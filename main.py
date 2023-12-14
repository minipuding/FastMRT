import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from argparse import ArgumentParser
import yaml
from yamlinclude import YamlIncludeConstructor
import thop
import os
from torchvision.transforms import Compose

from fastmrt.data.mask import RandomMaskFunc, EquiSpacedMaskFunc, apply_mask
from fastmrt.data.transforms import FastmrtDataTransform2D
from fastmrt.data.prf import PrfHeader, PrfFunc
from fastmrt.data.augs import ComplexAugs, IdentityAugs
from fastmrt.modules.data_module import FastmrtDataModule
from fastmrt.modules.model_module import CasNetModule, KDNetModule
from fastmrt.modules.base_module import FastmrtModule
from fastmrt.models.runet import Unet
from fastmrt.models.cunet import ComplexUnet
from fastmrt.models.resunet import UNet as ResUnet
from fastmrt.models.casnet import CasNet
from fastmrt.models.swtnet import SwinIR
from fastmrt.models.kdnet import KDNet
from fastmrt.utils.seed import randomness
from fastmrt.pretrain.transforms import FastmrtPretrainTransform

def inherit_constructor(loader, node):
    def merge_dict(dict1, dict2):
        for key in dict2:
            if key in dict1 and isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                merge_dict(dict1[key], dict2[key])
            else:
                dict1[key] = dict2[key]
    
    kwargs = loader.construct_mapping(node, deep=True)
    merge = kwargs.pop("_BASE_")
    merge_dict(merge, kwargs)
    return merge

# build args
def build_args():
    """
    Add command inplements and read configs, then flatten it as a dict.
    Returns:
        args: all argments from `parser.parse_args()`
        log_cfgs: the args should add to logger.
    """

    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f'{parent_key}{sep}{k.lower()}' if parent_key else k.lower()
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def parse_args_from_dict(cfgs, parser):
        flat_cfgs = flatten_dict(cfgs)
        for key, value in flat_cfgs.items():
            parser.add_argument(f'--{key}', type=type(value), default=value)
        return parser.parse_args(), flat_cfgs
    
    parser = ArgumentParser()

    # model choice & load dir config
    parser.add_argument('-n', '--net', type=str, required=True,
                        help="(str request) One of 'r-unet', 'c-unet', 'casnet', 'gannet', 'complexnet'")
    parser.add_argument('-s', '--stage', type=str, required=True,
                        help="(str request) One of 'train', 'test' and 'train-test'")
    parser.add_argument('-g', '--gpus', type=int, required=False, default=[0], nargs='+',
                        help="(int request) gpu(s) index")
    parser.add_argument('-cd', '--cfg_dir', type=str, default='./configs',
                        help="(str optional) the directory of config that saves other paths.")
    parser.add_argument('-os', '--only_source', action='store_true', default=False,
                        help="(bool optional) only use source datas or use source and diffusion augmented datas.")
    parser.add_argument('-nul', '--no_use_logger', action='store_true', default=False,
                        help="(bool optional) whether using logger.")

    # load directory config
    core_args = parser.parse_args()
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir="./configs")
    yaml.add_constructor("!inherit", inherit_constructor)

    # load net config (hypeparameters)
    cfgs_dir = os.path.join(core_args.cfg_dir, f"{core_args.net}.yaml")
    assert os.path.exists(cfgs_dir), \
        f"You try to run `{core_args.net}`, but there is no config file named `{core_args.net}.yaml` on your config directory ({core_args.cfg_dir})"
    with open(cfgs_dir) as fconfig:
        cfgs = yaml.load(fconfig.read(), Loader=yaml.FullLoader)

    # collect all configs and turn to args
    args, flat_cfgs = parse_args_from_dict(cfgs, parser)

    # turn str to float for some special params
    args.model_lr = flat_cfgs["model_lr"] = float(args.model_lr)
    args.model_weight_decay = flat_cfgs["model_weight_decay"] = float(args.model_weight_decay)

    return args, flat_cfgs, cfgs

class FastmrtRunner:

    def __init__(self, args, log_cfgs, source_cfgs) -> None:

        self.args = args

        # initialize mask function
        if args.data_sampling_mode == "RANDOM":
            self.mask_func = RandomMaskFunc(center_fraction=args.data_center_fraction,
                                            acceleration=self.args.data_acceleration)
        elif args.data_sampling_mode == "EQUISPACED":
            self.mask_func = EquiSpacedMaskFunc(center_fraction=args.data_center_fraction,
                                                acceleration=self.args.data_acceleration)
        
        # initialize prf function
        self.prf_func = PrfFunc(
            prf_header=PrfHeader(
                B0=self.args.prf_b0,
                gamma=self.args.prf_gamma,
                alpha=self.args.prf_alpha,
                TE=self.args.prf_te,
        ))

        # initialize augmentation function
        if args.augs_enabled is False:
            self.augs_func = IdentityAugs()
        else:
            self.augs_func = ComplexAugs(
                ca_rate=self.args.augs_ca_rate,
                objs=self.args.augs_objs,
                ap_logic=self.args.augs_ap_logic,
                augs_list=self.args.augs_list,
                compose_num=self.args.augs_compose_num
            )

        # initialize transforms
        self.train_transform = FastmrtDataTransform2D(
            mask_func=self.mask_func,
            prf_func=self.prf_func,
            aug_func=self.augs_func,
            data_format=self.args.data_format,
            use_random_seed=True,
            fftshift_dim=(-2, -1)
        )
        self.val_transform = FastmrtDataTransform2D(
            mask_func=self.mask_func,
            prf_func=self.prf_func,
            aug_func=None,
            data_format=self.args.data_format,
            use_random_seed=False,
            fftshift_dim=(-2, -1),
        )
        self.test_transform = self.val_transform

        # initialize data module
        self.data_module = FastmrtDataModule(
            root=self.args.data_dir,
            only_source=args.only_source,
            train_transform=self.train_transform,
            val_transform=self.val_transform,
            test_transform=self.test_transform,
            batch_size=self.args.data_batch_size,
            dataset_type=self.args.data_type,
            workers=args.data_workers,
        )

        # initialize model module
        self.model_module = self._init_model_module(args=args)

        # initialize logger
        if args.no_use_logger is False:

            # define wandb logger
            self.logger = loggers.WandbLogger(save_dir=self.args.log_dir, name=self.args.log_name, project=self.args.net.upper())

            # add FLOPs and params to log
            pesudo_img = torch.randn(1, 1, 96, 96) + 1j if args.net == 'cunet' else torch.randn(1, 2, 96, 96)
            flops, params = thop.profile(self.model_module.model, (pesudo_img, ))
            log_cfgs["FLOPs"] = "%.2fG" % (flops * 1e-9)
            log_cfgs["params"] = "%.2fM" % (params * 1e-6)

            # save configs to wandb
            self.logger.log_hyperparams(log_cfgs)
        else:
            
            self.logger = False

        # create Traner
        self.trainer = pl.Trainer(
            gpus=args.gpus,
            strategy='ddp' if len(args.gpus) > 1 else None,
            enable_progress_bar=False,
            max_epochs=args.model_max_epochs,
            logger=self.logger
        )

        # save source configs
        with open("last_config.yaml", 'w') as f:
            yaml.dump(source_cfgs, f)

    def run(self):
        if self.args.stage == 'train':
            self.trainer.fit(self.model_module, datamodule=self.data_module)
        elif self.args.stage == 'train-test':
            self.trainer.fit(self.model_module, datamodule=self.data_module)
            self.trainer.test(self.model_module, datamodule=self.data_module)
        elif self.args.stage == 'test':
            if args.net != 'zf':  # zero filled should not load state dict
                ckpt = torch.load(args.test_ckpt_dir)
                self.model_module.load_state_dict(ckpt["state_dict"], strict=False)
            self.trainer.test(self.model_module, datamodule=self.data_module)
    
    def _get_model(self, args):

        if args.net == 'zf':
            return torch.nn.Identity()
        
        elif args.net == 'runet':
            return Unet(
                in_channels=args.model_in_channels,
                out_channels=args.model_out_channels,
                base_channels=args.model_base_channels,
                level_num=args.model_level_num,
                drop_prob=args.model_drop_prob,
                leakyrelu_slope=args.model_leakyrelu_slope,
            )
        elif args.net == 'cunet':
            return ComplexUnet(
                in_channels=args.model_in_channels,
                out_channels=args.model_out_channels,
                base_channels=args.model_base_channels,
                level_num=args.model_level_num,
                drop_prob=args.model_drop_prob,
            )
        elif args.net == 'resunet':
            return ResUnet(
                in_channels=args.model_in_channels,
                out_channels=args.model_out_channels,
                base_channels=args.model_base_channels,
                ch_mult=args.model_ch_mult,
                attn=args.model_attn,
                num_res_blocks=args.model_num_res_blocks,
                drop_prob=args.model_drop_prob,
            )
        elif args.net == 'casnet':
            return CasNet(
                in_channels=args.model_in_channels,
                out_channels=args.model_out_channels,
                base_channels=args.model_base_channels,
                res_block_num=args.model_res_block_num,
                res_conv_ksize=args.model_res_conv_ksize,
                res_conv_num=args.model_res_conv_num,
                drop_prob=args.model_drop_prob,
                leakyrelu_slope=args.model_leakyrelu_slope,
            )
        elif args.net == 'swtnet':
            return SwinIR(
                upscale=args.model_upscale,
                in_chans=args.model_in_channels,
                img_size=args.model_img_size,
                patch_size=args.model_patch_size,
                window_size=args.model_window_size,
                img_range=args.model_img_range,
                depths=args.model_depths,
                embed_dim=args.model_embed_dim,
                num_heads=args.model_num_heads,
                mlp_ratio=args.model_mlp_ratio,
                upsampler=args.model_upsampler,
                resi_connection=args.model_resi_connection
            )

    def _init_model_module(self, args):
        
        if args.kd_enabled == True:

            # construct student net and teacher net
            stu_net = self._get_model(args)
            if args.net == "swtnet":
                args.model_num_heads = [num * args.kd_channels_ratio for num in args.model_num_heads]
            else:
                args.model_base_channels *= args.kd_channels_ratio
            tea_net = self._get_model(args)

            # load teacher state dict
            model_state_dict = torch.load(args.kd_tea_dir)["state_dict"]
            tea_state_dict = {}
            for key, value in model_state_dict.items():
                if 'total' in key:
                    continue
                new_key = key[len("model."):]
                tea_state_dict[new_key] = value
            tea_net.load_state_dict(tea_state_dict)

            # create module with knowledge distillation
            return KDNetModule(
                model=KDNet(tea_net, stu_net),
                loss_type=args.model_loss_type,
                max_epochs=args.model_max_epochs,
                lr=args.model_lr,
                weight_decay=args.model_weight_decay,
                tmap_prf_func=self.prf_func,
                tmap_patch_rate=args.log_tmap_patch_rate,
                tmap_heated_thresh=args.log_tmap_heated_thresh,
                log_images_frame_idx=args.log_images_frame_idx,
                log_images_freq=args.log_images_freq
            )
        
        if args.net == 'casnet':
            return CasNetModule(
                model=self._get_model(args),
                loss_type=args.model_loss_type,
                max_epochs=args.model_max_epochs,
                lr=args.model_lr,
                weight_decay=args.model_weight_decay,
                tmap_prf_func=self.prf_func,
                tmap_patch_rate=args.log_tmap_patch_rate,
                tmap_heated_thresh=args.log_tmap_heated_thresh,
                log_images_frame_idx=args.log_images_frame_idx,
                log_images_freq=args.log_images_freq
            )            
        else:
            return FastmrtModule(
                model=self._get_model(args),
                loss_type=args.model_loss_type,
                max_epochs=args.model_max_epochs,
                lr=args.model_lr,
                weight_decay=args.model_weight_decay,
                tmap_prf_func=self.prf_func,
                tmap_patch_rate=args.log_tmap_patch_rate,
                tmap_heated_thresh=args.log_tmap_heated_thresh,
                log_images_frame_idx=args.log_images_frame_idx,
                log_images_freq=args.log_images_freq
            )

if __name__ == "__main__":

    seed = 3407
    randomness(seed=seed)

    args, flat_cfgs, source_cfgs = build_args()
    runner = FastmrtRunner(args, flat_cfgs, source_cfgs)
    runner.run()