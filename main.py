import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from argparse import ArgumentParser
import yaml
import thop
from torchvision.transforms import Compose

from fastmrt.data.mask import RandomMaskFunc, EquiSpacedMaskFunc, apply_mask
from fastmrt.data.transforms import FastmrtDataTransform2D
from fastmrt.data.prf import PrfHeader, PrfFunc
from fastmrt.data.augs import ComplexAugs, IdentityAugs
from fastmrt.modules.data_module import FastmrtDataModule
from fastmrt.modules.model_module import (
    UNetModule, 
    CUNetModule, 
    ResUNetModule,
    CasNetModule,
    SwtNetModule,
    KDNetModule,
)
from fastmrt.models.runet import Unet
from fastmrt.utils.seed import randomness
from fastmrt.pretrain.transforms import FastmrtPretrainTransform

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
        return parser.parse_args()
    
    parser = ArgumentParser()
    cfgs, log_cfgs = {}, {}

    # model choice & load dir config
    parser.add_argument('-n', '--net', type=str, required=True,
                        help="(str request) One of 'r-unet', 'c-unet', 'casnet', 'gannet', 'complexnet'")
    parser.add_argument('-s', '--stage', type=str, required=True,
                        help="(str request) One of 'train', 'pre-train', 'fine-tune', 'test'")
    parser.add_argument('-g', '--gpus', type=int, required=False, default=[0], nargs='+',
                        help="(int request) gpu(s) index")
    parser.add_argument('-dc', '--dir_config', type=str, default='./configs/dir.yaml',
                        help="(str optional) the directory of config that saves other paths.")
    parser.add_argument('-os', '--only_source', action='store_true', default=False,
                        help="(bool optional) only use source datas or use source and diffusion augmented datas.")
    parser.add_argument('-nul', '--no_use_logger', action='store_true', default=False,
                        help="(bool optional) whether using logger.")

    # load directory config
    with open(parser.parse_args().dir_config) as fconfig:
        dir_cfg = yaml.load(fconfig.read(), Loader=yaml.FullLoader)

    # load prf config
    with open(dir_cfg['PRF_CONFIG_DIR']) as fconfig:
        prf_cfg = yaml.load(fconfig.read(), Loader=yaml.FullLoader)

    # load net config (hypeparameters)
    if parser.parse_args().net == 'runet':
        with open(dir_cfg['RUNET_CONFIG_DIR']) as fconfig:
            net_cfg = yaml.load(fconfig.read(), Loader=yaml.FullLoader)
    elif parser.parse_args().net == 'cunet':
        with open(dir_cfg['CUNET_CONFIG_DIR']) as fconfig:
            net_cfg = yaml.load(fconfig.read(), Loader=yaml.FullLoader)
    elif parser.parse_args().net == 'resunet':
        with open(dir_cfg['RESUNET_CONFIG_DIR']) as fconfig:
            net_cfg = yaml.load(fconfig.read(), Loader=yaml.FullLoader)
    elif parser.parse_args().net == 'swtnet':
        with open(dir_cfg['SWTNET_CONFIG_DIR']) as fconfig:
            net_cfg = yaml.load(fconfig.read(), Loader=yaml.FullLoader)
    elif parser.parse_args().net == 'casnet':
        with open(dir_cfg['CASNET_CONFIG_DIR']) as fconfig:
            net_cfg = yaml.load(fconfig.read(), Loader=yaml.FullLoader)
    elif parser.parse_args().net == 'kdnet':
        with open(dir_cfg['KDNET_CONFIG_DIR']) as fconfig:
            net_cfg = yaml.load(fconfig.read(), Loader=yaml.FullLoader)

    # collect all configs and turn to args
    cfgs.update(dir_cfg)
    cfgs.update(prf_cfg)
    cfgs.update(net_cfg)
    args = parse_args_from_dict(cfgs, parser)

    # collect configs requires to log
    log_cfgs.update(net_cfg)
    log_cfgs.update(dict(DATA_DIR=cfgs["DATA_DIR"]))
    log_cfgs = flatten_dict(log_cfgs)

    # turn str to float for some special params
    args.model_lr = log_cfgs["model_lr"] = float(args.model_lr)
    args.model_weight_decay = log_cfgs["model_weight_decay"] = float(args.model_weight_decay)

    return args, log_cfgs

class FastmrtRunner:

    def __init__(self, args, log_cfgs) -> None:

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
                B0=self.args.b0,
                gamma=self.args.gamma,
                alpha=self.args.alpha,
                TE=self.args.te,
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
            dataset_type='2D',
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
            self.logger.log_hyperparams(log_cfgs)
        else:
            
            self.logger = False

        # Create Traner
        self.trainer = pl.Trainer(
            gpus=args.gpus,
            strategy='ddp' if len(args.gpus) > 1 else None,
            enable_progress_bar=False,
            max_epochs=args.model_max_epochs,
            logger=self.logger
        )

    def run(self):
        self.trainer.fit(self.model_module, datamodule=self.data_module)

    def _init_model_module(self, args):
        if args.net == 'runet':
            return UNetModule(
                in_channels=args.model_in_channels,
                out_channels=args.model_out_channels,
                base_channels=args.model_base_channels,
                level_num=args.model_level_num,
                drop_prob=args.model_drop_prob,
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
        elif args.net == 'cunet':
            return CUNetModule(
                in_channels=args.model_in_channels,
                out_channels=args.model_out_channels,
                base_channels=args.model_base_channels,
                level_num=args.model_level_num,
                drop_prob=args.model_drop_prob,
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
        elif args.net == 'resunet':
            return ResUNetModule(
                in_channels=args.model_in_channels,
                out_channels=args.model_out_channels,
                base_channels=args.model_base_channels,
                ch_mult=args.model_ch_mult,
                attn=args.model_attn,
                num_res_blocks=args.model_num_res_block,
                drop_prob=args.model_drop_prob,
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
        elif args.net == 'swtnet':
            return SwtNetModule(
                upscale=args.model_upscale,
                in_channels=args.model_in_channels,
                img_size=args.model_img_size,
                patch_size=args.model_patch_size,
                window_size=args.model_window_size,
                img_range=args.model_img_range,
                depths=args.model_depths,
                embed_dim=args.model_embed_dim,
                num_heads=args.model_num_heads,
                mlp_ratio=args.model_mlp_ratio,
                upsampler=args.model_upsampler,
                resi_connection=args.model_resi_connection,
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
        elif args.net == 'casnet':
            return CasNetModule(
                in_channels=args.model_in_channels,
                out_channels=args.model_out_channels,
                base_channels=args.model_base_channels,
                res_block_num=args.model_res_block_num,
                res_conv_ksize=args.model_res_conv_ksize,
                res_conv_num=args.model_res_conv_num,
                drop_prob=args.model_drop_prob,
                leakyrelu_slope=args.model_leakyrelu_slope,
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
        elif args.net == 'kdnet':
            tea_net = Unet(
                in_channels=args.model_in_channels,
                out_channels=args.model_out_channels,
                base_channels=args.model_base_channels_tea
            )
            module_state_dict = torch.load(args.model_tea_dir)["state_dict"]
            model_state_dict = tea_net.state_dict()
            del_list=[]
            for key in module_state_dict:
                if 'total' in key:
                    del_list.append(key)
            for key in del_list:
                module_state_dict.pop(key, None)
            for model_key, module_key in zip(model_state_dict, module_state_dict):
                model_state_dict[model_key] = module_state_dict[module_key]
            tea_net.load_state_dict(model_state_dict)
            stu_net = Unet(
                in_channels=args.model_in_channels,
                out_channels=args.model_out_channels,
                base_channels=args.model_base_channels_stu
            )

            # Create RUnet Module
            return KDNetModule(
                tea_net=tea_net,
                stu_net=stu_net,
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
            raise ValueError("``--net`` must be one of 'runet', 'cunet', 'resunet', 'swtnet', \
                             'casnet' and 'kdnet' but {} was got.".format(args.net))

if __name__ == "__main__":

    seed = 3407
    randomness(seed=seed)

    args, cfgs = build_args()
    runner = FastmrtRunner(args, cfgs)
    runner.run()