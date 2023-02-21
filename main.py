import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers
from argparse import ArgumentParser
import yaml
from cli import FastmrtCLI
from typing import NamedTuple

from fastmrt.data.dataset import SliceDataset
from fastmrt.data.mask import RandomMaskFunc, EquiSpacedMaskFunc, apply_mask
from fastmrt.data.transforms import UNetDataTransform, CasNetDataTransform, RFTNetDataTransform, ComposeTransform
from fastmrt.data.prf import PrfHeader, PrfFunc
from fastmrt.modules.data_module import FastmrtDataModule
from fastmrt.modules.unet_module import UNetModule
from fastmrt.modules.casnet_module import CasNetModule
from fastmrt.modules.rftnet_module import RFTNetModule
from fastmrt.models.runet import Unet
from fastmrt.pretrain.transforms import FastmrtPretrainTransform


# build args
def build_args():
    parser = ArgumentParser()

    # model choice & load dir config
    parser.add_argument('--net', type=str, required=False, default="r-unet",
                        help="(str request) One of 'r-unet', 'casnet', 'gannet', 'complexnet'")
    parser.add_argument('--stage', type=str, required=False, default="pre-train",
                        help="(str request) One of 'train', 'pre-train', 'fine-tune', 'test'")
    parser.add_argument('--gpus', type=int, required=False, default=[1, 0], nargs='+',
                        help="(int request) gpu(s) index")
    parser.add_argument('--dir_config', type=str, default='./configs/dir.yaml',
                        help="(str optional) the directory of config that saves other paths.")

    # load directory config
    with open(parser.parse_args().dir_config) as fconfig:
        dir_cfg = yaml.load(fconfig.read(), Loader=yaml.FullLoader)
    parser = FastmrtCLI.dir_cli(parser, dir_cfg)

    # load prf config
    with open(parser.parse_args().prf_config_dir) as fconfig:
        prf_cfg = yaml.load(fconfig.read(), Loader=yaml.FullLoader)
    parser = FastmrtCLI.prf_cli(parser, prf_cfg)

    # load net config (hypeparameters)
    if parser.parse_args().net == 'r-unet':
        with open(parser.parse_args().runet_config_dir) as fconfig:
            runet_cfg = yaml.load(fconfig.read(), Loader=yaml.FullLoader)
        parser = FastmrtCLI.runet_cli(parser, runet_cfg)
    elif parser.parse_args().net == 'casnet':
        with open(parser.parse_args().casnet_config_dir) as fconfig:
            casnet_cfg = yaml.load(fconfig.read(), Loader=yaml.FullLoader)
        parser = FastmrtCLI.casnet_cli(parser, casnet_cfg)
    elif parser.parse_args().net == 'rftnet':
        with open(parser.parse_args().rftnet_config_dir) as fconfig:
            rftnet_cfg = yaml.load(fconfig.read(), Loader=yaml.FullLoader)
        parser = FastmrtCLI.rftnet_cli(parser, rftnet_cfg)

    return parser.parse_args()

def run(args):

    if args.net == 'r-unet':

        gpus_num = len(args.gpus)
        # Obtain Mask Function
        if args.sampling_mode == "RANDOM":
            mask_func = RandomMaskFunc(center_fraction=args.center_fraction,
                                       acceleration=args.acceleration)
        elif args.sampling_mode == "EQUISPACED":
            mask_func = EquiSpacedMaskFunc(center_fraction=args.center_fraction,
                                           acceleration=args.acceleration)
        # Obtain PRF Function
        prf_func = PrfFunc(prf_header=PrfHeader(
            B0=args.b0,
            gamma=args.gamma,
            alpha=args.alpha,
            TE=args.te,
        ))

        # Obtain Transforms
        project_name = "RUNET"
        dataset_type = "2D"
        root = args.data_dir
        train_transform = UNetDataTransform(mask_func=mask_func,
                                            prf_func=prf_func,
                                            data_format=args.data_format,
                                            use_random_seed=True,
                                            resize_size=args.resize_size,
                                            resize_mode=args.resize_mode,
                                            fftshift_dim=(-2, -1))
        val_transform = UNetDataTransform(mask_func=mask_func,
                                          prf_func=prf_func,
                                          data_format=args.data_format,
                                          use_random_seed=False,
                                          resize_size=args.resize_size,
                                          resize_mode=args.resize_mode,
                                          fftshift_dim=(-2, -1))
        if args.stage == 'pre-train':
            project_name = "PRETRAIN_RUNET"
            dataset_type = "PT"
            root = args.pt_data_dir
            train_transform = ComposeTransform([
                FastmrtPretrainTransform(simufocus_type=args.sf_type,
                                         use_random_seed=True,
                                         frame_num=args.sf_frame_num,
                                         cooling_time_rate=args.sf_cooling_time_rate,
                                         max_delta_temp=args.sf_max_delta_temp,
                                         ), train_transform])
            val_transform = ComposeTransform([
                FastmrtPretrainTransform(simufocus_type=args.sf_type,
                                         use_random_seed=True,
                                         frame_num=args.sf_frame_num,
                                         cooling_time_rate=args.sf_cooling_time_rate,
                                         max_delta_temp=args.sf_max_delta_temp,
                                         ), val_transform])

        # Create Data Module
        # args.batch_size *= gpus_num
        data_module = FastmrtDataModule(root=root,
                                        train_transform=train_transform,
                                        val_transform=val_transform,
                                        test_transform=val_transform,
                                        batch_size=args.batch_size,
                                        dataset_type=dataset_type)

        # Create RUnet Module
        # args.lr *= gpus_num
        unet_module = UNetModule(in_channels=args.in_channels,
                                 out_channels=args.out_channels,
                                 base_channels=args.base_channels,
                                 level_num=args.level_num,
                                 drop_prob=args.drop_prob,
                                 leakyrelu_slope=args.leakyrelu_slope,
                                 last_layer_with_act=args.last_layer_with_act,
                                 lr=args.lr,
                                 lr_step_size=args.lr_step_size,
                                 lr_gamma=args.lr_gamma,
                                 weight_decay=args.weight_decay,
                                 tmap_prf_func=prf_func,
                                 tmap_patch_rate=args.tmap_patch_rate,
                                 tmap_max_temp_thresh=args.tmap_max_temp_thresh,
                                 tmap_ablation_thresh=args.tmap_ablation_thresh,
                                 log_images_frame_idx=args.log_images_frame_idx,
                                 log_images_freq=args.log_images_freq)

        # Judge whether the stage is ``fine-tune``
        if args.stage == "fine-tune":
            unet_module.load_state_dict(torch.load(args.model_dir)["state_dict"])
            # unet_module.model.down_convs.modules()
            # unet_module.model.down_convs.requires_grad_(False)  # freeze encoder

        # Create Logger & Add Hparams
        logger = loggers.WandbLogger(save_dir=args.log_dir, name=args.log_name, project=project_name)
        hparams = {
            "net": args.net,
            "batch_size": args.batch_size,
            "sampling_mode": args.sampling_mode,
            "acceleration": args.acceleration,
            "center_fraction": args.center_fraction,
            "resize_size": args.resize_size,
            "resize_mode": args.resize_mode,

            "base_channels": args.base_channels,
            "level_num": args.level_num,
            "drop_prob": args.drop_prob,
            "leakyrelu_slope": args.leakyrelu_slope,
            "last_layer_with_act": args.last_layer_with_act,
            "lr": args.lr,
            "lr_step_size": args.lr_step_size,
            "lr_gamma": args.lr_gamma,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
        }
        logger.log_hyperparams(hparams)

        # Create Traner
        strategy = 'ddp' if gpus_num > 1 else None
        print(strategy)
        trainer = pl.Trainer(gpus=args.gpus,
                             strategy=strategy,
                             enable_progress_bar=False,
                             max_epochs=args.max_epochs,
                             logger=logger)

        # Start Training
        trainer.fit(unet_module, datamodule=data_module)
    elif args.net == 'casnet':

        # Obtain Mask Function
        if args.sampling_mode == "RANDOM":
            mask_func = RandomMaskFunc(center_fraction=args.center_fraction,
                                       acceleration=args.acceleration)
        elif args.sampling_mode == "EQUISPACED":
            mask_func = EquiSpacedMaskFunc(center_fraction=args.center_fraction,
                                           acceleration=args.acceleration)
        # Obtain PRF Function
        prf_func = PrfFunc(prf_header=PrfHeader(
            B0=args.b0,
            gamma=args.gamma,
            alpha=args.alpha,
            TE=args.te,
        ))

        # Obtain Transforms
        train_transform = CasNetDataTransform(mask_func, data_format=args.data_format)
        val_transform = CasNetDataTransform(mask_func, data_format=args.data_format, use_random_seed=False)

        # Create Data Module
        data_module = FastmrtDataModule(root=args.data_dir,
                                        train_transform=train_transform,
                                        val_transform=val_transform,
                                        test_transform=val_transform,
                                        batch_size=args.batch_size)

        # Create RUnet Module
        casnet_module = CasNetModule(in_channels=args.in_channels,
                                     out_channels=args.out_channels,
                                     base_channels=args.base_channels,
                                     res_block_num=args.res_block_num,
                                     res_conv_ksize=args.res_conv_ksize,
                                     res_conv_num=args.res_conv_num,
                                     drop_prob=args.drop_prob,
                                     leakyrelu_slope=args.leakyrelu_slope,
                                     lr=args.lr,
                                     lr_step_size=args.lr_step_size,
                                     lr_gamma=args.lr_gamma,
                                     weight_decay=args.weight_decay,
                                     tmap_prf_func=prf_func,
                                     tmap_patch_rate=args.tmap_patch_rate,
                                     tmap_max_temp_thresh=args.tmap_max_temp_thresh,
                                     tmap_ablation_thresh=args.tmap_ablation_thresh,
                                     log_images_frame_idx=args.log_images_frame_idx,
                                     log_images_freq=args.log_images_freq)

        # Create Logger & Add Hparams
        logger = loggers.WandbLogger(save_dir=args.log_dir, name=args.log_name, project="CASNET")
        hparams = {
            "net": args.net,
            "batch_size": args.batch_size,
            "sampling_mode": args.sampling_mode,
            "acceleration": args.acceleration,
            "center_fraction": args.center_fraction,
            "resize_size": args.resize_size,
            "resize_mode": args.resize_mode,

            "base_channels": args.base_channels,
            "res_block_num": args.res_block_num,
            "res_conv_ksize": args.res_conv_ksize,
            "res_conv_num": args.res_conv_num,
            "drop_prob": args.drop_prob,
            "leakyrelu_slope": args.leakyrelu_slope,
            "lr": args.lr,
            "lr_step_size": args.lr_step_size,
            "lr_gamma": args.lr_gamma,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
        }
        logger.log_hyperparams(hparams)

        # Create Traner
        trainer = pl.Trainer(gpus=[0],
                             enable_progress_bar=False,
                             max_epochs=args.max_epochs,
                             logger=logger)

        # Start Training
        trainer.fit(casnet_module, datamodule=data_module)
    elif args.net == 'rftnet':
        # Obtain Mask Function
        if args.sampling_mode == "RANDOM":
            mask_func = RandomMaskFunc(center_fraction=args.center_fraction,
                                       acceleration=args.acceleration)
        elif args.sampling_mode == "EQUISPACED":
            mask_func = EquiSpacedMaskFunc(center_fraction=args.center_fraction,
                                           acceleration=args.acceleration)
        # Obtain PRF Function
        prf_func = PrfFunc(prf_header=PrfHeader(
            B0=args.b0,
            gamma=args.gamma,
            alpha=args.alpha,
            TE=args.te,
        ))

        # Obtain Transforms
        train_transform = RFTNetDataTransform(mask_func, data_format=args.data_format)
        val_transform = RFTNetDataTransform(mask_func, data_format=args.data_format, use_random_seed=False)

        # Create Data Module
        data_module = FastmrtDataModule(root=args.data_dir,
                                        train_transform=train_transform,
                                        val_transform=val_transform,
                                        test_transform=val_transform,
                                        batch_size=args.batch_size)

        # Create RUnet Module
        unet_module = RFTNetModule(in_channels=args.in_channels,
                                 out_channels=args.out_channels,
                                 base_channels=args.base_channels,
                                 level_num=args.level_num,
                                 drop_prob=args.drop_prob,
                                 leakyrelu_slope=args.leakyrelu_slope,
                                 last_layer_with_act=args.last_layer_with_act,
                                 lr=args.lr,
                                 lr_step_size=args.lr_step_size,
                                 lr_gamma=args.lr_gamma,
                                 weight_decay=args.weight_decay,
                                 tmap_prf_func=prf_func,
                                 tmap_patch_rate=args.tmap_patch_rate,
                                 tmap_max_temp_thresh=args.tmap_max_temp_thresh,
                                 tmap_ablation_thresh=args.tmap_ablation_thresh,
                                 log_images_frame_idx=args.log_images_frame_idx,
                                 log_images_freq=args.log_images_freq)

        # Create Logger & Add Hparams
        logger = loggers.WandbLogger(save_dir=args.log_dir, name=args.log_name, project="RFTNET")
        hparams = {
            "net": args.net,
            "batch_size": args.batch_size,
            "sampling_mode": args.sampling_mode,
            "acceleration": args.acceleration,
            "center_fraction": args.center_fraction,
            "resize_size": args.resize_size,
            "resize_mode": args.resize_mode,

            "base_channels": args.base_channels,
            "level_num": args.level_num,
            "drop_prob": args.drop_prob,
            "leakyrelu_slope": args.leakyrelu_slope,
            "last_layer_with_act": args.last_layer_with_act,
            "lr": args.lr,
            "lr_step_size": args.lr_step_size,
            "lr_gamma": args.lr_gamma,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
        }
        logger.log_hyperparams(hparams)

        # Create Traner
        trainer = pl.Trainer(gpus=[0],
                             enable_progress_bar=False,
                             max_epochs=args.max_epochs,
                             logger=logger)

        # Start Training
        trainer.fit(unet_module, datamodule=data_module)
    else:
        raise ValueError("``--net`` must be one of 'r-unet', 'casnet', rftnet, but {} was got.".format(args.net))

if __name__ == "__main__":
    import random
    torch.manual_seed(3407)  #设置随机种子，使每次训练方式固定
    torch.cuda.manual_seed(3407)
    random.seed(3407)
    np.random.seed(3407)
    torch.backends.cudnn.deterministic = True

    args = build_args()
    run(args)