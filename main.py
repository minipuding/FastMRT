import torch
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
from fastmrt.data.transforms import UNetDataTransform, CasNetDataTransform, RFTNetDataTransform
from fastmrt.data.prf import PrfHeader, PrfFunc
from fastmrt.modules.data_module import FastmrtDataModule
from fastmrt.modules.unet_module import UNetModule
from fastmrt.modules.casnet_module import CasNetModule
from fastmrt.modules.rftnet_module import RFTNetModule
from fastmrt.models.runet import Unet
from fastmrt.pretrain.transforms import FastmrtPretrainTransform

# class Hparams(NamedTuple):
#     net: str
#     batch_size: int
#     acceleration: int
#     center_fraction: float
#     resize_size: tuple
#     resize_mode: str
#
#     base_channels: int
#     level_num: int
#     drop_prob: float
#     leakyrelu_slope: float
#     last_layer_with_act: bool
#     lr: float
#     lr_step_size: int
#     lr_gamma: float
#     weight_decay: float
#     max_epochs: int

# build args
def build_args():
    parser = ArgumentParser()

    # model choice & load dir config
    parser.add_argument('--net', type=str, required=True,
                        help="(str request) One of 'r-unet', 'casnet', 'gannet', 'complexnet'")
    parser.add_argument('--stage', type=str, required=True,
                        help="(str request) One of 'train', 'pre-train', 'fine-tune', 'test'")
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
        if args.stage == 'train':
            project_name = "RUNET"
            dataset_type = "2D"
            root = args.data_dir
            train_transform = UNetDataTransform(mask_func=mask_func,
                                                prf_func=prf_func,
                                                data_format=args.data_format,
                                                use_random_seed=True,
                                                resize_size=args.resize_size,
                                                resize_mode=args.resize_mode,
                                                fftshift_dim=-2)
            val_transform = UNetDataTransform(mask_func=mask_func,
                                              prf_func=prf_func,
                                              data_format=args.data_format,
                                              use_random_seed=False,
                                              resize_size=args.resize_size,
                                              resize_mode=args.resize_mode,
                                              fftshift_dim=-2)
        elif args.stage == 'pre-train':
            project_name = "PRETRAIN_RUNET"
            dataset_type = "PT"
            root = args.pt_data_dir
            train_transform = FastmrtPretrainTransform(mask_func=mask_func,
                                                       prf_func=prf_func,
                                                       data_format=args.data_format,
                                                       use_random_seed=True,
                                                       resize_size=args.resize_size,
                                                       resize_mode=args.resize_mode,
                                                       fftshift_dim=(-2, -1),
                                                       simufocus_type=args.sf_type,
                                                       net=args.net,
                                                       frame_num=args.sf_frame_num,
                                                       cooling_time_rate=args.sf_cooling_time_rate,
                                                       center_crop_size=args.sf_center_crop_size,
                                                       random_crop_size=args.sf_random_crop_size,
                                                       max_delta_temp=args.sf_max_delta_temp,
                                                       )
            val_transform = FastmrtPretrainTransform(mask_func=mask_func,
                                                     prf_func=prf_func,
                                                     data_format=args.data_format,
                                                     use_random_seed=False,
                                                     resize_size=args.resize_size,
                                                     resize_mode=args.resize_mode,
                                                     fftshift_dim=(-2, -1),
                                                     simufocus_type=args.sf_type,
                                                     net=args.net,
                                                     frame_num=args.sf_frame_num,
                                                     cooling_time_rate=args.sf_cooling_time_rate,
                                                     center_crop_size=args.sf_center_crop_size,
                                                     random_crop_size=args.sf_random_crop_size,
                                                     max_delta_temp=args.sf_max_delta_temp,
                                                     )

        # Create Data Module
        data_module = FastmrtDataModule(root=root,
                                        train_transform=train_transform,
                                        val_transform=val_transform,
                                        test_transform=val_transform,
                                        batch_size=args.batch_size,
                                        dataset_type=dataset_type)

        # Create RUnet Module
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
        trainer = pl.Trainer(gpus=[0],
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
    args = build_args()
    run(args)