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
import copy
import thop

from fastmrt.data.dataset import SliceDataset
from fastmrt.data.mask import RandomMaskFunc, EquiSpacedMaskFunc, apply_mask
from fastmrt.data.transforms import (
    UNetDataTransform,
    CasNetDataTransform,
    RFTNetDataTransform,
    KDNetDataTransform,
    ComposeTransform,
)
from fastmrt.data.prf import PrfHeader, PrfFunc
from fastmrt.data.augs import AugsCollateFunction
from fastmrt.modules.data_module import FastmrtDataModule
from fastmrt.modules.unet_module import UNetModule
from fastmrt.modules.resunet_module import ResUNetModule
from fastmrt.modules.swtnet_module import SwtNetModule
from fastmrt.modules.casnet_module import CasNetModule
from fastmrt.modules.rftnet_module import RFTNetModule
from fastmrt.modules.kdnet_module import KDNetModule
from fastmrt.models.runet import Unet
from fastmrt.models.resunet import UNet as ResUNet
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
    elif parser.parse_args().net == 'resunet':
        with open(parser.parse_args().resunet_config_dir) as fconfig:
            resunet_cfg = yaml.load(fconfig.read(), Loader=yaml.FullLoader)
        parser = FastmrtCLI.resunet_cli(parser, resunet_cfg)
    elif parser.parse_args().net == 'swtnet':
        with open(parser.parse_args().swtnet_config_dir) as fconfig:
            swtnet_cfg = yaml.load(fconfig.read(), Loader=yaml.FullLoader)
        parser = FastmrtCLI.swtnet_cli(parser, swtnet_cfg)
    elif parser.parse_args().net == 'casnet':
        with open(parser.parse_args().casnet_config_dir) as fconfig:
            casnet_cfg = yaml.load(fconfig.read(), Loader=yaml.FullLoader)
        parser = FastmrtCLI.casnet_cli(parser, casnet_cfg)
    elif parser.parse_args().net == 'rftnet':
        with open(parser.parse_args().rftnet_config_dir) as fconfig:
            rftnet_cfg = yaml.load(fconfig.read(), Loader=yaml.FullLoader)
        parser = FastmrtCLI.rftnet_cli(parser, rftnet_cfg)
    elif parser.parse_args().net == 'kdnet':
        with open(parser.parse_args().kdnet_config_dir) as fconfig:
            kdnet_cfg = yaml.load(fconfig.read(), Loader=yaml.FullLoader)
        parser = FastmrtCLI.kdnet_cli(parser, kdnet_cfg)

    return parser.parse_args()

def run_runet(args):

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
    project_name = "AUGS"
    dataset_type = "2D"
    root = args.data_dir
    train_transform = UNetDataTransform(mask_func=mask_func,
                                        prf_func=prf_func,
                                        data_format=args.data_format,
                                        use_random_seed=True,
                                        fftshift_dim=(-2, -1))
    val_transform = UNetDataTransform(mask_func=mask_func,
                                      prf_func=prf_func,
                                      data_format=args.data_format,
                                      use_random_seed=False,
                                      fftshift_dim=(-2, -1))

    # define augs
    collate_fn = AugsCollateFunction(transforms=train_transform,
                                     ap_shuffle=args.ap_shuffle,
                                     union=args.union,
                                     objs=args.objs,
                                     ap_logic=args.ap_logic,
                                     augs_list=args.augs_list,
                                     compose_num=args.compose_num
                                     )
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
                                    dataset_type=dataset_type,
                                    collate_fn=collate_fn)

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
    flops, params = thop.profile(unet_module.model, (torch.randn(1, 2, 96, 96), ))
    hparams = {
        "net": args.net,
        "dataset": args.data_dir,
        "batch_size": args.batch_size,
        "sampling_mode": args.sampling_mode,
        "acceleration": args.acceleration,
        "center_fraction": args.center_fraction,
        "FLOPs": "%.2fG" % (flops * 1e-9),
        "params": "%.2fM" % (params * 1e-6),
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
        "augs-ap_shuffle": args.ap_shuffle,
        "augs-union": args.union,
        "augs-objs": args.objs,
        "augs-ap_logic": args.ap_logic,
        "augs-augs_list": args.augs_list,
        "augs-compose_num": args.compose_num,
    }
    logger.log_hyperparams(hparams)

    # Create Traner
    strategy = 'ddp' if gpus_num > 1 else None

    trainer = pl.Trainer(gpus=args.gpus,
                         strategy=strategy,
                         enable_progress_bar=False,
                         max_epochs=args.max_epochs,
                         logger=logger)

    # Start Training
    trainer.fit(unet_module, datamodule=data_module)

def run_resunet(args):

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
    project_name = "RESUNET"
    dataset_type = "2D"
    root = args.data_dir
    train_transform = UNetDataTransform(mask_func=mask_func,
                                        prf_func=prf_func,
                                        data_format=args.data_format,
                                        use_random_seed=True,
                                        fftshift_dim=(-2, -1))
    val_transform = UNetDataTransform(mask_func=mask_func,
                                      prf_func=prf_func,
                                      data_format=args.data_format,
                                      use_random_seed=False,
                                      fftshift_dim=(-2, -1))

    # define augs
    collate_fn = AugsCollateFunction(transforms=train_transform,
                                     ap_shuffle=args.ap_shuffle,
                                     union=args.union,
                                     objs=args.objs,
                                     ap_logic=args.ap_logic,
                                     augs_list=args.augs_list,
                                     compose_num=args.compose_num
                                     )
    if args.stage == 'pre-train':
        project_name = "PRETRAIN"
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
                                    dataset_type=dataset_type,
                                    collate_fn=collate_fn)

    # Create RUnet Module
    # args.lr *= gpus_num
    resunet_module = ResUNetModule(in_channels=args.in_channels,
                                   out_channels=args.out_channels,
                                   base_channels=args.base_channels,
                                   ch_mult=args.ch_mult,
                                   attn=args.attn,
                                   num_res_blocks=args.num_res_block,
                                   drop_prob=args.drop_prob,
                                   lr=args.lr,
                                   weight_decay=args.weight_decay,
                                   tmap_prf_func=prf_func,
                                   tmap_patch_rate=args.tmap_patch_rate,
                                   tmap_max_temp_thresh=args.tmap_max_temp_thresh,
                                   tmap_ablation_thresh=args.tmap_ablation_thresh,
                                   log_images_frame_idx=args.log_images_frame_idx,
                                   log_images_freq=args.log_images_freq)

    # Judge whether the stage is ``fine-tune``
    if args.stage == "fine-tune":
        resunet_module.load_state_dict(torch.load(args.model_dir)["state_dict"])
        # unet_module.model.down_convs.modules()
        # unet_module.model.down_convs.requires_grad_(False)  # freeze encoder

    # Create Logger & Add Hparams
    logger = loggers.WandbLogger(save_dir=args.log_dir, name=args.log_name, project=project_name)
    flops, params = thop.profile(resunet_module.model, (torch.randn(1, 2, 96, 96), ))
    hparams = {
        "net": args.net,
        "dataset": args.data_dir,
        "batch_size": args.batch_size,
        "sampling_mode": args.sampling_mode,
        "acceleration": args.acceleration,
        "center_fraction": args.center_fraction,
        "FLOPs": "%.2fG" % (flops * 1e-9),
        "params": "%.2fM" % (params * 1e-6),
        "base_channels": args.base_channels,
        "ch_mult": args.ch_mult,
        "attn": args.attn,
        "num_res_block": args.num_res_block,
        "drop_prob": args.drop_prob,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "augs-ap_shuffle": args.ap_shuffle,
        "augs-union": args.union,
        "augs-objs": args.objs,
        "augs-ap_logic": args.ap_logic,
        "augs-augs_list": args.augs_list,
        "augs-compose_num": args.compose_num,
    }
    logger.log_hyperparams(hparams)

    # Create Traner
    strategy = 'ddp' if gpus_num > 1 else None

    trainer = pl.Trainer(gpus=args.gpus,
                         strategy=strategy,
                         enable_progress_bar=False,
                         max_epochs=args.max_epochs,
                         logger=logger,
                         gradient_clip_algorithm="norm")

    # Start Training
    trainer.fit(resunet_module, datamodule=data_module)

def run_swtnet(args):

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
    project_name = "SWTNET"
    dataset_type = "2D"
    root = args.data_dir
    train_transform = UNetDataTransform(mask_func=mask_func,
                                        prf_func=prf_func,
                                        data_format=args.data_format,
                                        use_random_seed=True,
                                        fftshift_dim=(-2, -1))
    val_transform = UNetDataTransform(mask_func=mask_func,
                                      prf_func=prf_func,
                                      data_format=args.data_format,
                                      use_random_seed=False,
                                      fftshift_dim=(-2, -1))

    # define augs
    collate_fn = AugsCollateFunction(transforms=train_transform,
                                     ap_shuffle=args.ap_shuffle,
                                     union=args.union,
                                     objs=args.objs,
                                     ap_logic=args.ap_logic,
                                     augs_list=args.augs_list,
                                     compose_num=args.compose_num
                                     )
    if args.stage == 'pre-train':
        project_name = "PRETRAIN"
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
                                    dataset_type=dataset_type,
                                    collate_fn=collate_fn)

    # Create RUnet Module
    # args.lr *= gpus_num
    swtnet_module = SwtNetModule(upscale=args.upscale,
                                 in_channels=args.in_channels,
                                 img_size=args.img_size,
                                 patch_size=args.patch_size,
                                 window_size=args.window_size,
                                 img_range=args.img_range,
                                 depths=args.depths,
                                 embed_dim=args.embed_dim,
                                 num_heads=args.num_heads,
                                 mlp_ratio=args.mlp_ratio,
                                 upsampler=args.upsampler,
                                 resi_connection=args.resi_connection,
                                 lr=args.lr,
                                 weight_decay=args.weight_decay,
                                 tmap_prf_func=prf_func,
                                 tmap_patch_rate=args.tmap_patch_rate,
                                 tmap_max_temp_thresh=args.tmap_max_temp_thresh,
                                 tmap_ablation_thresh=args.tmap_ablation_thresh,
                                 log_images_frame_idx=args.log_images_frame_idx,
                                 log_images_freq=args.log_images_freq)

    # Judge whether the stage is ``fine-tune``
    if args.stage == "fine-tune":
        swtnet_module.load_state_dict(torch.load(args.model_dir)["state_dict"])
        # unet_module.model.down_convs.modules()
        # unet_module.model.down_convs.requires_grad_(False)  # freeze encoder

    # Create Logger & Add Hparams
    logger = loggers.WandbLogger(save_dir=args.log_dir, name=args.log_name, project=project_name)
    flops, params = thop.profile(swtnet_module.model, (torch.randn(1, 2, 96, 96), ))
    hparams = {
        "net": args.net,
        "dataset": args.data_dir,
        "batch_size": args.batch_size,
        "sampling_mode": args.sampling_mode,
        "acceleration": args.acceleration,
        "center_fraction": args.center_fraction,
        "FLOPs": "%.2fG" % (flops * 1e-9),
        "params": "%.2fM" % (params * 1e-6),
        "upscale": args.upscale,
        "img_size": args.img_size,
        "window_size": args.window_size,
        "img_range": args.img_range,
        "depths": args.depths,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "mlp_ratio": args.mlp_ratio,
        "upsampler": args.upsampler,
        "resi_connection": args.resi_connection,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "augs-ap_shuffle": args.ap_shuffle,
        "augs-union": args.union,
        "augs-objs": args.objs,
        "augs-ap_logic": args.ap_logic,
        "augs-augs_list": args.augs_list,
        "augs-compose_num": args.compose_num,
    }
    logger.log_hyperparams(hparams)

    # Create Traner
    strategy = 'ddp' if gpus_num > 1 else None

    trainer = pl.Trainer(gpus=args.gpus,
                         strategy=strategy,
                         enable_progress_bar=False,
                         max_epochs=args.max_epochs,
                         logger=logger)

    # Start Training
    trainer.fit(swtnet_module, datamodule=data_module)

def run_casnet(args):
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
    flops, params = thop.profile(casnet_module.model, (torch.randn(1, 2, 96, 96), ))
    hparams = {
        "net": args.net,
        "batch_size": args.batch_size,
        "sampling_mode": args.sampling_mode,
        "acceleration": args.acceleration,
        "center_fraction": args.center_fraction,
        "FLOPs": "%.2fG" % (flops * 1e-9),
        "params": "%.2fM" % (params * 1e-6),
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

def run_rftnet(args):
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
    rft_module = RFTNetModule(in_channels=args.in_channels,
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
    trainer.fit(rft_module, datamodule=data_module)

def run_kdnet(args):
    gpus_num = len(args.gpus)
    # Obtain Mask Function
    if args.sampling_mode == "RANDOM":
        mask_func_tea = RandomMaskFunc(center_fraction=args.center_fraction_tea,
                                       acceleration=args.acceleration_tea)
        mask_func_stu = RandomMaskFunc(center_fraction=args.center_fraction_stu,
                                       acceleration=args.acceleration_stu)
    elif args.sampling_mode == "EQUISPACED":
        mask_func_tea = EquiSpacedMaskFunc(center_fraction=args.center_fraction_tea,
                                           acceleration=args.acceleration_tea)
        mask_func_stu = EquiSpacedMaskFunc(center_fraction=args.center_fraction_stu,
                                           acceleration=args.acceleration_stu)
    # Obtain PRF Function
    prf_func = PrfFunc(prf_header=PrfHeader(
        B0=args.b0,
        gamma=args.gamma,
        alpha=args.alpha,
        TE=args.te,
    ))

    # Obtain Transforms
    project_name = "KD"
    dataset_type = "2D"
    root = args.data_dir
    train_transform = KDNetDataTransform(mask_func_tea=mask_func_tea,
                                         mask_func_stu=mask_func_stu,
                                         prf_func=prf_func,
                                         data_format=args.data_format,
                                         use_random_seed=True,
                                         fftshift_dim=(-2, -1))
    val_transform = KDNetDataTransform(mask_func_tea=mask_func_tea,
                                       mask_func_stu=mask_func_stu,
                                       prf_func=prf_func,
                                       data_format=args.data_format,
                                       use_random_seed=False,
                                       fftshift_dim=(-2, -1))

    # define augs
    collate_fn = AugsCollateFunction(transforms=train_transform,
                                     ap_shuffle=args.ap_shuffle,
                                     union=args.union,
                                     objs=args.objs,
                                     ap_logic=args.ap_logic,
                                     augs_list=args.augs_list,
                                     compose_num=args.compose_num
                                     )
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
    data_module = FastmrtDataModule(root=root,
                                    train_transform=train_transform,
                                    val_transform=val_transform,
                                    test_transform=val_transform,
                                    batch_size=args.batch_size * gpus_num,
                                    dataset_type=dataset_type,
                                    collate_fn=collate_fn)

    # Create teacher and student model
    tea_net = Unet(in_channels=args.in_channels,
                   out_channels=args.out_channels,
                   base_channels=args.base_channels_tea)
    module_state_dict = torch.load(args.model_dir)["state_dict"]
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
    stu_net = Unet(in_channels=args.in_channels,
                   out_channels=args.out_channels,
                   base_channels=args.base_channels_stu)

    # Create RUnet Module
    kdnet_module = KDNetModule(tea_net=tea_net,
                               stu_net=stu_net,
                               use_ema=args.use_ema,
                               soft_label_weight=args.soft_label_weight,
                               lr_stu=args.lr_stu,
                               weight_decay_stu=args.weight_decay_stu,
                               tmap_prf_func=prf_func,
                               tmap_patch_rate=args.tmap_patch_rate,
                               tmap_max_temp_thresh=args.tmap_max_temp_thresh,
                               tmap_ablation_thresh=args.tmap_ablation_thresh,
                               log_images_frame_idx=args.log_images_frame_idx,
                               log_images_freq=args.log_images_freq)

    # Judge whether the stage is ``fine-tune``
    if args.stage == "fine-tune":
        kdnet_module.load_state_dict(torch.load(args.model_dir)["state_dict"])
        # kdnet_module.model.down_convs.modules()
        # kdnet_module.model.down_convs.requires_grad_(False)  # freeze encoder

    # Create Logger & Add Hparams
    logger = loggers.WandbLogger(save_dir=args.log_dir, name=args.log_name, project=project_name)
    flops_tea, params_tea = thop.profile(tea_net, (torch.randn(1, 2, 96, 96), ))
    flops_stu, params_stu = thop.profile(stu_net, (torch.randn(1, 2, 96, 96), ))
    hparams = {
        "net_tea": tea_net._get_name(),
        "net_stu": stu_net._get_name(),
        "batch_size": args.batch_size,
        "sampling_mode": args.sampling_mode,
        "acceleration_tea": args.acceleration_tea,
        "acceleration_stu": args.acceleration_stu,
        "center_fraction_tea": args.center_fraction_tea,
        "center_fraction_stu": args.center_fraction_stu,
        "FLOPs_tea": "%.2fG" % (flops_tea * 1e-9),
        "FLOPs_stu": "%.2fG" % (flops_stu * 1e-9),
        "params_tea": "%.2fM" % (params_tea * 1e-6),
        "params_stu": "%.2fM" % (params_stu * 1e-6),
        "base_channels_tea": args.base_channels_tea,
        "base_channels_stu": args.base_channels_stu,
        "level_num_tea": args.level_num_tea,
        "level_num_stu": args.level_num_stu,
        "drop_prob_stu": args.drop_prob_stu,
        "leakyrelu_slope_stu": args.leakyrelu_slope_stu,
        "last_layer_with_act_tea": args.last_layer_with_act_tea,
        "last_layer_with_act_stu": args.last_layer_with_act_stu,
        "lr_stu": args.lr_stu,
        "weight_decay_stu": args.weight_decay_stu,
        "max_epochs": args.max_epochs,
        "soft_label_weight": args.soft_label_weight,
        "augs-ap_shuffle": args.ap_shuffle,
        "augs-union": args.union,
        "augs-objs": args.objs,
        "augs-ap_logic": args.ap_logic,
        "augs-augs_list": args.augs_list,
        "augs-compose_num": args.compose_num,
    }
    logger.log_hyperparams(hparams)

    # Create Traner
    strategy = 'ddp' if gpus_num > 1 else None

    trainer = pl.Trainer(gpus=args.gpus,
                         strategy=strategy,
                         enable_progress_bar=False,
                         max_epochs=args.max_epochs,
                         logger=logger)

    # Start Training
    trainer.fit(kdnet_module, datamodule=data_module)

def run(args):

    if args.net == 'r-unet':
        run_runet(args)
    elif args.net == 'resunet':
        run_resunet(args)
    elif args.net == 'swtnet':
        run_swtnet(args)
    elif args.net == 'casnet':
        run_casnet(args)
    elif args.net == 'rftnet':
        run_rftnet(args)
    elif args.net == 'kdnet':
        run_kdnet(args)
    else:
        raise ValueError("``--net`` must be one of 'r-unet', 'casnet', rftnet, but {} was got.".format(args.net))



if __name__ == "__main__":
    import random
    import os
    seed = 3407
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args = build_args()
    run(args)