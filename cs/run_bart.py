"""
Modified from FastMRI.
"""

import logging
import multiprocessing
import pathlib
import time
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
from torch import Tensor
import yaml
import os

from fastmrt.data import SliceDataset
from fastmrt.data import FastmrtDataTransform2D as T
from fastmrt.data.mask import RandomMaskFunc, EquiSpacedMaskFunc, apply_mask
from fastmrt.data.transforms import FastmrtDataTransform2D
from fastmrt.data.prf import PrfHeader, PrfFunc
from fastmrt.utils.seed import randomness
from fastmrt.utils.trans import complex_tensor_to_real_tensor as ct2rt
from fastmrt.utils.trans import complex_np_to_real_np as cn2rn
from fastmrt.utils.fftc import ifft2c_numpy
from fastmrt.utils.vis import draw_tmap
from transforms import CSTransform
import eval
import sys
from tqdm import tqdm
import json


def cs_total_variation(kspace: Tensor, reg_wt: float, num_low_freqs: int, num_iters: int=200):
    """
    Run ESPIRIT coil sensitivity estimation and Total Variation Minimization
    based reconstruction algorithm using the BART toolkit.

    Args:
        args (argparse.Namespace): Arguments including ESPIRiT parameters.
        reg_wt (float): Regularization parameter.
        crop_size (tuple): Size to crop final image to.

    Returns:
        np.array: Reconstructed image.
    """
    kspace = torch.from_numpy(kspace)

    kspace = kspace.unsqueeze(0).unsqueeze(-1)

    kspace = kspace.numpy()

    pred = bart.bart(
        1, f"pics -d0 -R T:7:0:{reg_wt} -i {num_iters}", kspace, np.ones_like(kspace)
    )

    return pred


def run_bart(args, dataset):
    """Run the BART reconstruction on the given data set."""

    outputs = []
    for data in tqdm(dataset):

        prediction = cs_total_variation(
            data.mask_kspace, args.reg_wt, data.num_low_freqs, args.num_iters,
        )
        prediction_ref = cs_total_variation(
            data.mask_kspace_ref, args.reg_wt, data.num_low_freqs, args.num_iters,
        )
        outputs.append(dict(
            pred=prediction.squeeze(),
            pred_ref=prediction_ref.squeeze(),
            kspace=data.kspace,
            kspace_ref=data.kspace_ref,
            mask_kspace=data.mask_kspace,
            mask_kspace_ref=data.mask_kspace_ref,
            tmap_mask=data.tmap_mask,
            mask=data.mask,
            metainfo=data.metainfo,
        ))
        
    return outputs

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
    parser.add_argument('-cd', '--cfg_dir', type=str, default='cs.yaml',
                        help="(str optional) the directory of config that saves other paths.")
    parser.add_argument('-nlf', '--num_iters', type=int, default=200,
                        help="(int optional) the number of low frequencies.")
    parser.add_argument('-rw', '--reg_wt', type=float, default=0.01,
                        help="")
    parser.add_argument('--save_dir', type=str, default='../logs/CS', 
                        help='')

    # load directory config
    core_args = parser.parse_args()

    # load net config (hypeparameters)
    with open(core_args.cfg_dir) as fconfig:
        cfgs = yaml.load(fconfig.read(), Loader=yaml.FullLoader)

    # collect all configs and turn to args
    args, flat_cfgs = parse_args_from_dict(cfgs, parser)

    return args, flat_cfgs, cfgs


if __name__ == "__main__":
    args, _, _ = build_args()

    # import bart
    os.environ['TOOLBOX_PATH'] = args.bart_dir
    sys.path.append(os.path.join(args.bart_dir, "python"))
    import bart

    # set randomness
    randomness(3407)

    # need this global for multiprocessing
    if args.data_sampling_mode == "RANDOM":
            mask_func = RandomMaskFunc(center_fraction=args.data_center_fraction,
                                            acceleration=args.data_acceleration)
    elif args.data_sampling_mode == "EQUISPACED":
        mask_func = EquiSpacedMaskFunc(center_fraction=args.data_center_fraction,
                                            acceleration=args.data_acceleration)

    prf_func = PrfFunc(
        prf_header=PrfHeader(
            B0=args.prf_b0,
            gamma=args.prf_gamma,
            alpha=args.prf_alpha,
            TE=args.prf_te,
    ))

    transforms = CSTransform(
        mask_func=mask_func,
        prf_func=prf_func,
        aug_func=None,
        use_random_seed=False,
        fftshift_dim=(-2, -1),
    )

    dataset = SliceDataset(
        root=os.path.join(args.data_dir[0], "test"),
        transform=transforms,
    )

    start_time = time.time()
    outputs = run_bart(args, dataset)
    end_time = time.time()

    # calculate image metrics
    img_metrics = eval.calc_image_metrics([cn2rn(sample['pred']) for sample in outputs], [cn2rn(ifft2c_numpy(sample['kspace'])) for sample in outputs])
    
    # calculate temperature metrics
    full_tmaps, recon_tmaps, file_names = [], [], []
    for sample in outputs:
        if sample["metainfo"]["frame_idx"] > 0: # we only focus on temperature maps after first frame.
            full_tmaps += [prf_func(ifft2c_numpy(sample["kspace"]),  ifft2c_numpy(sample["kspace_ref"])) * sample["tmap_mask"]]
            recon_tmaps += [prf_func(sample["pred"], sample["pred_ref"]) * sample["tmap_mask"]]
            file_names += [f"{sample['metainfo']['file_name']}_" \
                            f"f{sample['metainfo']['frame_idx']:02d}" \
                            f"s{sample['metainfo']['slice_idx']}" \
                            f"c{sample['metainfo']['coil_idx']}"]

    tmap_metrics = eval.calc_tmap_metrics(full_tmaps, recon_tmaps, args.log_tmap_patch_rate, args.log_tmap_heated_thresh)

    # combine metrics
    metrics = {"cost_time": (end_time - start_time) / len(dataset)}
    for k, v in img_metrics.items():
        metrics.update({k: float(v.to('cpu'))})
    for k, v in tmap_metrics.items():
        metrics.update({k: float(v.to('cpu'))})
    
    # save result
    save_root = os.path.join(args.save_dir, f'{os.path.basename(args.data_dir[0])}_{args.data_acceleration}x')
    os.makedirs(save_root, exist_ok=True)
    with open(os.path.join(save_root, 'metrics.json'), 'w') as fp:
        json.dump(metrics, fp)

    for full_tmap, recon_tmap, file_name in tqdm(zip(full_tmaps, recon_tmaps, file_names)):
        save_dir = os.path.join(save_root, file_name)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "full_tmap"), full_tmap)
        np.save(os.path.join(save_dir, "recon_tmap"), recon_tmap)
        with draw_tmap(torch.from_numpy(full_tmap)) as plt:
            plt.savefig(os.path.join(save_dir, "full_tmap.png"))
        with draw_tmap(torch.from_numpy(recon_tmap)) as plt:
            plt.savefig(os.path.join(save_dir, "recon_tmap.png"))
