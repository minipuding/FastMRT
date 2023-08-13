from fastmrt.utils.metrics import FastmrtMetrics
import torch
from typing import List

def calc_image_metrics(preds, srcs):

    # initialize scales
    mse = torch.tensor(0, dtype=torch.float32, device='cuda')
    ssim = torch.tensor(0, dtype=torch.float32, device='cuda')
    psnr = torch.tensor(0, dtype=torch.float32, device='cuda')

    batch_num = len(preds)

    # calculate metrics
    for pred, src in zip(preds, srcs):

        pred = torch.from_numpy(pred).unsqueeze(0).to('cuda')
        src = torch.from_numpy(src).unsqueeze(0).to('cuda')

        mse += FastmrtMetrics.mse(pred, src)
        ssim += FastmrtMetrics.ssim(pred, src)
        psnr += FastmrtMetrics.psnr(pred, src)

    return {
        'mse': mse / batch_num,
        'ssim': ssim / batch_num,
        'psnr': psnr / batch_num,
    }

def calc_tmap_metrics(
        full_tmaps: List,
        recon_tmaps: List,
        tmap_patch_rate: int,
        tmap_heated_thresh: int,
    ) -> None:

    # init metrics
    tmap_error = torch.tensor(0, dtype=torch.float32, device='cuda')
    patch_mse = torch.tensor(0, dtype=torch.float32, device='cuda')
    heated_area_dice = torch.tensor(0, dtype=torch.float32, device='cuda')
    patch_error_std = torch.tensor(0, dtype=torch.float32, device='cuda')

    # parameters
    tmap_num = len(full_tmaps)
    tmap_height = full_tmaps[0].shape[0]
    tmap_width = full_tmaps[0].shape[1]
    patch_height = tmap_height // tmap_patch_rate
    patch_width = tmap_width // tmap_patch_rate
    dice_calc_num = 0

    # calculate metrics
    for full_tmap, recon_tmap in zip(full_tmaps, recon_tmaps):

        full_tmap = torch.from_numpy(full_tmap).to('cuda')
        recon_tmap = torch.from_numpy(recon_tmap).to('cuda')

        # metric 1: mean of temperature maps error
        tmap_error += torch.mean(torch.abs(full_tmap - recon_tmap))

        # metric 2: patch root-mean-square error
        patch_full_tmap = full_tmap[(tmap_height - patch_height) // 2: tmap_height - (tmap_height - patch_height) // 2,
                                    (tmap_width - patch_width) // 2: tmap_width - (tmap_width - patch_width) // 2]
        patch_recon_tmap = recon_tmap[(tmap_height - patch_height) // 2: tmap_height - (tmap_height - patch_height) // 2,
                                    (tmap_width - patch_width) // 2: tmap_width - (tmap_width - patch_width) // 2]
        patch_mse += FastmrtMetrics.mse(patch_recon_tmap.unsqueeze(0), patch_full_tmap.unsqueeze(0))

        # metric 3: dice coefficient of heated areas.
        area_full_tmap = (patch_full_tmap > tmap_heated_thresh).type(torch.FloatTensor)
        area_recon_tmap = (patch_recon_tmap > tmap_heated_thresh).type(torch.FloatTensor)
        if torch.sum(area_full_tmap) > 1: # ensuring the ablation area exist
            heated_area_dice += FastmrtMetrics.dice(area_recon_tmap.unsqueeze(0), area_full_tmap.unsqueeze(0))
            dice_calc_num += 1

        # metric 4: bland-altman analysis
        _, _, _, ba_error_std, _ = FastmrtMetrics.bland_altman(patch_recon_tmap, patch_full_tmap)
        patch_error_std += ba_error_std


    return {
        'T_error': tmap_error / tmap_num,
        'T_patch_rmse': torch.sqrt(patch_mse / tmap_num),
        'T_heated_area_dice': heated_area_dice / dice_calc_num,
        'T_patch_error_std': patch_error_std / tmap_num
    }