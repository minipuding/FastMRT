"""
Copyright (c) Sijie Xu with email:sijie.x@sjtu.edu.cn.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import pytorch_ssim as pssim
from typing import Union, Any


class FastmrtMetrics:

    @staticmethod
    def ssim(pred: torch.Tensor, gt: torch.Tensor, device: str='cuda') -> torch.Tensor:
        """
        calculate Structural Similarity between predicted output and ground truth.
        Args:
            pred: an output tensor predicted from model. (b, c, h, w)
            gt: ground truth.
            device:
        Return:
            ssim_metric: a ssim value.
        """
        ssim_metric = torch.tensor(0, dtype=torch.float32, device='cuda')
        # normalize to 0 ~ 1.
        pred = (pred - torch.min(pred, dim=0, keepdim=True)[0]) / \
               (torch.max(pred, dim=0, keepdim=True)[0] - torch.min(pred, dim=0, keepdim=True)[0])
        gt = (gt - torch.min(gt, dim=0, keepdim=True)[0]) / \
             (torch.max(gt, dim=0, keepdim=True)[0] - torch.min(gt, dim=0, keepdim=True)[0])
        
        for batch_idx in range(pred.shape[0]):
            ssim_metric += pssim.ssim(pred[batch_idx].unsqueeze(0).cuda(), gt[batch_idx].unsqueeze(0).cuda()).item()
        return ssim_metric / pred.shape[0]

    @staticmethod
    def psnr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        psnr_metric = torch.tensor(0, dtype=torch.float32, device='cuda')
        for batch_idx in range(pred.shape[0]):
            mse = torch.mean((pred[batch_idx] - gt[batch_idx]) ** 2)
            psnr_metric += 10 * torch.log10(gt[batch_idx].max() ** 2 / mse + 1.0)
        return psnr_metric / pred.shape[0]

    @staticmethod
    def mse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        mse_metric = torch.tensor(0, dtype=torch.float32, device='cuda')
        for batch_idx in range(pred.shape[0]):
            mse_metric += torch.mean((pred[batch_idx] - gt[batch_idx]) ** 2)
        return mse_metric / pred.shape[0]

    @staticmethod
    def rmse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        rmse_metric = torch.tensor(0, dtype=torch.float32, device='cuda')
        for batch_idx in range(pred.shape[0]):
            rmse_metric += torch.mean((pred[batch_idx] - gt[batch_idx]) ** 2)
        return torch.sqrt(rmse_metric / pred.shape[0])

    @staticmethod
    def dice(pred: torch.Tensor, gt: torch.Tensor, smooth: float=1e-5) -> torch.Tensor:
        dice_metric = torch.tensor(0, dtype=torch.float32, device='cuda')
        for batch_idx in range(pred.shape[0]):
            inter = (pred[batch_idx] * gt[batch_idx]).sum()
            union = (pred[batch_idx] + gt[batch_idx]).sum()
            dice_metric += 2 * (inter + smooth) / (union + smooth)
        return dice_metric / pred.shape[0]

    @staticmethod
    def bland_altman(pred: torch.Tensor, gt: torch.Tensor) -> tuple[Union[float, Any], Any, Any, Any, Any]:
        ba_mean = (pred.flatten() + gt.flatten()) / 2
        ba_error = pred.flatten() - gt.flatten()
        ba_error_mean = ba_error.mean()
        ba_error_std = ba_error.std()
        ba_out_loa = torch.sum((ba_error > ba_error_mean + 1.96 * ba_error_std) | \
                               (ba_error < ba_error_mean - 1.96 * ba_error_std)) / ba_mean.shape[0]
        return ba_mean, ba_error, ba_error_mean, ba_error_std, ba_out_loa
