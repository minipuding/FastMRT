import torch
import pytorch_ssim as pssim
from typing import Union, Any


# import torch.nn.functional as F
# from torch.autograd import Variable
# import numpy as np
# from math import exp


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

# class SSIM:
#     def __init__(self, device):
#         self.device = device
#
#     def _gaussian(self, window_size, sigma):
#         gauss = torch.Tensor([exp(-(x - window_size / 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
#         return gauss / gauss.sum()
#
#     def _create_window(self, window_size, channel):
#         _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
#         _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#         window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
#         return window.cuda()
#
#     def ssim(self, img1, img2, window_size, size_average=True):
#         (_, channel, _, _) = img1.size()
#         if self.device == 'cuda':
#             window = self._create_window(window_size, channel).cuda()
#         elif self.device == 'cpu':
#             window = self._create_window(window_size, channel).cpu()
#
#         mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
#         mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
#
#         mu1_sq = mu1.pow(2)
#         mu2_sq = mu2.pow(2)
#         mu1_mu2 = mu1 * mu2
#
#         sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
#         sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
#         sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
#
#         C1 = 0.01 ** 2
#         C2 = 0.03 ** 2
#
#         ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#
#         if size_average:
#             return ssim_map.mean()
#         else:
#             return ssim_map.mean(1).mean(1).mean(1)
