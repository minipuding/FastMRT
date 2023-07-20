"""
Copyright (c) Sijie Xu with email: sijie.x@foxmail.com.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os.path

import pytorch_lightning as pl
from typing import Dict, Sequence

from torch import nn

from fastmrt.models.cunet import ComplexUnet
from fastmrt.utils.metrics import FastmrtMetrics
from fastmrt.utils.normalize import denormalize
from fastmrt.utils.trans import real_tensor_to_complex_tensor as rt2ct
from fastmrt.utils.trans import complex_tensor_to_real_tensor as ct2rt
from fastmrt.utils.vis import draw_tmap, draw_bland_altman_fig, draw_linear_regression_fig
from fastmrt.data.prf import PrfFunc
import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, List
import matplotlib
import wandb
import time
import pandas

matplotlib.use('agg')

from matplotlib import pyplot as plt


class BaseModule(pl.LightningModule):
    """
    This is a PyTorch-Lightning module that defines the standard FastMRT metrics for temperature reconstruction.
    The metrics are divided into three categories: global, local, and graphs. 

    Global metrics reflect the reconstruction performance of temperature maps, but they may contain a lot of noise and areas without heating. 
    Local metrics are calculated only on the patch cropped center of the heated point, which better reflects the reconstruction performance. 
    The cropped size is 1/4 of the image size, and the heated point is always at the center of the field of view in FastMRT datasets.

    The module includes the following metrics:
        - Train stage: 
            - Training loss
        - Validation stage: 
            - Validation loss
            - Image metrics:
                - PSNR: Peak Signal to Noise Ratio 
                - SSIM: Structural Similarity
            - Temperature metrics: 
                - Temperature error: The absolute error of temperature maps between reconstructed images and original images
                - Patch RMSE: Root Mean Square of temperature map within the center patch of temperature images
                - Heated area dice: The dice coefficient between reconstructed temperature image and origin one of heating area. 
                    We assume the temperature over 43℃ is heated, and to eliminate ambient noise, we use the patch around the heating point.
                - Patch STD: The standard deviation of the patch used in `Patch RMSE` and `Heated area dice`
            - Graphs(Medias):
                - Amplitude Image: Original Image, Down-sampling Inputs, Reconstructed Image, Error Image
                - Temperature Maps: Original TMaps, Down-sampling TMaps, Reconstructed TMaps, Error TMaps
                - Bland-Altman analysis
                - Linear regression analysis
        - Test stage:
            - The same as validation.
    
    Args:
        tmap_prf_func: Proton resonance frequency function. 
        tmap_patch_rate: Patch size rate related to image size. Default is 4.
        tmap_heated_thresh: The threshold of the heated temperature when calculating the heated area.
        log_images_frame_idx: The frame index in a thermometry sequence we want to log. Since the temperature of each frame of a 
            temperature measuring sequence increases gradually, there may be no obvious heating signs in the early frame. 
            Therefore, for the convenience of observation, we recommend the sequence of frames 4 to 8 as the logging maps (most of 
            the sequences are 9 in length). Default is 5.
        log_images_freq: How many epochs logging images. Default is 50.
        device: The device for calculating metrics.
        enable_logger: Whether to calculate all metrics and save logs.
        is_log_image_metrics: Whether to calculate amplitude metrics and save logs.
        is_log_tmap_metrics: Whether to calculate temperature maps metrics and save logs.
        is_log_media_metrics: Whether to show medias to logs.
    """

    def __init__(
            self,
            model: nn.Module,
            tmap_prf_func: PrfFunc = None,
            tmap_patch_rate: int = 4,
            tmap_heated_thresh = 43,
            log_images_frame_idx: int = 5, # recommend 4 ~ 8
            log_images_freq: int = 50,
            device: str='cuda',
            is_log_image_metrics: bool=True,
            is_log_tmap_metrics: bool=True,
            is_log_media_metrics: bool=True,
    ):
        super(BaseModule, self).__init__()
        self.tmap_prf_func = tmap_prf_func
        self.tmap_patch_rate = tmap_patch_rate
        self.tmap_heated_thresh = tmap_heated_thresh
        self.log_images_frame_idx = log_images_frame_idx
        self.log_images_freq = log_images_freq
        self.to(device)
        self.is_log_image_metrics = is_log_image_metrics
        self.is_log_tmap_metrics = is_log_tmap_metrics
        self.is_log_media_metrics = is_log_media_metrics
        self.model = model

    def training_epoch_end(self, train_logs: Sequence[Dict]) -> None:
        """log train loss after training epoch end.
        """
        train_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        for log in train_logs:
            train_loss += log["loss"]
        self.log("loss", train_loss / len(train_logs), on_epoch=True, on_step=False)

    def validation_epoch_end(self, val_logs: Sequence[Dict]) -> None:

        # save image(amplitude) metrics
        if self.is_log_image_metrics is True:
            image_metrics = self._calc_image_metrics(val_logs)
            self._log_scalar(image_metrics, stage="val")

        # save temperature maps metrics
        if self.is_log_tmap_metrics is True:
            full_tmaps, recon_tmaps = [], []
            for log in val_logs:
                for sample_idx in range(log["input"].shape[0]):
                    if log["frame_idx"][sample_idx] > 0: # we only focus on temperature maps after first frame.
                        full_tmaps += [self.tmap_prf_func(rt2ct(log["label"][sample_idx]),
                                                        rt2ct(log["label_ref"][sample_idx])) * log["tmap_mask"][sample_idx]]
                        recon_tmaps += [self.tmap_prf_func(rt2ct(log["output"][sample_idx]),
                                                        rt2ct(log["output_ref"][sample_idx])) * log["tmap_mask"][sample_idx]]
            tmap_metrics = self._calc_tmap_metrics(full_tmaps, recon_tmaps)
            self._log_scalar(tmap_metrics)

        # save log medias (images & temperature maps)
        if self.is_log_media_metrics is True:
            if (self.current_epoch + 1) % self.log_images_freq == 0:
                self._log_medias(val_logs, f"val_medias")
    
    def test_epoch_end(self, outputs) -> None:

        # calculate image(amplitude) metrics
        image_metrics = self._calc_image_metrics(outputs)

        # calculate temperature maps metrics
        full_tmaps, recon_tmaps, file_names = [], [], []
        for log in outputs:
            for sample_idx in range(log["input"].shape[0]):
                if log["frame_idx"][sample_idx] > 0: # we only focus on temperature maps after first frame.
                    full_tmaps += [self.tmap_prf_func(rt2ct(log["label"][sample_idx]),
                                                    rt2ct(log["label_ref"][sample_idx])) * log["tmap_mask"][sample_idx]]
                    recon_tmaps += [self.tmap_prf_func(rt2ct(log["output"][sample_idx]),
                                                    rt2ct(log["output_ref"][sample_idx])) * log["tmap_mask"][sample_idx]]
                    
                    file_names += [f"{log['file_name'][sample_idx]}_" \
                                  f"f{log['frame_idx'][sample_idx]:02d}" \
                                  f"s{log['slice_idx'][sample_idx]}" \
                                  f"c{log['coil_idx'][sample_idx]}"]
        
        tmap_metrics = self._calc_tmap_metrics(full_tmaps, recon_tmaps)

        # save cost time
        self.to("cpu")
        pesudo_input = torch.ones_like(log["input"][0]).unsqueeze(0).cpu()  # batch_size=1
        if isinstance(self.model, ComplexUnet):
            pesudo_input = torch.ones((1, 1, log["input"].shape[-2], log["input"].shape[-1])) + 1j
        loop_times = 1000  # hack: for calculating average time
        start_time = time.time()
        for _ in range(loop_times):
            self.model(pesudo_input)
        end_time = time.time()
        cost_metrics = {"cost_time(s)": (end_time - start_time) / loop_times}

        # save metrics to table
        if self.logger is not None:
            all_metrics = dict(Experiment_Names=self.logger.experiment.name)
            all_metrics.update(tmap_metrics)
            all_metrics.update(image_metrics)
            all_metrics.update(cost_metrics)
            df_metrics = pandas.DataFrame({"metrics": all_metrics}).transpose()
            self.logger.experiment.log({"test_metrics": wandb.Table(dataframe=df_metrics)})

        # save medias
        save_root = os.path.join(self.logger.save_dir, self.logger.name, self.logger.experiment.id, "medias")
        os.makedirs(save_root, exist_ok=True)
        for full_tmap, recon_tmap, file_name in zip(full_tmaps, recon_tmaps, file_names):
            save_dir = os.path.join(save_root, file_name)
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "full_tmap"), full_tmap.cpu().numpy())
            np.save(os.path.join(save_dir, "recon_tmap"), recon_tmap.cpu().numpy())
            with draw_tmap(full_tmap) as plt:
                plt.savefig(os.path.join(save_dir, "full_tmap.png"))
            with draw_tmap(recon_tmap) as plt:
                plt.savefig(os.path.join(save_dir, "recon_tmap.png"))    

    def on_train_end(self) -> None:
        t = time.localtime()
        ts = f"{t.tm_year}{t.tm_mon:02d}{t.tm_mday:02d}_{t.tm_hour:02d}{t.tm_sec:02d}{t.tm_yday:02d}"
        torch.save(self.model.state_dict(), os.path.join(self.logger.save_dir, self.logger.name, self.logger.experiment.id,
                                                         f"model_epoch_{self.current_epoch}_t{ts}.pth"))

    def _calc_image_metrics(self, logs: Sequence[Dict]):

        # initialize scales
        mse = torch.tensor(0, dtype=torch.float32, device=self.device)
        ssim = torch.tensor(0, dtype=torch.float32, device=self.device)
        psnr = torch.tensor(0, dtype=torch.float32, device=self.device)
        loss = torch.tensor(0, dtype=torch.float32, device=self.device)

        batch_num = len(logs)

        # calculate metrics
        for log in logs:
            mse += FastmrtMetrics.mse(log["output"], log["label"])
            ssim += FastmrtMetrics.ssim(log["output"], log["label"])
            psnr += FastmrtMetrics.psnr(log["output"], log["label"])
            loss += log[f"loss"]

        return {
            'mse': mse / batch_num,
            'ssim': ssim / batch_num,
            'psnr': psnr / batch_num,
            'loss': loss / batch_num,
        }

    def _calc_tmap_metrics(
            self,
            full_tmaps: List,
            recon_tmaps: List
    ) -> None:
        if self.tmap_prf_func is None:
            return

        # init metrics
        tmap_error = torch.tensor(0, dtype=torch.float32, device=self.device)
        patch_mse = torch.tensor(0, dtype=torch.float32, device=self.device)
        heated_area_dice = torch.tensor(0, dtype=torch.float32, device=self.device)
        patch_error_std = torch.tensor(0, dtype=torch.float32, device=self.device)

        # parameters
        tmap_num = len(full_tmaps)
        tmap_height = full_tmaps[0].shape[0]
        tmap_width = full_tmaps[0].shape[1]
        patch_height = tmap_height // self.tmap_patch_rate
        patch_width = tmap_width // self.tmap_patch_rate
        dice_calc_num = 0

        # calculate metrics
        for full_tmap, recon_tmap in zip(full_tmaps, recon_tmaps):

            # metric 1: mean of temperature maps error
            tmap_error += torch.mean(torch.abs(full_tmap - recon_tmap))

            # metric 2: patch root-mean-square error
            patch_full_tmap = full_tmap[(tmap_height - patch_height) // 2: tmap_height - (tmap_height - patch_height) // 2,
                                        (tmap_width - patch_width) // 2: tmap_width - (tmap_width - patch_width) // 2]
            patch_recon_tmap = recon_tmap[(tmap_height - patch_height) // 2: tmap_height - (tmap_height - patch_height) // 2,
                                          (tmap_width - patch_width) // 2: tmap_width - (tmap_width - patch_width) // 2]
            patch_mse += FastmrtMetrics.mse(patch_recon_tmap.unsqueeze(0), patch_full_tmap.unsqueeze(0))

            # metric 3: dice coefficient of heated areas.
            area_full_tmap = (patch_full_tmap > self.tmap_heated_thresh).type(torch.FloatTensor)
            area_recon_tmap = (patch_recon_tmap > self.tmap_heated_thresh).type(torch.FloatTensor)
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

    def _log_medias(
            self,
            logs: Sequence[Dict],
            prefix: str
    ):
        # obtain batch index and sample index according to `self.log_images_frame_idx`
        batch_indices, sample_indices = [], []
        for batch_idx in range(len(logs)):
            for sample_idx in range(logs[batch_idx]["input"].shape[0]):
                if logs[batch_idx]["frame_idx"][sample_idx] == self.log_images_frame_idx:
                    batch_indices += [batch_idx]
                    sample_indices += [sample_idx]

        # hack: control the logging sample number of medias, here we set 20 default.
        if len(batch_indices) > 20:
            batch_indices = batch_indices[::len(batch_indices) // 20]
            sample_indices = sample_indices[::len(sample_indices) // 20]

        for batch_idx, sample_idx in zip(batch_indices, sample_indices):
            
            # obtain section name
            section_name = f"{prefix}_" \
                           f"{logs[batch_idx]['file_name'][sample_idx]}_" \
                           f"f{self.log_images_frame_idx:02d}" \
                           f"s{logs[batch_idx]['slice_idx'][sample_idx]}" \
                           f"c{logs[batch_idx]['coil_idx'][sample_idx]}"

            # obtain input, label & output images
            log_input = logs[batch_idx]["input"][sample_idx].squeeze(0)
            log_label = logs[batch_idx]["label"][sample_idx].squeeze(0)
            log_output = logs[batch_idx]["output"][sample_idx].squeeze(0)

            # calculate root square of images
            log_input_rss = torch.sqrt((log_input**2).sum(dim=0))
            log_label_rss = torch.sqrt((log_label**2).sum(dim=0))
            log_output_rss = torch.sqrt((log_output**2).sum(dim=0))
            log_error_rss = torch.abs(log_output_rss - log_label_rss)

            # add amplitude images to log
            vmin, vmax = log_label_rss.min(), log_label_rss.max()
            self.logger.log_image(f"{section_name}/A_input", [self._vmin_max(log_input_rss, vmin=vmin, vmax=vmax)], self.current_epoch + 1)
            self.logger.log_image(f"{section_name}/B_label", [self._vmin_max(log_label_rss, vmin=vmin, vmax=vmax)], self.current_epoch + 1)
            self.logger.log_image(f"{section_name}/C_output", [self._vmin_max(log_output_rss, vmin=vmin, vmax=vmax)], self.current_epoch + 1)
            self.logger.log_image(f"{section_name}/D_error", [self._vmin_max(log_error_rss, vmin=vmin, vmax=vmax)], self.current_epoch + 1)

            # obtain input, label & output reference images
            log_input_ref = logs[batch_idx]["input_ref"][sample_idx].squeeze(0)
            log_label_ref = logs[batch_idx]["label_ref"][sample_idx].squeeze(0)
            log_output_ref = logs[batch_idx]["output_ref"][sample_idx].squeeze(0)
            log_tmap_mask = logs[batch_idx]["tmap_mask"][sample_idx].squeeze(0)

            # calculate temperature maps
            mask_tmap = self.tmap_prf_func(rt2ct(log_input), rt2ct(log_input_ref)) * log_tmap_mask
            full_tmap = self.tmap_prf_func(rt2ct(log_label), rt2ct(log_label_ref)) * log_tmap_mask
            recon_tmap = self.tmap_prf_func(rt2ct(log_output), rt2ct(log_output_ref)) * log_tmap_mask
            error_tmap = full_tmap - recon_tmap

            # add temperature maps to log
            self._log_tmap(mask_tmap, fig_name=f"{section_name}/E_mask_tmap")
            self._log_tmap(full_tmap, fig_name=f"{section_name}/F_full_tmap")
            self._log_tmap(recon_tmap, fig_name=f"{section_name}/G_recon_tmap")
            self._log_tmap(error_tmap, fig_name=f"{section_name}/H_error_tmap", vmin=-10, vmax=10)

            # add bland-altman analysis & linear regression to log
            tmap_height = full_tmap.shape[0]
            tmap_width = recon_tmap.shape[1]
            patch_height = tmap_height // self.tmap_patch_rate
            patch_width = tmap_width // self.tmap_patch_rate
            patch_full_tmap = full_tmap[(tmap_height - patch_height) // 2: tmap_height - (tmap_height - patch_height) // 2,
                              (tmap_width - patch_width) // 2: tmap_width - (tmap_width - patch_width) // 2]
            patch_recon_tmap = recon_tmap[(tmap_height - patch_height) // 2: tmap_height - (tmap_height - patch_height) // 2,
                              (tmap_width - patch_width) // 2: tmap_width - (tmap_width - patch_width) // 2]
            ba_mean, ba_error, ba_error_mean, ba_error_std, _ = FastmrtMetrics.bland_altman(patch_recon_tmap, patch_full_tmap)
            with draw_bland_altman_fig(ba_mean, ba_error, ba_error_mean, ba_error_std) as plt:
                self.logger.experiment.log({f"{section_name}/I_bland_altman": wandb.Image(plt)})
            with draw_linear_regression_fig(patch_recon_tmap, patch_full_tmap) as plt:
                self.logger.experiment.log({f"{section_name}/J_linear_regression": wandb.Image(plt)})


    def _log_tmap(self, tmap: torch.Tensor, fig_name: str, vmin: Any=0.0, vmax: Any=70.0) -> None:
        fig = plt.figure(fig_name)
        plt.imshow(tmap.cpu(), cmap='jet', vmin=vmin, vmax=vmax)
        plt.colorbar()
        self.logger.experiment.log({fig_name: plt})
        plt.close(fig)
    
    def _log_scalar(self, metrics: Dict, stage: str="") -> None:
        for key, val in metrics.items():
            if stage:
                key = f"{stage}_{key}"
            self.log(key, val)

    @staticmethod
    def _vmin_max(image, vmin, vmax):
        return (image - image.min()) / (vmax - vmin)



class FastmrtModule(BaseModule):
    """
    This is a PyTorch-Lightning module general template that defines 
    the standard FastMRT training, validation and test processing.

    This class provides some default settings:
    - The batch data type is real-type, which means using two channels as real and imag of a complex image, and the batch shape is [B, 2, H, W].
    - The optimizer and scheduler are set to use `AdamW` optimizer and `CosineAnnealingLR` as the default settings.
    - The loss function has two options: l1-loss and decoupled-loss.

    If you have custom requirements, you can inherit this class and 
    override the corresponding functions.

    Args:
        loss_type (str): One of `l1` and `decoupled`.
            Specifies the type of loss function to use.
        max_epochs (int): The number of training epochs.
            Defaults to 10.
        lr (float): The learning rate.
            Defaults to 0.001.
        weight_decay (float): The weight decay.
            Defaults to 0.0.
    """

    def __init__(
            self, 
            *args, 
            loss_type: str='l1',
            max_epochs: int=200,
            lr: float = 5e-4,
            weight_decay: float = 1e-4,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        
        if loss_type == 'l1':
            self.loss_fn = self._l1_loss
        elif loss_type == 'decoupled':
            self.loss_fn = self._decoupled_loss
        else:
            raise ValueError(f"`loss_type` must be one of `l1` and `decoupled`, but {loss_type} was got.")
        
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay

        # to_eval: denormalize (and to real-type data format)
        self.to_eval = lambda x, m, s : ct2rt(denormalize(x, m, s).squeeze()) if torch.is_complex(x) else denormalize(x, m, s).squeeze()
    
    def training_step(self, batch, batch_idx, **kwargs):

        return {"loss": self.loss_fn(self.model(batch.input, **kwargs), batch.label)}

    def validation_step(self, batch, batch_idx, **kwargs):

        # obtain output and reference frame output
        output = self.model(batch.input, **kwargs)
        output_ref = self.model(batch.input_ref, **kwargs)

        # calculate validation loss
        val_loss = self.loss_fn(output, batch.label)

        # obtain mean and standard diviation for de-normailzation
        mean = batch.mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        std = batch.std.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        return {
            "input": self.to_eval(batch.input, mean, std),
            "label": self.to_eval(batch.label, mean, std),
            "output": self.to_eval(output, mean, std),
            "input_ref": self.to_eval(batch.input_ref, mean, std),
            "label_ref": self.to_eval(batch.label_ref, mean, std),
            "output_ref": self.to_eval(output_ref, mean, std),
            "tmap_mask": batch.tmap_mask,
            "loss": val_loss,
            "file_name": batch.metainfo['file_name'],
            "frame_idx": batch.metainfo['frame_idx'],
            "slice_idx": batch.metainfo['slice_idx'],
            "coil_idx": batch.metainfo['coil_idx'],
        }

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def predict_step(self, batch):
        output = self.model(batch.input)
        pred_loss = self.loss_fn(output, batch.label)
        mean = batch.mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        std = batch.std.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return {
            "output": self.to_eval(output, mean, std),
            "label": self.to_eval(batch.label, mean, std),
            "pred_loss": pred_loss,
        }

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.lr,
                                      weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=self.max_epochs,
                                                               last_epoch=-1)
        return [optimizer], [scheduler]

    def _l1_loss(self, output, label):

        return F.l1_loss(output, label)

    def _decoupled_loss(self, output, label):

        alpha = 2   # phase loss coefficient
        
        # turn to complex data format
        if torch.is_complex(label):
            output_complex = output
            label_complex = label
        else:
            output_complex = rt2ct(output)
            label_complex = rt2ct(label)

        # amplitude loss
        amp_loss = F.l1_loss(output_complex.abs(), label_complex.abs())

        # phase loss
        phs_loss = (torch.angle(output_complex * torch.conj(label_complex)) * label_complex.abs()).abs().mean(0).mean()

        return amp_loss + phs_loss * alpha