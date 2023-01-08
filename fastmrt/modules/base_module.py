"""
base_module主要做两件事情
1. 重写training_epoch_end()、valid_epoch_end()、test_epoch_end()
    用于写进log。因为所有模型在这一部分是通用的。
2. 重写training_step_end()、valid_step_end()、test_step_end()
    用于在分布式训练中收集所有GPU计算结果汇总。
"""


import pytorch_lightning as pl
from typing import Dict, Sequence
from fastmrt.utils.metrics import FastmrtMetrics
from fastmrt.utils.rss import rss_tensor
from fastmrt.utils.trans import real_tensor_to_complex_tensor as rt2ct
from fastmrt.data.prf import PrfFunc
import torch
import numpy as np
from typing import Any, List
import matplotlib
import wandb

matplotlib.use('agg')

from matplotlib import pyplot as plt


class BaseModule(pl.LightningModule):

    def __init__(
            self,
            tmap_prf_func: PrfFunc = None,
            tmap_patch_rate: int = 12,
            tmap_max_temp_thresh = 45,
            tmap_ablation_thresh = 57,
            log_images_frame_idx: int = 6, # recommend 4 ~ 8
            log_images_freq: int = 50,
    ):
        super(BaseModule, self).__init__()
        self.tmap_prf_func = tmap_prf_func
        self.tmap_patch_rate = tmap_patch_rate
        self.tmap_max_temp_thresh = tmap_max_temp_thresh
        self.tmap_ablation_thresh = tmap_ablation_thresh
        self.log_images_frame_idx = log_images_frame_idx
        self.log_images_freq = log_images_freq

    def training_epoch_end(self, train_logs: Sequence[Dict]) -> None:
        train_loss = torch.tensor(0, dtype=torch.float32, device='cuda')
        for log in train_logs:
            train_loss += log["loss"]
        self.log("loss", train_loss, on_epoch=True, on_step=False)

    def validation_epoch_end(self, val_logs: Sequence[Dict]) -> None:
        # save image metrics
        self._log_image_metrics(val_logs, stage='val')

        # save tmap metrics
        full_tmaps, recon_tmaps = [], []
        for log in val_logs:
            for sample_idx in range(log["input"].shape[0]):
                if log["frame_idx"][sample_idx] > 0: # we only focus on temperature maps after first frame.
                    full_tmaps += [self.tmap_prf_func(rt2ct(log["label"][sample_idx]),
                                               rt2ct(log["label_ref"][sample_idx])) * log["tmap_mask"][sample_idx]]
                    recon_tmaps += [self.tmap_prf_func(rt2ct(log["output"][sample_idx]),
                                               rt2ct(log["output_ref"][sample_idx])) * log["tmap_mask"][sample_idx]]
        self._log_tmap_metrics(full_tmaps, recon_tmaps)

        # save log medias (images & tmaps)
        if (self.current_epoch + 1) % self.log_images_freq == 0:
            self._log_medias(val_logs, f"val_medias")

    def test_epoch_end(self, test_logs) -> None:
        pass

    def training_step_end(self, train_log) -> Dict:
        pass

    def validation_step_end(self, val_log) -> Dict:
        pass

    def test_step_end(self, test_log) -> Dict:
        pass

    def _log_image_metrics(self, logs: Sequence[Dict], stage: str = "val"):
        # initialize scales
        mse = torch.tensor(0, dtype=torch.float32, device='cuda')
        ssim = torch.tensor(0, dtype=torch.float32, device='cuda')
        psnr = torch.tensor(0, dtype=torch.float32, device='cuda')
        loss = torch.tensor(0, dtype=torch.float32, device='cuda')

        batch_num = len(logs)

        # calculate metrics
        for log in logs:
            mse += FastmrtMetrics.mse(log["output"], log["label"])
            ssim += FastmrtMetrics.ssim(log["output"], log["label"])
            psnr += FastmrtMetrics.psnr(log["output"], log["label"])
            loss += log[f"{stage}_loss"]

        # save validation logs of metrics and loss
        self.log(f"{stage}_mse", mse / batch_num)
        self.log(f"{stage}_ssim", ssim / batch_num)
        self.log(f"{stage}_psnr", psnr / batch_num)
        self.log(f"{stage}_loss", loss / batch_num)

    def _log_tmap_metrics(
            self,
            full_tmaps: List,
            recon_tmaps: List
    ) -> None:
        if self.tmap_prf_func is None:
            return

        # init metrics
        temp_error = torch.tensor(0, dtype=torch.float32, device="cuda")
        patch_rmse = torch.tensor(0, dtype=torch.float32, device="cuda")
        max_temp_error = torch.tensor(0, dtype=torch.float32, device="cuda")
        max_temp_dist = torch.tensor(0, dtype=torch.float32, device="cuda")
        ablation_area_dice = torch.tensor(0, dtype=torch.float32, device="cuda")
        patch_ba_error_mean = torch.tensor(0, dtype=torch.float32, device="cuda")
        patch_ba_error_std = torch.tensor(0, dtype=torch.float32, device="cuda")
        patch_ba_out_loa = torch.tensor(0, dtype=torch.float32, device="cuda")

        # tmap and patch parameters
        tmap_num = len(full_tmaps)
        tmap_height = full_tmaps[0].shape[0]
        tmap_width = full_tmaps[0].shape[1]
        patch_height = tmap_height // self.tmap_patch_rate
        patch_width = tmap_width // self.tmap_patch_rate
        dice_calc_num = 0
        max_temp_num = 0

        # calculate metrics
        for full_tmap, recon_tmap in zip(full_tmaps, recon_tmaps):

            # metric 1: mean of tmap error
            temp_error += torch.mean(torch.abs(full_tmap - recon_tmap))

            # metric 2: patch root-mean-square error
            patch_full_tmap = full_tmap[(tmap_height - patch_height) // 2: tmap_height - (tmap_height - patch_height) // 2,
                              (tmap_width - patch_width) // 2: tmap_width - (tmap_width - patch_width) // 2]
            patch_recon_tmap = recon_tmap[(tmap_height - patch_height) // 2: tmap_height - (tmap_height - patch_height) // 2,
                              (tmap_width - patch_width) // 2: tmap_width - (tmap_width - patch_width) // 2]
            patch_rmse += FastmrtMetrics.mse(patch_recon_tmap.unsqueeze(0), patch_full_tmap.unsqueeze(0))

            # metric 3 & 4: max temperature error & distance
            max_temp_full_tmap = torch.max(patch_full_tmap)
            max_temp_recon_tmap = torch.max(patch_recon_tmap)
            if max_temp_recon_tmap > self.tmap_max_temp_thresh:
                max_temp_pos_full_tmap = torch.argmax(patch_full_tmap)      # max value position of full tmap
                max_temp_pos_recon_tmap = torch.argmax(patch_recon_tmap)    # max value position of recon tmap
                error_vector = [max_temp_pos_full_tmap % patch_height - max_temp_pos_recon_tmap % patch_height,
                                max_temp_pos_full_tmap // patch_width - max_temp_pos_recon_tmap // patch_width]
                max_temp_error += torch.abs(max_temp_full_tmap - max_temp_recon_tmap)
                max_temp_dist += torch.sqrt((error_vector[0] / tmap_height) ** 2 + (error_vector[1] / tmap_width) ** 2)
                max_temp_num += 1

            # metric 5: dice coefficient of ablation area
            area_full_tmap = patch_full_tmap > self.tmap_ablation_thresh
            area_recon_tmap = patch_recon_tmap > self.tmap_ablation_thresh
            if torch.sum(area_full_tmap) > 1: # ensuring the ablation area exist
                ablation_area_dice += FastmrtMetrics.dice(area_recon_tmap.unsqueeze(0), area_full_tmap.unsqueeze(0))
                dice_calc_num += 1

            # metric 6: bland-altman analysis outer LoA
            _, _, ba_error_mean, ba_error_std, ba_out_loa = FastmrtMetrics.bland_altman(patch_recon_tmap, patch_full_tmap)
            patch_ba_error_mean += ba_error_mean
            patch_ba_error_std += ba_error_std
            patch_ba_out_loa += ba_out_loa


        # add tmap metrics to log
        self.log("T_error", temp_error / tmap_num)
        self.log("T_patch_rmse", patch_rmse / tmap_num)
        self.log("T_max_error", max_temp_error / max_temp_num)
        self.log("T_max_dist", max_temp_dist / max_temp_num)
        self.log("T_ablation_area_dice", ablation_area_dice / dice_calc_num)
        self.log("T_patch_ba_error_mean", patch_ba_error_mean / tmap_num)
        self.log("T_patch_ba_error_std", patch_ba_error_std / tmap_num)
        self.log("T_patch_ba_out_loa", patch_ba_out_loa / tmap_num)

    def _log_medias(
            self,
            logs: Sequence[Dict],
            prefix: str
    ):
        # obtain index
        batch_indices, sample_indices = [], []
        for batch_idx in range(len(logs)):
            for sample_idx in range(logs[batch_idx]["input"].shape[0]):
                if logs[batch_idx]["frame_idx"][sample_idx] == self.log_images_frame_idx:
                    batch_indices += [batch_idx]
                    sample_indices += [sample_idx]

        # control the logging sample number of medias
        if len(batch_indices) > 20:
            batch_indices = batch_indices[::len(batch_indices) // 20]
            sample_indices = sample_indices[::len(sample_indices) // 20]

        for batch_idx, sample_idx in zip(batch_indices, sample_indices):
            # obtain section name
            section_name = f"{prefix}_" \
                           f"{logs[batch_idx]['file_name'][sample_idx]}_" \
                           f"f{self.log_images_frame_idx}" \
                           f"s{logs[batch_idx]['slice_idx'][sample_idx]}" \
                           f"c{logs[batch_idx]['coil_idx'][sample_idx]}"

            # obtain input, label & output images
            log_input = logs[batch_idx]["input"][sample_idx].squeeze(0)
            log_label = logs[batch_idx]["label"][sample_idx].squeeze(0)
            log_output = logs[batch_idx]["output"][sample_idx].squeeze(0)

            # calculate root square of images
            log_input_rss = rss_tensor(log_input, dim=0).unsqueeze(0)
            log_label_rss = rss_tensor(log_label, dim=0).unsqueeze(0)
            log_output_rss = rss_tensor(log_output, dim=0).unsqueeze(0)

            # normalization
            log_input_rss = (log_input_rss - log_input_rss.min()) / (log_input_rss.max() - log_input_rss.min())
            log_label_rss = (log_label_rss - log_label_rss.min()) / (log_label_rss.max() - log_label_rss.min())
            log_output_rss = (log_output_rss - log_output_rss.min()) / (log_output_rss.max() - log_output_rss.min())
            log_error_rss = torch.abs(log_output_rss - log_label_rss)

            # add amplitude images to log
            self.logger.log_image(f"{section_name}/A_input", [log_input_rss], self.current_epoch + 1)
            self.logger.log_image(f"{section_name}/B_label", [log_label_rss], self.current_epoch + 1)
            self.logger.log_image(f"{section_name}/C_output", [log_output_rss], self.current_epoch + 1)
            self.logger.log_image(f"{section_name}/D_error", [log_error_rss], self.current_epoch + 1)

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
            self._log_tmap(recon_tmap, fig_name=f"{section_name}/G_recon_temp")
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
            self._log_bland_altman_fig(f"{section_name}/I_bland_altman", ba_mean, ba_error, ba_error_mean, ba_error_std)
            self._log_linear_regression_fig(f"{section_name}/J_linear_regression", patch_recon_tmap, patch_full_tmap)


    def _log_tmap(self, tmap: torch.Tensor, fig_name: str, vmin: Any=0.0, vmax: Any=70.0) -> None:
        fig = plt.figure(fig_name)
        plt.imshow(tmap.cpu(), cmap='jet', vmin=vmin, vmax=vmax)
        plt.colorbar()
        self.logger.experiment.log({fig_name: plt})
        plt.close(fig)

    def _log_bland_altman_fig(self, fig_name: str, ba_mean, ba_error, ba_error_mean, ba_error_std):
        # calculate datas
        mean_error = ba_error_mean.cpu().numpy()
        loa_upper_limit = ba_error_mean.cpu().numpy() + 1.96 * ba_error_std.cpu().numpy()
        loa_lower_limit = ba_error_mean.cpu().numpy() - 1.96 * ba_error_std.cpu().numpy()
        # start plot
        fig = plt.figure(fig_name)
        plt.scatter(ba_mean.cpu().numpy(), ba_error.cpu().numpy())
        plt.axhline(mean_error, color='gray', linestyle='-')
        plt.axhline(loa_upper_limit, color='red', linestyle='--')
        plt.axhline(loa_lower_limit, color='red', linestyle='--')
        plt.text(0, mean_error, "mean: {:.3f}".format(mean_error))
        plt.text(0, loa_upper_limit, "upper limit: {:.3f}".format(loa_upper_limit))
        plt.text(0, loa_lower_limit, "lower limit: {:.3f}".format(loa_lower_limit))
        plt.xlabel("Mean of Recon and Full Tmap Patches (℃)")
        plt.ylabel("Difference (℃)")
        plt.title("Bland-Altman Analysis")
        # save to log
        self.logger.experiment.log({fig_name: wandb.Image(plt)})
        plt.close(fig)

    def _log_linear_regression_fig(self, fig_name: str,  patch_recon_tmap, patch_full_tmap):
        # calculate datas
        data_x = patch_recon_tmap.flatten().cpu().numpy()
        data_y = patch_full_tmap.flatten().cpu().numpy()
        [k, b] = np.polyfit(data_x, data_y, deg=1)
        ref_x = np.linspace(np.min(data_x), np.max(data_x), 10)
        ref_y = ref_x
        fit_x = ref_x
        fit_y = k * fit_x + b
        # start plot
        fig = plt.figure(fig_name)
        plt.plot(data_x, data_y, '.')
        plt.plot(ref_x, ref_y, color="red", linestyle="--")
        plt.plot(fit_x, fit_y, color="blue", linestyle="-")
        plt.text(ref_x[7], ref_y[4], "y={:.3f}x+{:.3f}".format(k, b))
        plt.xlabel("Temperature of Recon Tmap (℃)")
        plt.ylabel("Temperature of Full Tmap (℃)")
        plt.title("Linear Regression Analysis")
        # save to log
        self.logger.experiment.log({fig_name: wandb.Image(plt)})
        plt.close(fig)

