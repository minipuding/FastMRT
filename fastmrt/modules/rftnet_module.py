import torch
import torch.nn.functional as F
from fastmrt.models.rftnet import RFTNet
from fastmrt.modules.base_module import BaseModule
from fastmrt.utils.normalize import denormalize
from fastmrt.utils.metrics import FastmrtMetrics
from fastmrt.utils.trans import real_tensor_to_complex_tensor as rt2ct
from fastmrt.data.prf import PrfFunc
from typing import Tuple, Sequence, Dict


class RFTNetModule(BaseModule):
    def __init__(
        self,
        in_channels: Tuple[int] = (2, 2, 2),
        out_channels: Tuple[int] = (2, 2, 1),
        base_channels: Tuple[int] = (32, 32, 32),
        level_num: Tuple[int] = (4, 4, 4),
        drop_prob: Tuple[float] = (0.0, 0.0, 0.0),
        leakyrelu_slope: Tuple[float] = (0.4, 0.4, 0.4),
        last_layer_with_act: bool = False,
        lr: float = 1e-3,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        tmap_prf_func: PrfFunc = None,
        tmap_patch_rate: int = 4,
        tmap_max_temp_thresh = 45,
        tmap_ablation_thresh = 43,
        log_images_frame_idx: int = 5, # recommend 4 ~ 8
        log_images_freq: int = 50,
    ):
        super(RFTNetModule, self).__init__(tmap_prf_func=tmap_prf_func,
                                           tmap_patch_rate=tmap_patch_rate,
                                           tmap_max_temp_thresh=tmap_max_temp_thresh,
                                           tmap_ablation_thresh=tmap_ablation_thresh,
                                           log_images_frame_idx=log_images_frame_idx,
                                           log_images_freq=log_images_freq)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.level_num = level_num
        self.drop_prob = drop_prob
        self.leakyrelu_slope = leakyrelu_slope
        self.last_layer_with_act = last_layer_with_act
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.model = RFTNet(in_channels=self.in_channels,
                          out_channels=self.out_channels,
                          base_channels=self.base_channels,
                          level_num=self.level_num,
                          drop_prob=self.drop_prob,
                          leakyrelu_slope=self.leakyrelu_slope,
                          last_layer_with_act=self.last_layer_with_act,
                          )

    def training_step(self, batch, batch_idx, weight=4):
        output_phs, output_ref, output_rec = self.model(batch.input)
        train_loss = F.l1_loss(output_phs, batch.label_phs) * weight + \
                     F.l1_loss(output_ref, batch.label_ref) + \
                     F.l1_loss(output_rec, batch.label_img)# + \
                     # 10 * torch.mean(torch.abs(rt2ct(output_phs)) * torch.abs(rt2ct(batch.label_phs)) - torch.sum(output_phs * batch.label_phs, dim=-3)) + \
                     # 10 * torch.mean(torch.abs(rt2ct(output_ref)) * torch.abs(rt2ct(batch.label_ref)) - torch.sum(output_ref * batch.label_ref, dim=-3)) + \
                     # 10 * torch.mean(torch.abs(rt2ct(output_rec)) * torch.abs(rt2ct(batch.label_img)) - torch.sum(output_rec * batch.label_img, dim=-3))
        # train_loss = torch.sum(torch.abs(torch.abs(rt2ct(output_phs)) - torch.abs(rt2ct(batch.label_phs)))) * weight + \
        #              torch.sum(torch.abs(torch.abs(rt2ct(output_ref)) - torch.abs(rt2ct(batch.label_ref)))) + \
        #              torch.sum(torch.abs(torch.abs(rt2ct(output_rec)) - torch.abs(rt2ct(batch.label_img)))) + \
        #              torch.sum(torch.abs(rt2ct(output_phs)) * torch.abs(rt2ct(batch.label_phs)) - torch.sum(
        #                  output_phs * batch.label_phs, dim=-3)) + \
        #              torch.sum(torch.abs(rt2ct(output_ref)) * torch.abs(rt2ct(batch.label_ref)) - torch.sum(
        #                  output_ref * batch.label_ref, dim=-3)) + \
        #              torch.sum(torch.abs(rt2ct(output_rec)) * torch.abs(rt2ct(batch.label_img)) - torch.sum(
        #                  output_rec * batch.label_img, dim=-3))
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        output_phs = self.model(batch.input, stage="val")
        val_loss = F.l1_loss(output_phs, batch.label_phs)
        return {
            "input": batch.input,
            "label_phs": batch.label_phs,
            "output_phs": output_phs,
            "tmap_mask": batch.tmap_mask,
            "val_loss": val_loss,
            "file_name": batch.file_name,
            "frame_idx": batch.frame_idx,
            "slice_idx": batch.slice_idx,
            "coil_idx": batch.coil_idx,
        }

    def validation_epoch_end(self, val_logs: Sequence[Dict]) -> None:
        # save validation loss
        loss = torch.tensor(0, dtype=torch.float32, device='cuda')
        for log in val_logs:
            loss += log["val_loss"]
        self.log("val_loss", loss / len(val_logs))

        # save tmap metrics
        full_tmaps, recon_tmaps = [], []
        for log in val_logs:
            for sample_idx in range(log["input"].shape[0]):
                if log["frame_idx"][sample_idx] > 0: # we only focus on temperature maps after first frame.
                    full_tmaps += [self.tmap_prf_func(delta_phs = log["label_phs"][sample_idx], is_phs=True) * log["tmap_mask"][sample_idx]]
                    recon_tmaps += [self.tmap_prf_func(delta_phs = log["output_phs"][sample_idx], is_phs=True) * log["tmap_mask"][sample_idx]]
        self._log_tmap_metrics(full_tmaps, recon_tmaps)

        # save log medias (images & tmaps)
        if (self.current_epoch + 1) % self.log_images_freq == 0:
            self._log_medias(val_logs, f"val_medias")

    def test_step(self, batch):
        output = self.model(batch.input, batch.origin_shape, batch.mask, batch.mean, batch.std)
        mean = batch.mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        std = batch.std.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return {
            "input": (batch.input * std + mean),
            "output": (output * std + mean),
            "target": (batch.label * std + mean),
        }

    def predict_step(self, batch):
        output = self.model(batch.input, batch.origin_shape, batch.mask)
        pred_loss = F.l1_loss(output, batch.label)
        mean = batch.mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        std = batch.std.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return {
            "output": denormalize(output, mean, std),
            "label": denormalize(batch.label, mean, std),
            "pred_loss": pred_loss,
        }

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(),
        #                              lr=self.lr,
        #                              weight_decay=self.weight_decay,
        #                             momentum=0.9,
        #                             nesterov=True)
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay,
                                     betas=(0.9, 0.999),
                                     eps=1e-8)
        # optimizer = torch.optim.RMSprop(self.parameters(),
        #                                 lr=self.lr,
        #                                 weight_decay=self.weight_decay,
        #                                 momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    step_size=self.lr_step_size,
                                                    gamma=self.lr_gamma)
        return [optimizer], [scheduler]


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

        for batch_idx, sample_idx in zip(batch_indices, sample_indices):
            # obtain section name
            section_name = f"{prefix}_" \
                           f"{logs[batch_idx]['file_name'][sample_idx]}_" \
                           f"f{self.log_images_frame_idx}" \
                           f"s{logs[batch_idx]['slice_idx'][sample_idx]}" \
                           f"c{logs[batch_idx]['coil_idx'][sample_idx]}"

            # obtain input, label & output images
            log_label_phs = logs[batch_idx]["label_phs"][sample_idx].squeeze(0)
            log_output_phs = logs[batch_idx]["output_phs"][sample_idx].squeeze(0)
            log_tmap_mask = logs[batch_idx]["tmap_mask"][sample_idx].squeeze(0)

            # calculate temperature maps
            full_tmap = self.tmap_prf_func(delta_phs = log_label_phs, is_phs=True) * log_tmap_mask
            recon_tmap = self.tmap_prf_func(delta_phs = log_output_phs, is_phs=True) * log_tmap_mask
            error_tmap = full_tmap - recon_tmap

            # add temperature maps to log
            self._log_tmap(full_tmap, fig_name=f"{section_name}/F_full_tmap")
            self._log_tmap(recon_tmap, fig_name=f"{section_name}/G_recon_temp")
            self._log_tmap(error_tmap, fig_name=f"{section_name}/H_error_tmap", vmin=-10, vmax=10)

            # add bland-altman analysis & linear regression to log
            tmap_height = full_tmap.shape[0]
            tmap_width = full_tmap.shape[1]
            patch_height = tmap_height // self.tmap_patch_rate
            patch_width = tmap_width // self.tmap_patch_rate
            patch_full_tmap = full_tmap[(tmap_height - patch_height) // 2: tmap_height - (tmap_height - patch_height) // 2,
                              (tmap_width - patch_width) // 2: tmap_width - (tmap_width - patch_width) // 2]
            patch_recon_tmap = recon_tmap[(tmap_height - patch_height) // 2: tmap_height - (tmap_height - patch_height) // 2,
                              (tmap_width - patch_width) // 2: tmap_width - (tmap_width - patch_width) // 2]
            ba_mean, ba_error, ba_error_mean, ba_error_std, _ = FastmrtMetrics.bland_altman(patch_recon_tmap, patch_full_tmap)
            self._log_bland_altman_fig(f"{section_name}/I_bland_altman", ba_mean, ba_error, ba_error_mean, ba_error_std)
            self._log_linear_regression_fig(f"{section_name}/J_linear_regression", patch_recon_tmap, patch_full_tmap)

