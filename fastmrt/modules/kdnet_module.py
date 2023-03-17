import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmrt.models.kdnet import KDNet
from fastmrt.modules.base_module import BaseModule
from fastmrt.utils.trans import complex_tensor_to_real_tensor as ct2rt
from fastmrt.utils.trans import real_tensor_to_complex_tensor as rt2ct
from fastmrt.utils.normalize import denormalize
from fastmrt.data.prf import PrfFunc
from typing import Sequence, Dict


class KDNetModule(BaseModule):
    """
    About manual optimizer:
    https://pytorch-lightning.readthedocs.io/en/stable/model/manual_optimization.html#manual-optimization
    """
    def __init__(
            self,
            tea_net: nn.Module,
            stu_net: nn.Module,
            use_ema: bool = False,
            soft_label_weight: float = 2.0,
            lr_stu: float = 5e-4,
            weight_decay_stu: float = 0.0,
            tmap_prf_func: PrfFunc = None,
            tmap_patch_rate: int = 4,
            tmap_max_temp_thresh: int = 45,
            tmap_ablation_thresh: int = 43,
            log_images_frame_idx: int = 5,  # recommend 4 ~ 8
            log_images_freq: int = 50,
    ):
        super(KDNetModule, self).__init__(tmap_prf_func=tmap_prf_func,
                                          tmap_patch_rate=tmap_patch_rate,
                                          tmap_max_temp_thresh=tmap_max_temp_thresh,
                                          tmap_ablation_thresh=tmap_ablation_thresh,
                                          log_images_frame_idx=log_images_frame_idx,
                                          log_images_freq=log_images_freq)
        self.tea_net = tea_net
        self.stu_net = stu_net
        self.use_ema = use_ema
        # assert (1 >= soft_label_weight >= 0), f"`soft_label_weight` must between 0 and 1, but got {soft_label_weight}"
        self.soft_label_weight = soft_label_weight
        self.lr_stu = lr_stu
        self.weight_decay_stu = weight_decay_stu
        self.model = KDNet(tea_net, stu_net, use_ema, soft_label_weight)

    def training_step(self, batch, batch_idx):
        tea_loss, stu_loss, soft_label_loss, kd_loss = self._decoupled_loss_v5(batch, beta1=0.63, beta2=0.37)

        return {"loss": kd_loss,
                "tea_loss": tea_loss,
                "stu_loss": stu_loss,
                "soft_label_loss": soft_label_loss}

    def training_epoch_end(self, train_logs: Sequence[Dict]) -> None:
        train_loss = torch.tensor(0, dtype=torch.float32, device='cuda')
        tea_loss = torch.tensor(0, dtype=torch.float32, device='cuda')
        stu_loss = torch.tensor(0, dtype=torch.float32, device='cuda')
        soft_label_loss = torch.tensor(0, dtype=torch.float32, device='cuda')
        for log in train_logs:
            train_loss += log["loss"]
            tea_loss += log["tea_loss"]
            stu_loss += log["stu_loss"]
            soft_label_loss += log["soft_label_loss"]
        self.log("loss", train_loss, on_epoch=True, on_step=False)
        self.log("tea_loss", tea_loss, on_epoch=True, on_step=False)
        self.log("stu_loss", stu_loss, on_epoch=True, on_step=False)
        self.log("soft_label_loss", soft_label_loss, on_epoch=True, on_step=False)

    def validation_step(self, batch, batch_idx):
        _, output = self.model(batch.input_stu)
        _, output_ref = self.model(batch.input_ref)
        val_loss = F.l1_loss(output, batch.label)
        mean = batch.mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        std = batch.std.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return {
            "input": denormalize(batch.input_stu, mean, std),
            "label": denormalize(batch.label, mean, std),
            "output": denormalize(output, mean, std),
            "input_ref": denormalize(batch.input_ref, mean, std),
            "label_ref": denormalize(batch.label_ref, mean, std),
            "output_ref": denormalize(output_ref, mean, std),
            "tmap_mask": batch.tmap_mask,
            "val_loss": val_loss,
            "file_name": batch.file_name,
            "frame_idx": batch.frame_idx,
            "slice_idx": batch.slice_idx,
            "coil_idx": batch.coil_idx,
        }

    def test_step(self, batch):
        output = self.model(batch.input)
        mean = batch.mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        std = batch.std.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return {
            "input": (batch.input * std + mean),
            "output": (output * std + mean),
            "target": (batch.label * std + mean),
        }

    def predict_step(self, batch):
        output = self.model(batch.input)
        pred_loss = F.l1_loss(output, batch.label)
        mean = batch.mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        std = batch.std.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return {
            "output": denormalize(output, mean, std),
            "label": denormalize(batch.label, mean, std),
            "pred_loss": pred_loss,
        }

    def configure_optimizers(self):
        optimizer_stu = torch.optim.Adam(self.model.stu_net.parameters(),
                                         lr=self.lr_stu,
                                         weight_decay=self.weight_decay_stu)

        scheduler_stu = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_stu,
                                                                   T_max=200,
                                                                   last_epoch=-1)
        return [optimizer_stu], [scheduler_stu]

    def _l1_loss(self, batch):
        # model forward (only forward stu_input)
        tea_output, stu_output = self.model(batch.input_stu)

        # calculate losses
        tea_loss = F.l1_loss(tea_output, batch.label)
        stu_loss = F.l1_loss(stu_output, batch.label)
        soft_label_loss = F.l1_loss(stu_output, tea_output.clone().detach())
        kd_loss = stu_loss + self.soft_label_weight * soft_label_loss

        return tea_loss, stu_loss, soft_label_loss, kd_loss

    def _decoupled_loss(self, batch, beta1=0.5, beta2=0.5):
        # model forward (only forward stu_input)
        tea_output, stu_output = self.model(batch.input_stu)

        tea_output_complex = rt2ct(tea_output.clone().detach())
        stu_output_complex = rt2ct(stu_output)
        label_complex = rt2ct(batch.label)

        # amplitude loss
        stu_amp_loss = F.l1_loss(stu_output_complex.abs(), label_complex.abs())
        tea_amp_loss = F.l1_loss(tea_output_complex.abs(), label_complex.abs())
        soft_amp_loss = F.l1_loss(stu_output_complex.abs(), tea_output_complex.abs())

        # phase loss
        stu_phs_loss = (torch.angle(stu_output_complex * torch.conj(label_complex)) * label_complex.abs()).abs().sum(0).mean()
        tea_phs_loss = (torch.angle(tea_output_complex * torch.conj(label_complex)) * label_complex.abs()).abs().sum(0).mean()
        soft_phs_loss = (torch.angle(stu_output_complex * torch.conj(tea_output_complex)) * tea_output_complex.abs()).abs().sum(0).mean()

        # merging
        amp_loss = (1 - beta1) * stu_amp_loss + beta1 * soft_amp_loss
        phs_loss = (1 - beta2) * stu_phs_loss + beta2 * soft_phs_loss

        tea_loss = tea_amp_loss + tea_phs_loss / torch.pi
        stu_loss = stu_amp_loss + stu_phs_loss / torch.pi
        soft_label_loss = soft_amp_loss + soft_phs_loss / torch.pi
        kd_loss = amp_loss + phs_loss / torch.pi

        return tea_loss, stu_loss, soft_label_loss, kd_loss

    def _decoupled_loss_v5(self, batch, beta1=0.5, beta2=0.5):
        # setting loss weight decay
        gamma1, gamma2 = 0.6, 0.4
        end_epoch = 200
        beta1 = max((1 - self.current_epoch / end_epoch) * gamma1, 0)
        beta2 = max((1 - self.current_epoch / end_epoch) * gamma2, 0)

        # model forward (only forward stu_input)
        tea_output, stu_output = self.model(batch.input_stu)

        tea_output_complex = rt2ct(tea_output.clone().detach())
        stu_output_complex = rt2ct(stu_output)
        label_complex = rt2ct(batch.label)

        # amplitude loss
        stu_amp_loss = F.l1_loss(stu_output_complex.abs(), label_complex.abs())
        tea_amp_loss = F.l1_loss(tea_output_complex.abs(), label_complex.abs())
        soft_amp_loss = F.l1_loss(stu_output_complex.abs(), tea_output_complex.abs())

        # phase loss
        stu_phs_loss = (torch.angle(stu_output_complex * torch.conj(label_complex)) * label_complex.abs()).abs().sum(0).mean()
        tea_phs_loss = (torch.angle(tea_output_complex * torch.conj(label_complex)) * label_complex.abs()).abs().sum(0).mean()
        soft_phs_loss = (torch.angle(stu_output_complex * torch.conj(tea_output_complex)) * tea_output_complex.abs()).abs().sum(0).mean()

        # merging
        amp_loss = (1 - beta1) * stu_amp_loss + beta1 * soft_amp_loss
        phs_loss = (1 - beta2) * stu_phs_loss + beta2 * soft_phs_loss

        tea_loss = tea_amp_loss + tea_phs_loss / torch.pi
        stu_loss = stu_amp_loss + stu_phs_loss / torch.pi
        soft_label_loss = soft_amp_loss + soft_phs_loss / torch.pi
        kd_loss = amp_loss + phs_loss / torch.pi

        return tea_loss, stu_loss, soft_label_loss, kd_loss


