import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmrt.models.runet import Unet
from fastmrt.models.kdnet import KDNet
from fastmrt.modules.base_module import BaseModule
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
            lr_tea: float = 5e-4,
            lr_stu: float = 5e-4,
            weight_decay_tea: float = 0.0,
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
        self.soft_label_weight = soft_label_weight
        self.lr_tea = lr_tea
        self.lr_stu = lr_stu
        self.weight_decay_tea = weight_decay_tea
        self.weight_decay_stu = weight_decay_stu
        self.model = KDNet(tea_net, stu_net, use_ema, soft_label_weight)
        self.automatic_optimization = False
        # self.model = ResUnet(inout_ch=2, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1)

    def training_step(self, batch, batch_idx):
        # model forward
        tea_output, stu_output = self.model(batch.input_stu, batch.input_tea)

        # calculate losses
        tea_loss = F.l1_loss(tea_output, batch.label)
        stu_loss = F.l1_loss(stu_output, batch.label)
        soft_label_loss = F.l1_loss(stu_output, tea_output.clone().detach())
        kd_loss = stu_loss + self.soft_label_weight * soft_label_loss

        # obtain optimizers & schedulers
        optimizer_tea, optimizer_stu = self.optimizers()
        scheduler_tea, scheduler_stu = self.lr_schedulers()

        # optimize teacher network
        optimizer_tea.zero_grad()
        self.manual_backward(tea_loss)
        optimizer_tea.step()

        # optimize student network
        optimizer_stu.zero_grad()
        self.manual_backward(kd_loss)
        optimizer_stu.step()

        # update lr scheduler
        scheduler_tea.step()
        scheduler_stu.step()

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
        optimizer_tea = torch.optim.Adam(self.model.tea_net.parameters(),
                                         lr=self.lr_tea,
                                         weight_decay=self.weight_decay_tea)
        optimizer_stu = torch.optim.Adam(self.model.stu_net.parameters(),
                                         lr=self.lr_stu,
                                         weight_decay=self.weight_decay_stu)

        scheduler_tea = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_tea,
                                                                   T_max=200,
                                                                   last_epoch=-1)
        scheduler_stu = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_stu,
                                                                   T_max=200,
                                                                   last_epoch=-1)
        return [optimizer_tea, optimizer_stu], [scheduler_tea, scheduler_stu]




