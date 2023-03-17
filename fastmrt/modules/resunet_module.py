import random

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from fastmrt.models.resunet import UNet as ResUnet
from fastmrt.modules.base_module import BaseModule
from fastmrt.utils.normalize import denormalize
from fastmrt.utils.trans import complex_tensor_to_real_tensor as ct2rt
from fastmrt.utils.trans import real_tensor_to_complex_tensor as rt2ct
from fastmrt.data.prf import PrfFunc
from typing import Sequence, Dict, List


class ResUNetModule(BaseModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            base_channels: int = 32,
            ch_mult: List[int] = [1, 2, 2, 2],
            attn: List[int] = [3],
            num_res_blocks: int = 2,
            drop_prob: float = 0.1,
            lr: float = 1e-3,
            weight_decay: float = 0.0,
            tmap_prf_func: PrfFunc = None,
            tmap_patch_rate: int = 4,
            tmap_max_temp_thresh: int = 45,
            tmap_ablation_thresh: int = 43,
            log_images_frame_idx: int = 5,  # recommend 4 ~ 8
            log_images_freq: int = 50,
    ):
        super(ResUNetModule, self).__init__(tmap_prf_func=tmap_prf_func,
                                            tmap_patch_rate=tmap_patch_rate,
                                            tmap_max_temp_thresh=tmap_max_temp_thresh,
                                            tmap_ablation_thresh=tmap_ablation_thresh,
                                            log_images_frame_idx=log_images_frame_idx,
                                            log_images_freq=log_images_freq)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.ch_mult = ch_mult
        self.attn = attn
        self.num_res_blocks = num_res_blocks
        self.drop_prob = drop_prob
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = ResUnet(in_channels=self.in_channels,
                             out_channels=self.out_channels,
                             base_channels=self.base_channels,
                             ch_mult=self.ch_mult,
                             attn=self.attn,
                             num_res_blocks=self.num_res_blocks,
                             drop_prob=self.drop_prob,
                             )

    def training_step(self, batch, batch_idx):
        amp_loss, phs_loss = self._decoupled_loss_v3(batch)
        train_loss = amp_loss + phs_loss

        return {"amp_loss": amp_loss,
                "phs_loss": phs_loss,
                "loss": train_loss}

    def training_epoch_end(self, train_logs: Sequence[Dict]) -> None:
        amp_loss = torch.tensor(0, dtype=torch.float32, device='cuda')
        phs_loss = torch.tensor(0, dtype=torch.float32, device='cuda')
        train_loss = torch.tensor(0, dtype=torch.float32, device='cuda')
        for log in train_logs:
            amp_loss += log["amp_loss"]
            phs_loss += log["phs_loss"]
            train_loss += log["loss"]
        self.log("amp_loss", amp_loss, on_epoch=True, on_step=False)
        self.log("phs_loss", phs_loss, on_epoch=True, on_step=False)
        self.log("loss", train_loss, on_epoch=True, on_step=False)

    def validation_step(self, batch, batch_idx):
        output = self.model(batch.input)
        output_ref = self.model(batch.input_ref)
        val_loss = F.l1_loss(output, batch.label)
        mean = batch.mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        std = batch.std.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return {
            "input": denormalize(batch.input, mean, std),
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
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=200,
                                                               last_epoch=-1)
        return [optimizer], [scheduler]

    def _normal_training(self, batch):
        output = self.model(batch.input)
        train_loss = F.l1_loss(output, batch.label)
        return train_loss

    def _decoupled_loss(self, batch):
        output = self.model(batch.input)
        output_complex = rt2ct(output)
        label_complex = rt2ct(batch.label)

        # amplitude loss
        amp_loss = F.l1_loss(output_complex.abs(), label_complex.abs())

        # phase loss
        phs_loss = (torch.angle(output_complex * torch.conj(label_complex)) * batch.tmap_mask).abs().sum(0).mean()

        return amp_loss, phs_loss

    def _decoupled_loss_v2(self, batch):
        output = self.model(batch.input)
        output_complex = rt2ct(output)
        label_complex = rt2ct(batch.label)

        # union loss
        union_loss = F.l1_loss(output, batch.input)

        # amplitude loss
        amp_loss = F.l1_loss(output_complex.abs(), label_complex.abs())

        # phase loss
        phs_loss = (torch.angle(output_complex * torch.conj(label_complex)) * batch.tmap_mask).abs().sum(0).mean()

        return amp_loss, phs_loss, union_loss

    def _decoupled_loss_v3(self, batch):
        output = self.model(batch.input)
        output_complex = rt2ct(output)
        label_complex = rt2ct(batch.label)

        # amplitude loss
        amp_loss = F.l1_loss(output_complex.abs(), label_complex.abs())

        # phase loss
        phs_loss = (torch.angle(output_complex * torch.conj(label_complex)) * batch.phs_scale).abs().sum(0).mean()

        return amp_loss, phs_loss / torch.pi
