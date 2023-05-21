"""
Copyright (c) Sijie Xu with email: sijie.x@foxmail.com.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmrt.modules.base_module import FastmrtModule
from fastmrt.utils.trans import real_tensor_to_complex_tensor as rt2ct
from typing import Dict, Sequence


class CasNetModule(FastmrtModule):

    def training_step(self, batch, batch_idx):

        return super().training_step(batch, batch_idx, mean=batch.mean, std=batch.std, mask=batch.mask)
    
    def validation_step(self, batch, batch_idx):

        return super().validation_step(batch, batch_idx, mean=batch.mean, std=batch.std, mask=batch.mask)


class KDNetModule(FastmrtModule):

    def training_step(self, batch, batch_idx):
        tea_loss, stu_loss, soft_label_loss, kd_loss = self.loss_fn(*self.model(batch.input), batch.label)

        return {"loss": kd_loss,
                "tea_loss": tea_loss,
                "stu_loss": stu_loss,
                "soft_label_loss": soft_label_loss}

    def training_epoch_end(self, train_logs: Sequence[Dict]) -> None:
        train_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        tea_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        stu_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        soft_label_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        for log in train_logs:
            train_loss += log["loss"]
            tea_loss += log["tea_loss"]
            stu_loss += log["stu_loss"]
            soft_label_loss += log["soft_label_loss"]
        self.log("loss", train_loss / len(train_logs), on_epoch=True, on_step=False)
        self.log("tea_loss", tea_loss / len(train_logs), on_epoch=True, on_step=False)
        self.log("stu_loss", stu_loss / len(train_logs), on_epoch=True, on_step=False)
        self.log("soft_label_loss", soft_label_loss / len(train_logs), on_epoch=True, on_step=False)

    def validation_step(self, batch, batch_idx, **kwargs):

        # obtain output and reference frame output
        output_tea, output = self.model(batch.input, **kwargs)
        _, output_ref = self.model(batch.input_ref, **kwargs)

        # calculate validation loss
        _, _, _, val_loss = self.loss_fn(output_tea, output, batch.label)

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
            "val_loss": val_loss,
            "file_name": batch.metainfo['file_name'],
            "frame_idx": batch.metainfo['frame_idx'],
            "slice_idx": batch.metainfo['slice_idx'],
            "coil_idx": batch.metainfo['coil_idx'],
        }

    def _l1_loss(self, tea_output, stu_output, label):

        # setting loss weight decay
        gamma = 0.4
        beta = max((1 - self.current_epoch / self.max_epochs) * gamma, 0)

        # calculate losses
        tea_loss = F.l1_loss(tea_output, label)
        stu_loss = F.l1_loss(stu_output, label)
        soft_label_loss = F.l1_loss(stu_output, tea_output.clone().detach())
        kd_loss = (1 - beta) *stu_loss + beta * soft_label_loss

        return tea_loss, stu_loss, soft_label_loss, kd_loss

    def _decoupled_loss(self, tea_output, stu_output, label):
        # setting loss weight decay
        alpha = 2
        gamma1, gamma2 = 0.4, 0.4
        beta1 = max((1 - self.current_epoch / self.max_epochs) * gamma1, 0)
        beta2 = max((1 - self.current_epoch / self.max_epochs) * gamma2, 0)

        # turn to complex data format
        if torch.is_complex(label):
            tea_output_complex = tea_output.clone().detach()
            stu_output_complex = stu_output
            label_complex = label
        else:
            tea_output_complex = rt2ct(tea_output.clone().detach())
            stu_output_complex = rt2ct(stu_output)
            label_complex = rt2ct(label)

        # amplitude loss
        stu_amp_loss = F.l1_loss(stu_output_complex.abs(), label_complex.abs())
        tea_amp_loss = F.l1_loss(tea_output_complex.abs(), label_complex.abs())
        soft_amp_loss = F.l1_loss(stu_output_complex.abs(), tea_output_complex.abs())

        # phase loss
        stu_phs_loss = (torch.angle(stu_output_complex * torch.conj(label_complex)) * label_complex.abs()).abs().mean(0).mean()
        tea_phs_loss = (torch.angle(tea_output_complex * torch.conj(label_complex)) * label_complex.abs()).abs().mean(0).mean()
        soft_phs_loss = (torch.angle(stu_output_complex * torch.conj(tea_output_complex)) * tea_output_complex.abs()).abs().mean(0).mean()

        # merging
        amp_loss = (1 - beta1) * stu_amp_loss + beta1 * soft_amp_loss
        phs_loss = (1 - beta2) * stu_phs_loss + beta2 * soft_phs_loss

        tea_loss = tea_amp_loss + tea_phs_loss * alpha
        stu_loss = stu_amp_loss + stu_phs_loss * alpha
        soft_label_loss = soft_amp_loss + soft_phs_loss * alpha
        kd_loss = amp_loss + phs_loss *alpha

        return tea_loss, stu_loss, soft_label_loss, kd_loss