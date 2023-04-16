"""
Copyright (c) Sijie Xu with email: sijie.x@foxmail.com.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmrt.models.runet import Unet
from fastmrt.models.cunet import ComplexUnet
from fastmrt.models.resunet import UNet as ResUnet
from fastmrt.models.casnet import CasNet
from fastmrt.models.swtnet import SwinIR
from fastmrt.models.kdnet import KDNet
from fastmrt.modules.base_module import FastmrtModule
from fastmrt.utils.trans import real_tensor_to_complex_tensor as rt2ct
from typing import List, Dict, Sequence


class UNetModule(FastmrtModule):
    """

    """
    def __init__(
            self,
            *args,
            in_channels: int,
            out_channels: int,
            base_channels: int = 32,
            level_num: int = 4,
            drop_prob: float = 0.0,
            leakyrelu_slope: float = 0.1,
            **kwargs,
    ):
        super(UNetModule, self).__init__(*args, **kwargs)
        self.model = Unet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            level_num=level_num,
            drop_prob=drop_prob,
            leakyrelu_slope=leakyrelu_slope,
        )

class CUNetModule(FastmrtModule):
    def __init__(
            self,
            *args,
            in_channels: int,
            out_channels: int,
            base_channels: int = 32,
            level_num: int = 4,
            drop_prob: float = 0.0,
            **kwargs,
    ):
        super(CUNetModule, self).__init__(*args, **kwargs)
        self.model = ComplexUnet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            level_num=level_num,
            drop_prob=drop_prob,
        )

class ResUNetModule(FastmrtModule):
    def __init__(
            self,
            *args,
            in_channels: int,
            out_channels: int,
            base_channels: int = 32,
            ch_mult: List[int] = [1, 2, 2, 2],
            attn: List[int] = [3],
            num_res_blocks: int = 2,
            drop_prob: float = 0.1,
            **kwargs,
    ):
        super(ResUNetModule, self).__init__(*args, **kwargs)
        self.model = ResUnet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            ch_mult=ch_mult,
            attn=attn,
            num_res_blocks=num_res_blocks,
            drop_prob=drop_prob,
        )

class CasNetModule(FastmrtModule):
    def __init__(
        self,
        *args,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        res_block_num: int = 5,
        res_conv_ksize: int = 3,
        res_conv_num: int = 5,
        drop_prob: float = 0.0,
        leakyrelu_slope = 0.1,
        **kwargs,
    ):
        super(CasNetModule, self).__init__(*args, **kwargs)
        self.model = CasNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            res_block_num=res_block_num,
            res_conv_ksize=res_conv_ksize,
            res_conv_num=res_conv_num,
            drop_prob=drop_prob,
            leakyrelu_slope=leakyrelu_slope,
        )

    def training_step(self, batch, batch_idx):

        return super().training_step(batch, batch_idx, mean=batch.mean, std=batch.std, mask=batch.mask)
    
    def validation_step(self, batch, batch_idx):

        return super().validation_step(batch, batch_idx, mean=batch.mean, std=batch.std, mask=batch.mask)

class SwtNetModule(FastmrtModule):
    def __init__(
            self,
            *args,
            upscale=1,
            in_channels=2,
            img_size=[96, 96],
            patch_size=1,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2.0,
            upsampler='',
            resi_connection='1conv',
            **kwargs,
    ):
        super(SwtNetModule, self).__init__(*args, **kwargs)
        self.model = SwinIR(
            upscale=upscale,
            in_chans=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            window_size=window_size,
            img_range=img_range,
            depths=depths,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            upsampler=upsampler,
            resi_connection=resi_connection
        )

class KDNetModule(FastmrtModule):
    """
    About manual optimizer:
    https://pytorch-lightning.readthedocs.io/en/stable/model/manual_optimization.html#manual-optimization
    """
    def __init__(
            self,
            *args,
            tea_net: nn.Module,
            stu_net: nn.Module,
            **kwargs,
    ):
        super(KDNetModule, self).__init__(*args, **kwargs)
        self.tea_net = tea_net
        self.stu_net = stu_net
        self.model = KDNet(tea_net, stu_net)

    def training_step(self, batch, batch_idx):
        tea_loss, stu_loss, soft_label_loss, kd_loss = self.loss_fn(batch)

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

    def _l1_loss(self, batch):

        # setting loss weight decay
        gamma = 0.7
        beta = max((1 - self.current_epoch / self.max_epoch) * gamma, 0)

        # model forward (only forward stu_input)
        tea_output, stu_output = self.model(batch.input_stu)

        # calculate losses
        tea_loss = F.l1_loss(tea_output, batch.label)
        stu_loss = F.l1_loss(stu_output, batch.label)
        soft_label_loss = F.l1_loss(stu_output, tea_output.clone().detach())
        kd_loss = (1 - beta) *stu_loss + beta * soft_label_loss

        return tea_loss, stu_loss, soft_label_loss, kd_loss

    def _decoupled_loss(self, batch):
        # setting loss weight decay
        gamma1, gamma2 = 0.7, 0.7
        beta1 = max((1 - self.current_epoch / self.max_epoch) * gamma1, 0)
        beta2 = max((1 - self.current_epoch / self.max_epoch) * gamma2, 0)

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