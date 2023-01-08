import torch
from torch import nn
from fastmrt.models.runet import Unet
from fastmrt.utils.trans import real_tensor_to_complex_tensor, complex_tensor_to_real_tensor
from fastmrt.utils.normalize import normalize_apply, normalize_paras, denormalize
from typing import Tuple

class RFTNet(nn.Module):
    def __init__(
        self,
        in_channels: Tuple[int] = (2, 2, 2),
        out_channels: Tuple[int] = (2, 2, 1),
        base_channels: Tuple[int] = (32, 32, 32),
        level_num: Tuple[int] = (4, 4, 4),
        drop_prob: Tuple[float] = (0.0, 0.0, 0.0),
        leakyrelu_slope: Tuple[float] = (0.4, 0.4, 0.4),
        last_layer_with_act: bool = False,
    ):
        super(RFTNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.level_num = level_num
        self.drop_prob = drop_prob
        self.leakyrelu_slope = leakyrelu_slope
        self.last_layer_with_act = last_layer_with_act
        self.refnet = Unet(
            in_channels=self.in_channels[0],
            out_channels=self.out_channels[0],
            base_channels=self.base_channels[0],
            level_num=self.level_num[0],
            drop_prob=self.drop_prob[0],
            leakyrelu_slope=self.leakyrelu_slope[0],
            last_layer_with_act=self.last_layer_with_act,
        )
        self.recnet = Unet(
            in_channels=self.in_channels[1],
            out_channels=self.out_channels[1],
            base_channels=self.base_channels[1],
            level_num=self.level_num[1],
            drop_prob=self.drop_prob[1],
            leakyrelu_slope=self.leakyrelu_slope[1],
            last_layer_with_act=self.last_layer_with_act,
        )
        self.tnet = Unet(
            in_channels=self.in_channels[2],
            out_channels=self.out_channels[2],
            base_channels=self.base_channels[2],
            level_num=self.level_num[2],
            drop_prob=self.drop_prob[2],
            leakyrelu_slope=self.leakyrelu_slope[2],
            last_layer_with_act=self.last_layer_with_act,
        )

    def forward(self, input, stage: str = "train"):
        mean, std = normalize_paras(input)
        output = normalize_apply(input, mean, std)
        output_ref = denormalize(self.refnet(output), mean, std)
        output_rec = denormalize(self.recnet(output), mean, std)
        output = real_tensor_to_complex_tensor(output_rec) * \
                 torch.conj(real_tensor_to_complex_tensor(output_ref))
        output_phs = complex_tensor_to_real_tensor(output)# / (2*torch.pi)
        # mean, std = normalize_paras(output)
        # output = normalize_apply(output, mean, std)
        # output_phs = denormalize(self.tnet(output), mean, std)
        if stage == "train":
            return output_phs, output_ref, output_rec
        elif stage == "val" or "test":
            return output_phs
        else:
            raise ValueError("``stage`` must be one of ``train``, ``val`` or ``test``.")