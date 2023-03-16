import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
import fastmrt.complexnn.complexLayers as clys
import fastmrt.complexnn.complexFunctions as cfuns


class Unet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        level_num: int = 4,
        drop_prob: float = 0.0,
        leakyrelu_slope: float = 0.4,
        last_layer_with_act: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.level_num = level_num
        self.drop_prob = drop_prob
        self.leakyrelu_slope = leakyrelu_slope
        self.last_layer_with_act = last_layer_with_act

        self.down_convs = nn.ModuleList([ComplexConvBlock(in_channels=self.in_channels,
                                                          out_channels=self.base_channels,
                                                          drop_prob=self.drop_prob,
                                                          leakyrelu_slope=self.leakyrelu_slope)])
        temp_channels = self.base_channels
        for _ in range(self.level_num):
            self.down_convs.append(ComplexConvBlock(in_channels=temp_channels,
                                                    out_channels=temp_channels * 2,
                                                    drop_prob=self.drop_prob,
                                                    leakyrelu_slope=self.leakyrelu_slope))
            temp_channels *= 2

        self.up_convs = nn.ModuleList()
        self.up_transpose_convs = nn.ModuleList()
        for _ in range(self.level_num):
            self.up_transpose_convs.append(ComplexTransposeConvBlock(in_channels=temp_channels,
                                                                     out_channels=temp_channels // 2,
                                                                     drop_prob=self.drop_prob,
                                                                     leakyrelu_slope=self.leakyrelu_slope))
            self.up_convs.append(ComplexConvBlock(in_channels=temp_channels,
                                                  out_channels=temp_channels // 2,
                                                  drop_prob=self.drop_prob,
                                                  leakyrelu_slope=self.leakyrelu_slope))
            temp_channels //= 2

        self.up_convs[-1] = ComplexConvBlock(in_channels=temp_channels * 2,
                                             out_channels=self.out_channels,
                                             drop_prob=self.drop_prob,
                                             leakyrelu_slope=self.leakyrelu_slope,
                                             with_act=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input
        stack = []
        for layer in self.down_convs[:-1]:
            output = layer(output)
            stack.append(output)
            output = cfuns.complex_max_pool2d(output, 2)
        output = self.down_convs[-1](output)

        for up_conv_layer, up_transpose_conv_layer in zip(self.up_convs, self.up_transpose_convs):
            output = up_transpose_conv_layer(output)
            down_conv_feature = stack.pop()
            output = torch.cat([output, down_conv_feature], dim=1)
            output = up_conv_layer(output)
        return output


class ComplexConvBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            drop_prob: float,
            leakyrelu_slope: float,
            with_act: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            clys.ComplexConv2d(self.in_channels, self.out_channels, kernel_size=3, padding='same'),
            clys.ComplexBatchNorm2d(self.out_channels),
            clys.ComplexReLU(),
            clys.ComplexDropout2d(self.drop_prob),
            clys.ComplexConv2d(self.out_channels, self.out_channels, kernel_size=3, padding='same'),
            clys.ComplexBatchNorm2d(self.out_channels),
        )
        if with_act is True:
            self.layers.append(clys.ComplexReLU())
        self.layers.append(clys.ComplexDropout2d(self.drop_prob))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class ComplexTransposeConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, drop_prob: float, leakyrelu_slope: float):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.drop_prob = drop_prob
        self.layers = nn.Sequential(
            clys.ComplexConvTranspose2d(self.in_channels, self.out_channels, kernel_size=2, stride=2),
            clys.ComplexBatchNorm2d(self.out_channels),
            clys.ComplexReLU(),
            clys.ComplexDropout2d(self.drop_prob),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)

if __name__ == "__main__":
    net = Unet(2, 2, 24)
    input_data = torch.randn(8, 2, 256, 256)
    summary(net, input_data, device='cpu', depth=5)