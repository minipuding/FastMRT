import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary


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

        self.down_convs = nn.ModuleList([ConvBlock(in_channels=self.in_channels,
                                                   out_channels=self.base_channels,
                                                   drop_prob=self.drop_prob,
                                                   leakyrelu_slope=self.leakyrelu_slope)])
        temp_channels = self.base_channels
        for _ in range(self.level_num):
            self.down_convs.append(ConvBlock(in_channels=temp_channels,
                                             out_channels=temp_channels * 2,
                                             drop_prob=self.drop_prob,
                                             leakyrelu_slope=self.leakyrelu_slope))
            temp_channels *= 2

        self.up_convs = nn.ModuleList()
        self.up_transpose_convs = nn.ModuleList()
        for _ in range(self.level_num):
            self.up_transpose_convs.append(TransposeConvBlock(in_channels=temp_channels,
                                                              out_channels=temp_channels // 2,
                                                              drop_prob=self.drop_prob,
                                                              leakyrelu_slope=self.leakyrelu_slope))
            self.up_convs.append(ConvBlock(in_channels=temp_channels,
                                           out_channels=temp_channels // 2,
                                           drop_prob=self.drop_prob,
                                           leakyrelu_slope=self.leakyrelu_slope))
            temp_channels //= 2

        self.up_convs[-1] = ConvBlock(in_channels=temp_channels * 2,
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
            output = F.max_pool2d(output, 2)
        output = self.down_convs[-1](output)

        for up_conv_layer, up_transpose_conv_layer in zip(self.up_convs, self.up_transpose_convs):
            output = up_transpose_conv_layer(output)
            down_conv_feature = stack.pop()
            output = torch.cat([output, down_conv_feature], dim=1)
            output = up_conv_layer(output)
        return output


class ConvBlock(nn.Module):

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
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(negative_slope=leakyrelu_slope, inplace=False),
            nn.Dropout2d(self.drop_prob),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(self.out_channels),
        )
        if with_act is True:
            self.layers.append(nn.LeakyReLU(negative_slope=leakyrelu_slope, inplace=False))
        self.layers.append(nn.Dropout2d(self.drop_prob))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class TransposeConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, drop_prob: float, leakyrelu_slope: float):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.drop_prob = drop_prob
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(negative_slope=leakyrelu_slope, inplace=False),
            nn.Dropout2d(self.drop_prob)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)

if __name__ == "__main__":
    net = Unet(2, 2, 24)
    input_data = torch.randn(8, 2, 256, 256)
    summary(net, input_data, device='cpu', depth=5)