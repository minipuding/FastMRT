import torch
import torch.nn as nn
from torch.nn import functional as F


class Unet(nn.Module):
    """
     This is a naive implementation of Unet with the following structure:
         _______________________________________________________________________________________________
        | input --> conv_block   ······························>   trans_conv --> conv_block --> output |
        |             |                                                   |                             |
        |         max_pool2d --> conv_block  ·····>  trans_conv --> conv_block                          |
        |                             ...               ...                                             |
        |                             max_pool2d --> conv_block                                         |
        |_______________________________________________________________________________________________|
    The conv_block consists of two `conv->bn->leaky_relu`.

    Args:
        in_channels: int, input channel, 2 for real type image.
        out_channels: int, output channels as `in_channels`.
        base_channels: int, the first conv block channels and subsequent conv block would double based
            on the `base_channels` with gradual downsampling, default is 32.
        level_num: int, the number of levels of Unet, default is 4.
        drop_prob: float, dropout probability applied in each conv and trans conv block.
        leakyrelu_slope: float, leaky ReLU slope.
    """


    def __init__(
        self,
        in_channels: int=2,
        out_channels: int=2,
        base_channels: int=32,
        level_num: int=4,
        drop_prob: float=0.0,
        leakyrelu_slope: float=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.level_num = level_num
        self.drop_prob = drop_prob
        self.leakyrelu_slope = leakyrelu_slope

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
    """
    Naive convolutional block with `(conv->bn->leaky_relu->dropout)*2`

    Args:
        in_channels: int, input channel
        out_channels: int, output channels.
        drop_prob: float, dropout probability applied in each conv and trans conv block.
        leakyrelu_slope: float, leaky ReLU slope.
        with_act: bool, use activate function on last layer or not.
    """

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
    """
    Naive transpose convolutional block with `trans_conv->bn->leaky_relu->dropout`.

    Args:
        in_channels: int, input channel
        out_channels: int, output channels.
        drop_prob: float, dropout probability applied in each conv and trans conv block.
        leakyrelu_slope: float, leaky ReLU slope.
    """
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            drop_prob: float, 
            leakyrelu_slope: float
        ):
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
