import torch
import torch.nn as nn
from torchsummary import summary
import fastmrt.complexnn.complexLayers as clys
import fastmrt.complexnn.complexFunctions as cfuns


class ComplexUnet(nn.Module):
    """
    The structure is the same as fastmrt.models.runet, but all of the blocks are 
    instantiated by complexnn layers and functions.
    Note: here we use CReLU, which means applying relu to both the real and imaginary parts of the complex image,
        since this activation function is better than other activation functions according to
        https://doi.org/10.1002/mrm.28733.

    Args:
        in_channels: int, input channel, 1 for complex type image.
        out_channels: int, output channels as `in_channels`.
        base_channels: int, the first conv block channels and subsequent conv block would double based
            on the `base_channels` with gradual downsampling, default is 16. Note that the base_channels is
            half of r-unet to keep the same FLOPs.
        level_num: int, the number of levels of Unet, default is 4.
        drop_prob: float, dropout probability applied in each conv and trans conv block.
    """

    def __init__(
        self,
        in_channels: int=1,
        out_channels: int=1,
        base_channels: int=16,
        level_num: int = 4,
        drop_prob: float = 0.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.level_num = level_num
        self.drop_prob = drop_prob

        self.down_convs = nn.ModuleList([ComplexConvBlock(in_channels=self.in_channels,
                                                          out_channels=self.base_channels,
                                                          drop_prob=self.drop_prob)])
        temp_channels = self.base_channels
        for _ in range(self.level_num):
            self.down_convs.append(ComplexConvBlock(in_channels=temp_channels,
                                                    out_channels=temp_channels * 2,
                                                    drop_prob=self.drop_prob))
            temp_channels *= 2

        self.up_convs = nn.ModuleList()
        self.up_transpose_convs = nn.ModuleList()
        for _ in range(self.level_num):
            self.up_transpose_convs.append(ComplexTransposeConvBlock(in_channels=temp_channels,
                                                                     out_channels=temp_channels // 2,
                                                                     drop_prob=self.drop_prob))
            self.up_convs.append(ComplexConvBlock(in_channels=temp_channels,
                                                  out_channels=temp_channels // 2,
                                                  drop_prob=self.drop_prob))
            temp_channels //= 2

        self.up_convs[-1] = ComplexConvBlock(in_channels=temp_channels * 2,
                                             out_channels=self.out_channels,
                                             drop_prob=self.drop_prob,
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
    
    """
    The structure is the same as fastmrt.models.runet.ConvBlock, but all of the blocks are 
    instantiated by complexnn layers and functions.

    Args:
        in_channels: int, input channel
        out_channels: int, output channels.
        drop_prob: float, dropout probability applied in each conv and trans conv block.
        with_act: bool, use activate function on last layer or not.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            drop_prob: float,
            with_act: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            clys.ComplexConv2d(self.in_channels, self.out_channels, kernel_size=3, padding='same'),
            clys.ComplexBatchNorm2d(self.out_channels),
            clys.ComplexReLU(),  # CReLU
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
    """
    The structure is the same as fastmrt.models.runet.TransposeConvBlock, but all of the blocks are 
    instantiated by complexnn layers and functions.

    Args:
        in_channels: int, input channel
        out_channels: int, output channels.
        drop_prob: float, dropout probability applied in each conv and trans conv block.
    """

    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            drop_prob: float
        ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.drop_prob = drop_prob
        self.layers = nn.Sequential(
            clys.ComplexConvTranspose2d(self.in_channels, self.out_channels, kernel_size=2, stride=2),
            clys.ComplexBatchNorm2d(self.out_channels),
            clys.ComplexReLU(),  # CReLU
            clys.ComplexDropout2d(self.drop_prob),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)
