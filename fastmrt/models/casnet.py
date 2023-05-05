import torch
from torchsummary import summary
from torch import nn
import torch.nn.functional as F
from fastmrt.utils.fftc import fft2c_tensor, ifft2c_tensor
from fastmrt.utils.trans import real_tensor_to_complex_tensor, complex_tensor_to_real_tensor
from fastmrt.utils.normalize import normalize_apply, denormalize
from fastmrt.models.runet import Unet

class CasNet(nn.Module):
    """
    Casnet is a classical MR reconstruction network as follows:
        CNN - DC - CNN - DC ... CNN - DC
    DC stands for data consistency. We use res-block CNNs here.
    
    Args:
        in_channels: int, the input channel, 2 for real type images.
        out_channels: int, the output channels are the same as `in_channels`.
        base_channels: int, the res-conv block channels, default is 32.
        res_block_num: int, the number of res-blocks, default is 5.
        res_conv_ksize: int, the conv kernel size in res-blocks, default is 3.
        res_conv_num: int, the number of res-conv layers in a res-block, default is 5.
        drop_prob: float, dropout probability applied in each conv and trans conv block.
        leakyrelu_slope: float, the slope of the leaky ReLU.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 32,
        res_block_num: int = 5,
        res_conv_ksize: int = 3,
        res_conv_num: int = 5,
        drop_prob: float = 0.0,
        leakyrelu_slope: float = 0.1,
    ):
        super(CasNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.res_block_num = res_block_num
        self.res_conv_ksize = res_conv_ksize
        self.res_conv_num = res_conv_num
        self.drop_prob = drop_prob
        self.leakyrelu_slope = leakyrelu_slope

        self.res_blocks = nn.ModuleList(
            [ResBlock(in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        base_channels=self.base_channels,
                        res_conv_ksize=self.res_conv_ksize,
                        res_conv_num=self.res_conv_num,
                        drop_prob=self.drop_prob,
                        leakyrelu_slope=self.leakyrelu_slope) for _ in range(self.res_block_num)]
        )
        self.dc_blocks = nn.ModuleList(
            [DCBlock() for _ in range(self.res_block_num)]
        )

    def forward(self, input, mean: float=0., std: float=1., mask: torch.Tensor=torch.ones(1, 96)):
        outputs = []
        # mask = mask.to(input.device)
        temp = input
        # identity = denormalize(input, mean, std)
        for res_block, dc_block in zip(self.res_blocks, self.dc_blocks):
            output = res_block(temp)
            # output = denormalize((output + temp) / 2, mean, std)
            temp = dc_block((output + temp) / 2, input, mask)
            # temp = normalize_apply(output, mean, std)
            outputs += [temp]
        return outputs[-1]


class ResBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        res_conv_ksize: int = 3,
        res_conv_num: int = 5,
        drop_prob: float = 0.0,
        leakyrelu_slope: float = 0.1,
    ):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.res_conv_ksize = res_conv_ksize
        self.res_conv_num = res_conv_num
        self.drop_prob = drop_prob
        self.leakyrelu_slope = leakyrelu_slope

        self.in_conv = ConvBlock(self.in_channels, self.base_channels, self.drop_prob, self.leakyrelu_slope)
        self.out_conv = ConvBlock(self.base_channels, self.out_channels, self.drop_prob, self.leakyrelu_slope, with_act=False)
        self.res_block = nn.ModuleList(
            self.res_conv_num * [ResConv(self.base_channels, self.res_conv_ksize, self.drop_prob, self.leakyrelu_slope)]
        )

    def forward(self, input):
        output = self.in_conv(input)
        for res_conv in self.res_block:
            output = res_conv(output)
        return self.out_conv(output)


class ResConv(nn.Module):
    """
    Res-Block with two convolution layers.

    Args:
        channels: int, input and output channels.
        kernel_size: int, convolution kernel size, default is 3.
        drop_prob: float, dropout probability applied in each conv and trans conv block.
        leakyrelu_slope: float, the slope of the leaky ReLU.
    """

    def __init__(
            self,
            channels: int = 32,
            kernel_size: int = 3,
            drop_prob: float = 0.0,
            leakyrelu_slope: float = 0.1,
    ):
        super(ResConv, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.drop_prob = drop_prob
        self.leakyrelu_slope = leakyrelu_slope

        self.res_block = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=self.kernel_size, padding='same'),
            nn.BatchNorm2d(num_features=self.channels),
            nn.LeakyReLU(negative_slope=self.leakyrelu_slope, inplace=False),
            nn.Dropout2d(self.drop_prob),
            nn.Conv2d(self.channels, self.channels, kernel_size=self.kernel_size, padding='same'),
            nn.BatchNorm2d(num_features=self.channels),
            nn.LeakyReLU(negative_slope=self.leakyrelu_slope, inplace=False),
            nn.Dropout2d(self.drop_prob),
        )

    def forward(self, input: torch.Tensor):
        output = self.res_block(input)
        return output + input


class ConvBlock(nn.Module):
    """
    This block contains one convolutional layer with batch normalization and ReLU activation.

    Args:
        in_channels: int, the number of input channels. For real images, this is 2.
        out_channels: int, the number of output channels, which is the same as `in_channels`.
        drop_prob: float, the probability of dropout applied in each convolutional and transposed convolutional block.
        leakyrelu_slope: float, the slope of the leaky ReLU activation function.
        with_act: bool, whether to use activation layers.
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
        self.leakyrelu_slope = leakyrelu_slope
        self.with_act = with_act

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.act = nn.LeakyReLU(negative_slope=self.leakyrelu_slope, inplace=False)
        self.drop = nn.Dropout2d(self.drop_prob)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv(input)
        if self.with_act is True:
            output = self.bn(output)
            output = self.act(output)
        output = self.drop(output)
        return output


class DCBlock(nn.Module):
    """
    Data consistency block.

    Args:
        output: the output of the CNN.
        origin_input: the original input image.
        mask: the down-sampling mask.
    """

    def __init__(
            self,
    ):
        super(DCBlock, self).__init__()

    def forward(self, output, origin_input, mask):
        output_kspace = fft2c_tensor(real_tensor_to_complex_tensor(output))
        origin_input_kspace = fft2c_tensor(real_tensor_to_complex_tensor(origin_input))
        dc_kspace = origin_input_kspace * mask + output_kspace * (1 - mask)
        return complex_tensor_to_real_tensor(ifft2c_tensor(dc_kspace))

    def _pad(self, kspace, size):
        height, width = kspace.shape
        resized_kspace = torch.complex(torch.zeros(size=size), torch.zeros(size=size))
        resized_kspace[(size[0] - height) // 2: size[0] - (size[0] - height) // 2,
                       (size[1] - width) // 2: size[1] - (size[1] - width) // 2] = kspace
        return resized_kspace

    def _unpad(self, kspace, size):
        height, width = kspace.shape
        return kspace[(height - size[0]) // 2: height - (height - size[0]) // 2,
                      (width - size[1]) // 2: width - (width - size[1]) // 2]



if __name__ == "__main__":
    net = CasNet(2, 2, 24)
    input_data = torch.randn(8, 2, 256, 256)
    summary(net, input_data, device='cpu', depth=5)