import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from typing import List


class Swish(nn.Module):
    """
    定义swish激活函数，可参考https://blog.csdn.net/bblingbbling/article/details/107105648
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class DownSample(nn.Module):
    """
    通过stride=2的卷积层进行降采样
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.down_sample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.down_sample.weight)
        init.zeros_(self.down_sample.bias)

    def forward(self, x):
        x = self.down_sample(x)
        return x


class UpSample(nn.Module):
    """
    通过conv+最近邻插值进行上采样
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.up_sample = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.up_sample.weight)
        init.zeros_(self.up_sample.bias)

    def forward(self, x):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.up_sample(x)
        return x


class AttnBlock(nn.Module):
    """
    自注意力模块，其中线性层均用kernel为1的卷积层表示
    """
    def __init__(self, in_channels: int):
        # ``self.proj_q``, ``self.proj_k``, ``self.proj_v``分别用于学习query, key, value
        # ``self.proj``作为自注意力后的线性投射层
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_channels)
        self.proj_q = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        # 输入经过组归一化以及全连接层后分别得到query, key, value
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        # 用矩阵乘法计算query与key的相似性权重w
        # 其中的``torch.bmm``的效果是第1维不动，第2，3维的矩阵做矩阵乘法，
        # 如a.shape=(_n, _h, _m), b.shape=(_n, _m, _w) --> torch.bmm(a, b).shape=(_n, _h, _w)
        # 矩阵运算后得到的权重要除以根号C, 归一化(相当于去除通道数对权重w绝对值的影响)
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        # 再用刚得到的权重w对value进行注意力加权，操作也是一次矩阵乘法运算
        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)

        # 最后经过线性投射层输出，返回值加上输入x构成跳跃连接(残差连接)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    """
    残差网络模块
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 drop_prob: float,
                 attn=False):
        """
        Args:
            in_channels: int, 输入通道数
            out_ch: int, 输出通道数
            dropout: float, dropout的比例
            attn: bool, 是否使用自注意力模块
        """
        super().__init__()
        # 模块1: gn -> swish -> conv
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
        )
        # 模块2: gn -> swish -> dropout -> conv
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            Swish(),
            nn.Dropout(drop_prob),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )
        # 如果输入输出通道数不一样，则添加一个过渡层``shortcut``, 卷积核为1, 否则什么也不做
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        # 如果需要加attention, 则添加一个``AttnBlock``, 否则什么也不做
        if attn:
            self.attn = AttnBlock(out_channels)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x):
        h = self.block1(x)                           # 输入特征经过模块1编码
        h = self.block2(h)                           # 将混合后的特征输入到模块2进一步编码
        h = h + self.shortcut(x)                     # 残差连接
        h = self.attn(h)                             # 经过自注意力模块(如果attn=True的话)
        return h


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 base_channels: int=32,
                 ch_mult: List[int]=[1,2,2,2],
                 attn: List[int]=[3],
                 num_res_blocks: int=2,
                 drop_prob: float=0.1):
        super().__init__()
        # assert确保需要加attention的位置小于总降采样次数
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        # 实例化头部卷积层
        self.head = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        # 实例化U-Net的编码器部分，即降采样部分，每一层次由``num_res_blocks``个残差块组成
        # 其中chs用于记录降采样过程中的各阶段通道数，now_ch表示当前阶段的通道数
        self.downblocks = nn.ModuleList()
        chs = [base_channels]  # record output channel when dowmsample for upsample
        now_ch = base_channels
        for i, mult in enumerate(ch_mult):  # i表示列表ch_mult的索引, mult表示ch_mult[i]
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_channels=now_ch, out_channels=out_ch,
                    drop_prob=drop_prob, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        # 实例化U-Net编码器和解码器的过渡层，由两个残差块组成
        # 这里我不明白为什么第一个残差块加attention, 第二个不加……问就是``工程科学``
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, drop_prob, attn=True),
            ResBlock(now_ch, now_ch, drop_prob, attn=False),
        ])

        # 实例化U-Net的解码器部分, 与编码器几乎对称
        # 唯一不同的是，每一层次的残差块比编码器多一个，
        # 原因是第一个残差块要用来融合当前特征图与跳转连接过来的特征图，第二、三个才是和编码器对称用来抽特征
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_channels=chs.pop() + now_ch, out_channels=out_ch,
                    drop_prob=drop_prob, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        # 尾部模块: gn -> swish -> conv, 目的是回到原图通道数
        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, out_channels, 3, stride=1, padding=1)
        )
        # 注意这里只初始化头部和尾部模块，因为其他模块在实例化的时候已经初始化过了
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x):
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h)
        h = self.tail(h)

        assert len(hs) == 0
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
    print(y.shape)
