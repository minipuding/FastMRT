import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from typing import List


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class DownSample(nn.Module):
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
    def __init__(self, in_channels: int):
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
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)

        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 drop_prob: float,
                 attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            Swish(),
            nn.Dropout(drop_prob),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
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
        h = self.block1(x)
        h = self.block2(h)
        h = h + self.shortcut(x)
        h = self.attn(h)
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
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        self.head = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        self.downblocks = nn.ModuleList()
        chs = [base_channels]  # record output channel when dowmsample for upsample
        now_ch = base_channels
        for i, mult in enumerate(ch_mult): 
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

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, drop_prob, attn=True),
            ResBlock(now_ch, now_ch, drop_prob, attn=False),
        ])

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

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, out_channels, 3, stride=1, padding=1)
        )
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
