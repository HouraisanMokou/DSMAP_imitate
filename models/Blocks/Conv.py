import torch
from torch import nn
from torch.nn import functional as F
from ..util.norm import *

from functools import partial


class Conv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel=4, stride=2, padding=1,
                 out_padding=0,
                 norm_mod=None,
                 active=None,
                 transposed=False):
        super(Conv, self).__init__()
        self.norm_mod = norm_mod
        if self.norm_mod != "modulation":
            func = partial(nn.ConvTranspose2d, output_padding=out_padding) if transposed else nn.Conv2d
            self.conv = func(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
            self.norm = None if norm_mod is None else norm_dict[norm_mod](out_channel)
        else:
            self.conv = AdaModulationConv(in_channel, out_channel, kernel, stride, padding)
            self.norm = None
        self.active = None if active is None else active()

    def forward(self, x):
        x = self.conv(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.active is not None:
            x = self.active(x)

        return x


class AdaModulationConv(nn.Module):
    def __init__(self, inc, outc, kernel_size, stride, padding=1, active=None):
        super(AdaModulationConv, self).__init__()
        self.in_channel = inc
        self.weight = nn.Parameter(torch.randn((outc, inc, kernel_size, kernel_size)))
        self.stride = stride
        self.padding = padding
        self.active = None if active is None else active()
        self.style = None

    def forward(self, x):
        style = self.style
        batch_size = x.size()[0]
        o, i, h, w = self.weight.size()
        style = style/ style.norm(float('inf'), dim=1, keepdim=True)
        style = style.reshape((-1, self.in_channel, 1, 1, 1))
        w = self.weight * (1 / float(i * h * w)**0.5 / self.weight.norm(float('inf'), dim=[1, 2, 3], keepdim=True))
        batch_out = []
        for _ in range(batch_size):
            w_ = w * (style[_, ::])
            coe = (w_.square().sum(dim=[1, 2, 3]) + 1e-8).rsqrt().reshape(i, 1, 1, 1)
            w_ = w_* coe
            batch_out.append(F.conv2d(x[_, ::].unsqueeze(0), w_, stride=self.stride, padding=self.padding))
        x = torch.cat(batch_out, dim=0)
        return x
