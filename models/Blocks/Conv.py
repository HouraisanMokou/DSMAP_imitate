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
        func = partial(nn.ConvTranspose2d, output_padding=out_padding) if transposed else nn.Conv2d
        self.conv = func(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.active = None if active is None else active()
        self.norm = None if norm_mod is None else norm_dict[norm_mod](out_channel)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.active is not None:
            x = self.active(x)
        return x
