from torch import nn
from torch.nn import functional as F
from ..util.norm import *
from ..Blocks.Conv import Conv
from ..Blocks.MultiBlocks import ResBlocks


class Decoder(nn.Module):
    def __init__(self, n_downsample, mid_downsample, n_res, dim, norm_mod='adain', active=nn.ReLU):
        super(Decoder, self).__init__()
        layers = [ResBlocks(n_res, dim, norm_mod=norm_mod, active=active)]

        for _ in range(mid_downsample):
            layers += [nn.Upsample(scale_factor=2),
                       Conv(dim, dim, 5, 1, 2, 'ln', active=active)]
        for _ in range(n_downsample):
            layers += [nn.Upsample(scale_factor=2),
                       Conv(dim, dim // 2, 5, 1, 2, 'ln', active=active)]
            dim //= 2

        layers += [Conv(dim, 3, 7, 1, 3)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
