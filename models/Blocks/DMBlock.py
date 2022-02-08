from torch import nn
from torch.nn import functional as F
from ..util.norm import *
from ..Blocks.Conv import Conv


class DMBlock(nn.Module):
    def __init__(self, dim, active=nn.ReLU):
        super(DMBlock, self).__init__()
        self.layer1 = Conv(dim, dim, 3, 2, 1, 1, norm_mod='ln', active=active,transposed=True)
        self.layer2 = Conv(dim, dim, 3, 2, 1, 1, norm_mod='ln')

    def forward(self, x, res):
        x = self.layer1(x)
        return self.layer2(x+res)
