from torch import nn
from torch.nn import functional as F
from ..util.norm import *
from ..Blocks.Conv import Conv

class ResBlock(nn.Module):
    def __init__(self,dim,norm_mod=None,active=nn.ReLU):
        super(ResBlock, self).__init__()
        self.model=nn.Sequential(*[
            Conv(dim,dim,3,1,1,norm_mod=norm_mod,active=active),
            Conv(dim, dim, 3, 1, 1, norm_mod=norm_mod)
        ])

    def forward(self,x):
        res=x
        x=self.model(x)
        return res+x