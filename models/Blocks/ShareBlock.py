from torch import nn
from torch.nn import functional as F
from ..util.norm import *
from ..Blocks.Conv import Conv

class ShareBlock(nn.Module):
    def __init__(self,dim,norm_mod=None,active=nn.ReLU):
        super(ShareBlock, self).__init__()
        self.model=nn.Sequential(*[
            Conv(dim,dim,3,2,1,norm_mod=norm_mod,active=active,transposed=True),
            Conv(dim, dim, 3, 2, 1, norm_mod=norm_mod)
        ])

    def forward(self,x):
        res=x
        x=self.model(x)
        return res+x