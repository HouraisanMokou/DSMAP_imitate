from torch import nn
from torch.nn import functional as F
from .ResBlock import ResBlock

class ResBlocks(nn.Module):
    def __init__(self,num_block,dim,norm_mod=None,active=nn.ReLU):
        super(ResBlocks, self).__init__()
        layers=[]
        for _ in range(num_block):
            layers += [ResBlock(dim,norm_mod=norm_mod,active=active)]
        self.model=nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self,in_fea,out_fea,dim,num_block,active=nn.ReLU):
        super(MLP, self).__init__()
        self.active=active
        layers=[nn.Linear(in_fea,dim)]
        layers=self.add_active(layers)
        for _ in range(num_block-2):
            layers+=[nn.Linear(dim,dim)]
            layers=self.add_active(layers)
        layers+=[nn.Linear(dim,out_fea)]
        layers=self.add_active(layers)
        self.model=nn.Sequential(*layers)


    def add_active(self,layers):
        if self.active is not None:
            layers+=[self.active()]
        return layers

    def forward(self,x):
        return self.model(x.view(x.size(0), -1))