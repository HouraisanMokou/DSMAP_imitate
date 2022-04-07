import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.Tensor(dim, 1, 1).uniform_())
        self.bias = nn.Parameter(torch.Tensor(dim, 1, 1))

    def forward(self, x):
        size = x.size()[1:]
        out = F.layer_norm(x, size, self.weight.expand(size), self.bias.expand(size))
        return out


class AdaIN(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super(AdaIN, self).__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.zeros(dim))

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        xx = x.contiguous().view(1, b * c, *x.size()[2:])

        # print(f'{x.shape};{running_mean.shape};{running_var.shape};{self.weight.shape};{self.bias.shape};{self.dim}')

        out = F.batch_norm(xx, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps)
        return out.view(*x.size())

norm_dict = {
    'ln': LayerNorm,
    'adain': AdaIN,
    'in': nn.InstanceNorm2d
}