from torch import nn
from torch.nn import functional as F
from ..Blocks.Conv import Conv
from ..util.active import *
from utils.util import hinge_loss, dis_ls_loss,gen_ls_loss


class Discriminator(nn.Module):
    ## discriminator for multi-scale
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.active = active_dict[opt.dis_active]
        self.num_scale = opt.num_scales
        self.num_layer = opt.num_layer
        self.models = nn.ModuleList()
        self.dim = opt.dim

        for _ in range(self.num_scale):
            self._make_layer()

        self.loss_mod = opt.loss_mod

    def _make_layer(self):
        dim = self.dim
        layer = [Conv(3, dim, 4, 2, 1, active=self.active)]
        for _ in range(self.num_layer - 1):
            layer += [Conv(dim, dim * 2, 4, 2, 1, active=self.active)]
            dim = dim * 2
        layer += [Conv(dim, 1, 1, 1, 0, active=nn.Tanh)]
        self.models.append(nn.Sequential(*layer))

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
            x = self.avg_pool(x)
        return outputs

    def dis_loss(self, real, fake):
        out0 = self(fake)
        out1 = self(real)
        losses = 0

        for iter, (fake, real) in enumerate(zip(out0, out1)):
            if self.loss_mod == 'ls':
                losses += dis_ls_loss(real, fake)
            elif self.loss_mod == 'hinge':
                losses += hinge_loss(real, fake)
        return losses

    def gen_loss(self, real, fake):
        out0 = self(fake)
        out1 = self(real)
        losses = 0

        for iter, (fake, real) in enumerate(zip(out0, out1)):
            if self.loss_mod == 'ls':
                losses += gen_ls_loss(real, fake)
            elif self.loss_mod == 'hinge':
                losses += hinge_loss(real, fake)
        return losses