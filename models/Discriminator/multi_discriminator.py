from torch import nn
from torch.nn import functional as F
from ..Blocks.Conv import Conv
from ..util.active import *
from utils.util import hinge_loss, dis_ls_loss


class MultiDiscriminator(nn.Module):
    def __init__(self, opt):
        super(MultiDiscriminator, self).__init__()
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.active = active_dict[opt.dis_active]
        self.num_scale = opt.num_scales
        self.num_layer = opt.num_layer
        self.models_share = nn.ModuleList()
        self.models_heads = nn.ModuleList()
        self.dim = opt.dim
        self.num_domain =len(opt.classes)

        for _ in range(self.num_scale):
            self._make_layer()

        self.loss_mod = opt.loss_mod

    def _make_layer(self):
        dim = self.dim
        layer = [Conv(3, dim, 4, 2, 1, active=self.active)]
        layer += [Conv(dim, dim * 2, 4, 2, 1, active=self.active)]
        dim = dim * 2
        self.models_share.append(nn.Sequential(*layer))
        heads = nn.ModuleList()
        for _ in range(self.num_domain):
            layer = []
            for _ in range(self.num_layer - 2):
                layer += [Conv(dim, dim * 2, 4, 2, 1, active=self.active)]
                dim = dim * 2
            layer += [Conv(dim, 1, 1, 1, 0, active=nn.Tanh)]
            heads.append( nn.Sequential(*layer))

        self.models_heads.append(heads)

    def forward(self, x, cls):
        outputs = []
        for i in range(len(self.models_share)):
            model_share = self.models_share[i]
            model_head = self.models_heads[i][cls]
            share = model_share(x)
            outputs.append(model_head(share))
            x = self.avg_pool(x)
        return outputs

    def loss(self, real, fake, cls):
        out0 = self(fake, cls)
        out1 = self(real, cls)
        losses = 0

        for iter, (fake, real) in enumerate(zip(out0, out1)):
            if self.loss_mod == 'ls':
                losses += dis_ls_loss(real, fake)
            elif self.loss_mod == 'hinge':
                losses += hinge_loss(real, fake)
        return losses
