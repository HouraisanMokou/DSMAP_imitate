from torch import nn
from torch.nn import functional as F
from ..util.norm import *
from ..util.active import *
from ..Blocks.Conv import Conv
from ..Blocks.MultiBlocks import ResBlocks
from ..Blocks.ResBlock import ResBlock
from ..Blocks.DMBlock import DMBlock


class StyleEncoder(nn.Module):
    def __init__(self, dim, style_dim, norm_mod=None, active=nn.ReLU):
        super(StyleEncoder, self).__init__()
        layers = [Conv(3, dim, 7, 1, 3, norm_mod=norm_mod, active=active)]
        for _ in range(2):
            layers += [Conv(dim, 2 * dim, 4, 2, 1, norm_mod=norm_mod, active=active)]
            dim *= 2
        for _ in range(2):
            layers += [Conv(dim, dim, 4, 2, 1, norm_mod=norm_mod, active=active)]
        layers += [nn.AdaptiveAvgPool2d(1)]
        layers += [Conv(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class StyleEncoder_share(nn.Module):
    def __init__(self, opt, norm_mod=None, active=nn.ReLU):
        super(StyleEncoder_share, self).__init__()
        self.dim = opt.dim
        self.style_dim = opt.style_dim
        self.norm_mod = norm_mod
        self.active = active
        self.num_domain = len(opt.classes)
        self.pres = nn.ModuleList()  # nn.Sequential(*layers)
        self.encoders = nn.ModuleList()
        cur_dim = -1
        for _ in range(self.num_domain):
            cur_dim=self._make_one_encoder()
        cur_dim = cur_dim if cur_dim != -1 else self.dim
        self.share_layer = ResBlock(cur_dim, norm_mod=norm_mod, active=nn.LeakyReLU)

        # self.share_layer = share_layer
        self.out_dim = cur_dim

    def forward(self, x, idx):
        pre = self.pres[idx](x)
        out = self.encoders[idx](pre)
        share = self.share_layer(out)
        return pre, out, share

    def _make_one_encoder(self):
        dim = self.dim
        layers = [Conv(3, dim, 7, 1, 3, norm_mod=self.norm_mod, active=self.active)]
        for _ in range(2):
            layers += [Conv(dim, 2 * dim, 4, 2, 1, norm_mod=self.norm_mod, active=self.active)]
            dim *= 2
        #self.out_dim = dim
        self.pres.append(nn.Sequential(*layers))
        layers=[]
        for _ in range(2):
            layers += [Conv(dim, dim, 4, 2, 1, norm_mod=self.norm_mod, active=self.active)]
        layers += [nn.AdaptiveAvgPool2d(1)]
        layers += [Conv(dim, self.style_dim, 1, 1, 0)]
        self.encoders.append(nn.Sequential(*layers))


class ContentEncoder_share(nn.Module):
    def __init__(self, opt, norm_mod, num_domain=2):
        super(ContentEncoder_share, self).__init__()
        self.n_downsample = opt.n_downsample
        self.mid_downsample = opt.mid_downsample
        self.n_res = opt.num_layer
        self.dim = opt.dim
        self.norm_mod = norm_mod
        self.active = active_dict[opt.gen_active]
        self.pre_list = nn.ModuleList()
        self.encode_list = nn.ModuleList()
        cur_dim = -1
        for _ in range(num_domain):
            pre, encoder, cur_dim = self._make_one_encoder()
            self.pre_list.append(pre)
            self.encode_list.append(encoder)
        cur_dim = cur_dim if cur_dim != -1 else self.dim

        self.share_layer = ResBlock(cur_dim, norm_mod=norm_mod, active=nn.LeakyReLU)
        self.out_dim = cur_dim

    def forward(self, x, idx):
        pre = self.pre_list[idx](x)
        out = self.encode_list[idx](pre)
        share = self.share_layer(out)
        return pre, out, share

    def _make_one_encoder(self):
        layers = [Conv(3, self.dim, 7, 1, 3),
                  nn.LeakyReLU(inplace=True)]
        cur_dim = self.dim
        for _ in range(self.n_downsample):
            layers += [Conv(cur_dim, 2 * cur_dim, 4, 2, 1, norm_mod=self.norm_mod, active=self.active)]
            cur_dim *= 2
        layers2 = []
        for _ in range(self.mid_downsample):
            layers2 += [Conv(cur_dim, cur_dim, 4, 2, 1, norm_mod=self.norm_mod, active=self.active)]
        layers2 += [ResBlocks(self.n_res, cur_dim, norm_mod=self.norm_mod, active=self.active)]
        return nn.Sequential(*layers), nn.Sequential(*layers2), cur_dim
