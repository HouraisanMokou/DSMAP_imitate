from torch import nn
from torch.nn import functional as F
from ..util.norm import *
from ..util.active import *
from ..Blocks.Conv import Conv
from ..Blocks.MultiBlocks import ResBlocks,MLP
from ..Blocks.DMBlock import DMBlock
from .encoder import StyleEncoder, ContentEncoder
from .decoder import Decoder


class Generator(nn.Module):
    def __init__(self, content_encoder: ContentEncoder, idx, opt):
        super(Generator, self).__init__()
        self.loss_mod = opt.loss_mod
        self.idx = idx
        self.dim = opt.dim
        self.mlp_dim = opt.mlp_dim
        self.style_dim = opt.style_dim
        self.n_downsample = opt.n_downsample
        self.mid_downsample = opt.mid_downsample
        self.n_res = opt.num_layer
        self.active = active_dict[opt.gen_active]

        self.content_encoder = content_encoder
        self.style_encoder = StyleEncoder(self.dim, self.style_dim, active=self.active)
        self.domain_mapping = DMBlock(self.content_encoder.out_dim)

        self.decoder = Decoder(self.n_downsample, self.mid_downsample, self.n_res, self.content_encoder.out_dim,
                               active=self.active)
        self.mlp = MLP(self.style_dim,self._cnt_mlp_fea(), self.mlp_dim, 3)

    def _assign_adain(self,style):
        cnt=0
        for m in self.decoder.modules():
            if m.__class__.__name__=='AdaIN':
                m.weight=style[:,cnt:cnt+m.dim].contiguous().view(-1)
                m.bias=style[:,cnt+m.dim:cnt+2*m.dim].contiguous().view(-1)
                cnt+=2*m.dim


    def _cnt_mlp_fea(self):
        cnt=0
        for m in self.decoder.modules():
            if m.__class__.__name__=='AdaIN':
                cnt+=2*m.dim
        return cnt

    def encode(self,img):
        style_code=self.style_encoder(img)
        pre_code, domain_code, share_content=self.content_encoder(img,self.idx)
        dm_out=self.domain_mapping(share_content,pre_code)
        return pre_code,domain_code,share_content,dm_out,style_code

    def decode(self,content,style):
        adain=self.mlp(style)
        self._assign_adain(adain)
        img=self.decoder(content)
        return img






