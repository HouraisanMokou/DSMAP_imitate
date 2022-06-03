from torch import nn
from torch.nn import functional as F
from ..util.norm import *
from ..util.active import *
from ..Blocks.Conv import Conv
from ..Blocks.MultiBlocks import ResBlocks,MLP
from ..Blocks.DMBlock import DMBlock
from .encoder import StyleEncoder, ContentEncoder_share
from .decoder import Decoder


class Generator(nn.Module):
    def __init__(self, content_encoder: ContentEncoder_share, idx, opt):
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
                               active=self.active,modulation_list=opt.modulation_list)
        self.mlp = MLP(self.style_dim,self._cnt_mlp_fea1()+self._cnt_mlp_fea2(), self.mlp_dim, 3)

    def _assign_adain(self,style):
        cnt=0
        for m in self.decoder.modules():
            if m.__class__.__name__=='AdaIN':
                # print(f'{cnt};{cnt+m.dim};{style[:,cnt:cnt+m.dim].contiguous().shape}')
                m.weight=style[:,cnt:cnt+m.dim].contiguous().view(-1)
                m.bias=style[:,cnt+m.dim:cnt+2*m.dim].contiguous().view(-1)
                cnt+=2*m.dim
            if m.__class__.__name__ == 'AdaModulationConv':
                # print(f'{cnt};{cnt+m.dim};{style[:,cnt:cnt+m.dim].contiguous().shape}')
                style_code = style[:, cnt:cnt + m.in_channel].contiguous()
                m.style = style_code
                cnt += m.in_channel


    def _cnt_mlp_fea1(self):
        cnt=0
        for m in self.decoder.modules():
            if m.__class__.__name__=='AdaIN':
                cnt+=2*m.dim
        return cnt



    def _cnt_mlp_fea2(self):
        cnt=0
        for m in self.decoder.modules():
            if m.__class__.__name__=='AdaModulationConv':
                cnt+=m.in_channel
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

class Generator_modulation(nn.Module):
    def __init__(self, content_encoder: ContentEncoder_share, idx, opt):
        super(Generator_modulation, self).__init__()
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
                               active=self.active,norm_mod='modulation')
        self.mlp = MLP(self.style_dim,self._cnt_mlp_fea(), self.mlp_dim, 3)

    def _assign_adain(self,style):
        cnt=0
        for m in self.decoder.modules():
            if m.__class__.__name__=='AdaModulationConv':
                # print(f'{cnt};{cnt+m.dim};{style[:,cnt:cnt+m.dim].contiguous().shape}')
                style_code=style[:,cnt:cnt+m.in_channel].contiguous()
                m.style=style_code
                cnt+=m.in_channel


    def _cnt_mlp_fea(self):
        cnt=0
        for m in self.decoder.modules():
            if m.__class__.__name__=='AdaModulationConv':
                cnt+=m.in_channel
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





