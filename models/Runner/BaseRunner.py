from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from ..Generator.generator import Generator
from ..Generator.decoder import Decoder
from ..Generator.encoder import StyleEncoder, ContentEncoder_share
from ..Discriminator.discriminator import Discriminator
from utils.util import vgg_preprocess,contextual_loss

from functools import partial


def init_fun(m, f):
    classname = m.__class__.__name__
    if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
        f(m.weight.data)

class BaseRunner(nn.Module):
    init_dict = {
        'kaiming': partial(init_fun,f=nn.init.kaiming_normal_),
        'gaussian': partial(init_fun,f=partial(nn.init.normal_, mean=0, std=0.02))
    }

    def __init__(self, opt):
        super(BaseRunner, self).__init__()
        self.device=opt.device
        self.ckpt_prefix=opt.checkpoints_prefix
        self.num_domain = len(opt.classes)
        self.content_encoder = ContentEncoder_share(opt, 'in', num_domain=self.num_domain).to(self.device)
        self.style_dim = opt.style_dim
        self.gen_list = nn.ModuleList()
        self.dis_list = nn.ModuleList()
        for idx in range(self.num_domain):
            gen = Generator(self.content_encoder, idx, opt)
            dis = Discriminator(opt)
            self.gen_list.append(gen.to(self.device))
            self.dis_list.append(dis.to(self.device))
        gen_paras, dis_paras = list(), list()
        for gen in self.gen_list:
            gen_paras += list(gen.parameters())
        for dis in self.dis_list:
            dis_paras += list(dis.parameters())
        gen_paras,dis_paras=list(set(gen_paras)),list(set(dis_paras))
        self.gen_optimizer = Adam(gen_paras, lr=opt.lr, weight_decay=opt.l2)
        self.dis_optimizer = Adam(dis_paras, lr=opt.lr/10, weight_decay=opt.l2)
        self.gen_scheduler = StepLR(self.gen_optimizer,opt.step,opt.gamma)
        self.dis_scheduler = StepLR(self.dis_optimizer,opt.step,opt.gamma)

        self.apply(self.init_dict['gaussian'])
        for gen in self.gen_list:
            gen.apply(self.init_dict['gaussian'])
        for dis in self.dis_list:
            dis.apply(self.init_dict['gaussian'])

        self.lambda_g=opt.lambda_g
        self.lambda_recon_x=opt.lambda_recon_x
        self.lambda_recon_s=opt.lambda_recon_s
        self.lambda_recon_c=opt.lambda_recon_c
        self.lambda_recon_d=opt.lambda_recon_d
        self.lambda_recon_cyc=opt.lambda_recon_cyc
        self.lambda_vgg=opt.lambda_vgg

        if self.lambda_vgg>0:
            from torchvision.models import vgg16
            self.vgg=vgg16(pretrained=True).to(self.device)
            self.vgg.eval()
            for p in self.vgg.parameters():
                p.requires_grad=False

    def vgg_perceptual_loss(self,img,tar):
        img=vgg_preprocess(img)
        tar=vgg_preprocess(tar)
        return contextual_loss(self.vgg.features(img),self.vgg.features(tar))

