import os.path
import time

import numpy as np
import torch
from torch.autograd import Variable
from ..Generator.generator import Generator,Generator_modulation

from .BaseRunner import *
from utils.util import l1_loss
from PIL import Image
from os import mkdir
from torch.autograd import grad
from torchvision.utils import make_grid,save_image

class Runner(BaseRunner):
    def __init__(self, opt):
        super(Runner, self).__init__(opt)
        self.gen_a = self.gen_list[0]
        self.gen_b = self.gen_list[1]

        self.dis_a = self.dis_list[0]
        self.dis_b = self.dis_list[1]
        self.iter=0

    def dis_step(self, x_a, x_b):
        t1 = time.time()
        self.dis_optimizer.zero_grad()
        s_a_r = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).to(self.device))
        s_b_r = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).to(self.device))

        pre_a, c_a, _, dm_ab, s_a = self.gen_a.encode(x_a)
        pre_b, c_b, _, dm_ba, s_b = self.gen_b.encode(x_b)

        x_ba = self.gen_a.decode(dm_ba, s_a)
        x_ab = self.gen_b.decode(dm_ab, s_b)
        x_ba_r = self.gen_a.decode(dm_ba, s_a_r)
        x_ab_r = self.gen_b.decode(dm_ab, s_b_r)
        loss_a_random = self.dis_a.dis_loss(x_a, x_ba_r.detach())
        loss_b_random = self.dis_b.dis_loss(x_b, x_ab_r.detach())
        loss_a = self.dis_a.dis_loss(x_a, x_ba.detach())
        loss_b = self.dis_b.dis_loss(x_b, x_ab.detach())

        total_loss = self.lambda_g * (loss_a + loss_b + loss_a_random + loss_b_random)
        if not torch.isnan(total_loss):
            total_loss.backward()
        self.dis_optimizer.step()
        self.dis_scheduler.step()
        info_dict = {
            'loss_a_random': loss_a_random.cpu().detach().numpy(),
            'loss_b_random': loss_b_random.cpu().detach().numpy(),
            'loss_a': loss_a.cpu().detach().numpy(), 'loss_b': loss_b.cpu().detach().numpy(),
            'total_loss': total_loss.cpu().detach().numpy(), 'used_time': time.time() - t1
        }
        return info_dict

    def gen_step(self, x_a, x_b):
        t1 = time.time()
        self.gen_optimizer.zero_grad()
        s_a_r = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).to(self.device))
        s_b_r = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).to(self.device))

        pre_a, c_a, c_a_share, dm_ab, s_a = self.gen_a.encode(x_a)
        pre_b, c_b, c_b_share, dm_ba, s_b = self.gen_b.encode(x_b)

        dm_aa = self.gen_b.domain_mapping(c_a_share, pre_a)
        dm_bb = self.gen_a.domain_mapping(c_b_share, pre_b)

        x_ba = self.gen_a.decode(dm_ba, s_a)
        x_ab = self.gen_b.decode(dm_ab, s_b)
        x_aa = self.gen_a.decode(dm_aa, s_a)
        x_bb = self.gen_b.decode(dm_bb, s_b)

        _, c_b_recon, _, dm_bb_recon, s_a_recon = self.gen_a.encode(x_ba)
        _, c_a_recon, _, dm_aa_recon, s_b_recon = self.gen_b.encode(x_ab)

        x_aba = self.gen_a.decode(dm_aa_recon, s_a_recon)
        x_bab = self.gen_b.decode(dm_bb_recon, s_b_recon)

        x_ba_r = self.gen_a.decode(dm_ba, s_a_r)
        x_ab_r = self.gen_b.decode(dm_ab, s_b_r)

        _, c_b_recon_r, _, _, s_a_recon_r = self.gen_a.encode(x_ba_r)
        _, c_a_recon_r, _, _, s_b_recon_r = self.gen_b.encode(x_ab_r)

        loss_recon_d_a, loss_recon_d_b = l1_loss(c_a_share, dm_aa), l1_loss(c_b_share, dm_bb)
        loss_recon_x_a, loss_recon_x_b = l1_loss(x_a, x_aa), l1_loss(x_b, x_bb)
        loss_recon_c_a, loss_recon_c_b = l1_loss(c_a_recon, c_a), l1_loss(c_b_recon, c_b)
        loss_cc_x_a, loss_cc_x_b = l1_loss(x_a, x_aba), l1_loss(x_b, x_bab)
        loss_adv_a, loss_adv_b = self.dis_a.gen_loss(x_a, x_ba), self.dis_b.gen_loss(x_b, x_ab)
        loss_perceptual_a, loss_perceptual_b = self.vgg_perceptual_loss(x_ba, x_a), self.vgg_perceptual_loss(x_ab, x_b)
        loss_recon_s_a_random, loss_recon_s_b_random = l1_loss(s_a_recon_r, s_a_r), l1_loss(s_b_recon_r, s_b_r)
        loss_recon_c_a_random, loss_recon_c_b_random = l1_loss(c_a_recon_r, c_a), l1_loss(c_b_recon_r, c_b)
        loss_adv_a_random, loss_adv_b_random = self.dis_a.gen_loss(x_a, x_ba_r), self.dis_a.gen_loss(x_b, x_ab_r)
        loss_perceptual_a_random, loss_perceptual_b_random = \
            self.vgg_perceptual_loss(x_ba_r, x_a), self.vgg_perceptual_loss(x_ab_r, x_b)

        # print(f"loss_recon_x_a {loss_recon_x_a} {loss_recon_x_b}")
        # print(f"loss_recon_c_a {loss_recon_c_a} {loss_recon_c_b}")
        # print(f"loss_recon_d_a {loss_recon_d_a} {loss_recon_d_b}")
        # print(f"loss_cc_x_a {loss_cc_x_a} {loss_cc_x_b}")
        # print(f"loss_adv_a {loss_adv_a} {loss_adv_b}")
        # print(f"loss_perceptual {loss_perceptual_a} {loss_perceptual_b}")
        # print(f"loss_recon_s_a_r {loss_recon_s_a_random} {loss_recon_s_b_random}")
        # print(f"loss_recon_c_a_r {loss_recon_c_a_random} {loss_recon_c_b_random}")
        # print(f"loss_adv_a_r {loss_adv_a_random} {loss_adv_b_random}")
        # print(f"loss_perceptual_a_r {loss_perceptual_a_random} {loss_perceptual_b_random}")

        total_loss = \
            self.lambda_g * (loss_adv_a + loss_adv_b + loss_adv_a_random + loss_adv_b_random) + \
            self.lambda_recon_c * (loss_recon_c_a + loss_recon_c_b + loss_recon_c_a_random + loss_recon_c_b_random) + \
            self.lambda_recon_x * (loss_recon_x_a + loss_recon_x_b) + \
            self.lambda_recon_d * (loss_recon_d_a + loss_recon_d_b) + \
            self.lambda_recon_cyc * (loss_cc_x_a + loss_cc_x_b) + \
            self.lambda_recon_s * (loss_recon_s_a_random + loss_recon_s_b_random) + \
            self.lambda_vgg * (
                    loss_perceptual_a + loss_perceptual_b + loss_perceptual_a_random + loss_perceptual_b_random)
        print(f'total {total_loss}')
        if not torch.isnan(total_loss):
            total_loss.backward()
        self.gen_optimizer.step()
        self.gen_scheduler.step()
        info_dict = {
            'loss_recon_d_a': loss_recon_d_a.cpu().detach().numpy(),
            'loss_recon_d_b': loss_recon_d_b.cpu().detach().numpy(),
            'loss_recon_x_a': loss_recon_x_a.cpu().detach().numpy(),
            'loss_recon_x_b': loss_recon_x_b.cpu().detach().numpy(),
            'loss_recon_c_a': loss_recon_c_a.cpu().detach().numpy(),
            'loss_recon_c_b': loss_recon_c_b.cpu().detach().numpy(),
            'loss_cc_x_a': loss_cc_x_a.cpu().detach().numpy(),
            'loss_cc_x_b': loss_cc_x_b.cpu().detach().numpy(),
            'loss_adv_a': loss_adv_a.cpu().detach().numpy(),
            'loss_adv_b': loss_adv_b.cpu().detach().numpy(),
            'loss_perceptual_a': loss_perceptual_a.cpu().detach().numpy(),
            'loss_perceptual_b': loss_perceptual_b.cpu().detach().numpy(),
            'loss_recon_s_a_random': loss_recon_s_a_random.cpu().detach().numpy(),
            'loss_recon_s_b_random': loss_recon_s_b_random.cpu().detach().numpy(),
            'loss_recon_c_a_random': loss_recon_c_a_random.cpu().detach().numpy(),
            'loss_recon_c_b_random': loss_recon_c_b_random.cpu().detach().numpy(),
            'loss_adv_a_random': loss_adv_a_random.cpu().detach().numpy(),
            'loss_adv_b_random': loss_adv_b_random.cpu().detach().numpy(),
            'loss_perceptual_a_random': loss_perceptual_a_random.cpu().detach().numpy(),
            'loss_perceptual_b_random': loss_perceptual_b_random.cpu().detach().numpy(),
            'total_loss': total_loss.cpu().detach().numpy(), 'used_time': time.time() - t1
        }
        return info_dict

    def sample(self, x_a, x_b):
        with torch.no_grad():
            pre_a, c_a, c_a_share, dm_ab, s_a = self.gen_a.encode(x_a)
            pre_b, c_b, c_b_share, dm_ba, s_b = self.gen_b.encode(x_b)

            x_ba = self.gen_a.decode(dm_ba, s_a)
            x_ab = self.gen_b.decode(dm_ab, s_b)
            return x_ba, x_ab

    def save(self, iter):
        torch.save({
            'gen_a': self.gen_a.state_dict(),
            'gen_b': self.gen_b.state_dict(),
            'dis_a': self.dis_a.state_dict(),
            'dis_b': self.dis_b.state_dict(),
            'gen_optimizer': self.gen_optimizer.state_dict(),
            'dis_optimizer': self.dis_optimizer.state_dict(),
            'gen_scheduler': self.gen_scheduler.state_dict(),
            'dis_scheduler': self.dis_scheduler.state_dict(),
        }, os.path.join(self.ckpt_prefix, f'iter_{iter}.pt'))
        return 'save successfully at iter ' + str(iter)

    def load(self, iter):
        ckpt = torch.load(os.path.join(self.ckpt_prefix, f'iter_{iter}.pt'))
        self.gen_a.load_state_dict(ckpt['gen_a'])
        self.gen_b.load_state_dict(ckpt['gen_b'])
        self.gen_optimizer.load_state_dict(ckpt['gen_optimizer'])
        self.gen_scheduler.load_state_dict(ckpt['gen_scheduler'])
        self.dis_a.load_state_dict(ckpt['dis_a'])
        self.dis_b.load_state_dict(ckpt['dis_b'])
        self.dis_optimizer.load_state_dict(ckpt['dis_optimizer'])
        self.dis_scheduler.load_state_dict(ckpt['dis_scheduler'])
        return 'load successfully from iter ' + str(iter)

    def gen(self, xa, xb,iter,opt):
        with torch.no_grad():
            xba, xab = self.sample(xa, xb)
            length = xa.size()[0]
            all_images = torch.cat([xa, xb, xab, xba], dim=0)
            result = make_grid(all_images, nrow=length)
            if not os.path.exists(f'{opt.checkpoints_prefix}/image'):
                mkdir(f'{opt.checkpoints_prefix}/image')
            save_image(result, f'{opt.checkpoints_prefix}/image/visual_{iter}.jpg')
