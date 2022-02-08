import os.path

import torch
from torch.autograd import Variable
from torch.nn import functional as F
import pickle


def load(opt, trainer,iter):
    result_filename = os.path.join(opt.checkpoints_prefix, f'result_{iter}.pkl')
    with open(result_filename, 'rb') as f:
        result = pickle.load(f)
    suc=trainer.load(iter)
    return result,suc


def save(opt, trainer, result_dict, iter):
    result_filename = os.path.join(opt.checkpoints_prefix, f'result_{iter}.pkl')
    with open(result_filename, 'wb') as f:
        pickle.dump(result_dict, f)
    return trainer.save(iter)


def hinge_loss(real, fake):
    return F.relu(1 - real).mean() + F.relu(1 + fake).mean()


def ls_loss(real, fake):
    return torch.mean((fake + 1) ** 2) + torch.mean((real - 1) ** 2)


def l1_loss(real, fake):
    return torch.mean(torch.abs(real - fake))


###############################################################################
# Code from
# https://github.com/NVlabs/MUNIT
###############################################################################
def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean))  # subtract mean
    return batch


###############################################################################
# Code from
# https://github.com/S-aiueo32/contextual_loss_pytorch
###############################################################################
def contextual_loss(x, y, h=0.5):
    """Computes contextual loss between x and y.

    Args:
      x: features of shape (N, C, H, W).
      y: features of shape (N, C, H, W).

    Returns:
      cx_loss = contextual loss between x and y (Eq (1) in the paper)
    """
    assert x.size() == y.size()
    N, C, H, W = x.size()  # e.g., 10 x 512 x 14 x 14. In this case, the number of points is 196 (14x14).

    y_mu = y.mean(3).mean(2).mean(0).reshape(1, -1, 1, 1)

    x_centered = x - y_mu
    y_centered = y - y_mu
    x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
    y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1, keepdim=True)

    # The equation at the bottom of page 6 in the paper
    # Vectorized computation of cosine similarity for each pair of x_i and y_j
    x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H*W, H*W)

    d = 1 - cosine_sim  # (N, H*W, H*W)  d[n, i, j] means d_ij for n-th data
    d_min, _ = torch.min(d, dim=2, keepdim=True)  # (N, H*W, 1)

    # Eq (2)
    d_tilde = d / (d_min + 1e-5)

    # Eq(3)
    w = torch.exp((1 - d_tilde) / h)

    # Eq(4)
    cx_ij = w / torch.sum(w, dim=2, keepdim=True)  # (N, H*W, H*W)

    # Eq (1)
    cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)  # (N, )
    cx_loss = torch.mean(-torch.log(cx + 1e-5))

    return cx_loss
