from torch import nn

active_dict = {
    'relu': nn.ReLU,
    'lrelu': nn.LeakyReLU
}