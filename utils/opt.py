import argparse
import os
import random
import time

import numpy as np
import torch


def baseOpt():
    parser = argparse.ArgumentParser(description='arguments of program')
    parser.add_argument('--data_directory', type=str, default='./datasets', help='original data directory')
    parser.add_argument('--checkpoints_directory', type=str, default='./checkpoint', help='checkpoints directory')

    # ilsvrc2012 SIGGRAPH
    parser.add_argument('--dataset_name', type=str, default='1;2', help='the name of sets (split with \';\' ')
    parser.add_argument('--model', type=str, default='DSMAP', help='the name of model')

    parser.add_argument('--stage', type=str, default='train', help='train/test')
    parser.add_argument('--logging_directory',
                        type=str, default='./log', help='the directory the log would be saved to')
    parser.add_argument('--random_state', type=int, default=2022, help='the random seed')

    # arguments for runner
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='cuda:0 / cpu')
    parser.add_argument('--start_iter', type=int, default=-1,
                        help='number of epochs')
    parser.add_argument('--train_iter', type=int, default=100000,
                        help='number of epochs')
    parser.add_argument('--save_period', type=int, default=10,
                        help='save period')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization in optimizer')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size while training/ validating')

    parser.add_argument('--size', type=int, default=192, help='h,w of img')
    # args for model
    parser.add_argument('--num_layer', type=int, default=4, help='# of layer of whole model')
    parser.add_argument('--dim', type=int, default=64, help='dim of encoded feature')
    parser.add_argument('--step', type=int, default=100000, help='step of lr scheduler')
    parser.add_argument('--gamma', type=int, default=0.5, help='decay rate of lr scheduler')
    parser.add_argument('--lambda_g', type=int, default=1, help='weight of gan loss')
    parser.add_argument('--lambda_recon_x', type=int, default=10, help='weight of x reconstruction loss')
    parser.add_argument('--lambda_recon_s', type=int, default=10, help='weight of s reconstruction loss')
    parser.add_argument('--lambda_recon_c', type=int, default=2, help='weight of c reconstruction loss')
    parser.add_argument('--lambda_recon_d', type=int, default=2, help='weight of d reconstruction loss')
    parser.add_argument('--lambda_recon_cyc', type=int, default=5, help='weight of cycle consistence loss')
    parser.add_argument('--lambda_vgg', type=int, default=1, help='weight of vgg perceptual loss')
    parser.add_argument('--loss_mod', type=str, default='ls', choices=['ls', 'hinge'],
                        help='weight of vgg perceptual loss')
    # args for discriminator
    parser.add_argument('--dis_active', type=str, default='lrelu', choices=['lrelu', 'relu'],
                        help='the active function of discriminator')
    parser.add_argument('--num_scales', type=int, default=3, help='# of scales in discriminator')
    # args for generator
    parser.add_argument('--mlp_dim', type=int, default=256, help='dim of mlp layer')
    parser.add_argument('--style_dim', type=int, default=8, help='dim of style feature')
    parser.add_argument('--n_downsample', type=int, default=2, help='# of a layer in decoder/encoder')
    parser.add_argument('--mid_downsample', type=int, default=1, help='# of a layer in decoder/encoder')
    parser.add_argument('--gen_active', type=str, default='relu', choices=['lrelu', 'relu'],
                        help='the active function of generator')

    parser.add_argument('--testing_iter', type=int, default=10,
                        help='iter to test')
    parser.add_argument('--show_samples', type=int, default=1,
                        help='# of testing samples')
    args, unknown = parser.parse_known_args()

    setattr(args, 'logging_file_name', os.path.join(
        args.logging_directory,
        '{}_{}_{}.txt'.format(args.model,
                              args.dataset_name,
                              time.strftime('%Y.%m.%d', time.localtime()))
    ))
    setattr(args, 'result_file_name', os.path.join(
        args.logging_directory,
        '{}_{}_{}.pkl'.format(args.model,
                              args.dataset_name,
                              time.strftime('%Y.%m.%d', time.localtime()))
    ))
    setattr(args, 'result_pics_path', os.path.join(
        args.logging_directory,
        '{}_{}'.format(args.model,
                       args.dataset_name)
    ))
    sets=args.dataset_name.split(';')
    paths=[os.path.join(args.data_directory, s) for s in sets]
    setattr(args, 'dataset_path', paths)
    setattr(args, 'classes', sets)
    prefix = args.checkpoints_directory + '/' + '{}_{}'.format(args.model, args.dataset_name)
    setattr(args, 'checkpoints_prefix',
            prefix)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    if not os.path.exists(args.data_directory):
        os.makedirs(args.data_directory)
    if not os.path.exists(args.checkpoints_directory):
        os.makedirs(args.checkpoints_directory)
    if not os.path.exists(args.logging_directory):
        os.makedirs(args.logging_directory)
    if not os.path.exists(args.result_pics_path):
        os.makedirs(args.result_pics_path)

    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    torch.cuda.manual_seed(args.random_state)
    torch.backends.cudnn.deterministic = True

    return args
