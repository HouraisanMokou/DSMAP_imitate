import logging
import os.path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import opt, util
from utils.data import ImageSet, GanLoader
from models.Runner.Runner import Runner
from torchvision.utils import make_grid,save_image
from collections import defaultdict

from PIL import Image

if __name__ == '__main__':
    opts = opt.baseOpt()
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    file_handler = logging.FileHandler(opts.logging_file_name)
    file_handler.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    logger.info(
        "{}: start to logging\n".format(
            time.strftime("%Y.%m.%d_%H:%M:%S", time.localtime())
        )
    )

    num_workers = 1
    root = opts.dataset_path[0]
    data = os.listdir(root)
    loader1 = GanLoader(
        DataLoader(ImageSet(data, opts, root, train=False),
                   opts.batch_size, shuffle=True, num_workers=num_workers))
    root = opts.dataset_path[1]
    data = os.listdir(root)
    loader2 = GanLoader(
        DataLoader(ImageSet(data, opts, root, train=False),
                   opts.batch_size, shuffle=True, num_workers=num_workers))

    runner = Runner(opts)

    _, suc = util.load(opts, runner, opts.testing_iter)
    logger.info(suc)
    results = []
    xa, was, has = next(loader1)
    xb, wbs, hbs = next(loader2)
    logger.info('start to save')
    xa, xb = xa.to(opts.device), xb.to(opts.device)
    xba, xab = runner.sample(xa, xb)
    length=xa.size()[0]
    all_images=torch.cat([xa,xb,xab,xba],dim=0)
    result=make_grid(all_images,nrow=length)
    save_image(result,'result.jpg')
