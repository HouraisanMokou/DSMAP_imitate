import logging
import os.path
import time

import numpy as np
from torch.utils.data import DataLoader

from utils import opt, util
from utils.data import ImageSet, GanLoader
from models.Runner.Runner import Runner
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
    max_iter = int(np.ceil(opts.show_samples / opts.batch_size))
    for iter in range(max_iter):
        xa, was,has = next(loader1)
        xb, wbs,hbs = next(loader2)
        xa,xb=xa.to(opts.device),xb.to(opts.device)
        xba, xab = runner.sample(xa, xb)
        results.append([
            xa.cpu().detach().numpy(), xb.cpu().detach().numpy(),
            xba.cpu().detach().numpy(), xab.cpu().detach().numpy(),
            was.cpu().detach().numpy(), has.cpu().detach().numpy(),
            wbs.cpu().detach().numpy(),hbs.cpu().detach().numpy()
        ])
    logger.info('start to save')
    cnt = 0
    for batch in results:
        for idx in range(batch[0].shape[0]):
            cnt += 1
            batch2 = []
            for p in range(4):
                x = batch[p]
                tmp = x * 127 + 127
                tmp[tmp < 0] = 0
                tmp[tmp > 255] = 255
                tmp = tmp.transpose(0, 2, 3, 1)
                batch2.append(tmp)

            xa = batch2[0][idx, :, :, :].astype('uint8')
            xb = batch2[1][idx, :, :, :].astype('uint8')
            xba = batch2[2][idx, :, :, :].astype('uint8')
            xab = batch2[3][idx, :, :, :].astype('uint8')
            wa = batch[4][idx]
            ha = batch[5][idx]
            wb = batch[6][idx]
            hb = batch[7][idx]

            ia = Image.fromarray(xa).resize((wa,ha))
            ib = Image.fromarray(xb).resize((wb,hb))
            iba = Image.fromarray(xba).resize((wa,ha))
            iab = Image.fromarray(xab).resize((wb,hb))

            ia.save(os.path.join(opts.result_pics_path, f'pair{cnt}_a.jpg'))
            ib.save(os.path.join(opts.result_pics_path, f'pair{cnt}_b.jpg'))
            iba.save(os.path.join(opts.result_pics_path, f'pair{cnt}_ba.jpg'))
            iab.save(os.path.join(opts.result_pics_path, f'pair{cnt}_ab.jpg'))
