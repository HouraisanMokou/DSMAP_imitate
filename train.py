import logging
import os.path
import time

from torch.utils.data import DataLoader

from utils import opt, util
from utils.data import ImageSet, GanLoader
from models.Runner.Runner import Runner
from collections import defaultdict

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

    num_workers=1
    root=opts.dataset_path[0]
    data=os.listdir(root)
    loader1 = GanLoader(
        DataLoader(ImageSet(data,opts,root),
        opts.batch_size, shuffle=True, num_workers=num_workers))
    root=opts.dataset_path[1]
    data=os.listdir(root)
    loader2 = GanLoader(
        DataLoader(ImageSet(data,opts,root),
        opts.batch_size, shuffle=True, num_workers=num_workers))

    trainer=Runner(opts)

    if opts.start_iter >-1:
        result,suc=util.load(opts,trainer,opts.start_iter)
        logger.info(suc)
    else:
        result={
            'gen':defaultdict(list),'dis':defaultdict(list)
        }

    for iter in range(opts.start_iter+1,opts.start_iter+1+opts.train_iter):
        xa=next(loader1).to(opts.device)
        xb=next(loader2).to(opts.device)

        dis_dict=trainer.dis_step(xa,xb)
        gen_dict=trainer.gen_step(xa,xb)
        for k in dis_dict.keys():
            result['dis'][k].append(dis_dict[k])
        for k in gen_dict.keys():
            result['gen'][k].append(gen_dict[k])

        if iter%opts.save_period==0:
            logger.info(util.save(opts,trainer,result,iter))

        # logger.info('finish a iter')