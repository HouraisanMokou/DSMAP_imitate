import logging
import os.path
import time

from torch.utils.data import DataLoader

from utils import opt, util
from utils.data import ImageSet, GanLoader
from models.Runner.Runner import Runner
from collections import defaultdict

from models.Discriminator.multi_discriminator import  MultiDiscriminator
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

    md=MultiDiscriminator(opts)
    print(md)