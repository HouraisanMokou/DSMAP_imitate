import os.path

from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
from PIL import Image


class ImageSet(VisionDataset):
    def __init__(self, data, opt, root,train=True):
        self.train=train
        self.data = data
        self.opt = opt
        super(ImageSet, self).__init__('')
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize(opt.size),
            transforms.CenterCrop(opt.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((opt.size,opt.size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.data[idx])
        img = Image.open(path)
        if self.train==True:
            img = self.transform(img)
            return img
        else:
            w,h=img.size
            img = self.test_transform(img)
            return img,w,h


def GanLoader(dataloader):
    while True:
        for d in dataloader:
            yield d