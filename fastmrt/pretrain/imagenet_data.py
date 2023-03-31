import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
import torchvision.transforms as tv_trans
import random
import PIL.Image as Image
from glob import glob
import os
import numpy as np
from typing import Optional, Callable, Union, Iterator, Sized, Dict
from pathlib import Path

from fastmrt.utils.fftc import fft2c_numpy, ifft2c_numpy, fft2c_tensor, ifft2c_tensor
from torchvision.transforms import Compose


class ImagenetDataset(Dataset):
    # root = '../ImageNet/ILSVRC/Data/CLS-LOC/'
    def __init__(self,
                 root: Union[str, Path, os.PathLike],
                 transform: Optional[Callable] = None, ):

        self.files = []
        pattern = "*.JPEG"
        for path, _, _ in os.walk(root):
            self.files.extend(glob(os.path.join(path, pattern)))
            # or uncomment in case paths are already saved
            #     self.files = torch.load('./imagenet_filepaths_train.pt')

        self.aug_trans = tv_trans.Compose([
            tv_trans.Resize(320, ),
            tv_trans.RandomCrop(96),
            tv_trans.Grayscale(1),
            tv_trans.RandomVerticalFlip(p=0.5),
            tv_trans.RandomHorizontalFlip(p=0.5),
            tv_trans.ToTensor(),
        ])

        self.transform = transform

    def __len__(self, ):
        return len(self.files)

    def __getitem__(self, idx):
        # Load Image
        fname = self.files[idx]
        image = Image.open(fname).convert("RGB")

        # Data Augmentation
        y = self.aug_trans(image)
        if random.uniform(0, 1) < 0.5:
            y = torch.rot90(y, 1, [-2, -1])

        x = y.squeeze()
        x = x * torch.exp(torch.tensor(0 + 0j))
        kspace = fft2c_tensor(x, fftshift_dim=(-2, -1))

        if self.transform is not None:
            return self.transform({
                "kspace": kspace,
                "fname": fname[-20:],
                "dataslice": 0
            })
        else:
            return {
                "kspace": kspace,
                "fname": fname[-20:],
                "dataslice": 0
            }


