import albumentations as A
from typing import Dict, Union, List
from fastmrt.utils.trans import complex_np_to_real_np as cn2rn
from fastmrt.utils.trans import real_np_to_complex_np as rn2cn
import numpy as np
import random


class ComplexAugs:

    def __init__(self,
                 height: int,
                 width: int,
                 union: bool = False,
                 objs=None,
                 augs_list=None,
                 compose_num=None,
                 crop_scale: tuple[float, float] = (0.7, 1.0),
                 gamma_limit: Union[float, tuple[float, float]] = (99, 101),
                 noise_var_limit: Union[tuple[float, float], float] = (0.5, 1.0),
                 blur_limit: Union[int, tuple[int, int]] = (1, 5),):
        if objs is None:
            objs = ["amp", "phs"]
        else:
            assert isinstance(objs, List)
        if augs_list is None:
            augs_list = ["crop", "rotate", "blur", "flip"]
        else:
            assert isinstance(objs, List)
        if compose_num is None:
            compose_num = len(augs_list)
        else:
            assert compose_num <= len(augs_list)
        self.union = union
        self.objs = objs
        self.augs_list = augs_list
        self.compose_num = compose_num

        self.crop = A.RandomResizedCrop(height=height, width=width, scale=crop_scale)
        self.flip = A.Flip()
        self.rotate = A.RandomRotate90()
        self.gamma = A.RandomGamma(gamma_limit=gamma_limit)
        self.noise = A.GaussNoise(var_limit=noise_var_limit)
        self.blur = A.GaussianBlur(blur_limit=blur_limit)

    def __call__(self, sample, tmap_mask):
        if self.union is True:
            sample_mask = np.concatenate((sample, tmap_mask[np.newaxis, :]), axis=0).transpose([1, 2, 0])
            sample_mask = self.apply_augs(sample_mask).transpose([-1, 0, 1])
            sample = rn2cn(sample_mask[:2])
            tmap_mask = sample_mask[-1]
        else:
            amp = np.abs(sample)
            phs = cn2rn(sample / amp)
            phs_mask = np.concatenate((phs, tmap_mask[np.newaxis, :]), axis=0).transpose([1, 2, 0])
            if "amp" in self.objs:
                amp = self.apply_augs(amp)
            if "phs" in self.objs:
                phs_mask = self.apply_augs(phs_mask).transpose([-1, 0, 1])
                phs = rn2cn(phs_mask[:2])
                tmap_mask = phs_mask[-1]
            sample = amp * phs
        return sample, tmap_mask

    def apply_augs(self, x):
        augs = random.sample(self.augs_list, self.compose_num)
        if "crop" in augs:
            x = self.crop(image=x)["image"]
        if "flip" in augs:
            x = self.flip(image=x)["image"]
        if "rotate" in augs:
            x = self.rotate(image=x)["image"]
        if "noise" in augs:
            x = self.noise(image=x)["image"]
        if "blur" in augs:
            x = self.blur(image=x)["image"]
        if "gamma" in augs:
            x = self.gamma(image=x)["image"]
        return x