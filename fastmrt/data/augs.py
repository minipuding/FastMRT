import albumentations as A
from typing import Dict, Union, List
from fastmrt.utils.trans import complex_np_to_real_np as cn2rn
from fastmrt.utils.trans import real_np_to_complex_np as rn2cn
import numpy as np
import random
from fastmrt.data.transforms import ComposeTransform
from torch.utils.data._utils.collate import default_collate
from fastmrt.utils.fftc import fft2c_numpy, ifft2c_numpy


class ComplexAugs:
    """
     Apply normal augmentations (e.g. random crop and resize) on complex image.
     We can use augmentations only on amplitude,  phase or both them. This class only
     support four basic augmentations: random crop and resize, random rotate (0, 90, 180 and 270 degrees),
     gaussian blur and random flip.

     Args:
         height: int, image height
         width: int, image width
         union: bool, if True, apply the same augmentations on both amplitude and phase like general augs,
                default is False
         objs: list[str], objects that augs applied on, only choose from `amp` and `phs`
         ap_logic: str, shuffle
    """

    def __init__(self,
                 height: int,
                 width: int,
                 union: bool = False,
                 objs=None,
                 ap_logic: str="and",
                 augs_list=None,
                 compose_num=None,
                 crop_scale: tuple[float, float] = (0.7, 1.0),
                 gamma_limit: Union[float, tuple[float, float]] = (99, 101),
                 noise_var_limit: Union[tuple[float, float], float] = (0.5, 1.0),
                 blur_limit: Union[int, tuple[int, int]] = (1, 5),):

        # assert objects
        if objs is None:
            objs = ["amp", "phs"]
        else:
            assert isinstance(objs, List)

        # assert augmentations list
        if augs_list is None:
            augs_list = ["crop", "rotate", "blur", "flip"]
        else:
            assert isinstance(augs_list, List)
        if compose_num is None or compose_num <= 0:
            compose_num = len(augs_list)
        elif len(augs_list) > 0:
            assert compose_num <= len(augs_list)
        self.union = union
        self.objs = objs
        self.ap_logic = ap_logic
        self.augs_list = augs_list
        self.compose_num = compose_num

        self.crop = A.RandomResizedCrop(height=height, width=width, scale=crop_scale)
        self.flip = A.Flip()
        self.rotate = A.RandomRotate90()
        self.gamma = A.RandomGamma(gamma_limit=gamma_limit)
        self.noise = A.GaussNoise(var_limit=noise_var_limit)
        self.blur = A.GaussianBlur(blur_limit=blur_limit)

    def __call__(self, sample, tmap_mask):
        if len(self.augs_list) == 0:
            return sample, tmap_mask
        elif self.union is True:
            sample = cn2rn(sample)
            sample_mask = np.concatenate((sample, tmap_mask[np.newaxis, :]), axis=0).transpose([1, 2, 0])
            sample_mask = self.apply_augs(sample_mask).transpose([-1, 0, 1])
            sample = rn2cn(sample_mask[:2])
            tmap_mask = sample_mask[-1]
        elif len(self.objs) != 0:
            if len(self.objs) > 1 and self.ap_logic == "or":
                objs = random.sample(self.objs, 1)
            else:
                objs = self.objs
            amp = np.abs(sample)
            phs = sample / amp
            if "amp" in objs:
                amp = self.apply_augs(amp)
            if "phs" in objs:
                phs = cn2rn(phs)
                phs_mask = np.concatenate((phs, tmap_mask[np.newaxis, :]), axis=0).transpose([1, 2, 0])
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


class AugsCollateFunction:

    def __init__(self,
                 *args,
                 transforms,
                 height: int = 96,
                 width: int = 96,
                 ap_shuffle: bool=False,
                 **kwargs
                 ):

        if isinstance(transforms, ComposeTransform):
            self.simu_focus_trans, self.under_sampling_trans = transforms.items()
        else:
            self.simu_focus_trans, self.under_sampling_trans = None, transforms
        self.augs_fun = ComplexAugs(*args, height=height, width=width, **kwargs)
        self.ap_shuffle = ap_shuffle

    def __call__(self, batch):
        if self.simu_focus_trans is not None:
            batch = [self.simu_focus_trans(sample) for sample in batch]

        batch = self._augs(batch)

        # apply down-sampling transform.
        batch = [self.under_sampling_trans(sample) for sample in batch]

        return default_collate(batch)

    def _augs(self, batch):

        # turn to image domain
        batch_imgs = [fft2c_numpy(sample["kspace"], fftshift_dim=(-2, -1)) for sample in batch]

        # shuffle phase and amplitude among samples in a batch.
        if self.ap_shuffle:
            batch_imgs_amp = [np.abs(img) for img in batch_imgs]
            batch_imgs_phs = [img / img_abs for img, img_abs in zip(batch_imgs, batch_imgs_amp)]
            random.shuffle(batch_imgs_amp)
            batch_imgs = [img_abs * img_phs for img_abs, img_phs in zip(batch_imgs_amp, batch_imgs_phs)]

        # apply augmentations
        batch_tmsks = [sample["tmap_mask"] for sample in batch]
        imgs_masks = [self.augs_fun(img, tmap_mask) for img, tmap_mask in zip(batch_imgs, batch_tmsks)]
        batch_imgs = [img_mask[0] for img_mask in imgs_masks]
        batch_tmsks = [img_mask[1] for img_mask in imgs_masks]

        # turn to kspace domain
        batch_ksps = [ifft2c_numpy(img, fftshift_dim=(-2, -1)) for img in batch_imgs]

        # reset batch
        for idx in range(len(batch_ksps)):
            batch[idx]["kspace"] = batch_ksps[idx]
            batch[idx]["tmap_mask"] = batch_tmsks[idx]
        return batch