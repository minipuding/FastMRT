from fastmrt.utils.trans import complex_np_to_real_np as cn2rn
from fastmrt.utils.trans import real_np_to_complex_np as rn2cn
from fastmrt.utils.fftc import fft2c_numpy, ifft2c_numpy
from fastmrt.utils.seed import temp_seed
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import Compose
import numpy as np
import random
import albumentations as A
from typing import Dict, Union, List


class ComplexAugs:
    """
     Apply normal augmentations (e.g. random crop and resize) on complex image.
     We can use augmentations only on amplitude, phase or both of them. This class only
     supports four basic augmentations: random crop and resize, random rotate (0, 90, 180 and 270 degrees),
     gaussian blur and random flip.

     Args:
         height: int, image height, default is fastmrt dataset height, 96
         width: int, image width, default is fastmrt dataset width, 96
         ca_rate: float, complex augments rate. That's the probability of triggering complex enhancement.
         objs: list[str], objects that augs applied on, only choose from `amp` and `phs` that represent amplitude and phs, respectively.
         ap_logic: str, amplitude and phase augs logic, if "and", the augs would be applied sequentially in amplitude and phase,
                 if "or", the augs would be applied one of amplitude and phase, default is ["amp", "phs"].
         augs_list: List[str], augmentation list, including `crop`(random resized crop), `rotate`(per 90 degree), 
                 `blur`(gaussian blur) and `flip`(vertical and horizontal), default is ["crop", "rotate", "blur", "flip"].
         compose_num: int, the number of augments to compose. It should be set between 0(not include) and len(augs_list), 
                 specially, if compose_num is 1, only one augments sample from augs_list would be applied, default is len(augs_list).
         crop_scale: tuple, the parameter `scale` of albumentations.RandomResizedCrop, default is (0.7, 1.)
         blur_limit: tuple, the parameter `blur_limit` of albumentations.GaussianBlur, default is (1, 5)
         
    """

    def __init__(self,
                 height: int=96,
                 width: int=96,
                 ca_rate: float=0.,
                 objs=None,
                 ap_logic: str="and",
                 augs_list=None,
                 compose_num=None,
                 crop_scale: tuple[float, float] = (0.7, 1.0),
                 blur_limit: Union[int, tuple[int, int]] = (1, 5),):
        
        # assert complex augment rate
        assert 0. <= ca_rate <= 1., f"`ca_rate` must be between 0. and 1. but got {ca_rate}."

        # assert objects
        if objs is None:
            objs = ["amp", "phs"]
        else:
            assert isinstance(objs, List)

        # assert augmentations list
        if augs_list is None:
            augs_list = ["crop", "rotate", "blur", "flip"]
        else:
            assert isinstance(augs_list, List), f"`augs_list` must be a list type, but got `{type(augs_list)}`"
        if compose_num is None or compose_num <= 0:
            compose_num = len(augs_list)
        elif len(augs_list) > 0:
            assert compose_num <= len(augs_list), f"`compose_num` must be in 0 < compose_num < len(augs_list), but got {compose_num}"

        # assignment
        self.ca_rate = ca_rate
        self.objs = objs
        self.ap_logic = ap_logic
        self.augs_list = augs_list
        self.compose_num = compose_num

        # define augments
        self.crop = A.RandomResizedCrop(height=height, width=width, scale=crop_scale)
        self.flip = A.Flip()
        self.rotate = A.RandomRotate90()
        self.blur = A.GaussianBlur(blur_limit=blur_limit)

    def __call__(self, sample, tmap_mask, seed=None):
        """
        Apply augmentations on the input sample and tmap_mask.

        Args:
            sample: complex64 numpy.ndarray, input sample
            tmap_mask: int or float32 numpy.ndarray, tmap mask

        Returns:
            sample: complex64, numpy.ndarray, augmented sample
            tmap_mask: int or float32 numpy.ndarray, augmented tmap mask
        """
        with temp_seed(seed=seed):
            if len(self.augs_list) == 0:
                Warning("The length os `augs_list` is 0, so the ComplexAugs would not be applied.")
                return sample, tmap_mask
            elif random.random() > self.ca_rate:  # Have a `1-ca_rate`` probability of using a normal augments.
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
            else:
                raise ValueError("The `objs` should not be empty when `ca_rate` is over 0.")
        return sample, tmap_mask

    def apply_augs(self, x):
        """
        Apply augmentations on the input x.

        Args:
            x: numpy.ndarray, input x

        Returns:
            x: numpy.ndarray, augmented x
        """
        augs = random.sample(self.augs_list, self.compose_num)
        if "crop" in augs:
            x = self.crop(image=x)["image"]
        if "flip" in augs:
            x = self.flip(image=x)["image"]
        if "rotate" in augs:
            x = self.rotate(image=x)["image"]
        if "blur" in augs:
            x = self.blur(image=x)["image"]
        return x


class IdentityAugs:
    """Do not apply anything. Just return inputs."""

    def __call__(self, sample, tmap_mask, seed=None):
        return sample, tmap_mask


class AugsCollateFunction:

    def __init__(self,
                 *args,
                 transforms,
                 height: int = 96,
                 width: int = 96,
                 ap_shuffle: bool=False,
                 **kwargs
                 ):
        """
        Initialize the AugsCollateFunction.

        Args:
            *args: tuple, positional arguments
            transforms: Compose, transforms
            height: int, image height
            width: int, image width
            ap_shuffle: bool, shuffle
            **kwargs: dict, keyword arguments
        """
        if isinstance(transforms, Compose):
            self.simu_focus_trans, self.under_sampling_trans = transforms.items()
        else:
            self.simu_focus_trans, self.under_sampling_trans = None, transforms
        self.augs_fun = ComplexAugs(*args, height=height, width=width, **kwargs)
        self.ap_shuffle = ap_shuffle

    def __call__(self, batch):
        """
        Apply augmentations on the input batch.

        Args:
            batch: list, input batch

        Returns:
            batch: list, augmented batch
        """
        if self.simu_focus_trans is not None:
            batch = [self.simu_focus_trans(sample) for sample in batch]

        batch = self._augs(batch)

        # apply down-sampling transform.
        batch = [self.under_sampling_trans(sample) for sample in batch]

        return default_collate(batch)

    def _augs(self, batch):
        """
        Apply augmentations on the input batch.

        Args:
            batch: list, input batch

        Returns:
            batch: list, augmented batch
        """
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
