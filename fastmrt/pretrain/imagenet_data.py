import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.sampler import RandomSampler
import torchvision.transforms as tv_trans
import random
import PIL.Image as Image
from glob import glob
import os
import numpy as np
from typing import Optional, Callable, Union, Iterator, Sized, Dict
from pathlib import Path
from fastmrt.data.transforms import ComplexAugs

from fastmrt.utils.fftc import fft2c_numpy, ifft2c_numpy, fft2c_tensor, ifft2c_tensor
from fastmrt.data.transforms import ComposeTransform


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


class ContractLearningRandomSampler(RandomSampler):

    def __init__(self, data_source: Sized, batch_size: int, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        super(ContractLearningRandomSampler, self).__init__(data_source=data_source,
                                                            replacement=replacement,
                                                            num_samples=num_samples,
                                                            generator=generator)
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64,
                                     generator=generator).tolist()

        # only overwrite when ``replacement`` is False.
        else:
            for _ in range(self.num_samples // n):
                indexes = torch.randperm(n, generator=generator).tolist()[: n - n % self.batch_size]  # must drop last
                loop_indexes = []
                for _ in range(self.batch_size):
                    indexes.insert(0, indexes.pop())
                    loop_indexes += indexes
                yield from loop_indexes
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]


class ContractLearningCollateFunction:

    def __init__(self, transforms):

        if isinstance(transforms, ComposeTransform):
            self.simu_focus_trans, self.under_sampling_trans = transforms.items()
        else:
            self.simu_focus_trans, self.under_sampling_trans = None, transforms

    def __call__(self, batch):
        if self.simu_focus_trans is not None:
            batch = [self.simu_focus_trans(sample) for sample in batch]

        batch = self.phase_augmentation(batch)

        # apply down-sampling transform.
        batch = [self.under_sampling_trans(sample) for sample in batch]

        return default_collate(batch)

    @staticmethod
    def cl_same_amp_same_phase_strategy(batch):
        # turn to image domain
        batch_imgs = [fft2c_numpy(sample["kspace"], fftshift_dim=(-2, -1)) for sample in batch]
        # keep phase and turn to the same amplitude.
        batch_imgs = [np.abs(batch_imgs[0]) / np.abs(img) * img for img in batch_imgs]
        # turn to kspace domain
        batch_ksps = [ifft2c_numpy(img, fftshift_dim=(-2, -1)) for img in batch_imgs]
        # reset batch
        for idx in range(len(batch_ksps)):
            batch[idx]["kspace"] = batch_ksps[idx]
        # construct positive and negative samples
        batch = 2 * batch
        return batch

    @staticmethod
    def cl_diff_amp_same_phase_strategy(batch):
        assert len(batch) % 2 == 0
        half_bs = len(batch) // 2

        # turn to image domain
        batch_imgs = [fft2c_numpy(sample["kspace"], fftshift_dim=(-2, -1)) for sample in batch]

        # # keep phase and turn to the corresponding amplitude.
        # batch_imgs = batch_imgs[:half_bs] + \
        #              [batch_imgs[idx] / np.abs(batch_imgs[idx]) * batch_imgs[half_bs+idx] for idx in range(half_bs)]

        # =============== DEBUG =======================
        # Experiment for comparing with ``cl_same_amp_same_phase_strategy`` to find whether
        # the batchnorm or matching is the essential reason.
        # Exp.01 verify matching
        # batch_imgs = [batch_imgs[0]] * len(batch)
        # # Exp.02 verify batchnorm
        batch_imgs_copy = batch_imgs[1:]
        batch_imgs_copy.insert(0, batch_imgs_copy.pop())
        batch_imgs = [batch_imgs[0]] + \
                     [img_c / np.abs(img_c) * img for img, img_c in zip(batch_imgs[1:], batch_imgs_copy)]


        # turn to kspace domain
        batch_ksps = [ifft2c_numpy(img, fftshift_dim=(-2, -1)) for img in batch_imgs]
        # reset batch
        for idx in range(len(batch_ksps)):
            batch[idx]["kspace"] = batch_ksps[idx]
        # construct positive and negative samples
        batch = 2 * batch
        return batch

    @staticmethod
    def phase_augmentation(batch):
        # turn to image domain
        batch_imgs = [fft2c_numpy(sample["kspace"], fftshift_dim=(-2, -1)) for sample in batch]
        # keep phase and turn to the same amplitude.
        batch_imgs_amp = [np.abs(img) for img in batch_imgs]
        batch_imgs_phs = [img / img_abs for img, img_abs in zip(batch_imgs, batch_imgs_amp)]
        random.shuffle(batch_imgs_phs)
        batch_imgs = [img_abs * img_phs for img_abs, img_phs in zip(batch_imgs_amp, batch_imgs_phs)]
        # turn to kspace domain
        batch_ksps = [ifft2c_numpy(img, fftshift_dim=(-2, -1)) for img in batch_imgs]
        # reset batch
        for idx in range(len(batch_ksps)):
            batch[idx]["kspace"] = batch_ksps[idx]
        # construct positive and negative samples
        # batch = 2 * batch
        return batch


class ContractLearningCollateFunctionV2(ContractLearningCollateFunction):

    def __init__(self,
                 transforms,
                 augs_strategy=None,
                 phs_aug: bool = True,
                 use_augs: bool = False,
                 n_views=4):
        super(ContractLearningCollateFunctionV2, self).__init__(transforms)
        if augs_strategy is None:
            self.augs_strategy = dict(
                union=False,                              # apply augmentations on amplitude and phase together
                objs=["amp", "phs"],                      # object applied on, only valid when `union` is `False`
                augs=["crop", "rotate", "blur", "flip"],  # augmentation types
                compose_num=None,                         # the number of augmentations sampling from `augs`,
                                                          # default is `None` that means apply all augmentations
                                                          # on `augs`.
            )
        self.phs_aug = phs_aug
        self.use_augs = use_augs
        self.complex_augs = ComplexAugs(strategy=self.augs_strategy, height=96, width=96)
        self.n_views = n_views

    def __call__(self, batch):
        if self.simu_focus_trans is not None:
            batch = [self.simu_focus_trans(sample) for sample in batch]

        collate_batches = [self.under_sampling_trans(sample) for sample in batch]
        collate_batches = [default_collate(collate_batches)]
        for _ in range(self.n_views - 1):
            cl_batch = self.amp_augmentation(batch)
            cl_batch = [self.under_sampling_trans(sample) for sample in cl_batch]
            collate_batches.append(default_collate(cl_batch))

        return collate_batches

    def amp_augmentation(self, batch):
        # turn to image domain
        batch_imgs = [fft2c_numpy(sample["kspace"], fftshift_dim=(-2, -1)) for sample in batch]
        # keep phase and turn to the same amplitude.
        if self.phs_aug is True:
            batch_imgs_amp = [np.abs(img) for img in batch_imgs]
            batch_imgs_phs = [img / img_abs for img, img_abs in zip(batch_imgs, batch_imgs_amp)]
            random.shuffle(batch_imgs_amp)
            batch_imgs = [img_abs * img_phs for img_abs, img_phs in zip(batch_imgs_amp, batch_imgs_phs)]
        # augmentations
        if self.use_augs is True:
            batch_masks = [sample["tmap_mask"] for sample in batch]
            imgs_masks = [self.complex_augs(img, mask) for img, mask in zip(batch_imgs, batch_masks)]
            batch_imgs = [img_mask[0] for img_mask in imgs_masks]
            batch_masks = [img_mask[1] for img_mask in imgs_masks]
            # save mask
            for idx in range(len(batch_masks)):
                batch[idx]["tmap_mask"] = batch_masks[idx]
        # turn to kspace domain
        batch_ksps = [ifft2c_numpy(img, fftshift_dim=(-2, -1)) for img in batch_imgs]
        # reset batch
        for idx in range(len(batch_ksps)):
            batch[idx]["kspace"] = batch_ksps[idx]
        # construct positive and negative samples
        # batch = 2 * batch
        return batch
