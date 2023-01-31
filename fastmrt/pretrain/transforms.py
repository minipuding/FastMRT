"""
针对fastmri数据集预训练做的数据变换，包括模拟加热、随机裁剪、其他数据增强
"""
import albumentations as A
from typing import Tuple, Union, Dict, Optional, Any

import numpy as np
import torch

from fastmrt.utils.fftc import ifft2c_numpy, fft2c_numpy
from fastmrt.utils.seed import temp_seed
from fastmrt.data.transforms import UNetDataTransform, CasNetDataTransform, RFTNetDataTransform
from fastmrt.data.mask import MaskFunc
from fastmrt.data.prf import PrfFunc

class SimuFocus:

    def __init__(
            self,
            frame_num: int = 11,
            cooling_time_rate: float = 0.2,
            center_crop_size: Union[int, Tuple[int, int]] = 224,
            random_crop_size: Union[int, Tuple[int, int]] = 96,
            max_delta_temp: float = 30,
            b0: float = 3.0,
            gamma: float = 42.576,
            alpha: float = 0.01,
            te: float = 12.0,
    ):
        super(SimuFocus, self).__init__()
        self.frame_num = frame_num - 1  # deduct t = 0
        self.cooling_frame_num = int(cooling_time_rate * frame_num)
        self.max_delta_phi = max_delta_temp * (gamma * 1e6) * b0 * (alpha * 1e-6) * (te * 1e-3) * (2 * np.pi)
        self.crop_func = Crop(center_crop_size, random_crop_size)

    def __call__(
        self,
        kspace,
        fname,
        dataslice,
        rng,
    ):
        delta_phis = np.append(np.sort(rng.rand(self.frame_num - self.cooling_frame_num)),
                               np.abs(np.sort(rng.rand(self.cooling_frame_num)) * 0.5 - 1)) * self.max_delta_phi
        cropped_image, cropped_kspace = self.crop_func(kspace)
        simufocus_mask = self.generate_simulated_focus(cropped_image.shape, rng)
        return {
            "image": cropped_image,
            "kspace": cropped_kspace,
            "delta_phis": delta_phis,
            "simufocus_mask": simufocus_mask,
            "fname": fname,
            "dataslice": dataslice,
        }

    def generate_simulated_focus(
            self,
            mask_size: Tuple[int, int],
            rng: np.random.RandomState,
    ):
        raise NotImplementedError


class SimuFocusGaussian(SimuFocus):
    """
    Re-write ``generate_simulated_focus`` function to generate gaussian typy focus.
    """
    def __init__(
            self,
            frame_num: int = 11,
            cooling_time_rate: float = 0.2,
            center_crop_size: Union[int, Tuple[int, int]] = 224,
            random_crop_size: Union[int, Tuple[int, int]] = 96,
            max_delta_temp: float = 30,
            shift_limit: float = 0.02,
            scale_limit: float = 0.05,
            sigma_scale_limit: Tuple[float, float] = (1, 6),
    ):
        super(SimuFocusGaussian, self).__init__(
            frame_num=frame_num,
            cooling_time_rate=cooling_time_rate,
            center_crop_size=center_crop_size,
            random_crop_size=random_crop_size,
            max_delta_temp=max_delta_temp,
        )
        self.sigma_scale_limit = sigma_scale_limit
        self.rotate_func = A.ShiftScaleRotate(shift_limit=shift_limit,
                                              scale_limit=scale_limit,
                                              rotate_limit=90,
                                              p=0.9)

    def generate_simulated_focus(
            self,
            mask_size: Tuple[int, int],
            rng: np.random.RandomState,
    ):
        sigma1 = 0.1
        sigma2 = sigma1 * (self.sigma_scale_limit[0] + rng.rand() *
                           (self.sigma_scale_limit[1] - self.sigma_scale_limit[0]))
        inv_A = 1 / (sigma1 ** 2)
        inv_B = 1 / (sigma2 ** 2)
        inv_C = 0
        gaussian_mask = self.gaussian_map(mask_size, mask_size[0] // 2, mask_size[1] // 2,
                                          inv_A, inv_B, inv_C)
        return self.rotate_func(image=gaussian_mask)["image"]

    @staticmethod
    def gaussian_map(
            shape: Tuple[int, int],
            cx: int,
            cy: int,
            _A: float = 2.0,
            _B: float = 2.0,
            _C: float = 0
    ):
        height, width = shape
        gauss_map = np.zeros(shape)
        x = np.linspace(0, width - 1, width).astype(np.int32)
        y = np.linspace(0, height - 1, height).astype(np.int32)
        X, Y = np.meshgrid(x, y)
        gauss_map[X, Y] = np.exp(- 0.5 * np.sqrt(_A * _B - _C ** 2) *
                                 (_A * ((X - cx) / width) ** 2 + _B * ((Y - cy) / height) ** 2
                                 + 2 * _C * ((X - cx) / width) * ((Y - cy) / height)))
        gauss_map = (gauss_map - np.min(gauss_map)) / (np.max(gauss_map) - np.min(gauss_map))
        return gauss_map


class SimuFocusKWave(SimuFocus):
    pass


class Crop:
    """
    Crop fastmri images to regular size that match to our dataset.
    """
    def __init__(
            self,
            center_crop_size: Union[int, Tuple[int, int]],
            random_crop_size: Union[int, Tuple[int, int]],
    ):
        super(Crop, self).__init__()

        if isinstance(center_crop_size, int):
            self.center_crop_height = center_crop_size
            self.center_crop_width = center_crop_size
        elif isinstance(center_crop_size, Tuple):
            self.center_crop_height = center_crop_size[1]
            self.center_crop_width = center_crop_size[0]

        if isinstance(random_crop_size, int):
            self.random_crop_height = random_crop_size
            self.random_crop_width = random_crop_size
        elif isinstance(random_crop_size, Tuple):
            self.random_crop_height = random_crop_size[1]
            self.random_crop_width = random_crop_size[0]

        self.center_crop_func = A.CenterCrop(self.center_crop_height, self.center_crop_width)
        self.random_crop_func = A.RandomCrop(self.random_crop_height, self.random_crop_width)

    def __call__(self, kspace):
        image = ifft2c_numpy(kspace, fftshift_dim=(-2, -1))
        cropped_image = self.random_crop_func(image=self.center_crop_func(image=image)["image"])["image"]
        cropped_kspace = fft2c_numpy(cropped_image, fftshift_dim=(-2, -1))
        return cropped_image, cropped_kspace


class FastmrtPretrainTransform:

    def __init__(
            self,
            mask_func: MaskFunc,
            prf_func: Optional[PrfFunc] = None,
            data_format: str = 'CF',
            use_random_seed: bool = True,
            resize_size: Tuple[int, int] = (224, 224),
            resize_mode: str = 'on_kspace',
            fftshift_dim: Union[int, Tuple[int, int]] = -2,
            simufocus_type: str = "gaussian",
            net: str = "r-unet",
            frame_num: int = 11,
            cooling_time_rate: float = 0.2,
            center_crop_size: Union[int, Tuple[int, int]] = 224,
            random_crop_size: Union[int, Tuple[int, int]] = 96,
            max_delta_temp: float = 30,
    ):
        super(FastmrtPretrainTransform, self).__init__()
        self.rng = np.random.RandomState()
        self.use_random_seed = use_random_seed
        if simufocus_type == "gaussian":
            self.simufocus_transform = SimuFocusGaussian(frame_num=frame_num,
                                                         cooling_time_rate=cooling_time_rate,
                                                         center_crop_size=center_crop_size,
                                                         random_crop_size=random_crop_size,
                                                         max_delta_temp=max_delta_temp)
        elif simufocus_type == "kwave":
            self.simufocus_transform = SimuFocusKWave()

        self.net = net
        if self.net == "r-unet":
            self.data_transform = UNetDataTransform(mask_func=mask_func,
                                                    prf_func=prf_func,
                                                    data_format=data_format,
                                                    use_random_seed=use_random_seed,
                                                    resize_size=resize_size,
                                                    resize_mode=resize_mode,
                                                    fftshift_dim=fftshift_dim)
        elif self.net == "casnet":
            self.data_transform = CasNetDataTransform(mask_func=mask_func,
                                                      prf_func=prf_func,
                                                      data_format=data_format,
                                                      use_random_seed=use_random_seed,
                                                      resize_size=resize_size,
                                                      fftshift_dim=fftshift_dim)
        elif self.net == "rftnet":
            self.data_transform = RFTNetDataTransform(mask_func=mask_func,
                                                      prf_func=prf_func,
                                                      data_format=data_format,
                                                      use_random_seed=use_random_seed,
                                                      resize_size=resize_size,
                                                      resize_mode=resize_mode,
                                                      fftshift_dim=fftshift_dim)

    def __call__(
            self,
            kspace,
            fname,
            dataslice,
            use_tmap_mask: bool = True,
    ):
        seed = None if self.use_random_seed is True else tuple(map(ord, fname))
        with temp_seed(self.rng, seed):
            simulated_data = self.simufocus_transform(kspace, fname, dataslice, self.rng)
            interface_data = self.transform_interface(simulated_data, self.rng)
        return self.data_transform(interface_data)

    @staticmethod
    def transform_interface(simulated_data, rng):
        max_phase = rng.choice(simulated_data["delta_phis"], 1)
        phase_map = np.exp(1j * max_phase * simulated_data["simufocus_mask"])
        kspace = fft2c_numpy(simulated_data["image"] * phase_map, fftshift_dim=(-2, -1))
        kspace_ref = fft2c_numpy(simulated_data["image"], fftshift_dim=(-2, -1))
        frame_idx = int(np.where(simulated_data["delta_phis"] == max_phase)[0])
        return {
            "kspace": kspace.astype(np.complex64),
            "kspace_ref": kspace_ref.astype(np.complex64),
            "tmap_mask": np.ones(kspace.shape),
            "file_name": simulated_data["fname"],
            "frame_idx": frame_idx,
            "slice_idx": simulated_data["dataslice"],
            "coil_idx": 0
        }


