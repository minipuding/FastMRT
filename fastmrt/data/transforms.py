import random
import time
from typing import Dict, NamedTuple, Union, Tuple, List
import torch
from fastmrt.data.mask import MaskFunc, apply_mask
from fastmrt.data.prf import PrfFunc
from fastmrt.utils.trans import (
    complex_tensor_to_real_tensor,
    complex_tensor_to_amp_phase_tensor,
)
from fastmrt.utils.trans import real_np_to_complex_np as rn2cn
from fastmrt.utils.trans import complex_np_to_real_np as cn2rn
from fastmrt.utils.fftc import ifft2c_tensor
from fastmrt.utils.resize import resize, resize_on_kspace, resize_on_image
from fastmrt.utils.normalize import normalize_apply, normalize_paras
import numpy as np
import albumentations as A


class Default(NamedTuple):
    kspace_torch: torch.Tensor
    target_torch: torch.Tensor


class DefaultTransform():

    def __call__(
            self,
            kspace: np.ndarray,
            target: np.ndarray,
    ):
        kspace_torch = torch.from_numpy(kspace)
        target_torch = torch.from_numpy(target)

        return Default(
            kspace_torch=kspace_torch,
            target_torch=target_torch
        )


class UNetSample(NamedTuple):
    # For training
    input: torch.Tensor
    label: torch.Tensor
    # For temperature map
    input_ref: torch.Tensor
    label_ref: torch.Tensor
    tmap_mask: torch.Tensor
    # For restore
    mask: torch.Tensor
    mean: Union[float, torch.Tensor]
    std: Union[float, torch.Tensor]
    origin_shape: Tuple[int]
    # For namer
    file_name: str
    frame_idx: int
    slice_idx: int
    coil_idx: int


class CasNetSample(NamedTuple):
    # For training
    input: torch.Tensor
    label: torch.Tensor
    # For temperature map
    input_ref: torch.Tensor
    label_ref: torch.Tensor
    tmap_mask: torch.Tensor
    # For restore
    mask: torch.Tensor
    # mean: Union[float, torch.Tensor]
    # std: Union[float, torch.Tensor]
    origin_shape: Tuple[int]
    # For namer
    file_name: str
    frame_idx: int
    slice_idx: int
    coil_idx: int


class RFTNetSample(NamedTuple):
    # For training
    input: torch.Tensor
    label_phs: torch.Tensor
    label_ref: torch.Tensor
    label_img: torch.Tensor
    # For temperature map
    tmap_mask: torch.Tensor
    # For restore
    mask: torch.Tensor
    # For namer
    file_name: str
    frame_idx: int
    slice_idx: int
    coil_idx: int


class KDNetSample(NamedTuple):
    # For training
    input_tea: torch.Tensor
    input_stu: torch.Tensor
    label: torch.Tensor
    # For temperature map
    input_ref: torch.Tensor
    label_ref: torch.Tensor
    tmap_mask: torch.Tensor
    # For restore
    mask: torch.Tensor
    mean: Union[float, torch.Tensor]
    std: Union[float, torch.Tensor]
    origin_shape: Tuple[int]
    # For namer
    file_name: str
    frame_idx: int
    slice_idx: int
    coil_idx: int


class UNetDataTransform:
    """
    Data Transformer for training U-Net model.
    """

    def __init__(
            self,
            mask_func: MaskFunc,
            prf_func: PrfFunc = None,
            data_format: str = 'CF',
            use_random_seed: bool = True,
            resize_size: Tuple[int, ...] = (224, 224),
            resize_mode: str = 'on_kspace',
            fftshift_dim: Union[int, Tuple[int, int]] = (-2, -1),
    ):
        """

        :param mask_func:
        """
        self.mask_func = mask_func
        self.prf_func = prf_func
        self.data_format = data_format
        self.use_random_seed = use_random_seed
        self.resize_size = resize_size
        self.resize_mode = resize_mode
        self.fftshift_dim = fftshift_dim

    def __call__(
            self,
            data: Dict,
            use_tmap_mask: bool = True,
    ) -> UNetSample:

        # to tensor
        kspace_torch = torch.from_numpy(data["kspace"])
        kspace_torch_ref = torch.from_numpy(data["kspace_ref"])

        # apply mask
        tag = f"{data['file_name']}_f{data['frame_idx']}s{data['slice_idx']}c{'coil_idx'}"
        seed = None if self.use_random_seed is True else tuple(map(ord, tag))
        mask_kspace, mask, _ = apply_mask(kspace_torch, self.mask_func, seed=seed)
        mask_kspace_ref = kspace_torch_ref * mask

        # apply inverse Fourier transform & resize
        if self.resize_mode == 'on_image':
            # mask kspace
            mask_image = ifft2c_tensor(mask_kspace, self.fftshift_dim)
            # mask_image = resize_on_image(mask_image, self.resize_size)
            # mask kspace reference
            mask_image_ref = ifft2c_tensor(mask_kspace_ref, self.fftshift_dim)
            # mask_image_ref = resize_on_image(mask_image_ref, self.resize_size)
            # full image
            full_image = ifft2c_tensor(kspace_torch, self.fftshift_dim)
            # full_image = resize_on_image(full_image, self.resize_size)
            # full image reference
            full_image_ref = ifft2c_tensor(kspace_torch_ref, self.fftshift_dim)
            # full_image_ref = resize_on_image(full_image_ref, self.resize_size)

        elif self.resize_mode == 'on_kspace':
            # mask kspace
            # mask_kspace = resize_on_kspace(mask_kspace, self.resize_size)
            mask_image = ifft2c_tensor(mask_kspace, self.fftshift_dim)
            # mask kspace reference
            # mask_kspace_ref = resize_on_kspace(mask_kspace_ref, self.resize_size)
            mask_image_ref = ifft2c_tensor(mask_kspace_ref, self.fftshift_dim)
            # full image
            # full_kspace = resize_on_kspace(kspace_torch, self.resize_size)
            full_image = ifft2c_tensor(kspace_torch, self.fftshift_dim)
            # full image reference
            # full_kspace_ref = resize_on_kspace(kspace_torch_ref, self.resize_size)
            full_image_ref = ifft2c_tensor(kspace_torch_ref, self.fftshift_dim)
        else:
            raise ValueError('``resize_type`` must be one of the ``on_image`` and ``on_kspace``, '
                             'but ``{}`` was got.'.format(self.resize_size))

        # apply data format transform
        if self.data_format == 'CF':    # Complex Float
            input = mask_image
            label = full_image
            input_ref = mask_image_ref
            label_ref = full_image_ref
        elif self.data_format == 'RF':  # Real Float
            input = complex_tensor_to_real_tensor(mask_image, mode='CHW')
            label = complex_tensor_to_real_tensor(full_image, mode='CHW')
            input_ref = complex_tensor_to_real_tensor(mask_image_ref, mode='CHW')
            label_ref = complex_tensor_to_real_tensor(full_image_ref, mode='CHW')
        elif self.data_format == 'TM':  # Temperature Map
            input = self.prf_func(mask_image, mask_image_ref)
            label = self.prf_func(full_image, full_image_ref)
            input_ref = None
            label_ref = None
        elif self.data_format == 'AP':  # Amplitude & Phase
            input = complex_tensor_to_amp_phase_tensor(mask_image)
            label = complex_tensor_to_amp_phase_tensor(full_image)
            input_ref = complex_tensor_to_amp_phase_tensor(mask_image_ref)
            label_ref = complex_tensor_to_amp_phase_tensor(full_image_ref)
        else:
            raise ValueError("``data_format`` must be one of ``CF``(Complex Float), ``RF``(Real Float),"
                             " ``TM``(Temperature Map) and ``AP``(Amplitude & Phase),"
                             " but ``{}`` was got.".format(self.data_format))

        # apply normalization
        mean, std = normalize_paras(input)
        input = normalize_apply(input, mean, std, eps=1e-12)
        label = normalize_apply(label, mean, std, eps=1e-12)
        input_ref = normalize_apply(input_ref, mean, std, eps=1e-12)
        label_ref = normalize_apply(label_ref, mean, std, eps=1e-12)

        # temperature map mask
        if use_tmap_mask is True:
            tmap_mask = torch.from_numpy(data["tmap_mask"])
            # tmap_mask = resize(tmap_mask_torch, self.resize_size, mode="nearest")
        else:
            tmap_mask = torch.ones(label.shape)

        return UNetSample(
            input=input,
            label=label,
            input_ref=input_ref,
            label_ref=label_ref,
            mask=mask,
            mean=mean,
            std=std,
            tmap_mask=tmap_mask,
            file_name=data["file_name"],
            frame_idx=data["frame_idx"],
            slice_idx=data["slice_idx"],
            coil_idx=data["coil_idx"],
            origin_shape=kspace_torch.shape,
        )


class CasNetDataTransform:
    """
    Data Transformer for training CasNet model.
    """

    def __init__(
            self,
            mask_func: MaskFunc,
            prf_func: PrfFunc = None,
            data_format: str = 'RF',
            use_random_seed: bool = True,
            resize_size: list[int, ...] = (256, 256),
            fftshift_dim: Union[int, Tuple[int, int]] = (-2, -1),
    ):
        """

        :param mask_func:
        """
        self.mask_func = mask_func
        self.prf_func = prf_func
        self.data_format = data_format
        self.use_random_seed = use_random_seed
        self.resize_size = resize_size
        self.fftshift_dim = fftshift_dim

    def __call__(
            self,
            data: Dict
    ) -> CasNetSample:

        # to tensor
        kspace_torch = torch.from_numpy(data["kspace"])
        kspace_torch_ref = torch.from_numpy(data["kspace_ref"])
        tmap_mask_torch = torch.from_numpy(data["tmap_mask"])

        # apply mask
        tag = f"{data['file_name']}_f{data['frame_idx']}s{data['slice_idx']}c{'coil_idx'}"
        seed = None if self.use_random_seed is True else tuple(map(ord, tag))
        mask_kspace, mask, _ = apply_mask(kspace_torch, self.mask_func, seed=seed)
        mask_kspace_ref = kspace_torch_ref * mask

        # apply resize by padding on kspace
        # mask_kspace = resize_on_kspace(mask_kspace, self.resize_size)
        # full_kspace = resize_on_kspace(kspace_torch, self.resize_size)
        # mask_kspace_ref = resize_on_kspace(mask_kspace_ref, self.resize_size)
        # full_kspace_ref = resize_on_kspace(kspace_torch_ref, self.resize_size)

        # apply inverse Fourier transform
        mask_image = ifft2c_tensor(mask_kspace, self.fftshift_dim)
        full_image = ifft2c_tensor(kspace_torch, self.fftshift_dim)
        mask_image_ref = ifft2c_tensor(mask_kspace_ref, self.fftshift_dim)
        full_image_ref = ifft2c_tensor(kspace_torch_ref, self.fftshift_dim)

        # apply data format transform
        if self.data_format == 'CF':  # Complex Float
            input = mask_image
            label = full_image
            input_ref = mask_image_ref
            label_ref = full_image_ref
        elif self.data_format == 'RF':  # Real Float
            input = complex_tensor_to_real_tensor(mask_image, mode='CHW')
            label = complex_tensor_to_real_tensor(full_image, mode='CHW')
            input_ref = complex_tensor_to_real_tensor(mask_image_ref, mode='CHW')
            label_ref = complex_tensor_to_real_tensor(full_image_ref, mode='CHW')
        else:
            raise ValueError("``data_format`` must be one of ``CF``(Complex Float), ``RF``(Real Float),"
                             " ``TM``(Temperature Map) and ``AP``(Amplitude & Phase),"
                             " but ``{}`` was got.".format(self.data_format))

        # apply normalization
        # mean, std = normalize_paras(input)
        # input = normalize_apply(input, mean, std, eps=1e-12)
        # label = normalize_apply(label, mean, std, eps=1e-12)
        # input_ref = normalize_apply(input_ref, mean, std, eps=1e-12)
        # label_ref = normalize_apply(label_ref, mean, std, eps=1e-12)

        # temperature map mask
        # tmap_mask = resize(tmap_mask_torch, self.resize_size, mode="nearest")
        tmap_mask = tmap_mask_torch

        return CasNetSample(
            input=input,
            label=label,
            input_ref=input_ref,
            label_ref=label_ref,
            mask=mask,
            origin_shape=kspace_torch.shape,
            tmap_mask=tmap_mask,
            file_name=data["file_name"],
            frame_idx=data["frame_idx"],
            slice_idx=data["slice_idx"],
            coil_idx=data["coil_idx"],
        )


class RFTNetDataTransform:
    """
    Data Transformer for training U-Net model.
    """

    def __init__(
            self,
            mask_func: MaskFunc,
            prf_func: PrfFunc = None,
            data_format: str = 'CF',
            use_random_seed: bool = True,
            resize_size: list[int, ...] = (256, 256),
            resize_mode: str = 'on_kspace',
            fftshift_dim: Union[int, Tuple[int, int]] = (-2, -1),
    ):
        """

        :param mask_func:
        """
        self.mask_func = mask_func
        self.prf_func = prf_func
        self.data_format = data_format
        self.use_random_seed = use_random_seed
        self.resize_size = resize_size
        self.resize_mode = resize_mode
        self.fftshift_dim = fftshift_dim

    def __call__(
            self,
            data: Dict
    ) -> RFTNetSample:

        # to tensor
        kspace_torch = torch.from_numpy(data["kspace"])
        kspace_torch_ref = torch.from_numpy(data["kspace_ref"])
        tmap_mask_torch = torch.from_numpy(data["tmap_mask"])

        # apply mask
        tag = f"{data['file_name']}_f{data['frame_idx']}s{data['slice_idx']}c{'coil_idx'}"
        seed = None if self.use_random_seed is True else tuple(map(ord, tag))
        mask_kspace, mask, _ = apply_mask(kspace_torch, self.mask_func, seed=seed)

        # apply inverse Fourier transform & resize
        if self.resize_mode == 'on_image':
            # mask kspace
            mask_image = ifft2c_tensor(mask_kspace, self.fftshift_dim)
            mask_image = resize_on_image(mask_image, self.resize_size)
            # full image
            full_image = ifft2c_tensor(kspace_torch, self.fftshift_dim)
            full_image = resize_on_image(full_image, self.resize_size)
            # full image reference
            full_image_ref = ifft2c_tensor(kspace_torch_ref, self.fftshift_dim)
            full_image_ref = resize_on_image(full_image_ref, self.resize_size)

        elif self.resize_mode == 'on_kspace':
            # mask kspace
            mask_kspace = resize_on_kspace(mask_kspace, self.resize_size)
            mask_image = ifft2c_tensor(mask_kspace, self.fftshift_dim)
            # full image
            full_kspace = resize_on_kspace(kspace_torch, self.resize_size)
            full_image = ifft2c_tensor(full_kspace, self.fftshift_dim)
            # full image reference
            full_kspace_ref = resize_on_kspace(kspace_torch_ref, self.resize_size)
            full_image_ref = ifft2c_tensor(full_kspace_ref, self.fftshift_dim)
        else:
            raise ValueError('``resize_type`` must be one of the ``on_image`` and ``on_kspace``, '
                             'but ``{}`` was got.'.format(self.resize_size))

        # apply data format transform
        if self.data_format == 'CF':  # Complex Float
            input = mask_image
            label_img = full_image
            label_ref = full_image_ref
        elif self.data_format == 'RF':  # Real Float
            input = complex_tensor_to_real_tensor(mask_image, mode='CHW')
            label_img = complex_tensor_to_real_tensor(full_image, mode='CHW')
            label_ref = complex_tensor_to_real_tensor(full_image_ref, mode='CHW')
        else:
            raise ValueError("``data_format`` must be one of ``CF``(Complex Float), ``RF``(Real Float),"
                             " ``TM``(Temperature Map) and ``AP``(Amplitude & Phase),"
                             " but ``{}`` was got.".format(self.data_format))

        # apply delta phase
        label_phs = complex_tensor_to_real_tensor(full_image * torch.conj(full_image_ref))

        # temperature map mask
        tmap_mask = resize(tmap_mask_torch, self.resize_size, mode="nearest")

        return RFTNetSample(
            input=input,
            label_phs=label_phs,
            label_ref=label_ref,
            label_img=label_img,
            mask=mask,
            tmap_mask=tmap_mask,
            file_name=data["file_name"],
            frame_idx=data["frame_idx"],
            slice_idx=data["slice_idx"],
            coil_idx=data["coil_idx"],
        )


class KDNetDataTransform:
    """
    Data Transformer for training Knowledge Distillation model.
    """

    def __init__(
            self,
            mask_func_tea: MaskFunc,
            mask_func_stu: MaskFunc,
            prf_func: PrfFunc = None,
            data_format: str = 'CF',
            use_random_seed: bool = True,
            fftshift_dim: Union[int, Tuple[int, int]] = (-2, -1),
    ):
        """

        :param mask_func:
        """
        self.mask_func_tea = mask_func_tea
        self.mask_func_stu = mask_func_stu
        self.prf_func = prf_func
        self.data_format = data_format
        self.use_random_seed = use_random_seed
        self.fftshift_dim = fftshift_dim

    def __call__(
            self,
            data: Dict,
            use_tmap_mask: bool = True,
    ) -> KDNetSample:

        # to tensor
        kspace_torch = torch.from_numpy(data["kspace"])
        kspace_torch_ref = torch.from_numpy(data["kspace_ref"])

        # apply mask
        tag = f"{data['file_name']}_f{data['frame_idx']}s{data['slice_idx']}c{'coil_idx'}"
        seed = None if self.use_random_seed is True else tuple(map(ord, tag))
        mask_kspace_tea, mask_tea, _ = apply_mask(kspace_torch, self.mask_func_tea, seed=seed)
        mask_kspace_stu, mask_stu, _ = apply_mask(kspace_torch, self.mask_func_stu, seed=seed)
        mask_kspace_ref = kspace_torch_ref * mask_stu

        # apply inverse Fourier transform
        # mask kspace
        mask_image_tea = ifft2c_tensor(mask_kspace_tea, self.fftshift_dim)
        mask_image_stu = ifft2c_tensor(mask_kspace_stu, self.fftshift_dim)
        # mask kspace reference
        mask_image_ref = ifft2c_tensor(mask_kspace_ref, self.fftshift_dim)
        # full image
        full_image = ifft2c_tensor(kspace_torch, self.fftshift_dim)
        # full image reference
        full_image_ref = ifft2c_tensor(kspace_torch_ref, self.fftshift_dim)

        # apply data format transform
        if self.data_format == 'CF':    # Complex Float
            input_tea = mask_image_tea
            input_stu = mask_image_stu
            label = full_image
            input_ref = mask_image_ref
            label_ref = full_image_ref
        elif self.data_format == 'RF':  # Real Float
            input_tea = complex_tensor_to_real_tensor(mask_image_tea, mode='CHW')
            input_stu = complex_tensor_to_real_tensor(mask_image_stu, mode='CHW')
            label = complex_tensor_to_real_tensor(full_image, mode='CHW')
            input_ref = complex_tensor_to_real_tensor(mask_image_ref, mode='CHW')
            label_ref = complex_tensor_to_real_tensor(full_image_ref, mode='CHW')
        elif self.data_format == 'TM':  # Temperature Map
            input_tea = self.prf_func(mask_image_tea, mask_image_ref)
            input_stu = self.prf_func(mask_image_stu, mask_image_ref)
            label = self.prf_func(full_image, full_image_ref)
            input_ref = None
            label_ref = None
        elif self.data_format == 'AP':  # Amplitude & Phase
            input_tea = complex_tensor_to_amp_phase_tensor(mask_image_tea)
            input_stu = complex_tensor_to_amp_phase_tensor(mask_image_stu)
            label = complex_tensor_to_amp_phase_tensor(full_image)
            input_ref = complex_tensor_to_amp_phase_tensor(mask_image_ref)
            label_ref = complex_tensor_to_amp_phase_tensor(full_image_ref)
        else:
            raise ValueError("``data_format`` must be one of ``CF``(Complex Float), ``RF``(Real Float),"
                             " ``TM``(Temperature Map) and ``AP``(Amplitude & Phase),"
                             " but ``{}`` was got.".format(self.data_format))

        # apply normalization
        mean, std = normalize_paras(input_stu)
        input_tea = normalize_apply(input_tea, mean, std, eps=1e-12)
        input_stu = normalize_apply(input_stu, mean, std, eps=1e-12)
        label = normalize_apply(label, mean, std, eps=1e-12)
        input_ref = normalize_apply(input_ref, mean, std, eps=1e-12)
        label_ref = normalize_apply(label_ref, mean, std, eps=1e-12)

        # temperature map mask
        if use_tmap_mask is True:
            tmap_mask = torch.from_numpy(data["tmap_mask"])
        else:
            tmap_mask = torch.ones(label.shape)

        return KDNetSample(
            input_tea=input_tea,
            input_stu=input_stu,
            label=label,
            input_ref=input_ref,
            label_ref=label_ref,
            mask=mask_stu,
            mean=mean,
            std=std,
            tmap_mask=tmap_mask,
            file_name=data["file_name"],
            frame_idx=data["frame_idx"],
            slice_idx=data["slice_idx"],
            coil_idx=data["coil_idx"],
            origin_shape=kspace_torch.shape,
        )


class ComposeTransform:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for tran in self.transforms:
            x = tran(x)
        return x

    def items(self):
        return self.transforms

