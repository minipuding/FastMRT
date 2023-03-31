from typing import Dict, NamedTuple, Union, Tuple, List
import torch
from fastmrt.data.mask import MaskFunc, apply_mask
from fastmrt.data.prf import PrfFunc
from fastmrt.data.augs import ComplexAugs, IdentityAugs
from fastmrt.utils.trans import (
    complex_tensor_to_real_tensor,
    complex_tensor_to_amp_phase_tensor,
)
from fastmrt.utils.trans import complex_np_to_amp_phase_np as cn2apn
from fastmrt.utils.trans import real_np_to_complex_np as rn2cn
from fastmrt.utils.trans import complex_np_to_real_np as cn2rn
from fastmrt.utils.fftc import ifft2c_tensor, fft2c_tensor, ifft2c_numpy, fft2c_numpy
from fastmrt.utils.resize import resize, resize_on_kspace, resize_on_image
from fastmrt.utils.normalize import normalize_apply, normalize_paras
import numpy as np


class FastmrtSample(NamedTuple):
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
    # For record data
    metainfo: Dict


class CasNetSample(NamedTuple):
    # For training
    input: torch.Tensor
    label: torch.Tensor
    phs_scale: torch.Tensor
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
    phs_scale: torch.Tensor
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

class FastmrtDataTransform2D:

    def __init__(
        self,
        mask_func: MaskFunc,
        prf_func: PrfFunc,
        aug_func: None,
        data_format: str = 'CF',
        use_augs: bool=True,
        use_random_seed: bool = True,
        fftshift_dim: Union[int, Tuple[int, int]] = (-2, -1),
    ):
        
        self.mask_func = mask_func
        self.prf_func = prf_func
        if use_augs is True and aug_func is not None:
            self.aug_func = aug_func
        else:
            self.aug_func = IdentityAugs()
        self.data_format = data_format
        self.use_random_seed = use_random_seed
        self.fftshift_dim = fftshift_dim
    
    def __call__(
            self,
            data: Dict,
            use_tmap_mask: bool = True,
    ):
        """
            Apply general fastmrt 2D data transforms:
                augs(optional) -> down-sampling -> ifft2c -> transfer data format -> normalization -> to tensor
            Dict include `kspace`, `kspace_ref`, `tmap_mask`, `metadata` at least.
        """

        # apply augmentations
        seed = self._generate_seed()
        kspace, tmap_mask = self._apply_augmentations(data["kspace"], data["tmap_mask"], seed=seed)
        kspace_ref, _ = self._apply_augmentations(data["kspace_ref"], data["tmap_mask"], seed=seed)

        # apply down-sampling mask
        mask_kspace, mask = self._apply_mask(kspace=kspace, data_info=data["metainfo"])
        mask_kspace_ref, _ = self._apply_mask(kspace=kspace_ref, data_info=data["metainfo"], mask=mask)

        # apply inverse Fourier transform
        [mask_image, mask_image_ref, full_image, full_image_ref] = \
            self._ifft2c([mask_kspace, mask_kspace_ref, kspace, kspace_ref])

        # apply data format transform
        image, image_ref = self._apply_data_format_trans(mask_image, mask_image_ref)
        label, label_ref = self._apply_data_format_trans(full_image, full_image_ref)

        # apply normalization
        [image, image_ref, label, label_ref], mean, std = \
            self._apply_normalize([image, image_ref, label, label_ref])

        # to tensor
        [image, image_ref, label, label_ref, tmap_mask, mean, std] = \
            self._to_tensor([image, image_ref, label, label_ref, tmap_mask, np.array(mean), np.array(std)])
        tmap_mask = tmap_mask if use_tmap_mask else torch.ones(image.shape)

        return FastmrtSample(
            input=image,
            label=label,
            input_ref=image_ref,
            label_ref=label_ref,
            tmap_mask=tmap_mask,
            mask=mask,
            mean=mean,
            std=std,
            metainfo=data["metainfo"],
        )
    
    def _apply_augmentations(self, kspace, tmap_mask, seed):
        """
        Apply augmentation on kspace and conrespodding temperature mask.

        Args:
            kspace: a complex64 numpy.ndarray, input kspace
            tmap_mask: temperature mask
        """
        # to time(image) domain
        image = fft2c_numpy(kspace, self.fftshift_dim)
        image, tmap_mask = self.aug_func(image, tmap_mask, seed=seed)
        kspace = ifft2c_numpy(image, self.fftshift_dim)
        return kspace, tmap_mask

    def _apply_mask(self, kspace, data_info, mask=None):
        """
        Apply down-sampling mask to kspace.
        
        Args:
            kspace: complex64 tensor, the 2-dim kspace of sample.
            data_info: a dict, should include `file_name`, `frame_idx`, `slice_idx` and `coil_idx`.
                It is used to generate a tag for seeding.
            mask: a given mask. If `mask` is None, the down-sampling would be down 
                by `self.mask_func` and return coorespoding mask, other wise the given mask
                would be applyed to the kspace directly. Default is None. 
        """
        if mask is None:
            tag = f"{data_info['file_name']}_f{data_info['frame_idx']}s{data_info['slice_idx']}c{'coil_idx'}"
            seed = None if self.use_random_seed is True else tuple(map(ord, tag))
            mask_kspace, mask, _ = apply_mask(kspace, self.mask_func, seed=seed)
            return mask_kspace, mask
        else:
            return kspace * mask, mask
    
    def _ifft2c(self, kspaces):
        """
        Apply inverse Fourier transform to a list of kspaces.

        Args:
            kspaces: a list of numpy.complex64 kspaces
        Returns:
            a list of numpy.complex64 images transfered from kspaces
        """
        assert len(kspaces) > 0, "the length of `datas` should be over 0."
        return [ifft2c_numpy(kspace, self.fftshift_dim) for kspace in kspaces]
    
    def _apply_data_format_trans(self, image, image_ref):
        """
        Apply data format transform.
        
        Args:
            image: a complex64 tensor image in time domain.
            image_ref: a complex64 tensor reference image in time domain.
        Returns:
            image: the image after data format transform.
            image_ref: the reference image after data format transform.
        """
        if self.data_format == 'CF':    # Complex Float
            image = image.unsqueeze(0)
            image_ref = image_ref.unsqueeze(0)
        elif self.data_format == 'RF':  # Real Float
            image = cn2rn(image, mode='CHW')
            image_ref = cn2rn(image_ref, mode='CHW')
        elif self.data_format == 'TM':  # Temperature Map
            image = self.prf_func(image, image_ref)
            image_ref = None
        elif self.data_format == 'AP':  # Amplitude & Phase
            image = cn2apn(image)
            image_ref = cn2apn(image_ref)
        else:
            raise ValueError("``data_format`` must be one of ``CF``(Complex Float), ``RF``(Real Float),"
                             " ``TM``(Temperature Map) and ``AP``(Amplitude & Phase),"
                             " but ``{}`` was got.".format(self.data_format))
        return image, image_ref
    
    def _apply_normalize(self, images: List, eps: float=1e-12):
        """
        Apply normalize to all elements of images list.
        The first one of images list would be used to calculate the mean and std,
        the apply the same mean and std to all elements of list.

        Args:
            images: a list of 2-dim tensor images. 
            eps: epsilon, a mini-float to avoid divide 0.
        
        Return:
            A list of images after normalization, mean and std.
        """
        assert len(images) > 0, "the length of `images` should be over 0."
        mean, std = normalize_paras(images[0])
        return [normalize_apply(img, mean, std, eps=eps) if img is not None else None for img in images], mean, std

    def _to_tensor(self, datas: List):
        """
        Transfer a list of numpy.ndarray to tensor.

        Args:
            datas: a list of numpy array
        Returns:
            a list of tensor transfered from datas
        """
        assert len(datas) > 0, "the length of `datas` should be over 0."
        return [torch.from_numpy(data)  if data is not None else None for data in datas]

    def _generate_seed(self):
        return np.random.randint(2 ** 32 - 1)


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
            fftshift_dim: Union[int, Tuple[int, int]] = (-2, -1),
    ):
        """

        :param mask_func:
        """
        self.mask_func = mask_func
        self.prf_func = prf_func
        self.data_format = data_format
        self.use_random_seed = use_random_seed
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
            fftshift_dim: Union[int, Tuple[int, int]] = (-2, -1),
    ):
        """

        :param mask_func:
        """
        self.mask_func = mask_func
        self.prf_func = prf_func
        self.data_format = data_format
        self.use_random_seed = use_random_seed
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


