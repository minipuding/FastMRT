from typing import Dict, NamedTuple, Union, Tuple, List, Callable
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
    

class FastmrtDataTransform2D:

    def __init__(
        self,
        mask_func: MaskFunc,
        prf_func: PrfFunc,
        aug_func: Callable=None,
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
        [image, image_ref, label, label_ref, tmap_mask, mask, mean, std] = \
            self._to_tensor([image, image_ref, label, label_ref, tmap_mask, mask, np.array(mean), np.array(std)])
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
            image = np.expand_dims(image, axis=0)
            image_ref = np.expand_dims(image_ref, axis=0)
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

