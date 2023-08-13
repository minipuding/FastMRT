from typing import Dict, NamedTuple
from fastmrt.data.transforms import FastmrtDataTransform2D
import numpy as np

class CSSample(NamedTuple):

    kspace: np.ndarray
    kspace_ref: np.ndarray
    mask_kspace: np.ndarray
    mask_kspace_ref: np.ndarray
    tmap_mask: np.ndarray
    mask: np.ndarray
    num_low_freqs: int
    metainfo: Dict

class CSTransform(FastmrtDataTransform2D):
        
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

        mask_size = mask_kspace.shape[-1]
        num_low_freqs = round(mask_size * self.mask_func.center_fraction)

        return CSSample(
            kspace=kspace,
            kspace_ref=kspace_ref,
            mask_kspace=mask_kspace,
            mask_kspace_ref=mask_kspace_ref,
            num_low_freqs=num_low_freqs,
            tmap_mask=tmap_mask,
            mask=mask,
            metainfo=data["metainfo"],
        )