"""
Copyright (c) Sijie Xu with email:sijie.x@sjtu.edu.cn.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
from typing import (
    Callable,
    Optional,
    Union,
    List,
)
from pathlib import Path
import os
import h5py


class Dataset(torch.utils.data.Dataset):
    """
    A pytorch Dataset that provides access to MR Thermomery image
    with (num_frame, num_coil, num_slice, height, width) shape.
    """

    def __init__(
            self,
            root: Union[str, List[str]],
    ):
        """
        Args:
            root: Paths to the datasets.
        """
        self.data = []

        # load 5-D complex64 .h5 dataset
        # [frames, slice, coils, height, width]
        if isinstance(root, str):
            root = [root]
        for sub_root in root:
            for path, _, file_names in os.walk(sub_root):
                self.data += [self._load_data(os.path.join(path, file_name)) for file_name in file_names]  # load 5-D data

    def __getitem__(self, idx : int):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def _load_data(self, file_name):
        with h5py.File(file_name, "r") as hf:
            header = dict(hf.attrs)
            kspace = hf["kspace"][()].transpose()
            tmap_masks = hf["tmap_masks"][()].transpose() \
                if hf["tmap_masks"][()].shape is not None else None
        log_file_name = '_'.join(file_name.split("/")[-4:])
        return header, kspace, tmap_masks, log_file_name

class SliceDataset(Dataset):
    """
    A pytorch Dataset that provides access to MR Thermomery image slices.
    We disassemble all datasets into two-dimensional slices, regardless of time series and inter-layer relationships.
    """

    def __init__(
        self,
        root : Union[str, List[str]],
        transform : Optional[Callable] = None,
        ref_idx: int = 0
    ):
        """
        Args:
            root: Paths to the datasets.
            transforms: Optional; A sequence of callable objects that
                preprocesses the raw data into appropriate form.
                The transform function should take 'kspace', 'kspace_ref', 'tmap_mask' and `metadata` as inputs.
            ref_idx: int; Index of reference slice for calculating
                temperature maps (TMap), default is 0.
        """
        super(SliceDataset, self).__init__(root)
        self.transform = transform
        self.slice_data = []

        # load slice kspace as dataset
        for header, kspace, tmap_masks, file_name in self.data:
            for slice_idx in range(int(header['slices'])):
                for coil_idx in range(int(header['coils'])):
                    tmap_mask = np.squeeze(tmap_masks[slice_idx, coil_idx, :, :]) \
                        if tmap_masks is not None else np.ones([header["width"], header["height"]])
                    for frame_idx in range(int(header['frames'])):
                        slice_kspace = np.squeeze(kspace[frame_idx, slice_idx, coil_idx, :, :])     # load current frame
                        slice_kspace_ref = np.squeeze(kspace[ref_idx, slice_idx, coil_idx, :, :])   # load reference frame
                        metainfo = dict(file_name=file_name, frame_idx=frame_idx, 
                                        slice_idx=slice_idx, coil_idx=coil_idx)
                        self.slice_data += [{"kspace": slice_kspace,
                                             "kspace_ref": slice_kspace_ref,
                                             "tmap_mask": tmap_mask,
                                             "metainfo": metainfo}]

    def __getitem__(self, idx : int):
        if self.transform is None:
            return self.slice_data[idx]
        else:
            return self.transform(self.slice_data[idx])

    def __len__(self):
        return len(self.slice_data)


class VolumeDataset(Dataset):
    """
        A pytorch Dataset that provides access to MR Thermomery image volumes.
        We disassemble all datasets into three-dimensional volumes, regardless of time series relationships.
        Note that we can transfer volume to image domain by 2d-ifft to each slice of volume rather than 3d-ifft.

        Args:
            root: paths to the datasets.
            transforms: optional; A sequence of callable objects that
                preprocesses the raw data into appropriate form.
                The transform function should take 'kspace', xxx as  inputs.
            ref_idx: int; index of reference slice for calculating
                temperature maps (TMap), default is 0.
    """
    def __init__(
        self,
        root : Union[str, List[str]],
        transform : Optional[Callable] = None,
        ref_idx: int = 0
    ):
        super(VolumeDataset, self).__init__(root)
        self.transform = transform
        self.volume_data = []

        # load volume kspace as dataset
        for header, kspace, tmap_mask, file_name in self.data:
            for frame_idx in range(int(header['frames'])):
                for coil_idx in range(int(header['coils'])):
                    volume_kspace = np.squeeze(kspace[frame_idx, :, coil_idx, :, :])
                    volume_kspace_ref = np.squeeze(kspace[ref_idx, :, coil_idx, :, :])
                    metainfo = dict(file_name=file_name, frame_idx=frame_idx, coil_idx=coil_idx)
                    self.volume_data += [{"volume_kspace": volume_kspace,
                                          "volume_kspace_ref": volume_kspace_ref,
                                          "tmap_mask": tmap_mask,
                                          "metainfo": metainfo}]

    def __getitem__(self, idx : int):
        if self.transform is None:
            return self.volume_data[idx]
        else:
            return self.transform(self.volume_data[idx])

    def __len__(self):
        return len(self.volume_data)
