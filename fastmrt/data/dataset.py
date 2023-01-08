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
    Tuple,
    Union,
)
from pathlib import Path
import os
import h5py
import fastmrt.utils.trans as tool
from fastmrt.utils.fftc import ifft2c_numpy


class Dataset(torch.utils.data.Dataset):
    """
    A pytorch Dataset that provides access to MR Thermomery image
    with (num_frame, num_coil, num_slice, height, width) shape.
    """

    def __init__(
            self,
            root: Union[str, Path, os.PathLike],
    ):
        """
        Args:
            root: Paths to the datasets.
        """
        # self.transform = transform
        # self.num_coils = num_coils
        self.data = []

        # load 5-D complex64 .h5 dataset
        file_names = os.listdir(root)
        for file_name in file_names:
            header, kspace, tmap_masks = self._load_data(os.path.join(root, file_name))
            self.data += [(header, kspace, tmap_masks, file_name)]  # load 5-D data

    def __getitem__(self, idx : int):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def _load_data(self, file_name):
        with h5py.File(file_name, "r") as hf:
            header = dict(hf.attrs)
            kspace = hf["kspace"][()].transpose()
            tmap_masks = hf["tmap_masks"][()].transpose()
        return header, kspace, tmap_masks

class SliceDataset(Dataset):
    """
    A pytorch Dataset that provides access to MR Thermomery image slices.
    We disassemble all datasets into two-dimensional slices, regardless of time series and inter-layer relationships.
    """

    def __init__(
        self,
        root : Union[str, Path, os.PathLike],
        transform : Optional[Callable] = None,
        ref_idx: int = 0
    ):
        """
        Args:
            root: Paths to the datasets.
            transforms: Optional; A sequence of callable objects that
                preprocesses the raw data into appropriate form.
                The transform function should take 'kspace', xxx as  inputs.
            ref_idx: int; Index of reference slice for calculating
                temperature maps (Tmap), default is 0.
        """
        super(SliceDataset, self).__init__(root)
        self.transform = transform
        self.slice_data = []

        # load slice kspace as dataset
        for header, kspace, tmap_masks, file_name in self.data:
            for slice_idx in range(int(header['slices'])):
                for coil_idx in range(int(header['coils'])):
                    tmap_mask = np.squeeze(tmap_masks[slice_idx, coil_idx, :, :])
                    for frame_idx in range(int(header['frames'])):
                        slice_kspace = np.squeeze(kspace[frame_idx, slice_idx, coil_idx, :, :])     # load current frame
                        slice_kspace_ref = np.squeeze(kspace[ref_idx, slice_idx, coil_idx, :, :])   # load reference frame
                        self.slice_data += [{"kspace": slice_kspace,
                                             "kspace_ref": slice_kspace_ref,
                                             "tmap_mask": tmap_mask,
                                             "file_name": file_name,
                                             "frame_idx": frame_idx,
                                             "slice_idx": slice_idx,
                                             "coil_idx": coil_idx,}]

    def __getitem__(self, idx : int):
        return self.transform(self.slice_data[idx])

    def __len__(self):
        return len(self.slice_data)

class VolumeDataset(Dataset):
    """
        A pytorch Dataset that provides access to MR Thermomery image volumes.
        We disassemble all datasets into three-dimensional volumes, regardless of time series relationships.
    """
    def __init__(
        self,
        root : Union[str, Path, os.PathLike],
        transform : Optional[Callable] = None,
    ):
        super(VolumeDataset, self).__init__(root)
        self.transform = transform
        self.volume_data = []

        # load volume kspace as dataset
        for header, kspace in self.data:
            for frame_idx in range(int(header['frames'])):
                for coil_idx in range(int(header['coils'])):
                    volume_kspace = np.squeeze(kspace[frame_idx, :, coil_idx, :, :])
                    volume = self.transform(volume_kspace)
                    self.volume_data += [volume]

    def __getitem__(self, idx : int):
        return self.volume_data[idx]

    def __len__(self):
        return len(self.volume_data)
