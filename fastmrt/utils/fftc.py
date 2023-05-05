"""
Copyright (c) Sijie Xu with email:sijie.x@sjtu.edu.cn.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import List, Optional, Union, Tuple
import numpy as np
import torch


def fft2c_tensor(data : torch.Tensor, fftshift_dim: Optional[Union[int, Tuple[int, int]]] = (-2, -1)) -> torch.Tensor:
    """
        Apply centered 2 dimensional Fast Fourier Transform for tensor data.

        Args:
            data: Complex valued input data containing at least 3 dimensions:
                dimensions -3 & -2 are spatial dimensions and dimension -1 has size
                2. All other dimensions are assumed to be batch dimensions.
            norm: Normalization mode. See ``torch.fft.fft``.

        Returns:
            The FFT of the input.
    """
    if torch.is_complex(data) is False:
        raise ValueError("Narray does not complex type.")

    data = torch.fft.ifftshift(data, dim=fftshift_dim)
    data = torch.fft.fftn(data, dim=(-2, -1))
    data = torch.fft.fftshift(data)

    return data

def ifft2c_tensor(data : torch.Tensor, fftshift_dim: Optional[Union[int, Tuple[int, int]]] = (-2,-1)) -> torch.Tensor:
    """
        Apply centered 2-dimensional Inverse Fast Fourier Transform for tensor data.

        Args:
            data: Complex valued input data containing at least 3 dimensions:
                dimensions -3 & -2 are spatial dimensions and dimension -1 has size
                2. All other dimensions are assumed to be batch dimensions.
            norm: Normalization mode. See ``torch.fft.ifft``.

        Returns:
            The IFFT of the input.
    """
    if torch.is_complex(data) is False:
        raise ValueError("Narray does not complex type.")

    data = torch.fft.ifftshift(data)
    data = torch.fft.ifftn(data, dim=(-2,-1))
    data = torch.fft.fftshift(data, dim=fftshift_dim)

    return data

def fft2c_numpy(data : np.complex, fftshift_dim: Optional[Union[int, Tuple[int, int]]] = (-2,-1)) -> np.complex:
    """
        Apply centered 2 dimensional Fast Fourier Transform for numpy data.

        Args:
            data: Complex valued input data containing at least 2 dimensions:
                dimensions -2 & -1 are spatial dimensions. All other dimensions are assumed to be batch dimensions.

        Returns:
            The FFT of the input.
    """
    if np.iscomplex(data) is False:
        raise ValueError("Narray does not complex type.")

    data = np.fft.ifftshift(data, axes=fftshift_dim)
    data = np.fft.fftn(data, axes=(-2, -1))
    data = np.fft.fftshift(data)

    return data.astype(np.complex64)

def ifft2c_numpy(data : np.complex, fftshift_dim: Optional[Union[int, Tuple[int,...]]] = (-2,-1)) -> np.complex:
    """
        Apply centered 2-dimensional Inverse Fast Fourier Transform for numpy data.

        Args:
            data: Complex valued input data containing at least 2 dimensions:
                dimensions -2 & -1 are spatial dimensions. All other dimensions are assumed to be batch dimensions.

        Returns:
            The IFFT of the input.
    """
    if np.iscomplex(data) is False:
        raise ValueError("Narray does not complex type.")

    data = np.fft.ifftshift(data)
    data = np.fft.ifftn(data, axes=(-2, -1))
    data = np.fft.fftshift(data, axes=fftshift_dim)

    return data.astype(np.complex64)
