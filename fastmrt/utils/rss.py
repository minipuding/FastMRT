"""
Copyright (c) Sijie Xu with email:sijie.x@sjtu.edu.cn.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
from typing import Union, Tuple

def rss_tensor(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for torch.Tensor type.

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt((data**2).sum(dim)).squeeze()


def rss_complex_tensor(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for complex inputs with torch.Tensor type.

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt((complex_abs(data)**2).sum(dim))

def rss_complex_numpy(data : np.complex, dim : int = 0) -> np.complex:
    """
    Compute the Root Sum of Squares (RSS) for complex inputs with np.complex type.

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input array
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return np.norm(np.abs(data), dim = dim)

