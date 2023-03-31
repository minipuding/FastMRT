import torch
import numpy as np
from typing import Union, Tuple


def normalize_paras(
    data: Union[torch.Tensor, np.ndarray]
) -> Tuple:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return mean, std


def normalize_apply(
        data: Union[torch.Tensor, np.ndarray],
        mean: Union[float, torch.Tensor],
        std: Union[float, torch.Tensor],
        eps: Union[float, torch.Tensor] = 0.0
) -> torch.Tensor:
    """

    :param data:
    :param mean:
    :param std:
    :param eps:
    :return:
    """
    return (data - mean) / (std + eps)


def denormalize(
        data: Union[torch.Tensor, np.ndarray],
        mean: Union[torch.Tensor, float],
        std: Union[torch.Tensor, float],
) -> Union[float, torch.Tensor]:

    return (data * std) + mean
