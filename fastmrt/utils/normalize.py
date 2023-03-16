import torch
from typing import Union, Tuple


def normalize_paras(
    data: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    mean = torch.mean(data)
    std = torch.std(data)

    return mean, std


def normalize_apply(
        data: torch.Tensor,
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
        data: torch.Tensor,
        mean: Union[torch.Tensor, float],
        std: Union[torch.Tensor, float],
) -> torch.Tensor:

    return (data * std) + mean
