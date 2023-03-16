import torch
from typing import Tuple
from torch.nn.functional import interpolate


def resize(image: torch.Tensor,
           size: Tuple[int, ...],
           mode: str = 'bicubic') -> torch.Tensor:
    return interpolate(input=image.unsqueeze(0).unsqueeze(0),
                       size=size,
                       mode=mode).squeeze(0).squeeze(0)


def resize_on_image(image: torch.complex64,
                    size: Tuple[int, ...]) -> torch.Tensor:
    """

    :param image:
    :param size:
    :return:
    """
    real = interpolate(input=image.real.unsqueeze(0).unsqueeze(0),
                       size=size,
                       mode='bicubic').squeeze(0).squeeze(0)
    imag = interpolate(input=image.imag.unsqueeze(0).unsqueeze(0),
                       size=size,
                       mode='bicubic').squeeze(0).squeeze(0)
    return torch.complex(real, imag)


def resize_on_kspace(kspace: torch.Tensor,
                     size: Tuple[int, ...]) -> torch.Tensor:
    """

    :param kspace:
    :param size:
    :return:
    """
    height, width = kspace.shape
    resized_kspace = torch.complex(torch.zeros(size=size), torch.zeros(size=size))
    resized_kspace[(size[0] - height) // 2: size[0] - (size[0] - height) // 2,
                   (size[1] - width) // 2: size[1] - (size[1] - width) // 2] = kspace
    return resized_kspace
