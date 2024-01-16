import torch
import numpy as np
from typing import Tuple


def complex_np_to_real_np(data: np.ndarray, mode: str = 'CHW') -> np.ndarray:
    if np.iscomplexobj(data):
        if mode == 'CHW':
            return np.stack((data.real, data.imag), axis=-3)
        elif mode == 'HWC':
            return np.stack((data.real, data.imag), axis=-1)
        else:
            raise ValueError("``mode`` must be one of ``CHW`` and ``HWC``, but {} was got.".format(mode))
    else:
        raise ValueError("``data`` must be ``torch.complex64`` or "
                         "``torch.complex128`` type, but a ``{}`` was got".format(data.dtype))


def complex_np_to_real_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-3)

    return torch.from_numpy(data)


def complex_tensor_to_real_np(data: torch.Tensor, mode: str = 'CHW') -> torch.Tensor:
    if torch.is_complex(data):
        if mode == 'CHW':
            return torch.stack((data.real, data.imag), dim=-3).numpy()
        elif mode == 'HWC':
            return torch.stack((data.real, data.imag), dim=-1).numpy()
        else:
            raise ValueError("``mode`` must be one of ``CHW`` and ``HWC``, but {} was got.".format(mode))
    else:
        raise ValueError("``data`` must be ``torch.complex64`` or "
                         "``torch.complex128`` type, but a ``{}`` was got".format(data.dtype))


def complex_tensor_to_real_tensor(data: torch.Tensor, mode: str = 'CHW') -> torch.Tensor:
    if torch.is_complex(data):
        if mode == 'CHW':
            return torch.stack((data.real, data.imag), dim=-3)
        elif mode == 'HWC':
            return torch.stack((data.real, data.imag), dim=-1)
        else:
            raise ValueError("``mode`` must be one of ``CHW`` and ``HWC``, but {} was got.".format(mode))
    else:
        raise ValueError("``data`` must be ``torch.complex64`` or "
                         "``torch.complex128`` type, but a ``{}`` was got".format(data.dtype))


def real_np_to_complex_tensor(data: np.ndarray) -> torch.Tensor:
    data = torch.from_numpy(data)
    if data.ndim > 2:
        permute_dims = [dim_idx for dim_idx in range(-data.ndim, 0) if dim_idx != -3]
        permute_dims.append(-3)
    else:
        raise ValueError(f"The ndim of tensor ``data`` should large than 3, but got `ndim={data.ndim}`.")
    return torch.view_as_complex(data.permute(permute_dims).contiguous())


def real_np_to_complex_np(data: np.ndarray) -> np.ndarray:
    return real_np_to_complex_tensor(data).numpy()


def real_tensor_to_complex_tensor(data: torch.Tensor) -> torch.Tensor:
    if data.ndim > 2:
        permute_dims = [dim_idx for dim_idx in range(-data.ndim, 0) if dim_idx != -3]
        permute_dims.append(-3)
    else:
        raise ValueError("The ndim of tensor ``data`` should large than 3.")
    return torch.view_as_complex(data.permute(permute_dims).contiguous())


def real_tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    return real_tensor_to_complex_tensor(data.cpu()).detach().numpy()


def complex_tensor_to_amp_phase_tensor(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if torch.is_complex(data):
        return torch.abs(data), torch.angle(data)
    else:
        raise ValueError("``data`` must be ``torch.complex64`` or "
                         "``torch.complex128`` type, but a ``{}`` was got".format(data.dtype))

def complex_np_to_amp_phase_np(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if np.iscomplexobj(data):
        return np.abs(data), np.angle(data)
    else:
        raise ValueError("``data`` must be ``np.complex64`` or "
                         "``np.complex128`` type, but a ``{}`` was got".format(data.dtype))