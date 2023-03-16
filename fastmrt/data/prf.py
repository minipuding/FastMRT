from typing import Union, NamedTuple
from fastmrt.utils.trans import real_tensor_to_complex_tensor
from pathlib import Path
import torch
import yaml

class PrfHeader(NamedTuple):
    B0: float
    gamma: float
    alpha: float
    TE: float


class PrfFunc():
    """
        Magnetic resonance temperature measurement based on proton resonance frequency (PRF)
    """
    def __init__(self, prf_header: PrfHeader):
        self.b0 = prf_header.B0
        self.gamma = prf_header.gamma
        self.alpha = prf_header.alpha
        self.te = prf_header.TE

    def __call__(
        self,
        cur_frame: torch.Tensor = None,
        ref_frame: torch.Tensor = None,
        delta_phs: torch.Tensor = None,
        is_phs: bool = False,
        offset: float = 37.0
    ) -> torch.Tensor:
        if is_phs is False and cur_frame is not None and ref_frame is not None:
            delta_phase = torch.angle(cur_frame * torch.conj(ref_frame))
        elif is_phs is True and delta_phs is not None:
            delta_phase = torch.angle(real_tensor_to_complex_tensor(delta_phs))
        else:
            raise ValueError("inputs must be one of frames or delta phase.")
        tempreture_map = delta_phase / ((self.gamma * 1e6) * (self.alpha * 1e-6)
                                        * self.b0 * self.te * 1e-3 * (2 * torch.pi))
        return tempreture_map + offset
