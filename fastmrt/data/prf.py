from typing import Union, NamedTuple
from pathlib import Path
import torch
import numpy as np

class PrfHeader(NamedTuple):
    B0: float
    gamma: float
    alpha: float
    TE: float


class PrfFunc():
    """
        This class calculates the temperature map between the current frame and reference frame or 
        from the phase difference map directly. It is based on proton resonance frequency (PRF) 
        magnetic resonance temperature measurement.
        Args:
            prf_header: a NamedTuple type, must include four parameters:
                prf_header.B0: Magnetic resonance The strength of the main magnetic field, here is 3T.
                prf_header.gamma: Magnetic rotation ratio of hydrogen atom, which is a fixed constant, 42.57MHz/T.
                prf_header.alpha: Temperature sensitivity coefficient of water molecule, default is 0.01ppm/℃.
                prf_header.TE: echo time, here is 12ms.
    """
    def __init__(self, prf_header: PrfHeader):
        self.b0 = prf_header.B0
        self.gamma = prf_header.gamma
        self.alpha = prf_header.alpha
        self.te = prf_header.TE

    def __call__(
        self,
        cur_frame: Union[np.ndarray, torch.Tensor] = None,
        ref_frame: Union[np.ndarray, torch.Tensor] = None,
        delta_phs: Union[np.ndarray, torch.Tensor] = None,
        offset: float = 37.0
    ) -> Union[np.ndarray, torch.Tensor]:
        """
            This function calculates the temperature map between the current frame and reference frame or 
            from the phase difference map directly.
            Args:
                cur_frame: current frame, a 2-dim complex64 type tensor or numpy array. If None, `delta_phs` should be supported.
                ref_frame: reference frame, a 2-dim complex64 type tensor or numpy array. If None, `delta_phs` should be supported.
                delta_phs: phase difference map, a float32 type tensor or numpy array. If None, `cur_frame` and `ref_frame` should be supported. 
                          `delta_phs` is mutually exclusive to `cur_frame` and `ref_frame`.
                offset: the base temperature without heating, such as 37 degrees centigrade as body temperature 
                        and 25 degrees centigrade as indoor temperature.
            Returns:
                temperature map, a 2-dim complex64 type tensor or numpy array.
        """
        
        backend = np if isinstance(cur_frame, np.ndarray) or isinstance(delta_phs, np.ndarray) else torch
        if delta_phs is None:
            if  cur_frame is not None and ref_frame is not None:
                delta_phase = backend.angle(cur_frame * backend.conj(ref_frame))
            else:
                raise ValueError("Insufficient parameter, must input `delta_phs` or `cur_frame` and `ref_frame`.")
        else:
            if  cur_frame is None and ref_frame is None:
                delta_phase = delta_phs
            else:
                raise ValueError("Excessive parameter, must keep mutually exclusive to `delta_phs` and `cur_frame`, `ref_frame`.")
        tempreture_map = delta_phase / ((self.gamma * 1e6) * (self.alpha * 1e-6) * self.b0 * self.te * 1e-3 * (2 * backend.pi))
        return tempreture_map + offset
