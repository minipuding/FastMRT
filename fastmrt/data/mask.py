"""
Modified from FastMRI: https://github.com/facebookresearch/fastMRI
"""

from fastmrt.utils.seed import temp_seed
from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch
from abc import ABC, abstractmethod

class MaskFunc(ABC):
    """
    An object for GRAPPA-style sampling masks like fastmri.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.

    When called, ``MaskFunc`` uses internal functions create mask by 1)
    creating a mask for the k-space center, 2) create a mask outside of the
    k-space center, and 3) combining them into a total mask. The internals are
    handled by ``sample_mask``, which calls ``calculate_center_mask`` for (1)
    and ``calculate_acceleration_mask`` for (2). The combination is executed
    in the ``MaskFunc`` ``__call__`` function.

    If you would like to implement a new mask, simply subclass ``MaskFunc``
    and overwrite the ``sample_mask`` logic. See examples in ``RandomMaskFunc``
    and ``EquispacedMaskFunc``.
    """

    def __init__(
        self,
        center_fraction: float,
        acceleration: int,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
            seed: Seed for starting the internal random number generator of the
                ``MaskFunc``.
        """
        if 1 / acceleration < center_fraction:
            raise ValueError("``center_fraction`` is too large,"
                             " it must be below {} when acceleration is {}.".format(1 / acceleration, acceleration))

        self.center_fraction = center_fraction
        self.acceleration = acceleration

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[np.float32, int]:
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space.
            offset: Offset from 0 to begin mask (for equispaced masks). If no
                offset is given, then one is selected randomly.
            seed: Seed for random number generator for reproducibility.

        Returns:
            A 2-tuple containing 1) the k-space mask and 2) the number of
            center frequency lines.
        """
        # if len(shape) < 3:
        #     raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(seed):
            mask_size = shape[-1]
            num_low_frequencies = round(mask_size * self.center_fraction)
            center_mask = self.calculate_center_mask(mask_size, shape, num_low_frequencies)
            accel_mask = self.calculate_acceleration_mask(mask_size, offset, num_low_frequencies)
            mask = np.maximum(center_mask, accel_mask)
            mask = self._reshape(mask, mask_size, shape)

        return mask, num_low_frequencies

    @abstractmethod
    def calculate_acceleration_mask(
        self,
        mask_size: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            mask_size: Number of columns or raws of k-space (2D subsampling).
            offset: Offset from 0 to begin masking (for equispaced masks).
            num_low_frequencies: Integer count of low-frequency lines sampled.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        raise NotImplementedError

    def calculate_center_mask(
        self,
        mask_size: int,
        shape: Sequence[int],
        num_low_freqs: int
    ) -> np.ndarray:
        """
        Build center mask based on number of low frequencies.

        Args:
            mask_size: Number of columns or raws of k-space (2D subsampling).
            shape: Shape of k-space to mask.
            num_low_freqs: Number of low-frequency lines to sample.

        Returns:
            A mask for hte low spatial frequencies of k-space.
        """
        center_mask = np.zeros(mask_size, dtype=np.float32)
        pad = (mask_size - num_low_freqs + 1) // 2
        center_mask[pad : pad + num_low_freqs] = 1
        assert center_mask.sum() == num_low_freqs

        return center_mask

    def _reshape(
            self,
            mask: np.ndarray,
            mask_size: int,
            shape: Sequence[int],
    ):
        reshaped_mask_shape = [1 for _ in shape]
        reshaped_mask_shape[-1] = mask_size
        reshaped_mask = mask.reshape(reshaped_mask_shape)

        return reshaped_mask

class RandomMaskFunc(MaskFunc):
    """
    Creates a random sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the ``RandomMaskFunc`` object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        prob = (num_cols / self.acceleration - num_low_frequencies) / (
            num_cols - num_low_frequencies
        )
        accel_mask = np.random.uniform(size=num_cols) < prob

        return accel_mask

class EquiSpacedMaskFunc(MaskFunc):
    """
    Sample data with equally-spaced k-space lines.

    Unlike FastMRI in this class, we modify it more make sense:
        1. Correcting that ``acceleration`` is greater than the true 
        acceleration rate due to the densely-sampled center.
        2. Adds a random offset to the sampling start position.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce a mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        # to ensure whole acceleration is correct
        if (1 - self.center_fraction * self.acceleration) <= 0:
            return np.zeros(num_cols, dtype=np.float32)

        # We set the real (corrected) acceleration as `A`, `c` as self.center_fraction, `a` as self.acceleration,
        # the number we expect to sample as `n`, and the total phase encoding number as `N`.
        # Then, N/n = A, N/((n * (1 - c)) + N * c) = a.
        # Therefore, A = (a * (1 - c)) / (1 - a * c).
        acceleration = (1 - self.center_fraction) * self.acceleration \
                        / (1 - self.center_fraction * self.acceleration)

        # set random offset
        if offset is None:
            offset = np.random.randint(0, high=acceleration)
        
        eq_masks = np.round(np.arange(offset, num_cols-1, acceleration)).astype(np.int32)

        accel_mask = np.zeros(num_cols, dtype=np.float32)
        accel_mask[eq_masks] = 1

        return accel_mask


def apply_mask(data: np.ndarray,
               mask_func: MaskFunc,
               seed = None):
    mask, num_low_frequencies = mask_func(data.shape, seed=seed)
    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros
    return masked_data, mask, num_low_frequencies
