"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import contextlib
from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch
from abc import ABC, abstractmethod

class MaskFunc(ABC):
    """
    An object for GRAPPA-style sampling masks.

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
        seed: Optional[int] = None,
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
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        shape: Sequence[int],
        direction: str = 'PE',
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[np.float32, int]:
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space.
            direction: The down-smapling direction using ``PE`` or ``FE``, which
                indicates ``phase-encoding direction`` and ``frequency-encoding``,
                default direction is ``PE``.
            offset: Offset from 0 to begin mask (for equispaced masks). If no
                offset is given, then one is selected randomly.
            seed: Seed for random number generator for reproducibility.

        Returns:
            A 2-tuple containing 1) the k-space mask and 2) the number of
            center frequency lines.
        """
        # if len(shape) < 3:
        #     raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            mask_size = shape[-1] if direction == 'PE' else shape[-2]
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
            direction: str = 'PE'
    ):
        reshaped_mask_shape = [1 for _ in shape]
        if direction == 'PE':
            reshaped_mask_shape[-1] = mask_size
        elif direction == 'FE':
            reshaped_mask_shape[-2] = mask_size
        else:
            raise ValueError('``direction`` paramter is uncorrect. It must be ``PE`` or ``FE``.')
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
        accel_mask = self.rng.uniform(size=num_cols) < prob

        return accel_mask

class EquiSpacedMaskFunc(MaskFunc):
    """
    Sample data with equally-spaced k-space lines.

    The lines are spaced exactly evenly, as is done in standard GRAPPA-style
    acquisitions. This means that with a densely-sampled center,
    ``acceleration`` will be greater than the true acceleration rate.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

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

        acceleration = round((1 - self.center_fraction) * self.acceleration
                             / (1 - self.center_fraction * self.acceleration))

        if offset is None:
            offset = self.rng.randint(0, high=acceleration)

        accel_mask = np.zeros(num_cols, dtype=np.float32)
        accel_mask[offset::acceleration] = 1

        return accel_mask

@contextlib.contextmanager
def temp_seed(rng: np.random.RandomState, seed: Optional[Union[int, Tuple[int, ...]]]):
    """A context manager for temporarily adjusting the random seed."""
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)

def apply_mask(data: torch.Tensor,
               mask_func: MaskFunc,
               direction: str = 'PE',
               seed = None):
    mask, num_low_frequencies = mask_func(data.shape, seed=seed)
    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros
    return masked_data, mask, num_low_frequencies
