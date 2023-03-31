import contextlib
import numpy as np
import random
import torch
import albumentations as A
from typing import Union, Optional, Tuple


@contextlib.contextmanager
def temp_seed(seed: Optional[Union[int, Tuple[int, ...]]]):
    """
        A context manager for setting temporary seed.
        The code block under the `with temp_seed(seed):` would be use temporary 
        random seed on torch, numpy and random packages. Finally the random state 
        before would be set back. 
    """
    if seed is not None:
        try:
            np_state = np.random.get_state()
            torch_state = torch.get_rng_state()
            rd_state = random.getstate()
            np.random.seed(seed)
            torch.random.manual_seed(seed if isinstance(seed, int) else sum(seed))
            random.seed(seed)
            yield
        finally:
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            random.setstate(rd_state)
    else:
        yield