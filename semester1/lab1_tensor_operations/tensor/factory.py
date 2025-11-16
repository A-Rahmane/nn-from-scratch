# tensor/factory.py
from .core import Tensor
import numpy as np
from typing import List

def zeros_like(t: Tensor) -> Tensor:
    return Tensor(np.zeros_like(t.data))

def ones_like(t: Tensor) -> Tensor:
    return Tensor(np.ones_like(t.data))

def rand_like(t: Tensor) -> Tensor:
    return Tensor(np.random.rand(*t.shape))

def randn_like(t: Tensor) -> Tensor:
    return Tensor(np.random.randn(*t.shape))

def concatenate(tensors: List[Tensor], axis=0) -> Tensor:
    return Tensor(np.concatenate([t.data for t in tensors], axis=axis))

def stack(tensors: List[Tensor], axis=0) -> Tensor:
    return Tensor(np.stack([t.data for t in tensors], axis=axis))
