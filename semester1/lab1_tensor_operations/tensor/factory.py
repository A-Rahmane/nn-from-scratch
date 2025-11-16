from .core import Tensor
import numpy as np
from typing import List, Union


def zeros_like(t: Tensor) -> Tensor:
    return Tensor(np.zeros_like(t.data))


def ones_like(t: Tensor) -> Tensor:
    return Tensor(np.ones_like(t.data))


def rand_like(t: Tensor) -> Tensor:
    return Tensor(np.random.rand(*t.shape).astype(np.float32))


def randn_like(t: Tensor) -> Tensor:
    return Tensor(np.random.randn(*t.shape).astype(np.float32))


def concatenate(tensors: List[Tensor], axis: int = 0) -> Tensor:
    if not tensors:
        raise ValueError("Cannot concatenate empty list of tensors")
    
    arrays = [t.data for t in tensors]
    return Tensor(np.concatenate(arrays, axis=axis))


def stack(tensors: List[Tensor], axis: int = 0) -> Tensor:
    if not tensors:
        raise ValueError("Cannot stack empty list of tensors")
    
    arrays = [t.data for t in tensors]
    return Tensor(np.stack(arrays, axis=axis))


def split(t: Tensor, sections: Union[int, List[int]], axis: int = 0) -> List[Tensor]:
    arrays = np.split(t.data, sections, axis=axis)
    return [Tensor(arr) for arr in arrays]