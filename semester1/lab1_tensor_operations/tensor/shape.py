from .core import Tensor
import numpy as np
from typing import Optional, Tuple

def reshape(t: Tensor, *shape: int) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(t.data.reshape(shape))

def flatten(t: Tensor) -> Tensor:
    return Tensor(t.data.flatten())

def transpose(t: Tensor, *axes: int) -> Tensor:
    if axes:
        return Tensor(np.transpose(t.data, axes))
    return Tensor(t.data.T)

def squeeze(t: Tensor, axis: Optional[int] = None) -> Tensor:
    return Tensor(np.squeeze(t.data, axis=axis))

def unsqueeze(t: Tensor, dim: int) -> Tensor:
    return Tensor(np.expand_dims(t.data, axis=dim))
