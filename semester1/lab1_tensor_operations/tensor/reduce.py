from .core import Tensor
import numpy as np
from typing import Optional, Tuple, Union

def sum(t: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims=False) -> Tensor:
    return Tensor(np.sum(t.data, axis=axis, keepdims=keepdims))

def mean(t: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims=False) -> Tensor:
    return Tensor(np.mean(t.data, axis=axis, keepdims=keepdims))

def max(t: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims=False) -> Tensor:
    return Tensor(np.max(t.data, axis=axis, keepdims=keepdims))

def min(t: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims=False) -> Tensor:
    return Tensor(np.min(t.data, axis=axis, keepdims=keepdims))
