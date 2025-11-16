from .core import Tensor
import numpy as np
from typing import Union


def getitem(t: Tensor, key) -> Tensor:
    if isinstance(key, Tensor):
        key = key.data
    elif isinstance(key, tuple):
        key = tuple(k.data if isinstance(k, Tensor) else k for k in key)
    
    result = t.data[key]
    return Tensor(result)


def setitem(t: Tensor, key, value: Union[Tensor, np.ndarray, float, int]) -> None:
    if isinstance(key, Tensor):
        key = key.data
    elif isinstance(key, tuple):
        key = tuple(k.data if isinstance(k, Tensor) else k for k in key)
    
    if isinstance(value, Tensor):
        t.data[key] = value.data
    else:
        t.data[key] = value