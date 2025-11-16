from .core import Tensor
import numpy as np
from typing import Union

Numeric = Union[Tensor, np.ndarray, float, int]

def _ensure_tensor(x: Numeric) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)

def add(a: Tensor, b: Numeric) -> Tensor:
    b = _ensure_tensor(b)
    return Tensor(a.data + b.data)

def sub(a: Tensor, b: Numeric) -> Tensor:
    b = _ensure_tensor(b)
    return Tensor(a.data - b.data)

def mul(a: Tensor, b: Numeric) -> Tensor:
    b = _ensure_tensor(b)
    return Tensor(a.data * b.data)

def div(a: Tensor, b: Numeric) -> Tensor:
    b = _ensure_tensor(b)
    return Tensor(a.data / b.data)

def pow(a: Tensor, exponent: float) -> Tensor:
    return Tensor(a.data ** exponent)

def neg(a: Tensor) -> Tensor:
    return Tensor(-a.data)
