from .core import Tensor
import numpy as np
from typing import Union

Numeric = Union[Tensor, np.ndarray, float, int]


def _ensure_tensor(x: Numeric) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def add(a: Tensor, b: Numeric) -> Tensor:
    b = _ensure_tensor(b)
    return Tensor(a.data + b.data)


def radd(a: Tensor, b: Numeric) -> Tensor:
    return add(a, b)


def sub(a: Tensor, b: Numeric) -> Tensor:
    b = _ensure_tensor(b)
    return Tensor(a.data - b.data)


def rsub(a: Tensor, b: Numeric) -> Tensor:
    b = _ensure_tensor(b)
    return Tensor(b.data - a.data)


def mul(a: Tensor, b: Numeric) -> Tensor:
    b = _ensure_tensor(b)
    return Tensor(a.data * b.data)


def rmul(a: Tensor, b: Numeric) -> Tensor:
    return mul(a, b)


def div(a: Tensor, b: Numeric) -> Tensor:
    b = _ensure_tensor(b)
    return Tensor(a.data / b.data)


def rdiv(a: Tensor, b: Numeric) -> Tensor:
    b = _ensure_tensor(b)
    return Tensor(b.data / a.data)


def pow(a: Tensor, exponent: float) -> Tensor:
    return Tensor(a.data ** exponent)


def neg(a: Tensor) -> Tensor:
    return Tensor(-a.data)