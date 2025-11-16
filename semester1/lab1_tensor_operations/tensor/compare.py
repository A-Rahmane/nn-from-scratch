from .core import Tensor
import numpy as np
from typing import Union

Numeric = Union[Tensor, np.ndarray, float, int]

def eq(a: Tensor, b: Numeric) -> Tensor:
    b = b if isinstance(b, Tensor) else Tensor(b)
    return Tensor((a.data == b.data).astype(bool))

def ne(a: Tensor, b: Numeric) -> Tensor:
    b = b if isinstance(b, Tensor) else Tensor(b)
    return Tensor((a.data != b.data).astype(bool))

def lt(a: Tensor, b: Numeric) -> Tensor:
    b = b if isinstance(b, Tensor) else Tensor(b)
    return Tensor((a.data < b.data).astype(bool))

def le(a: Tensor, b: Numeric) -> Tensor:
    b = b if isinstance(b, Tensor) else Tensor(b)
    return Tensor((a.data <= b.data).astype(bool))

def gt(a: Tensor, b: Numeric) -> Tensor:
    b = b if isinstance(b, Tensor) else Tensor(b)
    return Tensor((a.data > b.data).astype(bool))

def ge(a: Tensor, b: Numeric) -> Tensor:
    b = b if isinstance(b, Tensor) else Tensor(b)
    return Tensor((a.data >= b.data).astype(bool))
