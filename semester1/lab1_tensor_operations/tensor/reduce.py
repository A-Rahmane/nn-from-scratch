from .core import Tensor
import numpy as np
from typing import Optional, Tuple, Union

AxisType = Optional[Union[int, Tuple[int, ...]]]


def sum(t: Tensor, axis: AxisType = None, keepdims: bool = False) -> Tensor:
    return Tensor(np.sum(t.data, axis=axis, keepdims=keepdims))


def mean(t: Tensor, axis: AxisType = None, keepdims: bool = False) -> Tensor:
    return Tensor(np.mean(t.data, axis=axis, keepdims=keepdims))


def max(t: Tensor, axis: AxisType = None, keepdims: bool = False) -> Tensor:
    return Tensor(np.max(t.data, axis=axis, keepdims=keepdims))


def min(t: Tensor, axis: AxisType = None, keepdims: bool = False) -> Tensor:
    return Tensor(np.min(t.data, axis=axis, keepdims=keepdims))


def std(t: Tensor, axis: AxisType = None, keepdims: bool = False) -> Tensor:
    return Tensor(np.std(t.data, axis=axis, keepdims=keepdims))


def var(t: Tensor, axis: AxisType = None, keepdims: bool = False) -> Tensor:
    return Tensor(np.var(t.data, axis=axis, keepdims=keepdims))