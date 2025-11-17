"""Aggregation (reduction) operations for Tensor class.

This module implements operations that reduce tensor dimensions:
- sum: Sum of elements
- mean: Average of elements
- max: Maximum element
- min: Minimum element
- std: Standard deviation
- var: Variance

All operations support:
- Reduction over all elements (axis=None)
- Reduction along specific axes
- Keeping reduced dimensions (keepdims=True)
"""

from .core import Tensor
import numpy as np
from typing import Optional, Tuple, Union

AxisType = Optional[Union[int, Tuple[int, ...]]]


def sum(t: Tensor, axis: AxisType = None, keepdims: bool = False) -> Tensor:
    """Sum of tensor elements.
    
    Args:
        t: Input tensor
        axis: Axis or axes along which to sum
              None sums all elements
        keepdims: Whether to keep reduced dimensions as size 1
    
    Returns:
        Sum tensor
    """
    return Tensor(np.sum(t.data, axis=axis, keepdims=keepdims))


def mean(t: Tensor, axis: AxisType = None, keepdims: bool = False) -> Tensor:
    """Mean (average) of tensor elements.
    
    Args:
        t: Input tensor
        axis: Axis or axes along which to compute mean
        keepdims: Whether to keep reduced dimensions as size 1
    
    Returns:
        Mean tensor
    """
    return Tensor(np.mean(t.data, axis=axis, keepdims=keepdims))


def max(t: Tensor, axis: AxisType = None, keepdims: bool = False) -> Tensor:
    """Maximum of tensor elements.
    
    Args:
        t: Input tensor
        axis: Axis or axes along which to find maximum
        keepdims: Whether to keep reduced dimensions as size 1
    
    Returns:
        Maximum tensor
    """
    return Tensor(np.max(t.data, axis=axis, keepdims=keepdims))


def min(t: Tensor, axis: AxisType = None, keepdims: bool = False) -> Tensor:
    """Minimum of tensor elements.
    
    Args:
        t: Input tensor
        axis: Axis or axes along which to find minimum
        keepdims: Whether to keep reduced dimensions as size 1
    
    Returns:
        Minimum tensor
    """
    return Tensor(np.min(t.data, axis=axis, keepdims=keepdims))


def std(t: Tensor, axis: AxisType = None, keepdims: bool = False) -> Tensor:
    """Standard deviation of tensor elements.
    
    Args:
        t: Input tensor
        axis: Axis or axes along which to compute std
        keepdims: Whether to keep reduced dimensions as size 1
    
    Returns:
        Standard deviation tensor
    """
    return Tensor(np.std(t.data, axis=axis, keepdims=keepdims))


def var(t: Tensor, axis: AxisType = None, keepdims: bool = False) -> Tensor:
    """Variance of tensor elements.
    
    Args:
        t: Input tensor
        axis: Axis or axes along which to compute variance
        keepdims: Whether to keep reduced dimensions as size 1
    
    Returns:
        Variance tensor
    """
    return Tensor(np.var(t.data, axis=axis, keepdims=keepdims))