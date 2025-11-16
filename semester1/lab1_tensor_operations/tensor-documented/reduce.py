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
    
    Examples:
        >>> t = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> print(sum(t))
        Tensor(21.)
        
        >>> print(sum(t, axis=0))
        Tensor([5. 7. 9.])
        
        >>> print(sum(t, axis=1))
        Tensor([ 6. 15.])
        
        >>> print(sum(t, axis=1, keepdims=True))
        Tensor([[ 6.]
                [15.]])
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
    
    Examples:
        >>> t = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> print(mean(t))
        Tensor(3.5)
        
        >>> print(mean(t, axis=0))
        Tensor([2.5 3.5 4.5])
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
    
    Examples:
        >>> t = Tensor([[1, 5, 3], [4, 2, 6]])
        >>> print(max(t))
        Tensor(6.)
        
        >>> print(max(t, axis=1))
        Tensor([5. 6.])
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
    
    Examples:
        >>> t = Tensor([[1, 5, 3], [4, 2, 6]])
        >>> print(min(t))
        Tensor(1.)
        
        >>> print(min(t, axis=0))
        Tensor([1. 2. 3.])
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
    
    Examples:
        >>> t = Tensor([1, 2, 3, 4, 5])
        >>> print(std(t))
        Tensor(1.4142135)
        
        >>> t2d = Tensor([[1, 2], [3, 4]])
        >>> print(std(t2d, axis=0))
        Tensor([1. 1.])
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
    
    Examples:
        >>> t = Tensor([1, 2, 3, 4, 5])
        >>> print(var(t))
        Tensor(2.)
        
        >>> t2d = Tensor([[1, 2], [3, 4]])
        >>> print(var(t2d, axis=1))
        Tensor([0.25 0.25])
    """
    return Tensor(np.var(t.data, axis=axis, keepdims=keepdims))