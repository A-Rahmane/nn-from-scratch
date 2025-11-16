"""Indexing and slicing operations for Tensor class.

This module implements NumPy-style indexing and slicing:
- Integer indexing: t[0], t[1, 2]
- Slicing: t[1:3], t[:, 1:]
- Boolean indexing: t[t > 5]
- Fancy indexing: t[[0, 2, 4]]
- Item assignment: t[0] = 5

All indexing operations support the full NumPy indexing API.
"""

from .core import Tensor
import numpy as np
from typing import Union


def getitem(t: Tensor, key) -> Tensor:
    """Get item or slice from tensor.
    
    Supports all NumPy indexing modes:
    - Integer indexing
    - Slicing
    - Boolean masking
    - Fancy indexing
    - Multi-dimensional indexing
    
    Args:
        t: Input tensor
        key: Index, slice, tuple of indices, or boolean mask
    
    Returns:
        Indexed tensor
    
    Examples:
        >>> t = Tensor([1, 2, 3, 4, 5])
        >>> print(getitem(t, 2))
        Tensor(3.)
        
        >>> print(getitem(t, slice(1, 4)))
        Tensor([2. 3. 4.])
        
        >>> mask = Tensor([True, False, True, False, True])
        >>> print(getitem(t, mask))
        Tensor([1. 3. 5.])
    """
    # Convert Tensor boolean masks to numpy arrays
    if isinstance(key, Tensor):
        key = key.data
    
    # Handle tuple of indices (for multi-dimensional indexing)
    elif isinstance(key, tuple):
        key = tuple(k.data if isinstance(k, Tensor) else k for k in key)
    
    result = t.data[key]
    return Tensor(result)


def setitem(t: Tensor, key, value: Union[Tensor, np.ndarray, float, int]) -> None:
    """Set item or slice in tensor (in-place operation).
    
    This is the only in-place operation in the Tensor class.
    
    Args:
        t: Input tensor (modified in-place)
        key: Index, slice, or boolean mask
        value: Value to set (Tensor, array, or scalar)
    
    Examples:
        >>> t = Tensor([1, 2, 3])
        >>> setitem(t, 1, 10)
        >>> print(t)
        Tensor([ 1. 10.  3.])
        
        >>> setitem(t, slice(0, 2), Tensor([5, 6]))
        >>> print(t)
        Tensor([ 5.  6.  3.])
        
        >>> mask = Tensor([True, False, True])
        >>> setitem(t, mask, 0)
        >>> print(t)
        Tensor([0. 6. 0.])
    """
    # Convert Tensor key to numpy array
    if isinstance(key, Tensor):
        key = key.data
    
    # Handle tuple of indices
    elif isinstance(key, tuple):
        key = tuple(k.data if isinstance(k, Tensor) else k for k in key)
    
    # Convert Tensor value to numpy array
    if isinstance(value, Tensor):
        t.data[key] = value.data
    else:
        t.data[key] = value