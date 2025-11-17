"""Comparison operations for Tensor class.

This module implements element-wise comparison operations:
- Equality: ==, !=
- Ordering: <, <=, >, >=

All operations return boolean tensors and support broadcasting.
"""

from .core import Tensor
import numpy as np
from typing import Union

Numeric = Union[Tensor, np.ndarray, float, int]


def eq(a: Tensor, b: Numeric) -> Tensor:
    """Element-wise equality comparison.
    
    Args:
        a: First tensor
        b: Second tensor or scalar
    
    Returns:
        Boolean tensor where True indicates equality
    """
    b = b if isinstance(b, Tensor) else Tensor(b)
    return Tensor((a.data == b.data).astype(bool))


def ne(a: Tensor, b: Numeric) -> Tensor:
    """Element-wise inequality comparison.
    
    Args:
        a: First tensor
        b: Second tensor or scalar
    
    Returns:
        Boolean tensor where True indicates inequality
    """
    b = b if isinstance(b, Tensor) else Tensor(b)
    return Tensor((a.data != b.data).astype(bool))


def lt(a: Tensor, b: Numeric) -> Tensor:
    """Element-wise less than comparison.
    
    Args:
        a: First tensor
        b: Second tensor or scalar
    
    Returns:
        Boolean tensor where True indicates a < b
    """
    b = b if isinstance(b, Tensor) else Tensor(b)
    return Tensor((a.data < b.data).astype(bool))


def le(a: Tensor, b: Numeric) -> Tensor:
    """Element-wise less than or equal comparison.
    
    Args:
        a: First tensor
        b: Second tensor or scalar
    
    Returns:
        Boolean tensor where True indicates a <= b
    """
    b = b if isinstance(b, Tensor) else Tensor(b)
    return Tensor((a.data <= b.data).astype(bool))


def gt(a: Tensor, b: Numeric) -> Tensor:
    """Element-wise greater than comparison.
    
    Args:
        a: First tensor
        b: Second tensor or scalar
    
    Returns:
        Boolean tensor where True indicates a > b
    """
    b = b if isinstance(b, Tensor) else Tensor(b)
    return Tensor((a.data > b.data).astype(bool))


def ge(a: Tensor, b: Numeric) -> Tensor:
    """Element-wise greater than or equal comparison.
    
    Args:
        a: First tensor
        b: Second tensor or scalar
    
    Returns:
        Boolean tensor where True indicates a >= b
    """
    b = b if isinstance(b, Tensor) else Tensor(b)
    return Tensor((a.data >= b.data).astype(bool))