"""Factory methods and batch operations for Tensor class.

This module provides:
- Factory methods: Create tensors based on existing tensor shapes
- Batch operations: Operations on multiple tensors

Factory Methods:
- zeros_like, ones_like: Create filled tensors with same shape
- rand_like, randn_like: Create random tensors with same shape

Batch Operations:
- concatenate: Join tensors along existing dimension
- stack: Join tensors along new dimension
- split: Divide tensor into multiple tensors
"""

from .core import Tensor
import numpy as np
from typing import List, Union


def zeros_like(t: Tensor) -> Tensor:
    """Create tensor of zeros with same shape as input.
    
    Args:
        t: Reference tensor for shape
    
    Returns:
        Tensor of zeros with same shape and dtype
    """
    return Tensor(np.zeros_like(t.data))


def ones_like(t: Tensor) -> Tensor:
    """Create tensor of ones with same shape as input.
    
    Args:
        t: Reference tensor for shape
    
    Returns:
        Tensor of ones with same shape and dtype
    """
    return Tensor(np.ones_like(t.data))


def rand_like(t: Tensor) -> Tensor:
    """Create tensor of random values [0, 1) with same shape as input.
    
    Args:
        t: Reference tensor for shape
    
    Returns:
        Tensor of random uniform values in [0, 1)
    """
    return Tensor(np.random.rand(*t.shape).astype(np.float32))


def randn_like(t: Tensor) -> Tensor:
    """Create tensor of random normal values with same shape as input.
    
    Values are sampled from standard normal distribution (mean=0, std=1).
    
    Args:
        t: Reference tensor for shape
    
    Returns:
        Tensor of random normal values
    """
    return Tensor(np.random.randn(*t.shape).astype(np.float32))


def concatenate(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """Concatenate tensors along an existing axis.
    
    All tensors must have the same shape except in the concatenation dimension.
    
    Args:
        tensors: List of tensors to concatenate
        axis: Axis along which to concatenate (default: 0)
    
    Returns:
        Concatenated tensor
    
    Raises:
        ValueError: If tensors have incompatible shapes
    """
    if not tensors:
        raise ValueError("Cannot concatenate empty list of tensors")
    
    arrays = [t.data for t in tensors]
    return Tensor(np.concatenate(arrays, axis=axis))


def stack(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """Stack tensors along a new axis.
    
    All tensors must have the same shape. Creates a new dimension at the specified axis.
    
    Args:
        tensors: List of tensors to stack
        axis: Position of new axis (default: 0)
    
    Returns:
        Stacked tensor with one additional dimension
    
    Raises:
        ValueError: If tensors have different shapes
    """
    if not tensors:
        raise ValueError("Cannot stack empty list of tensors")
    
    arrays = [t.data for t in tensors]
    return Tensor(np.stack(arrays, axis=axis))


def split(t: Tensor, sections: Union[int, List[int]], axis: int = 0) -> List[Tensor]:
    """Split tensor into multiple tensors along an axis.
    
    Args:
        t: Input tensor to split
        sections: Either:
                 - int: Number of equal-sized sections
                 - list: Indices where to split
        axis: Axis along which to split (default: 0)
    
    Returns:
        List of tensors
    
    Raises:
        ValueError: If tensor cannot be evenly divided
    """
    arrays = np.split(t.data, sections, axis=axis)
    return [Tensor(arr) for arr in arrays]