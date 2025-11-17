"""Shape manipulation operations for Tensor class.

This module implements operations that change or query tensor shape:
- reshape: Change shape while preserving data
- flatten: Convert to 1D
- transpose: Permute dimensions
- squeeze: Remove singleton dimensions
- unsqueeze: Add singleton dimensions
- T property: Transpose for 2D tensors

All operations return new tensors (no in-place modification).
"""

from .core import Tensor
import numpy as np
from typing import Optional, Tuple, Union


def reshape(t: Tensor, *shape: Union[int, Tuple[int, ...]]) -> Tensor:
    """Reshape tensor to new shape.
    
    Args:
        t: Input tensor
        *shape: New shape as individual arguments or tuple
    
    Returns:
        Reshaped tensor
    
    Examples:
        >>> t = Tensor([1, 2, 3, 4, 5, 6])
        >>> print(reshape(t, 2, 3))
        Tensor([[1. 2. 3.]
                [4. 5. 6.]])
        
        >>> print(reshape(t, (3, 2)))
        Tensor([[1. 2.]
                [3. 4.]
                [5. 6.]])
    """
    # Handle both reshape(2, 3) and reshape((2, 3))
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(t.data.reshape(shape))


def flatten(t: Tensor) -> Tensor:
    """Flatten tensor to 1D.
    
    Args:
        t: Input tensor
    
    Returns:
        Flattened 1D tensor
    """
    return Tensor(t.data.flatten())


def transpose(t: Tensor, *axes: int) -> Tensor:
    """Transpose tensor dimensions.
    
    Args:
        t: Input tensor
        *axes: Permutation of dimensions (optional)
               If not provided, reverses all dimensions
    
    Returns:
        Transposed tensor
    
    Examples:
        >>> t = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> print(transpose(t))
        Tensor([[1. 4.]
                [2. 5.]
                [3. 6.]])
        
        >>> t3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        >>> print(transpose(t3d, 2, 0, 1).shape)
        (2, 2, 2)
    """
    if axes:
        return Tensor(np.transpose(t.data, axes))
    return Tensor(t.data.T)


def get_T(t: Tensor) -> Tensor:
    """Get transpose of tensor (helper for T property).
    
    Args:
        t: Input tensor
    
    Returns:
        Transposed tensor
    """
    return Tensor(t.data.T)


def squeeze(t: Tensor, axis: Optional[int] = None) -> Tensor:
    """Remove dimensions of size 1.
    
    Args:
        t: Input tensor
        axis: Specific axis to squeeze (optional)
              If None, removes all singleton dimensions
    
    Returns:
        Squeezed tensor
    
    Examples:
        >>> t = Tensor([[[1, 2, 3]]])
        >>> print(squeeze(t).shape)
        (3,)
        
        >>> t = Tensor([[[1], [2], [3]]])
        >>> print(squeeze(t, axis=2).shape)
        (1, 3)
    """
    return Tensor(np.squeeze(t.data, axis=axis))


def unsqueeze(t: Tensor, dim: int) -> Tensor:
    """Add dimension of size 1 at specified position.
    
    Args:
        t: Input tensor
        dim: Position to insert new dimension
             Negative values count from the end
    
    Returns:
        Tensor with added dimension
    
    Examples:
        >>> t = Tensor([1, 2, 3])
        >>> print(unsqueeze(t, 0).shape)
        (1, 3)
        
        >>> print(unsqueeze(t, 1).shape)
        (3, 1)
        
        >>> print(unsqueeze(t, -1).shape)
        (3, 1)
    """
    return Tensor(np.expand_dims(t.data, axis=dim))