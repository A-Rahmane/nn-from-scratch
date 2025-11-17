"""Linear algebra operations for Tensor class.

This module implements matrix operations:
- matmul: Matrix multiplication (generalized for batches)
- dot: Dot product

Both operations follow NumPy/PyTorch conventions and include
proper error handling for incompatible shapes.
"""

from .core import Tensor
import numpy as np


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication.
    
    Performs standard matrix multiplication with broadcasting for batch operations.
    Follows the same rules as numpy.matmul and torch.matmul.
    
    Args:
        a: First tensor
        b: Second tensor
    
    Returns:
        Result of matrix multiplication
    
    Raises:
        ValueError: If shapes are incompatible for matrix multiplication
    """
    if not isinstance(b, Tensor):
        b = Tensor(b)
    
    try:
        result = np.matmul(a.data, b.data)
        return Tensor(result)
    except ValueError as e:
        raise ValueError(
            f"Incompatible shapes for matmul: {a.shape} and {b.shape}. "
            f"Inner dimensions must match for matrix multiplication."
        ) from e


def dot(a: Tensor, b: Tensor) -> Tensor:
    """Dot product.
    
    Computes the dot product of two tensors:
    - For 1D tensors: sum of element-wise products
    - For 2D tensors: matrix multiplication
    - For higher dimensions: sum product over last axis of a and second-to-last axis of b
    
    Args:
        a: First tensor
        b: Second tensor
    
    Returns:
        Dot product result
    """
    if not isinstance(b, Tensor):
        b = Tensor(b)
    
    return Tensor(np.dot(a.data, b.data))