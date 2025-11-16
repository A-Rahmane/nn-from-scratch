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
    
    Examples:
        >>> a = Tensor([[1, 2], [3, 4]])
        >>> b = Tensor([[5, 6], [7, 8]])
        >>> print(matmul(a, b))
        Tensor([[19. 22.]
                [43. 50.]])
        
        >>> # 1D @ 2D
        >>> v = Tensor([1, 2])
        >>> m = Tensor([[3, 4], [5, 6]])
        >>> print(matmul(v, m))
        Tensor([13. 16.])
        
        >>> # Batch matrix multiplication
        >>> a_batch = Tensor([[[1, 2]], [[3, 4]]])  # shape (2, 1, 2)
        >>> b_batch = Tensor([[[5], [6]], [[7], [8]]])  # shape (2, 2, 1)
        >>> result = matmul(a_batch, b_batch)
        >>> print(result.shape)
        (2, 1, 1)
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
    
    Examples:
        >>> # 1D dot product
        >>> a = Tensor([1, 2, 3])
        >>> b = Tensor([4, 5, 6])
        >>> print(dot(a, b))
        Tensor(32.)
        
        >>> # 2D matrix multiplication
        >>> a = Tensor([[1, 2], [3, 4]])
        >>> b = Tensor([[5, 6], [7, 8]])
        >>> print(dot(a, b))
        Tensor([[19. 22.]
                [43. 50.]])
    """
    if not isinstance(b, Tensor):
        b = Tensor(b)
    
    return Tensor(np.dot(a.data, b.data))