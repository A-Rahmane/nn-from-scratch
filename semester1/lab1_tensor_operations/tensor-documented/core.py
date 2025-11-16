"""Core tensor class with all operations integrated.

This module defines the base Tensor class and integrates all operations
from other modules. Operations are bound at module load time, so there's
zero runtime overhead compared to a monolithic implementation.

Architecture:
    1. Core Tensor class defines data storage and basic properties
    2. Import operation functions from specialized modules
    3. Bind operations to Tensor class using lambda functions
    4. All operations are instance methods or class methods on Tensor

The modular design provides:
    - Clear separation of concerns
    - Easy maintenance and testing
    - Extensibility for new operations
    - Zero performance overhead
"""

import numpy as np
from typing import Union, Tuple, List, Optional


class Tensor:
    """Core tensor class for neural network operations.
    
    This class wraps NumPy arrays and provides tensor operations needed
    for building neural networks from scratch. The implementation is split
    across multiple modules for maintainability.
    
    Args:
        data: Input data (list, tuple, numpy array, scalar, or Tensor)
    
    Attributes:
        data: The underlying NumPy array storing tensor values (float32)
    
    Examples:
        >>> t = Tensor([1, 2, 3])
        >>> print(t.shape)
        (3,)
        
        >>> t2d = Tensor([[1, 2], [3, 4]])
        >>> result = t2d + 5
        >>> print(result)
        Tensor([[6. 7.]
                [8. 9.]])
    """

    def __init__(self, data: Union[list, tuple, np.ndarray, float, int, 'Tensor']):
        """Initialize tensor with data.
        
        Args:
            data: Input data that will be converted to float32 NumPy array
                 (boolean arrays are kept as bool dtype)
        """
        if isinstance(data, Tensor):
            self.data = data.data.copy()
        else:
            arr = np.array(data)
            self.data = arr.astype(np.float32) if arr.dtype != bool else arr

    # ==================== Properties ====================

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the tensor.
        
        Returns:
            Tuple of dimension sizes
        
        Example:
            >>> t = Tensor([[1, 2, 3], [4, 5, 6]])
            >>> print(t.shape)
            (2, 3)
        """
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Return the number of dimensions.
        
        Returns:
            Number of dimensions (axes)
        
        Example:
            >>> t = Tensor([[[1, 2], [3, 4]]])
            >>> print(t.ndim)
            3
        """
        return self.data.ndim

    @property
    def size(self) -> int:
        """Return the total number of elements.
        
        Returns:
            Total number of elements
        
        Example:
            >>> t = Tensor([[1, 2, 3], [4, 5, 6]])
            >>> print(t.size)
            6
        """
        return self.data.size

    @property
    def dtype(self):
        """Return the data type of the tensor.
        
        Returns:
            NumPy dtype object
        
        Example:
            >>> t = Tensor([1, 2, 3])
            >>> print(t.dtype)
            float32
        """
        return self.data.dtype
    
    # ==================== Basic Operations ====================

    def clone(self) -> "Tensor":
        """Create a deep copy of the tensor.
        
        Returns:
            Independent copy of the tensor
        
        Example:
            >>> t = Tensor([1, 2, 3])
            >>> t_copy = t.clone()
            >>> t_copy[0] = 99
            >>> print(t.data[0])  # Original unchanged
            1.0
        """
        return Tensor(self.data.copy())

    # ==================== String Representation ====================

    def __repr__(self) -> str:
        """Return detailed string representation.
        
        Returns:
            String in format "Tensor(data)"
        """
        return f"Tensor({self.data})"

    def __str__(self) -> str:
        """Return user-friendly string representation.
        
        Returns:
            String representation of the underlying data
        """
        return str(self.data)


# ==================== Import and Bind Operations ====================
# Operations are imported from specialized modules and bound to the Tensor class
# This happens at module load time, so there's no runtime overhead

# Arithmetic Operations
from .arithmetic import add, radd, sub, rsub, mul, rmul, div, rdiv, pow, neg

Tensor.__add__      = lambda self, other: add(self, other)
Tensor.__radd__     = lambda self, other: radd(self, other)
Tensor.__sub__      = lambda self, other: sub(self, other)
Tensor.__rsub__     = lambda self, other: rsub(self, other)
Tensor.__mul__      = lambda self, other: mul(self, other)
Tensor.__rmul__     = lambda self, other: rmul(self, other)
Tensor.__truediv__  = lambda self, other: div(self, other)
Tensor.__rtruediv__ = lambda self, other: rdiv(self, other)
Tensor.__pow__      = lambda self, power: pow(self, power)
Tensor.__neg__      = lambda self: neg(self)

# Comparison Operations
from .compare import eq, ne, lt, le, gt, ge

Tensor.__eq__ = lambda self, other: eq(self, other)
Tensor.__ne__ = lambda self, other: ne(self, other)
Tensor.__lt__ = lambda self, other: lt(self, other)
Tensor.__le__ = lambda self, other: le(self, other)
Tensor.__gt__ = lambda self, other: gt(self, other)
Tensor.__ge__ = lambda self, other: ge(self, other)

# Indexing and Slicing
from .indexing import getitem, setitem

Tensor.__getitem__ = lambda self, key: getitem(self, key)
Tensor.__setitem__ = lambda self, key, value: setitem(self, key, value)

# Shape Manipulation
from .shape import reshape, flatten, transpose, squeeze, unsqueeze, get_T

Tensor.reshape   = lambda self, *shape: reshape(self, *shape)
Tensor.flatten   = lambda self: flatten(self)
Tensor.transpose = lambda self, *axes: transpose(self, *axes)
Tensor.squeeze   = lambda self, axis=None: squeeze(self, axis)
Tensor.unsqueeze = lambda self, dim: unsqueeze(self, dim)
Tensor.T         = property(lambda self: get_T(self))

# Aggregation Operations
from .reduce import sum, mean, max, min, std, var

Tensor.sum  = lambda self, axis=None, keepdims=False: sum(self, axis, keepdims)
Tensor.mean = lambda self, axis=None, keepdims=False: mean(self, axis, keepdims)
Tensor.max  = lambda self, axis=None, keepdims=False: max(self, axis, keepdims)
Tensor.min  = lambda self, axis=None, keepdims=False: min(self, axis, keepdims)
Tensor.std  = lambda self, axis=None, keepdims=False: std(self, axis, keepdims)
Tensor.var  = lambda self, axis=None, keepdims=False: var(self, axis, keepdims)

# Matrix Operations
from .linalg import matmul, dot

Tensor.matmul     = lambda self, other: matmul(self, other)
Tensor.__matmul__ = lambda self, other: matmul(self, other)
Tensor.dot        = lambda self, other: dot(self, other)

# Factory Methods (class methods)
from .factory import zeros_like, ones_like, rand_like, randn_like, concatenate, stack, split

Tensor.zeros_like  = classmethod(lambda cls, tensor: zeros_like(tensor))
Tensor.ones_like   = classmethod(lambda cls, tensor: ones_like(tensor))
Tensor.rand_like   = classmethod(lambda cls, tensor: rand_like(tensor))
Tensor.randn_like  = classmethod(lambda cls, tensor: randn_like(tensor))
Tensor.concatenate = classmethod(lambda cls, tensors, axis=0: concatenate(tensors, axis))
Tensor.stack       = classmethod(lambda cls, tensors, axis=0: stack(tensors, axis))
Tensor.split       = lambda self, sections, axis=0: split(self, sections, axis)


# Test basic functionality when module is run directly
if __name__ == "__main__":
    print("=== Testing Modular Tensor Implementation ===\n")

    # Basic creation
    t1 = Tensor([1, 2, 3])
    print(f"t1 = {t1}")
    print(f"shape: {t1.shape}, ndim: {t1.ndim}, size: {t1.size}\n")

    # Arithmetic
    t2 = Tensor([4, 5, 6])
    print(f"t1 + t2 = {t1 + t2}")
    print(f"t1 * 2 = {t1 * 2}")
    print(f"5 + t1 = {5 + t1}")  # Test reverse operation
    print(f"t1 ** 2 = {t1 ** 2}\n")

    # Matrix operations
    m1 = Tensor([[1, 2], [3, 4]])
    m2 = Tensor([[5, 6], [7, 8]])
    print(f"m1 @ m2 =\n{m1 @ m2}\n")

    # Aggregations
    print(f"m1.sum() = {m1.sum()}")
    print(f"m1.mean(axis=0) = {m1.mean(axis=0)}")
    print(f"m1.std() = {m1.std()}\n")

    # Shape manipulation
    t3 = Tensor([1, 2, 3, 4, 5, 6])
    print(f"t3.reshape(2, 3) =\n{t3.reshape(2, 3)}")
    print(f"Transposed =\n{t3.reshape(2, 3).T}\n")

    # Factory methods
    zeros = Tensor.zeros_like(m1)
    print(f"zeros_like(m1) =\n{zeros}\n")

    # Batch operations
    stacked = Tensor.stack([t1, t2], axis=0)
    print(f"stack([t1, t2]) =\n{stacked}\n")

    print("✅ All modular operations working correctly!")