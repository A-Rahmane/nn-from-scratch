"""Tensor module - Modular implementation of tensor operations.

This module provides a comprehensive Tensor class with operations 
organized into separate files for maintainability:

- core.py: Basic tensor class and properties
- arithmetic.py: Arithmetic operations (+, -, *, /, **)
- compare.py: Comparison operations (==, !=, <, >, <=, >=)
- indexing.py: Indexing and slicing operations
- shape.py: Shape manipulation (reshape, transpose, flatten, etc.)
- reduce.py: Aggregation operations (sum, mean, max, min, std, var)
- linalg.py: Linear algebra operations (matmul, dot)
- factory.py: Factory methods and batch operations

Architecture:
    The Tensor class is split into focused modules:
    - Each module handles a specific category of operations
    - core.py imports and binds all operations at module load time
    - Zero runtime overhead compared to monolithic implementation
    - Easier to maintain, test, and extend
"""

from .core import Tensor

__all__ = ['Tensor']

__version__ = '1.0.0'
__author__ = 'MENOUER Abderrahmane'

# Module-level convenience functions
def tensor(data):
    """Create a tensor from data.
    
    Args:
        data: Input data (list, tuple, numpy array, or scalar)
    
    Returns:
        Tensor object
    """
    return Tensor(data)


def zeros(shape):
    """Create a tensor of zeros.
    
    Args:
        shape: Shape of the tensor
    
    Returns:
        Tensor of zeros
    """
    import numpy as np
    return Tensor(np.zeros(shape, dtype=np.float32))


def ones(shape):
    """Create a tensor of ones.
    
    Args:
        shape: Shape of the tensor
    
    Returns:
        Tensor of ones
    """
    import numpy as np
    return Tensor(np.ones(shape, dtype=np.float32))


def rand(shape):
    """Create a tensor of random values in [0, 1).
    
    Args:
        shape: Shape of the tensor
    
    Returns:
        Tensor of random values
    """
    import numpy as np
    return Tensor(np.random.rand(*shape))


def randn(shape):
    """Create a tensor of random normal values.
    
    Args:
        shape: Shape of the tensor
    
    Returns:
        Tensor of random normal values
    """
    import numpy as np
    return Tensor(np.random.randn(*shape))