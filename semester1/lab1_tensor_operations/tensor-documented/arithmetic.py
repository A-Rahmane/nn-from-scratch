"""Arithmetic operations for Tensor class.

This module implements all arithmetic operations including:
- Basic operations: add, subtract, multiply, divide, power, negate
- Reverse operations: radd, rsub, rmul, rdiv (for scalar op tensor)
- Type coercion: Automatic conversion of scalars and arrays to Tensors

All operations support broadcasting as per NumPy conventions.
"""

from .core import Tensor
import numpy as np
from typing import Union

Numeric = Union[Tensor, np.ndarray, float, int]


def _ensure_tensor(x: Numeric) -> Tensor:
    """Convert input to Tensor if it isn't already.
    
    Args:
        x: Input value (Tensor, array, or scalar)
    
    Returns:
        Tensor object
    """
    return x if isinstance(x, Tensor) else Tensor(x)


def add(a: Tensor, b: Numeric) -> Tensor:
    """Add two tensors or tensor and scalar.
    
    Args:
        a: First tensor
        b: Second tensor or scalar
    
    Returns:
        Result of a + b
    
    Example:
        >>> a = Tensor([1, 2, 3])
        >>> b = Tensor([4, 5, 6])
        >>> result = add(a, b)
        >>> print(result)
        Tensor([5. 7. 9.])
    """
    b = _ensure_tensor(b)
    return Tensor(a.data + b.data)


def radd(a: Tensor, b: Numeric) -> Tensor:
    """Reverse addition (scalar + tensor).
    
    Args:
        a: Tensor
        b: Scalar or tensor
    
    Returns:
        Result of b + a
    
    Example:
        >>> t = Tensor([1, 2, 3])
        >>> result = radd(t, 5)  # 5 + t
        >>> print(result)
        Tensor([6. 7. 8.])
    """
    return add(a, b)  # Addition is commutative


def sub(a: Tensor, b: Numeric) -> Tensor:
    """Subtract two tensors or tensor and scalar.
    
    Args:
        a: First tensor
        b: Second tensor or scalar
    
    Returns:
        Result of a - b
    
    Example:
        >>> a = Tensor([5, 7, 9])
        >>> b = Tensor([1, 2, 3])
        >>> result = sub(a, b)
        >>> print(result)
        Tensor([4. 5. 6.])
    """
    b = _ensure_tensor(b)
    return Tensor(a.data - b.data)


def rsub(a: Tensor, b: Numeric) -> Tensor:
    """Reverse subtraction (scalar - tensor).
    
    Args:
        a: Tensor
        b: Scalar or tensor
    
    Returns:
        Result of b - a
    
    Example:
        >>> t = Tensor([1, 2, 3])
        >>> result = rsub(t, 10)  # 10 - t
        >>> print(result)
        Tensor([9. 8. 7.])
    """
    b = _ensure_tensor(b)
    return Tensor(b.data - a.data)


def mul(a: Tensor, b: Numeric) -> Tensor:
    """Element-wise multiplication.
    
    Args:
        a: First tensor
        b: Second tensor or scalar
    
    Returns:
        Result of a * b
    
    Example:
        >>> a = Tensor([1, 2, 3])
        >>> b = Tensor([2, 3, 4])
        >>> result = mul(a, b)
        >>> print(result)
        Tensor([ 2.  6. 12.])
    """
    b = _ensure_tensor(b)
    return Tensor(a.data * b.data)


def rmul(a: Tensor, b: Numeric) -> Tensor:
    """Reverse multiplication (scalar * tensor).
    
    Args:
        a: Tensor
        b: Scalar or tensor
    
    Returns:
        Result of b * a
    
    Example:
        >>> t = Tensor([1, 2, 3])
        >>> result = rmul(t, 5)  # 5 * t
        >>> print(result)
        Tensor([ 5. 10. 15.])
    """
    return mul(a, b)  # Multiplication is commutative


def div(a: Tensor, b: Numeric) -> Tensor:
    """Element-wise division.
    
    Args:
        a: First tensor (numerator)
        b: Second tensor or scalar (denominator)
    
    Returns:
        Result of a / b
    
    Example:
        >>> a = Tensor([10, 20, 30])
        >>> b = Tensor([2, 4, 5])
        >>> result = div(a, b)
        >>> print(result)
        Tensor([5. 5. 6.])
    """
    b = _ensure_tensor(b)
    return Tensor(a.data / b.data)


def rdiv(a: Tensor, b: Numeric) -> Tensor:
    """Reverse division (scalar / tensor).
    
    Args:
        a: Tensor (denominator)
        b: Scalar or tensor (numerator)
    
    Returns:
        Result of b / a
    
    Example:
        >>> t = Tensor([2, 4, 5])
        >>> result = rdiv(t, 20)  # 20 / t
        >>> print(result)
        Tensor([10.  5.  4.])
    """
    b = _ensure_tensor(b)
    return Tensor(b.data / a.data)


def pow(a: Tensor, exponent: float) -> Tensor:
    """Raise tensor to a power.
    
    Args:
        a: Base tensor
        exponent: Power to raise to
    
    Returns:
        Result of a ** exponent
    
    Example:
        >>> t = Tensor([2, 3, 4])
        >>> result = pow(t, 2)
        >>> print(result)
        Tensor([ 4.  9. 16.])
    """
    return Tensor(a.data ** exponent)


def neg(a: Tensor) -> Tensor:
    """Negate tensor (unary minus).
    
    Args:
        a: Input tensor
    
    Returns:
        Result of -a
    
    Example:
        >>> t = Tensor([1, -2, 3])
        >>> result = neg(t)
        >>> print(result)
        Tensor([-1.  2. -3.])
    """
    return Tensor(-a.data)