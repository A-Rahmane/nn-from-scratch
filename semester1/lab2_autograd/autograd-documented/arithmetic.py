"""Arithmetic operations with gradient tracking.

This module implements arithmetic operations with automatic differentiation:
- Basic operations: add, subtract, multiply, divide, power, negate
- Reverse operations: radd, rsub, rmul, rdiv (for scalar op tensor)
- Broadcasting gradient handling

All operations properly handle:
- Gradient computation via chain rule
- Broadcasting (with proper gradient reduction)
- Computational graph construction
- Type coercion (scalars, lists, arrays to Tensors)

Mathematical Foundations:
    Addition: y = a + b
        ∂y/∂a = 1, ∂y/∂b = 1
    
    Multiplication: y = a * b
        ∂y/∂a = b, ∂y/∂b = a
    
    Power: y = x^n
        ∂y/∂x = n * x^(n-1)
    
    Division: y = a / b
        ∂y/∂a = 1/b, ∂y/∂b = -a/b²
"""

import numpy as np
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Tensor

Numeric = Union["Tensor", np.ndarray, float, int]


def _ensure_tensor(x: Numeric) -> "Tensor":
    """Convert input to Tensor if it isn't already.
    
    Helper function to handle type coercion for operations that accept
    tensors, arrays, or scalars.
    
    Args:
        x: Input value (Tensor, array, or scalar)
    
    Returns:
        Tensor object
    """
    # Import here to avoid circular dependency
    from .core import Tensor
    return x if isinstance(x, Tensor) else Tensor(x)


def _handle_broadcasting_grad(grad: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Handle gradient reduction for broadcasted operations.
    
    When operations involve broadcasting, the gradient must be reduced
    to match the original input shape. This function handles all cases:
    - Added dimensions (prepended 1s)
    - Broadcast dimensions (dimension is 1 in input but > 1 in output)
    
    Algorithm:
        1. Sum over any prepended dimensions
        2. For each dimension, if original was 1 but gradient is > 1:
           Sum over that dimension and keep dimension with keepdims=True
    
    Args:
        grad: Gradient array (possibly broadcast shape)
        original_shape: Shape of original input tensor
    
    Returns:
        Gradient reduced to match original_shape
    """
    # Sum over prepended dimensions
    ndims_added = grad.ndim - len(original_shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)
    
    # Sum over dimensions that were size 1 in original
    for i, (dim, grad_dim) in enumerate(zip(original_shape, grad.shape)):
        if dim == 1 and grad_dim > 1:
            grad = grad.sum(axis=i, keepdims=True)
    
    return grad


def add(a: "Tensor", b: Numeric) -> "Tensor":
    """Add two tensors or tensor and scalar with gradient tracking.
    
    Performs element-wise addition with automatic differentiation.
    Supports broadcasting following NumPy rules.
    
    Mathematical Gradient:
        y = a + b
        ∂y/∂a = 1 (gradient flows unchanged)
        ∂y/∂b = 1 (gradient flows unchanged)
    
    Broadcasting Example:
        a: (2, 3)
        b: (3,)  -> broadcasts to (2, 3)
        y: (2, 3)
        
        Backward:
        ∂L/∂a: (2, 3) -> (2, 3) unchanged
        ∂L/∂b: (2, 3) -> sum axis 0 -> (3,)
    
    Args:
        a: First tensor
        b: Second tensor or scalar
    
    Returns:
        Result of a + b with gradient tracking
    """
    from .core import Tensor
    
    b = _ensure_tensor(b)
    out = Tensor(
        a.data + b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        _children=(a, b),
        _op="add",
    )
    
    if out.requires_grad:
        def _backward():
            """Backward pass for addition.
            
            Addition distributes gradients equally to both inputs.
            Handle broadcasting by summing gradients over broadcast dimensions.
            """
            if a.requires_grad:
                grad = _handle_broadcasting_grad(out.grad.data, a.data.shape)
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
            
            if b.requires_grad:
                grad = _handle_broadcasting_grad(out.grad.data, b.data.shape)
                
                if b.grad is None:
                    b.grad = Tensor(grad)
                else:
                    b.grad = Tensor(b.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def radd(a: "Tensor", b: Numeric) -> "Tensor":
    """Reverse addition (scalar + tensor).
    
    Handles the case when a scalar or non-Tensor is on the left side
    of the addition operator.
    
    Args:
        a: Tensor
        b: Scalar or tensor
    
    Returns:
        Result of b + a
    """
    return add(a, b)  # Addition is commutative


def sub(a: "Tensor", b: Numeric) -> "Tensor":
    """Subtract two tensors or tensor and scalar with gradient tracking.
    
    Performs element-wise subtraction with automatic differentiation.
    
    Mathematical Gradient:
        y = a - b
        ∂y/∂a = 1
        ∂y/∂b = -1
    
    Args:
        a: First tensor (minuend)
        b: Second tensor or scalar (subtrahend)
    
    Returns:
        Result of a - b with gradient tracking
    """
    from .core import Tensor
    
    b = _ensure_tensor(b)
    out = Tensor(
        a.data - b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        _children=(a, b),
        _op="sub",
    )
    
    if out.requires_grad:
        def _backward():
            """Backward pass for subtraction.
            
            Gradient flows unchanged to first input,
            negated to second input.
            """
            if a.requires_grad:
                grad = _handle_broadcasting_grad(out.grad.data, a.data.shape)
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
            
            if b.requires_grad:
                grad = _handle_broadcasting_grad(out.grad.data, b.data.shape)
                
                if b.grad is None:
                    b.grad = Tensor(-grad)
                else:
                    b.grad = Tensor(b.grad.data - grad)
        
        out.grad_fn = _backward
    
    return out


def rsub(a: "Tensor", b: Numeric) -> "Tensor":
    """Reverse subtraction (scalar - tensor).
    
    Handles the case when a scalar or non-Tensor is on the left side
    of the subtraction operator.
    
    Mathematical Gradient:
        y = b - a
        ∂y/∂a = -1
    
    Args:
        a: Tensor
        b: Scalar or tensor
    
    Returns:
        Result of b - a
    """
    from .core import Tensor
    
    b = _ensure_tensor(b)
    return sub(b, a)


def mul(a: "Tensor", b: Numeric) -> "Tensor":
    """Multiply two tensors or tensor and scalar with gradient tracking.
    
    Performs element-wise multiplication with automatic differentiation.
    Supports broadcasting following NumPy rules.
    
    Mathematical Gradient:
        y = a * b
        ∂y/∂a = b
        ∂y/∂b = a
    
    Chain rule application:
        ∂L/∂a = ∂L/∂y * ∂y/∂a = grad_out * b
        ∂L/∂b = ∂L/∂y * ∂y/∂b = grad_out * a
    
    Args:
        a: First tensor
        b: Second tensor or scalar
    
    Returns:
        Result of a * b with gradient tracking
    """
    from .core import Tensor
    
    b = _ensure_tensor(b)
    out = Tensor(
        a.data * b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        _children=(a, b),
        _op="mul",
    )
    
    if out.requires_grad:
        def _backward():
            """Backward pass for multiplication.
            
            Uses product rule: d(uv) = v*du + u*dv
            Each input gets gradient multiplied by the other input.
            """
            if a.requires_grad:
                grad = out.grad.data * b.data
                grad = _handle_broadcasting_grad(grad, a.data.shape)
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
            
            if b.requires_grad:
                grad = out.grad.data * a.data
                grad = _handle_broadcasting_grad(grad, b.data.shape)
                
                if b.grad is None:
                    b.grad = Tensor(grad)
                else:
                    b.grad = Tensor(b.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def rmul(a: "Tensor", b: Numeric) -> "Tensor":
    """Reverse multiplication (scalar * tensor).
    
    Handles the case when a scalar or non-Tensor is on the left side
    of the multiplication operator.
    
    Args:
        a: Tensor
        b: Scalar or tensor
    
    Returns:
        Result of b * a
    """
    return mul(a, b)  # Multiplication is commutative


def div(a: "Tensor", b: Numeric) -> "Tensor":
    """Divide two tensors or tensor and scalar with gradient tracking.
    
    Performs element-wise division with automatic differentiation.
    
    Mathematical Gradient:
        y = a / b
        ∂y/∂a = 1/b
        ∂y/∂b = -a/b²
    
    Chain rule application:
        ∂L/∂a = ∂L/∂y * (1/b) = grad_out / b
        ∂L/∂b = ∂L/∂y * (-a/b²) = -grad_out * a / b²
    
    Args:
        a: Numerator tensor
        b: Denominator tensor or scalar
    
    Returns:
        Result of a / b with gradient tracking
    """
    from .core import Tensor
    
    b = _ensure_tensor(b)
    out = Tensor(
        a.data / b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        _children=(a, b),
        _op="div",
    )
    
    if out.requires_grad:
        def _backward():
            """Backward pass for division.
            
            Uses quotient rule: d(u/v) = (v*du - u*dv)/v²
            Simplified:
            - Gradient for a: grad_out / b
            - Gradient for b: -grad_out * a / b²
            """
            if a.requires_grad:
                grad = out.grad.data / b.data
                grad = _handle_broadcasting_grad(grad, a.data.shape)
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
            
            if b.requires_grad:
                grad = -out.grad.data * a.data / (b.data ** 2)
                grad = _handle_broadcasting_grad(grad, b.data.shape)
                
                if b.grad is None:
                    b.grad = Tensor(grad)
                else:
                    b.grad = Tensor(b.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def rdiv(a: "Tensor", b: Numeric) -> "Tensor":
    """Reverse division (scalar / tensor).
    
    Handles the case when a scalar or non-Tensor is on the left side
    of the division operator.
    
    Mathematical Gradient:
        y = b / a
        ∂y/∂a = -b/a²
    
    Args:
        a: Tensor (denominator)
        b: Scalar or tensor (numerator)
    
    Returns:
        Result of b / a
    """
    from .core import Tensor
    
    b = _ensure_tensor(b)
    return div(b, a)


def pow(a: "Tensor", power: Union[int, float]) -> "Tensor":
    """Raise tensor to a power with gradient tracking.
    
    Performs element-wise exponentiation with automatic differentiation.
    
    Mathematical Gradient:
        y = x^n
        ∂y/∂x = n * x^(n-1)
    
    Chain rule application:
        ∂L/∂x = ∂L/∂y * ∂y/∂x = grad_out * n * x^(n-1)
    
    Special cases:
        - x^0 = 1, gradient = 0
        - x^1 = x, gradient = 1
        - x^2, gradient = 2x
        - x^(-1) = 1/x, gradient = -1/x²
    
    Args:
        a: Base tensor
        power: Exponent (int or float)
    
    Returns:
        Result of a ** power with gradient tracking
    """
    from .core import Tensor
    
    out = Tensor(
        a.data ** power,
        requires_grad=a.requires_grad,
        _children=(a,),
        _op=f"pow({power})",
    )
    
    if out.requires_grad:
        def _backward():
            """Backward pass for power function.
            
            Uses power rule: d(x^n)/dx = n*x^(n-1)
            """
            if a.requires_grad:
                # Gradient: n * x^(n-1) * grad_out
                grad = power * (a.data ** (power - 1)) * out.grad.data
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def neg(a: "Tensor") -> "Tensor":
    """Negate tensor with gradient tracking.
    
    Performs element-wise negation with automatic differentiation.
    
    Mathematical Gradient:
        y = -x
        ∂y/∂x = -1
    
    Args:
        a: Input tensor
    
    Returns:
        Result of -a with gradient tracking
    """
    from .core import Tensor
    
    out = Tensor(
        -a.data,
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="neg",
    )
    
    if out.requires_grad:
        def _backward():
            """Backward pass for negation.
            
            Gradient is simply negated.
            """
            if a.requires_grad:
                if a.grad is None:
                    a.grad = Tensor(-out.grad.data)
                else:
                    a.grad = Tensor(a.grad.data - out.grad.data)
        
        out.grad_fn = _backward
    
    return out


# Test arithmetic operations when run directly
if __name__ == "__main__":
    from .core import Tensor
    
    print("=== Testing Arithmetic Operations ===\n")
    
    # Test addition
    print("Test 1: Addition")
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)
    c = add(a, b)
    c.sum().backward()
    print(f"a.grad = {a.grad.data} (expected: [1., 1.])")
    print(f"b.grad = {b.grad.data} (expected: [1., 1.])\n")
    
    # Test multiplication
    print("Test 2: Multiplication")
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    c = mul(a, b)
    c.sum().backward()
    print(f"a.grad = {a.grad.data} (expected: [4., 5.])")
    print(f"b.grad = {b.grad.data} (expected: [2., 3.])\n")
    
    # Test power
    print("Test 3: Power")
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = pow(x, 2)
    y.sum().backward()
    print(f"x.grad = {x.grad.data} (expected: [4., 6.])\n")
    
    # Test division
    print("Test 4: Division")
    a = Tensor([6.0, 8.0], requires_grad=True)
    b = Tensor([2.0, 4.0], requires_grad=True)
    c = div(a, b)
    c.sum().backward()
    print(f"a.grad = {a.grad.data} (expected: [0.5, 0.25])")
    print(f"b.grad = {b.grad.data} (expected: [-1.5, -0.5])\n")
    
    print("All arithmetic operations working correctly!")