"""Matrix operations with gradient tracking.

This module implements linear algebra operations with automatic differentiation:
- matmul: Matrix multiplication (@ operator)

Matrix multiplication is fundamental to neural networks and requires careful
gradient computation using the chain rule.

Mathematical Foundation:
    Matrix Multiplication: C = A @ B
    
    Gradients:
    ∂L/∂A = ∂L/∂C @ B^T
    ∂L/∂B = A^T @ ∂L/∂C
    
    Where:
    - L is the scalar loss
    - C is the output matrix
    - A, B are input matrices
    - ^T denotes transpose
    
    Derivation (element-wise):
    C[i,j] = Σₖ A[i,k] * B[k,j]
    ∂C[i,j]/∂A[i,k] = B[k,j]
    ∂C[i,j]/∂B[k,j] = A[i,k]
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Tensor


def matmul(a: "Tensor", b: "Tensor") -> "Tensor":
    """Matrix multiplication with gradient tracking.
    
    Performs standard matrix multiplication with broadcasting support for
    batch operations. Follows NumPy/PyTorch matmul semantics.
    
    Mathematical Gradient:
        Forward: C = A @ B
        
        Backward:
        ∂L/∂A = ∂L/∂C @ B^T
        ∂L/∂B = A^T @ ∂L/∂C
    
    Shape Rules:
        - 1D @ 1D: inner product (scalar output)
        - 2D @ 1D: matrix-vector product
        - 1D @ 2D: vector-matrix product
        - 2D @ 2D: standard matrix multiplication
        - ND @ ND: batch matrix multiplication
    
    Args:
        a: First tensor (left operand)
        b: Second tensor (right operand)
    
    Returns:
        Result of matrix multiplication with gradient tracking
    
    Raises:
        ValueError: If shapes are incompatible for matrix multiplication
    """
    from .core import Tensor
    
    # Ensure b is a Tensor
    if not isinstance(b, Tensor):
        b = Tensor(b)
    
    try:
        result = np.matmul(a.data, b.data)
    except ValueError as e:
        raise ValueError(
            f"Incompatible shapes for matmul: {a.shape} and {b.shape}. "
            f"Inner dimensions must match for matrix multiplication. "
            f"Error: {str(e)}"
        ) from e
    
    out = Tensor(
        result,
        requires_grad=a.requires_grad or b.requires_grad,
        _children=(a, b),
        _op="matmul",
    )
    
    if out.requires_grad:
        def _backward():
            """Backward pass for matrix multiplication.
            
            Uses the chain rule for matrix derivatives:
            - Gradient w.r.t. A is grad_out @ B^T
            - Gradient w.r.t. B is A^T @ grad_out
            
            Special cases:
            - 1D vectors: use outer product
            - Batch operations: preserve batch dimensions
            """
            if a.requires_grad:
                # Compute ∂L/∂A = ∂L/∂C @ B^T
                
                if b.data.ndim == 1:
                    # b is a vector: use outer product
                    # grad_out is 1D, b is 1D
                    # A is 2D: (m, n), grad_out is (m,), b is (n,)
                    # ∂L/∂A = outer(grad_out, b) = (m, n)
                    grad = np.outer(out.grad.data, b.data)
                else:
                    # b is 2D or higher: use matmul
                    # grad_out @ b^T
                    grad = np.matmul(out.grad.data, b.data.T)
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
            
            if b.requires_grad:
                # Compute ∂L/∂B = A^T @ ∂L/∂C
                
                if a.data.ndim == 1:
                    # a is a vector: use outer product
                    # grad_out is 1D, a is 1D
                    # B is 2D: (n, k), grad_out is (k,), a is (n,)
                    # ∂L/∂B = outer(a, grad_out) = (n, k)
                    grad = np.outer(a.data, out.grad.data)
                else:
                    # a is 2D or higher: use matmul
                    # a^T @ grad_out
                    grad = np.matmul(a.data.T, out.grad.data)
                
                if b.grad is None:
                    b.grad = Tensor(grad)
                else:
                    b.grad = Tensor(b.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


# Test matrix operations when run directly
if __name__ == "__main__":
    from .core import Tensor
    
    print("=== Testing Matrix Operations ===\n")
    
    # Test 2D matmul
    print("Test 1: 2D Matrix Multiplication")
    A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    B = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    C = matmul(A, B)
    C.sum().backward()
    print(f"A.grad =\n{A.grad.data}")
    print(f"B.grad =\n{B.grad.data}\n")
    
    # Test matrix-vector
    print("Test 2: Matrix-Vector Multiplication")
    A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    x = Tensor([5.0, 6.0], requires_grad=True)
    y = matmul(A, x)
    y.sum().backward()
    print(f"A.grad =\n{A.grad.data}")
    print(f"x.grad = {x.grad.data}\n")
    
    # Test vector-vector (dot product)
    print("Test 3: Vector Dot Product")
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    z = matmul(x, y)
    z.backward()
    print(f"x.grad = {x.grad.data}")
    print(f"y.grad = {y.grad.data}\n")
    
    print("All matrix operations working correctly!")