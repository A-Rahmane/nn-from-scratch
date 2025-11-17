"""Shape manipulation operations with gradient tracking.

This module implements shape operations with automatic differentiation:
- reshape: Change tensor shape without changing data
- transpose: Permute tensor dimensions
- T property: Convenient matrix/vector transpose

All operations properly handle:
- Gradient flow through shape changes
- Axis permutations
- Edge cases (1D vectors, empty tensors)

Mathematical Foundation:
    Shape operations don't change values, only their arrangement.
    Therefore, gradients simply need to be rearranged back to
    match the original input shape.
    
    Reshape: gradient.reshape(original_shape)
    Transpose: gradient.transpose(inverse_permutation)
"""

import numpy as np
from typing import Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Tensor


def reshape(a: "Tensor", *shape: Union[int, Tuple[int, ...]]) -> "Tensor":
    """Reshape tensor to new shape with gradient tracking.
    
    Returns a tensor with the same data but different shape. The new shape
    must be compatible with the original shape (same total number of elements).
    
    Mathematical Gradient:
        Reshape doesn't change values, only arrangement.
        Therefore: ∂L/∂x = ∂L/∂y.reshape(x.shape)
        
        The gradient is simply reshaped back to the input's original shape.
    
    Args:
        a: Input tensor
        *shape: New shape as individual arguments or tuple
                Can use -1 for one dimension to infer size
    
    Returns:
        Reshaped tensor with gradient tracking
    """
    from .core import Tensor
    
    # Handle both reshape(2, 3) and reshape((2, 3))
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    
    out = Tensor(
        a.data.reshape(shape),
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="reshape",
    )
    
    if out.requires_grad:
        original_shape = a.data.shape
        
        def _backward():
            """Backward pass for reshape.
            
            Simply reshape gradient back to original input shape.
            No values are changed, only the arrangement.
            """
            if a.requires_grad:
                grad = out.grad.data.reshape(original_shape)
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def transpose(a: "Tensor", *axes: int) -> "Tensor":
    """Transpose tensor dimensions with gradient tracking.
    
    Permutes the dimensions of the tensor according to the given axes.
    If no axes are given, reverses all dimensions (equivalent to .T for 2D).
    
    Mathematical Gradient:
        Transpose permutes dimensions, so gradient must be permuted back.
        If forward: y = transpose(x, axes)
        Then backward: ∂L/∂x = transpose(∂L/∂y, inverse_axes)
        
        Where inverse_axes reverses the permutation.
    
    Args:
        a: Input tensor
        *axes: Permutation of dimensions (optional)
               If not provided, reverses all dimensions
    
    Returns:
        Transposed tensor with gradient tracking
    """
    from .core import Tensor
    
    if len(axes) == 0:
        # No axes specified, use simple transpose
        out_data = a.data.T
        axes_used = None
    else:
        out_data = np.transpose(a.data, axes)
        axes_used = axes
    
    out = Tensor(
        out_data,
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="transpose",
    )
    
    if out.requires_grad:
        def _backward():
            """Backward pass for transpose.
            
            Compute inverse permutation and transpose gradient back.
            """
            if a.requires_grad:
                if axes_used is None:
                    # Simple transpose, reverse it
                    grad = out.grad.data.T
                else:
                    # Compute inverse permutation
                    # If forward permutation is (1, 2, 0), inverse is (2, 0, 1)
                    inv_axes = np.argsort(axes_used)
                    grad = np.transpose(out.grad.data, inv_axes)
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def get_T(a: "Tensor") -> "Tensor":
    """Get transpose of tensor (helper for T property).
    
    Convenience function for 2D matrix transpose. For higher dimensional
    tensors, reverses all axes.
    
    Args:
        a: Input tensor
    
    Returns:
        Transposed tensor
    """
    return transpose(a)


# Test shape operations when run directly
if __name__ == "__main__":
    from .core import Tensor
    
    print("=== Testing Shape Operations ===\n")
    
    # Test reshape
    print("Test 1: Reshape")
    x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    y = reshape(x, 2, 2)
    y.sum().backward()
    print(f"x.grad = {x.grad.data}")
    print("Expected: [1. 1. 1. 1.]\n")
    
    # Test transpose
    print("Test 2: Transpose")
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = transpose(x)
    y.sum().backward()
    print(f"x.grad =\n{x.grad.data}")
    print("Expected: [[1. 1.] [1. 1.]]\n")
    
    # Test T property
    print("Test 3: T property")
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = get_T(x)
    y.sum().backward()
    print(f"x.grad = {x.grad.data}")
    print("Expected: [[1. 1. 1.]]\n")
    
    print("All shape operations working correctly!")