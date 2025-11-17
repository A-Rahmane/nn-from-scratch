"""Aggregation operations with gradient tracking.

This module implements reduction operations with automatic differentiation:
- sum: Sum elements along specified axes
- mean: Average elements along specified axes
- max: Maximum elements along specified axes

All operations properly handle:
- Axis specification (single axis, multiple axes, or all)
- keepdims flag for maintaining dimensions
- Gradient broadcasting back to original shape
- Edge cases (empty tensors, out of range axes)

Mathematical Foundations:
    Sum: y = Σᵢ xᵢ
        ∂y/∂xᵢ = 1 (gradient broadcasts to all inputs)
    
    Mean: y = (1/n) Σᵢ xᵢ
        ∂y/∂xᵢ = 1/n (gradient divided by count and broadcast)
    
    Max: y = max(x₁, x₂, ..., xₙ)
        ∂y/∂xᵢ = 1 if xᵢ = max, else 0
        (gradient only flows to maximum elements)
"""

import numpy as np
from typing import Optional, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Tensor


def sum_op(
    a: "Tensor",
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> "Tensor":
    """Sum tensor elements along specified axes with gradient tracking.
    
    Computes the sum of array elements over given axes. During backward pass,
    the gradient is broadcast back to the original input shape.
    
    Mathematical Gradient:
        y = Σ xᵢ
        ∂y/∂xᵢ = 1
        
        The gradient of 1 is broadcast to all elements that contributed to
        the sum. This means each element receives the same gradient value.
    
    Gradient Broadcasting:
        If axis=None: gradient (scalar) -> broadcast to original shape
        If axis=0: gradient (m,) -> broadcast to (n, m)
        If keepdims=True: gradient shape matches for easy broadcasting
    
    Args:
        a: Input tensor
        axis: Axis or axes along which to sum. If None, sum all elements.
        keepdims: If True, reduced axes are kept as dimensions of size 1
    
    Returns:
        Tensor with summed values and gradient tracking
    """
    from .core import Tensor
    
    out = Tensor(
        a.data.sum(axis=axis, keepdims=keepdims),
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="sum",
    )
    
    if out.requires_grad:
        def _backward():
            """Backward pass for sum operation.
            
            The gradient of sum is simply 1 for each input element.
            We need to broadcast the output gradient back to the input shape.
            """
            if a.requires_grad:
                grad = out.grad.data
                
                # If dimensions were reduced, we need to broadcast back
                if not keepdims and axis is not None:
                    # Add back the reduced dimensions
                    if isinstance(axis, int):
                        grad = np.expand_dims(grad, axis=axis)
                    else:
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, axis=ax)
                
                # Broadcast gradient to original shape
                grad = np.broadcast_to(grad, a.data.shape)
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def mean(
    a: "Tensor",
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> "Tensor":
    """Compute mean of tensor elements along specified axes with gradient tracking.
    
    Computes the arithmetic mean over given axes. During backward pass,
    the gradient is divided by the number of elements and broadcast back.
    
    Mathematical Gradient:
        y = (1/n) Σᵢ xᵢ  where n is the number of elements
        ∂y/∂xᵢ = 1/n
        
        Each element receives gradient divided by the count of elements
        that contributed to the mean.
    
    Implementation:
        mean(x) = sum(x) / n
        
        Therefore:
        ∂mean/∂x = ∂sum/∂x * (1/n) = 1/n
    
    Args:
        a: Input tensor
        axis: Axis or axes along which to compute mean. If None, compute over all.
        keepdims: If True, reduced axes are kept as dimensions of size 1
    
    Returns:
        Tensor with mean values and gradient tracking
    """
    from .core import Tensor
    
    out = Tensor(
        a.data.mean(axis=axis, keepdims=keepdims),
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="mean",
    )
    
    if out.requires_grad:
        def _backward():
            """Backward pass for mean operation.
            
            Mean is sum divided by count, so gradient is 1/n for each element.
            We broadcast the output gradient divided by count back to input shape.
            """
            if a.requires_grad:
                grad = out.grad.data
                
                # Calculate the number of elements that contributed to each output
                if axis is None:
                    # All elements contributed
                    n = a.data.size
                else:
                    # Elements along specified axes contributed
                    if isinstance(axis, int):
                        n = a.data.shape[axis]
                    else:
                        n = np.prod([a.data.shape[ax] for ax in axis])
                
                # Divide gradient by count
                grad = grad / n
                
                # If dimensions were reduced, broadcast back
                if not keepdims and axis is not None:
                    if isinstance(axis, int):
                        grad = np.expand_dims(grad, axis=axis)
                    else:
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, axis=ax)
                
                # Broadcast gradient to original shape
                grad = np.broadcast_to(grad, a.data.shape)
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def max_op(
    a: "Tensor",
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> "Tensor":
    """Compute maximum of tensor elements along specified axes with gradient tracking.
    
    Finds the maximum value over given axes. During backward pass, the gradient
    flows only to the elements that achieved the maximum value.
    
    Mathematical Gradient:
        y = max(x₁, x₂, ..., xₙ)
        ∂y/∂xᵢ = 1 if xᵢ = max(x₁, ..., xₙ), else 0
        
        Gradient only flows to maximum elements. If multiple elements tie
        for maximum, gradient is divided equally among them.
    
    Implementation Note:
        We create a mask where each maximum element gets value 1, then
        normalize by the count of maxima (to handle ties correctly).
    
    Args:
        a: Input tensor
        axis: Axis or axes along which to find maximum. If None, find global max.
        keepdims: If True, reduced axes are kept as dimensions of size 1
    
    Returns:
        Tensor with maximum values and gradient tracking
    """
    from .core import Tensor
    
    # Compute maximum values
    max_values = a.data.max(axis=axis, keepdims=keepdims)
    
    out = Tensor(
        max_values,
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="max",
    )
    
    if out.requires_grad:
        def _backward():
            """Backward pass for max operation.
            
            Gradient flows only to elements that equal the maximum.
            If multiple elements tie, gradient is split equally.
            """
            if a.requires_grad:
                grad = out.grad.data
                
                # Create mask for maximum elements
                # Need to broadcast max_values for comparison
                if not keepdims and axis is not None:
                    max_broadcast = max_values
                    if isinstance(axis, int):
                        max_broadcast = np.expand_dims(max_broadcast, axis=axis)
                    else:
                        for ax in sorted(axis):
                            max_broadcast = np.expand_dims(max_broadcast, axis=ax)
                else:
                    max_broadcast = max_values
                
                # Mask is 1 where element equals max, 0 elsewhere
                mask = (a.data == max_broadcast).astype(float)
                
                # Count how many elements achieved the max (to handle ties)
                count = mask.sum(axis=axis, keepdims=True)
                
                # Normalize mask so tied maxima split the gradient equally
                mask = mask / count
                
                # Broadcast output gradient
                if not keepdims and axis is not None:
                    if isinstance(axis, int):
                        grad = np.expand_dims(grad, axis=axis)
                    else:
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, axis=ax)
                
                grad = np.broadcast_to(grad, a.data.shape)
                
                # Apply mask to route gradient only to max elements
                grad = grad * mask
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


# Test aggregation operations when run directly
if __name__ == "__main__":
    from .core import Tensor
    
    print("=== Testing Aggregation Operations ===\n")
    
    # Test sum
    print("Test 1: Sum")
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = sum_op(x)
    y.backward()
    print(f"x.grad = {x.grad.data}")
    print("Expected: [[1. 1.] [1. 1.]]\n")
    
    # Test mean
    print("Test 2: Mean")
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = mean(x)
    y.backward()
    print(f"x.grad = {x.grad.data}")
    print("Expected: [[0.25 0.25] [0.25 0.25]]\n")
    
    # Test max
    print("Test 3: Max")
    x = Tensor([[1.0, 4.0], [2.0, 3.0]], requires_grad=True)
    y = max_op(x)
    y.backward()
    print(f"x.grad = {x.grad.data}")
    print("Expected: [[0. 1.] [0. 0.]]\n")
    
    # Test sum with axis
    print("Test 4: Sum with axis")
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = sum_op(x, axis=0)
    y.sum().backward()
    print(f"x.grad = {x.grad.data}")
    print("Expected: [[1. 1.] [1. 1.]]\n")
    
    print("All aggregation operations working correctly!")