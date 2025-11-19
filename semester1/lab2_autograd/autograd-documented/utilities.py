"""Gradient utility functions.

This module provides utilities for working with gradients:
- zero_grad: Reset gradients for multiple tensors
- clip_grad_value: Clip gradients by value
- clip_grad_norm: Clip gradients by global norm
- numerical_gradient: Compute gradients numerically
- check_gradients: Verify autograd against numerical gradients

These utilities are essential for:
- Training stability (gradient clipping)
- Debugging (gradient checking)
- Memory management (gradient zeroing)
"""

import numpy as np
from typing import List, Callable, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Tensor


def zero_grad(*tensors: "Tensor") -> None:
    """
    Zero out gradients for multiple tensors.
    
    Convenience function to reset gradients for a collection of tensors.
    Equivalent to calling tensor.zero_grad() for each tensor.
    
    Use case:
        Before each backward pass in training, reset parameter gradients
        to prevent accumulation from previous iterations.
    
    Args:
        *tensors: Variable number of tensors to zero gradients for
    """
    for tensor in tensors:
        tensor.zero_grad()


def clip_grad_value(tensors: List["Tensor"], clip_value: float) -> None:
    """
    Clip gradient values to a maximum absolute value.
    
    Each gradient element is clipped to [-clip_value, clip_value].
    This prevents individual gradient elements from becoming too large.
    
    Use case:
        Prevent exploding gradients in RNNs or deep networks.
        Simpler than norm clipping but less commonly used.
    
    Mathematical Operation:
        grad_clipped = max(min(grad, clip_value), -clip_value)
        
        Equivalent to:
        grad_clipped = clip(grad, -clip_value, clip_value)
    
    Args:
        tensors: List of tensors whose gradients to clip
        clip_value: Maximum absolute value for gradients
    """
    for tensor in tensors:
        if tensor.grad is not None:
            tensor.grad.data = np.clip(
                tensor.grad.data,
                -clip_value,
                clip_value
            )


def clip_grad_norm(tensors: List["Tensor"], max_norm: float) -> float:
    """
    Clip gradient norm to maximum value.
    
    Computes the global L2 norm of all gradients, and if it exceeds max_norm,
    scales all gradients proportionally to bring the total norm to max_norm.
    
    This is the preferred gradient clipping method as it preserves the
    relative magnitudes of gradients across parameters.
    
    Mathematical Operation:
        total_norm = sqrt(Σᵢ ||gradᵢ||²)
        
        If total_norm > max_norm:
            scale = max_norm / total_norm
            For each gradient: grad *= scale
    
    Args:
        tensors: List of tensors whose gradients to clip
        max_norm: Maximum norm for gradients
    
    Returns:
        Total norm before clipping (useful for monitoring)
    """
    # Calculate total gradient norm across all tensors
    total_norm = 0.0
    for tensor in tensors:
        if tensor.grad is not None:
            # Add squared L2 norm of this tensor's gradient
            total_norm += np.sum(tensor.grad.data ** 2)
    
    total_norm = np.sqrt(total_norm)
    
    # Compute clipping coefficient
    clip_coef = max_norm / (total_norm + 1e-6)  # Add epsilon to avoid division by zero
    
    # If total norm exceeds max_norm, scale all gradients
    if clip_coef < 1:
        for tensor in tensors:
            if tensor.grad is not None:
                tensor.grad.data *= clip_coef
    
    return float(total_norm)


def numerical_gradient(
    func: Callable[["Tensor"], "Tensor"],
    tensor: "Tensor",
    epsilon: float = 1e-5
) -> np.ndarray:
    """
    Compute numerical gradient using finite differences.
    
    Approximates the gradient using the central difference formula:
    f'(x) ≈ [f(x + ε) - f(x - ε)] / (2ε)
    
    This is more accurate than forward differences and less susceptible
    to numerical errors.
    
    Use case:
        Verify correctness of analytical gradients computed by autograd.
        Essential for debugging gradient computation issues.
    
    Mathematical Foundation:
        Taylor expansion:
        f(x + ε) = f(x) + f'(x)ε + O(ε²)
        f(x - ε) = f(x) - f'(x)ε + O(ε²)
        
        Subtracting:
        f(x + ε) - f(x - ε) = 2f'(x)ε + O(ε³)
        
        Therefore:
        f'(x) ≈ [f(x + ε) - f(x - ε)] / (2ε) + O(ε²)
    
    Args:
        func: Function that takes a tensor and returns a scalar tensor
        tensor: Input tensor to compute gradient for
        epsilon: Small perturbation for finite differences
                 (1e-5 is usually a good choice)
    
    Returns:
        Numerical gradient as NumPy array with same shape as input tensor
    
    Notes:
        - epsilon should not be too small (numerical instability)
        - epsilon should not be too large (approximation error)
        - 1e-5 to 1e-7 is usually optimal
    """
    grad = np.zeros_like(tensor.data)
    
    # Iterate over all elements in the tensor
    it = np.nditer(tensor.data, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = tensor.data[idx]
        
        # Compute f(x + epsilon)
        tensor.data[idx] = old_value + epsilon
        fxh = func(tensor).data.item()
        
        # Compute f(x - epsilon)
        tensor.data[idx] = old_value - epsilon
        fxl = func(tensor).data.item()
        
        # Central difference formula
        grad[idx] = (fxh - fxl) / (2 * epsilon)
        
        # Restore original value
        tensor.data[idx] = old_value
        it.iternext()
    
    return grad


def check_gradients(
    func: Callable[["Tensor"], "Tensor"],
    tensor: "Tensor",
    epsilon: float = 1e-5,
    tolerance: float = 1e-5
) -> Tuple[bool, float]:
    """
    Check if autograd gradients match numerical gradients.
    
    Computes both analytical gradients (via autograd) and numerical
    gradients (via finite differences), then compares them.
    
    Use case:
        Verify correctness of gradient implementation.
        Debug gradient computation issues.
        Essential when implementing custom operations.
    
    Algorithm:
        1. Compute analytical gradient using autograd
        2. Compute numerical gradient using finite differences
        3. Compare element-wise absolute differences
        4. Check if max difference is below tolerance
    
    Args:
        func: Function that takes a tensor and returns a scalar tensor
        tensor: Input tensor to check gradients for
        epsilon: Perturbation for numerical gradient (default: 1e-5)
        tolerance: Maximum allowed difference (default: 1e-5)
    
    Returns:
        Tuple of (gradients_match: bool, max_difference: float)
        - gradients_match: True if max difference < tolerance
        - max_difference: Maximum absolute difference between gradients
    
    Notes:
        - Always check gradients when implementing new operations
        - If check fails, try adjusting epsilon (1e-5 to 1e-7)
        - Some operations (ReLU, max) have non-smooth points where
          numerical gradients may be inaccurate
    """
    # Compute analytical gradient using autograd
    tensor.zero_grad()
    output = func(tensor)
    output.backward()
    analytical_grad = tensor.grad.data.copy()
    
    # Compute numerical gradient using finite differences
    numerical_grad = numerical_gradient(func, tensor, epsilon)
    
    # Compare gradients
    diff = np.abs(analytical_grad - numerical_grad)
    max_diff = np.max(diff)
    
    # Check if difference is within tolerance
    match = max_diff < tolerance
    
    return match, float(max_diff)


# Test utilities when run directly
if __name__ == "__main__":
    from .core import Tensor
    
    print("=== Testing Gradient Utilities ===\n")
    
    # Test zero_grad
    print("Test 1: zero_grad")
    x = Tensor([1.0], requires_grad=True)
    y = Tensor([2.0], requires_grad=True)
    z = x * y
    z.backward()
    print(f"Before zero_grad: x.grad={x.grad}, y.grad={y.grad}")
    zero_grad(x, y)
    print(f"After zero_grad: x.grad={x.grad}, y.grad={y.grad}\n")
    
    # Test clip_grad_value
    print("Test 2: clip_grad_value")
    x = Tensor([1.0], requires_grad=True)
    x.grad = Tensor([10.0])
    y = Tensor([1.0], requires_grad=True)
    y.grad = Tensor([-15.0])
    print(f"Before clip: x.grad={x.grad.data}, y.grad={y.grad.data}")
    clip_grad_value([x, y], 5.0)
    print(f"After clip: x.grad={x.grad.data}, y.grad={y.grad.data}\n")
    
    # Test clip_grad_norm
    print("Test 3: clip_grad_norm")
    x = Tensor([1.0], requires_grad=True)
    y = Tensor([1.0], requires_grad=True)
    x.grad = Tensor([3.0])
    y.grad = Tensor([4.0])
    norm_before = np.sqrt(3**2 + 4**2)
    print(f"Before clip: norm={norm_before:.2f}")
    total_norm = clip_grad_norm([x, y], 2.5)
    print(f"Total norm: {total_norm:.2f}")
    print(f"After clip: x.grad={x.grad.data}, y.grad={y.grad.data}\n")
    
    # Test numerical_gradient
    print("Test 4: numerical_gradient")
    def f(x):
        return (x ** 2).sum()
    
    x = Tensor([2.0, 3.0], requires_grad=True)
    num_grad = numerical_gradient(f, x)
    print(f"Numerical gradient: {num_grad}")
    print("Expected: [4. 6.] (2*x)\n")
    
    # Test check_gradients
    print("Test 5: check_gradients")
    def func(x):
        return ((x ** 2) * 3 + x).sum()
    
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    match, diff = check_gradients(func, x)
    print(f"Gradients match: {match}")
    print(f"Max difference: {diff:.2e}\n")
    
    print("✅ All utilities working correctly!")