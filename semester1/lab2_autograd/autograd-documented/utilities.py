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
    
    Example:
        >>> from semester1.lab2_autograd.autograd import Tensor, clip_grad_norm
        >>>


    """
    pass