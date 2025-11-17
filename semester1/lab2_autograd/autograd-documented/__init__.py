"""Autograd module - Modular implementation of automatic differentiation.

This module provides automatic differentiation capabilities, extending the 
Tensor class from Lab 1 with computational graph tracking and gradient computation.

Modules:
- core.py: Core Tensor class with autograd support
- arithmetic.py: Arithmetic operations with gradient tracking
- aggregation.py: Aggregation operations (sum, mean, max) with gradients
- shape.py: Shape manipulation operations with gradients
- matrix.py: Matrix operations with gradients
- context.py: Context managers for gradient control
- utilities.py: Gradient utilities (clipping, checking, etc.)
- visualization.py: Computational graph visualization

Architecture:
    The Tensor class extends the base Tensor from Lab 1 with:
    - Computational graph tracking (_prev, _op, grad_fn)
    - Automatic gradient computation via backpropagation
    - Context managers for gradient control
    - Gradient utilities for training

Key Concepts:
    - Computational Graph: DAG of operations between tensors
    - Chain Rule: Automatic gradient propagation
    - Topological Sort: Correct ordering of gradient computation
    - Gradient Accumulation: Gradients sum across multiple paths
"""

from .core import Tensor
from .context import no_grad, enable_grad
from .utilities import (
    zero_grad,
    clip_grad_value,
    clip_grad_norm,
    numerical_gradient,
    check_gradients,
)

# Optional: only import if graphviz is available
try:
    from .visualization import visualize_computation_graph
    __all__ = [
        'Tensor',
        'no_grad',
        'enable_grad',
        'zero_grad',
        'clip_grad_value',
        'clip_grad_norm',
        'numerical_gradient',
        'check_gradients',
        'visualize_computation_graph',
    ]
except ImportError:
    __all__ = [
        'Tensor',
        'no_grad',
        'enable_grad',
        'zero_grad',
        'clip_grad_value',
        'clip_grad_norm',
        'numerical_gradient',
        'check_gradients',
    ]

__version__ = '1.0.0'
__author__ = 'MENOUER Abderrahmane'

# Module-level convenience functions
def tensor(data, requires_grad=False):
    """Create a tensor with optional gradient tracking.
    
    Args:
        data: Input data (list, tuple, numpy array, or scalar)
        requires_grad: Whether to track gradients
    
    Returns:
        Tensor object with autograd support
    """
    return Tensor(data, requires_grad=requires_grad)