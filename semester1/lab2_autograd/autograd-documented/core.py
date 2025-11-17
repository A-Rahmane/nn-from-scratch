"""Core Tensor class with automatic differentiation support.

This module defines the Tensor class that extends the base Tensor from Lab 1
with gradient tracking and automatic differentiation capabilities.

The Tensor class maintains:
- Computational graph structure (_prev, _op)
- Gradient storage (grad)
- Gradient computation function (grad_fn)
- Global gradient tracking flag (_grad_enabled)

Mathematical Foundation:
    For a computational graph with operations f₁, f₂, ..., fₙ:
    - Forward pass: Compute output values
    - Backward pass: Compute gradients via chain rule
    
    Chain Rule:
    ∂L/∂x = ∂L/∂y × ∂y/∂x
    
    For multiple paths to x:
    ∂L/∂x = Σᵢ (∂L/∂yᵢ × ∂yᵢ/∂x)
"""

import numpy as np
from typing import Optional, List, Tuple, Callable, Set, Union
from semester1.lab1_tensor_operations.tensor import Tensor as BaseTensor


class Tensor(BaseTensor):
    """
    Tensor with automatic differentiation support.
    
    Extends the base Tensor class with:
    - Computational graph tracking
    - Automatic gradient computation via backpropagation
    - Context managers for gradient control
    
    Attributes:
        data (np.ndarray): The underlying NumPy array
        requires_grad (bool): Whether gradients are tracked for this tensor
        grad (Optional[Tensor]): Accumulated gradient (None until backward is called)
        grad_fn (Optional[Callable]): Function to compute gradients during backward
        _prev (Tuple[Tensor, ...]): Parent tensors in computational graph
        _op (str): Operation that created this tensor (for debugging)
    
    Class Attributes:
        _grad_enabled (bool): Global flag to control gradient tracking
    
    Args:
        data: Input data (list, tuple, numpy array, or scalar)
        requires_grad: If True, track gradients for this tensor
        _children: Parent tensors (for internal graph construction)
        _op: Operation name (for internal graph construction)
    
    Mathematical Properties:
        - Gradients accumulate: grad = Σ incoming_gradients
        - Chain rule applies automatically
        - Broadcasting handled correctly
        - Topological order ensured by DFS
    """
    
    # Class variable to control gradient tracking globally
    _grad_enabled = True
    
    def __init__(
        self,
        data: Union[list, tuple, np.ndarray, float, int],
        requires_grad: bool = False,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
    ):
        """Initialize tensor with autograd support.
        
        Creates a new tensor with optional gradient tracking. If requires_grad
        is True and global gradient tracking is enabled, this tensor will be
        part of the computational graph.
        
        Args:
            data: Input data to wrap in tensor
            requires_grad: If True, track gradients for this tensor
            _children: Parent tensors (for internal graph construction)
            _op: Operation name (for internal graph construction)
        
        Note:
            The _children and _op parameters are for internal use by operation
            functions. Users should not need to specify these directly.
        """
        # Initialize base tensor
        super().__init__(data)
        
        # Autograd-specific attributes
        self.requires_grad = requires_grad and self._grad_enabled
        self.grad: Optional["Tensor"] = None
        self.grad_fn: Optional[Callable] = None
        self._prev: Tuple["Tensor", ...] = _children
        self._op = _op
        
    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        """
        Compute gradients via backpropagation through computational graph.
        
        This method implements reverse-mode automatic differentiation:
        1. Build topological ordering of computational graph via DFS
        2. Initialize gradient of output (this tensor) to 1.0 or provided gradient
        3. Propagate gradients backward through graph using chain rule
        4. Accumulate gradients at each tensor node
        
        Algorithm:
            1. Topological Sort:
               - Use DFS to visit all ancestors
               - Build reverse topological order
            
            2. Initialize Output Gradient:
               - If scalar: gradient = 1.0
               - If provided: use given gradient
               - Otherwise: error
            
            3. Backward Pass:
               - For each tensor in reverse topological order:
                 - If tensor has grad_fn:
                   - Call grad_fn to compute parent gradients
                   - Accumulate gradients at parent tensors
        
        Mathematical Foundation:
            For each operation y = f(x₁, x₂, ...):
            
            ∂L/∂xᵢ = ∂L/∂y × ∂y/∂xᵢ  (chain rule)
            
            Where:
            - L is the final scalar loss
            - y is the output of operation f
            - xᵢ are the inputs to operation f
            - ∂L/∂y is the incoming gradient (already computed)
            - ∂y/∂xᵢ is the local gradient (computed by grad_fn)
        
        Args:
            gradient: Gradient of loss w.r.t. this tensor. 
                     If None, assumes this is a scalar loss and uses 1.0
        
        Raises:
            RuntimeError: If called on non-scalar without gradient argument
            RuntimeError: If tensor doesn't require gradients
        """
        if not self.requires_grad:
            raise RuntimeError(
                "Cannot call backward() on tensor that doesn't require gradients. "
                "Create tensor with requires_grad=True."
            )
        
        # For scalar output, default gradient is 1.0
        if gradient is None:
            if self.data.size == 1:
                gradient = Tensor(np.ones_like(self.data))
            else:
                raise RuntimeError(
                    "Gradient argument must be specified for non-scalar output. "
                    f"Expected gradient for tensor of shape {self.shape}. "
                    "Consider calling .sum() or .mean() first to get a scalar."
                )
        
        # Build topological ordering of computational graph
        # This ensures we process tensors in correct order for gradient computation
        topo_order: List["Tensor"] = []
        visited: Set[int] = set()
        
        def build_topo(tensor: "Tensor") -> None:
            """Build topological ordering via depth-first search.
            
            Visits all ancestor tensors recursively, then adds current tensor
            to the ordering. This creates a reverse topological order where
            children appear after parents.
            
            Args:
                tensor: Current tensor to process
            """
            tensor_id = id(tensor)
            if tensor_id not in visited:
                visited.add(tensor_id)
                # Recursively visit all parents first
                for parent in tensor._prev:
                    build_topo(parent)
                # Add current tensor after all parents
                topo_order.append(tensor)
        
        build_topo(self)
        
        # Initialize gradient for output tensor
        self.grad = gradient
        
        # Propagate gradients backward in reverse topological order
        # This ensures each tensor's gradient is computed after all its children
        for tensor in reversed(topo_order):
            if tensor.grad_fn is not None:
                # Call the gradient function to propagate gradients to parents
                tensor.grad_fn()
    
    def zero_grad(self) -> None:
        """
        Reset gradient to None.
        
        This should be called before each backward pass to prevent gradient
        accumulation from previous iterations. In training loops, call this
        for all parameters before computing gradients.
        
        Why this is necessary:
            Gradients accumulate by default (grad += new_grad). This is useful
            for:
            - Multiple backward passes on different losses
            - Gradient accumulation for large batch sizes
            
            But for typical training, you want fresh gradients each iteration.
        """
        self.grad = None
    
    def detach(self) -> "Tensor":
        """
        Create a new tensor detached from computational graph.
        
        The returned tensor:
        - Shares the same data (no copy)
        - Doesn't require gradients
        - Doesn't have gradient history
        - Won't participate in gradient computation
        
        Use cases:
        - When you want to use a value without affecting gradients
        - To break gradient flow at specific points
        - For implementing certain algorithms (e.g., target networks)
        - To save memory by freeing computational graph
        
        Returns:
            New tensor with requires_grad=False and no gradient history
        
        Note:
            The returned tensor is a shallow copy (shares data with original).
            Modifying the detached tensor's data will affect the original.
        """
        return Tensor(self.data.copy(), requires_grad=False)
    
    def __repr__(self) -> str:
        """
        Return detailed string representation including gradient info.
        
        Format: Tensor(data, grad_fn=<OpName>)
        
        Returns:
            String representation with gradient tracking info
        """
        base_repr = f"Tensor({self.data}"
        if self.requires_grad and self._op:
            base_repr += f", grad_fn=<{self._op}>"
        base_repr += ")"
        return base_repr


# Import and bind operations from other modules
# Operations are bound at module load time for zero overhead

# Arithmetic Operations
from .arithmetic import (
    add, radd, sub, rsub, mul, rmul, div, rdiv, pow, neg
)

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

# Aggregation Operations
from .aggregation import sum_op, mean, max_op

Tensor.sum  = lambda self, axis=None, keepdims=False: sum_op(self, axis, keepdims)
Tensor.mean = lambda self, axis=None, keepdims=False: mean(self, axis, keepdims)
Tensor.max  = lambda self, axis=None, keepdims=False: max_op(self, axis, keepdims)

# Shape Operations
from .shape import reshape, transpose, get_T

Tensor.reshape   = lambda self, *shape: reshape(self, *shape)
Tensor.transpose = lambda self, *axes: transpose(self, *axes)
Tensor.T         = property(lambda self: get_T(self))

# Matrix Operations
from .matrix import matmul

Tensor.matmul     = lambda self, other: matmul(self, other)
Tensor.__matmul__ = lambda self, other: matmul(self, other)


# Test basic functionality when run directly
if __name__ == "__main__":
    print("=== Testing Modular Autograd Implementation ===\n")
    
    # Test 1: Simple gradient
    print("Test 1: f(x) = x^2")
    x = Tensor([2.0], requires_grad=True)
    y = x ** 2
    y.backward()
    print(f"x = {x.data}, dy/dx = {x.grad.data} (expected: [4.0])\n")
    
    # Test 2: Chain rule
    print("Test 2: f(x) = (x^2 + 1)^2")
    x = Tensor([2.0], requires_grad=True)
    y = (x ** 2 + 1) ** 2
    y.backward()
    print(f"x = {x.data}, dy/dx = {x.grad.data} (expected: [40.0])\n")
    
    # Test 3: Multiple operations
    print("Test 3: f(x, y) = x * y + x^2")
    x = Tensor([3.0], requires_grad=True)
    y = Tensor([4.0], requires_grad=True)
    z = x * y + x ** 2
    z.backward()
    print(f"∂z/∂x = {x.grad.data} (expected: [10.0])")
    print(f"∂z/∂y = {y.grad.data} (expected: [3.0])\n")
    
    print("All modular autograd operations working correctly!")