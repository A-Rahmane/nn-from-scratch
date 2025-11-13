"""
Automatic differentiation module.

This module extends the Tensor class from Lab 1 with automatic gradient
computation capabilities, enabling backpropagation through computational graphs.

Key Concepts:
    - Computational Graph: DAG tracking operations between tensors
    - Gradient Functions: Store how to compute gradients for each operation
    - Chain Rule: Automatically propagate gradients backward through the graph
    - Topological Sort: Ensure correct ordering of gradient computations

Mathematical Foundation:
    For a function y = f(x), the gradient ∂L/∂x is computed via chain rule:
    ∂L/∂x = ∂L/∂y × ∂y/∂x
    
    Where:
    - L is the final scalar loss
    - y is the output of operation f
    - x is the input to operation f
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
    
    Args:
        data: Input data (list, tuple, numpy array, or scalar)
        requires_grad: Whether to track gradients for this tensor
        _children: Parent tensors that created this tensor (internal use)
        _op: Operation that created this tensor (internal use)
        
    Attributes:
        data: The underlying NumPy array
        requires_grad: Whether gradients are tracked
        grad: Accumulated gradient (None until backward is called)
        grad_fn: Function to compute gradients during backward pass
        _prev: Set of parent tensors in computational graph
        _op: String describing the operation that created this tensor
    
    Example:
        >>> x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> y = x * 2
        >>> z = y.sum()
        >>> z.backward()
        >>> print(x.grad)  # Tensor([2., 2., 2.])
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
        
        Args:
            data: Input data to wrap in tensor
            requires_grad: If True, track gradients for this tensor
            _children: Parent tensors (for internal graph construction)
            _op: Operation name (for internal graph construction)
        """
        # Initialize base tensor
        super().__init__(data)
        
        # Autograd-specific attributes
        self.requires_grad = requires_grad and self._grad_enabled
        self.grad: Optional["Tensor"] = None
        self.grad_fn: Optional[Callable] = None
        self._prev: Set["Tensor"] = set(_children)
        self._op = _op
        
    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        """
        Compute gradients via backpropagation through computational graph.
        
        This method implements reverse-mode automatic differentiation:
        1. Build topological ordering of computational graph
        2. Initialize gradient of output (this tensor) to 1.0
        3. Propagate gradients backward using chain rule
        
        Mathematical Foundation:
            For each operation y = f(x₁, x₂, ...):
            ∂L/∂xᵢ = ∂L/∂y × ∂y/∂xᵢ
            
        Args:
            gradient: Gradient of loss w.r.t. this tensor. 
                     If None, assumes this is a scalar loss and uses 1.0
                     
        Raises:
            RuntimeError: If called on non-scalar without gradient argument
            RuntimeError: If tensor doesn't require gradients
            
        Example:
            >>> x = Tensor([2.0], requires_grad=True)
            >>> y = x ** 2
            >>> y.backward()
            >>> print(x.grad)  # Tensor([4.0]) because dy/dx = 2x = 4
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
                    f"Expected gradient for tensor of shape {self.shape}"
                )
        
        # Build topological ordering of computational graph
        topo_order: List["Tensor"] = []
        visited: Set["Tensor"] = set()
        
        def build_topo(tensor: "Tensor") -> None:
            """Build topological ordering via depth-first search."""
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._prev:
                    build_topo(parent)
                topo_order.append(tensor)
        
        build_topo(self)
        
        # Initialize gradient for output tensor
        self.grad = gradient
        
        # Propagate gradients backward in reverse topological order
        for tensor in reversed(topo_order):
            if tensor.grad_fn is not None:
                tensor.grad_fn()
    
    def zero_grad(self) -> None:
        """
        Reset gradient to None.
        
        Call this before each backward pass to prevent gradient accumulation
        from previous iterations.
        
        Example:
            >>> x = Tensor([1.0], requires_grad=True)
            >>> y = x * 2
            >>> y.backward()
            >>> print(x.grad)  # Tensor([2.0])
            >>> x.zero_grad()
            >>> print(x.grad)  # None
        """
        self.grad = None
    
    def detach(self) -> "Tensor":
        """
        Create a new tensor detached from computational graph.
        
        The returned tensor shares the same data but doesn't track gradients.
        Useful when you want to use a value without affecting gradients.
        
        Returns:
            New tensor with requires_grad=False and no gradient history
            
        Example:
            >>> x = Tensor([1.0], requires_grad=True)
            >>> y = x * 2
            >>> z = y.detach()  # z shares data with y but no gradients
            >>> w = z * 3       # This operation won't be in x's graph
        """
        return Tensor(self.data.copy(), requires_grad=False)
    
    # ==================== Arithmetic Operations with Gradients ====================
    
    def __add__(self, other: Union["Tensor", np.ndarray, float, int]) -> "Tensor":
        """
        Add tensors with gradient tracking.
        
        Gradient computation:
            y = a + b
            ∂y/∂a = 1 (gradient flows unchanged)
            ∂y/∂b = 1 (gradient flows unchanged)
            
        Broadcasting is handled automatically by NumPy and gradient
        is reduced appropriately during backward pass.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="add",
        )
        
        if out.requires_grad:
            def _backward():
                """Backward pass for addition.
                
                Addition distributes gradients equally to both inputs.
                Handle broadcasting by summing gradients over broadcast dimensions.
                """
                if self.requires_grad:
                    # Sum out any dimensions that were broadcast
                    grad = out.grad.data
                    
                    # Handle broadcasting: reduce gradient to match input shape
                    ndims_added = grad.ndim - self.data.ndim
                    for _ in range(ndims_added):
                        grad = grad.sum(axis=0)
                    
                    # Sum over dimensions that were broadcast
                    for i, (dim, grad_dim) in enumerate(zip(self.data.shape, grad.shape)):
                        if dim == 1 and grad_dim > 1:
                            grad = grad.sum(axis=i, keepdims=True)
                    
                    if self.grad is None:
                        self.grad = Tensor(grad)
                    else:
                        self.grad = Tensor(self.grad.data + grad)
                
                if other.requires_grad:
                    grad = out.grad.data
                    
                    # Handle broadcasting for other
                    ndims_added = grad.ndim - other.data.ndim
                    for _ in range(ndims_added):
                        grad = grad.sum(axis=0)
                    
                    for i, (dim, grad_dim) in enumerate(zip(other.data.shape, grad.shape)):
                        if dim == 1 and grad_dim > 1:
                            grad = grad.sum(axis=i, keepdims=True)
                    
                    if other.grad is None:
                        other.grad = Tensor(grad)
                    else:
                        other.grad = Tensor(other.grad.data + grad)
            
            out.grad_fn = _backward
        
        return out
    
    def __mul__(self, other: Union["Tensor", np.ndarray, float, int]) -> "Tensor":
        """
        Multiply tensors with gradient tracking.
        
        Gradient computation:
            y = a * b
            ∂y/∂a = b (multiply incoming gradient by other input)
            ∂y/∂b = a (multiply incoming gradient by other input)
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="mul",
        )
        
        if out.requires_grad:
            def _backward():
                """Backward pass for multiplication.
                
                Product rule: gradient of each input is incoming gradient
                times the other input value.
                """
                if self.requires_grad:
                    grad = out.grad.data * other.data
                    
                    # Handle broadcasting
                    ndims_added = grad.ndim - self.data.ndim
                    for _ in range(ndims_added):
                        grad = grad.sum(axis=0)
                    
                    for i, (dim, grad_dim) in enumerate(zip(self.data.shape, grad.shape)):
                        if dim == 1 and grad_dim > 1:
                            grad = grad.sum(axis=i, keepdims=True)
                    
                    if self.grad is None:
                        self.grad = Tensor(grad)
                    else:
                        self.grad = Tensor(self.grad.data + grad)
                
                if other.requires_grad:
                    grad = out.grad.data * self.data
                    
                    # Handle broadcasting
                    ndims_added = grad.ndim - other.data.ndim
                    for _ in range(ndims_added):
                        grad = grad.sum(axis=0)
                    
                    for i, (dim, grad_dim) in enumerate(zip(other.data.shape, grad.shape)):
                        if dim == 1 and grad_dim > 1:
                            grad = grad.sum(axis=i, keepdims=True)
                    
                    if other.grad is None:
                        other.grad = Tensor(grad)
                    else:
                        other.grad = Tensor(other.grad.data + grad)
            
            out.grad_fn = _backward
        
        return out
    
    def __pow__(self, power: Union[float, int]) -> "Tensor":
        """
        Raise tensor to a power with gradient tracking.
        
        Gradient computation:
            y = x^n
            ∂y/∂x = n * x^(n-1) (power rule from calculus)
        """
        out = Tensor(
            self.data ** power,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"pow({power})",
        )
        
        if out.requires_grad:
            def _backward():
                """Backward pass for power operation.
                
                Uses power rule: d/dx(x^n) = n*x^(n-1)
                """
                if self.requires_grad:
                    grad = out.grad.data * (power * self.data ** (power - 1))
                    
                    if self.grad is None:
                        self.grad = Tensor(grad)
                    else:
                        self.grad = Tensor(self.grad.data + grad)
            
            out.grad_fn = _backward
        
        return out
    
    def __neg__(self) -> "Tensor":
        """
        Negate tensor with gradient tracking.
        
        Gradient computation:
            y = -x
            ∂y/∂x = -1 (gradient is negated)
        """
        return self * (-1)
    
    def __sub__(self, other: Union["Tensor", np.ndarray, float, int]) -> "Tensor":
        """
        Subtract tensors with gradient tracking.
        
        Gradient computation:
            y = a - b
            ∂y/∂a = 1
            ∂y/∂b = -1
        """
        return self + (-other if isinstance(other, Tensor) else Tensor(-other))
    
    def __truediv__(self, other: Union["Tensor", np.ndarray, float, int]) -> "Tensor":
        """
        Divide tensors with gradient tracking.
        
        Gradient computation:
            y = a / b
            ∂y/∂a = 1/b
            ∂y/∂b = -a/b²
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)
    
    def __radd__(self, other: Union[np.ndarray, float, int]) -> "Tensor":
        """Right addition: other + self"""
        return self + other
    
    def __rmul__(self, other: Union[np.ndarray, float, int]) -> "Tensor":
        """Right multiplication: other * self"""
        return self * other
    
    def __rsub__(self, other: Union[np.ndarray, float, int]) -> "Tensor":
        """Right subtraction: other - self"""
        return Tensor(other) + (-self)
    
    def __rtruediv__(self, other: Union[np.ndarray, float, int]) -> "Tensor":
        """Right division: other / self"""
        return Tensor(other) * (self ** -1)
    
    # ==================== Aggregation Operations with Gradients ====================
    
    def sum(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> "Tensor":
        """
        Sum tensor elements with gradient tracking.
        
        Gradient computation:
            y = sum(x)
            ∂y/∂x_i = 1 for all i (gradient broadcasts back to original shape)
            
        The gradient of sum is broadcasted back to the input shape,
        as each input contributes equally (+1) to the output.
        """
        out = Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )
        
        if out.requires_grad:
            def _backward():
                """Backward pass for sum.
                
                Sum operation broadcasts gradient of 1 to all inputs.
                Must handle keepdims and axis parameters correctly.
                """
                if self.requires_grad:
                    grad = out.grad.data
                    
                    # If keepdims=False, we need to add back dimensions
                    if not keepdims and axis is not None:
                        if isinstance(axis, int):
                            grad = np.expand_dims(grad, axis=axis)
                        else:
                            for ax in sorted(axis):
                                grad = np.expand_dims(grad, axis=ax)
                    
                    # Broadcast gradient to match input shape
                    grad = np.broadcast_to(grad, self.data.shape)
                    
                    if self.grad is None:
                        self.grad = Tensor(grad)
                    else:
                        self.grad = Tensor(self.grad.data + grad)
            
            out.grad_fn = _backward
        
        return out
    
    def mean(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> "Tensor":
        """
        Mean of tensor elements with gradient tracking.
        
        Gradient computation:
            y = mean(x) = sum(x) / n
            ∂y/∂x_i = 1/n for all i
            
        where n is the number of elements being averaged.
        """
        out = Tensor(
            np.mean(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="mean",
        )
        
        if out.requires_grad:
            def _backward():
                """Backward pass for mean.
                
                Mean distributes gradient equally across all inputs,
                divided by the count of elements.
                """
                if self.requires_grad:
                    grad = out.grad.data
                    
                    # Calculate number of elements that were averaged
                    if axis is None:
                        n = self.data.size
                    elif isinstance(axis, int):
                        n = self.data.shape[axis]
                    else:
                        n = np.prod([self.data.shape[ax] for ax in axis])
                    
                    # Distribute gradient
                    grad = grad / n
                    
                    # Handle keepdims
                    if not keepdims and axis is not None:
                        if isinstance(axis, int):
                            grad = np.expand_dims(grad, axis=axis)
                        else:
                            for ax in sorted(axis):
                                grad = np.expand_dims(grad, axis=ax)
                    
                    # Broadcast to input shape
                    grad = np.broadcast_to(grad, self.data.shape)
                    
                    if self.grad is None:
                        self.grad = Tensor(grad)
                    else:
                        self.grad = Tensor(self.grad.data + grad)
            
            out.grad_fn = _backward
        
        return out
    
    def max(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
    ) -> "Tensor":
        """
        Maximum of tensor elements with gradient tracking.
        
        Gradient computation:
            y = max(x)
            ∂y/∂x_i = 1 if x_i == max(x), else 0
            
        Gradient flows only to the maximum element(s).
        """
        out_data = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="max",
        )
        
        if out.requires_grad:
            def _backward():
                """Backward pass for max.
                
                Gradient flows only to maximum elements.
                Creates a mask where max elements are 1, others are 0.
                """
                if self.requires_grad:
                    grad = out.grad.data
                    
                    # Expand dimensions if keepdims=False
                    out_expanded = out_data
                    if not keepdims and axis is not None:
                        if isinstance(axis, int):
                            out_expanded = np.expand_dims(out_expanded, axis=axis)
                        else:
                            for ax in sorted(axis):
                                out_expanded = np.expand_dims(out_expanded, axis=ax)
                        grad = np.expand_dims(grad, axis=axis) if isinstance(axis, int) else grad
                        for ax in sorted(axis) if not isinstance(axis, int) else []:
                            grad = np.expand_dims(grad, axis=ax)
                    
                    # Create mask for maximum values
                    mask = (self.data == out_expanded).astype(np.float32)
                    
                    # Distribute gradient among all maximum values
                    count = np.sum(mask, axis=axis, keepdims=True) if axis is not None else np.sum(mask)
                    grad_input = mask * np.broadcast_to(grad, self.data.shape) / count
                    
                    if self.grad is None:
                        self.grad = Tensor(grad_input)
                    else:
                        self.grad = Tensor(self.grad.data + grad_input)
            
            out.grad_fn = _backward
        
        return out
    
    # ==================== Shape Operations with Gradients ====================
    
    def reshape(self, *shape: int) -> "Tensor":
        """
        Reshape tensor with gradient tracking.
        
        Gradient computation:
            y = reshape(x, new_shape)
            ∂y/∂x: reshape gradient back to original shape
            
        Reshaping doesn't change values, so gradient just needs to be
        reshaped back to match input.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        
        out = Tensor(
            self.data.reshape(shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )
        
        if out.requires_grad:
            original_shape = self.data.shape
            
            def _backward():
                """Backward pass for reshape.
                
                Simply reshape gradient back to input shape.
                """
                if self.requires_grad:
                    grad = out.grad.data.reshape(original_shape)
                    
                    if self.grad is None:
                        self.grad = Tensor(grad)
                    else:
                        self.grad = Tensor(self.grad.data + grad)
            
            out.grad_fn = _backward
        
        return out
    
    def transpose(self, *axes: int) -> "Tensor":
        """
        Transpose tensor with gradient tracking.
        
        Gradient computation:
            y = transpose(x, axes)
            ∂y/∂x: transpose gradient with inverse permutation
        """
        if len(axes) == 0:
            out_data = self.data.T
            axes_used = None
        else:
            out_data = np.transpose(self.data, axes)
            axes_used = axes
        
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="transpose",
        )
        
        if out.requires_grad:
            def _backward():
                """Backward pass for transpose.
                
                Transpose gradient with inverse permutation to match input.
                """
                if self.requires_grad:
                    if axes_used is None:
                        # Simple transpose, reverse it
                        grad = out.grad.data.T
                    else:
                        # Compute inverse permutation
                        inv_axes = np.argsort(axes_used)
                        grad = np.transpose(out.grad.data, inv_axes)
                    
                    if self.grad is None:
                        self.grad = Tensor(grad)
                    else:
                        self.grad = Tensor(self.grad.data + grad)
            
            out.grad_fn = _backward
        
        return out
    
    @property
    def T(self) -> "Tensor":
        """Transpose property with gradient tracking."""
        return self.transpose()
    
    # ==================== Matrix Operations with Gradients ====================
    
    def matmul(self, other: "Tensor") -> "Tensor":
        """
        Matrix multiplication with gradient tracking.
        
        Gradient computation:
            C = A @ B
            ∂C/∂A = ∂L/∂C @ B^T
            ∂C/∂B = A^T @ ∂L/∂C
            
        These are the standard matrix multiplication gradient formulas.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        try:
            out = Tensor(
                np.matmul(self.data, other.data),
                requires_grad=self.requires_grad or other.requires_grad,
                _children=(self, other),
                _op="matmul",
            )
        except ValueError as e:
            raise ValueError(
                f"Incompatible shapes for matmul: {self.shape} and {other.shape}"
            ) from e
        
        if out.requires_grad:
            def _backward():
                """Backward pass for matrix multiplication.
                
                Uses the chain rule for matrix derivatives:
                - Gradient w.r.t. A is grad @ B^T
                - Gradient w.r.t. B is A^T @ grad
                """
                if self.requires_grad:
                    # ∂L/∂A = ∂L/∂C @ B^T
                    if other.data.ndim == 1:
                        # Vector case
                        grad = np.outer(out.grad.data, other.data)
                    else:
                        grad = np.matmul(out.grad.data, other.data.T)
                    
                    if self.grad is None:
                        self.grad = Tensor(grad)
                    else:
                        self.grad = Tensor(self.grad.data + grad)
                
                if other.requires_grad:
                    # ∂L/∂B = A^T @ ∂L/∂C
                    if self.data.ndim == 1:
                        # Vector case
                        grad = np.outer(self.data, out.grad.data)
                    else:
                        grad = np.matmul(self.data.T, out.grad.data)
                    
                    if other.grad is None:
                        other.grad = Tensor(grad)
                    else:
                        other.grad = Tensor(other.grad.data + grad)
            
            out.grad_fn = _backward
        
        return out
    
    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication operator @ with gradients."""
        return self.matmul(other)


# ==================== Context Managers ====================


class no_grad:
    """
    Context manager to disable gradient tracking.
    
    Useful for inference or when performing operations that shouldn't
    be part of the computational graph (e.g., parameter updates).
    
    Example:
        >>> x = Tensor([1.0], requires_grad=True)
        >>> with no_grad():
        ...     y = x * 2  # This operation won't track gradients
        ...     print(y.requires_grad)  # False
    """
    
    def __enter__(self):
        """Enter no_grad context."""
        self.prev = Tensor._grad_enabled
        Tensor._grad_enabled = False
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit no_grad context."""
        Tensor._grad_enabled = self.prev
        return False


class enable_grad:
    """
    Context manager to enable gradient tracking.
    
    Useful when you want to temporarily enable gradients within a no_grad block.
    
    Example:
        >>> with no_grad():
        ...     with enable_grad():
        ...         x = Tensor([1.0], requires_grad=True)
        ...         y = x * 2  # This WILL track gradients
    """
    
    def __enter__(self):
        """Enter enable_grad context."""
        self.prev = Tensor._grad_enabled
        Tensor._grad_enabled = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit enable_grad context."""
        Tensor._grad_enabled = self.prev
        return False


# ==================== Gradient Utilities ====================


def zero_grad(*tensors: Tensor) -> None:
    """
    Zero out gradients for multiple tensors.
    
    Convenience function to reset gradients for a collection of tensors.
    
    Args:
        *tensors: Variable number of tensors to zero gradients for
        
    Example:
        >>> x = Tensor([1.0], requires_grad=True)
        >>> y = Tensor([2.0], requires_grad=True)
        >>> z = x * y
        >>> z.backward()
        >>> zero_grad(x, y)  # Reset both gradients
    """
    for tensor in tensors:
        tensor.zero_grad()


def clip_grad_value(tensors: List[Tensor], clip_value: float) -> None:
    """
    Clip gradient values to a maximum absolute value.
    
    Helps prevent exploding gradients during training.
    
    Args:
        tensors: List of tensors whose gradients to clip
        clip_value: Maximum absolute value for gradients
        
    Example:
        >>> parameters = [weight, bias]
        >>> clip_grad_value(parameters, 1.0)  # Clip to [-1, 1]
    """
    for tensor in tensors:
        if tensor.grad is not None:
            tensor.grad.data = np.clip(tensor.grad.data, -clip_value, clip_value)


def clip_grad_norm(tensors: List[Tensor], max_norm: float) -> float:
    """
    Clip gradient norm to maximum value.
    
    Scales all gradients proportionally if their total norm exceeds max_norm.
    This is preferred over value clipping for most applications.
    
    Args:
        tensors: List of tensors whose gradients to clip
        max_norm: Maximum norm for gradients
        
    Returns:
        Total norm of gradients before clipping
        
    Example:
        >>> parameters = [weight, bias]
        >>> total_norm = clip_grad_norm(parameters, 1.0)
    """
    # Calculate total norm
    total_norm = 0.0
    for tensor in tensors:
        if tensor.grad is not None:
            total_norm += np.sum(tensor.grad.data ** 2)
    total_norm = np.sqrt(total_norm)
    
    # Clip if necessary
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for tensor in tensors:
            if tensor.grad is not None:
                tensor.grad.data *= clip_coef
    
    return total_norm


def numerical_gradient(
    func: Callable[[Tensor], Tensor],
    tensor: Tensor,
    epsilon: float = 1e-5
) -> np.ndarray:
    """
    Compute numerical gradient using finite differences.
    
    Used for gradient checking to verify autograd implementation.
    
    Mathematical Foundation:
        f'(x) ≈ [f(x + ε) - f(x - ε)] / (2ε)
        
    This is the central difference formula, which is more accurate
        than forward differences and less susceptible to numerical errors.
    
    Args:
        func: Function that takes a tensor and returns a scalar tensor
        tensor: Input tensor to compute gradient for
        epsilon: Small perturbation for finite differences
        
    Returns:
        Numerical gradient as NumPy array
        
    Example:
        >>> def f(x):
        ...     return (x ** 2).sum()
        >>> x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> num_grad = numerical_gradient(f, x)
        >>> # Compare with autograd
        >>> y = f(x)
        >>> y.backward()
        >>> print(np.allclose(num_grad, x.grad.data))  # Should be True
    """
    grad = np.zeros_like(tensor.data)
    
    # Compute gradient for each element
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
        
        # Central difference
        grad[idx] = (fxh - fxl) / (2 * epsilon)
        
        # Restore original value
        tensor.data[idx] = old_value
        it.iternext()
    
    return grad


def check_gradients(
    func: Callable[[Tensor], Tensor],
    tensor: Tensor,
    epsilon: float = 1e-5,
    tolerance: float = 1e-5
) -> Tuple[bool, float]:
    """
    Check if autograd gradients match numerical gradients.
    
    Computes both automatic and numerical gradients and compares them.
    
    Args:
        func: Function that takes a tensor and returns a scalar tensor
        tensor: Input tensor to check gradients for
        epsilon: Perturbation for numerical gradient
        tolerance: Maximum allowed difference
        
    Returns:
        Tuple of (gradients_match, max_difference)
        
    Example:
        >>> def f(x):
        ...     return (x ** 3 + 2 * x).sum()
        >>> x = Tensor([1.0, 2.0], requires_grad=True)
        >>> match, diff = check_gradients(f, x)
        >>> print(f"Gradients match: {match}, Max diff: {diff}")
    """
    # Compute autograd gradient
    tensor.zero_grad()
    output = func(tensor)
    output.backward()
    autograd_grad = tensor.grad.data.copy()
    
    # Compute numerical gradient
    numerical_grad = numerical_gradient(func, tensor, epsilon)
    
    # Compare
    diff = np.abs(autograd_grad - numerical_grad)
    max_diff = np.max(diff)
    match = max_diff < tolerance
    
    return match, max_diff


# ==================== Visualization Utilities ====================


def visualize_computation_graph(
    tensor: Tensor,
    format: str = "png",
    filename: str = "computation_graph"
) -> None:
    """
    Visualize the computational graph (requires graphviz).
    
    Creates a visual representation of the computational graph leading
    to the given tensor.
    
    Args:
        tensor: Output tensor to visualize graph for
        format: Output format ('png', 'pdf', 'svg')
        filename: Output filename (without extension)
        
    Note:
        Requires graphviz package: pip install graphviz
        
    Example:
        >>> x = Tensor([1.0], requires_grad=True)
        >>> y = x ** 2
        >>> z = y + 3
        >>> visualize_computation_graph(z)  # Creates computation_graph.png
    """
    try:
        from graphviz import Digraph
    except ImportError:
        print("graphviz package required. Install with: pip install graphviz")
        return
    
    dot = Digraph(format=format)
    dot.attr(rankdir='LR')  # Left to right layout
    
    nodes = set()
    edges = set()
    
    def build_graph(tensor: Tensor):
        """Recursively build graph structure."""
        if tensor not in nodes:
            nodes.add(tensor)
            
            # Add node for tensor
            label = f"{tensor._op}\nshape: {tensor.shape}"
            if tensor.requires_grad:
                label += f"\ngrad: {tensor.grad is not None}"
            
            dot.node(str(id(tensor)), label, shape='box')
            
            # Add parent nodes and edges
            for parent in tensor._prev:
                if parent not in nodes:
                    build_graph(parent)
                edge = (str(id(parent)), str(id(tensor)))
                if edge not in edges:
                    edges.add(edge)
                    dot.edge(*edge)
    
    build_graph(tensor)
    dot.render(filename, cleanup=True)
    print(f"Computation graph saved to {filename}.{format}")


# Test basic functionality
if __name__ == "__main__":
    print("=== Testing Autograd ===\n")
    
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
    
    # Test 4: Sum operation
    print("Test 4: f(x) = sum(x)")
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.sum()
    y.backward()
    print(f"∂y/∂x =\n{x.grad.data}\n(expected: all ones)\n")
    
    # Test 5: Matrix multiplication
    print("Test 5: Matrix multiplication")
    A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    B = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    C = A @ B
    loss = C.sum()
    loss.backward()
    print(f"∂L/∂A =\n{A.grad.data}")
    print(f"∂L/∂B =\n{B.grad.data}\n")
    
    # Test 6: Numerical gradient checking
    print("Test 6: Gradient checking")
    def func(x):
        return ((x ** 2) * 3 + x).sum()
    
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    match, diff = check_gradients(func, x)
    print(f"Gradients match: {match}, Max difference: {diff:.2e}\n")
    
    # Test 7: no_grad context
    print("Test 7: no_grad context")
    x = Tensor([1.0], requires_grad=True)
    with no_grad():
        y = x * 2
        print(f"y.requires_grad = {y.requires_grad} (expected: False)\n")
    
    print("All basic tests passed!")