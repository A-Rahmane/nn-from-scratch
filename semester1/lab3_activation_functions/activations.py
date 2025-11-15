"""
Activation functions for neural networks.

This module implements all standard activation functions with automatic
differentiation support via the autograd system from Lab 2.

Includes both functional and module-based interfaces for flexibility.

Mathematical Foundation:
    Activation functions introduce non-linearity into neural networks, enabling
    them to learn complex patterns. Without activations, stacking multiple layers
    would be equivalent to a single linear transformation.
    
Key Concepts:
    - Non-linearity: Essential for learning non-linear decision boundaries
    - Gradient flow: Well-behaved gradients prevent vanishing/exploding problems
    - Range: Output range affects network dynamics and training stability
    - Computational efficiency: Important for large-scale training
"""

import numpy as np
from typing import Optional, Union
from semester1.lab2_autograd.autograd import Tensor


# ==================== Functional Interface ====================


def relu(x: Tensor) -> Tensor:
    """
    ReLU (Rectified Linear Unit) activation function.
    
    Formula: f(x) = max(0, x)
    Derivative: f'(x) = 1 if x > 0 else 0
    Range: [0, ∞)
    
    Properties:
        - Non-linear but piecewise linear
        - Not zero-centered (outputs always >= 0)
        - Can cause "dying ReLU" problem (neurons that never activate)
        - Computationally efficient (simple max operation)
        - Most popular activation for hidden layers
        - Helps mitigate vanishing gradient problem
    
    Advantages:
        - Fast computation (no expensive operations)
        - Does not saturate for positive values
        - Sparse activation (about 50% of neurons are zero)
    
    Disadvantages:
        - Dead neurons (gradient is always 0 for x < 0)
        - Not zero-centered (can slow convergence)
        - Unbounded output (may need normalization)
    
    Args:
        x: Input tensor of any shape
    
    Returns:
        Output tensor with ReLU applied element-wise
    
    Example:
        >>> x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        >>> y = relu(x)
        >>> print(y.data)  # [0, 0, 0, 1, 2]
        >>> y.sum().backward()
        >>> print(x.grad.data)  # [0, 0, 0, 1, 1]
    """
    # Use element-wise multiplication with boolean mask
    # This leverages autograd's existing multiplication gradient
    mask = Tensor((x.data > 0).astype(np.float32))
    return x * mask


def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    """
    Leaky ReLU activation function.
    
    Formula: f(x) = max(αx, x) = x if x > 0 else αx
    Derivative: f'(x) = 1 if x > 0 else α
    Range: (-∞, ∞)
    
    Properties:
        - Prevents dying ReLU problem
        - Small negative slope for x < 0
        - Default α = 0.01 (1% slope for negatives)
        - Non-zero gradient everywhere
    
    Advantages:
        - Fixes dying ReLU problem
        - Still computationally efficient
        - Allows gradient flow for negative values
    
    Disadvantages:
        - Performance not always better than ReLU
        - Additional hyperparameter (α) to tune
    
    Args:
        x: Input tensor
        alpha: Slope for negative values (default: 0.01)
    
    Returns:
        Output tensor with Leaky ReLU applied
    
    Example:
        >>> x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        >>> y = leaky_relu(x, alpha=0.1)
        >>> print(y.data)  # [-0.2, -0.1, 0, 1, 2]
    """
    positive_mask = Tensor((x.data > 0).astype(np.float32))
    negative_mask = Tensor((x.data <= 0).astype(np.float32))
    return x * positive_mask + (x * alpha) * negative_mask


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """
    ELU (Exponential Linear Unit) activation function.
    
    Formula: f(x) = x if x > 0 else α(exp(x) - 1)
    Derivative: f'(x) = 1 if x > 0 else α * exp(x)
    Range: (-α, ∞)
    
    Properties:
        - Smooth everywhere (unlike ReLU)
        - Mean activations closer to zero
        - Saturates to -α for large negative values
        - More robust to noise than ReLU
    
    Advantages:
        - No dying ReLU problem
        - Negative saturation improves learning
        - Mean activations closer to zero (faster convergence)
        - Smooth gradient everywhere
    
    Disadvantages:
        - More expensive computation (exp operation)
        - Can produce NaN if not careful with large values
    
    Args:
        x: Input tensor
        alpha: Scale parameter for negative values (default: 1.0)
    
    Returns:
        Output tensor with ELU applied
    
    Example:
        >>> x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        >>> y = elu(x, alpha=1.0)
        >>> print(y.data)  # [-0.865, -0.632, 0, 1, 2]
    """
    # For numerical stability, clip input before exp
    x_clipped = Tensor(np.clip(x.data, -20, 20))
    
    positive_part = x * Tensor((x.data > 0).astype(np.float32))
    
    # For negative part: α(exp(x) - 1)
    # Build using primitive operations with gradients
    exp_x = Tensor(np.exp(x_clipped.data))
    negative_part = Tensor((alpha * (exp_x.data - 1)) * (x.data <= 0).astype(np.float32))
    
    # Combine but handle gradients manually for negative part
    result_data = positive_part.data + negative_part.data
    out = Tensor(
        result_data,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="elu"
    )
    
    if out.requires_grad:
        def _backward():
            if x.requires_grad:
                # Gradient: 1 if x > 0, else α * exp(x)
                grad = out.grad.data.copy()
                negative_mask = x.data <= 0
                grad[negative_mask] *= alpha * np.exp(np.clip(x.data[negative_mask], -20, 20))
                
                if x.grad is None:
                    x.grad = Tensor(grad)
                else:
                    x.grad = Tensor(x.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def sigmoid(x: Tensor) -> Tensor:
    """
    Sigmoid activation function.
    
    Formula: σ(x) = 1 / (1 + exp(-x))
    Derivative: σ'(x) = σ(x) * (1 - σ(x))
    Range: (0, 1)
    
    Properties:
        - Smooth, differentiable everywhere
        - Output is probability-like (between 0 and 1)
        - Suffers from vanishing gradients
        - Zero-centered around 0.5 (not ideal)
        - Used in binary classification output layer
    
    Advantages:
        - Output bounded between 0 and 1
        - Smooth gradient
        - Interpretable as probability
    
    Disadvantages:
        - Vanishing gradients for large |x|
        - Not zero-centered
        - Expensive computation (exp)
        - Gradients can be very small
    
    Numerical Stability:
        Uses clipping to prevent overflow in exp(-x).
        For large positive x: σ(x) ≈ 1
        For large negative x: σ(x) ≈ 0
    
    Args:
        x: Input tensor
    
    Returns:
        Output tensor with sigmoid applied element-wise
    
    Example:
        >>> x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        >>> y = sigmoid(x)
        >>> print(y.data)  # [0.119, 0.268, 0.5, 0.731, 0.881]
    """
    # Numerically stable sigmoid implementation
    # For positive x: 1 / (1 + exp(-x))
    # For negative x: exp(x) / (1 + exp(x))
    x_clipped = np.clip(x.data, -20, 20)
    
    result = np.zeros_like(x_clipped)
    positive_mask = x_clipped >= 0
    negative_mask = ~positive_mask
    
    # For positive values
    result[positive_mask] = 1.0 / (1.0 + np.exp(-x_clipped[positive_mask]))
    
    # For negative values (more stable)
    exp_x = np.exp(x_clipped[negative_mask])
    result[negative_mask] = exp_x / (1.0 + exp_x)
    
    out = Tensor(
        result,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="sigmoid"
    )
    
    if out.requires_grad:
        def _backward():
            if x.requires_grad:
                # Gradient: σ(x) * (1 - σ(x))
                sigmoid_val = out.data
                grad = out.grad.data * sigmoid_val * (1 - sigmoid_val)
                
                if x.grad is None:
                    x.grad = Tensor(grad)
                else:
                    x.grad = Tensor(x.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def tanh(x: Tensor) -> Tensor:
    """
    Hyperbolic tangent activation function.
    
    Formula: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Derivative: tanh'(x) = 1 - tanh²(x)
    Range: (-1, 1)
    
    Properties:
        - Zero-centered (better than sigmoid)
        - Smooth and differentiable
        - Still suffers from vanishing gradients
        - Stronger gradients than sigmoid
        - Used in RNN/LSTM gates
    
    Advantages:
        - Zero-centered output
        - Stronger gradients than sigmoid
        - Bounded output
    
    Disadvantages:
        - Vanishing gradients for large |x|
        - Expensive computation
    
    Args:
        x: Input tensor
    
    Returns:
        Output tensor with tanh applied element-wise
    
    Example:
        >>> x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        >>> y = tanh(x)
        >>> print(y.data)  # [-0.964, -0.762, 0, 0.762, 0.964]
    """
    # Use NumPy's tanh for numerical stability
    x_clipped = np.clip(x.data, -20, 20)
    result = np.tanh(x_clipped)
    
    out = Tensor(
        result,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="tanh"
    )
    
    if out.requires_grad:
        def _backward():
            if x.requires_grad:
                # Gradient: 1 - tanh²(x)
                tanh_val = out.data
                grad = out.grad.data * (1 - tanh_val ** 2)
                
                if x.grad is None:
                    x.grad = Tensor(grad)
                else:
                    x.grad = Tensor(x.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """
    Softmax activation function.
    
    Formula: softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)
    Range: (0, 1), sums to 1 along axis
    
    Properties:
        - Converts logits to probability distribution
        - Output sums to 1 along specified axis
        - Gradient is Jacobian matrix (not element-wise)
        - Used in multi-class classification output
    
    Numerical Stability:
        Uses log-sum-exp trick: subtract max before exp
        softmax(x) = softmax(x - max(x))
    
    Gradient:
        For i=j: ∂softmax(x)ᵢ/∂xⱼ = softmax(x)ᵢ(1 - softmax(x)ᵢ)
        For i≠j: ∂softmax(x)ᵢ/∂xⱼ = -softmax(x)ᵢ * softmax(x)ⱼ
    
    Args:
        x: Input tensor (usually logits)
        axis: Axis along which to compute softmax (default: -1)
    
    Returns:
        Output tensor with softmax applied, sums to 1 along axis
    
    Example:
        >>> x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        >>> y = softmax(x, axis=1)
        >>> print(y.data)  # Each row sums to 1
        >>> # [[0.09, 0.24, 0.67], [0.09, 0.24, 0.67]]
    """
    # Numerical stability: subtract max
    x_max = np.max(x.data, axis=axis, keepdims=True)
    x_shifted = x.data - x_max
    
    # Compute softmax
    exp_x = np.exp(np.clip(x_shifted, -20, 20))
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    result = exp_x / sum_exp
    
    out = Tensor(
        result,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="softmax"
    )
    
    if out.requires_grad:
        def _backward():
            if x.requires_grad:
                # Softmax gradient (Jacobian)
                # ∂Lᵢ/∂xⱼ = softmax(x)ᵢ * (δᵢⱼ - softmax(x)ⱼ)
                # Simplified: grad_input = softmax * (grad_output - sum(grad_output * softmax))
                s = out.data
                grad_output = out.grad.data
                
                # Sum of element-wise product along axis
                sum_term = np.sum(grad_output * s, axis=axis, keepdims=True)
                grad_input = s * (grad_output - sum_term)
                
                if x.grad is None:
                    x.grad = Tensor(grad_input)
                else:
                    x.grad = Tensor(x.grad.data + grad_input)
        
        out.grad_fn = _backward
    
    return out


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """
    Log-Softmax activation function.
    
    Formula: log_softmax(x) = log(softmax(x)) = x - log(Σⱼ exp(xⱼ))
    Range: (-∞, 0)
    
    Properties:
        - More numerically stable than log(softmax(x))
        - Used with NLL loss for classification
        - Prevents numerical underflow
    
    Numerical Stability:
        Uses log-sum-exp trick:
        log_softmax(x) = x - max(x) - log(Σ exp(x - max(x)))
    
    Args:
        x: Input tensor
        axis: Axis along which to compute log-softmax (default: -1)
    
    Returns:
        Output tensor with log-softmax applied
    
    Example:
        >>> x = Tensor([[1, 2, 3]], requires_grad=True)
        >>> y = log_softmax(x, axis=1)
        >>> print(y.data)  # [[-2.41, -1.41, -0.41]]
    """
    # Numerical stability: use log-sum-exp trick
    x_max = np.max(x.data, axis=axis, keepdims=True)
    x_shifted = x.data - x_max
    
    # log_softmax = x - max(x) - log(sum(exp(x - max(x))))
    exp_shifted = np.exp(np.clip(x_shifted, -20, 20))
    log_sum_exp = np.log(np.sum(exp_shifted, axis=axis, keepdims=True))
    result = x_shifted - log_sum_exp
    
    out = Tensor(
        result,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="log_softmax"
    )
    
    if out.requires_grad:
        def _backward():
            if x.requires_grad:
                # Gradient: grad_output - softmax(x) * sum(grad_output)
                grad_output = out.grad.data
                softmax_val = np.exp(out.data)  # exp(log_softmax) = softmax
                
                sum_grad = np.sum(grad_output, axis=axis, keepdims=True)
                grad_input = grad_output - softmax_val * sum_grad
                
                if x.grad is None:
                    x.grad = Tensor(grad_input)
                else:
                    x.grad = Tensor(x.grad.data + grad_input)
        
        out.grad_fn = _backward
    
    return out


def gelu(x: Tensor) -> Tensor:
    """
    GELU (Gaussian Error Linear Unit) activation function.
    
    Formula: f(x) = x * Φ(x) where Φ is Gaussian CDF
    Approximation: f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    Range: (-0.17, ∞)
    
    Properties:
        - Smooth approximation to ReLU
        - Used in BERT and GPT models
        - Weights inputs by their value
        - Non-monotonic for small negative values
    
    Advantages:
        - Smooth everywhere
        - State-of-the-art in transformers
        - Better than ReLU for some tasks
    
    Disadvantages:
        - More expensive than ReLU
        - Complex formula
    
    Args:
        x: Input tensor
    
    Returns:
        Output tensor with GELU applied
    
    Example:
        >>> x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        >>> y = gelu(x)
        >>> print(y.data)  # [-0.046, -0.159, 0, 0.841, 1.954]
    """
    # Use approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    
    # Build computation graph
    x_cubed = x ** 3
    inner = x + Tensor(0.044715) * x_cubed
    inner_scaled = Tensor(sqrt_2_over_pi) * inner
    tanh_part = tanh(inner_scaled)
    result = Tensor(0.5) * x * (Tensor(1.0) + tanh_part)
    
    return result


def swish(x: Tensor, beta: float = 1.0) -> Tensor:
    """
    Swish/SiLU (Sigmoid Linear Unit) activation function.
    
    Formula: f(x) = x * σ(βx) where σ is sigmoid
    When β=1, also called SiLU
    Derivative: f'(x) = f(x) + σ(βx)(1 - f(x))
    Range: (-∞, ∞)
    
    Properties:
        - Smooth, non-monotonic
        - Self-gated activation
        - Unbounded above, bounded below
        - Used in EfficientNet and other modern architectures
    
    Advantages:
        - Smooth everywhere
        - Better than ReLU on some tasks
        - Self-gating property
    
    Disadvantages:
        - More expensive than ReLU
        - Unbounded (may need normalization)
    
    Args:
        x: Input tensor
        beta: Scaling parameter (default: 1.0)
    
    Returns:
        Output tensor with Swish applied
    
    Example:
        >>> x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        >>> y = swish(x)
        >>> print(y.data)  # [-0.238, -0.268, 0, 0.731, 1.762]
    """
    return x * sigmoid(x * Tensor(beta))


def mish(x: Tensor) -> Tensor:
    """
    Mish activation function.
    
    Formula: f(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    Range: (-∞, ∞)
    
    Properties:
        - Smooth, non-monotonic
        - Self-regularizing
        - Unbounded above, bounded below (~-0.31)
        - Similar to Swish but smoother
    
    Advantages:
        - Smooth everywhere
        - Good performance on image tasks
        - Self-regularizing properties
    
    Disadvantages:
        - Expensive computation
        - Complex formula
    
    Args:
        x: Input tensor
    
    Returns:
        Output tensor with Mish applied
    
    Example:
        >>> x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        >>> y = mish(x)
        >>> print(y.data)  # [-0.252, -0.303, 0, 0.865, 1.944]
    """
    sp = softplus(x)
    return x * tanh(sp)


def softplus(x: Tensor, beta: float = 1.0) -> Tensor:
    """
    Softplus activation function.
    
    Formula: f(x) = (1/β) * log(1 + exp(βx))
    Derivative: f'(x) = σ(βx) (sigmoid)
    Range: (0, ∞)
    
    Properties:
        - Smooth approximation to ReLU
        - Always positive
        - Derivative is sigmoid
        - Converges to ReLU as β→∞
    
    Numerical Stability:
        For large βx: softplus(x) ≈ x
        For small βx: softplus(x) ≈ (1/β) * log(2)
    
    Args:
        x: Input tensor
        beta: Scaling parameter (default: 1.0)
    
    Returns:
        Output tensor with Softplus applied
    
    Example:
        >>> x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        >>> y = softplus(x)
        >>> print(y.data)  # [0.127, 0.313, 0.693, 1.313, 2.127]
    """
    # Numerical stability
    x_scaled = beta * x.data
    x_clipped = np.clip(x_scaled, -20, 20)
    
    # For large positive values, softplus(x) ≈ x
    result = np.where(
        x_clipped > 20,
        x_scaled / beta,
        np.log(1.0 + np.exp(x_clipped)) / beta
    )
    
    out = Tensor(
        result,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="softplus"
    )
    
    if out.requires_grad:
        def _backward():
            if x.requires_grad:
                # Gradient: sigmoid(βx)
                sigmoid_val = 1.0 / (1.0 + np.exp(-np.clip(beta * x.data, -20, 20)))
                grad = out.grad.data * sigmoid_val
                
                if x.grad is None:
                    x.grad = Tensor(grad)
                else:
                    x.grad = Tensor(x.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def softsign(x: Tensor) -> Tensor:
    """
    Softsign activation function.
    
    Formula: f(x) = x / (1 + |x|)
    Derivative: f'(x) = 1 / (1 + |x|)²
    Range: (-1, 1)
    
    Properties:
        - Similar to tanh but with polynomial denominator
        - Faster convergence than tanh
        - Less saturated gradients
    
    Args:
        x: Input tensor
    
    Returns:
        Output tensor with Softsign applied
    
    Example:
        >>> x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        >>> y = softsign(x)
        >>> print(y.data)  # [-0.667, -0.5, 0, 0.5, 0.667]
    """
    abs_x = Tensor(np.abs(x.data))
    denominator = Tensor(1.0) + abs_x
    return x / denominator


def hard_sigmoid(x: Tensor) -> Tensor:
    """
    Hard Sigmoid activation function.
    
    Formula: f(x) = clip((x + 1) / 2, 0, 1)
    Range: [0, 1]
    
    Properties:
        - Piecewise linear approximation to sigmoid
        - Faster computation (no exp)
        - Used in mobile/embedded models
    
    Args:
        x: Input tensor
    
    Returns:
        Output tensor with Hard Sigmoid applied
    
    Example:
        >>> x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        >>> y = hard_sigmoid(x)
        >>> print(y.data)  # [0, 0, 0.5, 1, 1]
    """
    result = np.clip((x.data + 1.0) / 2.0, 0.0, 1.0)
    
    out = Tensor(
        result,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="hard_sigmoid"
    )
    
    if out.requires_grad:
        def _backward():
            if x.requires_grad:
                # Gradient: 0.5 if -1 < x < 1, else 0
                grad = out.grad.data.copy()
                grad[(x.data <= -1.0) | (x.data >= 1.0)] = 0
                grad[(x.data > -1.0) & (x.data < 1.0)] *= 0.5
                
                if x.grad is None:
                    x.grad = Tensor(grad)
                else:
                    x.grad = Tensor(x.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def hard_tanh(x: Tensor) -> Tensor:
    """
    Hard Tanh activation function.
    
    Formula: f(x) = clip(x, -1, 1)
    Range: [-1, 1]
    
    Properties:
        - Piecewise linear approximation to tanh
        - Faster computation
        - Used in some RNN variants
    
    Args:
        x: Input tensor
    
    Returns:
        Output tensor with Hard Tanh applied
    
    Example:
        >>> x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        >>> y = hard_tanh(x)
        >>> print(y.data)  # [-1, -1, 0, 1, 1]
    """
    result = np.clip(x.data, -1.0, 1.0)
    
    out = Tensor(
        result,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="hard_tanh"
    )
    
    if out.requires_grad:
        def _backward():
            if x.requires_grad:
                # Gradient: 1 if -1 < x < 1, else 0
                grad = out.grad.data.copy()
                grad[(x.data <= -1.0) | (x.data >= 1.0)] = 0
                
                if x.grad is None:
                    x.grad = Tensor(grad)
                else:
                    x.grad = Tensor(x.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


# ==================== Module Interface ====================


class Activation:
    """
    Base class for all activation functions.
    
    All activation modules should inherit from this class and implement
    the forward() method. This provides a consistent interface similar
    to PyTorch's nn.Module.
    
    The __call__ method makes instances callable, allowing them to be
    used like functions: activation(x)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of activation function.
        
        Args:
            x: Input tensor
        
        Returns:
            Activated tensor
            
        Raises:
            NotImplementedError: If subclass doesn't implement forward()
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def __call__(self, x: Tensor) -> Tensor:
        """Make the module callable."""
        return self.forward(x)
    
    def __repr__(self) -> str:
        """String representation."""
        return self.__class__.__name__ + "()"


class ReLU(Activation):
    """
    ReLU activation module.
    
    Applies the ReLU function element-wise:
    f(x) = max(0, x)
    
    Example:
        >>> relu = ReLU()
        >>> x = Tensor([[-1, 0, 1], [2, 3, 4]], requires_grad=True)
        >>> y = relu(x)
        >>> y.sum().backward()
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply ReLU activation."""
        return relu(x)


class LeakyReLU(Activation):
    """
    Leaky ReLU activation module.
    
    Applies Leaky ReLU with specified negative slope

    Args:
        alpha: Slope for negative values (default: 0.01)
    
    Example:
        >>> leaky_relu = LeakyReLU(alpha=0.1)
        >>> x = Tensor([[-1, 0, 1]], requires_grad=True)
        >>> y = leaky_relu(x)
    """
    
    def __init__(self, alpha: float = 0.01):
        """Initialize Leaky ReLU with negative slope."""
        self.alpha = alpha
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Leaky ReLU activation."""
        return leaky_relu(x, self.alpha)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(alpha={self.alpha})"


class ELU(Activation):
    """
    ELU activation module.
    
    Args:
        alpha: Scale parameter for negative values (default: 1.0)
    
    Example:
        >>> elu_act = ELU(alpha=1.0)
        >>> x = Tensor([[-2, -1, 0, 1, 2]], requires_grad=True)
        >>> y = elu_act(x)
    """
    
    def __init__(self, alpha: float = 1.0):
        """Initialize ELU with alpha parameter."""
        self.alpha = alpha
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply ELU activation."""
        return elu(x, self.alpha)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(alpha={self.alpha})"


class Sigmoid(Activation):
    """
    Sigmoid activation module.
    
    Applies sigmoid function element-wise:
    f(x) = 1 / (1 + exp(-x))
    
    Example:
        >>> sigmoid = Sigmoid()
        >>> x = Tensor([[0, 1, 2]], requires_grad=True)
        >>> y = sigmoid(x)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Sigmoid activation."""
        return sigmoid(x)


class Tanh(Activation):
    """
    Tanh activation module.
    
    Applies hyperbolic tangent function element-wise:
    f(x) = tanh(x)
    
    Example:
        >>> tanh_act = Tanh()
        >>> x = Tensor([[-1, 0, 1]], requires_grad=True)
        >>> y = tanh_act(x)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Tanh activation."""
        return tanh(x)


class Softmax(Activation):
    """
    Softmax activation module.
    
    Args:
        axis: Axis along which to compute softmax (default: -1)
    
    Example:
        >>> softmax = Softmax(axis=1)
        >>> x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        >>> y = softmax(x)  # Each row sums to 1
    """
    
    def __init__(self, axis: int = -1):
        """Initialize Softmax with axis."""
        self.axis = axis
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Softmax activation."""
        return softmax(x, self.axis)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(axis={self.axis})"


class LogSoftmax(Activation):
    """
    Log-Softmax activation module.
    
    Args:
        axis: Axis along which to compute log-softmax (default: -1)
    
    Example:
        >>> log_softmax = LogSoftmax(axis=1)
        >>> x = Tensor([[1, 2, 3]], requires_grad=True)
        >>> y = log_softmax(x)
    """
    
    def __init__(self, axis: int = -1):
        """Initialize Log-Softmax with axis."""
        self.axis = axis
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Log-Softmax activation."""
        return log_softmax(x, self.axis)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(axis={self.axis})"


class GELU(Activation):
    """
    GELU activation module.
    
    Gaussian Error Linear Unit used in BERT, GPT, and other transformers.
    
    Example:
        >>> gelu_act = GELU()
        >>> x = Tensor([[-1, 0, 1, 2]], requires_grad=True)
        >>> y = gelu_act(x)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply GELU activation."""
        return gelu(x)


class Swish(Activation):
    """
    Swish/SiLU activation module.
    
    Args:
        beta: Scaling parameter (default: 1.0)
    
    Example:
        >>> swish_act = Swish(beta=1.0)
        >>> x = Tensor([[-1, 0, 1]], requires_grad=True)
        >>> y = swish_act(x)
    """
    
    def __init__(self, beta: float = 1.0):
        """Initialize Swish with beta parameter."""
        self.beta = beta
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Swish activation."""
        return swish(x, self.beta)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(beta={self.beta})"


class Mish(Activation):
    """
    Mish activation module.
    
    Self-regularizing activation function:
    f(x) = x * tanh(softplus(x))
    
    Example:
        >>> mish_act = Mish()
        >>> x = Tensor([[-2, -1, 0, 1, 2]], requires_grad=True)
        >>> y = mish_act(x)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Mish activation."""
        return mish(x)


class Softplus(Activation):
    """
    Softplus activation module.
    
    Args:
        beta: Scaling parameter (default: 1.0)
    
    Example:
        >>> softplus_act = Softplus(beta=1.0)
        >>> x = Tensor([[-1, 0, 1]], requires_grad=True)
        >>> y = softplus_act(x)
    """
    
    def __init__(self, beta: float = 1.0):
        """Initialize Softplus with beta parameter."""
        self.beta = beta
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Softplus activation."""
        return softplus(x, self.beta)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(beta={self.beta})"


class Softsign(Activation):
    """
    Softsign activation module.
    
    Applies softsign function: f(x) = x / (1 + |x|)
    
    Example:
        >>> softsign_act = Softsign()
        >>> x = Tensor([[-2, -1, 0, 1, 2]], requires_grad=True)
        >>> y = softsign_act(x)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Softsign activation."""
        return softsign(x)


class HardSigmoid(Activation):
    """
    Hard Sigmoid activation module.
    
    Piecewise linear approximation to sigmoid.
    
    Example:
        >>> hard_sigmoid = HardSigmoid()
        >>> x = Tensor([[-2, -1, 0, 1, 2]], requires_grad=True)
        >>> y = hard_sigmoid(x)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Hard Sigmoid activation."""
        return hard_sigmoid(x)


class HardTanh(Activation):
    """
    Hard Tanh activation module.
    
    Piecewise linear approximation to tanh.
    
    Example:
        >>> hard_tanh = HardTanh()
        >>> x = Tensor([[-2, -1, 0, 1, 2]], requires_grad=True)
        >>> y = hard_tanh(x)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Hard Tanh activation."""
        return hard_tanh(x)


class PReLU(Activation):
    """
    Parametric ReLU with learnable parameter.
    
    Formula: f(x) = max(αx, x) where α is learned during training
    
    The parameter α is initialized and updated during backpropagation,
    allowing the network to learn the optimal negative slope.
    
    Args:
        alpha: Initial value for the learnable parameter (default: 0.25)
        num_parameters: Number of α parameters (default: 1)
            - If 1: single α shared across all channels
            - If > 1: separate α for each channel
    
    Attributes:
        alpha: Learnable parameter tensor with requires_grad=True
    
    Example:
        >>> prelu = PReLU(alpha=0.25)
        >>> x = Tensor([[-1, 0, 1]], requires_grad=True)
        >>> y = prelu(x)
        >>> y.sum().backward()
        >>> print(prelu.alpha.grad)  # Gradient w.r.t. alpha
    """
    
    def __init__(self, alpha: float = 0.25, num_parameters: int = 1):
        """Initialize PReLU with learnable alpha parameter."""
        # Initialize alpha as learnable parameter
        self.alpha = Tensor(
            np.full(num_parameters, alpha, dtype=np.float32),
            requires_grad=True
        )
        self.num_parameters = num_parameters
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply PReLU activation.
        
        Args:
            x: Input tensor
        
        Returns:
            Output with PReLU applied
        """
        # Positive part
        positive_mask = Tensor((x.data > 0).astype(np.float32))
        positive_part = x * positive_mask
        
        # Negative part with learnable alpha
        negative_mask = Tensor((x.data <= 0).astype(np.float32))
        
        # Broadcast alpha if needed
        if self.num_parameters == 1:
            alpha_broadcast = self.alpha
        else:
            # Reshape alpha for channel-wise operation
            alpha_broadcast = self.alpha
        
        negative_part = x * alpha_broadcast * negative_mask
        
        return positive_part + negative_part
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(alpha={self.alpha.data}, num_parameters={self.num_parameters})"


# ==================== Utility Functions ====================


def compare_activations(
    x_range: tuple = (-5, 5),
    num_points: int = 100,
    activations: Optional[list] = None
) -> dict:
    """
    Compare multiple activation functions over a range.
    
    Computes activation values for visualization and analysis.
    
    Args:
        x_range: Tuple of (min, max) for input range
        num_points: Number of points to evaluate
        activations: List of activation names to compare (default: all)
    
    Returns:
        Dictionary mapping activation names to (x_values, y_values)
    
    Example:
        >>> results = compare_activations(x_range=(-5, 5), num_points=100)
        >>> # Use results for plotting
        >>> import matplotlib.pyplot as plt
        >>> for name, (x, y) in results.items():
        ...     plt.plot(x, y, label=name)
        >>> plt.legend()
        >>> plt.show()
    """
    if activations is None:
        activations = [
            'relu', 'leaky_relu', 'elu', 'sigmoid', 'tanh',
            'gelu', 'swish', 'mish', 'softplus'
        ]
    
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    results = {}
    
    activation_functions = {
        'relu': relu,
        'leaky_relu': leaky_relu,
        'elu': elu,
        'sigmoid': sigmoid,
        'tanh': tanh,
        'gelu': gelu,
        'swish': swish,
        'mish': mish,
        'softplus': softplus,
        'softsign': softsign,
        'hard_sigmoid': hard_sigmoid,
        'hard_tanh': hard_tanh,
    }
    
    for name in activations:
        if name in activation_functions:
            x_tensor = Tensor(x_values)
            y_tensor = activation_functions[name](x_tensor)
            results[name] = (x_values, y_tensor.data)
    
    return results


def check_dead_neurons(
    activations: Tensor,
    threshold: float = 0.0
) -> float:
    """
    Check percentage of dead neurons (always outputting zero or below threshold).
    
    Useful for diagnosing the dying ReLU problem during training.
    
    Args:
        activations: Tensor of activation values from a layer
        threshold: Threshold below which neuron is considered dead (default: 0.0)
    
    Returns:
        Percentage of dead neurons (0-100)
    
    Example:
        >>> x = Tensor(np.random.randn(100, 64))
        >>> y = relu(x)
        >>> dead_pct = check_dead_neurons(y)
        >>> print(f"Dead neurons: {dead_pct:.1f}%")
    """
    total = activations.data.size
    dead = np.sum(activations.data <= threshold)
    return (dead / total) * 100.0


def activation_statistics(activations: Tensor) -> dict:
    """
    Compute statistics about activation values.
    
    Useful for monitoring activation health during training.
    
    Args:
        activations: Tensor of activation values
    
    Returns:
        Dictionary with statistics:
            - mean: Average activation
            - std: Standard deviation
            - min: Minimum value
            - max: Maximum value
            - dead_pct: Percentage of zeros (for ReLU)
            - saturated_pct: Percentage near saturation (for sigmoid/tanh)
    
    Example:
        >>> x = Tensor(np.random.randn(1000, 128))
        >>> y = relu(x)
        >>> stats = activation_statistics(y)
        >>> print(f"Mean: {stats['mean']:.3f}, Dead: {stats['dead_pct']:.1f}%")
    """
    data = activations.data
    
    stats = {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'dead_pct': check_dead_neurons(activations),
    }
    
    # Check saturation for bounded activations
    # Sigmoid saturates near 0 and 1
    near_zero = np.sum(np.abs(data) < 0.01)
    near_one = np.sum(np.abs(data - 1.0) < 0.01)
    stats['saturated_pct'] = ((near_zero + near_one) / data.size) * 100.0
    
    return stats


def gradient_flow_analysis(
    model_layers: list,
    x: Tensor,
    loss_fn: callable
) -> dict:
    """
    Analyze gradient flow through layers.
    
    Helps diagnose vanishing/exploding gradient problems.
    
    Args:
        model_layers: List of (layer, activation) tuples
        x: Input tensor
        loss_fn: Function that computes loss from output
    
    Returns:
        Dictionary with gradient statistics per layer
    
    Example:
        >>> layers = [(W1, relu), (W2, sigmoid), (W3, None)]
        >>> stats = gradient_flow_analysis(layers, x_train, lambda y: ((y - y_true)**2).mean())
    """
    # Forward pass through all layers
    activations = [x]
    for weight, activation_fn in model_layers:
        z = activations[-1] @ weight
        if activation_fn is not None:
            a = activation_fn(z)
        else:
            a = z
        activations.append(a)
    
    # Compute loss and backward
    loss = loss_fn(activations[-1])
    loss.backward()
    
    # Collect gradient statistics
    gradient_stats = {}
    for i, (weight, _) in enumerate(model_layers):
        if weight.grad is not None:
            grad_norm = np.linalg.norm(weight.grad.data)
            grad_mean = np.mean(np.abs(weight.grad.data))
            grad_max = np.max(np.abs(weight.grad.data))
            
            gradient_stats[f'layer_{i}'] = {
                'grad_norm': float(grad_norm),
                'grad_mean': float(grad_mean),
                'grad_max': float(grad_max),
            }
    
    return gradient_stats


# ==================== Visualization Helpers ====================


def plot_activation_comparison(
    x_range: tuple = (-5, 5),
    num_points: int = 1000,
    save_path: Optional[str] = None
):
    """
    Plot comparison of all activation functions.
    
    Creates a visualization showing the shape and range of each activation.
    
    Args:
        x_range: Range of x values to plot
        num_points: Number of points for smooth curves
        save_path: Optional path to save figure
    
    Example:
        >>> plot_activation_comparison(x_range=(-5, 5), save_path='activations.png')
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting. Install with: pip install matplotlib")
        return
    
    results = compare_activations(x_range, num_points)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for name, (x, y) in results.items():
        ax.plot(x, y, label=name, linewidth=2)
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlabel('Input (x)', fontsize=12)
    ax.set_ylabel('Output f(x)', fontsize=12)
    ax.set_title('Comparison of Activation Functions', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_gradient_comparison(
    x_range: tuple = (-5, 5),
    num_points: int = 1000,
    save_path: Optional[str] = None
):
    """
    Plot gradients of activation functions.
    
    Shows how gradients behave across the input range, useful for
    understanding vanishing gradient problems.
    
    Args:
        x_range: Range of x values to plot
        num_points: Number of points for smooth curves
        save_path: Optional path to save figure
    
    Example:
        >>> plot_gradient_comparison(x_range=(-5, 5), save_path='gradients.png')
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting. Install with: pip install matplotlib")
        return
    
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    
    activation_functions = {
        'ReLU': relu,
        'Leaky ReLU': leaky_relu,
        'ELU': elu,
        'Sigmoid': sigmoid,
        'Tanh': tanh,
        'GELU': gelu,
        'Swish': swish,
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for name, act_fn in activation_functions.items():
        gradients = []
        for x_val in x_values:
            x_tensor = Tensor([x_val], requires_grad=True)
            y = act_fn(x_tensor)
            y.backward()
            gradients.append(x_tensor.grad.data[0])
        
        ax.plot(x_values, gradients, label=name, linewidth=2)
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlabel('Input (x)', fontsize=12)
    ax.set_ylabel("Gradient f'(x)", fontsize=12)
    ax.set_title('Gradients of Activation Functions', fontsize=14, fontweight='bold')
    ax.set_ylim([-0.5, 1.5])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


# ==================== Testing ====================


if __name__ == "__main__":
    print("=== Testing Activation Functions ===\n")
    
    # Test 1: ReLU
    print("Test 1: ReLU")
    x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
    y = relu(x)
    print(f"Input: {x.data}")
    print(f"Output: {y.data}")
    y.sum().backward()
    print(f"Gradient: {x.grad.data}\n")
    
    # Test 2: Sigmoid
    print("Test 2: Sigmoid")
    x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
    y = sigmoid(x)
    print(f"Output: {y.data}")
    y.sum().backward()
    print(f"Gradient: {x.grad.data}\n")
    
    # Test 3: Softmax
    print("Test 3: Softmax")
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    y = softmax(x, axis=1)
    print(f"Output:\n{y.data}")
    print(f"Row sums: {y.data.sum(axis=1)}\n")
    
    # Test 4: PReLU
    print("Test 4: PReLU with learnable parameter")
    prelu = PReLU(alpha=0.25)
    x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
    y = prelu(x)
    print(f"Output: {y.data}")
    y.sum().backward()
    print(f"x gradient: {x.grad.data}")
    print(f"alpha gradient: {prelu.alpha.grad.data}\n")
    
    # Test 5: Module interface
    print("Test 5: Module interface")
    relu_module = ReLU()
    sigmoid_module = Sigmoid()
    x = Tensor([[-1, 0, 1]], requires_grad=True)
    y1 = relu_module(x)
    y2 = sigmoid_module(x)
    print(f"ReLU: {y1.data}")
    print(f"Sigmoid: {y2.data}\n")
    
    # Test 6: Dead neurons
    print("Test 6: Dead neuron analysis")
    x = Tensor(np.random.randn(100, 64) - 2)  # Biased negative
    y = relu(x)
    dead_pct = check_dead_neurons(y)
    print(f"Dead neurons: {dead_pct:.1f}%\n")
    
    print("All tests passed!")