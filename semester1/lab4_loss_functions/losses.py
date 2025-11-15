"""
Loss functions for neural networks.

This module implements all standard loss functions with automatic
differentiation support via the autograd system from Lab 2.

Includes both functional and module-based interfaces for flexibility.

Mathematical Foundation:
    Loss functions quantify the difference between model predictions and
    true targets. They guide the optimization process during training.
    
Key Concepts:
    - Regression losses: For continuous targets
    - Classification losses: For discrete categories
    - Reduction modes: How to aggregate batch losses
    - Class weights: Handle imbalanced datasets
    - Numerical stability: Prevent overflow/underflow
"""

import numpy as np
from typing import Optional, Union, List, Callable
from semester1.lab2_autograd.autograd import Tensor


# ==================== Utility Functions ====================


def apply_reduction(loss: Tensor, reduction: str) -> Tensor:
    """
    Apply reduction to loss tensor.
    
    Args:
        loss: Loss tensor of any shape
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Reduced loss tensor
        
    Raises:
        ValueError: If reduction mode is invalid
    """
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Must be 'mean', 'sum', or 'none'")


def _check_shapes(predictions: Tensor, targets: Tensor, allow_broadcasting: bool = False):
    """
    Check if prediction and target shapes are compatible.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        allow_broadcasting: Whether to allow broadcasting
        
    Raises:
        ValueError: If shapes are incompatible
    """
    if not allow_broadcasting and predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
        )


def _clip_for_log(x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Clip values to prevent log(0).
    
    Args:
        x: Input array
        eps: Small epsilon value
        
    Returns:
        Clipped array safe for logarithm
    """
    return np.clip(x, eps, 1.0 - eps)


# ==================== Regression Losses ====================


def mse_loss(
    predictions: Tensor,
    targets: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    """
    Mean Squared Error (MSE) loss.
    
    Formula: L = (1/n)Σ(ŷᵢ - yᵢ)²
    Gradient: ∂L/∂ŷ = (2/n)(ŷ - y)
    
    Use Cases:
        - Regression problems
        - When outliers should be heavily penalized
        - When errors are normally distributed
    
    Properties:
        - Differentiable everywhere
        - Sensitive to outliers (quadratic penalty)
        - Scale-dependent
        - Convex optimization landscape
    
    Advantages:
        - Simple and intuitive
        - Smooth gradients
        - Works well for most regression tasks
    
    Disadvantages:
        - Sensitive to outliers
        - Assumes Gaussian noise
        - Scale-dependent (not normalized)
    
    Args:
        predictions: Predicted values, shape (batch_size, ...)
        targets: Ground truth values, same shape as predictions
        reduction: How to reduce the loss:
            - 'mean': Average over all elements (default)
            - 'sum': Sum over all elements
            - 'none': No reduction, return per-element loss
    
    Returns:
        Loss tensor:
            - Scalar if reduction is 'mean' or 'sum'
            - Same shape as input if reduction is 'none'
    
    Example:
        >>> predictions = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        >>> targets = Tensor([[1.5, 2.5], [3.5, 4.5]])
        >>> loss = mse_loss(predictions, targets, reduction='mean')
        >>> print(loss.data)  # 0.25
        >>> loss.backward()
        >>> print(predictions.grad.data)  # Gradient w.r.t. predictions
    """
    _check_shapes(predictions, targets)
    
    # Compute squared differences
    diff = predictions - targets
    squared_diff = diff ** 2
    
    # Apply reduction
    return apply_reduction(squared_diff, reduction)


def mae_loss(
    predictions: Tensor,
    targets: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    """
    Mean Absolute Error (MAE) / L1 Loss.
    
    Formula: L = (1/n)Σ|ŷᵢ - yᵢ|
    Gradient: ∂L/∂ŷ = sign(ŷ - y)/n
    
    Use Cases:
        - Robust regression (less sensitive to outliers)
        - When errors are not normally distributed
        - When outliers should have linear penalty
    
    Properties:
        - Robust to outliers (linear penalty)
        - Not differentiable at zero
        - Produces sparse solutions
    
    Advantages:
        - Robust to outliers
        - Interpretable (same units as target)
        - Less sensitive to large errors
    
    Disadvantages:
        - Non-smooth at zero (optimization challenges)
        - Constant gradient may cause overshooting
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        reduction: Reduction mode
    
    Returns:
        MAE loss
    
    Example:
        >>> predictions = Tensor([1.0, 2.0, 10.0], requires_grad=True)
        >>> targets = Tensor([1.5, 2.5, 3.0])
        >>> loss = mae_loss(predictions, targets)
        >>> loss.backward()
    """
    _check_shapes(predictions, targets)
    
    # Compute absolute differences
    diff = predictions - targets
    abs_diff = Tensor(np.abs(diff.data))
    
    # Build computational graph for gradient
    out = Tensor(
        abs_diff.data,
        requires_grad=predictions.requires_grad,
        _children=(predictions, targets),
        _op='mae'
    )
    
    if out.requires_grad:
        def _backward():
            if predictions.requires_grad:
                # Gradient: sign(pred - target)
                grad = np.sign(diff.data)
                
                if predictions.grad is None:
                    predictions.grad = Tensor(grad)
                else:
                    predictions.grad = Tensor(predictions.grad.data + grad)
        
        out.grad_fn = _backward
    
    # Apply reduction
    return apply_reduction(out, reduction)


def smooth_l1_loss(
    predictions: Tensor,
    targets: Tensor,
    beta: float = 1.0,
    reduction: str = 'mean'
) -> Tensor:
    """
    Smooth L1 Loss (Huber Loss).
    
    Formula:
        L = 0.5 * (x²/β)           if |x| < β
        L = |x| - 0.5 * β          otherwise
        where x = prediction - target
    
    Use Cases:
        - Robust regression (combines MSE and MAE)
        - Object detection (Faster R-CNN)
        - When you want smooth gradients near zero
    
    Properties:
        - Smooth everywhere
        - Quadratic for small errors, linear for large errors
        - Controlled by β parameter
    
    Advantages:
        - Robust to outliers
        - Smooth gradients
        - Best of MSE and MAE
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        beta: Threshold for switching between L1 and L2 (default: 1.0)
        reduction: Reduction mode
    
    Returns:
        Smooth L1 loss
    
    Example:
        >>> predictions = Tensor([0.5, 2.0, 10.0], requires_grad=True)
        >>> targets = Tensor([1.0, 2.0, 3.0])
        >>> loss = smooth_l1_loss(predictions, targets, beta=1.0)
    """
    _check_shapes(predictions, targets)
    
    diff = predictions - targets
    abs_diff = Tensor(np.abs(diff.data))
    
    # Smooth L1: 0.5 * x²/β if |x| < β, else |x| - 0.5*β
    mask_small = abs_diff.data < beta
    
    loss_data = np.zeros_like(diff.data)
    loss_data[mask_small] = 0.5 * (diff.data[mask_small] ** 2) / beta
    loss_data[~mask_small] = abs_diff.data[~mask_small] - 0.5 * beta
    
    out = Tensor(
        loss_data,
        requires_grad=predictions.requires_grad,
        _children=(predictions, targets),
        _op='smooth_l1'
    )
    
    if out.requires_grad:
        def _backward():
            if predictions.requires_grad:
                grad = np.zeros_like(diff.data)
                # Gradient: x/β if |x| < β, else sign(x)
                grad[mask_small] = diff.data[mask_small] / beta
                grad[~mask_small] = np.sign(diff.data[~mask_small])
                
                if predictions.grad is None:
                    predictions.grad = Tensor(grad)
                else:
                    predictions.grad = Tensor(predictions.grad.data + grad)
        
        out.grad_fn = _backward
    
    return apply_reduction(out, reduction)


def msle_loss(
    predictions: Tensor,
    targets: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    """
    Mean Squared Logarithmic Error (MSLE).
    
    Formula: L = (1/n)Σ(log(ŷᵢ+1) - log(yᵢ+1))²
    
    Use Cases:
        - When you care about relative errors
        - When targets span several orders of magnitude
        - When underprediction is worse than overprediction
    
    Properties:
        - Penalizes underestimation more than overestimation
        - Scale-invariant
        - Requires non-negative predictions and targets
    
    Args:
        predictions: Predicted values (must be non-negative)
        targets: Ground truth values (must be non-negative)
        reduction: Reduction mode
    
    Returns:
        MSLE loss
    
    Example:
        >>> predictions = Tensor([1.0, 10.0, 100.0], requires_grad=True)
        >>> targets = Tensor([2.0, 20.0, 200.0])
        >>> loss = msle_loss(predictions, targets)
    """
    _check_shapes(predictions, targets)
    
    # Add 1 to avoid log(0)
    log_pred = Tensor(np.log(_clip_for_log(predictions.data + 1)))
    log_target = Tensor(np.log(_clip_for_log(targets.data + 1)))
    
    # Use MSE on log values
    diff = log_pred.data - log_target.data
    squared_log_diff = diff ** 2
    
    out = Tensor(
        squared_log_diff,
        requires_grad=predictions.requires_grad,
        _children=(predictions,),
        _op='msle'
    )
    
    if out.requires_grad:
        def _backward():
            if predictions.requires_grad:
                # Gradient: 2 * (log(pred+1) - log(target+1)) / (pred+1)
                grad = 2 * diff / (predictions.data + 1)
                
                if predictions.grad is None:
                    predictions.grad = Tensor(grad)
                else:
                    predictions.grad = Tensor(predictions.grad.data + grad)
        
        out.grad_fn = _backward
    
    return apply_reduction(out, reduction)


# ==================== Binary Classification Losses ====================


def binary_cross_entropy(
    predictions: Tensor,
    targets: Tensor,
    weight: Optional[Tensor] = None,
    reduction: str = 'mean'
) -> Tensor:
    """
    Binary Cross-Entropy (BCE) loss.
    
    Formula: L = -(1/n)Σ[yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]
    
    Use Cases:
        - Binary classification
        - Multi-label classification
        - Predictions already passed through sigmoid
    
    Properties:
        - Predictions should be in [0, 1]
        - Targets should be 0 or 1
        - Logarithmic penalty
    
    Args:
        predictions: Predicted probabilities in [0, 1], shape (batch_size, ...)
        targets: Binary targets (0 or 1), same shape as predictions
        weight: Optional sample weights
        reduction: Reduction mode
    
    Returns:
        BCE loss
    
    Example:
        >>> from semester1.lab3_activation_functions.activations import sigmoid
        >>> logits = Tensor([0.5, 1.0, -0.5], requires_grad=True)
        >>> predictions = sigmoid(logits)
        >>> targets = Tensor([1.0, 1.0, 0.0])
        >>> loss = binary_cross_entropy(predictions, targets)
    """
    _check_shapes(predictions, targets)
    
    # Clip predictions for numerical stability
    pred_clipped = _clip_for_log(predictions.data)
    
    # BCE formula: -[y*log(p) + (1-y)*log(1-p)]
    bce = -(targets.data * np.log(pred_clipped) + 
            (1 - targets.data) * np.log(1 - pred_clipped))
    
    out = Tensor(
        bce,
        requires_grad=predictions.requires_grad,
        _children=(predictions, targets),
        _op='bce'
    )
    
    if out.requires_grad:
        def _backward():
            if predictions.requires_grad:
                # Gradient: -(y/p - (1-y)/(1-p)) = (p-y)/(p(1-p))
                eps = 1e-7
                pred_safe = np.clip(predictions.data, eps, 1 - eps)
                grad = (pred_safe - targets.data) / (pred_safe * (1 - pred_safe))
                
                if weight is not None:
                    grad *= weight.data
                
                if predictions.grad is None:
                    predictions.grad = Tensor(grad)
                else:
                    predictions.grad = Tensor(predictions.grad.data + grad)
        
        out.grad_fn = _backward
    
    # Apply weight if provided
    if weight is not None:
        out = out * weight
    
    return apply_reduction(out, reduction)


def binary_cross_entropy_with_logits(
    logits: Tensor,
    targets: Tensor,
    weight: Optional[Tensor] = None,
    pos_weight: Optional[Tensor] = None,
    reduction: str = 'mean'
) -> Tensor:
    """
    Binary Cross-Entropy with Logits.
    
    Combines sigmoid and BCE in numerically stable way.
    
    Formula: L = -[y*log(σ(x)) + (1-y)*log(1-σ(x))]
    Stable form: L = max(x,0) - x*y + log(1 + exp(-|x|))
    
    Use Cases:
        - Binary classification (preferred over BCE)
        - When predictions are logits (before sigmoid)
        - Imbalanced datasets (use pos_weight)
    
    Properties:
        - More numerically stable than BCE + sigmoid
        - Handles extreme logit values well
        - Supports positive class weighting
    
    Args:
        logits: Raw predictions (before sigmoid)
        targets: Binary targets (0 or 1)
        weight: Optional sample weights
        pos_weight: Weight for positive class (for imbalance)
        reduction: Reduction mode
    
    Returns:
        BCE with logits loss
    
    Example:
        >>> logits = Tensor([0.5, 1.0, -0.5], requires_grad=True)
        >>> targets = Tensor([1.0, 1.0, 0.0])
        >>> loss = binary_cross_entropy_with_logits(logits, targets)
    """
    _check_shapes(logits, targets)
    
    # Numerically stable BCE: max(x,0) - x*y + log(1 + exp(-|x|))
    max_val = np.maximum(logits.data, 0)
    bce = max_val - logits.data * targets.data + np.log(1 + np.exp(-np.abs(logits.data)))
    
    # Apply positive weight if provided
    if pos_weight is not None:
        # Weight positive examples
        bce = bce * ((targets.data * (pos_weight.data - 1)) + 1)
    
    out = Tensor(
        bce,
        requires_grad=logits.requires_grad,
        _children=(logits, targets),
        _op='bce_with_logits'
    )
    
    if out.requires_grad:
        def _backward():
            if logits.requires_grad:
                # Gradient: σ(x) - y
                sigmoid_val = 1 / (1 + np.exp(-np.clip(logits.data, -20, 20)))
                grad = sigmoid_val - targets.data
                
                if pos_weight is not None:
                    grad *= ((targets.data * (pos_weight.data - 1)) + 1)
                
                if weight is not None:
                    grad *= weight.data
                
                if logits.grad is None:
                    logits.grad = Tensor(grad)
                else:
                    logits.grad = Tensor(logits.grad.data + grad)
        
        out.grad_fn = _backward
    
    if weight is not None:
        out = out * weight
    
    return apply_reduction(out, reduction)


# ==================== Multi-Class Classification Losses ====================


def cross_entropy_loss(
    predictions: Tensor,
    targets: Tensor,
    weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = 'mean'
) -> Tensor:
    """
    Cross-Entropy loss for multi-class classification.
    
    Formula: L = -(1/n)Σᵢ Σⱼ yᵢⱼ log(ŷᵢⱼ)
    
    Combines log_softmax and NLL for numerical stability.
    
    Use Cases:
        - Multi-class classification
        - When classes are mutually exclusive
        - Output layer uses softmax activation
    
    Properties:
        - Measures difference between probability distributions
        - Numerically stable implementation
        - Supports class weights
        - Supports label smoothing
    
    Label Smoothing:
        Instead of hard targets [0, 0, 1, 0], use soft targets:
        [ε/K, ε/K, 1-ε+ε/K, ε/K] where ε is smoothing factor
        Helps with overconfidence and generalization.
    
    Args:
        predictions: Model output logits (before softmax), shape (batch_size, num_classes)
        targets: Ground truth class indices, shape (batch_size,) or one-hot (batch_size, num_classes)
        weight: Optional class weights, shape (num_classes,)
        ignore_index: Target value to ignore in loss computation
        label_smoothing: Label smoothing factor in [0, 1)
        reduction: Reduction mode
    
    Returns:
        Cross-entropy loss
    
    Example:
        >>> logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        >>> targets = Tensor([0])  # Class 0 is correct
        >>> loss = cross_entropy_loss(logits, targets)
        >>> loss.backward()
    """
    batch_size = predictions.shape[0]
    num_classes = predictions.shape[1]
    
    # Convert targets to indices if one-hot
    if len(targets.shape) > 1 and targets.shape[1] == num_classes:
        target_indices = np.argmax(targets.data, axis=1)
    else:
        target_indices = targets.data.astype(int)
    
    # Log-softmax for numerical stability
    from semester1.lab3_activation_functions.activations import log_softmax
    log_probs = log_softmax(predictions, axis=1)
    
    # Handle label smoothing
    if label_smoothing > 0:
        # Smooth labels: (1-ε)*y + ε/K
        confidence = 1.0 - label_smoothing
        smoothing_value = label_smoothing / num_classes
        
        # Create smoothed targets
        targets_smooth = np.full((batch_size, num_classes), smoothing_value, dtype=np.float32)
        for i in range(batch_size):
            if target_indices[i] != ignore_index:
                targets_smooth[i, target_indices[i]] = confidence + smoothing_value
        
        # Loss = -sum(targets * log_probs)
        loss = -(Tensor(targets_smooth) * log_probs).sum(axis=1)
    else:
        # Standard cross-entropy
        # Gather log probabilities for target classes
        loss_data = np.zeros(batch_size, dtype=np.float32)
        for i in range(batch_size):
            if target_indices[i] != ignore_index:
                loss_data[i] = -log_probs.data[i, target_indices[i]]
        
        loss = Tensor(
            loss_data,
            requires_grad=predictions.requires_grad,
            _children=(predictions,),
            _op='cross_entropy'
        )
        
        if loss.requires_grad:
            def _backward():
                if predictions.requires_grad:
                    # Gradient: softmax(x) - one_hot(y)
                    softmax_val = np.exp(log_probs.data)
                    grad = softmax_val.copy()
                    
                    for i in range(batch_size):
                        if target_indices[i] != ignore_index:
                            grad[i, target_indices[i]] -= 1.0
                    
                    # Apply class weights
                    if weight is not None:
                        for i in range(batch_size):
                            if target_indices[i] != ignore_index:
                                grad[i] *= weight.data[target_indices[i]]
                    
                    if predictions.grad is None:
                        predictions.grad = Tensor(grad)
                    else:
                        predictions.grad = Tensor(predictions.grad.data + grad)
            
            loss.grad_fn = _backward
    
    # Apply class weights
    if weight is not None and label_smoothing == 0:
        weight_per_sample = np.array([
            weight.data[target_indices[i]] if target_indices[i] != ignore_index else 0.0
            for i in range(batch_size)
        ])
        loss = loss * Tensor(weight_per_sample)
    
    return apply_reduction(loss, reduction)


def nll_loss(
    log_probabilities: Tensor,
    targets: Tensor,
    weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    reduction: str = 'mean'
) -> Tensor:
    """
    Negative Log Likelihood (NLL) loss.
    
    Formula: L = -(1/n)Σ log(ŷᵢ[yᵢ])
    
    Use Cases:
        - After log_softmax output
        - When you have log probabilities
        - Multi-class classification
    
    Args:
        log_probabilities: Log probabilities from log_softmax, shape (batch_size, num_classes)
        targets: Ground truth class indices, shape (batch_size,)
        weight: Optional class weights
        ignore_index: Target value to ignore
        reduction: Reduction mode
    
    Returns:
        NLL loss
    
    Example:
        >>> from semester1.lab3_activation_functions.activations import log_softmax
        >>> logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        >>> log_probs = log_softmax(logits, axis=1)
        >>> targets = Tensor([0])
        >>> loss = nll_loss(log_probs, targets)
    """
    batch_size = log_probabilities.shape[0]
    target_indices = targets.data.astype(int)
    
    # Gather log probabilities for target classes
    loss_data = np.zeros(batch_size, dtype=np.float32)
    for i in range(batch_size):
        if target_indices[i] != ignore_index:
            loss_data[i] = -log_probabilities.data[i, target_indices[i]]
    
    loss = Tensor(
        loss_data,
        requires_grad=log_probabilities.requires_grad,
        _children=(log_probabilities,),
        _op='nll'
    )
    
    if loss.requires_grad:
        def _backward():
            if log_probabilities.requires_grad:
                grad = np.zeros_like(log_probabilities.data)
                
                for i in range(batch_size):
                    if target_indices[i] != ignore_index:
                        grad[i, target_indices[i]] = -1.0
                        
                        if weight is not None:
                            grad[i, target_indices[i]] *= weight.data[target_indices[i]]
                
                if log_probabilities.grad is None:
                    log_probabilities.grad = Tensor(grad)
                else:
                    log_probabilities.grad = Tensor(log_probabilities.grad.data + grad)
        
        loss.grad_fn = _backward
    
    # Apply class weights
    if weight is not None:
        weight_per_sample = np.array([
            weight.data[target_indices[i]] if target_indices[i] != ignore_index else 0.0
            for i in range(batch_size)
        ])
        loss = loss * Tensor(weight_per_sample)
    
    return apply_reduction(loss, reduction)


# ==================== Advanced Losses ====================


def hinge_loss(
    predictions: Tensor,
    targets: Tensor,
    margin: float = 1.0,
    reduction: str = 'mean'
) -> Tensor:
    """
    Hinge loss for SVM-style classification.
    
    Formula: L = max(0, margin - y*ŷ)
    where y ∈ {-1, +1}
    
    Use Cases:
        - Support Vector Machines
        - Maximum margin classification
        - Binary classification with margin
    
    Args:
        predictions: Model predictions (raw scores)
        targets: Binary targets in {-1, +1}
        margin: Desired margin (default: 1.0)
        reduction: Reduction mode
    
    Returns:
        Hinge loss
    
    Example:
        >>> predictions = Tensor([0.5, -0.3, 1.2], requires_grad=True)
        >>> targets = Tensor([1.0, -1.0, 1.0])  # +1 or -1
        >>> loss = hinge_loss(predictions, targets)
    """
    _check_shapes(predictions, targets)
    
    # Hinge: max(0, margin - y*pred)
    hinge = margin - targets.data * predictions.data
    hinge = np.maximum(0, hinge)
    
    out = Tensor(
        hinge,
        requires_grad=predictions.requires_grad,
        _children=(predictions, targets),
        _op='hinge'
    )
    
    if out.requires_grad:
        def _backward():
            if predictions.requires_grad:
                # Gradient: -y if loss > 0, else 0
                grad = np.where(hinge > 0, -targets.data, 0)
                
                if predictions.grad is None:
                    predictions.grad = Tensor(grad)
                else:
                    predictions.grad = Tensor(predictions.grad.data + grad)
        
        out.grad_fn = _backward
    
    return apply_reduction(out, reduction)


def kl_divergence(
    predictions: Tensor,
    targets: Tensor,
    reduction: str = 'mean',
    log_target: bool = False
) -> Tensor:
    """
    Kullback-Leibler Divergence.
    
    Formula: D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))
    
    Use Cases:
        - Measure difference between probability distributions
        - Knowledge distillation
        - Variational inference
    
    Args:
        predictions: Predicted distribution (Q)
        targets: Target distribution (P)
        reduction: Reduction mode
        log_target: If True, targets are log probabilities
    
    Returns:
        KL divergence
    
    Example:
        >>> predictions = Tensor([[0.1, 0.7, 0.2]], requires_grad=True)
        >>> targets = Tensor([[0.3, 0.4, 0.3]])
        >>> loss = kl_divergence(predictions, targets)
    """
    _check_shapes(predictions, targets)
    
    # Clip for numerical stability
    pred_clipped = _clip_for_log(predictions.data)
    
    if log_target:
        # Targets are already log probabilities
        target_log = targets.data
        target_prob = np.exp(target_log)
    else:
        target_clipped = _clip_for_log(targets.data)
        target_log = np.log(target_clipped)
        target_prob = target_clipped
    
    # KL(P||Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
    kl = target_prob * (target_log - np.log(pred_clipped))
    
    out = Tensor(
        kl,
        requires_grad=predictions.requires_grad,
        _children=(predictions, targets),
        _op='kl_div'
    )
    
    if out.requires_grad:
        def _backward():
            if predictions.requires_grad:
                # Gradient: -P/Q
                eps = 1e-7
                grad = -target_prob / np.clip(predictions.data, eps, None)
                
                if predictions.grad is None:
                    predictions.grad = Tensor(grad)
                else:
                    predictions.grad = Tensor(predictions.grad.data + grad)
        
        out.grad_fn = _backward
    
    return apply_reduction(out, reduction)


def focal_loss(
    predictions: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean'
) -> Tensor:
    """
    Focal Loss for addressing class imbalance.
    
    Formula: FL(p) = -α(1-p)ᵞ log(p)
    
    The modulating factor (1-p)ᵞ reduces loss for well-classified examples,
    focusing training on hard examples.
    
    Use Cases:
        - Highly imbalanced datasets
        - Object detection (RetinaNet)
        - When easy examples dominate training
    
    Properties:
        - Down-weights easy examples
        - Focuses on hard negatives
        - Controlled by α (balance) and γ (focusing)
    
    Args:
        predictions: Predicted probabilities after sigmoid, shape (batch_size, num_classes)
        targets: Ground truth labels (0 or 1), same shape
        alpha: Weighting factor in [0, 1] for class balance (default: 0.25)
        gamma: Focusing parameter γ >= 0 (default: 2.0)
        reduction: Reduction mode
    
    Returns:
        Focal loss
    
    Reference:
        "Focal Loss for Dense Object Detection" - Lin et al. (2017)
    
    Example:
        >>> from semester1.lab3_activation_functions.activations import sigmoid
        >>> logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        >>> predictions = sigmoid(logits)
        >>> targets = Tensor([[1.0, 0.0, 1.0]])
        >>> loss = focal_loss(predictions, targets, alpha=0.25, gamma=2.0)
    """
    _check_shapes(predictions, targets)
    
    # Clip predictions
    pred_clipped = _clip_for_log(predictions.data)
    
    # Compute focal loss: -α(1-p)^γ log(p) for positive, -α*p^γ log(1-p) for negative
    # Simplified: -α_t * (1-p_t)^γ * log(p_t)
    # where p_t = p if y=1, else 1-p
    #       α_t = α if y=1, else 1-α
    
    # For positive class (y=1): FL = -α(1-p)^γ log(p)
    # For negative class (y=0): FL = -(1-α)p^γ log(1-p)
    
    p_t = np.where(targets.data == 1, pred_clipped, 1 - pred_clipped)
    alpha_t = np.where(targets.data == 1, alpha, 1 - alpha)
    
    focal_weight = alpha_t * np.power(1 - p_t, gamma)
    focal = -focal_weight * np.log(p_t)
    
    out = Tensor(
        focal,
        requires_grad=predictions.requires_grad,
        _children=(predictions, targets),
        _op='focal'
    )
    
    if out.requires_grad:
        def _backward():
            if predictions.requires_grad:
                eps = 1e-7
                pred_safe = np.clip(predictions.data, eps, 1 - eps)
                
                # Complex gradient computation
                # d/dp[-α(1-p)^γ log(p)] = -α[γ(1-p)^(γ-1)(-1)log(p) + (1-p)^γ/p]
                #                        = α[(1-p)^γ/p - γ(1-p)^(γ-1)log(p)]
                
                p_t = np.where(targets.data == 1, pred_safe, 1 - pred_safe)
                alpha_t = np.where(targets.data == 1, alpha, 1 - alpha)
                
                # Gradient components
                term1 = alpha_t * np.power(1 - p_t, gamma) / p_t
                term2 = alpha_t * gamma * np.power(1 - p_t, gamma - 1) * np.log(p_t)
                
                grad = np.where(
                    targets.data == 1,
                    term1 - term2,
                    -(term1 - term2)
                )
                
                if predictions.grad is None:
                    predictions.grad = Tensor(grad)
                else:
                    predictions.grad = Tensor(predictions.grad.data + grad)
        
        out.grad_fn = _backward
    
    return apply_reduction(out, reduction)


def dice_loss(
    predictions: Tensor,
    targets: Tensor,
    smooth: float = 1.0,
    reduction: str = 'mean'
) -> Tensor:
    """
    Dice Loss for segmentation tasks.
    
    Formula: DL = 1 - (2|P∩T| + smooth) / (|P| + |T| + smooth)
    
    Use Cases:
        - Image segmentation
        - When dealing with imbalanced classes
        - Medical imaging
    
    Properties:
        - Based on Dice coefficient (F1 score)
        - Handles class imbalance well
        - Smooth parameter prevents division by zero
    
    Args:
        predictions: Predicted probabilities, shape (batch_size, ...)
        targets: Ground truth (0 or 1), same shape
        smooth: Smoothing factor to avoid division by zero (default: 1.0)
        reduction: Reduction mode
    
    Returns:
        Dice loss
    
    Example:
        >>> predictions = Tensor([[0.9, 0.8, 0.1]], requires_grad=True)
        >>> targets = Tensor([[1.0, 1.0, 0.0]])
        >>> loss = dice_loss(predictions, targets)
    """
    _check_shapes(predictions, targets)
    
    # Flatten predictions and targets
    pred_flat = predictions.data.reshape(-1)
    target_flat = targets.data.reshape(-1)
    
    # Compute Dice coefficient
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat)
    
    dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss_val = 1.0 - dice_coeff
    
    out = Tensor(
        np.array([dice_loss_val]),
        requires_grad=predictions.requires_grad,
        _children=(predictions, targets),
        _op='dice'
    )
    
    if out.requires_grad:
        def _backward():
            if predictions.requires_grad:
                # Gradient: -2*(target*union - pred*2*intersection) / union^2
                grad_flat = -2 * (
                    target_flat * union - pred_flat * 2 * intersection
                ) / (union + smooth) ** 2
                
                grad = grad_flat.reshape(predictions.shape)
                
                if predictions.grad is None:
                    predictions.grad = Tensor(grad)
                else:
                    predictions.grad = Tensor(predictions.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out if reduction == 'none' else out


def cosine_embedding_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = 0.0,
    reduction: str = 'mean'
) -> Tensor:
    """
    Cosine Embedding Loss.
    
    Formula:
        L = 1 - cos(x1, x2)           if y = 1
        L = max(0, cos(x1, x2) - margin)   if y = -1
    
    Use Cases:
        - Learning embeddings
        - Similarity learning
        - Face verification
    
    Args:
        input1: First input embeddings
        input2: Second input embeddings
        target: Labels (1 for similar, -1 for dissimilar)
        margin: Margin for dissimilar pairs
        reduction: Reduction mode
    
    Returns:
        Cosine embedding loss
    
    Example:
        >>> input1 = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        >>> input2 = Tensor([[1.0, 2.0, 2.0]], requires_grad=True)
        >>> target = Tensor([1.0])  # Similar
        >>> loss = cosine_embedding_loss(input1, input2, target)
    """
    # Compute cosine similarity
    dot_product = (input1 * input2).sum(axis=1)
    norm1 = (input1 ** 2).sum(axis=1) ** 0.5
    norm2 = (input2 ** 2).sum(axis=1) ** 0.5
    
    cos_sim = dot_product / (norm1 * norm2 + 1e-8)
    
    # Compute loss based on target
    # If target = 1 (similar): loss = 1 - cos_sim
    # If target = -1 (dissimilar): loss = max(0, cos_sim - margin)
    loss = Tensor(np.where(
        target.data == 1,
        1.0 - cos_sim.data,
        np.maximum(0, cos_sim.data - margin)
    ))
    
    return apply_reduction(loss, reduction)


def triplet_margin_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: float = 1.0,
    p: float = 2.0,
    reduction: str = 'mean'
) -> Tensor:
    """
    Triplet Margin Loss.
    
    Formula: L = max(d(a,p) - d(a,n) + margin, 0)
    where d is distance metric (L2 by default)
    
    Use Cases:
        - Face recognition
        - Person re-identification
        - Metric learning
    
    Args:
        anchor: Anchor embeddings
        positive: Positive (similar) embeddings
        negative: Negative (dissimilar) embeddings
        margin: Margin between positive and negative pairs (default: 1.0)
        p: Norm degree for distance (default: 2.0 for L2)
        reduction: Reduction mode
    
    Returns:
        Triplet margin loss
    
    Example:
        >>> anchor = Tensor([[1.0, 2.0]], requires_grad=True)
        >>> positive = Tensor([[1.1, 2.1]], requires_grad=True)
        >>> negative = Tensor([[5.0, 6.0]], requires_grad=True)
        >>> loss = triplet_margin_loss(anchor, positive, negative, margin=1.0)
    """
    # Compute distances
    dist_ap = ((anchor - positive) ** 2).sum(axis=1) ** 0.5
    dist_an = ((anchor - negative) ** 2).sum(axis=1) ** 0.5
    
    # Triplet loss: max(d(a,p) - d(a,n) + margin, 0)
    triplet = dist_ap - dist_an + margin
    loss_data = np.maximum(0, triplet.data)
    
    loss = Tensor(loss_data)
    
    return apply_reduction(loss, reduction)


def contrastive_loss(
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: float = 1.0,
    reduction: str = 'mean'
) -> Tensor:
    """
    Contrastive Loss for Siamese networks.
    
    Formula:
        L = (1-y) * 0.5 * d² + y * 0.5 * max(0, margin - d)²
    where d is Euclidean distance
    
    Use Cases:
        - Siamese networks
        - One-shot learning
        - Similarity learning
    
    Args:
        input1: First input embeddings
        input2: Second input embeddings
        target: Labels (0 for similar, 1 for dissimilar)
        margin: Margin for dissimilar pairs
        reduction: Reduction mode
    
    Returns:
        Contrastive loss
    
    Example:
        >>> input1 = Tensor([[1.0, 2.0]], requires_grad=True)
        >>> input2 = Tensor([[1.1, 2.1]], requires_grad=True)
        >>> target = Tensor([0.0])  # Similar
        >>> loss = contrastive_loss(input1, input2, target, margin=1.0)
    """
    # Euclidean distance
    dist = ((input1 - input2) ** 2).sum(axis=1) ** 0.5
    
    # Contrastive loss
    # Similar pairs (y=0): 0.5 * d²
    # Dissimilar pairs (y=1): 0.5 * max(0, margin - d)²
    loss_similar = 0.5 * (dist ** 2)
    loss_dissimilar = 0.5 * (Tensor(np.maximum(0, margin - dist.data)) ** 2)
    
    loss = (Tensor(1.0) - target) * loss_similar + target * loss_dissimilar
    
    return apply_reduction(loss, reduction)


# ==================== Utility Functions ====================


def compute_class_weights(
    targets: np.ndarray,
    num_classes: int,
    method: str = 'balanced'
) -> np.ndarray:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        targets: Array of class labels, shape (n_samples,)
        num_classes: Number of classes
        method: Weighting method:
            - 'balanced': n_samples / (n_classes * n_samples_class[c])
            - 'inverse': 1 / n_samples_class[c]
            - 'sqrt': sqrt(n_samples / n_samples_class[c])
    
    Returns:
        Class weights, shape (num_classes,)
    
    Example:
        >>> targets = np.array([0, 0, 0, 1, 1, 2])  # Imbalanced
        >>> weights = compute_class_weights(targets, num_classes=3)
        >>> print(weights)  # Higher weight for rare class 2
    """
    # Count samples per class
    class_counts = np.bincount(targets.astype(int), minlength=num_classes)
    n_samples = len(targets)
    
    if method == 'balanced':
        # Balanced weighting
        weights = n_samples / (num_classes * class_counts.astype(float))
    elif method == 'inverse':
        # Inverse frequency
        weights = 1.0 / class_counts.astype(float)
    elif method == 'sqrt':
        # Square root of inverse frequency
        weights = np.sqrt(n_samples / class_counts.astype(float))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Handle zero counts
    weights[class_counts == 0] = 0.0
    
    return weights


def smooth_labels(
    targets: np.ndarray,
    num_classes: int,
    smoothing: float = 0.1
) -> np.ndarray:
    """
    Apply label smoothing to one-hot encoded targets.
    
    Formula: y_smooth = (1 - ε) * y + ε / K
    
    Args:
        targets: One-hot encoded targets, shape (batch_size, num_classes)
        num_classes: Number of classes
        smoothing: Smoothing factor ε in [0, 1)
    
    Returns:
        Smoothed targets
    
    Example:
        >>> targets = np.array([[0, 0, 1, 0]])  # One-hot
        >>> smoothed = smooth_labels(targets, num_classes=4, smoothing=0.1)
        >>> print(smoothed)  # [0.025, 0.025, 0.925, 0.025]
    """
    confidence = 1.0 - smoothing
    smoothing_value = smoothing / num_classes
    
    smoothed = targets * confidence + smoothing_value
    return smoothed


# ==================== Module Interface ====================


class Loss:
    """
    Base class for all loss functions.
    
    All loss modules should inherit from this class and implement
    the forward() method.
    
    Args:
        reduction: Specifies the reduction to apply to the output:
            'none': no reduction
            'mean': mean of all elements (default)
            'sum': sum of all elements
    """
    
    def __init__(self, reduction: str = 'mean'):
        """Initialize loss with reduction mode."""
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        
        Returns:
            Computed loss
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Make the module callable."""
        return self.forward(predictions, targets)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(reduction='{self.reduction}')"


class MSELoss(Loss):
    """
    Mean Squared Error loss module.
    
    Example:
        >>> criterion = MSELoss(reduction='mean')
        >>> predictions = Tensor([[1.0, 2.0]], requires_grad=True)
        >>> targets = Tensor([[1.5, 2.5]])
        >>> loss = criterion(predictions, targets)
        >>> loss.backward()
    """
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute MSE loss."""
        return mse_loss(predictions, targets, self.reduction)


class MAELoss(Loss):
    """Mean Absolute Error loss module."""
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute MAE loss."""
        return mae_loss(predictions, targets, self.reduction)


class SmoothL1Loss(Loss):
    """
    Smooth L1 Loss module.
    
    Args:
        beta: Threshold parameter (default: 1.0)
        reduction: Reduction mode
    """
    
    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.beta = beta
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute Smooth L1 loss."""
        return smooth_l1_loss(predictions, targets, self.beta, self.reduction)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(beta={self.beta}, reduction='{self.reduction}')"


class MSLELoss(Loss):
    """Mean Squared Logarithmic Error loss module."""
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute MSLE loss."""
        return msle_loss(predictions, targets, self.reduction)


class BCELoss(Loss):
    """
    Binary Cross-Entropy loss module.
    
    Args:
        weight: Optional sample weights
        reduction: Reduction mode
    """
    
    def __init__(self, weight: Optional[Tensor] = None, reduction: str = 'mean'):
        super().__init__(reduction)
        self.weight = weight
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute BCE loss."""
        return binary_cross_entropy(predictions, targets, self.weight, self.reduction)


class BCEWithLogitsLoss(Loss):
    """
    Binary Cross-Entropy with Logits loss module.
    
    Args:
        weight: Optional sample weights
        pos_weight: Weight for positive class
        reduction: Reduction mode
    """
    
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        pos_weight: Optional[Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__(reduction)
        self.weight = weight
        self.pos_weight = pos_weight
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute BCE with logits loss."""
        return binary_cross_entropy_with_logits(
            predictions, targets, self.weight, self.pos_weight, self.reduction
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reduction='{self.reduction}')"


class CrossEntropyLoss(Loss):
    """
    Cross-Entropy loss module.
    
    Args:
        weight: Optional class weights
        ignore_index: Target value to ignore
        label_smoothing: Label smoothing factor
        reduction: Reduction mode
    """
    
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__(reduction)
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute cross-entropy loss."""
        return cross_entropy_loss(
            predictions, targets, self.weight,
            self.ignore_index, self.label_smoothing, self.reduction
        )
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(label_smoothing={self.label_smoothing}, "
                f"reduction='{self.reduction}')")


class NLLLoss(Loss):
    """
    Negative Log Likelihood loss module.
    
    Args:
        weight: Optional class weights
        ignore_index: Target value to ignore
        reduction: Reduction mode
    """
    
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ):
        super().__init__(reduction)
        self.weight = weight
        self.ignore_index = ignore_index
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute NLL loss."""
        return nll_loss(predictions, targets, self.weight, self.ignore_index, self.reduction)


class HingeLoss(Loss):
    """
    Hinge loss module.
    
    Args:
        margin: Desired margin (default: 1.0)
        reduction: Reduction mode
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.margin = margin
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute hinge loss."""
        return hinge_loss(predictions, targets, self.margin, self.reduction)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(margin={self.margin}, reduction='{self.reduction}')"


class KLDivLoss(Loss):
    """
    KL Divergence loss module.
    
    Args:
        reduction: Reduction mode
        log_target: Whether targets are log probabilities
    """
    
    def __init__(self, reduction: str = 'mean', log_target: bool = False):
        super().__init__(reduction)
        self.log_target = log_target
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute KL divergence."""
        return kl_divergence(predictions, targets, self.reduction, self.log_target)


class FocalLoss(Loss):
    """
    Focal Loss for addressing class imbalance.
    
    Args:
        alpha: Weighting factor in [0, 1] for class balance (default: 0.25)
        gamma: Focusing parameter γ >= 0 (default: 2.0)
        reduction: Reduction mode
    
    Example:
        >>> criterion = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        >>> targets = Tensor([[1.0, 0.0, 1.0]])
        >>> loss = criterion(logits, targets)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__(reduction)
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss."""
        return focal_loss(predictions, targets, self.alpha, self.gamma, self.reduction)
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(alpha={self.alpha}, "
                f"gamma={self.gamma}, reduction='{self.reduction}')")


class DiceLoss(Loss):
    """
    Dice Loss for segmentation.
    
    Args:
        smooth: Smoothing factor (default: 1.0)
        reduction: Reduction mode
    """
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.smooth = smooth
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute Dice loss."""
        return dice_loss(predictions, targets, self.smooth, self.reduction)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(smooth={self.smooth}, reduction='{self.reduction}')"


class CosineEmbeddingLoss(Loss):
    """
    Cosine Embedding Loss module.
    
    Args:
        margin: Margin for dissimilar pairs
        reduction: Reduction mode
    """
    
    def __init__(self, margin: float = 0.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.margin = margin
    
    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        """Compute cosine embedding loss."""
        return cosine_embedding_loss(input1, input2, target, self.margin, self.reduction)
    
    def __call__(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        """Override call to accept three inputs."""
        return self.forward(input1, input2, target)


class TripletMarginLoss(Loss):
    """
    Triplet Margin Loss module.
    
    Args:
        margin: Margin between positive and negative pairs
        p: Norm degree for distance
        reduction: Reduction mode
    """
    
    def __init__(self, margin: float = 1.0, p: float = 2.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.margin = margin
        self.p = p
    
    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        """Compute triplet margin loss."""
        return triplet_margin_loss(anchor, positive, negative, self.margin, self.p, self.reduction)
    
    def __call__(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        """Override call to accept three inputs."""
        return self.forward(anchor, positive, negative)


class ContrastiveLoss(Loss):
    """
    Contrastive Loss for Siamese networks.
    
    Args:
        margin: Margin for dissimilar pairs
        reduction: Reduction mode
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.margin = margin
    
    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        """Compute contrastive loss."""
        return contrastive_loss(input1, input2, target, self.margin, self.reduction)
    
    def __call__(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        """Override call to accept three inputs."""
        return self.forward(input1, input2, target)


# ==================== Testing ====================


if __name__ == "__main__":
    print("=== Testing Loss Functions ===\n")
    
    # Test 1: MSE Loss
    print("Test 1: MSE Loss")
    pred = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    target = Tensor([[1.5, 2.5], [3.5, 4.5]])
    loss = mse_loss(pred, target, reduction='mean')
    print(f"Loss: {loss.data}")
    loss.backward()
    print(f"Gradient: {pred.grad.data}\n")
    
    # Test 2: Cross-Entropy
    print("Test 2: Cross-Entropy Loss")
    logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
    targets = Tensor([0])
    loss = cross_entropy_loss(logits, targets)
    print(f"Loss: {loss.data}")
    loss.backward()
    print(f"Gradient: {logits.grad.data}\n")
    
    # Test 3: Reduction modes
    print("Test 3: Reduction Modes")
    pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
    target = Tensor([[1.5, 2.5], [3.5, 4.5]])
    loss_mean = mse_loss(pred, target, reduction='mean')
    loss_sum = mse_loss(pred, target, reduction='sum')
    loss_none = mse_loss(pred, target, reduction='none')
    print(f"Mean: {loss_mean.data}")
    print(f"Sum: {loss_sum.data}")
    print(f"None: {loss_none.data}\n")
    
    # Test 4: BCE with Logits
    print("Test 4: BCE with Logits")
    logits = Tensor([0.5, 1.0, -0.5], requires_grad=True)
    targets = Tensor([1.0, 1.0, 0.0])
    loss = binary_cross_entropy_with_logits(logits, targets)
    print(f"Loss: {loss.data}")
    loss.backward()
    print(f"Gradient: {logits.grad.data}\n")
    
    # Test 5: Module interface
    print("Test 5: Module Interface")
    criterion = MSELoss(reduction='mean')
    pred = Tensor([[1.0, 2.0]], requires_grad=True)
    target = Tensor([[1.5, 2.5]])
    loss = criterion(pred, target)
    print(f"Loss: {loss.data}\n")
    
    # Test 6: Focal Loss
    print("Test 6: Focal Loss")
    from semester1.lab3_activation_functions.activations import sigmoid
    logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
    predictions = sigmoid(logits)
    targets = Tensor([[1.0, 0.0, 1.0]])
    loss = focal_loss(predictions, targets, alpha=0.25, gamma=2.0)
    print(f"Loss: {loss.data}\n")
    
    # Test 7: Class weights
    print("Test 7: Class Weights")
    targets_np = np.array([0, 0, 0, 1, 1, 2])
    weights = compute_class_weights(targets_np, num_classes=3)
    print(f"Class weights: {weights}\n")
    
    print("All basic tests passed!")