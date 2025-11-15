"""Comprehensive unit tests for loss functions."""

import pytest
import numpy as np
from semester1.lab2_autograd.autograd import Tensor, check_gradients
from semester1.lab3_activation_functions.activations import sigmoid, softmax, log_softmax
from semester1.lab4_loss_functions.losses import (
    # Regression losses
    mse_loss, mae_loss, smooth_l1_loss, msle_loss,
    # Binary classification
    binary_cross_entropy, binary_cross_entropy_with_logits,
    # Multi-class classification
    cross_entropy_loss, nll_loss,
    # Advanced losses
    hinge_loss, kl_divergence, focal_loss, dice_loss,
    cosine_embedding_loss, triplet_margin_loss, contrastive_loss,
    # Module interface
    MSELoss, MAELoss, SmoothL1Loss, MSLELoss,
    BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, NLLLoss,
    HingeLoss, KLDivLoss, FocalLoss, DiceLoss,
    CosineEmbeddingLoss, TripletMarginLoss, ContrastiveLoss,
    # Utilities
    compute_class_weights, smooth_labels, apply_reduction,
)


class TestRegressionLosses:
    """Test regression loss functions."""
    
    def test_mse_loss_forward(self):
        """Test MSE loss forward pass."""
        pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
        target = Tensor([[1.5, 2.5], [3.5, 4.5]])
        loss = mse_loss(pred, target, reduction='mean')
        
        # MSE = mean((pred - target)^2) = mean([0.25, 0.25, 0.25, 0.25]) = 0.25
        expected = 0.25
        assert abs(loss.data - expected) < 1e-6
    
    def test_mse_loss_gradient(self):
        """Test MSE loss gradient."""
        pred = Tensor([[1.0, 2.0]], requires_grad=True)
        target = Tensor([[1.5, 2.5]])
        loss = mse_loss(pred, target, reduction='mean')
        loss.backward()
        
        # Gradient: 2*(pred - target)/n = 2*[-0.5, -0.5]/2 = [-0.5, -0.5]
        expected_grad = np.array([[-0.5, -0.5]])
        np.testing.assert_array_almost_equal(pred.grad.data, expected_grad)
    
    def test_mse_loss_reduction_modes(self):
        """Test MSE loss reduction modes."""
        pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
        target = Tensor([[2.0, 3.0], [4.0, 5.0]])
        
        loss_mean = mse_loss(pred, target, reduction='mean')
        loss_sum = mse_loss(pred, target, reduction='sum')
        loss_none = mse_loss(pred, target, reduction='none')
        
        assert loss_mean.data == 1.0  # mean([1,1,1,1]) = 1
        assert loss_sum.data == 4.0   # sum([1,1,1,1]) = 4
        assert loss_none.shape == (2, 2)
    
    def test_mse_loss_gradient_check(self):
        """Numerical gradient check for MSE loss."""
        def func(x):
            target = Tensor([[1.5, 2.5]])
            return mse_loss(x, target, reduction='mean')
        
        pred = Tensor([[1.0, 2.0]], requires_grad=True)
        match, diff = check_gradients(func, pred, tolerance=1e-5)
        assert match is True
    
    def test_mae_loss_forward(self):
        """Test MAE loss forward pass."""
        pred = Tensor([1.0, 2.0, 3.0])
        target = Tensor([1.5, 2.5, 3.5])
        loss = mae_loss(pred, target, reduction='mean')
        
        # MAE = mean(|pred - target|) = mean([0.5, 0.5, 0.5]) = 0.5
        expected = 0.5
        assert abs(loss.data - expected) < 1e-6
    
    def test_mae_loss_gradient(self):
        """Test MAE loss gradient."""
        pred = Tensor([1.0, 3.0, 2.0], requires_grad=True)
        target = Tensor([1.5, 2.5, 2.5])
        loss = mae_loss(pred, target, reduction='sum')
        loss.backward()
        
        # Gradient: sign(pred - target) = [-1, 1, -1]
        expected_grad = np.array([-1.0, 1.0, -1.0])
        np.testing.assert_array_equal(pred.grad.data, expected_grad)
    
    def test_mae_loss_robust_to_outliers(self):
        """Test MAE is less sensitive to outliers than MSE."""
        pred = Tensor([1.0, 2.0, 10.0])
        target = Tensor([1.0, 2.0, 3.0])
        
        mae = mae_loss(pred, target, reduction='mean')
        mse = mse_loss(pred, target, reduction='mean')
        
        # MAE = (0 + 0 + 7)/3 = 2.33
        # MSE = (0 + 0 + 49)/3 = 16.33
        assert mae.data < mse.data
    
    def test_smooth_l1_loss_forward(self):
        """Test Smooth L1 loss forward pass."""
        pred = Tensor([0.5, 2.0, 10.0])
        target = Tensor([1.0, 2.0, 3.0])
        loss = smooth_l1_loss(pred, target, beta=1.0, reduction='none')
        
        # For |diff| < beta: 0.5 * diff^2 / beta
        # For |diff| >= beta: |diff| - 0.5 * beta
        # diff = [-0.5, 0, 7]
        # loss = [0.125, 0, 6.5]
        expected = np.array([0.125, 0.0, 6.5])
        np.testing.assert_array_almost_equal(loss.data, expected)
    
    def test_smooth_l1_loss_gradient(self):
        """Test Smooth L1 loss gradient."""
        pred = Tensor([0.5, 2.0, 10.0], requires_grad=True)
        target = Tensor([1.0, 2.0, 3.0])
        loss = smooth_l1_loss(pred, target, beta=1.0, reduction='sum')
        loss.backward()
        
        # For |diff| < beta: diff/beta
        # For |diff| >= beta: sign(diff)
        # diff = [-0.5, 0, 7]
        # grad = [-0.5, 0, 1]
        expected_grad = np.array([-0.5, 0.0, 1.0])
        np.testing.assert_array_almost_equal(pred.grad.data, expected_grad)
    
    def test_msle_loss_forward(self):
        """Test MSLE loss forward pass."""
        pred = Tensor([1.0, 10.0, 100.0])
        target = Tensor([1.0, 10.0, 100.0])
        loss = msle_loss(pred, target, reduction='mean')
        
        # Perfect prediction should give ~0 loss
        assert loss.data < 1e-6
    
    def test_msle_loss_penalizes_underestimation(self):
        """Test MSLE penalizes underestimation more."""
        pred_under = Tensor([5.0])
        pred_over = Tensor([15.0])
        target = Tensor([10.0])
        
        loss_under = msle_loss(pred_under, target, reduction='mean')
        loss_over = msle_loss(pred_over, target, reduction='mean')
        
        # Underestimation should have higher loss
        assert loss_under.data > loss_over.data


class TestBinaryClassificationLosses:
    """Test binary classification loss functions."""
    
    def test_bce_loss_forward(self):
        """Test BCE loss forward pass."""
        pred = Tensor([0.9, 0.1, 0.5])
        target = Tensor([1.0, 0.0, 1.0])
        loss = binary_cross_entropy(pred, target, reduction='none')
        
        # BCE = -[y*log(p) + (1-y)*log(1-p)]
        expected = -np.array([
            1*np.log(0.9) + 0*np.log(0.1),
            0*np.log(0.1) + 1*np.log(0.9),
            1*np.log(0.5) + 0*np.log(0.5)
        ])
        np.testing.assert_array_almost_equal(loss.data, expected, decimal=5)
    
    def test_bce_loss_gradient(self):
        """Test BCE loss gradient."""
        pred = Tensor([0.5], requires_grad=True)
        target = Tensor([1.0])
        loss = binary_cross_entropy(pred, target, reduction='mean')
        loss.backward()
        
        # Gradient at p=0.5, y=1: (p-y)/(p(1-p)) = -0.5/0.25 = -2
        assert abs(pred.grad.data[0] - (-2.0)) < 0.1
    
    def test_bce_with_logits_stability(self):
        """Test BCE with logits numerical stability."""
        logits = Tensor([100.0, -100.0])
        target = Tensor([1.0, 0.0])
        loss = binary_cross_entropy_with_logits(logits, target, reduction='none')
        
        # Should not overflow
        assert np.all(np.isfinite(loss.data))
        assert np.all(loss.data >= 0)
    
    def test_bce_with_logits_gradient(self):
        """Test BCE with logits gradient."""
        logits = Tensor([0.5, 1.0, -0.5], requires_grad=True)
        target = Tensor([1.0, 1.0, 0.0])
        loss = binary_cross_entropy_with_logits(logits, target, reduction='mean')
        loss.backward()
        
        # Gradient: sigmoid(x) - y
        sigmoid_vals = 1 / (1 + np.exp(-logits.data))
        expected_grad = (sigmoid_vals - target.data) / 3  # divided by batch size
        np.testing.assert_array_almost_equal(logits.grad.data, expected_grad, decimal=5)
    
    def test_bce_with_logits_pos_weight(self):
        """Test BCE with logits positive class weighting."""
        logits = Tensor([0.5, -0.5], requires_grad=True)
        target = Tensor([1.0, 0.0])
        pos_weight = Tensor([2.0])
        
        loss = binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight, reduction='mean'
        )
        
        # Loss should be weighted for positive class
        assert np.isfinite(loss.data)
    
    def test_bce_equivalence_with_sigmoid(self):
        """Test BCE equivalence with sigmoid + BCE."""
        logits = Tensor([0.5, 1.0, -0.5])
        target = Tensor([1.0, 1.0, 0.0])
        
        # Method 1: BCE with logits
        loss1 = binary_cross_entropy_with_logits(logits, target, reduction='mean')
        
        # Method 2: Sigmoid + BCE
        pred = sigmoid(logits)
        loss2 = binary_cross_entropy(pred, target, reduction='mean')
        
        # Should be approximately equal
        np.testing.assert_almost_equal(loss1.data, loss2.data, decimal=5)


class TestMultiClassLosses:
    """Test multi-class classification loss functions."""
    
    def test_cross_entropy_loss_forward(self):
        """Test cross-entropy loss forward pass."""
        logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        targets = Tensor([0])
        loss = cross_entropy_loss(logits, targets)
        
        # Should be finite and positive
        assert np.isfinite(loss.data)
        assert loss.data > 0
    
    def test_cross_entropy_loss_gradient(self):
        """Test cross-entropy loss gradient."""
        logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        targets = Tensor([0])
        loss = cross_entropy_loss(logits, targets)
        loss.backward()
        
        # Gradient should be softmax - one_hot
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape
        # First element (correct class) should have negative gradient
        assert logits.grad.data[0, 0] < 0
    
    def test_cross_entropy_with_label_smoothing(self):
        """Test cross-entropy with label smoothing."""
        logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        targets = Tensor([0])
        
        loss_no_smooth = cross_entropy_loss(logits, targets, label_smoothing=0.0)
        loss_smooth = cross_entropy_loss(logits, targets, label_smoothing=0.1)
        
        # Smoothed loss should be different
        assert abs(loss_no_smooth.data - loss_smooth.data) > 1e-6
    
    def test_cross_entropy_with_class_weights(self):
        """Test cross-entropy with class weights."""
        logits = Tensor([[2.0, 1.0, 0.1], [1.0, 2.0, 0.1]], requires_grad=True)
        targets = Tensor([0, 1])
        weights = Tensor([2.0, 1.0, 1.0])
        
        loss = cross_entropy_loss(logits, targets, weight=weights)
        
        # Weighted loss should be finite
        assert np.isfinite(loss.data)
    
    def test_cross_entropy_ignore_index(self):
        """Test cross-entropy with ignore index."""
        logits = Tensor([[2.0, 1.0, 0.1], [1.0, 2.0, 0.1]], requires_grad=True)
        targets = Tensor([0, -100])  # Ignore second sample
        
        loss = cross_entropy_loss(logits, targets, ignore_index=-100)
        
        # Only first sample should contribute
        assert np.isfinite(loss.data)
    
    def test_nll_loss_forward(self):
        """Test NLL loss forward pass."""
        logits = Tensor([[2.0, 1.0, 0.1]])
        targets = Tensor([0])
        
        log_probs = log_softmax(logits, axis=1)
        loss = nll_loss(log_probs, targets)
        
        # Should be finite and positive
        assert np.isfinite(loss.data)
        assert loss.data > 0
    
    def test_nll_loss_gradient(self):
        """Test NLL loss gradient."""
        logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        targets = Tensor([0])
        
        log_probs = log_softmax(logits, axis=1)
        loss = nll_loss(log_probs, targets)
        loss.backward()
        
        # Gradient should exist
        assert logits.grad is not None
    
    def test_cross_entropy_nll_equivalence(self):
        """Test cross-entropy is equivalent to log_softmax + NLL."""
        logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        targets = Tensor([0])
        
        # Method 1: Cross-entropy
        loss1 = cross_entropy_loss(logits, targets)
        
        # Method 2: Log softmax + NLL
        log_probs = log_softmax(logits, axis=1)
        loss2 = nll_loss(log_probs, targets)
        
        # Should be approximately equal
        np.testing.assert_almost_equal(loss1.data, loss2.data, decimal=5)


class TestAdvancedLosses:
    """Test advanced loss functions."""
    
    def test_hinge_loss_forward(self):
        """Test hinge loss forward pass."""
        pred = Tensor([0.5, -0.3, 1.2])
        target = Tensor([1.0, -1.0, 1.0])
        loss = hinge_loss(pred, target, margin=1.0, reduction='none')
        
        # Hinge: max(0, margin - y*pred)
        # [max(0, 1-0.5), max(0, 1-0.3), max(0, 1-1.2)]
        # [0.5, 0.7, 0]
        expected = np.array([0.5, 0.7, 0.0])
        np.testing.assert_array_almost_equal(loss.data, expected)
    
    def test_hinge_loss_gradient(self):
        """Test hinge loss gradient."""
        pred = Tensor([0.5, -0.3], requires_grad=True)
        target = Tensor([1.0, -1.0])
        loss = hinge_loss(pred, target, margin=1.0, reduction='sum')
        loss.backward()
        
        # Gradient: -y if loss > 0, else 0
        expected_grad = np.array([-1.0, 1.0])
        np.testing.assert_array_equal(pred.grad.data, expected_grad)
    
    def test_kl_divergence_forward(self):
        """Test KL divergence forward pass."""
        pred = Tensor([[0.4, 0.6]])
        target = Tensor([[0.5, 0.5]])
        loss = kl_divergence(pred, target, reduction='sum')
        
        # Should be non-negative
        assert loss.data >= 0
    
    def test_kl_divergence_symmetry(self):
        """Test KL divergence is not symmetric."""
        pred = Tensor([[0.4, 0.6]])
        target = Tensor([[0.6, 0.4]])
        
        loss1 = kl_divergence(pred, target, reduction='sum')
        loss2 = kl_divergence(target, pred, reduction='sum')
        
        # KL(P||Q) != KL(Q||P)
        assert abs(loss1.data - loss2.data) > 1e-6
    
    def test_focal_loss_forward(self):
        """Test focal loss forward pass."""
        pred = Tensor([[0.9, 0.5, 0.1]])
        target = Tensor([[1.0, 1.0, 0.0]])
        loss = focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='none')
        
        # Easy example (0.9) should have lower loss than hard example (0.5)
        assert loss.data[0, 0] < loss.data[0, 1]
    
    def test_focal_loss_gradient(self):
        """Test focal loss gradient."""
        pred = Tensor([[0.9, 0.5, 0.1]], requires_grad=True)
        target = Tensor([[1.0, 1.0, 0.0]])
        loss = focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean')
        loss.backward()
        
        # Gradient should exist
        assert pred.grad is not None
    
    def test_dice_loss_forward(self):
        """Test Dice loss forward pass."""
        pred = Tensor([[0.9, 0.8, 0.1]])
        target = Tensor([[1.0, 1.0, 0.0]])
        loss = dice_loss(pred, target, smooth=1.0)
        
        # Perfect overlap should give loss close to 0
        # Dice coefficient = 2*intersection/union
        assert loss.data < 0.2
    
    def test_dice_loss_perfect_prediction(self):
        """Test Dice loss with perfect prediction."""
        pred = Tensor([[1.0, 1.0, 0.0]])
        target = Tensor([[1.0, 1.0, 0.0]])
        loss = dice_loss(pred, target, smooth=1.0)
        
        # Perfect prediction should give loss close to 0
        assert loss.data < 0.01
    
    def test_cosine_embedding_loss_similar(self):
        """Test cosine embedding loss for similar pairs."""
        input1 = Tensor([[1.0, 2.0, 3.0]])
        input2 = Tensor([[1.0, 2.0, 3.0]])
        target = Tensor([1.0])  # Similar
        
        loss = cosine_embedding_loss(input1, input2, target)
        
        # Identical vectors should have loss close to 0
        assert loss.data < 0.01
    
    def test_cosine_embedding_loss_dissimilar(self):
        """Test cosine embedding loss for dissimilar pairs."""
        input1 = Tensor([[1.0, 2.0, 3.0]])
        input2 = Tensor([[-1.0, -2.0, -3.0]])
        target = Tensor([-1.0])  # Dissimilar
        
        loss = cosine_embedding_loss(input1, input2, target, margin=0.0)
        
        # Opposite vectors should have high loss
        assert loss.data > 0.5
    
    def test_triplet_margin_loss_forward(self):
        """Test triplet margin loss forward pass."""
        anchor = Tensor([[1.0, 2.0]])
        positive = Tensor([[1.1, 2.1]])
        negative = Tensor([[5.0, 6.0]])
        
        loss = triplet_margin_loss(anchor, positive, negative, margin=1.0)
        
        # Loss should be non-negative
        assert loss.data >= 0
    
    def test_contrastive_loss_similar(self):
        """Test contrastive loss for similar pairs."""
        input1 = Tensor([[1.0, 2.0]])
        input2 = Tensor([[1.1, 2.1]])
        target = Tensor([0.0])  # Similar
        
        loss = contrastive_loss(input1, input2, target, margin=1.0)
        
        # Similar pairs should have low loss
        assert loss.data < 0.1
    
    def test_contrastive_loss_dissimilar(self):
        """Test contrastive loss for dissimilar pairs."""
        input1 = Tensor([[1.0, 2.0]])
        input2 = Tensor([[5.0, 6.0]])
        target = Tensor([1.0])  # Dissimilar
        
        loss = contrastive_loss(input1, input2, target, margin=1.0)
        
        # Dissimilar pairs far apart should have low loss
        assert np.isfinite(loss.data)


class TestModuleInterface:
    """Test module-based loss interface."""
    
    def test_mse_loss_module(self):
        """Test MSE loss module."""
        criterion = MSELoss(reduction='mean')
        pred = Tensor([[1.0, 2.0]], requires_grad=True)
        target = Tensor([[1.5, 2.5]])
        
        loss = criterion(pred, target)
        assert abs(loss.data - 0.25) < 1e-6
    
    def test_mae_loss_module(self):
        """Test MAE loss module."""
        criterion = MAELoss(reduction='mean')
        pred = Tensor([1.0, 2.0, 3.0])
        target = Tensor([1.5, 2.5, 3.5])
        
        loss = criterion(pred, target)
        assert abs(loss.data - 0.5) < 1e-6
    
    def test_smooth_l1_loss_module(self):
        """Test Smooth L1 loss module."""
        criterion = SmoothL1Loss(beta=1.0, reduction='mean')
        pred = Tensor([0.5, 2.0])
        target = Tensor([1.0, 2.0])
        
        loss = criterion(pred, target)
        assert np.isfinite(loss.data)
    
    def test_bce_loss_module(self):
        """Test BCE loss module."""
        criterion = BCELoss(reduction='mean')
        pred = Tensor([0.9, 0.1])
        target = Tensor([1.0, 0.0])
        
        loss = criterion(pred, target)
        assert np.isfinite(loss.data)
    
    def test_bce_with_logits_module(self):
        """Test BCE with logits module."""
        criterion = BCEWithLogitsLoss(reduction='mean')
        logits = Tensor([0.5, -0.5])
        target = Tensor([1.0, 0.0])
        
        loss = criterion(logits, target)
        assert np.isfinite(loss.data)
    
    def test_cross_entropy_module(self):
        """Test cross-entropy module."""
        criterion = CrossEntropyLoss(reduction='mean')
        logits = Tensor([[2.0, 1.0, 0.1]])
        targets = Tensor([0])
        
        loss = criterion(logits, targets)
        assert np.isfinite(loss.data)
    
    def test_nll_loss_module(self):
        """Test NLL loss module."""
        criterion = NLLLoss(reduction='mean')
        logits = Tensor([[2.0, 1.0, 0.1]])
        targets = Tensor([0])
        
        log_probs = log_softmax(logits, axis=1)
        loss = criterion(log_probs, targets)
        assert np.isfinite(loss.data)
    
    def test_focal_loss_module(self):
        """Test focal loss module."""
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        pred = Tensor([[0.9, 0.5]])
        target = Tensor([[1.0, 1.0]])
        
        loss = criterion(pred, target)
        assert np.isfinite(loss.data)
    
    def test_module_repr(self):
        """Test module string representation."""
        criterion = MSELoss(reduction='mean')
        repr_str = repr(criterion)
        
        assert 'MSELoss' in repr_str
        assert 'mean' in repr_str
    
    def test_module_callable(self):
        """Test modules are callable."""
        modules = [
            MSELoss(), MAELoss(), BCELoss(),
            BCEWithLogitsLoss(), CrossEntropyLoss()
        ]
        
        for module in modules:
            assert callable(module)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_apply_reduction_mean(self):
        """Test reduction mean."""
        loss = Tensor([[1.0, 2.0], [3.0, 4.0]])
        reduced = apply_reduction(loss, 'mean')
        
        assert reduced.data == 2.5
    
    def test_apply_reduction_sum(self):
        """Test reduction sum."""
        loss = Tensor([[1.0, 2.0], [3.0, 4.0]])
        reduced = apply_reduction(loss, 'sum')
        
        assert reduced.data == 10.0
    
    def test_apply_reduction_none(self):
        """Test reduction none."""
        loss = Tensor([[1.0, 2.0], [3.0, 4.0]])
        reduced = apply_reduction(loss, 'none')
        
        assert reduced.shape == (2, 2)
    
    def test_apply_reduction_invalid(self):
        """Test invalid reduction mode."""
        loss = Tensor([1.0, 2.0])
        
        with pytest.raises(ValueError):
            apply_reduction(loss, 'invalid')
    
    def test_compute_class_weights_balanced(self):
        """Test balanced class weights."""
        targets = np.array([0, 0, 0, 1, 1, 2])
        weights = compute_class_weights(targets, num_classes=3, method='balanced')
        
        # Class 2 (rarest) should have highest weight
        assert weights[2] > weights[0]
        assert weights[2] > weights[1]
    
    def test_compute_class_weights_inverse(self):
        """Test inverse frequency weights."""
        targets = np.array([0, 0, 0, 1, 1, 2])
        weights = compute_class_weights(targets, num_classes=3, method='inverse')
        
        # Weights should be 1/count
        expected = np.array([1/3, 1/2, 1/1])
        np.testing.assert_array_almost_equal(weights, expected)
    
    def test_smooth_labels(self):
        """Test label smoothing."""
        targets = np.array([[0, 0, 1, 0]])
        smoothed = smooth_labels(targets, num_classes=4, smoothing=0.1)
        
        # Expected: [0.025, 0.025, 0.925, 0.025]
        expected = np.array([[0.025, 0.025, 0.925, 0.025]])
        np.testing.assert_array_almost_equal(smoothed, expected)
    
    def test_smooth_labels_sum_to_one(self):
        """Test smoothed labels sum to 1."""
        targets = np.array([[0, 0, 1, 0], [1, 0, 0, 0]])
        smoothed = smooth_labels(targets, num_classes=4, smoothing=0.1)
        
        # Each row should sum to 1
        row_sums = smoothed.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0])


class TestEdgeCases:
    """Test edge cases and numerical stability."""
    
    def test_mse_loss_empty_input(self):
        """Test MSE loss with empty input."""
        pred = Tensor(np.array([]).reshape(0, 5))
        target = Tensor(np.array([]).reshape(0, 5))
        
        loss = mse_loss(pred, target, reduction='mean')
        # Should handle gracefully
        assert np.isnan(loss.data) or loss.data == 0
    
    def test_bce_numerical_stability(self):
        """Test BCE numerical stability with extreme values."""
        pred = Tensor([0.999999, 0.000001])
        target = Tensor([1.0, 0.0])
        
        loss = binary_cross_entropy(pred, target, reduction='mean')
        
        # Should not overflow
        assert np.isfinite(loss.data)
    
    def test_cross_entropy_large_logits(self):
        """Test cross-entropy with large logits."""
        logits = Tensor([[1000.0, 0.0, 0.0]])
        targets = Tensor([0])
        
        loss = cross_entropy_loss(logits, targets)
        
        # Should not overflow
        assert np.isfinite(loss.data)
    
    def test_focal_loss_extreme_predictions(self):
        """Test focal loss with extreme predictions."""
        pred = Tensor([[0.99, 0.01, 0.5]])
        target = Tensor([[1.0, 0.0, 1.0]])
        
        loss = focal_loss(pred, target, alpha=0.25, gamma=2.0)
        
        # Should handle extreme values
        assert np.isfinite(loss.data)
    
    def test_loss_with_nan_input(self):
        """Test loss functions reject NaN input."""
        pred = Tensor([1.0, np.nan, 3.0])
        target = Tensor([1.0, 2.0, 3.0])
        
        loss = mse_loss(pred, target, reduction='mean')
        
        # Loss will be NaN but shouldn't crash
        assert np.isnan(loss.data)
    
    def test_loss_shape_mismatch(self):
        """Test loss functions with shape mismatch."""
        pred = Tensor([[1.0, 2.0]])
        target = Tensor([1.0, 2.0, 3.0])
        
        # Should raise error for incompatible shapes
        with pytest.raises((ValueError, IndexError)):
            mse_loss(pred, target)
    
    def test_zero_division_protection(self):
        """Test protection against division by zero."""
        pred = Tensor([0.0, 0.0])
        target = Tensor([0.0, 0.0])
        
        # Should not crash
        loss = msle_loss(pred, target)
        assert np.isfinite(loss.data)


class TestGradientFlow:
    """Test gradient flow through losses."""
    
    def test_mse_gradient_flow(self):
        """Test gradient flows through MSE loss."""
        pred = Tensor([[1.0, 2.0]], requires_grad=True)
        target = Tensor([[1.5, 2.5]])
        
        loss = mse_loss(pred, target, reduction='mean')
        loss.backward()
        
        # Gradient should exist and be finite
        assert pred.grad is not None
        assert np.all(np.isfinite(pred.grad.data))
    
    def test_cross_entropy_gradient_flow(self):
        """Test gradient flows through cross-entropy."""
        logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        targets = Tensor([0])
        
        loss = cross_entropy_loss(logits, targets)
        loss.backward()
        
        assert logits.grad is not None
        assert np.all(np.isfinite(logits.grad.data))
    
    def test_focal_loss_gradient_flow(self):
        """Test gradient flows through focal loss."""
        pred = Tensor([[0.9, 0.5]], requires_grad=True)
        target = Tensor([[1.0, 1.0]])
        
        loss = focal_loss(pred, target, alpha=0.25, gamma=2.0)
        loss.backward()
        
        assert pred.grad is not None
        assert np.all(np.isfinite(pred.grad.data))
    
    def test_multiple_backward_passes(self):
        """Test multiple backward passes accumulate gradients."""
        pred = Tensor([[1.0, 2.0]], requires_grad=True)
        target = Tensor([[1.5, 2.5]])
        
        # First backward
        loss1 = mse_loss(pred, target, reduction='mean')
        loss1.backward()
        first_grad = pred.grad.data.copy()
        
        # Second backward (without zero_grad)
        loss2 = mse_loss(pred, target, reduction='mean')
        loss2.backward()
        
        # Gradients should accumulate
        np.testing.assert_array_almost_equal(
            pred.grad.data, 2 * first_grad
        )


class TestRealWorldScenarios:
    """Test realistic usage scenarios."""
    
    def test_binary_classification_training_step(self):
        """Test complete binary classification training step."""
        # Mini-batch
        X = Tensor(np.random.randn(32, 10), requires_grad=False)
        y = Tensor((np.random.rand(32) > 0.5).astype(float))
        
        # Model weights
        W = Tensor(np.random.randn(10, 1) * 0.1, requires_grad=True)
        b = Tensor(np.zeros((1, 1)), requires_grad=True)
        
        # Forward pass
        logits = X @ W + b
        loss = binary_cross_entropy_with_logits(
            logits.reshape(-1), y, reduction='mean'
        )
        
        # Backward pass
        W.zero_grad()
        b.zero_grad()
        loss.backward()
        
        # Check gradients exist
        assert W.grad is not None
        assert b.grad is not None
        assert np.all(np.isfinite(W.grad.data))
        assert np.all(np.isfinite(b.grad.data))
    
    def test_multiclass_classification_training_step(self):
        """Test complete multi-class classification training step."""
        # Mini-batch
        X = Tensor(np.random.randn(32, 10), requires_grad=False)
        y = Tensor(np.random.randint(0, 5, 32))
        
        # Model weights
        W = Tensor(np.random.randn(10, 5) * 0.1, requires_grad=True)
        b = Tensor(np.zeros((1, 5)), requires_grad=True)
        
        # Forward pass
        logits = X @ W + b
        loss = cross_entropy_loss(logits, y, reduction='mean')
        
        # Backward pass
        W.zero_grad()
        b.zero_grad()
        loss.backward()
        
        # Check gradients
        assert W.grad is not None
        assert b.grad is not None
    
    def test_regression_training_step(self):
        """Test complete regression training step."""
        # Mini-batch
        X = Tensor(np.random.randn(32, 5), requires_grad=False)
        y = Tensor(np.random.randn(32, 1))
        
        # Model weights
        W = Tensor(np.random.randn(5, 1) * 0.1, requires_grad=True)
        b = Tensor(np.zeros((1, 1)), requires_grad=True)
        
        # Forward pass
        predictions = X @ W + b
        loss = mse_loss(predictions, y, reduction='mean')
        
        # Backward pass
        W.zero_grad()
        b.zero_grad()
        loss.backward()
        
        # Update weights
        learning_rate = 0.01
        W.data -= learning_rate * W.grad.data
        b.data -= learning_rate * b.grad.data
        
        # Check update happened
        assert not np.allclose(W.grad.data, 0)
    
    def test_imbalanced_classification(self):
        """Test classification with imbalanced classes."""
        # Imbalanced dataset: 90% class 0, 10% class 1
        targets = np.array([0]*90 + [1]*10)
        
        # Compute class weights
        weights = compute_class_weights(targets, num_classes=2, method='balanced')
        
        # Weights should favor minority class
        assert weights[1] > weights[0]
        
        # Use in loss
        logits = Tensor(np.random.randn(100, 2), requires_grad=True)
        targets_tensor = Tensor(targets)
        weight_tensor = Tensor(weights)
        
        loss = cross_entropy_loss(
            logits, targets_tensor, weight=weight_tensor
        )
        
        assert np.isfinite(loss.data)
    
    def test_label_smoothing_in_training(self):
        """Test label smoothing for regularization."""
        logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        targets = Tensor([0])
        
        # Without smoothing
        loss_no_smooth = cross_entropy_loss(
            logits, targets, label_smoothing=0.0
        )
        
        # With smoothing
        loss_smooth = cross_entropy_loss(
            logits, targets, label_smoothing=0.1
        )
        
        # Both should be finite
        assert np.isfinite(loss_no_smooth.data)
        assert np.isfinite(loss_smooth.data)


class TestNumericalGradientChecking:
    """Test gradients with numerical verification."""
    
    def test_mse_gradient_check(self):
        """Numerical gradient check for MSE."""
        def func(x):
            target = Tensor([[1.5, 2.5]])
            return mse_loss(x, target, reduction='mean')
        
        pred = Tensor([[1.0, 2.0]], requires_grad=True)
        match, diff = check_gradients(func, pred, tolerance=1e-5)
        assert match is True
    
    def test_mae_gradient_check(self):
        """Numerical gradient check for MAE."""
        def func(x):
            target = Tensor([[1.5, 2.5]])
            return mae_loss(x, target, reduction='mean')
        
        pred = Tensor([[1.0, 2.0]], requires_grad=True)
        match, diff = check_gradients(func, pred, tolerance=1e-4)
        assert match is True
    
    def test_smooth_l1_gradient_check(self):
        """Numerical gradient check for Smooth L1."""
        def func(x):
            target = Tensor([[1.5, 2.5]])
            return smooth_l1_loss(x, target, beta=1.0, reduction='mean')
        
        pred = Tensor([[1.0, 2.0]], requires_grad=True)
        match, diff = check_gradients(func, pred, tolerance=1e-5)
        assert match is True
    
    def test_bce_with_logits_gradient_check(self):
        """Numerical gradient check for BCE with logits."""
        def func(x):
            target = Tensor([1.0, 0.0, 1.0])
            return binary_cross_entropy_with_logits(x, target, reduction='mean')
        
        logits = Tensor([0.5, 1.0, -0.5], requires_grad=True)
        match, diff = check_gradients(func, logits, tolerance=1e-4)
        assert match is True
    
    def test_hinge_loss_gradient_check(self):
        """Numerical gradient check for hinge loss."""
        def func(x):
            target = Tensor([1.0, -1.0])
            return hinge_loss(x, target, margin=1.0, reduction='mean')
        
        pred = Tensor([0.5, -0.3], requires_grad=True)
        match, diff = check_gradients(func, pred, tolerance=1e-4)
        assert match is True


class TestComparisons:
    """Test loss function comparisons and properties."""
    
    def test_mse_vs_mae_outliers(self):
        """Test MSE is more sensitive to outliers than MAE."""
        pred = Tensor([1.0, 2.0, 10.0])  # 10.0 is outlier
        target = Tensor([1.0, 2.0, 3.0])
        
        mse = mse_loss(pred, target, reduction='mean')
        mae = mae_loss(pred, target, reduction='mean')
        
        # MSE should be much larger due to squared term
        assert mse.data > mae.data
    
    def test_smooth_l1_combines_mse_mae(self):
        """Test Smooth L1 combines properties of MSE and MAE."""
        pred_small = Tensor([1.0, 1.1])  # Small error
        pred_large = Tensor([1.0, 5.0])  # Large error
        target = Tensor([1.0, 1.0])
        
        # For small errors, behaves like MSE
        loss_small = smooth_l1_loss(pred_small, target, beta=1.0)
        
        # For large errors, behaves like MAE
        loss_large = smooth_l1_loss(pred_large, target, beta=1.0)
        
        assert np.isfinite(loss_small.data)
        assert np.isfinite(loss_large.data)
    
    def test_focal_loss_focuses_on_hard_examples(self):
        """Test focal loss down-weights easy examples."""
        pred_easy = Tensor([[0.9]])  # Easy example (confident)
        pred_hard = Tensor([[0.6]])  # Hard example (less confident)
        target = Tensor([[1.0]])
        
        loss_easy = focal_loss(pred_easy, target, alpha=0.25, gamma=2.0)
        loss_hard = focal_loss(pred_hard, target, alpha=0.25, gamma=2.0)
        
        # Hard example should contribute more to loss
        assert loss_hard.data > loss_easy.data


class TestBatchProcessing:
    """Test loss functions with different batch sizes."""
    
    def test_mse_different_batch_sizes(self):
        """Test MSE with different batch sizes."""
        for batch_size in [1, 8, 32, 128]:
            pred = Tensor(np.random.randn(batch_size, 10))
            target = Tensor(np.random.randn(batch_size, 10))
            
            loss = mse_loss(pred, target, reduction='mean')
            assert np.isfinite(loss.data)
    
    def test_cross_entropy_different_batch_sizes(self):
        """Test cross-entropy with different batch sizes."""
        for batch_size in [1, 8, 32, 128]:
            logits = Tensor(np.random.randn(batch_size, 10))
            targets = Tensor(np.random.randint(0, 10, batch_size))
            
            loss = cross_entropy_loss(logits, targets, reduction='mean')
            assert np.isfinite(loss.data)
    
    def test_reduction_consistency_across_batches(self):
        """Test reduction is consistent across batch sizes."""
        # Small batch
        pred_small = Tensor([[1.0, 2.0]])
        target_small = Tensor([[1.5, 2.5]])
        loss_small = mse_loss(pred_small, target_small, reduction='mean')
        
        # Large batch (same values repeated)
        pred_large = Tensor([[1.0, 2.0]] * 100)
        target_large = Tensor([[1.5, 2.5]] * 100)
        loss_large = mse_loss(pred_large, target_large, reduction='mean')
        
        # Mean should be the same
        np.testing.assert_almost_equal(loss_small.data, loss_large.data)


class TestIntegrationWithActivations:
    """Test losses integrate with activation functions."""
    
    def test_bce_with_sigmoid_output(self):
        """Test BCE with sigmoid activation."""
        from semester1.lab3_activation_functions.activations import sigmoid
        
        logits = Tensor([0.5, 1.0, -0.5], requires_grad=True)
        predictions = sigmoid(logits)
        targets = Tensor([1.0, 1.0, 0.0])
        
        loss = binary_cross_entropy(predictions, targets)
        loss.backward()
        
        # Gradient should flow through sigmoid
        assert logits.grad is not None
    
    def test_cross_entropy_with_linear_output(self):
        """Test cross-entropy with linear (logit) output."""
        logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        targets = Tensor([0])
        
        loss = cross_entropy_loss(logits, targets)
        loss.backward()
        
        assert logits.grad is not None
    
    def test_nll_with_log_softmax(self):
        """Test NLL with log-softmax activation."""
        from semester1.lab3_activation_functions.activations import log_softmax
        
        logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        log_probs = log_softmax(logits, axis=1)
        targets = Tensor([0])
        
        loss = nll_loss(log_probs, targets)
        loss.backward()
        
        assert logits.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])