"""Comprehensive unit tests for activation functions."""

import pytest
import numpy as np
from semester1.lab2_autograd.autograd import Tensor, check_gradients
from semester1.lab3_activation_functions.activations import (
    # Functional interface
    relu, leaky_relu, elu, sigmoid, tanh,
    softmax, log_softmax, gelu, swish, mish,
    softplus, softsign, hard_sigmoid, hard_tanh,
    # Module interface
    ReLU, LeakyReLU, ELU, Sigmoid, Tanh,
    Softmax, LogSoftmax, GELU, Swish, Mish,
    Softplus, Softsign, HardSigmoid, HardTanh, PReLU,
    # Utilities
    check_dead_neurons, activation_statistics, compare_activations,
)


class TestReLUActivation:
    """Test ReLU activation function."""
    
    def test_relu_forward(self):
        """Test ReLU forward pass."""
        x = Tensor([-2, -1, 0, 1, 2])
        y = relu(x)
        expected = np.array([0, 0, 0, 1, 2], dtype=np.float32)
        np.testing.assert_array_equal(y.data, expected)
    
    def test_relu_gradient(self):
        """Test ReLU gradient computation."""
        x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        y = relu(x)
        y.sum().backward()
        expected_grad = np.array([0, 0, 0, 1, 1], dtype=np.float32)
        np.testing.assert_array_equal(x.grad.data, expected_grad)
    
    def test_relu_gradient_check(self):
        """Test ReLU gradient with numerical verification."""
        def func(x):
            return relu(x).sum()
        
        x = Tensor([0.5, 1.5, -0.5], requires_grad=True)
        match, diff = check_gradients(func, x, tolerance=1e-4)
        assert match is True
    
    def test_relu_2d(self):
        """Test ReLU on 2D tensor."""
        x = Tensor([[-1, 0, 1], [2, -2, 3]], requires_grad=True)
        y = relu(x)
        expected = np.array([[0, 0, 1], [2, 0, 3]], dtype=np.float32)
        np.testing.assert_array_equal(y.data, expected)
    
    def test_relu_module(self):
        """Test ReLU module interface."""
        relu_mod = ReLU()
        x = Tensor([[-1, 0, 1]], requires_grad=True)
        y = relu_mod(x)
        expected = np.array([[0, 0, 1]], dtype=np.float32)
        np.testing.assert_array_equal(y.data, expected)


class TestLeakyReLUActivation:
    """Test Leaky ReLU activation function."""
    
    def test_leaky_relu_forward(self):
        """Test Leaky ReLU forward pass."""
        x = Tensor([-2, -1, 0, 1, 2])
        y = leaky_relu(x, alpha=0.1)
        expected = np.array([-0.2, -0.1, 0, 1, 2], dtype=np.float32)
        np.testing.assert_array_almost_equal(y.data, expected)
    
    def test_leaky_relu_gradient(self):
        """Test Leaky ReLU gradient."""
        x = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        y = leaky_relu(x, alpha=0.1)
        y.sum().backward()
        expected_grad = np.array([0.1, 0.1, 0.1, 1, 1], dtype=np.float32)
        np.testing.assert_array_almost_equal(x.grad.data, expected_grad)
    
    def test_leaky_relu_no_dead_neurons(self):
        """Test that Leaky ReLU has no dead neurons."""
        x = Tensor(np.random.randn(100, 64) - 5)  # Very negative
        y = leaky_relu(x, alpha=0.01)
        dead_pct = check_dead_neurons(y)
        assert dead_pct == 0.0
    
    def test_leaky_relu_module(self):
        """Test Leaky ReLU module."""
        leaky = LeakyReLU(alpha=0.2)
        x = Tensor([[-1, 0, 1]], requires_grad=True)
        y = leaky(x)
        expected = np.array([[-0.2, 0, 1]], dtype=np.float32)
        np.testing.assert_array_almost_equal(y.data, expected)


class TestELUActivation:
    """Test ELU activation function."""
    
    def test_elu_forward_positive(self):
        """Test ELU forward for positive values."""
        x = Tensor([0, 1, 2])
        y = elu(x, alpha=1.0)
        expected = np.array([0, 1, 2], dtype=np.float32)
        np.testing.assert_array_almost_equal(y.data, expected)
    
    def test_elu_forward_negative(self):
        """Test ELU forward for negative values."""
        x = Tensor([-1, -2])
        y = elu(x, alpha=1.0)
        # ELU: α(exp(x) - 1) for x < 0
        expected = 1.0 * (np.exp([-1, -2]) - 1)
        np.testing.assert_array_almost_equal(y.data, expected, decimal=3)
    
    def test_elu_gradient(self):
        """Test ELU gradient."""
        x = Tensor([1.0, -1.0], requires_grad=True)
        y = elu(x, alpha=1.0)
        y.sum().backward()
        
        # Gradient: 1 for x>0, α*exp(x) for x<0
        assert x.grad.data[0] == 1.0  # Positive side
        assert np.testing.assert_almost_equal(x.grad.data[1], np.exp(-1.0), decimal=3)  # Negative side
    
    def test_elu_module(self):
        """Test ELU module."""
        elu_mod = ELU(alpha=1.0)
        x = Tensor([[-1, 0, 1]], requires_grad=True)
        y = elu_mod(x)
        assert y.shape == (1, 3)


class TestSigmoidActivation:
    """Test Sigmoid activation function."""
    
    def test_sigmoid_forward(self):
        """Test Sigmoid forward pass."""
        x = Tensor([0])
        y = sigmoid(x)
        assert abs(y.data[0] - 0.5) < 1e-6
    
    def test_sigmoid_range(self):
        """Test Sigmoid output range."""
        x = Tensor([-10, -5, 0, 5, 10])
        y = sigmoid(x)
        assert np.all(y.data > 0)
        assert np.all(y.data < 1)
    
    def test_sigmoid_gradient(self):
        """Test Sigmoid gradient: σ'(x) = σ(x)(1-σ(x))."""
        x = Tensor([0.0], requires_grad=True)
        y = sigmoid(x)
        y.backward()
        # At x=0, σ(0)=0.5, σ'(0)=0.5*0.5=0.25
        assert abs(x.grad.data[0] - 0.25) < 1e-6
    
    def test_sigmoid_numerical_stability(self):
        """Test Sigmoid numerical stability for large values."""
        x = Tensor([-100, -50, 50, 100])
        y = sigmoid(x)
        assert np.all(np.isfinite(y.data))
        assert np.all(y.data >= 0)
        assert np.all(y.data <= 1)
    
    def test_sigmoid_gradient_check(self):
        """Test Sigmoid gradient numerically."""
        def func(x):
            return sigmoid(x).sum()
        
        x = Tensor([0.5, 1.0, -0.5], requires_grad=True)
        match, diff = check_gradients(func, x, tolerance=1e-4)
        assert match is True
    
    def test_sigmoid_module(self):
        """Test Sigmoid module."""
        sig = Sigmoid()
        x = Tensor([[0, 1, -1]], requires_grad=True)
        y = sig(x)
        assert y.shape == (1, 3)
        assert np.all(y.data > 0) and np.all(y.data < 1)


class TestTanhActivation:
    """Test Tanh activation function."""
    
    def test_tanh_forward(self):
        """Test Tanh forward pass."""
        x = Tensor([0])
        y = tanh(x)
        assert abs(y.data[0]) < 1e-6
    
    def test_tanh_range(self):
        """Test Tanh output range."""
        x = Tensor([-10, -5, 0, 5, 10])
        y = tanh(x)
        assert np.all(y.data > -1)
        assert np.all(y.data < 1)
    
    def test_tanh_gradient(self):
        """Test Tanh gradient: tanh'(x) = 1 - tanh²(x)."""
        x = Tensor([0.0], requires_grad=True)
        y = tanh(x)
        y.backward()
        # At x=0, tanh(0)=0, tanh'(0)=1
        assert abs(x.grad.data[0] - 1.0) < 1e-6
    
    def test_tanh_symmetry(self):
        """Test Tanh symmetry: tanh(-x) = -tanh(x)."""
        x = Tensor([1.0, 2.0, 3.0])
        y_pos = tanh(x)
        y_neg = tanh(-x)
        np.testing.assert_array_almost_equal(y_pos.data, -y_neg.data)
    
    def test_tanh_module(self):
        """Test Tanh module."""
        tanh_mod = Tanh()
        x = Tensor([[0, 1, -1]], requires_grad=True)
        y = tanh_mod(x)
        assert y.shape == (1, 3)


class TestSoftmaxActivation:
    """Test Softmax activation function."""
    
    def test_softmax_forward(self):
        """Test Softmax forward pass."""
        x = Tensor([[1, 2, 3]])
        y = softmax(x, axis=1)
        # Should sum to 1
        assert abs(y.data.sum() - 1.0) < 1e-6
    
    def test_softmax_properties(self):
        """Test Softmax properties."""
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        y = softmax(x, axis=1)
        # Each row should sum to 1
        row_sums = y.data.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0])
        # All values should be positive
        assert np.all(y.data > 0)
    
    def test_softmax_gradient(self):
        """Test Softmax gradient."""
        x = Tensor([[1, 2, 3]], requires_grad=True)
        y = softmax(x, axis=1)
        loss = (y ** 2).sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_softmax_numerical_stability(self):
        """Test Softmax numerical stability."""
        x = Tensor([[100, 200, 300]])
        y = softmax(x, axis=1)
        assert np.all(np.isfinite(y.data))
        assert abs(y.data.sum() - 1.0) < 1e-6
    
    def test_softmax_2d(self):
        """Test Softmax on 2D tensor."""
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        y = softmax(x, axis=1)
        assert y.shape == (2, 2)
        np.testing.assert_array_almost_equal(y.data.sum(axis=1), [1.0, 1.0])
    
    def test_softmax_module(self):
        """Test Softmax module."""
        softmax_mod = Softmax(axis=1)
        x = Tensor([[1, 2, 3]], requires_grad=True)
        y = softmax_mod(x)
        assert abs(y.data.sum() - 1.0) < 1e-6


class TestLogSoftmaxActivation:
    """Test Log-Softmax activation function."""
    
    def test_log_softmax_forward(self):
        """Test Log-Softmax forward pass."""
        x = Tensor([[1, 2, 3]])
        y = log_softmax(x, axis=1)
        # All values should be negative
        assert np.all(y.data <= 0)
    
    def test_log_softmax_equivalence(self):
        """Test Log-Softmax equivalence to log(softmax)."""
        x = Tensor([[1, 2, 3]])
        y1 = log_softmax(x, axis=1)
        y2_soft = softmax(x, axis=1)
        y2 = Tensor(np.log(y2_soft.data + 1e-10))
        np.testing.assert_array_almost_equal(y1.data, y2.data, decimal=5)
    
    def test_log_softmax_numerical_stability(self):
        """Test Log-Softmax numerical stability."""
        x = Tensor([[100, 200, 300]])
        y = log_softmax(x, axis=1)
        assert np.all(np.isfinite(y.data))
    
    def test_log_softmax_gradient(self):
        """Test Log-Softmax gradient."""
        x = Tensor([[1, 2, 3]], requires_grad=True)
        y = log_softmax(x, axis=1)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_log_softmax_module(self):
        """Test Log-Softmax module."""
        log_softmax_mod = LogSoftmax(axis=1)
        x = Tensor([[1, 2, 3]], requires_grad=True)
        y = log_softmax_mod(x)
        assert np.all(y.data <= 0)


class TestGELUActivation:
    """Test GELU activation function."""
    
    def test_gelu_forward(self):
        """Test GELU forward pass."""
        x = Tensor([0])
        y = gelu(x)
        # GELU(0) ≈ 0
        assert abs(y.data[0]) < 0.1
    
    def test_gelu_positive_values(self):
        """Test GELU for positive values."""
        x = Tensor([1, 2, 3])
        y = gelu(x)
        # GELU(x) ≈ x for large positive x
        assert np.all(y.data > 0)
    
    def test_gelu_gradient(self):
        """Test GELU gradient."""
        x = Tensor([0.5, 1.0, 1.5], requires_grad=True)
        y = gelu(x)
        y.sum().backward()
        assert x.grad is not None
        assert np.all(x.grad.data > 0)  # Positive gradients
    
    def test_gelu_module(self):
        """Test GELU module."""
        gelu_mod = GELU()
        x = Tensor([[-1, 0, 1]], requires_grad=True)
        y = gelu_mod(x)
        assert y.shape == (1, 3)


class TestSwishActivation:
    """Test Swish/SiLU activation function."""
    
    def test_swish_forward(self):
        """Test Swish forward pass."""
        x = Tensor([0])
        y = swish(x)
        # Swish(0) = 0 * sigmoid(0) = 0
        assert abs(y.data[0]) < 1e-6
    
    def test_swish_properties(self):
        """Test Swish properties."""
        x = Tensor([1, 2, 3])
        y = swish(x)
        # Swish is unbounded above
        assert np.all(y.data > 0)
    
    def test_swish_gradient(self):
        """Test Swish gradient."""
        x = Tensor([0.5, 1.0, 1.5], requires_grad=True)
        y = swish(x)
        y.sum().backward()
        assert x.grad is not None
    
    def test_swish_module(self):
        """Test Swish module."""
        swish_mod = Swish(beta=1.0)
        x = Tensor([[-1, 0, 1]], requires_grad=True)
        y = swish_mod(x)
        assert y.shape == (1, 3)


class TestMishActivation:
    """Test Mish activation function."""
    
    def test_mish_forward(self):
        """Test Mish forward pass."""
        x = Tensor([0])
        y = mish(x)
        # Mish(0) ≈ 0
        assert abs(y.data[0]) < 0.1
    
    def test_mish_positive_values(self):
        """Test Mish for positive values."""
        x = Tensor([1, 2, 3])
        y = mish(x)
        assert np.all(y.data > 0)
    
    def test_mish_gradient(self):
        """Test Mish gradient."""
        x = Tensor([0.5, 1.0, 1.5], requires_grad=True)
        y = mish(x)
        y.sum().backward()
        assert x.grad is not None
    
    def test_mish_module(self):
        """Test Mish module."""
        mish_mod = Mish()
        x = Tensor([[-1, 0, 1]], requires_grad=True)
        y = mish_mod(x)
        assert y.shape == (1, 3)


class TestSoftplusActivation:
    """Test Softplus activation function."""
    
    def test_softplus_forward(self):
        """Test Softplus forward pass."""
        x = Tensor([0])
        y = softplus(x)
        # Softplus(0) = log(2) ≈ 0.693
        assert abs(y.data[0] - np.log(2)) < 1e-3
    
    def test_softplus_positive(self):
        """Test Softplus always positive."""
        x = Tensor([-10, -5, 0, 5, 10])
        y = softplus(x)
        assert np.all(y.data > 0)
    
    def test_softplus_approximates_relu(self):
        """Test Softplus approximates ReLU for large values."""
        x = Tensor([10])
        y = softplus(x)
        # For large x, softplus(x) ≈ x
        assert abs(y.data[0] - 10.0) < 0.1
    
    def test_softplus_gradient(self):
        """Test Softplus gradient is sigmoid."""
        x = Tensor([0.0], requires_grad=True)
        y = softplus(x)
        y.backward()
        # softplus'(0) = sigmoid(0) = 0.5
        assert abs(x.grad.data[0] - 0.5) < 1e-3
    
    def test_softplus_module(self):
        """Test Softplus module."""
        softplus_mod = Softplus(beta=1.0)
        x = Tensor([[-1, 0, 1]], requires_grad=True)
        y = softplus_mod(x)
        assert np.all(y.data > 0)


class TestSoftsignActivation:
    """Test Softsign activation function."""
    
    def test_softsign_forward(self):
        """Test Softsign forward pass."""
        x = Tensor([0])
        y = softsign(x)
        assert abs(y.data[0]) < 1e-6
    
    def test_softsign_range(self):
        """Test Softsign output range."""
        x = Tensor([-10, -5, 0, 5, 10])
        y = softsign(x)
        assert np.all(y.data > -1)
        assert np.all(y.data < 1)
    
    def test_softsign_gradient(self):
        """Test Softsign gradient."""
        x = Tensor([1.0], requires_grad=True)
        y = softsign(x)
        y.backward()
        # softsign'(1) = 1/(1+|1|)² = 0.25
        assert abs(x.grad.data[0] - 0.25) < 1e-3
    
    def test_softsign_module(self):
        """Test Softsign module."""
        softsign_mod = Softsign()
        x = Tensor([[-2, -1, 0, 1, 2]], requires_grad=True)
        y = softsign_mod(x)
        assert np.all(y.data > -1) and np.all(y.data < 1)


class TestHardSigmoidActivation:
    """Test Hard Sigmoid activation function."""
    
    def test_hard_sigmoid_forward(self):
        """Test Hard Sigmoid forward pass."""
        x = Tensor([-2, -1, 0, 1, 2])
        y = hard_sigmoid(x)
        expected = np.array([0, 0, 0.5, 1, 1], dtype=np.float32)
        np.testing.assert_array_almost_equal(y.data, expected)
    
    def test_hard_sigmoid_range(self):
        """Test Hard Sigmoid output range."""
        x = Tensor([-10, -5, 0, 5, 10])
        y = hard_sigmoid(x)
        assert np.all(y.data >= 0)
        assert np.all(y.data <= 1)
    
    def test_hard_sigmoid_gradient(self):
        """Test Hard Sigmoid gradient."""
        x = Tensor([0.0], requires_grad=True)
        y = hard_sigmoid(x)
        y.backward()
        # Gradient in linear region is 0.5
        assert abs(x.grad.data[0] - 0.5) < 1e-6
    
    def test_hard_sigmoid_module(self):
        """Test Hard Sigmoid module."""
        hard_sig = HardSigmoid()
        x = Tensor([[-2, -1, 0, 1, 2]], requires_grad=True)
        y = hard_sig(x)
        assert np.all(y.data >= 0) and np.all(y.data <= 1)


class TestHardTanhActivation:
    """Test Hard Tanh activation function."""
    
    def test_hard_tanh_forward(self):
        """Test Hard Tanh forward pass."""
        x = Tensor([-2, -1, 0, 1, 2])
        y = hard_tanh(x)
        expected = np.array([-1, -1, 0, 1, 1], dtype=np.float32)
        np.testing.assert_array_equal(y.data, expected)
    
    def test_hard_tanh_range(self):
        """Test Hard Tanh output range."""
        x = Tensor([-10, -5, 0, 5, 10])
        y = hard_tanh(x)
        assert np.all(y.data >= -1)
        assert np.all(y.data <= 1)
    
    def test_hard_tanh_gradient(self):
        """Test Hard Tanh gradient."""
        x = Tensor([0.0], requires_grad=True)
        y = hard_tanh(x)
        y.backward()
        # Gradient in linear region is 1
        assert x.grad.data[0] == 1.0
    
    def test_hard_tanh_gradient_saturated(self):
        """Test Hard Tanh gradient in saturated region."""
        x = Tensor([2.0], requires_grad=True)
        y = hard_tanh(x)
        y.backward()
        # Gradient in saturated region is 0
        assert x.grad.data[0] == 0.0
    
    def test_hard_tanh_module(self):
        """Test Hard Tanh module."""
        hard_tanh_mod = HardTanh()
        x = Tensor([[-2, -1, 0, 1, 2]], requires_grad=True)
        y = hard_tanh_mod(x)
        assert np.all(y.data >= -1) and np.all(y.data <= 1)


class TestPReLUActivation:
    """Test Parametric ReLU activation function."""
    
    def test_prelu_forward(self):
        """Test PReLU forward pass."""
        prelu = PReLU(alpha=0.25)
        x = Tensor([-2, -1, 0, 1, 2])
        y = prelu(x)
        expected = np.array([-0.5, -0.25, 0, 1, 2], dtype=np.float32)
        np.testing.assert_array_almost_equal(y.data, expected)
    
    def test_prelu_learnable_parameter(self):
        """Test PReLU has learnable parameter."""
        prelu = PReLU(alpha=0.25)
        assert prelu.alpha.requires_grad is True
    
    def test_prelu_gradient_x(self):
        """Test PReLU gradient w.r.t. input."""
        prelu = PReLU(alpha=0.25)
        x = Tensor([-1, 0, 1], requires_grad=True)
        y = prelu(x)
        y.sum().backward()
        expected_grad = np.array([0.25, 0.25, 1], dtype=np.float32)
        np.testing.assert_array_almost_equal(x.grad.data, expected_grad)
    
    def test_prelu_gradient_alpha(self):
        """Test PReLU gradient w.r.t. alpha."""
        prelu = PReLU(alpha=0.25)
        x = Tensor([-2, -1, 0, 1, 2])
        y = prelu(x)
        y.sum().backward()
        # Alpha gradient should be sum of negative values
        assert prelu.alpha.grad is not None
    
    def test_prelu_module_repr(self):
        """Test PReLU string representation."""
        prelu = PReLU(alpha=0.25)
        repr_str = repr(prelu)
        assert "PReLU" in repr_str
        assert "0.25" in repr_str


class TestEdgeCases:
    """Test edge cases and numerical stability."""
    
    def test_large_positive_values(self):
        """Test activations with large positive values."""
        x = Tensor([100, 200, 300])
        
        # Sigmoid should not overflow
        y_sig = sigmoid(x)
        assert np.all(np.isfinite(y_sig.data))
        
        # Softmax should not overflow
        y_soft = softmax(x.reshape(1, 3), axis=1)
        assert np.all(np.isfinite(y_soft.data))
    
    def test_large_negative_values(self):
        """Test activations with large negative values."""
        x = Tensor([-100, -200, -300])
        
        # Sigmoid should not underflow
        y_sig = sigmoid(x)
        assert np.all(np.isfinite(y_sig.data))
        assert np.all(y_sig.data >= 0)
    
    def test_zero_input(self):
        """Test activations with zero input."""
        x = Tensor([0])
        
        activations = [relu, leaky_relu, elu, sigmoid, tanh, gelu, swish, mish]
        for act_fn in activations:
            y = act_fn(x)
            assert np.isfinite(y.data[0])
    
    def test_batch_processing(self):
        """Test activations with batch input."""
        x = Tensor(np.random.randn(32, 64))
        
        # All activations should handle batches
        y_relu = relu(x)
        assert y_relu.shape == (32, 64)
        
        y_sigmoid = sigmoid(x)
        assert y_sigmoid.shape == (32, 64)
    
    def test_gradient_flow_deep_network(self):
        """Test gradient flow through multiple activations."""
        x = Tensor(np.random.randn(10, 20), requires_grad=True)
        
        # Stack multiple activations
        y = relu(x)
        y = sigmoid(y)
        y = tanh(y)
        loss = y.sum()
        
        loss.backward()
        assert x.grad is not None
        assert np.all(np.isfinite(x.grad.data))


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_compare_activations(self):
        """Test activation comparison utility."""
        results = compare_activations(
            x_range=(-5, 5),
            num_points=100,
            activations=['relu', 'sigmoid', 'tanh']
        )
        
        assert 'relu' in results
        assert 'sigmoid' in results
        assert 'tanh' in results
        
        for name, (x, y) in results.items():
            assert len(x) == 100
            assert len(y) == 100
    
    def test_check_dead_neurons(self):
        """Test dead neuron detection."""
        # All negative input -> 100% dead with ReLU
        x = Tensor(np.ones((100, 64)) * -1)
        y = relu(x)
        dead_pct = check_dead_neurons(y)
        assert dead_pct == 100.0
        
        # All positive input -> 0% dead
        x = Tensor(np.ones((100, 64)))
        y = relu(x)
        dead_pct = check_dead_neurons(y)
        assert dead_pct == 0.0
    
    def test_activation_statistics(self):
        """Test activation statistics computation."""
        x = Tensor(np.random.randn(1000, 128))
        y = relu(x)
        stats = activation_statistics(y)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'dead_pct' in stats
        
        # ReLU output should have non-negative values
        assert stats['min'] >= 0


class TestModuleInterface:
    """Test module interface consistency."""
    
    def test_all_modules_callable(self):
        """Test that all activation modules are callable."""
        modules = [
            ReLU(), LeakyReLU(), ELU(), Sigmoid(), Tanh(),
            Softmax(), LogSoftmax(), GELU(), Swish(), Mish(),
            Softplus(), Softsign(), HardSigmoid(), HardTanh(), PReLU()
        ]
        
        x = Tensor([[1, 2, 3]], requires_grad=True)
        
        for module in modules:
            y = module(x)
            assert isinstance(y, Tensor)
    
    def test_module_repr(self):
        """Test module string representations."""
        relu_mod = ReLU()
        assert "ReLU" in repr(relu_mod)
        
        leaky_mod = LeakyReLU(alpha=0.1)
        assert "LeakyReLU" in repr(leaky_mod)
        assert "0.1" in repr(leaky_mod)
    
    def test_module_consistency_with_functional(self):
        """Test module interface gives same results as functional."""
        x = Tensor([[1, 2, 3]], requires_grad=True)
        
        # ReLU
        y_func = relu(x)
        y_mod = ReLU()(x)
        np.testing.assert_array_equal(y_func.data, y_mod.data)
        
        # Sigmoid
        y_func = sigmoid(x)
        y_mod = Sigmoid()(x)
        np.testing.assert_array_almost_equal(y_func.data, y_mod.data)


class TestGradientChecking:
    """Test gradient computation with numerical verification."""
    
    def test_relu_gradient_check(self):
        """Numerical gradient check for ReLU."""
        def func(x):
            return relu(x).sum()
        
        x = Tensor([0.5, 1.5, 2.5], requires_grad=True)
        match, diff = check_gradients(func, x, tolerance=1e-4)
        assert match is True
    
    def test_sigmoid_gradient_check(self):
        """Numerical gradient check for Sigmoid."""
        def func(x):
            return sigmoid(x).sum()
        
        x = Tensor([0.5, 1.0, 1.5], requires_grad=True)
        match, diff = check_gradients(func, x, tolerance=1e-4)
        assert match is True
    
    def test_tanh_gradient_check(self):
        """Numerical gradient check for Tanh."""
        def func(x):
            return tanh(x).sum()
        
        x = Tensor([0.5, 1.0, 1.5], requires_grad=True)
        match, diff = check_gradients(func, x, tolerance=1e-4)
        assert match is True
    
    def test_softmax_gradient_check(self):
        """Numerical gradient check for Softmax."""
        def func(x):
            return softmax(x, axis=1).sum()
        
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        match, diff = check_gradients(func, x, tolerance=1e-4)
        assert match is True
    
    def test_gelu_gradient_check(self):
        """Numerical gradient check for GELU."""
        def func(x):
            return gelu(x).sum()
        
        x = Tensor([0.5, 1.0, 1.5], requires_grad=True)
        match, diff = check_gradients(func, x, tolerance=1e-3)
        assert match is True


class TestComplexScenarios:
    """Test complex usage scenarios."""
    
    def test_multi_layer_network(self):
        """Test activations in multi-layer network."""
        # Simple 2-layer network
        x = Tensor(np.random.randn(16, 10), requires_grad=False)
        W1 = Tensor(np.random.randn(10, 20) * 0.1, requires_grad=True)
        W2 = Tensor(np.random.randn(20, 5) * 0.1, requires_grad=True)
        
        # Forward pass
        h1 = x @ W1
        a1 = relu(h1)
        h2 = a1 @ W2
        a2 = sigmoid(h2)
        loss = (a2 ** 2).sum()
        
        # Backward pass
        W1.zero_grad()
        W2.zero_grad()
        loss.backward()
        
        # Check gradients exist
        assert W1.grad is not None
        assert W2.grad is not None
        assert np.all(np.isfinite(W1.grad.data))
        assert np.all(np.isfinite(W2.grad.data))
    
    def test_activation_comparison_on_same_input(self):
        """Compare different activations on same input."""
        x = Tensor(np.linspace(-3, 3, 50))
        
        activations = {
            'relu': relu(x),
            'leaky_relu': leaky_relu(x),
            'elu': elu(x),
            'sigmoid': sigmoid(x),
            'tanh': tanh(x),
        }
        
        # All should produce valid outputs
        for name, y in activations.items():
            assert np.all(np.isfinite(y.data))
    
    def test_gradient_accumulation_across_activations(self):
        """Test gradient accumulation with multiple activation paths."""
        x = Tensor([1.0], requires_grad=True)
        
        # Multiple paths with different activations
        y1 = relu(x)
        y2 = sigmoid(x)
        y3 = tanh(x)
        
        # Combine
        y = y1 + y2 + y3
        y.backward()
        
        # Gradient should be sum of individual
        # gradients from each path
        assert x.grad is not None
        assert np.isfinite(x.grad.data[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])