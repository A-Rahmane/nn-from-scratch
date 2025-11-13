"""Comprehensive unit tests for automatic differentiation."""

import pytest
import numpy as np
from semester1.lab2_autograd.autograd import (
    Tensor,
    no_grad,
    enable_grad,
    zero_grad,
    clip_grad_value,
    clip_grad_norm,
    numerical_gradient,
    check_gradients,
)


class TestTensorCreationWithGrad:
    """Test tensor creation with gradient tracking."""
    
    def test_default_requires_grad(self):
        """Test that requires_grad defaults to False."""
        t = Tensor([1, 2, 3])
        assert t.requires_grad is False
        assert t.grad is None
    
    def test_explicit_requires_grad(self):
        """Test explicit requires_grad flag."""
        t = Tensor([1, 2, 3], requires_grad=True)
        assert t.requires_grad is True
        assert t.grad is None
    
    def test_grad_initially_none(self):
        """Test that grad is initially None."""
        t = Tensor([1, 2, 3], requires_grad=True)
        assert t.grad is None
    
    def test_computational_graph_attributes(self):
        """Test that graph tracking attributes exist."""
        t = Tensor([1, 2, 3], requires_grad=True)
        assert hasattr(t, '_prev')
        assert hasattr(t, '_op')
        assert hasattr(t, 'grad_fn')


class TestBasicGradients:
    """Test gradients for basic operations."""
    
    def test_addition_gradient(self):
        """Test gradient of addition."""
        x = Tensor([2.0, 3.0], requires_grad=True)
        y = Tensor([4.0, 5.0], requires_grad=True)
        z = x + y
        z.sum().backward()
        
        np.testing.assert_array_equal(x.grad.data, np.ones(2))
        np.testing.assert_array_equal(y.grad.data, np.ones(2))
    
    def test_subtraction_gradient(self):
        """Test gradient of subtraction."""
        x = Tensor([5.0, 6.0], requires_grad=True)
        y = Tensor([2.0, 3.0], requires_grad=True)
        z = x - y
        z.sum().backward()
        
        np.testing.assert_array_equal(x.grad.data, np.ones(2))
        np.testing.assert_array_equal(y.grad.data, -np.ones(2))
    
    def test_multiplication_gradient(self):
        """Test gradient of multiplication."""
        x = Tensor([2.0, 3.0], requires_grad=True)
        y = Tensor([4.0, 5.0], requires_grad=True)
        z = x * y
        z.sum().backward()
        
        np.testing.assert_array_equal(x.grad.data, np.array([4.0, 5.0]))
        np.testing.assert_array_equal(y.grad.data, np.array([2.0, 3.0]))
    
    def test_division_gradient(self):
        """Test gradient of division."""
        x = Tensor([6.0, 8.0], requires_grad=True)
        y = Tensor([2.0, 4.0], requires_grad=True)
        z = x / y
        z.sum().backward()
        
        # ∂(x/y)/∂x = 1/y
        expected_x_grad = 1.0 / np.array([2.0, 4.0])
        np.testing.assert_array_almost_equal(x.grad.data, expected_x_grad)
        
        # ∂(x/y)/∂y = -x/y^2
        expected_y_grad = -np.array([6.0, 8.0]) / np.array([2.0, 4.0])**2
        np.testing.assert_array_almost_equal(y.grad.data, expected_y_grad)
    
    def test_power_gradient(self):
        """Test gradient of power operation."""
        x = Tensor([2.0, 3.0], requires_grad=True)
        y = x ** 2
        y.sum().backward()
        
        # ∂(x^2)/∂x = 2x
        expected = np.array([4.0, 6.0])
        np.testing.assert_array_almost_equal(x.grad.data, expected)
    
    def test_power_gradient_higher_order(self):
        """Test gradient of higher order power."""
        x = Tensor([2.0, 3.0], requires_grad=True)
        y = x ** 3
        y.sum().backward()
        
        # ∂(x^3)/∂x = 3x^2
        expected = 3 * np.array([2.0, 3.0])**2
        np.testing.assert_array_almost_equal(x.grad.data, expected)
    
    def test_negation_gradient(self):
        """Test gradient of negation."""
        x = Tensor([2.0, 3.0], requires_grad=True)
        y = -x
        y.sum().backward()
        
        np.testing.assert_array_equal(x.grad.data, -np.ones(2))


class TestChainRule:
    """Test chain rule propagation."""
    
    def test_simple_chain(self):
        """Test simple chain: f(x) = (x^2)^2."""
        x = Tensor([2.0], requires_grad=True)
        y = x ** 2  # y = 4
        z = y ** 2  # z = 16
        z.backward()
        
        # dz/dx = dz/dy * dy/dx = 2y * 2x = 2*4 * 2*2 = 32
        expected = np.array([32.0])
        np.testing.assert_array_almost_equal(x.grad.data, expected)
    
    def test_complex_chain(self):
        """Test complex chain: f(x) = (x^2 + 1)^3."""
        x = Tensor([2.0], requires_grad=True)
        y = x ** 2 + 1  # y = 5
        z = y ** 3      # z = 125
        z.backward()
        
        # dz/dx = 3y^2 * 2x = 3*25 * 4 = 300
        expected = np.array([300.0])
        np.testing.assert_array_almost_equal(x.grad.data, expected)
    
    def test_multiple_paths(self):
        """Test gradient accumulation from multiple paths."""
        x = Tensor([2.0], requires_grad=True)
        y1 = x ** 2  # Path 1
        y2 = x * 3   # Path 2
        z = y1 + y2  # Combine
        z.backward()
        
        # dz/dx = dy1/dx + dy2/dx = 2x + 3 = 4 + 3 = 7
        expected = np.array([7.0])
        np.testing.assert_array_almost_equal(x.grad.data, expected)
    
    def test_branching_computation(self):
        """Test branching in computational graph."""
        x = Tensor([3.0], requires_grad=True)
        y = x * x  # Reuse x
        z = y + x
        z.backward()
        
        # dz/dx = dy/dx + 1 = 2x + 1 = 7
        expected = np.array([7.0])
        np.testing.assert_array_almost_equal(x.grad.data, expected)


class TestAggregationGradients:
    """Test gradients for aggregation operations."""
    
    def test_sum_gradient(self):
        """Test gradient of sum."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.sum()
        y.backward()
        
        # Gradient of sum is 1 for all elements
        expected = np.ones((2, 2))
        np.testing.assert_array_equal(x.grad.data, expected)
    
    def test_sum_axis_gradient(self):
        """Test gradient of sum along axis."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.sum(axis=0)
        y.sum().backward()
        
        # Gradient broadcasts back to original shape
        expected = np.ones((2, 2))
        np.testing.assert_array_equal(x.grad.data, expected)
    
    def test_mean_gradient(self):
        """Test gradient of mean."""
        x = Tensor([[2.0, 4.0], [6.0, 8.0]], requires_grad=True)
        y = x.mean()
        y.backward()
        
        # Gradient of mean is 1/n for each element
        expected = np.ones((2, 2)) / 4
        np.testing.assert_array_almost_equal(x.grad.data, expected)
    
    def test_mean_axis_gradient(self):
        """Test gradient of mean along axis."""
        x = Tensor([[2.0, 4.0], [6.0, 8.0]], requires_grad=True)
        y = x.mean(axis=1)
        y.sum().backward()
        
        # Each element contributes 1/2 to its row mean
        expected = np.ones((2, 2)) / 2
        np.testing.assert_array_almost_equal(x.grad.data, expected)
    
    def test_max_gradient(self):
        """Test gradient of max."""
        x = Tensor([[1.0, 5.0], [3.0, 2.0]], requires_grad=True)
        y = x.max()
        y.backward()
        
        # Only maximum element gets gradient
        expected = np.array([[0.0, 1.0], [0.0, 0.0]])
        np.testing.assert_array_equal(x.grad.data, expected)
    
    def test_max_axis_gradient(self):
        """Test gradient of max along axis."""
        x = Tensor([[1.0, 5.0], [3.0, 2.0]], requires_grad=True)
        y = x.max(axis=1)
        y.sum().backward()
        
        # Maximum in each row gets gradient
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_array_equal(x.grad.data, expected)


class TestBroadcastingGradients:
    """Test gradients with broadcasting."""
    
    def test_scalar_addition(self):
        """Test gradient when adding scalar."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x + 5.0
        y.sum().backward()
        
        expected = np.ones((2, 2))
        np.testing.assert_array_equal(x.grad.data, expected)
    
    def test_broadcast_addition(self):
        """Test gradient with broadcast addition."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = Tensor([10.0, 20.0], requires_grad=True)
        z = x + y
        z.sum().backward()
        
        # x gets gradient of ones
        np.testing.assert_array_equal(x.grad.data, np.ones((2, 2)))
        
        # y's gradient is summed over broadcast dimension
        np.testing.assert_array_equal(y.grad.data, np.array([2.0, 2.0]))
    
    def test_broadcast_multiplication(self):
        """Test gradient with broadcast multiplication."""
        x = Tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
        y = Tensor([10.0, 20.0], requires_grad=True)
        z = x * y
        z.sum().backward()
        
        # x gradient is y value
        expected_x = np.array([[10.0, 20.0], [10.0, 20.0]])
        np.testing.assert_array_equal(x.grad.data, expected_x)
        
        # y gradient is sum of x values in each column
        expected_y = np.array([6.0, 8.0])
        np.testing.assert_array_equal(y.grad.data, expected_y)


class TestMatrixGradients:
    """Test gradients for matrix operations."""
    
    def test_matmul_2d(self):
        """Test gradient of 2D matrix multiplication."""
        A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        B = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        C = A @ B
        loss = C.sum()
        loss.backward()
        
        # ∂L/∂A = ones @ B^T
        expected_A = np.ones((2, 2)) @ np.array([[5.0, 6.0], [7.0, 8.0]]).T
        np.testing.assert_array_almost_equal(A.grad.data, expected_A)
        
        # ∂L/∂B = A^T @ ones
        expected_B = np.array([[1.0, 2.0], [3.0, 4.0]]).T @ np.ones((2, 2))
        np.testing.assert_array_almost_equal(B.grad.data, expected_B)
    
    def test_matmul_vector(self):
        """Test gradient of matrix-vector multiplication."""
        A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        x = Tensor([5.0, 6.0], requires_grad=True)
        y = A @ x
        y.sum().backward()
        
        # Gradient shapes should match inputs
        assert A.grad.shape == A.shape
        assert x.grad.shape == x.shape


class TestShapeOperationGradients:
    """Test gradients for shape operations."""
    
    def test_reshape_gradient(self):
        """Test gradient of reshape."""
        x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
        y = x.reshape(2, 3)
        z = (y ** 2).sum()
        z.backward()
        
        # Gradient should match original shape
        assert x.grad.shape == x.shape
        expected = 2 * np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_array_almost_equal(x.grad.data, expected)
    
    def test_transpose_gradient(self):
        """Test gradient of transpose."""
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        y = x.T
        z = (y ** 2).sum()
        z.backward()
        
        # Gradient should match original shape
        assert x.grad.shape == x.shape
        expected = 2 * np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_array_almost_equal(x.grad.data, expected)


class TestGradientAccumulation:
    """Test gradient accumulation."""
    
    def test_multiple_backward_calls(self):
        """Test that gradients accumulate across backward calls."""
        x = Tensor([2.0], requires_grad=True)
        
        # First backward
        y1 = x ** 2
        y1.backward()
        first_grad = x.grad.data.copy()
        
        # Second backward without zero_grad
        y2 = x * 3
        y2.backward()
        
        # Gradient should be sum of both
        expected = first_grad + 3.0
        np.testing.assert_array_almost_equal(x.grad.data, expected)
    
    def test_zero_grad(self):
        """Test zero_grad resets gradients."""
        x = Tensor([2.0], requires_grad=True)
        y = x ** 2
        y.backward()
        
        assert x.grad is not None
        x.zero_grad()
        assert x.grad is None
    
    def test_shared_tensor_gradient(self):
        """Test gradient accumulation for shared tensors."""
        x = Tensor([3.0], requires_grad=True)
        y = x * x * x  # x appears three times
        y.backward()
        
        # dy/dx = 3x^2 = 27
        expected = np.array([27.0])
        np.testing.assert_array_almost_equal(x.grad.data, expected)


class TestContextManagers:
    """Test gradient context managers."""
    
    def test_no_grad_disables_tracking(self):
        """Test that no_grad disables gradient tracking."""
        x = Tensor([1.0], requires_grad=True)
        
        with no_grad():
            y = x * 2
            assert y.requires_grad is False
            assert y.grad_fn is None
    
    def test_no_grad_nested(self):
        """Test nested no_grad contexts."""
        x = Tensor([1.0], requires_grad=True)
        
        with no_grad():
            y = x * 2
            with no_grad():
                z = y * 3
                assert z.requires_grad is False
    
    def test_enable_grad_inside_no_grad(self):
        """Test enable_grad inside no_grad."""
        x = Tensor([1.0], requires_grad=True)
        
        with no_grad():
            y = x * 2
            assert y.requires_grad is False
            
            with enable_grad():
                z = Tensor([3.0], requires_grad=True)
                w = z * 4
                assert w.requires_grad is True
    
    def test_context_manager_restores_state(self):
        """Test that context managers restore previous state."""
        assert Tensor._grad_enabled is True
        
        with no_grad():
            assert Tensor._grad_enabled is False
        
        assert Tensor._grad_enabled is True


class TestDetach:
    """Test detach functionality."""
    
    def test_detach_stops_gradient(self):
        """Test that detach stops gradient flow."""
        x = Tensor([2.0], requires_grad=True)
        y = x ** 2
        z = y.detach()
        w = z * 3
        w.backward()
        
        # x should not receive gradients through detached path
        assert x.grad is None
    
    def test_detach_creates_copy(self):
        """Test that detach creates independent copy."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x.detach()
        
        assert y.requires_grad is False
        assert not np.shares_memory(x.data, y.data)


class TestGradientUtilities:
    """Test gradient utility functions."""
    
    def test_zero_grad_multiple_tensors(self):
        """Test zero_grad with multiple tensors."""
        x = Tensor([1.0], requires_grad=True)
        y = Tensor([2.0], requires_grad=True)
        z = x * y
        z.backward()
        
        assert x.grad is not None
        assert y.grad is not None
        
        zero_grad(x, y)
        
        assert x.grad is None
        assert y.grad is None
    
    def test_clip_grad_value(self):
        """Test gradient value clipping."""
        tensors = [
            Tensor([1.0], requires_grad=True),
            Tensor([2.0], requires_grad=True)
        ]
        
        # Set large gradients
        tensors[0].grad = Tensor([10.0])
        tensors[1].grad = Tensor([-15.0])
        
        clip_grad_value(tensors, 5.0)
        
        assert tensors[0].grad.data[0] == 5.0
        assert tensors[1].grad.data[0] == -5.0
    
    def test_clip_grad_norm(self):
        """Test gradient norm clipping."""
        tensors = [
            Tensor([1.0], requires_grad=True),
            Tensor([2.0], requires_grad=True)
        ]
        
        # Set gradients
        tensors[0].grad = Tensor([3.0])
        tensors[1].grad = Tensor([4.0])
        
        # Total norm is 5, clip to 2.5
        total_norm = clip_grad_norm(tensors, 2.5)
        
        assert abs(total_norm - 5.0) < 1e-6
        # Gradients should be scaled by 0.5
        assert abs(tensors[0].grad.data[0] - 1.5) < 1e-6
        assert abs(tensors[1].grad.data[0] - 2.0) < 1e-6


class TestNumericalGradients:
    """Test numerical gradient computation."""
    
    def test_numerical_gradient_simple(self):
        """Test numerical gradient for simple function."""
        def func(x):
            return (x ** 2).sum()
        
        x = Tensor([2.0, 3.0], requires_grad=True)
        num_grad = numerical_gradient(func, x)
        
        # Analytical gradient is 2x
        expected = np.array([4.0, 6.0])
        np.testing.assert_array_almost_equal(num_grad, expected, decimal=5)
    
    def test_check_gradients(self):
        """Test gradient checking utility."""
        def func(x):
            return ((x ** 2) * 3 + x).sum()
        
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        match, diff = check_gradients(func, x)
        
        assert match is True
        assert diff < 1e-5
    
    def test_check_gradients_complex(self):
        """Test gradient checking for complex function."""
        def func(x):
            y = x ** 3
            z = y * 2 + x
            return z.sum()
        
        x = Tensor([1.0, 2.0], requires_grad=True)
        match, diff = check_gradients(func, x, tolerance=1e-4)
        
        assert match is True
        assert diff < 1e-4


class TestComplexComputations:
    """Test complex computational graphs."""
    
    def test_linear_regression_gradient(self):
        """Test gradients for simple linear regression."""
        # y = wx + b
        w = Tensor([2.0], requires_grad=True)
        b = Tensor([1.0], requires_grad=True)
        x = Tensor([3.0])
        
        # Forward pass
        y_pred = w * x + b
        y_true = Tensor([8.0])
        loss = ((y_pred - y_true) ** 2).sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert w.grad is not None
        assert b.grad is not None
        
        # dL/dw = 2(y_pred - y_true) * x = 2(7-8) * 3 = -6
        expected_w_grad = np.array([-6.0])
        np.testing.assert_array_almost_equal(w.grad.data, expected_w_grad)
        
        # dL/db = 2(y_pred - y_true) = 2(7-8) = -2
        expected_b_grad = np.array([-2.0])
        np.testing.assert_array_almost_equal(b.grad.data, expected_b_grad)
    
    def test_two_layer_network_gradient(self):
        """Test gradients for simple two-layer network."""
        # Input
        x = Tensor([[1.0, 2.0]], requires_grad=False)
        
        # Layer 1: y = x @ W1
        W1 = Tensor([[0.5, 0.5], [0.5, 0.5]], requires_grad=True)
        y = x @ W1
        
        # Layer 2: z = y @ W2
        W2 = Tensor([[1.0], [1.0]], requires_grad=True)
        z = y @ W2
        
        # Loss
        loss = z.sum()
        loss.backward()
        
        # Both weight matrices should have gradients
        assert W1.grad is not None
        assert W2.grad is not None
        assert W1.grad.shape == W1.shape
        assert W2.grad.shape == W2.shape
    
    def test_deep_computation_graph(self):
        """Test deep computational graph."""
        x = Tensor([2.0], requires_grad=True)
        
        # Build deep graph
        y = x
        for _ in range(10):
            y = y * 2 + 1
        
        y.backward()
        
        # Should have gradient
        assert x.grad is not None
        assert np.isfinite(x.grad.data).all()
    
    def test_wide_computation_graph(self):
        """Test wide computational graph with many operations."""
        x = Tensor([1.0], requires_grad=True)
        
        # Create many parallel paths
        outputs = []
        for i in range(10):
            outputs.append(x * (i + 1))
        
        # Sum all outputs
        result = outputs[0]
        for out in outputs[1:]:
            result = result + out
        
        result.backward()
        
        # Gradient should be sum of all coefficients: 1+2+3+...+10 = 55
        expected = np.array([55.0])
        np.testing.assert_array_almost_equal(x.grad.data, expected)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_backward_on_non_scalar_without_gradient(self):
        """Test that backward on non-scalar requires gradient argument."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x * 2
        
        with pytest.raises(RuntimeError):
            y.backward()  # Should fail: not a scalar
    
    def test_backward_without_requires_grad(self):
        """Test that backward fails if tensor doesn't require grad."""
        x = Tensor([1.0], requires_grad=False)
        y = x * 2
        
        with pytest.raises(RuntimeError):
            y.backward()
    
    def test_gradient_of_constant(self):
        """Test gradient when constant is involved."""
        x = Tensor([2.0], requires_grad=True)
        c = Tensor([3.0], requires_grad=False)
        y = x * c
        y.backward()
        
        assert x.grad is not None
        assert c.grad is None
        
        expected = np.array([3.0])
        np.testing.assert_array_equal(x.grad.data, expected)
    
    def test_zero_gradient(self):
        """Test operations that result in zero gradients."""
        x = Tensor([2.0], requires_grad=True)
        y = Tensor([3.0], requires_grad=True)
        z = x * 0 + y  # x's contribution is zero
        z.backward()
        
        assert x.grad.data[0] == 0.0
        assert y.grad.data[0] == 1.0
    
    def test_disconnected_graph(self):
        """Test gradient with disconnected computational graph."""
        x = Tensor([2.0], requires_grad=True)
        y = Tensor([3.0], requires_grad=True)
        
        # z doesn't depend on x
        z = y * 2
        z.backward()
        
        assert x.grad is None  # x not in graph
        assert y.grad is not None


class TestMemoryManagement:
    """Test memory management and graph cleanup."""
    
    def test_detach_breaks_graph(self):
        """Test that detach breaks computational graph."""
        x = Tensor([1.0], requires_grad=True)
        y = x * 2
        z = y.detach()
        w = z * 3
        
        # w should not have grad_fn since z was detached
        assert w.grad_fn is None
    
    def test_no_grad_prevents_graph_building(self):
        """Test that no_grad prevents graph building."""
        x = Tensor([1.0], requires_grad=True)
        
        with no_grad():
            y = x * 2
            z = y * 3
        
        # No computational graph should be built
        assert y._prev == set()
        assert z._prev == set()


class TestRealWorldExamples:
    """Test realistic use cases."""
    
    def test_mini_batch_linear_regression(self):
        """Test gradient computation for batched linear regression."""
        # Batch of 3 samples, 2 features
        X = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=False)
        y_true = Tensor([[5.0], [11.0], [17.0]], requires_grad=False)
        
        # Parameters
        W = Tensor([[1.0], [1.0]], requires_grad=True)
        b = Tensor([[0.0]], requires_grad=True)
        
        # Forward pass
        y_pred = X @ W + b
        
        # MSE loss
        diff = y_pred - y_true
        loss = (diff ** 2).mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and have correct shape
        assert W.grad is not None
        assert b.grad is not None
        assert W.grad.shape == W.shape
        assert b.grad.shape == b.shape
    
    def test_simple_optimization_step(self):
        """Test a simple gradient descent step."""
        # Simple function: f(x) = x^2
        x = Tensor([10.0], requires_grad=True)
        
        learning_rate = 0.1
        
        for _ in range(5):
            # Forward
            loss = (x ** 2).sum()
            
            # Backward
            x.zero_grad()
            loss.backward()
            
            # Update (manual gradient descent)
            with no_grad():
                x.data -= learning_rate * x.grad.data
        
        # x should have moved toward 0
        assert abs(x.data[0]) < 10.0
    
    def test_multi_output_function(self):
        """Test function with multiple outputs."""
        x = Tensor([1.0, 2.0], requires_grad=True)
        
        # Multiple outputs
        y1 = (x ** 2).sum()
        y2 = (x * 3).sum()
        
        # Total loss
        loss = y1 + y2
        loss.backward()
        
        # dy1/dx = 2x, dy2/dx = 3
        # Total: 2x + 3
        expected = 2 * np.array([1.0, 2.0]) + 3
        np.testing.assert_array_almost_equal(x.grad.data, expected)


class TestGradientChecking:
    """Test gradient checking utilities."""
    
    def test_gradient_check_polynomial(self):
        """Test gradient checking for polynomial."""
        def func(x):
            return (x ** 3 + 2 * x ** 2 + x).sum()
        
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        match, diff = check_gradients(func, x)
        
        assert match is True
        assert diff < 1e-5
    
    def test_gradient_check_matrix_multiply(self):
        """Test gradient checking for matrix operations."""
        A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        B_fixed = np.array([[5.0, 6.0], [7.0, 8.0]])
        
        def func(A):
            B = Tensor(B_fixed, requires_grad=False)
            C = A @ B
            return C.sum()
        
        match, diff = check_gradients(func, A)
        
        assert match is True
        assert diff < 1e-5
    
    def test_gradient_check_complex_function(self):
        """Test gradient checking for complex function."""
        def func(x):
            y = x ** 2
            z = y * 3 + x
            w = z.mean()
            return w
        
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        match, diff = check_gradients(func, x)
        
        assert match is True
        assert diff < 1e-5


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])