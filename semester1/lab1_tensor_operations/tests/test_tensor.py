"""Unit tests for Tensor class."""
import pytest
import numpy as np
from semester1.lab1_tensor_operations.tensor import Tensor


class TestTensorCreation:
    """Test tensor creation and properties."""
    
    def test_create_from_list(self):
        """Test creating tensor from list."""
        t = Tensor([1, 2, 3])
        assert t.shape == (3,)
        assert t.ndim == 1
        assert t.size == 3
    
    def test_create_from_nested_list(self):
        """Test creating tensor from nested list."""
        t = Tensor([[1, 2], [3, 4]])
        assert t.shape == (2, 2)
        assert t.ndim == 2
        assert t.size == 4
    
    def test_create_from_scalar(self):
        """Test creating tensor from scalar."""
        t = Tensor(5)
        assert t.shape == ()
        assert t.ndim == 0
        assert t.size == 1
    
    def test_dtype_is_float32(self):
        """Test that dtype is float32."""
        t = Tensor([1, 2, 3])
        assert t.dtype == np.float32


class TestArithmeticOperations:
    """Test arithmetic operations."""
    
    def test_addition(self):
        """Test tensor addition."""
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        result = a + b
        expected = np.array([5, 7, 9], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_addition_with_scalar(self):
        """Test tensor + scalar."""
        a = Tensor([1, 2, 3])
        result = a + 5
        expected = np.array([6, 7, 8], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_reverse_addition(self):
        """Test scalar + tensor."""
        a = Tensor([1, 2, 3])
        result = 5 + a
        expected = np.array([6, 7, 8], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_subtraction(self):
        """Test tensor subtraction."""
        a = Tensor([5, 7, 9])
        b = Tensor([1, 2, 3])
        result = a - b
        expected = np.array([4, 5, 6], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_multiplication(self):
        """Test element-wise multiplication."""
        a = Tensor([1, 2, 3])
        b = Tensor([2, 3, 4])
        result = a * b
        expected = np.array([2, 6, 12], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_division(self):
        """Test element-wise division."""
        a = Tensor([10, 20, 30])
        b = Tensor([2, 4, 5])
        result = a / b
        expected = np.array([5, 5, 6], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_power(self):
        """Test power operation."""
        a = Tensor([2, 3, 4])
        result = a ** 2
        expected = np.array([4, 9, 16], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_negation(self):
        """Test negation operation."""
        a = Tensor([1, -2, 3])
        result = -a
        expected = np.array([-1, 2, -3], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_broadcasting(self):
        """Test broadcasting in operations."""
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        b = Tensor([10, 20, 30])
        result = a + b
        expected = np.array([[11, 22, 33], [14, 25, 36]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)


class TestComparisonOperations:
    """Test comparison operations."""
    
    def test_equal(self):
        """Test equality comparison."""
        a = Tensor([1, 2, 3])
        b = Tensor([1, 0, 3])
        result = a == b
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(result.data, expected)
    
    def test_not_equal(self):
        """Test inequality comparison."""
        a = Tensor([1, 2, 3])
        b = Tensor([1, 0, 3])
        result = a != b
        expected = np.array([False, True, False])
        np.testing.assert_array_equal(result.data, expected)
    
    def test_greater_than(self):
        """Test greater than comparison."""
        a = Tensor([1, 2, 3])
        b = Tensor([0, 2, 4])
        result = a > b
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(result.data, expected)
    
    def test_less_than(self):
        """Test less than comparison."""
        a = Tensor([1, 2, 3])
        b = Tensor([2, 2, 2])
        result = a < b
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(result.data, expected)
    
    def test_greater_equal(self):
        """Test greater than or equal comparison."""
        a = Tensor([1, 2, 3])
        b = Tensor([1, 1, 4])
        result = a >= b
        expected = np.array([True, True, False])
        np.testing.assert_array_equal(result.data, expected)
    
    def test_less_equal(self):
        """Test less than or equal comparison."""
        a = Tensor([1, 2, 3])
        b = Tensor([1, 3, 3])
        result = a <= b
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(result.data, expected)


class TestIndexingSlicing:
    """Test indexing and slicing operations."""
    
    def test_single_index(self):
        """Test single index access."""
        t = Tensor([1, 2, 3, 4, 5])
        result = t[2]
        assert result.data == 3.0
    
    def test_slice(self):
        """Test slicing."""
        t = Tensor([1, 2, 3, 4, 5])
        result = t[1:4]
        expected = np.array([2, 3, 4], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_2d_indexing(self):
        """Test 2D indexing."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t[0, 1]
        assert result.data == 2.0
    
    def test_row_slicing(self):
        """Test row slicing."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t[0]
        expected = np.array([1, 2, 3], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_column_slicing(self):
        """Test column slicing."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t[:, 1]
        expected = np.array([2, 5], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_boolean_indexing(self):
        """Test boolean indexing."""
        t = Tensor([1, 2, 3, 4, 5])
        mask = t > 3
        result = t[mask]
        expected = np.array([4, 5], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_setitem(self):
        """Test item assignment."""
        t = Tensor([1, 2, 3])
        t[1] = 10
        expected = np.array([1, 10, 3], dtype=np.float32)
        np.testing.assert_array_equal(t.data, expected)


class TestShapeManipulation:
    """Test shape manipulation operations."""
    
    def test_reshape(self):
        """Test reshape operation."""
        t = Tensor([1, 2, 3, 4, 5, 6])
        result = t.reshape(2, 3)
        assert result.shape == (2, 3)
        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_reshape_with_tuple(self):
        """Test reshape with tuple argument."""
        t = Tensor([1, 2, 3, 4, 5, 6])
        result = t.reshape((3, 2))
        assert result.shape == (3, 2)
    
    def test_transpose(self):
        """Test transpose operation."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.transpose()
        expected = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_transpose_property(self):
        """Test T property."""
        t = Tensor([[1, 2], [3, 4]])
        result = t.T
        expected = np.array([[1, 3], [2, 4]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_flatten(self):
        """Test flatten operation."""
        t = Tensor([[1, 2], [3, 4], [5, 6]])
        result = t.flatten()
        expected = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_squeeze(self):
        """Test squeeze operation."""
        t = Tensor([[[1, 2, 3]]])
        result = t.squeeze()
        assert result.shape == (3,)
    
    def test_squeeze_specific_axis(self):
        """Test squeeze on specific axis."""
        t = Tensor([[[1], [2], [3]]])
        result = t.squeeze(2)
        assert result.shape == (1, 3)
    
    def test_unsqueeze(self):
        """Test unsqueeze operation."""
        t = Tensor([1, 2, 3])
        result = t.unsqueeze(0)
        assert result.shape == (1, 3)
        result = t.unsqueeze(1)
        assert result.shape == (3, 1)


class TestAggregationOperations:
    """Test aggregation operations."""
    
    def test_sum_all(self):
        """Test sum over all elements."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.sum()
        assert result.data == 21.0
    
    def test_sum_axis0(self):
        """Test sum along axis 0."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.sum(axis=0)
        expected = np.array([5, 7, 9], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_sum_axis1(self):
        """Test sum along axis 1."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.sum(axis=1)
        expected = np.array([6, 15], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_sum_keepdims(self):
        """Test sum with keepdims."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.sum(axis=1, keepdims=True)
        assert result.shape == (2, 1)
    
    def test_mean(self):
        """Test mean operation."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.mean()
        assert result.data == 3.5
    
    def test_mean_axis(self):
        """Test mean along axis."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.mean(axis=0)
        expected = np.array([2.5, 3.5, 4.5], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.data, expected)
    
    def test_max(self):
        """Test max operation."""
        t = Tensor([[1, 5, 3], [4, 2, 6]])
        result = t.max()
        assert result.data == 6.0
    
    def test_max_axis(self):
        """Test max along axis."""
        t = Tensor([[1, 5, 3], [4, 2, 6]])
        result = t.max(axis=1)
        expected = np.array([5, 6], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_min(self):
        """Test min operation."""
        t = Tensor([[1, 5, 3], [4, 2, 6]])
        result = t.min()
        assert result.data == 1.0
    
    def test_std(self):
        """Test standard deviation."""
        t = Tensor([1, 2, 3, 4, 5])
        result = t.std()
        expected = np.std([1, 2, 3, 4, 5], dtype=np.float32)
        np.testing.assert_almost_equal(result.data, expected)
    
    def test_var(self):
        """Test variance."""
        t = Tensor([1, 2, 3, 4, 5])
        result = t.var()
        expected = np.var([1, 2, 3, 4, 5], dtype=np.float32)
        np.testing.assert_almost_equal(result.data, expected)


class TestMatrixOperations:
    """Test matrix operations."""
    
    def test_matmul_2d(self):
        """Test 2D matrix multiplication."""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        result = a @ b
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_matmul_method(self):
        """Test matmul method."""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        result = a.matmul(b)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_dot_product_1d(self):
        """Test dot product of vectors."""
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        result = a.dot(b)
        assert result.data == 32.0
    
    def test_dot_product_2d(self):
        """Test dot product with 2D tensors."""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        result = a.dot(b)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_matmul_incompatible_shapes(self):
        """Test matmul with incompatible shapes raises error."""
        a = Tensor([[1, 2, 3]])
        b = Tensor([[4, 5]])
        with pytest.raises(ValueError):
            a @ b
    
    def test_matmul_1d_2d(self):
        """Test matmul with 1D and 2D tensors."""
        a = Tensor([1, 2])
        b = Tensor([[3, 4], [5, 6]])
        result = a @ b
        expected = np.array([13, 16], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)


class TestAdvancedOperations:
    """Test advanced operations."""
    
    def test_clone(self):
        """Test clone creates deep copy."""
        t = Tensor([[1, 2], [3, 4]])
        t_clone = t.clone()
        t_clone[0, 0] = 99
        assert t.data[0, 0] == 1.0  # Original unchanged
        assert t_clone.data[0, 0] == 99.0
    
    def test_zeros_like(self):
        """Test zeros_like."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        zeros = Tensor.zeros_like(t)
        expected = np.zeros((2, 3), dtype=np.float32)
        np.testing.assert_array_equal(zeros.data, expected)
    
    def test_ones_like(self):
        """Test ones_like."""
        t = Tensor([[1, 2], [3, 4]])
        ones = Tensor.ones_like(t)
        expected = np.ones((2, 2), dtype=np.float32)
        np.testing.assert_array_equal(ones.data, expected)
    
    def test_rand_like(self):
        """Test rand_like creates correct shape."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        rand = Tensor.rand_like(t)
        assert rand.shape == t.shape
        assert np.all((rand.data >= 0) & (rand.data < 1))
    
    def test_randn_like(self):
        """Test randn_like creates correct shape."""
        t = Tensor([[1, 2], [3, 4], [5, 6]])
        randn = Tensor.randn_like(t)
        assert randn.shape == t.shape
    
    def test_concatenate_axis0(self):
        """Test concatenate along axis 0."""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        result = Tensor.concatenate([a, b], axis=0)
        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_concatenate_axis1(self):
        """Test concatenate along axis 1."""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        result = Tensor.concatenate([a, b], axis=1)
        expected = np.array([[1, 2, 5, 6], [3, 4, 7, 8]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_stack_axis0(self):
        """Test stack along axis 0."""
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        result = Tensor.stack([a, b], axis=0)
        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_stack_axis1(self):
        """Test stack along axis 1."""
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        result = Tensor.stack([a, b], axis=1)
        expected = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)
    
    def test_split_equal_sections(self):
        """Test split into equal sections."""
        t = Tensor([1, 2, 3, 4, 5, 6])
        result = t.split(3)
        assert len(result) == 3
        assert result[0].shape == (2,)
        np.testing.assert_array_equal(result[0].data, np.array([1, 2], dtype=np.float32))
    
    def test_split_2d(self):
        """Test split on 2D tensor."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.split(2, axis=0)
        assert len(result) == 2
        assert result[0].shape == (1, 3)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_tensor(self):
        """Test creating empty tensor."""
        t = Tensor([])
        assert t.shape == (0,)
        assert t.size == 0
    
    def test_scalar_operations(self):
        """Test operations with scalar tensors."""
        a = Tensor(5)
        b = Tensor(3)
        result = a + b
        assert result.data == 8.0
    
    def test_large_tensor(self):
        """Test with large tensor."""
        t = Tensor(np.random.randn(1000, 1000))
        assert t.shape == (1000, 1000)
        result = t.sum()
        assert isinstance(result, Tensor)
    
    def test_negative_indexing(self):
        """Test negative indexing."""
        t = Tensor([1, 2, 3, 4, 5])
        assert t[-1].data == 5.0
        assert t[-2].data == 4.0
    
    def test_chained_operations(self):
        """Test chaining multiple operations."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = (t + 1) * 2 - 3
        expected = np.array([[1, 3, 5], [7, 9, 11]], dtype=np.float32)
        np.testing.assert_array_equal(result.data, expected)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
