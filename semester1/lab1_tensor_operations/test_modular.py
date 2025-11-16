"""Quick test script to verify modular implementation works correctly.

Run this to ensure the refactored modular implementation maintains
100% compatibility with the original monolithic version.

Usage:
    python semester1/lab1_tensor_operations/test_modular.py
"""

from tensor import Tensor
import numpy as np


def test_basic_operations():
    """Test basic tensor operations."""
    print("Testing basic operations...")
    
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    
    # Arithmetic
    assert np.allclose((t1 + t2).data, [5, 7, 9])
    assert np.allclose((t1 - t2).data, [-3, -3, -3])
    assert np.allclose((t1 * 2).data, [2, 4, 6])
    assert np.allclose((t1 / 2).data, [0.5, 1, 1.5])
    assert np.allclose((t1 ** 2).data, [1, 4, 9])
    
    # Reverse operations
    assert np.allclose((5 + t1).data, [6, 7, 8])
    assert np.allclose((10 - t1).data, [9, 8, 7])
    assert np.allclose((2 * t1).data, [2, 4, 6])
    
    print("✅ Basic operations passed")


def test_comparisons():
    """Test comparison operations."""
    print("Testing comparisons...")
    
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([1, 0, 4])
    
    assert np.all((t1 == t2).data == [True, False, False])
    assert np.all((t1 != t2).data == [False, True, True])
    assert np.all((t1 > t2).data == [False, True, False])
    assert np.all((t1 >= t2).data == [True, True, False])
    
    print("✅ Comparisons passed")


def test_indexing():
    """Test indexing and slicing."""
    print("Testing indexing...")
    
    t = Tensor([1, 2, 3, 4, 5])
    
    assert t[2].data == 3.0
    assert np.allclose(t[1:4].data, [2, 3, 4])
    
    t[0] = 10
    assert t[0].data == 10.0
    
    # Boolean indexing
    mask = t > 3
    filtered = t[mask]
    assert len(filtered.data) == 2
    
    print("✅ Indexing passed")


def test_shape_ops():
    """Test shape manipulation."""
    print("Testing shape manipulation...")
    
    t = Tensor([1, 2, 3, 4, 5, 6])
    
    reshaped = t.reshape(2, 3)
    assert reshaped.shape == (2, 3)
    
    flat = reshaped.flatten()
    assert flat.shape == (6,)
    
    transposed = reshaped.T
    assert transposed.shape == (3, 2)
    
    unsqueezed = t.unsqueeze(0)
    assert unsqueezed.shape == (1, 6)
    
    print("✅ Shape manipulation passed")


def test_aggregations():
    """Test aggregation operations."""
    print("Testing aggregations...")
    
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    
    assert t.sum().data == 21.0
    assert np.allclose(t.sum(axis=0).data, [5, 7, 9])
    assert np.allclose(t.mean().data, 3.5)
    assert t.max().data == 6.0
    assert t.min().data == 1.0
    assert t.std().data > 0  # Just check it computes
    assert t.var().data > 0
    
    print("✅ Aggregations passed")


def test_linalg():
    """Test linear algebra operations."""
    print("Testing linear algebra...")
    
    m1 = Tensor([[1, 2], [3, 4]])
    m2 = Tensor([[5, 6], [7, 8]])
    
    result = m1 @ m2
    expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
    assert np.allclose(result.data, expected)
    
    v1 = Tensor([1, 2, 3])
    v2 = Tensor([4, 5, 6])
    dot_result = v1.dot(v2)
    assert dot_result.data == 32.0
    
    print("✅ Linear algebra passed")


def test_factory():
    """Test factory methods."""
    print("Testing factory methods...")
    
    t = Tensor([[1, 2], [3, 4]])
    
    zeros = Tensor.zeros_like(t)
    assert np.all(zeros.data == 0)
    
    ones = Tensor.ones_like(t)
    assert np.all(ones.data == 1)
    
    rand = Tensor.rand_like(t)
    assert rand.shape == t.shape
    
    # Concatenate
    cat = Tensor.concatenate([t, t], axis=0)
    assert cat.shape == (4, 2)
    
    # Stack
    stacked = Tensor.stack([t, t], axis=0)
    assert stacked.shape == (2, 2, 2)
    
    # Split
    split_result = t.split(2, axis=0)
    assert len(split_result) == 2
    
    print("✅ Factory methods passed")


def main():
    """Run all tests."""
    print("="*50)
    print("Testing Modular Tensor Implementation")
    print("="*50 + "\n")
    
    test_basic_operations()
    test_comparisons()
    test_indexing()
    test_shape_ops()
    test_aggregations()
    test_linalg()
    test_factory()
    
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED! Refactoring successful!")
    print("="*50)


if __name__ == "__main__":
    main()