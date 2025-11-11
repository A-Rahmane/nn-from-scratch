# Lab 1: Tensor Operations

## Status
🟢 Completed

## Description
Implementation of a comprehensive Tensor class for neural network operations. This lab builds the fundamental data structure used throughout the entire course, providing NumPy-based tensor operations with an interface similar to PyTorch.

## Learning Outcomes
After completing this lab, you will be able to:
- Create and manipulate multi-dimensional tensors
- Perform element-wise and matrix operations
- Understand broadcasting mechanics
- Use indexing and slicing effectively
- Apply shape manipulation techniques
- Implement aggregation operations
- Build the foundation for automatic differentiation

## Files
- `tensor.py` - Main Tensor class implementation (500+ lines)
- `tests/test_tensor.py` - Comprehensive unit tests (400+ lines)
- `notebooks/demo.ipynb` - Interactive demonstration notebook
- `README.md` - This file

## Implementation Highlights

### Core Features
1. **Data Storage**: Float32 NumPy arrays for numerical stability
2. **Arithmetic Operations**: Full support for +, -, *, /, **, with broadcasting
3. **Comparison Operations**: ==, !=, <, >, <=, >=
4. **Indexing**: NumPy-style indexing, slicing, and boolean masking
5. **Shape Manipulation**: reshape, transpose, flatten, squeeze, unsqueeze
6. **Aggregations**: sum, mean, max, min, std, var with axis support
7. **Matrix Operations**: matmul (@), dot product, batch operations
8. **Advanced Utilities**: clone, factory methods, concatenate, stack, split

### Design Decisions
- **Immutability**: Operations return new tensors (except `__setitem__`)
- **Type Coercion**: Automatic conversion of scalars and lists to tensors
- **Broadcasting**: Leverages NumPy's broadcasting for shape compatibility
- **Error Handling**: Clear error messages for shape mismatches

## How to Run

### Installation
```bash
# From repository root
pip install -e .
```

### Run Tests
```bash
# Run all tests
pytest semester1/lab1_tensor_operations/tests/ -v

# Run specific test class
pytest semester1/lab1_tensor_operations/tests/test_tensor.py::TestArithmeticOperations -v

# Run with coverage
pytest semester1/lab1_tensor_operations/tests/ --cov=semester1.lab1_tensor_operations
```

### Interactive Demo
```bash
# Launch Jupyter notebook
jupyter notebook semester1/lab1_tensor_operations/notebooks/demo.ipynb

# Or run the main file
python semester1/lab1_tensor_operations/tensor.py
```

## Key Concepts

### Broadcasting Rules
NumPy/PyTorch broadcasting rules are followed:
1. If tensors have different number of dimensions, prepend 1s to the smaller shape
2. Tensors are compatible if, for each dimension, they are equal or one of them is 1
3. After broadcasting, each tensor behaves as if it had the larger shape

Example:
```python
a = Tensor([[1, 2, 3]])      # shape (1, 3)
b = Tensor([[1], [2], [3]])  # shape (3, 1)
c = a + b                     # shape (3, 3) via broadcasting
```

### Memory Efficiency
Operations create new tensors, but NumPy's copy-on-write behavior means memory
is only duplicated when necessary. Use `.clone()` for explicit deep copies.

### Axis Convention
- `axis=0`: Operations along rows (vertically)
- `axis=1`: Operations along columns (horizontally)
- `axis=None`: Operations over all elements

## Test Results
```bash
$ pytest semester1/lab1_tensor_operations/tests/ -v

========================== test session starts ==========================
collected 85 items

test_tensor.py::TestTensorCreation::test_create_from_list PASSED    [  1%]
test_tensor.py::TestTensorCreation::test_create_from_nested_list PASSED [  2%]
...
test_tensor.py::TestEdgeCases::test_chained_operations PASSED      [100%]

========================== 85 passed in 2.34s ===========================
```

## Performance Notes

Vectorized operations are significantly faster than Python loops:
- 1000x1000 matrix addition: ~2ms (vectorized) vs ~2000ms (loops)
- Use NumPy's optimized C implementations under the hood
- Broadcasting avoids explicit memory duplication

## Common Pitfalls

1. **Shape Mismatch**: Always check shapes before operations
```python
   a = Tensor([[1, 2]])     # shape (1, 2)
   b = Tensor([[1], [2]])   # shape (2, 1)
   c = a @ b                # ❌ Error: incompatible shapes
```

2. **Reference vs Copy**: Regular assignment creates references
```python
   a = Tensor([1, 2, 3])
   b = a           # b references same data
   b = a.clone()   # b is independent copy
```

3. **Scalar vs Tensor**: Check return types
```python
   t = Tensor([[1, 2], [3, 4]])
   s = t.sum()      # Returns Tensor, not Python scalar
   v = s.data.item()  # Extract Python scalar
```

## Extensions (Optional Challenges)

For students seeking additional challenge:

1. **Implement einsum**: Einstein summation notation
2. **Add GPU support**: Using CuPy instead of NumPy
3. **Optimize memory**: Implement in-place operations
4. **Add strided operations**: View-based operations without copying
5. **Implement gather/scatter**: Advanced indexing operations

## Connection to Future Labs

This Tensor class will be extended in Lab 2 with:
- Computational graph tracking
- Automatic gradient computation
- Backward pass implementation

The foundation built here is crucial for understanding how modern deep learning
frameworks like PyTorch work under the hood.

## References

1. **NumPy Documentation**: https://numpy.org/doc/stable/
2. **Broadcasting Tutorial**: https://numpy.org/doc/stable/user/basics.broadcasting.html
3. **PyTorch Tensor Docs**: https://pytorch.org/docs/stable/tensors.html
4. **Deep Learning Book (Goodfellow et al.)**: Chapter 2 - Linear Algebra

## Troubleshooting

### Import Errors
```bash
# If you get import errors, ensure the package is installed
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/nn-from-scratch"
```

### Test Failures
```bash
# Run tests with verbose output
pytest semester1/lab1_tensor_operations/tests/ -vv

# Run specific failing test
pytest semester1/lab1_tensor_operations/tests/test_tensor.py::test_name -vv
```

### Type Errors
All operations should work with:
- Other Tensor objects
- NumPy arrays
- Python scalars (int, float)
- Python lists

If you encounter type errors, ensure proper conversion in the operation methods.

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| **Implementation** | 40 | All methods correctly implemented |
| - Basic operations | 10 | Arithmetic, comparison ops |
| - Indexing/slicing | 8 | All indexing modes work |
| - Shape manipulation | 8 | Reshape, transpose, etc. |
| - Aggregations | 7 | Sum, mean, max, etc. |
| - Matrix operations | 7 | Matmul, dot product |
| **Tests** | 20 | Comprehensive test coverage |
| - Test completeness | 10 | All features tested |
| - Edge cases | 10 | Boundary conditions |
| **Documentation** | 20 | Clear, complete docs |
| - Docstrings | 10 | All public methods |
| - Comments | 5 | Complex logic explained |
| - README | 5 | Complete lab README |
| **Demo** | 20 | Working demonstration |
| - Notebook completeness | 10 | All features shown |
| - Explanations | 10 | Clear explanations |

**Total: 100 points**

## Author
Lab developed for Neural Networks from Scratch course

## License
MIT License - See repository root for details
