# Lab 1: Tensor Operations

## Status
🟢 Completed (Refactored to Modular Architecture)

## Description
Implementation of a comprehensive Tensor class for neural network operations. This lab builds the fundamental data structure used throughout the entire course, providing NumPy-based tensor operations with an interface similar to PyTorch.

**New in v1.0**: The Tensor class has been refactored into a modular architecture for improved maintainability and code organization.

## Architecture

### Modular Structure
The Tensor implementation is now split into focused modules:
```
semester1/lab1_tensor_operations/tensor/
├── __init__.py          # Module initialization and exports
├── core.py              # Core Tensor class and properties
├── arithmetic.py        # Arithmetic operations (+, -, *, /, **)
├── compare.py           # Comparison operations (==, !=, <, >, <=, >=)
├── indexing.py          # Indexing and slicing operations
├── shape.py             # Shape manipulation (reshape, transpose, etc.)
├── reduce.py            # Aggregation operations (sum, mean, max, min, std, var)
├── linalg.py            # Linear algebra (matmul, dot)
└── factory.py           # Factory methods and batch operations
```

### Design Benefits
1. **Separation of Concerns**: Each module handles a specific category of operations
2. **Maintainability**: Easy to locate and modify specific functionality
3. **Testability**: Individual modules can be tested in isolation
4. **Extensibility**: New operations can be added without modifying existing code
5. **Readability**: Smaller, focused files are easier to understand

## Learning Outcomes
After completing this lab, you will be able to:
- Create and manipulate multi-dimensional tensors
- Perform element-wise and matrix operations
- Understand broadcasting mechanics
- Use indexing and slicing effectively
- Apply shape manipulation techniques
- Implement aggregation operations
- Build the foundation for automatic differentiation
- **Design modular, maintainable code architectures**

## Files
- `tensor/` - Modular Tensor implementation
  - `core.py` - Core class (100 lines)
  - `arithmetic.py` - Arithmetic ops (60 lines)
  - `compare.py` - Comparison ops (40 lines)
  - `indexing.py` - Indexing ops (30 lines)
  - `shape.py` - Shape manipulation (50 lines)
  - `reduce.py` - Aggregations (40 lines)
  - `linalg.py` - Linear algebra (20 lines)
  - `factory.py` - Factory methods (50 lines)
  - `__init__.py` - Module init (30 lines)
- `tensor.py` - Legacy monolithic implementation (500+ lines) [DEPRECATED]
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
- **Modular Architecture**: Operations organized by category for maintainability
- **Immutability**: Operations return new tensors (except `__setitem__`)
- **Type Coercion**: Automatic conversion of scalars and lists to tensors
- **Broadcasting**: Leverages NumPy's broadcasting for shape compatibility
- **Error Handling**: Clear error messages for shape mismatches
- **Lazy Imports**: Core module imports operation modules to avoid circular dependencies

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

# Or test the modular implementation
python -c "from semester1.lab1_tensor_operations.tensor import Tensor; print(Tensor([1,2,3]) + Tensor([4,5,6]))"
```

## Usage Examples

### Basic Operations
```python
from semester1.lab1_tensor_operations.tensor import Tensor

# Creation
t1 = Tensor([1, 2, 3])
t2 = Tensor([[1, 2], [3, 4]])

# Arithmetic (automatically uses arithmetic.py)
result = t1 + 5          # Element-wise addition
result = t1 * 2          # Scalar multiplication
result = t1 ** 2         # Power operation

# Matrix operations (automatically uses linalg.py)
result = t2 @ t2.T       # Matrix multiplication

# Aggregations (automatically uses reduce.py)
mean = t2.mean(axis=0)   # Column-wise mean
total = t2.sum()         # Sum all elements

# Shape manipulation (automatically uses shape.py)
reshaped = t1.reshape(3, 1)
flat = t2.flatten()
transposed = t2.T
```

### Advanced Operations
```python
# Factory methods (automatically uses factory.py)
zeros = Tensor.zeros_like(t2)
ones = Tensor.ones_like(t2)
random = Tensor.rand_like(t2)

# Batch operations
concatenated = Tensor.concatenate([t1, t1], axis=0)
stacked = Tensor.stack([t1, t1], axis=0)

# Indexing (automatically uses indexing.py)
element = t2[0, 1]       # Single element
row = t2[0]              # First row
column = t2[:, 1]        # Second column
mask = t1 > 2            # Boolean indexing
filtered = t1[mask]
```

## Module Details

### core.py
- **Purpose**: Define the base Tensor class with data storage and properties
- **Key Components**: `__init__`, shape, ndim, size, dtype, clone, `__repr__`, `__str__`
- **Integration**: Imports and binds all operations from other modules

### arithmetic.py
- **Purpose**: Implement arithmetic operations
- **Operations**: add, sub, mul, div, pow, neg + reverse variants (radd, rsub, etc.)
- **Helper**: `_ensure_tensor()` for type coercion

### compare.py
- **Purpose**: Implement comparison operations
- **Operations**: eq, ne, lt, le, gt, ge
- **Returns**: Boolean tensors

### indexing.py
- **Purpose**: Handle indexing and slicing
- **Operations**: getitem, setitem
- **Supports**: Integer indexing, slicing, boolean masking, NumPy-style indexing

### shape.py
- **Purpose**: Shape manipulation operations
- **Operations**: reshape, flatten, transpose, squeeze, unsqueeze, T property
- **Flexibility**: Supports multiple input formats (tuples, args)

### reduce.py
- **Purpose**: Aggregation operations along axes
- **Operations**: sum, mean, max, min, std, var
- **Features**: axis parameter, keepdims option

### linalg.py
- **Purpose**: Linear algebra operations
- **Operations**: matmul, dot
- **Error Handling**: Clear messages for incompatible shapes

### factory.py
- **Purpose**: Factory methods and batch operations
- **Class Methods**: zeros_like, ones_like, rand_like, randn_like, concatenate, stack
- **Instance Methods**: split

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
test_tensor.py::TestArithmeticOperations::test_addition PASSED      [  3%]
test_tensor.py::TestArithmeticOperations::test_reverse_addition PASSED [  4%]
...
test_tensor.py::TestEdgeCases::test_chained_operations PASSED      [100%]

========================== 85 passed in 2.34s ===========================
```

All tests pass with the new modular architecture!

## Migration Guide

### From Monolithic to Modular

The old monolithic `tensor.py` has been refactored, but the API remains **100% compatible**:
```python
# Old way (still works)
from semester1.lab1_tensor_operations.tensor import Tensor

# New way (recommended)
from semester1.lab1_tensor_operations.tensor import Tensor

# Usage is identical
t = Tensor([1, 2, 3])
result = t + 5
```

**No code changes required!** The refactoring is internal only.

### For Contributors

When adding new operations:

1. **Identify the category**: arithmetic, comparison, shape, reduction, etc.
2. **Add to appropriate module**: Create function in relevant .py file
3. **Bind to Tensor class**: Add lambda binding in `core.py`
4. **Write tests**: Add test cases in `test_tensor.py`
5. **Update documentation**: Document the new operation

Example: Adding a new operation
```python
# In reduce.py
def median(t: Tensor, axis=None, keepdims=False) -> Tensor:
    """Compute median of tensor elements."""
    return Tensor(np.median(t.data, axis=axis, keepdims=keepdims))

# In core.py (after imports)
from .reduce import median
Tensor.median = lambda self, axis=None, keepdims=False: median(self, axis, keepdims)
```

## Performance Notes

Vectorized operations are significantly faster than Python loops:
- 1000x1000 matrix addition: ~2ms (vectorized) vs ~2000ms (loops)
- Use NumPy's optimized C implementations under the hood
- Broadcasting avoids explicit memory duplication
- Modular architecture has **zero performance overhead** (functions are bound at import time)

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
6. **Add more modules**: Create modules for special functions (exp, log, sin, cos)

## Connection to Future Labs

This modular Tensor class will be extended in Lab 2 with:
- Computational graph tracking
- Automatic gradient computation
- Backward pass implementation

The modular architecture makes it easy to add these features:
- Add `autograd.py` module for gradient tracking
- Extend `core.py` with `requires_grad` flag
- Each operation module can define its own backward pass

The foundation built here is crucial for understanding how modern deep learning
frameworks like PyTorch work under the hood.

## Architecture Comparison

### Monolithic (tensor.py)
```
tensor.py (500+ lines)
├── Properties (20 lines)
├── Arithmetic (80 lines)
├── Comparison (60 lines)
├── Indexing (30 lines)
├── Shape (70 lines)
├── Aggregation (80 lines)
├── Linear Algebra (40 lines)
└── Factory Methods (120 lines)
```

**Issues**: Hard to navigate, difficult to maintain, everything in one file

### Modular (tensor/)
```
tensor/
├── __init__.py (30 lines)      # Clean exports
├── core.py (100 lines)         # Core class + integration
├── arithmetic.py (60 lines)    # Just arithmetic
├── compare.py (40 lines)       # Just comparisons
├── indexing.py (30 lines)      # Just indexing
├── shape.py (50 lines)         # Just shapes
├── reduce.py (40 lines)        # Just reductions
├── linalg.py (20 lines)        # Just linear algebra
└── factory.py (50 lines)       # Just factories
```

**Benefits**: Clear organization, easy to navigate, maintainable, testable

## References

1. **NumPy Documentation**: https://numpy.org/doc/stable/
2. **Broadcasting Tutorial**: https://numpy.org/doc/stable/user/basics.broadcasting.html
3. **PyTorch Tensor Docs**: https://pytorch.org/docs/stable/tensors.html
4. **Deep Learning Book (Goodfellow et al.)**: Chapter 2 - Linear Algebra
5. **Clean Code (Martin)**: Principles of modular design

## Troubleshooting

### Import Errors
```bash
# If you get import errors, ensure the package is installed
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/nn-from-scratch"

# Test the import
python -c "from semester1.lab1_tensor_operations.tensor import Tensor; print('Import successful!')"
```

### Test Failures
```bash
# Run tests with verbose output
pytest semester1/lab1_tensor_operations/tests/ -vv

# Run specific failing test
pytest semester1/lab1_tensor_operations/tests/test_tensor.py::test_name -vv

# Check which tests use the old vs new implementation
pytest semester1/lab1_tensor_operations/tests/ -v --tb=short
```

### Type Errors
All operations should work with:
- Other Tensor objects
- NumPy arrays
- Python scalars (int, float)
- Python lists

If you encounter type errors, check the `_ensure_tensor()` helper in `arithmetic.py`.

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| **Implementation** | 40 | All methods correctly implemented |
| - Basic operations | 10 | Arithmetic, comparison ops |
| - Indexing/slicing | 8 | All indexing modes work |
| - Shape manipulation | 8 | Reshape, transpose, etc. |
| - Aggregations | 7 | Sum, mean, max, etc. |
| - Matrix operations | 7 | Matmul, dot product |
| **Modular Design** | 20 | Clean architecture |
| - Separation of concerns | 10 | Each module focused |
| - Code organization | 5 | Logical file structure |
| - Integration | 5 | Proper binding in core.py |
| **Tests** | 20 | Comprehensive test coverage |
| - Test completeness | 10 | All features tested |
| - Edge cases | 10 | Boundary conditions |
| **Documentation** | 20 | Clear, complete docs |
| - Module docstrings | 5 | All modules documented |
| - Function docstrings | 5 | All functions documented |
| - README | 5 | Complete lab README |
| - Architecture docs | 5 | Design decisions explained |

**Total: 100 points**

**Bonus Points (+10)**: Successfully refactored from monolithic to modular architecture with 100% test compatibility

## Author
Lab developed for Neural Networks from Scratch course
Refactored to modular architecture by MENOUER Abderrahmane

## License
MIT License - See repository root for details

## Changelog

### Version 1.0.0 (Current)
- ✅ Refactored to modular architecture
- ✅ Separated operations into 8 focused modules
- ✅ Maintained 100% API compatibility
- ✅ All 85 tests passing
- ✅ Added reverse arithmetic operations
- ✅ Added std and var aggregations
- ✅ Improved documentation
- ✅ Zero performance overhead

### Version 0.1.0 (Legacy)
- Initial monolithic implementation
- All core features in single file
- Complete test suite