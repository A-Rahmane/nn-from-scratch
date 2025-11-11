# Lab 1: Tensor Operations

## Objectives
- Understand the fundamental data structure (tensors) used in neural networks
- Implement basic tensor operations from scratch using only NumPy
- Learn about broadcasting, reshaping, and indexing operations
- Build a foundation for automatic differentiation in future labs
- Write comprehensive unit tests for tensor operations

## Background

### What is a Tensor?
A **tensor** is a generalization of vectors and matrices to higher dimensions. In neural networks, tensors are the fundamental data structure for representing:
- **Scalars** (0D tensor): A single number
- **Vectors** (1D tensor): An array of numbers
- **Matrices** (2D tensor): A 2D array of numbers
- **Higher-dimensional tensors**: 3D, 4D, or more

### Why Build Tensors from Scratch?
Understanding tensor operations at a fundamental level is crucial because:
1. **Foundation of Deep Learning**: All neural network operations are tensor operations
2. **Computational Efficiency**: Vectorized operations are much faster than loops
3. **Preparation for Autograd**: Understanding forward operations is essential for backward passes
4. **Framework Independence**: These concepts apply to PyTorch, TensorFlow, and JAX

### Key Concepts

#### Broadcasting
Broadcasting allows operations between tensors of different shapes by automatically expanding dimensions:
```python
# Example:
a = [1, 2, 3]        # shape (3,)
b = [[1], [2], [3]]  # shape (3, 1)
# a + b will broadcast to shape (3, 3)
```

#### Reshaping
Changing the shape of a tensor without changing its data:
```python
# Example:
x = [1, 2, 3, 4, 5, 6]  # shape (6,)
x.reshape(2, 3)          # shape (2, 3) -> [[1,2,3], [4,5,6]]
```

---

## Implementation Tasks

### Task 1: Basic Tensor Class

**Description**: Implement a `Tensor` class that wraps NumPy arrays and provides essential properties.

**Requirements**:
1. Store data as a NumPy array with `float32` dtype
2. Implement `shape` property to return tensor dimensions
3. Implement `ndim` property to return number of dimensions
4. Implement `size` property to return total number of elements
5. Implement `__repr__` for readable string representation
6. Implement `__str__` for user-friendly output

**Hints**:
- Use `np.array()` with `dtype=np.float32` for consistency
- Handle various input types: lists, tuples, NumPy arrays, scalars
- Properties use `@property` decorator

**Expected Behavior**:
```python
t = Tensor([[1, 2, 3], [4, 5, 6]])
print(t.shape)  # (2, 3)
print(t.ndim)   # 2
print(t.size)   # 6
```

---

### Task 2: Arithmetic Operations

**Description**: Implement element-wise arithmetic operations with proper broadcasting support.

**Requirements**:
1. Implement `__add__` and `__radd__` for addition
2. Implement `__sub__` and `__rsub__` for subtraction
3. Implement `__mul__` and `__rmul__` for multiplication
4. Implement `__truediv__` and `__rtruediv__` for division
5. Implement `__pow__` for exponentiation
6. Implement `__neg__` for negation
7. Support operations with other Tensors, NumPy arrays, and scalars
8. Return new Tensor objects (don't modify in place)

**Hints**:
- Use NumPy's built-in broadcasting
- Handle type checking: if not Tensor, convert to Tensor
- `__radd__` handles `5 + tensor` (reverse addition)
- Each operation should return a new `Tensor` instance

**Expected Behavior**:
```python
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])
c = a + b           # Tensor([5, 7, 9])
d = a * 2           # Tensor([2, 4, 6])
e = 10 - a          # Tensor([9, 8, 7])
```

---

### Task 3: Comparison Operations

**Description**: Implement comparison operations that return boolean tensors.

**Requirements**:
1. Implement `__eq__` for equality
2. Implement `__ne__` for inequality
3. Implement `__lt__` for less than
4. Implement `__le__` for less than or equal
5. Implement `__gt__` for greater than
6. Implement `__ge__` for greater than or equal
7. Return Tensor with boolean data type

**Hints**:
- Use NumPy comparison operators
- Result should be a Tensor with `dtype=bool`
- Support comparison with scalars and other tensors

**Expected Behavior**:
```python
a = Tensor([1, 2, 3])
b = Tensor([2, 2, 2])
result = a > b      # Tensor([False, False, True])
```

---

### Task 4: Indexing and Slicing

**Description**: Implement NumPy-style indexing and slicing operations.

**Requirements**:
1. Implement `__getitem__` for indexing and slicing
2. Implement `__setitem__` for item assignment
3. Support single index: `tensor[0]`
4. Support slicing: `tensor[1:3]`
5. Support multi-dimensional indexing: `tensor[0, 1]`
6. Support boolean indexing: `tensor[tensor > 0]`
7. Return Tensor objects (not NumPy arrays)

**Hints**:
- Delegate to NumPy array's indexing
- Wrap results in Tensor class
- For `__setitem__`, convert input to NumPy array if needed

**Expected Behavior**:
```python
t = Tensor([[1, 2, 3], [4, 5, 6]])
print(t[0])        # Tensor([1, 2, 3])
print(t[:, 1])     # Tensor([2, 5])
t[0, 0] = 10       # Modifies tensor
```

---

### Task 5: Shape Manipulation

**Description**: Implement methods for reshaping and transposing tensors.

**Requirements**:
1. Implement `reshape(*shape)` method
2. Implement `transpose(*axes)` method
3. Implement `T` property for matrix transpose
4. Implement `flatten()` method
5. Implement `squeeze()` to remove dimensions of size 1
6. Implement `unsqueeze(dim)` to add dimension of size 1
7. Return new Tensor objects (don't modify in place)

**Hints**:
- Use NumPy's `reshape()`, `transpose()`, `flatten()`, `squeeze()`, `expand_dims()`
- Handle variable arguments: `reshape(2, 3)` or `reshape((2, 3))`
- For transpose with no arguments, reverse all axes

**Expected Behavior**:
```python
t = Tensor([1, 2, 3, 4, 5, 6])
t2 = t.reshape(2, 3)           # shape (2, 3)
t3 = t2.T                       # shape (3, 2)
t4 = t2.flatten()              # shape (6,)
t5 = t.unsqueeze(0)            # shape (1, 6)
```

---

### Task 6: Aggregation Operations

**Description**: Implement reduction operations that aggregate tensor values.

**Requirements**:
1. Implement `sum(axis=None, keepdims=False)` method
2. Implement `mean(axis=None, keepdims=False)` method
3. Implement `max(axis=None, keepdims=False)` method
4. Implement `min(axis=None, keepdims=False)` method
5. Implement `std(axis=None, keepdims=False)` for standard deviation
6. Implement `var(axis=None, keepdims=False)` for variance
7. Support both full reduction and axis-specific reduction
8. Support `keepdims` to maintain dimensionality

**Hints**:
- Use NumPy's reduction functions
- When `axis=None`, reduce over all dimensions
- `keepdims=True` preserves the number of dimensions

**Expected Behavior**:
```python
t = Tensor([[1, 2, 3], [4, 5, 6]])
print(t.sum())          # Tensor(21)
print(t.sum(axis=0))    # Tensor([5, 7, 9])
print(t.mean())         # Tensor(3.5)
print(t.max(axis=1))    # Tensor([3, 6])
```

---

### Task 7: Matrix Operations

**Description**: Implement linear algebra operations essential for neural networks.

**Requirements**:
1. Implement `matmul(other)` or `__matmul__` for matrix multiplication
2. Implement `dot(other)` for dot product
3. Support `@` operator for matrix multiplication
4. Implement proper dimension checking and error messages
5. Support batch matrix multiplication for 3D tensors

**Hints**:
- Use `np.matmul()` or `@` operator
- For 2D tensors: matrix multiplication
- For 1D tensors: dot product
- For higher dimensions: batch operations
- Raise `ValueError` for incompatible shapes

**Expected Behavior**:
```python
a = Tensor([[1, 2], [3, 4]])
b = Tensor([[5, 6], [7, 8]])
c = a @ b                    # Matrix multiplication
# Result: [[19, 22], [43, 50]]

v1 = Tensor([1, 2, 3])
v2 = Tensor([4, 5, 6])
d = v1.dot(v2)              # Dot product: 32
```

---

### Task 8: Advanced Operations

**Description**: Implement utility methods useful for neural networks.

**Requirements**:
1. Implement `clone()` to create a deep copy
2. Implement `zeros_like()` class method
3. Implement `ones_like()` class method
4. Implement `rand_like()` class method for random values
5. Implement `randn_like()` class method for normal distribution
6. Implement `concatenate(tensors, axis=0)` class method
7. Implement `stack(tensors, axis=0)` class method
8. Implement `split(sections, axis=0)` method

**Hints**:
- Use `np.copy()` for deep copy
- Class methods use `@classmethod` decorator
- `concatenate` joins tensors along existing axis
- `stack` creates new axis
- Handle list of tensors as input

**Expected Behavior**:
```python
t = Tensor([[1, 2], [3, 4]])
t2 = t.clone()                           # Deep copy

zeros = Tensor.zeros_like(t)            # [[0, 0], [0, 0]]
ones = Tensor.ones_like(t)              # [[1, 1], [1, 1]]

a = Tensor([1, 2])
b = Tensor([3, 4])
c = Tensor.concatenate([a, b])          # [1, 2, 3, 4]
d = Tensor.stack([a, b])                # [[1, 2], [3, 4]]
```