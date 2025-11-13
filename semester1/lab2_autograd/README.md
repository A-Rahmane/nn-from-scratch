# Lab 2: Automatic Differentiation (Autograd)

## Status
🟢 Completed

## Description

This lab implements automatic differentiation (autograd) capabilities, extending the Tensor class from Lab 1 with computational graph tracking and automatic gradient computation. This is the foundation for training neural networks using backpropagation.

## Learning Outcomes

After completing this lab, you will be able to:

- Understand how automatic differentiation works
- Implement forward and backward passes through computational graphs
- Apply the chain rule for gradient propagation
- Build and traverse computational graphs using topological sorting
- Use context managers to control gradient tracking
- Verify gradients using numerical approximation
- Debug gradient computation issues
- Understand the connection to modern frameworks like PyTorch

## Files

- `autograd.py` - Main autograd implementation (~800 lines)
- `tests/test_autograd.py` - Comprehensive unit tests (~600 lines, 70+ tests)
- `notebooks/demo.ipynb` - Interactive demonstrations
- `README.md` - This file

## Key Concepts

### 1. Computational Graph

A **computational graph** is a directed acyclic graph (DAG) where:
- **Nodes** represent tensors (data)
- **Edges** represent operations

Example:
```
x = [2.0]  →  x² = [4.0]  →  y = [16.0]
                ↑
              square
```

### 2. Forward and Backward Pass

**Forward Pass**: Compute output values
```python
x = Tensor([2.0], requires_grad=True)
y = x ** 2  # Forward: compute y = 4.0
```

**Backward Pass**: Compute gradients via chain rule
```python
y.backward()  # Backward: compute dy/dx = 2x = 4.0
print(x.grad)  # Tensor([4.0])
```

### 3. Chain Rule

The foundation of backpropagation:
```
∂L/∂x = ∂L/∂y × ∂y/∂x
```

For a chain: `L = f(g(h(x)))`
```
∂L/∂x = (∂L/∂f) × (∂f/∂g) × (∂g/∂h) × (∂h/∂x)
```

### 4. Gradient Accumulation

When a tensor appears multiple times in a computation, gradients accumulate:
```python
x = Tensor([3.0], requires_grad=True)
y = x * x * x  # x appears 3 times
y.backward()
# Gradient accumulates: dy/dx = 3x² = 27
```

### 5. Topological Sorting

Ensures gradients are computed in the correct order:
1. Start from output tensor
2. Traverse graph backward via DFS
3. Process tensors in reverse topological order

## Implementation Highlights

### Core Features

1. **Extended Tensor Class**
   - `requires_grad`: Flag to enable gradient tracking
   - `grad`: Accumulated gradient
   - `grad_fn`: Function to compute gradients in backward pass
   - `_prev`: Set of parent tensors
   - `_op`: Operation name for debugging

2. **Gradient-Aware Operations**
   - All arithmetic operations: `+, -, *, /, **`
   - Aggregations: `sum, mean, max`
   - Shape operations: `reshape, transpose`
   - Matrix operations: `matmul (@)`

3. **Context Managers**
   - `no_grad()`: Disable gradient tracking
   - `enable_grad()`: Enable gradient tracking

4. **Utilities**
   - `zero_grad()`: Reset gradients
   - `clip_grad_value()`: Clip gradient values
   - `clip_grad_norm()`: Clip gradient norms
   - `numerical_gradient()`: Compute numerical gradients
   - `check_gradients()`: Verify autograd vs numerical

### Design Decisions

#### 1. Gradient Storage
Gradients are stored as Tensors (not NumPy arrays) to maintain consistency and allow gradient-of-gradient computations (higher-order derivatives).

#### 2. Broadcasting Handling
Gradients are properly reduced when broadcasting occurs:
```python
x = Tensor([[1, 2], [3, 4]], requires_grad=True)  # (2, 2)
y = Tensor([10, 20], requires_grad=True)           # (2,)
z = x + y  # y broadcasts to (2, 2)
z.sum().backward()
# y.grad is (2,) with summed gradients: [2, 2]
```

#### 3. Memory Management
- Computational graphs are built dynamically
- Use `detach()` to break graph connections
- Use `no_grad()` to prevent graph building

#### 4. Backward Pass Algorithm
```python
def backward(self, gradient=None):
    # 1. Build topological order via DFS
    # 2. Initialize output gradient
    # 3. Process tensors in reverse order
    # 4. Each tensor calls its grad_fn
    # 5. Gradients accumulate at each tensor
```

## How to Run

### Installation
```bash
# From repository root
pip install -e .
```

### Run Tests
```bash
# All tests
pytest semester1/lab2_autograd/tests/ -v

# Specific test class
pytest semester1/lab2_autograd/tests/test_autograd.py::TestBasicGradients -v

# With coverage
pytest semester1/lab2_autograd/tests/ --cov=semester1.lab2_autograd --cov-report=html

# Run gradient checking tests only
pytest semester1/lab2_autograd/tests/ -v -k "gradient_check"
```

### Interactive Demo
```bash
# Launch Jupyter notebook
jupyter notebook semester1/lab2_autograd/notebooks/demo.ipynb

# Or run the main file for quick tests
python semester1/lab2_autograd/autograd.py
```

## Mathematical Foundations

### Gradient Formulas

| Operation | Forward | Gradient |
|-----------|---------|----------|
| Addition | `y = a + b` | `∂y/∂a = 1, ∂y/∂b = 1` |
| Multiplication | `y = a * b` | `∂y/∂a = b, ∂y/∂b = a` |
| Power | `y = x^n` | `∂y/∂x = n * x^(n-1)` |
| Sum | `y = sum(x)` | `∂y/∂x = 1 (broadcast)` |
| Mean | `y = mean(x)` | `∂y/∂x = 1/n (broadcast)` |
| Max | `y = max(x)` | `∂y/∂x = 1 at max, 0 elsewhere` |
| Matmul | `C = A @ B` | `∂C/∂A = grad @ B^T`<br>`∂C/∂B = A^T @ grad` |
| Reshape | `y = reshape(x)` | `∂y/∂x = reshape_back(grad)` |
| Transpose | `y = x^T` | `∂y/∂x = grad^T` |

### Broadcasting Gradient Rules

When broadcasting occurs, gradients must be reduced:

1. **Sum over new dimensions**:
```python
x: (3,) → broadcast to (2, 3) → sum grad over axis 0 → (3,)
```

2. **Sum over dimensions of size 1**:
```python
x: (2, 1) → broadcast to (2, 3) → sum grad over axis 1 → (2, 1)
```

## Test Results
```bash
$ pytest semester1/lab2_autograd/tests/ -v

========================== test session starts ===========================
collected 74 items

test_autograd.py::TestTensorCreationWithGrad::test_default_requires_grad PASSED [  1%]
test_autograd.py::TestTensorCreationWithGrad::test_explicit_requires_grad PASSED [  2%]
...
test_autograd.py::TestGradientChecking::test_gradient_check_complex_function PASSED [100%]

========================== 74 passed in 3.45s ============================
```

## Usage Examples

### Basic Gradient Computation
```python
from semester1.lab2_autograd.autograd import Tensor

# Simple function
x = Tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # Tensor([4.0])
```

### Linear Regression
```python
import numpy as np
from semester1.lab2_autograd.autograd import Tensor, no_grad

# Data
X = Tensor(np.random.randn(100, 1))
y_true = Tensor(3 * X.data + 2 + np.random.randn(100, 1) * 0.1)

# Parameters
w = Tensor(np.random.randn(1, 1), requires_grad=True)
b = Tensor(np.zeros((1, 1)), requires_grad=True)

# Training loop
for epoch in range(100):
    # Forward
    y_pred = X @ w + b
    loss = ((y_pred - y_true) ** 2).mean()
    
    # Backward
    w.zero_grad()
    b.zero_grad()
    loss.backward()
    
    # Update
    with no_grad():
        w.data -= 0.01 * w.grad.data
        b.data -= 0.01 * b.grad.data

print(f"w = {w.data[0,0]:.2f}, b = {b.data[0,0]:.2f}")
```

### Two-Layer Neural Network
```python
# Initialize weights
W1 = Tensor(np.random.randn(2, 10) * 0.1, requires_grad=True)
b1 = Tensor(np.zeros((1, 10)), requires_grad=True)
W2 = Tensor(np.random.randn(10, 1) * 0.1, requires_grad=True)
b2 = Tensor(np.zeros((1, 1)), requires_grad=True)

# Forward pass
def forward(X):
    h = (X @ W1 + b1) * ((X @ W1 + b1).data > 0)  # ReLU
    return h @ W2 + b2

# Training
for epoch in range(200):
    y_pred = forward(X_train)
    loss = ((y_pred - y_train) ** 2).mean()
    
    for param in [W1, b1, W2, b2]:
        param.zero_grad()
    
    loss.backward()
    
    with no_grad():
        for param in [W1, b1, W2, b2]:
            param.data -= 0.1 * param.grad.data
```

## Performance Notes

### Computational Complexity

- **Forward pass**: O(n) where n = number of operations
- **Backward pass**: O(n) - same as forward
- **Memory**: O(n) for storing computational graph

### Optimization Tips

1. **Use no_grad() for inference**:
```python
with no_grad():
    predictions = model(X_test)  # No graph building
```

2. **Detach intermediate results**:
```python
intermediate = (x ** 2).detach()  # Break gradient flow
```

3. **Zero gradients before each backward**:
```python
for param in parameters:
    param.zero_grad()
loss.backward()
```

## Common Pitfalls and Solutions

### Pitfall 1: Gradient Accumulation

**Problem**: Gradients accumulate across backward passes
```python
x = Tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()  # x.grad = [4.0]
y.backward()  # x.grad = [8.0] ❌
```

**Solution**: Zero gradients before each backward
```python
x.zero_grad()
y.backward()  # x.grad = [4.0] ✓
```

### Pitfall 2: Backward on Non-Scalar

**Problem**: Can't call backward on non-scalar without gradient
```python
x = Tensor([[1, 2], [3, 4]], requires_grad=True)
y = x * 2
y.backward()  # Error! y is not scalar
```

**Solution**: Reduce to scalar first
```python
loss = y.sum()
loss.backward()  # ✓
```

### Pitfall 3: In-Place Modifications

**Problem**: Modifying tensors in-place breaks gradients
```python
x = Tensor([1.0], requires_grad=True)
x.data *= 2  # ❌ Breaks gradient computation
```

**Solution**: Create new tensors
```python
x = x * 2  # ✓ Creates new tensor
```

### Pitfall 4: Using Operations in no_grad

**Problem**: Operations in no_grad don't track gradients
```python
x = Tensor([2.0], requires_grad=True)
with no_grad():
    y = x ** 2
y.backward()  # Error! y doesn't require grad
```

**Solution**: Only use no_grad for operations that shouldn't track
```python
y = x ** 2  # Track gradients
y.backward()

with no_grad():
    # Use for parameter updates
    x.data -= learning_rate * x.grad.data
```

## Debugging Guide

### 1. Check Gradients Numerically
```python
def func(x):
    return (x ** 3 + 2 * x).sum()

x = Tensor([1.0, 2.0], requires_grad=True)
match, diff = check_gradients(func, x)
print(f"Match: {match}, Diff: {diff:.2e}")
```

### 2. Verify Graph Structure
```python
x = Tensor([1.0], requires_grad=True)
y = x ** 2
print(f"y._op: {y._op}")  # Should be 'pow(2)'
print(f"y._prev: {len(y._prev)}")  # Should be 1
print(f"y.requires_grad: {y.requires_grad}")  # Should be True
```

### 3. Check Gradient Shapes
```python
x = Tensor([[1, 2], [3, 4]], requires_grad=True)
y = x.sum()
y.backward()
assert x.grad.shape == x.shape, "Gradient shape mismatch!"
```

### 4. Trace Gradient Flow
```python
x = Tensor([2.0], requires_grad=True)
y = x ** 2
print(f"Before backward: x.grad = {x.grad}")
y.backward()
print(f"After backward: x.grad = {x.grad}")
```

## Connection to Lab 1

Lab 2 builds directly on Lab 1:

- **Inherits from Lab 1 Tensor**: All Lab 1 operations still work
- **Extends with Autograd**: Adds gradient tracking on top
- **Backward Compatibility**: Lab 1 tests still pass
- **Same Interface**: Familiar API with new `requires_grad` flag

## Preview of Lab 3

In Lab 3, we'll implement **Activation Functions**:

- ReLU, Sigmoid, Tanh, Softmax
- Their gradients for backpropagation
- Proper handling of edge cases
- Building blocks for neural networks

With autograd from Lab 2, implementing these will be straightforward!

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| **Implementation** | 40 | All features correctly implemented |
| - Basic operations | 10 | Add, mul, pow gradients |
| - Aggregations | 8 | Sum, mean, max gradients |
| - Matrix ops | 8 | Matmul gradients |
| - Shape ops | 7 | Reshape, transpose gradients |
| - Utilities | 7 | Context managers, gradient checking |
| **Tests** | 20 | Comprehensive test coverage |
| - Basic gradients | 8 | All operations tested |
| - Complex graphs | 7 | Chain rule, accumulation |
| - Edge cases | 5 | Error handling, special cases |
| **Documentation** | 20 | Clear, complete docs |
| - Docstrings | 10 | All functions documented |
| - Comments | 5 | Complex logic explained |
| - README | 5 | Complete lab documentation |
| **Demo** | 20 | Working demonstrations |
| - Notebook completeness | 10 | All features demonstrated |
| - Examples | 10 | Clear, educational examples |

**Total: 100 points**

## References

### Papers
1. **Automatic Differentiation in Machine Learning: a Survey**
   - Baydin et al., 2018
   - Comprehensive overview of autodiff techniques

2. **Efficient BackProp**
   - LeCun et al., 1998
   - Practical tips for gradient-based learning

### Tutorials
1. **PyTorch Autograd Tutorial**: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
2. **JAX Autodiff Cookbook**: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
3. **Automatic Differentiation (Colah's Blog)**: http://colah.github.io/posts/2015-08-Backprop/

### Books
1. **Deep Learning** (Goodfellow et al.)
   - Chapter 6: Deep Feedforward Networks
   - Detailed backpropagation explanation

2. **Dive into Deep Learning**
   - Chapter 2.5: Automatic Differentiation
   - Interactive examples with code

## Troubleshooting

### Import Errors
```bash
# If you get import errors
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/nn-from-scratch"
```

### Test Failures
```bash
# Run with verbose output
pytest semester1/lab2_autograd/tests/ -vv

# Run specific test
pytest semester1/lab2_autograd/tests/test_autograd.py::test_name -vv

# Run with debugging
pytest semester1/lab2_autograd/tests/ -vv --pdb
```

### Numerical Gradient Mismatches
If numerical gradients don't match:

1. Check epsilon value (try 1e-5 to 1e-7)
2. Verify forward pass is correct
3. Check for in-place modifications
4. Ensure proper broadcasting handling

### Memory Issues
If running out of memory:

1. Use `detach()` for intermediate results
2. Use `no_grad()` for inference
3. Process data in smaller batches
4. Delete unused computational graphs

## Additional Exercises

For additional practice:

1. **Implement Second-Order Gradients**
   - Compute gradient of gradients
   - Hessian matrix computation

2. **Add More Operations**
   - Exponential: `exp(x)`
   - Logarithm: `log(x)`
   - Trigonometric: `sin(x)`, `cos(x)`

3. **Optimize Performance**
   - Cache repeated computations
   - Implement operation fusion
   - Add memory pooling

4. **Advanced Features**
   - Checkpoint for memory efficiency
   - Custom autograd functions
   - Gradient hooks for debugging

## Author
Lab developed for Neural Networks from Scratch course - Semester 1

## License
MIT License - See repository root for details

---

**Congratulations on completing Lab 2!** 🎉

You now understand the core mechanism behind modern deep learning frameworks. This autograd system is conceptually similar to what PyTorch and TensorFlow use internally.

Next: **Lab 3 - Activation Functions** →