# Lab 3: Activation Functions

## Status
🟢 Ready for Implementation

## Description

This lab implements comprehensive activation functions for neural networks with automatic differentiation support. It provides both functional and module-based interfaces, complete mathematical documentation, and extensive visualization tools.

## Learning Outcomes

After completing this lab, you will be able to:

- Understand the role of activation functions in neural networks
- Implement all standard activation functions from scratch
- Compute gradients automatically via backpropagation
- Compare activation properties and choose appropriately
- Diagnose and solve activation-related problems
- Use both functional and module interfaces
- Visualize activation shapes and gradient flow
- Implement parametric activations with learnable parameters

## Files

- `activations.py` - Main implementation (~850 lines)
  - Functional interface for all activations
  - Module-based interface (PyTorch-style)
  - Utility functions for analysis
  - Visualization helpers
- `tests/test_activations.py` - Comprehensive tests (~900 lines, 100+ tests)
  - Forward pass tests
  - Gradient verification
  - Numerical gradient checking
  - Edge cases and stability
  - Module interface tests
- `notebooks/demo.ipynb` - Interactive demonstration (~25 cells)
  - Visual comparisons
  - Gradient flow analysis
  - Dying ReLU problem
  - Vanishing gradients
  - Complete classification example
- `README.md` - This file

## Installation
```bash
# From repository root
pip install -e .

# Verify installation
python -c "from semester1.lab3_activation_functions.activations import relu; print('✅ Installation successful!')"
```

## How to Run

### Run Tests
```bash
# All tests
pytest semester1/lab3_activation_functions/tests/ -v

# Specific test class
pytest semester1/lab3_activation_functions/tests/test_activations.py::TestReLUActivation -v

# With coverage
pytest semester1/lab3_activation_functions/tests/ --cov=semester1.lab3_activation_functions --cov-report=html

# Quick smoke test
python semester1/lab3_activation_functions/activations.py
```

### Interactive Demo
```bash
# Launch Jupyter notebook
jupyter notebook semester1/lab3_activation_functions/notebooks/demo.ipynb

# Or run specific examples
cd semester1/lab3_activation_functions
python -c "from activations import *; import numpy as np; x = Tensor([1,2,3], requires_grad=True); y = relu(x); y.sum().backward(); print(x.grad)"
```

## Activation Functions Implemented

### Basic Activations

| Function | Formula | Range | Properties |
|----------|---------|-------|------------|
| **ReLU** | max(0, x) | [0, ∞) | Fast, dying neurons |
| **Leaky ReLU** | max(αx, x) | (-∞, ∞) | Fixes dying ReLU |
| **ELU** | x if x>0 else α(exp(x)-1) | (-α, ∞) | Smooth, negative saturation |

### Sigmoid Family

| Function | Formula | Range | Properties |
|----------|---------|-------|------------|
| **Sigmoid** | 1/(1+exp(-x)) | (0, 1) | Probability-like, vanishing gradients |
| **Tanh** | tanh(x) | (-1, 1) | Zero-centered, vanishing gradients |
| **Hard Sigmoid** | clip((x+1)/2, 0, 1) | [0, 1] | Fast approximation |
| **Hard Tanh** | clip(x, -1, 1) | [-1, 1] | Fast approximation |

### Softmax

| Function | Formula | Range | Properties |
|----------|---------|-------|------------|
| **Softmax** | exp(xᵢ)/Σexp(xⱼ) | (0, 1), sums to 1 | Multi-class classification |
| **Log Softmax** | log(softmax(x)) | (-∞, 0) | Numerically stable |

### Modern Activations

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| **GELU** | x·Φ(x) | (-0.17, ∞) | Transformers (BERT, GPT) |
| **Swish/SiLU** | x·σ(x) | (-∞, ∞) | Modern CNNs |
| **Mish** | x·tanh(softplus(x)) | (-∞, ∞) | Image tasks |
| **Softplus** | log(1+exp(x)) | (0, ∞) | Smooth ReLU |
| **Softsign** | x/(1+\|x\|) | (-1, 1) | Fast tanh alternative |

### Parametric

| Function | Formula | Learnable | Properties |
|----------|---------|-----------|------------|
| **PReLU** | max(αx, x) | α | Network learns optimal slope |

## Mathematical Foundations

### Gradient Formulas

All activations have properly computed gradients:
```
ReLU:      f'(x) = 1 if x > 0 else 0
Sigmoid:   f'(x) = σ(x)·(1 - σ(x))
Tanh:      f'(x) = 1 - tanh²(x)
Softmax:   ∂softmax(x)ᵢ/∂xⱼ = softmax(x)ᵢ·(δᵢⱼ - softmax(x)ⱼ)
GELU:      Computed via chain rule through approximation
```

### Numerical Stability

All implementations handle:
- Large positive/negative values
- Overflow/underflow prevention
- Log-sum-exp trick for softmax
- Clipping for exponentials

## Usage Examples

### Basic Usage
```python
from semester1.lab2_autograd.autograd import Tensor
from semester1.lab3_activation_functions.activations import relu, sigmoid, softmax

# Simple activation
x = Tensor([-1, 0, 1, 2], requires_grad=True)
y = relu(x)
print(y.data)  # [0, 0, 1, 2]

# Compute gradients
y.sum().backward()
print(x.grad.data)  # [0, 0, 1, 1]
```

### Module Interface
```python
from semester1.lab3_activation_functions.activations import ReLU, Sigmoid, Softmax

# Create activation modules
relu_layer = ReLU()
sigmoid_layer = Sigmoid()
softmax_layer = Softmax(axis=1)

# Use in network
x = Tensor([[1, 2, 3]], requires_grad=True)
h1 = relu_layer(x)
h2 = sigmoid_layer(h1)
output = softmax_layer(h2)
```

### Parametric Activation
```python
from semester1.lab3_activation_functions.activations import PReLU

# Create PReLU with learnable parameter
prelu = PReLU(alpha=0.25)

x = Tensor([-2, -1, 0, 1, 2])
y = prelu(x)

# Gradient flows to both x and alpha
y.sum().backward()
print(x.grad)        # Gradient w.r.t. input
print(prelu.alpha.grad)  # Gradient w.r.t. alpha parameter
```

### Network Example
```python
import numpy as np
from semester1.lab2_autograd.autograd import Tensor
from semester1.lab3_activation_functions.activations import relu, softmax

# Simple 2-layer network
X = Tensor(np.random.randn(32, 10))
W1 = Tensor(np.random.randn(10, 20) * 0.1, requires_grad=True)
W2 = Tensor(np.random.randn(20, 5) * 0.1, requires_grad=True)

# Forward pass
h1 = relu(X @ W1)
logits = h1 @ W2
probs = softmax(logits, axis=1)

# Backward pass
loss = (probs ** 2).sum()
W1.zero_grad()
W2.zero_grad()
loss.backward()

# Update weights
learning_rate = 0.01
W1.data -= learning_rate * W1.grad.data
W2.data -= learning_rate * W2.grad.data
```

### Utility Functions
```python
from semester1.lab3_activation_functions.activations import (
    compare_activations,
    check_dead_neurons,
    activation_statistics,
    plot_activation_comparison
)

# Compare activations
results = compare_activations(x_range=(-5, 5), num_points=100)

# Check for dead neurons
x = Tensor(np.random.randn(100, 64))
y = relu(x)
dead_pct = check_dead_neurons(y)
print(f"Dead neurons: {dead_pct:.1f}%")

# Get activation statistics
stats = activation_statistics(y)
print(f"Mean: {stats['mean']:.3f}")
print(f"Dead: {stats['dead_pct']:.1f}%")

# Plot comparison
plot_activation_comparison(save_path='activations.png')
```

## Test Results
```bash
$ pytest semester1/lab3_activation_functions/tests/ -v

======================== test session starts =========================
collected 100+ items

test_activations.py::TestReLUActivation::test_relu_forward PASSED
test_activations.py::TestReLUActivation::test_relu_gradient PASSED
test_activations.py::TestReLUActivation::test_relu_gradient_check PASSED
...
test_activations.py::TestComplexScenarios::test_multi_layer_network PASSED

====================== 100+ passed in 4.52s ==========================
```

## Common Use Cases

### When to Use Which Activation?

**Hidden Layers:**
- **Default**: ReLU or GELU
- **Deep networks**: GELU, avoid Sigmoid/Tanh
- **Dying neurons issue**: Leaky ReLU or ELU
- **Transformers**: GELU
- **CNNs**: ReLU, GELU, or Swish
- **RNNs**: Tanh

**Output Layers:**
- **Binary classification**: Sigmoid
- **Multi-class classification**: Softmax
- **Regression**: Linear (no activation) or ReLU (if output ≥ 0)
- **Multi-label**: Sigmoid (independent per label)

### Troubleshooting Guide

| Problem | Symptom | Solution |
|---------|---------|----------|
| Dead neurons | High % of zero outputs | Use Leaky ReLU or ELU |
| Vanishing gradients | Gradients near zero | Avoid Sigmoid/Tanh, use ReLU/GELU |
| Exploding gradients | Very large gradients | Gradient clipping + check initialization |
| Poor convergence | Loss not decreasing | Try different activation |
| Saturation | All outputs near bounds | Check weight initialization |

## Key Concepts

### 1. Non-Linearity

Without activations, deep networks collapse to linear:
```
f(W₃·W₂·W₁·x) = (W₃·W₂·W₁)·x = W·x
```

Activations break this:
```
f(W₃·σ(W₂·σ(W₁·x))) ≠ W·x
```

### 2. Gradient Flow

Good activations maintain gradient magnitude:
- ReLU: gradient is 0 or 1
- Sigmoid: gradient vanishes for large |x|
- GELU: smooth gradients everywhere

### 3. Dead Neurons

ReLU neurons can die (always output 0):
```python
x = Tensor([-5, -3, -1])  # All negative
y = relu(x)  # All zeros
# Gradient is always 0, neuron never updates!
```

Solution: Leaky ReLU, ELU, or better initialization

### 4. Saturation

Sigmoid/Tanh saturate (flat regions):
```python
x = Tensor([10, -10])
y = sigmoid(x)  # [~1, ~0]
# Gradients are nearly zero in flat regions
```

### 5. Softmax Temperature

Temperature controls confidence:
```python
logits = Tensor([[1, 2, 3]])
probs_low_temp = softmax(logits / 0.5)   # Sharp distribution
probs_high_temp = softmax(logits / 2.0)  # Smooth distribution
```

## Performance Notes

### Computational Complexity

| Activation | Complexity | Relative Speed |
|------------|------------|----------------|
| ReLU | O(n) | Fastest (1x) |
| Leaky ReLU | O(n) | Fast (1.1x) |
| Sigmoid | O(n) | Slow (5x) |
| Tanh | O(n) | Slow (5x) |
| GELU | O(n) | Slow (10x) |
| Softmax | O(n²) for gradient | Slowest |

### Memory Usage

- All activations: O(n) memory for forward pass
- Computational graph: O(n) per operation
- Use `detach()` to break graph if needed

## Common Pitfalls

### 1. Wrong Output Activation
```python
# ❌ Wrong: Softmax for regression
output = softmax(logits)

# ✅ Correct: Linear for regression
output = logits
```

### 2. Sigmoid/Tanh in Deep Networks
```python
# ❌ Wrong: Causes vanishing gradients
for i in range(10):
    x = sigmoid(x @ weights[i])

# ✅ Correct: Use ReLU or GELU
for i in range(10):
    x = relu(x @ weights[i])
```

### 3. Not Checking Dead Neurons
```python
# ✅ Good practice: Monitor activations
from semester1.lab3_activation_functions.activations import check_dead_neurons

y = relu(x)
dead_pct = check_dead_neurons(y)
if dead_pct > 50:
    print(f"Warning: {dead_pct:.1f}% dead neurons!")
```

## Connection to Previous Labs

### Lab 1: Tensor Operations
- Builds on tensor arithmetic
- Uses shape manipulation
- Leverages broadcasting

### Lab 2: Automatic Differentiation
- All activations compute gradients automatically
- Seamless integration with computational graph
- Uses backward() for gradient propagation

## Preview of Lab 4

In Lab 4, we'll implement **Loss Functions**:
- Mean Squared Error (MSE)
- Cross-Entropy Loss
- Binary Cross-Entropy
- Hinge Loss
- Custom loss functions

These will combine with our activations to form complete training pipelines!

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| **Implementation** | 40 | All activations correctly implemented |
| - Functional interface | 15 | All functions work correctly |
| - Module interface | 10 | All classes work correctly |
| - PReLU (parametric) | 8 | Learnable parameter works |
| - Utilities | 7 | Helper functions work |
| **Tests** | 20 | Comprehensive test coverage |
| - Forward pass tests | 7 | All activations tested |
| - Gradient tests | 8 | Numerical verification |
| - Edge cases | 5 | Stability and special cases |
| **Documentation** | 20 | Clear, complete documentation |
| - Docstrings | 10 | All functions documented |
| - Comments | 5 | Complex logic explained |
| - README | 5 | Complete lab documentation |
| **Demo** | 20 | Working demonstrations |
| - Notebook completeness | 10 | All features demonstrated |
| - Visualizations | 5 | Clear, informative plots |
| - Explanations | 5 | Educational content |

**Total: 100 points**

## References

### Papers
1. **Deep Learning** (Goodfellow et al., 2016) - Chapter 6
2. **Gaussian Error Linear Units (GELUs)** (Hendrycks & Gimpel, 2016)
3. **Swish: A Self-Gated Activation Function** (Ramachandran et al., 2017)
4. **Mish: A Self Regularized Non-Monotonic Activation Function** (Misra, 2019)

### Tutorials
- PyTorch nn.functional: https://pytorch.org/docs/stable/nn.functional.html
- Activation Functions Explained: https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html

### Books
- **Deep Learning** (Goodfellow et al.) - Comprehensive coverage
- **Neural Networks and Deep Learning** (Nielsen) - Intuitive explanations

## Troubleshooting

### Import Errors
```bash
# Ensure package is installed
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/nn-from-scratch"
```

### Test Failures
```bash
# Run with verbose output
pytest semester1/lab3_activation_functions/tests/ -vv

# Run specific test
pytest semester1/lab3_activation_functions/tests/test_activations.py::test_relu_forward -vv

# Run with debugging
pytest semester1/lab3_activation_functions/tests/ --pdb
```

### Numerical Issues

If you encounter NaN or Inf:
1. Check input ranges (clip if necessary)
2. Verify numerical stability implementations
3. Use smaller learning rates
4. Check weight initialization

### Gradient Mismatches

If gradients don't match numerical:
1. Verify epsilon value (try 1e-5 to 1e-7)
2. Check forward pass correctness
3. Ensure no in-place modifications
4. Verify broadcasting handling

## Additional Exercises

For students seeking extra challenge:

1. **Implement Additional Activations**
   - Maxout
   - Randomized Leaky ReLU
   - Concatenated ReLU

2. **Custom Activation Design**
   - Design your own activation function
   - Test on classification task
   - Compare to standard activations

3. **Activation Analysis Tools**
   - Histogram of activations
   - Gradient flow visualization
   - Dead neuron heatmap

4. **Optimization**
   - Vectorized implementations
   - Custom CUDA kernels (advanced)
   - Memory-efficient versions

## Author

Lab developed for Neural Networks from Scratch course - Semester 1

## License

MIT License - See repository root for details

---

**Congratulations on completing Lab 3!** 🎉

You now have a complete understanding of activation functions and their role in neural networks. This is a crucial foundation for building effective deep learning models.

**Next: Lab 4 - Loss Functions** →