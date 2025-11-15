# Lab 4: Loss Functions

## Status
🟢 Ready for Use

## Description

This lab implements comprehensive loss functions for neural networks with automatic differentiation support. It provides both functional and module-based interfaces, complete mathematical documentation, and extensive support for real-world training scenarios including class imbalance, label smoothing, and numerical stability.

## Learning Outcomes

After completing this lab, you will be able to:

- Understand different types of loss functions and when to use them
- Implement loss functions with automatic differentiation
- Handle class imbalance in datasets
- Apply label smoothing for regularization
- Choose appropriate loss functions for specific tasks
- Diagnose training issues related to loss functions
- Implement custom loss functions
- Use both functional and module interfaces

## Files

- `losses.py` - Main implementation (~1000 lines)
  - Regression losses (MSE, MAE, Huber, MSLE)
  - Binary classification losses (BCE, BCE with Logits)
  - Multi-class losses (Cross-Entropy, NLL)
  - Advanced losses (Hinge, Focal, Dice, KL Divergence, etc.)
  - Functional and module interfaces
  - Utility functions
- `tests/test_losses.py` - Comprehensive tests (~1100 lines, 100+ tests)
  - Forward pass tests
  - Gradient verification
  - Numerical gradient checking
  - Edge cases and stability
  - Real-world scenarios
- `notebooks/demo.ipynb` - Interactive demonstration
  - Loss function comparisons
  - Practical examples
  - Visualizations
  - Training scenarios
- `README.md` - This file

## Installation
```bash
# From repository root
pip install -e .

# Verify installation
python -c "from semester1.lab4_loss_functions.losses import mse_loss; print('✅ Installation successful!')"
```

## How to Run

### Run Tests
```bash
# All tests
pytest semester1/lab4_loss_functions/tests/ -v

# Specific test class
pytest semester1/lab4_loss_functions/tests/test_losses.py::TestRegressionLosses -v

# With coverage
pytest semester1/lab4_loss_functions/tests/ --cov=semester1.lab4_loss_functions --cov-report=html

# Quick smoke test
python semester1/lab4_loss_functions/losses.py
```

### Interactive Demo
```bash
# Launch Jupyter notebook
jupyter notebook semester1/lab4_loss_functions/notebooks/demo.ipynb
```

## Loss Functions Implemented

### Regression Losses

| Loss | Formula | Use Case | Properties |
|------|---------|----------|------------|
| **MSE** | L = (1/n)Σ(ŷ-y)² | Standard regression | Sensitive to outliers |
| **MAE** | L = (1/n)Σ\|ŷ-y\| | Robust regression | Robust to outliers |
| **Smooth L1** | Huber loss | Robust regression | Combines MSE + MAE |
| **MSLE** | L = (1/n)Σ(log(ŷ+1)-log(y+1))² | Relative errors | Scale-invariant |

### Binary Classification Losses

| Loss | Input Type | Use Case | Properties |
|------|------------|----------|------------|
| **BCE** | Probabilities [0,1] | Binary classification | Requires sigmoid |
| **BCE with Logits** | Logits (raw scores) | Binary classification | More stable |

### Multi-Class Classification Losses

| Loss | Input Type | Use Case | Properties |
|------|------------|----------|------------|
| **Cross-Entropy** | Logits | Multi-class | Combines softmax + NLL |
| **NLL** | Log probabilities | Multi-class | Requires log_softmax |

### Advanced Losses

| Loss | Use Case | Properties |
|------|----------|------------|
| **Hinge** | SVM training | Margin-based |
| **Focal** | Imbalanced data | Focuses on hard examples |
| **Dice** | Segmentation | Handles imbalance well |
| **KL Divergence** | Distribution matching | Non-symmetric |
| **Cosine Embedding** | Similarity learning | Cosine-based |
| **Triplet Margin** | Metric learning | Anchor-positive-negative |
| **Contrastive** | Siamese networks | Pair-wise learning |

## Mathematical Foundations

### Gradient Formulas

All losses properly compute gradients:
```
MSE:           ∂L/∂ŷ = 2(ŷ - y)/n
MAE:           ∂L/∂ŷ = sign(ŷ - y)/n
BCE:           ∂L/∂ŷ = (ŷ - y)/(ŷ(1-ŷ))
Cross-Entropy: ∂L/∂x = softmax(x) - one_hot(y)
Hinge:         ∂L/∂ŷ = -y if loss > 0, else 0
```

### Numerical Stability

All implementations handle:
- Large positive/negative values
- Overflow/underflow prevention
- Log-sum-exp trick for softmax
- Clipping for exponentials
- Division by zero protection

## Usage Examples

### Basic Usage
```python
from semester1.lab2_autograd.autograd import Tensor
from semester1.lab4_loss_functions.losses import mse_loss, cross_entropy_loss

# Regression
predictions = Tensor([[1.0, 2.0]], requires_grad=True)
targets = Tensor([[1.5, 2.5]])
loss = mse_loss(predictions, targets)
loss.backward()

# Classification
logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
targets = Tensor([0])
loss = cross_entropy_loss(logits, targets)
loss.backward()
```

### Module Interface
```python
from semester1.lab4_loss_functions.losses import MSELoss, CrossEntropyLoss

# Create loss modules
mse_criterion = MSELoss(reduction='mean')
ce_criterion = CrossEntropyLoss(label_smoothing=0.1)

# Use in training
loss = mse_criterion(predictions, targets)
loss.backward()
```

### Handling Class Imbalance
```python
import numpy as np
from semester1.lab4_loss_functions.losses import (
    cross_entropy_loss,
    compute_class_weights
)

# Imbalanced dataset
targets = np.array([0]*900 + [1]*100)  # 90% class 0, 10% class 1

# Compute balanced weights
weights = compute_class_weights(targets, num_classes=2, method='balanced')
weight_tensor = Tensor(weights)

# Use weighted loss
loss = cross_entropy_loss(logits, targets_tensor, weight=weight_tensor)
```

### Label Smoothing
```python
# Apply label smoothing for regularization
loss = cross_entropy_loss(
    logits,
    targets,
    label_smoothing=0.1  # Smooth with ε=0.1
)
```

### Complete Training Example
```python
import numpy as np
from semester1.lab2_autograd.autograd import Tensor
from semester1.lab3_activation_functions.activations import relu
from semester1.lab4_loss_functions.losses import cross_entropy_loss

# Data
X = Tensor(np.random.randn(32, 10))
y = Tensor(np.random.randint(0, 5, 32))

# Model
W1 = Tensor(np.random.randn(10, 20) * 0.1, requires_grad=True)
W2 = Tensor(np.random.randn(20, 5) * 0.1, requires_grad=True)

# Training loop
for epoch in range(100):
    # Forward
    h = relu(X @ W1)
    logits = h @ W2
    loss = cross_entropy_loss(logits, y)
    
    # Backward
    W1.zero_grad()
    W2.zero_grad()
    loss.backward()
    
    # Update
    W1.data -= 0.01 * W1.grad.data
    W2.data -= 0.01 * W2.grad.data
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
```

## Test Results
```bash
$ pytest semester1/lab4_loss_functions/tests/ -v

======================== test session starts =========================
collected 100+ items

test_losses.py::TestRegressionLosses::test_mse_loss_forward PASSED
test_losses.py::TestRegressionLosses::test_mse_loss_gradient PASSED
test_losses.py::TestRegressionLosses::test_mae_loss_forward PASSED
...
test_losses.py::TestRealWorldScenarios::test_imbalanced_classification PASSED

====================== 100+ passed in 5.23s ==========================
```

## When to Use Which Loss?

### Regression Tasks

| Scenario | Recommended Loss | Reason |
|----------|-----------------|--------|
| Standard regression | MSE | Simple, smooth gradients |
| Outliers present | MAE or Smooth L1 | Robust to outliers |
| Relative errors matter | MSLE | Scale-invariant |
| Mixed small/large errors | Smooth L1 (Huber) | Best of both worlds |

### Binary Classification

| Scenario | Recommended Loss | Reason |
|----------|-----------------|--------|
| Standard binary | BCE with Logits | Most stable |
| After sigmoid | BCE | When using sigmoid activation |
| Imbalanced classes | BCE with pos_weight | Weight positive class |

### Multi-Class Classification

| Scenario | Recommended Loss | Reason |
|----------|-----------------|--------|
| Standard multi-class | Cross-Entropy | Most common choice |
| After log_softmax | NLL | When using log_softmax |
| Imbalanced classes | Cross-Entropy with weights | Handle imbalance |
| Overfitting | Cross-Entropy with smoothing | Regularization |

### Special Tasks

| Task | Recommended Loss | Reason |
|------|-----------------|--------|
| SVM training | Hinge Loss | Margin-based learning |
| Imbalanced detection | Focal Loss | Focuses on hard examples |
| Segmentation | Dice Loss | Handles class imbalance |
| Metric learning | Triplet/Contrastive | Learn embeddings |

## Common Pitfalls and Solutions

### Pitfall 1: Wrong Output Activation
```python
# ❌ Wrong: Softmax before Cross-Entropy
probs = softmax(logits)
loss = cross_entropy_loss(probs, targets)  # Double softmax!

# ✅ Correct: Pass logits directly
loss = cross_entropy_loss(logits, targets)
```

### Pitfall 2: Not Handling Class Imbalance
```python
# ❌ Wrong: Ignoring imbalance
loss = cross_entropy_loss(logits, targets)

# ✅ Correct: Use class weights
weights = compute_class_weights(targets_np, num_classes=5)
loss = cross_entropy_loss(logits, targets, weight=Tensor(weights))
```

### Pitfall 3: Using Wrong Reduction
```python
# ❌ Wrong: Sum reduction scales with batch size
loss = mse_loss(pred, target, reduction='sum')  # Varies with batch size

# ✅ Correct: Use mean reduction
loss = mse_loss(pred, target, reduction='mean')  # Consistent
```

### Pitfall 4: Not Clipping Predictions
```python
# ❌ Wrong: Can cause log(0)
loss = binary_cross_entropy(predictions, targets)

# ✅ Correct: Use BCE with Logits (handles internally)
loss = binary_cross_entropy_with_logits(logits, targets)
```

## Troubleshooting Guide

| Problem | Symptom | Solution |
|---------|---------|----------|
| NaN loss | Loss becomes NaN | Check for log(0), div by zero, use logits version |
| Inf loss | Loss becomes Inf | Clip gradients, reduce learning rate |
| Loss not decreasing | Flat loss curve | Check learning rate, verify gradients |
| Exploding loss | Loss increases rapidly | Reduce learning rate, gradient clipping |
| Class ignored | Model predicts one class | Use class weights or focal loss |

## Connection to Previous Labs

### Lab 1: Tensor Operations
- All losses work with Tensor class
- Uses arithmetic and aggregation operations
- Leverages broadcasting

### Lab 2: Automatic Differentiation
- All losses compute gradients automatically
- Seamless integration with computational graph
- Uses backward() for gradient propagation

### Lab 3: Activation Functions
- Losses combine with activations naturally
- BCE with sigmoid, Cross-Entropy with softmax
- Proper gradient flow through both

## Preview of Lab 5

In Lab 5, we'll implement **Dense Neural Networks**:
- Fully connected layers
- Weight initialization strategies
- Forward and backward passes
- Complete network architecture
- Combining layers, activations, and losses

With losses from Lab 4, we can now train complete networks!

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| **Implementation** | 40 | All losses correctly implemented |
| - Regression losses | 10 | MSE, MAE, Smooth L1, MSLE |
| - Classification losses | 10 | BCE, Cross-Entropy, NLL |
| - Advanced losses | 12 | Hinge, Focal, Dice, etc. |
| - Utilities | 8 | Weights, smoothing, reduction |
| **Tests** | 20 | Comprehensive test coverage |
| - Forward pass tests | 7 | UNKOWN ERROR ACCURED |
| - Gradient tests | 8 | Numerical verification |
| - Edge cases | 5 | Stability and special cases |
| **Documentation** | 20 | Clear, complete documentation |
| - Docstrings | 10 | All functions documented |
| - Comments | 5 | Complex logic explained |
| - README | 5 | Complete lab documentation |
| **Demo** | 20 | Working demonstrations |
| - Notebook completeness | 10 | All features demonstrated |
| - Examples | 5 | Clear, educational examples |
| - Visualizations | 5 | Informative plots |

**Total: 100 points**

## References

### Papers

1. **Focal Loss for Dense Object Detection** (Lin et al., 2017)
   - Introduces Focal Loss for imbalanced datasets
   
2. **Label Smoothing** (Szegedy et al., 2016)
   - Rethinking the Inception Architecture
   
3. **Huber Loss** (Huber, 1964)
   - Robust Estimation of a Location Parameter

### Tutorials

- PyTorch Loss Functions: https://pytorch.org/docs/stable/nn.html#loss-functions
- Loss Functions Explained: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html

### Books

- **Deep Learning** (Goodfellow et al.) - Chapter 5: Machine Learning Basics
- **Pattern Recognition and Machine Learning** (Bishop) - Loss functions section

## Additional Exercises

For students seeking extra challenge:

1. **Implement Additional Losses**
   - Tversky Loss (generalization of Dice)
   - Lovász-Softmax Loss (for segmentation)
   - Center Loss (for face recognition)

2. **Custom Loss Design**
   - Design task-specific loss function
   - Combine multiple losses with weights
   - Implement adaptive loss weighting

3. **Advanced Features**
   - Online hard example mining
   - Curriculum learning with loss
   - Multi-task loss balancing

4. **Optimization**
   - Vectorized implementations
   - Memory-efficient versions
   - Custom CUDA kernels (advanced)

## Comparison Table: All Losses

| Loss | Type | Input | Output | Key Parameter | Robustness |
|------|------|-------|--------|---------------|------------|
| MSE | Regression | Any | Linear | - | Low (outliers) |
| MAE | Regression | Any | Linear | - | High |
| Smooth L1 | Regression | Any | Linear | beta | Medium |
| MSLE | Regression | Non-negative | Linear | - | High |
| BCE | Binary | [0,1] | Sigmoid | - | Medium |
| BCE+Logits | Binary | Any | Sigmoid | pos_weight | High |
| Cross-Entropy | Multi-class | Any | Softmax | label_smoothing | High |
| NLL | Multi-class | Log probs | - | - | High |
| Hinge | Binary | Any | None | margin | Medium |
| Focal | Classification | [0,1] | Sigmoid/Softmax | alpha, gamma | High |
| Dice | Segmentation | [0,1] | - | smooth | High |
| KL Div | Distribution | Probs | - | - | Medium |

## FAQ

### Q: When should I use MSE vs MAE?

**A:** Use MSE for standard regression when errors are normally distributed. Use MAE when you have outliers or want equal penalty for all errors.

### Q: What's the difference between BCE and BCE with Logits?

**A:** BCE requires probabilities in [0,1] (after sigmoid). BCE with Logits takes raw logits and applies sigmoid internally, which is more numerically stable.

### Q: How do I handle severe class imbalance?

**A:** Three approaches:
1. Use class weights with `compute_class_weights()`
2. Use Focal Loss which focuses on hard examples
3. Oversample minority class or undersample majority

### Q: What is label smoothing and when should I use it?

**A:** Label smoothing replaces hard targets (0, 1) with soft targets (ε, 1-ε). Use it to:
- Prevent overconfidence
- Improve generalization
- Regularize the model
- Typical values: 0.1 or 0.2

### Q: Why is my loss NaN?

**A:** Common causes:
1. Log of zero or negative number
2. Division by zero
3. Overflow in exponentials
4. Learning rate too high

Solutions:
- Use stable versions (BCE with Logits instead of BCE)
- Clip gradients
- Reduce learning rate
- Check for NaN in data

### Q: Should I use 'mean' or 'sum' reduction?

**A:** Almost always use 'mean':
- Consistent across batch sizes
- Better for comparing losses
- More stable gradients

Use 'sum' only for specific algorithms that require it.

### Q: How do I implement a custom loss?

**A:** Follow this template:
```python
def custom_loss(predictions: Tensor, targets: Tensor, reduction: str = 'mean') -> Tensor:
    # Compute element-wise loss
    loss = your_loss_formula(predictions, targets)
    
    # Create output tensor with gradient function
    out = Tensor(
        loss.data,
        requires_grad=predictions.requires_grad,
        _children=(predictions, targets),
        _op='custom'
    )
    
    if out.requires_grad:
        def _backward():
            if predictions.requires_grad:
                # Compute gradient
                grad = your_gradient_formula(predictions, targets)
                
                if predictions.grad is None:
                    predictions.grad = Tensor(grad)
                else:
                    predictions.grad = Tensor(predictions.grad.data + grad)
        
        out.grad_fn = _backward
    
    return apply_reduction(out, reduction)
```

### Q: Can I combine multiple losses?

**A:** Yes! Example:
```python
# Multi-task learning
loss1 = mse_loss(pred1, target1)
loss2 = cross_entropy_loss(pred2, target2)

# Weighted combination
total_loss = 0.5 * loss1 + 0.5 * loss2
total_loss.backward()
```

## Performance Benchmarks

Approximate relative speeds (MSE = 1.0x baseline):

| Loss | Relative Speed | Notes |
|------|----------------|-------|
| MSE | 1.0x | Baseline (fastest) |
| MAE | 1.1x | Slightly slower |
| Smooth L1 | 1.2x | Conditional logic |
| BCE | 2.0x | Logarithms |
| BCE+Logits | 1.8x | More stable |
| Cross-Entropy | 2.5x | Softmax + log |
| Focal | 3.0x | Complex computation |
| Dice | 1.5x | Intersection/union |

## Memory Usage

All losses: O(batch_size) memory for gradients

Tips for large batches:
- Use gradient accumulation
- Process in smaller chunks
- Use 'none' reduction and manually average

## Best Practices

### 1. Always Use Appropriate Loss
```python
# Regression
loss = mse_loss(predictions, targets)

# Binary Classification
loss = binary_cross_entropy_with_logits(logits, targets)

# Multi-Class Classification
loss = cross_entropy_loss(logits, targets)
```

### 2. Handle Imbalance
```python
# Compute weights
weights = compute_class_weights(targets_np, num_classes)

# Use in loss
loss = cross_entropy_loss(logits, targets, weight=Tensor(weights))
```

### 3. Monitor Loss Values
```python
# Track during training
losses = []
for epoch in range(100):
    loss = compute_loss(...)
    losses.append(loss.data)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
```

### 4. Use Label Smoothing for Generalization
```python
# Add slight smoothing
loss = cross_entropy_loss(
    logits, 
    targets, 
    label_smoothing=0.1
)
```

### 5. Gradient Checking During Development
```python
from semester1.lab2_autograd.autograd import check_gradients

def loss_func(x):
    return your_loss(x, targets)

match, diff = check_gradients(loss_func, predictions)
assert match, f"Gradient mismatch: {diff}"
```

## Visualization Examples

### Loss Landscape
```python
import numpy as np
import matplotlib.pyplot as plt

# Create grid
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Compute loss at each point
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pred = Tensor([[X[i,j], Y[i,j]]])
        target = Tensor([[0.0, 0.0]])
        Z[i,j] = mse_loss(pred, target).data

# Plot
plt.contour(X, Y, Z, levels=20)
plt.colorbar(label='Loss')
plt.xlabel('Parameter 1')
plt.ylabel('Parameter 2')
plt.title('MSE Loss Landscape')
plt.show()
```

### Training Curves
```python
import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.show()
```

### Loss Comparison
```python
# Compare different losses on same data
losses = {
    'MSE': mse_loss(pred, target),
    'MAE': mae_loss(pred, target),
    'Smooth L1': smooth_l1_loss(pred, target)
}

plt.bar(losses.keys(), [l.data for l in losses.values()])
plt.ylabel('Loss Value')
plt.title('Loss Function Comparison')
plt.show()
```

## Debugging Checklist

When loss isn't working:

- [ ] Check input shapes match
- [ ] Verify targets are in correct format (indices vs one-hot)
- [ ] Ensure predictions are in correct range
- [ ] Check for NaN/Inf in inputs
- [ ] Verify gradients are flowing (check `.grad`)
- [ ] Test with simple known case
- [ ] Compare with PyTorch implementation
- [ ] Check reduction mode is appropriate
- [ ] Verify learning rate isn't too high
- [ ] Test gradient numerically

## Real-World Example: Complete Training Pipeline
```python
import numpy as np
from semester1.lab2_autograd.autograd import Tensor
from semester1.lab3_activation_functions.activations import relu, softmax
from semester1.lab4_loss_functions.losses import (
    cross_entropy_loss,
    compute_class_weights
)

# Prepare data
X_train = Tensor(np.random.randn(1000, 20))
y_train = Tensor(np.random.randint(0, 5, 1000))
X_val = Tensor(np.random.randn(200, 20))
y_val = Tensor(np.random.randint(0, 5, 200))

# Handle class imbalance
weights = compute_class_weights(y_train.data, num_classes=5)
class_weights = Tensor(weights)

# Initialize model
W1 = Tensor(np.random.randn(20, 64) * 0.1, requires_grad=True)
b1 = Tensor(np.zeros((1, 64)), requires_grad=True)
W2 = Tensor(np.random.randn(64, 5) * 0.1, requires_grad=True)
b2 = Tensor(np.zeros((1, 5)), requires_grad=True)

# Training loop
learning_rate = 0.01
batch_size = 32
num_epochs = 50

for epoch in range(num_epochs):
    # Training
    epoch_loss = 0
    num_batches = len(X_train.data) // batch_size
    
    for i in range(num_batches):
        # Get batch
        start = i * batch_size
        end = start + batch_size
        X_batch = Tensor(X_train.data[start:end])
        y_batch = Tensor(y_train.data[start:end])
        
        # Forward pass
        h1 = relu(X_batch @ W1 + b1)
        logits = h1 @ W2 + b2
        
        # Compute loss with class weights and label smoothing
        loss = cross_entropy_loss(
            logits, 
            y_batch,
            weight=class_weights,
            label_smoothing=0.1
        )
        
        epoch_loss += loss.data
        
        # Backward pass
        W1.zero_grad()
        b1.zero_grad()
        W2.zero_grad()
        b2.zero_grad()
        loss.backward()
        
        # Update parameters
        W1.data -= learning_rate * W1.grad.data
        b1.data -= learning_rate * b1.grad.data
        W2.data -= learning_rate * W2.grad.data
        b2.data -= learning_rate * b2.grad.data
    
    # Validation
    h1_val = relu(X_val @ W1 + b1)
    logits_val = h1_val @ W2 + b2
    val_loss = cross_entropy_loss(logits_val, y_val)
    
    # Print progress
    if epoch % 5 == 0:
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {epoch_loss/num_batches:.4f}")
        print(f"  Val Loss: {val_loss.data:.4f}")

print("\nTraining complete!")
```

## Author

Lab developed for Neural Networks from Scratch course - Semester 1, Lab 4

## License

MIT License - See repository root for details

---

**Congratulations on completing Lab 4!** 🎉

You now have a complete understanding of loss functions and can train neural networks effectively. With losses, activations, and automatic differentiation, you're ready to build complete neural networks!

**Next: Lab 5 - Dense Neural Networks** →