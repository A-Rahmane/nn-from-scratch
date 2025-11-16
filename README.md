# Neural Networks from Scratch

A comprehensive 6-semester journey through building neural networks from scratch in Python.

## Overview
This repository contains implementations of neural network components built entirely from scratch, using only NumPy for numerical operations. No high-level ML frameworks (PyTorch, TensorFlow, etc.) are used.

**Latest Update**: Lab 1 has been refactored to a modular architecture for improved maintainability and code quality.

## Structure
- `semester1/` - Foundations & Basic Neural Networks
  - ✅ **Lab 1**: Tensor Operations (Modular Architecture - COMPLETED)
  - ⚪ Lab 2: Autograd
  - ⚪ Lab 3-6: Coming soon
- `semester2/` - Deep Neural Networks & Optimization
- `semester3/` - Convolutional Neural Networks
- `semester4/` - Advanced CNNs & Training
- `semester5/` - Recurrent Neural Networks
- `semester6/` - Attention & Transformers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/A-Rahmane/nn-from-scratch.git
cd nn-from-scratch
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

## Running Tests

### All Tests
```bash
pytest semester1/lab1_tensor_operations/tests/ -v
```

### Quick Verification
```bash
# Test modular implementation
python semester1/lab1_tensor_operations/test_modular.py

# Test core module
python semester1/lab1_tensor_operations/tensor/core.py
```

## Features

### Modular Architecture (Lab 1)
The Tensor class now uses a modular design:
- **8 focused modules** instead of 1 monolithic file
- **Zero performance overhead** (operations bound at import time)
- **100% API compatibility** with original implementation
- **Easier to maintain** and extend
```python
from semester1.lab1_tensor_operations.tensor import Tensor

# All original functionality preserved
t = Tensor([1, 2, 3])
result = t + 5  # Works exactly as before
```

## Lab Completion Checklist
- [x] Semester 1: Lab 1 (Modular Refactoring Complete)
- [ ] Semester 1: Labs 2-6
- [ ] Semester 2: Labs 1-6
- [ ] Semester 3: Labs 1-6
- [ ] Semester 4: Labs 1-6
- [ ] Semester 5: Labs 1-6
- [ ] Semester 6: Labs 1-6

## Recent Updates

### v1.0.0 - Modular Architecture
- ✅ Refactored Tensor class into 8 focused modules
- ✅ Maintained 100% backward compatibility
- ✅ All 85 tests passing
- ✅ Added comprehensive documentation
- ✅ Zero performance impact

## Contributing
When contributing new features:
1. Follow the modular architecture pattern
2. Add appropriate tests
3. Update documentation
4. Ensure all existing tests pass

## License
MIT License - See LICENSE file for details

## Author
MENOUER Abderrahmane