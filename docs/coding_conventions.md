# Coding Conventions

## General Guidelines
1. **Language**: Python 3.8+
2. **Style Guide**: Follow PEP 8
3. **Formatting**: Use Black formatter
4. **Linting**: Use Flake8

## Naming Conventions

### Classes
- Use PascalCase: `Tensor`, `DenseLayer`, `SGDOptimizer`
- Suffix descriptive names: `Conv2DLayer`, `MaxPooling2D`

### Functions and Methods
- Use snake_case: `forward()`, `backward()`, `compute_gradient()`
- Verbs for actions: `calculate_loss()`, `update_weights()`

### Variables
- Use snake_case: `learning_rate`, `batch_size`, `hidden_dims`
- Descriptive names: avoid single letters except for loops

### Constants
- Use UPPER_SNAKE_CASE: `DEFAULT_LEARNING_RATE`, `EPSILON`

## Code Structure

### Class Template
```python
import numpy as np
from typing import Optional, Tuple


class LayerName:
    """Brief description of the layer.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
    
    Attributes:
        attr1: Description of attribute 1
        attr2: Description of attribute 2
    """
    
    def __init__(self, param1: int, param2: float = 0.01):
        self.param1 = param1
        self.param2 = param2
        self._initialize_parameters()
    
    def _initialize_parameters(self) -> None:
        """Initialize layer parameters."""
        pass
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.
        
        Args:
            x: Input array of shape (batch_size, input_dim)
        
        Returns:
            Output array of shape (batch_size, output_dim)
        """
        pass
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass.
        
        Args:
            grad_output: Gradient from next layer
        
        Returns:
            Gradient with respect to input
        """
        pass
```

## Documentation

### Docstrings
- Use Google-style docstrings
- Document all public methods and classes
- Include type hints

### Comments
- Explain "why", not "what"
- Keep comments up-to-date with code
- Use inline comments sparingly

## Testing

### Test File Naming
- Prefix with `test_`: `test_tensor.py`, `test_autograd.py`

### Test Function Naming
- Prefix with `test_`: `test_forward_pass()`, `test_gradient_computation()`
- Be descriptive: `test_tensor_addition_broadcasting()`

### Test Structure
```python
import pytest
import numpy as np
from semester1.lab1_tensor_operations.tensor import Tensor


def test_tensor_addition():
    """Test basic tensor addition."""
    # Arrange
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    
    # Act
    result = a + b
    
    # Assert
    expected = Tensor([5, 7, 9])
    np.testing.assert_array_equal(result.data, expected.data)
```

## Version Control

### Commit Messages
Format: `<type>(<scope>): <subject>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding tests
- `refactor`: Code refactoring
- `style`: Formatting changes

Examples:
```
feat(tensor): implement broadcasting for addition
fix(autograd): correct gradient computation for multiplication
docs(readme): add installation instructions
test(dense): add tests for weight initialization
```

### Branch Naming
- Feature branches: `feature/lab1-tensor-operations`
- Bug fixes: `fix/autograd-gradient-bug`
- Experiments: `experiment/alternative-initialization`

## Code Quality

### Before Committing
```bash
# Format code
black semester1/lab1_tensor_operations/

# Check linting
flake8 semester1/lab1_tensor_operations/

# Run tests
pytest semester1/lab1_tensor_operations/tests/
```

## Imports
Order imports as follows:
1. Standard library imports
2. Third-party imports (numpy, matplotlib)
3. Local application imports
```python
import math
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from semester1.lab1_tensor_operations.tensor import Tensor
```
