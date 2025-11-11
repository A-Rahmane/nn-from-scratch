# **Lab 0: Initialization & Project Setup**

## **Objective**
Set up a professional development environment and project structure that will be used throughout all 6 semesters. Learn version control best practices and establish coding conventions for building neural networks from scratch.

---

## **Part 1: GitHub Repository Setup**

### Step 1: Create Repository
1. Go to [GitHub](https://github.com) and sign in
2. Click "New Repository"
3. Repository settings:
   - **Name**: `nn-from-scratch`
   - **Description**: "Building Neural Networks from Scratch - 6 Semester Lab Series"
   - **Visibility**: Public (recommended) or Private
   - **Initialize with**: 
     - ✅ Add a README file
     - ✅ Add .gitignore (select Python template)
     - ✅ Choose a license (MIT recommended)
4. Click "Create repository"

### Step 2: Clone Repository Locally
```bash
git clone https://github.com/YOUR_USERNAME/nn-from-scratch.git
cd nn-from-scratch
```

---

## **Part 2: Project Structure**

Create the following directory structure:

```
nn-from-scratch/
│
├── README.md
├── .gitignore
├── LICENSE
├── requirements.txt
├── setup.py
│
├── semester1/
│   ├── __init__.py
│   ├── lab1_tensor_operations/
│   │   ├── __init__.py
│   │   ├── tensor.py
│   │   ├── tests/
│   │   │   ├── __init__.py
│   │   │   └── test_tensor.py
│   │   ├── notebooks/
│   │   │   └── demo.ipynb
│   │   └── README.md
│   │
│   ├── lab2_autograd/
│   │   ├── __init__.py
│   │   ├── autograd.py
│   │   ├── tests/
│   │   │   ├── __init__.py
│   │   │   └── test_autograd.py
│   │   ├── notebooks/
│   │   │   └── demo.ipynb
│   │   └── README.md
│   │
│   └── [lab3-6 following same structure]
│
├── semester2/
│   └── [same structure as semester1]
│
├── semester3/
│   └── [same structure as semester1]
│
├── semester4/
│   └── [same structure as semester1]
│
├── semester5/
│   └── [same structure as semester1]
│
├── semester6/
│   └── [same structure as semester1]
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── visualization.py
│   └── metrics.py
│
├── docs/
│   ├── coding_conventions.md
│   ├── lab_template.md
│   └── resources.md
│
└── tests/
    └── run_all_tests.py
```

### Step 3: Create Directory Structure

Run these commands in your repository root:

```bash
# Create semester directories
for i in {1..6}; do
    mkdir -p "semester$i"
    touch "semester$i/__init__.py"
done

# Create lab directories for semester 1 (example)
for i in {1..6}; do
    mkdir -p "semester1/lab${i}_placeholder"
    touch "semester1/lab${i}_placeholder/__init__.py"
    touch "semester1/lab${i}_placeholder/README.md"
    mkdir -p "semester1/lab${i}_placeholder/tests"
    touch "semester1/lab${i}_placeholder/tests/__init__.py"
    mkdir -p "semester1/lab${i}_placeholder/notebooks"
done

# Create utility directories
mkdir -p utils docs tests
touch utils/__init__.py
touch tests/run_all_tests.py
```

---

## **Part 3: Essential Files**

### Create `requirements.txt`
```txt
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
pytest>=7.0.0
black>=22.0.0
flake8>=4.0.0
```

### Create `setup.py`
```python
from setuptools import setup, find_packages

setup(
    name="nn-from-scratch",
    version="0.1.0",
    description="Neural Networks implemented from scratch",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    python_requires=">=3.8",
)
```

### Update `README.md`
```markdown
# Neural Networks from Scratch

A comprehensive 6-semester journey through building neural networks from scratch in Python.

## Overview
This repository contains implementations of neural network components built entirely from scratch, using only NumPy for numerical operations. No high-level ML frameworks (PyTorch, TensorFlow, etc.) are used.

## Structure
- `semester1/` - Foundations & Basic Neural Networks
- `semester2/` - Deep Neural Networks & Optimization
- `semester3/` - Convolutional Neural Networks
- `semester4/` - Advanced CNNs & Training
- `semester5/` - Recurrent Neural Networks
- `semester6/` - Attention & Transformers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/nn-from-scratch.git
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
```bash
pytest semester1/lab1_tensor_operations/tests/
```

## Lab Completion Checklist
- [ ] Semester 1: Labs 1-6
- [ ] Semester 2: Labs 1-6
- [ ] Semester 3: Labs 1-6
- [ ] Semester 4: Labs 1-6
- [ ] Semester 5: Labs 1-6
- [ ] Semester 6: Labs 1-6

## License
MIT License
```

### Create `docs/coding_conventions.md`
```markdown
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
```

### Create `docs/lab_template.md`
```markdown
# Lab X: [Lab Title]

## Objectives
- Objective 1
- Objective 2
- Objective 3

## Background
Brief theoretical background needed for this lab.

## Implementation Tasks

### Task 1: [Task Name]
**Description**: What to implement

**Requirements**:
- Requirement 1
- Requirement 2

**Hints**:
- Helpful hint 1
- Helpful hint 2

### Task 2: [Task Name]
[Similar structure]

## Testing Requirements
- Test case 1
- Test case 2
- Test case 3

## Submission Checklist
- [ ] All functions implemented
- [ ] All tests passing
- [ ] Code documented with docstrings
- [ ] Demo notebook completed
- [ ] README.md updated

## Grading Rubric
- Implementation (40%)
- Tests (20%)
- Documentation (20%)
- Demo & Results (20%)

## Resources
- Link to relevant papers
- Link to tutorials
- Link to documentation
```

---

## **Part 4: Development Environment Setup**

### Step 1: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Step 3: Configure Git
```bash
# Set your identity
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Optional: Set default editor
git config --global core.editor "code --wait"  # For VS Code
```

---

## **Part 5: First Commit**

### Step 1: Stage All Files
```bash
git add .
```

### Step 2: Commit
```bash
git commit -m "feat(setup): initialize project structure for 6-semester lab series"
```

### Step 3: Push to GitHub
```bash
git push origin main
```

---

## **Part 6: Verify Setup**

### Create a Test File
Create `semester1/lab1_tensor_operations/tensor.py`:
```python
"""Tensor operations module."""
import numpy as np


class Tensor:
    """Basic tensor class for neural network operations."""
    
    def __init__(self, data):
        """Initialize tensor with data.
        
        Args:
            data: Input data (list, tuple, or numpy array)
        """
        self.data = np.array(data, dtype=np.float32)
    
    def __repr__(self):
        return f"Tensor({self.data})"


# Test if import works
if __name__ == "__main__":
    t = Tensor([1, 2, 3])
    print(t)
```

### Run Test
```bash
python semester1/lab1_tensor_operations/tensor.py
```

Expected output: `Tensor([1. 2. 3.])`

---

## **Part 7: Lab README Template**

For each lab, create a README.md with this structure:

```markdown
# Lab X: [Lab Title]

## Status
🟢 Completed | 🟡 In Progress | ⚪ Not Started

## Description
Brief description of what this lab implements.

## Files
- `module_name.py` - Main implementation
- `tests/test_module_name.py` - Unit tests
- `notebooks/demo.ipynb` - Demonstration notebook

## How to Run

### Tests
```bash
pytest semester1/labX_name/tests/ -v
```

### Demo
```bash
jupyter notebook semester1/labX_name/notebooks/demo.ipynb
```

## Implementation Details
Key implementation decisions and approaches.

## Results
Summary of results, performance metrics, or visualizations.

## References
- Link 1
- Link 2
```

---

## **Part 8: Workflow for Each Lab**

### Starting a New Lab
```bash
# Create branch
git checkout -b feature/semester1-lab1-tensor-operations

# Create lab structure
mkdir -p semester1/lab1_tensor_operations/{tests,notebooks}
touch semester1/lab1_tensor_operations/__init__.py
touch semester1/lab1_tensor_operations/tensor.py
touch semester1/lab1_tensor_operations/tests/test_tensor.py
touch semester1/lab1_tensor_operations/README.md
```

### During Development
```bash
# Regular commits
git add semester1/lab1_tensor_operations/
git commit -m "feat(tensor): implement addition operation"

# Run tests frequently
pytest semester1/lab1_tensor_operations/tests/ -v

# Format code
black semester1/lab1_tensor_operations/
```

### Completing a Lab
```bash
# Final commit
git add semester1/lab1_tensor_operations/
git commit -m "feat(tensor): complete lab 1 - tensor operations"

# Push to GitHub
git push origin feature/semester1-lab1-tensor-operations

# Merge to main (or create pull request on GitHub)
git checkout main
git merge feature/semester1-lab1-tensor-operations
git push origin main
```

---

## **Deliverables**

Submit a document containing:

1. **GitHub Repository Link**: URL to your public/private repository
2. **Screenshot**: Repository structure showing all directories
3. **Verification**: Screenshot of successful test run
4. **Reflection** (2-3 paragraphs):
   - What you learned about project organization
   - Any challenges faced during setup
   - How this structure will help in future labs

---

## **Grading Criteria**

| Criteria | Points |
|----------|--------|
| Repository created and properly initialized | 15 |
| Correct directory structure | 25 |
| All required files present (requirements.txt, setup.py, etc.) | 20 |
| Documentation files created | 15 |
| Virtual environment setup and working | 10 |
| First commit pushed to GitHub | 10 |
| Verification test runs successfully | 5 |

**Total: 100 points**

---

## **Additional Resources**

- [GitHub Guides](https://guides.github.com/)
- [PEP 8 Style Guide](https://pep8.org/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
