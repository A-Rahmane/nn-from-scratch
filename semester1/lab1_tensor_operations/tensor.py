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
