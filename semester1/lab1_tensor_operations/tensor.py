"""Tensor operations module."""
import numpy as np
from typing import Union, Tuple, List, Optional


class Tensor:
    """Basic tensor class for neural network operations.
    
    This class wraps NumPy arrays and provides common tensor operations
    needed for building neural networks from scratch.
    
    Args:
        data: Input data (list, tuple, numpy array, or scalar)
    
    Attributes:
        data: The underlying NumPy array storing tensor values
    """
    
    def __init__(self, data: Union[list, tuple, np.ndarray, float, int]):
        """Initialize tensor with data.
        
        Args:
            data: Input data that will be converted to float32 NumPy array
        """
        if isinstance(data, Tensor):
            self.data = data.data.copy()
        else:
            arr = np.array(data)
            if arr.dtype == bool:
                self.data = arr
            else:
                self.data = arr.astype(np.float32)
    
    # ==================== Properties ====================
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the tensor."""
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self.data.ndim
    
    @property
    def size(self) -> int:
        """Return the total number of elements."""
        return self.data.size
    
    @property
    def dtype(self):
        """Return the data type of the tensor."""
        return self.data.dtype
    
    # ==================== String Representation ====================
    
    def __repr__(self) -> str:
        """Return detailed string representation."""
        return f"Tensor({self.data})"
    
    def __str__(self) -> str:
        """Return user-friendly string representation."""
        return str(self.data)
    
    # ==================== Arithmetic Operations ====================
    
    def __add__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Add two tensors or tensor and scalar."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data + other.data)
    
    def __radd__(self, other: Union[np.ndarray, float, int]) -> 'Tensor':
        """Right addition (for scalar + tensor)."""
        return self.__add__(other)
    
    def __sub__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Subtract two tensors or tensor and scalar."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data - other.data)
    
    def __rsub__(self, other: Union[np.ndarray, float, int]) -> 'Tensor':
        """Right subtraction (for scalar - tensor)."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(other.data - self.data)
    
    def __mul__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Element-wise multiplication."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data * other.data)
    
    def __rmul__(self, other: Union[np.ndarray, float, int]) -> 'Tensor':
        """Right multiplication (for scalar * tensor)."""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Element-wise division."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data / other.data)
    
    def __rtruediv__(self, other: Union[np.ndarray, float, int]) -> 'Tensor':
        """Right division (for scalar / tensor)."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(other.data / self.data)
    
    def __pow__(self, power: Union[float, int]) -> 'Tensor':
        """Raise tensor to a power."""
        return Tensor(self.data ** power)
    
    def __neg__(self) -> 'Tensor':
        """Negate tensor."""
        return Tensor(-self.data)
    
    # ==================== Comparison Operations ====================
    
    def __eq__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Element-wise equality comparison."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor((self.data == other.data).astype(bool))
    
    def __ne__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Element-wise inequality comparison."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor((self.data != other.data).astype(bool))
    
    def __lt__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Element-wise less than comparison."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor((self.data < other.data).astype(bool))
    
    def __le__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Element-wise less than or equal comparison."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor((self.data <= other.data).astype(bool))
    
    def __gt__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Element-wise greater than comparison."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor((self.data > other.data).astype(bool))
    
    def __ge__(self, other: Union['Tensor', np.ndarray, float, int]) -> 'Tensor':
        """Element-wise greater than or equal comparison."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor((self.data >= other.data).astype(bool))
    
    # ==================== Indexing and Slicing ====================
    
    def __getitem__(self, key) -> 'Tensor':
        """Get item or slice from tensor."""
        if isinstance(key, Tensor):
            key = key.data
        result = self.data[key]
        return Tensor(result)
    
    def __setitem__(self, key, value: Union['Tensor', np.ndarray, float, int]):
        """Set item or slice in tensor."""
        if isinstance(value, Tensor):
            self.data[key] = value.data
        else:
            self.data[key] = value
    
    # ==================== Shape Manipulation ====================
    
    def reshape(self, *shape: int) -> 'Tensor':
        """Reshape tensor to new shape.
        
        Args:
            *shape: New shape dimensions
        
        Returns:
            Reshaped tensor
        """
        # Handle both reshape(2, 3) and reshape((2, 3))
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return Tensor(self.data.reshape(shape))
    
    def transpose(self, *axes: int) -> 'Tensor':
        """Transpose tensor dimensions.
        
        Args:
            *axes: Permutation of dimensions (optional)
        
        Returns:
            Transposed tensor
        """
        if len(axes) == 0:
            return Tensor(self.data.T)
        return Tensor(np.transpose(self.data, axes))
    
    @property
    def T(self) -> 'Tensor':
        """Transpose property for 2D matrices."""
        return self.transpose()
    
    def flatten(self) -> 'Tensor':
        """Flatten tensor to 1D.
        
        Returns:
            Flattened tensor
        """
        return Tensor(self.data.flatten())
    
    def squeeze(self, axis: Optional[int] = None) -> 'Tensor':
        """Remove dimensions of size 1.
        
        Args:
            axis: Specific axis to squeeze (optional)
        
        Returns:
            Squeezed tensor
        """
        return Tensor(np.squeeze(self.data, axis=axis))
    
    def unsqueeze(self, dim: int) -> 'Tensor':
        """Add dimension of size 1 at specified position.
        
        Args:
            dim: Position to insert new dimension
        
        Returns:
            Tensor with added dimension
        """
        return Tensor(np.expand_dims(self.data, axis=dim))
    
    # ==================== Aggregation Operations ====================
    
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, 
            keepdims: bool = False) -> 'Tensor':
        """Sum of tensor elements.
        
        Args:
            axis: Axis or axes along which to sum
            keepdims: Whether to keep reduced dimensions
        
        Returns:
            Sum tensor
        """
        return Tensor(np.sum(self.data, axis=axis, keepdims=keepdims))
    
    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, 
             keepdims: bool = False) -> 'Tensor':
        """Mean of tensor elements.
        
        Args:
            axis: Axis or axes along which to compute mean
            keepdims: Whether to keep reduced dimensions
        
        Returns:
            Mean tensor
        """
        return Tensor(np.mean(self.data, axis=axis, keepdims=keepdims))
    
    def max(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, 
            keepdims: bool = False) -> 'Tensor':
        """Maximum of tensor elements.
        
        Args:
            axis: Axis or axes along which to find maximum
            keepdims: Whether to keep reduced dimensions
        
        Returns:
            Maximum tensor
        """
        return Tensor(np.max(self.data, axis=axis, keepdims=keepdims))
    
    def min(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, 
            keepdims: bool = False) -> 'Tensor':
        """Minimum of tensor elements.
        
        Args:
            axis: Axis or axes along which to find minimum
            keepdims: Whether to keep reduced dimensions
        
        Returns:
            Minimum tensor
        """
        return Tensor(np.min(self.data, axis=axis, keepdims=keepdims))
    
    def std(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, 
            keepdims: bool = False) -> 'Tensor':
        """Standard deviation of tensor elements.
        
        Args:
            axis: Axis or axes along which to compute std
            keepdims: Whether to keep reduced dimensions
        
        Returns:
            Standard deviation tensor
        """
        return Tensor(np.std(self.data, axis=axis, keepdims=keepdims))
    
    def var(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, 
            keepdims: bool = False) -> 'Tensor':
        """Variance of tensor elements.
        
        Args:
            axis: Axis or axes along which to compute variance
            keepdims: Whether to keep reduced dimensions
        
        Returns:
            Variance tensor
        """
        return Tensor(np.var(self.data, axis=axis, keepdims=keepdims))
    
    # ==================== Matrix Operations ====================
    
    def matmul(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication.
        
        Args:
            other: Tensor to multiply with
        
        Returns:
            Result of matrix multiplication
        
        Raises:
            ValueError: If shapes are incompatible
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        try:
            result = np.matmul(self.data, other.data)
            return Tensor(result)
        except ValueError as e:
            raise ValueError(f"Incompatible shapes for matmul: {self.shape} and {other.shape}") from e
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication operator @."""
        return self.matmul(other)
    
    def dot(self, other: 'Tensor') -> 'Tensor':
        """Dot product.
        
        Args:
            other: Tensor to compute dot product with
        
        Returns:
            Dot product result
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(np.dot(self.data, other.data))
    
    # ==================== Advanced Operations ====================
    
    def clone(self) -> 'Tensor':
        """Create a deep copy of the tensor.
        
        Returns:
            Cloned tensor
        """
        return Tensor(self.data.copy())
    
    @classmethod
    def zeros_like(cls, tensor: 'Tensor') -> 'Tensor':
        """Create tensor of zeros with same shape.
        
        Args:
            tensor: Reference tensor for shape
        
        Returns:
            Tensor of zeros
        """
        return cls(np.zeros_like(tensor.data))
    
    @classmethod
    def ones_like(cls, tensor: 'Tensor') -> 'Tensor':
        """Create tensor of ones with same shape.
        
        Args:
            tensor: Reference tensor for shape
        
        Returns:
            Tensor of ones
        """
        return cls(np.ones_like(tensor.data))
    
    @classmethod
    def rand_like(cls, tensor: 'Tensor') -> 'Tensor':
        """Create tensor of random values [0, 1) with same shape.
        
        Args:
            tensor: Reference tensor for shape
        
        Returns:
            Tensor of random values
        """
        return cls(np.random.rand(*tensor.shape))
    
    @classmethod
    def randn_like(cls, tensor: 'Tensor') -> 'Tensor':
        """Create tensor of random normal values with same shape.
        
        Args:
            tensor: Reference tensor for shape
        
        Returns:
            Tensor of random normal values
        """
        return cls(np.random.randn(*tensor.shape))
    
    @classmethod
    def concatenate(cls, tensors: List['Tensor'], axis: int = 0) -> 'Tensor':
        """Concatenate tensors along an axis.
        
        Args:
            tensors: List of tensors to concatenate
            axis: Axis along which to concatenate
        
        Returns:
            Concatenated tensor
        """
        arrays = [t.data for t in tensors]
        return cls(np.concatenate(arrays, axis=axis))
    
    @classmethod
    def stack(cls, tensors: List['Tensor'], axis: int = 0) -> 'Tensor':
        """Stack tensors along a new axis.
        
        Args:
            tensors: List of tensors to stack
            axis: Axis along which to stack
        
        Returns:
            Stacked tensor
        """
        arrays = [t.data for t in tensors]
        return cls(np.stack(arrays, axis=axis))
    
    def split(self, sections: Union[int, List[int]], axis: int = 0) -> List['Tensor']:
        """Split tensor into multiple tensors.
        
        Args:
            sections: Number of equal sections or list of sizes
            axis: Axis along which to split
        
        Returns:
            List of tensors
        """
        arrays = np.split(self.data, sections, axis=axis)
        return [Tensor(arr) for arr in arrays]


# Test basic functionality
if __name__ == "__main__":
    print("=== Testing Tensor Class ===\n")
    
    # Basic creation
    t1 = Tensor([1, 2, 3])
    print(f"t1 = {t1}")
    print(f"shape: {t1.shape}, ndim: {t1.ndim}, size: {t1.size}\n")
    
    # Arithmetic
    t2 = Tensor([4, 5, 6])
    print(f"t1 + t2 = {t1 + t2}")
    print(f"t1 * 2 = {t1 * 2}")
    print(f"t1 ** 2 = {t1 ** 2}\n")
    
    # Matrix operations
    m1 = Tensor([[1, 2], [3, 4]])
    m2 = Tensor([[5, 6], [7, 8]])
    print(f"m1 @ m2 =\n{m1 @ m2}\n")
    
    # Aggregations
    print(f"m1.sum() = {m1.sum()}")
    print(f"m1.mean(axis=0) = {m1.mean(axis=0)}\n")
    
    # Shape manipulation
    t3 = Tensor([1, 2, 3, 4, 5, 6])
    print(f"t3.reshape(2, 3) =\n{t3.reshape(2, 3)}")
    print(f"Transposed =\n{t3.reshape(2, 3).T}\n")
    
    print("All basic tests passed!")
