from .core import Tensor
import numpy as np


def matmul(a: Tensor, b: Tensor) -> Tensor:
    if not isinstance(b, Tensor):
        b = Tensor(b)
    
    try:
        result = np.matmul(a.data, b.data)
        return Tensor(result)
    except ValueError as e:
        raise ValueError(
            f"Incompatible shapes for matmul: {a.shape} and {b.shape}. "
            f"Inner dimensions must match for matrix multiplication."
        ) from e


def dot(a: Tensor, b: Tensor) -> Tensor:
    if not isinstance(b, Tensor):
        b = Tensor(b)
    
    return Tensor(np.dot(a.data, b.data))