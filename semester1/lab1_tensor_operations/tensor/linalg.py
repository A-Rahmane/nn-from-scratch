from .core import Tensor
import numpy as np

def matmul(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(np.matmul(a.data, b.data))

def dot(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(np.dot(a.data, b.data))
