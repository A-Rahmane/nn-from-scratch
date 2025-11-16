import numpy as np
from typing import Union, Tuple

class Tensor:
    """Core tensor class for data storage and basic properties."""

    def __init__(self, data: Union[list, tuple, np.ndarray, float, int]):
        if isinstance(data, Tensor):
            self.data = data.data.copy()
        else:
            arr = np.array(data)
            self.data = arr.astype(np.float32) if arr.dtype != bool else arr

    # ==================== Properties ====================

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype
    
    # ==================== Advanced Operations ====================

    def clone(self) -> "Tensor":
        return Tensor(self.data.copy())

    # ==================== String Representation ====================

    def __repr__(self):
        return f"Tensor({self.data})"

    def __str__(self):
        return str(self.data)
    
# ==================== Arithmetic Operations ====================

from .arithmetic import add, sub, mul, div, pow, neg

Tensor.__add__     = lambda self, other: add(self, other)
Tensor.__sub__     = lambda self, other: sub(self, other)
Tensor.__mul__     = lambda self, other: mul(self, other)
Tensor.__truediv__ = lambda self, other: div(self, other)
Tensor.__pow__     = lambda self, power: pow(self, power)
Tensor.__neg__     = lambda self       : neg(self       )

# TODO : implement remaining arithmetic ops
# __radd__, __rsub__, __rmul__, __rtruediv__

# ==================== Comparison Operations ====================

from .compare import eq, ne, lt, le, gt, ge

Tensor.__eq__ = lambda self, other: eq(self, other)
Tensor.__ne__ = lambda self, other: ne(self, other)
Tensor.__lt__ = lambda self, other: lt(self, other)
Tensor.__le__ = lambda self, other: le(self, other)
Tensor.__gt__ = lambda self, other: gt(self, other)
Tensor.__ge__ = lambda self, other: ge(self, other)

# ==================== Shape Manipulation ====================

from .shape import reshape, flatten, transpose, squeeze, unsqueeze

Tensor.reshape   = lambda self, shape: reshape  (self, shape)
Tensor.flatten   = lambda self       : flatten  (self       )
Tensor.transpose = lambda self, axes : transpose(self, axes )
Tensor.squeeze   = lambda self, axis : squeeze  (self, axis )
Tensor.unsqueeze = lambda self, dim  : unsqueeze(self, dim  )

# ==================== Aggregation Operations ====================

from .reduce import sum, mean, max, min

Tensor.sum  = lambda self, axis: sum (self, axis)
Tensor.mean = lambda self, axis: mean(self, axis)
Tensor.max  = lambda self, axis: max (self, axis)
Tensor.mix  = lambda self, axis: min (self, axis)

# ==================== Matrix Operations ====================

from .linalg import matmul, dot

Tensor.mat