import numpy as np
from typing import Union, Tuple, List, Optional


class Tensor:

    def __init__(self, data: Union[list, tuple, np.ndarray, float, int, 'Tensor']):
        if isinstance(data, Tensor):
            self.data = data.data.copy()
        else:
            arr = np.array(data)
            self.data = arr.astype(np.float32) if arr.dtype != bool else arr

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

    def clone(self) -> "Tensor":
        return Tensor(self.data.copy())

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype})"

    def __str__(self) -> str:
        return str(self.data)

from .arithmetic import add, radd, sub, rsub, mul, rmul, div, rdiv, pow, neg

Tensor.__add__      = lambda self, other: add(self, other)
Tensor.__radd__     = lambda self, other: radd(self, other)
Tensor.__sub__      = lambda self, other: sub(self, other)
Tensor.__rsub__     = lambda self, other: rsub(self, other)
Tensor.__mul__      = lambda self, other: mul(self, other)
Tensor.__rmul__     = lambda self, other: rmul(self, other)
Tensor.__truediv__  = lambda self, other: div(self, other)
Tensor.__rtruediv__ = lambda self, other: rdiv(self, other)
Tensor.__pow__      = lambda self, power: pow(self, power)
Tensor.__neg__      = lambda self: neg(self)

from .compare import eq, ne, lt, le, gt, ge

Tensor.__eq__ = lambda self, other: eq(self, other)
Tensor.__ne__ = lambda self, other: ne(self, other)
Tensor.__lt__ = lambda self, other: lt(self, other)
Tensor.__le__ = lambda self, other: le(self, other)
Tensor.__gt__ = lambda self, other: gt(self, other)
Tensor.__ge__ = lambda self, other: ge(self, other)

from .indexing import getitem, setitem

Tensor.__getitem__ = lambda self, key: getitem(self, key)
Tensor.__setitem__ = lambda self, key, value: setitem(self, key, value)

from .shape import reshape, flatten, transpose, squeeze, unsqueeze, get_T

Tensor.reshape   = lambda self, *shape: reshape(self, *shape)
Tensor.flatten   = lambda self: flatten(self)
Tensor.transpose = lambda self, *axes: transpose(self, *axes)
Tensor.squeeze   = lambda self, axis=None: squeeze(self, axis)
Tensor.unsqueeze = lambda self, dim: unsqueeze(self, dim)
Tensor.T         = property(lambda self: get_T(self))

from .reduce import sum, mean, max, min, std, var

Tensor.sum  = lambda self, axis=None, keepdims=False: sum(self, axis, keepdims)
Tensor.mean = lambda self, axis=None, keepdims=False: mean(self, axis, keepdims)
Tensor.max  = lambda self, axis=None, keepdims=False: max(self, axis, keepdims)
Tensor.min  = lambda self, axis=None, keepdims=False: min(self, axis, keepdims)
Tensor.std  = lambda self, axis=None, keepdims=False: std(self, axis, keepdims)
Tensor.var  = lambda self, axis=None, keepdims=False: var(self, axis, keepdims)

from .linalg import matmul, dot

Tensor.matmul     = lambda self, other: matmul(self, other)
Tensor.__matmul__ = lambda self, other: matmul(self, other)
Tensor.dot        = lambda self, other: dot(self, other)

from .factory import zeros_like, ones_like, rand_like, randn_like, concatenate, stack, split

Tensor.zeros_like  = classmethod(lambda cls, tensor: zeros_like(tensor))
Tensor.ones_like   = classmethod(lambda cls, tensor: ones_like(tensor))
Tensor.rand_like   = classmethod(lambda cls, tensor: rand_like(tensor))
Tensor.randn_like  = classmethod(lambda cls, tensor: randn_like(tensor))
Tensor.concatenate = classmethod(lambda cls, tensors, axis=0: concatenate(tensors, axis))
Tensor.stack       = classmethod(lambda cls, tensors, axis=0: stack(tensors, axis))
Tensor.split       = lambda self, sections, axis=0: split(self, sections, axis)
