import numpy as np
from typing import Optional, List, Tuple, Callable, Set, Union
from semester1.lab1_tensor_operations.tensor import Tensor as BaseTensor


class Tensor(BaseTensor):
    _grad_enabled = True
    
    def __init__(
        self,
        data: Union[list, tuple, np.ndarray, float, int],
        requires_grad: bool = False,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
    ):
        super().__init__(data)
        
        self.requires_grad = requires_grad and self._grad_enabled
        self.grad: Optional["Tensor"] = None
        self.grad_fn: Optional[Callable] = None
        self._prev: Tuple["Tensor", ...] = _children
        self._op = _op
        
    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        if not self.requires_grad:
            raise RuntimeError(
                "Cannot call backward() on tensor that doesn't require gradients. "
                "Create tensor with requires_grad=True."
            )
        
        if gradient is None:
            if self.data.size == 1:
                gradient = Tensor(np.ones_like(self.data))
            else:
                raise RuntimeError(
                    "Gradient argument must be specified for non-scalar output. "
                    f"Expected gradient for tensor of shape {self.shape}. "
                    "Consider calling .sum() or .mean() first to get a scalar."
                )
        
        topo_order: List["Tensor"] = []
        visited: Set[int] = set()
        
        def build_topo(tensor: "Tensor") -> None:
            tensor_id = id(tensor)
            if tensor_id not in visited:
                visited.add(tensor_id)
                for parent in tensor._prev:
                    build_topo(parent)
                topo_order.append(tensor)
        
        build_topo(self)
        
        self.grad = gradient
        
        for tensor in reversed(topo_order):
            if tensor.grad_fn is not None:
                tensor.grad_fn()
    
    def zero_grad(self) -> None:
        self.grad = None
    
    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)
    
    def __repr__(self) -> str:
        base_repr = f"Tensor({self.data}"
        if self.requires_grad and self._op:
            base_repr += f", grad_fn=<{self._op}>"
        base_repr += ")"
        return base_repr


from .arithmetic import (
    add, radd, sub, rsub, mul, rmul, div, rdiv, pow, neg
)

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

from .aggregation import sum_op, mean, max_op

Tensor.sum  = lambda self, axis=None, keepdims=False: sum_op(self, axis, keepdims)
Tensor.mean = lambda self, axis=None, keepdims=False: mean(self, axis, keepdims)
Tensor.max  = lambda self, axis=None, keepdims=False: max_op(self, axis, keepdims)

from .shape import reshape, transpose, get_T

Tensor.reshape   = lambda self, *shape: reshape(self, *shape)
Tensor.transpose = lambda self, *axes: transpose(self, *axes)
Tensor.T         = property(lambda self: get_T(self))

from .matrix import matmul

Tensor.matmul     = lambda self, other: matmul(self, other)
Tensor.__matmul__ = lambda self, other: matmul(self, other)