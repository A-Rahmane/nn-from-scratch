import numpy as np
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Tensor

Numeric = Union["Tensor", np.ndarray, float, int]


def _ensure_tensor(x: Numeric) -> "Tensor":
    from .core import Tensor
    return x if isinstance(x, Tensor) else Tensor(x)


def _handle_broadcasting_grad(grad: np.ndarray, original_shape: tuple) -> np.ndarray:
    ndims_added = grad.ndim - len(original_shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)
    
    for i, (dim, grad_dim) in enumerate(zip(original_shape, grad.shape)):
        if dim == 1 and grad_dim > 1:
            grad = grad.sum(axis=i, keepdims=True)
    
    return grad


def add(a: "Tensor", b: Numeric) -> "Tensor":
    from .core import Tensor
    
    b = _ensure_tensor(b)
    out = Tensor(
        a.data + b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        _children=(a, b),
        _op="add",
    )
    
    if out.requires_grad:
        def _backward():
            if a.requires_grad:
                grad = _handle_broadcasting_grad(out.grad.data, a.data.shape)
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
            
            if b.requires_grad:
                grad = _handle_broadcasting_grad(out.grad.data, b.data.shape)
                
                if b.grad is None:
                    b.grad = Tensor(grad)
                else:
                    b.grad = Tensor(b.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def radd(a: "Tensor", b: Numeric) -> "Tensor":
    return add(a, b)


def sub(a: "Tensor", b: Numeric) -> "Tensor":
    from .core import Tensor
    
    b = _ensure_tensor(b)
    out = Tensor(
        a.data - b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        _children=(a, b),
        _op="sub",
    )
    
    if out.requires_grad:
        def _backward():
            if a.requires_grad:
                grad = _handle_broadcasting_grad(out.grad.data, a.data.shape)
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
            
            if b.requires_grad:
                grad = _handle_broadcasting_grad(out.grad.data, b.data.shape)
                
                if b.grad is None:
                    b.grad = Tensor(-grad)
                else:
                    b.grad = Tensor(b.grad.data - grad)
        
        out.grad_fn = _backward
    
    return out


def rsub(a: "Tensor", b: Numeric) -> "Tensor":
    from .core import Tensor
    
    b = _ensure_tensor(b)
    return sub(b, a)


def mul(a: "Tensor", b: Numeric) -> "Tensor":
    from .core import Tensor
    
    b = _ensure_tensor(b)
    out = Tensor(
        a.data * b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        _children=(a, b),
        _op="mul",
    )
    
    if out.requires_grad:
        def _backward():
            if a.requires_grad:
                grad = out.grad.data * b.data
                grad = _handle_broadcasting_grad(grad, a.data.shape)
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
            
            if b.requires_grad:
                grad = out.grad.data * a.data
                grad = _handle_broadcasting_grad(grad, b.data.shape)
                
                if b.grad is None:
                    b.grad = Tensor(grad)
                else:
                    b.grad = Tensor(b.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def rmul(a: "Tensor", b: Numeric) -> "Tensor":
    return mul(a, b)


def div(a: "Tensor", b: Numeric) -> "Tensor":
    from .core import Tensor
    
    b = _ensure_tensor(b)
    out = Tensor(
        a.data / b.data,
        requires_grad=a.requires_grad or b.requires_grad,
        _children=(a, b),
        _op="div",
    )
    
    if out.requires_grad:
        def _backward():
            if a.requires_grad:
                grad = out.grad.data / b.data
                grad = _handle_broadcasting_grad(grad, a.data.shape)
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
            
            if b.requires_grad:
                grad = -out.grad.data * a.data / (b.data ** 2)
                grad = _handle_broadcasting_grad(grad, b.data.shape)
                
                if b.grad is None:
                    b.grad = Tensor(grad)
                else:
                    b.grad = Tensor(b.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def rdiv(a: "Tensor", b: Numeric) -> "Tensor":
    from .core import Tensor
    
    b = _ensure_tensor(b)
    return div(b, a)


def pow(a: "Tensor", power: Union[int, float]) -> "Tensor":
    from .core import Tensor
    
    out = Tensor(
        a.data ** power,
        requires_grad=a.requires_grad,
        _children=(a,),
        _op=f"pow({power})",
    )
    
    if out.requires_grad:
        def _backward():
            if a.requires_grad:
                grad = power * (a.data ** (power - 1)) * out.grad.data
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def neg(a: "Tensor") -> "Tensor":
    from .core import Tensor
    
    out = Tensor(
        -a.data,
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="neg",
    )
    
    if out.requires_grad:
        def _backward():
            if a.requires_grad:
                if a.grad is None:
                    a.grad = Tensor(-out.grad.data)
                else:
                    a.grad = Tensor(a.grad.data - out.grad.data)
        
        out.grad_fn = _backward
    
    return out