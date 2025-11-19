import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Tensor


def matmul(a: "Tensor", b: "Tensor") -> "Tensor":
    from .core import Tensor
    
    if not isinstance(b, Tensor):
        b = Tensor(b)
    
    try:
        result = np.matmul(a.data, b.data)
    except ValueError as e:
        raise ValueError(
            f"Incompatible shapes for matmul: {a.shape} and {b.shape}. "
            f"Inner dimensions must match for matrix multiplication. "
            f"Error: {str(e)}"
        ) from e
    
    out = Tensor(
        result,
        requires_grad=a.requires_grad or b.requires_grad,
        _children=(a, b),
        _op="matmul",
    )
    
    if out.requires_grad:
        def _backward():
            if a.requires_grad:
                
                if b.data.ndim == 1:
                    grad = np.outer(out.grad.data, b.data)
                else:
                    grad = np.matmul(out.grad.data, b.data.T)
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
            
            if b.requires_grad:
                
                if a.data.ndim == 1:
                    grad = np.outer(a.data, out.grad.data)
                else:
                    grad = np.matmul(a.data.T, out.grad.data)
                
                if b.grad is None:
                    b.grad = Tensor(grad)
                else:
                    b.grad = Tensor(b.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out