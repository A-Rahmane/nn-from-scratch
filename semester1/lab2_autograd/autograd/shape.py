import numpy as np
from typing import Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Tensor


def reshape(a: "Tensor", *shape: Union[int, Tuple[int, ...]]) -> "Tensor":
    from .core import Tensor
    
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    
    out = Tensor(
        a.data.reshape(shape),
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="reshape",
    )
    
    if out.requires_grad:
        original_shape = a.data.shape
        
        def _backward():
            if a.requires_grad:
                grad = out.grad.data.reshape(original_shape)
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def transpose(a: "Tensor", *axes: int) -> "Tensor":
    from .core import Tensor
    
    if len(axes) == 0:
        out_data = a.data.T
        axes_used = None
    else:
        out_data = np.transpose(a.data, axes)
        axes_used = axes
    
    out = Tensor(
        out_data,
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="transpose",
    )
    
    if out.requires_grad:
        def _backward():
            if a.requires_grad:
                if axes_used is None:
                    grad = out.grad.data.T
                else:
                    inv_axes = np.argsort(axes_used)
                    grad = np.transpose(out.grad.data, inv_axes)
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def get_T(a: "Tensor") -> "Tensor":
    return transpose(a)