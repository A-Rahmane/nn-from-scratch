import numpy as np
from typing import Optional, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Tensor


def sum_op(
    a: "Tensor",
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> "Tensor":
    from .core import Tensor
    
    out = Tensor(
        a.data.sum(axis=axis, keepdims=keepdims),
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="sum",
    )
    
    if out.requires_grad:
        def _backward():
            if a.requires_grad:
                grad = out.grad.data
                
                if not keepdims and axis is not None:
                    if isinstance(axis, int):
                        grad = np.expand_dims(grad, axis=axis)
                    else:
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, axis=ax)
                
                grad = np.broadcast_to(grad, a.data.shape)
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def mean(
    a: "Tensor",
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> "Tensor":
    from .core import Tensor
    
    out = Tensor(
        a.data.mean(axis=axis, keepdims=keepdims),
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="mean",
    )
    
    if out.requires_grad:
        def _backward():
            if a.requires_grad:
                grad = out.grad.data
                
                if axis is None:
                    n = a.data.size
                else:
                    if isinstance(axis, int):
                        n = a.data.shape[axis]
                    else:
                        n = np.prod([a.data.shape[ax] for ax in axis])
                
                grad = grad / n
                
                if not keepdims and axis is not None:
                    if isinstance(axis, int):
                        grad = np.expand_dims(grad, axis=axis)
                    else:
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, axis=ax)
                
                grad = np.broadcast_to(grad, a.data.shape)
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out


def max_op(
    a: "Tensor",
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> "Tensor":
    from .core import Tensor
    
    max_values = a.data.max(axis=axis, keepdims=keepdims)
    
    out = Tensor(
        max_values,
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="max",
    )
    
    if out.requires_grad:
        def _backward():
            if a.requires_grad:
                grad = out.grad.data
                
                if not keepdims and axis is not None:
                    max_broadcast = max_values
                    if isinstance(axis, int):
                        max_broadcast = np.expand_dims(max_broadcast, axis=axis)
                    else:
                        for ax in sorted(axis):
                            max_broadcast = np.expand_dims(max_broadcast, axis=ax)
                else:
                    max_broadcast = max_values
                
                mask = (a.data == max_broadcast).astype(float)
                
                count = mask.sum(axis=axis, keepdims=True)
                
                mask = mask / count
                
                if not keepdims and axis is not None:
                    if isinstance(axis, int):
                        grad = np.expand_dims(grad, axis=axis)
                    else:
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, axis=ax)
                
                grad = np.broadcast_to(grad, a.data.shape)
                
                grad = grad * mask
                
                if a.grad is None:
                    a.grad = Tensor(grad)
                else:
                    a.grad = Tensor(a.grad.data + grad)
        
        out.grad_fn = _backward
    
    return out