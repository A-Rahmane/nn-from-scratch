import numpy as np
from typing import List, Callable, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Tensor


def zero_grad(*tensors: "Tensor") -> None:
    for tensor in tensors:
        tensor.zero_grad()


def clip_grad_value(tensors: List["Tensor"], clip_value: float) -> None:
    for tensor in tensors:
        if tensor.grad is not None:
            tensor.grad.data = np.clip(
                tensor.grad.data,
                -clip_value,
                clip_value
            )


def clip_grad_norm(tensors: List["Tensor"], max_norm: float) -> float:
    for tensor in tensors:
        if tensor.grad is not None:
            total_norm += np.sum(tensor.grad.data ** 2)
    
    total_norm = np.sqrt(total_norm)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        for tensor in tensors:
            if tensor.grad is not None:
                tensor.grad.data *= clip_coef
    
    return float(total_norm)


def numerical_gradient(
    func: Callable[["Tensor"], "Tensor"],
    tensor: "Tensor",
    epsilon: float = 1e-5
) -> np.ndarray:
    grad = np.zeros_like(tensor.data)
    
    it = np.nditer(tensor.data, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = tensor.data[idx]
        
        tensor.data[idx] = old_value + epsilon
        fxh = func(tensor).data.item()
        
        tensor.data[idx] = old_value - epsilon
        fxl = func(tensor).data.item()
        
        grad[idx] = (fxh - fxl) / (2 * epsilon)
        
        tensor.data[idx] = old_value
        it.iternext()
    
    return grad


def check_gradients(
    func: Callable[["Tensor"], "Tensor"],
    tensor: "Tensor",
    epsilon: float = 1e-5,
    tolerance: float = 1e-5
) -> Tuple[bool, float]:
    tensor.zero_grad()
    output = func(tensor)
    output.backward()
    analytical_grad = tensor.grad.data.copy()
    
    numerical_grad = numerical_gradient(func, tensor, epsilon)
    
    diff = np.abs(analytical_grad - numerical_grad)
    max_diff = np.max(diff)
    
    match = max_diff < tolerance
    
    return match, float(max_diff)
