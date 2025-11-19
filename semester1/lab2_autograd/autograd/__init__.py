from .core import Tensor
from .context import no_grad, enable_grad
from .utilities import (
    zero_grad,
    clip_grad_value,
    clip_grad_norm,
    numerical_gradient,
    check_gradients,
)

try:
    from .visualization import visualize_computation_graph
    __all__ = [
        'Tensor',
        'no_grad',
        'enable_grad',
        'zero_grad',
        'clip_grad_value',
        'clip_grad_norm',
        'numerical_gradient',
        'check_gradients',
        'visualize_computation_graph',
    ]
except ImportError:
    __all__ = [
        'Tensor',
        'no_grad',
        'enable_grad',
        'zero_grad',
        'clip_grad_value',
        'clip_grad_norm',
        'numerical_gradient',
        'check_gradients',
    ]

__version__ = '1.0.0'
__author__ = 'MENOUER Abderrahmane'

def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad)