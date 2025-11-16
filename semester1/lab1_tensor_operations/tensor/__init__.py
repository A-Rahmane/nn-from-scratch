from .core import Tensor

__all__ = ['Tensor']

__version__ = '1.0.0'
__author__ = 'MENOUER Abderrahmane'

def tensor(data):
    return Tensor(data)


def zeros(shape):
    import numpy as np
    return Tensor(np.zeros(shape, dtype=np.float32))


def ones(shape):
    import numpy as np
    return Tensor(np.ones(shape, dtype=np.float32))


def rand(shape):
    import numpy as np
    return Tensor(np.random.rand(*shape))


def randn(shape):
    import numpy as np
    return Tensor(np.random.randn(*shape))