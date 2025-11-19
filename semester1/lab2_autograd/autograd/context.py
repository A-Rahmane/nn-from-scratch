from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Tensor


class no_grad:
    def __enter__(self):
        from .core import Tensor
        
        self.prev = Tensor._grad_enabled
        Tensor._grad_enabled = False
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        from .core import Tensor
        
        Tensor._grad_enabled = self.prev
        return False


class enable_grad:
    def __enter__(self):
        from .core import Tensor
        
        self.prev = Tensor._grad_enabled
        Tensor._grad_enabled = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        from .core import Tensor
        
        Tensor._grad_enabled = self.prev
        return False