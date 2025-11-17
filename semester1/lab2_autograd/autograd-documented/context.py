"""Context managers for gradient control.

This module provides context managers to control gradient tracking:
- no_grad: Disable gradient tracking temporarily
- enable_grad: Enable gradient tracking temporarily

These are useful for:
- Inference (no need for gradients)
- Parameter updates (in-place operations without tracking)
- Memory optimization (free computational graph)
- Debugging (isolate gradient issues)

Implementation:
    Context managers modify the global _grad_enabled flag on the Tensor class.
    This flag is checked during tensor creation to determine if gradients
    should be tracked.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Tensor


class no_grad:
    """
    Context manager to disable gradient tracking.
    
    Within this context, all operations on tensors will not build the
    computational graph, even if tensors have requires_grad=True.
    
    Use cases:
        - Inference: Don't need gradients for predictions
        - Parameter updates: In-place modifications without tracking
        - Memory optimization: Save memory by not building graph
        - Performance: Faster operations without gradient bookkeeping
    
    Implementation:
        Sets Tensor._grad_enabled = False on entry
        Restores previous value on exit
    
    Attributes:
        prev (bool): Previous state of gradient tracking (saved on entry)
    """
    
    def __enter__(self):
        """Enter no_grad context.
        
        Saves the current gradient tracking state and disables tracking.
        
        Returns:
            self (for context manager protocol)
        """
        from .core import Tensor
        
        self.prev = Tensor._grad_enabled
        Tensor._grad_enabled = False
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit no_grad context.
        
        Restores the previous gradient tracking state.
        
        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)
        
        Returns:
            False (don't suppress exceptions)
        """
        from .core import Tensor
        
        Tensor._grad_enabled = self.prev
        return False


class enable_grad:
    """
    Context manager to enable gradient tracking.
    
    Within this context, gradient tracking is enabled even if it was
    disabled by an outer no_grad context.
    
    Use cases:
        - Enable gradients within a no_grad block
        - Testing gradient computation
        - Selective gradient tracking
    
    Implementation:
        Sets Tensor._grad_enabled = True on entry
        Restores previous value on exit
    
    Attributes:
        prev (bool): Previous state of gradient tracking (saved on entry)
    """
    
    def __enter__(self):
        """Enter enable_grad context.
        
        Saves the current gradient tracking state and enables tracking.
        
        Returns:
            self (for context manager protocol)
        """
        from .core import Tensor
        
        self.prev = Tensor._grad_enabled
        Tensor._grad_enabled = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit enable_grad context.
        
        Restores the previous gradient tracking state.
        
        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)
        
        Returns:
            False (don't suppress exceptions)
        """
        from .core import Tensor
        
        Tensor._grad_enabled = self.prev
        return False


# Test context managers when run directly
if __name__ == "__main__":
    from .core import Tensor
    
    print("=== Testing Context Managers ===\n")
    
    # Test no_grad
    print("Test 1: no_grad context")
    x = Tensor([1.0], requires_grad=True)
    
    y_with_grad = x * 2
    print(f"With grad: y.requires_grad = {y_with_grad.requires_grad}")
    
    with no_grad():
        y_no_grad = x * 2
        print(f"No grad: y.requires_grad = {y_no_grad.requires_grad}")
    
    y_with_grad_again = x * 2
    print(f"After context: y.requires_grad = {y_with_grad_again.requires_grad}\n")
    
    # Test enable_grad inside no_grad
    print("Test 2: enable_grad inside no_grad")
    with no_grad():
        y1 = x * 2
        print(f"Inside no_grad: y1.requires_grad = {y1.requires_grad}")
        
        with enable_grad():
            z = Tensor([3.0], requires_grad=True)
            y2 = z * 4
            print(f"Inside enable_grad: y2.requires_grad = {y2.requires_grad}")
        
        y3 = x * 5
        print(f"Back to no_grad: y3.requires_grad = {y3.requires_grad}\n")
    
    # Test nesting
    print("Test 3: Nested contexts")
    print(f"Initial: Tensor._grad_enabled = {Tensor._grad_enabled}")
    
    with no_grad():
        print(f"In no_grad: Tensor._grad_enabled = {Tensor._grad_enabled}")
        
        with enable_grad():
            print(f"In enable_grad: Tensor._grad_enabled = {Tensor._grad_enabled}")
        
        print(f"Back to no_grad: Tensor._grad_enabled = {Tensor._grad_enabled}")
    
    print(f"After all: Tensor._grad_enabled = {Tensor._grad_enabled}\n")
    
    print("All context managers working correctly!")