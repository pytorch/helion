"""Demonstration of FX tracing issues with symbolic bounds"""

import torch
from torch.fx import symbolic_trace
from torch.fx.experimental import proxy_tensor

# Example 1: Regular slice - works fine
def regular_slice_example(tensor):
    return tensor[0:10]

# Example 2: Slice with SymInt - causes issues
def symint_slice_attempt(tensor, size: torch.SymInt):
    # This would fail during FX tracing:
    # return tensor[0:size]
    
    # Why it fails:
    # 1. Python creates slice(0, size) where size is SymInt
    # 2. FX tries to trace through __getitem__(tensor, slice(0, size))
    # 3. FX needs to store/reconstruct the slice object
    # 4. But slice objects aren't FX-traceable with SymInt
    
    # The slice object itself becomes problematic because:
    # - hash(slice(0, SymInt)) may fail
    # - FX graph serialization can't handle slice with SymInt
    # - The symbolic relationship gets lost
    pass

# Example 3: What SliceProxy does
class SliceProxy(torch.Tensor):
    """Simplified version showing the concept"""
    def __new__(cls, slice_id: int):
        # Create a dummy tensor that carries just an ID
        return torch.Tensor._make_wrapper_subclass(
            cls, size=(), dtype=torch.int64, device="meta"
        )
    
    def __init__(self, slice_id: int):
        self.slice_id = slice_id

# Usage with SliceProxy:
def slice_proxy_solution(tensor, size: torch.SymInt):
    # Store bounds externally
    slice_id = register_slice_bounds(0, size, 1)  # Returns simple int
    
    # Pass simple object through FX
    slice_proxy = SliceProxy(slice_id)
    
    # FX sees: __getitem__(tensor, SliceProxy(0))
    # Much easier to trace - no SymInt in the data structure
    return tensor[slice_proxy]

# Example 4: Why we can't just use tuples
def tuple_attempt(tensor, size: torch.SymInt):
    # This also has issues:
    # return tensor[(0, size)]
    
    # Problems:
    # 1. Ambiguous - is it multi-dimensional indexing or a slice?
    # 2. Still has SymInt in a data structure
    # 3. FX still needs to handle the tuple with SymInt
    pass

# Example 5: The FX graph difference
def show_fx_graph_difference():
    """Demonstrates what FX sees in each case"""
    
    # What FX wants to see (concrete values):
    # %arg0 : Tensor
    # %getitem : Tensor = call_function[target=operator.getitem](
    #     args=(%arg0, slice(0, 10, None)), kwargs={}
    # )
    
    # What happens with SymInt (problematic):
    # %arg0 : Tensor
    # %arg1 : SymInt 
    # %slice : slice = ???  # How to create slice with SymInt in graph?
    # %getitem : Tensor = call_function[target=operator.getitem](
    #     args=(%arg0, %slice), kwargs={}
    # )
    
    # What happens with SliceProxy (clean):
    # %arg0 : Tensor
    # %slice_proxy : SliceProxy = call_function[target=SliceProxy](
    #     args=(0,), kwargs={}  # Just an ID!
    # )
    # %getitem : Tensor = call_function[target=operator.getitem](
    #     args=(%arg0, %slice_proxy), kwargs={}
    # )
    pass

# Example 6: Alternative without SliceProxy - Direct FX node creation
def alternative_solution(tensor, start: torch.SymInt, stop: torch.SymInt):
    """What we could do if FX was enhanced"""
    
    # Instead of creating Python slice object
    # Directly create FX nodes for the indexing operation
    
    # Pseudo-code for enhanced FX:
    # %start : SymInt = get_arg(1)
    # %stop : SymInt = get_arg(2) 
    # %result : Tensor = call_function[target=symbolic_slice](
    #     args=(%tensor, %start, %stop), kwargs={}
    # )
    
    # This avoids slice objects entirely
    # But requires modifying how Python indexing is traced
    pass

if __name__ == "__main__":
    print("This file demonstrates FX tracing challenges with symbolic bounds")
    print("Run with appropriate PyTorch symbolic shape environment to see details")