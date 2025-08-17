#!/usr/bin/env python3
"""Trace where the FX error happens in Helion compilation"""

import torch
import helion
import helion.language as hl

# Add some debug prints to understand the flow
print("\n=== COMPILATION FLOW ANALYSIS ===\n")

@helion.kernel(use_default_config=True)
def simple_slice_kernel(x: torch.Tensor) -> torch.Tensor:
    N = x.size(0)
    block_size = hl.register_block_size(N)
    
    out = torch.zeros_like(x)
    
    for tile in hl.tile(N, block_size=block_size):
        # This line causes the error
        data = x[tile][0:block_size]
        out[tile] = data
    
    return out

# Let's trace the error path from the stack trace:
print("1. User calls kernel:")
print("   simple_slice_kernel(x)")
print("   ↓")
print("2. Kernel.__call__ → bind(args)(*args)")
print("   ↓")
print("3. BoundKernel.__init__ creates HostFunction:")
print("   HostFunction(kernel.fn, fake_args, constexpr_args)")
print("   ↓")
print("4. HostFunction.__init__ calls lower_to_device_ir(self)")
print("   ↓")
print("5. lower_to_device_ir creates WalkHostAST and visits AST nodes")
print("   ↓")
print("6. WalkHostAST.visit_For detects a GRID loop (hl.tile)")
print("   It calls: _make_fx(lambda: WalkDeviceAST(...).visit(node))")
print("   ↓")
print("7. _make_fx uses torch.fx.experimental.proxy_tensor.make_fx")
print("   This starts FX tracing of the device code")
print("   ↓")
print("8. WalkDeviceAST.visit_For processes the device loop body")
print("   ↓")
print("9. WalkDeviceAST.visit_Assign processes: data = x[tile][0:block_size]")
print("   ↓")
print("10. WalkDeviceAST.visit_Subscript handles x[tile][0:block_size]")
print("    It calls hl.subscript() with the slice")
print("    ↓")
print("11. FX tracer tries to create_arg for the slice object")
print("    slice(0, block_size) where block_size is SymInt")
print("    ↓")
print("12. FX's create_arg fails on SymInt in slice.stop")
print("    AssertionError: a.node.constant is not None")
print("\n=== KEY INSIGHT ===")
print("The error happens during FX tracing of device code,")
print("when FX tries to serialize a slice object containing SymInt.")
print("\nThis is EXACTLY why SliceProxy exists - to avoid passing")
print("slice objects with SymInt through FX tracing!")

# Try to run it to see the error
x = torch.randn(64, device='cuda')
try:
    result = simple_slice_kernel(x)
except Exception as e:
    print(f"\nError confirmed: {type(e).__name__}")
    
print("\n=== COMPILATION STAGES ===")
print("1. Host AST Analysis (Python AST)")
print("2. Device/Host Separation (Grid loops → Device IR)")
print("3. FX Tracing of Device Code ← ERROR HAPPENS HERE")
print("4. Type Propagation")
print("5. Code Generation")
print("6. Kernel Launch")