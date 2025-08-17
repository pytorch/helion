"""Debug script to understand SliceProxy issue."""

import torch
from helion.language.slice_proxy import SliceProxy
from helion._compiler.compile_environment import CompileEnvironment

# Test SliceProxy.to_slice() with symbolic bounds
env = CompileEnvironment()
CompileEnvironment.set_current(env)

# Create a SymInt
block_size_m = torch.SymInt(32)
block_size_n = torch.SymInt(16)
end_idx = block_size_m - block_size_n + 1  # Should be 17

# Register a slice with symbolic bounds
slice_id = env.register_slice(0, end_idx, None)
proxy = SliceProxy(slice_id)

print(f"SliceProxy created: {proxy}")
print(f"Slice ID: {proxy.slice_id}")
print(f"Bounds in env: {env.slice_bounds.get(proxy.slice_id, 'Not found')}")

# Try to convert to slice
try:
    slice_obj = proxy.to_slice()
    print(f"Converted slice: {slice_obj}")
    print(f"  start: {slice_obj.start} (type: {type(slice_obj.start)})")
    print(f"  stop: {slice_obj.stop} (type: {type(slice_obj.stop)})")
    print(f"  step: {slice_obj.step} (type: {type(slice_obj.step)})")
except Exception as e:
    print(f"Error converting to slice: {e}")
    import traceback
    traceback.print_exc()
    
# Let's also check what happens with concrete values
env2 = CompileEnvironment()
CompileEnvironment.set_current(env2)

slice_id2 = env2.register_slice(0, 17, None)
proxy2 = SliceProxy(slice_id2)

print(f"\nWith concrete values:")
print(f"SliceProxy created: {proxy2}")
slice_obj2 = proxy2.to_slice()
print(f"Converted slice: {slice_obj2}")