# Where the FX Tracing Error Happens in Helion Compilation

## Overview

When using symbolic slices (e.g., `x[0:block_size]` where `block_size` is a SymInt), Helion encounters an error during the FX tracing phase of compilation. This document explains exactly where and why this error occurs.

## The Error

```
AssertionError: a.node.constant is not None
```

This error occurs when FX tries to process a slice object containing a SymInt value.

## Compilation Pipeline

The Helion compilation process follows these stages:

```
User Code
    ↓
Kernel.__call__
    ↓
BoundKernel.__init__
    ↓
HostFunction.__init__
    ↓
lower_to_device_ir()
    ↓
WalkHostAST (processes host code)
    ↓
    Finds hl.tile loop → Grid loop
    ↓
_make_fx(lambda: WalkDeviceAST.visit(node))  ← FX TRACING STARTS
    ↓
WalkDeviceAST (processes device code)
    ↓
    visit_Slice creates slice(0, SymInt)
    ↓
    visit_Subscript passes slice to hl.load
    ↓
FX tracer.create_proxy() → create_arg()
    ↓
    FAILS on SymInt in slice  ← ERROR HERE
```

## Detailed Error Flow

### 1. AST Processing (`device_ir.py:817-830`)

When processing the Python AST, `WalkDeviceAST.visit_Slice` creates a Python slice object:

```python
def visit_Slice(self, node: ast.Slice) -> slice:
    if node.lower is None:
        lower = None
    else:
        lower = self.visit(node.lower)
    if node.upper is None:
        upper = None
    else:
        upper = self.visit(node.upper)  # This returns a SymInt
    if node.step is None:
        step = None
    else:
        step = self.visit(node.step)
    return slice(lower, upper, step)  # Creates slice(0, SymInt, None)
```

For a slice like `0:block_size`, this creates `slice(0, SymInt(block_size), None)`.

### 2. Subscript Handling (`device_ir.py:941`)

The subscript operation passes the slice to a Helion operation:

```python
def visit_Subscript(self, node: ast.Subscript) -> object:
    value = node.value
    # ...
    return hl.load(self.visit(value), self._subscript_slice_proxy(node.slice))
```

The slice object (containing SymInt) is passed to `hl.load()`.

### 3. FX Proxy Creation (`_decorators.py:169`)

When the Helion operation is traced by FX:

```python
proxy_out = tracer.create_proxy(
    "call_function",
    wrapper,
    *args_to_proxies(tracer, flat_args, {}),
)
```

FX needs to convert all arguments to a form it can trace. This calls `create_arg` on the slice.

### 4. FX create_arg Failure (`torch/fx/proxy.py:379`)

FX tries to process the slice recursively:

```python
def create_arg(self, a):
    if isinstance(a, slice):
        return slice(
            self.create_arg(a.start),
            self.create_arg(a.stop),  # ← FAILS HERE
            self.create_arg(a.step)
        )
```

When it tries to process the SymInt in `a.stop`, it expects the SymInt to have a `node.constant` attribute (concrete value), which symbolic values don't have:

```python
# In torch/fx/experimental/proxy_tensor.py:1126
assert a.node.constant is not None  # ← AssertionError
```

## Why This Happens

The error occurs because:

1. **FX expects concrete values**: FX tracing is designed to work with concrete values or properly tracked symbolic values through its proxy system.

2. **Slice objects hide symbolic nature**: When a SymInt is placed inside a Python slice object, FX loses track of its symbolic nature.

3. **No proxy for slice objects**: FX knows how to create proxies for tensors and basic types, but slice objects with symbolic bounds fall through the cracks.

## Example Code That Triggers the Error

```python
@helion.kernel(use_default_config=True)
def simple_slice_kernel(x: torch.Tensor) -> torch.Tensor:
    N = x.size(0)
    block_size = hl.register_block_size(N)  # Creates SymInt
    
    out = torch.zeros_like(x)
    
    for tile in hl.tile(N, block_size=block_size):
        # This line causes the error:
        data = x[tile][0:block_size]  # slice(0, SymInt)
        out[tile] = data
    
    return out
```

## The Solution: SliceProxy

SliceProxy was created to solve this exact problem by:

1. **Storing bounds separately**: Instead of creating `slice(0, SymInt)`, it stores the bounds in `CompileEnvironment`
2. **Passing only an ID**: Through FX tracing, only a simple integer ID is passed
3. **Reconstructing later**: During code generation, the slice bounds are retrieved and used

This avoids the FX serialization issue entirely by never passing slice objects with SymInt through the FX tracing machinery.

## Summary

The FX tracing error with symbolic slices occurs during the compilation phase when:
- Helion separates device code from host code
- FX traces the device code to build a computation graph
- A slice object containing SymInt is encountered
- FX cannot serialize the SymInt within the slice object

This is a fundamental limitation of PyTorch's FX system and is the primary motivation for the SliceProxy design.