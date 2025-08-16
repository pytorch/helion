# Issue #493: Arithmetic on input floats makes them fp64

## Metadata
- **State**: OPEN
- **Author**: [v0i0](https://github.com/v0i0)
- **Created**: August 13, 2025 at 16:51 UTC
- **Updated**: August 13, 2025 at 16:51 UTC

## Description

**Describe the bug**
Passing a python `float` into a kernel is fine as long as it is used directly, but any use in arithmetic seems to make it fp64, and thus upcast other parts of the program.

**To Reproduce**

This compiles:
```
import helion
import helion.language as hl
import torch

@helion.kernel
def kernel(a, beta):
    for t in hl.tile(a.shape[0]):
        b = a[t]
        a[t] = beta * b
    return a

print(kernel(torch.randn(1000, device="cuda"), 1.5))
```

This does not:
```
import helion
import helion.language as hl
import torch

@helion.kernel
def kernel(a, beta):
    for t in hl.tile(a.shape[0]):
        b = a[t]
        a[t] = (1 - beta) * b
    return a

print(kernel(torch.randn(1000, device="cuda"), 1.5))
```

and yields 

```
triton.compiler.errors.CompilationError: at 7:4:
def _kernel_kernel(a, a_size_0, a_stride_0, beta, _BLOCK_SIZE_0: tl.constexpr):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0 * _BLOCK_SIZE_0
    b = tl.load(tl.make_block_ptr(a, [a_size_0], [a_stride_0], [offset_0], [_BLOCK_SIZE_0], [0]), boundary_check=[0], padding_option='zero')
    sub = tl.full([], 1.0, tl.float64) + -1 * beta
    v_0 = b * sub
    tl.store(tl.make_block_ptr(a, [a_size_0], [a_stride_0], [offset_0], [_BLOCK_SIZE_0], [0]), v_0, boundary_check=[0])
    ^
Traceback (most recent call last):
  File "/home/mhoehnerbach/local/env/helion/lib/python3.12/site-packages/triton/language/core.py", line 42, in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/mhoehnerbach/local/env/helion/lib/python3.12/site-packages/triton/language/core.py", line 2216, in store
    return _semantic.store(pointer, value, mask, boundary_check, cache_modifier, eviction_policy)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mhoehnerbach/local/env/helion/lib/python3.12/site-packages/triton/language/semantic.py", line 1289, in store
    return self._store_block_pointer(ptr, val, mask, boundary_check, cache, eviction)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/mhoehnerbach/local/env/helion/lib/python3.12/site-packages/triton/language/semantic.py", line 1219, in _store_block_pointer
    assert ptr.type.element_ty.element_ty == val.type.element_ty, f"Block element type({ptr.type.element_ty.element_ty}) and value element type({val.type.element_ty}) mismatch"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Block element type(fp32) and value element type(fp64) mismatch
```

**Expected behavior**
Arithmetic on floats should work.

**Versions**
main

**Additional context**
-

## Comments

*No comments yet.*
