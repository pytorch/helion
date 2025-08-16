# Issue #458: Type propagation error in pre-processing code

## Metadata
- **State**: OPEN
- **Author**: [StrongerXi](https://github.com/StrongerXi)
- **Created**: August 07, 2025 at 22:50 UTC
- **Updated**: August 07, 2025 at 23:51 UTC

## Description

**Describe the bug**
Error:
```
  File "/home/ryanguo99/repos/helion/examples/grouped_gemm.py", line 122, in <module>
    main()
  File "/home/ryanguo99/repos/helion/examples/grouped_gemm.py", line 109, in main
    run_example(
  File "/home/ryanguo99/repos/helion/helion/_testing.py", line 372, in run_example
    func(*args).to(torch.float32),
    ^^^^^^^^^^^
  File "/home/ryanguo99/repos/helion/helion/runtime/kernel.py", line 272, in __call__
    return self.bind(args)(*args)
           ^^^^^^^^^^^^^^^
  File "/home/ryanguo99/repos/helion/helion/runtime/kernel.py", line 158, in bind
    bound_kernel = BoundKernel(self, args)
                   ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ryanguo99/repos/helion/helion/runtime/kernel.py", line 338, in __init__
    self.host_function: HostFunction = HostFunction(
                                       ^^^^^^^^^^^^^
  File "/home/ryanguo99/repos/helion/helion/_compiler/host_function.py", line 108, in __init__
    propagate_types(self, fake_args)
  File "/home/ryanguo99/repos/helion/helion/_compiler/type_propagation.py", line 2251, in propagate_types
    prop.visit(stmt)
  File "/home/ryanguo99/repos/helion/helion/_compiler/type_propagation.py", line 1580, in visit
    type_info = visitor(node)
                ^^^^^^^^^^^^^
  File "/home/ryanguo99/repos/helion/helion/_compiler/type_propagation.py", line 2086, in visit_For
    body = self._loop_body(node.body)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ryanguo99/repos/helion/helion/_compiler/type_propagation.py", line 2050, in _loop_body
    self.visit(stmt)
  File "/home/ryanguo99/repos/helion/helion/_compiler/type_propagation.py", line 1580, in visit
    type_info = visitor(node)
                ^^^^^^^^^^^^^
  File "/home/ryanguo99/repos/helion/helion/_compiler/type_propagation.py", line 1974, in visit_Assign
    type_info = self.visit(node.value)
                ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ryanguo99/repos/helion/helion/_compiler/type_propagation.py", line 1580, in visit
    type_info = visitor(node)
                ^^^^^^^^^^^^^
  File "/home/ryanguo99/repos/helion/helion/_compiler/type_propagation.py", line 1945, in visit_Subscript
    return value_type.propagate_getitem(slice_type, self.origin())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ryanguo99/repos/helion/helion/_compiler/type_propagation.py", line 1238, in propagate_getitem
    return super().propagate_getitem(key, origin)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ryanguo99/repos/helion/helion/_compiler/type_propagation.py", line 1200, in propagate_getitem
    return super().propagate_getitem(key, origin)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ryanguo99/repos/helion/helion/_compiler/type_propagation.py", line 323, in propagate_getitem
    raise exc.TypeInferenceError(
helion.exc.TypeInferenceError: Subscript not supported with self=SequenceType([TensorType([As_size0, As_size1], torch.float16), TensorType([As_size0, As_size1], torch.float16), TensorType([As_size0, As_size1], torch.float16), TensorType([As_size0, As_size1], torch.float16)]) key=SymIntType(u2)
While processing:
  File "/home/ryanguo99/repos/helion/examples/grouped_gemm.py", line 34, in grouped_gemm
    A = As[i]
        ^^^^^
```

**To Reproduce**
```python
"""
Helion Grouped GEMM Kernel Example
============================
TODO
"""

# %%
from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from torch import Tensor

import helion
from helion._testing import run_example
import helion.language as hl


# %%
@helion.kernel
def grouped_gemm(
    As: list[Tensor],
    Bs: list[Tensor],
) -> list[Tensor]:
    group_size = len(As)
    device = As[0].device

    flattened_As = []
    flattened_Bs = []
    g_sizes = []
    max_m, max_n = 0, 0
    A_offset, B_offset, C_offset = 0, 0, 0
    for i in range(group_size):
        A = As[i]
        B = Bs[i]
        M, K = A.shape
        K, N = B.shape
        flattened_As.append(A.reshape(-1))
        flattened_Bs.append(B.reshape(-1))
        g_sizes += [M, N, K, A_offset, B_offset, C_offset]
        max_m = max(max_m, M)
        max_n = max(max_n, N)
        A_offset += M * K
        B_offset += K * N
        C_offset += M * N
    flattened_A = torch.cat(flattened_As)
    flattened_B = torch.cat(flattened_Bs)
    flattened_C = torch.empty(C_offset, dtype=x.dtype, device=x.device)

    # note these are device tensors
    g_sizes_tensor = torch.tensor(g_sizes, dtype=torch.int32, device=device)

    for group in hl.grid(group_size):
        # Get metadata for this GEMM
        m = hl.load(g_sizes_tensor, [group*3])
        n = hl.load(g_sizes_tensor, [group*3 + 1])
        k = hl.load(g_sizes_tensor, [group*3 + 2])
        A_offset = hl.load(g_sizes_tensor, [group*3 + 3])
        B_offset = hl.load(g_sizes_tensor, [group*3 + 4])
        C_offset = hl.load(g_sizes_tensor, [group*3 + 5])

        # Get the input and output tensors
        a = flattened_A[A_offset: A_offset+m*k]
        b = flattened_B[B_offset: B_offset+k*n]
        c = flattened_C[C_offset: C_offset+m*n]

        # Actual GEMM
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            out[tile_m, tile_n] = acc.to(C.dtype)

    results = []
    offset = 0
    for m, _, n in g_sizes:
        next_offset = offset + m * n
        result = flattened_C[offset:next_offset].reshape(m, n)
        results.append(result)
        offset = next_offset
    return out


def grouped_gemm_reference(As, Bs):
    out = []
    for a, b in zip(As, Bs):
        out.append(torch.matmul(a, b))
    return out


# %%
def main() -> None:
    batch_size = 32
    dim = 64
    device = "cuda"

    shapes = (
        (1024, 512, 2048),
        (512, 1024, 2048),
        (1024, 2048, 4096),
        (4096, 2048, 4096),
    )
    As, Bs = [], []
    for m, k, n in shapes:
        As.append(torch.rand(m, k, device="cuda", dtype=torch.float16))
        Bs.append(torch.rand(k, n, device="cuda", dtype=torch.float16))

    run_example(
        grouped_gemm,
        grouped_gemm_reference,
        (As, Bs),
        kernel_name="helion",
        baseline_name="torch",
        rtol=1e-3,
        atol=1e-3,
    )

# %%
if __name__ == "__main__":
    main()
```

**Expected behavior**
No error, or better error message to suggest workaround (in case it's a user error).

**Versions**
PyTorch/Triton/Helion versions and any other relevant library version.

**Additional context**
Add any other context about the problem here.


## Comments

### Comment 1 by [yf225](https://github.com/yf225)
*Posted on August 07, 2025 at 22:58 UTC*

One thought: since the code before the first hl.grid / hl.tile is NOT part of the generated Triton kernel, maybe we can move those code into the caller instead, and maybe this will work.

---

### Comment 2 by [StrongerXi](https://github.com/StrongerXi)
*Posted on August 07, 2025 at 23:51 UTC*

Yeah I manually did that and it worked. But I _thought_ helion does that automatically (e.g., for the output tensor allocation in examples). Maybe it's just a use education thing.

---
