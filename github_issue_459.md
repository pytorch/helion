# Issue #459: Shape specialization error from reshape

## Metadata
- **State**: OPEN
- **Author**: [StrongerXi](https://github.com/StrongerXi)
- **Created**: August 07, 2025 at 22:52 UTC
- **Updated**: August 12, 2025 at 18:22 UTC
- **Assignees**: [yf225](https://github.com/yf225)

## Description

**Describe the bug**
```
Traceback (most recent call last):
  File "/home/ryanguo99/repos/helion/examples/grouped_gemm.py", line 138, in <module>
    main()
  File "/home/ryanguo99/repos/helion/examples/grouped_gemm.py", line 125, in main
    run_example(
  File "/home/ryanguo99/repos/helion/helion/_testing.py", line 372, in run_example
    func(*args).to(torch.float32),
    ^^^^^^^^^^^
  File "/home/ryanguo99/repos/helion/examples/grouped_gemm.py", line 87, in grouped_gemm
    grouped_gemm_impl(flattened_A, flattened_B, flattened_C, g_sizes_tensor, g_size_dummy)
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
  File "/home/ryanguo99/repos/helion/helion/_compiler/type_propagation.py", line 1916, in visit_Call
    return func.propagate_call(tuple(args), kwargs, self.origin())  # pyright: ignore[reportReturnType]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ryanguo99/repos/helion/helion/_compiler/type_propagation.py", line 615, in propagate_call
    output_type = TypeInfo.from_example(
                  ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ryanguo99/repos/helion/helion/_compiler/type_propagation.py", line 195, in from_example
    return TensorType(origin, fake_value=value)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ryanguo99/repos/helion/helion/_compiler/type_propagation.py", line 383, in __init__
    CompileEnvironment.current().add_kernel_tensor_size(fake_value.size())
  File "/home/ryanguo99/repos/helion/helion/_compiler/compile_environment.py", line 83, in add_kernel_tensor_size
    raise exc.ShapeSpecializingAllocation
helion.exc.ShapeSpecializingAllocation: Using a tensor size in a device allocation requires specialization. Use `hl.specialize` or `hl.constexpr` to specialize the size.
While processing:
  File "/home/ryanguo99/repos/helion/examples/grouped_gemm.py", line 40, in grouped_gemm_impl
    A = flattened_A[A_offset: A_offset+m*k].reshape(m, k)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
def grouped_gemm_impl(
    flattened_A: Tensor,
    flattened_B: Tensor,
    flattened_C: Tensor,
    group_sizes: Tensor,
    group_size_dummy: Tensor, # of shape [group_size]
) -> list[Tensor]:
    group_size = group_size_dummy.shape[0]

    for group in hl.grid(group_size):
        # Get metadata for this GEMM
        m = hl.load(group_sizes, [group*3])
        n = hl.load(group_sizes, [group*3 + 1])
        k = hl.load(group_sizes, [group*3 + 2])
        A_offset = hl.load(group_sizes, [group*3 + 3])
        B_offset = hl.load(group_sizes, [group*3 + 4])
        C_offset = hl.load(group_sizes, [group*3 + 5])

        # Get the input and output tensors, and unflatten them.
        A = flattened_A[A_offset: A_offset+m*k].reshape(m, k)
        B = flattened_B[B_offset: B_offset+k*n].reshape(k, n)
        C = flattened_C[C_offset: C_offset+m*n].reshape(m, n)

        # Actual GEMM
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, A[tile_m, tile_k], B[tile_k, tile_n])
            out[tile_m, tile_n] = acc.to(C.dtype)

    return out


def grouped_gemm(
    As: list[Tensor],
    Bs: list[Tensor],
) -> list[Tensor]:
    group_size = len(As)
    device = As[0].device

    # flatten inputs, and constrct a flat output buffer.
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
    flattened_C = torch.empty(C_offset, dtype=flattened_A.dtype, device=device)
    g_sizes_tensor = torch.tensor(g_sizes, dtype=torch.int32, device=device)
    g_size_dummy = torch.empty(group_size)

    # kernel
    grouped_gemm_impl(flattened_A, flattened_B, flattened_C, g_sizes_tensor, g_size_dummy)

    # unflatten the output into individual tensors.
    results = []
    offset = 0
    for m, _, n in g_sizes:
        next_offset = offset + m * n
        result = flattened_C[offset:next_offset].reshape(m, n)
        results.append(result)
        offset = next_offset

    return results


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

*No comments yet.*
