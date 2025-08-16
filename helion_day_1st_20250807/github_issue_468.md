# Issue #468: better error message: rank mismatch in control flow: 3 != 2

## Metadata
- **State**: CLOSED
- **Author**: [pianpwk](https://github.com/pianpwk)
- **Created**: August 08, 2025 at 05:28 UTC
- **Updated**: August 15, 2025 at 22:02 UTC
- **Closed**: August 15, 2025 at 22:02 UTC
- **Assignees**: [jansel](https://github.com/jansel)

## Description

Probably I'm doing something wrong, but there's no error with `HELION_INTERPRET=1`, and I can't tell what the issue is from the error message (jagged softmax):
```python
import re
import torch

import helion
from helion._testing import run_example
import helion.language as hl
from torch._C import device

@helion.kernel()
def jagged_softmax_kernel_2loop(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
) -> torch.Tensor:

    N = x_offsets[-1].item()
    num_rows, M = x_offsets.size(0) - 1, x_data.size(1)
    out = torch.zeros([N * M], dtype=x_data.dtype, device=x_data.device)
    
    # flatten
    x_flat = x_data.view(-1)

    for tile_b in hl.tile(num_rows):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        seqlens = ends - starts
        max_seqlen = seqlens.amax()

        for tile_m in hl.tile(M):

            block_max = hl.full([tile_b, tile_m], 0.0, dtype=x_data.dtype)
            block_new_max = hl.full([tile_b, tile_m], 0.0, dtype=x_data.dtype)
            block_L = hl.full([tile_b, tile_m], 0.0, dtype=x_data.dtype)

            for tile_k in hl.tile(max_seqlen):
                base_indices = starts[:, None] + tile_k.index[None, :]
                flat_indices = base_indices[:, :, None] * M + tile_m.index[None, None, :]
                row_mask = tile_k.index[None, :] < seqlens[:, None]
                combined_mask = row_mask[:, :, None] & (tile_m.index < M)[None, None, :]
                x_slice = hl.load(
                    x_flat,
                    [flat_indices],
                    extra_mask=combined_mask,
                )
                slice_max = torch.where(combined_mask, x_slice, float("-inf")).amax(dim=1)
                block_new_max = torch.maximum(block_max, slice_max)
                block_L *= torch.exp(block_new_max - block_max)
                block_L += torch.where(row_mask[:, :, None], torch.exp(x_slice - block_new_max[:, None, :]), 0.0).sum(dim=1)
                block_max = block_new_max
            block_max = block_max[:, None, :]
            block_L = block_L[:, None, :]

            for tile_k in hl.tile(max_seqlen):
                base_indices = starts[:, None] + tile_k.index[None, :]
                flat_indices = base_indices[:, :, None] * M + tile_m.index[None, None, :]
                row_mask = tile_k.index[None, :] < seqlens[:, None]
                combined_mask = row_mask[:, :, None] & (tile_m.index < M)[None, None, :]
                x_slice = hl.load(
                    x_flat,
                    [flat_indices],
                    extra_mask=combined_mask,
                )
                x_exp = torch.exp(x_slice)
                z = x_exp / torch.exp(torch.log(block_L) + block_max)
                block_out = torch.where(
                    combined_mask,
                    z,
                    0.0,
                )
                out_indices = torch.where(flat_indices >= N * M, 0, flat_indices)
                hl.store(
                    out,
                    [out_indices],
                    block_out,
                    extra_mask=combined_mask,
                )

    out = out.reshape([N, M])
    return out


if __name__ == "__main__":
    num_rows, max_cols = 1024, 128
    device = "cuda"

    lengths = torch.randint(1, max_cols + 1, (num_rows,), device=device)
    x_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(lengths, dim=0)]
    )
    nnz = int(x_offsets[-1])
    M = 8  # number of features
    x_data = torch.randn(nnz, M, dtype=torch.float32, device=device)
    
    out_hl = jagged_softmax_kernel_2loop(x_data, x_offsets)
```

error:
```
Traceback (most recent call last):
  File "/data/users/pianpwk/helion/examples/jagged_softmax.py", line 226, in <module>
    out_hl = jagged_softmax_kernel_2loop(x_data, x_offsets)
  File "/data/users/pianpwk/helion/helion/runtime/kernel.py", line 272, in __call__
    return self.bind(args)(*args)
  File "/data/users/pianpwk/helion/helion/runtime/kernel.py", line 158, in bind
    bound_kernel = BoundKernel(self, args)
  File "/data/users/pianpwk/helion/helion/runtime/kernel.py", line 338, in __init__
    self.host_function: HostFunction = HostFunction(
  File "/data/users/pianpwk/helion/helion/_compiler/host_function.py", line 108, in __init__
    propagate_types(self, fake_args)
  File "/data/users/pianpwk/helion/helion/_compiler/type_propagation.py", line 2252, in propagate_types
    prop.visit(stmt)
  File "/data/users/pianpwk/helion/helion/_compiler/type_propagation.py", line 1581, in visit
    type_info = visitor(node)
  File "/data/users/pianpwk/helion/helion/_compiler/type_propagation.py", line 2087, in visit_For
    body = self._loop_body(node.body)
  File "/data/users/pianpwk/helion/helion/_compiler/type_propagation.py", line 2051, in _loop_body
    self.visit(stmt)
  File "/data/users/pianpwk/helion/helion/_compiler/type_propagation.py", line 1581, in visit
    type_info = visitor(node)
  File "/data/users/pianpwk/helion/helion/_compiler/type_propagation.py", line 2092, in visit_For
    self.scope.merge_if_else(body, orelse)
  File "/data/users/pianpwk/helion/helion/_compiler/type_propagation.py", line 158, in merge_if_else
    self.merge(true)
  File "/data/users/pianpwk/helion/helion/_compiler/type_propagation.py", line 143, in merge
    merged = existing.merge(v)
  File "/data/users/pianpwk/helion/helion/_compiler/type_propagation.py", line 526, in merge
    raise exc.TypeInferenceError(
helion.exc.TypeInferenceError: rank mismatch in control flow: 3 != 2
While processing:
  File "/data/users/pianpwk/helion/examples/jagged_softmax.py", line 151, in jagged_softmax_kernel_2loop
    for tile_m in hl.tile(M):
```

## Comments

### Comment 1 by [jansel](https://github.com/jansel)
*Posted on August 15, 2025 at 14:38 UTC*

After #501 the error is:
```py
  File "/home/jansel/helion/helion/_compiler/type_propagation.py", line 143, in merge
    merged = existing.merge(v, var_name=k)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jansel/helion/helion/_compiler/type_propagation.py", line 529, in merge
    raise exc.ControlFlowTensorMismatch(
helion.exc.ControlFlowTensorMismatch: Tensor mismatch in control flow for variable 'block_max': rank 3 != 2
Hint: ensure the same tensor rank/shape/dtype/device for this variable across branches/iterations.
While processing:
  File "/home/jansel/helion/scratch.py", line 28, in jagged_softmax_kernel_2loop
```

---
