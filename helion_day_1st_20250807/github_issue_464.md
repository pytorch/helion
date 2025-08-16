# Issue #464: numerical error with HELION_INTERPRET=1

## Metadata
- **State**: OPEN
- **Author**: [pianpwk](https://github.com/pianpwk)
- **Created**: August 08, 2025 at 01:19 UTC
- **Updated**: August 08, 2025 at 01:28 UTC

## Description

This jagged softmax script produces matching outputs in normal mode, but has mismatching values with `HELION_INTERPRET=1`
```python
import re
import torch

import helion
from helion._testing import run_example
import helion.language as hl
from torch._C import device


def reference_jagged_softmax(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
) -> torch.Tensor:
    vals = []
    for i, j in zip(x_offsets[:-1], x_offsets[1:]):
        y = x_data[i:j]
        vals.append(torch.softmax(y, dim=0))
    return torch.cat(vals, dim=0)


@helion.kernel()
def jagged_softmax_kernel(
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

            # 1.
            block_max = hl.full([tile_b, tile_m], float("-inf"), dtype=x_data.dtype)
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

                block_max = torch.maximum(block_max, x_slice.amax(dim=1))
            block_max = block_max[:, None, :]

            # 2.
            block_sumexp = hl.zeros([tile_b, tile_m], dtype=x_data.dtype)
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
                exp = torch.exp(x_slice - block_max)
                block_sumexp = torch.where(row_mask[:, :, None], exp, 0.0).sum(dim=1)
            block_sumexp = block_sumexp[:, None, :]

            # 3.
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
                
                block_out = torch.exp(x_slice - block_max) / block_sumexp
                out_indices = starts[:, None, None] * M + tile_k.index[None, :, None] * M + tile_m.index[None, None, :]
                out_mask = out_indices < N * M
                out_indices = torch.where(out_indices >= N * M, 0, out_indices)
                hl.store(
                    out,
                    [out_indices],
                    block_out,
                    extra_mask=out_mask,
                )

    out = out.reshape([N, M])
    return out


if __name__ == "__main__":
    x_data = torch.randn(13, 4).cuda()
    x_offsets = torch.tensor([0, 6, 8, 12]).cuda()
    
    out_eager = reference_jagged_softmax(x_data, x_offsets)
    # out_explicit = explicit_reference_jagged_softmax(x_data, x_offsets, 2)
    print(out_eager)
    # print(out_explicit)

    out_hl = jagged_softmax_kernel(x_data, x_offsets)
    print(out_hl)
```

output (in interpret mode):
```
tensor([[0.1098, 0.1023, 0.0854, 0.3640],
        [0.1477, 0.0294, 0.0533, 0.3492],
        [0.0841, 0.0511, 0.0823, 0.0909],
        [0.4831, 0.2017, 0.0875, 0.0515],
        [0.0320, 0.4493, 0.6637, 0.1218],
        [0.1433, 0.1663, 0.0278, 0.0226],
        [0.6483, 0.2705, 0.1160, 0.3992],
        [0.3517, 0.7295, 0.8840, 0.6008],
        [0.1083, 0.1481, 0.1833, 0.1722],
        [0.1755, 0.5477, 0.0600, 0.3668],
        [0.5966, 0.1792, 0.6298, 0.4478],
        [0.1195, 0.1250, 0.1269, 0.0131]], device='cuda:0')
tensor([[0.0000, 0.1023, 0.0854, 0.3640],
        [0.1477, 0.0294, 0.0533, 0.3492],
        [0.0841, 0.0511, 0.0823, 0.0909],
        [0.4831, 0.2017, 0.0875, 0.0515],
        [0.0320, 0.4493, 0.6637, 0.1218],
        [0.1433, 0.1663, 0.0278, 0.0226],
        [0.6483, 0.2705, 0.1160, 0.3992],
        [0.3517, 0.7295, 0.8840, 0.6008],
        [0.2917, 2.1171, 0.4209, 0.2798],
        [0.2917, 2.1171, 0.4209, 0.2798],
        [0.2917, 2.1171, 0.4209, 0.2798],
        [0.2917, 2.1171, 0.4209, 0.2798]], device='cuda:0')
```

torch: '2.9.0a0+git21392c0'
helion: helion@334095fbd68555506f46c7adb52db654178fffe3

## Comments

*No comments yet.*
