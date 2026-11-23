"""better-benchmark repro pointwise_f7babc5c978c (torchbench nvidia_deeprecommender).

Inductor emits one pointwise kernel,
``triton_poi_fused_convert_element_type_slice_0``: the graph's
``x[:, 0:-1].to(float32)``, a bf16 ``[rows, cols]`` activation without its last
column widened to a new contiguous fp32 ``[rows, cols - 1]`` buffer (one bf16
load and one fp32 store per element, no arithmetic). The slice is free: Inductor
indexes the output as one flat dimension of ``rows * (cols - 1)`` elements and
loads element ``index % (cols - 1) + cols * (index // (cols - 1))`` of the
input's storage, so the dropped column is never read. The port takes the same
unsliced input and tiles the output's two dimensions, reading the input at the
same ``[row, col]``.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def slice_to_fp32(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    kept = cols - 1  # the graph's slice(x, 1, 0, -1)
    out = torch.empty([rows, kept], dtype=torch.float32, device=x.device)
    for tile_r, tile_c in hl.tile([rows, kept]):
        out[tile_r, tile_c] = x[tile_r, tile_c].to(torch.float32)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1 = inputs[0]
    out = slice_to_fp32(arg0_1)
    return (out,)
