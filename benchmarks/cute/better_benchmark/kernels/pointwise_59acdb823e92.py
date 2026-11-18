"""better-benchmark repro pointwise_59acdb823e92 (hf/infer openai/gpt-oss-20b).

The routing postprocess of a mixture-of-experts layer: the top 4 of each
token's 32 bf16 router logits, the flattened expert ids sorted (with the sort
permutation), a ``>= 32`` mask and an int32 copy of the sorted ids, and the
bf16 hidden rows gathered in sorted order (row ``permutation // 4``, zeroed
where the mask is set). Inductor emits three kernels around one ATen call:

- ``triton_per_fused_topk_0``: Inductor's ``topk`` decomposition, a persistent
  reduction over the 32 logits of a row. The bf16 row is widened to fp32 (one
  load), NaNs are canonicalized, and the fp32 bits become an int32 key whose
  integer order is the floats' total order (``bits ^ 0x7fffffff`` for negative
  floats, NaN largest). Four first-index argmaxes follow, each after masking
  the previous winner's column to ``INT32_MIN``. The four indices are stored
  to the columns of an int64 ``[rows, 4]`` buffer and the four logits they
  select, gathered from the input again, to the columns of the bf16
  ``[rows, 4]`` values output: eight stores per row.
- ``torch.ops.aten.sort.stable(indices.view(-1), stable=False, ...)``, called
  by Inductor's wrapper between its kernels (Inductor has no Triton sort).
  The port makes the same ATen call (``torch.sort``) at the same place: it is
  the wrapper's own library call, not work moved out of a kernel, and a
  Helion kernel cannot express it on both backends (PORTING.md, "Same
  kernels, same traffic"). Helion's ``torch.sort`` compares every pair of
  elements: for the 4000 ids the Triton backend does not compile it (a 4000
  x 4000 tensor is past Triton's size limit), and the CuTe backend would
  put quadratic work where Inductor's wrapper makes the library call.
- ``triton_poi_fused_convert_element_type_ge_1``: a pointwise kernel over the
  sorted ids, storing ``id >= 32`` (bool) and ``int32(id)``.
- ``triton_poi_fused_div_full_ge_index_unsqueeze_where_2``: a pointwise
  kernel over the bf16 ``[rows * 4, hidden]`` output. Each element loads its
  row's mask and sort position, floor-divides the position by 4, wraps a
  negative source row (``aten.index``), gathers the bf16 hidden element,
  widens it to fp32, replaces it by ``0.0`` where the mask is set and stores
  it in bf16. The port tiles the output's two dimensions (Inductor indexes
  it as one flat dimension and recovers the row and column with a division
  and a remainder).

Ties: Inductor's decomposition returns tied logits in index order (first
index first), while eager's CUDA ``topk`` returns them in an
implementation-defined order. The repro's bf16 randn rows have ties among
their top four (20 of the 1000 rows are ordered differently by eager), so
Inductor's sort permutation, and this port's, differs from eager's in the
positions of those ties (40 of 4000 elements); every other output is equal.
The checks therefore compare the permutation with Inductor's compiled
output, which it must equal exactly (``common.check_outputs``).
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def topk4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # x: [rows, n] bf16, contiguous.
    rows, n = x.shape
    flat = x.view(-1)
    values = torch.empty([rows, 4], dtype=x.dtype, device=x.device)
    indices = torch.empty([rows, 4], dtype=torch.int64, device=x.device)
    for tile_r in hl.tile(rows):
        v = x[tile_r, :].to(torch.float32)
        v = torch.where(torch.isnan(v), float("nan"), v)
        # The fp32 bits as a key whose int32 order is the floats' order.
        bits = v.view(torch.int32)
        key = torch.where(bits < 0, bits ^ 2147483647, bits)
        cols = hl.arange(v.size(1), dtype=torch.int64)

        idx0 = torch.argmax(key, dim=1)
        key = torch.where(cols[None, :] == idx0[:, None], -2147483648, key)
        idx1 = torch.argmax(key, dim=1)
        key = torch.where(cols[None, :] == idx1[:, None], -2147483648, key)
        idx2 = torch.argmax(key, dim=1)
        key = torch.where(cols[None, :] == idx2[:, None], -2147483648, key)
        idx3 = torch.argmax(key, dim=1)

        indices[tile_r, 0] = idx0
        indices[tile_r, 1] = idx1
        indices[tile_r, 2] = idx2
        indices[tile_r, 3] = idx3
        # gather(x, 1, idx): one more load per selected logit, at the flat
        # offset Inductor uses (idx + n * row). x[tile_r, idx] would index the
        # two as an outer product, not row by row.
        base = tile_r.index * n
        values[tile_r, 0] = flat[base + idx0].to(torch.float32).to(values.dtype)
        values[tile_r, 1] = flat[base + idx1].to(torch.float32).to(values.dtype)
        values[tile_r, 2] = flat[base + idx2].to(torch.float32).to(values.dtype)
        values[tile_r, 3] = flat[base + idx3].to(torch.float32).to(values.dtype)
    return values, indices


@helion.kernel(static_shapes=True)
def invalid_mask_and_int32(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # ids: [n] int64.
    n = ids.size(0)
    invalid = torch.empty([n], dtype=torch.bool, device=ids.device)
    ids32 = torch.empty([n], dtype=torch.int32, device=ids.device)
    for tile_n in hl.tile(n):
        v = ids[tile_n]
        invalid[tile_n] = v >= 32
        ids32[tile_n] = v.to(torch.int32)
    return invalid, ids32


@helion.kernel(static_shapes=True)
def gather_sorted_rows(
    invalid: torch.Tensor, positions: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    # invalid: [n] bool; positions: [n] int64; x: [rows, hidden] bf16.
    n = positions.size(0)
    rows, hidden = x.shape
    out = torch.empty([n, hidden], dtype=x.dtype, device=x.device)
    for tile_n, tile_h in hl.tile([n, hidden]):
        # index(x, [div(positions, 4, rounding_mode="floor")])
        src = positions[tile_n] // 4
        src = torch.where(src < 0, src + rows, src)
        v = x[src, tile_h].to(torch.float32)
        masked = torch.where(invalid[tile_n][:, None], 0.0, v)
        out[tile_n, tile_h] = masked.to(out.dtype)
    return out


def forward(inputs: list) -> tuple[torch.Tensor, ...]:
    arg0_1, arg1_1 = inputs[:2]
    values, indices = topk4(arg0_1)
    # Inductor's wrapper calls ATen here too (aten.sort.stable, stable=False).
    sorted_ids, positions = torch.sort(indices.view(-1))
    invalid, ids32 = invalid_mask_and_int32(sorted_ids)
    gathered = gather_sorted_rows(invalid, positions, arg1_1)
    return (
        values,
        sorted_ids,
        positions,
        invalid.view(-1, 1),
        gathered,
        ids32,
    )
