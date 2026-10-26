"""Column-reduction kernels and configs shared by the vector-loop sinking tests."""

from __future__ import annotations

from typing import Any
from typing import Callable

import torch

import helion
import helion.language as hl

FRAGMENT_REDUCE = "_cute_grouped_reduce_shared_two_stage_fragment("
TWO_STAGE_REDUCE = "_cute_grouped_reduce_shared_two_stage("


def _col_reduce_sum_fn(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.zeros(n, dtype=x.dtype, device=x.device)
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    for tile_n in hl.tile(n, block_size=block_n):
        col_acc = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m, block_size=block_m):
            col_acc += torch.sum(x[tile_m, tile_n].to(torch.float32), dim=0)
        out[tile_n] = col_acc.to(out.dtype)
    return out


def _col_reduce_max_fn(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty(n, dtype=x.dtype, device=x.device)
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    for tile_n in hl.tile(n, block_size=block_n):
        col_acc = hl.full([tile_n], float("-inf"), dtype=torch.float32)
        for tile_m in hl.tile(m, block_size=block_m):
            col_acc = torch.maximum(
                col_acc, torch.amax(x[tile_m, tile_n].to(torch.float32), dim=0)
            )
        out[tile_n] = col_acc.to(out.dtype)
    return out


def _col_reduce_sum_from8_fn(x: torch.Tensor) -> torch.Tensor:
    """The grid tile starts at column 8: its chunks are not V-aligned and its
    bounds mask differs per element."""
    m, n = x.size()
    out = torch.zeros(n, dtype=x.dtype, device=x.device)
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    for tile_n in hl.tile(8, n, block_size=block_n):
        col_acc = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m, block_size=block_m):
            col_acc += torch.sum(x[tile_m, tile_n].to(torch.float32), dim=0)
        out[tile_n] = col_acc.to(out.dtype)
    return out


def _col_reduce_sum_lower_triangle_fn(x: torch.Tensor) -> torch.Tensor:
    """Each element is loaded under its own row/column condition."""
    m, n = x.size()
    out = torch.zeros(n, dtype=x.dtype, device=x.device)
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    for tile_n in hl.tile(n, block_size=block_n):
        col_acc = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m, block_size=block_m):
            keep = tile_m.index[:, None] >= tile_n.index[None, :]
            vals = hl.load(x, [tile_m, tile_n], extra_mask=keep)
            col_acc += torch.sum(vals.to(torch.float32), dim=0)
        out[tile_n] = col_acc.to(out.dtype)
    return out


def _col_reduce_sum_pair_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Two sibling row loops over the same rows block."""
    m, n = x.size()
    out = torch.zeros(n, dtype=x.dtype, device=x.device)
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    for tile_n in hl.tile(n, block_size=block_n):
        acc_x = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m, block_size=block_m):
            acc_x += torch.sum(x[tile_m, tile_n].to(torch.float32), dim=0)
        acc_y = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m, block_size=block_m):
            acc_y += torch.sum(y[tile_m, tile_n].to(torch.float32), dim=0)
        out[tile_n] = (acc_x + acc_y).to(out.dtype)
    return out


def _col_weighted_mean_fn(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """A row-only load (``w[tile_m]``) and a row-only accumulator (``wsum``)
    next to the column accumulator."""
    m, n = x.size()
    out = torch.zeros(n, dtype=torch.float32, device=x.device)
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    for tile_n in hl.tile(n, block_size=block_n):
        acc = hl.zeros([tile_n], dtype=torch.float32)
        wsum = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m, block_size=block_m):
            wt = w[tile_m].to(torch.float32)
            acc += torch.sum(x[tile_m, tile_n].to(torch.float32) * wt[:, None], dim=0)
            wsum += torch.sum(wt)
        out[tile_n] = acc / wsum
    return out


def _col_reduce_sum_rescaled_fn(
    x: torch.Tensor, scale: torch.Tensor, flag: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A lane-invariant scalar (``s = scale[0]``) used by one store, then
    conditionally redefined by an ``if`` and used again: two definitions of
    ``s`` in one straight-line segment."""
    m, n = x.size()
    out1 = torch.zeros(n, dtype=torch.float32, device=x.device)
    out2 = torch.zeros(n, dtype=torch.float32, device=x.device)
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    for tile_n in hl.tile(n, block_size=block_n):
        acc = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m, block_size=block_m):
            acc += torch.sum(x[tile_m, tile_n].to(torch.float32), dim=0)
        s = scale[0]
        out1[tile_n] = acc * s
        if flag[0] != 0:
            s = s * 2.0
        out2[tile_n] = acc * s
    return out1, out2


def _col_reduce_sum_rescaled_const_fn(
    x: torch.Tensor, flag: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The same with a pure definition (``hl.full([], 1.5)``) instead of a
    load."""
    m, n = x.size()
    out1 = torch.zeros(n, dtype=torch.float32, device=x.device)
    out2 = torch.zeros(n, dtype=torch.float32, device=x.device)
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    for tile_n in hl.tile(n, block_size=block_n):
        acc = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m, block_size=block_m):
            acc += torch.sum(x[tile_m, tile_n].to(torch.float32), dim=0)
        s = hl.full([], 1.5, dtype=torch.float32)
        out1[tile_n] = acc * s
        if flag[0] != 0:
            s = s * 2.0
        out2[tile_n] = acc * s
    return out1, out2


def _col_reduce_sum_atomic_then_load_fn(
    x: torch.Tensor, count: torch.Tensor
) -> torch.Tensor:
    """An atomic whose result is used (``old = hl.atomic_add(...)``) followed
    by a lane-invariant load of the tensor it updates: the load must not run
    before the V-loop that holds the atomic."""
    m, n = x.size()
    out = torch.zeros(n, dtype=torch.float32, device=x.device)
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    for tile_n in hl.tile(n, block_size=block_n):
        old = hl.atomic_add(
            count, [tile_n], hl.full([tile_n], 1.0, dtype=torch.float32)
        )
        first = count[tile_n.id * block_n]
        acc = old * first
        for tile_m in hl.tile(m, block_size=block_m):
            acc += torch.sum(x[tile_m, tile_n].to(torch.float32), dim=0)
        out[tile_n] = acc
    return out


def _col_reduce_sum_gather_then_atomic_fn(
    x: torch.Tensor, count: torch.Tensor
) -> torch.Tensor:
    """A gather of each lane's own element of ``count`` (a subscript read
    with no load call), then an atomic bump of that element, with the
    gathered value used after the row loop: it must stay the value read
    before the atomic."""
    m, n = x.size()
    out = torch.zeros(n, dtype=torch.float32, device=x.device)
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    for tile_n in hl.tile(n, block_size=block_n):
        before = torch.gather(count[tile_n], 0, tile_n.index.to(torch.int64))
        hl.atomic_add(count, [tile_n], hl.full([tile_n], 1.0, dtype=torch.float32))
        acc = hl.zeros([tile_n], dtype=torch.float32)
        for tile_m in hl.tile(m, block_size=block_m):
            acc += torch.sum(x[tile_m, tile_n].to(torch.float32), dim=0)
        out[tile_n] = acc + before
    return out


def _kernel(fn: Callable[..., Any], *, static_shapes: bool) -> helion.Kernel:
    return helion.kernel(
        fn, backend="cute", static_shapes=static_shapes, autotune_effort="none"
    )


col_reduce_sum_static = _kernel(_col_reduce_sum_fn, static_shapes=True)
col_reduce_sum_dynamic = _kernel(_col_reduce_sum_fn, static_shapes=False)
col_reduce_max_dynamic = _kernel(_col_reduce_max_fn, static_shapes=False)
col_reduce_sum_from8_static = _kernel(_col_reduce_sum_from8_fn, static_shapes=True)
col_reduce_sum_from8_dynamic = _kernel(_col_reduce_sum_from8_fn, static_shapes=False)
col_reduce_sum_lower_triangle_static = _kernel(
    _col_reduce_sum_lower_triangle_fn, static_shapes=True
)
col_reduce_sum_pair_static = _kernel(_col_reduce_sum_pair_fn, static_shapes=True)
col_reduce_sum_pair_dynamic = _kernel(_col_reduce_sum_pair_fn, static_shapes=False)
col_weighted_mean_static = _kernel(_col_weighted_mean_fn, static_shapes=True)
col_weighted_mean_dynamic = _kernel(_col_weighted_mean_fn, static_shapes=False)
col_reduce_sum_rescaled_static = _kernel(
    _col_reduce_sum_rescaled_fn, static_shapes=True
)
col_reduce_sum_rescaled_dynamic = _kernel(
    _col_reduce_sum_rescaled_fn, static_shapes=False
)
col_reduce_sum_rescaled_const_static = _kernel(
    _col_reduce_sum_rescaled_const_fn, static_shapes=True
)
col_reduce_sum_atomic_then_load_static = _kernel(
    _col_reduce_sum_atomic_then_load_fn, static_shapes=True
)
col_reduce_sum_gather_then_atomic_static = _kernel(
    _col_reduce_sum_gather_then_atomic_fn, static_shapes=True
)


def _sink_config(
    *,
    block_sizes: list[int],
    num_threads: list[int],
    vec: list[int],
    layouts: list[str] | None = None,
    sink: bool = True,
    unroll: int = 1,
) -> dict[str, object]:
    return {
        "block_sizes": block_sizes,
        "num_threads": num_threads,
        "cute_vector_widths": vec,
        "cute_lane_layouts": layouts or ["strided", "blocked"],
        "cute_vloop_sink": sink,
        "cute_lane_unroll": unroll,
    }
