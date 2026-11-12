"""The masks the device IR keeps before a reduction of a loop carry's lanes
(``node_masking._reduces_stale_carry_lanes``), counted in kernels bound on
CPU.

A carry with a dim on its own loop's block holds earlier iterations' values
in the padded lanes of a partial tile, as do the loop's results after it, so
a sum over that block keeps the mask before it wherever such lanes reach it:
in the carry's loop, in a later loop over the same block, and through the
results of the loops and branches between them.  test_masking.py runs the
kernels and checks their values.
"""

from __future__ import annotations

import os

import pytest
import torch

import helion
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._testing import cpu_render_target
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl
from helion.language._tracing_ops import _mask_to

pytestmark = [
    pytest.mark.cpu_only,
    skipUnlessBackends(["triton", "cute"]),
    pytest.mark.skipif(
        os.environ.get("HELION_INTERPRET") == "1",
        reason="inspects the device IR; ref mode has none",
    ),
]


def _masks(kernel: helion.Kernel, *args: object) -> list[tuple[list[int], int]]:
    """The block ids of each device loop body of ``kernel`` bound for
    ``args`` on CPU and its number of ``_mask_to`` nodes, in graph order (an
    inner loop's body before its outer loop's)."""
    with cpu_render_target():
        bound = cpu_bind(kernel, args)
    return [
        (
            info.block_ids,
            len(info.graph.find_nodes(op="call_function", target=_mask_to)),
        )
        for info in bound.host_function.device_ir.graphs
        if isinstance(info, ForLoopGraphInfo)
    ]


@helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
def flat_carry(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    block_n = hl.register_block_size(n)
    out = torch.empty([m], device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n]).flatten()
        total = hl.zeros([tile_m])
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + x[tile_m, tile_n].flatten()
            total = total + acc.view(tile_m, block_n).sum(dim=1)
        out[tile_m] = total
    return out


@helion.kernel(config={"block_sizes": [16, 4]}, static_shapes=True)
def flat_later_loop(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    block_n = hl.register_block_size(n)
    out = torch.empty([m], device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n]).flatten()
        for tile_n in hl.tile(n, block_size=block_n):
            acc = acc + x[tile_m, tile_n].flatten()
        total = hl.zeros([tile_m])
        for tile_n in hl.tile(n, block_size=block_n):
            rows = acc.view(tile_m, block_n) + x[tile_m, tile_n]
            total = total + rows.sum(dim=1)
        out[tile_m] = total
    return out


# ``acc.flatten()`` holds the carry's stale lanes in its one ``tile_m * bn``
# dim: summed viewed back in its loop or in a later loop over the same block,
# it keeps the mask (neither backend runs such a carry yet).
@pytest.mark.parametrize("kernel", [flat_carry, flat_later_loop])
def test_a_flattened_carry_keeps_its_mask(kernel: helion.Kernel) -> None:
    assert sum(count for _, count in _masks(kernel, torch.ones([64, 20]))) == 1


@helion.kernel(config={"block_sizes": [16, 4, 8]}, static_shapes=True)
def read_only(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    block_n = hl.register_block_size(n)
    out = torch.empty([m], device=x.device)
    for tile_m in hl.tile(m):
        total = hl.zeros([tile_m])
        for tile_n in hl.tile(n, block_size=block_n):
            row = x[tile_m, tile_n]
            for _tile_k in hl.tile(y.size(0)):
                total = total + row.sum(dim=1)
        out[tile_m] = total
    return out


@helion.kernel(config={"block_sizes": [16, 4, 8]}, static_shapes=True)
def fresh_carry(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    block_n = hl.register_block_size(n)
    out = torch.empty([m], device=x.device)
    for tile_m in hl.tile(m):
        total = hl.zeros([tile_m])
        for tile_n in hl.tile(n, block_size=block_n):
            inner = hl.zeros([tile_m, block_n])
            for _tile_k in hl.tile(y.size(0)):
                inner = inner + x[tile_m, tile_n]
                total = total + inner.sum(dim=1)
        out[tile_m] = total
    return out


# An inner loop's input it only reads, and its own carry started in the outer
# loop, hold no lanes an earlier iteration of the outer loop left: their sums
# over the outer loop's block keep no mask.
@pytest.mark.parametrize("kernel", [read_only, fresh_carry])
def test_inner_loop_values_without_earlier_lanes_need_no_mask(
    kernel: helion.Kernel,
) -> None:
    masks = _masks(kernel, torch.ones([64, 20]), torch.ones([16]))
    assert sum(count for _, count in masks) == 0


@helion.kernel(config={"block_sizes": [16, 4, 4]}, static_shapes=True)
def summed_in_an_enclosing_graph(w: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    nx, ny = w.size()
    out = torch.empty([nx], device=w.device)
    block_y = hl.register_block_size(ny)
    for tile_g in hl.tile(nx):
        total = hl.zeros([tile_g])
        for tile_x in hl.tile(nx):
            c = hl.zeros([block_y, tile_x])
            for tile_y in hl.tile(ny, block_size=block_y):
                c = c + w[tile_x, tile_y].T
            r = c.sum(1)
            for tile_y in hl.tile(ny, block_size=block_y):
                total = total + (r + v[tile_y]).sum(0)
        out[tile_g] = total
    return out


@helion.kernel(config={"block_sizes": [16, 64, 1]}, static_shapes=True)
def from_a_nested_loop(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    block_n = hl.register_block_size(n)
    out = torch.empty([m], device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n])
        for _tile_k in hl.tile(y.size(0)):
            acc = hl.zeros([tile_m, block_n])
            for tile_n in hl.tile(n, block_size=block_n):
                acc = acc + x[tile_m, tile_n]
        total = hl.zeros([tile_m])
        for tile_n in hl.tile(n, block_size=block_n):
            total = total + (acc + x[tile_m, tile_n]).sum(dim=1)
        out[tile_m] = total
    return out


@helion.kernel(config={"block_sizes": [16, 64]}, static_shapes=True)
def from_branches(x: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    block_n = hl.register_block_size(n)
    out = torch.empty([m], device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, block_n])
        if flag[0] > 0:
            for tile_n in hl.tile(n, block_size=block_n):
                acc = acc + x[tile_m, tile_n]
        else:
            for tile_n in hl.tile(n, block_size=block_n):
                acc = acc + 2 * x[tile_m, tile_n]
        total = hl.zeros([tile_m])
        for tile_n in hl.tile(n, block_size=block_n):
            total = total + (acc + x[tile_m, tile_n]).sum(dim=1)
        out[tile_m] = total
    return out


# A value with a loop's stale lanes along a block reaches the sum over that
# block in the last loop: through ``c.sum(1)``, a reduction over another block
# in an enclosing graph, whose mask was decided without the loops over the
# summed one; out of an outer loop that reruns the carry's loop each
# iteration; and out of either branch of an ``if``.  The last loop over that
# block (block 0, the registered one) keeps the mask before its sum.
@pytest.mark.parametrize(
    ("kernel", "args"),
    [
        (summed_in_an_enclosing_graph, (torch.ones([8, 20]), torch.ones([20]))),
        (from_a_nested_loop, (torch.ones([64, 20]), torch.ones([3]))),
        (from_branches, (torch.ones([64, 20]), torch.ones([1], dtype=torch.int32))),
    ],
    ids=["enclosing-reduction", "nested-loop", "branches"],
)
def test_a_later_loop_keeps_its_mask_for_stale_lanes_of_any_nesting(
    kernel: helion.Kernel, args: tuple[torch.Tensor, ...]
) -> None:
    later = [count for blocks, count in _masks(kernel, *args) if blocks == [0]][-1]
    assert later == 1
