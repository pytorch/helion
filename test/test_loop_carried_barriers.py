"""Barriers before loads that a store of an earlier loop iteration reaches.

A store late in one iteration of a device loop and a load early in the next
are a read-after-write across threads like a store and a later load in one
body: the load gets the barrier.  A storage every access of which in the loop
goes through the same view, whose strides map distinct elements to distinct
memory, at indices that tell the iterations apart along every axis of the
loop by one same item at one same position (a tile or grid index), or at one
same index whose full slice a rolled reduction loop maps to its chunk, is
touched in disjoint tiles by different iterations and needs none.  The
kernels render on CPU (TileIR emits no barriers, and ref-eager mode generates
no code).
"""

from __future__ import annotations

import functools
import itertools

import pytest
import torch

import helion
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._compiler.loop_dependency_checker import intra_loop_raw_reread_block_ids
from helion._testing import cpu_render_target
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessTritonBarriers
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _reads_what_the_last_iteration_stored(
    x: torch.Tensor, scratch: torch.Tensor
) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            # The first 16 columns, which the first iteration stores.
            first = scratch[tile_m, 0:16].sum(-1, keepdim=True)
            out[tile_m, tile_n] = first + x[tile_m, tile_n]
            scratch[tile_m, tile_n] = x[tile_m, tile_n]
    return out


def _reads_the_previous_row_reversed(
    x: torch.Tensor, scratch: torch.Tensor
) -> torch.Tensor:
    for _ in hl.tile(1, block_size=1):
        for step in hl.grid(4):
            # The previous iteration stored this row, other threads each
            # element; this iteration stores the next row.
            value = scratch[step, 63 - hl.arange(64)]
            scratch[step + 1, hl.arange(64)] = value + 1
    return scratch


def _writes_a_shifted_view(x: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    shifted = scratch[:, 1:]
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n - 1):
            # The next iteration reads the column this one writes last.
            shifted[tile_m, tile_n] = scratch[tile_m, tile_n] + 1
    return scratch


def _updates_an_overlapping_argument(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    for _ in hl.tile(1, block_size=1):
        for tile in hl.tile(a.size(0)):
            # ``b`` may be ``a`` shifted: the next iteration loads what this
            # one stores.
            b[tile] = a[tile] + 1
    return b


def _writes_a_transposed_view(x: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
    m = scratch.size(0)
    columns = scratch.transpose(0, 1)
    for _ in hl.tile(1, block_size=1):
        for rows in hl.tile(m):
            # Later iterations read rows across the columns this one writes.
            columns[rows, 0:m] = scratch[rows, 0:m] + 1
    return scratch


def _updates_an_overlapping_view(
    x: torch.Tensor, scratch: torch.Tensor
) -> torch.Tensor:
    windows = scratch.view(-1).unfold(0, 64, 1)
    for _ in hl.tile(1, block_size=1):
        for row in hl.tile(16, block_size=1):
            # Row ``r + 1`` overlaps 63 of the elements row ``r`` writes.
            windows[row, :] = windows[row, :] + 1
    return scratch


def _updates_a_broadcast_view(x: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
    rows = scratch[0:1].expand(16, 64)
    for _ in hl.tile(1, block_size=1):
        for row in hl.tile(16, block_size=1):
            # Every row is the same memory.
            rows[row, :] = rows[row, :] + 1
    return scratch


def _updates_one_axis_of_a_2d_loop(
    x: torch.Tensor, scratch: torch.Tensor
) -> torch.Tensor:
    m, n = x.shape
    for tile_m in hl.tile(m):
        for tile_n, _tile_k in hl.tile([n, 2]):
            # Iterations ``(n, k0)`` and ``(n, k1)`` update the same elements.
            scratch[tile_m, tile_n] = scratch[tile_m, tile_n] + x[tile_m, tile_n]
    return scratch


def _updates_its_own_2d_tile(x: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    for _ in hl.tile(1, block_size=1):
        for tile_m, tile_n in hl.tile([m, n]):
            scratch[tile_m, tile_n] = scratch[tile_m, tile_n] + x[tile_m, tile_n]
    return scratch


def _updates_its_own_row(x: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    for tile_n in hl.tile(n):
        for i in hl.grid(m):
            scratch[i, tile_n] = scratch[i, tile_n] + x[i, tile_n]
    return scratch


def _reads_a_reinterpreted_view(x: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
    flat = scratch.view(-1)
    halves = flat.view(torch.bfloat16)[:64]
    head = flat[:64]
    for _ in hl.tile(1, block_size=1):
        for t in hl.tile(64, block_size=16):
            # Same sizes, strides and offset, but in elements half as wide:
            # iteration 1 reads bytes iteration 0 stored.
            head[t] = halves[t].to(torch.float32) + 1
    return scratch


def _rereads_rows_sized_like_its_tile(
    x: torch.Tensor, scratch: torch.Tensor
) -> torch.Tensor:
    m, n = x.shape
    bm = hl.register_block_size(m)
    rows = torch.zeros([bm, n], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(n):
        for tile_m in hl.tile(m, block_size=bm):
            # Every iteration reads and writes rows 0:bm: sized like the tile,
            # not indexed by it.
            prev = rows[:, tile_n]
            rows[:, tile_n] = x[tile_m, tile_n] + prev.sum(0, keepdim=True)
    return scratch


def _updates_its_own_tile(x: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            scratch[tile_m, tile_n] = scratch[tile_m, tile_n] + x[tile_m, tile_n]
    return scratch


def _updates_its_own_tile_under_an_if(
    x: torch.Tensor, scratch: torch.Tensor, scale: float
) -> torch.Tensor:
    m, n = x.shape
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            value = x[tile_m, tile_n]
            if scale > 0:
                value = value + scratch[tile_m, tile_n]
            scratch[tile_m, tile_n] = value
    return scratch


def _updates_its_own_columns_under_an_if(
    x: torch.Tensor, scratch: torch.Tensor, scale: float
) -> torch.Tensor:
    m, n = x.shape
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            # The ``if`` reads a copy of ``columns`` (``_new_var``).
            columns = tile_n.index
            if scale > 0:
                value = scratch[tile_m, columns] * scale
                scratch[tile_m, columns] = value + value.sum(-1, keepdim=True)
    return scratch


def _shifts_its_tile_in_an_inner_loop(
    x: torch.Tensor, scratch: torch.Tensor
) -> torch.Tensor:
    m, n = x.shape
    for tile_m in hl.tile(m, block_size=1):
        for tile_n in hl.tile(n // 2, block_size=16):
            columns = tile_n.index + n // 4
            value = scratch[tile_m, columns]
            for _ in hl.tile(2, block_size=1):
                # The second pass stores the next iteration's tile.
                scratch[tile_m, columns] = value + 1.0
                columns = columns + 16
    return scratch


def _stages_its_rows_in_two_inner_loops(
    x: torch.Tensor, scratch: torch.Tensor
) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_b in hl.tile(m, block_size=32):
        for tile_m in hl.tile(tile_b.begin, tile_b.end):
            # Two tiles of columns, but each iteration's own rows.
            for tile_n in hl.tile(n):
                scratch[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
            for tile_n in hl.tile(n):
                out[tile_m, tile_n] = scratch[tile_m, tile_n] + 1.0
    return out


def _scales_its_columns_in_place(
    x: torch.Tensor, scratch: torch.Tensor
) -> torch.Tensor:
    out = torch.empty([x.size(1)], dtype=x.dtype, device=x.device)
    for tile_n in hl.tile(x.size(1)):
        # A rolled reduction loop maps ``:`` to its chunk.
        columns = scratch[:, tile_n]
        scratch[:, tile_n] = columns * 2.0
        out[tile_n] = columns.sum(0)
    return out


def _flushes_an_earlier_write_once(
    x: torch.Tensor, a: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        c[tile_m, :] = x[tile_m, :]
        for tile_n in hl.tile(n):
            # The barrier the previous iteration's store to ``a`` needs here
            # also flushes the store to ``c`` before the loop.
            first = a[tile_m, 0:16].sum(-1, keepdim=True)
            out[tile_m, tile_n] = first + c[tile_m, tile_n]
            a[tile_m, tile_n] = x[tile_m, tile_n]
    return out


def _rewrites_a_size_one_row(
    x: torch.Tensor, scratch: torch.Tensor, row: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        for i in hl.grid(4):
            # ``row`` has one row: indexing drops ``i``, so every iteration
            # rewrites the same elements.
            row[i, :] = row[i, :] + 1
        out[tile_m, :] = x[tile_m, :]
    return out


def _counts_up_in_a_while_loop(x: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        while flag[0, 0] < 4:
            flag[0, 0] = flag[0, 0] + 1
        out[tile_m, :] = x[tile_m, :]
    return out


@functools.cache
def _bind(fn: object, *extra: object, static_shapes: bool = True) -> object:
    # Once per module: the tests only read the bound kernel's device IR.
    with cpu_render_target():
        return cpu_bind(
            helion.kernel(fn, static_shapes=static_shapes),
            (torch.randn(32, 64), torch.zeros(32, 64), *extra),
        )


def _barriered_loads(code: str) -> int:
    """The loads that follow a barrier directly."""
    lines = [
        line.strip()
        for line in code.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return sum(
        line == "tl.debug_barrier()" and "tl.load(" in after
        for line, after in itertools.pairwise(lines)
    )


def _code(
    fn: object,
    *extra: object,
    reduction_loop: int | None = None,
    static_shapes: bool = True,
) -> str:
    bound = _bind(fn, *extra, static_shapes=static_shapes)
    config = bound.config_spec.default_config()
    config.config["block_sizes"] = [16] * len(config.config["block_sizes"])
    if reduction_loop is not None:
        config.config["reduction_loops"] = [reduction_loop]
    with cpu_render_target():
        return bound.to_code(config)


@skipUnlessTritonBarriers
@pytest.mark.parametrize(
    "fn",
    [
        _reads_what_the_last_iteration_stored,
        _reads_the_previous_row_reversed,
        # Another view of the storage: the same index reaches other elements.
        _writes_a_shifted_view,
        _writes_a_transposed_view,
        # A view of another dtype: the same index reaches other bytes.
        _reads_a_reinterpreted_view,
        # The same view and index, but the view's rows overlap.
        _updates_an_overlapping_view,
        _updates_a_broadcast_view,
        # The index tells apart one axis of the loop's two.
        _updates_one_axis_of_a_2d_loop,
        # A full slice is the loop's tile only in a rolled reduction loop.
        _rereads_rows_sized_like_its_tile,
    ],
)
def test_a_load_an_earlier_iteration_stored_follows_a_barrier(fn: object) -> None:
    # (The store after the load may take a barrier of its own, before it
    # overwrites what other threads loaded.)
    code = _code(fn)
    assert _barriered_loads(code) == 1, code
    assert code.index("tl.debug_barrier()") < code.index("tl.load("), code


# Triton orders overlapping arguments.
@skipUnlessTritonBarriers
def test_a_load_of_an_argument_another_overlaps_follows_a_barrier() -> None:
    # The walk counts ``a`` and ``b`` as one storage; each is one view at
    # one index, but the two views reach different elements.
    base = torch.zeros(96)
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_updates_an_overlapping_argument, static_shapes=True),
            (base[:64], base[32:]),
        )
        code = bound.to_code(helion.Config(block_sizes=[32]))
    assert code.index("tl.debug_barrier()") < code.index("tl.load(a"), code


@skipUnlessTritonBarriers
def test_an_index_of_a_size_one_dimension_tells_no_iterations_apart() -> None:
    # (Every thread also stores the one row the others loaded: the store
    # takes a barrier too.)
    code = _code(_rewrites_a_size_one_row, torch.zeros(1, 64))
    assert _barriered_loads(code) == 1, code
    assert code.index("tl.debug_barrier()") < code.index("tl.load(row"), code


@skipUnlessTritonBarriers
@pytest.mark.parametrize("static_shapes", [True, False])
def test_an_update_through_a_size_one_dimension_follows_a_barrier(
    static_shapes: bool,
) -> None:
    # ``scratch`` has one row, so every row tile reaches row 0: no iteration
    # owns its tile.  With dynamic shapes the size bound decides, as it does
    # code generation's broadcast (under a guard).
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_updates_its_own_2d_tile, static_shapes=static_shapes),
            (torch.randn(32, 64), torch.zeros(1, 64)),
        )
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = [1, 16]
        code = bound.to_code(config)
    assert _barriered_loads(code) == 1, code


@skipUnlessTritonBarriers
def test_one_barrier_flushes_the_entry_and_the_carried_writes() -> None:
    # From the loop's entry alone (the store to ``c``) the load of ``c`` would
    # take the barrier; the barrier the carried store to ``a`` needs before
    # the earlier load of ``a`` flushes ``c`` too, so that one is the only one.
    # (The store to ``a`` takes its own, before it overwrites what other
    # threads loaded.)
    code = _code(_flushes_an_earlier_write_once, torch.zeros(32, 64))
    assert _barriered_loads(code) == 1, code
    assert code.index("tl.debug_barrier()") < code.index("tl.load(a"), code


@skipUnlessTritonBarriers
@pytest.mark.parametrize(
    ("fn", "extra"),
    [
        (_updates_its_own_tile, ()),
        (_updates_its_own_tile_under_an_if, (1.0,)),
        (_updates_its_own_columns_under_an_if, (1.0,)),
        (_updates_its_own_2d_tile, ()),
        (_updates_its_own_row, ()),
    ],
)
def test_an_update_of_the_iterations_own_tile_needs_no_barrier(
    fn: object, extra: tuple[object, ...]
) -> None:
    # No load waits for an earlier iteration.  (Through an ``if`` the store
    # takes a barrier before it overwrites what it loaded: that does not
    # prove each thread stores only what it loaded.)
    code = _code(fn, *extra)
    assert _barriered_loads(code) == 0, code


@skipUnlessTritonBarriers
def test_accesses_sharing_the_iterations_rows_need_no_carried_barrier() -> None:
    # The one barrier orders the second inner loop's loads after the first's
    # stores; none orders a store after the previous iteration's loads.
    code = _code(_stages_its_rows_in_two_inner_loops)
    assert code.count("tl.debug_barrier()") == 1, code
    barrier = code.index("tl.debug_barrier()")
    assert code.index("tl.store(scratch") < barrier < code.index("tl.load(scratch")


@skipUnlessTritonBarriers
def test_a_tile_an_inner_loop_shifts_is_not_the_iterations_own() -> None:
    # The inner loop carries ``columns``: its store's index is not the
    # ``columns`` of the load, though both name it.
    code = _code(_shifts_its_tile_in_an_inner_loop)
    assert code.index("tl.debug_barrier()") < code.index("tl.load(scratch"), code


@skipUnlessTritonBarriers
def test_an_update_of_its_own_tile_under_dynamic_shapes_follows_a_barrier() -> None:
    # The cache key leaves strides out: a later call may pass an expanded
    # view, whose iterations' tiles share memory.
    code = _code(_updates_its_own_tile, static_shapes=False)
    assert _barriered_loads(code) == 1, code


@skipUnlessTritonBarriers
def test_an_update_of_a_rolled_reduction_chunk_needs_no_barrier() -> None:
    code = _code(_scales_its_columns_in_place, reduction_loop=8)
    assert "for roffset_" in code, code
    assert _barriered_loads(code) == 0, code


@skipIfTileIR("TileIR runs no intra-loop barrier walk")
@skipIfRefEager("ref-eager mode builds no device IR")
@pytest.mark.parametrize(
    ("fn", "rereads"),
    [(_reads_what_the_last_iteration_stored, True), (_updates_its_own_tile, False)],
)
def test_a_loop_carried_reread_pins_the_loop(fn: object, rereads: bool) -> None:
    # Warp specialization would split the barrier per partition, so a loop
    # whose iterations re-read each other's stores is not warp-specialized.
    bound = _bind(fn)
    device_ir = bound.host_function.device_ir
    inner = {
        block_id
        for info in device_ir.graphs
        if isinstance(info, ForLoopGraphInfo)
        for block_id in info.block_ids
    }
    with bound.env, bound.host_function:
        pinned = intra_loop_raw_reread_block_ids(
            device_ir.graphs, device_ir.root_ids, device_ir.grid_block_ids
        )
    assert bool(pinned & inner) == rereads


@skipIfTileIR("TileIR runs no intra-loop barrier walk")
@skipIfRefEager("ref-eager mode builds no device IR")
def test_without_carries_a_while_loop_is_walked_once() -> None:
    # A while loop's condition re-reads what its body stored in the previous
    # iteration; a backend whose own barrier pass orders consecutive iterations
    # (``carries=False``: CuTe) walks the condition and the body once.
    bound = _bind(_counts_up_in_a_while_loop)
    device_ir = bound.host_function.device_ir
    walk = (device_ir.graphs, device_ir.root_ids, device_ir.grid_block_ids)
    with bound.env, bound.host_function:
        assert intra_loop_raw_reread_block_ids(*walk)
        assert not intra_loop_raw_reread_block_ids(*walk, carries=False)
