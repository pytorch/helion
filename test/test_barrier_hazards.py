"""Barriers between accesses of one element by different threads.

Two accesses of an element by different threads, one of them a write, race
unless a CTA barrier comes between (``mark_intra_loop_raw_barriers``): a
read after a write, and on Triton a write after a read or after a write.  A
loop body starts with the accesses of its earlier iterations pending, and a
sibling loop with those of the loops before it (``needs_barrier_before``).
Accesses take none when their indices never meet on some dimension, when
they are the same store or index the loop's own tile in another iteration,
or when a store writes back what a load at its index read
(``x[t] = f(x[t])``).  The kernels render on CPU (TileIR emits no barriers,
and ref-eager mode generates no code); the last tests run two across warps.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cross_loop_codegen import _allocation
from helion._compiler.device_function import SyncRole
from helion._compiler.loop_dependency_checker import INTRA_LOOP_RAW_BARRIER_META
from helion._compiler.loop_dependency_checker import emit_intra_loop_raw_barrier
from helion._compiler.loop_dependency_checker import intra_loop_raw_reread_block_ids
from helion._compiler.loop_dependency_checker import mark_intra_loop_raw_barriers
from helion._testing import DEVICE
from helion._testing import cpu_render_target
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessTritonBarriers
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only

_BARRIER = "tl.debug_barrier()"


def _overwrites_what_it_read(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), 64], device=x.device)
    for tile_m in hl.tile(x.size(0)):
        # Each thread reads the row's upper half, which other threads own in
        # the whole-row store.
        upper = x[tile_m, hl.arange(64) + 64]
        x[tile_m, :] = x.new_zeros([tile_m, 128])
        out[tile_m, :] = upper
    return out


def _reads_what_it_overwrote(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), 64], device=x.device)
    for tile_m in hl.tile(x.size(0)):
        x[tile_m, :] = x.new_zeros([tile_m, 128])
        out[tile_m, :] = x[tile_m, hl.arange(64) + 64]
    return out


def _releases_into_what_it_read(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), 64], device=x.device)
    for tile_m in hl.tile(x.size(0)):
        half = hl.arange(64)
        upper = x[tile_m, half + 64]
        hl.atomic_add(x, [tile_m, half + 64], x.new_ones([tile_m, 64]), sem="release")
        out[tile_m, :] = upper
    return out


def _resets_what_it_added_into(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        hl.atomic_add(x, [tile_m, hl.arange(64) + 64], x.new_ones([tile_m, 64]))
        x[tile_m, :] = x.new_zeros([tile_m, 128])
    return x


def _gathers_and_updates(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    for tile in hl.tile(indices.size(0)):
        # Repeated indices put one element in several threads.
        idx = indices[tile]
        x[idx] = x[idx] + 1.0
    return x


def _transposes_its_own_tile(x: torch.Tensor) -> torch.Tensor:
    for tile_a, tile_b in hl.tile([x.size(0), x.size(1)], block_size=[32, 32]):
        # The thread that stores an element may have loaded its mirror.
        x[tile_a, tile_b] = x[tile_a, tile_b].T
    return x


def _packed_load_after_a_store(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    out = torch.empty([offsets.size(0), 16], dtype=torch.float16, device=x.device)
    for tile in hl.tile(offsets.size(0)):
        x[:] = x.new_zeros([128])
        # A packed load reads anywhere in its tensor.
        lanes = hl.load_bfloat16_x16_to_float16(x, offsets[tile])
        for i in hl.static_range(16):
            out[tile, i] = lanes[i]
    return out


def _restores_part_of_what_it_stored(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        x[tile_m, :] = x.new_ones([tile_m, 128])
        # Other threads own the upper half in the whole-row store.
        x[tile_m, hl.arange(64) + 64] = x.new_full([tile_m, 64], 2.0)
    return x


def _stores_wrapping_halves(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        x[tile_m, hl.arange(128, dtype=torch.uint8)] = x.new_ones([tile_m, 128])
        # 224 + 32 wraps to 0: the upper half's tail lands on the first.
        x[tile_m, hl.arange(64, dtype=torch.uint8) + 224] = x.new_zeros([tile_m, 64])
    return x


def _stores_the_halves_of_a_row(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        half = hl.arange(64)
        x[tile_m, half] = x.new_ones([tile_m, 64])
        x[tile_m, half + 64] = x.new_zeros([tile_m, 64])
    return x


def _reads_what_the_last_iteration_stored(
    x: torch.Tensor, scratch: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        for tile_n in hl.tile(x.size(1)):
            # The first 16 columns, which the first iteration stores.
            first = scratch[tile_m, 0:16].sum(-1, keepdim=True)
            out[tile_m, tile_n] = first + x[tile_m, tile_n]
            scratch[tile_m, tile_n] = x[tile_m, tile_n]
    return out


def _writes_a_shifted_view(x: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
    shifted = scratch[:, 1:]
    for tile_m in hl.tile(x.size(0)):
        for tile_n in hl.tile(x.size(1) - 1):
            # The next iteration reads the column this one writes last.
            shifted[tile_m, tile_n] = scratch[tile_m, tile_n] + 1
    return scratch


def _stores_reversed_every_other_iteration(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        columns = hl.arange(64)
        for tile_k in hl.tile(y.size(1)):
            # The loop carries the index: other threads store the columns.
            x[tile_m, columns] = y[tile_m, tile_k]
            columns = 63 - columns
    return x


def _reads_the_slot_the_last_iteration_stored(
    x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(y)
    for tile_m in hl.tile(x.size(0)):
        for i in hl.grid(y.size(1)):
            # The last iteration's ``i + 1`` is this one's ``i``.
            value = x[tile_m, i]
            out[tile_m, i] = value + y[tile_m, i]
            x[tile_m, i + 1] = value + 1.0
    return out


def _stores_windows_that_overlap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        for k in hl.grid(4):
            # Each window begins 3 columns on: its elements move to other
            # threads.
            for tile_n in hl.tile(k * 3, k * 3 + 64, block_size=64):
                x[tile_m, tile_n] = y[tile_m, k][:, None] + x.new_zeros(
                    [tile_m, tile_n]
                )
    return x


def _updates_its_own_tile(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        x[tile_m, :] = x[tile_m, :] * 2.0 + 1.0
    return x


def _updates_in_a_loop(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0), block_size=1):
        for tile_n in hl.tile(x.size(1)):
            x[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
    return x


def _overwrites_in_a_loop(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m, block_size=1):
        for tile_n in hl.tile(n):
            mirrored = x[tile_m, n - 1 - tile_n.index]
            x[tile_m, tile_n] = x.new_zeros([tile_m, tile_n])
            out[tile_m, tile_n] = mirrored
    return out


def _stores_two_slots(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        x[tile_m, 0, :] = x.new_ones([tile_m, 128])
        x[tile_m, 1, :] = x.new_zeros([tile_m, 128])
    return x


def _stores_slots_through_a_zero_stride(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        rows = tile_m.index[:, None]
        # A zero-stride slot index (the broadcast spelling of a constant).
        zero = rows * 0
        columns = hl.arange(128)[None, :]
        hl.store(x, [rows, zero, columns], x.new_ones([tile_m, 128]))
        hl.store(x, [rows, zero + 1, columns], x.new_zeros([tile_m, 128]))
    return x


def _stores_shifted_column_blocks(x: torch.Tensor) -> torch.Tensor:
    for tile_m, tile_h in hl.tile([x.size(0), 2], block_size=[None, 1]):
        # A block per head: the head's offset is in both indices.
        columns = tile_h.begin * 128 + hl.arange(128)
        x[tile_m, columns] = x.new_ones([tile_m, 128])
        x[tile_m, 256 + columns] = x.new_zeros([tile_m, 128])
    return x


def _stores_a_block_past_the_first(x: torch.Tensor) -> torch.Tensor:
    for tile_m, tile_v in hl.tile([x.size(0), 128], block_size=[None, 16]):
        x[tile_m, hl.arange(128)] = x.new_ones([tile_m, 128])
        # The loop runs from 0: its tile begins at 0 or later.
        x[tile_m, 128 + tile_v.begin + hl.arange(16)] = x.new_zeros([tile_m, 16])
    return x


def _stores_a_block_into_the_first(x: torch.Tensor) -> torch.Tensor:
    for tile_m, tile_v in hl.tile([x.size(0), 128], block_size=[None, 16]):
        x[tile_m, hl.arange(128)] = x.new_ones([tile_m, 128])
        # Columns 120 to 127 of the first block, in the first tile.
        x[tile_m, 120 + tile_v.begin + hl.arange(16)] = x.new_zeros([tile_m, 16])
    return x


def _doubles_one_half_into_the_other(x: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        x[tile_m, 64:128] = x[tile_m, 0:64] * 2.0
    return x


def _stores_its_rows_each_iteration(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        for tile_k in hl.tile(y.size(1)):
            # Each iteration overwrites the block the one before stored, from
            # the same threads.
            x[tile_m, 0:64] = y[tile_m, tile_k]
    return x


def _stores_its_tile_while_counting(
    x: torch.Tensor, y: torch.Tensor, count: torch.Tensor
) -> torch.Tensor:
    for tile_m, tile_n in hl.tile([x.size(0), x.size(1)]):
        left = count[0]
        while left > 0:
            x[tile_m, tile_n] = y[tile_m, tile_n] + left
            left = left - 1
    return x


def _sibling_loop_overwrites(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m, block_size=1):
        for tile_n in hl.tile(n, block_size=32):
            out[tile_m, tile_n] = x[tile_m, n - 1 - tile_n.index]
        for tile_n in hl.tile(n, block_size=32):
            x[tile_m, tile_n] = x.new_zeros([tile_m, tile_n])
    return out


def _sibling_loop_restores(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    for tile_m in hl.tile(m, block_size=1):
        for tile_n in hl.tile(n, block_size=32):
            x[tile_m, tile_n] = x.new_ones([tile_m, tile_n])
        for tile_n in hl.tile(n, block_size=64):
            x[tile_m, tile_n] = x.new_zeros([tile_m, tile_n])
    return x


def _reads_three_then_overwrites_two(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    out = torch.empty([x.size(0), 64], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(x.size(0)):
        upper = hl.arange(64) + 64
        out[tile_m, :] = x[tile_m, upper] + y[tile_m, upper] + z[tile_m, upper]
        x[tile_m, :] = x.new_zeros([tile_m, 128])
        # The barrier before the store to ``x`` completed the read of ``y``.
        y[tile_m, :] = y.new_zeros([tile_m, 128])
    return out


def _pair() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(32, 128), torch.randn(32, 128)


# The arguments of the kernels that take other than one 32 x 128 tensor.
_ARGS: dict[object, tuple[object, ...]] = {
    _gathers_and_updates: (torch.randn(64), torch.randint(0, 64, [32])),
    _transposes_its_own_tile: (torch.randn(64, 64),),
    _packed_load_after_a_store: (torch.randn(128).bfloat16(), torch.randint(0, 8, [8])),
    _reads_what_the_last_iteration_stored: _pair(),
    _writes_a_shifted_view: _pair(),
    _stores_reversed_every_other_iteration: (torch.randn(32, 64), torch.randn(32, 128)),
    _stores_two_slots: (torch.randn(32, 2, 128),),
    _stores_slots_through_a_zero_stride: (torch.randn(32, 2, 128),),
    _stores_shifted_column_blocks: (torch.randn(32, 512),),
    _stores_a_block_past_the_first: (torch.randn(32, 256),),
    _stores_a_block_into_the_first: (torch.randn(32, 256),),
    _reads_the_slot_the_last_iteration_stored: (
        torch.randn(32, 33),
        torch.randn(32, 32),
    ),
    _stores_windows_that_overlap: (torch.randn(32, 128), torch.randn(32, 4)),
    _stores_its_rows_each_iteration: (torch.randn(128, 128), torch.randn(128, 128)),
    _stores_its_tile_while_counting: (
        torch.randn(128, 128),
        torch.randn(128, 128),
        torch.tensor([3], dtype=torch.int32),
    ),
    _reads_three_then_overwrites_two: (*_pair(), torch.randn(32, 128)),
}


def _bind(fn: object, args: tuple[object, ...] = ()) -> object:
    with cpu_render_target():
        return cpu_bind(
            helion.kernel(fn, static_shapes=True),
            args or _ARGS.get(fn, (torch.randn(32, 128),)),
        )


def _code(fn: object, args: tuple[object, ...] = (), **config: object) -> str:
    bound = _bind(fn, args)
    default = bound.config_spec.default_config()
    with cpu_render_target():
        # A row per program, a whole row per access, unless ``config`` says.
        return bound.to_code(
            helion.Config(
                **{
                    **default.config,
                    "block_sizes": [1] * len(default.block_sizes),
                    **config,
                }
            )
        )


def _tma_code(fn: object, indexing: list[str]) -> str:
    """``fn`` rendered with 64 x 64 blocks and ``indexing``, its
    ``tensor_descriptor`` accesses through TMA."""
    with (
        patch(
            "helion._compiler.device_function._selects_tma_access", return_value=True
        ),
        # CPU tensors take no TMA; render the descriptor stores anyway.
        patch(
            "helion._compiler.indexing_strategy."
            "TensorDescriptorIndexingStrategy.is_supported",
            return_value=True,
        ),
    ):
        return _code(fn, block_sizes=[64, 64], indexing=indexing)


@skipUnlessTritonBarriers
@pytest.mark.parametrize(
    ("fn", "first", "second", "args"),
    [
        # A store, an atomic and a packed load after another thread's access.
        (_overwrites_what_it_read, "tl.load(x", "tl.store(x", ()),
        (_reads_what_it_overwrote, "tl.store(x", "tl.load(x", ()),
        (_resets_what_it_added_into, "tl.atomic_add(x", "tl.store(x", ()),
        (_packed_load_after_a_store, "tl.store(x", "tl.load(x.to(", ()),
        # Loaded indices may name one element twice; a transpose moves values.
        (_gathers_and_updates, "tl.load(x", "tl.store(x", ()),
        (_transposes_its_own_tile, "tl.load(x", "tl.store(x", ()),
        # Stores that meet: narrow indices wrap, and a dimension of one
        # element takes both halves to it.
        (_restores_part_of_what_it_stored, "tl.store(x", "tl.store(x", ()),
        (_stores_wrapping_halves, "tl.store(x", "tl.store(x", ()),
        (_stores_a_block_into_the_first, "tl.store(x", "tl.store(x", ()),
        (
            _stores_the_halves_of_a_row,
            "tl.store(x",
            "tl.store(x",
            (torch.randn(32, 1),),
        ),
        # A load of what an earlier iteration stored, through the same view
        # or another, and a store whose index the loop carries.
        (_reads_what_the_last_iteration_stored, None, "tl.load(scratch", ()),
        (_writes_a_shifted_view, None, "tl.load(scratch", ()),
        (_stores_reversed_every_other_iteration, None, "tl.store(x", ()),
        # Another iteration's loop index, and a window that begins elsewhere.
        (_reads_the_slot_the_last_iteration_stored, None, "tl.load(x", ()),
        (_stores_windows_that_overlap, None, "tl.store(x", ()),
    ],
)
def test_accesses_that_may_meet_follow_a_barrier(
    fn: object, first: str | None, second: str, args: tuple[object, ...]
) -> None:
    code = _code(fn, args)
    barrier = code.index(_BARRIER)
    if first is not None:
        assert code.index(first) < barrier, code
    assert barrier < code.index(second, barrier), code


@skipUnlessTritonBarriers
@pytest.mark.parametrize(
    "fn",
    [
        # ``x[t] = f(x[t])``, also in a loop over the row's tiles.
        _updates_its_own_tile,
        _updates_in_a_loop,
        # Indices that never meet: ``hl.arange`` offsets, slots, slices.
        _stores_the_halves_of_a_row,
        _stores_two_slots,
        _doubles_one_half_into_the_other,
        # A zero-stride slot, a shared symbolic offset, and a tile's begin
        # in a loop from 0.
        _stores_slots_through_a_zero_stride,
        _stores_shifted_column_blocks,
        _stores_a_block_past_the_first,
        # The same store each iteration, from the same threads.
        _stores_its_rows_each_iteration,
    ],
)
def test_accesses_that_never_meet_take_no_barrier(fn: object) -> None:
    assert _BARRIER not in _code(fn)


@skipUnlessTritonBarriers
def test_an_update_under_dynamic_shapes_follows_a_barrier() -> None:
    # The cache key leaves strides out: the binding may run on an expanded
    # view, whose rows share their elements.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_updates_its_own_tile, static_shapes=False),
            (torch.randn(32, 128),),
        )
        assert _BARRIER in bound.to_code(helion.Config(block_sizes=[1]))


@skipUnlessTritonBarriers
@pytest.mark.parametrize("fn", [_sibling_loop_overwrites, _sibling_loop_restores])
def test_a_sibling_loop_follows_one_barrier(fn: object) -> None:
    # One barrier between the loops, not one per iteration of the second.
    kernel = next(
        node
        for node in ast.parse(_code(fn)).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    loops = [statement for statement in kernel.body if isinstance(statement, ast.For)]
    assert len(loops) == 2
    before = kernel.body[kernel.body.index(loops[1]) - 1]
    assert ast.unparse(before) == _BARRIER
    assert _BARRIER not in ast.unparse(loops[1])


@skipIfTileIR("TileIR runs no intra-loop barrier walk")
@skipIfRefEager("ref-eager mode builds no device IR")
@pytest.mark.parametrize(
    ("fn", "pinned"), [(_overwrites_in_a_loop, True), (_updates_in_a_loop, False)]
)
def test_a_loop_taking_a_barrier_is_not_warp_specialized(
    fn: object, pinned: bool
) -> None:
    # Warp specialization would split the loop's barrier per partition.
    bound = _bind(fn)
    device_ir = bound.host_function.device_ir
    with bound.env, bound.host_function:
        block_ids = intra_loop_raw_reread_block_ids(
            device_ir.graphs, device_ir.root_ids, device_ir.grid_block_ids, reads=True
        )
    assert bool(block_ids) == pinned


@skipUnlessTritonBarriers
def test_a_barrier_drains_tma_stores_after_writes_alone() -> None:
    # bar.sync completes other threads' loads, but not a TMA store: after
    # reads alone no drain comes first, after a write (and before a release)
    # it does.  A store each iteration repeats from the same threads drains
    # its earlier copy if it goes through TMA, and takes no barrier.
    with patch(
        "helion._compiler.device_function._selects_tma_access", return_value=True
    ):
        reads = _code(_overwrites_what_it_read)
        writes = [_code(_reads_what_it_overwrote), _code(_releases_into_what_it_read)]
    assert "async_store_drain" not in reads, reads
    for code in writes:
        assert code.index("async_store_drain") < code.index(_BARRIER), code
    tma = _tma_code(_stores_its_rows_each_iteration, ["pointer", "tensor_descriptor"])
    assert _BARRIER not in tma, tma
    assert tma.index("async_store_drain") < tma.index("x_desc.store"), tma
    pointer = _tma_code(_stores_its_rows_each_iteration, ["pointer", "pointer"])
    assert "async_store_drain" not in pointer, pointer
    counted = _tma_code(
        _stores_its_tile_while_counting, ["pointer", "pointer", "tensor_descriptor"]
    )
    assert counted.index("async_store_drain") < counted.index("x_desc.store"), counted


@skipUnlessTritonBarriers
def test_a_barrier_fence_covers_what_it_orders() -> None:
    # The fence orders reads of the access's tensor (``x``), of what the
    # barrier flushes (``out``) and of what it completes that the kernel
    # writes (``y``), not ``z``: a ring partition reading none drops it.
    bound = _bind(_reads_three_then_overwrites_two)
    device_ir = bound.host_function.device_ir
    graphs = [info.copy() for info in device_ir.graphs]
    host = bound.host_function
    with bound.env, host:
        mark_intra_loop_raw_barriers(graphs, device_ir.root_ids, reads=True)
    (marked,) = (
        node
        for info in graphs
        for node in info.graph.nodes
        if node.meta.get(INTRA_LOOP_RAW_BARRIER_META)
    )
    fakes = {origin.host_str(): fake for fake, origin in host.tensor_to_origin.items()}
    roles = {"barrier_fence": SyncRole("wait")}
    function = SimpleNamespace(
        cta_barrier=lambda: [ast.parse("barrier_fence = 0").body[0]],
        cta_read_barrier=lambda: [ast.parse("barrier_fence = 0").body[0]],
        sync_roles=roles,
        sync_names=lambda *kinds: {n for n, r in roles.items() if r.kind in kinds},
    )
    state = SimpleNamespace(
        fx_node=marked, device_function=function, add_statement=lambda _: None
    )
    with bound.env, host:
        emit_intra_loop_raw_barrier(state, fakes["x"])  # pyrefly: ignore
        assert roles["barrier_fence"].covers == {
            _allocation(fakes[name]) for name in ("x", "y", "out")
        }


@skipUnlessTritonBarriers
@pytest.mark.cuda
def test_a_write_after_a_read_runs_correctly_across_warps() -> None:
    # Each warp reads half a row other warps overwrite with zeros.
    kernel = helion.kernel(
        _overwrites_what_it_read, config=helion.Config(block_sizes=[1], num_warps=8)
    )
    x = torch.randn(4096, 128, device=DEVICE)
    torch.testing.assert_close(kernel(x.clone()), x[:, 64:])


@skipUnlessTritonBarriers
@pytest.mark.cuda
def test_a_store_over_another_threads_store_runs_correctly_across_warps() -> None:
    # Each warp stores twos over ones other warps stored.
    kernel = helion.kernel(
        _restores_part_of_what_it_stored,
        config=helion.Config(block_sizes=[1], num_warps=8),
    )
    x = kernel(torch.empty(4096, 128, device=DEVICE))
    expected = torch.ones_like(x)
    expected[:, 64:] = 2.0
    torch.testing.assert_close(x, expected)
