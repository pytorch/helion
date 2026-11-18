"""Barriers around ``hl.inline_triton`` snippets that store or load.

A snippet may access any pointer it is given, so it reads and writes every
tensor of the kernel's memory it gets
(``loop_dependency_checker.opaque_access_target``): a load after a snippet
that stored to its tensor, in a sibling loop, a nested loop or the same loop
body, needs the barrier a plain store there gets, and so does a snippet
after a store to a tensor it is given.  So does a load after an atomic,
where the barrier is what orders atomics (on Triton; CuTe's thread-barrier
pass orders them itself), an atomic after a store to its target or after an
atomic it does not commute with (the atomic reads its target; after the same
self-commuting op, an add after an add, it needs none unless either uses its
old value), and a load after a store through device pointers
(a stack tensor), which may reach any memory.  An op given only values the
device code computed reaches no memory.  The kernels render on CPU (TileIR
emits no barriers, and ref-eager mode generates no code).
"""

from __future__ import annotations

import ast
import functools
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._compiler.cross_loop_codegen import _allocation
from helion._compiler.device_function import SyncRole
from helion._compiler.loop_dependency_checker import INTRA_LOOP_RAW_BARRIER_FLUSHES_META
from helion._compiler.loop_dependency_checker import INTRA_LOOP_RAW_BARRIER_META
from helion._compiler.loop_dependency_checker import emit_intra_loop_raw_barrier
from helion._compiler.loop_dependency_checker import intra_loop_raw_reread_block_ids
from helion._compiler.loop_dependency_checker import mark_intra_loop_raw_barriers
from helion._testing import cpu_render_target
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessBackends
from helion._testing import skipUnlessTritonBarriers
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

if TYPE_CHECKING:
    from torch.fx import Node

    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only
# ``tl.store`` of a tile of ``x`` into ``scratch``, through its pointer.
_STORE = "tl.store({0} + {1}[:, None] * {2} + {3}[None, :], {4})"


def _sibling_loops(
    x: torch.Tensor, scratch: torch.Tensor, reload: hl.constexpr
) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            hl.inline_triton(
                _STORE,
                args=(scratch, tile_m.index, n, tile_n.index, x[tile_m, tile_n]),
                output_like=None,
            )
        for tile_n in hl.tile(n):
            source = scratch if reload else x
            out[tile_m, tile_n] = source[tile_m, tile_n] + 1.0
    return out


def _store_then_nested_loop(x: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        hl.inline_triton(
            "tl.store({0} + {1}[:, None] * {2} + tl.arange(0, {2})[None, :], {3})",
            args=(scratch, tile_m.index, n, x[tile_m, :]),
            output_like=None,
        )
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = scratch[tile_m, tile_n] + 1.0
    return out


def _store_then_load_in_one_body(
    x: torch.Tensor, scratch: torch.Tensor
) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            hl.inline_triton(
                _STORE,
                args=(scratch, tile_m.index, n, tile_n.index, x[tile_m, tile_n]),
                output_like=None,
            )
            out[tile_m, tile_n] = scratch[tile_m, tile_n] + 1.0
    return out


def _store_then_snippet_load(x: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            scratch[tile_m, tile_n] = x[tile_m, tile_n]
            out[tile_m, tile_n] = hl.inline_triton(
                "tl.load({0} + {1}[:, None] * {2} + {3}[None, :])",
                args=(scratch, tile_m.index, n, tile_n.index),
                output_like=x[tile_m, tile_n],
            )
    return out


def _atomic_then_load_in_one_body(
    x: torch.Tensor, scratch: torch.Tensor
) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            hl.atomic_add(scratch, [tile_m, tile_n], x[tile_m, tile_n])
            out[tile_m, tile_n] = scratch[tile_m, tile_n] + 1.0
    return out


def _store_then_atomics(
    x: torch.Tensor, scratch: torch.Tensor, store: hl.constexpr, release: hl.constexpr
) -> torch.Tensor:
    m, n = x.shape
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            if store:
                scratch[tile_m, tile_n] = x[tile_m, tile_n]
            if release:
                hl.atomic_add(
                    scratch, [tile_m, tile_n], x[tile_m, tile_n], sem="release"
                )
            else:
                hl.atomic_add(scratch, [tile_m, tile_n], x[tile_m, tile_n])
            hl.atomic_add(scratch, [tile_m, tile_n], x[tile_m, tile_n])
    return x


def _mixes_atomics(
    x: torch.Tensor, scratch: torch.Tensor, exchange: hl.constexpr
) -> torch.Tensor:
    m, n = x.shape
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            if exchange:
                hl.atomic_xchg(scratch, [tile_m, tile_n], x[tile_m, tile_n])
            else:
                hl.atomic_max(scratch, [tile_m, tile_n], x[tile_m, tile_n])
            hl.atomic_add(scratch, [tile_m, tile_n], x[tile_m, tile_n])
    return x


def _adds_then_reads_the_old_value(
    x: torch.Tensor, scratch: torch.Tensor
) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            hl.atomic_add(scratch, [tile_m, tile_n], x[tile_m, tile_n])
            out[tile_m, tile_n] = hl.atomic_add(
                scratch, [tile_m, tile_n], x[tile_m, tile_n]
            )
    return out


def _adds_to_one_column_each_step(
    x: torch.Tensor, scratch: torch.Tensor
) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        for _step in hl.grid(4):
            # Every step adds to the same elements; the first add returns the
            # old value.
            out[tile_m] = hl.atomic_add(scratch, [tile_m, 0], x[tile_m, 0])
            hl.atomic_add(scratch, [tile_m, 0], x[tile_m, 0])
    return out


def _store_two_then_load(
    x: torch.Tensor, scratch: torch.Tensor, other: torch.Tensor | None
) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            scratch[tile_m, tile_n] = x[tile_m, tile_n]
            if other is not None:
                other[tile_m, tile_n] = x[tile_m, tile_n] * 2
            out[tile_m, tile_n] = scratch[tile_m, tile_n] + 1.0
    return out


def _pointer_store_then_load(
    x: torch.Tensor,
    scratch: torch.Tensor,
    dev_ptrs: torch.Tensor,
    example: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.specialize(dev_ptrs.size(0))
    for _ in hl.tile(1, block_size=1):
        for i in hl.grid(x.size(0)):
            tensors = hl.stacktensor_like(example, dev_ptrs[:])
            tensors[i] = x[None, i]
            out[i] = scratch[i] + 1
    return out


def _pointer_store_then_packed_load(
    x: torch.Tensor,
    scratch: torch.Tensor,
    dev_ptrs: torch.Tensor,
    example: torch.Tensor,
    w: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.specialize(dev_ptrs.size(0))
    for _ in hl.tile(1, block_size=1):
        for i in hl.grid(x.size(0)):
            group = offsets[i]
            tensors = hl.stacktensor_like(example, dev_ptrs[:])
            tensors[i] = x[None, i]
            lanes = hl.load_bfloat16_x16_to_float16(w, group)
            out[i] = scratch[i] + lanes[0].to(torch.float32)
    return out


def _store_then_pointer_load(
    x: torch.Tensor,
    scratch: torch.Tensor,
    dev_ptrs: torch.Tensor,
    example: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.specialize(dev_ptrs.size(0))
    for _ in hl.tile(1, block_size=1):
        for i in hl.grid(x.size(0)):
            tensors = hl.stacktensor_like(example, dev_ptrs[:])
            out[i] = tensors[i].sum(0)
            scratch[i] = x[i]
    return out


def _load_then_pointer_store(
    x: torch.Tensor,
    scratch: torch.Tensor,
    dev_ptrs: torch.Tensor,
    example: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.specialize(dev_ptrs.size(0))
    for _ in hl.tile(1, block_size=1):
        for i in hl.grid(x.size(0)):
            out[i] = scratch[i] + 1
            tensors = hl.stacktensor_like(example, dev_ptrs[:])
            tensors[i] = x[None, i]
    return out


def _pointer_store_then_sibling_loop(
    x: torch.Tensor,
    scratch: torch.Tensor,
    dev_ptrs: torch.Tensor,
    example: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.specialize(dev_ptrs.size(0))
    for _ in hl.tile(1, block_size=1):
        for i in hl.grid(x.size(0)):
            tensors = hl.stacktensor_like(example, dev_ptrs[:])
            tensors[i] = x[None, i]
        for j in hl.grid(x.size(0)):
            out[j] = scratch[j] + 1
    return out


def _store_then_sibling_pointer_loop(
    x: torch.Tensor,
    scratch: torch.Tensor,
    dev_ptrs: torch.Tensor,
    example: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(x)
    hl.specialize(dev_ptrs.size(0))
    for _ in hl.tile(1, block_size=1):
        for i in hl.grid(x.size(0)):
            scratch[i] = x[i]
        for j in hl.grid(x.size(0)):
            tensors = hl.stacktensor_like(example, dev_ptrs[:])
            out[j] = tensors[j].sum(0)
    return out


def _code(fn: object, *args: object) -> str:
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(fn, static_shapes=True),
            (torch.randn(32, 64), torch.empty(32, 64), *args),
        )
        return bound.to_code(bound.config_spec.default_config())


@skipUnlessTritonBarriers
@pytest.mark.parametrize("reload", [True, False])
def test_a_snippet_store_orders_a_sibling_loop_like_a_store(reload: bool) -> None:
    # The sibling loop that loads ``scratch`` after the snippet stored to it
    # gets the inter-loop barrier; one that loads only ``x`` does not.  (The
    # snippet itself follows a barrier: it may read what the previous
    # iteration's snippet stored.)
    code = _code(_sibling_loops, reload)
    second_loop = code.rindex("for offset_")
    assert code.count("tl.debug_barrier()") == 1 + reload, code
    assert code.count("tl.debug_barrier()", 0, second_loop) == 1 + reload, code


@skipUnlessTritonBarriers
@pytest.mark.parametrize(
    ("fn", "before_store"),
    [(_store_then_nested_loop, 0), (_store_then_load_in_one_body, 1)],
)
def test_a_snippet_store_orders_a_later_load(fn: object, before_store: int) -> None:
    # The load of ``scratch`` after the snippet's store, in a nested loop or
    # in the same loop body, follows a barrier.  In one body the snippet
    # takes one too, before its store: it overwrites what the previous
    # iteration loaded.
    code = _code(fn)
    store = code.index("tl.store(scratch")
    assert code.count("tl.debug_barrier()", 0, store) == before_store, code
    assert code.count("tl.debug_barrier()", store) == 1, code
    assert code.index("tl.debug_barrier()", store) < code.index("tl.load(", store)


@skipUnlessTritonBarriers
def test_a_snippet_reading_a_stored_tensor_follows_a_barrier() -> None:
    # The snippet may read what the store before it wrote.  The store takes
    # a barrier too, before it: it overwrites what the previous iteration's
    # snippet read.
    code = _code(_store_then_snippet_load)
    store = code.index("tl.store(scratch")
    assert code.count("tl.debug_barrier()", 0, store) == 1, code
    assert code.count("tl.debug_barrier()", store) == 1, code
    assert code.index("tl.debug_barrier()", store) < code.index("tl.load(scratch")


@skipUnlessTritonBarriers
def test_a_load_after_an_atomic_follows_a_barrier() -> None:
    # Another warp's atomic add into ``scratch`` may not be visible yet.
    code = _code(_atomic_then_load_in_one_body)
    assert code.count("tl.debug_barrier()") == 1, code
    assert code.index("tl.atomic_add(scratch") < code.index("tl.debug_barrier()")


@skipUnlessTritonBarriers
@pytest.mark.parametrize("release", [False, True])
@pytest.mark.parametrize("store", [False, True])
def test_an_atomic_after_a_store_follows_a_barrier(store: bool, release: bool) -> None:
    # The first atomic reads ``scratch``, which another warp's store, earlier
    # in the body or in an earlier iteration, may not have reached yet; the
    # barrier before it is also its release barrier.  An add after the adds
    # of the body and of earlier iterations needs none.
    code = _code(_store_then_atomics, store, release)
    assert code.count("tl.debug_barrier()") == (store or release), code
    if store:
        barrier = code.index("tl.debug_barrier()")
        assert code.index("tl.store(scratch") < barrier, code
        assert barrier < code.index("tl.atomic_add(scratch"), code


@skipUnlessTritonBarriers
@pytest.mark.parametrize("exchange", [False, True])
def test_an_atomic_after_one_it_does_not_commute_with_follows_a_barrier(
    exchange: bool,
) -> None:
    # Another warp's exchange (or max) may land before or after this add:
    # each order gives another result.  The atomics touch the iteration's own
    # tile, so the previous iteration's need no barrier.
    code = _code(_mixes_atomics, exchange)
    first = code.index("tl.atomic_xchg" if exchange else "tl.atomic_max")
    assert code.count("tl.debug_barrier()") == 1, code
    assert first < code.index("tl.debug_barrier()") < code.index("tl.atomic_add")


@skipUnlessTritonBarriers
def test_an_add_whose_old_value_is_used_follows_a_barrier() -> None:
    # The old value the second add returns depends on whether another warp's
    # first add landed before it.
    code = _code(_adds_then_reads_the_old_value)
    first = code.index("tl.atomic_add")
    assert code.count("tl.debug_barrier()") == 1, code
    assert first < code.index("tl.debug_barrier()") < code.rindex("tl.atomic_add")


@skipUnlessTritonBarriers
def test_an_add_whose_old_value_is_used_follows_the_previous_steps_adds() -> None:
    # Every step adds to the same elements: the old value the first add returns
    # depends on whether another warp's add of the previous step landed, and
    # the second add on whether another warp's first add did.
    code = _code(_adds_to_one_column_each_step)
    first = code.index("tl.atomic_add")
    assert code.count("tl.debug_barrier()") == 2, code
    assert code.index("tl.debug_barrier()") < first, code
    assert first < code.rindex("tl.debug_barrier()") < code.rindex("tl.atomic_add")


@skipIfTileIR("TileIR runs no intra-loop barrier walk")
@skipIfRefEager("ref-eager mode builds no device IR")
def test_a_load_after_an_atomic_rereads_only_where_atomics_count() -> None:
    # The walk behind the marker and the warp-specialization pinning counts
    # the atomic as a write only with ``atomics``
    # (``Backend.intra_loop_barriers_order_atomics``: Triton, not CuTe).
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_atomic_then_load_in_one_body, static_shapes=True),
            (torch.randn(32, 64), torch.empty(32, 64)),
        )
    device_ir = bound.host_function.device_ir
    walk = (device_ir.graphs, device_ir.root_ids, device_ir.grid_block_ids)
    with bound.env, bound.host_function:
        assert intra_loop_raw_reread_block_ids(*walk, atomics=True)
        assert not intra_loop_raw_reread_block_ids(*walk, atomics=False)


@functools.cache
def _bind_pointer_kernel(fn: object) -> BoundKernel:
    # ``x``, ``scratch``, a table of two device pointers and a tensor like
    # the ones they point to; once per kernel and module.
    with cpu_render_target():
        return cpu_bind(
            helion.kernel(fn, static_shapes=True),
            (
                torch.randn(64),
                torch.empty(64),
                torch.zeros(2, dtype=torch.uint64),
                torch.empty(64),
            ),
        )


def _pointer_code(fn: object) -> str:
    bound = _bind_pointer_kernel(fn)
    with cpu_render_target():
        return bound.to_code(bound.config_spec.default_config())


@skipUnlessTritonBarriers
def test_a_load_after_a_pointer_store_follows_a_barrier() -> None:
    # A store through device pointers (a stack tensor) may reach any memory:
    # the next load gets the barrier, whatever it reads.  (The store takes one
    # too: it may overwrite what the previous iteration loaded.)
    code = _pointer_code(_pointer_store_then_load)
    store = code.index("tl.store(load.to(tl.pointer_type")
    assert code.count("tl.debug_barrier()", store) == 1, code
    assert code.index("tl.debug_barrier()", store) < code.index("tl.load(scratch")


@skipUnlessTritonBarriers
def test_a_load_after_a_previous_pointer_store_follows_a_barrier() -> None:
    # The previous iteration's store through device pointers may have
    # reached ``scratch``.  (The store takes a barrier too: it may overwrite
    # what the load read.)
    code = _pointer_code(_load_then_pointer_store)
    load = code.index("tl.load(scratch")
    assert code.count("tl.debug_barrier()", 0, load) == 1, code
    assert code.index("tl.debug_barrier()") < load, code


@skipUnlessBackends(["cute"])
@skipIfRefEager("ref-eager mode generates no code")
@pytest.mark.parametrize(
    ("fn", "read"),
    [
        (_pointer_store_then_load, "scratch.iterator"),
        # CuTe's own barrier pass orders the iterations' other accesses, but
        # does not take a pointer access to reach ``scratch``.
        (_store_then_pointer_load, "cute.make_ptr"),
        (_load_then_pointer_store, "scratch.iterator"),
    ],
)
def test_a_pointer_hazard_follows_a_cute_barrier(fn: object, read: str) -> None:
    # On CuTe too, a load after a store through device pointers, or a load
    # through them after a store, in the same body or the previous iteration,
    # follows the barrier.
    code = _pointer_code(fn)
    barrier = "cute.arch.sync_threads()"
    assert code.count(barrier) == 1, code
    assert code.index(barrier) < code.index(read), code


@skipUnlessBackends(["cute"])
@skipIfRefEager("ref-eager mode generates no code")
def test_a_cute_packed_load_leaves_a_pointer_hazard_pending() -> None:
    # CuTe's packed-load codegen places no barrier, so the packed load does
    # not take the pointer store's: the next load does, as without it.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_pointer_store_then_packed_load, static_shapes=True),
            (
                torch.randn(64),
                torch.empty(64),
                torch.zeros(2, dtype=torch.uint64),
                torch.empty(64),
                torch.zeros(1024, dtype=torch.bfloat16),
                torch.zeros(64, dtype=torch.int64),
            ),
        )
        code = bound.to_code(bound.config_spec.default_config())
    barrier = "cute.arch.sync_threads()"
    assert code.count(barrier) == 1, code
    assert code.index(barrier) < code.index("scratch.iterator"), code


@skipUnlessTritonBarriers
def test_a_pointer_load_after_a_store_follows_a_barrier() -> None:
    # A load through device pointers may read what the previous iteration's
    # store to ``scratch`` wrote: its pointers may name ``scratch``.  (The
    # store takes a barrier too: it may overwrite what that load read.)
    code = _pointer_code(_store_then_pointer_load)
    pointer_load = code.index("tl.load(load.to(tl.pointer_type")
    assert code.count("tl.debug_barrier()", 0, pointer_load) == 1, code
    assert code.index("tl.debug_barrier()") < pointer_load, code


@skipUnlessTritonBarriers
def test_a_pointer_store_orders_a_sibling_loop_once() -> None:
    # The store through device pointers may reach ``scratch``: the sibling
    # loop that reads it follows one barrier, not one per iteration.
    code = _pointer_code(_pointer_store_then_sibling_loop)
    second_loop = code.rindex("for offset_")
    assert code.count("tl.debug_barrier()", second_loop) == 0, code
    assert code.rindex("tl.debug_barrier()", 0, second_loop) > code.index(
        "tl.store(load"
    ), code


@skipUnlessTritonBarriers
def test_a_sibling_pointer_loop_after_a_store_follows_a_barrier() -> None:
    # The second loop's device pointers may name ``scratch``: it follows the
    # barrier before it (and one per iteration for its own store to ``out``,
    # which they may name too).
    code = _pointer_code(_store_then_sibling_pointer_loop)
    second_loop = code.rindex("for offset_")
    assert code.rindex("tl.debug_barrier()", 0, second_loop) > code.index(
        "tl.store(scratch"
    ), code


def _marked_barrier(bound: BoundKernel, *, reads: bool = False) -> Node:
    """The one access ``mark_intra_loop_raw_barriers`` (with ``reads``)
    marks in copies of ``bound``'s graphs: the cached kernel's own graphs stay
    unmarked (code generation copies them with their nodes' meta)."""
    device_ir = bound.host_function.device_ir
    graphs = [info.copy() for info in device_ir.graphs]
    with bound.env, bound.host_function:
        mark_intra_loop_raw_barriers(graphs, device_ir.root_ids, reads=reads)
    (marked,) = (
        node
        for info in graphs
        for node in info.graph.nodes
        if node.meta.get(INTRA_LOOP_RAW_BARRIER_META)
    )
    return marked


def _fence_covers(
    bound: BoundKernel, marked: Node, accessed: str = "scratch"
) -> object:
    """What the fence of the barrier before ``marked`` (an access of
    ``accessed``) covers, emitted into a stub device function."""
    host = bound.host_function
    fakes = {origin.host_str(): fake for fake, origin in host.tensor_to_origin.items()}
    roles = {"barrier_fence": SyncRole("wait")}
    function = SimpleNamespace(
        cta_barrier=lambda: [ast.parse("barrier_fence = 0").body[0]],
        cta_read_barrier=lambda: [ast.parse("barrier_fence = 0").body[0]],
        sync_roles=roles,
        sync_names=lambda *kinds: {n for n, r in roles.items() if r.kind in kinds},
    )
    emitted: list[ast.stmt] = []
    state = SimpleNamespace(
        fx_node=marked, device_function=function, add_statement=emitted.append
    )
    with bound.env, host:
        emit_intra_loop_raw_barrier(state, fakes[accessed])  # pyrefly: ignore
    assert [ast.unparse(statement) for statement in emitted] == ["barrier_fence = 0"]
    return roles["barrier_fence"].covers


@functools.cache
def _bound_store_two(second_store: bool) -> BoundKernel:
    other = torch.empty(32, 64) if second_store else None
    with cpu_render_target():
        return cpu_bind(
            helion.kernel(_store_two_then_load, static_shapes=True),
            (torch.randn(32, 64), torch.empty(32, 64), other),
        )


@skipIfTileIR("TileIR runs no intra-loop barrier walk")
@skipIfRefEager("ref-eager mode builds no device IR")
@pytest.mark.parametrize("second_store", [False, True])
def test_a_barrier_records_the_writes_it_flushes(second_store: bool) -> None:
    # The barrier before the load of ``scratch`` also flushes the store to
    # ``other`` when there is one (``emit_intra_loop_raw_barrier`` covers its
    # fence by what it flushes).
    bound = _bound_store_two(second_store)
    marked = _marked_barrier(bound)
    stored = ("scratch", "other") if second_store else ("scratch",)
    expected = {
        id(fake.untyped_storage())
        for fake, origin in bound.host_function.tensor_to_origin.items()
        if origin.host_str() in stored
    }
    flushed = marked.meta[INTRA_LOOP_RAW_BARRIER_FLUSHES_META]
    assert len(expected) == len(flushed) == len(stored)
    assert {id(tensor.untyped_storage()) for tensor in flushed} == expected


@skipIfTileIR("TileIR runs no intra-loop barrier walk")
@skipIfRefEager("ref-eager mode builds no device IR")
def test_a_barrier_fence_covers_the_read_and_what_it_flushes() -> None:
    # ``emit_intra_loop_raw_barrier`` records the barrier's waits as ordering
    # reads of the loaded tensor and of every tensor whose writes it flushes
    # (a ring partition reading none of them drops them).
    bound = _bound_store_two(True)
    covers = _fence_covers(bound, _marked_barrier(bound))
    host = bound.host_function
    fakes = {origin.host_str(): fake for fake, origin in host.tensor_to_origin.items()}
    with bound.env, host:
        assert covers == {_allocation(fakes["scratch"]), _allocation(fakes["other"])}


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


@skipUnlessBackends(["triton"])  # orders reads
@skipIfTileIR("TileIR runs no intra-loop barrier walk")
@skipIfRefEager("ref-eager mode builds no device IR")
def test_a_barrier_fence_covers_the_reads_it_completes() -> None:
    # The store to ``y`` takes no barrier of its own: the one before the
    # store to ``x`` orders it, so its fence covers ``y`` too.  No write
    # reaches ``z``, whose read it also completes.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_reads_three_then_overwrites_two, static_shapes=True),
            (torch.randn(32, 128), torch.randn(32, 128), torch.randn(32, 128)),
        )
    host = bound.host_function
    fakes = {origin.host_str(): fake for fake, origin in host.tensor_to_origin.items()}
    covers = _fence_covers(bound, _marked_barrier(bound, reads=True), "x")
    with bound.env, host:
        assert covers == {_allocation(fakes["x"]), _allocation(fakes["y"])}


@skipUnlessBackends(["triton"])  # orders overlapping arguments
@skipIfTileIR("TileIR runs no intra-loop barrier walk")
@skipIfRefEager("ref-eager mode builds no device IR")
def test_a_barrier_over_overlapping_arguments_keeps_its_whole_fence() -> None:
    # ``scratch`` and ``other`` overlap: the walk counts them as one storage,
    # so the barrier before the load of ``scratch`` orders reads of either,
    # which name different allocations.  Its fence orders every read.
    base = torch.empty(32, 128)
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_store_two_then_load, static_shapes=True),
            (torch.randn(32, 64), base[:, :64], base[:, 32:96]),
        )
    assert bound.env.input_storage_aliases
    assert _fence_covers(bound, _marked_barrier(bound)) is None


@skipUnlessBackends(["triton", "cute"])  # Pallas binds no stack tensor
@skipIfTileIR("TileIR runs no intra-loop barrier walk")
@skipIfRefEager("ref-eager mode builds no device IR")
def test_a_barrier_after_a_pointer_store_keeps_its_whole_fence() -> None:
    # A store through device pointers may reach any memory: the next load
    # gets a barrier, which records the store (None), and its fence orders
    # every read.
    bound = _bind_pointer_kernel(_pointer_store_then_load)
    marked = _marked_barrier(bound)
    assert None in marked.meta[INTRA_LOOP_RAW_BARRIER_FLUSHES_META]
    assert _fence_covers(bound, marked) is None


def _printed_sum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.size(0), device=x.device)
    for i in hl.tile(x.size(0)):
        out[i] = x[i, :, :].sum((1, 2))
        hl.device_print("debug", x[i, 0, 0])
    return out


@skipUnlessBackends(["cute"])
@skipIfRefEager("ref-eager mode generates no code")
def test_an_op_given_no_memory_writes_none() -> None:
    # ``hl.device_print`` of a loaded element reaches no memory: the sum over
    # dims 1 and 2 still merges them into one reduction, which CuTe needs.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_printed_sum, static_shapes=True), (torch.randn(4, 8, 8),)
        )
        bound.to_code(bound.config_spec.default_config())
