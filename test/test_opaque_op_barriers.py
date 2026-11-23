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

import functools
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._compiler.loop_dependency_checker import intra_loop_raw_reread_block_ids
from helion._testing import cpu_render_target
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessBackends
from helion._testing import skipUnlessTritonBarriers
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

if TYPE_CHECKING:
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
