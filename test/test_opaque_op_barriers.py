"""Barriers around ``hl.inline_triton`` snippets that store or load.

A snippet may access any pointer it is given, so it reads and writes every
tensor of the kernel's memory it gets
(``loop_dependency_checker.opaque_access_target``): a load after a snippet
that stored to its tensor, in a sibling loop, a nested loop or the same loop
body, needs the barrier a plain store there gets, and so does a snippet
after a store to a tensor it is given.  So does a load after an atomic,
where the barrier is what orders atomics (on Triton; CuTe's thread-barrier
pass orders them itself).  An op given only values the device code
computed reaches no memory.  The kernels render on CPU (TileIR emits no
barriers, and ref-eager mode generates no code).
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.loop_dependency_checker import intra_loop_raw_reread_block_ids
from helion._testing import cpu_render_target
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only
_TRITON_BARRIERS = (
    skipUnlessBackends(["triton"]),
    skipIfTileIR("TileIR does not legalize tl.debug_barrier"),
    skipIfRefEager("ref-eager mode generates no code"),
)


def _triton_barriers(test: object) -> object:
    for skip in _TRITON_BARRIERS:
        test = skip(test)
    return test


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


def _code(fn: object, *args: object) -> str:
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(fn, static_shapes=True),
            (torch.randn(32, 64), torch.empty(32, 64), *args),
        )
        return bound.to_code(bound.config_spec.default_config())


@_triton_barriers
@pytest.mark.parametrize("reload", [True, False])
def test_a_snippet_store_orders_a_sibling_loop_like_a_store(reload: bool) -> None:
    # The sibling loop that loads ``scratch`` after the snippet stored to it
    # gets the inter-loop barrier; one that loads only ``x`` does not.
    code = _code(_sibling_loops, reload)
    assert ("tl.debug_barrier()" in code) == reload, code


@_triton_barriers
@pytest.mark.parametrize("fn", [_store_then_nested_loop, _store_then_load_in_one_body])
def test_a_snippet_store_orders_a_later_load(fn: object) -> None:
    # The load of ``scratch`` after the snippet's store, in a nested loop or
    # in the same loop body, follows a barrier.
    code = _code(fn)
    assert code.count("tl.debug_barrier()") == 1, code
    assert code.index("tl.debug_barrier()") > code.index("tl.store(scratch"), code


@_triton_barriers
def test_a_snippet_reading_a_stored_tensor_follows_a_barrier() -> None:
    # The snippet may read what the store before it wrote.
    code = _code(_store_then_snippet_load)
    assert code.count("tl.debug_barrier()") == 1, code
    assert code.index("tl.debug_barrier()") < code.index("tl.load(scratch"), code


@_triton_barriers
def test_a_load_after_an_atomic_follows_a_barrier() -> None:
    # Another warp's atomic add into ``scratch`` may not be visible yet.
    code = _code(_atomic_then_load_in_one_body)
    assert code.count("tl.debug_barrier()") == 1, code
    assert code.index("tl.atomic_add(scratch") < code.index("tl.debug_barrier()")


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
    assert intra_loop_raw_reread_block_ids(*walk, atomics=True)
    assert not intra_loop_raw_reread_block_ids(*walk, atomics=False)


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
