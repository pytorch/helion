"""Memory effects that keep CuTe tile-vector loads and stores in program order.

The tile-unroll vector hoist (``_cute_register_tile_unroll_vec_hoist``) moves a
per-lane load above the constexpr V-loop and the tile-vector store
(``_cute_register_tile_unroll_vec_store``) delays a per-lane store to one flush
after it.  Neither may cross a statement of the V-loop body that touches the
same tensor, or a tensor not proven disjoint from it: an atomic or store before
the load keeps the load scalar, and a load or atomic after the store restores
the scalar store.  Effects on tensors proven disjoint (a fresh allocation) keep
both optimizations.  Only statements that stay inside the lane loop count: a
lane-invariant statement the lane-loop distribution places after the loop
follows the flush.  A restored scalar store is decided before the thread-barrier
pass (``add_thread_barriers``) reads the nest, so the barriers separate the
statements as they are emitted.
"""

from __future__ import annotations

import ast
import re
from typing import Any

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._compiler.cute.memory_ops import _cute_statement_written_tensors
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language.memory_ops import CuteTileVecStoreSite

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _col_sum_atomic_then_lane_load(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Grid lane loop: every lane bumps ``out`` and must read back its bump."""
    m, n = x.size()
    res = torch.zeros(n, dtype=torch.float32, device=x.device)
    block_m = hl.register_block_size(m)
    block_n = hl.register_block_size(n)
    for tile_n in hl.tile(n, block_size=block_n):
        old = hl.atomic_add(out, [tile_n], hl.full([tile_n], 1.0, dtype=torch.float32))
        c = out[tile_n]
        acc = c - old
        for tile_m in hl.tile(m, block_size=block_m):
            acc += torch.sum(x[tile_m, tile_n].to(torch.float32), dim=0)
        res[tile_n] = acc
    return res


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_atomic_then_load(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Device lane loop (inner tile) with the same atomic-then-load body."""
    m, n = x.size()
    res = torch.zeros(m, n, dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m, block_size=1):
        for tile_n in hl.tile(n):
            old = hl.atomic_add(
                out,
                [tile_m, tile_n],
                hl.full([tile_m, tile_n], 1.0, dtype=torch.float32),
            )
            c = out[tile_m, tile_n]
            res[tile_m, tile_n] = c - old
    return res


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_store_then_reload(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Device lane loop: ``hl.store`` then a reload of the stored tile."""
    m, n = x.size()
    res = torch.zeros(m, n, dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m, block_size=1):
        for tile_n in hl.tile(n):
            hl.store(out, [tile_m, tile_n], x[tile_m, tile_n] + 1.0)
            c = out[tile_m, tile_n]
            res[tile_m, tile_n] = c * 2.0
    return res


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_store_then_reload_alias(
    x: torch.Tensor, out: torch.Tensor, out2: torch.Tensor
) -> torch.Tensor:
    """Device lane loop: ``hl.store`` through ``out``, reload through ``out2``.

    Passing one tensor for both keeps the pair unproven disjoint, so the store
    stays scalar and the reload stays in the V-loop; distinct storages keep
    both vector forms.
    """
    m, n = x.size()
    res = torch.zeros(m, n, dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m, block_size=1):
        for tile_n in hl.tile(n):
            hl.store(out, [tile_m, tile_n], x[tile_m, tile_n] + 1.0)
            c = out2[tile_m, tile_n]
            res[tile_m, tile_n] = c * 2.0
    return res


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _flat_store_then_reload(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Grid lane loop: ``hl.store`` then a reload of the stored tile."""
    n = x.size(0)
    res = torch.zeros(n, dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        hl.store(out, [tile_n], x[tile_n] + 1.0)
        c = out[tile_n]
        res[tile_n] = c * 2.0
    return res


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _grid_store_then_atomic(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Grid lane loop: a collected store followed by an atomic on its tensor.

    No read-after-write barrier is inserted for the atomic, so the store's
    flush must not be delayed past it: ``old`` reads the stored value.
    """
    n = x.size(0)
    res = torch.zeros(n, dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        hl.store(out, [tile_n], x[tile_n] + 1.0)
        old = hl.atomic_add(out, [tile_n], hl.full([tile_n], 1.0, dtype=torch.float32))
        res[tile_n] = old
    return res


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _grid_store_then_invariant_store(
    x: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """Grid lane loop: a collected store, then a lane-invariant store to its tensor.

    The zeroing store depends on no lane, so the lane-loop distribution places
    it after the loop, past the flush, and the collected store keeps its flush.
    """
    n = x.size(0)
    for tile_n in hl.tile(n):
        hl.store(out, [tile_n], x[tile_n] + 1.0)
        out[tile_n.begin] = 0.0
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _grid_store_then_uniform_atomic(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Grid lane loops on two thread axes: a collected store, then an atomic on
    its tensor that is uniform along the row axis.

    The atomic addresses the tile's first row on every row thread, so the
    restored scalar store and the atomic race across those threads: the
    barrier pass separates them, and must see the scalar store to do so.
    """
    m, n = x.shape
    for tile_m, tile_n in hl.tile([m, n]):
        hl.store(out, [tile_m, tile_n], x[tile_m, tile_n] + 1)
        hl.atomic_add(out, [tile_m.begin, tile_n], 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _nested_store_then_uniform_atomic(
    x: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """A device loop inside the grid's row axis: the same store and atomic."""
    m, n = x.shape
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            hl.store(out, [tile_m, tile_n], x[tile_m, tile_n] + 1)
            hl.atomic_add(out, [tile_m.begin, tile_n], 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _store_fresh_then_load(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A store to a fresh allocation precedes the load of an input."""
    n = x.size(0)
    a = torch.empty_like(x)
    b = torch.empty_like(x)
    for tile_n in hl.tile(n):
        a[tile_n] = x[tile_n] + 1.0
        c = y[tile_n]
        b[tile_n] = c * 2.0
    return a, b


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _atomic_arg_then_load(
    x: torch.Tensor, a: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """An atomic on one input precedes the load of another input.

    Whether the load keeps its packet depends on the runtime alias proof for
    ``a`` and ``y``: distinct storages are cache-specialized as disjoint,
    overlapping views are not.  No read-after-write barrier is inserted for an
    atomic, so the hoist gate alone decides.
    """
    n = x.size(0)
    b = torch.empty_like(x)
    for tile_n in hl.tile(n):
        hl.atomic_add(a, [tile_n], x[tile_n])
        c = y[tile_n]
        b[tile_n] = c * 2.0
    return b


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _load_only(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        out[tile_n] = x[tile_n] + y[tile_n]
    return out


def _grid_config() -> helion.Config:
    return helion.Config(
        block_sizes=[1024, 16],
        num_threads=[1, 4],
        cute_vector_widths=[1, 4],
        cute_lane_layouts=["strided", "blocked"],
    )


def _row_config() -> helion.Config:
    return helion.Config(
        block_sizes=[64],
        num_threads=[4],
        cute_vector_widths=[1, 4],
        cute_lane_layouts=["blocked", "blocked"],
    )


def _flat_config() -> helion.Config:
    return helion.Config(
        block_sizes=[16],
        num_threads=[4],
        cute_vector_widths=[4],
        cute_lane_layouts=["blocked"],
    )


def _two_axis_config() -> helion.Config:
    """Two row threads, each owning one row, around a vectorized column loop."""
    return helion.Config(
        block_sizes=[2, 64],
        num_threads=[2, 16],
        cute_vector_widths=[1, 4],
    )


def _two_axis_args() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty(8, 256, dtype=torch.int32),
        torch.empty(8, 256, dtype=torch.int32),
    )


def _code(kernel: Any, args: tuple[Any, ...], config: helion.Config) -> str:
    with _cpu_codegen():
        return kernel._bind_isolated(args).to_code(config)


def _vloop_body(code: str) -> str:
    """The generated statements from the constexpr V-loop to the kernel end."""
    return code[code.index("range_constexpr") :]


def _line_of(text: str, position: int) -> str:
    return text[text.rfind("\n", 0, position) + 1 : text.index("\n", position)]


def _atomic_args() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty(1024, 1024, dtype=torch.bfloat16),
        torch.empty(1024, dtype=torch.float32),
    )


def _row_args() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty(64, 1024, dtype=torch.float32),
        torch.empty(64, 1024, dtype=torch.float32),
    )


def _flat_args(count: int = 2) -> tuple[torch.Tensor, ...]:
    return tuple(torch.empty(1024, dtype=torch.float32) for _ in range(count))


def _assert_grid_atomic_then_load_stays_scalar(code: str) -> None:
    assert "cute.arch.load(out.iterator" not in code
    body = _vloop_body(code)
    atomic = body.index("cute.arch.atomic_add((out.iterator")
    reload = body.index("c = (out.iterator + cutlass.Int32(indices_1)")
    assert atomic < reload
    assert ".load()" in _line_of(body, reload)
    # The result store follows every effect and keeps its vector flush.
    assert "_cute_store_u32_vec(res.iterator" in code


def _assert_device_loop_atomic_then_load_stays_scalar(code: str) -> None:
    assert "cute.arch.load(out.iterator" not in code
    body = _vloop_body(code)
    assert body.index("cute.arch.atomic_add((out.iterator") < body.index(
        "c = (out.iterator"
    )
    assert "_cute_store_u32_vec(res.iterator" in code


def _assert_store_then_reload_stays_scalar(
    code: str, reloaded: str = "out", *, expect_barrier: bool = True
) -> None:
    """The scalar store precedes the reload (and its RAW barrier when the
    reload is marked as reading the stored storage) inside the V-loop."""
    assert f"cute.arch.load({reloaded}.iterator" not in code
    assert "_cute_store_u32_vec(out.iterator" not in code
    body = _vloop_body(code)
    store = body.index(".store(cutlass.Float32(v_1))")
    reload = body.index(f"c = ({reloaded}.iterator")
    assert store < reload
    if expect_barrier:
        assert store < body.index("cute.arch.sync_threads()") < reload
    assert _line_of(body, store).strip().startswith("(out.iterator")


def _assert_store_then_atomic_stays_scalar(code: str) -> None:
    assert "_cute_store_u32_vec(out.iterator" not in code
    assert "_tile_store_vals_0_0" not in code
    body = _vloop_body(code)
    store = body.index(".store(cutlass.Float32(v_1))")
    atomic = body.index("old = cute.arch.atomic_add((out.iterator")
    assert store < atomic
    assert _line_of(body, store).strip().startswith("(out.iterator")
    # The input load precedes every effect and the result store follows the
    # atomic: both keep their vector forms.
    assert "cute.arch.load(x.iterator" in code
    assert "_cute_store_u32_vec(res.iterator" in code


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _assert_barrier_between_the_scalar_store_and_the_atomic(code: str) -> None:
    """The store of ``out`` is a scalar store inside the V-loop again, and the
    barrier pass put a block-wide barrier between it and the atomic that the
    row threads issue for the tile's first row; the flush is gone."""
    assert "_cute_store_u32_vec(out.iterator" not in code
    assert "_tile_store_vals" not in code
    body = _vloop_body(code)
    store = body.index(".store(cutlass.Int32(v_1))")
    barrier = body.index("cute.arch.sync_threads()")
    atomic = body.index("cute.arch.atomic_add((out.iterator")
    assert store < barrier < atomic
    assert body.count("cute.arch.sync_threads()") == 1
    store_line = _line_of(body, store)
    assert store_line.strip().startswith("(out.iterator")
    assert _indent(_line_of(body, barrier)) == _indent(store_line)
    # The input load precedes every effect and keeps its packet.
    assert "cute.arch.load(x.iterator" in code


def _assert_invariant_store_follows_the_flush(code: str) -> None:
    """The collected store keeps its flush; the single scalar store of ``out``
    (the zeroing) follows it outside the lane loop, reading no lane."""
    flush = code.index("_cute_store_u32_vec(out.iterator")
    zeroing = code.index(".store(cutlass.Float32(0.0))")
    assert flush < zeroing
    assert code.count(".store(cutlass.Float32(") == 1
    zeroing_line = _line_of(code, zeroing)
    assert _indent(zeroing_line) <= _indent(_line_of(code, flush))
    assert "vec_lane" not in zeroing_line and "lane_base" not in zeroing_line


def test_packed_store_site_counts_as_a_write_before_purity() -> None:
    """A packed signed-byte store site binds the packet under the flush operand's
    name: a plain assignment with no calls that must still count as a write."""
    binding = ast.parse("_tile_store_vals_0_0 = _tile_unroll_vec_0_0").body[0]
    flush = ast.parse(
        "_cute_store_u16_vec((out.iterator + lane_base), "
        "_cute_signed_bitfield_to_bf16_packed(_tile_store_vals_0_0, 0, 4, 8))"
    ).body[0]
    scalar = ast.parse("(out.iterator + indices_0).store(value)").body[0]
    site = CuteTileVecStoreSite(
        "_tile_store_vals_0_0", "out", binding, scalar, None, flush
    )
    assert _cute_statement_written_tensors(binding, []) == set()
    assert _cute_statement_written_tensors(binding, [site]) == {"out"}
    # The append site and a grid's scalar placeholder are writes as well.
    append = ast.parse(
        "_tile_store_vals_0_1.append((value).bitcast(cutlass.Uint32))"
    ).body[0]
    assert _cute_statement_written_tensors(append, []) is None
    assert _cute_statement_written_tensors(
        append,
        [
            CuteTileVecStoreSite(
                "_tile_store_vals_0_1", "res", append, scalar, None, flush
            )
        ],
    ) == {"res"}
    assert _cute_statement_written_tensors(scalar, []) == {"out"}


@pytest.mark.parametrize(
    "source",
    [
        "cute.arch.sync_threads()",
        "total = _cute_grouped_reduce_shared_tree(acc, smem, 0, 32)",
        "total = cute.arch.warp_reduction_sum(acc)",
        "helper(out)",
    ],
)
def test_barriers_and_cross_thread_helpers_have_unknown_effects(source: str) -> None:
    assert _cute_statement_written_tensors(ast.parse(source).body[0], []) is None


def test_atomic_before_load_keeps_grid_load_in_vloop() -> None:
    _assert_grid_atomic_then_load_stays_scalar(
        _code(_col_sum_atomic_then_lane_load, _atomic_args(), _grid_config())
    )


def test_atomic_before_load_keeps_device_loop_load_in_vloop() -> None:
    _assert_device_loop_atomic_then_load_stays_scalar(
        _code(_row_atomic_then_load, _row_args(), _row_config())
    )


@pytest.mark.parametrize(
    ("kernel", "args", "config"),
    [
        (_row_store_then_reload, _row_args(), _row_config()),
        (_flat_store_then_reload, _flat_args(), _flat_config()),
    ],
    ids=["device_loop", "grid"],
)
def test_store_then_reload_keeps_scalar_store_before_reload(
    kernel: Any, args: tuple[Any, ...], config: helion.Config
) -> None:
    _assert_store_then_reload_stays_scalar(_code(kernel, args, config))


def test_store_then_reload_through_alias_keeps_scalar_store_before_reload() -> None:
    x, out = _row_args()
    # The same lane stores and reloads the element, so program order alone
    # keeps the reload exact; no barrier is marked for the second parameter.
    _assert_store_then_reload_stays_scalar(
        _code(_row_store_then_reload_alias, (x, out, out), _row_config()),
        reloaded="out2",
        expect_barrier=False,
    )


def test_store_then_reload_through_distinct_storage_keeps_vector_ops() -> None:
    x, out = _row_args()
    out2 = torch.empty_like(out)
    code = _code(_row_store_then_reload_alias, (x, out, out2), _row_config())
    assert "cute.arch.load(out2.iterator" in code
    assert "_cute_store_u32_vec(out.iterator" in code
    assert "_cute_store_u32_vec(res.iterator" in code
    assert ".load()" not in _vloop_body(code)


def test_store_then_atomic_keeps_scalar_store_before_atomic() -> None:
    _assert_store_then_atomic_stays_scalar(
        _code(_grid_store_then_atomic, _flat_args(), _flat_config())
    )


def test_lane_invariant_store_after_a_collected_store_keeps_the_flush() -> None:
    # The lane-loop distribution, not the demotion, orders a lane-invariant
    # store against the flush; only statements kept inside the loop demote.
    _assert_invariant_store_follows_the_flush(
        _code(_grid_store_then_invariant_store, _flat_args(), _flat_config())
    )


@pytest.mark.parametrize(
    "kernel",
    [_grid_store_then_uniform_atomic, _nested_store_then_uniform_atomic],
    ids=["grid", "device_loop"],
)
def test_store_then_uniform_atomic_gets_a_barrier_after_the_scalar_store(
    kernel: Any,
) -> None:
    code = _code(kernel, _two_axis_args(), _two_axis_config())
    _assert_barrier_between_the_scalar_store_and_the_atomic(code)


def test_device_loop_store_then_reload_keeps_unrelated_vector_ops() -> None:
    code = _code(_row_store_then_reload, _row_args(), _row_config())
    # The input load precedes every effect and the result store follows the
    # reload: both keep their vector forms.
    assert "cute.arch.load(x.iterator" in code
    assert "_cute_store_u32_vec(res.iterator" in code
    assert "_tile_store_vals_1_0 = []" not in code


def test_store_to_proven_disjoint_tensor_keeps_hoist() -> None:
    code = _code(_store_fresh_then_load, _flat_args(), _flat_config())
    assert "cute.arch.load(x.iterator" in code
    assert "cute.arch.load(y.iterator" in code
    assert "_cute_store_u32_vec(a.iterator" in code
    assert "_cute_store_u32_vec(b.iterator" in code
    assert ".load()" not in _vloop_body(code)


def test_atomic_on_runtime_proven_disjoint_input_keeps_hoist() -> None:
    code = _code(_atomic_arg_then_load, _flat_args(3), _flat_config())
    assert "cute.arch.load(x.iterator" in code
    assert "cute.arch.load(y.iterator" in code
    assert "_cute_store_u32_vec(b.iterator" in code
    body = _vloop_body(code)
    assert body.index("cute.arch.atomic_add((a.iterator") < body.index(
        "c = cutlass.Uint32(_tile_unroll_vec_0_1"
    )


def test_atomic_on_overlapping_view_keeps_load_in_vloop() -> None:
    x = torch.empty(1024, dtype=torch.float32)
    base = torch.empty(1280, dtype=torch.float32)
    code = _code(_atomic_arg_then_load, (x, base[:1024], base[256:]), _flat_config())
    # ``x`` is loaded before any effect; ``y`` overlaps the atomically
    # updated ``a`` and must read after its lane's atomic.
    assert "cute.arch.load(x.iterator" in code
    assert "cute.arch.load(y.iterator" not in code
    body = _vloop_body(code)
    assert body.index("cute.arch.atomic_add((a.iterator") < body.index(
        "c = (y.iterator"
    )
    assert "_cute_store_u32_vec(b.iterator" in code


def test_load_only_kernel_keeps_every_vector_op() -> None:
    code = _code(_load_only, _flat_args(), _flat_config())
    assert "cute.arch.load(x.iterator" in code
    assert "cute.arch.load(y.iterator" in code
    assert "_cute_store_u32_vec(out.iterator" in code
    body = _vloop_body(code)
    assert ".load()" not in body
    assert ".store(" not in body
    assert len(re.findall(r"_tile_unroll_vec_\d+_\d+ = cute.arch.load", code)) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_atomic_then_load_runtime_exact() -> None:
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device=DEVICE)
    out = torch.zeros(1024, dtype=torch.float32, device=DEVICE)
    bound = _col_sum_atomic_then_lane_load._bind_isolated((x, out))
    _assert_grid_atomic_then_load_stays_scalar(bound.to_code(_grid_config()))
    run = bound.compile_config(_grid_config())
    res = run(x, out)
    # ``c - old`` is exactly one in every lane; the column sums add to it.
    torch.testing.assert_close(
        res, 1.0 + x.to(torch.float32).sum(dim=0), rtol=1e-4, atol=1e-2
    )
    assert torch.equal(out, torch.ones_like(out))
    zeros = torch.zeros_like(x)
    out.zero_()
    assert torch.equal(run(zeros, out), torch.ones(1024, device=DEVICE))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_device_loop_atomic_then_load_runtime_exact() -> None:
    x = torch.empty(64, 1024, dtype=torch.float32, device=DEVICE)
    out = torch.zeros(64, 1024, dtype=torch.float32, device=DEVICE)
    bound = _row_atomic_then_load._bind_isolated((x, out))
    _assert_device_loop_atomic_then_load_stays_scalar(bound.to_code(_row_config()))
    run = bound.compile_config(_row_config())
    res = run(x, out)
    assert torch.equal(res, torch.ones_like(res))
    assert torch.equal(out, torch.ones_like(out))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("kernel", "shape", "config"),
    [
        (_row_store_then_reload, (64, 1024), _row_config()),
        (_flat_store_then_reload, (1024,), _flat_config()),
    ],
    ids=["device_loop", "grid"],
)
def test_store_then_reload_runtime_exact(
    kernel: Any, shape: tuple[int, ...], config: helion.Config
) -> None:
    x = torch.randn(*shape, dtype=torch.float32, device=DEVICE)
    out = torch.zeros(*shape, dtype=torch.float32, device=DEVICE)
    bound = kernel._bind_isolated((x, out))
    _assert_store_then_reload_stays_scalar(bound.to_code(config))
    run = bound.compile_config(config)
    res = run(x, out)
    torch.testing.assert_close(out, x + 1.0, rtol=0, atol=0)
    torch.testing.assert_close(res, (x + 1.0) * 2.0, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_store_then_reload_through_alias_runtime_exact() -> None:
    x = torch.randn(64, 1024, dtype=torch.float32, device=DEVICE)
    out = torch.zeros(64, 1024, dtype=torch.float32, device=DEVICE)
    bound = _row_store_then_reload_alias._bind_isolated((x, out, out))
    _assert_store_then_reload_stays_scalar(
        bound.to_code(_row_config()), reloaded="out2", expect_barrier=False
    )
    run = bound.compile_config(_row_config())
    res = run(x, out, out)
    torch.testing.assert_close(out, x + 1.0, rtol=0, atol=0)
    torch.testing.assert_close(res, (x + 1.0) * 2.0, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lane_invariant_store_after_a_collected_store_runtime_exact() -> None:
    x = torch.randn(1024, dtype=torch.float32, device=DEVICE)
    out = torch.zeros(1024, dtype=torch.float32, device=DEVICE)
    bound = _grid_store_then_invariant_store._bind_isolated((x, out))
    _assert_invariant_store_follows_the_flush(bound.to_code(_flat_config()))
    bound.compile_config(_flat_config())(x, out)
    # Every tile zeroes its first element after the flushed copy.
    expected = x + 1.0
    expected[::16] = 0.0
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "kernel",
    [_grid_store_then_uniform_atomic, _nested_store_then_uniform_atomic],
    ids=["grid", "device_loop"],
)
def test_store_then_uniform_atomic_runtime_exact(kernel: Any) -> None:
    x = torch.randint(-1000, 1000, (8, 256), dtype=torch.int32, device=DEVICE)
    out = torch.zeros_like(x)
    bound = kernel._bind_isolated((x, out))
    _assert_barrier_between_the_scalar_store_and_the_atomic(
        bound.to_code(_two_axis_config())
    )
    bound.compile_config(_two_axis_config())(x, out)
    # Every element holds its stored value; a tile's first row also the one
    # increment issued for it once the barrier has ordered every store.
    expected = x + 1
    expected[::2] += 1
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_store_then_atomic_runtime_exact() -> None:
    x = torch.randn(1024, dtype=torch.float32, device=DEVICE)
    out = torch.zeros(1024, dtype=torch.float32, device=DEVICE)
    bound = _grid_store_then_atomic._bind_isolated((x, out))
    _assert_store_then_atomic_stays_scalar(bound.to_code(_flat_config()))
    run = bound.compile_config(_flat_config())
    res = run(x, out)
    # ``old`` is the value the lane stored a statement earlier.
    torch.testing.assert_close(res, x + 1.0, rtol=0, atol=0)
    torch.testing.assert_close(out, (x + 1.0) + 1.0, rtol=0, atol=0)
