"""Source generation and native checks for integer MMA-DAG loop bounds."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._compiler.cute import cute_mma_dag as dag
from helion._testing import DEVICE
from helion._testing import pairwise
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.cpu_only


def _bounded_matmul(
    a: torch.Tensor, b: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(b.size(2))
    hl.specialize(a.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        start = torch.clamp(bounds[0] // 2, min=0)
        stop = torch.minimum(bounds[1] + 1, bounds[2])
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for step in hl.grid(start, stop):
            acc = hl.dot(a[tm, :], b[step, :, :], acc=acc)
        out[tm, :] = acc
    return out


def _offset_bounded_matmul(
    a: torch.Tensor, b: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(b.size(1))
    hl.specialize(a.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        start = bounds[0] + 1
        stop = bounds[1] + 1
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for _step in hl.grid(start, stop):
            acc = hl.dot(a[tm, :], b[:, :], acc=acc)
        out[tm, :] = acc
    return out


def _direct_bounded_matmul(
    a: torch.Tensor, b: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(b.size(1))
    hl.specialize(a.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        start = bounds[0]
        stop = bounds[1]
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for _step in hl.grid(start, stop):
            acc = hl.dot(a[tm, :], b[:, :], acc=acc)
        out[tm, :] = acc
    return out


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize(
    ("dtype", "kind"),
    [
        (torch.int32, "composed"),
        (torch.int64, "composed"),
        (torch.int64, "offset"),
        (torch.int64, "direct"),
    ],
)
def test_composed_uniform_integer_loop_bounds(
    mode: str, dtype: torch.dtype, kind: str
) -> None:
    fn = {
        "composed": _bounded_matmul,
        "offset": _offset_bounded_matmul,
        "direct": _direct_bounded_matmul,
    }[kind]
    args = (
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty(
            (4, 64, 64) if kind == "composed" else (64, 64), dtype=torch.bfloat16
        ),
        torch.empty((3,), dtype=dtype),
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(
                fn,
                backend="cute",
                static_shapes=True,
            ),
            args,
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = dag.match_mma_dag(bound.host_function.device_ir.graphs)
            assert program is not None
            (loop,) = program.loops
            assert loop.static_trip is None
            assert loop.begin in program.uniform_values
            assert loop.end in program.uniform_values
            assert len(program.scalar_loads) == (3 if kind == "composed" else 2)
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64],
                    cute_mma_dag=mode,
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=2,
                    cute_mma_dag_loads="lane" if mode == "warp" else "tma",
                )
            )
        source = bound.to_code(config)
    assert "_cute_grouped_reduce_shared_two_stage" not in source
    assert "_helion_mma_dag" in source
    assert ("md_scalar" if kind == "direct" else "md_bound") in source
    assert (".ldg_s32(" if dtype == torch.int32 else ".ldg_s64(") in source
    if dtype == torch.int32:
        assert "md_bound = cutlass.Int32(" in source
    if dtype == torch.int64:
        if kind != "direct":
            assert "md_bound = cutlass.Int64(" in source
        assert "cutlass.Int32(_helion_mma_dag.ldg_s64(" not in source


def _float_bound(
    a: torch.Tensor, b: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(b.size(1))
    hl.specialize(a.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        stop = (bounds[0].float() * 0.5).to(torch.int64)
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for _step in hl.grid(0, stop):
            acc = hl.dot(a[tm, :], b[:, :], acc=acc)
        out[tm, :] = acc
    return out


def _iteration_bound(
    a: torch.Tensor, b: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(b.size(1))
    hl.specialize(a.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for outer in hl.grid(2):
            stop = bounds[outer] + 1
            for _step in hl.grid(0, stop):
                acc = hl.dot(a[tm, :], b[:, :], acc=acc)
        out[tm, :] = acc
    return out


def _loop_local_invariant_bound(
    a: torch.Tensor, b: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(b.size(1))
    hl.specialize(a.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for _outer in hl.grid(bounds[0]):
            # Hoisting this load needs proof that bounds has a second element,
            # even when the outer loop has zero trips.
            stop = bounds[1] + 1
            for _step in hl.grid(stop):
                acc = hl.dot(a[tm, :], b[:, :], acc=acc)
        out[tm, :] = acc
    return out


@pytest.mark.parametrize(("size", "static_shapes"), [(1, True), (2, False)])
def test_loop_local_bound_without_bounds_proof_is_not_speculated(
    size: int, static_shapes: bool
) -> None:
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(
                _loop_local_invariant_bound,
                backend="cute",
                static_shapes=static_shapes,
            ),
            (
                torch.empty((64, 64), dtype=torch.bfloat16),
                torch.empty((64, 64), dtype=torch.bfloat16),
                torch.empty((size,), dtype=torch.int64),
            ),
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            assert dag.match_mma_dag(bound.host_function.device_ir.graphs) is None


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
def test_loop_local_bound_with_constant_in_bounds_address(mode: str) -> None:
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(
                _loop_local_invariant_bound, backend="cute", static_shapes=True
            ),
            (
                torch.empty((64, 64), dtype=torch.bfloat16),
                torch.empty((64, 64), dtype=torch.bfloat16),
                torch.empty((2,), dtype=torch.int64),
            ),
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = dag.match_mma_dag(bound.host_function.device_ir.graphs)
            assert program is not None
            assert len(program.scalar_loads) == 2
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64],
                    cute_mma_dag=mode,
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=2,
                )
            )
        source = bound.to_code(config)
    assert "_helion_mma_dag" in source
    assert "md_bound = cutlass.Int64(" in source


def _loop_local_divided_bound(
    a: torch.Tensor, b: torch.Tensor, bounds: torch.Tensor, divisor: hl.constexpr
) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(b.size(1))
    hl.specialize(a.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for _outer in hl.grid(bounds[0]):
            if divisor is None:
                stop = bounds[1] // bounds[2]
            else:
                stop = bounds[1] // divisor
            for _step in hl.grid(stop):
                acc = hl.dot(a[tm, :], b[:, :], acc=acc)
        out[tm, :] = acc
    return out


@pytest.mark.parametrize("divisor", [None, -1, 2])
def test_loop_local_bound_requires_safe_integer_arithmetic(divisor: int | None) -> None:
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(
                _loop_local_divided_bound, backend="cute", static_shapes=True
            ),
            (
                torch.empty((64, 64), dtype=torch.bfloat16),
                torch.empty((64, 64), dtype=torch.bfloat16),
                torch.empty((3,), dtype=torch.int64),
                divisor,
            ),
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = dag.match_mma_dag(bound.host_function.device_ir.graphs)
        assert (program is not None) is (divisor == 2)


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
def test_loop_local_scalar_bound_matches_reference_on_device(mode: str) -> None:
    if torch.version.cuda is None or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("Both MMA modes require a Blackwell test target")
    generator = torch.Generator(device=DEVICE).manual_seed(41)
    a = torch.randn((64, 64), device=DEVICE, dtype=torch.bfloat16, generator=generator)
    b = torch.randn((64, 64), device=DEVICE, dtype=torch.bfloat16, generator=generator)
    bounds = torch.tensor([0, 1], device=DEVICE, dtype=torch.int64)
    bound = helion.kernel(
        _loop_local_invariant_bound, backend="cute", static_shapes=True
    ).bind((a, b, bounds))
    with bound.env:
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[64],
                cute_mma_dag=mode,
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
            )
        )
    compiled = bound.compile_config(config)
    product = a.float() @ b.float()
    # Reuse one compiled launch across zero and nonzero trip counts, including
    # a return to zero after nonzero carries have occupied its registers.
    for outer in (0, 1, 3, 0):
        bounds[0] = outer
        actual = compiled(a, b, bounds)
        torch.testing.assert_close(actual, product * (outer * 2), atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("fn", [_float_bound, _iteration_bound])
def test_uniform_bound_requires_integer_invariant_inputs(
    fn: Callable[..., object],
) -> None:
    args = (
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((2,), dtype=torch.int64),
    )
    with cpu_target():
        bound = cpu_bind(helion.kernel(fn, backend="cute", static_shapes=True), args)
        assert bound.host_function is not None
        with bound.env, bound.host_function, pytest.raises(dag._NotMine) as error:
            dag._Matcher(bound.host_function.device_ir.graphs)._match()
    assert any(entry.name == "uniform_value" for entry in error.traceback)


def _filled_bounded_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    bounds: torch.Tensor,
    dynamic: hl.constexpr,
) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(b.size(1))
    hl.specialize(a.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        start = bounds[0]
        if dynamic:
            stop = hl.full([], bounds[1], dtype=torch.int64)
        else:
            stop = hl.full([], 2**31 + 1, dtype=torch.int64)
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for _step in hl.grid(start, stop):
            acc = hl.dot(a[tm, :], b[:, :], acc=acc)
        out[tm, :] = acc
    return out


def _nested_bounded_matmul(
    a: torch.Tensor, b: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(b.size(1))
    hl.specialize(a.size(1))
    out = torch.empty([2, m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        start = bounds[0] + 1
        stop = bounds[1] + 1
        for outer in hl.grid(2):
            acc = hl.zeros([tm, n], dtype=torch.float32)
            for _step in hl.grid(start, stop):
                acc = hl.dot(a[tm, :], b[:, :], acc=acc)
            out[outer, tm, :] = acc
    return out


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("kind", ["literal", "dynamic", "nested"])
def test_integer_fill_and_nested_bound_aliases(mode: str, kind: str) -> None:
    args: tuple[object, ...] = (
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((2,), dtype=torch.int64),
    )
    fn: Callable[..., object]
    if kind == "nested":
        fn = _nested_bounded_matmul
    else:
        fn = _filled_bounded_matmul
        args = (*args, kind == "dynamic")
    with cpu_target():
        bound = cpu_bind(helion.kernel(fn, backend="cute", static_shapes=True), args)
        assert bound.host_function is not None
        if kind == "nested":
            with bound.env, bound.host_function:
                program = dag.match_mma_dag(bound.host_function.device_ir.graphs)
            assert program is not None
            assert any(
                isinstance(value, torch.fx.Node) and program.resolve(value) is not value
                for loop in program.loops
                for value in (loop.begin, loop.end)
            )
        config = bound.config_spec.default_config()
        config.config.update(
            cute_mma_dag=mode,
            cute_mma_dag_warps=4,
            cute_mma_dag_stages=2,
            cute_mma_dag_loads="lane" if mode == "warp" else "tma",
            block_sizes=[64],
        )
        with bound.env:
            config = bound.config_spec.normalized_config(config)
        source = bound.to_code(config)
    assert "_helion_mma_dag" in source
    assert "cutlass.Float32(2147483649" not in source
    if kind == "literal":
        assert "cutlass.Int64(2147483649)" in source


def _nested_tma_bounded_matmul(
    a: torch.Tensor, b: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(b.size(2))
    hl.specialize(a.size(1))
    out = torch.empty([2, m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        start = bounds[0] + 1
        stop = bounds[1] + 1
        for outer in hl.grid(2):
            acc = hl.zeros([tm, n], dtype=torch.float32)
            for step in hl.grid(start, stop):
                acc = hl.dot(a[tm, :], b[step, :, :], acc=acc)
            out[outer, tm, :] = acc
    return out


def _loadfree_bounded_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(b.size(2))
    hl.specialize(a.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        start = hl.full([], 0, dtype=torch.int64)
        stop = hl.full([], 2, dtype=torch.int64)
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for step in hl.grid(start, stop):
            acc = hl.dot(a[tm, :], b[step, :, :], acc=acc)
        out[tm, :] = acc
    return out


def _lead_bounded_matmul(
    a: torch.Tensor, b: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    batches, m, _k = a.shape
    n = hl.specialize(b.size(2))
    hl.specialize(a.size(2))
    out = torch.empty([batches, m, n], dtype=torch.float32, device=a.device)
    for tb, tm in hl.tile([batches, m], block_size=[1, None]):
        start = torch.where(bounds[0] > 0, bounds[1], bounds[2])
        stop = bounds[3] + tb.id * 2
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for step in hl.grid(start, stop):
            acc = hl.dot(a[tb.id, tm, :], b[step, :, :], acc=acc)
        out[tb.id, tm, :] = acc
    return out


def _shared_symint_bounded_matmul(
    a: torch.Tensor, b: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    batches, m, _k = a.shape
    n = hl.specialize(b.size(2))
    hl.specialize(a.size(2))
    out = torch.empty([batches, m, n], dtype=torch.float32, device=a.device)
    for tb, tm in hl.tile([batches, m], block_size=[1, None]):
        start = tb.id * 2 + 1
        stop = bounds[0] + start
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for step in hl.grid(start, stop):
            acc = hl.dot(a[tb.id, tm, :], b[step, :, :], acc=acc)
        out[tb.id, tm, :] = acc
    return out


# Each case targets a distinct scheduler or value path: nested warp TMA's
# carried ring counter, rejection of dynamic persistent tcgen05, and symbolic
# leading coordinates / boolean selections on both arithmetic emitters.
# The compound-start case uses lane copies on warp and TMA on tcgen05.
@pytest.mark.parametrize(
    ("mode", "kind"),
    [
        ("warp", "nested_tma"),
        ("tcgen05", "loadfree"),
        ("warp", "lead"),
        ("tcgen05", "lead"),
        ("warp", "shared_symint"),
        ("tcgen05", "shared_symint"),
    ],
)
def test_uniform_bound_rings_and_symbolic_inputs(mode: str, kind: str) -> None:
    fn = {
        "nested_tma": _nested_tma_bounded_matmul,
        "loadfree": _loadfree_bounded_matmul,
        "lead": _lead_bounded_matmul,
        "shared_symint": _shared_symint_bounded_matmul,
    }[kind]
    args: tuple[object, ...] = (
        torch.empty(
            (2, 128, 64) if kind in ("lead", "shared_symint") else (128, 64),
            dtype=torch.bfloat16,
        ),
        torch.empty((6, 64, 64), dtype=torch.bfloat16),
    )
    if kind != "loadfree":
        args = (*args, torch.empty((4,), dtype=torch.int64))
    with cpu_target():
        bound = cpu_bind(helion.kernel(fn, backend="cute", static_shapes=True), args)
        config = bound.config_spec.default_config()
        config.config.update(
            cute_mma_dag=mode,
            cute_mma_dag_warps=4,
            cute_mma_dag_stages=3,
            cute_mma_dag_loads="lane"
            if kind == "shared_symint" and mode == "warp"
            else "tma",
            block_sizes=[64],
        )
        if kind == "loadfree":
            with bound.env, pytest.raises(helion.exc.InvalidConfig, match="persistent"):
                bound.config_spec.normalized_config(
                    helion.Config(**{**config.config, "cute_mma_dag_persistent": 1})
                )
        with bound.env:
            config = bound.config_spec.normalized_config(config)
        source = bound.to_code(config)
    assert "_helion_mma_dag" in source
    if kind == "shared_symint" and mode == "warp":
        assert "cp_async_16" in source
        assert "md_stage = md_ring0 % 3" in source
        assert "cutlass.range(1 + 2 * (tile_offset_0 // _BLOCK_SIZE_0)," in source
    else:
        assert "md_tma" in source
    if kind == "nested_tma":
        assert "md_cnt1 = md_cnt1 + 1" in source
        assert "md_stage1 = md_cnt1 % 3" in source
    elif kind == "loadfree":
        assert "md_npid" not in source
    else:
        assert "md_bound = cutlass.Int64(" in source
    if kind == "shared_symint" and mode == "tcgen05":
        assert "md_bound - (1 + 2 * (tile_offset_0 // _BLOCK_SIZE_0))" in source


def _aten_filled_bounded_matmul(
    a: torch.Tensor, b: torch.Tensor, bounds: torch.Tensor, fill: hl.constexpr
) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(b.size(1))
    hl.specialize(a.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        start = bounds[1]
        stop = torch.full_like(bounds[0], fill) * 2
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for _step in hl.grid(start, stop):
            acc = hl.dot(a[tm, :], b[:, :], acc=acc)
        out[tm, :] = acc
    return out


@pytest.mark.parametrize(
    ("mode", "dtype", "fill"),
    pairwise(
        mode=["warp", "tcgen05"], dtype=[torch.int32, torch.int64], fill=[1.9, -1.9]
    ),
)
def test_aten_integer_fill_is_cast_before_arithmetic(
    mode: str, dtype: torch.dtype, fill: float
) -> None:
    args = (
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((2,), dtype=dtype),
        fill,
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(
                _aten_filled_bounded_matmul, backend="cute", static_shapes=True
            ),
            args,
        )
        with bound.env:
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64],
                    cute_mma_dag=mode,
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=2,
                )
            )
        source = bound.to_code(config)
    width = 32 if dtype == torch.int32 else 64
    assert re.search(rf"v_\d+ = cutlass\.Int{width}\({int(fill)}\)", source)
    assert f"{fill} *" not in source


def _narrow_fill_epilogue(
    a: torch.Tensor, b: torch.Tensor, dtype: hl.constexpr
) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(b.size(1))
    hl.specialize(a.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for _step in hl.grid(2):
            acc = hl.dot(a[tm, :], b[:, :], acc=acc)
        out[tm, :] = acc + torch.full_like(acc, 1, dtype=dtype)
    return out


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.uint8])
def test_narrow_epilogue_fills_keep_existing_lowering(
    mode: str, dtype: torch.dtype
) -> None:
    args = (
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((64, 64), dtype=torch.bfloat16),
        dtype,
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(_narrow_fill_epilogue, backend="cute", static_shapes=True),
            args,
        )
        with bound.env:
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64],
                    cute_mma_dag=mode,
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=2,
                )
            )
        source = bound.to_code(config)
    assert "_helion_mma_dag" in source


def _dual_window_bound(
    a: torch.Tensor,
    data: torch.Tensor,
    offset: torch.Tensor,
    limit: torch.Tensor,
    entry_limit: hl.constexpr,
) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(data.size(1))
    width = hl.specialize(a.size(1))
    out = torch.empty([m, n], device=a.device, dtype=torch.float32)
    for tm in hl.tile(m):
        begin = offset[0]
        end = begin + 2
        idx = hl.arange(width)
        acc = hl.zeros([tm, n], dtype=torch.float32)
        if entry_limit:
            cached_limit = limit[0]
        else:
            cached_limit = hl.full([], 0, dtype=torch.int64)
        for _step in hl.grid(begin, end):
            tile = hl.load(
                data,
                [begin + idx, slice(None)],
                extra_mask=(idx < (cached_limit if entry_limit else limit[0]))[:, None],
            )
            tile = (tile.float() * 1.25).to(data.dtype)
            acc = hl.dot(a[tm, :], tile, acc=acc)
        out[tm, :] = acc
    return out


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("entry_limit", [False, True])
def test_same_int64_scalar_is_a_window_offset_and_loop_bound(
    mode: str, entry_limit: bool
) -> None:
    args = (
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((256, 128), dtype=torch.bfloat16),
        torch.empty((1,), dtype=torch.int64),
        torch.empty((1,), dtype=torch.int64),
        entry_limit,
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(_dual_window_bound, backend="cute", static_shapes=True), args
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = dag._Matcher(bound.host_function.device_ir.graphs)._match()
            offset = next(
                access for access in program.scalar_loads if access.name == "offset"
            )
            assert offset.node in program.uniform_values
            assert any(
                window.offset.node is offset.node
                for access in program.accesses
                for _position, window in access.windows
            )
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64],
                    cute_mma_dag=mode,
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=3,
                )
            )
        source = bound.to_code(config)
    assert "md_scalar = _helion_mma_dag.ldg_s64(" in source


def _aten_symbolic_fill_bound(
    a: torch.Tensor, b: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    m, n = hl.specialize(a.size(0)), hl.specialize(b.size(1))
    hl.specialize(a.size(1))
    hl.specialize(b.size(0))
    out = torch.empty([m, n], device=a.device, dtype=torch.float32)
    for tm in hl.tile(m):
        stop = torch.full_like(bounds[0], bounds.size(0))
        acc = hl.zeros([tm, n], dtype=torch.float32)
        for _step in hl.grid(stop):
            acc = hl.dot(a[tm, :], b[:, :], acc=acc)
        out[tm, :] = acc
    return out


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
def test_aten_integer_fill_uses_dynamic_host_shape(mode: str) -> None:
    args = (
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((3,), dtype=torch.int64),
    )
    with cpu_target():
        bound = cpu_bind(
            helion.kernel(
                _aten_symbolic_fill_bound, backend="cute", static_shapes=False
            ),
            args,
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = dag._Matcher(bound.host_function.device_ir.graphs)._match()
            fills = [
                node
                for node in program.uniform_values
                if node.target is torch.ops.aten.full.default
            ]
            assert len(fills) == 1 and isinstance(fills[0].args[1], torch.fx.Node)
            config = bound.config_spec.normalized_config(
                helion.Config(
                    block_sizes=[64],
                    cute_mma_dag=mode,
                    cute_mma_dag_warps=4,
                    cute_mma_dag_stages=2,
                )
            )
        source = bound.to_code(config)
    assert "_helion_mma_dag" in source
