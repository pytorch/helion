"""CPU source-generation coverage of MMA-DAG maxima and vector carry seeds."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._compiler.cute import cute_mma_dag as dag
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.cpu_only


def _maxima(
    a: torch.Tensor, b: torch.Tensor, axis: int, scalar: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    # Render-only fixture: scalar_out assumes the single M tile used below.
    m, n = a.size(0), hl.specialize(b.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    scalar_out = torch.empty([1], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(a.size(1)):
            value = hl.dot(a[tm, tk], b[tk, :], acc=value)
        maximum = torch.amax(value - 7.0, dim=axis)
        if axis == 0:
            out[tm, :] = value - maximum[None, :]
        else:
            out[tm, :] = value - maximum[:, None]
        if scalar:
            scalar_out[0] = torch.amax(maximum, dim=0)
    return out, scalar_out


def _running_max(a: torch.Tensor, b: torch.Tensor, initial: float) -> torch.Tensor:
    m, steps = a.size(0), b.size(0)
    hl.specialize(a.size(1))
    hl.specialize(b.size(2))
    out = torch.empty([m], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        maximum = hl.full([tm], initial, dtype=torch.float32)
        for step in hl.grid(steps):
            value = hl.dot(a[tm, :], b[step, :, :])
            maximum = torch.maximum(maximum, torch.amax(value, dim=1))
        out[tm] = maximum
    return out


def _running_column_max(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    steps, m, n = a.size(0), hl.specialize(a.size(1)), b.size(1)
    hl.specialize(a.size(2))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tn in hl.tile(n):
        maximum = hl.full([tn], float("-inf"), dtype=torch.float32)
        value = hl.zeros([m, tn], dtype=torch.float32)
        for step in hl.grid(steps):
            value = hl.dot(a[step, :, :], b[:, tn])
            maximum = torch.maximum(maximum, torch.amax(value, dim=0))
        out[:, tn] = value - maximum[None, :]
    return out


def _render(
    fn: Callable[..., object],
    args: tuple[object, ...],
    mode: str,
    blocks: list[int],
    *,
    cluster: int = 1,
    epilogue: str = "row",
) -> tuple[dag.MmaDagProgram, str]:
    with cpu_target():
        bound = cpu_bind(helion.kernel(fn, backend="cute", static_shapes=True), args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = dag.match_mma_dag(bound.host_function.device_ir.graphs)
        assert program is not None
        config = helion.Config(
            block_sizes=blocks,
            cute_mma_dag=mode,
            cute_mma_dag_warps=4,
            cute_mma_dag_stages=2,
            cute_mma_dag_loads="lane" if mode == "warp" else "tma",
            cute_mma_dag_cluster=cluster,
            cute_mma_dag_epilogue=epilogue,
        )
        with bound.env:
            config = bound.config_spec.normalized_config(config)
        source = bound.to_code(config)
    assert "_cute_grouped_reduce_shared_two_stage" not in source
    assert "_helion_mma_dag" in source
    return program, source


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize(("axis", "scalar"), [(0, False), (1, False), (1, True)])
def test_fragment_and_vector_maxima(mode: str, axis: int, scalar: bool) -> None:
    a = torch.empty((64, 64), dtype=torch.bfloat16)
    b = torch.empty((64, 64), dtype=torch.bfloat16)
    program, source = _render(
        _maxima, (a, b, hl.constexpr(axis), hl.constexpr(scalar)), mode, [64, 32]
    )
    assert [step.kind for step in program.reduces] == ["max"] * (1 + scalar)
    assert program.reduces[0].orient == ("cols" if axis == 0 else "rows")
    if scalar:
        assert program.reduces[1].orient == "scalar"
    assert "propagate_nan=True" in source
    assert ".lane_group_sum(" not in source
    assert ".octet_sum(" not in source
    if mode == "warp":
        assert (".octet_max(" if axis == 0 else ".lane_group_max(") in source
    elif axis == 0 or scalar:
        assert ".lane_group_max(" in source


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("initial", [2.5, float("-inf"), float("inf")])
def test_running_max_preserves_scalar_carry_initializer(
    mode: str, initial: float
) -> None:
    a = torch.empty((64, 64), dtype=torch.bfloat16)
    b = torch.empty((3, 64, 64), dtype=torch.bfloat16)
    program, source = _render(_running_max, (a, b, hl.constexpr(initial)), mode, [64])
    (loop,) = program.loops
    (carry,) = loop.vec_carries
    assert carry.init_vec is None
    assert carry.init_fill == initial
    literal = "2.5" if initial == 2.5 else f"float('{initial}')"
    register = rf"md_r{carry.slot}_\d+_\d+" if mode == "warp" else f"md_vv{carry.slot}"
    assert re.search(rf"{register} = cutlass.Float32\({re.escape(literal)}\)", source)
    assert "Float32(-inf)" not in source


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
def test_scalar_fill_carry_acquires_column_layout(mode: str) -> None:
    a = torch.empty((3, 64, 64), dtype=torch.bfloat16)
    b = torch.empty((64, 64), dtype=torch.bfloat16)
    program, source = _render(_running_column_max, (a, b), mode, [64])
    (carry,) = program.loops[0].vec_carries
    assert program.vec_slots[carry.slot].orient == "cols"
    assert program.vec_frag_slots[carry.slot]
    assert "cutlass.Float32(float('-inf'))" in source


def test_scalar_max_preserves_ragged_input_mask() -> None:
    a = torch.empty((24, 64), dtype=torch.bfloat16)
    b = torch.empty((64, 64), dtype=torch.bfloat16)
    _, source = _render(
        _maxima, (a, b, hl.constexpr(1), hl.constexpr(True)), "warp", [32, 32]
    )
    assert re.search(
        r"md_r\d+_\d+_\d+ if .* < 24 else cutlass.Float32\(float\('-inf'\)\)", source
    )


def test_column_scalar_max_masks_unused_tcgen05_lanes() -> None:
    a = torch.empty((128, 64), dtype=torch.bfloat16)
    b = torch.empty((64, 64), dtype=torch.bfloat16)
    _, source = _render(
        _maxima, (a, b, hl.constexpr(0), hl.constexpr(True)), "tcgen05", [128, 32]
    )
    # Inactive lanes keep the max identity, and only live lanes evaluate
    # the vector expression. The collective must still run on every lane.
    assert re.search(
        r"(?m)^(?P<indent>[ ]+)(?P<value>md_s(?:_\d+)?) = "
        r"cutlass.Float32\(float\('-inf'\)\)\n"
        r"(?P=indent)if md_row128 < 64:\n"
        r"(?P=indent)    (?P=value) = md_vv\d+\n"
        r"(?P=indent)md_s(?:_\d+)? = _helion_mma_dag\.lane_group_max"
        r"\((?P=value), 32\)",
        source,
    )


def _two_maxima(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n = a.size(0), hl.specialize(b.size(1))
    out = torch.empty([m, n], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        value = hl.zeros([tm, n], dtype=torch.float32)
        for tk in hl.tile(a.size(1)):
            value = hl.dot(a[tm, tk], b[tk, :], acc=value)
        # This resembles the centered-variance fusion structurally, but both
        # reductions are max, so each must exchange its own partials.
        center = torch.amax(value, dim=1, keepdim=True) / n
        scale = torch.amax((value - center) ** 2, dim=1, keepdim=True)
        out[tm, :] = value / scale
    return out


@pytest.mark.parametrize("epilogue", ["row", "streamed"])
def test_cluster_maxima_do_not_use_sum_variance_fusion(epilogue: str) -> None:
    a = torch.empty((128, 64), dtype=torch.bfloat16)
    b = torch.empty((64, 512), dtype=torch.bfloat16)
    program, source = _render(
        _two_maxima, (a, b), "tcgen05", [128, 32], cluster=2, epilogue=epilogue
    )
    assert [step.kind for step in program.reduces] == ["max", "max"]
    assert source.count(".mbar_wait_cluster(") == 2
    assert "propagate_nan=True" in source
    assert re.search(r"md_s(?:_\d+)? = md_s(?:_\d+)? \+ md_s(?:_\d+)?", source) is None


def _scalar_broadcast_before_vector_link(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    m, steps = a.size(0), b.size(0)
    hl.specialize(a.size(1))
    hl.specialize(b.size(2))
    out = torch.empty([m], dtype=torch.float32, device=a.device)
    for tm in hl.tile(m):
        carry = hl.zeros([tm], dtype=torch.float32)
        for step in hl.grid(steps):
            value = hl.dot(a[tm, :], b[step, :, :])
            rows = torch.sum(value, dim=1)
            scalar = torch.sum(rows, dim=0)
            carry = carry * scalar + rows
        out[tm] = carry
    return out


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
def test_scalar_broadcast_does_not_set_vector_carry_layout(mode: str) -> None:
    a = torch.empty((64, 64), dtype=torch.bfloat16)
    b = torch.empty((3, 64, 64), dtype=torch.bfloat16)
    program, _ = _render(_scalar_broadcast_before_vector_link, (a, b), mode, [64])
    (carry,) = program.loops[0].vec_carries
    assert program.vec_slots[carry.slot].orient == "rows"
    assert program.vec_frag_slots[carry.slot]
