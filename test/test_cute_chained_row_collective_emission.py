from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_pipeline import _config as _pipeline_config
import helion
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_preparation_pipeline as pipeline_module
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_pointwise_unroll import BoundedProducerUnroll
from helion._compiler.cute.chained_row_collective_emission import (
    emit_row_collective_group,
)
from helion._compiler.cute.chained_row_collectives import plan_row_collective_group
from helion._compiler.cute.chained_vector_stage import VectorStageOperand
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_loop(
    a,
    b,
    initial,
    keepdim: hl.constexpr,
    nonlinear: hl.constexpr,
    cross: hl.constexpr,
    mask_enabled: hl.constexpr = True,  # pyrefly: ignore [bad-function-definition]
):
    steps, size, width = a.shape
    history = torch.empty((steps, size, size), dtype=torch.float32, device=a.device)
    final = torch.empty_like(initial)
    for rows in hl.tile(size, block_size=16):
        state = initial[rows, rows]
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(width)
            if mask_enabled:
                raw_x = hl.load(
                    a, [step.id, rows, kk], extra_mask=(rows.index % 3 != 0)[:, None]
                ).float()
            else:
                raw_x = a[step.id, rows, kk].float()
            raw_y = b[step.id, rows, kk].float()
            x = (raw_x * 0.125).to(torch.float16).float()
            y = (raw_y * 0.25).to(torch.bfloat16).float()
            if nonlinear:
                x = torch.sigmoid(x)
                y = torch.exp(y)
            sx = torch.sum(x * x, dim=1, keepdim=keepdim)  # pyrefly: ignore [bad-argument-type]
            sy = torch.sum(y + y, dim=1, keepdim=keepdim)  # pyrefly: ignore [bad-argument-type]
            if keepdim:
                expanded_sx = sx
                expanded_sy = sy
            else:
                expanded_sx = sx[:, None]
                expanded_sy = sy[:, None]
            xout = x
            if cross:
                xout = x.T
            left = (xout + expanded_sx).to(a.dtype)
            right = (y + expanded_sy).to(a.dtype)
            first = hl.dot(left, right.T, out_dtype=torch.float32)
            second = hl.dot(
                left, (right.float() + 1).to(a.dtype).T, out_dtype=torch.float32
            )
            state = hl.dot(
                state.to(a.dtype),
                (first + second).to(a.dtype),
                acc=state,
                out_dtype=torch.float32,
            )
            history[step.id, rows, rows] = state
        final[rows, rows] = state
    return history, final


def _args(
    dtype=torch.bfloat16, width=64, *, keepdim=False, nonlinear=False, cross=False
):
    return (
        torch.empty((3, 16, width), dtype=dtype),
        torch.empty((3, 16, width), dtype=dtype),
        torch.empty((16, 16), dtype=torch.float32),
        keepdim,
        nonlinear,
        cross,
    )


def _config():
    return helion.Config(
        num_warps=16,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_group_contractions=True,
        cute_chained_warp_mma_rows=32,
        cute_chained_preparation_pipeline=True,
    )


def _operands(plan, candidate):
    group = candidate.stage.group
    result = []
    members = tuple(zip(group.stages, group.geometries, group.offsets, strict=True))
    for role in ("a", "b"):
        for stage, geometry, offset in members[:1] if role == "a" else members:
            arg, _ = geometry.operand(role, "row", "column")
            result.append(
                VectorStageOperand(
                    plan.dots[stage].args[arg],
                    geometry,
                    role,
                    candidate.shape,
                    f"chain_{group.stages[0]}_{role}",
                    offset if role == "b" else 0,
                )
            )
    return tuple(result)


def _capture(
    *, args=None, kernel=_row_loop, config=None, change=None, check=None, threads=128
):
    args = _args() if args is None else args
    config = _config() if config is None else config
    found = []
    original = pipeline_module._prepare

    def observe(cg, plan, pipeline, execution, vector, unroll, scratch):
        frame = pipeline.frame
        shapes = {
            node: chain._shape(node)
            for node in plan.region.nodes
            if isinstance(node.meta.get("val"), torch.Tensor)
        }
        published = {}
        for action in frame.actions:
            candidate = plan_row_collective_group(plan, frame, action.event, shapes)
            if candidate is not None:
                operands = _operands(plan, candidate)
                local = ChainedExecution(
                    threads, thread="test_thread", sync="test_barrier.arrive_and_wait()"
                )
                tracker = BoundedProducerUnroll(1)
                values: Any = (plan, frame, candidate, dict(published), operands)
                if change is not None:
                    values = change(*values)
                before = dict(values[3])
                lines = emit_row_collective_group(
                    cg, *values, execution=local, producer_unroll=tracker
                )
                assert values[3] == before
                if check is not None:
                    check(cg, *values, lines)
                found.append((lines, candidate, tracker))
                break
            for name in action.writes:
                buffer = next(item for item in frame.buffers if item.name == name)
                if buffer.node is not None:
                    published[buffer.node] = name
        return original(cg, plan, pipeline, execution, vector, unroll, scratch)

    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        with (
            bound.env.use_runtime_arg_values(_runtime_values(kernel, args)),
            patch.object(pipeline_module, "_prepare", observe),
        ):
            source = bound.to_code(config)
    assert len(found) == 1
    return *found[0], source


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width", [16, 48, 64, 128])
@pytest.mark.parametrize("keepdim", [False, True])
def test_original_row_order_typed_snapshots_keepdim_and_complete_ownership(
    dtype, width, keepdim
):
    lines, candidate, _, _ = _capture(args=_args(dtype, width, keepdim=keepdim))
    assert lines is not None
    text = "\n".join(lines)
    ast.parse(text)
    tag = f"chain_row_collective_{candidate.first_event}"
    physical_width = candidate.shape[1]
    parts = (physical_width + 31) // 32
    assert text.count(f"cute.make_rmem_tensor(({parts},), cutlass.Float32)") == 2
    assert f"{tag}_column = test_thread % 32 + {tag}_part * 32" in text
    assert text.count("= cutlass.Float32(0)") == 2
    for offset in (16, 8, 4, 2, 1):
        assert text.count(f"offset={offset})") == 2
    assert text.count("cute.arch.shuffle_sync(") == 2
    assert lines[-2:] == ["test_barrier.arrive_and_wait()"] * 2
    assert (
        "alloc_smem" not in text
        and "fence_view" not in text
        and "cute.copy" not in text
    )
    for buffer in candidate.buffers:
        coords = f"{tag}_row, 0" if keepdim else f"{tag}_row"
        assert f"{buffer.name}[{coords}]" in text
    owners = [
        (thread // 32 + step * 4, thread % 32 + part * 32)
        for thread in range(128)
        for step in range(4)
        for part in range(parts)
        if thread % 32 + part * 32 < physical_width
    ]
    assert len(owners) == len(set(owners)) == 16 * physical_width


@pytest.mark.parametrize("threads", [128, 256, 384])
def test_original_nonlinear_roots_are_retained_without_formula_recognition(threads):
    lines, candidate, _, _ = _capture(args=_args(nonlinear=True), threads=threads)
    assert lines is not None
    text = "\n".join(lines)
    assert text.count("cute.make_rmem_tensor") == 2
    assert "cutlass.Float16" in text and "cutlass.BFloat16" in text
    assert "test_thread % 32 == 0" in text
    tag = f"chain_row_collective_{candidate.first_event}"
    assert f"test_thread // 32 + {tag}_step * {threads // 32}" in text


@pytest.mark.parametrize(
    "reason",
    [
        "offset",
        "shape",
        "geometry",
        "target",
        "omit",
        "stale",
        "missing_input",
        "domain",
        "sparse",
    ],
)
def test_rejections_do_not_publish_boundaries_or_activate_unroll(reason):
    def change(plan, frame, candidate, boundaries, operands):
        if reason in ("offset", "shape", "geometry", "target"):
            changes = {
                "offset": {"offset": 1},
                "shape": {"shape": (16, 32)},
                "geometry": {"geometry": replace(operands[0].geometry, transpose=True)},
                "target": {"target": "different_target"},
            }[reason]
            operands = (replace(operands[0], **changes), *operands[1:])
        elif reason == "omit":
            operands = operands[:-1]
        elif reason == "stale":
            candidate = replace(candidate, stop_event=candidate.stop_event + 1)
        elif reason == "missing_input":
            # Force an existing graph read to claim an unavailable frame value.
            first = (
                candidate.read_regions[0]
                if candidate.read_regions
                else candidate.affected_regions[0]
            )
            candidate = replace(candidate, read_regions=(first,))
        return plan, frame, candidate, boundaries, operands

    from helion._compiler.cute import chained_row_collective_emission as emission

    original_domain = chain._operand_domain

    def domain(cg, node, coords, plan):
        if reason == "domain" and any(
            "chain_row_collective_" in coord for coord in coords
        ):
            raise chain._UnsupportedChain("unproven original domain")
        return original_domain(cg, node, coords, plan)

    with (
        patch.object(chain, "_operand_domain", domain),
        patch.object(
            emission,
            "emit_sparse_reduction",
            return_value=["original_sparse"] if reason == "sparse" else None,
        ),
    ):
        lines, _, tracker, _ = _capture(change=change)
    assert lines is None and not tracker.activated


@pytest.mark.parametrize("leaf_count", [1, 4])
def test_real_kda_capture_uses_original_nodes_and_all_three_native_operands(leaf_count):
    root = Path(__file__).resolve().parents[1]
    namespace = ModuleType("benchmarks")
    namespace.__path__ = [str(root / "benchmarks")]
    previous = sys.modules.get("benchmarks")
    sys.modules["benchmarks"] = namespace
    try:
        kernel, args = _kda_fixture()
        config = _pipeline_config(16, pipeline=True)
        config.config.update(
            block_sizes=[128],
            cute_chained_group_contractions=True,
            cute_chained_warp_mma_rows=32,
            cute_chained_collective_retention=True,
            cute_chained_leaf_pipeline="rectangular_tma",
            cute_chained_leaf_count=leaf_count,
        )
        lines, candidate, _, source = _capture(kernel=kernel, args=args, config=config)
    finally:
        if previous is None:
            sys.modules.pop("benchmarks", None)
        else:
            sys.modules["benchmarks"] = previous
    assert lines is not None
    assert len(candidate.collectives) == (2 if leaf_count == 1 else 1)
    assert candidate.shape == (32, 128)
    text = "\n".join(lines)
    assert text.count("cute.make_rmem_tensor((4,), cutlass.Float32)") == len(
        candidate.collectives
    )
    assert "chain_row_collective_" in source
    assert f"chain_0_b[chain_row_collective_{candidate.first_event}_row + 32" in text
