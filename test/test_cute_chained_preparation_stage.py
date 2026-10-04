from __future__ import annotations

import ast
from dataclasses import replace
import math
from typing import TYPE_CHECKING
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from ._cute_aux import _cpu_codegen
from .test_cute_chained_execution import _LEGACY_DIGESTS
from .test_cute_chained_execution import _digest
from .test_cute_chained_execution import _emissions
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_group_guards import _plan
from .test_cute_chained_preparation_cut import _kda_fixture
import helion
from helion._compiler.cute import chained_tcgen_stage
from helion._compiler.cute.chained_mma_selection import warp_mma_shape
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
import helion.language as hl

if TYPE_CHECKING:
    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion.runtime.config import Config
    from helion.runtime.kernel import Kernel


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mixed_loop(left, right, other, feedback, initial):
    steps, m, k = left.shape
    n = feedback.size(-1)
    history = torch.empty((steps, m, n), device=left.device)
    output = torch.empty_like(initial)
    for rows, cols in hl.tile([m, n], block_size=[32, 48]):
        state = initial[rows, cols]
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(k)
            pp = hl.arange(feedback.size(-2))
            a = hl.load(
                left, [step.id, rows, kk], extra_mask=(rows.index % 3 != 1)[:, None]
            )
            first = hl.dot(a, right[step.id, kk, pp], out_dtype=torch.float32)
            second = hl.dot(a, other[step.id, kk, cols], out_dtype=torch.float32)
            state = hl.dot(
                first.to(left.dtype),
                feedback[step.id, pp, cols],
                acc=state * 0.5 + second,
                out_dtype=torch.float32,
            )
            history[step.id, rows, cols] = state
        output[rows, cols] = state
    return history, output


def _mixed_inputs(dtype: torch.dtype, vector: bool) -> tuple:
    k = 64 if vector else 48
    return (
        *(
            torch.empty((3, m, n), dtype=dtype)
            for m, n in ((29, k), (k, 48), (k, 37), (48, 37))
        ),
        torch.empty((29, 37), dtype=torch.float32),
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transposed_group(left, other, right, initial):
    steps, _, k = left.shape
    n = hl.specialize(right.size(-1))
    history = torch.empty((steps, 16, n), device=left.device)
    output = torch.empty_like(initial)
    for rr, cols in hl.tile([32, n], block_size=[32, n]):
        state = initial[rr, cols]
        for step in hl.tile(steps, block_size=1):
            mm, kk = hl.arange(16), hl.arange(k)
            b = right[step.id, kk, cols]
            first = hl.dot(left[step.id, mm, kk], b, out_dtype=torch.float32)
            second = hl.dot(other[step.id, rr, kk], b, out_dtype=torch.float32)
            state = state * 0.5 + second
            history[step.id, mm, cols] = first
        output[rr, cols] = state
    return history, output


def _transposed_inputs(n: int) -> tuple:
    return (
        *(
            torch.empty(shape, dtype=torch.bfloat16)
            for shape in ((3, 16, 16), (3, 32, 16), (3, 16, n))
        ),
        torch.empty((32, n), dtype=torch.float32),
    )


def _config(grouped: bool = True, *, vector: bool = False) -> helion.Config:
    return helion.Config(
        num_warps=16,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_group_contractions=grouped,
        cute_chained_warp_mma_rows=32,
        cute_chained_scratch_layout="xor",
        cute_chained_pointwise_unroll=8,
        cute_chained_pointwise_vectorize=vector,
    )


@pytest.mark.parametrize(
    "case",
    [
        "stage",
        "vector",
        "warp",
        "collective_serial",
        "collective_warp",
        "cache",
        "root",
    ],
)
def test_explicit_none_preserves_frozen_default_emission_bytes(case: str) -> None:
    original = chained_tcgen_stage.emit_stage

    def explicit_none(*args, **kwargs):
        return original(*args, **kwargs, prepared_shape=None)

    with patch.object(chained_tcgen_stage, "emit_stage", explicit_none):
        assert _digest(_emissions(case)) == _LEGACY_DIGESTS[case]


def _validation_plan() -> ChainedMatmulPlan:
    graph = Graph()
    lhs = _input(graph, "lhs", (32, 16), torch.bfloat16)
    rhs = _input(graph, "rhs", (16, 48), torch.bfloat16)
    result = _dot(graph, lhs, rhs)
    return replace(_plan(graph, (result,)), warp_mma_stages=frozenset({0}))


@pytest.mark.parametrize(
    "shape",
    [
        False,
        32,
        [32, 48, 16],
        (),
        (32, 48),
        (32, 48, 16, 1),
        (True, 48, 16),
        (32.0, 48, 16),
        (0, 48, 16),
        (-32, 48, 16),
        (16, 48, 16),
        (128, 48, 16),
        (32, 32, 16),
        (32, 64, 16),
        (32, 48, 32),
    ],
)
def test_invalid_prepared_shape_rejected_before_codegen(shape) -> None:
    with pytest.raises(ValueError, match="exact selected warp MMA shape"):
        chained_tcgen_stage.emit_stage(
            Mock(),
            _validation_plan(),
            {},
            0,
            StageGeometry((32, 48, 16), False),
            "0",
            prepared_shape=shape,
        )


def test_shape_override_rejected_for_tcgen_stage() -> None:
    plan = replace(_validation_plan(), warp_mma_stages=frozenset())
    with pytest.raises(ValueError, match="exact selected warp MMA shape"):
        chained_tcgen_stage.emit_stage(
            Mock(),
            plan,
            {},
            0,
            StageGeometry((32, 48, 16), False),
            "0",
            prepared_shape=(32, 48, 16),
        )


def _capture(kernel: Kernel, args: tuple, config: Config, prepared: bool) -> list[dict]:
    original = chained_tcgen_stage.emit_stage
    vector = chained_tcgen_stage.emit_vector_stage
    records = []
    vector_shapes = []

    def record_vector(*args, **kwargs):
        vector_shapes.append((kwargs["tag"], kwargs["shape"]))
        return vector(*args, **kwargs)

    def record(*args, **kwargs):
        plan, stage, geometry = args[1], args[3], args[4]
        group = args[6] if len(args) > 6 else kwargs.get("group")
        warp = stage in plan.warp_mma_stages
        shape = warp_mma_shape(geometry, group)
        if prepared and warp:
            kwargs["prepared_shape"] = shape
        begin = len(vector_shapes)
        lines = original(*args, **kwargs)
        records.append(
            {
                "stage": stage,
                "warp": warp,
                "shape": shape,
                "threads": plan.threads,
                "source": "\n".join(lines),
                "vectors": vector_shapes[begin:],
            }
        )
        return lines

    with (
        _cpu_codegen(),
        patch.object(chained_tcgen_stage, "emit_stage", record),
        patch.object(chained_tcgen_stage, "emit_vector_stage", record_vector),
    ):
        kernel._bind_isolated(args).to_code(config)
    assert records
    return records


def _layout_shape(source: str, name: str) -> tuple[int, int]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            assert isinstance(node.value, ast.Call)
            return ast.literal_eval(node.value.args[1])
    raise AssertionError(name)


def _check_shapes(before: list[dict], after: list[dict], *, vector: bool) -> None:
    assert len(before) == len(after)
    selected = 0
    for old, new in zip(before, after, strict=True):
        if not new["warp"]:
            assert old["source"] == new["source"]
            continue
        selected += 1
        stage, (m, n, k) = new["stage"], new["shape"]
        prefix = f"chain_{stage}"
        assert _layout_shape(old["source"], f"{prefix}_a_layout") == (128, k)
        assert _layout_shape(new["source"], f"{prefix}_a_layout") == (m, k)
        assert _layout_shape(new["source"], f"{prefix}_b_layout") == (n, k)
        # Only A storage ownership changes. Grouped B offsets, all expressions,
        # seed/publication and logical C stores remain byte-identical.
        assert (
            old["source"].split(f"{prefix}_b_ptr =", 1)[1]
            == new["source"].split(f"{prefix}_b_ptr =", 1)[1]
        )
        if vector:
            old_a = [shape for tag, shape in old["vectors"] if f"{prefix}_a_" in tag]
            new_a = [shape for tag, shape in new["vectors"] if f"{prefix}_a_" in tag]
            assert old_a == [(128, k)] and new_a == [(m, k)]
        else:
            tree = ast.parse(new["source"])
            loop = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and node.target.id == f"{prefix}_a_{stage}_step"
            )
            assert isinstance(loop.iter, ast.Call)
            trips = ast.literal_eval(loop.iter.args[0])
            count, threads = m * k, new["threads"]
            assert trips == math.ceil(count / threads)
            assert {
                thread + step * threads
                for step in range(trips)
                for thread in range(threads)
                if thread + step * threads < count
            } == set(range(count))
            if count % threads:
                assert f"if {prefix}_a_{stage}_index < {count}:" in new["source"]
    assert selected


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_trimmed_scalar_and_vector_producers_preserve_b_and_logical_c(
    grouped: bool, vector: bool, dtype: torch.dtype
) -> None:
    args = _mixed_inputs(dtype, vector)
    config = _config(grouped, vector=vector)
    before = _capture(_mixed_loop, args, config, False)
    after = _capture(_mixed_loop, args, config, True)
    _check_shapes(before, after, vector=vector)


@pytest.mark.parametrize("width", [8, 24])
def test_transposed_group_keeps_padding_and_member_offsets(width: int) -> None:
    args = _transposed_inputs(width)
    config = _config()
    config.config["cute_chained_pointwise_unroll"] = 1
    config.config["cute_chained_scratch_layout"] = "row_major"
    before = _capture(_transposed_group, args, config, False)
    after = _capture(_transposed_group, args, config, True)
    _check_shapes(before, after, vector=False)
    assert after[0]["shape"][0] == math.ceil(width / 16) * 16


def test_trimmed_all_single_trip_producers_do_not_falsely_activate_unroll() -> None:
    args = _transposed_inputs(8)
    with pytest.raises(
        helion.exc.BackendUnsupported, match="multi-trip loop operand producer"
    ):
        _capture(_transposed_group, args, _config(), True)


def test_actual_preparation_first_group_uses_8192_byte_a_shape() -> None:
    kernel, args = _kda_fixture()
    config = _config(vector=True)
    config.config.update(
        block_sizes=[128],
        cute_chained_scan_schedule="warp",
        cute_chained_pointwise_cache_bytes=4096,
    )
    before = _capture(kernel, args, config, False)
    after = _capture(kernel, args, config, True)
    _check_shapes(before, after, vector=True)
    assert after[0]["shape"] == (32, 64, 128)
    assert _layout_shape(after[0]["source"], "chain_0_a_layout") == (32, 128)
    assert math.prod(after[0]["shape"][::2]) * 2 == 8192
