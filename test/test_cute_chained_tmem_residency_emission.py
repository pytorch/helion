from __future__ import annotations

import ast
from dataclasses import replace
import importlib
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_execution import _LEGACY_DIGESTS
from .test_cute_chained_execution import _digest
from .test_cute_chained_execution import _emissions
from .test_cute_chained_preparation_cut import _kda_fixture
import helion
from helion import exc
from helion._compiler.cute import chained_tcgen_stage as stages
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_tmem_accumulator import (
    plan_tmem_accumulator_residency,
)
from helion._testing import DEVICE
import helion.language as hl

if TYPE_CHECKING:
    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.cute.chained_tmem_accumulator import TmemAccumulatorResidency
    from helion.runtime.kernel import Kernel


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _resident_loop(a, b, c, d, e, initial, initial_side, seed_side: hl.constexpr):
    steps, m, k = a.shape
    n = hl.specialize(initial.size(-1))
    other_n = hl.specialize(initial_side.size(-1))
    history = torch.empty((steps, m, n), device=a.device)
    other_history = torch.empty((steps, m, other_n), device=a.device)
    output = torch.empty_like(initial)
    other_output = torch.empty_like(initial_side)
    for rows in hl.tile(m, block_size=128):
        state = initial[rows, :]
        side = initial_side[rows, :]
        for step in hl.tile(steps, block_size=1):
            cols, other_cols = hl.arange(n), hl.arange(other_n)
            kk = hl.arange(k)
            source = hl.dot(
                a[step.id, rows, kk],
                b[step.id, kk, cols],
                acc=state,
                out_dtype=torch.float32,
            )
            common = c[step.id, rows, kk]
            state = hl.dot(
                common, d[step.id, kk, cols], acc=source, out_dtype=torch.float32
            )
            if seed_side:
                side = hl.dot(
                    common,
                    e[step.id, kk, other_cols],
                    acc=side * 0.5,
                    out_dtype=torch.float32,
                )
            else:
                side = hl.dot(
                    common, e[step.id, kk, other_cols], out_dtype=torch.float32
                )
            history[step.id, rows, cols] = state
            other_history[step.id, rows, other_cols] = side
        output[rows, :] = state
        other_output[rows, :] = side
    return history, other_history, output, other_output


def _args(dtype: torch.dtype, seed: bool) -> tuple:
    return (
        *(
            torch.empty(shape, dtype=dtype)
            for shape in (
                (3, 128, 16),
                (3, 16, 32),
                (3, 128, 16),
                (3, 16, 32),
                (3, 16, 16),
            )
        ),
        torch.empty((128, 32), dtype=torch.float32),
        torch.empty((128, 16), dtype=torch.float32),
        seed,
    )


def _proof(plan: ChainedMatmulPlan) -> TmemAccumulatorResidency:
    groups = plan.contraction_groups
    if groups is None:
        geometries = tuple(stages.stage_geometry(shape) for shape in plan.shapes)
        assert all(geometry is not None for geometry in geometries)
        groups = tuple(
            ContractionGroup((index,), (geometry,))
            for index, geometry in enumerate(geometries)
            if geometry is not None
        )
    groups = tuple(
        group for group in groups if group.stages[0] not in plan.warp_mma_stages
    )
    result = plan_tmem_accumulator_residency(plan, groups)
    assert result is not None
    return result


def _capture(
    kernel: Kernel,
    args: tuple,
    config: helion.Config,
    *,
    role: bool = False,
    invalid: bool = False,
) -> list[dict]:
    original = stages.emit_stage
    records = []

    def emit(*args, **kwargs):
        plan, stage = args[1], args[3]
        residency = _proof(plan)
        if invalid:
            residency = replace(residency, columns=residency.columns + 16)
        kwargs["residency"] = residency
        if role:
            kwargs["execution"] = ChainedExecution(
                256,
                thread="consumer_thread",
                warp="consumer_warp",
                sync="consumer_barrier.arrive_and_wait()",
            )
        lines = original(*args, **kwargs)
        records.append(
            {
                "stage": stage,
                "proof": residency,
                "source": "\n".join(lines),
                "published": tuple(args[2]),
            }
        )
        return lines

    with _cpu_codegen(), patch.object(stages, "emit_stage", emit):
        kernel._bind_isolated(args).to_code(config)
    return records


@pytest.mark.parametrize("case", list(_LEGACY_DIGESTS))
def test_none_residency_keeps_original_seed_and_publication_bytes(case: str) -> None:
    original = stages.emit_stage

    def emit(*args, **kwargs):
        return original(*args, **kwargs, residency=None)

    with patch.object(stages, "emit_stage", emit):
        assert _digest(_emissions(case)) == _LEGACY_DIGESTS[case]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("seed", [False, True])
def test_resident_source_skips_smem_and_destination_seeds_only_other_members(
    dtype: torch.dtype, grouped: bool, seed: bool
) -> None:
    records = _capture(
        _resident_loop,
        _args(dtype, seed),
        helion.Config(
            num_warps=8,
            cute_chained_mma_schedule="tcgen05_tmem",
            cute_chained_group_contractions=grouped,
        ),
        role=True,
    )
    proof = records[0]["proof"]
    producer = next(
        record for record in records if record["stage"] == proof.source_stage
    )
    consumer = next(
        record for record in records if record["stage"] == proof.destination_group
    )
    pfx, cfx = f"chain_{proof.source_stage}", f"chain_{proof.destination_group}"
    assert f"{pfx}_c" not in producer["source"]
    assert f"{pfx}_result_row" not in producer["source"]
    assert proof.source not in producer["published"]
    assert (
        "tcgen05.commit" in producer["source"] and "mbarrier_wait" in producer["source"]
    )
    assert producer["source"].endswith("consumer_barrier.arrive_and_wait()")
    assert f"{pfx}_c" not in consumer["source"]
    assert f"{cfx}_mma.set(tcgen05.Field.ACCUMULATE, True)" in consumer["source"]
    assert f"{cfx}_seed_{proof.destination_member}_segment" not in consumer["source"]
    if grouped:
        assert (
            f"{cfx}_seed_2_segment = cute.composition(cute.domain_offset(((0, 32), 0, 0), {cfx}_acc), cute.make_layout(((128, 16), 1, 1)))"
            in consumer["source"]
        )
        assert f"{cfx}_seed_2_values.fill(0.0)" in consumer["source"]
        assert (f"{cfx}_seed_2_values[" in consumer["source"]) is seed
    else:
        assert "_segment =" not in consumer["source"]
    tree = ast.parse(consumer["source"])
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and ast.unparse(node.test) == "consumer_thread < 128"
        ):
            assert "consumer_barrier.arrive_and_wait" not in ast.unparse(node)


def test_stale_residency_is_rejected_before_special_emission() -> None:
    with pytest.raises(exc.BackendUnsupported, match="failed late validation"):
        _capture(
            _resident_loop,
            _args(torch.bfloat16, True),
            helion.Config(
                num_warps=4,
                cute_chained_mma_schedule="tcgen05_tmem",
                cute_chained_group_contractions=True,
            ),
            invalid=True,
        )


def test_actual_kda_retains_nonresident_fp32_carry_seed() -> None:
    kernel, args = _kda_fixture()
    records = _capture(
        kernel,
        args,
        helion.Config(
            block_sizes=[128],
            num_warps=16,
            cute_chained_mma_schedule="tcgen05_tmem",
            cute_chained_group_contractions=True,
            cute_chained_scratch_layout="xor",
            cute_chained_pointwise_vectorize=True,
            cute_chained_scan_schedule="warp",
            cute_chained_pointwise_cache_bytes=4096,
            cute_chained_pointwise_unroll=8,
            cute_chained_warp_mma_rows=32,
        ),
    )
    proof = records[0]["proof"]
    assert (
        proof.source_stage,
        proof.destination_group,
        proof.destination_member,
        proof.columns,
    ) == (12, 13, 13, 32)
    source = next(record["source"] for record in records if record["stage"] == 12)
    destination = next(record["source"] for record in records if record["stage"] == 13)
    assert "chain_12_c" not in source + destination
    assert "chain_13_seed_14_segment" in destination
    assert "cute.make_layout(((128, 128), 1, 1))" in destination
    assert "chain_loop_carry_2[" in destination
    assert (
        "chain_13_seed_14_values[chain_13_seed_14_index] = cutlass.Float32("
        in destination
    )
    assert "cute.arch.fence_view_async_tmem_store()" in destination


@pytest.mark.parametrize("columns,other", [(16, 16), (32, 16), (32, 128), (64, 48)])
def test_actual_cute_segment_identity_copy_has_exact_disjoint_coverage(
    columns: int, other: int
) -> None:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05
    from cutlass.utils import blackwell_helpers

    ir = importlib.import_module("cutlass._mlir.ir")
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            n = columns + other
            mma = blackwell_helpers.make_trivial_tiled_mma(
                cutlass.BFloat16,
                cutlass.BFloat16,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
                cutlass.Float32,
                tcgen05.CtaGroup.ONE,
                (128, n),
                tcgen05.OperandSource.SMEM,
            )
            layout = mma.make_fragment_C(mma.partition_shape_C((128, n))).layout
            pointer = cute.make_ptr(
                cutlass.Float32, 0, cute.AddressSpace.tmem, assumed_align=16
            )
            acc = cute.make_tensor(pointer, layout)
            origin, shape = ((0, columns), 0, 0), ((128, other), 1, 1)
            segment = cute.composition(
                cute.domain_offset(origin, acc), cute.make_layout(shape)
            )
            identity = cute.composition(
                cute.domain_offset(
                    origin,
                    mma.get_slice(0).partition_C(cute.make_identity_tensor((128, n))),
                ),
                cute.make_layout(shape),
            )
            copy = tcgen05.make_tmem_copy(
                cute.make_copy_atom(
                    tcgen05.St32x32bOp(tcgen05.Repetition(min(32, other & -other))),
                    cutlass.Float32,
                ),
                segment,
            )
            seen = []
            for thread in range(128):
                owner = copy.get_slice(thread)
                owner.partition_D(segment)
                coords = owner.partition_S(identity)
                seen.extend(
                    tuple(map(int, coords[index]))
                    for index in range(int(cute.size(coords)))
                )
            expected = {(row, col) for row in range(128) for col in range(columns, n)}
            assert len(seen) == len(set(seen)) and set(seen) == expected
        assert module.operation.verify()


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _multiple_segments(
    a,
    b,
    c,
    d,
    e,
    f,
    initial,
    initial_seeded,
    initial_unseeded,
    steps: int,
    transpose: hl.constexpr,
):
    steps = hl.specialize(steps)
    width = hl.specialize(b.size(-1))
    seeded_width = hl.specialize(e.size(-1))
    unseeded_width = hl.specialize(f.size(-1))
    k = hl.specialize(a.size(-1))
    history = torch.empty((a.size(0), *initial.shape), device=a.device)
    seeded_history = torch.empty((a.size(0), *initial_seeded.shape), device=a.device)
    unseeded_history = torch.empty(
        (a.size(0), *initial_unseeded.shape), device=a.device
    )
    output = torch.empty_like(initial)
    seeded_output = torch.empty_like(initial_seeded)
    unseeded_output = torch.empty_like(initial_unseeded)
    for rows in hl.tile(128, block_size=128):
        if transpose:
            state = initial[:, rows]
            seeded = initial_seeded[:, rows]
            unseeded = initial_unseeded[:, rows]
        else:
            state = initial[rows, :]
            seeded = initial_seeded[rows, :]
            unseeded = initial_unseeded[rows, :]
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(k)
            cols = hl.arange(width)
            seeded_cols = hl.arange(seeded_width)
            unseeded_cols = hl.arange(unseeded_width)
            left = a[step.id, rows, kk]
            right = b[step.id, kk, cols]
            common = c[step.id, rows, kk]
            destination = d[step.id, kk, cols]
            other_seeded = e[step.id, kk, seeded_cols]
            other_unseeded = f[step.id, kk, unseeded_cols]
            if transpose:
                source = hl.dot(right.T, left.T, acc=state, out_dtype=torch.float32)
                state = hl.dot(
                    destination.T, common.T, acc=source, out_dtype=torch.float32
                )
                seeded = hl.dot(
                    other_seeded.T,
                    common.T,
                    acc=seeded * 0.5,
                    out_dtype=torch.float32,
                )
                unseeded = hl.dot(other_unseeded.T, common.T, out_dtype=torch.float32)
                history[step.id, cols, rows] = state
                seeded_history[step.id, seeded_cols, rows] = seeded
                unseeded_history[step.id, unseeded_cols, rows] = unseeded
            else:
                source = hl.dot(left, right, acc=state, out_dtype=torch.float32)
                state = hl.dot(common, destination, acc=source, out_dtype=torch.float32)
                seeded = hl.dot(
                    common, other_seeded, acc=seeded * 0.5, out_dtype=torch.float32
                )
                unseeded = hl.dot(common, other_unseeded, out_dtype=torch.float32)
                history[step.id, rows, cols] = state
                seeded_history[step.id, rows, seeded_cols] = seeded
                unseeded_history[step.id, rows, unseeded_cols] = unseeded
        if transpose:
            output[:, rows] = state
            seeded_output[:, rows] = seeded
            unseeded_output[:, rows] = unseeded
        else:
            output[rows, :] = state
            seeded_output[rows, :] = seeded
            unseeded_output[rows, :] = unseeded
    return (
        history,
        seeded_history,
        unseeded_history,
        output,
        seeded_output,
        unseeded_output,
    )


_SEGMENT_CASES = [(16, 16, 16, 0), (32, 16, 48, 1), (64, 32, 64, 3)]


def _segment_args(device, dtype, transpose, widths_steps) -> tuple:
    width, seeded_width, unseeded_width, steps = widths_steps
    torch.manual_seed(726)
    count = max(1, steps)
    operands = tuple(
        torch.randn((count, m, n), device=device, dtype=dtype) * 0.125
        for m, n in (
            (128, 16),
            (16, width),
            (128, 16),
            (16, width),
            (16, seeded_width),
            (16, unseeded_width),
        )
    )
    initial = tuple(
        torch.randn((n, 128) if transpose else (128, n), device=device) * 0.125
        for n in (width, seeded_width, unseeded_width)
    )
    return (*operands, *initial, steps, transpose)


def _segment_config() -> helion.Config:
    return helion.Config(
        num_warps=8,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_group_contractions=True,
    )


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("widths_steps", _SEGMENT_CASES)
def test_multiple_seeded_unseeded_segments_cpu(transpose, widths_steps) -> None:
    records = _capture(
        _multiple_segments,
        _segment_args("cpu", torch.bfloat16, transpose, widths_steps),
        _segment_config(),
    )
    proof = records[0]["proof"]
    assert proof.source_geometry.transpose is transpose
    assert proof.destination_geometry.transpose is transpose
    assert proof.columns == widths_steps[0]
    assert [record["stage"] for record in records] == [0, 1]
    source = records[1]["source"]
    assert "chain_1_seed_2_segment" in source
    assert "chain_1_seed_3_segment" in source
    assert "chain_1_seed_2_values[" in source
    assert "chain_1_seed_3_values[" not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("widths_steps", _SEGMENT_CASES)
def test_multiple_seeded_unseeded_segments_gpu(dtype, transpose, widths_steps) -> None:
    args = _segment_args(DEVICE, dtype, transpose, widths_steps)
    tensors = args[:9]
    saved = tuple(tensor.clone() for tensor in tensors)
    config = _segment_config()
    original = _multiple_segments._bind_isolated(args).compile_config(config)
    emit_original = stages.emit_stage

    def emit(*stage_args, **kwargs):
        proof = _proof(stage_args[1])
        assert proof.source_geometry.transpose is transpose
        kwargs["residency"] = proof
        return emit_original(*stage_args, **kwargs)

    with patch.object(stages, "emit_stage", emit):
        bound = _multiple_segments._bind_isolated(args)
        source = bound.to_code(config)
        assert "chain_1_seed_2_segment" in source
        assert "chain_1_seed_3_segment" in source
        resident = bound.compile_config(config)
    expected, actual, repeated = original(*args), resident(*args), resident(*args)
    steps = widths_steps[-1]
    if steps == 0:
        # Histories are intentionally unwritten when the dynamic loop is empty.
        expected, actual, repeated = expected[3:], actual[3:], repeated[3:]
        torch.testing.assert_close(actual, args[6:9], rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
    torch.testing.assert_close(tensors, saved, rtol=0, atol=0)
    if steps:
        state, seeded, unseeded = (tensor.clone() for tensor in args[6:9])
        # This flag is a dynamic property: mock.patch's delattr teardown does
        # not restore it. Save/set/restore through PyTorch's public setter.
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for step in range(steps):
                a, b, c, d, e, f = (tensor[step].float() for tensor in args[:6])
                if transpose:
                    state = b.T @ a.T + state
                    state = d.T @ c.T + state
                    seeded = e.T @ c.T + seeded * 0.5
                    unseeded = f.T @ c.T
                else:
                    state = a @ b + state
                    state = c @ d + state
                    seeded = c @ e + seeded * 0.5
                    unseeded = c @ f
                torch.testing.assert_close(
                    tuple(history[step] for history in actual[:3]),
                    (state, seeded, unseeded),
                    rtol=2e-4,
                    atol=2e-5,
                )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.testing.assert_close(
            actual[3:], (state, seeded, unseeded), rtol=2e-4, atol=2e-5
        )
