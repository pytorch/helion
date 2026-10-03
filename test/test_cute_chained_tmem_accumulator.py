from __future__ import annotations

from dataclasses import FrozenInstanceError
from dataclasses import replace
import importlib
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph
from torch.fx import Node

from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_group_guards import _plan
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_frame import _capture
import helion
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_recurrence_workspace import plan_recurrence_workspace
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
from helion._compiler.cute.chained_tmem_accumulator import (
    plan_tmem_accumulator_residency,
)
from helion._compiler.cute.chained_tmem_accumulator import (
    validate_tmem_accumulator_residency,
)


def _candidate(width=32, *, transpose=False, dtype=torch.bfloat16, mode="direct"):
    graph = Graph()
    m, n = (width, 128) if transpose else (128, width)
    left = _input(graph, "left", (m, 16), dtype)
    right = _input(graph, "right", (16, n), dtype)
    source = _dot(graph, left, right)
    seed = source
    if mode == "cast":
        seed = _convert(graph, source, torch.float32)
    elif mode == "arithmetic":
        seed = _call(
            graph, torch.ops.aten.mul.Scalar, (source, 1.0), (m, n), torch.float32
        )
    intervening = (
        _dot(graph, left, right) if mode in ("intervening", "other_role") else None
    )
    destination = _dot(graph, left, right, seed)
    if transpose:
        common = _call(graph, torch.ops.aten.t.default, (right,), (128, 16), dtype)
    else:
        common = left
    other_right = _input(graph, "other_right", (16, 64), dtype)
    other = _dot(graph, common, other_right)
    plan = _plan(
        graph,
        (destination, other, source) if mode == "extra_user" else (destination, other),
    )
    geometry = StageGeometry((m, n, 16), transpose)
    following = 2 if intervening is not None else 1
    groups = (ContractionGroup((0,), (geometry,)),)
    if intervening is not None:
        groups += (ContractionGroup((1,), (geometry,)),)
    groups += (
        ContractionGroup(
            (following, following + 1), (geometry, StageGeometry((128, 64, 16), False))
        ),
    )
    plan = replace(
        plan,
        strategy="tcgen05_tmem",
        contraction_groups=groups,
        warp_mma_stages=frozenset({1}) if mode == "other_role" else frozenset(),
    )
    role_groups = tuple(
        group for group in groups if group.stages[0] not in plan.warp_mma_stages
    )
    return plan, role_groups


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("width", [16, 32, 64])
def test_exact_direct_accumulator_retains_original_nodes_and_geometry(
    dtype, transpose, width
):
    plan, groups = _candidate(width, transpose=transpose, dtype=dtype)
    assert plan.region is not None
    before = tuple((node, node.args, dict(node.kwargs)) for node in plan.region.nodes)
    proof = plan_tmem_accumulator_residency(plan, groups)
    assert proof is not None and validate_tmem_accumulator_residency(plan, proof)
    assert proof.region is plan.region
    assert proof.source is plan.dots[0] and proof.destination is plan.dots[1]
    assert (proof.source_stage, proof.destination_group, proof.destination_member) == (
        0,
        1,
        1,
    )
    assert proof.offset == 0 and proof.columns == width
    assert proof.source_geometry.transpose is transpose
    assert proof.source_geometry.logical[:2] == proof.destination_geometry.logical[:2]
    assert before == tuple(
        (node, node.args, dict(node.kwargs)) for node in plan.region.nodes
    )
    with pytest.raises(FrozenInstanceError):
        proof.columns = 0  # pyrefly: ignore [read-only]


@pytest.mark.parametrize("mode", ["cast", "arithmetic", "extra_user", "intervening"])
def test_no_reassociation_additional_use_or_intervening_tcgen_issue(mode):
    plan, groups = _candidate(mode=mode)
    assert plan_tmem_accumulator_residency(plan, groups) is None
    if mode == "intervening":
        assert plan_tmem_accumulator_residency(plan, (groups[0], groups[-1])) is None


def test_other_role_warp_preparation_does_not_renumber_adjacent_tcgen_issues():
    plan, groups = _candidate(mode="other_role")
    proof = plan_tmem_accumulator_residency(plan, groups)
    assert proof is not None
    assert (proof.source_stage, proof.destination_group) == (0, 2)


def test_initial_policy_rejects_nonzero_destination_member_offset():
    graph = Graph()
    left = _input(graph, "left", (128, 16), torch.bfloat16)
    right = _input(graph, "right", (16, 32), torch.bfloat16)
    source = _dot(graph, left, right)
    independent = _dot(graph, left, right)
    destination = _dot(graph, left, right, source)
    geometry = StageGeometry((128, 32, 16), False)
    groups = (
        ContractionGroup((0,), (geometry,)),
        ContractionGroup((1, 2), (geometry, geometry)),
    )
    plan = replace(
        _plan(graph, (independent, destination)),
        strategy="tcgen05_tmem",
        contraction_groups=groups,
    )
    assert plan_tmem_accumulator_residency(plan, groups) is None


@pytest.mark.parametrize(
    "bad",
    [
        "foreign",
        "stage",
        "offset",
        "columns",
        "graph_revision",
        "kwargs",
        "same_user_operand",
        "half_product",
        "result_shape",
        "operand_dtype",
        "source_group",
        "orientation",
        "sparse",
        "warp",
        "direct",
    ],
)
def test_malformed_or_stale_proofs_fail_closed(bad):
    plan, groups = _candidate(128 if bad == "orientation" else 32)
    assert plan.region is not None
    proof = plan_tmem_accumulator_residency(plan, groups)
    assert proof is not None
    if bad == "foreign":
        other, _ = _candidate()
        assert not validate_tmem_accumulator_residency(other, proof)
        return
    if bad == "stage":
        proof = replace(proof, source_stage=99)
    elif bad == "offset":
        proof = replace(proof, offset=99)
    elif bad == "columns":
        proof = replace(proof, columns=99)
    elif bad == "graph_revision":
        plan.region.graph.placeholder("different_revision")
    elif bad == "kwargs":
        proof.destination.kwargs = {"extra": (proof.source,)}
    elif bad == "same_user_operand":
        proof.destination.args = (proof.source, *proof.destination.args[1:])
    elif bad == "half_product":
        proof.source.args = (*proof.source.args[:3], torch.float16)
    elif bad == "result_shape":
        proof.source.meta["val"] = torch.empty((128, 64), dtype=torch.float32)
    elif bad == "operand_dtype":
        operand = proof.source.args[0]
        assert isinstance(operand, Node)
        operand.meta["val"] = operand.meta["val"].float()
    elif bad == "source_group":
        plan = replace(
            plan,
            contraction_groups=(
                ContractionGroup(
                    (0, 1), (groups[0].geometries[0], groups[1].geometries[0])
                ),
                ContractionGroup((2,), (groups[1].geometries[1],)),
            ),
        )
    elif bad == "orientation":
        geometry = replace(groups[1].geometries[0], transpose=True)
        plan = replace(
            plan,
            contraction_groups=(
                groups[0],
                replace(groups[1], geometries=(geometry, groups[1].geometries[1])),
            ),
        )
    elif bad == "sparse":
        geometry = StageGeometry((64, 32, 16), False)
        plan = replace(
            plan,
            shapes=((64, 32, 16), (64, 32, 16), plan.shapes[2]),
            contraction_groups=(
                ContractionGroup((0,), (geometry,)),
                ContractionGroup((1, 2), (geometry, groups[1].geometries[1])),
            ),
        )
    elif bad == "warp":
        plan = replace(plan, warp_mma_stages=frozenset({0}))
    else:
        plan = replace(plan, direct_output=True)
    assert not validate_tmem_accumulator_residency(plan, proof)


def test_largest_edge_then_source_order_selection_is_deterministic():
    graph = Graph()
    left = _input(graph, "left", (128, 16), torch.bfloat16)
    outputs = []
    for index, width in enumerate((32, 64, 64)):
        right = _input(graph, f"right_{index}", (16, width), torch.bfloat16)
        first = _dot(graph, left, right)
        outputs.append(_dot(graph, left, right, first))
    plan = replace(_plan(graph, tuple(outputs)), strategy="tcgen05_tmem")
    groups = tuple(
        ContractionGroup((i,), (StageGeometry(shape, False),))
        for i, shape in enumerate(plan.shapes)
    )
    proof = plan_tmem_accumulator_residency(plan, groups)
    assert proof is not None and proof.source_stage == 2


@pytest.mark.parametrize("dtype_name", ["BFloat16", "Float16"])
@pytest.mark.parametrize("resident_width,seed_width", [(16, 16), (32, 128), (64, 64)])
def test_actual_cute_fp32_prefix_and_segmented_seed_layout_cpu(
    dtype_name, resident_width, seed_width
):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05
    from cutlass.utils import blackwell_helpers

    ir = importlib.import_module("cutlass._mlir.ir")
    dtype = {"BFloat16": cutlass.BFloat16, "Float16": cutlass.Float16}[dtype_name]
    before = torch.cuda.is_initialized()
    width = resident_width + seed_width
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            layouts, identities = {}, {}
            for columns in {resident_width, seed_width, width}:
                mma = blackwell_helpers.make_trivial_tiled_mma(
                    dtype,
                    dtype,
                    cute.nvgpu.OperandMajorMode.K,
                    cute.nvgpu.OperandMajorMode.K,
                    cutlass.Float32,
                    tcgen05.CtaGroup.ONE,
                    (128, columns),
                    tcgen05.OperandSource.SMEM,
                )
                layouts[columns] = mma.make_fragment_C(
                    mma.partition_shape_C((128, columns))
                ).layout
                identities[columns] = mma.get_slice(0).partition_C(
                    cute.make_identity_tensor((128, columns))
                )
            source = {
                tuple(map(int, identities[resident_width][index])): int(
                    layouts[resident_width](index)
                )
                for index in range(128 * resident_width)
            }
            whole = {
                tuple(map(int, identities[width][index])): int(layouts[width](index))
                for index in range(128 * width)
            }
            assert all(whole[coord] == address for coord, address in source.items())
            pointer = cute.make_ptr(
                cutlass.Float32, 0, cute.AddressSpace.tmem, assumed_align=16
            )
            target = cute.make_tensor(pointer, layouts[width])
            segment = cute.composition(
                cute.domain_offset(((0, resident_width), 0, 0), target),
                cute.make_layout(((128, seed_width), 1, 1)),
            )
            assert str(segment.layout) == str(layouts[seed_width])
            copy = tcgen05.make_tmem_copy(
                cute.make_copy_atom(
                    tcgen05.St32x32bOp(
                        tcgen05.Repetition(min(32, seed_width & -seed_width))
                    ),
                    cutlass.Float32,
                ),
                segment,
            )
            writes = set()
            for thread in range(128):
                owner = copy.get_slice(thread)
                destination = owner.partition_D(segment)
                coords = owner.partition_S(identities[seed_width])
                values = cute.make_rmem_tensor(coords.shape, cutlass.Float32)
                # Emit the actual copy op as well as proving its static cells.
                cute.copy(copy, values, destination)
                for index in range(int(cute.size(coords))):
                    row, col = map(int, coords[index])
                    assert row == thread
                    assert (
                        int(layouts[seed_width](((row, col), 0, 0))) + resident_width
                        == whole[row, col + resident_width]
                    )
                    writes.add((row, col + resident_width))
            assert writes == {
                (row, col) for row in range(128) for col in range(resident_width, width)
            }
            assert not set(source) & writes
        assert module.operation.verify()
    assert torch.cuda.is_initialized() == before


def test_actual_kda_optional_workspace_omits_only_resident_c_and_preserves_gamma_seed():
    kernel, args = _kda_fixture()
    plan, cut, shapes = _capture(
        kernel,
        args,
        helion.Config(
            block_sizes=[128],
            num_warps=16,
            num_stages=2,
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
    original = plan_recurrence_workspace(plan, cut, shapes)
    assert original is not None
    assert original == plan_recurrence_workspace(plan, cut, shapes, None)
    groups = tuple(stage.group for stage in original.stages)
    proof = plan_tmem_accumulator_residency(plan, groups)
    assert proof is not None and proof.source_stage == 12
    packed = plan_recurrence_workspace(plan, cut, shapes, proof)
    assert packed is not None and validate_tmem_accumulator_residency(plan, proof)
    assert original.layout.allocated_bytes == 98304
    assert packed.layout.allocated_bytes == packed.peak_live_bytes == 81920
    assert packed.a_bytes == original.a_bytes and packed.b_bytes == original.b_bytes
    assert packed.stages == original.stages
    assert proof.source not in dict(packed.bindings)
    assert {item.name for item in original.layout.regions} - {
        item.name for item in packed.layout.regions
    } == {"chain_12_c"}
    assert packed.layout.region("chain_loop_carry_2").live_until == 27
    for region in packed.layout.regions:
        previous = original.layout.region(region.name)
        assert (region.byte_size, region.live_from, region.live_until) == (
            previous.byte_size,
            previous.live_from,
            previous.live_until,
        )
    assert (
        plan_recurrence_workspace(plan, cut, shapes, replace(proof, columns=64)) is None
    )
