from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import TypeAlias
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.fx import Graph
from torch.fx import Node

from ._cute_aux import _cpu_codegen
from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_group_guards import _plan
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_seed_tile_integration import _config as _kda_config
import helion
from helion._compiler.cute import chained_collectives as collectives
from helion._compiler.cute import chained_sparse_reduction as sparse
from helion._compiler.cute import chained_tcgen05
from helion._compiler.cute.chained_sparse_reduction import emit_sparse_reduction
from helion._compiler.cute.chained_sparse_reduction import plan_sparse_reduction
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import _tracing_ops
from helion.language import tile_index
from helion.language import view_ops

if TYPE_CHECKING:
    from helion._compiler.cute.chained_sparse_reduction import SparseReduction

Tree: TypeAlias = str | tuple["Tree", "Tree"]


def _fixture(extent=32, position=31, axis=0, keepdim=False, false=0.0):
    graph = Graph()
    shape = (extent, 128) if axis == 0 else (128, extent)
    payload = _input(graph, "payload", shape, torch.float32)
    iota = _call(graph, torch.ops.prims.iota.default, (extent,), (extent,), torch.int32)
    iota.kwargs = {"start": 0, "step": 1, "dtype": torch.int32}
    equality = _call(
        graph, torch.ops.aten.eq.Scalar, (iota, position), (extent,), torch.bool
    )
    condition = _call(
        graph,
        view_ops.subscript,
        (equality, [slice(None), None] if axis == 0 else [None, slice(None)]),
        (extent, 1) if axis == 0 else (1, extent),
        torch.bool,
    )
    zero = _call(
        graph, torch.ops.aten.scalar_tensor.default, (false,), (), torch.float32
    )
    selection = _call(
        graph,
        torch.ops.aten.where.self,
        (condition, payload, zero),
        shape,
        torch.float32,
    )
    result_shape = (
        tuple(1 if i == axis else size for i, size in enumerate(shape))
        if keepdim
        else (128,)
    )
    reduction = _call(
        graph,
        torch.ops.aten.sum.dim_IntList,
        (selection, [axis], keepdim),
        result_shape,
        torch.float32,
    )
    left = _input(graph, "lhs", (128, 16), torch.bfloat16)
    right = _input(graph, "rhs", (16, 128), torch.bfloat16)
    result = _dot(graph, left, right)
    plan = _plan(graph, (result, reduction))
    return plan, reduction, iota, equality, selection


def _original_tree(extent, position):
    zero = "positive_zero"
    lanes: list[Tree] = [
        (zero, "payload") if lane == position else zero for lane in range(32)
    ]
    for offset in (16, 8, 4, 2, 1):
        previous = lanes.copy()
        for lane in range(32):
            left = previous[lane]
            right = previous[lane + offset] if lane + offset < 32 else previous[lane]
            lanes[lane] = zero if left == right == zero else (left, right)
    return lanes[0]


def _planned_tree(proof: SparseReduction):
    zero = "positive_zero"
    value: Tree = (zero, "payload")
    for side in proof.zero_sides:
        value = (zero, value) if side == "left" else (value, zero)
    return value


@pytest.mark.parametrize("extent", range(1, 33))
@pytest.mark.parametrize("axis,keepdim", [(0, False), (1, False), (0, True), (1, True)])
def test_every_position_has_exact_original_warp_live_tree(extent, axis, keepdim):
    for position in range(extent):
        plan, reduction, _, _, selection = _fixture(extent, position, axis, keepdim)
        proof = plan_sparse_reduction(plan, reduction)
        assert proof is not None
        assert proof.node is reduction and proof.source is selection
        assert proof.position == position and proof.axis == axis
        assert _planned_tree(proof) == _original_tree(extent, position)
        assert proof.output_shape == tuple(reduction.meta["val"].shape)


@pytest.mark.parametrize("position", range(32))
def test_float32_bits_including_signed_zero_and_poisoned_unselected(position):
    plan, reduction, *_ = _fixture(position=position)
    proof = plan_sparse_reduction(plan, reduction)
    assert proof is not None
    bits = np.array(
        [
            0,
            0x80000000,
            1,
            0x80000001,
            0x3F800000,
            0xBF800000,
            0x7F800000,
            0xFF800000,
            0x7FC12345,
            0xFFC54321,
            0x7F812345,
        ],
        dtype=np.uint32,
    )
    payload = bits.view(np.float32)
    with np.errstate(invalid="ignore"):
        original = np.zeros((32, len(bits)), dtype=np.float32)
        # Unselected source values may be NaN: the original where contributes
        # literal positive zero, not that source value, to the reduction.
        source = np.full_like(original, np.float32(np.nan))
        source[position] = payload
        original += np.where(np.arange(32)[:, None] == position, source, np.float32(0))
        for offset in (16, 8, 4, 2, 1):
            previous = original.copy()
            for lane in range(32):
                original[lane] = np.add(
                    previous[lane],
                    previous[lane + offset] if lane + offset < 32 else previous[lane],
                )
        actual = np.add(np.float32(0), payload)
        for side in proof.zero_sides:
            actual = (
                np.add(np.float32(0), actual)
                if side == "left"
                else np.add(actual, np.float32(0))
            )
    np.testing.assert_array_equal(actual.view(np.uint32), original[0].view(np.uint32))
    assert actual.view(np.uint32)[1] != bits[1]  # A bare load is not equivalent.


@pytest.mark.parametrize("false", [-0.0, 1.0, -1.0, float("nan"), True])
def test_false_arm_must_be_literal_positive_fp32_zero(false):
    plan, reduction, *_ = _fixture(false=false)
    assert plan_sparse_reduction(plan, reduction) is None


@pytest.mark.parametrize(
    "change",
    [
        "start",
        "step",
        "shift",
        "tile",
        "dynamic",
        "multihot",
        "wrong_axis",
        "cast",
        "false_dtype",
        "selected_dtype",
        "extent",
        "negative",
        "outside",
    ],
)
def test_unproven_coordinate_or_typed_contract_rejected(change):
    plan, reduction, iota, equality, selection = _fixture(
        extent=33 if change == "extent" else 32
    )
    if change in ("start", "step"):
        iota.kwargs = {**iota.kwargs, change: 1 if change == "start" else 2}
    elif change == "shift":
        iota.target, iota.args, iota.kwargs = (
            torch.ops.aten.add.Scalar,
            (selection.args[1], 1),
            {},
        )
    elif change == "tile":
        iota.target, iota.kwargs = tile_index, {}
    elif change == "dynamic":
        equality.args = (iota, selection.args[1])
    elif change == "multihot":
        equality.target = torch.ops.aten.le.Scalar
    elif change == "wrong_axis":
        reduction.args = (selection, [1], False)
        reduction.meta["val"] = torch.empty((32,), dtype=torch.float32)
    elif change == "cast":
        iota.target, iota.args, iota.kwargs = (
            torch.ops.prims.convert_element_type.default,
            (selection.args[1], torch.int32),
            {},
        )
    elif change == "false_dtype":
        cast("Node", selection.args[2]).meta["val"] = torch.empty(
            (), dtype=torch.bfloat16
        )
    elif change == "selected_dtype":
        cast("Node", selection.args[1]).meta["val"] = torch.empty(
            (32, 128), dtype=torch.bfloat16
        )
    elif change in ("negative", "outside"):
        equality.args = (iota, -1 if change == "negative" else 32)
    assert plan_sparse_reduction(plan, reduction) is None


@pytest.mark.parametrize("zero", [0, 0.0, -0.0, 1])
def test_nested_mask_to_preserves_original_source_or_rejects(zero):
    plan, reduction, _, _, selection = _fixture()
    with reduction.graph.inserting_before(reduction):
        first = _call(
            reduction.graph,
            _tracing_ops._mask_to,
            (selection, 0),
            (32, 128),
            torch.float32,
        )
        second = _call(
            reduction.graph,
            _tracing_ops._mask_to,
            (first, zero),
            (32, 128),
            torch.float32,
        )
    reduction.args = (second, [0], False)
    assert plan.region is not None
    plan = replace(
        plan, region=replace(plan.region, nodes=tuple(reduction.graph.nodes))
    )
    proof = plan_sparse_reduction(plan, reduction)
    if zero == 0 and not np.signbit(zero):
        assert proof is not None and proof.source is second
    else:
        assert proof is None


def test_foreign_graph_and_immutable_record():
    plan, reduction, *_ = _fixture()
    foreign, other, *_ = _fixture()
    assert plan_sparse_reduction(plan, other) is None
    assert plan_sparse_reduction(foreign, reduction) is None
    proof = plan_sparse_reduction(plan, reduction)
    assert proof is not None
    with pytest.raises(FrozenInstanceError):
        proof.position = 0  # pyrefly: ignore [read-only]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sparse_chain(
    x,
    left,
    right,
    axis: hl.constexpr,
    position: hl.constexpr,
    keepdim: hl.constexpr,
    masked: hl.constexpr,
):
    axis_value: int = axis  # pyrefly: ignore [bad-assignment]
    position_value: int = position  # pyrefly: ignore [bad-assignment]
    extent = x.shape[axis_value]
    result = torch.empty((128, 128), device=left.device, dtype=torch.float32)
    for rows, columns in hl.tile([128, 128], block_size=[128, 128]):
        index = hl.arange(extent)
        kk = hl.arange(16)
        if axis_value == 0:
            if masked:
                raw = hl.load(
                    x, [index, columns], extra_mask=(columns.index % 2 == 0)[None, :]
                )
            else:
                raw = x[index, columns]
            condition = (index == position_value)[:, None]
        else:
            if masked:
                raw = hl.load(
                    x, [columns, index], extra_mask=(columns.index % 2 == 0)[:, None]
                )
            else:
                raw = x[columns, index]
            condition = (index == position_value)[None, :]
        selected = torch.where(condition, torch.sigmoid(raw.float()), 0.0)
        if keepdim:
            coefficient = torch.sum(selected, dim=axis_value, keepdim=True).squeeze(
                axis_value
            )
        else:
            coefficient = torch.sum(selected, dim=axis_value, keepdim=False)
        product = hl.dot(left[rows, kk], right[kk, columns], out_dtype=torch.float32)
        result[rows, columns] = product + coefficient[None, :]
    return result


def _arguments(axis, position, keepdim=False, masked=False, device="cpu"):
    return (
        torch.randn(
            (32, 128) if axis == 0 else (128, 32), dtype=torch.float32, device=device
        ),
        torch.randn((128, 16), dtype=torch.bfloat16, device=device),
        torch.randn((16, 128), dtype=torch.bfloat16, device=device),
        axis,
        position,
        keepdim,
        masked,
    )


def _capture(kernel, args, config):
    captured = []
    original = collectives.emit_collectives_before

    def observe(cg, plan, boundaries, stage, **kwargs):
        lines = original(cg, plan, boundaries, stage, **kwargs)
        if not captured:
            for node in plan.region.reductions:
                proof = plan_sparse_reduction(plan, node)
                if proof is not None:
                    snapshot = dict(boundaries)
                    emitted = emit_sparse_reduction(
                        cg,
                        plan,
                        boundaries,
                        node,
                        "sparse_result",
                        execution=kwargs.get("execution"),
                    )
                    assert boundaries == snapshot
                    captured.append((proof, emitted))
        return lines

    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        with (
            bound.env.use_runtime_arg_values(_runtime_values(kernel, args)),
            patch.object(collectives, "emit_collectives_before", observe),
            patch.object(chained_tcgen05, "emit_collectives_before", observe),
        ):
            bound.to_code(config)
    assert len(captured) == 1 and captured[0][1] is not None
    return captured[0]


@pytest.mark.parametrize(
    "axis,position,keepdim,masked",
    [
        (0, 0, False, False),
        (0, 31, True, True),
        (1, 7, False, True),
        (1, 16, True, False),
    ],
)
def test_real_source_keeps_selected_expression_masks_and_six_adds(
    axis, position, keepdim, masked
):
    proof, lines = _capture(
        _sparse_chain,
        _arguments(axis, position, keepdim, masked),
        helion.Config(num_warps=4, cute_chained_mma_schedule="tcgen05_tmem"),
    )
    assert proof.position == position and proof.axis == axis
    source = "\n".join(lines)
    assert "cute.math.exp2" in source and ".load() if" in source
    assert (
        "shuffle" not in source
        and "alloc_smem" not in source
        and "barrier" not in source
    )
    tree = ast.parse(source)
    updates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "sparse_result_acc"
    ]
    assert len(updates) == 6
    assert all(
        isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add)
        for node in updates
    )
    assert "sparse_result_vector = chain_thread +" in source
    if masked:
        assert "operator.eq" in source
        assert any(
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.BitAnd)
            and isinstance(node.right, ast.Constant)
            and node.right.value == 1
            for node in ast.walk(tree)
        )


def test_actual_kda_onehot_prefix_is_admitted_without_model_matching():
    kernel, args = _kda_fixture()
    proof, lines = _capture(kernel, args, _kda_config(32, pipeline=True))
    assert proof.position == 31 and proof.extent == 32 and proof.axis == 0
    assert proof.source.target is _tracing_ops._mask_to
    assert proof.zero_sides == ("left",) * 5
    assert "chain_collective_0[31, sparse_result_vector]" in "\n".join(lines)
    assert "chain_prep_thread" in "\n".join(lines)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("axis,position", [(0, 0), (0, 31), (1, 7), (1, 16)])
def test_sparse_reduction_gpu_preserves_masks_and_inputs(axis, position):
    # Prepared only; the component task never launches these GPU cases.
    args = _arguments(axis, position, masked=True, device=DEVICE)
    saved = tuple(value.clone() for value in args[:3])
    config = helion.Config(num_warps=4, cute_chained_mma_schedule="tcgen05_tmem")
    with patch.object(sparse, "emit_sparse_reduction", return_value=None):
        baseline_bound = _sparse_chain._bind_isolated(args)
        baseline = baseline_bound.compile_config(config)
    bound = _sparse_chain._bind_isolated(args)
    compiled = bound.compile_config(config)
    actual = compiled(*args)
    assert torch.equal(actual.view(torch.int32), baseline(*args).view(torch.int32))
    x, left, right = args[:3]
    valid = torch.arange(128, device=x.device) % 2 == 0
    masked = torch.where(valid[None, :] if axis == 0 else valid[:, None], x, 0)
    coefficient = torch.sigmoid(masked).select(axis, position)
    torch.testing.assert_close(
        actual,
        left.float() @ right.float() + coefficient[None, :],
        atol=1e-4,
        rtol=1e-4,
    )
    torch.testing.assert_close(compiled(*args), actual, atol=0, rtol=0)
    for value, original in zip(args[:3], saved, strict=True):
        torch.testing.assert_close(value, original, atol=0, rtol=0)
