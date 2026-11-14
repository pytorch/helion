"""Semantic recognition of an authored integer S2F6/BF16 decode."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import helion
from helion._compiler.cute.analysis.native_s2f6_match import match_native_s2f6_decode
from helion._compiler.device_ir import ReductionLoopGraphInfo
from helion._compiler.device_ir import _get_custom_decomp_table
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_native_s2f6 import integer_s2f6_decode
import helion.language as hl
from helion.language import memory_ops

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True)
def _written_decode(raw: torch.Tensor, scale: torch.Tensor, out: torch.Tensor) -> None:
    for tile in hl.tile(raw.size(0), block_size=32):
        out[tile, :] = integer_s2f6_decode(raw[tile, :], scale[tile, :])


@pytest.fixture(scope="module")
@forbid_cuda_work()
def decomposed() -> torch.fx.GraphModule:
    return make_fx(
        integer_s2f6_decode,
        decomposition_table=_get_custom_decomp_table(),
        tracing_mode="fake",
    )(torch.empty(32, 128, dtype=torch.int8), torch.empty(32, 128, dtype=torch.uint8))


def _output(graph: torch.fx.Graph) -> torch.fx.Node:
    return next(n for n in graph.nodes if n.op == "output").args[0]


def _constant_user(graph: torch.fx.Graph, target: object, value: int) -> torch.fx.Node:
    matches = [
        n
        for n in graph.nodes
        if n.target is target
        and len(n.args) >= 1
        and type(n.args[-1]) is int
        and n.args[-1] == value
    ]
    assert len(matches) == 1
    return matches[0]


@forbid_cuda_work()
def test_decomposed_recipe_has_input_proof_without_names(decomposed) -> None:
    graph = copy.deepcopy(decomposed).graph
    for i, node in enumerate(graph.nodes):
        node.name = f"unrelated_{i}"
    raw, scale = (n for n in graph.nodes if n.op == "placeholder")
    output = _output(graph)
    proof = match_native_s2f6_decode(output)
    assert proof is not None
    assert proof.raw is raw and proof.scale is scale and proof.value is output
    assert {raw, scale, output} <= proof.nodes
    # The leaf does not alter the source graph or initialize CUDA.
    assert all(n.name.startswith("unrelated_") for n in graph.nodes)


@pytest.mark.parametrize(
    "target,old,new",
    [
        (torch.ops.aten.ge.Scalar, 128, 127),  # highest magnitude bit
        (torch.ops.aten.scalar_tensor.default, 7, 6),  # leading-bit index
        (torch.ops.aten.lt.Scalar, 0, 1),  # signed zero / sign predicate
        (torch.ops.aten.sub.Tensor, 6, 7),  # scale bias
        (torch.ops.aten.sub.Tensor, 128, 127),  # mantissa
        (torch.ops.aten.__lshift__.Scalar, 7, 6),  # exponent field shift
        (torch.ops.aten.clamp_max.default, 6, 5),  # subnormal shift
        (torch.ops.aten.le.Scalar, 0, -1),  # normal/subnormal boundary
        (torch.ops.aten.clamp_min.default, 0, 1),  # structural, dead for zero
        (torch.ops.aten.eq.Scalar, 0, 1),  # magnitude-zero select
        (torch.ops.aten.clamp_max.default, 0x7F7F, 0x7F80),  # saturation
        (torch.ops.aten.scalar_tensor.default, 0x8000, 0x4000),
        (torch.ops.aten.eq.Scalar, 255, 254),  # exceptional scale
        (torch.ops.aten.scalar_tensor.default, 0x7FC0, 0x7F80),  # source NaN
        (torch.ops.aten.ge.Scalar, 2, 2.0),  # exact integer recipe
    ],
)
@forbid_cuda_work()
def test_changed_recipe_is_rejected(decomposed, target, old, new) -> None:
    graph = copy.deepcopy(decomposed).graph
    node = _constant_user(graph, target, old)
    node.args = (*node.args[:-1], new)
    assert match_native_s2f6_decode(_output(graph)) is None


@forbid_cuda_work()
def test_integer_width_operator_and_dataflow_are_proved(decomposed) -> None:
    graph = copy.deepcopy(decomposed).graph
    raw, scale = (n for n in graph.nodes if n.op == "placeholder")
    raw.meta["val"] = torch.empty(32, 128, dtype=torch.uint8)
    assert match_native_s2f6_decode(_output(graph)) is None

    graph = copy.deepcopy(decomposed).graph
    node = next(n for n in graph.nodes if n.target is torch.ops.aten.bitwise_or.Tensor)
    node.target = torch.ops.aten.add.Tensor
    assert match_native_s2f6_decode(_output(graph)) is None

    graph = copy.deepcopy(decomposed).graph
    node = _constant_user(graph, torch.ops.aten.eq.Scalar, 255)
    # A signed input cannot replace the unsigned scale input.
    signed = next(
        n
        for n in graph.nodes
        if n.target is torch.ops.prims.convert_element_type.default
    )
    node.args = (signed, 255)
    assert match_native_s2f6_decode(_output(graph)) is None

    graph = copy.deepcopy(decomposed).graph
    node = next(n for n in graph.nodes if n.target is torch.ops.aten.add.Tensor)
    node.kwargs = {"alpha": 2}
    assert match_native_s2f6_decode(_output(graph)) is None


@forbid_cuda_work()
def test_mixed_same_dtype_scale_inputs_are_rejected(decomposed) -> None:
    graph = copy.deepcopy(decomposed).graph
    scale = next(
        n
        for n in graph.nodes
        if n.op == "placeholder" and n.meta["val"].dtype == torch.uint8
    )
    convert = next(
        n
        for n in scale.users
        if n.target is torch.ops.prims.convert_element_type.default
    )
    with graph.inserting_after(scale):
        other_scale = graph.placeholder("different_scale")
    other_scale.meta = scale.meta.copy()
    with graph.inserting_after(other_scale):
        other_convert = graph.call_function(convert.target, (other_scale, torch.int32))
    other_convert.meta = convert.meta.copy()
    shift = _constant_user(graph, torch.ops.aten.clamp_max.default, 6)
    shift.args = (other_convert, 6)
    # Both local conversion subgraphs match. Only their shared input identity
    # distinguishes this mixed normal/subnormal recipe from the admitted one.
    assert convert.meta["val"].dtype == other_convert.meta["val"].dtype
    assert match_native_s2f6_decode(_output(graph)) is None


@pytest.mark.parametrize("mutation", ["mantissa_shift", "branches", "narrow", "view"])
@forbid_cuda_work()
def test_shift_branch_and_tail_mutations_fail(decomposed, mutation) -> None:
    graph = copy.deepcopy(decomposed).graph
    if mutation == "mantissa_shift":
        node = next(
            n
            for n in graph.nodes
            if n.target is torch.ops.aten.sub.Tensor and n.args[0] == 7
        )
        node.args = (6, node.args[1])
    elif mutation == "branches":
        predicate = _constant_user(graph, torch.ops.aten.le.Scalar, 0)
        node = next(iter(predicate.users))
        node.args = (node.args[0], node.args[2], node.args[1])
    elif mutation == "narrow":
        node = next(
            n
            for n in graph.nodes
            if n.target is torch.ops.prims.convert_element_type.default
            and n.args[1] is torch.int16
        )
        node.args = (node.args[0], torch.int32)
    else:
        node = next(n for n in graph.nodes if n.target is torch.ops.aten.view.dtype)
        node.target = torch.ops.prims.convert_element_type.default
    assert match_native_s2f6_decode(_output(graph)) is None


@forbid_cuda_work()
def test_lifted_constant_needs_explicit_caller_resolution(decomposed) -> None:
    graph = copy.deepcopy(decomposed).graph
    constant = _constant_user(graph, torch.ops.aten.scalar_tensor.default, 0x7FC0)
    with graph.inserting_before(constant):
        lifted = graph.placeholder("lifted_constant")
    lifted.meta = constant.meta.copy()
    constant.replace_all_uses_with(lifted)
    assert match_native_s2f6_decode(_output(graph)) is None
    proof = match_native_s2f6_decode(
        _output(graph), resolve=lambda n: constant if n is lifted else n
    )
    assert proof is not None
    assert {constant, lifted, _output(graph)} <= proof.nodes


@forbid_cuda_work()
def test_actual_cpu_bound_integer_decode() -> None:
    args = (
        torch.empty(32, 128, dtype=torch.int8),
        torch.empty(32, 128, dtype=torch.uint8),
        torch.empty(32, 128, dtype=torch.bfloat16),
    )
    # Inspect the portable recipe before its independent native fusion.
    with (
        cpu_render_target(),
        patch(
            "helion._compiler.cute.fuse_native_s2f6.fuse_native_s2f6", return_value=0
        ),
    ):
        bound = cpu_bind(cast("Kernel[object]", _written_decode), args)
        assert bound.host_function is not None
        graph = bound.host_function.device_ir.graphs[0].graph
        store = next(n for n in graph.nodes if n.target is memory_ops.store)
        value = store.args[2]
        assert isinstance(value, torch.fx.Node)
        proof = match_native_s2f6_decode(value)
        loop = next(
            g
            for g in bound.host_function.device_ir.graphs
            if isinstance(g, ReductionLoopGraphInfo)
        )
        loop_store = next(n for n in loop.graph.nodes if n.target is memory_ops.store)
        loop_value = loop_store.args[2]
        assert isinstance(loop_value, torch.fx.Node)
        assert match_native_s2f6_decode(loop_value) is None
        loop_proof = match_native_s2f6_decode(
            loop_value,
            resolve=lambda n: (
                loop.placeholder_to_outer_arg(n)
                if n.graph is loop.graph and n.op == "placeholder"
                else n
            ),
        )
    assert proof is not None
    assert proof.raw.target is memory_ops.load
    assert proof.scale.target is memory_ops.load
    assert loop_proof is not None
    assert loop_proof.raw.target is loop_proof.scale.target is memory_ops.load
    assert set(loop.graph.find_nodes(op="placeholder")) <= loop_proof.nodes
