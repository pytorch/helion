from __future__ import annotations

import ast

import pytest
import sympy
import torch
from torch._inductor.ir import ComputedBuffer
from torch._inductor.ir import FixedLayout
from torch._inductor.ir import Pointwise
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

import helion
from helion import exc
from helion._compiler.inductor_lowering import InductorLowering
from helion._compiler.inductor_lowering import PointwiseLowering
from helion._compiler.inductor_lowering import lowering_input_nodes
from helion._compiler.inductor_lowering import opmath_scalar_dtype
from helion._compiler.inductor_lowering import prepare_graph_lowerings
from helion._compiler.inductor_lowering import strip_unused_inputs
from helion._testing import DEVICE
from helion._testing import import_example
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner._metadata.ir_features import _has_networkx_node_link
from helion.autotuner._metadata.ir_features import _lowering_features
from helion.autotuner._metadata.ir_features import extract_ir_graph
import helion.language as hl

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.cpu_render,
    skipUnlessBackends(["triton", "cute"]),
]


@pytest.mark.skipif(not _has_networkx_node_link(), reason="networkx>=3.4 required")
def test_rolled_placeholder_metadata_is_not_a_lowered_call():
    bound = cpu_bind(import_example("softmax").softmax, (torch.empty(128, 128),))
    ir = bound.host_function.device_ir
    before = ir.semantic_debug_str()
    placeholders = [
        node
        for info in ir.graphs
        for node in info.graph.nodes
        if node.op == "placeholder"
        and isinstance(node.meta.get("lowering"), InductorLowering)
    ]
    assert placeholders
    for node in placeholders:
        assert not node.all_input_nodes
        assert node.meta["lowering"].input_slots
        assert node.meta["lowering"].input_nodes(node) == []
        assert node.meta["lowering"].input_fake_tensors(node) == []
    # Dataset extraction is valid outside CompileEnvironment/HostFunction.
    graph = extract_ir_graph(ir)
    assert graph["graph"]["rolled_reductions"]
    assert any(
        node["op_kind"] == "call_function"
        and node["lowering_class"] == "ReductionLowering"
        and node["input_shapes"]
        for node in graph["nodes"]
    )
    assert ir.semantic_debug_str() == before


class _CpuPointwise:
    """Execute the actual Inductor comparison recipe on IEEE-class CPU values."""

    def __init__(self, values):
        self.values = values

    def load(self, name, index):
        return self.values

    @staticmethod
    def ne(left, right):
        return left != right

    @staticmethod
    def isnan(value):
        return torch.isnan(value)

    @staticmethod
    def to_dtype(value, dtype, src_dtype=None, use_compute_types=True):
        return value.to(dtype)


@helion.kernel(static_shapes=True, autotune_effort="none")
def _self_compare(x: torch.Tensor, reduce: hl.constexpr, isnan: hl.constexpr):
    out = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for rows in hl.tile(x.size(0)):
        if reduce:
            value = torch.amax(x[rows, :], dim=1)
        else:
            value = x[rows, 0]
        if isnan:
            result = torch.isnan(value)
        else:
            result = value != value
        out[rows] = result
    return out


@pytest.mark.parametrize("reduce", [False, True])
@pytest.mark.parametrize("isnan", [False, True])
def test_bound_self_comparison_operands(reduce, isnan):
    cuda_initialized = torch.cuda.is_initialized()
    bound = cpu_bind(_self_compare, (torch.empty(31, 8), reduce, isnan))
    ir = bound.host_function.device_ir
    before = ir.semantic_debug_str()
    nodes = [
        node
        for info in ir.graphs
        for node in info.graph.nodes
        if node.target in (torch.ops.aten.ne.Tensor, torch.ops.aten.isnan.default)
    ]
    assert len(nodes) == 1
    if nodes[0].target is torch.ops.aten.ne.Tensor:
        assert nodes[0].args[0] is nodes[0].args[1]
    lowering = nodes[0].meta["lowering"]
    assert len(lowering.input_names) == len(lowering.input_nodes(nodes[0])) == 1
    values = torch.tensor([0.0, -0.0, 1.0, -float("inf"), float("inf"), float("nan")])
    with V.set_ops_handler(_CpuPointwise(values)):
        result = lowering.buffer.data.inner_fn([0] * len(lowering.buffer.data.ranges))
    assert torch.equal(result.value, torch.isnan(values))
    config = bound.config_spec.default_config()
    source = bound.to_code(config)
    ast.parse(source)
    assert ir.semantic_debug_str() == before
    assert torch.cuda.is_initialized() == cuda_initialized


@helion.kernel(static_shapes=True, autotune_effort="none")
def _repeated_binary(x: torch.Tensor, multiply: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        value = x[tile]
        if multiply:
            out[tile] = value * value
        else:
            out[tile] = value + value
    return out


@pytest.mark.parametrize("multiply", [False, True])
def test_repeated_binary_keeps_semantic_args_and_unique_lowered_inputs(multiply):
    bound = cpu_bind(_repeated_binary, (torch.empty(31), multiply))
    target = torch.ops.aten.mul.Tensor if multiply else torch.ops.aten.add.Tensor
    nodes = [
        node
        for info in bound.host_function.device_ir.graphs
        for node in info.graph.nodes
        if node.target is target
    ]
    assert len(nodes) == 1
    node = nodes[0]
    assert node.args[0] is node.args[1]
    lowering = node.meta["lowering"]
    assert lowering.input_nodes(node) == [node.args[0]]
    assert len(lowering.input_names) == 1
    assert lowering.input_fake_tensors(node) == [node.args[0].meta["val"]]
    features = _lowering_features(node)
    assert len(features.input_shapes) == len(features.input_dtypes) == 1
    # Rendering includes padded-lane value analysis for a 31-element tail.
    ast.parse(bound.to_code(helion.Config(block_sizes=[32])))


@helion.kernel(static_shapes=True, autotune_effort="none")
def _merged_comparison(x: torch.Tensor):
    out = torch.empty((x.size(0),), dtype=torch.bool, device=x.device)
    for rows in hl.tile(x.size(0)):
        value = x[rows, :]
        first = torch.amax(value, dim=1)
        second = torch.amax(value, dim=1)
        out[rows] = first != second
    return out


def test_cse_preserves_distinct_lowering_names_after_input_merge():
    bound = cpu_bind(_merged_comparison, (torch.empty(31, 8),))
    nodes = [
        node
        for info in bound.host_function.device_ir.graphs
        for node in info.graph.nodes
        if node.target is torch.ops.aten.ne.Tensor
    ]
    assert len(nodes) == 1
    node = nodes[0]
    lowering = node.meta["lowering"]
    assert len(lowering.input_names) == 2
    assert lowering.input_slots == (0, 1)
    assert lowering.input_nodes(node) == list(node.args)
    assert lowering_input_nodes(node) == list(node.args)
    if bound.env.backend.name == "cute":
        # Actual CSE has joined the reductions, but both Inductor names
        # must still receive the surviving value during code generation.
        assert node.args[0] is node.args[1]
    ast.parse(bound.to_code(bound.config_spec.default_config()))


def test_pruning_keeps_keyword_order_and_excludes_scheduler_dependencies():
    graph = torch.fx.Graph()
    left, right, unused, dependency = [
        graph.placeholder(name) for name in ("left", "right", "unused", "dependency")
    ]
    # A multi-output operation's individual lowered buffer can read only a
    # subset of the original operands, including repeated keyword arguments.
    node = graph.call_function(
        torch.ops.aten.native_layer_norm.default,
        (left, [8]),
        {"weight": left, "bias": unused, "eps": right},
    )
    names, slots = strip_unused_inputs(
        node,
        OrderedSet(["right_buffer", "left_buffer"]),
        {left: "left_buffer", right: "right_buffer", unused: "unused_buffer"},
    )
    assert names == ["left_buffer", "right_buffer"]
    assert slots == (0, 0, 1)
    assert node.kwargs["weight"] is left
    assert node.kwargs["bias"] is None
    buffer = ComputedBuffer(
        name="result",
        layout=FixedLayout(torch.device("cpu"), torch.float32, [sympy.Integer(8)]),
        data=Pointwise(
            device=torch.device("cpu"),
            dtype=torch.float32,
            inner_fn=lambda index: 0,
            ranges=[sympy.Integer(8)],
        ),
    )
    lowering = PointwiseLowering(buffer, names, slots)
    node.kwargs = {**node.kwargs, "_extra_deps": [dependency]}
    assert lowering.input_nodes(node) == [left, right]
    assert lowering_input_nodes(node) == [left, right]
    # FX replacement does not rewrite arbitrary metadata. Occurrence indices
    # keep the association correct without storing old Node references there.
    right.replace_all_uses_with(left)
    assert lowering.input_nodes(node) == [left, left]
    node.meta["lowering"] = lowering
    assert lowering_input_nodes(node) == [left, left]
    node.meta.pop("lowering")
    assert lowering_input_nodes(node) == [left]
    copied_graph = torch.fx.Graph()
    mapping = {}
    for current in graph.nodes:
        mapping[current] = copied_graph.node_copy(current, mapping.__getitem__)
    assert lowering.input_nodes(mapping[node]) == [mapping[left], mapping[left]]


@pytest.mark.parametrize("replace_first", [False, True])
def test_changed_lowered_occurrence_count_refuses_before_render(replace_first):
    bound = cpu_bind(_repeated_binary, (torch.empty(31), True))
    node = next(
        node
        for info in bound.host_function.device_ir.graphs
        for node in info.graph.nodes
        if node.target is torch.ops.aten.mul.Tensor
    )
    source = node.args[0]
    node.args = (2.0, source) if replace_first else (source, 2.0)
    with pytest.raises(exc.InternalError, match="lowered value inputs changed"):
        node.meta["lowering"].input_nodes(node)
    # Removing one repeated occurrence used to silently retain the old
    # square recipe. The normal codegen boundary must reject stale IR.
    with pytest.raises(exc.InternalError, match="lowered value inputs changed"):
        bound.to_code(helion.Config(block_sizes=[32]))


@pytest.mark.cuda
@pytest.mark.parametrize("reduce", [False, True])
def test_self_comparison_native_ieee_classes(reduce):
    classes = torch.tensor([0.0, -0.0, 1.0, -float("inf"), float("inf"), float("nan")])
    values = classes[torch.arange(31) % len(classes), None].expand(31, 8).contiguous()
    source = values.to(DEVICE)
    expected = torch.isnan(values.amax(1) if reduce else values[:, 0]).to(DEVICE)
    for isnan in (False, True):
        assert torch.equal(_self_compare(source, reduce, isnan), expected)
    if reduce:
        assert torch.equal(_merged_comparison(source), expected)


@helion.kernel(static_shapes=True, autotune_effort="none")
def _square_and_other(x: torch.Tensor, y: torch.Tensor):
    square = torch.empty_like(x)
    other = torch.empty_like(y)
    for rows in hl.tile(x.size(0)):
        left = x[rows]
        right = y[rows]
        square[rows] = left * left
        other[rows] = right + 1.0
    return square, other


@pytest.mark.parametrize("keyword", [False, True])
def test_splitting_repeated_lowering_input_refuses_before_render(keyword):
    bound = cpu_bind(_square_and_other, (torch.empty(31), torch.empty(31)))
    nodes = [
        node
        for info in bound.host_function.device_ir.graphs
        for node in info.graph.nodes
    ]
    multiply = next(n for n in nodes if n.target is torch.ops.aten.mul.Tensor)
    addition = next(n for n in nodes if n.target is torch.ops.aten.add.Tensor)
    left, right = multiply.args[0], addition.args[0]
    assert left is not right
    assert multiply.args == (left, left)
    if keyword:
        multiply.args = (left,)
        multiply.kwargs = {"other": right}
    else:
        multiply.args = (left, right)
    # Both source operands still exist, but one original repeated-input
    # group has split. Reusing the old x*x recipe would ignore y entirely.
    with pytest.raises(exc.InternalError, match="repeated lowered value input changed"):
        bound.to_code(helion.Config(block_sizes=[32]))


@helion.kernel(static_shapes=True, autotune_effort="none")
def _square_and_scale(x: torch.Tensor, scale: float):
    square = torch.empty_like(x)
    scaled = torch.empty_like(x)
    for rows in hl.tile(x.size(0)):
        value = x[rows]
        scaled[rows] = value * scale
        square[rows] = value * value
    return square, scaled


def test_reprepare_uses_current_operands_for_opmath_dtype():
    bound = cpu_bind(_square_and_scale, (torch.empty(31, dtype=torch.bfloat16), 1.1))
    graph = next(
        info.graph
        for info in bound.host_function.device_ir.graphs
        if any(n.target is torch.ops.aten.mul.Tensor for n in info.graph.nodes)
    )
    multiplies = [n for n in graph.nodes if n.target is torch.ops.aten.mul.Tensor]
    square = next(n for n in multiplies if n.args[0] is n.args[1])
    scaled = next(n for n in multiplies if n.args[0] is not n.args[1])
    scalar = scaled.args[1]
    assert isinstance(scalar, torch.fx.Node)
    assert isinstance(scalar.meta["val"], torch.SymFloat)
    previous = square.meta["lowering"]
    square.args = (square.args[0], scalar)
    graph.lint()
    # Re-preparation consumes the rewritten operands. The old slot contract
    # remains useful for detecting consumption before a new recipe is built.
    with pytest.raises(exc.InternalError, match="repeated lowered value input changed"):
        previous.input_nodes(square)
    with bound.env, bound.host_function:
        prepare_graph_lowerings(graph)
    current = square.meta["lowering"]
    assert current is not previous
    assert current.input_slots == (0, 1)
    assert current.input_nodes(square) == list(square.args)
    assert opmath_scalar_dtype(square) is torch.bfloat16
    ast.parse(bound.to_code(helion.Config(block_sizes=[32])))
