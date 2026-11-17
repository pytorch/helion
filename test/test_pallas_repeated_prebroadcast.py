from __future__ import annotations

import operator
from typing import cast

import pytest
import torch
from torch._inductor.virtualized import V

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.inductor_lowering import FakeGraphLowering
from helion._compiler.inductor_lowering import InductorLowering
from helion._compiler.inductor_lowering import prepare_node_lowering
from helion._compiler.pallas.tracing_ops import _annotate_pre_broadcast
from helion._compiler.pallas.tracing_ops import (
    _rewrite_outer_subscripts_for_pre_broadcast,
)
from helion._compiler.source_location import SourceLocation
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion.language._tracing_ops import _pre_broadcast_tile
from helion.language.view_ops import subscript

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("repeated", [False, True])
def test_pre_broadcast_preserves_consumer_input_groups(outer, repeated):
    # Two consumers require different tile widths. A tile may be shared among
    # repeated operands of one consumer, but not between widths or source nodes.
    location = SourceLocation(1, 0, 1, 0, "pre_broadcast", __file__)
    graph = torch.fx.Graph()
    with (
        mock_cuda_unavailable(),
        forbid_cuda_work(),
        CompileEnvironment(
            torch.device("cpu"), helion.Settings(backend="pallas")
        ) as env,
    ):

        def node_value(node, shape):
            node.meta.update(val=torch.empty(shape), location=location)
            return node

        if outer:
            loop = graph.placeholder("loop_result")
            bases = [
                node_value(graph.call_function(operator.getitem, (loop, i)), (2,))
                for i in range(2)
            ]
        else:
            loop = None
            bases = [node_value(graph.placeholder(f"state{i}"), (2,)) for i in range(2)]
        values = [
            node_value(
                graph.call_function(subscript, (base, [slice(None), None])), (2, 1)
            )
            for base in bases
        ]
        consumers = []
        for width in (64, 256):
            wide = node_value(graph.placeholder(f"wide{width}"), (2, width))
            right = values[0] if repeated else values[1]
            consumer = node_value(
                graph.call_function(
                    torch.ops.aten.addcmul.default, (wide, values[0], right)
                ),
                (2, width),
            )
            consumers.append(consumer)
        graph.output(tuple(consumers))
        with V.set_graph_handler(lowering := FakeGraphLowering()):
            for consumer in consumers:
                prepare_node_lowering(lowering, consumer)
        original_lowerings = [consumer.meta["lowering"] for consumer in consumers]
        original_inputs = [
            tuple(lowering.input_nodes(consumer))
            for lowering, consumer in zip(original_lowerings, consumers, strict=True)
        ]
        if outer:
            assert loop is not None
            _rewrite_outer_subscripts_for_pre_broadcast(
                loop, dict(enumerate(bases)), helion.Config()
            )
        else:
            _annotate_pre_broadcast(
                graph,
                {node.name for node in (*bases, *values)},
                [],
                env,
                helion.Config(),
            )
        graph.lint()
        for consumer, lowering, original in zip(
            consumers, original_lowerings, original_inputs, strict=True
        ):
            assert isinstance(lowering, InductorLowering)
            assert consumer.meta["lowering"] is lowering
            actual = lowering.input_nodes(consumer)
            assert len(actual) == len(original)
            left, right = consumer.args[1:]
            assert (left is right) == repeated
            assert (
                isinstance(left, torch.fx.Node) and left.target is _pre_broadcast_tile
            )
            assert (
                isinstance(right, torch.fx.Node) and right.target is _pre_broadcast_tile
            )
            assert left.args[0] is values[0]
            assert right.args[0] is values[0 if repeated else 1]
        tiles = [node for node in graph.nodes if node.target is _pre_broadcast_tile]
        assert len(tiles) == (2 if repeated else 4)
        assert consumers[0].args[1] is not consumers[1].args[1]

    source_values = [torch.tensor([2.0, -3.0]), torch.tensor([5.0, 7.0])]
    concrete = {
        base: source[:, None].expand(2, 128)
        for base, source in zip(bases, source_values, strict=True)
    }
    for value, base in zip(values, bases, strict=True):
        concrete[value] = concrete[base]
    for consumer in consumers:
        wide = cast("torch.fx.Node", consumer.args[0])
        width = wide.meta["val"].shape[-1]
        concrete[wide] = torch.arange(2 * width, dtype=torch.float32).reshape(2, width)
        for arg in consumer.args[1:]:
            assert isinstance(arg, torch.fx.Node)
            source, target = arg.args
            value = concrete[source]
            concrete[arg] = (
                value[..., :target] if target < 128 else value.repeat(1, target // 128)
            )
        actual = consumer.target(*(concrete[arg] for arg in consumer.args))
        expected = torch.addcmul(
            concrete[wide],
            source_values[0][:, None],
            source_values[0 if repeated else 1][:, None],
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
