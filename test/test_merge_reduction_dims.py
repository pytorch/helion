from __future__ import annotations

import copy
from itertools import starmap
import operator
import pickle
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock
from unittest.mock import patch

import sympy
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from helion._compiler.active_blocks import unused_block_ids
from helion._compiler.aten_lowering import resolve_reduction_sizes
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.fragment_storage import configured_fragment_expr
from helion._compiler.device_ir import RootGraphInfo
from helion._compiler.device_ir import _get_custom_decomp_table
from helion._compiler.host_function import HostFunction
from helion._compiler.merge_reduction_dims import _META_KEY
from helion._compiler.variable_origin import BlockSizeOrigin
from helion.language._tracing_ops import _get_symnode
from helion.language._tracing_ops import _host_tensor
from helion.language.memory_ops import store


def test_reduction_metadata_survives_copy_and_pickle() -> None:
    op = torch.ops.aten.sum.dim_IntList
    x = torch.randn(2, 3)
    graph = make_fx(
        lambda value: op(value, [0, 1]) + op(x, [0, 1]),
        decomposition_table=_get_custom_decomp_table(),
    )(x.clone())
    tags = [
        node.meta[_META_KEY] for node in graph.graph.nodes if _META_KEY in node.meta
    ]
    assert tags
    nodes = {node.name: node for node in graph.graph.nodes}
    for tag in tags:
        assert pickle.loads(pickle.dumps(tag)) == tag
        assert all(
            isinstance(nodes[name].meta["val"], torch.Tensor)
            and nodes[name].target is not _host_tensor
            for name in tag.nodes
        )
    cloned = copy.deepcopy(graph)
    torch.testing.assert_close(cloned(x), 2 * op(x, [0, 1]))


def test_logical_size_preserves_capture_identity_and_arange_extent() -> None:
    graph = torch.fx.Graph()
    tensor = graph.placeholder("tensor")
    size = graph.call_function(torch.ops.aten.sym_size.int, (tensor, 1))
    physical_extent = ShapeEnv().create_unbacked_symint()
    size.meta["val"] = physical_extent
    indices = list(
        starmap(
            graph.call_function,
            (
                (torch.ops.aten.arange.default, (size,)),
                (torch.ops.aten.arange.start, (0, size)),
                (torch.ops.aten.arange.start_step, (0, size, 1)),
            ),
        )
    )
    output = graph.output((size, *indices))
    captured_args = [size]
    env = SimpleNamespace(
        block_sizes=[SimpleNamespace(reduction=True, size=33)],
        get_block_id=lambda value: 0,
        sympy_debug=str,
    )
    with patch.object(CompileEnvironment, "current", return_value=env):
        resolve_reduction_sizes(graph)
    assert captured_args[0] is size and size in graph.nodes
    assert size.target is _get_symnode and size.meta["val"] == 33
    output_values = output.args[0]
    assert isinstance(output_values, tuple)
    assert output_values[0] == 33
    for index, position in zip(indices, (0, 1, 1), strict=True):
        physical = index.args[position]
        assert isinstance(physical, torch.fx.Node) and physical is not size
        assert physical.meta["helion_host_scalar"] is False
        assert physical.meta["val"] is physical_extent
        assert physical.args == (str(physical_extent),)
    graph.lint()

    class ExtentInterpreter(torch.fx.Interpreter):
        def call_function(self, target, args, kwargs):
            if target is _get_symnode:
                return 64 if args == (str(physical_extent),) else int(args[0])
            return super().call_function(target, args, kwargs)

    logical, *ranges = ExtentInterpreter(torch.fx.GraphModule({}, graph)).run(None)
    assert logical == 33
    for value in ranges:
        torch.testing.assert_close(value, torch.arange(64))


def test_scalar_store_liveness_does_not_allocate_reduction_dimensions() -> None:
    graph = torch.fx.Graph()
    tensor = graph.placeholder("tensor")
    tensor.meta["val"] = torch.empty(2, 3)
    graph.call_function(store, (tensor, [0, slice(None)], 1))
    graph.output(None)
    allocate = Mock(side_effect=AssertionError("liveness allocated a dimension"))
    env = SimpleNamespace(
        block_sizes=[SimpleNamespace(size=3, reduction=False, retired=True)],
        get_block_id=lambda value: None,
        known_equal=operator.eq,
        allocate_reduction_dimension=allocate,
        should_broadcast_tensor_indexers=lambda index: False,
        is_singleton_size=lambda size: size == 1,
        backend=SimpleNamespace(pad_factory_tensors_to_power_of_2=True),
    )
    with patch.object(CompileEnvironment, "current", return_value=env):
        assert unused_block_ids([RootGraphInfo(0, graph)], {0}) == set()
    allocate.assert_not_called()


def test_retired_fragment_extent_keeps_metadata_concretization() -> None:
    rdim, length = sympy.symbols("rdim length", integer=True, positive=True)
    env = SimpleNamespace(
        block_sizes=[SimpleNamespace(is_tile=False, reduction=False, numel=length)],
        specialize_expr=lambda value: value,
        shape_env=object(),
        backend=SimpleNamespace(static_rdim_size=lambda size: 64),
    )
    host = SimpleNamespace(
        expr_to_origin={rdim: SimpleNamespace(origin=BlockSizeOrigin(0))}
    )
    resolve_tile = Mock(side_effect=AssertionError("retired axis treated as a tile"))
    with (
        patch.object(HostFunction, "current", return_value=host),
        patch(
            "helion._compiler.cute.fragment_storage.metadata_guarded", return_value=True
        ),
        patch(
            "helion._compiler.cute.fragment_storage.shape_env_size_hint",
            return_value=33,
        ),
    ):
        assert (
            configured_fragment_expr(
                cast("CompileEnvironment", env), rdim, resolve_tile
            )
            == 64
        )
    resolve_tile.assert_not_called()
