from __future__ import annotations

import ast
import copy
import operator
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph
from torch.fx import Node

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_cut import _runtime_values
import helion
from helion._compiler.ast_extension import expr_from_string
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute.work_order import UnsupportedWorkOrder
from helion._compiler.cute.work_order import project_scalar_slice
from helion._compiler.cute.work_order import scalar_slice
from helion._compiler.generate_ast import GenerateAST
from helion._compiler.inductor_lowering import GraphInterpreter
import helion.language as hl
from helion.language import _tracing_ops
from helion.language import memory_ops


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _ragged_sum(data, starts, ends, result):
    for row in hl.grid(result.size(0)):
        begin = starts[row].long() + 1
        end = ends[row].long() - 2
        accum = hl.full([], 0, dtype=torch.float32)
        for columns in hl.tile(begin, end):
            accum = accum + data[row, columns].sum()
        result[row] = accum


def _assert_load_addresses(text, expected):
    assignments = {}
    addresses = []

    class Expand(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in assignments:
                return copy.deepcopy(assignments[node.id])
            return node

    for statement in ast.parse(text).body:
        assert isinstance(statement, ast.Assign) and len(statement.targets) == 1
        value = statement.value
        for node in ast.walk(value):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
            ):
                addresses.append(Expand().visit(copy.deepcopy(node.func.value)))
        target = statement.targets[0]
        assert isinstance(target, ast.Name)
        assignments[target.id] = Expand().visit(copy.deepcopy(value))
    assert len(addresses) == len(expected) == 2
    for address, wanted in zip(addresses, expected, strict=True):
        assert ast.dump(address) == ast.dump(ast.parse(wanted, mode="eval").body)


def _project(
    cg, graph, outputs, tmp_path, expected_indices=("candidate_axis", "candidate_axis")
):
    loads = [node for node in graph.nodes if node.target is memory_ops.load]
    leaf = loads[0].args[1][0]
    assert isinstance(leaf, Node)
    spec = scalar_slice(graph, outputs, (leaf,))
    df = cg.device_function
    arguments = tuple(df.arguments)
    assert arguments
    codegen = tuple((node, node.meta.get("codegen")) for node in graph.nodes)
    symbols = df.expr_to_var_info
    symbol_values = dict(symbols)
    grid = cg.current_grid_state
    indices, offsets, sizes = (
        grid.strategy.index_vars,
        grid.strategy.offset_vars,
        df.block_size_var_cache,
    )
    mappings = (dict(indices), dict(offsets), dict(sizes))
    load_index = df.device_load_index
    before_statements = tuple(cg.statements_stack[-1])
    result_map = cg._codegen_results_by_owner_node_id
    result_contents = dict(result_map)
    owner_map = cg._statements_by_owner_node_id
    owner_contents = {key: tuple(value) for key, value in owner_map.items()}
    tracked, owner = cg._track_statement_owners, cg._statement_owner_fx_node
    for _ in range(2):
        result = project_scalar_slice(
            cg, spec, {leaf: expr_from_string("candidate_axis")}
        )
        assert (
            cg._codegen_results_by_owner_node_id is result_map
            and result_map == result_contents
        )
        assert cg._statements_by_owner_node_id is owner_map
        assert {key: tuple(value) for key, value in owner_map.items()} == owner_contents
    with (
        patch.object(
            GraphInterpreter, "run", side_effect=RuntimeError("projection interruption")
        ),
        pytest.raises(RuntimeError, match="projection interruption"),
    ):
        project_scalar_slice(cg, spec, {leaf: expr_from_string("candidate_axis")})
    assert (
        cg._codegen_results_by_owner_node_id is result_map
        and result_map == result_contents
    )
    assert cg._statements_by_owner_node_id is owner_map
    assert {key: tuple(value) for key, value in owner_map.items()} == owner_contents
    assert (
        cg._track_statement_owners is tracked and cg._statement_owner_fx_node is owner
    )
    assert df.expr_to_var_info is symbols
    assert df.expr_to_var_info == symbol_values
    assert grid.strategy.index_vars is indices and grid.strategy.offset_vars is offsets
    assert df.block_size_var_cache is sizes
    assert (indices, offsets, sizes) == mappings
    assert df.device_load_index == load_index
    assert len(df.arguments) == len(arguments)
    assert all(a is b for a, b in zip(df.arguments, arguments, strict=True))
    assert tuple(cg.statements_stack[-1]) == before_statements
    assert all(node.meta.get("codegen") is value for node, value in codegen)
    text = ast.unparse(ast.Module(body=list(result.statements), type_ignores=[]))
    sources = [
        df._tensor_args[node.args[0].meta["val"]].name
        for node in spec.nodes
        if node.target is memory_ops.load
    ]
    addresses = [
        f"{source}.iterator + cutlass.Int32({index}) * cutlass.Int32({source}.layout.stride[0])"
        for source, index in zip(sources, expected_indices, strict=True)
    ]
    _assert_load_addresses(text, addresses)
    # A candidate mention alone does not establish the actual memory binding.
    with pytest.raises(AssertionError):
        _assert_load_addresses(
            text.replace(
                "cutlass.Int32(candidate_axis)", "cutlass.Int32(indices_original)", 1
            ),
            addresses,
        )
    assert len(result.values) == 2
    (tmp_path / "projection.py").write_text(text)
    return spec, leaf


def _ordinary(action):
    original = GenerateAST._try_lower_direct_affine_root
    seen = []

    def observe(cg, grid_state, body):
        graph = cg.current_root_graph_info.graph
        loops = [
            node for node in graph.nodes if _tracing_ops.is_for_loop_target(node.target)
        ]
        assert len(loops) == 1
        loop = loops[0]
        seen.append(action(cg, graph, (loop.args[1][0], loop.args[2][0])))
        return original(cg, grid_state, body)

    args = (
        torch.empty((5, 64)),
        torch.arange(5, dtype=torch.int64),
        torch.arange(5, dtype=torch.int64) + 32,
        torch.empty(5),
    )
    with (
        _cpu_codegen(),
        patch.object(GenerateAST, "_try_lower_direct_affine_root", observe),
    ):
        bound = _ragged_sum._bind_isolated(args)
        source = bound.to_code(
            helion.Config(block_sizes=[8], num_warps=4, pid_type="flat")
        )
    assert len(seen) == 1
    return source


def test_original_ordinary_scalar_lowering_and_host_identity(tmp_path):
    reference = _ordinary(lambda *args: None)
    source = _ordinary(
        lambda cg, graph, outputs: _project(cg, graph, outputs, tmp_path)
    )
    assert source == reference
    (tmp_path / "source.py").write_text(source)


def test_original_fused_scalar_lowering_and_host_identity(tmp_path):
    original = chain._install_chained_body
    seen = []

    def observe(cg, plan, lines):
        aliases = tuple(plan.tensor_aliases.items())
        result = _project(
            cg,
            plan.loop.root.graph,
            (plan.loop.begin, plan.loop.end),
            tmp_path,
            ("candidate_axis", "1 + candidate_axis"),
        )
        assert tuple(plan.tensor_aliases.items()) == aliases
        seen.append(result)
        return original(cg, plan, lines)

    kernel, args = _kda_fixture()
    config = helion.Config(
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
    )
    sources = []
    for selected in (False, True):
        with _cpu_codegen():
            bound = kernel._bind_isolated(args)
            with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
                if selected:
                    with patch.object(chain, "_install_chained_body", observe):
                        sources.append(bound.to_code(config))
                else:
                    sources.append(bound.to_code(config))
    assert len(seen) == 1 and sources[0] == sources[1]
    (tmp_path / "source.py").write_text(sources[1])


@pytest.mark.parametrize(
    "mutation", ("target", "arguments", "value", "lowering", "bindings", "foreign")
)
def test_original_scalar_projection_rejects_mutations(mutation, tmp_path):
    def action(cg, graph, outputs):
        spec, leaf = _project(cg, graph, outputs, tmp_path)
        node = spec.outputs[0]
        assert isinstance(node, Node)
        target, args, value, lowering = (
            node.target,
            node.args,
            node.meta["val"],
            node.meta["lowering"],
        )
        bindings = {leaf: expr_from_string("candidate_axis")}
        try:
            if mutation == "target":
                node.target = torch.ops.aten.neg.default
            elif mutation == "arguments":
                node.args = (node.args[0], 2)
            elif mutation == "value":
                node.meta["val"] = torch.empty((), dtype=torch.float32)
            elif mutation == "lowering":
                node.meta["lowering"] = object()
            elif mutation == "bindings":
                bindings = {}
            elif mutation == "foreign":
                bindings = {outputs[1]: expr_from_string("candidate_axis")}
            with pytest.raises(UnsupportedWorkOrder):
                project_scalar_slice(cg, spec, bindings)
        finally:
            node.target, node.args = target, args
            node.meta["val"], node.meta["lowering"] = value, lowering

    _ordinary(action)


@pytest.mark.parametrize("mutation", ("erase", "reorder"))
def test_scalar_discovery_pins_original_membership_and_order(mutation):
    graph = Graph()
    first = graph.call_function(operator.add, (1, 2))
    second = graph.call_function(operator.add, (3, 4))
    first.meta["val"], second.meta["val"] = 3, 7
    spec = scalar_slice(graph, (first,), ())
    if mutation == "erase":
        graph.erase_node(second)
    else:
        second.append(first)
    assert first.graph is graph and second.graph is graph
    with pytest.raises(UnsupportedWorkOrder, match="membership or order"):
        spec.check()
