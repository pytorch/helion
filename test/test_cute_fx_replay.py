from __future__ import annotations

import ast
import contextlib
import operator
from types import SimpleNamespace
from typing import Any
from typing import cast

import pytest
import torch

from helion._compiler.ast_extension import create
from helion._compiler.generate_ast import GenerateAST
from helion._compiler.inductor_lowering import GraphInterpreter


class _Lowering:
    def __init__(self, result: ast.expr) -> None:
        self.result = result

    def codegen(self, interpreter: GraphInterpreter, node: torch.fx.Node) -> object:
        return self.result


def _codegen() -> GenerateAST:
    codegen = object.__new__(GenerateAST)
    codegen._statement_owner_fx_node = None
    codegen._statements_by_owner_node_id = {}
    codegen._codegen_results_by_owner_node_id = {}
    codegen._track_statement_owners = True
    codegen.referenced_thread_block_dims = [1, 1, 1]
    codegen.statements_stack = [[]]
    codegen.device_function = cast(
        "Any",
        SimpleNamespace(
            cute_state=SimpleNamespace(has_tcgen05_fragment_epilogue_plan=False),
            expr_to_var_info={},
            new_var=lambda prefix: f"{prefix}_normalized",
        ),
    )
    codegen._record_statement_thread_references = lambda statements: None
    codegen._record_tcgen05_owned_statement = lambda statement: None
    return codegen


def _node() -> tuple[torch.fx.Graph, torch.fx.Node, _Lowering]:
    graph = torch.fx.Graph()
    placeholder = graph.placeholder("input")
    node = graph.call_function(operator.neg, (placeholder,))
    graph.output(node)
    lowering = _Lowering(
        create(
            ast.BinOp,
            left=create(ast.Name, id="lhs", ctx=ast.Load()),
            op=create(ast.Add),
            right=create(ast.Name, id="rhs", ctx=ast.Load()),
        )
    )
    node.meta.update(
        location=contextlib.nullcontext(),
        lowering=lowering,
        val=1,
    )
    return graph, node, lowering


def test_final_result_is_recorded_without_changing_existing_metadata() -> None:
    graph, node, lowering = _node()
    codegen = _codegen()
    interpreter = GraphInterpreter(graph, codegen)
    interpreter._create_named_result = cast(
        "Any", lambda current, result: f"{current.name}_normalized"
    )

    result = interpreter.run_node(node)

    assert node.meta["codegen"] is lowering.result
    assert isinstance(result, ast.Name)
    assert codegen.codegen_result_for_node(node) == (True, result)


def test_statement_lookup_keeps_exact_owner_and_container() -> None:
    _, node, _ = _node()
    codegen = _codegen()
    first: list[ast.AST] = []
    second: list[ast.AST] = []
    statement = create(ast.Pass)

    with codegen.statement_owner_node(node):
        codegen.append_statement(first, statement)
        codegen.append_statement(second, create(ast.Pass))

    entries = codegen.statements_owned_by_node(node)
    assert len(entries) == 2
    assert entries[0] == (first, statement)

    codegen.remove_statements_owned_by_nodes((node,))
    assert first == []
    assert second == []
    assert codegen.statements_owned_by_node(node) == ()


def test_owned_span_replacement_is_atomic() -> None:
    graph = torch.fx.Graph()
    first_node = graph.call_function(operator.neg, (graph.placeholder("x"),))
    second_node = graph.call_function(operator.neg, (first_node,))
    codegen = _codegen()
    prefix = create(ast.Pass)
    first = create(ast.Pass)
    second = create(ast.Pass)
    suffix = create(ast.Pass)
    body = [prefix]
    with codegen.statement_owner_node(first_node):
        codegen.append_statement(body, first)
    with codegen.statement_owner_node(second_node):
        codegen.append_statement(body, second)
    body.append(suffix)
    replacement = create(ast.Expr, value=create(ast.Constant, value=1))

    assert codegen.replace_owned_statement_span(
        body, (first_node, second_node), (replacement,)
    )
    assert body == [prefix, replacement, suffix]
    assert codegen.statements_owned_by_node(first_node) == ()
    assert codegen.statements_owned_by_node(second_node) == ()


def test_owned_span_replacement_rejects_gap_without_mutation() -> None:
    graph = torch.fx.Graph()
    first_node = graph.call_function(operator.neg, (graph.placeholder("x"),))
    second_node = graph.call_function(operator.neg, (first_node,))
    codegen = _codegen()
    first = create(ast.Pass)
    gap = create(ast.Pass)
    second = create(ast.Pass)
    body: list[ast.AST] = []
    with codegen.statement_owner_node(first_node):
        codegen.append_statement(body, first)
    body.append(gap)
    with codegen.statement_owner_node(second_node):
        codegen.append_statement(body, second)
    original = tuple(body)

    assert not codegen.replace_owned_statement_span(
        body, (first_node, second_node), (create(ast.Pass),)
    )
    assert tuple(body) == original
    assert codegen.statements_owned_by_node(first_node) == ((body, first),)
    assert codegen.statements_owned_by_node(second_node) == ((body, second),)


def test_owned_span_replacement_rejects_aliased_ast_without_mutation() -> None:
    _, node, _ = _node()
    codegen = _codegen()
    statement = create(ast.Expr, value=create(ast.Constant, value=1))
    body: list[ast.AST] = []
    with codegen.statement_owner_node(node):
        codegen.append_statement(body, statement)

    assert not codegen.replace_owned_statement_span(body, (node,), (statement,))
    assert body == [statement]
    assert codegen.statements_owned_by_node(node) == ((body, statement),)


def test_owned_span_replacement_rejects_duplicate_source_identity() -> None:
    _, node, _ = _node()
    codegen = _codegen()
    statement = create(ast.Pass)
    body: list[ast.AST] = []
    with codegen.statement_owner_node(node):
        codegen.append_statement(body, statement)
    body.append(statement)

    assert not codegen.replace_owned_statement_span(body, (node,), (create(ast.Pass),))
    assert body == [statement, statement]
    assert codegen.statements_owned_by_node(node) == ((body, statement),)


def test_owned_span_replacement_rolls_back_thread_dimensions_on_error() -> None:
    _, node, _ = _node()
    codegen = _codegen()
    source = create(ast.Pass)
    body: list[ast.AST] = []
    with codegen.statement_owner_node(node):
        codegen.append_statement(body, source)
    replacement = create(ast.Expr, value=create(ast.Constant, value=1))

    def fail_after_update(statements: list[ast.AST]) -> None:
        codegen.referenced_thread_block_dims[0] = 32
        raise RuntimeError("injected recorder failure")

    codegen._record_statement_thread_references = fail_after_update
    with pytest.raises(RuntimeError, match="injected recorder failure"):
        codegen.replace_owned_statement_span(body, (node,), (replacement,))

    assert body == [source]
    assert codegen.referenced_thread_block_dims == [1, 1, 1]
    assert codegen.statements_owned_by_node(node) == ((body, source),)
