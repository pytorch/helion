from __future__ import annotations

import ast
from unittest.mock import patch

import pytest

from ._cute_prepared_source import assert_prepared_root_equivalent
from .test_cute_chained_root_initialized import _source
from helion._compiler.cute import chained_plain_root as roots


@pytest.fixture(scope="module")
def original_and_prepared():
    with patch.object(roots, "codegen_plain_root", return_value=False):
        before = _source(late=False)
    return before, _source(late=False)


def test_actual_root_preserves_original_primitives_and_surrounding_math(
    original_and_prepared,
):
    assert_prepared_root_equivalent(*original_and_prepared)


@pytest.mark.parametrize(
    "mutation",
    (
        "lhs",
        "rhs",
        "accumulator",
        "phase",
        "issuer",
        "k_count",
        "initialized",
        "missing_wait",
        "read_source",
        "store_destination",
        "completion_order",
        "seed_math",
    ),
)
def test_source_equivalence_rejects_operand_phase_and_math_changes(
    original_and_prepared, mutation
):
    before, after = original_and_prepared
    tree = ast.parse(after)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    issue = next(
        node
        for node in calls
        if ast.unparse(node.func) == "execute_prepared_continuation"
    )
    port = issue.args[1].elts[0]
    action = issue.args[0].elts[0]
    if mutation in ("lhs", "rhs", "accumulator"):
        index = ("lhs", "rhs", "accumulator").index(mutation)
        port.elts[index] = ast.Name(id="foreign_operand", ctx=ast.Load())
    elif mutation == "phase":
        port.elts[5] = ast.Constant(1)
        issue.args[3].elts[0] = ast.Constant(1)
    elif mutation == "issuer":
        issue.args[5] = ast.Constant(True)
    elif mutation == "k_count":
        action.elts[3] = ast.Constant(4)
    elif mutation == "initialized":
        action.elts[4] = ast.Constant(True)
    elif mutation == "missing_wait":
        action.elts[6] = ast.Constant(False)
    elif mutation in ("read_source", "store_destination"):
        operation = "read" if mutation == "read_source" else "store"
        call = next(
            node
            for node in calls
            if ast.unparse(node.func)
            == f"prepared_tcgen_edge.execute_prepared_{operation}"
        )
        call.args[0 if operation == "read" else 1] = ast.Name(
            id="foreign_storage", ctx=ast.Load()
        )
    elif mutation == "completion_order":
        kernel = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and any(
                "execute_prepared_store_completion" in ast.unparse(s) for s in node.body
            )
        )
        index = next(
            i
            for i, node in enumerate(kernel.body)
            if ast.unparse(node).startswith(
                "prepared_tcgen_edge.execute_prepared_store_completion("
            )
        )
        kernel.body[index - 1], kernel.body[index] = (
            kernel.body[index],
            kernel.body[index - 1],
        )
    else:
        seed = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "chain_0_values[chain_seed_index]"
        )
        seed.value = ast.Constant(0.0)
    with pytest.raises(AssertionError):
        assert_prepared_root_equivalent(
            before, ast.unparse(ast.fix_missing_locations(tree))
        )
