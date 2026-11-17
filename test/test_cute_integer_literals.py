"""Integer constants retain the eager scalar conversion before comparison."""

from __future__ import annotations

import ast
import operator

import cutlass
import pytest
import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import CuteDSLOpOverrides

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_integer_literals import integer_literal_comparisons
from helion._testing.cute_integer_literals import integer_literal_inputs

pytestmark = pytest.mark.cpu_only


def test_out_of_range_int32_constants_match_eager_comparisons() -> None:
    # Six comparisons over neighboring limits and multiple wraps exercise the
    # scalar conversion, without compiling a kernel for every combination.
    literals = (2**31, -(2**31) - 1, 2**32 - 1, 2**32, 2**32 + 1, -(2**32) - 1)
    comparisons = (
        operator.lt,
        operator.le,
        operator.eq,
        operator.ne,
        operator.ge,
        operator.gt,
    )
    values = integer_literal_inputs(torch.device("cpu"))
    with cpu_render_target():
        bound = cpu_bind(integer_literal_comparisons, (values,))
        overrides = bound.env.backend.inductor_op_overrides()
        for literal in literals:
            expression = overrides.constant(literal, torch.int32)
            scalar = eval(expression, {"cutlass": cutlass})
            for compare in comparisons:
                actual = torch.tensor(
                    [bool(compare(cutlass.Int32(int(x)), scalar).value) for x in values]
                )
                expected = compare(values, literal)
                torch.testing.assert_close(actual, expected)
                reverse = torch.tensor(
                    [bool(compare(scalar, cutlass.Int32(int(x))).value) for x in values]
                )
                torch.testing.assert_close(reverse, compare(literal, values))


def test_int64_large_scalar_domain_already_matches_eager() -> None:
    values = torch.tensor([-(2**63), -1, 0, 1, 2**63 - 1], dtype=torch.int64)
    with cpu_render_target():
        bound = cpu_bind(integer_literal_comparisons, (values,))
        overrides = bound.env.backend.inductor_op_overrides()
        for literal in (-(2**63), 2**63 - 1, 2**63, 2**64 - 1):
            expression = overrides.constant(literal, torch.int64)
            assert expression == CuteDSLOpOverrides.constant(literal, torch.int64)
            scalar = eval(expression, {"cutlass": cutlass})
            for compare in (
                operator.lt,
                operator.le,
                operator.eq,
                operator.ne,
                operator.ge,
                operator.gt,
            ):
                actual = torch.tensor(
                    [bool(compare(cutlass.Int64(int(x)), scalar).value) for x in values]
                )
                torch.testing.assert_close(actual, compare(values, literal))


def test_in_range_int32_constants_keep_existing_expression() -> None:
    with cpu_render_target():
        bound = cpu_bind(
            integer_literal_comparisons, (torch.empty(37, dtype=torch.int32),)
        )
        overrides = bound.env.backend.inductor_op_overrides()
        for literal in (-(2**31), -1, 0, 1, 2**31 - 1):
            before = CuteDSLOpOverrides.constant(literal, torch.int32)
            after = overrides.constant(literal, torch.int32)
            assert ast.dump(ast.parse(after, mode="eval")) == ast.dump(
                ast.parse(before, mode="eval")
            )


def test_generic_render_has_no_out_of_range_int32_constants() -> None:
    with cpu_render_target():
        bound = cpu_bind(
            integer_literal_comparisons, (torch.empty(37, dtype=torch.int32),)
        )
        source = bound.to_code(helion.Config(block_sizes=[32]))
    tree = ast.parse(source)
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    constants = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            parent = parents.get(node)
            constants.add(
                -node.value
                if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.USub)
                else node.value
            )
    assert all(-(2**31) <= value < 2**31 for value in constants)
