from __future__ import annotations

import ast
import importlib
import math
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._compiler.ast_extension import fix_missing_locations
from helion._compiler.ast_extension import unparse
from helion._compiler.cute.chained_matmul import _balanced_mask
from helion._compiler.cute.chained_matmul import _masked_operand
from helion._compiler.generate_ast import GenerateAST
from helion._compiler.output_header import get_needed_import_lines
import helion.language as hl


def _mask_terms(tree: ast.expr) -> tuple[list[str], int]:
    pending = [(tree, 0)]
    terms = []
    maximum = 0
    while pending:
        node, depth = pending.pop()
        maximum = max(maximum, depth)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            pending.extend(((node.right, depth + 1), (node.left, depth + 1)))
        else:
            terms.append(ast.unparse(node))
    return terms, maximum


@pytest.mark.parametrize("count", [0, 1, 2, 3, 7, 1200])
def test_balanced_domain_mask_keeps_order_and_logarithmic_depth(count: int) -> None:
    bounds = [f"index + {i} < limit" for i in range(count)]
    original = bounds.copy()
    predicate = ast.parse(_balanced_mask(bounds), mode="eval").body
    terms, depth = _mask_terms(predicate)
    assert terms == (bounds or ["True"])
    assert depth <= (math.ceil(math.log2(count)) if count else 0)
    assert bounds == original


def test_domain_mask_only_deduplicates_exact_pure_bounds() -> None:
    bounds = ["index < size", "0 <= index", "index < size", "(index) < size"]
    predicate = _balanced_mask(bounds)
    assert predicate.count("index < size") == 1
    assert predicate.count("(index) < size") == 1
    assert _mask_terms(ast.parse(predicate, mode="eval").body)[0] == [
        "index < size",
        "0 <= index",
        "index < size",
    ]


def test_balanced_conjunction_keeps_eager_left_to_right_evaluation() -> None:
    calls: list[int] = []

    def check(index: int) -> bool:
        calls.append(index)
        return index != 2

    # Unique probes observe evaluation order; deduplication is only for the
    # pure proof-owned predicates used by the production callers.
    predicate = _balanced_mask([f"check({i})" for i in range(7)])
    assert eval(predicate, {"check": check}) is False
    assert calls == list(range(7))


def test_masked_operand_preserves_float_grouping_and_false_value() -> None:
    expression = _masked_operand("a + (b + c)", "float", ["index < limit"] * 3)
    tree = ast.parse(expression, mode="eval").body
    assert isinstance(tree, ast.IfExp) and isinstance(tree.body, ast.Call)
    assert ast.dump(tree.body.args[0]) == ast.dump(
        ast.parse("a + (b + c)", mode="eval").body
    )
    values = {"a": 1e20, "b": -1e20, "c": 3.0, "index": 0, "limit": 1}
    assert eval(expression, values) == 0.0
    assert eval(expression, values | {"index": 1}) == 0.0
    assert eval(expression, values | {"a": 1.0, "b": 2.0}) == 6.0
    assert eval(expression, values | {"a": 1.0, "b": 2.0, "index": 1}) == 0.0
    assert _masked_operand("a + (b + c)", "float", []) == "float(a + (b + c))"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _metadata_copy(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    for i in hl.tile(x.size(0)):
        output[i] = x[i]
    return output


def test_deep_mask_passes_actual_device_function_postpasses() -> None:
    bounds = [f"index + {i} < limit" for i in range(1200)]
    value = _masked_operand("value", "cutlass.Float32", bounds)
    hoist = importlib.import_module(
        "helion._compiler.cute.hoist_lane_invariant_reductions"
    )
    with _cpu_codegen():
        bound = _metadata_copy._bind_isolated((torch.zeros(8),))
        config = bound.config_spec.normalized_config({"block_sizes": [8]})
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            # Bind graph metadata only, then exercise the real device-function
            # post-passes on a representative nested masked operand. No graph
            # visitor, full kernel source generation, or SDK compiler is used.
            context = GenerateAST(bound.host_function, config)
            with context.device_function:
                argument = bound.fake_args[0]
                assert isinstance(argument, torch.Tensor)
                output = context.device_function.tensor_arg(argument).name
                context.device_function.body.extend(
                    ast.parse(
                        "index = cutlass.Int32(cute.arch.thread_idx()[0])\n"
                        "limit = cutlass.Int32(4096)\n"
                        "value = cutlass.Float32(1.0)\n"
                        "for step in range(2):\n"
                        f"    {output}[step] = {value}\n"
                    ).body
                )
                with patch.object(
                    hoist, "_update_uniform_names", wraps=hoist._update_uniform_names
                ) as update:
                    result = context.device_function.codegen_function_def()
                assert update.call_count > 0
            module = ast.Module(body=result, type_ignores=[])
            get_needed_import_lines(module)
            source = unparse(fix_missing_locations(module), output_origin_lines=False)
    # The real post-passes must retain the complete mask and the output write.
    rendered = ast.parse(source)
    masks = [
        node.test
        for node in ast.walk(rendered)
        if isinstance(node, ast.IfExp)
        and isinstance(node.test, ast.BinOp)
        and isinstance(node.test.op, ast.BitAnd)
    ]
    assert len(masks) == 1
    assert _mask_terms(masks[0]) == (bounds, math.ceil(math.log2(len(bounds))))
    assert f"{output}[step] =" in source
