from __future__ import annotations

import ast
import copy
from unittest.mock import patch

import pytest
import torch
from torch._inductor.codecache import PyCodeCache

from .test_cute_bounded_cache_codegen import _bound
from .test_cute_bounded_cache_codegen import _config
from .test_cute_bounded_cache_codegen import _cpu_target
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


class _FlattenConjunctions(ast.NodeTransformer):
    def visit_BoolOp(self, node: ast.BoolOp) -> ast.expr:
        self.generic_visit(node)
        if isinstance(node.op, ast.And):
            values = []
            for value in node.values:
                if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.And):
                    values.extend(value.values)
                else:
                    values.append(value)
            node.values = values
        return node


def _comparable(source: str) -> str:
    module = ast.parse(source)
    module.body = [
        statement
        for statement in module.body
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Attribute)
            and statement.targets[0].attr == "_helion_cute_source_hash"
        )
    ]
    return ast.dump(_FlattenConjunctions().visit(module))


def _long_guards(source: str) -> list[ast.BoolOp]:
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.BoolOp)
        and isinstance(node.op, ast.And)
        and len(node.values) >= 8
    ]


@pytest.mark.parametrize("mode", ["scalar", "bounded_layout", "resident"])
def test_actual_reduction_codegen_only_changes_boolean_tree_and_source_hash(mode):
    width, threads = (2048, 256) if mode == "resident" else (4096, 512)
    config = _config(mode, width=width, threads=threads, rows=2, vector=8)
    with _cpu_target():
        bound = _bound(shape=(4096, 4096))
        with patch(
            "helion._compiler.cute.boolean_guards.reassociate_boolean_guards",
            side_effect=lambda body: body,
        ):
            before = bound.to_code(config)
        after = bound.to_code(config)
    assert before != after
    assert _long_guards(before)
    assert not _long_guards(after)
    assert _comparable(before) == _comparable(after)


def test_actual_guarded_bundle_preserves_zero_tail_and_oversized_launch_abi():
    config = _config("bounded_layout", threads=512, rows=2, vector=8)
    with _cpu_target():
        bound = _bound(shape=(4096, 4096))
        with patch(
            "helion._compiler.cute.boolean_guards.reassociate_boolean_guards",
            side_effect=lambda body: body,
        ):
            before = bound.to_code(config)
        after = bound.to_code(config)
        assert _comparable(before) == _comparable(after)
        modules = [PyCodeCache.load(source) for source in (before, after)]
        for columns in (0, 1, 513, 4096, 4097):
            tensor = torch.empty((3, columns), dtype=torch.bfloat16)
            records = []
            for module in modules:
                calls = []

                def capture(kernel, grid, *arguments, _calls=calls, **keywords):
                    _calls.append((kernel, grid, arguments, keywords))

                output = module.row_softmax(tensor, _launcher=capture)
                assert len(calls) == 1
                kernel, grid, arguments, keywords = calls[0]
                assert arguments[0] is tensor and arguments[1] is output
                assert output.shape == tensor.shape and output.dtype == tensor.dtype
                expected = (
                    module._helion_row_softmax_bounded
                    if 0 < columns <= 4096
                    else module._helion_row_softmax
                )
                assert kernel is expected
                records.append(
                    (kernel.__name__, grid, arguments[2:], copy.deepcopy(keywords))
                )
            assert records[0] == records[1]
            assert records[0][1] == (2,)
            assert records[0][2] == (3, columns)
            assert records[0][3]["block"] == (512, 2, 1)
