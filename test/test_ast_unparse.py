from __future__ import annotations

import ast
import copy
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest

from helion._compiler.ast_extension import ExtendedAST
from helion._compiler.ast_extension import LoopType
from helion._compiler.ast_extension import create
from helion._compiler.ast_extension import expr_from_string
from helion._compiler.ast_extension import fix_missing_locations
from helion._compiler.ast_extension import unparse
from helion._compiler.ast_read_writes import ReadWrites
from helion._compiler.ast_read_writes import _ReadWriteVisitor
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.device_state import CuteDeviceFunctionState
from helion._compiler.device_function import DeviceFunction
from helion._compiler.output_header import get_needed_import_lines
from helion._compiler.type_printer import ASTPrinter
from helion._compiler.type_printer import print_ast
from helion.runtime.config import Config
from helion.runtime.kernel import fix_missing_locations as runtime_fix_missing_locations

if TYPE_CHECKING:
    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan


@pytest.mark.parametrize(
    "source",
    [
        "a - (b - c)",
        "a + (b + c)",
        "(a ** b) ** c",
        "a ** b ** c",
        "-a ** b + a ** -b",
        "a * (b / c)",
        "a @ (b @ c)",
        "a & b | c ^ d",
        "(a | b) & c",
        "(a < b) & (c < d)",
        "a + f(b - c, value=d ** e)[i] * -g",
        "(a if p else b) + (c and d)",
        "(a << b) >> (c + d)",
        "a // (b % c)",
    ],
)
def test_binary_unparse_preserves_stdlib_grouping(source: str) -> None:
    tree = ast.parse(source, mode="eval")
    expected = ast.unparse(tree)
    assert unparse(tree, output_origin_lines=False) == expected
    assert ast.dump(ast.parse(expected, mode="eval")) == ast.dump(tree)


def test_binary_unparse_preserves_float_evaluation_and_operand_order() -> None:
    source = "take(0) + (take(1) + take(2)) - take(3) ** (take(4) ** take(5))"
    tree = ast.parse(source, mode="eval")
    calls: list[int] = []
    values = (1e20, -1e20, 3.0, 2.0, 3.0, 2.0)

    def take(index: int) -> float:
        calls.append(index)
        return values[index]

    expected = eval(compile(tree, "<original-expression>", "eval"), {"take": take})
    assert calls == list(range(6))
    calls.clear()
    actual = eval(unparse(tree, output_origin_lines=False), {"take": take})
    assert actual == expected == -512.0
    assert calls == list(range(6))


@pytest.mark.parametrize(
    "source",
    [
        "worker_id + i * sm_count",
        "a - (b - c)",
        "(a ** b) ** c",
        "a ** b ** c",
    ],
)
def test_binary_debug_annotations_match_recursive_traversal(source: str) -> None:
    tree = expr_from_string(source)
    binary_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.BinOp)]
    for node in binary_nodes:
        assert isinstance(node, ExtendedAST)
        node._loop_type = LoopType.DEVICE
    with patch.object(ASTPrinter, "visit_BinOp", ast._Unparser.visit_BinOp):
        expected = print_ast(tree)
    actual = print_ast(tree)
    assert actual == expected
    assert actual.count("# BinOp: loop_type=DEVICE") == len(binary_nodes)


def test_deep_binary_debug_annotations_do_not_recurse() -> None:
    tree: ast.AST = ast.Name(id="value", ctx=ast.Load())
    depth = 1200
    for index in range(depth):
        tree = create(
            ast.BinOp,
            left=tree,
            op=ast.BitAnd(),
            right=ast.Constant(value=index),
            _loop_type=LoopType.DEVICE,
        )
    limit = sys.getrecursionlimit()
    actual = print_ast(tree)
    assert actual.count("# BinOp: loop_type=DEVICE") == depth
    assert sys.getrecursionlimit() == limit


def test_binary_read_write_walk_retains_existing_side_effect_accounting() -> None:
    tree = ast.parse(
        """
for ignored in range(count):
    output[index] += (
        hl.atomic_add(state, [index], value)
        + tensor[index] * scale
        - hl.store(other, [index], tensor.dtype)
    )
"""
    )
    with patch.object(_ReadWriteVisitor, "visit_BinOp", ast.NodeVisitor.generic_visit):
        expected = ReadWrites.from_ast(tree)
    actual = ReadWrites.from_ast(tree)
    assert actual == expected
    assert list(actual.reads) == list(expected.reads)
    assert list(actual.writes) == list(expected.writes)
    assert "ignored" not in actual.writes
    assert actual.atomic_reads == {"state"}
    assert actual.inplace_writes == {"output": 1, "state": 1, "other": 1}


@pytest.mark.parametrize("locations", ["defaults", "partial", "shared"])
def test_iterative_locations_match_stdlib_inheritance(locations: str) -> None:
    shared = ast.Name(id="value", ctx=ast.Load())
    first = ast.Expr(value=shared)
    second = ast.Expr(value=shared)
    tree = ast.Module(body=[first, second], type_ignores=[])
    if locations != "defaults":
        first.lineno, first.col_offset = 8, 4
        first.end_lineno, first.end_col_offset = 9, 20
        second.lineno, second.col_offset = 80, 40
        second.end_lineno, second.end_col_offset = 90, 200
    if locations == "partial":
        shared.lineno, shared.col_offset = 3, 7
        shared.end_lineno, shared.end_col_offset = None, None
    expected = ast.fix_missing_locations(copy.deepcopy(tree))
    assert fix_missing_locations(tree) is tree
    assert ast.dump(tree, include_attributes=True) == ast.dump(
        expected, include_attributes=True
    )
    assert first.value is second.value
    if locations == "shared":
        assert (shared.lineno, shared.col_offset, shared.end_lineno) == (8, 4, 9)


@pytest.mark.parametrize("depth", [600, 1200])
def test_deep_predicate_passes_launcher_import_scan_and_final_renderer(
    depth: int,
) -> None:
    terms = [f"(cutlass.Int32(index) + {i} < limit)" for i in range(depth)]
    predicate = " & ".join(terms)
    original_source = (
        "def generated(index, limit, value):\n"
        "    lane = cute.arch.thread_idx()[0]\n"
        "    for step in range(2):\n"
        f"        result = cutlass.Float32(value) if {predicate} else cutlass.Float32(0)\n"
        "    return result\n"
    )
    tree = ast.parse(original_source)
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    limit = sys.getrecursionlimit()
    with pytest.raises(RecursionError):
        ast.unparse(tree)
    with (
        patch.object(_ReadWriteVisitor, "visit_BinOp", ast.NodeVisitor.generic_visit),
        pytest.raises(RecursionError),
    ):
        ReadWrites.from_ast(tree)

    state = CuteDeviceFunctionState()
    state.chained_matmul_plan = cast("ChainedMatmulPlan", SimpleNamespace(threads=128))
    state.shared_reduce_launch_block = (128, 1, 1)
    device = SimpleNamespace(
        codegen=object(),
        tile_strategy=object(),
        preamble=[],
        body=function.body,
        cute_state=state,
    )
    with patch.object(DeviceFunction, "current", return_value=device):
        with (
            patch(
                "helion._compiler.cute.backend.unparse",
                side_effect=lambda node, **kwargs: ast.unparse(node),
            ),
            pytest.raises(RecursionError),
        ):
            CuteBackend().launcher_keyword_args(Config(), has_barrier=False)
        launcher = CuteBackend().launcher_keyword_args(Config(), has_barrier=False)
    assert launcher == ["block=(128, 1, 1)"]
    with patch(
        "helion._compiler.output_header._active_library_imports",
        return_value={
            "cute": "import cutlass.cute as cute",
            "cutlass": "import cutlass",
        },
    ):
        assert get_needed_import_lines(tree) == [
            "import cutlass.cute as cute",
            "import cutlass",
        ]
    if depth == 1200:
        with pytest.raises(RecursionError):
            ast.fix_missing_locations(tree)
    assert runtime_fix_missing_locations is fix_missing_locations
    assert runtime_fix_missing_locations(tree) is tree
    source = unparse(tree)
    assert source == unparse(tree, output_origin_lines=False)
    assert predicate in source
    assert source == original_source.rstrip()
    assert sys.getrecursionlimit() == limit
