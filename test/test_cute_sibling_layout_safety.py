from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import operator
from types import CodeType
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

from examples.welford import welford
import numpy as np
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_shared_tile_extents import _launch_dims
from test.test_cute_sibling_loop_threads import _context
from test.test_cute_sibling_loop_threads import _row_ir

import helion
from helion._compiler.cute.layout_propagation import _plan_warp_per_row_execution
from helion._compiler.cute.memory_ops import _cute_vector_load_mask_is_lane_only
from helion._compiler.tile_dispatch import TileStrategyDispatch
from helion._testing import skipUnlessBackends

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion._compiler.device_ir import RootGraphInfo


@pytest.mark.parametrize("threads", ((1, 32), (32, 1)))
def test_scalar_sibling_does_not_change_the_shared_row_axis(
    threads: tuple[int, int],
) -> None:
    ir = _row_ir()
    config = helion.Config(block_sizes=[8, 128, 128], num_threads=[8, *threads])
    with _context(ir, config) as (_env, fn):
        dispatch = TileStrategyDispatch(fn, config)
        fn.tile_strategy = dispatch
        _plan_warp_per_row_execution(ir.graphs[0], dispatch)
        root = cast("RootGraphInfo", ir.graphs[0])
        assert not root.cute_grid_execution_plans
        assert dispatch.thread_block_dims() == (8, 32, 1)
        assert dispatch.thread_axis_for_block_id(0) == 0
        threaded_sibling = threads.index(32) + 1
        assert dispatch.thread_axis_for_block_id(threaded_sibling) == 1


def test_scalar_child_inside_threaded_pass_keeps_valid_row_plan() -> None:
    ir = _row_ir(nested=True)
    config = helion.Config(block_sizes=[8, 128, 128], num_threads=[8, 32, 1])
    with _context(ir, config) as (_env, fn):
        dispatch = TileStrategyDispatch(fn, config)
        fn.tile_strategy = dispatch
        _plan_warp_per_row_execution(ir.graphs[0], dispatch)
        root = cast("RootGraphInfo", ir.graphs[0])
        assert root.cute_grid_execution_plans
        assert dispatch.thread_block_dims() == (32, 8, 1)
        assert dispatch.thread_axis_for_block_id(0) == 1
        assert dispatch.thread_axis_for_block_id(1) == 0


@pytest.mark.parametrize(
    ("mask", "allowed"),
    (
        (None, True),
        ("True", True),
        ("mask_n", True),
        ("mask_n and (True and mask_n)", True),
        ("mask_m", False),
        ("mask_m and mask_n", False),
        ("mask_n or mask_m", False),
        ("not mask_n", False),
        ("1", False),
        ("rows.load() < limit", False),
    ),
)
def test_vector_load_anchor_only_discharges_its_own_mask(
    mask: str | None, allowed: bool
) -> None:
    assert _cute_vector_load_mask_is_lane_only(mask, "mask_n") is allowed


def _config(kind: str) -> helion.Config:
    if kind == "scalar_stats":
        # Replay the captured config, including legacy cache-policy aliases.
        return helion.Config.from_json(
            """{
                "block_sizes": [8, 128, 32],
                "num_threads": [0, 1, 32],
                "cute_vector_widths": [4, 4, 2],
                "cute_lane_layouts": ["blocked", "blocked", "strided"],
                "cute_cluster_n": 2,
                "cute_min_blocks_per_mp": 2,
                "load_eviction_policies": ["", "streaming", "l2_last", "first"]
            }"""
        )
    if kind == "row_tail":
        return helion.Config(
            block_sizes=[8, 32, 32],
            num_threads=[8, 8, 8],
            cute_vector_widths=[1, 4, 4],
            cute_lane_layouts=["blocked", "strided", "strided"],
            cute_cluster_n=1,
        )
    if kind == "broadcast_count":
        return helion.Config(
            block_sizes=[128, 2, 64],
            num_threads=[0, 0, 8],
            cute_vector_widths=[4, 2, 2],
            cute_lane_layouts=["strided", "strided", "blocked"],
            cute_cluster_n=2,
            cute_min_blocks_per_mp=3,
            load_eviction_policies=["last", "last", "first", "last"],
        )
    assert kind == "warp_per_row"
    return helion.Config(
        block_sizes=[4, 128, 128],
        num_threads=[4, 32, 32],
        cute_vector_widths=[1, 4, 4],
        cute_lane_layouts=["blocked", "strided", "strided"],
        cute_cluster_n=1,
    )


@lru_cache(None)
def _code(rows: int, columns: int, kind: str, shared_budget: int = 232448) -> str:
    kernel = helion.kernel(
        welford.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=shared_budget,
        ),
    ):
        bound = _cpu_bind(
            kernel,
            (torch.empty(columns), torch.empty(columns), torch.empty(rows, columns)),
        )
        return bound.to_code(_config(kind))


def _effects(
    expr: ast.expr, rewrite: Callable[[ast.Call], ast.expr | None]
) -> ast.expr | None:
    """Keep selected calls and their actual predicates, discarding data arithmetic."""
    if isinstance(expr, ast.Call) and (replacement := rewrite(expr)) is not None:
        return replacement
    if isinstance(expr, ast.IfExp):
        left, right = _effects(expr.body, rewrite), _effects(expr.orelse, rewrite)
        if left is None and right is None:
            return None
        return ast.IfExp(
            deepcopy(expr.test), left or ast.Constant(None), right or ast.Constant(None)
        )
    children = [
        effect
        for child in ast.iter_child_nodes(expr)
        if isinstance(child, ast.expr)
        and (effect := _effects(child, rewrite)) is not None
    ]
    if not children:
        return None
    assert not isinstance(expr, ast.BoolOp), "fixture has a short-circuit memory effect"
    return children[0] if len(children) == 1 else ast.Tuple(children, ast.Load())


def _projection(
    tree: ast.Module, rewrite: Callable[[ast.Call], ast.expr | None]
) -> tuple[CodeType, str]:
    """Execute the emitted address/control statements, without floating point math.

    This fixture has input-independent addresses. Preserve all assignments to
    their dependencies (including shadowing and loop-carried coordinates), and
    reject a projection whose address or predicate would depend on loaded data.
    """
    device = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    statements = list(ast.walk(device))
    effects = {
        id(node): effect
        for node in statements
        if isinstance(node, (ast.Assign, ast.Expr))
        and (effect := _effects(node.value, rewrite)) is not None
    }
    controls = [
        node.iter if isinstance(node, ast.For) else node.test
        for node in statements
        if isinstance(node, (ast.For, ast.If))
    ]

    def reads(expr: ast.AST) -> set[str]:
        return {
            node.id
            for node in ast.walk(expr)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }

    needed = set().union(*(reads(expr) for expr in [*effects.values(), *controls]))
    while True:
        previous = needed.copy()
        for node in statements:
            if isinstance(node, ast.Assign):
                assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                if node.targets[0].id in needed:
                    assert id(node) not in effects, "data-dependent address fixture"
                    needed.update(reads(node.value))
        if needed == previous:
            break

    def body(nodes: list[ast.stmt]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for node in nodes:
            if (
                isinstance(node, ast.Assign)
                and cast("ast.Name", node.targets[0]).id in needed
            ):
                result.append(deepcopy(node))
            elif id(node) in effects:
                result.append(ast.Expr(effects[id(node)]))
            elif isinstance(node, (ast.For, ast.If)):
                new = deepcopy(node)
                new.body = body(node.body)
                new.orelse = body(node.orelse) if node.orelse else []
                result.append(new)
        return result or [ast.Pass()]

    function = deepcopy(device)
    function.decorator_list = []
    function.body = body(device.body)
    constants = [
        deepcopy(node)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("_BLOCK_SIZE_")
    ]
    module = ast.fix_missing_locations(ast.Module([*constants, function], []))
    return compile(module, "<emitted memory projection>", "exec"), function.name


@dataclass(frozen=True)
class _Pointer:
    name: str
    size: int
    offset: int = 0

    def __add__(self, offset: int) -> _Pointer:
        return _Pointer(self.name, self.size, self.offset + offset)


def _memory_coverage(
    code: str, rows: int, columns: int, block_id: int
) -> tuple[Counter[int], Counter[str]]:
    def rewrite(call: ast.Call) -> ast.expr | None:
        name = ast.unparse(call.func)
        if name == "cute.arch.load":
            dtype = call.args[1]
            width = (
                ast.literal_eval(dtype.args[0])[0] if isinstance(dtype, ast.Call) else 1
            )
            return ast.Call(
                ast.Name("_read", ast.Load()),
                [deepcopy(call.args[0]), ast.Constant(width)],
                [],
            )
        if isinstance(call.func, ast.Attribute) and call.func.attr == "load":
            return ast.Call(
                ast.Name("_read", ast.Load()),
                [deepcopy(call.func.value), ast.Constant(1)],
                [],
            )
        if isinstance(call.func, ast.Attribute) and call.func.attr == "store":
            return ast.Call(
                ast.Name("_write", ast.Load()),
                [deepcopy(call.func.value), ast.Constant(1)],
                [],
            )
        if name == "_cute_store_u32_vec":
            length = ast.Call(ast.Name("len", ast.Load()), [deepcopy(call.args[1])], [])
            return ast.Call(
                ast.Name("_write", ast.Load()), [deepcopy(call.args[0]), length], []
            )
        if isinstance(call.func, ast.Attribute) and call.func.attr == "append":
            return ast.Call(deepcopy(call.func), [ast.Constant(0)], [])
        return None

    tree = ast.parse(code)
    program, name = _projection(tree, rewrite)
    current_thread = (0, 0, 0)
    output: Counter[int] = Counter()
    reads: Counter[str] = Counter()

    def read(pointer: _Pointer, width: int) -> None:
        assert 0 <= pointer.offset <= pointer.size - width, (pointer, width)
        reads[pointer.name] += width

    def write(pointer: _Pointer, width: int) -> None:
        assert pointer.name == "out"
        assert 0 <= pointer.offset <= pointer.size - width, (pointer, width)
        output.update(range(pointer.offset, pointer.offset + width))

    scope: dict[str, object] = {
        "operator": operator,
        "cutlass": SimpleNamespace(Int32=int, Int64=int, range_constexpr=range),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                thread_idx=lambda: current_thread, block_idx=lambda: (block_id, 0, 0)
            )
        ),
        "_read": read,
        "_write": write,
    }
    exec(program, scope)
    run = cast("Callable[..., None]", scope[name])
    tensors = [
        SimpleNamespace(
            iterator=_Pointer(label, size), layout=SimpleNamespace(stride=strides)
        )
        for label, size, strides in (
            ("x", rows * columns, (columns, 1)),
            ("weight", columns, (1,)),
            ("bias", columns, (1,)),
            ("out", rows * columns, (columns, 1)),
        )
    ]
    for thread in product(*(range(size) for size in _launch_dims(tree))):
        current_thread = thread
        run(*tensors, 1e-5)
    return output, reads


@pytest.mark.parametrize(
    ("kind", "rows", "columns"),
    (
        ("scalar_stats", 16384, 1024),
        ("scalar_stats", 17, 80),
        ("row_tail", 17, 80),
        ("row_tail", 17, 77),
        ("warp_per_row", 16, 256),
        ("warp_per_row", 17, 80),
    ),
)
@skipUnlessBackends(["cute"])
def test_emitted_sibling_addresses_and_output_coverage(
    kind: str, rows: int, columns: int
) -> None:
    code = _code(rows, columns, kind)
    block_rows = _config(kind).block_sizes[0]
    for block_id in {0, (rows - 1) // block_rows}:
        output, reads = _memory_coverage(code, rows, columns, block_id)
        first = block_id * block_rows * columns
        end = min((block_id + 1) * block_rows, rows) * columns
        assert output == Counter(range(first, end))
        assert set(reads) == {"x", "weight", "bias"}
    if kind == "warp_per_row" and rows % block_rows == 0 and columns % 128 == 0:
        # Preserve the existing fully valid vector-load/warp-per-row path.
        assert "ir.VectorType.get([4]" in code
        assert _launch_dims(ast.parse(code)) == (32, 4, 1)


def _broadcast_counts(code: str, columns: int) -> list[list[int]]:
    tree = ast.parse(code)
    device = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(device)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    pending = [assignments["Tn"]]
    visited: set[int] = set()
    reductions: list[ast.Call] = []
    while pending:
        expression = pending.pop()
        if id(expression) in visited:
            continue
        visited.add(id(expression))
        for node in ast.walk(expression):
            if isinstance(node, ast.Call) and ast.unparse(node.func) in {
                "_cute_grouped_reduce_shared_two_stage",
                "cute.arch.warp_reduction_sum",
            }:
                reductions.append(node)
            elif isinstance(node, ast.Name) and node.id in assignments:
                pending.append(assignments[node.id])
    (reduction,) = reductions
    grouped = ast.unparse(reduction.func) == "_cute_grouped_reduce_shared_two_stage"
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in reduction.keywords}
    launch = _launch_dims(tree)
    pre, span = (
        (kwargs["pre"], kwargs["group_span"])
        if grouped
        else (1, kwargs["threads_in_group"])
    )
    current_thread = (0, 0, 0)
    current_inputs: list[int] = []

    def rewrite(call: ast.Call) -> ast.expr | None:
        if call is not reduction:
            return None
        args = [deepcopy(call.args[0])]
        if grouped:
            args.extend((deepcopy(call.args[2]), deepcopy(call.args[3])))
        return ast.Call(ast.Name("_input", ast.Load()), args, [])

    program, name = _projection(tree, rewrite)

    def record(
        value: np.int64, identity: np.int64 | None = None, linear: int | None = None
    ) -> None:
        assert isinstance(value, np.int64), "sum must retain its promoted input"
        if identity is not None:
            assert type(value) is type(identity), "shared selection types must agree"
        expected = current_thread[0] + launch[0] * (
            current_thread[1] + launch[1] * current_thread[2]
        )
        if linear is not None:
            assert linear == expected
        current_inputs.append(int(value))

    scope: dict[str, object] = {
        "operator": operator,
        "cutlass": SimpleNamespace(
            Int32=np.int32, Int64=np.int64, range_constexpr=range
        ),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                thread_idx=lambda: current_thread, block_idx=lambda: (0, 0, 0)
            )
        ),
        "_input": record,
    }
    exec(program, scope)
    run = cast("Callable[..., None]", scope[name])
    values: list[list[int]] = []
    for z, y, x in product(range(launch[2]), range(launch[1]), range(launch[0])):
        current_thread = (x, y, z)
        current_inputs = []
        run(None, None, None, None, 1e-5)
        values.append(current_inputs)
    assert all(len(value) == (columns + 1) // 2 for value in values)
    groups: dict[tuple[int, int], list[int]] = {}
    for lane, entries in enumerate(values):
        totals = groups.setdefault((lane // span, lane % pre), [0] * len(entries))
        for iteration, entry in enumerate(entries):
            totals[iteration] += entry
    return [groups[lane // span, lane % pre] for lane in range(len(values))]


@pytest.mark.parametrize("columns", (16, 17, 1024))
@skipUnlessBackends(["cute"])
def test_broadcast_count_uses_physical_row_stride(columns: int) -> None:
    actual = _broadcast_counts(_code(128, columns, "broadcast_count"), columns)
    expected = [min(2, columns - offset) for offset in range(0, columns, 2)]
    assert actual == [expected] * (128 * 8)


@pytest.mark.parametrize("shared_budget", (0, 20000))
@skipUnlessBackends(["cute"])
def test_unavailable_strided_reduce_storage_does_not_fall_back_to_wrong_warp(
    shared_budget: int,
) -> None:
    # The count needs Int64 storage: 20,000 bytes would only fit the old Int32
    # estimate. Never fall through to a consecutive-lane warp reduction.
    with pytest.raises(helion.exc.BackendUnsupported, match="shared-memory budget"):
        _code(128, 16, "broadcast_count", shared_budget=shared_budget)
