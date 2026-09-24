"""Two full independent tiles per CTA, sharing the typed scalar coefficient."""

from __future__ import annotations

import ast
import copy
import dataclasses
import math
from typing import TYPE_CHECKING
from typing import cast

from ..ast_read_writes import ast_rename
from ..compile_environment import CompileEnvironment
from ..program_id import FlatProgramIDs
from .serial_lane_recurrence import _axis
from .serial_lane_recurrence import _names
from .serial_lane_recurrence import _require
from .serial_lane_recurrence import _typed
from .serial_lane_recurrence import _value

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..device_function import DeviceFunction
    from ..tile_strategy import PerThreadNDTileStrategy
    from .serial_lane_recurrence import Recurrence

KEY = "cute_serial_lane_coarsen"


@dataclasses.dataclass(frozen=True)
class CoarsenPlan:
    index: str
    block: int
    grid: int
    stride: int
    tiles: int


def paired_axis(
    rec: Recurrence, vector: int, root: list[int], blocks: dict[int, int]
) -> int | None:
    """Select by typed coordinates/strides, never source names or native shapes."""
    env = CompileEnvironment.current()
    scalar_axes = {
        _axis(index)
        for load in rec.loads
        if load is not rec.state
        for index in cast("list[object]", load.args[1])
    }
    output = _value(rec.store.args[0])
    candidates = [
        axis
        for axis in root
        if axis != vector
        and axis not in scalar_axes
        and axis in rec.axes
        and axis in blocks
        and env.block_sizes[axis].size_hint() % (2 * blocks[axis]) == 0
    ]
    return (
        min(candidates, key=lambda a: output.stride(rec.axes.index(a)))
        if candidates
        else None
    )


def capture_pair(
    df: DeviceFunction,
    strategy: PerThreadNDTileStrategy,
    rec: Recurrence,
    vector: int,
    width: int,
) -> CoarsenPlan:
    _require(width == 4)
    _require(df.config.config.get("cute_serial_lane_schedule") == "step_major_vector")
    _require(
        df.config.config.get("cute_serial_lane_load_schedule")
        in ("prefetch2", "prefetch4")
    )
    pids = _typed(df.pid, FlatProgramIDs)
    _require(
        pids.shared_pid_var is None and not df.config.config.get("xcd_remap", False)
    )
    root = [p.block_id for p in pids.pid_info]
    _require(set(root) == set(strategy.block_ids) and len(root) == len(set(root)))
    blocks = {a: _typed(df.resolved_block_size(a), int) for a in root}
    axis = paired_axis(rec, vector, root, blocks)
    _require(axis is not None)
    assert axis is not None
    _require(strategy._elements_per_thread_for_block(axis) == 1)
    env = CompileEnvironment.current()
    counts = [env.block_sizes[a].size_hint() // blocks[a] for a in root]
    _require(all(type(n) is int and n > 0 for n in counts))
    grid = math.prod(counts)
    _require(grid < 2**31 and grid % 2 == 0)
    position = root.index(axis)
    # Actual FlatProgramIDs consumes pid_info in fast-to-slow grid order.
    return CoarsenPlan(
        strategy.index_var(axis),
        blocks[axis],
        grid,
        math.prod(counts[:position]),
        counts[position],
    )


def expand_banks(
    df: DeviceFunction,
    body: list[ast.stmt],
    index: str,
    block: int,
    private: set[str],
    *,
    split_tail: bool = False,
) -> list[ast.stmt]:
    """Clone only bank-local statements; serial/future guards remain shared."""
    body = ast.parse(ast.unparse(ast.Module(body=body, type_ignores=[]))).body
    if split_tail:
        # A peeled suffix is straight-line; only the earlier dynamic group is
        # shared across banks. Lane-local constexpr loops still clone intact.
        dynamic_groups = [
            node
            for node in body
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ]
        _require(len(dynamic_groups) <= 1)
        group = dynamic_groups[0] if dynamic_groups else None
    else:
        group = _typed(body[-1], ast.For)
    tainted = {*private, index}

    def groups(nodes: list[ast.stmt]) -> Iterator[ast.stmt]:
        for node in nodes:
            if node is group or (
                isinstance(node, ast.If) and not (_names(node.test) & tainted)
            ):
                yield from groups(node.body)
                yield from groups(node.orelse)
            else:
                yield node

    # Fixed-point SSA closure includes vector-pointer temporaries and the
    # original scalar-copy fallback. Scalar coefficient loads/exp are outside.
    while True:
        old = set(tainted)
        for node in groups(body):
            if any(isinstance(n, ast.Name) and n.id in tainted for n in ast.walk(node)):
                tainted.update(
                    n.id
                    for n in ast.walk(node)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                )
        if old == tainted:
            break
    mapping = {
        name: df.new_var(name + "_bank1", dce=False)
        for name in sorted(tainted - {index})
    }

    class Shift(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id == index and isinstance(node.ctx, ast.Load):
                return ast.BinOp(
                    left=copy.deepcopy(node), op=ast.Add(), right=ast.Constant(block)
                )
            return node

    def expand(nodes: list[ast.stmt]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for node in nodes:
            if node is group or (
                isinstance(node, ast.If) and not (_names(node.test) & tainted)
            ):
                node.body = expand(node.body)
                node.orelse = expand(node.orelse)
                result.append(node)
            else:
                result.append(node)
                if any(
                    isinstance(n, ast.Name) and n.id in tainted for n in ast.walk(node)
                ):
                    result.append(
                        _typed(
                            Shift().visit(ast_rename(copy.deepcopy(node), mapping)),
                            ast.stmt,
                        )
                    )
        return result

    return expand(body)


def remap_pid(
    df: DeviceFunction, fn: ast.FunctionDef, pair: CoarsenPlan
) -> ast.FunctionDef:
    """Map one physical CTA to the first of two adjacent logical-axis tiles."""
    span = pair.stride * (pair.tiles // 2)
    name = df.new_var("serial_pair_pid", dce=False)
    value = df.new_var("serial_physical_pid", dce=False)
    count = 0

    class Remap(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            nonlocal count
            if ast.unparse(node) == "cute.arch.block_idx()[0]":
                count += 1
                return ast.Name(id=name, ctx=ast.Load())
            return self.generic_visit(node)

    # Mutate this candidate only; keep existing node identities for the later
    # lane-loop replacement visitor.
    Remap().visit(fn)
    _require(count > 0)
    fn.body[:0] = ast.parse(
        f"{value} = cutlass.Int32(cute.arch.block_idx()[0])\n"
        f"{name} = ({value} // {span}) * {2 * span} + "
        f"(({value} % {span}) // {pair.stride}) * {2 * pair.stride} + "
        f"{value} % {pair.stride}"
    ).body
    return fn
