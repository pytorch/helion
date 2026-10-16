"""Prove a shared-memory ``hl.split`` exchange is race-free under lane loops.

The non-load ``hl.split`` fallback writes each thread's element to shared
memory, calls ``sync_threads`` and reads the two pair elements back.  Under
CuTe lane loops the write, the barrier and the reads all sit inside the same
lane iteration, so a pair element produced by *another* iteration of an
enclosing lane loop is not in shared memory yet when its partner reads it.

Whether that happens depends on the lane layout of the pair dim's block
(blocked ``tid * EPT + lane`` keeps ``lane`` as the fast index, strided or
synthetic ``tid + lane * T`` makes it the slow index, vector partitions add a
constexpr inner lane, ...) and on the pair geometry (halves vs. interleaved).
Instead of re-deriving every layout, this module evaluates the emitted index
expressions numerically over the whole thread block and every lane iteration
and checks, per lane iteration, that each slot a thread reads was written in
that same iteration and that every slot lies inside the staging buffer.  A
slot may be staged by several threads: a tile element is a function of its
logical coordinates, so redundant lanes (e.g. the surplus lanes of an
``hl.split`` output) write the same value.  Anything the evaluator cannot
resolve is rejected loudly.
"""

from __future__ import annotations

import ast
import itertools
from typing import TYPE_CHECKING

import numpy as np

from ... import exc
from ..compile_environment import CompileEnvironment
from ..tile_strategy import FlattenedTileStrategy

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST
    from ..tile_strategy import DeviceGridState

_Value = np.ndarray | int

_UNSUPPORTED_PREFIX = "hl.split of a non-load tile"


def _unsupported(reason: str) -> exc.BackendUnsupported:
    return exc.BackendUnsupported("cute", f"{_UNSUPPORTED_PREFIX}: {reason}")


def _static_range_extent(loop: ast.For) -> int | None:
    """Trip count of ``for v in range(N)`` / ``cutlass.range_constexpr(N)``."""
    call = loop.iter
    if not isinstance(call, ast.Call) or len(call.args) != 1 or call.keywords:
        return None
    extent = call.args[0]
    if isinstance(extent, ast.Constant) and type(extent.value) is int:
        return extent.value
    return None


def _lane_iterations(grid_state: DeviceGridState) -> list[tuple[str, int]] | None:
    """Every lane variable the wrapped body iterates, with its trip count.

    A vec-partitioned lane loop materializes as the wrapper's outer loop plus
    a constexpr inner loop; both are separate lane iterations as far as the
    per-iteration barrier is concerned.
    """
    lanes: dict[str, int] = {}
    for lane_var, extent in grid_state.lane_loops:
        wrapper = grid_state.vec_lane_wrappers.get(lane_var)
        if wrapper is None:
            lanes[lane_var] = extent
            continue
        outer = _static_range_extent(wrapper.outer_for)
        inner = _static_range_extent(wrapper.vloop)
        if outer is None or inner is None:
            return None
        lanes[lane_var] = outer
        lanes[wrapper.vec_lane_var] = inner
    return list(lanes.items())


def _collect_definitions(
    statement_lists: list[list[ast.AST]],
) -> dict[str, ast.expr]:
    """Map plain ``name = expr`` assignments to their right-hand sides."""
    definitions: dict[str, ast.expr] = {}
    for statements in statement_lists:
        for stmt in statements:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                definitions[stmt.targets[0].id] = stmt.value
    return definitions


def emitted_definitions(
    cg: GenerateAST, grid_state: DeviceGridState
) -> dict[str, ast.expr]:
    """The ``name = expr`` assignments emitted so far around the current body.

    The enclosing statement lists, the root body, the grid's prefix, its
    per-lane index setup and its vectorized lane wrappers.
    """
    return _collect_definitions(
        [
            *cg.statements_stack,
            cg.device_function.body,
            grid_state.outer_prefix,
            grid_state.lane_setup_statements,
            *(
                list(wrapper.outer_for.body)
                for wrapper in grid_state.vec_lane_wrappers.values()
            ),
        ]
    )


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


class _IndexEvaluator:
    """Evaluate integer index expressions over the thread block.

    Thread indices become broadcastable ``numpy`` arrays, lane variables take
    the current iteration's value, block-size constexprs their configured
    value, and every other name is looked up in the emitted definitions and
    evaluated recursively.  ``None`` means the expression is outside the
    supported fragment.
    """

    def __init__(
        self,
        definitions: dict[str, ast.expr],
        constants: dict[str, int],
        thread_ids: dict[int, np.ndarray],
    ) -> None:
        self.definitions = definitions
        self.constants = constants
        self.thread_ids = thread_ids
        self.block_idx = 0
        self.lane_values: dict[str, int] = {}
        self.used_lanes: set[str] = set()
        self._cache: dict[str, _Value | None] = {}
        self._resolving: set[str] = set()

    def set_lanes(self, lane_values: dict[str, int]) -> None:
        self.lane_values = lane_values
        self._cache = {}

    def set_block_idx(self, block_idx: int) -> None:
        self.block_idx = block_idx
        self._cache = {}

    def eval(self, node: ast.AST) -> _Value | None:
        if isinstance(node, ast.Constant):
            return node.value if type(node.value) is int else None
        if isinstance(node, ast.Name):
            return self._name(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = self.eval(node.operand)
            return None if operand is None else -operand
        if isinstance(node, ast.BinOp):
            return self._binop(node)
        if isinstance(node, ast.Call):
            # ``cutlass.Int32(x)`` and friends are value-preserving casts.
            func = _dotted_name(node.func)
            if (
                func is not None
                and func.startswith("cutlass.")
                and len(node.args) == 1
                and not node.keywords
            ):
                return self.eval(node.args[0])
            return None
        if isinstance(node, ast.Subscript):
            func = node.value
            if (
                isinstance(func, ast.Call)
                and not func.args
                and isinstance(node.slice, ast.Constant)
                and type(node.slice.value) is int
            ):
                name = _dotted_name(func.func)
                if name == "cute.arch.thread_idx":
                    return self.thread_ids.get(node.slice.value, 0)
                if name == "cute.arch.block_idx":
                    # Uniform across the block.  A staging slot must not depend
                    # on it (local coordinates subtract the tile base), which
                    # the caller checks by evaluating at two block indices.
                    return self.block_idx
            return None
        return None

    def _name(self, name: str) -> _Value | None:
        if name in self.lane_values:
            self.used_lanes.add(name)
            return self.lane_values[name]
        if name in self.constants:
            return self.constants[name]
        if name in self._cache:
            return self._cache[name]
        definition = self.definitions.get(name)
        if definition is None or name in self._resolving:
            return None
        self._resolving.add(name)
        value = self.eval(definition)
        self._resolving.discard(name)
        self._cache[name] = value
        return value

    def _binop(self, node: ast.BinOp) -> _Value | None:
        left = self.eval(node.left)
        right = self.eval(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
            if not np.all(np.asarray(right) != 0):
                return None
            return left // right if isinstance(node.op, ast.FloorDiv) else left % right
        return None


def _thread_ids(
    grid_state: DeviceGridState,
) -> tuple[tuple[int, ...], dict[int, np.ndarray]]:
    sizes = tuple(grid_state.thread_axis_sizes.get(axis, 1) for axis in range(3))
    thread_ids: dict[int, np.ndarray] = {}
    for axis, size in enumerate(sizes):
        shape = [1, 1, 1]
        shape[axis] = size
        thread_ids[axis] = np.arange(size, dtype=np.int64).reshape(shape)
    return sizes, thread_ids


def verify_split_smem_exchange(
    cg: GenerateAST,
    grid_state: DeviceGridState,
    numel: int,
    write_index: str,
    read_indices: list[str],
) -> None:
    """Raise ``BackendUnsupported`` unless the per-iteration exchange is safe.

    ``write_index`` is the flat staging slot this thread writes in the current
    lane iteration and ``read_indices`` the slots it reads back after the
    barrier; all are expressions over the emitted per-thread index variables.
    """
    lanes = _lane_iterations(grid_state)
    if lanes is None:
        raise _unsupported("lane loop with a non-static trip count")

    env = CompileEnvironment.current()
    df = cg.device_function
    constants: dict[str, int] = {}
    for info in env.block_sizes:
        name = df.block_size_var(info.block_id)
        size = df.resolved_block_size(info.block_id)
        if name is not None and isinstance(size, int):
            constants[name] = size
    for strategy in df.tile_strategy.strategies:
        # The blocks of a flattened tile share one variable, the product of
        # their sizes.
        if isinstance(strategy, FlattenedTileStrategy) and isinstance(
            strategy.block_size, int
        ):
            constants[strategy.block_size_var(-1)] = strategy.block_size
    grid_shape, thread_ids = _thread_ids(grid_state)
    evaluator = _IndexEvaluator(
        emitted_definitions(cg, grid_state), constants, thread_ids
    )
    expressions = [
        ast.parse(expr, mode="eval").body for expr in (write_index, *read_indices)
    ]

    def evaluate(index: int) -> np.ndarray:
        value = evaluator.eval(expressions[index])
        if value is None:
            raise _unsupported(
                "staging slot index uses an expression the exchange check "
                "cannot evaluate"
            )
        return np.broadcast_to(np.asarray(value, dtype=np.int64), grid_shape).reshape(
            -1
        )

    # Only lane loops the slot indices depend on change the staged element;
    # iterating an unrelated live lane loop just rewrites the same slots.
    evaluator.set_lanes(dict.fromkeys((name for name, _ in lanes), 0))
    for index in range(len(expressions)):
        evaluate(index)
    used = [(name, extent) for name, extent in lanes if name in evaluator.used_lanes]

    for values in itertools.product(*(range(extent) for _, extent in used)):
        evaluator.set_lanes(dict(zip((name for name, _ in used), values, strict=True)))
        evaluator.set_block_idx(0)
        slots = evaluate(0)
        if slots.min() < 0 or slots.max() >= numel:
            raise _unsupported("staging slot index falls outside the tile")
        # Several threads may stage one slot (redundant lanes of a split
        # output hold the same logical element), so only read coverage within
        # the iteration matters.
        reads = [evaluate(index) for index in range(1, len(expressions))]
        for read in reads:
            if not np.isin(read, slots).all():
                raise _unsupported(
                    "a pair element is produced by a different iteration of "
                    "an enclosing lane loop than the one that reads it"
                )
        # Block-local slots are the same in every block; an index that kept
        # its tile base would validate in block 0 and fault everywhere else.
        evaluator.set_block_idx(1)
        if not np.array_equal(evaluate(0), slots) or any(
            not np.array_equal(evaluate(index), read)
            for index, read in enumerate(reads, start=1)
        ):
            raise _unsupported("staging slot index depends on the block index")
