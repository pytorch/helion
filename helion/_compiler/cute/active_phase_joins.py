"""Limit local grid notifications to CTAs that can execute neighboring phases."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ...exc import InvalidConfig
from ..ast_extension import expr_from_string
from ..program_id import NUM_SM_VAR
from .autotune.config_keys import STAGED_GRID_KEY

if TYPE_CHECKING:
    from ..device_function import DeviceFunction

# Avoid different boundaries and their spin loads sharing a cache line.
COUNTER_STRIDE = 32


def _count(loop: ast.For) -> int | None:
    """Recognize the exact zero-trip condition of the interleaved PID emitter."""
    if loop.orelse:
        return None
    for x in ast.walk(loop.iter):
        if not (
            isinstance(x, ast.BinOp)
            and isinstance(x.op, ast.Sub)
            and isinstance(x.left, ast.Constant)
            and isinstance(x.right, ast.Constant)
            and type(x.left.value) is int
            and type(x.right.value) is int
            and 0 <= x.right.value < x.left.value < 2**31
        ):
            continue
        diff = f"({x.left.value} - {x.right.value})"
        nsm = NUM_SM_VAR
        expected = expr_from_string(
            "range(cutlass.Int32(0), "
            f"cutlass.Int32({diff} // {nsm} + cutlass.Int32("
            f"cutlass.Int32(cute.arch.block_idx()[0]) < {diff} % {nsm})))"
        )
        if ast.dump(loop.iter) == ast.dump(expected):
            return x.left.value - x.right.value
    return None


_ALLOCATORS = frozenset(
    {
        "cute.arch.alloc_smem",
        "cute.make_layout",
        "cute.make_rmem_tensor",
        "cute.make_tensor",
        "cutlass.utils.SmemAllocator",
    }
)


def _local_setup(stmt: ast.stmt, allocators: set[str]) -> bool:
    """Only descriptors, private storage allocations and scalar literals."""
    if not (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(target := stmt.targets[0], ast.Name)
    ):
        return False
    # Track the origin of a bound method; an unrelated allocator may have
    # external effects, even if it also calls its method 'allocate_tensor'.
    was_allocator = {
        f"{name}.allocate_tensor" for name in allocators if name != target.id
    }
    allocators.discard(target.id)
    for node in ast.walk(stmt.value):
        if isinstance(node, ast.Call):
            if ast.unparse(node.func) not in _ALLOCATORS | was_allocator:
                return False
        elif not isinstance(
            node,
            (ast.Name, ast.Constant, ast.Tuple, ast.Attribute, ast.Load, ast.keyword),
        ):
            return False
    if (
        isinstance(stmt.value, ast.Call)
        and ast.unparse(stmt.value.func) == "cutlass.utils.SmemAllocator"
    ):
        allocators.add(target.id)
    return True


def participants(region: list[ast.stmt]) -> int | None:
    """Return a conservative active prefix, or None for the whole grid.

    Every effect must occur in a recognized loop. Locally allocated storage
    is initialized independently on each CTA. A hoisted load, store, or an
    unknown operation outside such a loop makes all CTAs participate.
    """
    counts: list[int] = []
    allocators: set[str] = set()
    for stmt in region:
        if isinstance(stmt, ast.For) and (n := _count(stmt)) is not None:
            counts.append(n)
            # Any name rebound by the region ceases to be a proven allocator.
            allocators.difference_update(
                node.id
                for node in ast.walk(stmt)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            )
        elif not _local_setup(stmt, allocators):
            return None
    return max(counts, default=None)


def apply_active_joins(
    fn: DeviceFunction,
    definition: ast.FunctionDef,
    checked_joins: dict[ast.stmt, ast.Call],
) -> None:
    """Called after occupancy and unconditional local joins have been checked."""
    from ..device_function import TensorArg

    if fn.config.get(STAGED_GRID_KEY, 1) == 1 or any(
        isinstance(n, ast.Name)
        and n.id == NUM_SM_VAR
        and isinstance(n.ctx, (ast.Store, ast.Del))
        for n in ast.walk(definition)
    ):
        raise InvalidConfig("active phase joins require a checked immutable grid")
    regions: list[list[ast.stmt]] = [[]]
    calls: list[ast.Call] = []
    semaphore = None
    for stmt in definition.body:
        call = checked_joins.get(stmt)
        if call is None:
            regions[-1].append(stmt)
            continue
        if not (
            isinstance(arg := call.args[0], ast.Attribute)
            and arg.attr == "iterator"
            and isinstance(arg.value, ast.Name)
            and (semaphore is None or arg.value.id == semaphore)
        ):
            raise InvalidConfig("active phase joins require the same local semaphore")
        semaphore = arg.value.id
        calls.append(call)
        regions.append([])
    if not calls or not any(
        isinstance(arg, TensorArg)
        and arg.name == semaphore
        and arg.fake_value.numel() >= COUNTER_STRIDE * len(calls)
        for arg in fn.arguments
    ):
        raise InvalidConfig(
            "active phase joins require a private counter for each edge"
        )

    def limit(n: int | None) -> str:
        return NUM_SM_VAR if n is None else f"min({NUM_SM_VAR}, {n})"

    counts = [participants(region) for region in regions]
    for i, call in enumerate(calls):
        replacement = expr_from_string(
            f"_cute_join_active({semaphore}.iterator + {i * COUNTER_STRIDE}, "
            f"{limit(counts[i])}, {limit(counts[i + 1])})"
        )
        assert isinstance(replacement, ast.Call)
        call.func = replacement.func
        call.args = replacement.args
