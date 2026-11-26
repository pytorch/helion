"""Recognize the scalar arithmetic and physical coordinates consumed by a paired column sum."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ..reduction_strategy import LoopedReductionStrategy
from ._ast_pass_utils import assigned_name
from ._ast_pass_utils import parse_expr
from ._ast_pass_utils import same_expr

if TYPE_CHECKING:
    from ..device_function import DeviceFunction
    from ..tile_strategy import DeviceGridState
    from .ordered_column_sum import _ColumnSum


def _rhs(node: ast.AST, text: str) -> bool:
    return isinstance(node, ast.Assign) and same_expr(node.value, parse_expr(text))


def _pointer(tensor: str, indices: list[str]) -> str:
    return (
        tensor
        + ".iterator"
        + "".join(
            f" + cutlass.Int32({idx}) * cutlass.Int32({tensor}.layout.stride[{i}])"
            for i, idx in enumerate(indices)
        )
    )


def ordered_column_region(
    body: list[ast.AST],
    grid: DeviceGridState,
    fn: DeviceFunction,
    site: _ColumnSum,
    x: str,
    y: str,
    tile: int,
) -> bool:
    if len(body) != 5:
        return False
    axis = grid.block_thread_axes.get(site.column)
    if axis is None:
        return False
    col = grid.strategy.index_var(site.column)
    mask = grid.strategy.mask_var(site.column)
    origin = grid.strategy.grid_origin_var(site.column)
    setups = {assigned_name(stmt): stmt for stmt in grid.lane_setup_statements}
    owners = [
        (var, count)
        for var, count in grid.lane_loops
        if grid.lane_loop_block_ids.get(var) == {site.column}
    ]
    if len(owners) != 1:
        return False
    lane, span = owners[0]
    hardware = f"cutlass.Int32(cute.arch.thread_idx()[{axis}])"
    if span <= 0 or tile % span or lane in grid.reversed_lane_vars or col not in setups:
        return False
    if not _rhs(setups[col], f"{origin} + {hardware} * {span} + cutlass.Int32({lane})"):
        return False
    # This is the complete ordinary column predicate. An explicit user
    # predicate, extra row/tile offset, or a domain ending early cannot be dropped.
    if mask:
        if mask not in setups or not (
            _rhs(setups[mask], f"{hardware} < {tile // span} and {col} < {site.width}")
            or _rhs(setups[mask], f"{col} < {site.width}")
        ):
            return False
    elif site.width % tile:
        return False
    leading = ["0"] * (site.destination.ndim - 1)

    def check_load(stmt: ast.AST, slot: str) -> str | None:
        read = f"({_pointer(x, [*leading, slot, col])}).load()"
        if mask:
            read += f" if {mask} else cutlass.BFloat16(0)"
        return (
            assigned_name(stmt)
            if isinstance(stmt, ast.Assign) and _rhs(stmt, read)
            else None
        )

    narrowed = body[-2]
    if not isinstance(narrowed, ast.Assign) or not (b := assigned_name(narrowed)):
        return False
    store = body[-1]
    if mask:
        if not (
            isinstance(store, ast.If)
            and same_expr(store.test, parse_expr(mask))
            and not store.orelse
            and len(store.body) == 1
        ):
            return False
        store = store.body[0]
    if not isinstance(store, ast.Expr) or not same_expr(
        store.value,
        parse_expr(f"({_pointer(y, [*leading, col])}).store(cutlass.BFloat16({b}))"),
    ):
        return False
    first, second, third = body[:3]
    if not (
        isinstance(third, ast.Assign)
        and (result := assigned_name(third))
        and _rhs(narrowed, f"cutlass.BFloat16({result})")
    ):
        return False
    # A synthetic per-thread owner is converted to the ordinary scalar
    # slot loop later. Tie its live coordinate and all arithmetic to the load.
    if isinstance(second, ast.Assign):
        owners = [
            v
            for v, n in grid.lane_loops
            if n == site.slots and grid.lane_loop_block_ids.get(v) == {site.reduction}
        ]
        if len(owners) != 1 or owners[0] in grid.reversed_lane_vars:
            return False
        slot_lane = owners[0]
        landed = [
            n
            for n, v in setups.items()
            if n and _rhs(v, f"cutlass.Int32(0) + cutlass.Int32({slot_lane}) * 1")
        ]
        if len(landed) != 1:
            return False
        read = check_load(first, landed[0])
        wide = assigned_name(second)
        return bool(
            read
            and wide
            and _rhs(second, f"cutlass.Float32({read})")
            and _rhs(
                third,
                f"cutlass.Float32(_helion_lane_reduce({wide}, 'sum', cutlass.Float32(0), threads_in_group=1, owner_lane='{slot_lane}'))",
            )
        )
    strategy = fn.tile_strategy.block_id_to_strategy.get((site.reduction,))
    if not isinstance(strategy, LoopedReductionStrategy) or strategy._thread_count != 1:
        return False
    if not (
        isinstance(first, ast.Assign)
        and (acc := assigned_name(first))
        and _rhs(first, "cutlass.Float32(0)")
        and isinstance(second, ast.For)
        and isinstance(second.target, ast.Name)
        and not second.orelse
        and len(second.body) == 1
        and isinstance(inner := second.body[0], ast.For)
        and isinstance(inner.target, ast.Name)
        and not inner.orelse
        and len(inner.body) == 4
    ):
        return False
    step = strategy._loop_block_size
    if (
        not isinstance(step, int)
        or step <= 0
        or site.slots % step
        or not same_expr(inner.iter, parse_expr(f"range({step})"))
    ):
        return False
    step_var = strategy.block_size_var(site.reduction)
    if not step_var or not same_expr(
        second.iter,
        parse_expr(
            f"range(cutlass.Int32(0), cutlass.Int32({site.slots}), cutlass.Int32({step_var}))"
        ),
    ):
        return False
    coord, load, widen, add = inner.body
    return bool(
        isinstance(coord, ast.Assign)
        and (slot := assigned_name(coord))
        and _rhs(
            coord,
            f"{second.target.id} + cutlass.Int32(0) + cutlass.Int32({inner.target.id}) * 1",
        )
        and (read := check_load(load, slot))
        and isinstance(widen, ast.Assign)
        and (wide := assigned_name(widen))
        and _rhs(widen, f"cutlass.Float32({read})")
        and isinstance(add, ast.Assign)
        and assigned_name(add) == acc
        and _rhs(add, f"{acc} + cutlass.Float32({wide})")
        and _rhs(
            third,
            f"cutlass.Float32(cute.arch.warp_reduction_sum({acc}, threads_in_group=1))",
        )
    )
