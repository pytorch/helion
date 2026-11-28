"""Unroll complete ordered sums of a scalar row when expansion is bounded."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...runtime.config import Config
from ._ast_pass_utils import parse_expr
from .autotune.config_keys import UNROLL_WARP_ROWS_KEY

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ...autotuner.config_spec import ConfigSpec
    from ..device_function import DeviceFunction

_WARP = 32


def bounded_row_sweeps(width: int, chunk: int, sweeps: int) -> bool:
    """Bound both constexpr loops and the total per-lane code expansion."""
    return (
        0 < chunk <= 16 * _WARP
        and chunk % _WARP == 0
        and width % chunk == 0
        and 1 <= width // chunk <= 32
        and 2 <= width // _WARP <= 128
        and 1 <= sweeps <= 256 // (width // _WARP)
    )


def unrolled_row_geometry(spec: ConfigSpec, opts: Mapping[str, object]) -> bool:
    sizes = opts.get("block_sizes")
    threads = opts.get("num_threads")
    chunks = opts.get("reduction_loops")
    if (
        opts.get("cute_warp_row_reductions") is not True
        or not isinstance(sizes, list)
        or not isinstance(threads, list)
        or not isinstance(chunks, list)
    ):
        return False
    return any(
        spec.block_sizes.config_get(sizes, row) == 1
        and spec.num_threads.config_get(threads, row) == 1
        and spec.num_threads.config_get(threads, axis) == 1
        and type(chunk := spec.reduction_loops.config_get(chunks, axis)) is int
        and bounded_row_sweeps(width, chunk, count)
        for row, axis, width, count in spec.cute_warp_row_reduction_roots.values()
    )


def add_unrolled_row_seed(spec: ConfigSpec, configs: list[Config]) -> list[Config]:
    """Keep the existing carrier and append one separate choice."""

    for parent in reversed(configs):
        if unrolled_row_geometry(spec, parent.config):
            return [
                *configs,
                Config.from_dict(parent.config | {UNROLL_WARP_ROWS_KEY: True}),
            ]
    return configs


@dataclass(frozen=True)
class WarpRowUnroll:
    outer: ast.For
    inner: ast.For
    width: int
    chunk: int
    outer_controls: tuple[str, str]
    inner_controls: tuple[str, str]

    @staticmethod
    def controls(loop: ast.For) -> tuple[str, str]:
        return ast.dump(loop.target), ast.dump(loop.iter)

    @classmethod
    def record(
        cls, outer: ast.For, inner: ast.For, width: int, chunk: int
    ) -> WarpRowUnroll:
        return cls(outer, inner, width, chunk, cls.controls(outer), cls.controls(inner))


def unroll_proved_sweeps(fn: DeviceFunction, body: Sequence[ast.stmt]) -> None:
    """Unroll the exact previously proved loop nodes after ordinary reload fusion."""
    pending = fn.cute_state.lanes.pending_warp_row_unrolls
    if not pending:
        return
    live = Counter(id(n) for st in body for n in ast.walk(st) if isinstance(n, ast.For))
    if any(
        live[id(plan.outer)] != 1
        or live[id(plan.inner)] != 1
        or plan.outer.body != [plan.inner]
        or plan.outer.orelse
        or plan.inner.orelse
        or plan.controls(plan.outer) != plan.outer_controls
        or plan.controls(plan.inner) != plan.inner_controls
        for plan in pending
    ):
        return
    for plan in pending:
        outer, inner, width, chunk = plan.outer, plan.inner, plan.width, plan.chunk
        assert isinstance(outer.target, ast.Name)
        offset = outer.target.id
        step = fn.new_var("warp_row_chunk", dce=False)
        outer.target = ast.Name(id=step, ctx=ast.Store())
        outer.iter = parse_expr(f"cutlass.range_constexpr({width // chunk})")
        outer.body.insert(
            0, ast.parse(f"{offset} = cutlass.Int32({step} * {chunk})").body[0]
        )
        inner.iter = parse_expr(f"cutlass.range_constexpr({chunk // _WARP})")
    fn.cute_state.lanes.unrolled_warp_row = True
