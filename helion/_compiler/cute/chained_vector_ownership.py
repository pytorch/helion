"""Logical eight-value ownership, independent of expression and copy legality.

This planner neither selects an optimization nor proves that a destination's
native layout supports a tiled copy. Callers retain their original masks,
active participant guards, publication barriers and activation policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class VectorOwnership:
    """A complete rectangular iteration space with eight values per thread.

    ``threads`` describes already selected producers, not the entire CTA or
    synchronization role. In particular, a physical-copy caller may need a
    power-of-two prefix even when scalar ownership covers a larger role.
    Expressions take caller-bound thread/step names. The zero/default planner
    option retains the old source text, including a one-trip row loop.
    """

    shape: tuple[int, int]
    threads: int
    tile_columns: int
    thread_rows: int
    thread_columns: int
    row_tiles: int
    column_tiles: int
    trips: int
    thread_order: Literal["row_major", "column_major"] = "row_major"

    def matches(self, shape: tuple[int, int], threads: int) -> bool:
        """Reject stale or inconsistent dependent fields at emitter boundaries."""
        return self == plan_vector_ownership(
            shape,
            threads,
            tile_columns=self.tile_columns,
            thread_order=self.thread_order,
        )

    @property
    def changed(self) -> bool:
        return self.tile_columns != self.shape[1] or self.thread_order != "row_major"

    @property
    def thread_strides(self) -> tuple[int, int]:
        """Physical thread numbering for the same rectangular logical tile."""
        if self.thread_order == "column_major":
            return (1, self.thread_rows)
        return (self.thread_columns, 1)

    def row_expression(self, thread: str, step: str) -> str:
        if self.thread_order == "column_major":
            row = f"{thread} % {self.thread_rows}"
            if self.row_tiles == 1:
                return row
            return f"{row} + ({step} // {self.column_tiles}) * {self.thread_rows}"
        if not self.changed:
            return f"{thread} // {self.thread_columns} + {step} * {self.thread_rows}"
        if self.row_tiles == 1:
            return f"{thread} // {self.thread_columns}"
        return (
            f"{thread} // {self.thread_columns} + "
            f"({step} // {self.column_tiles}) * {self.thread_rows}"
        )

    def base_expression(self, thread: str, step: str) -> str:
        if self.thread_order == "column_major":
            column = step if self.row_tiles == 1 else f"({step} % {self.column_tiles})"
            return f"({thread} // {self.thread_rows} + {column} * {self.thread_columns}) * 8"
        base = f"{thread} % {self.thread_columns} * 8"
        if not self.changed:
            return base
        column = step if self.row_tiles == 1 else f"({step} % {self.column_tiles})"
        return f"{base} + {column} * {self.tile_columns}"

    def copy_indices(self, step: str) -> str:
        if not self.changed:
            return f"None, {step}, 0"
        if self.row_tiles == 1:
            return f"None, 0, {step}"
        return f"None, {step} // {self.column_tiles}, {step} % {self.column_tiles}"


def plan_vector_ownership(
    shape: tuple[int, int],
    threads: int,
    *,
    tile_columns: int = 0,
    thread_order: Literal["row_major", "column_major"] = "row_major",
) -> VectorOwnership | None:
    """Plan full eight-value vectors; reject partial physical column tiles.

    A ragged final row tile requires the caller's original ``row < height``
    guard. Logical node padding inside the physical rectangle is separately
    owned by its original expression/mask emitter. No storage padding is added.
    Non-power-of-two producer teams are valid for logical ownership when the
    column thread count divides the team; native-copy constraints remain with
    the caller. No hardware launch, lifetime or synchronization proof is made.
    """
    if (
        type(shape) is not tuple
        or len(shape) != 2
        or any(type(extent) is not int or extent <= 0 for extent in shape)
        or type(threads) is not int
        or threads <= 0
        or type(tile_columns) is not int
        or tile_columns < 0
        or type(thread_order) is not str
        or thread_order not in ("row_major", "column_major")
    ):
        return None
    height, width = shape
    tile_columns = tile_columns or width
    if width % 8 or tile_columns % 8 or width % tile_columns:
        return None
    columns = tile_columns // 8
    if columns & (columns - 1) or threads % columns:
        return None
    rows = threads // columns
    row_tiles = (height + rows - 1) // rows
    column_tiles = width // tile_columns
    return VectorOwnership(
        shape=shape,
        threads=threads,
        tile_columns=tile_columns,
        thread_rows=rows,
        thread_columns=columns,
        row_tiles=row_tiles,
        column_tiles=column_tiles,
        trips=row_tiles * column_tiles,
        thread_order=thread_order,
    )
