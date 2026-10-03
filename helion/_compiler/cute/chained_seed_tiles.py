"""Disjoint panels of one full-M128 FP32 accumulator member.

This is geometry only: the caller retains its exact seed expression, masks,
128-thread ownership, ready/read ordering and TMEM load/store fences. Splitting
a member grants no arena reuse or permission to overwrite resident neighbors.
"""

from __future__ import annotations

from dataclasses import dataclass

from ... import exc
from .chained_tmem_segments import emit_tmem_segment_views
from .chained_tmem_segments import validate_tmem_segment


def _validate_max_columns(max_columns: int) -> None:
    if type(max_columns) is not int or max_columns not in (0, 32, 64):
        raise ValueError("seed panel maximum columns must be 0, 32 or 64")


@dataclass(frozen=True)
class SeedPanel:
    full_shape: tuple[int, int]
    member_offset: int
    member_width: int
    offset: int
    width: int

    def views(self, prefix: str, source_prefix: str) -> list[str]:
        """FP32 physical and identity views with original grouped coordinates.

        Expressions must subtract ``member_offset``, not this panel's offset,
        when mapping a global column back to the original logical member.
        """
        return emit_tmem_segment_views(
            prefix, source_prefix, self.full_shape, self.offset, self.width
        )

    def store_views(self, prefix: str, source_prefix: str, thread: str) -> list[str]:
        """Matching FP32 target/coordinate partitions; no allocation or fence."""
        repetition = min(32, self.width & -self.width)
        return [
            *self.views(prefix, source_prefix),
            f"{prefix}_copy = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition({repetition})), cutlass.Float32), {prefix}_segment)",
            f"{prefix}_thread = {prefix}_copy.get_slice({thread})",
            f"{prefix}_target = {prefix}_thread.partition_D({prefix}_segment)",
            f"{prefix}_coords = {prefix}_thread.partition_S({prefix}_identity)",
        ]


def plan_seed_panels(
    full_shape: tuple[int, int],
    member_offset: int,
    member_width: int,
    max_columns: int = 0,
) -> tuple[SeedPanel, ...]:
    """Cover one member exactly once, retaining its original column origin.

    Zero means the original whole-member view, even for non-power-of-two
    widths. A positive bound partitions in source order with an exact, possibly
    smaller final panel. All widths/offsets use physical FP32 TMEM columns.
    """
    _validate_max_columns(max_columns)
    if type(full_shape) is not tuple:
        raise ValueError("seed panels require a full-M128 physical shape tuple")
    validate_tmem_segment(full_shape, member_offset, member_width)
    step = max_columns or member_width
    return tuple(
        SeedPanel(
            full_shape,
            member_offset,
            member_width,
            offset,
            min(step, member_offset + member_width - offset),
        )
        for offset in range(member_offset, member_offset + member_width, step)
    )


@dataclass
class SeedTiling:
    """Per-codegen activation; only an emitted split seed is effective."""

    max_columns: int
    activated: bool = False

    def __post_init__(self) -> None:
        _validate_max_columns(self.max_columns)

    def panels(
        self, full_shape: tuple[int, int], member_offset: int, member_width: int
    ) -> tuple[SeedPanel, ...]:
        panels = plan_seed_panels(
            full_shape, member_offset, member_width, self.max_columns
        )
        self.activated |= len(panels) > 1
        return panels

    def validate(self) -> None:
        if self.max_columns and not self.activated:
            raise exc.BackendUnsupported(
                "cute", "seed tiling requires an admitted multi-panel accumulator seed"
            )
