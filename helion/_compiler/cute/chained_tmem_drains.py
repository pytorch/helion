"""Bounded complete FP32 readback into already-owned shared result views.

Callers establish MMA completion, full source/destination lifetimes and the
128-participant team. No consumer may observe a partial destination. This
transport issues no allocation, arithmetic, narrowing, role join or publication
receipt; the original enclosing result action remains the authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .chained_result_transport import load_operation
from .chained_tmem_segments import validate_tmem_segment

if TYPE_CHECKING:
    from .chained_execution import ChainedExecution


@dataclass(frozen=True)
class FP32DrainPanels:
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        if type(self.shape) is not tuple:
            raise ValueError("FP32 drain requires an original physical shape tuple")
        validate_tmem_segment(
            self.shape, 0, self.shape[1] if len(self.shape) == 2 else 0
        )
        if self.shape[1] < 64 or self.shape[1] % 32:
            raise ValueError("FP32 drain requires multiple complete N32 panels")

    @property
    def count(self) -> int:
        return self.shape[1] // 32


def plan_fp32_drain(shape: tuple[int, int], columns: int) -> FP32DrainPanels | None:
    if type(columns) is not int or columns not in (0, 32):
        raise ValueError("FP32 drain tile columns must be 0 or 32")
    if not columns or shape[0] != 128 or not 64 <= shape[1] <= 256 or shape[1] % 32:
        return None
    return FP32DrainPanels(shape)


@dataclass
class DrainTiling:
    """Codegen activation only; original result actions own all readiness."""

    columns: int = 0
    activated: bool = False

    def __post_init__(self) -> None:
        self.check(self.columns)

    def check(self, selected: object) -> None:
        if (
            type(self.columns) is not int
            or type(selected) is not int
            or self.columns not in (0, 32)
            or self.columns != selected
        ):
            raise ValueError("FP32 drain selection changed")

    def emit(
        self,
        panels: FP32DrainPanels,
        prefix: str,
        source_prefix: str,
        publication: ScalarPublication | PartitionedPublication,
        *,
        execution: ChainedExecution,
    ) -> list[str]:
        self.check(32)
        lines = emit_streamed_fp32_drain(
            panels, prefix, source_prefix, publication, execution=execution
        )
        self.activated = True
        return lines

    def validate(self) -> None:
        from ... import exc

        self.check(self.columns)
        if self.columns and not self.activated:
            raise exc.BackendUnsupported(
                "cute", "FP32 drain tiling requires an emitted complete materialization"
            )


@dataclass(frozen=True)
class FP32SharedTarget:
    """One original shared view and its logical member coordinate map."""

    tensor: str
    shape: tuple[int, int]
    transpose: bool = False
    column_offset: int = 0
    physical_width: int = 0


@dataclass(frozen=True)
class ScalarPublication:
    index: str
    row: str
    column: str
    targets: tuple[FP32SharedTarget, ...]
    guarded: bool = False

    def coordinates(self, target: FP32SharedTarget) -> tuple[str, str, str]:
        column = (
            f"({self.column} - {target.column_offset})" if self.guarded else self.column
        )
        row, column = (column, self.row) if target.transpose else (self.row, column)
        predicate = f"({self.column} >= {target.column_offset}) & ({self.column} < {target.column_offset + target.physical_width}) & ({row} < {target.shape[0]}) & ({column} < {target.shape[1]})"
        return row, column, predicate

    def lines(self, values: str, coords: str) -> list[str]:
        """Original scalar publication, shared by whole and panel readbacks."""
        lines = [
            f"for {self.index} in cutlass.range_constexpr(cute.size({values})):",
            f"    {self.row}, {self.column} = {coords}[{self.index}]",
        ]
        for target in self.targets:
            row, column, predicate = self.coordinates(target)
            if self.guarded:
                lines.append(f"    if {predicate}:")
            lines.append(
                f"{'        ' if self.guarded else '    '}{target.tensor}[{row}, {column}] = {values}[{self.index}]"
            )
        return lines


@dataclass(frozen=True)
class PartitionedPublication:
    """Original full shared tensor; its native C partition is retained."""

    tensor: str


def emit_streamed_fp32_drain(
    panels: FP32DrainPanels,
    prefix: str,
    source_prefix: str,
    publication: ScalarPublication | PartitionedPublication,
    *,
    execution: ChainedExecution,
) -> list[str]:
    panels.__post_init__()
    if execution.threads != 128:
        raise ValueError("FP32 drain requires exactly 128 local participants")

    def view(name: str, source: str, offset: str) -> str:
        return f"{name} = cute.composition(cute.domain_offset(((0, {offset}), 0, 0), {source}), cute.make_layout(((128, 32), 1, 1)))"

    body = [
        f"{prefix}_column = {prefix}_panel * 32",
        view(f"{prefix}_acc", f"{source_prefix}_acc", f"{prefix}_column"),
        view(f"{prefix}_identity", f"{prefix}_full_identity", f"{prefix}_column"),
        f"{prefix}_source = {prefix}_thread.partition_S({prefix}_acc)",
        f"{prefix}_coords = {prefix}_thread.partition_D({prefix}_identity)",
        f"cute.copy({prefix}_copy, {prefix}_source, {prefix}_values)",
        "cute.arch.fence_view_async_tmem_load()",
    ]
    if isinstance(publication, ScalarPublication):
        body.extend(publication.lines(f"{prefix}_values", f"{prefix}_coords"))
    else:
        body.extend(
            [
                view(
                    f"{prefix}_shared_panel",
                    f"{prefix}_shared_full",
                    f"{prefix}_column",
                ),
                f"{prefix}_shared_target = {prefix}_thread.partition_D({prefix}_shared_panel)",
                f"cute.autovec_copy({prefix}_values, {prefix}_shared_target)",
            ]
        )
    return [
        f"{prefix}_full_identity = {source_prefix}_slice.partition_C(cute.make_identity_tensor({panels.shape!r}))",
        view(f"{prefix}_acc0", f"{source_prefix}_acc", "0"),
        view(f"{prefix}_identity0", f"{prefix}_full_identity", "0"),
        f"{prefix}_copy = tcgen05.make_tmem_copy(cute.make_copy_atom({load_operation((128, 32))}, cutlass.Float32), {prefix}_acc0)",
        f"{prefix}_thread = {prefix}_copy.get_slice({execution.thread})",
        f"{prefix}_coords0 = {prefix}_thread.partition_D({prefix}_identity0)",
        f"{prefix}_values = cute.make_rmem_tensor({prefix}_coords0.shape, cutlass.Float32)",
        *(
            [
                f"{prefix}_shared_full = {source_prefix}_slice.partition_C({publication.tensor})"
            ]
            if isinstance(publication, PartitionedPublication)
            else []
        ),
        f"for {prefix}_panel in cutlass.range({panels.count}, unroll=1):",
        *("    " + line for line in body),
    ]
