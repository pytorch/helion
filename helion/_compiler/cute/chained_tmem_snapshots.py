"""Bounded FP32-to-half snapshots of an original TMEM C view.

The caller proves original expression coordinates, immutable side inputs, MMA
readiness and publication lifetimes. This helper neither selects a graph edge
nor allocates/reuses TMEM. The expression is the ordinary fragment-bound
``_Expression`` result, including its original casts and operand-domain mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .chained_result_transport import load_operation
from .chained_tmem_segments import validate_tmem_segment
from .warp_specialized_plan import VALID_TMEM_COLUMNS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .chained_execution import ChainedExecution


@dataclass(frozen=True)
class PackedSnapshotPanels:
    """Column units are FP32 TMEM words, not bytes or half elements."""

    shape: tuple[int, int]
    source_offset: int
    destination_offset: int
    allocation_columns: int
    ordered: OrderedSnapshotAccess | None = None

    def __post_init__(self) -> None:
        if type(self.shape) is not tuple:
            raise ValueError("snapshot requires a full-M128 shape tuple")
        validate_tmem_segment(
            self.shape, 0, self.shape[1] if len(self.shape) == 2 else 0
        )
        if self.shape[1] < 64 or self.shape[1] % 32:
            raise ValueError("snapshot requires multiple complete N32 panels")
        if (
            any(
                type(value) is not int
                for value in (
                    self.source_offset,
                    self.destination_offset,
                    self.allocation_columns,
                )
            )
            or self.source_offset < 0
            or self.source_offset % 16
            or self.destination_offset < 0
            or self.destination_offset % 32
            or not 0 < self.allocation_columns <= VALID_TMEM_COLUMNS[-1]
            or self.source_offset + self.shape[1] > self.allocation_columns
            or self.destination_offset + self.shape[1] // 2 > self.allocation_columns
        ):
            raise ValueError("snapshot TMEM column bounds or alignment")
        if max(self.source_offset, self.destination_offset) < min(
            self.source_offset + self.shape[1],
            self.destination_offset + self.shape[1] // 2,
        ):
            if self.ordered is None:
                raise ValueError("snapshot source and destination must be disjoint")
        if self.ordered is not None and not self.ordered.matches(self):
            raise ValueError("snapshot ordered access proof changed")

    @property
    def count(self) -> int:
        return self.shape[1] // 32


@dataclass(frozen=True)
class OrderedSnapshotAccess:
    """Forward N32 reads followed by N16 word stores, on the same 128 lanes.

    For the original full-M128 Ld32x32b/Rep32 and packed St32x32b/Rep16
    partitions, participant r owns TMEM row r in both operations. Half pair
    (2j, 2j+1) becomes word j. A store may therefore overwrite an already-read
    word of that participant, but never a future panel's word. This is address
    authority only: the caller separately proves exact fragment coordinates,
    exclusive use, immutable side inputs and completion of the original MMA.
    """

    shape: tuple[int, int]
    source_offset: int
    destination_offset: int
    allocation_columns: int

    def matches(self, panels: PackedSnapshotPanels) -> bool:
        if (
            type(self.shape) is not tuple
            or any(
                type(v) is not int
                for v in (
                    *self.shape,
                    self.source_offset,
                    self.destination_offset,
                    self.allocation_columns,
                )
            )
            or self.shape != panels.shape
            or self.source_offset != panels.source_offset
            or self.destination_offset != panels.destination_offset
            or self.allocation_columns != panels.allocation_columns
        ):
            return False
        for panel in range(panels.count):
            start = self.destination_offset + panel * 16
            stop = start + 16
            for future in range(panel + 1, panels.count):
                read = self.source_offset + future * 32
                if max(start, read) < min(stop, read + 32):
                    return False
        return True


def emit_streamed_snapshot(
    panels: PackedSnapshotPanels,
    prefix: str,
    source_prefix: str,
    dtype: str,
    coords: tuple[str, str],
    expression_lines: Sequence[str],
    masked_value: str,
    *,
    execution: ChainedExecution,
    source_update: tuple[Sequence[str], str] | None = None,
) -> list[str]:
    """Render one uniform N32 stream, keeping original global coordinates.

    ``source_prefix_acc/slice`` must be the original full member view at
    ``execution.tmem + panels.source_offset``. Existing expression lines read
    ``source_prefix_values[prefix_index]`` and use ``coords``. No re-lowering,
    textual substitution, implicit cast, dense-layout replacement or rebasing
    of logical indices occurs here.

    All128 participants belong to a caller-established aligned hardware team.
    The first rendezvous moves before the stream: this requires no cross-thread
    expression exchange, immutable source/side inputs, and no visible partial
    destination. The final store fence and second rendezvous publish all panels.
    Wider role joins must remain outside the first128 predicate in the caller.

    An optional source update uses the same original FP32 values after packing.
    Its caller proves that no old-state reader remains, all side inputs are
    ready and immutable, and source/destination storage is disjoint. Its N32
    pre-store join is moved here from the original accumulator seed.
    """
    if execution.threads != 128:
        raise ValueError("snapshot requires exactly 128 local participants")
    if dtype not in ("cutlass.BFloat16", "cutlass.Float16"):
        raise ValueError("snapshot requires a BF16 or FP16 destination")
    # Also reject a record corrupted after construction, before any emission.
    panels.__post_init__()
    if source_update is not None and panels.ordered is not None:
        raise ValueError("snapshot source update requires disjoint storage")
    packed_shape = (128, panels.shape[1] // 2)

    def views(offset: str, suffix: str = "") -> list[str]:
        return [
            f"{prefix}_segment{suffix} = cute.composition(cute.domain_offset(((0, {offset}), 0, 0), {source_prefix}_acc), cute.make_layout(((128, 32), 1, 1)))",
            f"{prefix}_identity{suffix} = cute.composition(cute.domain_offset(((0, {offset}), 0, 0), {source_prefix}_identity), cute.make_layout(((128, 32), 1, 1)))",
            f"{prefix}_panel_target{suffix} = cute.composition(cute.domain_offset((0, ({offset}) // 2), {prefix}_target), cute.make_layout((128, 16)))",
            f"{prefix}_panel_coords{suffix} = cute.composition(cute.domain_offset((0, ({offset}) // 2), {prefix}_coords_tensor), cute.make_layout((128, 16)))",
        ]

    body = [
        *views(f"{prefix}_panel * 32"),
        f"{prefix}_source = {prefix}_reader.partition_S({prefix}_segment)",
        f"{source_prefix}_coords = {prefix}_reader.partition_D({prefix}_identity)",
        f"{prefix}_destination = {prefix}_writer.partition_D({prefix}_panel_target)",
        f"cute.copy({prefix}_load, {prefix}_source, {source_prefix}_values)",
        "cute.arch.fence_view_async_tmem_load()",
        f"for {prefix}_index in cutlass.range_constexpr(cute.size({source_prefix}_values)):",
        f"    {coords[0]}, {coords[1]} = {source_prefix}_coords[{prefix}_index]",
        *("    " + line.replace("\n", "\n    ") for line in expression_lines),
        f"    {prefix}_values[{prefix}_index] = {masked_value}",
        f"cute.copy({prefix}_store, {prefix}_packed, {prefix}_destination)",
    ]
    update_setup = []
    if source_update is not None:
        update_lines, update_value = source_update
        update_setup = [
            f"{prefix}_update_copy = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(32)), cutlass.Float32), {prefix}_segment0)",
            f"{prefix}_update_writer = {prefix}_update_copy.get_slice({execution.thread})",
        ]
        body.extend(
            [
                f"for {prefix}_index in cutlass.range_constexpr(cute.size({source_prefix}_values)):",
                f"    {coords[0]}, {coords[1]} = {source_prefix}_coords[{prefix}_index]",
                *("    " + line.replace("\n", "\n    ") for line in update_lines),
                f"    {source_prefix}_values[{prefix}_index] = cutlass.Float32({update_value})",
                execution.sync,
                f"{prefix}_update_target = {prefix}_update_writer.partition_D({prefix}_segment)",
                f"cute.copy({prefix}_update_copy, {source_prefix}_values, {prefix}_update_target)",
            ]
        )
    return [
        f"{source_prefix}_identity = {source_prefix}_slice.partition_C(cute.make_identity_tensor({panels.shape!r}))",
        f"{prefix}_layout = cute.composition({source_prefix}_acc.layout, cute.make_layout({packed_shape!r}))",
        f"{prefix}_coords_layout = cute.composition({source_prefix}_identity.layout, cute.make_layout({packed_shape!r}))",
        f"{prefix}_target = cute.make_tensor({execution.tmem} + {panels.destination_offset}, {prefix}_layout)",
        f"{prefix}_coords_tensor = cute.make_tensor({source_prefix}_identity.iterator, {prefix}_coords_layout)",
        # Static offset-zero tensors differ structurally from dynamic-offset
        # tensors. Never carry the setup containers through the DSL loop.
        *views("0", "0"),
        f"{prefix}_load = tcgen05.make_tmem_copy(cute.make_copy_atom({load_operation((128, 32))}, cutlass.Float32), {prefix}_segment0)",
        f"{prefix}_reader = {prefix}_load.get_slice({execution.thread})",
        f"{prefix}_load_coords0 = {prefix}_reader.partition_D({prefix}_identity0)",
        f"{source_prefix}_values = cute.make_rmem_tensor({prefix}_load_coords0.shape, cutlass.Float32)",
        f"{prefix}_store = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(16)), cutlass.Float32), {prefix}_panel_target0)",
        f"{prefix}_writer = {prefix}_store.get_slice({execution.thread})",
        f"{prefix}_store_coords0 = {prefix}_writer.partition_S({prefix}_panel_coords0)",
        f"{prefix}_packed = cute.make_rmem_tensor({prefix}_store_coords0.shape, cutlass.Float32)",
        f"{prefix}_values = cute.make_tensor(cute.recast_ptr({prefix}_packed.iterator, dtype={dtype}), {source_prefix}_values.layout)",
        *update_setup,
        execution.sync,
        f"for {prefix}_panel in cutlass.range({panels.count}, unroll=1):",
        *("    " + line.replace("\n", "\n    ") for line in body),
        "cute.arch.fence_view_async_tmem_store()",
        execution.sync,
    ]
