"""Checked tile ownership within an existing prepared, complete-team interval."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING
from typing import Protocol

import torch
from torch.fx import Node

from ...language.view_ops import subscript
from . import chained_matmul as chain
from .warp_specialized_primitives import swizzle_b16_index

if TYPE_CHECKING:
    from .chained_execution import ChainedExecution
    from .contraction_region import ContractionSpec
    from .prepared_graph_schedule import ContractionGraph
    from .prepared_graph_schedule import ContractionPartition
    from .warp_specialized_plan import WarpRole


class PreparedWarpOwner(Protocol):
    @property
    def graph(self) -> ContractionGraph: ...

    def check(self) -> None: ...


@dataclass(frozen=True)
class RuntimeTileAxis:
    divisor: int
    modulus: int | None
    scale: int

    def value(self, warp: int) -> int:
        if min(self.divisor, self.scale) < 1 or (
            self.modulus is not None and self.modulus < 1
        ):
            raise chain._UnsupportedChain("invalid runtime tile coordinate")
        value = warp // self.divisor
        if self.modulus is not None:
            value %= self.modulus
        return value * self.scale

    def payload(self) -> tuple[int, int | None, int]:
        self.value(0)
        return self.divisor, self.modulus, self.scale


@dataclass(frozen=True)
class WarpTile:
    warp: int
    ordinal: int
    row: int
    column: int
    axes: tuple[int, int]
    k_interval: tuple[int, int]
    omit_contraction: bool


@dataclass(frozen=True)
class SharedTileView:
    """One actual byte-addressed view in a caller-owned cyclic frame."""

    source: Node
    offset: int
    shape: tuple[int, int]
    origin: tuple[int, int] = (0, 0)
    row_bytes: int = 128
    columns_per_tile: int = 64
    rows_per_tile: int = 32
    phase_mask: int = 7
    transposed: bool = False

    def address(self, row: int, column: int) -> int:
        if self.transposed:
            row, column = column, row
        return self.offset + swizzle_b16_index(
            row + self.origin[0],
            column + self.origin[1],
            self.row_bytes,
            self.columns_per_tile,
            self.rows_per_tile,
            self.phase_mask,
        )

    def bytes(self) -> frozenset[int]:
        return frozenset(
            self.address(row, column) + byte
            for row in range(self.shape[0])
            for column in range(self.shape[1])
            for byte in (0, 1)
        )

    def payload(self) -> tuple[int, ...]:
        if not self.transposed or self.origin[0] != 0:
            raise chain._UnsupportedChain("unsupported native result store image")
        return (
            self.offset,
            self.origin[1],
            self.row_bytes,
            self.columns_per_tile,
            self.rows_per_tile,
            self.phase_mask,
        )


@dataclass(frozen=True)
class SharedLease:
    """Half-open byte and phase intervals of actual caller-owned shared data."""

    source: Node
    begin: int
    end: int
    first_phase: int
    last_phase: int


@dataclass(frozen=True)
class WarpTileInterval:
    owner: PreparedWarpOwner
    role: WarpRole
    execution: ChainedExecution
    teams: int
    named_barrier_base: int
    ready_phase: int
    join_phase: int
    frame: tuple[int, int]
    inputs: tuple[SharedTileView, SharedTileView]
    leases: tuple[SharedLease, ...]

    def check(self) -> None:
        self.owner.check()
        if (
            self.execution.threads % 32
            or self.teams * self.execution.threads != self.role.warp_count * 32
            or self.ready_phase != 0
            or self.join_phase != 1
            or self.named_barrier_base < 1
            or self.named_barrier_base + self.teams > 16
            or self.frame[0] >= self.frame[1]
        ):
            raise chain._UnsupportedChain("invalid prepared warp interval or team join")
        for view in self.inputs:
            if not view.bytes() <= frozenset(range(*self.frame)):
                raise chain._UnsupportedChain("prepared operand outside caller frame")
        for lease in self.leases:
            if (
                lease.begin < self.frame[0]
                or lease.end > self.frame[1]
                or lease.begin >= lease.end
                or lease.first_phase >= lease.last_phase
            ):
                raise chain._UnsupportedChain("invalid caller shared lease")

    def payload(self) -> tuple[int, int]:
        self.check()
        return self.named_barrier_base, self.execution.threads


@dataclass(frozen=True)
class CausalWarpPublication:
    """Original BF16 where(row >= column, original contraction, +0) cut."""

    spec: ContractionSpec
    publication: Node
    row_coordinate: Node
    column_coordinate: Node
    shape: tuple[int, int]

    def check(self) -> None:
        node = self.publication
        if (
            node.target is not torch.ops.prims.convert_element_type.default
            or len(node.args) != 2
            or node.args[1] is not torch.bfloat16
            or node.kwargs
            or not isinstance(node.args[0], Node)
        ):
            raise chain._UnsupportedChain("changed typed warp publication")
        where = node.args[0]
        if (
            where.target is not torch.ops.aten.where.self
            or len(where.args) != 3
            or where.kwargs
            or where.args[1] is not self.spec.node
            or not isinstance(where.args[0], Node)
            or not isinstance(where.args[2], Node)
        ):
            raise chain._UnsupportedChain("changed original warp mask value")
        mask = where.args[0]
        zero = where.args[2]
        if (
            mask.target is not torch.ops.aten.ge.Tensor
            or len(mask.args) != 2
            or mask.kwargs
            or zero.target is not torch.ops.aten.scalar_tensor.default
            or zero.args != (0.0,)
            or not isinstance(zero.args[0], (int, float))
            or math.copysign(1.0, zero.args[0]) != 1.0
            or zero.kwargs.get("dtype") is not torch.float32
        ):
            raise chain._UnsupportedChain("changed causal mask or positive-zero arm")
        for value, coordinate, indices, extent in zip(
            mask.args,
            (self.row_coordinate, self.column_coordinate),
            ([slice(None), None], [None, slice(None)]),
            self.shape,
            strict=True,
        ):
            if (
                not isinstance(value, Node)
                or value.target is not subscript
                or value.args != (coordinate, indices)
                or value.kwargs
                or coordinate.target is not torch.ops.prims.iota.default
                or coordinate.args != (extent,)
                or coordinate.kwargs.get("start") != 0
                or coordinate.kwargs.get("step") != 1
            ):
                raise chain._UnsupportedChain(
                    "changed actual row/column mask coordinates"
                )


class WarpTileBinding(Protocol):
    """Backend authority connecting a schedule to actual callers and consumers."""

    def check(self, schedule: WarpTileSchedule) -> None: ...


@dataclass(frozen=True)
class WarpTileSchedule:
    """Physical work over original nodes, not a replacement contraction graph."""

    partition: ContractionPartition
    publication: CausalWarpPublication
    interval: WarpTileInterval
    fixed: tuple[ContractionSpec, ...]
    result: SharedTileView
    axes: tuple[RuntimeTileAxis, ...]
    tiles: tuple[WarpTile, ...]
    operand_port: tuple[int, int, int, int, int]
    binding: WarpTileBinding

    def check(self) -> None:
        self.partition.check()
        self.interval.check()
        self.publication.check()
        spec = self.publication.spec
        graph = self.partition.graph
        if self.interval.owner.graph is not graph:
            raise chain._UnsupportedChain("foreign prepared interval authority")
        if self.publication.publication not in graph.region.nodes:
            raise chain._UnsupportedChain("foreign original typed publication")
        if (
            self.partition.specs != (spec,)
            or self.result.source is not self.publication.publication
        ):
            raise chain._UnsupportedChain("foreign tile/publication partition")
        if (
            spec.accumulator is not None
            or spec.operand_dtypes != (torch.bfloat16, torch.bfloat16)
            or spec.result_dtype is not torch.float32
            or self.result.shape != self.publication.shape
        ):
            raise chain._UnsupportedChain("unsupported warp tile seed, type, or extent")
        m, n = self.publication.shape
        lhs_shape, rhs_shape = (
            chain._host_shape(spec.lhs.meta["val"]),
            chain._host_shape(spec.rhs.meta["val"]),
        )
        if lhs_shape[0] != m or rhs_shape[1] != n or lhs_shape[1] != rhs_shape[0]:
            raise chain._UnsupportedChain("changed warp contraction geometry")
        k = lhs_shape[1]
        a, b = self.interval.inputs
        if a.source is not spec.lhs or b.source is not spec.rhs:
            raise chain._UnsupportedChain("foreign prepared operands")
        if a.shape != lhs_shape or b.shape != (rhs_shape[1], rhs_shape[0]):
            raise chain._UnsupportedChain("changed original prepared operand image")
        # The ordered-K port consumes offsets, not arbitrary SharedTileView
        # layouts. Admit exactly the native leaf's actual prepared input image.
        if any(
            (
                view.origin,
                view.row_bytes,
                view.columns_per_tile,
                view.rows_per_tile,
                view.phase_mask,
                view.transposed,
            )
            != ((0, 0), 128, 64, 32, 7, False)
            for view in (a, b)
        ):
            raise chain._UnsupportedChain("unsupported ordered-K leaf input layout")
        if self.operand_port != (a.offset, b.offset, k // 16, 2, m) or k % 16:
            raise chain._UnsupportedChain("changed prepared operand/K representation")
        originals = {id(item) for item in graph.region.contractions}
        if len({id(item) for item in self.fixed}) != len(self.fixed) or any(
            id(item) not in originals for item in self.fixed
        ):
            raise chain._UnsupportedChain("foreign fixed interval contraction")
        ancestors = chain._ancestors(self.publication.publication)
        if any(
            item.node in ancestors
            or spec.node in chain._ancestors(item.node)
            or self.publication.publication in chain._ancestors(item.node)
            for item in self.fixed
        ):
            raise chain._UnsupportedChain("cross-warp interval contraction dependency")
        warps = self.interval.execution.threads // 32
        covered: set[tuple[int, int]] = set()
        physical: set[int] = set()
        ordinals = [[] for _ in range(warps)]
        for tile in self.tiles:
            if (
                not 0 <= tile.warp < warps
                or min(tile.axes) < 0
                or max(tile.axes) >= len(self.axes)
            ):
                raise chain._UnsupportedChain("out-of-team tile owner or runtime axis")
            if (
                self.axes[tile.axes[0]].value(tile.warp),
                self.axes[tile.axes[1]].value(tile.warp),
            ) != (tile.row, tile.column):
                raise chain._UnsupportedChain(
                    "runtime coordinate differs from logical tile"
                )
            if tile.k_interval != (0, k):
                raise chain._UnsupportedChain("changed original increasing K interval")
            cells = {
                (
                    tile.row + lane // 4 + ((slot % 4) // 2) * 8,
                    tile.column + (slot // 4) * 8 + lane % 4 * 2 + slot % 2,
                )
                for lane in range(32)
                for slot in range(8)
            }
            if (
                len(cells) != 256
                or covered & cells
                or any(not (0 <= r < m and 0 <= c < n) for r, c in cells)
            ):
                raise chain._UnsupportedChain(
                    "duplicate or out-of-domain warp publication"
                )
            if tile.omit_contraction != all(r < c for r, c in cells):
                raise chain._UnsupportedChain("unproved omitted contraction domain")
            tile_bytes = {
                self.result.address(r, c) + byte for r, c in cells for byte in (0, 1)
            }
            if len(tile_bytes) != 512 or physical & tile_bytes:
                raise chain._UnsupportedChain("duplicate physical halfword publication")
            covered.update(cells)
            physical.update(tile_bytes)
            ordinals[tile.warp].append(tile.ordinal)
        if (
            covered != {(r, c) for r in range(m) for c in range(n)}
            or physical != self.result.bytes()
        ):
            raise chain._UnsupportedChain(
                "incomplete tile or physical publication coverage"
            )
        if any(sorted(items) != list(range(len(items))) for items in ordinals):
            raise chain._UnsupportedChain("duplicate or missing serial tile ordinal")
        if not physical <= frozenset(range(*self.interval.frame)):
            raise chain._UnsupportedChain("result outside caller frame")
        for view in self.interval.inputs:
            if physical & view.bytes():
                raise chain._UnsupportedChain("result aliases live prepared operand")
        for lease in self.interval.leases:
            if (
                lease.first_phase < self.interval.join_phase
                and self.interval.ready_phase < lease.last_phase
                and physical & set(range(lease.begin, lease.end))
            ):
                raise chain._UnsupportedChain("result aliases live caller shared lease")
        self.result.payload()
        self.binding.check(self)
        graph.check()

    def payload(self) -> tuple[object, ...]:
        self.check()
        counts = tuple(
            sum(tile.warp == warp for tile in self.tiles)
            for warp in range(self.interval.execution.threads // 32)
        )
        default = Counter(tile.axes for tile in self.tiles).most_common(1)[0][0]
        overrides = tuple(
            (
                warp,
                tuple(
                    (tile.ordinal, *tile.axes)
                    for tile in self.tiles
                    if tile.warp == warp and tile.axes != default
                ),
            )
            for warp in range(len(counts))
            if any(tile.warp == warp and tile.axes != default for tile in self.tiles)
        )
        return (
            counts,
            default,
            overrides,
            self.operand_port,
            self.result.payload(),
            self.interval.payload(),
            tuple(axis.payload() for axis in self.axes),
        )
