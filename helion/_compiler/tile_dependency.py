from __future__ import annotations

import dataclasses
import enum
from functools import cached_property
import itertools
import math
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

import sympy

if TYPE_CHECKING:
    import ast

    from .device_ir import DeviceIR


TILE_DEPENDENCY_SCOPE_IDS_META = "_tile_dependency_scope_ids"
TILE_DEPENDENCY_SCOPE_ID_ATTR = "_tile_dependency_scope_id"
_ALLOCATION_ADDRESS_AXIS = -1
# A memory hazard at one concrete producer/consumer callsite pairing.
DependencyPoint = tuple[int, int | None, int | None]


class TileDependencyKind(enum.Enum):
    """The memory hazard represented by a cross-loop dependency edge."""

    READ_AFTER_WRITE = "read_after_write"
    WRITE_AFTER_READ = "write_after_read"
    WRITE_AFTER_WRITE = "write_after_write"


def tile_dependency_scope_id(node: ast.AST) -> int | None:
    """Return the stable DeviceIR execution scope attached to a lowered loop."""
    scope_id = getattr(node, TILE_DEPENDENCY_SCOPE_ID_ATTR, None)
    return scope_id if isinstance(scope_id, int) else None


def execution_scopes_by_graph_id(
    device_ir: DeviceIR,
    *,
    execution_scopes: tuple[ExecutionScope, ...] | None = None,
) -> tuple[tuple[ExecutionScope, ...], ...]:
    """Resolve every DeviceIR graph to its static execution-scope occurrences."""
    if execution_scopes is None:
        execution_scopes = build_execution_scopes(device_ir)
    _validate_execution_scopes(device_ir, execution_scopes)
    scopes_by_graph: list[list[ExecutionScope]] = [[] for _ in device_ir.graphs]
    for scope in execution_scopes:
        scopes_by_graph[scope.graph_id].append(scope)
    return tuple(tuple(scopes) for scopes in scopes_by_graph)


@dataclasses.dataclass(frozen=True)
class LogicalTaskAxis:
    """One source-level axis in a root's logical task space.

    ``extent`` comes directly from the block-size registration performed while
    tracing ``hl.tile``.  It is independent of the later physical PID order or
    an L2 traversal chosen by a concrete configuration.
    """

    block_id: int
    extent: sympy.Expr | str | None
    canonical_origin: bool = True


@dataclasses.dataclass(frozen=True)
class TaskFamily:
    """One opaque top-level loop and its authoritative logical task domain."""

    axes: tuple[LogicalTaskAxis, ...]

    @property
    def logical_axis_order(self) -> tuple[int, ...]:
        return tuple(axis.block_id for axis in self.axes)

    def axis(self, block_id: int) -> LogicalTaskAxis | None:
        return next((axis for axis in self.axes if axis.block_id == block_id), None)


@dataclasses.dataclass(frozen=True)
class LogicalDomain:
    """One configured Cartesian domain in canonical logical coordinates.

    Axis identity and geometry belong here.  Linear traversal is deliberately
    supplied to :meth:`coordinates` and :meth:`index` by the caller so event
    identity, action order, and physical PID traversal cannot accidentally
    become the same policy.
    """

    axis_order: tuple[int, ...]
    axis_counts_items: tuple[tuple[int, int], ...]
    block_sizes_items: tuple[tuple[int, int], ...] = ()
    kind: Literal["scope", "allocation", "event", "worker", "value"] = "scope"
    identity: int | None = None

    def __post_init__(self) -> None:
        if len(set(self.axis_order)) != len(self.axis_order):
            raise ValueError("logical domain axes must be unique")
        if tuple(axis for axis, _count in self.axis_counts_items) != self.axis_order:
            raise ValueError("logical domain counts must follow axis order")
        if (
            self.block_sizes_items
            and tuple(axis for axis, _size in self.block_sizes_items) != self.axis_order
        ):
            raise ValueError("logical domain block sizes must follow axis order")
        if any(count <= 0 for _axis, count in self.axis_counts_items):
            raise ValueError("logical domain axis counts must be positive")
        if any(size <= 0 for _axis, size in self.block_sizes_items):
            raise ValueError("logical domain block sizes must be positive")

    @property
    def axis_counts(self) -> dict[int, int]:
        return dict(self.axis_counts_items)

    @property
    def block_sizes(self) -> dict[int, int]:
        return dict(self.block_sizes_items)

    @property
    def size(self) -> int:
        return math.prod(count for _axis, count in self.axis_counts_items)

    def _validate_traversal(self, traversal: tuple[int, ...]) -> None:
        if len(traversal) != len(self.axis_order) or set(traversal) != set(
            self.axis_order
        ):
            raise ValueError("logical traversal must permute the domain axes")

    def coordinates(
        self,
        index: int,
        *,
        traversal: tuple[int, ...] | None = None,
    ) -> dict[int, int]:
        """Decode an integer using the requested fastest-to-slowest axes."""
        if not 0 <= index < self.size:
            raise IndexError(index)
        traversal = self.axis_order if traversal is None else traversal
        self._validate_traversal(traversal)
        counts = self.axis_counts
        coordinates: dict[int, int] = {}
        remainder = index
        for axis in traversal:
            count = counts[axis]
            coordinates[axis] = remainder % count
            remainder //= count
        if remainder:
            raise AssertionError("index exceeds its logical coordinate domain")
        return coordinates

    def index(
        self,
        coordinates: dict[int, int],
        *,
        traversal: tuple[int, ...] | None = None,
    ) -> int:
        """Encode coordinates using the requested fastest-to-slowest axes."""
        traversal = self.axis_order if traversal is None else traversal
        self._validate_traversal(traversal)
        counts = self.axis_counts
        result = 0
        multiplier = 1
        for axis in traversal:
            coordinate = coordinates[axis]
            count = counts[axis]
            if not 0 <= coordinate < count:
                raise IndexError(coordinate)
            result += coordinate * multiplier
            multiplier *= count
        return result


def logical_axis_symbol(axis: int) -> sympy.Symbol:
    """Return the canonical integer symbol for one logical-domain axis."""
    suffix = str(axis) if axis >= 0 else f"m{-axis}"
    return sympy.Symbol(f"logical_axis_{suffix}", integer=True, nonnegative=True)


@dataclasses.dataclass(frozen=True)
class _LogicalRelationPiece:
    """One guarded source box mapped to a Cartesian target range."""

    source_bounds_items: tuple[tuple[int, int, int, int], ...]
    target_ranges: tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...]

    def contains(self, coordinates: dict[int, int]) -> bool:
        return all(
            begin <= coordinates[axis] < end and (coordinates[axis] - begin) % step == 0
            for axis, begin, end, step in self.source_bounds_items
        )


@dataclasses.dataclass(frozen=True)
class LogicalRelation:
    """Restricted symbolic relation between two Cartesian integer domains.

    Each piece maps one guarded source box to a Cartesian product of target
    ranges.  Expressions may use source-axis symbols, static affine arithmetic,
    floor division, modulo, and min/max.  Operations that cannot stay in this
    deliberately small grammar must decline rather than enumerate runtime
    instances.
    """

    source_domain: LogicalDomain
    target_domain: LogicalDomain
    pieces: tuple[_LogicalRelationPiece, ...]

    def __post_init__(self) -> None:
        for piece in self.pieces:
            if (
                tuple(axis for axis, _begin, _end, _step in piece.source_bounds_items)
                != self.source_domain.axis_order
            ):
                raise ValueError("relation source bounds must follow domain order")
            if (
                tuple(axis for axis, _begin, _end, _step in piece.target_ranges)
                != self.target_domain.axis_order
            ):
                raise ValueError("relation target ranges must follow domain order")
            if any(
                step <= 0 for _axis, _begin, _end, step in piece.source_bounds_items
            ):
                raise ValueError("relation source strides must be positive")
            if any(step <= 0 for _axis, _begin, _end, step in piece.target_ranges):
                raise ValueError("relation target strides must be positive")

    @classmethod
    def identity(
        cls,
        source_domain: LogicalDomain,
        target_domain: LogicalDomain,
    ) -> LogicalRelation:
        """Return the pointwise identity between equivalent coordinate spaces."""
        if (
            source_domain.axis_order != target_domain.axis_order
            or source_domain.axis_counts_items != target_domain.axis_counts_items
        ):
            raise ValueError("identity relation requires equal coordinate geometry")
        return cls(
            source_domain=source_domain,
            target_domain=target_domain,
            pieces=(
                _LogicalRelationPiece(
                    source_bounds_items=tuple(
                        (axis, 0, source_domain.axis_counts[axis], 1)
                        for axis in source_domain.axis_order
                    ),
                    target_ranges=tuple(
                        (
                            axis,
                            logical_axis_symbol(axis),
                            logical_axis_symbol(axis) + 1,  # pyrefly: ignore[unsupported-operation]
                            1,
                        )
                        for axis in target_domain.axis_order
                    ),
                ),
            ),
        )

    @classmethod
    def point_map(
        cls,
        source_domain: LogicalDomain,
        target_domain: LogicalDomain,
        pieces: tuple[
            tuple[
                tuple[tuple[int, int, int, int], ...],
                tuple[sympy.Expr, ...],
            ],
            ...,
        ],
    ) -> LogicalRelation:
        """Build a piecewise single-valued relation in domain axis order."""
        return cls(
            source_domain=source_domain,
            target_domain=target_domain,
            pieces=tuple(
                _LogicalRelationPiece(
                    source_bounds_items=source_bounds,
                    target_ranges=tuple(
                        (
                            axis,
                            expression,
                            expression + 1,  # pyrefly: ignore[unsupported-operation]
                            1,
                        )
                        for axis, expression in zip(
                            target_domain.axis_order,
                            target_expressions,
                            strict=True,
                        )
                    ),
                )
                for source_bounds, target_expressions in pieces
            ),
        )

    @classmethod
    def total(
        cls,
        source_domain: LogicalDomain,
        target_domain: LogicalDomain,
    ) -> LogicalRelation:
        """Return the complete relation between two bounded domains."""
        return cls(
            source_domain=source_domain,
            target_domain=target_domain,
            pieces=(
                _LogicalRelationPiece(
                    source_bounds_items=tuple(
                        (axis, 0, source_domain.axis_counts[axis], 1)
                        for axis in source_domain.axis_order
                    ),
                    target_ranges=tuple(
                        (
                            axis,
                            sympy.Integer(0),
                            sympy.Integer(target_domain.axis_counts[axis]),
                            1,
                        )
                        for axis in target_domain.axis_order
                    ),
                ),
            ),
        )

    @classmethod
    def projection(
        cls,
        source_domain: LogicalDomain,
        target_domain: LogicalDomain,
    ) -> LogicalRelation | None:
        """Project a domain onto a coordinate-compatible subdomain."""
        source_counts = source_domain.axis_counts
        if any(
            axis not in source_counts or source_counts[axis] != count
            for axis, count in target_domain.axis_counts_items
        ):
            return None
        return cls(
            source_domain=source_domain,
            target_domain=target_domain,
            pieces=(
                _LogicalRelationPiece(
                    source_bounds_items=tuple(
                        (axis, 0, source_counts[axis], 1)
                        for axis in source_domain.axis_order
                    ),
                    target_ranges=tuple(
                        (
                            axis,
                            logical_axis_symbol(axis),
                            logical_axis_symbol(axis) + 1,  # pyrefly: ignore[unsupported-operation]
                            1,
                        )
                        for axis in target_domain.axis_order
                    ),
                ),
            ),
        )

    def retarget(self, target_domain: LogicalDomain) -> LogicalRelation | None:
        """Retype the codomain without changing its coordinate geometry."""
        if (
            self.target_domain.axis_order != target_domain.axis_order
            or self.target_domain.axis_counts_items != target_domain.axis_counts_items
        ):
            return None
        return LogicalRelation(
            source_domain=self.source_domain,
            target_domain=target_domain,
            pieces=self.pieces,
        )

    def project_target(
        self,
        target_domain: LogicalDomain,
    ) -> LogicalRelation | None:
        """Existentially drop target axes while preserving the remaining map."""
        current_counts = self.target_domain.axis_counts
        if any(
            axis not in current_counts or current_counts[axis] != count
            for axis, count in target_domain.axis_counts_items
        ):
            return None
        retained_axes = frozenset(target_domain.axis_order)
        return LogicalRelation(
            source_domain=self.source_domain,
            target_domain=target_domain,
            pieces=tuple(
                _LogicalRelationPiece(
                    source_bounds_items=piece.source_bounds_items,
                    target_ranges=tuple(
                        target_range
                        for target_range in piece.target_ranges
                        if target_range[0] in retained_axes
                    ),
                )
                for piece in self.pieces
            ),
        )

    def project_source(
        self,
        source_domain: LogicalDomain,
    ) -> LogicalRelation | None:
        """Union dropped source axes when their images remain rectilinear."""
        current_counts = self.source_domain.axis_counts
        if any(
            axis not in current_counts or current_counts[axis] != count
            for axis, count in source_domain.axis_counts_items
        ):
            return None
        retained_axes = frozenset(source_domain.axis_order)
        dropped_axes = frozenset(self.source_domain.axis_order) - retained_axes
        pieces: list[_LogicalRelationPiece] = []
        for piece in self.pieces:
            source_bounds = {
                axis: (begin, end, step)
                for axis, begin, end, step in piece.source_bounds_items
            }
            eliminated_uses: set[int] = set()
            target_ranges: list[tuple[int, sympy.Expr, sympy.Expr, int]] = []
            for target_axis, begin, end, target_step in piece.target_ranges:
                symbols: dict[sympy.Basic, int] = {
                    logical_axis_symbol(axis): axis
                    for axis in self.source_domain.axis_order
                }
                expression_axes = {
                    symbols[symbol]
                    for symbol in begin.free_symbols | end.free_symbols
                    if symbol in symbols
                }
                if len(expression_axes) != len(begin.free_symbols | end.free_symbols):
                    return None
                eliminated_axes = expression_axes & dropped_axes
                if not eliminated_axes:
                    target_ranges.append((target_axis, begin, end, target_step))
                    continue
                if len(eliminated_axes) != 1 or expression_axes & retained_axes:
                    return None
                (eliminated_axis,) = eliminated_axes
                if eliminated_axis in eliminated_uses:
                    # Projecting one coordinate into several target dimensions
                    # creates a diagonal rather than a Cartesian product.
                    return None
                eliminated_uses.add(eliminated_axis)
                interval = _single_axis_interval(
                    begin,
                    end,
                    domain=self.source_domain,
                )
                if interval is None or target_step != 1:
                    return None
                interval_axis, stride, offset, width = interval
                if interval_axis != eliminated_axis:
                    return None
                source_begin, source_end, source_step = source_bounds[eliminated_axis]
                final_source = (
                    source_begin
                    + (source_end - source_begin - 1) // source_step * source_step
                )
                projected_begin = sympy.Integer(offset + source_begin * stride)
                projected_end = sympy.Integer(offset + final_source * stride + width)
                if width == stride * source_step:
                    projected_step = 1
                elif width == 1:
                    projected_step = stride * source_step
                else:
                    return None
                target_ranges.append(
                    (
                        target_axis,
                        projected_begin,
                        projected_end,
                        projected_step,
                    )
                )
            pieces.append(
                _LogicalRelationPiece(
                    source_bounds_items=tuple(
                        (axis, *source_bounds[axis])
                        for axis in source_domain.axis_order
                    ),
                    target_ranges=tuple(target_ranges),
                )
            )
        return LogicalRelation(
            source_domain=source_domain,
            target_domain=self.target_domain,
            pieces=tuple(dict.fromkeys(pieces)),
        )

    def lift_source(self, source_domain: LogicalDomain) -> LogicalRelation | None:
        """Add unused source axes without changing any relation fiber."""
        source_counts = source_domain.axis_counts
        if any(
            axis not in source_counts or source_counts[axis] != count
            for axis, count in self.source_domain.axis_counts_items
        ):
            return None
        current_axes = frozenset(self.source_domain.axis_order)
        pieces: list[_LogicalRelationPiece] = []
        for piece in self.pieces:
            bounds = {
                axis: (begin, end, step)
                for axis, begin, end, step in piece.source_bounds_items
            }
            pieces.append(
                _LogicalRelationPiece(
                    source_bounds_items=tuple(
                        (
                            (axis, *bounds[axis])
                            if axis in current_axes
                            else (axis, 0, source_counts[axis], 1)
                        )
                        for axis in source_domain.axis_order
                    ),
                    target_ranges=piece.target_ranges,
                )
            )
        return LogicalRelation(
            source_domain=source_domain,
            target_domain=self.target_domain,
            pieces=tuple(dict.fromkeys(pieces)),
        )

    def then(self, following: LogicalRelation) -> LogicalRelation | None:
        """Compose a projection/full-fiber relation with another relation.

        This is the program-order composition needed for nested checkpoints.
        ``self`` maps a later scope to preceding scope instances; ``following``
        maps those preceding instances to their acquired predecessors.
        """
        if self.target_domain != following.source_domain:
            return None
        point_composition = _compose_point_relations(self, following)
        if point_composition is not None:
            return point_composition
        if len(self.pieces) != 1:
            return None
        piece = self.pieces[0]
        if piece.source_bounds_items != tuple(
            (axis, 0, self.source_domain.axis_counts[axis], 1)
            for axis in self.source_domain.axis_order
        ):
            return None

        retained_axes: list[int] = []
        source_counts = self.source_domain.axis_counts
        for axis, begin, end, step in piece.target_ranges:
            if step != 1:
                return None
            count = self.target_domain.axis_counts[axis]
            symbol = logical_axis_symbol(axis)
            if (
                axis in source_counts
                and source_counts[axis] == count
                and sympy.simplify(begin - symbol) == 0  # pyrefly: ignore[unsupported-operation]
                and sympy.simplify(end - symbol - 1) == 0  # pyrefly: ignore[unsupported-operation]
            ):
                retained_axes.append(axis)
            elif not (
                sympy.simplify(begin) == 0 and sympy.simplify(end - count) == 0  # pyrefly: ignore[unsupported-operation]
            ):
                return None

        retained_domain = LogicalDomain(
            axis_order=tuple(retained_axes),
            axis_counts_items=tuple(
                (axis, source_counts[axis]) for axis in retained_axes
            ),
            block_sizes_items=tuple(
                (axis, self.source_domain.block_sizes[axis])
                for axis in retained_axes
                if axis in self.source_domain.block_sizes
            ),
            kind=self.source_domain.kind,
            identity=self.source_domain.identity,
        )
        projected = following.project_source(retained_domain)
        return None if projected is None else projected.lift_source(self.source_domain)

    def covers(self, required: LogicalRelation) -> bool:
        """Conservatively prove that this relation contains ``required``."""
        if (
            self.source_domain != required.source_domain
            or self.target_domain != required.target_domain
        ):
            return False
        return all(
            any(
                _relation_piece_covers(
                    available,
                    needed,
                    target_domain=self.target_domain,
                )
                for available in self.pieces
            )
            for needed in required.pieces
        )

    def source_axes_used(self) -> tuple[int, ...] | None:
        """Return source axes that can change a relation fiber."""
        symbols: dict[sympy.Basic, int] = {
            logical_axis_symbol(axis): axis for axis in self.source_domain.axis_order
        }
        used: set[int] = set()
        full_source_bounds = {
            axis: (0, self.source_domain.axis_counts[axis], 1)
            for axis in self.source_domain.axis_order
        }
        for piece in self.pieces:
            for axis, begin, end, step in piece.source_bounds_items:
                if (begin, end, step) != full_source_bounds[axis]:
                    used.add(axis)
            for _axis, begin, end, _step in piece.target_ranges:
                for symbol in begin.free_symbols | end.free_symbols:
                    source_axis = symbols.get(symbol)
                    if source_axis is None:
                        return None
                    used.add(source_axis)
        return tuple(axis for axis in self.source_domain.axis_order if axis in used)

    def inverse(self) -> LogicalRelation | None:
        """Invert supported affine point/range maps without enumeration."""
        return self._cached_inverse

    @cached_property
    def _cached_inverse(self) -> LogicalRelation | None:
        pieces: list[_LogicalRelationPiece] = []
        used_mixed_radix_inverse = False
        for piece in self.pieces:
            mixed_radix_pieces = _mixed_radix_interval_inverse(
                piece,
                source_domain=self.source_domain,
                target_domain=self.target_domain,
            )
            if mixed_radix_pieces is not None:
                pieces.extend(mixed_radix_pieces)
                used_mixed_radix_inverse = True
                continue
            lower_bounds: dict[int, list[sympy.Expr]] = {}
            upper_bounds: dict[int, list[sympy.Expr]] = {}
            target_steps: dict[int, int] = {}
            for axis, begin, end, step in piece.source_bounds_items:
                if (
                    begin == 0
                    and end == self.source_domain.axis_counts[axis]
                    and step == 1
                ):
                    lower_bounds[axis] = []
                    upper_bounds[axis] = []
                    target_steps[axis] = 1
                else:
                    lower_bounds[axis] = [sympy.Integer(begin)]
                    upper_bounds[axis] = [sympy.Integer(end)]
                    target_steps[axis] = step

            inverse_source_bounds = {
                axis: [0, self.target_domain.axis_counts[axis], 1]
                for axis in self.target_domain.axis_order
            }
            for target_axis, begin, end, step in piece.target_ranges:
                if step != 1:
                    return None
                target_count = self.target_domain.axis_counts[target_axis]
                if (
                    sympy.simplify(begin) == 0
                    and sympy.simplify(end - target_count)  # pyrefly: ignore[unsupported-operation]
                    == 0
                ):
                    continue
                interval = _single_axis_interval(
                    begin,
                    end,
                    domain=self.source_domain,
                )
                if interval is None:
                    floor_point = _single_axis_floor_point(
                        begin,
                        end,
                        domain=self.source_domain,
                    )
                    if floor_point is not None:
                        (
                            source_axis,
                            numerator_stride,
                            numerator_offset,
                            divisor,
                            output_offset,
                        ) = floor_point
                        target_coordinate = logical_axis_symbol(target_axis)
                        source_begin, source_end, source_step = next(
                            (source_begin, source_end, source_step)
                            for (
                                axis,
                                source_begin,
                                source_end,
                                source_step,
                            ) in piece.source_bounds_items
                            if axis == source_axis
                        )
                        if source_step != 1:
                            return None
                        lower_bounds[source_axis].append(
                            sympy.ceiling(  # pyrefly: ignore[bad-argument-type]
                                (
                                    divisor * (target_coordinate - output_offset)  # pyrefly: ignore[unsupported-operation]
                                    - numerator_offset
                                )
                                / numerator_stride
                            )
                        )
                        upper_bounds[source_axis].append(
                            sympy.ceiling(  # pyrefly: ignore[bad-argument-type]
                                (
                                    divisor * (target_coordinate - output_offset + 1)  # pyrefly: ignore[unsupported-operation]
                                    - numerator_offset
                                )
                                / numerator_stride
                            )
                        )
                        lower_bounds[source_axis].append(sympy.Integer(source_begin))
                        upper_bounds[source_axis].append(sympy.Integer(source_end))
                        continue
                    if begin.free_symbols or end.free_symbols:
                        return None
                    inverse_source_bounds[target_axis][0] = max(
                        inverse_source_bounds[target_axis][0], int(begin)
                    )
                    inverse_source_bounds[target_axis][1] = min(
                        inverse_source_bounds[target_axis][1], int(end)
                    )
                    continue
                source_axis, stride, offset, width = interval
                source_begin, source_end, source_step = next(
                    (source_begin, source_end, source_step)
                    for axis, source_begin, source_end, source_step in piece.source_bounds_items
                    if axis == source_axis
                )
                final_source = (
                    source_begin
                    + (source_end - source_begin - 1) // source_step * source_step
                )
                inverse_source_bounds[target_axis][0] = max(
                    inverse_source_bounds[target_axis][0],
                    offset + source_begin * stride,
                )
                inverse_source_bounds[target_axis][1] = min(
                    inverse_source_bounds[target_axis][1],
                    offset + final_source * stride + width,
                )
                target_coordinate = logical_axis_symbol(target_axis)
                if width == stride:
                    target_begin = cast(
                        "sympy.Expr",
                        sympy.floor(  # pyrefly: ignore[bad-argument-type, unsupported-operation]
                            (target_coordinate - offset) / stride  # pyrefly: ignore[unsupported-operation]
                        ),
                    )
                    lower_bounds[source_axis].append(target_begin)
                    upper_bounds[source_axis].append(
                        target_begin + 1  # pyrefly: ignore[unsupported-operation]
                    )
                else:
                    lower_bounds[source_axis].append(
                        sympy.floor(  # pyrefly: ignore[bad-argument-type, unsupported-operation]
                            (target_coordinate - offset - width) / stride  # pyrefly: ignore[unsupported-operation]
                        )
                        + 1  # pyrefly: ignore[unsupported-operation]
                    )
                    upper_bounds[source_axis].append(
                        cast(
                            "sympy.Expr",
                            sympy.ceiling(  # pyrefly: ignore[bad-argument-type]
                                (target_coordinate + 1 - offset) / stride  # pyrefly: ignore[unsupported-operation]
                            ),
                        )
                    )

            pieces.append(
                _LogicalRelationPiece(
                    source_bounds_items=tuple(
                        (
                            axis,
                            inverse_source_bounds[axis][0],
                            inverse_source_bounds[axis][1],
                            inverse_source_bounds[axis][2],
                        )
                        for axis in self.target_domain.axis_order
                    ),
                    target_ranges=tuple(
                        (
                            axis,
                            (
                                sympy.Max(*lower_bounds[axis])
                                if lower_bounds[axis]
                                else sympy.Integer(0)
                            ),
                            (
                                sympy.Min(*upper_bounds[axis])
                                if upper_bounds[axis]
                                else sympy.Integer(self.source_domain.axis_counts[axis])
                            ),
                            target_steps[axis],
                        )
                        for axis in self.source_domain.axis_order
                    ),
                )
            )
        inverse = LogicalRelation(
            source_domain=self.target_domain,
            target_domain=self.source_domain,
            pieces=tuple(dict.fromkeys(pieces)),
        )
        if used_mixed_radix_inverse:
            return _compact_mixed_radix_point_map(inverse) or inverse
        return inverse

    @staticmethod
    def _evaluate(expression: sympy.Expr, coordinates: dict[int, int]) -> int:
        value = expression.xreplace(
            {
                logical_axis_symbol(axis): sympy.Integer(coordinate)
                for axis, coordinate in coordinates.items()
            }
        )
        if value.free_symbols or not isinstance(value, sympy.Integer):
            raise ValueError(
                f"relation expression did not evaluate to an integer: {value}"
            )
        return int(value)

    def target_coordinates(
        self,
        source_coordinates: dict[int, int],
    ) -> frozenset[tuple[int, ...]]:
        result: set[tuple[int, ...]] = set()
        for piece in self.pieces:
            if not piece.contains(source_coordinates):
                continue
            ranges: list[range] = []
            for target_axis, begin, end, step in piece.target_ranges:
                target_count = self.target_domain.axis_counts[target_axis]
                concrete_begin = max(0, self._evaluate(begin, source_coordinates))
                concrete_end = min(
                    target_count,
                    self._evaluate(end, source_coordinates),
                )
                ranges.append(range(concrete_begin, concrete_end, step))
            result.update(itertools.product(*ranges))
        return frozenset(result)

    def targets(
        self,
        source_index: int,
        *,
        source_traversal: tuple[int, ...] | None = None,
        target_traversal: tuple[int, ...] | None = None,
    ) -> frozenset[int]:
        """Enumerate one source fiber for differential testing."""
        source_coordinates = self.source_domain.coordinates(
            source_index,
            traversal=source_traversal,
        )
        return frozenset(
            self.target_domain.index(
                dict(zip(self.target_domain.axis_order, coordinates, strict=True)),
                traversal=target_traversal,
            )
            for coordinates in self.target_coordinates(source_coordinates)
        )

    def materialize(
        self,
        *,
        source_traversal: tuple[int, ...] | None = None,
        target_traversal: tuple[int, ...] | None = None,
    ) -> tuple[frozenset[int], ...]:
        """Enumerate the relation only for tests and small-domain validation."""
        return tuple(
            self.targets(
                index,
                source_traversal=source_traversal,
                target_traversal=target_traversal,
            )
            for index in range(self.source_domain.size)
        )

    def union(self, other: LogicalRelation) -> LogicalRelation | None:
        """Return an exact finite union when both relations share typed domains."""
        if (
            self.source_domain != other.source_domain
            or self.target_domain != other.target_domain
        ):
            return None
        if self.covers(other):
            return self
        if other.covers(self):
            return other
        return LogicalRelation(
            source_domain=self.source_domain,
            target_domain=self.target_domain,
            pieces=tuple(dict.fromkeys((*self.pieces, *other.pieces))),
        )

    def has_disjoint_source_support(self, other: LogicalRelation) -> bool:
        """Prove that no source coordinate participates in both relations."""
        if self.source_domain != other.source_domain:
            return False
        return all(
            _source_boxes_are_disjoint(left, right)
            for left in self.pieces
            for right in other.pieces
        )

    def is_total(self) -> bool:
        """Return whether one canonical piece covers the complete product."""
        if len(self.pieces) != 1:
            return False
        (piece,) = self.pieces
        return piece.source_bounds_items == tuple(
            (axis, 0, self.source_domain.axis_counts[axis], 1)
            for axis in self.source_domain.axis_order
        ) and piece.target_ranges == tuple(
            (
                axis,
                sympy.Integer(0),
                sympy.Integer(self.target_domain.axis_counts[axis]),
                1,
            )
            for axis in self.target_domain.axis_order
        )

    def has_total_source(self) -> bool:
        """Return whether every source coordinate has at least one target."""
        cells = _relation_source_cells(self, include_domain=True)
        if cells is None:
            return False
        return all(
            any(
                _source_box_covers(piece.source_bounds_items, bounds)
                and _target_box_is_nonempty_for_all_sources(
                    piece.target_ranges,
                    source_domain=self.source_domain,
                    source_bounds=bounds,
                    target_domain=self.target_domain,
                )
                for piece in self.pieces
            )
            for bounds in cells
        )

    def is_single_valued(self) -> bool:
        """Return whether every source instance maps to at most one target."""
        normalized_targets = tuple(
            tuple(
                (
                    axis,
                    _simplify_logical_expression(
                        begin,
                        domain=self.source_domain,
                        source_bounds=piece.source_bounds_items,
                    ),
                    _simplify_logical_expression(
                        end,
                        domain=self.source_domain,
                        source_bounds=piece.source_bounds_items,
                    ),
                    step,
                )
                for axis, begin, end, step in piece.target_ranges
            )
            for piece in self.pieces
        )
        if any(
            step != 1
            or sympy.simplify(end - begin)  # pyrefly: ignore[unsupported-operation]
            != 1
            for target_ranges in normalized_targets
            for _axis, begin, end, step in target_ranges
        ):
            return False
        for left_index, left in enumerate(self.pieces):
            for right_index, right in enumerate(
                self.pieces[left_index + 1 :],
                start=left_index + 1,
            ):
                if _source_boxes_are_disjoint(left, right):
                    continue
                if normalized_targets[left_index] != normalized_targets[right_index]:
                    return False
        return True

    def canonical_single_valued(self) -> LogicalRelation | None:
        """Return a disjoint-source form for an at-most-one-valued relation.

        The transformation partitions only at the constant boundaries already
        present in relation pieces.  Its cost therefore depends on relation
        complexity rather than the number of runtime source instances.  A
        strided source guard or two different values on an overlapping source
        region is rejected instead of being expanded.
        """
        cells = _relation_source_cells(self)
        if cells is None:
            return None
        pieces: list[_LogicalRelationPiece] = []
        for bounds in cells:
            active_targets = tuple(
                tuple(
                    (
                        axis,
                        _simplify_logical_expression(
                            begin,
                            domain=self.source_domain,
                            source_bounds=bounds,
                        ),
                        _simplify_logical_expression(
                            end,
                            domain=self.source_domain,
                            source_bounds=bounds,
                        ),
                        step,
                    )
                    for axis, begin, end, step in piece.target_ranges
                )
                for piece in self.pieces
                if _source_box_covers(piece.source_bounds_items, bounds)
            )
            if not active_targets:
                continue
            target_ranges = active_targets[0]
            if any(active != target_ranges for active in active_targets[1:]):
                return None
            if any(
                step != 1
                or sympy.simplify(end - begin)  # pyrefly: ignore[unsupported-operation]
                != 1
                for _axis, begin, end, step in target_ranges
            ):
                return None
            pieces.append(
                _LogicalRelationPiece(
                    source_bounds_items=bounds,
                    target_ranges=target_ranges,
                )
            )
        return LogicalRelation(
            source_domain=self.source_domain,
            target_domain=self.target_domain,
            pieces=tuple(pieces),
        )

    def is_total_function(self) -> bool:
        """Return whether every source instance maps to exactly one target."""
        canonical = self.canonical_single_valued()
        if canonical is None:
            return False
        cells = _relation_source_cells(self, include_domain=True)
        if cells is None:
            return False
        return all(
            any(
                _source_box_covers(piece.source_bounds_items, bounds)
                and _target_point_is_in_domain(
                    piece.target_ranges,
                    source_domain=self.source_domain,
                    source_bounds=bounds,
                    target_domain=self.target_domain,
                )
                for piece in canonical.pieces
            )
            for bounds in cells
        )

    def is_positional_bijection(self) -> bool:
        """Return whether coordinates are renamed position-for-position."""
        if (
            len(self.source_domain.axis_order) != len(self.target_domain.axis_order)
            or tuple(self.source_domain.axis_counts.values())
            != tuple(self.target_domain.axis_counts.values())
            or len(self.pieces) != 1
        ):
            return False
        (piece,) = self.pieces
        if piece.source_bounds_items != tuple(
            (axis, 0, self.source_domain.axis_counts[axis], 1)
            for axis in self.source_domain.axis_order
        ):
            return False
        return all(
            target_axis == expected_target_axis
            and step == 1
            and sympy.simplify(begin - logical_axis_symbol(source_axis)) == 0  # pyrefly: ignore[unsupported-operation]
            and sympy.simplify(end - begin) == 1  # pyrefly: ignore[unsupported-operation]
            for source_axis, expected_target_axis, (
                target_axis,
                begin,
                end,
                step,
            ) in zip(
                self.source_domain.axis_order,
                self.target_domain.axis_order,
                piece.target_ranges,
                strict=True,
            )
        )

    def fiber_cardinality(self) -> LogicalRelation | None:
        """Return the exact number of distinct targets in every source fiber.

        The result is another single-valued ``LogicalRelation`` whose one
        target coordinate is the cardinality.  This keeps aggregation inside
        the relation algebra while avoiding a separate scalar-expression IR.
        Source boxes are partitioned only at existing structural boundaries.
        Overlapping target boxes must be identical or provably disjoint.
        """
        cells = _relation_source_cells(self, include_domain=True)
        if cells is None:
            return None
        value_axis = 0
        value_domain = LogicalDomain(
            axis_order=(value_axis,),
            axis_counts_items=((value_axis, self.target_domain.size + 1),),
            kind="value",
        )
        pieces: list[_LogicalRelationPiece] = []
        for bounds in cells:
            active_targets = tuple(
                dict.fromkeys(
                    piece.target_ranges
                    for piece in self.pieces
                    if _source_box_covers(piece.source_bounds_items, bounds)
                )
            )
            if any(
                not _target_boxes_are_disjoint(
                    left,
                    right,
                    source_domain=self.source_domain,
                    source_bounds=bounds,
                )
                for left_index, left in enumerate(active_targets)
                for right in active_targets[left_index + 1 :]
            ):
                return None
            cardinality = sympy.Add(
                *(
                    _target_box_cardinality(
                        target_ranges,
                        target_domain=self.target_domain,
                        source_domain=self.source_domain,
                        source_bounds=bounds,
                    )
                    for target_ranges in active_targets
                )
            )
            cardinality = sympy.simplify(cardinality)
            pieces.append(
                _LogicalRelationPiece(
                    source_bounds_items=bounds,
                    target_ranges=(
                        (
                            value_axis,
                            cardinality,
                            cardinality + 1,  # pyrefly: ignore[unsupported-operation]
                            1,
                        ),
                    ),
                )
            )
        return LogicalRelation(
            source_domain=self.source_domain,
            target_domain=value_domain,
            pieces=tuple(pieces),
        )

    def fiber_maximum(
        self,
        values: LogicalRelation,
    ) -> LogicalRelation | None:
        """Reduce each target fiber through a single-valued integer map.

        ``self`` maps source coordinates to a set of target coordinates and
        ``values`` maps those target coordinates to one scalar value.  The
        result maps every nonempty source fiber to its maximum value without
        enumerating either domain.  Unsupported intersections decline rather
        than approximating the dependency.
        """
        if (
            self.target_domain != values.source_domain
            or len(values.target_domain.axis_order) != 1
            or not values.is_total_function()
        ):
            return None
        pieces_by_source_bounds: dict[
            tuple[tuple[int, int, int, int], ...],
            list[_LogicalRelationPiece],
        ] = {}
        for piece in self.pieces:
            pieces_by_source_bounds.setdefault(piece.source_bounds_items, []).append(
                piece
            )
        if _source_boxes_partition_domain(
            tuple(pieces_by_source_bounds),
            self.source_domain,
        ):
            active_pieces_by_cell = tuple(
                (bounds, tuple(active_pieces))
                for bounds, active_pieces in pieces_by_source_bounds.items()
            )
        else:
            cells = _relation_source_cells(self, include_domain=True)
            if cells is None:
                return None
            active_pieces_by_cell = tuple(
                (
                    bounds,
                    tuple(
                        piece
                        for piece in self.pieces
                        if _source_box_covers(piece.source_bounds_items, bounds)
                    ),
                )
                for bounds in cells
            )
        value_axis = values.target_domain.axis_order[0]
        pieces: list[_LogicalRelationPiece] = []
        for source_bounds, active_pieces in active_pieces_by_cell:
            maxima: list[sympy.Expr] = []
            for relation_piece in active_pieces:
                for value_piece in values.pieces:
                    intersection = _intersect_target_with_source_box(
                        relation_piece.target_ranges,
                        value_piece.source_bounds_items,
                        source_domain=self.source_domain,
                        relation_source_bounds=source_bounds,
                    )
                    if intersection is None:
                        return None
                    if intersection is False:
                        continue
                    if len(value_piece.target_ranges) != 1:
                        return None
                    _axis, begin, end, step = value_piece.target_ranges[0]
                    if (
                        step != 1
                        or sympy.simplify(end - begin)  # pyrefly: ignore[unsupported-operation]
                        != 1
                    ):
                        return None
                    maximum = _target_box_expression_extreme(
                        begin,
                        target_domain=self.target_domain,
                        target_ranges=cast(
                            "tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...]",
                            intersection,
                        ),
                        maximize=True,
                    )
                    if maximum is None:
                        return None
                    maxima.append(maximum)
            if not maxima:
                continue
            maximum = _fiber_maximum_expression(
                tuple(maxima),
                source_domain=self.source_domain,
                source_bounds=source_bounds,
            )
            pieces.append(
                _LogicalRelationPiece(
                    source_bounds_items=source_bounds,
                    target_ranges=(
                        (
                            value_axis,
                            maximum,
                            maximum + 1,  # pyrefly: ignore[unsupported-operation]
                            1,
                        ),
                    ),
                )
            )
        return LogicalRelation(
            source_domain=self.source_domain,
            target_domain=values.target_domain,
            pieces=tuple(pieces),
        )

    def fiber_enumeration(self) -> LogicalRelation | None:
        """Enumerate uniform rectangular fibers with one symbolic local axis.

        The returned relation maps ``(within_fiber, source coordinates)`` to
        one target coordinate.  It is a compact bijection when this relation's
        target boxes are disjoint and every source fiber has the same static
        cardinality.  No source or target instance is materialized.
        """
        cells = _relation_source_cells(self, include_domain=True)
        if cells is None:
            return None
        cell_targets: list[
            tuple[
                tuple[tuple[int, int, int, int], ...],
                tuple[
                    tuple[
                        tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...],
                        int,
                    ],
                    ...,
                ],
            ]
        ] = []
        fiber_size: int | None = None
        for bounds in cells:
            active_targets = tuple(
                dict.fromkeys(
                    piece.target_ranges
                    for piece in self.pieces
                    if _source_box_covers(piece.source_bounds_items, bounds)
                )
            )
            if not active_targets or any(
                not _target_boxes_are_disjoint(
                    left,
                    right,
                    source_domain=self.source_domain,
                    source_bounds=bounds,
                )
                for left_index, left in enumerate(active_targets)
                for right in active_targets[left_index + 1 :]
            ):
                return None
            boxes: list[
                tuple[
                    tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...],
                    int,
                ]
            ] = []
            for target_ranges in active_targets:
                cardinality = _target_box_cardinality(
                    target_ranges,
                    target_domain=self.target_domain,
                    source_domain=self.source_domain,
                    source_bounds=bounds,
                )
                if (
                    cardinality.free_symbols or cardinality.is_integer is not True  # pyrefly: ignore[missing-attribute]
                ):
                    return None
                count = int(cardinality)
                if count <= 0:
                    continue
                boxes.append((target_ranges, count))
            total = sum(count for _ranges, count in boxes)
            if not total or (fiber_size is not None and total != fiber_size):
                return None
            fiber_size = total
            cell_targets.append((bounds, tuple(boxes)))
        if fiber_size is None:
            return None

        all_axes = {
            *self.source_domain.axis_order,
            *self.target_domain.axis_order,
        }
        within_axis = min(all_axes, default=0) - 1
        enumeration_domain = LogicalDomain(
            axis_order=(within_axis, *self.source_domain.axis_order),
            axis_counts_items=(
                (within_axis, fiber_size),
                *self.source_domain.axis_counts_items,
            ),
            kind="worker",
        )
        within = logical_axis_symbol(within_axis)
        pieces: list[_LogicalRelationPiece] = []
        for source_bounds, target_boxes in cell_targets:
            within_begin = 0
            for target_ranges, count in target_boxes:
                local = within - within_begin  # pyrefly: ignore[unsupported-operation]
                multiplier = 1
                target_points: list[tuple[int, sympy.Expr, sympy.Expr, int]] = []
                for axis, begin, end, step in target_ranges:
                    simplified_begin = _simplify_logical_expression(
                        begin,
                        domain=self.source_domain,
                        source_bounds=source_bounds,
                    )
                    simplified_end = _simplify_logical_expression(
                        end,
                        domain=self.source_domain,
                        source_bounds=source_bounds,
                    )
                    extent = sympy.simplify(
                        (simplified_end - simplified_begin) / step  # pyrefly: ignore[unsupported-operation]
                    )
                    if extent.free_symbols or extent.is_integer is not True:
                        return None
                    axis_count = int(extent)
                    if axis_count <= 0:
                        return None
                    coordinate = simplified_begin
                    if axis_count != 1:
                        coordinate += (  # pyrefly: ignore[unsupported-operation]
                            sympy.floor(local / multiplier) % axis_count  # pyrefly: ignore[unsupported-operation]
                        ) * step
                    target_points.append(
                        (
                            axis,
                            coordinate,
                            coordinate + 1,  # pyrefly: ignore[unsupported-operation]
                            1,
                        )
                    )
                    multiplier *= axis_count
                if multiplier != count:
                    return None
                pieces.append(
                    _LogicalRelationPiece(
                        source_bounds_items=(
                            (within_axis, within_begin, within_begin + count, 1),
                            *source_bounds,
                        ),
                        target_ranges=tuple(target_points),
                    )
                )
                within_begin += count
            if within_begin != fiber_size:
                return None
        return LogicalRelation(
            source_domain=enumeration_domain,
            target_domain=self.target_domain,
            pieces=tuple(pieces),
        )

    def constant_value(self) -> int | None:
        """Return one integer value when this is a total constant function."""
        canonical = self.canonical_single_valued()
        if canonical is None or not canonical.is_total_function():
            return None
        values: set[int] = set()
        for piece in canonical.pieces:
            if len(piece.target_ranges) != 1:
                return None
            _axis, begin, end, step = piece.target_ranges[0]
            begin = _simplify_logical_expression(
                begin,
                domain=canonical.source_domain,
                source_bounds=piece.source_bounds_items,
            )
            end = _simplify_logical_expression(
                end,
                domain=canonical.source_domain,
                source_bounds=piece.source_bounds_items,
            )
            if (
                step != 1
                or sympy.simplify(end - begin)  # pyrefly: ignore[unsupported-operation]
                != 1
            ):
                return None
            bounds = _logical_expression_bounds(
                begin,
                domain=canonical.source_domain,
                source_bounds=piece.source_bounds_items,
            )
            if (
                bounds is None
                or sympy.simplify(bounds[1] - bounds[0]) != 0  # pyrefly: ignore[unsupported-operation]
                or bounds[0].is_integer is not True  # pyrefly: ignore[missing-attribute]
            ):
                return None
            values.add(int(bounds[0]))
        if len(values) != 1:
            return None
        return values.pop()

    def value_bounds(
        self,
        fixed_coordinates: dict[int, int] | None = None,
    ) -> tuple[int, int] | None:
        """Return exact-enough scalar bounds under fixed source coordinates."""
        if len(self.target_domain.axis_order) != 1:
            return None
        fixed_coordinates = {} if fixed_coordinates is None else fixed_coordinates
        substitutions = {
            logical_axis_symbol(axis): sympy.Integer(coordinate)
            for axis, coordinate in fixed_coordinates.items()
        }
        minima: list[sympy.Expr] = []
        maxima: list[sympy.Expr] = []
        for piece in self.pieces:
            bounds: list[tuple[int, int, int, int]] = []
            active = True
            for axis, begin, end, step in piece.source_bounds_items:
                fixed = fixed_coordinates.get(axis)
                if fixed is None:
                    bounds.append((axis, begin, end, step))
                elif begin <= fixed < end and (fixed - begin) % step == 0:
                    bounds.append((axis, fixed, fixed + 1, 1))
                else:
                    active = False
                    break
            if not active or len(piece.target_ranges) != 1:
                continue
            _axis, begin, end, step = piece.target_ranges[0]
            if (
                step != 1
                or sympy.simplify(end - begin)  # pyrefly: ignore[unsupported-operation]
                != 1
            ):
                return None
            value_range = _logical_expression_bounds(
                begin.xreplace(substitutions),
                domain=self.source_domain,
                source_bounds=tuple(bounds),
            )
            if value_range is None:
                return None
            minima.append(value_range[0])
            maxima.append(value_range[1])
        if not minima:
            return None
        minimum = sympy.Min(*minima)
        maximum = sympy.Max(*maxima)
        if (
            minimum.free_symbols
            or maximum.free_symbols
            or minimum.is_integer is not True  # pyrefly: ignore[missing-attribute]
            or maximum.is_integer is not True  # pyrefly: ignore[missing-attribute]
        ):
            return None
        return int(minimum), int(maximum)

    def overlapping_sources(
        self,
        other: LogicalRelation,
    ) -> LogicalRelation | None:
        """Relate ``other`` sources to ``self`` sources by target overlap.

        This is the dependency composition used by memory accesses: ``self``
        maps producer instances to allocation coordinates and ``other`` maps
        consumer instances to the same allocation coordinates.  Unsupported
        relation shapes decline instead of expanding either source domain.
        """
        if self.target_domain != other.target_domain:
            return None
        target_counts = self.source_domain.axis_counts
        pieces: list[_LogicalRelationPiece] = []
        for producer_piece in self.pieces:
            full_producer_bounds = tuple(
                (axis, 0, self.source_domain.axis_counts[axis], 1)
                for axis in self.source_domain.axis_order
            )
            if producer_piece.source_bounds_items != full_producer_bounds:
                return None
            producer_ranges = {
                axis: (begin, end, step)
                for axis, begin, end, step in producer_piece.target_ranges
            }
            for consumer_piece in other.pieces:
                consumer_ranges = {
                    axis: (begin, end, step)
                    for axis, begin, end, step in consumer_piece.target_ranges
                }
                lower_bounds: dict[int, list[sympy.Expr]] = {
                    axis: [] for axis in self.source_domain.axis_order
                }
                upper_bounds: dict[int, list[sympy.Expr]] = {
                    axis: [] for axis in self.source_domain.axis_order
                }
                for allocation_axis in self.target_domain.axis_order:
                    producer_range = producer_ranges[allocation_axis]
                    consumer_range = consumer_ranges[allocation_axis]
                    if producer_range[2] != 1 or consumer_range[2] != 1:
                        return None
                    producer_begin, producer_end, _ = producer_range
                    consumer_begin, consumer_end, _ = consumer_range
                    allocation_count = self.target_domain.axis_counts[allocation_axis]
                    if (
                        sympy.simplify(producer_begin) == 0
                        and sympy.simplify(producer_end - allocation_count)  # pyrefly: ignore[unsupported-operation]
                        == 0
                    ):
                        continue
                    producer_interval = _single_axis_interval(
                        producer_begin,
                        producer_end,
                        domain=self.source_domain,
                    )
                    if producer_interval is None:
                        return None
                    producer_axis, stride, offset, width = producer_interval
                    consumer_interval = _single_axis_interval(
                        consumer_begin,
                        consumer_end,
                        domain=other.source_domain,
                    )
                    if consumer_interval is not None:
                        (
                            _consumer_axis,
                            consumer_stride,
                            consumer_offset,
                            consumer_width,
                        ) = consumer_interval
                        if (
                            width == stride
                            and consumer_width == consumer_stride
                            and stride % consumer_width == 0
                            and (consumer_offset - offset) % consumer_width == 0
                        ):
                            target_begin = cast(
                                "sympy.Expr",
                                sympy.floor(  # pyrefly: ignore[bad-argument-type]
                                    (consumer_begin - offset) / stride  # pyrefly: ignore[unsupported-operation]
                                ),
                            )
                            lower_bounds[producer_axis].append(target_begin)
                            upper_bounds[producer_axis].append(
                                target_begin + 1  # pyrefly: ignore[unsupported-operation]
                            )
                            continue
                    lower_bounds[producer_axis].append(
                        sympy.floor((consumer_begin - offset - width) / stride)  # pyrefly: ignore[unsupported-operation]
                        + 1  # pyrefly: ignore[unsupported-operation]
                    )
                    upper_bounds[producer_axis].append(
                        cast(
                            "sympy.Expr",
                            sympy.ceiling(  # pyrefly: ignore[bad-argument-type]
                                (consumer_end - offset) / stride  # pyrefly: ignore[unsupported-operation]
                            ),
                        )
                    )
                pieces.append(
                    _LogicalRelationPiece(
                        source_bounds_items=consumer_piece.source_bounds_items,
                        target_ranges=tuple(
                            (
                                axis,
                                (
                                    sympy.Max(*lower_bounds[axis])
                                    if lower_bounds[axis]
                                    else sympy.Integer(0)
                                ),
                                (
                                    sympy.Min(*upper_bounds[axis])
                                    if upper_bounds[axis]
                                    else sympy.Integer(target_counts[axis])
                                ),
                                1,
                            )
                            for axis in self.source_domain.axis_order
                        ),
                    )
                )
        return LogicalRelation(
            source_domain=other.source_domain,
            target_domain=self.source_domain,
            pieces=tuple(dict.fromkeys(pieces)),
        )


def _source_boxes_are_disjoint(
    left: _LogicalRelationPiece,
    right: _LogicalRelationPiece,
) -> bool:
    """Prove two concrete strided source boxes have no common point."""
    return _source_bounds_are_disjoint(
        left.source_bounds_items,
        right.source_bounds_items,
    )


def _source_bounds_are_disjoint(
    left: tuple[tuple[int, int, int, int], ...],
    right: tuple[tuple[int, int, int, int], ...],
) -> bool:
    """Prove two concrete strided source bounds have no common point."""
    for left_bound, right_bound in zip(
        left,
        right,
        strict=True,
    ):
        left_axis, left_begin, left_end, left_step = left_bound
        right_axis, right_begin, right_end, right_step = right_bound
        if left_axis != right_axis:
            return True
        overlap_begin = max(left_begin, right_begin)
        overlap_end = min(left_end, right_end)
        if overlap_begin >= overlap_end:
            return True
        if (right_begin - left_begin) % math.gcd(left_step, right_step):
            return True
    return False


def _source_boxes_partition_domain(
    boxes: tuple[tuple[tuple[int, int, int, int], ...], ...],
    domain: LogicalDomain,
) -> bool:
    """Prove that distinct unit-stride boxes partition a finite domain."""
    if not boxes:
        return False
    if not domain.axis_order:
        return boxes == ((),)
    counts = domain.axis_counts
    if any(
        tuple(axis for axis, _begin, _end, _step in box) != domain.axis_order
        or any(
            step != 1 or begin < 0 or end > counts[axis] or begin >= end
            for axis, begin, end, step in box
        )
        for box in boxes
    ):
        return False
    if (
        sum(math.prod(end - begin for _axis, begin, end, _step in box) for box in boxes)
        != domain.size
    ):
        return False

    sweep_index = max(
        range(len(domain.axis_order)),
        key=lambda index: len({(box[index][1], box[index][2]) for box in boxes}),
    )
    active: list[tuple[tuple[int, int, int, int], ...]] = []
    for box in sorted(boxes, key=lambda item: item[sweep_index][1]):
        begin = box[sweep_index][1]
        active = [other for other in active if other[sweep_index][2] > begin]
        if any(not _source_bounds_are_disjoint(other, box) for other in active):
            return False
        active.append(box)
    return True


def _source_box_covers(
    outer: tuple[tuple[int, int, int, int], ...],
    inner: tuple[tuple[int, int, int, int], ...],
) -> bool:
    """Return whether one unit-stride source box contains another."""
    return all(
        outer_axis == inner_axis
        and outer_step == inner_step == 1
        and outer_begin <= inner_begin
        and inner_end <= outer_end
        for (
            outer_axis,
            outer_begin,
            outer_end,
            outer_step,
        ), (
            inner_axis,
            inner_begin,
            inner_end,
            inner_step,
        ) in zip(outer, inner, strict=True)
    )


def _relation_source_cells(
    relation: LogicalRelation,
    *,
    include_domain: bool = False,
) -> tuple[tuple[tuple[int, int, int, int], ...], ...] | None:
    """Partition source space at relation-piece boundaries without enumeration."""
    if any(
        step != 1
        for piece in relation.pieces
        for _axis, _begin, _end, step in piece.source_bounds_items
    ):
        return None
    cuts: dict[int, set[int]] = {
        axis: ({0, count} if include_domain else set())
        for axis, count in relation.source_domain.axis_counts_items
    }
    for piece in relation.pieces:
        for axis, begin, end, _step in piece.source_bounds_items:
            cuts[axis].update((begin, end))
    if any(len(axis_cuts) < 2 for axis_cuts in cuts.values()):
        return ()
    intervals = tuple(
        tuple(
            (axis, begin, end, 1)
            for begin, end in itertools.pairwise(sorted(cuts[axis]))
            if begin < end
        )
        for axis in relation.source_domain.axis_order
    )
    return tuple(
        tuple(cell)
        for cell in itertools.product(*intervals)
        if any(
            _source_box_covers(piece.source_bounds_items, tuple(cell))
            for piece in relation.pieces
        )
        or include_domain
    )


def _target_boxes_are_disjoint(
    left: tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...],
    right: tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...],
    *,
    source_domain: LogicalDomain,
    source_bounds: tuple[tuple[int, int, int, int], ...],
) -> bool:
    """Conservatively prove two symbolic Cartesian target boxes disjoint."""
    for left_range, right_range in zip(left, right, strict=True):
        left_axis, left_begin, left_end, left_step = left_range
        right_axis, right_begin, right_end, right_step = right_range
        if left_axis != right_axis:
            return False
        if left_step != 1 or right_step != 1:
            continue
        left_before_right = sympy.simplify(right_begin - left_end)  # pyrefly: ignore[unsupported-operation]
        right_before_left = sympy.simplify(left_begin - right_end)  # pyrefly: ignore[unsupported-operation]
        left_before_right_bounds = _logical_expression_bounds(
            left_before_right,
            domain=source_domain,
            source_bounds=source_bounds,
        )
        right_before_left_bounds = _logical_expression_bounds(
            right_before_left,
            domain=source_domain,
            source_bounds=source_bounds,
        )
        if (
            left_before_right.is_nonnegative is True
            or right_before_left.is_nonnegative is True
            or (
                left_before_right_bounds is not None
                and left_before_right_bounds[0] >= 0  # pyrefly: ignore[unsupported-operation]
            )
            or (
                right_before_left_bounds is not None
                and right_before_left_bounds[0] >= 0  # pyrefly: ignore[unsupported-operation]
            )
        ):
            return True
    return False


def _logical_expression_bounds(
    expression: sympy.Expr,
    *,
    domain: LogicalDomain,
    source_bounds: tuple[tuple[int, int, int, int], ...],
    symbol_substitutions: dict[sympy.Basic, sympy.Expr] | None = None,
) -> tuple[sympy.Expr, sympy.Expr] | None:
    """Return conservative inclusive bounds for the restricted expression IR."""
    if expression.is_number:
        return expression, expression
    if isinstance(expression, sympy.Symbol):
        replacement = (
            None
            if symbol_substitutions is None
            else symbol_substitutions.get(expression)
        )
        if replacement is not None and replacement != expression:
            return _logical_expression_bounds(
                replacement,
                domain=domain,
                source_bounds=source_bounds,
            )
        axis_by_symbol = {logical_axis_symbol(axis): axis for axis in domain.axis_order}
        axis = axis_by_symbol.get(expression)
        if axis is None:
            return None
        begin, end, step = next(
            (begin, end, step)
            for bound_axis, begin, end, step in source_bounds
            if bound_axis == axis
        )
        final = begin + (end - begin - 1) // step * step
        return sympy.Integer(begin), sympy.Integer(final)
    if isinstance(expression, sympy.Add):
        child_bounds = tuple(
            _logical_expression_bounds(
                child,
                domain=domain,
                source_bounds=source_bounds,
                symbol_substitutions=symbol_substitutions,
            )
            for child in expression.args
        )
        if any(bounds is None for bounds in child_bounds):
            return None
        concrete = tuple(bounds for bounds in child_bounds if bounds is not None)
        return (
            sympy.Add(*(bounds[0] for bounds in concrete)),
            sympy.Add(*(bounds[1] for bounds in concrete)),
        )
    if isinstance(expression, sympy.Mul):
        numeric = sympy.Integer(1)
        symbolic: list[sympy.Expr] = []
        for child in expression.args:
            if child.is_number:
                numeric *= child  # pyrefly: ignore[unsupported-operation]
            else:
                symbolic.append(child)
        if len(symbolic) != 1:
            return None
        bounds = _logical_expression_bounds(
            symbolic[0],
            domain=domain,
            source_bounds=source_bounds,
            symbol_substitutions=symbol_substitutions,
        )
        if bounds is None or numeric.is_real is not True:
            return None
        values = (
            numeric * bounds[0],  # pyrefly: ignore[unsupported-operation]
            numeric * bounds[1],  # pyrefly: ignore[unsupported-operation]
        )
        if numeric >= 0:
            return values
        return values[1], values[0]
    if expression.func in (sympy.floor, sympy.ceiling):
        bounds = _logical_expression_bounds(
            cast("sympy.Expr", expression.args[0]),
            domain=domain,
            source_bounds=source_bounds,
            symbol_substitutions=symbol_substitutions,
        )
        if bounds is None:
            return None
        return expression.func(bounds[0]), expression.func(bounds[1])
    if expression.func in (sympy.Min, sympy.Max):
        child_bounds = tuple(
            _logical_expression_bounds(
                cast("sympy.Expr", child),
                domain=domain,
                source_bounds=source_bounds,
                symbol_substitutions=symbol_substitutions,
            )
            for child in expression.args
        )
        if any(bounds is None for bounds in child_bounds):
            return None
        concrete = tuple(bounds for bounds in child_bounds if bounds is not None)
        return (
            expression.func(*(bounds[0] for bounds in concrete)),
            expression.func(*(bounds[1] for bounds in concrete)),
        )
    if isinstance(expression, sympy.Mod):
        modulus = expression.args[1]
        if not isinstance(modulus, sympy.Integer) or modulus <= 0:
            return None
        return sympy.Integer(0), modulus - 1
    return None


def _intersect_target_with_source_box(
    target_ranges: tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...],
    value_source_bounds: tuple[tuple[int, int, int, int], ...],
    *,
    source_domain: LogicalDomain,
    relation_source_bounds: tuple[tuple[int, int, int, int], ...],
) -> tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...] | bool | None:
    """Return a contained target box, ``False`` if disjoint, else unknown."""
    bounds_by_axis = {
        axis: (begin, end, step) for axis, begin, end, step in value_source_bounds
    }
    for axis, begin, end, step in target_ranges:
        source_begin, source_end, source_step = bounds_by_axis[axis]
        if step != source_step:
            return None
        before = _logical_expression_bounds(
            end - source_begin,  # pyrefly: ignore[unsupported-operation]
            domain=source_domain,
            source_bounds=relation_source_bounds,
        )
        after = _logical_expression_bounds(
            begin - source_end,  # pyrefly: ignore[unsupported-operation]
            domain=source_domain,
            source_bounds=relation_source_bounds,
        )
        if (before is not None and before[1] <= 0) or (  # pyrefly: ignore[unsupported-operation]
            after is not None and after[0] >= 0  # pyrefly: ignore[unsupported-operation]
        ):
            return False
        lower = _logical_expression_bounds(
            begin - source_begin,  # pyrefly: ignore[unsupported-operation]
            domain=source_domain,
            source_bounds=relation_source_bounds,
        )
        upper = _logical_expression_bounds(
            source_end - end,  # pyrefly: ignore[unsupported-operation]
            domain=source_domain,
            source_bounds=relation_source_bounds,
        )
        if (
            lower is None
            or lower[0] < 0  # pyrefly: ignore[unsupported-operation]
            or upper is None
            or upper[0] < 0  # pyrefly: ignore[unsupported-operation]
        ):
            return None
    return target_ranges


def _target_box_expression_extreme(
    expression: sympy.Expr,
    *,
    target_domain: LogicalDomain,
    target_ranges: tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...],
    maximize: bool,
) -> sympy.Expr | None:
    """Substitute a box endpoint into a coordinatewise-monotone expression."""
    ranges = {
        logical_axis_symbol(axis): (begin, end, step)
        for axis, begin, end, step in target_ranges
    }
    if expression.is_number:
        return expression
    if isinstance(expression, sympy.Symbol):
        target_range = ranges.get(expression)
        if target_range is None:
            return None
        begin, end, step = target_range
        return (
            end - step  # pyrefly: ignore[unsupported-operation]
            if maximize
            else begin
        )
    if isinstance(expression, sympy.Add):
        children = tuple(
            _target_box_expression_extreme(
                child,
                target_domain=target_domain,
                target_ranges=target_ranges,
                maximize=maximize,
            )
            for child in expression.args
        )
        if any(child is None for child in children):
            return None
        return sympy.Add(*(child for child in children if child is not None))
    if isinstance(expression, sympy.Mul):
        numeric = sympy.Integer(1)
        symbolic: list[sympy.Expr] = []
        for child in expression.args:
            if child.is_number:
                numeric *= child  # pyrefly: ignore[unsupported-operation]
            else:
                symbolic.append(child)
        if len(symbolic) != 1 or numeric.is_real is not True:
            return None
        child = _target_box_expression_extreme(
            symbolic[0],
            target_domain=target_domain,
            target_ranges=target_ranges,
            maximize=maximize if numeric >= 0 else not maximize,
        )
        return None if child is None else numeric * child  # pyrefly: ignore[unsupported-operation]
    if expression.func in (sympy.floor, sympy.ceiling, sympy.Min, sympy.Max):
        children = tuple(
            _target_box_expression_extreme(
                cast("sympy.Expr", child),
                target_domain=target_domain,
                target_ranges=target_ranges,
                maximize=maximize,
            )
            for child in expression.args
        )
        if any(child is None for child in children):
            return None
        return expression.func(*(child for child in children if child is not None))
    return None


def _fiber_maximum_expression(
    expressions: tuple[sympy.Expr, ...],
    *,
    source_domain: LogicalDomain,
    source_bounds: tuple[tuple[int, int, int, int], ...],
) -> sympy.Expr:
    """Select a provably dominant fiber value without a costly symbolic Max."""
    unique = tuple(dict.fromkeys(expressions))
    if len(unique) == 1:
        return unique[0]
    bounds = tuple(
        _logical_expression_bounds(
            expression,
            domain=source_domain,
            source_bounds=source_bounds,
        )
        for expression in unique
    )
    for index, candidate in enumerate(bounds):
        if candidate is None or any(value.free_symbols for value in candidate):
            continue
        candidate_minimum = int(candidate[0])
        if all(
            other is not None
            and not any(value.free_symbols for value in other)
            and candidate_minimum >= int(other[1])
            for other_index, other in enumerate(bounds)
            if other_index != index
        ):
            return unique[index]
    return sympy.Max(*unique, evaluate=False)


def _simplify_logical_expression(
    expression: sympy.Expr,
    *,
    domain: LogicalDomain,
    source_bounds: tuple[tuple[int, int, int, int], ...],
) -> sympy.Expr:
    """Simplify min/max expressions using the relation source bounds."""
    if not expression.args:
        bounds = _logical_expression_bounds(
            expression,
            domain=domain,
            source_bounds=source_bounds,
        )
        if bounds is not None and sympy.simplify(bounds[1] - bounds[0]) == 0:  # pyrefly: ignore[unsupported-operation]
            return sympy.simplify(bounds[0])
        return expression
    children = tuple(
        _simplify_logical_expression(
            child,
            domain=domain,
            source_bounds=source_bounds,
        )
        if isinstance(child, sympy.Expr)
        else child
        for child in expression.args
    )
    rebuilt = expression.func(*children)
    bounds = _logical_expression_bounds(
        rebuilt,
        domain=domain,
        source_bounds=source_bounds,
    )
    if bounds is not None and sympy.simplify(bounds[1] - bounds[0]) == 0:  # pyrefly: ignore[unsupported-operation]
        return sympy.simplify(bounds[0])
    if rebuilt.func not in (sympy.Min, sympy.Max):
        return sympy.simplify(rebuilt)
    child_bounds = tuple(
        _logical_expression_bounds(
            cast("sympy.Expr", child),
            domain=domain,
            source_bounds=source_bounds,
        )
        for child in children
    )
    if any(bounds is None for bounds in child_bounds):
        return rebuilt
    concrete = tuple(bounds for bounds in child_bounds if bounds is not None)
    for index, child in enumerate(children):
        if rebuilt.func == sympy.Min and all(
            concrete[index][1] <= other[0]  # pyrefly: ignore[unsupported-operation]
            for other_index, other in enumerate(concrete)
            if other_index != index
        ):
            return cast("sympy.Expr", child)
        if rebuilt.func == sympy.Max and all(
            concrete[index][0] >= other[1]  # pyrefly: ignore[unsupported-operation]
            for other_index, other in enumerate(concrete)
            if other_index != index
        ):
            return cast("sympy.Expr", child)
    return rebuilt


def _target_box_cardinality(
    target_ranges: tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...],
    *,
    target_domain: LogicalDomain,
    source_domain: LogicalDomain,
    source_bounds: tuple[tuple[int, int, int, int], ...],
) -> sympy.Expr:
    """Return the clipped Cartesian cardinality of one target box."""
    cardinality: sympy.Expr = sympy.Integer(1)
    for axis, begin, end, step in target_ranges:
        begin = _simplify_logical_expression(
            begin,
            domain=source_domain,
            source_bounds=source_bounds,
        )
        end = _simplify_logical_expression(
            end,
            domain=source_domain,
            source_bounds=source_bounds,
        )
        begin_bounds = _logical_expression_bounds(
            begin,
            domain=source_domain,
            source_bounds=source_bounds,
        )
        end_bounds = _logical_expression_bounds(
            end,
            domain=source_domain,
            source_bounds=source_bounds,
        )
        clipped_begin = (
            begin
            if begin_bounds is not None and begin_bounds[0] >= 0  # pyrefly: ignore[unsupported-operation]
            else sympy.Max(sympy.Integer(0), begin)
        )
        clipped_end = (
            end
            if end_bounds is not None
            and end_bounds[1] <= target_domain.axis_counts[axis]  # pyrefly: ignore[unsupported-operation]
            else sympy.Min(
                sympy.Integer(target_domain.axis_counts[axis]),
                end,
            )
        )
        width = sympy.Max(  # pyrefly: ignore[unsupported-operation]
            sympy.Integer(0),
            clipped_end - clipped_begin,  # pyrefly: ignore[unsupported-operation]
        )
        extent = (
            width
            if step == 1
            else sympy.floor(  # pyrefly: ignore[bad-argument-type]
                (width + step - 1) / step  # pyrefly: ignore[unsupported-operation]
            )
        )
        cardinality *= extent  # pyrefly: ignore[unsupported-operation]
    return sympy.simplify(cardinality)


def _target_box_is_nonempty_for_all_sources(
    target_ranges: tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...],
    *,
    source_domain: LogicalDomain,
    source_bounds: tuple[tuple[int, int, int, int], ...],
    target_domain: LogicalDomain,
) -> bool:
    """Prove that a clipped target box is nonempty for every source point."""
    for axis, begin, end, step in target_ranges:
        if step != 1:
            return False
        begin_bounds = _logical_expression_bounds(
            begin,
            domain=source_domain,
            source_bounds=source_bounds,
        )
        end_bounds = _logical_expression_bounds(
            end,
            domain=source_domain,
            source_bounds=source_bounds,
        )
        width_bounds = _logical_expression_bounds(
            end - begin,  # pyrefly: ignore[unsupported-operation]
            domain=source_domain,
            source_bounds=source_bounds,
        )
        if (
            begin_bounds is None
            or end_bounds is None
            or width_bounds is None
            or begin_bounds[1] >= target_domain.axis_counts[axis]  # pyrefly: ignore[unsupported-operation]
            or end_bounds[0] <= 0  # pyrefly: ignore[unsupported-operation]
            or width_bounds[0] <= 0  # pyrefly: ignore[unsupported-operation]
        ):
            return False
    return True


def _target_point_is_in_domain(
    target_ranges: tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...],
    *,
    source_domain: LogicalDomain,
    source_bounds: tuple[tuple[int, int, int, int], ...],
    target_domain: LogicalDomain,
) -> bool:
    """Prove that a single-valued target remains inside its typed domain."""
    for axis, begin, end, step in target_ranges:
        if (
            step != 1
            or sympy.simplify(end - begin)  # pyrefly: ignore[unsupported-operation]
            != 1
        ):
            return False
        bounds = _logical_expression_bounds(
            begin,
            domain=source_domain,
            source_bounds=source_bounds,
        )
        if (
            bounds is None
            or bounds[0] < 0  # pyrefly: ignore[unsupported-operation]
            or bounds[1] >= target_domain.axis_counts[axis]  # pyrefly: ignore[unsupported-operation]
        ):
            return False
    return True


def _relation_piece_covers(
    available: _LogicalRelationPiece,
    required: _LogicalRelationPiece,
    *,
    target_domain: LogicalDomain,
) -> bool:
    available_bounds = {
        axis: (begin, end, step)
        for axis, begin, end, step in available.source_bounds_items
    }
    for axis, begin, end, step in required.source_bounds_items:
        available_begin, available_end, available_step = available_bounds[axis]
        if (
            available_begin > begin
            or available_end < end
            or (
                available_step != 1
                and (
                    step % available_step != 0
                    or (begin - available_begin) % available_step != 0
                )
            )
        ):
            return False

    available_ranges = {
        axis: (begin, end, step) for axis, begin, end, step in available.target_ranges
    }
    for axis, begin, end, step in required.target_ranges:
        available_begin, available_end, available_step = available_ranges[axis]
        if available_step != 1:
            phase = sympy.simplify(begin - available_begin)  # pyrefly: ignore[unsupported-operation]
            if (
                step % available_step != 0
                or sympy.simplify(sympy.Mod(phase, available_step)) != 0
            ):
                return False
        if (
            sympy.simplify(available_begin) == 0
            and sympy.simplify(  # pyrefly: ignore[unsupported-operation]
                available_end - sympy.Integer(target_domain.axis_counts[axis])  # pyrefly: ignore[unsupported-operation]
            )
            == 0
        ):
            continue
        begin_delta = sympy.simplify(begin - available_begin)  # pyrefly: ignore[unsupported-operation]
        end_delta = sympy.simplify(available_end - end)  # pyrefly: ignore[unsupported-operation]
        if (
            begin_delta.is_nonnegative is not True
            or end_delta.is_nonnegative is not True
        ):
            return False
    return True


def _single_axis_interval(
    begin: sympy.Expr,
    end: sympy.Expr,
    *,
    domain: LogicalDomain,
) -> tuple[int, int, int, int] | None:
    """Recognize ``[stride * axis + offset, ... + width)`` exactly."""
    source_symbols: dict[sympy.Basic, int] = {
        logical_axis_symbol(axis): axis for axis in domain.axis_order
    }
    used_symbols = begin.free_symbols | end.free_symbols
    if len(used_symbols) != 1:
        return None
    (symbol,) = used_symbols
    axis = source_symbols.get(symbol)
    if axis is None:
        return None
    expanded_begin = sympy.expand(begin)
    stride_expression = expanded_begin.coeff(symbol)
    offset_expression = sympy.simplify(expanded_begin - stride_expression * symbol)
    width_expression = sympy.simplify(end - begin)  # pyrefly: ignore[unsupported-operation]
    if (
        stride_expression.free_symbols
        or offset_expression.free_symbols
        or width_expression.free_symbols
        or stride_expression.is_integer is not True
        or offset_expression.is_integer is not True
        or width_expression.is_integer is not True
    ):
        return None
    stride = int(stride_expression)
    offset = int(offset_expression)
    width = int(width_expression)
    if stride <= 0 or width <= 0:
        return None
    return axis, stride, offset, width


def _mixed_radix_interval_inverse(
    piece: _LogicalRelationPiece,
    *,
    source_domain: LogicalDomain,
    target_domain: LogicalDomain,
) -> tuple[_LogicalRelationPiece, ...] | None:
    """Invert a dense inner mixed-radix interval into a point relation.

    This handles relations such as ``[256 * slot + 16 * tile, ... + 16)``.
    Axes whose strides form a dense chain beginning at the interval width are
    decoded symbolically; any remaining outer axes are represented by one
    compact piece per coordinate. The transformation is exact and does not
    enumerate either relation domain.
    """
    if (
        source_domain.kind != "scope"
        or target_domain.kind != "scope"
        or len(source_domain.axis_order) <= 1
        or len(target_domain.axis_order) != 1
        or len(piece.target_ranges) != 1
        or len(piece.target_ranges[0][1].free_symbols) <= 1
    ):
        return None
    target_axis, begin, end, step = piece.target_ranges[0]
    if step != 1:
        return None
    width_expression = sympy.simplify(end - begin)  # pyrefly: ignore[unsupported-operation]
    if width_expression.free_symbols or width_expression.is_integer is not True:
        return None
    width = int(width_expression)
    if width <= 0:
        return None

    source_symbols = {
        logical_axis_symbol(axis): axis for axis in source_domain.axis_order
    }
    expanded_begin = sympy.expand(begin)
    coefficients: dict[int, int] = {}
    remainder = expanded_begin
    for symbol, axis in source_symbols.items():
        coefficient = sympy.simplify(expanded_begin.coeff(symbol))
        if coefficient.free_symbols or coefficient.is_integer is not True:
            return None
        value = int(coefficient)
        if value:
            if value < 0:
                return None
            coefficients[axis] = value
            remainder = sympy.simplify(remainder - coefficient * symbol)
    if remainder.free_symbols or remainder.is_integer is not True:
        return None
    constant = int(remainder)

    bounds = {
        axis: (source_begin, source_end, source_step)
        for axis, source_begin, source_end, source_step in piece.source_bounds_items
    }
    if set(bounds) != set(source_domain.axis_order) or any(
        source_step != 1 for _begin, _end, source_step in bounds.values()
    ):
        return None
    if any(
        axis not in coefficients and source_end - source_begin != 1
        for axis, (source_begin, source_end, _step) in bounds.items()
    ):
        return None

    ordered_axes = sorted(coefficients, key=coefficients.__getitem__)
    dense_axes: list[int] = []
    expected_stride = width
    for axis in ordered_axes:
        if coefficients[axis] != expected_stride:
            break
        dense_axes.append(axis)
        source_begin, source_end, _step = bounds[axis]
        expected_stride *= source_end - source_begin
    if not dense_axes:
        return None
    outer_axes = tuple(axis for axis in source_domain.axis_order if axis not in dense_axes)
    if any(
        0 < coefficients.get(axis, 0) < expected_stride for axis in outer_axes
    ):
        return None

    target_symbol = logical_axis_symbol(target_axis)
    result: list[_LogicalRelationPiece] = []
    outer_ranges = tuple(
        range(bounds[axis][0], bounds[axis][1]) for axis in outer_axes
    )
    for outer_coordinates in itertools.product(*outer_ranges):
        coordinate_by_axis = dict(zip(outer_axes, outer_coordinates, strict=True))
        interval_begin = constant + sum(
            coefficients.get(axis, 0) * coordinate_by_axis[axis]
            for axis in outer_axes
        )
        interval_begin += sum(
            coefficients[axis] * bounds[axis][0] for axis in dense_axes
        )
        interval_end = interval_begin + expected_stride
        if interval_begin < 0 or interval_end > target_domain.axis_counts[target_axis]:
            return None

        target_ranges: list[tuple[int, sympy.Expr, sympy.Expr, int]] = []
        for axis in source_domain.axis_order:
            if axis in coordinate_by_axis:
                value: sympy.Expr = sympy.Integer(coordinate_by_axis[axis])
            else:
                source_begin, source_end, _step = bounds[axis]
                count = source_end - source_begin
                quotient = sympy.floor(
                    (target_symbol - interval_begin) / coefficients[axis]  # pyrefly: ignore[bad-argument-type, unsupported-operation]
                )
                value = sympy.Integer(source_begin) + sympy.Mod(quotient, count)
            target_ranges.append((axis, value, value + 1, 1))  # pyrefly: ignore[unsupported-operation]
        result.append(
            _LogicalRelationPiece(
                source_bounds_items=((target_axis, interval_begin, interval_end, 1),),
                target_ranges=tuple(target_ranges),
            )
        )
    return tuple(result)


def _compact_mixed_radix_point_map(
    relation: LogicalRelation,
) -> LogicalRelation | None:
    """Collapse a piecewise one-dimensional point map to quotient/remainder.

    Mixed-radix interval inversion can naturally produce one piece per outer
    coordinate. When their union is a total function, recognize the equivalent
    ``floor(index / stride) % radix`` coordinates so code generation remains
    constant-sized.
    """
    if (
        len(relation.pieces) <= 1
        or len(relation.source_domain.axis_order) != 1
        or len(relation.target_domain.axis_order) <= 1
    ):
        return None
    canonical = relation.canonical_single_valued()
    if canonical is None or not canonical.is_total_function():
        return None
    source_axis = relation.source_domain.axis_order[0]
    source_symbol = logical_axis_symbol(source_axis)
    source_size = relation.source_domain.size
    values_by_axis: dict[int, list[int]] = {
        axis: [] for axis in relation.target_domain.axis_order
    }
    for source in range(source_size):
        targets = relation.target_coordinates({source_axis: source})
        if len(targets) != 1:
            return None
        coordinates = next(iter(targets))
        for axis, value in zip(
            relation.target_domain.axis_order,
            coordinates,
            strict=True,
        ):
            values_by_axis[axis].append(value)

    expressions: list[sympy.Expr] = []
    for axis in relation.target_domain.axis_order:
        values = values_by_axis[axis]
        if all(value == values[0] for value in values):
            expressions.append(sympy.Integer(values[0]))
            continue
        first_change = next(
            (index for index, value in enumerate(values[1:], start=1) if value != values[0]),
            None,
        )
        if first_change is None:
            return None
        radix = relation.target_domain.axis_counts[axis]
        expression = sympy.Mod(
            sympy.floor(source_symbol / first_change) + values[0],  # pyrefly: ignore[bad-argument-type, unsupported-operation]
            radix,
        )
        if any(
            relation._evaluate(expression, {source_axis: source}) != value
            for source, value in enumerate(values)
        ):
            return None
        expressions.append(expression)
    return LogicalRelation.point_map(
        relation.source_domain,
        relation.target_domain,
        (
            (
                (
                    (
                        source_axis,
                        0,
                        relation.source_domain.axis_counts[source_axis],
                        1,
                    ),
                ),
                tuple(expressions),
            ),
        ),
    )


def _single_axis_floor_point(
    begin: sympy.Expr,
    end: sympy.Expr,
    *,
    domain: LogicalDomain,
) -> tuple[int, int, int, int, int] | None:
    """Recognize ``floor((a * axis + b) / d) + c`` point mappings."""
    if end != begin + 1 and sympy.simplify(end - begin) != 1:  # pyrefly: ignore[unsupported-operation]
        return None
    source_symbols: dict[sympy.Basic, int] = {
        logical_axis_symbol(axis): axis for axis in domain.axis_order
    }
    used_symbols = begin.free_symbols
    if len(used_symbols) != 1:
        return None
    (symbol,) = used_symbols
    axis = source_symbols.get(symbol)
    if axis is None:
        return None

    floor_terms = tuple(
        term for term in sympy.Add.make_args(begin) if term.func == sympy.floor
    )
    if len(floor_terms) != 1:
        return None
    (floor_term,) = floor_terms
    output_offset_expression = sympy.simplify(begin - floor_term)  # pyrefly: ignore[unsupported-operation]
    if (
        output_offset_expression.free_symbols
        or output_offset_expression.is_integer is not True
    ):
        return None
    numerator, denominator = sympy.fraction(sympy.together(floor_term.args[0]))
    numerator = sympy.expand(numerator)
    numerator_stride_expression = numerator.coeff(symbol)
    numerator_offset_expression = sympy.simplify(
        numerator - numerator_stride_expression * symbol
    )
    if (
        denominator.free_symbols
        or numerator_stride_expression.free_symbols
        or numerator_offset_expression.free_symbols
        or denominator.is_integer is not True  # pyrefly: ignore[missing-attribute]
        or numerator_stride_expression.is_integer is not True
        or numerator_offset_expression.is_integer is not True
    ):
        return None
    divisor = int(denominator)
    numerator_stride = int(numerator_stride_expression)
    numerator_offset = int(numerator_offset_expression)
    output_offset = int(output_offset_expression)
    if divisor <= 0 or numerator_stride <= 0:
        return None
    return (
        axis,
        numerator_stride,
        numerator_offset,
        divisor,
        output_offset,
    )


def _ceil_div(numerator: int, denominator: int) -> int:
    return -((-numerator) // denominator)


def _point_expression_preimage(
    expression: sympy.Expr,
    *,
    lower: int,
    upper: int,
    domain: LogicalDomain,
) -> tuple[int, int, int] | bool | None:
    """Invert one point expression over a constant half-open target interval."""
    if not expression.free_symbols:
        if expression.is_integer is not True:  # pyrefly: ignore[missing-attribute]
            return None
        return lower <= int(expression) < upper
    affine = _single_axis_interval(
        expression,
        expression + 1,  # pyrefly: ignore[unsupported-operation]
        domain=domain,
    )
    if affine is not None:
        axis, stride, offset, _width = affine
        return (
            axis,
            _ceil_div(lower - offset, stride),
            _ceil_div(upper - offset, stride),
        )
    floor_point = _single_axis_floor_point(
        expression,
        expression + 1,  # pyrefly: ignore[unsupported-operation]
        domain=domain,
    )
    if floor_point is None:
        return None
    axis, numerator_stride, numerator_offset, divisor, output_offset = floor_point
    return (
        axis,
        _ceil_div(
            divisor * (lower - output_offset) - numerator_offset,
            numerator_stride,
        ),
        _ceil_div(
            divisor * (upper - output_offset) - numerator_offset,
            numerator_stride,
        ),
    )


def _substitute_composed_expression(
    expression: sympy.Expr,
    *,
    substitutions: dict[sympy.Basic, sympy.Expr],
    source_domain: LogicalDomain,
    source_bounds: tuple[tuple[int, int, int, int], ...],
) -> sympy.Expr:
    """Substitute a point map, simplifying only bounded piecewise operators."""
    bounds = _logical_expression_bounds(
        expression,
        domain=source_domain,
        source_bounds=source_bounds,
        symbol_substitutions=substitutions,
    )
    if bounds is not None and bounds[0] == bounds[1]:
        return bounds[0]
    result = expression.xreplace(substitutions)
    if result.has(sympy.Mod, sympy.Min, sympy.Max):
        return _simplify_logical_expression(
            result,
            domain=source_domain,
            source_bounds=source_bounds,
        )
    return result


def _compose_point_relations(
    first: LogicalRelation,
    following: LogicalRelation,
) -> LogicalRelation | None:
    """Compose point-valued relation pieces by exact box preimage.

    Composition is distributive over the union of relation pieces, so it does
    not require globally canonicalizing either relation.  Avoiding that
    partitioning is important for compact physical traversals, whose pieces
    are already disjoint by construction but can number in the thousands.
    """
    if any(
        step != 1
        or (
            end != begin + 1  # pyrefly: ignore[unsupported-operation]
            and sympy.simplify(end - begin)  # pyrefly: ignore[unsupported-operation]
            != 1
        )
        for piece in first.pieces
        for _axis, begin, end, step in piece.target_ranges
    ):
        return None
    pieces: list[_LogicalRelationPiece] = []
    for first_piece in first.pieces:
        first_targets = {
            axis: begin for axis, begin, _end, _step in first_piece.target_ranges
        }
        substitutions: dict[sympy.Basic, sympy.Expr] = {
            logical_axis_symbol(axis): expression
            for axis, expression in first_targets.items()
        }
        for following_piece in following.pieces:
            bounds = {
                axis: [begin, end, step]
                for axis, begin, end, step in first_piece.source_bounds_items
            }
            valid = True
            for axis, begin, end, step in following_piece.source_bounds_items:
                if step != 1:
                    return None
                if begin == 0 and end == following.source_domain.axis_counts[axis]:
                    continue
                expression_bounds = _logical_expression_bounds(
                    first_targets[axis],
                    domain=first.source_domain,
                    source_bounds=first_piece.source_bounds_items,
                )
                if expression_bounds is not None:
                    minimum, maximum = expression_bounds
                    if minimum >= begin and maximum < end:  # pyrefly: ignore[unsupported-operation]
                        continue
                    if maximum < begin or minimum >= end:  # pyrefly: ignore[unsupported-operation]
                        valid = False
                        break
                preimage = _point_expression_preimage(
                    first_targets[axis],
                    lower=begin,
                    upper=end,
                    domain=first.source_domain,
                )
                if preimage is None:
                    return None
                if isinstance(preimage, bool):
                    if not preimage:
                        valid = False
                        break
                    continue
                source_axis, preimage_begin, preimage_end = preimage
                source_begin, source_end, source_step = bounds[source_axis]
                restricted_begin = max(source_begin, preimage_begin)
                restricted_end = min(source_end, preimage_end)
                restricted_begin += (source_begin - restricted_begin) % source_step
                if restricted_begin >= restricted_end:
                    valid = False
                    break
                bounds[source_axis] = [
                    restricted_begin,
                    restricted_end,
                    source_step,
                ]
            if not valid:
                continue
            source_bounds = tuple(
                (
                    axis,
                    bounds[axis][0],
                    bounds[axis][1],
                    bounds[axis][2],
                )
                for axis in first.source_domain.axis_order
            )

            pieces.append(
                _LogicalRelationPiece(
                    source_bounds_items=source_bounds,
                    target_ranges=tuple(
                        (
                            axis,
                            _substitute_composed_expression(
                                begin,
                                substitutions=substitutions,
                                source_domain=first.source_domain,
                                source_bounds=source_bounds,
                            ),
                            _substitute_composed_expression(
                                end,
                                substitutions=substitutions,
                                source_domain=first.source_domain,
                                source_bounds=source_bounds,
                            ),
                            step,
                        )
                        for axis, begin, end, step in following_piece.target_ranges
                    ),
                )
            )
    return LogicalRelation(
        source_domain=first.source_domain,
        target_domain=following.target_domain,
        pieces=tuple(dict.fromkeys(pieces)),
    )


def physical_traversal_relation(
    logical_domain: LogicalDomain,
    physical_axis_order: tuple[int, ...],
    *,
    l2_group_size: int | None = None,
) -> LogicalRelation:
    """Map a configured physical PID traversal to logical task coordinates."""
    if set(logical_domain.axis_order) != set(physical_axis_order):
        raise ValueError("physical traversal must permute the logical axes")
    counts = logical_domain.axis_counts
    if l2_group_size is None or len(physical_axis_order) < 2:
        source_domain = LogicalDomain(
            axis_order=physical_axis_order,
            axis_counts_items=tuple(
                (axis, counts[axis]) for axis in physical_axis_order
            ),
            kind="worker",
            identity=logical_domain.identity,
        )
        return LogicalRelation.point_map(
            source_domain,
            logical_domain,
            (
                (
                    tuple((axis, 0, counts[axis], 1) for axis in physical_axis_order),
                    tuple(
                        logical_axis_symbol(axis) for axis in logical_domain.axis_order
                    ),
                ),
            ),
        )

    first_axis, second_axis, *outer_axes = physical_axis_order
    first_count = counts[first_axis]
    second_count = counts[second_axis]
    inner_axis = min(logical_domain.axis_order, default=0) - 1
    while inner_axis in logical_domain.axis_order:
        inner_axis -= 1
    source_domain = LogicalDomain(
        axis_order=(inner_axis, *outer_axes),
        axis_counts_items=(
            (inner_axis, first_count * second_count),
            *((axis, counts[axis]) for axis in outer_axes),
        ),
        kind="worker",
        identity=logical_domain.identity,
    )
    inner = logical_axis_symbol(inner_axis)
    pieces: list[
        tuple[
            tuple[tuple[int, int, int, int], ...],
            tuple[sympy.Expr, ...],
        ]
    ] = []
    for first_in_group in range(0, first_count, l2_group_size):
        actual_group_size = min(first_count - first_in_group, l2_group_size)
        group = first_in_group // l2_group_size
        group_begin = group * l2_group_size * second_count
        for second in range(second_count):
            begin = group_begin + second * actual_group_size
            expressions = {
                first_axis: inner - begin + first_in_group,  # pyrefly: ignore[unsupported-operation]
                second_axis: sympy.Integer(second),
                **{axis: logical_axis_symbol(axis) for axis in outer_axes},
            }
            pieces.append(
                (
                    (
                        (inner_axis, begin, begin + actual_group_size, 1),
                        *((axis, 0, counts[axis], 1) for axis in outer_axes),
                    ),
                    tuple(expressions[axis] for axis in logical_domain.axis_order),
                )
            )
    return LogicalRelation.point_map(source_domain, logical_domain, tuple(pieces))


@dataclasses.dataclass(frozen=True)
class ExecutionScope:
    """One reachable DeviceIR callsite in an outer task's execution strand.

    ``graph_id`` identifies the called body, while ``callsite_path`` identifies
    this particular invocation of that body.  Nested loop actions inherit the
    worker assigned to their owning root task; this record describes their
    logical coordinate domain and program-order identity, not an independently
    movable scheduling unit.
    """

    scope_id: int
    root: int
    graph_id: int
    callsite_path: tuple[tuple[int, int], ...]
    parent_scope_id: int | None
    kind: Literal["root", "loop", "branch", "while_condition", "while_body"]
    local_axis_order: tuple[int, ...]
    logical_axis_order: tuple[int, ...]
    guaranteed: bool
    segmentable: bool

    @property
    def is_root(self) -> bool:
        return self.kind == "root"


def _validate_execution_scopes(
    device_ir: DeviceIR,
    execution_scopes: tuple[ExecutionScope, ...],
) -> None:
    if len(device_ir.root_ids) != len(device_ir.task_families):
        raise ValueError("DeviceIR roots and task families must have equal length")
    if tuple(scope.scope_id for scope in execution_scopes) != tuple(
        range(len(execution_scopes))
    ):
        raise ValueError("execution scope IDs must be contiguous and ordered")

    root_scope_ids: dict[int, int] = {}
    for scope in execution_scopes:
        if not 0 <= scope.graph_id < len(device_ir.graphs):
            raise ValueError(
                f"execution scope {scope.scope_id} references unknown graph "
                f"{scope.graph_id}"
            )
        if not 0 <= scope.root < len(device_ir.root_ids):
            raise ValueError(
                f"execution scope {scope.scope_id} references unknown root "
                f"{scope.root}"
            )
        if scope.parent_scope_id is None:
            if not scope.is_root or scope.callsite_path:
                raise ValueError(
                    f"execution scope {scope.scope_id} is not a valid root scope"
                )
            if device_ir.root_ids[scope.root] != scope.graph_id:
                raise ValueError(
                    f"execution scope {scope.scope_id} does not match root "
                    f"{scope.root}'s graph"
                )
            if scope.root in root_scope_ids:
                raise ValueError(f"root {scope.root} has multiple execution scopes")
            root_scope_ids[scope.root] = scope.scope_id
            continue
        if not 0 <= scope.parent_scope_id < scope.scope_id:
            raise ValueError(
                f"execution scope {scope.scope_id} has invalid parent "
                f"{scope.parent_scope_id}"
            )
        parent = execution_scopes[scope.parent_scope_id]
        if parent.root != scope.root:
            raise ValueError(
                f"execution scope {scope.scope_id} and its parent have "
                "different roots"
            )

    if frozenset(root_scope_ids) != frozenset(range(len(device_ir.root_ids))):
        raise ValueError("each DeviceIR root must have exactly one execution scope")


def build_execution_scopes(device_ir: DeviceIR) -> tuple[ExecutionScope, ...]:
    """Build the reachable DeviceIR callsite tree used by dependency actions.

    A DeviceIR graph body is not itself a unique execution point: one body may
    be referenced by several callsites, and control-flow graphs have different
    execution guarantees from ordinary device loops.  Paths therefore use the
    lexical call node and child argument slot within each owning root.
    """
    from ..language import _tracing_ops
    from .device_ir import ForLoopGraphInfo

    if len(device_ir.root_ids) != len(device_ir.task_families):
        raise ValueError("DeviceIR roots and task families must have equal length")

    scopes: list[ExecutionScope] = []

    def add_scope(
        *,
        root: int,
        graph_id: int,
        callsite_path: tuple[tuple[int, int], ...],
        parent_scope_id: int | None,
        kind: Literal["root", "loop", "branch", "while_condition", "while_body"],
        local_axis_order: tuple[int, ...],
        logical_axis_order: tuple[int, ...],
        guaranteed: bool,
        segmentable: bool,
    ) -> int:
        scope_id = len(scopes)
        scopes.append(
            ExecutionScope(
                scope_id=scope_id,
                root=root,
                graph_id=graph_id,
                callsite_path=callsite_path,
                parent_scope_id=parent_scope_id,
                kind=kind,
                local_axis_order=local_axis_order,
                logical_axis_order=logical_axis_order,
                guaranteed=guaranteed,
                segmentable=segmentable,
            )
        )
        return scope_id

    def walk(
        *,
        root: int,
        scope_id: int,
        ancestor_graph_ids: frozenset[int],
    ) -> None:
        scope = scopes[scope_id]
        graph = device_ir.graphs[scope.graph_id].graph
        for node_index, node in enumerate(graph.nodes):
            if node.op != "call_function":
                continue

            child_specs: list[
                tuple[
                    int,
                    int,
                    Literal["loop", "branch", "while_condition", "while_body"],
                    bool,
                ]
            ] = []
            if (
                _tracing_ops.is_for_loop_target(node.target)
                and node.args
                and isinstance(node.args[0], int)
            ):
                child_specs.append((0, node.args[0], "loop", scope.guaranteed))
            elif node.target is _tracing_ops._if and len(node.args) >= 3:
                if isinstance(node.args[1], int):
                    child_specs.append((1, node.args[1], "branch", False))
                if isinstance(node.args[2], int):
                    child_specs.append((2, node.args[2], "branch", False))
            elif node.target is _tracing_ops._while_loop and len(node.args) >= 2:
                if isinstance(node.args[0], int):
                    child_specs.append((0, node.args[0], "while_condition", False))
                if isinstance(node.args[1], int):
                    child_specs.append((1, node.args[1], "while_body", False))

            callsite_scope_ids: list[tuple[int, int]] = []
            for child_slot, child_graph_id, kind, guaranteed in child_specs:
                if not 0 <= child_graph_id < len(device_ir.graphs):
                    continue
                child_info = device_ir.graphs[child_graph_id]
                local_axes = (
                    tuple(child_info.block_ids)
                    if kind == "loop" and isinstance(child_info, ForLoopGraphInfo)
                    else ()
                )
                axes_are_unique = not set(local_axes).intersection(
                    scope.logical_axis_order
                )
                child_scope_id = add_scope(
                    root=root,
                    graph_id=child_graph_id,
                    callsite_path=(*scope.callsite_path, (node_index, child_slot)),
                    parent_scope_id=scope_id,
                    kind=kind,
                    local_axis_order=local_axes,
                    logical_axis_order=(*scope.logical_axis_order, *local_axes),
                    guaranteed=guaranteed,
                    segmentable=(
                        kind == "loop"
                        and guaranteed
                        and axes_are_unique
                        and not any(
                            axis in device_ir.noncanonical_task_origin_block_ids
                            for axis in local_axes
                        )
                    ),
                )
                callsite_scope_ids.append((child_slot, child_scope_id))
                if child_graph_id not in ancestor_graph_ids:
                    walk(
                        root=root,
                        scope_id=child_scope_id,
                        ancestor_graph_ids=ancestor_graph_ids
                        | frozenset((child_graph_id,)),
                    )
            if callsite_scope_ids:
                node.meta[TILE_DEPENDENCY_SCOPE_IDS_META] = tuple(callsite_scope_ids)

    for root, graph_id in enumerate(device_ir.root_ids):
        family = device_ir.task_families[root]
        root_scope_id = add_scope(
            root=root,
            graph_id=graph_id,
            callsite_path=(),
            parent_scope_id=None,
            kind="root",
            local_axis_order=family.logical_axis_order,
            logical_axis_order=family.logical_axis_order,
            guaranteed=True,
            segmentable=False,
        )
        walk(
            root=root,
            scope_id=root_scope_id,
            ancestor_graph_ids=frozenset((graph_id,)),
        )
    result = tuple(scopes)
    _validate_execution_scopes(device_ir, result)
    return result


@dataclasses.dataclass(frozen=True)
class TileAccess:
    """The memory facts needed to prove a cross-root readiness relation."""

    access_id: int
    memory_op_index: int
    atomic_op_index: int
    graph_id: int
    scope_id: int | None
    root: int
    allocation_id: int
    kind: Literal["load", "store"]
    tensor_name: str | None
    tensor_shape: tuple[int, ...]
    tensor_strides: tuple[int, ...]
    storage_offset: int
    subscript_dims: tuple[int, ...]
    subscript_affine_block_ids: tuple[int | None, ...]
    subscript_index_scales: tuple[int, ...]
    subscript_offsets: tuple[int | None, ...]
    subscript_is_scalar: tuple[bool, ...]
    has_explicit_mask: bool
    layout_is_static: bool
    subscript_is_full_slice: tuple[bool, ...] = ()
    subscript_static_extents: tuple[int | None, ...] = ()
    is_atomic: bool = False
    graph_node_index: int = -1


@dataclasses.dataclass(frozen=True)
class AllocationRegion:
    """A conservative region in allocation-address coordinates.

    ``address_interval`` is always a may-access hull.  When
    ``is_exact_contiguous`` is true, it is also the exact set of addresses.
    ``coordinate_bounds`` retain an exact rectangular view when one is known;
    they let equal-layout views prove disjointness or coverage without turning
    the dependency pass into a general symbolic set solver.
    """

    address_interval: tuple[int, int] | None
    is_exact_contiguous: bool
    layout: tuple[tuple[int, ...], tuple[int, ...], int] | None = None
    coordinate_bounds: tuple[tuple[int, int], ...] = ()
    coordinates_are_exact: bool = False


@dataclasses.dataclass(frozen=True)
class AccessDependency:
    """One source-ordered memory hazard over an allocation region."""

    kind: TileDependencyKind
    producer_access_id: int
    consumer_access_id: int
    region: AllocationRegion
    dependency_id: int = -1


@dataclasses.dataclass(frozen=True)
class TileDependencyRelation:
    """One symbolic dependency between execution-scope instance domains.

    ``relation`` maps each consumer instance to the producer instances it must
    observe.  A missing relation means that dependency scheduling must lift to
    an enclosing scope or family completion.
    """

    kind: TileDependencyKind
    dependency_id: int
    producer_access_id: int
    consumer_access_id: int
    producer_root: int
    consumer_root: int
    producer_scope_id: int | None
    consumer_scope_id: int | None
    relation: LogicalRelation | None


@dataclasses.dataclass(frozen=True)
class TileDependency:
    """One allocation hazard between two source-ordered root families."""

    producer_root: int
    consumer_root: int
    allocation_id: int
    tensor_names: frozenset[str]
    kinds: frozenset[TileDependencyKind]
    producer_accesses: tuple[TileAccess, ...]
    consumer_accesses: tuple[TileAccess, ...]
    access_dependencies: tuple[AccessDependency, ...]

    @property
    def is_raw_only(self) -> bool:
        return self.kinds == frozenset((TileDependencyKind.READ_AFTER_WRITE,))


@dataclasses.dataclass(frozen=True)
class TileDependencyGraph:
    """Allocation-derived dependencies and DeviceIR execution scopes."""

    task_families: tuple[TaskFamily, ...]
    accesses: tuple[TileAccess, ...]
    edges: tuple[TileDependency, ...]
    execution_scopes: tuple[ExecutionScope, ...] = ()
    scope_ids_by_access: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if tuple(scope.scope_id for scope in self.execution_scopes) != tuple(
            range(len(self.execution_scopes))
        ):
            raise ValueError("execution scope IDs must be contiguous")
        if any(
            not 0 <= scope_id < len(self.execution_scopes)
            for scope_ids in self.scope_ids_by_access
            for scope_id in scope_ids
        ):
            raise ValueError("access references an unknown execution scope")

    def edges_between(
        self,
        producer_root: int,
        consumer_root: int,
    ) -> tuple[TileDependency, ...]:
        return tuple(
            edge
            for edge in self.edges
            if edge.producer_root == producer_root
            and edge.consumer_root == consumer_root
        )

    def scopes_for_access(self, access_id: int) -> tuple[ExecutionScope, ...]:
        if not 0 <= access_id < len(self.scope_ids_by_access):
            return ()
        return tuple(
            self.execution_scopes[scope_id]
            for scope_id in self.scope_ids_by_access[access_id]
        )

    def dependency_points(
        self,
        dependency: AccessDependency,
    ) -> frozenset[DependencyPoint]:
        """Return every producer/consumer callsite obligation for one hazard."""

        def access_scope_ids(access_id: int) -> tuple[int | None, ...]:
            if not 0 <= access_id < len(self.scope_ids_by_access):
                return (None,)
            scope_ids = self.scope_ids_by_access[access_id]
            return scope_ids or (None,)

        return frozenset(
            (
                dependency.dependency_id,
                producer_scope_id,
                consumer_scope_id,
            )
            for producer_scope_id in access_scope_ids(dependency.producer_access_id)
            for consumer_scope_id in access_scope_ids(dependency.consumer_access_id)
        )


@dataclasses.dataclass(frozen=True)
class _ReachingAccess:
    root: int
    access: TileAccess
    region: AllocationRegion


def _access_region(
    access: TileAccess,
    task_family: TaskFamily,
) -> AllocationRegion:
    """Conservatively summarize one root's union of an access.

    Canonical non-scalar tile axes cover their source-level iteration extent
    independently of the configured block size. Unknown, scalar, masked, or
    indirect dimensions retain a may-access bound but are not allowed to kill
    an earlier reaching definition.
    """
    if not access.layout_is_static:
        return AllocationRegion(None, False)
    shape = access.tensor_shape
    strides = access.tensor_strides
    if len(shape) != len(strides) or any(size < 0 for size in shape):
        return AllocationRegion(None, False)

    position_by_dim: dict[int, int] = {}
    for position, tensor_dim in enumerate(access.subscript_dims):
        if tensor_dim in position_by_dim or not 0 <= tensor_dim < len(shape):
            return AllocationRegion(None, False)
        position_by_dim[tensor_dim] = position

    bounds: list[tuple[int, int]] = []
    exact_dimensions: list[bool] = []
    for tensor_dim, size in enumerate(shape):
        position = position_by_dim.get(tensor_dim)
        if position is None:
            bounds.append((0, size))
            exact_dimensions.append(not access.has_explicit_mask)
            continue
        if position >= len(access.subscript_is_full_slice):
            return AllocationRegion(None, False)
        if access.subscript_is_full_slice[position]:
            bounds.append((0, size))
            exact_dimensions.append(not access.has_explicit_mask)
            continue
        if (
            position >= len(access.subscript_affine_block_ids)
            or position >= len(access.subscript_index_scales)
            or position >= len(access.subscript_offsets)
            or position >= len(access.subscript_is_scalar)
        ):
            return AllocationRegion(None, False)
        block_id = access.subscript_affine_block_ids[position]
        offset = access.subscript_offsets[position]
        axis = task_family.axis(block_id) if block_id is not None else None
        symbolic_extent = axis.extent if axis is not None else None
        static_extent = (
            access.subscript_static_extents[position]
            if position < len(access.subscript_static_extents)
            else None
        )
        if (
            axis is None
            and not access.subscript_is_scalar[position]
            and offset is not None
            and static_extent is not None
        ):
            begin = offset if offset >= 0 else size + offset
            end = begin + static_extent
            if 0 <= begin <= end <= size:
                bounds.append((begin, end))
                exact_dimensions.append(not access.has_explicit_mask)
                continue
        if (
            axis is None
            or not axis.canonical_origin
            or not isinstance(symbolic_extent, int | sympy.Integer)
            or symbolic_extent < 0
            or access.subscript_index_scales[position] != 1
            or offset is None
            or access.subscript_is_scalar[position]
        ):
            bounds.append((0, size))
            exact_dimensions.append(False)
            continue
        extent = int(symbolic_extent)
        begin = offset
        end = offset + extent
        if begin < 0 or end > size:
            bounds.append((0, size))
            exact_dimensions.append(False)
            continue
        bounds.append((begin, end))
        exact_dimensions.append(not access.has_explicit_mask)

    return _allocation_region_from_bounds(
        access,
        tuple(bounds),
        tuple(exact_dimensions),
    )


def _access_positions_by_dimension(access: TileAccess) -> dict[int, int] | None:
    result: dict[int, int] = {}
    for position, dimension in enumerate(access.subscript_dims):
        if dimension in result or not 0 <= dimension < len(access.tensor_shape):
            return None
        result[dimension] = position
    return result


def _access_interval_expression(
    access: TileAccess,
    *,
    position: int,
    domain: LogicalDomain,
) -> tuple[sympy.Expr, sympy.Expr] | None:
    if position >= len(access.subscript_is_full_slice):
        return None
    tensor_dimension = access.subscript_dims[position]
    size = access.tensor_shape[tensor_dimension]
    if access.subscript_is_full_slice[position]:
        return sympy.Integer(0), sympy.Integer(size)
    if (
        position >= len(access.subscript_affine_block_ids)
        or position >= len(access.subscript_index_scales)
        or position >= len(access.subscript_offsets)
        or position >= len(access.subscript_is_scalar)
    ):
        return None
    axis = access.subscript_affine_block_ids[position]
    offset = access.subscript_offsets[position]
    if axis is None and access.subscript_is_scalar[position]:
        if offset is None:
            return None
        size = access.tensor_shape[tensor_dimension]
        normalized_offset = offset if offset >= 0 else size + offset
        if not 0 <= normalized_offset < size:
            return None
        begin = sympy.Integer(normalized_offset)
        return begin, begin + 1
    if axis is None:
        static_extent = (
            access.subscript_static_extents[position]
            if position < len(access.subscript_static_extents)
            else None
        )
        if offset is None or static_extent is None:
            return None
        normalized_offset = offset if offset >= 0 else size + offset
        if not 0 <= normalized_offset <= normalized_offset + static_extent <= size:
            return None
        begin = sympy.Integer(normalized_offset)
        return begin, begin + static_extent
    if axis is None or offset is None or axis not in domain.axis_counts:
        return None
    scale = access.subscript_index_scales[position]
    if scale != 1:
        return None
    coordinate: sympy.Expr = (
        sympy.Integer(0) if domain.axis_counts[axis] == 1 else logical_axis_symbol(axis)
    )
    if access.subscript_is_scalar[position]:
        begin = coordinate + offset  # pyrefly: ignore[unsupported-operation]
        return begin, begin + 1
    block_size = domain.block_sizes.get(axis)
    if block_size is None:
        return None
    begin = coordinate * block_size + offset  # pyrefly: ignore[unsupported-operation]
    return begin, begin + block_size


def _symbolic_coordinate_access_relation(
    access: TileAccess,
    *,
    source_domain: LogicalDomain,
    allocation_domain: LogicalDomain,
    tensor_dimensions: tuple[int, ...],
) -> LogicalRelation | None:
    """Map one access scope to its exact allocation-coordinate footprint."""
    if (
        not access.layout_is_static
        or access.has_explicit_mask
        or allocation_domain.kind != "allocation"
        or allocation_domain.identity != access.allocation_id
        or allocation_domain.axis_counts_items
        != tuple(
            (allocation_axis, access.tensor_shape[tensor_dimension])
            for allocation_axis, tensor_dimension in enumerate(tensor_dimensions)
        )
    ):
        return None
    positions = _access_positions_by_dimension(access)
    if positions is None:
        return None
    target_ranges: list[tuple[int, sympy.Expr, sympy.Expr, int]] = []
    for allocation_axis, tensor_dimension in zip(
        allocation_domain.axis_order,
        tensor_dimensions,
        strict=True,
    ):
        position = positions.get(tensor_dimension)
        interval = (
            (
                sympy.Integer(0),
                sympy.Integer(access.tensor_shape[tensor_dimension]),
            )
            if position is None
            else _access_interval_expression(
                access,
                position=position,
                domain=source_domain,
            )
        )
        if interval is None:
            return None
        begin, end = interval
        target_ranges.append((allocation_axis, begin, end, 1))
    return LogicalRelation(
        source_domain=source_domain,
        target_domain=allocation_domain,
        pieces=(
            _LogicalRelationPiece(
                source_bounds_items=tuple(
                    (axis, 0, source_domain.axis_counts[axis], 1)
                    for axis in source_domain.axis_order
                ),
                target_ranges=tuple(target_ranges),
            ),
        ),
    )


def _allocation_storage_size(access: TileAccess) -> int | None:
    if (
        len(access.tensor_shape) != len(access.tensor_strides)
        or access.storage_offset < 0
        or any(size <= 0 for size in access.tensor_shape)
        or any(stride < 0 for stride in access.tensor_strides)
    ):
        return None
    return (
        access.storage_offset
        + 1
        + sum(
            (size - 1) * stride
            for size, stride in zip(
                access.tensor_shape,
                access.tensor_strides,
                strict=True,
            )
        )
    )


def _normalized_coordinate_layout(
    access: TileAccess,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]] | None:
    """Return non-size-one dimensions and their allocation geometry."""
    layout = (access.tensor_shape, access.tensor_strides, access.storage_offset)
    if not _layout_is_injective(layout):
        return None
    dimensions = tuple(
        dimension for dimension, size in enumerate(access.tensor_shape) if size != 1
    )
    return dimensions, tuple(
        (access.tensor_shape[dimension], access.tensor_strides[dimension])
        for dimension in dimensions
    )


def _symbolic_linear_access_relation(
    access: TileAccess,
    *,
    source_domain: LogicalDomain,
    allocation_domain: LogicalDomain,
) -> LogicalRelation | None:
    """Map a provably contiguous view tile to linear allocation addresses."""
    if (
        not access.layout_is_static
        or access.has_explicit_mask
        or allocation_domain.kind != "allocation"
        or allocation_domain.identity != access.allocation_id
        or allocation_domain.axis_order != (_ALLOCATION_ADDRESS_AXIS,)
    ):
        return None
    positions = _access_positions_by_dimension(access)
    if positions is None:
        return None

    intervals: list[tuple[sympy.Expr, sympy.Expr]] = []
    widths: list[int] = []
    for tensor_dimension, size in enumerate(access.tensor_shape):
        position = positions.get(tensor_dimension)
        interval = (
            (sympy.Integer(0), sympy.Integer(size))
            if position is None
            else _access_interval_expression(
                access,
                position=position,
                domain=source_domain,
            )
        )
        if interval is None:
            return None
        begin, end = interval
        width_expression = sympy.simplify(end - begin)  # pyrefly: ignore[unsupported-operation]
        if not isinstance(width_expression, sympy.Integer):
            return None
        width = int(width_expression)
        if width <= 0:
            return None

        if position is not None and not access.subscript_is_full_slice[position]:
            axis = access.subscript_affine_block_ids[position]
            offset = access.subscript_offsets[position]
            if axis is not None:
                if offset is None:
                    return None
                final_end = (
                    (source_domain.axis_counts[axis] - 1)
                    * (1 if access.subscript_is_scalar[position] else width)
                    + offset
                    + width
                )
                if offset < 0 or final_end > size:
                    return None
        intervals.append(interval)
        widths.append(width)

    contiguous_span = 1
    for tensor_dimension in sorted(
        range(len(access.tensor_shape)),
        key=access.tensor_strides.__getitem__,
    ):
        width = widths[tensor_dimension]
        if width == 1:
            continue
        if access.tensor_strides[tensor_dimension] != contiguous_span:
            return None
        contiguous_span *= width

    begin = sympy.Integer(access.storage_offset)
    for (dimension_begin, _dimension_end), stride in zip(
        intervals,
        access.tensor_strides,
        strict=True,
    ):
        begin += dimension_begin * stride  # pyrefly: ignore[unsupported-operation]
    return LogicalRelation(
        source_domain=source_domain,
        target_domain=allocation_domain,
        pieces=(
            _LogicalRelationPiece(
                source_bounds_items=tuple(
                    (axis, 0, source_domain.axis_counts[axis], 1)
                    for axis in source_domain.axis_order
                ),
                target_ranges=(
                    (
                        _ALLOCATION_ADDRESS_AXIS,
                        begin,
                        begin + contiguous_span,  # pyrefly: ignore[unsupported-operation]
                        1,
                    ),
                ),
            ),
        ),
    )


def _symbolic_access_predecessors(
    *,
    producer_access: TileAccess,
    producer_domain: LogicalDomain,
    consumer_access: TileAccess,
    consumer_domain: LogicalDomain,
) -> LogicalRelation | None:
    """Compose two scope-to-allocation maps into exact predecessors."""
    if not producer_access.layout_is_static or not consumer_access.layout_is_static:
        return None
    producer_layout = _normalized_coordinate_layout(producer_access)
    consumer_layout = _normalized_coordinate_layout(consumer_access)
    if (
        producer_layout is not None
        and consumer_layout is not None
        and producer_layout[1] == consumer_layout[1]
        and producer_access.storage_offset == consumer_access.storage_offset
    ):
        producer_dimensions, normalized_layout = producer_layout
        consumer_dimensions, _ = consumer_layout
        coordinate_domain = LogicalDomain(
            axis_order=tuple(range(len(normalized_layout))),
            axis_counts_items=tuple(
                (axis, size) for axis, (size, _stride) in enumerate(normalized_layout)
            ),
            kind="allocation",
            identity=producer_access.allocation_id,
        )
        producer_relation = _symbolic_coordinate_access_relation(
            producer_access,
            source_domain=producer_domain,
            allocation_domain=coordinate_domain,
            tensor_dimensions=producer_dimensions,
        )
        consumer_relation = _symbolic_coordinate_access_relation(
            consumer_access,
            source_domain=consumer_domain,
            allocation_domain=coordinate_domain,
            tensor_dimensions=consumer_dimensions,
        )
        if producer_relation is not None and consumer_relation is not None:
            relation = producer_relation.overlapping_sources(consumer_relation)
            if relation is not None:
                return relation

    producer_storage_size = _allocation_storage_size(producer_access)
    consumer_storage_size = _allocation_storage_size(consumer_access)
    if producer_storage_size is None or consumer_storage_size is None:
        return None
    linear_domain = LogicalDomain(
        axis_order=(_ALLOCATION_ADDRESS_AXIS,),
        axis_counts_items=(
            (
                _ALLOCATION_ADDRESS_AXIS,
                max(producer_storage_size, consumer_storage_size),
            ),
        ),
        kind="allocation",
        identity=producer_access.allocation_id,
    )
    producer_relation = _symbolic_linear_access_relation(
        producer_access,
        source_domain=producer_domain,
        allocation_domain=linear_domain,
    )
    consumer_relation = _symbolic_linear_access_relation(
        consumer_access,
        source_domain=consumer_domain,
        allocation_domain=linear_domain,
    )
    if producer_relation is None or consumer_relation is None:
        return None
    return producer_relation.overlapping_sources(consumer_relation)


def _logical_domain_for_axes(
    axis_order: tuple[int, ...],
    *,
    axis_geometry: dict[int, tuple[int, int]],
    identity: int,
) -> LogicalDomain | None:
    geometry = tuple(axis_geometry.get(axis) for axis in axis_order)
    if any(item is None for item in geometry):
        return None
    concrete_geometry = tuple(item for item in geometry if item is not None)
    if any(count <= 0 or block_size <= 0 for count, block_size in concrete_geometry):
        return None
    return LogicalDomain(
        axis_order=axis_order,
        axis_counts_items=tuple(
            (axis, concrete_geometry[index][0]) for index, axis in enumerate(axis_order)
        ),
        block_sizes_items=tuple(
            (axis, concrete_geometry[index][1]) for index, axis in enumerate(axis_order)
        ),
        kind="scope",
        identity=identity,
    )


def instantiate_scope_domains(
    dependency_graph: TileDependencyGraph,
    *,
    axis_geometry: dict[int, tuple[int, int]],
) -> tuple[LogicalDomain | None, ...]:
    """Bind every DeviceIR execution scope to the selected tile geometry.

    The tuple is indexed by ``ExecutionScope.scope_id``.  No traversal is
    attached: these domains describe semantic coordinates, while physical PID
    and within-strand order remain scheduler/lowering choices.
    """
    return tuple(
        _logical_domain_for_axes(
            scope.logical_axis_order,
            axis_geometry=axis_geometry,
            identity=scope.scope_id,
        )
        for scope in dependency_graph.execution_scopes
    )


def instantiate_root_domains(
    dependency_graph: TileDependencyGraph,
    *,
    axis_geometry: dict[int, tuple[int, int]],
) -> tuple[LogicalDomain | None, ...]:
    """Bind top-level task families without attaching a physical traversal."""
    root_scope_ids = {
        scope.root: scope.scope_id
        for scope in dependency_graph.execution_scopes
        if scope.is_root
    }
    return tuple(
        _logical_domain_for_axes(
            family.logical_axis_order,
            axis_geometry=axis_geometry,
            identity=root_scope_ids.get(root, root),
        )
        for root, family in enumerate(dependency_graph.task_families)
    )


def instantiate_symbolic_dependencies(
    dependency_graph: TileDependencyGraph,
    *,
    axis_geometry: dict[int, tuple[int, int]],
) -> tuple[TileDependencyRelation, ...]:
    """Instantiate scope dependencies without enumerating task instances.

    Unsupported access geometry returns ``relation=None`` so the caller can
    monotonically retain family completion.
    """
    scope_domains = instantiate_scope_domains(
        dependency_graph,
        axis_geometry=axis_geometry,
    )
    scope_by_id = {scope.scope_id: scope for scope in dependency_graph.execution_scopes}
    root_domains = instantiate_root_domains(
        dependency_graph,
        axis_geometry=axis_geometry,
    )
    access_by_id = {access.access_id: access for access in dependency_graph.accesses}

    def endpoints(
        access: TileAccess,
    ) -> tuple[tuple[int | None, LogicalDomain], ...]:
        scope_ids = (
            dependency_graph.scope_ids_by_access[access.access_id]
            if 0 <= access.access_id < len(dependency_graph.scope_ids_by_access)
            else ()
        )
        if not scope_ids:
            root_domain = root_domains[access.root]
            return () if root_domain is None else ((None, root_domain),)
        result: list[tuple[int | None, LogicalDomain]] = []
        for scope_id in scope_ids:
            scope = scope_by_id[scope_id]
            domain = scope_domains[scope_id]
            if domain is not None and scope.guaranteed:
                result.append((scope_id, domain))
        return tuple(result)

    result: list[TileDependencyRelation] = []
    for edge in dependency_graph.edges:
        axes_have_canonical_origins = all(
            axis.canonical_origin
            for root in (edge.producer_root, edge.consumer_root)
            for axis in dependency_graph.task_families[root].axes
        )
        for access_dependency in edge.access_dependencies:
            producer_access = access_by_id[access_dependency.producer_access_id]
            consumer_access = access_by_id[access_dependency.consumer_access_id]
            producer_endpoints = endpoints(producer_access)
            consumer_endpoints = endpoints(consumer_access)
            if not producer_endpoints or not consumer_endpoints:
                result.append(
                    TileDependencyRelation(
                        kind=access_dependency.kind,
                        dependency_id=access_dependency.dependency_id,
                        producer_access_id=producer_access.access_id,
                        consumer_access_id=consumer_access.access_id,
                        producer_root=edge.producer_root,
                        consumer_root=edge.consumer_root,
                        producer_scope_id=None,
                        consumer_scope_id=None,
                        relation=None,
                    )
                )
                continue
            for producer_scope_id, producer_domain in producer_endpoints:
                for consumer_scope_id, consumer_domain in consumer_endpoints:
                    result.append(
                        TileDependencyRelation(
                            kind=access_dependency.kind,
                            dependency_id=access_dependency.dependency_id,
                            producer_access_id=producer_access.access_id,
                            consumer_access_id=consumer_access.access_id,
                            producer_root=edge.producer_root,
                            consumer_root=edge.consumer_root,
                            producer_scope_id=producer_scope_id,
                            consumer_scope_id=consumer_scope_id,
                            relation=(
                                _symbolic_access_predecessors(
                                    producer_access=producer_access,
                                    producer_domain=producer_domain,
                                    consumer_access=consumer_access,
                                    consumer_domain=consumer_domain,
                                )
                                if axes_have_canonical_origins
                                else None
                            ),
                        )
                    )
    return tuple(result)


def preceding_scope_relation(
    dependency_graph: TileDependencyGraph,
    *,
    scope_domains: tuple[LogicalDomain | None, ...],
    source_scope_id: int,
    consumer_scope_id: int,
    consumer_access_id: int,
) -> LogicalRelation | None:
    """Map a consumer scope to a preceding scope in the same task strand.

    An ancestor maps to its single enclosing instance.  A lexically earlier
    sibling subtree maps to every source instance under the shared enclosing
    instance.  Both are ordinary relations; no flattened action IDs are
    constructed.
    """
    scopes = dependency_graph.execution_scopes
    if len(scope_domains) != len(scopes):
        raise ValueError("scope domain count disagrees with the dependency graph")
    source_scope = scopes[source_scope_id]
    consumer_scope = scopes[consumer_scope_id]
    source_domain = scope_domains[source_scope_id]
    consumer_domain = scope_domains[consumer_scope_id]
    if (
        source_scope.root != consumer_scope.root
        or source_domain is None
        or consumer_domain is None
    ):
        return None
    try:
        consumer_access = next(
            access
            for access in dependency_graph.accesses
            if access.access_id == consumer_access_id
        )
    except StopIteration:
        return None
    if consumer_access.graph_node_index < 0:
        return None

    def lineage(scope_id: int) -> tuple[int, ...]:
        result: list[int] = []
        current: int | None = scope_id
        while current is not None:
            result.append(current)
            current = scopes[current].parent_scope_id
        result.reverse()
        return tuple(result)

    source_lineage = lineage(source_scope_id)
    consumer_lineage = lineage(consumer_scope_id)
    common_length = 0
    for source_ancestor, consumer_ancestor in zip(
        source_lineage, consumer_lineage, strict=False
    ):
        if source_ancestor != consumer_ancestor:
            break
        common_length += 1
    if not common_length:
        return None

    if common_length == len(source_lineage):
        equal_axes = source_domain.axis_order
    else:
        source_child = scopes[source_lineage[common_length]]
        source_node_index = source_child.callsite_path[-1][0]
        if common_length == len(consumer_lineage):
            consumer_node_index = consumer_access.graph_node_index
        else:
            consumer_child = scopes[consumer_lineage[common_length]]
            consumer_node_index = consumer_child.callsite_path[-1][0]
        if source_node_index >= consumer_node_index:
            return None
        common_scope_id = source_lineage[common_length - 1]
        common_domain = scope_domains[common_scope_id]
        if common_domain is None:
            return None
        equal_axes = common_domain.axis_order

    if any(axis not in consumer_domain.axis_counts for axis in equal_axes):
        return None
    equal_axis_set = frozenset(equal_axes)
    return LogicalRelation(
        source_domain=consumer_domain,
        target_domain=source_domain,
        pieces=(
            _LogicalRelationPiece(
                source_bounds_items=tuple(
                    (axis, 0, consumer_domain.axis_counts[axis], 1)
                    for axis in consumer_domain.axis_order
                ),
                target_ranges=tuple(
                    (
                        axis,
                        logical_axis_symbol(axis),
                        logical_axis_symbol(axis) + 1,  # pyrefly: ignore[unsupported-operation]
                        1,
                    )
                    if axis in equal_axis_set
                    else (
                        axis,
                        sympy.Integer(0),
                        sympy.Integer(source_domain.axis_counts[axis]),
                        1,
                    )
                    for axis in source_domain.axis_order
                ),
            ),
        ),
    )


def _allocation_region_from_bounds(
    access: TileAccess,
    bounds: tuple[tuple[int, int], ...],
    exact_dimensions: tuple[bool, ...],
) -> AllocationRegion:
    shape = access.tensor_shape
    strides = access.tensor_strides
    if any(begin >= end for begin, end in bounds):
        return AllocationRegion(
            (access.storage_offset, access.storage_offset),
            True,
            (shape, strides, access.storage_offset),
            bounds,
            all(exact_dimensions),
        )

    address_begin = access.storage_offset
    address_end = access.storage_offset
    for (begin, end), stride in zip(bounds, strides, strict=True):
        first = begin * stride
        last = (end - 1) * stride
        address_begin += min(first, last)
        address_end += max(first, last)
    address_end += 1

    coordinates_are_exact = all(exact_dimensions)
    active_strides = sorted(
        (abs(stride), end - begin)
        for (begin, end), stride in zip(bounds, strides, strict=True)
        if end - begin > 1
    )
    expected_stride = 1
    is_contiguous = coordinates_are_exact
    for stride, length in active_strides:
        if stride != expected_stride:
            is_contiguous = False
            break
        expected_stride *= length

    return AllocationRegion(
        (address_begin, address_end),
        is_contiguous,
        (shape, strides, access.storage_offset),
        bounds,
        coordinates_are_exact,
    )


def allocation_regions_may_overlap(
    left: AllocationRegion,
    right: AllocationRegion,
) -> bool:
    left_interval = left.address_interval
    right_interval = right.address_interval
    if left_interval is not None and right_interval is not None:
        if (
            left_interval[1] <= right_interval[0]
            or right_interval[1] <= left_interval[0]
        ):
            return False
    return not (
        left.layout is not None
        and left.layout == right.layout
        and _layout_is_injective(left.layout)
        and left.coordinate_bounds
        and len(left.coordinate_bounds) == len(right.coordinate_bounds)
        and any(
            left_end <= right_begin or right_end <= left_begin
            for (left_begin, left_end), (right_begin, right_end) in zip(
                left.coordinate_bounds,
                right.coordinate_bounds,
                strict=True,
            )
        )
    )


def _layout_is_injective(
    layout: tuple[tuple[int, ...], tuple[int, ...], int],
) -> bool:
    """Conservatively prove that distinct coordinates have distinct addresses."""
    shape, strides, _storage_offset = layout
    span = 1
    for stride, size in sorted(
        (abs(stride), size)
        for size, stride in zip(shape, strides, strict=True)
        if size > 1
    ):
        if stride < span:
            return False
        span += stride * (size - 1)
    return True


def _region_must_cover(cover: AllocationRegion, target: AllocationRegion) -> bool:
    cover_interval = cover.address_interval
    target_interval = target.address_interval
    if (
        cover.is_exact_contiguous
        and cover_interval is not None
        and target_interval is not None
        and cover_interval[0] <= target_interval[0]
        and target_interval[1] <= cover_interval[1]
    ):
        return True
    return (
        cover.coordinates_are_exact
        and cover.layout is not None
        and cover.layout == target.layout
        and len(cover.coordinate_bounds) == len(target.coordinate_bounds)
        and all(
            cover_begin <= target_begin and target_end <= cover_end
            for (cover_begin, cover_end), (target_begin, target_end) in zip(
                cover.coordinate_bounds,
                target.coordinate_bounds,
                strict=True,
            )
        )
    )


def _linear_region(begin: int, end: int) -> AllocationRegion:
    return AllocationRegion((begin, end), True)


def _intersect_regions(
    left: AllocationRegion,
    right: AllocationRegion,
) -> AllocationRegion:
    left_interval = left.address_interval
    right_interval = right.address_interval
    if left_interval is None or right_interval is None:
        return AllocationRegion(None, False)
    begin = max(left_interval[0], right_interval[0])
    end = min(left_interval[1], right_interval[1])
    if left.is_exact_contiguous and right.is_exact_contiguous:
        return _linear_region(begin, end)
    return AllocationRegion((begin, end), False)


def _subtract_regions(
    target: AllocationRegion,
    covers: tuple[AllocationRegion, ...],
) -> tuple[AllocationRegion, ...]:
    """Return the definitely-uncovered portion of ``target``.

    Exact contiguous regions can be split. Other layouts are retained unless
    one new write is proven to cover them completely. Retaining an imprecise
    region may add dependencies but can never lose a reaching definition.
    """
    pieces = (target,)
    for cover in covers:
        next_pieces: list[AllocationRegion] = []
        for piece in pieces:
            if _region_must_cover(cover, piece):
                continue
            piece_interval = piece.address_interval
            cover_interval = cover.address_interval
            if (
                piece.is_exact_contiguous
                and cover.is_exact_contiguous
                and piece_interval is not None
                and cover_interval is not None
            ):
                overlap_begin = max(piece_interval[0], cover_interval[0])
                overlap_end = min(piece_interval[1], cover_interval[1])
                if overlap_begin < overlap_end:
                    if piece_interval[0] < overlap_begin:
                        next_pieces.append(
                            _linear_region(piece_interval[0], overlap_begin)
                        )
                    if overlap_end < piece_interval[1]:
                        next_pieces.append(
                            _linear_region(overlap_end, piece_interval[1])
                        )
                    continue
            next_pieces.append(piece)
        pieces = tuple(next_pieces)
    return pieces


def _subtract_reaching_accesses(
    reaching: list[_ReachingAccess],
    writes: tuple[_ReachingAccess, ...],
) -> list[_ReachingAccess]:
    cover_regions = tuple(write.region for write in writes)
    return [
        _ReachingAccess(entry.root, entry.access, residual)
        for entry in reaching
        for residual in _subtract_regions(entry.region, cover_regions)
    ]


def build_tile_dependency_graph(
    accesses: tuple[TileAccess, ...],
    grid_block_ids: list[list[int]] | None = None,
    *,
    device_ir: DeviceIR | None = None,
    task_families: tuple[TaskFamily, ...] | None = None,
    root_phases: tuple[int, ...] | None = None,
    noncanonical_task_origin_block_ids: frozenset[int] = frozenset(),
    execution_scopes: tuple[ExecutionScope, ...] | None = None,
) -> TileDependencyGraph:
    """Build the minimal source-ordered allocation hazard graph.

    This pass is deliberately independent of code generation. It identifies the
    most recent writer and intervening readers of every allocation, then proves
    task readiness for the strict affine subset. Anything else remains a
    root-completion dependency.
    """
    if task_families is None and device_ir is not None:
        task_families = tuple(device_ir.task_families)
    if task_families is None:
        if grid_block_ids is None:
            raise TypeError(
                "device_ir, grid_block_ids, or task_families must be provided"
            )
        task_families = tuple(
            TaskFamily(
                axes=tuple(
                    LogicalTaskAxis(
                        block_id=block_id,
                        extent=None,
                        canonical_origin=(
                            block_id not in noncanonical_task_origin_block_ids
                        ),
                    )
                    for block_id in block_ids
                ),
            )
            for block_ids in grid_block_ids
        )
    elif grid_block_ids is not None and tuple(
        tuple(block_ids) for block_ids in grid_block_ids
    ) != tuple(family.logical_axis_order for family in task_families):
        raise ValueError("grid_block_ids disagree with task_families")

    root_count = len(task_families)
    if root_phases is None:
        root_phases = (0,) * root_count
    elif len(root_phases) != root_count:
        raise ValueError("root_phases must have one entry per task family")
    grid_block_ids = [list(family.logical_axis_order) for family in task_families]
    invalid_access = next(
        (
            access
            for access in accesses
            if access.allocation_id >= 0 and not 0 <= access.root < root_count
        ),
        None,
    )
    if invalid_access is not None:
        raise ValueError(
            f"tile dependency access {invalid_access.access_id} references unknown "
            f"root {invalid_access.root}; expected 0 <= root < {root_count}"
        )
    accesses_by_root: list[list[TileAccess]] = [[] for _ in range(root_count)]
    for access in accesses:
        if access.allocation_id >= 0:
            accesses_by_root[access.root].append(access)

    # Views can carry different source names at different roots while still
    # naming the same storage.  Keep one diagnostic alias set per allocation so
    # diagnostics can describe the DeviceIR edge without manufacturing one
    # duplicate edge per source spelling.
    tensor_names_by_allocation: dict[int, set[str]] = {}
    for access in accesses:
        if access.allocation_id >= 0 and access.tensor_name is not None:
            tensor_names_by_allocation.setdefault(access.allocation_id, set()).add(
                access.tensor_name
            )

    reads_by_root = [
        _accesses_by_allocation(root_accesses, "load")
        for root_accesses in accesses_by_root
    ]
    writes_by_root = [
        _accesses_by_allocation(root_accesses, "store")
        for root_accesses in accesses_by_root
    ]

    access_by_id = {access.access_id: access for access in accesses}
    region_by_access_id = {
        access.access_id: _access_region(access, task_families[access.root])
        for access in accesses
        if 0 <= access.root < root_count and access.allocation_id >= 0
    }
    dependencies_by_edge: dict[tuple[int, int, int], set[AccessDependency]] = {}
    reaching_writes: dict[int, list[_ReachingAccess]] = {}
    reaching_reads: dict[int, list[_ReachingAccess]] = {}

    def record(
        producer: _ReachingAccess,
        consumer: _ReachingAccess,
        kind: TileDependencyKind,
    ) -> None:
        dependencies_by_edge.setdefault(
            (producer.root, consumer.root, consumer.access.allocation_id), set()
        ).add(
            AccessDependency(
                kind=kind,
                producer_access_id=producer.access.access_id,
                consumer_access_id=consumer.access.access_id,
                region=_intersect_regions(producer.region, consumer.region),
            )
        )

    current_phase: int | None = None
    for consumer_root in range(root_count):
        phase = root_phases[consumer_root]
        if phase != current_phase:
            reaching_writes.clear()
            reaching_reads.clear()
            current_phase = phase
        reads = {
            allocation_id: tuple(
                _ReachingAccess(
                    consumer_root,
                    access,
                    region_by_access_id[access.access_id],
                )
                for access in allocation_accesses
            )
            for allocation_id, allocation_accesses in reads_by_root[
                consumer_root
            ].items()
        }
        writes = {
            allocation_id: tuple(
                _ReachingAccess(
                    consumer_root,
                    access,
                    region_by_access_id[access.access_id],
                )
                for access in allocation_accesses
            )
            for allocation_id, allocation_accesses in writes_by_root[
                consumer_root
            ].items()
        }

        for allocation_id, consumer_reads in reads.items():
            for consumer in consumer_reads:
                for producer in reaching_writes.get(allocation_id, ()):
                    if allocation_regions_may_overlap(producer.region, consumer.region):
                        record(
                            producer,
                            consumer,
                            TileDependencyKind.READ_AFTER_WRITE,
                        )
        for allocation_id, consumer_writes in writes.items():
            for consumer in consumer_writes:
                for producer in reaching_writes.get(allocation_id, ()):
                    if allocation_regions_may_overlap(producer.region, consumer.region):
                        record(
                            producer,
                            consumer,
                            TileDependencyKind.WRITE_AFTER_WRITE,
                        )
                for producer in reaching_reads.get(allocation_id, ()):
                    if allocation_regions_may_overlap(producer.region, consumer.region):
                        record(
                            producer,
                            consumer,
                            TileDependencyKind.WRITE_AFTER_READ,
                        )

        for allocation_id in reads.keys() | writes.keys():
            consumer_writes = writes.get(allocation_id, ())
            if consumer_writes:
                reaching_writes[allocation_id] = [
                    *_subtract_reaching_accesses(
                        reaching_writes.get(allocation_id, []), consumer_writes
                    ),
                    *consumer_writes,
                ]
                reaching_reads[allocation_id] = _subtract_reaching_accesses(
                    reaching_reads.get(allocation_id, []), consumer_writes
                )
            consumer_reads = reads.get(allocation_id, ())
            if consumer_reads:
                reaching_reads.setdefault(allocation_id, []).extend(
                    _ReachingAccess(consumer.root, consumer.access, residual)
                    for consumer in consumer_reads
                    for residual in _subtract_regions(
                        consumer.region,
                        tuple(write.region for write in consumer_writes),
                    )
                )

    edges: list[TileDependency] = []
    next_dependency_id = 0
    for (producer_root, consumer_root, allocation_id), dependency_set in sorted(
        dependencies_by_edge.items()
    ):
        ordered_dependencies = sorted(
            dependency_set,
            key=lambda dependency: (
                dependency.kind.value,
                dependency.producer_access_id,
                dependency.consumer_access_id,
                dependency.region.address_interval or (-1, -1),
            ),
        )
        access_dependencies = tuple(
            dataclasses.replace(
                dependency,
                dependency_id=next_dependency_id + index,
            )
            for index, dependency in enumerate(ordered_dependencies)
        )
        next_dependency_id += len(access_dependencies)
        kinds = frozenset(dependency.kind for dependency in access_dependencies)
        producer_accesses = tuple(
            access_by_id[access_id]
            for access_id in sorted(
                {dependency.producer_access_id for dependency in access_dependencies}
            )
        )
        consumer_accesses = tuple(
            access_by_id[access_id]
            for access_id in sorted(
                {dependency.consumer_access_id for dependency in access_dependencies}
            )
        )
        edges.append(
            TileDependency(
                producer_root=producer_root,
                consumer_root=consumer_root,
                allocation_id=allocation_id,
                tensor_names=frozenset(
                    tensor_names_by_allocation.get(allocation_id, ())
                ),
                kinds=kinds,
                producer_accesses=producer_accesses,
                consumer_accesses=consumer_accesses,
                access_dependencies=access_dependencies,
            )
        )

    if execution_scopes is None:
        execution_scopes = (
            build_execution_scopes(device_ir) if device_ir is not None else ()
        )
    scope_ids_by_access: list[tuple[int, ...]] = [
        ()
        for _ in range(max((access.access_id for access in accesses), default=-1) + 1)
    ]
    for access in accesses:
        if access.scope_id is None:
            if device_ir is not None:
                raise ValueError(
                    f"tile dependency access {access.access_id} has no execution scope"
                )
            continue
        if not 0 <= access.scope_id < len(execution_scopes):
            raise ValueError(
                f"tile dependency access {access.access_id} references unknown "
                f"execution scope {access.scope_id}"
            )
        scope = execution_scopes[access.scope_id]
        if scope.graph_id != access.graph_id or scope.root != access.root:
            raise ValueError(
                f"tile dependency access {access.access_id} references execution "
                f"scope {access.scope_id} with incompatible graph or root"
            )
        scope_ids_by_access[access.access_id] = (access.scope_id,)
    return TileDependencyGraph(
        task_families=task_families,
        accesses=accesses,
        edges=tuple(edges),
        execution_scopes=execution_scopes,
        scope_ids_by_access=tuple(scope_ids_by_access),
    )


def _accesses_by_allocation(
    accesses: list[TileAccess],
    kind: Literal["load", "store"],
) -> dict[int, tuple[TileAccess, ...]]:
    result: dict[int, list[TileAccess]] = {}
    for access in accesses:
        if access.kind == kind:
            result.setdefault(access.allocation_id, []).append(access)
    return {
        allocation_id: tuple(allocation_accesses)
        for allocation_id, allocation_accesses in result.items()
    }
