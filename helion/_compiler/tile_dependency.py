from __future__ import annotations

import dataclasses
import enum
from functools import cached_property
import itertools
import math
import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

import sympy
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.functions import Max as SymbolicMax
from torch.utils._sympy.functions import Min as SymbolicMin

from .. import exc

if TYPE_CHECKING:
    import ast
    from collections.abc import Callable
    from collections.abc import Mapping

    from .device_ir import DeviceIR


TILE_DEPENDENCY_SITE_IDS_META = "_tile_dependency_site_ids"
TILE_DEPENDENCY_SITE_ID_ATTR = "_tile_dependency_site_id"
_ALLOCATION_ADDRESS_AXIS = -1
_MAX_RELATION_PIECES = 4_096
_MAX_RELATION_PRODUCT_STATES = 65_536
# A memory hazard at one concrete producer/consumer callsite pairing.
DependencyObligation = tuple[int, int | None, int | None]
# SymPy's stubs do not expose a common arithmetic protocol shared with Python
# integers. Runtime validation in ``_integer_expression`` keeps this alias
# narrow while avoiding a false type-error fanout through concrete-only code.
IntegerExpression = Any
# One exact quasi-affine subset of a one-dimensional tensor subscript.  The
# first field contains ``(root_axis, coefficient, static_divisor)`` terms in
# logical-task coordinates; the remaining fields describe offsets as
# ``range(begin, end, step)`` over host-backed shape expressions. A divisor of
# one is an ordinary affine term.
AffineSubscriptRange = tuple[
    tuple[tuple[int, IntegerExpression, int], ...],
    IntegerExpression,
    IntegerExpression,
    int,
]


def _relation_product_is_within_budget(*factor_sizes: int) -> bool:
    """Check a prospective Cartesian product without forming it."""
    product_size = 1
    for factor_size in factor_sizes:
        if factor_size < 0:
            raise ValueError("relation product factors must be nonnegative")
        if factor_size == 0:
            return True
        if product_size > _MAX_RELATION_PRODUCT_STATES // factor_size:
            return False
        product_size *= factor_size
    return True


def _is_provably_nonnegative(
    expression: sympy.Expr,
    prove_nonnegative: Callable[[sympy.Expr], bool] | None,
) -> bool:
    """Use intrinsic SymPy facts, then an optional enclosing shape proof."""
    expression = sympy.sympify(expression)
    if expression.is_nonnegative is True:  # pyrefly: ignore[missing-attribute]
        return True
    interval = _bounded_parameter_expression_interval(expression)
    return (
        interval is not None and interval[0].is_nonnegative is True
    ) or (prove_nonnegative is not None and prove_nonnegative(expression))


def _integer_expression(value: IntegerExpression, *, description: str) -> sympy.Expr:
    """Return an integer-valued SymPy expression without specializing it."""
    if isinstance(value, int):
        return sympy.Integer(value)
    if not isinstance(value, sympy.Expr) or value.is_integer is not True:  # pyrefly: ignore[missing-attribute]
        raise ValueError(f"{description} must be an integer expression")
    return value


def _concrete_integer(value: IntegerExpression, *, description: str) -> int:
    """Require an integer expression to have no remaining parameters."""
    expression = sympy.simplify(_integer_expression(value, description=description))
    if expression.free_symbols:
        raise ValueError(
            f"{description} is symbolic; substitute parameters before enumeration"
        )
    if not isinstance(expression, sympy.Integer):
        raise ValueError(f"{description} did not evaluate to an integer: {expression}")
    return int(expression)


def _bounded_parameter_expression_interval(
    expression: sympy.Expr,
) -> tuple[sympy.Expr, sympy.Expr] | None:
    """Bound a finite integer expression independently of runtime parameters."""
    expression = sympy.sympify(expression)
    if expression.is_number:
        return expression, expression
    if isinstance(expression, sympy.Mod) and len(expression.args) == 2:
        modulus = expression.args[1]
        if (
            not modulus.free_symbols
            and modulus.is_integer is True
            and modulus.is_positive is True
        ):
            return sympy.Integer(0), modulus - 1
        return None
    quotient = _static_integer_quotient(expression)
    if quotient is not None:
        numerator, denominator = quotient
        numerator_interval = _bounded_parameter_expression_interval(numerator)
        if numerator_interval is None:
            return None
        return (
            sympy.floor(numerator_interval[0] / denominator),
            sympy.floor(numerator_interval[1] / denominator),
        )
    if isinstance(expression, sympy.Add):
        intervals = tuple(
            _bounded_parameter_expression_interval(child)
            for child in expression.args
        )
        if any(interval is None for interval in intervals):
            return None
        bounded = tuple(interval for interval in intervals if interval is not None)
        return (
            sympy.Add(*(interval[0] for interval in bounded)),
            sympy.Add(*(interval[1] for interval in bounded)),
        )
    if isinstance(expression, sympy.Mul):
        interval: tuple[sympy.Expr, sympy.Expr] = (
            sympy.Integer(1),
            sympy.Integer(1),
        )
        for child in expression.args:
            child_interval = _bounded_parameter_expression_interval(child)
            if child_interval is None:
                return None
            products = tuple(
                sympy.simplify(left * right)
                for left in interval
                for right in child_interval
            )
            if any(not value.is_number for value in products):
                return None
            interval = (min(products), max(products))
        return interval
    if expression.func in (sympy.Min, sympy.Max):
        intervals = tuple(
            _bounded_parameter_expression_interval(cast("sympy.Expr", child))
            for child in expression.args
        )
        if any(interval is None for interval in intervals):
            return None
        bounded = tuple(interval for interval in intervals if interval is not None)
        if expression.func == sympy.Min:
            return (
                min(interval[0] for interval in bounded),
                min(interval[1] for interval in bounded),
            )
        return (
            max(interval[0] for interval in bounded),
            max(interval[1] for interval in bounded),
        )
    return None


def _parameter_substitutions(
    substitutions: Mapping[sympy.Symbol, int],
) -> dict[sympy.Symbol, sympy.Integer]:
    """Validate the deliberately narrow Symbol-to-concrete-integer API."""
    result: dict[sympy.Symbol, sympy.Integer] = {}
    for parameter, value in substitutions.items():
        if not isinstance(parameter, sympy.Symbol):
            raise TypeError("parameter substitutions require SymPy Symbol keys")
        concrete = _concrete_integer(value, description=f"value for {parameter}")
        result[parameter] = sympy.Integer(concrete)
    return result


class TileDependencyKind(enum.Enum):
    """The memory hazard represented by a cross-loop dependency edge."""

    READ_AFTER_WRITE = "read_after_write"
    WRITE_AFTER_READ = "write_after_read"
    WRITE_AFTER_WRITE = "write_after_write"


def tile_dependency_site_id(node: ast.AST) -> int | None:
    """Return the stable DeviceIR execution site attached to a lowered loop."""
    site_id = getattr(node, TILE_DEPENDENCY_SITE_ID_ATTR, None)
    return site_id if isinstance(site_id, int) else None


def owner_roots_by_graph_id(device_ir: DeviceIR) -> tuple[tuple[int, ...], ...]:
    """Resolve every DeviceIR graph to all reachable top-level roots."""
    roots_by_graph: list[set[int]] = [set() for _ in device_ir.graphs]
    for site in build_execution_sites(device_ir):
        roots_by_graph[site.graph_id].add(site.root)
    return tuple(tuple(sorted(roots)) for roots in roots_by_graph)


@dataclasses.dataclass(frozen=True)
class TaskAxis:
    """One source-level axis in a root's logical task space.

    ``extent`` comes directly from the block-size registration performed while
    tracing ``hl.tile``. It is independent of the later PID task order or an
    L2 grouping chosen by a concrete configuration.
    """

    block_id: int
    extent: sympy.Expr | str | None
    canonical_origin: bool = True


@dataclasses.dataclass(frozen=True)
class TaskFamily:
    """One opaque top-level loop and its authoritative logical task domain."""

    axes: tuple[TaskAxis, ...]

    @property
    def logical_axis_order(self) -> tuple[int, ...]:
        return tuple(axis.block_id for axis in self.axes)

    def axis(self, block_id: int) -> TaskAxis | None:
        return next((axis for axis in self.axes if axis.block_id == block_id), None)


@dataclasses.dataclass(frozen=True)
class CoordinateDomain:
    """One configured Cartesian domain in canonical logical coordinates.

    Axis identity and geometry belong here. Linearization order is deliberately
    supplied to :meth:`coordinates` and :meth:`index` by the caller so event
    identity, task-local program order, and PID task order cannot accidentally
    become the same policy.
    """

    axis_order: tuple[int, ...]
    axis_counts_items: tuple[tuple[int, IntegerExpression], ...]
    block_sizes_items: tuple[tuple[int, int], ...] = ()
    kind: Literal["site", "allocation", "event", "task_order", "worker", "value"] = (
        "site"
    )
    identity: int | None = None
    _allow_empty: bool = dataclasses.field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(set(self.axis_order)) != len(self.axis_order):
            raise ValueError("coordinate-domain axes must be unique")
        if tuple(axis for axis, _count in self.axis_counts_items) != self.axis_order:
            raise ValueError("coordinate-domain counts must follow axis order")
        if (
            self.block_sizes_items
            and tuple(axis for axis, _size in self.block_sizes_items) != self.axis_order
        ):
            raise ValueError("coordinate-domain block sizes must follow axis order")
        normalized_counts = tuple(
            (
                axis,
                _integer_expression(
                    count,
                    description="coordinate-domain axis count",
                ),
            )
            for axis, count in self.axis_counts_items
        )
        object.__setattr__(self, "axis_counts_items", normalized_counts)
        for _axis, expression in normalized_counts:
            if expression.is_nonnegative is not True or (  # pyrefly: ignore[missing-attribute]
                expression.is_zero is True and not self._allow_empty
            ):
                raise ValueError("coordinate-domain axis counts must be positive")
        if any(size <= 0 for _axis, size in self.block_sizes_items):
            raise ValueError("coordinate-domain block sizes must be positive")

    @property
    def axis_count_expressions(self) -> dict[int, IntegerExpression]:
        """Return axis counts without requiring parameter substitution."""
        return dict(self.axis_counts_items)

    @property
    def block_sizes(self) -> dict[int, int]:
        return dict(self.block_sizes_items)

    @property
    def shape_expr(self) -> tuple[IntegerExpression, ...]:
        """Return the possibly parameterized Cartesian shape."""
        return tuple(count for _axis, count in self.axis_counts_items)

    @property
    def axis_counts(self) -> dict[int, int]:
        """Return concrete axis counts for legacy finite-domain operations."""
        return self._concrete_axis_counts()

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the concrete Cartesian shape."""
        counts = self._concrete_axis_counts()
        return tuple(counts[axis] for axis in self.axis_order)

    @property
    def size_expr(self) -> sympy.Expr:
        """Return the possibly parameterized number of domain points."""
        return sympy.prod(
            _integer_expression(count, description="coordinate-domain axis count")
            for count in self.shape_expr
        )

    @property
    def concrete_size(self) -> int:
        """Return the domain size, rejecting unresolved symbolic parameters."""
        return _concrete_integer(
            self.size_expr,
            description="coordinate-domain size",
        )

    @property
    def size(self) -> int:
        """Compatibility spelling for the explicitly concrete domain size."""
        return self.concrete_size

    @property
    def parameter_symbols(self) -> frozenset[sympy.Symbol]:
        """Return parameters used by this domain's axis counts."""
        return cast(
            "frozenset[sympy.Symbol]",
            frozenset(
                symbol
                for count in self.shape_expr
                for symbol in _integer_expression(
                    count,
                    description="coordinate-domain axis count",
                ).free_symbols
            ),
        )

    def substitute_parameters(
        self,
        substitutions: Mapping[sympy.Symbol, int],
    ) -> CoordinateDomain:
        """Return a concrete domain after a complete Symbol-to-int substitution."""
        concrete_substitutions = _parameter_substitutions(substitutions)
        missing = self.parameter_symbols - concrete_substitutions.keys()
        if missing:
            names = ", ".join(sorted(symbol.name for symbol in missing))
            raise ValueError(f"missing coordinate-domain parameters: {names}")
        counts = tuple(
            (
                axis,
                _concrete_integer(
                    _integer_expression(
                        count,
                        description="coordinate-domain axis count",
                    ).xreplace(concrete_substitutions),
                    description="substituted coordinate-domain axis count",
                ),
            )
            for axis, count in self.axis_counts_items
        )
        return CoordinateDomain(
            axis_order=self.axis_order,
            axis_counts_items=counts,
            block_sizes_items=self.block_sizes_items,
            kind=self.kind,
            identity=self.identity,
            _allow_empty=any(count == 0 for _axis, count in counts),
        )

    def _concrete_axis_counts(self) -> dict[int, int]:
        return {
            axis: _concrete_integer(
                count,
                description="coordinate-domain axis count",
            )
            for axis, count in self.axis_counts_items
        }

    def _validate_linearization_order(
        self,
        linearization_order: tuple[int, ...],
    ) -> None:
        if len(linearization_order) != len(self.axis_order) or set(
            linearization_order
        ) != set(self.axis_order):
            raise ValueError("linearization order must permute the domain axes")

    def coordinates(
        self,
        index: int,
        *,
        linearization_order: tuple[int, ...] | None = None,
    ) -> dict[int, int]:
        """Decode an integer using the requested fastest-to-slowest axes."""
        if not 0 <= index < self.size:
            raise IndexError(index)
        linearization_order = (
            self.axis_order if linearization_order is None else linearization_order
        )
        self._validate_linearization_order(linearization_order)
        counts = self._concrete_axis_counts()
        coordinates: dict[int, int] = {}
        remainder = index
        for axis in linearization_order:
            count = counts[axis]
            coordinates[axis] = remainder % count
            remainder //= count
        if remainder:
            raise AssertionError("index exceeds its coordinate domain")
        return coordinates

    def index(
        self,
        coordinates: dict[int, int],
        *,
        linearization_order: tuple[int, ...] | None = None,
    ) -> int:
        """Encode coordinates using the requested fastest-to-slowest axes."""
        linearization_order = (
            self.axis_order if linearization_order is None else linearization_order
        )
        self._validate_linearization_order(linearization_order)
        counts = self._concrete_axis_counts()
        result = 0
        multiplier = 1
        for axis in linearization_order:
            coordinate = coordinates[axis]
            count = counts[axis]
            if not 0 <= coordinate < count:
                raise IndexError(coordinate)
            result += coordinate * multiplier
            multiplier *= count
        return result


def coordinate_axis_symbol(axis: int) -> sympy.Symbol:
    """Return the canonical integer symbol for one coordinate-domain axis."""
    suffix = str(axis) if axis >= 0 else f"m{-axis}"
    return sympy.Symbol(f"coordinate_axis_{suffix}", integer=True, nonnegative=True)


def _simplify_bounded_coordinate_constants(
    expression: sympy.Expr,
    source_bounds: dict[
        int,
        tuple[IntegerExpression, IntegerExpression, int],
    ],
    *,
    axes: frozenset[int] | None = None,
) -> sympy.Expr:
    """Fold expressions that are constant over small bounded coordinates."""
    result = expression
    for axis, (begin, end, step) in source_bounds.items():
        if axes is not None and axis not in axes:
            continue
        symbol = coordinate_axis_symbol(axis)
        if symbol not in result.free_symbols:
            continue
        try:
            concrete_begin = _concrete_integer(
                begin,
                description="relation source bound",
            )
            concrete_end = _concrete_integer(
                end,
                description="relation source bound",
            )
        except ValueError:
            # Sampling parameter values would turn a regression oracle into a
            # proof.  Keeping the coordinate term is conservative.
            continue
        values = range(concrete_begin, concrete_end, step)
        if len(values) > 64:
            continue
        evaluated = tuple(
            sympy.simplify(result.xreplace({symbol: sympy.Integer(value)}))
            for value in values
        )
        if evaluated and all(
            sympy.simplify(value - evaluated[0]) == 0 for value in evaluated[1:]
        ):
            result = evaluated[0]
    return sympy.simplify(result)


def nested_logical_axes(
    root_domain: CoordinateDomain,
    site_domain: CoordinateDomain,
) -> tuple[int, ...]:
    """Return site axes that are not part of its owning root domain."""
    root_axes = frozenset(root_domain.axis_order)
    return tuple(axis for axis in site_domain.axis_order if axis not in root_axes)


@dataclasses.dataclass(frozen=True)
class _CoordinateRelationPiece:
    """One guarded source box mapped to a Cartesian target range."""

    source_bounds_items: tuple[
        tuple[int, IntegerExpression, IntegerExpression, int], ...
    ]
    target_ranges: tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_bounds_items",
            tuple(
                (
                    axis,
                    _integer_expression(begin, description="relation source bound"),
                    _integer_expression(end, description="relation source bound"),
                    _concrete_integer(step, description="relation source stride"),
                )
                for axis, begin, end, step in self.source_bounds_items
            ),
        )
        object.__setattr__(
            self,
            "target_ranges",
            tuple(
                (
                    axis,
                    _integer_expression(begin, description="relation target bound"),
                    _integer_expression(end, description="relation target bound"),
                    _concrete_integer(step, description="relation target stride"),
                )
                for axis, begin, end, step in self.target_ranges
            ),
        )

    def contains(self, coordinates: dict[int, int]) -> bool:
        return all(
            (
                concrete_begin <= coordinates[axis] < concrete_end
                and (coordinates[axis] - concrete_begin) % step == 0
            )
            for axis, begin, end, step in self.source_bounds_items
            for concrete_begin, concrete_end in (
                (
                    _concrete_integer(begin, description="relation source bound"),
                    _concrete_integer(end, description="relation source bound"),
                ),
            )
        )


@dataclasses.dataclass(frozen=True)
class CoordinateRelation:
    """Restricted symbolic relation between two Cartesian integer domains.

    Each piece maps one guarded source box to a Cartesian product of target
    ranges.  Expressions may use source-axis symbols, static affine arithmetic,
    floor division, modulo, and min/max.  Operations that cannot stay in this
    deliberately small grammar must decline rather than enumerate runtime
    instances.
    """

    source_domain: CoordinateDomain
    target_domain: CoordinateDomain
    pieces: tuple[_CoordinateRelationPiece, ...]

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
                _concrete_integer(step, description="relation source stride") <= 0
                for _axis, _begin, _end, step in piece.source_bounds_items
            ):
                raise ValueError("relation source strides must be positive")
            if any(
                _concrete_integer(step, description="relation target stride") <= 0
                for _axis, _begin, _end, step in piece.target_ranges
            ):
                raise ValueError("relation target strides must be positive")

    @property
    def parameter_symbols(self) -> frozenset[sympy.Symbol]:
        """Return non-coordinate parameters used by this relation."""
        coordinate_symbols = frozenset(
            coordinate_axis_symbol(axis)
            for axis in (
                *self.source_domain.axis_order,
                *self.target_domain.axis_order,
            )
        )
        symbols: set[sympy.Symbol] = set(self.source_domain.parameter_symbols)
        symbols.update(self.target_domain.parameter_symbols)
        for piece in self.pieces:
            for _axis, begin, end, _step in piece.source_bounds_items:
                symbols.update(
                    cast(
                        "set[sympy.Symbol]",
                        _integer_expression(
                            begin,
                            description="relation source bound",
                        ).free_symbols,
                    )
                )
                symbols.update(
                    cast(
                        "set[sympy.Symbol]",
                        _integer_expression(
                            end,
                            description="relation source bound",
                        ).free_symbols,
                    )
                )
            for _axis, begin, end, _step in piece.target_ranges:
                symbols.update(
                    cast("set[sympy.Symbol]", sympy.sympify(begin).free_symbols)
                )
                symbols.update(
                    cast("set[sympy.Symbol]", sympy.sympify(end).free_symbols)
                )
        return frozenset(symbols) - coordinate_symbols

    def substitute_parameters(
        self,
        substitutions: Mapping[sympy.Symbol, int],
    ) -> CoordinateRelation:
        """Concretize parameter bounds while retaining coordinate variables."""
        concrete_substitutions = _parameter_substitutions(substitutions)
        coordinate_symbols = frozenset(
            coordinate_axis_symbol(axis)
            for axis in (
                *self.source_domain.axis_order,
                *self.target_domain.axis_order,
            )
        )
        replaced_coordinates = coordinate_symbols & concrete_substitutions.keys()
        if replaced_coordinates:
            names = ", ".join(sorted(symbol.name for symbol in replaced_coordinates))
            raise ValueError(f"coordinate symbols are not parameters: {names}")
        missing = self.parameter_symbols - concrete_substitutions.keys()
        if missing:
            names = ", ".join(sorted(symbol.name for symbol in missing))
            raise ValueError(f"missing coordinate-relation parameters: {names}")

        def substitute_expression(expression: IntegerExpression) -> sympy.Expr:
            return sympy.simplify(
                _integer_expression(
                    expression,
                    description="relation expression",
                ).xreplace(concrete_substitutions)
            )

        pieces = tuple(
            _CoordinateRelationPiece(
                source_bounds_items=tuple(
                    (
                        axis,
                        _concrete_integer(
                            substitute_expression(begin),
                            description="substituted relation source bound",
                        ),
                        _concrete_integer(
                            substitute_expression(end),
                            description="substituted relation source bound",
                        ),
                        step,
                    )
                    for axis, begin, end, step in piece.source_bounds_items
                ),
                target_ranges=tuple(
                    (
                        axis,
                        substitute_expression(begin),
                        substitute_expression(end),
                        step,
                    )
                    for axis, begin, end, step in piece.target_ranges
                ),
            )
            for piece in self.pieces
        )
        result = CoordinateRelation(
            source_domain=self.source_domain.substitute_parameters(substitutions),
            target_domain=self.target_domain.substitute_parameters(substitutions),
            pieces=pieces,
        )
        converse = _memoized_exact_converse(self)
        if converse is not None:
            unmemoized_converse = CoordinateRelation(
                source_domain=converse.source_domain,
                target_domain=converse.target_domain,
                pieces=converse.pieces,
            )
            _remember_exact_converse(
                result,
                unmemoized_converse.substitute_parameters(substitutions),
            )
        return result

    @classmethod
    def identity(
        cls,
        source_domain: CoordinateDomain,
        target_domain: CoordinateDomain,
    ) -> CoordinateRelation:
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
                _CoordinateRelationPiece(
                    source_bounds_items=tuple(
                        (axis, 0, source_domain.axis_count_expressions[axis], 1)
                        for axis in source_domain.axis_order
                    ),
                    target_ranges=tuple(
                        (
                            axis,
                            coordinate_axis_symbol(axis),
                            coordinate_axis_symbol(axis) + 1,  # pyrefly: ignore[unsupported-operation]
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
        source_domain: CoordinateDomain,
        target_domain: CoordinateDomain,
        pieces: tuple[
            tuple[
                tuple[tuple[int, IntegerExpression, IntegerExpression, int], ...],
                tuple[sympy.Expr, ...],
            ],
            ...,
        ],
    ) -> CoordinateRelation:
        """Build a piecewise single-valued relation in domain axis order."""
        return cls(
            source_domain=source_domain,
            target_domain=target_domain,
            pieces=tuple(
                _CoordinateRelationPiece(
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
        source_domain: CoordinateDomain,
        target_domain: CoordinateDomain,
    ) -> CoordinateRelation:
        """Return the complete relation between two bounded domains."""
        return cls(
            source_domain=source_domain,
            target_domain=target_domain,
            pieces=(
                _CoordinateRelationPiece(
                    source_bounds_items=tuple(
                        (axis, 0, source_domain.axis_count_expressions[axis], 1)
                        for axis in source_domain.axis_order
                    ),
                    target_ranges=tuple(
                        (
                            axis,
                            sympy.Integer(0),
                            _integer_expression(
                                target_domain.axis_count_expressions[axis],
                                description="coordinate-domain axis count",
                            ),
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
        source_domain: CoordinateDomain,
        target_domain: CoordinateDomain,
    ) -> CoordinateRelation | None:
        """Project a domain onto a coordinate-compatible subdomain."""
        source_counts = source_domain.axis_count_expressions
        if any(
            axis not in source_counts
            or sympy.simplify(source_counts[axis] - count) != 0
            for axis, count in target_domain.axis_counts_items
        ):
            return None
        return cls(
            source_domain=source_domain,
            target_domain=target_domain,
            pieces=(
                _CoordinateRelationPiece(
                    source_bounds_items=tuple(
                        (axis, 0, source_counts[axis], 1)
                        for axis in source_domain.axis_order
                    ),
                    target_ranges=tuple(
                        (
                            axis,
                            coordinate_axis_symbol(axis),
                            coordinate_axis_symbol(axis) + 1,  # pyrefly: ignore[unsupported-operation]
                            1,
                        )
                        for axis in target_domain.axis_order
                    ),
                ),
            ),
        )

    def rename_target_axes(
        self, target_domain: CoordinateDomain
    ) -> CoordinateRelation | None:
        """Rename target axes positionally without changing coordinates."""
        old_axes = self.target_domain.axis_order
        new_axes = target_domain.axis_order
        if len(old_axes) != len(new_axes) or any(
            sympy.simplify(left - right) != 0
            for left, right in zip(
                self.target_domain.shape_expr,
                target_domain.shape_expr,
                strict=True,
            )
        ):
            return None
        renamed_axes = dict(zip(old_axes, new_axes, strict=True))
        return CoordinateRelation(
            source_domain=self.source_domain,
            target_domain=target_domain,
            pieces=tuple(
                dataclasses.replace(
                    piece,
                    target_ranges=tuple(
                        (renamed_axes[axis], begin, end, step)
                        for axis, begin, end, step in piece.target_ranges
                    ),
                )
                for piece in self.pieces
            ),
        )

    def rename_source_axes(
        self, source_domain: CoordinateDomain
    ) -> CoordinateRelation | None:
        """Rename source axes positionally without changing coordinates."""
        old_axes = self.source_domain.axis_order
        new_axes = source_domain.axis_order
        if len(old_axes) != len(new_axes) or any(
            sympy.simplify(left - right) != 0
            for left, right in zip(
                self.source_domain.shape_expr,
                source_domain.shape_expr,
                strict=True,
            )
        ):
            return None
        renamed_axes = dict(zip(old_axes, new_axes, strict=True))
        substitutions = {
            coordinate_axis_symbol(axis): coordinate_axis_symbol(renamed_axes[axis])
            for axis in old_axes
        }
        return CoordinateRelation(
            source_domain=source_domain,
            target_domain=self.target_domain,
            pieces=tuple(
                dataclasses.replace(
                    piece,
                    source_bounds_items=tuple(
                        (renamed_axes[axis], begin, end, step)
                        for axis, begin, end, step in piece.source_bounds_items
                    ),
                    target_ranges=tuple(
                        (
                            axis,
                            begin.xreplace(substitutions),
                            end.xreplace(substitutions),
                            step,
                        )
                        for axis, begin, end, step in piece.target_ranges
                    ),
                )
                for piece in self.pieces
            ),
        )

    def reorder_source_axes(
        self,
        source_axis_order: tuple[int, ...],
    ) -> CoordinateRelation | None:
        """Change only the fastest-to-slowest order of existing source axes."""
        if len(source_axis_order) != len(self.source_domain.axis_order) or set(
            source_axis_order
        ) != set(self.source_domain.axis_order):
            return None
        counts = self.source_domain.axis_counts
        block_sizes = self.source_domain.block_sizes
        source_domain = CoordinateDomain(
            axis_order=source_axis_order,
            axis_counts_items=tuple((axis, counts[axis]) for axis in source_axis_order),
            block_sizes_items=tuple(
                (axis, block_sizes[axis]) for axis in source_axis_order
            )
            if block_sizes
            else (),
            kind=self.source_domain.kind,
            identity=self.source_domain.identity,
        )
        return CoordinateRelation(
            source_domain=source_domain,
            target_domain=self.target_domain,
            pieces=tuple(
                dataclasses.replace(
                    piece,
                    source_bounds_items=tuple(
                        {
                            axis: (axis, begin, end, step)
                            for axis, begin, end, step in piece.source_bounds_items
                        }[axis]
                        for axis in source_axis_order
                    ),
                )
                for piece in self.pieces
            ),
        )

    def project_target(
        self,
        target_domain: CoordinateDomain,
    ) -> CoordinateRelation | None:
        """Existentially drop target axes while preserving the remaining map."""
        current_counts = self.target_domain.axis_count_expressions
        if any(
            axis not in current_counts
            or sympy.simplify(current_counts[axis] - count) != 0
            for axis, count in target_domain.axis_counts_items
        ):
            return None
        retained_axes = frozenset(target_domain.axis_order)
        return CoordinateRelation(
            source_domain=self.source_domain,
            target_domain=target_domain,
            pieces=tuple(
                _CoordinateRelationPiece(
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
        source_domain: CoordinateDomain,
    ) -> CoordinateRelation | None:
        """Union dropped source axes when their images remain rectilinear."""
        current_counts = self.source_domain.axis_count_expressions
        if (
            any(
                axis not in current_counts
                or sympy.simplify(current_counts[axis] - count) != 0
                for axis, count in source_domain.axis_counts_items
            )
            or len(self.pieces) > _MAX_RELATION_PIECES
        ):
            return None
        retained_axes = frozenset(source_domain.axis_order)
        dropped_axes = frozenset(self.source_domain.axis_order) - retained_axes
        pieces: list[_CoordinateRelationPiece] = []
        for piece in self.pieces:
            source_bounds = {
                axis: (begin, end, step)
                for axis, begin, end, step in piece.source_bounds_items
            }
            eliminated_uses: set[int] = set()
            target_ranges: list[tuple[int, sympy.Expr, sympy.Expr, int]] = []
            for target_axis, begin, end, target_step in piece.target_ranges:
                begin = _simplify_bounded_coordinate_constants(
                    begin, source_bounds, axes=dropped_axes
                )
                end = _simplify_bounded_coordinate_constants(
                    end, source_bounds, axes=dropped_axes
                )
                symbols: dict[sympy.Basic, int] = {
                    coordinate_axis_symbol(axis): axis
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
                if len(eliminated_axes) != 1:
                    return None
                (eliminated_axis,) = eliminated_axes
                if eliminated_axis in eliminated_uses:
                    # Projecting one coordinate into several target dimensions
                    # creates a diagonal rather than a Cartesian product.
                    return None
                eliminated_uses.add(eliminated_axis)
                eliminated_symbol = coordinate_axis_symbol(eliminated_axis)
                expanded_begin = sympy.expand(begin)
                stride_expression = expanded_begin.coeff(eliminated_symbol)
                base_expression = sympy.simplify(
                    expanded_begin - stride_expression * eliminated_symbol
                )
                width_expression = sympy.simplify(end - begin)  # pyrefly: ignore[unsupported-operation]
                if (
                    target_step != 1
                    or stride_expression.free_symbols
                    or width_expression.free_symbols
                    or stride_expression.is_integer is not True
                    or width_expression.is_integer is not True
                    or eliminated_symbol in base_expression.free_symbols
                ):
                    return None
                stride = int(stride_expression)
                width = int(width_expression)
                if stride <= 0 or width <= 0:
                    return None
                source_begin, source_end, source_step = source_bounds[eliminated_axis]
                source_count = self.source_domain.axis_count_expressions[
                    eliminated_axis
                ]
                target_count = self.target_domain.axis_count_expressions[target_axis]
                if (
                    source_step == 1
                    and stride == 1
                    and width == 1
                    and sympy.simplify(source_begin) == 0
                    and sympy.simplify(source_end - source_count) == 0  # pyrefly: ignore[unsupported-operation]
                    and sympy.simplify(base_expression) == 0
                    and sympy.simplify(target_count - source_count) == 0  # pyrefly: ignore[unsupported-operation]
                ):
                    # Existentially removing a complete runtime-sized
                    # positional factor produces the complete corresponding
                    # target axis.  No runtime extent is enumerated.
                    target_ranges.append(
                        (
                            target_axis,
                            sympy.Integer(0),
                            _integer_expression(
                                target_count,
                                description="coordinate-domain axis count",
                            ),
                            1,
                        )
                    )
                    continue
                try:
                    concrete_source_begin = _concrete_integer(
                        source_begin,
                        description="projected source begin",
                    )
                    concrete_source_end = _concrete_integer(
                        source_end,
                        description="projected source end",
                    )
                except ValueError:
                    # Retained axes may be parameterized, but eliminating a
                    # runtime-sized axis would require a non-static union.
                    return None
                if concrete_source_end <= concrete_source_begin:
                    return None
                final_source = (
                    concrete_source_begin
                    + (concrete_source_end - concrete_source_begin - 1)
                    // source_step
                    * source_step
                )
                projected_begin = sympy.simplify(
                    base_expression + concrete_source_begin * stride
                )
                projected_end = sympy.simplify(
                    base_expression + final_source * stride + width
                )
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
                _CoordinateRelationPiece(
                    source_bounds_items=tuple(
                        (axis, *source_bounds[axis])
                        for axis in source_domain.axis_order
                    ),
                    target_ranges=tuple(target_ranges),
                )
            )
        unique_pieces = tuple(dict.fromkeys(pieces))
        if len(unique_pieces) > _MAX_RELATION_PIECES:
            return None
        return CoordinateRelation(
            source_domain=source_domain,
            target_domain=self.target_domain,
            pieces=unique_pieces,
        )

    def lift_source(self, source_domain: CoordinateDomain) -> CoordinateRelation | None:
        """Add unused source axes without changing any related target set."""
        source_counts = source_domain.axis_count_expressions
        if any(
            axis not in source_counts
            or sympy.simplify(source_counts[axis] - count) != 0
            for axis, count in self.source_domain.axis_counts_items
        ):
            return None
        current_axes = frozenset(self.source_domain.axis_order)
        pieces: list[_CoordinateRelationPiece] = []
        for piece in self.pieces:
            bounds = {
                axis: (begin, end, step)
                for axis, begin, end, step in piece.source_bounds_items
            }
            pieces.append(
                _CoordinateRelationPiece(
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
        return CoordinateRelation(
            source_domain=source_domain,
            target_domain=self.target_domain,
            pieces=tuple(dict.fromkeys(pieces)),
        )

    def then(self, following: CoordinateRelation) -> CoordinateRelation | None:
        """Compose two relations and retain a proved reverse composition."""
        result = self._then_without_converse(following)
        if result is None:
            return None
        first_converse = _memoized_exact_converse(self)
        following_converse = _memoized_exact_converse(following)
        if first_converse is None or following_converse is None:
            return result
        converse = following_converse._then_without_converse(first_converse)
        if converse is not None:
            _remember_exact_converse(result, converse)
        return result

    def _then_without_converse(
        self,
        following: CoordinateRelation,
    ) -> CoordinateRelation | None:
        """Compose a projection/full-target-set relation with another relation.

        This is the program-order composition needed for nested checkpoints.
        ``self`` maps a later site to preceding site instances; ``following``
        maps those preceding instances to their acquired producer instances.
        """
        if self.target_domain != following.source_domain:
            return None
        if (
            following.source_domain == following.target_domain
            and following
            == CoordinateRelation.identity(
                following.source_domain,
                following.target_domain,
            )
        ):
            return self
        if following.is_positional_bijection():
            renamed = self.rename_target_axes(following.target_domain)
            if renamed is not None:
                return renamed
        coordinate_permutation = _coordinate_permutation_axes(following)
        if coordinate_permutation is not None:
            pieces = tuple(
                _CoordinateRelationPiece(
                    source_bounds_items=piece.source_bounds_items,
                    target_ranges=tuple(
                        (
                            target_axis,
                            *{
                                axis: (begin, end, step)
                                for axis, begin, end, step in piece.target_ranges
                            }[coordinate_permutation[target_axis]],
                        )
                        for target_axis in following.target_domain.axis_order
                    ),
                )
                for piece in self.pieces
            )
            return CoordinateRelation(
                source_domain=self.source_domain,
                target_domain=following.target_domain,
                pieces=pieces,
            )
        point_composition = _compose_point_relations(self, following)
        if point_composition is not None:
            return point_composition
        if self.parameter_symbols or following.parameter_symbols:
            # The remaining projection/full-set fallback is intentionally
            # finite-domain machinery.  Unsupported symbolic compositions
            # decline instead of asking ``axis_counts`` to specialize them.
            return None
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
            symbol = coordinate_axis_symbol(axis)
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

        retained_domain = CoordinateDomain(
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

    def factor_through(
        self,
        quotient: CoordinateRelation,
    ) -> CoordinateRelation | None:
        """Factor this relation through an exact source-coordinate quotient.

        Given ``self: C -> P`` and ``quotient: C -> K``, return ``F: K -> P``
        only when ``self == quotient ; F`` is proved structurally.  The
        currently supported quotient is a total coordinate projection or
        renaming.  Dropped coordinates must neither restrict source support nor
        occur in a target expression.
        """
        if quotient.source_domain != self.source_domain:
            return None
        if (
            ordinal_inverse := _source_support_ordinalization(
                quotient,
                reverse=True,
            )
        ) is not None:
            factored = _factor_through_source_ordinalization(
                self,
                quotient,
                ordinal_inverse,
            )
            if factored is not None:
                return factored
        if len(quotient.pieces) != 1:
            return None
        (quotient_piece,) = quotient.pieces
        if quotient_piece.source_bounds_items != tuple(
            (axis, 0, self.source_domain.axis_count_expressions[axis], 1)
            for axis in self.source_domain.axis_order
        ):
            return None

        source_axis_by_key_axis: dict[int, int] = {}
        for key_axis, begin, end, step in quotient_piece.target_ranges:
            interval = _single_axis_interval(
                begin,
                end,
                domain=self.source_domain,
            )
            if interval is None:
                return None
            source_axis, stride, offset, width = interval
            if stride != 1 or offset != 0 or width != 1 or step != 1:
                return None
            if source_axis in source_axis_by_key_axis.values():
                return None
            if (
                sympy.simplify(
                    self.source_domain.axis_count_expressions[source_axis]
                    - quotient.target_domain.axis_count_expressions[key_axis]
                )
                != 0
            ):
                return None
            source_axis_by_key_axis[key_axis] = source_axis
        if tuple(source_axis_by_key_axis) != quotient.target_domain.axis_order:
            return None

        key_axis_by_source_axis = {
            source_axis: key_axis
            for key_axis, source_axis in source_axis_by_key_axis.items()
        }
        dropped_axes = (
            frozenset(self.source_domain.axis_order) - key_axis_by_source_axis.keys()
        )
        substitutions = {
            coordinate_axis_symbol(source_axis): coordinate_axis_symbol(key_axis)
            for source_axis, key_axis in key_axis_by_source_axis.items()
        }
        pieces: list[_CoordinateRelationPiece] = []
        for piece in self.pieces:
            source_bounds = {
                axis: (begin, end, step)
                for axis, begin, end, step in piece.source_bounds_items
            }
            if any(
                source_bounds[axis]
                != (0, self.source_domain.axis_count_expressions[axis], 1)
                for axis in dropped_axes
            ):
                return None
            simplified_target_ranges = tuple(
                (
                    target_axis,
                    _simplify_bounded_coordinate_constants(
                        begin, source_bounds, axes=dropped_axes
                    ),
                    _simplify_bounded_coordinate_constants(
                        end, source_bounds, axes=dropped_axes
                    ),
                    step,
                )
                for target_axis, begin, end, step in piece.target_ranges
            )
            if any(
                symbol == coordinate_axis_symbol(axis)
                for _target_axis, begin, end, _step in simplified_target_ranges
                for symbol in begin.free_symbols | end.free_symbols
                for axis in dropped_axes
            ):
                return None
            pieces.append(
                _CoordinateRelationPiece(
                    source_bounds_items=tuple(
                        (
                            key_axis,
                            *source_bounds[source_axis_by_key_axis[key_axis]],
                        )
                        for key_axis in quotient.target_domain.axis_order
                    ),
                    target_ranges=tuple(
                        (
                            target_axis,
                            begin.xreplace(substitutions),
                            end.xreplace(substitutions),
                            step,
                        )
                        for target_axis, begin, end, step in simplified_target_ranges
                    ),
                )
            )
        return CoordinateRelation(
            source_domain=quotient.target_domain,
            target_domain=self.target_domain,
            pieces=tuple(dict.fromkeys(pieces)),
        )

    def covers(self, required: CoordinateRelation) -> bool:
        """Conservatively prove that this relation contains ``required``."""
        if (
            self.source_domain != required.source_domain
            or self.target_domain != required.target_domain
        ):
            return False
        if not required.pieces or self == required or self.is_total():
            return True
        if (
            len(self.pieces) > _MAX_RELATION_PIECES
            or len(required.pieces) > _MAX_RELATION_PIECES
        ):
            return False
        if self.is_pointwise_equal_on_same_support(required):
            return True
        positional_product = self._parameterized_positional_product
        required_positional_product = required._parameterized_positional_product
        if (
            positional_product is not None
            and required_positional_product is not None
            and positional_product[0] == required_positional_product[0]
        ):
            return positional_product[1].covers(required_positional_product[1])
        if not _relation_product_is_within_budget(
            len(self.pieces), len(required.pieces)
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

    def source_axes_affecting_targets(self) -> tuple[int, ...] | None:
        """Return source axes that can change the related target set."""
        symbols: dict[sympy.Basic, int] = {
            coordinate_axis_symbol(axis): axis for axis in self.source_domain.axis_order
        }
        domain_parameters = (
            self.source_domain.parameter_symbols | self.target_domain.parameter_symbols
        )
        used: set[int] = set()
        full_source_bounds = {
            axis: (0, self.source_domain.axis_count_expressions[axis], 1)
            for axis in self.source_domain.axis_order
        }
        for piece in self.pieces:
            source_bounds = {
                axis: (begin, end, step)
                for axis, begin, end, step in piece.source_bounds_items
            }
            for axis, begin, end, step in piece.source_bounds_items:
                if (begin, end, step) != full_source_bounds[axis]:
                    used.add(axis)
            for _axis, begin, end, _step in piece.target_ranges:
                begin = _simplify_bounded_coordinate_constants(begin, source_bounds)
                end = _simplify_bounded_coordinate_constants(end, source_bounds)
                for symbol in begin.free_symbols | end.free_symbols:
                    source_axis = symbols.get(symbol)
                    if source_axis is None:
                        if symbol in domain_parameters:
                            continue
                        return None
                    used.add(source_axis)
        return tuple(axis for axis in self.source_domain.axis_order if axis in used)

    @cached_property
    def _parameterized_positional_product(
        self,
    ) -> tuple[tuple[tuple[int, int], ...], CoordinateRelation] | None:
        """Factor shape-varying positional axes from a static inner relation.

        A runtime-sized axis is removable only when every relation piece spans
        that complete source axis and maps it point-for-point onto one equally
        sized target axis. The residual relation must be fully concrete. This
        is an exact Cartesian-product proof, not a sampled-shape shortcut.
        """
        if not self.parameter_symbols or not self.pieces:
            return None
        source_counts = self.source_domain.axis_count_expressions
        target_counts = self.target_domain.axis_count_expressions
        parameterized_source_axes = tuple(
            axis
            for axis in self.source_domain.axis_order
            if _integer_expression(
                source_counts[axis],
                description="coordinate-domain axis count",
            ).free_symbols
        )
        if not parameterized_source_axes:
            return None

        pairs: list[tuple[int, int]] = []
        used_target_axes: set[int] = set()
        for source_axis in parameterized_source_axes:
            source_count = _integer_expression(
                source_counts[source_axis],
                description="coordinate-domain axis count",
            )
            source_symbol = coordinate_axis_symbol(source_axis)
            candidates = tuple(
                target_axis
                for target_axis in self.target_domain.axis_order
                if target_axis not in used_target_axes
                and sympy.simplify(target_counts[target_axis] - source_count) == 0
                and all(
                    next(
                        target_range
                        for target_range in piece.target_ranges
                        if target_range[0] == target_axis
                    )[3]
                    == 1
                    and sympy.simplify(
                        next(
                            target_range
                            for target_range in piece.target_ranges
                            if target_range[0] == target_axis
                        )[1]
                        - source_symbol
                    )
                    == 0
                    and sympy.simplify(
                        next(
                            target_range
                            for target_range in piece.target_ranges
                            if target_range[0] == target_axis
                        )[2]
                        - source_symbol
                        - 1
                    )
                    == 0
                    for piece in self.pieces
                )
            )
            if len(candidates) != 1:
                return None
            target_axis = candidates[0]
            if any(
                next(
                    bounds
                    for bounds in piece.source_bounds_items
                    if bounds[0] == source_axis
                )
                != (source_axis, 0, source_count, 1)
                for piece in self.pieces
            ):
                return None
            pairs.append((source_axis, target_axis))
            used_target_axes.add(target_axis)

        removed_source_symbols = frozenset(
            coordinate_axis_symbol(source_axis) for source_axis, _target_axis in pairs
        )
        if any(
            removed_source_symbols & (begin.free_symbols | end.free_symbols)
            for piece in self.pieces
            for target_axis, begin, end, _step in piece.target_ranges
            if target_axis not in used_target_axes
        ):
            return None

        paired_source_axes = frozenset(source_axis for source_axis, _ in pairs)
        residual_source = CoordinateDomain(
            axis_order=tuple(
                axis
                for axis in self.source_domain.axis_order
                if axis not in paired_source_axes
            ),
            axis_counts_items=tuple(
                (axis, count)
                for axis, count in self.source_domain.axis_counts_items
                if axis not in paired_source_axes
            ),
            block_sizes_items=tuple(
                (axis, size)
                for axis, size in self.source_domain.block_sizes_items
                if axis not in paired_source_axes
            ),
            kind=self.source_domain.kind,
            identity=self.source_domain.identity,
        )
        residual_target = CoordinateDomain(
            axis_order=tuple(
                axis
                for axis in self.target_domain.axis_order
                if axis not in used_target_axes
            ),
            axis_counts_items=tuple(
                (axis, count)
                for axis, count in self.target_domain.axis_counts_items
                if axis not in used_target_axes
            ),
            block_sizes_items=tuple(
                (axis, size)
                for axis, size in self.target_domain.block_sizes_items
                if axis not in used_target_axes
            ),
            kind=self.target_domain.kind,
            identity=self.target_domain.identity,
        )
        residual = CoordinateRelation(
            source_domain=residual_source,
            target_domain=residual_target,
            pieces=tuple(
                dict.fromkeys(
                    _CoordinateRelationPiece(
                        source_bounds_items=tuple(
                            bounds
                            for bounds in piece.source_bounds_items
                            if bounds[0] not in paired_source_axes
                        ),
                        target_ranges=tuple(
                            target_range
                            for target_range in piece.target_ranges
                            if target_range[0] not in used_target_axes
                        ),
                    )
                    for piece in self.pieces
                )
            ),
        )
        if residual.parameter_symbols:
            return None
        return tuple(pairs), residual

    def producer_set_quotient(
        self,
    ) -> tuple[CoordinateRelation, CoordinateRelation] | None:
        """Factor a separable dense producer box through its set identity.

        Returns ``(keys_by_source, targets_by_key)`` when each nontrivial
        producer-coordinate range is a complete, equal-width partition driven
        by one distinct source axis.  This captures equivalence classes such
        as four adjacent consumer heads sharing the same producer set, while
        rejecting diagonal, partial, masked, or otherwise nonseparable maps.
        """
        if len(self.pieces) != 1:
            return None
        (piece,) = self.pieces
        full_source_bounds = tuple(
            (axis, 0, self.source_domain.axis_count_expressions[axis], 1)
            for axis in self.source_domain.axis_order
        )
        if piece.source_bounds_items != full_source_bounds:
            return None

        key_expression_by_source_axis: dict[int, sympy.Expr] = {}
        key_count_by_source_axis: dict[int, sympy.Expr] = {}
        target_partition_by_axis: dict[int, tuple[int, int]] = {}
        for target_axis, begin, end, step in piece.target_ranges:
            target_count = _integer_expression(
                self.target_domain.axis_count_expressions[target_axis],
                description="coordinate-domain axis count",
            )
            width_expression = sympy.simplify(end - begin)  # pyrefly: ignore[unsupported-operation]
            if step != 1 or not isinstance(width_expression, sympy.Integer):
                return None
            width = int(width_expression)
            if width <= 0:
                return None
            if sympy.simplify(begin) == 0 and sympy.simplify(end - target_count) == 0:  # pyrefly: ignore[unsupported-operation]
                continue
            key_count = sympy.simplify(target_count / width)  # pyrefly: ignore[unsupported-operation]
            if (
                key_count.is_integer is not True  # pyrefly: ignore[missing-attribute]
                or key_count.is_nonnegative is not True  # pyrefly: ignore[missing-attribute]
            ):
                return None
            begin_bounds = _logical_expression_bounds(
                begin,
                domain=self.source_domain,
                source_bounds=piece.source_bounds_items,
            )
            if (
                begin_bounds is None
                or sympy.simplify(begin_bounds[0]) != 0
                or sympy.simplify(begin_bounds[1] - (target_count - width)) != 0  # pyrefly: ignore[unsupported-operation]
                or sympy.simplify(sympy.Mod(begin, width)) != 0
            ):
                return None
            key_expression = sympy.simplify(begin / width)
            if key_expression.is_integer is not True:
                return None
            source_symbols = tuple(
                axis
                for axis in self.source_domain.axis_order
                if coordinate_axis_symbol(axis) in key_expression.free_symbols
            )
            if len(source_symbols) != 1:
                return None
            (source_axis,) = source_symbols
            if source_axis in key_expression_by_source_axis:
                return None
            source_count = _integer_expression(
                self.source_domain.axis_count_expressions[source_axis],
                description="coordinate-domain axis count",
            )
            group_size_expression = sympy.simplify(source_count / key_count)  # pyrefly: ignore[unsupported-operation]
            if (
                not isinstance(group_size_expression, sympy.Integer)
                or int(group_size_expression) <= 0
            ):
                return None
            group_size = int(group_size_expression)
            source_symbol = coordinate_axis_symbol(source_axis)
            expected = (
                source_symbol
                if group_size == 1
                else sympy.floor(source_symbol / group_size)
            )
            if sympy.simplify(key_expression - expected) != 0:
                return None
            key_expression_by_source_axis[source_axis] = key_expression
            key_count_by_source_axis[source_axis] = key_count
            target_partition_by_axis[target_axis] = (source_axis, width)

        if not key_expression_by_source_axis:
            return None
        key_axes = tuple(
            axis
            for axis in self.source_domain.axis_order
            if axis in key_expression_by_source_axis
        )
        key_domain = CoordinateDomain(
            axis_order=key_axes,
            axis_counts_items=tuple(
                (axis, key_count_by_source_axis[axis]) for axis in key_axes
            ),
            kind="event",
        )
        keys_by_source = CoordinateRelation.point_map(
            self.source_domain,
            key_domain,
            (
                (
                    piece.source_bounds_items,
                    tuple(key_expression_by_source_axis[axis] for axis in key_axes),
                ),
            ),
        )
        if not keys_by_source.is_total_function():
            return None
        target_ranges: list[tuple[int, sympy.Expr, sympy.Expr, int]] = []
        for target_axis, begin, end, step in piece.target_ranges:
            partition = target_partition_by_axis.get(target_axis)
            if partition is None:
                target_ranges.append((target_axis, begin, end, step))
                continue
            source_axis, width = partition
            target_begin = coordinate_axis_symbol(source_axis) * width
            target_ranges.append(
                (
                    target_axis,
                    target_begin,
                    target_begin + width,  # pyrefly: ignore[unsupported-operation]
                    1,
                )
            )
        targets_by_key = CoordinateRelation(
            source_domain=key_domain,
            target_domain=self.target_domain,
            pieces=(
                _CoordinateRelationPiece(
                    source_bounds_items=tuple(
                        (axis, 0, key_domain.axis_count_expressions[axis], 1)
                        for axis in key_axes
                    ),
                    target_ranges=tuple(target_ranges),
                ),
            ),
        )
        if targets_by_key._separable_fixed_width_partition() is None:
            return None
        return keys_by_source, targets_by_key

    @cached_property
    def _ordinalized_source_support(self) -> CoordinateRelation | None:
        """Canonical dense ordinal for this relation's represented support."""
        return _source_support_ordinalization(self)

    @cached_property
    def _factored_source_support_converse(self) -> CoordinateRelation | None:
        """Prove and invert this relation through its support ordinalization."""
        ordinalization = self._ordinalized_source_support
        if ordinalization is None:
            return None
        ordinal_inverse = _source_support_ordinalization(
            ordinalization,
            reverse=True,
        )
        if ordinal_inverse is None:
            return None
        _remember_exact_converse(ordinalization, ordinal_inverse)
        quotient = _factor_through_source_ordinalization(
            self,
            ordinalization,
            ordinal_inverse,
        )
        if quotient is None or not quotient.is_total_function():
            return None
        quotient_inverse = quotient.converse()
        if quotient_inverse is None or not quotient_inverse.is_total_function():
            return None
        return quotient_inverse._then_without_converse(ordinal_inverse)

    def converse(self) -> CoordinateRelation | None:
        """Return the exact converse when representable without enumeration."""
        if (converse := _memoized_exact_converse(self)) is not None:
            return converse
        if self.target_domain.size_expr.is_zero is True or not self.pieces:
            converse = CoordinateRelation(
                source_domain=self.target_domain,
                target_domain=self.source_domain,
                pieces=(),
            )
            return _remember_exact_converse(self, converse)
        if (converse := _coordinate_permutation_converse(self)) is not None:
            return _remember_exact_converse(self, converse)
        if self.is_positional_bijection():
            converse = self.derive_converse_and_target_counts()[0]
            return (
                None if converse is None else _remember_exact_converse(self, converse)
            )
        cached_converse = self.__dict__.get("_cached_converse")
        if (
            isinstance(cached_converse, CoordinateRelation)
            and cached_converse.is_single_valued()
        ):
            return _remember_exact_converse(self, cached_converse)
        if (converse := _symbolic_single_source_mixed_radix_converse(self)) is not None:
            return _remember_exact_converse(self, converse)
        if (converse := _cheap_source_support_converse(self)) is not None:
            return converse
        if (converse := self._factored_source_support_converse) is not None:
            return _remember_exact_converse(self, converse)
        if self.parameter_symbols:
            converse = self.derive_converse_and_target_counts()[0]
            return (
                None if converse is None else _remember_exact_converse(self, converse)
            )
        converse = self._cached_converse
        if converse is not None and converse.is_single_valued():
            return _remember_exact_converse(self, converse)
        target_counts = self.target_count_by_source()
        if target_counts is None:
            return (
                None if converse is None else _remember_exact_converse(self, converse)
            )
        converse = _derived_converse(self, target_counts) or converse
        return None if converse is None else _remember_exact_converse(self, converse)

    def derive_converse_and_target_counts(
        self,
    ) -> tuple[CoordinateRelation | None, CoordinateRelation | None]:
        """Derive the converse and per-source target counts from one proof."""
        if self.is_positional_bijection():
            converse = CoordinateRelation.point_map(
                self.target_domain,
                self.source_domain,
                (
                    (
                        tuple(
                            (
                                axis,
                                0,
                                self.target_domain.axis_count_expressions[axis],
                                1,
                            )
                            for axis in self.target_domain.axis_order
                        ),
                        tuple(
                            coordinate_axis_symbol(axis)
                            for axis in self.target_domain.axis_order
                        ),
                    ),
                ),
            )
            value_axis = 0
            value_domain = CoordinateDomain(
                axis_order=(value_axis,),
                axis_counts_items=((value_axis, 2),),
                kind="value",
            )
            target_counts = CoordinateRelation.point_map(
                self.source_domain,
                value_domain,
                (
                    (
                        tuple(
                            (
                                axis,
                                0,
                                self.source_domain.axis_count_expressions[axis],
                                1,
                            )
                            for axis in self.source_domain.axis_order
                        ),
                        (sympy.Integer(1),),
                    ),
                ),
            )
            return converse, target_counts
        separable_partition = self._separable_fixed_width_partition()
        if separable_partition is not None:
            partition_axes, _full_target_axes, fan_in = separable_partition
            partition_by_source = {
                source_axis: (target_axis, width)
                for source_axis, target_axis, width in partition_axes
            }
            converse = CoordinateRelation.point_map(
                self.target_domain,
                self.source_domain,
                (
                    (
                        tuple(
                            (
                                axis,
                                0,
                                self.target_domain.axis_count_expressions[axis],
                                1,
                            )
                            for axis in self.target_domain.axis_order
                        ),
                        tuple(
                            (
                                coordinate_axis_symbol(target_axis)
                                if width == 1
                                else sympy.floor(  # pyrefly: ignore[bad-argument-type]
                                    coordinate_axis_symbol(target_axis) / width  # pyrefly: ignore[unsupported-operation]
                                )
                            )
                            for target_axis, width in (
                                partition_by_source[source_axis]
                                for source_axis in self.source_domain.axis_order
                            )
                        ),
                    ),
                ),
            )
            value_axis = 0
            value_domain = CoordinateDomain(
                axis_order=(value_axis,),
                axis_counts_items=((value_axis, fan_in + 1),),
                kind="value",
            )
            target_counts = CoordinateRelation.point_map(
                self.source_domain,
                value_domain,
                (
                    (
                        tuple(
                            (
                                axis,
                                0,
                                self.source_domain.axis_count_expressions[axis],
                                1,
                            )
                            for axis in self.source_domain.axis_order
                        ),
                        (fan_in,),
                    ),
                ),
            )
            return converse, target_counts
        positional_product = self._parameterized_positional_product
        if positional_product is not None:
            positional_axes, residual = positional_product
            residual_converse = residual.converse()
            residual_target_counts = residual.target_count_by_source()
            if residual_converse is None or residual_target_counts is None:
                return None, None
            converse = _restore_positional_product(
                residual_converse,
                source_domain=self.target_domain,
                target_domain=self.source_domain,
                positional_axes=tuple(
                    (target_axis, source_axis)
                    for source_axis, target_axis in positional_axes
                ),
            )
            target_counts = residual_target_counts.lift_source(self.source_domain)
            if target_counts is None:
                return None, None
            return converse, target_counts
        if self.parameter_symbols:
            return None, None
        target_counts = self.target_count_by_source()
        converse = self._cached_converse
        if converse is not None and converse.is_single_valued():
            return converse, target_counts
        if target_counts is None:
            return converse, None
        return _derived_converse(self, target_counts) or converse, target_counts

    def _separable_fixed_width_partition(
        self,
    ) -> (
        tuple[
            tuple[tuple[int, int, int], ...],
            tuple[int, ...],
            sympy.Expr,
        ]
        | None
    ):
        """Prove an exact Cartesian fixed-width partition.

        Returns ``(partition_axes, full_target_axes, fan_in)``. Each
        ``(source_axis, target_axis, width)`` entry proves that one key owns
        exactly ``[width * key, width * key + width)`` on a producer axis.
        Every remaining producer axis is covered in full. This one structural
        certificate derives publication and arrival counts without sampling.
        """
        if not self.source_domain.axis_order or len(self.pieces) != 1:
            return None
        (piece,) = self.pieces
        if piece.source_bounds_items != tuple(
            (axis, 0, self.source_domain.axis_count_expressions[axis], 1)
            for axis in self.source_domain.axis_order
        ):
            return None
        partition_axes: list[tuple[int, int, int]] = []
        full_target_axes: list[int] = []
        used_source_axes: set[int] = set()
        fan_in: sympy.Expr = sympy.Integer(1)
        for target_axis, begin, end, step in piece.target_ranges:
            target_count = _integer_expression(
                self.target_domain.axis_count_expressions[target_axis],
                description="coordinate-domain axis count",
            )
            if (
                step == 1
                and sympy.simplify(begin) == 0
                and sympy.simplify(end - target_count) == 0  # pyrefly: ignore[unsupported-operation]
            ):
                full_target_axes.append(target_axis)
                fan_in *= target_count
                continue
            interval = _single_axis_interval(
                begin,
                end,
                domain=self.source_domain,
            )
            if step != 1 or interval is None:
                return None
            source_axis, stride, offset, width = interval
            if (
                source_axis in used_source_axes
                or offset != 0
                or width != stride
                or width <= 0
                or sympy.simplify(
                    target_count
                    - width
                    * _integer_expression(
                        self.source_domain.axis_count_expressions[source_axis],
                        description="coordinate-domain axis count",
                    )
                )
                != 0
            ):
                return None
            used_source_axes.add(source_axis)
            partition_axes.append((source_axis, target_axis, width))
            fan_in *= width
        if used_source_axes != set(self.source_domain.axis_order):
            return None
        return tuple(partition_axes), tuple(full_target_axes), sympy.simplify(fan_in)

    def _separable_fixed_width_point_quotient(
        self,
    ) -> tuple[tuple[int, int, int], ...] | None:
        """Prove a complete separable map ``item -> floor(item / C)``.

        The exact per-axis equations ``items == C * keys`` rule out tails.
        Source axes omitted from the target are repeated fibers. This is the
        consumer-side dual of :meth:`_separable_fixed_width_partition`.
        """
        if not self.target_domain.axis_order or len(self.pieces) != 1:
            return None
        (piece,) = self.pieces
        if piece.source_bounds_items != tuple(
            (axis, 0, self.source_domain.axis_count_expressions[axis], 1)
            for axis in self.source_domain.axis_order
        ):
            return None
        source_axes = {
            coordinate_axis_symbol(axis): axis for axis in self.source_domain.axis_order
        }
        used_source_axes: set[int] = set()
        quotient_axes: list[tuple[int, int, int]] = []
        for target_axis, begin, end, step in piece.target_ranges:
            if step != 1 or sympy.simplify(end - begin) != 1:  # pyrefly: ignore[unsupported-operation]
                return None
            expression_source_axes = tuple(
                source_axes[symbol]
                for symbol in begin.free_symbols
                if symbol in source_axes
            )
            if len(expression_source_axes) != 1:
                return None
            (source_axis,) = expression_source_axes
            if source_axis in used_source_axes:
                return None
            group_size = _single_ordinal_quotient_stride(
                begin,
                coordinate_axis_symbol(source_axis),
            )
            if (
                group_size is None
                or sympy.simplify(
                    self.source_domain.axis_count_expressions[source_axis]
                    - group_size
                    * self.target_domain.axis_count_expressions[target_axis]
                )
                != 0
            ):
                return None
            used_source_axes.add(source_axis)
            quotient_axes.append((source_axis, target_axis, group_size))
        return tuple(quotient_axes)

    @cached_property
    def _cached_converse(self) -> CoordinateRelation | None:
        pieces: list[_CoordinateRelationPiece] = []
        for piece in self.pieces:
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

            converse_source_bounds = {
                axis: [0, self.target_domain.axis_counts[axis], 1]
                for axis in self.target_domain.axis_order
            }
            for target_axis, begin, end, step in piece.target_ranges:
                if step != 1:
                    return None
                if begin.has(sympy.Mod):
                    begin = _simplify_logical_expression(
                        begin,
                        domain=self.source_domain,
                        source_bounds=piece.source_bounds_items,
                    )
                if end.has(sympy.Mod):
                    end = _simplify_logical_expression(
                        end,
                        domain=self.source_domain,
                        source_bounds=piece.source_bounds_items,
                    )
                target_count = self.target_domain.axis_counts[target_axis]
                if (
                    sympy.simplify(begin) == 0
                    and sympy.simplify(end - target_count)  # pyrefly: ignore[unsupported-operation]
                    == 0
                ):
                    continue
                if (
                    sympy.simplify(end - begin) == 1  # pyrefly: ignore[unsupported-operation]
                    and begin.has(sympy.Mod)
                    and len(begin.free_symbols) == 1
                ):
                    (source_symbol,) = begin.free_symbols
                    source_axis = next(
                        (
                            axis
                            for axis in self.source_domain.axis_order
                            if coordinate_axis_symbol(axis) == source_symbol
                        ),
                        None,
                    )
                    source_count = (
                        None
                        if source_axis is None
                        else self.source_domain.axis_counts[source_axis]
                    )
                    digit = (
                        None
                        if source_count is None
                        else _single_ordinal_digit(
                            begin,
                            source_symbol=source_symbol,
                            source_count=source_count,
                        )
                    )
                    source_bound = (
                        None
                        if source_axis is None
                        else next(
                            (
                                (source_begin, source_end, source_step)
                                for (
                                    axis,
                                    source_begin,
                                    source_end,
                                    source_step,
                                ) in piece.source_bounds_items
                                if axis == source_axis
                            ),
                            None,
                        )
                    )
                    if (
                        source_axis is not None
                        and source_count is not None
                        and digit is not None
                        and digit == (1, target_count)
                        and source_bound == (0, source_count, 1)
                        and not lower_bounds[source_axis]
                        and not upper_bounds[source_axis]
                    ):
                        target_coordinate = coordinate_axis_symbol(target_axis)
                        lower_bounds[source_axis].append(target_coordinate)
                        upper_bounds[source_axis].append(sympy.Integer(source_count))
                        target_steps[source_axis] = target_count
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
                        target_coordinate = coordinate_axis_symbol(target_axis)
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
                    converse_source_bounds[target_axis][0] = max(
                        converse_source_bounds[target_axis][0], int(begin)
                    )
                    converse_source_bounds[target_axis][1] = min(
                        converse_source_bounds[target_axis][1], int(end)
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
                converse_source_bounds[target_axis][0] = max(
                    converse_source_bounds[target_axis][0],
                    offset + source_begin * stride,
                )
                converse_source_bounds[target_axis][1] = min(
                    converse_source_bounds[target_axis][1],
                    offset + final_source * stride + width,
                )
                target_coordinate = coordinate_axis_symbol(target_axis)
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

            converse_source_bounds_items = tuple(
                (
                    axis,
                    converse_source_bounds[axis][0],
                    converse_source_bounds[axis][1],
                    converse_source_bounds[axis][2],
                )
                for axis in self.target_domain.axis_order
            )
            pieces.append(
                _CoordinateRelationPiece(
                    source_bounds_items=converse_source_bounds_items,
                    target_ranges=tuple(
                        (
                            axis,
                            _simplify_logical_expression(
                                (
                                    sympy.Max(*lower_bounds[axis])
                                    if lower_bounds[axis]
                                    else sympy.Integer(0)
                                ),
                                domain=self.target_domain,
                                source_bounds=converse_source_bounds_items,
                            ),
                            _simplify_logical_expression(
                                (
                                    sympy.Min(*upper_bounds[axis])
                                    if upper_bounds[axis]
                                    else sympy.Integer(
                                        self.source_domain.axis_counts[axis]
                                    )
                                ),
                                domain=self.target_domain,
                                source_bounds=converse_source_bounds_items,
                            ),
                            target_steps[axis],
                        )
                        for axis in self.source_domain.axis_order
                    ),
                )
            )
        return CoordinateRelation(
            source_domain=self.target_domain,
            target_domain=self.source_domain,
            pieces=tuple(dict.fromkeys(pieces)),
        )

    @staticmethod
    def _evaluate(expression: sympy.Expr, coordinates: dict[int, int]) -> int:
        value = expression.xreplace(
            {
                coordinate_axis_symbol(axis): sympy.Integer(coordinate)
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
        source_axis_order: tuple[int, ...] | None = None,
        target_axis_order: tuple[int, ...] | None = None,
    ) -> frozenset[int]:
        """Enumerate one source coordinate's targets for differential testing."""
        source_coordinates = self.source_domain.coordinates(
            source_index,
            linearization_order=source_axis_order,
        )
        return frozenset(
            self.target_domain.index(
                dict(zip(self.target_domain.axis_order, coordinates, strict=True)),
                linearization_order=target_axis_order,
            )
            for coordinates in self.target_coordinates(source_coordinates)
        )

    def materialize(
        self,
        *,
        source_axis_order: tuple[int, ...] | None = None,
        target_axis_order: tuple[int, ...] | None = None,
    ) -> tuple[frozenset[int], ...]:
        """Enumerate the relation only for tests and small-domain validation."""
        return tuple(
            self.targets(
                index,
                source_axis_order=source_axis_order,
                target_axis_order=target_axis_order,
            )
            for index in range(self.source_domain.size)
        )

    def union(self, other: CoordinateRelation) -> CoordinateRelation | None:
        """Return an exact union, retaining already-proved exact converses."""
        result = self._union_without_converse(other)
        if result is None or result is self or result is other:
            return result
        left_converse = _memoized_exact_converse(self)
        right_converse = _memoized_exact_converse(other)
        if left_converse is None or right_converse is None:
            return result
        converse = left_converse._union_without_converse(right_converse)
        if converse is not None:
            _remember_exact_converse(result, converse)
        return result

    def _union_without_converse(
        self,
        other: CoordinateRelation,
    ) -> CoordinateRelation | None:
        """Return an exact finite union without recursively propagating proof."""
        if (
            self.source_domain != other.source_domain
            or self.target_domain != other.target_domain
            or len(self.pieces) > _MAX_RELATION_PIECES
            or len(other.pieces) > _MAX_RELATION_PIECES
            or not _relation_product_is_within_budget(
                len(self.pieces), len(other.pieces)
            )
        ):
            return None
        if self.covers(other):
            return self
        if other.covers(self):
            return other
        pieces = dict.fromkeys(self.pieces)
        for piece in other.pieces:
            pieces.setdefault(piece, None)
            if len(pieces) > _MAX_RELATION_PIECES:
                return None
        return CoordinateRelation(
            source_domain=self.source_domain,
            target_domain=self.target_domain,
            pieces=tuple(pieces),
        )

    def coalesce_adjacent_target_boxes(
        self,
        *,
        prove_nonnegative: Callable[[sympy.Expr], bool] | None = None,
    ) -> CoordinateRelation:
        """Normalize adjacent target boxes within a bounded structural budget."""
        pieces = _coalesce_adjacent_target_boxes(
            self.pieces,
            source_domain=self.source_domain,
            prove_nonnegative=prove_nonnegative,
        )
        if pieces == self.pieces:
            return self
        return CoordinateRelation(
            source_domain=self.source_domain,
            target_domain=self.target_domain,
            pieces=pieces,
        )

    def coalesce_adjacent_source_boxes(
        self,
        *,
        fold_static_offsets: bool = False,
    ) -> CoordinateRelation:
        """Merge adjacent unit-stride source boxes into one exact point map.

        Identical mappings can always share one larger source box.  Two
        equal-width adjacent boxes whose point expressions differ by a static
        integer offset can also be represented exactly: select that offset
        with the quotient of the varying source coordinate.  This keeps
        mixed-radix permutations compact without enumerating their repeated
        residues after flattening.
        """
        if len(self.pieces) > _MAX_RELATION_PIECES or not (
            _relation_product_is_within_budget(
                len(self.pieces),
                len(self.pieces),
                max(1, len(self.source_domain.axis_order)),
            )
        ):
            # Returning the original relation is an exact conservative
            # decline; callers may reject it if compactness is required.
            return self

        def piece_key(piece: _CoordinateRelationPiece) -> tuple[object, ...]:
            return (
                tuple(
                    (axis, sympy.srepr(begin), sympy.srepr(end), step)
                    for axis, begin, end, step in piece.source_bounds_items
                ),
                tuple(
                    (axis, sympy.srepr(begin), sympy.srepr(end), step)
                    for axis, begin, end, step in piece.target_ranges
                ),
            )

        def merge_pair(
            lower: _CoordinateRelationPiece,
            upper: _CoordinateRelationPiece,
            axis: int,
        ) -> _CoordinateRelationPiece | None:
            lower_bounds = {
                bound_axis: (begin, end, step)
                for bound_axis, begin, end, step in lower.source_bounds_items
            }
            upper_bounds = {
                bound_axis: (begin, end, step)
                for bound_axis, begin, end, step in upper.source_bounds_items
            }
            lower_begin, lower_end, lower_step = lower_bounds[axis]
            upper_begin, upper_end, upper_step = upper_bounds[axis]
            if lower_step != 1 or upper_step != 1 or lower_end != upper_begin:
                return None
            merged_targets = lower.target_ranges
            if lower.target_ranges != upper.target_ranges:
                lower_width = lower_end - lower_begin
                if (
                    not fold_static_offsets
                    or lower_width <= 0
                    or lower_width != upper_end - upper_begin
                ):
                    return None
                source_symbol = coordinate_axis_symbol(axis)
                quotient = sympy.floor(  # pyrefly: ignore[bad-argument-type]
                    (source_symbol - lower_begin)  # pyrefly: ignore[unsupported-operation]
                    / lower_width
                )
                local_coordinate = lower_begin + sympy.Mod(  # pyrefly: ignore[unsupported-operation]
                    source_symbol - lower_begin,  # pyrefly: ignore[unsupported-operation]
                    lower_width,
                )
                corresponding_upper = source_symbol + lower_width  # pyrefly: ignore[unsupported-operation]
                folded_targets: list[tuple[int, sympy.Expr, sympy.Expr, int]] = []
                for lower_target, upper_target in zip(
                    lower.target_ranges,
                    upper.target_ranges,
                    strict=True,
                ):
                    lower_axis, lower_value, lower_end_value, lower_target_step = (
                        lower_target
                    )
                    upper_axis, upper_value, upper_end_value, upper_target_step = (
                        upper_target
                    )
                    value_delta = sympy.simplify(
                        upper_value.xreplace({source_symbol: corresponding_upper})  # pyrefly: ignore[unsupported-operation]
                        - lower_value  # pyrefly: ignore[unsupported-operation]
                    )
                    if (
                        lower_axis != upper_axis
                        or lower_target_step != 1
                        or upper_target_step != 1
                        or sympy.simplify(lower_end_value - lower_value) != 1  # pyrefly: ignore[unsupported-operation]
                        or sympy.simplify(upper_end_value - upper_value) != 1  # pyrefly: ignore[unsupported-operation]
                        or value_delta.free_symbols
                        or value_delta.is_integer is not True
                    ):
                        return None
                    folded_value = sympy.simplify(
                        lower_value.xreplace({source_symbol: local_coordinate})
                        + value_delta * quotient
                    )
                    folded_targets.append(
                        (
                            lower_axis,
                            folded_value,
                            folded_value + 1,
                            1,
                        )
                    )
                merged_targets = tuple(folded_targets)
            return _CoordinateRelationPiece(
                source_bounds_items=tuple(
                    (
                        (bound_axis, lower_begin, upper_end, 1)
                        if bound_axis == axis
                        else (bound_axis, *lower_bounds[bound_axis])
                    )
                    for bound_axis in self.source_domain.axis_order
                ),
                target_ranges=merged_targets,
            )

        pieces = sorted(dict.fromkeys(self.pieces), key=piece_key)
        while True:
            for axis in self.source_domain.axis_order:
                groups: dict[
                    tuple[tuple[int, int, int, int], ...],
                    list[_CoordinateRelationPiece],
                ] = {}
                unmergeable: list[_CoordinateRelationPiece] = []
                for piece in pieces:
                    axis_bound = next(
                        bound for bound in piece.source_bounds_items if bound[0] == axis
                    )
                    if axis_bound[3] != 1:
                        unmergeable.append(piece)
                        continue
                    group = tuple(
                        bound for bound in piece.source_bounds_items if bound[0] != axis
                    )
                    groups.setdefault(group, []).append(piece)

                next_pieces = list(unmergeable)
                merged = False
                for group_pieces in groups.values():
                    ordered = sorted(group_pieces, key=piece_key)
                    current = ordered[0]
                    for candidate in ordered[1:]:
                        combined = merge_pair(current, candidate, axis)
                        if combined is None:
                            next_pieces.append(current)
                            current = candidate
                        else:
                            current = combined
                            merged = True
                    next_pieces.append(current)
                if merged:
                    pieces = sorted(next_pieces, key=piece_key)
                    break
            else:
                break
        return CoordinateRelation(
            source_domain=self.source_domain,
            target_domain=self.target_domain,
            pieces=tuple(pieces),
        )

    def has_disjoint_source_support(self, other: CoordinateRelation) -> bool:
        """Prove that no source coordinate participates in both relations."""
        if (
            self.source_domain != other.source_domain
            or len(self.pieces) > _MAX_RELATION_PIECES
            or len(other.pieces) > _MAX_RELATION_PIECES
            or not _relation_product_is_within_budget(
                len(self.pieces), len(other.pieces)
            )
        ):
            return False
        if _ordinalized_source_supports_are_disjoint(
            self._ordinalized_source_support,
            other._ordinalized_source_support,
        ):
            return True
        return all(
            _source_boxes_are_disjoint(left, right)
            for left in self.pieces
            for right in other.pieces
        )

    def source_support_cardinality(self) -> int | sympy.Expr | None:
        """Return the exact number of source points where this relation is defined.

        Relation source domains are intentionally allowed to be larger than a
        relation's support.  Static worker schedules use that distinction to
        share one global ``(launch stage, worker, wave)`` domain while each
        root owns only some of its slots.  Count the disjoint canonical source
        cells rather than assuming that ``source_domain.size`` is the number
        of mapped tasks.

        This operation is structural: its cost depends on relation pieces,
        never on the number of source points.  Strided or overlapping support
        that cannot be canonicalized exactly declines instead of enumerating.
        """
        if self.target_domain.size_expr.is_zero is True:
            return 0
        if len(
            self.pieces
        ) <= _MAX_RELATION_PIECES and _relation_product_is_within_budget(
            len(self.pieces), len(self.pieces)
        ):
            piece_cardinalities = tuple(
                _source_box_cardinality(
                    piece.source_bounds_items,
                    domain=self.source_domain,
                )
                for piece in self.pieces
            )
            if (
                all(cardinality is not None for cardinality in piece_cardinalities)
                and all(
                    cardinality == 0
                    or _target_box_is_nonempty_for_all_sources(
                        piece.target_ranges,
                        source_domain=self.source_domain,
                        source_bounds=piece.source_bounds_items,
                        target_domain=self.target_domain,
                    )
                    for piece, cardinality in zip(
                        self.pieces,
                        piece_cardinalities,
                        strict=True,
                    )
                )
                and all(
                    _source_boxes_are_disjoint(left, right)
                    for index, left in enumerate(self.pieces)
                    for right in self.pieces[index + 1 :]
                )
            ):
                cardinality = sympy.simplify(
                    sum(
                        cast("sympy.Expr", piece_cardinality)
                        for piece_cardinality in piece_cardinalities
                    )
                )
                return (
                    int(cardinality)
                    if isinstance(cardinality, sympy.Integer)
                    else cardinality
                )
        converse = _memoized_exact_converse(self)
        if (
            converse is not None
            and self.is_single_valued()
            and converse.is_single_valued()
            and _source_boxes_partition_domain(
                tuple(piece.source_bounds_items for piece in converse.pieces),
                converse.source_domain,
            )
            and all(
                _target_point_is_in_domain(
                    piece.target_ranges,
                    source_domain=converse.source_domain,
                    source_bounds=piece.source_bounds_items,
                    target_domain=converse.target_domain,
                )
                for piece in converse.pieces
            )
        ):
            target_size = sympy.simplify(self.target_domain.size_expr)
            return (
                int(target_size)
                if isinstance(target_size, sympy.Integer)
                else target_size
            )
        if self._factored_source_support_converse is not None:
            target_size = sympy.simplify(self.target_domain.size_expr)
            return (
                int(target_size)
                if isinstance(target_size, sympy.Integer)
                else target_size
            )
        if self.parameter_symbols:
            # Symbolic overlap normalization requires an ordering proof for
            # every source cut.  Decline instead of sampling a runtime size.
            return None
        canonical = self.canonical_single_valued()
        if canonical is None:
            return None
        cardinality = 0
        for piece in canonical.pieces:
            if not _target_box_is_nonempty_for_all_sources(
                piece.target_ranges,
                source_domain=canonical.source_domain,
                source_bounds=piece.source_bounds_items,
                target_domain=canonical.target_domain,
            ):
                return None
            piece_cardinality = 1
            for axis, begin, end, step in piece.source_bounds_items:
                count = canonical.source_domain.axis_counts[axis]
                if begin < 0 or end > count or begin >= end:
                    return None
                piece_cardinality *= len(range(begin, end, step))
            cardinality += piece_cardinality
        return cardinality

    def is_total(self) -> bool:
        """Return whether one canonical piece covers the complete product."""
        if len(self.pieces) != 1:
            return False
        (piece,) = self.pieces
        return piece.source_bounds_items == tuple(
            (axis, 0, self.source_domain.axis_count_expressions[axis], 1)
            for axis in self.source_domain.axis_order
        ) and piece.target_ranges == tuple(
            (
                axis,
                sympy.Integer(0),
                _integer_expression(
                    self.target_domain.axis_count_expressions[axis],
                    description="coordinate-domain axis count",
                ),
                1,
            )
            for axis in self.target_domain.axis_order
        )

    def has_total_source(self) -> bool:
        """Return whether every source coordinate has at least one target."""
        if self.source_domain.size_expr.is_zero is True:
            return True
        cardinality = self.source_support_cardinality()
        if (
            cardinality is not None
            and sympy.simplify(
                cardinality - self.source_domain.size_expr  # pyrefly: ignore[unsupported-operation]
            )
            == 0
        ):
            # Support is a proved subset of the source domain. Equal finite
            # cardinality therefore proves coverage, including symbolic tails.
            return True
        converse = _memoized_exact_converse(self)
        if converse is not None and self.is_single_valued():
            converse_cardinality = converse.source_support_cardinality()
            if (
                converse_cardinality is not None
                and _integer_partition_expressions_equal(
                    converse_cardinality,
                    self.source_domain.size_expr,
                )
            ):
                # ``self`` being single-valued makes its exact converse
                # injective.  Its full-cardinality support therefore maps
                # onto every point of this finite source domain.
                return True
        positional_product = self._parameterized_positional_product
        if positional_product is not None:
            _positional_axes, residual = positional_product
            return residual.has_total_source()
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
        if len(self.pieces) > _MAX_RELATION_PIECES:
            return False
        if all(
            step == 1
            and _integer_partition_expressions_equal(end - begin, 1)
            for piece in self.pieces
            for _axis, begin, end, step in piece.target_ranges
        ) and (
            len(self.pieces) <= 1
            or (
                _relation_product_is_within_budget(
                    len(self.pieces),
                    len(self.pieces),
                )
                and all(
                    _source_boxes_are_disjoint(left, right)
                    for index, left in enumerate(self.pieces)
                    for right in self.pieces[index + 1 :]
                )
            )
        ):
            return True
        positional_product = self._parameterized_positional_product
        if positional_product is not None:
            _positional_axes, residual = positional_product
            return residual.is_single_valued()
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
        if len(self.pieces) <= 1 or (
            normalized_targets
            and all(
                target_ranges == normalized_targets[0]
                for target_ranges in normalized_targets[1:]
            )
        ):
            return True
        if _source_boxes_partition_domain(
            tuple(piece.source_bounds_items for piece in self.pieces),
            self.source_domain,
        ):
            return True
        if not _relation_product_is_within_budget(len(self.pieces), len(self.pieces)):
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

    def canonical_single_valued(self) -> CoordinateRelation | None:
        """Return a disjoint-source form for an at-most-one-valued relation.

        The transformation partitions only at the constant boundaries already
        present in relation pieces.  Its cost therefore depends on relation
        complexity rather than the number of runtime source instances.  A
        strided source guard or two different values on an overlapping source
        region is rejected instead of being expanded.
        """
        return self._cached_canonical_single_valued

    @cached_property
    def _cached_canonical_single_valued(self) -> CoordinateRelation | None:
        if len(self.pieces) == 1:
            (piece,) = self.pieces
            is_full_source = piece.source_bounds_items == tuple(
                (axis, 0, self.source_domain.axis_count_expressions[axis], 1)
                for axis in self.source_domain.axis_order
            )
            is_symbolic_partial_source = bool(
                self.parameter_symbols
            ) and _source_bounds_are_symbolically_within_domain(
                piece.source_bounds_items,
                self.source_domain,
            )
            is_point_map = all(
                step == 1 and sympy.simplify(end - begin) == 1  # pyrefly: ignore[unsupported-operation]
                for _axis, begin, end, step in piece.target_ranges
            )
            if is_full_source and is_point_map:
                return self
            if (
                is_symbolic_partial_source
                and is_point_map
                and _target_point_is_in_domain(
                    piece.target_ranges,
                    source_domain=self.source_domain,
                    source_bounds=piece.source_bounds_items,
                    target_domain=self.target_domain,
                )
            ):
                return self
        positional_product = self._parameterized_positional_product
        if positional_product is not None:
            positional_axes, residual = positional_product
            canonical_residual = residual.canonical_single_valued()
            if canonical_residual is None:
                return None
            return _restore_positional_product(
                canonical_residual,
                source_domain=self.source_domain,
                target_domain=self.target_domain,
                positional_axes=positional_axes,
            )
        if _source_boxes_partition_domain(
            tuple(piece.source_bounds_items for piece in self.pieces),
            self.source_domain,
        ) and all(
            _target_point_is_in_domain(
                piece.target_ranges,
                source_domain=self.source_domain,
                source_bounds=piece.source_bounds_items,
                target_domain=self.target_domain,
            )
            for piece in self.pieces
        ):
            return self
        cells = _relation_source_cells(self)
        if cells is None:
            return None
        pieces: list[_CoordinateRelationPiece] = []
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
                _CoordinateRelationPiece(
                    source_bounds_items=bounds,
                    target_ranges=target_ranges,
                )
            )
        return CoordinateRelation(
            source_domain=self.source_domain,
            target_domain=self.target_domain,
            pieces=tuple(pieces),
        )

    def is_total_function(self) -> bool:
        """Return whether every source instance maps to exactly one target."""
        if self.is_positional_bijection():
            return True
        if self._separable_fixed_width_point_quotient() is not None:
            return True
        if (
            self.parameter_symbols
            and len(self.source_domain.axis_order) == 1
            and _symbolic_single_source_mixed_radix_converse(self) is not None
        ):
            return True
        positional_product = self._parameterized_positional_product
        if positional_product is not None:
            _positional_axes, residual = positional_product
            return residual.is_total_function()
        if len(self.pieces) == 1:
            (piece,) = self.pieces
            if (
                piece.source_bounds_items
                == tuple(
                    (axis, 0, self.source_domain.axis_count_expressions[axis], 1)
                    for axis in self.source_domain.axis_order
                )
                and not self.target_domain.parameter_symbols
                and all(
                    not begin.free_symbols and not end.free_symbols
                    for _axis, begin, end, _step in piece.target_ranges
                )
                and _target_point_is_in_domain(
                    piece.target_ranges,
                    source_domain=self.source_domain,
                    source_bounds=piece.source_bounds_items,
                    target_domain=self.target_domain,
                )
            ):
                return True
        source_boxes = tuple(piece.source_bounds_items for piece in self.pieces)
        if _source_boxes_partition_domain(source_boxes, self.source_domain) and all(
            _target_point_is_in_domain(
                piece.target_ranges,
                source_domain=self.source_domain,
                source_bounds=piece.source_bounds_items,
                target_domain=self.target_domain,
            )
            for piece in self.pieces
        ):
            return True
        if self.is_single_valued() and self.has_total_source():
            return True
        if self.parameter_symbols:
            return False
        if (
            all(
                _source_bounds_are_within_domain(
                    piece.source_bounds_items,
                    self.source_domain,
                )
                for piece in self.pieces
            )
            and all(
                _source_boxes_are_disjoint(left, right)
                for index, left in enumerate(self.pieces)
                for right in self.pieces[index + 1 :]
            )
            and sum(
                math.prod(
                    len(range(begin, end, step))
                    for _axis, begin, end, step in piece.source_bounds_items
                )
                for piece in self.pieces
            )
            == self.source_domain.size
            and all(
                _target_point_is_in_domain(
                    piece.target_ranges,
                    source_domain=self.source_domain,
                    source_bounds=piece.source_bounds_items,
                    target_domain=self.target_domain,
                )
                for piece in self.pieces
            )
        ):
            return True
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

    def is_bijection_from_source_support(self) -> bool:
        """Prove that this relation bijects its support onto its target domain.

        The source relation may be partial: only participating source points
        are counted.  Exact cardinality, point-valuedness, and a total inverse
        are all required, so equal source/target counts alone cannot certify a
        duplicate-target map.
        """
        converse = _memoized_exact_converse(self)
        if (
            converse is not None
            and self.is_single_valued()
            and converse.is_total_function()
        ):
            return True
        if converse is None and self._factored_source_support_converse is not None:
            # The factorization constructs an exact total inverse through the
            # dense ordinal of this relation's semantic support.
            return True
        source_cardinality = self.source_support_cardinality()
        if (
            source_cardinality is None
            or not _integer_partition_expressions_equal(
                source_cardinality,
                self.target_domain.size_expr,
            )
            or not self.is_single_valued()
        ):
            return False
        if self.target_domain.size_expr.is_zero is True:
            return True
        if converse is None:
            converse = self.converse()
        return converse is not None and converse.is_total_function()

    def _pointwise_difference_bounds(
        self,
        other: CoordinateRelation,
        *,
        require_total_self: bool = True,
    ) -> tuple[tuple[sympy.Expr, sympy.Expr], ...] | None:
        """Bound ``self(source) - other(source)`` on a common partition."""
        if (
            self.source_domain != other.source_domain
            or len(self.target_domain.axis_order) != 1
            or len(other.target_domain.axis_order) != 1
        ):
            return None
        left = self.canonical_single_valued()
        right = other.canonical_single_valued()
        if (
            left is None
            or right is None
            or (require_total_self and not left.is_total_function())
            or not right.is_total_function()
        ):
            return None

        cells = _relations_source_cells((left, right), include_domain=True)
        if cells is None:
            return None
        result: list[tuple[sympy.Expr, sympy.Expr]] = []
        for bounds in cells:
            left_pieces = tuple(
                piece
                for piece in left.pieces
                if _source_box_covers(piece.source_bounds_items, bounds)
            )
            right_pieces = tuple(
                piece
                for piece in right.pieces
                if _source_box_covers(piece.source_bounds_items, bounds)
            )
            if not left_pieces and not require_total_self:
                continue
            if len(left_pieces) != 1 or len(right_pieces) != 1:
                return None
            left_range = left_pieces[0].target_ranges
            right_range = right_pieces[0].target_ranges
            if len(left_range) != 1 or len(right_range) != 1:
                return None
            _left_axis, left_begin, left_end, left_step = left_range[0]
            _right_axis, right_begin, right_end, right_step = right_range[0]
            if (
                left_step != 1
                or right_step != 1
                or sympy.simplify(left_end - left_begin) != 1  # pyrefly: ignore[unsupported-operation]
                or sympy.simplify(right_end - right_begin) != 1  # pyrefly: ignore[unsupported-operation]
            ):
                return None
            difference = _simplify_logical_expression(
                sympy.simplify(left_begin - right_begin),  # pyrefly: ignore[unsupported-operation]
                domain=self.source_domain,
                source_bounds=bounds,
            )
            difference_bounds = _logical_expression_bounds(
                difference,
                domain=self.source_domain,
                source_bounds=bounds,
            )
            if difference_bounds is None:
                return None
            result.append(tuple(sympy.simplify(value) for value in difference_bounds))
        return tuple(result)

    def is_pointwise_strictly_less_than(
        self,
        other: CoordinateRelation,
    ) -> bool:
        """Prove ``self(source) < other(source)`` without enumeration.

        Both operands must be total scalar functions over the same source
        domain. A partial relation is deliberately rejected: callers must
        prove every prerequisite independently before combining frontiers.
        """
        bounds = self._pointwise_difference_bounds(other)
        return bounds is not None and all(
            not upper.free_symbols
            and upper.is_integer is True  # pyrefly: ignore[missing-attribute]
            and int(upper) <= -1  # pyrefly: ignore[bad-argument-type]
            for _lower, upper in bounds
        )

    def is_pointwise_strictly_less_than_where_defined(
        self,
        other: CoordinateRelation,
    ) -> bool:
        """Prove strict order on this scalar function's exact source support."""
        bounds = self._pointwise_difference_bounds(
            other,
            require_total_self=False,
        )
        return bool(bounds) and all(
            not upper.free_symbols
            and upper.is_integer is True  # pyrefly: ignore[missing-attribute]
            and int(upper) <= -1  # pyrefly: ignore[bad-argument-type]
            for _lower, upper in bounds
        )

    def is_pointwise_equal_to(self, other: CoordinateRelation) -> bool:
        """Prove equality of two total scalar functions without enumeration."""
        if self.target_domain != other.target_domain:
            return False
        bounds = self._pointwise_difference_bounds(other)
        return bounds is not None and all(
            not lower.free_symbols
            and not upper.free_symbols
            and lower.is_integer is True  # pyrefly: ignore[missing-attribute]
            and upper.is_integer is True  # pyrefly: ignore[missing-attribute]
            and int(lower) == 0  # pyrefly: ignore[bad-argument-type]
            and int(upper) == 0  # pyrefly: ignore[bad-argument-type]
            for lower, upper in bounds
        )

    def is_pointwise_equal_on_same_support(
        self,
        other: CoordinateRelation,
    ) -> bool:
        """Prove equal point maps on the same exact source-box partition.

        Unlike :meth:`is_pointwise_equal_to`, this operation accepts partial
        and multidimensional point maps.  It is intentionally conservative:
        source bounds may be reordered or duplicated, but the two relations
        must expose the same symbolically normalized boxes.  Proving equality
        across two differently partitioned symbolic supports would require an
        ordering of runtime cuts, so that case declines rather than sampling.
        """
        if (
            self.source_domain != other.source_domain
            or self.target_domain != other.target_domain
            or len(self.pieces) > _MAX_RELATION_PIECES
            or len(other.pieces) > _MAX_RELATION_PIECES
        ):
            return False

        def normalized_pieces(
            relation: CoordinateRelation,
        ) -> (
            dict[
                tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...],
                tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...],
            ]
            | None
        ):
            result: dict[
                tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...],
                tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...],
            ] = {}
            for piece in relation.pieces:
                source_bounds = tuple(
                    (axis, sympy.simplify(begin), sympy.simplify(end), step)
                    for axis, begin, end, step in piece.source_bounds_items
                )
                target_ranges = tuple(
                    (
                        axis,
                        _simplify_logical_expression(
                            begin,
                            domain=relation.source_domain,
                            source_bounds=source_bounds,
                        ),
                        _simplify_logical_expression(
                            end,
                            domain=relation.source_domain,
                            source_bounds=source_bounds,
                        ),
                        step,
                    )
                    for axis, begin, end, step in piece.target_ranges
                )
                previous = result.setdefault(source_bounds, target_ranges)
                if previous != target_ranges:
                    return None
            return result

        left_pieces = normalized_pieces(self)
        right_pieces = normalized_pieces(other)
        if (
            left_pieces is None
            or right_pieces is None
            or left_pieces.keys() != right_pieces.keys()
        ):
            return False
        return all(
            all(
                left_axis == right_axis
                and left_step == right_step
                and _integer_partition_expressions_equal(left_begin, right_begin)
                and _integer_partition_expressions_equal(left_end, right_end)
                for (
                    left_axis,
                    left_begin,
                    left_end,
                    left_step,
                ), (
                    right_axis,
                    right_begin,
                    right_end,
                    right_step,
                ) in zip(
                    left_pieces[source_bounds],
                    right_pieces[source_bounds],
                    strict=True,
                )
            )
            for source_bounds in left_pieces
        )

    def is_positional_bijection(self) -> bool:
        """Return whether coordinates are renamed position-for-position."""
        if (
            len(self.source_domain.axis_order) != len(self.target_domain.axis_order)
            or any(
                sympy.simplify(left - right) != 0
                for left, right in zip(
                    self.source_domain.shape_expr,
                    self.target_domain.shape_expr,
                    strict=True,
                )
            )
            or len(self.pieces) != 1
        ):
            return False
        (piece,) = self.pieces
        if piece.source_bounds_items != tuple(
            (axis, 0, self.source_domain.axis_count_expressions[axis], 1)
            for axis in self.source_domain.axis_order
        ):
            return False
        return all(
            target_axis == expected_target_axis
            and step == 1
            and sympy.simplify(begin - coordinate_axis_symbol(source_axis)) == 0  # pyrefly: ignore[unsupported-operation]
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

    def target_count_by_source(self) -> CoordinateRelation | None:
        """Return the exact number of distinct targets for every source.

        The result is another single-valued ``CoordinateRelation`` whose one
        target coordinate is the cardinality.  This keeps aggregation inside
        the relation algebra while avoiding a separate scalar-expression IR.
        Source boxes are partitioned only at existing structural boundaries.
        Overlapping target boxes must be identical or provably disjoint.
        """
        positional_product = self._parameterized_positional_product
        if positional_product is not None:
            _positional_axes, residual = positional_product
            residual_counts = residual.target_count_by_source()
            return (
                None
                if residual_counts is None
                else residual_counts.lift_source(self.source_domain)
            )
        cells = _relation_source_cells(self, include_domain=True)
        if cells is None:
            return None
        value_axis = 0
        value_domain = CoordinateDomain(
            axis_order=(value_axis,),
            axis_counts_items=((value_axis, self.target_domain.size + 1),),
            kind="value",
        )
        pieces: list[_CoordinateRelationPiece] = []
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
                _CoordinateRelationPiece(
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
        return CoordinateRelation(
            source_domain=self.source_domain,
            target_domain=value_domain,
            pieces=tuple(pieces),
        )

    def max_target_value_by_source(
        self,
        values: CoordinateRelation,
    ) -> CoordinateRelation | None:
        """Find the maximum mapped value among each source's targets.

        ``self`` maps source coordinates to a set of target coordinates and
        ``values`` maps those target coordinates to one scalar value.  The
        result maps every source with targets to its maximum value without
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
            list[_CoordinateRelationPiece],
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
        pieces: list[_CoordinateRelationPiece] = []
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
            maximum = _max_target_value_expression(
                tuple(maxima),
                source_domain=self.source_domain,
                source_bounds=source_bounds,
            )
            pieces.append(
                _CoordinateRelationPiece(
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
        return CoordinateRelation(
            source_domain=self.source_domain,
            target_domain=values.target_domain,
            pieces=tuple(pieces),
        )

    def enumerate_targets_by_source(self) -> CoordinateRelation | None:
        """Enumerate uniform rectangular target sets by source coordinate.

        The returned relation maps ``(target_index, source coordinates)`` to
        one target coordinate.  It is a compact bijection when this relation's
        target boxes are disjoint and every source has the same static target
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
        targets_per_source: int | None = None
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
            if not total or (
                targets_per_source is not None and total != targets_per_source
            ):
                return None
            targets_per_source = total
            cell_targets.append((bounds, tuple(boxes)))
        if targets_per_source is None:
            return None

        all_axes = {
            *self.source_domain.axis_order,
            *self.target_domain.axis_order,
        }
        target_index_axis = min(all_axes, default=0) - 1
        enumeration_domain = CoordinateDomain(
            axis_order=(target_index_axis, *self.source_domain.axis_order),
            axis_counts_items=(
                (target_index_axis, targets_per_source),
                *self.source_domain.axis_counts_items,
            ),
            kind="task_order",
        )
        target_index = coordinate_axis_symbol(target_index_axis)
        pieces: list[_CoordinateRelationPiece] = []
        for source_bounds, target_boxes in cell_targets:
            target_index_begin = 0
            for target_ranges, count in target_boxes:
                local = target_index - target_index_begin  # pyrefly: ignore[unsupported-operation]
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
                    _CoordinateRelationPiece(
                        source_bounds_items=(
                            (
                                target_index_axis,
                                target_index_begin,
                                target_index_begin + count,
                                1,
                            ),
                            *source_bounds,
                        ),
                        target_ranges=tuple(target_points),
                    )
                )
                target_index_begin += count
            if target_index_begin != targets_per_source:
                return None
        return CoordinateRelation(
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
            coordinate_axis_symbol(axis): sympy.Integer(coordinate)
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
        other: CoordinateRelation,
        *,
        prove_nonnegative: Callable[[sympy.Expr], bool] | None = None,
    ) -> CoordinateRelation | None:
        """Relate ``other`` sources to ``self`` sources by target overlap.

        This is the dependency composition used by memory accesses: ``self``
        maps producer instances to allocation coordinates and ``other`` maps
        consumer instances to the same allocation coordinates.  Unsupported
        relation shapes decline instead of expanding either source domain.
        """
        if self.target_domain != other.target_domain:
            return None
        target_counts = self.source_domain.axis_count_expressions
        pieces: list[_CoordinateRelationPiece] = []
        for producer_piece in self.pieces:
            full_producer_bounds = tuple(
                (axis, 0, self.source_domain.axis_count_expressions[axis], 1)
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
                    allocation_count = self.target_domain.axis_count_expressions[
                        allocation_axis
                    ]
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
                target_ranges = tuple(
                    (
                        axis,
                        _simplify_logical_expression(
                            (
                                sympy.Max(*lower_bounds[axis])
                                if lower_bounds[axis]
                                else sympy.Integer(0)
                            ),
                            domain=other.source_domain,
                            source_bounds=consumer_piece.source_bounds_items,
                        ),
                        _simplify_logical_expression(
                            (
                                sympy.Min(*upper_bounds[axis])
                                if upper_bounds[axis]
                                else _integer_expression(
                                    target_counts[axis],
                                    description="coordinate-domain axis count",
                                )
                            ),
                            domain=other.source_domain,
                            source_bounds=consumer_piece.source_bounds_items,
                        ),
                        1,
                    )
                    for axis in self.source_domain.axis_order
                )
                outside_target_domain = False
                for axis, begin, end, _step in target_ranges:
                    begin_bounds = _logical_expression_bounds(
                        begin,
                        domain=other.source_domain,
                        source_bounds=consumer_piece.source_bounds_items,
                    )
                    end_bounds = _logical_expression_bounds(
                        end,
                        domain=other.source_domain,
                        source_bounds=consumer_piece.source_bounds_items,
                    )
                    width_bounds = _logical_expression_bounds(
                        end - begin,  # pyrefly: ignore[unsupported-operation]
                        domain=other.source_domain,
                        source_bounds=consumer_piece.source_bounds_items,
                    )
                    target_count = _integer_expression(
                        target_counts[axis],
                        description="coordinate-domain axis count",
                    )
                    if (
                        (
                            end_bounds is not None
                            and _is_provably_nonnegative(
                                -end_bounds[1],  # pyrefly: ignore[unsupported-operation]
                                prove_nonnegative,
                            )
                        )
                        or (
                            begin_bounds is not None
                            and _is_provably_nonnegative(
                                begin_bounds[0] - target_count,  # pyrefly: ignore[unsupported-operation]
                                prove_nonnegative,
                            )
                        )
                        or (
                            width_bounds is not None
                            and _is_provably_nonnegative(
                                -width_bounds[1],  # pyrefly: ignore[unsupported-operation]
                                prove_nonnegative,
                            )
                        )
                    ):
                        outside_target_domain = True
                        break
                if outside_target_domain:
                    continue
                pieces.append(
                    _CoordinateRelationPiece(
                        source_bounds_items=consumer_piece.source_bounds_items,
                        target_ranges=target_ranges,
                    )
                )
        return CoordinateRelation(
            source_domain=other.source_domain,
            target_domain=self.source_domain,
            pieces=tuple(pieces),
        ).coalesce_adjacent_target_boxes(prove_nonnegative=prove_nonnegative)


def _source_boxes_are_disjoint(
    left: _CoordinateRelationPiece,
    right: _CoordinateRelationPiece,
) -> bool:
    """Prove two concrete strided source boxes have no common point."""
    return _source_bounds_are_disjoint(
        left.source_bounds_items,
        right.source_bounds_items,
    )


def _source_bounds_are_within_domain(
    bounds: tuple[tuple[int, int, int, int], ...],
    domain: CoordinateDomain,
) -> bool:
    """Return whether a concrete source box is wholly inside its domain."""
    return tuple(
        axis for axis, _begin, _end, _step in bounds
    ) == domain.axis_order and all(
        0 <= begin <= end <= domain.axis_counts[axis] and step > 0
        for axis, begin, end, step in bounds
    )


def _source_bounds_are_symbolically_within_domain(
    bounds: tuple[tuple[int, IntegerExpression, IntegerExpression, int], ...],
    domain: CoordinateDomain,
) -> bool:
    """Prove a unit-stride symbolic source box lies inside its domain."""
    if tuple(axis for axis, _begin, _end, _step in bounds) != domain.axis_order:
        return False
    counts = domain.axis_count_expressions
    for axis, raw_begin, raw_end, step in bounds:
        begin = _integer_expression(raw_begin, description="relation source bound")
        end = _integer_expression(raw_end, description="relation source bound")
        count = _integer_expression(
            counts[axis],
            description="coordinate-domain axis count",
        )
        if (
            step != 1
            or not _is_provably_nonnegative(begin, None)
            or not _is_provably_nonnegative(sympy.simplify(end - begin), None)
            or not _is_provably_nonnegative(sympy.simplify(count - end), None)
        ):
            return False
    return True


def _source_box_cardinality(
    bounds: tuple[tuple[int, IntegerExpression, IntegerExpression, int], ...],
    *,
    domain: CoordinateDomain,
) -> sympy.Expr | None:
    """Return one source box's exact cardinality when its bounds are proved.

    Source strides are compile-time positive integers, so a symbolic width has
    the ordinary nonnegative ceil-div cardinality.  The proof deliberately
    requires the whole box to lie inside its domain; clipping a symbolic box
    would introduce a piecewise support that this representation has not
    normalized.
    """
    if tuple(axis for axis, _begin, _end, _step in bounds) != domain.axis_order:
        return None
    cardinality: sympy.Expr = sympy.Integer(1)
    for axis, raw_begin, raw_end, step in bounds:
        begin = _integer_expression(raw_begin, description="relation source bound")
        end = _integer_expression(raw_end, description="relation source bound")
        count = _integer_expression(
            domain.axis_count_expressions[axis],
            description="coordinate-domain axis count",
        )
        width = sympy.simplify(end - begin)
        if (
            step <= 0
            or not _is_provably_nonnegative(begin, None)
            or not _is_provably_nonnegative(width, None)
            or not _is_provably_nonnegative(sympy.simplify(count - end), None)
        ):
            return None
        extent = width if step == 1 else FloorDiv(width + step - 1, step)
        cardinality *= extent
    return sympy.simplify(cardinality)


def _has_unclipped_point_source_support(relation: CoordinateRelation) -> bool:
    """Prove every raw source-box point denotes one in-domain target point.

    An exact converse that is total proves that every target has one source.
    If the disjoint raw source boxes contain exactly that many points and the
    forward relation is point-valued, no raw point can have been removed by
    target-domain clipping.  This is the generic support fact needed to
    compose through a full intermediate-domain piece without re-solving each
    mixed-radix range expression.
    """
    converse = _memoized_exact_converse(relation)
    if converse is None or not converse.is_total_function():
        return False
    pieces = _nonempty_relation_pieces(relation)
    if any(
        not _source_bounds_are_symbolically_within_domain(
            piece.source_bounds_items,
            relation.source_domain,
        )
        or any(
            step != 1
            or not _integer_partition_expressions_equal(end - begin, 1)
            for _axis, begin, end, step in piece.target_ranges
        )
        for piece in pieces
    ) or any(
        not _source_boxes_are_disjoint(left, right)
        for index, left in enumerate(pieces)
        for right in pieces[index + 1 :]
    ):
        return False
    cardinalities = tuple(
        _source_box_cardinality(
            piece.source_bounds_items,
            domain=relation.source_domain,
        )
        for piece in pieces
    )
    return None not in cardinalities and _integer_partition_expressions_equal(
        sympy.Add(*(value for value in cardinalities if value is not None)),
        relation.target_domain.size_expr,
    )


_DERIVED_EXACT_CONVERSE_ATTRIBUTE = "_derived_exact_converse"


def _memoized_exact_converse(
    relation: CoordinateRelation,
) -> CoordinateRelation | None:
    """Read an internally derived converse without initiating another proof."""
    converse = relation.__dict__.get(_DERIVED_EXACT_CONVERSE_ATTRIBUTE)
    if converse is None:
        return None
    if not isinstance(converse, CoordinateRelation) or (
        converse.source_domain != relation.target_domain
        or converse.target_domain != relation.source_domain
    ):
        raise AssertionError("invalid derived exact-converse memo")
    return converse


def _remember_exact_converse(
    relation: CoordinateRelation,
    converse: CoordinateRelation,
) -> CoordinateRelation:
    """Memoize a proved converse pair outside dataclass value semantics."""
    if (
        converse.source_domain != relation.target_domain
        or converse.target_domain != relation.source_domain
    ):
        raise AssertionError("exact converse has reversed domains")
    existing = _memoized_exact_converse(relation)
    if existing is not None:
        return existing
    relation.__dict__[_DERIVED_EXACT_CONVERSE_ATTRIBUTE] = converse
    if _memoized_exact_converse(converse) is None:
        converse.__dict__[_DERIVED_EXACT_CONVERSE_ATTRIBUTE] = relation
    return converse


def _cheap_source_support_converse(
    relation: CoordinateRelation,
) -> CoordinateRelation | None:
    """Return a cheap structural converse of an exact support ordinal."""
    if (converse := _memoized_exact_converse(relation)) is not None:
        return converse
    if (
        len(relation.target_domain.axis_order) == 1
        and len(relation.source_domain.axis_order) >= 2
        and (
            converse := _source_support_ordinalization(
                relation,
                reverse=True,
            )
        )
        is not None
    ):
        return _remember_exact_converse(relation, converse)
    return None


def _integer_partition_expressions_equal(
    left: IntegerExpression,
    right: IntegerExpression,
) -> bool:
    """Compare integer expressions after canonical quotient simplification."""
    difference = _simplify_integer_quotients(
        sympy.simplify(sympy.sympify(left) - sympy.sympify(right))
    )
    return sympy.simplify(difference) == 0


def _coordinate_permutation_axes(
    relation: CoordinateRelation,
) -> dict[int, int] | None:
    """Return target-to-source axes for a complete coordinate permutation."""
    if (
        len(relation.pieces) != 1
        or len(relation.source_domain.axis_order)
        != len(relation.target_domain.axis_order)
    ):
        return None
    (piece,) = relation.pieces
    if piece.source_bounds_items != tuple(
        (axis, 0, relation.source_domain.axis_count_expressions[axis], 1)
        for axis in relation.source_domain.axis_order
    ):
        return None
    source_axis_by_symbol = {
        coordinate_axis_symbol(axis): axis
        for axis in relation.source_domain.axis_order
    }
    result: dict[int, int] = {}
    for target_axis, begin, end, step in piece.target_ranges:
        source_axis = source_axis_by_symbol.get(begin)
        if (
            source_axis is None
            or source_axis in result.values()
            or step != 1
            or not _integer_partition_expressions_equal(end - begin, 1)
            or not _integer_partition_expressions_equal(
                relation.source_domain.axis_count_expressions[source_axis],
                relation.target_domain.axis_count_expressions[target_axis],
            )
        ):
            return None
        result[target_axis] = source_axis
    return (
        result
        if tuple(result) == relation.target_domain.axis_order
        and len(result) == len(relation.source_domain.axis_order)
        else None
    )


def _coordinate_permutation_converse(
    relation: CoordinateRelation,
) -> CoordinateRelation | None:
    """Invert a complete coordinate permutation without radix reconstruction."""
    target_to_source = _coordinate_permutation_axes(relation)
    if target_to_source is None:
        return None
    source_to_target = {
        source_axis: target_axis
        for target_axis, source_axis in target_to_source.items()
    }
    return CoordinateRelation.point_map(
        relation.target_domain,
        relation.source_domain,
        (
            (
                tuple(
                    (
                        axis,
                        0,
                        relation.target_domain.axis_count_expressions[axis],
                        1,
                    )
                    for axis in relation.target_domain.axis_order
                ),
                tuple(
                    coordinate_axis_symbol(source_to_target[axis])
                    for axis in relation.source_domain.axis_order
                ),
            ),
        ),
    )


def _nonempty_relation_pieces(
    relation: CoordinateRelation,
) -> tuple[_CoordinateRelationPiece, ...]:
    """Discard only pieces whose source box is proved empty."""
    return tuple(
        piece
        for piece in relation.pieces
        if not any(
            sympy.simplify(sympy.sympify(end) - sympy.sympify(begin)).is_nonpositive
            is True
            for _axis, begin, end, _step in piece.source_bounds_items
        )
    )


def _source_bounds_equal(
    left: tuple[tuple[int, IntegerExpression, IntegerExpression, int], ...],
    right: tuple[tuple[int, IntegerExpression, IntegerExpression, int], ...],
) -> bool:
    return len(left) == len(right) and all(
        left_axis == right_axis
        and left_step == right_step
        and _integer_partition_expressions_equal(left_begin, right_begin)
        and _integer_partition_expressions_equal(left_end, right_end)
        for (
            left_axis,
            left_begin,
            left_end,
            left_step,
        ), (
            right_axis,
            right_begin,
            right_end,
            right_step,
        ) in zip(left, right, strict=True)
    )


def _flattened_capacity_is_sufficient(
    outer_count: sympy.Expr,
    inner_count: int,
    required_slots: sympy.Expr,
) -> bool:
    """Prove a dense two-axis container can hold a required prefix."""
    if _is_provably_nonnegative(
        _simplify_integer_quotients(
            sympy.simplify(outer_count * inner_count - required_slots)
        ),
        None,
    ):
        return True
    quotient = _static_integer_quotient(outer_count)
    if quotient is None:
        return False
    numerator, denominator = quotient
    if denominator != inner_count:
        return False
    # ceildiv(available, inner_count) is represented as
    # FloorDiv(available + inner_count - 1, inner_count).
    available = sympy.simplify(numerator - inner_count + 1)
    return _is_provably_nonnegative(
        _simplify_integer_quotients(
            sympy.simplify(available - required_slots)
        ),
        None,
    )


def _fixed_source_coordinates(
    relation: CoordinateRelation,
    piece: _CoordinateRelationPiece,
    varying_axes: frozenset[int],
) -> dict[int, sympy.Expr] | None:
    """Return proved in-domain singleton coordinates on all other axes."""
    counts = relation.source_domain.axis_count_expressions
    result: dict[int, sympy.Expr] = {}
    for axis, begin, end, step in piece.source_bounds_items:
        if axis in varying_axes:
            continue
        begin = sympy.sympify(begin)
        end = sympy.sympify(end)
        if (
            step != 1
            or not _integer_partition_expressions_equal(end - begin, 1)
            or not _is_provably_nonnegative(begin, None)
            or not _is_provably_nonnegative(
                sympy.simplify(sympy.sympify(counts[axis]) - end),
                None,
            )
        ):
            return None
        result[axis] = begin
    return result


def _relations_equal_after_source_coalescing(
    left: CoordinateRelation,
    right: CoordinateRelation,
) -> bool:
    """Compare point maps after the existing bounded source normalization."""
    if left == right:
        return True
    left = dataclasses.replace(left, pieces=_nonempty_relation_pieces(left))
    right = dataclasses.replace(right, pieces=_nonempty_relation_pieces(right))
    left = left.coalesce_adjacent_source_boxes()
    right = right.coalesce_adjacent_source_boxes()
    return left.is_pointwise_equal_on_same_support(right)


def _source_support_ordinalization(
    relation: CoordinateRelation,
    *,
    reverse: bool = False,
) -> CoordinateRelation | None:
    """Derive or invert a dense ordinalization of semantic source support.

    The proof depends only on relation geometry.  It accepts a bounded
    partition of one dense two-axis interval, or one phased strided traversal
    whose final row is clipped by the ordinal target domain.
    """
    if (
        not relation.pieces
        or len(relation.pieces) > _MAX_RELATION_PIECES
        or len(relation.source_domain.axis_order) < 2
    ):
        return None
    target_count = sympy.sympify(relation.target_domain.size_expr)
    if target_count.is_nonnegative is not True:
        return None
    if len(relation.target_domain.axis_order) == 1:
        ordinal_domain = relation.target_domain
    else:
        used_axes = frozenset(
            (*relation.source_domain.axis_order, *relation.target_domain.axis_order)
        )
        ordinal_axis = min(used_axes, default=0) - 1
        while ordinal_axis in used_axes:
            ordinal_axis -= 1
        ordinal_domain = CoordinateDomain(
            (ordinal_axis,),
            ((ordinal_axis, target_count),),
            kind="task_order",
            identity=relation.target_domain.identity,
            _allow_empty=target_count.is_zero is True,
        )
    (ordinal_axis,) = ordinal_domain.axis_order
    ordinal = coordinate_axis_symbol(ordinal_axis)
    source_counts = relation.source_domain.axis_count_expressions
    pieces = _nonempty_relation_pieces(relation)
    if not pieces:
        return None

    def finish(
        point_expression: sympy.Expr,
        inverse: dict[int, sympy.Expr],
    ) -> CoordinateRelation | None:
        ordinalization = CoordinateRelation.point_map(
            relation.source_domain,
            ordinal_domain,
            tuple(
                (piece.source_bounds_items, (point_expression,))
                for piece in relation.pieces
            ),
        )
        if reverse:
            if not _relations_equal_after_source_coalescing(
                ordinalization,
                relation,
            ):
                return None
            return CoordinateRelation.point_map(
                ordinal_domain,
                relation.source_domain,
                (
                    (
                        ((ordinal_axis, 0, target_count, 1),),
                        tuple(
                            inverse[axis]
                            for axis in relation.source_domain.axis_order
                        ),
                    ),
                ),
            )
        return ordinalization

    # A bounded partition of one row-major interval.
    for inner_axis in relation.source_domain.axis_order:
        try:
            inner_count = _concrete_integer(
                source_counts[inner_axis],
                description="ordinalization inner-axis count",
            )
        except ValueError:
            continue
        if inner_count <= 0:
            continue
        inner = coordinate_axis_symbol(inner_axis)
        for outer_axis in relation.source_domain.axis_order:
            if outer_axis == inner_axis:
                continue
            outer = coordinate_axis_symbol(outer_axis)
            for candidate_piece in pieces:
                candidate_bounds = {
                    axis: (begin, end, step)
                    for axis, begin, end, step in candidate_piece.source_bounds_items
                }
                first_inner, _inner_end, inner_step = candidate_bounds[inner_axis]
                first_outer, outer_end, outer_step = candidate_bounds[outer_axis]
                first_inner = sympy.sympify(first_inner)
                first_outer = sympy.sympify(first_outer)
                fixed = _fixed_source_coordinates(
                    relation,
                    candidate_piece,
                    frozenset((inner_axis, outer_axis)),
                )
                first_inner_interval = _bounded_parameter_expression_interval(
                    first_inner
                )
                if (
                    fixed is None
                    or inner_step != 1
                    or outer_step != 1
                    or not _integer_partition_expressions_equal(
                        outer_end - first_outer,
                        1,
                    )
                    or not _is_provably_nonnegative(first_outer, None)
                    or first_inner_interval is None
                    or not _is_provably_nonnegative(first_inner_interval[0], None)
                    or not _is_provably_nonnegative(
                        sympy.simplify(inner_count - 1 - first_inner_interval[1]),
                        None,
                    )
                    or not _flattened_capacity_is_sufficient(
                        sympy.sympify(source_counts[outer_axis]),
                        inner_count,
                        sympy.simplify(
                            first_outer * inner_count
                            + first_inner
                            + target_count
                        ),
                    )
                ):
                    continue
                point_expression = _simplify_integer_quotients(
                    sympy.simplify(
                        (outer - first_outer) * inner_count + inner - first_inner
                    )
                )
                first_count = SymbolicMin(
                    target_count,
                    inner_count - first_inner,
                )
                remaining = SymbolicMax(
                    sympy.simplify(target_count - first_count),
                    sympy.Integer(0),
                )
                full_outer = FloorDiv(remaining, inner_count)
                tail_count = sympy.Mod(remaining, inner_count)
                middle_begin = sympy.simplify(first_outer + 1)
                middle_end = sympy.simplify(middle_begin + full_outer)
                final_end = sympy.simplify(
                    middle_end
                    + FloorDiv(tail_count + inner_count - 1, inner_count)
                )

                def source_box(
                    inner_begin: IntegerExpression,
                    inner_end: IntegerExpression,
                    outer_begin: IntegerExpression,
                    outer_end: IntegerExpression,
                ) -> tuple[
                    tuple[int, IntegerExpression, IntegerExpression, int], ...
                ]:
                    bounds = {
                        **{
                            axis: (value, value + 1, 1)
                            for axis, value in fixed.items()
                        },
                        inner_axis: (inner_begin, inner_end, 1),
                        outer_axis: (outer_begin, outer_end, 1),
                    }
                    return tuple(
                        (axis, *bounds[axis])
                        for axis in relation.source_domain.axis_order
                    )

                expected_bounds = tuple(
                    bounds
                    for bounds in (
                        source_box(
                            first_inner,
                            first_inner + first_count,
                            first_outer,
                            first_outer + 1,
                        ),
                        source_box(0, inner_count, middle_begin, middle_end),
                        source_box(0, tail_count, middle_end, final_end),
                    )
                    if not any(
                        sympy.simplify(
                            sympy.sympify(end) - sympy.sympify(begin)
                        ).is_nonpositive
                        is True
                        for _axis, begin, end, _step in bounds
                    )
                )
                expected = CoordinateRelation.point_map(
                    relation.source_domain,
                    ordinal_domain,
                    tuple((bounds, (point_expression,)) for bounds in expected_bounds),
                )
                actual = CoordinateRelation.point_map(
                    relation.source_domain,
                    ordinal_domain,
                    tuple(
                        (piece.source_bounds_items, (point_expression,))
                        for piece in relation.pieces
                    ),
                )
                if not _relations_equal_after_source_coalescing(actual, expected):
                    continue
                inverse = {
                    **fixed,
                    inner_axis: sympy.Mod(ordinal + first_inner, inner_count),
                    outer_axis: first_outer
                    + sympy.floor((ordinal + first_inner) / inner_count),
                }
                return finish(point_expression, inverse)

    # A phased row-major traversal represented by exact full rows plus one
    # exact tail row.  Unlike a padded source box, every raw source point maps
    # inside the ordinal domain, so later composition never has to recover
    # support that was implicit in target clipping.
    for inner_axis in relation.source_domain.axis_order:
        try:
            inner_count = _concrete_integer(
                source_counts[inner_axis],
                description="ordinalization inner-axis count",
            )
        except ValueError:
            continue
        if inner_count <= 0:
            continue
        for outer_axis in relation.source_domain.axis_order:
            if outer_axis == inner_axis:
                continue
            for full_piece in pieces:
                full_bounds = {
                    axis: (begin, end, step)
                    for axis, begin, end, step in full_piece.source_bounds_items
                }
                if not _source_bounds_equal(
                    ((inner_axis, *full_bounds[inner_axis]),),
                    ((inner_axis, 0, inner_count, 1),),
                ):
                    continue
                phase, _full_end, period = full_bounds[outer_axis]
                fixed = _fixed_source_coordinates(
                    relation,
                    full_piece,
                    frozenset((inner_axis, outer_axis)),
                )
                if (
                    fixed is None
                    or period <= 0
                    or not isinstance(phase, sympy.Integer)
                    or not 0 <= int(phase) < period
                    or any(
                        _fixed_source_coordinates(
                            relation,
                            piece,
                            frozenset((inner_axis, outer_axis)),
                        )
                        != fixed
                        for piece in pieces
                    )
                    or not _integer_partition_expressions_equal(
                        source_counts[outer_axis],
                        period
                        * FloorDiv(
                            target_count + inner_count - 1,
                            inner_count,
                        ),
                    )
                ):
                    continue
                full_rows = FloorDiv(target_count, inner_count)
                tail_count = sympy.Mod(target_count, inner_count)
                tail_row = sympy.simplify(phase + period * full_rows)
                tail_rows = FloorDiv(tail_count + inner_count - 1, inner_count)

                def phased_box(
                    inner_begin: IntegerExpression,
                    inner_end: IntegerExpression,
                    outer_begin: IntegerExpression,
                    outer_end: IntegerExpression,
                    outer_step: int,
                ) -> tuple[
                    tuple[int, IntegerExpression, IntegerExpression, int], ...
                ]:
                    bounds = {
                        **{
                            axis: (value, value + 1, 1)
                            for axis, value in fixed.items()
                        },
                        inner_axis: (inner_begin, inner_end, 1),
                        outer_axis: (outer_begin, outer_end, outer_step),
                    }
                    return tuple(
                        (axis, *bounds[axis])
                        for axis in relation.source_domain.axis_order
                    )

                expected_bounds = tuple(
                    bounds
                    for bounds in (
                        phased_box(0, inner_count, phase, tail_row, period),
                        phased_box(
                            0,
                            tail_count,
                            tail_row,
                            sympy.simplify(tail_row + tail_rows),
                            1,
                        ),
                    )
                    if not any(
                        sympy.simplify(
                            sympy.sympify(end) - sympy.sympify(begin)
                        ).is_nonpositive
                        is True
                        for _axis, begin, end, _step in bounds
                    )
                )
                point_expression = _simplify_integer_quotients(
                    sympy.floor(coordinate_axis_symbol(outer_axis) / period)
                    * inner_count
                    + coordinate_axis_symbol(inner_axis)
                )
                expected = CoordinateRelation.point_map(
                    relation.source_domain,
                    ordinal_domain,
                    tuple((bounds, (point_expression,)) for bounds in expected_bounds),
                )
                actual = CoordinateRelation.point_map(
                    relation.source_domain,
                    ordinal_domain,
                    tuple(
                        (piece.source_bounds_items, (point_expression,))
                        for piece in relation.pieces
                    ),
                )
                if not _relations_equal_after_source_coalescing(actual, expected):
                    continue
                inverse = {
                    **fixed,
                    inner_axis: sympy.Mod(ordinal, inner_count),
                    outer_axis: period * sympy.floor(ordinal / inner_count) + phase,
                }
                return finish(point_expression, inverse)
    return None


def _factor_through_source_ordinalization(
    relation: CoordinateRelation,
    ordinalization: CoordinateRelation,
    ordinal_inverse: CoordinateRelation,
) -> CoordinateRelation | None:
    """Factor a point map through an exact dense support ordinalization."""
    if _relations_equal_after_source_coalescing(relation, ordinalization):
        return CoordinateRelation.identity(
            ordinalization.target_domain,
            ordinalization.target_domain,
        )
    if (
        len(ordinalization.target_domain.axis_order) != 1
        or ordinal_inverse.source_domain != ordinalization.target_domain
        or ordinal_inverse.target_domain != relation.source_domain
        or not _relation_product_is_within_budget(
            len(relation.pieces),
            len(ordinalization.pieces),
            len(ordinal_inverse.pieces),
        )
    ):
        return None
    target_count = ordinalization.target_domain.size_expr
    (ordinal_axis,) = ordinalization.target_domain.axis_order
    ordinal = coordinate_axis_symbol(ordinal_axis)
    if len(ordinal_inverse.pieces) != 1 or not _source_bounds_equal(
        ordinal_inverse.pieces[0].source_bounds_items,
        ((ordinal_axis, 0, target_count, 1),),
    ):
        return None
    (inverse_piece,) = ordinal_inverse.pieces
    inverse_substitutions = {
        coordinate_axis_symbol(axis): begin
        for axis, begin, _end, _step in inverse_piece.target_ranges
    }
    full_ordinal_bounds = ((ordinal_axis, 0, target_count, 1),)

    def full_candidate_recomposes(candidate: CoordinateRelation) -> bool:
        """Prove ``relation == ordinalization ; candidate`` piecewise."""
        if len(candidate.pieces) != 1:
            return False
        (candidate_piece,) = candidate.pieces
        if not _source_bounds_equal(
            candidate_piece.source_bounds_items,
            full_ordinal_bounds,
        ):
            return False
        matched_ordinal_pieces: set[int] = set()
        for relation_piece in relation.pieces:
            matching = tuple(
                (index, ordinal_piece)
                for index, ordinal_piece in enumerate(ordinalization.pieces)
                if _source_bounds_equal(
                    ordinal_piece.source_bounds_items,
                    relation_piece.source_bounds_items,
                )
            )
            if len(matching) != 1:
                return False
            ordinal_index, ordinal_piece = matching[0]
            matched_ordinal_pieces.add(ordinal_index)
            ordinal_expression = ordinal_piece.target_ranges[0][1]
            substitutions = {
                coordinate_axis_symbol(ordinal_axis): ordinal_expression
            }
            recomposed_targets = tuple(
                (
                    target_axis,
                    _substitute_composed_expression(
                        begin,
                        substitutions=substitutions,
                        source_domain=relation.source_domain,
                        source_bounds=relation_piece.source_bounds_items,
                    ),
                    _substitute_composed_expression(
                        end,
                        substitutions=substitutions,
                        source_domain=relation.source_domain,
                        source_bounds=relation_piece.source_bounds_items,
                    ),
                    step,
                )
                for target_axis, begin, end, step in candidate_piece.target_ranges
            )
            if any(
                actual_axis != expected_axis
                or actual_step != expected_step
                or not _integer_partition_expressions_equal(
                    _simplify_logical_expression(
                        actual_begin,
                        domain=relation.source_domain,
                        source_bounds=relation_piece.source_bounds_items,
                    ),
                    _simplify_logical_expression(
                        expected_begin,
                        domain=relation.source_domain,
                        source_bounds=relation_piece.source_bounds_items,
                    ),
                )
                or not _integer_partition_expressions_equal(
                    _simplify_logical_expression(
                        actual_end,
                        domain=relation.source_domain,
                        source_bounds=relation_piece.source_bounds_items,
                    ),
                    _simplify_logical_expression(
                        expected_end,
                        domain=relation.source_domain,
                        source_bounds=relation_piece.source_bounds_items,
                    ),
                )
                for (
                    actual_axis,
                    actual_begin,
                    actual_end,
                    actual_step,
                ), (
                    expected_axis,
                    expected_begin,
                    expected_end,
                    expected_step,
                ) in zip(
                    relation_piece.target_ranges,
                    recomposed_targets,
                    strict=True,
                )
            ):
                return False
        return len(matched_ordinal_pieces) == len(ordinalization.pieces)

    # Packed schedule pieces normally share one semantic ordinal expression.
    # Replace that expression directly before falling back to substitution
    # through the (necessarily more complicated) mixed-radix inverse.  This
    # is an exact quotient rewrite: ``full_candidate_recomposes`` remains the
    # acceptance proof, while avoiding expensive simplification of identities
    # such as ``W * floor((i + offset) / W) + Mod(i + offset, W)``.
    ordinal_expressions = tuple(
        dict.fromkeys(
            piece.target_ranges[0][1]
            for piece in ordinalization.pieces
            if len(piece.target_ranges) == 1
        )
    )
    if len(ordinal_expressions) == 1:
        ordinal_expression = ordinal_expressions[0]
        source_symbols = frozenset(
            coordinate_axis_symbol(axis)
            for axis in relation.source_domain.axis_order
        )
        direct_targets: dict[
            tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...], None
        ] = {}
        for piece in relation.pieces:
            target_ranges = tuple(
                (
                    target_axis,
                    cast(
                        "sympy.Expr",
                        target_begin.subs(
                            ordinal_expression,
                            ordinal,
                            simultaneous=True,
                        ),
                    ),
                    cast(
                        "sympy.Expr",
                        target_end.subs(
                            ordinal_expression,
                            ordinal,
                            simultaneous=True,
                        ),
                    ),
                    target_step,
                )
                for target_axis, target_begin, target_end, target_step in piece.target_ranges
            )
            if any(
                (begin.free_symbols | end.free_symbols) & source_symbols
                for _axis, begin, end, _step in target_ranges
            ):
                direct_targets.clear()
                break
            direct_targets.setdefault(target_ranges, None)
        for target_ranges in direct_targets:
            candidate = CoordinateRelation(
                source_domain=ordinalization.target_domain,
                target_domain=relation.target_domain,
                pieces=(
                    _CoordinateRelationPiece(
                        source_bounds_items=full_ordinal_bounds,
                        target_ranges=target_ranges,
                    ),
                ),
            )
            if candidate.is_total_function() and full_candidate_recomposes(candidate):
                return candidate

    # Piece boundaries in ``relation`` may be artifacts of the packed source
    # support rather than changes in the logical task map.  If the direct
    # quotient rewrite above did not apply, derive candidates by substituting
    # the proved inverse over the complete ordinal domain.
    candidate_targets: dict[
        tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...], None
    ] = {}
    for piece in relation.pieces:
        target_ranges = tuple(
            (
                target_axis,
                _substitute_composed_expression(
                    target_begin,
                    substitutions=inverse_substitutions,
                    source_domain=ordinalization.target_domain,
                    source_bounds=full_ordinal_bounds,
                ),
                _substitute_composed_expression(
                    target_end,
                    substitutions=inverse_substitutions,
                    source_domain=ordinalization.target_domain,
                    source_bounds=full_ordinal_bounds,
                ),
                target_step,
            )
            for target_axis, target_begin, target_end, target_step in piece.target_ranges
        )
        candidate_targets.setdefault(target_ranges, None)

    for target_ranges in candidate_targets:
        candidate = CoordinateRelation(
            source_domain=ordinalization.target_domain,
            target_domain=relation.target_domain,
            pieces=(
                _CoordinateRelationPiece(
                    source_bounds_items=full_ordinal_bounds,
                    target_ranges=target_ranges,
                ),
            ),
        )
        if candidate.is_total_function() and full_candidate_recomposes(candidate):
            return candidate

    factored_pieces: list[_CoordinateRelationPiece] = []
    for piece in relation.pieces:
        matching = tuple(
            candidate
            for candidate in ordinalization.pieces
            if _source_bounds_equal(
                candidate.source_bounds_items,
                piece.source_bounds_items,
            )
        )
        if len(matching) != 1 or len(matching[0].target_ranges) != 1:
            return None
        axis, begin, end, step = matching[0].target_ranges[0]
        bounds = _logical_expression_bounds(
            begin,
            domain=relation.source_domain,
            source_bounds=piece.source_bounds_items,
        )
        if (
            axis != ordinal_axis
            or step != 1
            or not _integer_partition_expressions_equal(end - begin, 1)
            or bounds is None
        ):
            return None
        ordinal_begin = SymbolicMax(sympy.Integer(0), bounds[0])
        ordinal_end = SymbolicMin(target_count, bounds[1] + 1)
        if sympy.simplify(ordinal_end - ordinal_begin).is_nonpositive is True:
            continue
        source_bounds = ((ordinal_axis, ordinal_begin, ordinal_end, 1),)
        factored_pieces.append(
            _CoordinateRelationPiece(
                source_bounds_items=source_bounds,
                target_ranges=tuple(
                    (
                        target_axis,
                        _substitute_composed_expression(
                            target_begin,
                            substitutions=inverse_substitutions,
                            source_domain=ordinalization.target_domain,
                            source_bounds=source_bounds,
                        ),
                        _substitute_composed_expression(
                            target_end,
                            substitutions=inverse_substitutions,
                            source_domain=ordinalization.target_domain,
                            source_bounds=source_bounds,
                        ),
                        target_step,
                    )
                    for target_axis, target_begin, target_end, target_step in piece.target_ranges
                ),
            )
        )
        if len(factored_pieces) > _MAX_RELATION_PIECES:
            return None
    factored = CoordinateRelation(
        source_domain=ordinalization.target_domain,
        target_domain=relation.target_domain,
        pieces=tuple(dict.fromkeys(factored_pieces)),
    ).coalesce_adjacent_source_boxes()
    recomposed = ordinalization.then(factored)
    if recomposed is None or not _relations_equal_after_source_coalescing(
        recomposed,
        relation,
    ):
        return None
    return factored


def _ordinalized_source_supports_are_disjoint(
    left: CoordinateRelation | None,
    right: CoordinateRelation | None,
) -> bool:
    """Prove two dense ordinalized supports occupy disjoint source intervals."""
    if (
        left is None
        or right is None
        or left.source_domain != right.source_domain
        or len(left.target_domain.axis_order) != 1
        or len(right.target_domain.axis_order) != 1
    ):
        return False
    left_piece = next(iter(_nonempty_relation_pieces(left)), None)
    right_piece = next(iter(_nonempty_relation_pieces(right)), None)
    if left_piece is None or right_piece is None:
        return True
    left_expression = left_piece.target_ranges[0][1]
    right_expression = right_piece.target_ranges[0][1]
    source_symbols = frozenset(
        coordinate_axis_symbol(axis) for axis in left.source_domain.axis_order
    )
    difference = _simplify_integer_quotients(
        sympy.simplify(left_expression - right_expression)
    )
    if difference.free_symbols & source_symbols:
        return False
    left_fixed = _fixed_source_coordinates(
        left,
        left_piece,
        frozenset(
            axis
            for axis in left.source_domain.axis_order
            if coordinate_axis_symbol(axis) in left_expression.free_symbols
        ),
    )
    right_fixed = _fixed_source_coordinates(
        right,
        right_piece,
        frozenset(
            axis
            for axis in right.source_domain.axis_order
            if coordinate_axis_symbol(axis) in right_expression.free_symbols
        ),
    )
    if left_fixed is None or left_fixed != right_fixed:
        return False
    return _is_provably_nonnegative(
        sympy.simplify(difference - left.target_domain.size_expr),
        None,
    ) or _is_provably_nonnegative(
        sympy.simplify(-difference - right.target_domain.size_expr),
        None,
    )


def _source_bounds_are_disjoint(
    left: tuple[tuple[int, IntegerExpression, IntegerExpression, int], ...],
    right: tuple[tuple[int, IntegerExpression, IntegerExpression, int], ...],
) -> bool:
    """Conservatively prove two strided source bounds have no common point."""
    for left_bound, right_bound in zip(
        left,
        right,
        strict=True,
    ):
        left_axis, left_begin, left_end, left_step = left_bound
        right_axis, right_begin, right_end, right_step = right_bound
        if left_axis != right_axis:
            return True
        if _is_provably_nonnegative(
            sympy.simplify(right_begin - left_end),  # pyrefly: ignore[unsupported-operation]
            None,
        ) or _is_provably_nonnegative(
            sympy.simplify(left_begin - right_end),  # pyrefly: ignore[unsupported-operation]
            None,
        ):
            return True
        left_last = _simplify_integer_quotients(
            sympy.simplify(
                left_begin
                + FloorDiv(left_end - left_begin - 1, left_step) * left_step  # pyrefly: ignore[unsupported-operation]
            )
        )
        right_last = _simplify_integer_quotients(
            sympy.simplify(
                right_begin
                + FloorDiv(right_end - right_begin - 1, right_step) * right_step  # pyrefly: ignore[unsupported-operation]
            )
        )
        if _is_provably_nonnegative(
            _simplify_integer_quotients(
                sympy.simplify(right_begin - left_last - 1)  # pyrefly: ignore[unsupported-operation]
            ),
            None,
        ) or _is_provably_nonnegative(
            _simplify_integer_quotients(
                sympy.simplify(left_begin - right_last - 1)  # pyrefly: ignore[unsupported-operation]
            ),
            None,
        ):
            return True
        residue = sympy.simplify(
            sympy.Mod(  # pyrefly: ignore[bad-argument-type]
                right_begin - left_begin,  # pyrefly: ignore[unsupported-operation]
                math.gcd(left_step, right_step),
            )
        )
        if residue.is_zero is False:
            return True
    return False


def _source_boxes_partition_domain(
    boxes: tuple[
        tuple[tuple[int, IntegerExpression, IntegerExpression, int], ...], ...
    ],
    domain: CoordinateDomain,
) -> bool:
    """Prove that distinct unit-stride boxes partition a coordinate domain."""
    if not boxes:
        return False
    if not domain.axis_order:
        return boxes == ((),)
    if domain.parameter_symbols:
        if not all(
            _source_bounds_are_symbolically_within_domain(box, domain) for box in boxes
        ):
            return False
        full_axes = frozenset(
            axis
            for axis in domain.axis_order
            if all(
                begin == 0
                and step == 1
                and sympy.simplify(
                    end - domain.axis_count_expressions[axis]  # pyrefly: ignore[unsupported-operation]
                )
                == 0
                for box in boxes
                for bound_axis, begin, end, step in box
                if bound_axis == axis
            )
        )
        residual_axes = tuple(
            axis for axis in domain.axis_order if axis not in full_axes
        )
        residual_counts = tuple(
            domain.axis_count_expressions[axis] for axis in residual_axes
        )
        residual_boxes = tuple(
            tuple(bound for bound in box if bound[0] in residual_axes) for box in boxes
        )
        if any(count.free_symbols for count in residual_counts) or any(
            sympy.sympify(value).free_symbols
            for box in residual_boxes
            for _axis, begin, end, _step in box
            for value in (begin, end)
        ):
            return False
        residual_domain = CoordinateDomain(
            axis_order=residual_axes,
            axis_counts_items=tuple(
                (axis, int(count))
                for axis, count in zip(residual_axes, residual_counts, strict=True)
            ),
            kind=domain.kind,
            identity=domain.identity,
        )
        return _source_boxes_partition_domain(residual_boxes, residual_domain)
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
    outer: tuple[tuple[int, IntegerExpression, IntegerExpression, int], ...],
    inner: tuple[tuple[int, IntegerExpression, IntegerExpression, int], ...],
) -> bool:
    """Return whether one unit-stride source box contains another."""
    return all(
        outer_axis == inner_axis
        and outer_step == inner_step == 1
        and _is_provably_nonnegative(
            sympy.simplify(inner_begin - outer_begin),  # pyrefly: ignore[unsupported-operation]
            None,
        )
        and _is_provably_nonnegative(
            sympy.simplify(outer_end - inner_end),  # pyrefly: ignore[unsupported-operation]
            None,
        )
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
    relation: CoordinateRelation,
    *,
    include_domain: bool = False,
) -> tuple[tuple[tuple[int, int, int, int], ...], ...] | None:
    """Partition source space at relation-piece boundaries without enumeration."""
    return _relations_source_cells((relation,), include_domain=include_domain)


def _relations_source_cells(
    relations: tuple[CoordinateRelation, ...],
    *,
    include_domain: bool = False,
) -> tuple[tuple[tuple[int, int, int, int], ...], ...] | None:
    """Return a bounded common concrete source partition for relations."""
    if not relations:
        return ()
    source_domain = relations[0].source_domain
    if (
        any(relation.source_domain != source_domain for relation in relations)
        or any(relation.parameter_symbols for relation in relations)
        or any(len(relation.pieces) > _MAX_RELATION_PIECES for relation in relations)
    ):
        return None
    if any(
        step != 1
        for relation in relations
        for piece in relation.pieces
        for _axis, _begin, _end, step in piece.source_bounds_items
    ):
        return None
    cuts: dict[int, set[int]] = {
        axis: ({0, count} if include_domain else set())
        for axis, count in source_domain.axis_counts_items
    }
    for relation in relations:
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
        for axis in source_domain.axis_order
    )
    if not _relation_product_is_within_budget(
        *(len(axis_intervals) for axis_intervals in intervals)
    ):
        return None
    pieces = tuple(piece for relation in relations for piece in relation.pieces)
    result: list[tuple[tuple[int, int, int, int], ...]] = []
    for cell in itertools.product(*intervals):
        bounds = tuple(cell)
        if include_domain or any(
            _source_box_covers(piece.source_bounds_items, bounds) for piece in pieces
        ):
            result.append(bounds)
            if len(result) > _MAX_RELATION_PIECES:
                return None
    return tuple(result)


def _target_boxes_are_disjoint(
    left: tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...],
    right: tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...],
    *,
    source_domain: CoordinateDomain,
    source_bounds: tuple[tuple[int, IntegerExpression, IntegerExpression, int], ...],
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
                and _is_provably_nonnegative(
                    left_before_right_bounds[0],
                    None,
                )
            )
            or (
                right_before_left_bounds is not None
                and _is_provably_nonnegative(
                    right_before_left_bounds[0],
                    None,
                )
            )
        ):
            return True
    return False


def _normalize_integer_rounding(expression: sympy.Expr) -> sympy.Expr:
    """Canonicalize floor/ceiling of an integer expression over a static divisor."""
    if expression.func not in (sympy.floor, sympy.ceiling) or len(expression.args) != 1:
        return expression
    numerator, denominator = sympy.fraction(sympy.together(expression.args[0]))
    if (
        numerator.is_integer is not True
        or denominator.is_integer is not True
        or denominator.free_symbols
        or int(denominator) <= 0
    ):
        return expression
    divisor = int(denominator)
    integer_part: sympy.Expr = sympy.Integer(0)
    remainder: sympy.Expr = sympy.Integer(0)
    for term in sympy.Add.make_args(sympy.expand(numerator)):
        coefficient, primitive = term.as_coeff_Mul()
        if coefficient.is_Integer and primitive.is_integer is True:
            quotient, residue = divmod(int(coefficient), divisor)
            integer_part += quotient * primitive
            remainder += residue * primitive
        else:
            remainder += term
    if expression.func == sympy.ceiling:
        remainder += divisor - 1
    return sympy.simplify(
        integer_part + sympy.floor(remainder / divisor)  # pyrefly: ignore[bad-argument-type, unsupported-operation]
    )


def _static_integer_quotient(
    expression: sympy.Expr,
) -> tuple[sympy.Expr, int] | None:
    """Parse either spelling of floor division by a positive static integer."""
    expression = sympy.sympify(expression)
    if expression.func == FloorDiv and len(expression.args) == 2:
        numerator, denominator = expression.args
    elif expression.func == sympy.floor and len(expression.args) == 1:
        numerator, denominator = sympy.fraction(sympy.together(expression.args[0]))
    else:
        return None
    if (
        numerator.is_integer is not True
        or denominator.free_symbols
        or denominator.is_integer is not True
        or int(denominator) <= 0
    ):
        return None
    return cast("sympy.Expr", numerator), int(denominator)


def _simplify_integer_quotients(expression: sympy.Expr) -> sympy.Expr:
    """Canonicalize exact integer quotient/remainder identities."""
    replacements = {
        node: sympy.floor(numerator / denominator)  # pyrefly: ignore[bad-argument-type, unsupported-operation]
        for node in sympy.preorder_traversal(expression)
        if (quotient := _static_integer_quotient(cast("sympy.Expr", node)))
        is not None
        for numerator, denominator in (quotient,)
        if node.func == FloorDiv
    }
    result = sympy.simplify(expression.xreplace(replacements))
    while result.func == sympy.Add:
        terms = list(result.args)
        replacement: sympy.Expr | None = None
        for modulo_index, term in enumerate(terms):
            modulo_coefficient, modulo = term.as_coeff_Mul()
            if not isinstance(modulo, sympy.Mod) or len(modulo.args) != 2:
                continue
            dividend, modulus = modulo.args
            if (
                modulo_coefficient.is_number is not True
                or dividend.is_integer is not True  # pyrefly: ignore[missing-attribute]
                or modulus.free_symbols
                or modulus.is_integer is not True  # pyrefly: ignore[missing-attribute]
                or int(modulus) <= 0  # pyrefly: ignore[bad-argument-type]
            ):
                continue
            quotient_term = (
                modulo_coefficient * modulus * sympy.floor(dividend / modulus)  # pyrefly: ignore[unsupported-operation]
            )
            quotient_index = next(
                (
                    index
                    for index, candidate in enumerate(terms)
                    if index != modulo_index
                    and sympy.simplify(candidate - quotient_term) == 0  # pyrefly: ignore[unsupported-operation]
                ),
                None,
            )
            if quotient_index is not None:
                replacement = sympy.Add(
                    *(
                        candidate
                        for index, candidate in enumerate(terms)
                        if index not in (modulo_index, quotient_index)
                    ),
                    modulo_coefficient * dividend,
                )
                break
        if replacement is None:
            break
        result = sympy.simplify(replacement)
    return result


def _logical_expression_bounds(
    expression: sympy.Expr,
    *,
    domain: CoordinateDomain,
    source_bounds: tuple[tuple[int, IntegerExpression, IntegerExpression, int], ...],
    symbol_substitutions: dict[sympy.Basic, sympy.Expr] | None = None,
) -> tuple[sympy.Expr, sympy.Expr] | None:
    """Return conservative inclusive bounds for the restricted expression IR."""
    if expression.is_number:
        return expression, expression
    if symbol_substitutions is not None:
        substituted = expression.xreplace(symbol_substitutions)
        if substituted != expression:
            # Apply the exact point-map substitution before classifying
            # coordinate-dependent factors.  In particular, an intermediate
            # coordinate may occur below a Mul/Floor even though only the
            # substituted coordinate belongs to ``domain``.
            return _logical_expression_bounds(
                cast("sympy.Expr", substituted),
                domain=domain,
                source_bounds=source_bounds,
            )
    coordinate_symbols = frozenset(
        coordinate_axis_symbol(axis) for axis in domain.axis_order
    )
    if (
        expression.free_symbols
        and not (expression.free_symbols & coordinate_symbols)
        and expression.free_symbols <= domain.parameter_symbols
    ):
        # Shape parameters are constant for one kernel invocation even though
        # their values are deliberately not specialized at compile time.
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
        axis_by_symbol = {
            coordinate_axis_symbol(axis): axis for axis in domain.axis_order
        }
        axis = axis_by_symbol.get(expression)
        if axis is None:
            return None
        begin, end, step = next(
            (begin, end, step)
            for bound_axis, begin, end, step in source_bounds
            if bound_axis == axis
        )
        final = begin + (end - begin - 1) // step * step
        return sympy.sympify(begin), sympy.sympify(final)
    if isinstance(expression, sympy.Add):
        piecewise_child = next(
            (
                child
                for child in expression.args
                if child.func in (sympy.Min, sympy.Max)
            ),
            None,
        )
        if piecewise_child is not None:
            common = sympy.Add(
                *(child for child in expression.args if child is not piecewise_child)
            )
            branch_bounds = tuple(
                _logical_expression_bounds(
                    sympy.simplify(common + branch),
                    domain=domain,
                    source_bounds=source_bounds,
                    symbol_substitutions=symbol_substitutions,
                )
                for branch in piecewise_child.args
            )
            if all(bounds is not None for bounds in branch_bounds):
                concrete = tuple(
                    bounds for bounds in branch_bounds if bounds is not None
                )
                return (
                    piecewise_child.func(*(bounds[0] for bounds in concrete)),
                    piecewise_child.func(*(bounds[1] for bounds in concrete)),
                )
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
        coefficient = sympy.Integer(1)
        coordinate_factors: list[sympy.Expr] = []
        for child in expression.args:
            if child.free_symbols & coordinate_symbols:
                coordinate_factors.append(child)
            else:
                if child.free_symbols - domain.parameter_symbols:
                    return None
                coefficient *= child  # pyrefly: ignore[unsupported-operation]
        if len(coordinate_factors) != 1:
            return None
        bounds = _logical_expression_bounds(
            coordinate_factors[0],
            domain=domain,
            source_bounds=source_bounds,
            symbol_substitutions=symbol_substitutions,
        )
        coefficient = sympy.simplify(coefficient)
        if bounds is None or coefficient.is_real is not True:
            return None
        values = (
            coefficient * bounds[0],  # pyrefly: ignore[unsupported-operation]
            coefficient * bounds[1],  # pyrefly: ignore[unsupported-operation]
        )
        if coefficient.is_nonnegative is True:
            return values
        if coefficient.is_nonpositive is True:
            return values[1], values[0]
        return None
    if expression.func in (sympy.floor, sympy.ceiling):
        bounds = _logical_expression_bounds(
            cast("sympy.Expr", expression.args[0]),
            domain=domain,
            source_bounds=source_bounds,
            symbol_substitutions=symbol_substitutions,
        )
        if bounds is None:
            return None
        return (
            _normalize_integer_rounding(expression.func(bounds[0])),
            _normalize_integer_rounding(expression.func(bounds[1])),
        )
    if expression.func is FloorDiv:
        numerator, denominator = expression.args
        if (
            denominator.free_symbols
            or denominator.is_integer is not True
            or denominator.is_positive is not True
        ):
            return None
        bounds = _logical_expression_bounds(
            cast("sympy.Expr", numerator),
            domain=domain,
            source_bounds=source_bounds,
            symbol_substitutions=symbol_substitutions,
        )
        if bounds is None:
            return None
        return (
            FloorDiv(bounds[0], denominator),
            FloorDiv(bounds[1], denominator),
        )
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
    value_source_bounds: tuple[
        tuple[int, IntegerExpression, IntegerExpression, int], ...
    ],
    *,
    source_domain: CoordinateDomain,
    relation_source_bounds: tuple[
        tuple[int, IntegerExpression, IntegerExpression, int], ...
    ],
) -> tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...] | bool | None:
    """Return a contained target box, ``False`` if disjoint, else unknown."""
    bounds_by_axis = {
        axis: (begin, end, step) for axis, begin, end, step in value_source_bounds
    }
    for axis, begin, end, step in target_ranges:
        source_begin, source_end, source_step = bounds_by_axis[axis]
        # A dense value piece contains every positive-stride subset of the
        # same interval.  For a strided value piece, the target stride must
        # preserve its lattice and the target begin must have the same
        # residue.  Equal strides alone are insufficient: even and odd
        # lattices have equal strides but are disjoint.
        if source_step != 1:
            if step % source_step:
                return None
            residue = sympy.simplify(
                sympy.Mod(  # pyrefly: ignore[bad-argument-type]
                    begin - source_begin,  # pyrefly: ignore[unsupported-operation]
                    source_step,
                )
            )
            if residue != 0:
                if residue.is_number:
                    return False
                residue_bounds = _logical_expression_bounds(
                    residue,
                    domain=source_domain,
                    source_bounds=relation_source_bounds,
                )
                if residue_bounds is None:
                    return None
                if residue_bounds[0] == residue_bounds[1] != 0:
                    return False
                if residue_bounds != (sympy.Integer(0), sympy.Integer(0)):
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
        if (before is not None and _is_provably_nonnegative(-before[1], None)) or (
            after is not None and _is_provably_nonnegative(after[0], None)
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
            or not _is_provably_nonnegative(lower[0], None)
            or upper is None
            or not _is_provably_nonnegative(upper[0], None)
        ):
            return None
    return target_ranges


def _target_box_expression_extreme(
    expression: sympy.Expr,
    *,
    target_domain: CoordinateDomain,
    target_ranges: tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...],
    maximize: bool,
) -> sympy.Expr | None:
    """Substitute a box endpoint into a coordinatewise-monotone expression."""
    ranges = {
        coordinate_axis_symbol(axis): (begin, end, step)
        for axis, begin, end, step in target_ranges
    }
    if expression.is_number:
        return expression
    if isinstance(expression, sympy.Symbol):
        target_range = ranges.get(expression)
        if target_range is None:
            return None
        begin, end, step = target_range
        if not maximize:
            return begin
        return (
            begin
            + sympy.floor(  # pyrefly: ignore[unsupported-operation]
                (end - 1 - begin) / step  # pyrefly: ignore[unsupported-operation]
            )
            * step
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


def _max_target_value_expression(
    expressions: tuple[sympy.Expr, ...],
    *,
    source_domain: CoordinateDomain,
    source_bounds: tuple[tuple[int, IntegerExpression, IntegerExpression, int], ...],
) -> sympy.Expr:
    """Select a provably dominant target value without a costly symbolic Max."""
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
    domain: CoordinateDomain,
    source_bounds: tuple[tuple[int, IntegerExpression, IntegerExpression, int], ...],
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
    rebuilt = _simplify_integer_quotients(expression.func(*children))
    if rebuilt.func == sympy.ceiling:
        rebuilt = _normalize_integer_rounding(rebuilt)
    if rebuilt.func == sympy.floor:
        numerator, divisor = sympy.fraction(sympy.together(rebuilt.args[0]))
        divisor_is_positive_on_support = divisor.is_positive is True or any(
            sympy.simplify(domain.axis_count_expressions[axis] - divisor) == 0
            for axis in domain.axis_order
        )
        numerator_bounds = _logical_expression_bounds(
            cast("sympy.Expr", numerator),
            domain=domain,
            source_bounds=source_bounds,
        )
        if (
            divisor.is_integer is True
            and divisor_is_positive_on_support
            and numerator_bounds is not None
            and _is_provably_nonnegative(numerator_bounds[0], None)
            and _is_provably_nonnegative(
                sympy.simplify(divisor - 1 - numerator_bounds[1]),
                None,
            )
        ):
            return sympy.Integer(0)
    if expression.func == sympy.Mod and rebuilt != expression:
        return _simplify_logical_expression(
            cast("sympy.Expr", rebuilt),
            domain=domain,
            source_bounds=source_bounds,
        )
    if rebuilt.func == sympy.Add:
        terms = list(rebuilt.args)
        # Reassociate two adjacent quotient digits without expanding their
        # domain.  For integer ``x`` and positive static ``d`` dividing ``m``:
        #
        #   c * (m / d) * floor(x / m) + c * floor(Mod(x, m) / d)
        #       == c * floor(x / d)
        #
        # Flattening a worker/wave schedule naturally produces the left-hand
        # spelling.  Keeping it canonical lets the existing mixed-radix proof
        # see one digit instead of an artificial wave boundary.
        for inner_index, term in enumerate(tuple(terms)):
            coefficient, atom = term.as_coeff_Mul()  # pyrefly: ignore[missing-attribute]
            if atom.func != sympy.floor or len(atom.args) != 1:
                continue
            numerator, denominator = sympy.fraction(sympy.together(atom.args[0]))
            if not isinstance(numerator, sympy.Mod) or len(numerator.args) != 2:
                continue
            dividend, modulus = numerator.args
            if (
                coefficient.is_integer is not True
                or dividend.is_integer is not True
                or denominator.free_symbols
                or denominator.is_integer is not True
                or modulus.free_symbols
                or modulus.is_integer is not True
            ):
                continue
            divisor = int(denominator)
            period = int(modulus)
            if divisor <= 0 or period <= 0 or period % divisor:
                continue
            outer_term = (
                coefficient * (period // divisor) * sympy.floor(dividend / period)
            )
            matching_index = next(
                (
                    index
                    for index, other in enumerate(terms)
                    if index != inner_index and sympy.simplify(other - outer_term) == 0  # pyrefly: ignore[unsupported-operation]
                ),
                None,
            )
            if matching_index is None:
                continue
            remaining = [
                other
                for index, other in enumerate(terms)
                if index not in (inner_index, matching_index)
            ]
            rebuilt = sympy.Add(
                *remaining,
                coefficient * sympy.floor(dividend / divisor),
            )
            return _simplify_logical_expression(
                cast("sympy.Expr", rebuilt),
                domain=domain,
                source_bounds=source_bounds,
            )
        for modulo_index, term in enumerate(tuple(terms)):
            modulo_coefficient, modulo_term = term.as_coeff_Mul()  # pyrefly: ignore[missing-attribute]
            if modulo_term.func != sympy.Mod or len(modulo_term.args) != 2:
                continue
            dividend, modulus = modulo_term.args
            if (
                modulo_coefficient.is_number is not True
                or dividend.is_integer is not True  # pyrefly: ignore[missing-attribute]
                or modulus.is_integer is not True  # pyrefly: ignore[missing-attribute]
                or modulus.free_symbols
                or int(modulus) <= 0  # pyrefly: ignore[bad-argument-type]
            ):
                continue
            quotient_term = (
                modulo_coefficient * modulus * sympy.floor(dividend / modulus)  # pyrefly: ignore[unsupported-operation]
            )
            matching_index = next(
                (
                    index
                    for index, term in enumerate(terms)
                    if index != modulo_index
                    and sympy.simplify(term - quotient_term) == 0  # pyrefly: ignore[unsupported-operation]
                ),
                None,
            )
            if matching_index is None:
                continue
            remaining = [
                term
                for index, term in enumerate(terms)
                if index not in (modulo_index, matching_index)
            ]
            rebuilt = sympy.Add(*remaining, modulo_coefficient * dividend)
            return _simplify_logical_expression(
                cast("sympy.Expr", rebuilt),
                domain=domain,
                source_bounds=source_bounds,
            )
    bounds = _logical_expression_bounds(
        rebuilt,
        domain=domain,
        source_bounds=source_bounds,
    )
    if bounds is not None and sympy.simplify(bounds[1] - bounds[0]) == 0:  # pyrefly: ignore[unsupported-operation]
        return sympy.simplify(bounds[0])
    if rebuilt.func == sympy.Mod and len(children) == 2:
        modulus = children[1]
        dividend_bounds = _logical_expression_bounds(
            cast("sympy.Expr", children[0]),
            domain=domain,
            source_bounds=source_bounds,
        )
        if (
            modulus.is_integer is True  # pyrefly: ignore[missing-attribute]
            and dividend_bounds is not None
            and _is_provably_nonnegative(dividend_bounds[0], None)
            and _is_provably_nonnegative(
                modulus - 1 - dividend_bounds[1],  # pyrefly: ignore[unsupported-operation]
                None,
            )
        ):
            return cast("sympy.Expr", children[0])
        if (
            modulus.is_integer is True  # pyrefly: ignore[missing-attribute]
            and not modulus.free_symbols
            and int(modulus) > 0  # pyrefly: ignore[bad-argument-type]
            and dividend_bounds is not None
            and not any(value.free_symbols for value in dividend_bounds)
        ):
            lower_period = sympy.floor(dividend_bounds[0] / modulus)  # pyrefly: ignore[unsupported-operation]
            upper_period = sympy.floor(dividend_bounds[1] / modulus)  # pyrefly: ignore[unsupported-operation]
            if sympy.simplify(lower_period - upper_period) == 0:  # pyrefly: ignore[unsupported-operation]
                return sympy.simplify(  # pyrefly: ignore[bad-return]
                    children[0] - lower_period * modulus  # pyrefly: ignore[unsupported-operation]
                )
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
            _is_provably_nonnegative(
                sympy.simplify(other[0] - concrete[index][1]),
                None,
            )
            for other_index, other in enumerate(concrete)
            if other_index != index
        ):
            return cast("sympy.Expr", child)
        if rebuilt.func == sympy.Max and all(
            _is_provably_nonnegative(
                sympy.simplify(concrete[index][0] - other[1]),
                None,
            )
            for other_index, other in enumerate(concrete)
            if other_index != index
        ):
            return cast("sympy.Expr", child)
    return rebuilt


def _target_box_cardinality(
    target_ranges: tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...],
    *,
    target_domain: CoordinateDomain,
    source_domain: CoordinateDomain,
    source_bounds: tuple[tuple[int, IntegerExpression, IntegerExpression, int], ...],
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
            if begin_bounds is not None
            and _is_provably_nonnegative(begin_bounds[0], None)
            else sympy.Max(sympy.Integer(0), begin)
        )
        target_count = target_domain.axis_count_expressions[axis]
        clipped_end = (
            end
            if end_bounds is not None
            and _is_provably_nonnegative(
                sympy.simplify(target_count - end_bounds[1]),
                None,
            )
            else sympy.Min(
                target_count,
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
    source_domain: CoordinateDomain,
    source_bounds: tuple[tuple[int, IntegerExpression, IntegerExpression, int], ...],
    target_domain: CoordinateDomain,
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
            or not _is_provably_nonnegative(
                sympy.simplify(
                    target_domain.axis_count_expressions[axis] - 1 - begin_bounds[1]
                ),
                None,
            )
            or not _is_provably_nonnegative(
                sympy.simplify(end_bounds[0] - 1),
                None,
            )
            or not _is_provably_nonnegative(
                sympy.simplify(width_bounds[0] - 1),
                None,
            )
        ):
            return False
    return True


def _target_point_is_in_domain(
    target_ranges: tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...],
    *,
    source_domain: CoordinateDomain,
    source_bounds: tuple[tuple[int, IntegerExpression, IntegerExpression, int], ...],
    target_domain: CoordinateDomain,
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
        lower_is_in_domain = bounds is not None and _is_provably_nonnegative(
            bounds[0],
            None,
        )
        upper_is_in_domain = bounds is not None and _is_provably_nonnegative(
            sympy.simplify(
                target_domain.axis_count_expressions[axis] - 1 - bounds[1]
            ),
            None,
        )
        quotient = _static_integer_quotient(begin)
        if not upper_is_in_domain and quotient is not None:
            numerator, denominator = quotient
            numerator_bounds = _logical_expression_bounds(
                numerator,
                domain=source_domain,
                source_bounds=source_bounds,
            )
            upper_is_in_domain = (
                numerator_bounds is not None
                and _is_provably_nonnegative(numerator_bounds[0], None)
                and _is_provably_nonnegative(
                    sympy.simplify(
                        denominator
                        * target_domain.axis_count_expressions[axis]
                        - 1
                        - numerator_bounds[1]
                    ),
                    None,
                )
            )
        if (
            bounds is None
            or not lower_is_in_domain
            or not upper_is_in_domain
        ):
            return False
    return True


def _relation_piece_covers(
    available: _CoordinateRelationPiece,
    required: _CoordinateRelationPiece,
    *,
    target_domain: CoordinateDomain,
) -> bool:
    available_bounds = {
        axis: (begin, end, step)
        for axis, begin, end, step in available.source_bounds_items
    }
    for axis, begin, end, step in required.source_bounds_items:
        available_begin, available_end, available_step = available_bounds[axis]
        if (
            not _is_provably_nonnegative(
                sympy.simplify(begin - available_begin),
                None,
            )
            or not _is_provably_nonnegative(
                sympy.simplify(available_end - end),
                None,
            )
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
                available_end
                - sympy.sympify(target_domain.axis_count_expressions[axis])  # pyrefly: ignore[unsupported-operation]
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
    domain: CoordinateDomain,
) -> tuple[int, int, int, int] | None:
    """Recognize ``[stride * axis + offset, ... + width)`` exactly."""
    source_symbols: dict[sympy.Basic, int] = {
        coordinate_axis_symbol(axis): axis for axis in domain.axis_order
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


def _restore_positional_product(
    relation: CoordinateRelation,
    *,
    source_domain: CoordinateDomain,
    target_domain: CoordinateDomain,
    positional_axes: tuple[tuple[int, int], ...],
) -> CoordinateRelation:
    """Restore exact pointwise axes around a factored inner relation."""
    target_axis_by_source = dict(positional_axes)
    source_axis_by_target = {
        target_axis: source_axis for source_axis, target_axis in positional_axes
    }
    return CoordinateRelation(
        source_domain=source_domain,
        target_domain=target_domain,
        pieces=tuple(
            _CoordinateRelationPiece(
                source_bounds_items=tuple(
                    (
                        (
                            axis,
                            0,
                            source_domain.axis_count_expressions[axis],
                            1,
                        )
                        if axis in target_axis_by_source
                        else next(
                            bounds
                            for bounds in piece.source_bounds_items
                            if bounds[0] == axis
                        )
                    )
                    for axis in source_domain.axis_order
                ),
                target_ranges=tuple(
                    (
                        (
                            axis,
                            coordinate_axis_symbol(source_axis_by_target[axis]),
                            coordinate_axis_symbol(source_axis_by_target[axis]) + 1,
                            1,
                        )
                        if axis in source_axis_by_target
                        else next(
                            target_range
                            for target_range in piece.target_ranges
                            if target_range[0] == axis
                        )
                    )
                    for axis in target_domain.axis_order
                ),
            )
            for piece in relation.pieces
        ),
    )


def _static_affine_coefficients(
    expression: sympy.Expr,
    *,
    domain: CoordinateDomain,
) -> tuple[dict[int, IntegerExpression], IntegerExpression] | None:
    """Return coefficients constant over one coordinate domain."""
    expanded = sympy.expand(expression)
    remainder = expanded
    coefficients: dict[int, IntegerExpression] = {}
    for axis in domain.axis_order:
        symbol = coordinate_axis_symbol(axis)
        coefficient = sympy.simplify(expanded.coeff(symbol))
        if (
            coefficient.free_symbols - domain.parameter_symbols
            or coefficient.is_integer is not True
            or coefficient.is_nonnegative is not True
        ):
            return None
        coefficients[axis] = (
            int(coefficient) if isinstance(coefficient, sympy.Integer) else coefficient
        )
        remainder -= coefficient * symbol  # pyrefly: ignore[unsupported-operation]
    remainder = sympy.simplify(remainder)
    if (
        remainder.free_symbols - domain.parameter_symbols
        or remainder.is_integer is not True
    ):
        return None
    return coefficients, (
        int(remainder) if isinstance(remainder, sympy.Integer) else remainder
    )


def _coalesce_adjacent_target_boxes(
    pieces: tuple[_CoordinateRelationPiece, ...],
    *,
    source_domain: CoordinateDomain,
    prove_nonnegative: Callable[[sympy.Expr], bool] | None = None,
) -> tuple[_CoordinateRelationPiece, ...]:
    """Merge adjacent target boxes without unbounded normalization work."""

    def piece_key(piece: _CoordinateRelationPiece) -> tuple[object, ...]:
        return (
            tuple(
                (axis, sympy.srepr(begin), sympy.srepr(end), step)
                for axis, begin, end, step in piece.source_bounds_items
            ),
            tuple(
                (axis, sympy.srepr(begin), sympy.srepr(end), step)
                for axis, begin, end, step in piece.target_ranges
            ),
        )

    unique_pieces = sorted(dict.fromkeys(pieces), key=piece_key)
    comparisons = 0
    while True:
        merge: tuple[int, int, _CoordinateRelationPiece] | None = None
        for left_index, left in enumerate(tuple(unique_pieces)):
            for right_index in range(left_index + 1, len(unique_pieces)):
                comparisons += 1
                if comparisons > _MAX_RELATION_PRODUCT_STATES:
                    # The unmerged relation remains exact.  Downstream proofs
                    # may conservatively decline if they require a more
                    # compact form.
                    return tuple(unique_pieces)
                right = unique_pieces[right_index]
                if left.source_bounds_items != right.source_bounds_items:
                    continue
                differing_axes: list[int] = []
                merged_ranges: list[tuple[int, sympy.Expr, sympy.Expr, int]] = []
                for left_range, right_range in zip(
                    left.target_ranges,
                    right.target_ranges,
                    strict=True,
                ):
                    if left_range == right_range:
                        merged_ranges.append(left_range)
                        continue
                    axis, left_begin, left_end, left_step = left_range
                    right_axis, right_begin, right_end, right_step = right_range
                    differing_axes.append(axis)
                    left_before_right = (
                        left_end == right_begin
                        or sympy.simplify(  # pyrefly: ignore[unsupported-operation]
                            left_end - right_begin
                        )
                        == 0
                    )
                    right_before_left = (
                        right_end == left_begin
                        or sympy.simplify(  # pyrefly: ignore[unsupported-operation]
                            right_end - left_begin
                        )
                        == 0
                    )
                    if (
                        axis != right_axis
                        or left_step != 1
                        or right_step != 1
                        or not (left_before_right or right_before_left)
                    ):
                        continue
                    left_width_bounds = _logical_expression_bounds(
                        left_end - left_begin,  # pyrefly: ignore[unsupported-operation]
                        domain=source_domain,
                        source_bounds=left.source_bounds_items,
                    )
                    right_width_bounds = _logical_expression_bounds(
                        right_end - right_begin,  # pyrefly: ignore[unsupported-operation]
                        domain=source_domain,
                        source_bounds=right.source_bounds_items,
                    )
                    if (
                        left_width_bounds is None
                        or right_width_bounds is None
                        or not _is_provably_nonnegative(
                            left_width_bounds[0],
                            prove_nonnegative,
                        )
                        or not _is_provably_nonnegative(
                            right_width_bounds[0],
                            prove_nonnegative,
                        )
                    ):
                        continue
                    merged_ranges.append(
                        (
                            axis,
                            left_begin if left_before_right else right_begin,
                            right_end if left_before_right else left_end,
                            1,
                        )
                    )
                if len(differing_axes) != 1 or len(merged_ranges) != len(
                    left.target_ranges
                ):
                    continue
                merge = (
                    left_index,
                    right_index,
                    _CoordinateRelationPiece(
                        source_bounds_items=left.source_bounds_items,
                        target_ranges=tuple(merged_ranges),
                    ),
                )
                break
            if merge is not None:
                break
        if merge is None:
            return tuple(unique_pieces)
        left_index, right_index, merged_piece = merge
        unique_pieces = sorted(
            [
                merged_piece if index == left_index else piece
                for index, piece in enumerate(unique_pieces)
                if index != right_index
            ],
            key=piece_key,
        )


def _dense_linear_overlap_relation(
    producer_relation: CoordinateRelation,
    consumer_relation: CoordinateRelation,
    *,
    prove_nonnegative: Callable[[sympy.Expr], bool] | None = None,
) -> CoordinateRelation | None:
    """Map exact linear-view accesses back to a dense producer task grid.

    The producer must tile one linear allocation densely in mixed-radix order.
    Each consumer piece may either stay within one producer tile or select an
    arithmetic progression of producer tiles along exactly one producer axis.
    This covers flattened reduction gathers without replacing them by an
    inexact contiguous hull.
    """
    if (
        producer_relation.target_domain != consumer_relation.target_domain
        or producer_relation.target_domain.axis_order != (_ALLOCATION_ADDRESS_AXIS,)
        or len(producer_relation.pieces) != 1
    ):
        return None
    producer_domain = producer_relation.source_domain
    producer_counts = producer_domain.axis_count_expressions
    producer_piece = producer_relation.pieces[0]
    if (
        producer_piece.source_bounds_items
        != tuple(
            (axis, 0, producer_counts[axis], 1) for axis in producer_domain.axis_order
        )
        or len(producer_piece.target_ranges) != 1
    ):
        return None
    _axis, producer_begin, producer_end, producer_step = producer_piece.target_ranges[0]
    producer_width_expr = sympy.simplify(producer_end - producer_begin)  # pyrefly: ignore[unsupported-operation]
    affine = _static_affine_coefficients(producer_begin, domain=producer_domain)
    if (
        producer_step != 1
        or affine is None
        or not isinstance(producer_width_expr, sympy.Integer)
    ):
        return None
    coefficients, producer_offset = affine
    producer_width = int(producer_width_expr)
    if producer_width <= 0:
        return None
    remaining_axes = {
        axis
        for axis in producer_domain.axis_order
        if sympy.simplify(producer_counts[axis] - 1) != 0
    }
    active_axes: list[int] = []
    dense_span: sympy.Expr = sympy.Integer(producer_width)
    tile_strides: dict[int, sympy.Expr] = {}
    while remaining_axes:
        matching_axes = tuple(
            axis
            for axis in remaining_axes
            if sympy.simplify(coefficients[axis] - dense_span) == 0
        )
        if len(matching_axes) != 1:
            return None
        (axis,) = matching_axes
        active_axes.append(axis)
        tile_stride = sympy.simplify(dense_span / producer_width)  # pyrefly: ignore[unsupported-operation]
        if tile_stride.is_integer is not True:
            return None
        tile_strides[axis] = tile_stride
        dense_span = sympy.simplify(dense_span * producer_counts[axis])
        remaining_axes.remove(axis)
    if any(
        coefficients[axis] != 0
        for axis in producer_domain.axis_order
        if sympy.simplify(producer_counts[axis] - 1) == 0
    ):
        return None
    allocation_count = producer_relation.target_domain.axis_count_expressions[
        _ALLOCATION_ADDRESS_AXIS
    ]
    remaining_allocation = sympy.simplify(
        allocation_count - producer_offset - dense_span
    )
    if not _is_provably_nonnegative(
        producer_offset, prove_nonnegative
    ) or not _is_provably_nonnegative(remaining_allocation, prove_nonnegative):
        return None

    pieces: list[_CoordinateRelationPiece] = []
    for consumer_piece in consumer_relation.pieces:
        if len(consumer_piece.target_ranges) != 1:
            return None
        _axis, consumer_begin, consumer_end, consumer_step = (
            consumer_piece.target_ranges[0]
        )
        begin_delta = sympy.simplify(consumer_begin - producer_offset)  # pyrefly: ignore[unsupported-operation]
        end_delta = sympy.simplify(consumer_end - producer_offset)  # pyrefly: ignore[unsupported-operation]
        begin_bounds = _logical_expression_bounds(
            begin_delta,
            domain=consumer_relation.source_domain,
            source_bounds=consumer_piece.source_bounds_items,
        )
        last_address = sympy.simplify(end_delta - consumer_step)  # pyrefly: ignore[unsupported-operation]
        last_bounds = _logical_expression_bounds(
            last_address,
            domain=consumer_relation.source_domain,
            source_bounds=consumer_piece.source_bounds_items,
        )
        width_expr = sympy.simplify(end_delta - begin_delta)
        if (
            begin_bounds is None
            or last_bounds is None
            or not _is_provably_nonnegative(begin_bounds[0], prove_nonnegative)
            or not _is_provably_nonnegative(
                sympy.simplify(
                    dense_span - 1 - last_bounds[1]  # pyrefly: ignore[unsupported-operation]
                ),
                prove_nonnegative,
            )
            or not isinstance(width_expr, sympy.Integer)
        ):
            return None
        width = int(width_expr)
        if width <= 0:
            return None

        first_ordinal = sympy.floor(begin_delta / producer_width)
        if (
            consumer_step == 1
            and width <= producer_width
            and producer_width % width == 0
            and sympy.simplify(sympy.Mod(begin_delta, width)) == 0
        ):
            ordinal_count = 1
            ordinal_step = 1
        elif (
            consumer_step >= producer_width
            and consumer_step % producer_width == 0
            and width % consumer_step == 0
        ):
            ordinal_count = width // consumer_step
            ordinal_step = consumer_step // producer_width
        else:
            return None

        varying_axis = None
        if ordinal_count > 1:
            varying_axis = next(
                (axis for axis in active_axes if tile_strides[axis] == ordinal_step),
                None,
            )
            if varying_axis is None:
                return None

        target_ranges: dict[int, tuple[sympy.Expr, sympy.Expr, int]] = {}
        for axis in producer_domain.axis_order:
            producer_count = producer_counts[axis]
            if sympy.simplify(producer_count - 1) == 0:
                coordinate: sympy.Expr = sympy.Integer(0)
            else:
                tile_stride = tile_strides[axis]
                coordinate = cast(
                    "sympy.Expr",
                    sympy.Mod(
                        sympy.floor(first_ordinal / tile_stride),
                        producer_count,
                    ),
                )
                for _ in range(2):
                    coordinate = _simplify_logical_expression(
                        coordinate,
                        domain=consumer_relation.source_domain,
                        source_bounds=consumer_piece.source_bounds_items,
                    )
            count = ordinal_count if axis == varying_axis else 1
            coordinate_bounds = _logical_expression_bounds(
                coordinate,
                domain=consumer_relation.source_domain,
                source_bounds=consumer_piece.source_bounds_items,
            )
            if (
                coordinate_bounds is None
                or not _is_provably_nonnegative(coordinate_bounds[0], prove_nonnegative)
                or not _is_provably_nonnegative(
                    sympy.simplify(producer_count - count - coordinate_bounds[1]),
                    prove_nonnegative,
                )
            ):
                return None
            target_ranges[axis] = (
                coordinate,
                coordinate + count,  # pyrefly: ignore[unsupported-operation]
                1,
            )
        pieces.append(
            _CoordinateRelationPiece(
                source_bounds_items=consumer_piece.source_bounds_items,
                target_ranges=tuple(
                    (axis, *target_ranges[axis]) for axis in producer_domain.axis_order
                ),
            )
        )
    return CoordinateRelation(
        source_domain=consumer_relation.source_domain,
        target_domain=producer_domain,
        pieces=tuple(pieces),
    ).coalesce_adjacent_target_boxes(prove_nonnegative=prove_nonnegative)


def _symbolic_single_source_mixed_radix_converse(
    relation: CoordinateRelation,
) -> CoordinateRelation | None:
    """Invert a full symbolic scalar traversal by a bounded radix search.

    This is the symbolic counterpart of the existing concrete mixed-radix
    proofs.  It permits one or more target extents to remain parameterized;
    the candidate inverse is accepted only when substituting the forward map
    reconstructs the source ordinal identically over its complete domain.
    Equal finite domain cardinalities then make that injection a bijection.
    """
    if (
        not relation.parameter_symbols
        or len(relation.source_domain.axis_order) != 1
        or len(relation.pieces) != 1
        or len(relation.target_domain.axis_order) > 6
        or not _integer_partition_expressions_equal(
            relation.source_domain.size_expr,
            relation.target_domain.size_expr,
        )
    ):
        return None
    (piece,) = relation.pieces
    (source_axis,) = relation.source_domain.axis_order
    full_source_bounds = (
        (
            source_axis,
            sympy.Integer(0),
            relation.source_domain.size_expr,
            1,
        ),
    )
    if piece.source_bounds_items != full_source_bounds or any(
        step != 1 or not _integer_partition_expressions_equal(end - begin, 1)
        for _axis, begin, end, step in piece.target_ranges
    ):
        return None
    target_counts = relation.target_domain.axis_count_expressions
    if any(sympy.sympify(count).is_positive is not True for count in target_counts.values()):
        return None

    target_axes = relation.target_domain.axis_order
    target_expression_by_axis = {
        axis: begin for axis, begin, _end, _step in piece.target_ranges
    }
    if any(
        not _integer_partition_expressions_equal(
            target_expression_by_axis[axis],
            0,
        )
        for axis in target_axes
        if _integer_partition_expressions_equal(target_counts[axis], 1)
    ):
        return None
    varying_axes = tuple(
        axis
        for axis in target_axes
        if not _integer_partition_expressions_equal(
            target_counts[axis],
            1,
        )
    )
    candidate_count = math.factorial(len(varying_axes)) * (1 << len(varying_axes))
    if candidate_count > _MAX_RELATION_PRODUCT_STATES:
        return None
    source_symbol = coordinate_axis_symbol(source_axis)
    target_bounds = tuple(
        (
            axis,
            sympy.Integer(0),
            target_counts[axis],
            1,
        )
        for axis in target_axes
    )
    for axis_order in itertools.permutations(varying_axes):
        for reflected_mask in range(1 << len(axis_order)):
            inverse: sympy.Expr = sympy.Integer(0)
            stride: sympy.Expr = sympy.Integer(1)
            matches = True
            for index, axis in enumerate(axis_order):
                count = target_counts[axis]
                coordinate = coordinate_axis_symbol(axis)
                quotient = (
                    source_symbol
                    if _integer_partition_expressions_equal(stride, 1)
                    else cast("sympy.Expr", FloorDiv(source_symbol, stride))
                )
                ordinary_digit = (
                    quotient
                    if index == len(axis_order) - 1
                    else sympy.Mod(quotient, count)
                    if not count.free_symbols
                    else sympy.simplify(
                        quotient
                        - cast("sympy.Expr", FloorDiv(quotient, count)) * count
                    )
                )
                expected_digit = (
                    count - 1 - ordinary_digit  # pyrefly: ignore[unsupported-operation]
                    if reflected_mask & (1 << index)
                    else ordinary_digit
                )
                if not _integer_partition_expressions_equal(
                    target_expression_by_axis[axis],
                    expected_digit,
                ):
                    matches = False
                    break
                digit = (
                    count - 1 - coordinate  # pyrefly: ignore[unsupported-operation]
                    if reflected_mask & (1 << index)
                    else coordinate
                )
                inverse += stride * digit  # pyrefly: ignore[unsupported-operation]
                stride = sympy.simplify(stride * count)
            if not matches:
                continue
            converse = CoordinateRelation.point_map(
                relation.target_domain,
                relation.source_domain,
                ((target_bounds, (inverse,)),),
            )
            return converse
    return None


def _derived_converse(
    relation: CoordinateRelation,
    target_counts: CoordinateRelation,
) -> CoordinateRelation | None:
    """Run the canonical non-enumerative converse derivations in one place."""
    for derive in (
        lambda: _dense_mixed_radix_converse(relation, target_counts),
        lambda: _piecewise_dense_point_converse(relation),
        lambda: _piecewise_source_grouped_mixed_radix_converse(relation),
        lambda: _piecewise_separable_dense_point_converse(relation),
        lambda: _piecewise_woven_mixed_radix_converse(relation),
        lambda: _piecewise_single_source_mixed_radix_converse(relation),
    ):
        if (converse := derive()) is not None:
            return converse
    return None


def _single_ordinal_quotient_stride(
    expression: sympy.Expr,
    source_symbol: sympy.Symbol,
) -> int | None:
    """Recognize ``floor(source / stride)`` (with stride one implicit)."""
    if sympy.simplify(expression - source_symbol) == 0:  # pyrefly: ignore[unsupported-operation]
        return 1
    quotient = _static_integer_quotient(expression)
    if (
        quotient is None
        or sympy.simplify(quotient[0] - source_symbol) != 0  # pyrefly: ignore[unsupported-operation]
    ):
        return None
    return quotient[1]


def _single_ordinal_digit(
    expression: sympy.Expr,
    *,
    source_symbol: sympy.Symbol,
    source_count: int,
) -> tuple[int, int] | None:
    """Recognize one dense mixed-radix digit of a bounded flat ordinal.

    The result is ``(input_stride, radix)``.  Both the conventional
    ``Mod(floor(x / stride), radix)`` spelling and the equivalent
    ``floor(Mod(x, period) / stride)`` spelling occur after SymPy
    simplification of configured PID permutations.
    """
    expression = sympy.simplify(expression)
    if isinstance(expression, sympy.Mod) and len(expression.args) == 2:
        dividend, modulus = expression.args
        stride = _single_ordinal_quotient_stride(
            cast("sympy.Expr", dividend), source_symbol
        )
        if (
            stride is None
            or modulus.free_symbols
            or modulus.is_integer is not True
            or int(modulus) <= 1
        ):
            return None
        radix = int(modulus)
        period = stride * radix
        return (stride, radix) if source_count % period == 0 else None

    quotient = _static_integer_quotient(expression)
    if quotient is None:
        if sympy.simplify(expression - source_symbol) == 0:  # pyrefly: ignore[unsupported-operation]
            return (1, source_count) if source_count > 1 else None
        return None
    numerator, stride = quotient
    if isinstance(numerator, sympy.Mod) and len(numerator.args) == 2:
        dividend, period = numerator.args
        if (
            sympy.simplify(dividend - source_symbol) != 0  # pyrefly: ignore[unsupported-operation]
            or period.free_symbols
            or period.is_integer is not True
            or int(period) <= stride
            or int(period) % stride
            or source_count % int(period)
        ):
            return None
        return stride, int(period) // stride
    if (
        sympy.simplify(numerator - source_symbol) != 0  # pyrefly: ignore[unsupported-operation]
        or source_count % stride
        or source_count // stride <= 1
    ):
        return None
    return stride, source_count // stride


def _piecewise_woven_mixed_radix_converse(
    relation: CoordinateRelation,
) -> CoordinateRelation | None:
    """Invert a dense partition of one ordinal's mixed-radix digits.

    Task-order permutations can split one source digit and weave its pieces
    across several logical axes.  A full multidimensional converse is then
    unnecessarily hard to represent even though logical task -> flat source
    ordinal remains a compact function.  This derivation proves that every
    source digit appears exactly once, that each target coordinate densely
    packs its assigned digits, and that different source pieces have disjoint
    target boxes.  It never enumerates domain points.
    """
    if not relation.pieces:
        return None
    nontrivial_source_axes = tuple(
        axis
        for axis in relation.source_domain.axis_order
        if relation.source_domain.axis_counts[axis] != 1
    )
    if len(nontrivial_source_axes) != 1:
        return None
    if len(relation.source_domain.axis_order) != 1:
        reduced_source = CoordinateDomain(
            axis_order=nontrivial_source_axes,
            axis_counts_items=tuple(
                (axis, relation.source_domain.axis_counts[axis])
                for axis in nontrivial_source_axes
            ),
            kind=relation.source_domain.kind,
            identity=relation.source_domain.identity,
        )
        projected = relation.project_source(reduced_source)
        reduced_converse = (
            None
            if projected is None
            else _piecewise_woven_mixed_radix_converse(projected)
        )
        if reduced_converse is None:
            return None
        reduced_ranges = {
            piece.source_bounds_items: {
                axis: (begin, end, step)
                for axis, begin, end, step in piece.target_ranges
            }
            for piece in reduced_converse.pieces
        }
        result = CoordinateRelation(
            source_domain=reduced_converse.source_domain,
            target_domain=relation.source_domain,
            pieces=tuple(
                _CoordinateRelationPiece(
                    source_bounds_items=source_bounds,
                    target_ranges=tuple(
                        (
                            (axis, *ranges[axis])
                            if axis in ranges
                            else (
                                axis,
                                sympy.Integer(0),
                                sympy.Integer(1),
                                1,
                            )
                        )
                        for axis in relation.source_domain.axis_order
                    ),
                )
                for source_bounds, ranges in reduced_ranges.items()
            ),
        )
        return result if result.canonical_single_valued() is not None else None
    (source_axis,) = relation.source_domain.axis_order
    source_symbol = coordinate_axis_symbol(source_axis)
    refined_pieces: list[_CoordinateRelationPiece] = []
    for piece in relation.pieces:
        ((bound_axis, source_begin, source_end, source_step),) = (
            piece.source_bounds_items
        )
        if bound_axis != source_axis or source_step != 1:
            return None
        target_box_cardinality = 1
        for _target_axis, begin, _end, _target_step in piece.target_ranges:
            bounds = _logical_expression_bounds(
                begin,
                domain=relation.source_domain,
                source_bounds=piece.source_bounds_items,
            )
            if (
                bounds is None
                or any(value.free_symbols for value in bounds)
                or any(value.is_integer is not True for value in bounds)
            ):
                target_box_cardinality = -1
                break
            target_box_cardinality *= int(bounds[1] - bounds[0] + 1)
        if target_box_cardinality == source_end - source_begin:
            refined_pieces.append(piece)
            continue
        cuts = {source_begin, source_end}
        for _target_axis, begin, _end, _target_step in piece.target_ranges:
            for subexpression in sympy.preorder_traversal(begin):
                if not isinstance(subexpression, sympy.Mod):
                    continue
                dividend, modulus = subexpression.args
                if (
                    modulus.free_symbols
                    or modulus.is_integer is not True
                    or int(modulus) <= 1
                ):
                    continue
                layout = _static_affine_coefficients(
                    cast("sympy.Expr", dividend),
                    domain=relation.source_domain,
                )
                if layout is None:
                    continue
                coefficients, offset = layout
                if coefficients[source_axis] != 1 or any(
                    coefficient
                    for axis, coefficient in coefficients.items()
                    if axis != source_axis
                ):
                    continue
                period = int(modulus)
                first_wrap = source_begin + (-offset - source_begin) % period
                if first_wrap == source_begin:
                    first_wrap += period
                last_wrap = source_end - (source_end + offset) % period
                if source_begin < first_wrap < source_end:
                    cuts.add(first_wrap)
                if source_begin < last_wrap < source_end:
                    cuts.add(last_wrap)
        refined_pieces.extend(
            dataclasses.replace(
                piece,
                source_bounds_items=((source_axis, begin, end, 1),),
            )
            for begin, end in itertools.pairwise(sorted(cuts))
        )
    converse_pieces: list[_CoordinateRelationPiece] = []
    source_intervals: list[tuple[int, int]] = []
    target_boxes: list[tuple[tuple[int, int, int, int], ...]] = []
    for piece in refined_pieces:
        ((bound_axis, source_begin, source_end, source_step),) = (
            piece.source_bounds_items
        )
        if (
            bound_axis != source_axis
            or source_step != 1
            or source_begin >= source_end
            or any(
                max(source_begin, other_begin) < min(source_end, other_end)
                for other_begin, other_end in source_intervals
            )
        ):
            return None
        source_intervals.append((source_begin, source_end))
        source_count = source_end - source_begin
        local_bounds = ((source_axis, 0, source_count, 1),)
        substitutions = {source_symbol: source_symbol + source_begin}
        target_box: list[tuple[int, int, int, int]] = []
        # (input stride, radix, target axis, packed-output stride, reflected)
        source_digits: list[tuple[int, int, int, int, bool]] = []
        for target_axis, begin, end, target_step in piece.target_ranges:
            begin = _simplify_logical_expression(
                cast("sympy.Expr", begin.xreplace(substitutions)),
                domain=relation.source_domain,
                source_bounds=local_bounds,
            )
            end = _simplify_logical_expression(
                cast("sympy.Expr", end.xreplace(substitutions)),
                domain=relation.source_domain,
                source_bounds=local_bounds,
            )
            if target_step != 1 or sympy.simplify(end - begin) != 1:  # pyrefly: ignore[unsupported-operation]
                return None
            bounds = _logical_expression_bounds(
                begin,
                domain=relation.source_domain,
                source_bounds=local_bounds,
            )
            if (
                bounds is None
                or any(value.free_symbols for value in bounds)
                or any(value.is_integer is not True for value in bounds)
            ):
                return None
            support_begin, support_last = (int(value) for value in bounds)
            support_end = support_last + 1
            if (
                support_begin < 0
                or support_end > relation.target_domain.axis_counts[target_axis]
            ):
                return None
            target_box.append((target_axis, support_begin, support_end, 1))
            local_expression = _simplify_logical_expression(
                begin - support_begin,  # pyrefly: ignore[unsupported-operation]
                domain=relation.source_domain,
                source_bounds=local_bounds,
            )
            packed_digits: list[tuple[int, int, int, bool]] = []
            constant = 0
            if local_expression != 0:
                for term in sympy.Add.make_args(sympy.expand(local_expression)):
                    if not term.free_symbols:
                        if term.is_integer is not True:
                            return None
                        constant += int(term)
                        continue
                    coefficient, atom = term.as_coeff_Mul()
                    if coefficient.is_integer is not True or int(coefficient) == 0:
                        return None
                    digit = _single_ordinal_digit(
                        cast("sympy.Expr", atom),
                        source_symbol=source_symbol,
                        source_count=source_count,
                    )
                    if digit is None:
                        return None
                    input_stride, radix = digit
                    signed_stride = int(coefficient)
                    packed_digits.append(
                        (
                            abs(signed_stride),
                            input_stride,
                            radix,
                            signed_stride < 0,
                        )
                    )
            reflected_constant = sum(
                output_stride * (radix - 1)
                for output_stride, _input_stride, radix, reflected in packed_digits
                if reflected
            )
            if constant != reflected_constant:
                return None
            expected_output_stride = 1
            for output_stride, input_stride, radix, reflected in sorted(packed_digits):
                if output_stride != expected_output_stride:
                    return None
                expected_output_stride *= radix
                source_digits.append(
                    (
                        input_stride,
                        radix,
                        target_axis,
                        output_stride,
                        reflected,
                    )
                )
            if expected_output_stride != support_end - support_begin:
                return None

        expected_input_stride = 1
        for (
            input_stride,
            radix,
            _target_axis,
            _output_stride,
            _reflected,
        ) in sorted(source_digits):
            if input_stride != expected_input_stride:
                return None
            expected_input_stride *= radix
        if expected_input_stride != source_count:
            return None
        target_box_tuple = tuple(target_box)
        if any(
            not _source_bounds_are_disjoint(previous, target_box_tuple)
            for previous in target_boxes
        ):
            return None
        target_boxes.append(target_box_tuple)
        support_begin_by_axis = {
            axis: begin for axis, begin, _end, _step in target_box_tuple
        }
        inverse_source: sympy.Expr = sympy.Integer(source_begin)
        for (
            input_stride,
            radix,
            target_axis,
            output_stride,
            reflected,
        ) in source_digits:
            local_target = (
                coordinate_axis_symbol(target_axis) - support_begin_by_axis[target_axis]
            )
            target_digit = sympy.Mod(  # pyrefly: ignore[bad-argument-type]
                sympy.floor(local_target / output_stride),  # pyrefly: ignore[bad-argument-type, unsupported-operation]
                radix,
            )
            source_digit = (
                radix - 1 - target_digit  # pyrefly: ignore[unsupported-operation]
                if reflected
                else target_digit
            )
            inverse_source += input_stride * source_digit  # pyrefly: ignore[unsupported-operation]
        converse_pieces.append(
            _CoordinateRelationPiece(
                source_bounds_items=target_box_tuple,
                target_ranges=(
                    (
                        source_axis,
                        inverse_source,
                        inverse_source + 1,  # pyrefly: ignore[unsupported-operation]
                        1,
                    ),
                ),
            )
        )
    result = CoordinateRelation(
        source_domain=relation.target_domain,
        target_domain=relation.source_domain,
        pieces=tuple(converse_pieces),
    )
    return result if result.canonical_single_valued() is not None else None


def _piecewise_single_source_mixed_radix_converse(
    relation: CoordinateRelation,
) -> CoordinateRelation | None:
    """Invert piecewise dense unflattening of one source coordinate.

    A sliced schedule commonly maps one dense ordinal to several logical task
    axes with ``Mod``/``floor`` digits.  For each source interval, prove that
    those digit expressions reconstruct the source ordinal and that the image
    has the same cardinality as its Cartesian target box.  The resulting
    inverse remains piecewise symbolic; no task coordinate is enumerated.
    """
    if not relation.pieces:
        return None
    converse_pieces: list[_CoordinateRelationPiece] = []
    source_boxes: list[tuple[tuple[int, int, int, int], ...]] = []
    target_boxes: list[tuple[tuple[int, int, int, int], ...]] = []
    for piece in relation.pieces:
        if any(
            not _source_bounds_are_disjoint(previous, piece.source_bounds_items)
            for previous in source_boxes
        ):
            return None
        source_boxes.append(piece.source_bounds_items)
        source_bounds = {
            axis: (begin, end, step)
            for axis, begin, end, step in piece.source_bounds_items
        }
        if any(step != 1 for _begin, _end, step in source_bounds.values()):
            return None
        varying_source_axes = tuple(
            axis
            for axis, (begin, end, _step) in source_bounds.items()
            if end - begin > 1
        )
        if len(varying_source_axes) != 1:
            return None
        (source_axis,) = varying_source_axes
        source_begin, source_end, _source_step = source_bounds[source_axis]
        source_symbol = coordinate_axis_symbol(source_axis)
        target_expressions: dict[int, sympy.Expr] = {}
        target_extents: dict[int, tuple[int, int]] = {}
        for target_axis, begin, end, step in piece.target_ranges:
            begin = _simplify_logical_expression(
                begin,
                domain=relation.source_domain,
                source_bounds=piece.source_bounds_items,
            )
            end = _simplify_logical_expression(
                end,
                domain=relation.source_domain,
                source_bounds=piece.source_bounds_items,
            )
            if step != 1 or sympy.simplify(end - begin) != 1:  # pyrefly: ignore[unsupported-operation]
                return None
            bounds = _logical_expression_bounds(
                begin,
                domain=relation.source_domain,
                source_bounds=piece.source_bounds_items,
            )
            if (
                bounds is None
                or any(value.free_symbols for value in bounds)
                or any(value.is_integer is not True for value in bounds)
            ):
                return None
            minimum, maximum = (int(value) for value in bounds)
            if (
                minimum < 0
                or maximum >= relation.target_domain.axis_counts[target_axis]
            ):
                return None
            target_expressions[target_axis] = begin
            target_extents[target_axis] = (minimum, maximum + 1)

        varying_target_axes = tuple(
            axis
            for axis in relation.target_domain.axis_order
            if target_extents[axis][1] - target_extents[axis][0] > 1
        )
        if len(varying_target_axes) > 6:
            return None
        source_count = source_end - source_begin
        if (
            math.prod(
                target_extents[axis][1] - target_extents[axis][0]
                for axis in relation.target_domain.axis_order
            )
            != source_count
        ):
            return None

        digit_strides: dict[int, int] | None = None
        for axis_order in itertools.permutations(varying_target_axes):
            candidate_strides: dict[int, int] = {}
            stride = 1
            reconstructed: sympy.Expr = sympy.Integer(source_begin)
            for target_axis in axis_order:
                minimum, end = target_extents[target_axis]
                candidate_strides[target_axis] = stride
                reconstructed += (target_expressions[target_axis] - minimum) * stride  # pyrefly: ignore[unsupported-operation]
                stride *= end - minimum
            difference = _simplify_logical_expression(
                reconstructed - source_symbol,  # pyrefly: ignore[unsupported-operation]
                domain=relation.source_domain,
                source_bounds=piece.source_bounds_items,
            )
            if difference == 0:
                digit_strides = candidate_strides
                break
        if digit_strides is None:
            return None

        target_box = tuple(
            (axis, *target_extents[axis], 1)
            for axis in relation.target_domain.axis_order
        )
        if any(
            not _source_bounds_are_disjoint(previous, target_box)
            for previous in target_boxes
        ):
            return None
        target_boxes.append(target_box)
        inverse_source = sympy.Integer(source_begin)
        for target_axis, stride in digit_strides.items():
            minimum, _end = target_extents[target_axis]
            inverse_source += (coordinate_axis_symbol(target_axis) - minimum) * stride  # pyrefly: ignore[unsupported-operation]
        converse_pieces.append(
            _CoordinateRelationPiece(
                source_bounds_items=target_box,
                target_ranges=tuple(
                    (
                        axis,
                        inverse_source
                        if axis == source_axis
                        else sympy.Integer(source_bounds[axis][0]),
                        (
                            inverse_source
                            if axis == source_axis
                            else sympy.Integer(source_bounds[axis][0])
                        )
                        + 1,  # pyrefly: ignore[unsupported-operation]
                        1,
                    )
                    for axis in relation.source_domain.axis_order
                ),
            )
        )
    result = CoordinateRelation(
        source_domain=relation.target_domain,
        target_domain=relation.source_domain,
        pieces=tuple(converse_pieces),
    )
    return result if result.canonical_single_valued() is not None else None


def _piecewise_source_grouped_mixed_radix_converse(
    relation: CoordinateRelation,
) -> CoordinateRelation | None:
    """Invert independent unflattenings of several source coordinates.

    A compact task order can preserve an outer cohort coordinate while
    unflattening an inner ordinal across several logical task coordinates.
    Each target coordinate must depend on at most one varying source axis;
    the target coordinates assigned to one source axis must jointly form an
    exact mixed-radix representation of that source interval.  This proves the
    inverse from rectangular pieces alone and never enumerates tasks.
    """
    if not relation.pieces:
        return None
    converse_pieces: list[_CoordinateRelationPiece] = []
    source_boxes: list[tuple[tuple[int, int, int, int], ...]] = []
    target_boxes: list[tuple[tuple[int, int, int, int], ...]] = []
    source_symbols = {
        coordinate_axis_symbol(axis): axis for axis in relation.source_domain.axis_order
    }
    for piece in relation.pieces:
        if any(
            not _source_bounds_are_disjoint(previous, piece.source_bounds_items)
            for previous in source_boxes
        ):
            return None
        source_boxes.append(piece.source_bounds_items)
        source_bounds = {
            axis: (begin, end, step)
            for axis, begin, end, step in piece.source_bounds_items
        }
        if any(step != 1 for _begin, _end, step in source_bounds.values()):
            return None

        target_expressions: dict[int, sympy.Expr] = {}
        target_extents: dict[int, tuple[int, int]] = {}
        target_axes_by_source: dict[int, list[int]] = {}
        for target_axis, begin, end, step in piece.target_ranges:
            begin = _simplify_logical_expression(
                begin,
                domain=relation.source_domain,
                source_bounds=piece.source_bounds_items,
            )
            end = _simplify_logical_expression(
                end,
                domain=relation.source_domain,
                source_bounds=piece.source_bounds_items,
            )
            if step != 1 or sympy.simplify(end - begin) != 1:  # pyrefly: ignore[unsupported-operation]
                return None
            expression_source_axes = {
                source_symbols[symbol]
                for symbol in begin.free_symbols
                if symbol in source_symbols
                and source_bounds[source_symbols[symbol]][1]
                - source_bounds[source_symbols[symbol]][0]
                > 1
            }
            if (
                len(expression_source_axes)
                != len(begin.free_symbols & source_symbols.keys())
                or len(expression_source_axes) > 1
            ):
                return None
            bounds = _logical_expression_bounds(
                begin,
                domain=relation.source_domain,
                source_bounds=piece.source_bounds_items,
            )
            if (
                bounds is None
                or any(value.free_symbols for value in bounds)
                or any(value.is_integer is not True for value in bounds)
            ):
                return None
            minimum, maximum = (int(value) for value in bounds)
            if (
                minimum < 0
                or maximum >= relation.target_domain.axis_counts[target_axis]
            ):
                return None
            target_expressions[target_axis] = begin
            target_extents[target_axis] = (minimum, maximum + 1)
            if expression_source_axes:
                (source_axis,) = expression_source_axes
                target_axes_by_source.setdefault(source_axis, []).append(target_axis)

        inverse_by_source_axis: dict[int, sympy.Expr] = {}
        for source_axis in relation.source_domain.axis_order:
            source_begin, source_end, _source_step = source_bounds[source_axis]
            source_count = source_end - source_begin
            if source_count == 1:
                inverse_by_source_axis[source_axis] = sympy.Integer(source_begin)
                continue
            target_axes = tuple(target_axes_by_source.get(source_axis, ()))
            if not target_axes or len(target_axes) > 6:
                return None
            if (
                math.prod(
                    target_extents[axis][1] - target_extents[axis][0]
                    for axis in target_axes
                )
                != source_count
            ):
                return None
            source_symbol = coordinate_axis_symbol(source_axis)
            inverse: sympy.Expr | None = None
            for axis_order in itertools.permutations(target_axes):
                stride = 1
                reconstructed: sympy.Expr = sympy.Integer(source_begin)
                for target_axis in axis_order:
                    minimum, target_end = target_extents[target_axis]
                    reconstructed += (  # pyrefly: ignore[unsupported-operation]
                        target_expressions[target_axis] - minimum  # pyrefly: ignore[unsupported-operation]
                    ) * stride
                    stride *= target_end - minimum
                difference = _simplify_logical_expression(
                    reconstructed - source_symbol,  # pyrefly: ignore[unsupported-operation]
                    domain=relation.source_domain,
                    source_bounds=piece.source_bounds_items,
                )
                if difference != 0:
                    continue
                inverse = sympy.Integer(source_begin)
                stride = 1
                for target_axis in axis_order:
                    minimum, target_end = target_extents[target_axis]
                    inverse += (  # pyrefly: ignore[unsupported-operation]
                        coordinate_axis_symbol(target_axis) - minimum  # pyrefly: ignore[unsupported-operation]
                    ) * stride
                    stride *= target_end - minimum
                break
            if inverse is None:
                return None
            inverse_by_source_axis[source_axis] = inverse

        target_box = tuple(
            (axis, *target_extents[axis], 1)
            for axis in relation.target_domain.axis_order
        )
        if any(
            not _source_bounds_are_disjoint(previous, target_box)
            for previous in target_boxes
        ):
            return None
        target_boxes.append(target_box)
        converse_pieces.append(
            _CoordinateRelationPiece(
                source_bounds_items=target_box,
                target_ranges=tuple(
                    (
                        axis,
                        inverse_by_source_axis[axis],
                        inverse_by_source_axis[axis] + 1,  # pyrefly: ignore[unsupported-operation]
                        1,
                    )
                    for axis in relation.source_domain.axis_order
                ),
            )
        )
    result = CoordinateRelation(
        source_domain=relation.target_domain,
        target_domain=relation.source_domain,
        pieces=tuple(converse_pieces),
    )
    return result if result.canonical_single_valued() is not None else None


def _piecewise_separable_dense_point_converse(
    relation: CoordinateRelation,
) -> CoordinateRelation | None:
    """Invert separable mixed-radix point maps over disjoint source boxes.

    Every target coordinate may densely encode its own disjoint subset of the
    source axes.  This is the common batched schedule form: a batch target
    preserves one source digit while another target packs tile digits.
    """
    if not relation.pieces:
        return None
    converse_pieces: list[_CoordinateRelationPiece] = []
    source_boxes: list[tuple[tuple[int, int, int, int], ...]] = []
    target_boxes: list[tuple[tuple[int, int, int, int], ...]] = []
    for piece in relation.pieces:
        if any(
            not _source_bounds_are_disjoint(previous, piece.source_bounds_items)
            for previous in source_boxes
        ):
            return None
        source_boxes.append(piece.source_bounds_items)
        source_bounds = {
            axis: (begin, end, step)
            for axis, begin, end, step in piece.source_bounds_items
        }
        if any(step != 1 for _begin, _end, step in source_bounds.values()):
            return None
        inverse_by_source_axis: dict[int, sympy.Expr] = {
            axis: sympy.Integer(begin)
            for axis, (begin, end, _step) in source_bounds.items()
            if end - begin == 1
        }
        target_box: list[tuple[int, int, int, int]] = []
        for target_axis, begin, end, target_step in piece.target_ranges:
            begin = _simplify_logical_expression(
                begin,
                domain=relation.source_domain,
                source_bounds=piece.source_bounds_items,
            )
            end = _simplify_logical_expression(
                end,
                domain=relation.source_domain,
                source_bounds=piece.source_bounds_items,
            )
            if target_step != 1 or sympy.simplify(end - begin) != 1:  # pyrefly: ignore[unsupported-operation]
                return None
            layout = _static_affine_coefficients(
                begin,
                domain=relation.source_domain,
            )
            if layout is None:
                return None
            coefficients, offset = layout
            digit_layout = sorted(
                (
                    coefficients[axis],
                    axis,
                    source_bounds[axis][0],
                    source_bounds[axis][1] - source_bounds[axis][0],
                )
                for axis in relation.source_domain.axis_order
                if source_bounds[axis][1] - source_bounds[axis][0] > 1
                and coefficients[axis]
            )
            expected_stride = 1
            for coefficient, axis, _source_begin, source_count in digit_layout:
                if axis in inverse_by_source_axis or coefficient != expected_stride:
                    return None
                expected_stride *= source_count
            support_begin_expression = sympy.simplify(
                offset
                + sum(
                    coefficients[axis] * source_begin
                    for axis, (
                        source_begin,
                        _source_end,
                        _source_step,
                    ) in source_bounds.items()
                )
            )
            if (
                support_begin_expression.free_symbols
                or support_begin_expression.is_integer is not True
            ):
                return None
            support_begin = int(support_begin_expression)
            support_end = support_begin + expected_stride
            if (
                support_begin < 0
                or support_end > relation.target_domain.axis_counts[target_axis]
            ):
                return None
            target_box.append((target_axis, support_begin, support_end, 1))
            local_target = coordinate_axis_symbol(target_axis) - support_begin  # pyrefly: ignore[unsupported-operation]
            for coefficient, axis, source_begin, source_count in digit_layout:
                quotient = (
                    sympy.floor(  # pyrefly: ignore[bad-argument-type]
                        local_target / coefficient  # pyrefly: ignore[unsupported-operation]
                    )
                    if coefficient
                    else sympy.Integer(0)
                )
                inverse_by_source_axis[axis] = cast(
                    "sympy.Expr",
                    source_begin
                    + (
                        quotient
                        if coefficient * source_count == expected_stride
                        else sympy.Mod(  # pyrefly: ignore[bad-argument-type]
                            quotient,
                            source_count,
                        )
                    ),
                )
        if set(inverse_by_source_axis) != set(relation.source_domain.axis_order):
            return None
        target_box_tuple = tuple(target_box)
        if any(
            not _source_bounds_are_disjoint(previous, target_box_tuple)
            for previous in target_boxes
        ):
            return None
        target_boxes.append(target_box_tuple)
        converse_pieces.append(
            _CoordinateRelationPiece(
                source_bounds_items=target_box_tuple,
                target_ranges=tuple(
                    (
                        axis,
                        inverse_by_source_axis[axis],
                        inverse_by_source_axis[axis] + 1,  # pyrefly: ignore[unsupported-operation]
                        1,
                    )
                    for axis in relation.source_domain.axis_order
                ),
            )
        )
    result = CoordinateRelation(
        source_domain=relation.target_domain,
        target_domain=relation.source_domain,
        pieces=tuple(converse_pieces),
    )
    return result if result.canonical_single_valued() is not None else None


def _piecewise_dense_point_converse(
    relation: CoordinateRelation,
) -> CoordinateRelation | None:
    """Invert disjoint dense affine point maps one source box at a time.

    This covers compact task orders produced by ``enumerate_targets_by_source``:
    each source box linearizes a Cartesian coordinate subset into a disjoint
    interval of one nontrivial target axis.  The proof is entirely symbolic;
    unsupported layouts decline instead of enumerating domain points.
    """
    if not relation.pieces:
        return None
    nontrivial_target_axes = tuple(
        axis
        for axis in relation.target_domain.axis_order
        if relation.target_domain.axis_counts[axis] != 1
    )
    if len(nontrivial_target_axes) != 1:
        return None
    (target_axis,) = nontrivial_target_axes
    dropped_target_axes = frozenset(relation.target_domain.axis_order) - {target_axis}
    target_coordinate = coordinate_axis_symbol(target_axis)
    converse_pieces: list[_CoordinateRelationPiece] = []
    support_intervals: list[tuple[int, int]] = []
    for piece in relation.pieces:
        source_bounds = {
            axis: (begin, end, step)
            for axis, begin, end, step in piece.source_bounds_items
        }
        if any(step != 1 for _begin, _end, step in source_bounds.values()):
            return None
        target_range = next(
            (
                (begin, end, step)
                for axis, begin, end, step in piece.target_ranges
                if axis == target_axis
            ),
            None,
        )
        if target_range is None:
            return None
        begin, end, target_step = target_range
        begin = _simplify_logical_expression(
            begin,
            domain=relation.source_domain,
            source_bounds=piece.source_bounds_items,
        )
        end = _simplify_logical_expression(
            end,
            domain=relation.source_domain,
            source_bounds=piece.source_bounds_items,
        )
        if target_step != 1 or sympy.simplify(end - begin) != 1:  # pyrefly: ignore[unsupported-operation]
            return None
        if any(
            axis not in dropped_target_axes
            and axis != target_axis
            or step != 1
            or sympy.simplify(
                _simplify_logical_expression(
                    dropped_begin,
                    domain=relation.source_domain,
                    source_bounds=piece.source_bounds_items,
                )
            )
            != 0
            or sympy.simplify(
                _simplify_logical_expression(
                    dropped_end,
                    domain=relation.source_domain,
                    source_bounds=piece.source_bounds_items,
                )
                - 1  # pyrefly: ignore[unsupported-operation]
            )
            != 0
            for axis, dropped_begin, dropped_end, step in piece.target_ranges
            if axis != target_axis
        ):
            return None
        layout = _static_affine_coefficients(
            begin,
            domain=relation.source_domain,
        )
        if layout is None:
            return None
        coefficients, offset = layout
        digit_layout: list[tuple[int, int, int, int]] = []
        for axis in relation.source_domain.axis_order:
            source_begin, source_end, _source_step = source_bounds[axis]
            source_count = source_end - source_begin
            coefficient = coefficients[axis]
            if source_count > 1 and coefficient <= 0:
                return None
            digit_layout.append((coefficient, axis, source_begin, source_count))
        expected_stride = 1
        for coefficient, _axis, _source_begin, source_count in sorted(digit_layout):
            if source_count == 1:
                continue
            if coefficient != expected_stride:
                return None
            expected_stride *= source_count
        support_begin_expression = sympy.simplify(
            offset
            + sum(
                coefficients[axis] * source_begin
                for axis, (
                    source_begin,
                    _source_end,
                    _source_step,
                ) in source_bounds.items()
            )
        )
        if (
            support_begin_expression.free_symbols
            or support_begin_expression.is_integer is not True
        ):
            return None
        support_begin = int(support_begin_expression)
        support_end = support_begin + expected_stride
        if (
            support_begin < 0
            or support_end > relation.target_domain.axis_counts[target_axis]
            or any(
                max(support_begin, other_begin) < min(support_end, other_end)
                for other_begin, other_end in support_intervals
            )
        ):
            return None
        support_intervals.append((support_begin, support_end))
        local_target = target_coordinate - support_begin  # pyrefly: ignore[unsupported-operation]
        source_expressions: list[sympy.Expr] = []
        for axis in relation.source_domain.axis_order:
            source_begin, source_end, _source_step = source_bounds[axis]
            source_count = source_end - source_begin
            coefficient = coefficients[axis]
            source_expressions.append(
                sympy.Integer(source_begin)
                if source_count == 1
                else cast(
                    "sympy.Expr",
                    source_begin
                    + (
                        sympy.floor(  # pyrefly: ignore[bad-argument-type]
                            local_target / coefficient  # pyrefly: ignore[unsupported-operation]
                        )
                        if coefficient * source_count == expected_stride
                        else sympy.Mod(  # pyrefly: ignore[bad-argument-type]
                            sympy.floor(  # pyrefly: ignore[bad-argument-type]
                                local_target / coefficient  # pyrefly: ignore[unsupported-operation]
                            ),
                            source_count,
                        )
                    ),
                )
            )
        converse_pieces.append(
            _CoordinateRelationPiece(
                source_bounds_items=tuple(
                    (
                        axis,
                        support_begin if axis == target_axis else 0,
                        support_end
                        if axis == target_axis
                        else relation.target_domain.axis_counts[axis],
                        1,
                    )
                    for axis in relation.target_domain.axis_order
                ),
                target_ranges=tuple(
                    (
                        axis,
                        expression,
                        expression + 1,  # pyrefly: ignore[unsupported-operation]
                        1,
                    )
                    for axis, expression in zip(
                        relation.source_domain.axis_order,
                        source_expressions,
                        strict=True,
                    )
                ),
            )
        )
    result = CoordinateRelation(
        source_domain=relation.target_domain,
        target_domain=relation.source_domain,
        pieces=tuple(converse_pieces),
    )
    return result if result.canonical_single_valued() is not None else None


def _dense_mixed_radix_converse(
    relation: CoordinateRelation,
    target_counts: CoordinateRelation,
) -> CoordinateRelation | None:
    """Reverse a dense mixed-radix source-to-target partition exactly.

    ``relation`` maps source coordinates to a bounded union of contiguous
    ranges on one target axis. Every range must share one affine source layout.
    The ranges are considered jointly, allowing target-coordinate digits that
    do not affect source identity to contribute to the target count.

    This intentionally recognizes only a total, disjoint target partition.
    Partial or overlapping relations remain valid dependency relations, but
    their exact converse is left unavailable unless ordinary
    :meth:`CoordinateRelation.converse` already represents it.
    """
    if not relation.pieces:
        return None
    if len(relation.target_domain.axis_order) != 1:
        nontrivial_axes = tuple(
            axis
            for axis in relation.target_domain.axis_order
            if relation.target_domain.axis_counts[axis] != 1
        )
        if len(nontrivial_axes) != 1:
            return None
        dropped_axes = frozenset(relation.target_domain.axis_order) - frozenset(
            nontrivial_axes
        )
        if any(
            step != 1
            or sympy.simplify(
                _simplify_logical_expression(
                    begin,
                    domain=relation.source_domain,
                    source_bounds=piece.source_bounds_items,
                )
            )
            != 0
            or sympy.simplify(
                _simplify_logical_expression(  # pyrefly: ignore[unsupported-operation]
                    end,
                    domain=relation.source_domain,
                    source_bounds=piece.source_bounds_items,
                )
                - 1
            )
            != 0
            for piece in relation.pieces
            for axis, begin, end, step in piece.target_ranges
            if axis in dropped_axes
        ):
            return None
        reduced_target = CoordinateDomain(
            axis_order=nontrivial_axes,
            axis_counts_items=tuple(
                (axis, relation.target_domain.axis_counts[axis])
                for axis in nontrivial_axes
            ),
            block_sizes_items=tuple(
                (axis, relation.target_domain.block_sizes[axis])
                for axis in nontrivial_axes
                if axis in relation.target_domain.block_sizes
            ),
            kind=relation.target_domain.kind,
            identity=relation.target_domain.identity,
        )
        projected = relation.project_target(reduced_target)
        if projected is None:
            return None
        reduced_converse = _dense_mixed_radix_converse(projected, target_counts)
        return (
            None
            if reduced_converse is None
            else reduced_converse.lift_source(relation.target_domain)
        )
    target_axis = relation.target_domain.axis_order[0]
    full_source_bounds = tuple(
        (axis, 0, relation.source_domain.axis_counts[axis], 1)
        for axis in relation.source_domain.axis_order
    )
    layouts: list[tuple[dict[int, int], int, int]] = []
    support_begins: list[int] = []
    support_ends: list[int] = []
    for piece in relation.pieces:
        if piece.source_bounds_items != full_source_bounds:
            return None
        if len(piece.target_ranges) != 1:
            return None
        piece_target_axis, begin, end, step = piece.target_ranges[0]
        if piece_target_axis != target_axis or step != 1:
            return None
        begin = _simplify_logical_expression(
            begin,
            domain=relation.source_domain,
            source_bounds=piece.source_bounds_items,
        )
        end = _simplify_logical_expression(
            end,
            domain=relation.source_domain,
            source_bounds=piece.source_bounds_items,
        )
        layout = _static_affine_coefficients(
            begin,
            domain=relation.source_domain,
        )
        width = sympy.simplify(end - begin)  # pyrefly: ignore[unsupported-operation]
        if (
            layout is None
            or width.free_symbols
            or width.is_integer is not True
            or int(width) <= 0
        ):
            return None
        expression_bounds = _logical_expression_bounds(
            begin,
            domain=relation.source_domain,
            source_bounds=piece.source_bounds_items,
        )
        end_bounds = _logical_expression_bounds(
            end,
            domain=relation.source_domain,
            source_bounds=piece.source_bounds_items,
        )
        if (
            expression_bounds is None
            or end_bounds is None
            or expression_bounds[0].free_symbols
            or end_bounds[1].free_symbols
            or expression_bounds[0].is_integer is not True  # pyrefly: ignore[missing-attribute]
            or end_bounds[1].is_integer is not True  # pyrefly: ignore[missing-attribute]
            or expression_bounds[0] < 0  # pyrefly: ignore[unsupported-operation]
            or end_bounds[1] > relation.target_domain.axis_counts[target_axis]  # pyrefly: ignore[unsupported-operation]
        ):
            return None
        support_begins.append(int(expression_bounds[0]))
        support_ends.append(int(end_bounds[1]))
        coefficients, offset = layout
        layouts.append((coefficients, offset, int(width)))

    coefficients = layouts[0][0]
    if any(piece_coefficients != coefficients for piece_coefficients, _, _ in layouts):
        return None
    targets_per_source = target_counts.constant_value()
    if targets_per_source is None or targets_per_source <= 0:
        return None
    support_begin = min(support_begins)
    support_end = max(support_ends)
    if targets_per_source * relation.source_domain.size != support_end - support_begin:
        return None

    target_coordinate = coordinate_axis_symbol(target_axis)
    source_expressions: list[sympy.Expr] = []
    for source_axis in relation.source_domain.axis_order:
        count = relation.source_domain.axis_counts[source_axis]
        if count == 1:
            source_expressions.append(sympy.Integer(0))
            continue
        stride = coefficients[source_axis]
        if stride <= 0:
            return None
        period = stride * count
        for piece_coefficients, offset, width in layouts:
            residual_min = (offset - support_begin) % period
            residual_max = residual_min + width - 1
            for other_axis in relation.source_domain.axis_order:
                if other_axis == source_axis:
                    continue
                other_stride = piece_coefficients[other_axis]
                if other_stride % period == 0:
                    continue
                other_count = relation.source_domain.axis_counts[other_axis]
                residual_max += other_stride * (other_count - 1)
            minimum_digit = residual_min // stride
            maximum_digit = residual_max // stride
            if minimum_digit != maximum_digit or minimum_digit % count != 0:
                return None
        source_expressions.append(
            sympy.Mod(  # pyrefly: ignore[bad-argument-type]
                sympy.floor(target_coordinate / stride),  # pyrefly: ignore[bad-argument-type, unsupported-operation]
                count,
            )
        )

    if support_begin:
        source_expressions = [
            expression.xreplace(
                {
                    target_coordinate: target_coordinate - support_begin  # pyrefly: ignore[unsupported-operation]
                }
            )
            for expression in source_expressions
        ]

    target_bounds = (
        (
            target_axis,
            support_begin,
            support_end,
            1,
        ),
    )
    result = CoordinateRelation.point_map(
        relation.target_domain,
        relation.source_domain,
        ((target_bounds, tuple(source_expressions)),),
    )
    return result if result.canonical_single_valued() is not None else None


def _single_axis_floor_point(
    begin: sympy.Expr,
    end: sympy.Expr,
    *,
    domain: CoordinateDomain,
) -> tuple[int, int, int, int, int] | None:
    """Recognize ``floor((a * axis + b) / d) + c`` point mappings."""
    if end != begin + 1 and sympy.simplify(end - begin) != 1:  # pyrefly: ignore[unsupported-operation]
        return None
    source_symbols: dict[sympy.Basic, int] = {
        coordinate_axis_symbol(axis): axis for axis in domain.axis_order
    }
    used_symbols = begin.free_symbols
    if len(used_symbols) != 1:
        return None
    (symbol,) = used_symbols
    axis = source_symbols.get(symbol)
    if axis is None:
        return None

    floor_terms = tuple(
        term
        for term in sympy.Add.make_args(begin)
        if _static_integer_quotient(cast("sympy.Expr", term)) is not None
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
    quotient = _static_integer_quotient(cast("sympy.Expr", floor_term))
    if quotient is None:
        return None
    numerator, divisor = quotient
    numerator = sympy.expand(numerator)
    numerator_stride_expression = numerator.coeff(symbol)
    numerator_offset_expression = sympy.simplify(
        numerator - numerator_stride_expression * symbol
    )
    if (
        numerator_stride_expression.free_symbols
        or numerator_offset_expression.free_symbols
        or numerator_stride_expression.is_integer is not True
        or numerator_offset_expression.is_integer is not True
    ):
        return None
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
    domain: CoordinateDomain,
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
    source_domain: CoordinateDomain,
    source_bounds: tuple[tuple[int, IntegerExpression, IntegerExpression, int], ...],
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
    first: CoordinateRelation,
    following: CoordinateRelation,
) -> CoordinateRelation | None:
    """Compose point-valued relation pieces by exact box preimage.

    Composition is distributive over the union of relation pieces, so it does
    not require globally canonicalizing either relation.  Avoiding that
    partitioning is important for compact PID task orders, whose pieces
    are already disjoint by construction but can number in the thousands.
    """
    if (
        len(first.pieces) > _MAX_RELATION_PIECES
        or len(following.pieces) > _MAX_RELATION_PIECES
        or not _relation_product_is_within_budget(
            len(first.pieces), len(following.pieces)
        )
        or any(
            step != 1
            or (
                end != begin + 1  # pyrefly: ignore[unsupported-operation]
                and sympy.simplify(end - begin)  # pyrefly: ignore[unsupported-operation]
                != 1
            )
            for piece in first.pieces
            for _axis, begin, end, step in piece.target_ranges
        )
    ):
        return None
    pieces: dict[_CoordinateRelationPiece, None] = {}
    unclipped_point_support: bool | None = None
    for first_piece in first.pieces:
        first_targets = {
            axis: begin for axis, begin, _end, _step in first_piece.target_ranges
        }
        substitutions: dict[sympy.Basic, sympy.Expr] = {
            coordinate_axis_symbol(axis): expression
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
                if (
                    begin == 0
                    and sympy.simplify(
                        end - following.source_domain.axis_count_expressions[axis]  # pyrefly: ignore[unsupported-operation]
                    )
                    == 0
                ):
                    target_begin = first_targets[axis]
                    target_bounds = _logical_expression_bounds(
                        target_begin,
                        domain=first.source_domain,
                        source_bounds=first_piece.source_bounds_items,
                    )
                    if (
                        target_bounds is not None
                        and _is_provably_nonnegative(target_bounds[0], None)
                        and _is_provably_nonnegative(
                            sympy.simplify(
                                first.target_domain.axis_count_expressions[axis]
                                - 1
                                - target_bounds[1]
                            ),
                            None,
                        )
                    ):
                        continue
                    if unclipped_point_support is None:
                        unclipped_point_support = (
                            _has_unclipped_point_source_support(first)
                        )
                    if not unclipped_point_support:
                        return None
                    continue
                expression_bounds = _logical_expression_bounds(
                    first_targets[axis],
                    domain=first.source_domain,
                    source_bounds=first_piece.source_bounds_items,
                )
                if expression_bounds is not None:
                    minimum, maximum = expression_bounds
                    if _is_provably_nonnegative(
                        sympy.simplify(minimum - begin),
                        None,
                    ) and _is_provably_nonnegative(
                        sympy.simplify(end - 1 - maximum),
                        None,
                    ):
                        continue
                    if _is_provably_nonnegative(
                        sympy.simplify(begin - 1 - maximum),
                        None,
                    ) or _is_provably_nonnegative(
                        sympy.simplify(minimum - end),
                        None,
                    ):
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

            composed = _CoordinateRelationPiece(
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
            pieces.setdefault(composed, None)
            if len(pieces) > _MAX_RELATION_PIECES:
                return None
    return CoordinateRelation(
        source_domain=first.source_domain,
        target_domain=following.target_domain,
        pieces=tuple(pieces),
    )


def pid_task_order(
    logical_domain: CoordinateDomain,
    pid_axis_order: tuple[int, ...],
    *,
    l2_group_size: int | None = None,
) -> CoordinateRelation:
    """Map configured PID task-order coordinates to logical tasks."""
    if set(logical_domain.axis_order) != set(pid_axis_order):
        raise ValueError("PID axis order must permute the logical task axes")
    counts = logical_domain.axis_count_expressions
    if l2_group_size is None or len(pid_axis_order) < 2:
        source_domain = CoordinateDomain(
            axis_order=pid_axis_order,
            axis_counts_items=tuple((axis, counts[axis]) for axis in pid_axis_order),
            kind="task_order",
            identity=logical_domain.identity,
        )
        return CoordinateRelation.point_map(
            source_domain,
            logical_domain,
            (
                (
                    tuple((axis, 0, counts[axis], 1) for axis in pid_axis_order),
                    tuple(
                        coordinate_axis_symbol(axis)
                        for axis in logical_domain.axis_order
                    ),
                ),
            ),
        )

    first_axis, second_axis, *outer_axes = pid_axis_order
    first_count = counts[first_axis]
    second_count = counts[second_axis]
    if l2_group_size <= 0:
        raise ValueError("L2 group size must be positive")
    concrete_first_count = _concrete_integer(
        first_count,
        description="L2 first-axis count",
    )
    concrete_second_count = _concrete_integer(
        second_count,
        description="L2 second-axis count",
    )
    group_count = (concrete_first_count + l2_group_size - 1) // l2_group_size
    piece_count = group_count * concrete_second_count
    if piece_count > _MAX_RELATION_PIECES or not (
        _relation_product_is_within_budget(group_count, concrete_second_count)
    ):
        raise ValueError("L2 task order exceeds the symbolic relation budget")
    inner_axis = min(logical_domain.axis_order, default=0) - 1
    while inner_axis in logical_domain.axis_order:
        inner_axis -= 1
    source_domain = CoordinateDomain(
        axis_order=(inner_axis, *outer_axes),
        axis_counts_items=(
            (inner_axis, first_count * second_count),
            *((axis, counts[axis]) for axis in outer_axes),
        ),
        kind="task_order",
        identity=logical_domain.identity,
    )
    inner = coordinate_axis_symbol(inner_axis)
    pieces: list[
        tuple[
            tuple[tuple[int, int, int, int], ...],
            tuple[sympy.Expr, ...],
        ]
    ] = []
    for first_in_group in range(0, concrete_first_count, l2_group_size):
        actual_group_size = min(
            concrete_first_count - first_in_group,
            l2_group_size,
        )
        group = first_in_group // l2_group_size
        group_begin = group * l2_group_size * concrete_second_count
        for second in range(concrete_second_count):
            begin = group_begin + second * actual_group_size
            expressions = {
                first_axis: inner - begin + first_in_group,  # pyrefly: ignore[unsupported-operation]
                second_axis: sympy.Integer(second),
                **{axis: coordinate_axis_symbol(axis) for axis in outer_axes},
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
    return CoordinateRelation.point_map(source_domain, logical_domain, tuple(pieces))


@dataclasses.dataclass(frozen=True)
class ExecutionSite:
    """One reachable DeviceIR callsite in an outer task's program order.

    ``graph_id`` identifies the called body, while ``callsite_path`` identifies
    this particular invocation of that body. Nested loop iterations inherit the
    worker assigned to their owning root task; this record describes their
    logical coordinate domain and program-order identity, not an independently
    movable scheduling unit.
    """

    site_id: int
    root: int
    graph_id: int
    callsite_path: tuple[tuple[int, int], ...]
    parent_site_id: int | None
    kind: Literal["root", "loop", "branch", "while_condition", "while_body"]
    local_axis_order: tuple[int, ...]
    logical_axis_order: tuple[int, ...]
    executes_unconditionally: bool
    can_split_loop: bool

    @property
    def is_root(self) -> bool:
        return self.kind == "root"


def build_execution_sites(device_ir: DeviceIR) -> tuple[ExecutionSite, ...]:
    """Build the reachable DeviceIR callsite tree used by dependency analysis.

    A DeviceIR graph body is not itself a unique execution point: one body may
    be referenced by several callsites, and control-flow graphs have different
    execution guarantees from ordinary device loops.  Paths therefore use the
    lexical call node and child argument slot within each owning root.
    """
    from ..language import _tracing_ops
    from .device_ir import ForLoopGraphInfo

    sites: list[ExecutionSite] = []

    def add_site(
        *,
        root: int,
        graph_id: int,
        callsite_path: tuple[tuple[int, int], ...],
        parent_site_id: int | None,
        kind: Literal["root", "loop", "branch", "while_condition", "while_body"],
        local_axis_order: tuple[int, ...],
        logical_axis_order: tuple[int, ...],
        executes_unconditionally: bool,
        can_split_loop: bool,
    ) -> int:
        site_id = len(sites)
        sites.append(
            ExecutionSite(
                site_id=site_id,
                root=root,
                graph_id=graph_id,
                callsite_path=callsite_path,
                parent_site_id=parent_site_id,
                kind=kind,
                local_axis_order=local_axis_order,
                logical_axis_order=logical_axis_order,
                executes_unconditionally=executes_unconditionally,
                can_split_loop=can_split_loop,
            )
        )
        return site_id

    def walk(
        *,
        root: int,
        site_id: int,
        ancestor_graph_ids: frozenset[int],
    ) -> None:
        site = sites[site_id]
        graph = device_ir.graphs[site.graph_id].graph
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
                child_specs.append(
                    (0, node.args[0], "loop", site.executes_unconditionally)
                )
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

            callsite_site_ids: list[tuple[int, int]] = []
            for (
                child_slot,
                child_graph_id,
                kind,
                executes_unconditionally,
            ) in child_specs:
                if not 0 <= child_graph_id < len(device_ir.graphs):
                    continue
                child_info = device_ir.graphs[child_graph_id]
                local_axes = (
                    tuple(child_info.block_ids)
                    if kind == "loop" and isinstance(child_info, ForLoopGraphInfo)
                    else ()
                )
                axes_are_unique = not set(local_axes).intersection(
                    site.logical_axis_order
                )
                child_site_id = add_site(
                    root=root,
                    graph_id=child_graph_id,
                    callsite_path=(*site.callsite_path, (node_index, child_slot)),
                    parent_site_id=site_id,
                    kind=kind,
                    local_axis_order=local_axes,
                    logical_axis_order=(*site.logical_axis_order, *local_axes),
                    executes_unconditionally=executes_unconditionally,
                    can_split_loop=(
                        kind == "loop"
                        and executes_unconditionally
                        and axes_are_unique
                        and not any(
                            axis in device_ir.noncanonical_task_origin_block_ids
                            for axis in local_axes
                        )
                    ),
                )
                callsite_site_ids.append((child_slot, child_site_id))
                if child_graph_id not in ancestor_graph_ids:
                    walk(
                        root=root,
                        site_id=child_site_id,
                        ancestor_graph_ids=ancestor_graph_ids
                        | frozenset((child_graph_id,)),
                    )
            if callsite_site_ids:
                node.meta[TILE_DEPENDENCY_SITE_IDS_META] = tuple(callsite_site_ids)

    for root, graph_id in enumerate(device_ir.root_ids):
        family = device_ir.task_families[root]
        root_site_id = add_site(
            root=root,
            graph_id=graph_id,
            callsite_path=(),
            parent_site_id=None,
            kind="root",
            local_axis_order=family.logical_axis_order,
            logical_axis_order=family.logical_axis_order,
            executes_unconditionally=True,
            can_split_loop=False,
        )
        walk(
            root=root,
            site_id=root_site_id,
            ancestor_graph_ids=frozenset((graph_id,)),
        )
    return tuple(sites)


@dataclasses.dataclass(frozen=True)
class TileAccess:
    """The memory facts needed to prove a cross-root readiness relation.

    Layout values are canonical SymPy integer expressions even though callers
    may pass Python integers during migration. ``layout_is_symbolically_exact``
    records only whether those expressions use guarded host-backed parameters;
    masks and indirect subscripts are independent access-relation facts.
    """

    access_id: int
    memory_op_index: int
    graph_id: int
    root: int
    allocation_id: int
    kind: Literal["load", "store"]
    tensor_name: str | None
    tensor_shape: tuple[IntegerExpression, ...]
    tensor_strides: tuple[IntegerExpression, ...]
    storage_offset: IntegerExpression
    subscript_dims: tuple[int, ...]
    subscript_affine_block_ids: tuple[int | None, ...]
    subscript_index_scales: tuple[int, ...]
    subscript_offsets: tuple[int | None, ...]
    subscript_is_scalar: tuple[bool, ...]
    has_explicit_mask: bool
    layout_is_symbolically_exact: bool
    subscript_is_full_slice: tuple[bool, ...] = ()
    subscript_static_extents: tuple[int | None, ...] = ()
    is_atomic: bool = False
    graph_node_index: int = -1
    affine_subscript_ranges: tuple[AffineSubscriptRange, ...] | None = None

    def __post_init__(self) -> None:
        """Canonicalize layout values once at the dependency-analysis boundary."""
        object.__setattr__(
            self,
            "tensor_shape",
            tuple(
                _integer_expression(value, description="access shape")
                for value in self.tensor_shape
            ),
        )
        object.__setattr__(
            self,
            "tensor_strides",
            tuple(
                _integer_expression(value, description="access stride")
                for value in self.tensor_strides
            ),
        )
        object.__setattr__(
            self,
            "storage_offset",
            _integer_expression(
                self.storage_offset,
                description="access storage offset",
            ),
        )
        if self.affine_subscript_ranges is not None:
            object.__setattr__(
                self,
                "affine_subscript_ranges",
                tuple(
                    (
                        tuple(
                            (
                                axis,
                                _integer_expression(
                                    coefficient,
                                    description="affine subscript coefficient",
                                ),
                                divisor,
                            )
                            for axis, coefficient, divisor in coordinate_terms
                        ),
                        _integer_expression(
                            begin,
                            description="affine subscript offset",
                        ),
                        _integer_expression(
                            end,
                            description="affine subscript offset",
                        ),
                        step,
                    )
                    for coordinate_terms, begin, end, step in self.affine_subscript_ranges
                ),
            )


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
    """One symbolic dependency between execution-site instance domains.

    ``producers_by_consumer`` maps each consumer instance to the producer
    instances it must observe. A missing relation means that dependency
    scheduling must lift to an enclosing site or root barrier.
    """

    kind: TileDependencyKind
    dependency_id: int
    producer_access_id: int
    consumer_access_id: int
    producer_root: int
    consumer_root: int
    producer_site_id: int | None
    consumer_site_id: int | None
    producers_by_consumer: CoordinateRelation | None


@dataclasses.dataclass(frozen=True)
class TileDependency:
    """One allocation hazard between two source-ordered root families."""

    producer_root: int
    consumer_root: int
    allocation_id: int
    tensor_names: frozenset[str]
    access_dependencies: tuple[AccessDependency, ...]


@dataclasses.dataclass(frozen=True)
class TileDependencyGraph:
    """Allocation-derived dependencies and DeviceIR execution sites."""

    task_families: tuple[TaskFamily, ...]
    accesses: tuple[TileAccess, ...]
    edges: tuple[TileDependency, ...]
    execution_sites: tuple[ExecutionSite, ...] = ()
    site_ids_by_access: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if tuple(site.site_id for site in self.execution_sites) != tuple(
            range(len(self.execution_sites))
        ):
            raise ValueError("execution site IDs must be contiguous")
        if any(
            not 0 <= site_id < len(self.execution_sites)
            for site_ids in self.site_ids_by_access
            for site_id in site_ids
        ):
            raise ValueError("access references an unknown execution site")

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

    def sites_for_access(self, access_id: int) -> tuple[ExecutionSite, ...]:
        if not 0 <= access_id < len(self.site_ids_by_access):
            return ()
        return tuple(
            self.execution_sites[site_id]
            for site_id in self.site_ids_by_access[access_id]
        )

    def dependency_obligations(
        self,
        dependency: AccessDependency,
    ) -> frozenset[DependencyObligation]:
        """Return every producer/consumer callsite obligation for one hazard."""

        def access_site_ids(access_id: int) -> tuple[int | None, ...]:
            if not 0 <= access_id < len(self.site_ids_by_access):
                return (None,)
            site_ids = self.site_ids_by_access[access_id]
            return site_ids or (None,)

        return frozenset(
            (
                dependency.dependency_id,
                producer_site_id,
                consumer_site_id,
            )
            for producer_site_id in access_site_ids(dependency.producer_access_id)
            for consumer_site_id in access_site_ids(dependency.consumer_access_id)
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
    if not access.layout_is_symbolically_exact:
        return AllocationRegion(None, False)
    try:
        shape = tuple(
            _concrete_integer(size, description="access shape")
            for size in access.tensor_shape
        )
        strides = tuple(
            _concrete_integer(stride, description="access stride")
            for stride in access.tensor_strides
        )
        storage_offset = _concrete_integer(
            access.storage_offset,
            description="access storage offset",
        )
    except ValueError:
        # Reaching-definition analysis may conservatively retain an edge while
        # configured symbolic dependency analysis later proves its exact map.
        return AllocationRegion(None, False)
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
            and access.subscript_is_scalar[position]
            and offset is not None
            and static_extent == 1
        ):
            begin = offset if offset >= 0 else size + offset
            end = begin + 1
            if 0 <= begin < end <= size:
                bounds.append((begin, end))
                exact_dimensions.append(not access.has_explicit_mask)
                continue
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
        dataclasses.replace(
            access,
            tensor_shape=shape,
            tensor_strides=strides,
            storage_offset=storage_offset,
        ),
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
    domain: CoordinateDomain,
) -> tuple[sympy.Expr, sympy.Expr] | None:
    if position >= len(access.subscript_is_full_slice):
        return None
    tensor_dimension = access.subscript_dims[position]
    size = _integer_expression(
        access.tensor_shape[tensor_dimension],
        description="access shape",
    )
    if access.subscript_is_full_slice[position]:
        return sympy.Integer(0), size
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
        if offset < 0 and size.free_symbols:
            return None
        normalized_offset = sympy.sympify(offset if offset >= 0 else size + offset)
        if (
            normalized_offset.is_nonnegative is not True
            or (size - normalized_offset - 1).is_nonnegative is not True
        ):
            return None
        return normalized_offset, normalized_offset + 1
    if axis is None:
        static_extent = (
            access.subscript_static_extents[position]
            if position < len(access.subscript_static_extents)
            else None
        )
        if offset is None or static_extent is None:
            return None
        if offset < 0 and size.free_symbols:
            return None
        normalized_offset = sympy.sympify(offset if offset >= 0 else size + offset)
        if (
            normalized_offset.is_nonnegative is not True
            or (size - normalized_offset - static_extent).is_nonnegative is not True
        ):
            return None
        return normalized_offset, normalized_offset + static_extent
    counts = domain.axis_count_expressions
    if axis is None or offset is None or axis not in counts:
        return None
    scale = access.subscript_index_scales[position]
    if scale != 1:
        return None
    coordinate: sympy.Expr = (
        sympy.Integer(0)
        if sympy.simplify(counts[axis] - 1) == 0
        else coordinate_axis_symbol(axis)
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
    source_domain: CoordinateDomain,
    allocation_domain: CoordinateDomain,
    tensor_dimensions: tuple[int, ...],
) -> CoordinateRelation | None:
    """Map one access site to its exact allocation-coordinate footprint."""
    if (
        not access.layout_is_symbolically_exact
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
                _integer_expression(
                    access.tensor_shape[tensor_dimension],
                    description="access shape",
                ),
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
    return CoordinateRelation(
        source_domain=source_domain,
        target_domain=allocation_domain,
        pieces=(
            _CoordinateRelationPiece(
                source_bounds_items=tuple(
                    (axis, 0, source_domain.axis_count_expressions[axis], 1)
                    for axis in source_domain.axis_order
                ),
                target_ranges=tuple(target_ranges),
            ),
        ),
    )


def _allocation_storage_size(
    access: TileAccess,
    *,
    prove_nonnegative: Callable[[sympy.Expr], bool] | None = None,
) -> sympy.Expr | None:
    """Return a symbolic upper bound for addresses in one exact strided view."""
    shape = tuple(
        _integer_expression(size, description="access shape")
        for size in access.tensor_shape
    )
    strides = tuple(
        _integer_expression(stride, description="access stride")
        for stride in access.tensor_strides
    )
    storage_offset = _integer_expression(
        access.storage_offset,
        description="access storage offset",
    )
    if (
        len(shape) != len(strides)
        or not _is_provably_nonnegative(storage_offset, prove_nonnegative)
        or any(not _is_provably_nonnegative(size, prove_nonnegative) for size in shape)
        or any(
            not _is_provably_nonnegative(stride, prove_nonnegative)
            for stride in strides
        )
    ):
        return None
    address_span = sympy.simplify(
        storage_offset
        + 1
        + sum(
            (size - 1) * stride
            for size, stride in zip(
                shape,
                strides,
                strict=True,
            )
        )
    )
    if address_span.is_zero is True:
        # A concrete empty view has no accessed address. A one-element domain
        # is a harmless carrier because its source relation is empty.
        return sympy.Integer(1)
    if _is_provably_nonnegative(address_span, prove_nonnegative):
        return address_span
    return sympy.simplify(sympy.Max(1, address_span))


def _normalized_coordinate_layout(
    access: TileAccess,
) -> (
    tuple[
        tuple[int, ...],
        tuple[tuple[sympy.Expr, sympy.Expr], ...],
    ]
    | None
):
    """Return non-size-one dimensions and their allocation geometry."""
    shape = tuple(
        _integer_expression(size, description="access shape")
        for size in access.tensor_shape
    )
    strides = tuple(
        _integer_expression(stride, description="access stride")
        for stride in access.tensor_strides
    )
    storage_offset = _integer_expression(
        access.storage_offset,
        description="access storage offset",
    )
    layout = (shape, strides, storage_offset)
    if not _layout_is_injective(layout):
        return None
    dimensions = tuple(
        dimension
        for dimension, size in enumerate(shape)
        if sympy.simplify(size - 1) != 0
    )
    return dimensions, tuple(
        (shape[dimension], strides[dimension]) for dimension in dimensions
    )


def _symbolic_linear_access_relation(
    access: TileAccess,
    *,
    source_domain: CoordinateDomain,
    allocation_domain: CoordinateDomain,
    prove_nonnegative: Callable[[sympy.Expr], bool] | None = None,
) -> CoordinateRelation | None:
    """Map a provably contiguous view tile to linear allocation addresses."""
    if (
        not access.layout_is_symbolically_exact
        or access.has_explicit_mask
        or allocation_domain.kind != "allocation"
        or allocation_domain.identity != access.allocation_id
        or allocation_domain.axis_order != (_ALLOCATION_ADDRESS_AXIS,)
    ):
        return None
    positions = _access_positions_by_dimension(access)
    if positions is None:
        return None

    if (
        len(access.tensor_shape) == 1
        and access.subscript_dims == (0,)
        and access.affine_subscript_ranges is not None
    ):
        (stride,) = access.tensor_strides
        if not isinstance(stride, sympy.Integer) or int(stride) <= 0:
            return None
        stride_value = int(stride)
        source_counts = source_domain.axis_count_expressions
        source_bounds = tuple(
            (axis, 0, source_counts[axis], 1) for axis in source_domain.axis_order
        )
        pieces: list[_CoordinateRelationPiece] = []
        for (
            coordinate_terms,
            offset_begin,
            offset_end,
            offset_step,
        ) in access.affine_subscript_ranges:
            offset_begin_expression = _integer_expression(
                offset_begin,
                description="affine subscript offset",
            )
            offset_end_expression = _integer_expression(
                offset_end,
                description="affine subscript offset",
            )
            offset_width = sympy.simplify(
                offset_end_expression - offset_begin_expression
            )
            if (
                offset_step <= 0
                or not isinstance(offset_width, sympy.Integer)
                or int(offset_width) <= 0
                or int(offset_width) % offset_step != 0
                or len(
                    {
                        (axis, divisor)
                        for axis, _coefficient, divisor in coordinate_terms
                    }
                )
                != len(coordinate_terms)
                or any(
                    axis not in source_counts
                    or not _is_provably_nonnegative(
                        _integer_expression(
                            coefficient,
                            description="affine subscript coefficient",
                        ),
                        prove_nonnegative,
                    )
                    or divisor <= 0
                    for axis, coefficient, divisor in coordinate_terms
                )
            ):
                return None
            index_begin = offset_begin_expression
            for axis, coefficient, divisor in coordinate_terms:
                coordinate = coordinate_axis_symbol(axis)
                quotient = (
                    coordinate if divisor == 1 else sympy.floor(coordinate / divisor)  # pyrefly: ignore[bad-argument-type, unsupported-operation]
                )
                index_begin += coefficient * quotient  # pyrefly: ignore[unsupported-operation]
            first_index_bounds = _logical_expression_bounds(
                index_begin,
                domain=source_domain,
                source_bounds=source_bounds,
            )
            last_index_bounds = _logical_expression_bounds(
                index_begin + offset_width - offset_step,
                domain=source_domain,
                source_bounds=source_bounds,
            )
            if first_index_bounds is None or last_index_bounds is None:
                return None
            minimum_index = first_index_bounds[0]
            maximum_index = last_index_bounds[1]
            remaining = sympy.simplify(  # pyrefly: ignore[unsupported-operation]
                access.tensor_shape[0] - 1 - maximum_index
            )
            if not _is_provably_nonnegative(
                minimum_index, prove_nonnegative
            ) or not _is_provably_nonnegative(remaining, prove_nonnegative):
                return None
            address_begin = access.storage_offset + index_begin * stride_value  # pyrefly: ignore[unsupported-operation]
            pieces.append(
                _CoordinateRelationPiece(
                    source_bounds_items=source_bounds,
                    target_ranges=(
                        (
                            _ALLOCATION_ADDRESS_AXIS,
                            address_begin,
                            address_begin + offset_width * stride_value,
                            offset_step * stride_value,
                        ),
                    ),
                )
            )
        return CoordinateRelation(
            source_domain=source_domain,
            target_domain=allocation_domain,
            pieces=tuple(pieces),
        )

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
                    (source_domain.axis_count_expressions[axis] - 1)
                    * (1 if access.subscript_is_scalar[position] else width)
                    + offset
                    + width
                )
                remaining = sympy.simplify(size - final_end)
                if offset < 0 or not _is_provably_nonnegative(
                    remaining, prove_nonnegative
                ):
                    return None
        intervals.append(interval)
        widths.append(width)

    contiguous_span = 1
    remaining_dimensions = {
        tensor_dimension
        for tensor_dimension in range(len(access.tensor_shape))
        if widths[tensor_dimension] != 1
    }
    while remaining_dimensions:
        candidates = tuple(
            tensor_dimension
            for tensor_dimension in remaining_dimensions
            if sympy.simplify(access.tensor_strides[tensor_dimension] - contiguous_span)
            == 0
        )
        if len(candidates) != 1:
            return None
        (tensor_dimension,) = candidates
        width = widths[tensor_dimension]
        contiguous_span *= width
        remaining_dimensions.remove(tensor_dimension)

    begin = _integer_expression(
        access.storage_offset,
        description="access storage offset",
    )
    for (dimension_begin, _dimension_end), stride in zip(
        intervals,
        access.tensor_strides,
        strict=True,
    ):
        begin += dimension_begin * stride  # pyrefly: ignore[unsupported-operation]
    return CoordinateRelation(
        source_domain=source_domain,
        target_domain=allocation_domain,
        pieces=(
            _CoordinateRelationPiece(
                source_bounds_items=tuple(
                    (axis, 0, source_domain.axis_count_expressions[axis], 1)
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


def _symbolic_producers_by_consumer(
    *,
    producer_access: TileAccess,
    producer_domain: CoordinateDomain,
    consumer_access: TileAccess,
    consumer_domain: CoordinateDomain,
    prove_nonnegative: Callable[[sympy.Expr], bool] | None = None,
) -> CoordinateRelation | None:
    """Compose two site-to-allocation maps into exact producer dependencies."""
    if (
        not producer_access.layout_is_symbolically_exact
        or not consumer_access.layout_is_symbolically_exact
    ):
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
        coordinate_domain = CoordinateDomain(
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
            relation = producer_relation.overlapping_sources(
                consumer_relation,
                prove_nonnegative=prove_nonnegative,
            )
            if relation is not None:
                return relation

    producer_storage_size = _allocation_storage_size(
        producer_access,
        prove_nonnegative=prove_nonnegative,
    )
    consumer_storage_size = _allocation_storage_size(
        consumer_access,
        prove_nonnegative=prove_nonnegative,
    )
    if producer_storage_size is None or consumer_storage_size is None:
        return None
    storage_size_delta = sympy.simplify(producer_storage_size - consumer_storage_size)
    if storage_size_delta == 0 or _is_provably_nonnegative(
        storage_size_delta, prove_nonnegative
    ):
        allocation_storage_size = producer_storage_size
    elif _is_provably_nonnegative(-storage_size_delta, prove_nonnegative):
        allocation_storage_size = consumer_storage_size
    else:
        allocation_storage_size = sympy.simplify(
            sympy.Max(producer_storage_size, consumer_storage_size)
        )
    linear_domain = CoordinateDomain(
        axis_order=(_ALLOCATION_ADDRESS_AXIS,),
        axis_counts_items=(
            (
                _ALLOCATION_ADDRESS_AXIS,
                allocation_storage_size,
            ),
        ),
        kind="allocation",
        identity=producer_access.allocation_id,
    )
    producer_relation = _symbolic_linear_access_relation(
        producer_access,
        source_domain=producer_domain,
        allocation_domain=linear_domain,
        prove_nonnegative=prove_nonnegative,
    )
    consumer_relation = _symbolic_linear_access_relation(
        consumer_access,
        source_domain=consumer_domain,
        allocation_domain=linear_domain,
        prove_nonnegative=prove_nonnegative,
    )
    if producer_relation is None or consumer_relation is None:
        return None
    relation = producer_relation.overlapping_sources(
        consumer_relation,
        prove_nonnegative=prove_nonnegative,
    )
    if relation is not None:
        return relation
    return _dense_linear_overlap_relation(
        producer_relation,
        consumer_relation,
        prove_nonnegative=prove_nonnegative,
    )


def _coordinate_domain_for_axes(
    axis_order: tuple[int, ...],
    *,
    axis_geometry: dict[int, tuple[IntegerExpression, int]],
    identity: int,
) -> CoordinateDomain | None:
    geometry = tuple(axis_geometry.get(axis) for axis in axis_order)
    if any(item is None for item in geometry):
        return None
    concrete_geometry = tuple(item for item in geometry if item is not None)
    for count, block_size in concrete_geometry:
        count_expression = _integer_expression(
            count,
            description="coordinate-domain axis count",
        )
        if count_expression.is_nonnegative is not True or block_size <= 0:
            return None
        if count_expression.is_zero is True:
            return None
    return CoordinateDomain(
        axis_order=axis_order,
        axis_counts_items=tuple(
            (axis, concrete_geometry[index][0]) for index, axis in enumerate(axis_order)
        ),
        block_sizes_items=tuple(
            (axis, concrete_geometry[index][1]) for index, axis in enumerate(axis_order)
        ),
        kind="site",
        identity=identity,
    )


def instantiate_coordinate_domains(
    dependency_graph: TileDependencyGraph,
    *,
    axis_geometry: dict[int, tuple[IntegerExpression, int]],
) -> tuple[
    tuple[CoordinateDomain | None, ...],
    tuple[CoordinateDomain | None, ...],
]:
    """Bind root and execution-site domains to one selected tile geometry.

    Root sites reuse the same domain objects returned in the site-indexed
    table. No task order is attached: these are semantic coordinates, while
    PID task order and task-local program order remain scheduling choices.
    """
    site_domains = tuple(
        _coordinate_domain_for_axes(
            site.logical_axis_order,
            axis_geometry=axis_geometry,
            identity=site.site_id,
        )
        for site in dependency_graph.execution_sites
    )
    root_site_ids = {
        site.root: site.site_id
        for site in dependency_graph.execution_sites
        if site.is_root
    }
    root_domains = tuple(
        (
            site_domains[root_site_ids[root]]
            if root in root_site_ids
            else _coordinate_domain_for_axes(
                family.logical_axis_order,
                axis_geometry=axis_geometry,
                identity=root,
            )
        )
        for root, family in enumerate(dependency_graph.task_families)
    )
    if any(
        domain is not None and domain.axis_order != family.logical_axis_order
        for domain, family in zip(
            root_domains,
            dependency_graph.task_families,
            strict=True,
        )
    ):
        raise ValueError("root site axes disagree with the task-family domain")
    return root_domains, site_domains


def instantiate_symbolic_dependencies(
    dependency_graph: TileDependencyGraph,
    *,
    root_domains: tuple[CoordinateDomain | None, ...],
    site_domains: tuple[CoordinateDomain | None, ...],
    prove_nonnegative: Callable[[sympy.Expr], bool] | None = None,
) -> tuple[TileDependencyRelation, ...]:
    """Instantiate site dependencies without enumerating task instances.

    Unsupported access geometry returns ``producers_by_consumer=None`` so the
    caller can monotonically retain root barrier.
    """
    if len(root_domains) != len(dependency_graph.task_families):
        raise ValueError("root domain count disagrees with the dependency graph")
    if len(site_domains) != len(dependency_graph.execution_sites):
        raise ValueError("site domain count disagrees with the dependency graph")
    site_by_id = {site.site_id: site for site in dependency_graph.execution_sites}
    access_by_id = {access.access_id: access for access in dependency_graph.accesses}

    def endpoints(
        access: TileAccess,
    ) -> tuple[tuple[int | None, CoordinateDomain], ...]:
        site_ids = (
            dependency_graph.site_ids_by_access[access.access_id]
            if 0 <= access.access_id < len(dependency_graph.site_ids_by_access)
            else ()
        )
        if not site_ids:
            root_domain = root_domains[access.root]
            return () if root_domain is None else ((None, root_domain),)
        result: list[tuple[int | None, CoordinateDomain]] = []
        for site_id in site_ids:
            site = site_by_id[site_id]
            domain = site_domains[site_id]
            if domain is not None and site.executes_unconditionally:
                result.append((site_id, domain))
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
                        producer_site_id=None,
                        consumer_site_id=None,
                        producers_by_consumer=None,
                    )
                )
                continue
            for producer_site_id, producer_domain in producer_endpoints:
                for consumer_site_id, consumer_domain in consumer_endpoints:
                    result.append(
                        TileDependencyRelation(
                            kind=access_dependency.kind,
                            dependency_id=access_dependency.dependency_id,
                            producer_access_id=producer_access.access_id,
                            consumer_access_id=consumer_access.access_id,
                            producer_root=edge.producer_root,
                            consumer_root=edge.consumer_root,
                            producer_site_id=producer_site_id,
                            consumer_site_id=consumer_site_id,
                            producers_by_consumer=(
                                _symbolic_producers_by_consumer(
                                    producer_access=producer_access,
                                    producer_domain=producer_domain,
                                    consumer_access=consumer_access,
                                    consumer_domain=consumer_domain,
                                    prove_nonnegative=prove_nonnegative,
                                )
                                if axes_have_canonical_origins
                                else None
                            ),
                        )
                    )
    return tuple(result)


def consumer_to_preceding_site_relation(
    dependency_graph: TileDependencyGraph,
    *,
    site_domains: tuple[CoordinateDomain | None, ...],
    preceding_site_id: int,
    consumer_site_id: int,
    consumer_access_id: int,
) -> CoordinateRelation | None:
    """Map a consumer site to a preceding site in task-local program order.

    An ancestor maps to its single enclosing instance.  A lexically earlier
    sibling subtree maps to every preceding-site instance under the shared
    enclosing instance. Both are ordinary relations; no flattened iteration IDs are
    constructed.
    """
    sites = dependency_graph.execution_sites
    if len(site_domains) != len(sites):
        raise ValueError("site domain count disagrees with the dependency graph")
    preceding_site = sites[preceding_site_id]
    consumer_site = sites[consumer_site_id]
    preceding_domain = site_domains[preceding_site_id]
    consumer_domain = site_domains[consumer_site_id]
    if (
        preceding_site.root != consumer_site.root
        or preceding_domain is None
        or consumer_domain is None
        or preceding_domain.parameter_symbols
        or consumer_domain.parameter_symbols
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

    def lineage(site_id: int) -> tuple[int, ...]:
        result: list[int] = []
        current: int | None = site_id
        while current is not None:
            result.append(current)
            current = sites[current].parent_site_id
        result.reverse()
        return tuple(result)

    preceding_lineage = lineage(preceding_site_id)
    consumer_lineage = lineage(consumer_site_id)
    common_length = 0
    for preceding_ancestor, consumer_ancestor in zip(
        preceding_lineage, consumer_lineage, strict=False
    ):
        if preceding_ancestor != consumer_ancestor:
            break
        common_length += 1
    if not common_length:
        return None

    if common_length == len(preceding_lineage):
        equal_axes = preceding_domain.axis_order
    else:
        preceding_child = sites[preceding_lineage[common_length]]
        preceding_node_index = preceding_child.callsite_path[-1][0]
        if common_length == len(consumer_lineage):
            consumer_node_index = consumer_access.graph_node_index
        else:
            consumer_child = sites[consumer_lineage[common_length]]
            consumer_node_index = consumer_child.callsite_path[-1][0]
        if preceding_node_index >= consumer_node_index:
            return None
        common_site_id = preceding_lineage[common_length - 1]
        common_domain = site_domains[common_site_id]
        if common_domain is None:
            return None
        equal_axes = common_domain.axis_order

    if any(axis not in consumer_domain.axis_counts for axis in equal_axes):
        return None
    equal_axis_set = frozenset(equal_axes)
    return CoordinateRelation(
        source_domain=consumer_domain,
        target_domain=preceding_domain,
        pieces=(
            _CoordinateRelationPiece(
                source_bounds_items=tuple(
                    (axis, 0, consumer_domain.axis_counts[axis], 1)
                    for axis in consumer_domain.axis_order
                ),
                target_ranges=tuple(
                    (
                        axis,
                        coordinate_axis_symbol(axis),
                        coordinate_axis_symbol(axis) + 1,  # pyrefly: ignore[unsupported-operation]
                        1,
                    )
                    if axis in equal_axis_set
                    else (
                        axis,
                        sympy.Integer(0),
                        sympy.Integer(preceding_domain.axis_counts[axis]),
                        1,
                    )
                    for axis in preceding_domain.axis_order
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
    layout: tuple[
        tuple[IntegerExpression, ...],
        tuple[IntegerExpression, ...],
        IntegerExpression,
    ],
) -> bool:
    """Conservatively prove that distinct coordinates have distinct addresses."""
    raw_shape, raw_strides, _storage_offset = layout
    shape = tuple(
        _integer_expression(size, description="layout shape") for size in raw_shape
    )
    try:
        strides = tuple(
            abs(_concrete_integer(stride, description="layout stride"))
            for stride in raw_strides
        )
    except ValueError:
        return False
    span: sympy.Expr = sympy.Integer(1)
    active_dimensions = sorted(
        (
            (stride, size)
            for size, stride in zip(shape, strides, strict=True)
            if sympy.simplify(size - 1) != 0
        ),
        key=operator.itemgetter(0),
    )
    for stride, size in active_dimensions:
        if not _is_provably_nonnegative(sympy.Integer(stride) - span, None):
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
    noncanonical_task_origin_block_ids: frozenset[int] | None = None,
) -> TileDependencyGraph:
    """Build the minimal source-ordered allocation hazard graph.

    This pass is deliberately independent of code generation. It identifies the
    most recent writer and intervening readers of every allocation, then proves
    task readiness for the strict affine subset. Anything else remains a
    root-barrier dependency.
    """
    if device_ir is not None:
        if task_families is None:
            task_families = tuple(device_ir.task_families)
        if noncanonical_task_origin_block_ids is None:
            noncanonical_task_origin_block_ids = frozenset(
                device_ir.noncanonical_task_origin_block_ids
            )
    if noncanonical_task_origin_block_ids is None:
        noncanonical_task_origin_block_ids = frozenset()
    if task_families is None:
        if grid_block_ids is None:
            raise TypeError(
                "device_ir, grid_block_ids, or task_families must be provided"
            )
        task_families = tuple(
            TaskFamily(
                axes=tuple(
                    TaskAxis(
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
    roots_per_phase = {phase: root_phases.count(phase) for phase in set(root_phases)}
    if root_count > 1 and any(
        0 <= access.root < root_count
        and access.allocation_id < 0
        and roots_per_phase[root_phases[access.root]] > 1
        for access in accesses
    ):
        raise exc.CrossLoopSchedulingError(
            "because a memory operation's allocation identity is unavailable"
        )
    accesses_by_root: list[list[TileAccess]] = [[] for _ in range(root_count)]
    for access in accesses:
        if 0 <= access.root < root_count and access.allocation_id >= 0:
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
        edges.append(
            TileDependency(
                producer_root=producer_root,
                consumer_root=consumer_root,
                allocation_id=allocation_id,
                tensor_names=frozenset(
                    tensor_names_by_allocation.get(allocation_id, ())
                ),
                access_dependencies=access_dependencies,
            )
        )

    execution_sites = build_execution_sites(device_ir) if device_ir is not None else ()
    site_ids_by_graph: dict[int, list[int]] = {}
    for site in execution_sites:
        site_ids_by_graph.setdefault(site.graph_id, []).append(site.site_id)
    site_ids_by_access: list[tuple[int, ...]] = [
        ()
        for _ in range(max((access.access_id for access in accesses), default=-1) + 1)
    ]
    for access in accesses:
        site_ids_by_access[access.access_id] = tuple(
            site_id
            for site_id in site_ids_by_graph.get(access.graph_id, ())
            if execution_sites[site_id].root == access.root
        )
    return TileDependencyGraph(
        task_families=task_families,
        accesses=accesses,
        edges=tuple(edges),
        execution_sites=execution_sites,
        site_ids_by_access=tuple(site_ids_by_access),
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
