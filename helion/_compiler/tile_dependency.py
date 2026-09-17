from __future__ import annotations

import dataclasses
import enum
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

    from .device_ir import DeviceIR


TILE_DEPENDENCY_SITE_IDS_META = "_tile_dependency_site_ids"
TILE_DEPENDENCY_SITE_ID_ATTR = "_tile_dependency_site_id"
_ALLOCATION_ADDRESS_AXIS = -1
_MAX_RELATION_PIECES = 4_096
_MAX_RELATION_PRODUCT_STATES = 65_536
DependencyObligation = tuple[int, int | None, int | None]
IntegerExpression = Any
RelationBounds = tuple[tuple[int, IntegerExpression, IntegerExpression, int], ...]
ConcreteRelationBounds = tuple[tuple[int, int, int, int], ...]
TargetRanges = tuple[tuple[int, sympy.Expr, sympy.Expr, int], ...]
RectangularAxis = tuple[int, int, Literal["block", "quotient"], int]
RectangularFiberSpec = tuple[tuple[RectangularAxis, ...], tuple[int, ...]]
AffineSubscriptRange = tuple[
    tuple[tuple[int, IntegerExpression, int], ...],
    IntegerExpression,
    IntegerExpression,
    int,
]
_AccessLayout = tuple[
    tuple[sympy.Expr, ...],
    tuple[sympy.Expr, ...],
    sympy.Expr,
    sympy.Expr | None,
    tuple[int, ...] | None,
    dict[int, int] | None,
]
_DenseAccessCodec = tuple[int, int, sympy.Expr, tuple[tuple[int, sympy.Expr], ...]]
_AccessMap = tuple["CoordinateRelation", _DenseAccessCodec | None]


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


def _positive_integer_shift_is_nonnegative(expression: sympy.Expr) -> bool:
    """Prove a small polynomial nonnegative over positive integer symbols."""
    positive_symbols = tuple(
        sorted(
            (s for s in expression.free_symbols if s.is_integer and s.is_positive),
            key=str,
        )
    )
    if not positive_symbols:
        return False
    positive = frozenset(positive_symbols)

    def expansion_bound(node: sympy.Basic) -> int | None:
        if not node.free_symbols:
            return 1
        if isinstance(node, sympy.Symbol):
            return 2 if node in positive else 1
        if isinstance(node, sympy.Add):
            bounds = tuple(expansion_bound(child) for child in node.args)
            return (
                None
                if None in bounds or sum(bounds) > _MAX_RELATION_PRODUCT_STATES
                else sum(bounds)
            )
        if isinstance(node, sympy.Mul):
            product = 1
            for child in node.args:
                bound = expansion_bound(child)
                if bound is None or not _relation_product_is_within_budget(
                    product, bound
                ):
                    return None
                product *= bound
            return product
        if isinstance(node, sympy.Pow):
            base, exponent = node.args
            bound = expansion_bound(base)
            if not isinstance(exponent, sympy.Integer) or exponent < 0 or bound is None:
                return None
            result = bound ** int(exponent)
            return result if result <= _MAX_RELATION_PRODUCT_STATES else None
        return None

    if expansion_bound(expression) is None:
        return False
    shifted_symbols = tuple(
        sympy.Dummy(f"{s.name}_minus_one", integer=True, nonnegative=True)
        for s in positive_symbols
    )
    shifted = sympy.expand(
        expression.xreplace(
            dict(zip(positive_symbols, (s + 1 for s in shifted_symbols), strict=True))
        )
    )
    try:
        polynomial = sympy.Poly(shifted, *sorted(shifted.free_symbols, key=str))
    except sympy.PolynomialError:
        return False
    return len(polynomial.terms()) <= _MAX_RELATION_PRODUCT_STATES and all(
        coefficient.is_nonnegative is True
        for _powers, coefficient in polynomial.terms()
    )


def _is_provably_nonnegative(
    expression: sympy.Expr,
    prove_nonnegative: Callable[[sympy.Expr], bool] | None,
) -> bool:
    """Use intrinsic SymPy facts, then an optional enclosing shape proof."""
    expression = sympy.sympify(expression)
    if expression.is_nonnegative is True:  # pyrefly: ignore[missing-attribute]
        return True
    if expression.func in (sympy.Min, SymbolicMin):
        return all(
            _is_provably_nonnegative(cast("sympy.Expr", argument), prove_nonnegative)
            for argument in expression.args
        )
    if expression.func in (sympy.Max, SymbolicMax):
        return any(
            _is_provably_nonnegative(cast("sympy.Expr", argument), prove_nonnegative)
            for argument in expression.args
        )
    quotient_difference = _static_quotient_difference(expression)
    if quotient_difference is not None:
        return True
    if isinstance(expression, sympy.Add):
        for term in expression.args:
            coefficient, primitive = term.as_coeff_Mul()
            if coefficient.is_negative is not True or primitive.func not in (
                sympy.Min,
                SymbolicMin,
            ):
                continue
            remainder = sympy.simplify(expression - term)
            if any(
                _is_provably_nonnegative(
                    sympy.simplify(remainder + coefficient * argument),
                    prove_nonnegative,
                )
                for argument in primitive.args
            ):
                return True
        for term in expression.args:
            coefficient, primitive = term.as_coeff_Mul()
            quotient = _static_integer_quotient(cast("sympy.Expr", primitive))
            remainder = sympy.simplify(expression - term)
            if (
                coefficient == 1
                and quotient is not None
                and isinstance(remainder, sympy.Integer)
                and _is_provably_nonnegative(
                    sympy.simplify(quotient[0] + remainder * quotient[1]),
                    prove_nonnegative,
                )
            ):
                # floor(n / d) + c >= 0 exactly when n + c*d >= 0 for
                # integer c and positive integer d.
                return True
    if not any(
        node.func in (sympy.Min, sympy.Max, SymbolicMin, SymbolicMax)
        for node in sympy.preorder_traversal(expression)
    ):
        quotient_simplified = _simplify_integer_quotients(expression)
        if (
            quotient_simplified != expression
            and quotient_simplified.is_nonnegative is True
        ):
            return True
    if _positive_integer_shift_is_nonnegative(expression):
        return True
    interval = _analyze_integer_expression(expression)[1]
    return (interval is not None and interval[0].is_nonnegative is True) or (
        prove_nonnegative is not None and prove_nonnegative(expression)
    )


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


def _analyze_integer_expression(
    expression: sympy.Expr,
    domain: CoordinateDomain | None = None,
    source_bounds: RelationBounds = (),
    simplify: bool = False,
) -> tuple[sympy.Expr, tuple[sympy.Expr, sympy.Expr] | None]:
    """Rewrite and bound the integer grammar used by relation expressions."""
    logical = domain is not None
    parameters = domain.parameter_symbols if domain is not None else frozenset()
    coordinates = {
        coordinate_axis_symbol(axis): (begin, end, step)
        for axis, begin, end, step in source_bounds
    }

    def quotient_bounds(value: sympy.Expr) -> tuple[sympy.Expr, sympy.Expr] | None:
        if (parsed := _static_quotient_difference(value)) is None:
            return None
        _base, divisor, offset = parsed
        result = sympy.Integer(0 if offset.is_zero is True else 1)
        exact = offset.is_zero is True or sympy.simplify(offset - divisor) == 0
        return (result, result) if exact else (sympy.Integer(0), sympy.Integer(1))

    def combine(
        values: tuple[sympy.Basic, ...], operation: Callable[..., sympy.Expr]
    ) -> tuple[sympy.Expr, sympy.Expr] | None:
        intervals = tuple(bounds(cast("sympy.Expr", value)) for value in values)
        if None in intervals:
            return None
        concrete = cast("tuple[tuple[sympy.Expr, sympy.Expr], ...]", intervals)
        return tuple(operation(*(item[side] for item in concrete)) for side in (0, 1))

    def bounds(value: sympy.Expr) -> tuple[sympy.Expr, sympy.Expr] | None:
        value = cast("sympy.Expr", sympy.sympify(value))
        symbols = value.free_symbols
        if value.is_number or (
            logical
            and symbols
            and symbols.isdisjoint(coordinates)
            and symbols <= parameters
        ):
            return value, value
        if not logical and (interval := quotient_bounds(value)) is not None:
            return interval
        if isinstance(value, sympy.Symbol):
            if (coordinate := coordinates.get(value)) is None:
                return None
            begin, end, step = coordinate
            return sympy.sympify(begin), sympy.sympify(
                begin + (end - begin - 1) // step * step
            )
        if isinstance(value, sympy.Mod) and len(value.args) == 2:
            dividend, modulus = value.args
            if not isinstance(modulus, sympy.Integer) or modulus <= 0:
                return None
            child = bounds(cast("sympy.Expr", dividend)) if logical else None
            if child is not None:
                quotients = tuple(sympy.floor(item / modulus) for item in child)
                if sympy.simplify(quotients[0] - quotients[1]) == 0:
                    return tuple(
                        sympy.simplify(item - quotients[0] * modulus) for item in child
                    )
            return sympy.Integer(0), modulus - 1
        if logical and value.func in (sympy.floor, sympy.ceiling):
            child = bounds(cast("sympy.Expr", value.args[0]))
            return None if child is None else tuple(map(value.func, child))
        quotient = _static_integer_quotient(value)
        if quotient is not None:
            if (child := bounds(quotient[0])) is None:
                return None
            return tuple(
                FloorDiv(item, quotient[1])
                if logical
                else sympy.floor(item / quotient[1])
                for item in child
            )
        if isinstance(value, sympy.Add):
            if logical:
                return combine(value.args, sympy.Add)
            if (collapsed := _exact_quotient_remainder_replacement(value)) is not None:
                return bounds(sympy.simplify(collapsed))
            terms, intervals = list(value.args), []
            while terms:
                term = terms.pop(0)
                interval = None
                for index, other in enumerate(terms):
                    interval = quotient_bounds(sympy.simplify(term + other))
                    if (
                        interval is None
                        and (reverse := quotient_bounds(sympy.simplify(-term - other)))
                        is not None
                    ):
                        interval = -reverse[1], -reverse[0]
                    if interval is not None:
                        terms.pop(index)
                        break
                interval = interval or bounds(cast("sympy.Expr", term))
                if interval is None:
                    return None
                intervals.append(interval)
            return tuple(
                sympy.Add(*(item[side] for item in intervals)) for side in (0, 1)
            )
        if isinstance(value, sympy.Mul):
            if not logical:
                result = (sympy.Integer(1), sympy.Integer(1))
                for item in value.args:
                    interval = bounds(cast("sympy.Expr", item))
                    if interval is None:
                        return None
                    products = tuple(
                        sympy.simplify(a * b) for a in result for b in interval
                    )
                    if any(not product.is_number for product in products):
                        return None
                    result = min(products), max(products)
                return result
            varying = tuple(
                item for item in value.args if item.free_symbols & coordinates.keys()
            )
            constant = sympy.prod(
                item
                for item in value.args
                if not item.free_symbols & coordinates.keys()
            )
            child = (
                bounds(cast("sympy.Expr", varying[0])) if len(varying) == 1 else None
            )
            if child is None or not (
                not constant.free_symbols - parameters
                and (constant.is_nonnegative or constant.is_nonpositive)
            ):
                return None
            result = constant * child[0], constant * child[1]
            return result if constant.is_nonnegative else result[::-1]
        if value.func in (sympy.Min, sympy.Max, SymbolicMin, SymbolicMax):
            operation = (
                value.func
                if logical
                else (lambda *items: min(items))
                if value.func in (sympy.Min, SymbolicMin)
                else (lambda *items: max(items))
            )
            return combine(value.args, operation)
        return None

    def rewrite(value: sympy.Expr) -> sympy.Expr:
        value = cast("sympy.Expr", sympy.sympify(value))
        children = tuple(rewrite(cast("sympy.Expr", child)) for child in value.args)
        rebuilt = (
            sympy.Mod(*children, evaluate=False)
            if value.func is sympy.Mod
            else value.func(*children)
            if children
            else value
        )
        interval = bounds(rebuilt)
        if interval is not None and sympy.simplify(interval[1] - interval[0]) == 0:  # pyrefly: ignore[unsupported-operation]
            return sympy.simplify(interval[0])
        if rebuilt.func is sympy.Mod and len(children) == 2:
            child = bounds(cast("sympy.Expr", children[0]))
            in_range = (
                child is not None
                and children[1].is_integer is True
                and _is_provably_nonnegative(child[0], None)
                and _is_provably_nonnegative(
                    sympy.simplify(children[1] - 1 - child[1]), None
                )
            )
            return cast("sympy.Expr", children[0]) if in_range else rebuilt
        if rebuilt.has(sympy.Mod):
            return rebuilt
        if rebuilt.func not in (sympy.Min, sympy.Max):
            return sympy.simplify(rebuilt)
        intervals = tuple(bounds(cast("sympy.Expr", child)) for child in children)
        if None in intervals:
            return rebuilt
        concrete = cast("tuple[tuple[sympy.Expr, sympy.Expr], ...]", intervals)
        side, operation = (
            (1, operator.le) if rebuilt.func == sympy.Min else (0, operator.ge)
        )
        for index, child in enumerate(children):
            if all(
                operation(concrete[index][side], other[1 - side])
                for other_index, other in enumerate(concrete)
                if other_index != index
            ):
                return cast("sympy.Expr", child)
        return rebuilt

    root = cast("sympy.Expr", sympy.sympify(expression))
    return (rewrite(root) if simplify else root), bounds(root)


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
    """One source-level axis in a root's logical task space."""

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
    """A typed Cartesian domain in canonical logical coordinates."""

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
                _integer_expression(count, description="coordinate-domain axis count"),
            )
            for axis, count in self.axis_counts_items
        )
        object.__setattr__(self, "axis_counts_items", normalized_counts)
        for _axis, expression in normalized_counts:
            if not _is_provably_nonnegative(expression, None) or (
                expression.is_zero is True and not self._allow_empty
            ):
                raise ValueError("coordinate-domain axis counts must be positive")
        if any(size <= 0 for _axis, size in self.block_sizes_items):
            raise ValueError("coordinate-domain block sizes must be positive")

    @classmethod
    def scalar(
        cls,
        size: IntegerExpression,
        *,
        axis: int = 0,
        kind: Literal[
            "site", "allocation", "event", "task_order", "worker", "value"
        ] = "value",
        identity: int | None = None,
    ) -> CoordinateDomain:
        """Return the canonical one-dimensional domain ``[0, size)``."""
        expression = _integer_expression(size, description="scalar-domain size")
        return cls(
            (axis,),
            ((axis, expression),),
            kind=kind,
            identity=identity,
            _allow_empty=expression.is_zero is True,
        )

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
        return {
            axis: _concrete_integer(count, description="coordinate-domain axis count")
            for axis, count in self.axis_counts_items
        }

    @property
    def size_expr(self) -> sympy.Expr:
        """Return the possibly parameterized number of domain points."""
        return sympy.Mul(*self.shape_expr)

    @property
    def size(self) -> int:
        return _concrete_integer(self.size_expr, description="coordinate-domain size")

    @property
    def parameter_symbols(self) -> frozenset[sympy.Symbol]:
        """Return parameters used by this domain's axis counts."""
        return frozenset(
            symbol for count in self.shape_expr for symbol in count.free_symbols
        )


def coordinate_axis_symbol(axis: int) -> sympy.Symbol:
    """Return the canonical integer symbol for one coordinate-domain axis."""
    suffix = str(axis) if axis >= 0 else f"m{-axis}"
    return sympy.Symbol(f"coordinate_axis_{suffix}", integer=True, nonnegative=True)


def _full_bounds(
    domain: CoordinateDomain,
) -> RelationBounds:
    return tuple((axis, 0, count, 1) for axis, count in domain.axis_counts_items)


def _next_axis(*axis_orders: tuple[int, ...]) -> int:
    return max(itertools.chain.from_iterable(axis_orders), default=-1) + 1


def _full_point_map(
    source: CoordinateDomain,
    target: CoordinateDomain,
    expressions: tuple[sympy.Expr, ...],
) -> CoordinateRelation:
    return CoordinateRelation.point_map(
        source, target, ((_full_bounds(source), expressions),)
    )


def _scalar_point_expression(piece: _CoordinateRelationPiece) -> sympy.Expr | None:
    if len(piece.target_ranges) != 1:
        return None
    _axis, begin, end, step = piece.target_ranges[0]
    return begin if step == 1 and sympy.simplify(end - begin) == 1 else None


def _total_relation(
    source: CoordinateDomain, target: CoordinateDomain
) -> CoordinateRelation:
    return CoordinateRelation(
        source,
        target,
        (_CoordinateRelationPiece(_full_bounds(source), _full_bounds(target)),),
    )


def _constant_value_map(
    domain: CoordinateDomain,
    value: IntegerExpression,
    *,
    other_axes: tuple[int, ...] = (),
) -> CoordinateRelation:
    axis = _next_axis(domain.axis_order, other_axes)
    value = sympy.sympify(value)
    return _full_point_map(
        domain, CoordinateDomain.scalar(value + 1, axis=axis, kind="value"), (value,)
    )


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

    source_bounds_items: RelationBounds
    target_ranges: TargetRanges

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


@dataclasses.dataclass(frozen=True)
class CoordinateRelation:
    """A bounded symbolic relation between typed Cartesian domains."""

    source_domain: CoordinateDomain
    target_domain: CoordinateDomain
    pieces: tuple[_CoordinateRelationPiece, ...]

    def __post_init__(self) -> None:
        # A relation denotes a set, so repeated pieces carry no information.
        # Canonicalize them here instead of requiring every producer and every
        # proof to account for duplicate union paths independently.
        object.__setattr__(self, "pieces", tuple(dict.fromkeys(self.pieces)))
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
    def scalar_floor_div(
        cls,
        source_domain: CoordinateDomain,
        divisor: int,
        *,
        offset: int = 0,
        target_domain: CoordinateDomain | None = None,
        target_kind: Literal[
            "site", "allocation", "event", "task_order", "worker", "value"
        ] = "value",
    ) -> CoordinateRelation:
        """Map a concrete scalar domain through ``i -> offset + i // divisor``."""
        if len(source_domain.axis_order) != 1:
            raise ValueError("scalar floor division requires a scalar source")
        if divisor <= 0:
            raise ValueError("scalar floor division divisor must be positive")
        if offset < 0:
            raise ValueError("scalar floor division offset must be nonnegative")
        size = source_domain.size
        value_count = offset + _ceil_div(size, divisor)
        if target_domain is None:
            target_domain = CoordinateDomain.scalar(
                value_count, kind=target_kind, identity=source_domain.identity
            )
        elif len(target_domain.axis_order) != 1 or target_domain.size < value_count:
            raise ValueError("scalar floor-division target domain is too small")
        (source_axis,) = source_domain.axis_order
        source = coordinate_axis_symbol(source_axis)
        return _full_point_map(
            source_domain,
            target_domain,
            (offset + FloorDiv(source, divisor),),
        )

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
        return _full_point_map(
            source_domain,
            target_domain,
            tuple(coordinate_axis_symbol(axis) for axis in target_domain.axis_order),
        )

    @classmethod
    def point_map(
        cls,
        source_domain: CoordinateDomain,
        target_domain: CoordinateDomain,
        pieces: tuple[
            tuple[RelationBounds, tuple[sympy.Expr, ...]],
            ...,
        ],
    ) -> CoordinateRelation:
        """Build a piecewise single-valued relation in domain axis order."""
        return cls(
            source_domain,
            target_domain,
            tuple(
                _CoordinateRelationPiece(
                    source_bounds,
                    tuple(
                        (axis, expression, expression + 1, 1)  # pyrefly: ignore[unsupported-operation]
                        for axis, expression in zip(
                            target_domain.axis_order, target_expressions, strict=True
                        )
                    ),
                )
                for source_bounds, target_expressions in pieces
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
        return _full_point_map(
            source_domain,
            target_domain,
            tuple(coordinate_axis_symbol(axis) for axis in target_domain.axis_order),
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
                self.target_domain.shape_expr, target_domain.shape_expr, strict=True
            )
        ):
            return None
        renamed_axes = dict(zip(old_axes, new_axes, strict=True))
        return CoordinateRelation(
            self.source_domain,
            target_domain,
            tuple(
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
                self.source_domain.shape_expr, source_domain.shape_expr, strict=True
            )
        ):
            return None
        renamed_axes = dict(zip(old_axes, new_axes, strict=True))
        substitutions = {
            coordinate_axis_symbol(axis): coordinate_axis_symbol(renamed_axes[axis])
            for axis in old_axes
        }
        return CoordinateRelation(
            source_domain,
            self.target_domain,
            tuple(
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
            self.source_domain,
            target_domain,
            tuple(
                _CoordinateRelationPiece(
                    piece.source_bounds_items,
                    tuple(
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
        if len(self.pieces) > _MAX_RELATION_PIECES or any(
            axis not in current_counts
            or sympy.simplify(current_counts[axis] - count) != 0
            for axis, count in source_domain.axis_counts_items
        ):
            return None
        retained_axes = frozenset(source_domain.axis_order)
        dropped_axes = frozenset(self.source_domain.axis_order) - retained_axes
        symbols = {
            coordinate_axis_symbol(axis): axis for axis in self.source_domain.axis_order
        }
        pieces: list[_CoordinateRelationPiece] = []
        for piece in self.pieces:
            source_bounds = {
                axis: (begin, end, step)
                for axis, begin, end, step in piece.source_bounds_items
            }
            eliminated_uses: set[int] = set()
            target_ranges: list[tuple[int, sympy.Expr, sympy.Expr, int]] = []
            for target_axis, begin, end, target_step in piece.target_ranges:
                begin = _simplify_logical_expression(
                    begin,
                    domain=self.source_domain,
                    source_bounds=piece.source_bounds_items,
                )
                end = _simplify_logical_expression(
                    end,
                    domain=self.source_domain,
                    source_bounds=piece.source_bounds_items,
                )
                free_symbols = begin.free_symbols | end.free_symbols
                if not free_symbols <= symbols.keys():
                    return None
                eliminated_axes = {
                    symbols[symbol] for symbol in free_symbols
                } & dropped_axes
                if not eliminated_axes:
                    target_ranges.append((target_axis, begin, end, target_step))
                    continue
                if len(eliminated_axes) != 1 or not eliminated_axes.isdisjoint(
                    eliminated_uses
                ):
                    return None
                (eliminated_axis,) = eliminated_axes
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
                source_count = current_counts[eliminated_axis]
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
                    target_ranges.append(
                        (
                            target_axis,
                            sympy.Integer(0),
                            _integer_expression(
                                target_count, description="coordinate-domain axis count"
                            ),
                            1,
                        )
                    )
                    continue
                try:
                    concrete_begin = _concrete_integer(
                        source_begin, description="projected source begin"
                    )
                    concrete_end = _concrete_integer(
                        source_end, description="projected source end"
                    )
                except ValueError:
                    return None
                if concrete_end <= concrete_begin:
                    return None
                final_source = concrete_begin + (
                    (concrete_end - concrete_begin - 1) // source_step * source_step
                )
                if width not in (1, stride * source_step):
                    return None
                target_ranges.append(
                    (
                        target_axis,
                        sympy.simplify(base_expression + concrete_begin * stride),
                        sympy.simplify(base_expression + final_source * stride + width),
                        1 if width == stride * source_step else stride * source_step,
                    )
                )
            pieces.append(
                _CoordinateRelationPiece(
                    tuple(
                        (axis, *source_bounds[axis])
                        for axis in source_domain.axis_order
                    ),
                    tuple(target_ranges),
                )
            )
        unique_pieces = tuple(dict.fromkeys(pieces))
        return (
            None
            if len(unique_pieces) > _MAX_RELATION_PIECES
            else CoordinateRelation(source_domain, self.target_domain, unique_pieces)
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
        pieces = []
        for piece in self.pieces:
            bounds = {
                axis: (begin, end, step)
                for axis, begin, end, step in piece.source_bounds_items
            }
            pieces.append(
                _CoordinateRelationPiece(
                    tuple(
                        (axis, *bounds[axis])
                        if axis in current_axes
                        else (axis, 0, source_counts[axis], 1)
                        for axis in source_domain.axis_order
                    ),
                    piece.target_ranges,
                )
            )
        return CoordinateRelation(
            source_domain, self.target_domain, tuple(dict.fromkeys(pieces))
        )

    def then(self, following: CoordinateRelation) -> CoordinateRelation | None:
        """Compose supported point/projection relations without enumeration."""
        if self.target_domain != following.source_domain:
            return None
        if following.is_positional_bijection():
            return self.rename_target_axes(following.target_domain)
        point_composition = _compose_point_relations(self, following)
        if point_composition is not None:
            return point_composition
        if len(following.pieces) == 1:
            (constant_piece,) = following.pieces
            intermediate_symbols = {
                coordinate_axis_symbol(axis)
                for axis in following.source_domain.axis_order
            }
            if (
                constant_piece.source_bounds_items
                == _full_bounds(following.source_domain)
                and not any(
                    (begin.free_symbols | end.free_symbols) & intermediate_symbols
                    for _axis, begin, end, _step in constant_piece.target_ranges
                )
                and self.has_total_source()
            ):
                return CoordinateRelation(
                    self.source_domain,
                    following.target_domain,
                    (
                        _CoordinateRelationPiece(
                            _full_bounds(self.source_domain),
                            constant_piece.target_ranges,
                        ),
                    ),
                )
        if len(self.pieces) != 1:
            return None
        piece = self.pieces[0]
        if piece.source_bounds_items != _full_bounds(self.source_domain):
            return None

        retained_axes: list[tuple[int, int]] = []
        source_counts = self.source_domain.axis_count_expressions
        for axis, begin, end, step in piece.target_ranges:
            if step != 1:
                return None
            count = self.target_domain.axis_count_expressions[axis]
            source_axis = next(
                (
                    source_axis
                    for source_axis, source_count in source_counts.items()
                    if source_count == count
                    and source_axis not in (source for _target, source in retained_axes)
                    and sympy.simplify(  # pyrefly: ignore[unsupported-operation]
                        begin - coordinate_axis_symbol(source_axis)
                    )
                    == 0
                    and sympy.simplify(  # pyrefly: ignore[unsupported-operation]
                        end - coordinate_axis_symbol(source_axis) - 1
                    )
                    == 0
                ),
                None,
            )
            if source_axis is not None:
                retained_axes.append((axis, source_axis))
            elif not (
                sympy.simplify(begin) == 0 and sympy.simplify(end - count) == 0  # pyrefly: ignore[unsupported-operation]
            ):
                return None

        retained_target_domain = CoordinateDomain(
            axis_order=tuple(target for target, _source in retained_axes),
            axis_counts_items=tuple(
                (target, self.target_domain.axis_count_expressions[target])
                for target, _source in retained_axes
            ),
            block_sizes_items=tuple(
                (target, self.target_domain.block_sizes[target])
                for target, _source in retained_axes
                if target in self.target_domain.block_sizes
            ),
            kind=self.target_domain.kind,
            identity=self.target_domain.identity,
        )
        retained_source_domain = CoordinateDomain(
            axis_order=tuple(source for _target, source in retained_axes),
            axis_counts_items=tuple(
                (source, source_counts[source]) for _target, source in retained_axes
            ),
            block_sizes_items=tuple(
                (source, self.source_domain.block_sizes[source])
                for _target, source in retained_axes
                if source in self.source_domain.block_sizes
            ),
            kind=self.source_domain.kind,
            identity=self.source_domain.identity,
        )
        projected = following.project_source(retained_target_domain)
        renamed = (
            None
            if projected is None
            else projected.rename_source_axes(retained_source_domain)
        )
        return None if renamed is None else renamed.lift_source(self.source_domain)

    def covers(self, required: CoordinateRelation) -> bool:
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

    def source_axes_affecting_targets(self) -> tuple[int, ...] | None:
        """Return source axes that can change the related target set."""
        symbols: dict[sympy.Basic, int] = {
            coordinate_axis_symbol(axis): axis for axis in self.source_domain.axis_order
        }
        used: set[int] = set()
        full_source_bounds = {
            axis: (0, self.source_domain.axis_count_expressions[axis], 1)
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

    @classmethod
    def union_all(
        cls,
        relations: tuple[CoordinateRelation, ...],
    ) -> CoordinateRelation | None:
        """Return one budgeted batch union without quadratic prefix proofs."""
        if not relations:
            return None
        source_domain = relations[0].source_domain
        target_domain = relations[0].target_domain
        input_piece_count = sum(len(relation.pieces) for relation in relations)
        if input_piece_count > _MAX_RELATION_PIECES or any(
            relation.source_domain != source_domain
            or relation.target_domain != target_domain
            for relation in relations
        ):
            return None
        pieces = tuple(
            dict.fromkeys(piece for relation in relations for piece in relation.pieces)
        )
        return cls(source_domain, target_domain, pieces)

    def has_disjoint_source_support(self, other: CoordinateRelation) -> bool:
        """Prove that no source coordinate participates in both relations."""
        if self.source_domain != other.source_domain:
            return False
        return all(
            _source_bounds_are_disjoint(
                left.source_bounds_items, right.source_bounds_items
            )
            for left in self.pieces
            for right in other.pieces
        )

    def source_support_is_empty(self) -> bool | None:
        """Return whether every guarded target fiber is provably empty."""
        if not self.pieces:
            return True
        for piece in self.pieces:
            if _target_ranges_are_valid(
                piece.target_ranges,
                source_domain=self.source_domain,
                source_bounds=piece.source_bounds_items,
                target_domain=self.target_domain,
                clipped=True,
            ):
                return False
        return None

    def shift_source_scalar(
        self, offset: int, *, ambient_domain: CoordinateDomain
    ) -> CoordinateRelation | None:
        """Translate scalar source guards and coordinate uses by ``offset``."""
        if (
            offset < 0
            or len(self.source_domain.axis_order) != 1
            or len(ambient_domain.axis_order) != 1
            or len(self.pieces) > _MAX_RELATION_PIECES
        ):
            return None
        source_axis, ambient_axis = (
            self.source_domain.axis_order[0],
            ambient_domain.axis_order[0],
        )
        source, ambient = (
            coordinate_axis_symbol(source_axis),
            coordinate_axis_symbol(ambient_axis),
        )
        if source_axis != ambient_axis and any(
            ambient in begin.free_symbols | end.free_symbols
            for piece in self.pieces
            for _axis, begin, end, _step in piece.target_ranges
        ):
            return None
        ambient_count = ambient_domain.axis_count_expressions[ambient_axis]
        substitutions = {source: ambient - offset}
        pieces = []
        for piece in self.pieces:
            _axis, begin, end, step = piece.source_bounds_items[0]
            shifted_begin, shifted_end = (
                sympy.simplify(begin + offset),
                sympy.simplify(end + offset),
            )
            if not _is_provably_nonnegative(
                shifted_begin, None
            ) or not _is_provably_nonnegative(ambient_count - shifted_end, None):
                return None
            pieces.append(
                _CoordinateRelationPiece(
                    ((ambient_axis, shifted_begin, shifted_end, step),),
                    tuple(
                        (
                            axis,
                            sympy.simplify(low.xreplace(substitutions)),
                            sympy.simplify(high.xreplace(substitutions)),
                            target_step,
                        )
                        for axis, low, high, target_step in piece.target_ranges
                    ),
                )
            )
        return CoordinateRelation(
            ambient_domain, self.target_domain, tuple(dict.fromkeys(pieces))
        )

    def is_total(self) -> bool:
        """Return whether one canonical piece covers the complete product."""
        if len(self.pieces) != 1:
            return False
        (piece,) = self.pieces
        return piece.source_bounds_items == _full_bounds(
            self.source_domain
        ) and piece.target_ranges == _full_bounds(self.target_domain)

    def has_total_source(self) -> bool:
        """Return whether every source coordinate has at least one target."""
        cells = _relation_source_cells(self, include_domain=True)
        if cells is None:
            return False
        return all(
            any(
                _source_box_covers(piece.source_bounds_items, bounds)
                and _target_ranges_are_valid(
                    piece.target_ranges,
                    source_domain=self.source_domain,
                    source_bounds=bounds,
                    target_domain=self.target_domain,
                    clipped=True,
                )
                for piece in self.pieces
            )
            for bounds in cells
        )

    def is_single_valued(self) -> bool:
        """Return whether every source instance maps to at most one target."""
        if not _relation_product_is_within_budget(len(self.pieces), len(self.pieces)):
            return False
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
            or (
                bounds := _analyze_integer_expression(
                    _simplify_integer_quotients(
                        _normalize_integer_rounding(end)
                        - _normalize_integer_rounding(begin)
                    )
                )[1]
            )
            is None
            or not _is_provably_nonnegative(bounds[0], None)
            or not _is_provably_nonnegative(1 - bounds[1], None)
            for target_ranges in normalized_targets
            for _axis, begin, end, step in target_ranges
        ):
            return False
        for left_index, left in enumerate(self.pieces):
            for right_index, right in enumerate(
                self.pieces[left_index + 1 :],
                start=left_index + 1,
            ):
                if _source_bounds_are_disjoint(
                    left.source_bounds_items, right.source_bounds_items
                ):
                    continue
                left_support = _in_domain_point_support(
                    left, self.source_domain, self.target_domain
                )
                right_support = _in_domain_point_support(
                    right, self.source_domain, self.target_domain
                )
                if (
                    left_support is not None
                    and right_support is not None
                    and _source_bounds_are_disjoint(left_support, right_support)
                ):
                    continue
                if normalized_targets[left_index] == normalized_targets[right_index]:
                    continue
                if any(
                    left_step != 1 or right_step != 1
                    for (*_left, left_step), (*_right, right_step) in zip(
                        left.source_bounds_items, right.source_bounds_items, strict=True
                    )
                ):
                    return False
                overlap = tuple(
                    (axis, max(left_begin, right_begin), min(left_end, right_end), 1)
                    for (axis, left_begin, left_end, _left_step), (
                        _axis,
                        right_begin,
                        right_end,
                        _right_step,
                    ) in zip(
                        left.source_bounds_items, right.source_bounds_items, strict=True
                    )
                )
                for left_range, right_range in zip(
                    normalized_targets[left_index],
                    normalized_targets[right_index],
                    strict=True,
                ):
                    width = _simplify_integer_quotients(
                        sympy.simplify(
                            _normalize_integer_rounding(left_range[2])
                            - _normalize_integer_rounding(left_range[1])
                            + _normalize_integer_rounding(right_range[2])
                            - _normalize_integer_rounding(right_range[1])
                        )
                    )
                    bounds = _logical_expression_bounds(
                        width, domain=self.source_domain, source_bounds=overlap
                    )
                    if bounds is None or not _is_provably_nonnegative(
                        1 - bounds[1], None
                    ):
                        bounds = _analyze_integer_expression(width)[1]
                    if bounds is not None and _is_provably_nonnegative(
                        1 - bounds[1], None
                    ):
                        break
                else:
                    return False
        return True

    def canonical_single_valued(self) -> CoordinateRelation | None:
        """Return this relation when every source has at most one target."""
        return self if self.is_single_valued() else None

    def is_total_function(self) -> bool:
        """Return whether every source instance maps to exactly one target."""
        if _source_boxes_partition_domain(
            tuple(piece.source_bounds_items for piece in self.pieces),
            self.source_domain,
        ) and all(
            all(
                step == 1 and sympy.simplify(end - begin) == 1
                for _axis, begin, end, step in piece.target_ranges
            )
            and _target_ranges_are_valid(
                piece.target_ranges,
                source_domain=self.source_domain,
                source_bounds=piece.source_bounds_items,
                target_domain=self.target_domain,
                clipped=False,
            )
            for piece in self.pieces
        ):
            return True
        return self.is_single_valued() and self.has_total_source()

    def is_positional_bijection(self) -> bool:
        """Return whether coordinates are renamed position-for-position."""
        if (
            len(self.source_domain.axis_order) != len(self.target_domain.axis_order)
            or self.source_domain.shape_expr != self.target_domain.shape_expr
            or len(self.pieces) != 1
        ):
            return False
        (piece,) = self.pieces
        if piece.source_bounds_items != _full_bounds(self.source_domain):
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

    def max_target_value_by_source(
        self,
        values: CoordinateRelation,
    ) -> CoordinateRelation | None:
        """Return maxima after composing to a scalar range relation."""
        if (
            self.target_domain != values.source_domain
            or len(values.target_domain.axis_order) != 1
            or not values.is_total_function()
        ):
            return None
        mapped = self if values.is_positional_bijection() else self.then(values)
        if mapped is None or len(mapped.target_domain.axis_order) != 1:
            return None
        cells = _relation_source_cells(mapped)
        if cells is None:
            return None
        value_axis = values.target_domain.axis_order[0]
        result = []
        for bounds in cells:
            maxima = []
            for piece in mapped.pieces:
                if not _source_box_covers(piece.source_bounds_items, bounds):
                    continue
                if len(piece.target_ranges) != 1:
                    return None
                _axis, _begin, end, step = piece.target_ranges[0]
                if step != 1:
                    return None
                maxima.append(sympy.simplify(end - 1))
            if not maxima:
                continue
            maximum = sympy.Max(*maxima)
            result.append(
                _CoordinateRelationPiece(
                    bounds, ((value_axis, maximum, maximum + 1, 1),)
                )
            )
        return CoordinateRelation(
            self.source_domain, values.target_domain, tuple(result)
        )

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

    def is_pointwise_strictly_less_than_where_defined(
        self, other: CoordinateRelation
    ) -> bool:
        """Prove strict scalar order over this relation's exact support."""
        left = self.canonical_single_valued()
        right = other.canonical_single_valued()
        if (
            left is None
            or right is None
            or left.source_domain != right.source_domain
            or len(left.target_domain.axis_order) != 1
            or len(right.target_domain.axis_order) != 1
            or not right.is_total_function()
        ):
            return False
        cells = _relation_source_cells(left)
        if cells is None:
            return False
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
            if len(left_pieces) != 1 or len(right_pieces) != 1:
                return False
            left_range, right_range = (
                left_pieces[0].target_ranges,
                right_pieces[0].target_ranges,
            )
            if len(left_range) != 1 or len(right_range) != 1:
                return False
            _, left_value, left_end, left_step = left_range[0]
            _, right_value, right_end, right_step = right_range[0]
            if (
                left_step != right_step
                or left_step != 1
                or sympy.simplify(left_end - left_value) != 1
                or sympy.simplify(right_end - right_value) != 1
            ):
                return False
            difference = _logical_expression_bounds(
                sympy.simplify(left_value - right_value),
                domain=left.source_domain,
                source_bounds=bounds,
            )
            if difference is None or not _is_provably_nonnegative(
                sympy.simplify(-difference[1] - 1), None
            ):
                return False
        return bool(cells)

    def prefix_before(self, threshold: int) -> int | None:
        """Return the exact nondecreasing scalar prefix below ``threshold``."""
        if not isinstance(threshold, int) or isinstance(threshold, bool):
            return None
        if (
            len(self.source_domain.axis_order) != 1
            or len(self.target_domain.axis_order) != 1
        ):
            return None
        try:
            extent = self.source_domain.size
        except ValueError:
            return None
        if extent == 0:
            return 0
        canonical = self.canonical_single_valued()
        if canonical is None:
            return None
        if not _scalar_relation_is_nondecreasing(canonical):
            return None
        (source_axis,) = canonical.source_domain.axis_order

        def is_before(index: int) -> bool | None:
            bounds = canonical.value_bounds({source_axis: index})
            if bounds is None or bounds[0] != bounds[1]:
                return None
            return bounds[0] < threshold

        first = is_before(0)
        last = is_before(extent - 1)
        if first is None or last is None:
            return None
        if not first:
            return 0
        if last:
            return extent
        lower, upper = 0, extent - 1
        while lower + 1 < upper:
            middle = (lower + upper) // 2
            before = is_before(middle)
            if before is None:
                return None
            if before:
                lower = middle
            else:
                upper = middle
        return upper


def _expression_is_nondecreasing_in(
    expression: sympy.Expr,
    source_symbol: sympy.Symbol,
) -> bool:
    """Recognize the intentionally small monotone scalar expression grammar."""
    expression = sympy.simplify(expression)
    if source_symbol not in expression.free_symbols:
        return not expression.free_symbols
    if expression.free_symbols != {source_symbol}:
        return False
    if expression == source_symbol:
        return True
    if isinstance(expression, sympy.Add) or expression.func in (
        sympy.Min,
        sympy.Max,
        SymbolicMin,
        SymbolicMax,
    ):
        return all(
            _expression_is_nondecreasing_in(term, source_symbol)
            for term in expression.args
        )
    if isinstance(expression, sympy.Mul):
        varying = tuple(
            cast("sympy.Expr", factor)
            for factor in expression.args
            if source_symbol in factor.free_symbols
        )
        constant = sympy.prod(
            factor
            for factor in expression.args
            if source_symbol not in factor.free_symbols
        )
        return (
            not constant.free_symbols
            and len(varying) == 1
            and constant.is_nonnegative is True
            and _expression_is_nondecreasing_in(varying[0], source_symbol)
        )
    if expression.func in (sympy.floor, sympy.ceiling):
        return _expression_is_nondecreasing_in(
            cast("sympy.Expr", expression.args[0]), source_symbol
        )
    if expression.func is FloorDiv:
        numerator, denominator = expression.args
        return (
            not denominator.free_symbols
            and denominator.is_integer is True
            and denominator.is_positive is True
            and _expression_is_nondecreasing_in(
                cast("sympy.Expr", numerator), source_symbol
            )
        )
    return False


def _scalar_relation_is_nondecreasing(relation: CoordinateRelation) -> bool:
    """Prove monotonicity on every piece and across all piece boundaries."""
    if (
        len(relation.source_domain.axis_order) != 1
        or len(relation.target_domain.axis_order) != 1
        or len(relation.pieces) > _MAX_RELATION_PIECES
    ):
        return False
    try:
        if not relation.is_total_function():
            return False
        extent = relation.source_domain.size
        (source_axis,) = relation.source_domain.axis_order

        def concrete(expression: IntegerExpression) -> int:
            return _concrete_integer(expression, description="scalar relation value")

        ordered = sorted(
            relation.pieces,
            key=lambda piece: concrete(piece.source_bounds_items[0][1]),
        )
    except ValueError:
        return False
    source_symbol = coordinate_axis_symbol(source_axis)
    cursor = 0
    previous_last: int | None = None
    for piece in ordered:
        if len(piece.source_bounds_items) != 1 or len(piece.target_ranges) != 1:
            return False
        piece_axis, begin_expr, end_expr, source_step = piece.source_bounds_items[0]
        _target_axis, value, value_end, target_step = piece.target_ranges[0]
        try:
            begin, end = concrete(begin_expr), concrete(end_expr)
            first = concrete(value.xreplace({source_symbol: sympy.Integer(begin)}))
            last = concrete(value.xreplace({source_symbol: sympy.Integer(end - 1)}))
        except ValueError:
            return False
        if (
            piece_axis != source_axis
            or begin != cursor
            or begin >= end
            or source_step != 1
            or target_step != 1
            or sympy.simplify(value_end - value) != 1
            or (previous_last is not None and first < previous_last)
            or (
                end - begin > 1
                and not _expression_is_nondecreasing_in(value, source_symbol)
            )
        ):
            return False
        previous_last = last
        cursor = end
    return cursor == extent


@dataclasses.dataclass(frozen=True, init=False)
class DenseTaskOrder:
    """A complete scalar task order and its construction-supplied inverse."""

    tasks_by_ordinal: CoordinateRelation
    ordinal_by_task: CoordinateRelation

    @classmethod
    def _from_constructed(
        cls, tasks_by_ordinal: CoordinateRelation, ordinal_by_task: CoordinateRelation
    ) -> DenseTaskOrder:
        ordinal_domain, task_domain = (
            tasks_by_ordinal.source_domain,
            tasks_by_ordinal.target_domain,
        )
        if (
            ordinal_domain.kind != "task_order"
            or len(ordinal_domain.axis_order) != 1
            or ordinal_by_task.source_domain != task_domain
            or ordinal_by_task.target_domain != ordinal_domain
            or ordinal_domain.size != task_domain.size
        ):
            raise ValueError("dense task order must be a complete scalar bijection")
        result = object.__new__(cls)
        object.__setattr__(result, "tasks_by_ordinal", tasks_by_ordinal)
        object.__setattr__(result, "ordinal_by_task", ordinal_by_task)
        return result

    @property
    def task_count(self) -> int:
        return self.tasks_by_ordinal.source_domain.size

    @classmethod
    def from_pid(
        cls,
        logical_domain: CoordinateDomain,
        pid_axis_order: tuple[int, ...],
        *,
        l2_group_size: int | None = None,
    ) -> DenseTaskOrder | None:
        """Construct a configured scalar PID order and inverse in one pass."""
        if set(logical_domain.axis_order) != set(pid_axis_order):
            raise ValueError("PID axis order must permute the logical task axes")
        try:
            task_count = logical_domain.size
            counts = logical_domain.axis_counts
        except ValueError:
            return None
        ordinal_axis = max(logical_domain.axis_order, default=-1) + 1
        ordinal_domain = CoordinateDomain.scalar(
            task_count,
            axis=ordinal_axis,
            kind="task_order",
            identity=logical_domain.identity,
        )
        if task_count == 0:
            forward = CoordinateRelation(ordinal_domain, logical_domain, ())
            inverse = CoordinateRelation(logical_domain, ordinal_domain, ())
            return cls._from_constructed(forward, inverse)
        ordinal = coordinate_axis_symbol(ordinal_axis)
        stride = 1
        strides = {}
        coordinates = {}
        for axis in pid_axis_order:
            strides[axis] = stride
            coordinates[axis] = sympy.Mod(FloorDiv(ordinal, stride), counts[axis])
            stride *= counts[axis]
        inverse_expression = sum(
            (coordinate_axis_symbol(axis) * strides[axis] for axis in pid_axis_order),
            sympy.Integer(0),
        )
        inverse_pieces = (
            (
                _full_bounds(logical_domain),
                (inverse_expression,),
            ),
        )

        if l2_group_size is not None and len(pid_axis_order) >= 2:
            if l2_group_size <= 0:
                raise ValueError("L2 group size must be positive")
            first_axis, second_axis, *outer_axes = pid_axis_order
            first_count, second_count = counts[first_axis], counts[second_axis]
            group = min(l2_group_size, first_count)
            span = group * second_count
            inner = sympy.Mod(ordinal, first_count * second_count)
            coordinates[first_axis] = FloorDiv(inner, span) * group + sympy.Mod(
                inner, group
            )
            coordinates[second_axis] = sympy.Mod(FloorDiv(inner, group), second_count)
            for axis in outer_axes:
                coordinates[axis] = sympy.Mod(
                    FloorDiv(ordinal, strides[axis]), counts[axis]
                )
            first, second = (
                coordinate_axis_symbol(first_axis),
                coordinate_axis_symbol(second_axis),
            )
            inner_inverse = (
                FloorDiv(first, group) * span + second * group + sympy.Mod(first, group)
            )
            outer_offset = sum(
                (coordinate_axis_symbol(axis) * strides[axis] for axis in outer_axes),
                sympy.Integer(0),
            )
            inverse_pieces = (
                (
                    _full_bounds(logical_domain),
                    (inner_inverse + outer_offset,),
                ),
            )
            tail = first_count % group
            if tail:
                full = first_count - tail
                tail_begin = full * second_count
                selector = FloorDiv(inner, tail_begin)
                tail_first = full + sympy.Mod(inner - tail_begin, tail)
                tail_second = FloorDiv(inner - tail_begin, tail)
                coordinates[first_axis] += selector * (
                    tail_first - coordinates[first_axis]
                )
                coordinates[second_axis] += selector * (
                    tail_second - coordinates[second_axis]
                )

                def bounds(begin: int, end: int) -> ConcreteRelationBounds:
                    return tuple(
                        (
                            axis,
                            begin if axis == first_axis else 0,
                            end if axis == first_axis else counts[axis],
                            1,
                        )
                        for axis in logical_domain.axis_order
                    )

                tail_inverse = tail_begin + second * tail + first - full
                inverse_pieces = (
                    (bounds(0, full), (inner_inverse + outer_offset,)),
                    (bounds(full, first_count), (tail_inverse + outer_offset,)),
                )
        forward = _full_point_map(
            ordinal_domain,
            logical_domain,
            tuple(coordinates[a] for a in logical_domain.axis_order),
        )
        inverse = CoordinateRelation.point_map(
            logical_domain, ordinal_domain, inverse_pieces
        )
        return cls._from_constructed(forward, inverse)


@dataclasses.dataclass(frozen=True, init=False)
class Incidence:
    """One set membership relation with optional constructive capabilities."""

    items_by_key: CoordinateRelation
    keys_by_item: CoordinateRelation | None = None
    count_by_key: CoordinateRelation | None = None
    grouped_items: DenseTaskOrder | None = None

    @classmethod
    def _from_constructed(
        cls,
        items_by_key: CoordinateRelation,
        *,
        keys_by_item: CoordinateRelation | None = None,
        count_by_key: CoordinateRelation | None = None,
        grouped_items: DenseTaskOrder | None = None,
    ) -> Incidence:
        key_domain, item_domain = items_by_key.source_domain, items_by_key.target_domain
        if keys_by_item is not None and (
            keys_by_item.source_domain != item_domain
            or keys_by_item.target_domain != key_domain
        ):
            raise ValueError("incidence reverse has incompatible domains")
        if count_by_key is not None and (
            keys_by_item is None
            or count_by_key.source_domain != key_domain
            or len(count_by_key.target_domain.axis_order) != 1
        ):
            raise ValueError("incidence count has incompatible domains")
        if grouped_items is not None and (
            grouped_items.tasks_by_ordinal.target_domain != item_domain
            or keys_by_item is None
            or count_by_key is None
        ):
            raise ValueError("incidence grouped order has incompatible item domain")
        result = object.__new__(cls)
        for name, value in (
            ("items_by_key", items_by_key),
            ("keys_by_item", keys_by_item),
            ("count_by_key", count_by_key),
            ("grouped_items", grouped_items),
        ):
            object.__setattr__(result, name, value)
        return result

    @classmethod
    def from_fibers(
        cls,
        items_by_key: CoordinateRelation,
        *,
        keys_by_item: CoordinateRelation | None = None,
        prove_nonnegative: Callable[[sympy.Expr], bool] | None = None,
    ) -> Incidence:
        """Construct an incidence, deriving separable reverse/count facts once."""
        items_by_key, structure = _rectangular_fiber_spec(
            items_by_key, prove_nonnegative=prove_nonnegative
        )
        count_by_key = None
        if structure is not None:
            mappings, full_item_axes = structure
            by_key = {key: rest for key, *rest in mappings}
            fan_in = sympy.prod(
                items_by_key.target_domain.axis_count_expressions[axis]
                for axis in full_item_axes
            ) * sympy.prod(
                width for _key, _item, mode, width in mappings if mode == "block"
            )

            def inverse_range(key_axis: int) -> tuple[int, sympy.Expr, sympy.Expr, int]:
                mapping = by_key.get(key_axis)
                if mapping is None:
                    return (
                        key_axis,
                        sympy.Integer(0),
                        items_by_key.source_domain.axis_count_expressions[key_axis],
                        1,
                    )
                item_axis, mode, width = mapping
                item = coordinate_axis_symbol(item_axis)
                if mode == "block":
                    value = FloorDiv(item, width)
                    return key_axis, value, value + 1, 1
                return key_axis, width * item, width * (item + 1), 1

            keys_by_item = CoordinateRelation(
                items_by_key.target_domain,
                items_by_key.source_domain,
                (
                    _CoordinateRelationPiece(
                        _full_bounds(items_by_key.target_domain),
                        tuple(
                            inverse_range(axis)
                            for axis in items_by_key.source_domain.axis_order
                        ),
                    ),
                ),
            )
            count_by_key = _constant_value_map(
                items_by_key.source_domain,
                sympy.simplify(fan_in),
                other_axes=items_by_key.target_domain.axis_order,
            )
        fiber_size, grouped_items = _key_major_order(
            items_by_key, keys_by_item, count_by_key, structure
        )
        if fiber_size is not None and keys_by_item is not None:
            if count_by_key is None:
                count_by_key = _constant_value_map(
                    items_by_key.source_domain,
                    fiber_size,
                    other_axes=items_by_key.target_domain.axis_order,
                )
            elif count_by_key.value_bounds() != (fiber_size, fiber_size):
                grouped_items = None
        return cls._from_constructed(
            items_by_key,
            keys_by_item=keys_by_item,
            count_by_key=count_by_key,
            grouped_items=grouped_items,
        )

    @classmethod
    def complete(
        cls,
        key_domain: CoordinateDomain,
        item_domain: CoordinateDomain,
        *,
        identity: bool = False,
    ) -> Incidence:
        """Construct an identity or complete Cartesian incidence directly."""
        constructor = CoordinateRelation.identity if identity else _total_relation
        items = constructor(key_domain, item_domain)
        keys = constructor(item_domain, key_domain)
        item_count = 1 if identity else item_domain.size
        certified = identity or key_domain.size == 1
        count = (
            _constant_value_map(
                key_domain, item_count, other_axes=item_domain.axis_order
            )
            if certified
            else None
        )
        return cls._from_constructed(
            items,
            keys_by_item=keys,
            count_by_key=count,
            grouped_items=(
                DenseTaskOrder.from_pid(item_domain, item_domain.axis_order)
                if certified
                else None
            ),
        )

    @classmethod
    def fixed_embedding(
        cls,
        key_domain: CoordinateDomain,
        item_domain: CoordinateDomain,
        fixed_coordinates: dict[int, int],
    ) -> Incidence | None:
        """Embed a domain by fixing every additional item coordinate."""
        key_counts, item_counts = (
            key_domain.axis_count_expressions,
            item_domain.axis_count_expressions,
        )
        if any(
            axis not in item_counts or sympy.simplify(count - item_counts[axis]) != 0
            for axis, count in key_counts.items()
        ) or set(fixed_coordinates) != set(item_domain.axis_order) - set(
            key_domain.axis_order
        ):
            return None
        if any(
            not 0 <= value < item_domain.axis_counts[axis]
            for axis, value in fixed_coordinates.items()
        ):
            return None
        item_bounds = tuple(
            (axis, 0, item_counts[axis], 1)
            if axis in key_counts
            else (axis, fixed_coordinates[axis], fixed_coordinates[axis] + 1, 1)
            for axis in item_domain.axis_order
        )
        items = _full_point_map(
            key_domain,
            item_domain,
            tuple(
                coordinate_axis_symbol(axis)
                if axis in key_counts
                else fixed_coordinates[axis]
                for axis in item_domain.axis_order
            ),
        )
        keys = CoordinateRelation.point_map(
            item_domain,
            key_domain,
            (
                (
                    item_bounds,
                    tuple(
                        coordinate_axis_symbol(axis) for axis in key_domain.axis_order
                    ),
                ),
            ),
        )
        return cls._from_constructed(items, keys_by_item=keys)

    def project_keys(self, key_domain: CoordinateDomain) -> Incidence | None:
        """Project the key side while preserving a supplied reverse."""
        items = self.items_by_key.project_source(key_domain)
        keys = (
            None
            if self.keys_by_item is None
            else self.keys_by_item.project_target(key_domain)
        )
        return (
            None
            if items is None or (self.keys_by_item is not None and keys is None)
            else Incidence._from_constructed(items, keys_by_item=keys)
        )

    def project_items(self, item_domain: CoordinateDomain) -> Incidence | None:
        """Project the item side while preserving a supplied reverse."""
        items = self.items_by_key.project_target(item_domain)
        keys = (
            None
            if self.keys_by_item is None
            else self.keys_by_item.project_source(item_domain)
        )
        return (
            None
            if items is None or (self.keys_by_item is not None and keys is None)
            else Incidence._from_constructed(items, keys_by_item=keys)
        )

    def reversed(self) -> Incidence | None:
        """Swap a construction-supplied pair without deriving an inverse."""
        return (
            None
            if self.keys_by_item is None
            else Incidence._from_constructed(
                self.keys_by_item, keys_by_item=self.items_by_key
            )
        )

    def coarsen(self, partition: KeyPartition) -> Incidence | None:
        """Compose a partition with a uniform fine incidence and its count."""
        if (
            self.items_by_key.source_domain
            != partition.coarse_key_by_fine_key.source_domain
        ):
            return None
        fine = self if self.count_by_key is not None else self.with_key_major_order()
        count_bounds = (
            None if fine.count_by_key is None else fine.count_by_key.value_bounds()
        )
        composed = partition.as_incidence().then(fine)
        if (
            composed is None
            or count_bounds is None
            or count_bounds[0] != count_bounds[1]
        ):
            return None
        assert fine.keys_by_item is not None
        partition_counts = partition.fine_key_count_by_coarse_key
        if (
            not fine.keys_by_item.is_single_valued()
            and partition_counts.value_bounds() != (1, 1)
        ):
            derived = composed.with_key_major_order()
            return derived if derived.count_by_key is not None else None
        factor = count_bounds[0]
        if factor == 1:
            counts = partition_counts
        else:
            (axis,) = partition_counts.target_domain.axis_order
            counts = CoordinateRelation.point_map(
                partition_counts.source_domain,
                CoordinateDomain.scalar(
                    factor * (partition_counts.target_domain.size - 1) + 1,
                    axis=axis,
                    kind="value",
                ),
                tuple(
                    (piece.source_bounds_items, (factor * piece.target_ranges[0][1],))
                    for piece in partition_counts.pieces
                ),
            )
        return Incidence._from_constructed(
            composed.items_by_key,
            keys_by_item=composed.keys_by_item,
            count_by_key=counts,
        )

    def rename_domains(
        self, key_domain: CoordinateDomain, item_domain: CoordinateDomain
    ) -> Incidence | None:
        """Rename both typed domains while preserving supplied capabilities."""
        items = self.items_by_key.rename_source_axes(key_domain)
        items = None if items is None else items.rename_target_axes(item_domain)
        keys = self.keys_by_item
        if keys is not None:
            keys = keys.rename_source_axes(item_domain)
            keys = None if keys is None else keys.rename_target_axes(key_domain)
        count = self.count_by_key
        if count is not None:
            count = count.rename_source_axes(key_domain)
        grouped = self.grouped_items
        if grouped is not None:
            tasks = grouped.tasks_by_ordinal.rename_target_axes(item_domain)
            ordinals = grouped.ordinal_by_task.rename_source_axes(item_domain)
            grouped = (
                None
                if tasks is None or ordinals is None
                else DenseTaskOrder._from_constructed(tasks, ordinals)
            )
        if (
            items is None
            or (self.keys_by_item is not None and keys is None)
            or (self.count_by_key is not None and count is None)
            or (self.grouped_items is not None and grouped is None)
        ):
            return None
        return Incidence._from_constructed(
            items, keys_by_item=keys, count_by_key=count, grouped_items=grouped
        )

    def reindex_items(self, order: DenseTaskOrder) -> Incidence | None:
        """Replace logical items by their certified dense ordinals."""
        keys = self.keys_by_item
        if (
            keys is None
            or self.items_by_key.target_domain != order.ordinal_by_task.source_domain
        ):
            return None
        items = self.items_by_key.then(order.ordinal_by_task)
        if items is None:
            items = _linearize_fibers(self.items_by_key, order.ordinal_by_task)
        reverse = order.tasks_by_ordinal.then(keys)
        return (
            None
            if items is None or reverse is None
            else Incidence._from_constructed(
                items,
                keys_by_item=reverse,
                count_by_key=self.count_by_key,
            )
        )

    def with_key_major_order(self) -> Incidence:
        """Attach construction-certified count and key-major item order."""
        if self.grouped_items is not None or self.keys_by_item is None:
            return self
        fiber_size, grouped = _key_major_order(
            self.items_by_key,
            self.keys_by_item,
            self.count_by_key,
            None,
        )
        if fiber_size is None and (grouped is None or self.count_by_key is None):
            return self
        count = self.count_by_key
        if count is None:
            assert fiber_size is not None
            count = _constant_value_map(
                self.items_by_key.source_domain,
                fiber_size,
                other_axes=self.items_by_key.target_domain.axis_order,
            )
        elif fiber_size is not None and count.value_bounds() != (
            fiber_size,
            fiber_size,
        ):
            grouped = None
        return Incidence._from_constructed(
            self.items_by_key,
            keys_by_item=self.keys_by_item,
            count_by_key=count,
            grouped_items=grouped,
        )

    def then(self, following: Incidence) -> Incidence | None:
        """Compose two paired incidences without rediscovering either reverse."""
        if self.keys_by_item is None or following.keys_by_item is None:
            return None
        items = self.items_by_key.then(following.items_by_key)
        if items is None:
            items = _linearize_fibers(self.items_by_key, following.items_by_key)
        keys = following.keys_by_item.then(self.keys_by_item)
        if keys is None:
            projection = CoordinateRelation.projection(
                self.keys_by_item.source_domain,
                self.keys_by_item.target_domain,
            )
            if projection == self.keys_by_item:
                keys = following.keys_by_item.project_target(
                    self.keys_by_item.target_domain
                )
        if items is not None and keys is None:
            derived = Incidence.from_fibers(items)
            keys = derived.keys_by_item
        if items is None and keys is not None:
            items = (
                CoordinateRelation.identity(
                    keys.target_domain, keys.target_domain
                ).rename_target_axes(keys.source_domain)
                if keys.is_positional_bijection()
                else _transpose_projection_or_singletons(keys)
            )
        if items is None or keys is None:
            return None
        return Incidence.from_fibers(items, keys_by_item=keys)

    def shift_items_scalar(
        self, offset: int, *, ambient_domain: CoordinateDomain
    ) -> Incidence | None:
        """Translate scalar item ordinals while retaining both orientations."""
        item_domain = self.items_by_key.target_domain
        if (
            self.keys_by_item is None
            or len(item_domain.axis_order) != 1
            or len(ambient_domain.axis_order) != 1
        ):
            return None
        ambient_axis = ambient_domain.axis_order[0]
        items = CoordinateRelation(
            self.items_by_key.source_domain,
            ambient_domain,
            tuple(
                _CoordinateRelationPiece(
                    piece.source_bounds_items,
                    tuple(
                        (ambient_axis, begin + offset, end + offset, step)
                        for _axis, begin, end, step in piece.target_ranges
                    ),
                )
                for piece in self.items_by_key.pieces
            ),
        )
        keys = self.keys_by_item.shift_source_scalar(
            offset, ambient_domain=ambient_domain
        )
        if keys is None:
            return None
        return Incidence._from_constructed(
            items, keys_by_item=keys, count_by_key=self.count_by_key
        )

    @classmethod
    def union_all(cls, incidences: tuple[Incidence, ...]) -> Incidence | None:
        """Union compatible paired incidences without dropping the reverse."""
        if not incidences:
            return None
        if len(incidences) == 1:
            return incidences[0]
        items = CoordinateRelation.union_all(tuple(i.items_by_key for i in incidences))
        supplied_keys = tuple(
            incidence.keys_by_item
            for incidence in incidences
            if incidence.keys_by_item is not None
        )
        keys = (
            CoordinateRelation.union_all(
                cast("tuple[CoordinateRelation, ...]", supplied_keys)
            )
            if len(supplied_keys) == len(incidences)
            else None
        )
        return None if items is None else cls.from_fibers(items, keys_by_item=keys)

    def last_item_by_residue(self, modulus: int) -> Incidence | None:
        """Map each occupied residue class to its greatest item and unique key."""
        keys = self.keys_by_item
        if keys is None or len(keys.source_domain.axis_order) != 1 or modulus <= 0:
            return None
        item_axis = keys.source_domain.axis_order[0]
        try:
            item_count = keys.source_domain.size
        except ValueError:
            return None
        worker_count = min(modulus, item_count)
        if worker_count > _MAX_RELATION_PIECES:
            return None
        result = []
        for worker in range(worker_count):
            candidates = []
            for key_piece in keys.pieces:
                support = _in_domain_point_support(
                    key_piece, keys.source_domain, keys.target_domain
                )
                if support is None:
                    return None
                _axis, key_begin, key_end, key_step = support[0]
                try:
                    begin, end = int(key_begin), int(key_end)
                except TypeError:
                    return None
                gcd = math.gcd(modulus, key_step)
                difference = begin - worker
                if difference % gcd:
                    continue
                reduced_modulus = key_step // gcd
                multiplier = (
                    0
                    if reduced_modulus == 1
                    else difference
                    // gcd
                    * pow(modulus // gcd, -1, reduced_modulus)
                    % reduced_modulus
                )
                period = modulus // gcd * key_step
                first = worker + modulus * multiplier
                if first < begin:
                    first += _ceil_div(begin - first, period) * period
                if first >= end:
                    continue
                last = first + (end - 1 - first) // period * period
                intersection = ((item_axis, last, last + 1, 1),)
                target = []
                for axis, start, stop, step in key_piece.target_ranges:
                    bounds = tuple(
                        _logical_expression_bounds(
                            expression,
                            domain=keys.source_domain,
                            source_bounds=intersection,
                        )
                        for expression in (start, stop)
                    )
                    if None in bounds or any(
                        sympy.simplify(value[1] - value[0]) != 0
                        for value in bounds
                        if value is not None
                    ):
                        return None
                    target.append((axis, bounds[0][0], bounds[1][0], step))  # type: ignore[index]
                candidates.append((last, tuple(target)))
            if not candidates:
                continue
            all_targets = {target for _value, target in candidates}
            if len(all_targets) != 1:
                return None
            last = max(value for value, _target in candidates)
            result.append(
                _CoordinateRelationPiece(
                    ((item_axis, last, last + 1, 1),), all_targets.pop()
                )
            )
        keys_by_item = CoordinateRelation(
            keys.source_domain, keys.target_domain, tuple(result)
        )
        items_by_key = CoordinateRelation(
            keys.target_domain,
            keys.source_domain,
            tuple(
                _CoordinateRelationPiece(
                    piece.target_ranges,
                    tuple(
                        (item_axis, begin, end, step)
                        for _axis, begin, end, step in piece.source_bounds_items
                    ),
                )
                for piece in result
            ),
        )
        return Incidence._from_constructed(items_by_key, keys_by_item=keys_by_item)


def _rectangular_fiber_spec(
    relation: CoordinateRelation,
    *,
    prove_nonnegative: Callable[[sympy.Expr], bool] | None = None,
    allow_block_tail: bool = False,
) -> tuple[CoordinateRelation, RectangularFiberSpec | None]:
    """Normalize rectangular fibers and recognize their separable axes."""
    pieces = list(dict.fromkeys(relation.pieces))
    comparisons = 0
    while True:
        if not _relation_product_is_within_budget(len(pieces), len(pieces)):
            return relation, None
        replacement: tuple[int, int, _CoordinateRelationPiece] | None = None
        for (left_index, left), (right_index, right) in itertools.combinations(
            enumerate(pieces), 2
        ):
            comparisons += 1
            if comparisons > _MAX_RELATION_PRODUCT_STATES:
                return relation, None
            differing = tuple(
                index
                for index, pair in enumerate(
                    zip(left.target_ranges, right.target_ranges, strict=True)
                )
                if pair[0] != pair[1]
            )
            if (
                left.source_bounds_items != right.source_bounds_items
                or len(differing) != 1
            ):
                continue
            (index,) = differing
            axis, begin, end, step = left.target_ranges[index]
            other_axis, other_begin, other_end, other_step = right.target_ranges[index]
            left_first = sympy.simplify(end - other_begin) == 0
            right_first = sympy.simplify(other_end - begin) == 0
            if (
                axis != other_axis
                or step != 1
                or other_step != 1
                or not (left_first or right_first)
                or not all(
                    _is_provably_nonnegative(width, prove_nonnegative)
                    for width in (end - begin, other_end - other_begin)
                )
            ):
                continue
            ranges = list(left.target_ranges)
            ranges[index] = (
                axis,
                begin if left_first else other_begin,
                other_end if left_first else end,
                1,
            )
            replacement = (
                left_index,
                right_index,
                _CoordinateRelationPiece(left.source_bounds_items, tuple(ranges)),
            )
            break
        if replacement is None:
            break
        left_index, right_index, merged = replacement
        pieces = [
            merged if index == left_index else piece
            for index, piece in enumerate(pieces)
            if index != right_index
        ]
    normalized = CoordinateRelation(
        relation.source_domain, relation.target_domain, tuple(pieces)
    )
    if (
        not normalized.source_domain.axis_order
        or len(normalized.pieces) != 1
        or normalized.pieces[0].source_bounds_items
        != _full_bounds(normalized.source_domain)
    ):
        return normalized, None
    (piece,) = normalized.pieces
    source_symbols = {
        coordinate_axis_symbol(axis): axis
        for axis in normalized.source_domain.axis_order
    }
    used_sources: set[int] = set()
    mappings, full_targets = [], []
    for target_axis, begin, end, step in piece.target_ranges:
        target_count = normalized.target_domain.axis_count_expressions[target_axis]
        if (
            step == 1
            and _is_provably_nonnegative(-begin, prove_nonnegative)
            and _is_provably_nonnegative(
                sympy.simplify(end - target_count), prove_nonnegative
            )
        ):
            full_targets.append(target_axis)
            continue
        interval = _single_axis_interval(begin, end, domain=normalized.source_domain)
        if interval is not None:
            source_axis, stride, offset, width = interval
            remaining = sympy.simplify(
                target_count
                - width * normalized.source_domain.axis_count_expressions[source_axis]
            )
            if (
                step != 1
                or source_axis in used_sources
                or offset
                or width != stride
                or width <= 0
                or not _is_provably_nonnegative(remaining, None)
                or (not allow_block_tail and remaining != 0)
            ):
                return normalized, None
            used_sources.add(source_axis)
            mappings.append((source_axis, target_axis, "block", width))
            continue
        source_axes = tuple(source_symbols.keys() & begin.free_symbols)
        if (
            step != 1
            or sympy.simplify(end - begin) != 1
            or len(source_axes) != 1
            or source_symbols[source_axes[0]] in used_sources
        ):
            return normalized, None
        source_symbol = source_axes[0]
        source_axis = source_symbols[source_symbol]
        quotient = _static_integer_quotient(begin)
        width = (
            1
            if sympy.simplify(begin - source_symbol) == 0
            else (
                quotient[1]
                if quotient is not None
                and sympy.simplify(quotient[0] - source_symbol) == 0
                else None
            )
        )
        if width is None or not _integer_partition_expressions_equal(
            normalized.source_domain.axis_count_expressions[source_axis],
            width * target_count,
        ):
            return normalized, None
        used_sources.add(source_axis)
        mappings.append((source_axis, target_axis, "quotient", width))
    return normalized, (tuple(mappings), tuple(full_targets))


@dataclasses.dataclass(frozen=True, init=False)
class KeyPartition:
    """A complete fine-to-coarse key partition with exact fiber counts."""

    fine_keys_by_coarse_key: CoordinateRelation
    coarse_key_by_fine_key: CoordinateRelation
    fine_key_count_by_coarse_key: CoordinateRelation

    def as_incidence(self, *, grouped_items: DenseTaskOrder | None = None) -> Incidence:
        """Return coarse keys and their fine members as one incidence."""
        return Incidence._from_constructed(
            self.fine_keys_by_coarse_key,
            keys_by_item=self.coarse_key_by_fine_key,
            count_by_key=self.fine_key_count_by_coarse_key,
            grouped_items=grouped_items,
        )

    def rekey_fine(self, fine_keys: Incidence) -> KeyPartition | None:
        """Pull a partition back through a total fine-key incidence."""
        reverse = fine_keys.reversed()
        if (
            reverse is None
            or fine_keys.items_by_key.target_domain
            != self.coarse_key_by_fine_key.source_domain
            or not fine_keys.items_by_key.is_total_function()
        ):
            return None
        composed = self.as_incidence().then(reverse)
        bijective = (
            _integer_partition_expressions_equal(
                fine_keys.items_by_key.source_domain.size_expr,
                fine_keys.items_by_key.target_domain.size_expr,
            )
            and reverse.items_by_key.is_total_function()
        )
        count = (
            self.fine_key_count_by_coarse_key
            if bijective
            else None
            if composed is None
            else composed.count_by_key
        )
        if composed is None or composed.keys_by_item is None or count is None:
            return None
        return KeyPartition._from_constructed(
            composed.items_by_key,
            composed.keys_by_item,
            count,
        )

    @classmethod
    def _from_constructed(
        cls,
        fine_keys_by_coarse_key: CoordinateRelation,
        coarse_key_by_fine_key: CoordinateRelation,
        fine_key_count_by_coarse_key: CoordinateRelation,
    ) -> KeyPartition:
        coarse_domain = fine_keys_by_coarse_key.source_domain
        fine_domain = fine_keys_by_coarse_key.target_domain
        if (
            coarse_key_by_fine_key.source_domain != fine_domain
            or coarse_key_by_fine_key.target_domain != coarse_domain
            or fine_key_count_by_coarse_key.source_domain != coarse_domain
            or len(fine_key_count_by_coarse_key.target_domain.axis_order) != 1
        ):
            raise ValueError("key partition requires complete compatible maps")
        result = object.__new__(cls)
        for name, value in (
            ("fine_keys_by_coarse_key", fine_keys_by_coarse_key),
            ("coarse_key_by_fine_key", coarse_key_by_fine_key),
            ("fine_key_count_by_coarse_key", fine_key_count_by_coarse_key),
        ):
            object.__setattr__(result, name, value)
        return result

    @classmethod
    def _rectangular(
        cls,
        fine_domain: CoordinateDomain,
        coarse_domain: CoordinateDomain,
        coarse_by_fine: CoordinateRelation,
        widths: dict[int, int],
        full_axes: frozenset[int],
        *,
        clipped_axes: frozenset[int],
        other_axes: tuple[int, ...] = (),
    ) -> KeyPartition:
        fine_counts = fine_domain.axis_count_expressions
        fine_by_coarse = CoordinateRelation(
            coarse_domain,
            fine_domain,
            (
                _CoordinateRelationPiece(
                    _full_bounds(coarse_domain),
                    tuple(
                        (axis, 0, fine_counts[axis], 1)
                        if axis in full_axes
                        else (
                            axis,
                            widths[axis] * coordinate_axis_symbol(axis),
                            widths[axis] * (coordinate_axis_symbol(axis) + 1),
                            1,
                        )
                        for axis in fine_domain.axis_order
                    ),
                ),
            ),
        )
        fiber = capacity = sympy.Integer(1)
        for axis, count in fine_domain.axis_counts_items:
            if axis in full_axes:
                factor = count
            else:
                capacity *= widths[axis]
                factor = (
                    sympy.Min(
                        widths[axis],
                        count - widths[axis] * coordinate_axis_symbol(axis),
                    )
                    if axis in clipped_axes
                    else widths[axis]
                )
            fiber *= factor
        capacity *= sympy.prod(fine_counts[axis] for axis in full_axes)
        count_axis = _next_axis(
            fine_domain.axis_order, coarse_domain.axis_order, other_axes
        )
        counts = _full_point_map(
            coarse_domain,
            CoordinateDomain.scalar(capacity + 1, axis=count_axis, kind="value"),
            (sympy.simplify(fiber),),
        )
        return cls._from_constructed(fine_by_coarse, coarse_by_fine, counts)

    @classmethod
    def projection(
        cls, fine_domain: CoordinateDomain, coarse_domain: CoordinateDomain
    ) -> KeyPartition | None:
        """Construct a coordinate projection and its complete fibers together."""
        coarse_by_fine = CoordinateRelation.projection(fine_domain, coarse_domain)
        if coarse_by_fine is None:
            return None
        coarse_axes = frozenset(coarse_domain.axis_order)
        return cls._rectangular(
            fine_domain,
            coarse_domain,
            coarse_by_fine,
            dict.fromkeys(coarse_axes, 1),
            frozenset(fine_domain.axis_order) - coarse_axes,
            clipped_axes=frozenset(),
        )

    @classmethod
    def contiguous_segments(
        cls,
        fine_domain: CoordinateDomain,
        coarse_domain: CoordinateDomain,
        segmented_axis: int,
        segments: tuple[tuple[int, int], ...],
    ) -> KeyPartition | None:
        """Construct a contiguous partition of one fine-domain axis."""
        try:
            coarse_stage, *coarse_outer = coarse_domain.axis_order
            segmented_extent = fine_domain.axis_counts[segmented_axis]
            stage_count = coarse_domain.axis_counts[coarse_stage]
        except (KeyError, ValueError):
            return None
        fine_outer = tuple(
            axis for axis in fine_domain.axis_order if axis != segmented_axis
        )
        if (
            not segments
            or segments[0][0] != 0
            or segments[-1][1] != segmented_extent
            or any(begin >= end for begin, end in segments)
            or any(left[1] != right[0] for left, right in itertools.pairwise(segments))
            or len(segments) != stage_count
            or len(fine_outer) != len(coarse_outer)
            or any(
                fine_domain.axis_count_expressions[fine_axis]
                != coarse_domain.axis_count_expressions[coarse_axis]
                for fine_axis, coarse_axis in zip(fine_outer, coarse_outer, strict=True)
            )
        ):
            return None
        coarse_counts = coarse_domain.axis_count_expressions
        fine_counts = fine_domain.axis_count_expressions
        outer_axes = dict(zip(fine_outer, coarse_outer, strict=True))

        def stage_bounds(stage: int) -> ConcreteRelationBounds:
            return tuple(
                (axis, stage, stage + 1, 1)
                if axis == coarse_stage
                else (axis, 0, coarse_counts[axis], 1)
                for axis in coarse_domain.axis_order
            )

        fine_by_coarse = CoordinateRelation(
            coarse_domain,
            fine_domain,
            tuple(
                _CoordinateRelationPiece(
                    stage_bounds(stage),
                    tuple(
                        (axis, begin, end, 1)
                        if axis == segmented_axis
                        else (
                            axis,
                            coordinate_axis_symbol(outer_axes[axis]),
                            coordinate_axis_symbol(outer_axes[axis]) + 1,
                            1,
                        )
                        for axis in fine_domain.axis_order
                    ),
                )
                for stage, (begin, end) in enumerate(segments)
            ),
        )
        coarse_by_fine = CoordinateRelation.point_map(
            fine_domain,
            coarse_domain,
            tuple(
                (
                    tuple(
                        (axis, begin, end, 1)
                        if axis == segmented_axis
                        else (axis, 0, fine_counts[axis], 1)
                        for axis in fine_domain.axis_order
                    ),
                    (stage, *(coordinate_axis_symbol(axis) for axis in fine_outer)),
                )
                for stage, (begin, end) in enumerate(segments)
            ),
        )
        count_axis = _next_axis(fine_domain.axis_order, coarse_domain.axis_order)
        counts = CoordinateRelation.point_map(
            coarse_domain,
            CoordinateDomain.scalar(
                max(end - begin for begin, end in segments) + 1,
                axis=count_axis,
                kind="value",
            ),
            tuple(
                (stage_bounds(stage), (end - begin,))
                for stage, (begin, end) in enumerate(segments)
            ),
        )
        return cls._from_constructed(fine_by_coarse, coarse_by_fine, counts)

    @classmethod
    def from_fixed_width_publication(
        cls, publication: CoordinateRelation
    ) -> tuple[KeyPartition, Incidence] | None:
        """Construct a fixed-width key quotient with exact clipped-tail counts."""
        publication, structure = _rectangular_fiber_spec(
            publication, allow_block_tail=True
        )
        if structure is None:
            return None
        mappings, full_axes = structure
        if not mappings:
            return None
        producer, fine = publication.source_domain, publication.target_domain
        fine_counts = fine.axis_count_expressions
        axes = tuple(target for _source, target, _mode, _width in mappings)
        partition_widths = {
            target: width if mode == "block" else 1
            for _source, target, mode, width in mappings
        }
        coarse = CoordinateDomain(
            axes,
            tuple(
                (axis, _ceil_div(fine_counts[axis], partition_widths[axis]))
                for axis in axes
            ),
            kind="event",
            identity=fine.identity,
            _allow_empty=any(fine_counts[axis].is_zero is True for axis in axes),
        )
        coarse_by_fine = _full_point_map(
            fine,
            coarse,
            tuple(
                FloorDiv(coordinate_axis_symbol(axis), partition_widths[axis])
                for axis in axes
            ),
        )
        partition = cls._rectangular(
            fine,
            coarse,
            coarse_by_fine,
            partition_widths,
            frozenset(full_axes),
            clipped_axes=frozenset(axes) - frozenset(full_axes),
            other_axes=producer.axis_order,
        )
        source_axes = tuple(source for source, _target, _mode, _width in mappings)
        producer_keys = CoordinateDomain(
            source_axes,
            tuple(
                (source, coarse.axis_count_expressions[target])
                for source, target, _mode, _width in mappings
            ),
            kind="event",
            identity=coarse.identity,
            _allow_empty=coarse._allow_empty,
        )
        producer_key_by_item = _full_point_map(
            producer,
            producer_keys,
            tuple(
                coordinate_axis_symbol(source)
                if mode == "block"
                else FloorDiv(coordinate_axis_symbol(source), width)
                for source, _target, mode, width in mappings
            ),
        )
        producer_partition = cls._rectangular(
            producer,
            producer_keys,
            producer_key_by_item,
            {
                source: 1 if mode == "block" else width
                for source, _target, mode, width in mappings
            },
            frozenset(producer.axis_order) - frozenset(source_axes),
            clipped_axes=frozenset(
                source for source, _target, mode, _width in mappings if mode == "block"
            ),
            other_axes=fine.axis_order,
        )
        incidence = producer_partition.as_incidence().rename_domains(coarse, producer)
        return None if incidence is None else (partition, incidence)


def _in_domain_point_support(
    piece: _CoordinateRelationPiece,
    source_domain: CoordinateDomain,
    target_domain: CoordinateDomain,
) -> ConcreteRelationBounds | None:
    """Tighten a point-map guard by its affine target-domain bounds."""
    try:
        target_counts = target_domain.axis_counts
    except ValueError:
        return None
    bounds = {
        axis: [begin, end, step] for axis, begin, end, step in piece.source_bounds_items
    }
    for target_axis, begin, end, step in piece.target_ranges:
        if step != 1 or sympy.simplify(end - begin) != 1:
            return None
        if not begin.free_symbols:
            if not isinstance(begin, sympy.Integer):
                return None
            if not 0 <= int(begin) < target_counts[target_axis]:
                first_axis = source_domain.axis_order[0]
                bounds[first_axis][1] = bounds[first_axis][0]
            continue
        value_bounds = _logical_expression_bounds(
            begin,
            domain=source_domain,
            source_bounds=tuple((axis, *values) for axis, values in bounds.items()),
        )
        if (
            value_bounds is not None
            and _is_provably_nonnegative(value_bounds[0], None)
            and _is_provably_nonnegative(
                target_counts[target_axis] - value_bounds[1] - 1, None
            )
        ):
            continue
        preimage = _point_expression_preimage(
            begin,
            lower=0,
            upper=target_counts[target_axis],
            domain=source_domain,
        )
        if preimage is True:
            continue
        if preimage is False:
            first_axis = source_domain.axis_order[0]
            bounds[first_axis][1] = bounds[first_axis][0]
            continue
        if preimage is None:
            return None
        source_axis, lower, upper, preimage_step = preimage
        source_begin, source_end, source_step = bounds[source_axis]
        lower, upper = max(source_begin, lower), min(source_end, upper)
        period = math.lcm(source_step, preimage_step)
        if period > _MAX_RELATION_PRODUCT_STATES:
            return None
        first = next(
            (
                value
                for value in range(lower, min(upper, lower + period))
                if (value - source_begin) % source_step == 0
                and (value - preimage[1]) % preimage_step == 0
            ),
            upper,
        )
        bounds[source_axis] = [first, upper, period]
    return tuple((axis, *bounds[axis]) for axis in source_domain.axis_order)


def _transpose_projection_or_singletons(
    relation: CoordinateRelation,
) -> CoordinateRelation | None:
    """Transpose a coordinate projection or explicit singleton pair cells."""
    projection = KeyPartition.projection(relation.target_domain, relation.source_domain)
    if projection is not None and projection.fine_keys_by_coarse_key == relation:
        return projection.coarse_key_by_fine_key
    pieces = []
    for piece in relation.pieces:
        if any(
            step != 1 or sympy.simplify(end - begin) != 1
            for _axis, begin, end, step in piece.source_bounds_items
        ):
            return None
        source_bounds = []
        for axis, begin, end, step in piece.target_ranges:
            bounds = _logical_expression_bounds(
                begin,
                domain=relation.source_domain,
                source_bounds=piece.source_bounds_items,
            )
            if (
                step != 1
                or bounds is None
                or sympy.simplify(bounds[1] - bounds[0]) != 0
            ):
                return None
            end_bounds = _logical_expression_bounds(
                end,
                domain=relation.source_domain,
                source_bounds=piece.source_bounds_items,
            )
            if end_bounds is None or sympy.simplify(end_bounds[0] - bounds[0]) != 1:
                return None
            source_bounds.append((axis, bounds[0], end_bounds[0], 1))
        pieces.append(
            _CoordinateRelationPiece(tuple(source_bounds), piece.source_bounds_items)
        )
    return CoordinateRelation(
        relation.target_domain, relation.source_domain, tuple(dict.fromkeys(pieces))
    )


def _source_bounds_are_disjoint(
    left: ConcreteRelationBounds,
    right: ConcreteRelationBounds,
) -> bool:
    """Prove two concrete strided source bounds have no common point."""
    for lhs, rhs in zip(left, right, strict=True):
        left_axis, left_begin, left_end, left_step = lhs
        right_axis, right_begin, right_end, right_step = rhs
        if (
            left_axis != right_axis
            or max(left_begin, right_begin) >= min(left_end, right_end)
            or (right_begin - left_begin) % math.gcd(left_step, right_step)
        ):
            return True
    return False


def _source_boxes_partition_domain(
    boxes: tuple[ConcreteRelationBounds, ...],
    domain: CoordinateDomain,
) -> bool:
    """Prove that distinct concrete boxes partition a finite domain."""
    if not boxes or not _relation_product_is_within_budget(len(boxes), len(boxes)):
        return False
    if not domain.axis_order:
        return boxes == ((),)
    if boxes == (_full_bounds(domain),):
        return True
    counts = domain.axis_counts
    if any(
        tuple(axis for axis, _begin, _end, _step in box) != domain.axis_order
        or any(
            step <= 0 or begin < 0 or end > counts[axis] or begin >= end
            for axis, begin, end, step in box
        )
        for box in boxes
    ):
        return False
    if (
        sum(
            math.prod(_ceil_div(end - begin, step) for _axis, begin, end, step in box)
            for box in boxes
        )
        != domain.size
    ):
        return False

    sweep_index = max(
        range(len(domain.axis_order)),
        key=lambda index: len({(box[index][1], box[index][2]) for box in boxes}),
    )
    active: list[ConcreteRelationBounds] = []
    for box in sorted(boxes, key=lambda item: item[sweep_index][1]):
        begin = box[sweep_index][1]
        active = [other for other in active if other[sweep_index][2] > begin]
        if any(not _source_bounds_are_disjoint(other, box) for other in active):
            return False
        active.append(box)
    return True


def _key_major_order(
    items: CoordinateRelation,
    keys: CoordinateRelation | None,
    counts: CoordinateRelation | None,
    structure: RectangularFiberSpec | None,
) -> tuple[int | None, DenseTaskOrder | None]:
    """Construct an order for a separable layout or regular scalar spans."""
    if not items.pieces or any(
        not _target_ranges_are_valid(
            piece.target_ranges,
            source_domain=items.source_domain,
            source_bounds=piece.source_bounds_items,
            target_domain=items.target_domain,
            clipped=False,
        )
        for piece in items.pieces
    ):
        return None, None
    key_domain, item_domain = items.source_domain, items.target_domain
    count_bounds = None if counts is None else counts.value_bounds()
    if structure is not None and count_bounds is not None:
        mappings, full_axes = structure
        by_key = {source: (target, width) for source, target, _mode, width in mappings}
        if set(by_key) == set(key_domain.axis_order) and (
            len(key_domain.axis_order) == 1
            or all(width == 1 for _target, width in by_key.values())
        ):
            order = (*full_axes, *(by_key[axis][0] for axis in key_domain.axis_order))
            fiber_size = count_bounds[0]
            if (
                count_bounds[0] == count_bounds[1]
                and _integer_partition_expressions_equal(
                    fiber_size * key_domain.size_expr, item_domain.size_expr
                )
                and set(order) == set(item_domain.axis_order)
            ):
                return fiber_size, DenseTaskOrder.from_pid(item_domain, order)
    if (
        counts is not None
        and keys is not None
        and len(key_domain.axis_order) == 1
        and len(item_domain.axis_order) == 1
        and _scalar_relation_is_nondecreasing(keys)
    ):
        return None, DenseTaskOrder.from_pid(item_domain, item_domain.axis_order)
    if len(key_domain.axis_order) != 1 or len(item_domain.axis_order) != 1:
        key_order = DenseTaskOrder.from_pid(key_domain, key_domain.axis_order)
        item_order = DenseTaskOrder.from_pid(item_domain, item_domain.axis_order)
        if key_order is None or item_order is None:
            return None, None
        scalar_items = key_order.tasks_by_ordinal.then(items)
        scalar_items = (
            None
            if scalar_items is None
            else _linearize_fibers(scalar_items, item_order.ordinal_by_task)
        )
        scalar_keys = None if keys is None else item_order.tasks_by_ordinal.then(keys)
        scalar_keys = (
            None if scalar_keys is None else scalar_keys.then(key_order.ordinal_by_task)
        )
        scalar_counts = (
            None if counts is None else key_order.tasks_by_ordinal.then(counts)
        )
        if scalar_items is None:
            return None, None
        fiber_size, grouped = _key_major_order(
            scalar_items, scalar_keys, scalar_counts, None
        )
        if grouped is None:
            return fiber_size, None
        tasks = grouped.tasks_by_ordinal.then(item_order.tasks_by_ordinal)
        inverse = item_order.ordinal_by_task.then(grouped.ordinal_by_task)
        return (
            (fiber_size, None)
            if tasks is None or inverse is None
            else (fiber_size, DenseTaskOrder._from_constructed(tasks, inverse))
        )
    (key_axis,), (item_axis,) = key_domain.axis_order, item_domain.axis_order
    key_symbol = coordinate_axis_symbol(key_axis)
    source_bounds = _full_bounds(key_domain)
    if any(piece.source_bounds_items != source_bounds for piece in items.pieces):
        return None, None
    ranges = []
    for piece in items.pieces:
        if len(piece.target_ranges) != 1:
            return None, None
        _axis, begin, end, step = piece.target_ranges[0]
        ranges.append((begin, end, step))
    base = ranges[0][0]
    spans = []
    for begin, end, step in ranges:
        offset = sympy.simplify(begin - base)
        count = sympy.simplify(_ceil_div(end - begin, step))
        if (
            step <= 0
            or not isinstance(offset, sympy.Integer)
            or not isinstance(count, sympy.Integer)
            or count <= 0
        ):
            return None, None
        spans.append((int(offset), int(count), step))
    spans.sort()
    span_width, item_step = spans[0][1:]
    segment_stride = 0 if len(spans) == 1 else spans[1][0] - spans[0][0]
    if any(
        count != span_width or step != item_step for _offset, count, step in spans
    ) or (
        len(spans) > 1
        and (
            segment_stride <= (span_width - 1) * item_step
            or any(
                right[0] - left[0] != segment_stride
                for left, right in itertools.pairwise(spans)
            )
        )
    ):
        return None, None
    fiber_size = len(spans) * span_width
    if not _integer_partition_expressions_equal(
        fiber_size * key_domain.size_expr, item_domain.size_expr
    ):
        return fiber_size, None
    try:
        _ = key_domain.size, item_domain.size
    except ValueError:
        return fiber_size, None
    if keys is None or not keys.is_total_function():
        return fiber_size, None
    begin = sympy.simplify(base + spans[0][0])
    segment_count = len(spans)
    grouped_domain = CoordinateDomain.scalar(
        item_domain.size,
        axis=_next_axis(key_domain.axis_order, item_domain.axis_order),
        kind="task_order",
        identity=item_domain.identity,
    )
    grouped_axis = grouped_domain.axis_order[0]
    grouped_symbol = coordinate_axis_symbol(grouped_axis)
    key = FloorDiv(grouped_symbol, fiber_size)
    local = grouped_symbol - fiber_size * key
    segment = FloorDiv(local, span_width) if segment_count > 1 else 0
    value = (
        begin.xreplace({key_symbol: key})
        + segment_stride * segment
        + item_step * (local - span_width * segment)
    )
    tasks = _full_point_map(grouped_domain, item_domain, (value,))
    item = coordinate_axis_symbol(item_axis)
    inverse_specs = []
    for piece in keys.pieces:
        key = _scalar_point_expression(piece)
        if key is None:
            return fiber_size, None
        delta = sympy.simplify(item - begin.xreplace({key_symbol: key}))
        segment = FloorDiv(delta, segment_stride) if segment_count > 1 else 0
        within_delta = delta - segment * segment_stride
        within = within_delta if item_step == 1 else FloorDiv(within_delta, item_step)
        local = segment * span_width + within
        inverse_specs.append(
            (
                piece.source_bounds_items,
                (
                    _simplify_logical_expression(
                        key * fiber_size + local,
                        domain=item_domain,
                        source_bounds=piece.source_bounds_items,
                    ),
                ),
            )
        )
    inverse = CoordinateRelation.point_map(
        item_domain, grouped_domain, tuple(dict.fromkeys(inverse_specs))
    )
    return fiber_size, DenseTaskOrder._from_constructed(tasks, inverse)


def _source_box_covers(
    outer: ConcreteRelationBounds,
    inner: ConcreteRelationBounds,
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
    relation: CoordinateRelation,
    *,
    include_domain: bool = False,
) -> tuple[ConcreteRelationBounds, ...] | None:
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
    sizes = tuple(map(len, intervals))
    if (
        not _relation_product_is_within_budget(*sizes)
        or math.prod(sizes) > _MAX_RELATION_PIECES
    ):
        return None
    return tuple(
        tuple(cell)
        for cell in itertools.product(*intervals)
        if any(
            _source_box_covers(piece.source_bounds_items, tuple(cell))
            for piece in relation.pieces
        )
        or include_domain
    )


def _logical_expression_bounds(
    expression: sympy.Expr,
    domain: CoordinateDomain,
    source_bounds: RelationBounds,
) -> tuple[sympy.Expr, sympy.Expr] | None:
    return _analyze_integer_expression(expression, domain, source_bounds)[1]


def _simplify_logical_expression(
    expression: sympy.Expr,
    domain: CoordinateDomain,
    source_bounds: ConcreteRelationBounds,
) -> sympy.Expr:
    return _analyze_integer_expression(expression, domain, source_bounds, True)[0]


def _target_ranges_are_valid(
    target_ranges: TargetRanges,
    *,
    source_domain: CoordinateDomain,
    source_bounds: ConcreteRelationBounds,
    target_domain: CoordinateDomain,
    clipped: bool,
) -> bool:
    for axis, begin, end, step in target_ranges:
        if step <= 0 or (clipped and step != 1):
            return False
        count = sympy.simplify(_ceil_div(end - begin, step))
        bounds = tuple(
            _logical_expression_bounds(
                expression, domain=source_domain, source_bounds=source_bounds
            )
            for expression in (
                (begin, end, end - begin)  # pyrefly: ignore[unsupported-operation]
                if clipped
                else (begin, count, begin + (count - 1) * step)
            )
        )
        if None in bounds:
            return False
        first, middle, last = cast("tuple[tuple[sympy.Expr, sympy.Expr], ...]", bounds)
        checks = (
            (
                target_domain.axis_count_expressions[axis] - 1 - first[1],
                middle[0] - 1,
                last[0] - 1,
            )
            if clipped
            else (
                first[0],
                middle[0] - 1,
                target_domain.axis_count_expressions[axis] - 1 - last[1],
            )
        )
        if not all(_is_provably_nonnegative(value, None) for value in checks):
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
                available_end - target_domain.axis_count_expressions[axis]  # pyrefly: ignore[unsupported-operation]
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
    affine = _static_affine_coefficients(begin, domain=domain)
    width_expression = sympy.simplify(end - begin)  # pyrefly: ignore[unsupported-operation]
    if (
        affine is None
        or width_expression.free_symbols
        or width_expression.is_integer is not True
    ):
        return None
    coefficients, offset = affine
    varying = tuple((axis, value) for axis, value in coefficients.items() if value)
    if len(varying) != 1:
        return None
    axis, stride = varying[0]
    width = int(width_expression)
    return (axis, stride, offset, width) if width > 0 else None


def _static_affine_coefficients(
    expression: sympy.Expr,
    *,
    domain: CoordinateDomain,
) -> tuple[dict[int, int], int] | None:
    """Return nonnegative static coefficients for one affine expression."""
    expanded = sympy.expand(expression)
    remainder = expanded
    coefficients: dict[int, int] = {}
    for axis in domain.axis_order:
        symbol = coordinate_axis_symbol(axis)
        coefficient = sympy.simplify(expanded.coeff(symbol))
        if coefficient.free_symbols or coefficient.is_integer is not True:
            return None
        value = int(coefficient)
        if value < 0:
            return None
        coefficients[axis] = value
        remainder -= coefficient * symbol  # pyrefly: ignore[unsupported-operation]
    remainder = sympy.simplify(remainder)
    if remainder.free_symbols or remainder.is_integer is not True:
        return None
    return coefficients, int(remainder)


def _linearize_fibers(
    incidence: CoordinateRelation,
    ordinal_by_item: CoordinateRelation,
) -> CoordinateRelation | None:
    """Map rectangular item fibers through a dense affine ordinal codec."""
    if (
        incidence.target_domain != ordinal_by_item.source_domain
        or len(ordinal_by_item.target_domain.axis_order) != 1
        or len(ordinal_by_item.pieces) != 1
    ):
        return None
    order_piece = ordinal_by_item.pieces[0]
    if (
        order_piece.source_bounds_items != _full_bounds(incidence.target_domain)
        or len(order_piece.target_ranges) != 1
    ):
        return None
    target_axis, expression, end, step = order_piece.target_ranges[0]
    affine = _static_affine_coefficients(expression, domain=incidence.target_domain)
    if step != 1 or sympy.simplify(end - expression) != 1 or affine is None:
        return None
    coefficients, offset = affine
    pieces = []
    for piece in incidence.pieces:
        start = sympy.Integer(offset)
        ranked = []
        for axis, begin, stop, axis_step in piece.target_ranges:
            extent = sympy.simplify(_ceil_div(stop - begin, axis_step))
            if (
                extent.free_symbols
                or not isinstance(extent, sympy.Integer)
                or extent <= 0
            ):
                return None
            ranked.append((coefficients[axis], axis_step, int(extent), begin))
            start += coefficients[axis] * begin
        varying = sorted(
            (coefficient * axis_step, extent)
            for coefficient, axis_step, extent, _begin in ranked
            if extent > 1
        )
        output_step = varying[0][0] if varying else 1
        if output_step <= 0:
            return None
        span = output_step
        for actual_stride, extent in varying:
            if actual_stride != span:
                return None
            span *= extent
        pieces.append(
            _CoordinateRelationPiece(
                piece.source_bounds_items,
                (
                    (
                        target_axis,
                        sympy.simplify(start),
                        sympy.simplify(start + span),
                        output_step,
                    ),
                ),
            )
        )
    return CoordinateRelation(
        incidence.source_domain, ordinal_by_item.target_domain, tuple(pieces)
    )


def _single_axis_floor_point(
    begin: sympy.Expr,
    end: sympy.Expr,
    *,
    domain: CoordinateDomain,
) -> tuple[int, int, int, int, int] | None:
    """Recognize ``floor((a * axis + b) / d) + c`` point mappings."""
    if end != begin + 1 and sympy.simplify(end - begin) != 1:  # pyrefly: ignore[unsupported-operation]
        return None
    source_symbols = {coordinate_axis_symbol(axis): axis for axis in domain.axis_order}
    if len(begin.free_symbols) != 1:
        return None
    (symbol,) = begin.free_symbols
    axis = source_symbols.get(symbol)
    if axis is None:
        return None
    floor_terms = tuple(
        term
        for term in sympy.Add.make_args(begin)
        if term.func in (sympy.floor, FloorDiv)
    )
    if len(floor_terms) != 1:
        return None
    (floor_term,) = floor_terms
    output_offset = sympy.simplify(begin - floor_term)  # pyrefly: ignore[unsupported-operation]
    if output_offset.free_symbols or output_offset.is_integer is not True:
        return None
    numerator, denominator = (
        floor_term.args
        if floor_term.func is FloorDiv
        else sympy.fraction(sympy.together(floor_term.args[0]))
    )
    numerator = sympy.expand(numerator)
    stride = numerator.coeff(symbol)
    offset = sympy.simplify(numerator - stride * symbol)
    if any(
        value.free_symbols or value.is_integer is not True
        for value in (denominator, stride, offset)
    ):
        return None
    divisor, stride_value = int(denominator), int(stride)
    if divisor <= 0 or stride_value <= 0:
        return None
    return axis, stride_value, int(offset), divisor, int(output_offset)


def _point_expression_preimage(
    expression: sympy.Expr,
    *,
    lower: int,
    upper: int,
    domain: CoordinateDomain,
) -> tuple[int, int, int, int] | bool | None:
    """Invert one point expression over a constant half-open target interval."""
    if not expression.free_symbols:
        if expression.is_integer is not True:  # pyrefly: ignore[missing-attribute]
            return None
        return lower <= int(expression) < upper
    if isinstance(expression, sympy.Mod) and upper == lower + 1:
        dividend, modulus = expression.args
        affine = _single_axis_interval(
            dividend,
            dividend + 1,
            domain=domain,  # pyrefly: ignore[unsupported-operation]
        )
        if (
            isinstance(modulus, sympy.Integer)
            and modulus > 0
            and affine is not None
            and affine[1] == affine[3] == 1
        ):
            axis, _stride, offset, _width = affine
            residue = (lower - offset) % int(modulus)
            try:
                extent = domain.axis_counts[axis]
            except ValueError:
                return None
            return axis, residue, extent, int(modulus)
    if len(expression.free_symbols) == 1:
        (symbol,) = expression.free_symbols
        axis = next(
            (
                axis
                for axis in domain.axis_order
                if coordinate_axis_symbol(axis) == symbol
            ),
            None,
        )
        coefficient = sympy.expand(expression).coeff(symbol)
        offset = sympy.simplify(expression - coefficient * symbol)
        if (
            axis is not None
            and coefficient.is_integer is True
            and not coefficient.free_symbols
            and coefficient.is_negative is True
            and offset.is_integer is True
            and not offset.free_symbols
        ):
            stride = -int(coefficient)
            constant = int(offset)
            return (
                axis,
                _ceil_div(constant - upper + 1, stride),
                (constant - lower) // stride + 1,
                1,
            )
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
            1,
        )
    floor_point = _single_axis_floor_point(
        expression,
        expression + 1,  # pyrefly: ignore[unsupported-operation]
        domain=domain,
    )
    if floor_point is None:
        return None
    axis, stride, offset, divisor, output_offset = floor_point
    return (
        axis,
        _ceil_div(divisor * (lower - output_offset) - offset, stride),
        _ceil_div(divisor * (upper - output_offset) - offset, stride),
        1,
    )


def _substitute_composed_expression(
    expression: sympy.Expr,
    *,
    substitutions: dict[sympy.Basic, sympy.Expr],
    source_domain: CoordinateDomain,
    source_bounds: ConcreteRelationBounds,
) -> sympy.Expr:
    """Substitute a point map, simplifying only bounded piecewise operators."""
    result = cast("sympy.Expr", expression.xreplace(substitutions))
    rewritten, bounds = _analyze_integer_expression(
        result,
        source_domain,
        source_bounds,
        result.has(sympy.Mod, sympy.Min, sympy.Max),
    )
    return bounds[0] if bounds is not None and bounds[0] == bounds[1] else rewritten


def _compose_point_relations(
    first: CoordinateRelation,
    following: CoordinateRelation,
) -> CoordinateRelation | None:
    """Compose point-valued relation pieces by exact box preimage."""
    pair_count = len(first.pieces) * len(following.pieces)
    if pair_count > _MAX_RELATION_PIECES or not _relation_product_is_within_budget(
        len(first.pieces), len(following.pieces)
    ):
        return None
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
    pieces: list[_CoordinateRelationPiece] = []
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
                if begin == 0 and end == following.source_domain.axis_counts[axis]:
                    continue
                source_expression = _simplify_logical_expression(
                    first_targets[axis],
                    domain=first.source_domain,
                    source_bounds=first_piece.source_bounds_items,
                )
                expression_bounds = _logical_expression_bounds(
                    source_expression,
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
                    source_expression,
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
                source_axis, preimage_begin, preimage_end, preimage_step = preimage
                source_begin, source_end, source_step = bounds[source_axis]
                restricted_end = min(source_end, preimage_end)
                lower = max(source_begin, preimage_begin)
                period = math.lcm(source_step, preimage_step)
                if period > _MAX_RELATION_PRODUCT_STATES:
                    return None
                restricted_begin = next(
                    (
                        value
                        for value in range(lower, min(restricted_end, lower + period))
                        if (value - source_begin) % source_step == 0
                        and (value - preimage_begin) % preimage_step == 0
                    ),
                    restricted_end,
                )
                if restricted_begin >= restricted_end:
                    valid = False
                    break
                bounds[source_axis] = [
                    restricted_begin,
                    restricted_end,
                    period,
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
                _CoordinateRelationPiece(
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
    return CoordinateRelation(
        source_domain=first.source_domain,
        target_domain=following.target_domain,
        pieces=tuple(dict.fromkeys(pieces)),
    )


def _integer_partition_expressions_equal(
    left: IntegerExpression,
    right: IntegerExpression,
) -> bool:
    """Compare integer expressions after canonical quotient simplification."""
    difference = _simplify_integer_quotients(
        sympy.simplify(sympy.sympify(left) - sympy.sympify(right))
    )
    return sympy.simplify(difference) == 0


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
    if expression.func == sympy.ceiling:
        numerator = sympy.expand(numerator + divisor - 1)
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


def _static_quotient_difference(
    expression: sympy.Expr,
) -> tuple[sympy.Expr, int, sympy.Expr] | None:
    """Recognize a bounded monotone difference of two static quotients."""
    terms = sympy.Add.make_args(sympy.expand(sympy.sympify(expression)))
    if len(terms) != 2:
        return None
    positive: tuple[sympy.Expr, int] | None = None
    negative: tuple[sympy.Expr, int] | None = None
    for term in terms:
        coefficient, primitive = term.as_coeff_Mul()
        quotient = _static_integer_quotient(cast("sympy.Expr", primitive))
        if quotient is None:
            return None
        if coefficient == 1 and positive is None:
            positive = quotient
        elif coefficient == -1 and negative is None:
            negative = quotient
        else:
            return None
    if positive is None or negative is None or positive[1] != negative[1]:
        return None
    base, denominator = negative
    offset = sympy.simplify(positive[0] - base)
    if (
        offset.is_integer is not True
        or offset.is_nonnegative is not True
        or sympy.simplify(denominator - offset).is_nonnegative is not True
    ):
        return None
    return base, denominator, offset


def _exact_quotient_remainder_replacement(expression: sympy.Expr) -> sympy.Expr | None:
    """Collapse one exact ``d * (x // d) + x % d`` pair."""
    if expression.func != sympy.Add:
        return None
    terms = expression.args
    for modulo_index, term in enumerate(terms):
        coefficient, modulo = term.as_coeff_Mul()
        if not isinstance(modulo, sympy.Mod) or len(modulo.args) != 2:
            continue
        dividend, modulus = modulo.args
        if (
            not coefficient.is_number
            or not dividend.is_integer
            or modulus.free_symbols
            or not modulus.is_integer
            or not modulus.is_positive
        ):
            continue
        for quotient_index, candidate in enumerate(terms):
            candidate_coefficient, primitive = candidate.as_coeff_Mul()
            quotient = _static_integer_quotient(cast("sympy.Expr", primitive))
            if (
                quotient_index == modulo_index
                or quotient is None
                or quotient[1] != int(modulus)
                or sympy.simplify(candidate_coefficient - coefficient * modulus) != 0
                or sympy.simplify(sympy.Mod(quotient[0], modulus) - modulo) != 0
            ):
                continue
            return sympy.Add(
                *(
                    value
                    for index, value in enumerate(terms)
                    if index not in (modulo_index, quotient_index)
                ),
                coefficient * quotient[0],
            )
    return None


def _simplify_integer_quotients(expression: sympy.Expr) -> sympy.Expr:
    """Apply the exact Euclidean identities needed by relation constructors."""
    result = sympy.sympify(expression).xreplace(
        {
            node: sympy.floor(numerator / divisor)
            for node in sympy.preorder_traversal(expression)
            if (quotient := _static_integer_quotient(cast("sympy.Expr", node)))
            is not None
            for numerator, divisor in (quotient,)
            if node.func == FloorDiv
        }
    )
    while True:
        replacement = _exact_quotient_remainder_replacement(result)
        if replacement is None and result.func == sympy.Add:
            terms = result.args
            for left_index, left in enumerate(terms):
                left_coefficient, left_quotient = left.as_coeff_Mul()
                if _static_integer_quotient(cast("sympy.Expr", left_quotient)) is None:
                    continue
                for right_index, right in enumerate(
                    terms[left_index + 1 :], left_index + 1
                ):
                    right_coefficient, right_quotient = right.as_coeff_Mul()
                    difference = _static_quotient_difference(
                        sympy.simplify(left_quotient - right_quotient)
                    )
                    if (
                        right_coefficient == -left_coefficient
                        and difference is not None
                        and sympy.simplify(difference[2] - difference[1]) == 0
                    ):
                        replacement = sympy.Add(
                            *(
                                value
                                for index, value in enumerate(terms)
                                if index not in (left_index, right_index)
                            ),
                            left_coefficient,
                        )
                        break
                if replacement is not None:
                    break
        if replacement is None:
            break
        simplified = sympy.simplify(replacement)
        if simplified == result:
            break
        result = simplified
    for node in sympy.preorder_traversal(result):
        if node.func not in (sympy.Min, sympy.Max, SymbolicMin, SymbolicMax):
            continue
        for candidate in node.args:
            smaller = node.func in (sympy.Min, SymbolicMin)
            if all(
                _is_provably_nonnegative(
                    sympy.simplify(other - candidate if smaller else candidate - other),
                    None,
                )
                for other in node.args
                if other != candidate
            ):
                return _simplify_integer_quotients(
                    sympy.simplify(result.xreplace({node: candidate}))
                )
    return sympy.simplify(result)


def _ceil_div(
    numerator: IntegerExpression,
    denominator: int,
) -> sympy.Expr:
    """Return exact integer ceil division for concrete or symbolic numerators."""
    return _normalize_integer_rounding(
        -FloorDiv(-sympy.sympify(numerator), denominator)
    )


@dataclasses.dataclass(frozen=True)
class ExecutionSite:
    """One reachable DeviceIR callsite in an outer task's program order."""

    site_id: int
    root: int
    graph_id: int
    callsite_path: tuple[tuple[int, int], ...]
    parent_site_id: int | None
    kind: Literal["root", "loop", "branch", "while_condition", "while_body"]
    logical_axis_order: tuple[int, ...]
    executes_unconditionally: bool
    can_split_loop: bool

    @property
    def is_root(self) -> bool:
        return self.kind == "root"


def build_execution_sites(device_ir: DeviceIR) -> tuple[ExecutionSite, ...]:
    """Build the reachable DeviceIR callsite tree used by dependency analysis."""
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
    """The memory facts needed to prove a cross-root readiness relation."""

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
    """A conservative region in allocation-address coordinates."""

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
    """One symbolic dependency between execution-site instance domains."""

    dependency_id: int
    consumer_access_id: int
    producer_root: int
    consumer_root: int
    producer_site_id: int | None
    consumer_site_id: int | None
    incidence: Incidence | None


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

    def obligations_by_root_pair(
        self,
    ) -> tuple[tuple[tuple[int, int], frozenset[DependencyObligation]], ...]:
        """Return the canonical dependency manifest grouped by root pair."""
        grouped: dict[tuple[int, int], set[DependencyObligation]] = {}
        for edge in self.edges:
            obligations = grouped.setdefault(
                (edge.producer_root, edge.consumer_root), set()
            )
            for dependency in edge.access_dependencies:
                producer_sites = self.site_ids_by_access[dependency.producer_access_id]
                consumer_sites = self.site_ids_by_access[dependency.consumer_access_id]
                obligations.update(
                    (dependency.dependency_id, producer_site, consumer_site)
                    for producer_site in producer_sites or (None,)
                    for consumer_site in consumer_sites or (None,)
                )
        return tuple((pair, frozenset(grouped[pair])) for pair in sorted(grouped))


@dataclasses.dataclass(frozen=True)
class _ReachingAccess:
    root: int
    access: TileAccess
    region: AllocationRegion


def _access_region(
    access: TileAccess,
    task_family: TaskFamily,
) -> AllocationRegion:
    """Conservatively summarize one root's union of an access."""
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

    position_by_dim = {
        dimension: position for position, dimension in enumerate(access.subscript_dims)
    }
    if len(position_by_dim) != len(access.subscript_dims) or any(
        not 0 <= dimension < len(shape) for dimension in position_by_dim
    ):
        return AllocationRegion(None, False)

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
    if axis is None:
        extent = (
            1
            if access.subscript_is_scalar[position]
            else (
                access.subscript_static_extents[position]
                if position < len(access.subscript_static_extents)
                else None
            )
        )
        if offset is None or extent is None or (offset < 0 and size.free_symbols):
            return None
        normalized_offset = sympy.sympify(offset if offset >= 0 else size + offset)
        if (
            normalized_offset.is_nonnegative is not True
            or (size - normalized_offset - extent).is_nonnegative is not True
        ):
            return None
        return normalized_offset, normalized_offset + extent
    counts = domain.axis_count_expressions
    if offset is None or axis not in counts:
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


def _access_layout(
    access: TileAccess,
    prove_nonnegative: Callable[[sympy.Expr], bool] | None,
) -> _AccessLayout:
    shape = tuple(
        _integer_expression(value, description="access shape")
        for value in access.tensor_shape
    )
    strides = tuple(
        _integer_expression(value, description="access stride")
        for value in access.tensor_strides
    )
    offset = _integer_expression(
        access.storage_offset, description="access storage offset"
    )
    linear_span = None
    if len(shape) == len(strides) and all(
        _is_provably_nonnegative(value, prove_nonnegative)
        for value in (*shape, *strides)
    ):
        linear_span = sympy.simplify(
            1
            + sum(
                (size - 1) * stride for size, stride in zip(shape, strides, strict=True)
            )
        )
    dimensions = None
    if len(shape) == len(strides) and _layout_is_injective((shape, strides, offset)):
        dimensions = tuple(
            index for index, size in enumerate(shape) if sympy.simplify(size - 1) != 0
        )
    positions = {
        dimension: index for index, dimension in enumerate(access.subscript_dims)
    }
    if len(positions) != len(access.subscript_dims) or any(
        not 0 <= dimension < len(shape) for dimension in positions
    ):
        positions = None
    return shape, strides, offset, linear_span, dimensions, positions


def _symbolic_access_map(
    access: TileAccess,
    *,
    layout: _AccessLayout,
    source_domain: CoordinateDomain,
    allocation_domain: CoordinateDomain,
    storage_offset: sympy.Expr | None = None,
    tensor_dimensions: tuple[int, ...] | None = None,
    prove_nonnegative: Callable[[sympy.Expr], bool] | None = None,
) -> _AccessMap | None:
    positions = layout[5]
    if positions is None:
        return None
    shape, strides = layout[:2]
    source_bounds = _full_bounds(source_domain)
    if tensor_dimensions is not None:
        ranges = []
        for allocation_axis, dimension in zip(
            allocation_domain.axis_order, tensor_dimensions, strict=True
        ):
            position = positions.get(dimension)
            interval = (
                (sympy.Integer(0), shape[dimension])
                if position is None
                else _access_interval_expression(
                    access, position=position, domain=source_domain
                )
            )
            if interval is None:
                return None
            ranges.append((allocation_axis, *interval, 1))
        relation = CoordinateRelation(
            source_domain,
            allocation_domain,
            (_CoordinateRelationPiece(source_bounds, tuple(ranges)),),
        )
        return relation, None
    if storage_offset is None:
        storage_offset = layout[2]

    if (
        len(shape) == 1
        and access.subscript_dims == (0,)
        and access.affine_subscript_ranges is not None
    ):
        (stride,) = strides
        if not isinstance(stride, sympy.Integer) or int(stride) <= 0:
            return None
        pieces: list[_CoordinateRelationPiece] = []
        counts = source_domain.axis_count_expressions
        for terms, raw_begin, raw_end, offset_step in access.affine_subscript_ranges:
            offset_begin = _integer_expression(
                raw_begin, description="affine subscript offset"
            )
            offset_width = sympy.simplify(
                _integer_expression(raw_end, description="affine subscript offset")
                - offset_begin
            )
            if (
                offset_step <= 0
                or not isinstance(offset_width, sympy.Integer)
                or int(offset_width) <= 0
                or int(offset_width) % offset_step != 0
                or len({(axis, divisor) for axis, _value, divisor in terms})
                != len(terms)
                or any(
                    axis not in counts
                    or divisor <= 0
                    or not _is_provably_nonnegative(
                        _integer_expression(value, description="affine coefficient"),
                        prove_nonnegative,
                    )
                    for axis, value, divisor in terms
                )
            ):
                return None
            index = offset_begin
            for axis, coefficient, divisor in terms:
                coordinate = coordinate_axis_symbol(axis)
                index += coefficient * (
                    coordinate if divisor == 1 else sympy.floor(coordinate / divisor)
                )
            first = _logical_expression_bounds(
                index, domain=source_domain, source_bounds=source_bounds
            )
            last = _logical_expression_bounds(
                index + offset_width - offset_step,
                domain=source_domain,
                source_bounds=source_bounds,
            )
            if (
                first is None
                or last is None
                or not _is_provably_nonnegative(first[0], prove_nonnegative)
                or not _is_provably_nonnegative(
                    sympy.simplify(shape[0] - 1 - last[1]), prove_nonnegative
                )
            ):
                return None
            address = storage_offset + index * stride
            pieces.append(
                _CoordinateRelationPiece(
                    source_bounds,
                    (
                        (
                            _ALLOCATION_ADDRESS_AXIS,
                            address,
                            address + offset_width * stride,
                            offset_step * int(stride),
                        ),
                    ),
                )
            )
        return CoordinateRelation(source_domain, allocation_domain, tuple(pieces)), None

    intervals: list[tuple[sympy.Expr, sympy.Expr]] = []
    widths: list[sympy.Expr] = []
    for dimension, size in enumerate(shape):
        position = positions.get(dimension)
        interval = (
            (sympy.Integer(0), size)
            if position is None
            else _access_interval_expression(
                access, position=position, domain=source_domain
            )
        )
        if interval is None:
            return None
        begin, end = interval
        width = sympy.simplify(end - begin)
        if not _is_provably_nonnegative(width - 1, prove_nonnegative):
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
                if offset < 0 or not _is_provably_nonnegative(
                    sympy.simplify(size - final_end), prove_nonnegative
                ):
                    return None
        intervals.append(interval)
        widths.append(width)

    span: sympy.Expr = sympy.Integer(1)
    remaining = {index for index, width in enumerate(widths) if width != 1}
    while remaining:
        matches = tuple(
            index for index in remaining if sympy.simplify(strides[index] - span) == 0
        )
        if len(matches) != 1:
            return None
        dimension = matches[0]
        span *= widths[dimension]
        remaining.remove(dimension)
    begin = storage_offset + sum(
        interval[0] * stride
        for interval, stride in zip(intervals, strides, strict=True)
    )
    relation = CoordinateRelation(
        source_domain,
        allocation_domain,
        (
            _CoordinateRelationPiece(
                source_bounds,
                ((_ALLOCATION_ADDRESS_AXIS, begin, begin + span, 1),),
            ),
        ),
    )
    codec = None
    affine = _static_affine_coefficients(begin, domain=source_domain)
    if isinstance(span, sympy.Integer) and int(span) > 0 and affine is not None:
        coefficients, offset = affine
        counts = source_domain.axis_count_expressions
        remaining = {
            axis
            for axis in source_domain.axis_order
            if sympy.simplify(counts[axis] - 1) != 0
        }
        dense_span: sympy.Expr = span
        strides_by_axis: list[tuple[int, sympy.Expr]] = []
        while remaining:
            matches = tuple(
                axis
                for axis in remaining
                if sympy.simplify(coefficients[axis] - dense_span) == 0
            )
            if len(matches) != 1:
                break
            axis = matches[0]
            strides_by_axis.append((axis, sympy.simplify(dense_span / int(span))))
            dense_span = sympy.simplify(dense_span * counts[axis])
            remaining.remove(axis)
        else:
            singleton_axes = set(source_domain.axis_order) - {
                axis for axis, _stride in strides_by_axis
            }
            allocation_count = allocation_domain.axis_count_expressions[
                _ALLOCATION_ADDRESS_AXIS
            ]
            if (
                all(coefficients[axis] == 0 for axis in singleton_axes)
                and _is_provably_nonnegative(sympy.Integer(offset), prove_nonnegative)
                and _is_provably_nonnegative(
                    sympy.simplify(allocation_count - offset - dense_span),
                    prove_nonnegative,
                )
            ):
                codec = offset, int(span), dense_span, tuple(strides_by_axis)
    return relation, codec


def _rectangular_overlap_sources(
    owner: CoordinateRelation,
    query: CoordinateRelation,
) -> CoordinateRelation | None:
    if (
        owner.target_domain != query.target_domain
        or not _relation_product_is_within_budget(len(owner.pieces), len(query.pieces))
        or any(
            piece.source_bounds_items != _full_bounds(owner.source_domain)
            for piece in owner.pieces
        )
    ):
        return None
    owner_counts = owner.source_domain.axis_count_expressions
    allocation_counts = owner.target_domain.axis_count_expressions
    pieces: list[_CoordinateRelationPiece] = []
    for owner_piece, query_piece in itertools.product(owner.pieces, query.pieces):
        owner_ranges = {axis: values for axis, *values in owner_piece.target_ranges}
        query_ranges = {axis: values for axis, *values in query_piece.target_ranges}
        lower = {axis: [] for axis in owner.source_domain.axis_order}
        upper = {axis: [] for axis in owner.source_domain.axis_order}
        for allocation_axis in owner.target_domain.axis_order:
            owner_begin, owner_end, owner_step = owner_ranges[allocation_axis]
            query_begin, query_end, query_step = query_ranges[allocation_axis]
            if owner_step != 1 or query_step != 1:
                return None
            if (
                sympy.simplify(owner_begin) == 0
                and sympy.simplify(owner_end - allocation_counts[allocation_axis]) == 0
            ):
                continue
            interval = _single_axis_interval(
                owner_begin, owner_end, domain=owner.source_domain
            )
            if interval is None:
                return None
            owner_axis, stride, offset, width = interval
            query_interval = _single_axis_interval(
                query_begin, query_end, domain=query.source_domain
            )
            if query_interval is not None:
                _, query_stride, query_offset, query_width = query_interval
                if (
                    width == stride
                    and query_width == query_stride
                    and stride % query_width == 0
                    and (query_offset - offset) % query_width == 0
                ):
                    value = sympy.floor((query_begin - offset) / stride)
                    lower[owner_axis].append(value)
                    upper[owner_axis].append(value + 1)
                    continue
            lower[owner_axis].append(
                sympy.floor((query_begin - offset - width) / stride) + 1
            )
            upper[owner_axis].append(sympy.ceiling((query_end - offset) / stride))
        pieces.append(
            _CoordinateRelationPiece(
                query_piece.source_bounds_items,
                tuple(
                    (
                        axis,
                        sympy.Max(*lower[axis]) if lower[axis] else sympy.Integer(0),
                        (
                            sympy.Min(*upper[axis])
                            if upper[axis]
                            else sympy.sympify(owner_counts[axis])
                        ),
                        1,
                    )
                    for axis in owner.source_domain.axis_order
                ),
            )
        )
    return CoordinateRelation(
        query.source_domain, owner.source_domain, tuple(dict.fromkeys(pieces))
    )


def _dense_overlap_sources(
    owner: _AccessMap,
    query: CoordinateRelation,
    prove_nonnegative: Callable[[sympy.Expr], bool] | None,
) -> CoordinateRelation | None:
    relation, codec = owner
    if codec is None or relation.target_domain != query.target_domain:
        return None
    offset, tile_width, dense_span, stride_items = codec
    tile_strides = dict(stride_items)
    counts = relation.source_domain.axis_count_expressions
    pieces: list[_CoordinateRelationPiece] = []
    for piece in query.pieces:
        if len(piece.target_ranges) != 1:
            return None
        _axis, begin, end, step = piece.target_ranges[0]
        begin_delta, end_delta = (
            sympy.simplify(begin - offset),
            sympy.simplify(end - offset),
        )
        first_bounds = _logical_expression_bounds(
            begin_delta,
            domain=query.source_domain,
            source_bounds=piece.source_bounds_items,
        )
        last_bounds = _logical_expression_bounds(
            sympy.simplify(end_delta - step),
            domain=query.source_domain,
            source_bounds=piece.source_bounds_items,
        )
        width_expression = sympy.simplify(end_delta - begin_delta)
        if (
            first_bounds is None
            or last_bounds is None
            or not isinstance(width_expression, sympy.Integer)
            or not _is_provably_nonnegative(first_bounds[0], prove_nonnegative)
            or not _is_provably_nonnegative(
                sympy.simplify(dense_span - 1 - last_bounds[1]), prove_nonnegative
            )
        ):
            return None
        width = int(width_expression)
        first_ordinal = sympy.floor(begin_delta / tile_width)
        if (
            0 < width <= tile_width
            and step == 1
            and tile_width % width == 0
            and sympy.simplify(sympy.Mod(begin_delta, width)) == 0
        ):
            ordinal_count, ordinal_step = 1, 1
        elif (
            step >= tile_width
            and step % tile_width == 0
            and width > 0
            and width % step == 0
        ):
            ordinal_count, ordinal_step = width // step, step // tile_width
        else:
            return None
        varying_axis = None
        if ordinal_count > 1:
            varying_axis = next(
                (axis for axis, stride in stride_items if stride == ordinal_step), None
            )
            if varying_axis is None:
                return None
        ranges = []
        for axis in relation.source_domain.axis_order:
            count = counts[axis]
            coordinate: sympy.Expr = sympy.Integer(0)
            if sympy.simplify(count - 1) != 0:
                coordinate = sympy.Mod(
                    sympy.floor(first_ordinal / tile_strides[axis]), count
                )
                for _ in range(2):
                    coordinate = _simplify_logical_expression(
                        coordinate,
                        domain=query.source_domain,
                        source_bounds=piece.source_bounds_items,
                    )
            extent = ordinal_count if axis == varying_axis else 1
            bounds = _logical_expression_bounds(
                coordinate,
                domain=query.source_domain,
                source_bounds=piece.source_bounds_items,
            )
            if (
                bounds is None
                or not _is_provably_nonnegative(bounds[0], prove_nonnegative)
                or not _is_provably_nonnegative(
                    sympy.simplify(count - extent - bounds[1]), prove_nonnegative
                )
            ):
                return None
            ranges.append((axis, coordinate, coordinate + extent, 1))
        pieces.append(
            _CoordinateRelationPiece(piece.source_bounds_items, tuple(ranges))
        )
    return CoordinateRelation(
        query.source_domain,
        relation.source_domain,
        tuple(dict.fromkeys(pieces)),
    )


def _access_incidence(
    owner: _AccessMap,
    query: _AccessMap,
    prove_nonnegative: Callable[[sympy.Expr], bool] | None,
) -> Incidence | None:
    owner_relation, _ = owner
    query_relation, _ = query
    items = _rectangular_overlap_sources(owner_relation, query_relation)
    if items is None:
        items = _dense_overlap_sources(owner, query_relation, prove_nonnegative)
    if items is None:
        return None
    keys = _transpose_projection_or_singletons(items)
    if keys is None:
        keys = _rectangular_overlap_sources(query_relation, owner_relation)
    if keys is None:
        keys = _dense_overlap_sources(query, owner_relation, prove_nonnegative)
    return Incidence.from_fibers(
        items, keys_by_item=keys, prove_nonnegative=prove_nonnegative
    )


def _symbolic_dependency_incidence(
    *,
    producer_access: TileAccess,
    producer_domain: CoordinateDomain,
    consumer_access: TileAccess,
    consumer_domain: CoordinateDomain,
    prove_nonnegative: Callable[[sympy.Expr], bool] | None = None,
) -> Incidence | None:
    if (
        not producer_access.layout_is_symbolically_exact
        or not consumer_access.layout_is_symbolically_exact
        or producer_access.has_explicit_mask
        or consumer_access.has_explicit_mask
        or producer_access.allocation_id != consumer_access.allocation_id
    ):
        return None
    producer_layout = _access_layout(producer_access, prove_nonnegative)
    consumer_layout = _access_layout(consumer_access, prove_nonnegative)
    producer_dimensions, consumer_dimensions = producer_layout[4], consumer_layout[4]
    if producer_dimensions is not None and consumer_dimensions is not None:
        producer_geometry = tuple(
            (producer_layout[0][axis], producer_layout[1][axis])
            for axis in producer_dimensions
        )
        consumer_geometry = tuple(
            (consumer_layout[0][axis], consumer_layout[1][axis])
            for axis in consumer_dimensions
        )
        if (
            producer_geometry == consumer_geometry
            and producer_access.storage_offset == consumer_access.storage_offset
        ):
            coordinate_domain = CoordinateDomain(
                tuple(range(len(producer_geometry))),
                tuple(
                    (axis, size)
                    for axis, (size, _stride) in enumerate(producer_geometry)
                ),
                kind="allocation",
                identity=producer_access.allocation_id,
            )
            producer_map = _symbolic_access_map(
                producer_access,
                layout=producer_layout,
                source_domain=producer_domain,
                allocation_domain=coordinate_domain,
                tensor_dimensions=producer_dimensions,
            )
            consumer_map = _symbolic_access_map(
                consumer_access,
                layout=consumer_layout,
                source_domain=consumer_domain,
                allocation_domain=coordinate_domain,
                tensor_dimensions=consumer_dimensions,
            )
            if producer_map is not None and consumer_map is not None:
                incidence = _access_incidence(
                    producer_map, consumer_map, prove_nonnegative
                )
                if incidence is not None:
                    return incidence

    common_offset = _integer_partition_expressions_equal(
        producer_access.storage_offset, consumer_access.storage_offset
    )
    producer_offset = sympy.Integer(0) if common_offset else producer_layout[2]
    consumer_offset = sympy.Integer(0) if common_offset else consumer_layout[2]
    storage_sizes = []
    for layout, offset in (
        (producer_layout, producer_offset),
        (consumer_layout, consumer_offset),
    ):
        span = layout[3]
        if span is None or not _is_provably_nonnegative(offset, prove_nonnegative):
            return None
        end = sympy.simplify(offset + span)
        if end.is_zero is True:
            end = sympy.Integer(1)
        elif not _is_provably_nonnegative(end, prove_nonnegative):
            end = sympy.simplify(sympy.Max(1, end))
        storage_sizes.append(end)
    producer_size, consumer_size = storage_sizes
    delta = sympy.simplify(producer_size - consumer_size)
    if delta == 0 or _is_provably_nonnegative(delta, prove_nonnegative):
        allocation_size = producer_size
    elif _is_provably_nonnegative(-delta, prove_nonnegative):
        allocation_size = consumer_size
    else:
        allocation_size = sympy.simplify(sympy.Max(producer_size, consumer_size))
    allocation_domain = CoordinateDomain(
        (_ALLOCATION_ADDRESS_AXIS,),
        ((_ALLOCATION_ADDRESS_AXIS, allocation_size),),
        kind="allocation",
        identity=producer_access.allocation_id,
    )
    producer_map = _symbolic_access_map(
        producer_access,
        layout=producer_layout,
        storage_offset=producer_offset,
        source_domain=producer_domain,
        allocation_domain=allocation_domain,
        prove_nonnegative=prove_nonnegative,
    )
    consumer_map = _symbolic_access_map(
        consumer_access,
        layout=consumer_layout,
        storage_offset=consumer_offset,
        source_domain=consumer_domain,
        allocation_domain=allocation_domain,
        prove_nonnegative=prove_nonnegative,
    )
    if producer_map is None or consumer_map is None:
        return None
    return _access_incidence(producer_map, consumer_map, prove_nonnegative)


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
    """Bind root and execution-site domains to selected tile geometry."""
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
    """Instantiate site dependencies without enumerating task instances."""
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
                        dependency_id=access_dependency.dependency_id,
                        consumer_access_id=consumer_access.access_id,
                        producer_root=edge.producer_root,
                        consumer_root=edge.consumer_root,
                        producer_site_id=None,
                        consumer_site_id=None,
                        incidence=None,
                    )
                )
                continue
            for producer_site_id, producer_domain in producer_endpoints:
                for consumer_site_id, consumer_domain in consumer_endpoints:
                    result.append(
                        TileDependencyRelation(
                            dependency_id=access_dependency.dependency_id,
                            consumer_access_id=consumer_access.access_id,
                            producer_root=edge.producer_root,
                            consumer_root=edge.consumer_root,
                            producer_site_id=producer_site_id,
                            consumer_site_id=consumer_site_id,
                            incidence=(
                                _symbolic_dependency_incidence(
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
    """Map a consumer site to a preceding site in local program order."""
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
                source_bounds_items=_full_bounds(consumer_domain),
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
    """Return the definitely-uncovered portion of ``target``."""
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
    """Build the minimal source-ordered allocation hazard graph."""
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
