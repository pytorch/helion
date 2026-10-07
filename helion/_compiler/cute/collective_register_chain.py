"""Late typed warp-MMA chains with explicit internal storage lifetimes.

The caller proves region placement, memory disjointness, direct carries and the
same-cell reaching definition of every seed. This module changes ownership and
storage, not scalar arithmetic. Planning is detached from compiler/AST state;
only a successfully planned template may be committed by the integration.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from itertools import count
import math
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import Protocol
from typing import cast

from ... import exc
from .contiguous_copy import ContiguousCopy
from .contiguous_copy import CopyTensorFacts
from .contiguous_copy import _Analysis
from .contiguous_copy import _replace
from .contiguous_copy import plan_contiguous_copy
from .packed_scalar_recipe import rounded_recipe_loop
from .scalar_recipe import ScalarRecipe
from .scalar_recipe import _clone
from .scalar_recipe import _read_names
from .scalar_recipe import build_recipe
from .scalar_recipe_cache import _Dependencies
from .scalar_recipe_rounding import is_rounded_fp32_multiply as _rounded_multiply
from .scalar_recipe_rounding import preserve_fp32_multiply_rounding

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence


class ScalarEmitter(Protocol):
    """A ScalarRecipe or the existing control-preserving scalar epilogue."""

    @property
    def boundary_names(self) -> frozenset[str]: ...

    def emit(
        self,
        replacements: Mapping[str, ast.expr],
        fresh_name: Callable[[str], str],
    ) -> tuple[Sequence[ast.stmt], ast.expr]: ...


@dataclass(frozen=True)
class RegisterChainSite:
    """One admitted contraction; coordinates retain their original scalar type.

    ``seed_name`` is a versioned scalar frontier, never an alias of an arbitrary
    tensor. The integration proves that it is exactly the same M/N cell of
    ``seed_from``. A missing seed recipe means the original direct zero seed.
    The full K domain is [0, static_k_extent), in unmodified BK-sized tiles.
    """

    identity: int
    bm: int
    bn: int
    bk: int
    dtype: str
    m_index: str
    n_index: str
    k_index: str
    m_offset: str
    n_offset: str
    k_offset: str
    reduction_iterator: ast.Call
    static_k_extent: int
    a_recipe: ScalarRecipe
    b_recipe: ScalarRecipe
    seed_recipe: ScalarRecipe | None = None
    seed_from: int | None = None
    seed_name: str | None = None


@dataclass(frozen=True)
class RegisterChainEpilogue:
    """The original terminal value, store pointer and store predicate.

    ``value_recipe`` may preserve conditional scalar statements through the
    existing scalar-epilogue emitter. Its returned expression executes only
    under the original predicate; preceding statements retain their own guards.
    ``pointer_recipe`` returns a pointer, not a synthetic memory load. Output
    layout facts, when supplied, are guarded against the actual output tensor.
    """

    m_index: str
    n_index: str
    accumulator_name: str
    value_recipe: ScalarEmitter
    pointer_recipe: ScalarRecipe
    predicate_recipe: ScalarRecipe
    dtype: str
    output_tensor: str
    output_facts: CopyTensorFacts | None = None


@dataclass(frozen=True)
class ProducerLayout:
    """A blocked contiguous packet layout, independent of tensor identity."""

    rows: int
    columns: int
    threads: int
    width: int

    @property
    def values_per_thread(self) -> int:
        return self.rows * self.columns // self.threads

    @property
    def slots(self) -> int:
        return self.values_per_thread // self.width

    def coordinates(self, tid: str, element: str) -> tuple[ast.expr, ast.expr]:
        flat = (
            f"(({element}) // {self.width} * {self.threads} + ({tid}))"
            f" * {self.width} + ({element}) % {self.width}"
        )
        return _expr(f"({flat}) // {self.columns}"), _expr(f"({flat}) % {self.columns}")


@dataclass(frozen=True)
class _SharedLayout:
    """The same layout description supplies the view and its packet proof."""

    shape: tuple[int, int]
    stride: tuple[int, int]
    swizzle: tuple[int, int, int]

    def source(self) -> str:
        bits, base, shift = self.swizzle
        return (
            f"cute.make_composed_layout(cute.make_swizzle({bits}, {base}, {shift}), "
            f"0, cute.make_layout({self.shape}, stride={self.stride}))"
        )

    def contiguous_packets(self, producer: ProducerLayout, *, transpose: bool) -> bool:
        bits, base, shift = self.swizzle
        # A packet starts at a multiple of width and cannot cross a producer
        # row. The affine layout maps it to consecutive elements even when B
        # is stored as (column, row). Swizzling changes only bits >= base, so
        # it leaves every element within this aligned low-bit packet in order.
        return (
            producer.width in (2, 4, 8)
            and producer.rows > 0
            and producer.columns > 0
            and producer.threads > 0
            and producer.columns % producer.width == 0
            and producer.rows * producer.columns % (producer.threads * producer.width)
            == 0
            and self.shape
            == (
                (producer.columns, producer.rows)
                if transpose
                else (producer.rows, producer.columns)
            )
            and self.stride
            == ((1, producer.columns) if transpose else (producer.columns, 1))
            and base >= producer.width.bit_length() - 1
            and 0 <= bits <= shift
        )


@dataclass(frozen=True)
class RegisterChainPlan:
    """A complete private template; no integration object was mutated."""

    sites: tuple[RegisterChainSite, ...]
    thread_count: int
    shared_bytes: int
    operand_layouts: tuple[tuple[ProducerLayout, ProducerLayout], ...]
    b_k_major: tuple[bool, ...]
    b_register_transpose: tuple[bool, ...]
    slice_transports: int
    matrix_transports: int
    _statements: tuple[ast.stmt, ...]
    _generated_names: tuple[str, ...]
    _tid: str


_HALF_TYPES = frozenset({"cutlass.Float16", "cutlass.BFloat16"})
_SHARED_ALIGNMENT_BYTES = 1024
_TYPE_BYTES = {
    "cutlass.Boolean": 1,
    "cutlass.Int32": 4,
    "cutlass.Int64": 8,
    "cutlass.Uint32": 4,
    "cutlass.Uint64": 8,
    "cutlass.Float16": 2,
    "cutlass.BFloat16": 2,
    "cutlass.Float32": 4,
    "cutlass.Float64": 8,
}


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _statements(source: str) -> list[ast.stmt]:
    return ast.parse(source).body


def _reject(reason: str) -> NoReturn:
    raise exc.BackendUnsupported("cute", f"warp register chain: {reason}")


def _integer(value: ast.expr) -> int | None:
    if isinstance(value, ast.Constant) and type(value.value) is int:
        return value.value
    if (
        isinstance(value, ast.Call)
        and ast.unparse(value.func)
        in {"cutlass.Int32", "cutlass.Int64", "cutlass.Uint32", "cutlass.Uint64"}
        and len(value.args) == 1
        and not value.keywords
    ):
        return _integer(value.args[0])
    return None


def _validate_sites(
    sites: tuple[RegisterChainSite, ...],
    epilogue: RegisterChainEpilogue,
    thread_count: int,
) -> None:
    if len(sites) < 2 or len({site.identity for site in sites}) != len(sites):
        _reject("requires at least two distinct ordered contractions")
    if thread_count != 64:
        _reject("the selected warp atom layout requires 64 physical threads")
    for index, site in enumerate(sites):
        if site.dtype not in _HALF_TYPES:
            _reject("requires original FP16/BF16 operands and FP32 C")
        if (
            site.bm <= 0
            or site.bn <= 0
            or site.bm % 16
            or site.bn % 16
            or site.bk not in (16, 32, 64, 128)
        ):
            _reject("unsupported m16n8k16 warp tile")
        iterator = site.reduction_iterator
        if (
            ast.unparse(iterator.func)
            not in {"range", "cutlass.range", "cutlass.range_constexpr"}
            or len(iterator.args) != 3
            or tuple(_integer(arg) for arg in iterator.args)
            != (0, site.static_k_extent, site.bk)
            or site.static_k_extent <= 0
            or site.static_k_extent % site.bk
            or any(keyword.arg != "unroll" for keyword in iterator.keywords)
        ):
            _reject("requires the complete proven static K range without padding")
        if index == 0:
            if site.seed_from is not None or site.seed_name is not None:
                _reject("the first site cannot read a chain accumulator")
        else:
            previous = sites[index - 1]
            if (
                site.seed_from != previous.identity
                or site.seed_recipe is None
                or site.seed_name is None
                or site.seed_name not in site.seed_recipe.boundary_names
                or (site.bm, site.bn, site.m_offset, site.n_offset)
                != (previous.bm, previous.bn, previous.m_offset, previous.n_offset)
            ):
                _reject("seed must name the adjacent same-cell FP32 frontier")
    if epilogue.dtype not in _HALF_TYPES | {"cutlass.Float32"}:
        _reject("unsupported original output dtype")
    if not epilogue.accumulator_name.isidentifier():
        _reject("output must name its scalar accumulator frontier")
    if epilogue.output_facts is not None and (
        epilogue.output_facts.dtype != epilogue.dtype
    ):
        _reject("output facts change the original store dtype")


class _Names:
    def __init__(self, occupied: set[str]) -> None:
        self.occupied = occupied.copy()
        self.names: list[str] = []
        self.counter = count()

    def __call__(self, hint: str) -> str:
        while True:
            name = f"_helion_rc_{next(self.counter)}_{hint}"
            if name not in self.occupied:
                self.names.append(name)
                self.occupied.add(name)
                return name


def _layout(rows: int, columns: int, threads: int) -> ProducerLayout:
    values = rows * columns // threads
    width = min(8, values, columns)
    while width > 1 and (values % width or columns % width):
        width //= 2
    return ProducerLayout(rows, columns, threads, width)


def _register_b_layout(layout: ProducerLayout) -> bool:
    """Four contiguous half-pairs exchange their two register/lane bits.

    m16n8k16 B ownership has four K values in each lane; retaining two K atoms
    makes four packed words. A 32-column, eight-value producer therefore needs
    a 4-by-4 transpose inside each four-lane group. Larger N tiles repeat that
    same transform. Other producer layouts use the complete shared conversion.
    """
    return (
        layout.threads == 64
        and layout.width == 8
        and layout.columns == 4 * layout.width
        and layout.rows % 16 == 0
    )


def _packet_plan(
    statements: Sequence[ast.Assign],
    value: ast.expr,
    *,
    coordinate: str,
    width: int,
    tensors: Mapping[str, CopyTensorFacts],
    aligned_names: Mapping[str, int],
) -> ContiguousCopy | None:
    """Use the existing address/mask analysis for a smaller typed packet.

    The general copy helper fixes packets at 16 bytes. A fragment output can
    own only four half values, so this local path also admits an 8-byte packet.
    It retains the same typed affine/predicate analysis and scalar tail path.
    """
    if width == 1:
        return None
    targets = {
        statement.targets[0].id
        for statement in statements
        if len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name)
    }
    if len(targets) != len(statements) or coordinate in targets:
        return None
    boundaries = (
        set().union(
            *(_read_names(statement.value) for statement in statements),
            _read_names(value),
        )
        - targets
    )
    recipe = build_recipe(value, statements, boundaries)
    if recipe is None:
        return None
    names = _Names(boundaries)
    copied, result = recipe.emit({}, names)
    definitions = {
        cast("ast.Name", statement.targets[0]).id: statement.value
        for statement in copied
    }
    analysis = _Analysis(definitions, coordinate, tensors, aligned_names)
    terminal = analysis.resolve(result)
    predicates: list[ast.expr] = []
    casts: list[str] = []
    positive_zero = True
    while True:
        terminal = analysis.resolve(terminal)
        if isinstance(terminal, ast.IfExp):
            zero = analysis.resolve(terminal.orelse)
            if isinstance(zero, ast.Call) and ast.unparse(zero.func) in (
                _HALF_TYPES | {"cutlass.Float32"}
            ):
                if len(zero.args) != 1 or zero.keywords:
                    return None
                casts.append(ast.unparse(zero.func))
                zero = analysis.resolve(zero.args[0])
            if (
                not isinstance(zero, ast.Constant)
                or type(zero.value) not in (int, float)
                or zero.value != 0
            ):
                return None
            positive_zero = positive_zero and math.copysign(1.0, zero.value) > 0
            predicates.append(terminal.test)
            terminal = terminal.body
        elif (
            isinstance(terminal, ast.Call)
            and ast.unparse(terminal.func) in _HALF_TYPES | {"cutlass.Float32"}
            and len(terminal.args) == 1
            and not terminal.keywords
        ):
            casts.append(ast.unparse(terminal.func))
            terminal = terminal.args[0]
        else:
            break
    if not (
        isinstance(terminal, ast.Call)
        and isinstance(terminal.func, ast.Attribute)
        and terminal.func.attr == "load"
        and not terminal.args
        and not terminal.keywords
    ):
        return None
    pointer = analysis.resolve(terminal.func.value)
    sources = [
        node.value.id
        for node in ast.walk(pointer)
        if isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
    ]
    if len(sources) != 1 or (facts := tensors.get(sources[0])) is None:
        return None
    if facts.dtype not in _HALF_TYPES | {"cutlass.Float32"}:
        return None
    packet_bytes = width * _TYPE_BYTES[facts.dtype]
    if (
        packet_bytes not in (8, 16)
        or facts.alignment_bytes < packet_bytes
        or any(dtype != facts.dtype for dtype in casts)
    ):
        return None
    analysis.width = width
    if (
        aligned_names.get(coordinate, 1) % width
        or analysis.delta(pointer) != 1
        or analysis.alignment(pointer) != width
    ):
        return None
    predicate = (
        ast.Constant(True)
        if not predicates
        else predicates[0]
        if len(predicates) == 1
        else ast.BoolOp(ast.And(), predicates)
    )
    if not analysis.expansion_fits((pointer, predicate)) or not analysis.interval(
        predicate
    ):
        return None
    return ContiguousCopy(
        sources[0],
        coordinate,
        tuple(copied),
        result,
        analysis.expand_variant(pointer),
        analysis.expand_variant(predicate),
        frozenset(
            name
            for name, expression in definitions.items()
            if not analysis.depends(expression)
        ),
        False,
        width=width,
        _uniform_positive_zero=positive_zero
        and analysis.packet_uniform_predicate(predicate),
    )


def _packet_alignment(statements: list[ast.stmt], alignment: int) -> list[ast.stmt]:
    """The reused copy emitter otherwise declares its fixed 16-byte alignment."""
    for statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "cute.make_ptr":
                for keyword in node.keywords:
                    if keyword.arg == "assumed_align":
                        assert _integer(keyword.value) == 16
                        keyword.value = ast.Constant(alignment)
    return statements


@dataclass(frozen=True)
class _Transport:
    target: str
    dtype: str
    axes: tuple[str, ...]
    recipe: ScalarRecipe
    dependencies: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _AxisProjection:
    """A quotient of the exact blocked producer coordinates, within one site.

    A row is constant within a packet. A column repeats across packet slots
    only when the thread stride is a whole number of rows. No source address,
    predicate, typed offset or scalar expression is simplified by this proof.
    """

    producer: ProducerLayout
    axis: str
    row: bool

    @property
    def size(self) -> int:
        return self.producer.slots if self.row else self.producer.width

    def index(self, element: str) -> ast.expr:
        operator = "//" if self.row else "%"
        return _expr(f"({element}) {operator} {self.producer.width}")

    def representative(self, element: str) -> str:
        return f"({element}) * {self.producer.width}" if self.row else element


@dataclass(frozen=True)
class _RegisterCache:
    name: str
    dtype: str
    projection: _AxisProjection | None = None

    def load(self, element: str) -> ast.expr:
        index = (
            self.projection.index(element)
            if self.projection is not None
            else _expr(element)
        )
        return ast.Subscript(_expr(self.name), index, ast.Load())


def _axis_projection(
    producer: ProducerLayout,
    axes: tuple[str, ...],
    *,
    row: str,
    column: str,
) -> _AxisProjection | None:
    # Generated coordinates are Int32. Prove the complete flat expression,
    # not just its final modulo, before identifying two consumer coordinates.
    if (
        producer.rows <= 0
        or producer.columns <= 0
        or producer.threads <= 0
        or producer.width <= 0
        or producer.columns % producer.width
        or producer.rows * producer.columns % (producer.threads * producer.width)
        or producer.rows * producer.columns > (1 << 31) - 1
    ):
        return None
    if axes == (row,):
        return _AxisProjection(producer, row, True)
    if axes == (column,) and producer.threads * producer.width % producer.columns == 0:
        return _AxisProjection(producer, column, False)
    return None


def _transport_batches(
    transfers: Sequence[_Transport],
    *,
    row: str,
    rows: int,
    columns: int,
    arena_bytes: int,
) -> tuple[tuple[tuple[_Transport, int], ...], ...]:
    """Partition independent read-only publications into disjoint arena ranges.

    Do not grow the existing arena just to combine phases. A dependency or a
    capacity limit starts a new phase, with its original retirement boundary.
    Each batch still has a publication barrier and a post-load reuse barrier.
    """
    sizes = [
        (
            rows * columns
            if len(item.axes) == 2
            else rows
            if item.axes == (row,)
            else columns
        )
        * _TYPE_BYTES[item.dtype]
        for item in transfers
    ]
    capacity = max((arena_bytes, *sizes))
    if capacity > 65536 or any(size <= 0 for size in sizes):
        _reject("transport phase exceeds the bounded shared arena")
    result: list[tuple[tuple[_Transport, int], ...]] = []
    current: list[tuple[_Transport, int]] = []
    end = 0
    for transfer, size in zip(transfers, sizes, strict=True):
        alignment = _TYPE_BYTES[transfer.dtype]
        offset = (end + alignment - 1) // alignment * alignment
        dependent = any(
            prior.target in transfer.dependencies
            or transfer.target in prior.dependencies
            for prior, _ in current
        )
        if current and (dependent or offset + size > capacity):
            result.append(tuple(current))
            current, offset = [], 0
        current.append((transfer, offset))
        end = offset + size
    if current:
        result.append(tuple(current))
    return tuple(result)


@dataclass(frozen=True)
class _TypedAxisValue:
    dtype: str
    axes: frozenset[str]


def _typed_axis_value(
    value: ast.expr, known: Mapping[str, _TypedAxisValue]
) -> _TypedAxisValue | None:
    """Follow exact casts/aliases and rounded multiplies, without reassociation.

    Unrounded FP arithmetic, loads, coordinate calculations, unknown calls,
    subscripts and conditional materialization are deliberately not hoisted.
    The boundary values are already captured, typed read-only transport loads.
    """
    if isinstance(value, ast.Name):
        return known.get(value.id)
    if isinstance(value, ast.Constant) and type(value.value) in (bool, int, float):
        return _TypedAxisValue(type(value.value).__name__, frozenset())
    if not isinstance(value, ast.Call):
        return None
    if (
        ast.unparse(value.func) in _TYPE_BYTES
        and len(value.args) == 1
        and not value.keywords
    ):
        argument = _typed_axis_value(value.args[0], known)
        return (
            None
            if argument is None
            else _TypedAxisValue(ast.unparse(value.func), argument.axes)
        )
    if _rounded_multiply(value):
        operands = cast("ast.Tuple", value.args[0]).elts
        left, right = (_typed_axis_value(operand, known) for operand in operands)
        if (
            left is not None
            and right is not None
            and left.dtype == right.dtype == "cutlass.Float32"
        ):
            return _TypedAxisValue("cutlass.Float32", left.axes | right.axes)
    return None


def _transports(
    statements: Sequence[ast.stmt],
    *,
    row: str,
    column: str,
    tensors: Mapping[str, CopyTensorFacts],
    forbidden: frozenset[str],
    matrices: bool,
) -> tuple[_Transport, ...]:
    # Do not flatten conditional materialization or move a load across it.
    if not all(
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        for statement in statements
    ):
        return ()
    assignments = cast("Sequence[ast.Assign]", statements)
    definitions = {
        cast("ast.Name", statement.targets[0]).id: statement.value
        for statement in assignments
    }
    if len(definitions) != len(assignments):
        return ()
    preceding: set[str] = set()
    for target, value in definitions.items():
        if (_read_names(value) & definitions.keys()) - preceding:
            return ()
        preceding.add(target)
    dependencies = _Dependencies(
        definitions, {name: facts.dtype for name, facts in tensors.items()}
    )
    boundaries = (
        set().union(*(_read_names(statement.value) for statement in assignments))
        - definitions.keys()
    )
    result = []
    for index, statement in enumerate(assignments):
        # A value alias is not a second load. Transport only the actual typed
        # load definition, retaining all later casts and arithmetic in place.
        if not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            for node in ast.walk(statement.value)
        ):
            continue
        leaves = dependencies.leaves(statement.value)
        axes = tuple(axis for axis in (row, column) if axis in leaves)
        dtype = dependencies.dtype(statement.value)
        if (
            not axes
            or len(axes) == 2
            and not matrices
            or leaves & forbidden
            or dtype not in _TYPE_BYTES
            or not dependencies.load_sources(statement.value)
        ):
            continue
        target = cast("ast.Name", statement.targets[0]).id
        recipe = build_recipe(_expr(target), assignments[: index + 1], boundaries)
        if recipe is not None:
            assert dtype is not None
            result.append(
                _Transport(
                    target,
                    dtype,
                    axes,
                    recipe,
                    frozenset(
                        cast("ast.Name", assignments[position].targets[0]).id
                        for position in recipe.source_statement_indices
                        if position != index
                    ),
                )
            )
    return tuple(result)


class _Builder:
    def __init__(
        self,
        names: _Names,
        tid: str,
        arena: str,
        tensor_facts: Mapping[str, CopyTensorFacts],
        fast_math: bool,
        thread_count: int,
        target_device_capability: tuple[int, int] | None = None,
    ) -> None:
        self.names = names
        self.tid = tid
        self.arena = arena
        self.tensors = tensor_facts
        self.fast_math = fast_math
        self.thread_count = thread_count
        self.target_device_capability = target_device_capability
        self.shared_bytes = 0
        self.shared_views: dict[str, tuple[_SharedLayout | None, str, int]] = {}
        self.slice_transports = 0
        self.matrix_transports = 0

    def materialize(
        self,
        recipe: ScalarEmitter | ScalarRecipe,
        replacements: Mapping[str, ast.expr],
        *,
        protect_products: bool = True,
    ) -> tuple[list[ast.stmt], ast.expr]:
        statements, value = recipe.emit(replacements, self.names)
        if (
            protect_products
            and not self.fast_math
            and all(isinstance(statement, ast.Assign) for statement in statements)
        ):
            statements, value = preserve_fp32_multiply_rounding(
                cast("list[ast.Assign]", statements), value
            )
        return list(statements), value

    def view(
        self,
        name: str,
        dtype: str,
        shape: str,
        *,
        offset: int = 0,
        layout: _SharedLayout | None = None,
    ) -> list[ast.stmt]:
        self.shared_views[name] = (layout, dtype, offset)
        layout_source = (
            layout.source() if layout is not None else f"cute.make_layout({shape})"
        )
        return _statements(
            f"{name} = cute.make_tensor(cute.recast_ptr({self.arena} + {offset}, "
            f"dtype={dtype}), {layout_source})"
        )

    def shared_packet(
        self, destination: str, layout: ProducerLayout, dtype: str, *, transpose: bool
    ) -> bool:
        view = self.shared_views.get(destination)
        if view is None:
            return False
        shared_layout, shared_dtype, byte_offset = view
        packet_bytes = layout.width * _TYPE_BYTES[dtype]
        return (
            shared_layout is not None
            and shared_dtype == dtype
            and shared_layout.contiguous_packets(layout, transpose=transpose)
            and byte_offset >= 0
            and byte_offset % packet_bytes == 0
            and _SHARED_ALIGNMENT_BYTES % packet_bytes == 0
        )

    def packet_loop(
        self,
        layout: ProducerLayout,
        statements: Sequence[ast.stmt],
        value: ast.expr,
        *,
        row: str,
        column: str,
        dtype: str,
        destination: str,
        transpose: bool = False,
        registers: bool = False,
        caches: Mapping[str, _RegisterCache] | None = None,
        aligned_names: Mapping[str, int] | None = None,
    ) -> ast.For:
        """Replay the original recipe at one complete producer partition."""
        assert all(isinstance(statement, ast.Assign) for statement in statements)
        assignments = cast("Sequence[ast.Assign]", statements)
        slot, lane = self.names("packet"), self.names("packet_lane")
        caches = {} if caches is None else caches
        varying = {column}
        common: list[ast.stmt] = []
        scalar: list[ast.stmt] = []
        substitutions = {column: _expr(f"{column} + {lane}")}
        element = f"{slot} * {layout.width} + {lane}"
        for index, statement in enumerate(assignments):
            target = cast("ast.Name", statement.targets[0]).id
            if target in caches:
                cache = caches[target]
                projection = cache.projection
                if projection is not None and (
                    projection.producer != layout
                    or projection.axis not in (row, column)
                ):
                    _reject("register cache has a different producer coordinate proof")
                invariant = projection is not None and projection.row
                if not invariant:
                    varying.add(target)
                (common if invariant else scalar).append(
                    ast.Assign(
                        [ast.Name(target, ast.Store())],
                        cache.load(
                            f"{slot} * {layout.width}" if invariant else element
                        ),
                    )
                )
                continue
            if not (_read_names(statement.value) & varying):
                common.append(_clone(statement))
                continue
            varying.add(target)
            has_load = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
                for node in ast.walk(statement.value)
            )
            plan = (
                _packet_plan(
                    assignments[: index + 1],
                    _expr(target),
                    coordinate=column,
                    width=layout.width,
                    tensors=self.tensors,
                    aligned_names={
                        **({} if aligned_names is None else aligned_names),
                        row: 1,
                        column: layout.width,
                    },
                )
                if has_load and not (_read_names(statement.value) & caches.keys())
                else None
            )
            if plan is not None:
                loaded, source = plan.emit_to_registers(self.names)
                common.extend(
                    _packet_alignment(
                        loaded,
                        plan.width
                        * _TYPE_BYTES[self.tensors[plan.source_tensor].dtype],
                    )
                )
                scalar.extend(_statements(f"{target} = {source}[{lane}]"))
            else:
                scalar.append(cast("ast.stmt", _replace(statement, substitutions)))
        converted = ast.Call(_expr(dtype), [_clone(value)], [])
        stored = cast("ast.expr", _replace(converted, substitutions))
        publication: list[ast.stmt] = []
        packet_values: str | None = None
        if not registers and self.shared_packet(
            destination, layout, dtype, transpose=transpose
        ):
            packet_values = self.names("shared_packet_values")
            packet_shared = self.names("shared_packet_destination")
            common.extend(
                _statements(
                    f"{packet_values} = cute.make_rmem_tensor(({layout.width},), {dtype})"
                )
            )
            coordinate = f"({column}, {row})" if transpose else f"({row}, {column})"
            # Do not reconstruct or align() this pointer: the SDK can move
            # the composed layout's swizzle into the iterator. crd2idx and the
            # original iterator apply it exactly once in either representation.
            publication.extend(
                _statements(
                    f"{packet_shared} = cute.make_tensor({destination}.iterator + "
                    f"cute.assume(cute.crd2idx({coordinate}, {destination}.layout), "
                    f"divby={layout.width}), "
                    f"cute.make_layout(({layout.width},), stride=(1,)))\n"
                    f"cute.autovec_copy({packet_values}, {packet_shared})"
                )
            )
        indices = (
            _expr(lane)
            if packet_values is not None
            else _expr(element)
            if registers
            else _expr(f"({column} + {lane}, {row})")
            if transpose
            else _expr(f"({row}, {column} + {lane})")
        )
        scalar.append(
            ast.Assign(
                [
                    ast.Subscript(
                        _expr(packet_values or destination), indices, ast.Store()
                    )
                ],
                stored,
            )
        )
        m, n = layout.coordinates(self.tid, f"{slot} * {layout.width}")
        return ast.For(
            ast.Name(slot, ast.Store()),
            _expr(f"cutlass.range_constexpr({layout.slots})"),
            [
                *_statements(
                    f"{row} = cutlass.Int32({ast.unparse(m)})\n"
                    f"{column} = cutlass.Int32({ast.unparse(n)})"
                ),
                *common,
                rounded_recipe_loop(
                    scalar,
                    lane=lane,
                    width=layout.width,
                    register_outputs=(
                        frozenset({packet_values})
                        if packet_values is not None
                        else frozenset()
                    ),
                    fresh_name=self.names,
                    target_device_capability=self.target_device_capability,
                ),
                *publication,
            ],
            [],
        )

    def transport(
        self,
        transfers: Sequence[_Transport],
        *,
        row: str,
        column: str,
        rows: int,
        columns: int,
        consumer_size: str,
        coordinates: Callable[[str], tuple[ast.expr, ast.expr]],
        aligned_names: Mapping[str, int],
        producer_layout: ProducerLayout | None = None,
    ) -> tuple[list[ast.stmt], dict[str, _RegisterCache]]:
        """Publish a disjoint batch, load every consumer value, then retire.

        Both barriers remain mandatory: the first publishes every member of
        the batch, and the second follows every shared load before arena reuse.
        Opaque SDK fragment coordinates keep their complete original cache.
        """
        result: list[ast.stmt] = []
        caches: dict[str, _RegisterCache] = {}
        batches = _transport_batches(
            transfers,
            row=row,
            rows=rows,
            columns=columns,
            arena_bytes=self.shared_bytes,
        )
        for batch in batches:
            reads: list[ast.stmt] = []
            for transfer, offset in batch:
                view = self.names("transfer_shared")
                cache = self.names("transfer_values")
                element = self.names("transfer_element")
                matrix = len(transfer.axes) == 2
                extent = rows if transfer.axes == (row,) else columns
                size = rows * columns if matrix else extent
                self.shared_bytes = max(
                    self.shared_bytes, offset + size * _TYPE_BYTES[transfer.dtype]
                )
                self.matrix_transports += int(matrix)
                self.slice_transports += int(not matrix)
                shape = (
                    f"({rows}, {columns}), stride=({columns}, 1)"
                    if matrix
                    else f"({extent},)"
                )
                result.extend(self.view(view, transfer.dtype, shape, offset=offset))
                statements, value = self.materialize(transfer.recipe, {})
                if matrix:
                    result.append(
                        self.packet_loop(
                            _layout(rows, columns, self.thread_count),
                            statements,
                            value,
                            row=row,
                            column=column,
                            dtype=transfer.dtype,
                            destination=view,
                            aligned_names=aligned_names,
                        )
                    )
                else:
                    slot, coordinate = self.names("slice_slot"), transfer.axes[0]
                    producer = _statements(
                        f"for {slot} in cutlass.range_constexpr({(extent + self.thread_count - 1) // self.thread_count}):\n"
                        f"    {coordinate} = cutlass.Int32({self.tid} + {slot} * {self.thread_count})\n"
                        f"    if {coordinate} < {extent}:\n"
                        f"        pass"
                    )[0]
                    assert isinstance(producer, ast.For)
                    branch = cast("ast.If", producer.body[-1])
                    branch.body = [
                        *statements,
                        ast.Assign(
                            [
                                ast.Subscript(
                                    _expr(view), _expr(coordinate), ast.Store()
                                )
                            ],
                            value,
                        ),
                    ]
                    result.append(producer)
                projection = None
                if (
                    producer_layout is not None
                    and (producer_layout.rows, producer_layout.columns)
                    == (rows, columns)
                    and producer_layout.threads == self.thread_count
                    and consumer_size == str(producer_layout.values_per_thread)
                ):
                    projection = _axis_projection(
                        producer_layout, transfer.axes, row=row, column=column
                    )
                caches[transfer.target] = _RegisterCache(
                    cache, transfer.dtype, projection
                )
                count = consumer_size if projection is None else str(projection.size)
                reads.extend(
                    _statements(
                        f"{cache} = cute.make_rmem_tensor(({count},), {transfer.dtype})"
                    )
                )
                m, n = (
                    coordinates(element)
                    if projection is None
                    else projection.producer.coordinates(
                        self.tid, projection.representative(element)
                    )
                )
                index = (
                    ast.Tuple([m, n], ast.Load())
                    if matrix
                    else m
                    if transfer.axes == (row,)
                    else n
                )
                reads.append(
                    ast.For(
                        ast.Name(element, ast.Store()),
                        _expr(f"cutlass.range_constexpr({count})"),
                        [
                            ast.Assign(
                                [
                                    ast.Subscript(
                                        _expr(cache), _expr(element), ast.Store()
                                    )
                                ],
                                ast.Subscript(_expr(view), index, ast.Load()),
                            )
                        ],
                        [],
                    )
                )
            result.extend(_statements("cute.arch.sync_threads()"))
            result.extend(reads)
            result.extend(_statements("cute.arch.sync_threads()"))
        return result, caches

    def cache_axis_recipes(
        self,
        layout: ProducerLayout,
        statements: Sequence[ast.stmt],
        value: ast.expr,
        *,
        column: str,
        caches: Mapping[str, _RegisterCache],
    ) -> tuple[list[ast.stmt], list[ast.stmt], dict[str, _RegisterCache]]:
        """Reuse exact rounded column DAGs across repeated row packets.

        Row-only DAGs already run once per row packet in ``packet_loop``.
        Here only existing explicit rounding boundaries become register caches;
        unrounded arithmetic retains its original fusion/materialization scope.
        All proof and cache state belongs to this one recipe and K iteration.
        """
        unchanged = ([], list(statements), dict(caches))
        if layout.slots <= 1 or not all(
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            for statement in statements
        ):
            return unchanged
        assignments = cast("Sequence[ast.Assign]", statements)
        definitions = {
            cast("ast.Name", statement.targets[0]).id: statement.value
            for statement in assignments
        }
        if len(definitions) != len(assignments):
            return unchanged
        known: dict[str, _TypedAxisValue] = {}
        roots: list[str] = []
        for target, expression in definitions.items():
            if target in caches:
                cache = caches[target]
                projection = cache.projection
                if projection is not None and projection.producer == layout:
                    known[target] = _TypedAxisValue(
                        cache.dtype, frozenset({projection.axis})
                    )
                continue
            typed = _typed_axis_value(expression, known)
            if typed is not None:
                known[target] = typed
                if typed.axes == {column} and _rounded_multiply(expression):
                    roots.append(target)
        if not roots:
            return unchanged
        # Only precise earlier SSA definitions may be replayed. Unknown names,
        # even if uniform in a larger scope, did not enter the typed proof.
        needed = set(roots)
        for target, expression in reversed(definitions.items()):
            if target in needed and target not in caches:
                needed.update(_read_names(expression) & definitions.keys())
        element = self.names("axis_element")
        projection = _AxisProjection(layout, column, False)
        replacements: dict[str, ast.expr] = {}
        setup: list[ast.stmt] = []
        body: list[ast.stmt] = []
        result_caches = dict(caches)
        for target in roots:
            cache = self.names("axis_values")
            setup.extend(
                _statements(
                    f"{cache} = cute.make_rmem_tensor(({layout.width},), cutlass.Float32)"
                )
            )
            result_caches[target] = _RegisterCache(cache, "cutlass.Float32", projection)
        for target, expression in definitions.items():
            if target not in needed:
                continue
            if target in caches:
                replacements[target] = caches[target].load(element)
                continue
            name = self.names("axis_value")
            body.append(
                ast.Assign(
                    [ast.Name(name, ast.Store())],
                    cast("ast.expr", _replace(expression, replacements)),
                )
            )
            replacements[target] = _expr(name)
            if target in roots:
                body.append(
                    ast.Assign(
                        [
                            ast.Subscript(
                                _expr(result_caches[target].name),
                                _expr(element),
                                ast.Store(),
                            )
                        ],
                        _expr(name),
                    )
                )
        setup.append(
            rounded_recipe_loop(
                body,
                lane=element,
                width=layout.width,
                register_outputs=frozenset(
                    result_caches[target].name for target in roots
                ),
                fresh_name=self.names,
                target_device_capability=self.target_device_capability,
            )
        )
        # Remove only typed, pure ancestors made dead by the new cache. Other
        # definitions retain their evaluation scope and original expression.
        live = _read_names(value)
        replay: list[ast.stmt] = []
        for statement in reversed(assignments):
            target = cast("ast.Name", statement.targets[0]).id
            if target not in live and target in known:
                continue
            replay.append(statement)
            live.discard(target)
            if target not in result_caches:
                live.update(_read_names(statement.value))
        replay.reverse()
        return setup, replay, result_caches

    def fragment_recipe(
        self,
        recipe: ScalarEmitter | ScalarRecipe,
        *,
        m_index: str,
        n_index: str,
        m_offset: str,
        n_offset: str,
        bm: int,
        bn: int,
        accumulator: str,
        coordinates: str,
        frontier: str | None,
        destination: str,
        dtype: str,
        predicate_recipe: ScalarRecipe | None = None,
        protect_products: bool = True,
    ) -> list[ast.stmt]:
        m, n, element = (
            self.names("fragment_m"),
            self.names("fragment_n"),
            self.names("fragment_element"),
        )
        replacements = {
            m_index: _expr(f"{m_offset} + {m}"),
            n_index: _expr(f"{n_offset} + {n}"),
        }
        if frontier is not None:
            replacements[frontier] = _expr(f"{accumulator}[{element}]")
        statements, value = self.materialize(
            recipe, replacements, protect_products=protect_products
        )
        transfers = _transports(
            statements,
            row=m,
            column=n,
            tensors=self.tensors,
            forbidden=frozenset({accumulator, element}),
            matrices=True,
        )
        prologue, caches = self.transport(
            transfers,
            row=m,
            column=n,
            rows=bm,
            columns=bn,
            consumer_size=f"cute.size({accumulator})",
            coordinates=lambda index: (
                _expr(f"{coordinates}[{index}][0]"),
                _expr(f"{coordinates}[{index}][1]"),
            ),
            aligned_names={m_offset: bm, n_offset: bn},
        )
        replay: list[ast.stmt] = []
        for statement in statements:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id in caches
            ):
                replay.append(
                    ast.Assign(
                        [_clone(statement.targets[0])],
                        caches[statement.targets[0].id].load(element),
                    )
                )
            else:
                replay.append(statement)
        target = (
            _expr(f"{destination}[{element}]")
            if predicate_recipe is None
            else _expr(f"{destination}[{m}, {n}]")
        )
        assert isinstance(target, ast.Subscript)
        target.ctx = ast.Store()
        stored = ast.Call(_expr(dtype), [value], [])
        if predicate_recipe is not None:
            predicate_setup, predicate = self.materialize(
                predicate_recipe, replacements
            )
            replay.extend(predicate_setup)
            stored = ast.IfExp(predicate, stored, _expr(f"{dtype}(0)"))
        replay.append(ast.Assign([target], stored))
        prologue.append(
            ast.For(
                ast.Name(element, ast.Store()),
                _expr(f"cutlass.range_constexpr(cute.size({accumulator}))"),
                [
                    *_statements(
                        f"{m} = cutlass.Int32({coordinates}[{element}][0])\n"
                        f"{n} = cutlass.Int32({coordinates}[{element}][1])"
                    ),
                    *replay,
                ],
                [],
            )
        )
        return prologue

    def register_b(
        self,
        layout: ProducerLayout,
        statements: Sequence[ast.stmt],
        value: ast.expr,
        *,
        row: str,
        column: str,
        dtype: str,
        fragment: str,
        thread: str,
        caches: Mapping[str, _RegisterCache],
        aligned_names: Mapping[str, int],
    ) -> list[ast.stmt]:
        """Retain complete B packets and transpose packed words within warps."""
        assert _register_b_layout(layout)
        values, packed = self.names("b_values"), self.names("b_packed")
        result = _statements(
            f"{values} = cute.make_rmem_tensor(({layout.values_per_thread},), {dtype})"
        )
        result.append(
            self.packet_loop(
                layout,
                statements,
                value,
                row=row,
                column=column,
                dtype=dtype,
                destination=values,
                registers=True,
                caches=caches,
                aligned_names=aligned_names,
            )
        )
        result.extend(
            _statements(f"{packed} = cute.recast_tensor({values}, cutlass.Uint32)")
        )
        lane = self.names("lane")
        result.extend(_statements(f"{lane} = {self.tid} % 32"))
        for bit in (1, 2):
            swapped = self.names("b_transpose")
            group, register, peer = (
                self.names("b_group"),
                self.names("b_word"),
                self.names("b_peer"),
            )
            result.extend(
                _statements(f"""
{swapped} = cute.make_rmem_tensor(({layout.values_per_thread // 2},), cutlass.Uint32)
for {group} in cutlass.range_constexpr({layout.rows // 16}):
    for {register} in cutlass.range_constexpr(4):
        {peer} = cute.arch.shuffle_sync({packed}[{group} * 4 + ({register} ^ {bit})], {lane} ^ {bit}, mask=-1, mask_and_clamp=31)
        {swapped}[{group} * 4 + {register}] = {packed}[{group} * 4 + {register}] if ({lane} & {bit}) == ({register} & {bit}) else {peer}
""")
            )
            packed = swapped
        half_values, coordinates, element = (
            self.names("b_half_values"),
            self.names("b_coordinates"),
            self.names("b_element"),
        )
        result.extend(
            _statements(f"""
{half_values} = cute.recast_tensor({packed}, {dtype})
{coordinates} = {thread}.partition_B(cute.make_identity_tensor(({layout.rows}, {layout.columns})))
for {element} in cutlass.range_constexpr(cute.size({fragment})):
    {fragment}[{element}] = {half_values}[{coordinates}[{element}][0] // 16 * 8 + {coordinates}[{element}][1] % 2 + 2 * ({coordinates}[{element}][1] // 8)]
""")
        )
        return result

    def output(
        self,
        site: RegisterChainSite,
        epilogue: RegisterChainEpilogue,
        accumulator: str,
        coordinates: str,
        *,
        terminal_arena_dead: bool = False,
    ) -> list[ast.stmt]:
        output = self.names("output_shared")
        self.shared_bytes = max(
            self.shared_bytes, site.bm * site.bn * _TYPE_BYTES[epilogue.dtype]
        )
        result = self.view(
            output,
            epilogue.dtype,
            f"({site.bm}, {site.bn}), stride=({site.bn}, 1)",
        )
        result.extend(
            self.fragment_recipe(
                epilogue.value_recipe,
                m_index=epilogue.m_index,
                n_index=epilogue.n_index,
                m_offset=site.m_offset,
                n_offset=site.n_offset,
                bm=site.bm,
                bn=site.bn,
                accumulator=accumulator,
                coordinates=coordinates,
                frontier=epilogue.accumulator_name,
                destination=output,
                dtype=epilogue.dtype,
                predicate_recipe=epilogue.predicate_recipe,
                # The original scalar epilogue supplies its fusion/cast policy.
                # Do not invent a new BF16 FMA or protect a previously fusible
                # product merely because the value is now in a fragment.
                protect_products=False,
            )
        )
        result.extend(_statements("cute.arch.sync_threads()"))
        layout = _layout(site.bm, site.bn, self.thread_count)
        slot, lane = self.names("output_packet"), self.names("output_lane")
        m, n = self.names("output_m"), self.names("output_n")
        values, shared = self.names("output_values"), self.names("output_source")
        replacements = {
            epilogue.m_index: _expr(f"{site.m_offset} + {m}"),
            epilogue.n_index: _expr(f"{site.n_offset} + {n}"),
        }
        pointer_setup, pointer = self.materialize(epilogue.pointer_recipe, replacements)
        mask_setup, predicate = self.materialize(
            epilogue.predicate_recipe, replacements
        )
        scalar_replacements = {n: _expr(f"{n} + {lane}")}
        scalar_pointer = cast("ast.expr", _replace(pointer, scalar_replacements))
        scalar_mask = cast("ast.expr", _replace(predicate, scalar_replacements))
        # Address definitions execute only under the original store predicate.
        # Their separate recipes cannot reference an accumulator after repartition.
        scalar_body = [
            *(
                cast("ast.stmt", _replace(statement, scalar_replacements))
                for statement in mask_setup
            ),
            ast.If(
                scalar_mask,
                [
                    *(
                        cast("ast.stmt", _replace(statement, scalar_replacements))
                        for statement in pointer_setup
                    ),
                    ast.Expr(
                        ast.Call(
                            ast.Attribute(scalar_pointer, "store", ast.Load()),
                            [_expr(f"{values}[{lane}]")],
                            [],
                        )
                    ),
                ],
                [],
            ),
        ]
        scalar = ast.For(
            ast.Name(lane, ast.Store()),
            _expr(f"cutlass.range_constexpr({layout.width})"),
            scalar_body,
            [],
        )
        stores: list[ast.stmt] = [scalar]
        if epilogue.output_facts is not None:
            synthetic_load = ast.IfExp(
                predicate,
                ast.Call(ast.Attribute(pointer, "load", ast.Load()), [], []),
                _expr(f"{epilogue.dtype}(0)"),
            )
            plan = _packet_plan(
                cast("list[ast.Assign]", [*pointer_setup, *mask_setup]),
                synthetic_load,
                coordinate=n,
                width=layout.width,
                tensors={**self.tensors, epilogue.output_tensor: epilogue.output_facts},
                aligned_names={
                    m: 1,
                    n: layout.width,
                    site.m_offset: site.bm,
                    site.n_offset: site.bn,
                },
            )
            if plan is not None:
                alignment = layout.width * _TYPE_BYTES[epilogue.dtype]
                vector = _packet_alignment(
                    plan.emit_from_registers(values, self.names), alignment
                )
                guards = [
                    f"{epilogue.output_tensor}.iterator.toint() % {alignment} == 0"
                ]
                guards.extend(
                    f"{epilogue.output_tensor}.layout.stride[{axis}] == {stride}"
                    for axis, stride in enumerate(epilogue.output_facts.strides)
                )
                stores = [ast.If(_expr(" and ".join(guards)), vector, [scalar])]
        row, column = layout.coordinates(self.tid, f"{slot} * {layout.width}")
        loop = ast.For(
            ast.Name(slot, ast.Store()),
            _expr(f"cutlass.range_constexpr({layout.slots})"),
            [
                *_statements(
                    f"{m} = cutlass.Int32({ast.unparse(row)})\n"
                    f"{n} = cutlass.Int32({ast.unparse(column)})\n"
                    f"{values} = cute.make_rmem_tensor(({layout.width},), {epilogue.dtype})\n"
                    f"{shared} = cute.make_tensor({output}.iterator + {m} * {site.bn} + {n}, cute.make_layout(({layout.width},), stride=(1,)))\n"
                    f"cute.autovec_copy({shared}, {values})"
                ),
                *stores,
            ],
            [],
        )
        result.append(loop)
        if not terminal_arena_dead:
            result.extend(_statements("cute.arch.sync_threads()"))
        return result


def plan_register_chain(
    sites: Sequence[RegisterChainSite],
    epilogue: RegisterChainEpilogue,
    *,
    thread_count: int,
    tensor_facts: Mapping[str, CopyTensorFacts],
    fast_math: bool = False,
    terminal_arena_dead: bool = False,
    constexpr_values: Mapping[str, int] | None = None,
    target_device_capability: tuple[int, int] | None = None,
) -> RegisterChainPlan:
    """Plan the whole selected chain before any integration-side mutation.

    The caller has already proved uniform placement, read-only recipe inputs,
    disjoint writes, original loop/live-out boundaries and same-cell frontiers.
    Unsupported local geometry, full-K domain or frontier rejects the entire
    selected configuration; this function does not silently lower part of it.
    ``terminal_arena_dead`` additionally requires a whole-device-region proof:
    no later shared alias/use, escape or enclosing backedge. Otherwise the final
    shared-load retirement barrier remains, including for standalone planning.
    """
    sites = tuple(sites)
    _validate_sites(sites, epilogue, thread_count)
    constexpr_values = {} if constexpr_values is None else constexpr_values
    occupied = set(tensor_facts) | {epilogue.output_tensor, epilogue.accumulator_name}
    occupied.update(constexpr_values)
    for site in sites:
        occupied.update(
            (
                site.m_index,
                site.n_index,
                site.k_index,
                site.m_offset,
                site.n_offset,
                site.k_offset,
            )
        )
        for recipe in (site.a_recipe, site.b_recipe, site.seed_recipe):
            if recipe is not None:
                occupied.update(recipe.boundary_names)
    occupied.update(epilogue.pointer_recipe.boundary_names)
    occupied.update(epilogue.predicate_recipe.boundary_names)
    occupied.update(epilogue.value_recipe.boundary_names)
    names = _Names(occupied)
    tid, arena, accumulator = names("tid"), names("arena"), names("accumulator")
    builder = _Builder(
        names,
        tid,
        arena,
        tensor_facts,
        fast_math,
        thread_count,
        target_device_capability,
    )
    # Address/mask replay occurs after the C fragment is redistributed. Reject
    # data-dependent addresses/masks rather than using a stale scalar frontier.
    for recipe in (epilogue.pointer_recipe, epilogue.predicate_recipe):
        statements, value = recipe.emit({}, names)
        if epilogue.accumulator_name in set().union(
            *(_read_names(statement) for statement in statements), _read_names(value)
        ):
            _reject("output address or predicate depends on the accumulator")
    result: list[ast.stmt] = []
    layouts: list[tuple[ProducerLayout, ProducerLayout]] = []
    b_majors: list[bool] = []
    register_b_layouts: list[bool] = []
    final_coordinates = ""
    for position, site in enumerate(sites):
        m, n, k = names("m"), names("n"), names("k")
        replacements = {
            site.m_index: _expr(f"{site.m_offset} + {m}"),
            site.n_index: _expr(f"{site.n_offset} + {n}"),
            site.k_index: _expr(f"{site.k_offset} + {k}"),
        }
        a_statements, a_value = builder.materialize(site.a_recipe, replacements)
        b_statements, b_value = builder.materialize(site.b_recipe, replacements)
        alignments = {
            **constexpr_values,
            site.m_offset: site.bm,
            site.n_offset: site.bn,
            site.k_offset: site.bk,
            m: 1,
            n: 1,
            k: 1,
        }
        b_k_major = (
            plan_contiguous_copy(
                cast("list[ast.Assign]", b_statements),
                b_value,
                coordinate=k,
                tensors=tensor_facts,
                aligned_names={**alignments, k: 8},
            )
            is not None
        )
        a_layout = _layout(site.bm, site.bk, thread_count)
        b_layout = (
            _layout(site.bn, site.bk, thread_count)
            if b_k_major
            else _layout(site.bk, site.bn, thread_count)
        )
        layouts.append((a_layout, b_layout))
        b_majors.append(b_k_major)
        register_b = b_k_major and _register_b_layout(b_layout)
        register_b_layouts.append(register_b)
        a_bytes = site.bm * site.bk * 2
        b_bytes = site.bn * site.bk * 2
        builder.shared_bytes = max(builder.shared_bytes, a_bytes + b_bytes)
        if builder.shared_bytes > 65536:
            _reject("operand phase exceeds the bounded shared arena")
        # Swizzles match the packet's contiguous major dimension. Every view
        # remains typed; byte offsets are used only for the shared allocation.
        a_swizzle = min(3, (site.bk // 8).bit_length() - 1)
        b_major = site.bk if b_k_major else site.bn
        b_swizzle = min(3, (b_major // 8).bit_length() - 1)
        a_shared, b_shared = names("a_shared"), names("b_shared")
        a_shared_layout = _SharedLayout(
            (site.bm, site.bk), (site.bk, 1), (a_swizzle, 3, 3)
        )
        b_shared_layout = _SharedLayout(
            (site.bn, site.bk),
            (site.bk, 1) if b_k_major else (1, site.bn),
            (b_swizzle, 3, 3),
        )
        result.extend(builder.view(a_shared, site.dtype, "", layout=a_shared_layout))
        result.extend(
            builder.view(
                b_shared, site.dtype, "", offset=a_bytes, layout=b_shared_layout
            )
        )
        # All generated binding names are allocated privately and later renamed
        # together. No name derives from a kernel or input tensor identity.
        mma, thread = names("mma"), names("mma_thread")
        ra, rb = names("a_fragment"), names("b_fragment")
        ca, cb = names("a_copy"), names("b_copy")
        pa, pb = names("a_partition"), names("b_partition")
        da, db = names("a_destination"), names("b_destination")
        coordinates = names("c_coordinates")
        final_coordinates = coordinates
        result.extend(
            _statements(f"""
{mma} = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op({site.dtype}, cutlass.Float32, (16, 8, 16)), atom_layout_mnk=(1, 2, 1))
{thread} = {mma}.get_slice({tid})
{coordinates} = {thread}.partition_C(cute.make_identity_tensor(({site.bm}, {site.bn})))
{ra} = {thread}.make_fragment_A({thread}.partition_shape_A(({site.bm}, {site.bk})))
{rb} = {thread}.make_fragment_B({thread}.partition_shape_B(({site.bn}, {site.bk})))
{ca} = cute.make_tiled_copy_A(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), {site.dtype}), {mma})
{cb} = cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose={not b_k_major}, num_matrices=4), {site.dtype}), {mma})
{pa} = {ca}.get_slice({tid}).partition_S({a_shared})
{pb} = {cb}.get_slice({tid}).partition_S({b_shared})
{da} = {ca}.get_slice({tid}).retile({ra})
{db} = {cb}.get_slice({tid}).retile({rb})
""")
        )
        if position == 0:
            result.extend(
                _statements(
                    f"{accumulator} = cute.make_rmem_tensor({mma}.partition_shape_C(({site.bm}, {site.bn})), cutlass.Float32)"
                )
            )
        if site.seed_recipe is None:
            result.extend(_statements(f"{accumulator}.fill(0.0)"))
        else:
            result.extend(
                builder.fragment_recipe(
                    site.seed_recipe,
                    m_index=site.m_index,
                    n_index=site.n_index,
                    m_offset=site.m_offset,
                    n_offset=site.n_offset,
                    bm=site.bm,
                    bn=site.bn,
                    accumulator=accumulator,
                    coordinates=coordinates,
                    frontier=site.seed_name,
                    destination=accumulator,
                    dtype="cutlass.Float32",
                )
            )
        reduction: list[ast.stmt] = []
        a_transfers = _transports(
            a_statements,
            row=m,
            column=k,
            tensors=tensor_facts,
            forbidden=frozenset(),
            matrices=False,
        )
        b_row, b_column = (n, k) if b_k_major else (k, n)
        b_transfers = _transports(
            b_statements,
            row=b_row,
            column=b_column,
            tensors=tensor_facts,
            forbidden=frozenset(),
            matrices=False,
        )
        a_transfer_body, a_caches = builder.transport(
            a_transfers,
            row=m,
            column=k,
            rows=site.bm,
            columns=site.bk,
            consumer_size=str(a_layout.values_per_thread),
            coordinates=lambda element, layout=a_layout: layout.coordinates(
                tid, element
            ),
            aligned_names=alignments,
            producer_layout=a_layout,
        )
        b_transfer_body, b_caches = builder.transport(
            b_transfers,
            row=b_row,
            column=b_column,
            rows=b_layout.rows,
            columns=b_layout.columns,
            consumer_size=str(b_layout.values_per_thread),
            coordinates=lambda element, layout=b_layout: layout.coordinates(
                tid, element
            ),
            aligned_names=alignments,
            producer_layout=b_layout,
        )
        a_reuse, a_statements, a_caches = builder.cache_axis_recipes(
            a_layout, a_statements, a_value, column=k, caches=a_caches
        )
        b_reuse, b_statements, b_caches = builder.cache_axis_recipes(
            b_layout, b_statements, b_value, column=b_column, caches=b_caches
        )
        reduction.extend([*a_transfer_body, *b_transfer_body, *a_reuse, *b_reuse])
        reduction.append(
            builder.packet_loop(
                a_layout,
                a_statements,
                a_value,
                row=m,
                column=k,
                dtype=site.dtype,
                destination=a_shared,
                caches=a_caches,
                aligned_names=alignments,
            )
        )
        if register_b:
            reduction.extend(
                builder.register_b(
                    b_layout,
                    b_statements,
                    b_value,
                    row=b_row,
                    column=b_column,
                    dtype=site.dtype,
                    fragment=rb,
                    thread=thread,
                    caches=b_caches,
                    aligned_names=alignments,
                )
            )
        else:
            reduction.append(
                builder.packet_loop(
                    b_layout,
                    b_statements,
                    b_value,
                    row=b_row,
                    column=b_column,
                    dtype=site.dtype,
                    destination=b_shared,
                    transpose=not b_k_major,
                    caches=b_caches,
                    aligned_names=alignments,
                )
            )
        ki = names("k_fragment")
        reduction.extend(
            _statements(f"""
cute.arch.sync_threads()
cute.copy({ca}, {pa}, {da})
{"" if register_b else f"cute.copy({cb}, {pb}, {db})"}
for {ki} in cutlass.range_constexpr({site.bk // 16}):
    cute.gemm({mma}, {accumulator}, {ra}[None, None, {ki}], {rb}[None, None, {ki}], {accumulator})
cute.arch.sync_threads()
""")
        )
        result.append(
            ast.For(
                ast.Name(site.k_offset, ast.Store()),
                _clone(site.reduction_iterator),
                reduction,
                [],
            )
        )
    result.extend(
        builder.output(
            sites[-1],
            epilogue,
            accumulator,
            final_coordinates,
            terminal_arena_dead=terminal_arena_dead,
        )
    )
    allocation = _statements(
        f"{arena} = cute.arch.alloc_smem(cutlass.Uint8, {builder.shared_bytes}, alignment={_SHARED_ALIGNMENT_BYTES})"
    )
    return RegisterChainPlan(
        sites,
        thread_count,
        builder.shared_bytes,
        tuple(layouts),
        tuple(b_majors),
        tuple(register_b_layouts),
        builder.slice_transports,
        builder.matrix_transports,
        tuple(
            ast.fix_missing_locations(statement) for statement in [*allocation, *result]
        ),
        tuple(names.names),
        tid,
    )


def emit_register_chain(
    plan: RegisterChainPlan,
    *,
    prefix: str,
    tid: str,
    fresh_name: Callable[[str], str],
) -> list[ast.stmt]:
    """Clone a fully accepted plan, then let the caller commit it atomically."""
    replacements = {
        name: _expr(fresh_name(f"{prefix}_{name.removeprefix('_helion_rc_')}"))
        for name in plan._generated_names
        if name != plan._tid
    }
    replacements[plan._tid] = _expr(tid)
    return [
        cast("ast.stmt", _replace(statement, replacements))
        for statement in plan._statements
    ]
