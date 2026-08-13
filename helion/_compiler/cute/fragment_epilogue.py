"""Generic pair-local tcgen05 fragment epilogues.

The planner owns a pure, single-store slice of the live FX graph.  It admits
operators individually and interprets their logical coordinates; it never
recognizes an expression or workload pattern.  The execution envelope is
deliberately narrow: a static tcgen05 layout whose adjacent register pair has
been validated to hold adjacent trailing-axis output elements.
"""

from __future__ import annotations

import ast
import dataclasses
from fractions import Fraction
import functools
import math
import operator
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast

import sympy
import torch
from torch._inductor.virtualized import V
from torch.fx.node import Node
from torch.fx.node import map_arg

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import tile_index
from ...language import view_ops
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..indexing_strategy import exact_tile_block_ids
from .cute_fx_walk import build_inner_outputs_index_from_graphs
from .cute_fx_walk import reach_matmul_anchors
from .cute_fx_walk import walk_carrier_to_tcgen05_matmul
from .cute_reshape import _get_tile_shape

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ...runtime.config import Config
    from ..device_ir import GraphInfo
    from ..inductor_lowering import CodegenState

    _IndexEvaluator = Callable[[dict[str, int]], int]


_T = TypeVar("_T")

# The only audited physical layout places adjacent even/odd trailing-axis
# elements in adjacent registers without crossing a subtile.  The renderer is
# intrinsically pair-local, so this is an implementation constant rather than
# a configurable capability.
_TCGEN05_PAIR_WIDTH = 2


class _UnsupportedFragment(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class Tcgen05PairEpiloguePlan:
    anchor: Node
    store_node: Node
    value_node: Node
    boundary_nodes: frozenset[Node]
    owned_nodes: frozenset[Node]


@dataclasses.dataclass(frozen=True)
class Tcgen05PairEpilogueCandidate:
    requires_pair_layout: bool


@dataclasses.dataclass(frozen=True)
class _Region:
    store: Node
    value: Node
    owned: frozenset[Node]
    boundaries: frozenset[Node]


_RESHAPE_TARGETS = {
    torch.ops.aten.reshape.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.view.default,
}
_PERMUTE_TARGETS = {
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.t.default,
}
_SHAPE_TARGETS = {
    *_RESHAPE_TARGETS,
    *_PERMUTE_TARGETS,
    torch.ops.aten.expand.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.dim,
    view_ops.subscript,
    view_ops.split,
    view_ops.join,
}


def _pointwise_inputs(node: Node) -> tuple[Node, ...] | None:
    from ..inductor_lowering import PointwiseLowering

    lowering = node.meta.get("lowering")
    if not isinstance(lowering, PointwiseLowering):
        return None
    inputs: list[Node] = []

    def visit(value: Node) -> Node:
        if isinstance(value.meta.get("val"), torch.Tensor):
            inputs.append(value)
        return value

    map_arg((node.args, {**node.kwargs, "_extra_deps": None}), visit)
    if not inputs or len(inputs) != len(lowering.input_names):
        return None
    return tuple(inputs)


def _is_split_getitem(node: Node) -> bool:
    base = node.args[0] if node.args else None
    index = node.args[1] if len(node.args) > 1 else None
    return (
        node.op == "call_function"
        and node.target is operator.getitem
        and isinstance(base, Node)
        and base.target is view_ops.split
        and index in (0, 1)
    )


def _is_host_tensor(node: Node) -> bool:
    return node.op == "call_function" and node.target is _tracing_ops._host_tensor


def _shape_only_subscript(node: Node) -> bool:
    source = node.args[0] if node.args else None
    indices = node.args[1] if len(node.args) > 1 else None
    source_value = source.meta.get("val") if isinstance(source, Node) else None
    output_value = node.meta.get("val")
    return (
        isinstance(source_value, torch.Tensor)
        and isinstance(output_value, torch.Tensor)
        and isinstance(indices, (list, tuple))
        and all(index is None or index == slice(None) for index in indices)
        and sum(index == slice(None) for index in indices) == source_value.ndim
        and len(indices) == output_value.ndim
    )


def _tile_index_block_ids(node: Node) -> set[int]:
    """Return tile-index provenance, rejecting arithmetic/data-derived indices."""
    if node.target is tile_index:
        size_node = node.args[0] if node.args else None
        size = size_node.meta.get("val") if isinstance(size_node, Node) else size_node
        block_id = (
            CompileEnvironment.current().get_block_id(size)
            if isinstance(size, (int, torch.SymInt, sympy.Basic))
            else None
        )
        if block_id is not None:
            return {block_id}
    source = node.args[0] if node.args else None
    shape_target = node.target in _RESHAPE_TARGETS | _PERMUTE_TARGETS | {
        torch.ops.aten.expand.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.dim,
        view_ops.subscript,
    }
    if isinstance(source, Node) and (
        shape_target or (node.target is memory_ops.load and not _is_host_tensor(source))
    ):
        if node.target is not view_ops.subscript or _shape_only_subscript(node):
            return _tile_index_block_ids(source)
    raise _UnsupportedFragment


def _extent_matches_block(extent: int | torch.SymInt, block_id: int) -> bool:
    env = CompileEnvironment.current()
    size = env.block_sizes[env.canonical_block_id(block_id)].size
    return isinstance(size, (int, torch.SymInt)) and env.known_equal(extent, size)


def _validate_host_load(node: Node, output_node: Node) -> None:
    source = node.args[0] if node.args else None
    indices = node.args[1] if len(node.args) > 1 else None
    source_value = source.meta.get("val") if isinstance(source, Node) else None
    output_value = node.meta.get("val")
    if not (
        isinstance(source, Node)
        and _is_host_tensor(source)
        and source is not output_node
        and isinstance(source_value, torch.Tensor)
        and isinstance(output_value, torch.Tensor)
        and isinstance(indices, (list, tuple))
        and len(indices) == source_value.ndim
        and (len(node.args) <= 2 or node.args[2] is None)
        and source_value.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        raise _UnsupportedFragment
    tensor_indices = [
        index
        for index in indices
        if isinstance(index, Node) and isinstance(index.meta.get("val"), torch.Tensor)
    ]
    env = CompileEnvironment.current()
    if tensor_indices and len(tensor_indices) != len(indices):
        raise _UnsupportedFragment
    used: set[int] = set()
    output_dim = 0
    for tensor_dim, index in enumerate(indices):
        block_ids: set[int] = set()
        if isinstance(index, Node) and isinstance(index.meta.get("val"), torch.Tensor):
            block_ids = _tile_index_block_ids(index)
        elif isinstance(index, Node) and isinstance(
            index.meta.get("val"), torch.SymInt
        ):
            if (block_id := env.get_block_id(index.meta["val"])) is not None:
                block_ids = {block_id}
        elif index == slice(None) and output_dim < output_value.ndim:
            if (
                block_id := env.resolve_block_id(output_value.shape[output_dim])
            ) is not None:
                block_ids = {block_id}
        elif not (
            isinstance(index, int)
            and isinstance(source_value.shape[tensor_dim], int)
            and 0 <= index < source_value.shape[tensor_dim]
        ):
            raise _UnsupportedFragment
        if block_ids:
            if len(block_ids) != 1:
                raise _UnsupportedFragment
            (block_id,) = block_ids
            canonical = env.canonical_block_id(block_id)
            expected = (
                None
                if tensor_indices
                else env.resolve_block_id(output_value.shape[output_dim])
            )
            if (
                canonical in used
                or (
                    expected is not None
                    and canonical != env.canonical_block_id(expected)
                )
                or not _extent_matches_block(source_value.shape[tensor_dim], block_id)
            ):
                raise _UnsupportedFragment
            used.add(canonical)
            output_dim += 1
    if not tensor_indices and output_dim != output_value.ndim:
        raise _UnsupportedFragment


def _node_inputs(node: Node, output_node: Node) -> tuple[Node, ...] | None:
    if node.op != "call_function":
        return None
    if (pointwise := _pointwise_inputs(node)) is not None:
        return pointwise
    if node.target is tile_index:
        return ()
    if node.target is memory_ops.load:
        source = node.args[0] if node.args else None
        if not isinstance(source, Node) or (
            len(node.args) > 2 and node.args[2] is not None
        ):
            return None
        if not _is_host_tensor(source):
            return (source,) if _shape_only_subscript(node) else None
        _validate_host_load(node, output_node)
        indices = node.args[1] if len(node.args) > 1 else ()
        if not isinstance(indices, (list, tuple)):
            raise _UnsupportedFragment
        return tuple(
            index
            for index in indices
            if isinstance(index, Node)
            and isinstance(index.meta.get("val"), torch.Tensor)
        )
    if node.target is view_ops.subscript and not _shape_only_subscript(node):
        return None
    if node.target is view_ops.join:
        return tuple(arg for arg in node.args[:2] if isinstance(arg, Node))
    if node.target not in _SHAPE_TARGETS and not _is_split_getitem(node):
        return None
    source = node.args[0] if node.args else None
    return (source,) if isinstance(source, Node) else ()


def _extract_region(
    graphs: Sequence[GraphInfo],
    anchor: Node,
    expected_output_block_ids: tuple[int, ...],
) -> _Region:
    inner_outputs = build_inner_outputs_index_from_graphs(graphs)
    reachable = [
        (store, value)
        for graph_info in graphs
        for store in graph_info.graph.nodes
        if store.target is memory_ops.store
        and len(store.args) > 2
        and isinstance((value := store.args[2]), Node)
        if anchor
        in reach_matmul_anchors(
            value,
            target_fx_nodes={anchor},
            inner_outputs_by_graph_id=inner_outputs,
        )
    ]
    if len(reachable) != 1:
        raise _UnsupportedFragment
    store, value = reachable[0]
    output_node = store.args[0] if store.args else None
    output = output_node.meta.get("val") if isinstance(output_node, Node) else None
    subscripts = store.args[1] if len(store.args) > 1 else None
    mask = store.args[3] if len(store.args) > 3 else None
    if (
        not isinstance(output_node, Node)
        or not isinstance(output, torch.Tensor)
        or output.ndim != 3
        or output.dtype not in (torch.float16, torch.bfloat16)
        or not isinstance(subscripts, (list, tuple))
        or exact_tile_block_ids(CompileEnvironment.current(), subscripts)
        != expected_output_block_ids
        or mask is not None
    ):
        raise _UnsupportedFragment

    owned: set[Node] = set()
    boundaries: set[Node] = set()
    visiting: set[Node] = set()

    def visit(node: Node) -> None:
        if node in owned or node in boundaries:
            return
        if walk_carrier_to_tcgen05_matmul(node, {anchor}, inner_outputs) is anchor:
            boundaries.add(node)
            return
        if node.graph is not store.graph or node in visiting:
            raise _UnsupportedFragment(
                f"unsupported region node {node.name}: {node.target}"
            )
        dependencies = _node_inputs(node, output_node)
        if dependencies is None:
            raise _UnsupportedFragment(
                f"unsupported region node {node.name}: {node.target}"
            )
        visiting.add(node)
        for dependency in dependencies:
            visit(dependency)
        visiting.remove(node)
        owned.add(node)

    visit(value)
    if not boundaries:
        raise _UnsupportedFragment
    allowed_users = {*owned, store}
    if any(
        user not in allowed_users
        for node in (*owned, *boundaries)
        for user in node.users
    ):
        raise _UnsupportedFragment

    body_graph_ids = [
        graph_info.graph_id for graph_info in graphs if graph_info.graph is anchor.graph
    ]
    if len(body_graph_ids) != 1:
        raise _UnsupportedFragment
    (body_graph_id,) = body_graph_ids
    loop_nodes = [
        node
        for node in store.graph.nodes
        if node.op == "call_function"
        and _tracing_ops.is_for_loop_target(node.target)
        and node.args
        and node.args[0] == body_graph_id
    ]
    if len(loop_nodes) != 1:
        raise _UnsupportedFragment
    (loop_node,) = loop_nodes
    order = {node: index for index, node in enumerate(store.graph.nodes)}
    if not all(order[loop_node] < order[node] < order[store] for node in owned):
        raise _UnsupportedFragment
    for node in store.graph.nodes:
        if not (order[loop_node] < order[node] < order[store]) or node in owned:
            continue
        is_loop_result = (
            node.target is operator.getitem and node.args and node.args[0] is loop_node
        )
        is_shape_scalar = isinstance(node.meta.get("val"), (int, torch.SymInt))
        if (
            node.op == "call_function"
            and node not in boundaries
            and not is_loop_result
            and not is_shape_scalar
            and not _is_host_tensor(node)
        ):
            raise _UnsupportedFragment(f"intervening node {node.name}: {node.target}")

    return _Region(store, value, frozenset(owned), frozenset(boundaries))


def analyze_tcgen05_pair_epilogue_candidate(
    graphs: Sequence[GraphInfo],
    anchor: Node,
    *,
    expected_output_block_ids: tuple[int, ...],
) -> Tcgen05PairEpilogueCandidate | None:
    """Side-effect-free preflight sharing the fragment planner's extraction."""
    try:
        region = _extract_region(graphs, anchor, expected_output_block_ids)
    except _UnsupportedFragment:
        return None
    return Tcgen05PairEpilogueCandidate(
        requires_pair_layout=any(
            node.target in _SHAPE_TARGETS or _is_split_getitem(node)
            for node in region.owned
        )
    )


def _interpret_index(
    value: sympy.Expr,
    variables: Mapping[str, _T],
    *,
    constant: Callable[[int], _T],
    add: Callable[[tuple[_T, ...]], _T],
    multiply: Callable[[tuple[_T, ...]], _T],
    floor_divide: Callable[[_T, _T], _T],
    modulo: Callable[[_T, _T], _T],
) -> _T:
    """Interpret the complete logical-index grammar into one target domain."""

    def visit(current: sympy.Expr) -> _T:
        if isinstance(current, sympy.Integer):
            return constant(int(current))
        if isinstance(current, sympy.Symbol):
            return variables[str(current)]
        if current.is_Add:
            return add(tuple(visit(cast("sympy.Expr", arg)) for arg in current.args))
        if current.is_Mul:
            return multiply(
                tuple(visit(cast("sympy.Expr", arg)) for arg in current.args)
            )
        if current.func is sympy.floor:
            numerator, denominator = cast(
                "sympy.Expr", current.args[0]
            ).as_numer_denom()
            return floor_divide(visit(numerator), visit(denominator))
        if current.func is sympy.Mod:
            return modulo(
                visit(cast("sympy.Expr", current.args[0])),
                visit(cast("sympy.Expr", current.args[1])),
            )
        raise _UnsupportedFragment(f"unsupported logical index {current}")

    return visit(value)


@dataclasses.dataclass(frozen=True)
class _Index:
    semantic: sympy.Expr
    bounds: tuple[tuple[sympy.Symbol, int, int], ...] = ()

    @staticmethod
    def constant(value: int) -> _Index:
        return _Index(sympy.Integer(value))

    @staticmethod
    def variable(name: str, upper: int) -> _Index:
        symbol = sympy.Symbol(name, integer=True, nonnegative=True)
        return _Index(symbol, ((symbol, 0, upper),))

    def compile(self) -> _IndexEvaluator:
        """Compile this index once for repeated exhaustive evaluation."""

        def variable(name: str) -> _IndexEvaluator:
            return operator.itemgetter(name)

        def constant(value: int) -> _IndexEvaluator:
            return lambda _values: value

        def add(parts: tuple[_IndexEvaluator, ...]) -> _IndexEvaluator:
            return lambda values: sum(part(values) for part in parts)

        def multiply(parts: tuple[_IndexEvaluator, ...]) -> _IndexEvaluator:
            return lambda values: math.prod(part(values) for part in parts)

        def floor_divide(
            numerator: _IndexEvaluator, denominator: _IndexEvaluator
        ) -> _IndexEvaluator:
            return lambda values: numerator(values) // denominator(values)

        def modulo(
            numerator: _IndexEvaluator, denominator: _IndexEvaluator
        ) -> _IndexEvaluator:
            return lambda values: numerator(values) % denominator(values)

        variables = {
            str(symbol): variable(str(symbol)) for symbol, _lower, _upper in self.bounds
        }
        return _interpret_index(
            self.semantic,
            variables,
            constant=constant,
            add=add,
            multiply=multiply,
            floor_divide=floor_divide,
            modulo=modulo,
        )

    def render(self, variables: dict[str, str]) -> str:
        return _interpret_index(
            self.semantic,
            variables,
            constant=lambda value: f"cutlass.Int32({value})",
            add=lambda parts: "(" + " + ".join(parts) + ")",
            multiply=lambda parts: "(" + " * ".join(parts) + ")",
            floor_divide=lambda numerator, denominator: (
                f"(({numerator}) // ({denominator}))"
            ),
            modulo=lambda numerator, denominator: f"(({numerator}) % ({denominator}))",
        )


def _merge_bounds(*indices: _Index) -> tuple[tuple[sympy.Symbol, int, int], ...]:
    merged: dict[sympy.Symbol, tuple[int, int]] = {}
    for index in indices:
        for symbol, lower, upper in index.bounds:
            if merged.setdefault(symbol, (lower, upper)) != (lower, upper):
                raise _UnsupportedFragment
    return tuple((symbol, *merged[symbol]) for symbol in sorted(merged, key=str))


def _semantic_interval(
    value: sympy.Expr, bounds: dict[sympy.Symbol, tuple[int, int]]
) -> tuple[Fraction, Fraction] | None:
    if value.is_Rational:
        rational = Fraction(str(value))
        return rational, rational
    if isinstance(value, sympy.Symbol):
        interval = bounds.get(value)
        if interval is None:
            return None
        return Fraction(interval[0]), Fraction(interval[1])
    if value.is_Add:
        parts = [
            _semantic_interval(cast("sympy.Expr", arg), bounds) for arg in value.args
        ]
        if any(part is None for part in parts):
            return None
        lower = Fraction(0)
        upper = Fraction(0)
        for part in parts:
            if part is not None:
                lower += part[0]
                upper += part[1]
        return lower, upper
    if value.is_Mul:
        result = (Fraction(1), Fraction(1))
        for arg in value.args:
            part = _semantic_interval(cast("sympy.Expr", arg), bounds)
            if part is None:
                return None
            products = tuple(left * right for left in result for right in part)
            result = min(products), max(products)
        return result
    if value.func is sympy.floor:
        part = _semantic_interval(cast("sympy.Expr", value.args[0]), bounds)
        if part is None:
            return None
        return Fraction(math.floor(part[0])), Fraction(math.floor(part[1]))
    if value.func is sympy.Mod:
        part = _semantic_interval(cast("sympy.Expr", value.args[0]), bounds)
        modulus = value.args[1]
        if not isinstance(modulus, sympy.Integer):
            return None
        modulus_value = int(modulus)
        if part is not None and 0 <= part[0] <= part[1] < modulus_value:
            return part
        if modulus_value > 0:
            return Fraction(0), Fraction(modulus_value - 1)
    return None


@functools.lru_cache(maxsize=512)
def _simplify_semantic(
    value: sympy.Expr, bounds_tuple: tuple[tuple[sympy.Symbol, int, int], ...]
) -> sympy.Expr:
    bounds = {symbol: (lower, upper) for symbol, lower, upper in bounds_tuple}

    def visit(current: sympy.Expr) -> sympy.Expr:
        if current.args:
            current = cast(
                "sympy.Expr",
                sympy.simplify(
                    current.func(
                        *(visit(cast("sympy.Expr", arg)) for arg in current.args)
                    )
                ),
            )
        if current.func is sympy.floor:
            interval = _semantic_interval(cast("sympy.Expr", current.args[0]), bounds)
            if interval is not None and math.floor(interval[0]) == math.floor(
                interval[1]
            ):
                return sympy.Integer(math.floor(interval[0]))
        if current.func is sympy.Mod:
            interval = _semantic_interval(cast("sympy.Expr", current.args[0]), bounds)
            modulus = current.args[1]
            if (
                interval is not None
                and isinstance(modulus, sympy.Integer)
                and 0 <= interval[0] <= interval[1] < int(modulus)
            ):
                return cast("sympy.Expr", current.args[0])
        return current

    previous = value
    for _ in range(4):
        current = visit(previous)
        if current == previous:
            return current
        previous = current
    return previous


def _binary(op: str, left: _Index | int, right: _Index | int) -> _Index:
    left = left if isinstance(left, _Index) else _Index.constant(left)
    right = right if isinstance(right, _Index) else _Index.constant(right)
    bounds = _merge_bounds(left, right)
    if op == "add":
        semantic = sympy.Add(left.semantic, right.semantic)
    elif op == "mul":
        semantic = sympy.Mul(left.semantic, right.semantic)
    elif op == "floordiv":
        quotient = sympy.Mul(left.semantic, sympy.Pow(right.semantic, -1))
        semantic = cast("sympy.Expr", sympy.floor(quotient))
    elif op == "mod":
        semantic = sympy.Mod(left.semantic, right.semantic)
    else:
        raise AssertionError(op)
    return _Index(_simplify_semantic(semantic, bounds), bounds)


def _add(left: _Index | int, right: _Index | int) -> _Index:
    return _binary("add", left, right)


def _mul(left: _Index | int, right: _Index | int) -> _Index:
    return _binary("mul", left, right)


def _floordiv(left: _Index | int, right: _Index | int) -> _Index:
    return _binary("floordiv", left, right)


def _mod(left: _Index | int, right: _Index | int) -> _Index:
    return _binary("mod", left, right)


def _constant_index(index: _Index) -> int | None:
    return int(index.semantic) if isinstance(index.semantic, sympy.Integer) else None


def _coords_from_flat(flat: _Index, shape: Sequence[int]) -> list[_Index]:
    coords: list[_Index] = []
    stride = 1
    for extent in reversed(shape):
        coords.append(_mod(_floordiv(flat, stride), extent))
        stride *= extent
    return list(reversed(coords))


def _flat_from_coords(coords: Sequence[_Index], shape: Sequence[int]) -> _Index:
    result = _Index.constant(0)
    for coord, extent in zip(coords, shape, strict=True):
        result = _add(_mul(result, extent), coord)
    return result


def _broadcast_flat(
    flat: _Index, output_shape: Sequence[int], source_shape: Sequence[int]
) -> _Index:
    if len(source_shape) > len(output_shape):
        raise _UnsupportedFragment
    output_coords = _coords_from_flat(flat, output_shape)
    offset = len(output_shape) - len(source_shape)
    source_coords: list[_Index] = []
    for dim, source_extent in enumerate(source_shape):
        output_extent = output_shape[offset + dim]
        if source_extent == 1:
            source_coords.append(_Index.constant(0))
        elif source_extent == output_extent:
            source_coords.append(output_coords[offset + dim])
        else:
            raise _UnsupportedFragment
    return _flat_from_coords(source_coords, source_shape)


def _shape(node: Node, config: Config) -> list[int]:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        raise _UnsupportedFragment
    return _get_tile_shape(value, CompileEnvironment.current(), config)


def _permutation(node: Node, ndim: int) -> list[int]:
    if node.target is torch.ops.aten.permute.default:
        dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dims")
        if not isinstance(dims, (list, tuple)):
            raise _UnsupportedFragment
        result: list[int] = []
        for dim in dims:
            if not isinstance(dim, int):
                raise _UnsupportedFragment
            result.append(dim % ndim)
        return result
    if node.target is torch.ops.aten.transpose.int:
        dim0 = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim0")
        dim1 = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim1")
        if not isinstance(dim0, int) or not isinstance(dim1, int):
            raise _UnsupportedFragment
        result = list(range(ndim))
        result[dim0 % ndim], result[dim1 % ndim] = (
            result[dim1 % ndim],
            result[dim0 % ndim],
        )
        return result
    if node.target is torch.ops.aten.t.default and ndim == 2:
        return [1, 0]
    raise _UnsupportedFragment


def _resolve_shape(
    node: Node,
    flat: _Index,
    *,
    config: Config,
    evaluate: Callable[[Node, _Index, int | None], object],
    choose: Callable[[_Index, object, object], object],
    projection: int | None = None,
) -> object:
    if node.target in _RESHAPE_TARGETS:
        source = node.args[0]
        if not isinstance(source, Node):
            raise _UnsupportedFragment
        return evaluate(source, flat, None)
    if node.target in _PERMUTE_TARGETS:
        source = node.args[0]
        if not isinstance(source, Node):
            raise _UnsupportedFragment
        output_coords = _coords_from_flat(flat, _shape(node, config))
        perm = _permutation(node, len(output_coords))
        source_coords = [output_coords[perm.index(dim)] for dim in range(len(perm))]
        return evaluate(
            source, _flat_from_coords(source_coords, _shape(source, config)), None
        )
    if node.target is torch.ops.aten.expand.default:
        source = node.args[0]
        if not isinstance(source, Node):
            raise _UnsupportedFragment
        return evaluate(
            source,
            _broadcast_flat(flat, _shape(node, config), _shape(source, config)),
            None,
        )
    if node.target is torch.ops.aten.unsqueeze.default:
        source = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        if not isinstance(source, Node) or not isinstance(dim, int):
            raise _UnsupportedFragment
        coords = _coords_from_flat(flat, _shape(node, config))
        dim %= len(coords)
        return evaluate(
            source,
            _flat_from_coords(coords[:dim] + coords[dim + 1 :], _shape(source, config)),
            None,
        )
    if node.target is torch.ops.aten.squeeze.dim:
        source = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        if not isinstance(source, Node) or not isinstance(dim, int):
            raise _UnsupportedFragment
        source_shape = _shape(source, config)
        dim %= len(source_shape)
        coords = _coords_from_flat(flat, _shape(node, config))
        if source_shape[dim] == 1:
            coords.insert(dim, _Index.constant(0))
        return evaluate(source, _flat_from_coords(coords, source_shape), None)
    source_arg = node.args[0] if node.args else None
    if node.target is view_ops.subscript or (
        node.target is memory_ops.load
        and isinstance(source_arg, Node)
        and not _is_host_tensor(source_arg)
    ):
        source = source_arg
        indices = node.args[1] if len(node.args) > 1 else None
        if not isinstance(source, Node) or not isinstance(indices, (list, tuple)):
            raise _UnsupportedFragment
        output_coords = iter(_coords_from_flat(flat, _shape(node, config)))
        source_coords: list[_Index] = []
        for index in indices:
            coord = next(output_coords)
            if index is None:
                continue
            if index != slice(None):
                raise _UnsupportedFragment
            source_coords.append(coord)
        return evaluate(
            source, _flat_from_coords(source_coords, _shape(source, config)), None
        )
    if _is_split_getitem(node):
        split = node.args[0]
        projection_arg = node.args[1]
        assert isinstance(split, Node) and isinstance(projection_arg, int)
        source = split.args[0]
        if not isinstance(source, Node):
            raise _UnsupportedFragment
        return evaluate(source, _add(_mul(flat, 2), projection_arg), None)
    if node.target is view_ops.split:
        if projection not in (0, 1):
            raise _UnsupportedFragment
        source = node.args[0]
        if not isinstance(source, Node):
            raise _UnsupportedFragment
        return evaluate(source, _add(_mul(flat, 2), projection), None)
    if node.target is view_ops.join:
        left = node.args[0] if node.args else None
        right = node.args[1] if len(node.args) > 1 else None
        if not isinstance(left, Node) or not isinstance(right, Node):
            raise _UnsupportedFragment
        coords = _coords_from_flat(flat, _shape(node, config))
        selector = coords[-1]
        source_flat = _flat_from_coords(coords[:-1], _shape(left, config))
        return choose(
            selector,
            evaluate(left, source_flat, None),
            evaluate(right, source_flat, None),
        )
    return evaluate(node, flat, projection)


def _collect_demands(
    node: Node,
    flat: _Index,
    *,
    region: _Region,
    config: Config,
    projection: int | None = None,
    memo: dict[tuple[Node, _Index, int | None], frozenset[tuple[Node, _Index]]]
    | None = None,
) -> frozenset[tuple[Node, _Index]]:
    if memo is None:
        memo = {}
    key = (node, flat, projection)
    if key in memo:
        return memo[key]

    def evaluate(
        current: Node, current_flat: _Index, current_projection: int | None
    ) -> object:
        if (
            current is not node
            or current_flat != flat
            or current_projection != projection
        ):
            return _collect_demands(
                current,
                current_flat,
                region=region,
                config=config,
                projection=current_projection,
                memo=memo,
            )
        if current in region.boundaries:
            return frozenset({(current, current_flat)})
        if current.target is tile_index:
            return frozenset()
        source = current.args[0] if current.args else None
        if (
            current.target is memory_ops.load
            and isinstance(source, Node)
            and _is_host_tensor(source)
        ):
            output_shape = _shape(current, config)
            demands: set[tuple[Node, _Index]] = set()
            indices = current.args[1] if len(current.args) > 1 else None
            if not isinstance(indices, (list, tuple)):
                raise _UnsupportedFragment
            for index in indices:
                if isinstance(index, Node) and isinstance(
                    index.meta.get("val"), torch.Tensor
                ):
                    index_flat = _broadcast_flat(
                        current_flat, output_shape, _shape(index, config)
                    )
                    demands.update(
                        _collect_demands(
                            index, index_flat, region=region, config=config, memo=memo
                        )
                    )
            return frozenset(demands)
        inputs = _pointwise_inputs(current)
        if inputs is not None:
            demands: set[tuple[Node, _Index]] = set()
            output_shape = _shape(current, config)
            for input_node in inputs:
                demands.update(
                    _collect_demands(
                        input_node,
                        _broadcast_flat(
                            current_flat, output_shape, _shape(input_node, config)
                        ),
                        region=region,
                        config=config,
                        memo=memo,
                    )
                )
            return frozenset(demands)
        raise _UnsupportedFragment

    def choose(selector: _Index, left: object, right: object) -> object:
        if not isinstance(left, frozenset) or not isinstance(right, frozenset):
            raise _UnsupportedFragment
        return left | right

    result = _resolve_shape(
        node,
        flat,
        config=config,
        evaluate=evaluate,
        choose=choose,
        projection=projection,
    )
    if not isinstance(result, frozenset):
        raise _UnsupportedFragment
    memo[key] = result
    return result


def analyze_tcgen05_pair_epilogue_plan(
    graphs: Sequence[GraphInfo],
    anchor: Node,
    *,
    expected_output_block_ids: tuple[int, ...],
    config: Config,
) -> Tcgen05PairEpiloguePlan | None:
    """Finalize logical pair locality for one static configuration."""
    try:
        region = _extract_region(graphs, anchor, expected_output_block_ids)
        output_shape = _shape(region.value, config)
        if len(output_shape) != 3 or output_shape[0] != 1:
            raise _UnsupportedFragment
        m_extent, n_extent = output_shape[-2:]
        if n_extent % _TCGEN05_PAIR_WIDTH:
            raise _UnsupportedFragment
        for boundary in region.boundaries:
            value = boundary.meta.get("val")
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype is not torch.float32
                or _shape(boundary, config) != output_shape
            ):
                raise _UnsupportedFragment

        row = _Index.variable("row", m_extent - 1)
        pair = _Index.variable("pair", n_extent // _TCGEN05_PAIR_WIDTH - 1)
        targets = [
            _flat_from_coords(
                [_Index.constant(0), row, _add(_mul(pair, 2), slot)],
                output_shape,
            )
            for slot in range(_TCGEN05_PAIR_WIDTH)
        ]
        demands = [
            _collect_demands(
                region.value,
                target,
                region=region,
                config=config,
            )
            for target in targets
        ]
        target_evaluators = [target.compile() for target in targets]
        demand_evaluators = [
            demand.compile() for slot_demands in demands for _, demand in slot_demands
        ]
        variables = {"row": 0, "pair": 0}
        for m in range(m_extent):
            variables["row"] = m
            for n_pair in range(n_extent // _TCGEN05_PAIR_WIDTH):
                variables["pair"] = n_pair
                allowed = {evaluate(variables) for evaluate in target_evaluators}
                if any(
                    evaluate(variables) not in allowed for evaluate in demand_evaluators
                ):
                    raise _UnsupportedFragment
        return Tcgen05PairEpiloguePlan(
            anchor,
            region.store,
            region.value,
            region.boundaries,
            region.owned,
        )
    except (_UnsupportedFragment, IndexError, StopIteration, ValueError):
        return None


def _render_statements(statements: Sequence[ast.AST], indent: str) -> str:
    return "".join(
        f"{indent}{line}\n"
        for statement in statements
        for line in ast.unparse(ast.fix_missing_locations(statement)).splitlines()
    )


class _Evaluator:
    def __init__(
        self,
        state: CodegenState,
        plan: Tcgen05PairEpiloguePlan,
        *,
        carrier: str,
        current_index: str,
        partner_index: str,
        pair_coordinate: str,
        current_flat: _Index,
        partner_flat: _Index,
        indent: str,
        terminals: dict[tuple[str, Node, _Index], ast.AST],
    ) -> None:
        self.state = state
        self.plan = plan
        self.carrier = carrier
        self.current_index = current_index
        self.partner_index = partner_index
        self.current_flat = current_flat
        self.partner_flat = partner_flat
        self.indent = indent
        self.terminals = terminals
        self.memo: dict[tuple[Node, _Index, int | None], ast.AST] = {}
        self.lines: list[str] = []
        tile_shape = _shape(plan.value_node, state.device_function.config)
        self.variables = {
            "row": (
                f"cutlass.Int32({pair_coordinate}[0]) % cutlass.Int32({tile_shape[-2]})"
            ),
            "pair": (
                f"(cutlass.Int32({pair_coordinate}[1]) % "
                f"cutlass.Int32({tile_shape[-1]})) // cutlass.Int32(2)"
            ),
        }

    def _bind(self, prefix: str, value: ast.AST) -> ast.AST:
        name = self.state.device_function.new_var(prefix)
        self.lines.append(
            _render_statements(
                [statement_from_string(f"{name} = {{value}}", value=value)], self.indent
            )
        )
        return expr_from_string(name)

    def _boundary(self, node: Node, flat: _Index) -> ast.AST:
        key = ("boundary", node, flat)
        if key in self.terminals:
            return self.terminals[key]
        if flat == self.current_flat:
            expression = f"{self.carrier}[{self.current_index}]"
        elif flat == self.partner_flat:
            expression = f"{self.carrier}[{self.partner_index}]"
        else:
            requested = flat.render(self.variables)
            current = self.current_flat.render(self.variables)
            expression = (
                f"{self.carrier}[{self.current_index}] if ({requested}) == ({current}) "
                f"else {self.carrier}[{self.partner_index}]"
            )
        result = self._bind("tcgen05_epi_acc", expr_from_string(expression))
        self.terminals[key] = result
        return result

    def _tile_index(self, node: Node, flat: _Index) -> ast.AST:
        from ...language.memory_ops import _cute_tile_begin_expr

        key = ("index", node, flat)
        if key in self.terminals:
            return self.terminals[key]
        size_node = node.args[0] if node.args else None
        size = size_node.meta.get("val") if isinstance(size_node, Node) else size_node
        coord = _coords_from_flat(
            flat, _shape(node, self.state.device_function.config)
        )[0]
        result = self._bind(
            "tcgen05_epi_index",
            expr_from_string(
                f"cutlass.Int32({_cute_tile_begin_expr(self.state, size)}) + "
                f"cutlass.Int32({coord.render(self.variables)})"
            ),
        )
        self.terminals[key] = result
        return result

    def _host_load(self, node: Node, flat: _Index) -> ast.AST:
        from ...language.memory_ops import _cute_scalar_load_expr
        from ...language.memory_ops import _cute_tile_begin_expr

        key = ("load", node, flat)
        if key in self.terminals:
            return self.terminals[key]
        source = node.args[0]
        indices = node.args[1]
        assert isinstance(source, Node) and isinstance(indices, (list, tuple))
        tensor = source.meta["val"]
        assert isinstance(tensor, torch.Tensor)
        tensor_name = self.state.device_function.tensor_arg(tensor).name
        self.state.device_function.placeholder_args.add(tensor_name)
        output_shape = _shape(node, self.state.device_function.config)
        output_coords = _coords_from_flat(flat, output_shape)
        tensor_indices = [
            index
            for index in indices
            if isinstance(index, Node)
            and isinstance(index.meta.get("val"), torch.Tensor)
        ]
        index_exprs: list[str] = []
        if tensor_indices:
            for index in tensor_indices:
                index_flat = _broadcast_flat(
                    flat, output_shape, _shape(index, self.state.device_function.config)
                )
                index_exprs.append(ast.unparse(self.evaluate(index, index_flat)))
        else:
            output_dim = 0
            for index in indices:
                if isinstance(index, Node) and isinstance(
                    index.meta.get("val"), torch.SymInt
                ):
                    extent = index.meta["val"]
                elif index == slice(None):
                    extent = node.meta["val"].shape[output_dim]
                else:
                    assert isinstance(index, int)
                    index_exprs.append(str(index))
                    continue
                index_exprs.append(
                    f"cutlass.Int32({_cute_tile_begin_expr(self.state, extent)}) + "
                    f"cutlass.Int32({output_coords[output_dim].render(self.variables)})"
                )
                output_dim += 1
        result = self._bind(
            "tcgen05_epi_load",
            expr_from_string(
                _cute_scalar_load_expr(tensor_name, index_exprs, tensor.dtype)
            ),
        )
        self.terminals[key] = result
        return result

    def _pointwise(self, node: Node, flat: _Index, inputs: tuple[Node, ...]) -> ast.AST:
        from ..aten_lowering import LoweringContext
        from ..inductor_lowering import PointwiseLowering

        lowering = node.meta.get("lowering")
        if not isinstance(lowering, PointwiseLowering):
            raise RuntimeError(f"non-pointwise fragment node {node.name}")
        output_shape = _shape(node, self.state.device_function.config)
        input_values = [
            self.evaluate(
                input_node,
                _broadcast_flat(
                    flat,
                    output_shape,
                    _shape(input_node, self.state.device_function.config),
                ),
            )
            for input_node in inputs
        ]
        ctx = LoweringContext.__new__(LoweringContext)
        ctx.cg = self.state.codegen
        ctx.env = self.state.env
        statements: list[ast.AST] = []
        with (
            self.state.codegen.set_statements(statements),
            V.set_current_node(node),
        ):
            result = lowering.codegen_from_input_asts(ctx, node, input_values)
        if not isinstance(result, ast.AST):
            raise RuntimeError(f"invalid pointwise fragment result for {node.name}")
        self.lines.append(_render_statements(statements, self.indent))
        return self._bind("tcgen05_epi_value", result)

    def evaluate(
        self, node: Node, flat: _Index, projection: int | None = None
    ) -> ast.AST:
        key = (node, flat, projection)
        if key in self.memo:
            return self.memo[key]

        def leaf(
            current: Node, current_flat: _Index, current_projection: int | None
        ) -> object:
            if (
                current is not node
                or current_flat != flat
                or current_projection != projection
            ):
                return self.evaluate(current, current_flat, current_projection)
            if current in self.plan.boundary_nodes:
                return self._boundary(current, current_flat)
            if current.target is tile_index:
                return self._tile_index(current, current_flat)
            source = current.args[0] if current.args else None
            if (
                current.target is memory_ops.load
                and isinstance(source, Node)
                and _is_host_tensor(source)
            ):
                return self._host_load(current, current_flat)
            inputs = _pointwise_inputs(current)
            if inputs is not None:
                return self._pointwise(current, current_flat, inputs)
            raise RuntimeError(f"unsupported committed fragment node {current.name}")

        def choose(selector: _Index, left: object, right: object) -> object:
            if not isinstance(left, ast.AST) or not isinstance(right, ast.AST):
                raise RuntimeError("invalid fragment selection")
            if (constant := _constant_index(selector)) is not None:
                return left if constant == 0 else right
            return self._bind(
                "tcgen05_epi_select",
                expr_from_string(
                    f"({{left}} if ({selector.render(self.variables)}) == "
                    "cutlass.Int32(0) else {right})",
                    left=left,
                    right=right,
                ),
            )

        result = _resolve_shape(
            node,
            flat,
            config=self.state.device_function.config,
            evaluate=leaf,
            choose=choose,
            projection=projection,
        )
        if not isinstance(result, ast.AST):
            raise RuntimeError(f"invalid committed fragment result for {node.name}")
        self.memo[key] = result
        return result


def render_tcgen05_pair_epilogue(
    state: CodegenState,
    plan: Tcgen05PairEpiloguePlan,
    *,
    carrier_name: str,
    coordinate_name: str,
    target_dtype: str,
    indent: str,
) -> tuple[str, str]:
    """Render the committed graph once for each adjacent register pair."""
    output_shape = _shape(plan.value_node, state.device_function.config)
    row = _Index.variable("row", output_shape[-2] - 1)
    pair = _Index.variable("pair", output_shape[-1] // _TCGEN05_PAIR_WIDTH - 1)
    current_flat = _flat_from_coords(
        [_Index.constant(0), row, _mul(pair, 2)], output_shape
    )
    partner_flat = _flat_from_coords(
        [_Index.constant(0), row, _add(_mul(pair, 2), 1)], output_shape
    )
    df = state.device_function
    output = df.new_var("tcgen05_epi_output")
    current_index = df.new_var("tcgen05_epi_pair")
    partner_index = df.new_var("tcgen05_epi_partner")
    pair_coordinate = df.new_var("tcgen05_epi_pair_coord")
    body_indent = indent + "    "
    terminals: dict[tuple[str, Node, _Index], ast.AST] = {}
    current = _Evaluator(
        state,
        plan,
        carrier=carrier_name,
        current_index=current_index,
        partner_index=partner_index,
        pair_coordinate=pair_coordinate,
        current_flat=current_flat,
        partner_flat=partner_flat,
        indent=body_indent,
        terminals=terminals,
    )
    partner = _Evaluator(
        state,
        plan,
        carrier=carrier_name,
        current_index=partner_index,
        partner_index=current_index,
        pair_coordinate=pair_coordinate,
        current_flat=partner_flat,
        partner_flat=current_flat,
        indent=body_indent,
        terminals=terminals,
    )
    current_value = current.evaluate(plan.value_node, current_flat)
    partner_value = partner.evaluate(plan.value_node, partner_flat)
    prelude = (
        f"{indent}assert cute.size({carrier_name}.shape) % 2 == 0, "
        '"tcgen05 pair carrier must be complete"\n'
        f"{indent}{output} = cute.make_rmem_tensor("
        f"cute.make_layout({carrier_name}.shape), {target_dtype})\n"
        f"{indent}for {current_index} in range(0, cute.size({carrier_name}.shape), 2):\n"
        f"{body_indent}{partner_index} = {current_index} + 1\n"
        f"{body_indent}{pair_coordinate} = {coordinate_name}[{current_index}]\n"
        f"{''.join(current.lines)}"
        f"{body_indent}{output}[{current_index}] = "
        f"{target_dtype}({ast.unparse(current_value)})\n"
        f"{''.join(partner.lines)}"
        f"{body_indent}{output}[{partner_index}] = "
        f"{target_dtype}({ast.unparse(partner_value)})\n"
    )
    return prelude, f"{output}.load()"
