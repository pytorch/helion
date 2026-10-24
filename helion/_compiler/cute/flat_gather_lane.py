"""Split a flattened gather index into a lane-invariant row term and a tile lane.

``hl.load(x_flat, [(starts[:, None] + tile_k.index[None, :])[:, :, None] * M
+ tile_m.index[None, None, :]])`` gathers rows of a flattened 2-D tensor: the
index tensor is ``X + tile_m.index`` where ``X`` is broadcast along the
``tile_m`` dimension and the tile term is broadcast along every other one.
Along ``tile_m`` the gathered addresses are contiguous (the tile index has
unit coefficient), so a vector of V consecutive lanes is one V-wide packet
whenever ``X`` is a multiple of V elements and the tensor's base is aligned.

The generic lowering lifts the whole index tensor into one scalar per lane,
which hides the tile term from the address analyses: the V-loop sink
(``cute/sink_vector_loops.py``) widens a load only when the vectorized
block's own per-element index variable appears in the pointer.  The load
lowering uses :func:`match_flat_gather_lane` to emit the pointer as
``base + X * stride + tile_index * stride`` instead, with the same value
modulo the index width.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._dynamo.source import LocalSource
from torch.fx import Node

from ..._compat import shape_env_size_hint
from ...language import _tracing_ops
from ...language import tile_ops
from ...language import view_ops
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Hashable
    from collections.abc import Sequence

    from ..device_ir import DeviceIR

_INTEGER_DTYPES = frozenset(
    {torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64}
)
_PASSTHROUGH_TARGETS = (
    _tracing_ops._new_var,
    torch.ops.aten.unsqueeze.default,
    torch.ops.prims.convert_element_type.default,
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
)
_ADD_TARGETS = (operator.add, torch.ops.aten.add.Tensor)
_SUB_TARGETS = (operator.sub, torch.ops.aten.sub.Tensor)
_MUL_TARGETS = (operator.mul, torch.ops.aten.mul.Tensor)


@dataclass(frozen=True)
class FlatGatherLaneSplit:
    """``index = row + tile_index(lane_block_id)`` broadcast along ``lane_dim``."""

    row: Node
    lane_block_id: int
    lane_dim: int


def _value(node: object) -> torch.Tensor | None:
    if isinstance(node, Node):
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            return value
    return None


def _is_broadcast_view(node: Node) -> bool:
    if node.target is view_ops.subscript:
        subscripts = node.args[1]
        return isinstance(subscripts, (list, tuple)) and all(
            item is None or (isinstance(item, slice) and item == slice(None))
            for item in subscripts
        )
    return node.target in _PASSTHROUGH_TARGETS


def _block_id_of_size(env: CompileEnvironment, size: object) -> int | None:
    """The block whose size symbol ``size`` is (no host-function context needed:
    the autotuner heuristics ask outside codegen)."""
    if not isinstance(size, torch.SymInt):
        return None
    expr = size._sympy_()
    for block_id, info in enumerate(env.block_sizes):
        var = info.var
        if isinstance(var, torch.SymInt) and var._sympy_() == expr:
            return block_id
    return None


def _tile_index_block(node: object, env: CompileEnvironment) -> int | None:
    """The block id of a (broadcast view of a) bare ``tile_index`` node."""
    current = node
    while isinstance(current, Node) and current.op == "call_function":
        if current.target is tile_ops.tile_index:
            value = _value(current)
            if value is None or value.ndim != 1:
                return None
            return _block_id_of_size(env, value.shape[0])
        if not _is_broadcast_view(current):
            return None
        current = current.args[0]
    return None


def _scalar(value: object) -> int | torch.SymInt | None:
    """The integer scalar ``value`` stands for: an int, or a node whose value
    is one (``_get_symnode``, ``sym_size.int``)."""
    if type(value) is int:
        return value
    if isinstance(value, Node):
        scalar = value.meta.get("val")
        if isinstance(scalar, (int, torch.SymInt)) and not isinstance(scalar, bool):
            return scalar
    return None


def _input_metadata_symbols(env: CompileEnvironment) -> set[sympy.Basic]:
    """The symbols of the input tensors' sizes and strides."""
    symbols: set[sympy.Basic] = set()
    for tensor in env.input_sources:
        for value in (*tensor.shape, *tensor.stride()):
            if isinstance(value, torch.SymInt):
                symbols.update(value._sympy_().free_symbols)
    return symbols


def _static_int(
    value: object, env: CompileEnvironment, *, hints: bool = False
) -> int | None:
    """``value`` as a static int: specialized symbols resolve; with ``hints``
    any symbol takes its size hint (a seed heuristic that will specialize the
    bound kernel on the input metadata may read it that way)."""
    scalar = _scalar(value)
    if scalar is None:
        return None
    if isinstance(scalar, int):
        return scalar
    resolved = env.specialize_expr(sympy.sympify(scalar))
    if isinstance(resolved, sympy.Integer):
        return int(resolved)
    if hints and scalar._sympy_().free_symbols <= _input_metadata_symbols(env):
        return int(shape_env_size_hint(env.shape_env, scalar._sympy_()))
    return None


def match_flat_gather_lane(
    index: object, env: CompileEnvironment
) -> FlatGatherLaneSplit | None:
    """Match an integer index tensor ``row + tile_index(block)``.

    The tile term must be the bare ``tile_index`` of a block (through
    broadcasting views only) whose extent is the only non-unit dimension of
    the term, and ``row`` must be a size-1 broadcast along that dimension, so
    the lane coordinate enters the index with unit coefficient and the rest
    of the index does not change with the lane.  Anything else (a scaled or
    shifted lane term, a lane term inside a product, a second occurrence of
    the lane) returns None.
    """
    if not isinstance(index, Node) or index.op != "call_function":
        return None
    if index.target not in _ADD_TARGETS or len(index.args) != 2:
        return None
    if index.kwargs.get("alpha", 1) != 1 or set(index.kwargs) - {"alpha"}:
        return None
    index_value = _value(index)
    if index_value is None or index_value.dtype not in _INTEGER_DTYPES:
        return None
    for lane_term, row in (index.args, index.args[::-1]):
        block_id = _tile_index_block(lane_term, env)
        if block_id is None or not isinstance(row, Node):
            continue
        lane_value = _value(lane_term)
        row_value = _value(row)
        if lane_value is None or row_value is None:
            continue
        if row_value.dtype not in _INTEGER_DTYPES:
            continue
        ndim = index_value.ndim
        if lane_value.ndim != ndim or row_value.ndim != ndim:
            continue
        lane_dims = [
            dim
            for dim in range(ndim)
            if not (
                isinstance(lane_value.shape[dim], int) and lane_value.shape[dim] == 1
            )
        ]
        if len(lane_dims) != 1:
            continue
        (lane_dim,) = lane_dims
        row_extent = row_value.shape[lane_dim]
        if not (isinstance(row_extent, int) and row_extent == 1):
            continue
        return FlatGatherLaneSplit(row=row, lane_block_id=block_id, lane_dim=lane_dim)
    return None


def row_is_multiple_of(
    node: object, multiple: int, env: CompileEnvironment, *, hints: bool = False
) -> bool:
    """Prove every value of integer expression ``node`` is a multiple of ``multiple``.

    Structural: a product with a factor the multiple divides (a static int, a
    specialized symbol, or a symbol the ``input_tensor_metadata`` fact pins
    to its hint: ``CompileEnvironment.specialized_multiple``), sums and
    differences of proven terms, broadcasting views of a proven term, static
    integers.  Unknown arithmetic (a loaded tensor, a tile index) fails
    closed; it may hold any value.  ``hints`` reads every symbol's size hint
    (for a seed heuristic that registers that fact when it emits seeds).
    """
    if multiple <= 1:
        return True
    memo: dict[Node, bool] = {}

    def factor_is_multiple(value: object) -> bool:
        if (constant := _static_int(value, env, hints=hints)) is not None:
            return constant % multiple == 0
        scalar = _scalar(value)
        if not isinstance(scalar, torch.SymInt):
            return False
        # ``specialized_multiple`` trusts a symbol's hint under the
        # ``input_tensor_metadata`` fact, which keys the bound kernel on the
        # input tensors' sizes and strides: only their symbols are pinned.
        # An ``int`` argument is keyed by type alone; its residue is pinned
        # only through the flat-gather factor specialization.
        if scalar._sympy_().free_symbols <= _input_metadata_symbols(env):
            return env.specialized_multiple(scalar, multiple)
        residue = int_factor_residue(env, scalar)
        return (
            residue is not None
            and _FACTOR_MODULUS % multiple == 0
            and residue % multiple == 0
        )

    def prove(value: object, depth: int) -> bool:
        if (constant := _static_int(value, env, hints=hints)) is not None:
            return constant % multiple == 0
        if not isinstance(value, Node) or value.op != "call_function":
            return False
        if value in memo:
            return memo[value]
        if depth > 64:
            return False
        memo[value] = False
        tensor = _value(value)
        result = False
        if tensor is not None and tensor.dtype in _INTEGER_DTYPES:
            if _is_broadcast_view(value):
                result = prove(value.args[0], depth + 1)
            elif (
                value.target in _MUL_TARGETS
                and len(value.args) == 2
                and not value.kwargs
            ):
                left, right = value.args
                for scalar, other in ((left, right), (right, left)):
                    if _scalar(scalar) is None:
                        continue
                    other_value = _value(other)
                    if not (
                        type(other) is int
                        or (
                            other_value is not None
                            and other_value.dtype in _INTEGER_DTYPES
                        )
                    ):
                        continue
                    result = factor_is_multiple(scalar) or prove(other, depth + 1)
                    break
                else:
                    result = prove(left, depth + 1) or prove(right, depth + 1)
            elif (
                value.target in (*_ADD_TARGETS, *_SUB_TARGETS)
                and len(value.args) == 2
                and value.kwargs.get("alpha", 1) == 1
                and set(value.kwargs) <= {"alpha"}
            ):
                result = prove(value.args[0], depth + 1) and prove(
                    value.args[1], depth + 1
                )
        memo[value] = result
        return result

    return prove(node, 0)


def flat_gather_sink_vec_width(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    lane_block_ids: Collection[int],
    loop_block_ids: Collection[int],
) -> int | None:
    """The packet width the V-loop sink can give every gather of the inner loops.

    The autotuner seeds that vectorize a nested row tile use this to decide
    whether such a seed may be the no-autotune default: a V-loop that is not
    sunk after all wraps the whole body and multiplies the serial work by V.
    Every ``hl.load`` / ``hl.store`` with an integer index tensor in a loop
    graph over ``loop_block_ids`` (the feature tiles and the jagged passes
    under them) must split as ``row + tile_index(lane)``
    (:func:`match_flat_gather_lane`) with the lane in ``lane_block_ids`` on
    the tensor's stride-1 dim.  V is one 16-byte packet of the widest such
    dtype; ``row`` must be a multiple of V and every tensor's base aligned
    for it.  None when any access fails or none exists.
    """
    from ...language.memory_ops import _CUTE_VECTOR_MAX_BYTES
    from ...language.memory_ops import load
    from ...language.memory_ops import store
    from ..device_ir import ForLoopGraphInfo

    # The alignment proof reads the bound inputs through the host function
    # and the compile environment; the autotuner heuristics may ask outside
    # their contexts.
    host_function = device_ir.host_function
    if host_function is None:
        return None
    accesses: list[tuple[torch.Tensor, int, Node]] = []
    for graph_info in device_ir.graphs:
        if not isinstance(graph_info, ForLoopGraphInfo) or not set(
            graph_info.block_ids
        ) <= set(loop_block_ids):
            continue
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or node.target not in (load, store):
                continue
            tensor = _value(node.args[0])
            subscript = node.args[1]
            if tensor is None or not isinstance(subscript, (list, tuple)):
                continue
            positions = [
                pos
                for pos, item in enumerate(subscript)
                if isinstance(item, Node) and _value(item) is not None
            ]
            if not positions:
                continue
            if len(positions) != 1:
                return None
            (position,) = positions
            tensor_dim = sum(1 for item in subscript[:position] if item is not None)
            if tensor_dim >= tensor.ndim:
                return None
            stride = tensor.stride(tensor_dim)
            split = match_flat_gather_lane(subscript[position], env)
            if (
                split is None
                or split.lane_block_id not in lane_block_ids
                or not (isinstance(stride, int) and stride == 1)
                or tensor.dtype.itemsize > _CUTE_VECTOR_MAX_BYTES
            ):
                return None
            accesses.append((tensor, tensor_dim, split.row))
    if not accesses:
        return None
    vec_width = min(
        _CUTE_VECTOR_MAX_BYTES // tensor.dtype.itemsize for tensor, _, _ in accesses
    )
    if vec_width <= 1:
        return None
    environment = contextlib.nullcontext() if CompileEnvironment.has_current() else env
    with environment, host_function:
        for tensor, tensor_dim, row in accesses:
            if not row_is_multiple_of(row, vec_width, env, hints=True) or not (
                _packet_layout_aligned(env, tensor, tensor_dim, vec_width)
            ):
                return None
    return vec_width


def flat_gather_rows_hint(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    lane_block_ids: Collection[int],
    loop_block_ids: Collection[int],
    columns: int,
    rows: int,
) -> int | None:
    """The average number of gathered rows per root tile, from size hints.

    The gathered tensors' lane extent (``nnz * M`` elements for a flattened
    ``[nnz, M]`` input) divided by the feature extent and the number of root
    rows: a seed heuristic uses it to size the jagged thread count (a
    ``int`` row-width argument's hint is read for this estimate only).
    """
    from ...language.memory_ops import load
    from ...language.memory_ops import store
    from ..device_ir import ForLoopGraphInfo

    elements: int | None = None
    for graph_info in device_ir.graphs:
        if not isinstance(graph_info, ForLoopGraphInfo) or not set(
            graph_info.block_ids
        ) <= set(loop_block_ids):
            continue
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or node.target not in (load, store):
                continue
            tensor = _value(node.args[0])
            subscript = node.args[1]
            if tensor is None or not isinstance(subscript, (list, tuple)):
                continue
            for position, item in enumerate(subscript):
                split = match_flat_gather_lane(item, env)
                if split is None or split.lane_block_id not in lane_block_ids:
                    continue
                tensor_dim = sum(
                    1 for entry in subscript[:position] if entry is not None
                )
                if tensor_dim >= tensor.ndim:
                    continue
                size = tensor.shape[tensor_dim]
                hint = (
                    size
                    if isinstance(size, int)
                    else int(shape_env_size_hint(env.shape_env, size._sympy_()))
                )
                elements = hint if elements is None else min(elements, hint)
    if elements is None or columns <= 0 or rows <= 0:
        return None
    return max(1, elements // (columns * rows))


def flat_gather_row_width_hint(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    lane_block_ids: Collection[int],
    loop_block_ids: Collection[int],
) -> int | None:
    """The hinted row width of the gathers: the static or hinted scalar
    multiplying their row term (``M`` in ``(start + k) * M + tile_m.index``),
    for a seed heuristic sizing a jagged feature tile; None when the row
    term is not such a product."""
    from ...language.memory_ops import load
    from ...language.memory_ops import store
    from ..device_ir import ForLoopGraphInfo

    widths: set[int] = set()
    for graph_info in device_ir.graphs:
        if not isinstance(graph_info, ForLoopGraphInfo) or not set(
            graph_info.block_ids
        ) <= set(loop_block_ids):
            continue
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or node.target not in (load, store):
                continue
            subscript = node.args[1]
            if not isinstance(subscript, (list, tuple)):
                continue
            for item in subscript:
                split = match_flat_gather_lane(item, env)
                if split is None or split.lane_block_id not in lane_block_ids:
                    continue
                row = split.row
                while isinstance(row, Node) and _is_broadcast_view(row):
                    row = row.args[0]
                if not (
                    isinstance(row, Node)
                    and row.op == "call_function"
                    and row.target in _MUL_TARGETS
                    and len(row.args) == 2
                ):
                    continue
                for operand in row.args:
                    scalar = _scalar(operand)
                    if scalar is None:
                        continue
                    if isinstance(scalar, int):
                        widths.add(scalar)
                    else:
                        widths.add(
                            int(shape_env_size_hint(env.shape_env, scalar._sympy_()))
                        )
    return min(widths) if widths else None


def _packet_layout_aligned(
    env: CompileEnvironment, tensor: torch.Tensor, lane_dim: int, vec_width: int
) -> bool:
    """``cute_reduction_vector_layout_aligned`` for a seed heuristic.

    Seeds are generated before the binding snapshots its runtime input
    specializations, so the bound alignment residues the codegen reads are
    not available yet.  Read the example input's residue live through the
    registered alignment specialization instead (the snapshot pins the same
    residue into the cache key, so the codegen will prove what is read here).
    """
    from torch._subclasses import FakeTensor

    from ..compile_environment import _replay_tensor_input_source
    from .memory_ops import _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
    from .memory_ops import cute_reduction_vector_layout_aligned

    if cute_reduction_vector_layout_aligned(env, tensor, lane_dim, vec_width):
        return True
    key = _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
    if key in env.bound_runtime_input_specialization_results:
        return False
    specialization = env.runtime_input_specializations.get(key)
    owner = env.tensor_alignment_owner(tensor)
    if specialization is None or owner is None:
        return False
    source, offset_bytes = owner
    if source not in specialization.sources:
        return False
    value = _replay_tensor_input_source(source, env.runtime_arg_values_by_name)
    if not isinstance(value, torch.Tensor) or isinstance(value, FakeTensor):
        return False
    if (int(value.data_ptr()) + offset_bytes) % (vec_width * tensor.dtype.itemsize):
        return False
    for dim in range(tensor.ndim):
        if dim == lane_dim:
            continue
        size = tensor.shape[dim]
        if isinstance(size, int) and size == 1:
            continue
        stride = tensor.stride(dim)
        if not (isinstance(stride, int) and stride % vec_width == 0):
            return False
    return True


# ``int`` arguments that scale a flat gather's row (``(start + k) * max_M``)
# are keyed by type alone, so their residue modulo the widest packet is pinned
# into the bound kernel's cache key by this runtime input specialization; the
# row proof reads it back (``int_factor_residue``).
FLAT_GATHER_FACTOR_SPECIALIZATION_KEY = "cute_flat_gather_int_factor_residue_v1"
_FACTOR_MODULUS = 16


def _symbol_int_argument_source(
    env: CompileEnvironment, symbol: sympy.Basic
) -> LocalSource | None:
    """The ``int`` argument a size symbol stands for, if it is one."""
    if not isinstance(symbol, sympy.Symbol):
        return None
    sources = env.shape_env.var_to_sources.get(symbol)
    if not sources:
        return None
    source = sources[0]
    if not isinstance(source, LocalSource):
        return None
    from ..host_function import HostFunction

    # An ``int`` argument is traced as a symbol (a tensor's size symbols
    # have tensor-property sources, not a local one).
    value = HostFunction.current().params.arguments.get(source.local_name)
    if not isinstance(value, (int, torch.SymInt)) or isinstance(value, bool):
        return None
    return source


def _factor_residues(values: Sequence[object]) -> Hashable:
    return tuple(
        int(value) % _FACTOR_MODULUS if type(value) is int else None for value in values
    )


def _factor_prepared_guard(
    source_extractors: Sequence[Callable[[Sequence[object]], Hashable]],
    expected: Hashable,
) -> Callable[[Sequence[object]], bool]:
    """The prepared argument guard pins each argument's type; the value's
    residue is the one fact left to check."""
    from ..runtime_tensor_guards import _prepared_guard_source

    namespace: dict[str, object] = {}
    checks: list[str] = []
    for index, (extract, residue) in enumerate(
        zip(source_extractors, cast("tuple[object, ...]", expected), strict=True)
    ):
        if residue is None:
            continue
        source = _prepared_guard_source(namespace, f"extract_{index}", extract)
        checks.append(f"{source} % {_FACTOR_MODULUS} == {residue}")
    return eval(f"lambda args: {' and '.join(checks) or 'True'}", namespace)


def flat_gather_int_factor_sources(
    env: CompileEnvironment, device_ir: DeviceIR
) -> tuple[LocalSource, ...]:
    """The ``int`` arguments multiplying a flat gather's row term."""
    from ...language.memory_ops import load
    from ...language.memory_ops import store

    sources: dict[str, LocalSource] = {}

    def visit(value: object, depth: int) -> None:
        if depth > 64 or not isinstance(value, Node) or value.op != "call_function":
            return
        if _is_broadcast_view(value):
            visit(value.args[0], depth + 1)
            return
        if value.target in (*_ADD_TARGETS, *_SUB_TARGETS, *_MUL_TARGETS):
            for operand in value.args[:2]:
                scalar = _scalar(operand)
                if isinstance(scalar, torch.SymInt):
                    expr = scalar._sympy_()
                    if len(expr.free_symbols) == 1:
                        (symbol,) = expr.free_symbols
                        source = _symbol_int_argument_source(env, symbol)
                        if source is not None:
                            sources.setdefault(source.local_name, source)
                elif scalar is None:
                    visit(operand, depth + 1)

    for graph_info in device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or node.target not in (load, store):
                continue
            subscript = node.args[1]
            if not isinstance(subscript, (list, tuple)):
                continue
            for item in subscript:
                split = match_flat_gather_lane(item, env)
                if split is not None:
                    visit(split.row, 0)
    return tuple(sources[name] for name in sorted(sources))


def register_flat_gather_factor_specializations(
    env: CompileEnvironment, device_ir: DeviceIR
) -> None:
    """Pin the packet residue of every ``int`` argument scaling a flat gather."""
    from ..compile_environment import RuntimeInputSpecialization

    sources = flat_gather_int_factor_sources(env, device_ir)
    if not sources:
        return
    env.register_runtime_input_specialization(
        FLAT_GATHER_FACTOR_SPECIALIZATION_KEY,
        RuntimeInputSpecialization(
            sources=sources,
            classifier_identity=(
                "int_argument_residue_v1",
                _FACTOR_MODULUS,
                tuple(map(repr, sources)),
            ),
            classifier=_factor_residues,
            reusable_tensor_properties=frozenset(("int_value",)),
            prepared_guard_factory=_factor_prepared_guard,
        ),
    )


def int_factor_residue(env: CompileEnvironment, value: torch.SymInt) -> int | None:
    """``value % 16`` for an ``int`` argument whose residue the bound kernel
    pins (the snapshot after binding), or the live example's residue while
    the binding is still being built (seed generation)."""
    from ..compile_environment import _replay_tensor_input_source

    expr = value._sympy_()
    if len(expr.free_symbols) != 1:
        return None
    (symbol,) = expr.free_symbols
    if expr != symbol:
        return None
    specialization = env.runtime_input_specializations.get(
        FLAT_GATHER_FACTOR_SPECIALIZATION_KEY
    )
    source = _symbol_int_argument_source(env, symbol)
    if specialization is None or source is None or source not in specialization.sources:
        return None
    index = specialization.sources.index(source)
    results = env.bound_runtime_input_specialization_results.get(
        FLAT_GATHER_FACTOR_SPECIALIZATION_KEY
    )
    if results is not None:
        residue = cast("tuple[object, ...]", results)[index]
        return residue if isinstance(residue, int) else None
    live = _replay_tensor_input_source(source, env.runtime_arg_values_by_name)
    return int(live) % _FACTOR_MODULUS if type(live) is int else None
