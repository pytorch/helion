"""Discover short, local affine scans without selecting a lowering.

The recognized recurrence is deliberately algebraic::

    decayed = state * diagonal[None, :]
    prediction = sum(decayed * prediction_vector[None, :], dim=-1)
    residual = row_input - prediction
    row_update = update_scale * residual
    state = decayed + row_update[:, None] * update_vector[None, :]
    observation = sum(state * observation_vector[None, :], dim=-1)

The observation may already have been distributed over the affine state update
by :mod:`factor_affine_reductions`.  Both forms prove the same contract.  The
coefficient producers are intentionally opaque live-ins: discovery neither
names them nor constrains their formulas.

This module is discovery-only.  It does not annotate FX nodes, mutate a graph,
or install a backend plan.  A future lowering can consume the explicit region
boundary after independently proving its scheduling and numerical-policy
requirements.
"""

from __future__ import annotations

import dataclasses
import enum
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import view_ops
from ...language.matmul_ops import _static_dim_value
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo


_ADD_TARGETS = frozenset((operator.add, torch.ops.aten.add.Tensor))
_MUL_TARGETS = frozenset((operator.mul, torch.ops.aten.mul.Tensor))
_SUB_TARGETS = frozenset((operator.sub, torch.ops.aten.sub.Tensor))
_TRANSPARENT_TARGETS = frozenset(
    (torch.ops.aten.clone.default, torch.ops.aten.detach.default)
)
_STORAGE_DTYPES = frozenset((torch.float16, torch.bfloat16))
_MIN_STEPS = 2
_MAX_STEPS = 8
_PACK_WIDTH = 16


class ShortAffineScanAccessKind(enum.Enum):
    READ = "read"
    WRITE = "write"


@dataclasses.dataclass(frozen=True)
class ShortAffineScanAccess:
    """Immutable description of one exact memory operation."""

    node: torch.fx.Node = dataclasses.field(repr=False)
    kind: ShortAffineScanAccessKind
    base: torch.fx.Node = dataclasses.field(repr=False)
    indices: tuple[object, ...] = dataclasses.field(repr=False)
    value: object = dataclasses.field(repr=False)
    mask: object = dataclasses.field(repr=False)
    other: object = dataclasses.field(repr=False)
    kwargs: tuple[tuple[str, object], ...] = ()


@dataclasses.dataclass(frozen=True)
class ShortAffineScanDependencySlice:
    """Pure producer closure for one region-boundary value.

    ``nodes`` may overlap another slice when two boundary values share pure
    producers. Ownership is region-wide through ``producer_nodes``; slices are
    dependency views, not competing owners.
    """

    value: torch.fx.Node = dataclasses.field(repr=False)
    nodes: tuple[torch.fx.Node, ...] = dataclasses.field(repr=False)


@dataclasses.dataclass(frozen=True)
class ShortAffineScanAccessFence:
    """Nearest same-base accesses outside the claimed source interval."""

    base: torch.fx.Node = dataclasses.field(repr=False)
    preceding: ShortAffineScanAccess | None
    following: ShortAffineScanAccess | None


@dataclasses.dataclass(frozen=True)
class ShortAffineScanStep:
    """One proved transition in a :class:`ShortAffineScanRegion`."""

    diagonal: torch.fx.Node = dataclasses.field(repr=False)
    prediction_vector: torch.fx.Node = dataclasses.field(repr=False)
    row_input: torch.fx.Node = dataclasses.field(repr=False)
    update_scale: object = dataclasses.field(repr=False)
    update_vector: torch.fx.Node = dataclasses.field(repr=False)
    state: torch.fx.Node = dataclasses.field(repr=False)
    observation_vector: torch.fx.Node = dataclasses.field(repr=False)
    observation: torch.fx.Node = dataclasses.field(repr=False)
    output_base: torch.fx.Node = dataclasses.field(repr=False)
    output_effect: torch.fx.Node = dataclasses.field(repr=False)
    state_effect: torch.fx.Node = dataclasses.field(repr=False)
    output_access: ShortAffineScanAccess
    state_access: ShortAffineScanAccess
    owned_nodes: tuple[torch.fx.Node, ...] = dataclasses.field(repr=False)


@dataclasses.dataclass(frozen=True)
class ShortAffineScanRegion:
    """A closed state chain and the exact boundary needed to lower it safely.

    Base-node identity proves only the local FX access contract.  A lowering
    must separately prove runtime disjointness between ``state_base`` and each
    output base before reordering or combining memory operations.  The
    contiguous ``source_interval`` and its adjacent same-base access fences
    make the source-order boundary explicit; a splice must remain inside that
    span and preserve both fences.
    """

    graph_id: int
    row_extent: int
    feature_extent: int
    storage_dtype: torch.dtype
    state_base: torch.fx.Node = dataclasses.field(repr=False)
    output_bases: tuple[torch.fx.Node, ...] = dataclasses.field(repr=False)
    entry_load: torch.fx.Node = dataclasses.field(repr=False)
    entry_state: torch.fx.Node = dataclasses.field(repr=False)
    entry_access: ShortAffineScanAccess
    steps: tuple[ShortAffineScanStep, ...]
    source_interval: tuple[torch.fx.Node, ...] = dataclasses.field(repr=False)
    owned_nodes: tuple[torch.fx.Node, ...] = dataclasses.field(repr=False)
    live_outs: tuple[torch.fx.Node, ...] = dataclasses.field(repr=False)
    dependency_slices: tuple[ShortAffineScanDependencySlice, ...]
    producer_nodes: tuple[torch.fx.Node, ...] = dataclasses.field(repr=False)
    read_accesses: tuple[ShortAffineScanAccess, ...]
    write_accesses: tuple[ShortAffineScanAccess, ...]
    read_bases: tuple[torch.fx.Node, ...] = dataclasses.field(repr=False)
    write_bases: tuple[torch.fx.Node, ...] = dataclasses.field(repr=False)
    access_fences: tuple[ShortAffineScanAccessFence, ...]

    @property
    def step_count(self) -> int:
        return len(self.steps)


@dataclasses.dataclass(frozen=True)
class _StepMatch:
    diagonal: torch.fx.Node
    prediction_vector: torch.fx.Node
    row_input: torch.fx.Node
    update_scale: object
    update_vector: torch.fx.Node
    state: torch.fx.Node
    observation_vector: torch.fx.Node
    observation: torch.fx.Node
    output_base: torch.fx.Node
    output_effect: torch.fx.Node
    state_effect: torch.fx.Node
    output_access: ShortAffineScanAccess
    state_access: ShortAffineScanAccess
    owned: frozenset[torch.fx.Node]


def _tensor(node: object) -> torch.Tensor | None:
    if not isinstance(node, torch.fx.Node):
        return None
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _static_shape(node: object) -> tuple[int, ...] | None:
    value = _tensor(node)
    if value is None:
        return None
    result = tuple(_static_int(size) for size in value.shape)
    if any(size is None for size in result):
        return None
    return tuple(size for size in result if size is not None)


def _static_int(value: object) -> int | None:
    """Resolve a tile extent from the selected config before shape hints."""

    if isinstance(value, int):
        return value
    if not isinstance(value, torch.SymInt) or not CompileEnvironment.has_current():
        return None
    env = CompileEnvironment.current()
    try:
        block_id = env.get_block_id(value)
    except (AttributeError, AssertionError, LookupError, RuntimeError, ValueError):
        block_id = None
    if block_id is not None:
        from ..device_function import DeviceFunction

        try:
            configured = DeviceFunction.current().resolved_block_size(block_id)
        except (AttributeError, AssertionError, LookupError, RuntimeError, ValueError):
            configured = None
        if isinstance(configured, int):
            return configured
        if isinstance(configured, torch.SymInt):
            try:
                return _static_dim_value(env, configured)
            except (AssertionError, LookupError, RuntimeError, ValueError):
                return None
    try:
        return _static_dim_value(env, value)
    except (AssertionError, LookupError, RuntimeError, ValueError):
        return None


def _is_fp32_shape(node: object, shape: tuple[int, ...]) -> bool:
    value = _tensor(node)
    return (
        value is not None
        and value.dtype is torch.float32
        and _static_shape(node) == shape
    )


def _is_scalar(node: object) -> bool:
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return True
    if _is_fp32_shape(node, ()):
        return True
    value = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
    return isinstance(value, torch.SymFloat)


def _access_descriptor(node: torch.fx.Node) -> ShortAffineScanAccess | None:
    """Capture one supported load/store exactly enough for later replay."""

    if (
        node.op != "call_function"
        or node.target not in (memory_ops.load, memory_ops.store)
        or len(node.args) != 4
        or node.kwargs
        or not isinstance(node.args[0], torch.fx.Node)
        or not isinstance(node.args[1], (list, tuple))
    ):
        return None
    base, raw_indices, third, fourth = node.args
    assert isinstance(base, torch.fx.Node)
    indices = tuple(
        _freeze_access_value(index) for index in cast("Sequence[object]", raw_indices)
    )
    if node.target is memory_ops.load:
        return ShortAffineScanAccess(
            node=node,
            kind=ShortAffineScanAccessKind.READ,
            base=base,
            indices=indices,
            value=None,
            mask=_freeze_access_value(third),
            other=_freeze_access_value(fourth),
        )
    return ShortAffineScanAccess(
        node=node,
        kind=ShortAffineScanAccessKind.WRITE,
        base=base,
        indices=indices,
        value=_freeze_access_value(third),
        mask=_freeze_access_value(fourth),
        other=None,
    )


def _freeze_access_value(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_access_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            (key, _freeze_access_value(item))
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
    return value


def _binary_pair(
    node: object, targets: frozenset[object]
) -> tuple[object, object] | None:
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target not in targets
        or len(node.args) != 2
        or node.kwargs
    ):
        return None
    return node.args[0], node.args[1]


def _transparent_source(
    node: object,
) -> tuple[torch.fx.Node, tuple[torch.fx.Node, ...]] | None:
    """Strip clone/detach nodes while retaining them as owned wrappers."""

    if not isinstance(node, torch.fx.Node):
        return None
    current = node
    wrappers: list[torch.fx.Node] = []
    for _ in range(8):
        if current.op != "call_function" or current.target not in _TRANSPARENT_TARGETS:
            return current, tuple(wrappers)
        if (
            len(current.args) != 1
            or current.kwargs
            or not isinstance(current.args[0], torch.fx.Node)
        ):
            return None
        wrappers.append(current)
        current = current.args[0]
    return None


def _storage_to_fp32(
    node: object,
) -> tuple[torch.fx.Node, torch.dtype, frozenset[torch.fx.Node]] | None:
    stripped = _transparent_source(node)
    if stripped is None:
        return None
    cast, outer = stripped
    if (
        cast.op != "call_function"
        or cast.target is not torch.ops.prims.convert_element_type.default
        or len(cast.args) != 2
        or cast.args[1] is not torch.float32
        or cast.kwargs
    ):
        return None
    inner = _transparent_source(cast.args[0])
    if inner is None:
        return None
    source, inner_wrappers = inner
    source_value = _tensor(source)
    cast_value = _tensor(cast)
    if (
        source_value is None
        or source_value.dtype not in _STORAGE_DTYPES
        or cast_value is None
        or cast_value.dtype is not torch.float32
    ):
        return None
    return source, source_value.dtype, frozenset((*outer, cast, *inner_wrappers))


def _fp32_to_storage(
    node: object, dtype: torch.dtype
) -> tuple[torch.fx.Node, frozenset[torch.fx.Node]] | None:
    stripped = _transparent_source(node)
    if stripped is None:
        return None
    cast, outer = stripped
    if (
        cast.op != "call_function"
        or cast.target is not torch.ops.prims.convert_element_type.default
        or len(cast.args) != 2
        or cast.args[1] is not dtype
        or cast.kwargs
        or not isinstance(cast.args[0], torch.fx.Node)
    ):
        return None
    source = cast.args[0]
    source_value = _tensor(source)
    cast_value = _tensor(cast)
    if (
        source_value is None
        or source_value.dtype is not torch.float32
        or cast_value is None
        or cast_value.dtype is not dtype
    ):
        return None
    return source, frozenset((*outer, cast))


def _broadcast(
    node: object,
    indices: tuple[object, ...],
    output_shape: tuple[int, ...],
) -> tuple[torch.fx.Node, frozenset[torch.fx.Node]] | None:
    stripped = _transparent_source(node)
    if stripped is None:
        return None
    view, wrappers = stripped
    if (
        view.op != "call_function"
        or view.target is not view_ops.subscript
        or len(view.args) != 2
        or view.kwargs
        or not isinstance(view.args[0], torch.fx.Node)
        or not isinstance(view.args[1], (list, tuple))
        or len(view.args[1]) != len(indices)
        or _static_shape(view) != output_shape
    ):
        return None
    for actual, expected in zip(view.args[1], indices, strict=True):
        if isinstance(expected, slice):
            if actual != slice(None):
                return None
        elif actual is not expected:
            return None
    return view.args[0], frozenset((*wrappers, view))


def _sum_input(
    node: object, input_shape: tuple[int, int]
) -> tuple[torch.fx.Node, frozenset[torch.fx.Node]] | None:
    stripped = _transparent_source(node)
    if stripped is None:
        return None
    reduction, wrappers = stripped
    if (
        reduction.op != "call_function"
        or reduction.target is not torch.ops.aten.sum.dim_IntList
        or len(reduction.args) not in (2, 3)
        or set(reduction.kwargs) - {"keepdim", "dtype"}
        or reduction.kwargs.get("dtype") is not None
        or not isinstance(reduction.args[0], torch.fx.Node)
        or not isinstance(reduction.args[1], (list, tuple))
    ):
        return None
    dims = tuple(reduction.args[1])
    keepdim = (
        reduction.args[2]
        if len(reduction.args) == 3
        else reduction.kwargs.get("keepdim", False)
    )
    if dims not in ((-1,), (1,)) or keepdim is not False:
        return None
    source = reduction.args[0]
    owned = set(wrappers)
    owned.add(reduction)
    if (
        source.op == "call_function"
        and source.target is _tracing_ops._mask_to
        and len(source.args) == 2
        and source.args[1] == 0
        and not source.kwargs
        and isinstance(source.args[0], torch.fx.Node)
    ):
        owned.add(source)
        source = source.args[0]
    if not _is_fp32_shape(source, input_shape):
        return None
    return source, frozenset(owned)


def _match_reduction_product(
    node: object,
    matrix: torch.fx.Node,
    rows: int,
    features: int,
) -> tuple[torch.fx.Node, torch.fx.Node, frozenset[torch.fx.Node]] | None:
    matched_sum = _sum_input(node, (rows, features))
    if matched_sum is None:
        return None
    product, owned = matched_sum
    pair = _binary_pair(product, _MUL_TARGETS)
    if pair is None:
        return None
    for maybe_matrix, maybe_vector_view in (pair, pair[::-1]):
        if maybe_matrix is not matrix:
            continue
        vector = _broadcast(maybe_vector_view, (None, slice(None)), (1, features))
        if vector is None:
            continue
        source, view_nodes = vector
        if not _is_fp32_shape(source, (features,)):
            continue
        return source, product, owned | view_nodes | {product}
    return None


def _unwrap_optional_scale(
    node: torch.fx.Node, rows: int
) -> tuple[torch.fx.Node, object, frozenset[torch.fx.Node]] | None:
    if not _is_fp32_shape(node, (rows,)):
        return None
    pair = _binary_pair(node, _MUL_TARGETS)
    if pair is not None:
        for value, scale in (pair, pair[::-1]):
            if isinstance(value, torch.fx.Node) and _is_scalar(scale):
                return value, scale, frozenset((node,))
    return node, 1.0, frozenset()


def _match_direct_observation(
    node: torch.fx.Node,
    state: torch.fx.Node,
    rows: int,
    features: int,
) -> tuple[torch.fx.Node, torch.fx.Node, frozenset[torch.fx.Node]] | None:
    matched = _match_reduction_product(node, state, rows, features)
    if matched is None:
        return None
    vector, _, owned = matched
    return vector, node, owned


def _match_factored_observation(
    node: torch.fx.Node,
    decayed: torch.fx.Node,
    row_update: torch.fx.Node,
    update_vector_view: torch.fx.Node,
    rows: int,
    features: int,
) -> tuple[torch.fx.Node, torch.fx.Node, frozenset[torch.fx.Node]] | None:
    outer = _binary_pair(node, _ADD_TARGETS)
    if outer is None:
        return None
    for base_sum, scaled_sum in (outer, outer[::-1]):
        base = _match_reduction_product(base_sum, decayed, rows, features)
        scaled = _sum_input(scaled_sum, (rows, 1))
        if base is None or scaled is None:
            continue
        observation_vector, _, base_owned = base
        scaled_product, scaled_owned = scaled
        scaled_pair = _binary_pair(scaled_product, _MUL_TARGETS)
        if scaled_pair is None:
            continue
        dot_sum_view: torch.fx.Node | None = None
        row_view_nodes: frozenset[torch.fx.Node] | None = None
        for maybe_row, maybe_dot in (scaled_pair, scaled_pair[::-1]):
            row_view = _broadcast(maybe_row, (slice(None), None), (rows, 1))
            if row_view is not None and row_view[0] is row_update:
                if isinstance(maybe_dot, torch.fx.Node):
                    dot_sum_view = maybe_dot
                    row_view_nodes = row_view[1]
                break
        if dot_sum_view is None or row_view_nodes is None:
            continue
        dot_view = _broadcast(dot_sum_view, (slice(None), None), (rows, 1))
        if dot_view is None:
            continue
        dot_sum, dot_view_nodes = dot_view
        dot_reduction = _sum_input(dot_sum, (rows, features))
        if dot_reduction is None:
            continue
        dot_product, dot_owned = dot_reduction
        expanded_owned: set[torch.fx.Node] = set()
        if (
            dot_product.op == "call_function"
            and dot_product.target is torch.ops.aten.expand.default
            and len(dot_product.args) == 2
            and not dot_product.kwargs
            and isinstance(dot_product.args[0], torch.fx.Node)
        ):
            expanded_owned.add(dot_product)
            dot_product = dot_product.args[0]
        dot_pair = _binary_pair(dot_product, _MUL_TARGETS)
        if dot_pair is None:
            continue
        observation_view = next(
            (
                candidate
                for candidate in dot_pair
                if isinstance(candidate, torch.fx.Node)
                and _broadcast(candidate, (None, slice(None)), (1, features))
                == (observation_vector, frozenset((candidate,)))
            ),
            None,
        )
        if observation_view is None or update_vector_view not in dot_pair:
            continue
        owned = (
            base_owned
            | scaled_owned
            | row_view_nodes
            | dot_view_nodes
            | dot_owned
            | expanded_owned
            | {node, scaled_product, dot_product, observation_view}
        )
        return observation_vector, node, frozenset(owned)
    return None


def _match_observation_store(
    store: torch.fx.Node,
    *,
    state_base: object,
    state: torch.fx.Node,
    decayed: torch.fx.Node,
    row_update: torch.fx.Node,
    update_vector_view: torch.fx.Node,
    rows: int,
    features: int,
    storage_dtype: torch.dtype,
) -> (
    tuple[
        torch.fx.Node,
        torch.fx.Node,
        frozenset[torch.fx.Node],
    ]
    | None
):
    if (
        store.op != "call_function"
        or store.target is not memory_ops.store
        or len(store.args) < 3
        or not isinstance(store.args[0], torch.fx.Node)
        or store.args[0] is state_base
    ):
        return None
    converted = _fp32_to_storage(store.args[2], storage_dtype)
    if converted is None:
        return None
    stored, cast_nodes = converted
    unscaled = _unwrap_optional_scale(stored, rows)
    if unscaled is None:
        return None
    observation, _, scale_nodes = unscaled
    direct = _match_direct_observation(observation, state, rows, features)
    if direct is not None:
        vector, result, observation_nodes = direct
        return (
            vector,
            result,
            cast_nodes | scale_nodes | observation_nodes | {store},
        )
    factored = _match_factored_observation(
        observation,
        decayed,
        row_update,
        update_vector_view,
        rows,
        features,
    )
    if factored is None:
        return None
    vector, result, observation_nodes = factored
    return (
        vector,
        result,
        cast_nodes | scale_nodes | observation_nodes | {store},
    )


def _match_step(
    state_store: torch.fx.Node,
    output_stores: Sequence[torch.fx.Node],
    incoming_state: torch.fx.Node,
    state_base: object,
    rows: int,
    features: int,
    storage_dtype: torch.dtype,
) -> _StepMatch | None:
    state_access = _access_descriptor(state_store)
    if (
        state_access is None
        or state_access.kind is not ShortAffineScanAccessKind.WRITE
        or state_access.base is not state_base
    ):
        return None
    converted = _fp32_to_storage(state_access.value, storage_dtype)
    if converted is None:
        return None
    state, state_cast_nodes = converted
    if not _is_fp32_shape(state, (rows, features)):
        return None
    update_pair = _binary_pair(state, _ADD_TARGETS)
    if update_pair is None:
        return None

    for decayed, outer in (update_pair, update_pair[::-1]):
        if not isinstance(decayed, torch.fx.Node) or not isinstance(
            outer, torch.fx.Node
        ):
            continue
        decay_pair = _binary_pair(decayed, _MUL_TARGETS)
        outer_pair = _binary_pair(outer, _MUL_TARGETS)
        if decay_pair is None or outer_pair is None:
            continue
        diagonal: torch.fx.Node | None = None
        decay_nodes: frozenset[torch.fx.Node] | None = None
        for maybe_state, maybe_diagonal in (decay_pair, decay_pair[::-1]):
            if maybe_state is not incoming_state:
                continue
            diagonal_view = _broadcast(
                maybe_diagonal, (None, slice(None)), (1, features)
            )
            if diagonal_view is not None and _is_fp32_shape(
                diagonal_view[0], (features,)
            ):
                diagonal, decay_nodes = diagonal_view
                break
        if diagonal is None or decay_nodes is None:
            continue

        row_update: torch.fx.Node | None = None
        update_vector: torch.fx.Node | None = None
        row_view_nodes: frozenset[torch.fx.Node] | None = None
        vector_view_nodes: frozenset[torch.fx.Node] | None = None
        update_vector_view: torch.fx.Node | None = None
        for maybe_row, maybe_vector in (outer_pair, outer_pair[::-1]):
            row_view = _broadcast(maybe_row, (slice(None), None), (rows, 1))
            vector_view = _broadcast(maybe_vector, (None, slice(None)), (1, features))
            if (
                row_view is not None
                and vector_view is not None
                and _is_fp32_shape(row_view[0], (rows,))
                and _is_fp32_shape(vector_view[0], (features,))
            ):
                row_update = row_view[0]
                update_vector = vector_view[0]
                row_view_nodes = row_view[1]
                vector_view_nodes = vector_view[1]
                if isinstance(maybe_vector, torch.fx.Node):
                    update_vector_view = maybe_vector
                break
        if any(
            item is None
            for item in (
                row_update,
                update_vector,
                row_view_nodes,
                vector_view_nodes,
                update_vector_view,
            )
        ):
            continue
        assert row_update is not None
        assert update_vector is not None
        assert row_view_nodes is not None
        assert vector_view_nodes is not None
        assert update_vector_view is not None

        scaled_residual = _binary_pair(row_update, _MUL_TARGETS)
        if scaled_residual is None:
            continue
        residual: torch.fx.Node | None = None
        update_scale: object | None = None
        for maybe_residual, maybe_scale in (
            scaled_residual,
            scaled_residual[::-1],
        ):
            if (
                isinstance(maybe_residual, torch.fx.Node)
                and _binary_pair(maybe_residual, _SUB_TARGETS) is not None
                and _is_scalar(maybe_scale)
            ):
                residual = maybe_residual
                update_scale = maybe_scale
                break
        if residual is None or update_scale is None:
            continue
        difference = _binary_pair(residual, _SUB_TARGETS)
        assert difference is not None
        row_input, prediction = difference
        if (
            not isinstance(row_input, torch.fx.Node)
            or not isinstance(prediction, torch.fx.Node)
            or not _is_fp32_shape(row_input, (rows,))
            or not _is_fp32_shape(prediction, (rows,))
            or not _is_fp32_shape(residual, (rows,))
            or not _is_fp32_shape(row_update, (rows,))
        ):
            continue
        prediction_match = _match_reduction_product(prediction, decayed, rows, features)
        if prediction_match is None:
            continue
        prediction_vector, _, prediction_nodes = prediction_match

        output_matches = [
            (store, matched)
            for store in output_stores
            if (
                matched := _match_observation_store(
                    store,
                    state_base=state_base,
                    state=state,
                    decayed=decayed,
                    row_update=row_update,
                    update_vector_view=update_vector_view,
                    rows=rows,
                    features=features,
                    storage_dtype=storage_dtype,
                )
            )
            is not None
        ]
        if len(output_matches) != 1:
            continue
        output_store, output_match = output_matches[0]
        output_access = _access_descriptor(output_store)
        if output_access is None:
            continue
        (
            observation_vector,
            observation,
            output_nodes,
        ) = output_match
        owned = (
            state_cast_nodes
            | decay_nodes
            | row_view_nodes
            | vector_view_nodes
            | prediction_nodes
            | output_nodes
            | {decayed, outer, residual, row_update, state, state_store}
        )
        return _StepMatch(
            diagonal=diagonal,
            prediction_vector=prediction_vector,
            row_input=row_input,
            update_scale=update_scale,
            update_vector=update_vector,
            state=state,
            observation_vector=observation_vector,
            observation=observation,
            output_base=cast("torch.fx.Node", output_store.args[0]),
            output_effect=output_store,
            state_effect=state_store,
            output_access=output_access,
            state_access=state_access,
            owned=frozenset(owned),
        )
    return None


def _input_nodes(node: torch.fx.Node) -> set[torch.fx.Node]:
    result: set[torch.fx.Node] = set()
    torch.fx.map_arg((node.args, node.kwargs), lambda value: result.add(value))
    return result


def _node_is_effectful(node: torch.fx.Node) -> bool:
    if node.op in {"call_method", "call_module"}:
        return True
    return node.op == "call_function" and node.is_impure()


def _same_memory_base(node: torch.fx.Node, base: object) -> bool:
    return (
        node.op == "call_function"
        and node.target in (memory_ops.load, memory_ops.store)
        and bool(node.args)
        and node.args[0] is base
    )


def _ordered(
    nodes: set[torch.fx.Node], positions: dict[torch.fx.Node, int]
) -> tuple[torch.fx.Node, ...]:
    return tuple(sorted(nodes, key=positions.__getitem__))


class _UnsupportedProducer(Exception):
    pass


def _is_external_source(node: torch.fx.Node) -> bool:
    return node.op in {"placeholder", "get_attr"} or (
        node.op == "call_function"
        and node.target in (_tracing_ops._host_tensor, _tracing_ops._get_symnode)
    )


def _ancestral_read_accesses(
    roots: set[torch.fx.Node], owned: set[torch.fx.Node]
) -> dict[torch.fx.Node, ShortAffineScanAccess]:
    reads: dict[torch.fx.Node, ShortAffineScanAccess] = {}
    visited: set[torch.fx.Node] = set()

    def visit(node: torch.fx.Node) -> None:
        if node in visited or node in owned:
            return
        visited.add(node)
        if node.op == "call_function" and node.target is memory_ops.load:
            access = _access_descriptor(node)
            if access is None:
                raise _UnsupportedProducer
            reads[node] = access
        for source in node.all_input_nodes:
            visit(source)

    for root in roots:
        visit(root)
    return reads


def _producer_closure(
    value: torch.fx.Node,
    owned: set[torch.fx.Node],
    interval: set[torch.fx.Node],
    positions: dict[torch.fx.Node, int],
    anchor: torch.fx.Node,
) -> set[torch.fx.Node]:
    producer_nodes: set[torch.fx.Node] = set()
    visited: set[torch.fx.Node] = set()

    def visit(node: torch.fx.Node) -> None:
        if node in visited or node in owned:
            return
        visited.add(node)
        if node not in interval or _is_external_source(node):
            return
        if positions[node] >= positions[anchor] or _node_is_effectful(node):
            raise _UnsupportedProducer
        if node.op != "call_function":
            raise _UnsupportedProducer
        producer_nodes.add(node)
        for source in node.all_input_nodes:
            visit(source)

    visit(value)
    return producer_nodes


def _dependency_slices(
    live_ins: set[torch.fx.Node],
    owned: set[torch.fx.Node],
    interval: tuple[torch.fx.Node, ...],
    positions: dict[torch.fx.Node, int],
) -> (
    tuple[
        tuple[ShortAffineScanDependencySlice, ...],
        tuple[torch.fx.Node, ...],
        tuple[ShortAffineScanAccess, ...],
    ]
    | None
):
    """Describe when each boundary value is produced without claiming its DAG."""

    interval_set = set(interval)
    slices: list[ShortAffineScanDependencySlice] = []
    all_producers: set[torch.fx.Node] = set()
    all_reads: dict[torch.fx.Node, ShortAffineScanAccess] = {}
    for value in _ordered(live_ins, positions):
        consumers = _ordered(
            {node for node in owned if value in _input_nodes(node)}, positions
        )
        if not consumers:
            return None
        anchor = consumers[0]
        try:
            producer_nodes = _producer_closure(
                value, owned, interval_set, positions, anchor
            )
            reads = _ancestral_read_accesses({value}, owned)
        except _UnsupportedProducer:
            return None
        ordered_nodes = _ordered(producer_nodes, positions)
        slices.append(
            ShortAffineScanDependencySlice(
                value=value,
                nodes=ordered_nodes,
            )
        )
        all_producers.update(producer_nodes)
        all_reads.update(reads)
    return (
        tuple(slices),
        _ordered(all_producers, positions),
        tuple(all_reads[node] for node in _ordered(set(all_reads), positions)),
    )


def _unique_bases(
    accesses: Sequence[ShortAffineScanAccess],
) -> tuple[torch.fx.Node, ...]:
    result: list[torch.fx.Node] = []
    seen: set[torch.fx.Node] = set()
    for access in accesses:
        if access.base not in seen:
            seen.add(access.base)
            result.append(access.base)
    return tuple(result)


def _access_fences(
    graph_nodes: Sequence[torch.fx.Node],
    bases: Sequence[torch.fx.Node],
    first_position: int,
    last_position: int,
) -> tuple[ShortAffineScanAccessFence, ...] | None:
    positions = {node: index for index, node in enumerate(graph_nodes)}
    result: list[ShortAffineScanAccessFence] = []
    for base in bases:
        accesses: list[ShortAffineScanAccess] = []
        for node in graph_nodes:
            if not _same_memory_base(node, base):
                continue
            access = _access_descriptor(node)
            if access is None:
                return None
            accesses.append(access)
        preceding = next(
            (
                access
                for access in reversed(accesses)
                if positions[access.node] < first_position
            ),
            None,
        )
        following = next(
            (access for access in accesses if positions[access.node] > last_position),
            None,
        )
        result.append(
            ShortAffineScanAccessFence(
                base=base,
                preceding=preceding,
                following=following,
            )
        )
    return tuple(result)


def _region_from_entry(
    info: GraphInfo,
    entry_state: torch.fx.Node,
    entry_load: torch.fx.Node,
    storage_dtype: torch.dtype,
    entry_wrappers: frozenset[torch.fx.Node],
) -> ShortAffineScanRegion | None:
    from ..device_ir import RootGraphInfo

    entry_access = _access_descriptor(entry_load)
    if (
        not isinstance(info, RootGraphInfo)
        or entry_access is None
        or entry_access.kind is not ShortAffineScanAccessKind.READ
    ):
        return None
    shape = _static_shape(entry_state)
    if shape is None or len(shape) != 2:
        return None
    rows, features = shape
    if rows <= 0 or features <= 0 or rows % _PACK_WIDTH or features % _PACK_WIDTH:
        return None
    graph_nodes = list(info.graph.nodes)
    positions = {node: index for index, node in enumerate(graph_nodes)}
    state_base = entry_load.args[0] if entry_load.args else None
    if not isinstance(state_base, torch.fx.Node):
        return None
    stores = [
        node
        for node in graph_nodes
        if node.op == "call_function" and node.target is memory_ops.store
    ]
    state_stores = [node for node in stores if node.args and node.args[0] is state_base]
    output_stores = [node for node in stores if node not in state_stores]
    incoming = entry_state
    previous_position = positions[entry_state]
    matches: list[_StepMatch] = []
    while len(matches) <= _MAX_STEPS:
        candidates = [
            matched
            for store in state_stores
            if positions[store] > previous_position
            and (
                matched := _match_step(
                    store,
                    output_stores,
                    incoming,
                    state_base,
                    rows,
                    features,
                    storage_dtype,
                )
            )
            is not None
        ]
        if not candidates:
            break
        if len(candidates) != 1:
            return None
        matched = candidates[0]
        matches.append(matched)
        incoming = matched.state
        previous_position = positions[matched.state_effect]
    if not _MIN_STEPS <= len(matches) <= _MAX_STEPS:
        return None

    owned: set[torch.fx.Node] = {entry_load, entry_state, *entry_wrappers}
    effects: set[torch.fx.Node] = set()
    for matched in matches:
        if owned.intersection(matched.owned):
            return None
        owned.update(matched.owned)
        effects.update((matched.output_effect, matched.state_effect))
    first_position = min(positions[node] for node in owned)
    last_position = max(positions[node] for node in owned)
    interval = tuple(graph_nodes[first_position : last_position + 1])
    if any(_node_is_effectful(node) and node not in effects for node in interval):
        return None

    live_outs: set[torch.fx.Node] = set()
    for node in owned:
        outside_users = [user for user in node.users if user not in owned]
        if any(user.op != "output" for user in outside_users):
            return None
        if outside_users:
            live_outs.add(node)
    live_ins = {
        source for node in owned for source in _input_nodes(node) if source not in owned
    }
    ordered_owned = _ordered(owned, positions)
    ordered_effects = _ordered(effects, positions)
    if any(
        max(
            positions[matches[index].output_effect],
            positions[matches[index].state_effect],
        )
        >= min(
            positions[matches[index + 1].output_effect],
            positions[matches[index + 1].state_effect],
        )
        for index in range(len(matches) - 1)
    ):
        return None
    dependency_match = _dependency_slices(live_ins, owned, interval, positions)
    if dependency_match is None:
        return None
    dependency_slices, producer_nodes, producer_reads = dependency_match
    replacement_nodes = {*owned, *producer_nodes}
    if any(
        user not in replacement_nodes for node in producer_nodes for user in node.users
    ):
        return None
    interval_reads: list[ShortAffineScanAccess] = []
    for node in interval:
        if node.op != "call_function" or node.target is not memory_ops.load:
            continue
        access = _access_descriptor(node)
        if access is None:
            return None
        interval_reads.append(access)
    reads_by_node = {
        access.node: access for access in (*producer_reads, *interval_reads)
    }
    read_accesses = tuple(
        reads_by_node[node] for node in _ordered(set(reads_by_node), positions)
    )
    write_accesses = tuple(
        access
        for node in ordered_effects
        for access in (_access_descriptor(node),)
        if access is not None
    )
    if len(write_accesses) != len(ordered_effects):
        return None
    read_bases = _unique_bases(read_accesses)
    write_bases = _unique_bases(write_accesses)
    write_base_set = set(write_bases)
    known_access_nodes = {entry_load, *effects}
    if any(
        access.base in write_base_set and access.node not in known_access_nodes
        for access in interval_reads
    ):
        return None
    fences = _access_fences(
        graph_nodes,
        (*read_bases, *(base for base in write_bases if base not in set(read_bases))),
        first_position,
        last_position,
    )
    if fences is None:
        return None
    steps = tuple(
        ShortAffineScanStep(
            diagonal=matched.diagonal,
            prediction_vector=matched.prediction_vector,
            row_input=matched.row_input,
            update_scale=matched.update_scale,
            update_vector=matched.update_vector,
            state=matched.state,
            observation_vector=matched.observation_vector,
            observation=matched.observation,
            output_base=matched.output_base,
            output_effect=matched.output_effect,
            state_effect=matched.state_effect,
            output_access=matched.output_access,
            state_access=matched.state_access,
            owned_nodes=_ordered(set(matched.owned), positions),
        )
        for matched in matches
    )
    return ShortAffineScanRegion(
        graph_id=info.graph_id,
        row_extent=rows,
        feature_extent=features,
        storage_dtype=storage_dtype,
        state_base=state_base,
        output_bases=tuple(matched.output_base for matched in matches),
        entry_load=entry_load,
        entry_state=entry_state,
        entry_access=entry_access,
        steps=steps,
        source_interval=interval,
        owned_nodes=ordered_owned,
        live_outs=_ordered(live_outs, positions),
        dependency_slices=dependency_slices,
        producer_nodes=producer_nodes,
        read_accesses=read_accesses,
        write_accesses=write_accesses,
        read_bases=read_bases,
        write_bases=write_bases,
        access_fences=fences,
    )


def discover_short_affine_scan_regions(
    graphs: Sequence[GraphInfo],
) -> tuple[ShortAffineScanRegion, ...]:
    """Return all non-overlapping, statically bounded affine scan regions.

    Discovery is fail-closed and side-effect free. Coefficient producer DAGs
    are recorded as dependency slices rooted at their boundary values.
    """

    regions: list[ShortAffineScanRegion] = []
    claimed_intervals: set[torch.fx.Node] = set()
    for info in graphs:
        for node in info.graph.nodes:
            converted = _storage_to_fp32(node)
            if converted is None:
                continue
            entry_load, storage_dtype, wrappers = converted
            if (
                entry_load.op != "call_function"
                or entry_load.target is not memory_ops.load
                or entry_load in claimed_intervals
            ):
                continue
            region = _region_from_entry(info, node, entry_load, storage_dtype, wrappers)
            if region is None or claimed_intervals.intersection(region.source_interval):
                continue
            regions.append(region)
            claimed_intervals.update(region.source_interval)
    return tuple(regions)


__all__ = [
    "ShortAffineScanAccess",
    "ShortAffineScanAccessFence",
    "ShortAffineScanAccessKind",
    "ShortAffineScanDependencySlice",
    "ShortAffineScanRegion",
    "ShortAffineScanStep",
    "discover_short_affine_scan_regions",
]
