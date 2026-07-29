"""Generic live-FX plans for tcgen05 register-fragment epilogues.

The plan is a validated slice of the existing FX graph.  It does not encode an
expression or recognize operation sequences: each owned node is admitted by
its own lowering capability, and the fragment evaluator executes those nodes
with their normal semantics.
"""

from __future__ import annotations

import dataclasses
import enum
import operator
from typing import TYPE_CHECKING

import torch
from torch.fx.node import Node
from torch.fx.node import map_arg

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import tile_index
from ...language import view_ops
from ..compile_environment import CompileEnvironment
from ..indexing_strategy import exact_tile_block_ids
from .cute_fx_walk import build_inner_outputs_index_from_graphs
from .cute_fx_walk import reach_matmul_anchors
from .cute_fx_walk import resolve_tcgen05_accumulator_boundary
from .cute_reshape import _get_tile_shape
from .cute_reshape import broadcast_logical_flat_index
from .cute_reshape import logical_coords_from_flat
from .cute_reshape import logical_flat_from_coords
from .cute_reshape import resolve_cute_logical_coordinate
from .tcgen05_constants import TCGEN05_FRAGMENT_PAIR_WIDTH

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...runtime.config import Config
    from ..device_ir import GraphInfo


class Tcgen05EpilogueLoadScope(enum.Enum):
    """How an external tensor is made available to a fragment evaluation."""

    OUTPUT_ALIGNED_SUBTILE = "output_aligned_subtile"
    DIRECT = "direct"


class Tcgen05PairTraversal(enum.Enum):
    """Physical register traversal proven by the centralized MMA gate."""

    CONTIGUOUS_EVEN_ODD_N_R2S = "contiguous_even_odd_n_r2s"


@dataclasses.dataclass(frozen=True)
class Tcgen05PairLocalCapability:
    """Proof that one physical register pair contains every boundary demand."""

    width: int
    traversal: Tcgen05PairTraversal


@dataclasses.dataclass(frozen=True)
class Tcgen05EpilogueLoadPlan:
    load_node: Node
    host_tensor_fx_node: Node
    host_tensor_val: torch.Tensor
    store_value_node: Node
    scope: Tcgen05EpilogueLoadScope
    broadcast_axis: int | None = None


@dataclasses.dataclass(frozen=True)
class Tcgen05EpilogueStorePlan:
    store_node: Node
    value_node: Node
    output_tensor: torch.Tensor
    output_block_ids: tuple[int, ...]
    boundary_nodes: tuple[Node, ...]
    load_plans: tuple[Tcgen05EpilogueLoadPlan, ...]
    requires_scalar_fragment: bool
    required_pair_width: int | None
    pair_local: Tcgen05PairLocalCapability | None = None


@dataclasses.dataclass(frozen=True)
class Tcgen05EpiloguePlan:
    anchor: Node
    stores: tuple[Tcgen05EpilogueStorePlan, ...]
    owned_nodes: tuple[Node, ...]

    @property
    def loads(self) -> tuple[Tcgen05EpilogueLoadPlan, ...]:
        return tuple(load for store in self.stores for load in store.load_plans)


@dataclasses.dataclass(frozen=True)
class Tcgen05EpilogueCandidate:
    """Node-free-enough preflight facts used before a config is selected."""

    explicit_epi_tile_loads_compatible: bool


def finalize_tcgen05_pair_local_plan(
    plan: Tcgen05EpiloguePlan,
    capability: Tcgen05PairLocalCapability,
) -> Tcgen05EpiloguePlan:
    """Attach the physical-layout capability after the MMA gate proves it."""
    if any(
        store.required_pair_width not in (None, capability.width)
        for store in plan.stores
    ):
        raise ValueError("pair-local capability does not satisfy the logical plan")
    return dataclasses.replace(
        plan,
        stores=tuple(
            dataclasses.replace(store, pair_local=capability)
            if store.required_pair_width is not None
            else store
            for store in plan.stores
        ),
    )


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


class _UnsupportedEpilogue(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class _EpilogueRegionStore:
    store: Node
    value: Node
    output: torch.Tensor
    owned: frozenset[Node]
    boundaries: frozenset[Node]


@dataclasses.dataclass(frozen=True)
class _EpilogueRegion:
    stores: tuple[_EpilogueRegionStore, ...]
    owned: frozenset[Node]


def _fragment_elementwise_inputs(node: Node) -> tuple[Node, ...] | None:
    """Return explicit inputs for one side-effect-free fragment lowering."""
    from ..inductor_lowering import APIFuncLowering
    from ..inductor_lowering import PointwiseLowering

    lowering = node.meta.get("lowering")
    if not isinstance(lowering, PointwiseLowering) and not (
        isinstance(lowering, APIFuncLowering)
        and "cute" in lowering.api_func._pure_fragment_codegen
    ):
        return None
    inputs: list[Node] = []
    map_arg(
        (node.args, {**node.kwargs, "_extra_deps": None}),
        lambda value: inputs.append(value) or value,
    )
    if (
        not inputs
        or any(
            not isinstance(value.meta.get("val"), (torch.Tensor, tuple, list))
            for value in inputs
        )
        or (
            isinstance(lowering, PointwiseLowering)
            and len(inputs) != len(lowering.input_names)
        )
    ):
        return None
    return tuple(inputs)


def _is_pointwise(node: Node) -> bool:
    return _fragment_elementwise_inputs(node) is not None


def _is_tuple_getitem(node: Node) -> bool:
    if node.op != "call_function" or node.target is not operator.getitem:
        return False
    base = node.args[0] if node.args else None
    index = node.args[1] if len(node.args) > 1 else None
    return (
        isinstance(base, Node)
        and base.op == "call_function"
        and base.target is view_ops.split
        and isinstance(base.meta.get("val"), (tuple, list))
        and index in (0, 1)
    )


def _coordinate_view_indices_supported(node: Node) -> bool:
    source = node.args[0] if node.args else None
    indices = node.args[1] if len(node.args) > 1 else None
    source_val = source.meta.get("val") if isinstance(source, Node) else None
    output_val = node.meta.get("val")
    if (
        not isinstance(indices, (list, tuple))
        or not isinstance(source_val, torch.Tensor)
        or not isinstance(output_val, torch.Tensor)
    ):
        return False
    return (
        all(index is None or index == slice(None) for index in indices)
        and len(indices) == output_val.ndim
        and sum(index == slice(None) for index in indices) == source_val.ndim
    )


def _host_load_supported(node: Node, source: Node) -> bool:
    indices = node.args[1] if len(node.args) > 1 else None
    source_val = source.meta.get("val")
    output_val = node.meta.get("val")
    if (
        not isinstance(indices, (list, tuple))
        or not isinstance(source_val, torch.Tensor)
        or not isinstance(output_val, torch.Tensor)
        or len(indices) != source_val.ndim
    ):
        return False
    advanced = [
        index
        for index in indices
        if isinstance(index, Node) and isinstance(index.meta.get("val"), torch.Tensor)
    ]
    if advanced:
        return len(advanced) == len(indices)
    coordinate_dims = 0
    env = CompileEnvironment.current()

    def source_dim_covers_block(tensor_dim: int, block_id: int) -> bool:
        block_size = env.block_sizes[env.canonical_block_id(block_id)].size
        return isinstance(block_size, (int, torch.SymInt)) and env.known_equal(
            source_val.shape[tensor_dim], block_size
        )

    def static_index_in_bounds(tensor_dim: int, index: int) -> bool:
        extent = source_val.shape[tensor_dim]
        return isinstance(extent, int) and 0 <= index < extent

    for tensor_dim, index in enumerate(indices):
        singleton = env.known_equal(source_val.shape[tensor_dim], 1) and not (
            isinstance(index, slice) and index == slice(None)
        )
        if isinstance(index, Node):
            index_value = index.meta.get("val")
            if isinstance(index_value, torch.SymInt):
                block_id = env.get_block_id(index_value)
                if block_id is None or (
                    not singleton and not source_dim_covers_block(tensor_dim, block_id)
                ):
                    return False
                coordinate_dims += 1
            elif not isinstance(index_value, int) or not static_index_in_bounds(
                tensor_dim, index_value
            ):
                return False
        elif isinstance(index, slice) and index == slice(None):
            if coordinate_dims >= output_val.ndim:
                return False
            block_id = env.resolve_block_id(output_val.shape[coordinate_dims])
            if block_id is None or not source_dim_covers_block(tensor_dim, block_id):
                return False
            coordinate_dims += 1
        elif not isinstance(index, int) or not static_index_in_bounds(
            tensor_dim, index
        ):
            return False
    return coordinate_dims == output_val.ndim


def _is_supported_node(node: Node) -> bool:
    if node.op != "call_function":
        return False
    if node.target is memory_ops.load:
        extra_mask = node.args[2] if len(node.args) > 2 else None
        source = node.args[0] if node.args else None
        if extra_mask is not None or not isinstance(source, Node):
            return False
        if _is_host_tensor_node(source):
            return _host_load_supported(node, source)
        return _coordinate_view_indices_supported(node)
    if node.target is view_ops.subscript:
        return _coordinate_view_indices_supported(node)
    return (
        _is_pointwise(node)
        or node.target in _SHAPE_TARGETS
        or node.target is tile_index
        or _is_tuple_getitem(node)
    )


def _is_host_tensor_node(node: Node) -> bool:
    return node.op == "call_function" and node.target is _tracing_ops._host_tensor


def _tensor_dependencies(node: Node) -> tuple[Node, ...]:
    """Return semantic tensor dependencies for one individually supported node."""
    if node.target is memory_ops.load:
        source = node.args[0] if node.args else None
        dependencies: list[Node] = []
        if isinstance(source, Node) and not _is_host_tensor_node(source):
            dependencies.append(source)
        indices = node.args[1] if len(node.args) > 1 else ()
        if isinstance(indices, (list, tuple)):
            dependencies.extend(
                index
                for index in indices
                if isinstance(index, Node)
                and isinstance(index.meta.get("val"), torch.Tensor)
            )
        return tuple(dict.fromkeys(dependencies))
    if node.target is tile_index:
        return ()
    return tuple(
        dependency
        for dependency in node.all_input_nodes
        if isinstance(dependency.meta.get("val"), (torch.Tensor, tuple, list))
        and not _is_host_tensor_node(dependency)
    )


def _store_pairs(graphs: Sequence[GraphInfo]) -> tuple[tuple[Node, Node], ...]:
    pairs: list[tuple[Node, Node]] = []
    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or node.target is not memory_ops.store:
                continue
            value = node.args[2] if len(node.args) > 2 else None
            if isinstance(value, Node):
                pairs.append((node, value))
    return tuple(pairs)


def _store_tensor(store: Node) -> torch.Tensor | None:
    tensor_node = store.args[0] if store.args else None
    tensor = tensor_node.meta.get("val") if isinstance(tensor_node, Node) else None
    return tensor if isinstance(tensor, torch.Tensor) else None


def _is_output_trailing_vector_load(
    load: Node,
    output: torch.Tensor,
    expected_output_block_ids: tuple[int, ...],
) -> bool:
    source = load.args[0] if load.args else None
    source_val = source.meta.get("val") if isinstance(source, Node) else None
    indices = load.args[1] if len(load.args) > 1 else None
    return (
        isinstance(source, Node)
        and _is_host_tensor_node(source)
        and isinstance(source_val, torch.Tensor)
        and source_val.ndim == 1
        and source_val.stride() == (1,)
        and CompileEnvironment.current().known_equal(
            source_val.shape[0], output.shape[-1]
        )
        and isinstance(indices, (list, tuple))
        and exact_tile_block_ids(CompileEnvironment.current(), indices)
        == expected_output_block_ids[-1:]
    )


def _slice_for_store(
    value: Node,
    *,
    anchor: Node,
    inner_outputs: dict[int, tuple[Node | None, ...]],
) -> tuple[set[Node], set[Node]]:
    owned: set[Node] = set()
    boundaries: set[Node] = set()
    visiting: set[Node] = set()

    def visit(node: Node) -> None:
        if node in owned or node in boundaries:
            return
        if (
            resolve_tcgen05_accumulator_boundary(node, {anchor}, inner_outputs)
            is anchor
        ):
            boundaries.add(node)
            return
        if node in visiting or not _is_supported_node(node):
            raise _UnsupportedEpilogue
        visiting.add(node)
        for dependency in _tensor_dependencies(node):
            visit(dependency)
        visiting.remove(node)
        owned.add(node)

    visit(value)
    if not boundaries:
        raise _UnsupportedEpilogue
    return owned, boundaries


def _validate_store(
    store: Node,
    *,
    expected_output_block_ids: tuple[int, ...],
) -> torch.Tensor:
    tensor = _store_tensor(store)
    if (
        tensor is None
        or tensor.ndim not in (2, 3)
        or tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        raise _UnsupportedEpilogue
    subscripts = store.args[1] if len(store.args) > 1 else None
    if not isinstance(subscripts, (list, tuple)):
        raise _UnsupportedEpilogue
    if exact_tile_block_ids(CompileEnvironment.current(), subscripts) != (
        expected_output_block_ids
    ):
        raise _UnsupportedEpilogue
    if len(store.args) > 3 and store.args[3] is not None:
        raise _UnsupportedEpilogue
    return tensor


def _extract_epilogue_region(
    graphs: Sequence[GraphInfo],
    anchor: Node,
    *,
    expected_output_block_ids: tuple[int, ...],
) -> _EpilogueRegion:
    inner_outputs = build_inner_outputs_index_from_graphs(graphs)
    stores: list[_EpilogueRegionStore] = []
    union_owned: set[Node] = set()
    for store, value in _store_pairs(graphs):
        if anchor not in reach_matmul_anchors(
            value,
            target_fx_nodes={anchor},
            inner_outputs_by_graph_id=inner_outputs,
        ):
            continue
        output = _validate_store(
            store, expected_output_block_ids=expected_output_block_ids
        )
        owned, boundaries = _slice_for_store(
            value, anchor=anchor, inner_outputs=inner_outputs
        )
        union_owned.update(owned)
        stores.append(
            _EpilogueRegionStore(
                store=store,
                value=value,
                output=output,
                owned=frozenset(owned),
                boundaries=frozenset(boundaries),
            )
        )
    if not stores:
        raise _UnsupportedEpilogue
    final_stores = {entry.store for entry in stores}
    if any(
        user not in union_owned and user not in final_stores
        for node in union_owned
        for user in node.users
    ):
        raise _UnsupportedEpilogue
    return _EpilogueRegion(tuple(stores), frozenset(union_owned))


def analyze_tcgen05_epilogue_candidate(
    graphs: Sequence[GraphInfo],
    anchor: Node,
    *,
    expected_output_block_ids: tuple[int, ...],
) -> Tcgen05EpilogueCandidate | None:
    """Check graph purity and per-node capabilities without classifying a formula."""
    explicit_epi_tile_loads_compatible = True
    try:
        region = _extract_epilogue_region(
            graphs,
            anchor,
            expected_output_block_ids=expected_output_block_ids,
        )
        for entry in region.stores:
            for owned_node in entry.owned:
                source = owned_node.args[0] if owned_node.args else None
                if (
                    owned_node.target is memory_ops.load
                    and isinstance(source, Node)
                    and _is_host_tensor_node(source)
                ):
                    explicit_epi_tile_loads_compatible &= (
                        _is_output_trailing_vector_load(
                            owned_node, entry.output, expected_output_block_ids
                        )
                    )
    except _UnsupportedEpilogue:
        return None
    return Tcgen05EpilogueCandidate(explicit_epi_tile_loads_compatible)


def _shape(node: Node, config: Config) -> list[int]:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        raise _UnsupportedEpilogue
    return _get_tile_shape(value, CompileEnvironment.current(), config)


def _coords(flat: int, shape: list[int]) -> tuple[int, ...]:
    result: list[int] = []
    for dim, extent in enumerate(shape):
        stride = 1
        for trailing in shape[dim + 1 :]:
            stride *= trailing
        result.append((flat // stride) % extent)
    return tuple(result)


_Request = tuple[str, Node, int]


def _coordinate_requests(
    node: Node,
    flat: int,
    *,
    boundaries: set[Node],
    config: Config,
    projection: int | None = None,
    memo: dict[tuple[Node, int, int | None], frozenset[_Request]],
) -> frozenset[_Request]:
    key = (node, flat, projection)
    if key in memo:
        return memo[key]

    def leaf(
        current: Node, current_flat: int | str, current_projection: int | None
    ) -> frozenset[_Request]:
        if not isinstance(current_flat, int):
            raise _UnsupportedEpilogue
        leaf_key = (current, current_flat, current_projection)
        if leaf_key != key and leaf_key in memo:
            return memo[leaf_key]
        if current in boundaries:
            return frozenset({("boundary", current, current_flat)})
        output_shape = _shape(current, config) if current_projection is None else None
        if _is_pointwise(current):
            if output_shape is None:
                raise _UnsupportedEpilogue
            requests: set[_Request] = set()
            for dependency in _tensor_dependencies(current):
                dependency_flat = broadcast_logical_flat_index(
                    current_flat,
                    output_shape=output_shape,
                    source_shape=_shape(dependency, config),
                )
                if not isinstance(dependency_flat, int):
                    raise _UnsupportedEpilogue
                requests.update(
                    _coordinate_requests(
                        dependency,
                        dependency_flat,
                        boundaries=boundaries,
                        config=config,
                        memo=memo,
                    )
                )
            return frozenset(requests)
        if current.target is memory_ops.load:
            source = current.args[0] if current.args else None
            if isinstance(source, Node) and _is_host_tensor_node(source):
                if output_shape is None:
                    raise _UnsupportedEpilogue
                requests = {("load", current, current_flat)}
                indices = current.args[1] if len(current.args) > 1 else ()
                if not isinstance(indices, (list, tuple)):
                    raise _UnsupportedEpilogue
                for index in indices:
                    if not (
                        isinstance(index, Node)
                        and isinstance(index.meta.get("val"), torch.Tensor)
                    ):
                        continue
                    index_flat = broadcast_logical_flat_index(
                        current_flat,
                        output_shape=output_shape,
                        source_shape=_shape(index, config),
                    )
                    if not isinstance(index_flat, int):
                        raise _UnsupportedEpilogue
                    requests.update(
                        _coordinate_requests(
                            index,
                            index_flat,
                            boundaries=boundaries,
                            config=config,
                            memo=memo,
                        )
                    )
                return frozenset(requests)
            if isinstance(source, Node) and output_shape is not None:
                indices = current.args[1] if len(current.args) > 1 else None
                if not isinstance(indices, (list, tuple)):
                    raise _UnsupportedEpilogue
                output_coords = iter(
                    logical_coords_from_flat(current_flat, output_shape)
                )
                source_coords: list[int | str] = []
                for index in indices:
                    coord = next(output_coords)
                    if index is None:
                        if coord != 0:
                            raise _UnsupportedEpilogue
                    elif isinstance(index, slice) and index == slice(None):
                        source_coords.append(coord)
                    else:
                        raise _UnsupportedEpilogue
                source_flat = logical_flat_from_coords(
                    source_coords, _shape(source, config)
                )
                if not isinstance(source_flat, int):
                    raise _UnsupportedEpilogue
                return _coordinate_requests(
                    source,
                    source_flat,
                    boundaries=boundaries,
                    config=config,
                    memo=memo,
                )
        if current.target is tile_index:
            return frozenset()
        raise _UnsupportedEpilogue

    result = resolve_cute_logical_coordinate(
        node,
        flat,
        config=config,
        leaf=leaf,
        select=lambda _selector, choices: frozenset().union(*choices),
        projection=projection,
    )
    memo[key] = result
    return result


def _trailing_matrix_coords(shape: list[int], flat: int) -> tuple[int, int, int]:
    coords = _coords(flat, shape)
    if len(coords) == 2:
        return (0, coords[0], coords[1])
    if len(coords) == 3:
        return coords
    raise _UnsupportedEpilogue


def _load_plan(
    load: Node,
    *,
    value: Node,
    output_shape: list[int],
    output_tensor: torch.Tensor,
    expected_output_block_ids: tuple[int, ...],
    requests_by_output: list[frozenset[_Request]],
    config: Config,
) -> Tcgen05EpilogueLoadPlan:
    source = load.args[0] if load.args else None
    if not isinstance(source, Node) or not _is_host_tensor_node(source):
        raise _UnsupportedEpilogue
    source_val = source.meta.get("val")
    if not isinstance(source_val, torch.Tensor):
        raise _UnsupportedEpilogue
    load_shape = _shape(load, config)
    demanded: list[set[int]] = []
    for requests in requests_by_output:
        demanded.append(
            {flat for kind, node, flat in requests if kind == "load" and node is load}
        )
    scope = Tcgen05EpilogueLoadScope.DIRECT
    broadcast_axis: int | None = None
    indices = load.args[1] if len(load.args) > 1 else None
    load_block_ids = (
        exact_tile_block_ids(CompileEnvironment.current(), indices)
        if isinstance(indices, (list, tuple))
        else None
    )
    if all(len(flats) == 1 for flats in demanded):
        flats = [next(iter(items)) for items in demanded]
        if (
            load_shape == output_shape
            and load_block_ids == expected_output_block_ids
            and tuple(source_val.shape) == tuple(output_tensor.shape)
            and all(
                load_flat == output_flat for output_flat, load_flat in enumerate(flats)
            )
        ):
            scope = Tcgen05EpilogueLoadScope.OUTPUT_ALIGNED_SUBTILE
            if source_val.ndim >= 2:
                trailing_strides = source_val.stride()[-2:]
                if trailing_strides == (0, 1):
                    broadcast_axis = 1
                elif trailing_strides == (1, 0):
                    broadcast_axis = 2
        elif (
            len(load_shape) == 1
            and len(output_shape) in (2, 3)
            and _is_output_trailing_vector_load(
                load, output_tensor, expected_output_block_ids
            )
            and all(
                load_flat == _coords(output_flat, output_shape)[-1]
                for output_flat, load_flat in enumerate(flats)
            )
        ):
            scope = Tcgen05EpilogueLoadScope.OUTPUT_ALIGNED_SUBTILE
            broadcast_axis = 1
    return Tcgen05EpilogueLoadPlan(
        load_node=load,
        host_tensor_fx_node=source,
        host_tensor_val=source_val,
        store_value_node=value,
        scope=scope,
        broadcast_axis=broadcast_axis,
    )


def analyze_tcgen05_epilogue_plan(
    graphs: Sequence[GraphInfo],
    anchor: Node,
    *,
    expected_output_block_ids: tuple[int, ...],
    config: Config,
) -> Tcgen05EpiloguePlan | None:
    """Build and fully validate one generic live-FX fragment plan."""
    try:
        region = _extract_epilogue_region(
            graphs,
            anchor,
            expected_output_block_ids=expected_output_block_ids,
        )

        store_plans: list[Tcgen05EpilogueStorePlan] = []
        for entry in region.stores:
            store = entry.store
            value = entry.value
            output = entry.output
            owned = set(entry.owned)
            boundaries = set(entry.boundaries)
            output_shape = _shape(value, config)
            output_numel = 1
            for extent in output_shape:
                output_numel *= extent
            memo: dict[tuple[Node, int, int | None], frozenset[_Request]] = {}
            requests_by_output = [
                _coordinate_requests(
                    value,
                    flat,
                    boundaries=boundaries,
                    config=config,
                    memo=memo,
                )
                for flat in range(output_numel)
            ]
            for output_flat, requests in enumerate(requests_by_output):
                output_hmn = _trailing_matrix_coords(output_shape, output_flat)
                for kind, request_node, request_flat in requests:
                    if kind != "boundary":
                        continue
                    source_hmn = _trailing_matrix_coords(
                        _shape(request_node, config), request_flat
                    )
                    if (
                        source_hmn[0] != output_hmn[0]
                        or source_hmn[1] != output_hmn[1]
                        or source_hmn[2] // TCGEN05_FRAGMENT_PAIR_WIDTH
                        != output_hmn[2] // TCGEN05_FRAGMENT_PAIR_WIDTH
                    ):
                        raise _UnsupportedEpilogue
            host_loads_list: list[Node] = []
            for graph_info in graphs:
                for owned_node in graph_info.graph.nodes:
                    source = owned_node.args[0] if owned_node.args else None
                    if (
                        owned_node in owned
                        and owned_node.target is memory_ops.load
                        and isinstance(source, Node)
                        and _is_host_tensor_node(source)
                    ):
                        host_loads_list.append(owned_node)
            host_loads = tuple(host_loads_list)
            load_plans = tuple(
                _load_plan(
                    load,
                    value=value,
                    output_shape=output_shape,
                    output_tensor=output,
                    expected_output_block_ids=expected_output_block_ids,
                    requests_by_output=requests_by_output,
                    config=config,
                )
                for load in host_loads
            )
            boundary_identity = all(
                all(
                    kind != "boundary"
                    or _trailing_matrix_coords(_shape(node, config), request_flat)
                    == _trailing_matrix_coords(output_shape, output_flat)
                    for kind, node, request_flat in requests
                )
                for output_flat, requests in enumerate(requests_by_output)
            )
            whole_fragment_nodes_supported = all(
                _is_pointwise(node)
                or any(
                    load.load_node is node
                    and load.scope is Tcgen05EpilogueLoadScope.OUTPUT_ALIGNED_SUBTILE
                    for load in load_plans
                )
                for node in owned
            )
            requires_scalar = (
                not boundary_identity or not whole_fragment_nodes_supported
            )
            required_pair_width = (
                TCGEN05_FRAGMENT_PAIR_WIDTH if not boundary_identity else None
            )
            if (
                required_pair_width is not None
                and len(output_shape) == 3
                and output_shape[0] != 1
            ):
                raise _UnsupportedEpilogue
            store_plans.append(
                Tcgen05EpilogueStorePlan(
                    store_node=store,
                    value_node=value,
                    output_tensor=output,
                    output_block_ids=expected_output_block_ids,
                    boundary_nodes=tuple(
                        node for node in value.graph.nodes if node in boundaries
                    ),
                    load_plans=load_plans,
                    requires_scalar_fragment=requires_scalar,
                    required_pair_width=required_pair_width,
                )
            )
        ordered_owned = tuple(
            node
            for graph_info in graphs
            for node in graph_info.graph.nodes
            if node in region.owned
        )
        return Tcgen05EpiloguePlan(
            anchor=anchor,
            stores=tuple(store_plans),
            owned_nodes=ordered_owned,
        )
    except (_UnsupportedEpilogue, IndexError, StopIteration, ValueError):
        return None
