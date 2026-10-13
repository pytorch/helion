"""Graph authority for a pointwise state update followed by a row sum.

Discovery is independent of physical tiling and does not rewrite the graph.
It retains the original pointwise DAG, memory indices, masks and store casts.
In particular, an indirect state index is not required to be unique: a later
schedule must refine the original parallel memory events, not strengthen the
caller's index contract.
"""

from __future__ import annotations

import dataclasses
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import tile_ops
from ...language import view_ops
from ..device_ir import RootGraphInfo

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo


class InapplicableStateUpdate(ValueError):
    """The original graph does not establish the selected schedule's contract."""


def _freeze(value: object) -> object:
    if isinstance(value, Node):
        return (Node, id(value))
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_freeze(item) for item in value))
    if isinstance(value, dict):
        return (dict, tuple((_freeze(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, slice):
        return (slice, _freeze(value.start), _freeze(value.stop), _freeze(value.step))
    if isinstance(value, torch.Tensor):
        # Keep the wrapper alive and bind its underlying owner. A temporary
        # wrapper's Python id can be reused after Tensor.set_ changes storage.
        storage = value.untyped_storage()
        return (
            torch.Tensor,
            id(value),
            storage,
            storage._cdata,
            value.dtype,
            tuple(_freeze(x) for x in value.shape),
            tuple(_freeze(x) for x in value.stride()),
            _freeze(value.storage_offset()),
        )
    if isinstance(value, torch.SymInt):
        return (torch.SymInt, str(value._sympy_()))
    if type(value) is float:
        return (float, value.hex())
    if (
        value is None
        or type(value) in (bool, int, str)
        or isinstance(value, torch.dtype)
    ):
        return (type(value), value)
    return (type(value), id(value))


@dataclasses.dataclass(frozen=True)
class NodeIdentity:
    node: Node = dataclasses.field(repr=False)
    op: str
    target: object = dataclasses.field(repr=False)
    args: object = dataclasses.field(repr=False)
    kwargs: object = dataclasses.field(repr=False)
    users: tuple[Node, ...] = dataclasses.field(repr=False)
    value: object = dataclasses.field(repr=False)

    @classmethod
    def capture(cls, node: Node) -> NodeIdentity:
        return cls(
            node,
            node.op,
            node.target,
            _freeze(node.args),
            _freeze(node.kwargs),
            tuple(node.users),
            _freeze(node.meta.get("val")),
        )


def _ancestors(node: Node) -> frozenset[Node]:
    pending = [node]
    seen: set[Node] = set()
    while pending:
        current = pending.pop()
        if current not in seen:
            seen.add(current)
            pending.extend(current.all_input_nodes)
    return frozenset(seen)


def _rank(node: Node) -> int | None:
    value = node.meta.get("val")
    return value.ndim if isinstance(value, torch.Tensor) else None


def _host(node: object) -> Node | None:
    if (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target is _tracing_ops._host_tensor
        and len(node.args) == 1
        and type(node.args[0]) is str
        and not node.kwargs
        and isinstance(node.meta.get("val"), torch.Tensor)
    ):
        return node
    return None


def _memory(node: Node, kind: object) -> bool:
    return (
        node.op == "call_function"
        and node.target is kind
        and len(node.args) == 4
        and not node.kwargs
        and _host(node.args[0]) is not None
        and isinstance(node.args[1], (list, tuple))
    )


def _pointwise(node: Node) -> bool:
    """Only original registered pointwise operators, never arbitrary callables."""
    if not isinstance(node.target, torch._ops.OpOverload):
        return False
    return (
        node.target.namespace in ("aten", "prims")
        and torch.Tag.pointwise in node.target.tags
        and torch.Tag.nondeterministic_seeded not in node.target.tags
        and not node.target._schema.is_mutable
    )


def _storage_input(store: Node) -> Node | None:
    value = store.args[2]
    if not isinstance(value, Node):
        return None
    source = _host(store.args[0])
    assert source is not None
    dtype = source.meta["val"].dtype
    if (
        value.target is torch.ops.prims.convert_element_type.default
        and len(value.args) == 2
        and not value.kwargs
        and value.args[1] is dtype
        and isinstance(value.args[0], Node)
    ):
        return value.args[0]
    if (
        isinstance(value.meta.get("val"), torch.Tensor)
        and value.meta["val"].dtype is dtype
    ):
        return value
    return None


@dataclasses.dataclass(frozen=True)
class PointwiseStateRegion:
    root: RootGraphInfo = dataclasses.field(repr=False)
    source: tuple[NodeIdentity, ...] = dataclasses.field(repr=False)
    state_load: Node = dataclasses.field(repr=False)
    state_store: Node = dataclasses.field(repr=False)
    output_store: Node = dataclasses.field(repr=False)
    update: Node = dataclasses.field(repr=False)
    reduction: Node = dataclasses.field(repr=False)
    row_selector: Node = dataclasses.field(repr=False)
    feature_selector: Node = dataclasses.field(repr=False)
    loads: tuple[Node, ...] = dataclasses.field(repr=False)

    def check(self) -> None:
        if self.source != tuple(NodeIdentity.capture(n) for n in self.root.graph.nodes):
            raise InapplicableStateUpdate("pointwise state graph changed")
        current = _discover_root(self.root)
        if current is None or (
            self.state_load,
            self.state_store,
            self.output_store,
            self.update,
            self.reduction,
            self.row_selector,
            self.feature_selector,
            self.loads,
        ) != (
            current.state_load,
            current.state_store,
            current.output_store,
            current.update,
            current.reduction,
            current.row_selector,
            current.feature_selector,
            current.loads,
        ):
            raise InapplicableStateUpdate("pointwise state region changed")

    def check_readonly_owners(self) -> None:
        """Recheck bound alias facts and require distinct mutable allocations."""
        from ..compile_environment import CompileEnvironment
        from .positional_host_effects import HostAliasOracle

        self.check()
        env = CompileEnvironment.current()
        aliases = HostAliasOracle(env)
        mutable = (self.state_store.args[0], self.output_store.args[0])
        for target in mutable:
            assert isinstance(target, Node)
            for load in self.loads:
                owner = load.args[0]
                assert isinstance(owner, Node)
                if owner is target and load is self.state_load:
                    continue
                if target.meta["val"].untyped_storage()._cdata == owner.meta[
                    "val"
                ].untyped_storage()._cdata or aliases.may_alias(target, owner):
                    raise InapplicableStateUpdate(
                        f"state/output may alias a reordered read: {target.args[0]}/{owner.args[0]}"
                    )
        first, second = mutable
        assert isinstance(first, Node) and isinstance(second, Node)
        if first.meta["val"].untyped_storage()._cdata == second.meta[
            "val"
        ].untyped_storage()._cdata or aliases.may_alias(first, second):
            raise InapplicableStateUpdate("state/output storage overlaps")


def _discover_root(root: RootGraphInfo) -> PointwiseStateRegion | None:
    nodes = tuple(root.graph.nodes)
    stores = tuple(n for n in nodes if n.target is memory_ops.store)
    loads = tuple(n for n in nodes if n.target is memory_ops.load)
    reductions = tuple(n for n in nodes if n.target is torch.ops.aten.sum.dim_IntList)
    if (
        len(stores) != 2
        or len(reductions) != 1
        or any(not _memory(n, memory_ops.store) for n in stores)
        or any(not _memory(n, memory_ops.load) for n in loads)
    ):
        return None
    reduction = reductions[0]
    if (
        len(reduction.args) != 2
        or reduction.kwargs
        or not isinstance(reduction.args[0], Node)
        or _rank(reduction.args[0]) != 2
        or _rank(reduction) != 1
        or not isinstance(reduction.args[1], (list, tuple))
        or tuple(reduction.args[1]) not in ((-1,), (1,))
        or any(type(dim) is not int for dim in reduction.args[1])
        or reduction.meta["val"].dtype is not torch.float32
    ):
        return None
    reduction_inputs = _ancestors(reduction)
    candidates = []
    for store in stores:
        update = _storage_input(store)
        if update is None or _rank(update) != 2 or update not in reduction_inputs:
            continue
        matching = [
            n
            for n in loads
            if n.args[0] is store.args[0]
            and _freeze(n.args[1]) == _freeze(store.args[1])
            and _freeze(n.args[2]) == _freeze(store.args[3])
        ]
        if len(matching) != 1:
            continue
        load = matching[0]
        if load not in _ancestors(update) or _rank(load) != 2:
            continue
        candidates.append((store, load, update))
    if len(candidates) != 1:
        return None
    state_store, state_load, update = candidates[0]
    output_store = next(n for n in stores if n is not state_store)
    output_value = output_store.args[2]
    if (
        not isinstance(output_value, Node)
        or _rank(output_value) != 1
        or reduction not in _ancestors(output_value)
        or state_store.args[0] is output_store.args[0]
        or sum(n.args[0] is state_store.args[0] for n in loads) != 1
        or state_load.args[3] is not None
    ):
        return None
    selectors = state_store.args[1]
    if len(selectors) < 2:
        return None
    row, feature = selectors[-2:]
    if (
        not isinstance(row, Node)
        or row.target is not _tracing_ops._get_symnode
        or not isinstance(feature, Node)
        or feature.target is not torch.ops.prims.iota.default
        or len(feature.args) != 1
        or type(feature.args[0]) is not int
        or feature.args[0] <= 0
        or feature.kwargs.get("start") != 0
        or feature.kwargs.get("step") != 1
        or feature.kwargs.get("dtype") is not torch.int32
        or not output_store.args[1]
        or output_store.args[1][-1] is not row
    ):
        return None
    # Every memory operation must contribute to one of the two original stores.
    # An unused load is still an effect and cannot be dropped by this schedule.
    live = _ancestors(state_store) | _ancestors(output_store)
    if any(n not in live for n in loads):
        return None
    structural = {
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        tile_ops.tile_id,
        torch.ops.prims.iota.default,
        view_ops.subscript,
        memory_ops.load,
        memory_ops.store,
        torch.ops.aten.sum.dim_IntList,
        operator.add,
        operator.mul,
        operator.floordiv,
        operator.mod,
    }
    for node in nodes:
        if node.op == "output":
            if node.args != (None,):
                return None
        elif node.op != "call_function" or (
            node.target not in structural and not _pointwise(node)
        ):
            return None
    # Cross-feature views/reductions are forbidden in the elementwise recipe.
    for node in _ancestors(update) | _ancestors(reduction.args[0]):
        if node.target is view_ops.subscript:
            if (
                len(node.args) != 2
                or node.kwargs
                or not isinstance(node.args[1], (tuple, list))
                or any(x is not None and x != slice(None) for x in node.args[1])
            ):
                return None
        if node is reduction:
            return None
    return PointwiseStateRegion(
        root,
        tuple(NodeIdentity.capture(n) for n in nodes),
        state_load,
        state_store,
        output_store,
        update,
        reduction,
        row,
        feature,
        loads,
    )


def discover_pointwise_state_region(
    graphs: Sequence[GraphInfo],
) -> PointwiseStateRegion | None:
    if len(graphs) != 1 or not isinstance(graphs[0], RootGraphInfo):
        return None
    return _discover_root(graphs[0])


def _proven_int(value: object) -> int | None:
    """Resolve only selected/static equalities, never a runtime size hint."""
    import sympy

    from ..compile_environment import CompileEnvironment

    if type(value) is int:
        return value
    if isinstance(value, torch.SymInt):
        value = value._sympy_()
    if isinstance(value, sympy.Expr):
        value = CompileEnvironment.current().specialize_expr(value)
        if isinstance(value, sympy.Integer):
            return int(value)
    return None


@dataclasses.dataclass(frozen=True)
class StateTaskDomain:
    """The original full parallel task domain and dense state plane."""

    region: PointwiseStateRegion = dataclasses.field(repr=False)
    grid_block_ids: tuple[int, ...]
    grid_extents: tuple[int, ...]
    row_block_id: int
    state_shape: tuple[int, ...]
    state_strides: tuple[int, ...]
    index_dtype: torch.dtype
    storage_dtype: torch.dtype

    def check(self) -> None:
        # The binder rechecks the current graph and owners before reading facts.
        current = bind_state_task_domain(self.region)
        if current != self:
            raise InapplicableStateUpdate("state task domain changed")


def bind_state_task_domain(region: PointwiseStateRegion) -> StateTaskDomain:
    """Bind canonical full axes, without assumptions on loaded slot contents."""
    from ..compile_environment import CompileEnvironment
    from ..device_function import DeviceFunction
    from ..host_function import HostFunction
    from .fx_matcher import _canonical_root_axis_ids

    region.check_readonly_owners()
    env = CompileEnvironment.current()
    df = DeviceFunction.current()
    state = cast("Node", region.state_store.args[0]).meta["val"]
    output = cast("Node", region.output_store.args[0]).meta["val"]
    state_shape = tuple(_proven_int(x) for x in state.shape)
    state_strides = tuple(_proven_int(x) for x in state.stride())
    extents = tuple(_proven_int(x) for x in output.shape)
    if (
        any(x is None or x <= 0 for x in (*state_shape, *extents))
        or any(x is None or x < 0 for x in state_strides)
        or state.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or region.update.meta["val"].dtype is not torch.float32
        or state_shape[-1] != region.feature_selector.args[0]
        or state_shape[-2] != extents[-1]
        or env.index_dtype not in (torch.int32, torch.int64)
    ):
        raise InapplicableStateUpdate("unproved state shape or scalar dtype")
    # Above rejects all None entries; use explicit tuples for the typed result.
    shape = tuple(int(x) for x in state_shape if x is not None)
    strides = tuple(int(x) for x in state_strides if x is not None)
    grid_extents = tuple(int(x) for x in extents if x is not None)
    if strides != tuple(math.prod(shape[i + 1 :]) for i in range(len(shape))):
        raise InapplicableStateUpdate(
            "state descriptor requires dense positive strides"
        )
    if math.prod(shape) > 2**31 - 1 or math.prod(grid_extents) > 2**31 - 1:
        raise InapplicableStateUpdate(
            "state or task extent exceeds descriptor index range"
        )
    indices = cast("Sequence[Node]", region.output_store.args[1])
    if len(indices) != len(grid_extents) or any(
        not isinstance(x, Node) for x in indices
    ):
        raise InapplicableStateUpdate("output is not the canonical full task domain")
    if any(x.target is not tile_ops.tile_id for x in indices[:-1]):
        raise InapplicableStateUpdate(
            "output outer axes are not scalar task coordinates"
        )
    row_id = env.resolve_block_id(region.row_selector.meta.get("val"))
    ids = _canonical_root_axis_ids(
        HostFunction.current().device_ir,
        root_phase_index=region.root.phase_index,
        coordinates=(*indices[:-1], None),
        expected_extents=grid_extents,
    )
    if (
        ids is None
        or ids[-1] != row_id
        or any(df.resolved_block_size(axis) != 1 for axis in ids[:-1])
    ):
        raise InapplicableStateUpdate("original root axes or scalar outer tiles differ")
    # The transported prefix may be indirect/nonlinear, but it must be uniform
    # over row/feature and independent of the mutable old-state load.
    for index in cast("Sequence[object]", region.state_load.args[1])[:-2]:
        if isinstance(index, Node):
            deps = _ancestors(index)
            if (
                region.row_selector in deps
                or region.feature_selector in deps
                or region.state_load in deps
            ):
                raise InapplicableStateUpdate(
                    "state plane prefix varies within a row plane"
                )
            if isinstance(index.meta.get("val"), torch.Tensor) and _rank(index) != 0:
                raise InapplicableStateUpdate("state plane prefix is not CTA uniform")
        elif type(index) is not int:
            raise InapplicableStateUpdate("unsupported state plane prefix")
    return StateTaskDomain(
        region,
        tuple(ids),
        grid_extents,
        ids[-1],
        shape,
        strides,
        env.index_dtype,
        state.dtype,
    )


@dataclasses.dataclass(frozen=True)
class HorizontalGeometry:
    """Finite distinct stages, with one row per consumer lane subgroup."""

    consumer_warps: int
    lanes_per_row: int
    stage_features: int
    feature_extent: int
    itemsize: int

    @property
    def rows(self) -> int:
        return self.consumer_warps * 32 // self.lanes_per_row

    @property
    def stages(self) -> int:
        return self.feature_extent // self.stage_features

    @property
    def threads(self) -> int:
        return (self.consumer_warps + 1) * 32

    @property
    def state_bytes(self) -> int:
        return self.rows * self.feature_extent * self.itemsize

    @property
    def arrivals(self) -> int:
        return self.consumer_warps * 32 + 1

    def check(self, shared_limit: int, extra_shared: int = 0) -> None:
        values = (
            self.consumer_warps,
            self.lanes_per_row,
            self.stage_features,
            self.feature_extent,
            self.itemsize,
            shared_limit,
        )
        if (
            any(type(v) is not int or v <= 0 for v in values)
            or type(extra_shared) is not int
            or extra_shared < 0
        ):
            raise InapplicableStateUpdate("invalid horizontal geometry")
        if (
            32 % self.lanes_per_row
            or self.threads > 1024
            or self.feature_extent % self.stage_features
            or self.stage_features % self.lanes_per_row
            or self.itemsize not in (2, 4)
            or self.state_bytes + self.stages * 16 + 8 + extra_shared > shared_limit
        ):
            raise InapplicableStateUpdate("unsupported horizontal resources")

    def logical_origin(
        self, warp: int, lane: int, stage: int, item: int
    ) -> tuple[int, int]:
        """Semantic map before the independently proved shared bank permutation."""
        if not (
            0 <= warp < self.consumer_warps
            and 0 <= lane < 32
            and 0 <= stage < self.stages
            and 0 <= item < self.stage_features // self.lanes_per_row
        ):
            raise InapplicableStateUpdate("outside horizontal consumer domain")
        rows_per_warp = 32 // self.lanes_per_row
        return (
            warp * rows_per_warp + lane % rows_per_warp,
            stage * self.stage_features
            + (lane // rows_per_warp) * (self.stage_features // self.lanes_per_row)
            + item,
        )
