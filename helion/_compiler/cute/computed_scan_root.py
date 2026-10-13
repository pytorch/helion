"""Bounded standalone computed prefixes after existing whole-root routes decline.

This owner does not widen generic scan load recovery. Its single fresh output,
closed pure source and complete scan/grid domains permit a CTA-local shared image
and one full warp per independent prefix. Existing positional roots keep their
original sequential scan emitter.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ... import exc
from ...language import _tracing_ops
from ...language import memory_ops
from ...language import scan_ops
from ...language import tile_index
from ...language import tile_ops
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction
from ..variable_origin import TileBeginOrigin
from ..variable_origin import TileIdOrigin
from . import chained_matmul as chain
from .chained_collectives import warp_prefix_point
from .chained_execution import ChainedExecution
from .chained_store_expression import _node_facts
from .computed_scan_staging import KEY as STAGING_KEY
from .computed_scan_staging import StagedScanLayout
from .computed_scan_staging import staged_scan_layout
from .positional_domains import UnprovenDomain
from .positional_host_effects import HostAliasOracle
from .positional_root import _EXPRESSION_TARGETS
from .positional_root import PositionalEmitter
from .positional_root import PositionalRootPlan
from .positional_root import _scan_helper
from .positional_root import _tile_bytes
from .positional_root import _Unsupported
from .positional_root import plan_root_layout

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_function import DeviceFunction
    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST


MAX_SCAN_EXTENT = 1024


def _raw_input(node: Node) -> bool:
    """Leave the existing load/cast/view scan families with their original owner."""
    seen: set[Node] = set()
    while node not in seen:
        seen.add(node)
        if node.target is memory_ops.load:
            return True
        if (
            node.target
            not in (
                *chain._VIEWS,
                torch.ops.prims.convert_element_type.default,
                _tracing_ops._new_var,
                _tracing_ops._mask_to,
            )
            or not node.args
            or not isinstance(node.args[0], Node)
        ):
            return False
        node = node.args[0]
    return False


def _scan_view(node: Node, scan: Node) -> bool:
    seen: set[Node] = set()
    while node is not scan and node not in seen:
        seen.add(node)
        if (
            node.target not in chain._VIEWS
            or not node.args
            or not isinstance(node.args[0], Node)
        ):
            return False
        node = node.args[0]
    return node is scan


def _storage_facts(nodes: tuple[Node, ...]) -> tuple[object, ...]:
    env = CompileEnvironment.current()
    return tuple(
        (
            node,
            node.meta["val"].untyped_storage()._cdata,
            node.meta["val"].untyped_storage() in env.fresh_allocation_storages,
            node.meta["val"].untyped_storage() in env.fresh_initialized_storages,
        )
        for node in nodes
        if node.target is _tracing_ops._host_tensor
    )


def _direct_axis(selector: Node) -> tuple[int, bool] | None:
    """A direct grid origin or tile coordinate, without index arithmetic."""
    env = CompileEnvironment.current()
    if selector.target in (tile_ops.tile_begin, tile_ops.tile_id, tile_index):
        block = env.resolve_block_id(cast("Node", selector.args[0]).meta.get("val"))
        return (
            (env.canonical_block_id(block), selector.target is not tile_index)
            if block is not None
            else None
        )
    if selector.target not in (_tracing_ops._get_symnode, torch.ops.aten.sym_size.int):
        return None
    value = selector.meta.get("val")
    if not isinstance(value, torch.SymInt):
        return None
    info = HostFunction.current().expr_to_origin.get(value._sympy_())
    if info is not None and type(info.origin) in (TileBeginOrigin, TileIdOrigin):
        origin = cast("TileBeginOrigin | TileIdOrigin", info.origin)
        return env.canonical_block_id(origin.block_id), True
    if chain._Expression._grid_origin(selector) is not None:
        return None
    block = env.resolve_block_id(value)
    return (env.canonical_block_id(block), False) if block is not None else None


def _complete_store(plan: PositionalRootPlan, store: Node) -> bool:
    """Complete fresh dense output; every launched axis has one disjoint selector."""
    if len(store.args) != 4 or store.kwargs or store.args[3] is not None:
        return False
    target, selectors, value = store.args[:3]
    if (
        not isinstance(target, Node)
        or target.target is not _tracing_ops._host_tensor
        or not isinstance(selectors, (list, tuple))
        or not isinstance(value, Node)
    ):
        return False
    tensor = target.meta["val"]
    if tensor.dtype != torch.float32 or not tensor.is_contiguous():
        return False
    env = CompileEnvironment.current()
    if tensor.untyped_storage() not in env.fresh_allocation_storages:
        return False
    shape = chain._host_shape(tensor)
    if len(selectors) != len(shape) or min(shape) <= 0:
        return False
    axes = {axis: (extent, block) for axis, extent, block in plan.axes}
    seen: set[int] = set()
    for selector, extent in zip(selectors, shape, strict=True):
        if selector == slice(None):
            continue
        if type(selector) is int:
            if selector != 0 or extent != 1:
                return False
            continue
        if not isinstance(selector, Node) or (item := _direct_axis(selector)) is None:
            return False
        axis, scalar = item
        if axis in seen or axis not in axes:
            return False
        size, block = axes[axis]
        if size != extent or scalar and block != 1:
            return False
        seen.add(axis)
    return all(axis in seen or extent <= block for axis, extent, block in plan.axes)


def _logical_scan_extent(plan: PositionalRootPlan, source: Node, axis: int) -> int:
    size = source.meta["val"].shape[axis]
    if isinstance(size, int):
        extent = plan.domains.static(source, axis)
    else:
        env = CompileEnvironment.current()
        block = env.resolve_block_id(size)
        if block is None:
            raise _Unsupported("unresolved computed scan axis")
        block = env.canonical_block_id(block)
        active = next((item for item in plan.axes if item[0] == block), None)
        if active is not None:
            _, extent, tile = active
            if extent > tile:
                raise _Unsupported("computed scan axis spans multiple CTAs")
        else:
            extent = env.block_sizes[block].size
    physical = chain._shape(source)[axis]
    if type(extent) is not int or not 1 <= extent <= physical <= MAX_SCAN_EXTENT:
        raise _Unsupported("unbounded computed scan extent")
    return extent


@dataclasses.dataclass(frozen=True)
class ComputedScanPlan(PositionalRootPlan):
    scan: Node
    extent: int
    nodes: tuple[Node, ...]
    helper_nodes: tuple[Node, ...]
    facts: tuple[object, ...]
    storage: tuple[object, ...]
    domains_snapshot: tuple[object, ...]
    physical: tuple[object, ...]
    staging: StagedScanLayout | None = None

    def check(self) -> None:
        if (
            self.facts != _node_facts((*self.nodes, *self.helper_nodes))
            or self.storage != _storage_facts(self.nodes)
            or self.domains_snapshot != tuple(self.domains.extents.items())
            or self.physical
            != (
                self.root_graph_id,
                self.axes,
                self.threads,
                self.shared_bytes,
                self.shared_capacity,
                self.store,
                self.scan,
                self.extent,
                chain._shape(cast("Node", self.scan.args[1])),
                self.staging,
            )
            or tuple(self.scan.graph.nodes) != self.nodes
            or tuple(_scan_helper(self.scan).graph.nodes) != self.helper_nodes
        ):
            raise _Unsupported("computed scan graph, storage or domain changed")
        if self.staging is not None:
            self.staging.check()


def _computed_scan_graph(
    graphs: Sequence[GraphInfo],
) -> tuple[tuple[Node, ...], Node, Node, Node] | None:
    root = chain._root_graph(graphs)
    if root is None:
        return None
    nodes = tuple(root.graph.nodes)
    scans = [node for node in nodes if node.target is scan_ops._associative_scan]
    stores = [node for node in nodes if node.target is memory_ops.store]
    if len(scans) != 1 or len(stores) != 1:
        return None
    scan, store = scans[0], stores[0]
    if len(store.args) != 4 or not chain._additive_scan(scan):
        return None
    source = cast("Node", scan.args[1])
    if _raw_input(source) or not isinstance(store.args[2], Node):
        return None
    if not _scan_view(store.args[2], scan):
        return None
    live = chain._ancestors(store) | {store}
    for node in nodes:
        if node.op in ("placeholder", "output"):
            continue
        if node not in live or node.op != "call_function":
            return None
        if node in (scan, store):
            continue
        if (
            node.target not in _EXPRESSION_TARGETS
            and chain._pointwise_inputs(node) is None
        ):
            return None
    return nodes, scan, store, source


def has_computed_scan_staging_candidate(graphs: Sequence[GraphInfo]) -> bool:
    """Search discovery only; physical, alias and domain admission is repeated."""
    candidate = _computed_scan_graph(graphs)
    return candidate is not None and candidate[3].meta["val"].ndim == 2


def plan_computed_scan_root(
    graphs: Sequence[GraphInfo],
    df: DeviceFunction,
) -> ComputedScanPlan | None:
    candidate = _computed_scan_graph(graphs)
    if candidate is None:
        return None
    nodes, scan, store, source = candidate
    try:
        plan = plan_root_layout(graphs, df)
        if not _complete_store(plan, store):
            return None
        oracle = HostAliasOracle(CompileEnvironment.current())
        output = cast("Node", store.args[0])
        if any(
            node.target is memory_ops.load
            and (
                not isinstance(node.args[0], Node)
                or node.args[0].target is not _tracing_ops._host_tensor
                or oracle.may_alias(output, node.args[0])
            )
            for node in nodes
        ):
            return None
        axis = cast("int", scan.args[2]) % source.meta["val"].ndim
        extent = _logical_scan_extent(plan, source, axis)
        helper_nodes = tuple(_scan_helper(scan).graph.nodes)
        if any(
            node.meta["val"].dtype != torch.float32
            for node in helper_nodes
            if node.op == "placeholder"
        ):
            return None
        staging = None
        schedule = df.config.config.get(STAGING_KEY, "warp")
        if schedule == "staged_warp":
            staging = staged_scan_layout(
                chain._shape(source), axis, extent, plan.threads
            )
            shared = plan.shared_bytes - _tile_bytes(source) + staging.allocated_bytes
            if plan.shared_capacity and shared > plan.shared_capacity:
                raise _Unsupported("computed scan staging exceeds shared capacity")
            plan = dataclasses.replace(plan, shared_bytes=shared)
        elif schedule != "warp":
            raise _Unsupported("unknown computed scan schedule")
        return ComputedScanPlan(
            root_graph_id=plan.root_graph_id,
            axes=plan.axes,
            threads=plan.threads,
            shared_bytes=plan.shared_bytes,
            shared_capacity=plan.shared_capacity,
            store=plan.store,
            collisions=plan.collisions,
            domains=plan.domains,
            scan=scan,
            extent=extent,
            nodes=nodes,
            helper_nodes=helper_nodes,
            facts=_node_facts((*nodes, *helper_nodes)),
            storage=_storage_facts(nodes),
            domains_snapshot=tuple(plan.domains.extents.items()),
            staging=staging,
            physical=(
                plan.root_graph_id,
                plan.axes,
                plan.threads,
                plan.shared_bytes,
                plan.shared_capacity,
                plan.store,
                scan,
                extent,
                chain._shape(source),
                staging,
            ),
        )
    except (
        _Unsupported,
        chain._UnsupportedChain,
        UnprovenDomain,
        exc.BackendUnsupported,
    ):
        return None


class ComputedScanEmitter(PositionalEmitter):
    def __init__(self, cg: GenerateAST, plan: ComputedScanPlan) -> None:
        plan.check()
        super().__init__(cg, plan)
        self.scan_plan = plan

    def emit_scan(self, node: Node) -> list[str]:
        plan = self.scan_plan
        plan.check()
        if node is not plan.scan or self.boundaries:
            raise _Unsupported("computed scan lost its original input frontier")
        source = cast("Node", node.args[1])
        shape = chain._shape(source)
        axis = cast("int", node.args[2]) % len(shape)
        vectors = math.prod(shape[:axis] + shape[axis + 1 :])
        staged = []
        if plan.staging is None:
            buffer = self.allocate(source)
        else:
            layout = plan.staging
            layout.check()
            buffer = self.fresh("tile")
            self.allocated_bytes += layout.allocated_bytes
            self.allocations.append(
                f"{buffer} = cute.make_tensor(cute.arch.alloc_smem(cutlass.Float32, "
                f"{layout.elements}, alignment=128), {layout.expression})"
            )
            header, positions = self.materialization_loop(shape)
            expression = self.expression()
            value = expression.value(source, positions)
            destination = f"{buffer}[{', '.join(positions)}]"
            # Match the original prefix's inactive scan identity without
            # evaluating its expression outside the logical scan extent.
            staged = self.phase(
                header,
                [
                    f"if {positions[axis]} < {plan.extent}:",
                    *self.indent(
                        [*expression.lines, f"{destination} = cutlass.Float32({value})"]
                    ),
                    "else:",
                    f"    {destination} = cutlass.Float32(0)",
                ],
            )
        name = self.fresh("computed_scan")
        vector, position = f"{name}_vector", f"{name}_position"
        coordinates = [vector] if len(shape) == 2 else []
        coordinates.insert(axis, position)
        coords = tuple(coordinates)
        expression = self.expression()
        value = (
            expression.value(source, coords)
            if plan.staging is None
            else f"{buffer}[{', '.join(coords)}]"
        )
        execution = ChainedExecution(plan.threads, thread="ptp_thread")
        body = [f"{name}_carry = cutlass.Float32(0)"] if plan.extent > 32 else []
        prefix = warp_prefix_point(
            name,
            plan.extent,
            expression.lines,
            f"cutlass.Float32({value})",
            execution=execution,
            initial_add=False,
        )
        body.extend(
            [
                f"for {name}_part in cutlass.range_constexpr({(shape[axis] + 31) // 32}):",
                *self.indent("\n".join(prefix).splitlines()),
                f"    if {position} < {shape[axis]}:",
                f"        {buffer}[{', '.join(coords)}] = {name}_acc",
            ]
        )
        self.boundaries[node] = buffer
        return staged + self.phase(
            [
                (
                    f"for {vector} in cutlass.range(ptp_thread // 32, {vectors}, "
                    f"{plan.threads // 32}, unroll=1):"
                ),
            ],
            body,
        )
