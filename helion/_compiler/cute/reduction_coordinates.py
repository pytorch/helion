"""Coordinate proofs for full reductions in the device IR.

Three proofs decide when a full reduction needs more than the native scalar
recipe's coordinates: ``independent_reduction_coordinates`` (its result is
consumed along another stored tile axis), ``captured_reduction_coordinates``
(its axis is captured across a device loop) and ``free_iota_reductions`` (a
complete static iota flows into it).  ``reduction_coordinate_proof`` bundles
the three for the fragment admission that consumes them together; the
``GenerateAST``-dependent filters (``physical_capture_axes``,
``owned_iota_reduction_axes``) stay outside the proof and are passed in as
axis sets.  All of them prove coordinate requirements, not operation or
storage support.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._inductor.ir import Reduction
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import tile_ops
from ..device_ir import ForLoopGraphInfo
from ..device_ir import ReductionLoopGraphInfo
from ..device_ir import RootGraphInfo
from ..device_ir import control_flow_parent_entries
from ..inductor_lowering import ReductionLowering

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST


def _full_reductions(
    env: CompileEnvironment, graphs: list[GraphInfo]
) -> Iterator[tuple[GraphInfo, Node, ReductionLowering]]:
    """Every node lowered as a reduction over a reduction block, in order."""
    for info in graphs:
        for node in info.graph.nodes:
            lowering = node.meta.get("lowering")
            if (
                isinstance(lowering, ReductionLowering)
                and env.block_sizes[lowering.block_index].reduction
            ):
                yield info, node, lowering


def independent_reduction_coordinates(
    env: CompileEnvironment,
    graphs: list[GraphInfo],
    *,
    include_rolled_outputs: bool = False,
) -> frozenset[Node]:
    """Find reductions whose result is consumed along another stored tile axis.

    This proves a need for complete logical coordinates, not general operation
    support or memory safety. The ordinary fragment admission and emitter still
    check every operation, extent, layout, predicate and shared allocation.
    Only dependencies inside one GraphInfo participate; cross-loop captures
    and arbitrary computed store indices keep their existing paths. The optional
    rolled-output projection is only for conservative metadata specialization
    discovery: a later persistent config may inline that full reduction.
    """

    def shape_axes(node: Node) -> frozenset[int]:
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            return frozenset()
        return frozenset(
            axis
            for size in value.shape
            if (axis := env.resolve_block_id(size)) is not None
        )

    def stored_axis(index: object) -> int | None:
        if not isinstance(index, Node):
            return None
        if index.target is tile_ops.tile_id:
            index = index.args[0]
        if not isinstance(index, Node) or index.target is not _tracing_ops._get_symnode:
            return None
        axis = env.resolve_block_id(index.meta.get("val"))
        if axis is None or not env.block_sizes[axis].is_tile:
            return None
        return axis

    def full_input_axis(
        node: Node, lowering: ReductionLowering, axes: dict[Node, frozenset[int]]
    ) -> bool:
        reduction = lowering.buffer.data
        assert isinstance(reduction, Reduction)
        (extent,) = reduction.reduction_ranges
        for source in node.all_input_nodes:
            if lowering.block_index in axes[source]:
                return True
            value = source.meta.get("val")
            if (
                isinstance(value, torch.Tensor)
                and isinstance(extent, (int, sympy.Integer))
                and int(extent) > 1
                and any(
                    isinstance(size, int) and size == int(extent)
                    for size in value.shape
                )
            ):
                return True
        return False

    axes = {node: shape_axes(node) for info in graphs for node in info.graph.nodes}
    producers = {
        node: (
            lowering.block_index,
            set().union(*(axes[source] for source in node.all_input_nodes)),
        )
        for _info, node, lowering in _full_reductions(env, graphs)
        if full_input_axis(node, lowering, axes)
    }
    if include_rolled_outputs:
        by_id = {info.graph_id: info for info in graphs}
        for info in graphs:
            for node in info.graph.nodes:
                if node.target is not operator.getitem or len(node.args) != 2:
                    continue
                call, slot = node.args
                if (
                    not isinstance(call, Node)
                    or not _tracing_ops.is_for_loop_target(call.target)
                    or not isinstance(slot, int)
                ):
                    continue
                child = by_id[cast("int", call.args[0])]
                if not isinstance(child, ReductionLoopGraphInfo):
                    continue
                output = next(iter(child.graph.find_nodes(op="output")))
                values = output.args[0]
                if (
                    isinstance(values, (list, tuple))
                    and 0 <= slot < len(values)
                    and isinstance(value := values[slot], Node)
                    and value in producers
                ):
                    producers[node] = producers[value]

    result: set[Node] = set()
    for info in graphs:
        reductions = {
            node: proof for node, proof in producers.items() if node.graph is info.graph
        }
        for store in info.graph.nodes:
            if store.target is not memory_ops.store:
                continue
            indices, value = store.args[1:3]
            if not isinstance(indices, (list, tuple)) or not isinstance(value, Node):
                continue
            output_axes = {
                axis for index in indices if (axis := stored_axis(index)) is not None
            }
            if not output_axes:
                continue
            # A full statistic must actually feed this store's payload. Stop
            # at global loads: memory aliasing is not a data-dependency proof.
            ancestors: set[Node] = set()
            pending = [value]
            while pending:
                node = pending.pop()
                if node.graph is not info.graph or node in ancestors:
                    continue
                ancestors.add(node)
                if node.target is not memory_ops.load:
                    pending.extend(node.all_input_nodes)
            for reduction, (full_axis, source_axes) in reductions.items():
                if reduction not in ancestors:
                    continue
                independent = output_axes - source_axes - {full_axis}
                if not independent:
                    continue
                # Require the independent tile coordinate on the actual value
                # path, not just an unrelated tiled input or an output index.
                pending = list(reduction.users)
                visited: set[Node] = set()
                while pending:
                    node = pending.pop()
                    if node not in ancestors or node in visited:
                        continue
                    visited.add(node)
                    # Broadcasting a statistic back onto its own full axis
                    # retains the existing native reduction coordinates. Later
                    # row broadcasts do not make that normalization independent.
                    if full_axis in axes[node]:
                        continue
                    if axes[node] & independent:
                        result.add(reduction)
                        break
                    pending.extend(node.users)
    return frozenset(result)


def captured_reduction_coordinates(
    env: CompileEnvironment,
    graphs: list[GraphInfo],
    *,
    root_graph_id: int | None = None,
    physical_axes: frozenset[int] = frozenset(),
) -> frozenset[Node]:
    """Identify full captured tensor reductions that need complete ownership.

    A native scalar recipe can represent one captured element per thread, but
    not the complete axis when lane loops surround the capturing device loop.
    Explicitly tiled reduction axes are excluded: their loads and reduction
    coordinates belong to the same tiled iteration instead of a full capture.
    This proves a coordinate requirement, not operation or storage support.
    """
    parents = control_flow_parent_entries(graphs)
    if not parents:
        return frozenset()
    by_graph = {info.graph: info for info in graphs}
    captures: dict[Node, Node] = {}
    for info in graphs:
        if info.graph_id not in parents:
            continue
        call, slot = parents[info.graph_id]
        captures.update(
            zip(
                info.graph.find_nodes(op="placeholder"),
                cast("list[Node]", call.args[slot]),
                strict=True,
            )
        )

    def root_id(info: GraphInfo) -> int:
        while info.graph_id in parents:
            call, _ = parents[info.graph_id]
            info = by_graph[call.graph]
        return info.graph_id

    def reads_full_capture(node: Node, axis: int) -> bool:
        pending = list(node.all_input_nodes)
        seen: set[Node] = set()
        while pending:
            value = pending.pop()
            if value in seen:
                continue
            seen.add(value)
            if value in captures:
                info = by_graph[value.graph]
                tensor = value.meta.get("val")
                if (
                    isinstance(info, ForLoopGraphInfo)
                    and not isinstance(info, ReductionLoopGraphInfo)
                    and axis not in info.block_ids
                    and isinstance(tensor, torch.Tensor)
                    and any(env.resolve_block_id(size) == axis for size in tensor.shape)
                ):
                    return True
                pending.append(captures[value])
            elif value.target not in (memory_ops.load, _tracing_ops._host_tensor):
                pending.extend(value.all_input_nodes)
        return False

    return frozenset(
        node
        for info, node, lowering in _full_reductions(env, graphs)
        if (root_graph_id is None or root_id(info) == root_graph_id)
        and lowering.block_index not in physical_axes
        and reads_full_capture(node, lowering.block_index)
    )


def free_iota_reductions(
    env: CompileEnvironment, graphs: list[GraphInfo]
) -> dict[Node, tuple[Node, ...]]:
    """Prove a full rank-one static iota flows into a root vector reduction.

    Scalar-grid indices and host tensor shapes are not element provenance.
    Follow only same-graph full-vector values, including tensor load indices;
    symbolic/tiled vectors already carry their own block identity. Captures and
    multi-axis contractions retain their existing ownership proofs.
    """
    result = {}
    for info, node, lowering in _full_reductions(env, graphs):
        if not isinstance(info, RootGraphInfo):
            continue
        reduction = lowering.buffer.data
        assert isinstance(reduction, Reduction)
        if reduction.ranges or len(reduction.reduction_ranges) != 1:
            continue
        extent = reduction.reduction_ranges[0]
        if not isinstance(extent, (int, sympy.Integer)) or extent <= 1:
            continue
        pending = list(node.all_input_nodes)
        seen = set()
        iotas = []
        while pending:
            value = pending.pop()
            if value in seen or value.graph is not info.graph:
                continue
            seen.add(value)
            tensor = value.meta.get("val")
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 1
                or not isinstance(tensor.shape[0], int)
                or tensor.shape[0] != extent
                or value.target is _tracing_ops._host_tensor
            ):
                continue
            if value.target is torch.ops.prims.iota.default:
                iotas.append(value)
            else:
                pending.extend(value.all_input_nodes)
        if iotas:
            result[node] = tuple(iotas)
    return result


@dataclass(frozen=True)
class ReductionCoordinates:
    """The reductions each coordinate proof admits for one set of graphs."""

    independent: frozenset[Node]
    captured: frozenset[Node]
    free_iota: frozenset[Node]

    def __contains__(self, node: object) -> bool:
        return (
            node in self.independent or node in self.captured or node in self.free_iota
        )


def reduction_coordinate_proof(
    env: CompileEnvironment,
    graphs: list[GraphInfo],
    *,
    physical_axes: frozenset[int] = frozenset(),
    owned_iota_axes: frozenset[int] = frozenset(),
) -> ReductionCoordinates:
    """Run the three coordinate proofs over ``graphs``.

    ``physical_axes`` (captures already covered by contiguous warp lanes) and
    ``owned_iota_axes`` (iota axes the producer already owns) come from the
    ``GenerateAST``-dependent filters below; the proof itself reads only the
    environment and the device IR.
    """
    return ReductionCoordinates(
        independent=independent_reduction_coordinates(env, graphs),
        captured=captured_reduction_coordinates(
            env, graphs, physical_axes=physical_axes
        ),
        free_iota=frozenset(
            node
            for node in free_iota_reductions(env, graphs)
            if node.meta["lowering"].block_index not in owned_iota_axes
        ),
    )


def physical_capture_axes(cg: GenerateAST) -> frozenset[int]:
    """Recognize full scalar captures already covered by contiguous warp lanes.

    Each captured vector has one scalar per physical lane, with the complete
    static axis mapped to threadIdx.x. No synthetic lane loop or vector packet
    can straddle the capturing loop. Other coordinate requirements still go
    through the ordinary fragment-support proof.
    """
    from ..compile_environment import CompileEnvironment
    from ..reduction_strategy import PersistentReductionStrategy

    env = CompileEnvironment.current()
    fn = cg.device_function
    graphs = cg.host_function.device_ir.graphs
    if (
        fn.cute_state.lanes.simt_cluster_n != 1
        or len(cg.host_function.device_ir.root_ids) != 1
        or any(
            not isinstance(info, (RootGraphInfo, ForLoopGraphInfo)) for info in graphs
        )
    ):
        return frozenset()
    reductions = captured_reduction_coordinates(env, graphs)
    candidates = {node.meta["lowering"].block_index for node in reductions}
    parents = control_flow_parent_entries(graphs)
    result = set()
    for axis in candidates:
        size = env.block_sizes[axis].numel
        if (
            not isinstance(size, (int, sympy.Integer))
            or not 1 <= size <= 32
            or int(size) & (int(size) - 1)
        ):
            continue
        strategies = [s for s in fn.tile_strategy.strategies if axis in s.block_ids]
        if len(strategies) != 1:
            continue
        strategy = strategies[0]
        if (
            not isinstance(strategy, PersistentReductionStrategy)
            or strategy._synthetic_cute_lane_var is not None
            or fn.tile_strategy.thread_axis_for_block_id(axis) != 0
            or fn.tile_strategy.thread_extent_for_block_id(axis) != size
            or fn.resolved_block_size(axis) != size
            or env.config_spec.cute_vector_widths.config_get(
                cast("list[int]", fn.config.config.get("cute_vector_widths", []) or []),
                axis,
                1,
            )
            != 1
        ):
            continue
        seen = False
        valid = True
        for info in graphs:
            if info.graph_id not in parents:
                continue
            for node in info.graph.find_nodes(op="placeholder"):
                value = node.meta.get("val")
                if not isinstance(value, torch.Tensor):
                    continue
                axes = [env.resolve_block_id(dim) for dim in value.shape]
                if axis not in axes:
                    continue
                seen = True
                if sum(bid == axis for bid in axes) != 1 or any(
                    bid != axis and not env.known_equal(dim, 1)
                    for bid, dim in zip(axes, value.shape, strict=True)
                ):
                    valid = False
        if seen and valid:
            result.add(axis)
    return frozenset(result)


def owned_iota_reduction_axes(cg: GenerateAST) -> frozenset[int]:
    """Retain native reductions when the producer uses that active full axis.

    Use the ordinary iota resolver itself, not coincidental selected extents.
    A rolled root producer cannot see its reduction loop's coordinate; it must
    be evaluated by the complete-fragment owner before rolling instead.
    """
    from ..compile_environment import CompileEnvironment
    from .cute_reshape import _resolve_dim_block_id

    env = CompileEnvironment.current()
    result = set()
    for node, iotas in free_iota_reductions(
        env, cg.host_function.device_ir.graphs
    ).items():
        lowering = node.meta["lowering"]
        assert isinstance(lowering, ReductionLowering)
        axis = lowering.block_index
        if all(
            _resolve_dim_block_id(cg, iota.meta["val"], 0) == axis for iota in iotas
        ):
            result.add(axis)
    return frozenset(result)
