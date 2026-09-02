from __future__ import annotations

import dataclasses
from itertools import starmap
import logging
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch

from ..autotuner.config_spec import SIZED_REDUCTION_CATEGORIES
from ..autotuner.config_spec import AccumulatorFact
from ..autotuner.config_spec import DotAxes
from ..autotuner.config_spec import DotAxisKind
from ..autotuner.config_spec import DotSite
from ..autotuner.config_spec import KernelGridFact
from ..autotuner.config_spec import KernelMatmulFact
from ..autotuner.config_spec import LiveTile
from ..autotuner.config_spec import LoopAxisFact
from ..autotuner.config_spec import MemoryOpFact
from ..autotuner.config_spec import PipelinedRegion
from ..autotuner.config_spec import PointwiseElementwiseFact
from ..autotuner.config_spec import ResidentRegion
from ..autotuner.config_spec import ResolvedMatmulFact
from ..autotuner.config_spec import RootGridFact
from ..autotuner.config_spec import SymbolicLoopBound
from ..language import _tracing_ops
from .compile_environment import FixedBlockSizeSource
from .compile_environment import _symint_free_symbols
from .compile_environment import _symint_sympy_expr
from .indexing_strategy import subscript_index_scale
from .indexing_strategy import subscript_tile_info

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Container
    from collections.abc import Iterable

    import sympy

    from ..autotuner.config_spec import ConfigSpec
    from .compile_environment import CompileEnvironment
    from .device_ir import DeviceIR
    from .device_ir import GraphInfo
    from .host_function import HostFunction


log = logging.getLogger(__name__)


def matmul_operand_positions() -> dict[object, tuple[int, int]]:
    """Matmul/dot FX targets mapped to their lhs/rhs argument positions."""
    from ..language import matmul_ops

    return {
        matmul_ops.dot_scaled: (0, 3),
        matmul_ops.dot: (0, 1),
        torch.ops.aten.mm.default: (0, 1),
        torch.ops.aten.bmm.default: (0, 1),
        torch.ops.aten.bmm.dtype: (0, 1),
        torch.ops.aten.addmm.default: (1, 2),
        torch.ops.aten.baddbmm.default: (1, 2),
    }


def trace_back_to_load(arg: object, load_op: object) -> torch.fx.Node | None:
    """Follow a matmul operand through pass-through ops to one load node."""
    cur = arg
    for _ in range(8):
        if not isinstance(cur, torch.fx.Node):
            return None
        if cur.target is load_op:
            return cur
        tensor_inputs = [
            value
            for value in cur.args
            if isinstance(value, torch.fx.Node)
            and isinstance(value.meta.get("val"), torch.Tensor)
        ]
        if len(tensor_inputs) != 1:
            return None
        cur = tensor_inputs[0]
    return None


def _rank_reduction_scaled_baddbmm_batch_block_id(
    node: torch.fx.Node,
    env: CompileEnvironment,
) -> int | None:
    """Detect the narrow Triton bug pattern for the H100 matmul heuristic.

    Triton's SM90 layout solver fails above ``num_stages=1`` when a dot-derived
    row reduction rescales a loop-carried ``baddbmm`` accumulator. Return its
    batch axis so the heuristic can recognize the singleton-batch WGMMA form
    and force one stage; this fact is not a general hardware constraint.
    """
    if node.target is not torch.ops.aten.baddbmm.default:
        return None
    output = node.meta.get("val")
    if not isinstance(output, torch.Tensor) or output.ndim != 3:
        return None
    scaled_acc = node.args[0]
    if not isinstance(scaled_acc, torch.fx.Node):
        return None
    if scaled_acc.target is not torch.ops.aten.mul.Tensor:
        return None

    def is_carried(value: object) -> bool:
        if not isinstance(value, torch.fx.Node):
            return False
        tensor = value.meta.get("val")
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != output.ndim
            or not all(map(env.known_equal, tensor.shape, output.shape))
        ):
            return False
        return value.op == "placeholder" or (
            value.target is _tracing_ops._new_var
            and bool(value.args)
            and isinstance(value.args[0], torch.fx.Node)
            and value.args[0].op == "placeholder"
        )

    accumulator, scale = scaled_acc.args[:2]
    if not is_carried(accumulator):
        accumulator, scale = scale, accumulator
    if not is_carried(accumulator) or not isinstance(scale, torch.fx.Node):
        return None

    from .inductor_lowering import ReductionLowering

    dot_targets = matmul_operand_positions()
    pending = [(scale, False)]
    seen: set[tuple[torch.fx.Node, bool]] = set()
    while pending:
        candidate, reduced = pending.pop()
        value = candidate.meta.get("val")
        reduced |= (
            isinstance(candidate.meta.get("lowering"), ReductionLowering)
            and isinstance(value, torch.Tensor)
            and len(value.shape) == 2
            and all(map(env.known_equal, value.shape, output.shape[:-1]))
        )
        state = (candidate, reduced)
        if state in seen:
            continue
        seen.add(state)
        if reduced and candidate.target in dot_targets:
            return env.get_block_id(output.shape[0])
        pending.extend((parent, reduced) for parent in candidate.all_input_nodes)
    return None


def _immovable_extent(
    env: CompileEnvironment,
    spec: ConfigSpec,
    block_id: int,
) -> int | None:
    """Return the per-program extent of an axis the config cannot move."""
    if not (0 <= block_id < len(env.block_sizes)):
        return None
    info = env.block_sizes[block_id]
    if not isinstance(info.block_size_source, FixedBlockSizeSource):
        return None
    try:
        value = info.block_size_source.from_config(
            spec._base_default_config(),
            info,
        )
    except Exception:
        return None
    if value is None:
        return None
    try:
        return max(1, int(env.size_hint(value)))
    except Exception:
        return None


def _load_needs_eviction_tunable(node: torch.fx.Node) -> bool:
    """A load gets an eviction-policy slot only when the user did not pass one."""
    eviction_policy_arg = node.kwargs.get("eviction_policy")
    if eviction_policy_arg is None and len(node.args) >= 4:
        eviction_policy_arg = node.args[3]
    return eviction_policy_arg is None


def _accessed_tensor_fake(node: torch.fx.Node) -> torch.Tensor | None:
    """Fake tensor of the buffer a load/store accesses."""
    arg = node.args[0] if node.args else None
    if isinstance(arg, torch.fx.Node):
        value = arg.meta.get("val")
        if isinstance(value, torch.Tensor):
            return value
    return None


def _subscript_block_id(env: CompileEnvironment, subscript: object) -> int | None:
    """Return the block axis indexed by a tile-provenance subscript."""
    info = subscript_tile_info(env, subscript)
    return info.block_id if info is not None else None


def _store_axis_key(
    env: CompileEnvironment,
    store_node: torch.fx.Node,
) -> tuple[int | None, ...]:
    """Return the full block-axis key of one store subscript."""
    fake = _accessed_tensor_fake(store_node)
    index_list = store_node.args[1] if len(store_node.args) >= 2 else None
    if fake is None or not isinstance(index_list, (list, tuple)):
        return ()
    key: list[int | None] = []
    for position, subscript in enumerate(index_list):
        if isinstance(subscript, int) or position >= fake.ndim:
            continue
        block_id = _subscript_block_id(env, subscript)
        if block_id is None:
            block_id = env.resolve_block_id(fake.shape[position])
        key.append(block_id)
    return tuple(key)


def _classify_load_dataflow(
    load_node: torch.fx.Node,
    reduction_nodes: set[int],
    env: CompileEnvironment,
) -> tuple[set[int], tuple[tuple[int | None, ...], ...]]:
    """Trace a load forward to reductions and stores without crossing either."""
    from ..language import memory_ops

    reductions_fed: set[int] = set()
    stores_fed: set[tuple[int | None, ...]] = set()
    seen: set[int] = set()
    stack = list(load_node.users)
    while stack:
        user = stack.pop()
        if id(user) in reduction_nodes:
            reductions_fed.add(id(user))
            continue
        if id(user) in seen:
            continue
        seen.add(id(user))
        if user.op == "call_function" and user.target is memory_ops.store:
            stores_fed.add(_store_axis_key(env, user))
            continue
        stack.extend(user.users)
    return reductions_fed, tuple(
        sorted(
            stores_fed,
            key=lambda axes: tuple(-1 if axis is None else axis for axis in axes),
        )
    )


def tile_rank(dims: tuple[int | None, ...]) -> int:
    """Number of block dimensions spanned by a tile."""
    return sum(dim is not None for dim in dims)


def tile_set_rank_profile(
    tiles: Iterable[tuple[int | None, ...]],
    max_rank: int,
) -> tuple[int, ...]:
    """Block-size-free lexicographic footprint key, highest rank first."""
    by_rank: dict[int, int] = {}
    for tile in tiles:
        rank = tile_rank(tile)
        if rank:
            by_rank[rank] = by_rank.get(rank, 0) + 1
    return tuple(by_rank.get(rank, 0) for rank in range(max_rank, 0, -1))


def _live_tile_kind(node: torch.fx.Node, dot_targets: frozenset[object]) -> str:
    from ..language import memory_ops

    if node.op == "placeholder":
        return "carry"
    if node.op == "call_function":
        if node.target in dot_targets:
            return "dot_out"
        if node.target is memory_ops.load:
            return "load"
    return "other"


def _tile_from_tensor(
    value: object,
    env: CompileEnvironment,
    *,
    kind: str,
    stageable: bool | None = None,
) -> LiveTile | None:
    if not (isinstance(value, torch.Tensor) and value.shape):
        return None
    dims = tuple(env.resolve_block_id(size) for size in value.shape)
    static_dims: list[int | None] = []
    for size, block_id in zip(value.shape, dims, strict=False):
        if block_id is not None:
            static_dims.append(None)
            continue
        try:
            static_dims.append(int(env.size_hint(size)))
        except Exception:
            static_dims.append(None)
    return LiveTile(
        dim_block_ids=dims,
        static_dims=tuple(static_dims),
        itemsize=value.dtype.itemsize,
        kind=kind,
        stageable=stageable,
    )


def _index_depends_on_loop(
    obj: object,
    env: CompileEnvironment,
    loop_block_ids: frozenset[int],
    seen: set[torch.fx.Node] | None = None,
) -> bool | None:
    """Whether an FX index is proven to depend on an enclosing loop axis."""
    if isinstance(obj, (list, tuple)):
        states = [
            _index_depends_on_loop(item, env, loop_block_ids, seen) for item in obj
        ]
        if any(state is True for state in states):
            return True
        if any(state is None for state in states):
            return None
        return False
    if isinstance(obj, dict):
        return _index_depends_on_loop(tuple(obj.values()), env, loop_block_ids, seen)
    if isinstance(obj, torch.fx.Node):
        seen = set() if seen is None else seen
        if obj in seen:
            return False
        seen.add(obj)
        value = obj.meta.get("val")
        known_origin = False
        try:
            block_id = env.resolve_block_id(value)
        except Exception:
            block_id = None
        if block_id is not None:
            known_origin = True
            if block_id in loop_block_ids:
                return True
        if isinstance(value, torch.Tensor):
            for dim in value.shape:
                try:
                    dim_block_id = env.resolve_block_id(dim)
                except Exception:
                    dim_block_id = None
                if dim_block_id is not None:
                    known_origin = True
                    if dim_block_id in loop_block_ids:
                        return True
        if obj.op == "placeholder":
            return False if known_origin else None
        children = (*obj.args, *obj.kwargs.values())
        if children:
            state = _index_depends_on_loop(children, env, loop_block_ids, seen)
            if state is not False:
                return state
            return False
        if obj.op == "get_attr":
            return False
        return False if known_origin else None
    if isinstance(obj, torch.SymInt):
        try:
            block_id = env.resolve_block_id(obj)
        except Exception:
            return None
        return block_id in loop_block_ids if block_id is not None else None
    if isinstance(obj, torch.Tensor):
        return None
    return False


def _live_node_steps(
    nodes: tuple[torch.fx.Node, ...],
    tracked_nodes: Container[torch.fx.Node],
    last_use: dict[torch.fx.Node, int],
    *,
    last_use_default: int = -1,
) -> Iterable[set[torch.fx.Node]]:
    """Yield the tracked nodes live after each graph node is introduced."""
    live: set[torch.fx.Node] = set()
    for index, node in enumerate(nodes):
        if node in tracked_nodes:
            live.add(node)
        yield live
        live = {
            value for value in live if last_use.get(value, last_use_default) > index
        }


@dataclasses.dataclass
class GraphAnalysis:
    """Shared structural and liveness observations for one DeviceIR graph."""

    graph_id: int
    nodes: tuple[torch.fx.Node, ...]
    is_reduction_loop: bool
    block_ids: frozenset[int]
    block_id_order: tuple[int, ...]
    live_tile_steps: tuple[tuple[LiveTile, ...], ...]
    peak_live_tiles: tuple[LiveTile, ...]
    peak_dot_output_tiles: tuple[LiveTile, ...]
    peak_promoted_lhs_tiles: tuple[LiveTile, ...]
    dot_nodes: tuple[torch.fx.Node, ...]
    reduction_occurrences: tuple[int, ...]
    reduction_axis_by_node_id: dict[int, int]
    reduction_input_itemsizes: tuple[tuple[int, int], ...]
    memory_tiles: tuple[tuple[torch.fx.Node, LiveTile], ...]
    _memory_tiles_by_loop_axes: dict[frozenset[int], tuple[LiveTile, ...]] = (
        dataclasses.field(
            default_factory=dict,
            repr=False,
        )
    )

    @classmethod
    def build(
        cls,
        graph_info: GraphInfo,
        env: CompileEnvironment,
        *,
        is_reduction_loop: bool,
        dot_targets: frozenset[object],
        resolve_placeholder: Callable[[torch.fx.Node], torch.fx.Node],
    ) -> GraphAnalysis:
        from ..language import memory_ops
        from .inductor_lowering import ReductionLowering

        graph = graph_info.graph
        nodes = tuple(graph.nodes)
        last_use: dict[torch.fx.Node, int] = {}
        tile_details: dict[torch.fx.Node, LiveTile] = {}
        dot_details: dict[torch.fx.Node, LiveTile] = {}
        promoted_lhs_details: dict[torch.fx.Node, LiveTile] = {}
        dot_nodes: list[torch.fx.Node] = []
        reduction_occurrences: list[int] = []
        seen_reductions: set[int] = set()
        reduction_axis_by_node_id: dict[int, int] = {}
        reduction_input_itemsizes: list[tuple[int, int]] = []
        memory_tiles: list[tuple[torch.fx.Node, LiveTile]] = []
        operand_positions = matmul_operand_positions()
        promoted_lhs_nodes: set[torch.fx.Node] = set()
        for node in nodes:
            positions = operand_positions.get(node.target)
            if (
                node.op != "call_function"
                or positions is None
                or positions[0] >= len(node.args)
            ):
                continue
            lhs = node.args[positions[0]]
            if (
                isinstance(lhs, torch.fx.Node)
                and resolve_placeholder(lhs).target is not memory_ops.load
            ):
                promoted_lhs_nodes.add(lhs)

        for index, node in enumerate(nodes):
            for input_node in node.all_input_nodes:
                last_use[input_node] = index

            kind = _live_tile_kind(node, dot_targets)
            tile = _tile_from_tensor(node.meta.get("val"), env, kind=kind)
            if tile is not None and kind == "dot_out":
                dot_details[node] = tile
            if tile is not None and node in promoted_lhs_nodes:
                tile = tile._replace(promoted_lhs=True)
                promoted_lhs_details[node] = tile
            if tile is not None and any(
                block_id is not None for block_id in tile.dim_block_ids
            ):
                tile_details[node] = tile

            if node.op == "call_function" and node.target in dot_targets:
                dot_nodes.append(node)

            lowering = node.meta.get("lowering")
            if isinstance(lowering, ReductionLowering):
                block_id = getattr(lowering, "block_index", None)
                if isinstance(block_id, int):
                    reduction_axis_by_node_id[id(node)] = block_id
                    if block_id not in seen_reductions:
                        seen_reductions.add(block_id)
                        reduction_occurrences.append(block_id)
                    for input_node in node.all_input_nodes:
                        input_value = input_node.meta.get("val")
                        if isinstance(input_value, torch.Tensor):
                            reduction_input_itemsizes.append(
                                (block_id, input_value.element_size())
                            )
                            break

            if node.op != "call_function":
                continue
            if node.target is memory_ops.load:
                memory_tile = _tile_from_tensor(
                    node.meta.get("val"),
                    env,
                    kind="load",
                )
            elif node.target is memory_ops.store:
                stored_value = None
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node):
                        candidate = arg.meta.get("val")
                        if isinstance(candidate, torch.Tensor) and candidate.shape:
                            stored_value = candidate
                memory_tile = _tile_from_tensor(stored_value, env, kind="store")
            else:
                memory_tile = None
            if memory_tile is not None:
                memory_tiles.append((node, memory_tile))

        live_tile_steps: list[tuple[LiveTile, ...]] = []
        seen_steps: set[frozenset[int]] = set()
        max_rank = max(
            (tile_rank(tile.dim_block_ids) for tile in tile_details.values()),
            default=0,
        )
        best_key: tuple[int, ...] = ()
        peak_live_tiles: tuple[LiveTile, ...] = ()
        for live in _live_node_steps(nodes, tile_details, last_use):
            if live:
                step_key = frozenset(id(value) for value in live)
                step = tuple(tile_details[value] for value in live)
                if step_key not in seen_steps:
                    seen_steps.add(step_key)
                    live_tile_steps.append(step)
                key = tile_set_rank_profile(
                    (tile_details[value].dim_block_ids for value in live),
                    max_rank,
                )
                if key > best_key:
                    best_key = key
                    peak_live_tiles = step

        def peak_role_tiles(
            details: dict[torch.fx.Node, LiveTile],
            *,
            last_use_default: int = -1,
        ) -> tuple[LiveTile, ...]:
            best: tuple[LiveTile, ...] = ()
            best_key = (-1, -1)
            for live in _live_node_steps(
                nodes,
                details,
                last_use,
                last_use_default=last_use_default,
            ):
                tiles = tuple(details[value] for value in live)
                key = (
                    sum(tile_rank(tile.dim_block_ids) for tile in tiles),
                    len(tiles),
                )
                if key > best_key:
                    best_key = key
                    best = tiles
            return best

        peak_dot_output_tiles = peak_role_tiles(
            dot_details,
            last_use_default=len(nodes),
        )
        peak_promoted_lhs_tiles = peak_role_tiles(promoted_lhs_details)

        return cls(
            graph_id=graph_info.graph_id,
            nodes=nodes,
            is_reduction_loop=is_reduction_loop,
            block_ids=frozenset(getattr(graph_info, "block_ids", ()) or ()),
            block_id_order=tuple(getattr(graph_info, "block_ids", ()) or ()),
            live_tile_steps=tuple(live_tile_steps),
            peak_live_tiles=peak_live_tiles,
            peak_dot_output_tiles=peak_dot_output_tiles,
            peak_promoted_lhs_tiles=peak_promoted_lhs_tiles,
            dot_nodes=tuple(dot_nodes),
            reduction_occurrences=tuple(reduction_occurrences),
            reduction_axis_by_node_id=reduction_axis_by_node_id,
            reduction_input_itemsizes=tuple(reduction_input_itemsizes),
            memory_tiles=tuple(memory_tiles),
        )

    def reaches_output(self, node: torch.fx.Node, limit: int = 64) -> bool:
        """Whether a value reaches the graph output under the legacy bounded walk."""
        frontier = [node]
        seen: set[torch.fx.Node] = {node}
        steps = 0
        while frontier and steps < limit:
            steps += 1
            current = frontier.pop()
            for user in current.users:
                if user.op == "output":
                    return True
                if user not in seen:
                    seen.add(user)
                    frontier.append(user)
        return False

    def memory_tiles_for_loop_axes(
        self,
        env: CompileEnvironment,
        loop_block_ids: frozenset[int] = frozenset(),
    ) -> tuple[LiveTile, ...]:
        """Memory tiles with load stageability resolved for the given loop axes."""
        cached = self._memory_tiles_by_loop_axes.get(loop_block_ids)
        if cached is not None:
            return cached
        out: list[LiveTile] = []
        for node, tile in self.memory_tiles:
            if tile.kind == "load":
                stageable = (
                    _index_depends_on_loop(node.args[1], env, loop_block_ids)
                    if len(node.args) > 1 and loop_block_ids
                    else False
                )
            else:
                stageable = False
            out.append(tile._replace(stageable=stageable))
        result = tuple(out)
        self._memory_tiles_by_loop_axes[loop_block_ids] = result
        return result


@dataclasses.dataclass
class DeviceIRAnalysis:
    """Shared graph interpretation for all fact builders of one DeviceIR."""

    graphs: tuple[GraphAnalysis, ...]
    by_id: dict[int, GraphAnalysis]
    child_loops: dict[int, tuple[tuple[int, frozenset[int]], ...]]
    parent_of: dict[int, int]
    loop_block_ids: dict[int, frozenset[int]]
    loop_calls: dict[int, tuple[torch.fx.Node, ...]]
    dot_nodes: tuple[tuple[int, torch.fx.Node], ...]
    accumulator_inputs: tuple[torch.fx.Node, ...]
    kernel_grid_fact: KernelGridFact

    @classmethod
    def build(
        cls,
        device_ir: DeviceIR,
        env: CompileEnvironment,
    ) -> DeviceIRAnalysis:
        from .device_ir import ForLoopGraphInfo
        from .device_ir import NodeArgsGraphInfo
        from .device_ir import ReductionLoopGraphInfo

        dot_targets = frozenset(matmul_operand_positions())
        graph_info_by_graph = {
            graph_info.graph: graph_info for graph_info in device_ir.graphs
        }

        def resolve_placeholder(node: torch.fx.Node) -> torch.fx.Node:
            seen: set[torch.fx.Node] = set()
            while node.op == "placeholder" and node not in seen:
                seen.add(node)
                graph_info = graph_info_by_graph.get(node.graph)
                if not isinstance(graph_info, NodeArgsGraphInfo):
                    break
                try:
                    node = graph_info.placeholder_to_outer_arg(node)
                except KeyError:
                    break
            return node

        graphs = tuple(
            GraphAnalysis.build(
                graph_info,
                env,
                is_reduction_loop=isinstance(
                    graph_info,
                    ReductionLoopGraphInfo,
                ),
                dot_targets=dot_targets,
                resolve_placeholder=resolve_placeholder,
            )
            for graph_info in device_ir.graphs
        )
        by_id = {graph.graph_id: graph for graph in graphs}
        child_loops: dict[int, tuple[tuple[int, frozenset[int]], ...]] = {}
        loop_calls: dict[int, list[torch.fx.Node]] = {}
        graph_parent_of: dict[int, int] = {}
        for graph in graphs:
            if graph.is_reduction_loop:
                continue
            edges: list[tuple[int, frozenset[int]]] = []
            for node in graph.nodes:
                if (
                    node.op != "call_function"
                    or not _tracing_ops.is_for_loop_target(node.target)
                    or not node.args
                    or not isinstance(node.args[0], int)
                ):
                    continue
                body_id = node.args[0]
                loop_calls.setdefault(body_id, []).append(node)
                body = by_id.get(body_id)
                if body is not None:
                    graph_parent_of[body_id] = graph.graph_id
                if body is not None and not body.is_reduction_loop:
                    edges.append((body_id, body.block_ids))
            if edges:
                child_loops[graph.graph_id] = tuple(edges)

        parent_of: dict[int, int] = {}
        loop_block_ids: dict[int, frozenset[int]] = {}
        for parent_id, loop_edges in child_loops.items():
            for body_id, block_ids in loop_edges:
                parent_of[body_id] = parent_id
                loop_block_ids[body_id] = block_ids

        dot_nodes = tuple(
            (graph.graph_id, node)
            for graph in graphs
            if not graph.is_reduction_loop
            for node in graph.dot_nodes
        )
        accumulator_inputs = tuple(
            node
            for graph_info in device_ir.graphs
            if isinstance(graph_info, ForLoopGraphInfo)
            for node in graph_info.node_args
        )
        roots = tuple(
            RootGridFact(graph_id, tuple(block_ids))
            for graph_id, block_ids in zip(
                device_ir.root_ids,
                device_ir.grid_block_ids,
                strict=True,
            )
        )
        root_ids = {root.root_graph_id for root in roots}
        graph_to_root: list[tuple[int, int]] = []
        for graph in graphs:
            current = graph.graph_id
            seen: set[int] = set()
            while current not in root_ids and current not in seen:
                seen.add(current)
                current = graph_parent_of.get(current, -1)
            if current in root_ids:
                graph_to_root.append((graph.graph_id, current))
        kernel_grid_fact = KernelGridFact(roots, tuple(graph_to_root))
        return cls(
            graphs=graphs,
            by_id=by_id,
            child_loops=child_loops,
            parent_of=parent_of,
            loop_block_ids=loop_block_ids,
            loop_calls={
                graph_id: tuple(calls) for graph_id, calls in loop_calls.items()
            },
            dot_nodes=dot_nodes,
            accumulator_inputs=accumulator_inputs,
            kernel_grid_fact=kernel_grid_fact,
        )

    @property
    def non_reduction_graphs(self) -> tuple[GraphAnalysis, ...]:
        return tuple(graph for graph in self.graphs if not graph.is_reduction_loop)

    def original_reductions(self) -> tuple[tuple[int, int], ...]:
        """Ordered, deduplicated reduction occurrences in original graphs."""
        return tuple(
            (graph.graph_id, block_id)
            for graph in self.non_reduction_graphs
            for block_id in graph.reduction_occurrences
        )

    def reduction_input_itemsize(self, block_id: int) -> int:
        """Legacy last-occurrence input width for one reduction axis."""
        itemsize = 0
        for graph in self.graphs:
            for axis, width in graph.reduction_input_itemsizes:
                if axis == block_id:
                    itemsize = width
        return itemsize

    def kernel_live_tile_steps(self) -> tuple[tuple[LiveTile, ...], ...]:
        return tuple(
            step
            for graph in self.non_reduction_graphs
            for step in graph.live_tile_steps
        )

    def _kernel_peak_role_tiles(self, field: str) -> tuple[LiveTile, ...]:
        """Peak role-specific tile set after adding ancestor loop graphs."""
        tiles_of = {
            graph.graph_id: cast("tuple[LiveTile, ...]", getattr(graph, field))
            for graph in self.non_reduction_graphs
        }
        best: tuple[LiveTile, ...] = ()
        best_key = (-1, -1)
        for graph_id, own in tiles_of.items():
            chain = list(own)
            current = self.parent_of.get(graph_id, -1)
            seen = {graph_id}
            while current in tiles_of and current not in seen:
                seen.add(current)
                chain.extend(tiles_of[current])
                current = self.parent_of.get(current, -1)
            key = (
                sum(tile_rank(tile.dim_block_ids) for tile in chain),
                len(chain),
            )
            if key > best_key:
                best_key = key
                best = tuple(chain)
        return best

    def kernel_peak_dot_outputs(self) -> tuple[LiveTile, ...]:
        """Peak dot-output set after adding accumulator ancestor chains."""
        return self._kernel_peak_role_tiles("peak_dot_output_tiles")

    def kernel_peak_promoted_lhs(self) -> tuple[LiveTile, ...]:
        """Peak transformed-LHS set after adding ancestor loop graphs."""
        return self._kernel_peak_role_tiles("peak_promoted_lhs_tiles")

    def group_live_tiles(
        self,
        group_graph_ids: list[int],
    ) -> dict[int, list[tuple[int | None, ...]]]:
        """Resident peak-live tiles attributed to reduction co-residency groups."""
        group_axes = {
            graph_id: set(self.by_id[graph_id].reduction_occurrences)
            for graph_id in group_graph_ids
        }
        peak_of = {
            graph.graph_id: [tile.dim_block_ids for tile in graph.peak_live_tiles]
            for graph in self.non_reduction_graphs
        }

        def max_by_profile(
            lhs: list[tuple[int | None, ...]],
            rhs: list[tuple[int | None, ...]],
        ) -> list[tuple[int | None, ...]]:
            max_rank = max(
                (tile_rank(tile) for tile in lhs + rhs),
                default=0,
            )
            lhs_key = tile_set_rank_profile(lhs, max_rank)
            rhs_key = tile_set_rank_profile(rhs, max_rank)
            return lhs if lhs_key >= rhs_key else rhs

        group_keys = set(group_graph_ids)
        result: dict[int, list[tuple[int | None, ...]]] = {}
        for graph_id in group_graph_ids:
            axes = group_axes[graph_id]
            tiles = list(peak_of.get(graph_id, ()))
            seen_bodies = {graph_id}
            frontier = [graph_id]
            while frontier:
                current = frontier.pop()
                for body_id, block_ids in self.child_loops.get(current, ()):
                    if body_id in seen_bodies or body_id in group_keys:
                        continue
                    if not axes or (block_ids & axes):
                        seen_bodies.add(body_id)
                        tiles = max_by_profile(tiles, peak_of.get(body_id, []))
                        frontier.append(body_id)
            result[graph_id] = tiles
        return result

    def accumulator_facts(
        self,
        env: CompileEnvironment,
    ) -> list[AccumulatorFact]:
        """Describe every loop-carried tensor accumulator."""
        from .._utils import next_power_of_2

        def resolve_dim(size: object) -> int | None:
            block_id = env.resolve_block_id(size)
            if block_id is not None:
                return block_id
            extent = env.size_hint(cast("int | torch.SymInt", size))
            matches = [
                info.block_id
                for info in env.block_sizes
                if info.reduction and next_power_of_2(info.size_hint()) == extent
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                log.warning(
                    "accumulator dim (padded extent %s) matches %d reduction axes %s; "
                    "cannot differentiate by extent -- left unresolved "
                    "(per_feature_accumulator may miss it)",
                    extent,
                    len(matches),
                    matches,
                )
            return None

        facts: list[AccumulatorFact] = []
        for node in self.accumulator_inputs:
            value = node.meta.get("val")
            if isinstance(value, torch.Tensor):
                facts.append(
                    AccumulatorFact(
                        dim_block_ids=tuple(resolve_dim(size) for size in value.shape),
                        itemsize=value.element_size(),
                    )
                )
        return facts

    def memory_op_facts(
        self,
        env: CompileEnvironment,
        host: HostFunction,
    ) -> list[MemoryOpFact]:
        """Record one fact for every load/store in indexing-slot order."""
        from ..language import memory_ops

        load_op = memory_ops.load
        store_op = memory_ops.store
        operand_positions = matmul_operand_positions()
        operands: dict[torch.fx.Node, str] = {}
        records: list[tuple[torch.fx.Node, MemoryOpFact]] = []
        memory_op_index = 0
        eviction_index = 0

        for graph_analysis in self.graphs:
            reduction_axis_by_id = graph_analysis.reduction_axis_by_node_id
            reduction_nodes = set(reduction_axis_by_id)
            for node in graph_analysis.nodes:
                if node.op != "call_function":
                    continue

                positions = operand_positions.get(node.target)
                if positions is not None:
                    for arg_index, operand in (
                        (positions[0], "lhs"),
                        (positions[1], "rhs"),
                    ):
                        if arg_index < len(node.args):
                            load = trace_back_to_load(node.args[arg_index], load_op)
                            if load is not None:
                                operands.setdefault(load, operand)
                    continue

                is_load = node.target is load_op
                if not (is_load or node.target is store_op):
                    continue

                this_eviction_index: int | None = None
                if is_load and _load_needs_eviction_tunable(node):
                    this_eviction_index = eviction_index
                    eviction_index += 1

                fake = _accessed_tensor_fake(node)
                origin = host.tensor_to_origin.get(fake) if fake is not None else None

                reductions_fed: tuple[tuple[int, int], ...] = ()
                stores_fed: tuple[tuple[int | None, ...], ...] = ()
                if is_load:
                    feeds, stores_fed = _classify_load_dataflow(
                        node,
                        reduction_nodes,
                        env,
                    )
                    per_axis: dict[int, int] = {}
                    for fed_id in feeds:
                        axis = reduction_axis_by_id[fed_id]
                        per_axis[axis] = per_axis.get(axis, 0) + 1
                    reductions_fed = tuple(sorted(per_axis.items()))

                indexed_block_ids: tuple[int | None, ...] = ()
                subscript_block_ids: tuple[int | None, ...] = ()
                subscript_strides: tuple[int, ...] = ()
                subscript_index_scales: tuple[int, ...] = ()
                subscript_affine_block_ids: tuple[int | None, ...] = ()
                subscript_extents: tuple[int, ...] = ()
                inner_extent: int | None = None
                accessed_numel = 0
                if fake is not None:
                    accessed_numel = 1
                    fake_strides = fake.stride()
                    for dim_index, dim_size in enumerate(fake.shape):
                        if fake_strides[dim_index] != 0:
                            accessed_numel *= env.size_hint(dim_size)
                    index_list = node.args[1] if len(node.args) >= 2 else None
                    if isinstance(index_list, (list, tuple)):
                        indexed_positions = [
                            position
                            for position, subscript in enumerate(index_list)
                            if not isinstance(subscript, int) and position < fake.ndim
                        ]
                        indexed_block_ids = tuple(
                            env.resolve_block_id(fake.shape[position])
                            for position in indexed_positions
                        )
                        subscript_block_ids = tuple(
                            _subscript_block_id(env, index_list[position])
                            for position in indexed_positions
                        )
                        subscript_strides = tuple(
                            env.size_hint(fake_strides[position])
                            for position in indexed_positions
                        )
                        affine = [
                            subscript_index_scale(env, index_list[position])
                            for position in indexed_positions
                        ]
                        subscript_affine_block_ids = tuple(
                            block_id for block_id, _scale in affine
                        )
                        subscript_index_scales = tuple(
                            scale for _block_id, scale in affine
                        )
                        subscript_extents = tuple(
                            env.size_hint(fake.shape[position])
                            for position in indexed_positions
                        )
                    if fake.ndim >= 2:
                        inner_extent = env.size_hint(fake.shape[-1])

                records.append(
                    (
                        node,
                        MemoryOpFact(
                            indexing_index=memory_op_index,
                            kind="load" if is_load else "store",
                            eviction_index=this_eviction_index,
                            tensor_name=origin.root_rw_name() if origin else None,
                            dtype=fake.dtype if fake is not None else None,
                            ndim=fake.ndim if fake is not None else 0,
                            num_reuses=len(node.users) if is_load else 0,
                            matmul_operand=None,
                            graph_id=graph_analysis.graph_id,
                            reductions_fed=reductions_fed,
                            stores_fed=stores_fed,
                            indexed_block_ids=indexed_block_ids,
                            inner_extent=inner_extent,
                            subscript_block_ids=subscript_block_ids,
                            subscript_strides=subscript_strides,
                            subscript_index_scales=subscript_index_scales,
                            subscript_affine_block_ids=subscript_affine_block_ids,
                            subscript_extents=subscript_extents,
                            accessed_numel=accessed_numel,
                        ),
                    )
                )
                memory_op_index += 1

        return [
            fact._replace(matmul_operand=operands.get(node)) for node, fact in records
        ]

    def pointwise_fact(
        self,
        spec: ConfigSpec,
    ) -> PointwiseElementwiseFact | None:
        """Build the fact for a pure, tile-independent elementwise kernel."""
        from ..language import memory_ops
        from ..language.atomic_ops import ATOMIC_OPS
        from ..language.inline_asm_ops import inline_asm_elementwise
        from ..language.inline_triton_ops import inline_triton
        from ..language.inline_triton_ops import triton_kernel
        from ..language.reduce_ops import _reduce
        from ..language.scan_ops import _associative_scan

        reduction_fact = spec.reduction_kernel_fact
        has_sized_reduction = reduction_fact is not None and any(
            descriptor.category in SIZED_REDUCTION_CATEGORIES
            for descriptor in reduction_fact.reductions
        )
        if has_sized_reduction or spec.matmul_facts or spec.accumulator_facts:
            return None
        if not spec.memory_op_facts or not spec.block_sizes:
            return None

        unsafe_pointwise_ops = frozenset(
            {
                _associative_scan,
                _reduce,
                inline_triton,
                triton_kernel,
                inline_asm_elementwise,
                *ATOMIC_OPS,
            }
        )
        total_numel = 1
        for block_size in spec.block_sizes:
            total_numel *= block_size.size_hint
        total_numel = max(1, total_numel)

        def value_itemsizes(value: object) -> list[int]:
            if isinstance(value, torch.Tensor):
                return [value.dtype.itemsize]
            if isinstance(value, (list, tuple)):
                return [
                    itemsize for item in value for itemsize in value_itemsizes(item)
                ]
            return []

        data_nodes: set[int] = set()
        data_path_buffer_width = 1
        for graph in self.graphs:
            for node in reversed(graph.nodes):
                if node.op != "call_function":
                    continue
                is_load = node.target is memory_ops.load
                if is_load or node.target is memory_ops.store:
                    for position in (2, 3) if not is_load else (2,):
                        if len(node.args) > position and isinstance(
                            node.args[position],
                            torch.fx.Node,
                        ):
                            data_nodes.add(id(node.args[position]))
                    accessed = _accessed_tensor_fake(node)
                    if accessed is not None:
                        data_path_buffer_width = max(
                            data_path_buffer_width,
                            accessed.dtype.itemsize,
                        )
                    if is_load:
                        continue
                if id(node) in data_nodes:
                    data_nodes.update(id(arg) for arg in node.all_input_nodes)

        sfu_ops = frozenset(
            {
                "sin",
                "cos",
                "tan",
                "tanh",
                "asin",
                "acos",
                "atan",
                "sinh",
                "cosh",
                "atanh",
                "asinh",
                "acosh",
                "exp",
                "exp2",
                "expm1",
                "log",
                "log2",
                "log10",
                "log1p",
                "sqrt",
                "rsqrt",
                "sigmoid",
                "erf",
                "erfc",
                "pow",
                "reciprocal",
            }
        )
        compute_itemsize = 1
        sfu_op_count = 0
        for graph in self.graphs:
            for node in graph.nodes:
                if id(node) in data_nodes:
                    for itemsize in value_itemsizes(node.meta.get("val")):
                        compute_itemsize = max(compute_itemsize, itemsize)
                if node.op == "call_function":
                    if node.target in unsafe_pointwise_ops:
                        return None
                    base = getattr(node.target, "__name__", str(node.target)).split(
                        "."
                    )[0]
                    if base in sfu_ops:
                        sfu_op_count += 1
        if compute_itemsize == 1:
            compute_itemsize = max(1, data_path_buffer_width)

        tiled_ids = {block_size.block_id for block_size in spec.block_sizes}
        contiguous_block_ids: set[int] = set()
        slab_numel = 0
        max_op_slab_numel = 1
        storage_itemsize = 1
        gather_stride = 1
        for memory_fact in spec.memory_op_facts:
            if memory_fact.dtype is None or memory_fact.accessed_numel < total_numel:
                continue
            slab_numel += memory_fact.accessed_numel // total_numel
            op_slab = 1
            for block_id, extent in zip(
                memory_fact.subscript_affine_block_ids,
                memory_fact.subscript_extents,
                strict=True,
            ):
                if block_id is None or block_id not in tiled_ids:
                    op_slab *= max(1, extent)
            max_op_slab_numel = max(max_op_slab_numel, op_slab)
            storage_itemsize = max(storage_itemsize, memory_fact.dtype.itemsize)
            for block_id, stride in zip(
                memory_fact.subscript_block_ids,
                memory_fact.subscript_strides,
                strict=True,
            ):
                if block_id is not None and stride == 1 and block_id in tiled_ids:
                    contiguous_block_ids.add(block_id)
            for block_id, scale, stride in zip(
                memory_fact.subscript_affine_block_ids,
                memory_fact.subscript_index_scales,
                memory_fact.subscript_strides,
                strict=True,
            ):
                if block_id is not None and block_id in tiled_ids and stride == 1:
                    gather_stride = max(gather_stride, scale)

        return PointwiseElementwiseFact(
            total_numel=total_numel,
            slab_numel=slab_numel,
            storage_itemsize=storage_itemsize,
            compute_itemsize=compute_itemsize,
            contig_block_ids=tuple(sorted(contiguous_block_ids)),
            sfu_ops=sfu_op_count,
            gather_stride=gather_stride,
            max_op_slab_numel=max_op_slab_numel,
        )

    def kernel_matmul_fact(
        self,
        env: CompileEnvironment,
        spec: ConfigSpec,
    ) -> KernelMatmulFact | None:
        """Compose the whole-kernel contraction fact from analyzed FX graphs."""
        facts = spec.matmul_facts
        if not facts:
            return None

        def axis_extent_or_none(block_id: int | None) -> int | None:
            if block_id is not None and 0 <= block_id < len(env.block_sizes):
                size = env.block_sizes[block_id].size
                if isinstance(size, (int, torch.SymInt)):
                    return max(1, env.size_hint(size))
            return None

        from ..language.matmul_ops import MATMUL_DIM_BLOCK_IDS_META
        from ..language.matmul_ops import MATMUL_FACT_ID_META

        nodes_by_fact_id: dict[int, tuple[int, torch.fx.Node]] = {}
        for graph_id, node in self.dot_nodes:
            fact_id = node.meta.get(MATMUL_FACT_ID_META)
            if (
                isinstance(fact_id, int)
                and 0 <= fact_id < len(facts)
                and fact_id not in nodes_by_fact_id
            ):
                nodes_by_fact_id[fact_id] = (graph_id, node)
        attribution_complete = (
            len(self.dot_nodes) == len(facts) == len(nodes_by_fact_id)
        )

        if attribution_complete:
            operand_positions = matmul_operand_positions()

            def shape_block_ids(node: torch.fx.Node) -> tuple[int | None, ...]:
                value = node.meta.get("val")
                if not isinstance(value, torch.Tensor):
                    return ()
                # Do not use ``resolve_block_id``: matching a concrete padded extent
                # could assign an unrelated fixed dimension to a reduction axis.
                result = [env.get_block_id(size) for size in value.shape]
                annotated = node.meta.get(MATMUL_DIM_BLOCK_IDS_META)
                if isinstance(annotated, (list, tuple)) and len(annotated) == len(
                    result
                ):
                    for index, block_id in enumerate(annotated):
                        if isinstance(block_id, int):
                            result[index] = block_id
                return tuple(result)

            def choose_axis(
                original: int | None,
                *inferred: int | None,
            ) -> int | None:
                if original is not None:
                    return original
                candidates = {value for value in inferred if value is not None}
                return candidates.pop() if len(candidates) == 1 else None

            # Keep block identity separate from representative fake sizes. A dot
            # establishes the identity of its output M/N dimensions; shape-preserving
            # pointwise operations carry it until a later dot consumes that tensor.
            for graph in self.graphs:
                for node in graph.nodes:
                    value = node.meta.get("val")
                    if not isinstance(value, torch.Tensor):
                        continue
                    result_ids = list(shape_block_ids(node))
                    fact_id = node.meta.get(MATMUL_FACT_ID_META)
                    if isinstance(fact_id, int) and fact_id in nodes_by_fact_id:
                        positions = operand_positions.get(node.target)
                        if positions is None:
                            continue
                        lhs = node.args[positions[0]]
                        rhs = node.args[positions[1]]
                        assert isinstance(lhs, torch.fx.Node)
                        assert isinstance(rhs, torch.fx.Node)
                        lhs_ids = shape_block_ids(lhs)
                        rhs_ids = shape_block_ids(rhs)
                        fact = facts[fact_id]
                        m_block_id = choose_axis(
                            fact.m_block_id,
                            lhs_ids[-2] if len(lhs_ids) >= 2 else None,
                        )
                        n_block_id = choose_axis(
                            fact.n_block_id,
                            rhs_ids[-1] if rhs_ids else None,
                        )
                        k_block_id = choose_axis(
                            fact.k_block_id,
                            lhs_ids[-1] if lhs_ids else None,
                            rhs_ids[-2] if len(rhs_ids) >= 2 else None,
                        )
                        fact = fact._replace(
                            m_block_id=m_block_id,
                            n_block_id=n_block_id,
                            k_block_id=k_block_id,
                            static_m=axis_extent_or_none(m_block_id) or fact.static_m,
                            static_n=axis_extent_or_none(n_block_id) or fact.static_n,
                            static_k=axis_extent_or_none(k_block_id) or fact.static_k,
                        )
                        facts[fact_id] = fact
                        if len(result_ids) >= 2:
                            result_ids[-2:] = [m_block_id, n_block_id]
                    elif torch.Tag.pointwise in getattr(node.target, "tags", ()):
                        shape = tuple(map(str, value.shape))
                        matching_inputs = [
                            shape_block_ids(input_node)
                            for input_node in node.all_input_nodes
                            if isinstance(
                                input_value := input_node.meta.get("val"), torch.Tensor
                            )
                            and tuple(map(str, input_value.shape)) == shape
                        ]
                        for index in range(len(result_ids)):
                            if result_ids[index] is not None:
                                continue
                            candidates = {
                                input_ids[index]
                                for input_ids in matching_inputs
                                if len(input_ids) == len(result_ids)
                                and input_ids[index] is not None
                            }
                            if len(candidates) == 1:
                                result_ids[index] = candidates.pop()
                    if any(block_id is not None for block_id in result_ids):
                        node.meta[MATMUL_DIM_BLOCK_IDS_META] = tuple(result_ids)

        valid_block_ids = set(spec.block_sizes.valid_block_ids())

        def classify_axis(
            block_id: int | None,
            static_extent: int | None,
        ) -> tuple[DotAxisKind, int | None]:
            if block_id is not None and block_id in valid_block_ids:
                return DotAxisKind.TUNABLE_TILED, static_extent
            if block_id is not None:
                extent = _immovable_extent(env, spec, block_id)
                if extent is not None:
                    return DotAxisKind.FIXED_FULL_EXTENT, extent
            if static_extent is None:
                return DotAxisKind.UNKNOWN, None
            return DotAxisKind.FIXED_FULL_EXTENT, static_extent

        axes: list[DotAxes] = []
        for fact in facts:
            m_kind, m_extent = classify_axis(fact.m_block_id, fact.static_m)
            n_kind, n_extent = classify_axis(fact.n_block_id, fact.static_n)
            k_kind, k_extent = classify_axis(fact.k_block_id, fact.static_k)
            axes.append(
                DotAxes(
                    m_kind,
                    n_kind,
                    k_kind,
                    m_extent,
                    n_extent,
                    k_extent,
                )
            )

        knob_users: dict[int, list[tuple[int, str]]] = {}
        for index, fact in enumerate(facts):
            for axis, block_id in (
                ("m", fact.m_block_id),
                ("n", fact.n_block_id),
                ("k", fact.k_block_id),
            ):
                if block_id is not None and block_id in valid_block_ids:
                    knob_users.setdefault(block_id, []).append((index, axis))

        from .host_function import HostFunction
        from .variable_origin import BlockSizeOrigin
        from .variable_origin import TileIdOrigin

        origins = HostFunction.current().expr_to_origin

        def symbolic_loop_bound(block_id: int) -> SymbolicLoopBound | None:
            """Preserve candidate-dependent symbolic loop bounds."""
            size = env.block_sizes[block_id].size
            if not isinstance(size, torch.SymInt):
                return None
            expr = _symint_sympy_expr(size)

            block_size_symbols: list[tuple[sympy.Symbol, int]] = []
            tile_id_symbols: list[tuple[sympy.Symbol, int]] = []
            for symbol in sorted(
                _symint_free_symbols(size), key=lambda item: item.name
            ):
                origin_info = origins.get(symbol)
                if origin_info is None:
                    continue
                origin = origin_info.origin
                if isinstance(origin, BlockSizeOrigin):
                    block_size_symbols.append((symbol, origin.block_id))
                elif isinstance(origin, TileIdOrigin):
                    tile_id_symbols.append((symbol, origin.block_id))
            if not block_size_symbols and not tile_id_symbols:
                return None
            return SymbolicLoopBound(
                expr,
                tuple(block_size_symbols),
                tuple(tile_id_symbols),
            )

        symbolic_loop_bounds = {
            block_id: symbolic_loop_bound(block_id)
            for block_id in range(len(env.block_sizes))
        }

        grid_block_ids = set(spec.grid_block_ids)
        matmul_output_block_ids = {fact.m_block_id for fact in facts} | {
            fact.n_block_id for fact in facts
        }
        outer_grid = 1
        for block_id in spec.grid_block_ids:
            if block_id not in matmul_output_block_ids:
                outer_grid *= axis_extent_or_none(block_id) or 1

        bounded_by: dict[int, int] = {}
        bounded_extent: dict[int, int] = {}
        for slot in range(len(spec.block_sizes)):
            block_spec = spec.block_sizes[slot]
            parent = block_spec.bounded_by_block_id
            if parent is not None:
                bounded_by[block_spec.block_id] = parent
                if 0 <= parent < len(env.block_sizes):
                    source = env.block_sizes[parent].block_size_source
                    if isinstance(source, FixedBlockSizeSource) and isinstance(
                        source.value,
                        int,
                    ):
                        bounded_extent[block_spec.block_id] = max(1, source.value)

        def loop_axes_for(graph_id: int) -> tuple[LoopAxisFact, ...]:
            axes_out: list[LoopAxisFact] = []
            seen_axes: set[int] = set()
            current = graph_id
            while current in self.loop_block_ids:
                for block_id in sorted(self.loop_block_ids[current]):
                    if block_id in grid_block_ids or block_id in seen_axes:
                        continue
                    seen_axes.add(block_id)
                    bound = (
                        None
                        if block_id in bounded_by
                        else symbolic_loop_bounds.get(block_id)
                    )
                    axes_out.append(
                        LoopAxisFact(
                            block_id=block_id,
                            extent=(
                                None
                                if bound is not None
                                else axis_extent_or_none(block_id)
                            ),
                            bounded_by_block_id=bounded_by.get(block_id),
                            bounded_extent=bounded_extent.get(block_id),
                            symbolic_bound=bound,
                        )
                    )
                current = self.parent_of.get(current, -1)
            return tuple(axes_out)

        def max_trips_for(graph_id: int) -> int | None:
            trips = 1
            for axis in loop_axes_for(graph_id):
                extent = axis.extent
                if axis.symbolic_bound is not None:
                    # There is no candidate-independent upper bound for arbitrary
                    # symbolic arithmetic.
                    return None
                if axis.bounded_by_block_id is not None:
                    extent = (
                        axis.bounded_extent
                        or _immovable_extent(
                            env,
                            spec,
                            axis.bounded_by_block_id,
                        )
                        or axis_extent_or_none(axis.bounded_by_block_id)
                    )
                if extent is None:
                    return None
                block = _immovable_extent(env, spec, axis.block_id)
                trips *= max(1, -(-extent // block)) if block else extent
            return trips

        def strip_index_cast(value: object) -> object:
            for _ in range(4):
                if not isinstance(value, torch.fx.Node) or value.target not in (
                    torch.ops.prims.convert_element_type.default,
                    torch.ops.aten._to_copy.default,
                ):
                    break
                value = value.args[0]
            return value

        def is_single_segment_loop(body_graph_id: int, block_id: int) -> bool:
            calls = self.loop_calls.get(body_graph_id, ())
            if len(calls) != 1:
                return False
            block_id_order = self.by_id[body_graph_id].block_id_order
            if block_id not in block_id_order:
                return False
            position = block_id_order.index(block_id)
            call = calls[0]
            ends = call.args[2] if len(call.args) > 2 else None
            if not isinstance(ends, (list, tuple)) or position >= len(ends):
                return False

            difference = strip_index_cast(ends[position])
            if (
                not isinstance(difference, torch.fx.Node)
                or difference.target is not torch.ops.aten.sub.Tensor
            ):
                return False
            end = strip_index_cast(difference.args[0])
            begin = strip_index_cast(difference.args[1])

            from ..language import memory_ops

            if (
                not isinstance(end, torch.fx.Node)
                or end.target is not memory_ops.load
                or not isinstance(begin, torch.fx.Node)
                or begin.target is not memory_ops.load
                or end.args[0] is not begin.args[0]
            ):
                return False
            offsets = _accessed_tensor_fake(end)
            if (
                offsets is None
                or offsets.ndim != 1
                or env.size_hint(offsets.shape[0]) != 2
            ):
                return False
            end_indices = end.args[1] if len(end.args) > 1 else None
            begin_indices = begin.args[1] if len(begin.args) > 1 else None
            if (
                not isinstance(end_indices, (list, tuple))
                or len(end_indices) != 1
                or not isinstance(begin_indices, (list, tuple))
                or len(begin_indices) != 1
            ):
                return False
            end_index = strip_index_cast(end_indices[0])
            begin_index = strip_index_cast(begin_indices[0])
            if not isinstance(end_index, torch.fx.Node):
                return False
            if end_index.target not in (
                operator.add,
                torch.ops.aten.add.Tensor,
                torch.ops.aten.add.Scalar,
            ):
                return False
            lhs, rhs = end_index.args[:2]
            return (lhs is begin_index and isinstance(rhs, int) and rhs == 1) or (
                rhs is begin_index and isinstance(lhs, int) and lhs == 1
            )

        def direct_operand_trip_count(
            node: torch.fx.Node,
            operand_index: int,
        ) -> tuple[int, int] | None:
            from ..language import memory_ops

            if operand_index >= len(node.args):
                return None
            operand = node.args[operand_index]
            load = trace_back_to_load(operand, memory_ops.load)
            if load is None or not isinstance(operand, torch.fx.Node):
                return None
            source = _accessed_tensor_fake(load)
            operand_value = operand.meta.get("val")
            if source is None or not isinstance(operand_value, torch.Tensor):
                return None

            source_numel = 1
            for size, stride in zip(source.shape, source.stride(), strict=True):
                if stride != 0:
                    source_numel *= max(1, env.size_hint(size))
            operand_numel = 1
            for size in operand_value.shape:
                operand_numel *= max(1, env.size_hint(size))
            denominator = max(1, outer_grid) * operand_numel
            if source_numel % denominator:
                return None
            return id(load), max(1, source_numel // denominator)

        dot_targets = matmul_operand_positions()
        dots_by_graph: dict[int, list[torch.fx.Node]] = {}
        for graph_id, node in self.dot_nodes:
            dots_by_graph.setdefault(graph_id, []).append(node)

        inferred_work_trips: dict[tuple[int, int], int] = {}
        for body_graph_id, block_ids in self.loop_block_ids.items():
            unresolved = [
                block_id
                for block_id in block_ids
                if block_id not in grid_block_ids
                and axis_extent_or_none(block_id) is None
                and block_id not in bounded_by
                and symbolic_loop_bounds.get(block_id) is None
            ]
            enclosing_axes = loop_axes_for(body_graph_id)
            if len(block_ids) != 1 or len(unresolved) != 1 or len(enclosing_axes) != 1:
                continue
            block_id = unresolved[0]
            if not is_single_segment_loop(body_graph_id, block_id):
                continue
            block = _immovable_extent(env, spec, block_id)
            if block is None:
                continue
            trip_by_load: dict[int, int] = {}
            for node in dots_by_graph.get(body_graph_id, ()):
                positions = dot_targets.get(node.target)
                if positions is None:
                    continue
                for operand_index in positions:
                    proof = direct_operand_trip_count(node, operand_index)
                    if proof is not None:
                        trip_by_load[proof[0]] = proof[1]
            trips = set(trip_by_load.values())
            if len(trip_by_load) >= 2 and len(trips) == 1:
                inferred_work_trips[(body_graph_id, block_id)] = trips.pop()

        sites: list[DotSite] = []
        if attribution_complete:
            for index, fact in enumerate(facts):
                graph_id, node = nodes_by_fact_id[index]
                value = node.meta.get("val")
                if isinstance(value, torch.Tensor) and value.ndim >= 2:
                    observed_extents: list[int] = []
                    annotated = node.meta.get(MATMUL_DIM_BLOCK_IDS_META)
                    annotated_tail = (
                        annotated[-2:]
                        if isinstance(annotated, (list, tuple))
                        and len(annotated) == value.ndim
                        else (None, None)
                    )
                    for dimension, annotated_block_id in zip(
                        value.shape[-2:],
                        annotated_tail,
                        strict=True,
                    ):
                        block_id = (
                            annotated_block_id
                            if isinstance(annotated_block_id, int)
                            else env.resolve_block_id(dimension)
                        )
                        if block_id is not None and 0 <= block_id < len(
                            env.block_sizes
                        ):
                            dimension = env.block_sizes[block_id].size
                        if not isinstance(dimension, (int, torch.SymInt)):
                            observed_extents.append(-1)
                            continue
                        observed_extents.append(env.size_hint(dimension))
                    expected_extents = [
                        fact.static_m,
                        fact.static_n,
                    ]
                    if any(
                        expected is not None and observed != -1 and expected != observed
                        for expected, observed in zip(
                            expected_extents,
                            observed_extents,
                            strict=False,
                        )
                    ):
                        attribution_complete = False
                updates_carry = (
                    self.by_id[graph_id].reaches_output(node)
                    and graph_id in self.loop_block_ids
                )
                rank_reduction_scaled_accumulator_batch_block_id = (
                    _rank_reduction_scaled_baddbmm_batch_block_id(
                        node,
                        env,
                    )
                    if updates_carry
                    else None
                )
                loop_axes = loop_axes_for(graph_id)
                exact_loop_trips = (
                    inferred_work_trips.get((graph_id, loop_axes[0].block_id))
                    if len(loop_axes) == 1
                    else None
                )
                sites.append(
                    DotSite(
                        graph_id,
                        updates_carry,
                        loop_axes,
                        exact_loop_trips,
                        max_trips_for(graph_id),
                        rank_reduction_scaled_accumulator_batch_block_id,
                    )
                )
        if not attribution_complete:
            sites = [DotSite(-1, False, (), None, None) for _ in facts]

        sequential_loop_trips = 1
        for body_graph_id, block_ids in self.loop_block_ids.items():
            if any(block_id in spec.grid_block_ids for block_id in block_ids):
                continue
            max_trips = max_trips_for(body_graph_id)
            if max_trips is not None:
                sequential_loop_trips = max(sequential_loop_trips, max_trips)

        pipelined_regions: list[PipelinedRegion] = []
        resident_regions: list[ResidentRegion] = []
        for graph in self.non_reduction_graphs:
            loop_axes = (
                loop_axes_for(graph.graph_id)
                if graph.graph_id in self.loop_block_ids
                else ()
            )
            tiles = graph.memory_tiles_for_loop_axes(
                env,
                frozenset(axis.block_id for axis in loop_axes),
            )
            if not tiles:
                continue
            if graph.graph_id in self.loop_block_ids:
                pipelined_regions.append(PipelinedRegion(loop_axes, tuple(tiles)))
            else:
                resident_regions.append(ResidentRegion(tuple(tiles)))

        resolved = tuple(
            starmap(ResolvedMatmulFact, zip(facts, axes, sites, strict=True))
        )
        return KernelMatmulFact(
            matmuls=resolved,
            knob_users=tuple(
                (block_id, tuple(knob_users[block_id]))
                for block_id in sorted(knob_users)
            ),
            sequential_loop_trips=sequential_loop_trips,
            live_dot_outputs=tuple(self.kernel_peak_dot_outputs()),
            live_promoted_lhs=tuple(self.kernel_peak_promoted_lhs()),
            live_tile_steps=tuple(self.kernel_live_tile_steps()),
            pipelined_regions=tuple(pipelined_regions),
            resident_regions=tuple(resident_regions),
            attribution_complete=attribution_complete,
        )
