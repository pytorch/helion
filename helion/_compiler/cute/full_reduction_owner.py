"""Full uint8 extrema that require a complete coordinate-owning root.

Scalar lane lowering cannot in general represent a reduction of an already
computed multidimensional value. Preserve those coordinates for the existing
computed-fragment owner rather than flattening a producer shared with another
consumer, or dropping a physical-thread ownership guard.

Logical decompositions are the motivating case, but direct full uint8 extrema
use the same exact owner. Only every local tensor axis is admitted here: a
retained tile axis, even of length one, is a partial reduction and needs a
separate surviving-coordinate ownership proof. Explicit grid and inner-loop
tile axes are also excluded, including their reshape expressions: their tail
domains need a separate complete-domain proof.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import torch

from ... import exc
from ..compile_environment import CompileEnvironment
from ..runtime_tensor_guards import tensor_pairs_with_proven_disjoint_storage
from . import target_facts

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo


_REQUIRED_OWNER = "helion_cute_full_reduction_owner"


def defer_full_logical_reduction(
    result: torch.fx.Node, source: torch.fx.Node, dims: tuple[int, ...]
) -> bool:
    """Mark only a full uint8 truth/extrema chain; later checks are mandatory.

    This hook is reached only after the normal static/adjacent-dimension merge
    admission. The fragment owner retains its own shape, layout, alias, control
    flow, supported-operation and shared-capacity checks at code generation.
    """
    # The emitter registry imports this module while Helion Config is still
    # initializing; DeviceIR imports that Config, so resolve it at bind time.
    from ..device_ir import DeviceIR
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import NodeArgsGraphInfo
    from ..device_ir import WhileConditionGraphInfo
    from ..device_ir import WhileLoopGraphInfo
    from ..device_ir import control_flow_child_graph_ids

    value = source.meta.get("val")
    if not (
        isinstance(value, torch.Tensor)
        and value.ndim > 1
        and value.dtype == torch.uint8
        and dims == tuple(range(value.ndim))
        and result.target in (torch.ops.aten.amax.default, torch.ops.aten.amin.default)
    ):
        return False
    ir = DeviceIR.current()
    if len(ir.root_ids) != 1 or len(ir.phases) != 1:
        from .dependent_phases import prepare_required_owner_phases

        if not prepare_required_owner_phases(ir):
            return False
    env = CompileEnvironment.current()
    tile_ids = {block_id for ids in ir.grid_block_ids for block_id in ids}
    tile_ids.update(
        block_id
        for info in ir.graphs
        if isinstance(info, ForLoopGraphInfo)
        for block_id in info.block_ids
    )
    # A literal reshape may erase every tile symbol from its result's shape.
    # Follow the producer domain, including direct captures and control-flow
    # results, so such views cannot turn a partial tile into a complete tensor.
    # This is conservative: even a contraction that consumes the tile axis is
    # left to a future domain proof. Scalar row.id indices do not add a tensor
    # axis, so ordinary per-row full tensors remain eligible.
    graph_infos = {info.graph: info for info in ir.graphs}
    while_bodies = {
        info.cond_graph_id: info
        for info in ir.graphs
        if isinstance(info, WhileLoopGraphInfo)
    }
    pending = [source]
    seen = set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor) and any(
            env.get_block_id(symbol) in tile_ids
            for size in value.shape
            if isinstance(size, torch.SymInt)
            for symbol in size._sympy_().free_symbols
        ):
            return False
        pending.extend(node.all_input_nodes)
        info = graph_infos[node.graph]
        if node.op == "placeholder" and isinstance(info, NodeArgsGraphInfo):
            # The direct argument is the zero-iteration/initial value. A loop
            # body may replace it on every iteration, so include the body's
            # result too. Traversing all body nodes is conservative and the
            # visited set closes the resulting carry cycle.
            pending.append(info.placeholder_to_outer_arg(node))
            if isinstance(info, (ForLoopGraphInfo, WhileLoopGraphInfo)):
                pending.extend(info.graph.nodes)
            elif isinstance(info, WhileConditionGraphInfo):
                # The condition reads the same carried state as the body.
                pending.extend(while_bodies[info.graph_id].graph.nodes)
        for child_id in control_flow_child_graph_ids(node):
            pending.extend(ir.graphs[child_id].graph.nodes)
    result.meta[_REQUIRED_OWNER] = True
    return True


def requires_full_reduction_owner(graphs: Sequence[GraphInfo]) -> bool:
    return any(
        node.meta.get(_REQUIRED_OWNER) is True
        for info in graphs
        for node in info.graph.nodes
    )


def validate_required_full_reduction_owner(
    env: CompileEnvironment, graphs: Sequence[GraphInfo], *, supported: bool
) -> None:
    """Refuse input/target limitations once during bind, before config search.

    Discovery may use live disjointness only through the registered runtime
    classifier. The emitter still checks the immutable bound facts and every
    config-dependent layout/capacity condition.
    """
    if not requires_full_reduction_owner(graphs):
        return
    if not supported:
        raise exc.BackendUnsupported(
            "cute",
            "unmerged full reductions require a supported complete fragment "
            "root; use the Triton backend for chained reductions",
        )
    if not target_facts.per_cta_smem_capacity_bytes(env.device):
        raise exc.BackendUnsupported(
            "cute", "unmerged full reductions require a known shared memory capacity"
        )
    destinations = fragment_store_destinations(graphs)
    disjoint = tensor_pairs_with_proven_disjoint_storage(
        env, {tensor: tensor for tensor in destinations}, allow_unbound=True
    )
    if any(
        frozenset((left, right)) not in disjoint
        for left, right in itertools.combinations(destinations, 2)
    ):
        raise exc.BackendUnsupported(
            "cute", "fragment store destinations require disjoint storage"
        )


def fragment_store_destinations(graphs: Sequence[GraphInfo]) -> set[torch.Tensor]:
    """Named global destinations, shared by bind and emitted-store proofs."""
    from ...language import memory_ops
    from ..host_function import HostFunction

    host = HostFunction.current()
    return {
        target.meta["val"]
        for info in graphs
        for node in info.graph.nodes
        if node.target is memory_ops.store
        and isinstance(target := node.args[0], torch.fx.Node)
        and isinstance(target.meta.get("val"), torch.Tensor)
        and target.meta["val"] in host.tensor_to_origin
    }
