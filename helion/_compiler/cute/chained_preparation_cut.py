"""Dependency cuts for preparation ahead of an ordered contraction loop.

This is a semantic analysis, not a pipeline schedule. It retains the original
FX revision and typed values; allocation, asynchronous completion, role-local
indices and slot lifetimes must be proved by a later physical planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import creation_ops
from ...language import memory_ops
from ...language import scan_ops
from ...language import tile_ops
from ...language.matmul_ops import dot
from ...language.tile_ops import tile_index
from .contraction_region import _domain

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo
    from .contraction_region import ContractionCarry
    from .contraction_region import ContractionRegion
    from .contraction_region import LogicalDomain


@dataclass(frozen=True)
class PreparationImage:
    """One exact value crossing the cut, before any physical layout selection.

    Keeping the producer node retains its original casts, indexing, predicates
    and default-zero semantics. Consumers are direct uses, including store
    indices/masks and explicit accumulators, not just subsequent dot operands.
    """

    node: Node
    dtype: torch.dtype
    logical_domain: LogicalDomain
    consumers: tuple[Node, ...]


@dataclass(frozen=True)
class PreparationCut:
    region: ContractionRegion
    carries: tuple[ContractionCarry, ...]
    captures: tuple[Node, ...]
    shared_inputs: tuple[Node, ...]
    preparation: tuple[Node, ...]
    recurrence: tuple[Node, ...]
    images: tuple[PreparationImage, ...]
    storage_proof_key: str


def is_load_only_preparation(cut: PreparationCut) -> bool:
    """A nonempty independent frontier without MMA or collective producers.

    This is a family predicate, not storage or expression proof. The complete
    cut, typed images and consumers are still validated by the frame planner.
    """
    return (
        bool(cut.images)
        and not cut.region.scans
        and not cut.region.reductions
        and all(spec.node in cut.recurrence for spec in cut.region.contractions)
        and bool(cut.region.contractions)
    )


def _known_effects(node: Node) -> bool:
    """Do not infer purity from a pointwise-lowering annotation alone."""
    from . import chained_matmul as chain
    from .chained_collectives import classify_collective

    if node.op in ("placeholder", "output"):
        return True
    if node.op != "call_function":
        return False
    if isinstance(node.target, torch._ops.OpOverload):
        return (
            not node.target._schema.is_mutable
            and not {
                torch.Tag.nondeterministic_seeded,
                torch.Tag.nondeterministic_bitwise,
            }.intersection(node.target.tags)
            and (
                torch.Tag.pointwise in node.target.tags
                or node.target in chain._VIEWS
                or node.target
                in (
                    torch.ops.prims.iota.default,
                    torch.ops.aten.scalar_tensor.default,
                    torch.ops.aten.sym_size.int,
                )
                or classify_collective(node) is not None
            )
        )
    return node.target in {
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        _tracing_ops._mask_to,
        tile_ops.tile_begin,
        tile_ops.tile_id,
        tile_index,
        creation_ops.full,
        memory_ops.load,
        memory_ops.store,
        scan_ops._associative_scan,
        dot,
        *chain._SCALAR_BINARY,
        *chain._VIEWS,
    }


def _shape_input(node: Node) -> bool:
    """A tensor's fixed logical shape does not depend on its carried values."""
    if (
        node.target is not torch.ops.aten.sym_size.int
        or len(node.args) != 2
        or node.kwargs
    ):
        return False
    source, axis = node.args
    if not isinstance(source, Node) or type(axis) is not int:
        return False
    tensor, value = source.meta.get("val"), node.meta.get("val")
    if not isinstance(tensor, torch.Tensor) or not -tensor.ndim <= axis < tensor.ndim:
        return False
    if isinstance(value, torch.SymInt):
        value = value.node.expr
    elif type(value) is not int:
        return False
    return value == _domain(tensor)[axis]


def late_widening_candidates(cut: PreparationCut) -> tuple[Node, ...]:
    """Pure raw-half widenings whose complete use lies beyond the cut.

    Moving the original conversion changes transport, not its typed value or
    evaluation order within any consumer. Keep the load's exact indices/mask
    and reject extra uses rather than inventing a different preparation image.
    """
    candidates = []
    for image in cut.images:
        node = image.node
        if (
            node not in cut.preparation
            or node.target is not torch.ops.prims.convert_element_type.default
            or len(node.args) != 2
            or node.args[1] is not torch.float32
            or node.kwargs
        ):
            continue
        source = node.args[0]
        if (
            not isinstance(source, Node)
            or source not in cut.preparation
            or source.target is not memory_ops.load
            or len(source.args) != 4
            or source.kwargs
            or tuple(source.users) != (node,)
        ):
            continue
        host, indices, mask, _eviction = source.args
        raw, widened = source.meta.get("val"), node.meta.get("val")
        if (
            not isinstance(host, Node)
            or not isinstance(host.meta.get("val"), torch.Tensor)
            or not isinstance(indices, (tuple, list))
            or not isinstance(raw, torch.Tensor)
            or raw.dtype not in (torch.float16, torch.bfloat16)
            or host.meta["val"].dtype != raw.dtype
            or not isinstance(widened, torch.Tensor)
            or widened.dtype != torch.float32
            or image.dtype != torch.float32
            or _domain(raw) != _domain(widened)
            or image.logical_domain != _domain(raw)
            or not node.users
            or set(node.users) != set(image.consumers)
            or any(user not in cut.recurrence for user in node.users)
            or (
                mask is not None
                and (
                    not isinstance(mask, Node)
                    or not isinstance(mask.meta.get("val"), torch.Tensor)
                    or mask.meta["val"].dtype != torch.bool
                )
            )
        ):
            continue
        candidates.append(node)
    return tuple(candidates)


def _images(
    region: ContractionRegion, preparation: set[Node], recurrent: set[Node]
) -> tuple[PreparationImage, ...] | None:
    images = []
    positions = {node: index for index, node in enumerate(region.nodes)}
    for node in region.nodes:
        if node not in preparation:
            continue
        consumers = tuple(
            sorted(
                (user for user in node.users if user in recurrent),
                key=positions.__getitem__,
            )
        )
        if not consumers:
            continue
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            return None
        images.append(PreparationImage(node, value.dtype, _domain(value), consumers))
    return tuple(images)


def plan_preparation_cut(
    graphs: Sequence[GraphInfo], *, late_widening: bool = False
) -> PreparationCut | None:
    """Partition an admitted loop's full DAG under its active host context.

    Admission reuses common expression/collective and loop-port proofs, then
    replays the existing disjoint-storage guard before allowing read-ahead.
    The caller supplies live inputs through ``env.use_runtime_arg_values``;
    an absent or incompatible runtime binding is not a storage proof.
    Returned records belong to this graph revision: recollect after rewrites.
    Host handles, captures and scalar index inputs must be rebound identically
    by both roles (including each role's selected loop iteration). They are not
    tensor images in a stage ring. Every write remains in the ordered role.
    """
    from .chained_loop import loop_storage_matches_runtime
    from .chained_matmul import _classify_chained_graph

    graph = _classify_chained_graph(graphs)
    if graph is None or graph.loop is None:
        return None
    loop = graph.loop
    region = loop.region
    if not all(map(_known_effects, region.nodes)) or not loop_storage_matches_runtime(
        loop
    ):
        return None
    carried = {carry.input for carry in region.carries}
    captures = tuple(loop.captures())
    shared = {
        node
        for node in region.nodes
        if node in captures
        or node.target in (_tracing_ops._host_tensor, _tracing_ops._get_symnode)
    }
    recurrent = set(carried)
    preparation: set[Node] = set()
    for node in region.nodes:
        inputs = node.all_input_nodes
        if node.target is torch.ops.aten.sym_size.int:
            if not _shape_input(node):
                return None
            shared.add(node)
            continue
        if (
            node in recurrent
            or any(source in recurrent for source in inputs)
            or node.target is memory_ops.store
            or node.op == "output"
        ):
            recurrent.add(node)
            shared.discard(node)
        elif node in shared:
            continue
        elif (
            type(node.meta.get("val")) in (bool, int, float)
            or isinstance(
                node.meta.get("val"), (torch.SymInt, torch.SymFloat, torch.SymBool)
            )
        ) and all(source in shared for source in inputs):
            # Shape/index scalars stay exact expressions over the shared ports;
            # do not guess a storage dtype or widen fixed-width arithmetic.
            shared.add(node)
        else:
            preparation.add(node)
    images = _images(region, preparation, recurrent)
    if images is None:
        return None
    cut = PreparationCut(
        region,
        region.carries,
        captures,
        tuple(node for node in region.nodes if node in shared),
        tuple(node for node in region.nodes if node in preparation),
        tuple(node for node in region.nodes if node in recurrent),
        images,
        loop.storage_key,
    )
    if not late_widening:
        return cut
    moved = late_widening_candidates(cut)
    if not moved:
        return None
    preparation.difference_update(moved)
    recurrent.update(moved)
    images = _images(region, preparation, recurrent)
    if images is None:
        return None
    return PreparationCut(
        region,
        cut.carries,
        captures,
        cut.shared_inputs,
        tuple(node for node in region.nodes if node in preparation),
        tuple(node for node in region.nodes if node in recurrent),
        images,
        cut.storage_proof_key,
    )
