"""Close the effects of an otherwise proved conditional attention region.

This is a semantic program certificate, not scheduling admission. A dedicated
emitter must retain the scalar prelude and output addresses, prove its physical
owners and resources, and recheck these facts on its final projected graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import operator
from typing import TYPE_CHECKING

import torch

from ....language import _tracing_ops
from ....language import creation_ops
from ....language import memory_ops
from ....language import tile_ops
from ...runtime_tensor_guards import tensor_pairs_with_proven_disjoint_storage
from ..producer_cache import pure_producer
from ..register_loads import host_load_is_readonly
from .conditional_attention_access import ConditionalAttentionAccess
from .conditional_attention_access import prove_conditional_attention_access
from .conditional_attention_recurrence import ConditionalAttentionRecurrence
from .conditional_attention_recurrence import match_conditional_attention_recurrence

if TYPE_CHECKING:
    from torch.fx import Node

    from ...compile_environment import CompileEnvironment
    from ...device_ir import DeviceIR


@dataclass(frozen=True)
class ConditionalAttentionProgram:
    recurrence: ConditionalAttentionRecurrence
    accesses: ConditionalAttentionAccess
    # Every read is explicit, including scalar metadata reads outside the page
    # loop. Pure prelude nodes remain in their original control-flow regions.
    reads: tuple[Node, ...]
    scalar_reads: tuple[Node, ...]
    writes: tuple[Node, ...]
    imported_reads: tuple[Node, ...] = ()


_SCALAR_OPERATORS = frozenset(
    (
        operator.add,
        operator.sub,
        operator.mul,
        operator.floordiv,
        operator.mod,
        operator.eq,
        operator.ne,
        operator.gt,
        operator.ge,
        operator.lt,
        operator.le,
    )
)


def prove_conditional_attention_program(
    ir: DeviceIR, env: CompileEnvironment
) -> ConditionalAttentionProgram | None:
    """Require exact opaque arithmetic and dispatch-proven readonly inputs.

    Schema-pure pointwise/view operations may remain in the prelude. An opaque
    ASM is accepted only as a node of the exact arithmetic/native leaf proofs;
    its user-supplied purity flag alone is never an admission fact here.
    """
    recurrence = match_conditional_attention_recurrence(ir, env)
    if recurrence is None:
        return None
    accesses = prove_conditional_attention_access(recurrence)
    if accesses is None:
        return None
    nodes = tuple(node for info in ir.graphs for node in info.graph.nodes)
    reads = tuple(node for node in nodes if node.target is memory_ops.load)
    writes = tuple(node for node in nodes if node.target is memory_ops.store)
    if writes != recurrence.stores:
        return None
    panel_reads = {
        accesses.query.node,
        accesses.page_table.node,
        accesses.key.raw.node,
        accesses.key.scale.node,
        accesses.value.raw.node,
        accesses.value.scale.node,
    }
    if not panel_reads <= set(reads):
        return None
    imported_reads = (
        recurrence.continuation.imported_reads if recurrence.continuation else ()
    )
    if not set(imported_reads) <= set(reads):
        return None
    scalar_reads = tuple(
        node for node in reads if node not in panel_reads and node not in imported_reads
    )
    page_regions = {recurrence.loop.graph}
    while True:
        added = {
            region.graph.graph
            for region in recurrence.regions.regions
            if region.parent is not None and region.parent.graph in page_regions
        } - page_regions
        if not added:
            break
        page_regions.update(added)
    # Only the exact seeded-entry envelope may introduce vector state loads.
    # They remain generic reads before the loop and need the same dispatch
    # readonly proof as every other input below.
    if any(node.graph in page_regions for node in imported_reads):
        return None
    for node in scalar_reads:
        # Keep source metadata reads in their original outer region. A scalar
        # read introduced inside the page/threshold region needs a new proof.
        value = node.meta.get("val")
        if (
            node.graph in page_regions
            or not isinstance(value, torch.Tensor)
            or value.ndim != 0
            or value.dtype not in (torch.int32, torch.int64)
            or len(node.args) != 4
            or node.args[2:] != (None, None)
            or node.kwargs
        ):
            return None
    if any(not host_load_is_readonly(node, env, ir.graphs) for node in reads):
        return None
    destinations = {}
    for write in writes:
        target = write.args[0]
        if not isinstance(target, torch.fx.Node):
            return None
        fake = target.meta.get("val")
        if target.target is not _tracing_ops._host_tensor or not isinstance(
            fake, torch.Tensor
        ):
            return None
        destinations[target] = fake
    disjoint = tensor_pairs_with_proven_disjoint_storage(env, destinations)
    if any(
        frozenset((left, right)) not in disjoint
        for left, right in itertools.combinations(destinations, 2)
    ):
        return None
    proved = (
        recurrence.proof_nodes
        | accesses.key.decode.nodes
        | accesses.value.decode.nodes
        | set(reads)
        | set(writes)
    )
    control_nodes = {
        region.parent
        for region in recurrence.regions.regions
        if region.parent is not None
    }
    for node in nodes:
        if node.op in ("placeholder", "output") or node in proved | control_nodes:
            continue
        if node.op != "call_function":
            return None
        if pure_producer(node):
            continue
        if node.target in (
            _tracing_ops._new_var,
            _tracing_ops._host_tensor,
            _tracing_ops._get_symnode,
            _tracing_ops._phi,
            operator.getitem,
            creation_ops.full,
            tile_ops.tile_id,
            torch.ops.prims.iota.default,
            torch.ops.aten.full.default,
            torch.ops.aten.scalar_tensor.default,
        ):
            continue
        if node.target in _SCALAR_OPERATORS and isinstance(
            node.meta.get("val"), (torch.SymInt, torch.SymBool, int, bool)
        ):
            continue
        return None
    return ConditionalAttentionProgram(
        recurrence, accesses, reads, scalar_reads, writes, imported_reads
    )
