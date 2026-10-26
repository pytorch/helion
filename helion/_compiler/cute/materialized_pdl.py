"""Join the proved SIMT materialization to its guarded native MMA consumer."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

from ...language import memory_ops
from .materialized_fission_codegen import _region_graph_ids
from .promote_output_axis import _access_tensor
from .promote_output_axis import _fresh_tensors

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .cute_mma import _CuteMmaNode


class MaterializedOperandPdl(NamedTuple):
    """Root identities of the proved producer/consumer edge and the MMA side
    (``"lhs"`` or ``"rhs"``) that reads the materialized scratch tensor."""

    producer: int
    consumer: int
    dependent_side: str


def prove_materialized_operand_pdl(
    env: CompileEnvironment, device_ir: DeviceIR, candidate: _CuteMmaNode
) -> MaterializedOperandPdl | None:
    """Recheck the typed materialization's fresh scratch edge after retracing.

    The original extractor proves the pure complete producer, unchanged inputs,
    fresh allocations and two ordered launches. The caller has additionally
    accepted exactly one guarded native MMA and its complete region SMEM facts.
    Join those independent proofs by actual graph and tensor identities.
    """
    plan = env.cute_fission_plan
    host = device_ir.host_function
    if (
        plan is None
        or host is None
        or plan.operand_interleave_factor is None
        or plan.region_count != 2
        or plan.pointwise_region_indices != (0,)
        or len(device_ir.root_ids) != 2
    ):
        return None
    producer, consumer = device_ir.root_ids
    producer_graphs = _region_graph_ids(device_ir.graphs, producer)
    consumer_graphs = _region_graph_ids(device_ir.graphs, consumer)
    if not any(
        info.graph is candidate.lhs.graph and info.graph_id in consumer_graphs
        for info in device_ir.graphs
    ):
        return None
    # The materialized operand is the fresh scratch tensor on either side of
    # the contraction (a transformed RHS recipe or a cast LHS recipe).
    fresh = _fresh_tensors(host)
    scratch = next(
        (
            operand.source_fake
            for operand in (candidate.operands.rhs, candidate.operands.lhs)
            if operand.source_fake in fresh
        ),
        None,
    )
    if scratch is None:
        return None
    stores = [
        node
        for info in device_ir.graphs
        if info.graph_id in producer_graphs
        for node in info.graph.nodes
        if node.target is memory_ops.store
    ]
    if not stores or any(_access_tensor(store) is not scratch for store in stores):
        return None
    side = "rhs" if candidate.operands.rhs.source_fake is scratch else "lhs"
    return MaterializedOperandPdl(producer, consumer, side)
