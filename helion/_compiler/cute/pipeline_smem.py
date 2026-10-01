"""Shared-memory accounting for a single native, row-major GEMM epilogue."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

import torch

from .tcgen05_constants import tcgen05_ab_smem_bytes_per_cta
from .tcgen05_constants import tcgen05_c_smem_bytes_per_cta

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node

    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from .cute_mma import _CuteMmaNode


TCGEN05_CTA_GROUP_CONFIG_KEY = "tcgen05_cta_group"
TCGEN05_CTA_GROUP_CHOICES = ("auto", "two")
TCGEN05_SMEM_AWARE_MAX_AB_STAGES = 16


class Tcgen05PipelineSmemFacts(NamedTuple):
    """A single store ring with no auxiliary or reduction scratch allocations."""

    input_dtype_bytes: int
    output_dtype_bytes: int
    capacity_bytes: int


def pipeline_region_graphs(
    device_ir: DeviceIR, node: Node, *, separately_launched: bool
) -> tuple[GraphInfo, ...] | None:
    """Charge only allocations in the candidate's proven device function."""
    if len(device_ir.root_ids) == 1:
        return tuple(device_ir.graphs)
    if not separately_launched:
        return None
    from .materialized_fission_codegen import _region_graph_ids

    graph_id = next(
        info.graph_id for info in device_ir.graphs if info.graph is node.graph
    )
    owners = [
        region
        for root in device_ir.root_ids
        if graph_id in (region := _region_graph_ids(device_ir.graphs, root))
    ]
    if len(owners) != 1:
        return None
    return tuple(info for info in device_ir.graphs if info.graph_id in owners[0])


def analyze_pipeline_smem_facts(
    candidate: _CuteMmaNode,
    node: Node,
    graphs: Sequence[GraphInfo],
    *,
    capacity_bytes: int,
    allow_leading_passthrough: bool = False,
) -> Tcgen05PipelineSmemFacts | None:
    from ...language import memory_ops
    from .aux_tensor import analyze_tcgen05_matmul_store_chains
    from .cute_mma import _tcgen05_tma_matrix_major

    operands = candidate.operands
    rank = 3 if operands.has_leading_passthrough else 2
    if (
        capacity_bytes <= 0
        or (operands.has_leading_passthrough and not allow_leading_passthrough)
        or operands.lhs.source_fake.ndim != rank
        or operands.rhs.source_fake.ndim != rank
        or operands.lhs.source_fake.dtype not in (torch.float16, torch.bfloat16)
        or operands.rhs.source_fake.dtype != operands.lhs.source_fake.dtype
        or operands.lhs.matrix_major != "row"
        or operands.rhs.matrix_major not in ("row", "col")
        or candidate.output_store_analysis is None
        or candidate.requires_accumulator_seed
    ):
        return None
    stores = analyze_tcgen05_matmul_store_chains(graphs, node)
    if stores is None or len(stores) != 1:
        return None
    store, chain = stores[0]
    if chain.auxiliary_tensor_loads:
        return None
    # A second output can introduce another store ring, even when it does not
    # consume this MMA. Do not borrow its memory for the AB pipeline.
    if any(
        other.op == "call_function"
        and other.target is memory_ops.store
        and other is not store
        for graph in graphs
        for other in graph.graph.nodes
    ):
        return None
    output_node = store.args[0]
    if not isinstance(output_node, torch.fx.Node):
        return None
    output = output_node.meta["val"]
    if (
        not isinstance(output, torch.Tensor)
        or output.ndim != rank
        or output.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or _tcgen05_tma_matrix_major(output) != "row"
    ):
        return None
    return Tcgen05PipelineSmemFacts(
        operands.lhs.source_fake.dtype.itemsize,
        output.dtype.itemsize,
        capacity_bytes,
    )


def pipeline_smem_bytes(
    facts: Tcgen05PipelineSmemFacts,
    *,
    bm: int,
    bn: int,
    bk: int,
    ab_stages: int,
    c_stages: int,
    acc_stages: int,
) -> int:
    """Upper-bound the allocations for the static CtaGroup.TWO schedule.

    The 128-row MMA uses a 64-row per-CTA, N-permuted epilogue. Its logical
    element count is the compact store-ring allocation size; the same CuTe
    sizing routine chooses the tile used by device and descriptor codegen.
    The 256-row family keeps its existing with-source tile convention.
    """
    from cutlass.utils.blackwell_helpers import compute_epilogue_tile_size

    assert bm in (128, 256) and bn in (64, 128, 256)
    assert bk in (64, 128, 256)
    assert facts.input_dtype_bytes == 2
    assert facts.output_dtype_bytes in (2, 4)
    assert ab_stages > 0 and c_stages in (2, 4) and acc_stages in (1, 2)
    two_cta_m128 = bm == 128
    output_width = facts.output_dtype_bytes * 8
    epi_m, epi_n = compute_epilogue_tile_size(
        bm // 2 if two_cta_m128 else bm,
        bn,
        two_cta_m128,
        output_width,
        None if two_cta_m128 else output_width,
        d_is_m_major=False,
        c_is_m_major=False,
    )
    ab_bytes = tcgen05_ab_smem_bytes_per_cta(
        bm=bm,
        bn=bn,
        bk=bk,
        dtype_bytes=facts.input_dtype_bytes,
        ab_stages=ab_stages,
        cluster_m=2,
    )
    c_bytes = tcgen05_c_smem_bytes_per_cta(
        epi_tile_m=epi_m,
        epi_tile_n=epi_n,
        dtype_bytes=facts.output_dtype_bytes,
        c_stages=c_stages,
    )
    # The plain static schedule has a TMEM pointer/deallocation cell, aligned
    # A/B allocations (128B), one D ring (1024B), and CuTe launch scratch.
    # 2KiB covers their fixed bytes and alignment gaps. AB completion uses
    # a full and an empty 8B barrier per stage, as does the accumulator ring.
    # CLC, extra output rings and auxiliary TMA allocations are excluded by
    # the caller rather than charged this smaller reservation.
    bookkeeping = 2 * 1024 + 16 * ab_stages + 16 * acc_stages
    return ab_bytes + c_bytes + bookkeeping


def max_pipeline_ab_stages(
    facts: Tcgen05PipelineSmemFacts,
    *,
    bm: int,
    bn: int,
    bk: int,
    c_stages: int,
    acc_stages: int,
) -> int:
    """Fill shared memory without changing the epilogue or accumulator ring."""
    for count in range(TCGEN05_SMEM_AWARE_MAX_AB_STAGES, 0, -1):
        if (
            pipeline_smem_bytes(
                facts,
                bm=bm,
                bn=bn,
                bk=bk,
                ab_stages=count,
                c_stages=c_stages,
                acc_stages=acc_stages,
            )
            <= facts.capacity_bytes
        ):
            return count
    return 0
