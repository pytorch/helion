"""Shared-memory accounting for a single native GEMM epilogue."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

import torch

from .tcgen05_constants import TCGEN05_AUX_LOAD_MODES_STAGED
from .tcgen05_constants import TCGEN05_AUX_STAGE_COUNT_DEFAULT
from .tcgen05_constants import Tcgen05RowvecAuxRow
from .tcgen05_constants import tcgen05_ab_smem_bytes_per_cta
from .tcgen05_constants import tcgen05_aux_ring_smem_bytes_per_cta
from .tcgen05_constants import tcgen05_c_smem_bytes_per_cta
from .tcgen05_constants import tcgen05_rowvec_stage_smem_bytes

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from torch.fx import Node

    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from .cute_mma import _CuteMmaNode


TCGEN05_CTA_GROUP_CONFIG_KEY = "tcgen05_cta_group"
TCGEN05_CTA_GROUP_CHOICES = ("auto", "two")
TCGEN05_SMEM_AWARE_MAX_AB_STAGES = 16


def tcgen05_bookkeeping_smem_bytes(*, ab_stages: int, acc_stages: int) -> int:
    """Fixed and per-stage allocations besides the AB and output rings.

    The plain static schedule has a TMEM pointer/deallocation cell, aligned
    A/B allocations (128B), one D ring (1024B), and CuTe launch scratch.
    2KiB covers their fixed bytes and alignment gaps. AB completion uses
    a full and an empty 8B barrier per stage, as does the accumulator ring.
    CLC, extra output rings and auxiliary TMA allocations are excluded by
    the caller rather than charged this smaller reservation.
    """
    return 2 * 1024 + 16 * ab_stages + 16 * acc_stages


def deepest_fitting_ab_stages(
    smem_bytes: Callable[[int], int],
    capacity_bytes: int,
    *,
    max_stages: int = TCGEN05_SMEM_AWARE_MAX_AB_STAGES,
) -> int:
    """Return the deepest AB ring in ``1..max_stages`` that fits, else 0."""
    return next(
        (
            stages
            for stages in range(max_stages, 0, -1)
            if smem_bytes(stages) <= capacity_bytes
        ),
        0,
    )


class Tcgen05PipelineSmemFacts(NamedTuple):
    """One store ring and its proved broadcast rows; no other scratch.

    ``rows`` lists the N-broadcast rows (FP8 FP32 scale rows, 16-bit or FP32
    bias rows) a ``pre_acc_wait`` epilogue stages in shared memory.
    ``has_auxiliary_loads`` records the auxiliary operands of the epilogue
    chain (exact-shape tiles, broadcast rows; FP8 columns): register-resident
    SIMT loads under ``post_acc_wait`` that offer the aux-placement seeds.
    ``exact_aux_itemsizes`` holds the element size of each exact-shape rank-2
    auxiliary operand: the staged aux load modes (``tcgen05_aux_load_mode`` in
    ``TCGEN05_AUX_LOAD_MODES_STAGED``) keep one ``tcgen05_aux_stages``-deep
    ring of epilogue subtiles per operand in its own dtype, which the AB ring
    must leave room for; ``ring_aux_itemsizes`` lists the operands those rings
    carry (``stages_through_smem_ring``: a boolean mask stays on the direct
    read, which the register mechanisms still prefetch and hoist).
    ``epilogue_transcendental`` records a transcendental unary op in the chain
    (``Tcgen05UnaryEpilogueChain.has_transcendental_op``).
    """

    input_dtype_bytes: int
    output_dtype_bytes: int
    capacity_bytes: int
    rows: tuple[Tcgen05RowvecAuxRow, ...] = ()
    has_auxiliary_loads: bool = False
    output_column_major: bool = False
    exact_aux_itemsizes: tuple[int, ...] = ()
    epilogue_transcendental: bool = False
    ring_aux_itemsizes: tuple[int, ...] = ()


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


def has_other_store(graphs: Sequence[GraphInfo], store: torch.fx.Node) -> bool:
    """Whether any graph stores to memory other than through ``store``."""
    from ...language import memory_ops

    return any(
        other.op == "call_function"
        and other.target is memory_ops.store
        and other is not store
        for graph in graphs
        for other in graph.graph.nodes
    )


def analyze_pipeline_smem_facts(
    candidate: _CuteMmaNode,
    node: Node,
    graphs: Sequence[GraphInfo],
    *,
    capacity_bytes: int,
    allow_leading_passthrough: bool = False,
    allow_broadcasts: bool = False,
) -> Tcgen05PipelineSmemFacts | None:
    """The allocation facts of ``candidate``'s paired pipeline, or ``None``.

    ``allow_broadcasts`` admits broadcast epilogue operands (FP8 scale rows
    and columns, 16-bit bias rows), and with them the whole FP8 family, where
    the caller's launch charges their stages: the single-root, unfissioned
    kernel, whose ``pre_acc_wait`` row stage the stage limit prices. A
    materialized region's limits price no placement, so its launches keep
    refusing them.
    """
    from .aux_tensor import analyze_tcgen05_matmul_store_chains
    from .aux_tensor import stages_through_smem_ring
    from .cute_epilogue import tcgen05_rowvec_aux_row
    from .cute_mma import _tcgen05_tma_matrix_major
    from .radix_store import proven_radix_store_layout

    operands = candidate.operands
    rank = 3 if operands.has_leading_passthrough else 2
    if (
        capacity_bytes <= 0
        or (operands.has_leading_passthrough and not allow_leading_passthrough)
        or operands.lhs.source_fake.ndim != rank
        or operands.rhs.source_fake.ndim != rank
        or operands.lhs.source_fake.dtype
        not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
        or operands.rhs.source_fake.dtype != operands.lhs.source_fake.dtype
        or not operands.lhs_tma_major_supported
        or operands.rhs.matrix_major not in ("row", "col")
        or candidate.output_store_analysis is None
        or candidate.requires_accumulator_seed
    ):
        return None
    stores = analyze_tcgen05_matmul_store_chains(graphs, node)
    if stores is None or len(stores) != 1:
        return None
    store, chain = stores[0]
    rows: list[Tcgen05RowvecAuxRow] = []
    fp8 = operands.lhs.source_fake.dtype == torch.float8_e4m3fn
    if fp8 and (
        not allow_broadcasts
        or operands.has_leading_passthrough
        or operands.lhs.source_to_logical_order is not None
        or operands.rhs.source_to_logical_order is not None
        or operands.rhs.rhs_is_grouped
    ):
        return None
    # FP8: every auxiliary leaf is an FP32 scale row (axis 1) or column (axes
    # 0, 2). 16-bit: exact-shape tiles (a residual, a mask of the output's
    # shape), which the SIMT aux path loads per subtile into registers with
    # no shared memory, plus 16- or 32-bit bias rows: ``bias[tile_n]`` (axis
    # 1) stages through shared memory under ``pre_acc_wait``, a row stage the
    # model charges, and the explicit (1, N) row (axis 0) is read in place.
    # Broadcast columns of a 16-bit epilogue are not modeled.
    axes, dtypes = (
        ((0, 1, 2), (torch.float32,))
        if fp8
        else ((0, 1), (torch.float16, torch.bfloat16, torch.float32))
    )
    for leaf in chain.auxiliary_tensor_loads:
        if leaf.broadcast_axis is None and not fp8:
            continue
        source = leaf.load_node.args[0]
        tensor = source.meta.get("val") if isinstance(source, torch.fx.Node) else None
        if (
            not allow_broadcasts
            or leaf.broadcast_axis not in axes
            or not isinstance(tensor, torch.Tensor)
            or tensor.dtype not in dtypes
            or any(
                extent != 1 and stride not in (0, 1)
                for extent, stride in zip(tensor.shape, tensor.stride(), strict=True)
            )
        ):
            return None
        if leaf.broadcast_axis == 1:
            rows.append(tcgen05_rowvec_aux_row(chain, leaf, tensor.dtype.itemsize))
    if fp8:
        for scalar in chain.runtime_scalars:
            if scalar.is_host_scalar:
                continue
            assert isinstance(scalar.source, torch.fx.Node)
            source = scalar.source.args[0]
            if not isinstance(source, torch.fx.Node):
                return None
            tensor = source.meta.get("val")
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or tensor.ndim != 0
            ):
                return None
    # A second output can introduce another store ring, even when it does not
    # consume this MMA. Do not borrow its memory for the AB pipeline.
    if has_other_store(graphs, store):
        return None
    output_node = store.args[0]
    if not isinstance(output_node, torch.fx.Node):
        return None
    output = output_node.meta["val"]
    radix_store = proven_radix_store_layout(store) is not None
    if radix_store:
        # A hierarchical (radix) destination stages the row-major [M, N] tile.
        output_major = "row"
    elif isinstance(output, torch.Tensor):
        output_major = _tcgen05_tma_matrix_major(output)
    else:
        output_major = None
    if (
        not isinstance(output, torch.Tensor)
        or (output.ndim != rank and not radix_store)
        or output.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or output_major not in ("row", "col")
        or (output_major == "col") != candidate.output_column_major
    ):
        return None
    exact_aux_itemsizes: list[int] = []
    ring_aux_itemsizes: list[int] = []
    for leaf in chain.auxiliary_tensor_loads:
        source = leaf.load_node.args[0]
        tensor = source.meta.get("val") if isinstance(source, torch.fx.Node) else None
        if (
            leaf.broadcast_axis is None
            and isinstance(tensor, torch.Tensor)
            and tensor.ndim == 2
        ):
            exact_aux_itemsizes.append(tensor.dtype.itemsize)
            if stages_through_smem_ring(leaf.broadcast_axis, tensor):
                ring_aux_itemsizes.append(tensor.dtype.itemsize)
    return Tcgen05PipelineSmemFacts(
        operands.lhs.source_fake.dtype.itemsize,
        output.dtype.itemsize,
        capacity_bytes,
        rows=tuple(rows),
        has_auxiliary_loads=bool(chain.auxiliary_tensor_loads),
        output_column_major=candidate.output_column_major,
        exact_aux_itemsizes=tuple(exact_aux_itemsizes),
        epilogue_transcendental=chain.has_transcendental_op,
        ring_aux_itemsizes=tuple(ring_aux_itemsizes),
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
    aux_load_placement: str = "post_acc_wait",
    store_mode: str = "normal",
    epi_warps: int = 4,
    aux_load_mode: str = "simt",
    aux_stages: int = TCGEN05_AUX_STAGE_COUNT_DEFAULT,
    epi_warp_groups: int = 1,
    epi_subtile_n: int | None = None,
) -> int:
    """Upper-bound the allocations for the static CtaGroup.TWO schedule.

    The 128-row MMA uses a 64-row per-CTA, N-permuted epilogue. Its logical
    element count is the compact store-ring allocation size; the same CuTe
    sizing routine chooses the tile used by device and descriptor codegen.
    The 256-row family takes CuTe's with-source tile unless ``epi_subtile_n``
    pins it (below), and both families use the destination's actual row- or
    column-major layout.
    A staged aux load mode adds one ``aux_stages``-deep ring of epilogue
    subtiles per operand a staged ring carries (``facts.ring_aux_itemsizes``)
    plus its full/empty barriers.  ``epi_warp_groups`` multiplies the C
    stages (each epilogue warp group owns ``c_stages`` stages of the ring,
    ``tcgen05_epi_warp_groups``) and ``epi_subtile_n`` pins the 256-row
    pair's C subtile to ``(128, epi_subtile_n)``: a configured
    ``tcgen05_epi_subtile_n`` (the explicit-tile path), or the plan's own
    (128, 32) subtile of a plain 16-bit store on static full tiles (or the
    aligned-N physical tile with a partial last M tile), which the paired
    caller resolves (the staged aux rings keep the plan's
    with-source subtile, as ``cute_mma`` sizes them).
    """
    from cutlass.utils.blackwell_helpers import compute_epilogue_tile_size

    # Non-power-of-two N widths are the aligned-N family's physical tiles.
    assert bn in (64, 128, 256) or (bn % 16 == 0 and 64 < bn < 256)
    assert bm in (128, 256) and bk in (64, 128, 256)
    assert facts.input_dtype_bytes in (1, 2)
    assert facts.output_dtype_bytes in (2, 4)
    assert ab_stages > 0 and c_stages in (2, 4) and acc_stages in (1, 2)
    assert epi_warp_groups >= 1
    two_cta_m128 = bm == 128
    output_width = facts.output_dtype_bytes * 8
    epi_m, epi_n = compute_epilogue_tile_size(
        bm // 2 if two_cta_m128 else bm,
        bn,
        two_cta_m128,
        output_width,
        None if two_cta_m128 else output_width,
        d_is_m_major=facts.output_column_major,
        c_is_m_major=facts.output_column_major,
    )
    c_tile_m, c_tile_n = epi_m, epi_n
    if epi_subtile_n is not None and not two_cta_m128:
        # The configured subtile of the 256-row pair: one CTA walks whole
        # 128-lane TMEM datapath tiles of its 128-row half
        # (``strategies.tcgen05_epilogue_subtile_m``).
        c_tile_m, c_tile_n = 128, epi_subtile_n
    ab_bytes = tcgen05_ab_smem_bytes_per_cta(
        bm=bm,
        bn=bn,
        bk=bk,
        dtype_bytes=facts.input_dtype_bytes,
        ab_stages=ab_stages,
        cluster_m=2,
    )
    c_bytes = tcgen05_c_smem_bytes_per_cta(
        epi_tile_m=c_tile_m,
        epi_tile_n=c_tile_n,
        dtype_bytes=facts.output_dtype_bytes,
        c_stages=c_stages * epi_warp_groups,
    )
    # Only the M256 normal epilogue stages FP32 N-broadcasts. The M128
    # instruction has a 64-row per-CTA epilogue, below the staging gate.
    # Direct stores allocate neither rows nor D; retaining D here is a
    # conservative bound and deliberately does not admit deeper direct rings.
    row_bytes = (
        tcgen05_rowvec_stage_smem_bytes(rows=facts.rows, bn=bn, epi_warps=epi_warps)
        if bm == 256 and store_mode == "normal" and aux_load_placement == "pre_acc_wait"
        else 0
    )
    aux_ring_bytes = (
        tcgen05_aux_ring_smem_bytes_per_cta(
            epi_tile_m=epi_m,
            epi_tile_n=epi_n,
            aux_stages=aux_stages,
            ring_itemsizes=facts.ring_aux_itemsizes,
        )
        if aux_load_mode in TCGEN05_AUX_LOAD_MODES_STAGED
        else 0
    )
    return (
        ab_bytes
        + c_bytes
        + row_bytes
        + aux_ring_bytes
        + tcgen05_bookkeeping_smem_bytes(ab_stages=ab_stages, acc_stages=acc_stages)
    )


def max_pipeline_ab_stages(
    facts: Tcgen05PipelineSmemFacts,
    *,
    bm: int,
    bn: int,
    bk: int,
    c_stages: int,
    acc_stages: int,
    aux_load_placement: str = "post_acc_wait",
    store_mode: str = "normal",
    epi_warps: int = 4,
    aux_load_mode: str = "simt",
    aux_stages: int = TCGEN05_AUX_STAGE_COUNT_DEFAULT,
    epi_warp_groups: int = 1,
    epi_subtile_n: int | None = None,
) -> int:
    """Fill shared memory without changing the epilogue or accumulator ring."""
    return deepest_fitting_ab_stages(
        lambda stages: pipeline_smem_bytes(
            facts,
            bm=bm,
            bn=bn,
            bk=bk,
            ab_stages=stages,
            c_stages=c_stages,
            acc_stages=acc_stages,
            aux_load_placement=aux_load_placement,
            store_mode=store_mode,
            epi_warps=epi_warps,
            aux_load_mode=aux_load_mode,
            aux_stages=aux_stages,
            epi_warp_groups=epi_warp_groups,
            epi_subtile_n=epi_subtile_n,
        ),
        facts.capacity_bytes,
    )
