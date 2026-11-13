"""Shared-memory accounting of one native tcgen05 GEMM launch.

Two byte models price the per-CTA arena -- ``pipeline_smem_bytes`` for the
static CtaGroup.TWO schedule and ``one_cta_pipeline_smem_bytes`` for the
role-local CtaGroup.ONE schedule -- and ``SmemArena`` is the one object the
stage searches reach them through (``max_pipeline_ab_stages`` and its
one-CTA alias).  Every budget here is ``Tcgen05PipelineSmemFacts.capacity_bytes``,
the device's opt-in capacity read once at bind.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import NamedTuple

import torch

from .tcgen05_constants import TCGEN05_AUX_LOAD_MODES_STAGED
from .tcgen05_constants import TCGEN05_AUX_STAGE_COUNT_DEFAULT
from .tcgen05_constants import Tcgen05RowvecAuxRow
from .tcgen05_constants import tcgen05_ab_smem_bytes_per_cta
from .tcgen05_constants import tcgen05_aux_ring_smem_bytes_per_cta
from .tcgen05_constants import tcgen05_c_smem_bytes_per_cta
from .tcgen05_constants import tcgen05_default_epilogue_tile_size
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
    """One store ring and its proved FP32 broadcast rows; no other scratch.

    ``has_auxiliary_loads`` records exact-shape (or, for FP8, broadcast)
    auxiliary operands of the epilogue chain: register-resident SIMT loads
    that add no shared memory but offer the aux-placement seeds.
    ``exact_aux_count`` is the number of exact-shape rank-2 auxiliary
    operands of the output's dtype: the staged aux load modes
    (``tcgen05_aux_load_mode`` in ``TCGEN05_AUX_LOAD_MODES_STAGED``) keep one
    ``tcgen05_aux_stages``-deep ring of epilogue subtiles per operand, which
    the AB ring must leave room for.
    """

    input_dtype_bytes: int
    output_dtype_bytes: int
    capacity_bytes: int
    rows: tuple[Tcgen05RowvecAuxRow, ...] = ()
    has_auxiliary_loads: bool = False
    output_column_major: bool = False
    exact_aux_count: int = 0


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
    allow_fp8_broadcasts: bool = False,
) -> Tcgen05PipelineSmemFacts | None:
    from .aux_tensor import analyze_tcgen05_matmul_store_chains
    from .cute_mma import _tcgen05_tma_matrix_major

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
    if operands.lhs.source_fake.dtype == torch.float8_e4m3fn:
        if (
            not allow_fp8_broadcasts
            or operands.has_leading_passthrough
            or operands.lhs.source_to_logical_order is not None
            or operands.rhs.source_to_logical_order is not None
            or operands.rhs.rhs_is_grouped
        ):
            return None
        for leaf in chain.auxiliary_tensor_loads:
            source = leaf.load_node.args[0]
            if not isinstance(source, torch.fx.Node):
                return None
            tensor = source.meta.get("val")
            if (
                leaf.broadcast_axis not in (0, 1, 2)
                or not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or any(
                    extent != 1 and stride not in (0, 1)
                    for extent, stride in zip(
                        tensor.shape, tensor.stride(), strict=True
                    )
                )
            ):
                return None
            if leaf.broadcast_axis == 1:
                rows.append(Tcgen05RowvecAuxRow(itemsize=4, promoted=False))
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
    elif any(leaf.broadcast_axis is not None for leaf in chain.auxiliary_tensor_loads):
        # A 16-bit epilogue may read exact-shape auxiliary tiles (a residual,
        # a mask of the output's shape): the SIMT aux path loads them per
        # subtile into registers and allocates no shared memory, so the
        # static two-CTA arena is the plain one-ring budget.  Broadcast rows
        # and columns stage through shared memory under ``pre_acc_wait`` and
        # are modeled only for the FP8 family above.
        return None
    # A second output can introduce another store ring, even when it does not
    # consume this MMA. Do not borrow its memory for the AB pipeline.
    if has_other_store(graphs, store):
        return None
    output_node = store.args[0]
    if not isinstance(output_node, torch.fx.Node):
        return None
    output = output_node.meta["val"]
    output_major = (
        _tcgen05_tma_matrix_major(output) if isinstance(output, torch.Tensor) else None
    )
    if (
        not isinstance(output, torch.Tensor)
        or output.ndim != rank
        or output.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or output_major not in ("row", "col")
        or (output_major == "col") != candidate.output_column_major
    ):
        return None
    exact_aux_count = 0
    for leaf in chain.auxiliary_tensor_loads:
        source = leaf.load_node.args[0]
        tensor = source.meta.get("val") if isinstance(source, torch.fx.Node) else None
        if (
            leaf.broadcast_axis is None
            and isinstance(tensor, torch.Tensor)
            and tensor.ndim == 2
            and tensor.dtype == output.dtype
        ):
            exact_aux_count += 1
    return Tcgen05PipelineSmemFacts(
        operands.lhs.source_fake.dtype.itemsize,
        output.dtype.itemsize,
        capacity_bytes,
        tuple(rows),
        bool(chain.auxiliary_tensor_loads),
        candidate.output_column_major,
        exact_aux_count,
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
    subtiles per exact-shape auxiliary operand (``facts.exact_aux_count``)
    plus its full/empty barriers.  ``epi_warp_groups`` multiplies the C
    stages (each epilogue warp group owns ``c_stages`` stages of the ring,
    ``tcgen05_epi_warp_groups``) and ``epi_subtile_n`` pins the 256-row
    pair's C subtile to ``(128, epi_subtile_n)``: a configured
    ``tcgen05_epi_subtile_n`` (the explicit-tile path), or the plan's own
    (128, 32) subtile of a plain 16-bit store on static full tiles, which
    the paired caller resolves (the staged aux rings keep the plan's
    with-source subtile, as ``cute_mma`` sizes them).
    """
    from cutlass.utils.blackwell_helpers import compute_epilogue_tile_size

    # Non-power-of-two N widths are proved only for the M128 instruction.
    assert bn in (64, 128, 256) or (bm == 128 and bn % 16 == 0 and 64 < bn < 256)
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
            dtype_bytes=facts.output_dtype_bytes,
            aux_stages=aux_stages,
            ring_count=facts.exact_aux_count,
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


def paired_pipeline_tile_modeled(
    *, bm: int, bn: int, bk: int, c_stages: int, acc_stages: int
) -> bool:
    """The tiles and ring counts ``pipeline_smem_bytes`` prices.

    Non-power-of-two N widths are proved only for the M128 instruction.
    """
    return (
        (bn in (64, 128, 256) or (bm == 128 and bn % 16 == 0 and 64 < bn < 256))
        and bm in (128, 256)
        and bk in (64, 128, 256)
        and c_stages in (2, 4)
        and acc_stages in (1, 2)
    )


def one_cta_pipeline_tile_modeled(
    *, bm: int, bn: int, bk: int, c_stages: int, acc_stages: int
) -> bool:
    """The tiles and ring counts ``one_cta_pipeline_smem_bytes`` prices."""
    return (
        bm in (64, 128)
        and bn in (16, 32, 64, 128, 256)
        and bk in (32, 64, 128, 256)
        and c_stages in (2, 4)
        and acc_stages in (1, 2)
    )


def one_cta_pipeline_smem_bytes(
    facts: Tcgen05PipelineSmemFacts,
    *,
    bm: int,
    bn: int,
    bk: int,
    ab_stages: int,
    c_stages: int,
    acc_stages: int,
    aux_load_mode: str = "simt",
    aux_stages: int = TCGEN05_AUX_STAGE_COUNT_DEFAULT,
    epi_warp_groups: int = 1,
    epi_subtile_n: int | None = None,
) -> int:
    """Conservative AB + actual-width DEFAULT D + bookkeeping bound.

    Only post-wait SIMT auxiliary loads are admitted, so scales need no SMEM.
    Direct stores retain the D-ring charge. The shared DEFAULT sizing helper
    follows the same with-source convention as the device/descriptor resolver.
    The model prices one- and two-byte operands (the FP8 persistent pipeline
    and the flat 16-bit launches of an ordered barrier chain run the same
    role-local CtaGroup.ONE schedule).  A staged aux load mode adds one
    ``aux_stages``-deep ring of the DEFAULT epilogue subtile per exact-shape
    auxiliary operand (``facts.exact_aux_count``), ``epi_warp_groups``
    multiplies the C stages and ``epi_subtile_n`` pins the C subtile to
    ``(min(bm, 128), epi_subtile_n)`` (``tcgen05_epi_subtile_n``), as the
    paired model does.  This function does not change any legacy one- or
    two-CTA allocation model.
    """
    assert one_cta_pipeline_tile_modeled(
        bm=bm, bn=bn, bk=bk, c_stages=c_stages, acc_stages=acc_stages
    )
    assert facts.input_dtype_bytes in (1, 2) and facts.output_dtype_bytes in (2, 4)
    assert ab_stages > 0 and epi_warp_groups >= 1
    width = facts.output_dtype_bytes * 8
    epi_m, epi_n = tcgen05_default_epilogue_tile_size(
        bm, bn, elem_width_d=width, elem_width_c=width
    )
    c_tile_m, c_tile_n = epi_m, epi_n
    if epi_subtile_n is not None:
        c_tile_m, c_tile_n = min(bm, 128), epi_subtile_n
    aux_ring_bytes = (
        tcgen05_aux_ring_smem_bytes_per_cta(
            epi_tile_m=epi_m,
            epi_tile_n=epi_n,
            dtype_bytes=facts.output_dtype_bytes,
            aux_stages=aux_stages,
            ring_count=facts.exact_aux_count,
        )
        if aux_load_mode in TCGEN05_AUX_LOAD_MODES_STAGED
        else 0
    )
    return (
        tcgen05_ab_smem_bytes_per_cta(
            bm=bm,
            bn=bn,
            bk=bk,
            dtype_bytes=facts.input_dtype_bytes,
            ab_stages=ab_stages,
            cluster_m=1,
        )
        + tcgen05_c_smem_bytes_per_cta(
            epi_tile_m=c_tile_m,
            epi_tile_n=c_tile_n,
            dtype_bytes=facts.output_dtype_bytes,
            c_stages=c_stages * epi_warp_groups,
        )
        + aux_ring_bytes
        + tcgen05_bookkeeping_smem_bytes(ab_stages=ab_stages, acc_stages=acc_stages)
    )


@dataclasses.dataclass(frozen=True)
class SmemArena:
    """The per-CTA shared-memory arena of one tcgen05 GEMM launch, one pricer.

    ``cluster_m`` selects the byte model: ``2`` is the static CtaGroup.TWO
    schedule (``pipeline_smem_bytes``: the 128-row MMA's 64-row per-CTA
    epilogue, the 256-row family's FP32 row stages under ``pre_acc_wait``),
    ``1`` the role-local CtaGroup.ONE schedule (``one_cta_pipeline_smem_bytes``:
    the DEFAULT epilogue tile, no row stages, so ``aux_load_placement`` /
    ``store_mode`` / ``epi_warps`` do not change its bytes).  Every ring must
    fit ``facts.capacity_bytes``; ``deepest_ab_stages`` is the one search over
    AB depth the seeds, the repair chain and the validators share.
    """

    facts: Tcgen05PipelineSmemFacts
    cluster_m: int = 2

    def __post_init__(self) -> None:
        assert self.cluster_m in (1, 2), self.cluster_m

    def bytes(
        self,
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
        """Bytes of the launch's rings under this arena's byte model."""
        if self.cluster_m == 1:
            return one_cta_pipeline_smem_bytes(
                self.facts,
                bm=bm,
                bn=bn,
                bk=bk,
                ab_stages=ab_stages,
                c_stages=c_stages,
                acc_stages=acc_stages,
                aux_load_mode=aux_load_mode,
                aux_stages=aux_stages,
                epi_warp_groups=epi_warp_groups,
                epi_subtile_n=epi_subtile_n,
            )
        return pipeline_smem_bytes(
            self.facts,
            bm=bm,
            bn=bn,
            bk=bk,
            ab_stages=ab_stages,
            c_stages=c_stages,
            acc_stages=acc_stages,
            aux_load_placement=aux_load_placement,
            store_mode=store_mode,
            epi_warps=epi_warps,
            aux_load_mode=aux_load_mode,
            aux_stages=aux_stages,
            epi_warp_groups=epi_warp_groups,
            epi_subtile_n=epi_subtile_n,
        )

    def deepest_ab_stages(
        self,
        *,
        bm: int,
        bn: int,
        bk: int,
        c_stages: int,
        acc_stages: int,
        max_stages: int = TCGEN05_SMEM_AWARE_MAX_AB_STAGES,
        aux_load_placement: str = "post_acc_wait",
        store_mode: str = "normal",
        epi_warps: int = 4,
        aux_load_mode: str = "simt",
        aux_stages: int = TCGEN05_AUX_STAGE_COUNT_DEFAULT,
        epi_warp_groups: int = 1,
        epi_subtile_n: int | None = None,
    ) -> int:
        """Deepest AB ring in ``1..max_stages`` that fits the capacity, else 0.

        The epilogue and accumulator rings are the caller's; only the AB
        depth is searched.
        """
        return deepest_fitting_ab_stages(
            lambda stages: self.bytes(
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
            self.facts.capacity_bytes,
            max_stages=max_stages,
        )


def max_pipeline_ab_stages(
    facts: Tcgen05PipelineSmemFacts,
    *,
    bm: int,
    bn: int,
    bk: int,
    c_stages: int,
    acc_stages: int,
    cluster_m: int = 2,
    max_stages: int = TCGEN05_SMEM_AWARE_MAX_AB_STAGES,
    aux_load_placement: str = "post_acc_wait",
    store_mode: str = "normal",
    epi_warps: int = 4,
    aux_load_mode: str = "simt",
    aux_stages: int = TCGEN05_AUX_STAGE_COUNT_DEFAULT,
    epi_warp_groups: int = 1,
    epi_subtile_n: int | None = None,
) -> int:
    """Fill shared memory without changing the epilogue or accumulator ring.

    ``SmemArena(facts, cluster_m).deepest_ab_stages`` by another name: the
    paired byte model unless ``cluster_m=1`` selects the one-CTA model.
    """
    return SmemArena(facts, cluster_m).deepest_ab_stages(
        bm=bm,
        bn=bn,
        bk=bk,
        c_stages=c_stages,
        acc_stages=acc_stages,
        max_stages=max_stages,
        aux_load_placement=aux_load_placement,
        store_mode=store_mode,
        epi_warps=epi_warps,
        aux_load_mode=aux_load_mode,
        aux_stages=aux_stages,
        epi_warp_groups=epi_warp_groups,
        epi_subtile_n=epi_subtile_n,
    )


def max_one_cta_pipeline_ab_stages(
    facts: Tcgen05PipelineSmemFacts,
    *,
    bm: int,
    bn: int,
    bk: int,
    c_stages: int,
    acc_stages: int,
    max_stages: int = TCGEN05_SMEM_AWARE_MAX_AB_STAGES,
    aux_load_mode: str = "simt",
    aux_stages: int = TCGEN05_AUX_STAGE_COUNT_DEFAULT,
    epi_warp_groups: int = 1,
    epi_subtile_n: int | None = None,
) -> int:
    """Deepest AB ring in ``1..max_stages`` the one-CTA launch fits, else 0.

    The one-CTA model's historical name for ``max_pipeline_ab_stages(...,
    cluster_m=1)``.
    """
    return max_pipeline_ab_stages(
        facts,
        bm=bm,
        bn=bn,
        bk=bk,
        c_stages=c_stages,
        acc_stages=acc_stages,
        cluster_m=1,
        max_stages=max_stages,
        aux_load_mode=aux_load_mode,
        aux_stages=aux_stages,
        epi_warp_groups=epi_warp_groups,
        epi_subtile_n=epi_subtile_n,
    )
