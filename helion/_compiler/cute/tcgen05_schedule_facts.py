"""Prove additional dense schedules after native MMA analysis finishes."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch

from .pipeline_smem import has_other_store
from .tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .cute_epilogue import Tcgen05UnaryEpilogueChain
    from .cute_mma import _CuteMmaNode


@dataclass(frozen=True)
class Tcgen05FullTileScheduleFacts:
    """One native, aligned, rank-two MMA with a supported single epilogue.

    Shape order is M, N, K. These are compile-time extents, not runtime hints.
    The proof excludes extra stores, reductions, grouped operands, and source-C
    rings. Broadcast and scalar epilogues use the existing direct load protocol.
    """

    block_ids: tuple[int, int, int]
    shape: tuple[int, int, int]
    dtype: torch.dtype
    rhs_k_major: bool = False
    # False for an M-major (column-major) A, e.g. ``x.T @ grad`` in a matmul
    # backward: the UMMA smem descriptor takes it directly
    # (``make_smem_layout_a(is_k_major=False)``), so the full-tile and
    # paired families apply; the proofs that read A through the K-major
    # partial-tile TMA protocol (N-edge clusters, fp8 K tails) stay K-major.
    lhs_k_major: bool = True


def n_edge_cluster_schedule_defaults() -> dict[str, object]:
    """The unsplit, direct-aux protocol for the additional N-edge family."""
    return {
        "tcgen05_strategy": "role_local_monolithic",
        "tcgen05_persistence_model": "static_persistent",
        "tcgen05_layout_strategy": "default",
        "tcgen05_cta_group": "auto",
        "tcgen05_warp_spec_scheduler_warps": 0,
        "tcgen05_warp_spec_c_input_warps": 0,
        "tcgen05_warp_spec_store_warps": 0,
        "tcgen05_warp_spec_epi_load_warps": 0,
        "tcgen05_num_epi_warps": 4,
        "tcgen05_l2_swizzle_size": 1,
        "l2_groupings": [1],
        "tcgen05_aux_load_mode": "simt",
    }


def n_edge_cluster_geometry_supported(
    facts: Tcgen05FullTileScheduleFacts, tile: tuple[int, int, int]
) -> bool:
    """Prove real tile origins for every peer in a partial-N 2x2 cluster.

    K-major B is required by the existing role-local partial-N TMA proof.
    Padded, entirely invalid peers need a different participation protocol.
    """
    if facts.dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
        return False
    if (
        not facts.rhs_k_major
        or not facts.lhs_k_major
        or any(type(value) is not int or value <= 0 for value in (*facts.shape, *tile))
    ):
        return False
    m, n, k = facts.shape
    bm, bn, bk = tile
    return (
        bm == 256
        and bn in (128, 256)
        and bk in (64, 128)
        and m % bm == 0
        and k % bk == 0
        and k // bk <= TCGEN05_TWO_CTA_MAX_K_TILES
        and n % bn != 0
        and ((n + bn - 1) // bn) % 2 == 0
    )


def n_edge_cluster_schedule_supported(
    facts: Tcgen05FullTileScheduleFacts,
    tile: tuple[int, int, int],
    config: Mapping[str, object],
) -> bool:
    """Shared codegen/search admission; never permit a per-CTA phase split."""
    return (
        n_edge_cluster_geometry_supported(facts, tile)
        and type(config.get("tcgen05_cluster_m")) is int
        and type(config.get("tcgen05_cluster_n")) is int
        and config.get("tcgen05_cluster_m") == config.get("tcgen05_cluster_n") == 2
        and config.get("pid_type") in ("persistent_blocked", "persistent_interleaved")
        and all(
            config.get(key, value) == value
            for key, value in n_edge_cluster_schedule_defaults().items()
        )
    )


def full_tile_schedule_supported(
    facts: Tcgen05FullTileScheduleFacts,
    tile: tuple[int, int, int],
    *,
    cluster_m: int,
    cluster_n: int,
) -> bool:
    """Shared legality predicate for registration, seeds and normalization.

    Admit only exact tiles and the existing automatic instruction families.
    In particular this proof does not admit padded clusters or any K/M/N tail.
    """
    if facts.dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
        return False
    if type(cluster_m) is not int or type(cluster_n) is not int or cluster_n != 1:
        return False
    bm, bn, bk = tile
    k_atom = 32 if facts.dtype == torch.float8_e4m3fn else 16
    if (
        any(type(value) is not int or value <= 0 for value in (*facts.shape, *tile))
        or bn not in (16, 32, 64, 128, 256)
        or bk not in (k_atom, 32, 64, 128, 256)
        or any(extent % value for extent, value in zip(facts.shape, tile, strict=True))
        or facts.shape[2] // bk > TCGEN05_TWO_CTA_MAX_K_TILES
    ):
        return False
    if cluster_m == 1:
        return bm in (64, 128)
    if cluster_m == 2:
        return (
            bn in (128, 256)
            and (bm == 256 or (bm == 128 and facts.dtype == torch.float8_e4m3fn))
        ) or (
            facts.dtype == torch.float8_e4m3fn
            and (bm, bn, bk) in ((128, 64, 256), (256, 64, 128))
        )
    return False


def paired_k_tail_schedule_supported(
    facts: Tcgen05FullTileScheduleFacts,
    tile: tuple[int, int, int],
    *,
    cluster_m: int,
    cluster_n: int,
) -> bool:
    """Admit the existing paired TMA zero-fill protocol with full M/N tiles.

    Every issued K tile starts inside the allocation; the descriptor zero-fills
    only its final partial box. Keep a partial first tile, output fringes and
    multicast clusters outside this proof.
    """
    if (
        facts.dtype != torch.float8_e4m3fn
        or not facts.rhs_k_major
        or not facts.lhs_k_major
        or type(cluster_m) is not int
        or type(cluster_n) is not int
        or (cluster_m, cluster_n) != (2, 1)
        or any(type(value) is not int or value <= 0 for value in (*facts.shape, *tile))
    ):
        return False
    m, n, k = facts.shape
    bm, bn, bk = tile
    return (
        bm in (128, 256)
        and bn in (128, 256)
        and bk in (64, 128, 256)
        and m % bm == 0
        and n % bn == 0
        and k > bk
        and k % bk != 0
        and (k + bk - 1) // bk <= TCGEN05_TWO_CTA_MAX_K_TILES
    )


def native_tma_schedule_supported(
    facts: Tcgen05FullTileScheduleFacts,
    tile: tuple[int, int, int],
    *,
    cluster_m: int,
    cluster_n: int,
) -> bool:
    """Shared search admission for exact tiles and proved paired K tails."""
    return full_tile_schedule_supported(
        facts, tile, cluster_m=cluster_m, cluster_n=cluster_n
    ) or paired_k_tail_schedule_supported(
        facts, tile, cluster_m=cluster_m, cluster_n=cluster_n
    )


@dataclass(frozen=True)
class NativeGemm:
    """The kernel's only native MMA, analyzed once for every schedule proof.

    The MMA is the single tcgen05 candidate of one device function, with
    compile-time M, N, K extents and exactly one store, fed by its analyzed
    epilogue chain.
    """

    axes: tuple[int, int, int]
    shape: tuple[int, int, int]
    candidate: _CuteMmaNode
    chain: Tcgen05UnaryEpilogueChain
    output: torch.Tensor
    # A store ring the allocation proofs can price completely: one unfissioned
    # function without atomics, and a fresh, TMA-legal destination
    # written by the plain (non-fragment) epilogue.
    owns_output_ring: bool


def analyze_native_gemm(
    env: CompileEnvironment, device_ir: DeviceIR
) -> NativeGemm | None:
    """Use the same native operand and epilogue proofs as generated code."""
    from ..compile_environment import CompileEnvironment
    from .aux_tensor import _output_tensor_from_store_node
    from .aux_tensor import _tcgen05_aux_detector_mma_nodes
    from .aux_tensor import analyze_tcgen05_matmul_store_chains
    from .cute_mma import _tcgen05_tma_destination_is_legal
    from .cute_mma import analyze_cute_mma_node
    from .promote_output_axis import _fresh_tensors

    state = env.config_spec._cute_tcgen05_config
    host = device_ir.host_function
    shape = state.matmul_compile_time_static_extents
    axes = state.matmul_block_ids
    if (
        not state.search_enabled
        or host is None
        or env.config_spec.target_device_capability not in ((10, 0), (10, 3))
        or shape is None
        or any(type(value) is not int or value <= 0 for value in shape)
        or axes is None
        or not state.matmul_operands_tma_provable
        or state.materialized_matmul_block_ids
        or len(device_ir.root_ids) != 1
        or env.config_spec.reduction_block_ids - {axes[2]}
    ):
        return None
    nodes = _tcgen05_aux_detector_mma_nodes(device_ir.graphs)
    if len(nodes) != 1:
        return None
    (node,) = nodes
    with ExitStack() as stack:
        if CompileEnvironment.has_current():
            assert CompileEnvironment.current() is env
        else:
            stack.enter_context(env)
        stack.enter_context(host)
        candidate = analyze_cute_mma_node(node, device_ir=device_ir)
        if (
            candidate is None
            or candidate.requires_accumulator_seed
            or candidate.operands.output_block_ids
            not in {tuple(ids) for ids in device_ir.grid_block_ids}
        ):
            return None
        stores = analyze_tcgen05_matmul_store_chains(device_ir.graphs, node)
        if stores is None or len(stores) != 1:
            return None
        store, chain = stores[0]
        output = _output_tensor_from_store_node(store)
        # Another output would add a store ring that no allocation proof prices.
        if output is None or has_other_store(device_ir.graphs, store):
            return None
        owns_output_ring = (
            env.cute_fission_plan is None
            and not device_ir.has_atomic_ops()
            and not candidate.requires_fragment_epilogue
            and output in _fresh_tensors(host)
            and _tcgen05_tma_destination_is_legal(
                env, store, output_column_major=candidate.output_column_major
            )
        )
    return NativeGemm(
        axes,
        cast("tuple[int, int, int]", shape),
        candidate,
        chain,
        output,
        owns_output_ring,
    )


def prove_full_tile_schedule_facts(
    gemm: NativeGemm | None,
) -> Tcgen05FullTileScheduleFacts | None:
    """Admit one aligned rank-two MMA with a supported single epilogue."""
    if gemm is None:
        return None
    operands = gemm.candidate.operands
    if (
        operands.has_leading_passthrough
        or (operands.m_block_id, operands.n_block_id, operands.k_block_id) != gemm.axes
        or operands.lhs.source_fake.ndim != 2
        or operands.rhs.source_fake.ndim != 2
        or operands.lhs.source_fake.dtype
        not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
        or operands.rhs.source_fake.dtype != operands.lhs.source_fake.dtype
        or operands.lhs.matrix_major not in ("row", "col")
        or operands.rhs.matrix_major not in ("row", "col")
        or operands.lhs.source_to_logical_order is not None
        or operands.rhs.source_to_logical_order is not None
        or operands.rhs.rhs_is_grouped
        or any(
            leaf.broadcast_axis not in (0, 1, 2)
            for leaf in gemm.chain.auxiliary_tensor_loads
        )
        or gemm.output.ndim != 2
        or gemm.output.dtype not in (torch.float16, torch.bfloat16)
    ):
        return None
    return Tcgen05FullTileScheduleFacts(
        gemm.axes,
        gemm.shape,
        operands.lhs.source_fake.dtype,
        rhs_k_major=operands.rhs.matrix_major == "col",
        lhs_k_major=operands.lhs.matrix_major == "row",
    )
