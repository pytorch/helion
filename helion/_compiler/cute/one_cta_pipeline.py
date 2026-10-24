"""Complete allocation proof for ordinary one-CTA FP8 GEMM pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .pipeline_smem import Tcgen05PipelineSmemFacts
from .pipeline_smem import deepest_fitting_ab_stages
from .pipeline_smem import tcgen05_bookkeeping_smem_bytes
from .strategies import TCGEN05_LAYOUT_OVERRIDES_KEYS
from .tcgen05_constants import tcgen05_ab_smem_bytes_per_cta
from .tcgen05_constants import tcgen05_c_smem_bytes_per_cta
from .tcgen05_constants import tcgen05_default_epilogue_tile_size
from .tcgen05_schedule_facts import Tcgen05FullTileScheduleFacts
from .tcgen05_schedule_facts import full_tile_schedule_supported

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..compile_environment import CompileEnvironment
    from .tcgen05_schedule_facts import NativeGemm


@dataclass(frozen=True)
class Tcgen05OneCtaPipelineFacts:
    schedule: Tcgen05FullTileScheduleFacts
    allocations: Tcgen05PipelineSmemFacts


def one_cta_pipeline_defaults() -> dict[str, object]:
    return {
        "cute_matmul_family": "tcgen05",
        "tcgen05_cta_group": "auto",
        "tcgen05_cluster_m": 1,
        "tcgen05_cluster_n": 1,
        "tcgen05_strategy": "role_local_monolithic",
        "tcgen05_persistence_model": "static_persistent",
        "tcgen05_layout_strategy": "default",
        "tcgen05_warp_spec_ab_load_warps": 1,
        "tcgen05_warp_spec_mma_warps": 1,
        "tcgen05_warp_spec_epi_load_warps": 0,
        "tcgen05_warp_spec_scheduler_warps": 0,
        "tcgen05_warp_spec_c_input_warps": 0,
        "tcgen05_warp_spec_store_warps": 0,
        "tcgen05_num_epi_warps": 4,
        "tcgen05_aux_load_mode": "simt",
        "tcgen05_aux_load_placement": "post_acc_wait",
        "tcgen05_tvm_ffi_launch": False,
        "tcgen05_flat_role_coordinates": False,
        "tcgen05_epilogue_fanout": "off",
        "tcgen05_epilogue_layout": "normal",
        "cute_materialized_operand_schedule": "off",
        "cute_materialized_schedule": "off",
        "cute_split_k_schedule": "legacy",
    }


def one_cta_pipeline_supported(
    facts: Tcgen05OneCtaPipelineFacts,
    tile: tuple[int, int, int],
    config: Mapping[str, object],
) -> bool:
    return (
        facts.schedule.dtype == torch.float8_e4m3fn
        and facts.allocations.input_dtype_bytes == 1
        and not facts.allocations.output_column_major
        and facts.allocations.output_dtype_bytes in (2, 4)
        and facts.allocations.capacity_bytes > 0
        and full_tile_schedule_supported(facts.schedule, tile, cluster_m=1, cluster_n=1)
        and config.get("pid_type") in ("persistent_blocked", "persistent_interleaved")
        and all(
            config.get(key, value) == value
            for key, value in one_cta_pipeline_defaults().items()
        )
        and all(config.get(key) is None for key in TCGEN05_LAYOUT_OVERRIDES_KEYS)
        and config.get("tcgen05_grouped_mode") is None
        and not config.get("cute_split_k_workspace")
        and config.get("tcgen05_c_store_mode", "normal") in ("normal", "direct")
        and config.get("tcgen05_c_stages", 2) in (2, 4)
        and config.get("tcgen05_acc_stages", 2) in (1, 2)
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
) -> int:
    """Conservative AB + actual-width DEFAULT D + bookkeeping bound.

    Only post-wait SIMT auxiliary loads are admitted, so scales need no SMEM.
    Direct stores retain the D-ring charge. The shared DEFAULT sizing helper
    follows the same with-source convention as the device/descriptor resolver.
    This function does not change any legacy one- or two-CTA allocation model.
    """
    assert bm in (64, 128) and bn in (16, 32, 64, 128, 256)
    assert bk in (32, 64, 128, 256)
    assert facts.input_dtype_bytes == 1 and facts.output_dtype_bytes in (2, 4)
    assert ab_stages > 0 and c_stages in (2, 4) and acc_stages in (1, 2)
    width = facts.output_dtype_bytes * 8
    epi_m, epi_n = tcgen05_default_epilogue_tile_size(
        bm, bn, elem_width_d=width, elem_width_c=width
    )
    return (
        tcgen05_ab_smem_bytes_per_cta(
            bm=bm, bn=bn, bk=bk, dtype_bytes=1, ab_stages=ab_stages, cluster_m=1
        )
        + tcgen05_c_smem_bytes_per_cta(
            epi_tile_m=epi_m,
            epi_tile_n=epi_n,
            dtype_bytes=facts.output_dtype_bytes,
            c_stages=c_stages,
        )
        + tcgen05_bookkeeping_smem_bytes(ab_stages=ab_stages, acc_stages=acc_stages)
    )


def max_one_cta_pipeline_ab_stages(
    facts: Tcgen05PipelineSmemFacts,
    *,
    bm: int,
    bn: int,
    bk: int,
    c_stages: int,
    acc_stages: int,
) -> int:
    return deepest_fitting_ab_stages(
        lambda stages: one_cta_pipeline_smem_bytes(
            facts,
            bm=bm,
            bn=bn,
            bk=bk,
            ab_stages=stages,
            c_stages=c_stages,
            acc_stages=acc_stages,
        ),
        facts.capacity_bytes,
    )


def prove_one_cta_pipeline_facts(
    env: CompileEnvironment, gemm: NativeGemm | None
) -> Tcgen05OneCtaPipelineFacts | None:
    # The allocation analyzer already excludes operand permutations,
    # leading/grouped inputs and unqualified scale loads.
    state = env.config_spec._cute_tcgen05_config
    allocations = state.pipeline_smem_facts
    if (
        gemm is None
        or not gemm.owns_output_ring
        or state.matmul_input_dtype != torch.float8_e4m3fn
        or allocations is None
        or allocations.input_dtype_bytes != 1
        or gemm.output.dtype not in (torch.bfloat16, torch.float32)
        or tuple(gemm.output.shape) != gemm.shape[:2]
        or gemm.candidate.operands.lhs.matrix_major != "row"
    ):
        return None
    return Tcgen05OneCtaPipelineFacts(
        Tcgen05FullTileScheduleFacts(
            gemm.axes,
            gemm.shape,
            torch.float8_e4m3fn,
            rhs_k_major=gemm.candidate.operands.rhs.matrix_major == "col",
        ),
        allocations,
    )
