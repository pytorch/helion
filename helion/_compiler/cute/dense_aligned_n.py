"""Aligned physical N tiles behind ordinary power-of-two public carriers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch

from .one_cta_pipeline import one_cta_pipeline_defaults
from .pipeline_smem import TCGEN05_SMEM_AWARE_MAX_AB_STAGES
from .pipeline_smem import Tcgen05PipelineSmemFacts
from .pipeline_smem import deepest_fitting_ab_stages
from .pipeline_smem import pipeline_smem_bytes
from .strategies import TCGEN05_LAYOUT_OVERRIDES_KEYS
from .tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .dense_gemm import DenseGemmFacts


TILE_N_KEY = "tcgen05_tile_n"
ALIGNED_N_CHOICES = tuple(n for n in range(64, 257, 16) if n & (n - 1))


def carrier_n(physical_n: int) -> int:
    return 1 << (physical_n - 1).bit_length()


def dense_aligned_n_defaults(
    *, acc_stages: int = 2, store_mode: str = "normal"
) -> dict[str, object]:
    assert store_mode in ("normal", "direct")
    return {
        **one_cta_pipeline_defaults(),
        "tcgen05_cta_group": "two",
        "tcgen05_cluster_m": 2,
        "tcgen05_c_store_mode": store_mode,
        "tcgen05_acc_stages": acc_stages,
        "tcgen05_l2_swizzle_size": 1,
        "l2_groupings": [1],
        "loop_orders": [[0, 1]],
        "num_sm_multiplier": 1,
    }


@dataclass(frozen=True)
class DenseAlignedNPlan:
    block_ids: tuple[int, int, int]
    public_tile: tuple[int, int, int]
    physical_tile: tuple[int, int, int]
    max_ab_stages: int


def dense_aligned_n_plan(
    facts: DenseGemmFacts | None,
    config: Mapping[str, object],
    tile: tuple[int, int, int],
    *,
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> DenseAlignedNPlan | None:
    """Reuse the strict ordinary native/fresh multiplicative-epilogue facts.

    DenseGemmFacts describes the native graph in the frontend's coordinates;
    this family leaves every auxiliary expression and coordinate view unchanged.
    """
    width = config.get(TILE_N_KEY, 0)
    if (
        facts is None
        or facts.input_dtype != torch.float8_e4m3fn
        or facts.capacity_bytes <= 0
        or type(width) is not int
        or width not in ALIGNED_N_CHOICES
        or any(type(value) is not int for value in tile)
    ):
        return None
    bm, _bn, bk = tile
    public = (128, carrier_n(width), bk)
    physical = (128, width, bk)
    m, n, k = facts.shape
    c_stages = config.get("tcgen05_c_stages", 2)
    acc_stages = config.get("tcgen05_acc_stages", 2)
    store_mode = config.get("tcgen05_c_store_mode", "normal")
    if (
        bm != 128
        or bk not in (64, 128)
        or tile not in (public, physical)
        or any(
            not low <= value <= high
            for candidate in (public, physical)
            for value, (low, high) in zip(candidate, bounds, strict=True)
        )
        or m % bm
        or n % width
        or k < bk
        or (k + bk - 1) // bk > TCGEN05_TWO_CTA_MAX_K_TILES
        or type(c_stages) is not int
        or c_stages not in (2, 4)
        or type(acc_stages) is not int
        or acc_stages not in (1, 2)
        or store_mode not in ("normal", "direct")
        or config.get("pid_type")
        not in ("persistent_blocked", "persistent_interleaved")
        or any(
            config.get(key, value) != value
            for key, value in dense_aligned_n_defaults(
                acc_stages=acc_stages, store_mode=cast("str", store_mode)
            ).items()
        )
        or any(config.get(key) is not None for key in TCGEN05_LAYOUT_OVERRIDES_KEYS)
        or config.get("tcgen05_grouped_mode") is not None
        or config.get("cute_split_k_workspace")
    ):
        return None
    # The physical tile is the paired M128 FP8 instruction with a 64-row,
    # N-permuted per-CTA BF16 D ring; its allocations are exactly those of
    # the native paired pipeline at this width.
    allocations = Tcgen05PipelineSmemFacts(
        1, 2, facts.capacity_bytes, output_column_major=facts.output_column_major
    )
    maximum = deepest_fitting_ab_stages(
        lambda stages: pipeline_smem_bytes(
            allocations,
            bm=128,
            bn=width,
            bk=bk,
            ab_stages=stages,
            c_stages=c_stages,
            acc_stages=acc_stages,
        ),
        facts.capacity_bytes,
        max_stages=min(TCGEN05_SMEM_AWARE_MAX_AB_STAGES, (k + bk - 1) // bk),
    )
    if maximum == 0:
        return None
    return DenseAlignedNPlan(facts.block_ids, public, physical, maximum)
