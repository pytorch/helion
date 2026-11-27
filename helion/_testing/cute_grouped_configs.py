"""Configs, arguments and seed filters that several grouped GEMM test modules share."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import helion

if TYPE_CHECKING:
    from helion.autotuner.config_spec import ConfigSpec


def d_shift_config(orientation: str, *, cluster_n: int = 1) -> helion.Config:
    """A worklist config with full-allocation A on the bounded D shift, its
    worklist in ``orientation`` (``"mn"`` or ``"nm"``)."""
    return helion.Config(
        block_sizes=[256, 128, 64],
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=cluster_n,
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
        tcgen05_grouped_mode="worklist_nm",
        tcgen05_grouped_worklist_orientation=orientation,
        tcgen05_grouped_worklist_mma_n=256,
        tcgen05_grouped_worklist_source_m_tile=128,
        tcgen05_grouped_full_allocation_a=True,
        tcgen05_grouped_d_shift=True,
    )


def grid_multiplier_config(multiplier: int) -> helion.Config:
    """A worklist config that launches ``multiplier`` CTAs per SM
    (``num_sm_multiplier``), on 32-row source tiles."""
    return helion.Config(
        block_sizes=[256, 128, 64],
        pid_type="persistent_interleaved",
        num_sm_multiplier=multiplier,
        tcgen05_cluster_m=1,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=5,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
        tcgen05_grouped_mode="worklist_nm",
        tcgen05_grouped_worklist_source_m_tile=32,
    )


def prefix_recipe_args(
    *,
    groups: int = 16,
    m: int = 2048,
    n: int = 1536,
    k: int = 2048,
    dtype: torch.dtype = torch.bfloat16,
    offset_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, ...]:
    """CPU arguments of a grouped GEMM over device offsets that split ``m``
    rows evenly between ``groups``; the defaults lie in the domain of the
    paired prefix recipes."""
    return (
        torch.empty((m, k), dtype=dtype),
        torch.empty((groups, n, k), dtype=dtype),
        torch.arange(groups + 1, dtype=offset_dtype) * (m // groups),
    )


def prefix_recipes(
    spec: ConfigSpec, source_m_tile: int | None = 128
) -> list[helion.Config]:
    """The compiler seeds of ``spec`` that carry a paired prefix recipe
    (``tcgen05_grouped_prefix_adaptive_n``) with ``source_m_tile`` rows per
    source tile; every such seed for ``None``."""
    return [
        seed
        for seed in spec.compiler_seed_configs
        if seed.get("tcgen05_grouped_prefix_adaptive_n") is True
        and (
            source_m_tile is None
            or seed.get("tcgen05_grouped_worklist_source_m_tile") == source_m_tile
        )
    ]
