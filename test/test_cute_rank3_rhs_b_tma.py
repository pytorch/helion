from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
import pickle
import re
from typing import Any
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs
from helion._compiler.cute.strategies import TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY
from helion._compiler.cute.strategies import TCGEN05_STRATEGY_CONFIG_KEY
from helion._compiler.cute.strategies import Tcgen05PersistenceModel
from helion._compiler.cute.strategies import Tcgen05Strategy
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._compiler.cute.tcgen05_constants import TCGEN05_AB_STAGES_AUTO_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_DIRECT_POINTER_METADATA_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_STATIC_PERSISTENT_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_STATIC_RESERVED_SMS_SEARCH_CHOICES,
)
from helion._compiler.program_id import Tcgen05PersistentProgramIDs
from helion._testing import DEVICE
from helion._testing import matchesBackends
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl
from helion.runtime import _CUTE_LAUNCH_ARG_CACHE_LIMIT
from helion.runtime import _TCGEN05_GROUPED_STATIC_METADATA_CACHE_LIMIT
from helion.runtime import _append_cute_wrapper_plan
from helion.runtime import _build_cached_cute_schema_and_args
from helion.runtime import _build_tcgen05_grouped_static_metadata
from helion.runtime import _cute_launch_arg_cache_key
from helion.runtime import _freeze_cute_wrapper_plans
from helion.runtime import _tcgen05_grouped_static_active_clusters
from helion.runtime import _tcgen05_grouped_static_metadata_cache_key
from helion.runtime import default_cute_launcher

pytestmark = skipUnlessBackends(["cute"])
if matchesBackends(["cute"]):
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")


def _rank3_rhs_tma_config() -> helion.Config:
    return helion.Config(
        block_sizes=[128, 128, 128],
        l2_groupings=[1],
        loop_orders=[[0, 1]],
        num_stages=2,
        num_warps=8,
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=1,
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
    )


def _rank3_rhs_grouped_static_config() -> helion.Config:
    config = _rank3_rhs_tma_config()
    config.config["tcgen05_grouped_static_persistent"] = True
    return config


def _rank3_rhs_grouped_static_default_ab_config() -> helion.Config:
    config = _rank3_rhs_grouped_static_config()
    config.config.pop("tcgen05_ab_stages", None)
    return config


def _rank3_rhs_clustered_tma_config() -> helion.Config:
    return helion.Config(
        block_sizes=[256, 128, 128],
        l2_groupings=[1],
        loop_orders=[[0, 1]],
        num_stages=2,
        num_warps=8,
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=2,
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
    )


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt(
    a: torch.Tensor, b_grouped: torch.Tensor, layout: torch.Tensor
) -> torch.Tensor:
    m, k = a.size()
    _g, n, _k = b_grouped.size()
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k],
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_bad_false_branch(
    a: torch.Tensor, b_grouped: torch.Tensor, layout: torch.Tensor
) -> torch.Tensor:
    m, k = a.size()
    _g, n, _k = b_grouped.size()
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 1)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k],
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_bad_group_n_tile(
    a: torch.Tensor, b_grouped: torch.Tensor, layout: torch.Tensor
) -> torch.Tensor:
    m, k = a.size()
    _g, n, _k = b_grouped.size()
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        group_id = layout[tile_n.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k],
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_bad_rhs_n_m_tile(
    a: torch.Tensor, b_grouped: torch.Tensor, layout: torch.Tensor
) -> torch.Tensor:
    m, k = a.size()
    _g, n, _k = b_grouped.size()
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k],
                b_grouped[safe_group_id, tile_m, tile_k].T,
            )
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_shared_safe_group(
    a: torch.Tensor, b_grouped: torch.Tensor, layout: torch.Tensor
) -> torch.Tensor:
    m, k = a.size()
    _g, n, _k = b_grouped.size()
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        safe_group_zero = (safe_group_id - safe_group_id).to(torch.float32)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k],
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = (acc + safe_group_zero).to(out.dtype)
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_shared_operand_loads(
    a: torch.Tensor, b_grouped: torch.Tensor, layout: torch.Tensor
) -> torch.Tensor:
    m, k = a.size()
    _g, n, _k = b_grouped.size()
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        acc_extra = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            a_tile = a[tile_m, tile_k]
            b_tile = b_grouped[safe_group_id, tile_n, tile_k].T
            acc = torch.addmm(acc, a_tile, b_tile)
            acc_extra = torch.addmm(acc_extra, a_tile, b_tile)
        out[tile_m, tile_n] = (acc + acc_extra).to(out.dtype)
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_with_n_sizes(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    n_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, k = a.size()
    _g, max_n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, max_n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        group_n = n_sizes[safe_group_id]
        valid_cols = tile_n.index < group_n
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k],
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = torch.where(
            valid_cols[None, :],
            acc.to(out.dtype),
            out[tile_m, tile_n],
        )
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_bad_n_sizes_zero_store(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    n_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, k = a.size()
    _g, max_n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, max_n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        group_n = n_sizes[safe_group_id]
        valid_cols = tile_n.index < group_n
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k],
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = torch.where(
            valid_cols[None, :],
            acc.to(out.dtype),
            torch.zeros_like(acc).to(out.dtype),
        )
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_bad_n_sizes_relative_index(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    n_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, k = a.size()
    _g, max_n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, max_n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        group_n = n_sizes[safe_group_id]
        valid_cols = (tile_n.index - tile_n.begin) < group_n
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k],
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = torch.where(
            valid_cols[None, :],
            acc.to(out.dtype),
            out[tile_m, tile_n],
        )
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_with_mn_tails(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    n_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, k = a.size()
    _g, max_n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, max_n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        row_group_ids = layout[tile_m]
        valid_rows = row_group_ids == safe_group_id
        group_n = n_sizes[safe_group_id]
        valid_cols = tile_n.index < group_n
        valid = valid_rows[:, None] & valid_cols[None, :]  # pyrefly: ignore[bad-index]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k],
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = torch.where(
            valid,
            acc.to(out.dtype),
            out[tile_m, tile_n],
        )
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_with_renamed_mn_tails(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    groups: torch.Tensor,
    cols: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, k = a.size()
    _g, max_n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, max_n]):
        group_id = groups[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        row_group_ids = groups[tile_m]
        valid_rows = row_group_ids == safe_group_id
        group_n = cols[safe_group_id]
        valid_cols = tile_n.index < group_n
        valid = valid_rows[:, None] & valid_cols[None, :]  # pyrefly: ignore[bad-index]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k],
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = torch.where(
            valid,
            acc.to(out.dtype),
            out[tile_m, tile_n],
        )
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_with_m_tail(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, k = a.size()
    _g, n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        row_group_ids = layout[tile_m]
        valid_rows = row_group_ids == safe_group_id
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k],
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        valid = valid_rows[:, None]  # pyrefly: ignore[bad-index]
        out[tile_m, tile_n] = torch.where(
            valid,
            acc.to(out.dtype),
            out[tile_m, tile_n],
        )
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_with_k_sizes(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    k_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, max_k = a.size()
    _g, n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        group_k = k_sizes[safe_group_id]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(max_k):
            valid_k = tile_k.index < group_k
            a_tile = a[tile_m, tile_k]
            b_tile = b_grouped[safe_group_id, tile_n, tile_k]
            masked_a = torch.where(valid_k[None, :], a_tile, torch.zeros_like(a_tile))
            masked_b = torch.where(valid_k[None, :], b_tile, torch.zeros_like(b_tile))
            acc = torch.addmm(acc, masked_a, masked_b.T)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    n_sizes: torch.Tensor,
    k_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, max_k = a.size()
    _g, max_n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, max_n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        row_group_ids = layout[tile_m]
        valid_rows = row_group_ids == safe_group_id
        group_n = n_sizes[safe_group_id]
        valid_cols = tile_n.index < group_n
        valid = valid_rows[:, None] & valid_cols[None, :]  # pyrefly: ignore[bad-index]
        group_k = k_sizes[safe_group_id]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(max_k):
            valid_k = tile_k.index < group_k
            a_tile = a[tile_m, tile_k]
            b_tile = b_grouped[safe_group_id, tile_n, tile_k]
            masked_a = torch.where(valid_k[None, :], a_tile, torch.zeros_like(a_tile))
            masked_b = torch.where(valid_k[None, :], b_tile, torch.zeros_like(b_tile))
            acc = torch.addmm(acc, masked_a, masked_b.T)
        out[tile_m, tile_n] = torch.where(
            valid,
            acc.to(out.dtype),
            out[tile_m, tile_n],
        )
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_bad_k_sizes_missing_b_mask(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    k_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, max_k = a.size()
    _g, n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        group_k = k_sizes[safe_group_id]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(max_k):
            valid_k = tile_k.index < group_k
            a_tile = a[tile_m, tile_k]
            masked_a = torch.where(valid_k[None, :], a_tile, torch.zeros_like(a_tile))
            acc = torch.addmm(
                acc,
                masked_a,
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_bad_k_sizes_arbitrary_mask(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    k_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, max_k = a.size()
    _g, n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        group_k = k_sizes[safe_group_id]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(max_k):
            valid_k = tile_k.index <= group_k
            a_tile = a[tile_m, tile_k]
            b_tile = b_grouped[safe_group_id, tile_n, tile_k]
            masked_a = torch.where(valid_k[None, :], a_tile, torch.zeros_like(a_tile))
            masked_b = torch.where(valid_k[None, :], b_tile, torch.zeros_like(b_tile))
            acc = torch.addmm(acc, masked_a, masked_b.T)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_bad_k_sizes_without_source_proof(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    k_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, max_k = a.size()
    _g, n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        group_k = k_sizes[safe_group_id]
        k_noop = (group_k - group_k).to(torch.float32)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(max_k):
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k],
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = (acc + k_noop).to(out.dtype)
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_bad_k_sizes_group_provenance(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    k_layout: torch.Tensor,
    k_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, max_k = a.size()
    _g, n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        k_group_id = k_layout[tile_m.begin]
        k_safe_group_id = torch.where(k_group_id >= 0, k_group_id, 0)
        group_k = k_sizes[k_safe_group_id]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(max_k):
            valid_k = tile_k.index < group_k
            a_tile = a[tile_m, tile_k]
            b_tile = b_grouped[safe_group_id, tile_n, tile_k]
            masked_a = torch.where(valid_k[None, :], a_tile, torch.zeros_like(a_tile))
            masked_b = torch.where(valid_k[None, :], b_tile, torch.zeros_like(b_tile))
            acc = torch.addmm(acc, masked_a, masked_b.T)
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_bad_k_sizes_extra_loop_use(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    k_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, max_k = a.size()
    _g, n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        group_k = k_sizes[safe_group_id]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(max_k):
            valid_k = tile_k.index < group_k
            a_tile = a[tile_m, tile_k]
            b_tile = b_grouped[safe_group_id, tile_n, tile_k]
            masked_a = torch.where(valid_k[None, :], a_tile, torch.zeros_like(a_tile))
            masked_b = torch.where(valid_k[None, :], b_tile, torch.zeros_like(b_tile))
            acc = torch.addmm(acc, masked_a, masked_b.T)

        extra = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for extra_k in hl.tile(max_k):
            extra_valid_k = extra_k.index < group_k
            extra_a = a[tile_m, extra_k]
            extra_b = b_grouped[safe_group_id, tile_n, extra_k]
            zero_a = extra_a - extra_a
            zero_b = extra_b - extra_b
            masked_zero_a = torch.where(extra_valid_k[None, :], zero_a, zero_a)
            masked_zero_b = torch.where(extra_valid_k[None, :], zero_b, zero_b)
            extra = torch.addmm(extra, masked_zero_a, masked_zero_b.T)

        out[tile_m, tile_n] = (acc + extra).to(out.dtype)
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_bad_mn_tails_zero_store(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    n_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, k = a.size()
    _g, max_n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, max_n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        row_group_ids = layout[tile_m]
        valid_rows = row_group_ids == safe_group_id
        group_n = n_sizes[safe_group_id]
        valid_cols = tile_n.index < group_n
        valid = valid_rows[:, None] & valid_cols[None, :]  # pyrefly: ignore[bad-index]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k],
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = torch.where(
            valid,
            acc.to(out.dtype),
            torch.zeros_like(acc).to(out.dtype),
        )
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_bad_m_tail_relative_index(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    n_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, k = a.size()
    _g, max_n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, max_n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        row_group_ids = layout[tile_m.index - tile_m.begin]
        valid_rows = row_group_ids == safe_group_id
        group_n = n_sizes[safe_group_id]
        valid_cols = tile_n.index < group_n
        valid = valid_rows[:, None] & valid_cols[None, :]  # pyrefly: ignore[bad-index]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k],
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = torch.where(
            valid,
            acc.to(out.dtype),
            out[tile_m, tile_n],
        )
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_bad_m_tail_arbitrary_row_mask(
    a: torch.Tensor,
    b_grouped: torch.Tensor,
    layout: torch.Tensor,
    n_sizes: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    m, k = a.size()
    _g, max_n, _k = b_grouped.size()
    for tile_m, tile_n in hl.tile([m, max_n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        row_group_ids = layout[tile_m]
        valid_rows = row_group_ids >= 0
        group_n = n_sizes[safe_group_id]
        valid_cols = tile_n.index < group_n
        valid = valid_rows[:, None] & valid_cols[None, :]  # pyrefly: ignore[bad-index]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k],
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = torch.where(
            valid,
            acc.to(out.dtype),
            out[tile_m, tile_n],
        )
    return out


@helion.kernel(backend="cute")
def _rank3_rhs_grouped_nt_nonzero_acc(
    a: torch.Tensor, b_grouped: torch.Tensor, layout: torch.Tensor
) -> torch.Tensor:
    m, k = a.size()
    _g, n, _k = b_grouped.size()
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        group_id = layout[tile_m.begin]
        safe_group_id = torch.where(group_id >= 0, group_id, 0)
        acc = hl.full([tile_m, tile_n], 1.0, dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k],
                b_grouped[safe_group_id, tile_n, tile_k].T,
            )
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def _make_args() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n, k, groups = 256, 256, 128, 4
    row_ids = torch.arange(m, device=DEVICE)
    k_ids = row_ids % k
    a = torch.zeros((m, k), device=DEVICE, dtype=torch.bfloat16)
    a[row_ids, k_ids] = 1
    group = torch.arange(groups, device=DEVICE, dtype=torch.float32)[:, None, None]
    col = torch.arange(n, device=DEVICE, dtype=torch.float32)[None, :, None]
    kk = torch.arange(k, device=DEVICE, dtype=torch.float32)[None, None, :]
    b_grouped = (group * 37.0 + (col % 17) * 1.25 + (kk % 13) * 0.125).to(
        torch.bfloat16
    )
    layout = torch.empty((m,), device=DEVICE, dtype=torch.int64)
    layout[:128] = 1
    layout[128:] = 3
    return a, b_grouped, layout


def _make_all_full_grouped_args(
    *,
    groups: int = 4,
    m_per_group: int = 128,
    n: int = 256,
    k: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = groups * m_per_group
    row_ids = torch.arange(m, device=DEVICE)
    k_ids = row_ids % k
    a = torch.zeros((m, k), device=DEVICE, dtype=dtype)
    a[row_ids, k_ids] = 1
    group = torch.arange(groups, device=DEVICE, dtype=torch.float32)[:, None, None]
    col = torch.arange(n, device=DEVICE, dtype=torch.float32)[None, :, None]
    kk = torch.arange(k, device=DEVICE, dtype=torch.float32)[None, None, :]
    b_grouped = (group * 37.0 + (col % 17) * 1.25 + (kk % 13) * 0.125).to(dtype)
    layout = torch.arange(groups, device=DEVICE, dtype=torch.int64).repeat_interleave(
        m_per_group
    )
    return a, b_grouped, layout


def _make_dense_all_full_grouped_args(
    *,
    groups: int = 4,
    m_per_group: int = 128,
    n: int = 256,
    k: int = 128,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b_grouped, layout = _make_all_full_grouped_args(
        groups=groups,
        m_per_group=m_per_group,
        n=n,
        k=k,
        dtype=dtype,
    )
    a_values = torch.arange(a.numel(), device=DEVICE, dtype=torch.float32).reshape(
        a.shape
    )
    b_values = torch.arange(
        b_grouped.numel(), device=DEVICE, dtype=torch.float32
    ).reshape(b_grouped.shape)
    a.copy_(((a_values % 17) - 8) / 17)
    b_grouped.copy_(((b_values % 19) - 9) / 19)
    return a, b_grouped, layout


def _make_nvidia_default_like_n_sizes_args() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    m_sizes = [128, 512, 128]
    n_sizes = [128, 128, 256]
    k = 128
    max_n = max(n_sizes)
    a = torch.randn((sum(m_sizes), k), device=DEVICE, dtype=torch.float16)
    b_grouped = torch.randn(
        (len(m_sizes), max_n, k),
        device=DEVICE,
        dtype=torch.float16,
    )
    layout = torch.cat(
        [
            torch.full((m_size,), group, device=DEVICE, dtype=torch.int32)
            for group, m_size in enumerate(m_sizes)
        ]
    )
    n_sizes_tensor = torch.tensor(n_sizes, device=DEVICE, dtype=torch.int32)
    out = torch.full(
        (sum(m_sizes), max_n),
        -77.0,
        device=DEVICE,
        dtype=torch.float16,
    )
    return a, b_grouped, layout, n_sizes_tensor, out


def _make_mn_tail_grouped_args(
    *,
    k: int = 128,
    dtype: torch.dtype = torch.float16,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    padded_m = 256
    max_n = 192
    m_sizes = [128, 16]
    n_sizes = [192, 160]
    a = torch.randn((padded_m, k), device=DEVICE, dtype=dtype)
    b_grouped = torch.randn(
        (len(m_sizes), max_n, k),
        device=DEVICE,
        dtype=dtype,
    )
    layout = torch.full((padded_m,), -1, device=DEVICE, dtype=torch.int32)
    cursor = 0
    for group_idx, m_size in enumerate(m_sizes):
        layout[cursor : cursor + m_size] = group_idx
        cursor += 128
    n_sizes_tensor = torch.tensor(n_sizes, device=DEVICE, dtype=torch.int32)
    out = torch.full((padded_m, max_n), -77.0, device=DEVICE, dtype=dtype)
    return a, b_grouped, layout, n_sizes_tensor, out


def _make_single_padded_n_tail_args(
    *,
    m: int = 640,
    n: int = 160,
    max_n: int = 192,
    k: int = 16,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.randn((m, k), device=DEVICE, dtype=dtype)
    b_grouped = torch.randn((1, max_n, k), device=DEVICE, dtype=dtype)
    layout = torch.zeros((m,), device=DEVICE, dtype=torch.int32)
    n_sizes_tensor = torch.tensor([n], device=DEVICE, dtype=torch.int32)
    out = torch.full((m, max_n), -77.0, device=DEVICE, dtype=dtype)
    return a, b_grouped, layout, n_sizes_tensor, out


def _set_single_group_m_tail_layout(
    layout: torch.Tensor,
    *,
    valid_m: int = 16,
) -> None:
    layout.fill_(-1)
    layout[:valid_m] = 0


def _set_single_group_no_m_tail_layout(layout: torch.Tensor) -> None:
    layout.fill_(0)


def _make_multi_group_n_tail_args(
    *,
    k: int = 16,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m_sizes = [128, 256, 128]
    n_sizes = [192, 160, 64]
    max_n = 192
    a = torch.randn((sum(m_sizes), k), device=DEVICE, dtype=dtype)
    b_grouped = torch.randn(
        (len(m_sizes), max_n, k),
        device=DEVICE,
        dtype=dtype,
    )
    layout = torch.cat(
        [
            torch.full((m_size,), group, device=DEVICE, dtype=torch.int32)
            for group, m_size in enumerate(m_sizes)
        ]
    )
    n_sizes_tensor = torch.tensor(n_sizes, device=DEVICE, dtype=torch.int32)
    out = torch.full((sum(m_sizes), max_n), -77.0, device=DEVICE, dtype=dtype)
    return a, b_grouped, layout, n_sizes_tensor, out


def _make_m_tail_grouped_args(
    *,
    m_sizes: tuple[int, ...] = (128, 16),
    k: int = 128,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    padded_m = len(m_sizes) * 128
    n = 128
    a = torch.randn((padded_m, k), device=DEVICE, dtype=dtype)
    b_grouped = torch.randn((len(m_sizes), n, k), device=DEVICE, dtype=dtype)
    layout = torch.full((padded_m,), -1, device=DEVICE, dtype=torch.int32)
    cursor = 0
    for group_idx, m_size in enumerate(m_sizes):
        layout[cursor : cursor + m_size] = group_idx
        cursor += 128
    out = torch.full((padded_m, n), -77.0, device=DEVICE, dtype=dtype)
    return a, b_grouped, layout, out


def _make_per_group_k_args(
    *,
    k: int = 32,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = 2
    m_per_group = 128
    n = 128
    a, b_grouped, layout = _make_dense_all_full_grouped_args(
        groups=groups,
        m_per_group=m_per_group,
        n=n,
        k=k,
        dtype=dtype,
    )
    k_sizes = torch.tensor((k // 2, k), device=DEVICE, dtype=torch.int32)
    out = torch.empty((groups * m_per_group, n), device=DEVICE, dtype=dtype)
    return a, b_grouped, layout, k_sizes, out


def _make_mismatched_k_mask_group_args(
    *,
    k: int = 32,
    dtype: torch.dtype = torch.float16,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    a, b_grouped, layout, k_sizes, out = _make_per_group_k_args(k=k, dtype=dtype)
    k_layout = (1 - layout).contiguous()
    return a, b_grouped, layout, k_layout, k_sizes, out


def _make_per_group_k64_args() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return _make_per_group_k_args(k=64)


def _make_mismatched_k_mask_group_k64_args() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return _make_mismatched_k_mask_group_args(k=64)


def _make_documented_mixed_k_args(
    *,
    dtype: torch.dtype = torch.float16,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    k_size_values = (32, 1536, 16, 16)
    m_size_values = (128, 16, 128, 16)
    n_size_values = (64, 128, 64, 64)
    groups = len(k_size_values)
    padded_m = groups * 128
    max_n = 128
    max_k = max(k_size_values)

    a_values = torch.arange(
        padded_m * max_k,
        device=DEVICE,
        dtype=torch.float32,
    ).reshape(padded_m, max_k)
    b_values = torch.arange(
        groups * max_n * max_k,
        device=DEVICE,
        dtype=torch.float32,
    ).reshape(groups, max_n, max_k)
    a = (((a_values % 23) - 11) / 257).to(dtype)
    b_grouped = (((b_values % 29) - 14) / 257).to(dtype)

    layout = torch.full((padded_m,), -1, device=DEVICE, dtype=torch.int32)
    for group_idx, (m_size, group_k) in enumerate(
        zip(m_size_values, k_size_values, strict=True)
    ):
        row_start = group_idx * 128
        rows = slice(row_start, row_start + m_size)
        layout[rows] = group_idx
        if group_k < max_k:
            a[rows, group_k:] = 3.0 + group_idx
            b_grouped[group_idx, :, group_k:] = -2.0 - group_idx

    n_sizes = torch.tensor(n_size_values, device=DEVICE, dtype=torch.int32)
    k_sizes = torch.tensor(k_size_values, device=DEVICE, dtype=torch.int32)
    out = torch.full((padded_m, max_n), -77.0, device=DEVICE, dtype=dtype)
    return a, b_grouped, layout, n_sizes, k_sizes, out


def _make_no_mn_tail_mixed_k_args(
    *,
    dtype: torch.dtype = torch.float16,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    shapes = (
        (256, 128, 64),
        (128, 64, 96),
        (128, 192, 192),
    )
    m_size_values = [shape[0] for shape in shapes]
    n_size_values = [shape[1] for shape in shapes]
    k_size_values = [shape[2] for shape in shapes]
    padded_m = sum(m_size_values)
    max_n = max(n_size_values)
    max_k = max(k_size_values)
    groups = len(shapes)

    a = torch.empty((padded_m, max_k), device=DEVICE, dtype=dtype)
    b_grouped = torch.empty((groups, max_n, max_k), device=DEVICE, dtype=dtype)
    layout = torch.empty((padded_m,), device=DEVICE, dtype=torch.int32)
    cursor = 0
    for group_idx, group_m in enumerate(m_size_values):
        layout[cursor : cursor + group_m] = group_idx
        cursor += group_m
    n_sizes = torch.tensor(n_size_values, device=DEVICE, dtype=torch.int32)
    k_sizes = torch.tensor(k_size_values, device=DEVICE, dtype=torch.int32)
    out = torch.full((padded_m, max_n), -77.0, device=DEVICE, dtype=dtype)
    return a, b_grouped, layout, n_sizes, k_sizes, out


def _make_blackwell_mixed_k_args(
    *,
    dtype: torch.dtype = torch.float16,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    shapes = (
        (8192, 1280, 32),
        (16, 384, 1536),
        (640, 1280, 16),
        (640, 160, 16),
    )
    block_m = 128
    m_size_values = [shape[0] for shape in shapes]
    n_size_values = [shape[1] for shape in shapes]
    k_size_values = [shape[2] for shape in shapes]
    aligned_m = [
        ((m_size + block_m - 1) // block_m) * block_m for m_size in m_size_values
    ]
    padded_m = sum(aligned_m)
    max_n = max(n_size_values)
    max_k = max(k_size_values)
    groups = len(shapes)

    a = torch.empty((padded_m, max_k), device=DEVICE, dtype=dtype)
    b_grouped = torch.empty((groups, max_n, max_k), device=DEVICE, dtype=dtype)
    layout = torch.empty((padded_m,), device=DEVICE, dtype=torch.int32)
    n_sizes = torch.empty((groups,), device=DEVICE, dtype=torch.int32)
    k_sizes = torch.empty((groups,), device=DEVICE, dtype=torch.int32)
    out = torch.empty((padded_m, max_n), device=DEVICE, dtype=dtype)
    return a, b_grouped, layout, n_sizes, k_sizes, out


def _make_mismatched_common_k_args(
    *,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b_grouped, layout = _make_dense_all_full_grouped_args(
        groups=2,
        m_per_group=128,
        n=128,
        k=64,
        dtype=dtype,
    )
    return a[:, :32].contiguous(), b_grouped, layout


def _make_documented_mixed_k_non_row_major_a_args() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    args = list(_make_documented_mixed_k_args())
    a = args[0]
    strided_a = torch.empty(
        (a.size(1), a.size(0)),
        device=DEVICE,
        dtype=a.dtype,
    ).T
    strided_a.copy_(a)
    assert strided_a.shape == a.shape
    assert strided_a.stride(1) != 1
    args[0] = strided_a
    return tuple(args)  # pyrefly: ignore[bad-return]


def _make_documented_mixed_k_non_k_contiguous_b_args() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    args = list(_make_documented_mixed_k_args())
    b_grouped = args[1]
    strided_b = torch.empty(
        (b_grouped.size(0), b_grouped.size(2), b_grouped.size(1)),
        device=DEVICE,
        dtype=b_grouped.dtype,
    ).transpose(1, 2)
    strided_b.copy_(b_grouped)
    assert strided_b.shape == b_grouped.shape
    assert strided_b.stride(2) != 1
    args[1] = strided_b
    return tuple(args)  # pyrefly: ignore[bad-return]


def _assert_grouped_matmul_close(
    out: torch.Tensor,
    args: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    a, b_grouped, layout = args
    expected = torch.empty_like(out)
    for group_idx in range(b_grouped.size(0)):
        rows = torch.nonzero(layout == group_idx, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        expected[rows] = (a[rows].float() @ b_grouped[group_idx].float().T).to(
            out.dtype
        )
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


def _assert_grouped_n_sizes_matmul_and_sentinel(
    out: torch.Tensor,
    args: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> None:
    a, b_grouped, layout, n_sizes, _out_arg = args
    for group_idx in range(b_grouped.size(0)):
        rows = torch.nonzero(layout == group_idx, as_tuple=False).flatten()
        group_n = int(n_sizes[group_idx].item())
        expected = (a[rows].float() @ b_grouped[group_idx, :group_n].float().T).to(
            out.dtype
        )
        torch.testing.assert_close(out[rows, :group_n], expected, rtol=2e-2, atol=2e-2)
        if group_n < out.size(1):
            torch.testing.assert_close(
                out[rows, group_n:],
                torch.full_like(out[rows, group_n:], -77.0),
                rtol=0,
                atol=0,
            )


def _assert_grouped_mn_tail_matmul_and_sentinel(
    out: torch.Tensor,
    args: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> None:
    a, b_grouped, layout, n_sizes, _out_arg = args
    sentinel = torch.full_like(out, -77.0)
    for group_idx in range(b_grouped.size(0)):
        rows = torch.nonzero(layout == group_idx, as_tuple=False).flatten()
        group_n = int(n_sizes[group_idx].item())
        expected = (a[rows].float() @ b_grouped[group_idx, :group_n].float().T).to(
            out.dtype
        )
        torch.testing.assert_close(out[rows, :group_n], expected, rtol=2e-2, atol=2e-2)
        if group_n < out.size(1):
            torch.testing.assert_close(
                out[rows, group_n:],
                sentinel[rows, group_n:],
                rtol=0,
                atol=0,
            )
    invalid_rows = torch.nonzero(layout < 0, as_tuple=False).flatten()
    torch.testing.assert_close(
        out[invalid_rows], sentinel[invalid_rows], rtol=0, atol=0
    )


def _assert_grouped_mn_tail_k_sizes_matmul_and_sentinel(
    out: torch.Tensor,
    args: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    *,
    rtol: float = 3e-2,
    atol: float = 3e-2,
) -> None:
    a, b_grouped, layout, n_sizes, k_sizes, _out_arg = args
    sentinel = torch.full_like(out, -77.0)
    for group_idx in range(b_grouped.size(0)):
        rows = torch.nonzero(layout == group_idx, as_tuple=False).flatten()
        group_n = int(n_sizes[group_idx].item())
        group_k = int(k_sizes[group_idx].item())
        expected = (a[rows, :group_k] @ b_grouped[group_idx, :group_n, :group_k].T).to(
            out.dtype
        )
        torch.testing.assert_close(
            out[rows, :group_n],
            expected,
            rtol=rtol,
            atol=atol,
        )
        if group_n < out.size(1):
            torch.testing.assert_close(
                out[rows, group_n:],
                sentinel[rows, group_n:],
                rtol=0,
                atol=0,
            )
    invalid_rows = torch.nonzero(layout < 0, as_tuple=False).flatten()
    torch.testing.assert_close(
        out[invalid_rows], sentinel[invalid_rows], rtol=0, atol=0
    )


def _assert_grouped_m_tail_matmul_and_sentinel(
    out: torch.Tensor,
    args: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    a, b_grouped, layout, _out_arg = args
    sentinel = torch.full_like(out, -77.0)
    for group_idx in range(b_grouped.size(0)):
        rows = torch.nonzero(layout == group_idx, as_tuple=False).flatten()
        expected = (a[rows].float() @ b_grouped[group_idx].float().T).to(out.dtype)
        torch.testing.assert_close(out[rows], expected, rtol=2e-2, atol=2e-2)
    invalid_rows = torch.nonzero(layout < 0, as_tuple=False).flatten()
    torch.testing.assert_close(
        out[invalid_rows], sentinel[invalid_rows], rtol=0, atol=0
    )


def _grouped_static_metadata_test_plan(
    *,
    layout_idx: int = 0,
    n_sizes_idx: int | None = 1,
    k_sizes_idx: int | None = None,
    suffix: str = "0",
) -> dict[str, object]:
    plan: dict[str, object] = {
        "kind": "tcgen05_grouped_static_persistent",
        "layout_idx": layout_idx,
        "group_count": 2,
        "bm": 128,
        "bn": 64,
        "bk": 128,
        "n_size": 256,
        "k_total_size": 128,
        "problem_sizes_arg": f"tcgen05_grouped_problem_sizes_{suffix}",
        "starts_arg": f"tcgen05_grouped_starts_{suffix}",
        "total_clusters_arg": f"tcgen05_grouped_total_clusters_{suffix}",
    }
    if n_sizes_idx is not None:
        plan["n_sizes_idx"] = n_sizes_idx
    if k_sizes_idx is not None:
        plan["k_sizes_idx"] = k_sizes_idx
    return plan


def test_grouped_static_reserved_sms_active_cluster_clamp() -> None:
    assert (
        _tcgen05_grouped_static_active_clusters(
            num_sm=148,
            cluster_m=1,
            reserved_sms=4,
        )
        == 144
    )
    assert (
        _tcgen05_grouped_static_active_clusters(
            num_sm=148,
            cluster_m=2,
            reserved_sms=4,
        )
        == 72
    )
    assert (
        _tcgen05_grouped_static_active_clusters(
            num_sm=148,
            cluster_m=1,
            reserved_sms=200,
        )
        == 1
    )


def test_grouped_static_wrapper_plan_uses_reserved_sms_cap() -> None:
    plan = _grouped_static_metadata_test_plan()
    plan[TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY] = 4
    plan["sched_params_arg"] = "tcgen05_grouped_sched_params_0"
    body: list[str] = []
    call_args: list[str] = []

    _append_cute_wrapper_plan(body, call_args, plan, num_sm=148)

    wrapper = "\n".join(body)
    assert "cutlass.Int32(144)" in wrapper
    assert "StaticPersistentGroupTileScheduler.get_grid_shape" in wrapper
    assert call_args == ["tcgen05_grouped_sched_params_0"]


def _make_grouped_static_metadata_args(
    m_sizes: tuple[int, int],
    n_sizes: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    layout = torch.cat(
        [
            torch.full((m_size,), group, device=DEVICE, dtype=torch.int32)
            for group, m_size in enumerate(m_sizes)
        ]
    )
    n_sizes_tensor = torch.tensor(n_sizes, device=DEVICE, dtype=torch.int32)
    return layout, n_sizes_tensor


def test_grouped_static_worklist_metadata_uses_pseudo_rows_and_real_groups() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("worklist metadata test needs CUDA")

    plan = {
        "kind": "tcgen05_grouped_static_persistent",
        "layout_idx": 2,
        "lhs_idx": 0,
        "rhs_idx": 1,
        "group_count": 3,
        "bm": 128,
        "bn": 64,
        "bk": 64,
        "n_size": 128,
        "k_total_size": 64,
        "problem_sizes_arg": "tcgen05_grouped_problem_sizes_0",
        "starts_arg": "tcgen05_grouped_starts_0",
        "real_groups_arg": "tcgen05_grouped_real_groups_0",
        "total_clusters_arg": "tcgen05_grouped_total_clusters_0",
        "dynamic_ab_tensormaps": True,
        "dynamic_d_tensormap": True,
        "worklist_metadata": True,
    }
    a = torch.empty((512, 64), device=DEVICE, dtype=torch.bfloat16)
    b = torch.empty((4, 128, 64), device=DEVICE, dtype=torch.bfloat16)
    worklist = torch.tensor(
        [[2, 0, 128, 0], [2, 128, 17, 0], [1, 384, 64, 0]],
        device=DEVICE,
        dtype=torch.int32,
    )

    problem_sizes, starts, real_groups, total_clusters = (
        _build_tcgen05_grouped_static_metadata(
            _FakeCuteKernel(), plan, (a, b, worklist)
        )
    )

    assert problem_sizes.detach().cpu().tolist() == [
        [128, 128, 64, 1],
        [17, 128, 64, 1],
        [64, 128, 64, 1],
    ]
    assert starts.detach().cpu().tolist() == [0, 128, 384]
    assert real_groups.detach().cpu().tolist() == [2, 2, 1]
    assert total_clusters == 6


class _FakeCuteKernel:
    _helion_cute_launch_arg_cache: Any
    _helion_cute_last_launch_cache: Any
    _helion_cute_wrapper_plans: Any
    _helion_tcgen05_grouped_static_metadata_cache: Any


def test_grouped_static_direct_pointer_metadata_builds_pointer_and_stride_tensors() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static metadata test needs CUDA")

    args = _make_documented_mixed_k_args()
    a, b_grouped, _layout, _n_sizes, _k_sizes, out = args
    plan = {
        "kind": "tcgen05_grouped_static_persistent",
        "layout_idx": 2,
        "n_sizes_idx": 3,
        "k_sizes_idx": 4,
        "lhs_idx": 0,
        "rhs_idx": 1,
        "group_count": 4,
        "bm": 128,
        "bn": 64,
        "bk": 64,
        "n_size": 128,
        "k_total_size": 1536,
        "problem_sizes_arg": "tcgen05_grouped_problem_sizes",
        "starts_arg": "tcgen05_grouped_starts",
        "direct_pointers_arg": "tcgen05_grouped_direct_pointers",
        "direct_strides_arg": "tcgen05_grouped_direct_strides",
        "total_clusters_arg": "tcgen05_grouped_total_clusters",
        "dynamic_ab_tensormaps": True,
        "dynamic_d_tensormap": True,
        "direct_pointer_metadata": True,
        "m_tail_preserve": True,
        "n_tail_preserve": True,
    }
    kernel = _FakeCuteKernel()
    kernel._helion_cute_wrapper_plans = (
        plan,
        {"kind": "tcgen05_d_tma", "d_idx": 5, "rank3_mnl_tensor": True},
    )

    problem_sizes, starts, direct_pointers, direct_strides, total_clusters = (
        _build_tcgen05_grouped_static_metadata(kernel, plan, args)
    )

    assert problem_sizes.detach().cpu().tolist() == [
        [128, 64, 32, 1],
        [16, 128, 1536, 1],
        [128, 64, 16, 1],
        [16, 64, 16, 1],
    ]
    assert starts.detach().cpu().tolist() == [0, 128, 256, 384]
    expected_pointers = [
        [
            int(a.data_ptr()) + start * int(a.stride(0)) * a.element_size(),
            int(b_grouped.data_ptr())
            + group * int(b_grouped.stride(0)) * b_grouped.element_size(),
            int(out.data_ptr()) + start * int(out.stride(0)) * out.element_size(),
        ]
        for group, start in enumerate((0, 128, 256, 384))
    ]
    assert direct_pointers.detach().cpu().tolist() == expected_pointers
    expected_strides = [
        [
            [int(a.stride(0)), int(a.stride(1))],
            [int(b_grouped.stride(1)), int(b_grouped.stride(2))],
            [int(out.stride(0)), int(out.stride(1))],
        ]
        for _ in range(4)
    ]
    assert direct_strides.detach().cpu().tolist() == expected_strides
    assert total_clusters == 5


def test_grouped_static_metadata_cache_revalidates_layout_identity() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static metadata cache test needs CUDA")

    plan = _grouped_static_metadata_test_plan()
    kernel = _FakeCuteKernel()
    stale_args = _make_grouped_static_metadata_args((128, 128), (256, 256))
    stale_result = _build_tcgen05_grouped_static_metadata(kernel, plan, stale_args)
    cache = kernel._helion_tcgen05_grouped_static_metadata_cache
    stale_entry = next(iter(cache.values()))

    current_args = _make_grouped_static_metadata_args((256, 128), (256, 256))
    current_key = _tcgen05_grouped_static_metadata_cache_key(
        plan, current_args[0], current_args[1]
    )
    cache.clear()
    cache[current_key] = stale_entry

    result = _build_tcgen05_grouped_static_metadata(kernel, plan, current_args)

    assert stale_result[2] == 8
    assert result[2] == 12
    assert result[1].detach().cpu().tolist() == [0, 256]


def test_grouped_static_metadata_cache_key_distinguishes_tail_kind() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static metadata cache test needs CUDA")

    layout, n_sizes = _make_grouped_static_metadata_args((128, 128), (64, 128))
    base_plan = _grouped_static_metadata_test_plan(n_sizes_idx=None)
    base_plan.update(
        {
            "m_tail_preserve": True,
            "n_tail_preserve": False,
            "dynamic_d_tensormap": True,
            "d_tensormap_tail_store": True,
            "n_size": 128,
            "k_total_size": 1536,
        }
    )
    no_tail_plan = {
        **base_plan,
        "grouped_static_has_m_tail": False,
        "grouped_static_has_n_tail": False,
    }
    m_tail_plan = {
        **base_plan,
        "grouped_static_has_m_tail": True,
        "grouped_static_has_n_tail": False,
    }
    n_tail_plan = {
        **base_plan,
        "n_tail_preserve": True,
        "grouped_static_has_m_tail": False,
        "grouped_static_has_n_tail": True,
    }

    no_tail_key = _tcgen05_grouped_static_metadata_cache_key(no_tail_plan, layout, None)
    m_tail_key = _tcgen05_grouped_static_metadata_cache_key(m_tail_plan, layout, None)
    n_tail_key = _tcgen05_grouped_static_metadata_cache_key(
        n_tail_plan, layout, n_sizes
    )

    assert no_tail_key != m_tail_key
    assert no_tail_key != n_tail_key
    assert m_tail_key != n_tail_key
    assert ("grouped_static_has_m_tail", False) in no_tail_key
    assert ("grouped_static_has_n_tail", False) in no_tail_key
    assert ("grouped_static_has_m_tail", True) in m_tail_key
    assert ("grouped_static_has_n_tail", False) in m_tail_key
    assert ("grouped_static_has_m_tail", False) in n_tail_key
    assert ("grouped_static_has_n_tail", True) in n_tail_key


def test_grouped_static_metadata_cache_key_distinguishes_dynamic_ab_rank() -> None:
    layout = torch.empty((0,), dtype=torch.int32)
    base_plan = _grouped_static_metadata_test_plan(n_sizes_idx=None)
    base_plan["dynamic_ab_tensormaps"] = True
    rank2_plan = {**base_plan, "dynamic_ab_tensormap_rank": 2}
    rank3_plan = {**base_plan, "dynamic_ab_tensormap_rank": 3}

    rank2_key = _tcgen05_grouped_static_metadata_cache_key(rank2_plan, layout, None)
    rank3_key = _tcgen05_grouped_static_metadata_cache_key(rank3_plan, layout, None)

    assert rank2_key != rank3_key
    assert ("dynamic_ab_tensormap_rank", 2) in rank2_key
    assert ("dynamic_ab_tensormap_rank", 3) in rank3_key


def test_grouped_static_metadata_cache_revalidates_n_sizes_identity_during_capture() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static metadata cache test needs CUDA")

    plan = _grouped_static_metadata_test_plan()
    kernel = _FakeCuteKernel()
    layout, stale_n_sizes = _make_grouped_static_metadata_args((128, 128), (64, 128))
    stale_args = (layout, stale_n_sizes)
    stale_result = _build_tcgen05_grouped_static_metadata(kernel, plan, stale_args)
    cache = kernel._helion_tcgen05_grouped_static_metadata_cache
    stale_entry = next(iter(cache.values()))

    current_n_sizes = torch.tensor((128, 256), device=DEVICE, dtype=torch.int32)
    current_args = (layout, current_n_sizes)
    current_key = _tcgen05_grouped_static_metadata_cache_key(
        plan, current_args[0], current_args[1]
    )
    cache.clear()
    cache[current_key] = stale_entry

    with (
        patch("helion.runtime._cuda_graph_capture_active", return_value=True),
        pytest.raises(helion.exc.BackendUnsupported, match="metadata is not cached"),
    ):
        _build_tcgen05_grouped_static_metadata(kernel, plan, current_args)

    result = _build_tcgen05_grouped_static_metadata(kernel, plan, current_args)

    assert stale_result[2] == 3
    assert result[2] == 6


def test_grouped_static_metadata_cache_revalidates_k_sizes_identity_during_capture() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static metadata cache test needs CUDA")

    plan = _grouped_static_metadata_test_plan(k_sizes_idx=2)
    plan["bk"] = 64
    kernel = _FakeCuteKernel()
    layout, n_sizes = _make_grouped_static_metadata_args((128, 128), (64, 128))
    stale_k_sizes = torch.tensor((64, 128), device=DEVICE, dtype=torch.int32)
    stale_args = (layout, n_sizes, stale_k_sizes)
    stale_result = _build_tcgen05_grouped_static_metadata(kernel, plan, stale_args)
    cache = kernel._helion_tcgen05_grouped_static_metadata_cache
    stale_entry = next(iter(cache.values()))

    current_k_sizes = torch.tensor((128, 128), device=DEVICE, dtype=torch.int32)
    current_args = (layout, n_sizes, current_k_sizes)
    current_key = _tcgen05_grouped_static_metadata_cache_key(
        plan,
        current_args[0],
        current_args[1],
        current_args[2],
    )
    cache.clear()
    cache[current_key] = stale_entry

    with (
        patch("helion.runtime._cuda_graph_capture_active", return_value=True),
        pytest.raises(helion.exc.BackendUnsupported, match="metadata is not cached"),
    ):
        _build_tcgen05_grouped_static_metadata(kernel, plan, current_args)

    result = _build_tcgen05_grouped_static_metadata(kernel, plan, current_args)

    assert stale_result[0].detach().cpu().tolist()[0][2] == 64
    assert result[0].detach().cpu().tolist()[0][2] == 128


def test_grouped_static_metadata_cache_rebuilds_after_versioned_value_mutations() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static metadata cache test needs CUDA")

    plan = _grouped_static_metadata_test_plan(k_sizes_idx=2)
    plan["bk"] = 64
    plan["m_tail_preserve"] = True
    kernel = _FakeCuteKernel()
    layout, n_sizes = _make_grouped_static_metadata_args((128, 128), (64, 128))
    k_sizes = torch.tensor((64, 128), device=DEVICE, dtype=torch.int32)
    args = (layout, n_sizes, k_sizes)

    problem_sizes, starts, total_clusters = _build_tcgen05_grouped_static_metadata(
        kernel, plan, args
    )
    assert problem_sizes.detach().cpu().tolist() == [
        [128, 64, 64, 1],
        [128, 128, 128, 1],
    ]
    assert starts.detach().cpu().tolist() == [0, 128]
    assert total_clusters == 3

    layout_version = int(layout._version)
    replacement_layout = torch.cat(
        [
            torch.full((64,), 0, device=DEVICE, dtype=torch.int32),
            torch.full((64,), -1, device=DEVICE, dtype=torch.int32),
            torch.full((128,), 1, device=DEVICE, dtype=torch.int32),
        ]
    )
    layout.copy_(replacement_layout)
    assert int(layout._version) > layout_version
    problem_sizes, starts, total_clusters = _build_tcgen05_grouped_static_metadata(
        kernel, plan, args
    )
    assert problem_sizes.detach().cpu().tolist() == [
        [64, 64, 64, 1],
        [128, 128, 128, 1],
    ]
    assert starts.detach().cpu().tolist() == [0, 128]
    assert total_clusters == 3

    n_sizes_version = int(n_sizes._version)
    n_sizes.fill_(128)
    assert int(n_sizes._version) > n_sizes_version
    problem_sizes, _starts, total_clusters = _build_tcgen05_grouped_static_metadata(
        kernel, plan, args
    )
    assert problem_sizes.detach().cpu().tolist() == [
        [64, 128, 64, 1],
        [128, 128, 128, 1],
    ]
    assert total_clusters == 4

    k_sizes_version = int(k_sizes._version)
    k_sizes[0] = 128
    assert int(k_sizes._version) > k_sizes_version
    problem_sizes, _starts, total_clusters = _build_tcgen05_grouped_static_metadata(
        kernel, plan, args
    )
    assert problem_sizes.detach().cpu().tolist() == [
        [64, 128, 128, 1],
        [128, 128, 128, 1],
    ]
    assert total_clusters == 4


def test_grouped_static_metadata_capture_requires_rewarm_after_versioned_mutation() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static metadata cache test needs CUDA")

    plan = _grouped_static_metadata_test_plan()
    kernel = _FakeCuteKernel()
    layout, n_sizes = _make_grouped_static_metadata_args((128, 128), (64, 128))
    args = (layout, n_sizes)
    warmup_result = _build_tcgen05_grouped_static_metadata(kernel, plan, args)
    assert warmup_result[2] == 3

    n_sizes.copy_(torch.tensor((128, 256), device=DEVICE, dtype=torch.int32))
    with (
        patch("helion.runtime._cuda_graph_capture_active", return_value=True),
        pytest.raises(
            helion.exc.BackendUnsupported,
            match="final grouped metadata values",
        ),
    ):
        _build_tcgen05_grouped_static_metadata(kernel, plan, args)

    rewarm_result = _build_tcgen05_grouped_static_metadata(kernel, plan, args)
    assert rewarm_result[2] == 6
    with patch("helion.runtime._cuda_graph_capture_active", return_value=True):
        capture_result = _build_tcgen05_grouped_static_metadata(kernel, plan, args)

    assert capture_result is rewarm_result


def test_grouped_static_metadata_data_mutation_bypasses_version_contract() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static metadata cache test needs CUDA")

    plan = _grouped_static_metadata_test_plan(k_sizes_idx=2)
    layout, n_sizes = _make_grouped_static_metadata_args((128, 128), (64, 128))
    k_sizes = torch.tensor((64, 128), device=DEVICE, dtype=torch.int32)
    cache_key = _tcgen05_grouped_static_metadata_cache_key(
        plan, layout, n_sizes, k_sizes
    )
    versions = (int(layout._version), int(n_sizes._version), int(k_sizes._version))

    layout.data.copy_(
        torch.cat(
            [
                torch.full((128,), 1, device=DEVICE, dtype=torch.int32),
                torch.full((128,), 0, device=DEVICE, dtype=torch.int32),
            ]
        )
    )
    n_sizes.data.copy_(torch.tensor((128, 256), device=DEVICE, dtype=torch.int32))
    k_sizes.data.fill_(128)
    torch.cuda.synchronize()

    assert layout.detach().cpu().tolist()[:2] == [1, 1]
    assert n_sizes.detach().cpu().tolist() == [128, 256]
    assert k_sizes.detach().cpu().tolist() == [128, 128]
    assert (int(layout._version), int(n_sizes._version), int(k_sizes._version)) == (
        versions
    )
    # This documents the unsupported version-bypass boundary, not stale output.
    assert (
        _tcgen05_grouped_static_metadata_cache_key(plan, layout, n_sizes, k_sizes)
        == cache_key
    )


def test_grouped_static_metadata_mn_tails_use_ceil_clusters() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static metadata test needs CUDA")

    plan = _grouped_static_metadata_test_plan()
    plan.update(
        {
            "n_size": 192,
            "m_tail_preserve": True,
            "n_tail_preserve": True,
        }
    )
    kernel = _FakeCuteKernel()
    layout = torch.full((256,), -1, device=DEVICE, dtype=torch.int32)
    layout[:128] = 0
    layout[128:144] = 1
    n_sizes = torch.tensor((128, 160), device=DEVICE, dtype=torch.int32)

    problem_sizes, starts, total_clusters = _build_tcgen05_grouped_static_metadata(
        kernel, plan, (layout, n_sizes)
    )

    assert problem_sizes.detach().cpu().tolist() == [
        [128, 128, 128, 1],
        [16, 160, 128, 1],
    ]
    assert starts.detach().cpu().tolist() == [0, 128]
    assert total_clusters == 5


def test_grouped_static_metadata_uses_per_group_k_sizes() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static metadata test needs CUDA")

    plan = _grouped_static_metadata_test_plan(
        layout_idx=2,
        n_sizes_idx=3,
        k_sizes_idx=4,
    )
    plan.update(
        {
            "group_count": 4,
            "n_size": 128,
            "k_total_size": 1536,
            "bk": 16,
            "m_tail_preserve": True,
            "n_tail_preserve": True,
        }
    )
    kernel = _FakeCuteKernel()
    args = _make_documented_mixed_k_args()

    problem_sizes, starts, total_clusters = _build_tcgen05_grouped_static_metadata(
        kernel,
        plan,
        args,
    )

    assert problem_sizes.detach().cpu().tolist() == [
        [128, 64, 32, 1],
        [16, 128, 1536, 1],
        [128, 64, 16, 1],
        [16, 64, 16, 1],
    ]
    assert starts.detach().cpu().tolist() == [0, 128, 256, 384]
    assert total_clusters == 5


def test_grouped_static_metadata_dynamic_bk64_relaxes_group_k_to_multiple_of_16() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static metadata test needs CUDA")

    plan = _grouped_static_metadata_test_plan(
        layout_idx=2,
        n_sizes_idx=3,
        k_sizes_idx=4,
    )
    plan.update(
        {
            "group_count": 4,
            "n_size": 128,
            "k_total_size": 1536,
            "bk": 64,
            "m_tail_preserve": True,
            "n_tail_preserve": True,
            "dynamic_ab_tensormaps": True,
            "ab_tensormaps_arg": "tcgen05_grouped_ab_tensormaps",
        }
    )
    kernel = _FakeCuteKernel()
    args = _make_documented_mixed_k_args()

    problem_sizes, starts, total_clusters = _build_tcgen05_grouped_static_metadata(
        kernel,
        plan,
        args,
    )

    assert problem_sizes.detach().cpu().tolist() == [
        [128, 64, 32, 1],
        [16, 128, 1536, 1],
        [128, 64, 16, 1],
        [16, 64, 16, 1],
    ]
    assert starts.detach().cpu().tolist() == [0, 128, 256, 384]
    assert total_clusters == 5

    bad_args = list(args)
    bad_args[4] = args[4].clone()
    bad_args[4][0] = 24
    with pytest.raises(helion.exc.BackendUnsupported, match="multiple of 16"):
        _build_tcgen05_grouped_static_metadata(kernel, plan, tuple(bad_args))

    bad_plan = dict(plan)
    bad_plan["bk"] = 128
    with pytest.raises(helion.exc.BackendUnsupported, match="BK64"):
        _build_tcgen05_grouped_static_metadata(_FakeCuteKernel(), bad_plan, args)


def test_grouped_static_dynamic_bk64_rejects_misaligned_tensormap_base() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static launch cache test needs CUDA")

    plan = _grouped_static_metadata_test_plan(
        layout_idx=2,
        n_sizes_idx=3,
        k_sizes_idx=4,
    )
    plan.update(
        {
            "group_count": 4,
            "n_size": 128,
            "k_total_size": 1536,
            "bk": 64,
            "m_tail_preserve": True,
            "n_tail_preserve": True,
            "dynamic_ab_tensormaps": True,
            "ab_tensormaps_arg": "tcgen05_grouped_ab_tensormaps",
            "lhs_idx": 0,
            "rhs_idx": 1,
        }
    )
    kernel = _FakeCuteKernel()
    kernel._helion_cute_wrapper_plans = [plan]
    args = list(_make_documented_mixed_k_args())
    a = args[0]
    misaligned_storage = torch.empty(
        (a.size(0), a.size(1) + 1),
        device=DEVICE,
        dtype=a.dtype,
    )
    misaligned_a = misaligned_storage[:, 1:]
    misaligned_a.copy_(a)
    assert misaligned_a.shape == a.shape
    assert misaligned_a.data_ptr() % 16 != 0
    args[0] = misaligned_a

    def fake_imports() -> tuple[object, object, object]:
        gmem = object()

        def make_ptr(
            dtype: object,
            data_ptr: int,
            _space: object,
            *,
            assumed_align: int,
        ) -> tuple[str, object, int, int]:
            return ("ptr", dtype, int(data_ptr), assumed_align)

        def current_stream() -> str:
            return "stream"

        return gmem, make_ptr, current_stream

    with (
        patch("helion.runtime._get_cute_launcher_imports", side_effect=fake_imports),
        patch("helion.runtime._torch_dtype_to_cutlass", side_effect=str),
        pytest.raises(helion.exc.BackendUnsupported, match="16-byte-aligned"),
    ):
        _build_cached_cute_schema_and_args(kernel, tuple(args), (1, 1, 1))


def test_grouped_static_dynamic_bk64_requires_launch_cache_before_capture() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static launch cache test needs CUDA")

    plan = _grouped_static_metadata_test_plan(
        layout_idx=2,
        n_sizes_idx=3,
        k_sizes_idx=4,
    )
    plan.update(
        {
            "group_count": 4,
            "n_size": 128,
            "k_total_size": 1536,
            "bk": 64,
            "m_tail_preserve": True,
            "n_tail_preserve": True,
            "dynamic_ab_tensormaps": True,
            "ab_tensormaps_arg": "tcgen05_grouped_ab_tensormaps",
            "lhs_idx": 0,
            "rhs_idx": 1,
        }
    )
    kernel = _FakeCuteKernel()
    kernel._helion_cute_wrapper_plans = [plan]
    args = _make_documented_mixed_k_args()
    _build_tcgen05_grouped_static_metadata(kernel, plan, args)

    def fake_imports() -> tuple[object, object, object]:
        gmem = object()

        def make_ptr(
            dtype: object,
            data_ptr: int,
            _space: object,
            *,
            assumed_align: int,
        ) -> tuple[str, object, int, int]:
            return ("ptr", dtype, int(data_ptr), assumed_align)

        def current_stream() -> str:
            return "stream"

        return gmem, make_ptr, current_stream

    with (
        patch("helion.runtime._get_cute_launcher_imports", side_effect=fake_imports),
        patch("helion.runtime._torch_dtype_to_cutlass", side_effect=str),
        patch("helion.runtime._cuda_graph_capture_active", return_value=True),
        pytest.raises(helion.exc.BackendUnsupported, match="workspace is not cached"),
    ):
        _build_cached_cute_schema_and_args(kernel, args, (1, 1, 1))


def test_grouped_static_metadata_rejects_bad_k_sizes() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static metadata test needs CUDA")

    plan = _grouped_static_metadata_test_plan(
        layout_idx=2,
        n_sizes_idx=3,
        k_sizes_idx=4,
    )
    plan.update(
        {
            "group_count": 4,
            "n_size": 128,
            "k_total_size": 1536,
            "bk": 16,
            "m_tail_preserve": True,
            "n_tail_preserve": True,
        }
    )
    base_args = list(_make_documented_mixed_k_args())
    kernel = _FakeCuteKernel()

    bad_args = list(base_args)
    bad_args[4] = base_args[4][:-1].contiguous()
    with pytest.raises(helion.exc.BackendUnsupported, match="k_sizes length"):
        _build_tcgen05_grouped_static_metadata(kernel, plan, tuple(bad_args))

    bad_args = list(base_args)
    bad_args[4] = base_args[4].clone()
    bad_args[4][0] = 1537
    with pytest.raises(helion.exc.BackendUnsupported, match="per-group K size"):
        _build_tcgen05_grouped_static_metadata(kernel, plan, tuple(bad_args))

    bad_args = list(base_args)
    bad_args[4] = base_args[4].clone()
    bad_args[4][0] = 0
    with pytest.raises(helion.exc.BackendUnsupported, match="positive"):
        _build_tcgen05_grouped_static_metadata(kernel, plan, tuple(bad_args))

    bad_args = list(base_args)
    bad_args[4] = base_args[4].clone()
    bad_args[4][0] = 24
    with pytest.raises(helion.exc.BackendUnsupported, match="CTA K tile"):
        _build_tcgen05_grouped_static_metadata(kernel, plan, tuple(bad_args))


def test_grouped_static_metadata_allows_interior_m_tail_padding() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static metadata test needs CUDA")

    plan = _grouped_static_metadata_test_plan()
    kernel = _FakeCuteKernel()
    layout = torch.full((256,), -1, device=DEVICE, dtype=torch.int32)
    layout[:16] = 0
    layout[128:] = 1
    n_sizes = torch.tensor((128, 128), device=DEVICE, dtype=torch.int32)

    full_m_padding_layout = torch.full((384,), -1, device=DEVICE, dtype=torch.int32)
    full_m_padding_layout[:128] = 0
    full_m_padding_layout[256:] = 1
    overlong_m_padding_layout = torch.full((384,), -1, device=DEVICE, dtype=torch.int32)
    overlong_m_padding_layout[:16] = 0
    overlong_m_padding_layout[256:] = 1
    with pytest.raises(helion.exc.BackendUnsupported, match="ordered complete groups"):
        _build_tcgen05_grouped_static_metadata(
            kernel, plan, (full_m_padding_layout, n_sizes)
        )
    with pytest.raises(helion.exc.BackendUnsupported, match="M size"):
        _build_tcgen05_grouped_static_metadata(kernel, plan, (layout, n_sizes))

    plan = dict(plan)
    plan["m_tail_preserve"] = True
    with pytest.raises(helion.exc.BackendUnsupported, match="M-tail padding"):
        _build_tcgen05_grouped_static_metadata(
            kernel, plan, (overlong_m_padding_layout, n_sizes)
        )

    problem_sizes, starts, total_clusters = _build_tcgen05_grouped_static_metadata(
        kernel, plan, (layout, n_sizes)
    )

    assert problem_sizes.detach().cpu().tolist() == [
        [16, 128, 128, 1],
        [128, 128, 128, 1],
    ]
    assert starts.detach().cpu().tolist() == [0, 128]
    assert total_clusters == 4


def test_grouped_static_metadata_rejects_tails_without_source_proof() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static metadata test needs CUDA")

    plan = _grouped_static_metadata_test_plan()
    plan["n_size"] = 192
    kernel = _FakeCuteKernel()
    layout = torch.full((256,), -1, device=DEVICE, dtype=torch.int32)
    layout[:128] = 0
    layout[128:144] = 1
    n_sizes = torch.tensor((128, 160), device=DEVICE, dtype=torch.int32)

    with pytest.raises(helion.exc.BackendUnsupported, match="M size"):
        _build_tcgen05_grouped_static_metadata(kernel, plan, (layout, n_sizes))

    plan = dict(plan)
    plan["m_tail_preserve"] = True
    with pytest.raises(helion.exc.BackendUnsupported, match="N size"):
        _build_tcgen05_grouped_static_metadata(kernel, plan, (layout, n_sizes))


def test_grouped_static_metadata_cache_keys_tail_proof_flags() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static metadata test needs CUDA")

    proven_plan = _grouped_static_metadata_test_plan()
    proven_plan.update(
        {
            "n_size": 192,
            "m_tail_preserve": True,
            "n_tail_preserve": True,
        }
    )
    unproven_plan = dict(proven_plan)
    unproven_plan["m_tail_preserve"] = False
    unproven_plan["n_tail_preserve"] = False
    kernel = _FakeCuteKernel()
    layout = torch.full((256,), -1, device=DEVICE, dtype=torch.int32)
    layout[:128] = 0
    layout[128:144] = 1
    n_sizes = torch.tensor((128, 160), device=DEVICE, dtype=torch.int32)

    problem_sizes, starts, total_clusters = _build_tcgen05_grouped_static_metadata(
        kernel, proven_plan, (layout, n_sizes)
    )

    assert problem_sizes.detach().cpu().tolist() == [
        [128, 128, 128, 1],
        [16, 160, 128, 1],
    ]
    assert starts.detach().cpu().tolist() == [0, 128]
    assert total_clusters == 5
    with pytest.raises(helion.exc.BackendUnsupported, match="M size"):
        _build_tcgen05_grouped_static_metadata(kernel, unproven_plan, (layout, n_sizes))


def test_grouped_static_metadata_rejects_invalid_layout_and_k_tail() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static metadata test needs CUDA")

    plan = _grouped_static_metadata_test_plan()
    plan.update({"n_size": 192, "m_tail_preserve": True, "n_tail_preserve": True})
    kernel = _FakeCuteKernel()
    layout = torch.full((256,), -1, device=DEVICE, dtype=torch.int32)
    layout[:128] = 0
    layout[128:144] = 1
    layout[200] = 1
    n_sizes = torch.tensor((128, 160), device=DEVICE, dtype=torch.int32)
    with pytest.raises(helion.exc.BackendUnsupported, match="layout rows"):
        _build_tcgen05_grouped_static_metadata(kernel, plan, (layout, n_sizes))

    bad_k_plan = dict(plan)
    bad_k_plan["k_total_size"] = 192
    layout[200] = -1
    with pytest.raises(helion.exc.BackendUnsupported, match="common N/K"):
        _build_tcgen05_grouped_static_metadata(kernel, bad_k_plan, (layout, n_sizes))


def test_grouped_static_launch_cache_hit_survives_metadata_lru_churn() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static launch cache test needs CUDA")

    plan_count = (
        _TCGEN05_GROUPED_STATIC_METADATA_CACHE_LIMIT // _CUTE_LAUNCH_ARG_CACHE_LIMIT + 1
    )
    kernel = _FakeCuteKernel()
    kernel._helion_cute_wrapper_plans = [
        _grouped_static_metadata_test_plan(
            layout_idx=2 * plan_idx,
            n_sizes_idx=2 * plan_idx + 1,
            suffix=str(plan_idx),
        )
        for plan_idx in range(plan_count)
    ]
    args_by_signature = [
        tuple(
            tensor
            for plan_idx in range(plan_count)
            for tensor in _make_grouped_static_metadata_args(
                (128, 128),
                (64 + 64 * ((signature + plan_idx) % 2), 128),
            )
        )
        for signature in range(_CUTE_LAUNCH_ARG_CACHE_LIMIT)
    ]

    def fake_imports() -> tuple[object, object, object]:
        gmem = object()

        def make_ptr(
            dtype: object,
            data_ptr: int,
            _space: object,
            *,
            assumed_align: int,
        ) -> tuple[str, object, int, int]:
            return ("ptr", dtype, int(data_ptr), assumed_align)

        def current_stream() -> str:
            return "stream"

        return gmem, make_ptr, current_stream

    with (
        patch("helion.runtime._get_cute_launcher_imports", side_effect=fake_imports),
        patch("helion.runtime._torch_dtype_to_cutlass", side_effect=str),
    ):
        first_schema, first_launch_args = _build_cached_cute_schema_and_args(
            kernel, args_by_signature[0], (1, 1, 1)
        )
        for args in args_by_signature[1:]:
            _build_cached_cute_schema_and_args(kernel, args, (1, 1, 1))

        assert (
            len(args_by_signature) * plan_count
            > _TCGEN05_GROUPED_STATIC_METADATA_CACHE_LIMIT
        )
        first_metadata_key = _tcgen05_grouped_static_metadata_cache_key(
            kernel._helion_cute_wrapper_plans[0],
            args_by_signature[0][0],
            args_by_signature[0][1],
        )
        metadata_cache = kernel._helion_tcgen05_grouped_static_metadata_cache
        assert len(metadata_cache) == _TCGEN05_GROUPED_STATIC_METADATA_CACHE_LIMIT
        assert first_metadata_key not in metadata_cache

        first_launch_key = _cute_launch_arg_cache_key(
            kernel, args_by_signature[0], (1, 1, 1)
        )
        first_launch_entry = kernel._helion_cute_launch_arg_cache[first_launch_key]
        assert len(kernel._helion_cute_launch_arg_cache) == _CUTE_LAUNCH_ARG_CACHE_LIMIT
        assert len(first_launch_entry.grouped_static_metadata) == plan_count

        with patch("helion.runtime._cuda_graph_capture_active", return_value=True):
            reused_schema, reused_launch_args = _build_cached_cute_schema_and_args(
                kernel, args_by_signature[0], (1, 1, 1)
            )

    assert reused_schema is first_schema
    assert reused_launch_args is first_launch_args


def test_grouped_static_launch_cache_misses_on_wrapper_plan_content_change() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static launch cache test needs CUDA")

    plan = _grouped_static_metadata_test_plan(
        layout_idx=2,
        n_sizes_idx=3,
        k_sizes_idx=4,
    )
    plan.update(
        {
            "group_count": 4,
            "n_size": 128,
            "k_total_size": 1536,
            "bk": 64,
            "m_tail_preserve": True,
            "n_tail_preserve": True,
            "dynamic_ab_tensormaps": True,
            "dynamic_d_tensormap": False,
            "ab_tensormaps_arg": "tcgen05_grouped_ab_tensormaps",
            "lhs_idx": 0,
            "rhs_idx": 1,
        }
    )
    kernel = _FakeCuteKernel()
    kernel._helion_cute_wrapper_plans = [plan]
    args = _make_documented_mixed_k_args()

    def fake_imports() -> tuple[object, object, object]:
        gmem = object()

        def make_ptr(
            dtype: object,
            data_ptr: int,
            _space: object,
            *,
            assumed_align: int,
        ) -> tuple[str, object, int, int]:
            return ("ptr", dtype, int(data_ptr), assumed_align)

        def current_stream() -> str:
            return "stream"

        return gmem, make_ptr, current_stream

    def workspace_shape(schema: tuple[tuple[object, ...], ...]) -> tuple[int, ...]:
        matches = [
            entry
            for entry in schema
            if entry[:2] == ("wrapper_tensor", "tcgen05_grouped_ab_tensormaps")
        ]
        assert len(matches) == 1
        sizes = matches[0][4]
        assert isinstance(sizes, tuple)
        return sizes

    with (
        patch("helion.runtime._get_cute_launcher_imports", side_effect=fake_imports),
        patch("helion.runtime._torch_dtype_to_cutlass", side_effect=str),
    ):
        first_schema, first_launch_args = _build_cached_cute_schema_and_args(
            kernel, args, (1, 1, 1)
        )
        first_key = _cute_launch_arg_cache_key(kernel, args, (1, 1, 1))
        plan["dynamic_d_tensormap"] = True
        second_schema, second_launch_args = _build_cached_cute_schema_and_args(
            kernel, args, (1, 1, 1)
        )
        second_key = _cute_launch_arg_cache_key(kernel, args, (1, 1, 1))

    assert first_key != second_key
    assert first_schema is not second_schema
    assert first_launch_args is not second_launch_args
    assert workspace_shape(first_schema)[1:] == (2, 16)
    assert workspace_shape(second_schema)[1:] == (3, 16)


def test_grouped_static_last_launch_misses_on_layout_n_k_version() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static last-launch cache test needs CUDA")

    plan = _grouped_static_metadata_test_plan(k_sizes_idx=2)
    kernel = _FakeCuteKernel()
    kernel._helion_cute_wrapper_plans = [plan]
    layout, n_sizes = _make_grouped_static_metadata_args((128, 128), (128, 128))
    k_sizes = torch.tensor((128, 128), device=DEVICE, dtype=torch.int32)
    args = (layout, n_sizes, k_sizes)
    compiled_calls: list[int] = []

    class FakeCompiled:
        def __call__(self, *launch_args: object) -> tuple[str, tuple[object, ...]]:
            return ("launched", launch_args)

    def fake_imports() -> tuple[object, object, object]:
        gmem = object()

        def make_ptr(
            dtype: object,
            data_ptr: int,
            _space: object,
            *,
            assumed_align: int,
        ) -> tuple[str, object, int, int]:
            return ("ptr", dtype, int(data_ptr), assumed_align)

        def current_stream() -> str:
            return "stream-from-imports"

        return gmem, make_ptr, current_stream

    def fake_get(*_args: object, **_kwargs: object) -> FakeCompiled:
        compiled_calls.append(len(compiled_calls) + 1)
        return FakeCompiled()

    with (
        patch("helion.runtime._get_cute_launcher_imports", side_effect=fake_imports),
        patch("helion.runtime._torch_dtype_to_cutlass", side_effect=str),
        patch("helion.runtime._cute_current_stream", return_value="stream"),
        patch("helion.runtime._get_compiled_cute_launcher", side_effect=fake_get),
    ):
        default_cute_launcher(kernel, (1,), *args, block=(32, 1, 1))
        default_cute_launcher(kernel, (1,), *args, block=(32, 1, 1))
        layout.add_(0)
        default_cute_launcher(kernel, (1,), *args, block=(32, 1, 1))
        n_sizes.add_(0)
        default_cute_launcher(kernel, (1,), *args, block=(32, 1, 1))
        k_sizes.add_(0)
        default_cute_launcher(kernel, (1,), *args, block=(32, 1, 1))

    assert compiled_calls == [1, 2, 3, 4]


def test_grouped_static_frozen_plan_fast_launcher_guards_metadata() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static last-launch cache test needs CUDA")

    plan = _grouped_static_metadata_test_plan(k_sizes_idx=2)
    kernel = _FakeCuteKernel()
    kernel._helion_cute_wrapper_plans = _freeze_cute_wrapper_plans([plan])
    layout, n_sizes = _make_grouped_static_metadata_args((128, 128), (128, 128))
    k_sizes = torch.tensor((128, 128), device=DEVICE, dtype=torch.int32)
    args = (layout, n_sizes, k_sizes)
    replacement_n_sizes = n_sizes.clone()
    compiled_calls: list[int] = []
    launched: list[tuple[int, tuple[object, ...]]] = []

    class FakeCompiled:
        def __init__(self, index: int) -> None:
            self.index = index

        def __call__(self, *launch_args: object) -> tuple[int, tuple[object, ...]]:
            launched.append((self.index, launch_args))
            return (self.index, launch_args)

    def fake_imports() -> tuple[object, object, object]:
        gmem = object()

        def make_ptr(
            dtype: object,
            data_ptr: int,
            _space: object,
            *,
            assumed_align: int,
        ) -> tuple[str, object, int, int]:
            return ("ptr", dtype, int(data_ptr), assumed_align)

        def current_stream() -> str:
            return "stream-from-imports"

        return gmem, make_ptr, current_stream

    def fake_get(*_args: object, **_kwargs: object) -> FakeCompiled:
        compiled_calls.append(len(compiled_calls) + 1)
        return FakeCompiled(len(compiled_calls))

    import helion.runtime as runtime_mod

    generic_matcher = runtime_mod._cute_last_launch_cache_entry
    with (
        patch("helion.runtime._get_cute_launcher_imports", side_effect=fake_imports),
        patch("helion.runtime._torch_dtype_to_cutlass", side_effect=str),
        patch("helion.runtime._cute_current_stream", return_value="stream"),
        patch("helion.runtime._get_compiled_cute_launcher", side_effect=fake_get),
        patch(
            "helion.runtime._cute_last_launch_cache_entry",
            wraps=generic_matcher,
        ) as generic_match,
    ):
        first = default_cute_launcher(kernel, (1,), *args, block=(32, 1, 1))
        assert kernel._helion_cute_last_launch_cache.fast_launcher is not None
        second = default_cute_launcher(kernel, (1,), *args, block=(32, 1, 1))
        assert generic_match.call_count == 1

        layout.add_(0)
        layout_version_miss = default_cute_launcher(
            kernel, (1,), *args, block=(32, 1, 1)
        )
        assert kernel._helion_cute_last_launch_cache.fast_launcher is not None
        assert generic_match.call_count == 2

        replacement_args = (layout, replacement_n_sizes, k_sizes)
        n_sizes_replacement_miss = default_cute_launcher(
            kernel, (1,), *replacement_args, block=(32, 1, 1)
        )
        assert kernel._helion_cute_last_launch_cache.fast_launcher is not None
        assert generic_match.call_count == 3

        k_sizes.add_(0)
        k_sizes_version_miss = default_cute_launcher(
            kernel, (1,), *replacement_args, block=(32, 1, 1)
        )
        assert kernel._helion_cute_last_launch_cache.fast_launcher is not None
        assert generic_match.call_count == 4

    assert first[0] == 1
    assert second[0] == 1
    assert layout_version_miss[0] == 2
    assert n_sizes_replacement_miss[0] == 3
    assert k_sizes_version_miss[0] == 4
    assert [entry[0] for entry in launched] == [1, 1, 2, 3, 4]
    assert compiled_calls == [1, 2, 3, 4]


def test_grouped_static_last_launch_retains_dynamic_tensormap_workspace() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static last-launch cache test needs CUDA")

    plan = _grouped_static_metadata_test_plan(
        layout_idx=2,
        n_sizes_idx=3,
        k_sizes_idx=4,
    )
    plan.update(
        {
            "group_count": 4,
            "n_size": 128,
            "k_total_size": 1536,
            "bk": 64,
            "m_tail_preserve": True,
            "n_tail_preserve": True,
            "dynamic_ab_tensormaps": True,
            "dynamic_d_tensormap": True,
            "ab_tensormaps_arg": "tcgen05_grouped_ab_tensormaps",
            "lhs_idx": 0,
            "rhs_idx": 1,
        }
    )
    kernel = _FakeCuteKernel()
    kernel._helion_cute_wrapper_plans = [plan]
    args = _make_documented_mixed_k_args()
    launched: list[tuple[object, ...]] = []
    compiled_calls: list[int] = []

    class FakeCompiled:
        def __call__(self, *launch_args: object) -> tuple[str, tuple[object, ...]]:
            launched.append(launch_args)
            return ("launched", launch_args)

    def fake_imports() -> tuple[object, object, object]:
        gmem = object()

        def make_ptr(
            dtype: object,
            data_ptr: int,
            _space: object,
            *,
            assumed_align: int,
        ) -> tuple[str, object, int, int]:
            return ("ptr", dtype, int(data_ptr), assumed_align)

        def current_stream() -> str:
            return "stream-from-imports"

        return gmem, make_ptr, current_stream

    def fake_get(*_args: object, **_kwargs: object) -> FakeCompiled:
        compiled_calls.append(len(compiled_calls) + 1)
        return FakeCompiled()

    with (
        patch("helion.runtime._get_cute_launcher_imports", side_effect=fake_imports),
        patch("helion.runtime._torch_dtype_to_cutlass", side_effect=str),
        patch("helion.runtime._cute_current_stream", return_value="stream"),
        patch("helion.runtime._get_compiled_cute_launcher", side_effect=fake_get),
    ):
        default_cute_launcher(kernel, (1,), *args, block=(32, 1, 1))
        entry = kernel._helion_cute_last_launch_cache
        assert any(tensor.shape[1:] == (3, 16) for tensor in entry.owned_tensors)
        with patch("helion.runtime._cuda_graph_capture_active", return_value=True):
            default_cute_launcher(kernel, (1,), *args, block=(32, 1, 1))

    assert compiled_calls == [1]
    assert len(launched) == 2


def test_grouped_static_last_launch_retains_d_only_tensormap_workspace() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("grouped-static last-launch cache test needs CUDA")

    plan = _grouped_static_metadata_test_plan(
        layout_idx=2,
        n_sizes_idx=3,
        suffix="d",
    )
    plan.update(
        {
            "group_count": 1,
            "n_size": 192,
            "k_total_size": 16,
            "bk": 16,
            "m_tail_preserve": True,
            "n_tail_preserve": True,
            "dynamic_d_tensormap": True,
            "d_tensormaps_arg": "tcgen05_grouped_d_tensormaps",
            "d_tensormap_slot": 0,
            "d_tensormap_tail_store": True,
        }
    )
    kernel = _FakeCuteKernel()
    kernel._helion_cute_wrapper_plans = [plan]
    args = _make_single_padded_n_tail_args()
    launched: list[tuple[object, ...]] = []
    compiled_calls: list[int] = []

    class FakeCompiled:
        def __call__(self, *launch_args: object) -> tuple[str, tuple[object, ...]]:
            launched.append(launch_args)
            return ("launched", launch_args)

    def fake_imports() -> tuple[object, object, object]:
        gmem = object()

        def make_ptr(
            dtype: object,
            data_ptr: int,
            _space: object,
            *,
            assumed_align: int,
        ) -> tuple[str, object, int, int]:
            return ("ptr", dtype, int(data_ptr), assumed_align)

        def current_stream() -> str:
            return "stream-from-imports"

        return gmem, make_ptr, current_stream

    def fake_get(*_args: object, **_kwargs: object) -> FakeCompiled:
        compiled_calls.append(len(compiled_calls) + 1)
        return FakeCompiled()

    with (
        patch("helion.runtime._get_cute_launcher_imports", side_effect=fake_imports),
        patch("helion.runtime._torch_dtype_to_cutlass", side_effect=str),
        patch("helion.runtime._cute_current_stream", return_value="stream"),
        patch("helion.runtime._get_compiled_cute_launcher", side_effect=fake_get),
    ):
        default_cute_launcher(kernel, (1,), *args, block=(32, 1, 1))
        entry = kernel._helion_cute_last_launch_cache
        assert any(tensor.shape[1:] == (1, 16) for tensor in entry.owned_tensors)
        assert not any(tensor.shape[1:] == (3, 16) for tensor in entry.owned_tensors)
        with patch("helion.runtime._cuda_graph_capture_active", return_value=True):
            default_cute_launcher(kernel, (1,), *args, block=(32, 1, 1))

    assert compiled_calls == [1]
    assert len(launched) == 2


def _make_mixed_boundary_grouped_args() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    a, b_grouped, layout = _make_all_full_grouped_args()
    mixed = layout.clone()
    mixed[:128] = 0
    mixed[128:192] = 1
    mixed[192:320] = 2
    mixed[320:] = 3
    return a, b_grouped, mixed


def _rank3_rhs_grouped_static_config_bn64() -> helion.Config:
    config = _rank3_rhs_grouped_static_config()
    config.config["block_sizes"] = [128, 64, 128]
    return config


def _rank3_rhs_grouped_static_config_bn64_bk(block_k: int) -> helion.Config:
    config = _rank3_rhs_grouped_static_config_bn64()
    config.config["block_sizes"] = [128, 64, block_k]
    return config


def _rank3_rhs_grouped_static_dynamic_bk64_config() -> helion.Config:
    config = _rank3_rhs_grouped_static_config_bn64_bk(64)
    config.config[TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY] = True
    return config


def _rank3_rhs_grouped_static_direct_bk64_config() -> helion.Config:
    config = _rank3_rhs_grouped_static_dynamic_bk64_config()
    config.config[TCGEN05_GROUPED_DIRECT_POINTER_METADATA_CONFIG_KEY] = True
    return config


def _make_partial_m_args() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b_grouped, layout = _make_args()
    return a[:192, :], b_grouped, layout[:192]


def _make_non_row_major_a_args() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b_grouped, layout = _make_args()
    m, k = a.shape
    strided_a = torch.empty((k, m), device=DEVICE, dtype=a.dtype).T
    strided_a.zero_()
    row_ids = torch.arange(m, device=DEVICE)
    strided_a[row_ids, row_ids % k] = 1
    assert strided_a.shape == a.shape
    assert strided_a.stride(1) != 1
    return strided_a, b_grouped, layout


def _make_non_k_contiguous_b_args() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a, _b_grouped, layout = _make_args()
    groups, n, k = 4, 256, 128
    b_grouped = torch.empty(
        (groups, k, n),
        device=DEVICE,
        dtype=torch.bfloat16,
    ).transpose(1, 2)
    assert b_grouped.shape == (groups, n, k)
    assert b_grouped.stride(2) != 1
    return a, b_grouped, layout


def _code_for(
    kernel: object,
    args: tuple[torch.Tensor, ...] | None = None,
    config: helion.Config | None = None,
) -> str:
    if args is None:
        args = _make_args()
    if config is None:
        config = _rank3_rhs_tma_config()
    bound = kernel.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    with (
        patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
        patch_cute_mma_support(),
    ):
        return bound.to_triton_code(config)


def _wrapper_plans_from_code(code: str) -> list[dict[str, object]]:
    marker = "._helion_cute_wrapper_plans = "
    line = next(line for line in code.splitlines() if marker in line)
    payload = line.split(marker, 1)[1]
    freeze_prefix = "helion.runtime._freeze_cute_wrapper_plans("
    if payload.startswith(freeze_prefix) and payload.endswith(")"):
        payload = payload[len(freeze_prefix) : -1]
    return list(ast.literal_eval(payload))


def test_rank3_rhs_wrapper_metadata_is_module_level_only() -> None:
    code = _code_for(
        _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes,
        _make_documented_mixed_k_args(),
        _rank3_rhs_grouped_static_dynamic_bk64_config(),
    )
    module = ast.parse(code)
    metadata_attrs = {"_helion_cute_wrapper_plans", "_helion_cute_cluster_shape"}
    module_assigns: list[tuple[int, str, str]] = []
    for index, stmt in enumerate(module.body):
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr in metadata_attrs
                and isinstance(target.value, ast.Name)
            ):
                module_assigns.append((index, target.value.id, target.attr))
    assert any(attr == "_helion_cute_wrapper_plans" for _, _, attr in module_assigns)
    for node in ast.walk(module):
        if not isinstance(node, ast.FunctionDef):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            for target in child.targets:
                assert not (
                    isinstance(target, ast.Attribute) and target.attr in metadata_attrs
                )
    for assign_index, kernel_name, _attr in module_assigns:
        kernel_index = next(
            index
            for index, stmt in enumerate(module.body)
            if isinstance(stmt, ast.FunctionDef) and stmt.name == kernel_name
        )
        next_function_index = next(
            index
            for index, stmt in enumerate(
                module.body[assign_index + 1 :], assign_index + 1
            )
            if isinstance(stmt, ast.FunctionDef)
        )
        assert kernel_index < assign_index < next_function_index


def _assert_grouped_static_codegen_markers(code: str) -> dict[str, object]:
    assert "StaticPersistentGroupTileScheduler.create" in code
    assert ".group_search_result" in code
    assert "virtual_pid" not in code
    assert "_cutlass_grouped_gemm_kernel" not in code
    assert "GroupedGemmKernel" not in code
    assert "tcgen05_rhs_safe_group" not in code
    plans = _wrapper_plans_from_code(code)
    return next(
        plan for plan in plans if plan["kind"] == "tcgen05_grouped_static_persistent"
    )


def _assert_no_grouped_static_codegen_leak(code: str) -> None:
    assert "tcgen05_grouped_static_persistent" not in code
    assert "StaticPersistentGroupTileScheduler.create" not in code
    assert "tcgen05_grouped_problem_n" not in code
    assert "tcgen05_grouped_tile_sched_params" not in code
    assert "tcgen05_grouped_group_idx" not in code
    assert "virtual_pid" in code


def _assert_backend_unsupported_match(
    error: helion.exc.BackendUnsupported, match: str
) -> None:
    message = str(error)
    assert re.search(match, message), message


def _assert_semantic_n_sizes_codegen_markers(code: str, *, expected_bn: int) -> None:
    grouped_plan = _assert_grouped_static_codegen_markers(code)
    assert grouped_plan["bn"] == expected_bn
    assert "n_sizes_idx" in grouped_plan
    assert ".problem_shape_n" in code
    assert "tcgen05_grouped_problem_n" in code
    assert re.search(r"^\s*valid_cols\s*=", code, re.MULTILINE) is None
    assert "b_grouped.iterator" not in code
    plans = _wrapper_plans_from_code(code)
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["input_dtype"] == "cutlass.Float16"


def _assert_semantic_mn_tail_codegen_markers(
    code: str, *, expected_bn: int
) -> dict[str, object]:
    grouped_plan = _assert_grouped_static_codegen_markers(code)
    assert grouped_plan["bn"] == expected_bn
    assert grouped_plan["m_tail_preserve"] is True
    assert grouped_plan["n_tail_preserve"] is True
    assert "n_sizes_idx" in grouped_plan
    assert ".problem_shape_m" in code
    assert ".problem_shape_n" in code
    assert "tcgen05_grouped_problem_m" in code
    assert "tcgen05_grouped_problem_n" in code
    assert re.search(r"^\s*valid_rows\s*=", code, re.MULTILINE) is None
    assert re.search(r"^\s*valid_cols\s*=", code, re.MULTILINE) is None
    assert "b_grouped.iterator" not in code
    return grouped_plan


def _assert_grouped_static_d_only_tail_tma_store(
    code: str,
    *,
    expected_bk: int,
    expected_has_m_tail: bool = False,
    expected_has_n_tail: bool = True,
) -> None:
    grouped_plan = _assert_grouped_static_codegen_markers(code)
    assert grouped_plan["bk"] == expected_bk
    assert grouped_plan["grouped_static_has_m_tail"] is expected_has_m_tail
    assert grouped_plan["grouped_static_has_n_tail"] is expected_has_n_tail
    assert grouped_plan["dynamic_d_tensormap"] is True
    assert grouped_plan["d_tensormap_tail_store"] is True
    assert grouped_plan["d_tensormap_slot"] == 0
    assert grouped_plan["d_tensormaps_arg"] == "tcgen05_grouped_d_tensormaps"
    assert grouped_plan.get("dynamic_ab_tensormaps") is not True
    assert "tcgen05_grouped_d_tensormaps" in code
    assert "tcgen05_grouped_ab_tensormaps" not in code
    assert "tcgen05_grouped_tensormap_manager" not in code
    assert "tma_desc_ptr=tcgen05_grouped_tensormap_a_desc_ptr" not in code
    assert "tma_desc_ptr=tcgen05_grouped_tensormap_b_desc_ptr" not in code
    assert "dynamic_ab_tensormaps" not in code
    assert "CopyR2GOp" not in code
    assert "tcgen05_simt_atom" not in code
    assert "tcgen05_edge_pred" not in code
    assert "_cutlass_grouped_gemm_kernel" not in code
    assert "GroupedGemmKernel" not in code
    assert "blackwell_grouped_gemm_nt" not in code
    assert "virtual_pid" not in code
    assert "tcgen05_grouped_d_tensormap_manager.update_tensormap(" in code
    assert "tcgen05_grouped_d_tensormap_manager.fence_tensormap_update(" in code
    assert "tma_desc_ptr=tcgen05_grouped_d_tensormap_desc_ptr" in code
    assert (
        "tcgen05_grouped_d_tensormap_base = out.iterator + "
        "cutlass.Int32(tcgen05_grouped_global_m_start) * "
        "cutlass.Int32(out.layout.stride[0])"
    ) in code
    assert (
        "cute.make_layout((tcgen05_grouped_problem_m, "
        "tcgen05_grouped_problem_n, cutlass.Int32(1)), "
        "stride=(out.layout.stride[0], out.layout.stride[1], cutlass.Int32(0)))"
    ) in code
    assert (
        "cute.local_tile(tcgen05_tail_tma_store_tensor, (128, 64), "
        "(tcgen05_grouped_cta_tile_idx_m, tcgen05_grouped_cta_tile_idx_n, 0))"
    ) in code
    plans = _wrapper_plans_from_code(code)
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["rhs_rank3_grouped_nt"] is True
    assert ab_plan.get("dynamic_ab_tensormaps") is not True
    d_plans = [plan for plan in plans if plan["kind"] == "tcgen05_d_tma"]
    assert any(plan.get("rank3_mnl_tensor") is True for plan in d_plans)


def _assert_grouped_static_no_tail_static_d_tma_store(
    code: str, *, expected_bk: int
) -> None:
    grouped_plan = _assert_grouped_static_codegen_markers(code)
    assert grouped_plan["bk"] == expected_bk
    assert grouped_plan["grouped_static_has_m_tail"] is False
    assert grouped_plan["grouped_static_has_n_tail"] is False
    assert grouped_plan.get("dynamic_d_tensormap") is not True
    assert grouped_plan.get("d_tensormap_tail_store") is not True
    assert "tcgen05_grouped_d_tensormaps" not in code
    assert "tcgen05_grouped_d_tensormap_manager" not in code
    assert "tcgen05_tail_tma_store_tensor" not in code
    assert "tcgen05_tail_tma_store_atom" not in code
    assert "update_tensormap" not in code
    assert "fence_tensormap_update" not in code
    assert "CopyR2GOp" not in code
    assert "tcgen05_simt_atom" not in code
    assert "tcgen05_edge_pred" not in code
    assert "_cutlass_grouped_gemm_kernel" not in code
    assert "GroupedGemmKernel" not in code
    assert "blackwell_grouped_gemm_nt" not in code
    assert "virtual_pid" not in code
    plans = _wrapper_plans_from_code(code)
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["rhs_rank3_grouped_nt"] is True
    assert ab_plan.get("dynamic_ab_tensormaps") is not True


def _assert_semantic_k_sizes_codegen_markers(
    code: str,
    *,
    expected_bn: int,
    expected_bk: int,
    expected_k_total: int,
    allow_dynamic_tensormap_bases: bool = False,
) -> dict[str, object]:
    grouped_plan = _assert_grouped_static_codegen_markers(code)
    assert grouped_plan["bn"] == expected_bn
    assert grouped_plan["bk"] == expected_bk
    assert grouped_plan["k_total_size"] == expected_k_total
    assert "k_sizes_idx" in grouped_plan
    assert ".problem_shape_k" in code
    assert "tcgen05_grouped_problem_k" in code
    assert f"tcgen05_grouped_problem_k, cutlass.Int32({expected_bk})" in code
    assert re.search(r"^\s*group_k\s*=", code, re.MULTILINE) is None
    assert re.search(r"^\s*valid_k\s*=", code, re.MULTILINE) is None
    assert "k_sizes.iterator" not in code
    if not allow_dynamic_tensormap_bases:
        assert "a.iterator" not in code
        assert "b_grouped.iterator" not in code
    return grouped_plan


def _assert_semantic_m_tail_codegen_markers(
    code: str, *, expected_bn: int
) -> dict[str, object]:
    grouped_plan = _assert_grouped_static_codegen_markers(code)
    assert grouped_plan["bn"] == expected_bn
    assert grouped_plan["m_tail_preserve"] is True
    assert grouped_plan.get("n_tail_preserve", False) is False
    assert "n_sizes_idx" not in grouped_plan
    assert ".problem_shape_m" in code
    assert ".problem_shape_n" in code
    assert "tcgen05_grouped_problem_m" in code
    assert "tcgen05_grouped_problem_n" in code
    assert re.search(r"^\s*row_group_ids\s*=", code, re.MULTILINE) is None
    assert re.search(r"^\s*valid_rows\s*=", code, re.MULTILINE) is None
    assert re.search(r"^\s*valid\s*=", code, re.MULTILINE) is None
    assert "b_grouped.iterator" not in code
    return grouped_plan


def _load_deepgemm_m_grouped_module() -> Any:
    with patch.dict(
        os.environ,
        {"HELION_BACKEND": "cute", "HELION_CUTE_MMA_IMPL": "tcgen05"},
        clear=False,
    ):
        from benchmarks.cute import deepgemm_m_grouped

    return deepgemm_m_grouped


def _example_grouped_static_bound(
    *,
    k: int = 384,
) -> tuple[Any, helion.Config]:
    deepgemm_m_grouped = _load_deepgemm_m_grouped_module()
    args, _expected = (
        deepgemm_m_grouped.make_deepgemm_m_grouped_bf16_gemm_nt_contiguous_args(
            (128, 128, 128, 128),
            n=256,
            k=k,
            m_alignment=128,
            tail_padding=0,
            device=DEVICE,
        )
    )
    bound = deepgemm_m_grouped._deepgemm_m_grouped_bf16_gemm_nt_contiguous_kernel.bind(
        args
    )
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    return bound, _rank3_rhs_grouped_static_default_ab_config()


def _stmt(src: str) -> ast.stmt:
    body = ast.parse(src).body
    assert len(body) == 1
    return body[0]


def test_grouped_static_dependency_stmts_drop_only_simple_coord_assigns() -> None:
    kept = Tcgen05PersistentProgramIDs._grouped_static_dependency_stmts(
        [
            _stmt("pid_0 = cutlass.Int32(0)"),
            _stmt("tile_offset_0 = pid_0 * cutlass.Int32(128)"),
            _stmt("ordinary = pid_0"),
        ]
    )

    assert [ast.unparse(stmt) for stmt in kept] == ["ordinary = pid_0"]


def test_grouped_static_dependency_stmts_reject_mixed_coord_write() -> None:
    with pytest.raises(AssertionError, match="mixed coordinate dependency"):
        Tcgen05PersistentProgramIDs._grouped_static_dependency_stmts(
            [_stmt("pid_0 = ordinary = cutlass.Int32(0)")]
        )


def test_grouped_static_omit_shared_loop_rejects_arbitrary_v_write() -> None:
    partition = Tcgen05PersistentProgramIDs._PartitionedRoleBody(
        role_blocks_inline=[],
        role_blocks_extracted=[],
        shared_body_extracted=[_stmt("v_probe = cutlass.Int32(0)")],
    )

    with pytest.raises(AssertionError, match="v_probe"):
        Tcgen05PersistentProgramIDs._assert_tcgen05_grouped_omit_shared_loop_safe(
            Tcgen05PersistentProgramIDs,
            partition,
        )


def test_grouped_static_omit_shared_loop_allows_known_group_scalar_rewrites() -> None:
    partition = Tcgen05PersistentProgramIDs._PartitionedRoleBody(
        role_blocks_inline=[],
        role_blocks_extracted=[],
        shared_body_extracted=[
            _stmt("group_id = grouped_layout.load()"),
            _stmt(
                "safe_group_id = group_id if group_id >= cutlass.Int64(0) "
                "else cutlass.Int64(0)"
            ),
            _stmt("cute.arch.sync_threads()"),
        ],
    )

    Tcgen05PersistentProgramIDs._assert_tcgen05_grouped_omit_shared_loop_safe(
        Tcgen05PersistentProgramIDs,
        partition,
    )


def test_rank3_rhs_grouped_nt_codegen_uses_nkg_tma_view() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(_rank3_rhs_grouped_nt)

    assert "tcgen05" in code
    assert "'rhs_rank3_grouped_nt': True" in code
    assert "StaticPersistentTileScheduler.create" in code
    assert "tcgen05_role_local_0_tile_sched" in code
    assert "block=(32, 6, 1)" in code
    assert "tcgen05_rhs_safe_group" in code
    assert "tcgen05_rhs_group = (" in code
    assert ".layout.stride[0])).load()" in code
    assert (
        re.search(
            r"^\s*group_id\s*=.*\.load\(\) if mask_0 else",
            code,
            re.MULTILINE,
        )
        is None
    )
    trailing_local_tile = [
        line.strip()
        for line in code.splitlines()
        if "cute.local_tile(tma_tensor_b" in line
    ]
    assert any(
        re.search(
            r"cute\.local_tile\(tma_tensor_b, \(128, 128\), "
            r"\([^,]+ // cutlass\.Int32\(128\), None, tcgen05_rhs_safe_group\)\)",
            line,
        )
        for line in trailing_local_tile
    )
    assert "cute.slice_(tma_tensor_b" not in code
    assert "tma_tensor_b[tcgen05_rhs_safe_group" not in code

    plans = _wrapper_plans_from_code(code)
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["rhs_rank3_grouped_nt"] is True
    rhs_idx = int(ab_plan["rhs_idx"])
    body: list[str] = []
    call_args: list[str] = []
    _append_cute_wrapper_plan(body, call_args, ab_plan)
    wrapper = "\n".join(body)
    assert f"(arg{rhs_idx}_shape1, arg{rhs_idx}_shape2, arg{rhs_idx}_shape0)" in wrapper
    assert (
        f"stride=(arg{rhs_idx}_stride1, arg{rhs_idx}_stride2, arg{rhs_idx}_stride0)"
    ) in wrapper
    assert ".mark_layout_dynamic(leading_dim=1)" in wrapper


def test_rank3_rhs_grouped_static_persistent_codegen_uses_group_scheduler() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt,
        _make_all_full_grouped_args(),
        _rank3_rhs_grouped_static_config(),
    )

    assert "'rhs_rank3_grouped_nt': True" in code
    assert "dynamic_ab_tensormap_rank" not in code
    assert "StaticPersistentGroupTileScheduler.create" in code
    assert "StaticPersistentTileScheduler.create" not in code
    assert "cutlass.utils.create_initial_search_state()" in code
    assert ".group_search_result" in code
    assert ".group_idx" in code
    assert ".cta_tile_idx_m" in code
    assert ".cta_tile_idx_n" in code
    assert ".cta_tile_count_k" in code
    assert ".problem_shape_m" in code
    assert ".problem_shape_n" in code
    assert ".problem_shape_k" in code
    assert "tcgen05_grouped_global_m_start" in code
    assert "tcgen05_grouped_group_idx" in code
    assert "cute.arch.setmaxregister_increase(240)" in code
    assert "cute.arch.setmaxregister_increase(256)" not in code
    assert "virtual_pid" not in code
    assert "tcgen05_rhs_safe_group" not in code
    assert ".layout.iterator" not in code
    assert "_cutlass_grouped_gemm_kernel" not in code
    assert "GroupedGemmKernel" not in code
    assert re.search(
        r"cute\.local_tile\(tma_tensor_b, \(128, 128\), "
        r"\([^,]+ // cutlass\.Int32\(128\), None, tcgen05_grouped_group_idx\)\)",
        code,
    )

    plans = _wrapper_plans_from_code(code)
    grouped_plans = [
        plan for plan in plans if plan["kind"] == "tcgen05_grouped_static_persistent"
    ]
    assert len(grouped_plans) == 1
    grouped_plan = grouped_plans[0]
    assert grouped_plan["group_count"] == 4
    assert grouped_plan["bm"] == 128
    assert grouped_plan["bn"] == 128
    assert grouped_plan["bk"] == 128
    assert grouped_plan["n_size"] == 256
    assert grouped_plan["k_total_size"] == 128
    assert "layout_idx" in grouped_plan

    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["rhs_rank3_grouped_nt"] is True
    assert ab_plan.get("dynamic_ab_tensormap_rank") is None


def test_rank3_rhs_grouped_static_persistent_codegen_accepts_fp16() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt,
        _make_all_full_grouped_args(dtype=torch.float16),
        _rank3_rhs_grouped_static_config(),
    )

    grouped_plan = _assert_grouped_static_codegen_markers(code)
    assert grouped_plan["bn"] == 128
    plans = _wrapper_plans_from_code(code)
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["input_dtype"] == "cutlass.Float16"


def test_rank3_rhs_grouped_static_persistent_codegen_accepts_block_n64() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt,
        _make_all_full_grouped_args(dtype=torch.float16),
        _rank3_rhs_grouped_static_config_bn64(),
    )

    grouped_plan = _assert_grouped_static_codegen_markers(code)
    assert grouped_plan["bn"] == 64
    plans = _wrapper_plans_from_code(code)
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["input_dtype"] == "cutlass.Float16"


def test_rank3_rhs_grouped_static_persistent_bn64_rejects_n_tail() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    with pytest.raises(helion.exc.BackendUnsupported, match="static_full_tiles"):
        _code_for(
            _rank3_rhs_grouped_nt,
            _make_all_full_grouped_args(n=160, dtype=torch.float16),
            _rank3_rhs_grouped_static_config_bn64(),
        )


def test_rank3_rhs_grouped_static_persistent_codegen_uses_semantic_n_sizes() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_with_n_sizes,
        _make_nvidia_default_like_n_sizes_args(),
        _rank3_rhs_grouped_static_config_bn64(),
    )

    _assert_semantic_n_sizes_codegen_markers(code, expected_bn=64)


def test_rank3_rhs_grouped_static_persistent_codegen_uses_semantic_n_sizes_bn128() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_with_n_sizes,
        _make_nvidia_default_like_n_sizes_args(),
        _rank3_rhs_grouped_static_config(),
    )

    _assert_semantic_n_sizes_codegen_markers(code, expected_bn=128)


def test_rank3_rhs_grouped_static_persistent_codegen_uses_semantic_mn_tails() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_with_mn_tails,
        _make_mn_tail_grouped_args(),
        _rank3_rhs_grouped_static_config_bn64(),
    )

    _assert_semantic_mn_tail_codegen_markers(code, expected_bn=64)


@pytest.mark.parametrize(
    ("k", "block_k"),
    [
        (32, 32),
        (64, 64),
        (96, 32),
        (160, 32),
        (192, 64),
    ],
)
def test_rank3_rhs_grouped_static_persistent_codegen_accepts_common_k_tail(
    k: int, block_k: int
) -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_with_mn_tails,
        _make_mn_tail_grouped_args(k=k),
        _rank3_rhs_grouped_static_config_bn64_bk(block_k),
    )

    grouped_plan = _assert_semantic_mn_tail_codegen_markers(code, expected_bn=64)
    assert grouped_plan["bk"] == block_k
    assert grouped_plan["k_total_size"] == k
    assert grouped_plan.get("dynamic_d_tensormap") is not True
    plans = _wrapper_plans_from_code(code)
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["bk"] == block_k
    assert ab_plan["ab_stage_count"] == 2


def test_rank3_rhs_grouped_static_persistent_codegen_accepts_bk16() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    k = 16
    code = _code_for(
        _rank3_rhs_grouped_nt_with_mn_tails,
        _make_mn_tail_grouped_args(k=k),
        _rank3_rhs_grouped_static_config_bn64_bk(k),
    )

    assert "PipelineTmaUmma.create(num_stages=2" in code
    assert "PipelineTmaUmma.create(num_stages=3" not in code
    assert "'rhs_rank3_grouped_nt': True" in code
    grouped_plan = _assert_semantic_mn_tail_codegen_markers(code, expected_bn=64)
    assert grouped_plan["bk"] == k
    assert grouped_plan["k_total_size"] == k
    assert grouped_plan.get("dynamic_d_tensormap") is not True
    plans = _wrapper_plans_from_code(code)
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["bk"] == k
    assert ab_plan["k_total_size"] == k
    assert ab_plan["ab_stage_count"] == 2
    assert ab_plan["rhs_rank3_grouped_nt"] is True
    d_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_d_tma")
    assert d_plan["output_dtype"] == "cutlass.Float16"


@pytest.mark.parametrize(
    ("k", "block_k"),
    [
        (32, 32),
        (64, 64),
        (96, 32),
        (160, 32),
        (192, 64),
    ],
)
def test_rank3_rhs_grouped_static_persistent_codegen_accepts_common_k_n_tail_d_only(
    k: int, block_k: int
) -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_with_mn_tails,
        _make_multi_group_n_tail_args(k=k),
        _rank3_rhs_grouped_static_config_bn64_bk(block_k),
    )

    grouped_plan = _assert_semantic_mn_tail_codegen_markers(code, expected_bn=64)
    assert grouped_plan["bk"] == block_k
    assert grouped_plan["k_total_size"] == k
    _assert_grouped_static_d_only_tail_tma_store(code, expected_bk=block_k)


def test_rank3_rhs_grouped_static_persistent_bk16_padded_n_tail_d_only_codegen() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_with_mn_tails,
        _make_single_padded_n_tail_args(),
        _rank3_rhs_grouped_static_config_bn64_bk(16),
    )

    grouped_plan = _assert_semantic_mn_tail_codegen_markers(code, expected_bn=64)
    assert grouped_plan["group_count"] == 1
    assert grouped_plan["n_size"] == 192
    assert grouped_plan["k_total_size"] == 16
    _assert_grouped_static_d_only_tail_tma_store(code, expected_bk=16)


def test_rank3_rhs_grouped_static_persistent_bk16_padded_no_tail_static_d_codegen() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_with_mn_tails,
        _make_single_padded_n_tail_args(n=128),
        _rank3_rhs_grouped_static_config_bn64_bk(16),
    )

    grouped_plan = _assert_semantic_mn_tail_codegen_markers(code, expected_bn=64)
    assert grouped_plan["group_count"] == 1
    assert grouped_plan["n_size"] == 192
    assert grouped_plan["k_total_size"] == 16
    _assert_grouped_static_no_tail_static_d_tma_store(code, expected_bk=16)


def test_rank3_rhs_grouped_static_persistent_has_n_tail_splits_bound_cache() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    args_list = list(_make_single_padded_n_tail_args(n=128))
    args = tuple(args_list)
    no_tail_bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(args)
    no_tail_bound.to_triton_code(_rank3_rhs_grouped_static_config_bn64_bk(16))
    no_tail_key = _rank3_rhs_grouped_nt_with_mn_tails.specialization_key(args)

    args_list[3].fill_(160)
    tail_args = tuple(args_list)
    tail_key = _rank3_rhs_grouped_nt_with_mn_tails.specialization_key(tail_args)
    tail_bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(tail_args)

    assert no_tail_key != tail_key
    assert no_tail_bound is not tail_bound


def test_rank3_rhs_grouped_static_persistent_has_m_tail_splits_bound_cache() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    args_list = list(_make_single_padded_n_tail_args(n=128))
    args = tuple(args_list)
    no_tail_bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(args)
    no_tail_bound.to_triton_code(_rank3_rhs_grouped_static_config_bn64_bk(16))
    no_tail_key = _rank3_rhs_grouped_nt_with_mn_tails.specialization_key(args)

    _set_single_group_m_tail_layout(args_list[2])
    m_tail_args = tuple(args_list)
    m_tail_key = _rank3_rhs_grouped_nt_with_mn_tails.specialization_key(m_tail_args)
    m_tail_bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(m_tail_args)

    assert no_tail_key != m_tail_key
    assert no_tail_bound is not m_tail_bound


def test_rank3_rhs_grouped_static_persistent_m_tail_only_splits_bound_cache() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    args_list = list(_make_m_tail_grouped_args(m_sizes=(128, 128), k=1536))
    args = tuple(args_list)
    config = _rank3_rhs_grouped_static_config_bn64()
    no_tail_bound = _rank3_rhs_grouped_nt_with_m_tail.bind(args)
    no_tail_code = no_tail_bound.to_triton_code(config)
    no_tail_key = _rank3_rhs_grouped_nt_with_m_tail.specialization_key(args)

    args_list[2].fill_(-1)
    args_list[2][:128] = 0
    args_list[2][128:144] = 1
    m_tail_args = tuple(args_list)
    m_tail_key = _rank3_rhs_grouped_nt_with_m_tail.specialization_key(m_tail_args)
    m_tail_bound = _rank3_rhs_grouped_nt_with_m_tail.bind(m_tail_args)
    m_tail_code = m_tail_bound.to_triton_code(config)

    assert no_tail_key != m_tail_key
    assert no_tail_bound is not m_tail_bound
    _assert_grouped_static_no_tail_static_d_tma_store(
        no_tail_code,
        expected_bk=128,
    )
    _assert_grouped_static_d_only_tail_tma_store(
        m_tail_code,
        expected_bk=128,
        expected_has_m_tail=True,
        expected_has_n_tail=False,
    )


def test_rank3_rhs_grouped_static_persistent_renamed_metadata_rebinds_n_tail() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    args_list = list(_make_single_padded_n_tail_args(n=128))
    args = tuple(args_list)
    config = _rank3_rhs_grouped_static_config_bn64_bk(16)
    no_tail_bound = _rank3_rhs_grouped_nt_with_renamed_mn_tails.bind(args)
    no_tail_code = no_tail_bound.to_triton_code(config)
    _assert_grouped_static_no_tail_static_d_tma_store(no_tail_code, expected_bk=16)
    no_tail_key = _rank3_rhs_grouped_nt_with_renamed_mn_tails.specialization_key(args)

    args_list[3].fill_(160)
    tail_args = tuple(args_list)
    tail_key = _rank3_rhs_grouped_nt_with_renamed_mn_tails.specialization_key(tail_args)
    tail_bound = _rank3_rhs_grouped_nt_with_renamed_mn_tails.bind(tail_args)
    tail_code = tail_bound.to_triton_code(config)
    _assert_grouped_static_d_only_tail_tma_store(tail_code, expected_bk=16)

    assert no_tail_key != tail_key
    assert no_tail_bound is not tail_bound


def test_rank3_rhs_grouped_static_persistent_renamed_metadata_rebinds_m_tail() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    args_list = list(_make_single_padded_n_tail_args(n=128))
    args = tuple(args_list)
    config = _rank3_rhs_grouped_static_config_bn64_bk(16)
    no_tail_bound = _rank3_rhs_grouped_nt_with_renamed_mn_tails.bind(args)
    no_tail_bound.to_triton_code(config)
    no_tail_key = _rank3_rhs_grouped_nt_with_renamed_mn_tails.specialization_key(args)

    _set_single_group_m_tail_layout(args_list[2])
    m_tail_args = tuple(args_list)
    m_tail_key = _rank3_rhs_grouped_nt_with_renamed_mn_tails.specialization_key(
        m_tail_args
    )
    m_tail_bound = _rank3_rhs_grouped_nt_with_renamed_mn_tails.bind(m_tail_args)

    assert no_tail_key != m_tail_key
    assert no_tail_bound is not m_tail_bound


def test_rank3_rhs_grouped_static_persistent_bk16_unpadded_n_tail_no_d_only_codegen() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    with pytest.raises(helion.exc.BackendUnsupported, match="static_full_tiles"):
        _code_for(
            _rank3_rhs_grouped_nt_with_mn_tails,
            _make_single_padded_n_tail_args(max_n=160),
            _rank3_rhs_grouped_static_config_bn64_bk(16),
        )


def test_rank3_rhs_grouped_static_persistent_rejects_bk8() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    with pytest.raises(helion.exc.InvalidConfig, match="block_k"):
        _code_for(
            _rank3_rhs_grouped_nt_with_mn_tails,
            _make_mn_tail_grouped_args(k=8),
            _rank3_rhs_grouped_static_config_bn64_bk(8),
        )


def test_rank3_rhs_grouped_static_persistent_codegen_uses_semantic_m_tail() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_with_m_tail,
        _make_m_tail_grouped_args(),
        _rank3_rhs_grouped_static_config_bn64(),
    )

    _assert_semantic_m_tail_codegen_markers(code, expected_bn=64)
    _assert_grouped_static_d_only_tail_tma_store(
        code,
        expected_bk=128,
        expected_has_m_tail=True,
        expected_has_n_tail=False,
    )


def test_rank3_rhs_grouped_static_persistent_k1536_m_tail_d_only_codegen() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_with_m_tail,
        _make_m_tail_grouped_args(k=1536),
        _rank3_rhs_grouped_static_config_bn64(),
    )

    grouped_plan = _assert_semantic_m_tail_codegen_markers(code, expected_bn=64)
    assert grouped_plan["k_total_size"] == 1536
    assert "n_sizes_idx" not in grouped_plan
    _assert_grouped_static_d_only_tail_tma_store(
        code,
        expected_bk=128,
        expected_has_m_tail=True,
        expected_has_n_tail=False,
    )


def test_rank3_rhs_grouped_static_persistent_rejects_non_preserving_n_sizes() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_bad_n_sizes_zero_store,
        _make_nvidia_default_like_n_sizes_args(),
        _rank3_rhs_grouped_static_config_bn64(),
    )

    _assert_no_grouped_static_codegen_leak(code)


def test_rank3_rhs_grouped_static_persistent_rejects_non_preserving_mn_tails() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_bad_mn_tails_zero_store,
        _make_mn_tail_grouped_args(),
        _rank3_rhs_grouped_static_config_bn64(),
    )

    _assert_no_grouped_static_codegen_leak(code)


def test_rank3_rhs_grouped_static_persistent_rejects_relative_n_index() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_bad_n_sizes_relative_index,
        _make_nvidia_default_like_n_sizes_args(),
        _rank3_rhs_grouped_static_config_bn64(),
    )

    _assert_no_grouped_static_codegen_leak(code)


def test_rank3_rhs_grouped_static_persistent_rejects_relative_m_index() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_bad_m_tail_relative_index,
        _make_mn_tail_grouped_args(),
        _rank3_rhs_grouped_static_config_bn64(),
    )

    _assert_no_grouped_static_codegen_leak(code)


def test_rank3_rhs_grouped_static_persistent_rejects_arbitrary_row_mask() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_bad_m_tail_arbitrary_row_mask,
        _make_mn_tail_grouped_args(),
        _rank3_rhs_grouped_static_config_bn64(),
    )

    _assert_no_grouped_static_codegen_leak(code)


def test_rank3_rhs_grouped_static_persistent_codegen_uses_semantic_k_sizes() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_with_k_sizes,
        _make_per_group_k_args(),
        _rank3_rhs_grouped_static_config_bn64_bk(16),
    )

    grouped_plan = _assert_semantic_k_sizes_codegen_markers(
        code,
        expected_bn=64,
        expected_bk=16,
        expected_k_total=32,
    )
    assert grouped_plan["group_count"] == 2


def test_rank3_rhs_grouped_static_persistent_codegen_uses_documented_mixed_k() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes,
        _make_documented_mixed_k_args(),
        _rank3_rhs_grouped_static_config_bn64_bk(16),
    )

    grouped_plan = _assert_semantic_k_sizes_codegen_markers(
        code,
        expected_bn=64,
        expected_bk=16,
        expected_k_total=1536,
    )
    assert grouped_plan["group_count"] == 4
    assert grouped_plan["m_tail_preserve"] is True
    assert grouped_plan["n_tail_preserve"] is True
    assert "n_sizes_idx" in grouped_plan


def test_rank3_rhs_grouped_static_persistent_codegen_uses_dynamic_bk64_mixed_k() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes,
        _make_documented_mixed_k_args(),
        _rank3_rhs_grouped_static_dynamic_bk64_config(),
    )

    grouped_plan = _assert_semantic_k_sizes_codegen_markers(
        code,
        expected_bn=64,
        expected_bk=64,
        expected_k_total=1536,
        allow_dynamic_tensormap_bases=True,
    )
    assert grouped_plan["group_count"] == 4
    assert grouped_plan["m_tail_preserve"] is True
    assert grouped_plan["n_tail_preserve"] is True
    assert grouped_plan["dynamic_ab_tensormaps"] is True
    assert grouped_plan["dynamic_ab_tensormap_rank"] == 2
    assert grouped_plan["dynamic_d_tensormap"] is True
    assert "ab_tensormaps_arg" in grouped_plan
    assert (
        "tcgen05_grouped_cta_tile_idx_m * cutlass.Int32(128) "
        "< tcgen05_grouped_problem_m"
    ) in code
    assert (
        "tcgen05_grouped_cta_tile_idx_n * cutlass.Int32(64) < tcgen05_grouped_problem_n"
    ) in code
    plans = _wrapper_plans_from_code(code)
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["dynamic_ab_tensormaps"] is True
    assert ab_plan["dynamic_ab_tensormap_rank"] == 2
    lhs_idx = int(ab_plan["lhs_idx"])
    rhs_idx = int(ab_plan["rhs_idx"])
    wrapper_body: list[str] = []
    wrapper_call_args: list[str] = []
    _append_cute_wrapper_plan(wrapper_body, wrapper_call_args, ab_plan)
    wrapper = "\n".join(wrapper_body)
    assert (
        f"(arg{lhs_idx}_shape0, arg{lhs_idx}_shape1), "
        f"stride=(arg{lhs_idx}_stride0, arg{lhs_idx}_stride1)"
    ) in wrapper
    assert f"(arg{lhs_idx}_shape0, arg{lhs_idx}_shape1, 1)" not in wrapper
    assert (
        f"(arg{rhs_idx}_shape1, arg{rhs_idx}_shape2), "
        f"stride=(arg{rhs_idx}_stride1, arg{rhs_idx}_stride2)"
    ) in wrapper
    assert (
        f"(arg{rhs_idx}_shape1, arg{rhs_idx}_shape2, arg{rhs_idx}_shape0)"
        not in wrapper
    )
    d_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_d_tma")
    assert d_plan["rank3_mnl_tensor"] is True
    assert "tcgen05_grouped_ab_tensormaps" in code
    assert "'dynamic_ab_tensormap_rank': 2" in code
    assert "cutlass.utils.TensorMapManager" in code
    assert "init_tensormap_from_atom" in code
    assert "update_tensormap" in code
    assert "fence_tensormap_update" in code
    assert "tma_desc_ptr=tcgen05_grouped_tensormap_a_desc_ptr" in code
    assert "tma_desc_ptr=tcgen05_grouped_tensormap_b_desc_ptr" in code
    assert (
        "tcgen05_grouped_d_tensormap_manager = cutlass.utils.TensorMapManager" in code
    )
    assert "tcgen05_grouped_d_tensormap_manager.init_tensormap_from_atom(" in code
    assert "tcgen05_grouped_d_tensormap_manager.update_tensormap(" in code
    assert "tcgen05_grouped_d_tensormap_manager.fence_tensormap_update(" in code
    assert re.search(
        r"cute\.copy\(\s*tcgen05_tma_store_atom,\s*"
        r"[\s\S]*?tma_desc_ptr=tcgen05_grouped_d_tensormap_desc_ptr\s*\)",
        code,
    )
    assert (
        "for tile_offset_2 in cutlass.range(cutlass.Int32(0), "
        "tcgen05_grouped_problem_k, cutlass.Int32(64), unroll=1):"
    ) in code
    assert "tile_offset_2 < tcgen05_grouped_problem_k" in code
    assert "tcgen05_tma_initial_full_tile" not in code
    assert (
        "cute.make_layout((tcgen05_grouped_problem_m, "
        "tcgen05_grouped_problem_k), "
        "stride=(a.layout.stride[0], a.layout.stride[1]))"
    ) in code
    assert (
        "cute.make_layout((tcgen05_grouped_problem_n, "
        "tcgen05_grouped_problem_k), "
        "stride=(b_grouped.layout.stride[1], b_grouped.layout.stride[2]))"
    ) in code
    assert (
        "tcgen05_grouped_problem_m, tcgen05_grouped_problem_k, cutlass.Int32(1)"
    ) not in code
    assert (
        "tcgen05_grouped_problem_n, tcgen05_grouped_problem_k, cutlass.Int32(1)"
    ) not in code
    assert (
        "cute.local_tile(tma_tensor_a, (128, 64), "
        "(tcgen05_grouped_cta_tile_idx_m, None))"
    ) in code
    assert (
        "cute.local_tile(tma_tensor_b, (64, 64), "
        "(tcgen05_grouped_cta_tile_idx_n, None))"
    ) in code
    assert (
        "cute.local_tile(tma_tensor_a, (128, 64), "
        "(tcgen05_grouped_cta_tile_idx_m, None, 0))"
    ) not in code
    assert (
        "cute.local_tile(tma_tensor_b, (64, 64), "
        "(tcgen05_grouped_cta_tile_idx_n, None, 0))"
    ) not in code
    assert (
        "cute.local_tile(tcgen05_tma_store_tensor, (128, 64), "
        "(tcgen05_grouped_cta_tile_idx_m, tcgen05_grouped_cta_tile_idx_n, 0))"
    ) in code
    assert "tcgen05_tiled_copy_r2g" not in code
    assert "tcgen05_store_mask" not in code


def test_rank3_rhs_grouped_static_persistent_dynamic_bk64_no_tail_elides_mn_tma_predicate() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes,
        _make_no_mn_tail_mixed_k_args(),
        _rank3_rhs_grouped_static_direct_bk64_config(),
    )

    grouped_plan = _assert_semantic_k_sizes_codegen_markers(
        code,
        expected_bn=64,
        expected_bk=64,
        expected_k_total=192,
        allow_dynamic_tensormap_bases=True,
    )
    assert grouped_plan["grouped_static_has_m_tail"] is False
    assert grouped_plan["grouped_static_has_n_tail"] is False
    assert grouped_plan["dynamic_ab_tensormaps"] is True
    assert grouped_plan["dynamic_d_tensormap"] is True
    assert grouped_plan["direct_pointer_metadata"] is True
    assert (
        "tcgen05_grouped_cta_tile_idx_m * cutlass.Int32(128) "
        "< tcgen05_grouped_problem_m"
    ) not in code
    assert (
        "tcgen05_grouped_cta_tile_idx_n * cutlass.Int32(64) < tcgen05_grouped_problem_n"
    ) not in code
    assert (
        "for tile_offset_2 in cutlass.range(cutlass.Int32(0), "
        "tcgen05_grouped_problem_k, cutlass.Int32(64), unroll=1):"
    ) in code
    assert "tile_offset_2 < tcgen05_grouped_problem_k" in code
    assert "tcgen05_tma_initial_full_tile" not in code
    assert "tma_desc_ptr=tcgen05_grouped_tensormap_a_desc_ptr" in code
    assert "tma_desc_ptr=tcgen05_grouped_tensormap_b_desc_ptr" in code
    assert "tcgen05_tiled_copy_r2g" not in code
    assert "tcgen05_store_mask" not in code


def test_rank3_rhs_grouped_static_persistent_codegen_direct_pointer_metadata() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes,
        _make_documented_mixed_k_args(),
        _rank3_rhs_grouped_static_direct_bk64_config(),
    )

    grouped_plan = _assert_semantic_k_sizes_codegen_markers(
        code,
        expected_bn=64,
        expected_bk=64,
        expected_k_total=1536,
        allow_dynamic_tensormap_bases=True,
    )
    assert grouped_plan["dynamic_ab_tensormaps"] is True
    assert grouped_plan["dynamic_d_tensormap"] is True
    assert grouped_plan["direct_pointer_metadata"] is True
    assert grouped_plan["direct_pointers_arg"] == "tcgen05_grouped_direct_pointers"
    assert grouped_plan["direct_strides_arg"] == "tcgen05_grouped_direct_strides"
    assert code.count("cute.nvgpu.cpasync.prefetch_descriptor(") == 3
    assert (
        "if tcgen05_tma_warp:\n"
        "        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)\n"
        "        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)"
    ) in code
    assert (
        "if tcgen05_warp_idx == cutlass.Int32(0):\n"
        "        cute.nvgpu.cpasync.prefetch_descriptor(tcgen05_tma_store_atom)"
    ) in code

    assert "tcgen05_grouped_direct_pointers" in code
    assert "tcgen05_grouped_direct_strides" in code
    assert "tcgen05_grouped_tensormap_a_addr = (" in code
    assert "tcgen05_grouped_tensormap_b_addr = (" in code
    assert "tcgen05_grouped_tensormap_a_stride_m = (" in code
    assert "tcgen05_grouped_tensormap_b_stride_n = (" in code
    assert (
        "tcgen05_grouped_tensormap_a_base = "
        "cute.make_ptr(cutlass.Float16, "
        "cutlass.Int64(tcgen05_grouped_tensormap_a_addr), "
        "cute.AddressSpace.gmem)"
    ) in code
    assert (
        "tcgen05_grouped_tensormap_b_base = "
        "cute.make_ptr(cutlass.Float16, "
        "cutlass.Int64(tcgen05_grouped_tensormap_b_addr), "
        "cute.AddressSpace.gmem)"
    ) in code
    assert "tcgen05_grouped_d_tensormap_addr = (" in code
    assert "tcgen05_grouped_d_tensormap_stride_m = (" in code
    assert (
        "tcgen05_grouped_d_tensormap_base = "
        "cute.make_ptr(cutlass.Float16, "
        "cutlass.Int64(tcgen05_grouped_d_tensormap_addr), "
        "cute.AddressSpace.gmem)"
    ) in code
    assert "tcgen05_grouped_tensormap_a_base = a.iterator +" not in code
    assert "tcgen05_grouped_tensormap_b_base = b_grouped.iterator +" not in code
    assert "tcgen05_grouped_d_tensormap_base = out.iterator +" not in code
    assert "_cutlass_grouped_gemm_kernel" not in code
    assert "GroupedGemmKernel" not in code
    assert "blackwell_grouped_gemm_nt" not in code


def test_rank3_rhs_grouped_static_persistent_dynamic_bk64_accepts_ab4() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    config = _rank3_rhs_grouped_static_dynamic_bk64_config()
    config.config["tcgen05_ab_stages"] = 4
    with patch.object(
        CuteTcgen05Config,
        "per_cta_raw_smem_cap_bytes",
        return_value=232448,
    ):
        code = _code_for(
            _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes,
            _make_documented_mixed_k_args(),
            config,
        )

    assert "PipelineTmaUmma.create(num_stages=4" in code
    assert "StaticPersistentGroupTileScheduler.create" in code
    assert "_cutlass_grouped_gemm_kernel" not in code
    assert "GroupedGemmKernel" not in code
    assert "virtual_pid" not in code
    assert "tcgen05_rhs_safe_group" not in code
    assert "tcgen05_grouped_ab_tensormaps" in code
    assert "tcgen05_grouped_d_tensormap_manager" in code
    plans = _wrapper_plans_from_code(code)
    grouped_plan = next(
        plan for plan in plans if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert grouped_plan["dynamic_ab_tensormaps"] is True
    assert grouped_plan["dynamic_d_tensormap"] is True
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["ab_stage_count"] == 4
    assert ab_plan["dynamic_ab_tensormaps"] is True
    d_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_d_tma")
    assert d_plan["c_stage_count"] == 2
    assert d_plan["rank3_mnl_tensor"] is True


def test_rank3_rhs_grouped_static_persistent_rejects_ab4_without_dynamic_bk64() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    config = _rank3_rhs_grouped_static_config_bn64()
    config.config["tcgen05_ab_stages"] = 4
    with pytest.raises(helion.exc.InvalidConfig, match="tcgen05_ab_stages"):
        _code_for(
            _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes,
            _make_documented_mixed_k_args(),
            config,
        )


def test_rank3_rhs_grouped_static_persistent_rejects_ab4_scheduler_variant() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    config = _rank3_rhs_grouped_static_dynamic_bk64_config()
    config.config["tcgen05_ab_stages"] = 4
    config.config[TCGEN05_STRATEGY_CONFIG_KEY] = (
        Tcgen05Strategy.ROLE_LOCAL_WITH_SCHEDULER.value
    )
    with pytest.raises(helion.exc.InvalidConfig, match="tcgen05_ab_stages"):
        _code_for(
            _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes,
            _make_documented_mixed_k_args(),
            config,
        )


@pytest.mark.parametrize(
    ("k", "expected_bk"),
    [
        (16, 16),
        (32, 32),
        (64, 64),
        (96, 32),
        (160, 32),
        (192, 64),
    ],
)
def test_rank3_rhs_grouped_static_persistent_compiler_seed_reaches_common_k(
    k: int,
    expected_bk: int,
) -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA compiler seed test needs CUDA fake inputs")

    with (
        patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
        patch_cute_mma_support(),
    ):
        bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(
            _make_mn_tail_grouped_args(k=k)
        )

    spec = bound.config_spec
    assert "cute_tcgen05_grouped_static_common_k" in spec.autotuner_heuristics
    assert spec.compiler_default_config is None
    raw_seeds = [
        config.config
        for config in spec.compiler_seed_configs
        if config.config.get(TCGEN05_GROUPED_STATIC_PERSISTENT_CONFIG_KEY)
    ]
    assert raw_seeds == [
        {
            "block_sizes": [128, 64, expected_bk],
            "loop_orders": [[0, 1]],
            "l2_groupings": [1],
            "num_warps": 8,
            "num_stages": 2,
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 1,
            "tcgen05_cluster_n": 1,
            TCGEN05_GROUPED_STATIC_PERSISTENT_CONFIG_KEY: True,
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                Tcgen05PersistenceModel.STATIC_PERSISTENT.value
            ),
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": 2,
            "tcgen05_num_epi_warps": 4,
        }
    ]

    normalized_seeds = [
        config.config
        for _flat, config in ConfigGeneration(spec).seed_flat_config_pairs()
        if config.config.get(TCGEN05_GROUPED_STATIC_PERSISTENT_CONFIG_KEY)
    ]
    assert len(normalized_seeds) == 1
    normalized_seed = normalized_seeds[0]
    assert normalized_seed["block_sizes"] == [128, 64, expected_bk]
    assert normalized_seed[TCGEN05_GROUPED_STATIC_PERSISTENT_CONFIG_KEY] is True
    assert normalized_seed["tcgen05_c_stages"] == 2
    assert normalized_seed["pid_type"] == "persistent_interleaved"
    assert (
        normalized_seed[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY]
        == Tcgen05PersistenceModel.STATIC_PERSISTENT.value
    )
    default_config = spec.default_config().config
    assert TCGEN05_GROUPED_STATIC_PERSISTENT_CONFIG_KEY not in default_config
    assert TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY not in default_config

    code = _code_for(
        _rank3_rhs_grouped_nt_with_mn_tails,
        _make_mn_tail_grouped_args(k=k),
        helion.Config.from_dict(normalized_seed),
    )
    assert "StaticPersistentGroupTileScheduler.create" in code
    assert "_cutlass_grouped_gemm_kernel" not in code
    assert "GroupedGemmKernel" not in code
    assert "virtual_pid" not in code
    grouped_plan = next(
        plan
        for plan in _wrapper_plans_from_code(code)
        if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert grouped_plan["bk"] == expected_bk
    assert grouped_plan["k_total_size"] == k
    assert grouped_plan.get("dynamic_ab_tensormaps") is not True
    ab_plan = next(
        plan
        for plan in _wrapper_plans_from_code(code)
        if plan["kind"] == "tcgen05_ab_tma"
    )
    assert ab_plan["bk"] == expected_bk
    assert ab_plan.get("dynamic_ab_tensormaps") is not True


def test_rank3_rhs_grouped_static_persistent_compiler_seed_reaches_dynamic_bk64_mixed_k() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA compiler seed test needs CUDA fake inputs")

    with (
        patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
        patch_cute_mma_support(),
    ):
        bound = _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes.bind(
            _make_documented_mixed_k_args()
        )

    spec = bound.config_spec
    assert "cute_tcgen05_grouped_static_common_k" not in spec.autotuner_heuristics
    assert "cute_tcgen05_grouped_dynamic_bk64" in spec.autotuner_heuristics
    assert spec.compiler_default_config is None
    raw_seeds = [
        config.config
        for config in spec.compiler_seed_configs
        if config.config.get(TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY)
    ]
    assert raw_seeds == [
        {
            "block_sizes": [128, 64, 64],
            "loop_orders": [[0, 1]],
            "l2_groupings": [1],
            "num_warps": 8,
            "num_stages": 2,
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 1,
            "tcgen05_cluster_n": 1,
            TCGEN05_GROUPED_STATIC_PERSISTENT_CONFIG_KEY: True,
            TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY: True,
            TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY: 3,
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: (
                Tcgen05PersistenceModel.STATIC_PERSISTENT.value
            ),
            "tcgen05_ab_stages": 4,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": 2,
            "tcgen05_num_epi_warps": 4,
        }
    ]

    normalized_seeds = [
        config.config
        for _flat, config in ConfigGeneration(spec).seed_flat_config_pairs()
        if config.config.get(TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY)
    ]
    assert len(normalized_seeds) == 1
    normalized_seed = normalized_seeds[0]
    assert normalized_seed["block_sizes"] == [128, 64, 64]
    assert normalized_seed[TCGEN05_GROUPED_STATIC_PERSISTENT_CONFIG_KEY] is True
    assert normalized_seed[TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY] is True
    assert normalized_seed["tcgen05_ab_stages"] == 4
    assert normalized_seed["tcgen05_c_stages"] == 2
    assert normalized_seed["pid_type"] == "persistent_interleaved"
    assert (
        normalized_seed[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY]
        == Tcgen05PersistenceModel.STATIC_PERSISTENT.value
    )
    default_config = spec.default_config().config
    assert TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY not in default_config
    assert TCGEN05_GROUPED_STATIC_PERSISTENT_CONFIG_KEY not in default_config

    code = _code_for(
        _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes,
        _make_documented_mixed_k_args(),
        helion.Config.from_dict(normalized_seed),
    )
    assert "PipelineTmaUmma.create(num_stages=4" in code
    assert "StaticPersistentGroupTileScheduler.create" in code
    assert "_cutlass_grouped_gemm_kernel" not in code
    assert "GroupedGemmKernel" not in code
    assert "virtual_pid" not in code
    assert "tcgen05_grouped_ab_tensormaps" in code
    assert "tcgen05_grouped_d_tensormap_manager" in code
    grouped_plan = next(
        plan
        for plan in _wrapper_plans_from_code(code)
        if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert grouped_plan["dynamic_ab_tensormaps"] is True
    assert grouped_plan["dynamic_d_tensormap"] is True
    ab_plan = next(
        plan
        for plan in _wrapper_plans_from_code(code)
        if plan["kind"] == "tcgen05_ab_tma"
    )
    assert ab_plan["dynamic_ab_tensormaps"] is True
    assert ab_plan["ab_stage_count"] == 4
    d_plan = next(
        plan
        for plan in _wrapper_plans_from_code(code)
        if plan["kind"] == "tcgen05_d_tma"
    )
    assert d_plan["rank3_mnl_tensor"] is True


def test_rank3_rhs_grouped_static_persistent_compiler_seed_reaches_dynamic_bk64_blackwell_mixed_k() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA compiler seed test needs CUDA fake inputs")

    with (
        patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
        patch_cute_mma_support(),
    ):
        bound = _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes.bind(
            _make_blackwell_mixed_k_args()
        )

    spec = bound.config_spec
    assert spec.allowed_pid_types == ("flat",)
    assert "cute_tcgen05_grouped_static_common_k" not in spec.autotuner_heuristics
    assert "cute_tcgen05_grouped_dynamic_bk64" in spec.autotuner_heuristics
    raw_seeds = [
        config.config
        for config in spec.compiler_seed_configs
        if config.config.get(TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY)
    ]
    assert len(raw_seeds) == 1
    raw_seed = raw_seeds[0]
    assert raw_seed["block_sizes"] == [128, 64, 64]
    assert raw_seed["pid_type"] == "persistent_interleaved"
    assert raw_seed[TCGEN05_GROUPED_STATIC_PERSISTENT_CONFIG_KEY] is True
    assert raw_seed[TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY] is True
    assert raw_seed[TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY] == 3
    assert raw_seed["tcgen05_ab_stages"] == 4
    assert (
        raw_seed[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY]
        == Tcgen05PersistenceModel.STATIC_PERSISTENT.value
    )

    config_gen = ConfigGeneration(spec)
    normalized_seeds = [
        config.config
        for _flat, config in config_gen.seed_flat_config_pairs()
        if config.config.get(TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY)
    ]
    assert len(normalized_seeds) == 1
    normalized_seed = normalized_seeds[0]
    assert normalized_seed["block_sizes"] == [128, 64, 64]
    assert normalized_seed["pid_type"] == "persistent_interleaved"
    assert normalized_seed[TCGEN05_GROUPED_STATIC_PERSISTENT_CONFIG_KEY] is True
    assert normalized_seed[TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY] is True
    assert normalized_seed[TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY] == 3
    assert normalized_seed["tcgen05_ab_stages"] == 4
    assert (
        normalized_seed[TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY]
        == Tcgen05PersistenceModel.STATIC_PERSISTENT.value
    )

    for flat_config in (
        config_gen.default_flat(),
        *(config_gen.random_flat() for _ in range(4)),
    ):
        config = config_gen.unflatten(flat_config).config
        assert config["pid_type"] == "flat"
        assert TCGEN05_GROUPED_STATIC_PERSISTENT_CONFIG_KEY not in config
        assert TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY not in config

    reserved_fragment = spec._flat_fields()[
        TCGEN05_GROUPED_STATIC_RESERVED_SMS_CONFIG_KEY
    ]
    assert (
        reserved_fragment.choices == TCGEN05_GROUPED_STATIC_RESERVED_SMS_SEARCH_CHOICES
    )


def test_rank3_rhs_grouped_static_persistent_compiler_seed_requires_exact_k_proof() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA compiler seed test needs CUDA fake inputs")

    with (
        patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
        patch_cute_mma_support(),
    ):
        bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(
            _make_mn_tail_grouped_args(k=64)
        )

    spec = bound.config_spec
    assert "cute_tcgen05_grouped_dynamic_bk64" not in spec.autotuner_heuristics
    assert not [
        config
        for config in spec.compiler_seed_configs
        if config.config.get(TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY)
    ]
    assert not [
        config
        for _flat, config in ConfigGeneration(spec).seed_flat_config_pairs()
        if config.config.get(TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY)
    ]


def test_rank3_rhs_grouped_static_persistent_compiler_seed_rejects_swapped_grid() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA compiler seed test needs CUDA fake inputs")

    with (
        patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
        patch_cute_mma_support(),
    ):
        bound = _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes.bind(
            _make_documented_mixed_k_args()
        )

    spec = bound.config_spec
    device_ir = bound.host_function.device_ir
    assert "cute_tcgen05_grouped_dynamic_bk64" in spec.autotuner_heuristics
    assert len(device_ir.grid_block_ids) == 1
    root_grid = tuple(device_ir.grid_block_ids[0])
    assert len(root_grid) == 2
    assert root_grid[0] != root_grid[1]

    try:
        device_ir.grid_block_ids[0] = [root_grid[1], root_grid[0]]
        spec.compiler_seed_configs = compiler_seed_configs(bound.env, device_ir)

        assert "cute_tcgen05_grouped_dynamic_bk64" not in spec.autotuner_heuristics
        assert not [
            config
            for config in spec.compiler_seed_configs
            if config.config.get(TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY)
        ]
        assert not [
            config
            for _flat, config in ConfigGeneration(spec).seed_flat_config_pairs()
            if config.config.get(TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY)
        ]
    finally:
        device_ir.grid_block_ids[0] = list(root_grid)
        spec.compiler_seed_configs = compiler_seed_configs(bound.env, device_ir)


def test_rank3_rhs_grouped_static_persistent_common_k_compiler_seed_rejects_swapped_grid() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA compiler seed test needs CUDA fake inputs")

    with (
        patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
        patch_cute_mma_support(),
    ):
        bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(
            _make_mn_tail_grouped_args(k=96)
        )

    spec = bound.config_spec
    device_ir = bound.host_function.device_ir
    assert "cute_tcgen05_grouped_static_common_k" in spec.autotuner_heuristics
    assert len(device_ir.grid_block_ids) == 1
    root_grid = tuple(device_ir.grid_block_ids[0])
    assert len(root_grid) == 2
    assert root_grid[0] != root_grid[1]

    try:
        device_ir.grid_block_ids[0] = [root_grid[1], root_grid[0]]
        spec.compiler_seed_configs = compiler_seed_configs(bound.env, device_ir)

        assert "cute_tcgen05_grouped_static_common_k" not in spec.autotuner_heuristics
        assert not [
            config
            for config in spec.compiler_seed_configs
            if config.config.get(TCGEN05_GROUPED_STATIC_PERSISTENT_CONFIG_KEY)
        ]
        assert not [
            config
            for _flat, config in ConfigGeneration(spec).seed_flat_config_pairs()
            if config.config.get(TCGEN05_GROUPED_STATIC_PERSISTENT_CONFIG_KEY)
        ]
    finally:
        device_ir.grid_block_ids[0] = list(root_grid)
        spec.compiler_seed_configs = compiler_seed_configs(bound.env, device_ir)


def _assert_no_grouped_dynamic_bk64_compiler_seed(
    kernel: object,
    args: tuple[torch.Tensor, ...],
) -> None:
    with (
        patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
        patch_cute_mma_support(),
    ):
        bound = kernel.bind(args)

    spec = bound.config_spec
    assert "cute_tcgen05_grouped_dynamic_bk64" not in spec.autotuner_heuristics
    assert not [
        config
        for config in spec.compiler_seed_configs
        if config.config.get(TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY)
    ]
    assert not [
        config
        for _flat, config in ConfigGeneration(spec).seed_flat_config_pairs()
        if config.config.get(TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY)
    ]


@pytest.mark.parametrize(
    ("kernel", "args_fn"),
    [
        (_rank3_rhs_grouped_nt_bad_k_sizes_missing_b_mask, _make_per_group_k64_args),
        (_rank3_rhs_grouped_nt_bad_k_sizes_arbitrary_mask, _make_per_group_k64_args),
        (
            _rank3_rhs_grouped_nt_bad_k_sizes_without_source_proof,
            _make_per_group_k64_args,
        ),
        (
            _rank3_rhs_grouped_nt_bad_k_sizes_group_provenance,
            _make_mismatched_k_mask_group_k64_args,
        ),
        (_rank3_rhs_grouped_nt_bad_k_sizes_extra_loop_use, _make_per_group_k64_args),
        (
            _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes,
            _make_documented_mixed_k_non_row_major_a_args,
        ),
        (
            _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes,
            _make_documented_mixed_k_non_k_contiguous_b_args,
        ),
    ],
)
def test_rank3_rhs_grouped_static_persistent_compiler_seed_rejects_unsafe_proof_shapes(
    kernel: object,
    args_fn: Any,
) -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA compiler seed test needs CUDA fake inputs")

    _assert_no_grouped_dynamic_bk64_compiler_seed(kernel, args_fn())


def test_rank3_rhs_grouped_static_persistent_compiler_seed_schema_is_seed_only() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA compiler seed test needs CUDA fake inputs")

    with (
        patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
        patch_cute_mma_support(),
    ):
        bound = _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes.bind(
            _make_documented_mixed_k_args()
        )

    spec = bound.config_spec
    config_gen = ConfigGeneration(spec)
    assert TCGEN05_GROUPED_STATIC_PERSISTENT_CONFIG_KEY in (
        config_gen._key_to_flat_indices
    )
    assert TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY in (
        config_gen._key_to_flat_indices
    )
    for flat_config in (
        config_gen.default_flat(),
        *(config_gen.random_flat() for _ in range(4)),
    ):
        config = config_gen.unflatten(flat_config).config
        assert TCGEN05_GROUPED_STATIC_PERSISTENT_CONFIG_KEY not in config
        assert TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY not in config


def test_rank3_rhs_grouped_static_persistent_common_k_compiler_seed_schema_is_seed_only() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA compiler seed test needs CUDA fake inputs")

    with (
        patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False),
        patch_cute_mma_support(),
    ):
        bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(
            _make_mn_tail_grouped_args(k=96)
        )

    spec = bound.config_spec
    config_gen = ConfigGeneration(spec)
    assert TCGEN05_GROUPED_STATIC_PERSISTENT_CONFIG_KEY in (
        config_gen._key_to_flat_indices
    )
    assert TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY not in (
        config_gen._key_to_flat_indices
    )
    for flat_config in (
        config_gen.default_flat(),
        *(config_gen.random_flat() for _ in range(4)),
    ):
        config = config_gen.unflatten(flat_config).config
        assert TCGEN05_GROUPED_STATIC_PERSISTENT_CONFIG_KEY not in config
        assert TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY not in config


def test_rank3_rhs_grouped_static_persistent_default_bn128_path_stays_non_dynamic() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    config = _rank3_rhs_grouped_static_config()
    assert config.config["block_sizes"] == [128, 128, 128]
    assert TCGEN05_GROUPED_DYNAMIC_AB_TENSORMAPS_CONFIG_KEY not in config.config
    code = _code_for(_rank3_rhs_grouped_nt, _make_args(), config)

    grouped_plan = next(
        plan
        for plan in _wrapper_plans_from_code(code)
        if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert grouped_plan.get("dynamic_ab_tensormaps") is not True
    ab_plan = next(
        plan
        for plan in _wrapper_plans_from_code(code)
        if plan["kind"] == "tcgen05_ab_tma"
    )
    assert ab_plan.get("dynamic_ab_tensormaps") is not True


def test_rank3_rhs_grouped_static_persistent_dynamic_bk64_rejects_missing_k_proof() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    with pytest.raises(helion.exc.BackendUnsupported, match="exact_k_sizes"):
        _code_for(
            _rank3_rhs_grouped_nt_with_mn_tails,
            _make_mn_tail_grouped_args(k=64),
            _rank3_rhs_grouped_static_dynamic_bk64_config(),
        )


def test_rank3_rhs_grouped_static_persistent_bk64_accepts_common_k() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_with_mn_tails,
        _make_mn_tail_grouped_args(k=64),
        _rank3_rhs_grouped_static_config_bn64_bk(64),
    )

    grouped_plan = _assert_semantic_mn_tail_codegen_markers(code, expected_bn=64)
    assert grouped_plan["bk"] == 64
    assert grouped_plan["k_total_size"] == 64
    assert grouped_plan.get("dynamic_ab_tensormaps") is not True


def test_rank3_rhs_grouped_static_persistent_bk64_without_dynamic_rejects_mixed_k() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    with pytest.raises(
        helion.exc.BackendUnsupported,
        match="block_k_64_static_common_or_dynamic",
    ):
        _code_for(
            _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes,
            _make_documented_mixed_k_args(),
            _rank3_rhs_grouped_static_config_bn64_bk(64),
        )


@pytest.mark.parametrize(
    ("args_fn", "match"),
    [
        (
            _make_documented_mixed_k_non_row_major_a_args,
            "tcgen05_grouped_static_persistent=True",
        ),
        (
            _make_documented_mixed_k_non_k_contiguous_b_args,
            "tcgen05_grouped_static_persistent=True",
        ),
    ],
)
def test_rank3_rhs_grouped_static_persistent_dynamic_bk64_rejects_bad_strides(
    args_fn: Any,
    match: str,
) -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    try:
        code = _code_for(
            _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes,
            args_fn(),
            _rank3_rhs_grouped_static_dynamic_bk64_config(),
        )
    except helion.exc.BackendUnsupported as error:
        _assert_backend_unsupported_match(error, match)
        return

    _assert_no_grouped_static_codegen_leak(code)


@pytest.mark.parametrize(
    ("kernel", "match"),
    [
        (
            _rank3_rhs_grouped_nt_bad_k_sizes_missing_b_mask,
            "tcgen05_grouped_static_persistent=True",
        ),
        (
            _rank3_rhs_grouped_nt_bad_k_sizes_arbitrary_mask,
            "tcgen05_grouped_static_persistent=True",
        ),
        (
            _rank3_rhs_grouped_nt_bad_k_sizes_without_source_proof,
            "common_k_block_pair_allowlisted",
        ),
    ],
)
def test_rank3_rhs_grouped_static_persistent_rejects_unsafe_k_sizes_source(
    kernel: object,
    match: str,
) -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    try:
        code = _code_for(
            kernel,
            _make_per_group_k_args(),
            _rank3_rhs_grouped_static_config_bn64_bk(16),
        )
    except helion.exc.BackendUnsupported as error:
        _assert_backend_unsupported_match(error, match)
        return

    _assert_no_grouped_static_codegen_leak(code)


def test_rank3_rhs_grouped_static_persistent_rejects_mismatched_k_mask_group() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    try:
        code = _code_for(
            _rank3_rhs_grouped_nt_bad_k_sizes_group_provenance,
            _make_mismatched_k_mask_group_args(),
            _rank3_rhs_grouped_static_config_bn64_bk(16),
        )
    except helion.exc.BackendUnsupported as error:
        _assert_backend_unsupported_match(
            error,
            "tcgen05_grouped_static_persistent=True",
        )
        return

    _assert_no_grouped_static_codegen_leak(code)


def test_rank3_rhs_grouped_static_persistent_rejects_extra_k_size_loop_use() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    try:
        config = _rank3_rhs_grouped_static_config_bn64_bk(16)
        config.config["block_sizes"] = [128, 64, 16, 16]
        code = _code_for(
            _rank3_rhs_grouped_nt_bad_k_sizes_extra_loop_use,
            _make_per_group_k_args(),
            config,
        )
    except helion.exc.BackendUnsupported as error:
        _assert_backend_unsupported_match(error, "collective_operand_loads")
        return

    _assert_no_grouped_static_codegen_leak(code)


def test_rank3_rhs_grouped_static_persistent_rejects_mismatched_common_k() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    with pytest.raises(helion.exc.BackendUnsupported, match="common_k"):
        _code_for(
            _rank3_rhs_grouped_nt,
            _make_mismatched_common_k_args(),
            _rank3_rhs_grouped_static_config_bn64_bk(32),
        )


@pytest.mark.parametrize(("k", "block_k"), [(24, 16), (48, 32)])
def test_rank3_rhs_grouped_static_persistent_rejects_k_tail(
    k: int, block_k: int
) -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    with pytest.raises(
        helion.exc.BackendUnsupported,
        match="common_k_block_multiple|static_full_tiles",
    ):
        _code_for(
            _rank3_rhs_grouped_nt_with_mn_tails,
            _make_mn_tail_grouped_args(k=k),
            _rank3_rhs_grouped_static_config_bn64_bk(block_k),
        )


def test_rank3_rhs_grouped_static_persistent_rejects_unallowlisted_common_k48_bk16() -> (
    None
):
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    with pytest.raises(
        helion.exc.BackendUnsupported,
        match="common_k_block_pair_allowlisted",
    ):
        _code_for(
            _rank3_rhs_grouped_nt_with_mn_tails,
            _make_mn_tail_grouped_args(k=48),
            _rank3_rhs_grouped_static_config_bn64_bk(16),
        )


def test_rank3_rhs_grouped_static_persistent_bk16_rejects_float32() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt_with_mn_tails,
        _make_mn_tail_grouped_args(k=16, dtype=torch.float32),
        _rank3_rhs_grouped_static_config_bn64_bk(16),
    )

    _assert_no_grouped_static_codegen_leak(code)


def test_rank3_rhs_grouped_static_persistent_bk16_rejects_unpadded_n160() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    with pytest.raises(helion.exc.BackendUnsupported, match="static_full_tiles"):
        _code_for(
            _rank3_rhs_grouped_nt,
            _make_all_full_grouped_args(
                groups=2,
                n=160,
                k=16,
                dtype=torch.float16,
            ),
            _rank3_rhs_grouped_static_config_bn64_bk(16),
        )


def test_rank3_rhs_grouped_static_persistent_rejects_small_k_with_default_bk() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    with pytest.raises(helion.exc.BackendUnsupported, match="static_full_tiles"):
        _code_for(
            _rank3_rhs_grouped_nt_with_mn_tails,
            _make_mn_tail_grouped_args(k=16),
            _rank3_rhs_grouped_static_config_bn64(),
        )


def test_rank3_rhs_grouped_static_persistent_rejects_k_tail_without_proof() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    with pytest.raises(helion.exc.BackendUnsupported, match="static_full_tiles"):
        _code_for(
            _rank3_rhs_grouped_nt_with_mn_tails,
            _make_mn_tail_grouped_args(k=192),
            _rank3_rhs_grouped_static_config_bn64(),
        )


def test_rank3_rhs_grouped_static_default_ab3_for_eligible_k() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt,
        _make_all_full_grouped_args(k=384),
        _rank3_rhs_grouped_static_default_ab_config(),
    )

    assert "PipelineTmaUmma.create(num_stages=3" in code
    plans = _wrapper_plans_from_code(code)
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["ab_stage_count"] == 3


def test_rank3_rhs_grouped_static_default_ab3_for_k1536() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt,
        _make_all_full_grouped_args(k=1536),
        _rank3_rhs_grouped_static_default_ab_config(),
    )

    assert "PipelineTmaUmma.create(num_stages=3" in code
    plans = _wrapper_plans_from_code(code)
    grouped_plan = next(
        plan for plan in plans if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert grouped_plan["k_total_size"] == 1536
    assert grouped_plan["bk"] == 128
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["ab_stage_count"] == 3


def test_rank3_rhs_grouped_static_default_ab2_for_explicit_c4() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    config = _rank3_rhs_grouped_static_default_ab_config()
    config.config["tcgen05_c_stages"] = 4
    code = _code_for(
        _rank3_rhs_grouped_nt,
        _make_all_full_grouped_args(k=384),
        config,
    )

    assert "PipelineTmaUmma.create(num_stages=2" in code
    assert "PipelineTmaUmma.create(num_stages=3" not in code
    plans = _wrapper_plans_from_code(code)
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["ab_stage_count"] == 2
    d_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_d_tma")
    assert d_plan["c_stage_count"] == 4


def test_examples_grouped_static_default_ab3_for_real_b200_path() -> None:
    _require_tcgen05_runtime_test()

    bound, config = _example_grouped_static_bound()

    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        code = bound.to_triton_code(config)

    assert bound.env.config_spec._tcgen05_ab_stages_three_fits_for_target(
        dtype_bytes=2,
        device=DEVICE,
        bm=128,
        bn=128,
        bk=128,
        cluster_m=1,
    )
    assert "PipelineTmaUmma.create(num_stages=3" in code
    assert "PipelineTmaUmma.create(num_stages=2" not in code
    plans = _wrapper_plans_from_code(code)
    grouped_plan = next(
        plan for plan in plans if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    assert grouped_plan["k_total_size"] == 384
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["ab_stage_count"] == 3


def test_examples_grouped_static_default_ab3_after_normalize() -> None:
    _require_tcgen05_runtime_test()

    bound, config = _example_grouped_static_bound()
    bound.env.config_spec.normalize(config)
    assert config.config["tcgen05_ab_stages"] == 2
    assert config.config[TCGEN05_AB_STAGES_AUTO_CONFIG_KEY] is True

    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        code = bound.to_triton_code(config)

    assert "PipelineTmaUmma.create(num_stages=3" in code
    assert "PipelineTmaUmma.create(num_stages=2" not in code
    plans = _wrapper_plans_from_code(code)
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["ab_stage_count"] == 3


def test_examples_grouped_static_raw_dict_normalize_preserves_ab_semantics() -> None:
    _require_tcgen05_runtime_test()

    bound, omitted_config = _example_grouped_static_bound()
    explicit_config = _rank3_rhs_grouped_static_config()
    cases = (
        (dict(omitted_config.config), True, 3),
        (dict(explicit_config.config), False, 2),
    )

    for raw, auto, stage_count in cases:
        bound.env.config_spec.normalize(raw)
        assert raw[TCGEN05_AB_STAGES_AUTO_CONFIG_KEY] is auto

        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            code = bound.to_triton_code(raw)

        assert f"PipelineTmaUmma.create(num_stages={stage_count}" in code
        assert f"PipelineTmaUmma.create(num_stages={5 - stage_count}" not in code
        plans = _wrapper_plans_from_code(code)
        ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
        assert ab_plan["ab_stage_count"] == stage_count


def test_examples_grouped_static_repr_replays_ab_semantics() -> None:
    _require_tcgen05_runtime_test()

    bound, omitted_config = _example_grouped_static_bound()
    explicit_config = _rank3_rhs_grouped_static_config()
    for config, auto, stage_count in (
        (omitted_config, True, 3),
        (explicit_config, False, 2),
    ):
        bound.env.config_spec.normalize(config)
        assert config.config[TCGEN05_AB_STAGES_AUTO_CONFIG_KEY] is auto
        decorator = bound.format_kernel_decorator(config, bound.settings)
        assert f"{TCGEN05_AB_STAGES_AUTO_CONFIG_KEY}={auto}" in decorator
        replayed = eval(repr(config), {"helion": helion})

        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            code = bound.to_triton_code(replayed)

        assert f"PipelineTmaUmma.create(num_stages={stage_count}" in code
        assert f"PipelineTmaUmma.create(num_stages={5 - stage_count}" not in code
        plans = _wrapper_plans_from_code(code)
        ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
        assert ab_plan["ab_stage_count"] == stage_count


def test_examples_grouped_static_default_ab3_does_not_widen_search_space() -> None:
    _require_tcgen05_runtime_test()

    bound, config = _example_grouped_static_bound()
    before = bound.env.config_spec._tcgen05_optional_fragments(for_search=True)[
        "tcgen05_ab_stages"
    ].high

    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        code = bound.to_triton_code(config)

    after = bound.env.config_spec._tcgen05_optional_fragments(for_search=True)[
        "tcgen05_ab_stages"
    ].high
    assert before == 2
    assert after == 2
    assert "PipelineTmaUmma.create(num_stages=3" in code


def test_examples_grouped_static_default_ab3_survives_dict_normalize_minimize_json() -> (
    None
):
    _require_tcgen05_runtime_test()

    bound, config = _example_grouped_static_bound()
    bound.env.config_spec.normalize(config.config)
    assert config.config["tcgen05_ab_stages"] == 2
    assert config.config[TCGEN05_AB_STAGES_AUTO_CONFIG_KEY] is True

    minimized = config.minimize(bound.env.config_spec)
    assert "tcgen05_ab_stages" not in minimized.config
    assert minimized.config[TCGEN05_AB_STAGES_AUTO_CONFIG_KEY] is True
    restored = helion.Config.from_json(minimized.to_json())

    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        code = bound.to_triton_code(restored)

    assert "PipelineTmaUmma.create(num_stages=3" in code
    assert "PipelineTmaUmma.create(num_stages=2" not in code
    plans = _wrapper_plans_from_code(code)
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["ab_stage_count"] == 3


def test_examples_grouped_static_default_ab3_survives_pickle_and_deepcopy() -> None:
    _require_tcgen05_runtime_test()

    bound, omitted_config = _example_grouped_static_bound()
    bound.env.config_spec.normalize(omitted_config)
    explicit_config = _rank3_rhs_grouped_static_config()
    bound.env.config_spec.normalize(explicit_config)

    cases = (
        (pickle.loads(pickle.dumps(omitted_config)), 3),
        (copy.deepcopy(omitted_config), 3),
        (pickle.loads(pickle.dumps(explicit_config)), 2),
        (copy.deepcopy(explicit_config), 2),
    )
    for restored, stage_count in cases:
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            code = bound.to_triton_code(restored)
        assert f"PipelineTmaUmma.create(num_stages={stage_count}" in code
        assert f"PipelineTmaUmma.create(num_stages={5 - stage_count}" not in code
        plans = _wrapper_plans_from_code(code)
        ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
        assert ab_plan["ab_stage_count"] == stage_count


def test_examples_grouped_static_raw_json_save_load_preserves_ab_semantics(
    tmp_path: Path,
) -> None:
    _require_tcgen05_runtime_test()

    bound, omitted_config = _example_grouped_static_bound()
    bound.env.config_spec.normalize(omitted_config)
    explicit_config = _rank3_rhs_grouped_static_config()
    bound.env.config_spec.normalize(explicit_config)

    cases = (
        ("omitted", omitted_config, 3),
        ("explicit", explicit_config, 2),
    )
    for label, config, stage_count in cases:
        json_restored = helion.Config.from_json(config.to_json())
        path = tmp_path / f"{label}.json"
        config.save(path)
        load_restored = helion.Config.load(path)

        for restored in (json_restored, load_restored):
            with patch.dict(
                os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False
            ):
                code = bound.to_triton_code(restored)
            assert f"PipelineTmaUmma.create(num_stages={stage_count}" in code
            assert f"PipelineTmaUmma.create(num_stages={5 - stage_count}" not in code
            plans = _wrapper_plans_from_code(code)
            ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
            assert ab_plan["ab_stage_count"] == stage_count


def test_examples_grouped_static_explicit_ab2_survives_minimize_round_trip() -> None:
    _require_tcgen05_runtime_test()

    bound, _default_ab_config = _example_grouped_static_bound()
    raw_config = _rank3_rhs_grouped_static_config()
    normalized_config = _rank3_rhs_grouped_static_config()
    bound.env.config_spec.normalize(normalized_config)
    assert normalized_config.config[TCGEN05_AB_STAGES_AUTO_CONFIG_KEY] is False

    for config in (raw_config, normalized_config):
        minimized = config.minimize(bound.env.config_spec)
        assert "tcgen05_ab_stages" not in minimized.config
        assert minimized.config[TCGEN05_AB_STAGES_AUTO_CONFIG_KEY] is False
        restored = helion.Config.from_json(minimized.to_json())
        assert restored.config[TCGEN05_AB_STAGES_AUTO_CONFIG_KEY] is False

        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            code = bound.to_triton_code(restored)

        assert "PipelineTmaUmma.create(num_stages=2" in code
        assert "PipelineTmaUmma.create(num_stages=3" not in code
        plans = _wrapper_plans_from_code(code)
        ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
        assert ab_plan["ab_stage_count"] == 2


def test_examples_grouped_static_union_update_marks_explicit_ab2() -> None:
    _require_tcgen05_runtime_test()

    bound, config = _example_grouped_static_bound()
    config.config |= {"tcgen05_ab_stages": 2}
    normalized = helion.Config.from_dict(config.config)
    bound.env.config_spec.normalize(normalized)
    assert normalized.config[TCGEN05_AB_STAGES_AUTO_CONFIG_KEY] is False

    with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
        code = bound.to_triton_code(config)

    assert "PipelineTmaUmma.create(num_stages=2" in code
    assert "PipelineTmaUmma.create(num_stages=3" not in code
    plans = _wrapper_plans_from_code(code)
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["ab_stage_count"] == 2


@pytest.mark.parametrize("order", ["explicit_first", "omitted_first"])
def test_examples_grouped_static_compile_cache_keeps_explicit_ab2_and_omitted_ab3(
    order: str,
) -> None:
    _require_tcgen05_runtime_test()

    bound, omitted_config = _example_grouped_static_bound()
    explicit_config = _rank3_rhs_grouped_static_config()
    bound.env.config_spec.normalize(omitted_config)
    bound.env.config_spec.normalize(explicit_config)

    configs = {
        "explicit": explicit_config,
        "omitted": omitted_config,
    }
    labels = (
        ("explicit", "omitted")
        if order == "explicit_first"
        else ("omitted", "explicit")
    )
    paths: dict[str, str] = {}
    codes: dict[str, str] = {}
    for label in labels:
        config = configs[label]
        bound.compile_config(config, allow_print=False)
        path = bound.get_cached_path(config)
        assert path is not None
        paths[label] = path
        codes[label] = Path(path).read_text()

    assert paths["explicit"] != paths["omitted"]
    assert "PipelineTmaUmma.create(num_stages=2" in codes["explicit"]
    assert "PipelineTmaUmma.create(num_stages=3" not in codes["explicit"]
    assert "PipelineTmaUmma.create(num_stages=3" in codes["omitted"]
    assert "PipelineTmaUmma.create(num_stages=2" not in codes["omitted"]


def test_rank3_rhs_grouped_static_default_ab2_for_small_k() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt,
        _make_all_full_grouped_args(k=256),
        _rank3_rhs_grouped_static_default_ab_config(),
    )

    assert "PipelineTmaUmma.create(num_stages=2" in code
    assert "PipelineTmaUmma.create(num_stages=3" not in code
    plans = _wrapper_plans_from_code(code)
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["ab_stage_count"] == 2


def test_rank3_rhs_grouped_static_preserves_explicit_ab2_for_eligible_k() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(
        _rank3_rhs_grouped_nt,
        _make_all_full_grouped_args(k=384),
        _rank3_rhs_grouped_static_config(),
    )

    assert "PipelineTmaUmma.create(num_stages=2" in code
    assert "PipelineTmaUmma.create(num_stages=3" not in code
    plans = _wrapper_plans_from_code(code)
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["ab_stage_count"] == 2


def test_rank3_rhs_non_grouped_static_default_stays_ab2_for_eligible_k() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    config = _rank3_rhs_tma_config()
    config.config.pop("tcgen05_ab_stages", None)
    code = _code_for(
        _rank3_rhs_grouped_nt,
        _make_all_full_grouped_args(k=384),
        config,
    )

    assert "PipelineTmaUmma.create(num_stages=2" in code
    assert "PipelineTmaUmma.create(num_stages=3" not in code
    plans = _wrapper_plans_from_code(code)
    ab_plan = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab_plan["ab_stage_count"] == 2


def test_rank3_rhs_grouped_static_persistent_rejects_partial_output_shape() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    with pytest.raises(helion.exc.BackendUnsupported, match="static_full_tiles"):
        _code_for(
            _rank3_rhs_grouped_nt,
            _make_partial_m_args(),
            _rank3_rhs_grouped_static_config(),
        )


def test_rank3_rhs_grouped_nt_rejects_bad_safe_group_false_branch() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(_rank3_rhs_grouped_nt_bad_false_branch)

    assert "'rhs_rank3_grouped_nt': True" not in code
    assert "tcgen05_rhs_safe_group" not in code


def test_rank3_rhs_grouped_nt_rejects_bad_group_tile_provenance() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(_rank3_rhs_grouped_nt_bad_group_n_tile)

    assert "'rhs_rank3_grouped_nt': True" not in code
    assert "tcgen05_rhs_safe_group" not in code
    assert "tcgen05_rhs_group =" not in code
    assert (
        re.search(
            r"^\s*group_id\s*=.*\.load\(\) if mask_\d+ else",
            code,
            re.MULTILINE,
        )
        is not None
    )


def test_rank3_rhs_grouped_nt_rejects_bad_rhs_n_block_provenance() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    try:
        code = _code_for(_rank3_rhs_grouped_nt_bad_rhs_n_m_tile)
    except helion.exc.BackendUnsupported as error:
        _assert_backend_unsupported_match(error, "tcgen05")
        return

    assert "'rhs_rank3_grouped_nt': True" not in code
    assert "tcgen05_rhs_safe_group" not in code
    assert "tcgen05_rhs_group =" not in code
    assert "b_grouped.iterator" in code


def test_rank3_rhs_grouped_nt_rejects_shared_safe_group_use() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(_rank3_rhs_grouped_nt_shared_safe_group)

    assert "'rhs_rank3_grouped_nt': True" not in code
    assert "tcgen05_rhs_safe_group" not in code
    assert "tcgen05_rhs_group =" not in code
    assert "b_grouped.iterator" in code


def test_rank3_rhs_grouped_nt_rejects_shared_operand_loads() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(_rank3_rhs_grouped_nt_shared_operand_loads)

    assert "'rhs_rank3_grouped_nt': True" not in code
    assert "tcgen05_rhs_safe_group" not in code
    assert "tcgen05_rhs_group =" not in code
    assert "a.iterator" in code
    assert "b_grouped.iterator" in code


def test_rank3_rhs_grouped_nt_rejects_nonzero_accumulator() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(_rank3_rhs_grouped_nt_nonzero_acc)

    assert "'rhs_rank3_grouped_nt': True" not in code
    assert "tcgen05_rhs_safe_group" not in code
    assert "tcgen05_rhs_group =" not in code
    assert "b_grouped.iterator" in code


def test_rank3_rhs_grouped_nt_partial_output_rejects_tma_marker() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(_rank3_rhs_grouped_nt, _make_partial_m_args())

    assert "'rhs_rank3_grouped_nt': True" not in code
    assert "tcgen05_rhs_safe_group" not in code
    assert "tcgen05_rhs_group =" not in code
    assert "b_grouped.iterator" in code


def test_rank3_rhs_grouped_nt_rejects_clustered_two_cta_config() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    try:
        code = _code_for(
            _rank3_rhs_grouped_nt, config=_rank3_rhs_clustered_tma_config()
        )
    except helion.exc.BackendUnsupported as error:
        _assert_backend_unsupported_match(error, "tcgen05")
        return

    assert "'rhs_rank3_grouped_nt': True" not in code
    assert "tcgen05_rhs_safe_group" not in code
    assert "tcgen05_rhs_group =" not in code
    assert "b_grouped.iterator" in code


def test_rank3_rhs_grouped_nt_rejects_non_row_major_a_layout() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    try:
        code = _code_for(_rank3_rhs_grouped_nt, _make_non_row_major_a_args())
    except helion.exc.BackendUnsupported as error:
        _assert_backend_unsupported_match(error, "tcgen05")
        return

    assert "'rhs_rank3_grouped_nt': True" not in code
    assert "tcgen05_rhs_safe_group" not in code
    assert "tcgen05_rhs_group =" not in code
    assert "b_grouped.iterator" in code


def test_rank3_rhs_grouped_nt_rejects_non_k_contiguous_b_layout() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA codegen test needs CUDA fake inputs")

    code = _code_for(_rank3_rhs_grouped_nt, _make_non_k_contiguous_b_args())

    assert "'rhs_rank3_grouped_nt': True" not in code
    assert "tcgen05_rhs_safe_group" not in code


def _require_tcgen05_runtime_test() -> None:
    if DEVICE.type != "cuda":
        pytest.skip("rank3 RHS B TMA runtime test needs CUDA")
    if os.environ.get("HELION_CUTE_MMA_IMPL", "").strip().lower() != "tcgen05":
        pytest.skip("rank3 RHS B TMA runtime test requires tcgen05 MMA")
    from helion._compiler.cute.mma_support import get_cute_mma_support

    with torch.cuda.device(DEVICE):
        major, _minor = torch.cuda.get_device_capability(DEVICE)
    if major < 10:
        pytest.skip("tcgen05 requires SM100+")
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("tcgen05 F16/BF16 MMA is not supported on this machine")


def test_rank3_rhs_grouped_nt_no_row_mask_fingerprint_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_args()
    bound = _rank3_rhs_grouped_nt.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    config = _rank3_rhs_tma_config()
    bound.set_config(config)
    out = bound(*args)
    a, b_grouped, layout = args
    expected = b_grouped[layout, :, torch.arange(a.size(0), device=DEVICE) % a.size(1)]
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_rank3_rhs_grouped_static_persistent_all_full_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_all_full_grouped_args()
    bound = _rank3_rhs_grouped_nt.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    config = _rank3_rhs_grouped_static_config()
    bound.set_config(config)
    out = bound(*args)
    a, b_grouped, layout = args
    expected = b_grouped[layout, :, torch.arange(a.size(0), device=DEVICE) % a.size(1)]
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_rank3_rhs_grouped_static_persistent_fp16_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_dense_all_full_grouped_args(dtype=torch.float16)
    bound = _rank3_rhs_grouped_nt.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    config = _rank3_rhs_grouped_static_config()
    bound.set_config(config)
    out = bound(*args)
    torch.cuda.synchronize()
    _assert_grouped_matmul_close(out, args)


def test_rank3_rhs_grouped_static_persistent_block_n64_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_dense_all_full_grouped_args(n=128, dtype=torch.float16)
    bound = _rank3_rhs_grouped_nt.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    config = _rank3_rhs_grouped_static_config_bn64()
    bound.set_config(config)
    out = bound(*args)
    torch.cuda.synchronize()
    _assert_grouped_matmul_close(out, args)


def test_rank3_rhs_grouped_static_persistent_semantic_n_sizes_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_nvidia_default_like_n_sizes_args()
    bound = _rank3_rhs_grouped_nt_with_n_sizes.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64())
    out = bound(*args)
    torch.cuda.synchronize()
    assert out is args[-1]
    _assert_grouped_n_sizes_matmul_and_sentinel(out, args)


def test_rank3_rhs_grouped_static_persistent_semantic_n_sizes_bn128_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_nvidia_default_like_n_sizes_args()
    bound = _rank3_rhs_grouped_nt_with_n_sizes.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config())
    out = bound(*args)
    torch.cuda.synchronize()
    assert out is args[-1]
    _assert_grouped_n_sizes_matmul_and_sentinel(out, args)


def test_rank3_rhs_grouped_static_persistent_semantic_mn_tails_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_mn_tail_grouped_args()
    bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64())
    out = bound(*args)
    torch.cuda.synchronize()
    assert out is args[-1]
    _assert_grouped_mn_tail_matmul_and_sentinel(out, args)


def test_rank3_rhs_grouped_static_persistent_small_common_k_mn_tails_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_mn_tail_grouped_args(k=32)
    bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64_bk(32))
    out = bound(*args)
    torch.cuda.synchronize()
    assert out is args[-1]
    _assert_grouped_mn_tail_matmul_and_sentinel(out, args)


def test_rank3_rhs_grouped_static_persistent_bk16_mn_tails_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_mn_tail_grouped_args(k=16)
    bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64_bk(16))
    out = bound(*args)
    torch.cuda.synchronize()
    assert out is args[-1]
    _assert_grouped_mn_tail_matmul_and_sentinel(out, args)


def test_rank3_rhs_grouped_static_persistent_bk16_padded_n_tail_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_single_padded_n_tail_args()
    bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64_bk(16))
    out = bound(*args)
    torch.cuda.synchronize()
    assert out is args[-1]
    _assert_grouped_mn_tail_matmul_and_sentinel(out, args)


def test_rank3_rhs_grouped_static_persistent_bk16_multi_group_n_tail_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_multi_group_n_tail_args()
    bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64_bk(16))
    out = bound(*args)
    torch.cuda.synchronize()
    assert out is args[-1]
    _assert_grouped_mn_tail_matmul_and_sentinel(out, args)


def test_rank3_rhs_grouped_static_persistent_bk16_padded_n_tail_graph_capture_runtime() -> (
    None
):
    _require_tcgen05_runtime_test()

    args = _make_single_padded_n_tail_args()
    bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64_bk(16))
    warmup = bound(*args)
    torch.cuda.synchronize()

    args[-1].fill_(-77.0)
    graph = torch.cuda.CUDAGraph()
    captured: list[torch.Tensor] = []
    with torch.cuda.graph(graph):
        captured.append(bound(*args))
    graph.replay()
    torch.cuda.synchronize()

    _assert_grouped_mn_tail_matmul_and_sentinel(warmup, args)
    _assert_grouped_mn_tail_matmul_and_sentinel(captured[0], args)


def test_rank3_rhs_grouped_static_persistent_bk16_padded_no_tail_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_single_padded_n_tail_args(n=128)
    bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64_bk(16))
    out = bound(*args)
    torch.cuda.synchronize()
    assert out is args[-1]
    _assert_grouped_mn_tail_matmul_and_sentinel(out, args)


def test_rank3_rhs_grouped_static_persistent_bk16_padded_no_tail_graph_capture_runtime() -> (
    None
):
    _require_tcgen05_runtime_test()

    args = _make_single_padded_n_tail_args(n=128)
    bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64_bk(16))
    warmup = bound(*args)
    torch.cuda.synchronize()

    args[-1].fill_(-77.0)
    graph = torch.cuda.CUDAGraph()
    captured: list[torch.Tensor] = []
    with torch.cuda.graph(graph):
        captured.append(bound(*args))
    graph.replay()
    torch.cuda.synchronize()

    _assert_grouped_mn_tail_matmul_and_sentinel(warmup, args)
    _assert_grouped_mn_tail_matmul_and_sentinel(captured[0], args)


def test_rank3_rhs_grouped_static_persistent_bk16_padded_n_tail_mutates_n_sizes_runtime() -> (
    None
):
    _require_tcgen05_runtime_test()

    args = list(_make_single_padded_n_tail_args())
    config = _rank3_rhs_grouped_static_config_bn64_bk(16)

    tail_bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(tuple(args))
    tail_bound.env.config_spec.cute_tcgen05_search_enabled = True
    tail_bound.set_config(config)
    out = tail_bound(*tuple(args))
    torch.cuda.synchronize()
    _assert_grouped_mn_tail_matmul_and_sentinel(out, tuple(args))

    args[3].fill_(128)
    args[-1].fill_(-77.0)
    with pytest.raises(helion.exc.BackendUnsupported, match="N-tail specialization"):
        tail_bound(*tuple(args))

    no_tail_bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(tuple(args))
    no_tail_bound.env.config_spec.cute_tcgen05_search_enabled = True
    no_tail_bound.set_config(config)
    out = no_tail_bound(*tuple(args))
    torch.cuda.synchronize()
    _assert_grouped_mn_tail_matmul_and_sentinel(out, tuple(args))

    args[3].fill_(160)
    args[-1].fill_(-77.0)
    with pytest.raises(helion.exc.BackendUnsupported, match="N-tail specialization"):
        no_tail_bound(*tuple(args))

    rebound_tail = _rank3_rhs_grouped_nt_with_mn_tails.bind(tuple(args))
    rebound_tail.env.config_spec.cute_tcgen05_search_enabled = True
    rebound_tail.set_config(config)
    out = rebound_tail(*tuple(args))
    torch.cuda.synchronize()
    _assert_grouped_mn_tail_matmul_and_sentinel(out, tuple(args))


def test_rank3_rhs_grouped_static_persistent_renamed_metadata_mutates_n_sizes_runtime() -> (
    None
):
    _require_tcgen05_runtime_test()

    args = list(_make_single_padded_n_tail_args(n=128))
    config = _rank3_rhs_grouped_static_config_bn64_bk(16)

    no_tail_bound = _rank3_rhs_grouped_nt_with_renamed_mn_tails.bind(tuple(args))
    no_tail_bound.env.config_spec.cute_tcgen05_search_enabled = True
    no_tail_bound.set_config(config)
    out = no_tail_bound(*tuple(args))
    torch.cuda.synchronize()
    _assert_grouped_mn_tail_matmul_and_sentinel(out, tuple(args))

    args[3].fill_(160)
    args[-1].fill_(-77.0)
    with pytest.raises(helion.exc.BackendUnsupported, match="N-tail specialization"):
        no_tail_bound(*tuple(args))

    tail_bound = _rank3_rhs_grouped_nt_with_renamed_mn_tails.bind(tuple(args))
    tail_bound.env.config_spec.cute_tcgen05_search_enabled = True
    tail_bound.set_config(config)
    out = tail_bound(*tuple(args))
    torch.cuda.synchronize()
    _assert_grouped_mn_tail_matmul_and_sentinel(out, tuple(args))

    args[3].fill_(128)
    args[-1].fill_(-77.0)
    with pytest.raises(helion.exc.BackendUnsupported, match="N-tail specialization"):
        tail_bound(*tuple(args))

    rebound_no_tail = _rank3_rhs_grouped_nt_with_renamed_mn_tails.bind(tuple(args))
    rebound_no_tail.env.config_spec.cute_tcgen05_search_enabled = True
    rebound_no_tail.set_config(config)
    out = rebound_no_tail(*tuple(args))
    torch.cuda.synchronize()
    _assert_grouped_mn_tail_matmul_and_sentinel(out, tuple(args))


def test_rank3_rhs_grouped_static_persistent_bk16_padded_no_tail_mutates_layout_runtime() -> (
    None
):
    _require_tcgen05_runtime_test()

    args = list(_make_single_padded_n_tail_args(n=128))
    config = _rank3_rhs_grouped_static_config_bn64_bk(16)

    no_tail_bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(tuple(args))
    no_tail_bound.env.config_spec.cute_tcgen05_search_enabled = True
    no_tail_bound.set_config(config)
    out = no_tail_bound(*tuple(args))
    torch.cuda.synchronize()
    _assert_grouped_mn_tail_matmul_and_sentinel(out, tuple(args))

    _set_single_group_m_tail_layout(args[2])
    args[-1].fill_(-77.0)
    with pytest.raises(helion.exc.BackendUnsupported, match="M-tail specialization"):
        no_tail_bound(*tuple(args))

    m_tail_bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(tuple(args))
    m_tail_bound.env.config_spec.cute_tcgen05_search_enabled = True
    m_tail_bound.set_config(config)
    out = m_tail_bound(*tuple(args))
    torch.cuda.synchronize()
    _assert_grouped_mn_tail_matmul_and_sentinel(out, tuple(args))

    _set_single_group_no_m_tail_layout(args[2])
    args[-1].fill_(-77.0)
    with pytest.raises(helion.exc.BackendUnsupported, match="M-tail specialization"):
        m_tail_bound(*tuple(args))

    rebound_no_tail = _rank3_rhs_grouped_nt_with_mn_tails.bind(tuple(args))
    rebound_no_tail.env.config_spec.cute_tcgen05_search_enabled = True
    rebound_no_tail.set_config(config)
    out = rebound_no_tail(*tuple(args))
    torch.cuda.synchronize()
    _assert_grouped_mn_tail_matmul_and_sentinel(out, tuple(args))


def test_rank3_rhs_grouped_static_persistent_bk16_padded_n_tail_rejects_capture_after_mutation_without_rewarm() -> (
    None
):
    _require_tcgen05_runtime_test()

    args = list(_make_single_padded_n_tail_args())
    bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(tuple(args))
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64_bk(16))
    bound(*tuple(args))
    torch.cuda.synchronize()

    args[3].fill_(128)
    with (
        patch("helion.runtime._cuda_graph_capture_active", return_value=True),
        pytest.raises(helion.exc.BackendUnsupported, match="metadata is not cached"),
    ):
        bound(*tuple(args))


def test_rank3_rhs_grouped_static_persistent_bk16_padded_no_tail_rejects_capture_after_layout_mutation_without_rewarm() -> (
    None
):
    _require_tcgen05_runtime_test()

    args = list(_make_single_padded_n_tail_args(n=128))
    bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(tuple(args))
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64_bk(16))
    bound(*tuple(args))
    torch.cuda.synchronize()

    _set_single_group_m_tail_layout(args[2])
    with (
        patch("helion.runtime._cuda_graph_capture_active", return_value=True),
        pytest.raises(helion.exc.BackendUnsupported, match="metadata is not cached"),
    ):
        bound(*tuple(args))


def test_rank3_rhs_grouped_static_persistent_bk16_n1280_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_dense_all_full_grouped_args(
        groups=5,
        m_per_group=128,
        n=1280,
        k=16,
        dtype=torch.float16,
    )
    bound = _rank3_rhs_grouped_nt.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    config = _rank3_rhs_grouped_static_config()
    config.config["block_sizes"] = [128, 128, 16]
    bound.set_config(config)
    out = bound(*args)
    torch.cuda.synchronize()
    _assert_grouped_matmul_close(out, args)


@pytest.mark.parametrize(
    ("k", "block_k"),
    [
        (64, 64),
        (96, 32),
        (160, 32),
        (192, 64),
    ],
)
def test_rank3_rhs_grouped_static_persistent_common_k_mn_tails_runtime(
    k: int,
    block_k: int,
) -> None:
    _require_tcgen05_runtime_test()

    args = _make_mn_tail_grouped_args(k=k)
    bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64_bk(block_k))
    out = bound(*args)
    torch.cuda.synchronize()
    assert out is args[-1]
    _assert_grouped_mn_tail_matmul_and_sentinel(out, args)


def test_rank3_rhs_grouped_static_persistent_common_k96_graph_capture_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_mn_tail_grouped_args(k=96)
    bound = _rank3_rhs_grouped_nt_with_mn_tails.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64_bk(32))
    warmup = bound(*args)
    torch.cuda.synchronize()

    args[-1].fill_(-77.0)
    graph = torch.cuda.CUDAGraph()
    captured: list[torch.Tensor] = []
    with torch.cuda.graph(graph):
        captured.append(bound(*args))
    graph.replay()
    torch.cuda.synchronize()

    _assert_grouped_mn_tail_matmul_and_sentinel(warmup, args)
    _assert_grouped_mn_tail_matmul_and_sentinel(captured[0], args)


def test_rank3_rhs_grouped_static_persistent_k1536_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_all_full_grouped_args(groups=2, n=128, k=1536)
    bound = _rank3_rhs_grouped_nt.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_default_ab_config())
    out = bound(*args)
    a, b_grouped, layout = args
    expected = b_grouped[layout, :, torch.arange(a.size(0), device=DEVICE) % a.size(1)]
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_rank3_rhs_grouped_static_persistent_documented_mixed_k_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_documented_mixed_k_args()
    bound = _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64_bk(16))
    out = bound(*args)
    torch.cuda.synchronize()
    assert out is args[-1]
    _assert_grouped_mn_tail_k_sizes_matmul_and_sentinel(out, args)


def test_rank3_rhs_grouped_static_persistent_dynamic_bk64_documented_mixed_k_runtime() -> (
    None
):
    _require_tcgen05_runtime_test()

    args = _make_documented_mixed_k_args()
    bound = _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_dynamic_bk64_config())
    out = bound(*args)
    torch.cuda.synchronize()
    assert out is args[-1]
    _assert_grouped_mn_tail_k_sizes_matmul_and_sentinel(
        out,
        args,
        rtol=4e-2,
        atol=4e-2,
    )


def test_rank3_rhs_grouped_static_persistent_direct_metadata_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_documented_mixed_k_args()
    bound = _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_direct_bk64_config())
    out = bound(*args)
    torch.cuda.synchronize()
    assert out is args[-1]
    _assert_grouped_mn_tail_k_sizes_matmul_and_sentinel(
        out,
        args,
        rtol=4e-2,
        atol=4e-2,
    )


def test_rank3_rhs_grouped_static_persistent_dynamic_bk64_no_mn_tail_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_no_mn_tail_mixed_k_args()
    args[0].normal_(mean=0.0, std=0.05)
    args[1].normal_(mean=0.0, std=0.05)
    bound = _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_direct_bk64_config())
    out = bound(*args)
    torch.cuda.synchronize()
    assert out is args[-1]
    _assert_grouped_mn_tail_k_sizes_matmul_and_sentinel(
        out,
        args,
        rtol=4e-2,
        atol=4e-2,
    )


def test_rank3_rhs_grouped_static_persistent_dynamic_bk64_ab4_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_documented_mixed_k_args()
    bound = _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    config = _rank3_rhs_grouped_static_dynamic_bk64_config()
    config.config["tcgen05_ab_stages"] = 4
    bound.set_config(config)
    out = bound(*args)
    torch.cuda.synchronize()
    assert out is args[-1]
    _assert_grouped_mn_tail_k_sizes_matmul_and_sentinel(
        out,
        args,
        rtol=4e-2,
        atol=4e-2,
    )


def test_rank3_rhs_grouped_static_persistent_dynamic_bk64_graph_capture_runtime() -> (
    None
):
    _require_tcgen05_runtime_test()

    args = _make_documented_mixed_k_args()
    bound = _rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_dynamic_bk64_config())
    warmup = bound(*args)
    torch.cuda.synchronize()

    args[-1].fill_(-77.0)
    graph = torch.cuda.CUDAGraph()
    captured: list[torch.Tensor] = []
    with torch.cuda.graph(graph):
        captured.append(bound(*args))
    graph.replay()
    torch.cuda.synchronize()

    _assert_grouped_mn_tail_k_sizes_matmul_and_sentinel(
        warmup,
        args,
        rtol=4e-2,
        atol=4e-2,
    )
    _assert_grouped_mn_tail_k_sizes_matmul_and_sentinel(
        captured[0],
        args,
        rtol=4e-2,
        atol=4e-2,
    )


def test_rank3_rhs_grouped_static_persistent_semantic_m_tail_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_m_tail_grouped_args()
    bound = _rank3_rhs_grouped_nt_with_m_tail.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64())
    out = bound(*args)
    torch.cuda.synchronize()
    assert out is args[-1]
    _assert_grouped_m_tail_matmul_and_sentinel(out, args)


def test_rank3_rhs_grouped_static_persistent_k1536_m_tail_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_m_tail_grouped_args(k=1536)
    bound = _rank3_rhs_grouped_nt_with_m_tail.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64())
    out = bound(*args)
    torch.cuda.synchronize()
    assert out is args[-1]
    _assert_grouped_m_tail_matmul_and_sentinel(out, args)


def test_rank3_rhs_grouped_static_persistent_interior_m_tail_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_m_tail_grouped_args(m_sizes=(16, 128))
    bound = _rank3_rhs_grouped_nt_with_m_tail.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64())
    out = bound(*args)
    torch.cuda.synchronize()
    assert out is args[-1]
    _assert_grouped_m_tail_matmul_and_sentinel(out, args)


def test_rank3_rhs_grouped_static_persistent_semantic_n_sizes_rejects_bad_n() -> None:
    _require_tcgen05_runtime_test()

    args_list = list(_make_nvidia_default_like_n_sizes_args())
    bad_n_sizes = args_list[3].clone()
    bad_n_sizes[1] = args_list[1].size(1) + 1
    args_list[3] = bad_n_sizes
    args = tuple(args_list)
    bound = _rank3_rhs_grouped_nt_with_n_sizes.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64())
    with pytest.raises(helion.exc.BackendUnsupported, match="per-group N size"):
        bound(*args)


def test_rank3_rhs_grouped_static_persistent_semantic_n_sizes_cache_invalidation() -> (
    None
):
    _require_tcgen05_runtime_test()

    args = _make_nvidia_default_like_n_sizes_args()
    bound = _rank3_rhs_grouped_nt_with_n_sizes.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64())
    out = bound(*args)
    torch.cuda.synchronize()
    _assert_grouped_n_sizes_matmul_and_sentinel(out, args)

    args[3][1] = 256
    args[4].fill_(-77.0)
    out = bound(*args)
    torch.cuda.synchronize()
    _assert_grouped_n_sizes_matmul_and_sentinel(out, args)


def test_rank3_rhs_grouped_static_persistent_semantic_n_sizes_graph_capture() -> None:
    _require_tcgen05_runtime_test()

    args = _make_nvidia_default_like_n_sizes_args()
    bound = _rank3_rhs_grouped_nt_with_n_sizes.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64())
    warmup = bound(*args)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = bound(*args)
    graph.replay()
    torch.cuda.synchronize()

    _assert_grouped_n_sizes_matmul_and_sentinel(warmup, args)
    _assert_grouped_n_sizes_matmul_and_sentinel(captured, args)


def test_rank3_rhs_grouped_static_persistent_semantic_n_sizes_requires_prewarm() -> (
    None
):
    _require_tcgen05_runtime_test()

    args = _make_nvidia_default_like_n_sizes_args()
    bound = _rank3_rhs_grouped_nt_with_n_sizes.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config_bn64())
    with (
        patch("helion.runtime._cuda_graph_capture_active", return_value=True),
        pytest.raises(helion.exc.BackendUnsupported, match="metadata is not cached"),
    ):
        bound(*args)


def test_rank3_rhs_grouped_static_persistent_graph_capture_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_all_full_grouped_args()
    bound = _rank3_rhs_grouped_nt.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config())
    warmup = bound(*args)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    captured: list[torch.Tensor] = []
    with torch.cuda.graph(graph):
        captured.append(bound(*args))
    graph.replay()
    torch.cuda.synchronize()

    a, b_grouped, layout = args
    expected = b_grouped[layout, :, torch.arange(a.size(0), device=DEVICE) % a.size(1)]
    torch.testing.assert_close(warmup, expected, rtol=0, atol=0)
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)


def test_rank3_rhs_grouped_static_persistent_rejects_mixed_boundary_runtime() -> None:
    _require_tcgen05_runtime_test()

    args = _make_mixed_boundary_grouped_args()
    bound = _rank3_rhs_grouped_nt.bind(args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(_rank3_rhs_grouped_static_config())
    with pytest.raises(helion.exc.BackendUnsupported, match="CTA M tile"):
        bound(*args)
