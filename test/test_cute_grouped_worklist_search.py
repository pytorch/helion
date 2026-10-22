"""CPU regression coverage for ordinary grouped-worklist proposals."""

from __future__ import annotations

import ast
import random
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _forbid_cuda_work
from test._cute_dense_tcgen05 import gb300_target

import helion
from helion._compiler.autotuner_heuristics.cute import (
    CuteTcgen05GroupedWorklistHeuristic,
)
from helion._compiler.cute.tcgen05_config import Tcgen05AbStagesThreeSearchConstraints
from helion._compiler.cute.tcgen05_constants import (
    resolve_tcgen05_grouped_worklist_mma_profile,
)
from helion._compiler.cute.tcgen05_constants import tcgen05_grouped_worklist_smem_bytes
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _grouped_gemm(
    a: torch.Tensor, weights: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    groups, n, _ = weights.shape
    block_m = hl.register_block_size(256)
    block_n = hl.register_block_size(128)
    block_k = hl.register_block_size(64, 128)
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for group_tile, rows, columns in hl.tile(
        [groups, m, n], block_size=[1, block_m, block_n]
    ):
        group = group_tile.index.sum()
        start = offsets[group]
        length = offsets[group + 1] - offsets[group]
        row_index = start + rows.index
        valid = rows.index < length
        acc = hl.zeros([rows, columns], dtype=torch.float32)
        for inner in hl.tile(k, block_size=block_k):
            left = hl.load(a, [row_index, inner], extra_mask=valid[:, None])
            acc = torch.addmm(acc, left, weights[group, columns, inner].T)
        hl.store(
            out,
            [row_index, columns],
            acc.to(out.dtype),
            extra_mask=valid[:, None],
        )
    return out


@pytest.fixture
def bound(monkeypatch: pytest.MonkeyPatch) -> Iterator[BoundKernel[object]]:
    for name in (
        "HELION_AUTOTUNE_EFFORT",
        "HELION_AUTOTUNER",
        "HELION_AUTOTUNE_CONFIG_OVERRIDES",
        "HELION_DISABLE_AUTOTUNER_HEURISTICS",
    ):
        monkeypatch.delenv(name, raising=False)
    with (
        _forbid_cuda_work(),
        gb300_target(),
        patch(
            "helion._compiler.autotuner_heuristics.cute."
            "tcgen05_runtime_n_ptx_compatible",
            return_value=False,
        ),
        patch(
            "helion._compiler.autotuner_heuristics.cute."
            "warn_tcgen05_runtime_n_ptx_fallback"
        ),
    ):
        args = (
            torch.empty((2048, 2048), dtype=torch.bfloat16),
            torch.empty((16, 1536, 2048), dtype=torch.bfloat16),
            torch.arange(17, dtype=torch.int32) * 128,
        )
        yield _grouped_gemm._bind_isolated(args)


def test_runtime_n_seed_guard_keeps_worklist_search_and_pure_default(
    bound: BoundKernel[object],
) -> None:
    spec = bound.config_spec
    tcgen = spec._cute_tcgen05_config
    assert tcgen.grouped_worklist_search_source_m_tiles == (224, 256, 32)
    assert tcgen.grouped_worklist_search_device_split_sizes is True
    assert spec.compiler_seed_configs == []
    assert spec.compiler_default_config is None
    fields = spec._flat_fields()
    assert fields["tcgen05_grouped_mode"].search_choices == (None, "worklist_nm")
    assert fields["tcgen05_grouped_worklist_source_m_tile"].search_choices == (
        None,
        224,
        256,
        32,
    )
    assert fields["tcgen05_ab_stages"].default() == 2
    assert fields["tcgen05_consumer_regs"].default() == 256
    for key in (
        "tcgen05_grouped_runtime_direct",
        "tcgen05_grouped_full_allocation_a",
        "tcgen05_warp_register_reallocation",
    ):
        assert key not in fields
    generation = ConfigGeneration(spec)
    default = generation.unflatten(generation.default_flat())
    assert "tcgen05_grouped_mode" not in default.config
    assert "tcgen05_grouped_worklist_source_m_tile" not in default.config

    assert bound.host_function is not None
    device_ir = bound.host_function.device_ir
    assert (
        CuteTcgen05GroupedWorklistHeuristic._eligible_inputs(bound.env, device_ir)
        is None
    )
    assert CuteTcgen05GroupedWorklistHeuristic._seed_configs(bound.env, device_ir) == []
    with patch(
        "helion._compiler.autotuner_heuristics.cute.tcgen05_runtime_n_ptx_compatible",
        side_effect=AssertionError("ordinary search must not require runtime-N PTX"),
    ):
        eligible = CuteTcgen05GroupedWorklistHeuristic._eligible_inputs(
            bound.env, device_ir, require_runtime_n_ptx=False
        )
    assert eligible is not None


def test_ordinary_random_worklists_reach_tiles_and_roundtrip(
    bound: BoundKernel[object],
) -> None:
    generation = ConfigGeneration(bound.config_spec)
    previous_state = random.getstate()
    try:
        random.seed(0)
        configs = [generation.random_config() for _ in range(128)]
    finally:
        random.setstate(previous_state)
    worklists = [
        config
        for config in configs
        if config.get("tcgen05_grouped_mode") == "worklist_nm"
    ]
    assert worklists
    assert {
        config.get("tcgen05_grouped_worklist_source_m_tile") for config in worklists
    } == {
        224,
        256,
        32,
    }
    assert {config.block_sizes[2] for config in worklists} == {64, 128}
    assert any(config.get("tcgen05_ab_stages") == 7 for config in worklists)
    assert any(config.get("tcgen05_consumer_regs") == 240 for config in worklists)
    for config in worklists:
        flat = generation.flatten(config)
        assert generation.unflatten(flat) == config
        assert generation.flatten(generation.unflatten(flat)) == flat
        assert config.block_sizes[:2] == [256, 128]
        assert config.num_warps == 8
        assert config.get("tcgen05_acc_stages") == 2
        assert config.get("tcgen05_c_stages") == 2
        assert config.get("tcgen05_cluster_n") == 1
        assert config.get("tcgen05_grouped_runtime_direct", False) is False
    assert bound.config_spec.compiler_seed_configs == []


def test_deep_worklist_roundtrip_and_smem_demotion(
    bound: BoundKernel[object],
) -> None:
    spec = bound.config_spec
    generation = ConfigGeneration(spec)
    requested = generation.unflatten(generation.default_flat())
    requested.config.update(
        block_sizes=[256, 128, 64],
        tcgen05_grouped_mode="worklist_nm",
        tcgen05_grouped_worklist_source_m_tile=224,
        tcgen05_ab_stages=7,
        tcgen05_consumer_regs=240,
    )
    spec.normalize(requested, _fix_invalid=True)
    assert requested.get("tcgen05_grouped_worklist_source_m_tile") == 224
    assert requested.get("tcgen05_ab_stages") == 7
    assert requested.get("tcgen05_consumer_regs") == 240
    assert generation.unflatten(generation.flatten(requested)) == requested
    source = bound.to_code(requested)
    ast.parse(source)
    assert "tcgen05_grouped_static_persistent" in source

    oversized = helion.Config.from_dict(
        requested.config | {"tcgen05_grouped_worklist_source_m_tile": 256}
    )
    tcgen = spec._cute_tcgen05_config
    capacity = 180 * 1024
    with patch.object(
        tcgen,
        "ab_stages_three_search_constraints",
        Tcgen05AbStagesThreeSearchConstraints(
            dtype_bytes=2, per_cta_smem_budget_bytes=capacity - 28 * 1024
        ),
    ):
        assert tcgen._grouped_worklist_nm_ab_config_mismatch(oversized.config, 7)
        spec.normalize(oversized, _fix_invalid=True)
        assert oversized.get("tcgen05_ab_stages") < 7
    profile = resolve_tcgen05_grouped_worklist_mma_profile(
        oversized.config, block_k=oversized.block_sizes[2]
    )
    facts = tcgen.grouped_worklist_smem_facts
    assert profile is not None and facts is not None
    required = tcgen05_grouped_worklist_smem_bytes(
        group_count=facts.group_count,
        device_split_sizes=facts.device_split_sizes,
        sched_stage_count=oversized.get("tcgen05_sched_stage_count", 1),
        bm=profile.mma_m,
        bn=profile.mma_n,
        bk=oversized.block_sizes[2],
        dtype_bytes=2,
        output_dtype_bytes=facts.output_dtype_bytes,
        ab_stages=oversized.get("tcgen05_ab_stages"),
        acc_stages=2,
        c_stages=2,
        cluster_m=profile.cluster_m,
    )
    assert required <= capacity
    assert spec.compiler_seed_configs == []
