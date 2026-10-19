"""CPU proof, allocation, normalization and codegen for deep one-CTA FP8."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from test._cute_binding import _forbid_cuda_work
from test._cute_dense_tcgen05 import atomic_output_mm
from test._cute_dense_tcgen05 import blackwell_capability
from test._cute_dense_tcgen05 import cpu_codegen_target
from test._cute_dense_tcgen05 import extra_store_mm
from test._cute_dense_tcgen05 import scaled_args
from test._cute_dense_tcgen05 import scaled_mm
from test._cute_dense_tcgen05 import scaled_mm_fp32

import helion
from helion._compiler.autotuner_heuristics.cute_tcgen05_schedule import (
    CuteTcgen05ScheduleHeuristic,
)
from helion._compiler.cute.one_cta_pipeline import max_one_cta_pipeline_ab_stages
from helion._compiler.cute.one_cta_pipeline import one_cta_pipeline_smem_bytes
from helion._compiler.cute.pipeline_smem import Tcgen05PipelineSmemFacts
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = skipUnlessBackends(["cute"])


@pytest.fixture(autouse=True)
def _cpu_codegen() -> Iterator[None]:
    with cpu_codegen_target(), _forbid_cuda_work():
        yield


def _config(**overrides: object) -> helion.Config:
    return helion.Config.from_dict(
        {
            "block_sizes": [64, 32, 128],
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 1,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": 16,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": 2,
            "tcgen05_aux_load_placement": "post_acc_wait",
        }
        | overrides
    )


@pytest.mark.parametrize("output_bytes", (2, 4))
@pytest.mark.parametrize("c_stages", (2, 4))
@pytest.mark.parametrize("bk", (32, 64, 128, 256))
def test_complete_budget_boundary(output_bytes: int, c_stages: int, bk: int) -> None:
    facts = Tcgen05PipelineSmemFacts(1, output_bytes, 232448)
    params = {"bm": 64, "bn": 32, "bk": bk, "c_stages": c_stages, "acc_stages": 2}
    limit = max_one_cta_pipeline_ab_stages(facts, **params)
    assert 0 < limit <= 16
    used = one_cta_pipeline_smem_bytes(facts, ab_stages=limit, **params)
    assert used <= facts.capacity_bytes
    assert (
        max_one_cta_pipeline_ab_stages(
            facts._replace(capacity_bytes=used - 1), **params
        )
        == limit - 1
    )
    if limit < 16:
        assert (
            one_cta_pipeline_smem_bytes(facts, ab_stages=limit + 1, **params) > 232448
        )
    if output_bytes == 2 and c_stages == 2 and bk == 128:
        assert used == 207136 and limit == 16
    if output_bytes == 2 and c_stages == 2 and bk == 256:
        assert used == 231600 and limit == 9


@pytest.mark.parametrize("mode", range(4))
@pytest.mark.parametrize("kernel", (scaled_mm, scaled_mm_fp32))
def test_deep_pipeline_all_scales_and_widths(kernel: helion.Kernel, mode: int) -> None:
    bound = kernel._bind_isolated(scaled_args(mode=mode))
    state = bound.config_spec._cute_tcgen05_config
    assert state.one_cta_pipeline_facts is not None
    assert state._get_dtype_ab_stages_hard_cap(1) == 12
    with bound.env:
        gen = ConfigGeneration(bound.config_spec)
        flat, config = gen.canonicalize_flat(gen.flatten(_config()))
        flat2, config2 = gen.canonicalize_flat(flat)
        assert flat2 == flat and config2 == config
        assert gen.unflatten(flat) == config
        assert config.block_sizes == [64, 32, 128]
        assert config.config["tcgen05_ab_stages"] == 16
        assert config.config["tcgen05_c_stages"] == 2
        seeds = state.one_cta_pipeline_seed_configs()
        assert any(
            s.block_sizes == [64, 32, 128] and s.config["tcgen05_ab_stages"] == 16
            for s in seeds
        )
        for seed in seeds:
            _flat, normalized = gen.canonicalize_flat(gen.flatten(seed))
            assert normalized.block_sizes == seed.block_sizes
            assert (
                normalized.config["tcgen05_ab_stages"]
                == seed.config["tcgen05_ab_stages"]
            )
    source = bound.to_code(config)
    assert "CtaGroup.ONE" in source
    assert "num_stages=16" in source
    assert "PipelineTmaUmma.create" in source
    assert "PipelineTmaStore.create" in source
    assert "aux_smem" not in source


@pytest.mark.parametrize("store_mode", ("normal", "direct"))
def test_explicit_bk256_depth_and_store_ring_preserved(store_mode: str) -> None:
    bound = scaled_mm._bind_isolated(scaled_args())
    state = bound.config_spec._cute_tcgen05_config
    config = _config(
        block_sizes=[64, 32, 256], tcgen05_ab_stages=9, tcgen05_c_store_mode=store_mode
    )
    assert state._one_cta_pipeline_stage_limit(config.config) == 9
    source = bound.to_code(config)
    assert "num_stages=9" in source and "CtaGroup.ONE" in source
    assert ("PipelineTmaStore.create" in source) == (store_mode == "normal")
    with pytest.raises(helion.exc.InvalidConfig):
        bound.to_code(_config(block_sizes=[64, 32, 256], tcgen05_ab_stages=10))
    with bound.env:
        repaired = _config(block_sizes=[64, 32, 256], tcgen05_ab_stages=16)
        bound.config_spec.normalize(repaired, _fix_invalid=True)
        assert repaired.config["tcgen05_ab_stages"] == 9


@pytest.mark.parametrize(
    "overrides",
    (
        {"tcgen05_aux_load_placement": "pre_acc_wait"},
        {"tcgen05_cluster_n": 2},
        {"tcgen05_cluster_m": 2},
        {"tcgen05_persistence_model": "clc_persistent"},
        {"tcgen05_warp_spec_scheduler_warps": 1},
        {"tcgen05_warp_spec_epi_load_warps": 1},
        {"tcgen05_warp_spec_store_warps": 1},
        {"tcgen05_warp_spec_c_input_warps": 1},
        {"tcgen05_layout_overrides_epi_tile_m": 64},
        {"tcgen05_epilogue_layout": "split_first_t2r"},
        {"cute_matmul_family": "warp_mma"},
        {"pid_type": "flat"},
    ),
)
def test_extra_protocols_cannot_borrow_complete_budget(
    overrides: dict[str, object],
) -> None:
    bound = scaled_mm._bind_isolated(scaled_args())
    config = _config(**overrides)
    assert (
        bound.config_spec._cute_tcgen05_config._one_cta_pipeline_parameters(
            config.config
        )
        is None
    )
    with pytest.raises(helion.exc.InvalidConfig):
        bound.to_code(config)


@pytest.mark.parametrize(
    "kind", ("scale_dtype", "scale_stride", "unaligned", "extra_store", "half_input")
)
def test_unproved_graph_cannot_gain_deep_stages(kind: str) -> None:
    args = list(scaled_args())
    kernel = scaled_mm
    if kind == "scale_dtype":
        args[3] = args[3].to(torch.bfloat16)
    elif kind == "scale_stride":
        args[3] = torch.empty(8192, dtype=torch.float32)[::2]
    elif kind == "unaligned":
        args[0] = torch.empty(64 * 4096 + 1, dtype=torch.float8_e4m3fn)[1:].view(
            64, 4096
        )
    elif kind == "half_input":
        args[:2] = [x.to(torch.bfloat16) for x in args[:2]]
    else:
        kernel, args = extra_store_mm, args[:2]
    bound = kernel._bind_isolated(tuple(args))
    assert bound.config_spec._cute_tcgen05_config.one_cta_pipeline_facts is None
    with pytest.raises(helion.exc.InvalidConfig):
        bound.to_code(_config())


def test_atomic_output_cannot_borrow_single_output_budget() -> None:
    bound = atomic_output_mm._bind_isolated(
        (*scaled_args()[:2], torch.zeros(1, dtype=torch.int32))
    )
    assert bound.config_spec._cute_tcgen05_config.one_cta_pipeline_facts is None
    with pytest.raises(helion.exc.InvalidConfig):
        bound.to_code(_config())


@pytest.mark.parametrize("kernel", (scaled_mm, scaled_mm_fp32))
@pytest.mark.parametrize("store", ("normal", "direct"))
def test_m128_pipeline_uses_one_cta_output_layout(
    kernel: helion.Kernel, store: str
) -> None:
    bound = kernel._bind_isolated(scaled_args(m=128))
    config = _config(block_sizes=[128, 32, 64], tcgen05_c_store_mode=store)
    state = bound.config_spec._cute_tcgen05_config
    assert state._one_cta_pipeline_stage_limit(config.config) == 16
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        flat, normalized = generation.canonicalize_flat(generation.flatten(config))
        assert normalized.block_sizes == [128, 32, 64]
        assert normalized.config["tcgen05_ab_stages"] == 16
        assert generation.unflatten(flat) == normalized
    source = bound.to_code(normalized)
    assert "CtaGroup.ONE" in source and "num_stages=16" in source
    assert ("PipelineTmaStore.create" in source) == (store == "normal")
    assert "aux_rowvec_smem_ptr" not in source


def test_wide_bn_fp32_c4_boundary_source() -> None:
    bound = scaled_mm_fp32._bind_isolated(scaled_args(m=128))
    state = bound.config_spec._cute_tcgen05_config
    config = _config(block_sizes=[128, 256, 32], tcgen05_c_stages=4)
    limit = state._one_cta_pipeline_stage_limit(config.config)
    assert limit is not None and 0 < limit < 16
    config.config["tcgen05_ab_stages"] = limit
    source = bound.to_code(config)
    assert "CtaGroup.ONE" in source and f"num_stages={limit}" in source
    assert "PipelineTmaStore.create(num_stages=4" in source
    assert "cute.arch.alloc_smem(cutlass.Float32" in source
    assert "aux_rowvec_smem_ptr" not in source
    config.config["tcgen05_ab_stages"] = limit + 1
    with pytest.raises(helion.exc.InvalidConfig):
        bound.to_code(config)


@pytest.mark.parametrize(
    "shape", ((65, 4096, 4096), (64, 4100, 4096), (64, 4096, 4100))
)
def test_partial_tiles_keep_old_budget(shape: tuple[int, int, int]) -> None:
    bound = scaled_mm._bind_isolated(scaled_args(*shape))
    assert (
        bound.config_spec._cute_tcgen05_config._one_cta_pipeline_parameters(
            _config().config
        )
        is None
    )
    with pytest.raises(helion.exc.InvalidConfig):
        bound.to_code(_config())


def test_pid_vetoes_and_registration_idempotence() -> None:
    bound = scaled_mm._bind_isolated(scaled_args())
    state = bound.config_spec._cute_tcgen05_config
    facts = state.one_cta_pipeline_facts
    assert facts is not None
    for pid in ("persistent_blocked", "persistent_interleaved"):
        bound.config_spec.disallow_pid_type(pid, reason="foreign dependency")
    previous = dict(bound.config_spec.disallowed_pid_type_causes)
    CuteTcgen05ScheduleHeuristic.register_facts(
        bound.env, bound.host_function.device_ir
    )
    assert state.one_cta_pipeline_facts == facts
    assert not state.one_cta_pipeline_seed_configs()
    assert state._one_cta_pipeline_parameters(_config().config) is None
    assert bound.config_spec.disallowed_pid_type_causes == previous


@pytest.mark.parametrize("capability", ((10, 0), (10, 3)))
def test_supported_architecture(capability: tuple[int, int]) -> None:
    with blackwell_capability(capability):
        bound = scaled_mm._bind_isolated(scaled_args())
        assert bound.config_spec._cute_tcgen05_config.one_cta_pipeline_facts is not None
        assert "num_stages=16" in bound.to_code(_config())
