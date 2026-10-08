from __future__ import annotations

import ast
import dataclasses
import os
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from ._cute_binding import _mock_cuda_unavailable
import helion
from helion._compiler.backend import CuteBackend
from helion._compiler.cute import cute_flash
from helion._compiler.cute.attention_plan import causal_score_plan
from helion._compiler.cute.attention_plan import dense_score_plan
from helion._compiler.cute.flash_schedule import FlashScheduleError
from helion._compiler.cute.flash_schedule import FlashScheduleSpec
from helion._compiler.cute.flash_schedule import build_fa4_schedule
from helion._compiler.cute.flash_schedule import max_fa4_kv_depth
from helion._compiler.cute.flash_schedule import verify_flash_schedule
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.device_function import DeviceFunction
    from helion.autotuner.config_fragment import EnumFragment

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}, clear=True),
        _mock_cuda_unavailable(),
        patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CPU-only test")
        ),
    ):
        yield


def _values(**overrides: object) -> dict[str, object]:
    return {
        "cute_flash_pipeline_family": "fa4",
        "cute_flash_stat_transport": "single",
        "cute_flash_e2e_schedule": "xu",
        "cute_flash_exp2_packet": "1x1",
        "cute_flash_softmax_disc": False,
        "cute_flash_rescale_threshold": 8.0,
        "cute_flash_wait_hint": 10_000_000,
        "cute_flash_persistent": True,
        "cute_flash_persistent_loop": "counted",
        "cute_flash_epi_tma": True,
        "cute_flash_p_store_rep": 16,
        "cute_flash_s_load_rep": 32,
        "cute_flash_rowmax": "software",
        "cute_flash_role_map": "helion",
        **overrides,
    }


def _resolve(
    head_dim: int,
    dtype: torch.dtype,
    values: dict[str, object],
    *,
    num_kv: int = 8,
    causal: bool = False,
) -> cute_flash.FlashAttentionConfig:
    return cute_flash.resolve_flash_config(
        head_dim,
        num_kv,
        values,
        dtype=dtype,
        num_bh=64,
        is_causal=causal,
        standard_dense_output=not causal,
        standard_causal_output=causal,
    )


def _source(
    head_dim: int,
    dtype: torch.dtype,
    values: dict[str, object],
    *,
    causal: bool = False,
) -> str:
    config = _resolve(head_dim, dtype, values, causal=causal)
    body = cute_flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=head_dim,
        num_kv=8,
        sequence_extent=1024,
        num_bh=64,
        total_tiles=128 if config.use_2cta_instrs else 256,
        cfg=config,
        has_lse=False,
        io_dtype="cutlass.Float16" if dtype is torch.float16 else "cutlass.BFloat16",
        score_plan=causal_score_plan(head_dim)
        if causal
        else dense_score_plan(head_dim),
    )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def _config_spec(
    head_dim: int,
    dtype: torch.dtype,
    *,
    num_kv: int = 48,
    causal: bool = False,
    requires_ws_overlap: bool = False,
) -> ConfigSpec:
    spec = ConfigSpec(
        backend=CuteBackend(),
        target_device_capability=(10, 3),
        device=torch.device("cpu"),
        num_sm=152,
    )
    for block_id, target in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
    spec.enable_cute_flash_search(
        head_dim=head_dim,
        num_kv=num_kv,
        num_bh=64,
        dtype=dtype,
        block_size_targets={0: 1, 1: 128, 2: 128},
        is_causal=causal,
        standard_dense_output=not causal,
        standard_causal_output=causal,
        requires_ws_overlap=requires_ws_overlap,
    )
    return spec


def _normalized_compiler_seeds(spec: ConfigSpec) -> list[helion.Config]:
    seeds = spec.autotune_seed_configs()
    spec.compiler_seed_configs = seeds
    generation = spec.create_config_generation()
    normalized = []
    for seed in seeds:
        # Compiler seeds are hints transferred through the flat search surface.
        # A WS seed can retain inactive FA4 values before that transfer.
        flat, config = generation.canonicalize_flat(generation.flatten(seed))
        assert spec.normalized_config(config) == config
        assert generation.unflatten(flat) == config
        assert spec._resolve_cute_flash_config(config.config) == (
            spec._resolve_cute_flash_config(seed.config)
        )
        normalized.append(config)
    messages: list[str] = []
    transferred = [
        config for flat, config in generation.seed_flat_config_pairs(messages.append)
    ]
    assert not messages
    assert transferred == list(dict.fromkeys(normalized))
    return normalized


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("schedule", ("xu", "16/4"))
@pytest.mark.parametrize("family", ("fa4", "fa4_2cta"))
@pytest.mark.parametrize("mma_interleave", (False, True))
@pytest.mark.parametrize("split_p", (False, True))
@pytest.mark.parametrize("persistent", (False, True))
def test_acknowledged_dense_probability_publication_precedes_empty_acquire(
    head_dim: int,
    dtype: torch.dtype,
    schedule: str,
    family: str,
    mma_interleave: bool,
    split_p: bool,
    persistent: bool,
) -> None:
    values = _values(
        cute_flash_pipeline_family=family,
        cute_flash_e2e_schedule=schedule,
        cute_flash_mma_interleave=mma_interleave,
        cute_flash_split_p_arrive=split_p,
        cute_flash_persistent=persistent,
        cute_flash_kv_stage=3,
    )
    config = _resolve(head_dim, dtype, values)
    assert (config.stat_transport, config.softmax_disc) == ("single", False)
    assert config.mma_interleave is mma_interleave
    assert config.split_p_arrive is split_p
    assert config.persistent is persistent
    source = _source(head_dim, dtype, values)
    initial_phase = "flash_s_corr_prod_phase = cutlass.Int32(1)"
    assert source.count(initial_phase) == 2
    first = source.index(initial_phase)
    second = source.index(initial_phase, first + 1)
    correction_start = source.index("(warp_idx >= 8) & (warp_idx < 12):")
    sections = (source[first:second], source[second:correction_start])
    for stage, section in enumerate(sections):
        empty = f"_helion_flash_rt.mbar_spin_wait(flash_s{stage}_corr_empty_ptr + 0, flash_s_corr_prod_phase, 10000000)"
        assert section.count(empty) == 3  # Entry, post-P, terminal drain.
        entry = section.index(empty)
        alpha = section.index(
            f"flash_scale_t[{stage} * 128 + flash_local_tidx] = flash_alpha"
        )
        acquire = section.index(empty, entry + len(empty))
        recurrence = section.index(
            "flash_row_sum = flash_row_sum * flash_alpha + flash_p_sum"
        )
        rowsum = section.index(
            f"flash_scale_t[{stage} * 128 + flash_local_tidx] = flash_row_sum"
        )
        tail = section.index(empty, acquire + len(empty))
        assert entry < alpha < acquire < recurrence < rowsum < tail
        assert section.count("flash_s_corr_prod_phase ^= 1") == 2
        if persistent:
            assert (
                section.index(initial_phase)
                < section.index("for flash_tile_iter")
                < entry
            )
        publication = section[alpha:acquire]
        if schedule == "xu":
            # XU uses inline stores. Every chunk, store fence and required
            # P-ready arrival must be before the potentially blocking acquire.
            assert publication.count(f"cute.copy(flash_tiled_st{stage}, tSTrS") == (
                2 if split_p else 1
            )
            assert publication.count("cute.arch.fence_view_async_tmem_store()") == (
                2 if split_p else 1
            )
            assert (
                f"_helion_flash_rt.mbarrier_arrive(flash_pfor_ptr + {stage}"
                in publication
            )
            assert (
                f"_helion_flash_rt.mbarrier_arrive(flash_pfor2_ptr + {stage}"
                in publication
            ) is split_p
            after_acquire = section[acquire:]
            assert f"cute.copy(flash_tiled_st{stage}, tSTrS" not in after_acquire
            assert "cute.arch.fence_view_async_tmem_store()" not in after_acquire
        else:
            # The existing split helper owns conversion, stores and P arrivals.
            assert "_helion_flash_rt.fa4_sp_exp_convert_store" in publication
            assert f"flash_pfor_ptr + {stage}" in publication
            assert (f"flash_pfor2_ptr + {stage}" in publication) is split_p

    correction = source[correction_start:]
    ready0 = "_helion_flash_rt.named_barrier_wait_unaligned(3 + warp_idx % 4, 64)"
    ready1 = "_helion_flash_rt.named_barrier_wait_unaligned(7 + warp_idx % 4, 64)"
    empty0 = "cute.arch.mbarrier_arrive(flash_s0_corr_empty_ptr + 0)"
    empty1 = "cute.arch.mbarrier_arrive(flash_s1_corr_empty_ptr + 0)"
    ordered = (
        ready0,
        empty0,
        ready1,
        "for flash_kv",
        "flash_a0 =",
        "_helion_flash_rt.mbarrier_arrive(flash_pfor_ptr + 0",
        empty1,
        "flash_a1 =",
        "_helion_flash_rt.mbarrier_arrive(flash_pfor_ptr + 1",
        empty0,
        empty1,
        ready0,
        "flash_inv_sum0 =",
        empty0,
        ready1,
        "flash_inv_sum1 =",
        empty1,
    )
    position = 0
    for token in ordered:
        position = correction.index(token, position) + len(token)


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("family", ("fa4", "fa4_2cta"))
@pytest.mark.parametrize("schedule", ("xu", "16/4"))
def test_dense_single_search_and_normalization_retain_every_legal_depth(
    head_dim: int, dtype: torch.dtype, family: str, schedule: str
) -> None:
    spec = _config_spec(head_dim, dtype)
    fragments = spec._cute_flash_autotune_fragments("fa4", family)
    transport = cast("EnumFragment", fragments[cute_flash.FLASH_STAT_TRANSPORT_KEY])
    assert {"ring2", "single"} <= set(transport.search_choices or transport.choices)
    schedule_spec = FlashScheduleSpec(
        head_dim=head_dim,
        kv_depth=2,
        cta_count=2 if family == "fa4_2cta" else 1,
        cooperative_mma=family == "fa4_2cta",
        multicast_kv=family == "fa4_2cta",
        stage_output=True,
        stat_depth=1,
        pipelined_stat_handoff=True,
    )
    for depth in range(2, max_fa4_kv_depth(schedule_spec) + 1):
        config = helion.Config.from_dict(
            {
                "block_sizes": [1, 128, 128],
                **_values(
                    cute_flash_pipeline_family=family,
                    cute_flash_e2e_schedule=schedule,
                    cute_flash_kv_stage=depth,
                ),
            }
        )
        spec.normalize(config)
        assert config.config[cute_flash.FLASH_STAT_TRANSPORT_KEY] == "single"
        assert config.config[cute_flash.FLASH_KV_STAGE_KEY] == depth


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("num_kv", (2, 4, 4096))
def test_compiler_seeds_retain_effective_single_split_and_xu_modes(
    head_dim: int, dtype: torch.dtype, num_kv: int
) -> None:
    spec = _config_spec(head_dim, dtype, num_kv=num_kv)
    resolved = [
        _resolve(head_dim, dtype, seed.config, num_kv=num_kv)
        for seed in _normalized_compiler_seeds(spec)
    ]
    modes = {
        config.exp2_impl
        for config in resolved
        if config.stat_transport == "single"
        and not config.softmax_disc
        and config.exp2_packet == "1x1"
        and config.rescale_threshold > 0.0
    }
    assert modes == {"split", "xu"}


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("causal", (False, True))
def test_dense_source_seeds_do_not_promote_ws_or_ordinary_causal(
    head_dim: int, dtype: torch.dtype, causal: bool
) -> None:
    spec = _config_spec(head_dim, dtype, causal=causal, requires_ws_overlap=not causal)
    for seed in _normalized_compiler_seeds(spec):
        resolved = spec._resolve_cute_flash_config(seed.config)
        assert resolved.stat_transport == "ring2"


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("schedule", ("xu", "16/4"))
def test_d128_ring2_depth_two_and_d64_existing_repair_are_preserved(
    dtype: torch.dtype, schedule: str
) -> None:
    values = _values(
        cute_flash_stat_transport="ring2",
        cute_flash_e2e_schedule=schedule,
        cute_flash_kv_stage=3,
    )
    assert (
        _resolve(128, dtype, values).stat_transport,
        _resolve(128, dtype, values).kv_stage,
    ) == ("ring2", 2)
    assert (
        _resolve(64, dtype, values).stat_transport,
        _resolve(64, dtype, values).kv_stage,
    ) == ("single", 3)
    source = _source(128, dtype, values)
    assert "flash_s_corr_prod_index ^= 1" in source
    assert "flash_s_corr_cons_index ^= 1" in source
    assert "flash_s_corr_prod_phase = cutlass.Int32(1)" not in source


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("schedule", ("xu", "16/4"))
def test_zero_threshold_keeps_conservative_single_protocol(
    head_dim: int, dtype: torch.dtype, schedule: str
) -> None:
    source = _source(
        head_dim,
        dtype,
        _values(cute_flash_e2e_schedule=schedule, cute_flash_rescale_threshold=0.0),
    )
    assert source.count("flash_s_corr_prod_phase = cutlass.Int32(0)") == 2
    assert "flash_s_corr_prod_phase = cutlass.Int32(1)" not in source


@pytest.mark.parametrize(
    ("head_dim", "transport", "threshold"),
    ((64, "single", 0.0), (128, "single", 0.0), (128, "ring2", 8.0)),
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_unacknowledged_xu_retains_late_probability_store_order(
    head_dim: int, transport: str, threshold: float, dtype: torch.dtype
) -> None:
    source = _source(
        head_dim,
        dtype,
        _values(
            cute_flash_stat_transport=transport, cute_flash_rescale_threshold=threshold
        ),
    )
    phase = "flash_s_corr_prod_phase = cutlass.Int32(0)"
    first = source.index(phase)
    second = source.index(phase, first + 1)
    softmax0 = source[first:second]
    recurrence = softmax0.index(
        "flash_row_sum = flash_row_sum * flash_alpha + flash_p_sum"
    )
    publication = softmax0.index("cute.copy(flash_tiled_st0, tSTrS")
    assert recurrence < publication


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_widened_transport_does_not_change_existing_defaults(
    head_dim: int, dtype: torch.dtype
) -> None:
    values: dict[str, object] = {cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4"}
    assert _resolve(head_dim, dtype, values).stat_transport == (
        "single" if head_dim == 64 else "ring2"
    )
    with patch.dict(os.environ, {"HELION_CUTE_FLASH_FA4_STAT_HANDOFF": "0"}):
        assert _resolve(head_dim, dtype, values).stat_transport == "ring2"


def test_short_reverse_packet_keeps_conservative_protocol() -> None:
    source = _source(
        64,
        torch.float16,
        _values(
            cute_flash_pipeline_family="fa4_2cta",
            cute_flash_exp2_packet="deg1_8x2_corr10",
            cute_flash_e2e_schedule="8/2",
        ),
    )
    assert source.count("flash_s_corr_prod_phase = cutlass.Int32(0)") == 2
    assert "flash_s_corr_prod_phase = cutlass.Int32(1)" not in source


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_ordinary_causal_and_ws_do_not_gain_dense_single_transport(
    head_dim: int, dtype: torch.dtype
) -> None:
    assert _resolve(head_dim, dtype, _values(), causal=True).stat_transport == "ring2"
    assert (
        _resolve(
            head_dim, dtype, _values(cute_flash_pipeline_family="ws_overlap")
        ).stat_transport
        == "ring2"
    )
    source = _source(head_dim, dtype, _values(), causal=True)
    assert "flash_s_corr_prod_phase = cutlass.Int32(1)" not in source


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("schedule", ("xu", "16/4"))
def test_single_final_does_not_expand_to_d128_or_xu(
    dtype: torch.dtype, schedule: str
) -> None:
    values = _values(
        cute_flash_stat_transport="single_final",
        cute_flash_e2e_schedule=schedule,
        cute_flash_persistent=False,
    )
    assert _resolve(128, dtype, values).stat_transport == "ring2"
    assert _resolve(64, dtype, values).stat_transport == (
        "single" if schedule == "xu" else "single_final"
    )


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("cta_count", (1, 2))
@pytest.mark.parametrize("kv_iterations", (1, 2, 3, 4, 7, 8))
def test_acknowledged_schedule_carries_odd_even_terminal_credit(
    head_dim: int, cta_count: int, kv_iterations: int
) -> None:
    spec = FlashScheduleSpec(
        head_dim=head_dim,
        kv_depth=2,
        cta_count=cta_count,
        multicast_kv=cta_count == 2,
        cooperative_mma=cta_count == 2,
        persistent=True,
        kv_iterations=kv_iterations,
        stat_depth=1,
        pipelined_stat_handoff=True,
    )
    spec = dataclasses.replace(spec, kv_depth=max_fa4_kv_depth(spec))
    verified = verify_flash_schedule(build_fa4_schedule(spec))
    stats = [
        cycle
        for cycle in verified.schedule.phase_cycles
        if cycle.barrier.startswith(("stat_ready", "stat_empty"))
    ]
    assert len(stats) == cta_count * 4
    assert all(cycle.uses_per_work == kv_iterations + 1 for cycle in stats)
    for cycle in stats:
        initial = 1 if cycle.barrier.startswith("stat_empty") else 0
        assert cycle.phases == (initial, initial ^ ((kv_iterations + 1) & 1), initial)


def test_causal_cross_slot_schedule_still_requires_equal_iteration_proof() -> None:
    with pytest.raises(FlashScheduleError, match="equal"):
        build_fa4_schedule(
            FlashScheduleSpec(
                head_dim=128,
                kv_depth=3,
                causal=True,
                stat_depth=1,
                pipelined_stat_handoff=True,
            )
        )
