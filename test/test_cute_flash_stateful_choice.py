from __future__ import annotations

import ast
from contextlib import ExitStack
import dataclasses
import inspect
import os
import random
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

from examples.attention import causal_attention_output
import pytest
import torch

from ._cute_binding import _forbid_native_compile
from ._cute_binding import _mock_cuda_unavailable
from .test_cute_flash_exp2 import _emit_causal_resident_native_source
from .test_cute_flash_exp2 import _flash_runtime
from .test_cute_flash_resident_choice import _emit as _emit_explicit_dense
import helion
from helion._compiler.autotuner_heuristics.cute import CuteFlashAttentionHeuristic
from helion._compiler.backend import CuteBackend
from helion._compiler.cute import cute_flash as flash
from helion._compiler.cute.attention_plan import SOFTCAP_KIND
from helion._compiler.cute.attention_plan import AttentionScoreModifier
from helion._compiler.cute.attention_plan import causal_score_plan
from helion._compiler.cute.flash_tuning import FlashTuningPolicy
from helion._compiler.cute.mma_support import CuteMmaSupport
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.exc import InvalidConfig

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.device_function import DeviceFunction
    from helion._compiler.device_ir import DeviceIR


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        _forbid_native_compile(),
        patch.dict(os.environ, {}, clear=True),
    ):
        yield


def _resolve(
    head_dim: int = 64,
    num_kv: int = 6,
    *,
    dtype: torch.dtype = torch.float16,
    overrides: dict[str, object] | None = None,
    proof: bool = True,
    standard_output: bool = True,
    capability: tuple[int, int] = (10, 3),
) -> flash.FlashAttentionConfig:
    return flash.resolve_flash_config(
        head_dim,
        num_kv,
        {
            flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
            flash.FLASH_SOFTMAX_LOWERING_KEY: "resident_stateful",
            **({} if overrides is None else overrides),
        },
        dtype=dtype,
        is_causal=True,
        standard_causal_output=standard_output,
        causal_resident_compatible=proof,
        target_device_capability=capability,
    )


def _emit(
    head_dim: int = 64,
    num_kv: int = 6,
    *,
    dtype: torch.dtype = torch.float16,
    overrides: dict[str, object] | None = None,
    capability: tuple[int, int] = (10, 3),
    extent: int | None = None,
    additional_modifier: bool = False,
    has_lse: bool = False,
) -> tuple[flash.FlashAttentionConfig, str]:
    config = _resolve(
        head_dim, num_kv, dtype=dtype, overrides=overrides, capability=capability
    )
    score = causal_score_plan(head_dim)
    if additional_modifier:
        score = dataclasses.replace(
            score,
            modifiers=(
                *score.modifiers,
                AttentionScoreModifier(SOFTCAP_KIND, value_log2=1.0),
            ),
        )
    body = flash.emit_flash_fa4_device_body(
        cast("DeviceFunction", None),
        head_dim=head_dim,
        num_kv=num_kv,
        sequence_extent=num_kv * 128 if extent is None else extent,
        num_bh=1,
        total_tiles=num_kv // 2,
        cfg=config,
        has_lse=has_lse,
        io_dtype="cutlass.Float16" if dtype is torch.float16 else "cutlass.BFloat16",
        score_plan=score,
        target_device_capability=capability,
    )
    return config, ast.unparse(ast.Module(body=body, type_ignores=[]))


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("rowmax", ("software", "tmem"))
def test_stateful_general_masked_unmasked_lowering(
    head_dim: int, dtype: torch.dtype, rowmax: str
) -> None:
    for num_kv in (2, 6, 32, 100, 512):
        config, source = _emit(
            head_dim, num_kv, dtype=dtype, overrides={flash.FLASH_ROWMAX_KEY: rowmax}
        )
        assert config.softmax_lowering == "resident_stateful"
        assert config.rowmax == rowmax
        assert not config.softmax_disc
        assert config.sp_row_sum == "whole"
        assert config.stat_transport == "single"
        assert source.count("ResidentSoftmaxState.create") == 2
        assert "resident_softmax_value_graph" not in source
        assert (
            "cutlass.BFloat16" in source
            if dtype is torch.bfloat16
            else "cutlass.Float16" in source
        )
        module = ast.parse(source)
        masked = [
            n
            for n in ast.walk(module)
            if isinstance(n, ast.For) and ast.unparse(n.target) == "flash_kv_mask_iter"
        ]
        unmasked = [
            n
            for n in ast.walk(module)
            if isinstance(n, ast.For)
            and ast.unparse(n.target) == "flash_kv_unmask_iter"
        ]
        assert len(masked) == len(unmasked) == 2
        for loop in masked:
            text = ast.unparse(loop)
            assert "causal_mask_t2r" in text
            assert "update_row_max_masked" in text
            assert "update_row_max_precomputed" not in text
            assert "tLDRed" not in text
        for loop in unmasked:
            text = ast.unparse(loop)
            assert "causal_mask_t2r" not in text
            assert ("update_row_max_precomputed" in text) == (rowmax == "tmem")
            assert ("tLDRed" in text) == (rowmax == "tmem")
        assert ("LdRed32x32bOp" in source) == (rowmax == "tmem")


@pytest.mark.parametrize("threshold", (0.0, 8.0))
@pytest.mark.parametrize("head_dim", (64, 128))
def test_stateful_p_publication_precedes_same_slot_credit(
    threshold: float, head_dim: int
) -> None:
    config, source = _emit(
        head_dim, overrides={flash.FLASH_RESCALE_THRESHOLD_KEY: threshold}
    )
    assert config.rescale_threshold == threshold
    previous = 0
    for stage in (0, 1):
        begin = source.index(
            "flash_softmax = _helion_flash_rt.ResidentSoftmaxState.create", previous
        )
        previous = begin + 1
        first = source[begin : source.index("for flash_kv_mask_iter", begin)]
        assert "update_row_max_masked(tLDrS.load(), True)" in first
        assert "update_row_sum(tLDrS.load(), flash_alpha, True)" in first
        assert (
            first.index("update_row_max_masked")
            < first.index("named_barrier_arrive_unaligned")
            < first.index("apply_exp2_convert")
        )
        assert (
            first.index("mbarrier_arrive(flash_pfor_ptr")
            < first.index("mbarrier_arrive(flash_pfor2_ptr")
            < first.index(f"mbar_spin_wait(flash_s{stage}_corr_empty_ptr")
            < first.index("update_row_sum")
        )
        assert f"flash_s{stage}_corr_empty_ptr" in first
    # First all-masked tile has safe zero state; conversion uses the destination
    # dtype, rather than copying the FP16-only value-graph helper.
    runtime = inspect.getsource(_flash_runtime.ResidentSoftmaxState)
    assert "else Float32(0.0)" in runtime
    assert "acc_scale = Float32(0.0)" in runtime
    assert "to(converted.element_type)" in runtime
    assert "cutlass.Float16" not in runtime


def test_stateful_software_lowering_is_independent_of_ldred_capability() -> None:
    config, source = _emit(
        128,
        dtype=torch.bfloat16,
        capability=(10, 0),
        overrides={flash.FLASH_ROWMAX_KEY: "tmem"},
    )
    assert config.rowmax == "software"
    assert "ResidentSoftmaxState.create" in source
    assert "LdRed32x32bOp" not in source


@pytest.mark.parametrize(
    "family",
    (
        "ws_overlap",
        "fa4_2cta",
        "fa4_2cta_causal",
        "fa4_deep_1cta",
        "fa4_local_tma",
        "fa4_cga2_local",
        "fa4_clc",
        "fa4_tma_4d",
    ),
)
def test_stateful_does_not_silently_change_incompatible_family(family: str) -> None:
    with pytest.raises(InvalidConfig, match="resident_stateful"):
        _resolve(overrides={flash.FLASH_PIPELINE_FAMILY_KEY: family})


@pytest.mark.parametrize("num_kv", (1, 3, 5))
def test_stateful_requires_complete_query_pairs(num_kv: int) -> None:
    with pytest.raises(InvalidConfig, match="paired coverage"):
        _resolve(num_kv=num_kv)


def test_stateful_rechecks_proof_and_rejects_other_semantics() -> None:
    for changes in ({"proof": False}, {"standard_output": False}, {"head_dim": 256}):
        with pytest.raises(InvalidConfig, match="paired coverage"):
            _resolve(**changes)
    for changes in ({"extent": 767}, {"additional_modifier": True}, {"has_lse": True}):
        with pytest.raises(AssertionError):
            _emit(**changes)
    with (
        patch.dict(os.environ, {"HELION_CUTE_FLASH_MMA_PTX": "0"}),
        pytest.raises(InvalidConfig, match="PTX"),
    ):
        _resolve()


def _spec(
    head_dim: int, num_kv: int, dtype: torch.dtype, proof: bool = True
) -> ConfigSpec:
    spec = ConfigSpec(
        backend=CuteBackend(),
        target_device_capability=(10, 3),
        device=torch.device("cpu"),
        num_sm=152,
    )
    for index, target in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=index, size_hint=target))
    spec.enable_cute_flash_search(
        head_dim=head_dim,
        num_kv=num_kv,
        dtype=dtype,
        block_size_targets={0: 1, 1: 128, 2: 128},
        is_causal=True,
        standard_causal_output=True,
        causal_resident_compatible=proof,
    )
    return spec


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_stateful_seed_routes_and_round_trip(head_dim: int, dtype: torch.dtype) -> None:
    for num_kv in (6, 32, 100, 512):
        spec = _spec(head_dim, num_kv, dtype)
        seeds = spec.autotune_seed_configs()
        env = cast("CompileEnvironment", SimpleNamespace(config_spec=spec))
        assert seeds == CuteFlashAttentionHeuristic.get_seed_configs(
            env, cast("DeviceIR", None)
        )
        assert seeds[0] == CuteFlashAttentionHeuristic.get_seed_config(
            env, cast("DeviceIR", None)
        )
        stateful = [
            s
            for s in seeds
            if s.config.get(flash.FLASH_SOFTMAX_LOWERING_KEY) == "resident_stateful"
        ]
        assert len(stateful) == 4
        assert [seed.config[flash.FLASH_ROWMAX_KEY] for seed in stateful] == [
            "software",
            "tmem",
            "software",
            "tmem",
        ]
        assert [
            seed.config.get(flash.FLASH_ROW_SUM_SCHEDULE_KEY, "post_acquire")
            for seed in stateful
        ] == ["post_acquire", "post_acquire", "pre_acquire", "pre_acquire"]
        spec.compiler_seed_configs = seeds
        generation = ConfigGeneration(spec)
        software = generation.unflatten(generation.flatten(stateful[0]))
        tmem = generation.unflatten(generation.flatten(stateful[1]))
        assert software.config | {flash.FLASH_ROWMAX_KEY: "tmem"} == tmem.config
        for seed in stateful:
            normalized = generation.unflatten(generation.flatten(seed))
            assert generation.unflatten(generation.flatten(normalized)) == normalized
            assert any(
                candidate == normalized
                for _, candidate in generation.seed_flat_config_pairs()
            )
        for rowmax in ("software", "tmem"):
            candidate = helion.Config.from_dict(
                {
                    **normalized.config,
                    flash.FLASH_ROWMAX_KEY: rowmax,
                    flash.FLASH_KV_STAGE_KEY: 3,
                    flash.FLASH_SOFTMAX_REGS_KEY: 176,
                    flash.FLASH_CORR_REGS_KEY: 88,
                    flash.FLASH_OTHER_REGS_KEY: 56,
                }
            )
            spec.normalize(candidate)
            effective = spec._resolve_cute_flash_config(candidate.config)
            assert effective.softmax_lowering == "resident_stateful"
            assert effective.rowmax == rowmax
            assert effective.kv_stage == 3
            assert (
                effective.softmax_regs,
                effective.corr_regs,
                effective.other_regs,
            ) == (176, 88, 56)
            assert effective.stat_transport == "single" and not effective.softmax_disc
            assert effective.exp2_packet == "1x1"
            assert effective.exp2_impl == "xu"
            assert effective.e2e_offset == effective.e2e_offset0 == 0


def test_stateful_is_unavailable_without_detector_proof() -> None:
    spec = _spec(64, 32, torch.float16, False)
    seeds = spec.autotune_seed_configs()
    assert all(
        seed.config.get(flash.FLASH_SOFTMAX_LOWERING_KEY) != "resident_stateful"
        for seed in seeds
    )
    fragment = spec._cute_flash_autotune_fragments()[flash.FLASH_SOFTMAX_LOWERING_KEY]
    assert "resident_stateful" not in fragment.search_choices


def test_stateful_override_rejects_conflicting_children() -> None:
    spec = _spec(64, 32, torch.float16)
    config = {
        flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
        flash.FLASH_SOFTMAX_LOWERING_KEY: "resident_stateful",
    }
    with pytest.raises(InvalidConfig, match="requires cute_flash_causal_loop_split"):
        spec.prepare_override_normalization(
            config,
            {
                flash.FLASH_SOFTMAX_LOWERING_KEY: "resident_stateful",
                flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: False,
            },
        )


@pytest.mark.parametrize("num_kv", (512, 1024, 2048, 4096))
def test_stateful_keeps_legacy_auto_and_standard_semantics(num_kv: int) -> None:
    legacy = _emit_causal_resident_native_source(num_kv=num_kv)
    assert legacy == _emit_causal_resident_native_source(
        num_kv=num_kv, config_overrides={flash.FLASH_SOFTMAX_LOWERING_KEY: "auto"}
    )
    standard = _emit_causal_resident_native_source(
        num_kv=num_kv, config_overrides={flash.FLASH_SOFTMAX_LOWERING_KEY: "standard"}
    )
    assert "ResidentSoftmaxState.create" not in standard
    assert "resident_softmax_value_graph" not in standard


@pytest.mark.parametrize("sequence", (127, 128, 256, 257, 384, 768))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_stateful_detector_proof_reaches_real_codegen(
    sequence: int, dtype: torch.dtype
) -> None:
    support = CuteMmaSupport(
        universal=True,
        warp_f16bf16=True,
        warpgroup_f16bf16=True,
        tcgen05_f16bf16=True,
        tcgen05_f8=True,
        tcgen05_tf32=True,
    )
    with ExitStack() as stack:
        for target, value in (
            ("helion.runtime.kernel.target_device_capability", (10, 3)),
            ("helion._compiler.compile_environment.target_device_capability", (10, 3)),
            ("helion.runtime.get_num_sm", 152),
            ("helion._compiler.cute.mma_support.get_cute_mma_support", support),
        ):
            stack.enter_context(patch(target, return_value=value))
        kernel = helion.kernel(
            causal_attention_output.fn,
            backend="cute",
            static_shapes=True,
            autotune_effort="none",
        )
        inputs = tuple(
            torch.empty(
                1,
                2,
                sequence,
                128,
                dtype=dtype,
                device="meta",  # @ignore-device-lint
            )
            for _ in range(3)
        )
        bound = kernel._bind_isolated(inputs)
        spec = bound.config_spec
        if sequence % 128:
            assert not spec.cute_flash_search_enabled
            assert not spec._cute_flash_causal_resident_compatible
            return
        assert spec.cute_flash_search_enabled
        assert spec._cute_flash_causal_resident_compatible == (sequence % 256 == 0)
        seeds = spec.autotune_seed_configs()
        stateful = [
            seed
            for seed in seeds
            if seed.config.get(flash.FLASH_SOFTMAX_LOWERING_KEY) == "resident_stateful"
        ]
        if sequence % 256:
            assert not stateful
            return
        expected_variants = {
            (width, schedule, rowmax)
            for width in (1, 2)
            for schedule in ("post_acquire", "pre_acquire")
            for rowmax in ("software", "tmem")
        }
        assert len(stateful) == len(expected_variants)
        assert {
            (
                seed.config[flash.FLASH_CAUSAL_LPT_SWIZZLE_KEY],
                seed.config.get(flash.FLASH_ROW_SUM_SCHEDULE_KEY, "post_acquire"),
                seed.config[flash.FLASH_ROWMAX_KEY],
            )
            for seed in stateful
        } == expected_variants
        for seed in stateful:
            source = bound.to_code(seed)
            assert source.count("ResidentSoftmaxState.create") == 2
            assert "resident_softmax_value_graph" not in source


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("causal", (False, True))
def test_explicit_resident_does_not_consult_length_policy(
    causal: bool, dtype: torch.dtype
) -> None:
    with (
        patch.object(
            FlashTuningPolicy,
            "dense_policy",
            side_effect=AssertionError("explicit dense policy lookup"),
        ),
        patch.object(
            FlashTuningPolicy,
            "causal_policy",
            side_effect=AssertionError("explicit causal policy lookup"),
        ),
    ):
        _, source = _emit(dtype=dtype) if causal else _emit_explicit_dense(dtype=dtype)
    assert (
        "ResidentSoftmaxState.create" if causal else "resident_softmax_value_graph"
    ) in source


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    ("key", "value"),
    (
        (flash.FLASH_SOFTMAX_DISC_KEY, True),
        (flash.FLASH_STAT_TRANSPORT_KEY, "ring2"),
        (flash.FLASH_CAUSAL_KV_ORDER_KEY, "ascending"),
        (flash.FLASH_CAUSAL_LOOP_SPLIT_KEY, False),
        (flash.FLASH_KV_TILE_N_KEY, 160),
    ),
)
def test_sampled_stateful_cannot_overwrite_fixed_controls(
    dtype: torch.dtype, key: str, value: object
) -> None:
    spec = _spec(64, 32, dtype)
    config = {flash.FLASH_SOFTMAX_LOWERING_KEY: "resident_stateful", key: value}
    before = dict(config)
    with pytest.raises(InvalidConfig, match=f"requires {key}="):
        spec.prepare_override_normalization(config, {key: value})
    assert config == before


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_sampled_stateful_keeps_compatible_fixed_controls(dtype: torch.dtype) -> None:
    spec = _spec(64, 32, dtype)
    config = {
        flash.FLASH_SOFTMAX_LOWERING_KEY: "resident_stateful",
        flash.FLASH_STAT_TRANSPORT_KEY: "single",
        flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
    }
    spec.prepare_override_normalization(
        config,
        {
            flash.FLASH_STAT_TRANSPORT_KEY: "single",
            flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
        },
    )
    assert config[flash.FLASH_SOFTMAX_LOWERING_KEY] == "resident_stateful"


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_stateful_sampling_retries_without_losing_fixed_control(
    dtype: torch.dtype,
) -> None:
    spec = _spec(64, 32, dtype)
    generation = spec.create_config_generation(
        overrides={flash.FLASH_SOFTMAX_DISC_KEY: True}
    )
    saved = random.getstate()
    try:
        random.seed(20261003)
        for _ in range(8):
            config = generation.random_config()
            assert config.config[flash.FLASH_SOFTMAX_DISC_KEY] is True
            assert (
                config.config[flash.FLASH_SOFTMAX_LOWERING_KEY] != "resident_stateful"
            )
    finally:
        random.setstate(saved)
