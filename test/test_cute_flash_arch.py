from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from helion._compiler.cute import cute_flash
from helion._compiler.cute import flash_tuning
from helion._compiler.cute.attention_plan import dense_score_plan
from helion._compiler.cute.flash_arch import FlashHardwareCapabilities
from helion._compiler.cute.flash_policy import FlashTargetPolicy
from helion._compiler.cute.flash_policy import flash_target_policy_cache_identity
from helion._compiler.cute.flash_policy import get_flash_target_policy
from helion._compiler.cute.flash_policy import registered_flash_target_policies
from helion._compiler.cute.flash_tuning import FlashCausalSeedTemplate
from helion._compiler.cute.flash_tuning import FlashCausalTuningPolicy
from helion._compiler.cute.flash_tuning import FlashDenseTuningPolicy
from helion._compiler.cute.flash_tuning import FlashPackedExp2Mode
from helion._compiler.cute.flash_tuning import FlashSoftmaxLowering
from helion._compiler.cute.flash_tuning import FlashTuningDType
from helion._compiler.cute.flash_tuning import FlashTuningPolicy
from helion._compiler.cute.flash_tuning import FlashTuningWorkload

if TYPE_CHECKING:
    from helion._compiler.device_function import DeviceFunction


def test_resident_causal_seed_template_is_head_dim_agnostic() -> None:
    """RESIDENT_V1 must not inherit DEGREE2_V1's FP16/hd64 restriction.

    DEGREE2_V1 pins the degree-2 compound exp2 packet, which only exists at
    FP16/hd64. The resident lowering rewrites exp2_packet to "1x1" anyway, so a
    resident seed carries no such restriction and may be registered for other
    workloads (e.g. head_dim=128).
    """
    resident = FlashCausalTuningPolicy(
        num_kv=512,
        kv_stage=3,
        seed_template=FlashCausalSeedTemplate.RESIDENT_V1,
        softmax_lowering=FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
    )
    assert not resident.requires_fp16_hd64

    degree2 = FlashCausalTuningPolicy(num_kv=512, kv_stage=3)
    assert degree2.seed_template is FlashCausalSeedTemplate.DEGREE2_V1
    assert degree2.requires_fp16_hd64

    # A resident seed template without a resident lowering is contradictory.
    with pytest.raises(ValueError, match="resident causal seed template"):
        FlashCausalTuningPolicy(
            num_kv=512,
            kv_stage=3,
            seed_template=FlashCausalSeedTemplate.RESIDENT_V1,
            softmax_lowering=FlashSoftmaxLowering.STANDARD,
        )

    # The resident template must not emit the hd64-only compound exp2 packet.
    overrides = cute_flash._flash_causal_tuning_overrides(resident)
    assert cute_flash.FLASH_EXP2_PACKET_KEY not in overrides
    assert overrides[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "fa4"


def test_sm103_flash_target_policy() -> None:
    target_policy = get_flash_target_policy((10, 3))
    capabilities = target_policy.hardware
    policy = target_policy.tuning

    assert dict(registered_flash_target_policies())[(10, 3)] == target_policy
    assert capabilities.supports_tmem_row_reduce
    assert capabilities.supports_packed_f16x2_exp2
    assert policy.workload == FlashTuningWorkload()
    assert policy.tmem_row_reduce_min_kv == 256
    assert (
        flash_target_policy_cache_identity(
            (10, 3), head_dim=64, torch_dtype="float16", num_kv=256
        )
        is not None
    )
    # head_dim=128 FP16 now carries causal resident-softmax seeds, so it
    # resolves a policy identity like head_dim=64 does.
    assert (
        flash_target_policy_cache_identity(
            (10, 3), head_dim=128, torch_dtype="float16", num_kv=256
        )
        is not None
    )
    hd128_tuning = target_policy.tuning_for_cute(128, "cutlass.Float16")
    assert hd128_tuning is not None
    assert hd128_tuning.workload == FlashTuningWorkload(
        head_dim=128, dtype=FlashTuningDType.FLOAT16
    )
    assert hd128_tuning.dense_policies == ()
    hd128_causal = hd128_tuning.causal_policy(512)
    assert hd128_causal is not None
    assert hd128_causal.seed_template is FlashCausalSeedTemplate.RESIDENT_V1
    assert (
        hd128_causal.softmax_lowering
        is FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH
    )
    # The resident template must not carry the FP16/hd64-only restriction.
    assert not hd128_causal.requires_fp16_hd64
    assert (
        flash_target_policy_cache_identity(
            (10, 3), head_dim=64, torch_dtype="bfloat16", num_kv=256
        )
        is None
    )
    bfloat_dense_policy = FlashDenseTuningPolicy(
        512,
        "1x1",
        "8/2",
        0,
        0,
        "ring2",
        pipeline_family="fa4_2cta",
        kv_stage=3,
        rescale_threshold=0,
    )
    bfloat_tuning = FlashTuningPolicy(
        workload=FlashTuningWorkload(
            head_dim=128,
            dtype=FlashTuningDType.BFLOAT16,
        ),
        tmem_row_reduce_min_kv=512,
        dense_policies=(bfloat_dense_policy,),
    )
    multi_workload_policy = FlashTargetPolicy(
        hardware=capabilities,
        tuning=policy,
        additional_tunings=(bfloat_tuning,),
    )
    assert multi_workload_policy.tuning_for_torch(128, "bfloat16") is bfloat_tuning
    assert (
        multi_workload_policy.tuning_for_cute(128, "cutlass.BFloat16") is bfloat_tuning
    )
    with patch.object(
        cute_flash, "get_flash_target_policy", return_value=multi_workload_policy
    ):
        bfloat_seed = cute_flash.flash_attention_seed_config(
            128,
            512,
            dtype=torch.bfloat16,
            standard_dense_output=True,
            target_device_capability=(10, 3),
        )
    assert bfloat_seed is not None
    bfloat_config = cute_flash.resolve_flash_config(
        128,
        512,
        bfloat_seed.config,
        dtype=torch.bfloat16,
        is_causal=False,
        standard_dense_output=True,
    )
    with patch.object(
        cute_flash, "get_flash_target_policy", return_value=multi_workload_policy
    ):
        bfloat_body = cute_flash.emit_flash_fa4_device_body(
            cast("DeviceFunction", None),
            head_dim=128,
            num_kv=512,
            sequence_extent=65_536,
            num_bh=64,
            total_tiles=8192,
            cfg=bfloat_config,
            has_lse=False,
            io_dtype="cutlass.BFloat16",
            score_plan=dense_score_plan(128),
            target_device_capability=(10, 3),
        )
    bfloat_source = ast.unparse(ast.Module(body=bfloat_body, type_ignores=[]))
    assert "LdRed32x32bOp" in bfloat_source
    with patch(
        "helion._compiler.cute.flash_policy.get_flash_target_policy",
        return_value=multi_workload_policy,
    ):
        bfloat_identity = flash_target_policy_cache_identity(
            (10, 3), head_dim=128, torch_dtype="bfloat16", num_kv=512
        )
    changed_bfloat_tuning = dataclasses.replace(
        bfloat_tuning,
        tmem_row_reduce_min_kv=1024,
    )
    with patch(
        "helion._compiler.cute.flash_policy.get_flash_target_policy",
        return_value=dataclasses.replace(
            multi_workload_policy, additional_tunings=(changed_bfloat_tuning,)
        ),
    ):
        changed_bfloat_identity = flash_target_policy_cache_identity(
            (10, 3), head_dim=128, torch_dtype="bfloat16", num_kv=512
        )
    assert bfloat_identity != changed_bfloat_identity

    expected_dense = {
        256: (
            "deg1_8x2_corr10",
            "8/2",
            5,
            1,
            "single",
            "fa4_2cta",
            6,
            False,
            8.0,
            FlashPackedExp2Mode.DISABLED,
            7,
            FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
            64,
            40,
        ),
        512: (
            "deg1_8x2_corr10",
            "8/2",
            2,
            1,
            "single",
            "fa4_2cta",
            6,
            False,
            8.0,
            FlashPackedExp2Mode.DISABLED,
            7,
            FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
            72,
            40,
        ),
        1024: (
            "deg1_8x2_corr10",
            "8/2",
            2,
            1,
            "single",
            "fa4_2cta",
            6,
            False,
            8.0,
            FlashPackedExp2Mode.DISABLED,
            7,
            FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
            72,
            40,
        ),
        2048: (
            "deg1_16x8",
            "16/8",
            0,
            10,
            "single_final",
            "fa4_2cta",
            6,
            False,
            8.0,
            FlashPackedExp2Mode.ALL_XU,
            7,
            FlashSoftmaxLowering.STANDARD,
            80,
            32,
        ),
    }
    assert {shape.num_kv for shape in policy.dense_policies} == set(expected_dense)
    for num_kv, expected in expected_dense.items():
        shape = policy.dense_policy(num_kv)
        assert shape is not None
        assert (
            shape.exp2_packet,
            shape.e2e_schedule,
            shape.e2e_offset,
            shape.e2e_offset0,
            shape.stat_transport,
            shape.pipeline_family,
            shape.kv_stage,
            shape.persistent,
            shape.rescale_threshold,
            shape.packed_exp2_mode,
            shape.probability_log2_shift,
            shape.softmax_lowering,
            shape.corr_regs,
            shape.other_regs,
        ) == expected
    assert policy.dense_policy(4096) is None

    expected_causal = {
        512: (
            8,
            FlashCausalSeedTemplate.DEGREE2_V1,
            15,
            3,
            "fa4",
            True,
            FlashSoftmaxLowering.STATEFUL,
            200,
            2,
            True,
            "descending",
        ),
        1024: (
            3,
            FlashCausalSeedTemplate.DEGREE2_V1,
            1,
            14,
            "fa4",
            False,
            FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
            184,
            0,
            True,
            "descending",
        ),
        2048: (
            3,
            FlashCausalSeedTemplate.DEGREE2_V1,
            14,
            12,
            "fa4",
            False,
            FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
            184,
            0,
            True,
            "descending",
        ),
        4096: (
            6,
            FlashCausalSeedTemplate.DEGREE2_V1,
            14,
            12,
            "helion",
            False,
            FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
            184,
            0,
            True,
            "descending",
        ),
    }
    assert {shape.num_kv for shape in policy.causal_policies} == set(expected_causal)
    for num_kv, expected in expected_causal.items():
        shape = policy.causal_policy(num_kv)
        assert shape is not None
        assert (
            shape.kv_stage,
            shape.seed_template,
            shape.e2e_offset,
            shape.e2e_offset0,
            shape.role_map,
            shape.epi_tma,
            shape.softmax_lowering,
            shape.softmax_regs,
            shape.first_load_order,
            shape.causal_loop_split,
            shape.causal_kv_order,
        ) == expected
    assert policy.causal_policy(256) is None


@pytest.mark.parametrize("capability", [None, (10, 0), (999, 999)])
def test_unknown_and_b200_flash_target_policy_is_generic(
    capability: tuple[int, int] | None,
) -> None:
    target_policy = get_flash_target_policy(capability)
    capabilities = target_policy.hardware
    policy = target_policy.tuning

    assert not capabilities.supports_tmem_row_reduce
    assert not capabilities.supports_packed_f16x2_exp2
    assert policy.tmem_row_reduce_min_kv is None
    assert not policy.dense_policies
    assert policy.dense_policy(256) is None
    assert not policy.causal_policies
    assert policy.causal_policy(512) is None


def test_flash_policy_rejects_duplicate_shapes() -> None:
    dense = FlashDenseTuningPolicy(256, "1x1", "8/2", 0, 0, "single")
    causal = FlashCausalTuningPolicy(512, 3)

    with pytest.raises(ValueError, match="dense KV sizes must be unique"):
        FlashTuningPolicy(dense_policies=(dense, dense))
    with pytest.raises(ValueError, match="causal KV sizes must be unique"):
        FlashTuningPolicy(causal_policies=(causal, causal))


def test_flash_policy_rejects_invalid_field_combinations() -> None:
    with pytest.raises(ValueError, match="head dimension"):
        FlashTuningWorkload(head_dim=0)
    with pytest.raises(ValueError, match="workloads must be unique"):
        FlashTargetPolicy(additional_tunings=(FlashTuningPolicy(),))
    with pytest.raises(ValueError, match="TMEM row-reduce minimum KV size"):
        FlashTuningPolicy(tmem_row_reduce_min_kv=0)
    for shift in (-1, 1.5, float("nan"), True):
        with pytest.raises(ValueError, match="probability log2 shift"):
            FlashDenseTuningPolicy(
                256,
                "1x1",
                "8/2",
                0,
                0,
                "single",
                probability_log2_shift=cast("int", shift),
            )
    for threshold in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="rescale threshold must be finite"):
            FlashDenseTuningPolicy(
                256,
                "1x1",
                "8/2",
                0,
                0,
                "single",
                rescale_threshold=threshold,
            )
    with pytest.raises(ValueError, match="register counts must be set together"):
        FlashDenseTuningPolicy(
            512,
            "1x1",
            "8/2",
            0,
            0,
            "single",
            corr_regs=72,
        )
    with pytest.raises(ValueError, match="first-load order must be 0-4"):
        FlashCausalTuningPolicy(512, 8, first_load_order=5)
    with pytest.raises(ValueError, match="positive multiple of 8"):
        FlashCausalTuningPolicy(512, 8, softmax_regs=199)
    with pytest.raises(ValueError, match="end-to-end offsets"):
        FlashCausalTuningPolicy(512, 8, e2e_offset=-1)
    with pytest.raises(ValueError, match="role map"):
        FlashCausalTuningPolicy(512, 8, role_map="")
    with pytest.raises(ValueError, match="exp2 packet"):
        FlashDenseTuningPolicy(256, "unknown", "8/2", 0, 0, "single")
    with pytest.raises(ValueError, match="end-to-end schedule"):
        FlashDenseTuningPolicy(256, "1x1", "unknown", 0, 0, "single")
    with pytest.raises(ValueError, match="statistics transport"):
        FlashDenseTuningPolicy(256, "1x1", "8/2", 0, 0, "unknown")
    with pytest.raises(ValueError, match="pipeline family"):
        FlashDenseTuningPolicy(
            256, "1x1", "8/2", 0, 0, "single", pipeline_family="unknown"
        )
    with pytest.raises(ValueError, match="role map"):
        FlashCausalTuningPolicy(512, 8, role_map="unknown")
    with pytest.raises(ValueError, match="unsupported causal softmax lowering"):
        FlashCausalTuningPolicy(
            512,
            8,
            softmax_lowering=cast("FlashSoftmaxLowering", "standard"),
        )
    with pytest.raises(ValueError, match="unsupported dense softmax lowering"):
        FlashDenseTuningPolicy(
            256,
            "1x1",
            "8/2",
            0,
            0,
            "single",
            softmax_lowering=FlashSoftmaxLowering.STATEFUL,
        )
    with pytest.raises(ValueError, match="unsupported dense softmax lowering"):
        FlashDenseTuningPolicy(
            256,
            "1x1",
            "8/2",
            0,
            0,
            "single",
            softmax_lowering=cast("FlashSoftmaxLowering", "standard"),
        )
    with pytest.raises(ValueError, match="unsupported dense packed exp2 mode"):
        FlashDenseTuningPolicy(
            256,
            "1x1",
            "8/2",
            0,
            0,
            "single",
            packed_exp2_mode=cast("FlashPackedExp2Mode", "disabled"),
        )
    with pytest.raises(ValueError, match="single statistics transport"):
        FlashDenseTuningPolicy(
            256,
            "1x1",
            "8/2",
            0,
            0,
            "ring2",
            softmax_lowering=FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
        )
    with pytest.raises(ValueError, match="nonpersistent fa4_2cta"):
        FlashDenseTuningPolicy(
            256,
            "deg1_8x2_corr10",
            "8/2",
            0,
            0,
            "single",
            pipeline_family="fa4",
            softmax_lowering=FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
        )
    with pytest.raises(ValueError, match="degree-1 exp2 packet"):
        FlashDenseTuningPolicy(
            2048,
            "8x2",
            "8/2",
            0,
            0,
            "single_final",
            packed_exp2_mode=FlashPackedExp2Mode.ALL_XU,
        )
    with pytest.raises(ValueError, match="exceed fp16 range"):
        FlashDenseTuningPolicy(
            2048,
            "deg1_16x8",
            "16/8",
            0,
            0,
            "single_final",
            probability_log2_shift=8,
            packed_exp2_mode=FlashPackedExp2Mode.ALL_XU,
        )
    with pytest.raises(ValueError, match="descending split loop"):
        FlashCausalTuningPolicy(
            768,
            4,
            softmax_lowering=FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
            causal_loop_split=False,
        )
    with pytest.raises(ValueError, match="descending split loop"):
        FlashCausalTuningPolicy(
            768,
            4,
            softmax_lowering=FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
            causal_kv_order="ascending",
        )


def test_target_policy_rejects_tuning_without_hardware_support() -> None:
    with pytest.raises(ValueError, match="TMEM row-reduce tuning"):
        FlashTargetPolicy(tuning=FlashTuningPolicy(tmem_row_reduce_min_kv=256))
    with pytest.raises(ValueError, match="packed exp2 tuning"):
        FlashTargetPolicy(
            tuning=FlashTuningPolicy(
                dense_policies=(
                    FlashDenseTuningPolicy(
                        256,
                        "deg1_16x8",
                        "16/8",
                        0,
                        0,
                        "single",
                        packed_exp2_mode=FlashPackedExp2Mode.ALL_XU,
                    ),
                )
            )
        )
    with pytest.raises(ValueError, match="causal resident softmax"):
        FlashTargetPolicy(
            tuning=FlashTuningPolicy(
                causal_policies=(
                    FlashCausalTuningPolicy(
                        512,
                        8,
                        softmax_lowering=FlashSoftmaxLowering.STATEFUL,
                    ),
                )
            )
        )
    with pytest.raises(ValueError, match="enabled TMEM row-reduce range"):
        FlashTargetPolicy(
            hardware=FlashHardwareCapabilities(supports_tmem_row_reduce=True),
            tuning=FlashTuningPolicy(
                tmem_row_reduce_min_kv=1024,
                causal_policies=(
                    FlashCausalTuningPolicy(
                        512,
                        8,
                        softmax_lowering=FlashSoftmaxLowering.STATEFUL,
                    ),
                ),
            ),
        )
    with pytest.raises(ValueError, match="FP16 head-dim-64"):
        FlashTargetPolicy(
            hardware=FlashHardwareCapabilities(supports_tmem_row_reduce=True),
            tuning=FlashTuningPolicy(
                workload=FlashTuningWorkload(
                    head_dim=128,
                    dtype=FlashTuningDType.BFLOAT16,
                ),
                tmem_row_reduce_min_kv=256,
                causal_policies=(
                    FlashCausalTuningPolicy(
                        512,
                        8,
                        softmax_lowering=FlashSoftmaxLowering.STATEFUL,
                    ),
                ),
            ),
        )
    FlashTargetPolicy(
        hardware=FlashHardwareCapabilities(supports_tmem_row_reduce=True),
        tuning=FlashTuningPolicy(
            workload=FlashTuningWorkload(
                head_dim=128,
                dtype=FlashTuningDType.BFLOAT16,
            ),
            tmem_row_reduce_min_kv=256,
        ),
    )
    with pytest.raises(ValueError, match="FP16 head-dim-64"):
        FlashTargetPolicy(
            tuning=FlashTuningPolicy(
                workload=FlashTuningWorkload(
                    head_dim=128,
                    dtype=FlashTuningDType.BFLOAT16,
                ),
                causal_policies=(FlashCausalTuningPolicy(512, 8),),
            )
        )
    with pytest.raises(ValueError, match="FP16 head-dim-64"):
        FlashTargetPolicy(
            tuning=FlashTuningPolicy(
                workload=FlashTuningWorkload(
                    head_dim=128,
                    dtype=FlashTuningDType.BFLOAT16,
                ),
                dense_policies=(
                    FlashDenseTuningPolicy(
                        512,
                        "1x1",
                        "8/2",
                        0,
                        0,
                        "ring2",
                        pipeline_family="fa4_tma_4d",
                        rescale_threshold=0,
                    ),
                ),
            )
        )

    FlashTargetPolicy(
        hardware=FlashHardwareCapabilities(
            supports_tmem_row_reduce=True,
            supports_packed_f16x2_exp2=True,
        ),
        tuning=FlashTuningPolicy(
            tmem_row_reduce_min_kv=256,
            dense_policies=(
                FlashDenseTuningPolicy(
                    256,
                    "deg1_16x8",
                    "16/8",
                    0,
                    0,
                    "single",
                    packed_exp2_mode=FlashPackedExp2Mode.ALL_XU,
                ),
            ),
            causal_policies=(
                FlashCausalTuningPolicy(
                    512,
                    8,
                    softmax_lowering=FlashSoftmaxLowering.STATEFUL,
                ),
            ),
        ),
    )


def test_flash_policy_choice_tables_match_codegen_schema() -> None:
    assert (
        frozenset(cute_flash.FLASH_PIPELINE_FAMILY_FLAGS)
        == flash_tuning._FLASH_POLICY_PIPELINE_FAMILIES
    )
    assert (
        frozenset(
            (
                *cute_flash._FLASH_EXP2_PACKET_PARAMS,
                *cute_flash._FLASH_MANUAL_EXP2_PACKET_PARAMS,
            )
        )
        == flash_tuning._FLASH_POLICY_EXP2_PACKETS
    )
    assert (
        frozenset(cute_flash._FLASH_EXP2_PACKET_PARAMS)
        == flash_tuning._FLASH_POLICY_BASE_EXP2_PACKETS
    )
    assert (
        cute_flash._FLASH_DEG1_EXP2_PACKETS
        == flash_tuning._FLASH_POLICY_DEGREE1_EXP2_PACKETS
    )
    assert (
        frozenset(
            family
            for family, flags in cute_flash.FLASH_PIPELINE_FAMILY_FLAGS.items()
            if flags.tensor_4d_tma
        )
        == flash_tuning._FLASH_POLICY_FP16_HD64_PIPELINE_FAMILIES
    )
