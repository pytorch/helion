from __future__ import annotations

from unittest.mock import patch

import pytest

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
    assert (
        flash_target_policy_cache_identity(
            (10, 3), head_dim=128, torch_dtype="float16", num_kv=256
        )
        is None
    )
    assert (
        flash_target_policy_cache_identity(
            (10, 3), head_dim=64, torch_dtype="bfloat16", num_kv=256
        )
        is None
    )
    bfloat_tuning = FlashTuningPolicy(
        workload=FlashTuningWorkload(
            head_dim=128,
            dtype=FlashTuningDType.BFLOAT16,
        ),
        tmem_row_reduce_min_kv=512,
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
    with patch(
        "helion._compiler.cute.flash_policy.get_flash_target_policy",
        return_value=multi_workload_policy,
    ):
        bfloat_identity = flash_target_policy_cache_identity(
            (10, 3), head_dim=128, torch_dtype="bfloat16", num_kv=512
        )
    changed_bfloat_policy = FlashTargetPolicy(
        hardware=capabilities,
        tuning=policy,
        additional_tunings=(
            FlashTuningPolicy(
                workload=bfloat_tuning.workload,
                tmem_row_reduce_min_kv=1024,
            ),
        ),
    )
    with patch(
        "helion._compiler.cute.flash_policy.get_flash_target_policy",
        return_value=changed_bfloat_policy,
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
    with pytest.raises(ValueError, match="probability log2 shift"):
        FlashDenseTuningPolicy(
            256, "1x1", "8/2", 0, 0, "single", probability_log2_shift=-1
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
    with pytest.raises(ValueError, match="unsupported for dense"):
        FlashDenseTuningPolicy(
            256,
            "1x1",
            "8/2",
            0,
            0,
            "single",
            softmax_lowering=FlashSoftmaxLowering.STATEFUL,
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
                        "1x1",
                        "8/2",
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
        hardware=FlashHardwareCapabilities(
            supports_tmem_row_reduce=True,
            supports_packed_f16x2_exp2=True,
        ),
        tuning=FlashTuningPolicy(
            tmem_row_reduce_min_kv=256,
            dense_policies=(
                FlashDenseTuningPolicy(
                    256,
                    "1x1",
                    "8/2",
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
