from __future__ import annotations

import pytest

from helion._compiler.cute.flash_arch import FlashHardwareCapabilities
from helion._compiler.cute.flash_policy import FlashTargetPolicy
from helion._compiler.cute.flash_policy import get_flash_target_policy
from helion._compiler.cute.flash_tuning import FlashCausalTuningPolicy
from helion._compiler.cute.flash_tuning import FlashDenseTuningPolicy
from helion._compiler.cute.flash_tuning import FlashPackedExp2Mode
from helion._compiler.cute.flash_tuning import FlashSoftmaxLowering
from helion._compiler.cute.flash_tuning import FlashTuningPolicy


def test_sm103_flash_hardware_capabilities() -> None:
    capabilities = get_flash_target_policy((10, 3)).hardware

    assert capabilities.supports_tmem_row_reduce
    assert capabilities.supports_packed_f16x2_exp2


@pytest.mark.parametrize(
    "capability",
    [None, (10, 0), (10, 2), (10, 4), (11, 0)],
)
def test_unknown_and_b200_flash_hardware_capabilities_are_generic(
    capability: tuple[int, int] | None,
) -> None:
    capabilities = get_flash_target_policy(capability).hardware

    assert not capabilities.supports_tmem_row_reduce
    assert not capabilities.supports_packed_f16x2_exp2


def test_sm103_flash_tuning_policy() -> None:
    policy = get_flash_target_policy((10, 3)).tuning

    assert policy.tmem_row_reduce_min_kv == 256

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
            2,
            FlashPackedExp2Mode.DISABLED,
            7,
            FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
            None,
            None,
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
            2,
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
            2,
            FlashPackedExp2Mode.DISABLED,
            7,
            FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
            None,
            None,
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
            2,
            FlashPackedExp2Mode.ALL_XU,
            7,
            FlashSoftmaxLowering.STANDARD,
            None,
            None,
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
            shape.q_tile_count,
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
            2,
            FlashSoftmaxLowering.STATEFUL,
            200,
            2,
        ),
        1024: (
            3,
            2,
            FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
            None,
            None,
        ),
        2048: (
            3,
            2,
            FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
            None,
            None,
        ),
        4096: (
            6,
            2,
            FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
            None,
            None,
        ),
    }
    assert {shape.num_kv for shape in policy.causal_policies} == set(expected_causal)
    for num_kv, expected in expected_causal.items():
        shape = policy.causal_policy(num_kv)
        assert shape is not None
        assert (
            shape.kv_stage,
            shape.q_tile_count,
            shape.softmax_lowering,
            shape.softmax_regs,
            shape.first_load_order,
        ) == expected
    assert policy.causal_policy(256) is None


@pytest.mark.parametrize(
    "capability",
    [None, (10, 0), (10, 2), (10, 4), (11, 0)],
)
def test_unknown_and_b200_flash_tuning_policy_is_generic(
    capability: tuple[int, int] | None,
) -> None:
    policy = get_flash_target_policy(capability).tuning

    assert policy.tmem_row_reduce_min_kv is None
    assert not policy.dense_policies
    assert policy.dense_policy(256) is None
    assert not policy.causal_policies
    assert policy.causal_policy(512) is None


def test_flash_policy_rejects_duplicate_shapes() -> None:
    dense = FlashDenseTuningPolicy(256, "packet", "8/2", 0, 0, "single")
    causal = FlashCausalTuningPolicy(512, 3)

    with pytest.raises(ValueError, match="dense KV sizes must be unique"):
        FlashTuningPolicy(dense_policies=(dense, dense))
    with pytest.raises(ValueError, match="causal KV sizes must be unique"):
        FlashTuningPolicy(causal_policies=(causal, causal))


def test_flash_policy_rejects_invalid_field_combinations() -> None:
    with pytest.raises(ValueError, match="TMEM row-reduce minimum KV size"):
        FlashTuningPolicy(tmem_row_reduce_min_kv=0)
    with pytest.raises(ValueError, match="probability log2 shift"):
        FlashDenseTuningPolicy(
            256, "packet", "8/2", 0, 0, "single", probability_log2_shift=-1
        )
    with pytest.raises(ValueError, match="register counts must be set together"):
        FlashDenseTuningPolicy(
            512,
            "packet",
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
    with pytest.raises(ValueError, match="query tile count must be 2"):
        FlashDenseTuningPolicy(256, "packet", "8/2", 0, 0, "single", q_tile_count=1)
    with pytest.raises(ValueError, match="query tile count must be 2"):
        FlashCausalTuningPolicy(512, 8, q_tile_count=1)
    with pytest.raises(ValueError, match="unsupported for dense"):
        FlashDenseTuningPolicy(
            256,
            "packet",
            "8/2",
            0,
            0,
            "single",
            softmax_lowering=FlashSoftmaxLowering.STATEFUL,
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
                        "packet",
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
                    "packet",
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
