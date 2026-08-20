from __future__ import annotations

import dataclasses
import enum

from .flash_arch import FlashHardwareCapabilities
from .flash_tuning import FlashCausalTuningPolicy
from .flash_tuning import FlashDenseTuningPolicy
from .flash_tuning import FlashPackedExp2Mode
from .flash_tuning import FlashSoftmaxLowering
from .flash_tuning import FlashTuningPolicy


@dataclasses.dataclass(frozen=True)
class FlashTargetPolicy:
    """Hardware capabilities and exact tuning choices for one target."""

    hardware: FlashHardwareCapabilities = dataclasses.field(
        default_factory=FlashHardwareCapabilities
    )
    tuning: FlashTuningPolicy = dataclasses.field(default_factory=FlashTuningPolicy)

    def __post_init__(self) -> None:
        if (
            self.tuning.tmem_row_reduce_min_kv is not None
            and not self.hardware.supports_tmem_row_reduce
        ):
            raise ValueError("TMEM row-reduce tuning requires hardware support")
        causal_resident_policies = tuple(
            policy
            for policy in self.tuning.causal_policies
            if policy.softmax_lowering is not FlashSoftmaxLowering.STANDARD
        )
        if causal_resident_policies and not self.hardware.supports_tmem_row_reduce:
            raise ValueError("causal resident softmax requires TMEM row-reduce support")
        row_reduce_min_kv = self.tuning.tmem_row_reduce_min_kv
        if causal_resident_policies and (
            row_reduce_min_kv is None
            or any(
                policy.num_kv < row_reduce_min_kv for policy in causal_resident_policies
            )
        ):
            raise ValueError(
                "causal resident softmax requires an enabled TMEM row-reduce range"
            )
        if (
            any(
                policy.packed_exp2_mode is not FlashPackedExp2Mode.DISABLED
                for policy in self.tuning.dense_policies
            )
            and not self.hardware.supports_packed_f16x2_exp2
        ):
            raise ValueError("packed exp2 tuning requires hardware support")


_GENERIC_FLASH_TARGET_POLICY = FlashTargetPolicy()
_FLASH_TARGET_POLICIES = {
    (10, 3): FlashTargetPolicy(
        hardware=FlashHardwareCapabilities(
            supports_tmem_row_reduce=True,
            supports_packed_f16x2_exp2=True,
        ),
        tuning=FlashTuningPolicy(
            tmem_row_reduce_min_kv=256,
            dense_policies=(
                FlashDenseTuningPolicy(
                    num_kv=256,
                    exp2_packet="deg1_8x2_corr10",
                    e2e_schedule="8/2",
                    e2e_offset=5,
                    e2e_offset0=1,
                    stat_transport="single",
                    probability_log2_shift=7,
                    softmax_lowering=FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
                ),
                FlashDenseTuningPolicy(
                    num_kv=512,
                    exp2_packet="deg1_8x2_corr10",
                    e2e_schedule="8/2",
                    e2e_offset=2,
                    e2e_offset0=1,
                    stat_transport="single",
                    probability_log2_shift=7,
                    softmax_lowering=FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
                    corr_regs=72,
                    other_regs=40,
                ),
                FlashDenseTuningPolicy(
                    num_kv=1024,
                    exp2_packet="deg1_8x2_corr10",
                    e2e_schedule="8/2",
                    e2e_offset=2,
                    e2e_offset0=1,
                    stat_transport="single",
                    probability_log2_shift=7,
                    softmax_lowering=FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
                ),
                FlashDenseTuningPolicy(
                    num_kv=2048,
                    exp2_packet="deg1_16x8",
                    e2e_schedule="16/8",
                    e2e_offset=0,
                    e2e_offset0=10,
                    stat_transport="single_final",
                    packed_exp2_mode=FlashPackedExp2Mode.ALL_XU,
                    probability_log2_shift=7,
                ),
            ),
            causal_policies=(
                FlashCausalTuningPolicy(
                    num_kv=512,
                    kv_stage=8,
                    softmax_lowering=FlashSoftmaxLowering.STATEFUL,
                    softmax_regs=200,
                    first_load_order=2,
                ),
                FlashCausalTuningPolicy(
                    num_kv=1024,
                    kv_stage=3,
                    softmax_lowering=FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
                ),
                FlashCausalTuningPolicy(
                    num_kv=2048,
                    kv_stage=3,
                    softmax_lowering=FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
                ),
                FlashCausalTuningPolicy(
                    num_kv=4096,
                    kv_stage=6,
                    softmax_lowering=FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
                ),
            ),
        ),
    ),
}


def get_flash_target_policy(
    capability: tuple[int, int] | None,
) -> FlashTargetPolicy:
    """Return flash-attention hardware and tuning policy for a target."""
    if capability is None:
        return _GENERIC_FLASH_TARGET_POLICY
    return _FLASH_TARGET_POLICIES.get(capability, _GENERIC_FLASH_TARGET_POLICY)


def _cache_identity_value(value: object) -> object:
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return tuple(
            (field.name, _cache_identity_value(getattr(value, field.name)))
            for field in dataclasses.fields(value)
        )
    if isinstance(value, tuple):
        return tuple(_cache_identity_value(item) for item in value)
    return value


def flash_target_policy_cache_identity(
    capability: tuple[int, int] | None,
) -> object | None:
    """Return a stable cache salt for non-generic target policy choices."""
    policy = get_flash_target_policy(capability)
    if policy is _GENERIC_FLASH_TARGET_POLICY:
        return None
    return _cache_identity_value(policy)
