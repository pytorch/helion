from __future__ import annotations

import dataclasses
import enum

from .flash_arch import FlashHardwareCapabilities
from .flash_tuning import FlashCausalTuningPolicy
from .flash_tuning import FlashCausalSeedTemplate
from .flash_tuning import FlashDenseTuningPolicy
from .flash_tuning import FlashPackedExp2Mode
from .flash_tuning import FlashSoftmaxLowering
from .flash_tuning import FlashTuningDType
from .flash_tuning import FlashTuningPolicy
from .flash_tuning import FlashTuningWorkload


@dataclasses.dataclass(frozen=True)
class FlashTargetPolicy:
    """Hardware capabilities and exact tuning choices for one target."""

    hardware: FlashHardwareCapabilities = dataclasses.field(
        default_factory=FlashHardwareCapabilities
    )
    tuning: FlashTuningPolicy = dataclasses.field(default_factory=FlashTuningPolicy)
    additional_tunings: tuple[FlashTuningPolicy, ...] = ()

    def __post_init__(self) -> None:
        workloads = [tuning.workload for tuning in self.tunings]
        if len(set(workloads)) != len(workloads):
            raise ValueError("flash target tuning workloads must be unique")
        for tuning in self.tunings:
            requires_fp16_hd64 = any(
                policy.requires_fp16_hd64 for policy in tuning.dense_policies
            ) or any(policy.requires_fp16_hd64 for policy in tuning.causal_policies)
            if requires_fp16_hd64 and (
                tuning.workload.head_dim != 64
                or tuning.workload.dtype is not FlashTuningDType.FLOAT16
            ):
                raise ValueError(
                    "flash target tuning currently requires the FP16 head-dim-64 workload"
                )
            if (
                tuning.tmem_row_reduce_min_kv is not None
                and not self.hardware.supports_tmem_row_reduce
            ):
                raise ValueError("TMEM row-reduce tuning requires hardware support")
            causal_resident_policies = tuple(
                policy
                for policy in tuning.causal_policies
                if policy.softmax_lowering is not FlashSoftmaxLowering.STANDARD
            )
            if causal_resident_policies and not self.hardware.supports_tmem_row_reduce:
                raise ValueError(
                    "causal resident softmax requires TMEM row-reduce support"
                )
            row_reduce_min_kv = tuning.tmem_row_reduce_min_kv
            if causal_resident_policies and (
                row_reduce_min_kv is None
                or any(
                    policy.num_kv < row_reduce_min_kv
                    for policy in causal_resident_policies
                )
            ):
                raise ValueError(
                    "causal resident softmax requires an enabled TMEM row-reduce range"
                )
            if (
                any(
                    policy.packed_exp2_mode is not FlashPackedExp2Mode.DISABLED
                    for policy in tuning.dense_policies
                )
                and not self.hardware.supports_packed_f16x2_exp2
            ):
                raise ValueError("packed exp2 tuning requires hardware support")

    @property
    def tunings(self) -> tuple[FlashTuningPolicy, ...]:
        """Return every workload-specific tuning table for this target."""
        return (self.tuning, *self.additional_tunings)

    def tuning_for_torch(
        self, head_dim: int, torch_dtype: str
    ) -> FlashTuningPolicy | None:
        """Resolve a tuning table from host-side workload types."""
        return next(
            (
                tuning
                for tuning in self.tunings
                if tuning.workload.head_dim == head_dim
                and tuning.workload.dtype.value == torch_dtype
            ),
            None,
        )

    def tuning_for_cute(
        self, head_dim: int, cute_dtype: str
    ) -> FlashTuningPolicy | None:
        """Resolve a tuning table from generated CuTe workload types."""
        return next(
            (
                tuning
                for tuning in self.tunings
                if tuning.workload.head_dim == head_dim
                and tuning.workload.dtype.cute_name == cute_dtype
            ),
            None,
        )


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
                    corr_regs=64,
                    other_regs=40,
                    epi_tma=True,
                    kv_order="descending",
                    precompute_qk_desc=True,
                    rescale_chunk_cols=8,
                    first_load_order=0,
                    corr_tile_size=8,
                    role_map="helion",
                    softmax_regs=200,
                    split_p_arrive=True,
                    softmax_disc=False,
                    disc_pipe_depth=1,
                    sp_row_sum="whole",
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
                    epi_tma=True,
                    kv_order="descending",
                    precompute_qk_desc=True,
                    rescale_chunk_cols=8,
                    first_load_order=4,
                    corr_tile_size=16,
                    role_map="fa4",
                    softmax_regs=200,
                    split_p_arrive=True,
                    softmax_disc=False,
                    disc_pipe_depth=1,
                    sp_row_sum="whole",
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
                    corr_regs=72,
                    other_regs=40,
                    epi_tma=True,
                    kv_order="descending",
                    precompute_qk_desc=True,
                    rescale_chunk_cols=8,
                    first_load_order=0,
                    corr_tile_size=8,
                    role_map="helion",
                    softmax_regs=200,
                    split_p_arrive=True,
                    softmax_disc=False,
                    disc_pipe_depth=1,
                    sp_row_sum="whole",
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
                    corr_regs=80,
                    other_regs=32,
                    epi_tma=True,
                    kv_order="descending",
                    precompute_qk_desc=True,
                    rescale_chunk_cols=8,
                    first_load_order=4,
                    corr_tile_size=8,
                    role_map="helion",
                    softmax_regs=192,
                    split_p_arrive=True,
                    softmax_disc=False,
                    disc_pipe_depth=1,
                    sp_row_sum="whole",
                ),
            ),
            causal_policies=(
                FlashCausalTuningPolicy(
                    num_kv=512,
                    kv_stage=8,
                    e2e_offset=15,
                    e2e_offset0=3,
                    role_map="fa4",
                    epi_tma=True,
                    softmax_lowering=FlashSoftmaxLowering.STATEFUL,
                    softmax_regs=200,
                    first_load_order=2,
                ),
                FlashCausalTuningPolicy(
                    num_kv=1024,
                    kv_stage=3,
                    e2e_offset=1,
                    e2e_offset0=14,
                    role_map="fa4",
                    softmax_lowering=FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
                    softmax_regs=184,
                    first_load_order=0,
                ),
                FlashCausalTuningPolicy(
                    num_kv=2048,
                    kv_stage=3,
                    e2e_offset=14,
                    e2e_offset0=12,
                    role_map="fa4",
                    softmax_lowering=FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
                    softmax_regs=184,
                    first_load_order=0,
                ),
                FlashCausalTuningPolicy(
                    num_kv=4096,
                    kv_stage=6,
                    e2e_offset=14,
                    e2e_offset0=12,
                    softmax_lowering=FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
                    softmax_regs=184,
                    first_load_order=0,
                ),
            ),
        ),
        additional_tunings=(
            # SM103 registered resident-softmax seeds only for hd64/FP16
            # (#3416-#3419), so head_dim=128 falls back to the STANDARD softmax
            # lowering. Seed an hd128 causal entry with the head-dim-agnostic
            # resident template so the same RESIDENT_VALUE_GRAPH lowering that
            # hd64 uses is reachable at hd128. Values come from the best
            # measured 1-CTA hd128 config (the resident path requires
            # not use_2cta_instrs).
            FlashTuningPolicy(
                workload=FlashTuningWorkload(
                    head_dim=128, dtype=FlashTuningDType.FLOAT16
                ),
                tmem_row_reduce_min_kv=256,
                causal_policies=(
                    FlashCausalTuningPolicy(
                        num_kv=512,
                        kv_stage=3,
                        seed_template=FlashCausalSeedTemplate.RESIDENT_V1,
                        e2e_offset=0,
                        e2e_offset0=0,
                        role_map="fa4",
                        epi_tma=False,
                        softmax_lowering=(
                            FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH
                        ),
                        softmax_regs=200,
                        first_load_order=2,
                    ),                    FlashCausalTuningPolicy(
                        num_kv=1024,
                        kv_stage=3,
                        seed_template=FlashCausalSeedTemplate.RESIDENT_V1,
                        e2e_offset=0,
                        e2e_offset0=0,
                        role_map="fa4",
                        epi_tma=False,
                        softmax_lowering=(
                            FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH
                        ),
                        softmax_regs=200,
                        first_load_order=2,
                    ),                    FlashCausalTuningPolicy(
                        num_kv=2048,
                        kv_stage=3,
                        seed_template=FlashCausalSeedTemplate.RESIDENT_V1,
                        e2e_offset=0,
                        e2e_offset0=0,
                        role_map="fa4",
                        epi_tma=False,
                        softmax_lowering=(
                            FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH
                        ),
                        softmax_regs=200,
                        first_load_order=2,
                    ),                    FlashCausalTuningPolicy(
                        num_kv=4096,
                        kv_stage=3,
                        seed_template=FlashCausalSeedTemplate.RESIDENT_V1,
                        e2e_offset=0,
                        e2e_offset0=0,
                        role_map="fa4",
                        epi_tma=True,
                        softmax_lowering=(
                            FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH
                        ),
                        softmax_regs=200,
                        first_load_order=2,
                    ),
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


def registered_flash_target_policies() -> tuple[
    tuple[tuple[int, int], FlashTargetPolicy], ...
]:
    """Return registered target policies in stable capability order."""
    return tuple(sorted(_FLASH_TARGET_POLICIES.items()))


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
    *,
    head_dim: int | None = None,
    torch_dtype: str | None = None,
    num_kv: int | None = None,
    is_causal: bool = False,
) -> object | None:
    """Return the stable cache salt for one target/workload policy."""
    policy = get_flash_target_policy(capability)
    if policy is _GENERIC_FLASH_TARGET_POLICY:
        return None
    if head_dim is None or torch_dtype is None:
        tuning = policy.tuning
    else:
        tuning = policy.tuning_for_torch(head_dim, torch_dtype)
    if tuning is None:
        return None
    workload = tuning.workload
    shape_policy = (
        tuning.causal_policy(num_kv)
        if is_causal and num_kv is not None
        else tuning.dense_policy(num_kv)
        if num_kv is not None
        else None
    )
    return _cache_identity_value(
        (
            policy.hardware,
            workload,
            tuning.tmem_row_reduce_min_kv,
            shape_policy,
        )
    )
