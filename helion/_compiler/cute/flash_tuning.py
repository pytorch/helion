from __future__ import annotations

import dataclasses
import enum

_FLASH_POLICY_EXP2_PACKETS = frozenset(
    {
        "1x1",
        "4x1",
        "4x2",
        "8x1",
        "8x2",
        "deg2_16x6",
        "hybrid_deg1_16x8",
        "deg1_16x8",
        "deg1_8x2_corr10",
    }
)
_FLASH_POLICY_E2E_SCHEDULES = frozenset({"xu", "8/2", "16/2", "16/4", "16/6", "16/8"})
_FLASH_POLICY_STAT_TRANSPORTS = frozenset({"ring2", "single", "single_final"})
_FLASH_POLICY_PIPELINE_FAMILIES = frozenset(
    {
        "ws_overlap",
        "fa4",
        "fa4_deep_1cta",
        "fa4_2cta_causal",
        "fa4_tma_4d",
        "fa4_local_tma",
        "fa4_local_tma_4d",
        "fa4_cga2_local",
        "fa4_cga2_local_tma_4d",
        "fa4_2cta",
        "fa4_2cta_tma_4d",
        "fa4_clc",
        "fa4_clc_tma_4d",
        "fa4_clc_local_tma",
        "fa4_clc_local_tma_4d",
    }
)
_FLASH_POLICY_ROLE_MAPS = frozenset({"helion", "fa4"})


def _validate_policy_choice(name: str, value: str, choices: frozenset[str]) -> None:
    if value not in choices:
        raise ValueError(f"unsupported flash {name}: {value!r}")


class FlashSoftmaxLowering(str, enum.Enum):
    """Available resident-softmax implementation strategies."""

    STANDARD = "standard"
    RESIDENT_VALUE_GRAPH = "resident_value_graph"
    STATEFUL = "stateful"


class FlashPackedExp2Mode(str, enum.Enum):
    """Dense packed-f16x2 exponential strategy."""

    DISABLED = "disabled"
    ALL_XU = "all_xu"


class FlashCausalSeedTemplate(str, enum.Enum):
    """Versioned causal seed implementation families."""

    DEGREE2_V1 = "degree2_v1"


class FlashTuningDType(str, enum.Enum):
    """Element types supported by target flash tuning policies."""

    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    @property
    def cute_name(self) -> str:
        if self is FlashTuningDType.FLOAT16:
            return "cutlass.Float16"
        if self is FlashTuningDType.BFLOAT16:
            return "cutlass.BFloat16"
        raise AssertionError(f"unsupported flash tuning dtype: {self!r}")


@dataclasses.dataclass(frozen=True)
class FlashTuningWorkload:
    """Workload envelope supported by one target tuning table."""

    head_dim: int = 64
    dtype: FlashTuningDType = FlashTuningDType.FLOAT16

    def __post_init__(self) -> None:
        if self.head_dim <= 0:
            raise ValueError("flash tuning head dimension must be positive")


@dataclasses.dataclass(frozen=True)
class FlashDenseTuningPolicy:
    """Exact dense seed and optional lowering choices for one KV size."""

    num_kv: int
    exp2_packet: str
    e2e_schedule: str
    e2e_offset: int
    e2e_offset0: int
    stat_transport: str
    pipeline_family: str = "fa4_2cta"
    kv_stage: int = 6
    persistent: bool = False
    packed_exp2_mode: FlashPackedExp2Mode = FlashPackedExp2Mode.DISABLED
    probability_log2_shift: int = 0
    softmax_lowering: FlashSoftmaxLowering = FlashSoftmaxLowering.STANDARD
    corr_regs: int | None = None
    other_regs: int | None = None

    def __post_init__(self) -> None:
        if self.num_kv <= 0:
            raise ValueError("dense KV size must be positive")
        _validate_policy_choice(
            "dense exp2 packet", self.exp2_packet, _FLASH_POLICY_EXP2_PACKETS
        )
        _validate_policy_choice(
            "dense end-to-end schedule",
            self.e2e_schedule,
            _FLASH_POLICY_E2E_SCHEDULES,
        )
        if self.e2e_offset < 0 or self.e2e_offset0 < 0:
            raise ValueError("dense end-to-end offsets must be nonnegative")
        _validate_policy_choice(
            "dense statistics transport",
            self.stat_transport,
            _FLASH_POLICY_STAT_TRANSPORTS,
        )
        _validate_policy_choice(
            "dense pipeline family",
            self.pipeline_family,
            _FLASH_POLICY_PIPELINE_FAMILIES,
        )
        if self.kv_stage <= 0:
            raise ValueError("dense KV stage must be positive")
        if self.probability_log2_shift < 0:
            raise ValueError("probability log2 shift must be nonnegative")
        if self.softmax_lowering is FlashSoftmaxLowering.STATEFUL:
            raise ValueError(
                "stateful softmax lowering is unsupported for dense policy"
            )
        if (
            self.softmax_lowering is FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH
            and self.stat_transport != "single"
        ):
            raise ValueError(
                "dense resident value-graph lowering requires single statistics transport"
            )
        if (self.corr_regs is None) != (self.other_regs is None):
            raise ValueError(
                "dense correction and other register counts must be set together"
            )
        if self.corr_regs is not None and (self.corr_regs <= 0 or self.corr_regs % 8):
            raise ValueError(
                "dense correction register count must be a positive multiple of 8"
            )
        if self.other_regs is not None and (
            self.other_regs < 24 or self.other_regs % 8
        ):
            raise ValueError(
                "dense other register count must be at least 24 and a multiple of 8"
            )


@dataclasses.dataclass(frozen=True)
class FlashCausalTuningPolicy:
    """Exact causal seed and resident-softmax choices for one KV size."""

    num_kv: int
    kv_stage: int
    seed_template: FlashCausalSeedTemplate = FlashCausalSeedTemplate.DEGREE2_V1
    e2e_offset: int = 0
    e2e_offset0: int = 0
    role_map: str = "helion"
    epi_tma: bool = False
    softmax_lowering: FlashSoftmaxLowering = FlashSoftmaxLowering.STANDARD
    softmax_regs: int | None = None
    first_load_order: int | None = None
    causal_loop_split: bool = True
    causal_kv_order: str = "descending"

    def __post_init__(self) -> None:
        if self.num_kv <= 0:
            raise ValueError("causal KV size must be positive")
        if self.kv_stage <= 0:
            raise ValueError("causal KV stage must be positive")
        if self.e2e_offset < 0 or self.e2e_offset0 < 0:
            raise ValueError("causal end-to-end offsets must be nonnegative")
        _validate_policy_choice(
            "causal role map", self.role_map, _FLASH_POLICY_ROLE_MAPS
        )
        if self.softmax_regs is not None and (
            self.softmax_regs <= 0 or self.softmax_regs % 8
        ):
            raise ValueError(
                "causal softmax register count must be a positive multiple of 8"
            )
        if self.first_load_order is not None and self.first_load_order not in range(5):
            raise ValueError("causal first-load order must be 0-4")
        if self.causal_kv_order not in ("ascending", "descending"):
            raise ValueError("causal KV order must be ascending or descending")
        if self.softmax_lowering is not FlashSoftmaxLowering.STANDARD and (
            not self.causal_loop_split or self.causal_kv_order != "descending"
        ):
            raise ValueError("causal resident softmax requires a descending split loop")


@dataclasses.dataclass(frozen=True)
class FlashTuningPolicy:
    """Architecture-specific flash-attention seeds and lowering choices."""

    workload: FlashTuningWorkload = dataclasses.field(
        default_factory=FlashTuningWorkload
    )
    tmem_row_reduce_min_kv: int | None = None
    dense_policies: tuple[FlashDenseTuningPolicy, ...] = ()
    causal_policies: tuple[FlashCausalTuningPolicy, ...] = ()

    def __post_init__(self) -> None:
        if self.tmem_row_reduce_min_kv is not None and self.tmem_row_reduce_min_kv <= 0:
            raise ValueError("TMEM row-reduce minimum KV size must be positive")
        dense_num_kv = [policy.num_kv for policy in self.dense_policies]
        if len(set(dense_num_kv)) != len(dense_num_kv):
            raise ValueError("dense KV sizes must be unique")
        causal_num_kv = [policy.num_kv for policy in self.causal_policies]
        if len(set(causal_num_kv)) != len(causal_num_kv):
            raise ValueError("causal KV sizes must be unique")

    def dense_policy(self, num_kv: int) -> FlashDenseTuningPolicy | None:
        """Return the exact dense policy for ``num_kv``, if one exists."""
        return next(
            (policy for policy in self.dense_policies if policy.num_kv == num_kv),
            None,
        )

    def causal_policy(self, num_kv: int) -> FlashCausalTuningPolicy | None:
        """Return the exact causal policy for ``num_kv``, if one exists."""
        return next(
            (policy for policy in self.causal_policies if policy.num_kv == num_kv),
            None,
        )
