from __future__ import annotations

import dataclasses
import enum
import math

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
        "causal_hd128_resident3_013_prefetch2_deg2_early_acquire",
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
_FLASH_POLICY_FP16_HD64_PIPELINE_FAMILIES = frozenset(
    {
        "fa4_tma_4d",
        "fa4_local_tma_4d",
        "fa4_cga2_local_tma_4d",
        "fa4_2cta_tma_4d",
        "fa4_clc_tma_4d",
        "fa4_clc_local_tma_4d",
    }
)
_FLASH_POLICY_ROLE_MAPS = frozenset({"helion", "fa4"})
_FLASH_POLICY_BASE_EXP2_PACKETS = frozenset({"1x1", "4x1", "4x2", "8x1", "8x2"})
_FLASH_POLICY_DEGREE1_EXP2_PACKETS = frozenset({"deg1_16x8", "deg1_8x2_corr10"})
_FLASH_FLOAT16_MAX_LOG2 = math.log2(65504.0)


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
    rescale_threshold: float = 8.0
    packed_exp2_mode: FlashPackedExp2Mode = FlashPackedExp2Mode.DISABLED
    probability_log2_shift: int = 0
    softmax_lowering: FlashSoftmaxLowering = FlashSoftmaxLowering.STANDARD
    corr_regs: int | None = None
    other_regs: int | None = None
    epi_tma: bool | None = None
    kv_order: str | None = None
    precompute_qk_desc: bool | None = None
    rescale_chunk_cols: int | None = None
    first_load_order: int | None = None
    corr_tile_size: int | None = None
    role_map: str | None = None
    softmax_regs: int | None = None
    split_p_arrive: bool | None = None
    softmax_disc: bool | None = None
    disc_pipe_depth: int | None = None
    sp_row_sum: str | None = None

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
        if not math.isfinite(self.rescale_threshold):
            raise ValueError("dense rescale threshold must be finite")
        if self.rescale_threshold < 0:
            raise ValueError("dense rescale threshold must be nonnegative")
        if (
            type(self.probability_log2_shift) is not int
            or self.probability_log2_shift < 0
        ):
            raise ValueError("probability log2 shift must be a nonnegative integer")
        if type(self.softmax_lowering) is not FlashSoftmaxLowering or (
            self.softmax_lowering
            not in (
                FlashSoftmaxLowering.STANDARD,
                FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
            )
        ):
            raise ValueError(
                f"unsupported dense softmax lowering: {self.softmax_lowering!r}"
            )
        if type(self.packed_exp2_mode) is not FlashPackedExp2Mode or (
            self.packed_exp2_mode
            not in (FlashPackedExp2Mode.DISABLED, FlashPackedExp2Mode.ALL_XU)
        ):
            raise ValueError(
                f"unsupported dense packed exp2 mode: {self.packed_exp2_mode!r}"
            )
        if (
            self.softmax_lowering is FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH
            and self.stat_transport != "single"
        ):
            raise ValueError(
                "dense resident value-graph lowering requires single statistics transport"
            )
        uses_resident_softmax = (
            self.softmax_lowering is FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH
        )
        uses_packed_exp2 = self.packed_exp2_mode is FlashPackedExp2Mode.ALL_XU
        if uses_resident_softmax and uses_packed_exp2:
            raise ValueError(
                "dense resident softmax and packed exp2 lowerings are mutually exclusive"
            )
        if uses_resident_softmax or uses_packed_exp2:
            if self.rescale_threshold <= 0:
                raise ValueError(
                    "specialized dense lowerings require a positive rescale threshold"
                )
            if self.pipeline_family != "fa4_2cta" or self.persistent:
                raise ValueError(
                    "specialized dense lowerings require the nonpersistent fa4_2cta pipeline"
                )
            if self.e2e_schedule == "xu":
                raise ValueError(
                    "specialized dense lowerings require a split end-to-end schedule"
                )
            if (
                self.probability_log2_shift + self.rescale_threshold
                >= _FLASH_FLOAT16_MAX_LOG2
            ):
                raise ValueError(
                    "dense probability shift and rescale threshold exceed fp16 range"
                )
        elif self.probability_log2_shift:
            raise ValueError(
                "dense probability shift requires a resident or packed exp2 lowering"
            )
        if (
            uses_packed_exp2
            and self.exp2_packet not in _FLASH_POLICY_DEGREE1_EXP2_PACKETS
        ):
            raise ValueError("packed exp2 lowering requires a degree-1 exp2 packet")
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
        if self.kv_order is not None:
            _validate_policy_choice(
                "dense KV order",
                self.kv_order,
                frozenset({"ascending", "descending"}),
            )
        if self.rescale_chunk_cols is not None and self.rescale_chunk_cols not in (
            8,
            16,
            32,
            64,
        ):
            raise ValueError("dense rescale chunk columns must be 8, 16, 32, or 64")
        if self.first_load_order is not None and self.first_load_order not in range(5):
            raise ValueError("dense first-load order must be between 0 and 4")
        if self.corr_tile_size is not None and self.corr_tile_size not in (8, 16, 32):
            raise ValueError("dense correction tile size must be 8, 16, or 32")
        if self.role_map is not None:
            _validate_policy_choice(
                "dense role map", self.role_map, _FLASH_POLICY_ROLE_MAPS
            )
        if self.softmax_regs is not None and self.softmax_regs not in (
            176,
            184,
            192,
            200,
        ):
            raise ValueError(
                "dense softmax register count must be 176, 184, 192, or 200"
            )
        if self.disc_pipe_depth is not None and self.disc_pipe_depth not in (
            1,
            2,
            3,
            4,
        ):
            raise ValueError("dense discrete softmax pipeline depth must be 1 to 4")
        if self.sp_row_sum not in (None, "fragment", "whole"):
            raise ValueError("dense SP row-sum mode must be fragment or whole")

    @property
    def requires_fp16_hd64(self) -> bool:
        """Whether this policy selects an implementation limited to FP16/hd64."""
        return (
            self.pipeline_family in _FLASH_POLICY_FP16_HD64_PIPELINE_FAMILIES
            or self.exp2_packet not in _FLASH_POLICY_BASE_EXP2_PACKETS
            or self.softmax_lowering is not FlashSoftmaxLowering.STANDARD
            or self.packed_exp2_mode is not FlashPackedExp2Mode.DISABLED
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
        if type(self.seed_template) is not FlashCausalSeedTemplate or (
            self.seed_template is not FlashCausalSeedTemplate.DEGREE2_V1
        ):
            raise ValueError(
                f"unsupported causal seed template: {self.seed_template!r}"
            )
        if type(self.softmax_lowering) is not FlashSoftmaxLowering or (
            self.softmax_lowering
            not in (
                FlashSoftmaxLowering.STANDARD,
                FlashSoftmaxLowering.RESIDENT_VALUE_GRAPH,
                FlashSoftmaxLowering.STATEFUL,
            )
        ):
            raise ValueError(
                f"unsupported causal softmax lowering: {self.softmax_lowering!r}"
            )
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

    @property
    def requires_fp16_hd64(self) -> bool:
        """Whether this policy's seed template is limited to FP16/hd64."""
        # DEGREE2_V1 is currently the only implemented causal seed template and
        # its emitter is specialized to the FP16/head-dim-64 workload.
        return self.seed_template is FlashCausalSeedTemplate.DEGREE2_V1


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
