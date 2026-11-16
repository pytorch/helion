"""Target tuning policies for the CuTe flash-attention forward.

A ``FlashTuningPolicy`` is one target's measured per-shape records.  A dense
or causal record names the knob values its body was measured with (projected
onto the autotune keys by ``knob_values``, in the seeds' key order) and the
lowering selections that are not knobs (the resident softmax lowering, the
packed exp2 mode, the probability shift, the causal seed template).  The
name-valued knobs validate against the knob registry's choice tables
(``flash_knobs``); the records mirror none of them.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import TYPE_CHECKING

from .flash_knobs import _FLASH_DEG1_EXP2_PACKETS
from .flash_knobs import _FLASH_DEG2_EXP2_PACKET
from .flash_knobs import FLASH_CAUSAL_KV_ORDER_KEY
from .flash_knobs import FLASH_CAUSAL_LOOP_SPLIT_KEY
from .flash_knobs import FLASH_CORR_REGS_KEY
from .flash_knobs import FLASH_CORR_TILE_SIZE_KEY
from .flash_knobs import FLASH_DISC_PIPE_KEY
from .flash_knobs import FLASH_E2E_OFFSET0_KEY
from .flash_knobs import FLASH_E2E_OFFSET_KEY
from .flash_knobs import FLASH_E2E_SCHEDULE_CHOICES
from .flash_knobs import FLASH_E2E_SCHEDULE_KEY
from .flash_knobs import FLASH_EPI_TMA_KEY
from .flash_knobs import FLASH_EXP2_PACKET_CHOICES
from .flash_knobs import FLASH_EXP2_PACKET_KEY
from .flash_knobs import FLASH_FIRST_LOAD_ORDER_KEY
from .flash_knobs import FLASH_KV_ORDER_KEY
from .flash_knobs import FLASH_KV_STAGE_KEY
from .flash_knobs import FLASH_KV_TILE_N_KEY
from .flash_knobs import FLASH_MASKED_E2E_SCHEDULE_KEY
from .flash_knobs import FLASH_OTHER_REGS_KEY
from .flash_knobs import FLASH_PERSISTENT_KEY
from .flash_knobs import FLASH_PIPELINE_FAMILY_FLAGS
from .flash_knobs import FLASH_PIPELINE_FAMILY_KEY
from .flash_knobs import FLASH_PRECOMPUTE_QK_DESC_KEY
from .flash_knobs import FLASH_RESCALE_CHUNK_COLS_KEY
from .flash_knobs import FLASH_RESCALE_THRESHOLD_KEY
from .flash_knobs import FLASH_ROLE_MAP_CHOICES
from .flash_knobs import FLASH_ROLE_MAP_KEY
from .flash_knobs import FLASH_SOFTMAX_DISC_KEY
from .flash_knobs import FLASH_SOFTMAX_REGS_KEY
from .flash_knobs import FLASH_SP_ROW_SUM_KEY
from .flash_knobs import FLASH_SPLIT_P_ARRIVE_KEY
from .flash_knobs import FLASH_STAT_TRANSPORT_CHOICES
from .flash_knobs import FLASH_STAT_TRANSPORT_KEY
from .flash_knobs import FLASH_WAIT_HINT_KEY
from .flash_knobs import flash_exp2_packet_is_compound

if TYPE_CHECKING:
    from collections.abc import Container

FLASH_FLOAT16_MAX_LOG2 = math.log2(65504.0)


def _validate_policy_choice(name: str, value: str, choices: Container[str]) -> None:
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


# The dense record's knobs in seed order: the measured structure, then the
# optional fields a record may leave unstated (``None``), then the register
# pair.
_DENSE_POLICY_KNOBS: tuple[tuple[str, str], ...] = (
    (FLASH_KV_TILE_N_KEY, "kv_tile_n"),
    (FLASH_PIPELINE_FAMILY_KEY, "pipeline_family"),
    (FLASH_KV_STAGE_KEY, "kv_stage"),
    (FLASH_PERSISTENT_KEY, "persistent"),
    (FLASH_E2E_SCHEDULE_KEY, "e2e_schedule"),
    (FLASH_E2E_OFFSET_KEY, "e2e_offset"),
    (FLASH_E2E_OFFSET0_KEY, "e2e_offset0"),
    (FLASH_EXP2_PACKET_KEY, "exp2_packet"),
    (FLASH_STAT_TRANSPORT_KEY, "stat_transport"),
    (FLASH_RESCALE_THRESHOLD_KEY, "rescale_threshold"),
    (FLASH_EPI_TMA_KEY, "epi_tma"),
    (FLASH_KV_ORDER_KEY, "kv_order"),
    (FLASH_PRECOMPUTE_QK_DESC_KEY, "precompute_qk_desc"),
    (FLASH_RESCALE_CHUNK_COLS_KEY, "rescale_chunk_cols"),
    (FLASH_FIRST_LOAD_ORDER_KEY, "first_load_order"),
    (FLASH_CORR_TILE_SIZE_KEY, "corr_tile_size"),
    (FLASH_ROLE_MAP_KEY, "role_map"),
    (FLASH_SOFTMAX_REGS_KEY, "softmax_regs"),
    (FLASH_SPLIT_P_ARRIVE_KEY, "split_p_arrive"),
    (FLASH_SOFTMAX_DISC_KEY, "softmax_disc"),
    (FLASH_DISC_PIPE_KEY, "disc_pipe_depth"),
    (FLASH_SP_ROW_SUM_KEY, "sp_row_sum"),
    (FLASH_CORR_REGS_KEY, "corr_regs"),
    (FLASH_OTHER_REGS_KEY, "other_regs"),
)


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
    kv_tile_n: int = 128

    def __post_init__(self) -> None:
        if self.num_kv <= 0:
            raise ValueError("dense KV size must be positive")
        if self.kv_tile_n <= 0 or self.kv_tile_n % 32:
            raise ValueError("dense KV tile width must be a positive multiple of 32")
        _validate_policy_choice(
            "dense exp2 packet", self.exp2_packet, FLASH_EXP2_PACKET_CHOICES
        )
        _validate_policy_choice(
            "dense end-to-end schedule",
            self.e2e_schedule,
            FLASH_E2E_SCHEDULE_CHOICES,
        )
        if self.e2e_offset < 0 or self.e2e_offset0 < 0:
            raise ValueError("dense end-to-end offsets must be nonnegative")
        _validate_policy_choice(
            "dense statistics transport",
            self.stat_transport,
            FLASH_STAT_TRANSPORT_CHOICES,
        )
        _validate_policy_choice(
            "dense pipeline family",
            self.pipeline_family,
            FLASH_PIPELINE_FAMILY_FLAGS,
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
                >= FLASH_FLOAT16_MAX_LOG2
            ):
                raise ValueError(
                    "dense probability shift and rescale threshold exceed fp16 range"
                )
        elif self.probability_log2_shift:
            raise ValueError(
                "dense probability shift requires a resident or packed exp2 lowering"
            )
        if uses_packed_exp2 and self.exp2_packet not in _FLASH_DEG1_EXP2_PACKETS:
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
                "dense role map", self.role_map, FLASH_ROLE_MAP_CHOICES
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

    def knob_values(self) -> dict[str, object]:
        """The record's values on the autotune keys (an optional field only
        when stated), in the order the dense seeds are written."""
        return {
            key: value
            for key, field in _DENSE_POLICY_KNOBS
            if (value := getattr(self, field)) is not None
        }

    @property
    def requires_fp16_hd64(self) -> bool:
        """Whether this policy selects an implementation limited to FP16/hd64."""
        return (
            FLASH_PIPELINE_FAMILY_FLAGS[self.pipeline_family].tensor_4d_tma
            or flash_exp2_packet_is_compound(self.exp2_packet)
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
            "causal role map", self.role_map, FLASH_ROLE_MAP_CHOICES
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

    def knob_values(self) -> dict[str, object]:
        """The record's values on the autotune keys: the causal degree-2 seed
        template (``DEGREE2_V1``: the fa4 family at the 16/6 cadence on the
        degree-2 packet, a three-deep PASS2 pipe and 16-column rescale chunks)
        with the record's offsets, output path, role map, ring depth, loop
        form and its stated registers and load order."""
        values: dict[str, object] = {
            FLASH_PIPELINE_FAMILY_KEY: "fa4",
            FLASH_E2E_SCHEDULE_KEY: "16/6",
            FLASH_MASKED_E2E_SCHEDULE_KEY: "16/6",
            FLASH_E2E_OFFSET_KEY: self.e2e_offset,
            FLASH_E2E_OFFSET0_KEY: self.e2e_offset0,
            FLASH_EXP2_PACKET_KEY: _FLASH_DEG2_EXP2_PACKET,
            FLASH_WAIT_HINT_KEY: 0,
            FLASH_DISC_PIPE_KEY: 3,
            FLASH_EPI_TMA_KEY: self.epi_tma,
            FLASH_RESCALE_CHUNK_COLS_KEY: 16,
            FLASH_ROLE_MAP_KEY: self.role_map,
            FLASH_KV_STAGE_KEY: self.kv_stage,
            FLASH_CAUSAL_LOOP_SPLIT_KEY: self.causal_loop_split,
            FLASH_CAUSAL_KV_ORDER_KEY: self.causal_kv_order,
        }
        if self.softmax_regs is not None:
            values[FLASH_SOFTMAX_REGS_KEY] = self.softmax_regs
        if self.first_load_order is not None:
            values[FLASH_FIRST_LOAD_ORDER_KEY] = self.first_load_order
        return values

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
