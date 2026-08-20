from __future__ import annotations

import dataclasses
import enum


class FlashSoftmaxLowering(str, enum.Enum):
    """Available resident-softmax implementation strategies."""

    STANDARD = "standard"
    RESIDENT_VALUE_GRAPH = "resident_value_graph"
    STATEFUL = "stateful"


class FlashPackedExp2Mode(str, enum.Enum):
    """Dense packed-f16x2 exponential strategy."""

    DISABLED = "disabled"
    ALL_XU = "all_xu"


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
    q_tile_count: int = 2
    packed_exp2_mode: FlashPackedExp2Mode = FlashPackedExp2Mode.DISABLED
    probability_log2_shift: int = 0
    softmax_lowering: FlashSoftmaxLowering = FlashSoftmaxLowering.STANDARD
    corr_regs: int | None = None
    other_regs: int | None = None

    def __post_init__(self) -> None:
        if self.num_kv <= 0:
            raise ValueError("dense KV size must be positive")
        if not self.exp2_packet:
            raise ValueError("dense exp2 packet must not be empty")
        if not self.e2e_schedule:
            raise ValueError("dense end-to-end schedule must not be empty")
        if self.e2e_offset < 0 or self.e2e_offset0 < 0:
            raise ValueError("dense end-to-end offsets must be nonnegative")
        if not self.stat_transport:
            raise ValueError("dense statistics transport must not be empty")
        if not self.pipeline_family:
            raise ValueError("dense pipeline family must not be empty")
        if self.kv_stage <= 0:
            raise ValueError("dense KV stage must be positive")
        if self.q_tile_count != 2:
            raise ValueError("dense query tile count must be 2")
        if self.probability_log2_shift < 0:
            raise ValueError("probability log2 shift must be nonnegative")
        if self.softmax_lowering is FlashSoftmaxLowering.STATEFUL:
            raise ValueError(
                "stateful softmax lowering is unsupported for dense policy"
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
    q_tile_count: int = 2
    softmax_lowering: FlashSoftmaxLowering = FlashSoftmaxLowering.STANDARD
    softmax_regs: int | None = None
    first_load_order: int | None = None

    def __post_init__(self) -> None:
        if self.num_kv <= 0:
            raise ValueError("causal KV size must be positive")
        if self.kv_stage <= 0:
            raise ValueError("causal KV stage must be positive")
        if self.q_tile_count != 2:
            raise ValueError("causal query tile count must be 2")
        if self.softmax_regs is not None and (
            self.softmax_regs <= 0 or self.softmax_regs % 8
        ):
            raise ValueError(
                "causal softmax register count must be a positive multiple of 8"
            )
        if self.first_load_order is not None and self.first_load_order not in range(5):
            raise ValueError("causal first-load order must be 0-4")


@dataclasses.dataclass(frozen=True)
class FlashTuningPolicy:
    """Architecture-specific flash-attention seeds and lowering choices."""

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
