from __future__ import annotations

import dataclasses
import math

DENSE_SCORE_KIND = "dense"
CAUSAL_MASK_KIND = "causal_mask"
TENSOR_BIAS_KIND = "tensor_bias"
RELATIVE_BIAS_KIND = "relative_bias"
ALIBI_BIAS_KIND = "alibi_bias"
SLIDING_WINDOW_MASK_KIND = "sliding_window_mask"
PREFIX_LM_MASK_KIND = "prefix_lm_mask"
DOCUMENT_MASK_KIND = "document_mask"
SOFTCAP_KIND = "softcap"
LOWERED_SCORE_KINDS = frozenset(
    {
        CAUSAL_MASK_KIND,
        TENSOR_BIAS_KIND,
        RELATIVE_BIAS_KIND,
        ALIBI_BIAS_KIND,
        SLIDING_WINDOW_MASK_KIND,
        PREFIX_LM_MASK_KIND,
        DOCUMENT_MASK_KIND,
        SOFTCAP_KIND,
    }
)
MASK_SCORE_KINDS = frozenset(
    {
        CAUSAL_MASK_KIND,
        SLIDING_WINDOW_MASK_KIND,
        PREFIX_LM_MASK_KIND,
        DOCUMENT_MASK_KIND,
    }
)
SPARSE_TILE_MASK_KINDS = frozenset(
    {
        SLIDING_WINDOW_MASK_KIND,
        PREFIX_LM_MASK_KIND,
        DOCUMENT_MASK_KIND,
    }
)


@dataclasses.dataclass(frozen=True)
class AttentionScoreModifier:
    """One tile-local score transform applied after QK and before softmax."""

    kind: str
    tensor_name: str | None = None
    scale_log2: float = 1.0
    value_log2: float | None = None
    window_size: int | None = None
    prefix_length: int | None = None
    index_mode: str | None = None
    index_divisor: int | None = None


@dataclasses.dataclass(frozen=True)
class AttentionScorePlan:
    """Composable score-transform plan for the CuTe flash attention path."""

    head_dim: int
    qk_scale_log2: float
    lse_scale: float = 1.0
    modifiers: tuple[AttentionScoreModifier, ...] = ()

    @property
    def is_causal(self) -> bool:
        return any(modifier.kind == CAUSAL_MASK_KIND for modifier in self.modifiers)

    @property
    def modifier_kinds(self) -> tuple[str, ...]:
        if not self.modifiers:
            return (DENSE_SCORE_KIND,)
        return tuple(modifier.kind for modifier in self.modifiers)

    @property
    def tensor_biases(self) -> tuple[AttentionScoreModifier, ...]:
        return tuple(
            modifier for modifier in self.modifiers if modifier.kind == TENSOR_BIAS_KIND
        )

    @property
    def has_tensor_bias(self) -> bool:
        return bool(self.tensor_biases)

    @property
    def alibi_biases(self) -> tuple[AttentionScoreModifier, ...]:
        return tuple(
            modifier for modifier in self.modifiers if modifier.kind == ALIBI_BIAS_KIND
        )

    @property
    def document_masks(self) -> tuple[AttentionScoreModifier, ...]:
        return tuple(
            modifier
            for modifier in self.modifiers
            if modifier.kind == DOCUMENT_MASK_KIND
        )

    @property
    def sliding_window_masks(self) -> tuple[AttentionScoreModifier, ...]:
        return tuple(
            modifier
            for modifier in self.modifiers
            if modifier.kind == SLIDING_WINDOW_MASK_KIND
        )

    @property
    def prefix_lm_masks(self) -> tuple[AttentionScoreModifier, ...]:
        return tuple(
            modifier
            for modifier in self.modifiers
            if modifier.kind == PREFIX_LM_MASK_KIND
        )

    @property
    def has_kv_tile_pruning(self) -> bool:
        return any(
            modifier.kind in SPARSE_TILE_MASK_KINDS for modifier in self.modifiers
        )

    @property
    def requires_ws_overlap(self) -> bool:
        return self.has_kv_tile_pruning

    def has_lowering(self) -> bool:
        seen_mask = False
        for modifier in self.modifiers:
            if modifier.kind in MASK_SCORE_KINDS:
                seen_mask = True
            elif modifier.kind == SOFTCAP_KIND and seen_mask:
                return False
        return (
            all(modifier.kind in LOWERED_SCORE_KINDS for modifier in self.modifiers)
            and len(self.tensor_biases) <= 1
            and len(self.alibi_biases) <= 1
            and len(self.sliding_window_masks) <= 1
            and len(self.prefix_lm_masks) <= 1
            and len(self.document_masks) <= 1
        )


def dense_score_plan(head_dim: int) -> AttentionScorePlan:
    return AttentionScorePlan(
        head_dim=head_dim,
        qk_scale_log2=math.log2(math.e) / math.sqrt(head_dim),
    )


def causal_score_plan(head_dim: int) -> AttentionScorePlan:
    return AttentionScorePlan(
        head_dim=head_dim,
        qk_scale_log2=math.log2(math.e) / math.sqrt(head_dim),
        modifiers=(AttentionScoreModifier(CAUSAL_MASK_KIND),),
    )
