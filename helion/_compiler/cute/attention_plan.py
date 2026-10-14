from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import torch.fx

DENSE_SCORE_KIND = "dense"
CAUSAL_MASK_KIND = "causal_mask"
TENSOR_BIAS_KIND = "tensor_bias"
RELATIVE_BIAS_KIND = "relative_bias"
ALIBI_BIAS_KIND = "alibi_bias"
SLIDING_WINDOW_MASK_KIND = "sliding_window_mask"
PREFIX_LM_MASK_KIND = "prefix_lm_mask"
DOCUMENT_MASK_KIND = "document_mask"
# ``kv_idx >= lb[b, q_idx]`` with the causal conjunct: a per-row KV lower bound
# (FA4 varlen's ``cu_seqlens`` data as a ``[B, S]`` table) lowered as a
# contiguous KV tile range plus a boundary-only per-row range mask. The kind
# folds the causal conjunct, so it is the plan's only mask modifier (a second
# one, including a separately applied causal mask, declines the lowering). A
# row whose bound lies past the row sees no key: the lowering returns 0 for it
# (the FlexAttention / FA4 convention) where the kernel source and the Triton
# lowering produce NaN from ``0 * inf``. As an interval kind it defaults to
# the fa4 pair body, whose normalizer is every fa4 plan's ``rcp.approx.ftz``
# of the row sum (plus the exp2 pipe split) rather than the ws_overlap body's
# exact divide: a non-empty row's sum is at least about 1 (its row max
# contributes ``exp2(0)``), so the reciprocal is within 1 ulp and the
# flush-to-zero never fires; an empty row's published sum is exactly 0
# (``row_kv_begin_row_sum``) and maps to a 0 output row.
ROW_KV_BEGIN_MASK_KIND = "row_kv_begin_mask"
SOFTCAP_KIND = "softcap"
# The FlexAttention envelope: a BlockMask-driven KV walk plus the user's
# score_mod / mask_mod captured as pointwise FX regions (``FlexRegion``). A
# flex plan reports this single kind; its structure lives in ``kv_walk``,
# ``score_region`` and ``mask_region`` rather than in modifiers.
FLEX_SCORE_KIND = "flex"
LOWERED_SCORE_KINDS = frozenset(
    {
        CAUSAL_MASK_KIND,
        TENSOR_BIAS_KIND,
        RELATIVE_BIAS_KIND,
        ALIBI_BIAS_KIND,
        SLIDING_WINDOW_MASK_KIND,
        PREFIX_LM_MASK_KIND,
        DOCUMENT_MASK_KIND,
        ROW_KV_BEGIN_MASK_KIND,
        SOFTCAP_KIND,
        FLEX_SCORE_KIND,
    }
)
MASK_SCORE_KINDS = frozenset(
    {
        CAUSAL_MASK_KIND,
        SLIDING_WINDOW_MASK_KIND,
        PREFIX_LM_MASK_KIND,
        DOCUMENT_MASK_KIND,
        ROW_KV_BEGIN_MASK_KIND,
    }
)
SPARSE_TILE_MASK_KINDS = frozenset(
    {
        SLIDING_WINDOW_MASK_KIND,
        PREFIX_LM_MASK_KIND,
        DOCUMENT_MASK_KIND,
        ROW_KV_BEGIN_MASK_KIND,
    }
)
# ``(q/k head dim, v/o head dim)`` pairs the fused flash path hosts. The
# asymmetric pair is the MLA prefill shape (192-wide Q/K, 128-wide V/O); only
# the FA4 topology carries it (sQ and the K ring scale with the first dim, the
# V box, TMEM output and epilogue with the second).
FLASH_HEAD_DIM_PAIRS = frozenset({(64, 64), (128, 128), (192, 128)})


def flash_head_dims_supported(head_dim: int, head_dim_v: int) -> bool:
    """True when ``(head_dim, head_dim_v)`` is a fused-flash head-dim pair."""
    return (head_dim, head_dim_v) in FLASH_HEAD_DIM_PAIRS


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
    # The kernel's ``fast_math`` setting, recorded by the detector so the
    # flash bodies can be rendered without a compile environment. Only the
    # softcap transform reads it (``tanh.approx`` instead of the exact form).
    fast_math: bool = False


@dataclasses.dataclass(frozen=True)
class FlexKvWalk:
    """BlockMask-driven KV walk of the FlexAttention envelope.

    The four tables are the BlockMask tensors as the kernel receives them:
    counts ``[Bm, Hm, R]`` and 128-wide KV tile ids ``[Bm, Hm, R, L]`` (int32)
    for the partial (masked) and full lists, one row per ``q_block`` query
    rows. ``Bm`` / ``Hm`` may be 1 (broadcast over batch / heads). With
    ``q_block == 128`` the fa4 pair body walks the union of two rows per
    256-row pair, with a per-stage membership status per step; with
    ``q_block == 256`` both stages share one row.
    """

    part_cnt_name: str
    part_idx_name: str
    full_cnt_name: str
    full_idx_name: str
    q_block: int
    kv_block: int
    table_batch: int
    table_heads: int
    # ``kv_block >= kv_seq``: every KV tile is in one block, the walk is dense
    # and the tables are never read (torch's trivial-mask case).
    trivial: bool = False


@dataclasses.dataclass(frozen=True)
class FlexAuxBuffer:
    """A tensor captured by a flex region (FlexAttention's ``other_buffers``)."""

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    # How the load's kv-dependent index relates to the KV lane; drives the
    # renderer's load form (uniform broadcast / 16-byte vector / gather):
    # "row_uniform" | "column_affine" | "head" | "diagonal" | "row_col" |
    # "value" | "general".
    index_class: str = "general"


@dataclasses.dataclass(frozen=True)
class FlexRegion:
    """A captured pointwise FX region of the flex envelope.

    ``graph`` placeholders, in order: ``score`` (fp32), ``b``, ``h``, ``q_idx``,
    ``kv_idx`` (int32), then one per ``AttentionScorePlan.aux_buffers`` entry,
    then one per ``host_scalars`` entry. ``output`` is the region's value: the
    transformed score (fp32) for a score region, the keep predicate (bool) for
    a mask region.
    """

    graph: torch.fx.Graph
    output: torch.fx.Node


@dataclasses.dataclass(frozen=True)
class AttentionScorePlan:
    """Composable score-transform plan for the CuTe flash attention path."""

    head_dim: int
    qk_scale_log2: float
    lse_scale: float = 1.0
    modifiers: tuple[AttentionScoreModifier, ...] = ()
    # Grouped-query attention: K/V are indexed by ``tile_b.index // kv_group``
    # (``heads_q // heads_kv``), so they carry ``batch // kv_group`` leading
    # entries. ``1`` is plain multi-head attention.
    kv_group: int = 1
    # V/O head dim when it differs from the Q/K head dim (``head_dim`` keeps
    # the Q/K width the softmax scale is derived from); ``None`` is the
    # symmetric case.
    head_dim_v: int | None = None
    # Flex envelope (``FLEX_SCORE_KIND``): Q and K/V lengths (they may differ
    # and need not be multiples of 128), the BlockMask walk and the captured
    # score_mod / mask_mod regions. ``scale_peeled`` records whether the
    # matcher folded the ``qk * scale`` multiply into the PASS2 multiplier
    # (mask-only regions, raw QK units on every tile) or left it in the score
    # region (natural units; the hook runs on every tile and PASS2 applies
    # ``log2(e)``). ``scale_arg`` names the runtime scale symbol when the scale
    # is not a literal.
    q_seq: int | None = None
    kv_seq: int | None = None
    kv_walk: FlexKvWalk | None = None
    score_region: FlexRegion | None = None
    mask_region: FlexRegion | None = None
    aux_buffers: tuple[FlexAuxBuffer, ...] = ()
    host_scalars: tuple[str, ...] = ()
    scale_arg: str | None = None
    scale_peeled: bool = True

    @property
    def is_flex(self) -> bool:
        return self.kv_walk is not None

    @property
    def has_list_walk(self) -> bool:
        """A data-dependent per-item KV list (variable trip count, like the
        causal / pruned-interval plans, but driven by tables)."""
        return self.kv_walk is not None and not self.kv_walk.trivial

    @property
    def flex_has_score_region(self) -> bool:
        return self.is_flex and (self.score_region is not None or not self.scale_peeled)

    @property
    def flex_has_mask_region(self) -> bool:
        return self.is_flex and self.mask_region is not None

    @property
    def value_head_dim(self) -> int:
        return self.head_dim if self.head_dim_v is None else self.head_dim_v

    @property
    def asymmetric_head_dims(self) -> bool:
        return self.value_head_dim != self.head_dim

    @property
    def is_causal(self) -> bool:
        return any(modifier.kind == CAUSAL_MASK_KIND for modifier in self.modifiers)

    @property
    def modifier_kinds(self) -> tuple[str, ...]:
        if self.is_flex:
            return (FLEX_SCORE_KIND,)
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
    def row_kv_begin_masks(self) -> tuple[AttentionScoreModifier, ...]:
        return tuple(
            modifier
            for modifier in self.modifiers
            if modifier.kind == ROW_KV_BEGIN_MASK_KIND
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
    def has_causal_structure(self) -> bool:
        """Whether every mask kind here only ever keeps keys at or below the
        diagonal (causal, sliding-window, prefix-LM and document masks all do),
        so a work item's KV range is a contiguous interval ending at its own
        tile and the trip count varies per work item."""
        return self.is_causal or self.has_kv_tile_pruning

    @property
    def has_contiguous_kv_interval(self) -> bool:
        """Whether the plan's KV tile pruning is one contiguous ``[first, last]``
        interval per query tile: exactly one mask modifier, so the range
        helpers in ``cute_flash`` can express it (two masks compose into a
        per-tile predicate that is not an interval)."""
        return (
            sum(modifier.kind in MASK_SCORE_KINDS for modifier in self.modifiers) == 1
        )

    @property
    def requires_ws_overlap(self) -> bool:
        """Pruned plans whose active tiles are not one contiguous interval per
        query tile keep the ws_overlap body (its per-tile activity predicate);
        interval plans (including the per-row KV lower bound, whose range
        prelude is evaluated per resident query tile) may also run on the fa4
        pair body (the two families that host KV-tile-pruned masks; the
        resolver pins the rest away)."""
        return self.has_kv_tile_pruning and not self.has_contiguous_kv_interval

    def has_lowering(self) -> bool:
        if self.is_flex:
            # The matcher admits the regions (pure pointwise over the canonical
            # coordinates); a flex plan carries no kind modifiers.
            return not self.modifiers
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
            and len(self.row_kv_begin_masks) <= 1
            # The per-row lower bound owns the KV tile range outright and
            # already carries the causal conjunct: it must be the plan's only
            # mask modifier. Its range prelude is emitted only for the
            # single-interval form, so a second mask (another range-pruned
            # kind or a separately applied causal mask, as chained
            # ``torch.where`` calls produce) would leave the boundary mask and
            # the tile predicate referencing an undefined range.
            and (
                not self.row_kv_begin_masks
                or sum(modifier.kind in MASK_SCORE_KINDS for modifier in self.modifiers)
                == 1
            )
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
