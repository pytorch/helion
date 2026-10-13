from __future__ import annotations

import dataclasses

import pytest

from helion._compiler.cute.attention_plan import ALIBI_BIAS_KIND
from helion._compiler.cute.attention_plan import CAUSAL_MASK_KIND
from helion._compiler.cute.attention_plan import ROW_KV_BEGIN_MASK_KIND
from helion._compiler.cute.attention_plan import SLIDING_WINDOW_MASK_KIND
from helion._compiler.cute.attention_plan import SOFTCAP_KIND
from helion._compiler.cute.attention_plan import AttentionScoreModifier
from helion._compiler.cute.attention_plan import AttentionScorePlan
from helion._compiler.cute.attention_plan import causal_score_plan
from helion._compiler.cute.causal_range import IntegerInterval
from helion._compiler.cute.causal_range import TileLayout
from helion._compiler.cute.causal_range import prefix_unmasked_kv_tiles
from helion._compiler.cute.causal_range import prove_causal_tile_range_unmasked
from helion._compiler.cute.causal_range import prove_descending_causal_prefix_unmasked
from helion._compiler.cute.causal_range import prove_prefix_tile_unmasked
from helion._compiler.cute.causal_range import prove_sliding_window_tile_unmasked
from helion._compiler.cute.causal_range import sliding_window_unmasked_tile_distances
from helion._compiler.cute.cute_flash import _flash_fa4_descending_causal_split_proof


@pytest.fixture
def full_tiles() -> TileLayout:
    return TileLayout(extent=512, stride=128, width=128)


def test_descending_causal_prefix_proves_complete_visible_tiles(
    full_tiles: TileLayout,
) -> None:
    proof = prove_descending_causal_prefix_unmasked(
        query_tiles=IntegerInterval(0, 4),
        query_layout=full_tiles,
        kv_layout=full_tiles,
    )
    assert proof.proven


def test_causal_range_rejects_diagonal_tile(full_tiles: TileLayout) -> None:
    proof = prove_causal_tile_range_unmasked(
        query_tiles=IntegerInterval(0, 4),
        kv_tiles=IntegerInterval(0, 4),
        kv_minus_query=IntegerInterval(-4, 1),
        query_layout=full_tiles,
        kv_layout=full_tiles,
    )
    assert not proof.proven
    assert proof.reason == "range includes a masked causal lane"


def test_causal_range_rejects_partial_sequence_tail() -> None:
    partial_tiles = TileLayout(extent=500, stride=128, width=128)
    proof = prove_descending_causal_prefix_unmasked(
        query_tiles=IntegerInterval(0, 4),
        query_layout=partial_tiles,
        kv_layout=partial_tiles,
    )
    assert not proof.proven
    assert proof.reason == "query tile has out-of-bounds lanes"


@pytest.mark.parametrize(
    ("query_tiles", "extent", "additional_modifiers", "tile_pruning", "reason"),
    [
        (None, 512, False, False, "symbolic range"),
        (IntegerInterval(0, 4), None, False, False, "dynamic extent"),
        (IntegerInterval(0, 4), 512, True, False, "additional score modifiers"),
        (IntegerInterval(0, 4), 512, False, True, "KV tile pruning"),
    ],
)
def test_descending_causal_prefix_conservative_fallbacks(
    query_tiles: IntegerInterval | None,
    extent: int | None,
    additional_modifiers: bool,
    tile_pruning: bool,
    reason: str,
) -> None:
    layout = TileLayout(extent=extent, stride=128, width=128)
    proof = prove_descending_causal_prefix_unmasked(
        query_tiles=query_tiles,
        query_layout=layout,
        kv_layout=layout,
        has_additional_modifiers=additional_modifiers,
        has_kv_tile_pruning=tile_pruning,
    )
    assert not proof.proven
    assert proof.reason == reason


def test_causal_range_handles_distinct_tile_geometries() -> None:
    proof = prove_causal_tile_range_unmasked(
        query_tiles=IntegerInterval(0, 4),
        kv_tiles=IntegerInterval(0, 7),
        kv_minus_query=IntegerInterval(-8, -1),
        query_layout=TileLayout(extent=512, stride=128, width=128),
        kv_layout=TileLayout(extent=512, stride=64, width=64),
    )
    assert proof.proven


def test_fa4_split_proof_requires_exact_static_sequence_coverage() -> None:
    score_plan = causal_score_plan(64)
    assert _flash_fa4_descending_causal_split_proof(
        sequence_extent=512,
        num_query_tiles=4,
        num_kv_tiles=4,
        score_plan=score_plan,
    ).proven

    tail = _flash_fa4_descending_causal_split_proof(
        sequence_extent=500,
        num_query_tiles=4,
        num_kv_tiles=4,
        score_plan=score_plan,
    )
    assert not tail.proven
    assert tail.reason == "partial or uncovered sequence tail"

    mismatch = _flash_fa4_descending_causal_split_proof(
        sequence_extent=512,
        num_query_tiles=4,
        num_kv_tiles=3,
        score_plan=score_plan,
    )
    assert not mismatch.proven
    assert mismatch.reason == "query/KV tile-count mismatch"


def test_fa4_split_proof_accepts_additive_modifiers_and_rejects_other_masks() -> None:
    # Additive transforms (ALiBi, softcap) change score values, not which
    # lanes the causal predicate masks, so the descending split stays proven
    # for them; a second mask changes the visible set and rejects it.
    base = causal_score_plan(64)
    for additive in (
        AttentionScoreModifier(ALIBI_BIAS_KIND, index_mode="mod", index_divisor=1),
        AttentionScoreModifier(SOFTCAP_KIND, value_log2=2.0),
    ):
        plan = dataclasses.replace(base, modifiers=(additive, *base.modifiers))
        assert _flash_fa4_descending_causal_split_proof(
            sequence_extent=512,
            num_query_tiles=4,
            num_kv_tiles=4,
            score_plan=plan,
        ).proven
    windowed = dataclasses.replace(
        base,
        modifiers=(
            *base.modifiers,
            AttentionScoreModifier(SLIDING_WINDOW_MASK_KIND, window_size=128),
        ),
    )
    proof = _flash_fa4_descending_causal_split_proof(
        sequence_extent=512,
        num_query_tiles=4,
        num_kv_tiles=4,
        score_plan=windowed,
    )
    assert not proof.proven
    assert proof.reason == "additional score modifiers"


def test_row_kv_begin_lowering_requires_a_single_mask_modifier() -> None:
    # ROW_KV_BEGIN folds the causal conjunct and owns the KV range: its range
    # prelude is emitted only for the single-interval form, so a plan that
    # also carries a separately applied causal mask (chained ``torch.where``
    # calls) or another range-pruned mask declines instead of rendering a
    # boundary mask over an undefined range.
    row_kv_begin = AttentionScoreModifier(
        ROW_KV_BEGIN_MASK_KIND, tensor_name="lb", index_mode="identity"
    )
    causal = AttentionScoreModifier(CAUSAL_MASK_KIND)
    window = AttentionScoreModifier(SLIDING_WINDOW_MASK_KIND, window_size=64)
    alone = AttentionScorePlan(
        head_dim=64, qk_scale_log2=1.0, modifiers=(row_kv_begin,)
    )
    assert alone.has_lowering()
    assert alone.requires_ws_overlap
    for modifiers in (
        (causal, row_kv_begin),
        (row_kv_begin, causal),
        (row_kv_begin, window),
    ):
        chained = AttentionScorePlan(
            head_dim=64, qk_scale_log2=1.0, modifiers=modifiers
        )
        assert not chained.has_lowering(), modifiers


@pytest.mark.parametrize(
    ("window_size", "tile_distance", "proven", "reason"),
    [
        (1024, 1, True, "complete tile lies inside the window"),
        (1024, 7, True, "complete tile lies inside the window"),
        (1024, 0, False, "tile reaches above the diagonal"),
        (1024, 8, False, "tile reaches beyond the window"),
        (300, 1, True, "complete tile lies inside the window"),
        (300, 2, False, "tile reaches beyond the window"),
        (100, 1, False, "tile reaches beyond the window"),
    ],
)
def test_sliding_window_tile_proof(
    full_tiles: TileLayout,
    window_size: int,
    tile_distance: int,
    proven: bool,
    reason: str,
) -> None:
    proof = prove_sliding_window_tile_unmasked(
        window_size=window_size,
        tile_distance=tile_distance,
        query_layout=full_tiles,
        kv_layout=full_tiles,
    )
    assert proof.proven is proven
    assert proof.reason == reason


def test_sliding_window_tile_proof_rejects_mismatched_geometry(
    full_tiles: TileLayout,
) -> None:
    proof = prove_sliding_window_tile_unmasked(
        window_size=1024,
        tile_distance=1,
        query_layout=full_tiles,
        kv_layout=TileLayout(extent=512, stride=64, width=64),
    )
    assert not proof.proven
    assert proof.reason == "query/KV tile geometry mismatch"


@pytest.mark.parametrize(
    ("window_size", "expected"),
    [
        (1024, IntegerInterval(1, 8)),
        (300, IntegerInterval(1, 2)),
        (100, IntegerInterval(1, 1)),
    ],
)
def test_sliding_window_unmasked_tile_distances(
    full_tiles: TileLayout, window_size: int, expected: IntegerInterval
) -> None:
    assert (
        sliding_window_unmasked_tile_distances(
            window_size=window_size, query_layout=full_tiles, kv_layout=full_tiles
        )
        == expected
    )


@pytest.mark.parametrize(
    ("prefix_length", "kv_tile", "proven", "reason"),
    [
        (200, 0, True, "complete tile lies inside the prefix"),
        (200, 1, False, "tile reaches past the prefix"),
        (1024, 7, True, "complete tile lies inside the prefix"),
        (1024, 8, False, "tile reaches past the prefix"),
        (64, 0, False, "tile reaches past the prefix"),
    ],
)
def test_prefix_tile_proof(
    full_tiles: TileLayout, prefix_length: int, kv_tile: int, proven: bool, reason: str
) -> None:
    proof = prove_prefix_tile_unmasked(
        prefix_length=prefix_length, kv_tile=kv_tile, kv_layout=full_tiles
    )
    assert proof.proven is proven
    assert proof.reason == reason


@pytest.mark.parametrize(
    ("prefix_length", "expected"),
    [
        (200, IntegerInterval(0, 1)),
        (1024, IntegerInterval(0, 8)),
        (64, IntegerInterval(0, 0)),
    ],
)
def test_prefix_unmasked_kv_tiles(
    full_tiles: TileLayout, prefix_length: int, expected: IntegerInterval
) -> None:
    assert (
        prefix_unmasked_kv_tiles(prefix_length=prefix_length, kv_layout=full_tiles)
        == expected
    )
