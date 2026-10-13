from __future__ import annotations

import dataclasses

import pytest

from helion._compiler.cute.attention_plan import ALIBI_BIAS_KIND
from helion._compiler.cute.attention_plan import CAUSAL_MASK_KIND
from helion._compiler.cute.attention_plan import DOCUMENT_MASK_KIND
from helion._compiler.cute.attention_plan import PREFIX_LM_MASK_KIND
from helion._compiler.cute.attention_plan import ROW_KV_BEGIN_MASK_KIND
from helion._compiler.cute.attention_plan import SLIDING_WINDOW_MASK_KIND
from helion._compiler.cute.attention_plan import SOFTCAP_KIND
from helion._compiler.cute.attention_plan import AttentionScoreModifier
from helion._compiler.cute.attention_plan import AttentionScorePlan
from helion._compiler.cute.attention_plan import causal_score_plan
from helion._compiler.cute.causal_range import CausalRangeProof
from helion._compiler.cute.causal_range import IntegerInterval
from helion._compiler.cute.causal_range import TileLayout
from helion._compiler.cute.causal_range import prefix_unmasked_kv_tiles
from helion._compiler.cute.causal_range import prove_causal_tile_range_unmasked
from helion._compiler.cute.causal_range import prove_descending_causal_prefix_unmasked
from helion._compiler.cute.causal_range import prove_prefix_tile_unmasked
from helion._compiler.cute.causal_range import prove_row_bound_interior_layout
from helion._compiler.cute.causal_range import prove_sliding_window_tile_unmasked
from helion._compiler.cute.causal_range import sliding_window_unmasked_tile_distances
from helion._compiler.cute.cute_flash import _flash_fa4_descending_causal_split_proof
from helion._compiler.cute.cute_flash import _flash_fa4_pruned_split_proof
from helion._compiler.cute.cute_flash import _flash_pruned_split_bounds
from helion._compiler.cute.cute_flash import _flash_pruned_split_softmax_segments


@pytest.fixture
def full_tiles() -> TileLayout:
    return TileLayout(extent=512, stride=128, width=128)


def _row_bound_tile_proof(
    *,
    kv_tile: int,
    query_tile: int,
    max_row_bound: int,
    layout: TileLayout,
) -> CausalRangeProof:
    """Oracle: KV tile ``kv_tile`` is fully visible to query tile
    ``query_tile`` under ``lb[row] <= key <= row`` when no row's bound
    exceeds ``max_row_bound``.  The causal half is the range proof; the
    lower-bound half holds when the tile's first key is at or past the
    largest bound, so no row can cut into it."""
    causal = prove_causal_tile_range_unmasked(
        query_tiles=IntegerInterval(query_tile, query_tile + 1),
        kv_tiles=IntegerInterval(kv_tile, kv_tile + 1),
        kv_minus_query=IntegerInterval(kv_tile - query_tile, kv_tile - query_tile + 1),
        query_layout=layout,
        kv_layout=layout,
    )
    if not causal.proven:
        return causal
    if layout.origin + kv_tile * layout.stride < max_row_bound:
        return CausalRangeProof(False, "tile starts before a row's lower bound")
    return CausalRangeProof(
        True, "complete tile lies past every row bound and below the diagonal"
    )


def _row_bound_unmasked_kv_tiles(
    *,
    max_row_bound: int,
    query_tile: int,
    layout: TileLayout,
) -> IntegerInterval:
    """The kernel's runtime interior ``[ceil(max_bound / 128), query_tile)``
    (``(max_bound + 127) // 128``, clamped at 0 and at the diagonal) in the
    terms of the layout the proof (``prove_row_bound_interior_layout``)
    accepts: gap-free equal tiles starting at the first key."""
    assert layout.stride == layout.width > 0
    first = max(0, -((layout.origin - max_row_bound) // layout.stride))
    return IntegerInterval(min(first, query_tile), query_tile)


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
    # The single-interval form is an interval kind: it runs on the fa4 pair
    # body (one range prelude per resident query tile) as well as ws_overlap.
    assert alone.has_contiguous_kv_interval
    assert not alone.requires_ws_overlap
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


@pytest.mark.parametrize(
    ("kv_tile", "query_tile", "max_row_bound", "proven", "reason"),
    [
        (
            2,
            3,
            256,
            True,
            "complete tile lies past every row bound and below the diagonal",
        ),
        (
            2,
            3,
            200,
            True,
            "complete tile lies past every row bound and below the diagonal",
        ),
        (2, 3, 257, False, "tile starts before a row's lower bound"),
        (3, 3, 0, False, "range includes a masked causal lane"),
        (1, 3, 129, False, "tile starts before a row's lower bound"),
        (
            0,
            3,
            0,
            True,
            "complete tile lies past every row bound and below the diagonal",
        ),
    ],
)
def test_row_bound_tile_proof(
    full_tiles: TileLayout,
    kv_tile: int,
    query_tile: int,
    max_row_bound: int,
    proven: bool,
    reason: str,
) -> None:
    proof = _row_bound_tile_proof(
        kv_tile=kv_tile,
        query_tile=query_tile,
        max_row_bound=max_row_bound,
        layout=full_tiles,
    )
    assert proof.proven is proven
    assert proof.reason == reason


def test_row_bound_unmasked_kv_tiles_matches_the_tile_proof(
    full_tiles: TileLayout,
) -> None:
    # The runtime formula ``ceil(max_bound / 128)`` picks exactly the tiles
    # the per-tile proof accepts, for every bound value and query tile.
    for query_tile in range(4):
        for max_row_bound in range(0, 513, 7):
            interval = _row_bound_unmasked_kv_tiles(
                max_row_bound=max_row_bound,
                query_tile=query_tile,
                layout=full_tiles,
            )
            assert interval.stop == query_tile
            for kv_tile in range(4):
                proven = _row_bound_tile_proof(
                    kv_tile=kv_tile,
                    query_tile=query_tile,
                    max_row_bound=max_row_bound,
                    layout=full_tiles,
                ).proven
                assert proven is (interval.start <= kv_tile < interval.stop)


def test_row_bound_unmasked_kv_tiles_brute_force(full_tiles: TileLayout) -> None:
    # Random per-row bound tables: every tile the formula calls interior has
    # no masked lane under ``lb[row] <= key <= row``.
    import random

    rng = random.Random(7)
    for _ in range(50):
        query_tile = rng.randrange(4)
        rows = range(query_tile * 128, query_tile * 128 + 128)
        bounds = [rng.randrange(0, row + 2) for row in rows]
        interval = _row_bound_unmasked_kv_tiles(
            max_row_bound=max(bounds), query_tile=query_tile, layout=full_tiles
        )
        for kv_tile in range(interval.start, interval.stop):
            for row, bound in zip(rows, bounds, strict=True):
                for key in range(kv_tile * 128, kv_tile * 128 + 128):
                    assert bound <= key <= row


def test_row_bound_interior_layout_proof(full_tiles: TileLayout) -> None:
    assert prove_row_bound_interior_layout(
        query_layout=full_tiles, kv_layout=full_tiles
    ).proven
    gapped = prove_row_bound_interior_layout(
        query_layout=full_tiles, kv_layout=TileLayout(extent=512, stride=128, width=64)
    )
    assert not gapped.proven
    assert gapped.reason == "query/KV tile geometry mismatch"
    overlapping = TileLayout(extent=512, stride=64, width=128)
    proof = prove_row_bound_interior_layout(
        query_layout=overlapping, kv_layout=overlapping
    )
    assert not proof.proven
    assert proof.reason == "tiles overlap or leave gaps"
    shifted = TileLayout(extent=512, stride=128, width=128, origin=5)
    proof = prove_row_bound_interior_layout(query_layout=shifted, kv_layout=shifted)
    assert not proof.proven
    assert proof.reason == "tile origin is not the first key"


def _single_mask_plan(modifier: AttentionScoreModifier) -> AttentionScorePlan:
    return dataclasses.replace(causal_score_plan(64), modifiers=(modifier,))


def test_fa4_pruned_split_proof_per_kind() -> None:
    window = _flash_fa4_pruned_split_proof(
        sequence_extent=2048,
        num_query_tiles=16,
        num_kv_tiles=16,
        score_plan=_single_mask_plan(
            AttentionScoreModifier(SLIDING_WINDOW_MASK_KIND, window_size=1024)
        ),
    )
    assert window.proof.proven
    assert window.kind == SLIDING_WINDOW_MASK_KIND
    assert window.interior_tiles == 7
    narrow = _flash_fa4_pruned_split_proof(
        sequence_extent=512,
        num_query_tiles=4,
        num_kv_tiles=4,
        score_plan=_single_mask_plan(
            AttentionScoreModifier(SLIDING_WINDOW_MASK_KIND, window_size=64)
        ),
    )
    assert narrow.proof.proven
    assert narrow.interior_tiles == 0
    prefix = _flash_fa4_pruned_split_proof(
        sequence_extent=1024,
        num_query_tiles=8,
        num_kv_tiles=8,
        score_plan=_single_mask_plan(
            AttentionScoreModifier(PREFIX_LM_MASK_KIND, prefix_length=200)
        ),
    )
    assert prefix.proof.proven
    assert prefix.kind == PREFIX_LM_MASK_KIND
    assert prefix.prefix_tiles == 1
    row_bound = _flash_fa4_pruned_split_proof(
        sequence_extent=512,
        num_query_tiles=4,
        num_kv_tiles=4,
        score_plan=_single_mask_plan(
            AttentionScoreModifier(ROW_KV_BEGIN_MASK_KIND, tensor_name="lb")
        ),
    )
    assert row_bound.proof.proven
    assert row_bound.kind == ROW_KV_BEGIN_MASK_KIND
    document = _flash_fa4_pruned_split_proof(
        sequence_extent=512,
        num_query_tiles=4,
        num_kv_tiles=4,
        score_plan=_single_mask_plan(
            AttentionScoreModifier(DOCUMENT_MASK_KIND, tensor_name="doc")
        ),
    )
    assert not document.proof.proven
    assert document.proof.reason == "mask has no static unmasked interior"
    tail = _flash_fa4_pruned_split_proof(
        sequence_extent=500,
        num_query_tiles=4,
        num_kv_tiles=4,
        score_plan=_single_mask_plan(
            AttentionScoreModifier(SLIDING_WINDOW_MASK_KIND, window_size=1024)
        ),
    )
    assert not tail.proof.proven
    assert tail.proof.reason == "partial or uncovered sequence tail"
    assert not _flash_fa4_pruned_split_proof(
        sequence_extent=512,
        num_query_tiles=4,
        num_kv_tiles=4,
        score_plan=causal_score_plan(64),
    ).proof.proven


def test_fa4_pruned_split_segments_cover_the_union() -> None:
    # Evaluate the emitted segment arithmetic on integers: for every pair,
    # window and order the three segments partition [first, last] and the
    # interior holds exactly the proven-visible tiles.
    import re

    def evaluate(expr: str, env: dict[str, int]) -> int:
        expr = re.sub(r"cutlass\.Int32\((\d+)\)", r"\1", expr)
        expr = expr.replace("cutlass.max", "max").replace("cutlass.min", "min")
        return int(eval(expr, {"max": max, "min": min}, env))

    for window_size in (64, 300, 1024):
        split = _flash_fa4_pruned_split_proof(
            sequence_extent=4096,
            num_query_tiles=32,
            num_kv_tiles=32,
            score_plan=_single_mask_plan(
                AttentionScoreModifier(
                    SLIDING_WINDOW_MASK_KIND, window_size=window_size
                )
            ),
        )
        back_tiles = (window_size + 127) // 128
        for pair in range(16):
            tiles = (2 * pair, 2 * pair + 1)
            first = max(0, tiles[0] - back_tiles)
            last = tiles[1]
            env = {"flash_kv_first": first, "flash_kv_last": last}
            for stage, m_tile in enumerate(tiles):
                env[f"flash_m_tile{stage}"] = m_tile
                bounds = _flash_pruned_split_bounds(split, str(stage), indent="")
                for line in bounds.splitlines():
                    name, _, value = line.partition(" = ")
                    env[name] = evaluate(value, env)
                for descending in (True, False):
                    visited: list[tuple[int, bool]] = []
                    for segment, masked in _flash_pruned_split_softmax_segments(
                        split, descending=descending
                    ):
                        count = evaluate(segment.loop_bound, env)
                        assert count >= 0
                        for i in range(count):
                            env[segment.loop_var] = i
                            assert segment.kv_expr is not None
                            visited.append((evaluate(segment.kv_expr, env), masked))
                    kv_tiles = [kv for kv, _ in visited]
                    expected = list(range(last, first - 1, -1))
                    if not descending:
                        expected.reverse()
                    assert kv_tiles == expected
                    for kv, masked in visited:
                        distance = m_tile - kv
                        fully_visible = 1 <= distance <= split.interior_tiles
                        assert masked is (not fully_visible)
