from __future__ import annotations

import pytest

from helion._compiler.cute.attention_plan import causal_score_plan
from helion._compiler.cute.causal_range import IntegerInterval
from helion._compiler.cute.causal_range import TileLayout
from helion._compiler.cute.causal_range import prove_causal_tile_range_unmasked
from helion._compiler.cute.causal_range import prove_descending_causal_prefix_unmasked
from helion._compiler.cute.cute_flash import _flash_fa4_descending_causal_split_proof
from helion._compiler.cute.cute_flash import _flash_runtime_range_header


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


def test_causal_split_loop_header_stays_runtime() -> None:
    source = _flash_runtime_range_header(
        "flash_kv_unmask_iter",
        "flash_m_tile0",
    )
    assert source == (
        "for flash_kv_unmask_iter in cutlass.range(flash_m_tile0, unroll=1):"
    )
    assert "range_constexpr" not in source


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
