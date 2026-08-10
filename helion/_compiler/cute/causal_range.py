from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class IntegerInterval:
    """A half-open interval of integer values."""

    start: int
    stop: int

    @property
    def is_empty(self) -> bool:
        return self.start >= self.stop


@dataclasses.dataclass(frozen=True)
class TileLayout:
    """Element coordinates covered by a tile-indexed tensor dimension."""

    extent: int | None
    stride: int
    width: int
    origin: int = 0


@dataclasses.dataclass(frozen=True)
class CausalRangeProof:
    proven: bool
    reason: str


def _element_interval(
    tile_interval: IntegerInterval,
    layout: TileLayout,
) -> IntegerInterval | None:
    if tile_interval.is_empty or layout.stride <= 0 or layout.width <= 0:
        return None
    return IntegerInterval(
        layout.origin + tile_interval.start * layout.stride,
        layout.origin + (tile_interval.stop - 1) * layout.stride + layout.width,
    )


def _affine_maximum(
    *,
    coefficient: int,
    constant: int,
    domain: IntegerInterval,
) -> int | None:
    if domain.is_empty:
        return None
    endpoint = domain.stop - 1 if coefficient >= 0 else domain.start
    return coefficient * endpoint + constant


def prove_causal_tile_range_unmasked(
    *,
    query_tiles: IntegerInterval | None,
    kv_tiles: IntegerInterval | None,
    kv_minus_query: IntegerInterval | None,
    query_layout: TileLayout,
    kv_layout: TileLayout,
    has_additional_modifiers: bool = False,
    has_kv_tile_pruning: bool = False,
) -> CausalRangeProof:
    """Prove every lane in a tile range satisfies the standard causal mask.

    ``kv_minus_query`` bounds the tile-index difference for every executed
    query/KV pair. The proof is deliberately interval based: callers must
    provide static bounds for the complete runtime range, and any missing or
    partial-tile bound rejects mask elision.
    """
    if has_additional_modifiers:
        return CausalRangeProof(False, "additional score modifiers")
    if has_kv_tile_pruning:
        return CausalRangeProof(False, "KV tile pruning")
    if query_layout.extent is None or kv_layout.extent is None:
        return CausalRangeProof(False, "dynamic extent")
    if query_tiles is None or kv_tiles is None or kv_minus_query is None:
        return CausalRangeProof(False, "symbolic range")
    query_elements = _element_interval(query_tiles, query_layout)
    kv_elements = _element_interval(kv_tiles, kv_layout)
    if query_elements is None or kv_elements is None or kv_minus_query.is_empty:
        return CausalRangeProof(False, "empty or invalid range")
    if query_layout.extent < 0 or kv_layout.extent < 0:
        return CausalRangeProof(False, "invalid extent")
    if query_elements.start < 0 or query_elements.stop > query_layout.extent:
        return CausalRangeProof(False, "query tile has out-of-bounds lanes")
    if kv_elements.start < 0 or kv_elements.stop > kv_layout.extent:
        return CausalRangeProof(False, "KV tile has out-of-bounds lanes")

    # For every pair, kv_tile <= query_tile + max_delta. The causal predicate
    # is true for the complete tiles only if the largest possible KV element
    # lies strictly before the first query element. Using exclusive ends turns
    # that condition into kv_end <= query_start.
    max_delta = kv_minus_query.stop - 1
    visibility_gap = _affine_maximum(
        coefficient=kv_layout.stride - query_layout.stride,
        constant=(
            kv_layout.origin
            + max_delta * kv_layout.stride
            + kv_layout.width
            - query_layout.origin
        ),
        domain=query_tiles,
    )
    if visibility_gap is None or visibility_gap > 0:
        return CausalRangeProof(False, "range includes a masked causal lane")
    return CausalRangeProof(True, "complete range is causally visible")


def prove_descending_causal_prefix_unmasked(
    *,
    query_tiles: IntegerInterval | None,
    query_layout: TileLayout,
    kv_layout: TileLayout,
    has_additional_modifiers: bool = False,
    has_kv_tile_pruning: bool = False,
) -> CausalRangeProof:
    """Prove ``for i in range(q_tile): kv_tile = q_tile - 1 - i``.

    This is the unmasked suffix of a descending causal traversal. The derived
    intervals are conservative unions over every query tile in
    ``query_tiles``; the affine difference interval retains the relationship
    needed to prove causal visibility.
    """
    if query_tiles is None:
        return CausalRangeProof(False, "symbolic range")
    if query_tiles.start < 0 or query_tiles.stop <= 1:
        return CausalRangeProof(False, "empty or invalid range")
    kv_tiles = IntegerInterval(0, query_tiles.stop - 1)
    kv_minus_query = IntegerInterval(-query_tiles.stop, 0)
    return prove_causal_tile_range_unmasked(
        query_tiles=query_tiles,
        kv_tiles=kv_tiles,
        kv_minus_query=kv_minus_query,
        query_layout=query_layout,
        kv_layout=kv_layout,
        has_additional_modifiers=has_additional_modifiers,
        has_kv_tile_pruning=has_kv_tile_pruning,
    )
