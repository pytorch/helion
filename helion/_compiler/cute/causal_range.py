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


def prove_sliding_window_tile_unmasked(
    *,
    window_size: int,
    tile_distance: int,
    query_layout: TileLayout,
    kv_layout: TileLayout,
) -> CausalRangeProof:
    """Prove a KV tile ``tile_distance`` tiles below a query tile is fully visible
    under the causal sliding window ``0 <= query - key <= window_size``.

    For equal tile geometries the element difference over the two complete
    tiles spans ``[d * stride - (width - 1), d * stride + (width - 1)]``; the
    tile is visible without any masked lane iff that whole span lies inside
    ``[0, window_size]``. The proof is independent of which query tile it is.
    """
    if window_size < 0 or tile_distance < 0:
        return CausalRangeProof(False, "invalid window or distance")
    if (
        query_layout.stride != kv_layout.stride
        or query_layout.width != kv_layout.width
        or query_layout.origin != kv_layout.origin
    ):
        return CausalRangeProof(False, "query/KV tile geometry mismatch")
    stride, width = kv_layout.stride, kv_layout.width
    if stride <= 0 or width <= 0:
        return CausalRangeProof(False, "empty or invalid range")
    if tile_distance * stride - (width - 1) < 0:
        return CausalRangeProof(False, "tile reaches above the diagonal")
    if tile_distance * stride + (width - 1) > window_size:
        return CausalRangeProof(False, "tile reaches beyond the window")
    return CausalRangeProof(True, "complete tile lies inside the window")


def sliding_window_unmasked_tile_distances(
    *,
    window_size: int,
    query_layout: TileLayout,
    kv_layout: TileLayout,
) -> IntegerInterval:
    """Half-open interval of tile distances below the diagonal whose tiles the
    window proof shows fully visible (``[1, 1)`` when there is none). Distances
    grow the maximum difference monotonically, so the proven set is a prefix of
    ``1, 2, ...`` and the scan stops at the first failure."""
    stop = 1
    if kv_layout.stride > 0:
        while stop <= window_size // kv_layout.stride + 1 and (
            prove_sliding_window_tile_unmasked(
                window_size=window_size,
                tile_distance=stop,
                query_layout=query_layout,
                kv_layout=kv_layout,
            ).proven
        ):
            stop += 1
    return IntegerInterval(1, stop)


def prove_prefix_tile_unmasked(
    *,
    prefix_length: int,
    kv_tile: int,
    kv_layout: TileLayout,
) -> CausalRangeProof:
    """Prove every key of KV tile ``kv_tile`` lies inside the prefix
    ``key < prefix_length`` (a prefix-LM tile that is visible to every query
    row regardless of the causal half)."""
    if prefix_length < 0 or kv_tile < 0:
        return CausalRangeProof(False, "invalid prefix or tile")
    elements = _element_interval(IntegerInterval(kv_tile, kv_tile + 1), kv_layout)
    if elements is None:
        return CausalRangeProof(False, "empty or invalid range")
    if elements.start < 0:
        return CausalRangeProof(False, "KV tile has out-of-bounds lanes")
    if elements.stop > prefix_length:
        return CausalRangeProof(False, "tile reaches past the prefix")
    return CausalRangeProof(True, "complete tile lies inside the prefix")


def prefix_unmasked_kv_tiles(
    *,
    prefix_length: int,
    kv_layout: TileLayout,
) -> IntegerInterval:
    """Half-open interval ``[0, n)`` of the leading KV tiles the prefix proof
    shows fully visible (a prefix of the tile index space, since tiles end
    monotonically later)."""
    stop = 0
    if kv_layout.stride > 0:
        while stop <= prefix_length // kv_layout.stride + 1 and (
            prove_prefix_tile_unmasked(
                prefix_length=prefix_length, kv_tile=stop, kv_layout=kv_layout
            ).proven
        ):
            stop += 1
    return IntegerInterval(0, stop)


def prove_row_bound_interior_layout(
    *,
    query_layout: TileLayout,
    kv_layout: TileLayout,
) -> CausalRangeProof:
    """Prove the runtime interior formula of the per-row KV lower bound is
    exact for these tile layouts: with equal, gap-free (``stride == width``),
    zero-origin geometries the first key of KV tile ``k`` is ``k * stride``
    and every key of the tile is at or past it, so ``k * stride >= max_bound``
    (``k >= ceil(max_bound / stride)``) means no row's bound cuts the tile,
    while ``k < query_tile`` keeps it below the diagonal.  The kernel
    evaluates ``(max_bound + 127) // 128`` per query tile;
    ``test_cute_causal_range`` checks that formula against the per-tile mask
    by brute force.
    """
    if (
        query_layout.stride != kv_layout.stride
        or query_layout.width != kv_layout.width
        or query_layout.origin != kv_layout.origin
    ):
        return CausalRangeProof(False, "query/KV tile geometry mismatch")
    if kv_layout.stride <= 0 or kv_layout.width <= 0:
        return CausalRangeProof(False, "empty or invalid range")
    if kv_layout.stride != kv_layout.width:
        return CausalRangeProof(False, "tiles overlap or leave gaps")
    if kv_layout.origin != 0:
        return CausalRangeProof(False, "tile origin is not the first key")
    return CausalRangeProof(True, "tile k starts at key k * stride")


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
