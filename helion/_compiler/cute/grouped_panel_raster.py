"""Signed cluster-index bounds for device grouped panel rastering."""

from __future__ import annotations

GROUPED_PANEL_ALTERNATE_KEY = "tcgen05_grouped_l2_swizzle_alternate"
GROUPED_PANEL_RATIO_KEY = "tcgen05_grouped_l2_swizzle_mn_ratio"
GROUPED_PANEL_RATIO_CHOICES = (None, 1, 2, 3, 4, 8, 16)


def grouped_panel_ratio_index_domain(
    columns: int, tile_n: int, cluster_n: int, ratio: int
) -> bool:
    """Bound the static M/N decision threshold before emitting Int32 math."""
    limit = (1 << 31) - 1
    if min(columns, tile_n, cluster_n, ratio) <= 0:
        return False
    width = tile_n * cluster_n
    if columns + width - 1 > limit:
        return False
    n_clusters = (columns + width - 1) // width
    return ratio * n_clusters <= limit


def grouped_device_panel_index_domain(
    groups: int,
    rows: int,
    columns: int,
    tile_m: int,
    tile_n: int,
    cluster_n: int,
) -> bool:
    """Bound SDK group prefixes, ceiling divisions and every panel product.

    Live intervals are clipped to ``rows``. A matched group has positive M/N
    cluster counts, and its panel span is at most their product. The worst
    group prefix also fits Int32 even if live intervals overlap.
    """
    limit = (1 << 31) - 1
    if min(groups, rows, columns, tile_m, tile_n, cluster_n) <= 0:
        return False
    cluster_width = tile_n * cluster_n
    if (
        rows * columns > limit
        or rows + tile_m - 1 > limit
        or columns + cluster_width - 1 > limit
    ):
        return False
    m_clusters = (rows + tile_m - 1) // tile_m
    n_clusters = (columns + cluster_width - 1) // cluster_width
    return groups * m_clusters * n_clusters <= limit
