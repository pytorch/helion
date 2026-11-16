"""Host arithmetic of the static persistent grouped tcgen05 scheduler: the
active cluster count of a device and the per-group CTA quotas.  Shared by
the ``tcgen05_grouped_static_persistent`` host wrapper emitter and the
launcher's grouped metadata builder."""

from __future__ import annotations


def grouped_static_active_clusters(
    *,
    num_sm: int,
    cluster_m: int,
    reserved_sms: int,
    num_sm_multiplier: int = 1,
    cluster_n: int = 1,
) -> int:
    if num_sm <= 0:
        raise ValueError("num_sm must be positive")
    if cluster_m <= 0:
        raise ValueError("cluster_m must be positive")
    if reserved_sms < 0:
        raise ValueError("reserved_sms must be non-negative")
    if type(num_sm_multiplier) is not int or num_sm_multiplier not in (1, 2):
        raise ValueError("num_sm_multiplier must be one or two")
    active_sms = max(1, num_sm - reserved_sms)
    return max(1, active_sms * num_sm_multiplier // (cluster_m * cluster_n))


def grouped_static_quotas(
    problem_shapes: tuple[tuple[int, int, int], ...],
    *,
    bm: int,
    bn: int,
    bk: int,
    max_active_clusters: int,
) -> tuple[int, ...]:
    """Balance each group's longest CTA without assigning idle CTAs.

    If fewer CTAs than groups are active, the returned ones are unused because
    the device selects the generic scheduler instead of the specialized path.
    """
    tile_counts = [
        ((problem_m + bm - 1) // bm) * ((problem_n + bn - 1) // bn)
        for problem_m, problem_n, _problem_k in problem_shapes
    ]
    k_tile_counts = [
        (problem_k + bk - 1) // bk
        for _problem_m, _problem_n, problem_k in problem_shapes
    ]
    quotas = [1] * len(problem_shapes)
    grid_size = min(sum(tile_counts), max_active_clusters)
    for _ in range(max(0, grid_size - len(quotas))):
        candidates = [
            index
            for index, (quota, tile_count) in enumerate(
                zip(quotas, tile_counts, strict=True)
            )
            if quota < tile_count
        ]
        group = max(
            candidates,
            key=lambda index: (
                (tile_counts[index] + quotas[index] - 1)
                // quotas[index]
                * k_tile_counts[index],
                tile_counts[index] * k_tile_counts[index],
                -index,
            ),
        )
        quotas[group] += 1
    return tuple(quotas)
