"""Co-resident cluster capacity for persistent CuTe cluster launches.

A persistent tcgen05 kernel sizes its grid by how many of its clusters the
device holds at once.  ``_NUM_SM // cluster_size`` is exact for the
CtaGroup.TWO pair (every TPC holds one pair) but over-counts larger
clusters: a cluster lives inside one GPC, and a B200's GPCs hold an uneven
number of TPCs, so its 148 SMs hold only 33 four-CTA clusters, not
``148 // 4 = 37`` (15 eight-CTA clusters, 7 sixteen-CTA ones).  The surplus
clusters of a ``_NUM_SM // 4`` persistent grid form a second hardware wave
that re-runs the prologue once the first wave's shortest cluster retires
(16 x 512x768x1024 fp16, 2x2 256x256x64 ab6: 19.3 -> 16.7 us at the exact
count).

The driver answers the question through ``cuOccupancyMaxActiveClusters``;
CUTLASS's ``HardwareInfo`` asks it for a dummy kernel that takes the whole
opt-in SMEM, i.e. one CTA per SM, which is the occupancy of every tcgen05
GEMM (their rings exceed half of the 228 KiB).  The answer only depends on
the device topology and the cluster size, so it is cached per process.
"""

from __future__ import annotations

from typing import cast

import torch

from ..triton.launcher import get_num_sm

_MAX_ACTIVE_CLUSTERS_CACHE: dict[tuple[int, int], int] = {}


def get_max_active_clusters(
    device: torch.device, cluster_size: int, *, reserved_sms: int = 0
) -> int:
    """Clusters of ``cluster_size`` one-CTA-per-SM CTAs the device co-schedules.

    Args:
        device: CUDA device of the launch.
        cluster_size: CTAs per thread-block cluster (``cluster_m * cluster_n``).
        reserved_sms: SMs to keep free for other work; caps the answer at
            ``(num_sm - reserved_sms) // cluster_size`` like
            :func:`helion.runtime.get_num_sm` does for the flat grid.

    Returns:
        The persistent grid size in clusters. Always at least 1.
    """
    assert device.type == "cuda", "cluster occupancy is a CUDA driver query"
    assert cluster_size >= 1
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, cluster_size)
    clusters = _MAX_ACTIVE_CLUSTERS_CACHE.get(key)
    if clusters is None:
        import cutlass.utils as cutlass_utils

        from .launcher import _ensure_cute_dsl_arch_env

        with torch.cuda.device(index):
            # ``HardwareInfo`` loads its probe kernel through the driver API,
            # which needs the device's primary context; entering the device
            # alone does not create one (nor does ``torch.cuda.init()``), but
            # asking for its current stream does.  The arch env var is read
            # from the same device.
            torch.cuda.current_stream(index)
            _ensure_cute_dsl_arch_env(())
            clusters = int(
                cutlass_utils.HardwareInfo(index).get_max_active_clusters(cluster_size)
            )
        _MAX_ACTIVE_CLUSTERS_CACHE[key] = clusters
    if reserved_sms > 0:
        clusters = min(
            clusters, get_num_sm(device, reserved_sms=reserved_sms) // cluster_size
        )
    return max(1, clusters)


def persistent_cluster_ctas(
    device: torch.device | None, cluster_size: int, num_sm: int
) -> int:
    """CTAs of a persistent one-CTA-per-SM cluster grid the device co-schedules.

    The driver's co-resident cluster count (``get_max_active_clusters``, a
    CUDA context query) where the device is known; ``num_sm // cluster_size``
    clusters otherwise (CPU renders, tests without a device).
    """
    if device is None or device.type != "cuda":
        return (num_sm // cluster_size) * cluster_size
    return get_max_active_clusters(device, cluster_size) * cluster_size


def resident_clusters(cluster_shape: object, device: torch.device | None) -> int | None:
    """The co-resident cluster count a wrapper of ``cluster_shape`` bakes in.

    Clusters above two CTAs take their persistent grid from the driver's
    count (:func:`persistent_cluster_ctas`), which varies with the GPC
    floorsweeping of same-SM-count devices, so the compiled-launcher keys
    carry it; ``None`` for the single CTA, the pair and CPU builds (their
    keys are unchanged).
    """
    if device is None or device.type != "cuda":
        return None
    if not isinstance(cluster_shape, (tuple, list)):
        return None
    cluster_size = 1
    for extent in cluster_shape:
        cluster_size *= int(cast("int", extent))
    if cluster_size <= 2:
        return None
    return get_max_active_clusters(device, cluster_size)
