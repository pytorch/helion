"""The threads and per-thread lanes of a CuTe reduction.

The reduction strategies size a reduction's threads and lane loop at code
generation (``CuteBackend.adjust_reduction_thread_count``,
``CuteLoopedReductionEmitter.plan_lane_loop``,
``CuteBackend.create_synthetic_reduction_lanes``,
``CutePersistentReductionEmitter.plan_synthetic_lanes``); config normalization
predicts the same lane loops from a config to decide whether
``cute_lane_unroll`` can take one (``_cute_config_forms_lane_loop``), and the
config generator's launch model fits the threads the same way
(``_reduction_axes``).  They all call these functions, so the predictions
follow the code they predict.
"""

from __future__ import annotations

from ..._utils import next_power_of_2
from ..per_thread_planning import MAX_THREADS_PER_BLOCK

# Threads of one warp: argmin/argmax reduce within one
# (``CutePersistentReductionEmitter.plan_max_threads``), a persistent row with
# synthetic lanes runs on one unless one vector per thread covers it
# (``plan_synthetic_lanes``), and an unpadded row takes whole warps
# (``unpadded_row_layout``).
WARP_THREADS = 32


def fit_reduction_threads(threads: int, other_threads: int) -> int:
    """``threads`` of a reduction, halved until they fit in one CTA beside
    the ``other_threads`` of the tiles and reductions it runs with."""
    while other_threads * threads > MAX_THREADS_PER_BLOCK and threads > 1:
        threads //= 2
    return threads


def looped_reduction_lanes(
    chunk: int, threads: int, vector_width: int
) -> tuple[int, int]:
    """``(lanes, width)`` of the lane loop a looped reduction walks its
    ``chunk`` in on ``threads`` threads: each thread takes ``ceil(chunk /
    threads)`` elements, in lanes of ``vector_width`` elements when that
    width divides them and of one element otherwise.  ``(1, 1)`` when the
    threads cover the chunk, which then needs no lane loop."""
    if threads <= 0 or chunk <= threads:
        return 1, 1
    elements = -(-chunk // threads)
    if vector_width > 1 and elements % vector_width == 0:
        return elements // vector_width, vector_width
    return elements, 1


def synthetic_reduction_lanes(threads: int, size: int) -> int | None:
    """The synthetic lanes a persistent reduction row of ``size`` elements
    walks on ``threads`` threads: its padded (power of two) extent over the
    threads, rounded up, or ``None`` when the threads cover it."""
    if threads <= 0:
        return None
    padded = next_power_of_2(max(1, size))
    if padded > threads:
        # Whole-warp thread counts need not divide the padded extent.
        return -(-padded // threads)
    return None


def tiled_row_threads(threads: int, size: int) -> int:
    """The threads of a persistent row of ``size`` elements that gets
    ``threads``: when the synthetic lanes of that many threads do not tile
    the row's padded extent exactly (a whole number of warps that is not a
    power of two, such as 96), the power of two below it, so that every
    thread's slice of lanes fits in the padded row
    (``plan_synthetic_lanes``)."""
    lanes = synthetic_reduction_lanes(threads, size)
    if lanes is None or threads * lanes == next_power_of_2(max(1, size)):
        return threads
    return 1 << (threads.bit_length() - 1)


def trims_padded_lanes(size: int, threads: int, *, proven_bounds: bool) -> bool:
    """Whether a masked persistent row of static ``size`` elements on
    ``threads`` threads walks only its live lanes (``plan_synthetic_lanes``'
    ``_live_lane_extent``): those before ``ceil(size / threads)``, after
    which every thread is masked.  Under ``cute_proven_bounds`` a row whose
    live lanes cover it exactly (``size`` a multiple of ``threads``) keeps
    its padded lanes: the proven-bounds cleanup would fold their masks,
    dropping the masked loads' branches, and LLVM may then contract the
    multiply-adds around them differently, so that config keeps its padded
    count and its results."""
    return not (proven_bounds and size % threads == 0)
