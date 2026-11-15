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

from typing import TYPE_CHECKING
from typing import cast

from ..._utils import next_power_of_2
from ..per_thread_planning import MAX_THREADS_PER_BLOCK

if TYPE_CHECKING:
    from collections.abc import Mapping

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


def fits_row_packets(
    row_itemsize: int | None, vector_width: int, capability: tuple[int, int] | None
) -> bool:
    """Whether ``vector_width`` of the widest elements a persistent row loads
    or stores, ``row_itemsize`` bytes each (``cute_row_itemsizes``; None when
    not recorded, which fits), fit one packet: 16 bytes, 32 on SM100+, as the
    memory ops' packet gate requires.  Only then does a row of several
    vectors per thread (``whole_vector_lanes``) hold them as packets
    ``threads * vector_width`` elements apart: a value no packet fits would
    be read an element per thread, the threads ``vector_width`` elements
    apart, so a warp instruction would span ``threads * vector_width``
    elements where the cyclic lanes read ``threads`` consecutive ones.  It
    sizes the packets only: a row whose accesses the packet gate refuses for
    their dtype or alignment still counts, and runs the chunked layout's
    order on scalar accesses."""
    if row_itemsize is None:
        return True
    max_bytes = 32 if capability is not None and capability >= (10, 0) else 16
    return vector_width * row_itemsize <= max_bytes


# The most lanes a persistent row's thread walks as several vectors
# (``whole_vector_lanes``).
MAX_VECTOR_ROW_LANES = 32


def whole_vector_lanes(lanes: int | None, vector_width: int) -> bool:
    """Whether the synthetic lanes a persistent row walks on one warp (or on
    whole warps, one vector per thread) are whole vectors of
    ``vector_width`` elements: one per thread, or several ``threads *
    vector_width`` elements apart (``plan_synthetic_lanes``' chunked rows),
    at most ``MAX_VECTOR_ROW_LANES`` lanes.  Either way one constexpr loop
    walks them, which ``cute_lane_unroll`` does not take.  Each vector's
    packet is loaded ahead of that loop, which unrolls every lane in every
    sweep, so more lanes (512 on a 16384-wide row on one warp at V = 2)
    keep the cyclic scalar lanes and their runtime loop."""
    return (
        lanes is not None
        and vector_width > 1
        and lanes % vector_width == 0
        and lanes <= MAX_VECTOR_ROW_LANES
    )


def requested_lane_unroll(config: Mapping[str, object]) -> int:
    """The ``cute_lane_unroll`` a masked persistent row plans its lanes for:
    1 under ``cute_vloop_sink``, a trade-off, since the sinking pass usually
    gives the knob to the sunk nest's lane loops
    (``_normalize_cute_lane_knobs``) and never generates the code again
    without it; where it leaves the nest rolled, the grid's lane loops take
    the knob, and an odd row keeps its live lanes with no unroll."""
    if config.get("cute_vloop_sink"):
        return 1
    return cast("int", config.get("cute_lane_unroll", 1))


def walks_live_lanes(
    size: int, threads: int, *, proven_bounds: bool, unroll: int
) -> bool:
    """Whether a masked persistent row of static ``size`` elements on
    ``threads`` threads walks exactly its live lanes (``masked_row_lanes``):
    those before ``ceil(size / threads)``, after which every thread is
    masked.  Two rows walk more:
    - under ``cute_proven_bounds``, a row whose live lanes cover it exactly
      (``size`` a multiple of ``threads``): the proven-bounds cleanup would
      fold their masks, dropping the masked loads' branches, and LLVM may
      then contract the multiply-adds around them differently, so that
      config keeps its padded count and its results;
    - under a lane unroll (``unroll`` above 1, ``requested_lane_unroll``), a
      row whose live lanes are odd in number: the unroll factors are powers
      of two that divide the lane count (``unroll_lane_loads._unroll_factor``),
      so the row walks its live lanes rounded up to a multiple of ``unroll``,
      which every factor up to it divides.  Only an odd count is rounded:
      the knob of 8 unrolls a 600-wide row's 19 live lanes, rounded to 24,
      by 8, but a 700-wide row's 22 by 2, the largest factor dividing 22.
      Rounding every count would make the knob uniform, but would change
      the code of rows whose live lanes already take a factor."""
    if proven_bounds and size % threads == 0:
        return False
    return unroll <= 1 or -(-size // threads) % 2 == 0


def masked_row_lanes(
    size: int, threads: int, lanes: int, *, proven_bounds: bool, unroll: int
) -> int:
    """The lanes a masked persistent row of static ``size`` elements on
    ``threads`` threads walks of its ``lanes`` padded ones: its live lanes,
    or where it walks more (``walks_live_lanes``) its padded count under
    ``cute_proven_bounds``, and under a lane unroll its odd live count
    rounded up to a multiple of ``unroll``, never past the padded count."""
    if proven_bounds and size % threads == 0:
        return lanes
    live = -(-size // threads)
    if unroll > 1 and live % 2:
        live = -(-live // unroll) * unroll
    return min(lanes, live)
