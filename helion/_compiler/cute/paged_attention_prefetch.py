"""Which boundary L2 prefetch a paired paged-attention bind can run.

The paired schedule's ``item`` partition launches one CTA per unit (one
request, the two heads of its pair), and the device dispatches the CTAs in
index order onto SMs as they free up. ``cute_mma_dag_boundary_prefetch=K``
lets the CTA's idle control warp request the first ``K`` K pages of a unit
that has not started into L2 (``_paged_attention_prefetch``), in one of two
forms that follow from the grid:

- ``wave_ahead``, more units than the device holds at once and at most two
  and a half waves of them: unit ``u`` serves unit ``u + capacity``, the
  unit that takes over its SM, right behind its own loader's last request.
- ``sibling``, fewer units than SMs with at most a fifth of the SMs left
  idle: the CTAs that start on the SMs the previous launch left idle serve
  the units still waiting for an SM, at entry. Only a launch the capture
  guard overlapped with a predecessor of the same kernel has such CTAs, and
  the host tells the kernel how many
  (``capture_pdl_cuda.finish_capture_launch``).

Both read only what the launch reads anyway: the served unit's bounds and
page ids, and addresses inside the K tensors. Neither changes a result.

A persistent partition launches one wave of CTAs, each of which streams
several units, so no unit waits for an SM and neither form applies. The
balanced partition of a grid of more units than SMs has a prefetch of its
own instead, with its own option, whose value is a distance in CTAs:

- ``range_ahead``: ``cute_mma_dag_boundary_prefetch_ahead=D`` lets the
  spare control warp of CTA ``c`` request the first pages of the stream of
  CTA ``c + D``, three of K and five of V, at the CTA's start. The device
  dispatches a grid's CTAs in index order, so that CTA starts later, and
  its loader then finds its first tiles on their way into L2. It reads the
  page table, one page id a lane, and the plan tables the CTA's scan left
  in shared memory; like the forms above it changes no result.

``cute_mma_dag_boundary_prefetch_cold=N`` gives the sibling form a depth
for the launches it otherwise leaves alone, those without a head start: the
CTAs an overlapped launch of the grid would start on idle SMs then request
the waiting units' first ``N`` K pages. Those requests are for a later
launch over the same pages, not for the units they name
(``cute_mma_dag.MMA_DAG_BOUNDARY_PREFETCH_COLD_KEY`` has what was measured).
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .paged_attention_transport import PagedAttentionTransport
    from .paged_attention_transport import PagedTmaOperand

BOUNDARY_PREFETCH_WAVE_AHEAD = "wave_ahead"
BOUNDARY_PREFETCH_SIBLING = "sibling"
# The distances, in CTAs, of the range-ahead prefetch
# (``cute_mma_dag_boundary_prefetch_ahead``). 48 is the handwritten
# kernel's constant for 148 SMs, 96 twice that; nothing derives either.
# Measured (B200, GPU 3, the SDG eval harness, 320 requests of a 2047-key
# window, frozen harness at two calls a sample, us per call against the
# tuned config, 4 legs, on top of the newest-first walk with the clipped
# page eight pages from a unit's end and a raw V ring of five tiles): 48
# CTAs -1.11 +- 0.07 and 96 CTAs -1.45 +- 0.10, one timer step each; one
# CTA ahead +0.29 +- 0.54 and no step; the same code with no CTA served
# -0.50 +- 0.11 and no step, so the requests carry the gain. At 148 and
# 296 requests the form alone (48 CTAs) read -0.47 +- 0.06 / -0.60 and
# -0.31 +- 0.03 / -0.70 at 3 or 2 / 12 calls, at 320 requests +0.01 /
# -0.14: there it gains only beside the walk and the ring, which is why
# the seeds name the three together.
BOUNDARY_PREFETCH_RANGE_AHEAD_DISTANCES: tuple[int, ...] = (48, 96)
# Pages of the served stream a range-ahead prefetch requests, from its first
# step: the handwritten kernel's counts, one lane of the spare warp a page.
RANGE_AHEAD_KEY_PAGES = 3
RANGE_AHEAD_VALUE_PAGES = 5
# Units up to which the spare warp of a persistent CTA scans no unit's
# length (``paged_attention_shared_memory.plan_geometry``: 19 scanning
# warps of 32 units). The range-ahead role waits at the scan's barrier, as
# a warp that scans cannot.
RANGE_AHEAD_UNIT_LIMIT = 19 * 32
# Both limits are measured on 148 SMs with 8 pages of a 17-page window,
# back-to-back calls over two or three inputs, in us per call against none.
#
# A grid past this many waves has no form: in long runs of calls 160 to 296
# units measured 0.5 to 1.8 us less and 320 to 380 units 0.1 to 0.8 us
# less, but 400 units about 1 us more, 443 units 0.3 more and 444 units 1.2
# more. One unit's requests serve one boundary; what a longer grid needs is
# a budget per boundary.
BOUNDARY_PREFETCH_WAVE_LIMIT = Fraction(5, 2)
# A one-wave grid that leaves more than this share of the SMs idle has no
# form either. In long runs of calls 90 to 108 units (39% to 27% idle)
# measured 0.3 to 0.5 us more with 8 pages, as they did with 4 and 12, 112
# and 116 units no change, and 119 to 144 units (20% to 3% idle) 0.1 to 0.9
# us less. Why the requests cost time there is not established: the same
# number of requests for one page cost nothing.
BOUNDARY_PREFETCH_IDLE_LIMIT = Fraction(1, 5)


def boundary_prefetch_form(units: int, capacity: int) -> str:
    """The form a grid of ``units`` CTAs takes on ``capacity`` resident CTAs,
    or the empty string for none.

    ``capacity`` is the SM count a paired launch may use (one CTA per SM),
    at most one when the target's is unknown. A grid that exactly fills the
    device has neither idle SMs nor a unit that waits for another's.
    """
    if capacity <= 1 or units <= 0:
        return ""
    if units < capacity:
        if capacity - units <= BOUNDARY_PREFETCH_IDLE_LIMIT * capacity:
            return BOUNDARY_PREFETCH_SIBLING
        return ""
    if capacity < units <= BOUNDARY_PREFETCH_WAVE_LIMIT * capacity:
        return BOUNDARY_PREFETCH_WAVE_AHEAD
    return ""


def range_ahead_distances(units: int, capacity: int) -> tuple[int, ...]:
    """The distances the range-ahead prefetch has on a balanced launch of
    ``units`` units on ``capacity`` resident CTAs, none where the grid has
    no such role.

    More units than SMs: the launch is one wave of ``capacity`` CTAs that
    each stream several units, the grid the role was measured on, where the
    balanced partition is seeded and the newest-first walk runs on it. The
    spare warp scans no length (``RANGE_AHEAD_UNIT_LIMIT``). And a distance
    names a CTA of the grid: one at which no CTA serves another would be
    the kernel without the requests under another name.
    """
    if not 1 < capacity < units <= RANGE_AHEAD_UNIT_LIMIT:
        return ()
    return tuple(
        distance
        for distance in BOUNDARY_PREFETCH_RANGE_AHEAD_DISTANCES
        if distance < capacity
    )


def boundary_prefetch_layout(
    transport: PagedAttentionTransport, operand: PagedTmaOperand | None = None
) -> tuple[int, int, int] | None:
    """What one byte-range request per page needs of an operand's layout,
    K's unless ``operand`` is given.

    A request names the rows of both heads of a pair at once, so the two
    heads of a key must be adjacent 128-byte rows and the pair's packed scale
    words adjacent 512-byte runs: a request has two heads, hence a pair's
    first head is zero. Returns the byte strides of a page in the raw and in
    the scale tensor and the number of pages they hold, or None for a layout
    whose pair is not one range.
    """
    key = transport.key if operand is None else operand
    if (
        transport.heads != 2
        or transport.subpages != 1
        or transport.key_width != 128
        or key.raw.dtype.itemsize != 1
        or key.scale.dtype.itemsize != 1
        or key.raw_strides[:3] != (1, 256, 128)
        or key.scale_strides[:2] != (1, 128)
        or key.raw_shape[4] != key.scale_shape[3]
    ):
        return None
    return key.raw_strides[4], key.scale_strides[3] * 4, key.raw_shape[4]
