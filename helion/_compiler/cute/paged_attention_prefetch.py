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
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .paged_attention_transport import PagedAttentionTransport

BOUNDARY_PREFETCH_WAVE_AHEAD = "wave_ahead"
BOUNDARY_PREFETCH_SIBLING = "sibling"
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


def boundary_prefetch_layout(
    transport: PagedAttentionTransport,
) -> tuple[int, int, int] | None:
    """What one byte-range request per page needs of the K layout.

    A request names the rows of both heads of a pair at once, so the two
    heads of a key must be adjacent 128-byte rows and the pair's packed scale
    words adjacent 512-byte runs: a request has two heads, hence a pair's
    first head is zero. Returns the byte strides of a page in the raw and in
    the scale tensor and the number of pages they hold, or None for a layout
    whose pair is not one range.
    """
    key = transport.key
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
