"""Generated-code names of the packet flushes (``_cute_store_*_vec``).

A packet flush ``NAME(PTR, VALUES)`` writes the lane values collected in
``VALUES`` as one vector store.  The branch-local vectorizer
(``persistent_branch_vec``) emits the plain forms; ``packet_store_policy``
rewrites a proven full global packet to its L2 evict-last or streaming
(evict-first, ``st.global.cs``) twin.  Kept apart
from the helpers themselves (``vec_utils`` / ``l2_policy`` /
``packet_store_helpers``), which import the CuTe DSL: the AST passes that
recognize these calls (``packet_access.match_packet_access``) run wherever
Helion imports.  ``test_cute_packet_access.py`` pins this table to the
dtype table of ``persistent_branch_vec`` and to the backend's
``library_imports``.
"""

from __future__ import annotations

# Plain packet flush -> its L2 evict-last form.
PACKET_FLUSH_L2_EVICT_LAST = {
    "_cute_store_u16_vec": "_cute_store_u16x8_l2_evict_last",
    "_cute_store_u32_vec": "_cute_store_u32x4_l2_evict_last",
    "_cute_store_fp8e4m3fn_vec": "_cute_store_fp8e4m3fn_vec_l2_evict_last",
    "_cute_store_fp8e5m2_vec": "_cute_store_fp8e5m2_vec_l2_evict_last",
}
# Plain packet flush -> its streaming (evict-first) form.
PACKET_FLUSH_STREAMING = {
    "_cute_store_u16_vec": "_cute_store_u16x8_streaming",
    "_cute_store_u32_vec": "_cute_store_u32x4_streaming",
    "_cute_store_fp8e4m3fn_vec": "_cute_store_fp8e4m3fn_vec_streaming",
    "_cute_store_fp8e5m2_vec": "_cute_store_fp8e5m2_vec_streaming",
}
# ``cute_store_policy`` value -> its flush table.
PACKET_FLUSH_BY_POLICY = {
    "l2_evict_last": PACKET_FLUSH_L2_EVICT_LAST,
    "streaming": PACKET_FLUSH_STREAMING,
}
# Every generated packet flush, plain or cache-hinted.
PACKET_FLUSH_HELPER_NAMES = (
    frozenset(PACKET_FLUSH_L2_EVICT_LAST)
    | frozenset(PACKET_FLUSH_L2_EVICT_LAST.values())
    | frozenset(PACKET_FLUSH_STREAMING.values())
)
