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

``streaming_store_rendered`` reads a streaming store off a generated module:
one of these flushes, or the paired attention body's own output stores.
"""

from __future__ import annotations

import ast

# Plain packet flush -> its L2 evict-last form.
PACKET_FLUSH_L2_EVICT_LAST = {
    "_cute_store_u16_vec": "_cute_store_u16x8_l2_evict_last",
    "_cute_store_u32_vec": "_cute_store_u32x4_l2_evict_last",
    "_cute_store_fp8e4m3fn_vec": "_cute_store_fp8e4m3fn_vec_l2_evict_last",
    "_cute_store_fp8e5m2_vec": "_cute_store_fp8e5m2_vec_l2_evict_last",
    "_cute_store_u8_vec": "_cute_store_u8_vec_l2_evict_last",
}
# Plain packet flush -> its streaming (evict-first) form.
PACKET_FLUSH_STREAMING = {
    "_cute_store_u16_vec": "_cute_store_u16x8_streaming",
    "_cute_store_u32_vec": "_cute_store_u32x4_streaming",
    "_cute_store_fp8e4m3fn_vec": "_cute_store_fp8e4m3fn_vec_streaming",
    "_cute_store_fp8e5m2_vec": "_cute_store_fp8e5m2_vec_streaming",
    "_cute_store_u8_vec": "_cute_store_u8_vec_streaming",
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


# The paired paged-attention body flushes no packet: it stores its outputs
# itself, with operators of its own template
# (``_paged_attention_io.store_output_rows``).  The kernel calls the body by
# this name and hands it a policy other than the default in this keyword
# (``paged_attention_codegen.codegen_paged_attention``).
PAIRED_BODY_NAME = "paged_attention_paired_body"
PAIRED_BODY_STORE_POLICY = "store_policy"


def _paired_body_streams(call: ast.Call) -> bool:
    """Whether ``call`` is the paired attention body's with the streaming
    policy for its output stores."""
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == PAIRED_BODY_NAME
        and any(
            keyword.arg == PAIRED_BODY_STORE_POLICY
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "streaming"
            for keyword in call.keywords
        )
    )


def streaming_store_rendered(source: str) -> bool:
    """Whether the generated module ``source`` stores with the streaming
    policy (``st.global.cs``, evict-first at L1 and L2).  That is a call of
    one of the ``PACKET_FLUSH_STREAMING`` helpers, the names
    ``packet_store_policy.lower_packet_store_policy`` gives the flushes that
    ``cute_store_policy='streaming'`` applies to, or the paired attention
    body's call with ``store_policy='streaming'``: that body flushes no
    packet and stores every element of its O output with the streaming
    operator of its own template.  A config with that policy whose stores
    are scalars, narrower packets or staged renders neither, and its kernel
    is the default policy's.  The autotuner's final selection asks this of a
    finalist's rendered source (``CuteBackend.finalist_preference``):
    ``source`` is the text ``BoundKernel.to_code`` rendered in this process,
    or the empty string for a kernel without generated code, so it parses.
    """
    module = ast.parse(source)
    streaming = frozenset(PACKET_FLUSH_STREAMING.values())
    return any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id in streaming)
            or _paired_body_streams(node)
        )
        for node in ast.walk(module)
    )
