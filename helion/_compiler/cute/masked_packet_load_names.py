"""Generated-code names of the predicated packet loads (``masked_packet_load``).

Kept apart from the helpers themselves, which import the CuTe DSL: the AST
passes that recognize these calls run wherever Helion imports.
"""

from __future__ import annotations

# Lane byte width -> generated-code helper name (32 and 64 bytes: two and
# four consecutive 16-byte packets under one predicate).
MASKED_PACKET_LOAD_HELPERS = {
    64: "_cute_load_v64b_if",
    32: "_cute_load_v32b_if",
    16: "_cute_load_v16b_if",
    8: "_cute_load_v8b_if",
    4: "_cute_load_v4b_if",
}
MASKED_PACKET_LOAD_HELPER_NAMES = frozenset(MASKED_PACKET_LOAD_HELPERS.values())
