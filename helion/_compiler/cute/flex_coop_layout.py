"""Layout constants of the flex hook's warp-cooperative table staging, shared by the
renderer (``flash_score_mod``) and the CuTe DSL runtime helpers (``flex_score_mod_runtime``).

Kept free of ``cutlass`` imports: the renderer is imported with ``helion`` itself and must
import without the DSL installed."""

from __future__ import annotations

# Request widths (bytes per table row per request) of the warp-cooperative table loads
# (``cute_flash_flex_coalesce``: whole lines of 4 / 8 / 16 rows at 128 / 64 / 32).
COOP_WIDTHS = (32, 64, 128)

# Every staged row is the chunk span plus this pad: it keeps every 8-lane phase of the
# copies' shared-memory writes (``coop_lane_map``'s row/slot assignment) and of the row
# reads on distinct four-bank groups.
COOP_PAD_BYTES = 16


def coop_pitch(row_bytes: int) -> int:
    """Byte pitch of one staged row: the row's chunk span plus the pad."""
    return row_bytes + COOP_PAD_BYTES
