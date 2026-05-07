from __future__ import annotations

# Validated CtaGroup.TWO autotune/runtime envelope for the B200 CuTe path.
# Re-verify the K-cap runtime and guard-boundary tests before raising the
# K-tile threshold or broadening the tile shape.
TCGEN05_TWO_CTA_BLOCK_M = 256
TCGEN05_TWO_CTA_BLOCK_N = 256
TCGEN05_TWO_CTA_MAX_K_TILES = 256
# Best measured seed L2 grouping for the validated 4096^3 CtaGroup.TWO row.
TCGEN05_TWO_CTA_SEED_L2_GROUPING = 4

# CtaGroup.ONE tcgen05 MMA covers 64/128 M tiles; 256 M tiles are validated only
# after projecting onto the CtaGroup.TWO path.
TCGEN05_ONE_CTA_MAX_BLOCK_M = 128
