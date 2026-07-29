"""
Full GLA (Diagonal Decay) Example
==================================

Chunked linear attention with per-dimension (diagonal) decay gates.
"""

from __future__ import annotations

from .linear_attention_engine import LinearAttentionVariant
from .linear_attention_harness import LinearAttentionExampleHarness

HARNESS = LinearAttentionExampleHarness(variant=LinearAttentionVariant.FULL_GLA)


def main() -> None:
    print(f"=== {HARNESS.title} ===")
    HARNESS.test()
    print()
    HARNESS.benchmark()


if __name__ == "__main__":
    main()
