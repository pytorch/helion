"""
Retention Example
=================

Chunked linear attention with fixed per-head scalar decay (RetNet). FLA's
chunk_retention computes its own internal decay, so we do NOT pass g to FLA.
"""

from __future__ import annotations

from .linear_attention_engine import LinearAttentionVariant
from .linear_attention_harness import LinearAttentionExampleHarness

HARNESS = LinearAttentionExampleHarness(variant=LinearAttentionVariant.RETENTION)


def main() -> None:
    print(f"=== {HARNESS.title} ===")
    HARNESS.test()
    print()
    HARNESS.benchmark()


if __name__ == "__main__":
    main()
