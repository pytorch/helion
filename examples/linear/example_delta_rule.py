"""
DeltaNet (Delta Rule) Example
==============================

Chunked linear attention with rank-1 correction and no decay (DeltaNet).
Keys are L2-normalized and beta is sigmoid-gated.
"""

from __future__ import annotations

from helion._testing.linear_attention_harness import LinearAttentionExampleHarness
from helion.experimental.linear_attention import LinearAttentionVariant

HARNESS = LinearAttentionExampleHarness(variant=LinearAttentionVariant.DELTA_RULE)


def main() -> None:
    print(f"=== {HARNESS.title} ===")
    HARNESS.test()
    print()
    HARNESS.benchmark()


if __name__ == "__main__":
    main()
