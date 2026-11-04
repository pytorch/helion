"""
Vanilla Linear Attention Example
=================================

Chunked linear attention with no decay and no correction term.
"""

from __future__ import annotations

from helion._testing.linear_attention_harness import LinearAttentionExampleHarness
from helion.experimental.linear_attention import LinearAttentionVariant

HARNESS = LinearAttentionExampleHarness(variant=LinearAttentionVariant.VANILLA)


def main() -> None:
    print(f"=== {HARNESS.title} ===")
    HARNESS.test()
    print()
    HARNESS.benchmark()


if __name__ == "__main__":
    main()
