"""
KDA (Kimi Delta Attention) Example
===================================

Diagonal (per-dimension) decay with rank-1 correction. This is the most
general variant: diagonal gating combined with the delta rule. Keys are the
L2-normalized constant, so only q and v carry gradients.
"""

from __future__ import annotations

from .linear_attention_engine import LinearAttentionVariant
from .linear_attention_harness import LinearAttentionExampleHarness

HARNESS = LinearAttentionExampleHarness(variant=LinearAttentionVariant.KDA)


def main() -> None:
    print(f"=== {HARNESS.title} ===")
    HARNESS.test()
    print()
    print(f"=== {HARNESS.title}: fused input preamble ===")
    HARNESS.test_fused_preamble()
    print()
    print(f"=== {HARNESS.title}: varlen ===")
    HARNESS.test_varlen()
    print()
    HARNESS.benchmark()


if __name__ == "__main__":
    main()
