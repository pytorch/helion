"""
Run all linear attention examples (test + benchmark).

Usage::

    python -m examples.linear.all
    HELION_USE_DEFAULT_CONFIG=1 python -m examples.linear.all
"""

from __future__ import annotations

import importlib
import sys
import traceback

EXAMPLES = [
    "example_simple_gla",
    "example_full_gla",
    "example_delta_rule",
    "example_gated_delta_rule",
    "example_vanilla_linear_attn",
    "example_retention",
    "example_mamba2_ssd",
    "example_rwkv6",
    "example_kda",
]


def main() -> None:
    results: list[tuple[str, str]] = []

    for name in EXAMPLES:
        print(f"\n{'=' * 70}")
        print(f" {name}")
        print(f"{'=' * 70}")
        try:
            mod = importlib.import_module(f"examples.linear.{name}")
            mod.main()
            results.append((name, "OK"))
        except Exception:
            traceback.print_exc()
            results.append((name, "FAIL"))

    print(f"\n{'=' * 70}")
    print(" Summary")
    print(f"{'=' * 70}")
    for name, status in results:
        print(f"  {name:<40} {status}")

    ok = sum(1 for _, s in results if s == "OK")
    print(f"\n{ok}/{len(results)} passed")

    if ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
