"""Cross-loop scheduled wrapper for the unchanged Qwen3 FFN probe."""

from __future__ import annotations

import argparse
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--strict-validation", action="store_true")
    args, remaining = parser.parse_known_args()

    from probes.qwen3 import helion_qwen3_ffn_tile_dependency as probe

    sys.argv = [sys.argv[0], *remaining]
    if not args.strict_validation:
        probe.main()
        return

    original_assert_close = torch.testing.assert_close

    def exact_assert_close(actual: object, expected: object, **kwargs: object) -> None:
        del kwargs
        original_assert_close(actual, expected, atol=0, rtol=0)

    torch.testing.assert_close = exact_assert_close
    try:
        probe.main()
    finally:
        torch.testing.assert_close = original_assert_close


if __name__ == "__main__":
    main()
