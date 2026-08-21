# ruff: noqa: ANN202
"""Opt-in TileDependencySchedule wrapper for the unchanged Qwen3 FFN probe."""

from __future__ import annotations

import argparse
import sys
import types

import torch

import helion

_compat_probe = types.ModuleType("triton_qwen3_sm_overlap_probe")


def _build_helion_reference(*args: object, **kwargs: object):
    from triton_qwen3_whole_layer_persistent import build_helion_reference

    return build_helion_reference(*args, **kwargs)


_compat_probe.build_helion_reference = _build_helion_reference
sys.modules.setdefault("triton_qwen3_sm_overlap_probe", _compat_probe)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--epoch-replicas", type=int)
    parser.add_argument("--tile-dependency-stages", type=int)
    parser.add_argument("--continuation-split", type=int)
    parser.add_argument("--producer-order", choices=("physical", "consumer_major"))
    parser.add_argument("--strict-validation", action="store_true")
    args, remaining = parser.parse_known_args()

    import helion_qwen3_ffn_tile_dependency as probe

    probe.qwen3_ffn_tile_dependency = helion.kernel(
        static_shapes=True,
        autotune_effort="none",
        tile_dependency_schedule=helion.TileDependencySchedule(
            epoch_replicas=args.epoch_replicas,
            tile_dependency_stages=args.tile_dependency_stages,
            continuation_split=args.continuation_split,
            producer_order=args.producer_order,
        ),
    )(probe.qwen3_ffn_tile_dependency.fn)
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
