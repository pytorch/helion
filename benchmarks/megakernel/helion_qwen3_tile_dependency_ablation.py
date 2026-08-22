# ruff: noqa: ANN001, ANN202
"""Ablate TileDependency schedule families without changing the source probe."""

from __future__ import annotations

import argparse
import dataclasses
import sys
import types

import torch

import helion
from helion._compiler import tile_dependency_planner

# The original comparison probe was superseded by the production persistent
# probe in the exploration worktree.  Keep this new ablation self-contained by
# supplying the one compatibility symbol the Helion harness imports, without
# editing either existing probe.
_compat_probe = types.ModuleType("triton_qwen3_sm_overlap_probe")


def _build_helion_reference(*args: object, **kwargs: object):
    from triton_qwen3_whole_layer_persistent import build_helion_reference

    return build_helion_reference(*args, **kwargs)


_compat_probe.build_helion_reference = _build_helion_reference
sys.modules.setdefault("triton_qwen3_sm_overlap_probe", _compat_probe)


def _without_on_ready(events):
    return tuple(dataclasses.replace(event, on_ready_root=None) for event in events)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--disable-continuation", action="store_true")
    parser.add_argument("--strict-validation", action="store_true")
    args, remaining = parser.parse_known_args()

    if args.disable_continuation:
        original_counted_events = tile_dependency_planner._derive_counted_events
        original_coalesced_events = (
            tile_dependency_planner._derive_coalesced_keyed_events
        )

        def derive_counted_events_without_continuation(**kwargs: object):
            return _without_on_ready(original_counted_events(**kwargs))

        def derive_coalesced_events_without_continuation(**kwargs: object):
            return _without_on_ready(original_coalesced_events(**kwargs))

        tile_dependency_planner._derive_counted_events = (
            derive_counted_events_without_continuation
        )
        tile_dependency_planner._derive_coalesced_keyed_events = (
            derive_coalesced_events_without_continuation
        )

    import helion_qwen3_tile_dependency as probe

    # Attach the new opt-in policy to a fresh Kernel while preserving the
    # existing probe module and generated Helion source verbatim.
    probe.qwen3_layer_tile_dependency = helion.kernel(
        static_shapes=True,
        autotune_effort="none",
        tile_dependency_schedule=helion.TileDependencySchedule(),
    )(probe.qwen3_layer_tile_dependency.fn)

    sys.argv = [sys.argv[0], *remaining]
    if not args.strict_validation:
        probe.main()
        return

    allocations: list[dict[str, torch.Tensor]] = []
    original_allocate_layer = probe.allocate_layer
    original_assert_close = torch.testing.assert_close
    validation_names = iter(
        (
            "output",
            "qkv",
            "partial_out",
            "partial_lse",
            "attention",
            "attention_out",
            "gate_up",
            "activation_q",
            "activation_scale",
        )
    )
    validation_failures: list[str] = []
    checked_bridge_state = False

    def tracked_allocate_layer(namespace):
        tensors = original_allocate_layer(namespace)
        allocations.append(tensors)
        return tensors

    def check_exact(name: str, actual: object, expected: object) -> None:
        actual_tensor = torch.as_tensor(actual)
        expected_tensor = torch.as_tensor(expected)
        if actual_tensor.numel() == expected_tensor.numel():
            actual_tensor = actual_tensor.view_as(expected_tensor)
        try:
            original_assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)
            print(f"STRICT_VALIDATION {name} exact", flush=True)
        except AssertionError:
            actual_float = actual_tensor.float()
            expected_float = expected_tensor.float()
            difference = (actual_float - expected_float).abs()
            mismatch = actual_tensor != expected_tensor
            validation_failures.append(name)
            print(
                "STRICT_VALIDATION",
                name,
                {
                    "mismatches": int(mismatch.sum().item()),
                    "elements": actual_tensor.numel(),
                    "max_abs": float(difference.max().item()),
                },
                flush=True,
            )

    def exact_assert_close(actual, expected, **kwargs: object) -> None:
        nonlocal checked_bridge_state
        if not checked_bridge_state and len(allocations) >= 2:
            checked_bridge_state = True
            persistent_tensors, reference_tensors = allocations[:2]
            for bridge_name in (
                "pre_q",
                "pre_scale",
                "kv_cache",
                "attention_q",
                "attention_scale",
                "ffn_q",
                "ffn_scale",
                "residual",
            ):
                check_exact(
                    bridge_name,
                    persistent_tensors[bridge_name],
                    reference_tensors[bridge_name],
                )
        name = next(validation_names, "unexpected")
        check_exact(name, actual, expected)

    probe.allocate_layer = tracked_allocate_layer
    torch.testing.assert_close = exact_assert_close
    try:
        probe.main()
    finally:
        probe.allocate_layer = original_allocate_layer
        torch.testing.assert_close = original_assert_close
    if validation_failures:
        raise AssertionError(
            "strict validation failed for " + ", ".join(validation_failures)
        )


if __name__ == "__main__":
    main()
