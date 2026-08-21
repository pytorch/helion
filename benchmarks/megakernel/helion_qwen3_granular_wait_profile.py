# ruff: noqa: ANN001, ANN003, ANN202
"""Profile generated TileDependency waits for the granular Qwen3 probe.

This is deliberately a separate diagnostic.  It leaves the reference probes
and production lowering untouched, and injects timing only around compiler-
generated acquire waits.  The opaque Helion tile bodies are unchanged.
"""

from __future__ import annotations

import argparse
import sys
import types
from typing import TYPE_CHECKING

import helion_qwen3_granular_tile_dependency as granular
import torch

import helion
from helion._compiler.ast_extension import statement_from_string
from helion._compiler.program_id import ForEachProgramID

if TYPE_CHECKING:
    import ast


def _globaltimer_assignment(name: str) -> ast.stmt:
    return statement_from_string(
        f"{name} = tl.inline_asm_elementwise("
        "asm='mov.u64 $0, %globaltimer;', constraints='=l', args=[], "
        "dtype=tl.int64, is_pure=False, pack=1)"
    )


def _trace_waits() -> dict[int, list[str]]:
    original = ForEachProgramID._wait_for_counter
    sites_by_device_function: dict[int, list[str]] = {}
    trace_arg_by_device_function: dict[int, str] = {}

    def traced_wait_for_counter(**kwargs):
        device_function = kwargs["device_function"]
        key = id(device_function)
        sites = sites_by_device_function.setdefault(key, [])
        site = len(sites)
        sites.append(kwargs["prefix"])
        trace_arg = trace_arg_by_device_function.get(key)
        if trace_arg is None:
            trace_arg = ForEachProgramID._register_tile_dependency_state(
                device_function,
                name_hint="tile_dependency_wait_profile",
                numel="128",
                dtype=torch.int64,
                zero_init=True,
            )
            trace_arg_by_device_function[key] = trace_arg
        begin = device_function.new_var("tile_dependency_wait_begin", dce=False)
        end = device_function.new_var("tile_dependency_wait_end", dce=False)
        return [
            _globaltimer_assignment(begin),
            *original(**kwargs),
            _globaltimer_assignment(end),
            statement_from_string(
                f"tl.atomic_add({trace_arg} + {site}, {end} - {begin})"
            ),
            statement_from_string(f"tl.atomic_add({trace_arg} + {64 + site}, 1)"),
        ]

    ForEachProgramID._wait_for_counter = staticmethod(traced_wait_for_counter)
    return sites_by_device_function


def _persistent_trace(compiled) -> torch.Tensor:
    matches: list[torch.Tensor] = []
    for value in compiled.__globals__.values():
        namespace = getattr(value, "__dict__", {})
        for state in namespace.get("_helion_persistent_state_cache", {}).values():
            if state.dtype == torch.int64 and state.numel() == 128:
                matches.append(state)
        device_caches = getattr(value, "device_caches", None)
        if not device_caches or torch.cuda.current_device() not in device_caches:
            continue
        for kernel in device_caches[torch.cuda.current_device()][0].values():
            for state in (
                vars(kernel).get("_helion_persistent_state_cache", {}).values()
            ):
                if state.dtype == torch.int64 and state.numel() == 128:
                    matches.append(state)
    if len(matches) != 1:
        raise RuntimeError(f"expected one wait-profile state, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-multiplier", type=int, default=8)
    parser.add_argument("--kernel-stages", type=int, default=2)
    parser.add_argument("--epoch-replicas", type=int)
    parser.add_argument("--tile-dependency-stages", type=int)
    parser.add_argument("--continuation-split", type=int)
    parser.add_argument("--producer-order", choices=("physical", "consumer_major"))
    args = parser.parse_args()

    compatibility = types.ModuleType("triton_qwen3_sm_overlap_probe")
    compatibility.build_helion_reference = granular._build_helion_reference
    sys.modules.setdefault("triton_qwen3_sm_overlap_probe", compatibility)

    import helion_qwen3_tile_dependency as probe

    probe.require_idle_visible_gpu()
    sites_by_device_function = _trace_waits()
    probe.rms_norm_per_block_quant = granular.tiled_rms_norm_per_block_quant
    probe.reshape_and_cache_flash = granular.tiled_reshape_and_cache_flash
    probe.merge_attention_splits = granular.tiled_merge_attention_splits
    probe._probe_matched_config = granular._probe_config
    kernel, _source = probe._build_composite_kernel()
    schedule = helion.TileDependencySchedule(
        epoch_replicas=args.epoch_replicas,
        tile_dependency_stages=args.tile_dependency_stages,
        continuation_split=args.continuation_split,
        producer_order=args.producer_order,
    )
    scheduled = helion.kernel(
        static_shapes=True,
        autotune_effort="none",
        tile_dependency_schedule=schedule,
    )(kernel.fn)

    layer_args = argparse.Namespace(
        seed=0,
        hidden=4096,
        intermediate=12288,
        q_heads=32,
        kv_heads=8,
        head_dim=128,
        context=8192,
        block_size=16,
        attention_splits=128,
        group=128,
        eps=1e-6,
        rope_theta=1_000_000.0,
        worker_multiplier=args.worker_multiplier,
        kernel_stages=args.kernel_stages,
        qk_head_block=1,
        attention_context_block=32,
    )
    tensors = probe.allocate_layer(layer_args)
    composite_args = probe._composite_args(tensors, layer_args)
    bound = scheduled.bind(composite_args)
    config = granular._probe_config(bound, layer_args)
    compiled = bound.compile_config(config)
    compiled(*composite_args)
    torch.cuda.synchronize()
    trace = _persistent_trace(compiled)
    trace.zero_()
    compiled(*composite_args)
    torch.cuda.synchronize()

    sites = max(sites_by_device_function.values(), key=len)
    values = trace.cpu()
    for index, name in enumerate(sites):
        count = int(values[64 + index].item())
        total_ns = int(values[index].item())
        print(
            "WAIT_PROFILE",
            index,
            name,
            {
                "calls": count,
                "total_cta_us": total_ns / 1000.0,
                "mean_ns": total_ns / max(1, count),
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
