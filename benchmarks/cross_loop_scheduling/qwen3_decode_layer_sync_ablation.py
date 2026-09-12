"""Diagnostic synchronization ablations for the untouched Qwen3 decode source.

This probe changes only compiler-selected synchronization mechanisms.  It does
not change source arithmetic, fusion boundaries, tile configuration, or model
constants.  Root IDs are intentionally confined to this diagnostic probe.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from pretuned_kernels._bench import bench_pre_captured_cudagraphs
from pretuned_kernels._bench import capture_cuda_graph
from pretuned_kernels._bench import thermal_warmup
from pretuned_kernels.megakernels.qwen3_decode_layer import qwen3_decode_layer as qwen
from pretuned_kernels.megakernels.qwen3_decode_layer._helion_aot_qwen3_decode_layer_cuda_sm100 import (
    CONFIG,
)
import torch

import helion
import helion._compiler.cross_loop_codegen as cross_loop_codegen
import helion._compiler.cross_loop_scheduler as cross_loop_scheduler

if TYPE_CHECKING:
    from helion.runtime.kernel import CompiledConfig


VARIANTS = {
    "current": (frozenset(), frozenset()),
    "barrier_5_6": (frozenset(), frozenset({((5,), 6)})),
    "no_cont_6_7": (frozenset({7}), frozenset()),
    "barrier_6_7": (frozenset({7}), frozenset({((6,), 7)})),
    "no_cont_12_13": (frozenset({13}), frozenset()),
    "historical_sync": (
        frozenset({7, 13}),
        frozenset({((5,), 6), ((6,), 7)}),
    ),
}


def _compiled_kernels(call: Callable[..., object]) -> tuple[Any, ...]:
    wrappers: list[Callable[..., object]] = [call]
    for cell in call.__closure__ or ():
        value = cell.cell_contents
        if isinstance(value, dict):
            wrappers.extend(item for item in value.values() if callable(item))
    kernels: list[Any] = []
    for wrapper in wrappers:
        for value in wrapper.__globals__.values():
            device_caches = getattr(value, "device_caches", None)
            if device_caches is None:
                continue
            for cache_pair in device_caches.values():
                for compiled in cache_pair[0].values():
                    if all(compiled is not previous for previous in kernels):
                        kernels.append(compiled)
    return tuple(kernels)


def _edge(
    plan: cross_loop_scheduler.ReadinessCounterPlan,
    consumer_index: int,
) -> tuple[tuple[int, ...], int]:
    return (
        tuple(producer.producer_root for producer in plan.producers),
        plan.consumers[consumer_index].consumer_root,
    )


def _compile_variant(
    args: tuple[object, ...],
    *,
    name: str,
) -> tuple[CompiledConfig, dict[str, object]]:
    disabled_continuation_roots, disabled_counter_edges = VARIANTS[name]
    bound = qwen.qwen3_decode_layer.bind(args)
    values = deepcopy(CONFIG)
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)

    original_continuations = (
        cross_loop_scheduler.choose_final_arrival_continuations
    )
    original_counters = cross_loop_scheduler.choose_readiness_counters
    original_global = cross_loop_scheduler._global_unit_list_schedule
    original_plan = cross_loop_codegen.build_static_pipeline_plan
    records: list[dict[str, object]] = []

    def continuations(*call_args: object, **call_kwargs: object):  # noqa: ANN202
        graph = call_args[0]
        return tuple(
            continuation
            for continuation in original_continuations(*call_args, **call_kwargs)
            if graph.event(continuation.event_id)
            .consumers[continuation.consumer_index]
            .consumer_root
            not in disabled_continuation_roots
        )

    def counters(*call_args: object, **call_kwargs: object):  # noqa: ANN202
        selected = original_counters(*call_args, **call_kwargs)
        result: list[cross_loop_scheduler.ReadinessCounterPlan] = []
        for plan in selected:
            retained_indices = tuple(
                index
                for index in range(len(plan.consumers))
                if _edge(plan, index) not in disabled_counter_edges
            )
            if not retained_indices:
                continue
            result.append(
                dataclasses.replace(
                    plan,
                    consumers=tuple(plan.consumers[index] for index in retained_indices),
                    continuation_consumer_index=(
                        retained_indices.index(plan.continuation_consumer_index)
                        if plan.continuation_consumer_index in retained_indices
                        else None
                    ),
                )
            )
        return tuple(result)

    def plan(*call_args: object, **call_kwargs: object):  # noqa: ANN202
        result = original_plan(*call_args, **call_kwargs)
        graph = cross_loop_scheduler.build_readiness_graph(
            dependency_graph=call_kwargs["dependency_graph"],
            root_task_orders=call_kwargs["root_task_orders"],
            site_domains=call_kwargs["site_domains"],
            publishable_site_ids=call_kwargs.get("publishable_site_ids"),
            prove_nonnegative=call_kwargs.get("prove_nonnegative"),
        )
        progress_safe = cross_loop_scheduler._schedule_is_progress_safe(
            result.worker_schedule,
            graph,
            result.readiness_counters,
            result.root_barrier_edges,
        )
        records.append(
            {
                "progress_safe": progress_safe,
                "barriers": sorted(result.root_barrier_edges),
                "counters": [
                    {
                        "producers": [p.producer_root for p in counter.producers],
                        "consumers": [c.consumer_root for c in counter.consumers],
                        "continuation": counter.continuation_consumer_index,
                        "keys": counter.readiness_key_count,
                        "fan_in": counter.uniform_arrival_count(),
                    }
                    for counter in result.readiness_counters
                ],
                "segments": [
                    {
                        "root": segment.root,
                        "task_count": segment.task_count,
                        "worker_begin": segment.worker_begin,
                        "worker_count": segment.worker_count,
                        "dispatch_offset": segment.dispatch_offset,
                    }
                    for segment in result.worker_schedule.segments
                ],
            }
        )
        return result

    cross_loop_scheduler.choose_final_arrival_continuations = continuations
    cross_loop_scheduler.choose_readiness_counters = counters
    cross_loop_scheduler._global_unit_list_schedule = lambda *args, **kwargs: None
    cross_loop_codegen.build_static_pipeline_plan = plan
    try:
        bound._compile_cache.clear()
        compiled = bound.compile_config(config)
        cache_path = next(iter(bound._cache_path_map.values()))
        if cache_path is None:
            raise RuntimeError("compiled kernel has no generated source")
        code = Path(cache_path).read_bytes()
    finally:
        cross_loop_scheduler.choose_final_arrival_continuations = (
            original_continuations
        )
        cross_loop_scheduler.choose_readiness_counters = original_counters
        cross_loop_scheduler._global_unit_list_schedule = original_global
        cross_loop_codegen.build_static_pipeline_plan = original_plan
    if len(records) != 1:
        raise RuntimeError(f"{name} captured {len(records)} plans")
    records[0]["generated_source_sha256"] = hashlib.sha256(code).hexdigest()
    return compiled, records[0]


def _make_inputs(
    base: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], tuple[object, ...], Callable[[], None]]:
    # Clone one common tensor set.  ``_make_inputs`` intentionally leaves the
    # rotary cache and compiler-owned intermediates uninitialized because the
    # production benchmark fills them through vLLM.  Recreating it per variant
    # therefore does not produce a valid cross-variant correctness reference.
    tensors = {name: value.clone() for name, value in base.items()}
    args = qwen._kernel_args(tensors)
    return tensors, args, qwen._make_reset(tensors)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=80)
    parser.add_argument("--warmup-ms", type=int, default=10_000)
    options = parser.parse_args()

    compiled: dict[str, CompiledConfig] = {}
    plans: dict[str, dict[str, object]] = {}
    inputs: dict[
        str,
        tuple[dict[str, torch.Tensor], tuple[object, ...], Callable[[], None]],
    ] = {}
    outputs: dict[str, tuple[torch.Tensor, ...]] = {}
    captures: dict[str, tuple[torch.cuda.CUDAGraph, Callable[[], None]]] = {}
    base = qwen._make_inputs()

    for name in VARIANTS:
        tensors, args, reset = _make_inputs(base)
        call, plan = _compile_variant(args, name=name)
        reset()
        output = call(*args)
        torch.cuda.synchronize()
        compiled[name] = call
        plans[name] = plan
        inputs[name] = (tensors, args, reset)
        outputs[name] = tuple(value.clone() for value in output)
        print(
            "COMPILED",
            name,
            json.dumps(
                {
                    "barriers": plan["barriers"],
                    "counters": plan["counters"],
                    "segments": len(plan["segments"]),
                }
            ),
            flush=True,
        )

    reference = outputs["current"]
    reference_cache = qwen._cache_slot(inputs["current"][0]).clone()
    correctness: dict[str, dict[str, float | bool]] = {}
    for name, output in outputs.items():
        output_max_abs = float((output[0].float() - reference[0].float()).abs().max())
        residual_max_abs = float(
            (output[-1].float() - reference[-1].float()).abs().max()
        )
        cache = qwen._cache_slot(inputs[name][0])
        cache_max_abs = float((cache.float() - reference_cache.float()).abs().max())
        close = True
        try:
            torch.testing.assert_close(output[0], reference[0], atol=0.25, rtol=0.05)
            torch.testing.assert_close(
                output[-1], reference[-1], atol=0.25, rtol=0.05
            )
            torch.testing.assert_close(cache, reference_cache, atol=0.125, rtol=0.03)
        except AssertionError:
            close = False
        correctness[name] = {
            "production_tolerance": close,
            "output_max_abs": output_max_abs,
            "residual_max_abs": residual_max_abs,
            "cache_max_abs": cache_max_abs,
        }
        print("CORRECTNESS", name, correctness[name], flush=True)
        if not close:
            continue
        _tensors, args, reset = inputs[name]
        graph, _ = capture_cuda_graph(
            lambda call=compiled[name], call_args=args: call(*call_args),
            reset,
        )
        captures[name] = (graph, reset)

    thermal_warmup(options.warmup_ms)
    isolated: dict[str, float] = {}
    for name, (graph, reset) in captures.items():
        isolated[name] = (
            bench_pre_captured_cudagraphs(
                [graph.replay],
                rep=options.repetitions,
                resets=[reset],
            )[0]
            * 1000
        )

    pairwise: dict[str, dict[str, float]] = {}
    current_graph, current_reset = captures["current"]
    for name, (graph, reset) in captures.items():
        if name == "current":
            continue
        elapsed = bench_pre_captured_cudagraphs(
            [current_graph.replay, graph.replay],
            rep=options.repetitions,
            resets=[current_reset, reset],
        )
        pairwise[name] = {
            "current": elapsed[0] * 1000,
            "variant": elapsed[1] * 1000,
        }

    resources: dict[str, dict[str, object]] = {}
    for name, call in compiled.items():
        kernels = _compiled_kernels(call)
        kernel = max(kernels, key=lambda item: int(item.metadata.shared))
        resources[name] = {
            "registers": int(kernel.n_regs),
            "spills": int(kernel.n_spills),
            "shared": int(kernel.metadata.shared),
            "warps": int(kernel.metadata.num_warps),
            "cubin_sha256": hashlib.sha256(kernel.asm["cubin"]).hexdigest(),
        }

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "source": "untouched pretuned Qwen3 B1/S8192/Q1",
                "config": CONFIG,
                "variants": {
                    name: {
                        "disabled_continuation_roots": sorted(spec[0]),
                        "disabled_counter_edges": [list(edge) for edge in spec[1]],
                    }
                    for name, spec in VARIANTS.items()
                },
                "correctness": correctness,
                "isolated_us_cold_l2": isolated,
                "pairwise_with_current_us_cold_l2": pairwise,
                "resources": resources,
                "plans": plans,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
