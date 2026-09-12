"""Dynamic-batch scheduler validation for the pretuned Gemma 4 A4B MoE.

The dynamic and static persistent variants are derived from the same checked-in
pretuned source.  The standalone control is the same eight-root Helion boundary
used by ``gemma4_a4b_moe_batched.py``.  Only the batch dimension is allowed to
vary; fusion, arithmetic, tensor dtypes, routing, and kernel boundaries stay
unchanged.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from itertools import starmap
import json
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cross_loop_scheduling import gemma4_a4b_moe_batched as base
from pretuned_kernels._bench import bench_pre_captured_cudagraphs
from pretuned_kernels._bench import capture_cuda_graph
from pretuned_kernels._bench import thermal_warmup
from pretuned_kernels.megakernels.gemma4_a4b_moe._helion_aot_gemma4_a4b_moe_cuda_sm100 import (
    CONFIG,
)
import torch

import helion
from helion._compiler import cross_loop_codegen
from helion._compiler import cross_loop_scheduler

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.kernel import CompiledConfig
    from helion.runtime.kernel import Kernel


def _compile(
    kernel: Kernel,
    args: tuple[object, ...],
    *,
    multiplier: int,
) -> tuple[CompiledConfig, dict[str, object], str, dict[str, object]]:
    bound = kernel.bind(args)
    values = deepcopy(CONFIG)
    values["num_sm_multiplier"] = multiplier
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)

    path_records: dict[str, object] = {}
    original_baseline = cross_loop_scheduler.build_baseline_worker_schedule
    original_event_frontier = cross_loop_scheduler._event_frontier_list_schedule
    original_global = cross_loop_scheduler._global_unit_list_schedule
    original_pipeline_plan = cross_loop_codegen.build_static_pipeline_plan

    def baseline(*call_args: object, **call_kwargs: object):  # noqa: ANN202
        result = original_baseline(*call_args, **call_kwargs)
        path_records["baseline_worker_schedule"] = {
            "called": True,
            "segments": len(result.segments),
            "roots": [segment.root for segment in result.segments],
        }
        return result

    def event_frontier(*call_args: object, **call_kwargs: object):  # noqa: ANN202
        result = original_event_frontier(*call_args, **call_kwargs)
        path_records["event_frontier_proposal"] = {
            "called": True,
            "returned_schedule": result is not None,
            "changed": result is not None and result != call_args[1],
            "segments": None if result is None else len(result.segments),
            "roots": (
                None
                if result is None
                else [segment.root for segment in result.segments]
            ),
        }
        return result

    def global_list(*call_args: object, **call_kwargs: object):  # noqa: ANN202
        result = original_global(*call_args, **call_kwargs)
        path_records["global_list_schedule"] = {
            "called": True,
            "returned_schedule": result is not None,
            "changed": result is not None and result != call_args[1],
            "segments": None if result is None else len(result.segments),
        }
        return result

    def pipeline_plan(*call_args: object, **call_kwargs: object):  # noqa: ANN202
        result = original_pipeline_plan(*call_args, **call_kwargs)
        path_records["static_pipeline_plan"] = {
            "readiness_counters": [
                {
                    "key_axes": list(plan.readiness_key_domain.axis_order),
                    "key_counts": [
                        [axis, str(count)]
                        for axis, count in (plan.readiness_key_domain.axis_counts_items)
                    ],
                    "key_count": str(plan.readiness_key_count_expr),
                    "fan_in": plan.uniform_arrival_count(),
                    "producer_roots": [
                        producer.producer_root for producer in plan.producers
                    ],
                    "consumer_roots": [
                        consumer.consumer_root for consumer in plan.consumers
                    ],
                    "continuation_consumer_index": (plan.continuation_consumer_index),
                }
                for plan in result.readiness_counters
            ],
            "root_barrier_edges": sorted(result.root_barrier_edges),
            "worker_schedule_segments": [
                {
                    "root": segment.root,
                    "worker_begin": segment.worker_begin,
                    "worker_count": segment.worker_count,
                    "dispatch_offset": segment.dispatch_offset,
                    "target_counts": [
                        [axis, str(count)]
                        for axis, count in (
                            segment.task_order.target_domain.axis_counts_items
                        )
                    ],
                }
                for segment in result.worker_schedule.segments
            ],
        }
        return result

    cross_loop_scheduler.build_baseline_worker_schedule = baseline
    cross_loop_scheduler._event_frontier_list_schedule = event_frontier
    cross_loop_scheduler._global_unit_list_schedule = global_list
    cross_loop_codegen.build_static_pipeline_plan = pipeline_plan
    try:
        compile_start = time.perf_counter()
        compiled = bound.compile_config(config)
        path_records["compile_seconds"] = time.perf_counter() - compile_start
        cache_path = next(iter(bound._cache_path_map.values()))
        if cache_path is None:
            raise RuntimeError("compiled Gemma kernel has no generated source path")
        code = Path(cache_path).read_text()
        path_records["generated_source_path"] = str(cache_path)
        path_records["generated_source_sha256"] = hashlib.sha256(
            code.encode()
        ).hexdigest()
    finally:
        cross_loop_scheduler.build_baseline_worker_schedule = original_baseline
        cross_loop_scheduler._event_frontier_list_schedule = original_event_frontier
        cross_loop_scheduler._global_unit_list_schedule = original_global
        cross_loop_codegen.build_static_pipeline_plan = original_pipeline_plan
    return compiled, values, code, path_records


def _compiled_kernels(call: Callable[..., object]) -> tuple[Any, ...]:
    wrappers: list[Callable[..., object]] = [call]
    for cell in call.__closure__ or ():
        value = cell.cell_contents
        if isinstance(value, dict):
            referenced_keys = {
                constant
                for constant in call.__code__.co_consts
                if isinstance(constant, str) and constant in value
            }
            selected = (
                (value[next(iter(referenced_keys))],)
                if len(referenced_keys) == 1
                else tuple(value.values())
            )
            wrappers.extend(item for item in selected if callable(item))
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


def _binary_summary(call: Callable[..., object]) -> dict[str, object]:
    kernels = _compiled_kernels(call)
    return {
        "specialization_count": len(kernels),
        "cubin_hashes": sorted(
            str(kernel.hash)
            for kernel in kernels
            if getattr(kernel, "hash", None) is not None
        ),
        "cubin_sha256": sorted(
            hashlib.sha256(kernel.asm["cubin"]).hexdigest()
            for kernel in kernels
            if isinstance(getattr(kernel, "asm", None), dict)
            and isinstance(kernel.asm.get("cubin"), bytes)
        ),
    }


def _resources(call: Callable[..., object]) -> dict[str, int]:
    kernels = _compiled_kernels(call)
    if not kernels:
        return {}
    compiled = max(kernels, key=lambda item: int(item.metadata.shared))
    return {
        "registers_per_thread": int(compiled.n_regs),
        "spills": int(compiled.n_spills),
        "shared_bytes": int(compiled.metadata.shared),
        "num_warps": int(compiled.metadata.num_warps),
    }


def _ptx_synchronization_summary(call: Callable[..., object]) -> dict[str, object]:
    kernels = _compiled_kernels(call)
    if not kernels:
        return {}
    compiled = max(kernels, key=lambda item: int(item.metadata.shared))
    ptx = compiled.asm.get("ptx", "")
    if isinstance(ptx, bytes):
        ptx = ptx.decode()
    lines = tuple(line.strip() for line in str(ptx).splitlines())
    return {
        "ptx_sha256": hashlib.sha256(str(ptx).encode()).hexdigest(),
        "atomic_u64_instructions": sum(
            "atom." in line and ".u64" in line for line in lines
        ),
        "atomic_u32_instructions": sum(
            "atom." in line and ".u32" in line for line in lines
        ),
        "acquire_load_u64_instructions": sum(
            "ld.acquire" in line and ".u64" in line for line in lines
        ),
        "acquire_load_u32_instructions": sum(
            "ld.acquire" in line and ".u32" in line for line in lines
        ),
        "release_store_u64_instructions": sum(
            "st.release" in line and ".u64" in line for line in lines
        ),
        "release_store_u32_instructions": sum(
            "st.release" in line and ".u32" in line for line in lines
        ),
        "barrier_instructions": sum("barrier." in line for line in lines),
        "membar_instructions": sum("membar." in line for line in lines),
    }


def _lowering_summary(code: str) -> dict[str, object]:
    return {
        "parameterized_root_loop": ("tile_dependency_parameterized_root_task" in code),
        "event_frontier_loop_count": code.count(
            "for tile_dependency_event_frontier_task in tl.range"
        ),
        "readiness_wait": "tile_dependency_readiness_wait" in code,
        "readiness_atomic_xchg": "tl.atomic_xchg" in code,
        "readiness_atomic_add": "tl.atomic_add" in code,
        "root_barrier": "tile_dependency_root_barrier" in code,
        "grid_barrier": "triton_helpers.x_grid_barrier(" in code,
        "dispatch_ticket": "tile_dependency_dispatch_ticket" in code,
    }


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    return float((left.float() - right.float()).abs().max())


@torch.inference_mode()
def benchmark(args: argparse.Namespace) -> dict[str, object]:
    base.gemma._require_sm100()
    batches = tuple(int(value) for value in args.batch_sizes.split(","))
    if not batches or any(batch <= 0 for batch in batches):
        raise ValueError("batch sizes must be positive")
    if args.exemplar_batch not in batches:
        raise ValueError("exemplar batch must be one of batch sizes")
    static_multiplier = (
        args.multiplier if args.static_multiplier is None else args.static_multiplier
    )

    tensors_by_batch = base._make_batch_series_inputs(batches, args.seed)
    args_by_batch = {
        batch: base.gemma._kernel_args(tensors_by_batch[batch]) for batch in batches
    }
    dynamic_kernel = base._make_persistent_kernel(dynamic_batch=True)
    dynamic_call, dynamic_config, dynamic_code, dynamic_path = _compile(
        dynamic_kernel,
        args_by_batch[args.exemplar_batch],
        multiplier=args.multiplier,
    )
    dynamic_call(*args_by_batch[args.exemplar_batch])
    torch.cuda.synchronize()
    static_calls: dict[int, CompiledConfig] = {}
    static_configs: dict[int, dict[str, object]] = {}
    static_paths: dict[int, dict[str, object]] = {}
    static_lowering: dict[int, dict[str, object]] = {}
    standalone_configs: dict[int, dict[str, object]] = {}
    standalone_resources: dict[int, dict[str, dict[str, int]]] = {}
    routing: dict[int, dict[str, object]] = {}
    correctness: dict[int, dict[str, object]] = {}
    captures: dict[int, tuple[tuple[str, torch.cuda.CUDAGraph, object], ...]] = {}
    expected_dynamic_binary = _binary_summary(dynamic_call)

    tolerances = (
        (0.15, 0.06),
        (0.05, 0.02),
        (2e-5, 2e-5),
        (0, 0),
        (0.2, 0.08),
        (0.25, 0.1),
        (0.25, 0.1),
    )
    for batch in batches:
        tensors = tensors_by_batch[batch]
        kernel_args = args_by_batch[batch]
        static_kernel = base._make_persistent_kernel(dynamic_batch=False)
        static_call, static_config, static_code, static_path = _compile(
            static_kernel,
            kernel_args,
            multiplier=static_multiplier,
        )
        static_calls[batch] = static_call
        static_configs[batch] = static_config
        static_paths[batch] = static_path
        static_lowering[batch] = _lowering_summary(static_code)

        standalone, configs, stage_calls = base._standalone_pipeline(
            tensors,
            config_mode="matched",
        )
        standalone_configs[batch] = configs
        expected = standalone()
        dynamic = dynamic_call(*kernel_args)
        static = static_call(*kernel_args)
        torch.cuda.synchronize()
        dynamic_errors: list[float] = []
        static_errors: list[float] = []
        for dynamic_value, static_value, expected_value, (atol, rtol) in zip(
            dynamic,
            static,
            expected,
            tolerances,
            strict=True,
        ):
            torch.testing.assert_close(
                dynamic_value, expected_value, atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                static_value, expected_value, atol=atol, rtol=rtol
            )
            dynamic_errors.append(_max_abs(dynamic_value, expected_value))
            static_errors.append(_max_abs(static_value, expected_value))
        correctness[batch] = {
            "dynamic_vs_standalone_max_abs_by_output": dynamic_errors,
            "static_vs_standalone_max_abs_by_output": static_errors,
            "dynamic_vs_static_bit_exact": all(
                starmap(torch.equal, zip(dynamic, static, strict=True))
            ),
        }
        routing[batch] = base._routing_summary(expected[3])

        current_binary = _binary_summary(dynamic_call)
        if current_binary != expected_dynamic_binary:
            raise AssertionError(
                f"dynamic B{batch} created a new binary: {current_binary}"
            )
        calls: list[tuple[str, Callable[[], object]]] = [
            ("standalone_eight_launch", standalone),
            (
                "persistent_static",
                lambda call=static_call, call_args=kernel_args: call(*call_args),
            ),
            (
                "persistent_dynamic",
                lambda call_args=kernel_args: dynamic_call(*call_args),
            ),
        ]
        captures[batch] = tuple(
            (name, *capture_cuda_graph(call)) for name, call in calls
        )
        standalone_resources[batch] = {
            name: _resources(call) for name, call in stage_calls.items()
        }

    thermal_warmup(args.warmup_ms)
    timings: dict[int, dict[str, float]] = {}
    for batch, batch_captures in captures.items():
        elapsed = bench_pre_captured_cudagraphs(
            [graph.replay for _name, graph, _output in batch_captures],
            rep=args.repetitions,
        )
        timings[batch] = {
            name: milliseconds * 1000
            for (name, _graph, _output), milliseconds in zip(
                batch_captures, elapsed, strict=True
            )
        }

    dynamic_binary = _binary_summary(dynamic_call)
    if dynamic_binary["specialization_count"] != 1:
        raise AssertionError(
            f"dynamic kernel did not retain one specialization: {dynamic_binary}"
        )
    return {
        "workload": "Gemma 4 26B-A4B MoE, matched eight-root boundary",
        "batch_sizes": batches,
        "exemplar_batch": args.exemplar_batch,
        "dynamic_multiplier": args.multiplier,
        "static_multiplier": static_multiplier,
        "device": torch.cuda.get_device_name(),
        "timings_us_cold_l2": timings,
        "correctness": correctness,
        "routing": routing,
        "one_dynamic_cubin_across_batch_sizes": True,
        "dynamic_binary": dynamic_binary,
        "dynamic_lowering": _lowering_summary(dynamic_code),
        "dynamic_ptx_synchronization": _ptx_synchronization_summary(dynamic_call),
        "dynamic_scheduler_path": dynamic_path,
        "static_lowering": static_lowering,
        "static_scheduler_path": static_paths,
        "resources": {
            "persistent_dynamic": _resources(dynamic_call),
            "persistent_static": {
                batch: _resources(call) for batch, call in static_calls.items()
            },
            "standalone": standalone_resources,
        },
        "configs": {
            "persistent_dynamic": dynamic_config,
            "persistent_static": static_configs,
            "standalone": standalone_configs,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,2")
    parser.add_argument("--exemplar-batch", type=int, default=2)
    parser.add_argument("--multiplier", type=int, default=2)
    parser.add_argument(
        "--static-multiplier",
        type=int,
        default=3,
        help="static-control multiplier",
    )
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmup-ms", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    print(json.dumps(benchmark(parser.parse_args()), indent=2, default=str))


if __name__ == "__main__":
    main()
