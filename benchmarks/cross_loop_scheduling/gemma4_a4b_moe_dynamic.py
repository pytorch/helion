"""Dynamic-batch scheduler validation for the pretuned Gemma 4 A4B MoE.

The dynamic and static persistent variants are derived from the same checked-in
pretuned source.  The standalone control is the same eight-root Helion boundary
used by ``gemma4_a4b_moe_batched.py``.  Only the batch dimension is allowed to
vary; fusion, arithmetic, tensor dtypes, routing, and kernel boundaries stay
unchanged.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import dataclasses
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
    root6_continuation_ablation: bool = False,
    root7_resident_ablation: bool = False,
    u32_epoch_ablation: bool = False,
    disable_root6_continuation: bool = False,
    static_packed_root_major: bool = False,
) -> tuple[CompiledConfig, dict[str, object], str, dict[str, object]]:
    if static_packed_root_major and not disable_root6_continuation:
        raise ValueError("static packed control requires resident root 6")
    if root7_resident_ablation and not root6_continuation_ablation:
        raise ValueError("root 7 can be retained only while forcing root 6")
    bound = kernel.bind(args)
    values = deepcopy(CONFIG)
    values["num_sm_multiplier"] = multiplier
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)

    path_records: dict[str, object] = {}
    original_root_major = (
        cross_loop_scheduler._build_root_major_worker_schedule
    )
    original_event_frontier = (
        cross_loop_scheduler._build_parametric_event_frontier_worker_schedule
    )
    original_global = cross_loop_scheduler._global_unit_list_schedule
    original_pipeline_plan = cross_loop_codegen.build_static_pipeline_plan
    original_emit = cross_loop_codegen.emit_cross_loop_schedule
    original_choose_continuations = (
        cross_loop_scheduler.choose_final_arrival_continuations
    )
    original_parametric_dtype = (
        cross_loop_codegen._PARAMETRIC_READINESS_COUNTER_DTYPE
    )
    original_parametric_alignment = (
        cross_loop_codegen._PARAMETRIC_READINESS_COUNTER_ALIGNMENT_WORDS
    )

    def root_major(*call_args: object, **call_kwargs: object):  # noqa: ANN202
        result = original_root_major(*call_args, **call_kwargs)
        path_records["parametric_root_major"] = {
            "called": True,
            "segments": len(result.segments),
            "roots": [segment.root for segment in result.segments],
        }
        return result

    def event_frontier(*call_args: object, **call_kwargs: object):  # noqa: ANN202
        result = original_event_frontier(*call_args, **call_kwargs)
        path_records["parametric_event_frontier"] = {
            "called": True,
            "accepted": result is not None,
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
        path_records["static_global_list"] = {
            "called": True,
            "accepted": result is not None,
            "segments": None if result is None else len(result.segments),
        }
        return result

    def choose_continuations(*call_args: object, **call_kwargs: object):  # noqa: ANN202
        result = original_choose_continuations(*call_args, **call_kwargs)
        readiness_graph = call_args[0]
        if not disable_root6_continuation:
            return result
        filtered = tuple(
            continuation
            for continuation in result
            if readiness_graph.event(continuation.event_id)
            .consumers[continuation.consumer_index]
            .consumer_root
            != 6
        )
        path_records["diagnostic_disabled_root6_continuation"] = {
            "candidate_count": len(result),
            "retained_count": len(filtered),
        }
        return filtered

    def pipeline_plan(*call_args: object, **call_kwargs: object):  # noqa: ANN202
        result = original_pipeline_plan(*call_args, **call_kwargs)
        root_task_orders = call_kwargs["root_task_orders"]
        if static_packed_root_major and not any(
            task_order.target_domain.parameter_symbols
            for task_order in root_task_orders
        ):
            root_domains = tuple(
                task_order.target_domain for task_order in root_task_orders
            )
            continuation_roots = frozenset(
                consumer.consumer_root
                for candidate in result.readiness_counters
                if (consumer := candidate.continuation_consumer) is not None
            )
            scheduled_roots = tuple(
                (root, domain)
                for root, domain in enumerate(root_domains)
                if root not in continuation_roots
            )
            first_slot = 0
            segments = []
            for root, domain in scheduled_roots:
                task_count = domain.size
                source_begin = 0
                while source_begin < task_count:
                    global_slot = first_slot + source_begin
                    worker_begin = global_slot % call_kwargs["worker_count"]
                    run_count = min(
                        task_count - source_begin,
                        call_kwargs["worker_count"] - worker_begin,
                    )
                    task_order = cross_loop_scheduler._task_order_slice(
                        root_task_orders[root],
                        source_begin,
                        run_count,
                    )
                    if task_order is None:
                        raise AssertionError(
                            "packed static task-order slice is not representable"
                        )
                    worker_step = global_slot // call_kwargs["worker_count"]
                    segments.append(
                        cross_loop_scheduler.WorkerScheduleSegment(
                            root=root,
                            task_order=task_order,
                            worker_begin=worker_begin,
                            worker_count=run_count,
                            dispatch_offset=worker_step * run_count,
                        )
                    )
                    source_begin += run_count
                first_slot += task_count
            result = dataclasses.replace(
                result,
                worker_schedule=cross_loop_scheduler.WorkerSchedule(
                    worker_count=call_kwargs["worker_count"],
                    segments=tuple(segments),
                ),
            )
            path_records["diagnostic_static_packed_root_major"] = {
                "resident_roots": [root for root, _domain in scheduled_roots],
                "excluded_roots": sorted(continuation_roots),
                "total_slots": str(first_slot),
            }
        if root6_continuation_ablation and any(
            task_order.target_domain.parameter_symbols
            for task_order in root_task_orders
        ):
            matches = tuple(
                (plan_index, plan, consumer_index)
                for plan_index, plan in enumerate(result.readiness_counters)
                if tuple(
                    producer.producer_root for producer in plan.producers
                )
                == (5,)
                for consumer_index, consumer in enumerate(plan.consumers)
                if consumer.consumer_root == 6
            )
            if len(matches) != 1:
                raise AssertionError(
                    "root-6 continuation ablation requires one exact 5->6 counter"
                )
            plan_index, plan, consumer_index = matches[0]
            if plan.continuation_consumer_index is not None:
                raise AssertionError("root 6 is already a continuation")
            forced = dataclasses.replace(
                plan,
                continuation_consumer_index=consumer_index,
            )
            if not cross_loop_scheduler._supports_parameterized_counter(forced):
                raise AssertionError(
                    "root-6 continuation does not satisfy the parameterized "
                    "counter certificate"
                )
            readiness_counters = tuple(
                forced if index == plan_index else candidate
                for index, candidate in enumerate(result.readiness_counters)
            )
            root6_to_root7 = tuple(
                (candidate_index, candidate, next_consumer_index)
                for candidate_index, candidate in enumerate(readiness_counters)
                if tuple(
                    producer.producer_root for producer in candidate.producers
                )
                == (6,)
                for next_consumer_index, consumer in enumerate(candidate.consumers)
                if consumer.consumer_root == 7
            )
            if len(root6_to_root7) != 1:
                raise AssertionError(
                    "root-6 continuation ablation requires one exact 6->7 counter"
                )
            next_plan_index, next_plan, next_consumer_index = root6_to_root7[0]
            if next_plan.continuation_consumer_index != next_consumer_index:
                raise AssertionError("the production plan did not continue 6->7")
            if root7_resident_ablation:
                readiness_counters = tuple(
                    dataclasses.replace(candidate, continuation_consumer_index=None)
                    if index == next_plan_index
                    else candidate
                    for index, candidate in enumerate(readiness_counters)
                )
            if not cross_loop_scheduler._parameterized_prerequisites_follow_root_order(
                readiness_counters,
                result.root_barrier_edges,
            ):
                raise AssertionError(
                    "root-6 continuation ablation broke root-order progress"
                )
            continuation_roots = frozenset(
                consumer.consumer_root
                for candidate in readiness_counters
                if (consumer := candidate.continuation_consumer) is not None
            )
            root_domains = tuple(
                task_order.target_domain for task_order in root_task_orders
            )
            worker_schedule = cross_loop_scheduler._build_root_major_worker_schedule(
                root_domains,
                root_task_orders,
                call_kwargs["worker_count"],
                excluded_roots=continuation_roots,
            )
            result = dataclasses.replace(
                result,
                worker_schedule=worker_schedule,
                readiness_counters=readiness_counters,
            )
            path_records["diagnostic_root6_continuation"] = {
                "forced_edge": [5, 6],
                "root7_resident": root7_resident_ablation,
                "chained_edge": None if root7_resident_ablation else [6, 7],
                "excluded_roots": sorted(continuation_roots),
            }
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

    class UInt64EpochToUInt32(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:  # noqa: N802
            node = self.generic_visit(node)
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "tl"
                and node.attr == "uint64"
            ):
                node.attr = "uint32"
            return node

    def emit(*call_args: object, **call_kwargs: object):  # noqa: ANN202
        result = original_emit(*call_args, **call_kwargs)
        if not u32_epoch_ablation:
            return result
        transformer = UInt64EpochToUInt32()
        transformed = [transformer.visit(statement) for statement in result]
        for statement in transformed:
            ast.fix_missing_locations(statement)
        path_records["diagnostic_u32_epoch"] = {
            "epoch_dtype": "uint32",
            "counter_alignment_bytes": (
                cross_loop_codegen._CROSS_LOOP_COUNTER_ALIGNMENT_BYTES
            ),
            "counter_alignment_words": (
                cross_loop_codegen._PARAMETRIC_READINESS_COUNTER_ALIGNMENT_WORDS
            ),
            "bounded_replay_only": True,
        }
        return transformed

    cross_loop_scheduler._build_root_major_worker_schedule = root_major
    cross_loop_scheduler._build_parametric_event_frontier_worker_schedule = (
        event_frontier
    )
    cross_loop_scheduler._global_unit_list_schedule = global_list
    cross_loop_scheduler.choose_final_arrival_continuations = choose_continuations
    cross_loop_codegen.build_static_pipeline_plan = pipeline_plan
    cross_loop_codegen.emit_cross_loop_schedule = emit
    if u32_epoch_ablation:
        cross_loop_codegen._PARAMETRIC_READINESS_COUNTER_DTYPE = torch.uint32
        cross_loop_codegen._PARAMETRIC_READINESS_COUNTER_ALIGNMENT_WORDS = (
            cross_loop_codegen._CROSS_LOOP_COUNTER_ALIGNMENT_WORDS
        )
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
        cross_loop_scheduler._build_root_major_worker_schedule = (
            original_root_major
        )
        cross_loop_scheduler._build_parametric_event_frontier_worker_schedule = (
            original_event_frontier
        )
        cross_loop_scheduler._global_unit_list_schedule = original_global
        cross_loop_scheduler.choose_final_arrival_continuations = (
            original_choose_continuations
        )
        cross_loop_codegen.build_static_pipeline_plan = original_pipeline_plan
        cross_loop_codegen.emit_cross_loop_schedule = original_emit
        cross_loop_codegen._PARAMETRIC_READINESS_COUNTER_DTYPE = (
            original_parametric_dtype
        )
        cross_loop_codegen._PARAMETRIC_READINESS_COUNTER_ALIGNMENT_WORDS = (
            original_parametric_alignment
        )
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


def _continuation_chain_summary(code: str) -> dict[str, object]:
    tree = ast.parse(code)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    def calls(function_name: str) -> list[str]:
        function = functions.get(function_name)
        if function is None:
            return []
        return sorted(
            {
                node.func.id
                for node in ast.walk(function)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
        )

    root5_scheduled = next(
        (
            name
            for name in functions
            if name.startswith("tile_dependency_root_5_scheduled_task")
        ),
        "",
    )
    root6 = next(
        (
            name
            for name in functions
            if name.startswith("tile_dependency_root_6")
            and "scheduled_task" not in name
        ),
        "",
    )
    entry = next(
        (name for name in functions if name.startswith("_helion_gemma4_a4b_moe")),
        "",
    )
    root5_calls = calls(root5_scheduled)
    entry_calls = calls(entry)
    return {
        "root5_scheduled_function": root5_scheduled or None,
        "root5_calls_root6": any(
            name.startswith("tile_dependency_root_6") for name in root5_calls
        ),
        "root6_function": root6 or None,
        # Root 6's publication and the nested root-7 continuation are emitted
        # in the owning root-5 scheduled helper after its inline root-6 call.
        "root5_chains_root7": any(
            name.startswith("tile_dependency_root_7") for name in root5_calls
        ),
        "resident_root6_call": any(
            name.startswith("tile_dependency_root_6") for name in entry_calls
        ),
        "resident_root7_call": any(
            name.startswith("tile_dependency_root_7") for name in entry_calls
        ),
        "readiness_wait_count": code.count("tile_dependency_readiness_wait"),
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
    root6_continuation_ablation = bool(
        getattr(args, "root6_continuation_ablation", False)
    )
    ablation_call: CompiledConfig | None = None
    ablation_config: dict[str, object] | None = None
    ablation_code: str | None = None
    ablation_path: dict[str, object] | None = None
    root6_only_call: CompiledConfig | None = None
    root6_only_config: dict[str, object] | None = None
    root6_only_code: str | None = None
    root6_only_path: dict[str, object] | None = None
    if root6_continuation_ablation:
        ablation_kernel = base._make_persistent_kernel(dynamic_batch=True)
        (
            ablation_call,
            ablation_config,
            ablation_code,
            ablation_path,
        ) = _compile(
            ablation_kernel,
            args_by_batch[args.exemplar_batch],
            multiplier=args.multiplier,
            root6_continuation_ablation=True,
        )
        ablation_call(*args_by_batch[args.exemplar_batch])
        torch.cuda.synchronize()
        root6_only_kernel = base._make_persistent_kernel(dynamic_batch=True)
        (
            root6_only_call,
            root6_only_config,
            root6_only_code,
            root6_only_path,
        ) = _compile(
            root6_only_kernel,
            args_by_batch[args.exemplar_batch],
            multiplier=args.multiplier,
            root6_continuation_ablation=True,
            root7_resident_ablation=True,
        )
        root6_only_call(*args_by_batch[args.exemplar_batch])
        torch.cuda.synchronize()

    u32_epoch_ablation = bool(getattr(args, "u32_epoch_ablation", False))
    u32_call: CompiledConfig | None = None
    u32_config: dict[str, object] | None = None
    u32_code: str | None = None
    u32_path: dict[str, object] | None = None
    if u32_epoch_ablation:
        u32_kernel = base._make_persistent_kernel(dynamic_batch=True)
        u32_call, u32_config, u32_code, u32_path = _compile(
            u32_kernel,
            args_by_batch[args.exemplar_batch],
            multiplier=args.multiplier,
            u32_epoch_ablation=True,
        )
        u32_call(*args_by_batch[args.exemplar_batch])
        torch.cuda.synchronize()

    static_calls: dict[int, CompiledConfig] = {}
    static_configs: dict[int, dict[str, object]] = {}
    static_paths: dict[int, dict[str, object]] = {}
    static_lowering: dict[int, dict[str, object]] = {}
    standalone_configs: dict[int, dict[str, object]] = {}
    standalone_resources: dict[int, dict[str, dict[str, int]]] = {}
    routing: dict[int, dict[str, object]] = {}
    correctness: dict[int, dict[str, object]] = {}
    ablation_correctness: dict[int, dict[str, object]] = {}
    root6_only_correctness: dict[int, dict[str, object]] = {}
    u32_correctness: dict[int, dict[str, object]] = {}
    captures: dict[int, tuple[tuple[str, torch.cuda.CUDAGraph, object], ...]] = {}
    expected_dynamic_binary = _binary_summary(dynamic_call)
    expected_ablation_binary = (
        _binary_summary(ablation_call) if ablation_call is not None else None
    )
    expected_root6_only_binary = (
        _binary_summary(root6_only_call) if root6_only_call is not None else None
    )
    expected_u32_binary = (
        _binary_summary(u32_call) if u32_call is not None else None
    )

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
        ablation = (
            ablation_call(*kernel_args) if ablation_call is not None else None
        )
        root6_only = (
            root6_only_call(*kernel_args)
            if root6_only_call is not None
            else None
        )
        u32 = u32_call(*kernel_args) if u32_call is not None else None
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
        if ablation is not None:
            ablation_errors: list[float] = []
            for ablation_value, expected_value, (atol, rtol) in zip(
                ablation,
                expected,
                tolerances,
                strict=True,
            ):
                torch.testing.assert_close(
                    ablation_value,
                    expected_value,
                    atol=atol,
                    rtol=rtol,
                )
                ablation_errors.append(_max_abs(ablation_value, expected_value))
            ablation_correctness[batch] = {
                "vs_standalone_max_abs_by_output": ablation_errors,
                "vs_static_bit_exact": all(
                    starmap(torch.equal, zip(ablation, static, strict=True))
                ),
                "vs_baseline_dynamic_bit_exact": all(
                    starmap(torch.equal, zip(ablation, dynamic, strict=True))
                ),
            }
        if root6_only is not None:
            root6_only_errors: list[float] = []
            for value, expected_value, (atol, rtol) in zip(
                root6_only,
                expected,
                tolerances,
                strict=True,
            ):
                torch.testing.assert_close(
                    value,
                    expected_value,
                    atol=atol,
                    rtol=rtol,
                )
                root6_only_errors.append(_max_abs(value, expected_value))
            root6_only_correctness[batch] = {
                "vs_standalone_max_abs_by_output": root6_only_errors,
                "vs_static_bit_exact": all(
                    starmap(torch.equal, zip(root6_only, static, strict=True))
                ),
                "vs_baseline_dynamic_bit_exact": all(
                    starmap(torch.equal, zip(root6_only, dynamic, strict=True))
                ),
            }
        if u32 is not None:
            u32_errors: list[float] = []
            for value, expected_value, (atol, rtol) in zip(
                u32,
                expected,
                tolerances,
                strict=True,
            ):
                torch.testing.assert_close(
                    value,
                    expected_value,
                    atol=atol,
                    rtol=rtol,
                )
                u32_errors.append(_max_abs(value, expected_value))
            u32_correctness[batch] = {
                "vs_standalone_max_abs_by_output": u32_errors,
                "vs_static_bit_exact": all(
                    starmap(torch.equal, zip(u32, static, strict=True))
                ),
                "vs_baseline_dynamic_bit_exact": all(
                    starmap(torch.equal, zip(u32, dynamic, strict=True))
                ),
            }
        routing[batch] = base._routing_summary(expected[3])

        current_binary = _binary_summary(dynamic_call)
        if current_binary != expected_dynamic_binary:
            raise AssertionError(
                f"dynamic B{batch} created a new binary: {current_binary}"
            )
        if ablation_call is not None:
            current_ablation_binary = _binary_summary(ablation_call)
            if current_ablation_binary != expected_ablation_binary:
                raise AssertionError(
                    f"root-6 continuation B{batch} created a new binary: "
                    f"{current_ablation_binary}"
                )
        if root6_only_call is not None:
            current_root6_only_binary = _binary_summary(root6_only_call)
            if current_root6_only_binary != expected_root6_only_binary:
                raise AssertionError(
                    f"root-6-only continuation B{batch} created a new binary: "
                    f"{current_root6_only_binary}"
                )
        if u32_call is not None:
            current_u32_binary = _binary_summary(u32_call)
            if current_u32_binary != expected_u32_binary:
                raise AssertionError(
                    f"u32 epoch B{batch} created a new binary: "
                    f"{current_u32_binary}"
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
        if ablation_call is not None:
            calls.append(
                (
                    "persistent_dynamic_root6_continuation",
                    lambda call=ablation_call, call_args=kernel_args: call(*call_args),
                )
            )
        if root6_only_call is not None:
            calls.append(
                (
                    "persistent_dynamic_root6_only_continuation",
                    lambda call=root6_only_call, call_args=kernel_args: call(
                        *call_args
                    ),
                )
            )
        if u32_call is not None:
            calls.append(
                (
                    "persistent_dynamic_u32_epoch",
                    lambda call=u32_call, call_args=kernel_args: call(*call_args),
                )
            )
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
    ablation_binary = (
        _binary_summary(ablation_call) if ablation_call is not None else None
    )
    if (
        ablation_binary is not None
        and ablation_binary["specialization_count"] != 1
    ):
        raise AssertionError(
            "root-6 continuation did not retain one specialization: "
            f"{ablation_binary}"
        )
    root6_only_binary = (
        _binary_summary(root6_only_call)
        if root6_only_call is not None
        else None
    )
    if (
        root6_only_binary is not None
        and root6_only_binary["specialization_count"] != 1
    ):
        raise AssertionError(
            "root-6-only continuation did not retain one specialization: "
            f"{root6_only_binary}"
        )
    u32_binary = _binary_summary(u32_call) if u32_call is not None else None
    if u32_binary is not None and u32_binary["specialization_count"] != 1:
        raise AssertionError(
            f"u32 epoch did not retain one specialization: {u32_binary}"
        )
    ablation_summary: dict[str, object] | None = None
    if ablation_call is not None:
        assert ablation_code is not None
        assert ablation_path is not None
        assert ablation_config is not None
        continuation_chain = _continuation_chain_summary(ablation_code)
        if not (
            continuation_chain["root5_scheduled_function"]
            == "tile_dependency_root_5_scheduled_task"
            and continuation_chain["root5_calls_root6"] is True
            and continuation_chain["root6_function"] == "tile_dependency_root_6"
            and continuation_chain["root5_chains_root7"] is True
            and continuation_chain["resident_root6_call"] is False
            and continuation_chain["resident_root7_call"] is False
            and continuation_chain["readiness_wait_count"] == 0
        ):
            raise AssertionError(
                f"unexpected root-6 continuation lowering: {continuation_chain}"
            )
        ablation_summary = {
            "correctness": ablation_correctness,
            "binary": ablation_binary,
            "lowering": _lowering_summary(ablation_code),
            "continuation_chain": continuation_chain,
            "scheduler_path": ablation_path,
            "resources": _resources(ablation_call),
            "ptx_synchronization": _ptx_synchronization_summary(ablation_call),
            "config": ablation_config,
        }
    root6_only_summary: dict[str, object] | None = None
    if root6_only_call is not None:
        assert root6_only_code is not None
        assert root6_only_path is not None
        assert root6_only_config is not None
        root6_only_chain = _continuation_chain_summary(root6_only_code)
        if not (
            root6_only_chain["root5_calls_root6"] is True
            and root6_only_chain["root5_chains_root7"] is False
            and root6_only_chain["resident_root6_call"] is False
            and root6_only_chain["resident_root7_call"] is True
            and int(root6_only_chain["readiness_wait_count"]) > 0
        ):
            raise AssertionError(
                f"unexpected root-6-only continuation lowering: {root6_only_chain}"
            )
        root6_only_summary = {
            "correctness": root6_only_correctness,
            "binary": root6_only_binary,
            "lowering": _lowering_summary(root6_only_code),
            "continuation_chain": root6_only_chain,
            "scheduler_path": root6_only_path,
            "resources": _resources(root6_only_call),
            "ptx_synchronization": _ptx_synchronization_summary(root6_only_call),
            "config": root6_only_config,
        }
    u32_summary: dict[str, object] | None = None
    if u32_call is not None:
        assert u32_code is not None
        assert u32_path is not None
        assert u32_config is not None
        u32_summary = {
            "correctness": u32_correctness,
            "binary": u32_binary,
            "lowering": _lowering_summary(u32_code),
            "scheduler_path": u32_path,
            "resources": _resources(u32_call),
            "ptx_synchronization": _ptx_synchronization_summary(u32_call),
            "config": u32_config,
        }
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
        "root6_continuation_ablation": ablation_summary,
        "root6_only_continuation_ablation": root6_only_summary,
        "u32_epoch_ablation": u32_summary,
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
    parser.add_argument(
        "--root6-continuation-ablation",
        action="store_true",
        help="benchmark diagnostic 5->6-only and 5->6->7 continuations",
    )
    parser.add_argument(
        "--u32-epoch-ablation",
        action="store_true",
        help="benchmark bounded-run uint32 readiness epochs",
    )
    print(json.dumps(benchmark(parser.parse_args()), indent=2, default=str))


if __name__ == "__main__":
    main()
