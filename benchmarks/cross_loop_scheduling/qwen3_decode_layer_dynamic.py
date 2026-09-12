"""Dynamic-batch scheduler validation for the pretuned Qwen3 decode layer.

The dynamic and static persistent variants are mechanically derived from the
same checked-in full-layer source by ``qwen3_decode_layer_batched.py``.  This
probe changes neither the fused boundary nor its arithmetic; it makes batch
symbolic, reuses one compiled dynamic callable for B1 and ragged B2, and
compares it with an exact-shape compilation of the same source at each case.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cross_loop_scheduling import qwen3_decode_layer_batched as base
from pretuned_kernels._bench import bench_pre_captured_cudagraphs
from pretuned_kernels._bench import capture_cuda_graph
from pretuned_kernels._bench import thermal_warmup
from pretuned_kernels.megakernels.qwen3_decode_layer._helion_aot_qwen3_decode_layer_cuda_sm100 import (
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


def _validate_active_participant_protocols() -> None:
    """Check both diagnostic barrier algebras, including wrapped cohorts."""
    worker_count = 1184
    for first_slot, task_count in (
        (0, 32),
        (32, 768),
        (1180, 10),
        (2357, 8),
        (17, worker_count),
        (31, worker_count + 7),
    ):
        active_count = min(worker_count, task_count)
        assert active_count > 0
        weights = []
        owners = []
        for worker in range(worker_count):
            local_worker = (
                worker + worker_count - first_slot % worker_count
            ) % worker_count
            if local_worker < active_count:
                owners.append(worker)
                weights.append(
                    worker_count // active_count
                    + int(local_worker < worker_count % active_count)
                )
        assert len(owners) == active_count
        assert sum(weights) == worker_count
        expected_owners = {
            (first_slot + ordinal) % worker_count for ordinal in range(active_count)
        }
        assert set(owners) == expected_owners
        # Protocol A uses one arrival per participant and a runtime target of
        # ``epoch * W + A``.  Its positive sentinel extends the same invariant
        # to an empty root without changing the fixed epoch stride.
        assert len(owners) == active_count
    empty_active_count = 0
    empty_participant_count = max(empty_active_count, 1)
    assert empty_participant_count == 1


def _compile(
    kernel: Kernel,
    args: tuple[object, ...],
    *,
    multiplier: int,
    request_major: bool,
    supported_intermediate_continuations: bool = False,
    active_participant_barriers: bool = False,
    bounded_unit_barriers: bool = False,
) -> tuple[CompiledConfig, dict[str, object], str, dict[str, object]]:
    if active_participant_barriers and bounded_unit_barriers:
        raise ValueError("select only one active-participant barrier protocol")
    if active_participant_barriers or bounded_unit_barriers:
        _validate_active_participant_protocols()
    bound = kernel.bind(args)
    values = deepcopy(CONFIG)
    values["num_sm_multiplier"] = multiplier
    if request_major:
        values["loop_orders"][5] = [2, 0, 1]
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)

    path_records: dict[str, object] = {}
    original_root_major = cross_loop_scheduler._build_root_major_worker_schedule
    original_event_frontier = (
        cross_loop_scheduler._build_parametric_event_frontier_worker_schedule
    )
    original_global = cross_loop_scheduler._global_unit_list_schedule
    original_pipeline_plan = cross_loop_codegen.build_static_pipeline_plan
    original_build_readiness_graph = cross_loop_scheduler.build_readiness_graph
    original_choose_continuations = (
        cross_loop_scheduler.choose_final_arrival_continuations
    )
    original_emit = cross_loop_codegen.emit_cross_loop_schedule
    original_root_counter_dtype = cross_loop_codegen._CROSS_LOOP_COUNTER_DTYPE
    captured_plan: list[cross_loop_scheduler.StaticPipelinePlan] = []

    def readiness_graph(*call_args: object, **call_kwargs: object):  # noqa: ANN202
        result = original_build_readiness_graph(*call_args, **call_kwargs)
        path_records["readiness_events"] = [
            {
                "event_id": event.event_id,
                "root_barrier_producer_root": event.root_barrier_producer_root,
                "producer_roots": [
                    producer.producer_root for producer in event.producers
                ],
                "producer_sites": [
                    producer.producer_site_id for producer in event.producers
                ],
                "key_counts": [
                    [axis, str(count)]
                    for axis, count in event.readiness_key_domain.axis_counts_items
                ],
                "fan_in": cross_loop_scheduler._uniform_arrival_count(event.producers),
                "consumers": [
                    {
                        "root": consumer.consumer_root,
                        "site": consumer.consumer_site_id,
                        "keys_total": consumer.keys_by_consumer.is_total_function(),
                        "keys_bijection": (
                            consumer.keys_by_consumer.is_positional_bijection()
                        ),
                        "continuation_counter_supported": (
                            cross_loop_scheduler._supports_parameterized_counter(
                                cross_loop_scheduler.ReadinessCounterPlan(
                                    producers=event.producers,
                                    consumers=(consumer,),
                                    continuation_consumer_index=0,
                                )
                            )
                        ),
                    }
                    for consumer in event.consumers
                ],
            }
            for event in result.events
        ]
        focused_events: dict[str, object] = {}
        for event in result.events:
            producer_roots = tuple(
                producer.producer_root for producer in event.producers
            )
            for consumer in event.consumers:
                edge = (producer_roots, consumer.consumer_root)
                if edge == ((7,), 8):
                    converse = consumer.keys_by_consumer.converse()
                    focused_events["7_to_8"] = {
                        "source_domain": repr(consumer.keys_by_consumer.source_domain),
                        "target_domain": repr(consumer.keys_by_consumer.target_domain),
                        "pieces": repr(consumer.keys_by_consumer.pieces),
                        "converse": repr(converse),
                        "converse_total": (
                            converse is not None and converse.is_total_function()
                        ),
                        "covered_obligations": len(consumer.covered_obligations),
                    }
                elif edge == ((8,), 9):
                    focused_events.setdefault("8_to_9", []).append(
                        {
                            "event_id": event.event_id,
                            "coarse_barrier": (
                                event.root_barrier_producer_root is not None
                            ),
                            "key_domain": repr(event.readiness_key_domain),
                            "covered_obligations": len(consumer.covered_obligations),
                        }
                    )
                elif edge == ((0, 9), 10):
                    focused_events["0_9_to_10"] = {
                        "event_id": event.event_id,
                        "key_domain": repr(event.readiness_key_domain),
                        "fan_in": cross_loop_scheduler._uniform_arrival_count(
                            event.producers
                        ),
                        "covered_obligations": len(consumer.covered_obligations),
                    }
        path_records["focused_events"] = focused_events
        return result

    def choose_continuations(  # noqa: ANN202
        *call_args: object, **call_kwargs: object
    ):
        result = original_choose_continuations(*call_args, **call_kwargs)
        graph = call_args[0]
        path_records["continuation_candidates"] = [
            {
                "event_id": continuation.event_id,
                "producer_roots": [
                    producer.producer_root
                    for producer in graph.event(continuation.event_id).producers
                ],
                "consumer_root": graph.event(continuation.event_id)
                .consumers[continuation.consumer_index]
                .consumer_root,
            }
            for continuation in result
        ]
        return result

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

    def pipeline_plan(*call_args: object, **call_kwargs: object):  # noqa: ANN202
        root_task_orders = call_kwargs["root_task_orders"]
        root_domains = tuple(
            task_order.target_domain for task_order in root_task_orders
        )
        if supported_intermediate_continuations and any(
            domain.parameter_symbols for domain in root_domains
        ):
            # Diagnostic-only ablation: use the production proof machinery and
            # change only admission of the two intermediate continuations that
            # already carry complete parameterized counter certificates.
            # The root IDs deliberately make this a Qwen attribution probe,
            # not a proposed compiler policy.
            worker_schedule = cross_loop_scheduler._build_root_major_worker_schedule(
                root_domains,
                root_task_orders,
                call_kwargs["worker_count"],
            )
            graph = cross_loop_scheduler.build_readiness_graph(
                dependency_graph=call_kwargs["dependency_graph"],
                root_task_orders=root_task_orders,
                site_domains=call_kwargs["site_domains"],
                publishable_site_ids=call_kwargs.get("publishable_site_ids"),
                prove_nonnegative=call_kwargs.get("prove_nonnegative"),
            )
            candidates = cross_loop_scheduler.choose_final_arrival_continuations(
                graph,
                worker_schedule,
            )
            selected = tuple(
                continuation
                for continuation in candidates
                if (
                    tuple(
                        producer.producer_root
                        for producer in graph.event(continuation.event_id).producers
                    ),
                    graph.event(continuation.event_id)
                    .consumers[continuation.consumer_index]
                    .consumer_root,
                )
                in {((6,), 7), ((12,), 13)}
            )
            counters = tuple(
                plan
                for plan in cross_loop_scheduler.choose_readiness_counters(
                    graph,
                    selected,
                )
                if cross_loop_scheduler._supports_parameterized_counter(plan)
            )
            if sum(
                plan.continuation_consumer_index is not None for plan in counters
            ) != len(selected):
                raise AssertionError(
                    "supported continuation ablation lost a selected continuation"
                )
            counters, barriers = cross_loop_scheduler._finalize_emitted_synchronization(
                readiness_graph=graph,
                readiness_counters=counters,
            )
            if not cross_loop_scheduler._parameterized_prerequisites_follow_root_order(
                counters,
                barriers,
            ):
                raise AssertionError("continuation ablation broke root progress")
            continuation_roots = frozenset(
                consumer.consumer_root
                for plan in counters
                if (consumer := plan.continuation_consumer) is not None
            )
            worker_schedule = cross_loop_scheduler._build_root_major_worker_schedule(
                root_domains,
                root_task_orders,
                call_kwargs["worker_count"],
                excluded_roots=continuation_roots,
            )
            result = cross_loop_scheduler.StaticPipelinePlan(
                worker_schedule=worker_schedule,
                root_task_orders=root_task_orders,
                readiness_counters=counters,
                root_barrier_edges=barriers,
            )
            path_records["diagnostic_continuation_ablation"] = {
                "selected": [
                    {
                        "producer_roots": [
                            producer.producer_root
                            for producer in graph.event(continuation.event_id).producers
                        ],
                        "consumer_root": graph.event(continuation.event_id)
                        .consumers[continuation.consumer_index]
                        .consumer_root,
                    }
                    for continuation in selected
                ],
                "excluded_roots": sorted(continuation_roots),
            }
        else:
            result = original_pipeline_plan(*call_args, **call_kwargs)
        path_records["static_pipeline_plan"] = {
            "readiness_counters": [
                {
                    "key_axes": list(plan.readiness_key_domain.axis_order),
                    "key_counts": [
                        [axis, str(count)]
                        for axis, count in plan.readiness_key_domain.axis_counts_items
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
        captured_plan[:] = [result]
        return result

    def emit(*call_args: object, **call_kwargs: object):  # noqa: ANN202
        result = original_emit(*call_args, **call_kwargs)
        if not (active_participant_barriers or bounded_unit_barriers):
            return result
        if len(captured_plan) != 1:
            raise AssertionError("active-barrier ablation did not capture one plan")
        (plan,) = captured_plan
        geometry = cross_loop_scheduler._parametric_root_major_schedule_geometry(
            plan.worker_schedule
        )
        if geometry is None:
            return result

        strategy = call_args[1]
        device_function = call_args[2]
        worker = cross_loop_codegen.typed_program_id(0)
        worker_count = plan.worker_schedule.worker_count

        def scheduled_loop(statement: ast.stmt) -> ast.For | None:
            candidates = (
                (statement,)
                if isinstance(statement, ast.For)
                else tuple(statement.body)
                if isinstance(statement, ast.If)
                else ()
            )
            return next(
                (
                    candidate
                    for candidate in candidates
                    if isinstance(candidate, ast.For)
                    and isinstance(candidate.target, ast.Name)
                    and candidate.target.id == strategy.virtual_pid_var
                ),
                None,
            )

        segment_indices = tuple(
            index
            for index, statement in enumerate(result)
            if scheduled_loop(statement) is not None
        )
        if len(segment_indices) != len(geometry):
            raise AssertionError(
                "active-barrier ablation could not identify every root segment"
            )

        barrier_producer_roots = {
            producer for producer, _consumer in plan.root_barrier_edges
        }
        task_count_by_root = {
            segment.root: task_count for segment, _first_slot, task_count in geometry
        }
        epoch_name = next(
            (
                node.id
                for statement in result
                for node in ast.walk(statement)
                if isinstance(node, ast.Name)
                and node.id.startswith("tile_dependency_epoch")
            ),
            None,
        )
        if epoch_name is None:
            raise AssertionError("active-barrier ablation could not find epoch")
        transformed: list[dict[str, object]] = []
        for result_index, (segment, first_slot, task_count) in zip(
            segment_indices,
            geometry,
            strict=True,
        ):
            statement = result[result_index]
            task_count_text = device_function.sympy_expr(task_count)
            first_slot_text = device_function.sympy_expr(first_slot)
            local_worker = (
                f"(({worker}) + {worker_count} - "
                f"(({first_slot_text}) % {worker_count})) % {worker_count}"
            )
            active_count = f"tl.minimum({worker_count}, ({task_count_text}))"
            participant_count = (
                f"tl.maximum(({active_count}), 1)"
                if bounded_unit_barriers
                else active_count
            )
            active_condition = cross_loop_codegen.expr_from_string(
                f"({local_worker}) < ({participant_count})"
            )
            if isinstance(statement, ast.If):
                statement.test = active_condition
                active_body = statement.body
            else:
                result[result_index] = ast.If(
                    test=active_condition,
                    body=[statement],
                    orelse=[],
                )
                active_body = result[result_index].body

            publication_count = 0
            if segment.root in barrier_producer_roots:
                safe_active_count = f"tl.maximum(({active_count}), 1)"
                weight = (
                    None
                    if bounded_unit_barriers
                    else cross_loop_codegen.expr_from_string(
                        f"({worker_count} // ({safe_active_count}) + "
                        f"tl.cast(({local_worker}) < "
                        f"({worker_count} % ({safe_active_count})), tl.int32))"
                    )
                )
                rewritten_body: list[ast.stmt] = []
                for child in active_body:
                    if not isinstance(child, ast.Expr) or not isinstance(
                        child.value,
                        ast.Call,
                    ):
                        rewritten_body.append(child)
                        continue
                    call = child.value
                    if (
                        isinstance(call.func, ast.Attribute)
                        and call.func.attr == "atomic_add"
                        and len(call.args) >= 2
                        and isinstance(call.args[1], ast.Constant)
                        and call.args[1].value == 1
                    ):
                        if bounded_unit_barriers:
                            counter = ast.unparse(call.args[0])
                            epoch_base = (
                                f"tl.cast({epoch_name}, tl.uint64) * "
                                f"tl.cast({worker_count}, tl.uint64)"
                            )
                            rewritten_body.append(
                                cross_loop_codegen.statement_from_string(
                                    f"tl.atomic_max({counter}, {epoch_base}, "
                                    "sem='relaxed', scope='gpu')"
                                )
                            )
                        else:
                            assert weight is not None
                            call.args[1] = weight
                        publication_count += 1
                    rewritten_body.append(child)
                active_body[:] = rewritten_body
                if publication_count != 1:
                    raise AssertionError(
                        f"root {segment.root} has {publication_count} direct "
                        "barrier publications"
                    )
            transformed.append(
                {
                    "root": segment.root,
                    "first_slot": str(first_slot),
                    "task_count": str(task_count),
                    "publishes_barrier": segment.root in barrier_producer_roots,
                }
            )
            if bounded_unit_barriers:
                incoming_roots = tuple(
                    sorted(
                        producer
                        for producer, consumer in plan.root_barrier_edges
                        if consumer == segment.root
                    )
                )
                waits = tuple(
                    child for child in active_body if isinstance(child, ast.While)
                )
                if len(waits) != len(incoming_roots):
                    raise AssertionError(
                        f"root {segment.root} has {len(waits)} waits for "
                        f"{len(incoming_roots)} incoming barriers"
                    )
                for wait, producer_root in zip(waits, incoming_roots, strict=True):
                    producer_task_count = device_function.sympy_expr(
                        task_count_by_root[producer_root]
                    )
                    producer_active_count = (
                        f"tl.minimum({worker_count}, ({producer_task_count}))"
                    )
                    producer_participants = f"tl.maximum(({producer_active_count}), 1)"
                    if (
                        not isinstance(wait.test, ast.Compare)
                        or len(wait.test.comparators) != 1
                    ):
                        raise AssertionError("unexpected root-barrier wait predicate")
                    wait.test.comparators[0] = cross_loop_codegen.expr_from_string(
                        f"tl.cast({epoch_name}, tl.uint64) * "
                        f"tl.cast({worker_count}, tl.uint64) + "
                        f"tl.cast(({producer_participants}), tl.uint64)"
                    )
        path_records["diagnostic_active_participant_barriers"] = {
            "worker_count": worker_count,
            "positive_batch_guard": True,
            "protocol": "bounded_unit" if bounded_unit_barriers else "weighted",
            "segments": transformed,
        }
        return result

    cross_loop_scheduler._build_root_major_worker_schedule = root_major
    cross_loop_scheduler._build_parametric_event_frontier_worker_schedule = (
        event_frontier
    )
    cross_loop_scheduler._global_unit_list_schedule = global_list
    cross_loop_scheduler.build_readiness_graph = readiness_graph
    cross_loop_scheduler.choose_final_arrival_continuations = choose_continuations
    cross_loop_codegen.build_static_pipeline_plan = pipeline_plan
    cross_loop_codegen.emit_cross_loop_schedule = emit
    if bounded_unit_barriers:
        cross_loop_codegen._CROSS_LOOP_COUNTER_DTYPE = torch.uint64
    try:
        compile_start = time.perf_counter()
        compiled = bound.compile_config(config)
        path_records["compile_seconds"] = time.perf_counter() - compile_start
        cache_path = next(iter(bound._cache_path_map.values()))
        if cache_path is None:
            raise RuntimeError("compiled Qwen kernel has no generated source path")
        code = Path(cache_path).read_text()
    finally:
        cross_loop_scheduler._build_root_major_worker_schedule = original_root_major
        cross_loop_scheduler._build_parametric_event_frontier_worker_schedule = (
            original_event_frontier
        )
        cross_loop_scheduler._global_unit_list_schedule = original_global
        cross_loop_scheduler.build_readiness_graph = original_build_readiness_graph
        cross_loop_scheduler.choose_final_arrival_continuations = (
            original_choose_continuations
        )
        cross_loop_codegen.build_static_pipeline_plan = original_pipeline_plan
        cross_loop_codegen.emit_cross_loop_schedule = original_emit
        cross_loop_codegen._CROSS_LOOP_COUNTER_DTYPE = original_root_counter_dtype
    return compiled, values, code, path_records


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


def _case_contexts() -> tuple[tuple[int, ...], ...]:
    return ((base.qwen.CONTEXT,), (2048, base.qwen.CONTEXT))


def _case_label(contexts: tuple[int, ...]) -> str:
    return f"B{len(contexts)}_S" + "_".join(str(value) for value in contexts)


def _make_case_inputs(contexts: tuple[int, ...]) -> dict[str, torch.Tensor]:
    if contexts == (base.qwen.CONTEXT,):
        tensors = base.qwen._make_inputs()
        tensors["cos_sin"].normal_()
        tensors["context_lens"] = torch.tensor(
            contexts,
            device="cuda",
            dtype=torch.int64,
        )
        return tensors
    return base._make_inputs(contexts)


@torch.inference_mode()
def benchmark(args: argparse.Namespace) -> dict[str, object]:
    base.qwen._require_sm100()
    cases = _case_contexts()
    if args.exemplar_batch != 2:
        raise ValueError("this B1/B2 probe uses B2 as its dynamic exemplar")

    base_inputs = {contexts: _make_case_inputs(contexts) for contexts in cases}
    dynamic_variants = {
        contexts: base._variant_inputs(tensors)
        for contexts, tensors in base_inputs.items()
    }
    exemplar_args = dynamic_variants[cases[-1]][1]
    dynamic_kernel = base._make_batched_kernel(dynamic_batch=True)
    dynamic_call, dynamic_config, dynamic_code, dynamic_path = _compile(
        dynamic_kernel,
        exemplar_args,
        multiplier=args.multiplier,
        request_major=args.request_major,
        supported_intermediate_continuations=(
            args.supported_intermediate_continuations
        ),
        active_participant_barriers=args.active_participant_barriers,
        bounded_unit_barriers=args.bounded_unit_barriers,
    )
    # Establish the replay epoch and binary with the exemplar before checking
    # the smaller case, yielding an explicit B2 -> B1 -> B2 reuse sequence.
    exemplar_reset = dynamic_variants[cases[-1]][2]
    exemplar_reset()
    dynamic_call(*exemplar_args)
    torch.cuda.synchronize()

    static_calls: dict[tuple[int, ...], CompiledConfig] = {}
    static_configs: dict[tuple[int, ...], dict[str, object]] = {}
    static_paths: dict[tuple[int, ...], dict[str, object]] = {}
    static_lowering: dict[tuple[int, ...], dict[str, object]] = {}
    correctness: dict[tuple[int, ...], dict[str, object]] = {}
    captures: dict[
        tuple[int, ...],
        tuple[tuple[str, torch.cuda.CUDAGraph, object, Callable[[], None]], ...],
    ] = {}
    expected_dynamic_binary: dict[str, object] | None = None

    for contexts in cases:
        _dynamic_tensors, dynamic_args, dynamic_reset = dynamic_variants[contexts]
        _static_tensors, static_args, static_reset = base._variant_inputs(
            base_inputs[contexts]
        )
        static_kernel = base._make_batched_kernel(dynamic_batch=False)
        static_call, static_config, static_code, static_path = _compile(
            static_kernel,
            static_args,
            multiplier=args.multiplier,
            request_major=args.request_major,
        )
        static_calls[contexts] = static_call
        static_configs[contexts] = static_config
        static_paths[contexts] = static_path
        static_lowering[contexts] = _lowering_summary(static_code)

        dynamic_reset()
        dynamic_outputs = dynamic_call(*dynamic_args)
        static_reset()
        static_outputs = static_call(*static_args)
        torch.cuda.synchronize()
        for dynamic_value, static_value in zip(
            dynamic_outputs, static_outputs, strict=True
        ):
            torch.testing.assert_close(dynamic_value, static_value, atol=0, rtol=0)
        correctness[contexts] = {
            "dynamic_vs_static_bit_exact": True,
            "output_shapes": [tuple(value.shape) for value in dynamic_outputs],
        }

        current_binary = _binary_summary(dynamic_call)
        if expected_dynamic_binary is None:
            expected_dynamic_binary = current_binary
        elif current_binary != expected_dynamic_binary:
            raise AssertionError(
                f"dynamic {contexts} created a new binary: {current_binary}"
            )

        calls: tuple[tuple[str, Callable[[], object], Callable[[], None]], ...] = (
            (
                "persistent_static",
                lambda call=static_call, call_args=static_args: call(*call_args),
                static_reset,
            ),
            (
                "persistent_dynamic",
                lambda call_args=dynamic_args: dynamic_call(*call_args),
                dynamic_reset,
            ),
        )
        captures[contexts] = tuple(
            (name, *capture_cuda_graph(call, reset), reset)
            for name, call, reset in calls
        )

    thermal_warmup(args.warmup_ms)
    timings: dict[tuple[int, ...], dict[str, float]] = {}
    for contexts, case_captures in captures.items():
        elapsed = bench_pre_captured_cudagraphs(
            [graph.replay for _name, graph, _output, _reset in case_captures],
            rep=args.repetitions,
            resets=[reset for _name, _graph, _output, reset in case_captures],
        )
        timings[contexts] = {
            name: milliseconds * 1000
            for (name, _graph, _output, _reset), milliseconds in zip(
                case_captures, elapsed, strict=True
            )
        }

    dynamic_binary = _binary_summary(dynamic_call)
    if dynamic_binary["specialization_count"] != 1:
        raise AssertionError(
            f"dynamic kernel did not retain one specialization: {dynamic_binary}"
        )
    return {
        "workload": "Qwen3-8B FP8 full decode layer, ragged B1/B2",
        "contexts": cases,
        "device": torch.cuda.get_device_name(),
        "timings_us_cold_l2": {
            _case_label(contexts): value for contexts, value in timings.items()
        },
        "correctness": {
            _case_label(contexts): value for contexts, value in correctness.items()
        },
        "one_dynamic_cubin_across_batch_sizes": True,
        "dynamic_binary": dynamic_binary,
        "dynamic_lowering": _lowering_summary(dynamic_code),
        "dynamic_scheduler_path": dynamic_path,
        "static_lowering": {
            _case_label(contexts): value for contexts, value in static_lowering.items()
        },
        "static_scheduler_path": {
            _case_label(contexts): value for contexts, value in static_paths.items()
        },
        "static_binary": {
            _case_label(contexts): _binary_summary(call)
            for contexts, call in static_calls.items()
        },
        "resources": {
            "persistent_dynamic": _resources(dynamic_call),
            "persistent_static": {
                _case_label(contexts): _resources(call)
                for contexts, call in static_calls.items()
            },
        },
        "configs": {
            "persistent_dynamic": dynamic_config,
            "persistent_static": {
                _case_label(contexts): value
                for contexts, value in static_configs.items()
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exemplar-batch", type=int, default=2)
    parser.add_argument("--multiplier", type=int, default=8)
    parser.add_argument("--request-major", action="store_true")
    parser.add_argument(
        "--supported-intermediate-continuations",
        action="store_true",
        help="diagnostic Qwen-only ablation for already-certified 6->7 and 12->13",
    )
    parser.add_argument(
        "--active-participant-barriers",
        action="store_true",
        help="diagnostic weighted active-worker root-barrier lowering",
    )
    parser.add_argument(
        "--bounded-unit-barriers",
        action="store_true",
        help="diagnostic active-worker uint64 epoch barriers with unit arrivals",
    )
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmup-ms", type=int, default=4000)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    result = benchmark(args)
    if args.compact:
        result = {
            key: result[key]
            for key in (
                "workload",
                "contexts",
                "device",
                "timings_us_cold_l2",
                "correctness",
                "one_dynamic_cubin_across_batch_sizes",
                "dynamic_binary",
                "static_binary",
                "dynamic_lowering",
                "resources",
            )
        } | {
            "dynamic_scheduler_path": {
                key: result["dynamic_scheduler_path"].get(key)
                for key in (
                    "parametric_root_major",
                    "diagnostic_continuation_ablation",
                    "diagnostic_active_participant_barriers",
                    "focused_events",
                    "static_pipeline_plan",
                    "compile_seconds",
                )
                if result["dynamic_scheduler_path"].get(key) is not None
            }
        }
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
