"""Dynamic-batch Muse/Glimmer FFN scheduler probe.

This is the five-launch boundary used by ``muse_glimmer_ffn.py``:

    gate/up split-K -> reduction -> SiLU/mul -> down split-K -> reduction

Only the token count is dynamic.  Intermediate allocations have a fixed
capacity so dependency analysis sees one stable address space while the
natural multi-axis roots retain a symbolic batch extent.  The dynamic and
static persistent kernels are decorated from the same generated function and
use the same tile/resource configuration.  The standalone path invokes the
same five component bodies in separate launches.
"""

from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import hashlib
import inspect
from itertools import starmap
import json
import linecache
from pathlib import Path
import sys
import textwrap
import time
import types
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pretuned_kernels._bench import bench_pre_captured_cudagraphs
from pretuned_kernels._bench import capture_cuda_graph
from pretuned_kernels._bench import thermal_warmup
import torch

import helion
from helion._compiler import cross_loop_codegen
from helion._compiler import cross_loop_scheduler
from helion._compiler.tile_dependency import CoordinateDomain
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.kernel import CompiledConfig
    from helion.runtime.kernel import Kernel


HIDDEN = 6656
INTERMEDIATE = 19968
GATE_SPLITS = 16
ACTIVATION_SPLITS = 16
SPLIT_HIDDEN = HIDDEN // GATE_SPLITS
SPLIT_INTERMEDIATE = INTERMEDIATE // ACTIVATION_SPLITS
GATE_N = 32
GATE_REDUCE_N = 64
ACTIVATION_N = 256
DOWN_N = 32
DOWN_REDUCE_N = 64
MAX_BATCH_SIZE = 4


@helion.kernel(
    static_shapes=False,
    autotune_effort="none",
    backend="triton",
    triton_do_not_specialize=True,
)
def dynamic_gate_up_splitk(
    ff_input: torch.Tensor,
    weight_t: torch.Tensor,
) -> torch.Tensor:
    """Natural multi-axis N32/K128 gate/up split-K root."""
    batch_size, hidden = ff_input.size()
    weight_hidden, twice_intermediate = weight_t.size()
    hidden = hl.specialize(hidden)
    weight_hidden = hl.specialize(weight_hidden)
    twice_intermediate = hl.specialize(twice_intermediate)
    assert hidden == HIDDEN
    assert weight_hidden == HIDDEN
    assert twice_intermediate == 2 * INTERMEDIATE
    torch._check(batch_size <= 4)
    partial = torch.empty(
        (4, 16, 2, 16, 1248),
        dtype=torch.float32,
        device=ff_input.device,
    )
    for tile_b, tile_a, tile_half, tile_n, tile_s in hl.tile(
        [batch_size, 16, 2, 1248, 16],
        block_size=[1, 1, 1, None, 1],
    ):
        output_n = tile_half.begin * 19968 + tile_a.begin * 1248 + tile_n.index
        accumulator = hl.zeros([tile_s, 1, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(416):
            global_k = (
                tile_s.index[:, None, None] * 416
                + tile_k.index[None, :, None]
            )
            lhs = ff_input[tile_b.begin, global_k].view(tile_s, 1, tile_k)
            rhs = weight_t[global_k, output_n[None, None, :]]
            accumulator = torch.baddbmm(accumulator, lhs, rhs)
        partial[
            tile_b.begin,
            tile_s,
            tile_half.begin,
            tile_a.begin,
            tile_n,
        ] = accumulator.squeeze(1)
    return partial


@helion.kernel(
    static_shapes=False,
    autotune_effort="none",
    backend="triton",
    triton_do_not_specialize=True,
)
def dynamic_gate_up_reduce(
    partial: torch.Tensor,
    batch_ref: torch.Tensor,
) -> torch.Tensor:
    """N64 FP32 split reduction to the production BF16 intermediate."""
    batch_size = batch_ref.size(0)
    torch._check(batch_size <= 4)
    output = torch.empty(
        (4, 2, 16, 1248),
        dtype=torch.bfloat16,
        device=partial.device,
    )
    for tile_b, tile_a, tile_half, tile_n in hl.tile(
        [batch_size, 16, 2, 1248], block_size=[1, 1, 1, None]
    ):
        values = partial[tile_b, :, tile_half, tile_a, tile_n]
        output[tile_b, tile_half, tile_a, tile_n] = torch.sum(
            values, dim=1
        ).to(output.dtype)
    return output


@helion.kernel(
    static_shapes=False,
    autotune_effort="none",
    backend="triton",
    triton_do_not_specialize=True,
)
def dynamic_silu_and_mul(
    gate_up: torch.Tensor,
    batch_ref: torch.Tensor,
) -> torch.Tensor:
    """Natural N256 activation loop partitioned by down-projection K slice."""
    batch_size = batch_ref.size(0)
    torch._check(batch_size <= 4)
    output = torch.empty(
        (4, 16, 1248),
        device=gate_up.device,
        dtype=gate_up.dtype,
    )
    for tile_b, tile_a in hl.tile([batch_size, 16], block_size=[1, 1]):
        for tile_i in hl.tile(1248):
            gate = gate_up[tile_b, 0, tile_a, tile_i]
            up = gate_up[tile_b, 1, tile_a, tile_i]
            silu = gate.to(torch.float32) * torch.sigmoid(gate.to(torch.float32))
            output[tile_b, tile_a, tile_i] = silu.to(up.dtype) * up
    return output


@helion.kernel(
    static_shapes=False,
    autotune_effort="none",
    backend="triton",
    triton_do_not_specialize=True,
)
def dynamic_down_splitk(
    activation: torch.Tensor,
    weight_t: torch.Tensor,
    batch_ref: torch.Tensor,
) -> torch.Tensor:
    """N32/K128 down projection with one CTA per activation slice."""
    batch_size = batch_ref.size(0)
    weight_intermediate, hidden = weight_t.size()
    weight_intermediate = hl.specialize(weight_intermediate)
    hidden = hl.specialize(hidden)
    assert weight_intermediate == INTERMEDIATE
    assert hidden == HIDDEN
    torch._check(batch_size <= 4)
    partial = torch.empty(
        (4, 16, 6656),
        dtype=torch.float32,
        device=activation.device,
    )
    for tile_b, tile_a, tile_n in hl.tile(
        [batch_size, 16, 6656], block_size=[1, 1, None]
    ):
        accumulator = hl.zeros([tile_a, 1, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(1248):
            lhs = activation[tile_b.begin, tile_a, tile_k].view(tile_a, 1, tile_k)
            global_k = (
                tile_a.index[:, None, None] * 1248
                + tile_k.index[None, :, None]
            )
            rhs = weight_t[global_k, tile_n.index[None, None, :]]
            accumulator = torch.baddbmm(accumulator, lhs, rhs)
        partial[tile_b.begin, tile_a, tile_n] = accumulator.squeeze(1)
    return partial


@helion.kernel(
    static_shapes=False,
    autotune_effort="none",
    backend="triton",
    triton_do_not_specialize=True,
)
def dynamic_down_reduce(
    partial: torch.Tensor,
    batch_ref: torch.Tensor,
) -> torch.Tensor:
    """N64 FP32 split reduction to the final dynamic BF16 output."""
    batch_size = batch_ref.size(0)
    torch._check(batch_size <= 4)
    output = torch.empty(
        (batch_size, HIDDEN), dtype=torch.bfloat16, device=partial.device
    )
    for tile_b, tile_n in hl.tile(
        [batch_size, 6656], block_size=[1, None]
    ):
        values = partial[tile_b, :, tile_n]
        output[tile_b, tile_n] = torch.sum(values, dim=1).to(output.dtype)
    return output


class _KernelWithFunction(Protocol):
    fn: object


@dataclasses.dataclass(frozen=True)
class _Invocation:
    prefix: str
    kernel: _KernelWithFunction
    arguments: dict[str, str]
    outputs: dict[str, str]


class _AssignedNames(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)


class _RenameNames(ast.NodeTransformer):
    def __init__(self, names: dict[str, str]) -> None:
        self.names = names

    def visit_Name(self, node: ast.Name) -> ast.Name:
        renamed = self.names.get(node.id)
        if renamed is None:
            return node
        return ast.copy_location(ast.Name(id=renamed, ctx=node.ctx), node)


def _inline_invocation(invocation: _Invocation) -> tuple[list[ast.stmt], list[ast.For]]:
    source = textwrap.dedent(inspect.getsource(invocation.kernel.fn))
    module = ast.parse(source)
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    assert len(functions) == 1
    function = functions[0]
    parameters = [argument.arg for argument in function.args.args]
    assigned = _AssignedNames()
    for statement in function.body:
        assigned.visit(statement)
    rename = {
        name: invocation.outputs.get(name, f"__muse_dynamic_{invocation.prefix}_{name}")
        for name in set(parameters) | assigned.names
    }
    transformer = _RenameNames(rename)
    preamble = [
        ast.Assign(
            targets=[ast.Name(id=rename[parameter], ctx=ast.Store())],
            value=ast.parse(invocation.arguments[parameter], mode="eval").body,
        )
        for parameter in parameters
    ]
    loops: list[ast.For] = []
    for statement in function.body:
        if isinstance(statement, ast.Return):
            continue
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        transformed = transformer.visit(ast.fix_missing_locations(statement))
        assert isinstance(transformed, ast.stmt)
        if isinstance(transformed, ast.For):
            loops.append(transformed)
        else:
            preamble.append(transformed)
    return preamble, loops


def _persistent_source() -> Callable[..., object]:
    events = (
        _Invocation(
            "gate_main",
            dynamic_gate_up_splitk,
            {"ff_input": "ff_input", "weight_t": "gate_up_weight_t"},
            {"partial": "gate_partial"},
        ),
        _Invocation(
            "gate_reduce",
            dynamic_gate_up_reduce,
            {"partial": "gate_partial", "batch_ref": "ff_input"},
            {"output": "gate_up"},
        ),
        _Invocation(
            "activation",
            dynamic_silu_and_mul,
            {"gate_up": "gate_up", "batch_ref": "ff_input"},
            {"output": "activation"},
        ),
        _Invocation(
            "down_main",
            dynamic_down_splitk,
            {
                "activation": "activation",
                "weight_t": "down_weight_t",
                "batch_ref": "ff_input",
            },
            {"partial": "down_partial"},
        ),
        _Invocation(
            "down_reduce",
            dynamic_down_reduce,
            {"partial": "down_partial", "batch_ref": "ff_input"},
            {"output": "down"},
        ),
    )
    preamble: list[ast.stmt] = []
    loops: list[ast.For] = []
    for event in events:
        event_preamble, event_loops = _inline_invocation(event)
        preamble.extend(event_preamble)
        loops.extend(event_loops)
    function = ast.FunctionDef(
        name="muse_glimmer_ffn_dynamic_batch_source",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="ff_input"),
                ast.arg(arg="gate_up_weight_t"),
                ast.arg(arg="down_weight_t"),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            *preamble,
            *loops,
            ast.Return(
                value=ast.Tuple(
                    elts=[
                        ast.Name(id=name, ctx=ast.Load())
                        for name in (
                            "down",
                            "gate_partial",
                            "gate_up",
                            "activation",
                            "down_partial",
                        )
                    ],
                    ctx=ast.Load(),
                )
            ),
        ],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    source = ast.unparse(module) + "\n"
    filename = "<muse_glimmer_ffn_dynamic_batch_source>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    module_name = "_muse_glimmer_ffn_dynamic_batch_source"
    generated_module = types.ModuleType(module_name)
    namespace = generated_module.__dict__
    namespace.update({
        "torch": torch,
        "hl": hl,
        "HIDDEN": HIDDEN,
        "INTERMEDIATE": INTERMEDIATE,
        "GATE_SPLITS": GATE_SPLITS,
        "ACTIVATION_SPLITS": ACTIVATION_SPLITS,
        "SPLIT_HIDDEN": SPLIT_HIDDEN,
        "SPLIT_INTERMEDIATE": SPLIT_INTERMEDIATE,
        "GATE_N": GATE_N,
        "GATE_REDUCE_N": GATE_REDUCE_N,
        "ACTIVATION_N": ACTIVATION_N,
        "DOWN_N": DOWN_N,
        "DOWN_REDUCE_N": DOWN_REDUCE_N,
        "MAX_BATCH_SIZE": MAX_BATCH_SIZE,
    })
    sys.modules[module_name] = generated_module
    exec(compile(source, filename, "exec"), namespace)
    return namespace["muse_glimmer_ffn_dynamic_batch_source"]


_SOURCE = _persistent_source()
MUSE_DYNAMIC = helion.kernel(
    static_shapes=False,
    autotune_effort="none",
    backend="triton",
    triton_do_not_specialize=True,
)(_SOURCE)
MUSE_STATIC = helion.kernel(
    static_shapes=True,
    autotune_effort="none",
    backend="triton",
)(_SOURCE)


def _compile(
    kernel: Kernel,
    args: tuple[object, ...],
    *,
    block_sizes: tuple[int, ...],
    persistent: bool,
    multiplier: int,
    num_stages: int = 2,
    enable_global_list: bool = True,
    maxnreg: int | None = None,
) -> tuple[CompiledConfig, dict[str, object], str, dict[str, object]]:
    bound = kernel.bind(args)
    values = copy.deepcopy(dict(bound.config_spec.default_config()))
    if len(values["block_sizes"]) != len(block_sizes):
        raise ValueError(
            f"{kernel.__name__}: expected {len(values['block_sizes'])} block "
            f"sizes, received {len(block_sizes)}"
        )
    values["block_sizes"] = list(block_sizes)
    values.update(
        {"num_warps": 1, "num_stages": num_stages, "maxnreg": maxnreg}
    )
    if persistent:
        values.update(
            {
                "pid_type": "persistent_blocked",
                "cross_loop_schedule": "static_pipeline",
                "num_sm_multiplier": multiplier,
            }
        )
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
        }
        return result

    def event_frontier(*call_args: object, **call_kwargs: object):  # noqa: ANN202
        result = original_event_frontier(*call_args, **call_kwargs)
        path_records["event_frontier_proposal"] = {
            "called": True,
            "returned_schedule": result is not None,
            "changed": result is not None and result != call_args[1],
            "segments": None if result is None else len(result.segments),
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
        dependency_graph = call_kwargs["dependency_graph"]
        root_task_orders = call_kwargs["root_task_orders"]
        site_domains = call_kwargs["site_domains"]
        publishable_site_ids = call_kwargs.get("publishable_site_ids")
        prove_nonnegative = call_kwargs.get("prove_nonnegative")
        root_domains = tuple(
            task_order.target_domain for task_order in root_task_orders
        )

        def relation_summary(relation: object) -> dict[str, object]:
            source_domain = relation.source_domain
            target_domain = relation.target_domain
            return {
                "source_axes": list(source_domain.axis_order),
                "source_counts": [
                    [axis, str(count)]
                    for axis, count in source_domain.axis_counts_items
                ],
                "target_axes": list(target_domain.axis_order),
                "target_counts": [
                    [axis, str(count)]
                    for axis, count in target_domain.axis_counts_items
                ],
                "source_axes_affecting_targets": (
                    None
                    if (
                        affecting := relation.source_axes_affecting_targets()
                    )
                    is None
                    else list(affecting)
                ),
                "pieces": [
                    {
                        "source_bounds": [
                            [axis, str(begin), str(end), step]
                            for axis, begin, end, step in piece.source_bounds_items
                        ],
                        "target_ranges": [
                            [axis, str(begin), str(end), step]
                            for axis, begin, end, step in piece.target_ranges
                        ],
                    }
                    for piece in relation.pieces
                ],
            }

        readiness_graph = cross_loop_scheduler.build_readiness_graph(
            dependency_graph,
            root_task_orders=root_task_orders,
            site_domains=site_domains,
            publishable_site_ids=publishable_site_ids,
            prove_nonnegative=prove_nonnegative,
        )
        event_records = []
        for event in readiness_graph.events:
            event_records.append(
                {
                    "event_id": event.event_id,
                    "event_key_rank": len(event.readiness_key_domain.axis_order),
                    "event_key_size": str(event.readiness_key_count_expr),
                    "root_barrier_producer_root": (
                        event.root_barrier_producer_root
                    ),
                    "uniform_fan_in": cross_loop_scheduler._uniform_arrival_count(
                        event.producers
                    ),
                    "fan_in_bounds": cross_loop_scheduler._arrival_count_bounds(
                        event.producers
                    ),
                    "producer_roots": [
                        producer.producer_root for producer in event.producers
                    ],
                    "producer_sites": [
                        producer.producer_site_id for producer in event.producers
                    ],
                    "producer_lowering_supported": [
                        cross_loop_scheduler._supports_readiness_counter_lowering(
                            producer
                        )
                        for producer in event.producers
                    ],
                    "exact_counter_supported": (
                        cross_loop_scheduler._supports_exact_counter_plan_lowering(
                            cross_loop_scheduler.ReadinessCounterPlan(
                                producers=event.producers,
                                consumers=tuple(
                                    consumer
                                    for consumer in event.consumers
                                    if consumer.consumer_site_id is None
                                ),
                            ),
                            root_domains,
                        )
                        if any(
                            consumer.consumer_site_id is None
                            for consumer in event.consumers
                        )
                        else False
                    ),
                    "arrival_count_relations": [
                        (
                            None
                            if producer.arrival_count_by_key is None
                            else relation_summary(producer.arrival_count_by_key)
                        )
                        for producer in event.producers
                    ],
                    "consumer_roots": [
                        consumer.consumer_root for consumer in event.consumers
                    ],
                    "consumer_sites": [
                        consumer.consumer_site_id for consumer in event.consumers
                    ],
                }
            )

        site_by_id = {
            site.site_id: site for site in dependency_graph.execution_sites
        }
        symbolic_dependencies = (
            cross_loop_scheduler.instantiate_symbolic_dependencies(
                dependency_graph,
                root_domains=root_domains,
                site_domains=site_domains,
                prove_nonnegative=prove_nonnegative,
            )
        )
        barrier_diagnostics = []
        root_relations_by_pair: dict[tuple[int, int], list[object]] = {}
        for dependency in symbolic_dependencies:
            pair = (dependency.producer_root, dependency.consumer_root)
            if pair not in result.root_barrier_edges:
                continue
            relation = dependency.producers_by_consumer
            record: dict[str, object] = {
                "dependency_id": dependency.dependency_id,
                "producer_root": dependency.producer_root,
                "consumer_root": dependency.consumer_root,
                "producer_site_id": dependency.producer_site_id,
                "consumer_site_id": dependency.consumer_site_id,
                "kind": str(dependency.kind),
                "exact_relation": relation is not None,
            }
            if relation is None:
                barrier_diagnostics.append(record)
                continue
            record["site_relation"] = relation_summary(relation)
            producer_site = (
                None
                if dependency.producer_site_id is None
                else site_by_id[dependency.producer_site_id]
            )
            consumer_site = (
                None
                if dependency.consumer_site_id is None
                else site_by_id[dependency.consumer_site_id]
            )
            producer_is_root = producer_site is None or producer_site.is_root
            consumer_is_root = consumer_site is None or consumer_site.is_root
            root_relation = relation
            if not producer_is_root:
                root_relation = root_relation.project_target(
                    root_domains[dependency.producer_root]
                )
            if root_relation is not None and not consumer_is_root:
                root_relation = root_relation.project_source(
                    root_domains[dependency.consumer_root]
                )
            record["root_projection_succeeded"] = root_relation is not None
            if root_relation is None:
                barrier_diagnostics.append(record)
                continue
            root_relations_by_pair.setdefault(pair, []).append(root_relation)
            record["root_relation"] = relation_summary(root_relation)
            quotient = root_relation.producer_set_quotient()
            record["producer_set_quotient_succeeded"] = quotient is not None
            if quotient is not None:
                keys_by_consumer, producers_by_key = quotient
                producer = cross_loop_scheduler.ReadinessProducer(
                    producer_root=dependency.producer_root,
                    producers_by_key=producers_by_key,
                )
                record["quotient"] = {
                    "keys_by_consumer": relation_summary(keys_by_consumer),
                    "producers_by_key": relation_summary(producers_by_key),
                    "arrival_count_relation": (
                        None
                        if producer.arrival_count_by_key is None
                        else relation_summary(producer.arrival_count_by_key)
                    ),
                    "uniform_fan_in": (
                        None
                        if producer.arrival_count_by_key is None
                        else producer.arrival_count_by_key.constant_value()
                    ),
                    "counter_lowering_supported": (
                        cross_loop_scheduler._supports_readiness_counter_lowering(
                            producer
                        )
                    ),
                }
            barrier_diagnostics.append(record)
        merged_pair_diagnostics = []
        for pair, relations in sorted(root_relations_by_pair.items()):
            merged = relations[0]
            union_succeeded = True
            for relation in relations[1:]:
                next_merged = merged.union(relation)
                if next_merged is None:
                    union_succeeded = False
                    break
                merged = next_merged
            pair_record: dict[str, object] = {
                "producer_root": pair[0],
                "consumer_root": pair[1],
                "relation_count": len(relations),
                "union_succeeded": union_succeeded,
            }
            if not union_succeeded:
                merged_pair_diagnostics.append(pair_record)
                continue
            pair_record["merged_relation"] = relation_summary(merged)
            used_axes = merged.source_axes_affecting_targets()
            pair_record["source_axes_affecting_targets"] = (
                None if used_axes is None else list(used_axes)
            )
            if used_axes is not None:
                consumer_domain = root_domains[pair[1]]
                consumer_counts = consumer_domain.axis_count_expressions
                consumer_blocks = consumer_domain.block_sizes
                key_domain = CoordinateDomain(
                    axis_order=used_axes,
                    axis_counts_items=tuple(
                        (axis, consumer_counts[axis]) for axis in used_axes
                    ),
                    block_sizes_items=tuple(
                        (axis, consumer_blocks[axis])
                        for axis in used_axes
                        if axis in consumer_blocks
                    ),
                    kind="event",
                )
                keys_by_consumer = type(merged).projection(
                    consumer_domain, key_domain
                )
                factor = (
                    None
                    if keys_by_consumer is None
                    else merged.factor_through(keys_by_consumer)
                )
                pair_record["projection_succeeded"] = (
                    keys_by_consumer is not None
                )
                pair_record["factor_through_succeeded"] = factor is not None
                if factor is not None:
                    factor_producer = cross_loop_scheduler.ReadinessProducer(
                        producer_root=pair[0],
                        producers_by_key=factor,
                    )
                    publication = factor_producer.keys_by_producer
                    arrival_count = factor_producer.arrival_count_by_key
                    pair_record["factor_proof"] = {
                        "counter_lowering_supported": (
                            cross_loop_scheduler._supports_readiness_counter_lowering(
                                factor_producer
                            )
                        ),
                        "publication_derived": publication is not None,
                        "publication_single_valued": (
                            publication is not None
                            and publication.canonical_single_valued() is not None
                        ),
                        "arrival_count_derived": arrival_count is not None,
                        "uniform_fan_in": (
                            None
                            if arrival_count is None
                            else arrival_count.constant_value()
                        ),
                    }
            pair_quotient = merged.producer_set_quotient()
            pair_record["producer_set_quotient_succeeded"] = (
                pair_quotient is not None
            )
            merged_pair_diagnostics.append(pair_record)
        path_records["static_pipeline_plan"] = {
            "readiness_counters": [
                {
                    "event_key_rank": len(plan.readiness_key_domain.axis_order),
                    "event_key_size": str(plan.readiness_key_count_expr),
                    "fan_in": plan.uniform_arrival_count(),
                    "producer_roots": [
                        producer.producer_root for producer in plan.producers
                    ],
                    "producer_task_ranks": [
                        len(producer.producers_by_key.target_domain.axis_order)
                        for producer in plan.producers
                    ],
                    "consumer_roots": [
                        consumer.consumer_root for consumer in plan.consumers
                    ],
                    "consumer_task_ranks": [
                        len(consumer.keys_by_consumer.source_domain.axis_order)
                        for consumer in plan.consumers
                    ],
                    "continuation_consumer_index": (
                        plan.continuation_consumer_index
                    ),
                }
                for plan in result.readiness_counters
            ],
            "root_barrier_edges": sorted(result.root_barrier_edges),
            "worker_schedule_segments": len(result.worker_schedule.segments),
            "readiness_graph_events": event_records,
            "barrier_edge_diagnostics": barrier_diagnostics,
            "merged_barrier_pair_diagnostics": merged_pair_diagnostics,
        }
        return result

    cross_loop_scheduler.build_baseline_worker_schedule = baseline
    cross_loop_scheduler._event_frontier_list_schedule = event_frontier
    cross_loop_scheduler._global_unit_list_schedule = (
        global_list
        if enable_global_list
        else lambda *unused_args, **unused_kwargs: None
    )
    cross_loop_codegen.build_static_pipeline_plan = pipeline_plan
    try:
        compile_start = time.perf_counter()
        code = bound.to_code(config)
        path_records["to_code_seconds"] = time.perf_counter() - compile_start
        binary_start = time.perf_counter()
        compiled = bound.compile_config(config)
        path_records["compile_config_seconds"] = time.perf_counter() - binary_start
        path_records["total_compile_seconds"] = time.perf_counter() - compile_start
    finally:
        cross_loop_scheduler.build_baseline_worker_schedule = original_baseline
        cross_loop_scheduler._event_frontier_list_schedule = original_event_frontier
        cross_loop_scheduler._global_unit_list_schedule = original_global
        cross_loop_codegen.build_static_pipeline_plan = original_pipeline_plan
    return compiled, values, code, path_records


def _compiled_kernels(call: Callable[..., object]) -> tuple[Any, ...]:
    kernels: list[Any] = []
    for value in call.__globals__.values():
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


def _make_weights(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    gate = (
        torch.randn(
            (2 * INTERMEDIATE, HIDDEN),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * (0.5 / HIDDEN**0.5)
    ).T
    down = (
        torch.randn(
            (HIDDEN, INTERMEDIATE),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * (0.5 / INTERMEDIATE**0.5)
    ).T
    return gate, down


def _make_input(batch_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.randn(
        (batch_size, HIDDEN),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )


def _active_outputs(
    outputs: tuple[torch.Tensor, ...], batch_size: int
) -> tuple[torch.Tensor, ...]:
    down, gate_partial, gate_up, activation, down_partial = outputs
    return (
        down[:batch_size],
        gate_partial[:batch_size],
        gate_up[:batch_size],
        activation[:batch_size],
        down_partial[:batch_size],
    )


def _lowering_summary(code: str) -> dict[str, object]:
    return {
        "parameterized_root_loop": "tile_dependency_parameterized_root_task" in code,
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


@torch.inference_mode()
def benchmark(args: argparse.Namespace) -> dict[str, object]:
    batch_sizes = tuple(int(value) for value in args.batch_sizes.split(","))
    if not batch_sizes or any(value < 1 or value > MAX_BATCH_SIZE for value in batch_sizes):
        raise ValueError(f"batch sizes must be in [1, {MAX_BATCH_SIZE}]")
    if args.exemplar_batch not in batch_sizes:
        raise ValueError("exemplar batch must be one of batch sizes")
    gate_weight, down_weight = _make_weights(args.seed)
    inputs = {
        batch: _make_input(batch, args.seed + batch) for batch in batch_sizes
    }
    exemplar_args = (
        inputs[args.exemplar_batch],
        gate_weight,
        down_weight,
    )

    dynamic_call, dynamic_config, dynamic_code, dynamic_path = _compile(
        MUSE_DYNAMIC,
        exemplar_args,
        block_sizes=(
            GATE_N,
            args.gate_block_k,
            GATE_REDUCE_N,
            ACTIVATION_N,
            DOWN_N,
            args.down_block_k,
            DOWN_REDUCE_N,
        ),
        persistent=True,
        multiplier=args.multiplier,
    )
    if args.compile_only:
        return {
            "workload": {
                "name": "Muse-Glimmer-30B BF16 FFN",
                "batch_sizes": batch_sizes,
                "hidden": HIDDEN,
                "intermediate": INTERMEDIATE,
            },
            "compile_only": True,
            "dynamic_binary": _binary_summary(dynamic_call),
            "dynamic_lowering": _lowering_summary(dynamic_code),
            "dynamic_scheduler_path": dynamic_path,
            "resources": {"persistent_dynamic": _resources(dynamic_call)},
            "config": dynamic_config,
        }
    component_specs = (
        ("gate_main", dynamic_gate_up_splitk, (GATE_N, args.gate_block_k)),
        ("gate_reduce", dynamic_gate_up_reduce, (GATE_REDUCE_N,)),
        ("activation", dynamic_silu_and_mul, (ACTIVATION_N,)),
        ("down_main", dynamic_down_splitk, (DOWN_N, args.down_block_k)),
        ("down_reduce", dynamic_down_reduce, (DOWN_REDUCE_N,)),
    )

    # Build the five dynamic standalone launchers once using exemplar tensors.
    gate_call, gate_config, _gate_code, _ = _compile(
        component_specs[0][1],
        (inputs[args.exemplar_batch], gate_weight),
        block_sizes=component_specs[0][2],
        persistent=False,
        multiplier=args.multiplier,
    )
    exemplar_gate = gate_call(inputs[args.exemplar_batch], gate_weight)
    gate_reduce_call, gate_reduce_config, _gate_reduce_code, _ = _compile(
        component_specs[1][1],
        (exemplar_gate, inputs[args.exemplar_batch]),
        block_sizes=component_specs[1][2],
        persistent=False,
        multiplier=args.multiplier,
    )
    exemplar_gate_up = gate_reduce_call(exemplar_gate, inputs[args.exemplar_batch])
    activation_call, activation_config, _activation_code, _ = _compile(
        component_specs[2][1],
        (exemplar_gate_up, inputs[args.exemplar_batch]),
        block_sizes=component_specs[2][2],
        persistent=False,
        multiplier=args.multiplier,
    )
    exemplar_activation = activation_call(exemplar_gate_up, inputs[args.exemplar_batch])
    down_call, down_config, _down_code, _ = _compile(
        component_specs[3][1],
        (exemplar_activation, down_weight, inputs[args.exemplar_batch]),
        block_sizes=component_specs[3][2],
        persistent=False,
        multiplier=args.multiplier,
    )
    exemplar_down = down_call(
        exemplar_activation, down_weight, inputs[args.exemplar_batch]
    )
    down_reduce_call, down_reduce_config, _down_reduce_code, _ = _compile(
        component_specs[4][1],
        (exemplar_down, inputs[args.exemplar_batch]),
        block_sizes=component_specs[4][2],
        persistent=False,
        multiplier=args.multiplier,
    )

    static_calls: dict[int, CompiledConfig] = {}
    static_configs: dict[int, dict[str, object]] = {}
    static_paths: dict[int, dict[str, object]] = {}
    static_lowering: dict[int, dict[str, object]] = {}
    captures: dict[int, tuple[tuple[str, torch.cuda.CUDAGraph, object], ...]] = {}
    correctness: dict[int, dict[str, object]] = {}

    def standalone(batch_size: int) -> tuple[torch.Tensor, ...]:
        ff_input = inputs[batch_size]
        gate_partial = gate_call(ff_input, gate_weight)
        gate_up = gate_reduce_call(gate_partial, ff_input)
        activation = activation_call(gate_up, ff_input)
        down_partial = down_call(activation, down_weight, ff_input)
        down = down_reduce_call(down_partial, ff_input)
        return down, gate_partial, gate_up, activation, down_partial

    expected_dynamic_binary: dict[str, object] | None = None
    for batch_size in batch_sizes:
        kernel_args = (inputs[batch_size], gate_weight, down_weight)
        static_call: CompiledConfig | None = None
        if not args.skip_static:
            static_call, static_config, static_code, static_path = _compile(
                MUSE_STATIC,
                kernel_args,
                block_sizes=(
                    GATE_N,
                    args.gate_block_k,
                    GATE_REDUCE_N,
                    ACTIVATION_N,
                    DOWN_N,
                    args.down_block_k,
                    DOWN_REDUCE_N,
                ),
                persistent=True,
                multiplier=args.multiplier,
                enable_global_list=False,
            )
            static_calls[batch_size] = static_call
            static_configs[batch_size] = static_config
            static_paths[batch_size] = static_path
            static_lowering[batch_size] = _lowering_summary(static_code)
        expected = _active_outputs(standalone(batch_size), batch_size)
        dynamic = _active_outputs(dynamic_call(*kernel_args), batch_size)
        static = (
            None
            if static_call is None
            else _active_outputs(static_call(*kernel_args), batch_size)
        )
        torch.cuda.synchronize()
        max_abs_dynamic = 0.0
        max_abs_static = 0.0
        for output_index, (dynamic_value, expected_value) in enumerate(
            zip(dynamic, expected, strict=True)
        ):
            torch.testing.assert_close(dynamic_value, expected_value, atol=5e-4, rtol=0.1)
            max_abs_dynamic = max(
                max_abs_dynamic,
                float((dynamic_value.float() - expected_value.float()).abs().max()),
            )
            if static is not None:
                static_value = static[output_index]
                torch.testing.assert_close(
                    static_value, expected_value, atol=5e-4, rtol=0.1
                )
                max_abs_static = max(
                    max_abs_static,
                    float(
                        (static_value.float() - expected_value.float()).abs().max()
                    ),
                )
        correctness[batch_size] = {
            "dynamic_vs_standalone_max_abs": max_abs_dynamic,
            "dynamic_vs_standalone_bit_exact": all(
                starmap(torch.equal, zip(dynamic, expected, strict=True))
            ),
        }
        if static is not None:
            correctness[batch_size].update(
                {
                    "static_vs_standalone_max_abs": max_abs_static,
                    "static_vs_standalone_bit_exact": all(
                        starmap(torch.equal, zip(static, expected, strict=True))
                    ),
                }
            )
        current_binary = _binary_summary(dynamic_call)
        if expected_dynamic_binary is None:
            expected_dynamic_binary = current_binary
        elif current_binary != expected_dynamic_binary:
            raise AssertionError(
                f"dynamic B{batch_size} created a new binary: {current_binary}"
            )

        calls: tuple[tuple[str, Callable[[], object]], ...] = (
            ("standalone_five_launch", lambda batch=batch_size: standalone(batch)),
            *(
                ()
                if static_call is None
                else (
                    (
                        "persistent_static",
                        lambda batch=batch_size, call=static_call: call(
                            inputs[batch], gate_weight, down_weight
                        ),
                    ),
                )
            ),
            (
                "persistent_dynamic",
                lambda batch=batch_size: dynamic_call(
                    inputs[batch], gate_weight, down_weight
                ),
            ),
        )
        captures[batch_size] = tuple(
            (name, *capture_cuda_graph(call)) for name, call in calls
        )

    thermal_warmup(args.warmup_ms)
    timings: dict[int, dict[str, float]] = {}
    for batch_size, batch_captures in captures.items():
        elapsed = bench_pre_captured_cudagraphs(
            [graph.replay for _name, graph, _output in batch_captures],
            rep=args.repetitions,
        )
        timings[batch_size] = {
            name: milliseconds * 1000
            for (name, _graph, _output), milliseconds in zip(
                batch_captures, elapsed, strict=True
            )
        }

    dynamic_binary = _binary_summary(dynamic_call)
    if dynamic_binary["specialization_count"] != 1:
        raise AssertionError(f"dynamic kernel did not retain one specialization: {dynamic_binary}")
    return {
        "workload": {
            "name": "Muse-Glimmer-30B BF16 FFN",
            "boundary": "five stages / five standalone launches / one persistent launch",
            "batch_sizes": batch_sizes,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "gate_splits": GATE_SPLITS,
            "activation_splits": ACTIVATION_SPLITS,
            "gate_geometry": f"N{GATE_N}xK{args.gate_block_k}",
            "gate_reduce_n": GATE_REDUCE_N,
            "activation_n": ACTIVATION_N,
            "down_geometry": f"N{DOWN_N}xK{args.down_block_k}",
            "down_reduce_n": DOWN_REDUCE_N,
            "scratch_batch_capacity": MAX_BATCH_SIZE,
        },
        "timings_us_cold_l2": timings,
        "correctness": correctness,
        "one_dynamic_cubin_across_batch_sizes": True,
        "dynamic_binary": dynamic_binary,
        "dynamic_lowering": _lowering_summary(dynamic_code),
        "dynamic_scheduler_path": dynamic_path,
        "static_lowering": static_lowering,
        "static_scheduler_path": static_paths,
        "resources": {
            "persistent_dynamic": _resources(dynamic_call),
            "persistent_static": {
                batch: _resources(call) for batch, call in static_calls.items()
            },
            "standalone": {
                "gate_main": _resources(gate_call),
                "gate_reduce": _resources(gate_reduce_call),
                "activation": _resources(activation_call),
                "down_main": _resources(down_call),
                "down_reduce": _resources(down_reduce_call),
            },
        },
        "configs": {
            "persistent_dynamic": dynamic_config,
            "persistent_static": static_configs,
            "standalone": {
                "gate_main": gate_config,
                "gate_reduce": gate_reduce_config,
                "activation": activation_config,
                "down_main": down_config,
                "down_reduce": down_reduce_config,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--exemplar-batch", type=int, default=2)
    parser.add_argument("--gate-block-k", type=int, default=128)
    parser.add_argument("--down-block-k", type=int, default=128)
    parser.add_argument("--multiplier", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmup-ms", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-static", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(args)
    rendered = json.dumps(result, indent=2, sort_keys=True, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
