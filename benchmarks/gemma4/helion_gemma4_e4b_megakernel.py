# ruff: noqa: ANN001, ANN202
# pyrefly: ignore-errors
"""Single-kernel Gemma 4 E4B decode-layer experiment.

The composite Helion function is assembled from the optimized separate-kernel
sources in ``helion_gemma4_e4b_layer.py``. Local names are alpha-renamed and
ordinary computation loops are copied unchanged; two residual joins use the
documented tiled variants below. Run this file with the tile-dependency-schedule
Helion checkout first on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import inspect
import json
import linecache
import math  # noqa: F401 - used by generated kernel source
from pathlib import Path
import textwrap
from typing import Protocol

from benchmarks.gemma4.common import Gemma4E4BShape
from benchmarks.gemma4.common import allocate_layer
from benchmarks.gemma4.common import benchmark_interleaved
from benchmarks.gemma4.common import capture
from benchmarks.gemma4.common import layer_reference
from benchmarks.gemma4.common import require_idle_visible_gpu
from benchmarks.gemma4.common import variant_name
from benchmarks.gemma4.common import visible_gpu_pids
import benchmarks.gemma4.helion_gemma4_e4b_layer as separate
import torch

import helion
import helion.language as hl

_gelu_tanh = separate._gelu_tanh


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def post_ff_residual_tiled(
    down: torch.Tensor,
    residual: torch.Tensor,
    post_ff_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Tile the output while redundantly computing the short RMS reduction."""
    m, hidden = down.size()
    hl.specialize(hidden)
    output = torch.empty_like(down)
    for tile_m, tile_n in hl.tile([m, hidden], block_size=[1, 1024]):
        values = down[tile_m, :].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(values * values, dim=-1) + eps)
        output_values = down[tile_m, tile_n].to(torch.float32)
        normalized = (output_values * inv_rms[:, None]).to(down.dtype)
        output[tile_m, tile_n] = (
            normalized * post_ff_weight[None, tile_n] + residual[tile_m, tile_n]
        )
    return output


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def final_ple_norm_residual_scale_tiled(
    ple_projection: torch.Tensor,
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    layer_scalar: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Tile the output while redundantly computing the short RMS reduction."""
    m, hidden_size = ple_projection.size()
    hl.specialize(hidden_size)
    output = torch.empty_like(hidden)
    for tile_m, tile_n in hl.tile([m, hidden_size], block_size=[1, 1024]):
        values = ple_projection[tile_m, :].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(values * values, dim=-1) + eps)
        output_values = ple_projection[tile_m, tile_n].to(torch.float32)
        normalized = (output_values * inv_rms[:, None]).to(ple_projection.dtype)
        scalar = hl.load(layer_scalar, [])
        output[tile_m, tile_n] = (
            normalized * norm_weight[None, tile_n] + hidden[tile_m, tile_n]
        ) * scalar
    return output


@dataclasses.dataclass(frozen=True)
class _Invocation:
    prefix: str
    kernel: _KernelWithFunction
    arguments: dict[str, str]
    outputs: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class _Bridge:
    source: str


class _KernelWithFunction(Protocol):
    fn: object


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


def _kernel_function_ast(kernel: _KernelWithFunction) -> ast.FunctionDef:
    source = textwrap.dedent(inspect.getsource(kernel.fn))
    module = ast.parse(source)
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    assert len(functions) == 1
    return functions[0]


def _inline_invocation(invocation: _Invocation) -> tuple[list[ast.stmt], list[ast.For]]:
    function = _kernel_function_ast(invocation.kernel)
    parameters = [argument.arg for argument in function.args.args]
    assert set(parameters) == set(invocation.arguments)

    assigned = _AssignedNames()
    for statement in function.body:
        assigned.visit(statement)
    rename = {
        name: invocation.outputs.get(name, f"__gemma4_{invocation.prefix}_{name}")
        for name in set(parameters) | assigned.names
    }
    transformer = _RenameNames(rename)
    preamble: list[ast.stmt] = [
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
    assert loops, invocation.prefix
    return preamble, loops


def _layer_events(kv_shared: bool) -> list[_Invocation | _Bridge]:
    events: list[_Invocation | _Bridge] = [
        _Invocation(
            "input_norm",
            separate.rms_norm,
            {"x": "hidden_states", "weight": "input_norm_weight", "eps": "eps"},
            {"out": "input_norm"},
        ),
        _Invocation(
            "q_projection" if kv_shared else "qkv_projection",
            separate.bf16_mm,
            {
                "x": "input_norm",
                "weight": "qkv_weight[:q_width]" if kv_shared else "qkv_weight",
            },
            {"out": "projected_qkv"},
        ),
    ]
    if kv_shared:
        events.extend(
            (
                _Invocation(
                    "q_norm_rope",
                    separate.q_norm_rope,
                    {
                        "query": "projected_qkv",
                        "q_weight": "q_norm_weight",
                        "cos_sin": "cos_sin",
                        "position": "position",
                        "num_q_heads": "q_heads",
                        "head_dim": "head_dim",
                        "eps": "eps",
                    },
                ),
                _Bridge("query = projected_qkv.view(1, q_heads, head_dim)"),
            )
        )
    else:
        events.extend(
            (
                _Invocation(
                    "qkv_norm_rope_cache",
                    separate.qkv_norm_rope_cache,
                    {
                        "qkv": "projected_qkv",
                        "q_weight": "q_norm_weight",
                        "k_weight": "k_norm_weight",
                        "cos_sin": "cos_sin",
                        "position": "position",
                        "kv_cache": "kv_cache",
                        "slot_mapping": "slot_mapping",
                        "num_q_heads": "q_heads",
                        "num_kv_heads": "kv_heads",
                        "head_dim": "head_dim",
                        "block_size": "cache_block",
                        "eps": "eps",
                    },
                ),
                _Bridge(
                    "query = projected_qkv[:, :q_width].view(1, q_heads, head_dim)"
                ),
            )
        )
    events.extend(
        (
            _Invocation(
                "attention_split",
                separate.paged_attention_split,
                {
                    "query": "query",
                    "kv_cache": "kv_cache",
                    "block_table": "block_table",
                    "context": "context",
                    "attention_context": "attention_context",
                    "block_size": "cache_block",
                    "q_per_kv": "q_per_kv",
                    "splits": "attention_splits",
                },
                {"partial_out": "partial_out", "partial_lse": "partial_lse"},
            ),
            _Invocation(
                "attention_merge",
                separate.merge_attention,
                {"partial_out": "partial_out", "partial_lse": "partial_lse"},
                {"output": "attention"},
            ),
            _Bridge("attention_view = attention.view(1, q_heads, head_dim)"),
            _Bridge("attention_flat = attention_view.view(1, q_width)"),
            _Invocation(
                "o_projection",
                separate.bf16_mm,
                {"x": "attention_flat", "weight": "o_weight"},
                {"out": "attention_out"},
            ),
            _Invocation(
                "post_attention",
                separate.post_attention_residual_pre_ff_norm,
                {
                    "attention_out": "attention_out",
                    "residual": "hidden_states",
                    "post_attention_weight": "post_attention_norm_weight",
                    "pre_ff_weight": "pre_ff_norm_weight",
                    "eps": "eps",
                },
                {"updated_residual": "residual", "ff_input": "ff_input"},
            ),
            _Invocation(
                "gate_up_projection",
                separate.bf16_mm,
                {"x": "ff_input", "weight": "gate_up_weight"},
                {"out": "gate_up"},
            ),
            _Invocation(
                "geglu",
                separate.geglu,
                {"gate_up": "gate_up"},
                {"output": "activation"},
            ),
            _Invocation(
                "down_projection",
                separate.bf16_mm,
                {"x": "activation", "weight": "down_weight"},
                {"out": "down"},
            ),
            _Invocation(
                "post_ff",
                post_ff_residual_tiled,
                {
                    "down": "down",
                    "residual": "residual",
                    "post_ff_weight": "post_ff_norm_weight",
                    "eps": "eps",
                },
                {"output": "hidden"},
            ),
            _Invocation(
                "ple_gate",
                separate.ple_gate_gelu_mul,
                {
                    "hidden": "hidden",
                    "gate_weight": "ple_gate_weight",
                    "per_layer_input": "per_layer_input",
                },
                {"output": "ple_input"},
            ),
            _Invocation(
                "ple_projection",
                separate.bf16_mm,
                {"x": "ple_input", "weight": "ple_projection_weight"},
                {"out": "ple_projection"},
            ),
            _Invocation(
                "final",
                final_ple_norm_residual_scale_tiled,
                {
                    "ple_projection": "ple_projection",
                    "hidden": "hidden",
                    "norm_weight": "post_ple_norm_weight",
                    "layer_scalar": "layer_scalar",
                    "eps": "eps",
                },
                {"output": "output"},
            ),
        )
    )
    return events


def _compose_layer_source(kv_shared: bool) -> str:
    preamble: list[ast.stmt] = []
    scheduled_statements: list[ast.stmt] = []
    for event in _layer_events(kv_shared):
        if isinstance(event, _Bridge):
            preamble.extend(ast.parse(textwrap.dedent(event.source)).body)
        else:
            event_preamble, event_loops = _inline_invocation(event)
            preamble.extend(event_preamble)
            scheduled_statements.extend(event_loops)

    arguments = [
        "hidden_states",
        "per_layer_input",
        "input_norm_weight",
        "post_attention_norm_weight",
        "pre_ff_norm_weight",
        "post_ff_norm_weight",
        "post_ple_norm_weight",
        "q_norm_weight",
        "k_norm_weight",
        "qkv_weight",
        "o_weight",
        "gate_up_weight",
        "down_weight",
        "ple_gate_weight",
        "ple_projection_weight",
        "layer_scalar",
        "position",
        "cos_sin",
        "kv_cache",
        "block_table",
        "slot_mapping",
        "hidden_size",
        "intermediate",
        "q_heads",
        "kv_heads",
        "q_per_kv",
        "head_dim",
        "context",
        "attention_context",
        "cache_block",
        "attention_splits",
        "eps",
    ]
    function = ast.FunctionDef(
        name=f"gemma4_e4b_{'shared' if kv_shared else 'nonshared'}_megakernel_source",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in arguments],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            ast.Assign(
                targets=[ast.Name(id="q_width", ctx=ast.Store())],
                value=ast.parse("q_heads * head_dim", mode="eval").body,
            ),
            *preamble,
            *scheduled_statements,
            ast.Return(
                value=ast.Tuple(
                    elts=[
                        ast.Name(id=name, ctx=ast.Load())
                        for name in (
                            "output",
                            "query",
                            "attention_view",
                            "residual",
                            "ff_input",
                            "gate_up",
                            "activation",
                            "down",
                            "hidden",
                            "ple_input",
                            "ple_projection",
                        )
                    ],
                    ctx=ast.Load(),
                )
            ),
        ],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    return ast.unparse(module) + "\n"


def _build_megakernel(kv_shared: bool):
    source = _compose_layer_source(kv_shared)
    filename = str(
        Path(__file__).with_name(
            f"_generated_gemma4_{'shared' if kv_shared else 'nonshared'}_megakernel.py"
        )
    )
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = globals()
    exec(compile(source, filename, "exec"), namespace)
    function = namespace[
        f"gemma4_e4b_{'shared' if kv_shared else 'nonshared'}_megakernel_source"
    ]
    return (
        helion.kernel(
            static_shapes=True,
            autotune_effort="none",
            backend="triton",
            tile_dependency_schedule=helion.TileDependencySchedule(),
        )(function),
        source,
    )


NONSHARED_MEGAKERNEL, NONSHARED_SOURCE = _build_megakernel(False)
SHARED_MEGAKERNEL, SHARED_SOURCE = _build_megakernel(True)


def _megakernel_args(tensors, shape, geometry, splits):
    return (
        tensors["hidden_states"],
        tensors["per_layer_input"],
        tensors["input_norm_weight"],
        tensors["post_attention_norm_weight"],
        tensors["pre_ff_norm_weight"],
        tensors["post_ff_norm_weight"],
        tensors["post_ple_norm_weight"],
        tensors["q_norm_weight"],
        tensors["k_norm_weight"],
        tensors["qkv_weight"],
        tensors["o_weight"],
        tensors["gate_up_weight"],
        tensors["down_weight"],
        tensors["ple_gate_weight"],
        tensors["ple_proj_weight"],
        tensors["layer_scalar"],
        tensors["position"],
        tensors["cos_sin"],
        tensors["kv_cache"],
        tensors["block_table"],
        tensors["slot_mapping"],
        shape.hidden,
        shape.intermediate,
        shape.q_heads,
        shape.kv_heads,
        shape.q_heads // shape.kv_heads,
        geometry.head_dim,
        shape.context,
        geometry.attention_context,
        shape.block_size,
        splits,
        shape.eps,
    )


def _megakernel_config(bound, args, geometry):
    values = dict(bound.config_spec.default_config())
    if args.config_mode == "matched":
        configs = json.loads(Path(args.config_path).read_text())
        q_projection = configs[
            f"{'q_mm' if geometry.kv_shared else 'qkv_mm'}_hd{geometry.head_dim}"
        ]
        o_projection = configs[f"o_mm_hd{geometry.head_dim}"]
        gate_projection = configs["gate_up_mm"]
        down_projection = configs["down_mm"]
        ple_gate = configs["ple_gate_gelu_mul"]
        block_size_by_id = {
            3: q_projection["block_sizes"][0],
            4: q_projection["block_sizes"][1],
            6: 1,
            10: 4,
            12: min(
                args.attention_block,
                geometry.attention_context
                // (
                    args.full_splits
                    if geometry.layer_type == "full"
                    else args.sliding_splits
                ),
            ),
            14: 1,
            15: (
                args.full_splits
                if geometry.layer_type == "full"
                else args.sliding_splits
            ),
            17: o_projection["block_sizes"][0],
            18: o_projection["block_sizes"][1],
            21: gate_projection["block_sizes"][0],
            22: gate_projection["block_sizes"][1],
            24: 256,
            26: down_projection["block_sizes"][0],
            27: down_projection["block_sizes"][1],
            31: ple_gate["block_sizes"][0],
            32: ple_gate["block_sizes"][1],
            34: 32,
            35: 32,
        }
        values["block_sizes"] = [
            block_size_by_id.get(spec.block_id, value)
            for spec, value in zip(
                bound.config_spec.block_sizes,
                values["block_sizes"],
                strict=True,
            )
        ]
        values["loop_orders"] = [
            q_projection["loop_orders"][0],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1],
            o_projection["loop_orders"][0],
            gate_projection["loop_orders"][0],
            [0, 1],
            down_projection["loop_orders"][0],
            [0, 1],
            ple_gate["loop_orders"][0],
            [0, 1],
            [0, 1],
        ]
        values["l2_groupings"] = [
            q_projection["l2_groupings"][0],
            1,
            1,
            1,
            o_projection["l2_groupings"][0],
            gate_projection["l2_groupings"][0],
            1,
            down_projection["l2_groupings"][0],
            1,
            ple_gate["l2_groupings"][0],
            1,
            1,
        ]

        range_choices = {
            (2, 3): (q_projection, 0),
            (4,): (q_projection, 1),
            (16, 17): (o_projection, 0),
            (18,): (o_projection, 1),
            (20, 21): (gate_projection, 0),
            (22,): (gate_projection, 1),
            (25, 26): (down_projection, 0),
            (27,): (down_projection, 1),
            (30, 31): (ple_gate, 0),
            (32,): (ple_gate, 1),
        }
        for key in (
            "range_num_stages",
            "range_unroll_factors",
            "range_multi_buffers",
            "range_flattens",
        ):
            updated = []
            for spec, value in zip(
                getattr(bound.config_spec, key), values[key], strict=True
            ):
                choice = range_choices.get(tuple(spec.block_ids))
                if choice is None:
                    updated.append(value)
                    continue
                stage_config, index = choice
                updated.append(stage_config[key][index])
            values[key] = updated

    values.update(
        {
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": args.worker_multiplier,
            "num_warps": args.num_warps
            if args.num_warps is not None
            else (4 if geometry.layer_type == "full" else 2),
            "num_stages": args.kernel_stages
            if args.kernel_stages is not None
            else (3 if geometry.layer_type == "full" else 4),
        }
    )
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return config


def _helion_resources(compiled_wrapper):
    kernels = []
    for value in compiled_wrapper.__globals__.values():
        device_caches = getattr(value, "device_caches", None)
        if not device_caches or torch.cuda.current_device() not in device_caches:
            continue
        kernels.extend(device_caches[torch.cuda.current_device()][0].values())
    if len(kernels) != 1:
        raise RuntimeError(f"expected one compiled Helion kernel, found {len(kernels)}")
    kernel = kernels[0]
    return {
        "registers": kernel.n_regs,
        "spills": kernel.n_spills,
        "shared": kernel.metadata.shared,
    }


def _assert_close(name, actual, expected, *, atol=2e-1, rtol=8e-2):
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
    maximum = float((actual.float() - expected.float()).abs().max().item())
    print(f"megakernel_correctness {name} max_abs={maximum:.6f}", flush=True)


def run(args) -> None:
    require_idle_visible_gpu()
    shape = Gemma4E4BShape(context=args.context, block_size=args.block_size)
    geometry = shape.layer_geometry(args.layer)
    splits = args.full_splits if geometry.layer_type == "full" else args.sliding_splits
    kernel = SHARED_MEGAKERNEL if geometry.kv_shared else NONSHARED_MEGAKERNEL
    source = SHARED_SOURCE if geometry.kv_shared else NONSHARED_SOURCE
    if args.print_source:
        print(source)
        return

    tensors = allocate_layer(shape, geometry, args.seed)
    reference = layer_reference(tensors, shape, geometry)
    kernel_args = _megakernel_args(tensors, shape, geometry, splits)
    bound = kernel.bind(kernel_args)
    host_function = bound.host_function
    assert host_function is not None
    config = _megakernel_config(bound, args, geometry)
    print(
        "SOURCE_ADAPTATIONS",
        [
            "post-FF residual RMSNorm is output-tiled with a redundant reduction",
            "final PLE RMSNorm/residual is output-tiled with a redundant reduction",
            "q_per_kv is passed directly to preserve static task geometry",
            f"attention reduction tiles are capped at {args.attention_block}",
        ],
        flush=True,
    )
    if args.dump_config:
        print("MEGAKERNEL_CONFIG", dict(config), flush=True)
        print("ROOT_BLOCK_IDS", host_function.device_ir.grid_block_ids, flush=True)
        dependency_plan = host_function.device_ir.cross_loop_dependency_plan
        assert dependency_plan is not None
        print("DEPENDENCY_EVENTS", dependency_plan.events, flush=True)
        print("DEPENDENCY_WAITS", dependency_plan.waits, flush=True)
        print(
            "BLOCK_SPECS",
            [
                {
                    "block_id": spec.block_id,
                    "size_hint": spec.size_hint,
                    "min_size": spec.min_size,
                    "max_size": spec.max_size,
                }
                for spec in bound.config_spec.block_sizes
            ],
            flush=True,
        )
        print(
            "RANGE_BLOCK_IDS",
            [spec.block_ids for spec in bound.config_spec.range_num_stages],
            flush=True,
        )
    if args.print_lowered:
        print(bound.to_triton_code(config, output_origin_lines=True))
        return
    if args.inspect_only:
        return
    compiled = bound.compile_config(config)
    outputs = compiled(*kernel_args)
    torch.cuda.synchronize()
    _assert_close("output", outputs[0], reference["output"])
    _assert_close("query", outputs[1], reference["query"], atol=8e-2, rtol=4e-2)
    _assert_close(
        "attention", outputs[2], reference["attention"], atol=1.5e-1, rtol=6e-2
    )
    if not geometry.kv_shared:
        slot = int(tensors["slot_mapping"][0].item())
        cache_block = slot // shape.block_size
        cache_offset = slot % shape.block_size
        _assert_close(
            "kv_cache_slot",
            tensors["kv_cache"][cache_block, cache_offset],
            reference["kv_cache"][cache_block, cache_offset],
            atol=8e-2,
            rtol=4e-2,
        )
    print("MEGAKERNEL_RESOURCES", _helion_resources(compiled), flush=True)
    print(
        "MEGAKERNEL_COMPILED",
        {
            "variant": variant_name(geometry),
            "roots": len(host_function.device_ir.root_ids),
            "tile_dependencies": len(
                host_function.device_ir.cross_loop_dependency_plan.edges
                if host_function.device_ir.cross_loop_dependency_plan is not None
                else ()
            ),
            "implicit_phase_starts": sorted(
                host_function.device_ir.tile_dependency_schedule.implicit_phase_starts
            ),
            "pid_type": config.pid_type,
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
            "worker_multiplier": config.num_sm_multiplier,
            "attention_block": args.attention_block,
            "attention_splits": splits,
        },
        flush=True,
    )
    if args.smoke and not args.benchmark:
        return
    if args.timing_only:
        graph, _ = capture(lambda: compiled(*kernel_args))
        pids = visible_gpu_pids()
        timings = benchmark_interleaved(
            {"helion_megakernel": graph.replay},
            args.repeats,
            args.batch_replays,
        )
        if visible_gpu_pids() != pids:
            raise RuntimeError("GPU process set changed during benchmark")
        print("TIMINGS", timings, flush=True)
        return

    config_path = Path(args.config_path)
    configs = json.loads(config_path.read_text()) if config_path.exists() else {}
    baseline_tensors = allocate_layer(shape, geometry, args.seed)
    built = separate.build_layer(
        args,
        baseline_tensors,
        shape,
        geometry,
        configs,
        config_path,
    )
    baseline_output = built["launch_optimized"]()
    torch.cuda.synchronize()
    _assert_close("separate_output", baseline_output, reference["output"])

    megakernel_graph, megakernel_output = capture(lambda: compiled(*kernel_args))
    megakernel_graph.replay()
    torch.cuda.synchronize()
    _assert_close("graph_output", megakernel_output[0], reference["output"])
    separate_graph, separate_output = capture(built["launch_optimized"])
    separate_graph.replay()
    torch.cuda.synchronize()
    _assert_close("separate_graph_output", separate_output, reference["output"])
    pids = visible_gpu_pids()
    timings = benchmark_interleaved(
        {
            "helion_megakernel": megakernel_graph.replay,
            "helion_separate_optimized": separate_graph.replay,
        },
        args.repeats,
        args.batch_replays,
    )
    if visible_gpu_pids() != pids:
        raise RuntimeError("GPU process set changed during benchmark")
    print(
        "RESULT_JSON",
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "helion_module": helion.__file__,
                "layer": args.layer,
                "variant": variant_name(geometry),
                "resources": _helion_resources(compiled),
                "timings": timings,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--sliding-splits", type=int, default=16)
    parser.add_argument("--full-splits", type=int, default=64)
    parser.add_argument("--attention-block", type=int, default=32)
    parser.add_argument("--worker-multiplier", type=int, default=2)
    parser.add_argument("--num-warps", type=int)
    parser.add_argument("--kernel-stages", type=int)
    parser.add_argument(
        "--config-mode", choices=("default", "matched"), default="matched"
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-replays", type=int, default=20)
    parser.add_argument(
        "--config-path",
        default="benchmarks/gemma4/gemma4_e4b_b200_configs.json",
    )
    parser.add_argument("--tune", nargs="*", default=[])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--print-source", action="store_true")
    parser.add_argument("--print-lowered", action="store_true")
    parser.add_argument("--dump-config", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
