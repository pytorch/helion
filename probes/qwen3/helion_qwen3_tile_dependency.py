# ruff: noqa: ANN001, ANN202
"""Shared-source Qwen3 layer probe for TileDependency compiler lowering.

The persistent function is assembled from the original separate Helion kernel
ASTs. Loop bodies are alpha-renamed but otherwise copied verbatim, so the
separate and persistent paths cannot silently drift in arithmetic or numerics.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import inspect
import linecache
import math  # noqa: F401 - used by the generated kernel's module globals
from pathlib import Path
import textwrap
from typing import Callable
from typing import Protocol

import torch

from probes.common import benchmark_interleaved
from probes.common import capture
from probes.common import require_idle_visible_gpu
from probes.common import visible_gpu_pids
from probes.qwen3.helion_qwen3_layer_baseline import allocate as allocate_layer
from probes.qwen3.helion_qwen3_layer_baseline import block_fp8_mm
from probes.qwen3.helion_qwen3_layer_baseline import fused_qk_norm_rope
from probes.qwen3.helion_qwen3_layer_baseline import merge_attention_splits
from probes.qwen3.helion_qwen3_layer_baseline import paged_gqa_decode_attention_split
from probes.qwen3.helion_qwen3_layer_baseline import per_token_group_fp8_quant
from probes.qwen3.helion_qwen3_layer_baseline import reshape_and_cache_flash
from probes.qwen3.helion_qwen3_layer_baseline import rms_norm_per_block_quant
from probes.qwen3.helion_qwen3_layer_baseline import silu_and_mul_per_block_quant

import helion
import helion.language as hl  # noqa: F401 - used by the generated kernel


def build_helion_reference(
    args, tensors
) -> tuple[Callable[[], object], dict[str, torch.Tensor]]:
    """Build the local same-source separate-kernel baseline."""
    from probes.qwen3.helion_qwen3_granular_tile_dependency import (
        _build_helion_reference,
    )

    return _build_helion_reference(args, tensors)


FP8_MAX = 448.0
FP8_MIN = -448.0
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)


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
    function = kernel.fn
    source = textwrap.dedent(inspect.getsource(function))
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
        name: invocation.outputs.get(name, f"__td_{invocation.prefix}_{name}")
        for name in set(parameters) | assigned.names
    }
    transformer = _RenameNames(rename)

    preamble: list[ast.stmt] = []
    for parameter in parameters:
        preamble.append(
            ast.Assign(
                targets=[ast.Name(id=rename[parameter], ctx=ast.Store())],
                value=ast.parse(invocation.arguments[parameter], mode="eval").body,
            )
        )

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


def _compose_qwen3_layer_source() -> str:
    qkv_width = "(q_heads + 2 * kv_heads) * head_dim"
    events: list[_Invocation | _Bridge] = [
        _Invocation(
            "pre",
            rms_norm_per_block_quant,
            {
                "result": "pre_q",
                "input": "hidden_states",
                "weight": "pre_weight",
                "scale": "pre_scale",
                "epsilon": "eps",
                "scale_ub": "None",
                "residual": "residual",
                "group_size": "group",
                "is_scale_transposed": "False",
            },
        ),
        _Invocation(
            "qkv_mm",
            block_fp8_mm,
            {
                "activation_q": "pre_q",
                "activation_scale": "pre_scale",
                "weight_q": "qkv_weight_q",
                "weight_scale": "qkv_weight_scale",
                "group_size": "group",
            },
            {"out": "qkv"},
        ),
        _Bridge(
            f"""
            batch = hidden_states.shape[0]
            query = qkv[:, : q_heads * head_dim].view(batch, q_heads, head_dim)
            key_begin = q_heads * head_dim
            key = qkv[:, key_begin : key_begin + kv_heads * head_dim].view(
                batch, kv_heads, head_dim
            )
            value = qkv[:, key_begin + kv_heads * head_dim : {qkv_width}].view(
                batch, kv_heads, head_dim
            )
            """
        ),
        _Invocation(
            "qk",
            fused_qk_norm_rope,
            {
                "qkv": "qkv",
                "num_heads_q": "q_heads",
                "num_heads_k": "kv_heads",
                "num_heads_v": "kv_heads",
                "head_dim": "head_dim",
                "eps": "eps",
                "q_weight": "q_weight",
                "k_weight": "k_weight",
                "cos_sin_cache": "cos_sin",
                "is_neox": "True",
                "position_ids": "position",
                "forced_token_heads_per_warp": "-1",
            },
        ),
        _Invocation(
            "cache",
            reshape_and_cache_flash,
            {
                "key": "key",
                "value": "value",
                "kv_cache": "kv_cache",
                "slot_mapping": "slot_mapping",
                "block_size": "cache_block",
            },
        ),
        _Invocation(
            "attention_split",
            paged_gqa_decode_attention_split,
            {
                "query": "query",
                "kv_cache": "kv_cache",
                "block_table": "block_table",
                "context": "context",
                "block_size": "cache_block",
                "q_per_kv": "q_heads // kv_heads",
                "splits": "attention_splits",
            },
            {"partial_out": "partial_out", "partial_lse": "partial_lse"},
        ),
        _Invocation(
            "attention_merge",
            merge_attention_splits,
            {"partial_out": "partial_out", "partial_lse": "partial_lse"},
            {"output": "attention"},
        ),
        _Bridge("attention_flat = attention.view(batch, hidden)"),
        _Invocation(
            "attention_quant",
            per_token_group_fp8_quant,
            {
                "input": "attention_flat",
                "output_q": "attention_q",
                "output_s": "attention_scale",
                "group_size": "group",
                "eps": "1e-10",
                "fp8_min": "FP8_MIN",
                "fp8_max": "FP8_MAX",
                "scale_ue8m0": "False",
                "dummy_is_scale_transposed": "False",
                "dummy_is_tma_aligned": "False",
            },
        ),
        _Invocation(
            "o_mm",
            block_fp8_mm,
            {
                "activation_q": "attention_q",
                "activation_scale": "attention_scale",
                "weight_q": "o_weight_q",
                "weight_scale": "o_weight_scale",
                "group_size": "group",
            },
            {"out": "attention_out"},
        ),
        _Invocation(
            "post",
            rms_norm_per_block_quant,
            {
                "result": "ffn_q",
                "input": "attention_out",
                "weight": "post_weight",
                "scale": "ffn_scale",
                "epsilon": "eps",
                "scale_ub": "None",
                "residual": "residual",
                "group_size": "group",
                "is_scale_transposed": "False",
            },
        ),
        _Invocation(
            "w13",
            block_fp8_mm,
            {
                "activation_q": "ffn_q",
                "activation_scale": "ffn_scale",
                "weight_q": "w13_q",
                "weight_scale": "w13_scale",
                "group_size": "group",
            },
            {"out": "gate_up"},
        ),
        _Invocation(
            "activation",
            silu_and_mul_per_block_quant,
            {"gate_up": "gate_up", "group_size": "group"},
            {
                "activation_q": "activation_q",
                "activation_scale": "activation_scale",
            },
        ),
        _Invocation(
            "w2",
            block_fp8_mm,
            {
                "activation_q": "activation_q",
                "activation_scale": "activation_scale",
                "weight_q": "w2_q",
                "weight_scale": "w2_scale",
                "group_size": "group",
            },
            {"out": "output"},
        ),
    ]

    preamble: list[ast.stmt] = []
    loops: list[ast.For] = []
    for event in events:
        if isinstance(event, _Bridge):
            preamble.extend(ast.parse(textwrap.dedent(event.source)).body)
        else:
            event_preamble, event_loops = _inline_invocation(event)
            preamble.extend(event_preamble)
            loops.extend(event_loops)

    arguments = [
        "hidden_states",
        "residual",
        "pre_weight",
        "pre_q",
        "pre_scale",
        "qkv_weight_q",
        "qkv_weight_scale",
        "q_weight",
        "k_weight",
        "cos_sin",
        "position",
        "kv_cache",
        "block_table",
        "slot_mapping",
        "o_weight_q",
        "o_weight_scale",
        "attention_q",
        "attention_scale",
        "post_weight",
        "ffn_q",
        "ffn_scale",
        "w13_q",
        "w13_scale",
        "w2_q",
        "w2_scale",
        "hidden",
        "intermediate",
        "q_heads",
        "kv_heads",
        "head_dim",
        "context",
        "cache_block",
        "attention_splits",
        "group",
        "eps",
    ]
    result_names = [
        "output",
        "pre_q",
        "pre_scale",
        "qkv",
        "partial_out",
        "partial_lse",
        "attention",
        "attention_q",
        "attention_scale",
        "attention_out",
        "ffn_q",
        "ffn_scale",
        "gate_up",
        "activation_q",
        "activation_scale",
        "residual",
    ]
    function = ast.FunctionDef(
        name="qwen3_layer_tile_dependency_source",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in arguments],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            *preamble,
            *loops,
            ast.Return(
                value=ast.Tuple(
                    elts=[ast.Name(id=name, ctx=ast.Load()) for name in result_names],
                    ctx=ast.Load(),
                )
            ),
        ],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    return ast.unparse(module) + "\n"


def _build_composite_kernel():
    source = _compose_qwen3_layer_source()
    filename = str(Path(__file__).with_name("_generated_qwen3_tile_dependency.py"))
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = globals()
    exec(compile(source, filename, "exec"), namespace)
    function = namespace["qwen3_layer_tile_dependency_source"]
    return helion.kernel(static_shapes=True, autotune_effort="none")(function), source


qwen3_layer_tile_dependency, GENERATED_SOURCE = _build_composite_kernel()


def _probe_matched_config(bound, args):
    """Map the winning Triton probe's arithmetic geometry onto Helion IDs."""
    values = dict(bound.config_spec.default_config())
    block_size_by_id = {
        1: 128,  # collaborative prefix reduction chunk
        2: 1,  # one prefix quant group per logical task
        5: 8,  # QKV N
        8: args.qk_head_block,
        11: 1,  # one KV head per cache task
        12: 128,  # full cache head dimension
        15: 4,  # four queries per attention task
        17: args.attention_context_block,
        19: args.merge_q_block,
        20: args.merge_split_block,
        22: 1,  # one attention quant group per task
        25: 8,  # O N
        28: 128,  # collaborative post-prefix reduction chunk
        29: 1,  # one post-prefix quant group per task
        32: 16,  # W13 N
        37: 8,  # W2 N
    }
    values["block_sizes"] = [
        block_size_by_id[spec.block_id] for spec in bound.config_spec.block_sizes
    ]
    values["loop_orders"] = [
        [0, 1],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2],
        [2, 1, 0],
        [0, 1],
        [0, 1, 2],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
    ]
    values["l2_groupings"] = [1, 2, 32, 16, 8, 1, 4, 1, 1, 1]

    def by_block_id(specs, choices, default):
        return [
            next(
                (
                    choices[block_id]
                    for block_id in spec.block_ids
                    if block_id in choices
                ),
                default,
            )
            for spec in specs
        ]

    staged_reductions = dict.fromkeys((6, 26, 33, 38), args.projection_stages)
    values["range_num_stages"] = by_block_id(
        bound.config_spec.range_num_stages, staged_reductions, 0
    )
    values["range_unroll_factors"] = by_block_id(
        bound.config_spec.range_unroll_factors,
        {6: 2, 26: 2, 33: 2, 38: 4},
        0,
    )
    values["range_multi_buffers"] = by_block_id(
        bound.config_spec.range_multi_buffers,
        {6: True, 17: True, 20: True, 26: False, 33: True, 38: False},
        None,
    )
    values["range_flattens"] = by_block_id(
        bound.config_spec.range_flattens,
        {
            1: True,
            2: True,
            6: False,
            17: True,
            20: False,
            26: False,
            28: True,
            29: True,
            33: False,
            38: True,
        },
        None,
    )
    values.update(
        {
            "num_warps": 1,
            "num_stages": args.kernel_stages,
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": args.worker_multiplier,
        }
    )
    if args.cross_loop_workers is not None:
        values["cross_loop_num_workers"] = args.cross_loop_workers
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return config


def _composite_args(tensors, args):
    return (
        tensors["hidden_states"],
        tensors["residual"],
        tensors["pre_weight"],
        tensors["pre_q"],
        tensors["pre_scale"],
        tensors["qkv_weight_q"],
        tensors["qkv_weight_scale"],
        tensors["q_weight"],
        tensors["k_weight"],
        tensors["cos_sin"],
        tensors["position"],
        tensors["kv_cache"],
        tensors["block_table"],
        tensors["slot_mapping"],
        tensors["o_weight_q"],
        tensors["o_weight_scale"],
        tensors["attention_q"],
        tensors["attention_scale"],
        tensors["post_weight"],
        tensors["ffn_q"],
        tensors["ffn_scale"],
        tensors["w13_q"],
        tensors["w13_scale"],
        tensors["w2_q"],
        tensors["w2_scale"],
        args.hidden,
        args.intermediate,
        args.q_heads,
        args.kv_heads,
        args.head_dim,
        args.context,
        args.block_size,
        args.attention_splits,
        args.group,
        args.eps,
    )


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
    launch_shared = getattr(
        kernel, "_helion_launch_dynamic_shared_bytes", kernel.metadata.shared
    )
    return {
        "registers": kernel.n_regs,
        "spills": kernel.n_spills,
        "shared": launch_shared,
        "triton_required_shared": kernel.metadata.shared,
        "resident_blocks_per_sm": getattr(
            kernel, "_helion_resident_blocks_per_sm", None
        ),
    }


def run(args) -> None:
    if not args.allow_busy:
        require_idle_visible_gpu()
    if args.print_source:
        print(GENERATED_SOURCE)
        return
    tensors = allocate_layer(args)
    composite_args = _composite_args(tensors, args)
    bound = qwen3_layer_tile_dependency.bind(composite_args)
    host_function = bound.host_function
    assert host_function is not None
    config = (
        _probe_matched_config(bound, args)
        if args.probe_config
        else bound.config_spec.default_config()
    )
    lowered = bound.to_triton_code(config)
    lowered_path = args.lowered_output.resolve()
    lowered_path.parent.mkdir(parents=True, exist_ok=True)
    lowered_path.write_text(lowered)
    print("LOWERED_TRITON", lowered_path, flush=True)
    if args.dump_config:
        print("CONFIG", dict(config), flush=True)
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
            "ROOT_BLOCK_IDS",
            host_function.device_ir.grid_block_ids,
            flush=True,
        )
        print(
            "RANGE_BLOCK_IDS",
            [spec.block_ids for spec in bound.config_spec.range_num_stages],
            flush=True,
        )
    if args.dump_ir:
        print(host_function.device_ir.debug_str(), flush=True)
        print(
            "CROSS_LOOP_DEPENDENCIES",
            [
                dataclasses.asdict(dependency)
                for dependency in (
                    host_function.device_ir.tile_dependency_graph.edges
                    if host_function.device_ir.tile_dependency_graph is not None
                    else ()
                )
            ],
            flush=True,
        )
    if args.dump_triton:
        print(lowered, flush=True)
        return
    if args.inspect_only:
        return
    compiled = bound.compile_config(config)
    outputs = compiled(*composite_args)
    torch.cuda.synchronize()
    print("RESOURCES", _helion_resources(compiled), flush=True)
    print(
        "COMPILED",
        {
            "roots": len(host_function.device_ir.root_ids),
            "tile_dependencies": len(
                host_function.device_ir.tile_dependency_graph.edges
                if host_function.device_ir.tile_dependency_graph is not None
                else ()
            ),
            "implicit_phase_starts": sorted(
                host_function.device_ir.implicit_dependency_starts
            ),
            "pid_type": config.pid_type,
        },
        flush=True,
    )
    if args.smoke:
        print("SMOKE_OK", tuple(tuple(output.shape) for output in outputs), flush=True)
        return
    if args.timing_only:
        graph, _ = capture(lambda: compiled(*composite_args))
        benchmark_pids = visible_gpu_pids()
        timings = benchmark_interleaved(
            {"helion_tile_dependency": graph.replay},
            args.repeats,
            args.batch_replays,
        )
        if visible_gpu_pids() != benchmark_pids:
            raise RuntimeError("GPU process set changed during benchmark")
        print("TIMINGS", timings, flush=True)
        return

    reference_tensors = allocate_layer(args)
    reference_launch, reference = build_helion_reference(args, reference_tensors)
    reference_launch()
    torch.cuda.synchronize()
    persistent = dict(
        zip(
            (
                "output",
                "pre_q",
                "pre_scale",
                "qkv",
                "partial_out",
                "partial_lse",
                "attention",
                "attention_q",
                "attention_scale",
                "attention_out",
                "ffn_q",
                "ffn_scale",
                "gate_up",
                "activation_q",
                "activation_scale",
                "residual",
            ),
            outputs,
            strict=True,
        )
    )
    tolerances = {
        "output": (0.25, 5e-2),
        "qkv": (6e-2, 5e-2),
        "partial_out": (8e-2, 3e-2),
        "partial_lse": (8e-2, 3e-2),
        "attention": (8e-2, 3e-2),
        "attention_out": (0.125, 3e-2),
        "gate_up": (0.125, 3e-2),
        "activation_q": (64.0, 3e-2),
        "activation_scale": (2e-3, 3e-2),
    }
    for name, expected in reference.items():
        actual = persistent[name].view_as(expected)
        atol, rtol = tolerances[name]
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=atol, rtol=rtol
        )

    graph, _ = capture(lambda: compiled(*composite_args))
    benchmark_reference_launch = reference_launch
    if args.helion_comparison_splits != args.attention_splits:
        comparison_args = argparse.Namespace(**vars(args))
        comparison_args.attention_splits = args.helion_comparison_splits
        comparison_tensors = allocate_layer(comparison_args)
        benchmark_reference_launch, _ = build_helion_reference(
            comparison_args, comparison_tensors
        )
    reference_graph, _ = capture(benchmark_reference_launch)
    graph.replay()
    torch.cuda.synchronize()
    benchmark_pids = visible_gpu_pids()
    timings = benchmark_interleaved(
        {
            "helion_tile_dependency": graph.replay,
            "helion_separate": reference_graph.replay,
        },
        args.repeats,
        args.batch_replays,
    )
    if visible_gpu_pids() != benchmark_pids:
        raise RuntimeError("GPU process set changed during benchmark")
    print("TIMINGS", timings, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=12288)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--attention-splits", type=int, default=128)
    parser.add_argument("--helion-comparison-splits", type=int, default=32)
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-replays", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dump-config", action="store_true")
    parser.add_argument("--dump-ir", action="store_true")
    parser.add_argument("--dump-triton", action="store_true")
    parser.add_argument(
        "--lowered-output",
        type=Path,
        default=Path("/tmp/qwen3_layer_clc_lowered.py"),
    )
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--probe-config", action="store_true")
    parser.add_argument("--projection-stages", type=int, default=4)
    parser.add_argument("--kernel-stages", type=int, default=2)
    parser.add_argument("--worker-multiplier", type=int, default=8)
    parser.add_argument("--cross-loop-workers", type=int)
    parser.add_argument("--merge-split-block", type=int, default=32)
    parser.add_argument("--merge-q-block", type=int, default=4)
    parser.add_argument("--attention-context-block", type=int, default=32)
    parser.add_argument("--qk-head-block", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--print-source", action="store_true")
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument(
        "--config-path",
        default=str(Path(__file__).with_name("qwen3_layer_helion_b200_configs.json")),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
