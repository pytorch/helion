# ruff: noqa: ANN001, ANN202
"""Shared-source narrow Qwen3 FFN probe for TileDependency lowering."""

from __future__ import annotations

import argparse
import ast
import linecache
from pathlib import Path

import torch

from probes.common import benchmark_interleaved
from probes.common import capture
from probes.common import require_idle_visible_gpu
from probes.common import visible_gpu_pids
from probes.qwen3.helion_qwen3_layer_baseline import FFN_CONFIGS
from probes.qwen3.helion_qwen3_layer_baseline import FP8_MAX  # noqa: F401
from probes.qwen3.helion_qwen3_layer_baseline import FP8_MIN  # noqa: F401
from probes.qwen3.helion_qwen3_layer_baseline import FP8_MIN_SCALE  # noqa: F401
from probes.qwen3.helion_qwen3_layer_baseline import block_fp8_mm
from probes.qwen3.helion_qwen3_layer_baseline import compile_config
from probes.qwen3.helion_qwen3_layer_baseline import silu_and_mul_per_block_quant
from probes.qwen3.helion_qwen3_tile_dependency import _inline_invocation
from probes.qwen3.helion_qwen3_tile_dependency import _Invocation

import helion
import helion.language as hl  # noqa: F401 - generated source global


def _compose_source() -> str:
    events = [
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
        event_preamble, event_loops = _inline_invocation(event)
        preamble.extend(event_preamble)
        loops.extend(event_loops)
    function = ast.FunctionDef(
        name="qwen3_ffn_tile_dependency_source",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg=name)
                for name in (
                    "ffn_q",
                    "ffn_scale",
                    "w13_q",
                    "w13_scale",
                    "w2_q",
                    "w2_scale",
                    "group",
                )
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
                            "output",
                            "gate_up",
                            "activation_q",
                            "activation_scale",
                        )
                    ],
                    ctx=ast.Load(),
                )
            ),
        ],
        decorator_list=[],
    )
    return (
        ast.unparse(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        )
        + "\n"
    )


def _build_kernel():
    source = _compose_source()
    filename = str(Path(__file__).with_name("_generated_qwen3_ffn_dependency.py"))
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = globals()
    exec(compile(source, filename, "exec"), namespace)
    return helion.kernel(static_shapes=True, autotune_effort="none")(
        namespace["qwen3_ffn_tile_dependency_source"]
    )


qwen3_ffn_tile_dependency = _build_kernel()


def _persistent_config(bound, args):
    values = dict(bound.config_spec.default_config())
    root_ids = bound.host_function.device_ir.grid_block_ids
    w13_n = root_ids[0][-1]
    w2_n = root_ids[2][-1]
    block_sizes = {w13_n: 16, w2_n: 8}
    values["block_sizes"] = [
        block_sizes[spec.block_id] for spec in bound.config_spec.block_sizes
    ]
    if args.batch > 1:
        values["loop_orders"] = [[1, 0] for _ in values["loop_orders"]]

    def by_root(specs, selected, default):
        return [
            next(
                (
                    selected[block_id]
                    for block_id in spec.block_ids
                    if block_id in selected
                ),
                default,
            )
            for spec in specs
        ]

    w13_reduction = next(iter(bound.config_spec.range_num_stages[1].block_ids))
    w2_reduction = next(iter(bound.config_spec.range_num_stages[-1].block_ids))
    range_roots = {w13_reduction: args.w13_stages, w2_reduction: args.w2_stages}
    values["range_num_stages"] = by_root(
        bound.config_spec.range_num_stages, range_roots, 0
    )
    values["range_unroll_factors"] = by_root(
        bound.config_spec.range_unroll_factors,
        {w13_reduction: args.w13_unroll, w2_reduction: args.w2_unroll},
        0,
    )
    values["range_multi_buffers"] = by_root(
        bound.config_spec.range_multi_buffers,
        {w13_reduction: True, w2_reduction: False},
        None,
    )
    values["range_flattens"] = by_root(
        bound.config_spec.range_flattens,
        {w13_reduction: False, w2_reduction: True},
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
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return config


def _helion_resources(compiled_wrapper):
    """Recover the single Triton kernel owned by a compiled Helion wrapper."""
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


def run(args) -> None:
    if not args.allow_busy:
        require_idle_visible_gpu()
    device = "cuda"
    ffn_q = torch.randn(
        (args.batch, args.hidden), device=device, dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    ffn_scale = torch.rand(
        (args.batch, args.hidden // args.group),
        device=device,
        dtype=torch.float32,
    )
    w13_q = torch.randn(
        (2 * args.intermediate, args.hidden), device=device, dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    w13_scale = torch.rand(
        (2 * args.intermediate // args.group, args.hidden // args.group),
        device=device,
        dtype=torch.float32,
    )
    w2_q = torch.randn(
        (args.hidden, args.intermediate), device=device, dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    w2_scale = torch.rand(
        (args.hidden // args.group, args.intermediate // args.group),
        device=device,
        dtype=torch.float32,
    )
    kernel_args = (ffn_q, ffn_scale, w13_q, w13_scale, w2_q, w2_scale, args.group)
    bound = qwen3_ffn_tile_dependency.bind(kernel_args)
    host_function = bound.host_function
    assert host_function is not None
    config = _persistent_config(bound, args)
    if args.dump_config:
        print("CONFIG", dict(config), flush=True)
        print("ROOT_BLOCK_IDS", host_function.device_ir.grid_block_ids, flush=True)
        print(
            "RANGE_BLOCK_IDS",
            [spec.block_ids for spec in bound.config_spec.range_num_stages],
            flush=True,
        )
        return
    if args.dump_triton:
        print(bound.to_triton_code(config), flush=True)
        return
    compiled = bound.compile_config(config)
    output, gate_up, activation_q, activation_scale = compiled(*kernel_args)

    _, w13 = compile_config(
        block_fp8_mm,
        (ffn_q, ffn_scale, w13_q, w13_scale, args.group),
        FFN_CONFIGS["w13"],
    )
    separate_gate = w13(ffn_q, ffn_scale, w13_q, w13_scale, args.group)
    _, activation = compile_config(
        silu_and_mul_per_block_quant,
        (separate_gate, args.group),
        FFN_CONFIGS["silu_quant"],
    )
    separate_q, separate_scale = activation(separate_gate, args.group)
    _, w2 = compile_config(
        block_fp8_mm,
        (separate_q, separate_scale, w2_q, w2_scale, args.group),
        FFN_CONFIGS["w2"],
    )
    separate_output = w2(separate_q, separate_scale, w2_q, w2_scale, args.group)
    torch.cuda.synchronize()
    if not args.skip_validation:
        for actual, expected, atol, rtol in (
            (gate_up, separate_gate, 0.125, 3e-2),
            (activation_q, separate_q, 64.0, 3e-2),
            (activation_scale, separate_scale, 2e-3, 3e-2),
            (output, separate_output, 0.25, 5e-2),
        ):
            torch.testing.assert_close(
                actual.float(), expected.float(), atol=atol, rtol=rtol
            )

    persistent_graph, _ = capture(lambda: compiled(*kernel_args))

    def launch_separate():
        gate = w13(ffn_q, ffn_scale, w13_q, w13_scale, args.group)
        quant, scale = activation(gate, args.group)
        return w2(quant, scale, w2_q, w2_scale, args.group)

    separate_graph, _ = capture(launch_separate)
    if args.profile:
        for _ in range(args.profile_warmups):
            persistent_graph.replay()
        torch.cuda.synchronize()
        print("RESOURCES", _helion_resources(compiled), flush=True)
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(args.profile_replays):
            persistent_graph.replay()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return
    pids = visible_gpu_pids()
    timings = benchmark_interleaved(
        {
            "helion_tile_dependency_ffn": persistent_graph.replay,
            "helion_separate_ffn": separate_graph.replay,
        },
        args.repeats,
        args.batch_replays,
    )
    if visible_gpu_pids() != pids:
        raise RuntimeError("GPU process set changed during benchmark")
    print("RESOURCES", _helion_resources(compiled), flush=True)
    print("TIMINGS", timings, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=12288)
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-replays", type=int, default=10)
    parser.add_argument("--w13-stages", type=int, default=4)
    parser.add_argument("--w13-unroll", type=int, default=2)
    parser.add_argument("--w2-stages", type=int, default=4)
    parser.add_argument("--w2-unroll", type=int, default=4)
    parser.add_argument("--kernel-stages", type=int, default=2)
    parser.add_argument("--worker-multiplier", type=int, default=8)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-warmups", type=int, default=5)
    parser.add_argument("--profile-replays", type=int, default=1)
    parser.add_argument("--dump-config", action="store_true")
    parser.add_argument("--dump-triton", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--allow-busy", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
