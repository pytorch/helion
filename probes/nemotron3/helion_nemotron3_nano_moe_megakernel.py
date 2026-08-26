# ruff: noqa: A002, ANN001, ANN202
# pyrefly: ignore-errors
"""Single-kernel CLC probe for the Nemotron-3 Nano FP8 MoE block.

The composite Helion source preserves the kernel boundaries modeled by
``helion_nemotron3_nano_moe``.  It changes only execution: all roots are
lowered into one dependency-scheduled Triton kernel whose physical CTAs use
Cluster Launch Control to drain the compiler-generated logical task stream.
"""

from __future__ import annotations

import argparse
import ast
import json
import linecache
import os
from pathlib import Path

import torch

from probes.common import benchmark_interleaved
from probes.common import capture
from probes.common import require_idle_visible_gpu
from probes.common import visible_gpu_pids
from probes.gemma4.helion_gemma4_e4b_megakernel import _Bridge
from probes.gemma4.helion_gemma4_e4b_megakernel import _helion_resources
from probes.gemma4.helion_gemma4_e4b_megakernel import _inline_invocation
from probes.gemma4.helion_gemma4_e4b_megakernel import _Invocation
import probes.nemotron3.helion_nemotron3_nano_moe as separate
from probes.nemotron3.helion_nemotron3_nano_moe import DEFAULT_CONFIG_PATH
from probes.nemotron3.helion_nemotron3_nano_moe import FP8_MAX
from probes.nemotron3.helion_nemotron3_nano_moe import Nemotron3NanoMoEShape
from probes.nemotron3.helion_nemotron3_nano_moe import allocate
from probes.nemotron3.helion_nemotron3_nano_moe import fp8_scaled_mm
from probes.nemotron3.helion_nemotron3_nano_moe import (
    fused_build_expert_maps_sort_first_token,
)
from probes.nemotron3.helion_nemotron3_nano_moe import initialize_autotune_inputs
from probes.nemotron3.helion_nemotron3_nano_moe import relu2_static_fp8_quant
from probes.nemotron3.helion_nemotron3_nano_moe import relu_squared
from probes.nemotron3.helion_nemotron3_nano_moe import routed_gemm1
from probes.nemotron3.helion_nemotron3_nano_moe import routed_gemm2_fused_finalize
from probes.nemotron3.helion_nemotron3_nano_moe import router_gemm_fp32
from probes.nemotron3.helion_nemotron3_nano_moe import scale_routed_and_add_shared
from probes.nemotron3.helion_nemotron3_nano_moe import static_scaled_fp8_quant
from probes.nemotron3.helion_nemotron3_nano_moe import topk_sigmoid

import helion
import helion.language as hl


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def zero_output(output: torch.Tensor) -> None:
    """Represent vLLM's routed-output memset as an explicit task family."""
    rows, width = output.size()
    for tile_m, tile_n in hl.tile([rows, width]):
        output[tile_m, tile_n] = 0.0


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def dense_relu2_static_fp8_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    quant_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> None:
    """Diagnostic equivalent of routed activation when every row is valid."""
    rows, width = input.size()
    hl.specialize(rows)
    hl.specialize(width)
    assert bias is None
    for tile_m, tile_n in hl.tile([rows, width]):
        values = input[tile_m, tile_n].to(torch.float32)
        activated = torch.maximum(values, torch.zeros_like(values))
        quantized = activated * activated * hl.load(quant_scale, [])
        out[tile_m, tile_n] = quantized.clamp(0.0, FP8_MAX).to(out.dtype)


def _events(*, dense_routed_activation: bool) -> tuple[_Invocation | _Bridge, ...]:
    return (
        _Bridge("tokens = hidden_states.size(0)"),
        _Bridge("assignments = tokens * top_k"),
        _Bridge(
            "topk_weights = torch.empty((tokens, top_k), "
            "dtype=torch.float32, device=hidden_states.device)"
        ),
        _Bridge(
            "topk_ids = torch.empty((tokens, top_k), "
            "dtype=torch.int32, device=hidden_states.device)"
        ),
        _Bridge(
            "token_expert_indices = torch.empty((tokens, top_k), "
            "dtype=torch.int32, device=hidden_states.device)"
        ),
        _Bridge(
            "routed_input_q = torch.empty(hidden_states.shape, "
            "dtype=torch.float8_e4m3fn, device=hidden_states.device)"
        ),
        _Bridge(
            "permuted_row_to_unpermuted_row = torch.empty((assignments,), "
            "dtype=torch.int32, device=hidden_states.device)"
        ),
        _Bridge(
            "unpermuted_row_to_permuted_row = torch.empty((assignments,), "
            "dtype=torch.int32, device=hidden_states.device)"
        ),
        _Bridge(
            "expert_first_token_offset = torch.empty((num_experts + 1,), "
            "dtype=torch.int64, device=hidden_states.device)"
        ),
        _Bridge(
            "permuted_input = torch.empty((assignments, hidden_size), "
            "dtype=torch.float8_e4m3fn, device=hidden_states.device)"
        ),
        _Bridge(
            "permuted_scales = torch.empty((assignments,), "
            "dtype=torch.float32, device=hidden_states.device)"
        ),
        _Bridge(
            "routed_gemm1_output = torch.empty((assignments, routed_intermediate), "
            "dtype=torch.bfloat16, device=hidden_states.device)"
        ),
        _Bridge(
            "routed_activation_q = torch.empty((assignments, routed_intermediate), "
            "dtype=torch.float8_e4m3fn, device=hidden_states.device)"
        ),
        _Bridge("routed_output = torch.empty_like(hidden_states)"),
        _Bridge(
            "shared_input_q = torch.empty(hidden_states.shape, "
            "dtype=torch.float8_e4m3fn, device=hidden_states.device)"
        ),
        _Bridge(
            "shared_activation = torch.empty((tokens, shared_intermediate), "
            "dtype=torch.bfloat16, device=hidden_states.device)"
        ),
        _Bridge(
            "shared_activation_q = torch.empty((tokens, shared_intermediate), "
            "dtype=torch.float8_e4m3fn, device=hidden_states.device)"
        ),
        _Invocation(
            "router",
            router_gemm_fp32,
            {
                "hidden_states": "hidden_states",
                "weight": "router_weight",
                "output_dtype": "torch.float32",
            },
            {"output": "router_logits"},
        ),
        # Preserve the production auxiliary-stream branch as ordinary roots.
        # The dependency graph, rather than a model-specific scheduler, exposes
        # that it is independent of routing until the final join.
        _Invocation(
            "shared_input_quant",
            static_scaled_fp8_quant,
            {
                "out": "shared_input_q",
                "input": "hidden_states",
                "scale": "shared_up_input_scale",
                "group_shape": "None",
            },
        ),
        _Invocation(
            "shared_up",
            fp8_scaled_mm,
            {
                "A": "shared_input_q",
                "B": "shared_up_weight",
                "scale_a": "shared_up_input_scale",
                "scale_b": "shared_up_weight_scale",
                "out_dtype": "torch.bfloat16",
                "bias": "None",
            },
            {"output": "shared_up"},
        ),
        _Invocation(
            "shared_activation",
            relu_squared,
            {"out": "shared_activation", "input": "shared_up"},
        ),
        _Invocation(
            "shared_activation_quant",
            static_scaled_fp8_quant,
            {
                "out": "shared_activation_q",
                "input": "shared_activation",
                "scale": "shared_down_input_scale",
                "group_shape": "None",
            },
        ),
        _Invocation(
            "shared_down",
            fp8_scaled_mm,
            {
                "A": "shared_activation_q",
                "B": "shared_down_weight",
                "scale_a": "shared_down_input_scale",
                "scale_b": "shared_down_weight_scale",
                "out_dtype": "torch.bfloat16",
                "bias": "None",
            },
            {"output": "shared_output"},
        ),
        _Invocation(
            "topk",
            topk_sigmoid,
            {
                "topk_weights": "topk_weights",
                "topk_indices": "topk_ids",
                "token_expert_indices": "token_expert_indices",
                "gating_output": "router_logits",
                "renormalize": "True",
                "e_score_correction_bias": "correction_bias",
                "routed_scaling_factor": "1.0",
                "is_padding": "None",
            },
        ),
        _Invocation(
            "routed_input_quant",
            static_scaled_fp8_quant,
            {
                "out": "routed_input_q",
                "input": "hidden_states",
                "scale": "routed_input_scale",
                "group_shape": "None",
            },
        ),
        _Invocation(
            "prologue",
            fused_build_expert_maps_sort_first_token,
            {
                "token_selected_experts": "topk_ids",
                "permuted_row_to_unpermuted_row": "permuted_row_to_unpermuted_row",
                "unpermuted_row_to_permuted_row": "unpermuted_row_to_permuted_row",
                "expert_first_token_offset": "expert_first_token_offset",
            },
        ),
        _Invocation(
            "expand",
            separate.expand_input_rows,
            {
                "unpermuted_input": "routed_input_q",
                "permuted_output": "permuted_input",
                "unpermuted_scales": "topk_weights",
                "permuted_scales": "permuted_scales",
                "permuted_row_to_unpermuted_row": "permuted_row_to_unpermuted_row",
                "expert_first_token_offset": "expert_first_token_offset",
            },
        ),
        _Invocation(
            "routed_gemm1",
            routed_gemm1,
            {
                "output": "routed_gemm1_output",
                "permuted_input": "permuted_input",
                "fc1_expert_weights": "routed_w1",
                "fc1_dequant_scales": "g1_alphas",
                "expert_first_token_offset": "expert_first_token_offset",
            },
        ),
        _Invocation(
            "routed_activation",
            (
                dense_relu2_static_fp8_quant
                if dense_routed_activation
                else relu2_static_fp8_quant
            ),
            (
                {
                    "out": "routed_activation_q",
                    "input": "routed_gemm1_output",
                    "quant_scale": "a2_gscale",
                    "bias": "None",
                }
                if dense_routed_activation
                else {
                    "out": "routed_activation_q",
                    "input": "routed_gemm1_output",
                    "quant_scale": "a2_gscale",
                    "bias": "None",
                    "expert_first_token_offset": "expert_first_token_offset",
                }
            ),
        ),
        _Invocation("routed_zero", zero_output, {"output": "routed_output"}),
        _Invocation(
            "routed_gemm2",
            routed_gemm2_fused_finalize,
            {
                "output": "routed_output",
                "activation": "routed_activation_q",
                "fc2_expert_weights": "routed_w2",
                "fc2_dequant_scales": "g2_alphas",
                "expert_first_token_offset": "expert_first_token_offset",
                "permuted_row_to_unpermuted_row": "permuted_row_to_unpermuted_row",
                "permuted_final_scales": "permuted_scales",
            },
        ),
        _Invocation(
            "merge",
            scale_routed_and_add_shared,
            {
                "shared_output": "shared_output",
                "routed_output": "routed_output",
                "routed_scaling_factor": "routed_scaling_factor",
            },
            {"output": "output"},
        ),
    )


OUTPUT_NAMES = (
    "output",
    "router_logits",
    "topk_weights",
    "topk_ids",
    "routed_gemm1_output",
    "routed_activation_q",
    "routed_output",
    "shared_up",
    "shared_activation",
    "shared_output",
)


def _compose_source(*, dense_routed_activation: bool) -> str:
    preamble: list[ast.stmt] = []
    loops: list[ast.For] = []
    for event in _events(dense_routed_activation=dense_routed_activation):
        if isinstance(event, _Bridge):
            preamble.extend(ast.parse(event.source).body)
            continue
        event_preamble, event_loops = _inline_invocation(event)
        preamble.extend(event_preamble)
        loops.extend(event_loops)

    arguments = (
        "hidden_states",
        "router_weight",
        "correction_bias",
        "routed_input_scale",
        "routed_w1",
        "g1_alphas",
        "a2_gscale",
        "routed_w2",
        "g2_alphas",
        "shared_up_input_scale",
        "shared_up_weight",
        "shared_up_weight_scale",
        "shared_down_input_scale",
        "shared_down_weight",
        "shared_down_weight_scale",
        "top_k",
        "num_experts",
        "hidden_size",
        "routed_intermediate",
        "shared_intermediate",
        "routed_scaling_factor",
    )
    function = ast.FunctionDef(
        name="nemotron3_nano_moe_megakernel_source",
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
                    elts=[ast.Name(id=name, ctx=ast.Load()) for name in OUTPUT_NAMES],
                    ctx=ast.Load(),
                )
            ),
        ],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    return ast.unparse(module) + "\n"


def _build_megakernel(*, dense_routed_activation: bool):
    source = _compose_source(dense_routed_activation=dense_routed_activation)
    suffix = "_dense_activation" if dense_routed_activation else ""
    filename = str(
        Path(__file__).with_name(f"_generated_nemotron3_nano_moe{suffix}.py")
    )
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = globals()
    exec(compile(source, filename, "exec"), namespace)
    return (
        helion.kernel(
            static_shapes=True,
            autotune_effort="none",
            backend="triton",
        )(namespace["nemotron3_nano_moe_megakernel_source"]),
        source,
    )


MEGAKERNELS = {
    dense_routed_activation: _build_megakernel(
        dense_routed_activation=dense_routed_activation
    )
    for dense_routed_activation in (False, True)
}


def _kernel_args(tensors, shape: Nemotron3NanoMoEShape):
    return (
        tensors["hidden_states"],
        tensors["router_weight"],
        tensors["correction_bias"],
        tensors["routed_input_scale"],
        tensors["routed_w1"],
        tensors["g1_alphas"],
        tensors["a2_gscale"],
        tensors["routed_w2"],
        tensors["g2_alphas"],
        tensors["shared_up_input_scale"],
        tensors["shared_up_weight"],
        tensors["shared_up_weight_scale"],
        tensors["shared_down_input_scale"],
        tensors["shared_down_weight"],
        tensors["shared_down_weight_scale"],
        shape.top_k,
        shape.num_experts,
        shape.hidden,
        shape.routed_intermediate,
        shape.shared_intermediate,
        shape.routed_scaling_factor,
    )


def _config(bound, args):
    values = dict(bound.config_spec.default_config())
    block_sizes = {
        1: args.router_block_n,
        2: args.router_block_k,
        3: 1,
        4: args.pointwise_block,
        6: args.shared_block_n,
        7: args.shared_up_block_k,
        8: 1,
        9: args.pointwise_block,
        10: 1,
        11: args.pointwise_block,
        13: args.shared_block_n,
        14: args.shared_down_block_k,
        18: 1,
        19: args.pointwise_block,
        23: args.expand_block,
        25: args.routed_block_n,
        26: args.routed_block_k,
        27: 1,
        28: args.pointwise_block,
        29: 1,
        30: args.pointwise_block,
        32: args.routed_block_n,
        33: args.routed_block_k,
        34: 1,
        35: args.merge_block,
    }
    values["block_sizes"] = [
        block_sizes[spec.block_id] for spec in bound.config_spec.block_sizes
    ]
    values["range_num_stages"] = [
        (
            args.shared_down_stages
            if tuple(spec.block_ids) == (14,)
            else args.shared_up_stages
            if tuple(spec.block_ids) == (7,)
            else args.routed_stages
            if tuple(spec.block_ids) in ((26,), (33,))
            else default
        )
        for spec, default in zip(
            bound.config_spec.range_num_stages,
            values["range_num_stages"],
            strict=True,
        )
    ]
    values["range_unroll_factors"] = [
        args.shared_down_unroll if tuple(spec.block_ids) == (14,) else default
        for spec, default in zip(
            bound.config_spec.range_unroll_factors,
            values["range_unroll_factors"],
            strict=True,
        )
    ]
    values.update(
        {
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": args.worker_multiplier,
            "num_warps": args.num_warps,
            "num_stages": args.kernel_stages,
            "cross_loop_num_workers": args.workers,
        }
    )
    if args.maxnreg is not None:
        values["maxnreg"] = args.maxnreg
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return config


def _assert_close(name, actual, expected, *, atol, rtol):
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
    maximum = float((actual.float() - expected.float()).abs().max().item())
    print(f"megakernel_correctness {name} max_abs={maximum:.6f}", flush=True)


def _validate(outputs, reference_tensors, reference_output) -> None:
    actual = dict(zip(OUTPUT_NAMES, outputs, strict=True))
    _assert_close("output", actual["output"], reference_output, atol=1.5e-1, rtol=1e-1)
    _assert_close(
        "router_logits",
        actual["router_logits"],
        reference_tensors["router_logits"],
        atol=4e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(actual["topk_ids"], reference_tensors["topk_ids"])
    _assert_close(
        "topk_weights",
        actual["topk_weights"],
        reference_tensors["topk_weights"],
        atol=2e-5,
        rtol=2e-5,
    )
    _assert_close(
        "routed_output",
        actual["routed_output"],
        reference_tensors["routed_output"],
        atol=8e-2,
        rtol=8e-2,
    )
    _assert_close(
        "shared_output",
        actual["shared_output"],
        reference_tensors["shared_output"],
        atol=8e-2,
        rtol=8e-2,
    )


def run(args) -> None:
    if not args.allow_busy:
        require_idle_visible_gpu()
    shape = Nemotron3NanoMoEShape(tokens=args.tokens)
    shape.validate()
    tensors = allocate(shape)
    initialize_autotune_inputs(shape, tensors)
    kernel_args = _kernel_args(tensors, shape)
    kernel, source = MEGAKERNELS[args.dense_routed_activation]
    bound = kernel.bind(kernel_args)
    config = _config(bound, args)

    if args.print_source:
        print(source)
        return
    if args.dump_config:
        host_function = bound.host_function
        assert host_function is not None
        print("MEGAKERNEL_CONFIG", dict(config), flush=True)
        print("ROOT_BLOCK_IDS", host_function.device_ir.grid_block_ids, flush=True)
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

    lowered = bound.to_triton_code(config, output_origin_lines=True)
    lowered_path = args.lowered_output.resolve()
    lowered_path.parent.mkdir(parents=True, exist_ok=True)
    lowered_path.write_text(lowered)
    print("LOWERED_TRITON", lowered_path, flush=True)
    if args.print_lowered:
        print(lowered)
        return
    if args.inspect_only:
        return

    compiled = bound.compile_config(config)
    outputs = compiled(*kernel_args)
    torch.cuda.synchronize()
    print("MEGAKERNEL_RESOURCES", _helion_resources(compiled), flush=True)

    baseline_args = argparse.Namespace(
        config=args.config,
        tune=[],
        tune_effort="full",
    )
    baseline_tensors = allocate(shape)
    initialize_autotune_inputs(shape, baseline_tensors)
    baseline, _ = separate.build_moe(baseline_args, shape, baseline_tensors)
    reference_output = baseline(overlap_shared=True)
    torch.cuda.synchronize()
    _validate(outputs, baseline_tensors, reference_output)

    if not args.benchmark:
        return
    megakernel_graph, graph_outputs = capture(lambda: compiled(*kernel_args))
    baseline_graph, baseline_output = capture(lambda: baseline(overlap_shared=True))
    megakernel_graph.replay()
    baseline_graph.replay()
    torch.cuda.synchronize()
    _validate(graph_outputs, baseline_tensors, baseline_output)
    pids = visible_gpu_pids()
    timings = benchmark_interleaved(
        {
            "helion_nemotron3_megakernel": megakernel_graph.replay,
            "helion_nemotron3_separate_overlap": baseline_graph.replay,
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
                "resources": _helion_resources(compiled),
                "cold_l2": os.environ.get("MEGAKERNEL_CLEAR_L2") == "1",
                "timings": timings,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--workers", type=int, default=592)
    parser.add_argument("--worker-multiplier", type=int, default=4)
    parser.add_argument("--num-warps", type=int, default=2)
    parser.add_argument("--kernel-stages", type=int, default=1)
    parser.add_argument("--router-block-n", type=int, default=32)
    parser.add_argument("--router-block-k", type=int, default=32)
    parser.add_argument("--pointwise-block", type=int, default=256)
    parser.add_argument("--merge-block", type=int, default=128)
    parser.add_argument("--expand-block", type=int, default=32)
    parser.add_argument("--shared-block-n", type=int, default=32)
    parser.add_argument("--shared-up-block-k", type=int, default=512)
    parser.add_argument("--shared-down-block-k", type=int, default=512)
    parser.add_argument("--routed-block-n", type=int, default=32)
    parser.add_argument("--routed-block-k", type=int, default=512)
    parser.add_argument("--shared-up-stages", type=int, default=1)
    parser.add_argument("--shared-down-stages", type=int, default=2)
    parser.add_argument("--shared-down-unroll", type=int, default=2)
    parser.add_argument("--routed-stages", type=int, default=1)
    parser.add_argument("--maxnreg", type=int, choices=(32, 64, 128, 256))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch-replays", type=int, default=10)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--dense-routed-activation", action="store_true")
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--dump-config", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--print-source", action="store_true")
    parser.add_argument("--print-lowered", action="store_true")
    parser.add_argument(
        "--lowered-output",
        type=Path,
        default=Path("/tmp/nemotron3_nano_moe_clc_lowered.py"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
