# ruff: noqa: ANN001, ANN201, ANN202
"""Generate standalone-over-megakernel SM Gantt charts for maintained probes."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import inspect
import json
import linecache
from pathlib import Path
import subprocess
import sys
import types
from unittest import mock

# Keep direct invocation (``python probes/generate_sm_gantts.py``) anchored to
# this worktree instead of whichever Helion/probes package happens to be installed.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# These are the canonical workload-probe worktrees used while developing the
# current compiler.  Keep this repository first on sys.path so every workload
# is compiled by helion-clc, never by another Helion worktree.
CROSS_KERNEL_ROOT = REPO_ROOT.parent / "helion-cross-kernel"
NEMOTRON_ROOT = REPO_ROOT.parent / "nemotron3_nano"
MUSE_GLIMMER_ROOT = REPO_ROOT.parent / "muse_glimmer"
for probe_root in (CROSS_KERNEL_ROOT, NEMOTRON_ROOT, MUSE_GLIMMER_ROOT):
    if probe_root.is_dir() and str(probe_root) not in sys.path:
        sys.path.append(str(probe_root))

import torch

import helion
import probes

canonical_probe_package = str(CROSS_KERNEL_ROOT / "probes")
if canonical_probe_package not in probes.__path__:
    probes.__path__.append(canonical_probe_package)

from probes.common import require_idle_visible_gpu
from probes.sm_gantt import TracedCompiled
from probes.sm_gantt import capture_with_reset
from probes.sm_gantt import compile_traced
from probes.sm_gantt import compile_traced_megakernel
from probes.sm_gantt import render_stacked_gantt
from probes.sm_gantt import safe_name
from probes.sm_gantt import serialize_intervals
from probes.sm_gantt import summarize
from probes.sm_gantt import trace_megakernel
from probes.sm_gantt import trace_separate

DEFAULT_OUTPUT = REPO_ROOT / "probes/sm_gantt_clc_current"


@dataclass
class Comparison:
    slug: str
    title: str
    stage_order: tuple[str, ...]
    root_stages: dict[int, str]
    separate_label: str
    megakernel_label: str
    separate_launch: object
    megakernel_launch: object
    separate_traced: list[TracedCompiled]
    megakernel: dict[str, object]
    reset: object
    standalone_lowered: dict[str, str]
    correctness: str


def _config_from_dict(bound, values) -> helion.Config:
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return config


def _compile_stage(
    kernel,
    kernel_args,
    config,
    stage_by_root: dict[int, str],
    root_by_stage: dict[str, int],
):
    bound = kernel.bind(kernel_args)
    lowered = bound.to_triton_code(config, output_origin_lines=True)
    compiled = compile_traced(
        bound,
        config,
        stage_by_root,
        root_by_stage=root_by_stage,
    )
    return compiled, lowered


def _clone_kernel(kernel, name: str):
    """Clone a Helion source so repeated launch sites get distinct trace state."""
    function = kernel.fn
    module = ast.parse(inspect.getsource(function))
    definition = next(node for node in module.body if isinstance(node, ast.FunctionDef))
    definition.decorator_list = []
    definition.name = name
    ast.fix_missing_locations(module)
    source = ast.unparse(module) + "\n"
    filename = f"<{name}_exact_source_clone>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    module_name = f"{__name__}_{name}_module"
    module_object = types.ModuleType(module_name)
    module_object.__dict__.update(
        {
            key: value
            for key, value in function.__globals__.items()
            if key not in {"__name__", "__loader__", "__package__", "__spec__"}
        }
    )
    module_object.__file__ = filename
    sys.modules[module_name] = module_object
    exec(compile(source, filename, "exec"), module_object.__dict__)
    return helion.kernel(
        static_shapes=True,
        autotune_effort="none",
        backend="triton",
    )(module_object.__dict__[name])


def _build_qwen3_ffn() -> Comparison:
    from probes.qwen3 import helion_qwen3_ffn_tile_dependency as probe
    from probes.qwen3 import helion_qwen3_layer_baseline as baseline

    stages = (
        "Gate/up projection",
        "SwiGLU + FP8 quant",
        "Down projection",
    )
    root_stages = dict(enumerate(stages))
    root_by_stage = {stage: root for root, stage in root_stages.items()}
    args = argparse.Namespace(
        batch=1,
        hidden=4096,
        intermediate=12288,
        group=128,
        w13_stages=4,
        w13_unroll=2,
        w13_block=16,
        w2_stages=4,
        w2_unroll=4,
        w2_block=8,
        kernel_stages=2,
        num_warps=1,
        maxnreg=None,
        worker_multiplier=8,
        cross_loop_workers=None,
        evict_first=[],
        evict_last=[],
    )
    torch.manual_seed(0)
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

    bound = probe.qwen3_ffn_tile_dependency.bind(kernel_args)
    config = probe._persistent_config(bound, args)
    mega = compile_traced_megakernel(bound, config, root_stages)
    megakernel_compiled = mega["compiled"]
    megakernel_outputs = megakernel_compiled(*kernel_args)

    traced: list[TracedCompiled] = []
    lowered: dict[str, str] = {}

    def build(stage, kernel, local_args, config_values):
        local_bound = kernel.bind(local_args)
        local_config = _config_from_dict(local_bound, config_values)
        compiled, source = _compile_stage(
            kernel,
            local_args,
            local_config,
            {0: stage},
            root_by_stage,
        )
        traced.append(compiled)
        lowered[stage] = source
        return compiled

    w13_args = (ffn_q, ffn_scale, w13_q, w13_scale, args.group)
    w13 = build(
        stages[0], baseline.block_fp8_mm, w13_args, baseline.FFN_CONFIGS["w13"]
    )
    gate_up = w13(*w13_args)
    activation_args = (gate_up, args.group)
    activation = build(
        stages[1],
        baseline.silu_and_mul_per_block_quant,
        activation_args,
        baseline.FFN_CONFIGS["silu_quant"],
    )
    activation_q, activation_scale = activation(*activation_args)
    w2_args = (activation_q, activation_scale, w2_q, w2_scale, args.group)
    w2 = build(
        stages[2], baseline.block_fp8_mm, w2_args, baseline.FFN_CONFIGS["w2"]
    )
    separate_output = w2(*w2_args)
    torch.cuda.synchronize()

    for actual, expected, atol, rtol in (
        (megakernel_outputs[1], gate_up, 0.125, 3e-2),
        (megakernel_outputs[2], activation_q, 64.0, 3e-2),
        (megakernel_outputs[3], activation_scale, 2e-3, 3e-2),
        (megakernel_outputs[0], separate_output, 0.25, 5e-2),
    ):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=atol, rtol=rtol
        )

    def launch_separate():
        local_gate = w13(*w13_args)
        local_q, local_scale = activation(local_gate, args.group)
        return w2(local_q, local_scale, w2_q, w2_scale, args.group)

    def launch_megakernel():
        return megakernel_compiled(*kernel_args)[0]

    return Comparison(
        slug="qwen3_ffn",
        title="Qwen3 FFN: SM-level execution",
        stage_order=stages,
        root_stages=root_stages,
        separate_label="Standalone Helion CUDA graph (3 launches)",
        megakernel_label="CLC Helion megakernel CUDA graph (1 launch)",
        separate_launch=launch_separate,
        megakernel_launch=launch_megakernel,
        separate_traced=traced,
        megakernel=mega,
        reset=lambda: None,
        standalone_lowered=lowered,
        correctness="all FFN outputs and intermediates passed production tolerances",
    )


QWEN3_LAYER_STAGES = (
    "Residual add + pre-RMS reduction",
    "Pre-RMS apply + FP8 quant",
    "QKV projection",
    "Q/K norm + RoPE",
    "KV-cache update",
    "Paged-attention split",
    "Attention partial merge",
    "Attention final merge",
    "Attention-output FP8 quant",
    "O projection",
    "Post-attn residual + RMS reduction",
    "Post-RMS apply + FP8 quant",
    "Gate/up projection",
    "SwiGLU + FP8 quant",
    "Down projection",
)


def _qwen3_layer_args() -> argparse.Namespace:
    return argparse.Namespace(
        seed=0,
        batch=1,
        hidden=4096,
        intermediate=12288,
        q_heads=32,
        kv_heads=8,
        head_dim=128,
        context=8192,
        block_size=16,
        attention_splits=128,
        helion_comparison_splits=32,
        group=128,
        eps=1e-6,
        rope_theta=1_000_000.0,
        projection_stages=4,
        kernel_stages=2,
        maxnreg=None,
        worker_multiplier=8,
        cross_loop_workers=1024,
        merge_split_block=32,
        merge_q_block=4,
        attention_context_block=32,
        qk_head_block=1,
        config_path=str(
            REPO_ROOT / "probes/qwen3/qwen3_layer_helion_b200_configs.json"
        ),
    )


def _qwen3_granular_config(bound, args) -> helion.Config:
    values = dict(bound.config_spec.default_config())
    values.pop("cross_loop_num_workers", None)
    values.update(
        {
            "num_warps": 1,
            "num_stages": args.kernel_stages,
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": args.worker_multiplier,
        }
    )
    return _config_from_dict(bound, values)


def _qwen3_single_stage(kernel, kernel_args) -> str:
    name = kernel.fn.__name__
    if name == "block_fp8_mm":
        input_width = kernel_args[0].shape[1]
        output_width = kernel_args[2].shape[0]
        if output_width == 6144:
            return "QKV projection"
        if output_width == 24576:
            return "Gate/up projection"
        if input_width == 12288:
            return "Down projection"
        return "O projection"
    return {
        "fused_qk_norm_rope": "Q/K norm + RoPE",
        "tiled_reshape_and_cache_flash": "KV-cache update",
        "canonical_paged_gqa_decode_attention_split": "Paged-attention split",
        "paged_gqa_decode_attention_split": "Paged-attention split",
        "per_token_group_fp8_quant": "Attention-output FP8 quant",
        "silu_and_mul_per_block_quant": "SwiGLU + FP8 quant",
    }[name]


def _build_qwen3_layer() -> Comparison:
    from probes.qwen3 import helion_qwen3_granular_tile_dependency as granular
    from probes.qwen3 import helion_qwen3_layer_baseline as baseline
    from probes.qwen3 import helion_qwen3_tile_dependency as composite

    args = _qwen3_layer_args()
    root_stages = dict(enumerate(QWEN3_LAYER_STAGES))
    root_by_stage = {stage: root for root, stage in root_stages.items()}
    granular._USE_CANONICAL_ATTENTION_VIEWS = False
    granular._USE_TASK_ALIGNED_ATTENTION = False
    composite.build_helion_reference = granular._build_helion_reference
    composite.rms_norm_per_block_quant = granular.tiled_rms_norm_per_block_quant
    composite.reshape_and_cache_flash = granular.tiled_reshape_and_cache_flash
    composite.merge_attention_splits = granular.tiled_merge_attention_splits
    composite._probe_matched_config = granular._probe_config
    kernel, _ = composite._build_composite_kernel()

    tensors = baseline.allocate(args)
    initial_residual = tensors["residual"].clone()
    initial_cache = tensors["kv_cache"].clone()

    def reset():
        tensors["residual"].copy_(initial_residual)
        tensors["kv_cache"].copy_(initial_cache)

    traced: list[TracedCompiled] = []
    lowered: dict[str, str] = {}
    original_rms = granular.tiled_rms_norm_per_block_quant
    pre_rms = _clone_kernel(original_rms, "qwen3_trace_pre_rms")
    post_rms = _clone_kernel(original_rms, "qwen3_trace_post_rms")

    def compile_known(local_kernel, kernel_args, config_values):
        local_bound = local_kernel.bind(kernel_args)
        local_config = _config_from_dict(local_bound, config_values)
        stage = _qwen3_single_stage(local_kernel, kernel_args)
        compiled, source = _compile_stage(
            local_kernel,
            kernel_args,
            local_config,
            {0: stage},
            root_by_stage,
        )
        traced.append(compiled)
        lowered[stage] = source
        return local_config, compiled

    def compile_granular(local_kernel, kernel_args, local_args):
        if local_kernel is original_rms:
            pairs = (
                (
                    pre_rms,
                    {
                        0: "Residual add + pre-RMS reduction",
                        1: "Pre-RMS apply + FP8 quant",
                    },
                    "pre_rms",
                ),
                (
                    post_rms,
                    {
                        0: "Post-attn residual + RMS reduction",
                        1: "Post-RMS apply + FP8 quant",
                    },
                    "post_rms",
                ),
            )
            wrappers = []
            for cloned, stage_map, artifact_name in pairs:
                local_bound = cloned.bind(kernel_args)
                local_config = _qwen3_granular_config(local_bound, local_args)
                compiled, source = _compile_stage(
                    cloned,
                    kernel_args,
                    local_config,
                    stage_map,
                    root_by_stage,
                )
                traced.append(compiled)
                lowered[artifact_name] = source
                wrappers.append((local_config, compiled))
            pre_ptr = kernel_args[0].data_ptr()

            def dispatch(*call_args):
                selected = wrappers[0][1] if call_args[0].data_ptr() == pre_ptr else wrappers[1][1]
                return selected(*call_args)

            return wrappers[0][0], dispatch

        local_bound = local_kernel.bind(kernel_args)
        local_config = _qwen3_granular_config(local_bound, local_args)
        stage_map = (
            {0: "Attention partial merge", 1: "Attention final merge"}
            if local_kernel is granular.tiled_merge_attention_splits
            else {0: _qwen3_single_stage(local_kernel, kernel_args)}
        )
        compiled, source = _compile_stage(
            local_kernel,
            kernel_args,
            local_config,
            stage_map,
            root_by_stage,
        )
        traced.append(compiled)
        lowered[" + ".join(stage_map.values())] = source
        return local_config, compiled

    separate_args = argparse.Namespace(**vars(args))
    separate_args.attention_splits = args.helion_comparison_splits
    with (
        mock.patch.object(baseline, "compile_config", compile_known),
        mock.patch.object(granular, "_compile_granular_separate_kernel", compile_granular),
    ):
        separate_launch, _ = granular._build_helion_reference(separate_args, tensors)
    reset()

    mega_args = composite._composite_args(tensors, args)
    bound = kernel.bind(mega_args)
    config = granular._probe_config(bound, args)
    mega = compile_traced_megakernel(bound, config, root_stages)
    megakernel_compiled = mega["compiled"]

    reset()
    separate_values = separate_launch()
    torch.cuda.synchronize()
    separate_output = separate_values[0].clone()
    separate_residual = tensors["residual"].clone()
    reset()
    mega_values = megakernel_compiled(*mega_args)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        mega_values[0].float(), separate_output.float(), atol=0.25, rtol=0.05
    )
    torch.testing.assert_close(
        mega_values[-1].float(), separate_residual.float(), atol=0.125, rtol=0.03
    )

    return Comparison(
        slug="qwen3_decode_layer",
        title="Qwen3 decode layer: SM-level execution",
        stage_order=QWEN3_LAYER_STAGES,
        root_stages=root_stages,
        separate_label="Standalone Helion CUDA graph (12 launches, 32 splits)",
        megakernel_label="CLC Helion megakernel CUDA graph (1 launch, 128 splits)",
        separate_launch=separate_launch,
        megakernel_launch=lambda: megakernel_compiled(*mega_args),
        separate_traced=traced,
        megakernel=mega,
        reset=reset,
        standalone_lowered=lowered,
        correctness="output and mutable residual passed production tolerances",
    )


GEMMA4_E4B_STAGES = (
    "Input RMSNorm",
    "QKV projection",
    "Q/K norm + RoPE + cache",
    "Attention split",
    "Attention merge",
    "O projection",
    "Residual add + post-attn/pre-FF norms",
    "Gate/up projection",
    "GeGLU",
    "Down projection",
    "Post-FF residual + RMSNorm",
    "PLE gate",
    "PLE projection",
    "PLE residual + final RMSNorm/scale",
)


def _gemma4_e4b_args() -> argparse.Namespace:
    return argparse.Namespace(
        seed=0,
        layer=0,
        context=8192,
        block_size=16,
        sliding_splits=16,
        full_splits=64,
        attention_block=32,
        attention_heads=4,
        o_block_n=None,
        o_block_k=None,
        down_block_n=None,
        down_block_k=None,
        worker_multiplier=2,
        cross_loop_workers=296,
        num_warps=None,
        kernel_stages=None,
        config_mode="fused",
        tune=[],
        config_path=str(
            REPO_ROOT / "probes/gemma4/gemma4_e4b_b200_configs.json"
        ),
    )


def _e4b_stage_for_kernel(kernel, kernel_args) -> str | None:
    name = kernel.fn.__name__
    if name == "bf16_mm":
        input_width = kernel_args[0].shape[1]
        output_width = kernel_args[1].shape[0]
        if output_width == 3072:
            return "QKV projection"
        if output_width == 20480:
            return "Gate/up projection"
        if input_width == 10240:
            return "Down projection"
        if input_width == 256:
            return "PLE projection"
        if input_width == 2048 and output_width == 2560:
            return "O projection"
        return None
    return {
        "rms_norm": "Input RMSNorm",
        "qkv_norm_rope_cache": "Q/K norm + RoPE + cache",
        "paged_attention_split": "Attention split",
        "merge_attention": "Attention merge",
        "post_attention_residual_pre_ff_norm": (
            "Residual add + post-attn/pre-FF norms"
        ),
        "geglu": "GeGLU",
        "post_ff_residual": "Post-FF residual + RMSNorm",
        "ple_gate_gelu_mul": "PLE gate",
        "final_ple_norm_residual_scale": "PLE residual + final RMSNorm/scale",
    }.get(name)


def _build_gemma4_e4b() -> Comparison:
    from probes.gemma4 import helion_gemma4_e4b_layer as separate
    from probes.gemma4 import helion_gemma4_e4b_megakernel as megakernel
    from probes.gemma4.common import Gemma4E4BShape
    from probes.gemma4.common import allocate_layer
    from probes.gemma4.common import layer_reference

    args = _gemma4_e4b_args()
    shape = Gemma4E4BShape(context=args.context, block_size=args.block_size)
    geometry = shape.layer_geometry(args.layer)
    tensors = allocate_layer(shape, geometry, args.seed)
    reference = layer_reference(tensors, shape, geometry)
    initial_cache = tensors["kv_cache"].clone()

    def reset():
        tensors["kv_cache"].copy_(initial_cache)

    root_stages = dict(enumerate(GEMMA4_E4B_STAGES))
    root_by_stage = {stage: root for root, stage in root_stages.items()}
    configs = json.loads(Path(args.config_path).read_text())
    traced_by_stage: dict[str, TracedCompiled] = {}
    lowered: dict[str, str] = {}

    def compile_one(kernel, kernel_args, config):
        stage = _e4b_stage_for_kernel(kernel, kernel_args)
        bound = kernel.bind(kernel_args)
        if stage is None:
            return bound.compile_config(config)
        if stage in traced_by_stage:
            raise RuntimeError(f"duplicate traced E4B stage {stage}")
        compiled, source = _compile_stage(
            kernel,
            kernel_args,
            config,
            {0: stage},
            root_by_stage,
        )
        traced_by_stage[stage] = compiled
        lowered[stage] = source
        return compiled

    def compile_config(kernel, kernel_args, values):
        bound = kernel.bind(kernel_args)
        config = _config_from_dict(bound, values)
        return config, compile_one(kernel, kernel_args, config)

    def compile_default(kernel, kernel_args):
        bound = kernel.bind(kernel_args)
        config = bound.config_spec.default_config()
        return config, compile_one(kernel, kernel_args, config)

    with (
        mock.patch.object(separate, "compile_config", compile_config),
        mock.patch.object(separate, "compile_default", compile_default),
    ):
        built = separate.build_layer(
            args,
            tensors,
            shape,
            geometry,
            configs,
            Path(args.config_path),
        )
    if set(traced_by_stage) != set(GEMMA4_E4B_STAGES):
        raise RuntimeError(
            "standalone E4B stages differ: "
            f"missing={sorted(set(GEMMA4_E4B_STAGES) - set(traced_by_stage))}, "
            f"extra={sorted(set(traced_by_stage) - set(GEMMA4_E4B_STAGES))}"
        )
    reset()

    splits = args.full_splits if geometry.layer_type == "full" else args.sliding_splits
    kernel = (
        megakernel.SHARED_MEGAKERNEL
        if geometry.kv_shared
        else megakernel.NONSHARED_MEGAKERNEL
    )
    kernel_args = megakernel._megakernel_args(tensors, shape, geometry, splits)
    bound = kernel.bind(kernel_args)
    config = megakernel._megakernel_config(bound, args, geometry)
    mega = compile_traced_megakernel(bound, config, root_stages)
    megakernel_compiled = mega["compiled"]

    reset()
    separate_output = built["launch_optimized"]().clone()
    torch.cuda.synchronize()
    reset()
    mega_outputs = megakernel_compiled(*kernel_args)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        separate_output.float(), reference["output"].float(), atol=0.2, rtol=0.08
    )
    torch.testing.assert_close(
        mega_outputs[0].float(), reference["output"].float(), atol=0.2, rtol=0.08
    )
    torch.testing.assert_close(
        mega_outputs[1].float(),
        built["stage_outputs"]["query"].float(),
        atol=0.08,
        rtol=0.04,
    )
    torch.testing.assert_close(
        mega_outputs[2].float(),
        built["stage_outputs"]["attention"].float(),
        atol=0.15,
        rtol=0.06,
    )

    return Comparison(
        slug="gemma4_e4b_decode_layer",
        title="Gemma 4 E4B decode layer: SM-level execution",
        stage_order=GEMMA4_E4B_STAGES,
        root_stages=root_stages,
        separate_label="Standalone Helion CUDA graph (14 launches)",
        megakernel_label="CLC Helion megakernel CUDA graph (1 launch)",
        separate_launch=built["launch_optimized"],
        megakernel_launch=lambda: megakernel_compiled(*kernel_args)[0],
        separate_traced=[traced_by_stage[stage] for stage in GEMMA4_E4B_STAGES],
        megakernel=mega,
        reset=reset,
        standalone_lowered=lowered,
        correctness="output, query, attention, and cache update passed tolerances",
    )


GEMMA4_A4B_STAGE_ORDER = (
    "Expert pre-RMSNorm",
    "Router RMSNorm + scale",
    "Router projection",
    "Router RMSNorm + projection",
    "Top-k routing",
    "Top-k candidates",
    "Top-k merge",
    "Expert gate/up",
    "Expert pre-RMSNorm + gate/up",
    "Expert GeGLU",
    "Expert down",
    "Expert down + routing weight",
    "Weighted expert reduce",
    "MoE post-norm",
)


def _gemma4_a4b_args() -> argparse.Namespace:
    return argparse.Namespace(
        seed=0,
        batch=1,
        route_skew=2.0,
        workers=444,
        worker_multiplier=4,
        num_warps=4,
        kernel_stages=1,
        maxnreg=128,
        disable_warp_specialize=False,
        source_mode="assignment_hierarchical_topk_unfused_geglu",
        router_block=8,
        router_block_k=256,
        router_stages=3,
        gate_block=16,
        gate_block_k=256,
        gate_stages=3,
        gate_l2_grouping=1,
        geglu_block=128,
        down_block=64,
        down_block_k=64,
        down_stages=5,
        down_l2_grouping=1,
        reduce_block=256,
        group_gate_block=64,
        group_gate_block_k=128,
        group_down_block=64,
        group_down_block_k=64,
        group_reduce_block=64,
        group_use_tma=False,
        config_mode="matched",
        config_path=str(
            REPO_ROOT / "probes/gemma4/gemma4_a4b_moe_b200_configs.json"
        ),
    )


def _build_gemma4_a4b() -> Comparison:
    from probes.gemma4 import helion_gemma4_a4b_moe as separate
    from probes.gemma4 import helion_gemma4_a4b_moe_megakernel as megakernel
    from probes.gemma4.gemma4_a4b_moe_common import Gemma4A4BMoEShape
    from probes.gemma4.gemma4_a4b_moe_common import allocate_moe
    from probes.gemma4.gemma4_a4b_moe_common import moe_reference

    args = _gemma4_a4b_args()
    shape = Gemma4A4BMoEShape(batch=args.batch)
    tensors = allocate_moe(shape, args.seed, route_skew=args.route_skew)
    reference = moe_reference(tensors, shape)
    root_by_stage = {
        stage: root for root, stage in enumerate(GEMMA4_A4B_STAGE_ORDER)
    }
    configs = json.loads(Path(args.config_path).read_text())
    traced: list[TracedCompiled] = []
    lowered: dict[str, str] = {}

    def build(config_name, stage, kernel, kernel_args):
        bound = kernel.bind(kernel_args)
        config = (
            _config_from_dict(bound, configs[config_name])
            if config_name in configs
            else bound.config_spec.default_config()
        )
        compiled, source = _compile_stage(
            kernel,
            kernel_args,
            config,
            {0: stage},
            root_by_stage,
        )
        traced.append(compiled)
        lowered[stage] = source
        return compiled

    pre_norm_kernel = _clone_kernel(separate.rms_norm, "a4b_trace_expert_pre_norm")
    post_norm_kernel = _clone_kernel(separate.rms_norm, "a4b_trace_moe_post_norm")
    prefix = "moe_b1_"
    pre_args = (tensors["residual"], tensors["pre_ff_norm_weight_2"], shape.eps)
    pre = build(
        prefix + "expert_pre_norm",
        "Expert pre-RMSNorm",
        pre_norm_kernel,
        pre_args,
    )
    expert_input = pre(*pre_args)
    router_norm_args = (
        tensors["residual"],
        tensors["router_scale"],
        tensors["root_size"],
        shape.eps,
    )
    router_norm = build(
        prefix + "router_norm_scale",
        "Router RMSNorm + scale",
        separate.router_norm_scale,
        router_norm_args,
    )
    router_hidden = router_norm(*router_norm_args)
    router_args = (router_hidden, tensors["router_weight"])
    router = build(
        prefix + "router_mm_fp32",
        "Router projection",
        separate.router_mm_fp32,
        router_args,
    )
    router_logits = router(*router_args)
    topk_args = (router_logits, tensors["per_expert_scale"], shape.top_k)
    topk = build(
        prefix + "route_topk",
        "Top-k routing",
        separate.gemma4_route_topk,
        topk_args,
    )
    topk_weights, topk_ids = topk(*topk_args)
    gate_args = (expert_input, tensors["expert_gate_up_weight"], topk_ids)
    gate = build(
        prefix + "expert_gate_up",
        "Expert gate/up",
        separate.expert_gate_up,
        gate_args,
    )
    gate_up = gate(*gate_args)
    geglu_args = (gate_up,)
    geglu = build(
        prefix + "expert_geglu",
        "Expert GeGLU",
        separate.geglu,
        geglu_args,
    )
    activation = geglu(*geglu_args)
    down_args = (activation, tensors["expert_down_weight"], topk_ids)
    down = build(
        prefix + "expert_down",
        "Expert down",
        separate.expert_down,
        down_args,
    )
    expert_outputs = down(*down_args)
    reduce_args = (expert_outputs, topk_weights)
    reduce = build(
        prefix + "expert_reduce",
        "Weighted expert reduce",
        separate.weighted_expert_reduce,
        reduce_args,
    )
    moe_down = reduce(*reduce_args)
    post_args = (moe_down, tensors["post_ff_norm_weight_2"], shape.eps)
    post = build(
        prefix + "moe_post_norm",
        "MoE post-norm",
        post_norm_kernel,
        post_args,
    )
    separate_output = post(*post_args)

    mega_root_stages = {
        0: "Router RMSNorm + projection",
        1: "Top-k candidates",
        2: "Top-k merge",
        3: "Expert pre-RMSNorm + gate/up",
        4: "Expert GeGLU",
        5: "Expert down + routing weight",
        6: "Weighted expert reduce",
        7: "MoE post-norm",
    }
    kernel, _ = megakernel.MEGAKERNELS[args.source_mode]
    kernel_args = megakernel._megakernel_args(tensors, shape)
    bound = kernel.bind(kernel_args)
    config = megakernel._config(bound, args)
    mega = compile_traced_megakernel(bound, config, mega_root_stages)
    megakernel_compiled = mega["compiled"]
    mega_outputs = megakernel_compiled(*kernel_args)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        separate_output.float(), reference["moe_branch"].float(), atol=0.3, rtol=0.1
    )
    expected_expert_outputs = (
        reference["expert_outputs"].float()
        * reference["topk_weights"][:, :, None]
    ).to(reference["expert_outputs"].dtype)
    expected_by_name = {
        "moe_branch": reference["moe_branch"],
        "router_logits": reference["router_logits"],
        "topk_weights": reference["topk_weights"],
        "topk_ids": reference["topk_ids"],
        "activation": reference["expert_activation"],
        "expert_outputs": expected_expert_outputs,
        "moe_down": reference["moe_down"],
    }
    for name, actual in zip(
        megakernel._source_outputs(args.source_mode), mega_outputs, strict=True
    ):
        wanted = expected_by_name[name].reshape(actual.shape)
        if name == "topk_ids":
            torch.testing.assert_close(actual, wanted)
        else:
            torch.testing.assert_close(
                actual.float(), wanted.float(), atol=0.3, rtol=0.1
            )

    def launch_separate():
        local_expert_input = pre(*pre_args)
        local_router_hidden = router_norm(*router_norm_args)
        local_logits = router(local_router_hidden, tensors["router_weight"])
        local_weights, local_ids = topk(
            local_logits, tensors["per_expert_scale"], shape.top_k
        )
        local_gate = gate(
            local_expert_input, tensors["expert_gate_up_weight"], local_ids
        )
        local_activation = geglu(local_gate)
        local_outputs = down(
            local_activation, tensors["expert_down_weight"], local_ids
        )
        local_moe_down = reduce(local_outputs, local_weights)
        return post(
            local_moe_down, tensors["post_ff_norm_weight_2"], shape.eps
        )

    return Comparison(
        slug="gemma4_a4b_moe",
        title="Gemma 4 26B-A4B MoE: SM-level execution",
        stage_order=GEMMA4_A4B_STAGE_ORDER,
        root_stages=mega_root_stages,
        separate_label="Standalone Helion matched CUDA graph (9 launches)",
        megakernel_label=(
            "CLC Helion assignment/hierarchical-top-k megakernel (1 launch)"
        ),
        separate_launch=launch_separate,
        megakernel_launch=lambda: megakernel_compiled(*kernel_args)[0],
        separate_traced=traced,
        megakernel=mega,
        reset=lambda: None,
        standalone_lowered=lowered,
        correctness="standalone and megakernel outputs/intermediates passed MoE tolerances",
    )


CANONICAL_NEMOTRON_STAGE_BY_PREFIX = {
    "fused_add_rms_norm": "Fused add + RMSNorm",
    "router_mm_fp32": "Router projection",
    "route_topk": "Top-k routing",
    "route_candidates": "Top-k group candidates",
    "route_merge": "Top-k candidate merge",
    "expert_up": "Routed expert up",
    "expert_relu2": "Routed ReLU²",
    "expert_down_reduce": "Routed expert down + reduce",
    "shared_up_mm": "Shared expert up",
    "shared_relu2": "Shared ReLU²",
    "shared_down_mm": "Shared expert down",
    "scale_add": "Final scale + add",
}


def _build_canonical_nemotron3() -> Comparison:
    """Trace the canonical nemotron3_nano probe with this worktree's compiler."""
    import importlib
    import importlib.util

    common_path = NEMOTRON_ROOT / "common.py"
    common_spec = importlib.util.spec_from_file_location("common", common_path)
    if common_spec is None or common_spec.loader is None:
        raise RuntimeError(f"cannot load canonical Nemotron common module: {common_path}")
    common = importlib.util.module_from_spec(common_spec)
    sys.modules["common"] = common
    common_spec.loader.exec_module(common)
    separate = importlib.import_module("helion_nemotron3_nano_layer")
    megakernel = importlib.import_module("helion_nemotron3_nano_moe_megakernel")

    args = argparse.Namespace(
        seed=0,
        context=8192,
        branch_order="interleaved",
        route_mode="iterative",
        workers=592,
        worker_multiplier=4,
        num_warps=2,
        kernel_stages=2,
        maxnreg=None,
        config_overrides=None,
        all_pointer=True,
        block_size_override=None,
        range_stage_override=None,
        range_unroll_override=None,
        range_multi_buffer_override=None,
        range_flatten_override=None,
        indexing_override=None,
        eviction_override=None,
        l2_grouping_override=None,
        loop_order_override=None,
        configs=str(NEMOTRON_ROOT / "nemotron3_nano_b200_configs.json"),
    )
    shape = common.Nemotron3NanoShape(context=args.context)
    tensors = common.allocate_moe(shape, args.seed)
    reference = common.moe_reference(tensors, shape)
    configs = json.loads(Path(args.configs).read_text())

    invocations = megakernel._invocations(args.branch_order, args.route_mode)
    root_stages = {
        root: CANONICAL_NEMOTRON_STAGE_BY_PREFIX[invocation.prefix]
        for root, invocation in enumerate(invocations)
    }
    stage_order = tuple(root_stages.values())
    root_by_stage = {stage: root for root, stage in root_stages.items()}
    traced: list[TracedCompiled] = []
    lowered: dict[str, str] = {}
    stage_by_config_name = {
        invocation.config_name: CANONICAL_NEMOTRON_STAGE_BY_PREFIX[
            invocation.prefix
        ]
        for invocation in invocations
    }
    stage_by_config_name["moe_route_topk"] = "Top-k routing"

    def build(config_name, kernel, kernel_args):
        stage = stage_by_config_name[config_name]
        bound = kernel.bind(kernel_args)
        config = (
            _config_from_dict(bound, configs[config_name])
            if config_name in configs
            else bound.config_spec.default_config()
        )
        compiled, source = _compile_stage(
            kernel,
            kernel_args,
            config,
            {0: stage},
            root_by_stage,
        )
        traced.append(compiled)
        lowered[stage] = source
        return compiled

    baseline_args = argparse.Namespace(
        disable_shared_overlap=False,
        shared_overlap_start="after_up",
        shared_stream_priority=0,
    )
    launch_separate = separate.build_moe(
        baseline_args,
        tensors,
        shape,
        build,
    )

    kernel, _ = megakernel.MEGAKERNELS[(args.branch_order, args.route_mode)]
    kernel_args = megakernel._megakernel_args(tensors, shape)
    bound = kernel.bind(kernel_args)
    roots = megakernel._root_configs(
        tensors,
        shape,
        configs,
        args.branch_order,
        args.route_mode,
    )
    config, _, _ = megakernel._matched_config(bound, roots, args)
    mega = compile_traced_megakernel(bound, config, root_stages)
    megakernel_compiled = mega["compiled"]

    separate_output, separate_residual = launch_separate()
    mega_output, mega_residual = megakernel_compiled(*kernel_args)
    torch.cuda.synchronize()
    for actual in (separate_output, mega_output):
        separate.assert_close(
            actual,
            reference["hidden_states"],
            atol=0.5,
            rtol=0.18,
        )
    for actual in (separate_residual, mega_residual):
        separate.assert_close(
            actual,
            reference["residual"],
            atol=0.025,
            rtol=0.025,
        )

    return Comparison(
        slug="nemotron3_nano_moe",
        title="Nemotron-3 Nano MoE: SM-level execution",
        stage_order=stage_order,
        root_stages=root_stages,
        separate_label="Canonical standalone Helion CUDA graph (10 launches)",
        megakernel_label="Current CLC Helion megakernel CUDA graph (1 launch)",
        separate_launch=launch_separate,
        megakernel_launch=lambda: megakernel_compiled(*kernel_args),
        separate_traced=traced,
        megakernel=mega,
        reset=lambda: None,
        standalone_lowered=lowered,
        correctness="canonical standalone and megakernel outputs passed tolerances",
    )


CANONICAL_DEEPSEEK_V3_STAGES = (
    "Router projection",
    "Top-k routing",
    "Routed W13",
    "Routed SwiGLU",
    "Routed W2",
    "Weighted reduce",
    "Shared W13",
    "Shared SwiGLU",
    "Shared W2",
    "Final add",
)


def _build_canonical_deepseek_v3() -> Comparison:
    """Trace the original DeepSeek-V3 probe with this worktree's compiler."""
    from probes.deepseek_v3.deepseek_v3_moe_common import DeepseekV3MoEShape
    from probes.deepseek_v3.deepseek_v3_moe_common import allocate_moe
    from probes.deepseek_v3.deepseek_v3_moe_common import moe_reference
    import probes.deepseek_v3.helion_deepseek_v3_moe as separate
    from probes.deepseek_v3 import helion_deepseek_v3_moe_megakernel as probe

    args = argparse.Namespace(
        seed=0,
        workers=592,
        worker_multiplier=8,
        num_warps=1,
        kernel_stages=2,
        router_block=2,
        router_k=512,
        router_stages=4,
        expert_w13_block=16,
        expert_w13_k=512,
        expert_w13_stages=4,
        expert_w2_block=32,
        expert_w2_k=512,
        expert_w2_stages=2,
        shared_w13_block=16,
        shared_w13_k=512,
        shared_w13_stages=2,
        shared_w2_block=32,
        shared_w2_k=256,
        shared_w2_stages=3,
        activation_block=256,
        reduce_block=256,
        add_block=256,
    )
    shape = DeepseekV3MoEShape(batch=1)
    tensors = allocate_moe(shape, args.seed)
    reference = moe_reference(tensors, shape)
    config_path = CROSS_KERNEL_ROOT / "probes/deepseek_v3/deepseek_v3_moe_b200_configs.json"
    configs = json.loads(config_path.read_text())
    root_stages = dict(enumerate(CANONICAL_DEEPSEEK_V3_STAGES))
    root_by_stage = {stage: root for root, stage in root_stages.items()}
    traced: list[TracedCompiled] = []
    lowered: dict[str, str] = {}

    def build(config_name, stage, kernel, kernel_args):
        bound = kernel.bind(kernel_args)
        config = (
            _config_from_dict(bound, configs[config_name])
            if config_name in configs
            else bound.config_spec.default_config()
        )
        compiled, source = _compile_stage(
            kernel,
            kernel_args,
            config,
            {0: stage},
            root_by_stage,
        )
        traced.append(compiled)
        lowered[stage] = source
        return compiled

    hidden = tensors["hidden_states"]
    router_args = (hidden, tensors["router_weight"])
    router = build("router_mm_fp32", root_stages[0], separate.router_mm_fp32, router_args)
    router_logits = router(*router_args)
    topk_args = (
        router_logits,
        tensors["correction_bias"],
        shape.top_k,
        shape.num_groups,
        shape.topk_groups,
        shape.routed_scale,
    )
    topk = build("grouped_topk", root_stages[1], separate.grouped_topk, topk_args)
    topk_weights, topk_ids = topk(*topk_args)
    routed_w13_args = (hidden, tensors["expert_w13"], topk_ids)
    routed_w13 = build(
        "expert_w13",
        root_stages[2],
        separate.selected_expert_w13,
        routed_w13_args,
    )
    expert_gate_up = routed_w13(*routed_w13_args)
    routed_activation = build(
        "expert_swiglu",
        root_stages[3],
        separate.silu_and_mul,
        (expert_gate_up,),
    )
    expert_activation = routed_activation(expert_gate_up)
    routed_w2_args = (expert_activation, tensors["expert_w2"], topk_ids)
    routed_w2 = build(
        "expert_w2",
        root_stages[4],
        separate.selected_expert_w2,
        routed_w2_args,
    )
    expert_outputs = routed_w2(*routed_w2_args)
    reduce_args = (expert_outputs, topk_weights)
    reduce = build(
        "expert_reduce",
        root_stages[5],
        separate.weighted_reduce,
        reduce_args,
    )
    routed_output = reduce(*reduce_args)
    shared_w13_args = (hidden, tensors["shared_w13"])
    shared_w13 = build(
        "shared_w13",
        root_stages[6],
        separate.bf16_mm,
        shared_w13_args,
    )
    shared_gate_up = shared_w13(*shared_w13_args)
    shared_activation_kernel = build(
        "shared_swiglu",
        root_stages[7],
        separate.silu_and_mul,
        (shared_gate_up,),
    )
    shared_activation = shared_activation_kernel(shared_gate_up)
    shared_w2_args = (shared_activation, tensors["shared_w2"])
    shared_w2 = build(
        "shared_w2",
        root_stages[8],
        separate.bf16_mm,
        shared_w2_args,
    )
    shared_output = shared_w2(*shared_w2_args)
    join_args = (routed_output, shared_output)
    join = build(
        "final_add",
        root_stages[9],
        separate.add_outputs,
        join_args,
    )
    join(*join_args)

    shared_stream = torch.cuda.Stream()

    def launch_separate():
        current = torch.cuda.current_stream()
        shared_stream.wait_stream(current)
        local_logits = router(hidden, tensors["router_weight"])
        with torch.cuda.stream(shared_stream):
            local_shared_gate = shared_w13(hidden, tensors["shared_w13"])
            local_shared_activation = shared_activation_kernel(local_shared_gate)
            local_shared = shared_w2(local_shared_activation, tensors["shared_w2"])
        local_weights, local_ids = topk(
            local_logits,
            tensors["correction_bias"],
            shape.top_k,
            shape.num_groups,
            shape.topk_groups,
            shape.routed_scale,
        )
        local_gate = routed_w13(hidden, tensors["expert_w13"], local_ids)
        local_activation = routed_activation(local_gate)
        local_expert_outputs = routed_w2(
            local_activation,
            tensors["expert_w2"],
            local_ids,
        )
        local_routed = reduce(local_expert_outputs, local_weights)
        current.wait_stream(shared_stream)
        return join(local_routed, local_shared)

    kernel_args = probe.megakernel_args(tensors, shape)
    bound = probe.MEGAKERNEL.bind(kernel_args)
    config = probe.persistent_config(bound, args)
    mega = compile_traced_megakernel(bound, config, root_stages)
    megakernel_compiled = mega["compiled"]
    separate_output = launch_separate()
    mega_outputs = megakernel_compiled(*kernel_args)
    torch.cuda.synchronize()
    probe._validate(mega_outputs, reference)
    torch.testing.assert_close(
        separate_output.float(),
        reference["output"].float(),
        atol=0.2,
        rtol=0.08,
    )

    return Comparison(
        slug="deepseek_v3_moe",
        title="DeepSeek-V3 decode MoE: SM-level execution",
        stage_order=CANONICAL_DEEPSEEK_V3_STAGES,
        root_stages=root_stages,
        separate_label="Canonical standalone Helion CUDA graph (10 launches)",
        megakernel_label="Current CLC Helion megakernel CUDA graph (1 launch)",
        separate_launch=launch_separate,
        megakernel_launch=lambda: megakernel_compiled(*kernel_args),
        separate_traced=traced,
        megakernel=mega,
        reset=lambda: None,
        standalone_lowered=lowered,
        correctness="canonical standalone and megakernel outputs passed tolerances",
    )


MUSE_SLIDING_STAGE_SPECS = (
    ("input_rms_norm", "Input RMSNorm"),
    ("qkv_splitk_main", "QKV split-K main"),
    ("qkv_splitk_reduce", "QKV split-K reduce"),
    ("q_stats_and_key_rope", "Q statistics + K RoPE"),
    ("query_from_stats_rope", "Query normalize + RoPE"),
    ("reshape_and_cache_flash", "KV-cache update"),
    ("flashinfer_split_attention", "Paged-attention split"),
    ("flashinfer_merge_states", "Attention merge"),
    ("attention_gate_splitk_main", "Attention-gate split-K main"),
    ("attention_gate_splitk_reduce", "Attention-gate split-K reduce"),
    ("sigmoid_attention_mul", "Sigmoid x attention"),
    ("o_projection", "O projection"),
    ("post_attention_state", "Post-attention residual + norms"),
    ("gate_up_splitk_main", "Gate/up split-K main"),
    ("gate_up_splitk_reduce", "Gate/up split-K reduce"),
    ("silu_and_mul", "SiLU x up"),
    ("down_splitk_main", "Down split-K main"),
    ("down_splitk_reduce", "Down split-K reduce"),
    ("final_recompute_residual", "Final residual"),
)

MUSE_FULL_STAGE_SPECS = (
    *MUSE_SLIDING_STAGE_SPECS[:3],
    ("qk_nope", "Q/K normalize (NoPE)"),
    *MUSE_SLIDING_STAGE_SPECS[5:],
)


def _muse_args(layer_idx: int, workers: int) -> argparse.Namespace:
    return argparse.Namespace(
        layer=layer_idx,
        exact_sliding_splits=17,
        exact_full_splits=64,
        split_k=16,
        down_split_k=16,
        worker_multiplier=4,
        cross_loop_workers=workers,
        num_warps=4,
        kernel_stages=2,
        maxnreg=None,
        qkv_block_n=64,
        gate_block_n=128,
        o_block_n=64,
        o_block_k=128,
        gate_up_block_n=128,
        down_block_n=128,
        projection_block_k=128,
        ffn_block_k=256,
        reduce_block=256,
        activation_block=256,
        attention_block=64,
        config_mode="coarse",
        ignore_stored_config=True,
        tune_megakernel=False,
        megakernel_config_path=str(
            REPO_ROOT / "probes/muse_glimmer_cross_source_clc/unused_configs.json"
        ),
    )


def _build_muse_glimmer(layer_idx: int, workers: int) -> Comparison:
    if canonical_probe_package in probes.__path__:
        probes.__path__.remove(canonical_probe_package)
    probes.__path__.insert(0, canonical_probe_package)
    sys.modules.pop("probes.common", None)

    import helion_muse_glimmer_exact as separate
    import helion_muse_glimmer_megakernel as probe
    import muse_glimmer_common as common

    for module in (common, separate, probe):
        if MUSE_GLIMMER_ROOT.resolve() not in Path(module.__file__).resolve().parents:
            raise RuntimeError(
                f"loaded the wrong Muse-Glimmer source: {module.__file__}"
            )

    args = _muse_args(layer_idx, workers)
    shape = common.MuseGlimmerShape(context=8192, block_size=16)
    geometry = shape.layer_geometry(layer_idx)
    stage_specs = (
        MUSE_SLIDING_STAGE_SPECS if geometry.use_rope else MUSE_FULL_STAGE_SPECS
    )
    stages = tuple(label for _, label in stage_specs)
    expected_signatures = tuple(name for name, _ in stage_specs)
    root_stages = dict(enumerate(stages))
    root_by_stage = {stage: root for root, stage in root_stages.items()}

    mega_tensors = common.allocate_layer(shape, geometry, seed=0)
    reference = common.layer_reference(mega_tensors, shape, geometry)
    kernel = (
        probe.SLIDING_MEGAKERNEL if geometry.use_rope else probe.FULL_MEGAKERNEL
    )
    kernel_args, production_cache = probe._prepare_megakernel(
        mega_tensors,
        shape,
        geometry,
        args,
    )
    initial_production_cache = production_cache.clone()
    bound = kernel.bind(kernel_args)
    config = probe._config(bound, args, geometry)
    mega = compile_traced_megakernel(bound, config, root_stages)
    megakernel_compiled = mega["compiled"]

    config_path = MUSE_GLIMMER_ROOT / "muse_glimmer_b200_configs.json"
    standalone_configs = json.loads(config_path.read_text())
    exact_tensors = common.allocate_layer(shape, geometry, seed=0)
    traced: list[TracedCompiled] = []
    lowered: dict[str, str] = {}
    next_stage = 0

    def compile_exact_stage(kernel, kernel_args, config_values=None):
        nonlocal next_stage
        if next_stage >= len(stages):
            raise RuntimeError("standalone emitted more kernels than expected")
        stage = stages[next_stage]
        next_stage += 1
        local_bound = kernel.bind(kernel_args)
        if config_values is None:
            local_config = local_bound.config_spec.default_config()
        else:
            local_config = _config_from_dict(local_bound, config_values)
        lowered[stage] = local_bound.to_triton_code(
            local_config,
            output_origin_lines=True,
        )
        compiled = compile_traced(
            local_bound,
            local_config,
            {0: stage},
            root_by_stage=root_by_stage,
        )
        traced.append(compiled)
        return local_config, compiled

    exact_args = argparse.Namespace(
        tune_exact=[],
        qkv_split_k=args.split_k,
        gate_split_k=args.split_k,
        gate_up_split_k=args.split_k,
        down_split_k=args.split_k,
        exact_sliding_splits=args.exact_sliding_splits,
        exact_full_splits=args.exact_full_splits,
    )
    with (
        mock.patch.object(
            separate,
            "compile_config",
            side_effect=lambda kernel, kernel_args, values: compile_exact_stage(
                kernel,
                kernel_args,
                values,
            ),
        ),
        mock.patch.object(
            separate,
            "compile_default",
            side_effect=lambda kernel, kernel_args: compile_exact_stage(
                kernel,
                kernel_args,
            ),
        ),
    ):
        exact = separate.build_exact(
            exact_args,
            exact_tensors,
            shape,
            geometry,
            standalone_configs,
            config_path,
        )

    actual_signatures = tuple(item["name"] for item in exact["signatures"])
    if actual_signatures != expected_signatures:
        raise RuntimeError(
            f"standalone signature order {actual_signatures} does not match "
            f"expected {expected_signatures}"
        )
    if next_stage != len(stages) or len(traced) != exact["launch_count"]:
        raise RuntimeError(
            f"traced {len(traced)} of {exact['launch_count']} standalone launches"
        )

    closure = inspect.getclosurevars(exact["launch_exact"]).nonlocals
    exact_cache_args = closure["cache_args"]
    exact_key_cache = exact_cache_args[2]
    exact_value_cache = exact_cache_args[3]
    exact_initial_cache = (
        exact_tensors["kv_cache"].permute(0, 2, 1, 3).contiguous()
    )
    exact_initial_key, exact_initial_value = exact_initial_cache.split(
        shape.head_dim,
        dim=-1,
    )

    def reset():
        exact_key_cache.copy_(exact_initial_key)
        exact_value_cache.copy_(exact_initial_value)
        production_cache.copy_(initial_production_cache)

    reset()
    separate_output = exact["launch_exact"]()
    reset()
    mega_outputs = megakernel_compiled(*kernel_args)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        separate_output.float(),
        reference["output"].float(),
        atol=0.12,
        rtol=0.08,
    )
    correctness = probe._validate(
        mega_outputs,
        reference,
        production_cache,
        mega_tensors,
        shape,
    )
    reset()

    return Comparison(
        slug=f"muse_glimmer_{geometry.layer_type}",
        title=f"Muse-Glimmer {geometry.layer_type} decode: SM-level execution",
        stage_order=stages,
        root_stages=root_stages,
        separate_label=(
            f"Standalone Helion CUDA graph ({exact['launch_count']} launches)"
        ),
        megakernel_label="helion-clc CLC megakernel CUDA graph (1 launch)",
        separate_launch=exact["launch_exact"],
        megakernel_launch=lambda: megakernel_compiled(*kernel_args)[0],
        separate_traced=traced,
        megakernel=mega,
        reset=reset,
        standalone_lowered=lowered,
        correctness=(
            "standalone output and all megakernel outputs/intermediates/cache "
            f"passed production tolerances ({len(correctness)} checks)"
        ),
    )


def _build_muse_glimmer_sliding() -> Comparison:
    return _build_muse_glimmer(layer_idx=0, workers=288)


def _build_muse_glimmer_full() -> Comparison:
    return _build_muse_glimmer(layer_idx=3, workers=296)


BUILDERS = {
    "qwen3-ffn": _build_qwen3_ffn,
    "qwen3-layer": _build_qwen3_layer,
    "gemma4-e4b": _build_gemma4_e4b,
    "gemma4-a4b": _build_gemma4_a4b,
    "nemotron3-moe": _build_canonical_nemotron3,
    "deepseek-v3-moe": _build_canonical_deepseek_v3,
    "muse-glimmer-sliding": _build_muse_glimmer_sliding,
    "muse-glimmer-full": _build_muse_glimmer_full,
}


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_artifacts(comparison: Comparison, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"{comparison.slug}_separate_vs_megakernel_sm_gantt.png"
    intervals_path = chart_path.with_name(f"{chart_path.stem}_intervals.json")
    result_path = chart_path.with_suffix(".json")
    lowered_dir = chart_path.with_name(f"{chart_path.stem}_lowered")
    lowered_dir.mkdir(parents=True, exist_ok=True)

    separate_graph, _ = capture_with_reset(
        comparison.separate_launch, comparison.reset
    )
    megakernel_graph, _ = capture_with_reset(
        comparison.megakernel_launch, comparison.reset
    )
    separate, l2_bytes = trace_separate(
        separate_graph,
        comparison.separate_traced,
        comparison.reset,
    )
    mega = comparison.megakernel
    megakernel, _ = trace_megakernel(
        megakernel_graph,
        mega["compiled"],
        mega["trace_numel"],
        mega["task_counts"],
        comparison.root_stages,
        comparison.reset,
    )
    render_stacked_gantt(
        separate,
        megakernel,
        chart_path,
        title_text=comparison.title,
        stage_order=comparison.stage_order,
        separate_label=comparison.separate_label,
        megakernel_label=comparison.megakernel_label,
    )

    standalone_paths = {}
    for index, (name, source) in enumerate(comparison.standalone_lowered.items()):
        path = lowered_dir / f"{index:02d}_{safe_name(name)}.py"
        path.write_text(source)
        standalone_paths[name] = str(path)
    mega_untraced = lowered_dir / "megakernel_untraced.py"
    mega_traced = lowered_dir / "megakernel_traced.py"
    mega_untraced.write_text(mega["untraced_lowered"])
    mega_traced.write_text(mega["traced_lowered"])
    intervals_path.write_text(
        json.dumps(
            {
                "separate": serialize_intervals(separate),
                "megakernel": serialize_intervals(megakernel),
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    result = {
        "workload": comparison.slug,
        "device": torch.cuda.get_device_name(),
        "helion_module": str(Path(helion.__file__).resolve()),
        "git_commit": _git_commit(),
        "cache_state": "cold_l2",
        "l2_flush_bytes": l2_bytes,
        "correctness": comparison.correctness,
        "trace_note": "instrumented tile span is not CUDA-event latency",
        "separate": summarize(separate, comparison.stage_order),
        "megakernel": summarize(megakernel, comparison.stage_order),
        "megakernel_config": mega["config"],
        "artifacts": {
            "gantt": str(chart_path),
            "intervals": str(intervals_path),
            "standalone_lowered": standalone_paths,
            "megakernel_untraced_lowered": str(mega_untraced),
            "megakernel_traced_lowered": str(mega_traced),
        },
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("SM_GANTT_RESULT", json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=(*BUILDERS, "muse-glimmer"),
        required=True,
        help="Use muse-glimmer to generate both representative layer variants.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-busy", action="store_true")
    args = parser.parse_args()
    if not args.allow_busy:
        require_idle_visible_gpu()
    if args.model == "muse-glimmer":
        for model in ("muse-glimmer-sliding", "muse-glimmer-full"):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--model",
                model,
                "--output-dir",
                str(args.output_dir.resolve()),
            ]
            if args.allow_busy:
                command.append("--allow-busy")
            subprocess.run(command, check=True)
        return
    comparison = BUILDERS[args.model]()
    _write_artifacts(comparison, args.output_dir.resolve())


if __name__ == "__main__":
    main()
