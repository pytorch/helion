# ruff: noqa: ANN001, ANN202, E402
"""Production-shape DeepSeek-V3 MoE standalone-kernel benchmark.

The Helion and vLLM paths use the same model tensors and preserve the same
operator boundaries:

  router -> grouped top-k -> routed W13 -> SwiGLU -> routed W2 -> reduce
  shared W13 -> SwiGLU -> shared W2
  routed + shared

The routed and shared branches are also measured with the shared branch on an
auxiliary stream, matching vLLM's production decode execution policy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("HELION_BACKEND", "triton")
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

REPO_ROOT = Path(__file__).resolve().parents[2]
VLLM_ROOT = Path(os.environ.get("VLLM_ROOT", REPO_ROOT.parent / "vllm")).resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(VLLM_ROOT))

import torch

from probes.common import benchmark_graphs_cold_l2
from probes.common import capture_with_reset
from probes.common import error_stats
from probes.common import gpu_snapshot
from probes.common import lowered_triton_summary
from probes.common import require_idle_visible_gpu
from probes.common import visible_gpu_pids
from probes.deepseek_v3.deepseek_v3_moe_common import DeepseekV3MoEShape
from probes.deepseek_v3.deepseek_v3_moe_common import allocate_moe
from probes.deepseek_v3.deepseek_v3_moe_common import moe_reference
from probes.deepseek_v3.deepseek_v3_moe_common import routing_histogram
from probes.deepseek_v3.helion_deepseek_v3_moe import build_moe

import helion


def _git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _initialize_vllm(args):
    from vllm.config import CacheConfig
    from vllm.config import CUDAGraphMode
    from vllm.config import VllmConfig
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment
    from vllm.distributed import initialize_model_parallel
    from vllm.utils.network_utils import get_open_port
    from vllm.v1.worker.workspace import init_workspace_manager

    torch.cuda.set_device(0)
    init_workspace_manager(torch.device("cuda"))
    cache_config = CacheConfig(block_size=16, cache_dtype="auto")
    cache_config.num_gpu_blocks = 1
    vllm_config = VllmConfig(cache_config=cache_config)
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    vllm_config.model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        is_moe=True,
        is_mm_prefix_lm=False,
        is_diffusion=False,
        is_hybrid=False,
        is_attention_free=False,
        runner_type="generate",
        architectures=["DeepseekV3ForCausalLM"],
        max_model_len=4096,
        compute_hash=lambda: "deepseek-v3-moe-standalone-benchmark",
    )
    if args.moe_backend != "auto":
        vllm_config.kernel_config.moe_backend = args.moe_backend
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
    )
    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(1, 1)
    return vllm_config


def _share_parameter(parameter: torch.nn.Parameter, tensor: torch.Tensor) -> None:
    if parameter.shape != tensor.shape:
        raise ValueError(
            f"parameter shape {parameter.shape} != tensor shape {tensor.shape}"
        )
    parameter.data = tensor


def _build_vllm(tensors, shape, vllm_config):
    from transformers import DeepseekV3Config
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE

    config = DeepseekV3Config()
    config.hidden_size = shape.hidden
    config.moe_intermediate_size = shape.intermediate
    config.n_routed_experts = shape.num_experts
    config.n_shared_experts = 1
    config.num_experts_per_tok = shape.top_k
    config.n_group = shape.num_groups
    config.topk_group = shape.topk_groups
    config.topk_method = "noaux_tc"
    config.norm_topk_prob = True
    config.routed_scaling_factor = shape.routed_scale
    config.scoring_func = "sigmoid"
    config.hidden_act = "silu"

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with set_current_vllm_config(vllm_config), torch.device("cuda"):
            layer = DeepseekV2MoE(
                config,
                vllm_config.parallel_config,
                prefix="model.layers.3.mlp",
            ).eval()
    finally:
        torch.set_default_dtype(old_dtype)

    _share_parameter(layer.gate.weight, tensors["router_weight"])
    assert layer.gate.e_score_correction_bias is not None
    _share_parameter(
        layer.gate.e_score_correction_bias,
        tensors["correction_bias"],
    )
    assert layer.shared_experts is not None
    _share_parameter(
        layer.shared_experts.gate_up_proj.weight,
        tensors["shared_w13"],
    )
    _share_parameter(
        layer.shared_experts.down_proj.weight,
        tensors["shared_w2"],
    )
    routed = layer.experts.routed_experts
    _share_parameter(routed.w13_weight, tensors["expert_w13"])
    _share_parameter(routed.w2_weight, tensors["expert_w2"])
    routed.quant_method.process_weights_after_loading(routed)

    # The CuteDSL router path cannot currently participate in the graph
    # capture used by this harness.  The next production fallback preserves
    # the same BF16xBF16 router GEMM boundary.
    layer.gate.allow_ll_bf16_gemm = False

    def launch():
        with set_forward_context(
            None,
            vllm_config=vllm_config,
            num_tokens=shape.batch,
            slot_mapping=None,
        ):
            return layer(tensors["hidden_states"])

    return layer, launch


def _validate(actual, expected, *, atol, rtol):
    if actual.dtype in (torch.int32, torch.int64):
        if not torch.equal(actual, expected.to(actual.dtype)):
            raise AssertionError(f"integer tensors differ: {actual} != {expected}")
        return {"exact": True}
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
    return error_stats(actual, expected)


def _validate_helion_stages(stages, reference):
    tolerances = {
        "router_logits": (0.05, 0.02),
        "topk_weights": (2e-5, 2e-5),
        "topk_ids": (0.0, 0.0),
        "expert_gate_up": (0.08, 0.04),
        "expert_activation": (0.08, 0.04),
        "expert_outputs": (0.12, 0.06),
        "routed_output": (0.15, 0.06),
        "shared_gate_up": (0.08, 0.04),
        "shared_activation": (0.08, 0.04),
        "shared_output": (0.12, 0.06),
        "output": (0.2, 0.08),
    }
    return {
        name: _validate(stages[name], reference[name], atol=atol, rtol=rtol)
        for name, (atol, rtol) in tolerances.items()
    }


def _write_lowerings(directory: Path, lowerings: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, source in lowerings.items():
        (directory / f"{name}.py").write_text(source)


def _task_geometry(configs, shape, sm_count: int) -> dict[str, dict[str, int]]:
    stage_outputs = {
        "expert_w13": shape.top_k * 2 * shape.intermediate,
        "expert_w2": shape.top_k * shape.hidden,
        "shared_w13": 2 * shape.intermediate,
        "shared_w2": shape.hidden,
    }
    result = {}
    for name, outputs in stage_outputs.items():
        block_n = int(configs[name]["block_sizes"][0])
        tasks = (outputs + block_n - 1) // block_n
        result[name] = {
            "output_elements": outputs,
            "output_block": block_n,
            "ctas": tasks,
            "full_waves": tasks // sm_count,
            "tail_ctas": tasks % sm_count,
        }
    return result


@torch.inference_mode()
def run(args) -> dict[str, Any]:
    require_idle_visible_gpu()
    shape = DeepseekV3MoEShape(batch=1)
    tensors = allocate_moe(shape, args.seed)
    reference = moe_reference(tensors, shape)
    configs_path = Path(args.config_path)
    configs = json.loads(configs_path.read_text())

    vllm_config = _initialize_vllm(args)
    from vllm.distributed import destroy_distributed_environment
    from vllm.distributed import destroy_model_parallel
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        fused_grouped_topk,
    )

    try:
        vllm_layer, vllm_launch = _build_vllm(tensors, shape, vllm_config)
        helion_args = argparse.Namespace(tune=args.tune)
        helion_build = build_moe(
            helion_args,
            tensors,
            shape,
            configs,
            configs_path,
        )

        with set_forward_context(
            None,
            vllm_config=vllm_config,
            num_tokens=shape.batch,
            slot_mapping=None,
        ):
            vllm_logits = vllm_layer.gate(tensors["hidden_states"])
            if isinstance(vllm_logits, tuple):
                vllm_logits = vllm_logits[0]
            vllm_weights, vllm_ids = fused_grouped_topk(
                tensors["hidden_states"],
                vllm_logits,
                shape.top_k,
                True,
                tensors["correction_bias"],
                shape.num_groups,
                shape.topk_groups,
                "sigmoid",
                shape.routed_scale,
            )
        torch.cuda.synchronize()

        correctness = {
            "helion_stages": _validate_helion_stages(
                helion_build["stage_outputs"], reference
            ),
            "vllm_routing": {
                "logits": error_stats(vllm_logits, reference["router_logits"]),
                "weights": _validate(
                    vllm_weights,
                    reference["topk_weights"],
                    atol=2e-3,
                    rtol=0.02,
                ),
                "ids": _validate(
                    vllm_ids,
                    reference["topk_ids"],
                    atol=0,
                    rtol=0,
                ),
            },
        }

        launches = {
            "helion_cudagraph_serial": helion_build["launch_serial"],
            "helion_cudagraph_shared_overlap": helion_build["launch_overlap"],
            "vllm_cudagraph": vllm_launch,
        }
        eager_outputs = {name: launch().clone() for name, launch in launches.items()}
        torch.cuda.synchronize()
        correctness["torch_reference"] = {
            name: _validate(output, reference["output"], atol=0.2, rtol=0.08)
            for name, output in eager_outputs.items()
        }
        correctness["vllm_reference"] = {
            name: _validate(
                output,
                eager_outputs["vllm_cudagraph"],
                atol=0.2,
                rtol=0.08,
            )
            for name, output in eager_outputs.items()
            if name != "vllm_cudagraph"
        }

        def noop() -> None:
            pass

        graphs = {}
        graph_outputs = {}
        for name, launch in launches.items():
            graphs[name], graph_outputs[name] = capture_with_reset(launch, noop)
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        correctness["cuda_graph"] = {
            name: _validate(output, reference["output"], atol=0.2, rtol=0.08)
            for name, output in graph_outputs.items()
        }

        process_set = visible_gpu_pids()
        telemetry_before = gpu_snapshot()
        timings = benchmark_graphs_cold_l2(
            {name: (graph.replay, noop) for name, graph in graphs.items()},
            args.repeats,
            flush_mib=args.l2_flush_mib,
            order_seed=args.order_seed,
        )
        stage_timings = {}
        if args.stage_repeats:
            stage_graphs = {
                name: capture_with_reset(launch, noop)[0]
                for name, launch in helion_build["stage_launches"].items()
            }
            stage_timings = benchmark_graphs_cold_l2(
                {name: (graph.replay, noop) for name, graph in stage_graphs.items()},
                args.stage_repeats,
                flush_mib=args.l2_flush_mib,
                order_seed=args.order_seed,
            )
        telemetry_after = gpu_snapshot()
        if visible_gpu_pids() != process_set:
            raise RuntimeError("GPU process set changed during benchmark")

        lowerings_dir = Path(args.lowerings_dir)
        _write_lowerings(lowerings_dir, helion_build["lowerings"])
        quant_method = vllm_layer.experts.routed_experts.quant_method
        selected_backend = getattr(quant_method, "unquantized_backend", None)
        same_storage = {
            "router_weight": (
                vllm_layer.gate.weight.data_ptr() == tensors["router_weight"].data_ptr()
            ),
            "expert_w13": (
                vllm_layer.experts.routed_experts.w13_weight.data_ptr()
                == tensors["expert_w13"].data_ptr()
            ),
            "expert_w2": (
                vllm_layer.experts.routed_experts.w2_weight.data_ptr()
                == tensors["expert_w2"].data_ptr()
            ),
            "shared_w13": (
                vllm_layer.shared_experts.gate_up_proj.weight.data_ptr()
                == tensors["shared_w13"].data_ptr()
            ),
            "shared_w2": (
                vllm_layer.shared_experts.down_proj.weight.data_ptr()
                == tensors["shared_w2"].data_ptr()
            ),
        }
        result = {
            "model": "deepseek-ai/DeepSeek-V3",
            "component": "decode_moe_tp1_bf16",
            "operator_boundaries": [
                "router_mm",
                "grouped_topk",
                "routed_w13",
                "silu_and_mul",
                "routed_w2",
                "weighted_reduce",
                "shared_w13",
                "shared_silu_and_mul",
                "shared_w2",
                "final_add",
            ],
            "benchmark": {
                "cache_state": "cold_l2",
                "l2_flush_mib": args.l2_flush_mib,
                "order": "randomized_forward_reverse_williams",
                "order_seed": args.order_seed,
                "observations_per_variant": args.repeats,
                "timed_replays_per_observation": 1,
                "compile_capture_and_warmup_excluded": True,
                "separate_graph_flush_scope": (
                    "once before the complete MoE graph; not between kernels"
                ),
                "mutable_state_restored": "not applicable; inputs are read-only",
            },
            "device": torch.cuda.get_device_name(),
            "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
            "shape": vars(shape),
            "routing": routing_histogram(reference["topk_ids"], shape.num_experts),
            "vllm": {
                "revision": _git_revision(VLLM_ROOT),
                "requested_backend": args.moe_backend,
                "selected_backend": (
                    selected_backend.value if selected_backend is not None else None
                ),
                "control": "official DeepseekV2MoE production module",
                "same_tensor_storage": same_storage,
            },
            "helion": {
                "revision": _git_revision(REPO_ROOT),
                "module": helion.__file__,
                "configs": helion_build["configs"],
                "resources": helion_build["resources"],
                "lowerings_dir": str(lowerings_dir),
                "lowering_summaries": {
                    name: lowered_triton_summary(source)
                    for name, source in helion_build["lowerings"].items()
                },
                "task_geometry": _task_geometry(
                    helion_build["configs"],
                    shape,
                    torch.cuda.get_device_properties(0).multi_processor_count,
                ),
            },
            "correctness": correctness,
            "timings": timings,
            "helion_stage_timings": stage_timings,
            "telemetry": {
                "before": telemetry_before,
                "after": telemetry_after,
            },
        }
        output_path = Path(
            args.output
            or Path(__file__).with_name(
                f"deepseek_v3_moe_standalone_{args.moe_backend}_result.json"
            )
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("RESULT_JSON", json.dumps(result, sort_keys=True), flush=True)
        return result
    finally:
        if torch.distributed.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--stage-repeats", type=int, default=0)
    parser.add_argument("--l2-flush-mib", type=int, choices=(256,), default=256)
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--tune", nargs="*", default=[])
    parser.add_argument(
        "--moe-backend",
        choices=("auto", "triton", "flashinfer_cutlass", "flashinfer_trtllm"),
        default="auto",
    )
    parser.add_argument(
        "--config-path",
        default=str(Path(__file__).with_name("deepseek_v3_moe_b200_configs.json")),
    )
    parser.add_argument(
        "--lowerings-dir",
        default=str(Path(__file__).with_name("deepseek_v3_moe_lowerings")),
    )
    parser.add_argument("--output")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
