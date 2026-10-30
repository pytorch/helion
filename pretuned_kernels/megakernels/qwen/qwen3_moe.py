# ruff: noqa: ANN001, ANN002, ANN201, ANN202
# pyrefly: ignore-errors
"""Qwen3-30B-A3B batch-one MoE block megakernel, pretuned for NVIDIA B200.

The single Helion function's four top-level tile loops implement the
post-attention RMSNorm and router projection; the experts' gate/up projection
with SiLU(gate) * up, where every expert program recomputes the routing (softmax,
top-8, renormalize) from the 128 router logits instead of waiting on a separate
routing stage; the experts' down projection scaled by the routing weights; and
the weighted expert sum plus residual.  The benchmark uses the production BF16
geometry and checks it against a PyTorch reference and vLLM's
Qwen3MoeSparseMoeBlock (fused MoE with its default B200 backend selection).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


BATCH = 1
HIDDEN = 2048
INTERMEDIATE = 768
NUM_EXPERTS = 128
TOP_K = 8
EPS = 1e-6


@helion.aot_kernel(
    static_shapes=False,
    backend="triton",
    triton_do_not_specialize=False,
)
def qwen3_moe(
    residual,
    norm_weight,
    router_weight,
    expert_gate_up_weight,
    expert_down_weight,
    top_k,
    eps,
):
    hl.specialize(residual.size(0))
    hl.specialize(residual.size(1))
    hl.specialize(norm_weight.size(0))
    hl.specialize(router_weight.size(0))
    hl.specialize(router_weight.size(1))
    hl.specialize(expert_gate_up_weight.size(0))
    hl.specialize(expert_gate_up_weight.size(1))
    hl.specialize(expert_gate_up_weight.size(2))
    hl.specialize(expert_down_weight.size(0))
    hl.specialize(expert_down_weight.size(1))
    hl.specialize(expert_down_weight.size(2))
    hl.specialize(residual.stride(0))
    hl.specialize(residual.stride(1))
    hl.specialize(norm_weight.stride(0))
    hl.specialize(router_weight.stride(0))
    hl.specialize(router_weight.stride(1))
    hl.specialize(expert_gate_up_weight.stride(0))
    hl.specialize(expert_gate_up_weight.stride(1))
    hl.specialize(expert_gate_up_weight.stride(2))
    hl.specialize(expert_down_weight.stride(0))
    hl.specialize(expert_down_weight.stride(1))
    hl.specialize(expert_down_weight.stride(2))
    top_k = hl.specialize(top_k)

    m, hidden = residual.size()
    num_experts, twice_intermediate, _ = expert_gate_up_weight.size()
    intermediate = twice_intermediate // 2
    flat_gate_up = expert_gate_up_weight.view(num_experts * twice_intermediate, hidden)
    flat_down = expert_down_weight.view(num_experts * hidden, intermediate)
    router_logits = torch.empty(
        (m, num_experts), dtype=torch.float32, device=residual.device
    )
    topk_weights = torch.empty((m, top_k), dtype=torch.float32, device=residual.device)
    topk_ids = torch.empty((m, top_k), dtype=torch.int32, device=residual.device)
    activation = torch.empty(
        (m, top_k, intermediate), dtype=residual.dtype, device=residual.device
    )
    selected_ids = torch.empty((m, top_k), dtype=torch.int32, device=residual.device)
    selected_weights = torch.empty(
        (m, top_k), dtype=torch.float32, device=residual.device
    )
    expert_outputs = torch.empty(
        (m, top_k, hidden), dtype=torch.float32, device=residual.device
    )
    output = torch.empty_like(residual)
    block_k1 = hl.register_block_size(hidden)
    block_k2 = hl.register_block_size(intermediate)

    # Qwen3MoeDecoderLayer.post_attention_layernorm + Qwen3MoeSparseMoeBlock.gate.
    for tile_m, tile_e in hl.tile([m, num_experts], block_size=[1, None]):
        token = tile_m.begin
        row = residual[token, :].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(row * row, dim=-1) + eps)
        normed = (row * inv_rms).to(residual.dtype) * norm_weight[:]
        w = router_weight[tile_e, :].to(torch.float32)
        router_logits[token, tile_e] = torch.sum(
            w * normed.to(torch.float32)[None, :], dim=-1
        )
    # Experts' gate/up projection with the RMSNorm recomputed inline and SiLU(gate) * up.
    for tile_m, tile_slot, tile_i in hl.tile(
        [m, top_k, intermediate], block_size=[1, 1, None]
    ):
        token = tile_m.begin
        slot = tile_slot.begin
        # Routing (softmax -> top-k -> renormalize == top-k of the logits, softmax over
        # the k), recomputed by every expert program from the 128 logits.
        vals, ids = torch.topk(router_logits[token, :], top_k, dim=-1, largest=True)
        probs = torch.exp(vals - torch.amax(vals, dim=-1, keepdim=True))
        probs = probs / torch.sum(probs, dim=-1, keepdim=True)
        pick = hl.arange(top_k) == slot
        expert = torch.sum(torch.where(pick, ids, 0), dim=-1).to(torch.int32)
        routing_weight = torch.sum(torch.where(pick, probs, 0.0), dim=-1)
        gate_row = expert * twice_intermediate + tile_i.index
        up_row = gate_row + intermediate
        row = residual[token, :].to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(row * row, dim=-1) + eps)
        gate_acc2 = hl.zeros([tile_i, block_k1], dtype=torch.float32)
        up_acc2 = hl.zeros([tile_i, block_k1], dtype=torch.float32)
        for tile_k in hl.tile(hidden, block_size=block_k1):
            x = (residual[token, tile_k].to(torch.float32) * inv_rms).to(residual.dtype)
            x = (x * norm_weight[tile_k]).to(torch.float32)
            gate_acc2 = (
                gate_acc2
                + flat_gate_up[gate_row, tile_k].to(torch.float32) * x[None, :]
            )
            up_acc2 = (
                up_acc2 + flat_gate_up[up_row, tile_k].to(torch.float32) * x[None, :]
            )
        gate_acc = torch.sum(gate_acc2, dim=-1)
        up_acc = torch.sum(up_acc2, dim=-1)
        gate = gate_acc.to(residual.dtype).to(torch.float32)
        up = up_acc.to(residual.dtype).to(torch.float32)
        activation[token, slot, tile_i] = (gate * torch.sigmoid(gate) * up).to(
            activation.dtype
        )
        if tile_i.begin == 0:
            selected_ids[token, slot] = expert
            selected_weights[token, slot] = routing_weight
            topk_ids[token, slot] = expert
            topk_weights[token, slot] = routing_weight
    # Experts' down projection, scaled by the routing weight.
    for tile_m, tile_slot, tile_n in hl.tile(
        [m, top_k, hidden], block_size=[1, 1, None]
    ):
        token = tile_m.begin
        slot = tile_slot.begin
        expert = selected_ids[token, slot]
        rows = expert * hidden + tile_n.index
        acc2 = hl.zeros([tile_n, block_k2], dtype=torch.float32)
        for tile_k in hl.tile(intermediate, block_size=block_k2):
            a = activation[token, slot, tile_k].to(torch.float32)
            acc2 = acc2 + flat_down[rows, tile_k].to(torch.float32) * a[None, :]
        acc = torch.sum(acc2, dim=-1)
        expert_outputs[token, slot, tile_n] = (
            acc.to(residual.dtype).to(torch.float32) * selected_weights[token, slot]
        )
    # Weighted expert sum plus the residual.
    for tile_m, tile_n in hl.tile([m, hidden], block_size=[1, None]):
        vals = expert_outputs[tile_m, :, tile_n]
        output[tile_m, tile_n] = (
            torch.sum(vals, dim=1) + residual[tile_m, tile_n].to(torch.float32)
        ).to(output.dtype)
    return output, router_logits, topk_weights, topk_ids, activation, expert_outputs


def use_cudagraph() -> bool:
    """The timed closures replay pre-captured CUDA graphs."""
    return True


def has_vllm() -> bool:
    try:
        from vllm.model_executor.models.qwen3_moe import (  # noqa: F401
            Qwen3MoeSparseMoeBlock,
        )
    except ImportError:
        return False
    return True


def _require_sm100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("qwen3_moe is pretuned only for NVIDIA SM100")


def _make_inputs(seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(
            *shape, generator=generator, device="cuda", dtype=torch.bfloat16
        )

    tensors = {
        "norm_weight": torch.rand(HIDDEN, generator=generator, device="cuda").to(
            torch.bfloat16
        )
        * 0.4
        + 0.8,
        "router_weight": randn(NUM_EXPERTS, HIDDEN) * 0.05,
        "expert_gate_up_weight": randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN)
        * HIDDEN**-0.5,
        "expert_down_weight": randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE)
        * INTERMEDIATE**-0.5,
    }
    tensors["residual"] = randn(BATCH, HIDDEN)
    return tensors


def _make_routing_cases() -> tuple[tuple[str, torch.Tensor], ...]:
    """Distinct random tokens; each token's top-8 experts follow from its values."""
    generator = torch.Generator(device="cuda").manual_seed(20261026 + BATCH)
    return tuple(
        (
            f"route_{index}",
            torch.randn(
                (BATCH, HIDDEN),
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            ),
        )
        for index in range(3)
    )


def _kernel_args(tensors: dict[str, torch.Tensor]) -> tuple[object, ...]:
    return (
        tensors["residual"],
        tensors["norm_weight"],
        tensors["router_weight"],
        tensors["expert_gate_up_weight"],
        tensors["expert_down_weight"],
        TOP_K,
        EPS,
    )


def reference(
    tensors: dict[str, torch.Tensor], bf16_logits: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference per token, rounded to BF16 where the kernel stores BF16.

    bf16_logits rounds the router logits to BF16 first, as vLLM's and SGLang's
    BF16 router linear does (this can flip near-tied top-k choices)."""
    outputs, ids = [], []
    for token in range(tensors["residual"].size(0)):
        token_tensors = dict(tensors, residual=tensors["residual"][token : token + 1])
        output, token_ids = _reference_token(token_tensors, bf16_logits)
        outputs.append(output)
        ids.append(token_ids)
    return torch.cat(outputs), torch.cat(ids)


def _reference_token(
    tensors: dict[str, torch.Tensor], bf16_logits: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    x = tensors["residual"].float()
    normed = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)).to(
        torch.bfloat16
    ) * tensors["norm_weight"]
    logits = normed.float() @ tensors["router_weight"].float().T
    if bf16_logits:
        logits = logits.to(torch.bfloat16).float()
    weights, ids = torch.topk(torch.softmax(logits, -1), TOP_K, -1)
    weights = weights / weights.sum(-1, keepdim=True)
    out = x.clone()
    for slot in range(TOP_K):
        expert = int(ids[0, slot])
        gate_up = (
            (normed.float() @ tensors["expert_gate_up_weight"][expert].float().T)
            .to(torch.bfloat16)
            .float()
        )
        act = (
            torch.nn.functional.silu(gate_up[:, :INTERMEDIATE])
            * gate_up[:, INTERMEDIATE:]
        ).to(torch.bfloat16)
        down = (act.float() @ tensors["expert_down_weight"][expert].float().T).to(
            torch.bfloat16
        )
        out += weights[0, slot] * down.float()
    return out.to(torch.bfloat16), ids


def _initialize_vllm() -> object:
    from vllm.config import CacheConfig
    from vllm.config import CUDAGraphMode
    from vllm.config import VllmConfig
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment
    from vllm.distributed import initialize_model_parallel
    from vllm.utils.network_utils import get_open_port
    from vllm.v1.worker.workspace import init_workspace_manager

    device_index = torch.cuda.current_device()
    init_workspace_manager(torch.device("cuda", device_index))
    try:
        vllm_config = VllmConfig(cache_config=CacheConfig(block_size=16))
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        hf_config = SimpleNamespace(
            hidden_size=HIDDEN,
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOP_K,
            moe_intermediate_size=INTERMEDIATE,
            norm_topk_prob=True,
            hidden_act="silu",
            rms_norm_eps=EPS,
        )
        vllm_config.model_config = SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=hf_config,
            hf_text_config=hf_config,
            is_mm_prefix_lm=False,
            is_moe=True,
            is_diffusion=False,
            is_hybrid=False,
            is_attention_free=True,
            runner_type="generate",
            architectures=["Qwen3MoeForCausalLM"],
            max_model_len=8192,
            rswa_window=None,
            compute_hash=lambda: "qwen3-moe-pretuned",
        )
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=device_index,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        )
        with set_current_vllm_config(vllm_config):
            initialize_model_parallel(1, 1)
    except Exception:
        _destroy_vllm()
        raise
    return vllm_config


def _destroy_vllm() -> None:
    from vllm.distributed import destroy_distributed_environment
    from vllm.distributed import destroy_model_parallel
    from vllm.v1.worker.workspace import reset_workspace_manager

    destroy_model_parallel()
    destroy_distributed_environment()
    reset_workspace_manager()


def _make_compiled_block_class() -> type[torch.nn.Module]:
    from torch import nn
    from vllm.compilation.decorators import support_torch_compile
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock

    @support_torch_compile(dynamic_arg_dims={"residual": 0})
    class CompiledQwen3MoeBlock(nn.Module):
        def __init__(self, *, vllm_config, prefix: str = "") -> None:
            super().__init__()
            self.norm = RMSNorm(HIDDEN, eps=EPS)
            self.moe = Qwen3MoeSparseMoeBlock(vllm_config, prefix=f"{prefix}.mlp")

        def forward(self, residual: torch.Tensor) -> torch.Tensor:
            return self.moe(self.norm(residual))

    return CompiledQwen3MoeBlock


def _make_vllm_call(
    tensors: dict[str, torch.Tensor],
) -> tuple[Callable[..., Callable[[], torch.Tensor]], str, Callable[[], None]]:
    """vLLM's post-attention RMSNorm, Qwen3MoeSparseMoeBlock, and residual add.

    The norm and MoE block are compiled as in vLLM's model (``support_torch_compile``).
    Returns a factory binding one residual tensor to a launch."""
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context

    vllm_config = _initialize_vllm()
    try:
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with set_current_vllm_config(vllm_config):
                block = (
                    _make_compiled_block_class()(
                        vllm_config=vllm_config, prefix="model.layers.0"
                    )
                    .eval()
                    .cuda()
                )
        finally:
            torch.set_default_dtype(old_dtype)
        norm, moe = block.norm, block.moe
        experts = moe.experts.routed_experts
        with torch.no_grad():
            norm.weight.copy_(tensors["norm_weight"])
            moe.gate.weight.copy_(tensors["router_weight"])
            experts.w13_weight.copy_(tensors["expert_gate_up_weight"])
            experts.w2_weight.copy_(tensors["expert_down_weight"])
            experts.quant_method.process_weights_after_loading(experts)
        backend = getattr(
            getattr(experts.quant_method, "unquantized_backend", None),
            "value",
            type(experts.quant_method).__name__,
        )
    except Exception:
        _destroy_vllm()
        raise

    def make_launch(
        residual: torch.Tensor, add_residual: bool = True
    ) -> Callable[[], torch.Tensor]:
        # In vLLM's model the residual add runs in the next layer's fused
        # add + RMSNorm; add_residual=False leaves it out, as in a decode step.
        def launch() -> torch.Tensor:
            with set_forward_context(
                None, vllm_config=vllm_config, num_tokens=residual.size(0)
            ):
                output = block(residual)
            return residual + output if add_residual else output

        return launch

    return make_launch, backend, _destroy_vllm


def _check(
    output: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    name: str,
    bf16_logits: bool = False,
) -> float:
    expected, _ids = reference(tensors, bf16_logits)
    delta = (expected.float() - tensors["residual"].float()).norm()
    rel = float((output.float() - expected.float()).norm() / delta)
    if rel > 2e-2:
        raise AssertionError(f"{name}: relative L2 error of the MoE delta {rel:.3e}")
    return rel


def _check_routing(helion_ids: torch.Tensor, tensors: dict[str, torch.Tensor]) -> None:
    _expected, ids = reference(tensors)
    for token in range(ids.size(0)):
        if sorted(helion_ids[token].tolist()) != sorted(ids[token].tolist()):
            raise AssertionError("routing mismatch vs the reference")


@torch.inference_mode()
def correctness_check() -> None:
    """Check every routing case against the FP32 reference."""
    _require_sm100()
    tensors = _make_inputs()
    for label, residual in _make_routing_cases():
        tensors["residual"].copy_(residual)
        helion_outputs = qwen3_moe(*_kernel_args(tensors))
        torch.cuda.synchronize()
        _check_routing(helion_outputs[3], tensors)
        _check(helion_outputs[0], tensors, label)


@torch.inference_mode()
def main(verbose: bool = True) -> dict:
    """Benchmark the megakernel and vLLM with cold-L2 CUDA-graph replay."""
    _require_sm100()
    if not has_vllm():
        raise RuntimeError("vLLM is required for the Qwen3 MoE comparison")

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from _bench import capture_cuda_graph
    from _bench import run_sweep

    def iter_benchmark_cases():
        tensors = _make_inputs()
        make_vllm_launch, backend, close_vllm = _make_vllm_call(tensors)
        try:
            vllm_call = make_vllm_launch(tensors["residual"], add_residual=False)
            cases = _make_routing_cases()
            for label, residual in cases:
                tensors["residual"].copy_(residual)
                helion_outputs = qwen3_moe(*_kernel_args(tensors))
                vllm_output = vllm_call()
                torch.cuda.synchronize()
                _check_routing(helion_outputs[3], tensors)
                helion_rel = _check(helion_outputs[0], tensors, "helion")
                vllm_rel = _check(
                    tensors["residual"] + vllm_output,
                    tensors,
                    "vllm",
                    bf16_logits=True,
                )
                if verbose:
                    print(f"{label}: relL2 helion {helion_rel:.2e} vllm {vllm_rel:.2e}")
            tensors["residual"].copy_(cases[0][1])
            helion_graph, _ = capture_cuda_graph(
                lambda: qwen3_moe(*_kernel_args(tensors))
            )
            vllm_graph, _ = capture_cuda_graph(vllm_call)
            yield backend, helion_graph, vllm_graph
        finally:
            close_vllm()

    def make_calls(case: tuple) -> tuple:
        backend, helion_graph, vllm_graph = case
        return (
            helion_graph.replay,
            [(f"vllm_auto ({backend})", vllm_graph.replay)],
            f"{BATCH:>5d}  {HIDDEN:>6d}  {NUM_EXPERTS:>7d}  {TOP_K:>4d}",
        )

    cases = iter_benchmark_cases()
    try:
        return run_sweep(
            cases,
            make_calls,
            use_cudagraph=False,
            pre_captured_cudagraph=True,
            thermal_warmup_ms=10_000,
            verbose=verbose,
            shape_header=f"{'batch':>5s}  {'hidden':>6s}  {'experts':>7s}  {'topk':>4s}",
        )
    finally:
        cases.close()


if __name__ == "__main__":
    main()
