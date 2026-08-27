"""Production-shape tensors and references for DeepSeek-V3's MoE block."""

from __future__ import annotations

from dataclasses import dataclass

import torch

MODEL_ID = "deepseek-ai/DeepSeek-V3"


@dataclass(frozen=True)
class DeepseekV3MoEShape:
    batch: int = 1
    hidden: int = 7168
    intermediate: int = 2048
    num_experts: int = 256
    top_k: int = 8
    num_groups: int = 8
    topk_groups: int = 4
    routed_scale: float = 2.5

    @property
    def experts_per_group(self) -> int:
        return self.num_experts // self.num_groups


def grouped_topk_reference(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    shape: DeepseekV3MoEShape,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match vLLM's DeepSeek ``noaux_tc`` sigmoid grouped routing."""
    scores = torch.sigmoid(logits)
    biased = scores + correction_bias[None, :]
    grouped = biased.view(shape.batch, shape.num_groups, shape.experts_per_group)
    group_scores = torch.topk(grouped, 2, dim=-1).values.sum(dim=-1)
    selected_groups = torch.topk(
        group_scores, shape.topk_groups, dim=-1, sorted=True
    ).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    expert_mask = (
        group_mask[:, :, None]
        .expand(shape.batch, shape.num_groups, shape.experts_per_group)
        .reshape(shape.batch, shape.num_experts)
    )
    selected_scores = biased.masked_fill(~expert_mask, float("-inf"))
    topk_ids = torch.topk(selected_scores, shape.top_k, dim=-1, sorted=True).indices
    topk_weights = scores.gather(1, topk_ids)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * shape.routed_scale
    return topk_weights.float(), topk_ids.to(torch.int32)


def _scaled_random(shape: tuple[int, ...], scale: float) -> torch.Tensor:
    return (torch.randn(shape, device="cuda", dtype=torch.float32) * scale).to(
        torch.bfloat16
    )


def allocate_moe(
    shape: DeepseekV3MoEShape,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Allocate the exact TP=1 BF16 DeepSeek-V3 MoE tensor geometry.

    The full 256-expert tensors are materialized so both Helion and vLLM see
    production strides and weight traffic.  Routing is made deterministic and
    selects experts 0..7; only those slices need nonzero random weights.
    """
    if shape.batch != 1:
        raise ValueError("the initial production decode probe requires batch=1")
    torch.manual_seed(seed)
    hidden_states = _scaled_random((shape.batch, shape.hidden), 0.25)
    router_weight = _scaled_random((shape.num_experts, shape.hidden), 0.002)
    correction_bias = torch.full(
        (shape.num_experts,), -2.0, device="cuda", dtype=torch.float32
    )
    correction_bias[: shape.top_k] = torch.linspace(
        2.0,
        1.65,
        shape.top_k,
        device="cuda",
        dtype=torch.float32,
    )

    # Zero initialization avoids NaNs in inactive experts while keeping the
    # allocation and vLLM weight layout fully production-sized.
    expert_w13 = torch.zeros(
        (shape.num_experts, 2 * shape.intermediate, shape.hidden),
        device="cuda",
        dtype=torch.bfloat16,
    )
    expert_w2 = torch.zeros(
        (shape.num_experts, shape.hidden, shape.intermediate),
        device="cuda",
        dtype=torch.bfloat16,
    )
    for expert in range(shape.top_k):
        expert_w13[expert].copy_(
            _scaled_random((2 * shape.intermediate, shape.hidden), shape.hidden**-0.5)
        )
        expert_w2[expert].copy_(
            _scaled_random((shape.hidden, shape.intermediate), shape.intermediate**-0.5)
        )

    shared_w13 = _scaled_random(
        (2 * shape.intermediate, shape.hidden), shape.hidden**-0.5
    )
    shared_w2 = _scaled_random(
        (shape.hidden, shape.intermediate), shape.intermediate**-0.5
    )
    return {
        "hidden_states": hidden_states.contiguous(),
        "router_weight": router_weight.contiguous(),
        "correction_bias": correction_bias.contiguous(),
        "expert_w13": expert_w13,
        "expert_w2": expert_w2,
        "shared_w13": shared_w13.contiguous(),
        "shared_w2": shared_w2.contiguous(),
    }


@torch.inference_mode()
def moe_reference(
    tensors: dict[str, torch.Tensor],
    shape: DeepseekV3MoEShape,
) -> dict[str, torch.Tensor]:
    hidden = tensors["hidden_states"]
    logits = torch.mm(
        hidden,
        tensors["router_weight"].T,
        out_dtype=torch.float32,
    )
    topk_weights, topk_ids = grouped_topk_reference(
        logits, tensors["correction_bias"], shape
    )

    gate_up_rows = []
    activation_rows = []
    expert_output_rows = []
    for expert_id in topk_ids[0].long().tolist():
        gate_up = torch.nn.functional.linear(hidden, tensors["expert_w13"][expert_id])
        gate, up = gate_up.chunk(2, dim=-1)
        activation = (gate.float() * torch.sigmoid(gate.float()) * up.float()).to(
            torch.bfloat16
        )
        expert_output = torch.nn.functional.linear(
            activation, tensors["expert_w2"][expert_id]
        )
        gate_up_rows.append(gate_up.squeeze(0))
        activation_rows.append(activation.squeeze(0))
        expert_output_rows.append(expert_output.squeeze(0))

    expert_gate_up = torch.stack(gate_up_rows)
    expert_activation = torch.stack(activation_rows)
    expert_outputs = torch.stack(expert_output_rows)
    routed_output = (
        (expert_outputs.float() * topk_weights[0, :, None])
        .sum(dim=0, keepdim=True)
        .to(torch.bfloat16)
    )

    shared_gate_up = torch.nn.functional.linear(hidden, tensors["shared_w13"])
    shared_gate, shared_up = shared_gate_up.chunk(2, dim=-1)
    shared_activation = (
        shared_gate.float() * torch.sigmoid(shared_gate.float()) * shared_up.float()
    ).to(torch.bfloat16)
    shared_output = torch.nn.functional.linear(shared_activation, tensors["shared_w2"])
    output = (routed_output + shared_output).to(torch.bfloat16)
    return {
        "router_logits": logits,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "expert_gate_up": expert_gate_up,
        "expert_activation": expert_activation,
        "expert_outputs": expert_outputs,
        "routed_output": routed_output,
        "shared_gate_up": shared_gate_up,
        "shared_activation": shared_activation,
        "shared_output": shared_output,
        "output": output,
    }


def routing_histogram(
    topk_ids: torch.Tensor, num_experts: int
) -> dict[str, int | list[int]]:
    counts = torch.bincount(topk_ids.flatten().long(), minlength=num_experts)
    return {
        "selected_ids": topk_ids.flatten().long().tolist(),
        "distinct_experts": int(torch.count_nonzero(counts).item()),
        "max_assignments_per_expert": int(counts.max().item()),
    }
