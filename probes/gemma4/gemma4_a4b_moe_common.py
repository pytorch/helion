"""Shared data and references for the standalone Gemma 4 26B-A4B MoE benchmark.

This module extracts the MoE sub-layer of ``google/gemma-4-26B-A4B`` from the
full decode layer in ``gemma4_26b_a4b_common.py``.  The extracted region starts
at the post-attention residual and ends at the normalized MoE branch, i.e. the
nine ops listed in ``MOE_STAGES`` below.  Everything else in the decoder layer
(attention, the dense MLP branch, the terminal branch add / post-FF norm /
layer scale) is deliberately outside the extraction.

Unlike the full-layer benchmark this module is parameterized by batch size.  At
``batch == 1`` it reproduces the decode geometry of the production layer bit for
bit; larger batches expose the ragged per-expert token distribution that only
appears once more than one token routes at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from probes.gemma4.gemma4_26b_a4b_common import Gemma4A4BShape
from probes.gemma4.gemma4_26b_a4b_common import allocate_layer
from probes.gemma4.gemma4_26b_a4b_common import layer_reference
from probes.gemma4.gemma4_26b_a4b_common import rms_norm_reference
from probes.gemma4.gemma4_26b_a4b_common import routing_reference

#: The ops the extraction covers, in dependency order.  ``matched`` names the
#: kernel boundary used by the unfused reference path; ``optimized`` names the
#: boundary the fused path uses (``fused into <stage>`` when it disappears).
MOE_STAGES = (
    ("expert_pre_norm", "rms_norm(residual, pre_ff_norm_weight_2)", "rms_norm"),
    ("router_norm_scale", "rms_norm(residual) * hidden**-0.5 * router_scale", "fused"),
    ("router_mm", "fp32 logits = router_hidden @ router_weight.T", "fused"),
    ("route_topk", "top-8 select, renormalized softmax, per_expert_scale", "router"),
    ("expert_gate_up", "gathered per-expert fused gate/up GEMV", "expert_geglu"),
    ("expert_geglu", "gelu_tanh(gate) * up", "fused"),
    ("expert_down", "gathered per-expert down GEMV", "expert_down_reduce"),
    ("expert_reduce", "routing-weight weighted sum over the top-8 experts", "fused"),
    ("moe_post_norm", "rms_norm(moe_down, post_ff_norm_weight_2)", "rms_norm"),
)


@dataclass(frozen=True)
class Gemma4A4BMoEShape:
    """The MoE-relevant subset of the A4B geometry, plus a batch dimension."""

    batch: int = 1
    hidden: int = 2816
    moe_intermediate: int = 704
    num_experts: int = 128
    top_k: int = 8
    eps: float = 1e-6

    @property
    def assignments(self) -> int:
        return self.batch * self.top_k

    def expert_weight_bytes(self) -> int:
        """Bytes of expert weight for one expert (gate/up plus down), BF16."""
        gate_up = 2 * self.moe_intermediate * self.hidden * 2
        down = self.hidden * self.moe_intermediate * 2
        return gate_up + down


def allocate_moe(
    shape: Gemma4A4BMoEShape,
    seed: int,
    *,
    route_skew: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Allocate the MoE tensors, using the real layer-0 residual for row 0.

    The weights come from ``allocate_layer`` with the same seed, so a
    ``batch == 1`` run of this benchmark consumes exactly the tensors the full
    A4B layer benchmark consumes.  Row 0 of ``residual`` is the true
    post-attention residual of that layer; any extra rows are drawn to the same
    RMS so the router sees realistically scaled inputs.

    ``route_skew`` makes some experts popular.  A Zipf profile over a random
    permutation of the experts is projected onto the mean input direction and
    added to the router weight, so hot experts win top-k for many tokens at
    once.  Zero leaves the allocation's own near-uniform routing alone.  Note
    that merely scaling the logits would not work: top-k is invariant to a
    positive scale, so only a per-expert bias changes which experts are chosen.
    """
    layer_shape = Gemma4A4BShape()
    geometry = layer_shape.layer_geometry(0)
    tensors = allocate_layer(layer_shape, geometry, seed)
    reference = layer_reference(tensors, layer_shape, geometry)

    device = "cuda"
    torch.manual_seed(seed + 977)
    row0 = reference["residual"].to(torch.bfloat16)
    if shape.batch == 1:
        residual = row0
    else:
        row_rms = row0.float().square().mean().sqrt()
        extra = torch.randn(
            (shape.batch - 1, shape.hidden), device=device, dtype=torch.float32
        )
        extra = extra * row_rms
        residual = torch.cat((row0, extra.to(torch.bfloat16)), dim=0)

    router_weight = tensors["router_weight"]
    if route_skew:
        mean_direction = residual.float().mean(dim=0)
        mean_direction = mean_direction / torch.linalg.vector_norm(mean_direction)
        ranks = torch.randperm(shape.num_experts, device=device)
        popularity = 1.0 / (ranks.float() + 1.0)
        popularity = popularity / popularity.max()
        bias = route_skew * popularity[:, None] * mean_direction[None, :]
        router_weight = (router_weight.float() + bias).to(torch.bfloat16)

    return {
        "residual": residual.contiguous(),
        "pre_ff_norm_weight_2": tensors["pre_ff_norm_weight_2"],
        "router_scale": tensors["router_scale"],
        "router_weight": router_weight,
        "per_expert_scale": tensors["per_expert_scale"],
        "expert_gate_up_weight": tensors["expert_gate_up_weight"],
        "expert_down_weight": tensors["expert_down_weight"],
        "post_ff_norm_weight_2": tensors["post_ff_norm_weight_2"],
        "root_size": torch.tensor(
            shape.hidden**-0.5, device=device, dtype=torch.bfloat16
        ),
    }


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    coefficient = 0.7978845608028654
    return 0.5 * x * (1.0 + torch.tanh(coefficient * (x + 0.044715 * x * x * x)))


def moe_reference(
    tensors: dict[str, torch.Tensor],
    shape: Gemma4A4BMoEShape,
) -> dict[str, torch.Tensor]:
    """Torch reference for the nine extracted ops, batch-generalized.

    The expert GEMVs are evaluated one token's top-k group at a time, which at
    ``batch == 1`` is exactly the ``torch.bmm`` form the full-layer reference
    uses, and at larger batches keeps the gathered weight slice bounded.
    """
    residual = tensors["residual"]
    expert_input = rms_norm_reference(
        residual, tensors["pre_ff_norm_weight_2"], shape.eps
    )
    router_hidden = rms_norm_reference(residual, None, shape.eps)
    router_hidden = router_hidden * tensors["root_size"]
    router_hidden = router_hidden * tensors["router_scale"]
    router_logits = torch.mm(
        router_hidden,
        tensors["router_weight"].T,
        out_dtype=torch.float32,
    )
    topk_weights, topk_ids = routing_reference(
        router_logits, shape.top_k, tensors["per_expert_scale"]
    )

    gate_up_weight = tensors["expert_gate_up_weight"]
    down_weight = tensors["expert_down_weight"]
    intermediate = shape.moe_intermediate
    gate_up = torch.empty(
        (shape.batch, shape.top_k, 2 * intermediate),
        device=residual.device,
        dtype=torch.bfloat16,
    )
    activation = torch.empty(
        (shape.batch, shape.top_k, intermediate),
        device=residual.device,
        dtype=torch.bfloat16,
    )
    outputs = torch.empty(
        (shape.batch, shape.top_k, shape.hidden),
        device=residual.device,
        dtype=torch.bfloat16,
    )
    for token in range(shape.batch):
        expert_ids = topk_ids[token].long()
        token_gate_up = torch.bmm(
            gate_up_weight[expert_ids],
            expert_input[token].expand(shape.top_k, -1).unsqueeze(-1),
        ).squeeze(-1)
        gate, up = token_gate_up.chunk(2, dim=-1)
        token_activation = _gelu_tanh(gate.float()).to(up.dtype) * up
        gate_up[token] = token_gate_up
        activation[token] = token_activation
        outputs[token] = torch.bmm(
            down_weight[expert_ids], token_activation.unsqueeze(-1)
        ).squeeze(-1)

    moe_down = (
        (outputs.float() * topk_weights[:, :, None]).sum(dim=1).to(torch.bfloat16)
    )
    moe_branch = rms_norm_reference(
        moe_down, tensors["post_ff_norm_weight_2"], shape.eps
    )
    return {
        "expert_input": expert_input,
        "router_hidden": router_hidden,
        "router_logits": router_logits,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "expert_gate_up": gate_up,
        "expert_activation": activation,
        "expert_outputs": outputs,
        "moe_down": moe_down,
        "moe_branch": moe_branch,
    }


def routing_histogram(topk_ids: torch.Tensor, num_experts: int) -> dict[str, float]:
    """Summarize how ragged one routing decision is."""
    counts = torch.bincount(topk_ids.reshape(-1).long(), minlength=num_experts)
    nonzero = counts[counts > 0]
    total = int(counts.sum().item())
    return {
        "assignments": total,
        "distinct_experts": int(nonzero.numel()),
        "max_tokens_per_expert": int(counts.max().item()),
        "mean_tokens_per_active_expert": float(nonzero.float().mean().item()),
        "imbalance": float(counts.max().item())
        / max(1e-9, float(nonzero.float().mean().item())),
    }


def expert_weight_traffic_bytes(
    topk_ids: torch.Tensor,
    shape: Gemma4A4BMoEShape,
) -> dict[str, float]:
    """DRAM floor for the expert weights under the two possible formulations."""
    distinct = int(torch.unique(topk_ids.reshape(-1)).numel())
    per_expert = shape.expert_weight_bytes()
    return {
        "grouped_bytes": distinct * per_expert,
        "gathered_bytes": shape.assignments * per_expert,
        "grouped_us_at_8tbs": distinct * per_expert / 8e12 * 1e6,
        "gathered_us_at_8tbs": shape.assignments * per_expert / 8e12 * 1e6,
    }


def align_experts_reference(
    topk_ids: torch.Tensor,
    num_experts: int,
    tile_tokens: int,
    tiles_per_expert: int,
) -> dict[str, torch.Tensor | int]:
    """Host-side model of ``moe_expert_tiles`` plus ``moe_assignment_order``.

    The layout is expert-major with a fixed stride, so expert ``e`` owns slots
    ``[e * tiles_per_expert * tile_tokens, ...)`` and tile ``j`` of that expert
    is the group id ``e * tiles_per_expert + j``.  ``active_tiles`` packs the
    occupied group ids so a consumer can walk exactly the non-empty work.
    """
    flat = topk_ids.reshape(-1)
    device = flat.device
    counts = torch.bincount(flat.long(), minlength=num_experts).to(torch.int32)
    tiles = (counts + tile_tokens - 1) // tile_tokens
    tile_end = torch.cumsum(tiles, dim=0).to(torch.int32)
    tile_start = tile_end - tiles
    num_active = int(tile_end[-1].item())
    stride = tiles_per_expert * tile_tokens
    active = torch.full((num_active,), -1, device=device, dtype=torch.int32)
    order = torch.full((num_experts * stride,), -1, device=device, dtype=torch.int32)
    for expert in range(num_experts):
        count = int(counts[expert].item())
        if count == 0:
            continue
        span = int(tiles[expert].item())
        start = int(tile_start[expert].item())
        active[start : start + span] = (
            torch.arange(span, device=device, dtype=torch.int32)
            + expert * tiles_per_expert
        )
        rows = (flat == expert).nonzero().squeeze(-1).to(torch.int32)
        order[expert * stride : expert * stride + count] = rows
    return {
        "expert_counts": counts,
        "tile_end": tile_end,
        "active_tiles": active,
        "order": order,
        "num_active_tiles": num_active,
    }


def tiles_per_expert(shape: Gemma4A4BMoEShape, tile_tokens: int) -> int:
    """A token routes to an expert at most once, so ``batch`` bounds its rows."""
    return math.ceil(shape.batch / tile_tokens)


def max_aligned_tiles(shape: Gemma4A4BMoEShape, tile_tokens: int) -> int:
    """Static upper bound on the number of occupied tiles.

    ``sum_e ceil(n_e / TM)`` is bounded three ways at once: every expert can
    occupy all of its tiles, every expert can waste at most one partial tile on
    top of a perfect packing, and no expert's tile count exceeds its token
    count.  The third bound is what keeps small batches cheap.
    """
    experts = shape.num_experts
    return min(
        experts * tiles_per_expert(shape, tile_tokens),
        math.ceil(shape.assignments / tile_tokens) + experts,
        shape.assignments,
    )
