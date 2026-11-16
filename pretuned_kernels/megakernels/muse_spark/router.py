"""Muse Spark spec-decode verify router for TP4 as one persistent Helion megakernel.

The expert-input side of the attention boundary for a T-token verify window
(T = 8 or 16): the pooler, router top-8 routing and the pre-expert NVFP4 quant.
The input is the pre-FFN-normed hidden state (replicated). The outputs fill the
MoE megakernel's input slot (``ExpertInputs``) and return the top-8 ids.

Roots (rank r):

    R1  router split-K partials                                         [S_r]
    P1  pooler split-K partials                                  [N/W/128, S_p]
    P2  pooler partial sum -> BF16 pooled quarter (symm)              [N/W/32]
    R2  router partial sum -> logit K-shard sum (symm)                [tokens]
    R3  rank-order logit sum and top-8 -> packed routes               [tokens]
    Q   every quarter -> RMSNorm and NVFP4                            [tokens]

Calls and graph replays must occur in identical collective order across ranks
on one ordered CUDA stream. ``main()`` compares against sglang's B200 default
(replicated pooler and router GEMMs, ``norm_fp4_quant`` and the Muse routing
kernel) and needs sglang (``muse-spark`` branch); ``--check`` runs only the FP64
correctness check::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \
      python -m torch.distributed.run --standalone --nproc-per-node=4 \
      pretuned_kernels/megakernels/muse_spark/router.py [--check]
"""

from __future__ import annotations

import dataclasses
import functools
import math
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from pretuned_kernels.megakernels._distributed import benchmark
from pretuned_kernels.megakernels._distributed import initialize
from pretuned_kernels.megakernels.muse_spark._common import EXPERT_HIDDEN
from pretuned_kernels.megakernels.muse_spark._common import HIDDEN
from pretuned_kernels.megakernels.muse_spark._common import NUM_EXPERTS
from pretuned_kernels.megakernels.muse_spark._common import SIGNAL_PAD_BYTES
from pretuned_kernels.megakernels.muse_spark._common import TOPK
from pretuned_kernels.megakernels.muse_spark._common import VERIFY_TOKENS
from pretuned_kernels.megakernels.muse_spark._common import WORLD_SIZE
from pretuned_kernels.megakernels.muse_spark._common import ExpertInputs
from pretuned_kernels.megakernels.muse_spark._common import e2m1_values
from pretuned_kernels.megakernels.muse_spark._common import encode_block_scale
from pretuned_kernels.megakernels.muse_spark._common import exp2_approx_ftz_f32
from pretuned_kernels.megakernels.muse_spark._common import pack_e2m1x2
from pretuned_kernels.megakernels.muse_spark._common import round_e2m1
from pretuned_kernels.megakernels.muse_spark._common import rsqrt_approx_ftz_f32
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

# Muse Spark routing and pre-expert norm scalars (compile-time constants).
NORM_EPS = 1.0e-5
ROUTER_MULTIPLIER = 1.0 / math.sqrt(32.0)
ROUTE_SCALE = 1.0
ROUTE_EPS = 1.0e-15

SPLIT_P = 16
SPLIT_R = 64
P_ROWS = 128
C_BLOCK = 32
# 1 / (fp8_e4m3_max * 6), the per-token NVFP4 global scale.
GLOBAL_SCALE_INV = 1.0 / (448.0 * 6.0)
_LOG2_E = 1.4426950408889634


def _fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return hl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;",
        "=f,f,f,f",
        [a, b, c],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


def _div_full(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return hl.inline_asm_elementwise(
        "div.full.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


def _max(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.maximum(a, b)


def _columns(value: torch.Tensor, width: int) -> list[torch.Tensor]:
    """Split the last axis into ``width`` tensors in increasing column order."""

    if width == 1:
        return [value.reshape(value.shape[:-1])]
    even, odd = hl.split(value.reshape(*value.shape[:-1], width // 2, 2))
    out: list[torch.Tensor] = []
    for e, o in zip(_columns(even, width // 2), _columns(odd, width // 2), strict=True):
        out.extend((e, o))
    return out


def _chain_sq(cols: list[torch.Tensor]) -> torch.Tensor:
    """fmaf(v_i, v_i, acc) from acc = 0 in column order (one thread's chain)."""

    acc = cols[0] * cols[0]
    for col in cols[1:]:
        acc = _fma(col, col, acc)
    return acc


def _butterfly(value: torch.Tensor, width: int) -> torch.Tensor:
    """xor-shuffle reduction of the last axis, partner distance width/2 first.

    Each size-2 sum is one commutative add, so the result equals
    ``warp::reduce_sum<width>`` bit for bit.
    """

    lead = value.shape[:-1]
    while width > 1:
        value = value.reshape(*lead, 2, width // 2).sum(dim=-2)
        width //= 2
    return value.reshape(lead)


def _pick(values: torch.Tensor, index: int, size: int) -> torch.Tensor:
    """Scalar ``values[index]`` of a small vector (exact: one nonzero term)."""

    return torch.sum(torch.where(hl.arange(size) == index, values, 0.0), dim=-1)


def _route(
    logits: torch.Tensor,
    bias: torch.Tensor,
    experts: torch.Tensor,
    num_experts: int,
    router_multiplier: float,
    route_scale: float,
    route_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Biased sigmoid, stable top-8 (key desc, id asc), renormalized weights."""

    scaled_logits = logits * router_multiplier
    expo = exp2_approx_ftz_f32((0.0 - scaled_logits) * _LOG2_E)
    score = _div_full(torch.ones_like(expo), expo + 1.0)
    key = score + bias
    # One int64 max per round: order-preserving key bits (+0.0 folds -0) above
    # the complemented id, so ties go to the lower id.
    kbits = (key + 0.0).view(dtype=torch.int32)
    ordered = torch.where(kbits >= 0, kbits, kbits ^ 0x7FFFFFFF).to(torch.int64)
    packed_key = (ordered << 32) | (num_experts - 1 - experts).to(torch.int64)
    taken = (hl.zeros([num_experts], dtype=torch.int64) - 1) << 63
    slots8 = hl.arange(8)
    sel_ids = hl.zeros([8], dtype=torch.int32)
    for kk in hl.static_range(8):
        best = hl.reduce(_max, packed_key, dim=0)
        best_id = (num_experts - 1) - (best & 0xFFFF).to(torch.int32)
        packed_key = torch.where(packed_key == best, taken, packed_key)
        sel_ids = torch.where(slots8 == kk, best_id, sel_ids)
    sel_scores = torch.sum(
        torch.where(experts[None, :] == sel_ids[:, None], score[None, :], 0.0), dim=-1
    )
    score_sum = _pick(sel_scores, 0, 8) + _pick(sel_scores, 1, 8)
    for kk in hl.static_range(2, 8):
        score_sum = score_sum + _pick(sel_scores, kk, 8)
    zeros8 = hl.zeros([8], dtype=torch.float32)
    scale = _div_full(zeros8 + route_scale, zeros8 + (score_sum + route_eps))
    return sel_scores * scale, sel_ids


def _muse_spark_router_megakernel(
    ffn_in: torch.Tensor,  # [T, N] bf16, the pre-FFN-normed hidden state
    wp: torch.Tensor,  # [Ne / W / P_ROWS, S_p, P_ROWS, N / S_p] bf16, pooler rows
    wr: torch.Tensor,  # [S_r, N / W / S_r, E] fp32, this rank's router K rows
    q_pre_w: torch.Tensor,  # [Ne] bf16
    router_bias: torch.Tensor,  # [E] fp32
    part_p: torch.Tensor,  # [S_p, Ne / W, T] fp32
    part_r: torch.Tensor,  # [S_r, T, E] fp32
    pooled: torch.Tensor,  # [T, Ne / W] bf16 symm
    logit_q: torch.Tensor,  # [T, E] fp32 symm
    x_q: torch.Tensor,  # [T, Ne / 2] u8
    x_sf: torch.Tensor,  # [T, Ne / 16] u8 (E4M3 bits)
    x_ts: torch.Tensor,  # [T] fp32
    packed: torch.Tensor,  # [T, 8] int32
    topk_ids: torch.Tensor,  # [T, 8] int32
    group_name: hl.ProcessGroupName,
    rank: hl.constexpr,
    quant_eps: hl.constexpr,
    global_scale_inv: hl.constexpr,
    router_multiplier: hl.constexpr,
    route_scale: hl.constexpr,
    route_eps: hl.constexpr,
) -> torch.Tensor:
    hl.specialize(ffn_in.size())
    hl.specialize(ffn_in.stride())
    hl.specialize(wp.size())
    hl.specialize(wp.stride())
    hl.specialize(wr.size())
    hl.specialize(wr.stride())
    hl.specialize(q_pre_w.size())
    hl.specialize(q_pre_w.stride())
    hl.specialize(router_bias.size())
    hl.specialize(router_bias.stride())
    hl.specialize(part_p.size())
    hl.specialize(part_p.stride())
    hl.specialize(part_r.size())
    hl.specialize(part_r.stride())
    hl.specialize(pooled.size())
    hl.specialize(pooled.stride())
    hl.specialize(logit_q.size())
    hl.specialize(logit_q.stride())
    hl.specialize(x_q.size())
    hl.specialize(x_q.stride())
    hl.specialize(x_sf.size())
    hl.specialize(x_sf.stride())
    hl.specialize(x_ts.size())
    hl.specialize(x_ts.stride())
    hl.specialize(packed.size())
    hl.specialize(packed.stride())
    hl.specialize(topk_ids.size())
    hl.specialize(topk_ids.stride())

    tokens = ffn_in.size(0)
    hidden = ffn_in.size(1)
    s_p = wp.size(1)
    k_p = wp.size(3)
    s_r = wr.size(0)
    k_r = wr.size(1)
    num_experts = wr.size(2)
    topk = packed.size(1)
    n_local = wp.size(0) * wp.size(2)
    world = hidden // (s_r * k_r)
    quant_lanes = q_pre_w.size(0) // 16
    q_local = n_local // 16
    kp_blk = hl.register_block_size(k_p)
    kr_blk = hl.register_block_size(k_r)
    x = ffn_in.view(tokens, s_p, k_p)
    xr = ffn_in.view(tokens, world, s_r, k_r)
    remote_pooled = torch.ops.symm_mem.get_remote_tensors(pooled, group_name)
    remote_logit = torch.ops.symm_mem.get_remote_tensors(logit_q, group_name)

    hl.pdl_wait()
    # Trigger at entry: the MoE megakernel's CTAs land on SMs this grid frees
    # early, and its own wait still orders it after this grid.
    hl.pdl_launch_dependents()
    # R1. Stock: the router GEMM (gate, FP32-accurate like the served F.linear;
    # see dot_precision below); each rank takes one K-shard here.
    for r1_s in hl.tile(s_r, block_size=1):
        acc_r = hl.zeros([tokens, num_experts], dtype=torch.float32)
        for r1_k in hl.tile(k_r, block_size=kr_blk):
            acc_r = hl.dot(
                xr[:, rank, r1_s.begin, r1_k].to(torch.float32),
                hl.load(wr, [r1_s.begin, r1_k, slice(None)]),
                acc=acc_r,
            )
        part_r[r1_s.begin, :, :] = acc_r

    # P1. Stock: the pooler GEMM (pre_expert_proj).
    for p1_n, p1_s in hl.tile([n_local, s_p], block_size=[P_ROWS, 1]):
        acc = hl.zeros([p1_n, tokens], dtype=torch.float32)
        for p1_k in hl.tile(k_p, block_size=kp_blk):
            acc = hl.dot(
                hl.load(wp, [p1_n.id, p1_s.begin, slice(None), p1_k]),
                x[:, p1_s.begin, p1_k].permute(1, 0),
                acc=acc,
            )
        part_p[p1_s.begin, p1_n, :] = acc

    # P2. Stock: the pooler GEMM's split-K reduce.
    for p2_c in hl.tile(n_local, block_size=C_BLOCK):
        total = torch.sum(part_p[:, p2_c, :], dim=0)
        pooled[:, p2_c] = total.permute(1, 0).to(torch.bfloat16)

    # R2. Stock: the router GEMM's split-K reduce.
    for r2_t in hl.tile(tokens, block_size=1):
        logit_q[r2_t, :] = torch.sum(part_r[:, r2_t, :], dim=0)

    # R3. Stock: FusedMuseSparkRouting (top-8), after summing the K-shard logits.
    for r3_t in hl.tile(tokens, block_size=1):
        experts = hl.arange(num_experts)
        bias = router_bias[experts]
        logits = hl.zeros([num_experts], dtype=torch.float32)
        # Rank order on every rank, so the routes are replicated.
        for peer_logit in remote_logit:
            logits = logits + peer_logit[r3_t.begin, experts]
        weights, sel_ids = _route(
            logits,
            bias,
            experts,
            num_experts,
            router_multiplier,
            route_scale,
            route_eps,
        )
        bits = weights.to(torch.bfloat16).view(dtype=torch.int16).to(torch.int32)
        packed[r3_t.begin, hl.arange(topk)] = (sel_ids << 16) | (bits & 0xFFFF)
        topk_ids[r3_t.begin, hl.arange(topk)] = sel_ids

    # Q. Every rank's pooled quarter (the gather sglang's gemm_ag does on GB300;
    # B200 serving replicates the pooler instead), then fused_norm_fp4_quant.
    for q_t in hl.tile(tokens, block_size=1):
        lanes = hl.arange(quant_lanes)
        elem = lanes[:, None] * 16 + hl.arange(16)[None, :]
        norm_w = q_pre_w[elem].to(torch.float32)
        quarters = hl.zeros([world, q_local, 16], dtype=torch.float32)
        owner = hl.arange(world)[:, None, None]
        for peer, peer_pooled in enumerate(remote_pooled):
            row = peer_pooled[q_t.begin, hl.arange(n_local)].to(torch.float32)
            quarters = torch.where(owner == peer, row.reshape(1, q_local, 16), quarters)
        xv = quarters.reshape(quant_lanes, 16)
        # fused_norm_fp4_quant order: 16-column chains, then two butterflies.
        sum_lane = _chain_sq(_columns(xv, 16))
        warp_sums = _butterfly(sum_lane.reshape(quant_lanes // 32, 32), 32)
        sum_sq = _butterfly(warp_sums, quant_lanes // 32)
        rstd = rsqrt_approx_ftz_f32(sum_sq * (1.0 / q_pre_w.size(0)) + quant_eps)
        normed = (xv * rstd * norm_w).to(torch.bfloat16).to(torch.float32)
        # Max via hl.reduce: an amax lowers to a Helion helper that would cost
        # these waiting roots their dry pass.
        block_amax = hl.reduce(_max, torch.abs(normed), dim=1)
        token_scale = hl.reduce(_max, block_amax, dim=0) * global_scale_inv
        fp8_scale, out_scale = encode_block_scale(
            token_scale + hl.zeros([quant_lanes], dtype=torch.float32), block_amax
        )
        low, high = hl.split((normed * out_scale[:, None]).reshape(quant_lanes, 8, 2))
        fp4_bytes = pack_e2m1x2(low, high).to(torch.uint8)
        x_q[q_t.begin, lanes[:, None] * 8 + hl.arange(8)[None, :]] = fp4_bytes
        x_sf[q_t.begin, lanes] = fp8_scale.to(torch.uint8)
        x_ts[q_t.begin + hl.arange(1)] = token_scale + hl.zeros(
            [1], dtype=torch.float32
        )
    return packed


muse_spark_router = helion.kernel(
    _muse_spark_router_megakernel,
    config=helion.Config(
        block_sizes=[64, 32],
        cross_loop_pipeline="dynamic",
        indexing="pointer",
        num_stages=3,
        num_warps=8,
        pid_type="persistent_blocked",
    ),
    static_shapes=True,
    backend="triton",
    # The FP32 router GEMM as 3xTF32, FP32-accurate on tensor cores (plain TF32
    # moves top-8 near-ties off the served F.linear; IEEE FMA costs ~3 us).
    dot_precision="tf32x3",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)


@dataclasses.dataclass(frozen=True)
class FullWeights:
    """One layer's checkpoint-order weights, replicated on every rank."""

    pre_expert: torch.Tensor  # [Ne, N] bf16, the pooler
    pre_expert_norm: torch.Tensor  # [Ne] bf16
    router: torch.Tensor  # [E, N] fp32
    router_bias: torch.Tensor  # [E] fp32


@dataclasses.dataclass(frozen=True)
class RouterWeights:
    """This rank's megakernel views (the router's K rows are a transposed copy)."""

    pooler: torch.Tensor  # [Ne / W / P_ROWS, S_p, P_ROWS, N / S_p] bf16
    router: torch.Tensor  # [N / W, E] fp32, this rank's router K rows
    pre_expert_norm: torch.Tensor  # [Ne] bf16
    router_bias: torch.Tensor  # [E] fp32


@dataclasses.dataclass(frozen=True)
class RouterBuffers:
    pooled: torch.Tensor  # [T, Ne / W] bf16 symm
    logit_q: torch.Tensor  # [T, E] fp32 symm
    part_p: torch.Tensor  # [S_p, Ne / W, T] fp32
    part_r: torch.Tensor  # [S_r, T, E] fp32


def make_full_weights(seed: int) -> FullWeights:
    def randn(shape: tuple[int, ...], offset: int, scale: float) -> torch.Tensor:
        g = torch.Generator(device="cuda").manual_seed(seed + offset)
        return torch.randn(shape, generator=g, device="cuda") * scale

    return FullWeights(
        pre_expert=randn((EXPERT_HIDDEN, HIDDEN), 4, HIDDEN**-0.5).bfloat16(),
        pre_expert_norm=(1.0 + randn((EXPERT_HIDDEN,), 5, 0.02)).bfloat16(),
        router=randn((NUM_EXPERTS, HIDDEN), 6, 0.01),
        router_bias=randn((NUM_EXPERTS,), 7, 0.01),
    )


def make_input(seed: int, tokens: int) -> torch.Tensor:
    """A replicated pre-FFN-normed hidden state [T, N] bf16."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(tokens, HIDDEN, generator=g, device="cuda").bfloat16()


def prepare_weights(full: FullWeights, rank: int) -> RouterWeights:
    """One-time per layer: this rank's pooler rows and router K rows."""
    n_local = EXPERT_HIDDEN // WORLD_SIZE
    k_local = HIDDEN // WORLD_SIZE
    return RouterWeights(
        # [N/W/P_ROWS, SPLIT_P, P_ROWS, K/SPLIT_P]: each P1 task's tile is dense.
        pooler=full.pre_expert[rank * n_local : (rank + 1) * n_local]
        .reshape(n_local // P_ROWS, P_ROWS, SPLIT_P, -1)
        .permute(0, 2, 1, 3)
        .contiguous(),
        router=full.router[:, rank * k_local : (rank + 1) * k_local]
        .t()
        .float()
        .contiguous(),
        pre_expert_norm=full.pre_expert_norm.contiguous(),
        router_bias=full.router_bias.contiguous(),
    )


# Symmetric buffers live for the process: torch finds a symmetric-memory handle by
# address, so a larger buffer at a freed smaller one's address gets its stale handle.
@functools.cache
def make_buffers(tokens: int, group: dist.ProcessGroup) -> RouterBuffers:
    device = torch.device("cuda", torch.cuda.current_device())
    n_local = EXPERT_HIDDEN // WORLD_SIZE
    pooled = symm_mem.empty(tokens, n_local, dtype=torch.bfloat16, device=device)
    logit_q = symm_mem.empty(tokens, NUM_EXPERTS, dtype=torch.float32, device=device)
    for buffer in (pooled, logit_q):
        symm_mem.rendezvous(buffer, group.group_name)
    return RouterBuffers(
        pooled=pooled,
        logit_q=logit_q,
        part_p=torch.empty(SPLIT_P, n_local, tokens, device=device),
        part_r=torch.empty(SPLIT_R, tokens, NUM_EXPERTS, device=device),
    )


def launch(
    ffn_in: torch.Tensor,
    weights: RouterWeights,
    buffers: RouterBuffers,
    expert_inputs: ExpertInputs,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Fill the first T rows of ``expert_inputs``; return the top-8 ids [T, 8]."""
    topk_ids = torch.empty(
        ffn_in.size(0), TOPK, dtype=torch.int32, device=ffn_in.device
    )
    muse_spark_router(
        ffn_in,
        weights.pooler,
        weights.router.view(SPLIT_R, -1, NUM_EXPERTS),
        weights.pre_expert_norm,
        weights.router_bias,
        buffers.part_p,
        buffers.part_r,
        buffers.pooled,
        buffers.logit_q,
        expert_inputs.fp4,
        expert_inputs.scales,
        expert_inputs.per_token_scale,
        expert_inputs.packed,
        topk_ids,
        group.group_name,
        group.rank(),
        NORM_EPS,
        GLOBAL_SCALE_INV,
        ROUTER_MULTIPLIER,
        ROUTE_SCALE,
        ROUTE_EPS,
    )
    return topk_ids


def _reference_quant(
    ffn_in: torch.Tensor, full: FullWeights
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashInfer's BF16 -> NVFP4 recipe on the FP64 pooler output after the
    pre-expert RMSNorm: (e2m1 values [T, Ne], E4M3 scale bytes, token scales)."""
    pooled = (ffn_in.double() @ full.pre_expert.double().T).bfloat16().double()
    rstd = torch.rsqrt(pooled.pow(2).mean(-1, keepdim=True) + NORM_EPS)
    normed = (pooled * rstd * full.pre_expert_norm.double()).bfloat16().float()
    blocks = normed.view(normed.size(0), -1, 16)
    block_amax = blocks.abs().amax(-1)
    token_scale = block_amax.amax(-1, keepdim=True) * GLOBAL_SCALE_INV
    encode = 1.0 / token_scale
    fp8 = (encode * (block_amax / 6.0)).clamp(max=448.0).to(torch.float8_e4m3fn)
    out_scale = 1.0 / (fp8.float() / encode)
    values = round_e2m1(blocks * out_scale[..., None]).flatten(-2).double()
    return values, fp8.view(torch.uint8), token_scale.view(-1)


def _check(
    ffn_in: torch.Tensor,
    full: FullWeights,
    expert_inputs: ExpertInputs,
    topk_ids: torch.Tensor,
) -> None:
    """Routes against the FP64 router and quant bytes against the FlashInfer
    recipe; pad rows stay zero and every rank holds the same outputs."""
    tokens = ffn_in.size(0)
    rank = dist.get_rank()
    failures = []
    if expert_inputs.fp4[tokens:].any() or expert_inputs.scales[tokens:].any():
        failures.append("the pad activation rows were written")
    logits = ffn_in.double() @ full.router.double().T
    score = torch.sigmoid(logits * ROUTER_MULTIPLIER)
    key = score + full.router_bias.double()
    eighth = torch.sort(key, dim=-1, descending=True).values[:, TOPK - 1 : TOPK]
    ids = topk_ids.long()
    distinct = torch.sort(ids, dim=-1).values.diff(dim=-1).ne(0).all()
    # The router GEMM is FP32-accurate, so only FP32-level near-ties may differ.
    if not (bool(distinct) and bool((key.gather(-1, ids) >= eighth - 1e-6).all())):
        failures.append(f"top-8 is not a top-8 of the FP64 keys: {ids.tolist()}")
    packed = expert_inputs.packed
    if not torch.equal(packed >> 16, topk_ids):
        failures.append("packed ids differ from the returned top-8")
    chosen = score.gather(-1, ids)
    weights = chosen * ROUTE_SCALE / (chosen.sum(-1, keepdim=True) + ROUTE_EPS)
    route_weights = (packed & 0xFFFF).to(torch.int16).view(torch.bfloat16).double()
    if float((route_weights - weights).abs().max()) > 1e-3:
        failures.append("route weights differ from the FP64 renormalized scores")
    values, scale_bytes, token_scale = _reference_quant(ffn_in, full)
    got_values = e2m1_values(expert_inputs.fp4[:tokens])
    # Pooler GEMMs round their BF16 output differently, which moves a few blocks.
    if float((got_values == values).double().mean()) < 0.999:
        failures.append("NVFP4 values differ from the FlashInfer recipe")
    if float((expert_inputs.scales[:tokens] == scale_bytes).double().mean()) < 0.999:
        failures.append("block scales differ from the FlashInfer recipe")
    got_scale = expert_inputs.per_token_scale[:tokens].double()
    if float(((got_scale - token_scale) / token_scale).abs().max()) > 1e-3:
        failures.append("token scales differ from the FlashInfer recipe")
    for name, value in (
        ("fp4", expert_inputs.fp4),
        ("scales", expert_inputs.scales),
        ("per_token_scale", expert_inputs.per_token_scale),
        ("packed", packed),
    ):
        mine = value.contiguous().view(torch.uint8)
        peers = [torch.empty_like(mine) for _ in range(WORLD_SIZE)]
        dist.all_gather(peers, mine)
        if not all(torch.equal(peer, peers[0]) for peer in peers):
            failures.append(f"ranks disagree on {name}")
    if failures:
        raise AssertionError(f"rank {rank}: {failures}")


def correctness_check(group: dist.ProcessGroup) -> None:
    """Two weight draws per T, alternated, so cross-launch state is cycled."""
    device = torch.device("cuda")
    for tokens in VERIFY_TOKENS:
        buffers = make_buffers(tokens, group)
        expert_inputs = ExpertInputs.zeros(tokens, device)
        cases = []
        for seed in (1000, 2000):
            full = make_full_weights(seed)
            cases.append((full, prepare_weights(full, group.rank())))
        for _ in range(2):
            for seed, (full, weights) in zip((1000, 2000), cases, strict=True):
                ffn_in = make_input(seed + 100, tokens)
                topk_ids = launch(ffn_in, weights, buffers, expert_inputs, group)
                _check(ffn_in, full, expert_inputs, topk_ids)
        dist.barrier()


def use_cudagraph() -> bool:
    return True


def _sglang_default(
    ffn_in: torch.Tensor, full: FullWeights
) -> Callable[[], tuple[torch.Tensor, ...]]:
    """sglang's B200 sequence, replicated on every rank: the pooler through its
    BF16 GEMM dispatcher on the served backend, then the fused pre-expert norm +
    NVFP4 quant, while the FP32 router GEMM and the Muse routing kernel run on a
    side stream, as served (no collective). Everything but the cuBLAS router GEMM
    launches with PDL, as in serving."""
    from sglang.kernels.jit.utils import is_arch_support_pdl
    from sglang.kernels.ops.muse_spark_v12.norm_fp4_quant import fused_norm_fp4_quant
    from sglang.kernels.ops.muse_spark_v12.routing import fused_muse_spark_routing
    from sglang.srt.layers.quantization import unquant
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.server_args import set_global_server_args_for_scheduler

    # The dispatcher stays on cuBLAS until configured: pick the served backend
    # (the CuTe DSL GEMM on SM100), as the scheduler does at startup.
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    unquant.initialize_bf16_gemm_config()

    # The sglang JIT kernels compile their PDL launch exactly when this holds.
    assert is_arch_support_pdl()
    tokens = ffn_in.size(0)
    packed = torch.empty(tokens, TOPK, dtype=torch.int32, device=ffn_in.device)
    side = torch.cuda.Stream()

    def call() -> tuple[torch.Tensor, ...]:
        # The served router reads the BF16 FFN input upcast, before the fork.
        router_input = ffn_in.float()
        main = torch.cuda.current_stream()
        side.wait_stream(main)
        expert_input = unquant.bf16_gemm_dispatch(ffn_in, full.pre_expert, None)
        fp4, scales, token_scale, _ = fused_norm_fp4_quant(
            expert_input, full.pre_expert_norm, NORM_EPS, GLOBAL_SCALE_INV
        )
        with torch.cuda.stream(side):
            logits = F.linear(router_input, full.router)
            _weights, topk_ids, _ = fused_muse_spark_routing(
                logits,
                full.router_bias,
                TOPK,
                ROUTE_SCALE,
                router_output_multiplier=ROUTER_MULTIPLIER,
                router_score_division_eps=ROUTE_EPS,
                packed_topk_ids=packed,
            )
        main.wait_stream(side)
        return fp4, scales, token_scale, topk_ids, packed

    return call


def _run(
    group: dist.ProcessGroup, *, verbose: bool, check_only: bool
) -> dict[str, Any]:
    correctness_check(group)
    if check_only:
        return {}
    rank = group.rank()
    full = make_full_weights(3000)
    weights = prepare_weights(full, rank)
    speedups = []
    for tokens in VERIFY_TOKENS:
        ffn_in = make_input(3100, tokens)
        buffers = make_buffers(tokens, group)
        expert_inputs = ExpertInputs.zeros(tokens, ffn_in.device)

        def helion_call(
            ffn_in: torch.Tensor = ffn_in,
            buffers: RouterBuffers = buffers,
            expert_inputs: ExpertInputs = expert_inputs,
        ) -> object:
            return launch(ffn_in, weights, buffers, expert_inputs, group)

        def validate(
            name: str,
            output: object,
            ffn_in: torch.Tensor = ffn_in,
            expert_inputs: ExpertInputs = expert_inputs,
        ) -> None:
            if name == "helion":
                _check(ffn_in, full, expert_inputs, cast("torch.Tensor", output))
                return
            fp4, scales, token_scale, topk_ids, packed = cast(
                "tuple[torch.Tensor, ...]", output
            )
            _check(
                ffn_in,
                full,
                ExpertInputs(
                    fp4=torch.cat((fp4, expert_inputs.fp4[fp4.size(0) :])),
                    scales=torch.cat(
                        (scales.view(torch.uint8), expert_inputs.scales[fp4.size(0) :])
                    ),
                    per_token_scale=token_scale.view(-1),
                    packed=packed,
                ),
                topk_ids,
            )

        results = benchmark(
            {
                "helion": helion_call,
                "sglang_default": _sglang_default(ffn_in, full),
            },
            validate=validate,
        )
        speedup = results["sglang_default"][1] / results["helion"][1]
        speedups.append(speedup)
        if verbose and rank == 0:
            print(f"T={tokens}")
            for name, (warm, cold) in results.items():
                print(f"{name:>16s}: {warm:7.2f} us warm, {cold:7.2f} us cold L2")
            print(f"{'speedup':>16s}: {speedup:.3f}x (cold L2)")
    return {
        "helion_wins": sum(s > 1 for s in speedups),
        "total": len(speedups),
        "geomean": math.prod(speedups) ** (1 / len(speedups)),
        "best_speedup": max(speedups),
    }


@torch.inference_mode()
def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """Run the TP4 FP64 check, then cold-L2 timings against the sglang default."""
    _rank, _local_rank, group = initialize(
        kernel_name="muse_spark_router",
        world_size=WORLD_SIZE,
        signal_pad_bytes=SIGNAL_PAD_BYTES,
    )
    try:
        return _run(group, verbose=verbose, check_only=check_only)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
