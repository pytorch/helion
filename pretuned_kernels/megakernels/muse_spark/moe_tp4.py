"""Muse Spark spec-decode verify expert-TP4 NVFP4 MoE as one persistent Helion
megakernel, including the post-expert norm finalize.

Every rank reads one weight bank: its FlashInfer TRT-LLM W13 I/W slices (aliased
by the stock parameters) followed by its H/W quarter of W2 over the full K,
stored as contiguous 256 x 256 B tiles. The routed activation z crosses ranks
once; the replicated y is gathered from every rank's quarter rows.

Roots (j / f: the routed experts in ascending order, ``slots`` bound):

    R1 W13 256-row tile, SwiGLU, NVFP4 z pushed to every rank     [2I/W/256, slots]
    R2 W2 quarter tile on all of z, routed rows and per-quarter
       sums of (norm_w * x)^2 scattered to symm by route          [H/W/256, slots]
    R3 per-route RMSNorm and weighted route sum of every rank's
       quarter -> y (replicated)                                  [tokens, H/W/128]

FC2 follows FlashInfer: the FC2 input scale is folded into ``up_alpha`` and
``out_alpha``, so z is quantized with a unit global scale.

Calls and graph replays must occur in identical collective order across ranks
on one ordered CUDA stream. The weights use FlashInfer's TRT-LLM layout, so this
example needs FlashInfer. ``main()`` compares against sglang's expert-TP default
(FlashInfer ``trtllm_fp4_block_scale_routed_moe``, the K-split all-reduce and the
CuTe DSL post-expert-norm finalize) and needs sglang (``muse-spark`` branch);
``--check`` runs only the FP64 correctness check::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \
      python -m torch.distributed.run --standalone --nproc-per-node=4 \
      pretuned_kernels/megakernels/muse_spark/moe_tp4.py [--check]
"""

from __future__ import annotations

import dataclasses
import functools
import math
import operator
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from pretuned_kernels.megakernels._distributed import benchmark
from pretuned_kernels.megakernels._distributed import initialize
from pretuned_kernels.megakernels.muse_spark._common import EXPERT_HIDDEN
from pretuned_kernels.megakernels.muse_spark._common import MOE_TOKENS
from pretuned_kernels.megakernels.muse_spark._common import NUM_EXPERTS
from pretuned_kernels.megakernels.muse_spark._common import SIGNAL_PAD_BYTES
from pretuned_kernels.megakernels.muse_spark._common import TOPK
from pretuned_kernels.megakernels.muse_spark._common import VERIFY_TOKENS
from pretuned_kernels.megakernels.muse_spark._common import WORLD_SIZE
from pretuned_kernels.megakernels.muse_spark._common import ExpertInputs
from pretuned_kernels.megakernels.muse_spark._common import dequantize_nvfp4
from pretuned_kernels.megakernels.muse_spark._common import encode_block_scale
from pretuned_kernels.megakernels.muse_spark._common import pack_e2m1x2
from pretuned_kernels.megakernels.muse_spark._common import relative_rmse_and_cosine
from pretuned_kernels.megakernels.muse_spark._common import round_e2m1
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

# The MoE runs in the pooled expert space: H here is EXPERT_HIDDEN, not HIDDEN.
HIDDEN = EXPERT_HIDDEN
INTERMEDIATE = 4096
NORM_EPS = 1e-8
FC2_GLOBAL_SCALE_INV = 1.0 / (448.0 * 6.0)
BF16_MAX = torch.finfo(torch.bfloat16).max
FP32_MAX = torch.finfo(torch.float32).max

MMA_M = 128
TILE_M = 256
TILE_K = 256  # e2m1x2 bytes per K step


def _popcount(v: torch.Tensor) -> torch.Tensor:
    """Set bits per int32 lane (SWAR: Triton cannot scan inside a WS loop)."""

    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    return (((v + (v >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24


def _distinct_table(
    ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Routed-expert bitmap words [8], their popcounts and exclusive prefix, and the
    distinct count; tables this narrow reduce within one warp."""

    lanes = hl.arange(8)
    row = ids.unsqueeze(0)
    bits = torch.where((row >> 5) == lanes.unsqueeze(1), 1 << (row & 31), 0)
    words = hl.reduce(operator.or_, bits, dim=1)
    counts = _popcount(words)
    before = hl.cumsum(counts, dim=0) - counts
    return words, counts, before, hl.reduce(operator.add, counts, dim=0)


def _expert_of(
    words: torch.Tensor, counts: torch.Tensor, before: torch.Tensor, j: object
) -> torch.Tensor:
    """The j-th distinct expert (ascending): its bitmap word, then the bit in it."""

    hit = (before <= j) & (before + counts > j)
    word = hl.reduce(operator.or_, torch.where(hit, words, 0), dim=0)
    key = hl.reduce(operator.or_, torch.where(hit, before * 8 + hl.arange(8), 0), dim=0)
    upto = _popcount(word & ((2 << hl.arange(32)) - 1))
    bit = hl.reduce(operator.add, (upto <= j - (key >> 3)).to(torch.int32), dim=0)
    return (key & 7) * 32 + bit


def _scale_tile(chunks: torch.Tensor) -> torch.Tensor:
    """[2, 16, 256] SWIZZLE_32_4_4 bytes of two 128-row tiles -> [256 rows, 32]."""

    return chunks.reshape(2, 8, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(256, 32)


def _logical_rows(acc: torch.Tensor) -> torch.Tensor:
    """[256 physical rows, t] -> logical order, per 128-row FI tile."""

    t = acc.size(1)
    return acc.reshape(2, 4, 4, 8, t).permute(0, 1, 3, 2, 4).reshape(256, t)


def _swiglu(
    acc: torch.Tensor,
    row_scale: torch.Tensor,
    up_alpha: torch.Tensor,
    gate_alpha: torch.Tensor,
) -> torch.Tensor:
    """Interleaved (up, gate) rows [256, t] -> SwiGLU [t, 128]."""

    t = acc.size(1)
    pairs = _logical_rows(acc).reshape(128, 2, t).permute(0, 2, 1)
    up_acc, gate_acc = hl.split(pairs)
    up = up_acc * row_scale[None, :] * up_alpha
    gate = gate_acc * row_scale[None, :] * gate_alpha
    silu = gate / (1.0 + torch.exp(-gate))
    return (silu * up).permute(1, 0)


def _quant_z(act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[t, 128] -> (e2m1x2 bytes [t, 64], E4M3 block scale bytes [t, 8])."""

    t = act.size(0)
    blocks = act.reshape(t, 8, 16)
    amax = torch.amax(torch.abs(blocks), dim=-1)
    fp8, mult = encode_block_scale(hl.full([t, 8], 1.0, dtype=torch.float32), amax)
    low, high = hl.split((blocks * mult[:, :, None]).reshape(t, 8, 8, 2))
    zq = pack_e2m1x2(low, high).to(torch.uint8).reshape(t, 64)
    return zq, fp8.to(torch.uint8)


def _muse_spark_moe_tp4_megakernel(
    weights: torch.Tensor,
    weights2: torch.Tensor,
    scales: torch.Tensor,
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    x_row_scale: torch.Tensor,
    packed_routing: torch.Tensor,
    routes: torch.Tensor,
    up_alpha: torch.Tensor,
    gate_alpha: torch.Tensor,
    out_alpha: torch.Tensor,
    z: torch.Tensor,
    z_scale: torch.Tensor,
    gemm2_ws: torch.Tensor,
    sumsq_ws: torch.Tensor,
    norm_weight: torch.Tensor,
    y: torch.Tensor,
    group_name: hl.ProcessGroupName,
    rank: hl.constexpr,
    world: hl.constexpr,
    norm_eps: hl.constexpr,
) -> torch.Tensor:
    hl.specialize(weights.size())
    hl.specialize(weights.stride())
    hl.specialize(weights2.size())
    hl.specialize(weights2.stride())
    hl.specialize(scales.size())
    hl.specialize(scales.stride())
    hl.specialize(x_q.size())
    hl.specialize(x_q.stride())
    hl.specialize(x_scale.size())
    hl.specialize(x_scale.stride())
    hl.specialize(x_row_scale.size())
    hl.specialize(x_row_scale.stride())
    hl.specialize(packed_routing.size())
    hl.specialize(packed_routing.stride())
    hl.specialize(routes.size())
    hl.specialize(routes.stride())
    hl.specialize(up_alpha.size())
    hl.specialize(up_alpha.stride())
    hl.specialize(gate_alpha.size())
    hl.specialize(gate_alpha.stride())
    hl.specialize(out_alpha.size())
    hl.specialize(out_alpha.stride())
    hl.specialize(z.size())
    hl.specialize(z.stride())
    hl.specialize(z_scale.size())
    hl.specialize(z_scale.stride())
    hl.specialize(gemm2_ws.size())
    hl.specialize(gemm2_ws.stride())
    hl.specialize(sumsq_ws.size())
    hl.specialize(sumsq_ws.stride())
    hl.specialize(norm_weight.size())
    hl.specialize(norm_weight.stride())
    hl.specialize(y.size())
    hl.specialize(y.stride())

    hidden = y.size(1)
    local = gemm2_ws.size(1)
    topk = packed_routing.size(1)
    slots = routes.size(0)
    tokens = packed_routing.size(0)
    rows = x_q.size(0)  # MMA token rows, T padded to 16
    k_steps = weights.size(1)
    q_tiles = z.size(2) * 4 // world // 256
    h_tiles = sumsq_ws.size(1)
    w13_tiles = up_alpha.size(0) * q_tiles
    remote_z = torch.ops.symm_mem.get_remote_tensors(z, group_name)
    remote_z_scale = torch.ops.symm_mem.get_remote_tensors(z_scale, group_name)
    remote_g2 = torch.ops.symm_mem.get_remote_tensors(gemm2_ws, group_name)
    remote_sumsq = torch.ops.symm_mem.get_remote_tensors(sumsq_ws, group_name)
    y3 = y.view(tokens, world, local)
    # x and its scales as one [1, rows, K] slot, laid out like a z slot.
    x3 = x_q.view(1, rows, x_q.size(1))
    xs3 = x_scale.view(1, rows, x_scale.size(1))
    # Activation block scales load as bytes and are viewed as e4m3.
    e4m3 = torch.float8_e4m3fn

    hl.pdl_wait()
    # Trigger at entry: the stock unpooler GEMM that follows streams its weights
    # in while R3 finishes; its own wait still orders it after this grid.
    hl.pdl_launch_dependents()
    # R1. Stock: the FlashInfer TRT-LLM MoE's routing, GEMM1 (W13, SwiGLU) and
    # its NVFP4 quant of the intermediate.
    for w13_tile_q, w13_tile_j in hl.tile([q_tiles, slots], block_size=[1, 1]):
        words, counts, before, distinct = _distinct_table(routes[:] >> 16)
        # Slots past the distinct count are skipped; their z is never read.
        if w13_tile_j.begin < distinct:
            expert = _expert_of(words, counts, before, w13_tile_j.begin)
            wt = expert * q_tiles + w13_tile_q.begin
            acc = hl.zeros([256, rows], dtype=torch.float32)
            # Staggered K order across tasks spreads FI weight rows over DRAM
            # channels. It is keyed on the expert, not its slot, so a token's sums
            # do not depend on the window's other routes.
            rot = 3 * (expert * q_tiles + w13_tile_q.begin)
            for w13_tile_k in hl.tile(k_steps, block_size=1):
                kk = (w13_tile_k.begin + rot) % k_steps
                # Weight bytes stream once: evict_first keeps L2 for the rest.
                lhs = hl.load(
                    weights,
                    [wt, kk, slice(None), slice(None)],
                    eviction_policy="evict_first",
                )
                lhs_scale = _scale_tile(
                    hl.load(
                        scales,
                        [wt, slice(None), kk * 16 + hl.arange(16), slice(None)],
                        eviction_policy="evict_first",
                    )
                )
                rhs = x3[0, :, kk * 256 + hl.arange(256)]
                rhs_scale = xs3[0, :, kk * 32 + hl.arange(32)].view(e4m3)
                acc = hl.dot_scaled(
                    lhs,
                    lhs_scale,
                    "e2m1",
                    rhs.permute(1, 0),
                    rhs_scale,
                    "e2m1",
                    acc=acc,
                    out_dtype=torch.float32,
                )
            act = _swiglu(
                acc,
                x_row_scale[hl.arange(rows)],
                up_alpha[expert],
                gate_alpha[expert],
            )
            zq, zsc = _quant_z(act)
            col = (rank * q_tiles + w13_tile_q.begin) * 64
            scol = (rank * q_tiles + w13_tile_q.begin) * 8
            for peer_z in remote_z:
                peer_z[w13_tile_j, :, col + hl.arange(64)] = zq[None, :, :]
            for peer_zs in remote_z_scale:
                peer_zs[w13_tile_j, :, scol + hl.arange(8)] = zsc[None, :, :]

    # R2. Stock: the FlashInfer TRT-LLM MoE's GEMM2 (W2).
    for w2_tile_h, w2_tile_f in hl.tile([h_tiles, slots], block_size=[1, 1]):
        words2, counts2, before2, distinct2 = _distinct_table(routes[:] >> 16)
        if w2_tile_f.begin < distinct2:
            expert2 = _expert_of(words2, counts2, before2, w2_tile_f.begin)
            wt2 = w13_tiles + expert2 * h_tiles + w2_tile_h.begin
            acc2 = hl.zeros([256, rows], dtype=torch.float32)
            # Same K staggering as R1.
            rot2 = 3 * (expert2 * h_tiles + w2_tile_h.begin)
            for w2_tile_k in hl.tile(k_steps, block_size=1):
                jj = (w2_tile_k.begin + rot2) % k_steps
                lhs2 = hl.load(
                    weights2,
                    [wt2, jj, slice(None), slice(None)],
                    eviction_policy="evict_first",
                )
                lhs2_scale = _scale_tile(
                    hl.load(
                        scales,
                        [wt2, slice(None), jj * 16 + hl.arange(16), slice(None)],
                        eviction_policy="evict_first",
                    )
                )
                rhs2 = z[w2_tile_f.begin, :, jj * 256 + hl.arange(256)]
                rhs2_scale = z_scale[w2_tile_f.begin, :, jj * 32 + hl.arange(32)].view(
                    e4m3
                )
                acc2 = hl.dot_scaled(
                    lhs2,
                    lhs2_scale,
                    "e2m1",
                    rhs2.permute(1, 0),
                    rhs2_scale,
                    "e2m1",
                    acc=acc2,
                    out_dtype=torch.float32,
                )
            val = (
                (_logical_rows(acc2) * out_alpha[expert2])
                .permute(1, 0)
                .to(torch.bfloat16)
            )
            # Each token routes an expert at most once: scatter to its route row.
            # MMA pad rows (T = 8) have no routes, so the scatter drops them.
            slot_id = hl.arange(rows)[:, None] * topk + hl.arange(topk)[None, :]
            live = slot_id < slots
            ids = hl.load(routes, [slot_id], extra_mask=live) >> 16
            match = ((ids == expert2) & live).to(torch.int32)
            route_row = torch.sum(match * slot_id, dim=-1)
            routed = torch.sum(match, dim=-1) > 0
            cols = w2_tile_h.begin * 256 + hl.arange(256)
            hl.store(gemm2_ws, [route_row, cols], val, extra_mask=routed[:, None])
            nw = norm_weight[rank * local + cols].to(torch.float32)
            gm = val.to(torch.float32) * nw[None, :]
            hl.store(
                sumsq_ws,
                [route_row, w2_tile_h.begin],
                torch.sum(gm * gm, dim=-1),
                extra_mask=routed,
            )

    # R3. Stock: the expert-TP all-reduce (reduce_partials), then
    # finalize_with_post_expert_norm (per-route RMSNorm, weighted top-k sum).
    for fin_tile_s, fin_tile_c in hl.tile([slots, local], block_size=[topk, 128]):
        # One token's routes: every rank's sum-of-squares partials and rows.
        bits = (routes[fin_tile_s] & 0xFFFF).to(torch.int16)
        weight = bits.view(torch.bfloat16).to(torch.float32)
        # Norm weights load before the first polled read, overlapping the wait.
        nws = [
            norm_weight[r * local + fin_tile_c.index] for r in hl.static_range(world)
        ]
        ss = hl.zeros([fin_tile_s], dtype=torch.float32)
        for peer_sumsq in remote_sumsq:
            ss = ss + torch.sum(peer_sumsq[fin_tile_s, :], dim=1)
        scale = torch.rsqrt(ss / hidden + norm_eps) * weight
        parts = [peer_g2[fin_tile_s, fin_tile_c] for peer_g2 in remote_g2]
        for r in hl.static_range(world):
            nw_r = nws[r].to(torch.float32)
            y3[fin_tile_s.id, r, fin_tile_c] = torch.sum(
                parts[r].to(torch.float32) * nw_r[None, :] * scale[:, None], dim=0
            ).to(torch.bfloat16)
    return y


def _config() -> helion.Config:
    # R1/R2 K loops are warp-specialized 3-stage TMA rings. Activation scales, R1's
    # row scale and alphas, and the gemm2 scatter are small: pointers beat TMA there.
    # Indexing slots follow the source's loads and stores: R1 0-15 (4 x scales,
    # 5 row scale, 6-7 alphas, 8-15 the z pushes), R2 16-25 (20 z scales, 23 the
    # gemm2 scatter), R3 26-42. The range lists are R1, its K loop, R2, its K
    # loop and R3; editing the kernel shifts both.
    indexing = ["tensor_descriptor"] * 43
    for slot in (4, 5, 6, 7, 20, 23):
        indexing[slot] = "pointer"
    return helion.Config(
        block_sizes=[],
        cross_loop_pipeline="static",
        host_tensor_descriptors=True,
        indexing=indexing,
        num_sm_multiplier=1,
        num_stages=2,
        num_warps=4,
        pid_type="persistent_blocked",
        range_num_stages=[0, 3, 0, 3, 0],
        range_warp_specializes=[True, None, True, None, None],
    )


muse_spark_moe_tp4 = helion.kernel(
    _muse_spark_moe_tp4_megakernel,
    config=_config(),
    static_shapes=True,
    backend="triton",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)


@dataclasses.dataclass(frozen=True)
class MoeTp4VerifyBank:
    """One rank's weight bank and the stock FlashInfer W13 views that alias it."""

    weights: torch.Tensor  # [W13 tiles, K / 512, 256, 256] u8 view of [rows, K / 2]
    weights2: torch.Tensor  # [tiles, K / 512, 256, 256] u8; R2 reads its W2 tiles
    scales: torch.Tensor  # [tiles, 2, K / 512 * 16, 256] e4m3
    w13: torch.Tensor  # [E, 2I / W, H / 2] u8
    w13_scale: torch.Tensor  # FlashInfer-shuffled W13 block scales, flat e4m3


def build_moe_tp4_verify_bank(
    *,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    group: dist.ProcessGroup,
) -> MoeTp4VerifyBank:
    """Copy this rank's FI W13 into a new bank and append its H/W quarter of W2
    over the full K as contiguous tiles, all-to-all'd from every rank's K slice."""

    world = group.size()
    experts, two_local, packed_k = w13.shape
    hidden, packed_local = w2.size(1), w2.size(2)
    # R1 and R2 share K steps: the full W2 K (I) must equal W13's K (H).
    assert world * packed_local == packed_k
    quarter = hidden // world
    w13_rows, w2_rows = experts * two_local, experts * quarter
    device = w13.device
    bank = torch.empty(w13_rows + w2_rows, packed_k, dtype=torch.uint8, device=device)
    bank[:w13_rows].copy_(w13.view(torch.uint8).reshape(w13_rows, packed_k))
    send = (
        w2.view(torch.uint8)
        .view(experts, world, quarter, packed_local)
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    # W2 tiles are [K steps, 256, 256 B] boxes: one 64 KiB DRAM run per TMA load.
    k_steps = packed_k // TILE_K
    w2_tiles = w2_rows // TILE_M
    rows2 = recv.permute(1, 2, 0, 3).reshape(w2_tiles, TILE_M, k_steps, TILE_K)
    bank[w13_rows:].view(w2_tiles, k_steps, TILE_M, TILE_K).copy_(
        rows2.permute(0, 2, 1, 3)
    )
    del send, recv, rows2

    # Block scales are contiguous [K steps, 32 * 128] chunks per 128-row tile.
    sf_bytes = packed_k // 8
    sbank = torch.empty(
        (w13_rows + w2_rows) * sf_bytes, dtype=torch.uint8, device=device
    )
    sbank[: w13_rows * sf_bytes].copy_(w13_scale.view(torch.uint8).reshape(-1))
    tile_bytes = MMA_M * packed_local // 8
    ssend = (
        w2_scale.view(torch.uint8)
        .reshape(experts, world, quarter // MMA_M, tile_bytes)
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    srecv = torch.empty_like(ssend)
    dist.all_to_all_single(srecv, ssend, group=group)
    sbank[w13_rows * sf_bytes :].view(
        experts, quarter // MMA_M, world, tile_bytes
    ).copy_(srecv.permute(1, 2, 0, 3))

    tiles = (w13_rows + w2_rows) // TILE_M
    w13_tiles = w13_rows // TILE_M
    e4m3 = torch.float8_e4m3fn
    return MoeTp4VerifyBank(
        weights=bank[:w13_rows]
        .view(w13_tiles, TILE_M, k_steps, TILE_K)
        .permute(0, 2, 1, 3),
        weights2=bank.view(tiles, k_steps, TILE_M, TILE_K),
        scales=sbank.view(e4m3).view(tiles, 2, k_steps * 16, TILE_M),
        w13=bank[:w13_rows].view(experts, two_local, packed_k),
        w13_scale=sbank[: w13_rows * sf_bytes].view(e4m3),
    )


@dataclasses.dataclass(frozen=True)
class MoeWorkspaces:
    """Symmetric per-rank buffers for one verify window size."""

    z: torch.Tensor  # [slots, 16, I / 2] u8, the routed NVFP4 activation
    z_scale: torch.Tensor  # [slots, 16, I / 16] u8, its E4M3 block scales
    gemm2_ws: torch.Tensor  # [slots, H / W] bf16 routed W2 rows
    sumsq_ws: torch.Tensor  # [slots, H / W / 256] fp32 sums of (norm_w * x)^2


# Symmetric buffers live for the process: torch finds a symmetric-memory handle by
# address, so a larger buffer at a freed smaller one's address gets its stale handle.
@functools.cache
def make_workspaces(tokens: int, group: dist.ProcessGroup) -> MoeWorkspaces:
    device = torch.device("cuda", torch.cuda.current_device())
    local, slots = HIDDEN // WORLD_SIZE, tokens * TOPK
    workspaces = MoeWorkspaces(
        z=symm_mem.empty(
            slots, MOE_TOKENS, INTERMEDIATE // 2, dtype=torch.uint8, device=device
        ),
        z_scale=symm_mem.empty(
            slots, MOE_TOKENS, INTERMEDIATE // 16, dtype=torch.uint8, device=device
        ),
        gemm2_ws=symm_mem.empty(slots, local, dtype=torch.bfloat16, device=device),
        sumsq_ws=symm_mem.empty(
            slots, local // TILE_M, dtype=torch.float32, device=device
        ),
    )
    # dataclasses.astuple would deep-copy the tensors; rendezvous needs the allocations.
    for field in dataclasses.fields(workspaces):
        symm_mem.rendezvous(getattr(workspaces, field.name), group.group_name)
    return workspaces


def launch(
    bank: MoeTp4VerifyBank,
    inputs: ExpertInputs,
    alphas: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    norm_weight: torch.Tensor,
    workspaces: MoeWorkspaces,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """The post-expert-normed route sum y [T, H], replicated on every rank."""
    tokens = inputs.packed.size(0)
    y = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=inputs.fp4.device)
    up_alpha, gate_alpha, out_alpha = alphas
    muse_spark_moe_tp4(
        bank.weights,
        bank.weights2,
        bank.scales,
        inputs.fp4,
        inputs.scales.view(torch.uint8).reshape(MOE_TOKENS, -1),
        inputs.per_token_scale,
        inputs.packed,
        inputs.packed.view(-1),
        up_alpha,
        gate_alpha,
        out_alpha,
        workspaces.z,
        workspaces.z_scale,
        workspaces.gemm2_ws,
        workspaces.sumsq_ws,
        norm_weight,
        y,
        group.group_name,
        group.rank(),
        WORLD_SIZE,
        NORM_EPS,
    )
    return y


# Routing draws: same8 routes every token to one set of 8 experts, shared picks
# from a 20-expert pool, uniform from all 256, disjoint gives every route its own
# expert.
ROUTE_SEEDS = {"uniform": 1, "shared": 2, "same8": 3, "disjoint": 4}


@dataclasses.dataclass(frozen=True)
class LogicalExperts:
    """Checkpoint-order weights of the loaded expert ids."""

    ids: list[int]
    w13: torch.Tensor  # [n, 2I, H/2] u8, rows [w1 (gate); w3 (up)]
    w13_scale: torch.Tensor  # [n, 2I, H/16] e4m3
    w13_scale_2: torch.Tensor  # [n, 2] fp32 (gate, up)
    w2: torch.Tensor  # [n, H, I/2] u8
    w2_scale: torch.Tensor  # [n, H, I/16] e4m3
    w2_scale_2: torch.Tensor  # [n] fp32


@dataclasses.dataclass(frozen=True)
class RankWeights:
    """This rank's expert-TP slices in FlashInfer's TRT-LLM layout, [256, ...]."""

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor


@dataclasses.dataclass(frozen=True)
class Draw:
    """One T-token routing, its NVFP4 activation and FP32 SwiGLU rows."""

    kind: str
    packed: torch.Tensor  # [T, 8] int32 expert << 16 | bf16 route weight
    hidden_states: torch.Tensor  # [T, H/2] u8
    hidden_states_scale: torch.Tensor  # [T, H/16] e4m3
    per_token_scale: torch.Tensor  # [T] fp32
    # (expert slot, tokens, top-k slots, FP32 SwiGLU rows [n, I])
    rows: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]


@dataclasses.dataclass(frozen=True)
class Case:
    """One bank holding every draw's experts, and the kernel's FC scalars."""

    logical: LogicalExperts
    rank_weights: RankWeights
    bank: MoeTp4VerifyBank
    fc2_scale: float
    alphas: tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # up, gate, out [E]
    norm_weight: torch.Tensor  # [H] bf16
    draws: list[Draw]


def _routes(kind: str, tokens: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(ROUTE_SEEDS[kind])
    order = torch.randperm(NUM_EXPERTS, generator=g)
    if kind == "disjoint":
        return order[: tokens * TOPK].view(tokens, TOPK)
    pool = order[: {"same8": TOPK, "shared": 20, "uniform": NUM_EXPERTS}[kind]]
    picks = [
        pool[torch.randperm(pool.numel(), generator=g)[:TOPK]] for _ in range(tokens)
    ]
    return torch.stack(picks)


def _random_experts(ids: list[int], seed: int) -> LogicalExperts:
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = len(ids)

    def scales(*shape: int) -> torch.Tensor:
        # E4M3 values in [2^-9, 2^-5): a spread of exponents, no zeros.
        exponent = torch.randint(-9, -5, shape, device="cuda", generator=g)
        mantissa = 1 + torch.rand(shape, device="cuda", generator=g)
        return (mantissa * torch.exp2(exponent.float())).to(torch.float8_e4m3fn)

    def nibbles(*shape: int) -> torch.Tensor:
        return torch.randint(
            0, 256, shape, dtype=torch.uint8, device="cuda", generator=g
        )

    return LogicalExperts(
        ids=ids,
        w13=nibbles(n, 2 * INTERMEDIATE, HIDDEN // 2),
        w13_scale=scales(n, 2 * INTERMEDIATE, HIDDEN // 16),
        w13_scale_2=0.5 + torch.rand(n, 2, device="cuda", generator=g),
        w2=nibbles(n, HIDDEN, INTERMEDIATE // 2),
        w2_scale=scales(n, HIDDEN, INTERMEDIATE // 16),
        w2_scale_2=0.5 + torch.rand(n, device="cuda", generator=g),
    )


def _flashinfer_layout(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashInfer's TRT-LLM NVFP4 MoE layout: per expert, rows permuted for the
    128-row epilogue tiles (W13 as [w3; w1] gated pairs) and the permuted block
    scales interleaved."""
    from flashinfer import nvfp4_block_scale_interleave
    from flashinfer.fused_moe.core import _maybe_get_cached_w3_w1_permute_indices
    from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache

    cache: dict[torch.Size, torch.Tensor] = {}
    u8 = torch.uint8
    out = [torch.empty_like(t.view(u8)) for t in (w13, w13_scale, w2, w2_scale)]
    for i in range(w13.size(0)):
        index = _maybe_get_cached_w3_w1_permute_indices(
            cache, w13[i].view(u8), 128, is_gated_act_gemm=True
        )
        out[0][i] = w13[i].view(u8)[index.to(w13.device)]
        index = _maybe_get_cached_w3_w1_permute_indices(
            cache,
            w13_scale[i].view(u8),
            128,
            num_elts_per_sf=16,
            is_gated_act_gemm=True,
        )
        out[1][i] = nvfp4_block_scale_interleave(
            w13_scale[i].view(u8)[index.to(w13.device)].contiguous()
        ).view(out[1][i].shape)
        index = get_w2_permute_indices_with_cache(cache, w2[i].view(u8), 128)
        out[2][i] = w2[i].view(u8)[index.to(w2.device)]
        index = get_w2_permute_indices_with_cache(
            cache, w2_scale[i].view(u8), 128, num_elts_per_sf=16
        )
        out[3][i] = nvfp4_block_scale_interleave(
            w2_scale[i].view(u8)[index.to(w2.device)].contiguous()
        ).view(out[3][i].shape)
    return out[0], out[1], out[2], out[3]


def _rank_weights(logical: LogicalExperts, rank: int) -> RankWeights:
    """Expert-TP narrow of the checkpoint tensors in the served FlashInfer layout."""
    local = INTERMEDIATE // WORLD_SIZE
    gate = slice(rank * local, (rank + 1) * local)
    up = slice(INTERMEDIATE + rank * local, INTERMEDIATE + (rank + 1) * local)
    # FlashInfer's gated GEMM takes [w3 (up); w1 (gate)].
    w13 = torch.cat((logical.w13[:, up], logical.w13[:, gate]), dim=1)
    w13_scale = torch.cat((logical.w13_scale[:, up], logical.w13_scale[:, gate]), dim=1)
    k2 = slice(rank * local // 2, (rank + 1) * local // 2)
    w2 = logical.w2[:, :, k2].contiguous()
    w2_scale = logical.w2_scale[:, :, rank * local // 16 : (rank + 1) * local // 16]
    w13, w13_scale, w2, w2_scale = _flashinfer_layout(
        w13.contiguous(), w13_scale.contiguous(), w2, w2_scale.contiguous()
    )
    ids = torch.tensor(logical.ids, device="cuda")

    def scatter(values: torch.Tensor, *shape: int) -> torch.Tensor:
        full = torch.zeros((NUM_EXPERTS, *shape), dtype=torch.uint8, device="cuda")
        full[ids] = values.view(torch.uint8).reshape(len(ids), *shape)
        return full

    e4m3 = torch.float8_e4m3fn
    return RankWeights(
        w13=scatter(w13, 2 * local, HIDDEN // 2),
        w13_scale=scatter(w13_scale, 2 * local, HIDDEN // 16).view(e4m3),
        w2=scatter(w2, HIDDEN, local // 2),
        w2_scale=scatter(w2_scale, HIDDEN, local // 16).view(e4m3),
    )


def _kernel_alphas(
    logical: LogicalExperts, fc2_scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(up, gate, out) FP32 alphas; as served, the static FC2 input scale is
    folded into up and out, so the kernel quantizes z with a unit global scale."""
    ids = torch.tensor(logical.ids, device="cuda")
    up, gate, out = (torch.ones(NUM_EXPERTS, device="cuda") for _ in range(3))
    up[ids] = logical.w13_scale_2[:, 1] / fc2_scale
    gate[ids] = logical.w13_scale_2[:, 0]
    out[ids] = logical.w2_scale_2 * fc2_scale
    return up, gate, out


def _swiglu_rows(
    logical: LogicalExperts,
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    per_token_scale: torch.Tensor,
) -> list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """FP64 W13 GEMMs of each routed expert's tokens, then the FP32 SwiGLU with
    the per-token scale and the unfolded W13 alphas."""
    topk_ids = topk_ids.cuda()
    out = []
    for slot, expert in enumerate(logical.ids):
        tokens, topk_slots = (topk_ids == expert).nonzero().unbind(-1)
        if tokens.numel() == 0:
            continue
        w13 = dequantize_nvfp4(logical.w13[slot], logical.w13_scale[slot])
        acc = (x[tokens] @ w13.T).float().clamp(-FP32_MAX, FP32_MAX)
        scale = per_token_scale[tokens][:, None] * logical.w13_scale_2[slot]
        gate = acc[:, :INTERMEDIATE] * scale[:, 0:1]
        up = acc[:, INTERMEDIATE:] * scale[:, 1:2]
        out.append((slot, tokens, topk_slots, gate / (1.0 + torch.exp(-gate)) * up))
    return out


def _make_draw(kind: str, index: int, logical: LogicalExperts, tokens: int) -> Draw:
    topk_ids = _routes(kind, tokens)
    g = torch.Generator(device="cuda").manual_seed(100 + ROUTE_SEEDS[kind])
    exponent = torch.randint(-8, -4, (tokens, HIDDEN // 16), device="cuda", generator=g)
    hidden_states = torch.randint(
        0, 256, (tokens, HIDDEN // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    hidden_states_scale = torch.exp2(exponent.float()).to(torch.float8_e4m3fn)
    per_token_scale = 0.5 + torch.rand(tokens, device="cuda", generator=g)
    g = torch.Generator(device="cuda").manual_seed(5 + index)
    raw = 0.25 + torch.rand(tokens, TOPK, device="cuda", generator=g)
    scores = (raw / raw.sum(-1, keepdim=True)).bfloat16().view(torch.int16)
    packed = (topk_ids.to(torch.int32).cuda() << 16) | (scores.to(torch.int32) & 0xFFFF)
    x = dequantize_nvfp4(hidden_states, hidden_states_scale)
    return Draw(
        kind=kind,
        packed=packed,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        per_token_scale=per_token_scale,
        rows=_swiglu_rows(logical, topk_ids, x, per_token_scale),
    )


def make_case(kinds: tuple[str, ...], group: dist.ProcessGroup, tokens: int) -> Case:
    """Every draw's experts in one bank; the static FC2 input scale is calibrated
    on the routed rows of all draws, as a checkpoint's input scale would be."""
    ids = sorted(
        set().union(*(_routes(kind, tokens).unique().tolist() for kind in kinds))
    )
    logical = _random_experts(ids, seed=7)
    rank_weights = _rank_weights(logical, group.rank())
    bank = build_moe_tp4_verify_bank(
        w13=rank_weights.w13,
        w13_scale=rank_weights.w13_scale,
        w2=rank_weights.w2,
        w2_scale=rank_weights.w2_scale,
        group=group,
    )
    draws = [_make_draw(kind, i, logical, tokens) for i, kind in enumerate(kinds)]
    amax = max(rows.abs().max().item() for d in draws for *_, rows in d.rows)
    fc2_scale = amax * FC2_GLOBAL_SCALE_INV
    g = torch.Generator(device="cuda").manual_seed(7)
    norm_weight = 1.0 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=g)
    return Case(
        logical=logical,
        rank_weights=rank_weights,
        bank=bank,
        fc2_scale=fc2_scale,
        alphas=_kernel_alphas(logical, fc2_scale),
        norm_weight=norm_weight.bfloat16(),
        draws=draws,
    )


def _quant_unit(act: torch.Tensor) -> torch.Tensor:
    """NVFP4 with a unit global scale (the FC2 input scale is pre-applied),
    dequantized to FP64: the megakernel's FC2 input."""
    blocks = act.float().reshape(act.shape[0], -1, 16)
    fp8 = (blocks.abs().amax(-1) / 6.0).clamp(max=448.0)
    fp8 = fp8.to(torch.float8_e4m3fn).float()
    x = blocks * torch.where(fp8 != 0, 1.0 / fp8, 0.0)[..., None]
    return (round_e2m1(x) * fp8[..., None]).reshape(act.shape).double()


def _quant_full_row(act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashInfer's dynamic BF16 -> NVFP4 FC2 requant of full rows (its B=1
    recipe), dequantized: (FP64 values [R, I], FP32 token scales [R])."""
    act = act.clamp(-BF16_MAX, BF16_MAX).bfloat16()
    blocks = act.float().reshape(act.shape[0], -1, 16)
    block_amax = blocks.abs().amax(-1)
    token_scale = block_amax.amax(-1, keepdim=True) * FC2_GLOBAL_SCALE_INV
    encode = torch.where(token_scale != 0, 1.0 / token_scale, FP32_MAX)
    fp8 = (encode * (block_amax * (1.0 / 6.0))).clamp(max=448.0)
    fp8 = fp8.to(torch.float8_e4m3fn).float()
    out_scale = torch.where(fp8 != 0, 1.0 / (fp8 * (1.0 / encode)), 0.0)
    values = (round_e2m1(blocks * out_scale[..., None]) * fp8[..., None]).reshape(
        act.shape
    )
    return values.double(), token_scale.view(-1)


def _references(case: Case, draw: Draw) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """(routed W2 rows [T * 8, H], finalized y [T, H]) per FC2 input: the
    megakernel's static recipe, FlashInfer's full-row requant and no requant."""
    logical, fc2 = case.logical, case.fc2_scale
    tokens = len(draw.packed)
    gemm2 = {
        mode: torch.empty(tokens, TOPK, HIDDEN, dtype=torch.bfloat16, device="cuda")
        for mode in ("static", "full_row", "exact")
    }
    for slot, rows_tokens, topk_slots, act in draw.rows:
        w2 = dequantize_nvfp4(logical.w2[slot], logical.w2_scale[slot])
        alpha = float(logical.w2_scale_2[slot])
        full_row, token_scale = _quant_full_row(act)
        inputs = {
            "static": (_quant_unit(act / fc2), fc2),
            "full_row": (full_row, token_scale.double()[:, None]),
            "exact": (act.double(), 1.0),
        }
        for mode, (z, scale) in inputs.items():
            rows = ((z @ w2.T) * (scale * alpha)).clamp(-BF16_MAX, BF16_MAX)
            gemm2[mode][rows_tokens, topk_slots] = rows.bfloat16()
    # The stock weight-first post-expert RMSNorm per route, then the weighted sum.
    weights = (draw.packed & 0xFFFF).to(torch.int16).view(torch.bfloat16).double()
    out = {}
    for mode, rows in gemm2.items():
        normed = (rows.float() * case.norm_weight.float()).double()
        rstd = torch.rsqrt(normed.pow(2).mean(-1, keepdim=True) + NORM_EPS)
        y = (weights[..., None] * normed * rstd).sum(1).float().bfloat16()
        out[mode] = (rows.view(tokens * TOPK, HIDDEN), y)
    return out


def _rrmse(got: torch.Tensor, expected: torch.Tensor) -> float:
    return relative_rmse_and_cosine(got, expected)[0]


def _load(inputs: ExpertInputs, draw: Draw) -> None:
    """Write one draw into the slot's first T rows, as the router does."""
    tokens = len(draw.packed)
    inputs.fp4[:tokens] = draw.hidden_states
    inputs.scales[:tokens] = draw.hidden_states_scale.view(torch.uint8)
    inputs.per_token_scale[:tokens] = draw.per_token_scale
    inputs.packed.copy_(draw.packed)


def _check(
    case: Case,
    draw: Draw,
    y: torch.Tensor,
    gemm2: torch.Tensor | None = None,
    *,
    recipe: str | None = "static",
) -> None:
    """Routed rows and y against the FP64 reference of ``recipe``, and no farther
    from no-requant than a full-row requant. Recipe None checks only the latter
    (an FC2 input requant this file does not model)."""
    rank = dist.get_rank()
    refs = _references(case, draw)
    columns = slice(rank * HIDDEN // WORLD_SIZE, (rank + 1) * HIDDEN // WORLD_SIZE)
    errors = {
        "y_vs_exact": _rrmse(y, refs["exact"][1]),
        "full_row_vs_exact": _rrmse(refs["full_row"][1], refs["exact"][1]),
    }
    if recipe is not None:
        errors["y"] = _rrmse(y, refs[recipe][1])
        if gemm2 is not None:
            errors["gemm2"] = _rrmse(gemm2, refs[recipe][0][:, columns])
    close = all(errors[k] < 1e-3 for k in ("y", "gemm2") if k in errors)
    no_worse = errors["y_vs_exact"] <= 1.05 * errors["full_row_vs_exact"]
    if not (close and no_worse):
        raise AssertionError(f"rank {rank} {draw.kind} ({recipe}): {errors}")
    mine = y.contiguous().view(torch.uint8)
    peers = [torch.empty_like(mine) for _ in range(WORLD_SIZE)]
    dist.all_gather(peers, mine)
    if not all(torch.equal(peer, peers[0]) for peer in peers):
        raise AssertionError(f"rank {rank} {draw.kind}: ranks disagree on y")


# (T, routing draws): bounded expert counts keep the FP64 references quick;
# disjoint fills every slot (64 distinct experts at T = 8).
CHECK_CASES = ((8, ("uniform",)), (8, ("disjoint",)), (16, ("same8", "shared")))


def correctness_check(group: dist.ProcessGroup) -> None:
    """Repeat and alternate draws to cycle the cross-launch state; y must be
    bitwise stable across calls."""
    device = torch.device("cuda", torch.cuda.current_device())
    for tokens, kinds in CHECK_CASES:
        case = make_case(kinds, group, tokens)
        workspaces = make_workspaces(tokens, group)
        inputs = ExpertInputs.zeros(tokens, device)
        outputs: list[list[torch.Tensor]] = [[] for _ in case.draws]
        for _ in range(3):
            for i, draw in enumerate(case.draws):
                _load(inputs, draw)
                y = launch(
                    case.bank, inputs, case.alphas, case.norm_weight, workspaces, group
                )
                outputs[i].append(y)
        for draw, ys in zip(case.draws, outputs, strict=True):
            if not all(torch.equal(y, ys[-1]) for y in ys):
                raise AssertionError(f"{draw.kind}: y differs across calls")
            _load(inputs, draw)
            y = launch(
                case.bank, inputs, case.alphas, case.norm_weight, workspaces, group
            )
            _check(case, draw, y, workspaces.gemm2_ws)
        dist.barrier()


def use_cudagraph() -> bool:
    return True


# Cached like make_workspaces.
@functools.cache
def _baseline_rows(tokens: int, group: dist.ProcessGroup) -> torch.Tensor:
    device = torch.device("cuda", torch.cuda.current_device())
    rows = symm_mem.empty(tokens * TOPK, HIDDEN, dtype=torch.bfloat16, device=device)
    symm_mem.rendezvous(rows, group.group_name)
    return rows


def _sglang_default(
    case: Case, draw: Draw, group: dist.ProcessGroup
) -> Callable[[], torch.Tensor]:
    """sglang's expert-TP sequence: FlashInfer's routed NVFP4 MoE without its
    finalize, the gather and one-shot all-reduce of the K-split routed rows
    (torch symmetric memory, standing in for sglang's custom all-reduce), and the
    CuTe DSL weight-first post-expert-norm finalize. The MoE and the finalize
    launch with PDL as in serving."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from sglang.kernels.jit.utils import is_arch_support_pdl
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_trtllm import (
        FusedPostExpertNormWeightedSumTRTLLM,
    )

    # The finalize launches with PDL exactly when this holds (one wave or less).
    assert is_arch_support_pdl()
    tokens = len(draw.packed)
    device = draw.packed.device
    weights = case.rank_weights
    up_alpha, gate_alpha, out_alpha = case.alphas
    finalize = FusedPostExpertNormWeightedSumTRTLLM(
        topk=TOPK, hidden_dim=HIDDEN, eps=NORM_EPS, weight_first=True
    )
    rows = _baseline_rows(tokens, group)
    slots = torch.arange(tokens * TOPK, dtype=torch.int32, device=device)
    y = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=device)

    def call() -> torch.Tensor:
        gemm2_out, expert_weights, expanded = trtllm_fp4_block_scale_routed_moe(
            topk_ids=draw.packed,
            routing_bias=None,
            hidden_states=draw.hidden_states,
            hidden_states_scale=draw.hidden_states_scale,
            # W13 from the bank: the stock parameters alias it, as served.
            gemm1_weights=case.bank.w13,
            gemm1_weights_scale=case.bank.w13_scale.view(weights.w13_scale.shape),
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=weights.w2,
            gemm2_weights_scale=weights.w2_scale,
            gemm2_bias=None,
            output1_scale_scalar=up_alpha,
            output1_scale_gate_scalar=gate_alpha,
            output2_scale_scalar=out_alpha,
            per_token_scale=draw.per_token_scale,
            num_experts=NUM_EXPERTS,
            top_k=TOPK,
            n_group=0,
            topk_group=0,
            intermediate_size=INTERMEDIATE // WORLD_SIZE,
            local_expert_offset=0,
            local_num_experts=NUM_EXPERTS,
            routed_scaling_factor=1.0,
            routing_method_type=1,
            do_finalize=False,
            activation_type=3,  # SwiGLU
            enable_pdl=True,
        )[:3]
        flat = expanded.reshape(-1)
        torch.index_select(gemm2_out, 0, flat.clamp(min=0), out=rows)
        reduced = torch.ops.symm_mem.one_shot_all_reduce(rows, "sum", group.group_name)
        finalize(
            gemm2_output=reduced,
            expert_weights=expert_weights,
            expanded_idx_to_permuted_idx=torch.where(flat >= 0, slots, -1),
            rms_norm_weight=case.norm_weight,
            output=y,
        )
        return y

    return call


def _run(
    group: dist.ProcessGroup, *, verbose: bool, check_only: bool
) -> dict[str, Any]:
    correctness_check(group)
    if check_only:
        return {}
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    speedups = []
    for tokens in VERIFY_TOKENS:
        # Uniform routing loads the most distinct experts per step.
        case = make_case(("uniform",), group, tokens)
        draw = case.draws[0]
        workspaces = make_workspaces(tokens, group)
        inputs = ExpertInputs.zeros(tokens, device)
        _load(inputs, draw)

        def helion_call(
            case: Case = case,
            inputs: ExpertInputs = inputs,
            workspaces: MoeWorkspaces = workspaces,
        ) -> torch.Tensor:
            return launch(
                case.bank, inputs, case.alphas, case.norm_weight, workspaces, group
            )

        def validate(
            name: str, y: object, case: Case = case, draw: Draw = draw
        ) -> None:
            # The megakernel uses the served static FC2 input scale; FlashInfer's
            # requant of z changes across versions, so it is held to accuracy only.
            _check(
                case,
                draw,
                cast("torch.Tensor", y),
                recipe="static" if name == "helion" else None,
            )

        results = benchmark(
            {
                "helion": helion_call,
                "sglang_default": _sglang_default(case, draw, group),
            },
            validate=validate,
        )
        speedup = results["sglang_default"][1] / results["helion"][1]
        speedups.append(speedup)
        if verbose and rank == 0:
            print(f"T={tokens} (uniform routing, {len(case.logical.ids)} experts)")
            for name, (warm, cold) in results.items():
                print(f"{name:>16s}: {warm:7.2f} us warm, {cold:7.2f} us cold L2")
            print(f"{'speedup':>16s}: {speedup:.3f}x (cold L2)")
        del case, workspaces
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
        kernel_name="muse_spark_moe_tp4",
        world_size=WORLD_SIZE,
        signal_pad_bytes=SIGNAL_PAD_BYTES,
    )
    try:
        return _run(group, verbose=verbose, check_only=check_only)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
