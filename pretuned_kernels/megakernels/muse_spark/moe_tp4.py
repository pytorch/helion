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
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from pretuned_kernels.megakernels._distributed import benchmark
from pretuned_kernels.megakernels._distributed import initialize
from pretuned_kernels.megakernels.muse_spark._common import MOE_TOKENS
from pretuned_kernels.megakernels.muse_spark._common import NUM_EXPERTS
from pretuned_kernels.megakernels.muse_spark._common import SIGNAL_PAD_BYTES
from pretuned_kernels.megakernels.muse_spark._common import TOPK
from pretuned_kernels.megakernels.muse_spark._common import VERIFY_TOKENS
from pretuned_kernels.megakernels.muse_spark._common import WORLD_SIZE
from pretuned_kernels.megakernels.muse_spark._common import ExpertInputs
from pretuned_kernels.megakernels.muse_spark._moe import HIDDEN
from pretuned_kernels.megakernels.muse_spark._moe import INTERMEDIATE
from pretuned_kernels.megakernels.muse_spark._moe import MMA_M
from pretuned_kernels.megakernels.muse_spark._moe import NORM_EPS
from pretuned_kernels.megakernels.muse_spark._moe import TILE_K
from pretuned_kernels.megakernels.muse_spark._moe import TILE_M
from pretuned_kernels.megakernels.muse_spark._moe import Draw
from pretuned_kernels.megakernels.muse_spark._moe import LogicalExperts
from pretuned_kernels.megakernels.muse_spark._moe import Problem
from pretuned_kernels.megakernels.muse_spark._moe import check
from pretuned_kernels.megakernels.muse_spark._moe import distinct_table
from pretuned_kernels.megakernels.muse_spark._moe import expert_of
from pretuned_kernels.megakernels.muse_spark._moe import flashinfer_layout
from pretuned_kernels.megakernels.muse_spark._moe import kernel_alphas
from pretuned_kernels.megakernels.muse_spark._moe import load
from pretuned_kernels.megakernels.muse_spark._moe import logical_rows
from pretuned_kernels.megakernels.muse_spark._moe import make_problem
from pretuned_kernels.megakernels.muse_spark._moe import quant_z
from pretuned_kernels.megakernels.muse_spark._moe import scale_tile
from pretuned_kernels.megakernels.muse_spark._moe import swiglu
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


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
        words, counts, before, distinct = distinct_table(routes[:] >> 16)
        # Slots past the distinct count are skipped; their z is never read.
        if w13_tile_j.begin < distinct:
            expert = expert_of(words, counts, before, w13_tile_j.begin)
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
                lhs_scale = scale_tile(
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
            act = swiglu(
                acc,
                x_row_scale[hl.arange(rows)],
                up_alpha[expert],
                gate_alpha[expert],
            )
            zq, zsc = quant_z(act)
            col = (rank * q_tiles + w13_tile_q.begin) * 64
            scol = (rank * q_tiles + w13_tile_q.begin) * 8
            for peer_z in remote_z:
                peer_z[w13_tile_j, :, col + hl.arange(64)] = zq[None, :, :]
            for peer_zs in remote_z_scale:
                peer_zs[w13_tile_j, :, scol + hl.arange(8)] = zsc[None, :, :]

    # R2. Stock: the FlashInfer TRT-LLM MoE's GEMM2 (W2).
    for w2_tile_h, w2_tile_f in hl.tile([h_tiles, slots], block_size=[1, 1]):
        words2, counts2, before2, distinct2 = distinct_table(routes[:] >> 16)
        if w2_tile_f.begin < distinct2:
            expert2 = expert_of(words2, counts2, before2, w2_tile_f.begin)
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
                lhs2_scale = scale_tile(
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
                (logical_rows(acc2) * out_alpha[expert2])
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


@dataclasses.dataclass(frozen=True)
class RankWeights:
    """This rank's expert-TP slices in FlashInfer's TRT-LLM layout, [256, ...]."""

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor


@dataclasses.dataclass(frozen=True)
class Case:
    """One bank holding every draw's experts, and the kernel's FC scalars."""

    problem: Problem
    rank_weights: RankWeights
    bank: MoeTp4VerifyBank
    alphas: tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # up, gate, out [E]


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
    w13, w13_scale, w2, w2_scale = flashinfer_layout(
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


def make_case(kinds: tuple[str, ...], group: dist.ProcessGroup, tokens: int) -> Case:
    """Every draw's experts in one bank."""
    problem = make_problem(kinds, tokens)
    rank_weights = _rank_weights(problem.logical, group.rank())
    bank = build_moe_tp4_verify_bank(
        w13=rank_weights.w13,
        w13_scale=rank_weights.w13_scale,
        w2=rank_weights.w2,
        w2_scale=rank_weights.w2_scale,
        group=group,
    )
    return Case(
        problem=problem,
        rank_weights=rank_weights,
        bank=bank,
        alphas=kernel_alphas(problem.logical, problem.fc2_scale),
    )


def _check(
    case: Case,
    draw: Draw,
    y: torch.Tensor,
    gemm2: torch.Tensor | None = None,
    *,
    recipe: str | None = "static",
) -> None:
    """y and this rank's H/W quarter of the routed rows against the references."""
    rank = dist.get_rank()
    columns = slice(rank * HIDDEN // WORLD_SIZE, (rank + 1) * HIDDEN // WORLD_SIZE)
    check(
        case.problem,
        draw,
        y,
        None if gemm2 is None else (gemm2, (slice(None), columns)),
        recipe=recipe,
    )


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
        outputs: list[list[torch.Tensor]] = [[] for _ in case.problem.draws]
        for _ in range(3):
            for i, draw in enumerate(case.problem.draws):
                load(inputs, draw)
                y = launch(
                    case.bank,
                    inputs,
                    case.alphas,
                    case.problem.norm_weight,
                    workspaces,
                    group,
                )
                outputs[i].append(y)
        for draw, ys in zip(case.problem.draws, outputs, strict=True):
            if not all(torch.equal(y, ys[-1]) for y in ys):
                raise AssertionError(f"{draw.kind}: y differs across calls")
            load(inputs, draw)
            y = launch(
                case.bank,
                inputs,
                case.alphas,
                case.problem.norm_weight,
                workspaces,
                group,
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
            rms_norm_weight=case.problem.norm_weight,
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
        draw = case.problem.draws[0]
        workspaces = make_workspaces(tokens, group)
        inputs = ExpertInputs.zeros(tokens, device)
        load(inputs, draw)

        def helion_call(
            case: Case = case,
            inputs: ExpertInputs = inputs,
            workspaces: MoeWorkspaces = workspaces,
        ) -> torch.Tensor:
            return launch(
                case.bank,
                inputs,
                case.alphas,
                case.problem.norm_weight,
                workspaces,
                group,
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
            print(
                f"T={tokens} (uniform routing, {len(case.problem.logical.ids)} experts)"
            )
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
