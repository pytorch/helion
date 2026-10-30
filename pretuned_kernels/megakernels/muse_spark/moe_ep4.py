"""Muse Spark spec-decode verify expert-parallel (EP4) NVFP4 MoE as one persistent
Helion megakernel, including the post-expert norm finalize and its all-reduce, for
the T = 8 DFlash2 verify window.

Each rank holds 64 whole experts, so it reads FlashInfer TRT-LLM's own W13 and W2
(256-row tiles of 256 B K steps) and their block scales in place. Only this
rank's finalize partial crosses ranks.

Roots (j / f: this rank's routed experts in ascending order, ``slots`` bound):

    R1  W13 256-row tile, SwiGLU, NVFP4 z                         [2I/256, slots]
    R2  W2 256-row tile on all of z, scattered to route rows, and
        each row's (norm_w * x)^2 sum of the tile                  [H/256, slots]
    FIN this rank's routes normed and summed per token (symm)     [T, H/256]
    AR  every rank's partial in rank order -> replicated y        [T, H/256]

FC2 follows FlashInfer: the FC2 input scale is folded into ``up_alpha`` and
``out_alpha``, so z is quantized with a unit global scale.

Calls and graph replays must occur in identical collective order across ranks
on one ordered CUDA stream. The weights use FlashInfer's TRT-LLM layout, so this
example needs FlashInfer. ``main()`` compares against sglang's EP4 default
(FlashInfer ``trtllm_fp4_block_scale_routed_moe`` over this rank's experts and
the CuTe DSL post-expert-norm finalize fused with its all-reduce) and needs
sglang (``muse-spark`` branch); ``--check`` runs only the FP64 correctness
check::

    NVSHMEM_DISABLE_CUDA_VMM=1 PYTHONPATH=. \
      python -m torch.distributed.run --standalone --nproc-per-node=4 \
      pretuned_kernels/megakernels/muse_spark/moe_ep4.py [--check]
"""

from __future__ import annotations

import dataclasses
import functools
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
from unittest import mock

from pretuned_kernels.megakernels._distributed import benchmark
from pretuned_kernels.megakernels._distributed import initialize
from pretuned_kernels.megakernels.muse_spark._common import MOE_TOKENS
from pretuned_kernels.megakernels.muse_spark._common import NUM_EXPERTS
from pretuned_kernels.megakernels.muse_spark._common import SIGNAL_PAD_BYTES
from pretuned_kernels.megakernels.muse_spark._common import TOPK
from pretuned_kernels.megakernels.muse_spark._common import WORLD_SIZE
from pretuned_kernels.megakernels.muse_spark._common import ExpertInputs
from pretuned_kernels.megakernels.muse_spark._moe import HIDDEN
from pretuned_kernels.megakernels.muse_spark._moe import INTERMEDIATE
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

LOCAL_EXPERTS = NUM_EXPERTS // WORLD_SIZE
TOKENS = 8
# Routing draws at T = 8: bounded expert counts keep the FP64 references quick;
# disjoint fills every slot (64 distinct experts).
CHECK_CASES = (("uniform",), ("disjoint",), ("same8", "shared"))


def _local_ids(ids: torch.Tensor, first: object, count: object) -> torch.Tensor:
    """Global expert ids -> this rank's local ids; other ranks' experts -> 256,
    which the distinct table ignores."""
    return torch.where((ids >= first) & (ids < first + count), ids - first, 256)


def _muse_spark_moe_ep4_megakernel(
    w13: torch.Tensor,
    w2: torch.Tensor,
    s13: torch.Tensor,
    s2: torch.Tensor,
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
    route_idx: torch.Tensor,
    route_w: torch.Tensor,
    norm_w: torch.Tensor,
    sumsq: torch.Tensor,
    fin_partial: torch.Tensor,
    y: torch.Tensor,
    group_name: hl.ProcessGroupName,
    first_expert: hl.constexpr,
    eps: hl.constexpr,
) -> torch.Tensor:
    hl.specialize(w13.size())
    hl.specialize(w13.stride())
    hl.specialize(w2.size())
    hl.specialize(w2.stride())
    hl.specialize(s13.size())
    hl.specialize(s13.stride())
    hl.specialize(s2.size())
    hl.specialize(s2.stride())
    hl.specialize(x_q.size())
    hl.specialize(x_q.stride())
    hl.specialize(x_scale.size())
    hl.specialize(x_scale.stride())
    hl.specialize(x_row_scale.size())
    hl.specialize(packed_routing.size())
    hl.specialize(packed_routing.stride())
    hl.specialize(routes.size())
    hl.specialize(up_alpha.size())
    hl.specialize(gate_alpha.size())
    hl.specialize(out_alpha.size())
    hl.specialize(z.size())
    hl.specialize(z.stride())
    hl.specialize(z_scale.size())
    hl.specialize(z_scale.stride())
    hl.specialize(gemm2_ws.size())
    hl.specialize(gemm2_ws.stride())
    hl.specialize(route_idx.size())
    hl.specialize(route_w.size())
    hl.specialize(norm_w.size())
    hl.specialize(sumsq.size())
    hl.specialize(fin_partial.size())
    hl.specialize(y.size())

    topk = packed_routing.size(1)
    slots = routes.size(0)
    rows = x_q.size(0)  # MMA token rows, T padded to 16
    experts = up_alpha.size(0)
    k_steps = w13.size(1)
    k2_steps = w2.size(1)
    q_tiles = w13.size(0) // experts
    h_tiles = w2.size(0) // experts
    # x and its scales as one [1, rows, K] slot, laid out like a z slot.
    x3 = x_q.view(1, rows, x_q.size(1))
    xs3 = x_scale.view(1, rows, x_scale.size(1))
    # Activation block scales load as bytes and are viewed as e4m3.
    e4m3 = torch.float8_e4m3fn

    tokens = packed_routing.size(0)
    hidden = gemm2_ws.size(1)
    gemm2_flat = gemm2_ws.view(-1)
    sumsq_flat = sumsq.view(-1)
    remote_fin = torch.ops.symm_mem.get_remote_tensors(fin_partial, group_name)

    hl.pdl_wait()
    # Trigger at entry: the next kernel's CTAs land on SMs this grid frees early,
    # and its own wait still orders it after this grid.
    hl.pdl_launch_dependents()
    # R1. Stock: the FlashInfer TRT-LLM MoE's routing, GEMM1 (W13, SwiGLU) and
    # its NVFP4 quant of the intermediate, for this rank's experts.
    for w13_tile_q, w13_tile_j in hl.tile([q_tiles, slots], block_size=[1, 1]):
        lid = _local_ids(routes[:] >> 16, first_expert, experts)
        words, counts, before, distinct = distinct_table(lid)
        # Slots past this rank's distinct count are skipped; their z is never read.
        if w13_tile_j.begin < distinct:
            expert = expert_of(words, counts, before, w13_tile_j.begin)
            wt = expert * q_tiles + w13_tile_q.begin
            acc = hl.zeros([TILE_M, rows], dtype=torch.float32)
            # Staggered K order across tasks spreads FI weight rows over DRAM
            # channels. It is keyed on the expert, not its slot, so a token's sums
            # do not depend on the window's other routes.
            rot = 3 * (expert * q_tiles + w13_tile_q.begin)
            for w13_tile_k in hl.tile(k_steps, block_size=1):
                kk = (w13_tile_k.begin + rot) % k_steps
                lhs = hl.load(w13, [wt, kk, slice(None), slice(None)])
                lhs_scale = scale_tile(
                    hl.load(
                        s13, [wt, slice(None), kk * 16 + hl.arange(16), slice(None)]
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
            z[w13_tile_j, :, w13_tile_q.begin * 64 + hl.arange(64)] = zq[None, :, :]
            z_scale[w13_tile_j, :, w13_tile_q.begin * 8 + hl.arange(8)] = zsc[
                None, :, :
            ]

    # R2. Stock: the FlashInfer TRT-LLM MoE's GEMM2 (W2), for this rank's experts.
    for w2_tile_h, w2_tile_f in hl.tile([h_tiles, slots], block_size=[1, 1]):
        lid2 = _local_ids(routes[:] >> 16, first_expert, experts)
        words2, counts2, before2, distinct2 = distinct_table(lid2)
        if w2_tile_f.begin < distinct2:
            expert2 = expert_of(words2, counts2, before2, w2_tile_f.begin)
            wt2 = expert2 * h_tiles + w2_tile_h.begin
            acc2 = hl.zeros([TILE_M, rows], dtype=torch.float32)
            # Same K staggering as R1.
            rot2 = 3 * (expert2 * h_tiles + w2_tile_h.begin)
            for w2_tile_k in hl.tile(k2_steps, block_size=1):
                jj = (w2_tile_k.begin + rot2) % k2_steps
                lhs2 = hl.load(w2, [wt2, jj, slice(None), slice(None)])
                lhs2_scale = scale_tile(
                    hl.load(
                        s2,
                        [wt2, slice(None), jj * 16 + hl.arange(16), slice(None)],
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
            match = ((ids == expert2 + first_expert) & live).to(torch.int32)
            route_row = torch.sum(match * slot_id, dim=-1)
            routed = torch.sum(match, dim=-1) > 0
            cols = w2_tile_h.begin * TILE_M + hl.arange(TILE_M)
            hl.store(gemm2_ws, [route_row, cols], val, extra_mask=routed[:, None])
            # The finalize's weight-first norm statistic of these columns.
            scaled = val.to(torch.float32) * norm_w[cols].to(torch.float32)[None, :]
            hl.store(
                sumsq_flat,
                [route_row * h_tiles + w2_tile_h.begin],
                torch.sum(scaled * scaled, dim=-1),
                extra_mask=routed,
            )

    # FIN. Stock: the EP finalize (weight-first post-expert norm, weighted top-8
    # sum) over this rank's routes, once every R2 row is in.
    for fin_t, fin_c in hl.tile([tokens, hidden], block_size=[1, 256]):
        slot = fin_t.begin * topk + hl.arange(topk)
        ridx = route_idx[slot]
        live = ridx >= 0
        row = torch.where(live, ridx, 0)
        sum_sq = torch.sum(
            hl.load(
                sumsq_flat,
                [row[:, None] * h_tiles + hl.arange(h_tiles)[None, :]],
                extra_mask=live[:, None],
            ),
            dim=-1,
        )
        scale = torch.where(
            live,
            route_w[slot].to(torch.float32) * torch.rsqrt(sum_sq / hidden + eps),
            0.0,
        )
        rows2 = (
            hl.load(
                gemm2_flat,
                [row[:, None] * hidden + fin_c.index[None, :]],
                extra_mask=live[:, None],
            ).to(torch.float32)
            * norm_w[fin_c].to(torch.float32)[None, :]
        )
        fin_partial[fin_t.begin, fin_c] = torch.sum(rows2 * scale[:, None], dim=0).to(
            torch.bfloat16
        )

    # AR. Stock: the finalize's fused all-reduce, summed in rank order.
    for ar_t, ar_c in hl.tile([tokens, hidden], block_size=[1, 256]):
        total = hl.zeros([ar_t, ar_c], dtype=torch.float32)
        for peer_fin in remote_fin:
            total = total + peer_fin[ar_t, ar_c].to(torch.float32)
        y[ar_t, ar_c] = total.to(torch.bfloat16)

    return y


def _config() -> helion.Config:
    # R1/R2 K loops are warp-specialized TMA rings (TMA producer / MMA) for the
    # weight, scale and activation tiles; routing, alphas, the z and gemm2
    # stores and the finalize use pointers. The range lists are R1, its K loop,
    # R2, its K loop, FIN and AR.
    tma, ptr = "tensor_descriptor", "pointer"
    return helion.Config(
        block_sizes=[],
        cross_loop_pipeline="static",
        host_tensor_descriptors=True,
        indexing=[*[tma] * 4, *[ptr] * 6, *[tma] * 4, *[ptr] * 17],
        num_sm_multiplier=1,
        num_stages=2,
        num_warps=4,
        pid_type="persistent_blocked",
        range_num_stages=[0, 3, 0, 3, 0, 0],
        range_warp_specializes=[True, None, True, None, None, None],
    )


muse_spark_moe_ep4 = helion.kernel(
    _muse_spark_moe_ep4_megakernel,
    config=_config(),
    static_shapes=True,
    backend="triton",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)


@dataclasses.dataclass(frozen=True)
class RankWeights:
    """This rank's whole experts in FlashInfer's TRT-LLM layout, [64, ...], as
    sglang's EP4 path stores them."""

    w13: torch.Tensor  # [E / W, 2I, H / 2] u8, rows [w3; w1] in 128-row tiles
    w13_scale: torch.Tensor  # [E / W, 2I, H / 16] e4m3, interleaved
    w2: torch.Tensor  # [E / W, H, I / 2] u8
    w2_scale: torch.Tensor  # [E / W, H, I / 16] e4m3, interleaved


@dataclasses.dataclass(frozen=True)
class MoeEp4Weights:
    """The megakernel's tile views of ``RankWeights``, read in place."""

    w13: torch.Tensor  # [E / W * 2I / 256, H / 512, 256, 256] u8
    w2: torch.Tensor  # [E / W * H / 256, I / 512, 256, 256] u8
    # Block scales are contiguous [K steps * 16, 2 * 128] chunks per 256-row tile.
    s13: torch.Tensor  # [E / W * 2I / 256, 2, H / 512 * 16, 256] e4m3
    s2: torch.Tensor  # [E / W * H / 256, 2, I / 512 * 16, 256] e4m3


def moe_ep4_weights(weights: RankWeights) -> MoeEp4Weights:
    w13, w2 = weights.w13.view(torch.uint8), weights.w2.view(torch.uint8)
    experts, two_i, packed_k = w13.shape
    hidden, packed_i = w2.shape[1], w2.shape[2]
    tiles13, tiles2 = experts * two_i // TILE_M, experts * hidden // TILE_M
    k_steps, k2_steps = packed_k // TILE_K, packed_i // TILE_K
    return MoeEp4Weights(
        w13=w13.reshape(tiles13, TILE_M, k_steps, TILE_K).permute(0, 2, 1, 3),
        w2=w2.reshape(tiles2, TILE_M, k2_steps, TILE_K).permute(0, 2, 1, 3),
        s13=weights.w13_scale.view(tiles13, 2, k_steps * 16, 256),
        s2=weights.w2_scale.view(tiles2, 2, k2_steps * 16, 256),
    )


@dataclasses.dataclass(frozen=True)
class LocalRoutes:
    """The finalize's route map, which sglang's EP4 attention-boundary megakernel
    writes beside the expert inputs."""

    route_idx: torch.Tensor  # [T, 8] int32, the route's gemm2 row if local, else -1
    route_w: torch.Tensor  # [T, 8] bf16 route weights

    @staticmethod
    def zeros(tokens: int, device: torch.device) -> LocalRoutes:
        return LocalRoutes(
            route_idx=torch.zeros(tokens, TOPK, dtype=torch.int32, device=device),
            route_w=torch.zeros(tokens, TOPK, dtype=torch.bfloat16, device=device),
        )


@dataclasses.dataclass(frozen=True)
class MoeWorkspaces:
    """Per-rank buffers for one verify window size."""

    z: torch.Tensor  # [slots, 16, I / 2] u8, the routed NVFP4 activation
    z_scale: torch.Tensor  # [slots, 16, I / 16] u8, its E4M3 block scales
    gemm2_ws: torch.Tensor  # [slots, H] bf16 routed W2 rows of this rank's experts
    sumsq: torch.Tensor  # [slots, H / 256] fp32 sums of (norm_w * x)^2
    fin_partial: torch.Tensor  # [T, H] bf16 symm, this rank's finalize partial


# The symmetric partial lives for the process: torch finds a symmetric-memory
# handle by address, so a larger buffer at a freed smaller one's address gets its
# stale handle.
@functools.cache
def make_workspaces(tokens: int, group: dist.ProcessGroup) -> MoeWorkspaces:
    device = torch.device("cuda", torch.cuda.current_device())
    slots = tokens * TOPK
    u8 = {"dtype": torch.uint8, "device": device}
    fin_partial = symm_mem.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=device)
    symm_mem.rendezvous(fin_partial, group.group_name)
    return MoeWorkspaces(
        z=torch.empty(slots, MOE_TOKENS, INTERMEDIATE // 2, **u8),
        z_scale=torch.empty(slots, MOE_TOKENS, INTERMEDIATE // 16, **u8),
        gemm2_ws=torch.empty(slots, HIDDEN, dtype=torch.bfloat16, device=device),
        sumsq=torch.zeros(slots, HIDDEN // TILE_M, dtype=torch.float32, device=device),
        fin_partial=fin_partial,
    )


def launch(
    weights: MoeEp4Weights,
    inputs: ExpertInputs,
    local_routes: LocalRoutes,
    alphas: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    norm_weight: torch.Tensor,
    workspaces: MoeWorkspaces,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """The post-expert-normed route sum y [T, H], replicated on every rank."""
    tokens = inputs.packed.size(0)
    y = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=inputs.fp4.device)
    up_alpha, gate_alpha, out_alpha = alphas
    muse_spark_moe_ep4(
        weights.w13,
        weights.w2,
        weights.s13,
        weights.s2,
        inputs.fp4,
        inputs.scales,
        inputs.per_token_scale,
        inputs.packed,
        inputs.packed.view(-1),
        up_alpha,
        gate_alpha,
        out_alpha,
        workspaces.z,
        workspaces.z_scale,
        workspaces.gemm2_ws,
        local_routes.route_idx.view(-1),
        local_routes.route_w.view(-1),
        norm_weight,
        workspaces.sumsq,
        workspaces.fin_partial,
        y,
        group.group_name,
        group.rank() * LOCAL_EXPERTS,
        NORM_EPS,
    )
    return y


@dataclasses.dataclass(frozen=True)
class Case:
    """This rank's experts of every draw, and the kernel's FC scalars."""

    problem: Problem
    rank_weights: RankWeights
    weights: MoeEp4Weights
    alphas: tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # up, gate, out [E / W]


def _rank_weights(logical: LogicalExperts, rank: int) -> RankWeights:
    """This rank's slice of the experts in the served FlashInfer layout; experts no
    draw routes stay zero."""
    first = rank * LOCAL_EXPERTS
    slots = [i for i, e in enumerate(logical.ids) if first <= e < first + LOCAL_EXPERTS]
    # FlashInfer's gated GEMM takes [w3 (up); w1 (gate)].
    w13 = torch.cat(
        (logical.w13[slots, INTERMEDIATE:], logical.w13[slots, :INTERMEDIATE]), dim=1
    )
    w13_scale = torch.cat(
        (
            logical.w13_scale[slots, INTERMEDIATE:],
            logical.w13_scale[slots, :INTERMEDIATE],
        ),
        dim=1,
    )
    w13, w13_scale, w2, w2_scale = flashinfer_layout(
        w13, w13_scale, logical.w2[slots], logical.w2_scale[slots]
    )
    local = torch.tensor(
        [logical.ids[i] - first for i in slots], dtype=torch.long, device="cuda"
    )

    def scatter(values: torch.Tensor, *shape: int) -> torch.Tensor:
        full = torch.zeros((LOCAL_EXPERTS, *shape), dtype=torch.uint8, device="cuda")
        full[local] = values.view(torch.uint8).reshape(len(slots), *shape)
        return full

    e4m3 = torch.float8_e4m3fn
    return RankWeights(
        w13=scatter(w13, 2 * INTERMEDIATE, HIDDEN // 2),
        w13_scale=scatter(w13_scale, 2 * INTERMEDIATE, HIDDEN // 16).view(e4m3),
        w2=scatter(w2, HIDDEN, INTERMEDIATE // 2),
        w2_scale=scatter(w2_scale, HIDDEN, INTERMEDIATE // 16).view(e4m3),
    )


def make_case(kinds: tuple[str, ...], group: dist.ProcessGroup, tokens: int) -> Case:
    problem = make_problem(kinds, tokens)
    rank_weights = _rank_weights(problem.logical, group.rank())
    first = group.rank() * LOCAL_EXPERTS
    alphas = kernel_alphas(problem.logical, problem.fc2_scale)
    return Case(
        problem=problem,
        rank_weights=rank_weights,
        weights=moe_ep4_weights(rank_weights),
        alphas=cast(
            "tuple[torch.Tensor, torch.Tensor, torch.Tensor]",
            tuple(a[first : first + LOCAL_EXPERTS] for a in alphas),
        ),
    )


def _load(
    inputs: ExpertInputs, local_routes: LocalRoutes, draw: Draw, rank: int
) -> None:
    """Write one draw into the expert inputs and the route map, as the boundary
    megakernel does."""
    load(inputs, draw)
    tokens = len(draw.packed)
    slots = torch.arange(tokens * TOPK, device=draw.packed.device).view(tokens, TOPK)
    local = (draw.packed >> 16) // LOCAL_EXPERTS == rank
    local_routes.route_idx.copy_(torch.where(local, slots, -1))
    local_routes.route_w.copy_(
        (draw.packed & 0xFFFF).to(torch.int16).view(torch.bfloat16)
    )


def _check(
    case: Case,
    draw: Draw,
    y: torch.Tensor,
    gemm2: torch.Tensor | None = None,
    *,
    recipe: str | None = "static",
) -> None:
    """y and the routed rows of this rank's experts against the references,
    which model the finalize's per-rank BF16 partials."""
    owners = (draw.packed >> 16) // LOCAL_EXPERTS
    local = (owners == dist.get_rank()).view(-1)
    check(
        case.problem,
        draw,
        y,
        None if gemm2 is None else (gemm2[local], local),
        recipe=recipe,
        owners=owners,
    )


def correctness_check(group: dist.ProcessGroup) -> None:
    """Repeat and alternate draws to cycle the cross-launch state; y must be
    bitwise stable across calls."""
    rank = group.rank()
    device = torch.device("cuda", torch.cuda.current_device())
    for kinds in CHECK_CASES:
        case = make_case(kinds, group, TOKENS)
        workspaces = make_workspaces(TOKENS, group)
        inputs = ExpertInputs.zeros(TOKENS, device)
        local_routes = LocalRoutes.zeros(TOKENS, device)

        def call(
            case: Case = case,
            inputs: ExpertInputs = inputs,
            local_routes: LocalRoutes = local_routes,
            workspaces: MoeWorkspaces = workspaces,
        ) -> torch.Tensor:
            return launch(
                case.weights,
                inputs,
                local_routes,
                case.alphas,
                case.problem.norm_weight,
                workspaces,
                group,
            )

        outputs: list[list[torch.Tensor]] = [[] for _ in case.problem.draws]
        for _ in range(3):
            for i, draw in enumerate(case.problem.draws):
                _load(inputs, local_routes, draw, rank)
                outputs[i].append(call())
        for draw, ys in zip(case.problem.draws, outputs, strict=True):
            if not all(torch.equal(y, ys[-1]) for y in ys):
                raise AssertionError(f"{draw.kind}: y differs across calls")
            _load(inputs, local_routes, draw, rank)
            _check(case, draw, call(), workspaces.gemm2_ws)
        dist.barrier()


def use_cudagraph() -> bool:
    return True


# Built once like make_workspaces: the custom all-reduce's push buffers are
# symmetric allocations.
@functools.cache
def _finalize_ar_workspace(group: dist.ProcessGroup) -> object:
    """The custom all-reduce v2 push workspace the fused finalize all-reduces
    through, on sglang's world group (its node and P2P checks read it), as
    sglang's TP group builds it."""
    from sglang.kernels.ops.muse_spark_v12.fi_trtllm_moe_finalize_with_rmsnorm_cutedsl import (
        FinalizePushWorkspace,
    )
    from sglang.srt.distributed import parallel_state
    from sglang.srt.distributed.device_communicators import custom_all_reduce_v2
    from sglang.srt.runtime_context import get_parallel

    group_name = group.group_name

    def allocate(
        nbytes: int, device: torch.device, group: dist.ProcessGroup
    ) -> tuple[torch.Tensor, object]:
        # Torch symmetric memory is NVSHMEM-backed here (initialize() selects it
        # for the megakernels), and NVSHMEM allocations name no group.
        tensor = symm_mem.empty(nbytes, dtype=torch.uint8, device=device)
        return tensor, symm_mem.rendezvous(tensor, group_name)

    device = torch.device("cuda", torch.cuda.current_device())
    world = parallel_state.init_world_group(
        ranks=list(range(WORLD_SIZE)), local_rank=device.index, backend="nccl"
    )
    get_parallel().override_permanently(world_group=world)
    with mock.patch.object(
        custom_all_reduce_v2, "_allocate_symmetric_memory", allocate
    ):
        communicator = custom_all_reduce_v2.CustomAllReduceV2(world.cpu_group, device)
    assert not communicator.disabled
    return FinalizePushWorkspace(communicator, group, hidden_dim=HIDDEN)


def _sglang_default(
    case: Case, draw: Draw, group: dist.ProcessGroup
) -> Callable[[], torch.Tensor]:
    """sglang's EP4 sequence: FlashInfer's routed NVFP4 MoE over this rank's
    experts without its finalize, then the CuTe DSL weight-first post-expert-norm
    finalize fused with its all-reduce (custom all-reduce v2 push buffers). Both
    launch with PDL as in serving."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from sglang.kernels.jit.utils import is_arch_support_pdl
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_trtllm import (
        FusedPostExpertNormWeightedSumTRTLLM,
    )

    # The finalize launches with PDL exactly when this holds (one wave or less).
    assert is_arch_support_pdl()
    tokens = len(draw.packed)
    weights = case.rank_weights
    up_alpha, gate_alpha, out_alpha = case.alphas
    finalize = FusedPostExpertNormWeightedSumTRTLLM(
        topk=TOPK, hidden_dim=HIDDEN, eps=NORM_EPS, weight_first=True
    )
    ar_workspace = _finalize_ar_workspace(group)
    y = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=draw.packed.device)

    def call() -> torch.Tensor:
        gemm2_out, expert_weights, expanded = trtllm_fp4_block_scale_routed_moe(
            topk_ids=draw.packed,
            routing_bias=None,
            hidden_states=draw.hidden_states,
            hidden_states_scale=draw.hidden_states_scale,
            gemm1_weights=weights.w13,
            gemm1_weights_scale=weights.w13_scale,
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
            intermediate_size=INTERMEDIATE,
            local_expert_offset=group.rank() * LOCAL_EXPERTS,
            local_num_experts=LOCAL_EXPERTS,
            routed_scaling_factor=1.0,
            routing_method_type=1,
            do_finalize=False,
            activation_type=3,  # SwiGLU
            enable_pdl=True,
        )[:3]
        finalize(
            gemm2_output=gemm2_out,
            expert_weights=expert_weights,
            expanded_idx_to_permuted_idx=expanded,
            rms_norm_weight=case.problem.norm_weight,
            output=y,
            ar_workspace=ar_workspace,
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
    # Uniform routing loads the most distinct experts per step.
    case = make_case(("uniform",), group, TOKENS)
    draw = case.problem.draws[0]
    workspaces = make_workspaces(TOKENS, group)
    inputs = ExpertInputs.zeros(TOKENS, device)
    local_routes = LocalRoutes.zeros(TOKENS, device)
    _load(inputs, local_routes, draw, rank)

    def helion_call() -> torch.Tensor:
        return launch(
            case.weights,
            inputs,
            local_routes,
            case.alphas,
            case.problem.norm_weight,
            workspaces,
            group,
        )

    def validate(name: str, y: object) -> None:
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
    if verbose and rank == 0:
        experts = len(case.problem.logical.ids)
        print(f"T={TOKENS} (uniform routing, {experts} experts)")
        for name, (warm, cold) in results.items():
            print(f"{name:>16s}: {warm:7.2f} us warm, {cold:7.2f} us cold L2")
        print(f"{'speedup':>16s}: {speedup:.3f}x (cold L2)")
    return {
        "helion_wins": int(speedup > 1),
        "total": 1,
        "geomean": speedup,
        "best_speedup": speedup,
    }


@torch.inference_mode()
def main(verbose: bool = True, check_only: bool = False) -> dict[str, Any]:
    """Run the EP4 FP64 check, then cold-L2 timings against the sglang default."""
    _rank, _local_rank, group = initialize(
        kernel_name="muse_spark_moe_ep4",
        world_size=WORLD_SIZE,
        signal_pad_bytes=SIGNAL_PAD_BYTES,
    )
    try:
        return _run(group, verbose=verbose, check_only=check_only)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
