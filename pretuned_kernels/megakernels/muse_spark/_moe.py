"""Shared pieces of the Muse Spark verify MoE megakernels (TP4 and EP4).

Device helpers: the routed-expert table each task decodes its expert from, and
the FlashInfer TRT-LLM tile layouts, SwiGLU and NVFP4 z quant of R1. Host
helpers: routing draws, random checkpoint experts in FlashInfer's layout, and
the FP64 references both megakernels are checked against.
"""

from __future__ import annotations

import dataclasses
import operator

from pretuned_kernels.megakernels.muse_spark._common import EXPERT_HIDDEN
from pretuned_kernels.megakernels.muse_spark._common import NUM_EXPERTS
from pretuned_kernels.megakernels.muse_spark._common import TOPK
from pretuned_kernels.megakernels.muse_spark._common import WORLD_SIZE
from pretuned_kernels.megakernels.muse_spark._common import ExpertInputs
from pretuned_kernels.megakernels.muse_spark._common import dequantize_nvfp4
from pretuned_kernels.megakernels.muse_spark._common import encode_block_scale
from pretuned_kernels.megakernels.muse_spark._common import pack_e2m1x2
from pretuned_kernels.megakernels.muse_spark._common import relative_rmse_and_cosine
from pretuned_kernels.megakernels.muse_spark._common import round_e2m1
import torch
import torch.distributed as dist

import helion.language as hl

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


def distinct_table(
    ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Routed-expert bitmap words [8], their popcounts and exclusive prefix, and the
    distinct count; tables this narrow reduce within one warp. Ids of 256 and up
    set no bit."""

    lanes = hl.arange(8)
    row = ids.unsqueeze(0)
    bits = torch.where((row >> 5) == lanes.unsqueeze(1), 1 << (row & 31), 0)
    words = hl.reduce(operator.or_, bits, dim=1)
    counts = _popcount(words)
    before = hl.cumsum(counts, dim=0) - counts
    return words, counts, before, hl.reduce(operator.add, counts, dim=0)


def expert_of(
    words: torch.Tensor, counts: torch.Tensor, before: torch.Tensor, j: object
) -> torch.Tensor:
    """The j-th distinct expert (ascending): its bitmap word, then the bit in it."""

    hit = (before <= j) & (before + counts > j)
    word = hl.reduce(operator.or_, torch.where(hit, words, 0), dim=0)
    key = hl.reduce(operator.or_, torch.where(hit, before * 8 + hl.arange(8), 0), dim=0)
    upto = _popcount(word & ((2 << hl.arange(32)) - 1))
    bit = hl.reduce(operator.add, (upto <= j - (key >> 3)).to(torch.int32), dim=0)
    return (key & 7) * 32 + bit


def scale_tile(chunks: torch.Tensor) -> torch.Tensor:
    """[2, 16, 256] SWIZZLE_32_4_4 bytes of two 128-row tiles -> [256 rows, 32]."""

    return chunks.reshape(2, 8, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(256, 32)


def logical_rows(acc: torch.Tensor) -> torch.Tensor:
    """[256 physical rows, t] -> logical order, per 128-row FI tile."""

    t = acc.size(1)
    return acc.reshape(2, 4, 4, 8, t).permute(0, 1, 3, 2, 4).reshape(256, t)


def swiglu(
    acc: torch.Tensor,
    row_scale: torch.Tensor,
    up_alpha: torch.Tensor,
    gate_alpha: torch.Tensor,
) -> torch.Tensor:
    """Interleaved (up, gate) rows [256, t] -> SwiGLU [t, 128]."""

    t = acc.size(1)
    pairs = logical_rows(acc).reshape(128, 2, t).permute(0, 2, 1)
    up_acc, gate_acc = hl.split(pairs)
    up = up_acc * row_scale[None, :] * up_alpha
    gate = gate_acc * row_scale[None, :] * gate_alpha
    silu = gate / (1.0 + torch.exp(-gate))
    return (silu * up).permute(1, 0)


def quant_z(act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[t, 128] -> (e2m1x2 bytes [t, 64], E4M3 block scale bytes [t, 8])."""

    t = act.size(0)
    blocks = act.reshape(t, 8, 16)
    amax = torch.amax(torch.abs(blocks), dim=-1)
    fp8, mult = encode_block_scale(hl.full([t, 8], 1.0, dtype=torch.float32), amax)
    low, high = hl.split((blocks * mult[:, :, None]).reshape(t, 8, 8, 2))
    zq = pack_e2m1x2(low, high).to(torch.uint8).reshape(t, 64)
    return zq, fp8.to(torch.uint8)


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
class Problem:
    """Every draw's experts, and the FC2 input scale and post-expert norm weight
    the references share."""

    logical: LogicalExperts
    fc2_scale: float
    norm_weight: torch.Tensor  # [H] bf16
    draws: list[Draw]


def routes(kind: str, tokens: int) -> torch.Tensor:
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


def flashinfer_layout(
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


def kernel_alphas(
    logical: LogicalExperts, fc2_scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(up, gate, out) FP32 alphas [256]; as served, the static FC2 input scale is
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
    topk_ids = routes(kind, tokens)
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


def make_problem(kinds: tuple[str, ...], tokens: int) -> Problem:
    """Every draw's experts; the static FC2 input scale is calibrated on the
    routed rows of all draws, as a checkpoint's input scale would be."""
    ids = sorted(
        set().union(*(routes(kind, tokens).unique().tolist() for kind in kinds))
    )
    logical = _random_experts(ids, seed=7)
    draws = [_make_draw(kind, i, logical, tokens) for i, kind in enumerate(kinds)]
    amax = max(rows.abs().max().item() for d in draws for *_, rows in d.rows)
    g = torch.Generator(device="cuda").manual_seed(7)
    norm_weight = 1.0 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=g)
    return Problem(
        logical=logical,
        fc2_scale=amax * FC2_GLOBAL_SCALE_INV,
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


def references(
    problem: Problem, draw: Draw, *, owners: torch.Tensor | None = None
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """(routed W2 rows [T * 8, H], finalized y [T, H]) per FC2 input: the
    megakernel's static recipe, FlashInfer's full-row requant and no requant.
    ``owners`` [T, 8] gives each route's rank under an EP finalize: the requant
    modes then sum every rank's BF16 partial in FP32 in rank order, as its
    all-reduce does; no requant stays the single-rounding ground truth."""
    logical, fc2 = problem.logical, problem.fc2_scale
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
        normed = (rows.float() * problem.norm_weight.float()).double()
        rstd = torch.rsqrt(normed.pow(2).mean(-1, keepdim=True) + NORM_EPS)
        terms = weights[..., None] * normed * rstd
        if owners is None or mode == "exact":
            y = terms.sum(1).float().bfloat16()
        else:
            y = sum(
                (terms * (owners == r)[..., None]).sum(1).float().bfloat16().float()
                for r in range(WORLD_SIZE)
            ).bfloat16()
        out[mode] = (rows.view(tokens * TOPK, HIDDEN), y)
    return out


def load(inputs: ExpertInputs, draw: Draw) -> None:
    """Write one draw into the slot's first T rows, as the router does."""
    tokens = len(draw.packed)
    inputs.fp4[:tokens] = draw.hidden_states
    inputs.scales[:tokens] = draw.hidden_states_scale.view(torch.uint8)
    inputs.per_token_scale[:tokens] = draw.per_token_scale
    inputs.packed.copy_(draw.packed)


def check(
    problem: Problem,
    draw: Draw,
    y: torch.Tensor,
    gemm2: tuple[torch.Tensor, object] | None = None,
    *,
    recipe: str | None = "static",
    owners: torch.Tensor | None = None,
) -> None:
    """y, and optionally the routed rows a megakernel keeps (``gemm2``: the kept
    rows and their index into the reference rows [T * 8, H]; every rank must
    pass them, possibly empty), against the FP64
    reference of ``recipe``, and y no farther from no-requant than a full-row
    requant. Recipe None checks only the latter (an FC2 input requant this file
    does not model). Every rank must hold the same y."""
    rank = dist.get_rank()
    refs = references(problem, draw, owners=owners)
    errors = {
        "y_vs_exact": relative_rmse_and_cosine(y, refs["exact"][1])[0],
        "full_row_vs_exact": relative_rmse_and_cosine(
            refs["full_row"][1], refs["exact"][1]
        )[0],
    }
    if recipe is not None:
        errors["y"] = relative_rmse_and_cosine(y, refs[recipe][1])[0]
        if gemm2 is not None:
            # Over every rank's kept rows, so all ranks reach one verdict.
            kept, index = gemm2
            expected = refs[recipe][0][index].double()
            sums = torch.stack(
                ((kept.double() - expected).square().sum(), expected.square().sum())
            )
            dist.all_reduce(sums)
            errors["gemm2"] = float(sums[0].sqrt() / sums[1].sqrt())
    close = all(errors[k] < 1e-3 for k in ("y", "gemm2") if k in errors)
    no_worse = errors["y_vs_exact"] <= 1.05 * errors["full_row_vs_exact"]
    if not (close and no_worse):
        raise AssertionError(f"rank {rank} {draw.kind} ({recipe}): {errors}")
    mine = y.contiguous().view(torch.uint8)
    peers = [torch.empty_like(mine) for _ in range(WORLD_SIZE)]
    dist.all_gather(peers, mine)
    if not all(torch.equal(peer, peers[0]) for peer in peers):
        raise AssertionError(f"rank {rank} {draw.kind}: ranks disagree on y")
