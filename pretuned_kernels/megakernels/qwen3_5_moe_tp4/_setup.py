# ruff: noqa: ANN001, ANN002, ANN202, C408
"""Synthetic Qwen3.5 TP4 MoE weights in FlashInfer's trtllm-gen layout, and a
torch reference on the logical (unshuffled) weights."""

from __future__ import annotations

import dataclasses

from pretuned_kernels.megakernels.gpt_oss_moe._standalone import _swizzle_scales
import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@dataclasses.dataclass(frozen=True)
class Shape:
    hidden: int = 4096
    intermediate: int = 256  # per rank (TP4)
    num_experts: int = 512
    top_k: int = 10
    shared_intermediate: int = 256  # per rank


SHAPE = Shape()


def e2m1_decode(packed: torch.Tensor) -> torch.Tensor:
    """[..., K/2] uint8 -> [..., K] float32 (low nibble first)."""
    table = torch.tensor(
        E2M1 + tuple(-v for v in E2M1), device=packed.device, dtype=torch.float32
    )
    low = table[(packed & 15).long()]
    high = table[(packed >> 4).long()]
    return torch.stack((low, high), dim=-1).flatten(-2)


def swizzle_scales(scale: torch.Tensor) -> torch.Tensor:
    """[..., rows, groups] E4M3 -> the 128x4-interleaved bytes (same shape): the
    TRT-LLM scale layout of ``gpt_oss_moe``."""
    return _swizzle_scales(scale.view(torch.uint8)).view(torch.float8_e4m3fn)


def _row_maps(device, intermediate, hidden):
    """trtllm-gen physical row -> logical row: W13 (in [up; gate] order, after the
    gated-act interleave and the epilogue-tile shuffle) and W2 (hidden): FlashInfer's
    ``get_reorder_rows_for_gated_act_gemm_row_indices`` indexed by
    ``get_shuffle_matrix_a_row_indices`` (epilogue tile 128), and the latter alone."""
    p = torch.arange(2 * intermediate)
    b, r = p // 32, p % 32
    j = 16 * b + 2 * (r % 8) + (r // 16)
    phys13 = torch.where((r // 8) % 2 == 0, j, intermediate + j)
    q = torch.arange(hidden)
    phys2 = (q // 32) * 32 + (q % 8) * 4 + (q % 32) // 8
    return phys13.to(device), phys2.to(device)


def make_weights(shape: Shape = SHAPE, seed: int = 0, device="cuda") -> dict:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    E, H, N, SHI = (
        shape.num_experts,
        shape.hidden,
        shape.intermediate,
        shape.shared_intermediate,
    )

    def rand_bytes(*size):
        return torch.randint(
            0, 256, size, generator=g, device=device, dtype=torch.uint8
        )

    def rand_scales(*size):
        # E4M3 block scales around 2^-3 .. 2^-1
        v = torch.exp2(torch.randint(-3, 0, size, generator=g, device=device).float())
        v = v * (1 + torch.randint(0, 8, size, generator=g, device=device).float() / 8)
        return v.to(torch.float8_e4m3fn)

    # Logical ModelOpt layout: w13 rows [gate (N); up (N)].
    w13_log = rand_bytes(E, 2 * N, H // 2)
    w13s_log = rand_scales(E, 2 * N, H // 16)
    w2_log = rand_bytes(E, H, N // 2)
    w2s_log = rand_scales(E, H, N // 16)
    # trtllm-gen: [up; gate], gated-act reorder, epilogue-tile shuffle.
    phys13, phys2 = _row_maps(device, N, H)
    upgate = torch.cat((w13_log[:, N:], w13_log[:, :N]), dim=1)
    upgate_s = torch.cat((w13s_log[:, N:], w13s_log[:, :N]), dim=1)
    w13 = upgate[:, phys13].contiguous()
    w13s = swizzle_scales(upgate_s[:, phys13].contiguous())
    w2 = w2_log[:, phys2].contiguous()
    w2s = swizzle_scales(w2s_log[:, phys2].contiguous())

    # alpha = 1 / (input global scale * weight global scale). The weights'
    # global scales (64 for W13, 16 for W2) keep the SwiGLU inputs and the routed
    # output of order 1, and the activation's global scale keeps its 16-blocks'
    # E4M3 scales (block amax * 64 / 6, up to about 250 here) below 448.
    input_global = torch.tensor([64.0], device=device)
    act_global = torch.tensor([64.0], device=device)
    alpha1 = (torch.rand(E, generator=g, device=device) * 0.5 + 0.75) / (
        input_global.item() * 64
    )
    alpha2 = (torch.rand(E, generator=g, device=device) * 0.5 + 0.75) / (
        act_global.item() * 16
    )

    router = (torch.randn(E, H, generator=g, device=device) * 0.02).to(torch.bfloat16)
    sw13 = (torch.randn(2 * SHI, H, generator=g, device=device) * 0.5).to(
        torch.float8_e4m3fn
    )
    sw2 = (torch.randn(H, SHI, generator=g, device=device) * 0.5).to(
        torch.float8_e4m3fn
    )
    # 1/in1, in1*w1, 1/in2, in2*w2: the shared output of order 1
    shared_scales = torch.tensor(
        [1 / 0.02, 0.02 * 0.05, 1 / 0.1, 0.1 * 0.1], device=device
    )
    gate_w = (torch.randn(H, generator=g, device=device) * 0.02).to(torch.bfloat16)
    return dict(
        w13_log=w13_log,
        w13s_log=w13s_log,
        w2_log=w2_log,
        w2s_log=w2s_log,
        w13=w13,
        w13s=w13s,
        w2=w2,
        w2s=w2s,
        alpha1=alpha1,
        alpha2=alpha2,
        input_global=input_global,
        act_global=act_global,
        router=router,
        sw13=sw13,
        sw2=sw2,
        shared_scales=shared_scales,
        gate_w=gate_w,
    )


def kernel_weight_args(w: dict, shape: Shape = SHAPE) -> tuple:
    E, H, N = shape.num_experts, shape.hidden, shape.intermediate
    return (
        w["router"],
        w["w13"],
        w["w13s"].view(E, (2 * N // 128) * (H // 512), 16, 256),
        w["w2"],
        w["w2s"].view(E, H // 128, 8, 256),
        w["alpha1"],
        w["alpha2"],
        w["input_global"],
        w["act_global"],
        w["sw13"],
        w["sw2"],
        w["shared_scales"],
        w["gate_w"],
        shape.top_k,
    )


def benchmark_weights(
    layer: int, rank: int, device, router_support: int = 0, shape: Shape = SHAPE
) -> dict:
    """Layer ``layer``'s weights on ``rank`` for the timings, without the logical
    copies: the expert shards per rank, the router and the shared expert's gate
    seeded by the layer alone (TP replicates them: every rank routes alike).
    ``router_support`` = S > 0 scales the router rows outside a layer-seeded set
    of S experts by 0.01 (about S distinct experts a layer)."""
    w = make_weights(shape, seed=1000 * layer + rank, device=device)
    for key in ("w13_log", "w13s_log", "w2_log", "w2s_log"):
        del w[key]
    g = torch.Generator(device=device)
    g.manual_seed(1000 * layer + 777)
    w["router"] = (
        torch.randn(shape.num_experts, shape.hidden, generator=g, device=device) * 0.02
    ).to(torch.bfloat16)
    if router_support:
        gs = torch.Generator(device=device)
        gs.manual_seed(1000 * layer + 999)
        off = torch.randperm(shape.num_experts, generator=gs, device=device)[
            router_support:
        ]
        w["router"][off] = (w["router"][off].float() * 0.01).to(torch.bfloat16)
    w["gate_w"] = (torch.randn(shape.hidden, generator=g, device=device) * 0.02).to(
        torch.bfloat16
    )
    return w


def _nvfp4_quant(x: torch.Tensor, global_scale: float, ties: torch.Tensor):
    """The NVFP4 copy of [..., K] float32 ``x * global_scale`` (E2M1 values times
    their 16-group E4M3 scales), FlashInfer rounding. A value exactly on a
    rounding boundary (about 1% of BF16 inputs) takes its value in ``ties``, the
    kernel's copy: its division may round the tie either way."""
    *b, k = x.shape
    grp = x.reshape(*b, k // 16, 16)
    bs = (grp.abs().amax(-1) * global_scale / 6.0).to(torch.float8_e4m3fn).float()
    div = torch.where(bs > 0, bs, torch.ones_like(bs))
    v = grp * global_scale / div[..., None]
    m = v.abs()
    code = (
        (m > 0.25).int()
        + (m >= 0.75).int()
        + (m > 1.25).int()
        + (m >= 1.75).int()
        + (m > 2.5).int()
        + (m >= 3.5).int()
        + (m > 5.0).int()
    )
    tie = sum(m == bound for bound in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0))
    table = torch.tensor(E2M1, device=x.device)
    q = table[code.long()] * torch.where(v < 0, -1.0, 1.0)
    out = (q * bs[..., None]).reshape(*b, k)
    return torch.where(tie.bool().reshape(*b, k), ties, out)


def reference_partial(x: torch.Tensor, w: dict, kernel_quant: tuple, shape=SHAPE):
    """This rank's MoE output (BF16 [T, H]) and top-k ids on the logical weights,
    the kernel's NVFP4 input and activations with their exact quantizations
    (label: (kernel's, exact), dequantized), and the stages before the combine:
    each pick's W2 row ([T, K, H], BF16, in the order of the returned ids) and
    the gated shared output ([T, H], BF16).

    ``kernel_quant``: the kernel's NVFP4 input [T, H / 2] and its scales, and its
    activation rows [distinct, T, N / 2] and their scales (row j: the window's
    j-th distinct expert, ascending). Each GEMM reads the kernel's copy of its
    input (W13 the NVFP4 input, W2 the activations), so a stage's error shows in
    that stage; a rounding tie resolves as the kernel's division does."""
    H, N, K = shape.hidden, shape.intermediate, shape.top_k
    T = x.shape[0]
    xf = x.float()
    logits = (xf @ w["router"].float().T).to(torch.bfloat16).float()
    # stable top-k: higher value, then lower index
    vals, ids = torch.sort(logits, dim=-1, descending=True, stable=True)
    ids = ids[:, :K]
    wts = torch.softmax(vals[:, :K], dim=-1)
    input_q, input_scale, act_q, act_scale = kernel_quant
    xq = e2m1_decode(input_q) * input_scale.float().repeat_interleave(16, 1)
    act_global = w["act_global"].item()
    distinct = sorted({int(e) for e in ids.flatten()})
    acts, exacts = [], []
    routed = torch.zeros(T, H, device=x.device)
    pick_rows = torch.zeros(T, K, H, device=x.device)
    for t in range(T):
        for k in range(K):
            e = int(ids[t, k])
            w13 = e2m1_decode(w["w13_log"][e]) * w["w13s_log"][
                e
            ].float().repeat_interleave(16, 1)
            h = (w13 @ xq[t]) * w["alpha1"][e]
            gate, up = h[:N], h[N:]
            act = gate * torch.sigmoid(gate) * up
            j = distinct.index(e)
            aq = e2m1_decode(act_q[j, t]) * act_scale[j, t].float().repeat_interleave(
                16
            )
            acts.append(aq)
            exacts.append(_nvfp4_quant(act, act_global, aq))
            w2 = e2m1_decode(w["w2_log"][e]) * w["w2s_log"][
                e
            ].float().repeat_interleave(16, 1)
            y = (w2 @ aq) * w["alpha2"][e]
            pick_rows[t, k] = y.to(torch.bfloat16).float()
            routed[t] += (
                wts[t, k].to(torch.bfloat16).float() * y.to(torch.bfloat16).float()
            )
    s = w["shared_scales"]
    x8 = torch.clamp(xf * s[0], -448, 448).to(torch.float8_e4m3fn).float()
    gu = (x8 @ w["sw13"].float().T) * s[1]
    g, u = gu[:, : shape.shared_intermediate], gu[:, shape.shared_intermediate :]
    a = (g * torch.sigmoid(g) * u).to(torch.bfloat16).float()
    a8 = torch.clamp(a * s[2], -448, 448).to(torch.float8_e4m3fn).float()
    sh = ((a8 @ w["sw2"].float().T) * s[3]).to(torch.bfloat16).float()
    gate = torch.sigmoid(xf @ w["gate_w"].float())
    sh = (sh * gate[:, None]).to(torch.bfloat16).float()
    quant = {
        "input (NVFP4)": (xq, _nvfp4_quant(xf, w["input_global"].item(), xq)),
        "activation (NVFP4)": (torch.stack(acts), torch.stack(exacts)),
    }
    stages = {"routed rows": pick_rows, "shared output": sh}
    return (routed + sh).to(torch.bfloat16), ids.int(), quant, stages
