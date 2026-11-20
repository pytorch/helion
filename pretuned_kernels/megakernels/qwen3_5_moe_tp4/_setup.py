# ruff: noqa: ANN001, ANN002, ANN202, C408
"""Synthetic Qwen3.5 TP4 MoE weights in FlashInfer's trtllm-gen layout, and a
torch reference on the logical (unshuffled) weights."""

from __future__ import annotations

import dataclasses

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
    """[..., rows, groups] E4M3 -> the 128x4-interleaved bytes (same shape)."""
    *batch, rows, groups = scale.shape
    n = len(batch)
    tiles = scale.view(torch.uint8).reshape(*batch, rows // 128, 4, 32, groups // 4, 4)
    # (tile, j, i, kb, k) -> (tile, kb, i, j, k)
    order = [*range(n), n, n + 3, n + 2, n + 1, n + 4]
    return (
        tiles.permute(order)
        .reshape(*batch, rows, groups)
        .contiguous()
        .view(torch.float8_e4m3fn)
    )


def _row_maps(device, intermediate=256, hidden=4096):
    """trtllm-gen physical row -> logical row: W13 (in [up; gate] order, after the
    gated-act interleave and the epilogue-tile shuffle) and W2 (hidden)."""
    p = torch.arange(2 * intermediate)
    b, r = p // 32, p % 32
    j = 16 * b + 2 * (r % 8) + (r // 16)
    phys13 = torch.where((r // 8) % 2 == 0, j, intermediate + j)
    q = torch.arange(hidden)
    phys2 = (q // 32) * 32 + (q % 8) * 4 + (q % 32) // 8
    return phys13.to(device), phys2.to(device)


def _check_row_maps():
    """The analytic maps equal FlashInfer's (run in a separate process)."""
    from flashinfer.fused_moe.core import (
        get_reorder_rows_for_gated_act_gemm_row_indices,
    )
    from flashinfer.utils import get_shuffle_matrix_a_row_indices

    w13 = torch.zeros(512, 2048, dtype=torch.uint8)
    w2 = torch.zeros(4096, 128, dtype=torch.uint8)
    reorder = get_reorder_rows_for_gated_act_gemm_row_indices(w13)
    fi13 = reorder[get_shuffle_matrix_a_row_indices(w13, 128)]
    fi2 = get_shuffle_matrix_a_row_indices(w2, 128)
    ours13, ours2 = _row_maps("cpu")
    assert torch.equal(fi13, ours13) and torch.equal(fi2, ours2)


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
    phys13, phys2 = _row_maps(device)
    upgate = torch.cat((w13_log[:, N:], w13_log[:, :N]), dim=1)
    upgate_s = torch.cat((w13s_log[:, N:], w13s_log[:, :N]), dim=1)
    w13 = upgate[:, phys13].contiguous()
    w13s = swizzle_scales(upgate_s[:, phys13].contiguous())
    w2 = w2_log[:, phys2].contiguous()
    w2s = swizzle_scales(w2s_log[:, phys2].contiguous())

    input_global = torch.tensor([64.0], device=device)
    act_global = torch.tensor([32.0], device=device)
    alpha1 = (
        torch.rand(E, generator=g, device=device) * 0.5 + 0.75
    ) / input_global.item()
    alpha2 = (
        torch.rand(E, generator=g, device=device) * 0.5 + 0.75
    ) / act_global.item()

    router = (torch.randn(E, H, generator=g, device=device) * 0.02).to(torch.bfloat16)
    sw13 = (torch.randn(2 * SHI, H, generator=g, device=device) * 0.5).to(
        torch.float8_e4m3fn
    )
    sw2 = (torch.randn(H, SHI, generator=g, device=device) * 0.5).to(
        torch.float8_e4m3fn
    )
    # 1/in1, in1*w1, 1/in2, in2*w2
    shared_scales = torch.tensor(
        [1 / 0.02, 0.02 * 0.01, 1 / 0.05, 0.05 * 0.01], device=device
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


def empty_weight_args(device, shape: Shape = SHAPE) -> tuple:
    """kernel_weight_args' shapes on uninitialized tensors (codegen / compile only)."""
    E, H, N = shape.num_experts, shape.hidden, shape.intermediate
    SHI = shape.shared_intermediate

    def empty(*size, dtype):
        return torch.empty(size, dtype=dtype, device=device)

    u8, e4, f32, bf = torch.uint8, torch.float8_e4m3fn, torch.float32, torch.bfloat16
    return (
        empty(E, H, dtype=bf),
        empty(E, 2 * N, H // 2, dtype=u8),
        empty(E, (2 * N // 128) * (H // 512), 16, 256, dtype=e4),
        empty(E, H, N // 2, dtype=u8),
        empty(E, H // 128, 8, 256, dtype=e4),
        empty(E, dtype=f32),
        empty(E, dtype=f32),
        empty(1, dtype=f32),
        empty(1, dtype=f32),
        empty(2 * SHI, H, dtype=e4),
        empty(H, SHI, dtype=e4),
        empty(4, dtype=f32),
        empty(H, dtype=bf),
        shape.top_k,
    )


def _nvfp4_quant(x: torch.Tensor, global_scale: float) -> torch.Tensor:
    """Quantize-dequantize [..., K] float32 by 16-groups, FlashInfer rounding."""
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
    table = torch.tensor(E2M1, device=x.device)
    q = table[code.long()] * torch.where(v < 0, -1.0, 1.0)
    return (q * bs[..., None] / global_scale).reshape(*b, k)


def reference_partial(x: torch.Tensor, w: dict, shape: Shape = SHAPE, quant=None):
    """This rank's MoE output (bf16 [T, H]) and the routing, on logical weights."""
    H, N, K = shape.hidden, shape.intermediate, shape.top_k
    T = x.shape[0]
    xf = x.float()
    logits = (xf @ w["router"].float().T).to(torch.bfloat16).float()
    # stable top-k: higher value, then lower index
    vals, ids = torch.sort(logits, dim=-1, descending=True, stable=True)
    ids = ids[:, :K]
    top = vals[:, :K]
    wts = torch.softmax(top, dim=-1)
    if quant is not None:
        # The kernel's own NVFP4 copies (ties round by its division): input
        # [T, H/2] + scales, activation per slot [S, T, N/2] + scales.
        iq, isc, aqk, asck = quant
        xq = e2m1_decode(iq) * isc.float().repeat_interleave(16, 1)
        distinct = sorted({int(e) for e in ids.flatten()})
    else:
        xq = _nvfp4_quant(xf, w["input_global"].item()) * w["input_global"].item()
    routed = torch.zeros(T, H, device=x.device)
    rows_out = torch.zeros(T, K, H, device=x.device, dtype=torch.bfloat16)
    for t in range(T):
        for k in range(K):
            e = int(ids[t, k])
            w13 = e2m1_decode(w["w13_log"][e]) * w["w13s_log"][
                e
            ].float().repeat_interleave(16, 1)
            h = (w13 @ xq[t]) * w["alpha1"][e]
            gate, up = h[:N], h[N:]
            act = gate * torch.sigmoid(gate) * up
            if quant is not None and aqk is not None:
                j = distinct.index(e)
                aq_ = e2m1_decode(aqk[j, t]) * asck[j, t].float().repeat_interleave(16)
                ref_aq = (
                    _nvfp4_quant(act[None], w["act_global"].item())[0]
                    * w["act_global"].item()
                )
                reference_partial.act_err = max(
                    getattr(reference_partial, "act_err", 0.0),
                    ((aq_ - ref_aq).norm() / ref_aq.norm()).item(),
                )
                aq = aq_
            else:
                aq = (
                    _nvfp4_quant(act[None], w["act_global"].item())[0]
                    * w["act_global"].item()
                )
            w2 = e2m1_decode(w["w2_log"][e]) * w["w2s_log"][
                e
            ].float().repeat_interleave(16, 1)
            y = (w2 @ aq) * w["alpha2"][e]
            rows_out[t, k] = y.to(torch.bfloat16)
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
    reference_partial.parts = (rows_out, sh.to(torch.bfloat16))
    return (routed + sh).to(torch.bfloat16), ids.int(), wts, logits
