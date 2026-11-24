"""Shared helpers for the Muse Spark spec-decode verify megakernels.

The PTX helpers reproduce the instructions the production CUDA kernels use
(``rsqrt.approx``, ``rcp.approx``, ``ex2.approx`` and the NVFP4 conversions), so
the megakernels match the served kernels' rounding.
"""

from __future__ import annotations

import dataclasses

import torch

import helion.language as hl

WORLD_SIZE = 4
SIGNAL_PAD_BYTES = 32 * 1024
# Verify windows the sglang integration serves: DFlash2 draft lengths.
VERIFY_TOKENS = (8, 16)
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
LOG2E = 1.4426950408889634
HIDDEN = 8192
EXPERT_HIDDEN = 4096
NUM_EXPERTS = 256
TOPK = 8
# dot_scaled needs N >= 16: a shorter verify window pads the MoE's MMA token rows
# to 16 with zero activations; routes and outputs keep T rows.
MOE_TOKENS = 16


@dataclasses.dataclass(frozen=True)
class ExpertInputs:
    """The MoE megakernel's NVFP4 input slot, which the router fills: 16
    activation rows (the first T live, the rest zero) and T route rows."""

    fp4: torch.Tensor  # [16, Ne / 2] u8
    scales: torch.Tensor  # [16, Ne / 16] u8 (E4M3 bits)
    per_token_scale: torch.Tensor  # [16] fp32
    packed: torch.Tensor  # [T, 8] int32 (expert << 16 | bf16 weight bits)

    @staticmethod
    def zeros(tokens: int, device: torch.device) -> ExpertInputs:
        return ExpertInputs(
            fp4=torch.zeros(
                MOE_TOKENS, EXPERT_HIDDEN // 2, dtype=torch.uint8, device=device
            ),
            scales=torch.zeros(
                MOE_TOKENS, EXPERT_HIDDEN // 16, dtype=torch.uint8, device=device
            ),
            per_token_scale=torch.zeros(MOE_TOKENS, dtype=torch.float32, device=device),
            packed=torch.zeros(tokens, TOPK, dtype=torch.int32, device=device),
        )


def rsqrt_approx_ftz_f32(value: torch.Tensor) -> torch.Tensor:
    """The reciprocal square root instruction the CUDA kernels use."""
    return hl.inline_asm_elementwise(
        "rsqrt.approx.ftz.f32 $0, $1;",
        "=f,f",
        [value],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


def rcp_approx_ftz_f32(value: torch.Tensor) -> torch.Tensor:
    """The fast reciprocal selected by CUDA's ``__fdividef(1, value)``."""
    return hl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        "=f,f",
        [value],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


def exp2_approx_ftz_f32(value: torch.Tensor) -> torch.Tensor:
    """The single-instruction exponential underlying CUDA's ``__expf``."""
    return hl.inline_asm_elementwise(
        "ex2.approx.ftz.f32 $0, $1;",
        "=f,f",
        [value],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


def encode_block_scale(
    token_scale: torch.Tensor,
    block_amax: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One E4M3 block scale byte and its quant multiplier; a zero token scale
    yields FlashInfer's zero-row result."""
    return hl.inline_asm_elementwise(
        """
        {
          .reg .pred nonzero, scale_nonzero;
          .reg .b16 fp8_pair;
          .reg .b32 h2_32;
          .reg .b16 h_lo, h_hi;
          .reg .f32 encode_scale, block_scaled, scale_f;
          .reg .f32 scale_f32, rcp_gs, product, result;
          setp.ne.f32 nonzero, $2, 0f00000000;
          @nonzero rcp.approx.ftz.f32 encode_scale, $2;
          @!nonzero mov.b32 encode_scale, 0x7f7fffff;
          mul.f32 block_scaled, $3, 0f3e2aaaab;
          mul.f32 scale_f, encode_scale, block_scaled;
          cvt.rn.satfinite.e4m3x2.f32 fp8_pair, 0f00000000, scale_f;
          @!nonzero mov.b16 fp8_pair, 0x0000;
          cvt.u32.u16 $0, fp8_pair;
          cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;
          mov.b32 {h_lo, h_hi}, h2_32;
          cvt.f32.f16 scale_f32, h_lo;
          rcp.approx.ftz.f32 rcp_gs, encode_scale;
          mul.f32 product, scale_f32, rcp_gs;
          rcp.approx.ftz.f32 result, product;
          setp.ne.f32 scale_nonzero, scale_f32, 0f00000000;
          selp.f32 $1, result, 0f00000000, scale_nonzero;
        }
        """,
        "=r,=f,f,f",
        [token_scale, block_amax],
        dtype=(torch.int32, torch.float32),
        is_pure=True,
        pack=1,
    )


def pack_e2m1x2(low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """One e2m1x2 byte (as int32) per (low, high) FP32 pair, RN saturating."""
    return hl.inline_asm_elementwise(
        """
        {
          .reg .b8 b;
          cvt.rn.satfinite.e2m1x2.f32 b, $2, $1;
          cvt.u32.u8 $0, b;
        }
        """,
        "=r,f,f",
        [low, high],
        dtype=torch.int32,
        is_pure=True,
        pack=1,
    )


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
    expo = exp2_approx_ftz_f32((0.0 - scaled_logits) * LOG2E)
    score = _div_full(torch.ones_like(expo), expo + 1.0)
    key = score + bias
    # Order-preserving key bits (+0.0 folds -0) above the complemented id, so
    # ties go to the lower id.
    kbits = (key + 0.0).view(dtype=torch.int32)
    ordered = torch.where(kbits >= 0, kbits, kbits ^ 0x7FFFFFFF).to(torch.int64)
    packed_key = (ordered << 32) | (num_experts - 1 - experts).to(torch.int64)
    # The keys are distinct, so one top-8 selects what eight max rounds would.
    best8 = torch.topk(packed_key, 8)[0]
    sel_ids = (num_experts - 1) - (best8 & 0xFFFF).to(torch.int32)
    sel_scores = torch.sum(
        torch.where(experts[None, :] == sel_ids[:, None], score[None, :], 0.0), dim=-1
    )
    score_sum = _pick(sel_scores, 0, 8) + _pick(sel_scores, 1, 8)
    for kk in hl.static_range(2, 8):
        score_sum = score_sum + _pick(sel_scores, kk, 8)
    zeros8 = hl.zeros([8], dtype=torch.float32)
    scale = _div_full(zeros8 + route_scale, zeros8 + (score_sum + route_eps))
    return sel_scores * scale, sel_ids


def e2m1_values(packed: torch.Tensor) -> torch.Tensor:
    """FP64 e2m1 values of e2m1x2 bytes ``[..., K / 2]`` (low nibble first)."""
    table = torch.tensor(
        [*E2M1_VALUES, *(-v for v in E2M1_VALUES)],
        dtype=torch.float64,
        device=packed.device,
    )
    data = packed.view(torch.uint8).to(torch.int64)
    return torch.stack((table[data & 0xF], table[data >> 4]), dim=-1).flatten(-2)


def round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """e2m1 round-to-nearest-even, saturating at 6."""
    a = x.abs()
    index = sum(
        (a >= bound if odd_below else a > bound).int()
        for bound, odd_below in (
            (0.25, False),
            (0.75, True),
            (1.25, False),
            (1.75, True),
            (2.5, False),
            (3.5, True),
            (5.0, False),
        )
    )
    return torch.tensor(E2M1_VALUES, device=x.device)[index] * torch.sign(x)


def dequantize_e2m1(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """FP64 e2m1 values ``[..., K]`` times their E4M3 16-value block scales
    ``[..., K / 16]``."""
    block = scales.view(torch.float8_e4m3fn).to(torch.float64)
    return (values.unflatten(-1, (-1, 16)) * block[..., None]).flatten(-2)


def dequantize_nvfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """FP64 values of e2m1x2 bytes ``[..., K / 2]`` with E4M3 16-value block
    scales ``[..., K / 16]`` (low nibble first)."""
    return dequantize_e2m1(e2m1_values(packed), scales)


def relative_rmse_and_cosine(
    actual: torch.Tensor, expected: torch.Tensor
) -> tuple[float, float]:
    """Relative RMS error and cosine similarity in FP64."""
    a = actual.double().reshape(-1)
    e = expected.double().reshape(-1)
    return float((a - e).norm() / e.norm()), float(
        torch.dot(a, e) / (a.norm() * e.norm())
    )
