"""Inkling-2 NVFP4 helpers for Helion device code: the checkpoint's "four over six" activation quantization
(per-row amax, per-16 E4M3 block scale picked between block_max/6 and block_max/4 by the smaller absolute
reconstruction error) and the routed experts' capped SwiGLU, op for op as SGLang's inkling2 kernels
(kernels/ops/inkling2/nvfp4.py _quantize_values, moe.py _capped_swiglu)."""

from __future__ import annotations

import struct

import torch

import helion.language as hl

FLT_MAX = 3.4028234663852886e38


# Triton lowers fp32 "/" to the approximate div.full: IEEE-rounded division goes through PTX, one operand per asm
# (Gluon's layout resolver rejects multi-operand inline asm whose operands resolve to different layouts).
def _f32_hex(c: float) -> str:
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", c))[0]


def rcp_rn(a: torch.Tensor) -> torch.Tensor:
    return hl.inline_asm_elementwise(
        "rcp.rn.f32 $0, $1;", "=f,f", [a], dtype=torch.float32, is_pure=True, pack=1
    )


def div_rn_by(a: torch.Tensor, c: float) -> torch.Tensor:
    """a / c, IEEE-rounded (c a compile-time constant)."""
    return hl.inline_asm_elementwise(
        f"div.rn.f32 $0, $1, {_f32_hex(c)};",
        "=f,f",
        [a],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


def rdiv_rn(c: float, a: torch.Tensor) -> torch.Tensor:
    """c / a, IEEE-rounded (c a compile-time constant)."""
    return hl.inline_asm_elementwise(
        f"div.rn.f32 $0, {_f32_hex(c)}, $1;",
        "=f,f",
        [a],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )


def pack_roundtrip(
    low: torch.Tensor, high: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """e2m1x2 byte (low nibble = low) as int32, and both values decoded back to FP32 (RN, saturating)."""
    return hl.inline_asm_elementwise(
        """
        {
          .reg .b8 b;
          .reg .b32 h;
          .reg .b16 lo, hi;
          cvt.rn.satfinite.e2m1x2.f32 b, $4, $3;
          cvt.u32.u8 $0, b;
          cvt.rn.f16x2.e2m1x2 h, b;
          mov.b32 {lo, hi}, h;
          cvt.f32.f16 $1, lo;
          cvt.f32.f16 $2, hi;
        }
        """,
        "=r,=f,=f,f,f",
        [low, high],
        dtype=(torch.int32, torch.float32, torch.float32),
        is_pure=True,
        pack=1,
    )


def _block_error(err: torch.Tensor) -> torch.Tensor:
    """[t, nb, 16] -> [t, nb]: adjacent-pair tree sum, as _block_error."""
    t, nb = err.size(0), err.size(1)
    e8 = err.reshape(t, nb, 8, 2)
    a, b = hl.split(e8)
    e4 = (a + b).reshape(t, nb, 4, 2)
    a, b = hl.split(e4)
    e2 = (a + b).reshape(t, nb, 2, 2)
    a, b = hl.split(e2)
    e1 = (a + b).reshape(t, nb, 1, 2)
    a, b = hl.split(e1)
    return (a + b).reshape(t, nb)


def _block_error_sum(err: torch.Tensor) -> torch.Tensor:
    """_block_error as two-element sums (one IEEE add each, so bitwise equal): no split/join layout round trips."""
    t, nb = err.size(0), err.size(1)
    e = err
    for k in (8, 4, 2, 1):
        e = torch.sum(e.reshape(t, nb, k, 2), dim=-1)
    return e.reshape(t, nb)


def _candidate(
    x: torch.Tensor,
    block_max: torch.Tensor,
    encoder: torch.Tensor,
    decoder: torch.Tensor,
    outer: torch.Tensor,
    inv_max: float,
    tree: bool = True,
):
    """encoder / decoder come per block ([t, nb], the row's value repeated), outer per row ([t])."""
    t, nb = x.size(0), x.size(1)
    scale = torch.minimum(
        block_max * (encoder * inv_max), hl.full([t, nb], 448.0, dtype=torch.float32)
    )
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    inverse = torch.minimum(
        rcp_rn(scale * decoder), hl.full([t, nb], FLT_MAX, dtype=torch.float32)
    )
    low, high = hl.split((x * inverse[:, :, None]).reshape(t, nb, 8, 2))
    packed, r_low, r_high = pack_roundtrip(low, high)
    rounded = hl.join(r_low, r_high).reshape(t, nb, 16)
    outer3 = outer[:, :, None] if outer.dim() == 2 else outer[:, None, None]
    diff = torch.abs((rounded * scale[:, :, None]) * outer3 - x)
    err = (
        (_block_error_sum(diff) if tree == "sum" else _block_error(diff))
        if tree
        else torch.sum(diff, dim=-1)
    )
    return packed, scale, err


def quant46(x: torch.Tensor, amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [t, nb, 16] FP32, amax [t] -> (e2m1x2 bytes [t, nb, 8] int32, E4M3 block scale bytes [t, nb] int32)."""
    t, nb = x.size(0), x.size(1)
    # per-row encoder / decoder computed per block: Gluon rejects inline asm on the [t] vector here
    row_amax = amax[:, None] + hl.zeros([t, nb], dtype=torch.float32)
    encoder = torch.minimum(
        rdiv_rn(1536.0, row_amax), hl.full([t, nb], FLT_MAX, dtype=torch.float32)
    )
    decoder = rcp_rn(encoder)
    outer = amax * (1.0 / 1536.0)
    block_max = torch.amax(torch.abs(x), dim=-1)
    packed6, scale6, err6 = _candidate(x, block_max, encoder, decoder, outer, 1.0 / 6.0)
    packed4, scale4, err4 = _candidate(x, block_max, encoder, decoder, outer, 0.25)
    choose4 = err4 < err6
    packed = torch.where(choose4[:, :, None], packed4, packed6)
    scale = (
        torch.where(choose4, scale4, scale6)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .to(torch.int32)
    )
    return packed, scale


def quant46s(x: torch.Tensor, amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """quant46, bitwise: the error tree as two-element sums, outer per block (no layout round trips)."""
    t, nb = x.size(0), x.size(1)
    row_amax = amax[:, None] + hl.zeros([t, nb], dtype=torch.float32)
    encoder = torch.minimum(
        rdiv_rn(1536.0, row_amax), hl.full([t, nb], FLT_MAX, dtype=torch.float32)
    )
    decoder = rcp_rn(encoder)
    outer = row_amax * (1.0 / 1536.0)
    block_max = torch.amax(torch.abs(x), dim=-1)
    packed6, scale6, err6 = _candidate(
        x, block_max, encoder, decoder, outer, 1.0 / 6.0, "sum"
    )
    packed4, scale4, err4 = _candidate(
        x, block_max, encoder, decoder, outer, 0.25, "sum"
    )
    choose4 = err4 < err6
    packed = torch.where(choose4[:, :, None], packed4, packed6)
    scale = (
        torch.where(choose4, scale4, scale6)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .to(torch.int32)
    )
    return packed, scale


def capped_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    gamma: torch.Tensor,
    gate_cap: float,
    up_cap: float,
):
    """((cap(gate) * sigmoid(gate)) * cap(up)) * gamma with the reference's ex2/rcp/tanh approximations (the caps'
    divisions as multiplications by their reciprocals: div.rn per element is ~5 us of R1 epilogue cold code)."""
    sigmoid = hl.inline_asm_elementwise(
        "{ .reg .f32 t; mul.rn.f32 t, $1, 0fBFB8AA3B; ex2.approx.ftz.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
        "rcp.approx.ftz.f32 $0, t; }",
        "=f,f",
        [gate],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )
    tanh_g = hl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;",
        "=f,f",
        [gate * (1.0 / gate_cap)],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )
    tanh_u = hl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;",
        "=f,f",
        [up * (1.0 / up_cap)],
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    )
    return ((gate_cap * tanh_g * sigmoid) * (up_cap * tanh_u)) * gamma


def quant46_codec(
    amax: torch.Tensor, nb: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """quant46's per-row encoder / decoder ([t, nb], the row's value repeated) and outer ([t]) for amax [t]."""
    t = amax.size(0)
    row_amax = amax[:, None] + hl.zeros([t, nb], dtype=torch.float32)
    encoder = torch.minimum(
        rdiv_rn(1536.0, row_amax), hl.full([t, nb], FLT_MAX, dtype=torch.float32)
    )
    return encoder, rcp_rn(encoder), amax * (1.0 / 1536.0)


def quant46_with(
    x: torch.Tensor,
    encoder: torch.Tensor,
    decoder: torch.Tensor,
    outer: torch.Tensor,
    tree: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """quant46 with the row codec precomputed (quant46_codec)."""
    block_max = torch.amax(torch.abs(x), dim=-1)
    packed6, scale6, err6 = _candidate(
        x, block_max, encoder, decoder, outer, 1.0 / 6.0, tree
    )
    packed4, scale4, err4 = _candidate(
        x, block_max, encoder, decoder, outer, 0.25, tree
    )
    choose4 = err4 < err6
    packed = torch.where(choose4[:, :, None], packed4, packed6)
    scale = (
        torch.where(choose4, scale4, scale6)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .to(torch.int32)
    )
    return packed, scale
