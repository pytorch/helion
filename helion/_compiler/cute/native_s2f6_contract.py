"""Exact local native decoder spelling and its proved broadcast shape."""

from __future__ import annotations

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

# Both native emitters use the compiler embedded in the pinned CuTe DSL,
# rather than an external toolkit's ptxas. Keep their hardware domain aligned.
NATIVE_S2F6_TARGETS = ((10, 0), (10, 3))

NATIVE_S2F6_ASM = """{
.reg .b16 raw, scale;
.reg .b32 decoded;
.reg .pred exceptional;
cvt.u16.u32 raw, $1;
cvt.u16.u32 scale, $2;
cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2 decoded, raw, scale;
setp.eq.u32 exceptional, $2, 255;
selp.b32 decoded, 0x7fc0, decoded, exceptional;
cvt.u16.u32 $0, decoded;
}"""


def native_s2f6_asm_matches(
    asm: object,
    constraints: object,
    dtype: object,
    pure: object,
    pack: object,
    operand_count: int,
) -> bool:
    """Recognize the exact lane-independent scalar decoder contract."""
    return (
        asm == NATIVE_S2F6_ASM
        and constraints == "=h,r,r"
        and dtype is torch.int16
        and pure is True
        and type(pack) is int
        and pack == 1
        and operand_count == 2
    )


def native_s2f6_broadcast_matches(
    raw: torch.Tensor, scale: torch.Tensor, value: torch.Tensor
) -> bool:
    rank = max(raw.ndim, scale.ndim)
    if value.ndim != rank:
        return False
    lhs = (1,) * (rank - raw.ndim) + tuple(raw.shape)
    rhs = (1,) * (rank - scale.ndim) + tuple(scale.shape)
    for a, b, result in zip(lhs, rhs, value.shape, strict=True):
        if statically_known_true(a == 1):
            expected = b
        elif statically_known_true(b == 1) or statically_known_true(a == b):
            expected = a
        else:
            return False
        if not statically_known_true(expected == result):
            return False
    return True
