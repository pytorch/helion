"""Ordered FP16/BF16 keys with deterministic index tie breaking.

The encoded signed Int32 maximum identifies the preferred value/index pair.
The 16-bit rank and at most 15 index bits leave every valid key above INT_MIN,
which callers can use to mask missing candidates. Paired encoding retains the
ordinal, NaNs-last policy used for register selection.
"""

from __future__ import annotations

import cutlass
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from ..._compiler.cute._mlir_compat import ir


@dsl_user_op
def encode_ordered_key_16(
    word: cutlass.Uint16,
    column: Int32,
    index_bits: int,
    largest: bool,
    rank_mode: str,
    infinity_bits: int,
    *,
    nan_order: str = "last",
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> Int32:
    """Pack an FP16/BF16 ordered value and a first-index tie breaker.

    ``rank_mode="signed"`` makes both zero signs equal; ``"ordinal"`` keeps
    their bitwise ordering. ``nan_order`` chooses the NaN end of ascending
    order before ``largest`` reverses the value order. All NaN payloads have
    the same rank. The original value must be recovered if its bits matter.
    """
    assert 0 <= index_bits <= 15
    assert rank_mode in ("signed", "ordinal")
    assert nan_order in ("first", "last")
    assert infinity_bits in (0x7C00, 0x7F80)
    nan_rank = 32767 if nan_order == "last" else -32767
    index_mask = (1 << index_bits) - 1
    rank_code = (
        "and.b32 rank, sign, 0x7fff; xor.b32 rank, signed_word, rank;"
        if rank_mode == "ordinal"
        else "xor.b32 rank, magnitude, sign; sub.s32 rank, rank, sign;"
    )
    reverse = "neg.s32 rank, rank;" if not largest else ""
    result = llvm.inline_asm(
        Int32.mlir_type,
        [word.ir_value(loc=loc, ip=ip), column.ir_value(loc=loc, ip=ip)],
        f"""
        {{
          .reg .b32 signed_word, magnitude, sign, rank, payload, packed;
          .reg .pred is_nan;
          cvt.s32.s16 signed_word, $1;
          and.b32 magnitude, signed_word, 0x7fff;
          shr.s32 sign, signed_word, 31;
          {rank_code}
          setp.gt.u32 is_nan, magnitude, {infinity_bits};
          selp.b32 rank, {nan_rank}, rank, is_nan;
          {reverse}
          shl.b32 packed, rank, {index_bits};
          sub.u32 payload, {index_mask}, $2;
          or.b32 packed, packed, payload;
          mov.b32 $0, packed;
        }}
        """,
        "=r,h,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def encode_ordinal_key_pair_16(
    words: cutlass.Uint32,
    column: Int32,
    index_bits: int,
    largest: bool,
    infinity_bits: int,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> tuple[Int32, Int32]:
    """Encode two adjacent 16-bit values without splitting their rank work.

    Each masked magnitude is at most 0x7fff. Adding 0x7fff-infinity
    cannot carry into its neighboring half; its sign bit detects only NaNs.
    PRMT expands those bits into masks for exact per-half canonicalization.
    """
    assert 1 <= index_bits <= 15
    assert infinity_bits in (0x7C00, 0x7F80)
    bias = 0x7FFF - infinity_bits
    packed_bias = bias | (bias << 16)
    index_mask = (1 << index_bits) - 1
    reverse = "neg.s32 lo, lo; neg.s32 hi, hi;" if not largest else ""
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [words.ir_value(loc=loc, ip=ip), column.ir_value(loc=loc, ip=ip)],
        f"""
        {{
          .reg .b32 sign, rank, magnitude, biased, nan_mask, lo, hi, payload;
          prmt.b32 sign, $2, 0, 0xbb99;
          lop3.b32 rank, $2, sign, 0x7fff7fff, 0x78;
          and.b32 magnitude, $2, 0x7fff7fff;
          add.u32 biased, magnitude, {packed_bias};
          prmt.b32 nan_mask, biased, 0, 0xbb99;
          lop3.b32 rank, rank, nan_mask, 0x7fff7fff, 0xb8;
          prmt.b32 lo, rank, 0, 0x9910;
          prmt.b32 hi, rank, 0, 0xbb32;
          {reverse}
          shl.b32 lo, lo, {index_bits};
          sub.u32 payload, {index_mask}, $3;
          or.b32 $0, lo, payload;
          shl.b32 hi, hi, {index_bits};
          sub.u32 payload, payload, 1;
          or.b32 $1, hi, payload;
        }}
        """,
        "=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(Int32.mlir_type, result, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(Int32.mlir_type, result, [1], loc=loc, ip=ip)),
    )
