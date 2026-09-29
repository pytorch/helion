# pyrefly: ignore-errors
"""Vector pack/store helpers for CuTe codegen.

Called during ``@cute.kernel`` tracing (plain Python; builds MLIR IR
directly), so no ``@cute.jit`` wrapper is needed.
"""

from __future__ import annotations

import cutlass
from cutlass._mlir import ir
from cutlass._mlir.dialects import vector as _vector_dialect
import cutlass.cute as cute


def store_u16_vec(ptr: object, vals: list) -> None:
    """Pack ``len(vals)`` ``cutlass.Uint16`` scalars into one vector value
    and store it through ``ptr`` (a ``cute.Pointer``), emitting a single
    ST.32/ST.64/ST.128 instead of per-element 2-byte stores.

    ``vals`` is a compile-time Python list collected across an unrolled
    ``cutlass.range_constexpr(V)`` lane loop.
    """
    vecty = ir.VectorType.get([len(vals)], cutlass.Uint16.mlir_type)
    packed = _vector_dialect.from_elements(vecty, [v.ir_value() for v in vals])
    cute.arch.store(ptr, packed)


def store_u32_vec(ptr: object, vals: list) -> None:
    """``store_u16_vec`` for ``cutlass.Uint32`` lanes (fp32 stores bitcast
    their values to Uint32 first): one ST.64/ST.128 instead of per-element
    4-byte stores."""
    vecty = ir.VectorType.get([len(vals)], cutlass.Uint32.mlir_type)
    packed = _vector_dialect.from_elements(vecty, [v.ir_value() for v in vals])
    cute.arch.store(ptr, packed)


def signed_bitfield_to_bf16_packed(
    packet: object, bit_offset: int, bit_width: int, lanes: int
) -> list:
    """Exactly convert contained signed byte fields in increasing lane order.

    Insert BF16 exponent bits into the magnitude/sign halves, then subtract
    the two exactly represented integers. The zero result is positive zero.
    Only the existing 2/4/8-byte packet is read; a two-byte packet emits one pair.
    """
    assert lanes in (2, 4, 8)
    assert 1 <= bit_width <= 8 and 0 <= bit_offset <= 8 - bit_width
    magnitude = (1 << (bit_width - 1)) - 1
    sign = 1 << (bit_width - 1)
    magnitude_mask = 0xFF00FF00 | magnitude | (magnitude << 16)
    sign_mask = 0xFF00FF00 | sign | (sign << 16)
    values = []
    for group in range(max(1, lanes // 4)):
        word = cutlass.Uint32(packet >> (32 * group))
        prefix = "{ .reg .b32 field; .reg .b32 l<3>; "
        if lanes != 2:
            prefix += ".reg .b32 h<3>; "
        prefix += (
            f"shr.b32 field, {{$r0}}, {bit_offset}; prmt.b32 l0, field, 0x43, 0x4140; "
        )
        if lanes != 2:
            prefix += "prmt.b32 h0, field, 0x43, 0x4342; "
        prefix += f"and.b32 l1, l0, {magnitude_mask}; "
        if lanes != 2:
            prefix += f"and.b32 h1, h0, {magnitude_mask}; "
        prefix += f"and.b32 l2, l0, {sign_mask}; "
        if lanes != 2:
            prefix += f"and.b32 h2, h0, {sign_mask}; "
        prefix += "sub.bf16x2 {$w0}, l1, l2; "
        if lanes == 2:
            low = cute.arch.inline_ptx(
                prefix + "}",
                write_only_types=[cutlass.Uint32],
                read_only_args=[word],
            )
            values.extend([cutlass.Uint16(low & 65535), cutlass.Uint16(low >> 16)])
        else:
            low, high = cute.arch.inline_ptx(
                prefix + "sub.bf16x2 {$w1}, h1, h2; }",
                write_only_types=[cutlass.Uint32, cutlass.Uint32],
                read_only_args=[word],
            )
            values.extend(
                [
                    cutlass.Uint16(low & 65535),
                    cutlass.Uint16(low >> 16),
                    cutlass.Uint16(high & 65535),
                    cutlass.Uint16(high >> 16),
                ]
            )
    return values
