# pyrefly: ignore-errors
"""Vector pack/store helpers for CuTe codegen.

Called during ``@cute.kernel`` tracing (plain Python; builds MLIR IR
directly), so no ``@cute.jit`` wrapper is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cutlass
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import vector as _vector_dialect
import cutlass.cute as cute

if TYPE_CHECKING:
    from collections.abc import Callable


def store_u8_vec(
    ptr: object,
    vals: list,
    *,
    packet_store: Callable[[object, object], None] | None = None,
) -> None:
    """Store 2/4/8/16 byte lanes as one naturally aligned integer packet."""
    store = cute.arch.store if packet_store is None else packet_store
    if len(vals) == 16:
        words = []
        for start in (0, 8):
            packed = cutlass.Uint64(0)
            for index, value in enumerate(vals[start : start + 8]):
                packed = packed | (cutlass.Uint64(value) << (8 * index))
            words.append(packed.ir_value())
        packet = _vector_dialect.from_elements(
            ir.VectorType.get([2], cutlass.Uint64.mlir_type), words
        )
        store(ptr, packet)
        return
    carrier = {2: cutlass.Uint16, 4: cutlass.Uint32, 8: cutlass.Uint64}[len(vals)]
    packed = carrier(0)
    for index, value in enumerate(vals):
        packed = packed | (carrier(value) << (8 * index))
    store(ptr, packed)


def _store_fp8_vec(
    ptr: object,
    vals: list,
    dtype: type,
    *,
    packet_store: Callable[[object, object], None] | None = None,
) -> None:
    store = cute.arch.store if packet_store is None else packet_store
    source_dtype = type(vals[0])
    if source_dtype in (cutlass.Float32, cutlass.Float16, cutlass.BFloat16) and all(
        type(value) is source_dtype and isinstance(value.value, ir.Value)
        for value in vals
    ):
        # Vector truncation has the scalar constructor's rounding and overflow
        # semantics, and lets the CUDA lowering pair its FP8 conversions.
        # Literal-backed wrappers still hold the original Python float;
        # emitting them in the source dtype first would add another rounding.
        source_type = ir.VectorType.get([len(vals)], source_dtype.mlir_type)
        values = [value.ir_value() for value in vals]
        source = _vector_dialect.from_elements(
            source_type,
            values,
        )
        converted = arith.truncf(
            ir.VectorType.get([len(vals)], dtype.mlir_type), source
        )
        carrier = {
            2: cutlass.Uint16,
            4: cutlass.Uint32,
            8: cutlass.Uint64,
            16: cutlass.Uint64,
        }[len(vals)]
        packed = _vector_dialect.bitcast(
            ir.VectorType.get([max(1, len(vals) // 8)], carrier.mlir_type), converted
        )
        if len(vals) == 16:
            store(ptr, packed)
            return
        scalar = _vector_dialect.extract(packed, [], [0])
        store(ptr, carrier(scalar))
    else:
        encoded = [dtype(value).bitcast(cutlass.Uint8) for value in vals]
        if packet_store is None:
            store_u8_vec(ptr, encoded)
        else:
            store_u8_vec(ptr, encoded, packet_store=packet_store)


def store_fp8e4m3fn_vec(ptr: object, vals: list) -> None:
    """Convert FP32/FP16/BF16 lanes to E4M3FN and store one byte packet."""
    _store_fp8_vec(ptr, vals, cutlass.Float8E4M3FN)


def store_fp8e5m2_vec(ptr: object, vals: list) -> None:
    """Convert FP32/FP16/BF16 lanes to E5M2 and store one byte packet."""
    _store_fp8_vec(ptr, vals, cutlass.Float8E5M2)


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


def red_add_f32_vec(ptr: object, vals: list) -> None:
    """One ``red.global.add.v2/v4.f32`` for ``len(vals)`` fp32 lanes at ``ptr``.

    ``ptr`` is the ``llvm_ptr`` of a ``cute.Pointer`` to a naturally aligned
    8- or 16-byte span of global memory; ``vals`` are the per-lane
    ``cutlass.Float32`` addends collected across an unrolled
    ``cutlass.range_constexpr(V)`` lane loop (the atomic twin of
    ``store_u32_vec``).  A fire-and-forget relaxed gpu-scope reduction, the
    ordering ``cute.arch.atomic_add(..., sem='relaxed')`` gives, issued as a
    single sm_90+ vector instruction instead of ``V`` scalar atomics.
    """
    width = len(vals)
    assert width in (2, 4), width
    operands = ", ".join(f"${i}" for i in range(1, width + 1))
    llvm.inline_asm(
        None,
        [ptr, *(cutlass.Float32(v).ir_value() for v in vals)],
        f"red.global.add.v{width}.f32 [$0], {{{operands}}};",
        ",".join(["l", *(["f"] * width)]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


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
