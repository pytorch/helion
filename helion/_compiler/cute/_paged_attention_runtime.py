"""Device building blocks for paged, quantized attention schedules.

The scaled S2F6 instruction is an SM100 feature. Its caller must establish
that target and that each decoded group shares the supplied E8M0 scale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import cutlass
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith
from cutlass._mlir.dialects import cute as cute_ir
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import nvvm
from cutlass._mlir.dialects import vector
import cutlass.cute as cute
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives as prims

from ._flash_row_mma_runtime import pack_bf16x2
from .device.ptx import add_bf16x2
from .device.ptx import l2_policy_evict_first
from .device.ptx import l2_policy_evict_last
from .device.ptx import mul_bf16x2

if TYPE_CHECKING:
    from typing import Any


def _tmem_store_address(row: cutlass.Int32, column: cutlass.Int32) -> cutlass.Int32:
    """The tensor-memory address of lane ``row`` and ``column``, as a sum.

    An address holds the lane in bits 16 and up and the column below them.
    ``row`` is one of a tile's 128 lanes, so ``row << 16`` has no bit below
    16. ``column`` is the allocation's base plus an offset inside the
    allocation; the base ``tcgen05.alloc`` wrote is a column at lane zero and
    an SM has 512 columns, so ``column`` has no bit at 16 or above. No bit is
    set in both terms, the sum carries nowhere, and it equals their OR.
    Only the sum folds: a role's stores differ in constant lane and column
    offsets, so they share one ``(row << 16) + base`` and each adds an
    immediate, where every OR is computed on its own.
    """
    return (row << 16) + column


@cute.jit
def stage_positive_zero_value_tile(
    dimension_base: cutlass.Int32, tmem_base: cutlass.Int32
) -> None:
    """Write 32 dimension rows by 128 keys as packed positive BF16 zeros."""
    zero = cutlass.Array(cutlass.Int32, 32, alignment=16)
    for i in cutlass.range_constexpr(32):
        zero[i] = cutlass.Int32(0)
    for half in cutlass.range_constexpr(2):
        address = _tmem_store_address(dimension_base, tmem_base + half * 32)
        prims.tcgen05_st(
            "32x32b", cutlass.inttoptr(address, 6, cutlass.Int32), zero[0:32]
        )


@dsl_user_op
def _scaled_s2f6x2_to_bf16x2(
    values: cutlass.Int32,
    scales: cutlass.Int32,
    *,
    loc: object = None,
    ip: object = None,
) -> cutlass.Int32:
    """Convert the two low signed bytes using two low UE8M0 scale bytes."""
    source = arith.trunci(T.i16(), values.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    source = llvm.bitcast(
        cast("Any", ir).VectorType.get([2], T.i8()), source, loc=loc, ip=ip
    )
    scale = arith.trunci(T.i16(), scales.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    result = nvvm.convert_s2f6x2_to_bf16x2(
        cast("Any", ir).VectorType.get([2], cutlass.BFloat16.mlir_type),
        source,
        scale_factor=scale,
        sat=nvvm.SaturationMode.SATFINITE,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(llvm.bitcast(T.i32(), result, loc=loc, ip=ip))


@dsl_user_op
def _native_s2f6x2_to_bf16x2(
    values: cutlass.Int32,
    scales: cutlass.Int32,
    *,
    loc: object = None,
    ip: object = None,
) -> cutlass.Int32:
    """Native signed-byte/UE8M0 conversion with exact per-halfword NaN bits.

    Scale bytes are unmodified. In particular, scale0 preserves subnormals;
    each scale255 independently selects the authored canonical BF16 NaN.
    The selects contain no control flow or collective operation.
    """
    result = _scaled_s2f6x2_to_bf16x2(values, scales, loc=loc, ip=ip)
    low = arith.select(
        ((scales & 255) == 255).ir_value(loc=loc, ip=ip),
        cutlass.Int32(0x7FC0).ir_value(loc=loc, ip=ip),
        (result & 0xFFFF).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    high = arith.select(
        ((cutlass.Int32(scales >> 8) & 255) == 255).ir_value(loc=loc, ip=ip),
        cutlass.Int32(0x7FC00000).ir_value(loc=loc, ip=ip),
        (result & cutlass.Int32(-65536)).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(arith.ori(low, high, loc=loc, ip=ip))


@cute.jit
def _pack_postscaled_bf16(
    low: cutlass.Float32,
    high: cutlass.Float32,
    post_scale_log2: cutlass.Constexpr,
) -> cutlass.Int32:
    """Preserve the first BF16 rounding before an explicit source scaling."""
    if cutlass.const_expr(post_scale_log2 != 0):
        low = cutlass.Float32(cutlass.BFloat16(low)) * (2.0**post_scale_log2)
        high = cutlass.Float32(cutlass.BFloat16(high)) * (2.0**post_scale_log2)
    return cutlass.Int32(pack_bf16x2(low, high))


@cute.jit
def scaled_s8_to_bf16_words(
    words: cute.Tensor,
    scale_byte: cutlass.Uint8,
    scale_codes_safe: cutlass.Boolean | None = None,
    post_scale_log2: cutlass.Constexpr = 0,
) -> cute.Tensor:
    """Decode packed INT8 words to twice as many packed BF16 words.

    A single scale read and branch serves the whole packet. The fast branch
    compensates S2F6's six fractional bits in the exponent. The exceptional
    scale range uses the source expression's FP32 multiply and BF16 rounding:
    UE8M0's special byte and SATFINITE must not replace those semantics.
    This includes INT8 -128 and scale zero, and assumes no workload-specific
    restriction on either byte's value.
    Passing scale_codes_safe=True must prove this invocation's scale is at
    most 247. False disables the fused S2F6 branch; the exact signed-zero
    branch remains valid independently. Ordinary callers check their scale.

    A source post-multiply by 2**p, -22 <= p <= 0, can join the guarded
    conversion for scale <=247: the original decoded value is exactly BF16.
    Negative adjusted scales use native scale0 followed by a packed BF16
    power-of-two multiply, preserving subnormal rounding and signed zero.
    Scales <=-14-p round every raw value to signed zero and use exact sign-bit
    transport without performing the conversion.
    Exceptional scales retain both source roundings; e.g. a first conversion
    that overflows must remain infinite after a finite negative power of two.
    """
    # The smallest nonzero decoded byte is 2**-127. This tested range keeps
    # its FP32 postmultiply >=2**-149, avoiding an extra FP32 underflow rounding.
    assert cutlass.const_expr(-22 <= post_scale_log2 <= 0)
    result = cute.make_rmem_tensor((2 * cute.size(words),), cutlass.Int32)
    scale = cutlass.Int32(scale_byte)
    if cutlass.const_expr(scale_codes_safe is None):
        safe = scale <= 247
    else:
        safe = scale_codes_safe
    underflows = cutlass.Boolean(False)
    if cutlass.const_expr(post_scale_log2 <= -14):
        underflows = scale <= -14 - post_scale_log2
    if underflows:
        # Every original BF16 value rounds to signed zero at this exponent.
        # Read every raw byte and retain its sign; this is not zero-data skipping.
        for index in cutlass.range_constexpr(cute.size(words)):
            word = cutlass.Int32(words[index])
            for half in cutlass.range_constexpr(2):
                pair = cutlass.Int32(word >> (16 * half))
                first = cutlass.Int32(pair & 0x80) << 8
                second = cutlass.Int32(pair & 0x8000) << 16
                result[2 * index + half] = cutlass.Int32(first) | cutlass.Int32(second)
    elif safe:
        if cutlass.const_expr(post_scale_log2 < -6):
            paired, factors = _postscale_pair_bits(
                scale | (scale << 8), post_scale_log2
            )
        else:
            corrected = scale + (6 + post_scale_log2)
            paired = corrected | (corrected << 8)
            # Dead in this constexpr arm; keep definite assignment explicit.
            factors = cutlass.Int32(0x3F803F80)
        for index in cutlass.range_constexpr(cute.size(words)):
            word = cutlass.Int32(words[index])
            low = _scaled_s2f6x2_to_bf16x2(word, paired)
            high = _scaled_s2f6x2_to_bf16x2(word >> 16, paired)
            if cutlass.const_expr(post_scale_log2 < -6):
                low = mul_bf16x2(low, factors)
                high = mul_bf16x2(high, factors)
            result[2 * index] = low
            result[2 * index + 1] = high
    else:
        factor = cute.math.exp2(cutlass.Float32(scale) - 127.0, fastmath=False)
        for index in cutlass.range_constexpr(cute.size(words)):
            word = cutlass.Int32(words[index])
            low = cutlass.Float32(cutlass.Int32(word << 24) >> 24) * factor
            high = cutlass.Float32(cutlass.Int32(word << 16) >> 24) * factor
            result[2 * index] = _pack_postscaled_bf16(low, high, post_scale_log2)
            low = cutlass.Float32(cutlass.Int32(word << 8) >> 24) * factor
            high = cutlass.Float32(word >> 24) * factor
            result[2 * index + 1] = _pack_postscaled_bf16(low, high, post_scale_log2)
    return result


@cute.jit
def _native_s8_to_bf16_words(words: cute.Tensor, paired: cutlass.Int32) -> cute.Tensor:
    """The native arm of ``scaled_s8_to_bf16_words`` for a corrected scale
    pair in the two low bytes of ``paired``: both bytes at most 253 after a
    correction that is not negative, which the caller proves."""
    result = cute.make_rmem_tensor((2 * cute.size(words),), cutlass.Int32)
    for index in cutlass.range_constexpr(cute.size(words)):
        word = cutlass.Int32(words[index])
        result[2 * index] = _scaled_s2f6x2_to_bf16x2(word, paired)
        result[2 * index + 1] = _scaled_s2f6x2_to_bf16x2(word >> 16, paired)
    return result


@dsl_user_op
def _load_words4(
    pointer: cute.Pointer,
    *,
    loc: object = None,
    ip: object = None,
) -> tuple[cutlass.Int32, ...]:
    """A sixteen-byte load in the pointer's space at a proven aligned address."""
    assert pointer.alignment >= 16
    values = cute.arch.load(
        pointer,
        cast("Any", ir).VectorType.get([4], T.i32()),
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Int32(
            vector.extract(
                values, dynamic_position=[], static_position=[i], loc=loc, ip=ip
            )
        )
        for i in range(4)
    )


@dsl_user_op
def _permute_scale_bytes(
    word: cutlass.Int32, selector: int, *, loc: object = None, ip: object = None
) -> cutlass.Int32:
    """Bytes of one scale word in another order: result byte ``i`` is the
    byte of ``word`` that nibble ``i`` of ``selector`` names (0 to 3).

    One ``prmt.b32``. A shift written in its place is re-derived by NVPTX at
    every use: sixteen conversions of a V tile then pay 24 more permutes.
    """
    assert 0 <= selector <= 0x3333 and not selector & 0xCCCC
    return cutlass.Int32(
        nvvm.prmt(
            word.ir_value(loc=loc, ip=ip),
            cutlass.Int32(selector).ir_value(loc=loc, ip=ip),
            nvvm.PermuteMode.DEFAULT,
            hi=cutlass.Int32(0).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


def _corrected_scale_word(word: cutlass.Int32, post_scale_log2: int) -> cutlass.Int32:
    """Four scale bytes with S2F6's six fractional bits and a source
    postscale of ``2**post_scale_log2`` added to each exponent, in one add.

    The caller proves ``post_scale_log2 >= -6``, so the correction is at
    most six and not negative. For bytes of at most 247, the native finite
    domain, no byte carries into its neighbour. A word with a byte above
    that may carry: its caller converts with it only what a failed scale
    vote replaces.
    """
    assert -6 <= post_scale_log2 <= 0
    return word + cutlass.Int32(0x01010101 * (6 + post_scale_log2))


@cute.jit
def _stage_scaled_s8_rows(
    raw: cute.Tensor,
    scales: cute.Tensor,
    row: cutlass.Int32,
    tmem_base: cutlass.Int32,
    columns: cutlass.Constexpr,
    scale_group: cutlass.Constexpr,
    post_scale_log2: cutlass.Constexpr,
    packed_scales: cutlass.Constexpr,
    scale_word: cutlass.Int32,
    row_scales_safe: cutlass.Boolean,
    all_rows_safe: cutlass.Constexpr,
    packed_tile: cutlass.Constexpr = False,
) -> None:
    """Stage rows using a proved warp-wide or original per-lane conversion.

    Only the wrapper's unanimous scale vote may select all_rows_safe. That
    vote makes both outer arms uniform, as required by collective TMEM stores.
    Mixed warps retain each row's original guard; generic layouts retain each
    group's original scale check.

    ``packed_tile`` (``packed_scale_tile_layout``) gives the unanimous arm
    its exponent correction once per row: one add over the row's scale word
    and one byte permute per group, where each group otherwise extracts its
    byte, corrects it and pairs it. Every byte is at most 247 in that arm,
    so the add carries nowhere and each group converts with the scale pair
    it had.
    """
    assert cutlass.const_expr(not packed_tile or packed_scales)
    corrected_word = scale_word
    if cutlass.const_expr(packed_tile and all_rows_safe):
        corrected_word = _corrected_scale_word(scale_word, post_scale_log2)
    warp_row = row // 32 * 32
    store_groups = 1
    if cutlass.const_expr(
        scale_group < 64 and 64 % scale_group == 0 and columns % 64 == 0
    ):
        # Adjacent groups keep their independent scale/rounding semantics but
        # share one 64-dimension TMEM transfer, reducing stores and address work.
        store_groups = 64 // scale_group
    store_words = store_groups * scale_group // 2
    for block in cutlass.range_constexpr(columns // (store_groups * scale_group)):
        packed = cutlass.Array(cutlass.Int32, store_words, alignment=16)
        for inner in cutlass.range_constexpr(store_groups):
            group = block * store_groups + inner
            words = cute.make_rmem_tensor((scale_group // 4,), cutlass.Int32)
            if cutlass.const_expr(packed_scales):
                scale = cutlass.Uint8(scale_word >> (8 * group))
            else:
                scale = cutlass.Uint8(scales[row, group])
            for packet in cutlass.range_constexpr(scale_group // 16):
                column = group * scale_group + packet * 16
                offset = cute.crd2idx((row, column), raw.layout)
                pointer = cute.recast_ptr(
                    raw.iterator + cute.assume(offset, divby=16), dtype=cutlass.Int32
                )
                values = _load_words4(pointer)
                for index in cutlass.range_constexpr(4):
                    words[packet * 4 + index] = values[index]
            # Conversion branches contain no collective. Keep the TMEM store
            # outside these lane-dependent branches.
            if cutlass.const_expr(packed_tile and all_rows_safe):
                converted = _native_s8_to_bf16_words(
                    words, _permute_scale_bytes(corrected_word, group | (group << 4))
                )
            elif cutlass.const_expr(all_rows_safe):
                converted = scaled_s8_to_bf16_words(words, scale, True, post_scale_log2)
            elif cutlass.const_expr(packed_scales):
                converted = scaled_s8_to_bf16_words(
                    words, scale, row_scales_safe, post_scale_log2
                )
            else:
                converted = scaled_s8_to_bf16_words(
                    words, scale, post_scale_log2=post_scale_log2
                )
            for index in cutlass.range_constexpr(scale_group // 2):
                packed[inner * (scale_group // 2) + index] = converted[index]
        address = _tmem_store_address(warp_row, tmem_base + block * store_words)
        destination = cutlass.inttoptr(address, 6, cutlass.Int32)
        prims.tcgen05_st("32x32b", destination, packed[0:store_words])


@cute.jit
def stage_scaled_s8_rows_to_tmem(
    raw: cute.Tensor,
    scales: cute.Tensor,
    row: cutlass.Int32,
    tmem_base: cutlass.Int32,
    columns: cutlass.Constexpr,
    scale_group: cutlass.Constexpr,
    post_scale_log2: cutlass.Constexpr = 0,
    packed_tile_forms: cutlass.Constexpr = False,
) -> None:
    """Convert one row per lane from a raw shared tile directly into TMEM.

    A participating warp owns 32 consecutive rows; ``row % 32`` is its lane.
    Its physical warp index modulo four must equal ``row // 32``. TMEM
    stores select that physical quarter; changing row-address bits cannot
    make one warp write all four quarters of a 128-row tile.
    The caller proves columns/group are multiples of 16, packet-aligned raw
    storage, and complete active warps. Four contiguous scales use one packed
    word load. Groups of 16 or 32 dimensions share 64-dimension TMEM stores
    when the complete row permits it; each group keeps its own scale.
    A unanimous warp scale check hoists the common conversion branch. Mixed
    warps keep the original per-row guards, so an exceptional row cannot force
    other rows onto full-domain arithmetic. Exact signed-zero groups remain
    independent of those conservative guards.
    Raw and scale tensors carry their actual shared layouts (including the
    raw TMA swizzle), so no global page pitch is baked into the conversion.
    The raw base carries at least 16-byte alignment; packet offsets retain
    their proved divisibility in the pointer type.
    Completion/fences and ready/free barriers belong to the owning pipeline.

    ``packed_tile_forms``: the caller takes the forms of the packed
    four-scales-per-row tile where this call has that layout
    (``packed_scale_tile_layout``); here the unanimous arm's exponent
    correction once per row (``_stage_scaled_s8_rows``). The guard, its
    vote and the mixed arm are unchanged, and every row converts with the
    scale bytes it had, so the staged K is the same for every input.
    """
    packed_scales = columns // scale_group == 4 and _packed_scale_layout_supported(
        scales
    )
    packed_tile = packed_tile_forms and packed_scale_tile_layout(
        scales, columns, scale_group, post_scale_log2
    )
    scale_word = cutlass.Int32(0)
    row_scales_safe = cutlass.Boolean(False)
    if cutlass.const_expr(packed_scales):
        # One word supplies all four independent scales for this row.
        offset = cute.crd2idx((row, 0), scales.layout)
        pointer = cute.recast_ptr(
            scales.iterator + cute.assume(offset, divby=4), dtype=cutlass.Int32
        )
        scale_word = cute.arch.load(pointer, cutlass.Int32)
        row_scales_safe = _scale_word_uses_s2f6(scale_word, post_scale_log2)
    if cutlass.const_expr(packed_scales):
        # One vote covers all four scales in each lane. Keep both collective
        # store paths uniform, while retaining the original guards on fallback.
        # The branch-local syncs preserve convergence for NVVM, keeping the
        # uniform fast path as a branch instead of predicating both bodies.
        # They are a code-generation constraint, not extra data synchronization.
        if cute.arch.vote_all_sync(row_scales_safe):
            cute.arch.sync_warp()
            _stage_scaled_s8_rows(
                raw,
                scales,
                row,
                tmem_base,
                columns,
                scale_group,
                post_scale_log2,
                packed_scales=True,
                scale_word=scale_word,
                row_scales_safe=row_scales_safe,
                all_rows_safe=True,
                packed_tile=packed_tile,
            )
        else:
            cute.arch.sync_warp()
            _stage_scaled_s8_rows(
                raw,
                scales,
                row,
                tmem_base,
                columns,
                scale_group,
                post_scale_log2,
                packed_scales=True,
                scale_word=scale_word,
                row_scales_safe=row_scales_safe,
                all_rows_safe=False,
            )
    else:
        _stage_scaled_s8_rows(
            raw,
            scales,
            row,
            tmem_base,
            columns,
            scale_group,
            post_scale_log2,
            packed_scales=False,
            scale_word=scale_word,
            row_scales_safe=row_scales_safe,
            all_rows_safe=False,
        )


@dsl_user_op
def _load_transposed_s8_16x16(
    pointer: cute.Pointer,
    *,
    loc: object = None,
    ip: object = None,
) -> tuple[cutlass.Int32, ...]:
    """Each lane receives four rows of two columns of a 16 by 16 byte tile."""
    assert pointer.alignment >= 16
    result = nvvm.ldmatrix(
        pointer.llvm_ptr,
        2,
        nvvm.MMALayout.col,
        shape=nvvm.LoadShape.M16N16,
        src_format=nvvm.LoadSrcFormat.B8,
        results=[cast("Any", llvm).StructType.get_literal([T.i32(), T.i32()])],
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Int32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(2)
    )


@cute.jit
def _scale_pair_with_factors(
    values: cutlass.Int32,
    first: cutlass.Float32,
    second: cutlass.Float32,
    post_scale_log2: cutlass.Constexpr = 0,
) -> cutlass.Int32:
    """Scale the low two signed bytes and pack their rounded BF16 values."""
    low = cutlass.Float32(cutlass.Int32(values << 24) >> 24) * first
    high = cutlass.Float32(cutlass.Int32(values << 16) >> 24) * second
    return _pack_postscaled_bf16(low, high, post_scale_log2)


@dsl_user_op
def _select_scale_pair(
    first: cutlass.Int32,
    second: cutlass.Int32,
    selector: cutlass.Int32,
    *,
    loc: object = None,
    ip: object = None,
) -> cutlass.Int32:
    """Select one byte from each scale word into the low two result bytes.

    The upper selector nibbles repeat the first input byte. Those result
    bytes are deliberately left unmasked and must be ignored by consumers.
    """
    return cutlass.Int32(
        nvvm.prmt(
            first.ir_value(loc=loc, ip=ip),
            selector.ir_value(loc=loc, ip=ip),
            nvvm.PermuteMode.DEFAULT,
            hi=second.ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _select_scale_group_word(
    words: tuple[cutlass.Int32, ...], group: cutlass.Int32
) -> cutlass.Int32:
    """Pack this warp's scale group from four consecutive key rows."""
    selector = group | ((group + 4) << 4)
    low = _select_scale_pair(words[0], words[1], selector) & 0xFFFF
    high = _select_scale_pair(words[2], words[3], selector) & 0xFFFF
    return cutlass.Int32(low | cutlass.Int32(high << 16))


@cute.jit
def _scale_word_uses_s2f6(
    word: cutlass.Int32, post_scale_log2: cutlass.Constexpr = 0
) -> cutlass.Boolean:
    """Prove four scales support native conversion with packed postscaling.

    The zero-byte test rejects 248..255 using each byte's high five bits.
    Negative corrected codes are clamped and rounded by a BF16 multiply.
    """
    inverse = cutlass.Uint32(~cutlass.Uint32(word) & cutlass.Uint32(0xF8F8F8F8))
    zero_bytes = (inverse - cutlass.Uint32(0x01010101)) & ~inverse
    return (zero_bytes & cutlass.Uint32(0x80808080)) == 0


@cute.jit
def _scale_pairs_use_s2f6(
    first: cutlass.Int32,
    second: cutlass.Int32,
    post_scale_log2: cutlass.Constexpr = 0,
) -> cutlass.Boolean:
    """Prove the finite native domain for four low-half scale bytes.

    Clearing each byte's low three bits and adding eight produces a carry
    exactly for values 248..255. Bit 8 records the low byte's carry because
    the upper byte's masked value+8 is even; bit 16 records the upper carry.
    Upper-half bits are deliberately ignored, as in the conversion itself.
    """
    first_carries = (first & 0xF8F8) + 0x0808
    second_carries = (second & 0xF8F8) + 0x0808
    return ((first_carries | second_carries) & 0x10100) == 0


@cute.jit
def _postscale_pair_bits(
    scales: cutlass.Int32, post_scale_log2: cutlass.Constexpr
) -> tuple[cutlass.Int32, cutlass.Int32]:
    """Prepare independent native scale bytes and BF16 multiplier lanes.

    Caller proves both low bytes <=247 and -22<=p<-6. For c=s+6+p,
    S2F6 uses max(c,0); a packed BF16 multiply by 2**min(c,0) supplies
    the remaining scale. At native scale0, raw*2**-133 is exactly BF16.
    The multiplier is normal BF16, since -16<=min(c,0)<=0, and native
    BF16 multiplication preserves the required subnormal RNE and zero sign.
    Extract each byte before correction to prevent cross-byte borrows;
    upper input bits are intentionally ignored as in the conversion itself.
    """
    first = cutlass.Int32(scales & 255) + (6 + post_scale_log2)
    second = cutlass.Int32(cutlass.Int32(scales >> 8) & 255) + (6 + post_scale_log2)
    first_native = cutlass.Int32(cutlass.max(first, 0))
    second_native = cutlass.Int32(cutlass.max(second, 0))
    corrected = cutlass.Int32(first_native | cutlass.Int32(second_native << 8))
    first_factor = cutlass.Int32((127 + cutlass.min(first, 0)) << 7)
    second_factor = cutlass.Int32((127 + cutlass.min(second, 0)) << 7)
    factors = cutlass.Int32(first_factor | cutlass.Int32(second_factor << 16))
    return corrected, factors


@dsl_user_op
def _packed_scale_layout_supported(
    scales: cute.Tensor, *, loc: object = None, ip: object = None
) -> bool:
    """The validated 128-key scale packet layout; other views use indexed loads."""
    scale_base = scales.iterator
    return (
        isinstance(scale_base, cute.Pointer)
        and scales.element_type in (cutlass.Int8, cutlass.Uint8)
        and not isinstance(scales.layout.type, cast("Any", cute_ir).ComposedLayoutType)
        and scales.shape == (128, 4)
        and scales.stride == (4, 1)
        and scale_base.memspace is cute.AddressSpace.smem
        and scale_base.alignment >= 16
    )


def packed_scale_tile_layout(
    scales: cute.Tensor,
    width: int,
    scale_group: int,
    post_scale_log2: int,
) -> bool:
    """Whether a conversion has the packed four-scales-per-row tile: 128
    dimensions in groups of 32 whose four scale bytes are one word of the
    staged 512-byte scale tile, and a source postscale of at least
    ``2**-6``. ``width`` is the tile's extent along the axis the scales
    group (K's columns, and for V four warps of 32 dimensions).

    With that postscale the corrected exponent of a native conversion is
    the scale byte plus zero to six: never negative, so no conversion
    needs a second multiply, and for a byte of at most 247 never above
    253. The forms that rest on it (one correction per row or lane, a V
    tile converted ahead of its scale vote) are for this layout alone;
    every other layout keeps the code it had.
    """
    return (
        width == 128
        and scale_group == 32
        and post_scale_log2 >= -6
        and _packed_scale_layout_supported(scales)
    )


@cute.jit
def _scale_word_at_least(
    word: cutlass.Int32, minimum: cutlass.Constexpr
) -> cutlass.Boolean:
    """Prove every scale byte meets a positive native-code lower bound."""
    assert cutlass.const_expr(1 <= minimum <= 16)
    unsigned = cutlass.Uint32(word)
    # A borrow can propagate only after an already-failing lower byte.
    # If every byte meets the bound, all four subtractions are independent.
    below = (unsigned - cutlass.Uint32(minimum * 0x01010101)) & ~unsigned
    return (below & cutlass.Uint32(0x80808080)) == 0


@cute.jit
def _stage_scaled_s8_transpose_tiles(
    raw: cute.Tensor,
    scales: cute.Tensor,
    dimension_base: cutlass.Int32,
    lane: cutlass.Int32,
    tmem_base: cutlass.Int32,
    keys: cutlass.Constexpr,
    packed_scales: cutlass.Constexpr,
    safe_tile: cutlass.Constexpr,
    post_scale_log2: cutlass.Constexpr,
    native_scale_only: cutlass.Constexpr = False,
    cached_group_word: cutlass.Int32 | None = None,
    positive_zero: cutlass.Constexpr = False,
    packed_tile: cutlass.Constexpr = False,
) -> None:
    """Stage key blocks with a static whole-tile or dynamic packet route.

    True requires the wrapper's full-tile proof; False conservatively uses
    full-domain arithmetic. None retains each packet's original warp vote.
    Both static modes are called only under a warp-uniform outer branch.
    native_scale_only=True additionally requires the wrapper to prove every
    scale >= -6-p. Corrected codes are then nonnegative, packed correction
    cannot borrow across bytes, and the identity multiply can be skipped.
    cached_group_word comes from _select_scale_group_word on this lane's
    lane*16-byte packet, using this dimension_base//32 group; the complete
    warp must remain converged while exchanging those words.
    positive_zero applies the source's final BF16 +0, except where the native
    finite conversion proof excludes negative zero and makes that add an identity.

    ``packed_tile`` (``packed_scale_tile_layout``) gives the native arm
    (``safe_tile=True``) its exponent correction once per lane: one add over
    the lane's cached group word ahead of the key blocks, which then
    exchange corrected words, and one byte permute for a block's upper
    pair. The arm converts with the scale pairs it had wherever every byte
    of the warp's group is at most 247. Where one is not, a pair may hold
    another value than the per-block correction gave; the caller runs the
    arm there only ahead of the vote that replaces what it staged
    (``_stage_guarded_s8_transpose_tiles``).
    """
    assert cutlass.const_expr(
        not native_scale_only
        or (safe_tile is True and packed_scales and -22 <= post_scale_log2 < -6)
    )
    assert cutlass.const_expr(
        cached_group_word is None or (keys == 128 and packed_scales)
    )
    corrected_once = packed_tile and safe_tile is True
    assert cutlass.const_expr(
        not corrected_once or (cached_group_word is not None and not native_scale_only)
    )
    exchanged_word = cached_group_word
    if cutlass.const_expr(corrected_once):
        assert cached_group_word is not None
        exchanged_word = _corrected_scale_word(cached_group_word, post_scale_log2)
    for key_tile in cutlass.range_constexpr(keys // 16):
        first_key = key_tile * 16 + 4 * (lane % 4)
        scale_group = dimension_base // 32
        if cutlass.const_expr(cached_group_word is not None):
            # Each lane cached four key rows from this exact scales/group.
            # The owner below is first_key // 4 within that 128-row load.
            owner = 4 * key_tile + lane % 4
            scales01 = cute.arch.shuffle_sync(
                exchanged_word, owner, mask=-1, mask_and_clamp=31
            )
            if cutlass.const_expr(corrected_once):
                scales23 = _permute_scale_bytes(scales01, 0x3232)
            else:
                scales23 = cutlass.Int32(scales01 >> 16)
        elif cutlass.const_expr(packed_scales):
            # first_key is a multiple of 4: its four scale words form one
            # aligned 16-byte packet, wholly inside the 128-row tile.
            scale_offset = cute.crd2idx((first_key, 0), scales.layout)
            pointer = cute.recast_ptr(
                scales.iterator + cute.assume(scale_offset, divby=16),
                dtype=cutlass.Int32,
            )
            words = _load_words4(pointer)
            selector = scale_group | ((scale_group + 4) << 4)
            scales01 = _select_scale_pair(words[0], words[1], selector)
            scales23 = _select_scale_pair(words[2], words[3], selector)
        else:
            scale0 = cutlass.Int32(cutlass.Uint8(scales[first_key, scale_group]))
            scale1 = cutlass.Int32(cutlass.Uint8(scales[first_key + 1, scale_group]))
            scale2 = cutlass.Int32(cutlass.Uint8(scales[first_key + 2, scale_group]))
            scale3 = cutlass.Int32(cutlass.Uint8(scales[first_key + 3, scale_group]))
            scales01 = cutlass.Int32(scale0 | (scale1 << 8))
            scales23 = cutlass.Int32(scale2 | (scale3 << 8))
        row = key_tile * 16 + lane % 16
        # All lanes must take the same branch: each branch contains collective
        # TMEM stores. Exceptional input anywhere in the warp uses full-domain
        # arithmetic for every lane, with one factor calculation per scale.
        if cutlass.const_expr(safe_tile is not None):
            safe = safe_tile
        else:
            safe = cute.arch.vote_all_sync(
                _scale_pairs_use_s2f6(scales01, scales23, post_scale_log2)
            )
        # In the two static modes, canonicalization folds this constant branch.
        if safe:
            if cutlass.const_expr(post_scale_log2 < -6 and not native_scale_only):
                corrected01, factors01 = _postscale_pair_bits(scales01, post_scale_log2)
                corrected23, factors23 = _postscale_pair_bits(scales23, post_scale_log2)
            elif cutlass.const_expr(corrected_once):
                # The lane's word was corrected ahead of the key blocks.
                corrected01 = scales01
                corrected23 = scales23
                factors01 = cutlass.Int32(0x3F803F80)
                factors23 = cutlass.Int32(0x3F803F80)
            else:
                corrected01 = scales01 + (6 + post_scale_log2) * 0x0101
                corrected23 = scales23 + (6 + post_scale_log2) * 0x0101
                # Dead in this constexpr arm; keep definite assignment explicit.
                factors01 = cutlass.Int32(0x3F803F80)
                factors23 = cutlass.Int32(0x3F803F80)
            for half in cutlass.range_constexpr(2):
                dimension = dimension_base + 16 * half
                offset = cute.crd2idx((row, dimension), raw.layout)
                values = _load_transposed_s8_16x16(
                    raw.iterator + cute.assume(offset, divby=16)
                )
                packed = cutlass.Array(cutlass.Int32, 4, alignment=16)
                packed[0] = _scaled_s2f6x2_to_bf16x2(values[0], corrected01)
                packed[1] = _scaled_s2f6x2_to_bf16x2(values[0] >> 16, corrected23)
                packed[2] = _scaled_s2f6x2_to_bf16x2(values[1], corrected01)
                packed[3] = _scaled_s2f6x2_to_bf16x2(values[1] >> 16, corrected23)
                if cutlass.const_expr(post_scale_log2 < -6 and not native_scale_only):
                    packed[0] = mul_bf16x2(packed[0], factors01)
                    packed[1] = mul_bf16x2(packed[1], factors23)
                    packed[2] = mul_bf16x2(packed[2], factors01)
                    packed[3] = mul_bf16x2(packed[3], factors23)
                # Native finite conversion produces +0 for a zero byte and
                # a nonzero BF16 value otherwise: s+p >= -6 bounds its minimum
                # magnitude by 2**-133. Only the negative-correction route can
                # round a negative byte to -0 and still needs the source add.
                if cutlass.const_expr(
                    positive_zero and post_scale_log2 < -6 and not native_scale_only
                ):
                    for packed_index in cutlass.range_constexpr(4):
                        packed[packed_index] = add_bf16x2(
                            packed[packed_index], cutlass.Int32(0)
                        )
                address = _tmem_store_address(dimension, tmem_base + key_tile * 8)
                destination = cutlass.inttoptr(address, 6, cutlass.Int32)
                prims.tcgen05_st("16x256b", destination, packed[0:4])
        else:
            factor0 = cute.math.exp2(
                cutlass.Float32(scales01 & 255) - 127.0, fastmath=False
            )
            factor1 = cute.math.exp2(
                cutlass.Float32(cutlass.Int32(scales01 >> 8) & 255) - 127.0,
                fastmath=False,
            )
            factor2 = cute.math.exp2(
                cutlass.Float32(scales23 & 255) - 127.0, fastmath=False
            )
            factor3 = cute.math.exp2(
                cutlass.Float32(cutlass.Int32(scales23 >> 8) & 255) - 127.0,
                fastmath=False,
            )
            for half in cutlass.range_constexpr(2):
                dimension = dimension_base + 16 * half
                offset = cute.crd2idx((row, dimension), raw.layout)
                values = _load_transposed_s8_16x16(
                    raw.iterator + cute.assume(offset, divby=16)
                )
                packed = cutlass.Array(cutlass.Int32, 4, alignment=16)
                packed[0] = _scale_pair_with_factors(
                    values[0], factor0, factor1, post_scale_log2
                )
                packed[1] = _scale_pair_with_factors(
                    values[0] >> 16, factor2, factor3, post_scale_log2
                )
                packed[2] = _scale_pair_with_factors(
                    values[1], factor0, factor1, post_scale_log2
                )
                packed[3] = _scale_pair_with_factors(
                    values[1] >> 16, factor2, factor3, post_scale_log2
                )
                if cutlass.const_expr(positive_zero):
                    for packed_index in cutlass.range_constexpr(4):
                        packed[packed_index] = add_bf16x2(
                            packed[packed_index], cutlass.Int32(0)
                        )
                address = _tmem_store_address(dimension, tmem_base + key_tile * 8)
                destination = cutlass.inttoptr(address, 6, cutlass.Int32)
                prims.tcgen05_st("16x256b", destination, packed[0:4])


@dsl_user_op
def _zero_bf16_pairs_from_s8_word(
    values: cutlass.Int32,
    *,
    loc: object = None,
    ip: object = None,
) -> tuple[cutlass.Int32, cutlass.Int32]:
    """Four signed raw bytes become four signed BF16 zeros: one mask, two PRMT."""
    mask = cutlass.Uint32(0x80808080).ir_value(loc=loc, ip=ip)
    signs = arith.andi(values.ir_value(loc=loc, ip=ip), mask, loc=loc, ip=ip)
    zero = cutlass.Int32(0).ir_value(loc=loc, ip=ip)
    low = nvvm.prmt(
        signs,
        cutlass.Int32(0x1404).ir_value(loc=loc, ip=ip),
        nvvm.PermuteMode.DEFAULT,
        hi=zero,
        loc=loc,
        ip=ip,
    )
    high = nvvm.prmt(
        signs,
        cutlass.Int32(0x3424).ir_value(loc=loc, ip=ip),
        nvvm.PermuteMode.DEFAULT,
        hi=zero,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(low), cutlass.Int32(high)


@cute.jit
def _stage_zero_s8_transpose_tiles(
    raw: cute.Tensor,
    dimension_base: cutlass.Int32,
    lane: cutlass.Int32,
    tmem_base: cutlass.Int32,
    keys: cutlass.Constexpr,
    positive_zero: cutlass.Constexpr = False,
) -> None:
    """Stage a proved zero tile, retaining signs unless source explicitly adds +0."""
    if cutlass.const_expr(positive_zero):
        assert cutlass.const_expr(keys == 128)
        stage_positive_zero_value_tile(dimension_base, tmem_base)
    else:
        for key_tile in cutlass.range_constexpr(keys // 16):
            row = key_tile * 16 + lane % 16
            for half in cutlass.range_constexpr(2):
                dimension = dimension_base + 16 * half
                offset = cute.crd2idx((row, dimension), raw.layout)
                values = _load_transposed_s8_16x16(
                    raw.iterator + cute.assume(offset, divby=16)
                )
                low0, high0 = _zero_bf16_pairs_from_s8_word(values[0])
                low1, high1 = _zero_bf16_pairs_from_s8_word(values[1])
                packed = cutlass.Array(cutlass.Int32, 4, alignment=16)
                packed[0] = low0
                packed[1] = high0
                packed[2] = low1
                packed[3] = high1
                address = _tmem_store_address(dimension, tmem_base + key_tile * 8)
                destination = cutlass.inttoptr(address, 6, cutlass.Int32)
                prims.tcgen05_st("16x256b", destination, packed[0:4])


@cute.jit
def _stage_voted_s8_transpose_tiles(
    raw: cute.Tensor,
    scales: cute.Tensor,
    dimension_base: cutlass.Int32,
    lane: cutlass.Int32,
    tmem_base: cutlass.Int32,
    keys: cutlass.Constexpr,
    cached_group_word: cutlass.Int32,
    packed_scales: cutlass.Constexpr,
    post_scale_log2: cutlass.Constexpr,
    positive_zero: cutlass.Constexpr,
) -> None:
    """Stage a packed tile (``packed_scale_tile_layout``) with the native
    conversion at once and vote on its scales behind it.

    The guard at the head of a tile (``_stage_guarded_s8_transpose_tiles``)
    holds back every load and conversion of the tile until the warp has
    voted, on the chain that sets a memory-hot stream's page period. Here
    the warp stages the whole tile with the native arm first, sixteen
    stores, and votes behind them. A failed vote runs the full-domain arm
    over the same sixteen column groups: its stores follow the native
    arm's, behind a store wait, and replace them. The caller publishes the
    stage only behind this function (its store wait, fence and arrival), so
    no reader sees the first pass.

    The vote covers the bytes this warp's conversion reads, its own scale
    group of the tile's 128 keys, and not the other three groups'. Where
    it passes, every one of those bytes is at most 247 and the native arm
    converted with the scale pairs the guarded arm corrects per block
    (``_stage_scaled_s8_transpose_tiles``). The guard at the head sends
    such a warp through the full-domain arm when another group holds a
    byte above 247. Both arms are exact for a byte of at most 247 with a
    postscale of at least ``2**-6``: the product of a signed byte and a
    power of two of at least ``2**-133`` is a BF16 value, which both
    produce, a zero byte as +0 (the full-domain arm's rounded product and
    postscale have nothing to round; mixed warps already send in-range
    rows of K through it). So the staged tile is the one the guard at the
    head stages, for every scale byte and every raw byte.

    Where the vote fails the native arm has converted with scale bytes
    outside its finite domain, or with pairs a carry of the lane's
    correction changed. The conversion saturates and does not trap, and
    all of it is replaced.
    """
    lane_safe = _scale_word_uses_s2f6(cached_group_word, post_scale_log2)
    _stage_scaled_s8_transpose_tiles(
        raw,
        scales,
        dimension_base,
        lane,
        tmem_base,
        keys,
        packed_scales=packed_scales,
        safe_tile=True,
        post_scale_log2=post_scale_log2,
        cached_group_word=cached_group_word,
        positive_zero=positive_zero,
        packed_tile=True,
    )
    if not cute.arch.vote_all_sync(lane_safe):
        # The vote is warp-uniform; the join keeps the collectives below
        # convergent for NVVM, as in the row decoder.
        cute.arch.sync_warp()
        # Two asynchronous stores to one column group: the replacement is
        # issued behind the completion of the first pass, so its order does
        # not rest on the order the device completes stores in.
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        _stage_scaled_s8_transpose_tiles(
            raw,
            scales,
            dimension_base,
            lane,
            tmem_base,
            keys,
            packed_scales=packed_scales,
            safe_tile=False,
            post_scale_log2=post_scale_log2,
            cached_group_word=cached_group_word,
            positive_zero=positive_zero,
        )


@cute.jit
def _stage_guarded_s8_transpose_tiles(
    raw: cute.Tensor,
    scales: cute.Tensor,
    dimension_base: cutlass.Int32,
    lane: cutlass.Int32,
    tmem_base: cutlass.Int32,
    keys: cutlass.Constexpr,
    tile_words: tuple[cutlass.Int32, ...],
    packed_scales: cutlass.Constexpr,
    post_scale_log2: cutlass.Constexpr,
    positive_zero: cutlass.Constexpr = False,
) -> None:
    """Prove complete-tile scale eligibility or use mixed/exceptional fallback."""
    cached_group_word = _select_scale_group_word(tile_words, dimension_base // 32)
    lane_safe = _scale_word_uses_s2f6(
        tile_words[0], post_scale_log2
    ) & _scale_word_uses_s2f6(tile_words[1], post_scale_log2)
    lane_safe = (
        lane_safe
        & _scale_word_uses_s2f6(tile_words[2], post_scale_log2)
        & _scale_word_uses_s2f6(tile_words[3], post_scale_log2)
    )
    tile_scales_safe = cute.arch.vote_all_sync(lane_safe)
    native_tile = cutlass.Boolean(False)
    if cutlass.const_expr(post_scale_log2 < -6):
        lane_native = cutlass.Boolean(True)
        for index in cutlass.range_constexpr(4):
            lane_native = lane_native & _scale_word_at_least(
                tile_words[index], -6 - post_scale_log2
            )
        native_tile = tile_scales_safe & cute.arch.vote_all_sync(lane_native)
    # Both votes are warp-uniform and dominate every collective in one
    # specialized key loop. Generic layouts retain their packet checks.
    if native_tile:
        _stage_scaled_s8_transpose_tiles(
            raw,
            scales,
            dimension_base,
            lane,
            tmem_base,
            keys,
            packed_scales=packed_scales,
            safe_tile=True,
            post_scale_log2=post_scale_log2,
            native_scale_only=post_scale_log2 < -6,
            cached_group_word=cached_group_word,
            positive_zero=positive_zero,
        )
    elif tile_scales_safe:
        _stage_scaled_s8_transpose_tiles(
            raw,
            scales,
            dimension_base,
            lane,
            tmem_base,
            keys,
            packed_scales=packed_scales,
            safe_tile=True,
            post_scale_log2=post_scale_log2,
            cached_group_word=cached_group_word,
            positive_zero=positive_zero,
        )
    else:
        _stage_scaled_s8_transpose_tiles(
            raw,
            scales,
            dimension_base,
            lane,
            tmem_base,
            keys,
            packed_scales=packed_scales,
            safe_tile=False,
            post_scale_log2=post_scale_log2,
            cached_group_word=cached_group_word,
            positive_zero=positive_zero,
        )


@cute.jit
def stage_scaled_s8_transpose_to_tmem(
    raw: cute.Tensor,
    scales: cute.Tensor,
    dimension_base: cutlass.Int32,
    lane: cutlass.Int32,
    tmem_base: cutlass.Int32,
    keys: cutlass.Constexpr,
    post_scale_log2: cutlass.Constexpr = 0,
    positive_zero: cutlass.Constexpr = False,
    packed_tile_forms: cutlass.Constexpr = False,
) -> None:
    """One warp decodes 32 dimensions of V and stages their transpose in TMEM.

    The shared raw tile has [key, dimension] coordinates and groups of 32
    dimensions share a UINT8 scale. Each LDSM byte fragment holds four keys
    at two dimensions. Decoding each pair gives exactly the four-word input
    of a 16x256b TMEM store, without a converted shared-memory transpose.
    The caller proves complete 16-key blocks, 32-aligned dimension_base,
    aligned LDSM addresses and the matching warp-to-TMEM row quarter.
    The raw base carries at least 16-byte alignment; packet offsets retain
    their proved divisibility in the pointer type.
    With a packed 128-key scale tile, an exceptional scale conservatively
    selects full-domain conversion for the complete tile in every V warp.
    A proved postscale zero domain transports only the raw sign bits, or writes
    constant +0 without reading raw bytes when the source explicitly adds +0.
    In particular, ``dimension_base // 32`` must match the physical warp
    index modulo four; looping a single warp over row offsets is invalid.

    ``packed_tile_forms``: the caller takes the forms of the packed
    four-scales-per-row tile where this call has that layout
    (``packed_scale_tile_layout``): the native conversion of the whole tile
    ahead of a vote on the warp's own scale group, and its exponent
    correction once per lane (``_stage_voted_s8_transpose_tiles``). The
    staged V is the same for every input.
    """
    # The production rings have this exact staged layout and 16-byte-aligned
    # slot offsets. Other shared views retain scalar byte loads.
    assert cutlass.const_expr(-22 <= post_scale_log2 <= 0)
    packed_scales = _packed_scale_layout_supported(scales)
    # One 16-byte load per lane spans the complete 512-byte staged scale tile.
    # Every V warp proves its own full tile, so no cross-warp barrier is needed.
    whole_tile_scale_guard = packed_scales and keys == 128
    # A V warp's tile is 32 dimensions, one scale group, of 128 keys.
    packed_tile = packed_tile_forms and packed_scale_tile_layout(
        scales, keys, 32, post_scale_log2
    )
    assert cutlass.const_expr(not packed_tile or whole_tile_scale_guard)
    if cutlass.const_expr(whole_tile_scale_guard):
        pointer = cute.recast_ptr(
            scales.iterator + cute.assume(lane * 16, divby=16), dtype=cutlass.Int32
        )
        tile_words = _load_words4(pointer)
        if cutlass.const_expr(post_scale_log2 <= -14):
            # For -22 <= p <= -14 and s <= -14-p, even raw -128 is at
            # most half the minimum BF16 subnormal after source scaling.
            # RNE produces signed zero. OR is an unsigned per-byte upper
            # bound; mixed low codes may conservatively keep the decoder.
            merged = tile_words[0] | tile_words[1] | tile_words[2] | tile_words[3]
            lane_zero = cutlass.Boolean(True)
            for byte in cutlass.range_constexpr(4):
                scale = cutlass.Int32(merged >> (8 * byte)) & 255
                lane_zero = lane_zero & (scale <= -14 - post_scale_log2)
            zero_tile = cute.arch.vote_all_sync(lane_zero)
            # Publish the warp-uniform branch as a convergence point before
            # the collective LDSM/TMEM operations, as in the row decoder.
            if zero_tile:
                cute.arch.sync_warp()
                _stage_zero_s8_transpose_tiles(
                    raw,
                    dimension_base,
                    lane,
                    tmem_base,
                    keys,
                    positive_zero=positive_zero,
                )
            else:
                cute.arch.sync_warp()
                _stage_guarded_s8_transpose_tiles(
                    raw,
                    scales,
                    dimension_base,
                    lane,
                    tmem_base,
                    keys,
                    tile_words,
                    packed_scales,
                    post_scale_log2,
                    positive_zero=positive_zero,
                )
        elif cutlass.const_expr(packed_tile):
            _stage_voted_s8_transpose_tiles(
                raw,
                scales,
                dimension_base,
                lane,
                tmem_base,
                keys,
                _select_scale_group_word(tile_words, dimension_base // 32),
                packed_scales,
                post_scale_log2,
                positive_zero,
            )
        else:
            _stage_guarded_s8_transpose_tiles(
                raw,
                scales,
                dimension_base,
                lane,
                tmem_base,
                keys,
                tile_words,
                packed_scales,
                post_scale_log2,
                positive_zero=positive_zero,
            )
    else:
        _stage_scaled_s8_transpose_tiles(
            raw,
            scales,
            dimension_base,
            lane,
            tmem_base,
            keys,
            packed_scales=packed_scales,
            safe_tile=None,
            post_scale_log2=post_scale_log2,
            positive_zero=positive_zero,
        )


@cute.jit
def make_paged_tma_operands(
    raw: cute.Tensor,
    scales: cute.Tensor,
    raw_shape: cutlass.Constexpr,
    raw_strides: cutlass.Constexpr,
    scale_shape: cutlass.Constexpr,
    scale_strides: cutlass.Constexpr,
    key_width: cutlass.Constexpr,
) -> tuple[cuda.TensorMap, cuda.TensorMap]:
    """Build descriptors from the transport plan and named runtime view pointers.

    Four contiguous scale bytes per key form one INT32 descriptor element;
    TMA preserves their bits. The planner proves alignment and every stride.
    """
    # Argument layouts may expose even stride1 as a dynamic host scalar.
    # Tensor-map construction needs a static leading mode; admission proved
    # these two contiguous axes, so retain that fact in the descriptor view.
    raw_view = cute.make_tensor(
        raw.iterator, cute.make_layout(raw_shape, stride=(1, *raw_strides[1:]))
    )
    scale_view = cute.make_tensor(
        cute.recast_ptr(scales.iterator, dtype=cutlass.Int32),
        cute.make_layout(scale_shape, stride=(1, *scale_strides[1:])),
    )
    raw_map = cuda.create_tensor_map_tiled_from_view(
        raw_view,
        box_dims=(128, key_width, 1, 1, 1),
        stride_order=(0, 1, 2, 3, 4),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    scale_map = cuda.create_tensor_map_tiled_from_view(
        scale_view,
        box_dims=(key_width, 1, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    return raw_map, scale_map


def _tma_l2_policy(hint: str) -> cutlass.Int64 | None:
    """Materialize a selected policy while staging a TMA issue."""
    if hint == "evict_first":
        return cast("cutlass.Int64", l2_policy_evict_first())
    if hint == "evict_last":
        return cast("cutlass.Int64", l2_policy_evict_last())
    assert hint == ""
    return None


@cute.jit
def enqueue_paged_tma_operand(
    raw_map: cuda.TensorMap,
    scale_map: cuda.TensorMap,
    raw_destination: cute.Pointer,
    scale_destination: cute.Pointer,
    ready: cute.Pointer,
    page: cutlass.Int32,
    subpage: cutlass.Int32,
    head: cutlass.Int32,
    key_width: cutlass.Constexpr,
    l2_hints: cutlass.Constexpr = ("", ""),
    include_scales: cutlass.Constexpr = True,
    key: cutlass.Int32 = cutlass.Int32(0),  # noqa: B008
) -> None:
    """Issue raw bytes and optional scales with one matching transaction count.

    key is the page key held by the first row of both boxes, a multiple of
    four so the packed scale words stay 16-byte aligned. Rows whose key is
    outside the descriptor's key extent are not fetched from their own
    addresses, are filled with zeros and still count toward the transaction.
    """
    raw_policy = _tma_l2_policy(l2_hints[0])
    scale_policy = raw_policy
    if cutlass.const_expr(include_scales and l2_hints[0] != l2_hints[1]):
        scale_policy = _tma_l2_policy(l2_hints[1])
    prims.mbarrier_arrive_expect_tx(ready, key_width * (132 if include_scales else 128))
    prims.cp_async_bulk_tensor_shared_cta_global(
        raw_destination,
        raw_map.get_ptr(),
        [cutlass.Int32(0), key, head, subpage, page],
        ready,
        l2_cache_hint=raw_policy,
    )
    if cutlass.const_expr(include_scales):
        prims.cp_async_bulk_tensor_shared_cta_global(
            scale_destination,
            scale_map.get_ptr(),
            [key, head, subpage, page],
            ready,
            l2_cache_hint=scale_policy,
        )


@cute.jit
def make_paged_query_map(
    query: cute.Tensor,
    shape: cutlass.Constexpr,
    strides: cutlass.Constexpr,
    box: cutlass.Constexpr,
) -> cuda.TensorMap:
    view = cute.make_tensor(query.iterator, cute.make_layout(shape, stride=strides))
    return cuda.create_tensor_map_tiled_from_view(
        view,
        box_dims=box,
        stride_order=(0, 1, 2, 3, 4),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


@cute.jit
def enqueue_paged_query(
    query_map: cuda.TensorMap,
    destination: cute.Pointer,
    ready: cute.Pointer,
    request: cutlass.Int32,
    first_head: cutlass.Int32,
    columns: cutlass.Constexpr,
) -> None:
    # Four disjoint boxes cover both 128-dimensional query panels, including
    # hardware zero-fill of any rows beyond the descriptor's logical extent.
    prims.mbarrier_arrive_expect_tx(ready, 2 * columns * 128 * 2)
    for head in cutlass.range_constexpr(2):
        for half in cutlass.range_constexpr(2):
            offset = head * columns * 128 + half * columns * 64
            prims.cp_async_bulk_tensor_shared_cta_global(
                destination + cute.assume(offset, divby=512),
                query_map.get_ptr(),
                [
                    cutlass.Int32(half * 64),
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    first_head + head,
                    request,
                ],
                ready,
            )
