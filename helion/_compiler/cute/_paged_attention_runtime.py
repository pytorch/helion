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

if TYPE_CHECKING:
    from typing import Any


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


@cute.jit
def scaled_s8_to_bf16_words(
    words: cute.Tensor,
    scale_byte: cutlass.Uint8,
    scale_codes_safe: cutlass.Boolean | None = None,
) -> cute.Tensor:
    """Decode packed INT8 words to twice as many packed BF16 words.

    A single scale read and branch serves the whole packet. The fast branch
    compensates S2F6's six fractional bits in the exponent. The exceptional
    scale range uses the source expression's FP32 multiply and BF16 rounding:
    UE8M0's special byte and SATFINITE must not replace those semantics.
    This includes INT8 -128 and scale zero, and assumes no workload-specific
    restriction on either byte's value.
    Passing scale_codes_safe=True must prove this invocation's scale is at
    most 247. False conservatively forces the full-domain fallback; ordinary
    callers retain their original per-scale check.
    """
    result = cute.make_rmem_tensor((2 * cute.size(words),), cutlass.Int32)
    scale = cutlass.Int32(scale_byte)
    if cutlass.const_expr(scale_codes_safe is None):
        safe = scale <= 247
    else:
        safe = scale_codes_safe
    if safe:
        corrected = scale + 6
        paired = corrected | (corrected << 8)
        for index in cutlass.range_constexpr(cute.size(words)):
            word = cutlass.Int32(words[index])
            result[2 * index] = _scaled_s2f6x2_to_bf16x2(word, paired)
            result[2 * index + 1] = _scaled_s2f6x2_to_bf16x2(word >> 16, paired)
    else:
        factor = cute.math.exp2(cutlass.Float32(scale) - 127.0, fastmath=False)
        for index in cutlass.range_constexpr(cute.size(words)):
            word = cutlass.Int32(words[index])
            low = cutlass.Float32(cutlass.Int32(word << 24) >> 24) * factor
            high = cutlass.Float32(cutlass.Int32(word << 16) >> 24) * factor
            result[2 * index] = pack_bf16x2(low, high)
            low = cutlass.Float32(cutlass.Int32(word << 8) >> 24) * factor
            high = cutlass.Float32(word >> 24) * factor
            result[2 * index + 1] = pack_bf16x2(low, high)
    return result


@dsl_user_op
def _load_words4(
    pointer: cute.Pointer,
    *,
    loc: object = None,
    ip: object = None,
) -> tuple[cutlass.Int32, ...]:
    """A sixteen-byte shared-memory load at a proven aligned address."""
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


@cute.jit
def stage_scaled_s8_rows_to_tmem(
    raw: cute.Tensor,
    scales: cute.Tensor,
    row: cutlass.Int32,
    tmem_base: cutlass.Int32,
    columns: cutlass.Constexpr,
    scale_group: cutlass.Constexpr,
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
    An exceptional packed scale conservatively selects full-domain conversion
    for every group in that row.
    Raw and scale tensors carry their actual shared layouts (including the
    raw TMA swizzle), so no global page pitch is baked into the conversion.
    The raw base carries at least 16-byte alignment; packet offsets retain
    their proved divisibility in the pointer type.
    Completion/fences and ready/free barriers belong to the owning pipeline.
    """
    warp_row = row // 32 * 32
    packed_scales = columns // scale_group == 4 and _packed_scale_layout_supported(
        scales
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
        row_scales_safe = _scale_word_uses_s2f6(scale_word)
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
            if cutlass.const_expr(packed_scales):
                converted = scaled_s8_to_bf16_words(words, scale, row_scales_safe)
            else:
                converted = scaled_s8_to_bf16_words(words, scale)
            for index in cutlass.range_constexpr(scale_group // 2):
                packed[inner * (scale_group // 2) + index] = converted[index]
        address = (warp_row << 16) | (tmem_base + block * store_words)
        destination = cutlass.inttoptr(address, 6, cutlass.Int32)
        prims.tcgen05_st("32x32b", destination, packed[0:store_words])


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
        cast("Any", llvm).StructType.get_literal([T.i32(), T.i32()]),
        pointer.llvm_ptr,
        2,
        nvvm.MMALayout.col,
        shape=nvvm.LoadShape.M16N16,
        src_format=nvvm.LoadSrcFormat.B8,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Int32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(2)
    )


@cute.jit
def _scale_pair_with_factors(
    values: cutlass.Int32, first: cutlass.Float32, second: cutlass.Float32
) -> cutlass.Int32:
    """Scale the low two signed bytes and pack their rounded BF16 values."""
    low = cutlass.Float32(cutlass.Int32(values << 24) >> 24) * first
    high = cutlass.Float32(cutlass.Int32(values << 16) >> 24) * second
    return cutlass.Int32(pack_bf16x2(low, high))


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
def _scale_word_uses_s2f6(word: cutlass.Int32) -> cutlass.Boolean:
    """Reject a word containing any exceptional scale byte in 248..255.

    The zero-byte test checks the inverted high five bits of all four bytes.
    """
    inverse = cutlass.Uint32(~cutlass.Uint32(word) & cutlass.Uint32(0xF8F8F8F8))
    zero_bytes = (inverse - cutlass.Uint32(0x01010101)) & ~inverse
    return (zero_bytes & cutlass.Uint32(0x80808080)) == 0


@cute.jit
def _scale_pairs_use_s2f6(
    first: cutlass.Int32, second: cutlass.Int32
) -> cutlass.Boolean:
    """Whether all four low-half scale bytes permit the exponent+6 route.

    Clearing each byte's low three bits and adding eight produces a carry
    exactly for values 248..255. Bit 8 records the low byte's carry because
    the upper byte's masked value+8 is even; bit 16 records the upper carry.
    Upper-half bits are deliberately ignored, as in the conversion itself.
    """
    first_carries = (first & 0xF8F8) + 0x0808
    second_carries = (second & 0xF8F8) + 0x0808
    return ((first_carries | second_carries) & 0x10100) == 0


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


@cute.jit
def stage_scaled_s8_transpose_to_tmem(
    raw: cute.Tensor,
    scales: cute.Tensor,
    dimension_base: cutlass.Int32,
    lane: cutlass.Int32,
    tmem_base: cutlass.Int32,
    keys: cutlass.Constexpr,
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
    With a packed 128-key scale tile, any exceptional scale conservatively
    selects full-domain conversion for the complete tile in every V warp.
    In particular, ``dimension_base // 32`` must match the physical warp
    index modulo four; looping a single warp over row offsets is invalid.
    """
    # The production rings have this exact staged layout and 16-byte-aligned
    # slot offsets. Other shared views retain scalar byte loads.
    packed_scales = _packed_scale_layout_supported(scales)
    # One 16-byte load per lane spans the complete 512-byte staged scale tile.
    # Every V warp proves its own full tile, so no cross-warp barrier is needed.
    whole_tile_scale_guard = packed_scales and keys == 128
    tile_scales_safe = cutlass.Boolean(False)
    if cutlass.const_expr(whole_tile_scale_guard):
        pointer = cute.recast_ptr(
            scales.iterator + cute.assume(lane * 16, divby=16), dtype=cutlass.Int32
        )
        tile_words = _load_words4(pointer)
        lane_safe = _scale_word_uses_s2f6(tile_words[0]) & _scale_word_uses_s2f6(
            tile_words[1]
        )
        lane_safe = (
            lane_safe
            & _scale_word_uses_s2f6(tile_words[2])
            & _scale_word_uses_s2f6(tile_words[3])
        )
        tile_scales_safe = cute.arch.vote_all_sync(lane_safe)
    for key_tile in cutlass.range_constexpr(keys // 16):
        first_key = key_tile * 16 + 4 * (lane % 4)
        scale_group = dimension_base // 32
        if cutlass.const_expr(packed_scales):
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
        if cutlass.const_expr(whole_tile_scale_guard):
            safe = tile_scales_safe
        else:
            safe = cute.arch.vote_all_sync(_scale_pairs_use_s2f6(scales01, scales23))
        if safe:
            corrected01 = scales01 + 0x0606
            corrected23 = scales23 + 0x0606
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
                address = (dimension << 16) | (tmem_base + key_tile * 8)
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
                packed[0] = _scale_pair_with_factors(values[0], factor0, factor1)
                packed[1] = _scale_pair_with_factors(values[0] >> 16, factor2, factor3)
                packed[2] = _scale_pair_with_factors(values[1], factor0, factor1)
                packed[3] = _scale_pair_with_factors(values[1] >> 16, factor2, factor3)
                address = (dimension << 16) | (tmem_base + key_tile * 8)
                destination = cutlass.inttoptr(address, 6, cutlass.Int32)
                prims.tcgen05_st("16x256b", destination, packed[0:4])


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
) -> None:
    """One elected producer issues raw bytes and scales into a free ring slot."""
    prims.mbarrier_arrive_expect_tx(ready, key_width * (128 + 4))
    prims.cp_async_bulk_tensor_shared_cta_global(
        raw_destination,
        raw_map.get_ptr(),
        [cutlass.Int32(0), cutlass.Int32(0), head, subpage, page],
        ready,
    )
    prims.cp_async_bulk_tensor_shared_cta_global(
        scale_destination,
        scale_map.get_ptr(),
        [cutlass.Int32(0), head, subpage, page],
        ready,
    )
