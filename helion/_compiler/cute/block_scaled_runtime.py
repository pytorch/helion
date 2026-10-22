"""SM100 native E2M1/UE4M3 and E4M3/UE8M0 block-scaled pipelines.

Prepared operands contain only byte permutations, sign normalization and zero
padding. The native consumer keeps them packed through TMA and shared memory.
MXFP8 operands are read directly from their original K-major tensors; only
their scale bytes need preparation into the native 128-row, four-group layout.
Its producer, MMA and output roles communicate through CUTLASS pipelines. Scale
TMEM is disjoint from the accumulator, including while the epilogue drains it.
"""

from __future__ import annotations

from math import gcd
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

from cuda.bindings.driver import (  # pyrefly: ignore [missing-import]
    CUstream,  # noqa: TC002
)
import cutlass
from cutlass._mlir.dialects import llvm
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import dsl_user_op
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.utils import blackwell_helpers
from cutlass.utils import blockscaled_layout
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout
from cutlass.utils.layout import LayoutEnum

from .block_scaled_config import block_scaled_workspace_shape

if TYPE_CHECKING:
    from ._mlir_compat import ir


@dsl_user_op
def _scale_copy(
    shared: cute.Tensor,
    tensor: cute.Tensor,
    stage: cutlass.Int32,
    group: tcgen05.CtaGroup,
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> None:
    source = cute.filter_zeros(shared, loc=loc, ip=ip)
    target = cute.filter_zeros(tensor, loc=loc, ip=ip)
    atom = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(group), tensor.element_type, loc=loc, ip=ip
    )
    copy = tcgen05.make_s2t_copy(atom, target, loc=loc, ip=ip)
    thread = copy.get_slice(0)
    descriptor = tcgen05.get_s2t_smem_desc_tensor(
        copy, thread.partition_S(source), loc=loc, ip=ip
    )
    cute.copy(
        copy,
        descriptor[None, None, None, None, stage],
        thread.partition_D(target),
        loc=loc,
        ip=ip,
    )


def _compact_a_layouts(
    full: cute.ComposedLayout, logical_rows: int, bk: int
) -> tuple[cute.ComposedLayout, cute.ComposedLayout]:
    """Keep the MMA row view while copying only its live 128-byte K windows."""
    k_stride = cute.get(full.outer.stride, 2)
    # A single 128-byte window has a scalar stride. Wider tiles retain a
    # second K mode whose window stride follows the compact row count.
    if cute.rank(cute.get(full.outer.shape, 2)) == 2:
        k_stride = (cute.get(k_stride, 0), logical_rows * 128)
    strides = (
        cute.get(full.outer.stride, 0),
        cute.get(full.outer.stride, 1),
        k_stride,
        logical_rows * bk,
    )
    mma = cute.make_composed_layout(
        full.inner, 0, cute.make_layout(full.outer.shape, stride=strides)
    )
    load = cute.make_composed_layout(
        full.inner,
        0,
        cute.make_layout(
            (logical_rows, (128, bk // 128), 1),
            stride=(128, (1, logical_rows * 128), 0),
        ),
    )
    return mma, load


@cute.jit
def _transpose_sf_stage_inplace(
    raw_sa: cute.Tensor,
    raw_sb: cute.Tensor,
    stage: cutlass.Int32,
    k_block: cutlass.Int32,
    m_start: cutlass.Int32,
    n_start: cutlass.Int32,
    m: cutlass.Constexpr,
    n: cutlass.Constexpr,
    k: cutlass.Constexpr,
    bk: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    logical_m: cutlass.Constexpr,
) -> None:
    """Read a complete TMA scale stage before permuting its storage in place."""
    lane = cutlass.Int32(cute.arch.thread_idx()[0]) % 32
    k_chunks = bk // 128
    base_a = stage * (128 * k_chunks)
    base_b = stage * (128 * n_chunks * k_chunks)
    words_a = cute.make_rmem_tensor((k_chunks, 4), cutlass.Int32)
    words_b = cute.make_rmem_tensor((k_chunks, n_chunks, 4), cutlass.Int32)
    for chunk in cutlass.range_constexpr(k_chunks):
        word_k = k_block * k_chunks + chunk
        for segment in cutlass.range_constexpr(4):
            row = lane + segment * 32
            value = cutlass.Int32(0)
            if m_start + row < m and row < logical_m and word_k < (k + 127) // 128:
                value = raw_sa[base_a + chunk * 128 + row]
            if cutlass.const_expr(k % 128 != 0):
                if word_k == k // 128:
                    value = value & ((1 << ((k % 128) // 32 * 8)) - 1)
            words_a[chunk, segment] = value
        for n_chunk in cutlass.range_constexpr(n_chunks):
            for segment in cutlass.range_constexpr(4):
                row = lane + segment * 32
                value = cutlass.Int32(0)
                if n_start + n_chunk * 128 + row < n and word_k < (k + 127) // 128:
                    value = raw_sb[
                        base_b + chunk * (128 * n_chunks) + n_chunk * 128 + row
                    ]
                if cutlass.const_expr(k % 128 != 0):
                    if word_k == k // 128:
                        value = value & ((1 << ((k % 128) // 32 * 8)) - 1)
                words_b[chunk, n_chunk, segment] = value
    # B also permutes its K-chunk and N-chunk modes. Preload both complete
    # buffers before any stores, even when these outer modes overlap.
    cute.arch.sync_warp()
    for chunk in cutlass.range_constexpr(k_chunks):
        for segment in cutlass.range_constexpr(4):
            raw_sa[base_a + chunk * 128 + lane * 4 + segment] = words_a[chunk, segment]
        for n_chunk in cutlass.range_constexpr(n_chunks):
            for segment in cutlass.range_constexpr(4):
                raw_sb[
                    base_b
                    + n_chunk * (128 * k_chunks)
                    + chunk * 128
                    + lane * 4
                    + segment
                ] = words_b[chunk, n_chunk, segment]


@dsl_user_op
def _ldmatrix_trans_b8(
    address: cutlass.Int32,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> tuple[cutlass.Uint32, cutlass.Uint32]:
    """``ldmatrix.m16n16.x1.trans.b8`` (sm_100a): lanes 0-15 address the
    sixteen 16-byte rows of a block; lane ``L`` receives rows ``4 (L % 4)``
    to ``4 (L % 4) + 3`` of column ``L // 4`` as the bytes of the first word
    and of column ``L // 4 + 8`` as the second."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal(  # pyrefly: ignore[missing-attribute]
            [cutlass.Uint32.mlir_type, cutlass.Uint32.mlir_type]
        ),
        [address.ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {$0, $1}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    low = llvm.extractvalue(cutlass.Uint32.mlir_type, result, [0], loc=loc, ip=ip)
    high = llvm.extractvalue(cutlass.Uint32.mlir_type, result, [1], loc=loc, ip=ip)
    return cutlass.Uint32(low), cutlass.Uint32(high)


class _LdsmUnit(NamedTuple):
    """One lane's loads of a sixteen-row-by-quad unit (``_ldsm_loads``)."""

    row0: cutlass.Int32
    word0: cutlass.Int32
    word1: cutlass.Int32
    # The scale words as stored and with the padding groups masked.
    stored0: cutlass.Uint32
    stored1: cutlass.Uint32
    scales0: cutlass.Uint32
    scales1: cutlass.Uint32
    low0: cutlass.Uint32
    high0: cutlass.Uint32
    low1: cutlass.Uint32
    high1: cutlass.Uint32


class _TransformSide(NamedTuple):
    """One operand of the fused consumer as the transform warps see it."""

    transpose: bool
    rows: int
    bk: int
    half: cutlass.Int32
    sf_words: cute.Tensor
    sf_pairs: cute.Tensor
    sf_stage_words: int
    words64: cute.Tensor
    words32: cute.Tensor
    stage_bytes: int
    raw_base: cutlass.Int32
    raw_stage_bytes: int


def _transpose_units(rows: int, bk: int) -> int:
    """Units of ``_ldsm_loads``: sixteen rows by one scale quad (32 k-bytes)."""
    return (rows // 16) * (bk // 64)


def _ldsm_loads(
    ttid: cutlass.Int32,
    threads: int,
    rows: int,
    bk: int,
    stage: cutlass.Int32,
    raw_base: cutlass.Int32,
    raw_stage_bytes: int,
    raw_mask: int,
    sf_words: cute.Tensor,
    sf_stage_words: int,
    sf_half: cutlass.Int32,
    tail: tuple[cutlass.Int32 | int, int, int],
) -> list[_LdsmUnit]:
    """Loads of one raw ``[k-byte][row]`` stage (row pitch ``rows`` bytes
    under the TMA swizzle of that width): per unit, the scale words of the
    lane's two rows and two transposed 16x16-byte ldmatrix blocks, which
    give lane ``L`` k-bytes ``16 c + 4 (L % 4) ..`` of rows ``L // 4`` and
    ``L // 4 + 8`` of the unit as 32-bit words.  A warp owns whole units
    (``_transpose_units`` is a multiple of the warps), and all of a stage's
    loads are issued before its first store (``_ldsm_stores``): the
    shared-memory accesses alias for the compiler, so a store would
    serialize the following loads behind the earlier loads' latency.
    ``tail`` is ``(k step, last k step, valid groups of the last quad)``; a
    nonzero last entry masks the padding groups' scale bytes
    (``_mask_tail_scales``)."""
    warp = ttid // 32
    lane = ttid % 32
    warps = threads // 32
    quads = bk // 64
    units = _transpose_units(rows, bk)
    assert units % warps == 0
    ki, last_step, keep = tail
    loads: list[_LdsmUnit] = []
    for item in range(units // warps):
        unit = warp + warps * item
        block = unit // quads
        quad = unit % quads
        row0 = block * 16 + lane // 4
        sf_row0 = sf_half + row0
        sf_row1 = sf_row0 + 8
        # Word w of a 512-byte scale block holds groups 4q .. 4q+3 of
        # row (w // 4) + 32 (w % 4) of a 128-row chunk.
        word0 = stage * sf_stage_words + (
            quad * 128 + (sf_row0 % 32) * 4 + sf_row0 // 32
        )
        word1 = stage * sf_stage_words + (
            quad * 128 + (sf_row1 % 32) * 4 + sf_row1 // 32
        )
        stored0 = cutlass.Uint32(sf_words[word0])
        stored1 = cutlass.Uint32(sf_words[word1])
        scales0 = stored0
        scales1 = stored1
        if keep:
            scales0 = _mask_tail_scales(scales0, quad, ki, last_step, quads - 1, keep)
            scales1 = _mask_tail_scales(scales1, quad, ki, last_step, quads - 1, keep)
        address = (
            raw_base
            + stage * raw_stage_bytes
            + (quad * 32 + lane % 16) * rows
            + block * 16
        )
        address0 = address ^ (((address >> 7) & raw_mask) << 4)
        address = address + 16 * rows
        address1 = address ^ (((address >> 7) & raw_mask) << 4)
        low0, high0 = _ldmatrix_trans_b8(address0)
        low1, high1 = _ldmatrix_trans_b8(address1)
        loads.append(
            _LdsmUnit(
                row0,
                word0,
                word1,
                stored0,
                stored1,
                scales0,
                scales1,
                low0,
                high0,
                low1,
                high1,
            )
        )
    return loads


def _ldsm_stores(
    loads: list[_LdsmUnit],
    ttid: cutlass.Int32,
    threads: int,
    bk: int,
    stage: cutlass.Int32,
    sf_words: cute.Tensor,
    words32: cute.Tensor,
    stage_bytes: int,
    row_bytes: int,
    chunk_mask: int,
) -> None:
    """Stores of ``_ldsm_loads``: each 32-bit word goes to its swizzled
    K-major slot with the sign of its group folded in, and a scale word
    whose stored bytes differ from the unsigned masked ones (signs set, or
    padding groups) is written back by the first of the four lanes of its
    row (the words belong to this warp alone)."""
    warp = ttid // 32
    lane = ttid % 32
    warps = threads // 32
    quads = bk // 64
    sign_mask = cutlass.Uint32(0x88888888)
    sub = (lane % 4) // 2
    for item, unit in enumerate(loads):
        quad = (warp + warps * item) % quads
        words = ((unit.low0, unit.high0), (unit.low1, unit.high1))
        for chunk in range(2):
            low, high = words[chunk]
            shift = 8 * (sub + 2 * chunk) + 7
            low = low ^ (((unit.scales0 >> shift) & cutlass.Uint32(1)) * sign_mask)
            high = high ^ (((unit.scales1 >> shift) & cutlass.Uint32(1)) * sign_mask)
            kb = quad * 32 + chunk * 16 + 4 * (lane % 4)
            offset0 = unit.row0 * row_bytes + kb
            offset0 = offset0 ^ (((offset0 >> 7) & chunk_mask) << 4)
            offset1 = (unit.row0 + 8) * row_bytes + kb
            offset1 = offset1 ^ (((offset1 >> 7) & chunk_mask) << 4)
            words32[(stage * stage_bytes + offset0) // 4] = low
            words32[(stage * stage_bytes + offset1) // 4] = high
        _clear_scale_signs(lane, unit.word0, unit.stored0, unit.scales0, sf_words)
        _clear_scale_signs(lane, unit.word1, unit.stored1, unit.scales1, sf_words)


@cute.jit
def _mask_tail_scales(
    value: cutlass.Uint32,
    quad: cutlass.Int32,
    ki: cutlass.Int32 | int,
    last_step: int,
    last_quad: int,
    keep: int,
) -> cutlass.Uint32:
    """Zero the scale bytes of the groups past K in the last quad of the last
    K step (``keep`` valid groups there): the sources' padding bytes are
    arbitrary, and a NaN scale would poison its row even against the
    zero-filled padding data."""
    if ki == last_step and quad == last_quad:
        value = cutlass.Uint32(value & cutlass.Uint32((1 << (8 * keep)) - 1))
    return value


@cute.jit
def _mask_tail_scale_pair(
    pair: cutlass.Uint64,
    quad: cutlass.Int32,
    ki: cutlass.Int32 | int,
    last_step: int,
    last_quad: int,
    keep: int,
) -> cutlass.Uint64:
    """``_mask_tail_scales`` for a pair of words of one quad."""
    if ki == last_step and quad == last_quad:
        mask = (1 << (8 * keep)) - 1
        pair = cutlass.Uint64(pair & cutlass.Uint64(mask | (mask << 32)))
    return pair


@cute.jit
def _clear_scale_signs(
    lane: cutlass.Int32,
    word: cutlass.Int32,
    stored: cutlass.Uint32,
    masked: cutlass.Uint32,
    sf_words: cute.Tensor,
) -> None:
    """Write scale word ``word`` back as the masked value without its sign
    bits, from the first of the four lanes that share it, when that differs
    from the stored word (signs set, or padding bytes masked): the MMA reads
    the scales from shared memory, not from the lanes' registers."""
    cleaned = cutlass.Uint32(masked & cutlass.Uint32(0x7F7F7F7F))
    if lane % 4 == 0:
        if stored != cleaned:
            sf_words[word] = cleaned


@cute.jit
def _direct_scale_load(
    index: cutlass.Int32,
    guarded: bool,
    stage: cutlass.Int32,
    sf_words: cute.Tensor,
    sf_stage_words: int,
    tail: tuple[cutlass.Int32 | int, int, int],
) -> tuple[cutlass.Uint64, cutlass.Uint64]:
    """Scale words ``2 index`` and ``2 index + 1`` of a direct side's stage as
    one 64-bit value (0 past the end), as stored and with the padding groups
    masked; ``tail`` as in ``_ldsm_loads`` (both words of a pair belong to
    one quad)."""
    ki, last_step, keep = tail
    stored = cutlass.Uint64(0)
    masked = cutlass.Uint64(0)
    if cutlass.const_expr(not guarded) or index < sf_stage_words // 2:
        word = stage * sf_stage_words + 2 * index
        low = cutlass.Uint64(cutlass.Uint32(sf_words[word]))
        high = cutlass.Uint64(cutlass.Uint32(sf_words[word + 1]))
        stored = cutlass.Uint64(low | (high << 32))
        masked = stored
        if cutlass.const_expr(keep != 0):
            masked = _mask_tail_scale_pair(
                stored,
                (2 * index) // 128,
                ki,
                last_step,
                sf_stage_words // 128 - 1,
                keep,
            )
    return stored, masked


@cute.jit
def _direct_scale_fold(
    stored: cutlass.Uint64,
    pair: cutlass.Uint64,
    index: cutlass.Int32,
    stage: cutlass.Int32,
    sf_pairs: cute.Tensor,
    sf_stage_words: int,
    words: cute.Tensor,
    stage_bytes: int,
    row_bytes: int,
    chunk_mask: int,
    half: cutlass.Int32,
    rows: int,
) -> None:
    """Write a direct side's scale word pair back as the masked value without
    its sign bits when that differs from the stored pair, and fold the signs
    into the side's K-major qwords (rows ``half`` .. ``half + rows`` of the
    128-row scale chunk are this tile's).  Word w of a 512-byte scale block
    holds groups 4q .. 4q+3 of row (w // 4) + 32 (w % 4) of a 128-row chunk."""
    cleaned = cutlass.Uint64(pair & cutlass.Uint64(0x7F7F7F7F7F7F7F7F))
    if stored != cleaned:
        sf_pairs[stage * (sf_stage_words // 2) + index] = cleaned
    signs = cutlass.Uint64(pair & cutlass.Uint64(0x8080808080808080))
    if signs != 0:
        for w in cutlass.range_constexpr(2):
            word = 2 * index + w
            quad = word // 128
            position = word % 128
            col = (position // 4) + 32 * (position % 4) - half
            if col >= 0 and col < rows:
                for sub in cutlass.range_constexpr(4):
                    if (signs >> (32 * w + 8 * sub + 7)) & cutlass.Uint64(1):
                        g = quad * 4 + sub
                        offset = col * row_bytes + g * 8
                        offset = offset ^ (((offset >> 7) & chunk_mask) << 4)
                        slot = (stage * stage_bytes + offset) // 8
                        words[slot] = cutlass.Uint64(words[slot]) ^ cutlass.Uint64(
                            0x8888888888888888
                        )


def _transform_stage(
    ttid: cutlass.Int32,
    threads: int,
    raw_pipe: pipeline.PipelineTmaAsync,
    ab_pipe: pipeline.PipelineAsyncUmma,
    raw_state: pipeline.PipelineState,
    ab_state: pipeline.PipelineState,
    sides: tuple[_TransformSide, _TransformSide],
    tail: tuple[cutlass.Int32 | int, int, int],
) -> tuple[pipeline.PipelineState, pipeline.PipelineState]:
    """Transform one stage: wait for its raw data and K-major buffer, issue
    every load, then the stores, then commit it.  ``tail`` is ``(k step,
    last k step, valid groups of the last quad)`` for the padding-scale mask.
    Returns the advanced states: the caller's loop carries them only through
    an assignment."""
    raw_pipe.consumer_wait(raw_state)
    ab_pipe.producer_acquire(ab_state)
    stage = raw_state.index
    transposed: list[list[_LdsmUnit]] = []
    direct: list[list[tuple[cutlass.Uint64, cutlass.Uint64]]] = []
    for side in sides:
        if side.transpose:
            transposed.append(
                _ldsm_loads(
                    ttid,
                    threads,
                    side.rows,
                    side.bk,
                    stage,
                    side.raw_base,
                    side.raw_stage_bytes,
                    side.rows // 16 - 1,
                    side.sf_words,
                    side.sf_stage_words,
                    side.half,
                    tail,
                )
            )
            direct.append([])
        else:
            pairs = side.sf_stage_words // 2
            direct.append(
                [
                    _direct_scale_load(
                        ttid + threads * item,
                        pairs % threads != 0,
                        stage,
                        side.sf_words,
                        side.sf_stage_words,
                        tail,
                    )
                    for item in range((pairs + threads - 1) // threads)
                ]
            )
            transposed.append([])
    for index, side in enumerate(sides):
        row_bytes = side.bk // 2
        chunk_mask = row_bytes // 16 - 1
        if side.transpose:
            _ldsm_stores(
                transposed[index],
                ttid,
                threads,
                side.bk,
                stage,
                side.sf_words,
                side.words32,
                side.stage_bytes,
                row_bytes,
                chunk_mask,
            )
        else:
            for item, (stored, pair) in enumerate(direct[index]):
                _direct_scale_fold(
                    stored,
                    pair,
                    ttid + threads * item,
                    stage,
                    side.sf_pairs,
                    side.sf_stage_words,
                    side.words64,
                    side.stage_bytes,
                    row_bytes,
                    chunk_mask,
                    side.half,
                    side.rows,
                )
    cute.arch.fence_view_async_shared()
    ab_pipe.producer_commit(ab_state)
    raw_pipe.consumer_release(raw_state)
    raw_state.advance()
    ab_state.advance()
    return raw_state, ab_state


def _raw_tile_layout(bk: int, rows: int) -> cute.ComposedLayout:
    """``[bk/2][rows]`` bytes with ``rows`` contiguous, swizzled like a TMA
    box of that width (SW64 for 64-byte rows, SW128 for 128)."""
    bits = {64: 2, 128: 3}.get(rows, 3)
    return cute.make_composed_layout(
        cute.make_swizzle(bits, 4, 3),
        0,
        cute.make_layout((bk // 2, rows), stride=(rows, 1)),
    )


class Sm100BlockScaledGemm:
    """One native tile family with independently tunable K and stage depth."""

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        bn: int,
        bk: int,
        stages: int,
        cluster_m: int,
        persistent: bool,
        num_sm: int,
        round_dtypes: tuple[type[cutlass.Numeric], ...] = (),
        *,
        operand_format: str = "nvfp4",
        fused_scales: bool = False,
        sf_tma: bool = False,
        sf_inplace: bool = False,
        accumulator_stages: int = 1,
        prefetch_output_registers: bool = False,
        tma_output: bool = False,
        compact_m: bool = False,
        logical_m_per_cta: int = 128,
        lhs_scale_row_stride: int = 1,
        lhs_scale_column_stride: int = 0,
        rhs_scale_row_stride: int = 1,
        rhs_scale_column_stride: int = 0,
        pdl: bool = False,
        fused: dict[str, tuple[str, int]] | None = None,
    ) -> None:
        assert operand_format in ("nvfp4", "mxfp8")
        self.operand_format = operand_format
        self.fused_scales = fused_scales
        self.sf_tma = sf_tma
        self.sf_inplace = sf_inplace
        assert not sf_tma or fused_scales
        assert not sf_inplace or (sf_tma and fused_scales)
        assert type(accumulator_stages) is int and accumulator_stages in (1, 2)
        assert type(prefetch_output_registers) is bool
        assert type(tma_output) is bool
        self.accumulator_stages = accumulator_stages
        self.prefetch_output_registers = prefetch_output_registers
        self.tma_output = tma_output
        self.output_stages = 2 if tma_output else 1
        self.compact_m = compact_m and logical_m_per_cta < 128 and m > 1
        self.logical_m_per_cta = logical_m_per_cta
        assert logical_m_per_cta in (32, 64, 128)
        assert fused_scales or logical_m_per_cta == 128
        self.lhs_scale_row_stride = lhs_scale_row_stride
        self.lhs_scale_column_stride = lhs_scale_column_stride
        self.rhs_scale_row_stride = rhs_scale_row_stride
        self.rhs_scale_column_stride = rhs_scale_column_stride
        # Fused preparation: no workspace kernel.  Each packed side is read
        # from its source tensor by TMA, a K-contiguous side ("direct",
        # (mode, row stride in bytes)) straight into its K-major stage and a
        # row-contiguous side ("transpose", (mode, k-byte stride in bytes))
        # into a raw [k-byte][row] staging tile that the transform warps
        # transpose into K-major words by ldmatrix.  The scales are read from their
        # swizzled source layout; the transform warps clear the sign bits
        # and fold them into the data, as the preparation kernel does.
        self.fused = fused is not None
        self.lhs_mode, self.lhs_stride = fused["lhs"] if fused else ("direct", 0)
        self.rhs_mode, self.rhs_stride = fused["rhs"] if fused else ("direct", 0)
        assert not self.fused or (operand_format == "nvfp4" and cluster_m == 1)
        assert not (self.fused and fused_scales)
        # Transform warps of the fused consumer (warps 6 and up).
        self.transform_threads = 256 if self.fused else 0
        # Warps 0-5 are the fixed roles; MXFP8 fused scales add their
        # scale warp (warp 6), the fused NVFP4 consumer its transform warps.
        self.threads = (224 if fused_scales else 192) + self.transform_threads
        # Programmatic dependent launch: the preparation kernel triggers its
        # dependents at its start, so this kernel's prologue (barrier and
        # pipeline init, TMEM allocation, descriptor prefetch) overlaps the
        # preparation; the TMA producer waits for its completion before the
        # first workspace load.
        self.pdl = pdl
        self.operand_dtype = (
            cutlass.Float4E2M1FN if operand_format == "nvfp4" else cutlass.Float8E4M3FN
        )
        self.scale_dtype = (
            cutlass.Float8E4M3FN if operand_format == "nvfp4" else cutlass.Float8E8M0FNU
        )
        self.scale_vector_size = 16 if operand_format == "nvfp4" else 32
        self.mma_k = 64 if operand_format == "nvfp4" else 32
        self.scale_chunk_k = 4 * self.scale_vector_size
        assert m > 0 and n > 0 and k > 0 and k % self.scale_vector_size == 0
        assert bn in (
            (8, 16, 32, 64, 128, 224, 256)
            if operand_format == "mxfp8"
            else (64, 128, 256)
        )
        assert bn != 8 or cluster_m == 1
        assert bk in (64, 128, 256)
        assert bk % self.scale_chunk_k == 0
        assert stages in (2, 3, 4, 5, 6, 8, 10, 12) and cluster_m in (1, 2)
        self.m, self.n, self.k = m, n, k
        self.mp, self.np, self.kp = block_scaled_workspace_shape(m, n, k, bn, bk)
        self.bm, self.bn, self.bk = 128 * cluster_m, bn, bk
        self.sf_n_chunks = (bn + 127) // 128
        self.stages = stages
        self.epi_n = (
            min(gcd(bn, 64), 32)
            if tma_output
            else gcd(bn, 64)
            if accumulator_stages > 1 or prefetch_output_registers
            else min(bn, 64)
        )
        required_tmem = (
            bn * accumulator_stages
            + (1 + self.sf_n_chunks) * bk // self.scale_vector_size
        )
        assert required_tmem <= 512
        assert not prefetch_output_registers or bn <= 224
        assert not tma_output or (
            operand_format == "mxfp8" and logical_m_per_cta == 128
        )
        self.cluster_m = cluster_m
        self.round_dtypes = round_dtypes
        self.group = tcgen05.CtaGroup.ONE if cluster_m == 1 else tcgen05.CtaGroup.TWO
        self.tile = (self.bm, bn, bk)
        self.qa_bytes = self.mp * self.kp // 2 if operand_format == "nvfp4" else 0
        self.qb_bytes = self.np * self.kp // 2 if operand_format == "nvfp4" else 0
        self.sa_bytes = self.mp * self.kp // self.scale_vector_size
        self.sb_bytes = self.np * self.kp // self.scale_vector_size
        self.qb_offset = self.qa_bytes
        self.sa_offset = self.qb_offset + self.qb_bytes
        self.sb_offset = self.sa_offset + self.sa_bytes
        self.workspace_bytes = self.sb_offset + self.sb_bytes
        self.m_tiles = (m + self.logical_m_per_cta * cluster_m - 1) // (
            self.logical_m_per_cta * cluster_m
        )
        self.n_tiles = (n + bn - 1) // bn
        self.tile_count = self.m_tiles * self.n_tiles
        self.clusters = (
            min(self.tile_count, max(1, num_sm // cluster_m))
            if persistent
            else self.tile_count
        )

    @cute.jit
    def __call__(
        self,
        workspace: cute.Tensor,
        output: cute.Tensor,
        alpha: cutlass.Float32,
        stream: CUstream,
    ) -> None:
        assert self.operand_format == "nvfp4"
        workspace_pointer = cast("cute.Pointer", workspace.iterator)
        qa = cute.make_tensor(
            cute.recast_ptr(workspace_pointer, dtype=cutlass.Float4E2M1FN),
            cute.make_layout((self.mp, self.kp, 1), stride=(self.kp, 1, 0)),
        )
        qb = cute.make_tensor(
            cute.recast_ptr(
                workspace_pointer + self.qb_offset, dtype=cutlass.Float4E2M1FN
            ),
            cute.make_layout((self.np, self.kp, 1), stride=(self.kp, 1, 0)),
        )
        sa = cute.make_tensor(
            cute.recast_ptr(workspace_pointer + self.sa_offset, dtype=cutlass.Int16),
            cute.make_layout((256, self.kp // 64, self.mp // 128)),
        )
        sb = cute.make_tensor(
            cute.recast_ptr(workspace_pointer + self.sb_offset, dtype=cutlass.Int16),
            cute.make_layout((256, self.kp // 64, self.np // 128)),
        )
        self._launch(qa, qb, sa, sb, output, alpha, stream, qa, qb)

    @cute.jit
    def fused_entry(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sa: cute.Tensor,
        sb: cute.Tensor,
        output: cute.Tensor,
        alpha: cutlass.Float32,
        stream: CUstream,
    ) -> None:
        """Consume the packed sources and swizzled scales directly."""
        assert self.operand_format == "nvfp4" and self.fused
        groups_padded = (self.k // 16 + 3) // 4 * 4
        assert groups_padded == self.kp // 16
        a_pointer = cast("cute.Pointer", a.iterator)
        b_pointer = cast("cute.Pointer", b.iterator)
        if cutlass.const_expr(self.lhs_mode == "direct"):
            qa = cute.make_tensor(
                cute.recast_ptr(a_pointer, dtype=cutlass.Float4E2M1FN),
                cute.make_layout(
                    (self.m, self.k, 1), stride=(2 * self.lhs_stride, 1, 0)
                ),
            )
            ra = qa
        else:
            qa = cute.make_tensor(
                cute.recast_ptr(a_pointer, dtype=cutlass.Float4E2M1FN),
                cute.make_layout((self.m, self.k, 1), stride=(self.k, 1, 0)),
            )
            ra = cute.make_tensor(
                cute.recast_ptr(a_pointer, dtype=cutlass.Uint8),
                cute.make_layout((self.k // 2, self.m), stride=(self.lhs_stride, 1)),
            )
        if cutlass.const_expr(self.rhs_mode == "direct"):
            qb = cute.make_tensor(
                cute.recast_ptr(b_pointer, dtype=cutlass.Float4E2M1FN),
                cute.make_layout(
                    (self.n, self.k, 1), stride=(2 * self.rhs_stride, 1, 0)
                ),
            )
            rb = qb
        else:
            qb = cute.make_tensor(
                cute.recast_ptr(b_pointer, dtype=cutlass.Float4E2M1FN),
                cute.make_layout((self.n, self.k, 1), stride=(self.k, 1, 0)),
            )
            rb = cute.make_tensor(
                cute.recast_ptr(b_pointer, dtype=cutlass.Uint8),
                cute.make_layout((self.k // 2, self.n), stride=(self.rhs_stride, 1)),
            )
        sfa = cute.make_tensor(
            cute.recast_ptr(cast("cute.Pointer", sa.iterator), dtype=cutlass.Int16),
            cute.make_layout((256, self.kp // 64, self.mp // 128)),
        )
        sfb = cute.make_tensor(
            cute.recast_ptr(cast("cute.Pointer", sb.iterator), dtype=cutlass.Int16),
            cute.make_layout((256, self.kp // 64, self.np // 128)),
        )
        self._launch(qa, qb, sfa, sfb, output, alpha, stream, ra, rb)

    @cute.jit
    def mxfp8(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        workspace: cute.Tensor,
        output: cute.Tensor,
        alpha: cutlass.Float32,
        stream: CUstream,
    ) -> None:
        """Read native FP8 operands directly and consume prepared E8M0 scales."""
        assert self.operand_format == "mxfp8"
        a_strides = cast("tuple[int | cutlass.Int64, int | cutlass.Int64]", a.stride)
        b_strides = cast("tuple[int | cutlass.Int64, int | cutlass.Int64]", b.stride)
        qa = cute.make_tensor(
            cute.recast_ptr(a.iterator, dtype=self.operand_dtype),
            cute.make_layout((self.m, self.k, 1), stride=(*a_strides, 0)),
        )
        qb = cute.make_tensor(
            cute.recast_ptr(b.iterator, dtype=self.operand_dtype),
            cute.make_layout((self.n, self.k, 1), stride=(*b_strides, 0)),
        )
        workspace_pointer = cast("cute.Pointer", workspace.iterator)
        sa = cute.make_tensor(
            cute.recast_ptr(workspace_pointer, dtype=cutlass.Int16),
            cute.make_layout((256, self.kp // self.scale_chunk_k, self.mp // 128)),
        )
        sb = cute.make_tensor(
            cute.recast_ptr(workspace_pointer + self.sb_offset, dtype=cutlass.Int16),
            cute.make_layout((256, self.kp // self.scale_chunk_k, self.np // 128)),
        )
        self._launch(qa, qb, sa, sb, output, alpha, stream, qa, qb)

    @cute.jit
    def mxfp8_packed(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sa: cute.Tensor,
        sb: cute.Tensor,
        output: cute.Tensor,
        alpha: cutlass.Float32,
        stream: CUstream,
    ) -> None:
        assert self.operand_format == "mxfp8" and self.fused_scales
        a_strides = cast("tuple[int | cutlass.Int64, int | cutlass.Int64]", a.stride)
        b_strides = cast("tuple[int | cutlass.Int64, int | cutlass.Int64]", b.stride)
        qa = cute.make_tensor(
            cute.recast_ptr(a.iterator, dtype=self.operand_dtype),
            cute.make_layout((self.m, self.k, 1), stride=(*a_strides, 0)),
        )
        qb = cute.make_tensor(
            cute.recast_ptr(b.iterator, dtype=self.operand_dtype),
            cute.make_layout((self.n, self.k, 1), stride=(*b_strides, 0)),
        )
        self._launch(qa, qb, sa, sb, output, alpha, stream, qa, qb)

    @cute.jit
    def _launch(
        self,
        qa: cute.Tensor,
        qb: cute.Tensor,
        sa: cute.Tensor,
        sb: cute.Tensor,
        output: cute.Tensor,
        alpha: cutlass.Float32,
        stream: CUstream,
        ra: cute.Tensor,
        rb: cute.Tensor,
    ) -> None:
        mma = blackwell_helpers.make_blockscaled_trivial_tiled_mma(
            self.operand_dtype,
            self.operand_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.scale_dtype,
            self.scale_vector_size,
            self.group,
            (self.bm, self.bn),
        )
        cluster = cute.tiled_divide(
            cute.make_layout((self.cluster_m, 1, 1)), (mma.thr_id.shape,)
        )
        al = blackwell_helpers.make_smem_layout_a(
            mma, self.tile, self.operand_dtype, self.stages
        )
        bl = blackwell_helpers.make_smem_layout_b(
            mma, self.tile, self.operand_dtype, self.stages
        )
        sal = blockscaled_layout.make_smem_layout_sfa(
            mma, self.tile, self.scale_vector_size, self.stages
        )
        sbl = blockscaled_layout.make_smem_layout_sfb(
            mma, self.tile, self.scale_vector_size, self.stages
        )
        sa_window = cute.make_layout((256, self.bk // self.scale_chunk_k, 1))
        sb_window = cute.make_layout(
            (256, self.bk // self.scale_chunk_k, self.sf_n_chunks)
        )
        assert cute.cosize(sal) == cute.cosize(sa_window) * 2 * self.stages
        assert cute.cosize(sbl) == cute.cosize(sb_window) * 2 * self.stages
        if cutlass.const_expr(self.compact_m):
            # Valid MMA rows occupy one logical stage. The remaining rows of
            # the physical MMA view are dead output lanes and overlap the next
            # stage; cosize keeps an explicit tail allocation for its final read.
            al, load_layout = _compact_a_layouts(al, self.logical_m_per_cta, self.bk)
            a_atom, a_tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(self.group),
                qa,
                load_layout,
                (self.logical_m_per_cta, self.bk, 1),
            )
        else:
            a_atom, a_tma = cute.nvgpu.make_tiled_tma_atom_A(
                cpasync.CopyBulkTensorTileG2SOp(self.group),
                qa,
                cute.slice_(al, (None, None, None, 0)),
                self.tile,
                mma,
                cluster.shape,
            )
        b_atom, b_tma = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(self.group),
            qb,
            cute.slice_(bl, (None, None, None, 0)),
            self.tile,
            mma,
            cluster.shape,
        )
        # A statically unit row mode disappears from CuTe's coordinate tensor,
        # although the TMA descriptor and its tile retain both K and row modes.
        # Restore only the coordinate domain; descriptor bounds still describe
        # the original allocation, so TMA zero-fills the extra coordinates.
        if cutlass.const_expr(self.operand_format == "mxfp8" and self.m == 1):
            a_tma = cute.make_tensor(
                (0, 0),
                cute.make_layout(
                    (2, self.k, 1),
                    stride=(
                        cute.ScaledBasis(1, mode=[1]),
                        cute.ScaledBasis(1, mode=[0]),
                        0,
                    ),
                ),
            )
        if cutlass.const_expr(self.operand_format == "mxfp8" and self.n == 1):
            b_tma = cute.make_tensor(
                (0, 0),
                cute.make_layout(
                    (2, self.k, 1),
                    stride=(
                        cute.ScaledBasis(1, mode=[1]),
                        cute.ScaledBasis(1, mode=[0]),
                        0,
                    ),
                ),
            )
        if cutlass.const_expr(self.fused_scales and self.sf_tma):
            sa_raw = cute.make_tensor(
                cute.recast_ptr(sa.iterator, dtype=cutlass.Int32),
                cute.make_layout(
                    (self.m, (self.k + 127) // 128, 1),
                    stride=(self.lhs_scale_row_stride, self.lhs_scale_column_stride, 0),
                ),
            )
            sb_raw = cute.make_tensor(
                cute.recast_ptr(sb.iterator, dtype=cutlass.Int32),
                cute.make_layout(
                    (self.n, (self.k + 127) // 128, 1),
                    stride=(self.rhs_scale_row_stride, self.rhs_scale_column_stride, 0),
                ),
            )
            sa_raw_layout = cute.make_layout(
                (128, self.bk // 128, 1), stride=(1, 128, 0)
            )
            sb_raw_layout = cute.make_layout(
                (128 * self.sf_n_chunks, self.bk // 128, 1),
                stride=(1, 128 * self.sf_n_chunks, 0),
            )
            sa_atom, sa_tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
                sa_raw,
                sa_raw_layout,
                sa_raw_layout.shape,
            )
            sb_atom, sb_tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
                sb_raw,
                sb_raw_layout,
                sb_raw_layout.shape,
            )
            if cutlass.const_expr(self.m == 1):
                sa_tma = cute.make_tensor(
                    (0, 0),
                    cute.make_layout(
                        (2, (self.k + 127) // 128, 1),
                        stride=(
                            cute.ScaledBasis(1, mode=[0]),
                            cute.ScaledBasis(1, mode=[1]),
                            0,
                        ),
                    ),
                )
            if cutlass.const_expr(self.n == 1):
                sb_tma = cute.make_tensor(
                    (0, 0),
                    cute.make_layout(
                        (2, (self.k + 127) // 128, 1),
                        stride=(
                            cute.ScaledBasis(1, mode=[0]),
                            cute.ScaledBasis(1, mode=[1]),
                            0,
                        ),
                    ),
                )
        elif cutlass.const_expr(self.fused_scales):
            sa_atom, sa_tma = a_atom, sa
            sb_atom, sb_tma = b_atom, sb
        else:
            sa_atom, sa_tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(self.group),
                sa,
                sa_window,
                sa_window.shape,
            )
            sb_atom, sb_tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(self.group),
                sb,
                sb_window,
                sb_window.shape,
            )
        if cutlass.const_expr(self.tma_output):
            assert output.element_type == cutlass.BFloat16
            assert cute.get(output.stride, 1) == 1
            output_row_stride = cute.get(output.stride, 0)
            assert output_row_stride >= self.n and output_row_stride % 8 == 0
            d_layout = blackwell_helpers.make_smem_layout_epi(
                output.element_type,
                LayoutEnum.ROW_MAJOR,
                (128, self.epi_n),
                self.output_stages,
            )
            d_atom, d_tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                output,
                cute.slice_(d_layout, (None, None, 0)),
                (128, self.epi_n),
            )
            assert cute.rank(d_tma.iterator) == 2
        else:
            d_atom, d_tma, d_layout = None, None, None
        self.tx_bytes = self.cluster_m * (
            (
                self.logical_m_per_cta * self.bk
                if self.compact_m
                else cute.size_in_bytes(self.operand_dtype, al) // self.stages
            )
            + cute.size_in_bytes(self.operand_dtype, bl) // self.stages
            + (0 if self.fused_scales else cute.cosize(sal) // self.stages)
            + (0 if self.fused_scales else cute.cosize(sbl) // self.stages)
        )
        # Raw staging tiles of the transposed sides: [bk/2 k-bytes][rows]
        # under the TMA swizzle of the row width (64 or 128 bytes), which
        # spreads the transform warps' ldmatrix rows over the banks.
        ral = _raw_tile_layout(self.bk, self.bm)
        rbl = _raw_tile_layout(self.bk, self.bn)
        ra_atom, ra_tma = a_atom, a_tma
        rb_atom, rb_tma = b_atom, b_tma
        if cutlass.const_expr(self.fused and self.lhs_mode == "transpose"):
            ra_atom, ra_tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(self.group), ra, ral, ral.shape
            )
        if cutlass.const_expr(self.fused and self.rhs_mode == "transpose"):
            rb_atom, rb_tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(self.group), rb, rbl, rbl.shape
            )
        # Fused: the raw tile replaces the K-major tile in the TMA byte count
        # (same size), the rest is unchanged.
        self.kernel(
            output,
            alpha,
            mma,
            cluster,
            a_atom,
            a_tma,
            b_atom,
            b_tma,
            sa_atom,
            sa_tma,
            sb_atom,
            sb_tma,
            al,
            bl,
            sal,
            sbl,
            d_atom,
            d_tma,
            d_layout,
            ra_atom,
            ra_tma,
            rb_atom,
            rb_tma,
            ral,
            rbl,
        ).launch(
            grid=(self.clusters * self.cluster_m, 1, 1),
            block=(self.threads, 1, 1),
            cluster=(self.cluster_m, 1, 1),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        output: cute.Tensor,
        alpha: cutlass.Float32,
        mma: cute.TiledMma,
        cluster: cute.Layout,
        a_atom: cute.CopyAtom,
        a_tma: cute.Tensor,
        b_atom: cute.CopyAtom,
        b_tma: cute.Tensor,
        sa_atom: cute.CopyAtom,
        sa_tma: cute.Tensor,
        sb_atom: cute.CopyAtom,
        sb_tma: cute.Tensor,
        al: cute.ComposedLayout,
        bl: cute.ComposedLayout,
        sal: cute.Layout,
        sbl: cute.Layout,
        d_atom: cute.CopyAtom | None,
        d_tma: cute.Tensor | None,
        d_layout: cute.Layout | cute.ComposedLayout | None,
        ra_atom: cute.CopyAtom,
        ra_tma: cute.Tensor,
        rb_atom: cute.CopyAtom,
        rb_tma: cute.Tensor,
        ral: cute.ComposedLayout,
        rbl: cute.ComposedLayout,
    ) -> None:
        output_pointer = cast("cute.Pointer", output.iterator)
        output_strides = cast(
            "tuple[int | cutlass.Int64, int | cutlass.Int64]", output.stride
        )
        tid = cutlass.Int32(cute.arch.thread_idx()[0])
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block = cutlass.Int32(cute.arch.block_idx()[0])
        v = block % self.cluster_m
        leader = v == 0
        first_tile = block // self.cluster_m
        tile_stride = cutlass.Int32(cute.arch.grid_dim()[0]) // self.cluster_m
        ap = cute.arch.alloc_smem(
            cutlass.Uint8,
            cute.size_in_bytes(self.operand_dtype, al),
            alignment=1024,
        )
        bp = cute.arch.alloc_smem(
            cutlass.Uint8,
            cute.size_in_bytes(self.operand_dtype, bl),
            alignment=1024,
        )
        a = cute.make_tensor(
            cute.recast_ptr(ap, al.inner, dtype=self.operand_dtype), al.outer
        )
        if cutlass.const_expr(self.compact_m):
            a_load = cute.make_tensor(
                cute.recast_ptr(ap, al.inner, dtype=self.operand_dtype),
                cute.make_layout(
                    (self.logical_m_per_cta, (128, self.bk // 128), 1, self.stages),
                    stride=(
                        128,
                        (1, self.logical_m_per_cta * 128),
                        0,
                        self.logical_m_per_cta * self.bk,
                    ),
                ),
            )
        else:
            a_load = a
        b = cute.make_tensor(
            cute.recast_ptr(bp, bl.inner, dtype=self.operand_dtype), bl.outer
        )
        sap = cute.arch.alloc_smem(self.scale_dtype, cute.cosize(sal), alignment=128)
        sbp = cute.arch.alloc_smem(self.scale_dtype, cute.cosize(sbl), alignment=128)
        sa = cute.make_tensor(sap, sal)
        sb = cute.make_tensor(sbp, sbl)
        sa_chunks = cute.make_tensor(
            cute.recast_ptr(sap, dtype=cutlass.Int16),
            cute.make_layout((256, self.bk // self.scale_chunk_k, 1, self.stages)),
        )
        sb_chunks = cute.make_tensor(
            cute.recast_ptr(sbp, dtype=cutlass.Int16),
            cute.make_layout(
                (256, self.bk // self.scale_chunk_k, self.sf_n_chunks, self.stages)
            ),
        )
        ab_barriers = cute.arch.alloc_smem(cutlass.Int64, 2 * self.stages, alignment=8)
        c_barriers = cute.arch.alloc_smem(
            cutlass.Int64, 2 * self.accumulator_stages, alignment=8
        )
        lhs_transpose = self.fused and self.lhs_mode == "transpose"
        rhs_transpose = self.fused and self.rhs_mode == "transpose"
        # Fused: raw [k-byte][row] staging tiles of the transposed sides and a
        # TMA -> transform-warps pipeline in front of the transform -> UMMA one.
        # 1024-byte alignment: the swizzle is a function of the address.
        rap = cute.arch.alloc_smem(
            cutlass.Uint8,
            cute.cosize(ral) * self.stages if lhs_transpose else 16,
            alignment=1024,
        )
        rbp = cute.arch.alloc_smem(
            cutlass.Uint8,
            cute.cosize(rbl) * self.stages if rhs_transpose else 16,
            alignment=1024,
        )
        raw_a = cute.make_tensor(
            cute.recast_ptr(rap, ral.inner, dtype=cutlass.Uint8),
            cute.make_layout(
                (self.bk // 2, self.bm, self.stages),
                stride=(self.bm, 1, cute.cosize(ral)),
            ),
        )
        raw_b = cute.make_tensor(
            cute.recast_ptr(rbp, rbl.inner, dtype=cutlass.Uint8),
            cute.make_layout(
                (self.bk // 2, self.bn, self.stages),
                stride=(self.bn, 1, cute.cosize(rbl)),
            ),
        )
        raw_barriers = cute.arch.alloc_smem(
            cutlass.Int64, 2 * self.stages if self.fused else 2, alignment=8
        )
        if cutlass.const_expr(self.fused):
            # The TMA-async consumer releases a stage from one lane per warp;
            # multicast signaling sizes the empty barrier by those lanes
            # (eight warps), not by the 256 threads of the group.
            raw_pipe = pipeline.PipelineTmaAsync.create(
                num_stages=self.stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.transform_threads
                ),
                tx_count=self.tx_bytes,
                barrier_storage=raw_barriers,
                cta_layout_vmnk=cluster,
                enable_multicast_signaling=True,
                defer_sync=True,
            )
            ab_pipe = pipeline.PipelineAsyncUmma.create(
                num_stages=self.stages,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.transform_threads
                ),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                barrier_storage=ab_barriers,
                cta_layout_vmnk=cluster,
                defer_sync=True,
            )
        else:
            # Unused: the TMA producer feeds the UMMA pipeline directly.
            raw_pipe = cast("pipeline.PipelineTmaAsync", None)
            ab_pipe = pipeline.PipelineTmaUmma.create(
                num_stages=self.stages,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    (1 + 32 * self.cluster_m) if self.sf_inplace else 1,
                ),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                tx_count=self.tx_bytes,
                barrier_storage=ab_barriers,
                cta_layout_vmnk=cluster,
                defer_sync=True,
            )
        if cutlass.const_expr(self.fused_scales and self.sf_tma):
            if cutlass.const_expr(self.sf_inplace):
                assert cute.cosize(sal) == 128 * (self.bk // 128) * self.stages * 4
                assert cute.cosize(sbl) == (
                    128 * self.sf_n_chunks * (self.bk // 128) * self.stages * 4
                )
                raw_sa_ptr = cute.recast_ptr(sap, dtype=cutlass.Int32)
                raw_sb_ptr = cute.recast_ptr(sbp, dtype=cutlass.Int32)
            else:
                raw_sa_ptr = cute.arch.alloc_smem(
                    cutlass.Int32, 128 * (self.bk // 128) * self.stages, alignment=128
                )
                raw_sb_ptr = cute.arch.alloc_smem(
                    cutlass.Int32,
                    128 * self.sf_n_chunks * (self.bk // 128) * self.stages,
                    alignment=128,
                )
            raw_sa_shared = cute.make_tensor(
                raw_sa_ptr,
                cute.make_layout(
                    (128, self.bk // 128, 1, self.stages),
                    stride=(1, 128, 0, 128 * (self.bk // 128)),
                ),
            )
            raw_sb_shared = cute.make_tensor(
                raw_sb_ptr,
                cute.make_layout(
                    (128 * self.sf_n_chunks, self.bk // 128, 1, self.stages),
                    stride=(
                        1,
                        128 * self.sf_n_chunks,
                        0,
                        128 * self.sf_n_chunks * (self.bk // 128),
                    ),
                ),
            )
            # These flat word views name the actual complete stage buffers.
            # Indexing a Tensor is the public DSL memory-access interface;
            # a Pointer only describes the buffer's base address.
            raw_sa_words = cute.make_tensor(
                raw_sa_ptr,
                cute.make_layout(128 * (self.bk // 128) * self.stages),
            )
            raw_sb_words = cute.make_tensor(
                raw_sb_ptr,
                cute.make_layout(
                    128 * self.sf_n_chunks * (self.bk // 128) * self.stages
                ),
            )
            if cutlass.const_expr(self.sf_inplace):
                raw_sf_pipe = None
                raw_sf_barriers = cute.arch.alloc_smem(
                    cutlass.Int64, self.stages, alignment=8
                )
                raw_sf_full = pipeline.MbarrierArray(
                    barrier_storage=raw_sf_barriers,
                    num_stages=self.stages,
                    agent=(
                        pipeline.PipelineOp.TmaLoad,
                        pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                    ),
                    tx_count=(128 + 128 * self.sf_n_chunks) * (self.bk // 128) * 4,
                )
            else:
                raw_sf_full = None
                raw_sf_barriers = cute.arch.alloc_smem(
                    cutlass.Int64, 2 * self.stages, alignment=8
                )
                raw_sf_pipe = pipeline.PipelineTmaAsync.create(
                    num_stages=self.stages,
                    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
                    tx_count=(128 + 128 * self.sf_n_chunks) * (self.bk // 128) * 4,
                    barrier_storage=raw_sf_barriers,
                    cta_layout_vmnk=cluster,
                    tidx=tid,
                    enable_multicast_signaling=True,
                    defer_sync=True,
                )
        else:
            # Disabled transfer modes have no buffers or pipeline. Keep this
            # absence explicit, then refine it inside the corresponding mode.
            raw_sa_shared, raw_sb_shared = None, None
            raw_sa_words, raw_sb_words = None, None
            raw_sf_full, raw_sf_pipe = None, None
        if cutlass.const_expr(self.fused_scales):
            if cutlass.const_expr(self.sf_inplace):
                sf_target = None
                if cutlass.const_expr(self.cluster_m == 2):
                    sf_target = pipeline.PipelineAsyncUmma._compute_leading_cta_rank(
                        self.cluster_m
                    )
                sf_pipe = pipeline.PipelineAsyncUmma(
                    ab_pipe.sync_object_full.recast_to_new_op_type(
                        pipeline.PipelineOp.AsyncThread
                    ),
                    ab_pipe.sync_object_empty,
                    self.stages,
                    sf_target,
                    ab_pipe.consumer_mask,
                    self.group,
                )
            else:
                sf_barriers = cute.arch.alloc_smem(
                    cutlass.Int64, 2 * self.stages, alignment=8
                )
                sf_pipe = pipeline.PipelineAsyncUmma.create(
                    num_stages=self.stages,
                    producer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread, 32 * self.cluster_m
                    ),
                    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                    barrier_storage=sf_barriers,
                    cta_layout_vmnk=cluster,
                    defer_sync=True,
                )
        else:
            sf_pipe = None
        c_pipe = pipeline.PipelineUmmaAsync.create(
            num_stages=self.accumulator_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 128 * self.cluster_m
            ),
            barrier_storage=c_barriers,
            cta_layout_vmnk=cluster,
            defer_sync=True,
        )
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster, is_relaxed=True)
        allocator = utils.TmemAllocator(
            barrier_for_retrieve=pipeline.NamedBarrier(
                barrier_id=1, num_threads=self.threads
            ),
            is_two_cta=self.cluster_m == 2,
        )
        c_layout = mma.make_fragment_C(mma.partition_shape_C((self.bm, self.bn))).layout
        sa_layout = blockscaled_layout.make_tmem_layout_sfa(
            mma,
            self.tile,
            self.scale_vector_size,
            cute.slice_(sal, (None, None, None, 0)),
        )
        sb_layout = blockscaled_layout.make_tmem_layout_sfb(
            mma,
            self.tile,
            self.scale_vector_size,
            cute.slice_(sbl, (None, None, None, 0)),
        )
        # Layout footprints are TMEM columns when inspected through a fake
        # pointer. Real pointers are retrieved only after allocation completes.
        c_columns = self.bn * self.accumulator_stages
        sa_columns = self.bk // self.scale_vector_size
        sb_columns = self.sf_n_chunks * (self.bk // self.scale_vector_size)
        columns = max(32, 1 << (c_columns + sa_columns + sb_columns - 1).bit_length())
        assert columns <= 512
        allocator.allocate(columns)
        allocator.wait_for_alloc()
        tmem = allocator.retrieve_ptr(cutlass.Float32)
        acc = cute.make_tensor(tmem, c_layout)
        tsa = cute.make_tensor(
            cute.recast_ptr(tmem + c_columns, dtype=self.scale_dtype), sa_layout
        )
        tsb = cute.make_tensor(
            cute.recast_ptr(tmem + c_columns + sa_columns, dtype=self.scale_dtype),
            sb_layout,
        )
        assert tcgen05.find_tmem_tensor_col_offset(acc) == self.bn
        assert tcgen05.find_tmem_tensor_col_offset(tsa) == sa_columns, (
            f"SFA columns {tcgen05.find_tmem_tensor_col_offset(tsa)} != {sa_columns}"
        )
        assert tcgen05.find_tmem_tensor_col_offset(tsb) == sb_columns, (
            f"SFB columns {tcgen05.find_tmem_tensor_col_offset(tsb)} != {sb_columns}"
        )
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster)
        if warp == 5:
            cpasync.prefetch_descriptor(a_atom)
            cpasync.prefetch_descriptor(b_atom)
            if cutlass.const_expr(not self.fused_scales or self.sf_tma):
                cpasync.prefetch_descriptor(sa_atom)
                cpasync.prefetch_descriptor(sb_atom)
            if cutlass.const_expr(lhs_transpose):
                cpasync.prefetch_descriptor(ra_atom)
            if cutlass.const_expr(rhs_transpose):
                cpasync.prefetch_descriptor(rb_atom)
            if cutlass.const_expr(self.pdl):
                # This warp's TMA reads the preparation kernel's workspace,
                # or (fused) the sources the preceding grid in the stream may
                # still be writing; the only other global reads before the
                # MMA are the fused-scales scale warp's, which waits itself.
                cute.arch.griddepcontrol_wait()
            state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.stages
            )
            # Fused: the K-major stages are refilled only once the MMA has
            # released them, which the raw pipeline alone does not track.
            free_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.stages
            )
            thread_mma = mma.get_slice(v)
            for tile_index in cutlass.range(
                first_tile, self.tile_count, tile_stride, unroll=1
            ):
                mi = tile_index // self.n_tiles
                ni = tile_index % self.n_tiles
                if cutlass.const_expr(self.compact_m):
                    ga = cute.local_tile(
                        a_tma,
                        (self.logical_m_per_cta, self.bk, 1),
                        (mi * self.cluster_m + v, None, 0),
                    )
                else:
                    ga = thread_mma.partition_A(
                        cute.local_tile(
                            cute.domain_offset(
                                (
                                    mi * self.logical_m_per_cta * self.cluster_m
                                    + v * (self.logical_m_per_cta - 128),
                                    0,
                                ),
                                a_tma[None, None, 0],
                            ),
                            (self.bm, self.bk),
                            (0, None),
                        )
                    )
                gb = thread_mma.partition_B(
                    cute.local_tile(
                        b_tma[None, None, 0], (self.bn, self.bk), (ni, None)
                    )
                )
                if cutlass.const_expr(self.sf_tma):
                    m_start = (
                        mi * self.logical_m_per_cta * self.cluster_m
                        + v * self.logical_m_per_cta
                    )
                    n_start = (
                        (ni // 2 * 128)
                        if cutlass.const_expr(self.bn == 64)
                        else ni * self.bn
                    )
                    raw_gsa = cute.local_tile(
                        cute.domain_offset((m_start, 0, 0), sa_tma),
                        (128, self.bk // 128, 1),
                        (0, None, 0),
                    )
                    raw_gsb = cute.local_tile(
                        cute.domain_offset((n_start, 0, 0), sb_tma),
                        (128 * self.sf_n_chunks, self.bk // 128, 1),
                        (0, None, 0),
                    )
                else:
                    raw_gsa, raw_gsb = None, None
                if cutlass.const_expr(not self.fused_scales):
                    gsa = cute.local_tile(
                        sa_tma,
                        (256, self.bk // self.scale_chunk_k, 1),
                        (0, None, mi * self.cluster_m + v),
                    )
                    gsb = cute.local_tile(
                        sb_tma,
                        (256, self.bk // self.scale_chunk_k, self.sf_n_chunks),
                        (0, None, ni // 2 if cutlass.const_expr(self.bn == 64) else ni),
                    )
                else:
                    gsa, gsb = None, None
                da, ga = cpasync.tma_partition(
                    a_atom,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(a_load, 0, 3),
                    cute.group_modes(ga, 0, 3),
                )
                db, gb = cpasync.tma_partition(
                    b_atom,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(b, 0, 3),
                    cute.group_modes(gb, 0, 3),
                )
                if cutlass.const_expr(not self.fused_scales):
                    assert gsa is not None and gsb is not None
                    dsa, gsa = cpasync.tma_partition(
                        sa_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sa_chunks, 0, 3),
                        cute.group_modes(gsa, 0, 3),
                    )
                    dsb, gsb = cpasync.tma_partition(
                        sb_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sb_chunks, 0, 3),
                        cute.group_modes(gsb, 0, 3),
                    )
                else:
                    dsa, dsb = None, None
                if cutlass.const_expr(self.sf_tma):
                    assert raw_sa_shared is not None and raw_sb_shared is not None
                    assert raw_gsa is not None and raw_gsb is not None
                    raw_dsa, raw_gsa = cpasync.tma_partition(
                        sa_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(raw_sa_shared, 0, 3),
                        cute.group_modes(raw_gsa, 0, 3),
                    )
                    raw_dsb, raw_gsb = cpasync.tma_partition(
                        sb_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(raw_sb_shared, 0, 3),
                        cute.group_modes(raw_gsb, 0, 3),
                    )
                else:
                    raw_dsa, raw_dsb = None, None
                # Placeholders where a side is not transposed (never copied).
                dra, gra = da, ga
                drb, grb = db, gb
                if cutlass.const_expr(lhs_transpose):
                    gra = cute.local_tile(ra_tma, (self.bk // 2, self.bm), (None, mi))
                    dra, gra = cpasync.tma_partition(
                        ra_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(raw_a, 0, 2),
                        cute.group_modes(gra, 0, 2),
                    )
                if cutlass.const_expr(rhs_transpose):
                    grb = cute.local_tile(rb_tma, (self.bk // 2, self.bn), (None, ni))
                    drb, grb = cpasync.tma_partition(
                        rb_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(raw_b, 0, 2),
                        cute.group_modes(grb, 0, 2),
                    )
                for ki in cutlass.range(self.kp // self.bk, unroll=1):
                    if cutlass.const_expr(self.fused):
                        raw_pipe.producer_acquire(state)
                        ab_pipe.producer_acquire(free_state)
                        barrier = raw_pipe.producer_get_barrier(state)
                    else:
                        if cutlass.const_expr(self.sf_inplace):
                            # The same empty signal protects operands and scales.
                            # Raw SF TMA may overwrite a stage only after UTCCP/MMA
                            # has finished consuming its prior contents.
                            ab_pipe.producer_acquire(state)
                        if cutlass.const_expr(self.sf_tma):
                            assert raw_gsa is not None and raw_gsb is not None
                            assert raw_dsa is not None and raw_dsb is not None
                            if cutlass.const_expr(self.sf_inplace):
                                assert raw_sf_full is not None
                                raw_sf_full.arrive_and_expect_tx(
                                    state.index, raw_sf_full.tx_count
                                )
                                raw_barrier = raw_sf_full.get_barrier(state.index)
                            else:
                                assert raw_sf_pipe is not None
                                raw_sf_pipe.producer_acquire(state)
                                raw_barrier = raw_sf_pipe.producer_get_barrier(state)
                            cute.copy(
                                sa_atom,
                                raw_gsa[None, ki],
                                raw_dsa[None, state.index],
                                tma_bar_ptr=raw_barrier,
                            )
                            cute.copy(
                                sb_atom,
                                raw_gsb[None, ki],
                                raw_dsb[None, state.index],
                                tma_bar_ptr=raw_barrier,
                            )
                            if cutlass.const_expr(not self.sf_inplace):
                                assert raw_sf_pipe is not None
                                raw_sf_pipe.producer_commit(state)
                        if cutlass.const_expr(not self.sf_inplace):
                            ab_pipe.producer_acquire(state)
                        barrier = ab_pipe.producer_get_barrier(state)
                    if cutlass.const_expr(lhs_transpose):
                        cute.copy(
                            ra_atom,
                            gra[None, ki],
                            dra[None, state.index],
                            tma_bar_ptr=barrier,
                        )
                    else:
                        cute.copy(
                            a_atom,
                            ga[None, ki],
                            da[None, state.index],
                            tma_bar_ptr=barrier,
                        )
                    if cutlass.const_expr(rhs_transpose):
                        cute.copy(
                            rb_atom,
                            grb[None, ki],
                            drb[None, state.index],
                            tma_bar_ptr=barrier,
                        )
                    else:
                        cute.copy(
                            b_atom,
                            gb[None, ki],
                            db[None, state.index],
                            tma_bar_ptr=barrier,
                        )
                    if cutlass.const_expr(not self.fused_scales):
                        assert gsa is not None and gsb is not None
                        assert dsa is not None and dsb is not None
                        cute.copy(
                            sa_atom,
                            gsa[None, ki],
                            dsa[None, state.index],
                            tma_bar_ptr=barrier,
                        )
                        cute.copy(
                            sb_atom,
                            gsb[None, ki],
                            dsb[None, state.index],
                            tma_bar_ptr=barrier,
                        )
                    if cutlass.const_expr(self.fused):
                        raw_pipe.producer_commit(state)
                        free_state.advance()
                    else:
                        ab_pipe.producer_commit(state)
                    state.advance()
            if cutlass.const_expr(self.fused):
                raw_pipe.producer_tail(state)
            else:
                ab_pipe.producer_tail(state.clone())
                if cutlass.const_expr(self.sf_tma and not self.sf_inplace):
                    assert raw_sf_pipe is not None
                    raw_sf_pipe.producer_tail(state.clone())
        if cutlass.const_expr(self.fused):
            if warp >= 6:
                # Transform warps (``_transform_stage``).  A direct side's
                # scale words are cleared of their sign bits and the signs
                # folded into its K-major qwords in place; a transposed
                # side's raw [k-byte][row] tile goes through ldmatrix into
                # the K-major stage with its signs folded and cleared on the
                # way.  Every load of a stage is issued before its first
                # store.
                ttid = tid - 192
                raw_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.stages
                )
                ab_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.stages
                )
                a_bytes = cute.size_in_bytes(self.operand_dtype, al)
                b_bytes = cute.size_in_bytes(self.operand_dtype, bl)
                a_words = cute.make_tensor(
                    cute.recast_ptr(ap, dtype=cutlass.Uint64),
                    cute.make_layout((a_bytes // 8,)),
                )
                b_words = cute.make_tensor(
                    cute.recast_ptr(bp, dtype=cutlass.Uint64),
                    cute.make_layout((b_bytes // 8,)),
                )
                a_words32 = cute.make_tensor(
                    cute.recast_ptr(ap, dtype=cutlass.Uint32),
                    cute.make_layout((a_bytes // 4,)),
                )
                b_words32 = cute.make_tensor(
                    cute.recast_ptr(bp, dtype=cutlass.Uint32),
                    cute.make_layout((b_bytes // 4,)),
                )
                sfa_words = cute.make_tensor(
                    cute.recast_ptr(sap, dtype=cutlass.Uint32),
                    cute.make_layout((cute.cosize(sal) // 4,)),
                )
                sfb_words = cute.make_tensor(
                    cute.recast_ptr(sbp, dtype=cutlass.Uint32),
                    cute.make_layout((cute.cosize(sbl) // 4,)),
                )
                sfa_pairs = cute.make_tensor(
                    cute.recast_ptr(sap, dtype=cutlass.Uint64),
                    cute.make_layout((cute.cosize(sal) // 8,)),
                )
                sfb_pairs = cute.make_tensor(
                    cute.recast_ptr(sbp, dtype=cutlass.Uint64),
                    cute.make_layout((cute.cosize(sbl) // 8,)),
                )
                for tile_index in cutlass.range(
                    first_tile, self.tile_count, tile_stride, unroll=1
                ):
                    half = cutlass.Int32(0)
                    if cutlass.const_expr(self.bn == 64):
                        half = cutlass.Int32((tile_index % self.n_tiles % 2) * 64)
                    sides = (
                        _TransformSide(
                            lhs_transpose,
                            self.bm,
                            self.bk,
                            cutlass.Int32(0),
                            sfa_words,
                            sfa_pairs,
                            cute.cosize(sal) // self.stages // 4,
                            a_words,
                            a_words32,
                            a_bytes // self.stages,
                            cutlass.Int32(rap.toint()),
                            cute.cosize(ral),
                        ),
                        _TransformSide(
                            rhs_transpose,
                            self.bn,
                            self.bk,
                            half,
                            sfb_words,
                            sfb_pairs,
                            cute.cosize(sbl) // self.stages // 4,
                            b_words,
                            b_words32,
                            b_bytes // self.stages,
                            cutlass.Int32(rbp.toint()),
                            cute.cosize(rbl),
                        ),
                    )
                    for _ki in cutlass.range(self.kp // self.bk, unroll=1):
                        raw_state, ab_state = _transform_stage(
                            ttid,
                            self.transform_threads,
                            raw_pipe,
                            ab_pipe,
                            raw_state,
                            ab_state,
                            sides,
                            (_ki, self.kp // self.bk - 1, self.k // 16 % 4),
                        )
                ab_pipe.producer_tail(ab_state)
        if warp == 4:
            if leader:
                ra = mma.make_fragment_A(a)
                rb = mma.make_fragment_B(b)
                state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.stages
                )
                cstate = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.accumulator_stages
                )
                for _tile_index in cutlass.range(
                    first_tile, self.tile_count, tile_stride, unroll=1
                ):
                    mma_sfb = tsb
                    if cutlass.const_expr(self.bn == 64):
                        # Two N64 tiles share one prepared 128-row SFB chunk.
                        # The upper 64 rows start two Float32 TMEM columns past
                        # the unshifted scale base, as required by tcgen05.
                        offset = cutlass.Int32((_tile_index % self.n_tiles % 2) * 2)
                        mma_sfb = cute.make_tensor(
                            cute.recast_ptr(
                                tmem + c_columns + sa_columns + offset,
                                dtype=self.scale_dtype,
                            ),
                            sb_layout,
                        )
                    c_pipe.producer_acquire(cstate)
                    tile_acc = cute.make_tensor(tmem + cstate.index * self.bn, c_layout)
                    mma.set(tcgen05.Field.ACCUMULATE, False)
                    for _ki in cutlass.range(self.kp // self.bk, unroll=1):
                        ab_pipe.consumer_wait(state)
                        if cutlass.const_expr(
                            self.fused_scales and not self.sf_inplace
                        ):
                            assert sf_pipe is not None
                            sf_pipe.consumer_wait(state)
                        _scale_copy(sa, tsa, state.index, self.group)
                        _scale_copy(sb, tsb, state.index, self.group)
                        for kk in cutlass.range_constexpr(self.bk // self.mma_k):
                            mma.set(tcgen05.Field.SFA, tsa[None, None, kk].iterator)
                            mma.set(tcgen05.Field.SFB, mma_sfb[None, None, kk].iterator)
                            cute.gemm(
                                mma,
                                tile_acc,
                                ra[None, None, kk, state.index],
                                rb[None, None, kk, state.index],
                                tile_acc,
                            )
                            mma.set(tcgen05.Field.ACCUMULATE, True)
                        ab_pipe.consumer_release(state)
                        if cutlass.const_expr(
                            self.fused_scales and not self.sf_inplace
                        ):
                            assert sf_pipe is not None
                            sf_pipe.consumer_release(state)
                        state.advance()
                    c_pipe.producer_commit(cstate)
                    cstate.advance()
                c_pipe.producer_tail(cstate)
        if cutlass.const_expr(self.fused_scales):
            assert sf_pipe is not None
            if warp == 6:
                lane = tid % 32
                if cutlass.const_expr(self.pdl and not self.sf_tma):
                    # This warp reads the packed scales from global memory
                    # (with sf_tma they arrive through warp 5's TMA, behind
                    # its wait).
                    cute.arch.griddepcontrol_wait()
                if cutlass.const_expr(not self.sf_tma):
                    raw_sa = cute.recast_ptr(sa_tma.iterator, dtype=cutlass.Int32)
                    raw_sb = cute.recast_ptr(sb_tma.iterator, dtype=cutlass.Int32)
                else:
                    raw_sa, raw_sb = None, None
                dst_sa = cute.recast_ptr(sap, dtype=cutlass.Int32)
                dst_sb = cute.recast_ptr(sbp, dtype=cutlass.Int32)
                state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.stages
                )
                if cutlass.const_expr(self.sf_tma):
                    raw_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.stages
                    )
                else:
                    raw_state = None
                for tile_index in cutlass.range(
                    first_tile, self.tile_count, tile_stride, unroll=1
                ):
                    mi = tile_index // self.n_tiles
                    ni = tile_index % self.n_tiles
                    m_start = (
                        mi * self.logical_m_per_cta * self.cluster_m
                        + v * self.logical_m_per_cta
                    )
                    n_start = (
                        (ni // 2 * 128)
                        if cutlass.const_expr(self.bn == 64)
                        else ni * self.bn
                    )
                    for ki in cutlass.range(self.kp // self.bk, unroll=1):
                        sf_pipe.producer_acquire(state)
                        if cutlass.const_expr(self.sf_tma):
                            assert raw_state is not None
                            if cutlass.const_expr(self.sf_inplace):
                                assert raw_sf_full is not None
                                raw_sf_full.wait(raw_state.index, raw_state.phase)
                            else:
                                assert raw_sf_pipe is not None
                                raw_sf_pipe.consumer_wait(raw_state)
                        if cutlass.const_expr(self.sf_inplace):
                            assert raw_sa_words is not None and raw_sb_words is not None
                            _transpose_sf_stage_inplace(
                                raw_sa_words,
                                raw_sb_words,
                                state.index,
                                ki,
                                m_start,
                                n_start,
                                self.m,
                                self.n,
                                self.k,
                                self.bk,
                                self.sf_n_chunks,
                                self.logical_m_per_cta,
                            )
                        else:
                            for chunk in cutlass.range_constexpr(self.bk // 128):
                                word_k = ki * (self.bk // 128) + chunk
                                for segment in cutlass.range_constexpr(4):
                                    row = lane + segment * 32
                                    gm = m_start + row
                                    value = cutlass.Int32(0)
                                    if (
                                        gm < self.m
                                        and row < self.logical_m_per_cta
                                        and word_k < (self.k + 127) // 128
                                    ):
                                        if cutlass.const_expr(self.sf_tma):
                                            assert raw_sa_words is not None
                                            value = raw_sa_words[
                                                state.index * (128 * (self.bk // 128))
                                                + chunk * 128
                                                + row
                                            ]
                                        else:
                                            assert raw_sa is not None
                                            value = raw_sa[
                                                cutlass.Int64(gm)
                                                * self.lhs_scale_row_stride
                                                + cutlass.Int64(word_k)
                                                * self.lhs_scale_column_stride
                                            ]
                                    if cutlass.const_expr(self.k % 128 != 0):
                                        if word_k == self.k // 128:
                                            value = value & (
                                                (1 << ((self.k % 128) // 32 * 8)) - 1
                                            )
                                    dst_sa[
                                        state.index * (128 * (self.bk // 128))
                                        + chunk * 128
                                        + lane * 4
                                        + segment
                                    ] = value
                                for n_chunk in cutlass.range_constexpr(
                                    self.sf_n_chunks
                                ):
                                    for segment in cutlass.range_constexpr(4):
                                        row = lane + segment * 32
                                        gn = n_start + n_chunk * 128 + row
                                        value = cutlass.Int32(0)
                                        if (
                                            gn < self.n
                                            and word_k < (self.k + 127) // 128
                                        ):
                                            if cutlass.const_expr(self.sf_tma):
                                                assert raw_sb_words is not None
                                                value = raw_sb_words[
                                                    state.index
                                                    * (
                                                        128
                                                        * self.sf_n_chunks
                                                        * (self.bk // 128)
                                                    )
                                                    + chunk * (128 * self.sf_n_chunks)
                                                    + n_chunk * 128
                                                    + row
                                                ]
                                            else:
                                                assert raw_sb is not None
                                                value = raw_sb[
                                                    cutlass.Int64(gn)
                                                    * self.rhs_scale_row_stride
                                                    + cutlass.Int64(word_k)
                                                    * self.rhs_scale_column_stride
                                                ]
                                        if cutlass.const_expr(self.k % 128 != 0):
                                            if word_k == self.k // 128:
                                                value = value & (
                                                    (1 << ((self.k % 128) // 32 * 8))
                                                    - 1
                                                )
                                        dst_sb[
                                            state.index
                                            * (
                                                128
                                                * (self.bk // 128)
                                                * self.sf_n_chunks
                                            )
                                            + n_chunk * (128 * (self.bk // 128))
                                            + chunk * 128
                                            + lane * 4
                                            + segment
                                        ] = value
                        cute.arch.fence_view_async_shared()
                        if cutlass.const_expr(self.sf_tma and not self.sf_inplace):
                            assert raw_sf_pipe is not None and raw_state is not None
                            raw_sf_pipe.consumer_release(raw_state)
                        sf_pipe.producer_commit(state)
                        state.advance()
                        if cutlass.const_expr(self.sf_tma):
                            assert raw_state is not None
                            raw_state.advance()
                sf_pipe.producer_tail(state.clone())
        if warp < 4:
            epi_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
            if cutlass.const_expr(self.tma_output):
                assert isinstance(d_layout, cute.ComposedLayout)
                assert d_atom is not None
                cp = cute.arch.alloc_smem(
                    output.element_type, cute.cosize(d_layout), alignment=1024
                )
                shared_d = cute.make_tensor(
                    cute.recast_ptr(cp, d_layout.inner, dtype=output.element_type),
                    d_layout.outer,
                )
                shared_c = shared_d[None, None, 0]
                store_pipe = pipeline.PipelineTmaStore.create(
                    num_stages=self.output_stages,
                    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                )
                store_buffer = cutlass.Int32(0)
                if warp == 0:
                    cpasync.prefetch_descriptor(d_atom)
            else:
                shared_d, store_pipe, store_buffer = None, None, None
                cp = cute.arch.alloc_smem(
                    output.element_type, 128 * self.epi_n, alignment=128
                )
                cl = cute.make_composed_layout(
                    cute.make_swizzle(3, 3, 3),
                    0,
                    cute.make_layout((128, self.epi_n), stride=(self.epi_n, 1)),
                )
                shared_c = cute.make_tensor(cp, cl)
            c_tiles = cute.local_tile(
                transform_partitioned_tensor_layout(acc), (128, self.epi_n), (0, None)
            )
            atom = blackwell_helpers.get_tmem_load_op(
                (128, self.bn, self.bk),
                LayoutEnum.ROW_MAJOR,
                output.element_type,
                cutlass.Float32,
                (128, self.epi_n),
                self.cluster_m == 2,
            )
            copy = tcgen05.make_tmem_copy(atom, c_tiles[None, None, 0])
            thread = copy.get_slice(tid)
            destination = thread.partition_D(shared_c)
            regs = cute.make_rmem_tensor(destination.shape, cutlass.Float32)
            narrowed = cute.make_rmem_tensor(destination.shape, output.element_type)
            if cutlass.const_expr(self.prefetch_output_registers):
                full_regs = cute.make_rmem_tensor(
                    (*destination.shape, self.bn // self.epi_n), cutlass.Float32
                )
            else:
                full_regs = None
            store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), output.element_type, num_bits_per_copy=128
            )
            if cutlass.const_expr(self.tma_output):
                assert shared_d is not None
                r2s_atom = blackwell_helpers.get_smem_store_op(
                    LayoutEnum.ROW_MAJOR, output.element_type, cutlass.Float32, copy
                )
                r2s_copy = cute.make_tiled_copy_D(r2s_atom, copy)
                r2s_thread = r2s_copy.get_slice(tid)
                r2s_target = r2s_thread.partition_D(shared_d)
                r2s_source = r2s_copy.retile(narrowed)
            else:
                r2s_copy, r2s_source, r2s_target = None, None, None
            cstate = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.accumulator_stages
            )
            for tile_index in cutlass.range(
                first_tile, self.tile_count, tile_stride, unroll=1
            ):
                m_start = (
                    tile_index // self.n_tiles * self.logical_m_per_cta * self.cluster_m
                    + v * self.logical_m_per_cta
                )
                n_start = tile_index % self.n_tiles * self.bn
                if cutlass.const_expr(self.tma_output):
                    assert d_tma is not None and d_atom is not None
                    assert shared_d is not None
                    # Apply the CTA origin before tiling: a single row tile
                    # has zero grid stride and would discard the second CTA's
                    # out-of-bounds row origin, aliasing the first CTA's output.
                    output_tile = cute.local_tile(
                        cute.domain_offset((m_start, n_start), d_tma),
                        (128, self.bn),
                        (0, 0),
                    )
                    output_chunks = cute.local_tile(
                        output_tile, (128, self.epi_n), (0, None)
                    )
                    tma_shared, tma_global = cpasync.tma_partition(
                        d_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(shared_d, 0, 2),
                        cute.group_modes(output_chunks, 0, 2),
                    )
                else:
                    tma_shared, tma_global = None, None
                c_pipe.consumer_wait(cstate)
                current_acc = cute.make_tensor(tmem + cstate.index * self.bn, c_layout)
                current_c_tiles = cute.local_tile(
                    transform_partitioned_tensor_layout(current_acc),
                    (128, self.epi_n),
                    (0, None),
                )
                if cutlass.const_expr(self.prefetch_output_registers):
                    assert full_regs is not None
                    for ni in cutlass.range_constexpr(self.bn // self.epi_n):
                        cute.copy(
                            copy,
                            thread.partition_S(current_c_tiles[None, None, ni]),
                            full_regs[None, None, None, ni],
                        )
                    cute.arch.fence_view_async_tmem_load()
                    c_pipe.consumer_release(cstate)
                for ni in cutlass.range(
                    (self.bn + self.epi_n - 1) // self.epi_n,
                    unroll_full=self.prefetch_output_registers,
                ):
                    if cutlass.const_expr(self.prefetch_output_registers):
                        assert full_regs is not None
                        rounded = full_regs[None, None, None, ni].load() * alpha
                    else:
                        cute.copy(
                            copy,
                            thread.partition_S(current_c_tiles[None, None, ni]),
                            regs,
                        )
                        rounded = regs.load() * alpha
                    for round_index in cutlass.range_constexpr(len(self.round_dtypes)):
                        rounded = rounded.to(self.round_dtypes[round_index])
                    narrowed.store(rounded.to(output.element_type))
                    if cutlass.const_expr(self.tma_output):
                        assert store_pipe is not None and store_buffer is not None
                        assert r2s_copy is not None and r2s_source is not None
                        assert r2s_target is not None
                        assert tma_shared is not None and tma_global is not None
                        if warp == 0:
                            with cute.arch.elect_one():
                                store_pipe.producer_acquire()
                        epi_barrier.arrive_and_wait()
                        cute.copy(
                            r2s_copy,
                            r2s_source,
                            r2s_target[None, None, None, store_buffer],
                        )
                        cute.arch.fence_view_async_shared()
                        epi_barrier.arrive_and_wait()
                        if warp == 0:
                            with cute.arch.elect_one():
                                cute.copy(
                                    d_atom,
                                    tma_shared[None, store_buffer],
                                    tma_global[None, ni],
                                )
                                store_pipe.producer_commit()
                        store_buffer = (store_buffer + 1) % self.output_stages
                    else:
                        cute.autovec_copy(narrowed, destination)
                        epi_barrier.arrive_and_wait()
                        for vi in cutlass.range_constexpr(self.epi_n // 8):
                            flat = (tid + vi * 128) * 8
                            row = flat // self.epi_n
                            col = flat % self.epi_n
                            gm = m_start + row
                            gn = n_start + ni * self.epi_n + col
                            if (
                                gm < self.m
                                and gn < self.n
                                and row < self.logical_m_per_cta
                                and ni * self.epi_n + col < self.bn
                            ):
                                address = (
                                    cutlass.Int64(gm) * output_strides[0]
                                    + cutlass.Int64(gn) * output_strides[1]
                                )
                                if (
                                    cutlass.const_expr(output_pointer.alignment >= 16)
                                    and gn + 7 < self.n
                                    and ni * self.epi_n + col + 7 < self.bn
                                    and address % 8 == 0
                                    and output_strides[1] == 1
                                ):
                                    src = cute.make_tensor(
                                        shared_c.iterator
                                        + cute.assume(
                                            cute.crd2idx((row, col), shared_c.layout),
                                            divby=8,
                                        ),
                                        cute.make_layout(8),
                                    )
                                    dst = cute.make_tensor(
                                        output_pointer + cute.assume(address, divby=8),
                                        cute.make_layout(8),
                                    )
                                    cute.copy(store_atom, src, dst)
                                else:
                                    for tail in cutlass.range_constexpr(8):
                                        if (
                                            gn + tail < self.n
                                            and ni * self.epi_n + col + tail < self.bn
                                        ):
                                            output[
                                                cutlass.Int64(gm),
                                                cutlass.Int64(gn) + tail,
                                            ] = shared_c[row, col + tail]
                        epi_barrier.arrive_and_wait()
                if cutlass.const_expr(not self.prefetch_output_registers):
                    cute.arch.fence_view_async_tmem_load()
                    c_pipe.consumer_release(cstate)
                cstate.advance()
            if cutlass.const_expr(self.tma_output):
                assert store_pipe is not None
                if warp == 0:
                    with cute.arch.elect_one():
                        store_pipe.producer_tail()
                epi_barrier.arrive_and_wait()
        cute.arch.sync_threads()
        if cutlass.const_expr(self.cluster_m == 2):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        allocator.relinquish_alloc_permit()
        allocator.free(tmem)
