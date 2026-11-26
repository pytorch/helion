"""Planner: the tcgen05 GEMM search surface for one ``hl.dot`` / matmul candidate.

``_plan_cute_tcgen05_search_candidate`` turns a candidate's operand facts into
a frozen :class:`CuteTcgen05SearchPlan` (search limits, admitted families)
without touching shared compile state; ``enable_cute_tcgen05_search`` applies
one accepted plan to the bound kernel's ``ConfigSpec``.  ``device_ir`` calls
both while detecting the search surface.  Moved verbatim from
``helion/language/matmul_ops.py``, which keeps call-time forwarders.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch._dynamo.source import TensorPropertySource

from ....language.matmul_ops import _dot_dimensions
from ....language.matmul_ops import _problem_extent
from ....language.matmul_ops import _static_dim_value
from ....language.matmul_ops import _static_problem_extent
from ...compile_environment import CompileEnvironment
from ...compile_environment import FixedBlockSizeSource
from ...compile_environment import _is_supported_tensor_input_source
from ...compile_environment import _symint_free_symbols
from ..cp_async_operand import tcgen05_cp_async_a_strided_layout
from ..cute_warp_mma_gemm import WARP_MMA_MIN_BM
from ..cute_warp_mma_gemm import WARP_MMA_MIN_BN
from ..cute_warp_mma_gemm import warp_mma_admission_reason
from ..dense_aligned_n import DENSE_ALIGNED_N_SIXTEEN_BIT_BLOCK_K
from ..dense_aligned_n import carrier_n
from ..dense_aligned_n import dense_aligned_n_mn_b_width
from ..dense_aligned_n import n_major_b_one_wave_pair128_gain
from ..dense_aligned_n import transposed_double_edge_without_pairs
from ..tcgen05_config import facts
from ..tcgen05_config import surface
from ..tcgen05_constants import RAGGED_BATCHED_MIN_BLOCK_N
from ..tcgen05_constants import TCGEN05_MANY_SMALL_BATCHES
from ..tcgen05_constants import TCGEN05_N_MAJOR_B_ONE_WAVE_MAX_K_BLOCKS
from ..tcgen05_constants import TCGEN05_ONE_CTA_MAX_BLOCK_M
from ..tcgen05_constants import TCGEN05_ONE_CTA_MIN_BLOCK_M
from ..tcgen05_constants import TCGEN05_ONE_ROW_TILE_BATCHED_BLOCK_N
from ..tcgen05_constants import TCGEN05_PAIR_TILE_FIXED_KBLOCKS
from ..tcgen05_constants import TCGEN05_TRANSPOSED_BATCH_MIN_WAVES
from ..tcgen05_constants import TCGEN05_TRANSPOSED_PRODUCT_MIN_GAIN
from ..tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from ..tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_DEEP_BLOCK_K
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_MIN_DIM
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_PAIR128_BLOCK
from ..tcgen05_constants import TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
from ..tcgen05_constants import TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N
from ..tcgen05_constants import TCGEN05_TWO_CTA_ONE_WAVE_BLOCK_NS
from ..tcgen05_constants import TCGEN05_TWO_CTA_PAIR128_ONE_WAVE_BLOCK_NS
from ..tcgen05_constants import TCGEN05_WHOLE_K_TILE_MIN
from ..tcgen05_constants import tcgen05_edge_pair128_narrow_width_fits
from ..tcgen05_constants import tcgen05_transposed_pair_prefetch_fits
from ..tcgen05_constants import tcgen05_two_cta_max_k_tiles

if TYPE_CHECKING:
    from collections.abc import Iterable

    import sympy


def _cuda_num_sms_or_zero(device: torch.device) -> int:
    """Return the device SM count, or 0 on devices ``get_num_sm`` does not support.

    The wave-quantization gates of ``enable_cute_tcgen05_search`` and the
    flash, materialized-MMA and warp-MMA planners read it. The 0 fallback
    keeps cluster_m=2 search live for configuration round-trip tests that
    bind on CPU or other unsupported device types. Tests patch it here.
    """
    if device.type != "cuda":
        return 0
    # Local import: ``helion.runtime`` is in the import chain that loads
    # this module, so a top-level import would be circular.
    from ....runtime import get_num_sm

    try:
        return get_num_sm(device)
    except (AssertionError, NotImplementedError):
        return 0


def tcgen05_mma_k(dtype: torch.dtype, *, tf32: bool = False) -> int:
    """K extent of one tcgen05 MMA instruction for ``dtype`` operands."""
    return 32 if dtype == torch.float8_e4m3fn else (8 if tf32 else 16)


def tcgen05_min_n(dtype: torch.dtype, *, tf32: bool = False) -> int:
    """The narrowest tcgen05 N tile for ``dtype`` operands (tf32's bn >= 32
    floor mirrors ``_mma_impl_matches_problem_shape``)."""
    return 16 if dtype == torch.float8_e4m3fn else (32 if tf32 else 8)


def tcgen05_small_n_zero_fills(m: int, n: int, k: int, dtype: torch.dtype) -> bool:
    """Whether an ``m x n x k`` GEMM whose static ``n`` is below the
    narrowest tcgen05 N tile runs one zero-filled tile of that width
    (``_plan_cute_tcgen05_search_candidate``): 16-bit operands of a
    memory-bound GEMM, ``m`` and ``k`` each at least 128, as a short M runs
    one zero-filled 64-row tile (``tcgen05_shape_rejected``). The planner
    asks it for an N of 2 to 4 whose N block may be 8 wide
    (``_n_block_admits``) and whose scalar-fallback block is narrower
    than 8 columns (an N of 5 to 7 rounds up to the 8-column tile anyway),
    and searches that tile where TMA loads B and the kernel's K block can be
    ``TCGEN05_SMALL_N_MIN_K_BYTES`` deep; helion.inductor asks it before it
    transposes an M under 8 (``tcgen05_transposes_skinny_product``)."""
    return (
        n < tcgen05_min_n(dtype)
        and dtype in (torch.bfloat16, torch.float16)
        and m >= 128
        and k >= 128
    )


# The K tile depth, in bytes, the zero-filled 8-column tile of a small N
# searches from: CuTe refuses an 8-column K-major B stage of 32-byte rows
# (``cute_mma``: "tcgen05 K-major B TMA with block_n=8 needs block_k *
# itemsize >= 64").
TCGEN05_SMALL_N_MIN_K_BYTES = 64


def _k_block_reaches(
    env: CompileEnvironment, k: int | torch.SymInt, minimum: int
) -> bool:
    """Whether a matmul's K tile, of size ``k`` (an operand tile's K extent),
    can be ``minimum`` deep: a K block whose range reaches it (a fixed
    integer ``block_size=`` at least that, an ``hl.register_block_size``
    range or the extent's power of two up to it), or a static K taken
    whole."""
    block_id = env.get_block_id(k)
    if block_id is None:
        static_k = _static_dim_value(env, k)
        return static_k is not None and static_k >= minimum
    try:
        spec = env.config_spec.block_sizes.block_id_lookup(block_id)
    except KeyError:
        source = env.block_sizes[block_id].block_size_source
        return (
            isinstance(source, FixedBlockSizeSource)
            and isinstance(source.value, int)
            and source.value >= minimum
        )
    return spec.max_size >= minimum


def _n_block_admits(env: CompileEnvironment, n: int | torch.SymInt, width: int) -> bool:
    """Whether a matmul's N tile, of size ``n`` (an operand tile's N extent),
    can be ``width`` wide: an N block left to the search, one registered
    (``hl.register_block_size``) over a range holding ``width``, or one fixed
    at that integer ``block_size=``. A registered range bounds the search,
    and the block size may size the kernel's allocations, so a range that
    leaves ``width`` out (``hl.register_block_size(4, 4)``) keeps its tiles."""
    block_id = env.get_block_id(n)
    if block_id is None:
        return False
    try:
        spec = env.config_spec.block_sizes.block_id_lookup(block_id)
    except KeyError:
        source = env.block_sizes[block_id].block_size_source
        return (
            isinstance(source, FixedBlockSizeSource)
            and isinstance(source.value, int)
            and source.value == width
        )
    bounds = spec.registered_bounds
    return bounds is None or bounds[0] <= width <= bounds[1]


def tcgen05_shape_rejected(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    *,
    tf32: bool = False,
    static_m: bool = True,
    static_n: bool = True,
    batch: int | None = None,
    small_n_supported: bool = False,
) -> bool:
    """Whether the planner leaves an ``m x n x k`` dot of ``dtype`` operands
    off the tcgen05 search for its shape alone (``m`` / ``n`` / ``k`` are the
    extents, or their hints where ``static_m`` / ``static_n`` say they are
    dynamic). ``batch``: the static leading batch of a rank-3 operand, if any;
    ``small_n_supported``: an N below the narrowest tile has a lowering (role-
    local TMA or the scalar fallback). helion.inductor asks it too, before it
    writes a dot's K loop (``helion.inductor.choices.dot_keeps_k_loop``)."""
    if not static_m:
        # A dynamic M keeps the 64-row floor.
        rows_rejected = m < 64
    else:
        rows_rejected = (
            # Fewer than 64 static rows run as one zero-filled 64-row tile
            # on the flat pipeline (decode GEMMs: bf16 16x14336x4096 27 us
            # against the SIMT fallback's 358 us), but a single row stays
            # SIMT (a TMA box over an extent-1 dimension ran 1x4096x4096 at
            # 327 us against 20 us for two rows).
            m < 2
            # Only the memory-bound decode shapes profit from the padded
            # tile; a tiny GEMM (33^3) keeps the SIMT lowering, which wins
            # there (2.8 us against the tile's 13.8).
            or (
                m < 64
                and (n < 128 or k < 128)
                # A batch of such matrices runs one zero-filled tile each
                # and fills the machine (window attention's 32768 x
                # 56x56x32: 14.0 ms on the SIMT lowering, 0.23 ms on the
                # one-CTA 64x64x32 tile).
                and (batch or 0) < TCGEN05_MANY_SMALL_BATCHES
            )
        )
    min_n = tcgen05_min_n(dtype, tf32=tf32)
    return (
        rows_rejected
        # Size-one tile axes are fixed to block size one before this
        # analysis, so they cannot carry the minimum tcgen05 instruction tile.
        or (n < 2 if static_n else n < min_n)
        or (static_n and n < min_n and not small_n_supported)
        or k < tcgen05_mma_k(dtype, tf32=tf32)
    )


def tcgen05_transposes_skinny_product(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    *,
    a_strides: tuple[int, int],
    b_strides: tuple[int, int],
    num_sms: int,
    a_offset: int = 0,
    b_offset: int = 0,
) -> bool:
    """Whether a 16-bit ``m x n x k`` product with a skinny ``m`` runs better
    as its transpose (``n x m x k``, out.T = b.T @ a.T) on the CuTe GEMM
    families, for a row-major ``a`` and a ``b`` that is K-major or N-major
    (``a_strides`` / ``b_strides``: the operands' element strides; any
    other layout stays as written). helion.inductor asks it before it
    lowers an ``aten.mm`` (``HelionChoices.transpose_matmul``).

    The tcgen05 MMA's M side holds 64 or more rows, its N side as few as 8
    columns. As written, a skinny M sits on the M side and the weight b
    streams on the N side in slabs as narrow as the grid needs. bf16
    128x768x1000 with an N-major b (a Linear's weight stored [K, N]):

        for tile_y, tile_x in hl.tile([128, 768]):  # [64, 32, 128]: 2 x 24 CTAs
            acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
            for tile_r in hl.tile(1000):
                tmp0 = in0[tile_y, tile_r]  # a: 64 x 128 box per stage
                tmp1 = in1[tile_r, tile_x]  # b: 128 rows of 32 contiguous
                                            # elements, a 64-byte TMA row each

    and every CTA re-reads 64 rows of a. Transposed, the weight is the MMA's
    A operand, 64-row slabs whose N-contiguous rows are 128-byte TMA rows,
    and torch M runs on the N side in 16-column tiles:

        for tile_x, tile_y in hl.tile([768, 128]):  # [64, 16, 128]: 12 x 8 CTAs
            acc = hl.zeros([tile_x, tile_y], dtype=torch.float32)
            for tile_r in hl.tile(1000):
                tmp0 = in0[tile_x, tile_r]  # b.T: 64 x 128 box, 128-byte rows
                tmp1 = in1[tile_r, tile_y]  # a.T: 16 x 128 box (K-major)

    GB300, both orientations autotuned at full effort in one process, of
    stock Inductor (cuBLAS): 0.711 -> 0.900 (128x640x1000, 32x1280x1000 the
    same). An N-major b transposes while 64-column slabs give the as-written
    grid fewer tiles than the device has SM pairs (narrower slabs are needed
    to fill it): 128x768x768 0.947 -> 0.987, 256x512x512 0.973 -> 0.995,
    64x4096x4096 (64 tiles) 0.976 -> 0.992, but 128x4096x4096 (128 tiles)
    0.991 -> 0.983 stays. A K-major b loads its slabs in 128-byte rows either
    way, so only an M below the one-CTA tile's 64 rows, which the as-written
    tile zero-fills, transposes: 32x1280x960 0.829 -> 0.925, while
    128x768x768 (1.003 -> 0.982) and 64x4096x4096 (1.004 -> 1.006) stay.
    Over whole 64-element K tiles, an N-major b of an M of one 128-row pair
    row also transposes where the transposed product runs a one-wave grid of
    128x64 CTA pairs whose operand footprint their L2 prefetch takes
    (``tcgen05_transposed_pair_prefetch_fits``, which counts b's row stride
    and the operands' offsets as the prefetch does), each CTA reading one
    whole line of b.T per K index where the as-written pair's CTAs read half
    lines of b: bf16 128x3072x768 (48 pairs) at full effort 0.95 as written
    (0.87 once the as-written pairs prefetch too: the search keeps a one-CTA
    tile) and 1.10 transposed; the best as-written and transposed configs
    by hand, of cuBLAS: 128x3072x768 1.05 / 1.11, 128x4096x768 1.03 / 1.09,
    128x4096x1024 1.03 / 1.09 (the same pair config either way: 1.05 ->
    1.09 at 128x3072x512, 1.03 -> 1.04 at 128x3072x1024). A K tail keeps it
    as written, where the 64-wide pair's 2x2 cluster with its shared
    prefetch runs at least as fast: 128x3072x1000 1.017 as written against
    1.002, 128x4096x1000 1.055 against 1.056. The pairs load both operands
    through TMA, so this clause asks both operands' strides on its 16-byte
    alignment (``_operands_tma_aligned``) and keeps other products as
    written, b's rows that the flat one-CTA kernel copies with ``cp.async``
    included: over b's rows sliced from [768, 3076] (6152 bytes apart) the
    CuTe backend leaves b.T, the transposed product's M-major A, to those
    copies and refuses the transposed 128x64 pair config. No clause
    transposes over an operand whose base (``a_offset`` / ``b_offset``) is
    off that alignment, which neither TMA nor ``cp.async`` loads either way
    round: bf16 128x768x1000 over a b 8 bytes into its storage runs the
    scalar producers as written and transposed.
    A quick-effort sweep of 60 shapes (M 16 to 512 by N x K in 768 x 768,
    1024 x 4096, 4096 x 1024, 4096 x 4096 and 1280 x 1000, both layouts),
    before the CTA pairs prefetched their operands, ran the shapes inside
    the slab rule 0.98 to 1.43 times as fast transposed and those outside it
    0.83 to 1.03 times (128x4096x1024 among them, which the pair clause
    above now transposes), but for three of the K-tail 1280 x 1000 shapes
    (1.06 to 1.16), which both orientations run on the flat one-CTA kernel
    at 0.61 to 0.80 of stock Inductor.

    The warp-MMA family tiles M in 16-row steps, so a skinny M fits it as
    written: a product it admits (``warp_mma_admission_reason``, the
    latency-bound regime) stays as written. bf16 16x768x768 with a K-major
    b: as written (warp MMA, [16, 32, 128]) 1.385 against 1.204 transposed
    (tcgen05 [64, 8, 128]); 32x768x768 1.387 against 1.158 (quick effort).
    An ``m`` below the narrowest tcgen05 N tile (8 for 16-bit operands) is
    the transposed product's N, which a memory-bound GEMM runs as one
    zero-filled 8-column tile (``tcgen05_small_n_zero_fills``). bf16
    4x4096x4096 with a K-major b, as written one zero-filled 64-row tile of
    a by 32-column slabs of b on 128 CTAs, every one of which reads all of
    a's 32 KB: 16.9 us (with a's four rows 64 KB apart instead, 8.6 us);
    transposed, 64 CTAs of [64, 8, 128] (a.T's 8 x 128 box, rows 4-7 zero):
    8.5 us, 1.06 of cuBLAS's split-K kernel.

    The measured operands loaded through TMA. An operand that TMA loads
    neither way round keeps the product as written, unmeasured either way
    (``_tma_strides_aligned``): a's rows off TMA's 16-byte alignment are
    a.T's, the transposed product's K-major B, and a K-major b's columns
    b.T's, its K-major A, either of which takes TMA off both operands (bf16
    32x768x768 over a's rows sliced from [32, 772], 1544 bytes apart and
    24,576 elements, too few for the CuTe backend to stage, renders the
    scalar producers both ways round), as does an overlapping view of
    strides (1, 1). An N-major b's rows off that alignment are b.T's
    columns, an M-major A, which the flat one-CTA kernel copies with
    ``cp.async`` beside a.T's TMA loads where its producer takes them
    (``tcgen05_cp_async_a_strided_layout``: rows an even count of elements
    apart over an even n), while as written TMA loads neither operand over
    them (the scalar producers run, or the CuTe backend first copies a b of
    ``STAGING_MIN_ELEMENTS`` elements or more): such rows do not keep the
    product as written in the slab clauses (unmeasured; the pair clause
    above keeps them), the others do. The warp-MMA admission
    reads the real strides, as the planner's does: over b's rows sliced
    from [256, 776] (16-byte aligned) the family runs bf16 32x768x256 as
    written; from [256, 772] it admits no warp MMA, and the product
    transposes, a.T through TMA and b.T through ``cp.async`` (the scalar
    producers as written), but from [256, 771] (odd) it stays.
    """
    if dtype not in (torch.bfloat16, torch.float16):
        return False
    b_k_major = b_strides[0] == 1
    if a_strides[1] != 1 or not (b_k_major or b_strides[1] == 1):
        return False
    if any(
        offset * dtype.itemsize % TCGEN05_TMA_ROW_ALIGN_BYTES
        for offset in (a_offset, b_offset)
    ):
        # (an operand based off TMA's 16-byte alignment: unmeasured either
        # way round)
        return False
    if not (1 < m < n) or tcgen05_shape_rejected(m, n, k, dtype):
        # (a product the planner leaves to the SIMT or warp-MMA lowering as
        # written is too small for its tiles to matter: bf16 40x96x64)
        return False
    if m < tcgen05_min_n(dtype) and not tcgen05_small_n_zero_fills(n, m, k, dtype):
        # (an M under the narrowest tcgen05 N tile is the transposed
        # product's N: its zero-filled N tile, or no tcgen05 tile)
        return False
    slabs = TCGEN05_ONE_CTA_MIN_BLOCK_M
    if not (
        m < slabs
        or (not b_k_major and -(-m // slabs) * -(-n // slabs) < num_sms // 2)
        or (
            not b_k_major
            # (one 128-row pair row as written, whose 64-wide pairs read half
            # lines of b: the shapes measured)
            and m <= TCGEN05_TWO_CTA_BLOCK_M // 2
            and k % TCGEN05_WHOLE_K_TILE_MIN == 0
            and tcgen05_transposed_pair_prefetch_fits(
                m,
                n,
                k,
                itemsize=dtype.itemsize,
                a_row_bytes=a_strides[0] * dtype.itemsize,
                b_row_bytes=b_strides[0] * dtype.itemsize,
                num_sms=num_sms,
                a_offset_bytes=a_offset * dtype.itemsize,
                b_offset_bytes=b_offset * dtype.itemsize,
            )
            # (the pairs load both operands through TMA)
            and _operands_tma_aligned(a_strides, b_strides, dtype.itemsize)
        )
    ):
        return False
    return (
        warp_mma_admission_reason(
            static_m=m,
            static_n=n,
            static_k=k,
            leading=1,
            dtype=dtype,
            lhs_major="row",
            rhs_major="col" if b_k_major else "row",
            num_sms=num_sms,
            lhs_strides=a_strides,
            rhs_strides=b_strides,
        )
        is not None
        # (operands TMA loads neither way round stay as written)
        and _tma_strides_aligned(a_strides, dtype.itemsize)
        and (
            _tma_strides_aligned(b_strides, dtype.itemsize)
            # (an N-major b's rows transposed are an M-major A's columns,
            # which the flat one-CTA kernel copies with cp.async)
            or (
                not b_k_major
                and tcgen05_cp_async_a_strided_layout(
                    torch.empty_strided(
                        (n, k), (1, b_strides[0]), dtype=dtype, device="meta"
                    )
                )
                is not None
            )
        )
    )


# Bytes per L2 sector: the granule a TMA row's L2 reads and an output row's
# stores move in.
TCGEN05_L2_SECTOR_BYTES = 32
# The row alignment a TMA descriptor needs (global strides in 16-byte units).
TCGEN05_TMA_ROW_ALIGN_BYTES = 16


def _tma_strides_aligned(strides: tuple[int, int], itemsize: int) -> bool:
    """Whether one operand's element strides pass the TensorMap rule the
    CuTe backend proves before it loads the operand through TMA
    (``cute_mma._tcgen05_tma_tensor_is_aligned``; over a kernel argument
    ``memory_ops.tensor_has_specialized_tma_alignment``): exactly one unit
    stride, and every other stride on TMA's 16-byte alignment. (The rule's
    16-byte aligned base is not in the strides.) An overlapping view fails
    it: ``torch.randn(639).unfold(0, 128, 1)`` is [512, 128] of strides
    (1, 1)."""
    return sum(stride == 1 for stride in strides) == 1 and all(
        stride == 1 or stride * itemsize % TCGEN05_TMA_ROW_ALIGN_BYTES == 0
        for stride in strides
    )


def _operands_tma_aligned(
    a_strides: tuple[int, int], b_strides: tuple[int, int], itemsize: int
) -> bool:
    """Whether both operands pass TMA's stride rule
    (``_tma_strides_aligned``), as the rules that transpose a product onto
    the CuTe CTA pairs ask. Transposed, a row-major a's rows are a.T's, the
    product's K-major B, and a K-major b's columns b.T's, its K-major A.
    Either way round an operand off the rule takes TMA off both operands
    (``plan_tcgen05_pipeline``: only an M-major A beside an aligned B keeps
    the pipeline, the flat one-CTA kernel copying A with ``cp.async``), and
    then no persistent CTA pair runs: 16-bit pairs fall back to the scalar
    producers behind a host guard that refuses every launch, and tf32 ones
    lower no tcgen05 pair. The transposed persistent 256x256x64 2x2 config
    of bf16 512x20008x128 loads neither operand through TMA over a's rows
    sliced from [512, 129] (258 bytes apart) or over an overlapping a of
    strides (1, 1), and both over a contiguous a. The CuTe backend copies
    such an operand of ``STAGING_MIN_ELEMENTS`` elements or more first
    (``stage_aligned_operands``), row-major, which makes a.T (transposed)
    and a K-major b (as written) N-major, layouts these rules did not
    measure; helion.inductor copies a 16-bit one, K-major still, before it
    asks them (``HelionChoices.stage_matmul_operand``)."""
    return _tma_strides_aligned(a_strides, itemsize) and _tma_strides_aligned(
        b_strides, itemsize
    )


def tcgen05_transposes_misaligned_b_rows(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    *,
    a_strides: tuple[int, int],
    b_strides: tuple[int, int],
    num_sms: int,
) -> bool:
    """Whether a 16-bit ``m x n x k`` product over an N-major ``b`` whose
    rows straddle L2 sectors runs better as its transpose (out.T = b.T @
    a.T) on the CuTe CTA pairs, for a row-major ``a`` (``a_strides`` /
    ``b_strides``: element strides). helion.inductor asks it before it
    lowers an ``aten.mm`` (``HelionChoices.transpose_matmul``).

    TMA needs 16-byte aligned rows, not sector-aligned ones: bf16 b of
    shape [768, 20008] stores its rows 40016 bytes apart, an odd multiple
    of 16, so every other 128-byte TMA row of b starts mid-sector and moves
    five L2 sectors, in two requests, for four sectors of data. As written,
    each CTA of the pair loads its own half of b's N tile; transposed, b.T
    is the product's A operand, which the 2x2 cluster (``tcgen05_cluster_n=
    2``: two CTA pairs side by side in the transposed N, so over the same b.T
    rows) multicasts, halving those requests per CTA. cuBLAS runs the same
    multicast (nvjet ``2x2f``, B shared by two pairs).

        for tile_y, tile_x in hl.tile([2048, 20008]):  # as written
            ...
            tmp1 = in1[tile_r, tile_x]  # b: half its N tile per CTA
        for tile_x, tile_y in hl.tile([20008, 2048]):  # transposed: 2x2
            ...                                       # clusters of pairs
            tmp0 = in0[tile_x, tile_r]  # b.T: multicast to both pairs

    GB300, of cuBLAS in one process, 256x256 pair tiles: as written /
    transposed in 2x2 clusters (persistent over the 36 co-resident clusters,
    L2 grouping 4), and the full-effort transposed autotune: bf16
    2048x20008x768 0.880-0.887 / 1.008-1.018 / 1.008; 4096x128104x1536
    0.960-0.965 / 1.032-1.040 / 1.011; 4096x50264x2048 0.900-0.905 /
    0.936-0.939 / 0.941. Sector-aligned rows are not this rule's: without
    its pair fallback the 2x2 cluster idles the 8 SMs the 36 co-resident
    clusters leave (GB300's 152): 8192x50272x768 0.997-0.999 as written,
    0.96-0.97 transposed in 2x2 clusters; 8192x50272x1024 1.015-1.021 /
    0.92-0.94; 4096x50272x2048 0.986-0.993 / 0.93-0.94 (with the fallback,
    ``tcgen05_transposes_n_edge_b_rows`` transposes the first, whose K loop
    of 12 K blocks is short enough; 16 and 32 stay as written). The same
    4096xNx2048 config as written runs 0.991 at a 128-byte aligned N,
    0.984-0.988 at 32- and 64-byte aligned ones, and 0.905-0.909 at both
    odd-16-byte ones tried (N = 50248, 50264).

    The transposed 2x2 cluster pairs the transposed product's N tiles
    (``m / 256`` must be whole and even), and a grid of at least two waves
    of co-resident 2x2 clusters keeps the clusters busy
    (``_transposed_clusters_fill_two_waves``). Rows TMA cannot read
    (not 16-byte aligned: bf16 N = 20001, 40002 bytes apart) are b.T's rows
    too, so neither orientation multicasts them, nor does either over a's
    rows off that alignment, a.T's rows (``_operands_tma_aligned``: bf16
    512x20008x128 over a's rows sliced from [512, 129]), and a product the
    tcgen05 planner rejects (``tcgen05_shape_rejected``: bf16 K = 8) runs
    neither orientation on the pairs; all stay as written.
    """
    if dtype not in (torch.bfloat16, torch.float16):
        return False
    itemsize = 2
    if a_strides[1] != 1 or b_strides[1] != 1:
        return False
    if b_strides[0] * itemsize % TCGEN05_L2_SECTOR_BYTES != TCGEN05_TMA_ROW_ALIGN_BYTES:
        # (sector-aligned rows need no multicast; rows off TMA's 16-byte
        # alignment cannot take it)
        return False
    if not _operands_tma_aligned(a_strides, b_strides, itemsize):
        return False
    if tcgen05_shape_rejected(m, n, k, dtype):
        return False
    if min(m, n) < TCGEN05_TWO_CTA_BLOCK_M:
        # (a skinny product is ``tcgen05_transposes_skinny_product``'s)
        return False
    return _transposed_clusters_fill_two_waves(m, n, num_sms)


def _transposed_clusters_fill_two_waves(m: int, n: int, num_sms: int) -> bool:
    """Whether the transposed product (out.T = b.T @ a.T) of an ``m x n``
    output runs its 2x2 clusters of 256x256 CTA pairs over whole cluster
    columns, at least two waves of them: the transposed product tiles ``m``
    by its 256-column pair width and ``n`` by its 256-row pair height, the
    cluster pairs two of its ``m / 256`` N tiles, and a wave counts
    ``num_sms // 4`` clusters, an upper bound on the co-resident ones
    (GB300's 152 SMs hold 36: ``tcgen05_transposes_misaligned_b_rows``).

    The transposed N tiles must be whole as well as even, as measured: ``n``
    may already end in a partial tile (the transposed M's; it always does
    under ``tcgen05_transposes_n_edge_b_rows``), and a partial N tile on top
    of it makes a double edge, which leaves the transposed product without
    the CTA pair's search family where 64-row tiles divide ``n``
    (``transposed_double_edge_without_pairs``: bf16 [2040, 512] @ [512,
    20032] transposed binds no ``cluster_m2_search_constraints`` and
    defaults to a flat one-CTA 128x16x16 tile, where as written it runs
    256x256 pairs)."""
    if m % (2 * TCGEN05_TWO_CTA_BLOCK_N):
        return False
    clusters = m // (2 * TCGEN05_TWO_CTA_BLOCK_N) * -(-n // TCGEN05_TWO_CTA_BLOCK_M)
    return clusters >= 2 * (num_sms // 4)


# The K loops, in 64-wide K blocks, over which the transposed 2x2 cluster's
# multicast of b.T outruns the as-written CTA pair on an N edge
# (``tcgen05_transposes_n_edge_b_rows``).
TCGEN05_N_EDGE_MULTICAST_MIN_K_BLOCKS = 6
TCGEN05_N_EDGE_MULTICAST_MAX_K_BLOCKS = 12
# The fewest 256-wide N tiles of such a product: its partial last tile then
# costs the transposed product at most a 64th of its 256-row tiles (the
# measured products run 120 to 197 N tiles).
TCGEN05_N_EDGE_MULTICAST_MIN_N_TILES = 64


def tcgen05_transposes_n_edge_b_rows(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    *,
    a_strides: tuple[int, int],
    b_strides: tuple[int, int],
    num_sms: int,
) -> bool:
    """Whether a 16-bit ``m x n x k`` product over an N-major ``b`` whose N
    ends in a partial 256-wide pair tile runs better as its transpose
    (out.T = b.T @ a.T) on the CuTe CTA pairs, for a row-major ``a``
    (``a_strides`` / ``b_strides``: element strides). helion.inductor asks
    it before it lowers an ``aten.mm`` (``HelionChoices.transpose_matmul``).

    As written no cluster multicasts such a ``b``: the 2x2 cluster's
    partial-N family reads a K-major B only
    (``n_edge_cluster_geometry_supported``: bf16 32768x30528x768 with b
    stored [K, N] refuses ``tcgen05_cluster_n=2``), and the M-paired
    512-row tile needs whole N tiles. Each CTA of the pair loads its own
    half of b's N tile:

        for tile_y, tile_x in hl.tile([32768, 30528]):  # as written
            ...
            tmp1 = in1[tile_r, tile_x]  # b: half its N tile per CTA

    Transposed, b.T is the product's A operand and the N edge is the
    transposed M's, which the 2x2 cluster takes, its pair fallback
    (``tcgen05_fallback_cluster``) keeping the SMs the co-resident 4-CTA
    clusters leave busy; the two pairs of a cluster share one b.T tile, a
    quarter of each pair tile's operand bytes from L2:

        for tile_x, tile_y in hl.tile([30528, 32768]):  # transposed: 2x2
            ...                                       # clusters of pairs
            tmp0 = in0[tile_x, tile_r]  # b.T: multicast to both pairs

    At the power cap the fewer L2 reads run at a higher clock. GB300, each
    alone in CUDA graphs over cold inputs (the blocks metric), of cuBLAS,
    the best of two as-written pair configs (256x256x64, six stages, L2
    grouping 32; 256x256x128, three stages, grouping 64) against the best
    transposed 2x2 config with the pair fallback (static or CLC schedule,
    grouping 32):

        K       16384x50272    32768x30528    8192x50272
        256     1.040 / 1.035  1.041 / 1.047
        384     1.026 / 1.034  1.012 / 1.025
        512     1.019 / 1.050  1.003 / 1.029  1.009 / 1.044
        640     1.010 / 1.046
        768     1.004 / 1.025  1.002 / 1.021  1.005 / 1.023
        1024    1.017 / 1.018  1.006 / 1.019  1.016 / 1.013
        1536    1.016 / 1.012

    and 4096x50272x768 1.001 / 1.021. A K loop of
    ``TCGEN05_N_EDGE_MULTICAST_MIN_K_BLOCKS`` to
    ``TCGEN05_N_EDGE_MULTICAST_MAX_K_BLOCKS`` K blocks of 64 transposes.
    Shorter ones are epilogue-bound, where the 2x2 cluster and its fallback
    gain nothing (32768x30528x128, two K blocks: 1.07 as written or
    transposed without the cluster, 0.99 with it); four and sixteen tie.
    Rows that straddle L2 sectors are
    ``tcgen05_transposes_misaligned_b_rows``'s, and the transposed 2x2
    cluster needs what that rule's cluster needs: a's rows on TMA's 16-byte
    alignment (``_operands_tma_aligned``: bf16 512x20032x384 over a's rows
    sliced from [512, 385] stays as written) and whole transposed N tiles
    (``_transposed_clusters_fill_two_waves``). A narrow N is not this
    rule's: N = 288 (two pair tiles) transposed would compute 512 rows for
    288, and ``tcgen05_copies_n_major_b`` runs it over a K-major copy of b
    instead (bf16 131072x288x576: 0.929 of cuBLAS as written, 1.064 over
    the copy); the rule asks ``TCGEN05_N_EDGE_MULTICAST_MIN_N_TILES``.
    """
    if dtype not in (torch.bfloat16, torch.float16):
        return False
    itemsize = 2
    if a_strides[1] != 1 or b_strides[1] != 1:
        return False
    if b_strides[0] * itemsize % TCGEN05_L2_SECTOR_BYTES:
        # (rows that straddle sectors are
        # ``tcgen05_transposes_misaligned_b_rows``'s; rows off TMA's 16-byte
        # alignment take no multicast either way round)
        return False
    if not _operands_tma_aligned(a_strides, b_strides, itemsize):
        return False
    if n % TCGEN05_TWO_CTA_BLOCK_N == 0:
        # (whole N tiles keep the as-written paired forms: the 2x2 cluster
        # over an even count, the M-paired 512-row tile)
        return False
    if -(-n // TCGEN05_TWO_CTA_BLOCK_N) < TCGEN05_N_EDGE_MULTICAST_MIN_N_TILES:
        return False
    k_blocks, k_tail = divmod(k, 64)
    if k_tail or not (
        TCGEN05_N_EDGE_MULTICAST_MIN_K_BLOCKS
        <= k_blocks
        <= TCGEN05_N_EDGE_MULTICAST_MAX_K_BLOCKS
    ):
        return False
    if tcgen05_shape_rejected(m, n, k, dtype):
        return False
    if min(m, n) < TCGEN05_TWO_CTA_BLOCK_M:
        return False
    return _transposed_clusters_fill_two_waves(m, n, num_sms)


def tcgen05_transposes_misaligned_a_rows(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    *,
    a_strides: tuple[int, int],
    b_strides: tuple[int, int],
    num_sms: int,
) -> bool:
    """Whether a 16-bit one-wave ``m x n x k`` product of a row-major ``a``
    whose rows straddle L2 sectors and an N-major ``b`` runs better as its
    transpose (out.T = b.T @ a.T) on the CuTe CTA pairs (``a_strides`` /
    ``b_strides``: element strides). helion.inductor asks it before it
    lowers an ``aten.mm`` (``HelionChoices.transpose_matmul``).

    bf16 a of shape [512, 1000] stores its rows 2000 bytes apart, an odd
    multiple of 16, so every other TMA row of a starts mid-sector; and b of
    shape [1000, 1280], stored [K, N], leaves the as-written 128-row CTA
    pairs (each CTA 64 rows of the tile) only the widths whose half rows of b
    hold 64-byte swizzle atoms. Its 2 x 5 tiles of 256x256 are one wave of
    GB300's 76 SM pairs, and so is either orientation's best grid:

        for tile_y, tile_x in hl.tile([512, 1280]):  # as written: 40 pairs
            ...                                       # of 128x128
            tmp0 = in0[tile_y, tile_r]  # a: 64 straddling rows per CTA
        for tile_x, tile_y in hl.tile([1280, 512]):  # transposed: 70 pairs
            ...                                       # of 128x80
            tmp1 = in1[tile_r, tile_y]  # a.T: 40 straddling rows per CTA

    (``dense_aligned_n.n_major_b_one_wave_pair128_gain`` prices the two grids
    at 1.13.) GB300, cold full-effort autotunes, both orientations forced, of
    cuBLAS (CUDA graphs over cold inputs, each mode in its own blocks), as
    written / transposed: 512x1280x1000 0.779 / 0.903, 640x2048x1000 0.674 /
    0.910. Padding a's rows to 2048 bytes speeds the as-written pick up 12%
    and the transposed one 4% (cuBLAS 3-4%). Over 1152 rows the transposed
    pairs run no aligned width (the pricing's gate), and 768x1152x1000
    (0.831 / 0.826) and 640x1152x1000 (0.793 / 0.797) stay as written.
    Sector-aligned rows of a stay as written too, where the model's gains
    (1.08-1.09) do not hold: 1024x768x1024 0.968 / 0.952 (1024x1280x1024,
    0.825 / 0.958, stays too). Measured at K = 1000 (16 K blocks); a K loop no
    longer than the pair tile's fixed cost (``TCGEN05_PAIR_TILE_FIXED_KBLOCKS``)
    or past ``TCGEN05_N_MAJOR_B_ONE_WAVE_MAX_K_BLOCKS`` keeps the
    orientation, as do a K-major b, a grid of a wave or more, and a skinny
    side (``tcgen05_transposes_skinny_product``'s).

    Both grids are CTA pairs, which load both operands through TMA, so b's
    rows off TMA's 16-byte alignment keep the product as written too
    (``_operands_tma_aligned``): as written they are the product's N-major
    B, transposed b.T's columns, an M-major A that only the flat one-CTA
    kernel copies (``cp.async``), and either way round TMA loads neither
    operand. bf16 960x640x360 over b's rows sliced from [360, 644] (1288
    bytes apart; 230,400 elements, too few to stage): the transposed 128x64
    pair and the as-written 128x128 one fall back to the scalar producers
    behind a host guard that refuses every launch, where over b's rows from
    [360, 640] the transposed pair loads both operands through TMA, and the
    product transposes there and over rows from [360, 648]. The CuTe
    backend first copies a b of ``STAGING_MIN_ELEMENTS`` elements or more
    (``stage_aligned_operands``), row-major, which makes the transposed
    product's b.T K-major, unless it leaves b.T to the flat one-CTA
    kernel's ``cp.async`` (beside an a.T at most ``CP_ASYNC_MAX_N`` wide):
    layouts the rule did not measure."""
    if dtype not in (torch.bfloat16, torch.float16):
        return False
    itemsize = 2
    if a_strides[1] != 1 or b_strides[1] != 1 or b_strides[0] == 1:
        return False
    if a_strides[0] * itemsize % TCGEN05_L2_SECTOR_BYTES != TCGEN05_TMA_ROW_ALIGN_BYTES:
        return False
    if not _operands_tma_aligned(a_strides, b_strides, itemsize):
        # (b's rows off TMA's alignment: neither orientation's pairs load
        # through TMA)
        return False
    if tcgen05_shape_rejected(m, n, k, dtype):
        return False
    if min(m, n) < TCGEN05_TWO_CTA_BLOCK_M:
        # (a skinny product is ``tcgen05_transposes_skinny_product``'s)
        return False
    sm_pairs = num_sms // 2
    if -(-m // TCGEN05_TWO_CTA_BLOCK_M) * -(-n // TCGEN05_TWO_CTA_BLOCK_N) >= sm_pairs:
        return False
    if not (
        TCGEN05_PAIR_TILE_FIXED_KBLOCKS
        < -(-k // DENSE_ALIGNED_N_SIXTEEN_BIT_BLOCK_K)
        <= TCGEN05_N_MAJOR_B_ONE_WAVE_MAX_K_BLOCKS
    ):
        return False
    gain = n_major_b_one_wave_pair128_gain(m, n, k, sm_pairs=sm_pairs)
    return gain is not None and gain >= TCGEN05_TRANSPOSED_PRODUCT_MIN_GAIN


# The longest K loop, in 64-wide K blocks, under which the epilogue bounds a
# 16-bit 256x256 pair tile (``tcgen05_transposes_epilogue_bound_product``).
TCGEN05_EPILOGUE_BOUND_MAX_K_BLOCKS = 8
# The waves of 256x256 pair tiles from which a grid runs in the steady state
# that rule was measured in (from 6.8 waves up), each tile's epilogue beside
# the next tile's loads; ``tcgen05_copies_n_major_b`` asks as many waves of
# its K-major pair tiles, which repay its copy of b (1.7 waves do not).
TCGEN05_EPILOGUE_BOUND_MIN_WAVES = 6


def tcgen05_transposes_epilogue_bound_product(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    *,
    a_strides: tuple[int, int],
    b_strides: tuple[int, int],
    num_sms: int,
) -> bool:
    """Whether a wide 16-bit ``m x n x k`` product over a K-major ``b``,
    whose 256x256 pair tiles the epilogue bounds and whose odd count of
    256-wide N tiles leaves out the as-written 2x2 cluster, runs better as
    its transpose (out.T = b.T @ a.T) on the CuTe CTA pairs, for a
    row-major ``a`` (``a_strides`` / ``b_strides``: element strides).
    helion.inductor asks it before it lowers an ``aten.mm``
    (``HelionChoices.transpose_matmul``).

    A K loop of at most ``TCGEN05_EPILOGUE_BOUND_MAX_K_BLOCKS`` K blocks of
    64 is shorter than its tile's epilogue, whose TMA stores share the TMA
    unit and the L2 path with the next tile's loads: bf16 4096x250112x512
    with its epilogue skipped (no TMEM loads, no stores) runs 1.18x cuBLAS
    on GB300, with it 0.96x.
    As written the output is row-major, and the epilogue stores each CTA's
    128x256 accumulator in (128, 32) subtiles, 128 TMA rows of 64 bytes:

        tcgen05_c_layout = cutlass.utils.layout.LayoutEnum.ROW_MAJOR
        tcgen05_epi_tile = (cute.make_layout(128), cute.make_layout(32))

    Transposed, out.T is column-major: the product's N runs down the TMEM
    lanes, and the epilogue stores CUTLASS's (128, 64) subtiles, 64 TMA rows
    of 256 bytes, a quarter of the rows for the same bytes (cuBLAS's
    column-major convention stores so too):

        tcgen05_c_layout = cutlass.utils.layout.LayoutEnum.COL_MAJOR
        tcgen05_epi_tile = cutlass.utils.blackwell_helpers.compute_epilogue_tile_shape(
            (256, 256), False, tcgen05_c_layout, cutlass.BFloat16, ...)  # (128, 64)

    The as-written 2x2 cluster (``tcgen05_cluster_n=2``) relieves those
    tiles about as much, multicasting each A tile to two pairs, but it pairs
    the 256-wide N tiles, which an odd count leaves out; the transposed
    product pairs ``cdiv(m, 256)`` of them instead, and its full-effort
    autotune of 4096x250112x512 runs both (2x2 clusters, L2 grouping 4):
    1.023-1.031 of cuBLAS, against 0.965 as written. GB300, of cuBLAS in
    one process, the same 256x256 pair configs (64-wide K blocks, six
    stages), as written / as written in 2x2 clusters / transposed:
    4096x250112x512 (977 N tiles) 0.972 / - / 0.995, 4096x65632x512 (257)
    0.970 / - / 0.996, 2048x131200x512 (513) 0.967 / - / 0.991,
    8192x32896x512 (129) 0.967 / - / 0.981; even counts, where the 2x2
    cluster ties the transposed product: 4096x65536x512 0.972 / 1.010 /
    1.000, 8192x32768x512 0.964 / 1.004 / 0.983, 2048x131072x512 0.965 /
    0.972 / 0.993, 4096x65536x256 1.017 / 1.044 / 1.053. With odd counts,
    the best of three such configs either way round, transposed over as
    written: 6144x24832x384 1.026, 4096x16640x256 1.017, 4096x8448x512
    (6.9 waves) 1.027, 2048x16640x512 (6.8 waves) 1.015, 8192x8448x512
    (n ~ m) 1.010, but the tall 16384x8448x512 0.981. Deeper K loops hide
    the epilogue, the more so the larger m (the transposed product's N):
    K = 640 1.012 (4096x250112), K = 768 0.952 (8192x50432), 0.950
    (8192x50272, 8192x65536); K = 1024 1.002 at m = 2048, 0.984-0.986 at
    4096, 0.970 at 8192; K = 2048 0.990 (4096x65536).

    The measured products tile ``m`` whole. A ragged ``m`` over a ragged
    ``n`` that 64-row tiles divide stays as written: the transposed product
    is a double edge, which over whole K tiles the search runs on no CTA
    pair (``transposed_double_edge_without_pairs``: bf16 2040x20032x512;
    over a K tail both orientations run the edge + K-tail pairs, unmeasured
    here). So does a product of an operand off TMA's 16-byte alignment, a's
    rows or b's columns, over which neither orientation runs the pairs
    (``_operands_tma_aligned``: bf16 2048x131200x64 over a's rows sliced
    from [2048, 65]).
    """
    if dtype not in (torch.bfloat16, torch.float16):
        return False
    if a_strides[1] != 1 or b_strides[0] != 1:
        return False
    if not _operands_tma_aligned(a_strides, b_strides, dtype.itemsize):
        return False
    if -(-k // 64) > TCGEN05_EPILOGUE_BOUND_MAX_K_BLOCKS:
        return False
    if tcgen05_shape_rejected(m, n, k, dtype):
        # (no pair tiles either way round: bf16 4096x11008x8, a rank-8 LoRA
        # up-projection, runs the SIMT or warp-MMA lowering as written)
        return False
    if m < TCGEN05_TWO_CTA_BLOCK_M or n < m:
        # measured on wide products; tall ones stay with the pair-tile
        # orientation model (``transposed_pair_tile_orientation``)
        return False
    if transposed_double_edge_without_pairs(m, n):
        return False
    n_tiles = -(-n // TCGEN05_TWO_CTA_BLOCK_N)
    if n_tiles % 2 == 0:
        return False
    tiles = -(-m // TCGEN05_TWO_CTA_BLOCK_M) * n_tiles
    return tiles >= TCGEN05_EPILOGUE_BOUND_MIN_WAVES * (num_sms // 2)


# Bytes per L2 line: the granule whose two halves a misaligned output row
# splits between two TMA store rows (``tcgen05_transposes_misaligned_batched_output``).
TCGEN05_L2_LINE_BYTES = 128
# The longest K under which that rule measured its gain: one 64-wide K block.
# Deeper K loops hide the epilogue either way round (GB300, bf16 64 x
# 1000x1000, as written -> transposed: K = 64 0.935 -> 0.994, K = 256 1.068 ->
# 1.066, K = 512 0.997 -> 0.999; 64 x 1024x1024x512 0.986 -> 0.959).
TCGEN05_MISALIGNED_OUTPUT_MAX_K = 64


def tcgen05_transposes_misaligned_batched_output(
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    *,
    a_strides: tuple[int, int],
    b_strides: tuple[int, int],
    num_sms: int,
) -> bool:
    """Whether a batched 16-bit ``batch x m x n x k`` product over a
    K-major ``b``, whose single K block leaves its pair tiles bound by the
    epilogue and whose output rows straddle L2 lines, runs better as its
    transpose (``out[i].T = b[i].T @ a[i].T``) on the CuTe CTA pairs, for a
    row-major ``a`` (``a_strides`` / ``b_strides``: element strides of one
    matrix). helion.inductor asks it before it lowers an ``aten.bmm``
    (``HelionChoices.transpose_bmm``).

    Either way round the epilogue's TMA boxes write 128-byte row pieces.
    As written the output is row-major and each (128, 64) subtile is one
    box of 128 rows (SASS: one ``UTMASTG.3D`` per subtile):

        tcgen05_c_layout = cutlass.utils.layout.LayoutEnum.ROW_MAJOR

    so where a row of ``n * 2`` bytes is not a multiple of 128, every piece
    shares its first and last L2 lines with the pieces of the neighbouring
    subtiles, which later stores write. Transposed the output is
    column-major, the product's N runs down the TMEM lanes, and each (128,
    64) subtile is two boxes of the same 64 rows, stored back to back (SASS:
    two ``UTMASTG.3D`` per subtile), the second completing the lines the
    first splits:

        tcgen05_c_layout = cutlass.utils.layout.LayoutEnum.COL_MAJOR

    The bytes, the L2 write sectors and requests are the same either way
    round (bmm 64 x 1000x1000x64 back to back: 125 MB written, 4.51M
    sectors, 1.90M requests); DRAM is busy 51.5% of the as-written kernel's
    33.8 us and 58.7% of the transposed one's 29.8 us (cuBLAS: 61.9% of
    28.2 us).

    GB300, bf16, K = 64, outputs of ~128 MB, two configs of the 256x256 pair
    either way round (the best of each), of cuBLAS in one process (CUDA
    graphs over cold inputs): rows split across lines, as written ->
    transposed: 116 x 744x744 1.085 -> 1.183, 64 x 1000x1000 0.935 -> 0.994,
    63 x 1008x1008 1.230 -> 1.257, 42 x 1240x1240 0.947 -> 1.030, 16 x
    2000x2000 1.375 -> 1.413, but 60 x 1032x1032 (16 bytes off, an 8-column
    edge tile) 0.936 -> 0.924; whole lines stay as written: 109 x 768x768
    1.022 -> 0.972, 61 x 1024x1024 1.051 -> 0.982, 39 x 1280x1280 1.049 ->
    0.970 (odd counts of 256-wide N tiles too, which the unbatched
    ``tcgen05_transposes_epilogue_bound_product`` transposes).

    The pairs run such ragged matrices only over operands TMA loads
    (``batched_pair_matrix``'s ``ab_tma``), and the measured operands were
    contiguous, so an outer stride off TMA's 16-byte alignment keeps the
    product as written: a's rows sliced from [2, 256, 65], 130 bytes apart,
    leave both orientations off the pairs (transposed, a.T is the product's
    B), and an operand large enough for the CuTe backend to stage first
    (``stage_aligned_operands``) is unmeasured.
    """
    if batch < 2 or dtype not in (torch.bfloat16, torch.float16):
        return False
    if a_strides[1] != 1 or b_strides[0] != 1:
        return False
    if (
        a_strides[0] * dtype.itemsize % TCGEN05_TMA_ROW_ALIGN_BYTES
        or b_strides[1] * dtype.itemsize % TCGEN05_TMA_ROW_ALIGN_BYTES
    ):
        return False
    if k > TCGEN05_MISALIGNED_OUTPUT_MAX_K:
        return False
    row_bytes = n * dtype.itemsize
    if row_bytes % TCGEN05_TMA_ROW_ALIGN_BYTES or not row_bytes % TCGEN05_L2_LINE_BYTES:
        # (rows off TMA's 16-byte alignment take the SIMT store either way
        # round; whole-line rows keep the row-major subtiles)
        return False
    if tcgen05_shape_rejected(m, n, k, dtype):
        return False
    if m < TCGEN05_TWO_CTA_BLOCK_M or n < m:
        # measured on square products
        return False
    tiles = batch * -(-m // TCGEN05_TWO_CTA_BLOCK_M) * -(-n // TCGEN05_TWO_CTA_BLOCK_N)
    return tiles >= TCGEN05_EPILOGUE_BOUND_MIN_WAVES * (num_sms // 2)


# The measured region of ``tcgen05_transposes_wide_tf32_product``: wide tf32
# products over 32 or more rows of 256-row pair tiles, K up to 2048.
TCGEN05_WIDE_TF32_MIN_M_TILES = 32
TCGEN05_WIDE_TF32_MAX_K = 2048


def tcgen05_transposes_wide_tf32_product(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    *,
    a_strides: tuple[int, int],
    b_strides: tuple[int, int],
    num_sms: int,
) -> bool:
    """Whether a wide fp32 ``m x n x k`` product computed as tf32, over a
    row-major ``a`` and a K-major ``b`` (``a_strides`` / ``b_strides``:
    element strides), runs better as its transpose (out.T = b.T @ a.T) on
    the CuTe CTA pairs. helion.inductor asks it before it lowers an
    ``aten.mm`` of fp32 operands under
    ``torch.backends.cuda.matmul.fp32_precision = "tf32"``
    (``HelionChoices.transpose_matmul``).

    The pair's persistent raster walks its tiles in groups of
    ``l2_groupings`` M tiles, M fastest. As written, N is the output's
    contiguous dimension, so a wave of 76 pairs at grouping 32 stores runs
    of only 2.4 tiles (2.4 KB of fp32) of each row, and each CTA's epilogue
    stores its 128x256 accumulator in (128, 32) TMA boxes of 128-byte rows:

        for tile_y, tile_x in hl.tile([16384, 32000]):
            ...
            out0[tile_y, tile_x] = acc  # row-major out: 128 B per box row

    Transposed, out.T is column-major: the grouped M' axis is the output's
    contiguous dimension, so the same wave stores runs of 32 tiles (32 KB)
    of each row, and the boxes store 512-byte rows:

        for tile_x, tile_y in hl.tile([32000, 16384]):
            ...
            out0[tile_x, tile_y] = acc  # column-major out.T: 512 B per box row

    GB300, of stock Inductor (cuBLAS) in one process, CUDA graphs over cold
    inputs (blocks metric, mean of two runs), the 256x256x32 x 6 pair as
    written / transposed, CLC-persistent at grouping 32 and static at
    grouping 8:

        16384x32000x768   0.991 / 1.006   0.951 / 0.957
        16384x32768x768   0.995 / 1.013   0.957 / 0.966
        16384x16640x768   1.001 / 1.018   0.966 / 0.973
        16384x32000x1024  0.988 / 1.006   0.958 / 0.966
        16384x32000x2048  1.145 / 1.153   1.221 / 1.223
        8192x32000x768    1.007 / 1.015   0.963 / 0.972
        8192x50272x768    1.004 / 1.011   0.960 / 0.974
        8192x32768x1024   0.980 / 0.982   0.943 / 0.953
        4096x50272x768    0.987 / 0.986   0.954 / 0.980
        4096x128000x768   0.981 / 0.975   0.942 / 0.975

    From 32 rows of pair tiles the transposed product leads by 0.2-1.8% in
    either schedule, odd and even counts of N tiles alike; at 16 rows (m =
    4096) the as-written grouping covers all of M, its runs grow to 4.75
    tiles, and the best of each orientation ties (0.987 / 0.986) or runs as
    written (0.981 / 0.975). A tall product (n < m) stays as written: its
    transpose groups the short axis (16384x3072x768 transposed 0.89-0.97,
    as written 0.98-1.00). So does a product of an operand off TMA's
    16-byte alignment, a's rows or b's columns, over which neither
    orientation runs the pairs (``_operands_tma_aligned``: 8192x32768x16
    over a's rows sliced from [8192, 17], 68 bytes apart, lowers no
    persistent tcgen05 pair either way round).
    """
    if dtype != torch.float32:
        return False
    if a_strides[1] != 1 or b_strides[0] != 1:
        return False
    if not _operands_tma_aligned(a_strides, b_strides, dtype.itemsize):
        return False
    if tcgen05_shape_rejected(m, n, k, dtype, tf32=True):
        return False
    m_tiles = -(-m // TCGEN05_TWO_CTA_BLOCK_M)
    if n < m or m_tiles < TCGEN05_WIDE_TF32_MIN_M_TILES or k > TCGEN05_WIDE_TF32_MAX_K:
        return False
    tiles = m_tiles * -(-n // TCGEN05_TWO_CTA_BLOCK_N)
    return tiles >= TCGEN05_EPILOGUE_BOUND_MIN_WAVES * (num_sms // 2)


def _pair_tile_columns(n: int, widths: Iterable[int]) -> tuple[int, int]:
    """``(columns, width)``: the fewest columns ``ceil(n / width) * width``
    that a 16-bit pair of one of ``widths`` (none wider than ``n``) computes
    over ``n``, the wider width on a tie."""
    columns, negative_width = min(
        (-(-n // width) * width, -width) for width in widths if width <= n
    )
    return columns, -negative_width


def tcgen05_one_row_tile_batched_narrow_n(
    *,
    batched_operand: bool,
    a_m_major: bool,
    itemsize: int,
    static_m: int | None,
    static_n: int | None,
    static_k: int | None,
    batch: int,
    num_sms: int,
) -> bool:
    """Whether the tcgen05 search tiles a batched 16-bit GEMM's N 64 wide
    (``TCGEN05_ONE_ROW_TILE_BATCHED_BLOCK_N``): each matrix one 64-row tile
    tall over an M-major A, N 128 to 512 and K 256 to 512 in whole 64-wide
    steps, the grid two 64x64 tiles a SM or more (the shapes measured; a
    row-major A keeps its 64x256 default, 1.03 of the ATen bmm on B200 at 384
    x 64x512x512). Memory-bound: every N tile of a matrix reads its whole 64 x
    K block of A, and B once.

    The final selection's per-call timing (an L2 flush before each launch)
    reads 64x128 tiles 0.2-0.5% ahead of the 64x64 tiles on a deep ring
    (``seeds._batched_one_row_tile_seed_configs``), but back to back in a
    CUDA graph they run 1-1.5% behind it, and the selection keeps them. bf16
    256 x 64x512x512 with A stored M-major (GB300, of cuBLAS, CUDA graphs over
    cold inputs): full-effort pick [1, 64, 128, 64] ab7 0.988 (its final
    selection read it 31.79 us, the other finalists 31.84-32.03), the 64x64
    ring by hand 0.999-1.000; 384 x 64x512x512: [1, 64, 128, 64] ab7 0.978
    (45.17 against 45.31-45.41 us), the ring 0.993. No measured shape runs a
    wider tile faster in the graph: 192 x 64x512x512 64x64 1.016-1.017
    against 64x256 0.978, 512 x 64x512x512 1.006-1.014 against 0.985, 1024 x
    64x256x256 0.969-0.972 against 64x128 0.966-0.973 and 64x256 0.960.
    """
    return (
        batched_operand
        and a_m_major
        and itemsize == 2
        and static_m is not None
        and static_n is not None
        and static_k is not None
        and 0 < static_m <= TCGEN05_ONE_CTA_MIN_BLOCK_M
        and static_n % TCGEN05_ONE_ROW_TILE_BATCHED_BLOCK_N == 0
        and 2 * TCGEN05_ONE_ROW_TILE_BATCHED_BLOCK_N <= static_n <= 512
        and static_k % TCGEN05_WHOLE_K_TILE_MIN == 0
        and 4 * TCGEN05_WHOLE_K_TILE_MIN <= static_k <= 8 * TCGEN05_WHOLE_K_TILE_MIN
        and num_sms > 0
        and batch * (static_n // TCGEN05_ONE_ROW_TILE_BATCHED_BLOCK_N) >= 2 * num_sms
    )


def tcgen05_copies_n_major_b(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    *,
    a_strides: tuple[int, int],
    b_strides: tuple[int, int],
    num_sms: int,
) -> bool:
    """Whether a 16-bit ``m x n x k`` product of a row-major ``a`` and an
    N-major ``b`` runs better as written over a K-major copy of ``b`` on the
    CuTe CTA pairs (``a_strides`` / ``b_strides``: element strides).
    helion.inductor asks it before it lowers an ``aten.mm`` that it keeps
    as written (``HelionChoices.copy_matmul_b``).

    Each CTA of the pair stages half of b's N tile as SMEM rows; over an
    N-major b the pair runs the widths whose half rows hold whole 64-byte
    swizzle atoms (``dense_aligned_n_mn_b_width``: 128, 192 and 256 columns
    from 128 up), over a K-major b every 16-column multiple. Where the
    N-major widths compute a third more columns than N needs and the K-major
    ones tile it (nearly) whole, the copy wins: bf16 131072x288x144 with b
    stored [K, N] runs 256x192 pairs over 288 columns (384 computed: the
    second tile 96 of 192 columns), over a K-major copy of b 256x144 pairs
    (288 computed):

        for tile_y, tile_x in hl.tile([131072, 288]):
            ...
            tmp1 = in1[tile_r, tile_x]  # in1: b itself ([144, 288], N-major)
        # with the copy, in1 is a [144, 288] buffer of strides (1, 144),
        # written by a copy kernel (16x16 tiles) ahead of the product

    The copy is a small fixed cost (1.5-1.8 us on GB300 for b up to 332 KB),
    which a product of at least ``TCGEN05_EPILOGUE_BOUND_MIN_WAVES`` waves of
    pair tiles at the K-major width repays; its columns are k elements
    apart, so K must keep them on TMA's 16-byte alignment
    (``TCGEN05_TMA_ROW_ALIGN_BYTES``). GB300, cold full-effort
    autotunes, cuBLAS time / Helion time (CUDA graphs over cold inputs, each
    mode in its own blocks), as written / over the copy: 131072x288x144 0.947
    / 1.055, 131072x288x576 0.929 / 1.064, 131072x288x1152 0.900 / 1.040,
    131072x160x144 0.576 / 1.047 (as written no pair width serves 160
    columns: 192 and 256 run past N); but 131072x416x144 (256-wide tiles
    either way: 512 of 416 columns) 1.170 / 1.107, 16384x288x144 (1.7 waves
    at 144 columns) 1.016 / 1.011, and 8192x288x144 (0.8 waves) 0.937 /
    0.803."""
    if dtype not in (torch.bfloat16, torch.float16):
        return False
    if a_strides[1] != 1 or b_strides[1] != 1 or b_strides[0] == 1:
        return False
    if n < TCGEN05_TWO_CTA_BLOCK_N // 2:
        return False
    if k * dtype.itemsize % TCGEN05_TMA_ROW_ALIGN_BYTES:
        # (the copy's columns, k elements apart, would be off TMA's 16-byte
        # alignment where b's rows are on it: bf16 K = 100 leaves a 200-byte
        # stride, whose operands fall back to the scalar SMEM producers)
        return False
    if tcgen05_shape_rejected(m, n, k, dtype):
        return False
    widths = range(TCGEN05_TWO_CTA_BLOCK_N // 2, TCGEN05_TWO_CTA_BLOCK_N + 1, 16)
    n_major_columns, _ = _pair_tile_columns(
        n, (w for w in widths if dense_aligned_n_mn_b_width(w, dtype.itemsize))
    )
    k_major_columns, k_major_width = _pair_tile_columns(n, widths)
    tiles = -(-m // TCGEN05_TWO_CTA_BLOCK_M) * -(-n // k_major_width)
    return (
        3 * n_major_columns >= 4 * k_major_columns
        and tiles >= TCGEN05_EPILOGUE_BOUND_MIN_WAVES * (num_sms // 2)
    )


def tcgen05_transposes_skinny_batched_product(
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    *,
    a_strides: tuple[int, int, int],
    b_strides: tuple[int, int, int],
    num_sms: int,
) -> bool:
    """Whether a 16-bit batched ``batch x (m x n x k)`` product whose matrices
    are one 64-row tile tall runs better as its transpose (``out[i].T =
    b[i].T @ a[i].T`` over ``[batch, n, m]``) on the CuTe GEMM families, for
    an M-major ``a`` and an N-major ``b`` (``a_strides`` / ``b_strides``: the
    operands' element strides; the transposed product has the same majors).
    helion.inductor asks it before it lowers an ``aten.bmm``
    (``HelionChoices.transpose_bmm``).

    As written each matrix is one 64-row MMA tile, every N tile of which
    reads the matrix's whole 64 x K block of ``a``; an M under 64 rows fills
    half of it and runs the flat launch only. bf16 1024 x 32x128x128 with
    ``a`` stored [1024, 128, 32] (attention's ``attn.transpose(1, 2)``):

        for tile_z, tile_y, tile_x in hl.tile([1024, 32, 128]):
            ...  # full-effort pick [1, 64, 64, 64] flat: 15.01 us

    Transposed, N runs on the MMA's M side in 128-row tiles and M on its N
    side, as cuBLAS runs these products (the transposed product on 128x64
    CTA pairs: ``nvjet_sm103_tst_128x64_64x10_2x1_2cta_v_bz_NTT``); the
    transposed product is the as-written ``batch x n x m x k`` one with a
    column-major output:

        for tile_z, tile_x, tile_y in hl.tile([1024, 128, 32]):
            ...  # [1, 128, 32, 128] persistent: 9.35 us
            # out0 [1024, 128, 32] N-contiguous: its transpose is the result

    GB300, cold full-effort autotunes through helion.inductor, CUDA graphs
    over cold inputs: 1024 x 32x128x128 0.688 -> 1.104 of stock Inductor (a
    Triton template), 1024 x 64x128x128 0.980 -> 0.996 of cuBLAS (12.05 ->
    11.86 us; no as-written config by hand above 0.987), 3072 x 64x128x128
    31.84 -> 31.82 us. A short grid runs the as-written tiles faster, so the
    batch holds six waves of the transposed product's tiles (one a matrix)
    at least: 128-row CTA pairs at M = 64 over a K loop of two or more
    64-deep k-blocks, the transposed one-tile matrices whose search runs only
    the pairs (``repair.batched_pair128_one_tile_matrices``), a wave of
    which holds half as many matrices (456 on GB300's 152 SMs); one-CTA
    128-row tiles otherwise (912): an M under 64, whose N' = M no CTA pair
    takes, or one K tile, whose transposed search keeps one-CTA tiles. Cold
    full-effort autotunes of 64x128x128, transposed against as written:
    512 matrices (6.7 waves of pairs) 6.94-6.98 against 7.14-7.15 us on the
    128x64 pair (6.93 for cuBLAS), 768 (10.1) 9.58 against 10.47. Those
    and the shorter grids, 384 (5.1) 6.14 against 5.88, 256 (3.4) 4.53
    both, 192 (2.5) 4.02 against 4.00, were timed before the transposed
    search ran only the pairs (its one-CTA picks); 192 by hand, the pair
    3.96-3.97 against 4.00 us. N is one 128-row tile: over two (1024 x 64x256x256)
    the tiles of a matrix share its a.mT only on a batch-slowest walk (the
    pairs by hand 0.976-0.987, batch-fastest 0.87, where the pair seeds
    start), against 0.977 as written. A longer K loop runs the as-written
    64x64 ring as fast (192 x 64x512x512: 1.016 / 1.024 by hand), so K holds
    at most four 64-deep k-blocks. a.mT is the transposed product's B
    operand, whose rows (a's K stride) and matrices (its batch stride) TMA
    reads on 16-byte boundaries: in attention's layout an M off a multiple of
    8 would stage B (1024 x 50x128x128: 100-byte rows) or, under 8 columns
    (M = 4), leave the tcgen05 search, so those stay as written.
    """
    if dtype not in (torch.float16, torch.bfloat16) or num_sms <= 0:
        return False
    if a_strides[1] != 1 or b_strides[2] != 1:
        return False
    if any(
        (stride * dtype.itemsize) % TCGEN05_TMA_ROW_ALIGN_BYTES
        for stride in (a_strides[0], a_strides[2])
    ):
        return False
    tile_rows = TCGEN05_TWO_CTA_BLOCK_M // 2
    # The transposed product's tile a matrix is a 128-row CTA pair where its
    # N (this M) reaches the narrowest pair width and its K loop spans two
    # or more 64-deep k-blocks (its search then runs only the pairs:
    # ``repair.batched_pair128_one_tile_matrices``), else one CTA: a wave of
    # pairs holds half as many matrices.
    pairs = (
        m >= min(TCGEN05_TWO_CTA_PAIR128_ONE_WAVE_BLOCK_NS)
        and k > TCGEN05_WHOLE_K_TILE_MIN
    )
    tiles_a_wave = num_sms // 2 if pairs else num_sms
    return (
        0 < m <= TCGEN05_ONE_CTA_MIN_BLOCK_M
        and n == tile_rows
        and k % TCGEN05_WHOLE_K_TILE_MIN == 0
        and k <= 4 * TCGEN05_WHOLE_K_TILE_MIN
        and batch >= TCGEN05_TRANSPOSED_BATCH_MIN_WAVES * tiles_a_wave
    )


def batched_pair_matrix(
    *,
    itemsize: int,
    rank3_operand: bool,
    static_m: int | None,
    static_n: int | None,
    static_k: int | None,
    ab_tma: bool,
) -> bool:
    """Whether a batched 16-bit GEMM runs its matrices on the persistent CTA
    pairs: each matrix taller and wider than a 128 x 128 tile, and either
    within one 256 x 256 CtaGroup.TWO tile or ragged against it in M or N
    (however many pair tiles it spans), which the persistent pair schedules
    run over the whole matrix. TMA zero-fills the A rows and B columns past
    the matrix, and the pair's 3-D TMA store clips the tiles at each matrix's
    edges (the batch is its own tensor-map dim). bf16 768 x 200x200x64
    (ViT-style attention scores): the 256x256 pair 19.9 us against the flat
    one-CTA 128x128 tiles' 27.2 (2 x 2 tiles per matrix, every one a fresh
    launch slot) and cuBLAS's 19.9; bf16 64 x 1000x1000x64, 4 x 4 pair tiles
    a matrix: the full-effort pick ran 27.7 us on the pairs against the flat
    tiles' 42.8 (cuBLAS 26.2).

    ``ab_tma``: both operands load through TMA (``cute_mma``'s
    ``batched_ab_tma_provable``), which the pairs need over ragged matrices
    (``_ragged_batched_pair_config``). bf16 40 x [200, 32] @ [32, 196] has
    392-byte B rows no TensorMap addresses and too few elements to stage:
    its pair seed would be the shared scalar-copy kernel, which refuses the
    double edge (``BackendUnsupported: tcgen05 SIMT edge epilogue
    double-edge output tiles ...``), so it keeps the flat one-CTA tiles."""
    return (
        itemsize == 2
        and ab_tma
        and rank3_operand
        and static_m is not None
        and static_n is not None
        and static_k is not None
        and static_m > 128
        and static_n > 128
        and (
            (
                static_m <= TCGEN05_TWO_CTA_BLOCK_M
                and static_n <= TCGEN05_TWO_CTA_BLOCK_N
            )
            or static_m % TCGEN05_TWO_CTA_BLOCK_M != 0
            or static_n % TCGEN05_TWO_CTA_BLOCK_N != 0
        )
    )


def tcgen05_batched_pair128_m_edge(static_m: int) -> bool:
    """Whether a batched GEMM's matrices are one partial 128-row CTA pair
    tile tall with rows for both CTAs: more than the 64 rows of the pair's
    first CTA, fewer than the pair tile's 128 (``batched_pair128_matrix``).

    The power-of-two floor of such an M (64 for 80) splits each matrix into a
    whole and a partial 64-row one-CTA tile, the second reading the matrix's
    B again. bf16 512 x [80, 128] @ [128, 128] (a M-major), the full-effort
    pick before:

        for tile_z, tile_y, tile_x in hl.tile([512, 80, 128]):
            ...  # [1, 64, 128, 32] flat: 1024 CTAs, 10.08 us

    and on one 128-row pair per matrix, its second CTA's A box zero-filled
    past row 80 and its 3-D TMA store clipped there (cuBLAS runs one 2-CTA
    pair of 128x96 a matrix, transposed):

        ...  # [1, 128, 128, 64] persistent pair: 7.53-7.59 us

    GB300, CUDA graphs over cold inputs, of cuBLAS's 7.45-7.49 us: 0.740 ->
    0.987-0.990; M = 96 / 112 run 0.980-0.982 / 0.988 as written. The
    transposed product (out.mT = b.mT @ a.mT: the same pair over N' = 80 on a
    128-wide carrier) runs 3.0-3.7% behind at every M (0.951-0.955): its
    column-major output's epilogue costs that much on a tile of two k-blocks,
    so these matrices stay as written. At 64 rows or fewer the pair's second
    CTA would hold none (``tcgen05_transposes_skinny_batched_product``
    transposes those)."""
    return TCGEN05_ONE_CTA_MIN_BLOCK_M < static_m < TCGEN05_TWO_CTA_EDGE_PAIR128_BLOCK


def batched_pair128_matrix(
    *,
    itemsize: int,
    rank3_operand: bool,
    static_m: int | None,
    static_n: int | None,
    static_k: int | None,
    ab_tma: bool,
) -> bool:
    """Whether a batched 16-bit GEMM runs its matrices on the persistent
    128-row CTA pairs (``tcgen05_cta_group="two"`` on ``block_m=128``: each
    CTA owns 64 rows of the pair tile) over an M edge, an N edge or a K
    tail: matrices whole in 128-row tiles whose N no 64-wide pair tile
    divides (wider than one, and ``batched_pair_matrix``'s 256-row pairs not
    taking them), or whose K ends in a tail past the first 64-deep k-block,
    or matrices shorter than one pair tile whose rows reach the pair's
    second CTA (``tcgen05_batched_pair128_m_edge``). The role-local TMA
    producer zero-fills each matrix's A rows past M, its B rows past N and
    its K tail (the batch is its own tensor-map dim), and the pair's 3-D TMA
    store clips the tile at the matrix's M and N edges. bf16 512 x [128,
    128] @ [128, 80] (b K-major) runs one 128x128 pair per matrix, as cuBLAS
    runs one 2-CTA pair of 128x80; before, only the flat one-CTA tiles ran
    it (the surface's paired pipeline takes whole 64-wide N and K tiles
    only).

    ``ab_tma`` as for ``batched_pair_matrix``: without TMA the pair would be
    the shared scalar-copy kernel (``_ragged_batched_pair_config``)."""
    if not (
        itemsize == 2
        and ab_tma
        and rank3_operand
        and static_m is not None
        and static_n is not None
        and static_k is not None
        and (
            static_m % TCGEN05_TWO_CTA_EDGE_PAIR128_BLOCK == 0
            or tcgen05_batched_pair128_m_edge(static_m)
        )
    ):
        return False
    pair_k = TCGEN05_TWO_CTA_EDGE_K_TAIL_DEEP_BLOCK_K
    pair_n = min(TCGEN05_TWO_CTA_PAIR128_ONE_WAVE_BLOCK_NS)
    n_edge = static_n % pair_n != 0 and static_n > pair_n
    k_tail = static_k % pair_k != 0 and static_k > pair_k
    return (
        (tcgen05_batched_pair128_m_edge(static_m) or n_edge or k_tail)
        # the narrowest pair tile is 64 wide: under it (N = 32) no pair tile
        # fits the search's N range, and the admission's persistent schedules
        # would only reach the one-CTA tiles
        and static_n >= pair_n
        # the pair's k-blocks are 64 deep and its K tail rides only past the
        # first (``cute_mma``'s ``_ragged_batched_pair_config``): K = 48 runs
        # on no pair
        and (static_k % pair_k == 0 or static_k > pair_k)
        and not batched_pair_matrix(
            itemsize=itemsize,
            rank3_operand=rank3_operand,
            static_m=static_m,
            static_n=static_n,
            static_k=static_k,
            ab_tma=ab_tma,
        )
    )


def _one_cta_heights_miss_m(static_m: int) -> bool:
    """Whether no one-CTA tile height (64 or 128 rows) divides M: no
    persistent one-CTA schedule runs it, only the CtaGroup.TWO pairs, whose
    TMA store clips the partial last M tile."""
    return static_m % (TCGEN05_ONE_CTA_MAX_BLOCK_M // 2) != 0


def _double_edge_pair_shape(
    *,
    itemsize: int,
    has_leading_passthrough: bool,
    static_m: int,
    static_k: int,
    max_search_k: int,
) -> bool:
    """Whether a GEMM ragged in both M and N runs on the 256-row pairs.

    16-bit operands, whole ``max_search_k`` tiles of K (a K tail on both
    edges is the edge + K-tail family's shape) and an M that no one-CTA tile
    height (64, 128 rows) divides: no persistent one-CTA schedule runs it, and
    the pair's TMA store clips both edges. ``_plan_cute_tcgen05_search_candidate``
    keeps its 256-row M ceiling and ``enable_cute_tcgen05_search`` admits it as
    an output edge on this one test.
    """
    return (
        itemsize == 2
        and not has_leading_passthrough
        and static_k % max_search_k == 0
        and _one_cta_heights_miss_m(static_m)
    )


@dataclasses.dataclass(frozen=True)
class CuteTcgen05SearchPlan:
    """Immutable tcgen05 search limits for one MMA candidate.

    Planning does not mutate shared compile state. ``static_*`` are proven
    compile-time extents; dynamic cache requirements live in the surrounding
    planning result so rejected candidates retain them too.
    """

    m: int | torch.SymInt
    n: int | torch.SymInt
    k: int | torch.SymInt
    static_m: int | None
    static_n: int | None
    static_k: int | None
    leading_work_multiplier: int
    is_fp8: bool
    is_small_n: bool
    # Small N is necessarily a partial tcgen05 N tile, so it needs explicit
    # edge-path capabilities. Regular N uses the existing full-tile
    # divisibility and validated edge-family gates instead.
    small_n_supports_role_local_tma: bool
    small_n_requires_role_local_tma: bool
    mma_k: int
    max_tcgen05_m: int
    max_tcgen05_n: int
    max_search_m: int
    max_search_n: int
    max_search_k: int
    min_search_m: int
    min_search_n: int
    # The register-MMA GEMM family (``cute_matmul_family="warp_mma"``) is
    # searched only on static, latency-bound problems with the operand
    # layouts its ldmatrix paths move (see ``warp_mma_admission_reason``).
    warp_mma_admitted: bool = False
    # False when an operand is a cast the tcgen05 pipeline cannot feed to the
    # tensor core (a bare ``load.to(bf16)`` of an int16 tile): only the
    # register-MMA family runs the kernel and the search offers that family
    # alone.
    tcgen05_family_admitted: bool = True
    # True when the search takes a small N's zero-filled tcgen05 tile
    # (``tcgen05_small_n_zero_fills``) over ``min_search_n``, the scalar
    # fallback's block, which stays the floor a provided config keeps.
    small_n_zero_fills: bool = False
    # ``batched_pair_matrix``: the search reaches the 256 x 256 pair tile
    # over each ragged matrix of a batch.
    batched_pair_matrix: bool = False
    # ``tcgen05_one_row_tile_batched_narrow_n``: the search tiles N 64 wide,
    # and so does the default.
    one_row_tile_narrow_n: bool = False
    # ``batched_pair128_matrix``: the search reaches the 128-row pairs over
    # each matrix's M edge, N edge or K tail.
    batched_pair128_matrix: bool = False


@dataclasses.dataclass(frozen=True)
class _CuteTcgen05SearchPlanningResult:
    """One side-effect-free candidate decision and its cache requirements.

    ``guards_complete`` is false when a dynamic extent cannot be traced back to
    runtime inputs. Such a candidate makes the whole tcgen05 search unsafe to
    cache, even when this particular binding rejects the candidate.
    """

    plan: CuteTcgen05SearchPlan | None
    required_specialized_vars: frozenset[sympy.Symbol]
    guards_complete: bool


def _plan_cute_tcgen05_search_candidate(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    has_leading_passthrough: bool,
    supports_small_n_role_local_tma: bool = False,
    supports_small_n_scalar_fallback: bool = False,
    allow_dynamic_hints: bool = False,
    lhs_source: torch.Tensor | None = None,
    rhs_source: torch.Tensor | None = None,
    operands_permuted: bool = False,
    operand_casts: bool = False,
    batched_ab_tma: bool = False,
) -> _CuteTcgen05SearchPlanningResult:
    """Plan one candidate and retain guards even when the candidate is rejected.

    ``lhs`` / ``rhs`` are the matmul's operand values (the loaded tiles);
    ``lhs_source`` / ``rhs_source`` are the host tensors they were loaded
    from, whose layouts decide the register-MMA family's admission.
    ``operand_casts`` says an operand is an in-register integer cast only
    the register-MMA family lowers: the candidate is rejected unless that
    family is admitted, and the plan withholds the tcgen05 family.
    ``batched_ab_tma`` says the tcgen05 pipeline loads both operands of a
    batched candidate through TMA (``batched_pair_matrix``); without the
    operand analysis (``plan_cute_tcgen05_search``) it is not proved.
    """
    m, n, k = _dot_dimensions(lhs, rhs)
    env = CompileEnvironment.current()

    problem_extents = tuple(_problem_extent(env, size) for size in (m, n, k))
    static_m, static_n, static_k = (
        None if problem_extent is None else _static_dim_value(env, problem_extent)
        for problem_extent in problem_extents
    )
    required_specialized_vars: set[sympy.Symbol] = set()
    guards_complete = True
    if allow_dynamic_hints:
        for problem_extent, static_size in zip(
            problem_extents,
            (static_m, static_n, static_k),
            strict=True,
        ):
            if static_size is not None:
                continue
            if not isinstance(problem_extent, torch.SymInt):
                guards_complete = False
                continue
            dynamic_symbols = _symint_free_symbols(problem_extent)
            if not dynamic_symbols:
                guards_complete = False
                continue
            for symbol in dynamic_symbols:
                sources = env.shape_env.var_to_sources.get(symbol)
                # The runtime builds guards from the first source. A traced
                # global (e.g. an AOT module's tile extent) has a source too,
                # but cannot be extracted from the kernel's arguments.
                if sources and (
                    _is_supported_tensor_input_source(sources[0])
                    or (
                        isinstance(sources[0], TensorPropertySource)
                        and _is_supported_tensor_input_source(sources[0].base)
                    )
                ):
                    required_specialized_vars.add(symbol)
                else:
                    guards_complete = False
    frozen_required_specialized_vars = frozenset(required_specialized_vars)

    def reject() -> _CuteTcgen05SearchPlanningResult:
        return _CuteTcgen05SearchPlanningResult(
            plan=None,
            required_specialized_vars=frozen_required_specialized_vars,
            guards_complete=guards_complete,
        )

    if not guards_complete:
        return reject()
    if not allow_dynamic_hints and (
        static_m is None or static_n is None or static_k is None
    ):
        return reject()
    m_hint = static_m if static_m is not None else env.size_hint(m)
    n_hint = static_n if static_n is not None else env.size_hint(n)
    k_hint = static_k if static_k is not None else env.size_hint(k)
    from ..mma_support import cute_fp32_dot_uses_tf32

    is_fp8 = lhs.dtype == torch.float8_e4m3fn
    is_tf32 = lhs.dtype == torch.float32 and cute_fp32_dot_uses_tf32()
    mma_k = tcgen05_mma_k(lhs.dtype, tf32=is_tf32)
    min_tcgen05_n = tcgen05_min_n(lhs.dtype, tf32=is_tf32)
    is_small_n = static_n is not None and static_n < min_tcgen05_n
    if (
        env.backend_name != "cute"
        or (
            lhs.dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
            and not is_tf32
        )
        or rhs.dtype != lhs.dtype
        or tcgen05_shape_rejected(
            m_hint,
            n_hint,
            k_hint,
            lhs.dtype,
            tf32=is_tf32,
            static_m=static_m is not None,
            static_n=static_n is not None,
            # the batch is a rank-3 operand's leading dimension (a leading
            # passthrough over two rank-2 operands has none)
            batch=(
                _static_problem_extent(env, (lhs if lhs.ndim == 3 else rhs).shape[0])
                if has_leading_passthrough and (lhs.ndim == 3 or rhs.ndim == 3)
                else None
            ),
            small_n_supported=(
                supports_small_n_role_local_tma or supports_small_n_scalar_fallback
            ),
        )
    ):
        return reject()
    small_n_requires_role_local_tma = (
        is_small_n and not supports_small_n_scalar_fallback
    )

    from ..mma_support import get_cute_mma_support

    support = get_cute_mma_support()
    from ..mma_support import tcgen05_supports_input_dtype

    if not tcgen05_supports_input_dtype(support, lhs.dtype):
        return reject()

    def pow2_floor_at_least(value: int, minimum: int) -> int:
        return 1 << (max(minimum, value).bit_length() - 1)

    max_tcgen05_n = (
        256
        if static_n is None
        else min(256, pow2_floor_at_least(static_n, min_tcgen05_n))
    )
    if (
        static_n is not None
        and lhs.dtype in (torch.bfloat16, torch.float16)
        and not has_leading_passthrough
        and 128 < static_n < 256
        and static_n % 16 == 0
    ):
        # A 16-bit N between 128 and 256 is itself an aligned pair width
        # (``dense_aligned_n.ALIGNED_N_CHOICES``), whose public carrier is
        # the 256-wide N tile: bf16 25216x192x768 then runs one 128x192 pair
        # tile per 128 rows (block_sizes [128, 256, 64], tcgen05_tile_n=192)
        # instead of two N tiles of 128 + 64 that read A twice, 10.14 us
        # against 11.31 (GB300, cuBLAS 10.47).
        max_tcgen05_n = 256
    max_tcgen05_m = (
        256 if max_tcgen05_n >= 128 and (static_m is None or static_m >= 256) else 128
    )
    max_search_m = (
        max_tcgen05_m
        if static_m is None
        else min(max_tcgen05_m, pow2_floor_at_least(static_m, 64))
    )
    # block_m=512 (tcgen05 M-paired tiles): two 256-row CtaGroup.TWO UMMA
    # subtiles share each K stage's B buffer, halving B's SMEM/L2/DRAM
    # traffic (fp16 16384^3: 0.92 -> 1.01 vs cuBLAS). Only searchable on the
    # 16-bit full-tile plain family the codegen validates; the tcgen05
    # config projections demote ineligible 512 samples back to 256.
    if (
        max_tcgen05_m == 256
        and not is_fp8
        and not is_tf32
        and static_m is not None
        and static_m % 512 == 0
        and static_n is not None
        and max_tcgen05_n >= 128
    ):
        max_search_m = 512
    max_search_n = max_tcgen05_n
    pair_matrix = (
        has_leading_passthrough
        and env.settings.cute_flat_edge_tma
        and batched_pair_matrix(
            itemsize=lhs.dtype.itemsize,
            rank3_operand=lhs.ndim == 3 or rhs.ndim == 3,
            static_m=static_m,
            static_n=static_n,
            static_k=static_k,
            ab_tma=batched_ab_tma,
        )
    )
    if pair_matrix:
        # The power-of-two floors (128 for 200) would split a matrix within
        # one pair tile into 2 x 2 tiles; the pair's tile covers it whole (a
        # larger ragged matrix keeps the 256 x 256 ceilings it has).
        max_tcgen05_m = max_search_m = TCGEN05_TWO_CTA_BLOCK_M
        max_tcgen05_n = max_search_n = TCGEN05_TWO_CTA_BLOCK_N
    pair128_matrix = (
        has_leading_passthrough
        and env.settings.cute_flat_edge_tma
        and batched_pair128_matrix(
            itemsize=lhs.dtype.itemsize,
            rank3_operand=lhs.ndim == 3 or rhs.ndim == 3,
            static_m=static_m,
            static_n=static_n,
            static_k=static_k,
            ab_tma=batched_ab_tma,
        )
    )
    max_search_k = (
        128 if static_k is None else min(128, pow2_floor_at_least(static_k, mma_k))
    )
    min_search_m = 128 if max_tcgen05_m >= 256 else 64
    if (
        static_n is not None
        and is_small_n
        and supports_small_n_scalar_fallback
        and not is_fp8
    ):
        min_search_n = 1 << (static_n - 1).bit_length()
    else:
        min_search_n = min_tcgen05_n
    # A 16-bit N of 2 to 4 columns, whose scalar-fallback block above is
    # narrower than the narrowest tcgen05 N tile (5 to 7 round up to 8
    # columns anyway), runs one zero-filled 8-column tile on the flat
    # pipeline, whose edge TMA fills the B rows past N with zeros and whose
    # store skips the output columns past it, as a short M runs one
    # zero-filled 64-row tile (``tcgen05_shape_rejected``) -- where the GEMM
    # is memory-bound, M and K each at least 128. helion.inductor's
    # transposed GEMV (bf16 4x4096x4096 with a K-major b, out.T = b.T @ a.T,
    # N = 4): the 8-column tile
    #     for tile_x, tile_y in hl.tile([4096, 4]):  # [64, 8, 128]: 64 CTAs
    #         ...
    #             tmp1 = in1[tile_r, tile_y]  # a.T: an 8 x 128 box, rows 4-7 zero
    # runs 8.48 us (GB300, 1.06 of cuBLAS's split-K kernel); the scalar
    # fallback's 4-column block ([64, 4, 16]) runs SIMT, 206 us.
    # The search and the default take that tile (``enable_cute_tcgen05_search``
    # raises the N search floor to it and the K one to
    # ``TCGEN05_SMALL_N_MIN_K_BYTES``) where the kernel's K block reaches
    # that depth; one that cannot (``hl.register_block_size(16, 16)``) keeps
    # the scalar fallback's search, its only working tiles. So does a kernel
    # whose N block cannot be 8 wide (``_n_block_admits``): a registered
    # range that leaves 8 out (``hl.register_block_size(4, 4)``, a bound on
    # the search whose value may size the kernel's allocations) or a fixed
    # ``block_size=`` other than 8. The block floor stays the scalar
    # fallback's, so a provided config keeps its tiles: a block_n of 2 or 4
    # runs the SIMT dot it ran before, where raising it to 8 would turn
    # [64, 4, 16] into the 32-byte K-major B stage CuTe refuses.
    # Only a B that TMA loads takes the tile (the TensorMap rule on its
    # static layout, ``stage_aligned_operands._tma_aligned``): an N-major B
    # of 8-byte rows (x @ w, w (K, 4) row-major) would run it on the scalar
    # shared-memory producer, which was not measured, and keeps the scalar
    # fallback. (The fake operands sit at storage offset 0: a base the bound
    # input misaligns is not visible here and runs the tile on that
    # producer.)
    from ..stage_aligned_operands import _tma_aligned

    small_n_zero_fills = (
        min_search_n < min_tcgen05_n
        and static_m is not None
        and static_n is not None
        and static_k is not None
        and tcgen05_small_n_zero_fills(static_m, static_n, static_k, lhs.dtype)
        and not has_leading_passthrough
        and env.settings.cute_flat_edge_tma
        and rhs_source is not None
        and _tma_aligned(rhs_source)
        and _k_block_reaches(env, k, TCGEN05_SMALL_N_MIN_K_BYTES // lhs.dtype.itemsize)
        and _n_block_admits(env, n, min_tcgen05_n)
    )
    leading_work_multiplier = 1
    if has_leading_passthrough:
        leading_operand = lhs if lhs.ndim == 3 else rhs
        static_leading = _static_problem_extent(env, leading_operand.shape[0])
        if static_leading is not None:
            leading_work_multiplier = max(static_leading, 1)
        if static_m is not None and static_n is not None and static_k is not None:
            if static_m % min_search_m != 0:
                min_search_m = 64
            if (
                (lhs.ndim < 3 and rhs.ndim < 3)
                or max_search_n < RAGGED_BATCHED_MIN_BLOCK_N
                or not env.settings.cute_flat_edge_tma
            ):
                # The passthrough axis is the kernel's, not an operand's (a
                # grouped GEMM whose group tile selects rows through offsets;
                # its families register whole native tiles), or N admits no
                # tile the ragged batched pipeline lowers: the search keeps
                # the power-of-two divisors of the extents.
                max_search_m = min(max_search_m, static_m & -static_m)
                max_search_n = min(max_search_n, static_n & -static_n)
                max_search_k = min(max_search_k, static_k & -static_k)
                if (
                    max_search_m < min_search_m
                    or max_search_n < min_search_n
                    or max_search_k < mma_k
                ):
                    return reject()
            # Ragged batched extents keep the plain GEMM's search ceilings:
            # the flat one-CTA pipeline loads partial tiles and the K tail
            # through TMA and stores them with predicated SIMT copies, as it
            # does for a rank-2 GEMM (bf16 16x1000^3 ran the SIMT fallback at
            # 182 us against cuBLAS's 37 us when the tiles had to divide the
            # extents).
    if (
        static_m is not None
        and static_n is not None
        and static_m % max_search_m != 0
        and static_n % max_search_n != 0
        # the 256-row pairs' double edges keep the pair tile
        and not (
            static_k is not None
            and _double_edge_pair_shape(
                itemsize=lhs.dtype.itemsize,
                has_leading_passthrough=has_leading_passthrough,
                static_m=static_m,
                static_k=static_k,
                max_search_k=max_search_k,
            )
        )
        and not pair_matrix
    ):
        max_search_m = min(max_search_m, 128)
    # A GEMM with fewer 128x128 output tiles than SMs is bound by per-CTA
    # fixed latency, not MMA throughput: more, narrower CTAs spread the work
    # (cuBLAS runs 256^3 as 128 CTAs of 64x8 tiles).  Operands that admit
    # 256-row tiles keep 128 rows as the search floor only when the grid can
    # fill the machine; below that the 64-row one-CTA tile is searchable.
    if (
        min_search_m > 64
        and static_m is not None
        and static_n is not None
        and static_m % 64 == 0
    ):
        num_sms = _cuda_num_sms_or_zero(lhs.device)
        if (
            num_sms > 0
            and leading_work_multiplier * -(-static_m // 128) * -(-static_n // 128)
            < num_sms
        ):
            min_search_m = 64
    from ..cute_mma import _tcgen05_tma_matrix_major

    one_row_tile_narrow_n = tcgen05_one_row_tile_batched_narrow_n(
        batched_operand=has_leading_passthrough and (lhs.ndim == 3 or rhs.ndim == 3),
        a_m_major=_tcgen05_tma_matrix_major(lhs if lhs_source is None else lhs_source)
        == "col",
        itemsize=lhs.dtype.itemsize,
        static_m=static_m,
        static_n=static_n,
        static_k=static_k,
        batch=leading_work_multiplier,
        num_sms=_cuda_num_sms_or_zero(lhs.device),
    )
    if one_row_tile_narrow_n:
        max_search_n = min(max_search_n, TCGEN05_ONE_ROW_TILE_BATCHED_BLOCK_N)
    if small_n_requires_role_local_tma and (
        static_m is None
        or static_k is None
        or static_m % max_search_m != 0
        or static_k % max_search_k != 0
    ):
        return reject()

    def static_strides(tensor: torch.Tensor | None) -> tuple[int, ...] | None:
        if tensor is None:
            return None
        strides = [_static_dim_value(env, stride) for stride in tensor.stride()]
        if any(stride is None for stride in strides):
            return None
        return tuple(cast("list[int]", strides))

    warp_mma_admitted = (
        warp_mma_admission_reason(
            static_m=static_m,
            static_n=static_n,
            static_k=static_k,
            leading=leading_work_multiplier,
            dtype=lhs.dtype,
            lhs_major=_tcgen05_tma_matrix_major(
                lhs if lhs_source is None else lhs_source
            ),
            rhs_major=_tcgen05_tma_matrix_major(
                rhs if rhs_source is None else rhs_source
            ),
            num_sms=_cuda_num_sms_or_zero(lhs.device),
            lhs_strides=static_strides(lhs_source),
            rhs_strides=static_strides(rhs_source),
            operands_permuted=operands_permuted,
        )
        is None
    )
    if operand_casts and not warp_mma_admitted:
        return reject()
    return _CuteTcgen05SearchPlanningResult(
        plan=CuteTcgen05SearchPlan(
            m=m,
            n=n,
            k=k,
            static_m=static_m,
            static_n=static_n,
            static_k=static_k,
            leading_work_multiplier=leading_work_multiplier,
            is_fp8=is_fp8,
            is_small_n=is_small_n,
            small_n_supports_role_local_tma=supports_small_n_role_local_tma,
            small_n_requires_role_local_tma=small_n_requires_role_local_tma,
            mma_k=mma_k,
            max_tcgen05_m=max_tcgen05_m,
            max_tcgen05_n=max_tcgen05_n,
            max_search_m=max_search_m,
            max_search_n=max_search_n,
            max_search_k=max_search_k,
            min_search_m=min_search_m,
            min_search_n=min_search_n,
            warp_mma_admitted=warp_mma_admitted,
            tcgen05_family_admitted=not operand_casts,
            small_n_zero_fills=small_n_zero_fills,
            batched_pair_matrix=pair_matrix,
            one_row_tile_narrow_n=one_row_tile_narrow_n,
            batched_pair128_matrix=pair128_matrix,
        ),
        required_specialized_vars=frozen_required_specialized_vars,
        guards_complete=True,
    )


def plan_cute_tcgen05_search(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    has_leading_passthrough: bool,
    supports_small_n_role_local_tma: bool = False,
    supports_small_n_scalar_fallback: bool = False,
    allow_dynamic_hints: bool = False,
) -> CuteTcgen05SearchPlan | None:
    """Return search limits without mutating shared compilation state."""
    return _plan_cute_tcgen05_search_candidate(
        lhs,
        rhs,
        has_leading_passthrough=has_leading_passthrough,
        supports_small_n_role_local_tma=supports_small_n_role_local_tma,
        supports_small_n_scalar_fallback=supports_small_n_scalar_fallback,
        allow_dynamic_hints=allow_dynamic_hints,
    ).plan


def enable_cute_tcgen05_search(
    lhs: torch.Tensor,
    *,
    plan: CuteTcgen05SearchPlan,
    m_block_id: int,
    n_block_id: int,
    k_block_id: int,
    input_dtype: torch.dtype,
    has_leading_passthrough: bool,
    explicit_epi_tile_compatible: bool,
    leading_block_id: int | None = None,
    warp_mma_plain_kernel: bool = True,
    output_dtype_bytes: int | None = None,
    output_stores: int = 1,
    plain_epilogue: bool = False,
    radix_store: bool = False,
) -> None:
    """Apply one preflighted tcgen05 search plan to the shared config.

    ``output_dtype_bytes`` / ``output_stores`` size the TMA-store rings the
    AB-stage fit estimate charges (``None``: AB-only estimate);
    ``plain_epilogue``: the epilogue stores the tile once and reads no other
    tensor (the 128-row pairs' plain GEMMs); ``radix_store``: it stores
    through a hierarchical (view/permute) destination layout
    (``radix_store``), which the 128-row pairs do not render.
    """
    env = CompileEnvironment.current()
    spec = env.config_spec
    facts.register_mma_analysis(
        spec.tcgen05,
        m_block_id=m_block_id,
        n_block_id=n_block_id,
        k_block_id=k_block_id,
        compile_time_static_extents=(plan.static_m, plan.static_n, plan.static_k),
        input_dtype=input_dtype,
        has_leading_passthrough=has_leading_passthrough,
        explicit_epi_tile_compatible=explicit_epi_tile_compatible,
        leading_work_multiplier=plan.leading_work_multiplier,
    )
    static_m = plan.static_m
    static_n = plan.static_n
    static_k = plan.static_k
    max_search_m = plan.max_search_m
    max_search_n = plan.max_search_n
    max_search_k = plan.max_search_k
    # The batched 128-row pairs (``batched_pair128_matrix``) run plain
    # epilogues: an auxiliary epilogue below one full N tile keeps the SIMT
    # edge store (``cute_mma``'s ``tcgen05_use_output_edge_tma_store_for_full_tiles``),
    # and a head merge's hierarchical store renders from one-CTA tiles only.
    batched_pair128 = plan.batched_pair128_matrix and plain_epilogue and not radix_store
    pair128_widths = TCGEN05_TWO_CTA_PAIR128_ONE_WAVE_BLOCK_NS
    if batched_pair128 and static_n is not None and static_n < max(pair128_widths):
        # The power-of-two floor of a ragged N under 128 columns (64 for 80)
        # would split each matrix into a whole and a 16-column tile, the
        # second reading the matrix's A again; the 128-wide pair tile covers
        # it whole, its MMA columns past N computing zeros (B's TMA box
        # zero-fills those rows) that the 3-D TMA store clips. bf16 512 x
        # [128, 128] @ [128, 80] (GB300, CUDA graphs over cold inputs, of
        # cuBLAS): block_sizes [1, 128, 128, 128] 1.037, [1, 128, 64, 128]
        # (64 + 16 columns) 1.015. A wider N keeps its power-of-two floor
        # (128 for 200): a 256-wide pair over it is unmeasured, and past six
        # AB stages beside four C stages its edge store no longer fits SMEM
        # and falls back to the SIMT store
        # (``cute_mma``'s ``tcgen05_output_edge_tma_store_fits_smem``).
        max_search_n = carrier_n(static_n)
    if (
        batched_pair128
        and static_m is not None
        and tcgen05_batched_pair128_m_edge(static_m)
    ):
        # Likewise the power-of-two floor of an M under 128 rows (64 for 80)
        # would split each matrix into a whole and a partial 64-row one-CTA
        # tile, the second reading its B again; the 128-row pair tile covers
        # it whole, its second CTA's A box zero-filled past M and the 3-D TMA
        # store clipped there (``tcgen05_batched_pair128_m_edge``).
        max_search_m = TCGEN05_TWO_CTA_EDGE_PAIR128_BLOCK
    spec.tcgen05.search_enabled = True
    if static_m is not None and static_m < plan.min_search_m:
        # The frontend left the M search ceiling at the extent; the tcgen05
        # tile needs ``min_search_m`` rows (TMA zero-fills the rows past M
        # and the SIMT epilogue masks their stores). bf16 16x14336x4096 ran
        # the SIMT fallback at 358 us against cuBLAS's 25 us without it.
        spec.block_sizes.block_id_lookup(m_block_id).allow_overshoot(plan.min_search_m)
    edge_k_tail_shape = False
    # 16-bit output edges (M or N not a multiple of the widest tile, K whole)
    # that the persistent schedules run: TMA zero-fills the A/B rows past the
    # extent and clips the D store. A ragged M runs only through CtaGroup.TWO
    # pairs (persistent cluster_m=1 needs whole M tiles: the host multi-tile
    # guard rejects it, so those samples are flattened by
    # ``_fix_cluster_m1_persistent_search_config``), so it needs the 256-row
    # pair tile. Shapes ragged in both M and N take the same pairs when K is
    # whole and no one-CTA tile height divides M (no persistent one-CTA
    # schedule runs them): the store clips both edges (bf16 1000x2880x4096:
    # the flat one-CTA winner ran 0.64 of cuBLAS, the 256x160 pair 1.03). An
    # M whole in one-CTA tiles keeps the one-CTA N-edge family's schedules,
    # and a K tail on both edges is the edge + K-tail family's shape.
    output_edge_shape = False
    # (whether M is ragged in the search's M tiles)
    m_edge = False
    # the K tile ceiling before an output edge's whole-K cap below
    flat_max_search_k = max_search_k
    if static_m is not None and static_n is not None and static_k is not None:
        # The edge + K-tail CtaGroup.TWO family (``allow_edge_cluster_m2_search``
        # below also asks for a K tail): ragged in both M and N, or, for
        # 16-bit operands, whole in both with a K that no 64-deep tile
        # divides (the family's edges with nothing clipped; bf16
        # 2048x2560x8008, fp32 out: 0.92 of cuBLAS on the family's
        # 256x256x64 seed, 0.75 on the flat pipeline's autotuned winner),
        # or ragged in one of them with a 16-bit K tail. That edge runs on
        # its 128-row pairs too (``cute_mma``'s single-edge K tail), so its
        # floor is one 128 x 128 pair tile: GB300, CUDA graphs over cold
        # inputs, full-effort autotune, of cuBLAS, flat winner -> family:
        # bf16 8192x240x240 0.85 -> 1.03 (the 128x128 pair, two tiles per
        # CTA), 4096x240x240 0.94 -> 1.02, 1280x1600x1000 0.70 -> 0.99,
        # 320x2048x1000 0.57 -> 0.71, 1000x2560x8008 0.39 -> 0.69.
        narrow_pair_n = min(TCGEN05_TWO_CTA_PAIR128_ONE_WAVE_BLOCK_NS)
        m_ragged = static_m % TCGEN05_TWO_CTA_BLOCK_M != 0
        n_ragged = static_n % TCGEN05_TWO_CTA_BLOCK_N != 0
        k_tail = static_k % TCGEN05_WHOLE_K_TILE_MIN != 0
        edge_k_tail_shape = (
            not has_leading_passthrough
            and plan.max_tcgen05_m >= TCGEN05_TWO_CTA_BLOCK_M
            and plan.max_tcgen05_n >= TCGEN05_TWO_CTA_BLOCK_N
            and min(static_m, static_n, static_k) >= TCGEN05_TWO_CTA_EDGE_K_TAIL_MIN_DIM
            and m_ragged == n_ragged
            and (m_ragged or (k_tail and input_dtype.itemsize == 2))
        ) or (
            not has_leading_passthrough
            and input_dtype.itemsize == 2
            and k_tail
            and m_ragged != n_ragged
            and plan.max_tcgen05_m >= TCGEN05_TWO_CTA_EDGE_PAIR128_BLOCK
            and min(static_m, static_k) >= TCGEN05_TWO_CTA_EDGE_PAIR128_BLOCK
            and (
                min(plan.max_tcgen05_n, static_n) >= TCGEN05_TWO_CTA_EDGE_PAIR128_BLOCK
                # or one N tile of the 64-wide pair, where a wave of them fits
                # (``tcgen05_edge_pair128_narrow_width_fits``): bf16
                # 16384x64x1000 would take 256 CTAs and stays one-CTA
                or (
                    plan.max_tcgen05_n == static_n == narrow_pair_n
                    and tcgen05_edge_pair128_narrow_width_fits(
                        rows=plan.leading_work_multiplier
                        * -(-static_m // TCGEN05_TWO_CTA_EDGE_PAIR128_BLOCK),
                        n=static_n,
                        num_sms=_cuda_num_sms_or_zero(lhs.device),
                    )
                )
            )
        )
        m_edge = static_m % max_search_m != 0
        n_edge = static_n % max_search_n != 0
        output_edge_shape = (
            # one ragged edge of a 16-bit GEMM
            (
                m_edge != n_edge
                and not has_leading_passthrough
                and input_dtype.itemsize == 2
            )
            # a batch of matrices on the CTA pairs (``batched_pair_matrix``),
            # K whole
            or (plan.batched_pair_matrix and static_k % max_search_k == 0)
            # or on the 128-row pairs over an M edge, an N edge or a K tail
            # (``batched_pair128_matrix``)
            or batched_pair128
            # both: the pairs' double-edge shape
            or (
                m_edge
                and n_edge
                and _double_edge_pair_shape(
                    itemsize=input_dtype.itemsize,
                    has_leading_passthrough=has_leading_passthrough,
                    static_m=static_m,
                    static_k=static_k,
                    max_search_k=max_search_k,
                )
            )
        ) and (
            not m_edge
            or max_search_m >= TCGEN05_TWO_CTA_BLOCK_M
            # the 128-row pairs clip a matrix's M edge too
            # (``tcgen05_batched_pair128_m_edge``)
            or batched_pair128
        )
        whole_k_tile = static_k & -static_k
        if (
            input_dtype.itemsize == 2
            and static_k % max_search_k != 0
            and not edge_k_tail_shape
            # (the batched 128-row pairs issue a K tail past their first
            # k-block themselves, and the flat tiles keep their 128-deep
            # k-blocks: bf16 512 x [128, 192] @ [192, 80] searches K up to 128)
            and not batched_pair128
            and (output_edge_shape or not (m_edge or n_edge))
            and whole_k_tile >= TCGEN05_WHOLE_K_TILE_MIN
        ):
            # Only the edge + K-tail family clamps a 16-bit K tail box (FP8
            # keeps its K128 tiles and the proved paired K-tail protocol);
            # any other K tail confines the search to the flat one-CTA
            # pipeline. Whole K tiles cost nothing there (bk=64 is the
            # deep-pipeline tile anyway), so where whole K tiles admit the
            # persistent and CtaGroup.TWO schedules, cap the K tile at the
            # largest power of two dividing K: bf16 1000x4096x2880 ran its
            # 64-deep two-CTA tile at 26.9 us against cuBLAS's 26.6 and the
            # flat winner's 30.7.
            max_search_k = whole_k_tile
    cluster_m2_max_k_tiles = tcgen05_two_cta_max_k_tiles(input_dtype.itemsize)
    max_cluster_m2_search_k = cluster_m2_max_k_tiles * max_search_k
    if static_m is None or static_n is None or static_k is None:
        allow_whole_k_persistent_pid_types = False
        allow_small_n_persistent_pid_types = False
        allow_k_tail_persistent_pid_types = False
        allow_full_tile_cluster_m2_search = False
        allow_edge_cluster_m2_search = False
        allow_fp8_small_grid_cluster_m2_search = False
        allow_one_wave_cluster_m2_search = False
    else:
        # Whole tiles, or a 16-bit output edge (``output_edge_shape``) with K
        # whole: only a K tail needs the edge family's clamped K box, and FP8
        # keeps its proved exact-tile restoration. (25216x768x3072 and
        # 4096x8008x2560 came out exact on the two-CTA tile as well.)
        allow_whole_k_persistent_pid_types = static_k % max_search_k == 0 and (
            output_edge_shape
            or (static_m % max_search_m == 0 and static_n % max_search_n == 0)
        )
        allow_small_n_persistent_pid_types = (
            plan.is_small_n
            and plan.small_n_supports_role_local_tma
            and 0 < static_n < max_search_n
            and static_m % max_search_m == 0
            and static_k % max_search_k == 0
        )
        # A 16-bit K tail over whole output tiles runs the persistent one-CTA
        # kernel, whose role-local producer issues the partial last K tile as
        # a whole box (``cute_mma``'s ``tcgen05_preserve_tma_for_one_cta_k_tail``);
        # without it the search ran only flat tiles. helion.inductor's
        # transposed bf16 128x768x1000 with b stored [K, N] (768 x 128, K =
        # 7 x 128 + 104), GB300, of cuBLAS: flat [64, 16, 128] 0.900, the same
        # tile persistent 0.935.
        allow_k_tail_persistent_pid_types = (
            static_k % max_search_k != 0
            and static_m % min(max_search_m, TCGEN05_ONE_CTA_MAX_BLOCK_M) == 0
            and static_n % max_search_n == 0
            and lhs.dtype in (torch.bfloat16, torch.float16)
            and not has_leading_passthrough
        )
        allow_full_tile_cluster_m2_search = (
            allow_whole_k_persistent_pid_types
            and max_search_m >= TCGEN05_TWO_CTA_BLOCK_M
            and (
                max_search_n >= TCGEN05_TWO_CTA_BLOCK_N
                # a GEMM one narrow N tile wide runs the 256 x N pair, unless
                # a registered block size leaves its 256 x 128 tile out
                or facts.narrow_n_pair_gemm(spec.tcgen05)
            )
            and static_k <= max_cluster_m2_search_k
        )
        allow_edge_cluster_m2_search = (
            edge_k_tail_shape
            and not allow_whole_k_persistent_pid_types
            and static_k <= max_cluster_m2_search_k
            and static_k % max_search_k != 0
        )
        allow_fp8_small_grid_cluster_m2_search = (
            not has_leading_passthrough
            and plan.is_fp8
            and allow_whole_k_persistent_pid_types
            and max_search_m >= TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
            and max_search_n >= TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N
            and static_k <= max_cluster_m2_search_k
        )
        # One-wave 256 x {128, 64} CtaGroup.TWO tiles for GEMMs whose
        # 256x256 two-CTA grid leaves most SMs idle (below, the small-grid
        # demotion turns cluster_m=2 search off when even the narrowest of
        # these tiles cannot fill a quarter of the SMs).
        allow_one_wave_cluster_m2_search = (
            allow_full_tile_cluster_m2_search
            # the one-wave two-CTA tiles run whole M tiles, or a 16-bit
            # output edge's partial last M tile, which the pair's TMA store
            # clips (bf16 1000x2304x2048: 72 256x128 pairs, one per SM pair,
            # 0.97 of cuBLAS against 0.77 for the flat one-CTA winner), and
            # whole N tiles, or a single N edge's partial last N tile (M
            # whole in the search's M tiles), which it clips too: the
            # transposed bf16 1000x1024x2048 (out.T 1024 x 1000) runs 72
            # pairs of 128x112 (``dense_aligned_n_block_ms``), 0.96 of
            # cuBLAS, where without this the search ran only the flat
            # one-CTA pipeline (16 pairs of 256x256 are under the
            # quarter-wave gate below). ``transposed_pair_tile_gain``
            # prices that pair on these grids only.
            and (static_m % TCGEN05_TWO_CTA_BLOCK_M == 0 or output_edge_shape)
            and (
                static_n % min(TCGEN05_TWO_CTA_ONE_WAVE_BLOCK_NS) == 0
                or (output_edge_shape and not m_edge)
            )
            and plan.leading_work_multiplier
            * -(-static_m // TCGEN05_TWO_CTA_BLOCK_M)
            * -(-static_n // TCGEN05_TWO_CTA_BLOCK_N)
            * 2
            < _cuda_num_sms_or_zero(lhs.device)
        )
    allow_cluster_m2_search = (
        allow_full_tile_cluster_m2_search
        or allow_edge_cluster_m2_search
        or allow_fp8_small_grid_cluster_m2_search
    )
    # True when only the one-wave tiles clear the quarter-wave gate below:
    # the 256x256 two-CTA families then stay off (their seeds, the FFI
    # coordinate and the aux-TMA surface) and the search keeps just the
    # narrow tiles.
    one_wave_only_cluster_m2_search = False
    num_sms = _cuda_num_sms_or_zero(lhs.device)
    spec._cute_tcgen05_config.device_sm_count = num_sms
    spec._cute_tcgen05_config.persistent_reserved_sms = (
        env.settings.persistent_reserved_sms
    )
    # The register-MMA family replaces the whole device body of ONE GEMM (its
    # root tile loop around its K loop; ``detect_warp_mma_gemm`` proves that
    # root holds nothing else); kernels with several matmuls keep the tcgen05
    # surface alone.
    warp_mma_admitted = plan.warp_mma_admitted and warp_mma_plain_kernel
    assert plan.tcgen05_family_admitted or warp_mma_admitted
    spec._cute_tcgen05_config.warp_mma_admitted = warp_mma_admitted
    spec._cute_tcgen05_config.tcgen05_family_admitted = plan.tcgen05_family_admitted
    spec._cute_tcgen05_config.tcgen05_min_search_m = plan.min_search_m
    spec._cute_tcgen05_config.output_edge_persistent_admitted = output_edge_shape and (
        allow_whole_k_persistent_pid_types or batched_pair128
    )
    spec._cute_tcgen05_config.batched_pair128_matrix = batched_pair128
    spec._cute_tcgen05_config.matmul_radix_store = radix_store
    spec._cute_tcgen05_config.tcgen05_min_search_n = plan.min_search_n
    spec._cute_tcgen05_config.one_row_tile_narrow_n = plan.one_row_tile_narrow_n
    spec._cute_tcgen05_config.matmul_leading_block_id = leading_block_id
    # True when the edge K-tail family clears the quarter-wave gate on its
    # 64-wide pairs alone (``repair.plain_pair128_widths``): the search then
    # keeps just those pairs.
    edge_pair128_only = False
    if allow_cluster_m2_search:
        assert static_m is not None and static_n is not None and static_k is not None
        if num_sms > 0:
            if allow_fp8_small_grid_cluster_m2_search:
                cluster_m = TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
                cluster_n = TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N
            else:
                cluster_m = TCGEN05_TWO_CTA_BLOCK_M
                cluster_n = TCGEN05_TWO_CTA_BLOCK_N
            # Partial last tiles run too (the persistent pair schedules clip
            # them): a 1000x2560 GEMM is 40 256x256 pairs, not 30.
            work_clusters = (
                plan.leading_work_multiplier
                * -(-static_m // cluster_m)
                * -(-static_n // cluster_n)
            )
            if allow_one_wave_cluster_m2_search:
                one_wave_work_clusters = (
                    plan.leading_work_multiplier
                    * -(-static_m // TCGEN05_TWO_CTA_BLOCK_M)
                    * -(-static_n // min(TCGEN05_TWO_CTA_ONE_WAVE_BLOCK_NS))
                )
                pair128_n = min(TCGEN05_TWO_CTA_PAIR128_ONE_WAVE_BLOCK_NS)
                if (
                    input_dtype.itemsize == 2
                    and plain_epilogue
                    and static_m % (TCGEN05_TWO_CTA_BLOCK_M // 2) != 0
                    and static_n % pair128_n == 0
                ):
                    # The 16-bit family also runs 128-row CTA pairs, each
                    # CTA 64 rows (``tcgen05_config.repair.plain_pair128_widths``),
                    # so its work is counted in its narrowest pair tile too:
                    # bf16 1500x384x384 has 36 tiles of 256x64, under a
                    # quarter of GB300's 152 SMs, but 72 pairs of 128x64
                    # (cuBLAS's tile), which ran 2.69 us against the flat
                    # one-CTA winner's 3.12 (cuBLAS 2.55). Only where the
                    # family takes those pairs: a plain epilogue (an
                    # auxiliary operand keeps its 256-row tiles) over an M
                    # whole 128-row tiles do not cover (over whole tiles
                    # the paired-pipeline search runs the 128-row pairs
                    # without this family).
                    one_wave_work_clusters = max(
                        one_wave_work_clusters,
                        plan.leading_work_multiplier
                        * -(-static_m // (TCGEN05_TWO_CTA_BLOCK_M // 2))
                        * (static_n // pair128_n),
                    )
                one_wave_only_cluster_m2_search = (
                    work_clusters < num_sms // 4 <= one_wave_work_clusters
                )
                work_clusters = max(work_clusters, one_wave_work_clusters)
            if allow_edge_cluster_m2_search and input_dtype.itemsize == 2:
                # The 16-bit edge family also runs 128x128 CTA pairs
                # (``tcgen05_config.repair.plain_pair128_admitted``: each CTA
                # owns 64 rows), so its work is counted in 128x128 tiles,
                # partial ones included: a ragged bf16 1000^3 GEMM has 16
                # 256x256 pairs but 64 of 128x128, enough to keep the two-CTA
                # search on (cuBLAS runs it as a 2x1-cluster 64x112 kernel;
                # the 128x128 pair ran 7.2 us against the flat one-CTA
                # winner's 8.4). fp8 kernels keep the pair count above.
                # Auxiliary epilogues are detected after planning
                # (``tcgen05.aux_kernel_detected``), so a 16-bit kernel
                # with one is admitted on this count too and runs the
                # family's 256x256 pairs, which the pair-128 gate excludes.
                edge_rows = plan.leading_work_multiplier * -(
                    -static_m // TCGEN05_TWO_CTA_EDGE_PAIR128_BLOCK
                )
                edge_work_clusters = edge_rows * -(
                    -static_n // TCGEN05_TWO_CTA_EDGE_PAIR128_BLOCK
                )
                narrow_n = min(TCGEN05_TWO_CTA_PAIR128_ONE_WAVE_BLOCK_NS)
                if plain_epilogue and tcgen05_edge_pair128_narrow_width_fits(
                    rows=edge_rows, n=static_n, num_sms=num_sms
                ):
                    # A plain epilogue's family also runs its 64-wide pairs on
                    # a grid of one wave of them, whose CTAs keep it on where
                    # the 128x128 pairs sit under the gate but they fill a
                    # quarter of the SMs, alone (``edge_pair128_only``): the
                    # transposed bf16 128x2304x1000 with b stored [K, N]
                    # (out.T 2304 x 128) has 18 tiles of 128x128 and 36 pairs
                    # of 128x64, 72 CTAs, which ran 0.90 of cuBLAS against the
                    # one-CTA pick's 0.79 (GB300, CUDA graphs over cold
                    # inputs). An auxiliary epilogue, which runs no 128-row
                    # pair, stays under the gate.
                    narrow_pairs = edge_rows * (static_n // narrow_n)
                    edge_pair128_only = (
                        max(work_clusters, edge_work_clusters) < num_sms // 4
                        and 2 * narrow_pairs >= num_sms // 4
                    )
                work_clusters = max(work_clusters, edge_work_clusters)
            if work_clusters < num_sms // 4 and not edge_pair128_only:
                allow_cluster_m2_search = False
                allow_edge_cluster_m2_search = False
                allow_fp8_small_grid_cluster_m2_search = False
                allow_one_wave_cluster_m2_search = False
    if (
        output_edge_shape
        and allow_whole_k_persistent_pid_types
        and static_m is not None
        and _one_cta_heights_miss_m(static_m)
        and not allow_cluster_m2_search
        # (the batched 128-row pairs reach a ragged M through the paired
        # pipeline: ``tcgen05_batched_pair128_m_edge``)
        and not batched_pair128
    ):
        # An M ragged for every one-CTA tile height runs persistent only
        # through the CtaGroup.TWO pairs: where none is reachable (bf16
        # 1000x128x2880: four pairs, under the quarter-wave gate) the
        # persistent pid types and the K cap would only cost the flat
        # pipeline its 128-deep K tiles. The admission flag stays set, so
        # the search repair still flattens a persistent one-CTA sample
        # (``_fix_cluster_m1_persistent_search_config``).
        allow_whole_k_persistent_pid_types = False
        max_search_k = flat_max_search_k
        if static_n is not None and static_n % max_search_n != 0:
            # A double edge kept the 256-row M ceiling for its pairs alone
            # (``_double_edge_pair_shape``); with none reachable it takes the
            # planner's one-CTA ceiling back (bf16 1000x1000x4096, 16 pairs).
            # A single M edge keeps 256 rows: the cp.async-A pipeline runs
            # them as two 128-row MMAs, and the repair projects the TMA
            # kernels' flat samples to 128 rows
            # (``_fix_cluster_m1_flat_block_m_search_config``).
            max_search_m = min(max_search_m, TCGEN05_ONE_CTA_MAX_BLOCK_M)
    # A batched 16-bit GEMM whose K is one K tile (64 deep at most) is
    # output-bound: every tile loads once and stores its whole output. Its
    # CTA pairs take that K in one k-block. A pair that splits it (bf16 512 x
    # 512x512x64: block_sizes [1, 256, 256, 16], four 16-deep k-blocks a
    # tile)
    #     for tile_r in hl.tile(64):  # _BLOCK_SIZE_3 = 16: four AB stages
    # issues four TMA stages per tile where one carries the same bytes; the
    # final selection's per-call timing (an L2 flush before each launch)
    # reads it 1-8% ahead of the K-whole tiles, but back to back in a CUDA
    # graph it runs behind them: 512^2 pair over k-blocks of 16 read 51.3 us
    # and ran 55.2 (0.947 of cuBLAS), the 128x64 one-CTA tile 50.9 / 52.0
    # (0.993); 256 x 512x512x64 27.3 / 28.0-29.4 against 28.2 / 27.2;
    # 384 x 512x512x64 39.9 / 42.4 against 38.9 / 39.4 (GB300). With K
    # whole the pairs read what they run (512^2: 57.0 / 57.3), so the
    # selection keeps the tile that wins in the graph. One-CTA tiles keep
    # their k-blocks: 256x512 runs 128x128 over 32-deep k-blocks at 1.056 of
    # cuBLAS against 0.96 over K whole. A K block size the kernel bounds
    # below K (``hl.register_block_size(16, 32)`` for K = 64) keeps its pairs
    # over its own k-blocks.
    matmul_fragments = facts._matmul_block_fragments(spec.tcgen05)
    cluster_m2_whole_k = (
        has_leading_passthrough
        and input_dtype.itemsize == 2
        and static_k is not None
        and static_k == max_search_k
        and static_k <= TCGEN05_WHOLE_K_TILE_MIN
        and matmul_fragments is not None
        and matmul_fragments[2].low <= static_k <= matmul_fragments[2].high
    )
    surface.narrow_autotune_to_validated_configs(
        spec.tcgen05,
        allow_persistent_pid_types=(
            allow_whole_k_persistent_pid_types
            or allow_small_n_persistent_pid_types
            or allow_k_tail_persistent_pid_types
            # (the 128-row pairs' own schedules: ``batched_pair128_matrix``)
            or batched_pair128
        ),
        enable_cluster_m2_search=allow_cluster_m2_search,
        cluster_m2_static_k=static_k if allow_cluster_m2_search else None,
        cluster_m2_max_k_tiles=cluster_m2_max_k_tiles,
        allow_cluster_m2_edge_k_tail_family=allow_edge_cluster_m2_search,
        allow_cluster_m2_fp8_small_grid=allow_fp8_small_grid_cluster_m2_search,
        allow_cluster_m2_one_wave_tiles=allow_one_wave_cluster_m2_search,
        cluster_m2_one_wave_only=one_wave_only_cluster_m2_search,
        cluster_m2_edge_pair128_only=edge_pair128_only,
        cluster_m2_whole_k=cluster_m2_whole_k,
        ab_stages_three_dtype_bytes=lhs.dtype.itemsize,
        ab_stages_three_device=lhs.device,
        ab_stages_three_output_dtype_bytes=output_dtype_bytes,
        ab_stages_three_output_stores=output_stores,
        reason="matmul kernel with CuTe tcgen05 backend",
    )
    if plan.small_n_requires_role_local_tma:
        spec.disallow_pid_type(
            "flat",
            reason="small-N MMA operands require role-local TMA loads",
        )
        spec.disallow_pid_type(
            "xyz",
            reason="small-N MMA operands require role-local TMA loads",
        )
    for axis_name, shape, max_size in (
        ("m", plan.m, max_search_m),
        ("n", plan.n, max_search_n),
        ("k", plan.k, max_search_k),
    ):
        block_idx = env.get_block_id(shape)
        if block_idx is None:
            continue
        if axis_name == "k":
            floor = min_size = plan.mma_k
        elif axis_name == "m":
            # The register-MMA family's 16-row tiles widen the M (and N)
            # search floor where it is admitted; ``cute_matmul_family=
            # "tcgen05"`` configs are clamped back to the tcgen05 floor by
            # ``tcgen05_config.repair._normalize_warp_mma_family``.
            floor = plan.min_search_m
            min_size = WARP_MMA_MIN_BM if warp_mma_admitted else floor
        else:
            floor = plan.min_search_n
            min_size = min(WARP_MMA_MIN_BN, floor) if warp_mma_admitted else floor
        env.block_sizes[block_idx].update_min_block(min_size, allow_flattened=True)
        if axis_name == "k":
            surface.cap_k_search(
                spec._cute_tcgen05_config, env.block_sizes[block_idx], max_size
            )
        else:
            env.block_sizes[block_idx].update_max_block(max_size)
        if min_size < floor:
            # The widened floor must not move the default (non-autotuned)
            # block: the default config stays the tcgen05 tile it was, and the
            # autotuner's flat default round-trips through normalization.
            with contextlib.suppress(KeyError):
                block_spec = spec.block_sizes.block_id_lookup(block_idx)
                if block_spec.default_size is None:
                    block_spec.default_size = max(
                        block_spec._fragment(spec).default(), floor
                    )
    if plan.small_n_zero_fills:
        # A small N's zero-filled tile (``_plan_cute_tcgen05_search_candidate``)
        # raises the search floors only, so the default and the search take
        # it and a provided config keeps its tiles: the 8-column N tile, past
        # the N extent's power of two, and the K tiles that feed it a K-major
        # B (``TCGEN05_SMALL_N_MIN_K_BYTES``; the planner proved the K block
        # reaches them and the N block's range holds the 8-column tile).
        for overshoots, shape, search_floor in (
            (True, plan.n, tcgen05_min_n(lhs.dtype)),
            (False, plan.k, TCGEN05_SMALL_N_MIN_K_BYTES // lhs.dtype.itemsize),
        ):
            block_idx = env.get_block_id(shape)
            if block_idx is None:
                continue
            with contextlib.suppress(KeyError):
                block_spec = spec.block_sizes.block_id_lookup(block_idx)
                if overshoots:
                    block_spec.allow_overshoot(search_floor)
                block_spec.autotuner_min = max(block_spec.autotuner_min, search_floor)
