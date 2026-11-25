"""Aligned physical N tiles behind ordinary power-of-two public carriers."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import math
from typing import TYPE_CHECKING
from typing import cast

import torch

from .one_cta_pipeline import one_cta_pipeline_defaults
from .pipeline_smem import TCGEN05_SMEM_AWARE_MAX_AB_STAGES
from .pipeline_smem import Tcgen05PipelineSmemFacts
from .pipeline_smem import deepest_fitting_ab_stages
from .pipeline_smem import paired_pipeline_max_ab_stages
from .pipeline_smem import pipeline_smem_bytes
from .strategies import TCGEN05_LAYOUT_OVERRIDES_KEYS
from .tcgen05_constants import TCGEN05_ALIGNED_N_MN_B_ROW_BYTES
from .tcgen05_constants import TCGEN05_EPI_SUBTILE_N_CONFIG_KEY
from .tcgen05_constants import TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY
from .tcgen05_constants import TCGEN05_MN_B_64B_ATOM_STREAM_COST
from .tcgen05_constants import TCGEN05_ODD_WIDTH_SUBTILE_N
from .tcgen05_constants import TCGEN05_ONE_CTA_MAX_BLOCK_M
from .tcgen05_constants import TCGEN05_PAIR_TILE_FIXED_KBLOCKS
from .tcgen05_constants import TCGEN05_PAIR_TILE_STREAM_RATIO
from .tcgen05_constants import TCGEN05_PLAIN_NARROW_SUBTILE_BLOCK_NS
from .tcgen05_constants import TCGEN05_PLAIN_NARROW_SUBTILE_N
from .tcgen05_constants import TCGEN05_TRANSPOSED_PRODUCT_MIN_GAIN
from .tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from .tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from .tcgen05_constants import TCGEN05_TWO_CTA_PAIR128_ONE_WAVE_BLOCK_NS
from .tcgen05_constants import TCGEN05_TWO_CTA_SIXTEEN_BIT_MAX_K_TILES
from .tcgen05_constants import tcgen05_odd_width_subtile
from .tcgen05_constants import tcgen05_pdl_l2_warmup_kblocks
from .tcgen05_constants import tcgen05_plain_narrow_subtile_covers_n
from .tcgen05_constants import tcgen05_two_cta_max_k_tiles

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .dense_gemm import DenseGemmFacts


TILE_N_KEY = "tcgen05_tile_n"
ALIGNED_N_CHOICES = tuple(n for n in range(64, 257, 16) if n & (n - 1))


def carrier_n(physical_n: int) -> int:
    return 1 << (physical_n - 1).bit_length()


def dense_aligned_n_block_m(input_dtype: torch.dtype) -> int | None:
    """Pair M of the family: the paired M128 FP8 instruction (64 rows per
    CTA), or the paired M256 instruction for 16-bit inputs (128 rows per CTA)."""
    if input_dtype == torch.float8_e4m3fn:
        return 128
    if dense_aligned_n_sixteen_bit(input_dtype):
        return TCGEN05_TWO_CTA_BLOCK_M
    return None


def dense_aligned_n_block_ms(
    input_dtype: torch.dtype,
    *,
    n: int,
    output_column_major: bool,
    plain_epilogue: bool,
) -> tuple[int, ...]:
    """Every pair M of the family, the default (``dense_aligned_n_block_m``)
    first. A plain 16-bit pair over a row-major output whose N one pair tile
    spans (at most 256 columns) also runs the paired M128 instruction, each
    CTA 64 rows of the tile (as the 128-row pairs of
    ``tcgen05_config.repair.plain_pair128_widths`` do, plain GEMMs only),
    where the 256-row pairs read A once per N tile but quantize M or the wave
    badly. bf16 25216x192x768
    (GB300, CUDA graphs over cold inputs, cuBLAS 10.47 us): 197 pairs of
    128x192 (one N tile: A read once) 10.14 us, 99 of 256x192 11.28 us (two
    rounds on 76 pairs for 1.3 rounds of work), 394 of the 128-row pair at
    128 + 64 columns (A read twice) 11.31 us.

    So does a plain 16-bit pair over a column-major output (the transposed
    product's out.T) at any N, where its CTA's last tile stages its
    16-column subtiles at once (``cute_mma``'s
    ``tcgen05_last_tile_staging``) instead of storing them in 16-column
    rounds. bf16 1000x1024xK transposed (out.T: 1024 x 1000) runs cuBLAS's
    swapped tile, 72 pairs of 128x112 (each CTA streams 64 + 56 operand
    rows per K step), against the as-written 64 pairs of 128x128 (64 + 64);
    GB300, in one process, two A/B-load warps, of cuBLAS: K=1024 0.950 ->
    0.955, 2048 0.937 -> 0.957, 4096 0.946 -> 0.970 (the 16-column rounds
    0.896 at K=1024, 0.918 at 2048).
    """
    block_m = dense_aligned_n_block_m(input_dtype)
    if block_m is None:
        return ()
    if (
        dense_aligned_n_sixteen_bit(input_dtype)
        and plain_epilogue
        and (output_column_major or n <= TCGEN05_TWO_CTA_BLOCK_N)
    ):
        return (block_m, TCGEN05_TWO_CTA_BLOCK_M // 2)
    return (block_m,)


def dense_aligned_n_sixteen_bit(input_dtype: torch.dtype) -> bool:
    """Whether the family runs the 16-bit paired instructions (M256, and M128
    where ``dense_aligned_n_block_ms`` admits it)."""
    return input_dtype in (torch.bfloat16, torch.float16)


def dense_aligned_n_mn_b_atom_bytes(width: int, itemsize: int) -> int:
    """The swizzle atom, in bytes, of the SMEM rows each CTA of the pair stages
    an N-major B in at a ``width``-column tile: the largest of 128, 64 and 32
    that divides the half row's bytes, as CuTe's ``make_smem_layout_b`` picks
    it (16: none)."""
    row_bytes = width // 2 * itemsize
    return next((atom for atom in (128, 64, 32) if row_bytes % atom == 0), 16)


def dense_aligned_n_mn_b_width(width: int, itemsize: int) -> bool:
    """Whether the pair runs a ``width``-column tile over an N-major B (a
    [K, N] row-major weight) at full speed: each CTA of the pair stages half
    of the tile's columns as one SMEM row of B, which must hold whole 64-byte
    swizzle atoms (``TCGEN05_ALIGNED_N_MN_B_ROW_BYTES``). The bf16 256x160
    pair's B ring over an N-major B is

        sB_layout = make_smem_layout_b(tiled_mma, (256, 160, 64), BFloat16, 8)
        # each CTA: 80-column rows of 160 B -> 32-byte swizzle atoms

    where 256 columns stage 256-byte rows (128-byte atoms) and 192 columns
    192-byte rows (64-byte atoms). GB300, of cuBLAS in one process: bf16
    2048x2560x2560 with B [K, N] row-major runs the 256x160 pair at 0.76
    (CUTLASS's dense_gemm_persistent at the same tile 0.77) and 256x192 at
    0.85; 8008x2560x4096 runs 160 columns at 0.74 and 256 at 1.09,
    20008x768x2048 96 columns at 0.66 and 256 at 0.88."""
    return (
        dense_aligned_n_mn_b_atom_bytes(width, itemsize)
        >= TCGEN05_ALIGNED_N_MN_B_ROW_BYTES
    )


# The 16-bit pair's short-K families run the deep 64-wide ring.
DENSE_ALIGNED_N_SIXTEEN_BIT_BLOCK_K = 64


def dense_aligned_n_defaults(
    *, acc_stages: int = 2, store_mode: str = "normal"
) -> dict[str, object]:
    assert store_mode in ("normal", "direct")
    return {
        **one_cta_pipeline_defaults(),
        "tcgen05_cta_group": "two",
        "tcgen05_cluster_m": 2,
        "tcgen05_c_store_mode": store_mode,
        "tcgen05_acc_stages": acc_stages,
        "tcgen05_l2_swizzle_size": 1,
        "l2_groupings": [1],
        "loop_orders": [[0, 1]],
        "num_sm_multiplier": 1,
        # The family's own epilogue tile: a seed merged onto a default that
        # names explicit layout overrides (the direct-entry default's
        # (128, 32) tile) must not inherit them.
        **dict.fromkeys(TCGEN05_LAYOUT_OVERRIDES_KEYS),
    }


def pair_tile_wave_cost(
    m: int,
    n: int,
    k: int,
    *,
    width: int,
    block_m: int,
    sm_pairs: int,
    stream_cost: float = 1.0,
) -> float:
    """Wave-quantized time of a 16-bit pair-tile grid over ``m`` x ``n``
    (arbitrary units: 256-column K-block MMAs). Each of the
    ``ceil(m / block_m) x ceil(n / width)`` tiles takes ``ceil(k / 64)`` K
    blocks, each the larger of its MMA time (its width, halved on the
    128-row pair's M128 instruction) and its operand stream
    (``TCGEN05_PAIR_TILE_STREAM_RATIO`` x (block_m + width): a narrow tile
    reloads the A tile per N tile), plus a fixed prologue and epilogue
    (``TCGEN05_PAIR_TILE_FIXED_KBLOCKS`` 256-wide K blocks); ``sm_pairs`` tiles
    run per wave. ``stream_cost`` scales the K blocks' time (an operand
    layout that slows them: ``TCGEN05_MN_B_64B_ATOM_STREAM_COST``)."""
    tiles = -(-m // block_m) * -(-n // width)
    k_blocks = -(-k // DENSE_ALIGNED_N_SIXTEEN_BIT_BLOCK_K)
    k_block = max(
        width * block_m / TCGEN05_TWO_CTA_BLOCK_M,
        TCGEN05_PAIR_TILE_STREAM_RATIO * (block_m + width),
    )
    fixed = TCGEN05_PAIR_TILE_FIXED_KBLOCKS * TCGEN05_TWO_CTA_BLOCK_N
    return -(-tiles // sm_pairs) * (k_blocks * k_block * stream_cost + fixed)


def transposed_double_edge_without_pairs(m: int, n: int) -> bool:
    """Whether the transposed product (out.T = b.T @ a.T, ``n`` x ``m``) of a
    16-bit ``m`` x ``n`` output is a double edge whose M (``n``) a 64-row
    one-CTA tile divides: ragged in both against the 256x256 pair tile. Over
    whole K tiles the search keeps the 256-row pairs on a double edge only
    where no one-CTA tile height divides its M
    (``search_plan._double_edge_pair_shape``) and caps any other M at 128
    rows, leaving it no CTA pair; over a K tail both orientations run the
    edge + K-tail pairs (unmeasured either way). bf16 2040x20032x512: as
    written (M = 2040 = 31 x 64 + 56) the pairs; transposed (M = 20032 =
    313 x 64) no ``cluster_m2_search_constraints``, only one-CTA tiles, where
    20008 rows (312 x 64 + 40) would keep the pairs."""
    return (
        m % TCGEN05_TWO_CTA_BLOCK_N != 0
        and n % TCGEN05_TWO_CTA_BLOCK_M != 0
        # a one-CTA tile height divides it (the opposite of
        # ``search_plan._one_cta_heights_miss_m``)
        and n % (TCGEN05_ONE_CTA_MAX_BLOCK_M // 2) == 0
    )


@dataclass(frozen=True)
class PairTileOrientation:
    """The pair-tile model's two orientations of a 16-bit GEMM over a K-major
    a (``transposed_pair_tile_orientation``)."""

    # as written / transposed ``pair_tile_wave_cost`` of each orientation's
    # best tiling; 1.0 where the model prices neither
    gain: float
    # whether the dependent-launch L2 warm-up runs on that best grid of
    # 256-row pairs (``tcgen05_pdl_l2_warmup_kblocks``: over the SMs its
    # previous call's last wave leaves idle)
    written_warms: bool = False
    transposed_warms: bool = False

    @property
    def transposes(self) -> bool:
        """Whether helion.inductor lowers the GEMM as its transposed product
        (``HelionChoices.transpose_matmul``): a gain of at least
        ``TCGEN05_TRANSPOSED_PRODUCT_MIN_GAIN``, or a tie the warm-up breaks.

        The model prices a wave by its tiles' K loop, so two grids of one
        wave tie however many SMs each leaves idle. Under a dependent launch
        the grid that leaves eight or more idle gets the warm-up: the early
        CTAs on those SMs pull the first k-blocks into L2 while the previous
        call runs. bf16 1000x2048x2304: as written 4 x 19 = 76 tiles of
        256x112 fill GB300's 76 SM pairs; transposed 8 x 9 = 72 leave 8 SMs
        idle. The same config either way round, in one process, of cuBLAS,
        back to back: 0.979/0.982 -> 1.002/1.003 (the transposed product
        without its warm-up 0.986/0.979); x2560 0.980/0.978 -> 1.009/1.013;
        the two-wave 6272x1024xK on 256x176 (150 tiles leave 4 idle, 144
        transposed 16) K=1024 0.972/0.973 -> 1.034/1.031, K=2048 0.998/0.998
        -> 1.035/1.043, K=4096 1.001/0.998 -> 1.011/1.015.
        """
        if self.gain >= TCGEN05_TRANSPOSED_PRODUCT_MIN_GAIN:
            return True
        return self.gain >= 1.0 and self.transposed_warms and not self.written_warms


def transposed_pair_tile_orientation(
    m: int, n: int, k: int, *, sm_pairs: int, b_n_major: bool = False
) -> PairTileOrientation:
    """How the 16-bit pair runs an ``m`` x ``n`` x ``k`` GEMM over a K-major
    ``a`` as written and as its transpose (``n`` x ``m`` x ``k``, out.T =
    b.T @ a.T): the ratio of the two orientations' best
    ``pair_tile_wave_cost`` over the pair's widths (the power-of-two ones
    and ``ALIGNED_N_CHOICES``; a partial last 256-row tile needs whole K
    tiles), and whether the dependent-launch warm-up runs on each best grid.

    The pair tiles its M side in 256-row steps and its N side in 16-column
    ones, so a tall output tiles finer transposed. bf16 16384x1024x4096 on
    76 SM pairs: 64 x 4 = 256 tiles of 256x256 (four waves) against 4 x 74
    = 296 tiles of 256x224 transposed (four waves of 7/8 the work): 1.13
    (measured: ``HelionChoices.transpose_matmul``).

    Over an N-major ``b`` (``b_n_major``: b stored [K, N]) the as-written
    pair runs only the widths whose half rows of B hold whole 64-byte
    swizzle atoms (``dense_aligned_n_mn_b_width``: 64, 128, 192 and 256
    columns of bf16), while the transposed product takes every width: its A
    operand is b.T, whose 128-row CTA slabs stage 256-byte rows whatever the
    width, and its B operand a.T, K-major. bf16 2048x2560x2560 with b stored
    [K, N]:

        for tile_y, tile_x in hl.tile([2048, 2560]):  # 8 x 14 = 112 tiles of
            ...                                       # 256x192: two waves
            tmp1 = in1[tile_r, tile_x]  # b: 96-column half rows per CTA
        for tile_x, tile_y in hl.tile([2560, 2048]):  # transposed: 10 x 15 =
            ...                                       # 150 tiles of 256x144
            tmp0 = in0[tile_x, tile_r]  # b.T: 128-row slabs, 256-byte rows

    two waves of 3/4 the work: 1.12. GB300, of cuBLAS in one process: 0.85
    as written (256x192), 1.02 transposed (256x144). A half row of 64-byte
    atoms (64 or 192 columns) also streams B slower than a K-major B
    (``TCGEN05_MN_B_64B_ATOM_STREAM_COST``, measured at 192).
    """
    if b_n_major and (
        min(m, n) < TCGEN05_TWO_CTA_BLOCK_M
        or -(-k // DENSE_ALIGNED_N_SIXTEEN_BIT_BLOCK_K)
        > TCGEN05_TWO_CTA_SIXTEEN_BIT_MAX_K_TILES
        or -(-m // TCGEN05_TWO_CTA_BLOCK_M) * -(-n // TCGEN05_TWO_CTA_BLOCK_N)
        < sm_pairs
    ):
        # The model prices 256-row pair tiles at the aligned widths in both
        # orientations, which these products do not run: an m under 256 runs
        # as written on the 128-row pair or one-CTA tiles
        # (``tcgen05_transposes_skinny_product`` weighs a skinny m); an n
        # under 256 leaves rows of every transposed tile empty; a K loop past
        # the pair's ``TCGEN05_TWO_CTA_SIXTEEN_BIT_MAX_K_TILES`` K blocks of 64
        # is no aligned-N plan at that depth (``dense_aligned_n_plan``) and no
        # pair at all; and a grid of 256x256 tiles under one per SM pair
        # (the one-wave regime ``pair128_tilings`` below prices for a K-major
        # b) runs one tile per CTA, which a short K loop leaves to the fixed
        # prologue and epilogue, while the as-written search runs narrower
        # pairs over two waves. GB300, bf16 with b stored [K, N], both
        # orientations autotuned at full effort in one process, as written /
        # transposed of stock Inductor: 128x16384x4096 (modeled 1.47) 1.162 /
        # 1.153, 128x9216x4096 (1.18) 1.071 / 1.068; 25216x192x768 1.002 /
        # 0.964, 25216x192x576 0.987 / 0.936; 4096x1024x512 (1.09: 64 tiles
        # of 256x256; as written the search runs 128 of 256x128) 1.104-1.114
        # / 1.064-1.074. Deeper K loops follow the model, the pair running
        # them both ways: 4096x4096x16384 (1.12, 256 K blocks) 0.998 / 1.102,
        # 16384x768x30528 (1.14) 0.988 / 1.126, 32768x512x30528 (1.14) 1.001
        # / 1.123, 8192x1024x29056 (1.14) 1.001 / 1.097, 32768x768x30528
        # (1.05) 1.110 / 1.123; over a K tail on the clipped widths (below),
        # 4096x2048x50264 (1.14) 0.98-1.01 / 1.10-1.11, 16384x768x50272
        # (1.14) 1.01-1.03 / 1.10-1.13, 8192x768x50272 (1.06) 0.96-1.05 /
        # 1.00-1.06, and 4096x1536x128104 (1.06) 1.05-1.07 / 1.03-1.05, the
        # miss: as written its search splits K three ways, which the model
        # does not price. These guards cover an N-major b only:
        # the K-major rule was measured on its own products (it transposes
        # the column-major-b twin of 4096x1024x512: 1.112 as written,
        # 1.056-1.064 transposed).
        return PairTileOrientation(1.0)
    whole_k = k % DENSE_ALIGNED_N_SIXTEEN_BIT_BLOCK_K == 0
    if -(-k // DENSE_ALIGNED_N_SIXTEEN_BIT_BLOCK_K) <= TCGEN05_PAIR_TILE_FIXED_KBLOCKS:
        # A K loop no longer than the fixed prologue and epilogue charged
        # per tile ranks the orientations by that charge alone, which the
        # persistent pair overlaps with its neighbouring tiles. bf16
        # 32768x256x256 (four K blocks), modeled 1.06 for the transposed
        # product: GB300, the same 128x256x64 CTA pair in one process, of
        # cuBLAS, 1.04 as written and 0.98 transposed (the full-effort
        # transposed picks 0.97-1.00 over four runs).
        return PairTileOrientation(1.0)
    if transposed_double_edge_without_pairs(m, n):
        # The model prices CTA pairs, which the search does not run on a
        # transposed double edge whose M 64-row tiles divide
        # (``transposed_double_edge_without_pairs``): bf16 264x2496x1024,
        # modeled 1.03, runs the pairs as written and only one-CTA tiles
        # transposed (2496 = 39 x 64 rows).
        return PairTileOrientation(1.0)
    widths = (64, 128, TCGEN05_TWO_CTA_BLOCK_N, *ALIGNED_N_CHOICES)
    pair128_m = TCGEN05_TWO_CTA_BLOCK_M // 2

    def stream_cost(width: int, n_major_b: bool) -> float:
        # (an N-major B half row of 64-byte atoms: 64 or 192 bf16 columns)
        if n_major_b and dense_aligned_n_mn_b_atom_bytes(width, 2) == 64:
            return TCGEN05_MN_B_64B_ATOM_STREAM_COST
        return 1.0

    def pair128_tilings(
        rows: int, cols: int, *, output_column_major: bool, n_major_b: bool
    ) -> list[tuple[int, int]]:
        # The 128-row pairs (each CTA 64 rows) on a grid the 256-row pairs
        # leave short of one pair per SM pair and the 128-row tiles fill to
        # at least a quarter of the SMs (``search_plan``'s gate) within one
        # wave:
        # - the one-wave family's (``repair.plain_pair128_widths``): a ragged
        #   M under whole N tiles of its widths. bf16 512x1000x1280: as
        #   written no pair tiles its N (1000 = 15.6 x 64; the flat one-CTA
        #   kernel runs 0.86 of cuBLAS), transposed 64 pairs of 128x64 do
        #   (0.96 with two A/B-load warps);
        # - the aligned-N family's, where it runs the 128-row pair over the
        #   output (``dense_aligned_n_block_ms``): an N no 64-column tile
        #   divides, under an aligned width that clips only its last tile
        #   (cuBLAS's swapped tile), on the whole 256-row M tiles whose
        #   single N edge the one-wave search takes (``search_plan``: its
        #   M ceiling divides such an M, and its quarter-wave gate counts
        #   256x64 tiles). bf16 1000x1024x2048: as written 64 pairs of
        #   128x128 (each CTA streams 64 + 64 operand rows per K step),
        #   transposed 72 pairs of 128x112 (64 + 56), 0.93 -> 0.96 of
        #   cuBLAS;
        # GB300, in one process.
        if not whole_k or rows < pair128_m:
            # (an M under one 128-row tile is no M tile the search reaches)
            return []
        if (
            -(-rows // TCGEN05_TWO_CTA_BLOCK_M) * -(-cols // TCGEN05_TWO_CTA_BLOCK_N)
            >= sm_pairs
        ):
            return []
        pair_widths = [
            width
            for width in TCGEN05_TWO_CTA_PAIR128_ONE_WAVE_BLOCK_NS
            if rows % pair128_m and cols % width == 0
        ]
        if (
            cols % 64
            and rows % TCGEN05_TWO_CTA_BLOCK_M == 0
            and -(-rows // TCGEN05_TWO_CTA_BLOCK_M) * -(-cols // 64) >= sm_pairs // 2
            and pair128_m
            in dense_aligned_n_block_ms(
                torch.bfloat16,
                n=cols,
                output_column_major=output_column_major,
                plain_epilogue=True,
            )
        ):
            pair_widths.extend(width for width in ALIGNED_N_CHOICES if width <= cols)
        return [
            (width, pair128_m)
            for width in pair_widths
            if sm_pairs // 2 <= -(-rows // pair128_m) * -(-cols // width) <= sm_pairs
            and (not n_major_b or dense_aligned_n_mn_b_width(width, 2))
        ]

    def best(
        rows: int, cols: int, *, output_column_major: bool, n_major_b: bool
    ) -> tuple[float, bool]:
        """The orientation's least cost, and whether the warm-up runs on a
        256-row pair grid of that cost: the plain pair the search runs either
        way round, unlike the 128-row pairs priced above (bf16 320x1600x384
        transposed admits none)."""
        if rows % TCGEN05_TWO_CTA_BLOCK_M and not whole_k:
            return math.inf, False
        tilings = [
            *(
                (width, TCGEN05_TWO_CTA_BLOCK_M)
                for width in widths
                if not n_major_b or dense_aligned_n_mn_b_width(width, 2)
            ),
            *pair128_tilings(
                rows,
                cols,
                output_column_major=output_column_major,
                n_major_b=n_major_b,
            ),
        ]
        if not tilings:
            return math.inf, False
        costs = [
            pair_tile_wave_cost(
                rows,
                cols,
                k,
                width=width,
                block_m=block_m,
                sm_pairs=sm_pairs,
                stream_cost=stream_cost(width, n_major_b),
            )
            for width, block_m in tilings
        ]
        cost = min(costs)

        def warms(width: int, block_m: int) -> bool:
            tiles = -(-rows // block_m) * -(-cols // width)
            return bool(
                tcgen05_pdl_l2_warmup_kblocks(
                    m=rows,
                    n=cols,
                    k=k,
                    bk=DENSE_ALIGNED_N_SIXTEEN_BIT_BLOCK_K,
                    block_m=block_m,
                    block_n=width,
                    itemsize=2,
                    work_ctas=2 * tiles,
                    num_sm=2 * sm_pairs,
                    one_tile_per_cta=tiles <= sm_pairs,
                )
            )

        return cost, any(
            warms(width, block_m)
            for (width, block_m), tiling_cost in zip(tilings, costs, strict=True)
            if tiling_cost == cost and block_m == TCGEN05_TWO_CTA_BLOCK_M
        )

    # (the transposed product stores out.T, a column-major output; its B
    # operand is a.T, K-major)
    as_written, written_warms = best(
        m, n, output_column_major=False, n_major_b=b_n_major
    )
    transposed, transposed_warms = best(n, m, output_column_major=True, n_major_b=False)
    if math.inf in (as_written, transposed):
        # an orientation the pair cannot tile (a partial last 256-row tile
        # over a K tail) runs other tiles, which the model does not price
        return PairTileOrientation(1.0)
    if b_n_major:
        # Only K-major operands warm (``cute_mma._tcgen05_pdl_l2_warmup_kblocks``):
        # as written b is an N-major B, and transposed b.T an M-major A.
        written_warms = transposed_warms = False
    return PairTileOrientation(as_written / transposed, written_warms, transposed_warms)


def n_major_b_one_wave_pair128_gain(
    m: int, n: int, k: int, *, sm_pairs: int
) -> float | None:
    """The as-written / transposed ``pair_tile_wave_cost`` of a 16-bit ``m``
    x ``n`` x ``k`` GEMM over a row-major ``a`` and an N-major ``b`` on the
    128-row CTA pairs (each CTA 64 rows of the tile) within one wave, each
    orientation at its best width: as written only the widths whose half
    rows of b hold 64-byte swizzle atoms or wider
    (``dense_aligned_n_mn_b_width``), transposed every 16-column multiple
    (its B operand is a.T, K-major) its search runs one-wave 128-row pairs
    at; None where either orientation runs no such grid. A width wider than
    its side (zero-filled columns) is left out, and so is an aligned width
    over a partial last 128-row tile with a K tail (``dense_aligned_n_plan``).
    bf16 512x1280x1000: as written 4 x 10 = 40 pairs of 128x128, each CTA
    streaming 64 + 64 operand rows per K step, transposed 10 x 7 = 70 of
    128x80 (64 + 40): 1.13.

    The transposed side, which the gain bets on, takes only the widths of its
    search's 128-row pairs (``repair.plain_pair128_widths``): the 128-wide
    pair, the 64-wide one over whole 64-column tiles
    (``tcgen05_edge_pair128_narrow_width_fits``), and an aligned width over
    whole 256-row tiles (the gate of the one-wave seeds' CTA pairs,
    ``seeds._one_wave_seed_configs``). Priced at every width, transposed
    264x1608x1000 read 1.18 on 128x64 pairs over its 264 columns and
    768x1152x1000 1.08 on 128x96 pairs over 1152 rows, neither of which its
    search runs: 128x128 pairs either way, 1.00 (GB300, both orientations
    autotuned: 768x1152x1000 0.831 as written, 0.826 transposed). The
    as-written side keeps every N-major width: a width its search cannot
    run only lowers the gain.

    The pricing alone does not rank the orientations: over sector-aligned
    rows of a the transposed search keeps the as-written pairs' grid and
    loses 2% (1024x768x1024, priced 1.08: 0.968 -> 0.952 of cuBLAS; GB300,
    both orientations autotuned).
    ``search_plan.tcgen05_transposes_misaligned_a_rows`` asks it
    for products whose rows of a straddle L2 sectors, which cost the
    as-written pairs more."""
    widths = (64, 128, TCGEN05_TWO_CTA_BLOCK_N, *ALIGNED_N_CHOICES)
    pair128_m = TCGEN05_TWO_CTA_BLOCK_M // 2
    whole_k = k % DENSE_ALIGNED_N_SIXTEEN_BIT_BLOCK_K == 0

    def best(rows: int, cols: int, *, n_major_b: bool) -> float:
        costs = []
        for width in widths:
            if width > cols or (n_major_b and not dense_aligned_n_mn_b_width(width, 2)):
                continue
            if width in ALIGNED_N_CHOICES and rows % pair128_m and not whole_k:
                continue
            if not n_major_b and (
                width == TCGEN05_TWO_CTA_BLOCK_N
                or (width == 64 and cols % 64)
                or (width in ALIGNED_N_CHOICES and rows % TCGEN05_TWO_CTA_BLOCK_M)
            ):
                # (the transposed side at its search's 128-row pair widths)
                continue
            if -(-rows // pair128_m) * -(-cols // width) > sm_pairs:
                continue
            stream_cost = (
                TCGEN05_MN_B_64B_ATOM_STREAM_COST
                if n_major_b and dense_aligned_n_mn_b_atom_bytes(width, 2) == 64
                else 1.0
            )
            costs.append(
                pair_tile_wave_cost(
                    rows,
                    cols,
                    k,
                    width=width,
                    block_m=pair128_m,
                    sm_pairs=sm_pairs,
                    stream_cost=stream_cost,
                )
            )
        return min(costs, default=math.inf)

    as_written = best(m, n, n_major_b=True)
    transposed = best(n, m, n_major_b=False)
    if math.inf in (as_written, transposed):
        return None
    return as_written / transposed


def transposed_pair_tile_gain(
    m: int, n: int, k: int, *, sm_pairs: int, b_n_major: bool = False
) -> float:
    """How much faster the 16-bit pair runs an ``m`` x ``n`` x ``k`` GEMM as
    its transpose: ``transposed_pair_tile_orientation``'s gain."""
    return transposed_pair_tile_orientation(
        m, n, k, sm_pairs=sm_pairs, b_n_major=b_n_major
    ).gain


@dataclass(frozen=True)
class DenseAlignedNPlan:
    block_ids: tuple[int, int, int]
    public_tile: tuple[int, int, int]
    physical_tile: tuple[int, int, int]
    max_ab_stages: int
    # The C-ring subtile width the configured AB ring was priced beside
    # (None: CuTe's own with-source tile); codegen takes the odd-width
    # (128, 48) subtile only where it was (``cute_mma``'s
    # ``tcgen05_odd_width_subtile_taken``).
    store_subtile_n: int | None = None


def dense_aligned_n_plan(
    facts: DenseGemmFacts | None,
    config: Mapping[str, object],
    tile: tuple[int, int, int],
    *,
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    sm_pairs: int,
) -> DenseAlignedNPlan | None:
    """Reuse the strict ordinary native multiplicative-epilogue facts.

    DenseGemmFacts describes the native graph in the frontend's coordinates;
    this family leaves every auxiliary expression and coordinate view unchanged.
    """
    width = config.get(TILE_N_KEY, 0)
    block_ms = (
        ()
        if facts is None
        else dense_aligned_n_block_ms(
            facts.input_dtype,
            n=facts.shape[1],
            output_column_major=facts.output_column_major,
            plain_epilogue=facts.plain_epilogue,
        )
    )
    if (
        facts is None
        or not block_ms
        or facts.capacity_bytes <= 0
        or type(width) is not int
        or width not in ALIGNED_N_CHOICES
        or any(type(value) is not int for value in tile)
    ):
        return None
    bm, _bn, bk = tile
    if bm not in block_ms:
        return None
    block_m = bm
    public = (block_m, carrier_n(width), bk)
    physical = (block_m, width, bk)
    m, n, k = facts.shape
    c_stages = config.get("tcgen05_c_stages", 2)
    acc_stages = config.get("tcgen05_acc_stages", 2)
    store_mode = config.get("tcgen05_c_store_mode", "normal")
    sixteen_bit = dense_aligned_n_sixteen_bit(facts.input_dtype)
    if (
        # Over a K-major B the 16-bit pair's physical width may stage an odd
        # count of 8-row swizzle atoms per CTA (144 -> 72 B rows: the TMA box
        # is the stage-0 ``select`` of the SMEM layout,
        # ``tcgen05_smem_stage_expr``); an N-major B keeps whole 16-row groups
        # (a multiple of 32). The output may be row- or column-major: the D
        # ring stages the tile in the destination's order either way (the
        # direct store, which skips the ring, row-major only).
        (sixteen_bit and not facts.rhs_k_major and width % 32)
        or (sixteen_bit and facts.output_column_major and store_mode == "direct")
        or bk not in (64, 128)
        or tile not in (public, physical)
        or any(
            not low <= value <= high
            for candidate in (public, physical)
            for value, (low, high) in zip(candidate, bounds, strict=True)
        )
        # The 16-bit pair clips a partial last M or N tile in its TMA store
        # (its loads zero-fill past the edge). A partial last N tile runs a K
        # tail too: one N tile per cluster (below), its producer issuing the
        # clipped N tile and the partial last K tile, both boxes zero-filled
        # past the edge (``cute_mma``'s single-edge K-tail family). The
        # transposed bf16 4096x2048x50264 with b stored [K, N] (2048 x 4096
        # on 256x224 pairs, 19 N tiles, K = 785 x 64 + 24):
        #
        #     tcgen05_tma_next_full_tile = tile_offset_0 + cutlass.Int32(256) <= cutlass.Int32(2048) and tile_offset_1 < cutlass.Int32(4096) and (tile_offset_2 + cutlass.Int32(448) < cutlass.Int32(50264))
        #
        # A partial last M tile keeps whole K tiles.
        or (m % bm and not (sixteen_bit and k % bk == 0))
        or (n % width and not sixteen_bit)
        or k < bk
        or (k + bk - 1) // bk > tcgen05_two_cta_max_k_tiles(facts.input_dtype.itemsize)
        or type(c_stages) is not int
        or c_stages not in (2, 4)
        or type(acc_stages) is not int
        or acc_stages not in (1, 2)
        or store_mode not in ("normal", "direct")
        or config.get("pid_type")
        not in ("persistent_blocked", "persistent_interleaved")
        or config.get("tcgen05_grouped_mode") is not None
        or config.get("cute_split_k_workspace")
    ):
        return None
    # Values the family pins; the 16-bit pair keeps the persistent raster's L2
    # grouping and may pair N tiles (A multicast) where they split evenly.
    pinned = dense_aligned_n_defaults(
        acc_stages=acc_stages, store_mode=cast("str", store_mode)
    )
    # The family's producer is the role-local TMA producer, which one or two
    # A/B-load warps run (``cute_mma._build_shared_ring_producer``).
    pinned.pop("tcgen05_warp_spec_ab_load_warps")
    if sixteen_bit:
        pinned.pop("l2_groupings")
        # N pairing (A multicast) takes whole, evenly split N tiles: the
        # N-edge cluster schedule covers the power-of-two widths only. A
        # column-major output keeps one N tile per cluster, as the
        # power-of-two pairs do (``seeds._paired_pipeline_parameters``).
        if (
            config.get("tcgen05_cluster_n") == 2
            and not facts.output_column_major
            and n % width == 0
            and (n // width) % 2 == 0
        ):
            pinned.pop("tcgen05_cluster_n")
    if any(config.get(key, value) != value for key, value in pinned.items()):
        return None
    # The physical tile is a paired instruction: M128 FP8 with a 64-row,
    # N-permuted per-CTA BF16 D ring, or 16-bit M256 with 128 rows per CTA
    # (M128 with 64: ``dense_aligned_n_block_ms``).
    # Its allocations are exactly those of the native paired pipeline at this
    # width and stored dtype, with the configured epilogue subtile and warp
    # groups; a plain 16-bit store on whole K tiles at a width ``cute_mma``'s
    # plain narrow rule covers takes the plan's (128, 32) subtile over whole
    # N tiles, and over a partial last N tile where
    # ``tcgen05_plain_narrow_subtile_covers_n`` admits one (the TMA store
    # clips a partial last M or N tile), a plain 16-bit store at 144 or 240
    # columns the (128, 48) one where its ring fits beside the configured AB
    # ring (below), any other C stage (an FP32 output, a partial last N tile
    # that rule leaves out, a column-major output) the model's own
    # with-source tile.
    allocations = Tcgen05PipelineSmemFacts(
        facts.input_dtype.itemsize,
        facts.output_dtype.itemsize,
        facts.capacity_bytes,
        output_column_major=facts.output_column_major,
    )
    epi_warp_groups = config.get(TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY, 1)
    configured_subtile_n = config.get(TCGEN05_EPI_SUBTILE_N_CONFIG_KEY)
    if type(epi_warp_groups) is not int or epi_warp_groups < 1:
        return None
    # The 16-bit pair's configured subtile partitions the physical width, not
    # the carrier (the FP8 M128 epilogue ignores the knob); a non-positive one
    # is left to the epilogue validation.
    if (
        not sixteen_bit
        or type(configured_subtile_n) is not int
        or configured_subtile_n <= 0
    ):
        configured_subtile_n = None
    elif width % configured_subtile_n or block_m != TCGEN05_TWO_CTA_BLOCK_M:
        # (the 128-row pair stores CuTe's with-source tile only: an explicit
        # subtile on its 64-row C tile fails codegen with "requires the
        # 256-row CtaGroup.TWO pair tile")
        return None
    # Mirrors ``cute_mma``'s ``tcgen05_plain_narrow_subtile`` decision (and
    # ``_paired_pipeline_store_subtile_n`` for power-of-two pairs): pricing a
    # (128, 32) ring that codegen then renders at (128, 64) would overflow
    # the seven-stage ring (``test_cute_dense_aligned_n`` renders the
    # 192-wide plain pair, whole and over an M or N tail, at the priced
    # subtile). Over an N tail the (128, 64) ring would leave room for six
    # stages only: bf16 1000x10240x2560 on the 192-wide pair (53.3 N tiles),
    # GB300, of cuBLAS in one process, 0.985-0.989 at six AB stages and
    # 1.001-1.009 at seven.
    # Those rules (and the odd-width one below) retile the row-major store
    # only; a column-major output keeps CuTe's with-source tile, which the
    # model prices itself:
    #
    #     tcgen05_store_epi_tile = compute_epilogue_tile_shape(
    #         (256, 224), False, LayoutEnum.COL_MAJOR, cutlass.BFloat16, ...)
    #     # (128, 32): a 128-row box, rows contiguous in the destination
    plain_sixteen_bit_store = (
        sixteen_bit
        and facts.output_dtype in (torch.float16, torch.bfloat16)
        and facts.plain_epilogue
        and not facts.output_column_major
        and store_mode == "normal"
        # the narrow and odd-width subtiles retile the 256-row pair's 128-row
        # C tile; the 128-row pair keeps (and is priced at) CuTe's own
        and block_m == TCGEN05_TWO_CTA_BLOCK_M
    )
    store_subtile_n = (
        configured_subtile_n
        if configured_subtile_n is not None
        else TCGEN05_PLAIN_NARROW_SUBTILE_N
        if plain_sixteen_bit_store
        and width in TCGEN05_PLAIN_NARROW_SUBTILE_BLOCK_NS
        and k % bk == 0
        and tcgen05_plain_narrow_subtile_covers_n(
            m, n, k, bm=block_m, bn=width, bk=bk, sm_pairs=sm_pairs
        )
        else None
    )

    # A pair that runs one tile fills at most its K loop's stages
    # (``repair.one_tile_ab_stage_cap``); a persistent walk over more tiles
    # than pairs keeps a deeper ring, which its producer fills with the next
    # tile's k-blocks while the current tile's MMAs and epilogue drain. bf16
    # 131072x144x288 (512 tiles of 256x144 on 76 pairs, five k-blocks a tile)
    # at six stages:
    #
    #     tcgen05_ab_pipeline = _helion_tc_pro.make_tma_umma_pipeline(num_stages=6, ...)
    #     ...
    #     # (the next tile's first k-block takes the sixth slot while the
    #     # MMA warp still holds the current tile's five)
    #     tcgen05_ab_pipeline.producer_acquire(tcgen05_ab_producer_state)
    #
    # GB300, the same pair in one process, of cuBLAS at the K loop's depth /
    # past it: 131072x144x288 0.98 / 1.01-1.02 (6 stages; 8 stages 0.99-1.00),
    # 32768x192x384 1.09 / 1.12 (7), 32768x192x192 1.02 / 1.15 (5-7).
    max_stages = (
        paired_pipeline_max_ab_stages(allocations)
        if facts.plain_epilogue
        else TCGEN05_SMEM_AWARE_MAX_AB_STAGES
    )
    if -(-m // block_m) * -(-n // width) <= sm_pairs:
        max_stages = min(max_stages, (k + bk - 1) // bk)
    # A plain 16-bit pair allocates the AB and C rings and nothing else, so
    # its arena is priced as ptxas lays it out (``plain_layout``, the
    # measured ``tcgen05_fixed_smem_overhead_bytes``): the 224-wide pair's
    # seventh AB stage beside two (128, 32) C stages takes 231 596 of the
    # 232 448 bytes, which the flat reserve (233 616) refuses. GB300, bf16
    # RC, the same kernel at six and seven stages in one process (median of
    # 15 per-round cuBLAS/Helion ratios), the transposed product's
    # column-major 256x224 tiles: 32768x768x768 0.981 -> 1.022,
    # 32768x512x512 0.980 -> 1.009.
    ring_bytes = functools.partial(
        pipeline_smem_bytes,
        allocations,
        bm=block_m,
        bn=width,
        bk=bk,
        c_stages=c_stages,
        acc_stages=acc_stages,
        epi_warp_groups=epi_warp_groups,
        plain_layout=sixteen_bit and facts.plain_epilogue,
    )
    maximum = deepest_fitting_ab_stages(
        lambda stages: ring_bytes(ab_stages=stages, epi_subtile_n=store_subtile_n),
        facts.capacity_bytes,
        max_stages=max_stages,
    )
    if maximum == 0:
        return None
    # A plain 16-bit pair of a width 48 divides and 32 does not (144, 240)
    # stores (128, 48) subtiles where that C ring fits beside the configured
    # AB ring (``cute_mma``'s ``tcgen05_odd_width_subtile_taken``); a deeper
    # ring keeps CuTe's (128, 16), so no AB depth is lost (144 wide: two C
    # stages fit beside eight AB stages either way, four beside seven only).
    if (
        store_subtile_n is None
        and plain_sixteen_bit_store
        and tcgen05_odd_width_subtile(width)
    ):
        odd_width_stages = deepest_fitting_ab_stages(
            lambda stages: ring_bytes(
                ab_stages=stages, epi_subtile_n=TCGEN05_ODD_WIDTH_SUBTILE_N
            ),
            facts.capacity_bytes,
            max_stages=max_stages,
        )
        ab_stages = config.get("tcgen05_ab_stages")
        if type(ab_stages) is int and 0 < ab_stages <= odd_width_stages:
            store_subtile_n = TCGEN05_ODD_WIDTH_SUBTILE_N
    return DenseAlignedNPlan(
        facts.block_ids, public, physical, maximum, store_subtile_n
    )
