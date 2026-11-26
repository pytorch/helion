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
from .tcgen05_constants import TCGEN05_EPI_SUBTILE_N_CONFIG_KEY
from .tcgen05_constants import TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY
from .tcgen05_constants import TCGEN05_ODD_WIDTH_SUBTILE_N
from .tcgen05_constants import TCGEN05_ONE_CTA_MAX_BLOCK_M
from .tcgen05_constants import TCGEN05_PAIR_TILE_FIXED_KBLOCKS
from .tcgen05_constants import TCGEN05_PAIR_TILE_STREAM_RATIO
from .tcgen05_constants import TCGEN05_PLAIN_NARROW_SUBTILE_BLOCK_NS
from .tcgen05_constants import TCGEN05_PLAIN_NARROW_SUBTILE_N
from .tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from .tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from .tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES
from .tcgen05_constants import TCGEN05_TWO_CTA_PAIR128_ONE_WAVE_BLOCK_NS
from .tcgen05_constants import tcgen05_odd_width_subtile
from .tcgen05_constants import tcgen05_plain_narrow_subtile_covers_n

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
    m: int, n: int, k: int, *, width: int, block_m: int, sm_pairs: int
) -> float:
    """Wave-quantized time of a 16-bit pair-tile grid over ``m`` x ``n``
    (arbitrary units: 256-column K-block MMAs). Each of the
    ``ceil(m / block_m) x ceil(n / width)`` tiles takes ``ceil(k / 64)`` K
    blocks, each the larger of its MMA time (its width, halved on the
    128-row pair's M128 instruction) and its operand stream
    (``TCGEN05_PAIR_TILE_STREAM_RATIO`` x (block_m + width): a narrow tile
    reloads the A tile per N tile), plus a fixed prologue and epilogue
    (``TCGEN05_PAIR_TILE_FIXED_KBLOCKS`` 256-wide K blocks); ``sm_pairs`` tiles
    run per wave."""
    tiles = -(-m // block_m) * -(-n // width)
    k_blocks = -(-k // DENSE_ALIGNED_N_SIXTEEN_BIT_BLOCK_K)
    k_block = max(
        width * block_m / TCGEN05_TWO_CTA_BLOCK_M,
        TCGEN05_PAIR_TILE_STREAM_RATIO * (block_m + width),
    )
    fixed = TCGEN05_PAIR_TILE_FIXED_KBLOCKS * TCGEN05_TWO_CTA_BLOCK_N
    return -(-tiles // sm_pairs) * (k_blocks * k_block + fixed)


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


def transposed_pair_tile_gain(m: int, n: int, k: int, *, sm_pairs: int) -> float:
    """How much faster the 16-bit pair runs an ``m`` x ``n`` x ``k`` GEMM as
    its transpose (``n`` x ``m`` x ``k``, out.T = b.T @ a.T): the ratio of
    the two orientations' best ``pair_tile_wave_cost`` over the pair's
    widths (the power-of-two ones and ``ALIGNED_N_CHOICES``; a width with a
    partial last tile, or a partial last 256-row tile, needs whole K tiles).

    The pair tiles its M side in 256-row steps and its N side in 16-column
    ones, so a tall output tiles finer transposed. bf16 16384x1024x4096 on
    76 SM pairs: 64 x 4 = 256 tiles of 256x256 (four waves) against 4 x 74
    = 296 tiles of 256x224 transposed (four waves of 7/8 the work): 1.13
    (measured: ``HelionChoices.transpose_matmul``).
    """
    whole_k = k % DENSE_ALIGNED_N_SIXTEEN_BIT_BLOCK_K == 0
    if -(-k // DENSE_ALIGNED_N_SIXTEEN_BIT_BLOCK_K) <= TCGEN05_PAIR_TILE_FIXED_KBLOCKS:
        # A K loop no longer than the fixed prologue and epilogue charged
        # per tile ranks the orientations by that charge alone, which the
        # persistent pair overlaps with its neighbouring tiles. bf16
        # 32768x256x256 (four K blocks), modeled 1.06 for the transposed
        # product: GB300, the same 128x256x64 CTA pair in one process, of
        # cuBLAS, 1.04 as written and 0.98 transposed (the full-effort
        # transposed picks 0.97-1.00 over four runs).
        return 1.0
    if transposed_double_edge_without_pairs(m, n):
        # The model prices CTA pairs, which the search does not run on a
        # transposed double edge whose M 64-row tiles divide
        # (``transposed_double_edge_without_pairs``): bf16 264x2496x1024,
        # modeled 1.03, runs the pairs as written and only one-CTA tiles
        # transposed (2496 = 39 x 64 rows).
        return 1.0
    widths = (64, 128, TCGEN05_TWO_CTA_BLOCK_N, *ALIGNED_N_CHOICES)
    pair128_m = TCGEN05_TWO_CTA_BLOCK_M // 2

    def pair128_costs(
        rows: int, cols: int, *, output_column_major: bool
    ) -> list[float]:
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
            pair_tile_wave_cost(
                rows, cols, k, width=width, block_m=pair128_m, sm_pairs=sm_pairs
            )
            for width in pair_widths
            if sm_pairs // 2 <= -(-rows // pair128_m) * -(-cols // width) <= sm_pairs
        ]

    def best(rows: int, cols: int, *, output_column_major: bool) -> float:
        if rows % TCGEN05_TWO_CTA_BLOCK_M and not whole_k:
            return math.inf
        return min(
            (
                *(
                    pair_tile_wave_cost(
                        rows,
                        cols,
                        k,
                        width=width,
                        block_m=TCGEN05_TWO_CTA_BLOCK_M,
                        sm_pairs=sm_pairs,
                    )
                    for width in widths
                    if cols % width == 0 or whole_k
                ),
                *pair128_costs(rows, cols, output_column_major=output_column_major),
            ),
            default=math.inf,
        )

    # (the transposed product stores out.T, a column-major output)
    as_written = best(m, n, output_column_major=False)
    transposed = best(n, m, output_column_major=True)
    if math.inf in (as_written, transposed):
        # an orientation the pair cannot tile (a partial last 256-row tile
        # over a K tail) runs other tiles, which the model does not price
        return 1.0
    return as_written / transposed


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
        # (its loads zero-fill past the edge); the edge schedule needs whole
        # K tiles.
        or ((m % bm or n % width) and not (sixteen_bit and k % bk == 0))
        or k < bk
        or (k + bk - 1) // bk > TCGEN05_TWO_CTA_MAX_K_TILES
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

    max_stages = min(
        paired_pipeline_max_ab_stages(allocations)
        if facts.plain_epilogue
        else TCGEN05_SMEM_AWARE_MAX_AB_STAGES,
        (k + bk - 1) // bk,
    )
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
