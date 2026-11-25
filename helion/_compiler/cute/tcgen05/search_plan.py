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
from ..cute_warp_mma_gemm import WARP_MMA_MIN_BM
from ..cute_warp_mma_gemm import WARP_MMA_MIN_BN
from ..tcgen05_config import facts
from ..tcgen05_config import surface
from ..tcgen05_constants import RAGGED_BATCHED_MIN_BLOCK_N
from ..tcgen05_constants import TCGEN05_MANY_SMALL_BATCHES
from ..tcgen05_constants import TCGEN05_ONE_CTA_MAX_BLOCK_M
from ..tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from ..tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_MIN_DIM
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_PAIR128_BLOCK
from ..tcgen05_constants import TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
from ..tcgen05_constants import TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N
from ..tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES
from ..tcgen05_constants import TCGEN05_TWO_CTA_ONE_WAVE_BLOCK_NS
from ..tcgen05_constants import TCGEN05_TWO_CTA_PAIR128_ONE_WAVE_BLOCK_NS
from ..tcgen05_constants import TCGEN05_WHOLE_K_TILE_MIN

if TYPE_CHECKING:
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
    ``TCGEN05_SMALL_N_MIN_K_BYTES`` deep."""
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
    if small_n_requires_role_local_tma and (
        static_m is None
        or static_k is None
        or static_m % max_search_m != 0
        or static_k % max_search_k != 0
    ):
        return reject()
    from ..cute_mma import _tcgen05_tma_matrix_major
    from ..cute_warp_mma_gemm import warp_mma_admission_reason

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
) -> None:
    """Apply one preflighted tcgen05 search plan to the shared config.

    ``output_dtype_bytes`` / ``output_stores`` size the TMA-store rings the
    AB-stage fit estimate charges (``None``: AB-only estimate);
    ``plain_epilogue``: the epilogue stores the tile once and reads no other
    tensor (the 128-row pairs' plain GEMMs).
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
            and min(plan.max_tcgen05_m, plan.max_tcgen05_n)
            >= TCGEN05_TWO_CTA_EDGE_PAIR128_BLOCK
            and min(static_m, static_n, static_k) >= TCGEN05_TWO_CTA_EDGE_PAIR128_BLOCK
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
        ) and (not m_edge or max_search_m >= TCGEN05_TWO_CTA_BLOCK_M)
        whole_k_tile = static_k & -static_k
        if (
            input_dtype.itemsize == 2
            and static_k % max_search_k != 0
            and not edge_k_tail_shape
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
    max_cluster_m2_search_k = TCGEN05_TWO_CTA_MAX_K_TILES * max_search_k
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
    spec._cute_tcgen05_config.output_edge_persistent_admitted = (
        output_edge_shape and allow_whole_k_persistent_pid_types
    )
    spec._cute_tcgen05_config.tcgen05_min_search_n = plan.min_search_n
    spec._cute_tcgen05_config.matmul_leading_block_id = leading_block_id
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
                edge_work_clusters = (
                    plan.leading_work_multiplier
                    * -(-static_m // TCGEN05_TWO_CTA_EDGE_PAIR128_BLOCK)
                    * -(-static_n // TCGEN05_TWO_CTA_EDGE_PAIR128_BLOCK)
                )
                work_clusters = max(work_clusters, edge_work_clusters)
            if work_clusters < num_sms // 4:
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
        ),
        enable_cluster_m2_search=allow_cluster_m2_search,
        cluster_m2_static_k=static_k if allow_cluster_m2_search else None,
        allow_cluster_m2_edge_k_tail_family=allow_edge_cluster_m2_search,
        allow_cluster_m2_fp8_small_grid=allow_fp8_small_grid_cluster_m2_search,
        allow_cluster_m2_one_wave_tiles=allow_one_wave_cluster_m2_search,
        cluster_m2_one_wave_only=one_wave_only_cluster_m2_search,
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
