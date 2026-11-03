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
from ...compile_environment import _is_supported_tensor_input_source
from ...compile_environment import _symint_free_symbols
from ..cute_warp_mma_gemm import WARP_MMA_MIN_BM
from ..cute_warp_mma_gemm import WARP_MMA_MIN_BN
from ..tcgen05_config import facts
from ..tcgen05_config import surface
from ..tcgen05_constants import RAGGED_BATCHED_MIN_BLOCK_N
from ..tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from ..tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_MIN_DIM
from ..tcgen05_constants import TCGEN05_TWO_CTA_EDGE_PAIR128_BLOCK
from ..tcgen05_constants import TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
from ..tcgen05_constants import TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N
from ..tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES
from ..tcgen05_constants import TCGEN05_TWO_CTA_ONE_WAVE_BLOCK_NS

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
) -> _CuteTcgen05SearchPlanningResult:
    """Plan one candidate and retain guards even when the candidate is rejected.

    ``lhs`` / ``rhs`` are the matmul's operand values (the loaded tiles);
    ``lhs_source`` / ``rhs_source`` are the host tensors they were loaded
    from, whose layouts decide the register-MMA family's admission.
    ``operand_casts`` says an operand is an in-register integer cast only
    the register-MMA family lowers: the candidate is rejected unless that
    family is admitted, and the plan withholds the tcgen05 family.
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
    mma_k = 32 if is_fp8 else (8 if is_tf32 else 16)
    # tf32's bn >= 32 floor mirrors _mma_impl_matches_problem_shape.
    min_tcgen05_n = 16 if is_fp8 else (32 if is_tf32 else 8)
    is_small_n = static_n is not None and static_n < min_tcgen05_n
    if (
        env.backend_name != "cute"
        or (
            lhs.dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
            and not is_tf32
        )
        or rhs.dtype != lhs.dtype
        # Fewer than 64 static rows run as one zero-filled 64-row tile on the
        # flat pipeline (decode GEMMs: bf16 16x14336x4096 27 us against the
        # SIMT fallback's 358 us); a dynamic M keeps the 64-row floor, and a
        # single row stays SIMT (a TMA box over an extent-1 dimension ran
        # 1x4096x4096 at 327 us against 20 us for two rows).
        or (
            m_hint < 64
            if static_m is None
            # Only the memory-bound decode shapes profit from the padded
            # tile; a tiny GEMM (33^3) keeps the SIMT lowering, which wins
            # there (2.8 us against the tile's 13.8).
            else static_m < 2 or (static_m < 64 and (n_hint < 128 or k_hint < 128))
        )
        # Size-one tile axes are fixed to block size one before this analysis,
        # so they cannot carry the minimum tcgen05 instruction tile.
        or (static_n is not None and static_n < 2)
        or (static_n is None and n_hint < min_tcgen05_n)
        or (
            is_small_n
            and not (
                supports_small_n_role_local_tma or supports_small_n_scalar_fallback
            )
        )
        or k_hint < mma_k
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
) -> None:
    """Apply one preflighted tcgen05 search plan to the shared config.

    ``output_dtype_bytes`` / ``output_stores`` size the TMA-store rings the
    AB-stage fit estimate charges (``None``: AB-only estimate).
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
    max_cluster_m2_search_k = TCGEN05_TWO_CTA_MAX_K_TILES * max_search_k
    if static_m is None or static_n is None or static_k is None:
        allow_full_tile_persistent_pid_types = False
        allow_small_n_persistent_pid_types = False
        allow_full_tile_cluster_m2_search = False
        allow_edge_cluster_m2_search = False
        allow_fp8_small_grid_cluster_m2_search = False
        allow_one_wave_cluster_m2_search = False
    else:
        allow_full_tile_persistent_pid_types = (
            static_m % max_search_m == 0
            and static_n % max_search_n == 0
            and static_k % max_search_k == 0
        )
        allow_small_n_persistent_pid_types = (
            plan.is_small_n
            and plan.small_n_supports_role_local_tma
            and 0 < static_n < max_search_n
            and static_m % max_search_m == 0
            and static_k % max_search_k == 0
        )
        allow_full_tile_cluster_m2_search = (
            allow_full_tile_persistent_pid_types
            and max_search_m >= TCGEN05_TWO_CTA_BLOCK_M
            and max_search_n >= TCGEN05_TWO_CTA_BLOCK_N
            and static_k <= max_cluster_m2_search_k
        )
        allow_edge_cluster_m2_search = (
            not has_leading_passthrough
            and not allow_full_tile_persistent_pid_types
            and plan.max_tcgen05_m >= TCGEN05_TWO_CTA_BLOCK_M
            and plan.max_tcgen05_n >= TCGEN05_TWO_CTA_BLOCK_N
            and static_m >= TCGEN05_TWO_CTA_EDGE_K_TAIL_MIN_DIM
            and static_n >= TCGEN05_TWO_CTA_EDGE_K_TAIL_MIN_DIM
            and static_k >= TCGEN05_TWO_CTA_EDGE_K_TAIL_MIN_DIM
            and static_k <= max_cluster_m2_search_k
            and static_m % TCGEN05_TWO_CTA_BLOCK_M != 0
            and static_n % TCGEN05_TWO_CTA_BLOCK_N != 0
            and static_k % max_search_k != 0
        )
        allow_fp8_small_grid_cluster_m2_search = (
            not has_leading_passthrough
            and plan.is_fp8
            and allow_full_tile_persistent_pid_types
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
            and static_n % min(TCGEN05_TWO_CTA_ONE_WAVE_BLOCK_NS) == 0
            and plan.leading_work_multiplier
            * (static_m // TCGEN05_TWO_CTA_BLOCK_M)
            * (static_n // TCGEN05_TWO_CTA_BLOCK_N)
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
    # The register-MMA family replaces the whole device body of ONE GEMM (its
    # root tile loop around its K loop; ``detect_warp_mma_gemm`` proves that
    # root holds nothing else); kernels with several matmuls keep the tcgen05
    # surface alone.
    warp_mma_admitted = plan.warp_mma_admitted and warp_mma_plain_kernel
    assert plan.tcgen05_family_admitted or warp_mma_admitted
    spec._cute_tcgen05_config.warp_mma_admitted = warp_mma_admitted
    spec._cute_tcgen05_config.tcgen05_family_admitted = plan.tcgen05_family_admitted
    spec._cute_tcgen05_config.tcgen05_min_search_m = plan.min_search_m
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
            work_clusters = (
                plan.leading_work_multiplier
                * (static_m // cluster_m)
                * (static_n // cluster_n)
            )
            if allow_one_wave_cluster_m2_search:
                one_wave_work_clusters = (
                    plan.leading_work_multiplier
                    * (static_m // TCGEN05_TWO_CTA_BLOCK_M)
                    * (static_n // min(TCGEN05_TWO_CTA_ONE_WAVE_BLOCK_NS))
                )
                one_wave_only_cluster_m2_search = (
                    work_clusters < num_sms // 4 <= one_wave_work_clusters
                )
                work_clusters = max(work_clusters, one_wave_work_clusters)
            if allow_edge_cluster_m2_search and input_dtype.itemsize == 2:
                # The 16-bit edge family also runs 128x128 CTA pairs
                # (``tcgen05_config.repair.plain_edge_pair128_admitted``: each CTA
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
    surface.narrow_autotune_to_validated_configs(
        spec.tcgen05,
        allow_persistent_pid_types=(
            allow_full_tile_persistent_pid_types or allow_small_n_persistent_pid_types
        ),
        enable_cluster_m2_search=allow_cluster_m2_search,
        cluster_m2_static_k=static_k if allow_cluster_m2_search else None,
        allow_cluster_m2_edge_k_tail_family=allow_edge_cluster_m2_search,
        allow_cluster_m2_fp8_small_grid=allow_fp8_small_grid_cluster_m2_search,
        allow_cluster_m2_one_wave_tiles=allow_one_wave_cluster_m2_search,
        cluster_m2_one_wave_only=one_wave_only_cluster_m2_search,
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
