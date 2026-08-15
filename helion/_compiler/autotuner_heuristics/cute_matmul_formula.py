"""Formulaic tcgen05 matmul autotuner-seeding heuristic — the CuTe analog of #3007.

The three shipped CuTe matmul seed producers
(``CuteTcgen05ClusterM2Heuristic``, ``CuteTcgen05ClusterM2FfiHeuristic``,
``CuteFp8GemmSkinnyMHeuristic``) are special-case, ``cluster_m=2``-only templates
that (a) structurally cannot even *propose* whole regimes (M=64 decode / cluster_m=1,
medium-M single-wave rectangular) and (b) couple their ``is_eligible`` to
``enforce_dot_requirements``' *search-restriction* gate — so a shape the wave-quant
gate suppressed from cm2 *search* also got NO seed (the "A1 gap").

``CuteTcgen05FormulaMatmulHeuristic`` replaces that with ONE shape-aware formula:
``f(M, K, N, dtype, epilogue, num_sm, smem_budget) -> Config``. Its genuine surface is
the ~5 knobs the codegen defaults get wrong for a given regime (see
``cute-matmul-heuristic-plan.md`` §4.3.5):

  1. ``tcgen05_cluster_m``   — the regime selector (1 decode / 2 compute+medium-M)
  2. ``block_sizes``         — [bm, bn, bk]: collective tile, wave-fill-shrunk
  3. ``tcgen05_ab_stages``   — depth-fill to the ~196 KB AB-SMEM isobar (dtype-capped)
  4. ``l2_groupings``        — wave-count-aware ([1] many-wave / [4] single-wave)
  5. ``pid_type`` / persistence — persistent_blocked (decode) / _interleaved (compute)

Everything else is INHERITED from the ``bn``/``num_stages``-keyed codegen defaults
(``acc_stages``, ``c_stages``, ``num_epi_warps``, role/warp-spec) — emitted for
completeness because a seed is a full Config, but written to the default value.

Two design invariants proven by the run-3 hill-climbs + the design-validation smoke:

  * **The depth-fill fills to a fixed ~196 608-byte AB-SMEM isobar.** ``bk`` and ``ab``
    trade off along it (the direct-entry stage-tuple table used to encode exactly
    this). So ``_pick_bk_ab`` picks the ``(bk, ab)`` that MAXIMIZES AB-SMEM bytes used
    within the per-CTA budget and the dtype cap (tie -> deepest ab, then larger ``bk``).
    This reproduces the pretuned fp8 decode bn=32/bk=256/ab=8 (196 608) over
    bk=128/ab=12 (147 456) AND the fp8 compute bk=128/ab=6 AND the fp8 decode
    bn=64/bk=128/ab=12. On the BF16 decode tile the tie now resolves to
    bk=128/ab=8, which is measured 4.6% FASTER than the recorded bk=256/ab=4 key
    (see the history note in ``_pick_bk_ab``).
  * **Wave/occupancy counts must be in CTAs, not output tiles** — ``cluster_m=2`` spends
    2 CTAs per output tile, so ``_wave_eff`` / ``l2_groupings`` multiply by ``cluster_m``.

The seed is ORTHOGONAL to ``enforce_dot_requirements``' search restrictions
(``cute-seed-orthogonal-to-search`` memory): ``is_eligible`` reads the ``MatmulFact``
directly and NEVER asks the search-restriction gate for permission. The formula emits
the best config that passes genuine (SMEM-budget / physical) validation; gate-3
"artificial-but-errors" caps (bf16 ab>3) are handled by the bundled prerequisite edits.

Bucket-A fills 16-bit ``ab`` to the dtype cap (8) on the DEFAULT path — the bundled
bf16-deep-AB prerequisite made deep 16-bit AB (e.g. compute ``bk=64/ab=6``) run on the plain
path, so it is NOT restricted to the FFI topology (measured +1–18% over the old ab≤3 cap).
The FFI ``explicit_epi_tile`` config still ships as a SECOND ranked seed
(``CuteTcgen05FormulaFfiAltHeuristic``, Bucket B); the autotuner keeps whichever wins.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Any

import torch

from ...runtime.config import Config
from ..cute.strategies import TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY
from ..cute.strategies import TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY
from ..cute.strategies import TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY
from ..cute.strategies import TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY
from ..cute.strategies import TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY
from ..cute.strategies import TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY
from ..cute.strategies import Tcgen05LayoutStrategy
from ..cute.strategies import Tcgen05PersistenceModel
from ..cute.tcgen05_config import CuteTcgen05Config
from ..cute.tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY
from ..cute.tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from ..cute.tcgen05_constants import TCGEN05_TWO_CTA_SEED_L2_GROUPING
from ..cute.tcgen05_constants import tcgen05_ab_smem_bytes_per_cta
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...autotuner.config_spec import ConfigSpec
    from ...autotuner.config_spec import MatmulFact
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR

# --- device / geometry constants (mirrors tcgen05_constants.py; sm100 / B200) ---
_STATIC_PERSISTENT = Tcgen05PersistenceModel.STATIC_PERSISTENT.value
_DTYPE_AB_CAP = {1: 12, 2: 8, 4: 3}
# Candidate block-K per dtype, largest first (the isobar fill tries all, keeps the best).
# 16-bit includes 256 because the narrow decode tile (bm=64/bn=32) fills the isobar at
# bk=256/ab=4 (196 608 = the R3 #8 bf16-decode key); for the 256² compute tile bk=256
# only reaches ab=1 so bk=128 still wins there — adding 256 is safe.
_DTYPE_BK_CHOICES = {1: (256, 128), 2: (256, 128, 64), 4: (64,)}
_WAVE_FULL = 0.8  # a tile "fills a wave" above this CTA occupancy (compute/medm classifier + FFI)
# Decode (memory-bound) accepts a WIDER bn at LOWER occupancy than the compute classifier does:
# the loop returns the widest bn clearing this bar, and a bandwidth-bound decode prefers the wider
# bar 0.8->0.5 improves 64x3584x3584 (bn64@0.38w 100 -> bn32@0.76w 115, +14%) and 64x1536x1536
_DECODE_WAVE_FULL = 0.5
_MANY_WAVE = 4  # waves >= this -> l2_groupings=[1]
_DECODE_M_MAX = 128  # M <= this is the cluster_m=1 decode regime (ONE_CTA bm cap)
# M < this is below the tcgen05 decode admission floor: matmul_ops.py
# enforce_dot_requirements gates the whole tcgen05 block on static_m>=64, so deep-AB is
# never admitted below it and the decode ab-lever is unavailable.
_DECODE_M_MIN = 64
# M <= this is the skinny-M SIMT regime (owned by CuteFp8GemmSkinnyMHeuristic).
_SKINNY_M_MAX = 16
_DECODE_BM = 64  # decode M-tile (bm=64 in every decode answer key / pretuned row)
_DECODE_BN_CHOICES = (128, 64, 32, 16)  # decode N-tile menu (from the pretuned table)
_FP8_SMALL_GRID = 128  # fp8 small-grid cluster_m=2 tile (bm=bn=128, per-CTA 64x128)

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_SIXTEEN_BIT_DTYPES = (torch.bfloat16, torch.float16)


def _itemsize(dtype: torch.dtype) -> int:
    if dtype in _FP8_DTYPES:
        return 1
    if dtype in _SIXTEEN_BIT_DTYPES:
        return 2
    return dtype.itemsize


def _n_ctas(bm: int, bn: int, m: int, n: int, cluster_m: int) -> int:
    """CTA count = output tiles x cluster_m (cluster_m=2 spends 2 CTAs / tile)."""
    tiles = max(1, math.ceil(m / bm)) * max(1, math.ceil(n / bn))
    return tiles * cluster_m


# ``num_sm <= 0`` is the "SM count UNKNOWN" sentinel from ``_num_sm`` (get_num_sm itself is
# always >= 1; the wrapper returns 0 only when it raises for a non-CUDA/unimplemented device).
# In the current formula this is unreachable — ``_formula_eligible`` gates on
# ``per_cta_ab_smem_budget_bytes(device) > 0``, which is 0 for non-CUDA, so these helpers only run
# once the device is CUDA with a real count. The guard is kept as a defensive floor (these are pure
# helpers a future caller could invoke off the eligibility-gated path) that also avoids a
# divide-by-zero; unknown SM count => "assume the tile fills a wave" (don't shrink for occupancy).
def _wave_eff(bm: int, bn: int, m: int, n: int, cluster_m: int, num_sm: int) -> float:
    """Occupancy in CTAs (NOT tiles). Getting this wrong (counting tiles) picked
    bn=64 instead of the correct bn=128 for fp8 medium-M."""
    if num_sm <= 0:
        return 1.0
    ctas = _n_ctas(bm, bn, m, n, cluster_m)
    waves = max(1, math.ceil(ctas / num_sm))
    return ctas / (waves * num_sm)


def _num_waves(bm: int, bn: int, m: int, n: int, cluster_m: int, num_sm: int) -> int:
    if num_sm <= 0:  # unknown SM count sentinel — see note above _wave_eff
        return 1
    return max(1, math.ceil(_n_ctas(bm, bn, m, n, cluster_m) / num_sm))


def _pick_bk_ab(
    itemsize: int, cluster_m: int, bm: int, bn: int, budget: int, k: int
) -> tuple[int, int]:
    """Fill the AB pipeline to the ~196 KB SMEM isobar; pick ``(bk, ab)`` on it."""
    cap = _DTYPE_AB_CAP.get(itemsize, 3)
    # Tie-break on a bytes_used tie: the DEEPEST ab, then the larger bk.
    #
    # 6 -> 8 (see ``_DTYPE_AB_CAP``) MANUFACTURES a tie on the narrow-decode tile: ``bk=128``
    best: tuple[int, int, int] | None = None
    best_bk_ab: tuple[int, int] | None = None
    for bk in _DTYPE_BK_CHOICES.get(itemsize, (128, 64)):
        # bk must tile K: bk<=K and K a multiple of bk (a partial K-tile is not a valid
        # static-full tcgen05 tile). Skips the degenerate small-K case (e.g. K=64/bk=256).
        if bk > k or k % bk != 0:
            continue
        per = tcgen05_ab_smem_bytes_per_cta(
            bm=bm, bn=bn, bk=bk, dtype_bytes=itemsize, ab_stages=1, cluster_m=cluster_m
        )
        if per <= 0:
            continue
        ab = min(cap, max(1, budget // per))
        bytes_used = per * ab
        cand = (bytes_used, ab, bk)
        if best is None or cand > best:
            best = cand
            best_bk_ab = (bk, ab)
    if best_bk_ab is None:
        # No dtype-menu bk divides K (e.g. K=48). Fall back to the largest power-of-2
        # that tiles K, so the seed still emits a valid tile.
        bk = 1
        while bk * 2 <= k and k % (bk * 2) == 0:
            bk *= 2
        per = tcgen05_ab_smem_bytes_per_cta(
            bm=bm, bn=bn, bk=bk, dtype_bytes=itemsize, ab_stages=1, cluster_m=cluster_m
        )
        ab = min(cap, max(1, budget // per)) if per > 0 else 1
        return bk, ab
    # Read the answer from ``best_bk_ab`` rather than unpacking ``best``, so the ranking key
    # and the returned value cannot drift apart if the tie-break changes again.
    return best_bk_ab


def _single_wave_rect_tile(m: int, n: int, num_sm: int) -> tuple[int, int]:
    """Medium-M: keep the tall bm=256 M-tile (needs cluster_m=2) but shrink bn until the
    CTA count reaches ~one wave. e.g. M512/N4096 -> [256,128] = 128 CTAs = 0.86 waves.

    bn candidates are capped at N (never emit bn>N — wasted padding) so a narrow-N tall shape
    routed here from the compute branch's ``n>=256`` guard still gets a sane bn<=N."""
    bm = TCGEN05_TWO_CTA_BLOCK_M
    for bn in (256, 128, 64):
        if bn <= n and _wave_eff(bm, bn, m, n, 2, num_sm) >= _WAVE_FULL:
            return bm, bn
    return bm, min(64, n)


def _decode_bm(m: int) -> int | None:
    """Pick the cluster_m=1 decode M-tile, or None if M is outside the decode envelope."""
    if not _is_pow2(m):
        return None
    return min(m, _DECODE_BM)


def _pick_decode_bn(m: int, bm: int, n: int, num_sm: int) -> int:
    """Decode (cluster_m=1): pick the decode N-tile — the widest bn in {128,64,32,16} whose CTA"""
    for bn in _DECODE_BN_CHOICES:
        if _wave_eff(bm, bn, m, n, 1, num_sm) >= _DECODE_WAVE_FULL:
            return bn
    return _DECODE_BN_CHOICES[-1]


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _epilogue_aux_rank(env: CompileEnvironment, device_ir: DeviceIR) -> int:
    """0 = pure matmul / transparent (unary act, rank-1 rowvec bias, rowwise scale);"""
    from ..cute.aux_tensor import (
        host_function_has_tcgen05_exact_shape_aux_kernel_pattern,
    )

    host_function = getattr(device_ir, "host_function", None)
    if host_function is None:
        return 0
    try:
        with env:
            is_exact = host_function_has_tcgen05_exact_shape_aux_kernel_pattern(
                host_function
            )
    except Exception:
        # The detector walks the FX graphs; on any unexpected graph shape fall back to the
        # transparent (aux_rank=0) assumption rather than wrongly capping ab.
        return 0
    return 2 if is_exact else 0


def _single_matmul_fact(spec: ConfigSpec) -> MatmulFact | None:
    facts = spec.matmul_facts
    if len(facts) != 1:
        return None
    fact = facts[0]
    if fact.static_m is None or fact.static_n is None or fact.static_k is None:
        return None
    if fact.lhs_ndim != 2 or fact.rhs_ndim != 2:
        return None
    if fact.lhs_dtype is not fact.rhs_dtype:
        return None
    return fact


def _num_sm(env: CompileEnvironment) -> int:
    from ...runtime import get_num_sm

    try:
        return get_num_sm(env.device)
    except (AssertionError, NotImplementedError):
        # get_num_sm is always >= 1 for CUDA but RAISES for non-CUDA / unimplemented devices.
        # Return 0 as the "SM count unknown" sentinel the wave-fill helpers key off (see the
        # note above _wave_eff). Unreachable in practice — the eligibility SMEM-budget gate
        # already excludes non-CUDA — but keeps the helpers total if called off that path.
        return 0


def _fp8_small_grid_fits_one_wave(m: int, n: int, num_sm: int) -> bool:
    """True when the fp8 small-grid 128x128 cluster grid fits within ~one wave."""
    if num_sm <= 0 or n < _FP8_SMALL_GRID:
        return False
    clusters = (m // _FP8_SMALL_GRID) * (n // _FP8_SMALL_GRID)
    return 0 < clusters <= num_sm // 2


def _regime_tile(
    m: int, n: int, num_sm: int, itemsize: int
) -> tuple[int, int, int, str] | None:
    """Classify the regime and return ``(cluster_m, bm, bn, pid_type)``, or None to decline.

    Declines shapes outside the tcgen05 static-full-tile envelope (M not tileable by the
    regime's bm): the cluster_m=1 decode path needs a power-of-2 M; the cluster_m=2 tiles
    need ``M % bm == 0`` (bm=256, or bm=128 for the fp8 small-grid). Declined shapes fall
    through to the default fragment (exactly as #3007's Triton formula declines jagged)."""
    # decode: 64 <= M <= 128, cluster_m=1
    if m <= _DECODE_M_MAX:
        if m < _DECODE_M_MIN:
            return None  # below the tcgen05 decode admission floor (static_m>=64)
        bm = _decode_bm(m)
        if bm is None:
            return (
                None  # non-pow2 M (e.g. 96) — outside the cluster_m=1 static envelope
            )
        return 1, bm, _pick_decode_bn(m, bm, n, num_sm), "persistent_blocked"
    # cluster_m=2 tiles carry bm=256 -> M must tile it cleanly (single-root static tile).
    if m % TCGEN05_TWO_CTA_BLOCK_M != 0:
        return None
    tile2_waves = _wave_eff(
        TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N, m, n, 2, num_sm
    )
    if (
        m >= TCGEN05_TWO_CTA_BLOCK_M
        and n >= TCGEN05_TWO_CTA_BLOCK_N
        and tile2_waves >= _WAVE_FULL
    ):
        # compute: the 256² TMEM square fills a wave.
        # The ``n >= 256`` guard is a STRUCTURAL safety floor, not just occupancy: without it a
        # tall+narrow shape (huge M, N in [64,256)) can clear tile2_waves>=0.8 on M-tiles alone
        # (e.g. M=16384/N=64 -> 0.86 waves) yet the emitted bn=256 tile is mostly wasted padding
        # over an N=64 output. Requiring n>=256 routes those to the medium-M rectangular tile
        # below, which shrinks bn to <=128 so it never emits bn>N. (Curriculum-rare — extreme
        # aspect ratios — but a real correctness/efficiency hole the occupancy check alone misses.)
        return (
            2,
            TCGEN05_TWO_CTA_BLOCK_M,
            TCGEN05_TWO_CTA_BLOCK_N,
            "persistent_interleaved",
        )
    # medium-M: the 256² tile underfills. The 128x128 SMALL-GRID collective (per-CTA 64x128) shrinks
    # BOTH dims -> more tiles -> more CTAs, beating the rectangular [256,bn] tile via OCCUPANCY where
    #
    if (
        itemsize == 1
        and m % _FP8_SMALL_GRID == 0
        and n % _FP8_SMALL_GRID == 0
        and _fp8_small_grid_fits_one_wave(m, n, num_sm)
    ):
        return 2, _FP8_SMALL_GRID, _FP8_SMALL_GRID, "persistent_interleaved"
    # else single-wave rectangular cluster_m=2 tile (bn shrunk) — wins at wider N.
    bm, bn = _single_wave_rect_tile(m, n, num_sm)
    return 2, bm, bn, "persistent_interleaved"


def _formula_seed(
    fact: MatmulFact,
    spec: ConfigSpec,
    num_sm: int,
    budget: int,
    aux_rank: int,
) -> dict[str, Any]:
    """The Bucket-A core: regime-classify -> pick collective -> depth-fill the pipeline."""
    m, n, k = fact.static_m, fact.static_n, fact.static_k
    assert m is not None and n is not None and k is not None
    itemsize = _itemsize(fact.lhs_dtype)

    # ---- 1+2. classify regime + select collective/tile (None => decline) ----
    tile = _regime_tile(m, n, num_sm, itemsize)
    assert tile is not None  # is_eligible already checked _regime_tile is not None
    cluster_m, bm, bn, pid = tile

    # ---- 3+4. bk + ab: fill the AB pipeline to the ~196 KB SMEM isobar (dtype-aware) ----
    #           A rank-2 source-C residual forces the aux-TMA path, hard-capped at ab=2 (the
    #           C2/C5 ceiling — its C-ring can't coexist with a deeper AB pipeline). That cap
    #           alone determines the depth; bk is still picked on the full budget (measured:
    #           for the residual rect [256,128] tile, bk128 = 200 TFLOP/s beats bk64 = 161).
    bk, ab = _pick_bk_ab(itemsize, cluster_m, bm, bn, budget, k)
    if aux_rank == 2:
        ab = min(ab, 2)

    # ---- 5. l2_groupings: wave-count-aware (waves in CTAs). Grouping (G×G tile-walk swizzle) aims
    #         to reuse a shared operand panel across the tiles of a block WHILE it's L2-resident.
    #         few-wave: block finishes fast, short reuse distance -> panel still cached -> fewer DRAM
    waves = _num_waves(bm, bn, m, n, cluster_m, num_sm)
    l2 = [1] if waves >= _MANY_WAVE else [TCGEN05_TWO_CTA_SEED_L2_GROUPING]

    # ---- 6. inherit the codegen defaults for the bn-keyed knobs (§4.3.5) ----
    acc = 2 if bn <= 256 else 1
    c = 4 if bn <= 16 else 2

    seed: dict[str, Any] = {
        "block_sizes": [bm, bn, bk],
        "l2_groupings": l2,
        "num_warps": 8,
        "num_stages": 4,
        "pid_type": pid,
        "tcgen05_cluster_m": cluster_m,
        "tcgen05_cluster_n": 1,
        "tcgen05_ab_stages": ab,
        "tcgen05_acc_stages": acc,
        "tcgen05_c_stages": c,
        "tcgen05_num_epi_warps": 4,
        TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: _STATIC_PERSISTENT,
        # Bucket-A is a DEFAULT-layout, non-FFI seed. Pin these EXPLICITLY: on a
        # direct-entry-eligible shape the fragment defaults are FFI-biased
        # (``_base_default_config`` -> layout=explicit_epi_tile, tvm_ffi_launch=True,
        # flat_role=True, because PR2's search fragment keeps True as choices[0]). When
        # ``default_config()`` layers this promoted seed over ``_base_default_config()``
        # via dict.update, any key this seed OMITS keeps the base's FFI value — grafting
        # explicit_epi_tile + ffi=True onto our deep-AB DEFAULT config (e.g. bf16
        # [256,256,64] ab6 c2) yields an invalid hybrid (the FFI direct-entry path needs
        # the (bk=64,ab=6,c=4) tuple, but we emit c=2) that raises InvalidConfig in the
        # no-autotune / baseline compile. Emitting them keeps the promoted default
        # self-consistent on the DEFAULT layout. (The FFI topology ships separately as
        # the Bucket-B alt-seed, which sets its own layout/ffi keys.)
        TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: Tcgen05LayoutStrategy.DEFAULT.value,
        TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY: False,
        TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: False,
        # Clear the explicit-epi-tile layout overrides too: ``_base_default_config``
        # carries epi_tile_m=128 / epi_tile_n=32 / d_store_box_n=32 (the FFI topology),
        # which under the DEFAULT layout trip the "tcgen05 strategy invariants violated"
        # check. None = let codegen derive the DEFAULT-layout epi tile.
        TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY: None,
        TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY: None,
        TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY: None,
    }
    # Pure matmul has exactly the A/B/C indexing slots; only emit the explicit indexing
    # list then (a fused epilogue adds memory ops -> leave those to the spec default).
    if spec.indexing.length == 3:
        seed["indexing"] = ["tensor_descriptor"] * 3
    return seed


def _formula_eligible(env: CompileEnvironment) -> bool:
    """Shared eligibility for both formula heuristics: a single static 2-D tcgen05-native"""
    spec = env.config_spec
    if not spec.cute_tcgen05_search_enabled:
        return False
    fact = _single_matmul_fact(spec)
    if fact is None:
        return False
    # tcgen05 covers fp8 e4m3 + 16-bit (bf16/fp16). e5m2 / fp32 are out of the tcgen05
    # search envelope (universal-atom only) -> decline, fall through.
    if fact.lhs_dtype not in (torch.float8_e4m3fn, *_SIXTEEN_BIT_DTYPES):
        return False
    # Skinny-M (M<=16) is the SIMT-vec regime owned by CuteFp8GemmSkinnyMHeuristic (a bm=1
    # row-per-block kernel, not a tcgen05 tile) — decline so it stays default.
    if fact.static_m is None or fact.static_n is None:
        return False
    if fact.static_m <= _SKINNY_M_MAX:
        return False
    if len(spec.block_sizes) != 3:
        return False
    # Genuine SMEM floor (gate-2): a device that can't hold the AB ring at all.
    if CuteTcgen05Config.per_cta_ab_smem_budget_bytes(env.device) <= 0:
        return False
    # Decline shapes outside the tcgen05 static-full-tile envelope (M not tileable by the
    # regime bm — e.g. non-pow2 decode M, or M not a multiple of 256 for cluster_m=2).
    return (
        _regime_tile(
            fact.static_m, fact.static_n, _num_sm(env), _itemsize(fact.lhs_dtype)
        )
        is not None
    )


class CuteTcgen05FormulaMatmulHeuristic(AutotunerHeuristic):
    """Bucket-A formula seed for tcgen05 matmul — the #3007 analog (promote-to-default).

    Reads the ``MatmulFact`` directly (NOT the search-restriction gate) so it covers the
    regimes the 3 cluster_m=2 producers can't: cluster_m=1 decode, single-wave medium-M.
    Registered AFTER the demoted 3 producers so ``compiler_default_config`` (last-promote-
    wins) is this formula's config."""

    name = "cute_tcgen05_formula_matmul"
    backend = "cute"
    promote_seed_to_default = True

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return _formula_eligible(env)

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        spec = env.config_spec
        fact = _single_matmul_fact(spec)
        if fact is None:
            return None
        if fact.static_m is None or fact.static_n is None:
            return None
        budget = CuteTcgen05Config.per_cta_ab_smem_budget_bytes(env.device)
        if budget <= 0:
            return None
        num_sm = _num_sm(env)
        if (
            _regime_tile(
                fact.static_m, fact.static_n, num_sm, _itemsize(fact.lhs_dtype)
            )
            is None
        ):
            return None
        aux_rank = _epilogue_aux_rank(env, device_ir)
        seed = _formula_seed(fact, spec, num_sm, budget, aux_rank)
        return Config(**seed)


class CuteTcgen05FormulaFfiAltHeuristic(AutotunerHeuristic):
    """Bucket-B FFI ``explicit_epi_tile`` alt-seed for 16-bit full-tile compute."""

    name = "cute_tcgen05_formula_ffi_alt"
    backend = "cute"
    promote_seed_to_default = False

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        # Share the capability gate (tcgen05 enabled, single static tcgen05-native fact).
        if not _formula_eligible(env):
            return False
        spec = env.config_spec
        fact = _single_matmul_fact(spec)
        if fact is None:
            return False
        # FFI explicit_epi_tile is validated only for 16-bit (bf16/fp16) operands.
        if fact.lhs_dtype not in _SIXTEEN_BIT_DTYPES:
            return False
        # Only the full-tile 256² compute regime (the FFI path is full-tile-only) — and
        # only when the 256² tile actually fills a wave (else Bucket-A picks a rectangular
        # / decode tile the FFI topology can't express).
        m, n = fact.static_m, fact.static_n
        assert m is not None and n is not None
        if m < TCGEN05_TWO_CTA_BLOCK_M:
            return False
        num_sm = _num_sm(env)
        tile2_waves = _wave_eff(
            TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N, m, n, 2, num_sm
        )
        if tile2_waves < _WAVE_FULL:
            return False
        # bk must be in the direct-entry stage-tuple table (64 admits the deep (6,4) tuple).
        return _ffi_bk_ab(fact, spec, env, device_ir) is not None

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        spec = env.config_spec
        fact = _single_matmul_fact(spec)
        if fact is None:
            return None
        if fact.static_m is None or fact.static_n is None:
            return None
        bk_ab = _ffi_bk_ab(fact, spec, env, device_ir)
        if bk_ab is None:
            return None
        bk, ab, c = bk_ab
        waves = _num_waves(
            TCGEN05_TWO_CTA_BLOCK_M,
            TCGEN05_TWO_CTA_BLOCK_N,
            fact.static_m,
            fact.static_n,
            2,
            _num_sm(env),
        )
        l2 = [1] if waves >= _MANY_WAVE else [TCGEN05_TWO_CTA_SEED_L2_GROUPING]
        seed: dict[str, Any] = {
            "block_sizes": [TCGEN05_TWO_CTA_BLOCK_M, TCGEN05_TWO_CTA_BLOCK_N, bk],
            "l2_groupings": l2,
            "num_warps": 8,
            "num_stages": 4,
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": ab,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": c,
            TCGEN05_L2_SWIZZLE_SIZE_CONFIG_KEY: 1,
            "tcgen05_num_epi_warps": 4,
            TCGEN05_PERSISTENCE_MODEL_CONFIG_KEY: _STATIC_PERSISTENT,
            TCGEN05_LAYOUT_STRATEGY_CONFIG_KEY: Tcgen05LayoutStrategy.EXPLICIT_EPI_TILE.value,
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_M_KEY: 128,
            TCGEN05_LAYOUT_OVERRIDES_EPI_TILE_N_KEY: 32,
            TCGEN05_LAYOUT_OVERRIDES_D_STORE_BOX_N_KEY: 32,
            TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY: True,
            TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY: True,
        }
        if spec.indexing.length in (3, 4):
            seed["indexing"] = ["tensor_descriptor"] * spec.indexing.length
        return Config(**seed)


# The direct-entry stage tuples that the FFI codegen accepts, tried in order
# enumerate them is gone — admission is now bk-legality + the per-tile SMEM budget):
# bk=128 admits only (3,2). The deep bk=64/ab=6 pipeline won the bf16 compute keys, so it
_FFI_STAGE_TUPLES_BY_BK: tuple[tuple[int, int, int], ...] = (
    (64, 6, 4),
    (128, 3, 2),
    (64, 3, 2),
)


def _ffi_bk_ab(
    fact: MatmulFact, spec: ConfigSpec, env: CompileEnvironment, device_ir: DeviceIR
) -> tuple[int, int, int] | None:
    """Pick the deepest FFI (bk, ab, c) direct-entry tuple that (a) tiles K evenly and"""
    k = fact.static_k
    assert k is not None
    itemsize = _itemsize(fact.lhs_dtype)
    aux_rank = _epilogue_aux_rank(env, device_ir)
    budget = CuteTcgen05Config.per_cta_ab_smem_budget_bytes(env.device)
    # Respect the bk fragment range if present (bk must be reachable in the spec).
    bk_low, bk_high = _bk_fragment_bounds(spec)
    for bk, ab, c in _FFI_STAGE_TUPLES_BY_BK:
        if aux_rank == 2 and ab > 3:
            continue  # deep AB loses to the shallow tuple under a source-C residual ring
        if k % bk != 0:
            continue
        if bk_low is not None and not (bk_low <= bk <= bk_high):
            continue
        per = tcgen05_ab_smem_bytes_per_cta(
            bm=TCGEN05_TWO_CTA_BLOCK_M,
            bn=TCGEN05_TWO_CTA_BLOCK_N,
            bk=bk,
            dtype_bytes=itemsize,
            ab_stages=ab,
            cluster_m=2,
        )
        if 0 < per <= budget:
            return bk, ab, c
    return None


def _bk_fragment_bounds(spec: ConfigSpec) -> tuple[int | None, int]:
    if len(spec.block_sizes) != 3:
        return None, 0
    frag = spec.block_sizes[2]._fragment(spec)
    low = getattr(frag, "low", None)
    high = getattr(frag, "high", None)
    if isinstance(low, int) and isinstance(high, int):
        return low, high
    return None, 0
