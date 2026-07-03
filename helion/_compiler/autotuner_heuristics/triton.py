from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import NamedTuple
from typing import cast

import torch

from ...autotuner.config_fragment import EnumFragment
from ...autotuner.config_spec import FULL_EXTENT_CATEGORIES
from ...autotuner.config_spec import SIZED_REDUCTION_CATEGORIES
from ...autotuner.config_spec import ReductionCategory
from ...runtime.config import Config
from .common import clamp_block_size_targets
from .common import matches_hardware
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...autotuner.config_spec import BlockSizeSpec
    from ...autotuner.config_spec import ConfigSpec
    from ...autotuner.config_spec import MatmulFact
    from ...autotuner.config_spec import ReductionDescriptor
    from ...autotuner.config_spec import ReductionKernelFact
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .common import HardwareTarget


log = logging.getLogger(__name__)

_B200_MATMUL_HEURISTICS_PATH = Path(__file__).resolve().parent / "matmul_b200.json"


# Heuristic was originally contributed by @umechand-amd
# in https://github.com/pytorch/helion/pull/2357.
class TritonSkinnyGemmHeuristic(AutotunerHeuristic):
    name = "triton_skinny_gemm"
    backend = "triton"
    MIN_ASPECT_RATIO = 8
    BLOCK_TARGETS = (64, 64, 256)
    HARDWARE_TARGETS = (("cuda", "sm90"), ("rocm", "gfx950"))

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            return False
        facts = env.config_spec.matmul_facts
        if len(facts) != 1:
            return False
        fact = facts[0]
        if fact.lhs_ndim != 2 or fact.rhs_ndim != 2:
            return False
        if (
            fact.static_m is None
            or fact.static_n is None
            or fact.static_k is None
            or fact.m_block_id is None
            or fact.n_block_id is None
            or fact.k_block_id is None
        ):
            return False
        if max(fact.static_m, fact.static_n) < cls.MIN_ASPECT_RATIO * min(
            fact.static_m, fact.static_n
        ):
            return False
        return (
            clamp_block_size_targets(
                env,
                [
                    (fact.m_block_id, fact.static_m, cls.BLOCK_TARGETS[0]),
                    (fact.n_block_id, fact.static_n, cls.BLOCK_TARGETS[1]),
                    (fact.k_block_id, fact.static_k, cls.BLOCK_TARGETS[2]),
                ],
            )
            is not None
        )

    @classmethod
    def get_seed_config(cls, env: CompileEnvironment, device_ir: DeviceIR) -> Config:
        assert len(env.config_spec.matmul_facts) == 1
        fact = env.config_spec.matmul_facts[0]
        assert fact.static_m is not None
        assert fact.static_n is not None
        assert fact.static_k is not None
        assert fact.m_block_id is not None
        assert fact.n_block_id is not None
        assert fact.k_block_id is not None
        block_sizes = clamp_block_size_targets(
            env,
            [
                (fact.m_block_id, fact.static_m, cls.BLOCK_TARGETS[0]),
                (fact.n_block_id, fact.static_n, cls.BLOCK_TARGETS[1]),
                (fact.k_block_id, fact.static_k, cls.BLOCK_TARGETS[2]),
            ],
        )
        assert block_sizes is not None
        return Config(block_sizes=block_sizes)


def _dtype_family_from_dtype(dtype: object) -> str:
    dtype = str(dtype)
    if "float16" in dtype or "bfloat16" in dtype:
        return "fp16_bf16"
    if "float32" in dtype:
        return "fp32"
    return "other"


def _single_2d_static_matmul_fact(config_spec: ConfigSpec) -> MatmulFact | None:
    facts = config_spec.matmul_facts
    if len(facts) != 1 or len(config_spec.block_sizes) != 3:
        return None
    fact = facts[0]
    if fact.lhs_ndim != 2 or fact.rhs_ndim != 2:
        return None
    if fact.static_m is None or fact.static_n is None or fact.static_k is None:
        return None
    if (fact.m_block_id, fact.n_block_id, fact.k_block_id) != (0, 1, 2):
        return None
    return fact


def _shape_bucket_from_fact(fact: MatmulFact) -> dict[str, object]:
    assert fact.static_m is not None
    assert fact.static_n is not None
    assert fact.static_k is not None
    return {
        "dtype": _dtype_family_from_dtype(fact.lhs_dtype),
        "m_value": fact.static_m,
        "n_value": fact.static_n,
        "k_value": fact.static_k,
    }


@functools.cache
def _heuristic_rules() -> tuple[dict[str, object], ...]:
    with _B200_MATMUL_HEURISTICS_PATH.open(encoding="utf-8") as handle:
        data = cast("dict[str, list[dict[str, object]]]", json.load(handle))
    return tuple(data["rules"])


def _interval_contains(interval: str, value: int) -> bool:
    lower_text, upper_text = interval[1:-1].split(",", maxsplit=1)
    lower = float(lower_text)
    upper = float("inf") if upper_text == "inf" else float(upper_text)

    lower_ok = value >= lower if interval[0] == "[" else value > lower
    upper_ok = value <= upper if interval[-1] == "]" else value < upper
    return lower_ok and upper_ok


def _shape_bucket_matches(
    rule_bucket: dict[str, object],
    query_bucket: dict[str, object],
) -> bool:
    for key, value in rule_bucket.items():
        if key in {"k_bucket", "m_bucket", "n_bucket"}:
            intervals = value if isinstance(value, list) else [value]
            dim_value = cast("int", query_bucket[f"{key[0]}_value"])
            if not any(
                _interval_contains(cast("str", interval), dim_value)
                for interval in intervals
            ):
                return False
            continue
        query_value = query_bucket.get(key)
        values = value if isinstance(value, list) else [value]
        if query_value not in values:
            return False
    return True


def _rules_for_bucket(
    shape_bucket: dict[str, object],
) -> list[dict[str, object]]:
    matches = [
        rule
        for rule in _heuristic_rules()
        if _shape_bucket_matches(
            cast("dict[str, object]", rule["shape_bucket"]),
            shape_bucket,
        )
    ]
    matches.sort(
        key=lambda rule: len(cast("dict[str, object]", rule["shape_bucket"])),
        reverse=True,
    )
    return matches


def _materialize_config(
    raw: dict[str, object],
    *,
    config_spec: ConfigSpec,
) -> Config:
    flat_fields = config_spec._flat_fields()
    supported = {key: value for key, value in raw.items() if key in flat_fields}
    allowed_pid_types = config_spec.allowed_pid_types
    if (
        "pid_type" in supported
        and allowed_pid_types
        and supported["pid_type"] not in allowed_pid_types
    ):
        supported.pop("pid_type")
    config_spec.normalize(supported, _fix_invalid=True)
    config = Config(**cast("dict[str, Any]", supported))
    config_spec._shrink_for_numel_constraints(config)
    return config


def _seed_config_for_bucket(
    shape_bucket: dict[str, object],
    *,
    config_spec: ConfigSpec,
) -> Config | None:
    rules = _rules_for_bucket(shape_bucket)
    if not rules:
        return None

    for rule in rules:
        for template in cast("list[dict[str, object]]", rule["templates"]):
            return _materialize_config(template, config_spec=config_spec)
    return None


def _seed_config_for_config_spec(config_spec: ConfigSpec) -> Config | None:
    fact = _single_2d_static_matmul_fact(config_spec)
    if fact is None:
        return None
    return _seed_config_for_bucket(
        _shape_bucket_from_fact(fact),
        config_spec=config_spec,
    )


class TritonB200MatmulHeuristic(AutotunerHeuristic):
    name = "triton_b200_matmul"
    backend = "triton"
    promote_seed_to_default = True
    HARDWARE_TARGETS = (("cuda", "sm100"),)

    @classmethod
    def is_eligible(
        cls,
        env: CompileEnvironment,
        device_ir: DeviceIR,
    ) -> bool:
        return matches_hardware(env, cls.HARDWARE_TARGETS)

    @classmethod
    def get_seed_config(
        cls,
        env: CompileEnvironment,
        device_ir: DeviceIR,
    ) -> Config | None:
        return _seed_config_for_config_spec(env.config_spec)


class TritonPointwiseSeedHeuristic(AutotunerHeuristic):
    """Seed a bandwidth-saturating tile for PURE elementwise/pointwise kernels.

    A pointwise kernel (no reduction / matmul / accumulator) is BANDWIDTH-bound, but the compiler
    defaults it to ``block_size=32`` (~10% of HBM). This seed sizes the tile from a byte budget + grid
    occupancy, keyed on the derived ``PointwiseElementwiseFact`` — never on the activation or a dtype
    literal. Fires on that fact's presence (built only on the ABSENCE of the reduction/matmul/
    accumulator facts, so it never claws a reducing kernel into this track).
    """

    name = "triton_pointwise"
    backend = "triton"

    # Hill-climbed constants (see _lab/pointwise/NOTEBOOK.md).
    TILE_BYTES = 8192  # target HBM bytes moved per tile
    MIN_WAVES = 8  # grid >= num_sm * MIN_WAVES (size_hint-aware grid floor)
    BLOCK_FLOOR = 256  # never regress toward the bs=32 default
    # Per-program register/working-set ceiling (fp32-compute bytes) before spill / block-numel
    # overflow: wide enough not to bind the flat family, tight enough to bind a heavy rope slab.
    REGISTER_BYTES = 65536
    # num_warps ramp: a transcendental-heavy tile is latency-bound and wants more warps to hide SFU
    # latency; capped at tile_numel // ELEMS_PER_WARP so a small tile does not starve its warps.
    DEFAULT_WARPS = 4
    MAX_WARPS = 16
    SFU_W8 = 3  # >= this many SFU ops -> >= 8 warps
    SFU_W16 = 9  # >= this many SFU ops -> 16 warps
    ELEMS_PER_WARP = (
        64  # each warp needs at least this many tile elements to be worth spawning
    )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return bool(env.config_spec.pointwise_facts)

    @classmethod
    def get_seed_config(cls, env: CompileEnvironment, device_ir: DeviceIR) -> Config:
        from ...runtime import get_num_sm

        spec = env.config_spec
        fact = spec.pointwise_facts[0]
        num_sm = max(1, get_num_sm(env.device))
        # slab_numel (untiled inner slab per tiled element) scaled by two widths into two caps:
        # budget_target = tiled elements per bandwidth-saturating program (STORAGE bytes); reg_cap =
        # how many fit before the fp32 working set spills (COMPUTE bytes; coarse proxy — see the fact).
        # A heavy rope slab makes both ~1, so it is not tiled past ~1 (vs the old spilling [1,256]).
        slab_bytes = max(1, fact.slab_numel * fact.storage_itemsize)
        reg_bytes = max(1, fact.slab_numel * fact.compute_itemsize)
        budget_target = max(1, cls.TILE_BYTES // slab_bytes)
        reg_cap = max(1, cls.REGISTER_BYTES // reg_bytes)
        # size_hint-aware: cap the tile so the grid keeps the SMs busy on small problems.
        occ_cap = max(1, fact.total_numel // (num_sm * cls.MIN_WAVES))
        target = max(1, min(budget_target, reg_cap, occ_cap))
        # Anti-undershoot floor, capped by the REGISTER budget only (NOT budget_target, NOT occ_cap):
        # keep a coalesced per-operand run, lowering it only on a genuine register overflow (a heavy
        # rope slab → reg_cap≈1). Low occupancy or a fan-in kernel's small byte budget is not worth it.
        inner_floor = min(cls.BLOCK_FLOOR, cls._pow2_floor(reg_cap))
        # Budget for _balanced_block_sizes: a coalescing CONFLICT (>1 contiguous axis, e.g. transposed
        # load + contiguous store) fills a square tile up to the register limit only — the bandwidth
        # budget is wasted on the strided operand, and a long coalescing run beats more programs.
        balance_cap = max(1, reg_cap)
        block_sizes = cls._seed_block_sizes(
            spec, target, inner_floor, fact.contig_block_ids, balance_cap
        )
        tile_numel = 1
        for b in block_sizes:
            tile_numel *= b
        # num_warps only when the SFU ramp raises it above the default; else None (Config drops None
        # keys), so the flat family stays block_sizes-only and byte-identical (no dead knob).
        num_warps = cls._warps_for(fact.sfu_ops, tile_numel)
        return Config(
            block_sizes=block_sizes,
            num_warps=num_warps if num_warps > cls.DEFAULT_WARPS else None,
        )

    @classmethod
    def _warps_for(cls, sfu_ops: int, tile_numel: int) -> int:
        """num_warps from SFU op count, capped by tile size (each warp needs >= ELEMS_PER_WARP
        elements or it starves)."""
        if sfu_ops >= cls.SFU_W16:
            target = cls.MAX_WARPS
        elif sfu_ops >= cls.SFU_W8:
            target = 8
        else:
            target = cls.DEFAULT_WARPS
        cap = cls._pow2_floor(max(1, tile_numel // cls.ELEMS_PER_WARP))
        return max(cls.DEFAULT_WARPS, min(cls.MAX_WARPS, target, cap))

    @staticmethod
    def _pow2_floor(value: int) -> int:
        return 1 << (value.bit_length() - 1) if value >= 1 else 1

    @classmethod
    def _clamp_dim(cls, target: int, bs_spec: BlockSizeSpec, floor: int) -> int:
        # Round DOWN to a pow2 within [floor (and the spec's correctness min), max_size]. max_size =
        # next_pow2(extent), so a short row is covered in one masked tile (768 -> 1024). autotuner_min
        # is the autotuner's search floor, not a seed constraint, so it is intentionally not applied.
        cand = cls._pow2_floor(max(1, target))
        cand = max(cand, floor, bs_spec.min_size)
        cand = min(cand, bs_spec.max_size)
        return max(1, cand)

    @classmethod
    def _seed_block_sizes(
        cls,
        spec: ConfigSpec,
        target: int,
        inner_floor: int = BLOCK_FLOOR,
        contig_block_ids: tuple[int, ...] = (),
        balance_cap: int = 1 << 30,
    ) -> list[int]:
        """Distribute the target tile across the block dims so the wide part lands on a CONTIGUOUS
        (stride-1) axis (from ``contig_block_ids``, not assumed to be the last dim):
        - single contiguous axis: fill it innermost-first, spilling leftover budget outward (row-major
          → the last dim, byte-identical to the prior seed; a transposed view → dim 0, e.g. [1024,1]
          instead of the uncoalesced [1,1024]).
        - CONFLICT (>1 contiguous axis, e.g. transposed load + contiguous store): no single wide axis
          coalesces every operand, so emit a BALANCED tile (see _balanced_block_sizes).
        The floor (register-capped) applies to the primary contiguous axis to keep a coalesced run."""
        n = len(spec.block_sizes)
        specs = [cast("BlockSizeSpec", spec.block_sizes[i]) for i in range(n)]
        # Positions whose block-id is a contiguous (stride-1) axis for some full-extent op.
        contig_pos = [i for i in range(n) if specs[i].block_id in contig_block_ids]
        if len(contig_pos) >= 2:
            return cls._balanced_block_sizes(specs, contig_pos, balance_cap)
        block = [1] * n
        # Root the wide tile at the contiguous axis; fall back to the last dim when unknown.
        primary = contig_pos[0] if contig_pos else (n - 1)
        order = [primary] + [i for i in reversed(range(n)) if i != primary]
        remaining = max(1, target)
        for (
            i
        ) in order:  # contiguous axis first (gets the floor), spill the rest outward
            floor = inner_floor if i == primary else 1
            block[i] = cls._clamp_dim(remaining, specs[i], floor)
            remaining = max(1, remaining // block[i])
            if remaining <= 1:
                break
        return block

    @classmethod
    def _balanced_block_sizes(
        cls, specs: list[BlockSizeSpec], contig_pos: list[int], balance_cap: int
    ) -> list[int]:
        """Balanced (square-ish) pow2 tile for a coalescing CONFLICT: give every contiguous axis an
        EQUAL run up to ``balance_cap`` — a single wide axis would stride the other operand. Non-
        contiguous axes stay 1."""
        n = len(specs)
        block = [1] * n
        k = len(contig_pos)
        run = 1
        while (
            run * 2
        ) ** k <= balance_cap:  # largest pow2 run with run**k within the budget
            run *= 2
        for i in contig_pos:
            block[i] = cls._clamp_dim(run, specs[i], 1)
        return block


def _triton_reduction_eligible(env: CompileEnvironment, device_ir: DeviceIR) -> bool:
    """Gate: the kernel has >= 1 SIZED reduction and no ``matmul_facts`` (GEMMs route to the matmul
    seeds). Admits both tracks (standard rollable, user-tiled), including a multi-reduction kernel.
    A reduction with no sized member (only GRID_TILE / DECLINED) declines.

    Keyed purely on the Stage-1 kernel fact. ``build_reduction_kernel_fact`` runs on every live
    compile, so the fact is absent only for a bare-spec unit test or a kernel with genuinely no
    reduction; both correctly decline.
    """
    spec = env.config_spec
    if spec.matmul_facts:
        return False
    kf = spec.reduction_kernel_fact
    if kf is None:
        return False
    return any(d.category in SIZED_REDUCTION_CATEGORIES for d in kf.reductions)


def _primary_descriptor_selected(env: CompileEnvironment) -> ReductionDescriptor | None:
    """The primary reduction descriptor: max ROW-BYTES (``size_hint * input_load_itemsize``) over
    the backed sized descriptors (not category tier-order, which would mis-rank the group-quant
    kernels). ``None`` if there is no sized reduction / no kernel fact.

    This is the single Stage-1 source the reduction tracks read every scalar lever off (num_warps
    / persistence / footprint caps). On the live compile path the kernel fact is present whenever
    there is a sized reduction, so ``None`` is the test-only / no-sized-reduction case.
    """
    from torch._inductor.utils import free_unbacked_symbols

    kf = env.config_spec.reduction_kernel_fact
    if kf is None:
        return None
    sized = [d for d in kf.reductions if d.category in SIZED_REDUCTION_CATEGORIES]
    if not sized:
        return None

    def _is_backed(d: ReductionDescriptor) -> bool:
        size = env.block_sizes[d.block_id].size
        # A concrete int or SymInt with no free unbacked symbols is backed; a non-int/SymInt
        # size (AutoSize / None) is treated as unbacked (conservative — same as today, where a
        # symbolic size makes ``free_unbacked_symbols`` truthy and drops it from ``backed``).
        if not isinstance(size, (int, torch.SymInt)):
            return False
        return not free_unbacked_symbols(size)

    backed = [d for d in sized if _is_backed(d)]
    pool = backed or sized
    return max(
        pool, key=lambda d: (d.size_hint * max(1, d.input_load_itemsize), d.size_hint)
    )


def _is_standard_reduction(pd: ReductionDescriptor) -> bool:
    """Standard vs user-tiled discriminator, keyed on the primary reduction's category: standard
    iff FULL_SLICE (a rolled rdim or a materialized full-width rdim the roller declined) or
    FULL_GRID; user-tiled is the USER_TILE case (the rdim is a ``block_sizes`` entry).
    """
    return pd.category in FULL_EXTENT_CATEGORIES


class _TileAllocation(NamedTuple):
    """The result of :meth:`_TritonReductionSeedBase.size_reduction_tiles` — the single
    per-co-residency-group budget allocation that produces every tile size the seed emits.

    Per co-residency group the allocator forms a register/byte capacity, then seats axes in
    priority order (full-extent reductions → user-tile reductions → grid-tile reductions → the
    grid-M rows), each taking first crack then floored by the budget remaining after everything
    already seated. Earlier groups' assignments are held fixed as inputs to later groups; the
    non-reduction loops are sized last against the remaining headroom. Floor-vs-resident and
    collapse-vs-widen are budget outcomes, not separate branches.

    - ``block_sizes``: the full ``Config.block_sizes`` vector — every tunable axis sized.
    - ``block_sizes_red_values``: ``{block_id -> r_block}`` for every tunable sized reduction that
      rides a ``block_sizes`` slot (the user-tiled track's reductions, including its primary). The
      standard track's rolled primary rides ``reduction_loops`` instead, surfaced via
      ``primary_r_block``/``persistent`` — so the name is about the emission target (block_sizes
      slot), not "secondary". Emission routing is the only standard-vs-user difference; every
      reduction gets a size from the same budget.
    - ``primary_r_block`` / ``persistent``: the primary reduction's chunk + persistence verdict
      (the byte budget admits the full extent AND the row is re-read).
    - ``rolled_loop_sizes``: ``{block_id -> (r_block, persistent)}`` for every rolled reduction axis
      OTHER than the primary (a kernel that rolls >1 reduction into separate ``reduction_loops``
      subgraphs). Empty unless a kernel rolls more than one reduction.
    """

    block_sizes: list[int]
    block_sizes_red_values: dict[int, int]
    primary_r_block: int
    persistent: bool
    rolled_loop_sizes: dict[int, tuple[int, bool]]


class _TritonReductionSeedBase(AutotunerHeuristic):
    """Shared base for the two Triton inner-reduction seed heuristics. Both consume the Stage-1
    ``ReductionKernelFact`` through ONE budget allocator (:meth:`size_reduction_tiles`); the
    subclasses differ ONLY in how they map the allocation onto knobs (EMISSION routing):

    - **standard** (:class:`TritonStandardReductionHeuristic`): Helion rolls the rdim into a
      ``reduction_loops`` loop, so the primary reduction's size lands on that knob.
    - **user-tiled** (:class:`TritonUserTiledReductionHeuristic`): the user hand-writes the
      ``hl.tile`` loop, so each reduction axis is a ``block_sizes`` entry.

    Not registered; only the subclasses are.
    """

    backend = "triton"
    # Widen the declared type so the sm100 subclass can retarget it (the base is sm90-only).
    HARDWARE_TARGETS: ClassVar[tuple[HardwareTarget, ...]] = (("cuda", "sm90"),)

    # ----- THE BUDGET (a register/byte capacity; everything else is a per-axis desire) -----
    # Per-program persistent byte ceiling: the group's resident working set — the sum over its live
    # tiles of ``itemsize × ∏(tile dims)`` — must fit this, else a tile floors. ~240 KiB, just over
    # H100 SMEM.
    ROW_PERSIST_MAX_BYTES = 245760
    # The tighter byte ceiling for a CARRIED reduction (an accumulator whose last dim is the rdim,
    # e.g. kl_div/jsd's ``[grid_M, R]``): that tile is held resident across the whole inner loop
    # rather than streamed-and-released, a heavier steady-state pressure, so the chunk sharing SRAM
    # with it wants a smaller extent. Half of ROW_PERSIST. This is the only place the
    # carried-vs-streamed distinction lives.
    CARRIED_PERSIST_MAX_BYTES = 245760 // 2
    # The PERSISTENCE-HOLD ceiling — the byte watermark under which a re-read row may hold its FULL
    # extent (vs the chunk budget, which sizes a streamed/looped tile). Only ``row_reread AND
    # carried_2d_count == 0`` reductions reach the hold, so a carried tile never loosens. The true
    # cutoff is not a single faithful byte budget (e.g. softmax flips at ~128-160 KiB, cross_entropy
    # at ~256-384 KiB with the same footprint), so these are two calibrated buckets selected by
    # ``_has_store_only_row_reread`` — a coarse proxy for whether persist's avoided HBM re-read lives
    # in the small L2 working set (tighter ceiling) or the large register file (looser ceiling):
    #  - no store-only re-read (cross_entropy/sum): reuse is register-resident, so holding a high
    #    watermark wins far out. 3x ROW.
    #  - a store-only re-reading pass exists (softmax/rms/layer_norm/welford): the row is re-swept
    #    from L2, so past ~a few KiB/row streaming beats holding it. ~1.2x ROW.
    PERSIST_HOLD_MAX_BYTES = 3 * 245760
    USER_TILE_PERSIST_HOLD_MAX_BYTES = 294912
    # Looped-fallback reduction chunk (pow2) for a row that does not fit the persistent budget.
    LOOPED_CHUNK = 16384
    # Occupancy floor for the grid-M widen: keep the post-tile grid >= num_sm * MIN_WAVES so
    # collapsing a fan-out sibling never under-occupies (mirrors the pointwise seed's MIN_WAVES).
    MIN_WAVES = 8
    # Diminishing-returns ceiling on the grid-M widen (rows/program): a memory-bound reduction does
    # not amortize past a handful of batched rows, and widening only trades away grid parallelism.
    # Bounds the widen the byte/occupancy caps alone would permit on a small-row huge-M kernel. Does
    # NOT bound the grad-param COLLAPSE branch (which intentionally batches rows to cut the
    # cross-grid finalize) nor a raised autotuner_min floor (max(floor, ...) still wins).
    WIDEN_MAX_ROWS = 8

    # num_warps ramp: keyed on the primary reduction extent (see ``_num_warps``).

    # =============================== Stage-1 fact accessors ================================= #
    @classmethod
    def _non_reduction_loop_ids(cls, spec: ConfigSpec) -> tuple[int, ...]:
        """The non-reduction user-tiled loops (welford's normalize pass) -- sized as a separate
        apply pass, NOT reduction-sized. Read off ``ReductionKernelFact.non_reduction_loop_block_ids``.
        """
        kf = spec.reduction_kernel_fact
        assert kf is not None
        return kf.non_reduction_loop_block_ids

    @classmethod
    def _resident_block_ids(cls, spec: ConfigSpec) -> set[int]:
        """The union of block_ids that appear (as a resolved dim) in some co-residency group's
        live-tile set — the "is this axis register-resident?" test. The single definition of
        residency, shared by the grid-M widen (a resident grid axis widens into the byte budget; a
        non-resident one is reduced away -> collapses) and ``_has_reduced_away_grid``. Empty if no
        kernel fact (a bare-spec unit test)."""
        kf = spec.reduction_kernel_fact
        if kf is None:
            return set()
        resident: set[int] = set()
        for g in kf.coresidency_groups:
            for tile in g.live_tiles:
                resident.update(d for d in tile if d is not None)
        return resident

    @classmethod
    def _has_reduced_away_grid(cls, spec: ConfigSpec) -> bool:
        """True iff some grid axis is REDUCED AWAY — a grid block_id that appears in NO live tile,
        i.e. a sequential cross-grid reduction loop whose partial is finalized by a later
        ``.sum(0)`` (the grad-parameter M-collapse idiom). Uses the shared ``_resident_block_ids``
        residency test. False if no kernel fact."""
        kf = spec.reduction_kernel_fact
        if kf is None:
            return False
        resident = cls._resident_block_ids(spec)
        return any(g not in resident for g in kf.grid_axis_block_ids)

    @staticmethod
    def _max_group_footprint(
        kf: ReductionKernelFact,
        axis: int,
        footprint_terms: Callable[
            [tuple[tuple[int | None, ...], ...], int], tuple[int, int]
        ],
        default_tiles: tuple[tuple[int | None, ...], ...],
    ) -> tuple[int, int]:
        """The ``(scale, flat)`` footprint for sizing ``axis``, taken from the heaviest co-residency
        group that spans it (largest ``scale``). A reduction axis is tiled the same width
        everywhere, so it must fit the worst group that uses it. ``flat`` comes from that same max
        group (mixing scale/flat across groups breaks the chunk solve). If the axis spans no group's
        tiles (a bare-spec / degenerate case), fall back to ``default_tiles`` (this descriptor's own
        group)."""
        best = None
        for g in kf.coresidency_groups:
            if not any(axis in t for t in g.live_tiles):
                continue
            scale, flat = footprint_terms(g.live_tiles, axis)
            if best is None or scale > best[0]:
                best = (scale, flat)
        return best if best is not None else footprint_terms(default_tiles, axis)

    @classmethod
    def _has_store_only_row_reread(
        cls, spec: ConfigSpec, pd: ReductionDescriptor
    ) -> bool:
        """True iff the primary reduction's row tensor is ALSO loaded by a store-only pass — a load
        of that tensor that feeds a store and no reduction (``stores_fed and not reductions_fed``).

        This selects the persist-hold ceiling. The physical question it stands in for is whether
        persistence's benefit (avoiding the row's HBM re-read) is served from the small L2 working
        set (tighter ceiling) or the large register file (looser ceiling). That quantity is not
        cleanly recoverable from any seed-time signal — kernels with the same byte footprint, load
        count, and output width can flip persist->chunk at ~2x-different points — so this is an
        ADMITTED PROXY: it classifies the tested kernels correctly but is not a faithful measure of
        the underlying cache-tier question and can be fooled (e.g. a 2-pass kernel whose 2nd pass
        reduces instead of storing re-reads the row identically but reads as False). If a kernel
        regresses on the persist ceiling, this proxy is the first suspect.

        Detected from the walker ``MemoryOpFact`` list (no re-walk). Not the same as
        ``non_reduction_loop_block_ids`` (a 2nd pass that reduces over the same axis leaves that set
        empty). Empty facts / no kernel fact -> False."""
        facts = spec.memory_op_facts
        if not facts:
            return False
        red_tensors = {
            f.tensor_name
            for f in facts
            if f.kind == "load"
            and f.tensor_name is not None
            and any(ax == pd.block_id for ax, _ in f.reductions_fed)
        }
        if not red_tensors:
            return False
        return any(
            f.kind == "load"
            and f.tensor_name in red_tensors
            and f.stores_fed
            and not f.reductions_fed
            for f in facts
        )

    @classmethod
    def non_reduction_loop_block_cap(
        cls, spec: ConfigSpec, pd: ReductionDescriptor
    ) -> int | None:
        """Optional element cap for a non-reduction apply loop. ``None`` = no extra cap beyond the
        shared ``loop_budget`` (sm90/H100 unchanged); a subclass may return a smaller budget."""
        return None

    # =============================== scalar levers (outside the budget) ===================== #
    @classmethod
    def _num_warps(cls, pd: ReductionDescriptor) -> int:
        """Scale num_warps with the reduction extent (pow2): rnumel <= 1024 -> 4, <= 4096 -> 8,
        <= 16384 -> 16, > 16384 -> 32."""
        rnumel = pd.size_hint
        warps32_min_elems = 16384
        if rnumel > warps32_min_elems:
            return 32
        if rnumel <= 1024:
            return 4
        if rnumel <= 4096:
            return 8
        return 16

    @classmethod
    def _block_floor(cls, bs_spec: BlockSizeSpec) -> int:
        """The smallest valid block size for an entry (honors a raised ``autotuner_min`` for
        large-M shapes rather than emitting an invalid ``block_size=1``)."""
        return max(1, bs_spec.min_size, bs_spec.autotuner_min)

    @classmethod
    def _m_axis_block_size(cls, spec: ConfigSpec, mbid: int) -> int:
        """Seed block size (rows/program) for one M-axis (grid) block_id, whether or not it is a
        tunable ``block_sizes`` entry. A grid-PINNED axis (``hl.tile(M, block_size=1)``) has no
        tunable slot and lives solely on the program grid -- read its FIXED value off
        ``env.block_sizes`` (the grid-pinned-M idiom every vLLM quant kernel uses)."""
        if mbid in spec.block_sizes.valid_block_ids():
            m_idx = spec.block_sizes.block_id_to_index(mbid)
            return cls._block_floor(cast("BlockSizeSpec", spec.block_sizes[m_idx]))
        from ...runtime.config import Config as _Config
        from ..compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        value = env.block_sizes[mbid].from_config(_Config(block_sizes=[]))
        if isinstance(value, (int, torch.SymInt)):
            return max(1, int(value))
        log.warning(
            "reduction seed: M-axis block_id=%s resolved to a non-static block size %r; "
            "falling back to block_size=1 (this should not happen for a pinned grid axis)",
            mbid,
            value,
        )
        return 1

    @classmethod
    def _eviction_policies(
        cls,
        env: CompileEnvironment,
        kind: str,
        reread_slot: int | None = None,
    ) -> list[str] | None:
        """``load_eviction_policies`` list (spec length); None leaves the autotuner default.
        - ``"stream"`` — single streamed input (read once): every load -> ``'first'`` (frees L2).
        - ``"reread"`` — the row is re-read across passes: its first load -> ``'last'``
          (L2-resident), rest -> ``'first'``. ``reread_slot`` from ``reread_eviction_index``."""
        n = env.config_spec.load_eviction_policies.length
        if n <= 0:
            return None
        if kind == "stream":
            return ["first"] * n
        if kind == "reread":
            if reread_slot is None or not 0 <= reread_slot < n:
                return None
            policy = ["first"] * n
            policy[reread_slot] = "last"
            return policy
        return None

    # ================================ THE BUDGET ALLOCATOR ============================== #
    @classmethod
    def size_reduction_tiles(
        cls,
        env: CompileEnvironment,
        spec: ConfigSpec,
        device_ir: DeviceIR,
        pd: ReductionDescriptor,
    ) -> _TileAllocation:
        """THE allocator: a per-co-residency-group BUDGET over the group's ACTUAL resident live
        tiles (``CoResidencyGroup.live_tiles``) assigns every tile size, in TWO passes.

        The footprint is faithful: ``resident_bytes = itemsize × Σ over the group's live tiles of
        ∏(tile dim widths)``. Sizing an axis A splits that sum into ``(scale, flat)`` — tiles
        CONTAINING A scale with ``block(A)``, tiles WITHOUT A are constant — and the budget test is
        ``itemsize × (scale × block(A) + flat) <= budget`` (the constant term SUBTRACTED, never
        divided). No ``num_live`` multiplier, no separate accumulator sum, no feature-extent
        reconstruction: the live tiles ARE the resident set (accumulators captured inline at real
        shape, scalar carries as rank-1 constant tiles).

        For each co-residency group:

          PASS 1 — seat the reductions with the grid axes pinned at their FLOOR (full-extent ->
            user-tile -> grid-tile). A re-read full-slice raises its floor to the full extent
            (PERSISTENCE) iff its resident tile fits the budget; else it chunks to
            ``min(LOOPED_CHUNK, byte budget, extent)``. A carried reduction (kl_div/norm-bwd) sizes
            against the tighter ``CARRIED_PERSIST`` budget.

          PASS 2 — the grid-M rows take the REMAINDER. A grid axis that is RESIDENT (appears in some
            live tile -> its row co-occupies the working set) WIDENS into the byte remainder (capped
            by occupancy + WIDEN_MAX_ROWS + extent) and FLOORS when the budget is spent. A grid axis
            in NO live tile is REDUCED AWAY (a sequential cross-grid ``.sum(0)`` finalize, holds no
            bytes) -> its floor raises to ``grid_rows / num_sm`` (collapse the finalize to ~1 SM
            wave). Both are pure per-axis MEMBERSHIP outcomes — no ``cdiv`` branch, no recognizer.

        Then the non-reduction loops LAST (welford's normalize, rms_norm_per_block's groups_per_row)
        — a separate pass co-resident with nothing in the group, sized against its own headroom.

        EMISSION is the ONLY standard-vs-user difference: a reduction's computed size is WRITTEN to
        ``reduction_loops`` (rolled/standard) or a ``block_sizes`` slot (user-tiled). Every
        reduction gets a size from the SAME budget; the split is codegen routing, not a different
        way to compute.
        """
        from ..._utils import next_power_of_2 as _np2
        from ..._utils import prev_power_of_2 as _pp2
        from ...runtime import get_num_sm

        num_sm = max(1, get_num_sm(env.device))
        occ_floor = num_sm * cls.MIN_WAVES
        itemsize = max(1, pd.itemsize)
        valid = set(spec.block_sizes.valid_block_ids())
        kf = spec.reduction_kernel_fact
        assert kf is not None
        grid_ids = set(kf.grid_axis_block_ids)
        non_reduction_loop_ids = set(cls._non_reduction_loop_ids(spec))
        reduction_ids = {d.block_id for d in kf.reductions}

        # Extent (pow2-padded) per block_id, read from STORED hints. The reason these maps exist at
        # all is TESTING: the reduction unit tests call ``get_seed_config`` on a bare spec OUTSIDE an
        # active CompileEnvironment, where ``env.block_sizes[bid]`` is unavailable — so extents must
        # come from data already persisted on the spec/fact. A reduction's extent is its descriptor
        # ``size_hint``; a tunable axis's is its ``BlockSizeSpec.size_hint``. The third fallback
        # (``env.block_sizes`` — a non-tunable pinned grid / materialized feature) is LIVE-PATH ONLY:
        # in the no-env test path every axis is in ``_spec_extent`` or ``_desc_extent`` by
        # construction, so that branch never executes (and would raise NoCurrentEnvironment if it did).
        _desc_extent = {d.block_id: d.size_hint for d in kf.reductions}
        _spec_extent = {
            cast("BlockSizeSpec", spec.block_sizes[i]).block_id: cast(
                "BlockSizeSpec", spec.block_sizes[i]
            ).size_hint
            for i in range(len(spec.block_sizes))
        }

        def extent_of(bid: int) -> int:
            if bid in _spec_extent:
                return _np2(_spec_extent[bid])
            if bid in _desc_extent:
                return _np2(_desc_extent[bid])
            return _np2(env.block_sizes[bid].size_hint())

        # The persistence/chunk budget a reduction sizes against — the only place the regime enters
        # (the footprint formula is identical everywhere; only this number changes). Two budgets,
        # keyed PER-REDUCTION on whether THIS reduction carries a >=2-D tile:
        #  - CARRIED (``carried_2d_count > 0``): the reduction's own ``[grid_M, R]`` accumulator is
        #    held resident across the whole inner loop, a heavier steady-state pressure than a
        #    streamed row -> the tighter budget -> smaller chunk.
        #  - STREAMED (``carried_2d_count == 0``): the ROW budget. Per-reduction, not kernel-wide: a
        #    grad-parameter norm-bwd carries its N accumulator on the materialized N axis, but the
        #    co-resident inner tile it sizes is itself non-carried and wants the looser ROW budget.
        #    A per-row scalar carry (e.g. welford mean/M2 ``[grid_M]``) has c2d=0, so it stays STREAMED.
        def persist_budget_for(d: ReductionDescriptor) -> int:
            return (
                cls.CARRIED_PERSIST_MAX_BYTES
                if d.carried_2d_count > 0
                else cls.ROW_PERSIST_MAX_BYTES
            )

        # The static grid-row count (program count before any widen), the occupancy numerator.
        from ..compile_environment import NoCurrentEnvironment

        grid_rows = 1
        # The try/except is NECESSARY (not defensive noise): this block dereferences the live
        # ``env`` (``env.block_sizes[gbid].size``, ``env.size_hint``), which the no-env unit-test
        # path (see ``extent_of`` above) cannot provide -> ``NoCurrentEnvironment``; a dynamic/None
        # grid size raises ``AttributeError``/``TypeError``. All three collapse to the SAME defined
        # fallback ``grid_rows = 0`` = "no compile-time occupancy", which the pass-2 occupancy widen
        # already handles (it simply does not fire). Scoped tightly to the env-touching loop so it
        # cannot mask an unrelated bug.
        try:
            for gbid in grid_ids:
                size = env.block_sizes[gbid].size
                if isinstance(size, (int, torch.SymInt)):
                    grid_rows *= env.size_hint(size)
                else:
                    grid_rows = 0  # dynamic grid -> no compile-time occupancy
                    break
        except (NoCurrentEnvironment, AttributeError, TypeError):
            grid_rows = 0

        # ``seated`` holds every tile assigned so far (held fixed for later sizing); ``sizes`` is
        # the subset that lands in tunable ``block_sizes`` slots. PASS 1 seats every grid axis at
        # its FLOOR; the reductions are sized against that floored grid, then PASS 2 widens the grid
        # into whatever budget the seated reductions left (the two-pass structure — reductions
        # first with the grid pinned low, then the grid).
        seated: dict[int, int] = {}
        for gbid in sorted(grid_ids):
            seated[gbid] = cls._m_axis_block_size(spec, gbid)
        sizes: dict[int, int] = {}
        block_sizes_red_values: dict[int, int] = {}
        rolled_loop_sizes: dict[int, tuple[int, bool]] = {}
        primary_r_block = 1
        persistent = False

        # Which axes are register-resident (see ``_resident_block_ids``). A grid axis in a live tile
        # widens into the byte budget; a grid axis in no live tile is reduced away by a later
        # ``.sum(0)``, holds no bytes, and collapses to ~1 SM wave instead.
        resident_block_ids = cls._resident_block_ids(spec)

        # A kernel with a loop-carried >=2-D accumulator (``carried_2d_count >= 1`` on any reduction)
        # pins that ``[grid_M, R]`` state in registers across the whole inner loop, so widening the
        # resident grid is risky (it multiplies the pinned register footprint and trips the
        # CTA-per-SM occupancy cliff). Such a kernel keeps its resident grid at FLOOR (no widen).
        carried_kernel = any(d.carried_2d_count > 0 for d in kf.reductions)

        def footprint_terms(
            tiles: tuple[tuple[int | None, ...], ...],
            axis: int,
        ) -> tuple[int, int]:
            """The group footprint as ``(scale, flat)``: resident bytes while sizing ``axis`` =
            ``itemsize × (scale × block(axis) + flat)`` — an axis-scaling term plus a constant term,
            kept separate (they ADD; folding the constant into a per-element coefficient over-counts
            it and wrongly denies persistence). Sum ``∏(dim widths)`` over the group's live tiles: a
            tile containing ``axis`` scales with it (its ``∏(other dims)`` adds to ``scale``), a tile
            without ``axis`` is constant (adds to ``flat``). A ``None`` dim is a size-1 broadcast.
            The tiles already ARE the resident set (loop-carried accumulators captured inline at
            real shape), so no separate accumulator sum is needed."""
            scale = 0
            flat = 0
            for tile in tiles:
                contains_axis = axis in tile
                prod = 1
                for d in tile:
                    if d is None or d == axis:
                        continue
                    prod *= conservatively_large_tile_width(d)
                if contains_axis:
                    scale += prod
                else:
                    flat += prod
            return max(1, scale), flat

        def conservatively_large_tile_width(bid: int) -> int:
            """One resident dim's width for the footprint bound: its SEATED width if already chosen,
            else its full extent. The full-extent fallback is safe BY SEATING ORDER, not a blind
            assumption — grid axes are seated first (the pass-1 preamble above), and a not-yet-seated
            *reduction* dim is later in the sizing ``order`` below, so over-approximating it at full
            extent only makes the footprint LARGER, keeping the axis currently being sized
            conservative (it can only end up smaller/safer, never over-sized into a spill). NB: the
            footprint is therefore ORDER-DEPENDENT (a later-sized reduction sees an earlier one at its
            seated width, but not vice-versa) — the ``order`` sort below is load-bearing for
            correctness, not cosmetic."""
            return max(1, seated.get(bid, extent_of(bid)))

        # The persistence-hold ceiling (used once, at ``expand_to_persist`` in the loop below; kept
        # here as it is loop-invariant — keyed on the primary ``pd``). Selects the SMALL vs BIG
        # bucket via ``_has_store_only_row_reread`` (an admitted proxy — see that method and
        # PERSIST_HOLD_MAX_BYTES).
        hold_ceiling = (
            cls.USER_TILE_PERSIST_HOLD_MAX_BYTES
            if cls._has_store_only_row_reread(spec, pd)
            else cls.PERSIST_HOLD_MAX_BYTES
        )

        for g in kf.coresidency_groups:
            descs = [kf.reductions[i] for i in g.descriptor_indices]
            sized = [d for d in descs if d.category in SIZED_REDUCTION_CATEGORIES]
            if not sized:
                continue
            tiles = g.live_tiles

            # ---- PASS 1: seat the reductions (full-extent -> user-tile -> grid-tile) against the
            # group's live-tile footprint with the grid axes at their floor. ----
            order = sorted(
                sized,
                key=lambda d: (
                    0
                    if d.category in FULL_EXTENT_CATEGORIES
                    else (1 if d.category is ReductionCategory.USER_TILE else 2),
                    -d.size_hint,
                ),
            )
            # ``order`` is ``sized`` (SIZED_REDUCTION_CATEGORIES only): FULL_SLICE / FULL_GRID /
            # USER_TILE. A GRID_TILE reduction (jsd's grid amax) is NOT sized here — it is a grid
            # axis, seated at its floor in the grid loop above and widened in PASS 2 like any grid
            # row. So this loop never sees a GRID_TILE.
            for d in order:
                raw_ext = d.size_hint  # the true reduction extent (NOT pow2-padded)
                ext = extent_of(d.block_id)  # pow2-padded — the seated tile width
                materialized_full_width = (
                    d.category is ReductionCategory.FULL_SLICE
                    and d.block_id not in valid
                    and d.block_id not in spec.reduction_loops.valid_block_ids()
                )
                if d.category is ReductionCategory.FULL_GRID or materialized_full_width:
                    # FULL_GRID (cdiv == 1) or a materialized full-width FULL_SLICE (the roller
                    # declined to roll it and it has no tunable block_sizes slot — e.g. a
                    # grad-parameter ``grad_weight[N]`` accumulator axis, or a specialized
                    # ``group_size``): the whole axis is one program's tile, full-extent resident by
                    # definition. Seat at the full extent, never chunk it through the byte budget — it
                    # cannot be split across programs and has nowhere to emit a chunk. Seating it
                    # full-width (not chunked to 1) is what lets the co-resident inner tile see the
                    # real N (else the inner tile reads N as 1 and grows to full extent — a spill).
                    seated[d.block_id] = ext
                    if d.block_id == pd.block_id:
                        primary_r_block = ext
                        # Seated at its full extent (r == ext), so it is persistent under the same
                        # ``persistent = (r >= ext)`` rule the normal sizing path uses.
                        persistent = True
                    if d.block_id in valid:
                        block_sizes_red_values[d.block_id] = ext
                    continue
                # Resident bytes(R) = itemsize × (scale × R + flat) over the live tiles. A reduction
                # axis is tiled the same width everywhere it appears, so it must fit the heaviest
                # co-residency group that spans it — take the footprint from the max-``scale`` group
                # over ``d.block_id``, not just this descriptor's own group. ``flat`` is taken from
                # that same max group (mixing terms across groups breaks the chunk arithmetic).
                scale, flat = cls._max_group_footprint(
                    kf, d.block_id, footprint_terms, default_tiles=tiles
                )
                # Size a streamed/chunked R from the byte budget first: the largest pow2 R whose
                # resident bytes fit, solving ``itemsize × (scale × R + flat) <= budget`` for R (the
                # constant term is subtracted, not divided), capped by LOOPED_CHUNK and the extent. A
                # carried reduction sizes against the tighter carried budget; a non-carried inner tile
                # against ROW.
                avail = persist_budget_for(d) // itemsize - flat
                byte_budget = _pp2(max(1, avail // scale))
                r = max(1, min(cls.LOOPED_CHUNK, byte_budget, ext))
                # THEN EXPAND TO PERSISTENT: lift R to the full extent iff the row is re-read (a
                # persistent pass fuses reduce+apply to one HBM load) AND there is no carried 2-D
                # tile (a carried tile is held resident the whole loop — it chunks, never persists)
                # AND the extent clears the per-program element limit AND the single resident tile
                # fits the persist ``hold_ceiling`` (apply-reread-keyed, computed above). The byte
                # test uses the RAW extent (true resident element count, not pow2-padded).
                element_cap = env.backend.max_tensor_numel
                expand_to_persist = (
                    d.row_reread
                    and d.carried_2d_count == 0
                    and (element_cap is None or raw_ext <= element_cap)
                    and itemsize * (scale * raw_ext + flat) <= hold_ceiling
                )
                if expand_to_persist:
                    r = ext
                seated[d.block_id] = r
                # THREE independent routing checks (the block_sizes and reduction_loops namespaces
                # are DISJOINT — an axis is a ``block_sizes`` tile XOR a rolled ``reduction_loops``
                # axis, never both — so these are plain ``if``s, not an if/elif chain):
                # (A) the PRIMARY's scalar levers (num_warps ramp + standard-track reduction_loops).
                if d.block_id == pd.block_id:
                    primary_r_block = r
                    persistent = r >= ext and d.category in FULL_EXTENT_CATEGORIES
                # (B) a tunable ``block_sizes`` reduction (user-tiled) -> its block_sizes slot.
                if d.block_id in valid:
                    block_sizes_red_values[d.block_id] = r
                # (C) a ROLLED NON-primary reduction -> surface its size for the standard track's
                # reduction_loops emission. ``!= pd.block_id`` excludes the ROLLED PRIMARY (whose
                # size is emitted via ``primary_r_block`` in (A) instead — it would otherwise be
                # double-routed here). Only reached by a kernel that rolls >1 reduction.
                if (
                    d.block_id != pd.block_id
                    and d.block_id in spec.reduction_loops.valid_block_ids()
                ):
                    rolled_loop_sizes[d.block_id] = (
                        r,
                        r >= ext and d.category in FULL_EXTENT_CATEGORIES,
                    )

            # ---- PASS 2: the grid-M rows take the remainder (widen / floor / collapse). ----
            for mbid in sorted(grid_ids):
                if mbid not in valid:
                    continue  # a grid-PINNED axis (FixedBlockSizeSource) -> fixed, not sized.
                ext = extent_of(mbid)
                floor = cls._block_floor(
                    cast(
                        "BlockSizeSpec",
                        spec.block_sizes[spec.block_sizes.block_id_to_index(mbid)],
                    )
                )
                if mbid not in resident_block_ids:
                    # a sequential cross-grid reduction loop (grad-param .sum(0)): in NO live tile ->
                    # NOT resident, holds no bytes. The byte budget cannot size it; raise the floor
                    # to ~1 SM wave to collapse the cross-grid finalize.
                    collapse = _np2(max(1, grid_rows // num_sm)) if grid_rows > 0 else 1
                    blk = max(floor, min(collapse, ext))
                elif carried_kernel:
                    # Register-occupancy guard (see ``carried_kernel`` above): the pinned
                    # ``[grid_M, R]`` accumulator makes widening the grid trip the CTA-per-SM
                    # occupancy cliff, which the leftover-byte widen and program-count ``occ_widen``
                    # cannot see. So a carried kernel keeps its resident grid at FLOOR.
                    blk = floor
                else:
                    # resident parallel rows: widen into the byte remainder (same faithful
                    # ``scale × block + flat`` footprint over the live tiles — a wider grid row
                    # scales every tile CONTAINING the grid axis), capped by occupancy (keep the
                    # post-widen grid >= num_sm·MIN_WAVES), a diminishing-returns ROWS ceiling, and
                    # the extent; floors when the budget is full.
                    scale_w, flat_w = footprint_terms(tiles, mbid)
                    avail_w = persist_budget_for(pd) // itemsize - flat_w
                    byte_widen = _pp2(max(1, avail_w // scale_w))
                    if grid_rows > 0:
                        occ_widen = _pp2(max(1, grid_rows // occ_floor))
                    else:
                        occ_widen = (
                            1  # dynamic grid -> no compile-time occupancy -> no widen
                        )
                    # ROWS ceiling: batching more than WIDEN_MAX_ROWS reduction ROWS/program only
                    # trades away grid parallelism for a resident-row reduction (softmax/rms_norm:
                    # memory-bound, does not amortize past ~8 rows). Does NOT apply when the primary
                    # is FULL_GRID (the grid axis batches tiny grid-resident per-group reductions —
                    # per_token_group's groups_per_row — which wants the wide occupancy-bound widen).
                    rows_ceiling = (
                        ext
                        if pd.category is ReductionCategory.FULL_GRID
                        else cls.WIDEN_MAX_ROWS
                    )
                    blk = max(floor, min(byte_widen, occ_widen, rows_ceiling, ext))
                seated[mbid] = blk
                sizes[mbid] = blk

        # ---- the non-reduction / independent loops LAST (own budget vs the headroom) ----
        # welford's normalize loop / rms_norm_per_block's groups_per_row. Co-resident with nothing
        # in a group's reduction tile (a separate sequential pass), so each gets a FRESH budget
        # against its own extent capped by the streamed ROW budget.
        loop_budget = _pp2(max(1, cls.ROW_PERSIST_MAX_BYTES // itemsize))
        # Optional tighter cap for a non-reduction apply loop (None on the base; set by sm100).
        loop_cap = cls.non_reduction_loop_block_cap(spec, pd)
        for i in range(len(spec.block_sizes)):
            bs_spec = cast("BlockSizeSpec", spec.block_sizes[i])
            bid = bs_spec.block_id
            if bid in block_sizes_red_values or bid in grid_ids or bid in reduction_ids:
                continue
            if bid in non_reduction_loop_ids or bid not in seated:
                # a non-reduction apply loop OR an independent standalone tiled loop: size it to
                # its own extent capped by the headroom (flooring it to 1 would serialize the pass).
                budget = loop_budget
                if loop_cap is not None and bid in non_reduction_loop_ids:
                    budget = min(budget, loop_cap)
                sizes[bid] = max(1, min(extent_of(bid), budget))

        # ---- assemble the full block_sizes vector ----
        block_sizes: list[int] = []
        for i in range(len(spec.block_sizes)):
            bs_spec = cast("BlockSizeSpec", spec.block_sizes[i])
            bid = bs_spec.block_id
            if bid in sizes:
                block_sizes.append(sizes[bid])
            elif bid in block_sizes_red_values:
                block_sizes.append(block_sizes_red_values[bid])
            else:
                block_sizes.append(cls._block_floor(bs_spec))

        return _TileAllocation(
            block_sizes=block_sizes,
            block_sizes_red_values=block_sizes_red_values,
            primary_r_block=primary_r_block,
            persistent=persistent,
            rolled_loop_sizes=rolled_loop_sizes,
        )


class TritonStandardReductionHeuristic(_TritonReductionSeedBase):
    """standard (Helion-rolled rdim) inner-reduction seed: Helion rolls the reduction axis
    into a ``reduction_loops`` loop from a single ``.sum(-1)``-style op — sum, long_sum,
    rms_norm, layer_norm, softmax-row, cross_entropy. Triton analog of
    ``CuteReductionTileHeuristic`` (keeps its registry name), deepening the original
    one-row/persistent/``['last']`` seed with the num_warps ramp, persistent-vs-looped,
    and per-slot eviction.

    Gated by ``_triton_reduction_eligible`` (standard track) — broader than upstream
    ``is_canonical_row_reduction`` (also multi-axis rollable rows and raised-``autotuner_min``
    large-M shapes). Off sm90 the H100-tuned levers are unvalidated, so it falls back to
    ``_narrow_seed`` (pre-existing behavior preserved).
    """

    name = "triton_reduction_tile"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if not _triton_reduction_eligible(env, device_ir):
            return False
        pd = _primary_descriptor_selected(env)
        return pd is not None and _is_standard_reduction(pd)

    @classmethod
    def _narrow_seed(cls, env: CompileEnvironment) -> Config:
        """The upstream conservative standard seed (one row/program, single persistent pass,
        ``['last']`` eviction where supported). A verbatim port used off sm90 so non-sm90
        behavior is unchanged.
        """
        spec = env.config_spec
        seed: dict[str, Any] = {
            "block_sizes": [1],
            "reduction_loops": [None],
        }
        # Emit 'last' only where the backend supports it; backends that restrict
        # eviction to ("",) keep the spec default so the seed stays valid.
        eviction = spec.load_eviction_policies
        if (
            eviction.length
            and isinstance(eviction.inner, EnumFragment)
            and "last" in eviction.inner.choices
        ):
            seed["load_eviction_policies"] = ["last"] * eviction.length
        return Config(**seed)

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            # sm100 has its own subclass (``TritonStandardReductionHeuristicSM100``); decline here so
            # exactly one reduction seed fires on B200.
            if matches_hardware(env, (("cuda", "sm100"),)):
                return None
            # Off any other target: keep the upstream conservative seed.
            return cls._narrow_seed(env)
        spec = env.config_spec
        pd = _primary_descriptor_selected(env)
        if pd is None:
            return None
        # The allocator sizes every axis from the per-co-residency-group budget: the reduction
        # chunk(s), the grid M (the remainder — widen / floor / collapse), and the apply/independent
        # loops, in one pass. The standard track maps that sizing onto the rolled ``reduction_loops``
        # knob + the num_warps ramp + eviction below (emission routing only).
        alloc = cls.size_reduction_tiles(env, spec, device_ir, pd)
        block_sizes = alloc.block_sizes
        r_block, persistent = alloc.primary_r_block, alloc.persistent
        num_warps = cls._num_warps(pd)
        # Grad-parameter M-collapse warp floor: a kernel that reduces its grid-M axis away (finalized
        # by a later ``.sum(0)`` — a grid block_id in no live tile) batches many M-rows per program
        # and accumulates a wide ``[inner, N]`` gradient. That cross-warp-parallelizable work wants
        # >=8 warps even when the primary reduction's extent is small. A floor, so it never lowers a
        # large-rdim ramp. Independent of co-residency, so not gated on a co-resident sibling.
        if cls._has_reduced_away_grid(spec):
            num_warps = max(8, num_warps)

        # standard rides persistent-vs-looped on the rolled ``reduction_loops`` knob (the primary
        # rdim is NOT a block_sizes entry). MATERIALIZED rdim (rms/ln/instance bwd, the roller
        # declined to roll it): emit an EMPTY reduction_loops -- already full-width persistent, and
        # a length-1 list would fail normalize against the 0-length spec.
        is_materialized = pd.block_id not in spec.reduction_loops.valid_block_ids()
        reduction_loops: list[int | None]
        if is_materialized:
            reduction_loops = []
        elif len(spec.reduction_loops) <= 1:
            # Single rolled reduction (the common case).
            reduction_loops = [None] if persistent else [r_block]
        else:
            # Multiple rolled reductions (e.g. two sequential rolled reductions in separate graphs).
            # One ``reduction_loops`` entry per spec in spec order: the primary spec uses
            # (r_block, persistent); the other rolled specs use ``alloc.rolled_loop_sizes`` (each
            # sized against its own extent — a rolled axis has no block_sizes slot, so the allocator
            # surfaces it here rather than in block_sizes_red_values).
            reduction_loops = []
            for rl_spec in spec.reduction_loops:
                bid = rl_spec.block_ids[0]
                if bid == pd.block_id:
                    reduction_loops.append(None if persistent else r_block)
                else:
                    rb, pers = alloc.rolled_loop_sizes[bid]
                    reduction_loops.append(None if pers else rb)
        seed: dict[str, Any] = {
            "block_sizes": block_sizes,
            "reduction_loops": reduction_loops,
            "num_warps": num_warps,
            "num_stages": 1,
            # 'flat': these reductions are grid-saturated at the M-grid.
            "pid_type": "flat",
        }
        # Eviction: a streamed input -> 'first' everywhere; a re-read row reloaded across a
        # grid-COLLAPSE loop -> pin it 'last' (first load), rest 'first'. Gated on
        # ``_has_reduced_away_grid`` (the grad-parameter ``.sum(0)`` M-collapse idiom: the program
        # batches many M-rows and re-fetches the row from L2 each row, so pinning it pays), not on
        # ``not persistent`` — a single fused persistent row does not reload from L2, so pinning
        # there only oversubscribes L2 and evicts store lines. Whether the row reloads is a
        # structural property, not a byte threshold, so ``_has_reduced_away_grid`` is the
        # discriminator. (A ``num_load == 1`` kernel hits the stream branch first.)
        evict = None
        if pd.num_load == 1:
            evict = cls._eviction_policies(env, "stream")
        elif pd.row_reread and cls._has_reduced_away_grid(spec):
            # Re-read row's eviction slot read directly from the descriptor (its load's
            # MemoryOpFact.eviction_index), not a per-config codegen re-walk.
            evict = cls._eviction_policies(env, "reread", pd.reread_eviction_index)
        if evict is not None:
            seed["load_eviction_policies"] = evict
        return Config(**seed)


class TritonUserTiledReductionHeuristic(_TritonReductionSeedBase):
    """user-tiled inner-reduction seed: fires when the user hand-writes the ``hl.tile`` loop
    over the reduction axis (so the rdim is an ordinary ``block_sizes`` entry, e.g.
    ``hl.tile(n, block_size=R_BLOCK)``), which the upstream gate rejects entirely.

    Every axis (the reduction r_block(s), the grid rows, the apply loops) is sized by the shared
    :meth:`size_reduction_tiles` ONE budget allocator — there are NO per-band branches. The kernel
    families this track covers (plain user-tiled softmax, carried-2-D kl_div/jsd, reduce-then-apply
    welford, grad-parameter bias_grad/dyt) differ only in their Stage-1 facts (carried accumulators,
    non-reduction loops, materialized features), which the budget consumes uniformly; the
    floor-vs-resident and chunk-vs-persistent decisions are budget OUTCOMES. This track maps the
    allocation onto its knobs (every reduction axis is a ``block_sizes`` entry; no
    ``reduction_loops``) + num_warps + reread eviction below.
    """

    name = "triton_reduction_user_tile"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if not _triton_reduction_eligible(env, device_ir):
            return False
        pd = _primary_descriptor_selected(env)
        return pd is not None and not _is_standard_reduction(pd)

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            # Off sm90: upstream never fired on user-tiled, so no prior seed to preserve. Decline.
            return None
        spec = env.config_spec
        pd = _primary_descriptor_selected(env)
        if pd is None:
            return None
        # The allocator sizes every axis from the per-co-residency-group budget: the user-tiled
        # reduction chunk(s) on their block_sizes slots, the grid M (the remainder), and the apply
        # loops, in one pass. The user-tiled track maps that sizing onto num_warps + eviction below
        # (no reduction_loops knob; the rdim rides a block_sizes entry).
        alloc = cls.size_reduction_tiles(env, spec, device_ir, pd)
        block_sizes = alloc.block_sizes
        num_warps = cls._num_warps(pd)
        non_reduction_loop_ids = set(cls._non_reduction_loop_ids(spec))
        seed: dict[str, Any] = {
            "block_sizes": block_sizes,
            "num_warps": num_warps,
            "num_stages": 1,
            "pid_type": "flat",  # see the standard branch.
        }
        # Reread eviction: keep the re-read row L2-resident ('last' on its load slot) whenever
        # it is re-read — welford (reduce-then-apply across combine + normalize) AND plain
        # user-tiled (softmax_two_pass loads x twice). Applies even when PERSISTENT: the second
        # pass still re-fetches x from HBM (profiler-confirmed), so 'last' cuts that re-read
        # traffic. kl_div/jsd (row_reread=False) unaffected.
        if non_reduction_loop_ids or pd.row_reread:
            # Re-read row's eviction slot read directly from the descriptor (its load's
            # MemoryOpFact.eviction_index), not a per-config codegen re-walk.
            ev = cls._eviction_policies(env, "reread", pd.reread_eviction_index)
            if ev is not None:
                seed["load_eviction_policies"] = ev
        return Config(**seed)


def _config_with_num_warps(cfg: Config, num_warps: int) -> Config:
    """Return a copy of ``cfg`` with ``num_warps`` overridden (the reduction seeds always set
    num_warps, so this replaces the existing value)."""
    merged: dict[str, Any] = {**cfg.config, "num_warps": num_warps}
    return Config(**merged)


# ============================ sm100 (B200) dedicated subclasses ============================ #
# Re-target the sm90 reduction seeds at sm100 via a subclass that overrides only the hardware gate +
# B200 constants; the sm90/H100 emit is a separate class + gate, so it stays frozen.
class _TritonReductionSeedSM100(_TritonReductionSeedBase):
    """sm100 (B200) constant/gate carrier for the two reduction seed tracks. Overrides the hardware
    gate + ``promote_seed_to_default`` and re-tunes constants (as class attributes) only where a B200
    measurement demands it. Not registered; the two concrete subclasses below are."""

    promote_seed_to_default = True
    HARDWARE_TARGETS = (("cuda", "sm100"),)
    # --- B200 constant overrides (re-tuned during the climb; unset = direct port of H100) ---
    # Load-traffic ceiling (bytes) below which a light streamed row drops to nw8 (see _b200_num_warps).
    NW8_MAX_ROW_TRAFFIC = 64 * 1024
    # Element cap for a non-reduction apply loop (see non_reduction_loop_block_cap).
    NON_REDUCTION_LOOP_MAX_ELEMS = 4096

    @classmethod
    def non_reduction_loop_block_cap(
        cls, spec: ConfigSpec, pd: ReductionDescriptor
    ) -> int | None:
        # Cap EVERY non-reduction apply loop (relieves register pressure on B200).
        return cls.NON_REDUCTION_LOOP_MAX_ELEMS

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        # Fire the inherited rich seed ONLY on sm100; decline elsewhere so this promoted class never
        # overrides the frozen sm90/H100 default.
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            return None
        cfg = super().get_seed_config(env, device_ir)
        if cfg is None:
            return None
        # Re-tune num_warps by the load-traffic key (both tracks, one place); see _b200_num_warps.
        pd = _primary_descriptor_selected(env)
        if pd is not None:
            nw = cls._b200_num_warps(env.config_spec, pd, cfg)
            if nw is not None:
                cfg = _config_with_num_warps(cfg, nw)
        return cfg

    @classmethod
    def _b200_num_warps(
        cls, spec: ConfigSpec, pd: ReductionDescriptor, cfg: Config
    ) -> int | None:
        """The B200 warp count for a light-traffic PERSISTENT streamed row (8, or 4 at small extent),
        or None to leave the base ramp untouched. Purely additive: only lowers, never raises."""
        # Skip M-collapse (grad-parameter .sum(0)): a cross-warp accumulate, not a streamed row.
        if cls._has_reduced_away_grid(spec):
            return None
        # Skip reduce-then-apply (welford / rms_norm_per_block): its reread lives on the non-reduction
        # loop, not in ``num_load``, so the traffic key can't see it.
        if cls._non_reduction_loop_ids(spec):
            return None
        # Restrict to a single sized reduction: a second one adds cross-warp compute the traffic key
        # can't see, with a non-monotonic warp optimum.
        kf = spec.reduction_kernel_fact
        if (
            kf is None
            or sum(1 for d in kf.reductions if d.category in SIZED_REDUCTION_CATEGORIES)
            != 1
        ):
            return None
        # Skip LOOPED reductions (positive ``reduction_loops`` chunk): they keep the base ramp.
        loops = cfg.config.get("reduction_loops")
        if isinstance(loops, (list, tuple)) and any(isinstance(x, int) for x in loops):
            return None
        # Load traffic per row = elems × load-width × #loads.
        traffic = pd.size_hint * max(1, pd.input_load_itemsize) * max(1, pd.num_load)
        if traffic <= cls.NW8_MAX_ROW_TRAFFIC:
            # keep the base's small-extent 4-warp floor (<=1024 elems); else 8.
            return 4 if pd.size_hint <= 1024 else 8
        return None  # heavy-traffic streamed row -> base ramp (16/32) is right


class TritonStandardReductionHeuristicSM100(
    _TritonReductionSeedSM100, TritonStandardReductionHeuristic
):
    """standard (Helion-rolled rdim) inner-reduction seed for sm100/B200: the rich
    :class:`TritonStandardReductionHeuristic` allocator with B200 constants from
    :class:`_TritonReductionSeedSM100`."""

    name = "triton_reduction_tile_sm100"


class TritonUserTiledReductionHeuristicSM100(
    _TritonReductionSeedSM100, TritonUserTiledReductionHeuristic
):
    """user-tiled inner-reduction seed for sm100/B200: the rich
    :class:`TritonUserTiledReductionHeuristic` allocator with B200 constants from
    :class:`_TritonReductionSeedSM100`."""

    name = "triton_reduction_user_tile_sm100"


class TritonMatmulReductionEpilogueHeuristic(AutotunerHeuristic):
    """Seed for a fused matmul + reduction-over-output-axis epilogue (matmul_rms_norm /
    matmul_layernorm / matmul_softmax / matmul_l2_normalize / matmul_sum / ...): a single
    grid loop over M does an inner K-loop ``addmm`` into a register-resident ``[M_BLOCK, N]``
    fp32 accumulator, then reduces over the matmul's N (output) axis on that accumulator. N
    is ``hl.specialize``'d (never tiled), so BOTH the ``[M_BLOCK, N]`` accumulator AND the
    ``[K_BLOCK, N]`` y-operand tile scale with N -> the kernel is SMEM/register-footprint
    bound and the win regime is small N (where a productive tile fits).

    Fires on the composed ``MatmulWithReductionEpilogueFact`` (a MatmulFact + an epilogue
    ReductionFact in one kernel) -- never on a pure matmul or a pure reduction, so those stay
    byte-identical. This sizes M_BLOCK by the resident fp32-accumulator footprint.
    """

    name = "triton_matmul_reduction_epilogue"
    backend = "triton"
    HARDWARE_TARGETS = (("cuda", "sm90"),)

    # The resident [M_BLOCK, N] fp32 accumulator must fit a per-program byte budget; M_BLOCK is the
    # largest pow2 under it, capped at MAX_M_BLOCK (an occupancy/register ceiling). ~128 KiB gives
    # the answer-key tile: M_BLOCK=64 at N<=512, 32 at N=1024, 16 at N=2048 (where the win vanishes).
    ACC_BUDGET_BYTES = 131072
    MAX_M_BLOCK = 64
    # Inner K tile (min 16 by the matmul min_dot_size; normalize clamps to <=K).
    K_BLOCK = 32
    # num_stages: pipeline the K-loop addmm (a matmul knob; the answer key uses 3).
    NUM_STAGES = 3
    # num_warps ramps with the resident accumulator elements (M_BLOCK * N).
    NUM_WARPS_ELEM_BREAK = 16384
    # Staged matmul-operand SMEM budget (sm90/H100 has ~227 KiB/SM). The [K_BLOCK, N]
    # y-operand x num_stages must fit this; past it the shipped [.,32]/st3 OOMs.
    # Calibrated to the measured feasibility boundary (KB=32/st3 fits N<=1024 bf16 /
    # N<=512 fp32; KB=16/st3 fits N<=2048 / N<=1024). The byte-cap (get_seed_config)
    # drops K_BLOCK 32->16 FIRST -- it halves the staged bytes AND avoids the measured
    # non-monotonic KB=32 ptxas cliffs -- keeping full stages; only past KB=16/st3 does
    # it drop num_stages (cliff-free once KB=16).
    SMEM_STAGED_BUDGET_BYTES = 196608  # 192 KiB

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            return False
        # Resident-only: fire when the composed fact's N axis is hl.specialize'd
        # (n_block_id is None). The looped/tiled-N shape is left to the default config.
        facts = env.config_spec.matmul_reduction_epilogue_facts
        return len(facts) == 1 and facts[0].matmul.n_block_id is None

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            return None
        from ..._utils import prev_power_of_2

        spec = env.config_spec
        fact = spec.matmul_reduction_epilogue_facts[0]
        n = max(1, fact.n_extent)
        # Per-program row ceiling: MAX_M_BLOCK at 2 bytes (bf16/fp16 tensor core),
        # scaled DOWN as the input dtype widens (fp32 = ~2x regs/elem -> //2 -> 32).
        # The factor only lowers the ceiling; MAX_M_BLOCK is the hard occupancy cap,
        # so a 1-byte dtype (fp8) is pinned to it by min(), not pushed above it.
        input_itemsize = fact.matmul.lhs_dtype.itemsize
        max_m = max(1, min(cls.MAX_M_BLOCK, cls.MAX_M_BLOCK * 2 // input_itemsize))

        # Resident N (hl.specialize'd, n_block_id is None -- guaranteed by is_eligible):
        # M_BLOCK = largest pow2 [M_BLOCK, N] fp32 accumulator under the ACC budget, capped
        # at max_m. The staged [K_BLOCK, N] operand is bounded separately by the SMEM
        # byte-cap below, which is what sets the feasible-N ceiling.
        m_block = max(
            1, min(max_m, prev_power_of_2(max(1, cls.ACC_BUDGET_BYTES // (n * 4))))
        )
        num_warps = 4 if m_block * n <= cls.NUM_WARPS_ELEM_BREAK else 8

        # K_BLOCK + num_stages via a priority-ordered footprint byte-cap. The staged
        # [K_BLOCK, N] y-operand (x num_stages) must fit SMEM; in the shipped small-N
        # regime [K_BLOCK=32, num_stages=3] fits, but past it (large N) it overflows.
        # Reduce K_BLOCK 32->16 FIRST -- it halves the staged bytes AND avoids the measured
        # non-monotonic K_BLOCK=32 ptxas cliffs, while keeping full stages -- then, only if
        # [16, st=3] still overflows (very large N), drop num_stages (cliff-free once
        # K_BLOCK=16). This EXTENDS the feasible N (KB=32/st3 to N<=1024 bf16, then KB=16/st3
        # to N<=2048) instead of OOMing into the bad default; small-N stays byte-identical.
        k_hint = next(
            (
                cast("BlockSizeSpec", spec.block_sizes[i]).size_hint
                for i in range(len(spec.block_sizes))
                if cast("BlockSizeSpec", spec.block_sizes[i]).block_id
                == fact.k_block_id
            ),
            cls.K_BLOCK,
        )
        k_block = min(cls.K_BLOCK, k_hint)
        num_stages = cls.NUM_STAGES
        if num_stages * k_block * n * input_itemsize > cls.SMEM_STAGED_BUDGET_BYTES:
            k_block = min(k_block, 16)
            while (
                num_stages > 1
                and num_stages * k_block * n * input_itemsize
                > cls.SMEM_STAGED_BUDGET_BYTES
            ):
                num_stages -= 1

        block_sizes: list[int] = []
        for i in range(len(spec.block_sizes)):
            bs_spec = cast("BlockSizeSpec", spec.block_sizes[i])
            bid = bs_spec.block_id
            if bid == fact.m_block_id:
                block_sizes.append(max(bs_spec.min_size, m_block))
            elif bid == fact.k_block_id:
                block_sizes.append(
                    max(bs_spec.min_size, min(k_block, bs_spec.size_hint))
                )
            else:
                block_sizes.append(max(1, bs_spec.min_size, bs_spec.autotuner_min))

        seed: dict[str, Any] = {
            "block_sizes": block_sizes,
            # The epilogue reduction is materialized on the resident accumulator, so
            # there is no reduction_loops knob to set.
            "reduction_loops": [],
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        return Config(**seed)
