from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

from ...autotuner.config_fragment import EnumFragment
from ...runtime.config import Config
from .common import REDUCTION_TARGET_NAMES
from .common import clamp_block_size_targets
from .common import matches_hardware
from .common import op_name_parts
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...autotuner.config_spec import BlockSizeSpec
    from ...autotuner.config_spec import ConfigSpec
    from ...autotuner.config_spec import MatmulFact
    from ...autotuner.config_spec import ReductionFact
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


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


class TritonSplitJoinRotateHeuristic(AutotunerHeuristic):
    """Seed all-ones ``block_sizes`` for split/join rotate kernels (rope).

    These kernels load a large untiled inner slab per program, so tiling any
    outer dim past 1 only wastes work and overflows Triton's block-numel cap.
    Detected by ``hl.split`` + ``hl.join`` with no matmul and no reduction op.
    """

    name = "triton_split_join_rotate"
    backend = "triton"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        # A GEMM (even fused) is not a rope-style rotate.
        if env.config_spec.matmul_facts:
            return False
        if not env.config_spec.block_sizes:
            return False
        # Local import avoids a circular import at module load
        # (runtime.kernel -> autotuner_heuristics -> helion.language).
        from ...language import join as hl_join
        from ...language import split as hl_split

        saw_split = False
        saw_join = False
        for graph_info in device_ir.graphs:
            for node in graph_info.graph.nodes:
                if node.op != "call_function":
                    continue
                target = node.target
                if target is hl_split:
                    saw_split = True
                elif target is hl_join:
                    saw_join = True
                elif op_name_parts(target) & REDUCTION_TARGET_NAMES:
                    # Fused reduction → not a pure rotate; keep its own tiling.
                    return False
        return saw_split and saw_join

    @classmethod
    def get_seed_config(cls, env: CompileEnvironment, device_ir: DeviceIR) -> Config:
        return Config(block_sizes=[1] * len(env.config_spec.block_sizes))


class TritonPointwiseSeedHeuristic(AutotunerHeuristic):
    """Seed a bandwidth-saturating tile for PURE elementwise/pointwise kernels.

    A pointwise kernel reads its inputs, computes, and writes its outputs with no
    reduction / matmul / loop-carried accumulator — it is BANDWIDTH-bound. The compiler
    default tiles it at ``block_size=32`` (``BlockSizeSpec._fragment``: ``total_ndim <= 2
    and reduction_numel <= 128``), moving only ~10% of HBM (~5-7x slower than a saturating
    tile; measured headroom in ``_lab/pointwise``). This seed sizes the tile to (a) a byte
    budget that saturates HBM and (b) the grid occupancy (size_hint-aware), keyed on the
    derived ``PointwiseElementwiseFact`` (bytes/elem + total numel) — NEVER on the
    activation identity or a dtype literal. Fires on the presence of that fact, which is
    built only on the ABSENCE of the reduction/matmul/accumulator facts (disjointness rule),
    so it never claws a reducing kernel into the pointwise track.
    """

    name = "triton_pointwise"
    backend = "triton"

    # Hill-climbed constants (see _lab/pointwise/NOTEBOOK.md). TILE_BYTES=8192 gives a ~1024-elem
    # tile for traffic-3 (swiglu/geglu/residual_add) and ~2048 for traffic-2 (relu²/bias_gelu) —
    # the robust zone: 1-D kernels are flat across 1-8K, but N-D/traffic-2 kernels (bias_gelu) lose
    # ~5-25% at 4096 vs 2048, so the smaller budget lifts the worst kernel with no 1-D regression.
    TILE_BYTES = 8192  # target HBM bytes moved per tile
    MIN_WAVES = 8  # grid >= num_sm * MIN_WAVES (size_hint-aware grid floor)
    BLOCK_FLOOR = 256  # never regress toward the bs=32 default
    # Per-program register/working-set ceiling (fp32-compute bytes of the resident tile) before it
    # spills / overflows Triton's block-numel cap. Not hill-climbed — bounded by two constraints
    # with a wide safe window: it must NOT bind the flat family (flat reg_bytes/elem ~ 12-16 * its
    # budget tile ~2048 => stay above ~32 KB) and it MUST bind a heavy partially-tiled slab (rope's
    # fp32 slab is ~80+ KB/elem => stay below ~170 KB). 64 KB sits in the middle and matches the
    # measured spill cliff (~block 8-16 at head_dim=256). Only the register cap reads it.
    REGISTER_BYTES = 65536
    # num_warps ramp (arithmetic-intensity keyed). The compiler default is 4; a transcendental-heavy
    # tile is latency-bound and wants more warps to hide the SFU latency (MEASURED: 4 SFU ops -> w8,
    # >=9 SFU -> w16; the flat GLU/activation family has 0-1 SFU -> stays w4, byte-identical). Over-
    # warping a SMALL tile starves each warp of work and regresses ([1,256] w16 = 0.5x measured), so
    # the ramp is capped at tile_numel // ELEMS_PER_WARP warps.
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
        bandwidth_bytes_per_elem = max(1, fact.bandwidth_bytes_per_elem)
        register_bytes_per_elem = max(1, fact.register_bytes_per_elem)
        # Two orthogonal per-program caps, both counting the FULL per-tiled-element cost (the untiled
        # inner slab folded in via accessed_numel//total_numel):
        #  - budget_target: HBM bandwidth — how many tiled elements make a coalesced, saturating
        #    program (storage-width bytes). This is what the flat family is tuned on.
        #  - reg_cap: register/occupancy — how many fit before the fp32-compute working set spills
        #    or overflows Triton's block-numel cap (compute-width bytes).
        # For a flat kernel both are large (small slab) so the tile is bandwidth-sized as before; for
        # a partially-tiled kernel (rope: seq tiled but the whole heads*head_dim slab is fp32-resident
        # per program) reg_cap (and budget_target) collapse to ~1, so it is not tiled past ~1 instead
        # of the old model's [1, 256] that overflowed the block/register cap.
        budget_target = max(1, cls.TILE_BYTES // bandwidth_bytes_per_elem)
        reg_cap = max(1, cls.REGISTER_BYTES // register_bytes_per_elem)
        # size_hint-aware: cap the tile so the grid keeps the SMs busy on small problems.
        occ_cap = max(1, fact.total_numel // (num_sm * cls.MIN_WAVES))
        target = max(1, min(budget_target, reg_cap, occ_cap))
        # Anti-undershoot floor, capped by the REGISTER budget only (NOT budget_target, NOT the
        # occ-capped target). The floor exists to keep a coalesced per-operand run; it must yield
        # ONLY when the program is genuinely too heavy to hold BLOCK_FLOOR elements — i.e. when an
        # untiled slab makes reg_cap tiny (rope: reg_cap≈1 → floor 1). It must NOT yield merely
        # because budget_target is small: a HIGH-FAN-IN kernel (many cheap full-extent operands)
        # has a small budget_target (TILE_BYTES split across all operands) but a LARGE reg_cap and
        # NO slab, so it can and should hold a BLOCK_FLOOR tile for per-operand coalescing — capping
        # the floor by budget_target there starved it to [1,16] (measured 1.83x SLOWER than default,
        # vs 1.48x faster at the [1,256] floor). For any flat kernel reg_cap is large, so this stays
        # 256 — byte-identical to the old hardcoded floor.
        inner_floor = min(cls.BLOCK_FLOOR, cls._pow2_floor(reg_cap))
        # A coalescing CONFLICT (>1 contiguous axis: a transposed load + a contiguous store) IGNORES
        # the bandwidth byte budget — it is wasted on the strided operand — and fills a BALANCED tile
        # up to the REGISTER limit (reg_cap) only. It does NOT apply occ_cap: a strided access needs a
        # long coalescing run to amortize the wasted cache-line bandwidth MORE than the grid needs many
        # programs, so under-saturating a small conflict shape with a bigger tile is faster (measured:
        # transposed_out_add [2048,512] [16,16]=0.84 -> [64,64]=1.02; large shapes are unchanged since
        # reg_cap bounds the run to ~64 either way). Single-contig / flat kernels use `target` (occ-capped).
        balance_cap = max(1, reg_cap)
        block_sizes = cls._seed_block_sizes(
            spec, target, inner_floor, fact.contig_block_ids, balance_cap
        )
        # num_warps ramp — keyed on arithmetic INTENSITY (SFU op count), tile-size-aware. Emitted only
        # when it exceeds the default (so the bandwidth-bound flat family stays block_sizes-only, the
        # dead-knob rule + byte-identical). num_stages/pid_type stay at defaults (1 / 'flat').
        tile_numel = 1
        for b in block_sizes:
            tile_numel *= b
        num_warps = cls._warps_for(fact.sfu_ops, tile_numel)
        if num_warps > cls.DEFAULT_WARPS:
            return Config(block_sizes=block_sizes, num_warps=num_warps)
        return Config(block_sizes=block_sizes)

    @classmethod
    def _warps_for(cls, sfu_ops: int, tile_numel: int) -> int:
        """num_warps from arithmetic intensity (SFU op count), capped by tile size. A transcendental-
        heavy program is latency-bound (more warps hide the SFU latency); a small tile cannot feed many
        warps (each warp needs >= ELEMS_PER_WARP elements, else it starves — measured regression)."""
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
        # Round DOWN to a pow2 within [floor (and the spec's correctness min), max_size]. max_size
        # is next_pow2(extent), so capping there (rather than pre-capping at the raw size_hint) lets
        # the inner tile COVER a short row in one masked tile (extent 768 -> 1024, not floored to
        # 512). autotuner_min is the AUTOTUNER's search floor, NOT a seed constraint (the reduction
        # and split-join seeds emit block=1 below it), so it is intentionally not applied here.
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
        """Distribute the target tile (in elements) across the block dims so the wide part lands on a
        CONTIGUOUS (stride-1) axis — the coalesced run. The contiguous axis is read from the fact's
        ``contig_block_ids`` (stride provenance), NOT assumed to be the last dim:

        - **Flat / single contiguous axis** (the common case — a row-major kernel's stride-1 axis IS
          the last block dim): distribute INNERMOST-first from the contiguous axis, spilling the
          leftover budget outward. Rooted at the last dim this is byte-identical to the prior seed:
          a WIDE row gives ``[1, 1024]``; a short-inner (tall-skinny) tensor spills outward to
          ``[128, 8]`` (a coalesced ``R*N`` block) instead of a starved ``[1, 8]``. For a TRANSPOSED
          view whose single contiguous axis is dim 0 (``xT = base.t()``), it roots the wide tile there
          → ``[1024, 1]`` instead of the uncoalesced ``[1, 1024]`` (measured 5.3x slower).
        - **Coalescing CONFLICT** (>1 contiguous axis — a transposed load wants dim 0, a contiguous
          store wants dim 1): no single wide axis coalesces every operand (``[2048,1]`` strides the
          store, measured 0.13x), so emit a BALANCED tile giving every contiguous axis an equal run.

        The floor (``inner_floor``, register-capped) applies to the primary contiguous axis so it keeps
        a ``BLOCK_FLOOR`` coalescing run except when a heavy untiled slab shrinks reg_cap (rope→1)."""
        n = len(spec.block_sizes)
        specs = [cast("BlockSizeSpec", spec.block_sizes[i]) for i in range(n)]
        # Positions whose block-id is a contiguous (stride-1) axis for some full-extent op.
        contig_pos = [i for i in range(n) if specs[i].block_id in contig_block_ids]
        if len(contig_pos) >= 2:
            return cls._balanced_block_sizes(specs, contig_pos, balance_cap)
        block = [1] * n
        # Root the wide tile at the single contiguous axis; fall back to the last (innermost) dim when
        # contiguity is unknown or already the last dim (the flat family — byte-identical).
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
        """Balanced (square-ish) pow2 tile for a coalescing CONFLICT: several tiled axes are each the
        contiguous axis for some operand (a transposed load + a contiguous store). Give every
        contiguous axis an EQUAL coalescing run — a single wide axis would stride the other operand
        (the ``[2048,1]`` disaster, 0.13x) — sized to fill the register/occupancy budget (balance_cap
        = min(reg_cap, occ_cap)); the bandwidth byte budget does not apply (it is wasted on the
        strided operand). Non-contiguous axes stay 1 (a wide tile on a strided axis cannot coalesce)."""
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
    """Gate: exactly one ``ReductionFact`` and no ``matmul_facts``. Admits both tracks
    (standard rollable, user-tiled); excludes GEMMs and multi-axis manual reductions."""
    spec = env.config_spec
    return len(spec.reduction_facts) == 1 and not spec.matmul_facts


def _is_standard_reduction(spec: ConfigSpec, fact: ReductionFact) -> bool:
    """standard vs user-tiled discriminator: standard iff the rdim is NOT a ``block_sizes``
    entry. Covers a roller-rolled rdim (a ``reduction_loops`` entry) AND a MATERIALIZED rdim
    (an inner ``reduction=True`` axis the roller declined to roll, in NEITHER spec --
    rms/ln/instance backward); user-tiled is the rdim-is-a-block_sizes case.
    """
    return fact.block_id not in spec.block_sizes.valid_block_ids()


def _grid_rows(env: CompileEnvironment, m_block_ids: tuple[int, ...]) -> int:
    """Product of the static M-axis (non-reduction grid) extents — the program count the
    reduction launches, the numerator of the occupancy ``grid_rows // num_sm``. Returns 0
    if any extent is not a statically-resolvable size (a dynamic grid has no compile-time
    occupancy, so the occupancy-gated narrow-w1 lever declines).
    """
    grid_rows = 1
    for mbid in m_block_ids:
        size = env.block_sizes[mbid].size
        if not isinstance(size, (int, torch.SymInt)):
            return 0
        grid_rows *= env.size_hint(size)
    return grid_rows


class _TritonReductionSeedBase(AutotunerHeuristic):
    """Shared base for the two Triton inner-reduction seed heuristics. Both share the
    workload facts (``ReductionFact``), the M_BLOCK-aware reduction-block lever
    (``_reduction_rblock``, from which each track derives ``persistent``), the ``num_warps``
    ramp, eviction provenance, and the block-size builders; the subclasses differ only in
    mapping that decision onto knobs:

    - **standard** (:class:`TritonStandardReductionHeuristic`): Helion rolls the rdim into
      a ``reduction_loops`` loop.
    - **user-tiled** (:class:`TritonUserTiledReductionHeuristic`): the user hand-writes the
      ``hl.tile`` loop, so the reduction axis is a ``block_sizes`` entry (plain user-tiled
      softmax, carried-2-D-tile kl_div/jsd, reduce-then-apply welford).

    Not registered; only the subclasses are.
    """

    backend = "triton"
    HARDWARE_TARGETS = (("cuda", "sm90"),)

    # Looped-fallback reduction chunk (pow2) for a row that does not fit the persistent
    # residency budget. ``_reduction_rblock`` shrinks it when a raised M_BLOCK divides the
    # footprint budget below it; at M_BLOCK==1 it is used as-is.
    LOOPED_CHUNK = 16384
    # Per-program byte budget for LOOP-CARRIED 2-D accumulator tiles ([M_BLOCK, R_BLOCK], e.g.
    # kl_div/jsd): caps R_BLOCK via num_carried_2d_tiles. Tightest budget -- resident the whole loop.
    CARRIED_TILE_MAX_BYTES = 16384
    # Per-program persistent byte ceiling. The resident reduction tile is [M_BLOCK, R_BLOCK]
    # in BOTH tracks (the persistent load and the looped accumulator both carry the M_BLOCK
    # dim), so the per-program footprint is ``m_block * r_block * itemsize`` and every cap
    # below divides the budget by M_BLOCK. Above it a wide resident tile spills register/SMEM,
    # so the reduction loops a chunk instead. ~240 KiB, just over H100 SMEM.
    ROW_PERSIST_MAX_BYTES = 245760
    # Per-program ELEMENT ceiling for a FULL-WIDTH-output row (stores the whole [M, N] row
    # back): its resident tile is fp32-promoted, so it spills at a WIDTH independent of input
    # dtype, which the byte cap above (input bytes) undercounts 2x for a half-precision row.
    # M_BLOCK-aware; gates only full_width_output rows, steering half-precision full-width
    # standard rows onto the looped path.
    FULL_WIDTH_PERSIST_MAX_ELEMS = 81920
    # Max resident bytes a PERSISTENT reduction body (body_live_tiles full-width tiles) may hold
    # before catastrophic register spill. Only REMOVES persistence from a heavy body, never grows it.
    LIVE_PERSIST_BUDGET = 3 * 245760
    # M-COLLAPSE (grad-parameter reduction, e.g. bias_grad): max rows one CTA reduces in a single
    # in-register inner tile, capped so the reduction tree + [rows, feature] tile don't spill.
    M_COLLAPSE_MAX_CTA = 256
    # M-COLLAPSE inner reduction tile byte budget: a grad-parameter collapse is memory-bound, so
    # the inner [rows, feature] tile wants the SMALLEST footprint (~2-8 rows) for CTA occupancy.
    M_COLLAPSE_TILE_BYTES = 32768
    # No welford "structured-combine floor": welford is memory-bound (profiler-confirmed),
    # so a wide combine tile only spills — register-residency via the reduction footprint cap
    # is what matters. The apply/normalize tile gets the SAME M_BLOCK-aware footprint cap as
    # the reduction tile, NOT a flat per-row cap (which needlessly narrowed the memory-bound
    # apply pass); applied inline in ``_build_block_sizes``.

    # NARROW-row single-warp (occupancy-gated): a narrow reduction extent wants ONE warp (the
    # cross-warp reduction tree is pure overhead; w1 reduces in-register via shuffle). The win
    # inverts past an occupancy ceiling (the SMs saturate), so it is gated on a row-byte cap
    # AND an occupancy cap, both keyed on input_load_itemsize (the HBM-load element width —
    # faithful and dtype-agnostic, unlike the fp32-promoted accumulator itemsize which is 4 at
    # both dtypes):
    #   - row cap: rnumel * input_load_itemsize <= NARROW_W1_MAX_BYTES.
    #   - occ cap: occ * row_bytes <= NARROW_W1_OCC_BYTE_LIMIT (a wider row saturates at lower
    #     occupancy, so the ceiling is on the product, not a flat occ).
    NARROW_W1_MAX_BYTES = 2048
    NARROW_W1_OCC_BYTE_LIMIT = 262144

    @classmethod
    def _carried_tile_r_block_cap(cls, fact: ReductionFact) -> int:
        """Pow2 R_BLOCK ceiling for a reduction carrying loop-resident 2-D accumulator tiles
        (kl_div, jsd): the per-program byte budget ``CARRIED_TILE_MAX_BYTES`` split across the
        accumulator itemsize and the carried-tile count. ``max(1, ..)`` guards a zero itemsize
        or tile count.
        """
        from ..._utils import next_power_of_2 as _np2

        cap = cls.CARRIED_TILE_MAX_BYTES // (
            max(1, fact.itemsize) * max(1, fact.num_carried_2d_tiles)
        )
        return _np2(max(1, cap))

    @classmethod
    def _num_warps(
        cls, fact: ReductionFact, num_sm: int = 0, grid_rows: int = 0
    ) -> int:
        """Scale num_warps with the reduction extent (pow2, per NumWarpsFragment):
        rnumel <= 1024 -> 4, <= 4096 -> 8, <= 16384 -> 16, > 16384 -> 32. Too few
        under-occupies the SM, too many wastes the reduction tree.

        NARROW-row single-warp refinement at the LOW end (the occupancy-gated lever): a
        narrow row at low/moderate occupancy wants ONE warp (the cross-warp reduction tree
        is pure overhead — see ``NARROW_W1_MAX_BYTES``). Fires only when the row-byte cap AND
        the resident-pressure cap (``occ * row_bytes <= NARROW_W1_OCC_BYTE_LIMIT``) hold; both
        key on ``input_load_itemsize`` (faithful, no dtype-kind branch) and the occ ceiling
        scales DOWN as the row grows (a wider row cliffs at lower occupancy). Needs ``num_sm``
        (0 disables it, e.g. an off-device caller). Disjoint from the wide-row branch below
        (``NARROW_W1_MAX_BYTES`` << the rnumel>16384 region), so the two never interact.
        """
        rnumel = fact.size_hint
        ils = fact.input_load_itemsize
        row_bytes = rnumel * ils
        # NARROW-row single-warp (see NARROW_W1_MAX_BYTES); needs a known device + static grid.
        have_enough_information = num_sm > 0 and ils > 0 and grid_rows > 0
        if have_enough_information:
            occ = grid_rows // num_sm
            if (
                fact.num_carried_2d_tiles
                == 0  # not a carried-2-D-tile reduction (kl_div/jsd)
                and row_bytes <= cls.NARROW_W1_MAX_BYTES
                and occ * row_bytes <= cls.NARROW_W1_OCC_BYTE_LIMIT
            ):
                return 1
        # >16384 (not >=) so a 16384-wide row stays w16, excluding the w32 regression there.
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
        """The smallest valid block size for an entry, used for every non-reduction axis
        the seed does not widen. Prefers one row/program but honors a raised
        ``autotuner_min`` (large-M shapes) rather than emitting an invalid ``block_size=1``.
        """
        return max(1, bs_spec.min_size, bs_spec.autotuner_min)

    @classmethod
    def _m_block_cap(cls, fact: ReductionFact) -> int:
        """Upper bound on M_BLOCK (rows/program) for a FULL-WIDTH-output reduction, so a huge-M
        grid-size ``autotuner_min`` raise cannot force an occupancy-starving M_BLOCK on a
        memory-bound held-row reduction. A seed below ``autotuner_min`` (raised only to cap the
        grid) is still valid -- it survives ``normalize``.

        The cap keeps the resident ``[M_BLOCK, rdim]`` live set inside the per-program register
        budget: ``M_BLOCK <= ROW_PERSIST_MAX_BYTES / (rdim * itemsize * body_live_tiles)``.
        Applied only via ``min`` with ``_block_floor`` (only ever LOWERS an over-raised floor)
        and only for full-width output -- streamed/scalar reductions ride occupancy on the
        chunk, not M_BLOCK, so they are uncapped.
        """
        from ..._utils import prev_power_of_2

        if not fact.full_width_output:
            return (
                1 << 30
            )  # no cap: scalar/streamed occupancy rides the reduction chunk
        live = max(1, fact.body_live_tiles)
        isz = max(1, fact.itemsize)
        sh = max(1, fact.size_hint)
        return max(
            1, prev_power_of_2(max(1, cls.ROW_PERSIST_MAX_BYTES // (sh * isz * live)))
        )

    @classmethod
    def _m_block_product(cls, spec: ConfigSpec, fact: ReductionFact) -> int:
        """Product of the seed's floored M-axis (grid) block sizes -- the number of rows each
        program processes (1 unless a huge-M shape raised ``autotuner_min``, capped by
        ``_m_block_cap`` for full-width reductions). Shared by the apply-loop stream cap
        (``_build_block_sizes``) and the Band-C combine cap so they read the same M_BLOCK.
        """
        m_block = 1
        cap = cls._m_block_cap(fact)
        for mbid in fact.m_block_ids:
            m_idx = spec.block_sizes.block_id_to_index(mbid)
            m_block *= min(
                cls._block_floor(cast("BlockSizeSpec", spec.block_sizes[m_idx])), cap
            )
        return m_block

    @classmethod
    def _build_block_sizes(
        cls,
        spec: ConfigSpec,
        fact: ReductionFact,
        red_block_id: int | None,
        red_value: int | None,
        non_reduction_loop_ids: frozenset[int] | set[int] = frozenset(),
    ) -> list[int]:
        """Build the ``block_sizes`` list: the reduction axis gets ``red_value``, each
        non-reduction loop tile (``non_reduction_loop_ids``, disjoint from the reduction
        block_id) gets ``loop_block``, every other axis its ``_block_floor``.
        ``red_block_id`` is None for standard (the reduction rides ``reduction_loops``, not
        a block_sizes entry).

        The non-reduction loop tile matches the reduction tile — ``red_value`` (user-tiled)
        or ``next_pow2(size_hint)`` (standard, where ``red_value`` is None). The normalize
        pass carries no accumulator, so this tile is a pure seed (a sane non-size-1 start,
        never a correctness constraint); the autotuner refines it from there.
        """
        from ..._utils import next_power_of_2 as _np2
        from ..._utils import prev_power_of_2

        loop_block: int | None = None
        if non_reduction_loop_ids:
            # The apply/normalize tile starts at the reduction tile — red_value (user-tiled) or
            # next_pow2(size_hint) (standard, where red_value is None)...
            loop_block = red_value if red_value is not None else _np2(fact.size_hint)
            # ...then is clamped to the same M_BLOCK-aware footprint cap as the reduction
            # tile (the apply tile is [M_BLOCK, loop_block] resident, so a wide one spills).
            # Only the Band-C reduce-then-apply kernels (welford, groupnorm) have a normalize
            # loop, so everything else is byte-identical (no non_reduction_loop_ids). The cap
            # clamps an otherwise-spilling apply pass back to register residency; the pass is
            # memory-bound so this is a net win. A flat per-row cap always narrowed it.
            m_block = cls._m_block_product(spec, fact)
            budget = cls.ROW_PERSIST_MAX_BYTES // (m_block * max(1, fact.itemsize))
            loop_block = min(loop_block, prev_power_of_2(budget))

        red_idx = (
            spec.block_sizes.block_id_to_index(red_block_id)
            if red_block_id is not None
            else None
        )
        out: list[int] = []
        for i in range(len(spec.block_sizes)):
            bs_spec = cast("BlockSizeSpec", spec.block_sizes[i])
            if i == red_idx:
                out.append(cast("int", red_value))
            elif bs_spec.block_id in non_reduction_loop_ids and loop_block is not None:
                out.append(loop_block)
            elif bs_spec.block_id in fact.m_block_ids:
                # M (grid) axis: floor it, but cap a full-width reduction's M_BLOCK by the
                # register budget (_m_block_cap) so a huge-M grid raise can't starve occupancy.
                out.append(min(cls._block_floor(bs_spec), cls._m_block_cap(fact)))
            else:
                out.append(cls._block_floor(bs_spec))
        return out

    @classmethod
    def _eviction_policies(
        cls,
        env: CompileEnvironment,
        kind: str,
        reread_slot: int | None = None,
    ) -> list[str] | None:
        """``load_eviction_policies`` list (spec length), keyed on per-load residency;
        None leaves the autotuner default.

        - ``"stream"`` — single streamed input (``num_load == 1``: sum, long_sum), read
          once: every load -> ``'first'`` (frees L2).
        - ``"reread"`` — the row is re-read across passes: its first load -> ``'last'``
          (L2-resident), rest -> ``'first'``. ``reread_slot`` is that load's actual slot,
          read directly from ``ReductionFact.reread_eviction_index`` (the re-read load's
          ``MemoryOpFact.eviction_index``), not guessed or re-walked per config.

        Other kinds leave the default until a per-slot win is confirmed.
        """
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

    @classmethod
    def _reduction_rblock(
        cls,
        env: CompileEnvironment,
        fact: ReductionFact,
        m_block: int,
        footprint_factor: int = 1,
    ) -> tuple[int, bool]:
        """The reduction-axis chunk (pow2) AND the persistent verdict, decided together in one
        budgeted formula and shared by both tracks. Returns ``(r_block, persistent)``.

        ``footprint_factor`` = how many resident rdim-shaped tiles one program holds live at
        the peak (1 = a single result tile). It bounds the decision two ways:
        - PERSISTENT additionally requires (besides ``row_reread`` and no carried 2-D tile) the
          single resident tile to fit ``ROW_PERSIST_MAX_BYTES`` AND the full
          ``footprint_factor``-tile resident set to fit ``LIVE_PERSIST_BUDGET`` (the multi-tile
          spill ceiling). This liveness term only ever REMOVES persistence from a heavy body.
        - the LOOPED chunk is shrunk by ``footprint_factor`` so a heavy body gets a smaller
          chunk, keeping the looped resident set inside the register budget.

        The standard track passes ``footprint_factor=body_live_tiles``; the user-tiled track
        keeps the default ``1`` (where ``LIVE_PERSIST_BUDGET`` collapses to the base byte cap,
        a no-op). The carried-2-D-tile cap (kl_div/jsd) is folded into the chunk decision.

        No welford "structured-combine floor": register-residency via the footprint cap is what
        matters; welford is memory-bound and a wide combine tile only spills.
        """
        from ..._utils import next_power_of_2 as _np2
        from ..._utils import prev_power_of_2

        rdim = _np2(fact.size_hint)
        itemsize = max(1, fact.itemsize)
        m = max(1, m_block)
        ff = max(1, footprint_factor)
        # Persistent iff (a) a SINGLE result tile fits the per-program caps (the element compile
        # limit element_cap AND the ROW_PERSIST_MAX_BYTES single-tile byte cap), AND (b) the full
        # ff-tile resident set fits LIVE_PERSIST_BUDGET. (b) is the liveness ceiling: it only
        # removes persistence from a heavy body, never grants it (no-op at ff==1).
        element_cap = env.backend.max_tensor_numel
        # Hold the full extent one-shot iff it clears every per-program ceiling (element compile
        # limit element_cap, ROW_PERSIST_MAX_BYTES byte cap, LIVE_PERSIST_BUDGET live-set cap) AND
        # there is NO carried 2-D accumulator -- a [M_BLOCK, R_BLOCK] tile carried across the loop
        # is too heavy to hold, so stream.
        extent_held = (
            # Persist captures a re-read prize iff looping would RE-READ the row: a full-width apply
            # (rms/layer_norm/softmax/welford) or a second reduction needing the first's result
            # (cross_entropy's logsumexp -- which full_width_output misses). row_reread catches both;
            # num_carried_2d_tiles == 0 keeps kl_div/jsd off persist (they re-read regardless).
            fact.row_reread
            and fact.num_carried_2d_tiles == 0
            and (element_cap is None or fact.size_hint <= element_cap)
            and (m * fact.size_hint * itemsize <= cls.ROW_PERSIST_MAX_BYTES)
            and (ff * m * fact.size_hint * itemsize <= cls.LIVE_PERSIST_BUDGET)
        )
        if extent_held:
            r_block = rdim
        else:
            # Can't hold the extent -> stream it at the occupancy-optimal LOOPED_CHUNK (M_BLOCK- and
            # liveness-shrunk); a capacity-sized chunk would re-read AND lower occupancy.
            budget = cls.ROW_PERSIST_MAX_BYTES // (m * itemsize * ff)
            r_block = min(cls.LOOPED_CHUNK, prev_power_of_2(budget))
            # Carried 2-D accumulators (kl_div, jsd) always reach this branch; cap R_BLOCK by its tighter
            # byte budget to avoid catastrophic register spills. No-op when num_carried_2d_tiles == 0.
            if fact.num_carried_2d_tiles >= 1:
                r_block = min(r_block, cls._carried_tile_r_block_cap(fact))
        # Never size the chunk past the (padded) extent: clamp to rdim so a sub-cap-N carried reduction
        # matches the old held branch and keeps `persistent` below correct.
        r_block = max(1, min(r_block, rdim))
        # `persistent` is READ OFF the final chunk -- held iff the chunk reached the extent
        # (reduction_loops=[None] standard, or R_BLOCK == rdim user-tiled), never set separately.
        return r_block, r_block >= rdim

    @classmethod
    def _m_collapse_grid_block(
        cls, env: CompileEnvironment, fact: ReductionFact, cap: int | None = None
    ) -> int:
        """Occupancy-sized grid M block for a grad-parameter M-collapse (rms/ln/instance/group
        backward on the standard track; bias_grad/dyt on the user-tiled track). The
        grad-parameter (``grad_weight[N]`` / ``grad_bias[N]``) is summed across the grid rows
        into a per-CTA partial finalized by a cross-CTA ``sum(0)``; with the block floored to 1
        that is a grid-wide M-way collapse (one partial/row), so size it to ~one SM wave
        (``next_pow2(grid_rows // num_sm)``) to cut it to ~``num_sm`` partials.

        ``cap`` bounds the block when it ALSO bears the reduction slab (user-tiled pure
        collapse, capped at ``M_COLLAPSE_MAX_CTA``); the standard track leaves it uncapped
        because the resident ``[inner, feature]`` set rides a separate inner re-tile, not this
        block. A dynamic/unbacked grid (``grid_rows == 0``) collapses to 1.
        """
        from ..._utils import next_power_of_2 as _np2
        from ...runtime import get_num_sm

        grid_rows = _grid_rows(env, fact.m_block_ids)
        num_sm = max(1, get_num_sm(env.device))
        block = _np2(max(1, grid_rows // num_sm))
        return max(1, block if cap is None else min(cap, block))

    @classmethod
    def _m_collapse_inner_byte_cap(cls, feat_bytes: int) -> int:
        """Largest pow2 inner reduction tile whose resident ``[inner, feature]`` fp32 set fits
        ``M_COLLAPSE_TILE_BYTES``, given the per-row feature footprint ``feat_bytes``. Both
        M-collapse tracks pass ``fact.feature_footprint * itemsize`` (the PRODUCT of the
        materialized feature axes); for a 2-D norm that product is the single feature axis.
        """
        from ..._utils import next_power_of_2 as _np2

        return max(1, _np2(max(1, cls.M_COLLAPSE_TILE_BYTES // max(1, feat_bytes))))


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
        spec = env.config_spec
        return _is_standard_reduction(spec, spec.reduction_facts[0])

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
            # Off the H100-validated target: keep the upstream conservative seed.
            return cls._narrow_seed(env)
        from ...runtime import get_num_sm

        spec = env.config_spec
        fact = spec.reduction_facts[0]
        # standard rides persistent-vs-looped on reduction_loops (sized by the shared _reduction_rblock).
        # footprint_factor=body_live_tiles routes a heavy body that would overflow the register file
        # persistent (e.g. fused_linear_jsd) to the looped path instead.
        m_block = cls._m_block_product(spec, fact)
        r_block, persistent = cls._reduction_rblock(
            env,
            fact,
            m_block,
            footprint_factor=fact.body_live_tiles,
        )
        num_warps = cls._num_warps(
            fact, max(1, get_num_sm(env.device)), _grid_rows(env, fact.m_block_ids)
        )

        # A standard reduction may be followed by a normalize loop (e.g. `s = x.sum(); out =
        # x/s`); its extra block_sizes tile(s) are sized by _build_block_sizes (matched to
        # the reduction tile). Only a seed (a worse tile costs autotuning time, never
        # correctness), so emit and let the autotuner refine.
        non_reduction_loop_ids = set(fact.non_reduction_loop_block_ids)

        # red_block_id=None: rdim is not a block_sizes entry, so every entry is a grid axis (floored)
        # or a normalize-loop tile. MATERIALIZED rdim (rms/ln/instance bwd, the roller declined to roll
        # it): emit an EMPTY reduction_loops -- already full-width persistent, and a length-1 list would
        # fail normalize against the 0-length spec.
        is_materialized = fact.block_id not in spec.reduction_loops.valid_block_ids()
        reduction_loops: list[int | None]
        if is_materialized:
            reduction_loops = []
        else:
            reduction_loops = [None] if persistent else [r_block]
        block_sizes = cls._build_block_sizes(
            spec, fact, None, None, non_reduction_loop_ids=non_reduction_loop_ids
        )
        # Dual-axis grad-parameter M-collapse (rms/ln/instance/group bwd): the grid M block is re-tiled
        # by an inner loop and feeds a per-feature grad accumulator finalized across CTAs.
        # _build_block_sizes floors it to 1 (leaving a grid-wide finalize); size it for occupancy so the
        # finalize shrinks to ~num_sm partials. Gated on per_feature_accumulator, which is False for the
        # 9 standard + 8 transfer kernels, so their seeds stay byte-identical.
        if fact.per_feature_accumulator:
            # The lever needs the grow-grid / byte-cap-inner decomposition to exist: a non-grid
            # inner tile bearing residency. per_feature_accumulator implies a device carry loop
            # (not the grid or rdim) that appears in inner_tile_ids, so an EMPTY inner_tile_ids
            # means no inner re-tile -- skip the lever and keep the base materialized seed.
            grid_ids = {b for bids in device_ir.grid_block_ids for b in bids}
            inner_tile_ids = [
                b
                for b in spec.block_sizes.valid_block_ids()
                if b not in grid_ids
                and b != fact.block_id
                and b not in non_reduction_loop_ids
            ]
            if inner_tile_ids:
                # Occupancy-size the grid M block (the dominant lever), byte-cap the inner
                # re-tile to the feature footprint, and drop the narrow-w1 warps lever: it keys
                # on rdim extent alone, but the resident tile here is [inner, feature]-wide, so
                # the plain extent ramp (>=4 warps) is faithful. Only instance_norm is affected.
                m_cta = cls._m_collapse_grid_block(env, fact)
                for mbid in fact.m_block_ids:
                    block_sizes[spec.block_sizes.block_id_to_index(mbid)] = m_cta
                inner = cls._m_collapse_inner_byte_cap(
                    max(1, fact.feature_footprint) * max(1, fact.itemsize)
                )
                for bid in inner_tile_ids:
                    block_sizes[spec.block_sizes.block_id_to_index(bid)] = inner
                num_warps = cls._num_warps(
                    fact
                )  # num_sm/grid_rows default 0 -> no narrow-w1
        seed: dict[str, Any] = {
            "block_sizes": block_sizes,
            "reduction_loops": reduction_loops,
            "num_warps": num_warps,
            "num_stages": 1,
            # 'flat': these reductions are grid-saturated at the M-grid.
            "pid_type": "flat",
        }
        # Eviction: streamed input -> 'first' everywhere; looped re-read -> first load
        # 'last', rest 'first'. PERSISTENT rows are left at default ON PURPOSE (the `not
        # persistent` gate below): a rolled persistent reduction fuses the reduce + apply to a
        # SINGLE HBM load of the row (profiler-confirmed), so a 'last' hint is a no-op and
        # actually regresses wide rows by pinning x and evicting weight/store lines. This is
        # the opposite of the user-tiled track, where softmax_two_pass has two PHYSICAL
        # reduction loops (two loads) so 'last' helps even when persistent.
        evict = None
        if fact.num_load == 1:
            evict = cls._eviction_policies(env, "stream")
        elif fact.row_reread and not persistent:
            # Re-read row's eviction slot read directly from the fact (its load's
            # MemoryOpFact.eviction_index), not a per-config codegen re-walk.
            evict = cls._eviction_policies(env, "reread", fact.reread_eviction_index)
        if evict is not None:
            seed["load_eviction_policies"] = evict
        return Config(**seed)


class TritonUserTiledReductionHeuristic(_TritonReductionSeedBase):
    """user-tiled inner-reduction seed: fires when the user hand-writes the ``hl.tile`` loop
    over the reduction axis (so the rdim is an ordinary ``block_sizes`` entry, e.g.
    ``hl.tile(n, block_size=R_BLOCK)``), which the upstream gate rejects entirely.
    R_BLOCK starts at the shared ``_reduction_rblock`` (M_BLOCK-aware footprint cap), then
    INDEPENDENT band predicates layer on via ``min`` (a kernel gets every cap it matches;
    today's kernels each match exactly one):

    - **plain user-tiled** (softmax_two_pass): no extra cap -- persistent full-pow2 R_BLOCK,
      standard-style reread-eviction for wide looped rows.
    - **carried 2-D tiles** (kl_div, jsd): carry ``[M_BLOCK, R_BLOCK]`` accumulator tiles
      across the loop, so R_BLOCK is capped by ``CARRIED_TILE_MAX_BYTES / (itemsize *
      num_carried_2d_tiles)`` -- folded into the shared ``_reduction_rblock`` decision.
    - **reduce-then-apply** (welford, ``non_reduction_loop_block_ids`` non-empty): no combine
      floor. Its normalize/apply tile starts at the reduction tile and gets the SAME
      M_BLOCK-aware footprint cap; see ``_build_block_sizes``.

    TODO(reductions): as more structured families land, promote each band into its own
    fact-keyed ``AutotunerHeuristic`` subclass rather than growing this method.
    """

    name = "triton_reduction_user_tile"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if not _triton_reduction_eligible(env, device_ir):
            return False
        spec = env.config_spec
        return not _is_standard_reduction(spec, spec.reduction_facts[0])

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            # Off sm90: upstream never fired on user-tiled, so no prior seed to preserve. Decline.
            return None
        from ...runtime import get_num_sm

        spec = env.config_spec
        fact = spec.reduction_facts[0]
        non_reduction_loop_ids = set(fact.non_reduction_loop_block_ids)
        m_block = cls._m_block_product(spec, fact)

        # user-tiled: rdim IS a block_sizes entry (no reduction_loops knob); persistent == R_BLOCK >=
        # next_pow2(N), other axes floored (u0*u1 <= 2**20). The shared lever sizes R_BLOCK from
        # residency (single-tile footprint + the folded-in carried-2-D cap for kl_div/jsd) and returns
        # it directly; _persistent is unused on this track.
        r_block, _persistent = cls._reduction_rblock(env, fact, m_block)
        # M-COLLAPSE (grad-parameter reduction, e.g. bias_grad/dyt): collapse the grid/row axis into a
        # per-feature accumulator, sizing the grid CTA for occupancy instead of T2's floored grid.
        m_collapse_block: int | None = None
        # Faithful signature: per_feature_accumulator -- a loop-carried accumulator over ALL
        # the materialized feature axis (bias_grad/dyt); per-row / 2-D accumulators are excluded.
        is_m_collapse = fact.per_feature_accumulator
        if is_m_collapse:
            # (a) grid CTA -> OCCUPANCY (_m_collapse_grid_block), capped at M_COLLAPSE_MAX_CTA since the
            #     grid block also bears the reduction slab (sum(0) finalize over ~num_sm partials). An
            #     unbacked/AOT grid (grid_rows == 0) falls through to block 1 -- a worse seed, not a bug.
            m_collapse_block = cls._m_collapse_grid_block(
                env, fact, cap=cls.M_COLLAPSE_MAX_CTA
            )
            # (b) inner reduction tile: depends on whether the collapse has PER-ROW WORK,
            #     which ``body_live_tiles`` measures (peak simultaneously-live full-width tiles).
            if fact.body_live_tiles <= 1:
                # PURE collapse (bias_grad: read + sum, ONE resident tile): a big inner tile is
                # cheap and cuts loop overhead, so reduce the whole CTA wave in one slab (bounded
                # by the grid block + 256 cap).
                r_block = m_collapse_block
            else:
                # Collapse WITH per-row work (dyt: full-width grad_x store + tanh intermediates):
                # a big inner tile spills, so byte-cap the resident [inner, feature] footprint
                # tight (~2-8 rows) for occupancy.
                feat_bytes = max(1, fact.feature_footprint) * max(1, fact.itemsize)
                inner_cap = cls._m_collapse_inner_byte_cap(feat_bytes)
                r_block = max(1, min(m_collapse_block, inner_cap))

        num_warps = cls._num_warps(
            fact, max(1, get_num_sm(env.device)), _grid_rows(env, fact.m_block_ids)
        )

        block_sizes = cls._build_block_sizes(
            spec,
            fact,
            fact.block_id,
            r_block,
            non_reduction_loop_ids=non_reduction_loop_ids,
        )
        if m_collapse_block is not None:
            # Raise the grid CTA tile(s) from the floor to the occupancy block (the reduction
            # tile was already set to m_collapse_block via r_block above).
            for mbid in fact.m_block_ids:
                idx = spec.block_sizes.block_id_to_index(mbid)
                block_sizes[idx] = m_collapse_block
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
        if non_reduction_loop_ids or fact.row_reread:
            # Re-read row's eviction slot read directly from the fact (its load's
            # MemoryOpFact.eviction_index), not a per-config codegen re-walk.
            ev = cls._eviction_policies(env, "reread", fact.reread_eviction_index)
            if ev is not None:
                seed["load_eviction_policies"] = ev
        return Config(**seed)


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
