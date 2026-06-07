from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

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


def _triton_reduction_eligible(env: CompileEnvironment, device_ir: DeviceIR) -> bool:
    """Gate for the Triton reduction seed: exactly one registered ``ReductionFact``
    (one inner reduction axis) and no ``matmul_facts``. Admits both tracks (T1
    rollable rdim and T2 user-tiled) and excludes GEMMs and multi-axis manual
    reductions (which leave ``reduction_facts`` empty)."""
    spec = env.config_spec
    return len(spec.reduction_facts) == 1 and not spec.matmul_facts


def _is_t1_reduction(spec: ConfigSpec, fact: ReductionFact) -> bool:
    """T1 (rollable rdim) vs T2 (user-tiled) discriminator: T1 iff the reduction
    axis is a rollable ``reduction_loops`` entry, T2 iff an ordinary ``block_sizes``
    entry. The two device_ir populators are mutually exclusive, so this is
    exhaustive over the eligible reductions.
    """
    return fact.block_id in spec.reduction_loops.valid_block_ids()


class _TritonReductionSeedBase(AutotunerHeuristic):
    """Shared base for the two Triton inner-reduction seed heuristics.

    Both tracks share the same workload facts (``ReductionFact``), the
    persistent-vs-looped first lever (``_persistent_looped``), the ``num_warps``
    ramp, the eviction-policy provenance, and the block-size builders — so those
    live here, and the two concrete heuristics only differ in how they map the
    shared decision onto their track's config knobs:

    - **T1** (:class:`TritonReductionTileHeuristic`): rollable rdim, the
      persistent-vs-looped choice rides the ``reduction_loops`` knob.
    - **T2** (:class:`TritonReductionUserTileHeuristic`): user-tiled, the reduction
      axis is a ``block_sizes`` entry (plain-T2 softmax, Band-B kl_div/jsd, Band-C
      welford).

    Cloned from ``cute.CuteReductionTileHeuristic`` for ``backend="triton"``: drops
    the CuTe-only knobs (``num_threads``, ``cute_vector_widths``) and adds
    ``num_warps`` / ``num_stages``. Not registered (no ``name``); only the two
    subclasses are in ``HEURISTICS_BY_BACKEND``.
    """

    backend = "triton"
    HARDWARE_TARGETS = (("cuda", "sm90"),)

    # Looped fallback chunk (pow2) for rows above the structural persistent cap
    # (rnumel > max_tensor_numel = 2**20), where a single pass cannot compile.
    LOOPED_CHUNK = 16384
    # num_warps for the looped branch (the huge-rnumel streaming class wants 32).
    LOOPED_NUM_WARPS = 32
    # Band-B (T2 carrying a [M_BLOCK, R_BLOCK] 2D accumulator across the inner loop:
    # kl_div, jsd) R_BLOCK cap, in bytes per accumulator. A full-N persistent R_BLOCK
    # over-allocates the live state and spills; the loop holds n_carried tiles at this
    # R_BLOCK, so cap the footprint at R_BLOCK * itemsize * n_carried. In bytes (via
    # itemsize) for dtype-generality. Scalar/row accumulators (n_carried==0) unaffected.
    BANDB_R_BLOCK_BYTES = 16384
    # Persistent byte ceiling per row, above which a reduction loops over a fixed chunk.
    # A wide row pinned resident across the reduction spills past the register/SMEM
    # budget. Caps on actual row bytes (size_hint * itemsize), not next_pow2, since
    # vocab sizes sharing a next_pow2 split across the crossover. 240 KiB sits in the
    # measured dead-zone (most impactful on re-read kernels like cross_entropy).
    ROW_PERSIST_MAX_BYTES = 245760
    # Band-C (welford-like reduce-then-apply) combine tile cap, in bytes. The combine
    # pass is a serial scalar recurrence over count/mean/M2, so it prefers to stay
    # persistent (single next_pow2(N) tile); looping pays the recurrence overhead.
    # 32 KiB keeps it persistent for the welford shapes (N<=8192).
    STRUCTURED_COMBINE_CAP_BYTES = 32768

    @classmethod
    def _num_warps(cls, fact: ReductionFact) -> int:
        """Scale num_warps with the reduction extent (a power of 2, as
        NumWarpsFragment requires). A wider row gives each warp more lane work to
        overlap; too few under-occupies the SM, too many waste the reduction tree::

            rnumel <= 1024  -> 4
            rnumel <= 4096  -> 8
            rnumel <= 16384 -> 16
            rnumel >  16384 -> 32
        """
        # rnumel breakpoint (elements) above which the persistent path wants max warps.
        # Set above 16384 so sum's widest in-sample row (rnumel=16384) stays at w16, and
        # the tiny-rnumel w32 regression is excluded.
        warps32_min_elems = 16384
        rnumel = fact.size_hint
        if rnumel > warps32_min_elems:
            return 32
        if rnumel <= 1024:
            return 4
        if rnumel <= 4096:
            return 8
        return 16

    @classmethod
    def _block_floor(cls, bs_spec: BlockSizeSpec) -> int:
        """The autotuner floor for a single block_sizes entry (the smallest valid
        block size), used for every non-reduction axis the seed does not widen.
        Prefers one row per program but honors a raised ``autotuner_min`` for
        large-M shapes rather than emitting an invalid ``block_size=1``.
        """
        return max(1, bs_spec.min_size, bs_spec.autotuner_min)

    @classmethod
    def _build_block_sizes(
        cls,
        spec: ConfigSpec,
        fact: ReductionFact,
        red_block_id: int | None,
        red_value: int | None,
        non_reduction_loop_ids: frozenset[int] | set[int] = frozenset(),
    ) -> list[int]:
        """Build the ``block_sizes`` list: the reduction axis gets ``red_value``, any
        non-reduction loop tile (``non_reduction_loop_ids``) gets the widened size
        derived here from ``fact``, and every other axis stays at its ``_block_floor``.
        ``non_reduction_loop_ids`` excludes the reduction block_id, so the two never
        collide.

        ``red_block_id`` is ``None`` for T1, where the reduction axis rides the
        ``reduction_loops`` knob and is NOT a ``block_sizes`` entry — then every
        ``block_sizes`` entry is either a grid axis (floored) or a non-reduction loop
        tile (widened).

        A non-reduction loop tile is a pure perf lever (the normalize pass carries no
        accumulator and masks its write): persistent ``next_pow2(N)`` when per-row valid
        work is small, looped above. It keys on per-row valid bytes
        (``n_valid * itemsize``), not ``next_pow2(N)``, so rows sharing a ``next_pow2``
        are still separated — and deliberately NOT tied to ``red_value`` (welford wants
        a narrower normalize tile than its combine tile; see ``get_seed_config``).

        Fallback: when the reduction extent is NOT statically known
        (``fact.static_rnumel is None`` — e.g. a dynamic/jagged reduce-then-apply
        reduction), the per-row-bytes cap has no extent to key on, so the tile matches
        the reduction tile: ``red_value`` for T2 (the reduction axis is a block_sizes
        entry), or ``next_pow2(size_hint)`` for T1 (the reduction rides
        ``reduction_loops`` and a persistent pass processes the full row). This is the
        "if unsure, match the reduction tile" default; it is NOT tuned on any kernel (no
        curriculum kernel has a dynamic-extent non-reduction loop).
        """
        from ..._utils import next_power_of_2 as _np2

        # Threshold (per-row valid bytes) below which the non-reduction loop tile stays
        # persistent at next_pow2(N); above it the tile loops a fixed chunk.
        persist_max_bytes = 12288
        # Looped chunk, in bytes, once per-row work exceeds the persist threshold. Do
        # not raise without re-gating: 4096 is a regression valley at the large-M /
        # N~5120 class.
        loop_chunk_bytes = 8192

        loop_block: int | None = None
        if non_reduction_loop_ids:
            if fact.static_rnumel is None:
                # Untuned dynamic-extent default: match the reduction tile. When the
                # reduction axis is a block_sizes entry (T2) that is ``red_value``; for
                # T1 the reduction rides ``reduction_loops`` (red_value is None), so
                # match ``next_pow2(size_hint)`` — the full row a persistent pass
                # processes. NOT tuned on any kernel — no curriculum kernel has a
                # dynamic-extent non-reduction loop.
                loop_block = (
                    red_value if red_value is not None else _np2(fact.size_hint)
                )
            else:
                np2_n = _np2(fact.size_hint)
                n_valid = fact.static_rnumel
                itemsize = max(1, fact.itemsize)
                if n_valid * itemsize <= persist_max_bytes:
                    loop_block = np2_n
                else:
                    loop_chunk_elems = max(1, loop_chunk_bytes // itemsize)
                    loop_block = min(np2_n, _np2(loop_chunk_elems))

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
        """``load_eviction_policies`` list (length == the live spec's), keyed on
        per-load cache residency. Returns None to leave the autotuner default.

        - ``"stream"`` — a single streamed reduction input (``num_load == 1``: sum,
          long_sum) is read once, so every load -> ``'first'`` (frees L2).
        - ``"reread"`` — the reduction-input row is re-read across passes. Its first
          load -> ``'last'`` (keep L2-resident for the re-read), every other slot ->
          ``'first'``. ``reread_slot`` is the actual ``load_eviction_policies`` slot
          index of that first load, resolved for the emitted config from the rolled
          codegen graphs (``DeviceIR.reread_eviction_slot_for_config``), not a
          positional guess.

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
    def _persistent_looped(
        cls, env: CompileEnvironment, fact: ReductionFact
    ) -> tuple[bool, int, int]:
        """The shared first lever, identical for both tracks; returns
        ``(persistent, extent, num_warps)``.

        Keys on the reduction extent and the backend's per-tile element cap.
        Persistent for every row the backend can compile as a single pass (up to
        max_tensor_numel = 2**20 elems), with an additional per-row byte ceiling
        (ROW_PERSIST_MAX_BYTES): a wide row held resident across the reduction
        spills past the register/SMEM budget, so above it the seed loops a fixed chunk.

        The byte cap is unconditional, though config-neutral off the re-read kernels:
        a single-load looped chunk ties the persistent pass (each element touched
        once), and the Band-B R_BLOCK cap is already tighter than LOOPED_CHUNK. So only
        the re-read kernels (rms_norm/layer_norm/softmax/cross_entropy/welford) are
        actually steered by it.
        """
        from ..._utils import next_power_of_2 as _np2

        # Persistent only if BOTH hold: the backend can compile the row in one pass
        # (structural element cap, None ⇒ no cap) AND the row fits the per-row byte
        # ceiling (the perf spill limit). The two are distinct: the element cap is a
        # backend/dtype compile limit, the byte cap a register/SMEM residency limit.
        element_cap = env.backend.max_tensor_numel
        can_persist = (element_cap is None or fact.size_hint <= element_cap) and (
            fact.size_hint * max(1, fact.itemsize) <= cls.ROW_PERSIST_MAX_BYTES
        )

        if can_persist:
            # Persistent (single-pass). T1 encodes the extent as reduction_loops=None;
            # T2 as the full pow2 R_BLOCK so the inner `for tile_n` loop runs once.
            return True, _np2(fact.size_hint), cls._num_warps(fact)
        # Looped: the row exceeds either the 2**20 structural cap or
        # ROW_PERSIST_MAX_BYTES. A fixed chunk plus high streaming warps.
        return False, cls.LOOPED_CHUNK, cls.LOOPED_NUM_WARPS


class TritonReductionTileHeuristic(_TritonReductionSeedBase):
    """T1 (rollable-rdim) inner-reduction seed: sum, long_sum, rms_norm, layer_norm,
    softmax-row, cross_entropy. The Triton analog of ``CuteReductionTileHeuristic``,
    keeping its registry name; it deepens the original one-row/persistent/``['last']``
    seed with an rnumel-scaled ``num_warps`` ramp, the persistent-vs-looped decision,
    per-slot eviction, and a ``persistent_interleaved`` grid for wide looped re-read
    rows.

    Gated by ``_triton_reduction_eligible`` restricted to the T1 track — broader than
    the upstream ``is_canonical_row_reduction`` gate, also covering multi-axis rollable
    rows and large-M shapes whose ``autotuner_min`` was raised above 1. Off sm90 the
    H100-tuned levers are unvalidated, so it falls back to the upstream conservative
    seed (``_narrow_seed``), preserving pre-existing behavior on other backends.
    """

    name = "triton_reduction_tile"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if not _triton_reduction_eligible(env, device_ir):
            return False
        spec = env.config_spec
        return _is_t1_reduction(spec, spec.reduction_facts[0])

    @classmethod
    def _narrow_seed(cls, env: CompileEnvironment) -> Config:
        """The upstream conservative T1 seed (one row per program, single persistent
        pass, ``['last']`` eviction where the backend supports it). Used off sm90,
        where the deep H100-tuned levers are unvalidated; a verbatim port of the
        original seed so non-sm90 behavior is unchanged.
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
        from ..._utils import next_power_of_2 as _np2

        spec = env.config_spec
        fact = spec.reduction_facts[0]
        # T1 encodes persistent-vs-looped via the `reduction_loops` knob, so the
        # shared lever's `extent` (the T2 R_BLOCK) is unused here.
        persistent, _extent, num_warps = cls._persistent_looped(env, fact)

        # A T1 reduction can be followed by a separate non-reduction loop that
        # normalizes the row, e.g. `s = x[m,:].sum(); for n: out = x/s`. Such a kernel
        # has extra `block_sizes` entries (the non-reduction loop tile(s)) that
        # _build_block_sizes widens, like welford's Band-C normalize tile. NOTE: this
        # T1+normalize path is not performance-validated — no curriculum kernel
        # exercises it — but it is only a seed (a worse starting tile costs autotuning
        # time, never correctness), so emit the widened config and let the autotuner
        # refine rather than decline.
        non_reduction_loop_ids = set(fact.non_reduction_loop_block_ids)

        # T1 (rollable rdim): persistent-vs-looped rides on the `reduction_loops`
        # knob (None = persistent). The reduction axis is NOT a block_sizes entry, so
        # pass red_block_id=None: every block_sizes entry is a grid axis (floored) or a
        # non-reduction loop tile (widened to next_pow2 of the reduction extent). With no
        # such loop this is the single grid block at its floor, as before.
        reduction_loops: list[int | None] = [None] if persistent else [cls.LOOPED_CHUNK]
        seed: dict[str, Any] = {
            "block_sizes": cls._build_block_sizes(
                spec, fact, None, None, non_reduction_loop_ids=non_reduction_loop_ids
            ),
            "reduction_loops": reduction_loops,
            "num_warps": num_warps,
            "num_stages": 1,
            # 'flat' is the default: these reductions are grid-saturated at the
            # M-grid. The wide-looped-reread path below is the one exception.
            "pid_type": "flat",
        }
        # Eviction: single streamed input -> 'first' everywhere; a looped re-read
        # row -> its first load 'last', rest 'first'. Persistent rows are
        # register/SMEM-resident across passes, so eviction is left at default.
        evict = None
        if fact.num_load == 1:
            evict = cls._eviction_policies(env, "stream")
        elif fact.row_reread and not persistent:
            slot = device_ir.reread_eviction_slot_for_config(
                fact.reread_buffer_name, Config(**seed), env
            )
            evict = cls._eviction_policies(env, "reread", slot)
        if evict is not None:
            seed["load_eviction_policies"] = evict
        # OVERFIT NOTE: the following ``persistent_interleaved`` grid is the most
        # workload-specific lever in this file — it was tuned for the wide-reread
        # few-long-rows class (cross_entropy at large V) and is NOT generalized. It is
        # shipped as-is (it beats 'flat' on that class) but the sm_mult formula and
        # maxnreg=64 are not validated beyond it; revisit before relying on it elsewhere.
        #
        # A wide looped re-read T1 reduction is a few-long-rows workload (M rows <<
        # program capacity, each grinding a heavy multi-pass re-read). A
        # ``persistent_interleaved`` grid of ~one resident program per row plus a
        # register cap beats the default 'flat' here. num_sm_multiplier sizes the
        # grid to the row count (clamp(np2(ceil(M / num_sm)), 1, 32)); maxnreg=64 is
        # a high-occupancy cap to hide the re-read memory latency.
        if fact.row_reread and not persistent:
            # local import: ``helion.runtime`` imports the heuristics, so a
            # module-level import here is circular.
            from ...runtime import get_num_sm

            # grid_rows = product of the M-axis extents, via env.size_hint
            # (BlockSizeInfo.size_hint() needs env to be the current context,
            # not guaranteed at seed-emit).
            grid_rows = 1
            for _mbid in fact.m_block_ids:
                # pyrefly: ignore [bad-argument-type]
                grid_rows *= env.size_hint(env.block_sizes[_mbid].size)
            num_sm = max(1, get_num_sm(env.device))
            sm_mult = min(32, max(1, _np2(-(-grid_rows // num_sm))))
            seed["pid_type"] = "persistent_interleaved"
            seed["num_sm_multiplier"] = sm_mult
            seed["maxnreg"] = 64
        return Config(**seed)


class TritonReductionUserTileHeuristic(_TritonReductionSeedBase):
    """T2 (user-tiled) inner-reduction seed: fires on a T2 reduction (the reduction
    axis is an ordinary ``block_sizes`` entry, i.e. a user
    ``hl.tile(n, block_size=R_BLOCK)``), which the upstream gate rejects entirely.
    Covers three sub-regimes via one linear path: the R_BLOCK tile starts at the shared
    persistent-vs-looped extent and is then capped by this workload's live state (the
    regimes are mutually exclusive), keyed on workload facts:

    - **plain T2** (softmax_two_pass): no cap — persistent full-pow2 R_BLOCK, with the
      same reread-eviction as T1 for wide looped rows.
    - **Band B** (kl_div, jsd): carries ``[M_BLOCK, R_BLOCK]`` 2-D tiles across the
      inner loop, so a full-N R_BLOCK spills — cap it by
      ``BANDB_R_BLOCK_BYTES / (itemsize * num_carried_2d_tiles)``.
    - **Band C** (welford, ``non_reduction_loop_block_ids`` non-empty): a
      reduce-then-apply kernel — cap the combine tile by ``STRUCTURED_COMBINE_CAP_BYTES``
      and widen the normalize loop tile(s) separately (see ``_build_block_sizes``).

    TODO(reductions): the bands are handled inline here as workload-fact caps. As more
    structured-reduction families land, consider promoting each into its own
    ``AutotunerHeuristic`` subclass keyed on a dedicated fact rather than growing this
    method — the registry already supports several heuristics per backend.
    """

    name = "triton_reduction_user_tile"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if not _triton_reduction_eligible(env, device_ir):
            return False
        spec = env.config_spec
        return not _is_t1_reduction(spec, spec.reduction_facts[0])

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            # The T2 seeds are H100-tuned, and the upstream heuristic never fired on
            # T2, so off sm90 there is no prior seed to preserve: decline and search.
            return None
        from ..._utils import next_power_of_2 as _np2

        spec = env.config_spec
        fact = spec.reduction_facts[0]
        persistent, extent, num_warps = cls._persistent_looped(env, fact)

        # T2 (user-tiled): the reduction axis IS a block_sizes entry (the inner
        # `hl.tile(n, block_size=R_BLOCK)`); there is no `reduction_loops` knob.
        # Persistent == R_BLOCK >= next_pow2(N). Every other block_size (the grid/row
        # axes) stays at its floor — for the Band-B loss kernels this keeps M_BLOCK at 1,
        # required by the u0*u1 <= 2**20 numel constraint. The reduction (R_BLOCK) tile
        # starts at the shared persistent-vs-looped extent and is then capped by this
        # workload's live state, if any (the three sub-regimes are mutually exclusive):
        r_block = extent
        non_reduction_loop_ids = set(fact.non_reduction_loop_block_ids)
        if fact.num_carried_2d_tiles >= 1:
            # Band B (kl_div, jsd): carries one-or-more [M_BLOCK, R_BLOCK] 2-D tiles
            # across the inner loop. A full-N persistent R_BLOCK over-allocates that live
            # state and spills, so cap R_BLOCK to keep the footprint
            # (R_BLOCK * itemsize * n_carried) SM-resident. num_carried_2d_tiles both
            # routes (>= 1) and sizes (the divisor); max(1, ...) guards a 0-divide.
            cap = cls.BANDB_R_BLOCK_BYTES // (
                max(1, fact.itemsize) * max(1, fact.num_carried_2d_tiles)
            )
            r_block = min(r_block, _np2(max(1, cap)))
        elif non_reduction_loop_ids:
            # Band C (welford): a reduce-then-apply combine. The combine pass is a serial
            # scalar recurrence (count/mean/M2) that prefers to stay persistent, so cap
            # its tile by the spill-safe byte budget; the normalize loop tile(s) are
            # widened separately inside _build_block_sizes (a single-axis seed would
            # floor them to width 1).
            #
            # TODO(reductions): this combine/normalize sizing — taken whenever a
            # reduction loop is followed by a non-reduction loop — uses per-N byte caps
            # (STRUCTURED_COMBINE_CAP_BYTES here, the normalize-tile cap in
            # _build_block_sizes) that are NOT M_BLOCK-aware. That is a PROXY: the real
            # spill driver is the coupled working-tile footprint M_BLOCK * tile *
            # itemsize, so the principled seed is a block-M-aware (M,N)-keyed rule, not
            # independent per-N caps. The caps are tuned conservatively and must NOT be
            # LOOSENED without a full-M-range A/B: raising the normalize cap
            # (2048->4096) once regressed welford(262144,5120) ~7.3x (a high-M shape a
            # narrow A/B missed). Make this path block-M-aware before relying on it
            # beyond the current curriculum.
            cap = cls.STRUCTURED_COMBINE_CAP_BYTES // max(1, fact.itemsize)
            r_block = min(r_block, _np2(max(1, cap)))

        seed: dict[str, Any] = {
            "block_sizes": cls._build_block_sizes(
                spec, fact, fact.block_id, r_block,
                non_reduction_loop_ids=non_reduction_loop_ids,
            ),
            "num_warps": num_warps,
            "num_stages": 1,
            "pid_type": "flat",  # see the T1 branch.
        }
        # Reread eviction: a reduce-then-apply combine (welford) always re-reads its
        # input across the combine + normalize passes, so it gets 'last' on the row's
        # first load regardless of persistence; the plain-T2 path (a wide softmax_two_pass
        # re-reading x across the max + exp-sum passes) only when the row is re-read AND
        # looped. kl_div/jsd are row_reread=False with no normalize loop, so unaffected.
        if non_reduction_loop_ids or (fact.row_reread and not persistent):
            slot = device_ir.reread_eviction_slot_for_config(
                fact.reread_buffer_name, Config(**seed), env
            )
            ev = cls._eviction_policies(env, "reread", slot)
            if ev is not None:
                seed["load_eviction_policies"] = ev
        return Config(**seed)
