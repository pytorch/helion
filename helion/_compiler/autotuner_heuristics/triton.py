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
    """Gate: exactly one ``ReductionFact`` and no ``matmul_facts``. Admits both tracks
    (T1 rollable, T2 user-tiled); excludes GEMMs and multi-axis manual reductions."""
    spec = env.config_spec
    return len(spec.reduction_facts) == 1 and not spec.matmul_facts


def _is_t1_reduction(spec: ConfigSpec, fact: ReductionFact) -> bool:
    """T1 vs T2 discriminator: T1 iff the rdim is a rollable ``reduction_loops`` entry,
    else T2 (a ``block_sizes`` entry). Exhaustive over eligible reductions (the two
    device_ir populators are mutually exclusive).
    """
    return fact.block_id in spec.reduction_loops.valid_block_ids()


class _TritonReductionSeedBase(AutotunerHeuristic):
    """Shared base for the two Triton inner-reduction seed heuristics. Both share the
    workload facts (``ReductionFact``), the persistent-vs-looped lever
    (``_persistent_looped``), the ``num_warps`` ramp, eviction provenance, and the
    block-size builders; the subclasses differ only in mapping that decision onto knobs:

    - **T1** (:class:`TritonReductionTileHeuristic`): rollable rdim, rides
      ``reduction_loops``.
    - **T2** (:class:`TritonReductionUserTileHeuristic`): user-tiled, the reduction axis
      is a ``block_sizes`` entry (plain-T2 softmax, Band-B kl_div/jsd, Band-C welford).

    Cloned from ``cute.CuteReductionTileHeuristic`` for triton (drops the CuTe-only
    knobs, adds ``num_warps`` / ``num_stages``). Not registered; only the subclasses are.
    """

    backend = "triton"
    HARDWARE_TARGETS = (("cuda", "sm90"),)

    # Looped-fallback chunk (pow2) for rows above the structural cap (rnumel > 2**20).
    LOOPED_CHUNK = 16384
    # num_warps for the looped branch (the huge-rnumel streaming class wants 32).
    LOOPED_NUM_WARPS = 32
    # Band-B (T2 carrying [M_BLOCK, R_BLOCK] 2-D accumulators: kl_div, jsd) R_BLOCK cap,
    # in bytes/accumulator: a full-N persistent R_BLOCK spills, so cap the footprint at
    # R_BLOCK * itemsize * n_carried. Bytes (via itemsize) for dtype-generality.
    BANDB_R_BLOCK_BYTES = 16384
    # Per-row persistent byte ceiling, above which the reduction loops a fixed chunk (a
    # wide resident row spills register/SMEM). Caps on real row bytes (size_hint *
    # itemsize), not next_pow2 (vocab sizes sharing a next_pow2 split at the crossover).
    # 240 KiB sits in the measured dead-zone (most impactful on re-read kernels like CE).
    ROW_PERSIST_MAX_BYTES = 245760
    # Band-C (welford reduce-then-apply) combine-tile cap, in bytes. The combine is a
    # serial scalar recurrence (count/mean/M2) that prefers persistent; 32 KiB keeps it
    # so for the welford shapes (N<=8192).
    STRUCTURED_COMBINE_CAP_BYTES = 32768

    @classmethod
    def _num_warps(cls, fact: ReductionFact) -> int:
        """Scale num_warps with the reduction extent (pow2, per NumWarpsFragment):
        rnumel <= 1024 -> 4, <= 4096 -> 8, <= 16384 -> 16, > 16384 -> 32. Too few
        under-occupies the SM, too many wastes the reduction tree.
        """
        # >16384 (not >=) so sum's widest in-sample row (16384) stays w16, excluding the
        # tiny-rnumel w32 regression.
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
        """The smallest valid block size for an entry, used for every non-reduction axis
        the seed does not widen. Prefers one row/program but honors a raised
        ``autotuner_min`` (large-M shapes) rather than emitting an invalid ``block_size=1``.
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
        """Build the ``block_sizes`` list: the reduction axis gets ``red_value``, each
        non-reduction loop tile (``non_reduction_loop_ids``, disjoint from the reduction
        block_id) gets the widened size derived from ``fact``, every other axis its
        ``_block_floor``. ``red_block_id`` is None for T1 (the reduction rides
        ``reduction_loops``, not a block_sizes entry).

        A non-reduction loop tile is a pure perf lever (the normalize pass carries no
        accumulator): persistent ``next_pow2(N)`` when per-row valid work is small,
        looped above. Keys on per-row valid bytes (``n_valid * itemsize``) so rows
        sharing a next_pow2 separate, and is NOT tied to ``red_value`` (welford wants a
        narrower normalize tile than its combine tile).

        Fallback when the extent is not static (``static_rnumel is None``): the byte cap
        has no extent to key on, so match the reduction tile — ``red_value`` (T2) or
        ``next_pow2(size_hint)`` (T1). NOT tuned on any kernel (none has a dynamic-extent
        non-reduction loop).
        """
        from ..._utils import next_power_of_2 as _np2

        # Per-row valid bytes below which the loop tile stays persistent at next_pow2(N).
        persist_max_bytes = 12288
        # Looped chunk (bytes) above that threshold. Do NOT raise without re-gating:
        # 4096 is a regression valley at the large-M / N~5120 class.
        loop_chunk_bytes = 8192

        loop_block: int | None = None
        if non_reduction_loop_ids:
            if fact.static_rnumel is None:
                # Untuned dynamic-extent default: match the reduction tile (``red_value``
                # for T2; ``next_pow2(size_hint)`` for T1, where red_value is None).
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
        """``load_eviction_policies`` list (spec length), keyed on per-load residency;
        None leaves the autotuner default.

        - ``"stream"`` — single streamed input (``num_load == 1``: sum, long_sum), read
          once: every load -> ``'first'`` (frees L2).
        - ``"reread"`` — the row is re-read across passes: its first load -> ``'last'``
          (L2-resident), rest -> ``'first'``. ``reread_slot`` is that load's actual slot,
          resolved for the config via ``reread_eviction_slot_for_config``, not guessed.

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
        """The shared first lever (both tracks); returns ``(persistent, extent,
        num_warps)``. Persistent for every row the backend compiles in one pass (up to
        max_tensor_numel = 2**20 elems) AND within the per-row byte ceiling
        (ROW_PERSIST_MAX_BYTES, the register/SMEM spill limit); else loop a fixed chunk.

        The byte cap is unconditional but config-neutral off the re-read kernels (a
        single-load looped chunk ties the persistent pass; the Band-B cap is already
        tighter), so only rms_norm/layer_norm/softmax/cross_entropy/welford are steered.
        """
        from ..._utils import next_power_of_2 as _np2

        # Persistent iff BOTH: the element cap (None => no cap, a compile limit) AND the
        # byte ceiling (a residency limit) — distinct limits.
        element_cap = env.backend.max_tensor_numel
        can_persist = (element_cap is None or fact.size_hint <= element_cap) and (
            fact.size_hint * max(1, fact.itemsize) <= cls.ROW_PERSIST_MAX_BYTES
        )

        if can_persist:
            # Persistent: T1 encodes the extent as reduction_loops=None; T2 as the full
            # pow2 R_BLOCK so the inner `for tile_n` runs once.
            return True, _np2(fact.size_hint), cls._num_warps(fact)
        # Looped: exceeds the 2**20 or byte cap. Fixed chunk + high streaming warps.
        return False, cls.LOOPED_CHUNK, cls.LOOPED_NUM_WARPS


class TritonReductionTileHeuristic(_TritonReductionSeedBase):
    """T1 (rollable-rdim) inner-reduction seed: sum, long_sum, rms_norm, layer_norm,
    softmax-row, cross_entropy. Triton analog of ``CuteReductionTileHeuristic`` (keeps
    its registry name), deepening the original one-row/persistent/``['last']`` seed with
    the num_warps ramp, persistent-vs-looped, per-slot eviction, and a
    ``persistent_interleaved`` grid for wide looped re-read rows.

    Gated by ``_triton_reduction_eligible`` (T1 track) — broader than upstream
    ``is_canonical_row_reduction`` (also multi-axis rollable rows, raised-``autotuner_min``
    large-M shapes). Off sm90 the H100-tuned levers are unvalidated, so it falls back to
    ``_narrow_seed`` (pre-existing behavior preserved).
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
        """The upstream conservative T1 seed (one row/program, single persistent pass,
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
        from ..._utils import next_power_of_2 as _np2

        spec = env.config_spec
        fact = spec.reduction_facts[0]
        # T1 rides persistent-vs-looped on `reduction_loops`, so the lever's `extent`
        # (the T2 R_BLOCK) is unused here.
        persistent, _extent, num_warps = cls._persistent_looped(env, fact)

        # A T1 reduction may be followed by a normalize loop (e.g. `s = x.sum(); out =
        # x/s`); its extra block_sizes tile(s) are widened by _build_block_sizes. NOT
        # perf-validated (no curriculum kernel), but it is only a seed (a worse tile
        # costs autotuning time, never correctness), so emit and let the autotuner refine.
        non_reduction_loop_ids = set(fact.non_reduction_loop_block_ids)

        # red_block_id=None: the rdim is not a block_sizes entry, so every entry is a
        # grid axis (floored) or a normalize loop tile (widened). None loop => the single
        # grid block at its floor, as before.
        reduction_loops: list[int | None] = [None] if persistent else [cls.LOOPED_CHUNK]
        seed: dict[str, Any] = {
            "block_sizes": cls._build_block_sizes(
                spec, fact, None, None, non_reduction_loop_ids=non_reduction_loop_ids
            ),
            "reduction_loops": reduction_loops,
            "num_warps": num_warps,
            "num_stages": 1,
            # 'flat': these reductions are grid-saturated at the M-grid. The
            # wide-looped-reread path below is the one exception.
            "pid_type": "flat",
        }
        # Eviction: streamed input -> 'first' everywhere; looped re-read -> first load
        # 'last', rest 'first'. Persistent rows stay resident, so left at default.
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
        # OVERFIT NOTE: this ``persistent_interleaved`` grid is tuned for the wide-reread
        # few-long-rows class (cross_entropy at large V) and is NOT generalized — it beats
        # 'flat' there, but the sm_mult formula and maxnreg=64 are unvalidated beyond it;
        # re-validate them before relying on it elsewhere.
        # ~one resident program per row + a register cap to hide the re-read latency;
        # num_sm_multiplier sizes the grid to the row count.
        if fact.row_reread and not persistent:
            # local import: helion.runtime imports the heuristics (circular at module level).
            from ...runtime import get_num_sm

            # grid_rows = product of M-axis extents (env.size_hint needs env current).
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
    Covers three mutually-exclusive sub-regimes in one linear path: R_BLOCK starts at the
    shared persistent-vs-looped extent, then is capped by this workload's live state:

    - **plain T2** (softmax_two_pass): no cap — persistent full-pow2 R_BLOCK, T1-style
      reread-eviction for wide looped rows.
    - **Band B** (kl_div, jsd): carries ``[M_BLOCK, R_BLOCK]`` 2-D tiles, so a full-N
      R_BLOCK spills — cap by ``BANDB_R_BLOCK_BYTES / (itemsize * num_carried_2d_tiles)``.
    - **Band C** (welford, ``non_reduction_loop_block_ids`` non-empty): reduce-then-apply
      — cap the combine tile by ``STRUCTURED_COMBINE_CAP_BYTES``, widen normalize tile(s)
      separately (see ``_build_block_sizes``).

    TODO(reductions): as more structured families land, promote each band into its own
    fact-keyed ``AutotunerHeuristic`` subclass rather than growing this method.
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
            # Off sm90: upstream never fired on T2, so no prior seed to preserve. Decline.
            return None
        from ..._utils import next_power_of_2 as _np2

        spec = env.config_spec
        fact = spec.reduction_facts[0]
        persistent, extent, num_warps = cls._persistent_looped(env, fact)

        # T2: the rdim IS a block_sizes entry (no reduction_loops knob); persistent ==
        # R_BLOCK >= next_pow2(N). Other axes stay at floor (keeps Band-B M_BLOCK at 1,
        # required by the u0*u1 <= 2**20 constraint). R_BLOCK starts at the lever extent,
        # then is capped by live state (the three sub-regimes are mutually exclusive):
        r_block = extent
        non_reduction_loop_ids = set(fact.non_reduction_loop_block_ids)
        if fact.num_carried_2d_tiles >= 1:
            # Band B (kl_div, jsd): a full-N R_BLOCK over-allocates the carried 2-D tiles
            # and spills, so cap the footprint (R_BLOCK * itemsize * n_carried)
            # SM-resident. num_carried_2d_tiles routes (>=1) and sizes; max(1,..) guards 0.
            cap = cls.BANDB_R_BLOCK_BYTES // (
                max(1, fact.itemsize) * max(1, fact.num_carried_2d_tiles)
            )
            r_block = min(r_block, _np2(max(1, cap)))
        elif non_reduction_loop_ids:
            # Band C (welford): the combine is a serial scalar recurrence that prefers
            # persistent, so cap its tile by the spill-safe budget; normalize tile(s) are
            # widened separately in _build_block_sizes.
            #
            # TODO(reductions): these per-N caps are a PROXY — the real spill driver is
            # the coupled M_BLOCK * tile * itemsize, so the principled seed is
            # block-M-aware. Do NOT LOOSEN without a full-M-range A/B: raising the
            # normalize cap (2048->4096) once regressed welford(262144,5120) ~7.3x.
            cap = cls.STRUCTURED_COMBINE_CAP_BYTES // max(1, fact.itemsize)
            r_block = min(r_block, _np2(max(1, cap)))

        seed: dict[str, Any] = {
            "block_sizes": cls._build_block_sizes(
                spec,
                fact,
                fact.block_id,
                r_block,
                non_reduction_loop_ids=non_reduction_loop_ids,
            ),
            "num_warps": num_warps,
            "num_stages": 1,
            "pid_type": "flat",  # see the T1 branch.
        }
        # Reread eviction: welford (reduce-then-apply) always re-reads across combine +
        # normalize, so 'last' regardless of persistence; plain-T2 (softmax_two_pass) only
        # when re-read AND looped. kl_div/jsd (row_reread=False, no normalize) unaffected.
        if non_reduction_loop_ids or (fact.row_reread and not persistent):
            slot = device_ir.reread_eviction_slot_for_config(
                fact.reread_buffer_name, Config(**seed), env
            )
            ev = cls._eviction_policies(env, "reread", slot)
            if ev is not None:
                seed["load_eviction_policies"] = ev
        return Config(**seed)
