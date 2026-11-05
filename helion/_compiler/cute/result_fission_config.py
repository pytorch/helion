"""Config keys of the CuTe result-fission schedule.

``cute_result_fission`` selects, per config, between the ordinary fused
lowering (default, ``False``) and a two-launch schedule for a kernel whose
proved epilogue stores one pure FP32 identity of the native accumulator beside
1-3 other FP32 outputs: a native TCGEN05 producer GEMM writes the identity
output, then a CuTe SIMT consumer reads it back and computes the other original
stores.  The producer and consumer have their own tile keys; the fused
kernel's pipeline keys are inert under the schedule (exactly like the split-K
workspace schedule's producer).
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import TypeGuard
from typing import cast

from ...autotuner.config_fragment import ConfigSpecFragment
from ...autotuner.config_fragment import EnumFragment
from ...autotuner.config_fragment import FragmentFingerprint
from ...autotuner.config_fragment import IntegerFragment
from ...autotuner.config_fragment import ListOf
from ...exc import InvalidConfig

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Mapping

RESULT_FISSION_KEY = "cute_result_fission"
PRODUCER_BLOCK_SIZES_KEY = "cute_result_fission_producer_block_sizes"
PRODUCER_PID_TYPE_KEY = "cute_result_fission_producer_pid_type"
PRODUCER_AB_STAGES_KEY = "cute_result_fission_producer_ab_stages"
PRODUCER_ACC_STAGES_KEY = "cute_result_fission_producer_acc_stages"
PRODUCER_C_STAGES_KEY = "cute_result_fission_producer_c_stages"
PRODUCER_CLUSTER_M_KEY = "cute_result_fission_producer_cluster_m"
PRODUCER_CTA_GROUP_KEY = "cute_result_fission_producer_cta_group"
CONSUMER_BLOCK_SIZES_KEY = "cute_result_fission_consumer_block_sizes"
CONSUMER_NUM_THREADS_KEY = "cute_result_fission_consumer_num_threads"
CONSUMER_VECTOR_WIDTH_KEY = "cute_result_fission_consumer_vector_width"

PRODUCER_KEYS = (
    PRODUCER_BLOCK_SIZES_KEY,
    PRODUCER_PID_TYPE_KEY,
    PRODUCER_AB_STAGES_KEY,
    PRODUCER_ACC_STAGES_KEY,
    PRODUCER_C_STAGES_KEY,
    PRODUCER_CLUSTER_M_KEY,
    PRODUCER_CTA_GROUP_KEY,
)
CONSUMER_KEYS = (
    CONSUMER_BLOCK_SIZES_KEY,
    CONSUMER_NUM_THREADS_KEY,
    CONSUMER_VECTOR_WIDTH_KEY,
)
RESULT_FISSION_CONFIG_KEYS = frozenset(
    (RESULT_FISSION_KEY, *PRODUCER_KEYS, *CONSUMER_KEYS)
)

PRODUCER_PID_TYPES = ("flat", "persistent_interleaved", "persistent_blocked")
# The producer template's own normalizer admits ``auto`` (one CTA, or the
# clustered CtaGroup.ONE / bm=256 pair family) and the explicit pair ``two``.
PRODUCER_CTA_GROUPS = ("auto", "two")
# Producer tile axes the TCGEN05 template admits (bf16/f16 operands).
PRODUCER_BLOCK_M = (128, 256, 64)
PRODUCER_BLOCK_N = (128, 256, 64, 32, 16)
PRODUCER_BLOCK_K = (64, 128, 32, 16)
# The explicit CTA pair (``_tcgen05_use_2cta_instrs`` with cta_group='two',
# and ``pipeline_smem_bytes``' pair tile asserts).
PAIR_BLOCK_M = (128, 256)
PAIR_BLOCK_N = (64, 128, 256)
PAIR_BLOCK_K = (64, 128)
_DEFAULT_PRODUCER_BLOCKS = (128, 128, 64)
# The producer normalizer admits one or two accumulator stages and two to
# four C stages (never one); the AB ring is priced per schedule.
PRODUCER_ACC_STAGES = (2, 1)
PRODUCER_C_STAGES = (2, 4, 3)
MAX_PRODUCER_AB_STAGES = 16
CONSUMER_VECTOR_WIDTHS = (1, 2, 4)
_POWERS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def default_consumer_block_sizes(m: int, n: int) -> list[int]:
    """Coalesced scalar default: a whole number of warps along N (the stride-1
    output axis, mapped onto ``threadIdx.x``), rows filling one 1024-thread
    block where the extents allow it."""
    bn = next((p for p in (256, 128, 64, 32) if n % p == 0), 32)
    bm = next((p for p in (4, 2, 1) if m % p == 0 and p * bn <= 1024), 1)
    return [bm, bn]


def _is_int_list(value: object, length: int) -> TypeGuard[list[int]]:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(type(item) is int for item in value)
    )


def _producer_defaults() -> dict[str, object]:
    return {
        PRODUCER_PID_TYPE_KEY: "persistent_interleaved",
        PRODUCER_AB_STAGES_KEY: 4,
        PRODUCER_ACC_STAGES_KEY: 2,
        PRODUCER_C_STAGES_KEY: 2,
        PRODUCER_CLUSTER_M_KEY: 1,
        PRODUCER_CTA_GROUP_KEY: "auto",
    }


def _pow2_ceiling(value: int) -> int:
    return 1 << max(value - 1, 0).bit_length()


def derived_consumer_threads(blocks: list[int], vector: int) -> list[int]:
    """One thread per row element group: ``vector`` contiguous N elements."""
    return [blocks[0], blocks[1] // vector]


def consumer_layout_valid(blocks: object, vector: object, m: int, n: int) -> bool:
    if not (_is_int_list(blocks, 2) and all(b in _POWERS for b in blocks)):
        return False
    if vector not in CONSUMER_VECTOR_WIDTHS:
        return False
    assert isinstance(vector, int)
    bm, bn = blocks
    threads_n = bn // vector
    return (
        bn % vector == 0
        and threads_n % 32 == 0
        and bm * threads_n <= 1024
        and bm <= _pow2_ceiling(m)
        and bn <= _pow2_ceiling(n)
    )


ProducerSchedule = tuple[int, int, int, int, str, int, int, str]
"""``(bm, bn, bk, cluster_m, cta_group, acc_stages, c_stages, pid_type)``."""


class ProducerAdmission(NamedTuple):
    """The deepest AB ring the producer template's own strict normalizer
    admits for each statically legal producer schedule of this kernel (0:
    none).  Computed once per kernel at registration from the compiled
    producer child, so pricing follows its effective CTA plan, C ring and
    shared-memory model exactly instead of a separate estimate."""

    max_ab_stages: Mapping[ProducerSchedule, int]


def producer_schedules() -> Iterator[ProducerSchedule]:
    """Every producer schedule the static checks admit (any AB depth)."""
    for bm, bn, bk, cluster_m, cta_group, acc, c, pid in itertools.product(
        PRODUCER_BLOCK_M,
        PRODUCER_BLOCK_N,
        PRODUCER_BLOCK_K,
        (1, 2),
        PRODUCER_CTA_GROUPS,
        PRODUCER_ACC_STAGES,
        PRODUCER_C_STAGES,
        PRODUCER_PID_TYPES,
    ):
        if _static_plan_problem([bm, bn, bk], cluster_m, cta_group, pid) is None:
            yield (bm, bn, bk, cluster_m, cta_group, acc, c, pid)


def producer_schedule_keys(
    schedule: ProducerSchedule, ab_stages: int
) -> dict[str, object]:
    bm, bn, bk, cluster_m, cta_group, acc, c, pid = schedule
    return {
        PRODUCER_BLOCK_SIZES_KEY: [bm, bn, bk],
        PRODUCER_PID_TYPE_KEY: pid,
        PRODUCER_AB_STAGES_KEY: ab_stages,
        PRODUCER_ACC_STAGES_KEY: acc,
        PRODUCER_C_STAGES_KEY: c,
        PRODUCER_CLUSTER_M_KEY: cluster_m,
        PRODUCER_CTA_GROUP_KEY: cta_group,
    }


def producer_runs_cta_pair(
    blocks: list[int], *, cluster_m: int, cta_group: str
) -> bool:
    """The producer's effective CTA plan (``_config_is_two_cta_mma``): a
    two-CTA cluster runs the tcgen05 pair for an explicit ``two`` or the
    bm=256 pair tile (the fp8 small-grid pair never applies: the proof admits
    bf16/f16 operands); any other two-CTA cluster is the clustered
    single-CTA family."""
    return cluster_m == 2 and (cta_group == "two" or blocks[0] == 256)


def _static_plan_problem(
    blocks: list[int], cluster_m: int, cta_group: str, pid_type: str
) -> str | None:
    if cluster_m == 2 and pid_type == "flat":
        # Clustered tcgen05 codegen is guarded to persistent pid schedules
        # (it raises BackendUnsupported for a flat launch).
        return (
            f"{PRODUCER_CLUSTER_M_KEY}=2 requires a persistent "
            f"{PRODUCER_PID_TYPE_KEY}, got 'flat'"
        )
    if cluster_m == 2 and not producer_runs_cta_pair(
        blocks, cluster_m=cluster_m, cta_group=cta_group
    ):
        # The clustered single-CTA family lowers without the TMA store ring
        # (a different, unreviewed epilogue protocol): not a producer plan.
        return (
            f"{PRODUCER_CLUSTER_M_KEY}=2 with {blocks!r} and "
            f"{PRODUCER_CTA_GROUP_KEY}={cta_group!r} is the clustered single-CTA "
            "family, which has no TMA store ring"
        )
    if cta_group == "two" and (
        cluster_m != 2
        or blocks[0] not in PAIR_BLOCK_M
        or blocks[1] not in PAIR_BLOCK_N
        or blocks[2] not in PAIR_BLOCK_K
    ):
        return (
            f"{PRODUCER_CTA_GROUP_KEY}='two' requires {PRODUCER_CLUSTER_M_KEY}=2 "
            f"and a pair tile (bm in {PAIR_BLOCK_M}, bn in {PAIR_BLOCK_N}, "
            f"bk in {PAIR_BLOCK_K}), got {blocks!r} cluster_m={cluster_m}"
        )
    return None


def _schedule_of(config: dict[str, object]) -> ProducerSchedule:
    blocks = cast("list[int]", config[PRODUCER_BLOCK_SIZES_KEY])
    return (
        blocks[0],
        blocks[1],
        blocks[2],
        cast("int", config[PRODUCER_CLUSTER_M_KEY]),
        cast("str", config[PRODUCER_CTA_GROUP_KEY]),
        cast("int", config[PRODUCER_ACC_STAGES_KEY]),
        cast("int", config[PRODUCER_C_STAGES_KEY]),
        cast("str", config[PRODUCER_PID_TYPE_KEY]),
    )


def producer_schedule_problem(
    config: dict[str, object], admission: ProducerAdmission | None
) -> str | None:
    """Why the producer keys are not an admissible schedule (``None`` if
    they are): the template's key domains, the pair's tile and cluster
    requirements, then the producer normalizer's own AB-ring admission."""
    for key in PRODUCER_KEYS:
        if _single_key_invalid(config, key):
            return f"invalid {key}={config.get(key)!r}"
    schedule = _schedule_of(config)
    problem = _static_plan_problem(
        list(schedule[:3]), schedule[3], schedule[4], schedule[7]
    )
    if problem is not None:
        return problem
    if admission is not None:
        limit = admission.max_ab_stages.get(schedule, 0)
        stages = cast("int", config[PRODUCER_AB_STAGES_KEY])
        if stages > limit:
            return (
                f"{PRODUCER_AB_STAGES_KEY}={stages} is not admitted by the producer "
                f"for schedule {schedule} ({limit} is its deepest AB ring)"
            )
    return None


def _clamp_ab(config: dict[str, object], admission: ProducerAdmission | None) -> None:
    if admission is None or producer_schedule_problem(config, None) is not None:
        return
    limit = admission.max_ab_stages.get(_schedule_of(config), 0)
    if 1 <= limit < cast("int", config[PRODUCER_AB_STAGES_KEY]):
        config[PRODUCER_AB_STAGES_KEY] = limit


def _repair_producer(
    config: dict[str, object], admission: ProducerAdmission | None
) -> bool:
    """Repair a search config's producer keys to an admissible schedule,
    revalidating every candidate; ``False`` when none is admissible."""
    defaults = {
        PRODUCER_BLOCK_SIZES_KEY: list(_DEFAULT_PRODUCER_BLOCKS),
        **_producer_defaults(),
    }
    for key, default in defaults.items():
        if _single_key_invalid(config, key):
            config[key] = default
    if (
        config.get(PRODUCER_CTA_GROUP_KEY) == "two"
        and producer_schedule_problem(config, None) is not None
    ):
        config[PRODUCER_CTA_GROUP_KEY] = "auto"
    _clamp_ab(config, admission)
    if producer_schedule_problem(config, admission) is None:
        return True
    # Last resort: the default producer schedule (its AB ring clamped to its
    # own admission), revalidated like any other candidate.
    fallback = {**config, **defaults}
    _clamp_ab(fallback, admission)
    if producer_schedule_problem(fallback, admission) is None:
        config.update(fallback)
        return True
    return False


def _single_key_invalid(config: dict[str, object], key: str) -> bool:
    value = config.get(key)
    if key == PRODUCER_BLOCK_SIZES_KEY:
        return not (
            _is_int_list(value, 3)
            and value[0] in PRODUCER_BLOCK_M
            and value[1] in PRODUCER_BLOCK_N
            and value[2] in PRODUCER_BLOCK_K
        )
    if key == PRODUCER_PID_TYPE_KEY:
        return value not in PRODUCER_PID_TYPES
    if key == PRODUCER_CTA_GROUP_KEY:
        return value not in PRODUCER_CTA_GROUPS
    if key == PRODUCER_CLUSTER_M_KEY:
        return value not in (1, 2)
    if key == PRODUCER_ACC_STAGES_KEY:
        return value not in PRODUCER_ACC_STAGES
    if key == PRODUCER_C_STAGES_KEY:
        return value not in PRODUCER_C_STAGES
    return type(value) is not int or not 1 <= value <= MAX_PRODUCER_AB_STAGES


def normalize_result_fission_config(
    config: dict[str, object],
    *,
    available: bool,
    fix_invalid: bool,
    shape: tuple[int, int, int] | None,
    producer_facts: ProducerAdmission | None = None,
) -> None:
    """Strict stage checks: shapes, the producer's one-C-ring SMEM price, and
    the consumer's thread geometry.  The producer and consumer compilers
    still normalize their stage configs strictly at codegen."""
    present = [key for key in RESULT_FISSION_CONFIG_KEYS if key in config]
    if not available or shape is None:
        if (
            config.get(RESULT_FISSION_KEY, False) is not False
            or any(key != RESULT_FISSION_KEY for key in present)
        ) and not fix_invalid:
            raise InvalidConfig(
                f"{RESULT_FISSION_KEY} requires a proved pure FP32 identity "
                "output beside computed FP32 epilogue stores"
            )
        for key in RESULT_FISSION_CONFIG_KEYS:
            config.pop(key, None)
        return
    enabled = config.get(RESULT_FISSION_KEY, False)
    if type(enabled) is not bool:
        if not fix_invalid:
            raise InvalidConfig(f"{RESULT_FISSION_KEY} must be a boolean")
        enabled = False
    if not enabled:
        # Absent means off: configs that never name the schedule (every saved
        # and default config) normalize exactly as before; the inactive stage
        # coordinates of a search config canonicalize away.
        # A flat search config carries every stage coordinate at its inert
        # search default (strict coverage transfers normalize those), so only
        # a stage key that differs from that default is an explicit misuse.
        inert = _inert_stage_defaults()
        stale = [
            key
            for key in present
            if key != RESULT_FISSION_KEY and config.get(key) != inert.get(key)
        ]
        if stale and not fix_invalid:
            raise InvalidConfig(
                f"result-fission stage keys {sorted(stale)} require {RESULT_FISSION_KEY}"
            )
        for key in RESULT_FISSION_CONFIG_KEYS:
            config.pop(key, None)
        return
    m, n, _k = shape
    if config.get("cute_pdl", False) is not False and not fix_invalid:
        # The consumer has no proved griddepcontrol wait dominance: both
        # stages launch as ordinary same-stream kernels.
        raise InvalidConfig(f"{RESULT_FISSION_KEY} requires cute_pdl=False")

    config.setdefault(PRODUCER_BLOCK_SIZES_KEY, list(_DEFAULT_PRODUCER_BLOCKS))
    for key, default in _producer_defaults().items():
        config.setdefault(key, default)
    problem = producer_schedule_problem(config, producer_facts)
    if problem is not None:
        if not fix_invalid:
            raise InvalidConfig(problem)
        if not _repair_producer(config, producer_facts):
            # No admissible producer schedule at all: the schedule is off.
            for key in RESULT_FISSION_CONFIG_KEYS:
                config.pop(key, None)
            return
    if "cute_pdl" in config:
        # Only once the schedule stays on: a search config whose producer has
        # no admissible schedule keeps its fused cute_pdl unchanged.
        config["cute_pdl"] = False
    default_blocks = default_consumer_block_sizes(m, n)
    cblocks = config.setdefault(CONSUMER_BLOCK_SIZES_KEY, list(default_blocks))
    vector = config.setdefault(CONSUMER_VECTOR_WIDTH_KEY, 1)
    if not consumer_layout_valid(cblocks, vector, m, n):
        if not fix_invalid:
            raise InvalidConfig(
                f"{CONSUMER_BLOCK_SIZES_KEY}={cblocks!r} with "
                f"{CONSUMER_VECTOR_WIDTH_KEY}={vector!r} must keep each row on "
                "its own threads with whole warps of contiguous N vectors"
            )
        config[CONSUMER_BLOCK_SIZES_KEY] = cblocks = list(default_blocks)
        config[CONSUMER_VECTOR_WIDTH_KEY] = vector = 1
    assert _is_int_list(cblocks, 2) and isinstance(vector, int)
    derived = derived_consumer_threads(cblocks, vector)
    threads = config.setdefault(CONSUMER_NUM_THREADS_KEY, list(derived))
    if threads != derived:
        if not fix_invalid:
            raise InvalidConfig(
                f"{CONSUMER_NUM_THREADS_KEY}={threads!r} must be {derived!r}: every "
                f"row on its own threads, each thread {vector} contiguous N elements"
            )
        config[CONSUMER_NUM_THREADS_KEY] = derived


class _SeedReachedEnum(EnumFragment):
    """Random sampling never reaches the schedule (seeds and their mutations
    do): ``random()`` returns the default without consuming a random draw,
    so adding these coordinates leaves every random fused sample of the
    search exactly as it was."""

    def random(self) -> object:
        return self.default()


class _SeedReachedInteger(IntegerFragment):
    def random(self) -> int:
        return self.default()


class _SeedReachedList(ListOf):
    def random(self) -> list[object]:
        return self.default()


class _ProducerTileFragment(ListOf):
    """The producer tile ``[bm, bn, bk]`` with per-axis choices (the
    template's admitted axes), seed-reached like the other stage keys."""

    def __init__(self, axes: tuple[EnumFragment, ...]) -> None:
        super().__init__(inner=axes[0], length=len(axes))
        self.axes = axes

    def __eq__(self, other: object) -> bool:
        # The inherited dataclass equality compares only ``inner``/``length``.
        return isinstance(other, _ProducerTileFragment) and self.axes == other.axes

    __hash__ = None

    def default(self) -> list[object]:
        return [axis.default() for axis in self.axes]

    def random(self) -> list[object]:
        return self.default()

    def pattern_neighbors(self, current: object, radius: int = 1) -> list[object]:
        if not isinstance(current, list) or len(current) != self.length:
            raise ValueError(f"Expected list of length {self.length}, got {current!r}")
        neighbors: list[object] = []
        for index, axis in enumerate(self.axes):
            for value in axis.pattern_neighbors(current[index], radius):
                neighbor = list(current)
                neighbor[index] = value
                neighbors.append(neighbor)
        return neighbors

    def differential_mutation(self, a: object, b: object, c: object) -> list[object]:
        assert isinstance(a, list) and isinstance(b, list) and isinstance(c, list)
        return [
            axis.differential_mutation(a[i], b[i], c[i])
            for i, axis in enumerate(self.axes)
        ]

    def fingerprint(self) -> FragmentFingerprint:
        return (
            "axes",
            *(part for axis in self.axes for part in ("|", *axis.fingerprint())),
        )

    def dim(self) -> int:
        return sum(axis.dim() for axis in self.axes)

    def cardinality(self) -> int | None:
        total = 1
        for axis in self.axes:
            count = axis.cardinality()
            if count is None:
                return None
            total *= count
        return total

    def encode(self, value: object) -> list[float]:
        assert isinstance(value, list)
        encoded: list[float] = []
        for axis, item in zip(self.axes, value, strict=True):
            encoded.extend(axis.encode(item))
        return encoded


def _inert_stage_defaults() -> dict[str, object]:
    """Each stage coordinate's search default: its value in a flat config
    while the schedule is off (the derived thread count has no field)."""
    return {
        key: field.default()
        for key, field in result_fission_search_fields().items()
        if key != RESULT_FISSION_KEY
    }


def result_fission_search_fields() -> dict[str, ConfigSpecFragment]:
    """The schedule and its stage coordinates.  None of them consumes a
    random draw (``_SeedReached*``): random samples keep the schedule off and
    the fused samples are unchanged; the search reaches the schedule through
    the result-fission seeds and then mutates these coordinates, which are
    inert (and canonicalized away) while the schedule is off.  The consumer
    thread count is derived from its tile and vector width."""
    return {
        RESULT_FISSION_KEY: _SeedReachedEnum((False, True), search_choices=(False,)),
        PRODUCER_BLOCK_SIZES_KEY: _ProducerTileFragment(
            (
                EnumFragment(PRODUCER_BLOCK_M),
                EnumFragment(PRODUCER_BLOCK_N),
                EnumFragment(PRODUCER_BLOCK_K),
            )
        ),
        PRODUCER_PID_TYPE_KEY: _SeedReachedEnum(
            ("persistent_interleaved", "persistent_blocked", "flat")
        ),
        PRODUCER_AB_STAGES_KEY: _SeedReachedInteger(1, MAX_PRODUCER_AB_STAGES, 4),
        PRODUCER_ACC_STAGES_KEY: _SeedReachedEnum(PRODUCER_ACC_STAGES),
        PRODUCER_C_STAGES_KEY: _SeedReachedEnum(PRODUCER_C_STAGES),
        PRODUCER_CLUSTER_M_KEY: _SeedReachedEnum((1, 2)),
        PRODUCER_CTA_GROUP_KEY: _SeedReachedEnum(PRODUCER_CTA_GROUPS),
        CONSUMER_BLOCK_SIZES_KEY: _SeedReachedList(
            EnumFragment((256, 4, 1, 2, 8, 16, 32, 64, 128, 512, 1024)), length=2
        ),
        CONSUMER_VECTOR_WIDTH_KEY: _SeedReachedEnum(CONSUMER_VECTOR_WIDTHS),
    }
