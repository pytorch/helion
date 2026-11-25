from __future__ import annotations

from typing import TYPE_CHECKING

from ...autotuner.config_fragment import ConfigSpecFragment
from ...autotuner.config_fragment import EnumFragment
from ...autotuner.config_fragment import IntegerFragment
from ...exc import InvalidConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...autotuner.config_spec import ConfigSpec

WORKSPACE_KEY = "cute_split_k_workspace"
STAGES_KEY = "cute_split_k_stages"
# A plain GEMM store (``out[tile_m, tile_n] = acc``) has no partition axis of
# its own: this knob is the partition count of its workspace schedule.
PARTITIONS_KEY = "cute_split_k_partitions"
# Three partitions fill the CTA pairs' waves where the powers of two do not:
# 24, 96 or 192 tiles of 256x256 run 72, 288 or 576 units on GB300's 76 SM
# pairs, 95% of their waves (two or four partitions of 96 tiles: 84%). bf16
# 2048 x 768 x 20008 (24 tiles), of cuBLAS: 1.085 in three partitions, 0.88
# in six.
PARTITION_CHOICES: tuple[int, ...] = (1, 2, 3, 4, 8, 16, 32)
# The producer reduces its partition straight into the (zero-filled) output
# with packed atomic adds instead of storing fp32 partials for a reducer
# launch: one kernel per call, the ``hl.atomic_add`` numerics of the source
# (every partition rounded into the output's dtype) where the reducer form
# rounds the fp32 sum once.
ATOMIC_EPILOGUE_KEY = "cute_split_k_atomic_epilogue"
# The element type of a plain GEMM store's partials: "accumulator" stores
# fp32 partials, "output" rounds each partition's fp32 sum to the 16-bit
# output dtype (half the bytes the producer stores and the reducer loads; the
# reducer still adds them in fp32). "output" is offered only under
# ``Settings.split_k_reduction="output"`` (torch's reduced-precision split-K
# flags, through helion.inductor): ``split_k_workspace.partials_admissible``.
PARTIALS_KEY = "cute_split_k_partials"
PARTIALS_ACCUMULATOR = "accumulator"
PARTIALS_OUTPUT = "output"
PARTIALS_CHOICES: tuple[str, ...] = (PARTIALS_ACCUMULATOR, PARTIALS_OUTPUT)
WORKSPACE_CONFIG_KEYS = frozenset(
    (WORKSPACE_KEY, STAGES_KEY, PARTITIONS_KEY, ATOMIC_EPILOGUE_KEY, PARTIALS_KEY)
)


def normalize_workspace_config(
    config: dict[str, object],
    *,
    available: bool,
    fix_invalid: bool,
    partition_choices: tuple[int, ...] = (),
    atomic_epilogue_available: bool = False,
    partials_available: bool = False,
) -> None:
    if not available:
        if (
            config.get(WORKSPACE_KEY, False) is not False
            or config.get(STAGES_KEY, 2) != 2
            or config.get(PARTITIONS_KEY, 1) != 1
            or config.get(ATOMIC_EPILOGUE_KEY, False) is not False
            or config.get(PARTIALS_KEY, PARTIALS_ACCUMULATOR) != PARTIALS_ACCUMULATOR
        ) and not fix_invalid:
            raise InvalidConfig(
                "native split-K workspace requires a proved split-K reduction "
                "(an fp32/fp16/bf16 atomic add or a plain GEMM store)"
            )
        for key in WORKSPACE_CONFIG_KEYS:
            config.pop(key, None)
        return
    enabled = config.setdefault(WORKSPACE_KEY, False)
    stages = config.setdefault(STAGES_KEY, 2)
    if type(enabled) is not bool:
        if not fix_invalid:
            raise InvalidConfig(f"{WORKSPACE_KEY} must be a boolean")
        config[WORKSPACE_KEY] = False
    if type(stages) is not int or not 1 <= stages <= 16:
        if not fix_invalid:
            raise InvalidConfig(f"{STAGES_KEY} must be an integer in [1, 16]")
        config[STAGES_KEY] = 2
    if not atomic_epilogue_available:
        if config.get(ATOMIC_EPILOGUE_KEY, False) is not False and not fix_invalid:
            raise InvalidConfig(
                f"{ATOMIC_EPILOGUE_KEY} requires a proved atomic split-K reduction "
                "into a float32, float16 or bfloat16 output kept in its dtype"
            )
        config.pop(ATOMIC_EPILOGUE_KEY, None)
    else:
        atomic = config.setdefault(ATOMIC_EPILOGUE_KEY, False)
        if type(atomic) is not bool:
            if not fix_invalid:
                raise InvalidConfig(f"{ATOMIC_EPILOGUE_KEY} must be a boolean")
            config[ATOMIC_EPILOGUE_KEY] = False
        elif atomic and not config[WORKSPACE_KEY]:
            # The epilogue belongs to the workspace schedule's producer; the
            # knob is inert on the plain atomic lowering.
            if not fix_invalid:
                raise InvalidConfig(f"{ATOMIC_EPILOGUE_KEY} requires {WORKSPACE_KEY}")
            config[ATOMIC_EPILOGUE_KEY] = False
    partials = config.get(PARTIALS_KEY, PARTIALS_ACCUMULATOR)
    if partials not in PARTIALS_CHOICES or (
        partials == PARTIALS_OUTPUT and not partials_available
    ):
        if not fix_invalid:
            raise InvalidConfig(
                f"{PARTIALS_KEY} must be one of {PARTIALS_CHOICES}; "
                f"{PARTIALS_OUTPUT!r} needs split_k_reduction='output', a 16-bit "
                "output and a plain GEMM store"
            )
        partials = PARTIALS_ACCUMULATOR
    if partials == PARTIALS_OUTPUT and not config[WORKSPACE_KEY]:
        # The partials belong to the workspace schedule; the knob is inert on
        # the plain kernel.
        if not fix_invalid:
            raise InvalidConfig(f"{PARTIALS_KEY} requires {WORKSPACE_KEY}")
        partials = PARTIALS_ACCUMULATOR
    if partials == PARTIALS_ACCUMULATOR:
        # The default: absent, so a config without the knob keeps the fp32
        # partials it always had.
        config.pop(PARTIALS_KEY, None)
    else:
        config[PARTIALS_KEY] = partials
    if not partition_choices:
        # The atomic split-K pattern partitions through its own tunable.
        if config.get(PARTITIONS_KEY, 1) != 1 and not fix_invalid:
            raise InvalidConfig(
                f"{PARTITIONS_KEY} applies to a plain GEMM store reduction only"
            )
        config.pop(PARTITIONS_KEY, None)
        return
    partitions = config.setdefault(PARTITIONS_KEY, 1)
    if type(partitions) is not int or partitions not in partition_choices:
        if not fix_invalid:
            raise InvalidConfig(
                f"{PARTITIONS_KEY} must be one of {partition_choices}, got {partitions!r}"
            )
        config[PARTITIONS_KEY] = 1
    if config.get(WORKSPACE_KEY) and config[PARTITIONS_KEY] == 1:
        # One partition is the plain kernel: keep the workspace schedule for
        # real splits only so the search does not sample a redundant twin.
        if not fix_invalid:
            raise InvalidConfig(f"{WORKSPACE_KEY} requires {PARTITIONS_KEY} > 1")
        config[WORKSPACE_KEY] = False
        # The 16-bit partials belong to the schedule: a workspace seed's
        # mutant to one partition, {workspace: True, partitions: 1, partials:
        # "output"}, would otherwise come out as {workspace: False, partials:
        # "output"}, which strict normalization (``to_code``) rejects with
        # "cute_split_k_partials requires cute_split_k_workspace" instead of
        # compiling the plain kernel.
        config.pop(PARTIALS_KEY, None)


def repair_workspace_geometry(
    config: dict[str, object],
    max_stages: Callable[[dict[str, object]], int | None],
) -> None:
    """Keep a search sample that turns the workspace schedule on to what its
    partition producer runs. ``max_stages`` reads the deepest AB ring the
    producer fits for the sample's tile, ``None`` for a tile it does not take
    (``split_k_workspace.workspace_tile_max_stages``: 128/256 rows,
    64/128/256 columns, 64/128/256-deep K steps dividing the GEMM and its
    partitions).

    The search mutates the schedule's knobs and the tile independently, and
    its tile projections rewrite the tile afterwards (a sampled
    ``tcgen05_cluster_m=2`` takes the CTA pair's 256x256 tile), so a workspace
    sample can name a tile the producer does not take or a ring deeper than
    the tile fits. Such a config failed to compile ("native split-K workspace
    requires full aligned partitions and a feasible paired pipeline"): 64 of
    the 142 configs a full-effort search of a bf16 384 x 768 x 16384 matmul
    sampled, among them [64, 128, 64] and [128, 128, 16] tiles and [128, 128,
    64] with a 16-deep ring (12 fit), and every workspace seed of that matmul
    mutated to ``tcgen05_cluster_m=2`` (no 256-row tile divides 384 rows).
    ``tcgen05_config.repair.fix_search_config`` runs the repair on the tile its
    projections leave, for the plain-store and the atomic proofs alike: the
    sample keeps its ring, cut to the deepest that tile fits, or runs the
    plain kernel of the tile (a valid candidate) where the producer does not
    take it. Strict validation leaves a config as written: code generation
    rejects it."""
    if config.get(WORKSPACE_KEY) is not True:
        return
    maximum = max_stages(config)
    if maximum is None:
        config[WORKSPACE_KEY] = False
        config[STAGES_KEY] = 2
        if PARTITIONS_KEY in config:
            config[PARTITIONS_KEY] = 1
        if config.get(ATOMIC_EPILOGUE_KEY):
            config[ATOMIC_EPILOGUE_KEY] = False
        config.pop(PARTIALS_KEY, None)
        return
    stages = config.get(STAGES_KEY, 2)
    if type(stages) is int and stages > maximum:
        config[STAGES_KEY] = maximum


def workspace_search_fields(config_spec: ConfigSpec) -> dict[str, ConfigSpecFragment]:
    """The search fields of the split-K workspace schedule (two-CTA partition
    producer + reducer launch) of a proved plain GEMM store or atomic
    reduction, none without the proof (``cute_split_k_workspace_available``).
    Every search surface that offers the schedule takes its fields from here,
    so a seed's workspace knobs survive the search's seed transfer (flatten
    and canonicalize drop a knob the surface has no field for). The schedule
    compiles only on its own tile geometries, so random samples keep it off
    (``search_choices=(False,)``) and the search reaches it through the
    workspace heuristic's seeds and their mutations;
    ``cute_split_k_partitions`` appears when the kernel has no partition
    tunable of its own (``plain_store_partition_choices``);
    ``cute_split_k_atomic_epilogue`` (the producer reduces into the output
    itself, no reducer launch) when the proof is an atomic reduction into an
    output kept in a dtype with a packed atomic add; ``cute_split_k_partials``
    (16-bit partials) when a plain store's 16-bit output may take them
    (``split_k_workspace.partials_admissible``)."""
    if not config_spec.cute_split_k_workspace_available:
        return {}
    fields: dict[str, ConfigSpecFragment] = {
        WORKSPACE_KEY: EnumFragment((False, True), search_choices=(False,)),
        STAGES_KEY: IntegerFragment(1, 16, 2),
    }
    if config_spec.cute_split_k_partition_choices:
        fields[PARTITIONS_KEY] = EnumFragment(
            config_spec.cute_split_k_partition_choices
        )
    if config_spec.cute_split_k_atomic_epilogue_available:
        fields[ATOMIC_EPILOGUE_KEY] = EnumFragment((False, True))
    if config_spec.cute_split_k_partials_available:
        fields[PARTIALS_KEY] = EnumFragment(PARTIALS_CHOICES)
    return fields
