from __future__ import annotations

from ...autotuner.config_fragment import ConfigSpecFragment
from ...autotuner.config_fragment import EnumFragment
from ...autotuner.config_fragment import IntegerFragment
from ...exc import InvalidConfig

WORKSPACE_KEY = "cute_split_k_workspace"
STAGES_KEY = "cute_split_k_stages"
# A plain GEMM store (``out[tile_m, tile_n] = acc``) has no partition axis of
# its own: this knob is the partition count of its workspace schedule.
PARTITIONS_KEY = "cute_split_k_partitions"
PARTITION_CHOICES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
# The producer reduces its partition straight into the (zero-filled) output
# with packed atomic adds instead of storing fp32 partials for a reducer
# launch: one kernel per call, the ``hl.atomic_add`` numerics of the source
# (every partition rounded into the output's dtype) where the reducer form
# rounds the fp32 sum once.
ATOMIC_EPILOGUE_KEY = "cute_split_k_atomic_epilogue"
WORKSPACE_CONFIG_KEYS = frozenset(
    (WORKSPACE_KEY, STAGES_KEY, PARTITIONS_KEY, ATOMIC_EPILOGUE_KEY)
)


def normalize_workspace_config(
    config: dict[str, object],
    *,
    available: bool,
    fix_invalid: bool,
    partition_choices: tuple[int, ...] = (),
    atomic_epilogue_available: bool = False,
) -> None:
    if not available:
        if (
            config.get(WORKSPACE_KEY, False) is not False
            or config.get(STAGES_KEY, 2) != 2
            or config.get(PARTITIONS_KEY, 1) != 1
            or config.get(ATOMIC_EPILOGUE_KEY, False) is not False
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


def workspace_search_fields(
    partition_choices: tuple[int, ...],
    *,
    atomic_epilogue_available: bool = False,
) -> dict[str, ConfigSpecFragment]:
    """The search fields of the split-K workspace schedule (two-CTA partition
    producer + reducer launch) of a proved plain GEMM store or atomic
    reduction. The schedule compiles only on its own tile geometries, so
    random samples keep it off (``search_choices=(False,)``) and the search
    reaches it through the workspace heuristic's seeds and their mutations;
    ``cute_split_k_partitions`` appears when the kernel has no partition
    tunable of its own (``plain_store_partition_choices``);
    ``cute_split_k_atomic_epilogue`` (the producer reduces into the output
    itself, no reducer launch) when the proof is an atomic reduction into an
    output kept in a dtype with a packed atomic add."""
    fields: dict[str, ConfigSpecFragment] = {
        WORKSPACE_KEY: EnumFragment((False, True), search_choices=(False,)),
        STAGES_KEY: IntegerFragment(1, 16, 2),
    }
    if partition_choices:
        fields[PARTITIONS_KEY] = EnumFragment(partition_choices)
    if atomic_epilogue_available:
        fields[ATOMIC_EPILOGUE_KEY] = EnumFragment((False, True))
    return fields
