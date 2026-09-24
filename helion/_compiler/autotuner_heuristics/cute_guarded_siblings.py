"""Bounded typed siblings; never promote a new default or mutate old seeds."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from ...runtime.config import Config
from ..cute.chained_matmul import detect_chained_matmul_search
from ..cute.host_fastpath import KEY as FAST_KEY
from ..cute.serial_lane_recurrence import KEY
from ..cute.serial_lane_recurrence import LOAD_KEY
from ..cute.serial_lane_recurrence import TAIL_KEY
from ..cute.serial_lane_recurrence import discover

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


def guarded_schedule_siblings(
    env: CompileEnvironment, device_ir: DeviceIR, parents: list[Config]
) -> list[Config]:
    """Select existing typed vector parents; full emission proofs still apply.

    The parent pool and its relative order remain intact. Separate coupled
    siblings make default-off knobs visible before random population filling.
    No native shape, kernel identity or measured configuration is consulted.
    """
    if not (
        env.config_spec.cute_serial_lane_schedule_enabled
        or env.config_spec.cute_chained_tcgen05_search_enabled
    ):
        return []
    if device_ir.host_function is None:
        return []
    with device_ir.host_function:
        recurrence = discover(device_ir)
    if recurrence is not None:
        vector = [
            parent
            for parent in parents
            if parent.config.get(KEY) == "step_major_vector"
            and parent.config.get(LOAD_KEY) in ("prefetch2", "prefetch4")
            and max(cast("list[int]", parent.config["cute_vector_widths"])) == 4
        ]
        peeled = [
            Config.from_dict(parent.config | {TAIL_KEY: "peel_final_group"})
            for parent in vector
            if recurrence.steps % int(cast("str", parent.config[LOAD_KEY])[-1]) == 0
        ]
        return [
            *peeled,
            *(
                Config.from_dict(p.config | {FAST_KEY: True})
                for p in [*vector, *peeled]
            ),
        ]

    with device_ir.host_function:
        chained = detect_chained_matmul_search(device_ir.graphs)
    if not (env.config_spec.cute_chained_tcgen05_search_enabled and chained):
        return []
    # The existing chained heuristic has already established full static MMA
    # tiles/axes and BF16/FP16 operands. Take two distinct vectorized geometries
    # in its ranked order, without altering that order or enabling random True.
    # Actual tensor ABI, pointer-domain and removable-copy proofs are still
    # mandatory in host_fastpath.variants; an unsupported sibling rejects.
    result = []
    geometries: set[tuple[int, ...]] = set()
    for parent in parents:
        config = parent.config
        if (
            config.get("cute_chained_mma_schedule") != "tcgen05_tmem"
            or config.get("cute_chained_pointwise_vectorize") is not True
            or config.get("cute_chained_pointwise_inplace_async") is True
        ):
            continue
        geometry = tuple(cast("list[int]", config["block_sizes"]))
        if geometry in geometries:
            continue
        geometries.add(geometry)
        # Preserve all explicit parent options, and carry its omitted defaults
        # through flattening. Never repair an unsupported parent's semantics.
        normalized = Config.from_dict(config)
        env.config_spec.normalize(normalized)
        result.append(Config.from_dict(normalized.config | {FAST_KEY: True}))
        if len(result) == 2:
            break
    return result
