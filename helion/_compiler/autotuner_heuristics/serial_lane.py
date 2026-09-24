"""Small generic sibling set for an independently owned typed recurrence."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from ...runtime.config import Config
from ..cute.serial_lane_coarsen import KEY as COARSEN_KEY
from ..cute.serial_lane_coarsen import paired_axis
from ..cute.serial_lane_recurrence import KEY
from ..cute.serial_lane_recurrence import LOAD_KEY
from ..cute.serial_lane_recurrence import LOAD_SCHEDULES
from ..cute.serial_lane_recurrence import discover
from ..cute.serial_lane_recurrence import load_schedule_bytes

if TYPE_CHECKING:
    from torch.fx import Node

    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


def serial_lane_seeds(env: CompileEnvironment, device_ir: DeviceIR) -> list[Config]:
    spec = env.config_spec
    if not spec.cute_serial_lane_schedule_enabled:
        return []
    if device_ir.host_function is None:
        return []
    with device_ir.host_function:
        recurrence = discover(device_ir)
    if recurrence is None or len(device_ir.grid_block_ids) != 1:
        return []
    output = cast("Node", recurrence.store.args[0]).meta["val"]
    state = cast("Node", recurrence.state.args[0]).meta["val"]
    root = list(device_ir.grid_block_ids[0])
    axes = recurrence.axes
    vector_axes = [
        axis
        for dim, axis in enumerate(axes)
        if axis in root
        and output.stride(dim) == state.stride(dim) == 1
        and axis in spec.cute_vector_widths.valid_block_ids()
    ]
    if len(vector_axes) != 1:
        return []
    vector = vector_axes[0]
    sizes = {axis: env.block_sizes[axis].size_hint() for axis in root}
    blocks = {}
    for item in spec.block_sizes:
        if len(item.block_ids) != 1 or item.block_id not in root:
            return []
        axis = item.block_id
        candidates = (64, 32, 16) if axis == vector else (4, 2, 1, 8, 16, 32)
        fragment = item._fragment(spec)
        legal = [
            value
            for value in candidates
            if fragment.low <= value <= fragment.high and sizes[axis] % value == 0
        ]
        if not legal:
            return []
        blocks[axis] = legal[0]
    if vector not in blocks:
        return []
    # Singleton/fixed axes keep their original slots. Remaining thread axes
    # put contiguous elements on x without changing logical ownership.
    ordered = sorted(
        root,
        key=lambda axis: (
            (0, root.index(axis))
            if axis not in blocks
            else (1, output.stride(axes.index(axis)))
        ),
    )
    order = [root.index(axis) for axis in ordered]
    result = []
    for width in (4, 2):
        threads = {
            axis: size // width if axis == vector else size
            for axis, size in blocks.items()
        }
        total = 1
        for value in threads.values():
            total *= value
        if total > 1024 or blocks[vector] % width:
            continue
        for mode in ("step_major", "step_major_vector"):
            result.append(
                Config.from_dict(
                    {
                        "block_sizes": [
                            blocks[item.block_id] for item in spec.block_sizes
                        ],
                        "num_threads": [
                            threads.get(item.block_id, 1) for item in spec.num_threads
                        ],
                        "cute_vector_widths": [
                            width if item.block_id == vector else 1
                            for item in spec.cute_vector_widths
                        ],
                        "loop_orders": [order],
                        "cute_cluster_n": 1,
                        "pid_type": "flat",
                        KEY: mode,
                    }
                )
            )
    # Append after the complete original block: never replace or reorder a
    # legacy sibling, promote a new default, or import a measured configuration.
    additions = []
    for parent in result:
        if parent.config[KEY] != "step_major_vector":
            continue
        width = max(cast("list[int]", parent.config["cute_vector_widths"]))
        for mode in LOAD_SCHEDULES:
            cap = 128 if mode.startswith("prefetch") else 64
            if load_schedule_bytes(recurrence, width, mode) <= cap:
                additions.append(Config.from_dict(parent.config | {LOAD_KEY: mode}))
    legacy = [*result, *additions]
    coarsened = []
    for parent in additions:
        if parent.config[LOAD_KEY] not in ("prefetch2", "prefetch4"):
            continue
        if max(cast("list[int]", parent.config["cute_vector_widths"])) != 4:
            continue
        paired_blocks = {
            axis: env.block_sizes[axis].from_config_assert(parent) for axis in root
        }
        if not all(
            type(value) is int and value > 0 for value in paired_blocks.values()
        ):
            continue
        with device_ir.host_function:
            axis = paired_axis(
                recurrence, vector, root, cast("dict[int, int]", paired_blocks)
            )
        if axis is not None:
            coarsened.append(Config.from_dict(parent.config | {COARSEN_KEY: 2}))
    return [*legacy, *coarsened]
