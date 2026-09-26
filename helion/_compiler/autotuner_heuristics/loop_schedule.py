"""Small schedule sibling set for independently owned tile-valued loops."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from ...language import memory_ops
from ...runtime.config import Config
from ..cute.loop_state import _index_axis
from ..cute.loop_state import find_loop

KEY = "cute_loop_vectorize"
LOAD_KEY = "cute_loop_load_schedule"
LOAD_SCHEDULES = ("group2", "prefetch2", "group4", "prefetch4")

if TYPE_CHECKING:
    from torch.fx import Node

    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


def loop_schedule_seeds(env: CompileEnvironment, device_ir: DeviceIR) -> list[Config]:
    spec = env.config_spec
    if not spec.cute_loop_schedule_enabled:
        return []
    if device_ir.host_function is None:
        return []
    with device_ir.host_function:
        loop = find_loop(device_ir)
    if loop is None or len(device_ir.grid_block_ids) != 1:
        return []
    stores = [node for node in loop.graph.nodes if node.target is memory_ops.store]
    if not stores:
        return []
    store = stores[0]
    output = cast("Node", store.args[0]).meta["val"]
    root = list(device_ir.grid_block_ids[0])
    with device_ir.host_function:
        axes = tuple(_index_axis(index) for index in store.args[1])
    vector_axes = [
        axis
        for dim, axis in enumerate(axes)
        if axis in root
        and output.stride(dim) == 1
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
        for vectorize in (False, True):
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
                        KEY: vectorize,
                    }
                )
            )
    # Append after the complete original block: never replace or reorder a
    # legacy sibling, promote a new default, or import a measured configuration.
    additions = []
    for parent in result:
        if parent.config[KEY] is not True:
            continue
        for mode in LOAD_SCHEDULES:
            additions.append(Config.from_dict(parent.config | {LOAD_KEY: mode}))
    return [*result, *additions]
