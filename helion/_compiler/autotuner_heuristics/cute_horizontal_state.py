"""Append one structural horizontal carrier without replacing ordinary seeds."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...runtime.config import Config
from ..cute.horizontal_state_config import CONFIG_KEY
from ..cute.horizontal_state_config import HORIZONTAL
from .cute import _seq_config_list
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from torch.fx import Node

    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact


def _carrier(
    env: CompileEnvironment, device_ir: DeviceIR
) -> tuple[int, int, tuple[int, ...]] | None:
    from ..cute.horizontal_state_update import _proven_int
    from ..cute.horizontal_state_update import discover_pointwise_state_region

    capability = env.config_spec.target_device_capability
    if capability is None or capability[0] != 10:
        return None
    region = discover_pointwise_state_region(device_ir.graphs)
    if region is None:
        return None
    state = cast("Node", region.state_store.args[0]).meta["val"]
    shape = tuple(_proven_int(x) for x in state.shape)
    strides = tuple(_proven_int(x) for x in state.stride())
    if any(x is None or x <= 0 for x in shape) or state.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        return None
    sizes = tuple(int(x) for x in shape if x is not None)
    if (
        strides != tuple(math.prod(sizes[i + 1 :]) for i in range(len(sizes)))
        or sizes[-1] % 32
    ):
        return None
    row_value = region.row_selector.meta.get("val")
    if not isinstance(row_value, torch.SymInt):
        return None
    rows = [
        block.block_id
        for block in env.block_sizes
        if block.var._sympy_() == row_value._sympy_()
    ]
    if len(rows) != 1:
        return None
    row_id = rows[0]
    # Leave runtime ownership and complete per-config geometry to the planner.
    # Resource floor is CUDA's 48-KiB per-block default shared capacity; no
    # opt-in carveout or hardware query is needed for these search witnesses.
    candidates = [
        r
        for r in (16, 32, 64, 128)
        if r <= sizes[-2]
        and r * sizes[-1] * 2 + sizes[-1] // 32 * 16 + 4096 <= 48 * 1024
    ]
    if not candidates:
        return None
    # Unlike seed preferences, this necessary-condition domain retains partial
    # row tiles and uses the planner's exact base shared-memory bound.
    legal_rows = tuple(
        r
        for r in (16, 32, 64, 128)
        if r * sizes[-1] * 2 + sizes[-1] // 32 * 16 + 8 <= 48 * 1024
    )
    return row_id, max(candidates), legal_rows


class CuteHorizontalStateHeuristic(AutotunerHeuristic):
    name = "cute_horizontal_state"
    backend = "cute"
    CACHE_SPECIALIZATION_FACTS = frozenset({"input_tensor_metadata"})

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.horizontal_state_update import discover_pointwise_state_region

        if discover_pointwise_state_region(device_ir.graphs) is None:
            return frozenset()
        carrier = _carrier(env, device_ir)
        env.config_spec.cute_horizontal_state_available = carrier is not None
        env.config_spec.cute_horizontal_state_row_tiles = (
            (carrier[0], carrier[2]) if carrier is not None else None
        )
        return cls.CACHE_SPECIALIZATION_FACTS

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        # Registration is early; the new carrier is appended only after all
        # legacy seed postprocessors have finished.
        return False

    @classmethod
    def horizontal_seed(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        carrier = _carrier(env, device_ir)
        if carrier is None:
            return None
        row_id, rows, _ = carrier
        values = dict(env.config_spec.autotune_reference_config().config)
        values["block_sizes"] = _seq_config_list(
            env.config_spec.block_sizes, {row_id: rows}
        )
        values[CONFIG_KEY] = HORIZONTAL
        values["pid_type"] = "flat"
        return Config.from_dict(values)
