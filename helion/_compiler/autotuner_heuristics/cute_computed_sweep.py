"""Appended sweep candidates for shared computed reduction inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...exc import InvalidConfig
from ...language.memory_ops import store
from .cute import CutePersistentSubwarpRowsHeuristic
from .cute import _seq_config_list

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


def _ancestors(node: torch.fx.Node) -> set[torch.fx.Node]:
    pending = [node]
    seen: set[torch.fx.Node] = set()
    while pending and len(seen) <= 128:
        current = pending.pop()
        if current not in seen:
            seen.add(current)
            pending.extend(current.all_input_nodes)
    return seen if not pending else set()


def has_computed_sweep_candidate(device_ir: DeviceIR) -> bool:
    """Discovery only: the late typed/alias/lane proof remains authoritative."""
    for info in device_ir.graphs:
        sums = [
            node.args[0]
            for node in info.graph.nodes
            if node.target is torch.ops.aten.sum.dim_IntList
            and node.args
            and isinstance(node.args[0], torch.fx.Node)
        ]
        outputs = [
            node.args[2]
            for node in info.graph.nodes
            if node.target is store
            and len(node.args) >= 3
            and isinstance(node.args[2], torch.fx.Node)
        ]
        for reduction in sums:
            for output in outputs:
                for node in _ancestors(reduction) & _ancestors(output):
                    value = node.meta.get("val")
                    if (
                        node.target
                        in (torch.ops.aten.add.Tensor, torch.ops.aten.mul.Tensor)
                        and isinstance(value, torch.Tensor)
                        and value.dtype == torch.float32
                        and value.ndim > 0
                    ):
                        return True
    return False


def append_computed_sweep_seed(
    env: CompileEnvironment, device_ir: DeviceIR, seeds: list[Config]
) -> list[Config]:
    plan = CutePersistentSubwarpRowsHeuristic._plan(env, device_ir)
    if plan is None or not has_computed_sweep_candidate(device_ir):
        return seeds
    row, reduction, _ = plan
    spec = env.config_spec
    parent = CutePersistentSubwarpRowsHeuristic._seed_config(
        env, row, reduction, 32, row_block_size=16, row_threads=1
    )
    proposed = parent.config | {
        "cute_vector_widths": _seq_config_list(
            spec.cute_vector_widths,
            dict.fromkeys(spec.cute_vector_widths.valid_block_ids(), 1),
        ),
        "cute_reduction_reloads": _seq_config_list(
            spec.cute_reduction_reloads, {reduction: "computed"}
        ),
    }
    if any(
        proposed.get(name) != value
        for name, value in env.settings.autotune_config_overrides.items()
    ):
        return seeds
    try:
        seed = spec.normalized_config(proposed)
    except InvalidConfig:
        return seeds
    if seed in seeds:
        return seeds
    spec.autotuner_heuristics.append("cute_computed_sweep_cache")
    return [*seeds, seed]


def append_vector_computed_sweep_seed(
    env: CompileEnvironment, device_ir: DeviceIR, seeds: list[Config]
) -> list[Config]:
    """Try one complete vector per reduction thread with computed reuse.

    Reuse the existing subwarp discovery, including its input-metadata
    specialization and dtype-sized vector bound. Sixteen reduction threads
    and sixteen independent row threads cover the complete static feature
    extent without a second vector. Late vector/address/cache proofs retain
    authority over individual loads, stores and computed values.
    """
    plan = CutePersistentSubwarpRowsHeuristic._plan(env, device_ir)
    if plan is None or not has_computed_sweep_candidate(device_ir):
        return seeds
    row, reduction, reduction_threads = plan
    # The parent derives its thread count from the full extent and a vector
    # of at most 16 bytes for the discovered input dtype. Keep that coverage.
    if reduction_threads != 16:
        return seeds
    extent = int(env.block_sizes[reduction].numel)
    width, remainder = divmod(extent, reduction_threads)
    if remainder or width not in (2, 4, 8):
        return seeds
    spec = env.config_spec
    parent = CutePersistentSubwarpRowsHeuristic._seed_config(
        env, row, reduction, reduction_threads, row_block_size=16, row_threads=16
    )
    proposed = parent.config | {
        "cute_vector_widths": _seq_config_list(
            spec.cute_vector_widths,
            dict.fromkeys(spec.cute_vector_widths.valid_block_ids(), 1)
            | {reduction: width},
        ),
        "cute_reduction_reloads": _seq_config_list(
            spec.cute_reduction_reloads, {reduction: "computed"}
        ),
    }
    if any(
        proposed.get(name) != value
        for name, value in env.settings.autotune_config_overrides.items()
    ):
        return seeds
    try:
        seed = spec.normalized_config(proposed)
    except InvalidConfig:
        return seeds
    if seed in seeds:
        return seeds
    spec.autotuner_heuristics.append("cute_vector_computed_sweep_cache")
    return [*seeds, seed]
