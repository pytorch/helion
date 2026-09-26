"""Additive coverage for a complete typed two-output native epilogue."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...runtime.config import Config
from ..cute.epilogue_fanout import FANOUT_CONFIG_KEY
from ..cute.epilogue_fanout import FANOUT_MODES
from ..cute.epilogue_fanout import prove_paired_fanout
from ..cute.epilogue_fanout import schedule_supported

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


def register_epilogue_fanout_coverage(
    env: CompileEnvironment, device_ir: DeviceIR
) -> None:
    from ...language import matmul_ops
    from ..cute.cute_mma import _trace_mma_to_stores

    spec = env.config_spec
    state = spec._cute_tcgen05_config
    host = device_ir.host_function
    if host is None or not state.search_enabled:
        return
    targets = (
        torch.ops.aten.addmm.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.baddbmm.default,
        matmul_ops.dot,
    )
    plans = []
    for info in device_ir.graphs:
        for node in info.graph.nodes:
            if node.op != "call_function" or node.target not in targets:
                continue
            stores = _trace_mma_to_stores(node, device_ir.graphs)
            if stores is None:
                continue
            with host:
                plan = prove_paired_fanout(env, host, device_ir.graphs, stores, {node})
            if plan is not None:
                plans.append(plan)
    state.epilogue_fanout_plans = tuple(plans)
    if not plans:
        return

    # Reuse ordinary compiler carriers, ranked by native tile work then their
    # existing block vector (including contraction depth).
    # No old seed is inserted, removed, mutated or interleaved. The two C-ring
    # depths are the existing memory/register tradeoff for this recipe.
    carriers: list[Config] = []
    for seed in spec.compiler_seed_configs:
        candidate = Config.from_dict(deepcopy(seed.config))
        try:
            spec.normalize(candidate)
        except InvalidConfig:
            # Existing seed lists can contain carriers for an earlier axis
            # namespace. Their ordinary search handling is unchanged; they
            # cannot prove a new additive witness for this complete schema.
            continue
        if not schedule_supported(candidate.config):
            continue
        for stages in (4, 2):
            trial = Config.from_dict(
                deepcopy(candidate.config)
                | {
                    "tcgen05_c_stages": stages,
                    FANOUT_CONFIG_KEY: "shared",
                }
            )
            if not state.epilogue_fanout_config_supported(trial.config):
                continue
            try:
                spec.normalize(trial)
            except InvalidConfig:
                continue
            trial.config.pop(FANOUT_CONFIG_KEY)
            if trial not in carriers:
                carriers.append(trial)

    def rank(config: Config) -> tuple[int, tuple[int, ...], int]:
        blocks = config.block_sizes
        axes = plans[0].block_ids
        m = blocks[spec.block_sizes.block_id_to_index(axes[0])]
        n = blocks[spec.block_sizes.block_id_to_index(axes[1])]
        stages = config.config["tcgen05_c_stages"]
        assert type(stages) is int
        return m * n, tuple(blocks), stages

    carriers.sort(key=rank, reverse=True)
    if not carriers:
        return

    # Keep one complete ordinary carrier, not merely one block vector: otherwise
    # distinct AB/ACC schedules with C4 could occupy both slots and displace C2.
    # At most four additions include any missing controls. The generic facility
    # enforces the same budget, vetoes and effectiveness check for every recipe.
    def without_c_stages(carrier: Config) -> dict[str, object]:
        return {
            key: value
            for key, value in carrier.config.items()
            if key != "tcgen05_c_stages"
        }

    chosen = [
        carrier
        for carrier in carriers
        if without_c_stages(carrier) == without_c_stages(carriers[0])
    ]
    assert len(chosen) <= 2
    state.epilogue_fanout_search_enabled = True
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.epilogue_fanout",
            version=1,
            key=FANOUT_CONFIG_KEY,
            domain=FANOUT_MODES,
            legacy="off",
            witnesses=tuple(
                CoverageWitness(carrier, mode)
                for carrier in chosen
                for mode in FANOUT_MODES
            ),
        )
    )
