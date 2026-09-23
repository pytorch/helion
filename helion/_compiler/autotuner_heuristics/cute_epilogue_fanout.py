"""Additive coverage for a complete typed two-output native epilogue."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageDependency
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...runtime.config import Config
from ..cute.epilogue_fanout import FANOUT_CONFIG_KEY
from ..cute.epilogue_fanout import FANOUT_MODES
from ..cute.epilogue_fanout import prove_paired_fanout
from ..cute.epilogue_fanout import schedule_supported
from ..cute.tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY
from ..cute.tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENTS

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
    _register_materialized_aux_coverage(env)


def _register_materialized_aux_coverage(env: CompileEnvironment) -> None:
    """Append one complete paired carrier with both auxiliary placements.

    The old seed prefix, default and fanout controls remain intact. Tile and
    pipeline choices depend on each proved region's output role; the strict
    normalizer checks all extents, complete memory budgets and dependencies.
    """
    spec = env.config_spec
    state = spec._cute_tcgen05_config
    axes = state.materialized_matmul_block_ids
    plans = state.epilogue_fanout_plans
    if (
        len(axes) < 2
        or not plans
        or not all(plan.pre_wait_aux_safe for plan in plans)
        or not any(
            chain.auxiliary_tensor_loads for plan in plans for chain in plan.chains
        )
        or not spec.compiler_seed_configs
    ):
        return
    carrier = Config.from_dict(deepcopy(spec.compiler_seed_configs[0].config))
    blocks = carrier.block_sizes.copy()
    ab_stages = []
    c_stages = []
    for triple in axes:
        paired = any(plan.block_ids == triple[:2] for plan in plans)
        tile = (256, 128, 64) if paired else (128, 256, 128)
        for block_id, value in zip(triple, tile, strict=True):
            blocks[spec.block_sizes.block_id_to_index(block_id)] = value
        ab_stages.append(8 if paired else 4)
        c_stages.append(2 if paired else 4)
    carrier.config.update(
        block_sizes=blocks,
        pid_type="persistent_interleaved",
        tcgen05_persistence_model="static_persistent",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=1,
        tcgen05_cta_group="two",
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
        tcgen05_epilogue_fanout="shared",
        tcgen05_region_ab_stages=ab_stages,
        tcgen05_region_c_stages=c_stages,
        tcgen05_materialized_pdl=True,
        tcgen05_c_acquire_placement="before_store",
    )
    try:
        spec.normalize(carrier)
    except InvalidConfig:
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.materialized_fanout_aux_placement",
            version=1,
            key=TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY,
            domain=TCGEN05_AUX_LOAD_PLACEMENTS,
            legacy=TCGEN05_AUX_LOAD_PLACEMENTS[0],
            witnesses=tuple(
                CoverageWitness(carrier, mode) for mode in TCGEN05_AUX_LOAD_PLACEMENTS
            ),
            dependencies=(
                CoverageDependency("cute.epilogue_fanout", FANOUT_CONFIG_KEY, "shared"),
            ),
        )
    )
