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
from ..cute.tcgen05_constants import TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY
from ..cute.tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY
from ..cute.tcgen05_constants import TCGEN05_AUX_LOAD_PLACEMENTS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...autotuner.config_spec import ConfigSpec
    from ..compile_environment import CompileEnvironment
    from ..cute.epilogue_fanout import PairedFanoutPlan
    from ..cute.tcgen05_config import CuteTcgen05Config
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
        # The ordinary seed builders cover single-output GEMMs (the pipeline
        # and full-tile facts need one store) and the machine-filling 256x256
        # CTA pairs, which the fanout does not admit at that width; a
        # two-output kernel with more 128x128 tiles than SMs therefore has no
        # carrier although its fanout is proved. Build one from the geometry.
        carriers = _synthesized_fanout_carriers(spec, state, plans)
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


def _synthesized_fanout_carriers(
    spec: ConfigSpec, state: CuteTcgen05Config, plans: Sequence[PairedFanoutPlan]
) -> list[Config]:
    """One-CTA full-tile carriers for a proved fanout no compiler seed carries.

    Geometry only: the 128x128 one-CTA tile (whole tiles of the fanout's
    output), the narrowest 16-bit-friendly K tile the search admits (64, then
    128), persistent interleaved role-local warps, and the deepest AB ring
    the whole static arena -- both outputs' C rings -- fits next to the deep
    C ring; the shallow-C twin keeps that depth so the pair differs in
    ``tcgen05_c_stages`` alone, as the seed-derived carriers do. Each carrier
    passes the same schedule, fanout-admission and strict-normalization
    checks as a seed-derived one.
    """
    extents = state.matmul_compile_time_static_extents
    fragments = state._matmul_block_fragments()
    if (
        extents is None
        or fragments is None
        or any(value is None or value <= 0 for value in extents)
    ):
        return []
    m, n, k = extents
    assert m is not None and n is not None and k is not None
    bm_fragment, bn_fragment, bk_fragment = fragments
    bm, bn = 128, 128
    if (
        m % bm
        or n % bn
        or not bm_fragment.low <= bm <= bm_fragment.high
        or not bn_fragment.low <= bn <= bn_fragment.high
    ):
        return []
    bk = next(
        (
            value
            for value in (64, 128)
            if k % value == 0 and bk_fragment.low <= value <= bk_fragment.high
        ),
        None,
    )
    if bk is None:
        return []
    blocks = state._matmul_seed_block_sizes(bm=bm, bn=bn, bk=bk)
    if blocks is None:
        return []
    ceiling = min(
        k // bk, state.max_ab_stages_that_fit(bm=bm, bn=bn, bk=bk, cluster_m=1)
    )
    ab_stages = next(
        (
            depth
            for depth in range(ceiling, 0, -1)
            if state.default_layout_smem_fits(
                bm=bm,
                bn=bn,
                bk=bk,
                cluster_m=1,
                ab_stages=depth,
                c_stages=4,
                stage_rows=False,
            )
        ),
        None,
    )
    if ab_stages is None:
        return []
    candidate = Config(
        block_sizes=blocks,
        pid_type="persistent_interleaved",
        tcgen05_cta_group="auto",
        tcgen05_cluster_m=1,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=ab_stages,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=4,
        tcgen05_num_epi_warps=4,
        tcgen05_persistence_model="static_persistent",
        tcgen05_strategy="role_local_monolithic",
        tcgen05_layout_strategy="default",
    )
    try:
        spec.normalize(candidate)
    except InvalidConfig:
        return []
    if not schedule_supported(candidate.config):
        return []
    carriers: list[Config] = []
    for stages in (4, 2):
        trial = Config.from_dict(
            deepcopy(candidate.config)
            | {"tcgen05_c_stages": stages, FANOUT_CONFIG_KEY: "shared"}
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
    return carriers


def _synthesized_pair_fanout_carriers(
    spec: ConfigSpec, state: CuteTcgen05Config
) -> list[Config]:
    """CTA-pair full-tile carriers for a proved fanout, from the geometry.

    The fanout admits the ``tcgen05_cluster_m=2`` pair at one tile shape only
    (the 256x128 16-bit tile, two C stages). Where the pair is on the search
    surface and the output is whole pair tiles, build that carrier for every
    K tile the search admits (128, then 64), each with the deepest AB ring
    the whole static arena -- both outputs' C rings -- fits; the pair halves
    the B operand's L2 feed per CTA, so its ring holds more K steps than the
    one-CTA 128x128 carrier's in the same SMEM. Each carrier passes the same
    schedule, fanout-admission and strict-normalization checks as the
    seed-derived ones. Empty where the pair is not searched or does not
    divide the output.
    """
    extents = state.matmul_compile_time_static_extents
    fragments = state._matmul_block_fragments()
    if (
        state.cluster_m2_search_constraints is None
        or extents is None
        or fragments is None
        or any(value is None or value <= 0 for value in extents)
    ):
        return []
    m, n, k = extents
    assert m is not None and n is not None and k is not None
    bm_fragment, bn_fragment, bk_fragment = fragments
    bm, bn = 256, 128
    if (
        m % bm
        or n % bn
        or not bm_fragment.low <= bm <= bm_fragment.high
        or not bn_fragment.low <= bn <= bn_fragment.high
    ):
        return []
    carriers: list[Config] = []
    for bk in (128, 64):
        if k % bk or not bk_fragment.low <= bk <= bk_fragment.high:
            continue
        blocks = state._matmul_seed_block_sizes(bm=bm, bn=bn, bk=bk)
        if blocks is None:
            continue
        ceiling = min(
            k // bk, state.max_ab_stages_that_fit(bm=bm, bn=bn, bk=bk, cluster_m=2)
        )
        ab_stages = next(
            (
                depth
                for depth in range(ceiling, 0, -1)
                if state.default_layout_smem_fits(
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    cluster_m=2,
                    ab_stages=depth,
                    c_stages=2,
                    stage_rows=False,
                )
            ),
            None,
        )
        if ab_stages is None:
            continue
        candidate = Config(
            block_sizes=blocks,
            pid_type="persistent_interleaved",
            tcgen05_cta_group="auto",
            tcgen05_cluster_m=2,
            tcgen05_cluster_n=1,
            tcgen05_ab_stages=ab_stages,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
            tcgen05_num_epi_warps=4,
            tcgen05_persistence_model="static_persistent",
            tcgen05_strategy="role_local_monolithic",
            tcgen05_layout_strategy="default",
        )
        try:
            spec.normalize(candidate)
        except InvalidConfig:
            continue
        if not schedule_supported(candidate.config):
            continue
        trial = Config.from_dict(
            deepcopy(candidate.config) | {FANOUT_CONFIG_KEY: "shared"}
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
    return carriers


def _carrier_geometry(config: Config, k_index: int | None) -> tuple[object, ...]:
    """The output tile a carrier walks: its M/N blocks and CTA cluster."""
    blocks = tuple(
        value for index, value in enumerate(config.block_sizes) if index != k_index
    )
    return (
        blocks,
        config.config.get("tcgen05_cluster_m", 1),
        config.config.get("tcgen05_cluster_n", 1),
    )


def register_aux_fragment_hoist_coverage(
    env: CompileEnvironment, device_ir: DeviceIR
) -> None:
    """Additive coverage for the pre-wait whole-fragment auxiliary hoist.

    ``tcgen05_aux_fragment_hoist`` acts on kernels with an exact-shape
    auxiliary operand. Its carriers take the ``pre_acc_wait`` placement the
    hoist requires: where every store loads an auxiliary operand, the
    ordinary compiler seeds that the fanout schedule predicate accepts (the
    hoist shares that store protocol); where an aux-free store needs the
    shared fanout to admit the placement, the fanout group's own carriers
    with the fanout on, declared as the dependency. Witnesses pair each
    carrier with the hoist off and on, so the search measures the hoist
    against the same rings.
    """
    spec = env.config_spec
    state = spec._cute_tcgen05_config
    if (
        device_ir.host_function is None
        or not state.search_enabled
        or not state.exact_shape_aux_kernel_detected
    ):
        return
    needs_fanout = not state.aux_loads_on_every_store
    sources: list[Config] = []
    if needs_fanout:
        groups = [
            g for g in spec.compiler_coverage_groups if g.key == FANOUT_CONFIG_KEY
        ]
        if not groups:
            return
        # The CTA-pair carrier first: the fanout group itself keeps the
        # one-CTA carriers (its witness budget holds one geometry), so the
        # pair's full recipe -- fanout, pre-wait placement, hoist -- enters
        # the search through this group where the pair divides the output.
        sources.extend(_synthesized_pair_fanout_carriers(spec, state))
        for witness in groups[0].witnesses:
            carrier = witness.carrier
            if carrier not in sources:
                sources.append(carrier)
    else:
        for seed in spec.compiler_seed_configs:
            candidate = Config.from_dict(deepcopy(seed.config))
            try:
                spec.normalize(candidate)
            except InvalidConfig:
                continue
            if schedule_supported(candidate.config) and candidate not in sources:
                sources.append(candidate)
    admitted: list[Config] = []
    for source in sources:
        values: dict[str, object] = {
            TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY: TCGEN05_AUX_LOAD_PLACEMENTS[1],
            TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY: True,
        }
        if needs_fanout:
            values[FANOUT_CONFIG_KEY] = "shared"
        trial = Config.from_dict(deepcopy(source.config) | values)
        if not state.aux_fragment_hoist_admissible(trial.config):
            continue
        try:
            spec.normalize(trial)
        except InvalidConfig:
            continue
        trial.config.pop(TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY)
        if trial not in admitted:
            admitted.append(trial)
    # Two carriers (the group's witness budget): one per output tile geometry
    # (M/N blocks and cluster; the K tile and ring depth are not a geometry)
    # first, in source order, so a CTA-pair carrier and the one-CTA carrier
    # both witness the hoist; a lone geometry keeps its C-ring twins as before.
    carriers: list[Config] = []
    seen_geometries: set[tuple[object, ...]] = set()
    indices = state._matmul_block_indices()
    k_index = indices[2] if indices is not None else None
    for trial in admitted:
        geometry = _carrier_geometry(trial, k_index)
        if geometry in seen_geometries:
            continue
        seen_geometries.add(geometry)
        carriers.append(trial)
        if len(carriers) == 2:
            break
    for trial in admitted:
        if len(carriers) == 2:
            break
        if trial not in carriers:
            carriers.append(trial)
    if not carriers:
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.aux_fragment_hoist",
            version=1,
            key=TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY,
            domain=(False, True),
            legacy=False,
            witnesses=tuple(
                CoverageWitness(carrier, mode)
                for carrier in carriers
                for mode in (False, True)
            ),
            dependencies=(
                (
                    CoverageDependency(
                        "cute.epilogue_fanout", FANOUT_CONFIG_KEY, "shared"
                    ),
                )
                if needs_fanout
                else ()
            ),
        )
    )
