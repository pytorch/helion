from __future__ import annotations

from copy import deepcopy
import logging
from typing import TYPE_CHECKING

from ...autotuner.compiler_coverage import (
    CompilerCoverageGroup as CompilerCoverageGroup,
)
from ...autotuner.compiler_coverage import CoverageDependency as CoverageDependency
from ...autotuner.compiler_coverage import CoverageWitness as CoverageWitness
from ...runtime.config import Config
from ..cute.grouped_row_union import CONFIG_KEY as GROUPED_ROW_UNION_KEY
from ..cute.grouped_row_union import LEGACY_SCHEDULE
from ..cute.grouped_row_union import PAIRED_CLC_SCHEDULE
from ..cute.grouped_row_union import RESIDENT_CTAS_KEY as GROUPED_RESIDENT_CTAS_KEY
from ..cute.grouped_row_union import SCHEDULE_KEY as GROUPED_ROW_UNION_SCHEDULE_KEY
from ..cute.grouped_row_union import STARTUP_PREFILL_KEY
from ..cute.grouped_row_union import TRANSPOSED_SCHEDULE
from .common import dedupe_configs
from .cute import CuteAffineScanHeuristic
from .cute import CuteAsyncPersistentSubwarpRowsHeuristic
from .cute import CuteAsyncStateLoadHeuristic
from .cute import CuteChunkPrepareHeuristic
from .cute import CuteChunkRecurrenceHeuristic
from .cute import CuteCollectiveMatmulHeuristic
from .cute import CuteFixedTokenRank1Heuristic
from .cute import CuteFlashAttentionHeuristic
from .cute import CuteFp8GemmSkinnyMHeuristic
from .cute import CuteNestedRowHeuristic
from .cute import CutePackedSingleTokenRank1Heuristic
from .cute import CutePersistentSubwarpRowsHeuristic
from .cute import CutePointwiseVecHeuristic
from .cute import CuteReductionTileHeuristic
from .cute import CuteReductionWideChunkHeuristic
from .cute import CuteResidentMultiRowHeuristic
from .cute import CuteResidentRowHeuristic
from .cute import CuteResidentRowWideClusterHeuristic
from .cute import CuteRolledClusterLadderHeuristic
from .cute import CuteRolledRowLadderHeuristic
from .cute import CuteSiblingRowHeuristic
from .cute import CuteTcgen05ClusterM2FfiHeuristic
from .cute import CuteTcgen05ClusterM2Heuristic
from .cute import CuteTcgen05GroupedDynamicBk64Heuristic
from .cute import CuteTcgen05GroupedSource64Heuristic
from .cute import CuteTcgen05GroupedStaticCommonKHeuristic
from .cute import CuteTcgen05GroupedWorklistHeuristic
from .cute import CuteTcgen05ThreadLocalEpilogueHeuristic
from .cute import CuteTileVecHeuristic
from .cute import CuteTileVecWarpPerRowHeuristic
from .cute import CuteTileVecWarpReduceHeuristic
from .cute import grouped_full_coverage_configs
from .cute import grouped_row_union_carrier
from .cute import grouped_row_union_cluster4_carrier
from .cute import grouped_row_union_paired_clc_carrier
from .cute_bounded_loop_cache import CuteBoundedLoopCacheHeuristic
from .cute_epilogue_fanout import register_epilogue_fanout_coverage
from .cute_host_paired_sum import CuteHostPairedSumHeuristic
from .cute_host_paired_sum import add_host_sum_seeds
from .cute_launch_bounds import register_matmul_min_blocks_coverage
from .cute_materialized import CuteMaterializedMmaHeuristic
from .cute_materialized_operand import CuteMaterializedOperandHeuristic
from .cute_materialized_pdl import register_materialized_pdl_coverage
from .cute_packed_operand import register_packed_operand_coverage
from .cute_resident_reductions import CuteResidentReductionHeuristic
from .cute_resident_sequence import CuteResidentSequenceHeuristic
from .cute_row_resident import register_row_resident_coverage
from .cute_signed_bitfield import add_signed_bitfield_seeds
from .pallas import PallasMatmulF32NoTilingSeedHeuristic
from .pallas import PallasMatmulNoTilingSeedHeuristic
from .register_chain import CuteRegisterChainHeuristic
from .triton import TritonB200FormulaMatmulHeuristic
from .triton import TritonB200MultiMatmulHeuristic
from .triton import TritonH100FormulaMatmulHeuristic
from .triton import TritonH100MatmulHeuristic as TritonH100MatmulHeuristic
from .triton import TritonH100MultiMatmulHeuristic
from .triton import TritonMatmulReductionEpilogueHeuristic
from .triton import TritonNarrowReductionHeuristic
from .triton import TritonPointwiseSeedHeuristic
from .triton import TritonReductionHeuristic
from .triton import TritonSkinnyGemmHeuristic

if TYPE_CHECKING:
    import torch

    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import AutotunerHeuristicType
    from .registry import CompilerHeuristicSpecializationFact

# All active heuristics by backend
HEURISTICS_BY_BACKEND: dict[str, tuple[AutotunerHeuristicType, ...]] = {
    "cute": (
        CuteAsyncStateLoadHeuristic,
        CuteFp8GemmSkinnyMHeuristic,
        CuteChunkRecurrenceHeuristic,
        CuteChunkPrepareHeuristic,
        CuteAffineScanHeuristic,
        CuteFlashAttentionHeuristic,
        CutePackedSingleTokenRank1Heuristic,
        CuteFixedTokenRank1Heuristic,
        CuteCollectiveMatmulHeuristic,
        CuteMaterializedMmaHeuristic,
        CuteMaterializedOperandHeuristic,
        CuteTcgen05ClusterM2FfiHeuristic,
        CuteTcgen05ClusterM2Heuristic,
        CuteTcgen05GroupedWorklistHeuristic,
        CuteTcgen05GroupedStaticCommonKHeuristic,
        CuteTcgen05GroupedDynamicBk64Heuristic,
        CuteTcgen05ThreadLocalEpilogueHeuristic,
        CuteReductionTileHeuristic,
        CuteReductionWideChunkHeuristic,
        CuteResidentReductionHeuristic,
        CuteResidentSequenceHeuristic,
        CutePersistentSubwarpRowsHeuristic,
        CuteAsyncPersistentSubwarpRowsHeuristic,
        CuteRolledRowLadderHeuristic,
        CuteRolledClusterLadderHeuristic,
        CuteTileVecHeuristic,
        CuteTileVecWarpReduceHeuristic,
        CuteTileVecWarpPerRowHeuristic,
        CuteSiblingRowHeuristic,
        CuteNestedRowHeuristic,
        CuteResidentRowHeuristic,
        CuteResidentRowWideClusterHeuristic,
        CuteResidentMultiRowHeuristic,
        CutePointwiseVecHeuristic,
        CuteRegisterChainHeuristic,
        CuteBoundedLoopCacheHeuristic,
        CuteHostPairedSumHeuristic,
        CuteTcgen05GroupedSource64Heuristic,
    ),
    "triton": (
        # The two sm90 front ends are disjoint and share the B200 decision flow,
        # with WGMMA/register-resident resource policy.
        TritonH100FormulaMatmulHeuristic,
        TritonH100MultiMatmulHeuristic,
        TritonSkinnyGemmHeuristic,
        # The two sm100 front ends are disjoint and both provide fast
        # autotune-off defaults as well as autotuner seeds.
        TritonB200FormulaMatmulHeuristic,
        TritonB200MultiMatmulHeuristic,
        TritonMatmulReductionEpilogueHeuristic,
        TritonReductionHeuristic,
        TritonNarrowReductionHeuristic,
        TritonPointwiseSeedHeuristic,
    ),
    "pallas": (
        PallasMatmulNoTilingSeedHeuristic,
        PallasMatmulF32NoTilingSeedHeuristic,
    ),
}

log: logging.Logger = logging.getLogger(__name__)


def get_heuristics(backend: str) -> tuple[AutotunerHeuristicType, ...]:
    return HEURISTICS_BY_BACKEND.get(backend, ())


def compiler_promotion_specialization_key(
    backend: str,
    device: torch.device,
) -> tuple[tuple[str, str | None], ...]:
    """Return named-target facts that can change seeds or their promotion.

    Compute capability is already part of the bound-kernel specialization key.
    Heuristics may request exact product identity either for a promotion fence
    or because product policy changes their emitted seeds. Non-matching products
    share the ``None`` bucket; capability-only heuristics add no key at all.
    ``get_hardware_info`` is cached per canonical device.
    """
    registry_signature = []
    for heuristic in get_heuristics(backend):
        named_targets = heuristic.CACHE_NAMED_TARGETS
        if named_targets is None and heuristic.promote_seed_to_default:
            named_targets = heuristic.PROMOTE_NAMED_TARGETS
        if named_targets:
            registry_signature.append((heuristic.name, named_targets))
    if not registry_signature:
        return ()

    from ..._argument_device import _canonicalize_argument_device
    from ..._hardware import get_hardware_info

    try:
        hardware = get_hardware_info(_canonicalize_argument_device(device))
        hardware_identity = (
            hardware.device_kind,
            hardware.hardware_name,
            hardware.compute_capability,
        )
    except RuntimeError:
        hardware_identity = None
    return tuple(
        (
            heuristic_name,
            hardware_identity[1]
            if hardware_identity is not None
            and (
                hardware_identity in named_targets
                or (*hardware_identity[:2], None) in named_targets
            )
            else None,
        )
        for heuristic_name, named_targets in registry_signature
    )


def compiler_seed_specialization_facts(
    backend: str,
    fired_heuristics: tuple[str, ...] | list[str],
) -> frozenset[CompilerHeuristicSpecializationFact]:
    """Return device facts needed by compiler seeds that actually fired.

    Eligibility is known only after tracing the kernel.  Deferring this lookup
    until then avoids putting SM count in every bound-kernel key merely because
    one heuristic registered for the backend happens to depend on it.
    """
    fired = frozenset(fired_heuristics)
    return frozenset(
        fact
        for heuristic in get_heuristics(backend)
        if heuristic.name in fired
        for fact in heuristic.CACHE_SPECIALIZATION_FACTS
    )


def compiler_seed_configs(
    env: CompileEnvironment,
    device_ir: DeviceIR,
) -> list[Config]:
    configs: list[Config] = []
    heuristics = get_heuristics(env.backend_name)
    registered_fact_specialization_facts: set[CompilerHeuristicSpecializationFact] = (
        set()
    )
    for heuristic in heuristics:
        registered_fact_specialization_facts.update(
            heuristic.register_facts(env, device_ir)
        )
    env.compiler_fact_specialization_facts = frozenset(
        registered_fact_specialization_facts
    )
    env.config_spec.autotuner_heuristics = []
    env.config_spec.compiler_default_config = None
    env.config_spec.compiler_seed_timeout_retry_repetitions = None
    if env.settings.disable_autotuner_heuristics:
        return configs

    for heuristic in heuristics:
        try:
            if not heuristic.is_eligible(env, device_ir):
                continue

            # A heuristic may plant a RANKED multi-seed list (get_seed_configs);
            # the single get_seed_config is the primary (== the list's [0]). The
            # default base hook returns None, so existing single-seed heuristics
            # keep their exact behavior.
            ranked = heuristic.get_seed_configs(env, device_ir)
            if ranked is None:
                config = heuristic.get_seed_config(env, device_ir)
                ranked = [config] if config is not None else []
        except Exception as e:
            log.debug(
                "Autotuner heuristic %s failed while generating compiler seed config: %s",
                heuristic.name,
                e,
                exc_info=True,
            )
            continue
        ranked = [c for c in ranked if c is not None]
        if not ranked:
            continue
        configs.extend(ranked)
        if heuristic.should_promote(env):
            # The primary (rank-0) is the promoted default.
            env.config_spec.compiler_default_config = ranked[0]
        env.config_spec.autotuner_heuristics.append(heuristic.name)
    if env.backend_name == "cute":
        configs = add_host_sum_seeds(env, dedupe_configs(configs))
        configs = add_signed_bitfield_seeds(env, configs)
    configs = dedupe_configs(configs)
    if env.backend_name == "cute":
        serial_rows = CuteResidentReductionHeuristic.serial_row_seed_configs(
            env, device_ir
        )
        if serial_rows:
            configs.extend(serial_rows)
            if (
                CuteResidentReductionHeuristic.name
                not in env.config_spec.autotuner_heuristics
            ):
                env.config_spec.autotuner_heuristics.append(
                    CuteResidentReductionHeuristic.name
                )
        configs.extend(
            CuteResidentReductionHeuristic.pipeline_depth_seed_configs(env, device_ir)
        )
    if env.backend_name == "cute" and env.config_spec.cute_resident_reduction_blocks:
        # Keep every existing witness in order and expose both summation trees
        # on the same general resident schedules.
        for seed in tuple(configs):
            if seed.config.get("cute_reduction_schedule") in ("resident", "pipelined"):
                values = deepcopy(seed.config)
                values["cute_reduction_local_tree"] = True
                configs.append(Config.from_dict(values))
    if env.backend_name == "cute":
        # Preserve the complete old prefix. A declared coverage witness below
        # admits the static-layout policy without displacing coupled schedules
        # from a bounded initial population.
        for seed in tuple(configs):
            if (
                seed.config.get("cute_collective_mma", False)
                and seed.config.get("cute_collective_compute", "warp") == "warp"
            ):
                values = deepcopy(seed.config)
                values["cute_collective_static_layouts"] = True
                configs.append(Config.from_dict(values))
    if env.backend_name == "cute":
        carrier = grouped_row_union_cluster4_carrier(env, device_ir)
        if carrier is not None:
            configs.append(carrier)
    if env.backend_name == "cute":
        serial_output = CuteResidentReductionHeuristic.serial_output_carrier(
            env, device_ir
        )
        if serial_output is not None:
            for row_schedule, packed in (
                ("serial", False),
                ("serial_deferred", False),
                ("serial_deferred", True),
            ):
                values = deepcopy(serial_output.config) | {
                    "cute_reduction_row_schedule": row_schedule
                }
                if packed:
                    values["cute_reduction_pack_output"] = True
                configs.append(Config.from_dict(values))
    if env.backend_name == "cute":
        paired = grouped_row_union_paired_clc_carrier(env, device_ir)
        if paired is not None:
            configs.append(paired)
            configs.append(
                Config.from_dict(paired.config | {STARTUP_PREFILL_KEY: True})
            )
    return dedupe_configs(configs)


def register_compiler_coverage_groups(
    env: CompileEnvironment, device_ir: DeviceIR
) -> None:
    """Declare independent coverage after every ordinary field is finalized.

    Sampling and explicit-override policy belong to the generic autotuner.
    Keep declarations present when automatic heuristics are disabled.
    """
    if env.backend_name != "cute":
        return
    register_epilogue_fanout_coverage(env, device_ir)
    register_materialized_pdl_coverage(env, device_ir)
    configs = grouped_full_coverage_configs(env, device_ir)
    deep_configs = grouped_full_coverage_configs(env, device_ir, block_k=128)
    if not configs:
        configs, deep_configs = deep_configs, []
    if configs:
        witnesses = [
            CoverageWitness(configs[0], "off"),
            CoverageWitness(configs[0], "fixed_tma_dense"),
        ]
        if deep_configs:
            # Append one independently validated dense pipeline. Its ordinary
            # carrier is checked by strict admission without adding a duplicate
            # control seed or changing any existing initial-population row.
            witnesses.append(CoverageWitness(deep_configs[0], "fixed_tma_dense"))
        witnesses.append(
            CoverageWitness(
                deep_configs[0] if deep_configs else configs[0],
                "fixed_tma_dense_local",
            )
        )
        env.config_spec.register_compiler_coverage_group(
            CompilerCoverageGroup(
                mechanism="cute.grouped_full_coverage",
                version=2,
                key="tcgen05_grouped_full_coverage",
                domain=("off", "fixed_tma_dense", "fixed_tma_dense_local"),
                legacy="off",
                witnesses=tuple(witnesses),
            )
        )
    row_union = grouped_row_union_carrier(env, device_ir)
    row_union_cluster4 = grouped_row_union_cluster4_carrier(env, device_ir)
    row_union_paired = grouped_row_union_paired_clc_carrier(env, device_ir)
    if row_union is not None:
        env.config_spec.register_compiler_coverage_group(
            CompilerCoverageGroup(
                mechanism="cute.grouped_dense_row_union",
                version=1,
                key=GROUPED_ROW_UNION_KEY,
                domain=(False, True),
                legacy=False,
                witnesses=(CoverageWitness(row_union, True),),
            )
        )
        if env.config_spec._cute_tcgen05_config.grouped_row_union_multi_resident_supported:
            carrier = Config.from_dict(
                deepcopy(row_union.config) | {GROUPED_ROW_UNION_KEY: True}
            )
            env.config_spec.register_compiler_coverage_group(
                CompilerCoverageGroup(
                    mechanism="cute.grouped_resident_ctas",
                    version=1,
                    key=GROUPED_RESIDENT_CTAS_KEY,
                    domain=(1, 2),
                    legacy=1,
                    witnesses=(CoverageWitness(carrier, 2),),
                    dependencies=(
                        CoverageDependency(
                            "cute.grouped_dense_row_union", GROUPED_ROW_UNION_KEY, True
                        ),
                    ),
                )
            )
    physical_carriers = [
        (name, carrier)
        for name, carrier in (
            (TRANSPOSED_SCHEDULE, row_union_cluster4),
            (PAIRED_CLC_SCHEDULE, row_union_paired),
        )
        if carrier is not None
    ]
    if physical_carriers:
        witnesses = []
        for name, carrier in physical_carriers:
            schedule_carrier = Config.from_dict(dict(carrier.config))
            schedule_carrier.config.pop(GROUPED_ROW_UNION_SCHEDULE_KEY)
            witnesses.append(CoverageWitness(schedule_carrier, name))
        env.config_spec.register_compiler_coverage_group(
            CompilerCoverageGroup(
                mechanism="cute.grouped_row_union_schedule",
                version=1,
                key=GROUPED_ROW_UNION_SCHEDULE_KEY,
                domain=(LEGACY_SCHEDULE, *(name for name, _ in physical_carriers)),
                legacy=LEGACY_SCHEDULE,
                witnesses=tuple(witnesses),
                dependencies=(
                    CoverageDependency(
                        "cute.grouped_dense_row_union", GROUPED_ROW_UNION_KEY, True
                    ),
                ),
            )
        )
    if row_union_paired is not None:
        env.config_spec.register_compiler_coverage_group(
            CompilerCoverageGroup(
                mechanism="cute.ab_startup_prefill",
                version=1,
                key=STARTUP_PREFILL_KEY,
                domain=(False, True),
                legacy=False,
                witnesses=(CoverageWitness(row_union_paired, True),),
                dependencies=(
                    CoverageDependency(
                        "cute.grouped_dense_row_union", GROUPED_ROW_UNION_KEY, True
                    ),
                    CoverageDependency(
                        "cute.grouped_row_union_schedule",
                        GROUPED_ROW_UNION_SCHEDULE_KEY,
                        PAIRED_CLC_SCHEDULE,
                    ),
                ),
            )
        )
    register_matmul_min_blocks_coverage(env, device_ir)
    serial_output = CuteResidentReductionHeuristic.serial_output_carrier(env, device_ir)
    if serial_output is not None:
        env.config_spec.register_compiler_coverage_group(
            CompilerCoverageGroup(
                mechanism="cute.resident_row_consumption",
                version=1,
                key="cute_reduction_row_schedule",
                domain=("batched", "serial", "serial_deferred"),
                legacy="batched",
                witnesses=tuple(
                    CoverageWitness(serial_output, mode)
                    for mode in ("serial", "serial_deferred")
                ),
            )
        )
        deferred = Config.from_dict(
            deepcopy(serial_output.config)
            | {"cute_reduction_row_schedule": "serial_deferred"}
        )
        env.config_spec.register_compiler_coverage_group(
            CompilerCoverageGroup(
                mechanism="cute.resident_terminal_product",
                version=1,
                key="cute_reduction_pack_output",
                domain=(False, True),
                legacy=False,
                witnesses=(CoverageWitness(deferred, True),),
                dependencies=(
                    CoverageDependency(
                        "cute.resident_row_consumption",
                        "cute_reduction_row_schedule",
                        "serial_deferred",
                    ),
                ),
            )
        )
    register_row_resident_coverage(env, device_ir)
    register_packed_operand_coverage(env, device_ir)
    carriers = env.config_spec.compiler_seed_configs
    if not carriers:
        # Declarations remain available when automatic seed use is disabled.
        # Reuse the same typed layout proof without promoting a default/seed.
        carriers = CuteCollectiveMatmulHeuristic.get_seed_configs(env, device_ir)
    for seed in carriers:
        if (
            seed.config.get("cute_collective_mma", False)
            and seed.config.get("cute_collective_compute", "warp") == "warp"
            and not seed.config.get("cute_collective_static_layouts", False)
        ):
            env.config_spec.register_compiler_coverage_group(
                CompilerCoverageGroup(
                    mechanism="cute.collective_static_layouts",
                    version=1,
                    key="cute_collective_static_layouts",
                    domain=(False, True),
                    legacy=False,
                    witnesses=(CoverageWitness(seed, True),),
                )
            )
            break
