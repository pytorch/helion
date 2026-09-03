from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .common import dedupe_configs
from .cute import CuteFlashAttentionHeuristic
from .cute import CuteFp8GemmSkinnyMHeuristic
from .cute import CuteReductionTileHeuristic
from .cute import CuteReductionWideChunkHeuristic
from .cute import CuteResidentMultiRowHeuristic
from .cute import CuteResidentRowHeuristic
from .cute import CuteResidentRowWideClusterHeuristic
from .cute import CuteRolledRowLadderHeuristic
from .cute import CuteTcgen05ClusterM2FfiHeuristic
from .cute import CuteTcgen05ClusterM2Heuristic
from .cute import CuteTcgen05GroupedDynamicBk64Heuristic
from .cute import CuteTcgen05GroupedStaticCommonKHeuristic
from .cute import CuteTcgen05GroupedWorklistHeuristic
from .cute import CuteTcgen05ThreadLocalEpilogueHeuristic
from .cute import CuteTileVecHeuristic
from .cute import CuteTileVecWarpPerRowHeuristic
from .cute import CuteTileVecWarpReduceHeuristic
from .pallas import PallasMatmulF32NoTilingSeedHeuristic
from .pallas import PallasMatmulNoTilingSeedHeuristic
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

    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import AutotunerHeuristicType
    from .registry import CompilerHeuristicSpecializationFact

# All active heuristics by backend
HEURISTICS_BY_BACKEND: dict[str, tuple[AutotunerHeuristicType, ...]] = {
    "cute": (
        CuteFp8GemmSkinnyMHeuristic,
        CuteFlashAttentionHeuristic,
        CuteTcgen05ClusterM2FfiHeuristic,
        CuteTcgen05ClusterM2Heuristic,
        CuteTcgen05GroupedWorklistHeuristic,
        CuteTcgen05GroupedStaticCommonKHeuristic,
        CuteTcgen05GroupedDynamicBk64Heuristic,
        CuteTcgen05ThreadLocalEpilogueHeuristic,
        CuteReductionTileHeuristic,
        CuteReductionWideChunkHeuristic,
        CuteRolledRowLadderHeuristic,
        CuteTileVecHeuristic,
        CuteTileVecWarpReduceHeuristic,
        CuteTileVecWarpPerRowHeuristic,
        CuteResidentRowHeuristic,
        CuteResidentRowWideClusterHeuristic,
        CuteResidentMultiRowHeuristic,
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
    return dedupe_configs(configs)
