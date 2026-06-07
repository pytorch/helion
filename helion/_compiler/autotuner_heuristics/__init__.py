from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .common import dedupe_configs
from .cute import CuteReductionTileHeuristic
from .cute import CuteReductionWideChunkHeuristic
from .cute import CuteTcgen05ClusterM2FfiHeuristic
from .cute import CuteTcgen05ClusterM2Heuristic
from .cute import CuteTileVecHeuristic
from .cute import CuteTileVecWarpPerRowHeuristic
from .cute import CuteTileVecWarpReduceHeuristic
from .pallas import PallasMatmulF32NoTilingSeedHeuristic
from .pallas import PallasMatmulNoTilingSeedHeuristic
from .triton import TritonB200MatmulHeuristic
from .triton import TritonReductionTileHeuristic
from .triton import TritonReductionUserTileHeuristic
from .triton import TritonSkinnyGemmHeuristic
from .triton import TritonSplitJoinRotateHeuristic

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import AutotunerHeuristicType

# All active heuristics by backend
HEURISTICS_BY_BACKEND: dict[str, tuple[AutotunerHeuristicType, ...]] = {
    "cute": (
        CuteTcgen05ClusterM2FfiHeuristic,
        CuteTcgen05ClusterM2Heuristic,
        CuteReductionTileHeuristic,
        CuteReductionWideChunkHeuristic,
        CuteTileVecHeuristic,
        CuteTileVecWarpReduceHeuristic,
        CuteTileVecWarpPerRowHeuristic,
    ),
    "triton": (
        TritonSkinnyGemmHeuristic,
        TritonB200MatmulHeuristic,
        TritonSplitJoinRotateHeuristic,
        TritonReductionTileHeuristic,
        TritonReductionUserTileHeuristic,
    ),
    "pallas": (
        PallasMatmulNoTilingSeedHeuristic,
        PallasMatmulF32NoTilingSeedHeuristic,
    ),
}

log: logging.Logger = logging.getLogger(__name__)


def get_heuristics(backend: str) -> tuple[AutotunerHeuristicType, ...]:
    return HEURISTICS_BY_BACKEND.get(backend, ())


def compiler_seed_configs(
    env: CompileEnvironment,
    device_ir: DeviceIR,
) -> list[Config]:
    configs: list[Config] = []
    env.config_spec.autotuner_heuristics = []
    env.config_spec.compiler_default_config = None
    if env.settings.disable_autotuner_heuristics:
        return configs

    for heuristic in get_heuristics(env.backend_name):
        try:
            if not heuristic.is_eligible(env, device_ir):
                continue

            config = heuristic.get_seed_config(env, device_ir)
        except Exception as e:
            log.debug(
                "Autotuner heuristic %s failed while generating compiler seed config: %s",
                heuristic.name,
                e,
                exc_info=True,
            )
            continue
        if config is None:
            continue
        configs.append(config)
        if heuristic.promote_seed_to_default:
            env.config_spec.compiler_default_config = config
        env.config_spec.autotuner_heuristics.append(heuristic.name)
    return dedupe_configs(configs)
