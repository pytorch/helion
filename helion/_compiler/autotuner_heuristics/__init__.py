from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .common import dedupe_configs
from .cute import CuteFlashAttentionCausalLptHeuristic
from .cute import CuteFlashAttentionHeuristic
from .cute import CuteFp8GemmSkinnyMHeuristic
from .cute import CuteReductionTileHeuristic
from .cute import CuteReductionWideChunkHeuristic
from .cute import CuteTcgen05ClusterM2FfiHeuristic
from .cute import CuteTcgen05ClusterM2Heuristic
from .cute import CuteTileVecHeuristic
from .cute import CuteTileVecWarpPerRowHeuristic
from .cute import CuteTileVecWarpReduceHeuristic
from .pallas import PallasMatmulF32NoTilingSeedHeuristic
from .pallas import PallasMatmulNoTilingSeedHeuristic
from .triton import TritonB200FormulaMatmulHeuristic
from .triton import TritonB200MatmulHeuristic
from .triton import TritonH100MatmulHeuristic
from .triton import TritonMatmulReductionEpilogueHeuristic
from .triton import TritonPointwiseSeedHeuristic
from .triton import TritonSkinnyGemmHeuristic
from .triton import TritonStandardReductionHeuristic
from .triton import TritonUserTiledReductionHeuristic

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import AutotunerHeuristicType

# All active heuristics by backend
HEURISTICS_BY_BACKEND: dict[str, tuple[AutotunerHeuristicType, ...]] = {
    "cute": (
        CuteFp8GemmSkinnyMHeuristic,
        CuteFlashAttentionHeuristic,
        CuteFlashAttentionCausalLptHeuristic,
        CuteTcgen05ClusterM2FfiHeuristic,
        CuteTcgen05ClusterM2Heuristic,
        CuteReductionTileHeuristic,
        CuteReductionWideChunkHeuristic,
        CuteTileVecHeuristic,
        CuteTileVecWarpReduceHeuristic,
        CuteTileVecWarpPerRowHeuristic,
    ),
    "triton": (
        # H100 dense matmul seed FIRST so its budget-formula config is the rank-0
        # (Product-A) seed for every clean 2-D static GEMM; the skinny rule below still
        # plants its config as a later search seed for aspect>=8 shapes.
        TritonH100MatmulHeuristic,
        TritonSkinnyGemmHeuristic,
        TritonB200MatmulHeuristic,
        # The sm100 formula, promoted; registered after the table so it wins the
        # last-promote-wins compiler_default_config loop.
        TritonB200FormulaMatmulHeuristic,
        TritonMatmulReductionEpilogueHeuristic,
        TritonStandardReductionHeuristic,
        TritonUserTiledReductionHeuristic,
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
        if heuristic.promote_seed_to_default:
            # The primary (rank-0) is the promoted default.
            env.config_spec.compiler_default_config = ranked[0]
        env.config_spec.autotuner_heuristics.append(heuristic.name)
    return dedupe_configs(configs)
