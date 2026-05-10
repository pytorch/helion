from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .common import dedupe_configs
from .cute import CuteTcgen05ClusterM2Heuristic
from .triton import TritonSkinnyGemmHeuristic

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import SeedHeuristicType

# All active heuristics by backend
HEURISTICS_BY_BACKEND: dict[str, tuple[SeedHeuristicType, ...]] = {
    "cute": (CuteTcgen05ClusterM2Heuristic,),
    "triton": (TritonSkinnyGemmHeuristic,),
}

log: logging.Logger = logging.getLogger(__name__)


def get_heuristics(backend: str) -> tuple[SeedHeuristicType, ...]:
    return HEURISTICS_BY_BACKEND.get(backend, ())


def compiler_seed_configs(
    env: CompileEnvironment,
    device_ir: DeviceIR,
) -> list[Config]:
    configs: list[Config] = []
    env.config_spec.compiler_seed_heuristics = []
    for heuristic in get_heuristics(env.backend_name):
        try:
            if not heuristic.is_eligible(env, device_ir):
                continue

            config = heuristic.get_config(env, device_ir)
        except Exception as e:
            log.debug(
                "Compiler seed heuristic %s failed: %s",
                heuristic.name,
                e,
                exc_info=True,
            )
            continue
        configs.append(config)
        env.config_spec.compiler_seed_heuristics.append(heuristic.name)
    return dedupe_configs(configs)
