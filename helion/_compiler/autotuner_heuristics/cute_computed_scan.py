"""Append a staged-input choice for complete rank-two computed prefixes."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

from ...autotuner.config_spec import CUTE_COMPUTED_SCAN_SCHEDULE_KEY
from ...exc import InvalidConfig
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact


class CuteComputedScanHeuristic(AutotunerHeuristic):
    name = "cute_computed_scan"
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.computed_scan_root import has_computed_scan_staging_candidate

        host = device_ir.host_function
        spec = env.config_spec
        if (
            host is not None
            and "flat" in spec.allowed_pid_types
            and not env.settings.autotune_force_persistent
        ):
            context = nullcontext() if env.has_current() else env
            with context, host:
                spec.cute_computed_scan_search_enabled = (
                    has_computed_scan_staging_candidate(device_ir.graphs)
                )
        return frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        spec = env.config_spec
        return (
            spec.cute_computed_scan_search_enabled
            and not spec.cute_chained_matmul_search_enabled
            and not spec.cute_tcgen05_search_enabled
            and not spec.cute_flash_search_enabled
            and not spec.cute_flash_bwd_search_enabled
            and spec.cute_chunk_prepare_schedule is None
        )

    @classmethod
    def seed_config(cls, env: CompileEnvironment, device_ir: DeviceIR) -> Config | None:
        if not cls.is_eligible(env, device_ir):
            return None
        config = env.config_spec.default_config()
        config.config.update(
            pid_type="flat", **{CUTE_COMPUTED_SCAN_SCHEDULE_KEY: "staged_warp"}
        )
        return config


def append_computed_scan_seed(
    env: CompileEnvironment, device_ir: DeviceIR, seeds: list[Config]
) -> list[Config]:
    candidate = CuteComputedScanHeuristic.seed_config(env, device_ir)
    if candidate is None or any(
        candidate.config.get(key) != value
        for key, value in env.settings.autotune_config_overrides.items()
    ):
        return seeds
    try:
        candidate = env.config_spec.normalized_config(candidate.config)
    except InvalidConfig:
        return seeds
    if candidate in seeds:
        return seeds
    env.config_spec.autotuner_heuristics.append(CuteComputedScanHeuristic.name)
    return [*seeds, candidate]
