"""Non-promoted shared-layout seeds for possible positional materializations."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from typing import TYPE_CHECKING

from ...autotuner.config_spec import CUTE_POSITIONAL_MMA_KEY
from ...autotuner.config_spec import CUTE_POSITIONAL_SHARED_LAYOUT_KEY
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact


class CutePositionalSharedLayoutHeuristic(AutotunerHeuristic):
    name = "cute_positional_shared_layout"
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.positional_root import has_positional_contraction_candidate

        spec = env.config_spec
        host = device_ir.host_function
        if (
            host is not None
            and not spec.cute_chained_matmul_search_enabled
            and not spec.cute_tcgen05_search_enabled
            and not spec.cute_flash_search_enabled
            and not spec.cute_flash_bwd_search_enabled
            and spec.cute_chunk_prepare_schedule is None
            and "flat" in spec.allowed_pid_types
            and not env.settings.autotune_force_persistent
        ):
            env_context = nullcontext() if env.has_current() else env
            with env_context, host:
                spec.cute_positional_shared_layout_search_enabled = (
                    has_positional_contraction_candidate(device_ir.graphs)
                )
        return frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        spec = env.config_spec
        # Other whole-root families can register after positional facts.
        # Resolve exclusivity again when the registry requests their seeds.
        return (
            spec.cute_positional_shared_layout_search_enabled
            and not spec.cute_chained_matmul_search_enabled
            and not spec.cute_tcgen05_search_enabled
            and not spec.cute_flash_search_enabled
            and not spec.cute_flash_bwd_search_enabled
            and spec.cute_chunk_prepare_schedule is None
        )

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        if not cls.is_eligible(env, device_ir):
            return None
        config = env.config_spec.default_config()
        config.config.update(
            pid_type="flat", **{CUTE_POSITIONAL_SHARED_LAYOUT_KEY: "row"}
        )
        return config

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config] | None:
        row = cls.get_seed_config(env, device_ir)
        if row is None:
            return None
        xor = deepcopy(row)
        xor.config[CUTE_POSITIONAL_SHARED_LAYOUT_KEY] = "xor"
        seeds = [row, xor]
        if env.config_spec.cute_positional_mma_search_enabled:
            for config in (row, xor):
                coupled = deepcopy(config)
                coupled.config[CUTE_POSITIONAL_MMA_KEY] = True
                seeds.append(coupled)
        return seeds
