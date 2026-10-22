"""Bounded paired FP8 pipeline seeds using one accumulator stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


class CuteTcgen05Fp8Acc1Heuristic(AutotunerHeuristic):
    name = "cute_tcgen05_fp8_acc1"
    backend = "cute"
    CACHE_SPECIALIZATION_FACTS = frozenset({"device_num_sm"})

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        state = env.config_spec._cute_tcgen05_config
        facts = state.pipeline_smem_facts
        return (
            facts is not None
            and facts.input_dtype_bytes == 1
            and state.paired_pipeline_search_enabled()
        )

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        seeds = cls.get_seed_configs(env, device_ir)
        return seeds[0] if seeds else None

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config]:
        # The registered SMEM proof prices every allocation, including auxiliary
        # rows. Keep only the deepest legal ring for each geometry/placement.
        # These are search seeds; the existing default is never promoted here.
        return env.config_spec._cute_tcgen05_config._paired_acc1_seed_configs()
