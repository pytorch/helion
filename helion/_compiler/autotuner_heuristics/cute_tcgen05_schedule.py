"""Register the proved dense tcgen05 schedule families and their seeds.

One native-GEMM analysis feeds full-tile persistent and paired schedules,
the N-edge 2x2 cluster, aligned-N physical tiles and one-CTA FP8 pipelines.
Registration is independent of seed preferences.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact


class CuteTcgen05ScheduleHeuristic(AutotunerHeuristic):
    name = "cute_tcgen05_dense_schedules"
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.dense_gemm import prove_dense_gemm_facts
        from ..cute.one_cta_pipeline import prove_one_cta_pipeline_facts
        from ..cute.tcgen05_schedule_facts import analyze_native_gemm
        from ..cute.tcgen05_schedule_facts import prove_full_tile_schedule_facts

        state = env.config_spec._cute_tcgen05_config
        # The preceding TMA registrar owns alignment/metadata specialization.
        # These proofs read only its result and compile-time shape/graph facts.
        gemm = analyze_native_gemm(env, device_ir)
        facts = prove_full_tile_schedule_facts(gemm)
        if facts is not None:
            state.register_full_tile_schedule_facts(facts)
            state.register_n_edge_cluster_schedule_facts(facts)
        state.dense_gemm_facts = prove_dense_gemm_facts(env, gemm)
        state.widen_paired_pipeline_k_search(env)
        one_cta = prove_one_cta_pipeline_facts(env, gemm)
        if state.one_cta_pipeline_facts is not None:
            assert state.one_cta_pipeline_facts == one_cta
        state.one_cta_pipeline_facts = one_cta
        state.widen_one_cta_pipeline_k_search(env)
        return frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        state = env.config_spec._cute_tcgen05_config
        return (
            state.full_tile_schedule_facts is not None
            or state.n_edge_cluster_schedule_facts is not None
            or state.dense_gemm_facts is not None
            or state.one_cta_pipeline_facts is not None
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
    ) -> list[Config] | None:
        state = env.config_spec._cute_tcgen05_config
        return [
            *state.full_tile_schedule_seed_configs(),
            *state.n_edge_cluster_schedule_seed_configs(),
            *state.dense_aligned_n_seed_configs(),
            *state.one_cta_pipeline_seed_configs(),
        ]
