"""Expose native FP32 partial/reduction schedules beside atomic schedules."""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING

from ...runtime.config import Config
from ..cute.pipeline_smem import Tcgen05PipelineSmemFacts
from ..cute.pipeline_smem import max_pipeline_ab_stages
from ..cute.split_k_workspace_config import STAGES_KEY
from ..cute.split_k_workspace_config import WORKSPACE_KEY
from ..cute.tcgen05_config import CuteTcgen05Config
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..compile_environment import ConfigValueExpression
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact


def _config_keys(expression: ConfigValueExpression | int) -> set[str]:
    if isinstance(expression, int):
        return set()
    return {
        key
        for argument in expression.arguments
        for key in ({argument} if isinstance(argument, str) else _config_keys(argument))
    }


class CuteSplitKWorkspaceHeuristic(AutotunerHeuristic):
    name = "cute_split_k_workspace"
    backend = "cute"
    CACHE_SPECIALIZATION_FACTS = frozenset({"config_num_sm"})

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.split_k_workspace import analyze_split_k_workspace

        if device_ir.host_function is None:
            return frozenset()
        with device_ir.host_function:
            env.config_spec.cute_split_k_workspace_available = (
                analyze_split_k_workspace(env, device_ir) is not None
            )
        return frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return env.config_spec.cute_split_k_workspace_available

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
        from ..cute.split_k_workspace import analyze_split_k_workspace
        from ..cute.split_k_workspace import workspace_schedule

        if not cls.is_eligible(env, device_ir):
            return []
        assert device_ir.host_function is not None
        with device_ir.host_function:
            proof = analyze_split_k_workspace(env, device_ir)
        assert proof is not None
        spec = env.config_spec
        capacity = CuteTcgen05Config.per_cta_smem_capacity_bytes(env.device)
        if capacity <= 0:
            return []
        base = spec._base_default_config()
        keys = _config_keys(proof.chunk)
        if len(keys) > 1:
            return []
        variations: list[dict[str, object]] = [{}]
        if keys:
            (key,) = keys
            fragment = spec.user_defined_tunables.get(key)
            values = fragment.search_values(limit=32) if fragment is not None else None
            if values is None:
                return []
            variations = [
                {key: value} for value in values if type(value) is int and value > 0
            ]
        facts = Tcgen05PipelineSmemFacts(2, 4, capacity)
        geometry = sorted(
            (
                (bm, bn, bk)
                for bm in (128, 256)
                for bn in (64, 128, 256)
                for bk in (64, 128)
                if proof.m % bm == proof.n % bn == proof.k % bk == 0
            ),
            key=lambda tile: (
                abs(math.log2(tile[0] / tile[1])),
                tile[0],
                tile[2],
                tile[1],
            ),
        )
        result: list[Config] = []
        for bm, bn, bk in geometry:
            overrides = {
                proof.m_block_id: bm,
                proof.n_block_id: bn,
                proof.k_block_id: bk,
            }
            block_sizes = []
            valid = True
            for item in spec.block_sizes:
                if len(item.block_ids) != 1:
                    valid = False
                    break
                fragment = item._fragment(spec)
                value = overrides.get(item.block_id, fragment.default())
                if not fragment.low <= value <= fragment.high:
                    valid = False
                    break
                block_sizes.append(value)
            if not valid:
                continue
            maximum = max_pipeline_ab_stages(
                facts, bm=bm, bn=bn, bk=bk, c_stages=2, acc_stages=2
            )
            if maximum <= 0:
                continue
            candidates: list[tuple[float, int, Config]] = []
            for variation in variations:
                candidate = Config.from_dict(
                    base.config
                    | variation
                    | {
                        "block_sizes": block_sizes,
                        WORKSPACE_KEY: True,
                        STAGES_KEY: maximum,
                        "cute_collective_mma": False,
                    }
                )
                schedule = workspace_schedule(
                    proof, env, candidate, capacity_bytes=capacity
                )
                if schedule is None:
                    continue
                # Target about one native cluster per SM pair. Retain both
                # nearest partition counts so traffic and occupancy can trade
                # off in the initial population. The parameter's spelling and
                # numerical values come only from its compiler expression.
                clusters = (proof.m // bm) * (proof.n // bn) * schedule.partitions
                occupancy = clusters / max(1, (spec.num_sm or 1) // 2)
                candidates.append(
                    (abs(math.log2(occupancy)), schedule.partitions, candidate)
                )
            chosen: set[int] = set()
            for _distance, partitions, candidate in sorted(
                candidates, key=operator.itemgetter(slice(2))
            ):
                if partitions in chosen:
                    continue
                chosen.add(partitions)
                result.append(candidate)
                if maximum > 2:
                    result.append(Config.from_dict(candidate.config | {STAGES_KEY: 2}))
                if len(chosen) == 2:
                    break
        return result
