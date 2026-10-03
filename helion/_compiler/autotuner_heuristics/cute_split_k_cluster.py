"""Append a typed cluster-K carrier without displacing legacy seeds."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageDependency
from ...autotuner.compiler_coverage import CoverageWitness
from ...runtime.config import Config
from ..cute.split_k_cluster import CLUSTER_K_SCHEDULE
from ..cute.split_k_cluster import cluster_facts
from ..cute.split_k_cluster_config import CLUSTER8_K4
from ..cute.split_k_cluster_config import FINALIZER_KEY
from ..cute.split_k_cluster_config import FINALIZER_WARPS
from ..cute.split_k_cluster_config import LEGACY
from ..cute.split_k_cluster_config import SCHEDULE_KEY
from ..cute.split_k_cluster_config import SCHEDULES
from .cute_split_k_workspace import _config_keys
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact


class CuteSplitKClusterHeuristic(AutotunerHeuristic):
    name = "cute_split_k_cluster"
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        if device_ir.host_function is not None:
            with device_ir.host_function:
                env.config_spec.cute_split_k_cluster_facts = cluster_facts(
                    env, device_ir
                )
        return frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        # The carrier is appended after all existing seeds, never promoted.
        return False


def cluster_carrier(env: CompileEnvironment, device_ir: DeviceIR) -> Config | None:
    spec = env.config_spec
    facts = spec.cute_split_k_cluster_facts
    if facts is None:
        return None
    schedule = CLUSTER_K_SCHEDULE
    keys = _config_keys(facts.chunk)
    if len(keys) > 1:
        return None
    variations: list[dict[str, object]] = [{}]
    if keys:
        (key,) = keys
        fragment = spec.user_defined_tunables.get(key)
        values = fragment.search_values(limit=32) if fragment is not None else None
        if values is None:
            return None
        variations = [{key: value} for value in values if type(value) is int]
    blocks = dict(zip(facts.axes, schedule.carrier_tile, strict=True))
    threads = dict(zip(facts.axes, (4, 8, 4), strict=True))
    base = spec._base_default_config()
    for variation in variations:
        candidate = Config.from_dict(
            base.config
            | variation
            | {
                SCHEDULE_KEY: CLUSTER8_K4,
                "cute_split_k_workspace": False,
                "cute_collective_mma": False,
                "block_sizes": [
                    blocks.get(item.block_id, item._fragment(spec).default())
                    for item in spec.block_sizes
                ],
                "num_threads": [
                    threads.get(item.block_id, 1) for item in spec.num_threads
                ],
            }
        )
        if facts.valid_config(spec, candidate):
            return candidate
    return None


def register_cluster_coverage(env: CompileEnvironment, device_ir: DeviceIR) -> None:
    carrier = cluster_carrier(env, device_ir)
    if carrier is None:
        return
    values = dict(carrier.config)
    values.pop(SCHEDULE_KEY)
    env.config_spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.split_k_cluster",
            version=1,
            key=SCHEDULE_KEY,
            domain=SCHEDULES,
            legacy=LEGACY,
            witnesses=(CoverageWitness(Config.from_dict(values), CLUSTER8_K4),),
        )
    )
    env.config_spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.split_k_finalizer",
            version=1,
            key=FINALIZER_KEY,
            domain=FINALIZER_WARPS,
            legacy=1,
            witnesses=(CoverageWitness(carrier, 4),),
            dependencies=(
                CoverageDependency("cute.split_k_cluster", SCHEDULE_KEY, CLUSTER8_K4),
            ),
        )
    )
