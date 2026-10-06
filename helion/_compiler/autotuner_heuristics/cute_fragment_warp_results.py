"""Additive coverage for warp-owned reduction and returned-atomic values."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageDependency
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...runtime.config import Config
from .cute_fragment_common import fragment_root_regions
from .cute_fragment_threads import THREADS
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fragment_warp_results"


class CuteFragmentWarpResultsHeuristic(AutotunerHeuristic):
    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.computed_fragment import computed_fragment_supported
        from ..cute.warp_results import warp_result_chain

        host = device_ir.host_function
        assert host is not None
        with host:
            env.config_spec.cute_fragment_warp_result_root_ids = frozenset(
                root
                for root, graphs in fragment_root_regions(device_ir)
                if computed_fragment_supported(env, graphs)
                and any(
                    warp_result_chain(node, env, max(THREADS))
                    for info in graphs
                    for node in info.graph.nodes
                )
            )
            groups = [
                count
                for _root, graphs in fragment_root_regions(device_ir)
                for info in graphs
                for node in info.graph.nodes
                for count in warp_result_chain(node, env, max(THREADS)).values()
            ]
            env.config_spec.cute_fragment_warp_result_min_threads = (
                min(groups, default=1) * 32
            )
        return (
            frozenset({"input_tensor_metadata"})
            if env.config_spec.cute_fragment_warp_result_root_ids
            else frozenset()
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_warp_results_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_warp_result_root_ids:
        return
    generation = spec.create_config_generation()
    try:
        carrier = resource_carrier
        if carrier is None:
            _, carrier = generation.canonicalize_flat(generation.default_flat())
        generation.strict_config_pair(carrier)
    except InvalidConfig:
        return
    spec.cute_fragment_warp_results_search_enabled = True
    generation = spec.create_config_generation()
    # A declared coverage coupling, without changing any existing seed/default.
    # The largest existing CTA domain witnesses every bounded group count.
    # It also stays explicit after normalization (128 is the omitted default).
    threads = max(THREADS)
    requested = Config.from_dict(
        deepcopy(carrier.config) | {"cute_fragment_threads": threads}
    )
    try:
        _, effective = generation.strict_config_pair(requested)
        generation.strict_config_pair(Config.from_dict(effective.config | {KEY: True}))
    except InvalidConfig:
        return
    thread_group = next(
        (
            group
            for group in spec.compiler_coverage_groups
            if group.key == "cute_fragment_threads"
        ),
        None,
    )
    if thread_group is None:
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.fragment_warp_results",
            version=1,
            key=KEY,
            domain=(False, True),
            legacy=False,
            witnesses=(CoverageWitness(effective, True),),
            dependencies=(
                CoverageDependency(thread_group.mechanism, thread_group.key, threads),
            ),
            deferred=True,
        )
    )
