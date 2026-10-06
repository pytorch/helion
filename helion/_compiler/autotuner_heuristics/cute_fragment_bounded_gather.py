"""Opt-in coverage for proved logical-coordinate complete-fragment gathers."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...runtime.config import Config
from .cute_fragment_common import fragment_root_regions
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fragment_bounded_gather"


def bounded_gather_roots(
    env: CompileEnvironment, device_ir: DeviceIR
) -> frozenset[int]:
    """The complete root proof, shared by explicit dependent capabilities."""
    from ..cute.bounded_gather import prove_gather
    from ..cute.computed_fragment import computed_fragment_supported
    from ..cute.topk import match_topk_root

    roots = set()
    host = device_ir.host_function
    assert host is not None
    with host:
        for root, graphs in fragment_root_regions(device_ir):
            if (
                match_topk_root(
                    graphs,
                    noncanonical_block_ids=device_ir.noncanonical_task_origin_block_ids,
                )
                is not None
            ):
                continue
            if not any(
                prove_gather(env, node, graphs=graphs) is not None
                for info in graphs
                for node in info.graph.nodes
            ):
                continue
            if computed_fragment_supported(env, graphs, bounded_gather_owned=True):
                roots.add(root)
    return frozenset(roots)


class CuteFragmentBoundedGatherHeuristic(AutotunerHeuristic):
    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        roots = bounded_gather_roots(env, device_ir)
        env.config_spec.cute_fragment_bounded_gather_root_ids = frozenset(roots)
        return frozenset({"input_tensor_metadata"}) if roots else frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_bounded_gather_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_bounded_gather_root_ids:
        return
    generation = spec.create_config_generation()
    try:
        carrier = resource_carrier
        if carrier is None:
            _, carrier = generation.canonicalize_flat(generation.default_flat())
        generation.strict_config_pair(carrier)
    except InvalidConfig:
        return
    previous = spec.cute_fragment_bounded_gather_search_enabled
    spec.cute_fragment_bounded_gather_search_enabled = True
    try:
        spec.create_config_generation().strict_config_pair(
            Config.from_dict(deepcopy(carrier.config) | {KEY: True})
        )
    except InvalidConfig:
        spec.cute_fragment_bounded_gather_search_enabled = previous
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.fragment_bounded_gather",
            version=1,
            key=KEY,
            domain=(False, True),
            legacy=False,
            witnesses=(CoverageWitness(carrier, True),),
            deferred=True,
        )
    )
