"""Deferred coverage for bounded same-coordinate computed register caches."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageDependency
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...runtime.config import Config
from .cute_fragment_common import fragment_root_regions
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fragment_register_producers"


class CuteFragmentRegisterProducersHeuristic(AutotunerHeuristic):
    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.producer_cache import pure_producer
        from ..cute.register_snapshots import snapshot_chains

        roots = set()
        host = device_ir.host_function
        assert host is not None
        with host:
            for root, graphs in fragment_root_regions(device_ir):
                if (
                    root
                    not in env.config_spec.cute_fragment_register_snapshots_root_ids
                ):
                    continue
                chains = snapshot_chains(graphs, env, allow_unbound=True)
                supported = set().union(*chains.values()) if chains else set()
                for graph in graphs:
                    costs = {}
                    for node in graph.graph.nodes:
                        costs[node] = (
                            1 + sum(costs.get(arg, 0) for arg in node.all_input_nodes)
                            if pure_producer(node)
                            else 0
                        )
                        value = node.meta.get("val")
                        if (
                            node in supported
                            and len(node.users) > 1
                            and isinstance(value, torch.Tensor)
                            and value.ndim == 1
                            and costs[node] >= 8
                        ):
                            roots.add(root)
        env.config_spec.cute_fragment_register_producer_root_ids = frozenset(roots)
        return frozenset({"input_tensor_metadata"}) if roots else frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_register_producers_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_register_producer_root_ids:
        return
    snapshot_group = next(
        (
            group
            for group in spec.compiler_coverage_groups
            if group.key == "cute_fragment_register_snapshots"
        ),
        None,
    )
    if snapshot_group is None:
        return
    # Reuse the existing bounded snapshot witness, including its thread geometry.
    witness = snapshot_group.witnesses[0]
    carrier = Config.from_dict(
        deepcopy(witness.carrier.config) | {"cute_fragment_register_snapshots": True}
    )
    previous = spec.cute_fragment_register_producers_search_enabled
    spec.cute_fragment_register_producers_search_enabled = True
    try:
        spec.create_config_generation().strict_config_pair(
            Config.from_dict(deepcopy(carrier.config) | {KEY: True})
        )
    except InvalidConfig:
        spec.cute_fragment_register_producers_search_enabled = previous
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.fragment_register_producers",
            version=1,
            key=KEY,
            domain=(False, True),
            legacy=False,
            witnesses=(CoverageWitness(carrier, True),),
            dependencies=(
                *snapshot_group.dependencies,
                CoverageDependency(snapshot_group.mechanism, snapshot_group.key, True),
            ),
            deferred=True,
        )
    )
