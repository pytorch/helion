"""Additive search for loop-exit publication of private integer updates."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

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

KEY = "cute_fragment_integer_atomic_epochs"


class CuteFragmentIntegerAtomicEpochsHeuristic(AutotunerHeuristic):
    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.computed_fragment import computed_fragment_supported
        from ..cute.integer_atomic_epochs import integer_atomic_loop_graphs

        host = device_ir.host_function
        assert host is not None
        roots = set()
        with host:
            for root, graphs in fragment_root_regions(device_ir):
                eligible = bool(integer_atomic_loop_graphs(graphs))
                # Includes the existing nonescape/no-alias, uniform-loop and
                # initialization -> updates -> final-read lifetime proof.
                if eligible and computed_fragment_supported(env, graphs):
                    roots.add(root)
        env.config_spec.cute_fragment_integer_atomic_epochs_root_ids = frozenset(roots)
        # Emission retains the logical-domain and wrapped-index bounds.
        return frozenset({"input_tensor_metadata"}) if roots else frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_integer_atomic_epochs_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_integer_atomic_epochs_root_ids:
        return
    previous = spec.create_config_generation()
    try:
        carrier = resource_carrier
        if carrier is None:
            _, carrier = previous.canonicalize_flat(previous.default_flat())
        previous.strict_config_pair(carrier)
    except InvalidConfig:
        return
    # The existing lane-private load proof removes shared input staging. Couple
    # only the new deferred witness to that storage choice; old seeds stay as-is.
    load_group = next(
        (
            g
            for g in spec.compiler_coverage_groups
            if g.key == "cute_fragment_register_loads"
        ),
        None,
    )
    if load_group is None:
        return
    spec.cute_fragment_integer_atomic_epochs_search_enabled = True
    generation = spec.create_config_generation()
    try:
        _, effective = generation.strict_config_pair(
            Config.from_dict(
                deepcopy(load_group.witnesses[0].carrier.config)
                | {
                    "cute_fragment_register_loads": True,
                    KEY: True,
                }
            )
        )
    except InvalidConfig:
        spec.cute_fragment_integer_atomic_epochs_search_enabled = False
        return
    witness = Config.from_dict({k: v for k, v in effective.config.items() if k != KEY})
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.fragment_integer_atomic_epochs",
            version=1,
            key=KEY,
            domain=(False, True),
            legacy=False,
            witnesses=(CoverageWitness(witness, True),),
            dependencies=(
                *load_group.dependencies,
                CoverageDependency(load_group.mechanism, load_group.key, True),
            ),
            deferred=True,
        )
    )
