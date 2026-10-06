"""Additive coverage for owner-private scalar finalizer loops."""

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

KEY = "cute_fragment_private_scalar_loops"


class CuteFragmentPrivateScalarLoopsHeuristic(AutotunerHeuristic):
    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.computed_fragment import computed_fragment_supported
        from ..cute.private_scalar_loops import private_scalar_loop_nodes

        host = device_ir.host_function
        assert host is not None
        with host:
            env.config_spec.cute_fragment_private_scalar_loop_root_ids = frozenset(
                root
                for root, graphs in fragment_root_regions(device_ir)
                if computed_fragment_supported(env, graphs)
                and private_scalar_loop_nodes(graphs)
            )
        return (
            frozenset({"input_tensor_metadata"})
            if env.config_spec.cute_fragment_private_scalar_loop_root_ids
            else frozenset()
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_private_scalar_loops_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_private_scalar_loop_root_ids:
        return
    generation = spec.create_config_generation()
    try:
        carrier = resource_carrier
        if carrier is None:
            _, carrier = generation.canonicalize_flat(generation.default_flat())
        generation.strict_config_pair(carrier)
    except InvalidConfig:
        return
    spec.cute_fragment_private_scalar_loops_search_enabled = True
    generation = spec.create_config_generation()
    effective = Config.from_dict(deepcopy(carrier.config))
    try:
        generation.strict_config_pair(Config.from_dict(effective.config | {KEY: True}))
    except InvalidConfig:
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.fragment_private_scalar_loops",
            version=1,
            key=KEY,
            domain=(False, True),
            legacy=False,
            witnesses=(CoverageWitness(effective, True),),
            deferred=True,
        )
    )
