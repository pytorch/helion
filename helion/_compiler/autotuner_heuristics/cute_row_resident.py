"""Additive coverage for the proved complete row-resident schedule."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageDependency
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...runtime.config import Config

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


def register_row_resident_coverage(
    env: CompileEnvironment, device_ir: DeviceIR
) -> None:
    from ..cute.row_matrix_transport import prove_row_matrix_transport
    from ..cute.row_resident import ROW_RESIDENT_KEY
    from ..cute.row_resident import prove_row_resident

    host = device_ir.host_function
    if host is None:
        return
    with host:
        plan = prove_row_resident(env, host, binding=True)
    if plan is None:
        return
    spec = env.config_spec
    spec.cute_materialized_schedule_available = True
    with host:
        spec.cute_row_matrix_transport_available = (
            prove_row_matrix_transport(env, host, plan, binding=True) is not None
        )
    modes = spec.cute_materialized_schedule_choices
    previous = spec.create_config_generation()
    try:
        _, carrier = previous.canonicalize_flat(previous.default_flat())
        previous.strict_config_pair(carrier)
    except InvalidConfig:
        return
    # When the independent min-blocks coordinate is present, this consumer
    # runs after that registration and carries its effective value explicitly.
    # No hidden/unrepresented launch-hint key is inserted into an old schema.
    dependencies: tuple[CoverageDependency, ...] = ()
    min_blocks = next(
        (
            group
            for group in spec.compiler_coverage_groups
            if group.key == "cute_min_blocks_per_mp"
        ),
        None,
    )
    if min_blocks is not None:
        requested = Config.from_dict(
            deepcopy(carrier.config) | {"cute_min_blocks_per_mp": 1}
        )
        try:
            _, carrier = previous.strict_config_pair(requested)
        except InvalidConfig:
            return
        if carrier.config.get("cute_min_blocks_per_mp") != 1:
            return
        dependencies = (CoverageDependency(min_blocks.mechanism, min_blocks.key, 1),)
    spec.cute_materialized_schedule_search_enabled = True
    generation = spec.create_config_generation()
    try:
        for mode in modes:
            requested = Config.from_dict(
                deepcopy(carrier.config) | {ROW_RESIDENT_KEY: mode}
            )
            _, effective = generation.strict_config_pair(requested)
            if effective.config.get(ROW_RESIDENT_KEY, "off") != mode:
                spec.cute_materialized_schedule_search_enabled = False
                return
    except InvalidConfig:
        spec.cute_materialized_schedule_search_enabled = False
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.materialized_rows",
            version=1,
            key=ROW_RESIDENT_KEY,
            domain=modes,
            legacy="off",
            witnesses=tuple(CoverageWitness(carrier, mode) for mode in modes),
            dependencies=dependencies,
        )
    )
