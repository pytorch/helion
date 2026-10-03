"""Additive ordinary coverage for the proved packed-operand warp schedule."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...runtime.config import Config

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


def register_packed_operand_coverage(
    env: CompileEnvironment, device_ir: DeviceIR
) -> None:
    from ..cute.packed_operand import KEY
    from ..cute.packed_operand import MODES
    from ..cute.packed_operand import prove_packed_operand

    host = device_ir.host_function
    if host is None:
        return
    with host:
        plan = prove_packed_operand(env, host, binding=True)
    if plan is None:
        return
    spec = env.config_spec
    spec.cute_materialized_operand_schedule_available = True
    previous = spec.create_config_generation()
    try:
        _, carrier = previous.canonicalize_flat(previous.default_flat())
        previous.strict_config_pair(carrier)
    except InvalidConfig:
        return
    spec.cute_materialized_operand_schedule_search_enabled = True
    generation = spec.create_config_generation()
    try:
        for mode in MODES:
            requested = Config.from_dict(deepcopy(carrier.config) | {KEY: mode})
            _, effective = generation.strict_config_pair(requested)
            if effective.config.get(KEY, "off") != mode:
                spec.cute_materialized_operand_schedule_search_enabled = False
                return
    except InvalidConfig:
        spec.cute_materialized_operand_schedule_search_enabled = False
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.materialized_byte_operand",
            version=1,
            key=KEY,
            domain=MODES,
            legacy="off",
            witnesses=tuple(CoverageWitness(carrier, mode) for mode in MODES),
        )
    )
