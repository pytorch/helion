"""Additive search coverage for structurally proved computed-fragment scans."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...language import scan_ops
from ...runtime.config import Config
from .cute_fragment_common import fragment_root_regions
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fragment_scan"
MODES = ("serial", "cooperative")


def fragment_scan_roots(env: CompileEnvironment, ir: DeviceIR) -> frozenset[int]:
    """Project the same roots used by explicit ordered phase code generation."""
    from ..cute.computed_fragment import computed_fragment_supported

    return frozenset(
        root
        for root, graphs in fragment_root_regions(ir)
        if any(
            node.target is scan_ops._associative_scan
            for info in graphs
            for node in info.graph.nodes
        )
        and computed_fragment_supported(env, graphs)
    )


class CuteFragmentScanHeuristic(AutotunerHeuristic):
    """Register capability independently of optional compiler seeds."""

    name = "cute_fragment_scan"
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        if not any(
            node.target is scan_ops._associative_scan
            for info in device_ir.graphs
            for node in info.graph.nodes
        ):
            return frozenset()
        host = device_ir.host_function
        assert host is not None
        with host:
            env.config_spec.cute_fragment_scan_root_ids = fragment_scan_roots(
                env, device_ir
            )
        # Shape/stride metadata determines both eligibility and the configured
        # full-axis extents, including an ineligible first dynamic binding.
        return frozenset({"input_tensor_metadata"})

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        # Search coverage below owns the new enum. No seed/default promotion.
        return False


def register_fragment_scan_coverage(
    env: CompileEnvironment, device_ir: DeviceIR
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_scan_root_ids:
        return
    previous = spec.create_config_generation()
    try:
        _, carrier = previous.canonicalize_flat(previous.default_flat())
        previous.strict_config_pair(carrier)
    except InvalidConfig:
        return
    spec.cute_fragment_scan_search_enabled = True
    generation = spec.create_config_generation()
    try:
        for mode in MODES:
            requested = Config.from_dict(deepcopy(carrier.config) | {KEY: mode})
            _, effective = generation.strict_config_pair(requested)
            if effective.config.get(KEY, "serial") != mode:
                spec.cute_fragment_scan_search_enabled = False
                return
    except InvalidConfig:
        spec.cute_fragment_scan_search_enabled = False
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.computed_fragment_scan",
            version=1,
            key=KEY,
            domain=MODES,
            legacy="serial",
            witnesses=(CoverageWitness(carrier, "cooperative"),),
        )
    )
