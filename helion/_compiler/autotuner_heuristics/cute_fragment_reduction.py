"""Ordinary search coverage for warp-parallel computed-fragment reductions."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...runtime.config import Config
from ..inductor_lowering import ReductionLowering
from .cute_fragment_common import computed_fragment_discovery_supported
from .cute_fragment_common import fragment_coverage_carrier
from .cute_fragment_common import fragment_root_regions
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from torch.fx import Node

    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fragment_reduction"
MODES = ("serial", "warp")


def fragment_warp_reduction_supported(node: Node) -> bool:
    """Match the scalar combines and accumulator types supported by the emitter."""
    lowering = node.meta.get("lowering")
    value = node.meta.get("val")
    return (
        isinstance(lowering, ReductionLowering)
        and lowering.reduction_type in ("sum", "min", "max", "prod")
        and isinstance(value, torch.Tensor)
        and value.dtype
        in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
        )
    )


def fragment_reduction_roots(env: CompileEnvironment, ir: DeviceIR) -> frozenset[int]:
    return frozenset(
        root
        for root, graphs in fragment_root_regions(ir)
        if any(
            fragment_warp_reduction_supported(node)
            for info in graphs
            for node in info.graph.nodes
        )
        and computed_fragment_discovery_supported(env, graphs)
    )


class CuteFragmentReductionHeuristic(AutotunerHeuristic):
    """Register capability even when optional compiler seed use is disabled."""

    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        if not any(
            fragment_warp_reduction_supported(node)
            for info in device_ir.graphs
            for node in info.graph.nodes
        ):
            return frozenset()
        host = device_ir.host_function
        assert host is not None
        with host:
            env.config_spec.cute_fragment_reduction_root_ids = fragment_reduction_roots(
                env, device_ir
            )
        # Metadata guards also protect an initially ineligible dynamic binding
        # whose next shape/stride may admit complete fragment ownership.
        return frozenset({"input_tensor_metadata"})

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        # Additive coverage provides the witness without promoting a default.
        return False


def register_fragment_reduction_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    supplemental_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_reduction_root_ids:
        return
    carrier = fragment_coverage_carrier(spec)
    if carrier is None:
        return
    spec.cute_fragment_reduction_search_enabled = True
    generation = spec.create_config_generation()
    try:
        for mode in MODES:
            requested = Config.from_dict(deepcopy(carrier.config) | {KEY: mode})
            _, effective = generation.strict_config_pair(requested)
            if effective.config.get(KEY, "serial") != mode:
                spec.cute_fragment_reduction_search_enabled = False
                return
    except InvalidConfig:
        spec.cute_fragment_reduction_search_enabled = False
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.computed_fragment_reduction",
            version=1,
            key=KEY,
            domain=MODES,
            legacy="serial",
            witnesses=(CoverageWitness(carrier, "warp"),),
            supplemental_witnesses=(CoverageWitness(supplemental_carrier, "warp"),)
            if supplemental_carrier is not None
            else (),
        )
    )
