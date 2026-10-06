"""Optional aligned packet loads into the existing dense shared layout."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...language import _tracing_ops
from ...language import memory_ops
from ...runtime.config import Config
from .cute_fragment_common import fragment_root_regions
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fragment_packet_loads"


class CuteFragmentPacketLoadsHeuristic(AutotunerHeuristic):
    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.computed_fragment import computed_fragment_supported
        from ..cute.register_loads import host_load_is_readonly

        host = device_ir.host_function
        assert host is not None
        roots = set()
        with host:
            for root, graphs in fragment_root_regions(device_ir):
                if not computed_fragment_supported(env, graphs):
                    continue
                for info in graphs:
                    for node in info.graph.nodes:
                        if node.target is not memory_ops.load:
                            continue
                        source = node.args[0]
                        value = node.meta.get("val")
                        if (
                            isinstance(source, Node)
                            and source.target is _tracing_ops._host_tensor
                            and isinstance(value, torch.Tensor)
                            and value.ndim
                            and value.dtype == torch.float32
                            and host_load_is_readonly(
                                node, env, graphs, allow_unbound=True
                            )
                        ):
                            roots.add(root)
        env.config_spec.cute_fragment_packet_load_root_ids = frozenset(roots)
        # The emitter additionally requires cache-specialized pointer
        # alignment and proves each packet's effective addresses and masks.
        return frozenset({"input_tensor_metadata"}) if roots else frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_packet_loads_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_packet_load_root_ids:
        return
    previous = spec.create_config_generation()
    try:
        carrier = resource_carrier
        if carrier is None:
            _, carrier = previous.canonicalize_flat(previous.default_flat())
        previous.strict_config_pair(carrier)
    except InvalidConfig:
        return
    previous_enabled = spec.cute_fragment_packet_loads_search_enabled
    spec.cute_fragment_packet_loads_search_enabled = True
    generation = spec.create_config_generation()
    try:
        generation.strict_config_pair(
            Config.from_dict(deepcopy(carrier.config) | {KEY: True})
        )
    except InvalidConfig:
        spec.cute_fragment_packet_loads_search_enabled = previous_enabled
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.fragment_packet_loads",
            version=1,
            key=KEY,
            domain=(False, True),
            legacy=False,
            witnesses=(CoverageWitness(carrier, True),),
            deferred=True,
        )
    )
