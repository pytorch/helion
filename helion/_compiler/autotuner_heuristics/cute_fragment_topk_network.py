"""Optional bounded network selection for complete shared fragment inputs."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

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

KEY = "cute_fragment_topk_network"


def network_topk_supported(dtype: torch.dtype, n: int | None, k: int) -> bool:
    # At most 32 Int64 register keys per lane. Larger/different-typed selections
    # keep the serial implementation, including Int64 extrema without packing.
    return (
        dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int32)
        and n is not None
        and 0 < k <= n <= 1024
    )


class CuteFragmentTopKNetworkHeuristic(AutotunerHeuristic):
    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.computed_fragment import _static_selection_extent
        from ..cute.computed_fragment import computed_fragment_supported
        from ..cute.topk import match_topk_root

        host = device_ir.host_function
        assert host is not None
        roots = set()
        with host:
            for root, graphs in fragment_root_regions(device_ir):
                # Preserve the native row owner, including its late layout and
                # alias guards. This coordinate belongs only to fragment roots.
                if (
                    match_topk_root(
                        graphs,
                        noncanonical_block_ids=device_ir.noncanonical_task_origin_block_ids,
                    )
                    is not None
                ):
                    continue
                if not computed_fragment_supported(env, graphs):
                    continue
                for info in graphs:
                    for node in info.graph.nodes:
                        if node.target is not torch.ops.aten.topk.default:
                            continue
                        source = node.args[0]
                        assert isinstance(source, Node)
                        value = source.meta["val"]
                        assert isinstance(value, torch.Tensor)
                        k = node.args[1]
                        if isinstance(k, int) and network_topk_supported(
                            value.dtype, _static_selection_extent(env, source), k
                        ):
                            roots.add(root)
        env.config_spec.cute_fragment_topk_network_root_ids = frozenset(roots)
        return frozenset({"input_tensor_metadata"}) if roots else frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_topk_network_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_topk_network_root_ids:
        return
    previous = spec.create_config_generation()
    try:
        carrier = resource_carrier
        if carrier is None:
            _, carrier = previous.canonicalize_flat(previous.default_flat())
        previous.strict_config_pair(carrier)
    except InvalidConfig:
        return
    spec.cute_fragment_topk_network_search_enabled = True
    generation = spec.create_config_generation()
    try:
        generation.strict_config_pair(
            Config.from_dict(deepcopy(carrier.config) | {KEY: True})
        )
    except InvalidConfig:
        return
    # Registered after every previous group: neither old witnesses nor their
    # sampling/RNG policy are changed by the new independent coordinate.
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.fragment_topk_network",
            version=1,
            key=KEY,
            domain=(False, True),
            legacy=False,
            witnesses=(CoverageWitness(carrier, True),),
            deferred=True,
        )
    )
