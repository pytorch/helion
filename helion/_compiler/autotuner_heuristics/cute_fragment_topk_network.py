"""Optional bounded network selection for complete shared fragment inputs."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageDependency
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...runtime.config import Config
from .cute_fragment_common import fragment_coverage_carrier
from .cute_fragment_common import fragment_root_regions
from .cute_fragment_common import register_fragment_boolean_coverage
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fragment_topk_network"
CTA_KEY = "cute_fragment_cta_topk"


def network_topk_supported(
    dtype: torch.dtype, n: int | None, k: int, *, values_only: bool = False
) -> bool:
    # At most 32 Int64 register keys per lane. Larger/different-typed selections
    # keep the serial implementation. Int64 values need no index packing when
    # the selection's public index output is proved unused.
    return (
        (
            dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int32)
            or dtype == torch.int64
            and values_only
        )
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
        from ..cute.ordered_selection import is_ordered_selection
        from ..cute.ordered_selection import ordered_selection_args
        from ..cute.ordered_selection import selection_values_only
        from ..cute.topk import match_topk_root

        host = device_ir.host_function
        assert host is not None
        roots = set()
        cta_roots = set()
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
                        if not is_ordered_selection(node):
                            continue
                        arguments = ordered_selection_args(node)
                        assert arguments is not None
                        source, k, *_ = arguments
                        value = source.meta["val"]
                        assert isinstance(value, torch.Tensor)
                        n = _static_selection_extent(env, source)
                        if k is None:
                            k = n
                        if isinstance(k, int) and network_topk_supported(
                            value.dtype, n, k, values_only=selection_values_only(node)
                        ):
                            roots.add(root)
                            if value.dtype == torch.int64 and selection_values_only(
                                node
                            ):
                                cta_roots.add(root)
        env.config_spec.cute_fragment_topk_network_root_ids = frozenset(roots)
        env.config_spec.cute_fragment_cta_topk_root_ids = frozenset(cta_roots)
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
    carrier = fragment_coverage_carrier(spec, resource_carrier)
    if carrier is None:
        return
    spec.cute_fragment_topk_network_search_enabled = True
    register_fragment_boolean_coverage(spec, KEY, carrier)
    if not spec.cute_fragment_cta_topk_root_ids:
        return
    dependency = next(
        (group for group in spec.compiler_coverage_groups if group.key == KEY),
        None,
    )
    if dependency is None:
        return
    cta_carrier = Config.from_dict(deepcopy(carrier.config) | {KEY: True})
    spec.cute_fragment_cta_topk_search_enabled = True
    try:
        generation = spec.create_config_generation()
        generation.strict_config_pair(
            Config.from_dict(deepcopy(cta_carrier.config) | {CTA_KEY: True})
        )
    except InvalidConfig:
        spec.cute_fragment_cta_topk_search_enabled = False
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.fragment_cta_topk",
            version=1,
            key=CTA_KEY,
            domain=(False, True),
            legacy=False,
            witnesses=(CoverageWitness(cta_carrier, True),),
            dependencies=(CoverageDependency(dependency.mechanism, KEY, True),),
            deferred=True,
        )
    )
