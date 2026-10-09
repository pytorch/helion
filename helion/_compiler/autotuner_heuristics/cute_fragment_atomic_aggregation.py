"""Default-off warp aggregation for private integer update epochs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...language import _tracing_ops
from .cute_fragment_common import computed_fragment_discovery_supported
from .cute_fragment_common import fragment_coverage_carrier
from .cute_fragment_common import fragment_root_regions
from .cute_fragment_common import register_fragment_boolean_coverage
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fragment_atomic_aggregation"


class CuteFragmentAtomicAggregationHeuristic(AutotunerHeuristic):
    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.local_atomic import atomic_target_origins

        host = device_ir.host_function
        assert host is not None
        roots = set()
        with host:
            for root, graphs in fragment_root_regions(device_ir):
                updates = atomic_target_origins(graphs)
                eligible = any(
                    target.target is not _tracing_ops._host_tensor
                    and isinstance(target.meta.get("val"), torch.Tensor)
                    and target.meta["val"].dtype == torch.int32
                    and not node.users
                    and node.args[3] == "relaxed"
                    for node, target in updates.items()
                )
                # Includes the existing nonescape/no-alias, uniform-loop and
                # initialization -> updates -> final-read lifetime proof.
                if eligible and computed_fragment_discovery_supported(env, graphs):
                    roots.add(root)
        env.config_spec.cute_fragment_atomic_aggregation_root_ids = frozenset(roots)
        # Emission retains the logical-domain and wrapped-index bounds.
        return frozenset({"input_tensor_metadata"}) if roots else frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_atomic_aggregation_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_atomic_aggregation_root_ids:
        return
    carrier = fragment_coverage_carrier(spec, resource_carrier)
    if carrier is None:
        return
    spec.cute_fragment_atomic_aggregation_search_enabled = True
    register_fragment_boolean_coverage(spec, KEY, carrier)
