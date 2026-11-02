"""``cute_fragment_batched_loads``: one barrier per run of independent host loads.

Every host load the fragment lowering stages ends its element walk with a
CTA barrier, so a root that loads several windows before using any of them
(the anchored chunk gram stages five per tier) pays one global-memory
latency and one barrier per window, in series.  With the knob on, a run of
consecutive host loads whose address and mask recipes read no shared memory
is emitted without the barriers between them: every load of the run is in
flight at once, and the first node that is not such a load issues the one
barrier that publishes them all.  Buffer recycling stays safe because the
unfenced phases read nothing from shared memory and every buffer they
write is either fresh or last read before the preceding barrier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from .cute_fragment_common import computed_fragment_discovery_supported
from .cute_fragment_common import fragment_coverage_carrier
from .cute_fragment_common import fragment_root_regions
from .cute_fragment_common import register_fragment_boolean_coverage
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fragment_batched_loads"


def _host_loads(graphs: list[GraphInfo]) -> int:
    return sum(
        1
        for info in graphs
        for node in info.graph.nodes
        if node.target is memory_ops.load
        and isinstance(node.args[0], Node)
        and node.args[0].target is _tracing_ops._host_tensor
    )


class CuteFragmentBatchedLoadsHeuristic(AutotunerHeuristic):
    """Register every complete fragment root with two or more host loads."""

    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        host = device_ir.host_function
        assert host is not None
        with host:
            env.config_spec.cute_fragment_batched_loads_root_ids = frozenset(
                root
                for root, graphs in fragment_root_regions(device_ir)
                if computed_fragment_discovery_supported(env, graphs)
                and _host_loads(graphs) >= 2
            )
        return (
            frozenset({"input_tensor_metadata"})
            if env.config_spec.cute_fragment_batched_loads_root_ids
            else frozenset()
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_batched_loads_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_batched_loads_root_ids:
        return
    carrier = fragment_coverage_carrier(spec, resource_carrier)
    if carrier is None:
        return
    previous_enabled = spec.cute_fragment_batched_loads_search_enabled
    spec.cute_fragment_batched_loads_search_enabled = True
    if not register_fragment_boolean_coverage(spec, KEY, carrier):
        spec.cute_fragment_batched_loads_search_enabled = previous_enabled
