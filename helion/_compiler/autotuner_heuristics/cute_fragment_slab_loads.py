"""``cute_fragment_slab_loads``: one slab per tensor for repeated static windows.

A fragment root that reads one host tensor through several static windows
(the anchored chunk gram stages ``q``, ``k`` and ``gc`` per 16-row tier:
twenty window loads, 160 ``k`` rows for 64 distinct ones) pays a global
round trip and a barrier per window.  With the knob on, the windows of one
tensor that differ only along one dimension by static, in-bounds offsets
read views of a single slab staged once at the start of the root
(``fragment_slab_loads.plan_slab_loads``); the slab never spans more rows
than the windows load together.  Only roots whose plan has at least one
group carry the knob.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

KEY = "cute_fragment_slab_loads"


class CuteFragmentSlabLoadsHeuristic(AutotunerHeuristic):
    """Register every complete fragment root with a slab group."""

    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.fragment_slab_loads import plan_slab_loads

        host = device_ir.host_function
        assert host is not None
        with host:
            env.config_spec.cute_fragment_slab_loads_root_ids = frozenset(
                root
                for root, graphs in fragment_root_regions(device_ir)
                if computed_fragment_discovery_supported(env, graphs)
                and plan_slab_loads(env, graphs)
            )
        # The plan reads static window bounds and tensor extents.
        return (
            frozenset({"input_tensor_metadata"})
            if env.config_spec.cute_fragment_slab_loads_root_ids
            else frozenset()
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_slab_loads_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_slab_loads_root_ids:
        return
    carrier = fragment_coverage_carrier(spec, resource_carrier)
    if carrier is None:
        return
    previous_enabled = spec.cute_fragment_slab_loads_search_enabled
    spec.cute_fragment_slab_loads_search_enabled = True
    if not register_fragment_boolean_coverage(spec, KEY, carrier):
        spec.cute_fragment_slab_loads_search_enabled = previous_enabled
