"""``cute_fragment_unrolled_loops``: constexpr element loops in fragment phases.

Every phase of the CTA-local fragment lowering walks its elements as
``for index in range(thread, count, threads)``: a loop with a dynamic start
whose body loads one element from global memory and stores it to shared
memory, so each slot of the walk waits for its own load before the next one
issues.  With the knob on, a phase of at most ``UNROLLED_SLOTS`` slots per
thread is emitted as a ``cutlass.range_constexpr`` walk instead, so the
compiler sees every load of the phase at once and issues them together.
The price is code size and register pressure, which can cost occupancy; the
autotuner decides.
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

KEY = "cute_fragment_unrolled_loops"


class CuteFragmentUnrolledLoopsHeuristic(AutotunerHeuristic):
    """Register every complete fragment root; all of them walk elements."""

    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        host = device_ir.host_function
        assert host is not None
        with host:
            env.config_spec.cute_fragment_unrolled_loops_root_ids = frozenset(
                root
                for root, graphs in fragment_root_regions(device_ir)
                if computed_fragment_discovery_supported(env, graphs)
            )
        # Shape/stride assumptions of an admitted root must survive rebinding.
        return (
            frozenset({"input_tensor_metadata"})
            if env.config_spec.cute_fragment_unrolled_loops_root_ids
            else frozenset()
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_unrolled_loops_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_unrolled_loops_root_ids:
        return
    carrier = fragment_coverage_carrier(spec, resource_carrier)
    if carrier is None:
        return
    previous_enabled = spec.cute_fragment_unrolled_loops_search_enabled
    spec.cute_fragment_unrolled_loops_search_enabled = True
    if not register_fragment_boolean_coverage(spec, KEY, carrier):
        spec.cute_fragment_unrolled_loops_search_enabled = previous_enabled
