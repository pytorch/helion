"""``cute_fragment_split_k``: idle warps share K in a tf32 fragment mma phase.

The tf32 ``mma.sync`` phase of the fragment lowering (``cute_fp32_dot``)
gives each warp whole 16 x 8 output tiles.  A 16-row dot over a long
contraction (the anchored chunk gram's ``[16, cols] x [cols, 128]`` tiers)
has fewer tiles than the CTA has warps, so most warps wait at the phase's
barrier while a few walk the whole K chain alone.  With the knob on, the
warps are grouped per tile: each walks a disjoint range of K steps (the
accumulator joins the first), writes fp32 partials, and the result reads
their fixed-order sum.  The phase's critical path shrinks by the number of
splits at the cost of one partial tile per split in shared memory.  Only
roots with a tf32-eligible fp32 dot carry the knob, and a config pins it on
only together with ``cute_fp32_dot='tf32_mma'``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageDependency
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...runtime.config import Config
from .cute_fp32_dot import fragment_fp32_dot_roots
from .cute_fragment_common import fragment_coverage_carrier
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fragment_split_k"


class CuteFragmentSplitKHeuristic(AutotunerHeuristic):
    """Register every complete fragment root with a tf32-eligible dot."""

    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        host = device_ir.host_function
        assert host is not None
        with host:
            env.config_spec.cute_fragment_split_k_root_ids = fragment_fp32_dot_roots(
                env, device_ir
            )
        return (
            frozenset({"input_tensor_metadata"})
            if env.config_spec.cute_fragment_split_k_root_ids
            else frozenset()
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_split_k_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_split_k_root_ids:
        return
    carrier = fragment_coverage_carrier(spec, resource_carrier)
    if carrier is None:
        return
    # The witness rides the tf32 mode: the carrier pins it on first and
    # declares it as a dependency on the tf32 group (a carrier that merely
    # sets another group's non-legacy mode is refused as "not ordinary legacy
    # modes" when the population is built and never reaches the search).
    tf32_group = next(
        (
            group
            for group in spec.compiler_coverage_groups
            if group.key == "cute_fp32_dot"
        ),
        None,
    )
    if tf32_group is None:
        return
    tf32_carrier = Config.from_dict(
        deepcopy(carrier.config) | {"cute_fp32_dot": "tf32_mma"}
    )
    previous_enabled = spec.cute_fragment_split_k_search_enabled
    spec.cute_fragment_split_k_search_enabled = True
    generation = spec.create_config_generation()
    try:
        _, tf32_carrier = generation.strict_config_pair(tf32_carrier)
        if tf32_carrier.config.get("cute_fp32_dot") != "tf32_mma":
            raise InvalidConfig("split-K carrier lost the tf32 mode")
        generation.strict_config_pair(
            Config.from_dict(deepcopy(tf32_carrier.config) | {KEY: True})
        )
    except InvalidConfig:
        spec.cute_fragment_split_k_search_enabled = previous_enabled
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism=KEY.replace("cute_", "cute.", 1),
            version=1,
            key=KEY,
            domain=(False, True),
            legacy=False,
            witnesses=(CoverageWitness(tf32_carrier, True),),
            dependencies=(
                CoverageDependency(tf32_group.mechanism, "cute_fp32_dot", "tf32_mma"),
            ),
            deferred=True,
        )
    )
