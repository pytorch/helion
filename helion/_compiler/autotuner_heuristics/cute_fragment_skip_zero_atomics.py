"""Default-off zero elision for proved private Int32 update epochs."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...language import _tracing_ops
from ...runtime.config import Config
from .cute_fragment_common import computed_fragment_discovery_supported
from .cute_fragment_common import fragment_root_regions
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fragment_skip_zero_atomics"


class CuteFragmentSkipZeroAtomicsHeuristic(AutotunerHeuristic):
    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.dead_zero_atomics import dead_zero_atomic_results
        from ..cute.local_atomic import atomic_target_origins

        host = device_ir.host_function
        assert host is not None
        roots = set()
        with host:
            for root, graphs in fragment_root_regions(device_ir):
                updates = atomic_target_origins(graphs)
                dead_results = dead_zero_atomic_results(graphs, env, allow_unbound=True)
                eligible = any(
                    target.target is not _tracing_ops._host_tensor
                    and isinstance(target.meta.get("val"), torch.Tensor)
                    and target.meta["val"].dtype == torch.int32
                    and (not node.users or node in dead_results)
                    and node.args[3] == "relaxed"
                    for node, target in updates.items()
                )
                # Includes the existing nonescape/no-alias, uniform-loop and
                # initialization -> updates -> final-read lifetime proof.
                if eligible and computed_fragment_discovery_supported(env, graphs):
                    roots.add(root)
        env.config_spec.cute_fragment_skip_zero_atomics_root_ids = frozenset(roots)
        # Emission retains the logical-domain and wrapped-index bounds.
        return frozenset({"input_tensor_metadata"}) if roots else frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_skip_zero_atomics_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_skip_zero_atomics_root_ids:
        return
    previous = spec.create_config_generation()
    try:
        carrier = resource_carrier
        if carrier is None:
            _, carrier = previous.canonicalize_flat(previous.default_flat())
        previous.strict_config_pair(carrier)
    except InvalidConfig:
        return
    spec.cute_fragment_skip_zero_atomics_search_enabled = True
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
            mechanism="cute.fragment_skip_zero_atomics",
            version=1,
            key=KEY,
            domain=(False, True),
            legacy=False,
            witnesses=(CoverageWitness(carrier, True),),
            deferred=True,
        )
    )
