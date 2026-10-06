"""Default-off bounded immutable host snapshots with integer CTA reductions."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import TYPE_CHECKING
from typing import cast

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageDependency
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...runtime.config import Config
from .cute_fragment_common import fragment_root_regions
from .cute_fragment_threads import THREADS
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact

KEY = "cute_fragment_register_snapshots"


class CuteFragmentRegisterSnapshotsHeuristic(AutotunerHeuristic):
    name = KEY
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.computed_fragment import computed_fragment_supported
        from ..cute.register_snapshots import MAX_SLOTS
        from ..cute.register_snapshots import snapshot_chains
        from ..cute.warp_results import static_shape

        host = device_ir.host_function
        assert host is not None
        roots = set()
        sizes = []
        with host:
            for root, graphs in fragment_root_regions(device_ir):
                chains = snapshot_chains(graphs, env, allow_unbound=True)
                if not chains or not computed_fragment_supported(
                    env, graphs, snapshot_owned=True
                ):
                    continue
                counts = [
                    math.prod(shape)
                    for node in chains
                    if (shape := static_shape(node.meta.get("val"), env)) is not None
                    and 0 < math.prod(shape) <= max(THREADS) * MAX_SLOTS
                ]
                # Includes the existing nonescape/no-alias, uniform-loop and
                # initialization -> updates -> final-read lifetime proof.
                if counts:
                    roots.add(root)
                    sizes.extend(counts)
        env.config_spec.cute_fragment_register_snapshots_root_ids = frozenset(roots)
        env.config_spec.cute_fragment_thread_root_ids |= frozenset(roots)
        env.config_spec.cute_fragment_register_snapshot_min_threads = (
            (min(sizes) + MAX_SLOTS - 1) // MAX_SLOTS if sizes else 0
        )
        # Emission retains the logical-domain and wrapped-index bounds.
        return frozenset({"input_tensor_metadata"}) if roots else frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def register_fragment_register_snapshots_coverage(
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
) -> None:
    spec = env.config_spec
    if not spec.cute_fragment_register_snapshots_root_ids:
        return
    previous = spec.create_config_generation()
    try:
        carrier = resource_carrier
        if carrier is None:
            _, carrier = previous.canonicalize_flat(previous.default_flat())
        previous.strict_config_pair(carrier)
    except InvalidConfig:
        return
    required = spec.cute_fragment_register_snapshot_min_threads
    if cast("int", carrier.get("cute_fragment_threads", 128)) < required:
        carrier = Config.from_dict(
            deepcopy(carrier.config)
            | {"cute_fragment_threads": min(t for t in THREADS if t >= required)}
        )
    spec.cute_fragment_register_snapshots_search_enabled = True
    generation = spec.create_config_generation()
    try:
        generation.strict_config_pair(
            Config.from_dict(deepcopy(carrier.config) | {KEY: True})
        )
    except InvalidConfig:
        return
    dependencies = ()
    threads = cast("int", carrier.get("cute_fragment_threads", 128))
    if threads != 128:
        thread_group = next(
            (
                group
                for group in spec.compiler_coverage_groups
                if group.key == "cute_fragment_threads"
            ),
            None,
        )
        if thread_group is None:
            return
        dependencies = (
            CoverageDependency(thread_group.mechanism, thread_group.key, threads),
        )
    # Registered after every previous group: neither old witnesses nor their
    # sampling/RNG policy are changed by the new independent coordinate.
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.fragment_register_snapshots",
            version=1,
            key=KEY,
            domain=(False, True),
            legacy=False,
            witnesses=(CoverageWitness(carrier, True),),
            dependencies=dependencies,
            deferred=True,
        )
    )
