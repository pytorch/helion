"""The CuTe computed-fragment knob table.

Every autotuner knob of the CTA-local fragment lowering (``computed_fragment``)
is one :class:`FragmentKnob` row: its config key, the roots it applies to, the
spec facts it records and how its coverage witness is built.  Two generic
pieces turn a row into the objects the autotuner surface runs:

* :func:`fragment_knob_heuristic` builds the ``AutotunerHeuristic`` the ranked
  registry calls: ``register_facts`` projects the row's roots (under the host
  function), stores them on the ``ConfigSpec`` and returns the metadata
  specialization fact; ``is_eligible`` is ``False`` for every row -- search
  coverage, never a seed or a promoted default, offers a fragment knob.
* :func:`fragment_knob_registrar` builds the coverage registrar
  ``heuristics.register_coverage_groups`` calls in its historical order: the
  root gate, the carrier (the admitted default with the MMA dataflow off, or a
  parent group's witness), the search-field enable, the strict witness probe
  and the ``CompilerCoverageGroup`` declaration.

Rows differ only in data: ordered root predicates, a custom root projection
where a knob records extra facts, and a :class:`KnobCoverage` whose optional
hooks (``admit``, ``carrier``, ``plan``) cover the coupled witnesses (thread
geometry, tf32 pins, a parent group's carrier, validated candidates).  The
generated class and registrar names are bound at the bottom of this module
and re-exported by ``heuristics/__init__`` where tests patch them.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import starmap
import math
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import cast

import torch
from torch.fx import Node

from ....autotuner.compiler_coverage import CompilerCoverageGroup
from ....autotuner.compiler_coverage import CoverageDependency
from ....autotuner.compiler_coverage import CoverageValue
from ....autotuner.compiler_coverage import CoverageWitness
from ....exc import InvalidConfig
from ....language import _tracing_ops
from ....language import memory_ops
from ....language import scan_ops
from ....runtime.config import Config
from ...autotuner_heuristics.registry import AutotunerHeuristic
from ..fragment_dot_support import fragment_half_dot_supported
from .config_keys import CUTE_FRAGMENT_KEYS
from .heuristics import cute_fragment_bounded_gather
from .heuristics import cute_fragment_threads
from .heuristics import cute_fragment_topk_network
from .heuristics.cute_fp32_dot import fragment_fp32_dot_roots
from .heuristics.cute_fragment_common import FragmentRootRequirement
from .heuristics.cute_fragment_common import computed_fragment_discovery_supported
from .heuristics.cute_fragment_common import fragment_coverage_carrier
from .heuristics.cute_fragment_common import fragment_root_regions
from .heuristics.cute_fragment_reduction import MODES as REDUCTION_MODES
from .heuristics.cute_fragment_reduction import fragment_warp_reduction_supported
from .heuristics.cute_fragment_resources import fragment_resource_carrier
from .heuristics.cute_fragment_scan_exports import validate_scan_exports
from .heuristics.cute_fragment_warp_producer_regions import (
    validate_warp_producer_regions,
)
from .heuristics.cute_fragment_warp_scan import WARP_SCAN_DTYPES

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Iterable

    from torch.fx import Graph

    from ....autotuner.config_generation import ConfigGeneration
    from ....autotuner.config_spec import ConfigSpec
    from ...autotuner_heuristics.registry import CompilerHeuristicSpecializationFact
    from ...compile_environment import CompileEnvironment
    from ...device_ir import DeviceIR
    from ...device_ir import GraphInfo

SCAN_MODES = ("serial", "cooperative")
# A dense row this many bytes long (one full sweep of the 32 banks) starts
# every row in the same bank.
BANK_SWEEP_BYTES = 128


# --------------------------------------------------------------------------
# Rows


@dataclass(frozen=True)
class FragmentRegion:
    """One root the explicit ordered phase codegen accepts, with its graphs."""

    root: int
    graphs: list[GraphInfo]

    @property
    def root_graph(self) -> Graph:
        return next(g.graph for g in self.graphs if g.graph_id == self.root)


def fragment_regions(ir: DeviceIR) -> list[FragmentRegion]:
    return list(starmap(FragmentRegion, fragment_root_regions(ir)))


@dataclass(frozen=True)
class RootFacts:
    """What a knob's root projection found: its roots and extra spec facts.

    ``spec_values`` are ``(attribute, value)`` pairs stored on the spec after
    the roots; ``specialized`` requests the metadata fact even without roots.
    """

    root_ids: frozenset[int]
    spec_values: tuple[tuple[str, object], ...] = ()
    specialized: bool = False


@dataclass(frozen=True)
class Carrier:
    """A coverage carrier and the group couplings its construction declared."""

    config: Config
    dependencies: tuple[CoverageDependency, ...] = ()
    pins: tuple[tuple[str, CoverageValue], ...] = ()


@dataclass(frozen=True)
class CoveragePlan:
    """The witnesses a probe admitted, with the group's complete dependencies."""

    witnesses: tuple[tuple[Config, CoverageValue], ...]
    dependencies: tuple[CoverageDependency, ...] = ()
    supplemental: tuple[tuple[Config, CoverageValue], ...] = ()


@dataclass(frozen=True)
class CoverageContext:
    """One registrar call: the knob, the environment and the shared carriers."""

    knob: FragmentKnob
    env: CompileEnvironment
    device_ir: DeviceIR
    resource_carrier: Config | None
    extended_only: bool

    @property
    def spec(self) -> ConfigSpec:
        return self.env.config_spec

    def group(self, key: str) -> CompilerCoverageGroup | None:
        return next(
            (g for g in self.spec.compiler_coverage_groups if g.key == key), None
        )

    def probe(
        self, generation: ConfigGeneration, config: Config, value: CoverageValue
    ) -> Config:
        """Strictly admit *config* with the knob set to *value*; the effective config."""
        _, effective = generation.strict_config_pair(
            Config.from_dict(deepcopy(config.config) | {self.knob.name: value})
        )
        return effective


@dataclass(frozen=True)
class KnobCoverage:
    """How a knob's coverage group is declared and witnessed.

    The default is one deferred Boolean witness on the admitted carrier.
    ``rollback`` restores the search-field flag when the probe refuses the
    witness; rows that historically left the field searchable keep ``False``.
    ``resource`` makes the shared storage-bounded carrier the carrier base; the
    enum rows instead register it as a supplemental witness.
    """

    mechanism: str | None = None
    version: int = 1
    domain: tuple[CoverageValue, ...] = (False, True)
    legacy: CoverageValue = False
    witness_values: tuple[CoverageValue, ...] = (True,)
    deferred: bool = True
    rollback: bool = True
    resource: bool = True
    admit: Callable[[CoverageContext], bool] | None = None
    resource_override: Callable[[CoverageContext], Config | None] | None = None
    carrier: Callable[[CoverageContext], Carrier | None] | None = None
    plan: Callable[[CoverageContext, Carrier], CoveragePlan | None] | None = None


@dataclass(frozen=True)
class FragmentKnob:
    """One computed-fragment knob: its key, its roots and its coverage."""

    name: str
    fragment: str
    # Position in the ranked heuristic tuple; None for a knob whose roots
    # another row's projection records (no heuristic class of its own).
    priority: int | None
    # Ordered per-region predicates (every one must hold), or a projection
    # that also records extra spec facts.
    applies_when: tuple[Callable[[CompileEnvironment, FragmentRegion], bool], ...] = ()
    roots: Callable[[CompileEnvironment, DeviceIR], RootFacts] | None = None
    # A whole-kernel pre-check: when it fails nothing is recorded; when it
    # holds the metadata fact is registered even for an empty root set.
    precheck: Callable[[CompileEnvironment, DeviceIR], bool] | None = None
    root_ids_attr: str | None = None
    search_enabled_attr: str | None = None
    coverage: KnobCoverage = KnobCoverage()

    @property
    def root_ids_attribute(self) -> str:
        return self.root_ids_attr or f"{self.name}_root_ids"

    @property
    def search_enabled_attribute(self) -> str:
        return self.search_enabled_attr or f"{self.name}_search_enabled"

    def project(self, env: CompileEnvironment, ir: DeviceIR) -> RootFacts:
        if self.roots is not None:
            return self.roots(env, ir)
        return RootFacts(
            frozenset(
                region.root
                for region in fragment_regions(ir)
                if all(applies(env, region) for applies in self.applies_when)
            )
        )


# --------------------------------------------------------------------------
# The generic heuristic and registrar


class FragmentKnobHeuristic(AutotunerHeuristic):
    """Records one knob's roots and facts; coverage, not a seed, offers it."""

    knob: ClassVar[FragmentKnob]
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        knob = cls.knob
        if knob.precheck is not None and not knob.precheck(env, device_ir):
            return frozenset()
        host = device_ir.host_function
        assert host is not None
        with host:
            facts = knob.project(env, device_ir)
        spec = env.config_spec
        setattr(spec, knob.root_ids_attribute, facts.root_ids)
        for attribute, value in facts.spec_values:
            setattr(spec, attribute, value)
        # Shape/stride metadata determines eligibility and the configured
        # extents of an admitted root, so both must survive rebinding.
        if knob.precheck is not None or facts.specialized or facts.root_ids:
            return frozenset({"input_tensor_metadata"})
        return frozenset()

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return False


def _class_name(fragment: str) -> str:
    return "CuteFragment{}Heuristic".format(
        "".join(
            "TopK" if part == "topk" else part.capitalize()
            for part in fragment.split("_")
        )
    )


def fragment_knob_heuristic(knob: FragmentKnob) -> type[FragmentKnobHeuristic]:
    assert knob.priority is not None
    return cast(
        "type[FragmentKnobHeuristic]",
        type(
            _class_name(knob.fragment),
            (FragmentKnobHeuristic,),
            {
                "knob": knob,
                "name": knob.name,
                "priority": knob.priority,
                "__module__": __name__,
            },
        ),
    )


def register_fragment_knob_coverage(
    knob: FragmentKnob,
    env: CompileEnvironment,
    device_ir: DeviceIR,
    *,
    resource_carrier: Config | None = None,
    extended_only: bool = False,
) -> None:
    """Declare *knob*'s coverage group after its field is proved and admitted.

    Structural requirements (the carrier, a parent group) are checked before
    the search field is enabled and leave the spec untouched when missing; a
    refused witness after enabling follows the row's ``rollback`` policy.
    """
    spec = env.config_spec
    if not getattr(spec, knob.root_ids_attribute):
        return
    coverage = knob.coverage
    ctx = CoverageContext(knob, env, device_ir, resource_carrier, extended_only)
    if coverage.admit is not None and not coverage.admit(ctx):
        return
    if coverage.carrier is not None:
        carrier = coverage.carrier(ctx)
    else:
        resource = None
        if coverage.resource_override is not None:
            resource = coverage.resource_override(ctx)
        elif coverage.resource:
            resource = resource_carrier
        config = fragment_coverage_carrier(spec, resource)
        carrier = None if config is None else Carrier(config)
    if carrier is None:
        return
    previous = getattr(spec, knob.search_enabled_attribute)
    setattr(spec, knob.search_enabled_attribute, True)
    plan = (coverage.plan or _default_plan)(ctx, carrier)
    if plan is None:
        if coverage.rollback:
            setattr(spec, knob.search_enabled_attribute, previous)
        return
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism=coverage.mechanism or knob.name.replace("cute_", "cute.", 1),
            version=coverage.version,
            key=knob.name,
            domain=coverage.domain,
            legacy=coverage.legacy,
            witnesses=tuple(starmap(CoverageWitness, plan.witnesses)),
            supplemental_witnesses=tuple(starmap(CoverageWitness, plan.supplemental)),
            dependencies=plan.dependencies,
            deferred=coverage.deferred,
        )
    )


def fragment_knob_registrar(
    knob: FragmentKnob,
) -> Callable[..., None]:
    def register(
        env: CompileEnvironment,
        device_ir: DeviceIR,
        *,
        resource_carrier: Config | None = None,
        extended_only: bool = False,
    ) -> None:
        register_fragment_knob_coverage(
            knob,
            env,
            device_ir,
            resource_carrier=resource_carrier,
            extended_only=extended_only,
        )

    register.__name__ = register.__qualname__ = (
        f"register_fragment_{knob.fragment}_coverage"
    )
    return register


# --------------------------------------------------------------------------
# Witness plans


def _default_plan(ctx: CoverageContext, carrier: Carrier) -> CoveragePlan | None:
    """One Boolean witness on the carrier, or every enum mode strictly kept."""
    coverage = ctx.knob.coverage
    generation = ctx.spec.create_config_generation()
    try:
        if coverage.domain == (False, True):
            ctx.probe(generation, carrier.config, True)
        else:
            for mode in coverage.domain:
                effective = ctx.probe(generation, carrier.config, mode)
                if effective.config.get(ctx.knob.name, coverage.legacy) != mode:
                    return None
    except InvalidConfig:
        return None
    supplemental = ()
    if not coverage.resource and ctx.resource_carrier is not None:
        supplemental = ((ctx.resource_carrier, coverage.witness_values[0]),)
    return CoveragePlan(
        tuple((carrier.config, value) for value in coverage.witness_values),
        carrier.dependencies,
        supplemental,
    )


def _thread_group_dependency(
    ctx: CoverageContext, threads: int
) -> CoverageDependency | None:
    group = ctx.group("cute_fragment_threads")
    if group is None:
        return None
    return CoverageDependency(group.mechanism, group.key, threads)


def _raise_threads(carrier: Config, required: int) -> Config:
    """*carrier* with its CTA worker count raised to the smallest legal count."""
    if cast("int", carrier.get("cute_fragment_threads", 128)) >= required:
        return carrier
    return Config.from_dict(
        deepcopy(carrier.config)
        | {
            "cute_fragment_threads": min(
                t for t in cute_fragment_threads.THREADS if t >= required
            )
        }
    )


def _thread_coupled_plan(
    min_threads_attr: str | None,
) -> Callable[[CoverageContext, Carrier], CoveragePlan | None]:
    """The witness rides the largest legacy worker domain (the widest
    one-scalar-per-lane proof) and declares that coupling on the thread group."""

    def plan(ctx: CoverageContext, carrier: Carrier) -> CoveragePlan | None:
        generation = ctx.spec.create_config_generation()
        threads = max(cute_fragment_threads.LEGACY_COVERAGE_THREADS)
        if min_threads_attr is not None:
            # Newly supported group counts can require a larger legal CTA.
            required = cast("int", getattr(ctx.spec, min_threads_attr))
            if required > threads:
                threads = min(
                    count
                    for count in cute_fragment_threads.THREADS
                    if count >= required
                )
        requested = Config.from_dict(
            deepcopy(carrier.config.config) | {"cute_fragment_threads": threads}
        )
        try:
            _, effective = generation.strict_config_pair(requested)
            ctx.probe(generation, effective, True)
        except InvalidConfig:
            return None
        dependency = _thread_group_dependency(ctx, threads)
        if dependency is None:
            return None
        return CoveragePlan(((effective, True),), (dependency, *carrier.dependencies))

    return plan


def _min_threads_plan(
    min_threads_attr: str,
) -> Callable[[CoverageContext, Carrier], CoveragePlan | None]:
    """The carrier's worker count is raised to what the register budget
    needs; a non-default count declares the coupling on the thread group."""

    def plan(ctx: CoverageContext, carrier: Carrier) -> CoveragePlan | None:
        config = _raise_threads(
            carrier.config, cast("int", getattr(ctx.spec, min_threads_attr))
        )
        generation = ctx.spec.create_config_generation()
        try:
            ctx.probe(generation, config, True)
        except InvalidConfig:
            return None
        dependencies: tuple[CoverageDependency, ...] = ()
        threads = cast("int", config.get("cute_fragment_threads", 128))
        if threads != 128:
            dependency = _thread_group_dependency(ctx, threads)
            if dependency is None:
                return None
            dependencies = (dependency,)
        return CoveragePlan(((config, True),), dependencies + carrier.dependencies)

    return plan


def _stripped_plan(ctx: CoverageContext, carrier: Carrier) -> CoveragePlan | None:
    """The effective config with the knob on, stripped of the knob, is the witness."""
    generation = ctx.spec.create_config_generation()
    try:
        effective = ctx.probe(generation, carrier.config, True)
    except InvalidConfig:
        return None
    witness = Config.from_dict(
        {k: v for k, v in effective.config.items() if k != ctx.knob.name}
    )
    return CoveragePlan(((witness, True),), carrier.dependencies)


def _pinned_plan(
    widen: bool,
) -> Callable[[CoverageContext, Carrier], CoveragePlan | None]:
    """The pinned carrier is strictly normalized (every pin kept) and probed;
    with *widen* a supplemental witness widens the root's innermost tile."""

    def plan(ctx: CoverageContext, carrier: Carrier) -> CoveragePlan | None:
        generation = ctx.spec.create_config_generation()
        try:
            _, pinned = generation.strict_config_pair(carrier.config)
            if any(pinned.config.get(key) != value for key, value in carrier.pins):
                raise InvalidConfig("coverage carrier lost a pinned mode")
            wide = _widest_inner_tile(ctx.spec, pinned) if widen else None
            for witness in (pinned, wide):
                if witness is not None:
                    ctx.probe(generation, witness, True)
        except InvalidConfig:
            return None
        return CoveragePlan(
            ((pinned, True),),
            carrier.dependencies,
            ((wide, True),) if wide is not None else (),
        )

    return plan


def _candidate_plan(
    scans: Callable[[CoverageContext], Iterable[Node]],
    validate: Callable[[CompileEnvironment, DeviceIR, Config], None],
) -> Callable[[CoverageContext, Carrier], CoveragePlan | None]:
    """The mechanism needs one complete row per CTA: the carrier is tried
    first, then its copies with a proved scan's leading tile coordinates set
    to 1; the first candidate the validator admits is the witness."""

    def plan(ctx: CoverageContext, carrier: Carrier) -> CoveragePlan | None:
        spec = ctx.spec
        host = ctx.device_ir.host_function
        assert host is not None
        carriers = [carrier.config]
        with host:
            for node in scans(ctx):
                candidate = Config.from_dict(deepcopy(carrier.config.config))
                value = cast("torch.Tensor", node.meta["val"])
                for extent in value.shape[:-1]:
                    bid = ctx.env.resolve_block_id(extent)
                    if bid is not None and bid in spec.block_sizes.valid_block_ids():
                        candidate.block_sizes[
                            spec.block_sizes.block_id_to_index(bid)
                        ] = 1
                if candidate not in carriers:
                    carriers.append(candidate)
        for config in carriers:
            try:
                requested = ctx.probe(spec.create_config_generation(), config, True)
                with host:
                    validate(ctx.env, ctx.device_ir, requested)
            except InvalidConfig:
                continue
            return CoveragePlan(((config, True),), carrier.dependencies)
        return None

    return plan


def _widest_inner_tile(spec: ConfigSpec, carrier: Config) -> Config | None:
    """*carrier* with its last block size doubled while strict normalization
    keeps every other key; None when the tile cannot widen."""
    sizes = carrier.config.get("block_sizes")
    if not isinstance(sizes, list) or not sizes or not spec.block_sizes:
        return None
    block = spec.block_sizes[-1]
    current = sizes[-1]
    if not isinstance(current, int) or current < 1:
        return None
    generation = spec.create_config_generation()
    best: Config | None = None
    size = current * 2
    while size <= block.dim_max_size:
        candidate = deepcopy(carrier.config)
        cast("list[int]", candidate["block_sizes"])[-1] = size
        try:
            _, normalized = generation.strict_config_pair(Config.from_dict(candidate))
        except InvalidConfig:
            break
        if (
            any(
                normalized.config.get(key) != carrier.config.get(key)
                for key in set(normalized.config) | set(carrier.config)
                if key != "block_sizes"
            )
            or cast("list[int]", normalized.config["block_sizes"])[-1] != size
        ):
            break
        best = normalized
        size *= 2
    return best


# --------------------------------------------------------------------------
# Carriers


def _parent_carrier(
    *parents: tuple[str, CoverageValue],
) -> Callable[[CoverageContext], Carrier | None]:
    """The first parent group's witness with every parent's mode pinned on;
    the group inherits the first parent's dependencies plus one per parent."""

    def carrier(ctx: CoverageContext) -> Carrier | None:
        groups = [ctx.group(key) for key, _ in parents]
        if any(group is None for group in groups):
            return None
        first = cast("CompilerCoverageGroup", groups[0])
        return Carrier(
            Config.from_dict(
                deepcopy(first.witnesses[0].carrier.config) | dict(parents)
            ),
            (
                *first.dependencies,
                *(
                    CoverageDependency(
                        cast("CompilerCoverageGroup", group).mechanism, key, value
                    )
                    for group, (key, value) in zip(groups, parents, strict=True)
                ),
            ),
        )

    return carrier


def _tf32_carrier(
    transport: bool,
) -> Callable[[CoverageContext], Carrier | None]:
    """The admitted carrier riding the tf32 mode (``cute_fp32_dot``) and, with
    *transport*, every present tile-transport group's non-legacy mode.  Each
    is another group's non-legacy mode, so the carrier declares them as
    dependencies: a carrier that merely sets them is refused as "not ordinary
    legacy modes" and never reaches the population."""

    def carrier(ctx: CoverageContext) -> Carrier | None:
        tf32 = ctx.group("cute_fp32_dot")
        if tf32 is None:
            return None
        config = fragment_coverage_carrier(ctx.spec, ctx.resource_carrier)
        if config is None:
            return None
        pins: list[tuple[str, CoverageValue]] = [("cute_fp32_dot", "tf32_mma")]
        dependencies = [CoverageDependency(tf32.mechanism, "cute_fp32_dot", "tf32_mma")]
        if transport:
            for key in (
                "cute_fragment_vector_io",
                "cute_fragment_unrolled_loops",
                "cute_fragment_padded_rows",
            ):
                group = ctx.group(key)
                if group is None:
                    continue
                dependencies.append(CoverageDependency(group.mechanism, key, True))
                pins.append((key, True))
        return Carrier(
            Config.from_dict(deepcopy(config.config) | dict(pins)),
            tuple(dependencies),
            tuple(pins),
        )

    return carrier


def _snapshot_carrier(ctx: CoverageContext) -> Carrier | None:
    """A snapshot root whose while loop needs the bounded gather rides it."""
    config = fragment_coverage_carrier(ctx.spec, ctx.resource_carrier)
    if config is None:
        return None
    if not ctx.spec.cute_fragment_register_snapshot_while_root_ids:
        return Carrier(config)
    group = ctx.group("cute_fragment_bounded_gather")
    if group is None:
        return None
    return Carrier(
        Config.from_dict(deepcopy(config.config) | {group.key: True}),
        (CoverageDependency(group.mechanism, group.key, True),),
    )


def _producer_cache_resource(ctx: CoverageContext) -> Config | None:
    """A cache-only root gets its own supplemental storage estimate, counting
    all lazy tensor outputs as possible allocations like the original no-reuse
    resource catalog; every old carrier/coverage witness is preserved."""
    spec = ctx.spec
    roots = spec.cute_fragment_producer_cache_root_ids
    if roots <= (
        spec.cute_fragment_scan_root_ids | spec.cute_fragment_reduction_root_ids
    ):
        return ctx.resource_carrier
    return fragment_resource_carrier(ctx.env, ctx.device_ir, root_ids=roots)


def _has_fragment_carrier(ctx: CoverageContext) -> bool:
    """The ordinary carrier is admitted (checked ahead of the parent lookup)."""
    return fragment_coverage_carrier(ctx.spec, ctx.resource_carrier) is not None


def _consumer_fusion_admits(ctx: CoverageContext) -> bool:
    """Every old eligible kernel keeps its witness at its historical position;
    newly eligible owner/store regions append after all prior groups."""
    from ..atomic_consumer_fusion import _local_regions

    spec = ctx.spec
    host = ctx.device_ir.host_function
    assert host is not None
    with host:
        legacy = any(
            region.root in spec.cute_fragment_atomic_consumer_fusion_root_ids
            and _local_regions(region.graphs, ctx.env)
            for region in fragment_regions(ctx.device_ir)
        )
    return ctx.extended_only != bool(legacy)


def _completed_scan_nodes(ctx: CoverageContext) -> Iterable[Node]:
    from ..scan_analysis import completed_scan_plans

    for region in fragment_regions(ctx.device_ir):
        for plan in completed_scan_plans(region.root_graph):
            yield plan.scan


def _root_scan_nodes(ctx: CoverageContext) -> Iterable[Node]:
    for region in fragment_regions(ctx.device_ir):
        for node in region.root_graph.nodes:
            if node.target is scan_ops._associative_scan:
                yield node


# --------------------------------------------------------------------------
# Region predicates (the ``applies_when`` columns)


def discovery_supported(env: CompileEnvironment, region: FragmentRegion) -> bool:
    """Complete fragment ownership, retrying with live alias facts."""
    return computed_fragment_discovery_supported(env, region.graphs)


def fragment_supported(env: CompileEnvironment, region: FragmentRegion) -> bool:
    """Complete fragment ownership from the bound facts only."""
    from ..computed_fragment import computed_fragment_supported

    return computed_fragment_supported(env, region.graphs)


def _nodes(region: FragmentRegion) -> Iterable[Node]:
    for info in region.graphs:
        yield from info.graph.nodes


def _has_scan(env: CompileEnvironment, region: FragmentRegion) -> bool:
    return any(node.target is scan_ops._associative_scan for node in _nodes(region))


def _has_warp_reduction(env: CompileEnvironment, region: FragmentRegion) -> bool:
    return any(fragment_warp_reduction_supported(node) for node in _nodes(region))


def _has_private_scalar_loops(env: CompileEnvironment, region: FragmentRegion) -> bool:
    from ..private_scalar_loops import private_scalar_loop_nodes

    return bool(private_scalar_loop_nodes(region.graphs))


def _has_lane_private_load(env: CompileEnvironment, region: FragmentRegion) -> bool:
    from ..register_loads import host_load_is_readonly
    from ..register_loads import lane_private_load

    return any(
        lane_private_load(node, env)
        and host_load_is_readonly(node, env, region.graphs, allow_unbound=True)
        for node in _nodes(region)
    )


def _has_producer_cache_candidates(
    env: CompileEnvironment, region: FragmentRegion
) -> bool:
    from ..producer_cache import producer_cache_candidates

    return bool(producer_cache_candidates(region.root_graph))


def _private_int32_updates(
    env: CompileEnvironment, region: FragmentRegion, *, dead_zero: bool
) -> bool:
    """A relaxed Int32 update epoch on a private (non-host) target whose
    results are unused (or, with *dead_zero*, proved dead zero adds)."""
    from ..local_atomic import atomic_target_origins

    updates = atomic_target_origins(region.graphs)
    dead_results: Collection[Node] = ()
    if dead_zero:
        from ..dead_zero_atomics import dead_zero_atomic_results

        dead_results = dead_zero_atomic_results(region.graphs, env, allow_unbound=True)
    return any(
        target.target is not _tracing_ops._host_tensor
        and isinstance(target.meta.get("val"), torch.Tensor)
        and target.meta["val"].dtype == torch.int32
        and (not node.users or node in dead_results)
        and node.args[3] == "relaxed"
        for node, target in updates.items()
    )


def _has_private_int32_updates(env: CompileEnvironment, region: FragmentRegion) -> bool:
    return _private_int32_updates(env, region, dead_zero=False)


def _has_dead_zero_int32_updates(
    env: CompileEnvironment, region: FragmentRegion
) -> bool:
    return _private_int32_updates(env, region, dead_zero=True)


def _has_scalar_sources(env: CompileEnvironment, region: FragmentRegion) -> bool:
    from ..published_scalars import scalar_source_candidates

    return bool(scalar_source_candidates(region.graphs))


def _has_atomic_consumers(env: CompileEnvironment, region: FragmentRegion) -> bool:
    from ..atomic_consumer_fusion import atomic_consumer_regions

    return bool(atomic_consumer_regions(region.graphs, env))


def _has_integer_atomic_loops(env: CompileEnvironment, region: FragmentRegion) -> bool:
    from ..integer_atomic_epochs import integer_atomic_loop_graphs

    return bool(integer_atomic_loop_graphs(region.graphs))


def _has_half_dot(env: CompileEnvironment, region: FragmentRegion) -> bool:
    return any(fragment_half_dot_supported(node, env) for node in _nodes(region))


def _has_packet_load(env: CompileEnvironment, region: FragmentRegion) -> bool:
    """A read-only host load of a packet dtype the aligned packet walk may stage."""
    from ..packet_loads import PACKET_DTYPES
    from ..register_loads import host_load_is_readonly

    for node in _nodes(region):
        if node.target is not memory_ops.load:
            continue
        source = node.args[0]
        value = node.meta.get("val")
        if (
            isinstance(source, Node)
            and source.target is _tracing_ops._host_tensor
            and isinstance(value, torch.Tensor)
            and value.ndim
            and value.dtype in PACKET_DTYPES
            and host_load_is_readonly(node, env, region.graphs, allow_unbound=True)
        ):
            return True
    return False


def conflict_prone_row(shape: tuple[int, ...], dtype: torch.dtype) -> bool:
    """Whether a dense tile of *shape* starts every row in the same bank."""
    if len(shape) < 2:
        return False
    row_bytes = shape[-1] * dtype.itemsize
    return row_bytes >= BANK_SWEEP_BYTES and row_bytes % BANK_SWEEP_BYTES == 0


def _has_conflict_prone_tile(env: CompileEnvironment, region: FragmentRegion) -> bool:
    for node in _nodes(region):
        if node.op != "call_function":
            continue
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor) and conflict_prone_row(
            tuple(env.size_hint(size) for size in value.shape), value.dtype
        ):
            return True
    return False


def _candidate_window(env: CompileEnvironment, node: Node) -> bool:
    """A host load/store whose innermost index could form row packets: the
    emitter proves the rest (alignment, static extents, index form)."""
    from ..fragment_vector_io import packet_lanes
    from ..fragment_vector_io import static_strides

    source = node.args[0]
    if not (isinstance(source, Node) and source.target is _tracing_ops._host_tensor):
        return False
    fake = source.meta.get("val")
    if not isinstance(fake, torch.Tensor) or not fake.ndim:
        return False
    lanes = packet_lanes(fake.dtype)
    strides = static_strides(fake)
    if lanes is None or strides is None or strides[-1] != 1:
        return False
    if any(stride % lanes for stride in strides[:-1]):
        return False
    indices = [index for index in node.args[1] if index is not None]  # type: ignore[union-attr]
    if len(indices) != fake.ndim:
        return False
    last = indices[-1]
    if isinstance(last, slice):
        return last == slice(None)
    if isinstance(last, Node):
        proxy = last.meta.get("val")
        if isinstance(proxy, torch.SymInt):
            return env.resolve_block_id(proxy) is not None
        return last.target is torch.ops.prims.iota.default
    return False


def _has_packet_window(env: CompileEnvironment, region: FragmentRegion) -> bool:
    return any(
        node.target in (memory_ops.load, memory_ops.store)
        and _candidate_window(env, node)
        for node in _nodes(region)
    )


def _has_two_host_loads(env: CompileEnvironment, region: FragmentRegion) -> bool:
    return (
        sum(
            1
            for node in _nodes(region)
            if node.target is memory_ops.load
            and isinstance(node.args[0], Node)
            and node.args[0].target is _tracing_ops._host_tensor
        )
        >= 2
    )


def _has_slab_plan(env: CompileEnvironment, region: FragmentRegion) -> bool:
    from ..fragment_slab_loads import plan_slab_loads

    return bool(plan_slab_loads(env, region.graphs))


def _has_completed_scan(env: CompileEnvironment, region: FragmentRegion) -> bool:
    from ..scan_analysis import completed_scan_plans

    return bool(completed_scan_plans(region.root_graph))


def _root_has_scan(env: CompileEnvironment, region: FragmentRegion) -> bool:
    return any(
        node.target is scan_ops._associative_scan for node in region.root_graph.nodes
    )


def _root_plan_is_lazy(env: CompileEnvironment, region: FragmentRegion) -> bool:
    from ..pure_producer_regions import pure_producer_plan

    return bool(pure_producer_plan(region.root_graph, env).lazy)


# --------------------------------------------------------------------------
# Whole-kernel pre-checks


def _kernel_has_scan(env: CompileEnvironment, ir: DeviceIR) -> bool:
    return any(
        node.target is scan_ops._associative_scan
        for info in ir.graphs
        for node in info.graph.nodes
    )


def _kernel_has_warp_reduction(env: CompileEnvironment, ir: DeviceIR) -> bool:
    return any(
        fragment_warp_reduction_supported(node)
        for info in ir.graphs
        for node in info.graph.nodes
    )


def _kernel_has_producer_cache_candidates(
    env: CompileEnvironment, ir: DeviceIR
) -> bool:
    from ..producer_cache import producer_cache_candidates

    return any(producer_cache_candidates(info.graph) for info in ir.graphs)


# --------------------------------------------------------------------------
# Root projections that record extra facts


def _warp_result_roots(env: CompileEnvironment, ir: DeviceIR) -> RootFacts:
    from ..warp_results import warp_result_chain

    threads = max(cute_fragment_threads.THREADS)
    regions = fragment_regions(ir)
    roots = frozenset(
        region.root
        for region in regions
        if fragment_supported(env, region)
        and any(warp_result_chain(node, env, threads) for node in _nodes(region))
    )
    groups = [
        count
        for region in regions
        for node in _nodes(region)
        for count in warp_result_chain(node, env, threads).values()
    ]
    return RootFacts(
        roots,
        (("cute_fragment_warp_result_min_threads", min(groups, default=1) * 32),),
    )


def _local_atomic_register_roots(env: CompileEnvironment, ir: DeviceIR) -> RootFacts:
    from ..local_atomic_registers import MAX_SLOTS
    from ..local_atomic_registers import local_atomic_register_chains
    from ..warp_results import static_shape

    roots = set()
    sizes = []
    for region in fragment_regions(ir):
        if not discovery_supported(env, region):
            continue
        counts = [
            math.prod(shape)
            for node in local_atomic_register_chains(region.graphs, env)
            if (shape := static_shape(node.meta.get("val"), env)) is not None
            and 0 < math.prod(shape) <= max(cute_fragment_threads.THREADS) * MAX_SLOTS
        ]
        # Includes the existing nonescape/no-alias, uniform-loop and
        # initialization -> updates -> final-read lifetime proof.
        if counts:
            roots.add(region.root)
            sizes.extend(counts)
    return RootFacts(
        frozenset(roots),
        (
            (
                "cute_fragment_local_atomic_register_min_threads",
                (min(sizes) + MAX_SLOTS - 1) // MAX_SLOTS if sizes else 0,
            ),
        ),
    )


def _register_snapshot_roots(env: CompileEnvironment, ir: DeviceIR) -> RootFacts:
    from ..computed_fragment import computed_fragment_supported
    from ..register_snapshots import MAX_SLOTS
    from ..register_snapshots import snapshot_chains
    from ..resident_while import resident_while_plan
    from ..warp_results import static_shape

    roots = set()
    while_roots = set()
    sizes = []
    for region in fragment_regions(ir):
        chains = snapshot_chains(region.graphs, env, allow_unbound=True)
        if not chains:
            continue
        has_while = any(
            node.target is _tracing_ops._while_loop for node in _nodes(region)
        )
        bounded = (
            has_while
            and region.root
            in cute_fragment_bounded_gather.bounded_gather_roots(
                env, ir, allow_unbound=True
            )
        )
        try:
            needs_bounded_gather = has_while and any(
                not resident_while_plan(node, region.graphs).local_allocations
                for node in _nodes(region)
                if node.target is _tracing_ops._while_loop
            )
        except InvalidConfig:
            continue
        if needs_bounded_gather and not bounded:
            continue
        if not computed_fragment_supported(
            env,
            region.graphs,
            snapshot_owned=True,
            bounded_gather_owned=bounded,
            allow_unbound=has_while,
        ):
            continue
        counts = [
            math.prod(shape)
            for node in chains
            if (shape := static_shape(node.meta.get("val"), env)) is not None
            and 0 < math.prod(shape) <= max(cute_fragment_threads.THREADS) * MAX_SLOTS
        ]
        # Includes the existing nonescape/no-alias, uniform-loop and
        # initialization -> updates -> final-read lifetime proof.
        if counts:
            roots.add(region.root)
            # Match every actual emitter prerequisite. A composed epoch does
            # not cancel a pure loop's bounded-gather need.
            if needs_bounded_gather:
                while_roots.add(region.root)
            sizes.extend(counts)
    spec = env.config_spec
    return RootFacts(
        frozenset(roots),
        (
            ("cute_fragment_register_snapshot_while_root_ids", frozenset(while_roots)),
            (
                "cute_fragment_thread_root_ids",
                spec.cute_fragment_thread_root_ids | roots,
            ),
            (
                "cute_fragment_register_snapshot_min_threads",
                (min(sizes) + MAX_SLOTS - 1) // MAX_SLOTS if sizes else 0,
            ),
        ),
    )


def _warp_scan_roots(env: CompileEnvironment, ir: DeviceIR) -> RootFacts:
    """Proved roots own the knob; roots a bounded gather would complete become
    explicit requirements the dependent gather witness admits."""
    roots = set()
    dependent = set()
    for region in fragment_regions(ir):
        scans = [
            node for node in _nodes(region) if node.target is scan_ops._associative_scan
        ]
        if scans and any(
            isinstance(node.meta.get("val"), torch.Tensor)
            and node.meta["val"].dtype in WARP_SCAN_DTYPES
            for node in scans
        ):
            if discovery_supported(env, region):
                roots.add(region.root)
            else:
                dependent.add(region.root)
    if dependent:
        dependent.intersection_update(
            cute_fragment_bounded_gather.bounded_gather_roots(
                env, ir, allow_unbound=True
            )
        )
    return RootFacts(
        frozenset(roots),
        (
            (
                "cute_fragment_warp_scan_requirements",
                tuple(
                    FragmentRootRequirement(
                        root, frozenset({"cute_fragment_bounded_gather"})
                    )
                    for root in sorted(dependent)
                ),
            ),
        ),
        specialized=bool(dependent),
    )


def _topk_network_roots(env: CompileEnvironment, ir: DeviceIR) -> RootFacts:
    from ..computed_fragment import _static_selection_extent
    from ..computed_fragment import computed_fragment_supported
    from ..selection.ordered_selection import is_ordered_selection
    from ..selection.ordered_selection import ordered_selection_args
    from ..selection.ordered_selection import selection_values_only
    from ..selection.topk import match_topk_root

    roots = set()
    cta_roots = set()
    for region in fragment_regions(ir):
        # Preserve the native row owner, including its late layout and alias
        # guards. This coordinate belongs only to fragment roots.
        if (
            match_topk_root(
                region.graphs,
                noncanonical_block_ids=ir.noncanonical_task_origin_block_ids,
            )
            is not None
        ):
            continue
        if not computed_fragment_supported(env, region.graphs):
            continue
        for node in _nodes(region):
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
            if not isinstance(k, int):
                continue
            values_only = selection_values_only(node)
            if cute_fragment_topk_network.network_topk_supported(
                value.dtype, n, k, values_only=values_only
            ):
                roots.add(region.root)
                if value.dtype == torch.int64 and values_only:
                    cta_roots.add(region.root)
    return RootFacts(
        frozenset(roots), (("cute_fragment_cta_topk_root_ids", frozenset(cta_roots)),)
    )


def _register_producer_roots(env: CompileEnvironment, ir: DeviceIR) -> RootFacts:
    """Snapshot roots with a costly pure producer feeding several consumers."""
    from ..producer_cache import pure_producer
    from ..register_snapshots import snapshot_chains

    roots = set()
    for region in fragment_regions(ir):
        if region.root not in env.config_spec.cute_fragment_register_snapshots_root_ids:
            continue
        chains = snapshot_chains(region.graphs, env, allow_unbound=True)
        supported = set().union(*chains.values()) if chains else set()
        for graph in region.graphs:
            costs = {}
            for node in graph.graph.nodes:
                costs[node] = (
                    1 + sum(costs.get(arg, 0) for arg in node.all_input_nodes)
                    if pure_producer(node)
                    else 0
                )
                value = node.meta.get("val")
                if (
                    node in supported
                    and len(node.users) > 1
                    and isinstance(value, torch.Tensor)
                    and value.ndim == 1
                    and costs[node] >= 8
                ):
                    roots.add(region.root)
    return RootFacts(frozenset(roots))


def _bounded_gather_roots(env: CompileEnvironment, ir: DeviceIR) -> RootFacts:
    return RootFacts(
        cute_fragment_bounded_gather.bounded_gather_roots(env, ir, allow_unbound=True)
    )


def _fp32_dot_roots(env: CompileEnvironment, ir: DeviceIR) -> RootFacts:
    return RootFacts(fragment_fp32_dot_roots(env, ir))


# --------------------------------------------------------------------------
# The table (heuristic order is ``priority``; coverage order is the registrar
# sequence in ``heuristics.register_coverage_groups``)

_BOOLEAN = KnobCoverage()
_BOOLEAN_KEEP_FIELD = KnobCoverage(rollback=False)

SCAN = FragmentKnob(
    "cute_fragment_scan",
    "scan",
    58,
    (_has_scan, discovery_supported),
    precheck=_kernel_has_scan,
    coverage=KnobCoverage(
        mechanism="cute.computed_fragment_scan",
        domain=SCAN_MODES,
        legacy="serial",
        witness_values=("cooperative",),
        deferred=False,
        resource=False,
    ),
)
REDUCTION = FragmentKnob(
    "cute_fragment_reduction",
    "reduction",
    59,
    (_has_warp_reduction, discovery_supported),
    precheck=_kernel_has_warp_reduction,
    coverage=KnobCoverage(
        mechanism="cute.computed_fragment_reduction",
        version=3,
        domain=REDUCTION_MODES,
        legacy="serial",
        witness_values=REDUCTION_MODES[1:],
        deferred=False,
        resource=False,
    ),
)
PRIVATE_SCALAR_LOOPS = FragmentKnob(
    "cute_fragment_private_scalar_loops",
    "private_scalar_loops",
    62,
    (fragment_supported, _has_private_scalar_loops),
    root_ids_attr="cute_fragment_private_scalar_loop_root_ids",
    coverage=_BOOLEAN_KEEP_FIELD,
)
REGISTER_LOADS = FragmentKnob(
    "cute_fragment_register_loads",
    "register_loads",
    63,
    (fragment_supported, _has_lane_private_load),
    root_ids_attr="cute_fragment_register_load_root_ids",
    coverage=KnobCoverage(rollback=False, plan=_thread_coupled_plan(None)),
)
WARP_RESULTS = FragmentKnob(
    "cute_fragment_warp_results",
    "warp_results",
    64,
    roots=_warp_result_roots,
    root_ids_attr="cute_fragment_warp_result_root_ids",
    coverage=KnobCoverage(
        rollback=False,
        plan=_thread_coupled_plan("cute_fragment_warp_result_min_threads"),
    ),
)
PRODUCER_CACHE = FragmentKnob(
    "cute_fragment_producer_cache",
    "producer_cache",
    65,
    (fragment_supported, _has_producer_cache_candidates),
    precheck=_kernel_has_producer_cache_candidates,
    coverage=KnobCoverage(rollback=False, resource_override=_producer_cache_resource),
)
WARP_SCAN = FragmentKnob(
    "cute_fragment_warp_scan",
    "warp_scan",
    66,
    roots=_warp_scan_roots,
    coverage=_BOOLEAN_KEEP_FIELD,
)
TOPK_NETWORK = FragmentKnob(
    "cute_fragment_topk_network",
    "topk_network",
    67,
    roots=_topk_network_roots,
    coverage=_BOOLEAN_KEEP_FIELD,
)
# Int64 values-only selections additionally offer the CTA-wide network; its
# roots are recorded by the network projection and its witness rides the
# network group's carrier.
CTA_TOPK = FragmentKnob(
    "cute_fragment_cta_topk",
    "cta_topk",
    None,
    coverage=KnobCoverage(
        carrier=_parent_carrier(("cute_fragment_topk_network", True))
    ),
)
ATOMIC_AGGREGATION = FragmentKnob(
    "cute_fragment_atomic_aggregation",
    "atomic_aggregation",
    68,
    (_has_private_int32_updates, discovery_supported),
    coverage=_BOOLEAN_KEEP_FIELD,
)
LOCAL_ATOMIC_REGISTERS = FragmentKnob(
    "cute_fragment_local_atomic_registers",
    "local_atomic_registers",
    69,
    roots=_local_atomic_register_roots,
    coverage=KnobCoverage(
        rollback=False,
        plan=_min_threads_plan("cute_fragment_local_atomic_register_min_threads"),
    ),
)
REGISTER_SNAPSHOTS = FragmentKnob(
    "cute_fragment_register_snapshots",
    "register_snapshots",
    70,
    roots=_register_snapshot_roots,
    coverage=KnobCoverage(
        rollback=False,
        carrier=_snapshot_carrier,
        plan=_min_threads_plan("cute_fragment_register_snapshot_min_threads"),
    ),
)
PUBLISHED_SCALARS = FragmentKnob(
    "cute_fragment_published_scalars",
    "published_scalars",
    71,
    (fragment_supported, _has_scalar_sources),
    root_ids_attr="cute_fragment_published_scalar_root_ids",
    coverage=_BOOLEAN_KEEP_FIELD,
)
SKIP_ZERO_ATOMICS = FragmentKnob(
    "cute_fragment_skip_zero_atomics",
    "skip_zero_atomics",
    72,
    (_has_dead_zero_int32_updates, discovery_supported),
    coverage=_BOOLEAN_KEEP_FIELD,
)
ATOMIC_CONSUMER_FUSION = FragmentKnob(
    "cute_fragment_atomic_consumer_fusion",
    "atomic_consumer_fusion",
    73,
    (fragment_supported, _has_atomic_consumers),
    coverage=KnobCoverage(admit=_consumer_fusion_admits),
)
# The lane-private load proof removes shared input staging: only the new
# deferred witness couples to that storage choice; old seeds stay as-is.
INTEGER_ATOMIC_EPOCHS = FragmentKnob(
    "cute_fragment_integer_atomic_epochs",
    "integer_atomic_epochs",
    75,
    (_has_integer_atomic_loops, fragment_supported),
    coverage=KnobCoverage(
        admit=_has_fragment_carrier,
        carrier=_parent_carrier(("cute_fragment_register_loads", True)),
        plan=_stripped_plan,
    ),
)
PACKET_LOADS = FragmentKnob(
    "cute_fragment_packet_loads",
    "packet_loads",
    76,
    (fragment_supported, _has_packet_load),
    root_ids_attr="cute_fragment_packet_load_root_ids",
)
PADDED_ROWS = FragmentKnob(
    "cute_fragment_padded_rows",
    "padded_rows",
    77,
    (discovery_supported, _has_conflict_prone_tile),
)
UNROLLED_LOOPS = FragmentKnob(
    "cute_fragment_unrolled_loops",
    "unrolled_loops",
    78,
    (discovery_supported,),
)
VECTOR_IO = FragmentKnob(
    "cute_fragment_vector_io",
    "vector_io",
    79,
    (discovery_supported, _has_packet_window),
)
BATCHED_LOADS = FragmentKnob(
    "cute_fragment_batched_loads",
    "batched_loads",
    80,
    (discovery_supported, _has_two_host_loads),
)
SPLIT_K = FragmentKnob(
    "cute_fragment_split_k",
    "split_k",
    81,
    roots=_fp32_dot_roots,
    coverage=KnobCoverage(carrier=_tf32_carrier(False), plan=_pinned_plan(False)),
)
MMA_ROWS = FragmentKnob(
    "cute_fragment_mma_rows",
    "mma_rows",
    82,
    roots=_fp32_dot_roots,
    coverage=KnobCoverage(carrier=_tf32_carrier(True), plan=_pinned_plan(True)),
)
SLAB_LOADS = FragmentKnob(
    "cute_fragment_slab_loads",
    "slab_loads",
    83,
    (discovery_supported, _has_slab_plan),
)
REGISTER_PRODUCERS = FragmentKnob(
    "cute_fragment_register_producers",
    "register_producers",
    84,
    roots=_register_producer_roots,
    root_ids_attr="cute_fragment_register_producer_root_ids",
    coverage=KnobCoverage(
        carrier=_parent_carrier(("cute_fragment_register_snapshots", True))
    ),
)
SCAN_EXPORTS = FragmentKnob(
    "cute_fragment_scan_exports",
    "scan_exports",
    85,
    (_has_completed_scan,),
    coverage=KnobCoverage(
        carrier=_parent_carrier(("cute_fragment_warp_scan", True)),
        plan=_candidate_plan(_completed_scan_nodes, validate_scan_exports),
    ),
)
WARP_PRODUCER_REGIONS = FragmentKnob(
    "cute_fragment_warp_producer_regions",
    "warp_producer_regions",
    86,
    (_root_has_scan,),
    coverage=KnobCoverage(
        carrier=_parent_carrier(
            ("cute_fragment_warp_scan", True), ("cute_fragment_reduction", "warp")
        ),
        plan=_candidate_plan(_root_scan_nodes, validate_warp_producer_regions),
    ),
)
BOUNDED_GATHER = FragmentKnob(
    "cute_fragment_bounded_gather",
    "bounded_gather",
    87,
    roots=_bounded_gather_roots,
)
PURE_PRODUCER_REGIONS = FragmentKnob(
    "cute_fragment_pure_producer_regions",
    "pure_producer_regions",
    88,
    (fragment_supported, _root_plan_is_lazy),
)

HALF_MMA = FragmentKnob(
    "cute_fragment_half_mma",
    "half_mma",
    90,
    (_has_half_dot, discovery_supported),
)

FRAGMENT_KNOBS: tuple[FragmentKnob, ...] = (
    SCAN,
    REDUCTION,
    PRIVATE_SCALAR_LOOPS,
    REGISTER_LOADS,
    WARP_RESULTS,
    PRODUCER_CACHE,
    WARP_SCAN,
    TOPK_NETWORK,
    CTA_TOPK,
    ATOMIC_AGGREGATION,
    LOCAL_ATOMIC_REGISTERS,
    REGISTER_SNAPSHOTS,
    PUBLISHED_SCALARS,
    SKIP_ZERO_ATOMICS,
    ATOMIC_CONSUMER_FUSION,
    INTEGER_ATOMIC_EPOCHS,
    PACKET_LOADS,
    PADDED_ROWS,
    UNROLLED_LOOPS,
    VECTOR_IO,
    BATCHED_LOADS,
    SPLIT_K,
    MMA_ROWS,
    SLAB_LOADS,
    REGISTER_PRODUCERS,
    SCAN_EXPORTS,
    WARP_PRODUCER_REGIONS,
    BOUNDED_GATHER,
    PURE_PRODUCER_REGIONS,
    HALF_MMA,
)

# ``config_keys.CUTE_FRAGMENT_KEYS`` is the key table the search surface
# derives from; it cannot import the rows (it loads while ``config_spec``
# initializes), so the rows check it here.
_UNLISTED_KNOBS = sorted(
    {knob.name for knob in FRAGMENT_KNOBS}.difference(CUTE_FRAGMENT_KEYS)
)
if _UNLISTED_KNOBS:
    raise RuntimeError(
        f"fragment knobs missing from config_keys.CUTE_FRAGMENT_KEYS: {_UNLISTED_KNOBS}"
    )

# --------------------------------------------------------------------------
# The generated heuristic classes and coverage registrars (static names:
# ``heuristics/__init__`` imports and re-exports them; tests patch them there)

CuteFragmentScanHeuristic = fragment_knob_heuristic(SCAN)
CuteFragmentReductionHeuristic = fragment_knob_heuristic(REDUCTION)
CuteFragmentPrivateScalarLoopsHeuristic = fragment_knob_heuristic(PRIVATE_SCALAR_LOOPS)
CuteFragmentRegisterLoadsHeuristic = fragment_knob_heuristic(REGISTER_LOADS)
CuteFragmentWarpResultsHeuristic = fragment_knob_heuristic(WARP_RESULTS)
CuteFragmentProducerCacheHeuristic = fragment_knob_heuristic(PRODUCER_CACHE)
CuteFragmentWarpScanHeuristic = fragment_knob_heuristic(WARP_SCAN)
CuteFragmentTopKNetworkHeuristic = fragment_knob_heuristic(TOPK_NETWORK)
CuteFragmentAtomicAggregationHeuristic = fragment_knob_heuristic(ATOMIC_AGGREGATION)
CuteFragmentLocalAtomicRegistersHeuristic = fragment_knob_heuristic(
    LOCAL_ATOMIC_REGISTERS
)
CuteFragmentRegisterSnapshotsHeuristic = fragment_knob_heuristic(REGISTER_SNAPSHOTS)
CuteFragmentPublishedScalarsHeuristic = fragment_knob_heuristic(PUBLISHED_SCALARS)
CuteFragmentSkipZeroAtomicsHeuristic = fragment_knob_heuristic(SKIP_ZERO_ATOMICS)
CuteFragmentAtomicConsumerFusionHeuristic = fragment_knob_heuristic(
    ATOMIC_CONSUMER_FUSION
)
CuteFragmentIntegerAtomicEpochsHeuristic = fragment_knob_heuristic(
    INTEGER_ATOMIC_EPOCHS
)
CuteFragmentPacketLoadsHeuristic = fragment_knob_heuristic(PACKET_LOADS)
CuteFragmentPaddedRowsHeuristic = fragment_knob_heuristic(PADDED_ROWS)
CuteFragmentUnrolledLoopsHeuristic = fragment_knob_heuristic(UNROLLED_LOOPS)
CuteFragmentVectorIoHeuristic = fragment_knob_heuristic(VECTOR_IO)
CuteFragmentBatchedLoadsHeuristic = fragment_knob_heuristic(BATCHED_LOADS)
CuteFragmentSplitKHeuristic = fragment_knob_heuristic(SPLIT_K)
CuteFragmentMmaRowsHeuristic = fragment_knob_heuristic(MMA_ROWS)
CuteFragmentSlabLoadsHeuristic = fragment_knob_heuristic(SLAB_LOADS)
CuteFragmentRegisterProducersHeuristic = fragment_knob_heuristic(REGISTER_PRODUCERS)
CuteFragmentScanExportsHeuristic = fragment_knob_heuristic(SCAN_EXPORTS)
CuteFragmentWarpProducerRegionsHeuristic = fragment_knob_heuristic(
    WARP_PRODUCER_REGIONS
)
CuteFragmentBoundedGatherHeuristic = fragment_knob_heuristic(BOUNDED_GATHER)
CuteFragmentPureProducerRegionsHeuristic = fragment_knob_heuristic(
    PURE_PRODUCER_REGIONS
)

CuteFragmentHalfMmaHeuristic = fragment_knob_heuristic(HALF_MMA)

register_fragment_scan_coverage = fragment_knob_registrar(SCAN)
register_fragment_reduction_coverage = fragment_knob_registrar(REDUCTION)
register_fragment_private_scalar_loops_coverage = fragment_knob_registrar(
    PRIVATE_SCALAR_LOOPS
)
register_fragment_register_loads_coverage = fragment_knob_registrar(REGISTER_LOADS)
register_fragment_warp_results_coverage = fragment_knob_registrar(WARP_RESULTS)
register_fragment_producer_cache_coverage = fragment_knob_registrar(PRODUCER_CACHE)
register_fragment_warp_scan_coverage = fragment_knob_registrar(WARP_SCAN)
register_fragment_topk_network_coverage = fragment_knob_registrar(TOPK_NETWORK)
register_fragment_cta_topk_coverage = fragment_knob_registrar(CTA_TOPK)
register_fragment_atomic_aggregation_coverage = fragment_knob_registrar(
    ATOMIC_AGGREGATION
)
register_fragment_local_atomic_registers_coverage = fragment_knob_registrar(
    LOCAL_ATOMIC_REGISTERS
)
register_fragment_register_snapshots_coverage = fragment_knob_registrar(
    REGISTER_SNAPSHOTS
)
register_fragment_published_scalars_coverage = fragment_knob_registrar(
    PUBLISHED_SCALARS
)
register_fragment_skip_zero_atomics_coverage = fragment_knob_registrar(
    SKIP_ZERO_ATOMICS
)
register_fragment_atomic_consumer_fusion_coverage = fragment_knob_registrar(
    ATOMIC_CONSUMER_FUSION
)
register_fragment_integer_atomic_epochs_coverage = fragment_knob_registrar(
    INTEGER_ATOMIC_EPOCHS
)
register_fragment_packet_loads_coverage = fragment_knob_registrar(PACKET_LOADS)
register_fragment_padded_rows_coverage = fragment_knob_registrar(PADDED_ROWS)
register_fragment_unrolled_loops_coverage = fragment_knob_registrar(UNROLLED_LOOPS)
register_fragment_vector_io_coverage = fragment_knob_registrar(VECTOR_IO)
register_fragment_batched_loads_coverage = fragment_knob_registrar(BATCHED_LOADS)
register_fragment_split_k_coverage = fragment_knob_registrar(SPLIT_K)
register_fragment_mma_rows_coverage = fragment_knob_registrar(MMA_ROWS)
register_fragment_slab_loads_coverage = fragment_knob_registrar(SLAB_LOADS)
register_fragment_register_producers_coverage = fragment_knob_registrar(
    REGISTER_PRODUCERS
)
register_fragment_scan_exports_coverage = fragment_knob_registrar(SCAN_EXPORTS)
register_fragment_warp_producer_regions_coverage = fragment_knob_registrar(
    WARP_PRODUCER_REGIONS
)
register_fragment_bounded_gather_coverage = fragment_knob_registrar(BOUNDED_GATHER)
register_fragment_pure_producer_regions_coverage = fragment_knob_registrar(
    PURE_PRODUCER_REGIONS
)
register_fragment_half_mma_coverage = fragment_knob_registrar(HALF_MMA)
