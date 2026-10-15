"""Additive coverage for a proved unordered radix selection region."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from ...autotuner.compiler_coverage import CompilerCoverageGroup
from ...autotuner.compiler_coverage import CoverageDependency
from ...autotuner.compiler_coverage import CoverageWitness
from ...exc import InvalidConfig
from ...runtime.config import Config
from ..cute.memory_ops import register_cute_tensor_alias_specializations
from ..cute.memory_ops import register_persistent_vec_alignment_specializations
from ..cute.ordered_selection import unordered_topk_args
from ..cute.radix_selection_config import ALGORITHM_KEY
from ..cute.radix_selection_config import ALGORITHMS
from ..cute.radix_selection_config import BOOLEAN_KEYS
from ..cute.radix_selection_config import GEOMETRY_CHOICES
from ..cute.radix_selection_config import HISTOGRAM_KEY
from ..cute.radix_selection_config import HISTOGRAM_MODES
from ..cute.radix_selection_config import PIPELINE_CHOICES
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact


class CuteRadixSelectionHeuristic(AutotunerHeuristic):
    name = "cute_radix_selection"
    backend = "cute"

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        from ..cute.radix_selection import match_radix_selection_root
        from ..cute.radix_selection import register_radix_packet_alignment
        from ..cute.radix_selection import tensor_value
        from ..cute.radix_selection import validate_radix_schedule

        host = device_ir.host_function
        if host is None:
            return frozenset()
        possible = any(
            (arguments := unordered_topk_args(node)) is not None
            and (value := tensor_value(arguments[0])) is not None
            and value.dtype == torch.int32
            for info in device_ir.graphs
            for node in info.graph.nodes
        )
        if not possible:
            return frozenset()
        with host:
            graph = match_radix_selection_root(device_ir.graphs, check_storage=False)
            if graph is None:
                return frozenset({"input_tensor_metadata"})
            register_cute_tensor_alias_specializations(env)
            register_persistent_vec_alignment_specializations(env)
            graph = match_radix_selection_root(device_ir.graphs, allow_unbound=True)
            if graph is not None:
                register_radix_packet_alignment(env, graph)
        env.config_spec.cute_radix_selection_graph = graph
        if graph is not None:
            try:
                validate_radix_schedule(env.config_spec, {})
            except InvalidConfig:
                env.config_spec.cute_radix_selection_graph = None
        return frozenset({"input_tensor_metadata"})

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        # Modes/geometry enter through additive coverage, never old seed rows.
        return False


def register_radix_selection_coverage(
    env: CompileEnvironment, device_ir: DeviceIR
) -> None:
    spec = env.config_spec
    if spec.cute_radix_selection_graph is None:
        return
    generation = spec.create_config_generation()
    _, default = generation.canonicalize_flat(generation.default_flat())
    generation.strict_config_pair(default)
    mechanism = "cute.radix_selection"
    algorithm_dependency = CoverageDependency(mechanism, ALGORITHM_KEY, "radix")
    domains = ((ALGORITHM_KEY, ALGORITHMS), *GEOMETRY_CHOICES.items())
    cluster = (
        4
        if spec.target_device_capability is not None
        and spec.target_device_capability >= (9, 0)
        and spec.cute_radix_selection_graph.rows * 4 <= (1 << 31) - 1
        else 1
    )
    candidates = 0
    for capacity in (4096, 2048, 1024):
        try:
            generation.strict_config_pair(
                Config.from_dict(
                    deepcopy(default.config)
                    | {
                        ALGORITHM_KEY: "radix",
                        "cute_radix_threads": 1024,
                        "cute_radix_cluster_ctas": cluster,
                        "cute_radix_candidate_capacity": capacity,
                    }
                )
            )
        except InvalidConfig:
            continue
        candidates = capacity
        break
    witness_values = ("radix", 1024, cluster, candidates)
    carrier = deepcopy(default.config)
    dependencies: list[CoverageDependency] = []
    for (key, domain), value in zip(domains, witness_values, strict=True):
        current = Config.from_dict(deepcopy(carrier))
        requested = Config.from_dict(deepcopy(carrier) | {key: value})
        generation.strict_config_pair(current)
        generation.strict_config_pair(requested)
        group_mechanism = mechanism if key == ALGORITHM_KEY else f"cute.{key[5:]}"
        spec.register_compiler_coverage_group(
            CompilerCoverageGroup(
                mechanism=group_mechanism,
                version=1,
                key=key,
                domain=domain,
                legacy=domain[0],
                witnesses=(CoverageWitness(current, value),),
                dependencies=tuple(dependencies),
                deferred=True,
            )
        )
        carrier[key] = value
        dependencies.append(
            algorithm_dependency
            if key == ALGORITHM_KEY
            else CoverageDependency(group_mechanism, key, value)
        )
    current = Config.from_dict(deepcopy(carrier))
    generation.strict_config_pair(current)
    generation.strict_config_pair(
        Config.from_dict(deepcopy(carrier) | {HISTOGRAM_KEY: "match"})
    )
    spec.register_compiler_coverage_group(
        CompilerCoverageGroup(
            mechanism="cute.radix_histogram_mode",
            version=1,
            key=HISTOGRAM_KEY,
            domain=HISTOGRAM_MODES,
            legacy="direct",
            witnesses=(CoverageWitness(current, "match"),),
            dependencies=tuple(dependencies),
            deferred=True,
        )
    )
    # Keep every new coordinate deferred and independently owned. These
    # witnesses neither alter the existing seed rows nor consume random draws
    # from the legacy initial population.
    for key in BOOLEAN_KEYS:
        if key == "cute_radix_cached_payload_inverse":
            from ..cute.radix_selection_codegen import has_exact_cached_payload_consumer

            eligible = (
                spec.cute_radix_selection_graph.largest
                and has_exact_cached_payload_consumer(spec.cute_radix_selection_graph)
            )
            required = (
                "cute_radix_candidate_key_cache",
                "cute_radix_valid_domain",
                "cute_radix_ticket_stores",
            )
            inverse_carrier = deepcopy(carrier) | dict.fromkeys(required, True)
            current = Config.from_dict(inverse_carrier)
            requested = Config.from_dict(deepcopy(inverse_carrier) | {key: True})
            witnesses = ()
            inverse_dependencies = tuple(dependencies)
            if eligible:
                witnesses = (CoverageWitness(current, True),)
                inverse_dependencies += tuple(
                    CoverageDependency(f"cute.{field[5:]}", field, True)
                    for field in required
                )
                try:
                    generation.strict_config_pair(current)
                    generation.strict_config_pair(requested)
                except InvalidConfig:
                    witnesses = ()
            spec.register_compiler_coverage_group(
                CompilerCoverageGroup(
                    mechanism=f"cute.{key[5:]}",
                    version=1,
                    key=key,
                    domain=(False, True),
                    legacy=False,
                    witnesses=witnesses,
                    dependencies=inverse_dependencies,
                    deferred=True,
                )
            )
            continue
        if (
            key == "cute_radix_valid_domain"
            and not spec.cute_radix_selection_graph.largest
        ):
            continue
        current = Config.from_dict(deepcopy(carrier))
        requested = Config.from_dict(deepcopy(carrier) | {key: True})
        witnesses = (CoverageWitness(current, True),)
        try:
            generation.strict_config_pair(current)
            generation.strict_config_pair(requested)
        except InvalidConfig:
            # A typed cache can exceed the target's static resource budget.
            # Manual smaller-capacity configurations remain independently legal.
            witnesses = ()
        spec.register_compiler_coverage_group(
            CompilerCoverageGroup(
                mechanism=f"cute.{key[5:]}",
                version=1,
                key=key,
                domain=(False, True),
                legacy=False,
                witnesses=witnesses,
                dependencies=tuple(dependencies),
                deferred=True,
            )
        )
    for key, domain in PIPELINE_CHOICES.items():
        current = Config.from_dict(deepcopy(carrier))
        pipeline_witnesses = tuple(
            CoverageWitness(current, value) for value in domain[1:]
        )
        for value in domain[1:]:
            generation.strict_config_pair(
                Config.from_dict(deepcopy(carrier) | {key: value})
            )
        spec.register_compiler_coverage_group(
            CompilerCoverageGroup(
                mechanism=f"cute.{key[5:]}",
                version=1,
                key=key,
                domain=domain,
                legacy=domain[0],
                witnesses=pipeline_witnesses,
                dependencies=tuple(dependencies),
                deferred=True,
            )
        )
