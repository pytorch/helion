"""Bounded compiler-derived proposals for CuTe LFBO and hybrid searches."""

from __future__ import annotations

import contextlib
import contextvars
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

from .. import exc
from .._compiler.compile_environment import CompileEnvironment
from ..runtime.config import Config
from .config_generation import ConfigGeneration

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..runtime.kernel import BoundKernel
    from .base_search import _AutotunableKernel
    from .config_spec import ConfigSpec

MAX_COMPILER_SEEDS = 83
MAX_ACC1_WITNESSES = 12
MAX_RADIX_WITNESSES = 6
Method = Literal["lfbo", "llm_lfbo", "llm"]


def radix_witness_configs(spec: ConfigSpec) -> list[Config]:
    """Derive resident-input packets/CTAs from the proved row and register budget."""
    from .._compiler.cute.radix_selection import radix_schedule
    from .._compiler.cute.radix_selection import validate_radix_schedule
    from .._compiler.cute.radix_selection_codegen import (
        has_exact_cached_payload_consumer,
    )
    from .._compiler.cute.radix_selection_config import GEOMETRY_CHOICES
    from .._compiler.cute.radix_selection_config import PIPELINE_CHOICES

    graph = spec.cute_radix_selection_graph
    if graph is None or not graph.largest:
        return []
    proposals = []
    for packet_bits in sorted(PIPELINE_CHOICES["cute_radix_packet_bits"]):
        packet = max(1, packet_bits // (graph.x.element_size() * 8))
        threads = next(
            (
                size
                for size in sorted(GEOMETRY_CHOICES["cute_radix_threads"])
                if size * packet >= graph.cols
            ),
            max(GEOMETRY_CHOICES["cute_radix_threads"]),
        )
        for inverse in (False, True):
            if inverse and not has_exact_cached_payload_consumer(graph):
                continue
            config = dict(spec.autotune_reference_config())
            config.update(
                cute_selection_algorithm="radix",
                cute_radix_threads=threads,
                cute_radix_cluster_ctas=1,
                cute_radix_candidate_capacity=0,
                cute_radix_histogram_mode="direct",
                cute_radix_packet_bits=packet_bits,
                cute_radix_candidate_key_cache=False,
                cute_radix_scalar_candidates=False,
                cute_radix_valid_domain=True,
                cute_radix_ticket_stores=True,
                cute_radix_cached_payload_inverse=inverse,
                cute_radix_resident_input=True,
                cute_radix_boundary_early_exit=graph.k < graph.logical_cols,
                cute_radix_defer_output=True,
                cute_radix_partial_scan=threads > 256,
            )
            plan = radix_schedule(graph, config)
            if plan.resident_elements * (1 if plan.resident_keys_only else 2) > 128:
                continue
            try:
                validate_radix_schedule(spec, config)
                spec.normalize(config)
            except exc.InvalidConfig:
                continue
            proposals.append(Config.from_dict(config))
    assert len(proposals) <= MAX_RADIX_WITNESSES
    return list(dict.fromkeys(proposals))


def register_cute_witnesses(
    bound: BoundKernel, *, method: Method, enabled: bool = True
) -> dict[str, Any]:
    """Register validated proposals after promotion, retaining every original seed."""
    if method not in ("lfbo", "llm_lfbo", "llm"):
        raise ValueError(f"Unsupported witness registration method: {method}")
    spec = bound.config_spec
    original = list(spec.compiler_seed_configs)
    record: dict[str, Any] = {
        "kernel_name": bound.kernel.name,
        "method": method,
        "enabled": enabled,
        "original_compiler_seeds": len(original),
        "added_acc1_witnesses": 0,
        "added_radix_witnesses": 0,
        "saved_config_seeds": False,
        "pure_llm_seed_injection": False,
        "promoted_default_changed": False,
    }
    if not enabled or method == "llm" or spec.backend.name != "cute":
        return record
    allowance = max(0, MAX_COMPILER_SEEDS - len(original))
    default_before = spec.compiler_default_config
    current = CompileEnvironment.current() if CompileEnvironment.has_current() else None
    if current is not None and current is not bound.env:
        raise RuntimeError("Witness registration entered another compile environment")
    with contextlib.nullcontext() if current is not None else bound.env:
        generation = ConfigGeneration(spec)
        seen = {config for _flat, config in generation.seed_flat_config_pairs()}
        acc1 = spec._cute_tcgen05_config._with_paired_acc1_witnesses(
            original, enabled=True
        )
        proposals = [("acc1", config) for config in acc1 if config not in original]
        proposals.extend(("radix", config) for config in radix_witness_configs(spec))
        additional = []
        counts = {"acc1": 0, "radix": 0}
        for family, raw in proposals:
            _flat, config = generation.canonicalize_flat(generation.flatten(raw))
            if config in seen:
                continue
            if family == "acc1" and (
                config.get("tcgen05_cta_group") != "two"
                or config.get("tcgen05_acc_stages") != 1
            ):
                raise RuntimeError("ACC1 witness lost its proved paired allocation")
            if family == "radix" and (
                config.get("cute_selection_algorithm") != "radix"
                or config.get("cute_radix_resident_input") is not True
            ):
                raise RuntimeError("Radix witness lost its proved resident input")
            seen.add(config)
            if len(additional) < allowance:
                additional.append(config)
                counts[family] += 1
    if counts["acc1"] > MAX_ACC1_WITNESSES or counts["radix"] > MAX_RADIX_WITNESSES:
        raise RuntimeError("Witness proposals exceed the bounded family allowance")
    spec.compiler_seed_configs = [*additional, *original]
    if spec.compiler_default_config != default_before:
        raise RuntimeError("Witness registration changed the promoted default")
    record.update(
        added_acc1_witnesses=counts["acc1"],
        added_radix_witnesses=counts["radix"],
        total_compiler_seeds=len(spec.compiler_seed_configs),
        minimum_random_slots_for_population_100=max(
            0, 99 - len(spec.compiler_seed_configs)
        ),
        witnesses=[dict(config) for config in additional],
    )
    return record


_ACTIVE_WITNESSES: contextvars.ContextVar[list[dict[str, Any]] | None] = (
    contextvars.ContextVar("helion_cute_witness_seeds", default=None)
)


@contextlib.contextmanager
def cute_witness_seeds() -> Iterator[list[dict[str, Any]]]:
    """Enable bounded proposals for newly constructed LFBO/hybrid searches.

    Proposals use compiler facts, legal coordinates and resource validation.
    Pure LLM receives no injected seeds. CUDA graph replay during timing uses
    Helion's existing HELION_BENCHMARK_CUDAGRAPH option.
    Registration preserves the promoted default and the original compiler seeds.
    """
    records: list[dict[str, Any]] = []
    token = _ACTIVE_WITNESSES.set(records)
    try:
        yield records
    finally:
        _ACTIVE_WITNESSES.reset(token)


def register_active_cute_witnesses(
    bound: _AutotunableKernel, *, method: Method
) -> None:
    from ..runtime.kernel import BoundKernel

    records = _ACTIVE_WITNESSES.get()
    if records is not None and isinstance(bound, BoundKernel):
        records.append(register_cute_witnesses(bound, method=method))
