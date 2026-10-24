"""Runtime tensor-storage proofs and their dispatch-cache guards."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch._dynamo.source import LocalSource
from torch._subclasses import FakeTensor

from .compile_environment import RuntimeInputSpecialization
from .compile_environment import _replay_tensor_input_source

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Hashable
    from collections.abc import Sequence

    from torch._guards import Source

    from .compile_environment import CompileEnvironment


def _prepared_guard_source(
    namespace: dict[str, object],
    name: str,
    extract: Callable[[Sequence[object]], Hashable],
) -> str:
    """The expression a generated prepared guard reads a source through:
    ``args[i]`` for a plain argument (the extractor carries its index), else
    a call of the extractor bound into ``namespace``."""
    index = getattr(extract, "_helion_arg_index", None)
    if type(index) is int:
        return f"args[{index}]"
    namespace[name] = extract
    return f"{name}(args)"


def _tensor_storage_disjoint_matrix_signature(
    values: Sequence[object],
) -> Hashable:
    """Classify every pair while reading each tensor's storage only once."""
    spans: list[tuple[torch.device, int, int] | None] = []
    for value in values:
        if not isinstance(value, torch.Tensor) or isinstance(value, FakeTensor):
            spans.append(None)
            continue
        storage = value.untyped_storage()
        start = int(storage.data_ptr())
        spans.append((value.device, start, start + storage.nbytes()))

    result: list[bool] = []
    for left, right in itertools.combinations(spans, 2):
        if left is None or right is None:
            result.append(False)
        elif left[1] == left[2] or right[1] == right[2] or left[0] != right[0]:
            result.append(True)
        else:
            result.append(left[2] <= right[1] or right[2] <= left[1])
    return tuple(result)


def _tensor_storage_disjoint_prepared_guard(
    source_extractors: Sequence[Callable[[Sequence[object]], Hashable]],
    expected: Hashable,
) -> Callable[[Sequence[object]], bool]:
    """Prepared-call predicate for ``_tensor_storage_disjoint_matrix_signature``.

    Source types and devices are pinned by the prepared argument guard, so a
    pair's entry can only change through the storage spans themselves.  Each
    span is read once and every pair is re-evaluated against its pinned
    entry.  A pair the classifier proved disjoint through differing devices
    may be rejected here (the spans are compared regardless), which only
    costs a fallback to the full dispatch path.
    """
    namespace: dict[str, object] = {}
    lines = ["def guard(args):"]
    for index, extract in enumerate(source_extractors):
        source = _prepared_guard_source(namespace, f"extract_{index}", extract)
        lines.extend(
            (
                f"    storage_{index} = {source}.untyped_storage()",
                f"    start_{index} = storage_{index}.data_ptr()",
                f"    end_{index} = start_{index} + storage_{index}.nbytes()",
            )
        )
    conditions: list[str] = []
    for (left, right), disjoint in zip(
        itertools.combinations(range(len(source_extractors)), 2),
        cast("tuple[bool, ...]", expected),
        strict=True,
    ):
        predicate = (
            f"(start_{left} == end_{left} or start_{right} == end_{right}"
            f" or end_{left} <= start_{right} or end_{right} <= start_{left})"
        )
        conditions.append(predicate if disjoint else f"not {predicate}")
    lines.append(f"    return {' and '.join(conditions) or 'True'}")
    exec("\n".join(lines), namespace)
    return cast("Callable[[Sequence[object]], bool]", namespace["guard"])


def _tensor_alias_sources(env: CompileEnvironment) -> tuple[Source, ...]:
    sources: list[Source] = []
    # Enumerate direct host tensor arguments by their argument names rather
    # than only asking FakeTensor -> Source.  FakeTensorConverter deliberately
    # reuses one FakeTensor when the same real tensor is passed in two argument
    # slots.  That makes the reverse mapping ambiguous, but the two explicit
    # argument Sources remain distinct and must both participate in the
    # cache-specialized alias matrix.
    from .host_function import HostFunction

    for name, value in HostFunction.current().params.arguments.items():
        if isinstance(value, torch.Tensor):
            sources.append(LocalSource(name, is_input=True))
    for tensor in env.input_sources:
        source = env.tensor_input_source(tensor)
        if source is not None and source not in sources:
            sources.append(source)
    return tuple(sources)


_TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY = "tensor_storage_disjoint_matrix_v1"


def register_tensor_alias_specializations(env: CompileEnvironment) -> None:
    """Cache-key runtime alias facts used by memory-reordering passes.

    Different kernel argument names are not a non-aliasing proof: callers may
    pass the same tensor or overlapping views.  Recording the storage-overlap
    predicate in the bound-kernel key lets codegen use a positive runtime fact
    without reusing that code for a later aliasing launch.
    """
    sources = _tensor_alias_sources(env)
    if len(sources) < 2:
        return
    env.register_runtime_input_specialization(
        _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY,
        RuntimeInputSpecialization(
            sources=sources,
            classifier_identity=(
                "storage_span_disjoint_matrix_v1",
                tuple(map(repr, sources)),
            ),
            classifier=_tensor_storage_disjoint_matrix_signature,
            reusable_tensor_properties=frozenset(("storage_span",)),
            prepared_guard_factory=_tensor_storage_disjoint_prepared_guard,
        ),
    )


def runtime_tensors_have_disjoint_storage(
    env: CompileEnvironment,
    left: torch.Tensor,
    right: torch.Tensor,
    recordable: Sequence[Source],
) -> bool:
    """Inspect live storage without registering a cache guard.

    Reuse ``recordable`` only within one read-only candidate proof.
    A transform using this provisional fact must register the alias
    specialization before committing its rewrite.
    """
    sources: list[Source] = []
    for tensor in (left, right):
        source = env.tensor_storage_input_source(tensor)
        if source is None or source not in recordable or source in sources:
            return False
        sources.append(source)
    values = tuple(
        _replay_tensor_input_source(source, env.runtime_arg_values_by_name)
        for source in sources
    )
    return _tensor_storage_disjoint_matrix_signature(values) == (True,)


def runtime_tensors_are_proven_disjoint(
    env: CompileEnvironment,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    allow_unbound: bool = False,
) -> bool:
    """Return a cache-specialized positive runtime storage-disjointness fact."""
    left_source = env.tensor_input_source(left)
    right_source = env.tensor_input_source(right)
    if left_source is None:
        left_source = env.tensor_storage_input_source(left)
    if right_source is None:
        right_source = env.tensor_storage_input_source(right)
    if left_source is None or right_source is None:
        return False
    return runtime_tensor_sources_are_proven_disjoint(
        env,
        left_source,
        right_source,
        allow_unbound=allow_unbound,
    )


def runtime_tensor_sources_are_proven_disjoint(
    env: CompileEnvironment,
    left_source: Source,
    right_source: Source,
    *,
    allow_unbound: bool = False,
) -> bool:
    """Return a cache-specialized storage fact for explicit input Sources.

    Search registration runs during binding, before its immutable facts are
    recorded. It may opt into live facts from the registered classifier.
    Codegen must retain the default requirement for a bound fact.
    """
    if left_source == right_source:
        return False
    specialization = env.runtime_input_specializations.get(
        _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
    )
    sources = _tensor_alias_sources(env)
    if (
        specialization is None
        or specialization.sources != sources
        or specialization.classifier_identity
        != ("storage_span_disjoint_matrix_v1", tuple(map(repr, sources)))
        or specialization.reusable_tensor_properties != frozenset(("storage_span",))
    ):
        return False
    # This immutable matrix was recorded from the bound kernel's dispatch key.
    # Recompilation may outlive its construction tensors, which are held only
    # weakly. Missing live arguments do not invalidate a cache-specialized fact.
    facts = env.bound_runtime_input_specialization_results.get(
        _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
    )
    runtime_values = tuple(
        _replay_tensor_input_source(source, env.runtime_arg_values_by_name)
        for source in sources
    )
    live_facts = specialization.classifier(runtime_values)
    assert isinstance(live_facts, tuple)
    if (
        allow_unbound
        and _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
        not in env.bound_runtime_input_specialization_results
    ):
        facts = live_facts
    if (
        not isinstance(facts, tuple)
        or len(facts) != len(sources) * (len(sources) - 1) // 2
        or any(type(fact) is not bool for fact in facts)
    ):
        return False
    live = tuple(
        isinstance(value, torch.Tensor) and not isinstance(value, FakeTensor)
        for value in runtime_values
    )
    # Keep rejecting contradictory live arguments, including a partial weak
    # fallback where only some construction tensors are still alive.
    if any(
        left_live and right_live and live_fact != fact
        for (left_live, right_live), live_fact, fact in zip(
            itertools.combinations(live, 2), live_facts, facts, strict=True
        )
    ):
        return False
    wanted = frozenset((left_source, right_source))
    for pair, fact in zip(itertools.combinations(sources, 2), facts, strict=True):
        if frozenset(pair) == wanted:
            return fact is True
    return False
