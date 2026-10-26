"""Storage aliasing among a kernel's tensor inputs.

Inputs that view one storage (a tensor passed twice, or views of one buffer),
or storages that overlap (separately constructed over one buffer), are one
allocation to every analysis that orders memory accesses, the
cross-loop dependency graph first of all.  Binding gives such inputs one fake
storage (:meth:`CompileEnvironment.record_input_storage_aliases`), and a
kernel whose code relies on that graph keys its bound kernel on the partition
of its inputs by storage (:func:`register_input_storage_partition`), so a
call that aliases its inputs differently compiles its own kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Hashable

import torch
from torch._dynamo.source import LocalSource
from torch._subclasses.fake_tensor import FakeTensor

from .compile_environment import RuntimeInputSpecialization

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch._guards import Source

    from .compile_environment import CompileEnvironment

# v2: storages whose address ranges overlap are one class.
INPUT_STORAGE_PARTITION_KEY = "input_storage_partition_v2"


def input_storage_extent(
    tensor: torch.Tensor,
) -> tuple[int, int] | tuple[str, torch.UntypedStorage] | None:
    """The memory ``tensor``'s storage covers: ``(start, end)`` byte addresses
    of a real storage, ``("fake", storage)`` for a fake input's storage, or
    None for an empty storage (it aliases nothing) or one that cannot be
    inspected (a tensor subclass without storage)."""
    try:
        storage = tensor.untyped_storage()
        if isinstance(tensor, FakeTensor):
            return ("fake", storage)
        start = storage.data_ptr()
        nbytes = storage.nbytes()
    except (NotImplementedError, RuntimeError, TypeError):
        return None
    return (start, start + nbytes) if start and nbytes else None


def input_storage_classes(values: Sequence[object]) -> list[Hashable | None]:
    """The storage class of every value: tensors whose storages overlap, also
    through others (``[data_ptr, data_ptr + nbytes)`` intervals merged),
    share one, so separately constructed storages over one buffer
    (``from_dlpack``, ``from_blob``) are one allocation; a fake input's class
    is its storage.  None for a value that is not a tensor or has an empty
    storage."""
    result: list[Hashable | None] = [None] * len(values)
    spans: list[tuple[int, int, int]] = []
    for index, value in enumerate(values):
        if not isinstance(value, torch.Tensor):
            continue
        # Inlined input_storage_extent: this runs on every eager call.
        try:
            storage = value.untyped_storage()
            if isinstance(value, FakeTensor):
                result[index] = ("fake", storage)
                continue
            start = storage.data_ptr()
            nbytes = storage.nbytes()
        except (NotImplementedError, RuntimeError, TypeError):
            continue
        if start and nbytes:
            spans.append((start, start + nbytes, index))
    spans.sort()
    key: Hashable = None
    reach = 0
    for start, end, index in spans:
        if start >= reach:  # disjoint from every earlier storage
            key = ("ptr", start)
        if end > reach:
            reach = end
        result[index] = key
    return result


def input_storage_partition(values: Sequence[object]) -> tuple[int, ...]:
    """The storage class of every value, numbered in first-seen order; -1 for
    a value that is not a tensor or has an empty storage."""
    numbers: dict[Hashable, int] = {}
    return tuple(
        -1 if key is None else numbers.setdefault(key, len(numbers))
        for key in input_storage_classes(values)
    )


def _input_tensor_sources(env: CompileEnvironment) -> tuple[Source, ...]:
    """Every tensor input slot: each direct tensor argument by name (one
    tensor passed in two slots has two sources), then tensors nested in
    container arguments."""
    from .host_function import HostFunction

    sources: list[Source] = [
        LocalSource(name, is_input=True)
        for name, value in HostFunction.current().params.arguments.items()
        if isinstance(value, torch.Tensor)
    ]
    for tensor in env.input_sources:
        source = env.tensor_input_source(tensor)
        if source is not None and source not in sources:
            sources.append(source)
    return tuple(sources)


def register_input_storage_partition(env: CompileEnvironment) -> None:
    """Key the bound kernel on which tensor inputs share a storage."""
    sources = _input_tensor_sources(env)
    if len(sources) < 2:
        return
    env.register_runtime_input_specialization(
        INPUT_STORAGE_PARTITION_KEY,
        RuntimeInputSpecialization(
            sources=sources,
            classifier_identity=(
                INPUT_STORAGE_PARTITION_KEY,
                tuple(map(repr, sources)),
            ),
            classifier=input_storage_partition,
            reusable_tensor_properties=frozenset(("storage_span",)),
        ),
    )
