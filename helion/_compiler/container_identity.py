"""Fold container identity without observing temporary tracing containers."""

from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import cast

from .compile_environment import CompileEnvironment
from .compile_environment import RuntimeInputSpecialization
from .compile_environment import _supports_container_identity
from .input_path import InputPathEntry
from .input_path import resolve_input_path
from .input_path import unwrap_constexpr
from .source_location import current_location
from .variable_origin import ArgumentOrigin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .variable_origin import Origin


@dataclass(frozen=True)
class _Operand:
    source_index: int | None
    indices: tuple[InputPathEntry, ...]
    anchor: object = field(compare=False, repr=False)

    def resolve(self, values: Sequence[object]) -> object:
        if self.source_index is None:
            return self.anchor
        return resolve_input_path(values[self.source_index], self.indices)


@dataclass(frozen=True)
class _IdentityComparison:
    left: _Operand
    right: _Operand
    inverted: bool

    def __call__(self, values: Sequence[object]) -> bool:
        same = self.left.resolve(values) is self.right.resolve(values)
        return not same if self.inverted else same


def _input_containers(
    values: Sequence[object],
) -> list[tuple[_Operand, object]]:
    """Find real input containers, independent of representative TypeInfo origins."""
    result = []

    def visit(
        value: object,
        source_index: int,
        path: tuple[InputPathEntry, ...],
        ancestors: frozenset[int],
    ) -> None:
        value = unwrap_constexpr(value)
        if _supports_container_identity(value):
            result.append((_Operand(source_index, path, None), value))
        if id(value) in ancestors:
            return
        ancestors = ancestors | {id(value)}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            children = (
                (InputPathEntry(field.name, True), getattr(value, field.name))
                for field in dataclasses.fields(value)
            )
        elif isinstance(value, tuple) and hasattr(type(value), "__match_args__"):
            children = (
                (InputPathEntry(name, True), getattr(value, name))
                # pyrefly: ignore [missing-attribute]
                for name in type(value).__match_args__
            )
        elif type(value) is dict:
            children = (
                (InputPathEntry(key, False), item) for key, item in value.items()
            )
        elif type(value) in (tuple, list):
            children = (
                (InputPathEntry(index, False), item)
                for index, item in enumerate(cast("Sequence[object]", value))
            )
        else:
            return
        for entry, child in children:
            visit(child, source_index, (*path, entry), ancestors)

    for index, value in enumerate(values):
        visit(value, index, (), frozenset())
    return result


@dataclass(frozen=True)
class _InputAliasPartition:
    operands: tuple[_Operand, ...]

    def __call__(self, values: Sequence[object]) -> tuple[int, ...]:
        # Addresses are temporary lookup keys only. The cache fact contains
        # canonical alias ordinals, and this classifier owns paths, not inputs.
        ordinals: dict[int, int] = {}
        return tuple(
            ordinals.setdefault(id(operand.resolve(values)), len(ordinals))
            for operand in self.operands
        )


def record_opaque_container_result(value: object) -> None:
    """Taint opaque results without guessing whether they came from an input.

    The existing witnesses are frontend-owned and release their originals on
    every compile exit. Marking an aliased input too is deliberately conservative:
    allocation/identity alone cannot prove an opaque helper's returned source.
    """
    env = CompileEnvironment.current()
    for _, container in _input_containers((value,)):
        witness = env.container_identity_witness(container)
        assert witness is not None
        witness.opaque_result = True


def compare_container_identity(
    op: ast.Is | ast.IsNot,
    left: object,
    left_origin: Origin | None,
    right: object,
    right_origin: Origin | None,
    *,
    left_constructed: bool = False,
    right_constructed: bool = False,
    left_captured: bool = False,
    right_captured: bool = False,
) -> bool:
    inverted = isinstance(op, ast.IsNot)
    env = CompileEnvironment.current()
    witnesses = tuple(env.container_identity_witness(value) for value in (left, right))
    left = env.container_identity(left)
    right = env.container_identity(right)
    # ClassType uses a dict proxy. Compare original types after recovering
    # holder identity, not the temporary representation's Python type.
    if type(left) is not type(right):
        return inverted
    same = left is right
    if any(witness is not None and witness.opaque_result for witness in witnesses):
        # An opaque result may be an input, a capture, or a fresh allocation.
        # No finite source-slot inventory proves arbitrary Python's choice.
        # Trace the correct branch now, but never reuse this binding.
        env.requires_fresh_binding = True
        return not same if inverted else same
    runtime_values = env.runtime_arg_values_by_name
    inputs = _input_containers(tuple(runtime_values.values()))
    input_names = tuple(runtime_values)

    def actual_path(
        value: object, origin: Origin | None, captured: bool
    ) -> tuple[ArgumentOrigin, tuple[InputPathEntry, ...]] | None:
        # A captured anchor stays captured even when this bind's input aliases
        # it. Replacing both operands with that input would lose the relation.
        if captured or (origin is not None and origin.is_global()):
            return None
        for operand, original in inputs:
            if original is value:
                assert operand.source_index is not None
                return ArgumentOrigin(
                    input_names[operand.source_index]
                ), operand.indices
        return None

    paths = (
        actual_path(left, left_origin, left_captured),
        actual_path(right, right_origin, right_captured),
    )
    if not any(path is not None for path in paths):
        return not same if inverted else same
    if any(
        path is None and constructed
        for path, constructed in zip(
            paths, (left_constructed, right_constructed), strict=True
        )
    ):
        # A new literal container cannot be a runtime input. Check
        # actual paths first: tuple(input_tuple) and the empty-tuple singleton
        # can preserve an input identity and still need the partition guard.
        # Opaque results were handled above; unproved host values stay anchors.
        return not same if inverted else same
    if inputs:
        # A host filter/helper can select different input aliases without
        # changing their value keys. Its representative origin does not prove
        # the selected path. Guard the input alias partition before using the
        # actual observed container paths for this comparison.
        alias_operands = tuple(operand for operand, _ in inputs)
        env.register_runtime_input_specialization(
            "container_identity:input_alias_partition",
            RuntimeInputSpecialization(
                sources=tuple(ArgumentOrigin(name).to_source() for name in input_names),
                classifier_identity=(
                    "container_input_aliases",
                    input_names,
                    alias_operands,
                ),
                classifier=_InputAliasPartition(alias_operands),
            ),
        )

    if all(path is not None for path in paths):
        # Both operands belong to the complete input inventory guarded above.
        # Its partition proves every pairwise identity, including when the
        # representative path changes between aliased and distinct inputs.
        # Captured anchors and opaque results never enter this case.
        return not same if inverted else same
    if any(path is not None for path in paths):
        sources = []
        operands = []
        for value, path in zip((left, right), paths, strict=True):
            if path is None:
                operands.append(_Operand(None, (), value))
            else:
                root, indices = path
                operands.append(_Operand(len(sources), indices, None))
                sources.append(root.to_source())
        left_operand, right_operand = operands
        classifier = _IdentityComparison(left_operand, right_operand, inverted)
        identity = (
            tuple((path[0].name, path[1]) if path else None for path in paths),
            inverted,
        )
        location = current_location()
        site = (
            f"container_identity:{location.filename}:{location.lineno}:"
            f"{location.colno}:{identity!r}"
        )
        ordinal = 0
        while (key := f"{site}:{ordinal}") in env.runtime_input_specializations:
            previous = env.runtime_input_specializations[key].classifier
            assert isinstance(previous, _IdentityComparison)
            if (
                previous.left.anchor is classifier.left.anchor
                and previous.right.anchor is classifier.right.anchor
            ):
                break
            ordinal += 1
        else:
            env.register_runtime_input_specialization(
                key,
                RuntimeInputSpecialization(
                    sources=tuple(sources),
                    classifier_identity=(*identity, ordinal),
                    binding_anchors=(left_operand.anchor, right_operand.anchor),
                    classifier=classifier,
                ),
            )
    return not same if inverted else same
