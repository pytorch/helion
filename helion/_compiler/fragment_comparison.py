"""Guard folded fragment comparisons against captured non-argument values."""

from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING
from typing import cast

from ..autotuner.config_fragment import ConfigSpecFragment
from ..autotuner.config_fragment import _plain_builtin_fragment_fields
from ..autotuner.config_fragment import _plain_builtin_fragment_key
from .compile_environment import CompileEnvironment
from .compile_environment import RuntimeInputSpecialization
from .input_path import InputPathEntry
from .input_path import argument_path
from .input_path import resolve_input_path
from .source_location import current_location
from .type_info import LiteralType
from .type_info import SequenceType
from .type_info import _eval_compare

if TYPE_CHECKING:
    from collections.abc import Hashable
    from collections.abc import Sequence


@dataclass(frozen=True)
class _CapturedComparison:
    indices: tuple[InputPathEntry, ...]
    anchor: object
    op: ast.cmpop
    argument_left: bool
    value_anchor_key: Hashable | None = None

    def __call__(self, values: Sequence[object]) -> Hashable:
        (value,) = values
        value = resolve_input_path(value, self.indices)
        left, right = (
            (value, self.anchor) if self.argument_left else (self.anchor, value)
        )
        result = cast("Hashable", _eval_compare(self.op, left, right))
        if self.value_anchor_key is not None:
            # Retain mutation detection for a captured anchor. A fresh inline
            # equal-valued anchor is interchangeable; a changed live anchor is
            # not, even when this particular comparison still returns False.
            assert isinstance(self.anchor, ConfigSpecFragment)
            return result, _plain_builtin_fragment_key(self.anchor)
        return result


def _value_comparison_anchor_key(
    op: ast.cmpop, argument: object, anchor: object
) -> Hashable | None:
    if (
        not isinstance(op, (ast.Eq, ast.NotEq))
        or type(argument) is not type(anchor)
        or not isinstance(anchor, ConfigSpecFragment)
        or not dataclasses.is_dataclass(anchor)
    ):
        return None
    fields = _plain_builtin_fragment_fields(anchor)
    key = _plain_builtin_fragment_key(anchor)
    if fields is None or key is None:
        return None
    # NaN equality inside dataclass tuples can observe the identity shortcut.
    # The bitwise domain key therefore cannot substitute for an anchor identity.
    pending = [value for _, value in fields]
    while pending:
        value = pending.pop()
        if type(value) is float and math.isnan(value):
            return None
        if type(value) in (tuple, list):
            pending.extend(cast("Sequence[object]", value))
        elif type(value) is dict:
            pending.extend(value.values())
    return type(anchor), key


def guard_fragment_comparison(
    op: ast.cmpop, left: LiteralType, right: LiteralType, result: object
) -> None:
    if (
        not isinstance(op, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot))
        or type(result) is not bool
    ):
        return
    if not isinstance(left.value, ConfigSpecFragment) and not isinstance(
        right.value, ConfigSpecFragment
    ):
        return
    argument_left = left.origin.is_argument()
    if argument_left == right.origin.is_argument():
        return
    argument, anchor = (left, right) if argument_left else (right, left)
    path = argument_path(argument.origin)
    if path is None:
        return
    root, indices = path
    location = current_location()
    identity = (root.name, indices, type(op).__name__, argument_left)
    site = (
        f"fragment_comparison:{location.filename}:{location.lineno}:"
        f"{location.colno}:{identity!r}"
    )
    env = CompileEnvironment.current()
    value_anchor_key = _value_comparison_anchor_key(op, argument.value, anchor.value)
    ordinal = 0
    # Static iterations may revisit one comparison with different anchors.
    # Keep published classifiers immutable, and number only this site's guards
    # in discovery order; object identities never enter a persistent key.
    while (key := f"{site}:{ordinal}") in env.runtime_input_specializations:
        previous = env.runtime_input_specializations[key].classifier
        assert isinstance(previous, _CapturedComparison)
        if previous.anchor is anchor.value or (
            value_anchor_key is not None
            and previous.value_anchor_key == value_anchor_key
        ):
            return
        ordinal += 1
    env.register_runtime_input_specialization(
        key,
        RuntimeInputSpecialization(
            sources=(root.to_source(),),
            classifier_identity=(*identity, ordinal, value_anchor_key),
            binding_anchors=(anchor.value,) if value_anchor_key is None else (),
            classifier=_CapturedComparison(
                indices=indices,
                anchor=anchor.value,
                op=op,
                argument_left=argument_left,
                value_anchor_key=value_anchor_key,
            ),
        ),
    )


def guard_fragment_sequence_equality(left: SequenceType, right: SequenceType) -> None:
    """Sequence equality observes the fragment identities of corresponding leaves."""
    for left_item, right_item in zip(
        left.element_types, right.element_types, strict=False
    ):
        if isinstance(left_item, SequenceType) and isinstance(right_item, SequenceType):
            guard_fragment_sequence_equality(left_item, right_item)
        elif (
            isinstance(left_item, LiteralType)
            and isinstance(right_item, LiteralType)
            and (
                isinstance(left_item.value, ConfigSpecFragment)
                or isinstance(right_item.value, ConfigSpecFragment)
            )
        ):
            guard_fragment_comparison(
                ast.Eq(),
                left_item,
                right_item,
                _eval_compare(ast.Eq(), left_item.value, right_item.value),
            )
