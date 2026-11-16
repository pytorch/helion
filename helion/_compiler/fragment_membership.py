"""Keep fragment membership sensitive to original input identities."""

from __future__ import annotations

from .fragment_call import _fragment_call_input
from .fragment_call import guard_fragment_call_input
from .type_info import DictType
from .type_info import LiteralType
from .type_info import SequenceType
from .type_info import TypeInfo


def guard_fragment_membership(left: TypeInfo, right: TypeInfo) -> None:
    pending = [left, right]
    operands = []
    observes_fragments = False
    while pending:
        operand = pending.pop()
        operands.append(operand)
        if isinstance(operand, LiteralType):
            observes_fragments |= bool(_fragment_call_input((operand.value,)))
        elif isinstance(operand, SequenceType):
            pending.extend(operand.element_types)
        elif isinstance(operand, DictType):
            pending.extend(operand.element_types.values())
    if observes_fragments:
        # Membership observes identity before equality, including nested and
        # constructed sequences. Guard complete originating arguments, just as
        # for an opaque Python observer, rather than a representative element
        # or the anchors reached by this trace's short-circuit path.
        for operand in operands:
            guard_fragment_call_input(operand.origin)
