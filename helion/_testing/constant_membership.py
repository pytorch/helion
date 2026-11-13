"""Constant and symbolic host collection predicates used by compiler tests."""

from __future__ import annotations

from typing import NamedTuple

import torch

import helion.language as hl


class MembershipTuple(NamedTuple):
    first: str | int
    second: str


class CustomMembershipTuple(MembershipTuple):
    def __contains__(self, item: object) -> bool:
        return item == "first"


def record_membership(x: torch.Tensor, form: hl.constexpr) -> torch.Tensor:
    if form == "custom":
        record = CustomMembershipTuple("fast", "slow")
    elif form == "symbolic":
        record = MembershipTuple(x.size(0), "slow")
    else:
        record = MembershipTuple("fast", "slow")
    # Containment uses positional values, never the record's field names.
    member = "fast" in record
    nonmember = "fast" not in record
    field_member = "first" in record
    out = torch.empty_like(x)
    for i in hl.tile(x.numel()):
        if form == "literal":
            out[i] = (
                x[i]
                + (3 if member else 7)
                + (11 if nonmember else 13)
                + (17 if field_member else 19)
            )
        else:
            # Negative controls retain symbolic predicates, not a device IfExp.
            out[i] = x[i]
    return out
