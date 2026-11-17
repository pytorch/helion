"""Constant and symbolic host collection predicates used by compiler tests."""

from __future__ import annotations

from typing import NamedTuple

import torch

import helion.language as hl


def collection_membership(
    x: torch.Tensor, mode: hl.constexpr, form: hl.constexpr
) -> torch.Tensor:
    if form == "tuple":
        choices = ("fast", "slow")
    elif form == "list":
        choices = ["fast", "slow"]
    else:
        choices = {"fast": x, "slow": x}
    member = mode in choices
    nonmember = mode not in choices
    out = torch.empty_like(x)
    for i in hl.tile(x.numel()):
        out[i] = x[i] + (3 if member else 7) + (11 if nonmember else 13)
    return out


def symbolic_collection_membership(x: torch.Tensor) -> torch.Tensor:
    # The CPU test inspects these local predicate types after binding.
    member = 7 in (x.size(0), 11)  # noqa: F841
    nonmember = 7 not in (x.size(0), 11)  # noqa: F841
    out = torch.empty_like(x)
    for i in hl.tile(x.numel()):
        out[i] = x[i]
    return out


def ordered_membership_phases(x: torch.Tensor, mode: hl.constexpr) -> torch.Tensor:
    if mode in ("fast", "slow"):
        bias = 3
    else:
        bias = 7
    temporary = torch.empty_like(x)
    output = torch.empty_like(x)
    for i in hl.tile(x.numel()):
        temporary[i] = x[i] + bias
    hl.barrier()
    for i in hl.tile(x.numel()):
        output[i] = temporary[i] * 2
    return output


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


def _collect_membership_args(*args: object) -> list[object]:
    return list(args)


def approximate_comprehension_membership(
    x: torch.Tensor, form: hl.constexpr
) -> torch.Tensor:
    if form in ("slice_key", "slice_key_merge"):
        cuts = {0: slice(0, v) for v in (1, 2) if v == 1}  # noqa: B035 - uncertain slice
        slice_key = cuts[0]
        if form == "slice_key_merge" and x.size(0) > 8:
            slice_key = slice(0, 2)
        choices = [1, 2][slice_key]
    elif form in ("nested_dict", "nested_unpack"):
        nested = {0: [v] for v in (1, 2) if v == 1}  # noqa: B035 - collide filtered keys
        if form == "nested_unpack":
            (choices,) = [nested[0]]
        else:
            choices = nested[0]
    elif form == "dict_string":
        choices = {0: v for v in ("a", "b") if True}  # noqa: B035 - non-mergeable values
    elif form == "dict_scalar":
        scalar_values = {0: v for v in (1, 2) if v == 1}  # noqa: B035 - collide filtered keys
        choices = [scalar_values[0]]
    elif form == "dict_kwargs":
        filtered_keywords = {v: 1 for v in ("k",) if False}
        choices = dict(**filtered_keywords)
    elif form in ("dict", "dict_copy", "dict_union"):
        filtered_dict = {v: x for v in (1, 2, 3) if v > 1}
        if form == "dict_copy":
            choices = dict(filtered_dict)
        elif form == "dict_union":
            choices = filtered_dict | {4: x}
        else:
            choices = filtered_dict
    else:
        filtered = [1 for v in (1, 2, 3) if v > 3]
        if form == "copy":
            choices = [v + 0 for v in filtered]
        elif form == "list_copy":
            choices = list(filtered)
        elif form == "tuple_copy":
            choices = tuple(filtered)
        elif form == "nested_copy":
            choices = tuple([filtered])[0]  # noqa: RUF015 - exercise nested copying
        elif form == "concat":
            choices = filtered + [4]  # noqa: RUF005 - exercise concatenation
        elif form == "repeat":
            choices = 2 * filtered
        elif form == "sorted":
            choices = sorted(filtered)
        elif form == "call_starred":
            choices = _collect_membership_args(*filtered)
        elif form == "starred":
            choices = [*filtered]
        elif form == "slice":
            choices = filtered[::1]
        elif form == "merge":
            if x.size(0) > 8:
                choices = filtered
            else:
                choices = [1]
        else:
            choices = filtered
    needle = "k" if form == "dict_kwargs" else 1
    member = needle in choices  # noqa: F841
    nonmember = needle not in choices  # noqa: F841
    kept = 2 in choices  # noqa: F841
    excluded = 2 not in choices  # noqa: F841
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile]
    return out


def approximate_membership_alias(x: torch.Tensor) -> torch.Tensor:
    original = [1]

    def return_original(value: int) -> list[int]:
        return original

    nested = {0: original for v in (0, 1) if v == 0}  # noqa: B035 - preserve source alias
    choices = nested[0]
    member = 1 in choices  # noqa: F841
    arguments = [0 for v in (0, 1) if v == 0]
    returned = return_original(*arguments)
    returned_member = 1 in returned  # noqa: F841
    source_member = 1 in original  # noqa: F841
    last_values = {0: v for v in (1, 2)}  # noqa: B035 - preserve last assignment
    last_choices = [last_values[0]]
    last_member = 2 in last_choices  # noqa: F841
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile]
    return out


def device_wrapped_membership(
    x: torch.Tensor, form: hl.constexpr, exact: hl.constexpr
) -> torch.Tensor:
    choices = [1] if exact else [1 for v in (1, 2, 3) if v > 3]
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if form == "zip":
            wrapped = zip(choices, choices, strict=True)
        else:
            wrapped = enumerate(choices)
        values = [value for _, value in wrapped]
        member = 1 in values  # noqa: F841
        out[tile] = x[tile]
    return out


def literal_container_slice(x: torch.Tensor) -> torch.Tensor:
    cuts = {0: slice(0, 2) for value in (0,) if True}  # noqa: B035 - uncertain provenance
    choices = list(range(3)[cuts[0]])
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + (3 if 1 in choices else 7)
    return out


def filtered_collection_predicate(
    x: torch.Tensor, limit: int, form: hl.constexpr, on_device: hl.constexpr
) -> torch.Tensor:
    filtered = [1 for value in (1, 2, 3) if value > limit]
    if not on_device:
        if form == "truth":
            if filtered:
                bias = 3
            else:
                bias = 7
        elif form == "not":
            bias = 7 if not filtered else 3
        elif form == "bool":
            bias = 3 if bool(filtered) else 7
        else:
            bias = len(filtered)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if not on_device:
            # Defined above under the same constexpr condition.
            # pyrefly: ignore [unbound-name]
            out[tile] = x[tile] + bias
        elif form == "truth":
            if filtered:
                out[tile] = x[tile] + 3
            else:
                out[tile] = x[tile] + 7
        elif form == "not":
            if not filtered:
                out[tile] = x[tile] + 7
            else:
                out[tile] = x[tile] + 3
        elif form == "bool":
            if bool(filtered):
                out[tile] = x[tile] + 3
            else:
                out[tile] = x[tile] + 7
        else:
            out[tile] = x[tile] + len(filtered)
    return out


def _filtered_length(values: list[int]) -> int:
    return len(values)


def filtered_collection_reduction(
    x: torch.Tensor, limit: int, form: hl.constexpr
) -> torch.Tensor:
    if form == "shape":
        filtered = [value for value in x.shape if value > limit]
    else:
        filtered = [value for value in (1, 2, 3) if value > limit]
    if form == "sum" or form == "shape":
        bias = sum(filtered)
    elif form == "float":
        bias = sum(filtered, 0.5)
    elif form == "any":
        bias = any(filtered)
    else:
        bias = _filtered_length(filtered)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + bias
    return out
