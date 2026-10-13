"""Bounded, bit-preserving same-warp transport of proven register fragments.

This is only an instruction selector for an ownership map. The caller must
provide disjoint, preallocated tensors with the map's dtype and scalar/word
layouts, and all 32 lanes must participate with ``lane`` in [0, 32). It grants
no graph, lifetime, synchronization, or mathematical-support permission.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .chained_register_fragments import WarpFragmentMap


def _valid_map(mapping: WarpFragmentMap) -> bool:
    if (
        mapping.role not in ("a", "b", "c")
        or type(mapping.transpose) is not bool
        or (
            mapping.dtype is not torch.float32
            if mapping.role == "c"
            else mapping.dtype not in (torch.float16, torch.bfloat16)
        )
        or any(
            len(table) != 32 or any(len(row) != 8 for row in table)
            for table in (
                mapping.sources,
                mapping.source_coordinates,
                mapping.destination_coordinates,
            )
        )
    ):
        return False
    coordinates = {(row, col) for row in range(16) for col in range(16)}
    if any(
        {coord for row in table for coord in row} != coordinates
        for table in (mapping.source_coordinates, mapping.destination_coordinates)
    ):
        return False
    if any(
        type(source.lane) is not int
        or type(source.slot) is not int
        or not 0 <= source.lane < 32
        or not 0 <= source.slot < 8
        or mapping.source_coordinates[source.lane][source.slot]
        != mapping.destination_coordinates[lane][slot]
        for lane, row in enumerate(mapping.sources)
        for slot, source in enumerate(row)
    ):
        return False
    pairs = (mapping.source_word_pairs, mapping.destination_word_pairs)
    if mapping.dtype is torch.float32:
        return pairs == ((), ())
    return all(
        len(table) == 4
        and all(len(pair) == 2 for pair in table)
        and all(type(slot) is int for pair in table for slot in pair)
        and sorted(slot for pair in table for slot in pair) == list(range(8))
        for table in pairs
    )


def _movmatrix_words(mapping: WarpFragmentMap) -> tuple[int, ...] | None:
    """Match each complete destination word against PTX's m8n8 transpose."""
    order = []
    for destination_pair in mapping.destination_word_pairs:
        matches = [
            word
            for word, source_pair in enumerate(mapping.source_word_pairs)
            if all(
                mapping.sources[lane][destination_pair[half]].lane
                == 8 * (lane % 4) + 4 * half + lane // 8
                and mapping.sources[lane][destination_pair[half]].slot
                == source_pair[(lane // 4) % 2]
                for lane in range(32)
                for half in range(2)
            )
        ]
        if len(matches) != 1:
            return None
        order.append(matches[0])
    return tuple(order) if sorted(order) == list(range(4)) else None


def _lane_expression(values: tuple[int, ...], lane: str) -> str | None:
    """Synthesize a small lane-bit permutation, checking the entire truth table."""
    constant = 0
    shifts: dict[int, int] = {}
    for output_bit in range(5):
        column = tuple((value >> output_bit) & 1 for value in values)
        if len(set(column)) == 1:
            constant |= column[0] << output_bit
            continue
        matches = [
            bit
            for bit in range(5)
            if column == tuple((index >> bit) & 1 for index in range(32))
        ]
        if len(matches) != 1:
            return None
        bit = matches[0]
        shift = output_bit - bit
        shifts[shift] = shifts.get(shift, 0) | (1 << bit)
    # Verify the synthesized integer function, rather than trusting a heuristic.
    if any(
        constant
        + sum(
            ((index & mask) << shift) if shift >= 0 else ((index & mask) >> -shift)
            for shift, mask in shifts.items()
        )
        != value
        for index, value in enumerate(values)
    ):
        return None
    terms = []
    for shift, mask in sorted(shifts.items()):
        term = f"(({lane}) & {mask})"
        if shift:
            term = f"({term} {'<<' if shift > 0 else '>>'} {abs(shift)})"
        terms.append(term)
    if constant or not terms:
        terms.append(str(constant))
    return "(" + " | ".join(terms) + ")"


def emit_fragment_transport(
    mapping: WarpFragmentMap,
    source: str,
    destination: str,
    *,
    prefix: str,
    lane: str,
) -> list[str] | None:
    """Emit scalar copies, four raw b16 movmatrix operations, or raw FP32 shuffles.

    ``cute`` and ``cutlass`` are caller globals; the b16 primitive is imported
    under a prefix-local name. No IR or caller state is changed on rejection.
    Identical source/destination expressions are rejected; distinct expressions
    must also refer to nonaliasing storage. Unsupported ownership maps return
    None, never a numeric conversion or an unbounded lane-selection fallback.
    """
    if source == destination or not _valid_map(mapping):
        return None
    if mapping.same_lane:
        slots = tuple(source.slot for source in mapping.sources[0])
        if any(
            tuple(source.slot for source in row) != slots for row in mapping.sources
        ):
            return None
        return [
            f"{destination}[{slot}] = {source}[{source_slot}]"
            for slot, source_slot in enumerate(slots)
        ]
    source_words = f"{prefix}_source_words"
    destination_words = f"{prefix}_destination_words"
    views = [
        f"{source_words} = cute.recast_tensor({source}, cutlass.Int32)",
        f"{destination_words} = cute.recast_tensor({destination}, cutlass.Int32)",
    ]
    if mapping.dtype in (torch.float16, torch.bfloat16):
        order = _movmatrix_words(mapping)
        if order is None:
            return None
        primitives = f"{prefix}_primitives"
        return [
            (
                "from helion._compiler.cute import affine_recurrence_primitives "
                f"as {primitives}"
            ),
            *views,
            *(
                f"{destination_words}[{word}] = {primitives}.movmatrix_b16("
                f"{source_words}[{source_word}])"
                for word, source_word in enumerate(order)
            ),
        ]
    lines = list(views)
    for slot in range(8):
        sources = tuple(row[slot] for row in mapping.sources)
        address = _lane_expression(tuple(source.lane for source in sources), lane)
        slots = sorted({source.slot for source in sources})
        if address is None or len(slots) > 2:
            return None
        selector = None
        if len(slots) == 2:
            for bit in range(5):
                if all(
                    source.slot == slots[(index >> bit) & 1]
                    for index, source in enumerate(sources)
                ):
                    selector = bit
                    break
            if selector is None:
                return None
        temporaries = []
        for index, source_slot in enumerate(slots):
            temporary = f"{prefix}_value_{slot}_{index}"
            temporaries.append(temporary)
            lines.append(
                f"{temporary} = cute.arch.shuffle_sync("
                f"{source_words}[{source_slot}], {address})"
            )
        value = temporaries[0]
        if selector is not None:
            value = (
                f"{temporaries[1]} if (({lane}) & {1 << selector}) != 0 else {value}"
            )
        lines.append(f"{destination_words}[{slot}] = {value}")
    return lines
