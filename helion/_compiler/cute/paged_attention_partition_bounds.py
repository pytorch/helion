"""Bounds of a persistent paged-attention launch's plan that the host and
the device compute alike.

The compile-time side (``paged_attention_partition``) decides from them how
wide a bind's slot products are and which range skews it offers, and sizes
a balanced launch's workspace by the record; the device template
(``_paged_attention_partition``) caps a launch's active CTAs and the skew
it applies with them, and lays a record out. Each is stated here once. This module
imports nothing, so the template, which imports nothing of the compiler,
and the compiler can both import it; that makes this file a source
dependency of the paged-attention wrapper.
"""

from __future__ import annotations

# The mean share of a launch divided by this bounds the skew it applies
# (``_paged_attention_partition.partition_plan``), and a bind offers only
# the skews its largest possible mean share can apply
# (``paged_attention_partition.partition_skew_choices``). A skew costs its
# own length at the tail of the launch, so it pays only as a small part of
# the share. B200, 148 CTAs, time against no skew: on shares of 458 and 529
# slots, 8 slots (under 2% of the share) took 0.2 to 0.4% off and 16 (3 to
# 3.5%) added 0.1 to 0.3%; on shares of 39 and 25 slots, 8 slots (a fifth
# and a third) added 7 and 10%.
SKEW_SHARE = 32

# A workspace record starts with its two counters, a 32-bit word per head
# (``_paged_attention_partition._count_arrival`` addresses a head's word); the
# host zeroes the counters of a workspace it allocates by this size
# (``helion.runtime.cute.paged_partition``).
RECORD_COUNTER_BYTES = 2 * 4

# The slots follow the counters at this offset, so no counter shares a
# cache line with a partial.
RECORD_HEADER_BYTES = 128


def plan_grid_bound(units: int, page_limit: int) -> int:
    """The most active CTAs whose slot products stay inside an Int32.

    A unit has one slot per page and one more, and finding a CTA's range
    multiplies the slots of all units with a CTA count. The device caps a
    launch's active CTAs here when its products are Int32; the compile-time
    side gives a bind Int32 products only while this covers twice the
    device's SMs (``paged_attention_partition.partition_slot_bits``).
    """
    return (2**31 - 1) // (units * (page_limit + 1))


def partial_slot_bytes(query_columns: int) -> tuple[int, int]:
    """Bytes of one head's partial panel, and of the slot that holds it.

    The panel is one 16-byte vector per thread and row quad, so a warp's
    lanes cover 512 contiguous bytes; the rows' ``(maximum, denominator)``
    pairs follow. A slot is a multiple of 128 bytes.
    """
    panel = (query_columns // 4) * 128 * 16
    return panel, -(-(panel + query_columns * 8) // 128) * 128


def partition_record_bytes(query_columns: int) -> int:
    """Bytes of one CTA's record in a balanced launch's workspace.

    ``RECORD_HEADER_BYTES`` for the two counters, one per head, then four
    slots, two per head (``partial_slot_bytes``). The compile-time side
    sizes the workspace by it, and the device code, which lays a record out
    from the same two definitions, checks the size it is given.
    """
    return RECORD_HEADER_BYTES + 4 * partial_slot_bytes(query_columns)[1]
