"""The routing and buffer sizes of the grouped-reduce dispatchers.

Role: device helper.  ``reduce_helpers`` (the device module generated code
imports) dispatches on these predicates and allocates by these sizes as it
traces; ``smem_footprint`` and the register-tile emitter size the
dispatchers' buffers by the same copies.  The module is DSL-free, and the
CuTe disk-cache key hashes it with ``reduce_helpers``
(``source_dependencies``), so it holds nothing else: an edit to
``reduce_call``, the passes' vocabulary, leaves compiled kernels cached, and
so does one to the CPU-side sizes there (``shared_columns_elements``).
"""

from __future__ import annotations

# Above this many warps the serial fold's per-thread chain outgrows the
# two-stage form's constant shuffle cost.
_SERIAL_MAX_WARPS = 8


def use_serial_block_reduce(pre: int, group_span: int, group_count: int) -> bool:
    """Whether ``_cute_grouped_reduce_shared_two_stage`` takes the serial
    form: one group of 2 to 8 warps spanning the whole CTA."""
    # group_count == 1 with pre == 1 means ONE group spanning the whole
    # CTA at every current emitter (they all size group_count as
    # num_threads // group_span); the serial body's warp indexing relies
    # on that invariant.
    return (
        pre == 1
        and group_count == 1
        and group_span % 32 == 0
        and 2 <= group_span // 32 <= _SERIAL_MAX_WARPS
    )


def use_interleaved_shared_reduce(pre: int, group_span: int) -> bool:
    """Whether ``_cute_grouped_reduce_shared_two_stage`` takes an interleaved
    form (the warp form below 32 siblings, the block form from 32): a
    power-of-two sibling count ``pre`` dividing a whole-warp span."""
    return (
        pre > 1
        and pre & (pre - 1) == 0
        and group_span % 32 == 0
        and group_span % pre == 0
        and (pre < 32 or pre % 32 == 0)
    )


def use_strided_warp_reduce(pre: int, group_span: int) -> bool:
    """Whether ``_cute_grouped_reduce_warp`` reduces its ``pre`` interleaved
    siblings with one partial butterfly: power-of-two ``pre`` and span, the
    span within a warp and above ``pre``.  With one lane per sibling the
    butterfly has no shuffle, only combines with constant identities, which
    the compiler may fold away: a NaN would keep its payload where the
    masked passes, combining with shuffled identities, canonicalize it."""
    return (
        pre > 1
        and pre & (pre - 1) == 0
        and group_span & (group_span - 1) == 0
        and pre < group_span <= 32
    )


def cooperative_column_share(pre: int, width: int, group_span: int) -> int:
    """Threads of a group that sum one output of the cooperative column fold
    (``reduce_helpers._cute_cooperative_column_sums``): the largest power of
    two that divides the reduction extent ``group_span // pre`` (so every
    lane reads the same number of partials) and is at most a warp and the
    group's threads per output."""
    extent = group_span // pre
    per_output = group_span // (pre * width)
    share = min(32, extent & -extent, max(1, per_output))
    return 1 << (share.bit_length() - 1)


def cooperative_column_slots(pre: int, width: int, group_span: int) -> int:
    """Output slots per group of the cooperative column fold, the size
    ``smem_footprint`` gives it: one per lane block and pass, so no two
    blocks store to one slot.  The ``pre * width`` outputs take the first
    slots; the blocks past the last output store their recomputed copies to
    slots nothing reads."""
    blocks = group_span // cooperative_column_share(pre, width, group_span)
    return blocks * -(-(pre * width) // blocks)
