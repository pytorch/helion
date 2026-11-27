"""Compile-time geometry of the paired attention schedule's work partitions.

A unit is the paired schedule's work item: two adjacent leading items. A
persistent partition runs several units in one CTA, which adds query stages
and the plan tables to that CTA's shared memory. The device side, which also
defines the slots a partition divides, is ``_paged_attention_partition``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from ..device_function import TensorArg
from ..host_function import HostFunction
from ..program_id import _compile_device_str
from .paged_attention_partition_bounds import SKEW_SHARE
from .paged_attention_partition_bounds import plan_grid_bound

if TYPE_CHECKING:
    import ast

    from ..device_function import DeviceFunction
    from ..generate_ast import GenerateAST

# The paired body's shared memory for one unit per CTA is 178,960 bytes with
# the query map and 40 query columns, inside the 184,320 the transport asks
# of a device (``paged_attention_transport``). A persistent CTA adds the
# barriers of its query stages, the compute groups' records, the plan
# tables, the plan and the scanned lengths, and with two query stages a
# second query panel of 20,480 bytes. Its largest layout, with every warp
# scanning (609 to 640 units), is 184,504 bytes with one query stage and
# 204,984 with two under ``whole``; the ``balanced`` partition's longer
# records and plan and the groups' posts add 160 bytes to each (184,664 and
# 205,144). The budgets are the next multiples of 4 KiB.
PARTITION_SMEM_BYTES_ONE_STAGE = 188416
PARTITION_SMEM_BYTES_TWO_STAGES = 208896
# The host name of the launch device's resident capacity in CTAs.
PARTITION_CTAS = "_PARTITION_CTAS"


def partition_slot_bits(units: int, page_limit: int, num_sm: int) -> int:
    """Width of the slot products of a CTA's plan, or 0 if it cannot plan.

    A unit has one slot per page and one more, so ``units * (page_limit +
    1)`` bounds the slots of all units, which must fit an Int32. Finding a
    CTA's range multiplies them with a CTA count. The products stay Int32
    while the largest count that cannot overflow them (``plan_grid_bound``,
    the bound the device code caps a launch at) covers twice the device's
    SMs, a margin for launches on a larger device: beyond that bound the
    device code idles CTAs. A bind with more slots uses Int64 products.
    """
    slots = units * (page_limit + 1)
    if not (units > 0 and page_limit > 0 and slots < 2**31):
        return 0
    return 32 if plan_grid_bound(units, page_limit) >= 2 * num_sm else 64


def paired_stream_page_bound(page_extent: object, units: int) -> int:
    """The most pages one CTA of the paired schedule streams, where the
    bind states it and it is below 2**31, else zero.

    A unit's trips are those of the source's loop over its pages, which
    stays inside the page table: at most ``page_extent``, the table's
    extent along its page axis. A CTA of one unit streams that many; a
    persistent CTA at most all ``units`` units, the product
    ``partition_slot_bits`` already requires below 2**31. A page table of
    dynamic extent states no bound. The body counts its per-tile loops in
    Int32 below a bound (``_paged_attention_paired_pipeline._tile_steps``)
    and in the source's Int64 without one.
    """
    if type(page_extent) is not int or page_extent <= 0 or units <= 0:
        return 0
    bound = units * page_extent
    return bound if bound < 2**31 else 0


def window_page_bound(
    window_left: int | None,
    query_rows: int,
    group: int,
    page_limit: int,
    key_width: int = 128,
) -> int | None:
    """Most pages of a unit's page table that hold a key some row of the
    unit sees under a static window, or None where the bind proves no such
    bound and the table's extent stands.

    What the bind proves. The mask is ``key <= position`` and ``key >=
    position - window_left`` with a constant ``window_left >= 0``
    (``LogicalAttentionMask``; None: no window, no bound). Row ``r`` of the
    unit's ``query_rows`` rows stands at ``(length - query_len) + r //
    group`` (``AttentionCoordinates``), so the rows stand at ``(query_rows
    - 1) // group + 1`` consecutive positions, wherever the first one lies.
    ``query_len`` is that first position's offset below the request's
    length, not a count: only the direct-I/O layout and the head-wide
    rescale predicate tie the rows to ``query_len * group``, elsewhere the
    matcher accepts any non-negative offset, zero included, and the bound
    does not read it. Page ``p`` of the table holds the ``key_width`` keys
    from ``p * key_width``.

    A row sees keys out of the ``window_left + 1`` up to its position, so
    all rows see keys out of ``window_left + positions`` consecutive ones;
    a row whose position the source's Int32 length arithmetic wraps below
    zero sees none. ``n`` consecutive keys touch at most ``(n + key_width -
    2) // key_width + 1`` pages: one more than they fill when they start on
    a page's last key. A 2047-key window with four or five positions gives
    18; the pages it covers at a given length are 17 or 18.

    That counts each page once only while no two pages of the table have
    the same key coordinates, which a table of more than ``2**31`` keys
    does not give an Int32 coordinate: such a bind gets no bound.
    """
    if (
        window_left is None
        or window_left < 0
        or query_rows <= 0
        or group <= 0
        or key_width <= 0
        or page_limit * key_width > 2**31
    ):
        return None
    positions = (query_rows - 1) // group + 1
    return (window_left + positions + key_width - 2) // key_width + 1


def partition_skew_choices(
    choices: tuple[int, ...],
    units: int,
    page_limit: int,
    capacity: int,
    window_pages: int | None = None,
) -> tuple[int, ...]:
    """The range skews of ``choices`` a balanced bind can apply at all.

    The device applies at most a thirty-second of the launch's mean share
    (``_paged_attention_partition.partition_plan``, by the same
    ``SKEW_SHARE``): a skew costs its own
    length at the end of the launch, so it pays only as a small fraction of
    the share, and no range is ever empty. The mean share is at most what
    units of the most pages a unit can walk give the launch's
    ``min(units, capacity)`` CTAs, ``capacity`` being the SMs the kernel
    may use (``ConfigSpec.num_sm``, which leaves the reserved ones out): a
    larger skew would be the same kernel as that bound for every launch.

    A unit walks at most ``page_limit`` pages, the page table's extent.
    Where the bind proves how many of them hold a key some row of a unit
    sees under a static sliding window (``window_pages``, from
    ``window_page_bound``; None where it proves none), the share is sized
    by the smaller of the two. A 2047-key window out of a 391-page table
    bounds 320 units on 148 SMs at a skew of 1, where the table's extent
    alone offered up to 16: each of those ran the schedule of the skew of
    1, as a kernel of its own that a search compiled and timed. The bound
    is the mask's, not the loop's: a source loop that visits more pages
    than its rows see has shares it understates, and its launches then
    apply the largest skew offered where they could have applied more. A
    launch of shorter units applies less than the config names, down to
    none.
    """
    pages = page_limit if window_pages is None else min(page_limit, window_pages)
    largest = units * (pages + 1) // min(units, capacity) // SKEW_SHARE
    return tuple(skew for skew in choices if skew <= largest)


def partition_q_stages(capacity: int) -> int:
    """Query stages a persistent CTA keeps, or 0 when it does not fit.

    Two stages let the loader request the next unit's queries while the
    previous unit's last QK still reads its own. With one, the loader waits
    for that MMA at every unit boundary. ``capacity`` is the device's
    per-CTA shared memory in bytes; 0 means it is unknown (CPU renders).
    """
    if not capacity or capacity >= PARTITION_SMEM_BYTES_TWO_STAGES:
        return 2
    if capacity >= PARTITION_SMEM_BYTES_ONE_STAGE:
        return 1
    return 0


def partition_capacity(fn: DeviceFunction) -> str:
    """Define the launch device's resident capacity on the host; its name.

    A 640-thread paired CTA takes a whole SM, so the SM count of the launch
    device (less the reserved SMs) is the resident capacity. It is a host
    value of each launch: the kernel reads the grid it was launched with
    and bakes no SM count.
    """
    reserved = CompileEnvironment.current().settings.persistent_reserved_sms
    return fn.host_constexpr_def(
        PARTITION_CTAS,
        f"helion.runtime.get_num_sm({_compile_device_str()}"
        f"{f', reserved_sms={reserved}' if reserved > 0 else ''})",
    )


def partition_launch_grid(fn: DeviceFunction, units: ast.AST) -> ast.AST:
    """The grid of a persistent partition: one CTA per unit up to one per SM.

    ``units`` is the one-dimensional grid of one CTA per unit.
    """
    return expr_from_string(
        f"(min({{units}}[0], {partition_capacity(fn)}),)", units=units
    )


def partition_workspace_arg(
    cg: GenerateAST, record_bytes: int, num_sm: int
) -> TensorArg:
    """Bind the workspace of a balanced launch as a hidden tensor argument.

    One record per resident CTA, through which the parts of a split unit
    exchange their results (``_paged_attention_partition``). The host
    expression runs in the generated launch on every call, ahead of the
    launcher, and returns a buffer the runtime owns per device and stream
    (``helion.runtime.cute.paged_partition``); the kernel reads its base
    and its extent, so no offset depends on the SM count.
    """
    df = cg.device_function
    env = CompileEnvironment.current()
    origin = HostFunction.current().first_tensor_origin(env.device)
    assert origin is not None
    like = origin.host_str()
    cg.ensure_module_statements(
        "from helion.runtime.cute.paged_partition import "
        "paged_partition_workspace as _helion_paged_partition_workspace\n"
    )
    with env.fake_mode:
        fake = torch.empty(
            (num_sm * record_bytes,), dtype=torch.uint8, device=env.device
        )
    argument = TensorArg(
        df.new_var("pa_partition_ws"),
        fake,
        f"_helion_paged_partition_workspace({like}, ctas={partition_capacity(df)}, "
        f"record_bytes={record_bytes})",
    )
    df.arguments.append(argument)
    df._tensor_args[fake] = argument
    df.placeholder_args.add(argument.name)
    return argument
