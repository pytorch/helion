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


def partition_skew_choices(
    choices: tuple[int, ...], units: int, page_limit: int, capacity: int
) -> tuple[int, ...]:
    """The range skews of ``choices`` a balanced bind can apply at all.

    The device applies at most a thirty-second of the launch's mean share
    (``_paged_attention_partition.partition_plan``, by the same
    ``SKEW_SHARE``): a skew costs its own
    length at the end of the launch, so it pays only as a small fraction of
    the share, and no range is ever empty. The mean share is at most what
    units of ``page_limit`` pages give the launch's ``min(units, capacity)``
    CTAs, ``capacity`` being the SMs the kernel may use
    (``ConfigSpec.num_sm``, which leaves the reserved ones out): a larger
    skew would be the same kernel as that bound for every launch. A launch
    of shorter units, or of a window's few pages out of a long page table,
    applies less than the config names, down to none.
    """
    largest = units * (page_limit + 1) // min(units, capacity) // SKEW_SHARE
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
