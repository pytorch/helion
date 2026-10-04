"""Typed register-to-TMEM transport for already-proven contraction operands.

These source helpers do not prove graph use, fragment compatibility, placement,
or lifetime. In particular, a narrowed operand never replaces its authoritative
FP32 value unless the caller has independently proved that value dead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .chained_execution import ChainedExecution

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from .chained_contraction_groups import ContractionGroup


@dataclass(frozen=True)
class TmemOperandBinding:
    """An already-published typed operand, independent of its producer kind.

    Placement, original-expression equality, readiness and lifetime are proved
    by its producer. The stage checks its exact physical group and dtype. Its
    base is independent of C: moving an accumulator must not move this image.
    """

    group: ContractionGroup
    physical_shape: tuple[int, int]
    dtype: torch.dtype
    column_offset: int
    base: str = "chain_tptr"


def emit_packed_tmem_fragment(
    prefix: str,
    source_prefix: str,
    shape: tuple[int, int],
    dtype: str,
    coords: tuple[str, str],
    expression_lines: Sequence[str],
    masked_value: str,
    *,
    execution: ChainedExecution | None = None,
    destination: str = "chain_tptr",
) -> list[str]:
    """Pack an original-expression result into a caller-owned TMEM operand.

    ``source_prefix`` binds the existing FP32 ``acc``, ``identity``, ``coords``
    and ``values`` fragments. ``masked_value`` includes the source's cast and
    padding policy. ``destination`` is a Float32 TMEM pointer expression, not a
    byte address. Its allocation, fragment layout, alias safety and lifetime
    remain the caller's responsibility.

    Every thread in the supplied TMEM participant context must execute both
    synchronization statements. A wider pipeline role must supply a separate
    128-thread participant context; placing a role-wide barrier inside a
    first-warpgroup predicate would deadlock and is not supported here.
    """
    execution = execution or ChainedExecution(128)
    if execution.threads != 128:
        raise ValueError("packed TMEM transport requires a 128-thread context")
    if dtype not in ("cutlass.BFloat16", "cutlass.Float16"):
        raise ValueError("packed TMEM transport requires a 16-bit floating dtype")
    if shape[0] <= 0 or shape[1] <= 0 or shape[1] % 2:
        raise ValueError("packed TMEM transport requires positive even columns")
    packed_shape = (shape[0], shape[1] // 2)
    store_repetition = min(32, packed_shape[1] & -packed_shape[1])
    expression = "\n".join(
        "    " + line.replace("\n", "\n    ") for line in expression_lines
    )
    return [
        f"{prefix}_layout = cute.composition({source_prefix}_acc.layout, cute.make_layout({packed_shape!r}))",
        f"{prefix}_coords_layout = cute.composition({source_prefix}_identity.layout, cute.make_layout({packed_shape!r}))",
        f"{prefix}_target = cute.make_tensor({destination}, {prefix}_layout)",
        f"{prefix}_coords_tensor = cute.make_tensor({source_prefix}_identity.iterator, {prefix}_coords_layout)",
        f"{prefix}_copy = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition({store_repetition})), cutlass.Float32), {prefix}_target)",
        f"{prefix}_thread = {prefix}_copy.get_slice({execution.thread})",
        f"{prefix}_destination = {prefix}_thread.partition_D({prefix}_target)",
        f"{prefix}_coords = {prefix}_thread.partition_S({prefix}_coords_tensor)",
        f"{prefix}_packed = cute.make_rmem_tensor({prefix}_coords.shape, cutlass.Float32)",
        f"{prefix}_values = cute.make_tensor(cute.recast_ptr({prefix}_packed.iterator, dtype={dtype}), {source_prefix}_values.layout)",
        f"for {prefix}_index in cutlass.range_constexpr(cute.size({source_prefix}_values)):",
        f"    {coords[0]}, {coords[1]} = {source_prefix}_coords[{prefix}_index]",
        expression,
        f"    {prefix}_values[{prefix}_index] = {masked_value}",
        execution.sync,
        f"cute.copy({prefix}_copy, {prefix}_packed, {prefix}_destination)",
        "cute.arch.fence_view_async_tmem_store()",
        execution.sync,
    ]


def emit_tmem_operand_view(
    prefix: str,
    shape: tuple[int, int, int],
    dtype: str,
    *,
    base: str = "chain_tptr",
) -> list[str]:
    """Rebase a TMEM A layout factory result in the operand's element units.

    ``base`` is a Float32 TMEM pointer name or parenthesized pointer expression.
    ``make_fragment_A`` returns a zero-based layout even for nonzero allocation
    bases. The width ratio below converts the supplied Float32 address to the
    BF16/FP16 iterator's units; this is not byte-pointer arithmetic.
    """
    if dtype not in ("cutlass.BFloat16", "cutlass.Float16"):
        raise ValueError("TMEM operand transport requires a 16-bit floating dtype")
    return [
        f"{prefix}_weight_layout = chain_sm100.make_smem_layout_a({prefix}_mma, {shape!r}, {dtype}, 1)",
        f"{prefix}_ra = {prefix}_mma.make_fragment_A({prefix}_weight_layout.outer)",
        f"{prefix}_ra = cute.make_tensor({prefix}_ra.iterator + (cutlass.Float32.width // {dtype}.width) * {base}.toint(), {prefix}_ra.layout)",
    ]
