"""Packet destinations in already admitted complete native prepared images.

These records add no image, allocation, or synchronization authority. The
original binder owns the native view; source identity and runtime guards remain
in the vector leaf emitter. Logical transposes cannot use row-wise packets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .chained_frontier_ownership import _group_matches
from .chained_frontier_ownership import _singleton_matches

if TYPE_CHECKING:
    from .chained_preparation_frame import PreparationFrame
    from .chained_prepared_groups import PreparedGroupBinding
    from .chained_prepared_operands import PreparedOperand


@dataclass(frozen=True)
class NativeVectorSink:
    """A contiguous eight-half packet through the unchanged native pointer."""

    target: str
    shape: tuple[int, int]
    dtype: str

    def pointer(self, row: str, column: str) -> str:
        return f"{self.target}.iterator + {self.target}.layout(({row}, {column}))"

    def aligned_pointer(self, pointer: str) -> str:
        # CuTe Pointer.align rebuilds a pointer when its inferred alignment is
        # too small. That conversion discards the native pointer's swizzle;
        # restore the original one after the caller's packet-alignment guard.
        # recast_ptr replaces the swizzle, so this also handles the static case
        # where align returns the original, already swizzled pointer.
        return (
            f"cute.recast_ptr({pointer}.align(16), "
            f"{self.target}_layout.inner, dtype={self.dtype})"
        )


def native_vector_sinks(
    frame: PreparationFrame,
    operands: tuple[PreparedOperand, ...],
    groups: tuple[PreparedGroupBinding, ...],
) -> dict[str, NativeVectorSink]:
    """Recheck exact full native bindings, then describe their packet views.

    The supported K16/32/64/128 layouts have a native BYTE-pointer swizzle whose
    target bits start at bit four. An aligned eight-half packet therefore keeps
    all sixteen consecutive byte addresses, including the K128 row-major panel
    order. Existing native binding proofs own dense member intervals, offsets,
    full logical domains and lifetimes. Source identity is not inferred here;
    the expression emitter still requires an exact raw load before using a sink.
    """
    result = {}

    def add(name: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        if (
            len(shape) != 2
            or type(shape[0]) is not int
            or shape[0] <= 0
            or shape[0] % 16
            or shape[1] not in (16, 32, 64, 128)
            or dtype not in (torch.bfloat16, torch.float16)
        ):
            return
        sink = NativeVectorSink(
            name,
            (shape[0], shape[1]),
            "cutlass.BFloat16" if dtype == torch.bfloat16 else "cutlass.Float16",
        )
        if name in result and result[name] != sink:
            raise ValueError("native packet image has incompatible views")
        result[name] = sink

    for operand in operands:
        if not _singleton_matches(frame, operand):
            raise ValueError("native packet operand binding changed")
        add(operand.buffer.name, operand.buffer.shape, operand.buffer.dtype)
    for binding in groups:
        if not _group_matches(frame, binding):
            raise ValueError("native packet group binding changed")
        for member in binding.candidate.members:
            if member.logical_modes == (0, 1):
                add(member.buffer.name, member.buffer.shape, member.buffer.dtype)
    return result
