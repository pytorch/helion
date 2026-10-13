"""Exact same-warp ownership maps for typed register-only contractions.

The bounded geometry is one 16x16 component of m16n8k16 warp MMA. Maps come
from actual CuTe partitions, not tensor-shape equality or graph-specific slot
permutations. They describe raw transport only: no arithmetic, narrowing,
instruction choice, lifetime permission, or synchronization is implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib
from typing import TYPE_CHECKING
from typing import Literal

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


FragmentRole = Literal["a", "b", "c"]
Coordinate = tuple[int, int]


@dataclass(frozen=True)
class RegisterElement:
    lane: int
    slot: int


@dataclass(frozen=True)
class WarpFragmentMap:
    """Destination lane/slot -> exact source lane/slot, without a dtype change.

    A/B source values use canonical logical C ownership; the destination is
    the native MMA operand. ``transpose`` retains the original physical dot
    orientation: transposed A comes from logical RHS, transposed B from LHS.
    For C, native FP32 C is the source and canonical logical C the destination.

    Coordinate tables are expressed in the source matrix's axes. Recast pairs
    specify the low/high scalar slots of each 32-bit word in actual storage
    order. C has no halfword pairs and must never narrow to use b16 transport.
    """

    role: FragmentRole
    dtype: torch.dtype
    transpose: bool
    sources: tuple[tuple[RegisterElement, ...], ...]
    source_coordinates: tuple[tuple[Coordinate, ...], ...]
    destination_coordinates: tuple[tuple[Coordinate, ...], ...]
    source_word_pairs: tuple[tuple[int, int], ...]
    destination_word_pairs: tuple[tuple[int, int], ...]

    @property
    def element_bits(self) -> int:
        return self.dtype.itemsize * 8

    @property
    def same_lane(self) -> bool:
        return all(
            source.lane == lane
            for lane, sources in enumerate(self.sources)
            for source in sources
        )


def _word_pairs(
    scalar_offsets: Sequence[int], word_offsets: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    """Resolve actual element-unit recasts; do not assume flat adjacent slots."""
    slots = {offset: slot for slot, offset in enumerate(scalar_offsets)}
    if len(slots) != len(scalar_offsets):
        raise ValueError("register scalar storage aliases within a fragment")
    pairs = tuple((slots[2 * offset], slots[2 * offset + 1]) for offset in word_offsets)
    if sorted(slot for pair in pairs for slot in pair) != list(range(len(slots))):
        raise ValueError("register recast does not cover each typed value once")
    return pairs


@lru_cache(maxsize=10)
def _derive_map(
    role: FragmentRole, dtype: torch.dtype, transpose: bool
) -> WarpFragmentMap:
    import cutlass
    import cutlass.cute as cute

    ir = importlib.import_module("cutlass._mlir.ir")
    # C ownership does not depend on the selected half input format. Its map
    # remains raw FP32; the input atom is needed only to query CuTe ownership.
    input_type = cutlass.BFloat16 if dtype is torch.bfloat16 else cutlass.Float16
    source_coordinates: list[tuple[Coordinate, ...]] = []
    destination_coordinates: list[tuple[Coordinate, ...]] = []
    source_pairs: tuple[tuple[int, int], ...] = ()
    destination_pairs: tuple[tuple[int, int], ...] = ()
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = cute.make_tiled_mma(
                cute.make_mma_atom(
                    cute.nvgpu.warp.MmaF16BF16Op(
                        input_type, cutlass.Float32, (16, 8, 16)
                    )
                ),
                atom_layout_mnk=(1, 1, 1),
            )
            identity = cute.make_identity_tensor((16, 16))
            for lane in range(32):
                thread = mma.get_slice(lane)
                source = thread.partition_C(identity)
                if role == "a":
                    destination = thread.partition_A(identity)
                elif role == "b":
                    destination = thread.partition_B(identity)
                else:
                    destination = source
                source_coordinates.append(
                    tuple(
                        (int(source[slot][0]), int(source[slot][1]))
                        for slot in range(cute.size(source))
                    )
                )
                exchange = transpose if role != "b" else not transpose
                destination_coordinates.append(
                    tuple(
                        (
                            int(destination[slot][int(exchange)]),
                            int(destination[slot][int(not exchange)]),
                        )
                        for slot in range(cute.size(destination))
                    )
                )
            if role != "c":
                source_registers = cute.make_rmem_tensor((8,), input_type)
                partition = (
                    mma.get_slice(0).partition_A(identity)
                    if role == "a"
                    else mma.get_slice(0).partition_B(identity)
                )
                destination_registers = (
                    mma.make_fragment_A(partition.shape)
                    if role == "a"
                    else mma.make_fragment_B(partition.shape)
                )
                pairs = []
                for registers in (source_registers, destination_registers):
                    words = cute.recast_tensor(registers, cutlass.Int32)
                    pairs.append(
                        _word_pairs(
                            tuple(
                                int(registers.layout(slot))
                                for slot in range(cute.size(registers))
                            ),
                            tuple(
                                int(words.layout(word))
                                for word in range(cute.size(words))
                            ),
                        )
                    )
                source_pairs, destination_pairs = pairs
        if not module.operation.verify():
            raise ValueError("invalid CuTe register fragment ownership IR")
    owners = {
        coord: RegisterElement(lane, slot)
        for lane, coordinates in enumerate(source_coordinates)
        for slot, coord in enumerate(coordinates)
    }
    all_coordinates = {(row, column) for row in range(16) for column in range(16)}
    if (
        len(owners) != 256
        or owners.keys() != all_coordinates
        or any(len(coordinates) != 8 for coordinates in source_coordinates)
        or any(len(coordinates) != 8 for coordinates in destination_coordinates)
        or {coord for coords in destination_coordinates for coord in coords}
        != all_coordinates
    ):
        raise ValueError("CuTe fragment layout is not one complete warp component")
    sources = tuple(
        tuple(owners[coord] for coord in coordinates)
        for coordinates in destination_coordinates
    )
    return WarpFragmentMap(
        role,
        dtype,
        transpose,
        sources,
        tuple(source_coordinates),
        tuple(destination_coordinates),
        source_pairs,
        destination_pairs,
    )


def plan_warp_fragment_map(
    role: FragmentRole, dtype: torch.dtype, *, transpose: bool = False
) -> WarpFragmentMap | None:
    """Admit only unchanged b16 A/B transport or full-precision FP32 C.

    A private CPU MLIR context resolves static layout facts on first use. No
    CUDA device is needed and no IR is emitted into the caller's kernel.
    """
    if (
        role not in ("a", "b", "c")
        or type(transpose) is not bool
        or (
            dtype not in (torch.bfloat16, torch.float16)
            if role != "c"
            else dtype is not torch.float32
        )
    ):
        return None
    return _derive_map(role, dtype, transpose)
