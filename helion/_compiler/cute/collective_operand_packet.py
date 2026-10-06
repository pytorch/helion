"""Typed shared stores for complete packets in collective operand layouts."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class SharedOperandPacket:
    """Four initialized TF32 words with the original pointer-carried swizzle.

    Admission belongs to ``plan_collective_operand_packet``. The recipe writes
    every register, including masked zeros, before this synchronous store. It
    does not move source predicates or arithmetic across the packet boundary.
    """

    width: int = 4

    def allocate(self, registers: str) -> ast.stmt:
        return ast.parse(
            f"{registers} = cute.make_rmem_tensor("
            f"cute.make_layout(({self.width},), stride=(1,)), cutlass.TFloat32)"
        ).body[0]

    def store(
        self,
        registers: str,
        destination: str,
        indices: tuple[ast.expr, ...],
        fresh_name: Callable[[str], str],
    ) -> list[ast.stmt]:
        coordinate = ast.unparse(ast.Tuple(elts=list(indices), ctx=ast.Load()))
        view = fresh_name("operand_packet_destination")
        # Rebuilding or aligning this pointer would discard its byte swizzle.
        # Its layout is the unswizzled outer layout in both admitted families.
        return ast.parse(f"""
{view} = cute.make_tensor({destination}.iterator + cute.assume(cute.crd2idx({coordinate}, {destination}.layout), divby={self.width}), cute.make_layout(({self.width},), stride=(1,)))
cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.TFloat32, num_bits_per_copy=128), {registers}, {view})
""").body


def plan_collective_operand_packet(
    *,
    dtype: str,
    native: bool,
    bm: int,
    bn: int,
    bk: int,
    operand_a: bool,
    k_major: bool,
    coordinate_k: bool,
    stages: int,
    width: int,
    base_alignment: int,
) -> SharedOperandPacket | None:
    """Prove a full packet in the existing native or warp TF32 layouts.

    Both layouts carry the swizzle on a 32-bit shared pointer. Native K_SW*
    has a contiguous 16/32-word inner K mode, and MN_SW128_32B a contiguous
    32-word MN mode; their byte swizzles preserve bits 0..3. The warp layout
    is row/column major with a byte swizzle whose base bit is 5. Every admitted
    extent is divisible by four. Thus the existing flat traversal multiplied
    by four starts on an aligned packet and its ``flat < elements`` guard
    admits either all four shared destinations or none. Per-lane *source*
    masks remain inside the recipe and need not agree.

    Keep this proof confined to those constructors and one staging buffer.
    Half operands, different traversal axes, and unknown layouts retain scalar
    stores. SDK address/descriptor tests cover every admitted tile, major mode,
    allocation phase, and lane, independently of this arithmetic proof.
    """
    if (
        dtype != "cutlass.TFloat32"
        or stages != 1
        or width != 4
        or base_alignment < 1024
        or base_alignment % 1024
        or bm not in ((64, 128) if native else (32, 64, 128))
        or bn not in (32, 64)
        or bk not in (16, 32, 64, 128)
        or coordinate_k != (operand_a or k_major)
    ):
        return None
    return SharedOperandPacket()
