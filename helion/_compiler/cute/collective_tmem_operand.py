"""Typed register-to-TMEM staging for a proven native collective A recipe.

The caller proves scalar recipe purity, aliasing, control placement and the
128-thread launch. This module changes only element ownership and storage.
It keeps every original load, predicate, dtype conversion and arithmetic node.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .collective_vector_recipe import emit_vector_recipe
from .scalar_recipe import _clone

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from .contiguous_copy import CopyTensorFacts


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


@dataclass(frozen=True)
class CollectiveTmemOperand:
    """The K-major half-precision A layout of one SM100 CTA.

    M64 uses the 16x32bx2 copy: each warp covers 16 rows and the two half
    warps cover opposite halves of K. M128 uses 32x32b: each physical thread
    covers one row. Both store all K elements and occupy BK/2 TMEM columns.
    The public CUTLASS copy partitions define the destination; the equivalent
    integer coordinates below let existing scalar load proofs see aligned K.
    """

    prefix: str
    tid: str
    bm: int
    bk: int
    dtype: str
    column_offset: int

    def __post_init__(self) -> None:
        if self.bm not in (64, 128) or self.bk not in (16, 32, 64, 128):
            raise ValueError("TMEM A requires M64/128 and K16/32/64/128")
        if self.dtype not in ("cutlass.Float16", "cutlass.BFloat16"):
            raise ValueError("TMEM A requires FP16 or BF16")
        if self.column_offset < 32 or self.column_offset % 32:
            raise ValueError("TMEM A must follow aligned accumulator columns")

    @property
    def values_per_thread(self) -> int:
        return self.bm * self.bk // 128

    @property
    def columns(self) -> int:
        return self.bk // 2

    def coordinates(self, element: str) -> tuple[ast.expr, ast.expr]:
        if self.bm == 64:
            return (
                _expr(f"{self.tid} % 16 + 16 * ({self.tid} // 32)"),
                _expr(f"({self.tid} // 16 % 2) * {self.bk // 2} + {element}"),
            )
        return _expr(self.tid), _expr(element)

    def setup(self, pointer: str) -> list[ast.stmt]:
        p = self.prefix
        operation, repetitions = (
            ("St16x32bx2Op", self.bk // 4)
            if self.bm == 64
            else ("St32x32bOp", self.bk // 2)
        )
        return ast.parse(f"""
{p}_ra_base = {p}_mma.make_fragment_A({p}_mma.partition_shape_A(({self.bm}, {self.bk})))
{p}_ra = cute.make_tensor(cute.recast_ptr({pointer} + {self.column_offset}, dtype={self.dtype}), {p}_ra_base.layout)
{p}_a_mk = cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout({p}_ra)
{p}_a_store_atom = cute.make_copy_atom(cute.nvgpu.tcgen05.{operation}(cute.nvgpu.tcgen05.Repetition({repetitions})), {self.dtype})
{p}_a_store_copy = cute.nvgpu.tcgen05.make_tmem_copy({p}_a_store_atom, {p}_a_mk)
{p}_a_store_thread = {p}_a_store_copy.get_slice({self.tid})
{p}_a_store_destination = {p}_a_store_thread.partition_D({p}_a_mk)
{p}_a_store_coordinates = {p}_a_store_thread.partition_S(cute.make_identity_tensor(({self.bm}, {self.bk})))
{p}_a_values = cute.make_rmem_tensor({p}_a_store_coordinates.shape, {self.dtype})
""").body

    def emit_recipe(
        self,
        statements: Sequence[ast.Assign],
        value: ast.expr,
        *,
        local_m: str,
        local_k: str,
        k_offset: str,
        static_k_extent: int | None,
        vectorize: bool,
        tensors: Mapping[str, CopyTensorFacts],
        aligned_names: Mapping[str, int],
        fresh_name: Callable[[str], str],
    ) -> ast.For:
        """Write the entire register fragment, retaining inactive-lane guards.

        Register indices are static to prevent local-memory spills. Vector
        recipes use the existing 128-bit load proof; a partial synthetic-K
        tile retains the exact scalar branch before any recipe evaluation.
        """
        p = self.prefix
        width = 8 if vectorize else 1
        assert (
            not vectorize or static_k_extent is None or static_k_extent % self.bk == 0
        )
        slot = f"{p}_ai"
        element = f"{slot} * {width}" if width != 1 else slot
        m, k = self.coordinates(element)
        converted = _expr(f"{self.dtype}({ast.unparse(value)})")
        copied: list[ast.stmt]
        if vectorize:
            copied = emit_vector_recipe(
                statements,
                converted,
                coordinate=local_k,
                width=width,
                tensors=tensors,
                aligned_names=aligned_names,
                destination=f"{p}_a_values",
                destination_indices=None,
                destination_offset=_expr(element),
                fresh_name=fresh_name,
            )
        else:
            copied = [
                *(_clone(statement) for statement in statements),
                ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=_expr(f"{p}_a_values"),
                            slice=_expr(element),
                            ctx=ast.Store(),
                        )
                    ],
                    value=converted,
                ),
            ]
            if static_k_extent is not None and static_k_extent % self.bk:
                copied = [
                    ast.If(
                        test=_expr(f"{k_offset} + {local_k} < {static_k_extent}"),
                        body=copied,
                        orelse=ast.parse(
                            f"{p}_a_values[{element}] = {self.dtype}(0)"
                        ).body,
                    )
                ]
        return ast.fix_missing_locations(
            ast.For(
                target=ast.Name(id=slot, ctx=ast.Store()),
                iter=_expr(
                    f"cutlass.range_constexpr({self.values_per_thread // width})"
                ),
                body=[
                    # Original cooperative indices are Int32. In M128 the K
                    # coordinate is otherwise a constexpr, which would change
                    # overflow before a later cast in an unchanged recipe.
                    ast.Assign(
                        [ast.Name(local_m, ast.Store())],
                        _expr(f"cutlass.Int32({ast.unparse(m)})"),
                    ),
                    ast.Assign(
                        [ast.Name(local_k, ast.Store())],
                        _expr(f"cutlass.Int32({ast.unparse(k)})"),
                    ),
                    *copied,
                ],
                orelse=[],
            )
        )

    def publish(self) -> list[ast.stmt]:
        p = self.prefix
        # All four warps issue their typed store. The native compute's existing
        # CTA barrier joins them before MMA; its completion wait protects reuse.
        return ast.parse(f"""
cute.copy({p}_a_store_copy, {p}_a_values, {p}_a_store_destination)
cute.arch.fence_view_async_tmem_store()
""").body
