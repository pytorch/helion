"""CuTe primitives for a proved, single-CTA gathered-row MMA pipeline."""

from __future__ import annotations

from typing import cast

import cutlass
from cutlass._mlir.dialects import cute_nvgpu
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import dsl_user_op
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.utils import blackwell_helpers
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout
from cutlass.utils.layout import LayoutEnum

from ._mlir_compat import ir

__all__ = [
    "LayoutEnum",
    "blackwell_helpers",
    "gather_four_rows",
    "make_tma_arguments",
    "pipeline",
    "tma_descriptor_address",
    "transform_partitioned_tensor_layout",
    "utils",
]


@dsl_user_op
def make_tma_arguments(
    lhs: cute.Tensor,
    rhs: cute.Tensor,
    bn: int,
    stages: int,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> tuple:
    """Build full-allocation descriptors and matching staged SMEM layouts.

    A gather transaction selects four independent rows. Its descriptor box
    contains one contiguous 64-element row; the destination layout groups four
    rows before repeating the 128-byte swizzle atom. B has logical E,K,N axes.
    """
    dtype = lhs.element_type
    mma = blackwell_helpers.make_trivial_tiled_mma(
        dtype,
        dtype,
        OperandMajorMode.K,
        OperandMajorMode.MN,
        cutlass.Float32,
        tcgen05.CtaGroup.ONE,
        (128, min(bn, 256)),
        tcgen05.OperandSource.SMEM,
        loc=loc,
        ip=ip,
    )
    atom = tcgen05.make_smem_layout_atom(
        tcgen05.SmemLayoutAtomKind.K_SW128, dtype, loc=loc, ip=ip
    )
    gather_atom = cute.make_composed_layout(
        atom.inner,
        0,
        cute.make_layout((4, 64), stride=atom.outer.stride, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    lhs_layout = cute.tile_to_shape(
        gather_atom, (128, 64, stages), order=(0, 1, 2), loc=loc, ip=ip
    )
    lhs_box = cute.make_composed_layout(
        atom.inner,
        0,
        cute.make_layout((1, 64), stride=atom.outer.stride, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    lhs_atom, _lhs_coordinates = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        lhs,
        lhs_box,
        lhs_box.shape,
        loc=loc,
        ip=ip,
    )
    rhs_layout = cute.tile_to_shape(
        tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.MN_SW128, dtype, loc=loc, ip=ip
        ),
        (bn, 64, stages),
        order=(1, 0, 2),
        loc=loc,
        ip=ip,
    )
    # Admission and the shape-baked wrapper establish a rank-three RHS.
    rhs_shape = cast("tuple[int, int, int]", rhs.shape)
    rhs_nke = cute.make_tensor(
        rhs.iterator,
        cute.make_layout(
            (rhs_shape[2], rhs_shape[1], rhs_shape[0]),
            stride=(rhs.stride[2], rhs.stride[1], rhs.stride[0]),
            loc=loc,
            ip=ip,
        ),
        loc=loc,
        ip=ip,
    )
    rhs_atom, rhs_coordinates = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        rhs_nke,
        cute.slice_(rhs_layout, (None, None, 0), loc=loc, ip=ip),
        (bn, 64),
        loc=loc,
        ip=ip,
    )
    return lhs_atom, rhs_atom, rhs_coordinates, mma, lhs_layout, rhs_layout


@dsl_user_op
def tma_descriptor_address(
    atom: cute.CopyAtom,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> cute.Pointer:
    executable = cute_nvgpu.atom_make_exec_tma(atom._trait.value, loc=loc, ip=ip)
    pointer_type = ir.Type.parse(
        "!cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic, align<128>>"
    )
    return cute_nvgpu.get_tma_desc_addr(pointer_type, executable, loc=loc, ip=ip)


@dsl_user_op
def gather_four_rows(
    descriptor: cute.Pointer,
    destination: cute.Pointer,
    barrier: cute.Pointer,
    column: cutlass.Int32,
    row0: cutlass.Int32,
    row1: cutlass.Int32,
    row2: cutlass.Int32,
    row3: cutlass.Int32,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> None:
    """Issue one SM100 gather4 transaction from an elected producer lane."""
    cute.arch.inline_ptx(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4."
        "mbarrier::complete_tx::bytes.cta_group::1 "
        "[{$r0}], [{$r1}, {{$r2}, {$r3}, {$r4}, {$r5}, {$r6}}], [{$r7}];",
        read_only_args=[
            destination.toint(loc=loc, ip=ip),
            descriptor.toint(loc=loc, ip=ip),
            cutlass.Int32(column),
            cutlass.Int32(row0),
            cutlass.Int32(row1),
            cutlass.Int32(row2),
            cutlass.Int32(row3),
            cutlass.Int32(barrier.toint(loc=loc, ip=ip)),
        ],
        loc=loc,
        ip=ip,
    )
