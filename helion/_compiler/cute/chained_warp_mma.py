"""Warp MMA over prepared shared operands, independent of region scheduling.

The caller owns operand layouts, participant predicates, accumulator seed
expressions, result publication and synchronization. This component emits only
the register fragments, shared-to-register copies and ordered K-tile issues.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .chained_execution import ChainedExecution
from .chained_scratch_layout import xor_swizzle

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .chained_warp_loop_carry import WarpLoopCarryEmission
    from .chained_warp_pair import WarpPairStage


def complete_warp_columns(columns: int, threads: int) -> bool:
    """No partial C panel under the original active-warp selection."""
    if columns < 8 or columns % 8 or threads < 32 or threads % 32:
        return False
    warps = min(threads // 32, 2 ** (columns.bit_length() - 4))
    return columns % (8 * warps) == 0


def emit_warp_publication(
    prefix: str, shape: tuple[int, int], *, xor_scratch: bool
) -> list[str]:
    """Publish the unchanged C fragment through its logical shared partition.

    CuTe autovec_copy asks right_inverse for a plain Layout; low-bit XOR
    scratch is a ComposedLayout and cannot use that policy. Scalar indexing
    applies the existing swizzle while keeping fragment coordinates unchanged.
    """
    lines = [f"{prefix}_sc = {prefix}_thr.partition_C({prefix}_c)"]
    if xor_scratch and xor_swizzle(shape) is not None:
        lines.extend(
            [
                f"for {prefix}_publish_index in cutlass.range_constexpr(cute.size({prefix}_acc)):",
                f"    {prefix}_sc[{prefix}_publish_index] = {prefix}_acc[{prefix}_publish_index]",
            ]
        )
    else:
        lines.append(f"cute.autovec_copy({prefix}_acc, {prefix}_sc)")
    return lines


def emit_warp_mma(
    prefix: str,
    dtype: str,
    shape: tuple[int, int, int],
    threads: int,
    inner_axes: dict[str, int],
    seed: Sequence[str],
    *,
    execution: ChainedExecution | None = None,
    pair: WarpPairStage | None = None,
    resident: WarpLoopCarryEmission | None = None,
) -> list[str]:
    if execution is not None and threads > execution.threads:
        raise ValueError("warp MMA participants exceed the execution team")
    execution = execution or ChainedExecution(threads)
    rows, columns, reduction = shape
    if reduction <= 0 or reduction % 8:
        raise ValueError("warp MMA requires complete K8 or K16 panels")
    atom_k = 16 if reduction % 16 == 0 else 8
    if pair is not None:
        pair.check_issue(prefix, dtype, shape, threads, seed, inner_axes, execution)
    if resident is not None:
        if pair is not None:
            raise ValueError("warp carry conflicts with paired register owner")
        resident.check_issue(prefix, dtype, shape, threads, seed, execution)
    register_a = None if pair is None else pair.register_a
    # Initial shared physical slice: exactly one native atom per warp, with the
    # existing canonical FP32 zero seed. Other original schedules stay intact.
    shared_k = (
        dtype == "cutlass.BFloat16"
        and rows == 16
        and columns == 8 * (threads // 32)
        and list(seed) == [f"{prefix}_acc.fill(0.0)"]
        and pair is None
        and resident is None
    )
    k_loop = (
        [
            "from helion._compiler.cute.prepared_warp_contraction import execute_prepared_warp_k",
            f"{prefix}_acc = execute_prepared_warp_k(({prefix}_copy_a, {prefix}_copy_sa, {prefix}_copy_ra, {prefix}_ra), ({prefix}_copy_b, {prefix}_copy_sb, {prefix}_copy_rb, {prefix}_rb), {prefix}_acc, {prefix}_mma, cute.size({prefix}_sa, mode=[2]), False, 1)",
        ]
        if shared_k
        else [
            f"for {prefix}_kk in cutlass.range_constexpr(cute.size({prefix}_sa, mode=[2])):",
            *(
                [
                    f"    cute.copy({prefix}_copy_a, {prefix}_copy_sa[None, None, {prefix}_kk], {prefix}_copy_ra[None, None, 0])"
                ]
                if register_a is None
                else []
            ),
            f"    cute.copy({prefix}_copy_b, {prefix}_copy_sb[None, None, {prefix}_kk], {prefix}_copy_rb[None, None, 0])",
            f"    cute.gemm({prefix}_mma, {prefix}_acc, {prefix}_ra[None, None, {'0' if register_a is None else prefix + '_kk'}], {prefix}_rb[None, None, 0], {prefix}_acc)",
        ]
    )
    topology = f"(1, {threads // 32}, 1)" if pair is None else repr(pair.topology)
    lines = [
        *(
            [
                f"{prefix}_mma = cute.make_tiled_mma(cute.make_mma_atom(cute.nvgpu.warp.MmaF16BF16Op({dtype}, cutlass.Float32, (16, 8, {atom_k}))), atom_layout_mnk={topology})",
                f"{prefix}_thr = {prefix}_mma.get_slice({execution.thread})",
            ]
            if resident is None
            else []
        ),
        f"{prefix}_sa = {prefix}_thr.partition_A({prefix}_a)",
        f"{prefix}_sb = {prefix}_thr.partition_B({prefix}_b)",
        f"{prefix}_ra = {prefix}_mma.make_fragment_A((cute.shape({prefix}_sa)[0], cute.shape({prefix}_sa)[1], 1))"
        if register_a is None
        else f"{prefix}_ra = {register_a}",
        f"{prefix}_rb = {prefix}_mma.make_fragment_B((cute.shape({prefix}_sb)[0], cute.shape({prefix}_sb)[1], 1))",
        *(
            [
                f"{prefix}_copy_a = cute.make_tiled_copy_A(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose={inner_axes['a'] == 0!r}, num_matrices={atom_k // 4}), {dtype}), {prefix}_mma)"
            ]
            if register_a is None
            else []
        ),
        f"{prefix}_copy_b = cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose={inner_axes['b'] == 0!r}, num_matrices={atom_k // 8}), {dtype}), {prefix}_mma)",
        *(
            [f"{prefix}_copy_thr_a = {prefix}_copy_a.get_slice({execution.thread})"]
            if register_a is None
            else []
        ),
        f"{prefix}_copy_thr_b = {prefix}_copy_b.get_slice({execution.thread})",
        *(
            [f"{prefix}_copy_sa = {prefix}_copy_thr_a.partition_S({prefix}_a)"]
            if register_a is None
            else []
        ),
        f"{prefix}_copy_sb = {prefix}_copy_thr_b.partition_S({prefix}_b)",
        *(
            [f"{prefix}_copy_ra = {prefix}_copy_thr_a.retile({prefix}_ra)"]
            if register_a is None
            else []
        ),
        f"{prefix}_copy_rb = {prefix}_copy_thr_b.retile({prefix}_rb)",
        *(
            [
                f"{prefix}_acc = cute.make_rmem_tensor({prefix}_mma.partition_shape_C(({rows}, {columns})), cutlass.Float32)"
            ]
            if resident is None
            else []
        ),
        *seed,
        *k_loop,
    ]
    if pair is not None:
        pair.record_issue(lines)
    if resident is not None:
        resident.record_issue(lines)
    return lines
