"""Warp MMA over prepared shared operands, independent of region scheduling.

The caller owns operand layouts, participant predicates, accumulator seed
expressions, result publication and synchronization. This component emits only
the register fragments, shared-to-register copies and ordered K-tile issues.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .chained_execution import ChainedExecution

if TYPE_CHECKING:
    from collections.abc import Sequence


def emit_warp_mma(
    prefix: str,
    dtype: str,
    shape: tuple[int, int, int],
    threads: int,
    inner_axes: dict[str, int],
    seed: Sequence[str],
    *,
    execution: ChainedExecution | None = None,
) -> list[str]:
    if execution is not None and threads > execution.threads:
        raise ValueError("warp MMA participants exceed the execution team")
    execution = execution or ChainedExecution(threads)
    rows, columns, _ = shape
    # Initial shared physical slice: exactly one native atom per warp, with the
    # existing canonical FP32 zero seed. Other original schedules stay intact.
    shared_k = (
        dtype == "cutlass.BFloat16"
        and rows == 16
        and columns == 8 * (threads // 32)
        and list(seed) == [f"{prefix}_acc.fill(0.0)"]
    )
    k_loop = (
        [
            "from helion._compiler.cute.prepared_warp_contraction import execute_prepared_warp_k",
            f"{prefix}_acc = execute_prepared_warp_k(({prefix}_copy_a, {prefix}_copy_sa, {prefix}_copy_ra, {prefix}_ra), ({prefix}_copy_b, {prefix}_copy_sb, {prefix}_copy_rb, {prefix}_rb), {prefix}_acc, {prefix}_mma, cute.size({prefix}_sa, mode=[2]), False, 1)",
        ]
        if shared_k
        else [
            f"for {prefix}_kk in cutlass.range_constexpr(cute.size({prefix}_sa, mode=[2])):",
            f"    cute.copy({prefix}_copy_a, {prefix}_copy_sa[None, None, {prefix}_kk], {prefix}_copy_ra[None, None, 0])",
            f"    cute.copy({prefix}_copy_b, {prefix}_copy_sb[None, None, {prefix}_kk], {prefix}_copy_rb[None, None, 0])",
            f"    cute.gemm({prefix}_mma, {prefix}_acc, {prefix}_ra[None, None, 0], {prefix}_rb[None, None, 0], {prefix}_acc)",
        ]
    )
    return [
        f"{prefix}_mma = cute.make_tiled_mma(cute.make_mma_atom(cute.nvgpu.warp.MmaF16BF16Op({dtype}, cutlass.Float32, (16, 8, 16))), atom_layout_mnk=(1, {threads // 32}, 1))",
        f"{prefix}_thr = {prefix}_mma.get_slice({execution.thread})",
        f"{prefix}_sa = {prefix}_thr.partition_A({prefix}_a)",
        f"{prefix}_sb = {prefix}_thr.partition_B({prefix}_b)",
        f"{prefix}_ra = {prefix}_mma.make_fragment_A((cute.shape({prefix}_sa)[0], cute.shape({prefix}_sa)[1], 1))",
        f"{prefix}_rb = {prefix}_mma.make_fragment_B((cute.shape({prefix}_sb)[0], cute.shape({prefix}_sb)[1], 1))",
        f"{prefix}_copy_a = cute.make_tiled_copy_A(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose={inner_axes['a'] == 0!r}, num_matrices=4), {dtype}), {prefix}_mma)",
        f"{prefix}_copy_b = cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose={inner_axes['b'] == 0!r}, num_matrices=2), {dtype}), {prefix}_mma)",
        f"{prefix}_copy_thr_a = {prefix}_copy_a.get_slice({execution.thread})",
        f"{prefix}_copy_thr_b = {prefix}_copy_b.get_slice({execution.thread})",
        f"{prefix}_copy_sa = {prefix}_copy_thr_a.partition_S({prefix}_a)",
        f"{prefix}_copy_sb = {prefix}_copy_thr_b.partition_S({prefix}_b)",
        f"{prefix}_copy_ra = {prefix}_copy_thr_a.retile({prefix}_ra)",
        f"{prefix}_copy_rb = {prefix}_copy_thr_b.retile({prefix}_rb)",
        f"{prefix}_acc = cute.make_rmem_tensor({prefix}_mma.partition_shape_C(({rows}, {columns})), cutlass.Float32)",
        *seed,
        *k_loop,
    ]
