"""Warp MMA for contractions in a positional root.

Operand expressions retain positional coordinates and typed masks. Each warp
loads those expressions directly into its MMA register fragments; this avoids
adding operand buffers to roots whose shared-memory allocation is already
large. The surrounding emitter owns loop lifetimes and result publication.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import cast

import torch

from ..compile_environment import CompileEnvironment
from . import chained_matmul as cm
from .mma_support import cute_fp32_dot_uses_tf32
from .positional_domains import contraction_of

if TYPE_CHECKING:
    from torch.fx import Node

    from .positional_root import PositionalEmitter


def supports_positional_warp_mma(node: Node) -> bool:
    contraction = contraction_of(node)
    if contraction is None:
        return False
    env = CompileEnvironment.current()
    capability = env.config_spec.target_device_capability
    if capability is None or capability[0] < 8:
        return False
    lhs, rhs = contraction.lhs, contraction.rhs
    dtype = lhs.meta["val"].dtype
    if (
        dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or rhs.meta["val"].dtype != dtype
        or node.meta["val"].dtype != torch.float32
        or dtype == torch.float32
        and not cute_fp32_dot_uses_tf32()
    ):
        return False
    shapes = [cm._shape(value) for value in (lhs, rhs, node)]
    if any(len(shape) < 2 or any(size != 1 for size in shape[:-2]) for shape in shapes):
        return False
    a, b, c = shapes
    return (
        a[-1] == b[-2]
        and a[-2] == c[-2]
        and b[-1] == c[-1]
        and all(size > 0 for shape in shapes for size in shape)
    )


def emit_positional_warp_mma(
    emitter: PositionalEmitter, node: Node
) -> list[str] | None:
    config = emitter.cg.device_function.config.config
    if not config.get("cute_positional_mma", False):
        return None
    if not supports_positional_warp_mma(node):
        return None
    warps = emitter.plan.threads // 32
    warps_m = cast(
        "int",
        config.get("cute_positional_mma_warps_m", 2 if warps % 2 == 0 else 1),
    )
    if warps < 1 or emitter.plan.threads % 32 or warps_m < 1 or warps % warps_m:
        return None
    warps_n = warps // warps_m
    contraction = contraction_of(node)
    assert contraction is not None
    lhs, rhs, acc = contraction.lhs, contraction.rhs, contraction.acc
    shape = cm._shape(node)
    m, n = shape[-2:]
    k_extent = cm._shape(lhs)[-1]
    dtype = lhs.meta["val"].dtype
    tf32 = dtype == torch.float32
    atom_k = 8 if tf32 else 16
    bm = math.ceil(m / (16 * warps_m)) * 16 * warps_m
    bn = math.ceil(n / (8 * warps_n)) * 8 * warps_n
    bk = math.ceil(k_extent / atom_k) * atom_k
    prefix = emitter.fresh("mma")
    env = CompileEnvironment.current()
    dtype_str = env.backend.dtype_str(dtype)
    operation = (
        f"cute.nvgpu.warp.MmaTF32Op((16, 8, {atom_k}))"
        if tf32
        else f"cute.nvgpu.warp.MmaF16BF16Op({dtype_str}, cutlass.Float32, (16, 8, {atom_k}))"
    )
    lines = [
        f"{prefix}_mma = cute.make_tiled_mma({operation}, atom_layout_mnk=({warps_m}, {warps_n}, 1))",
        f"{prefix}_thr = {prefix}_mma.get_slice(ptp_thread)",
        f"{prefix}_ra = {prefix}_thr.make_fragment_A({prefix}_thr.partition_shape_A(({bm}, {atom_k})))",
        f"{prefix}_rb = {prefix}_thr.make_fragment_B({prefix}_thr.partition_shape_B(({bn}, {atom_k})))",
        f"{prefix}_acc = cute.make_rmem_tensor({prefix}_mma.partition_shape_C(({bm}, {bn})), cutlass.Float32)",
        f"{prefix}_ia = {prefix}_thr.partition_A(cute.make_identity_tensor(({bm}, {atom_k})))",
        f"{prefix}_ib = {prefix}_thr.partition_B(cute.make_identity_tensor(({bn}, {atom_k})))",
        f"{prefix}_ic = {prefix}_thr.partition_C(cute.make_identity_tensor(({bm}, {bn})))",
    ]
    row, col, k = (f"{prefix}_{axis}" for axis in ("row", "col", "k"))
    index = f"{prefix}_i"

    def coordinates(value: Node, last: tuple[str, str]) -> tuple[str, ...]:
        value_shape = cm._shape(value)
        mapped = (*("0" for _ in value_shape[:-2]), *last[-len(value_shape) :])
        if not value_shape:
            return ()
        return tuple(
            "0" if size == 1 else coordinate
            for size, coordinate in zip(value_shape, mapped, strict=True)
        )

    c_coordinates = [
        f"{row} = {emitter.index_dtype}({prefix}_ic[{index}][0])",
        f"{col} = {emitter.index_dtype}({prefix}_ic[{index}][1])",
    ]
    initial = "cutlass.Float32(0)"
    seed = emitter.expression()
    if acc is not None:
        initial = f"cutlass.Float32({seed.value(acc, coordinates(acc, (row, col)))})"
    lines.extend(
        [
            f"for {index} in cutlass.range_constexpr(cute.size({prefix}_acc)):",
            *emitter.indent(
                [*c_coordinates, *seed.lines, f"{prefix}_acc[{index}] = {initial}"]
            ),
        ]
    )
    lines.append(f"for {prefix}_base in cutlass.range(0, {bk}, {atom_k}, unroll=1):")
    for side, operand, varying, pair in (
        ("a", lhs, row, (row, k)),
        ("b", rhs, col, (k, col)),
    ):
        expression = emitter.expression()
        value = expression.value(operand, coordinates(operand, pair))
        # A masked scalar product contributes zero even if an expression in
        # another operand yields Inf/NaN. Zero BOTH MMA operands on invalid K;
        # masking only one would turn 0*Inf into a NaN accumulator.
        bounds = [] if bk == k_extent else [f"{k} < {k_extent}"]
        if not emitter.plan.domains.k_promoted(node):
            for source, dimension in (
                (lhs, len(cm._shape(lhs)) - 1),
                (rhs, len(cm._shape(rhs)) - 2),
            ):
                bound = expression.dim_bound(source, dimension, k)
                if bound is not None and bound not in bounds:
                    bounds.append(bound)
        if bounds:
            value = f"({value} if {' and '.join(bounds)} else {dtype_str}(0))"
        # TF32 MMA exposes its register operands as packed i32 values. Use
        # the same explicit rounding instruction as the collective MMA path.
        converted = (
            f"cutlass.Int32(cute.arch.cvt_f32_tf32({value}))"
            if tf32
            else f"{dtype_str}({value})"
        )
        load = [
            f"{varying} = {emitter.index_dtype}({prefix}_i{side}[{index}][0])",
            f"{k} = {emitter.index_dtype}({prefix}_base + {prefix}_i{side}[{index}][1])",
            *expression.lines,
            f"{prefix}_r{side}[{index}] = {converted}",
        ]
        lines.extend(
            emitter.indent(
                [
                    f"for {index} in cutlass.range_constexpr(cute.size({prefix}_r{side})):",
                    *emitter.indent(load),
                ]
            )
        )
    lines.append(
        f"    cute.gemm({prefix}_mma, {prefix}_acc, {prefix}_ra, {prefix}_rb, {prefix}_acc)"
    )
    buffer = emitter.allocate(node)
    out_coords = (*("0" for _ in shape[:-2]), row, col)
    lines.extend(
        [
            f"for {index} in cutlass.range_constexpr(cute.size({prefix}_acc)):",
            *emitter.indent(
                [
                    *c_coordinates,
                    f"if {row} < {m} and {col} < {n}:",
                    f"    {buffer}[{', '.join(out_coords)}] = {prefix}_acc[{index}]",
                ]
            ),
            "cute.arch.sync_threads()",
        ]
    )
    emitter.boundaries[node] = buffer
    return lines
