"""A guarded affine global view for an already completed warp C fragment.

The caller owns full same-thread C coverage, completion and the original store
cut. This adapter retains the original point expression and scalar fallback;
it supplies no ordering, alias, allocation or fragment-permutation permission.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ...language.memory_ops import _cute_scalar_pointer_expr
from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain
from .chained_rectangular_leaf import prove_rectangular_leaf

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan
    from .chained_store_expression import StorePoint


def render_warp_global_store(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    store: Node,
    point: StorePoint,
    *,
    accumulator: str,
    thread_slice: str,
    shape: tuple[int, int],
    prefix: str,
    index: str,
    fallback: tuple[str, ...],
) -> tuple[str, ...] | None:
    """Use the original typed point on a complete, runtime-guarded rectangle.

    ``index`` is the exact name embedded in the caller's materialized C read.
    Uniform origins/captures retain their typed original scalar definitions.
    Unknown geometry declines; failed bounds/alignment use ``fallback`` verbatim.
    The actual thread slice defines both coordinate and destination ownership,
    including legal layouts whose warp factors differ from (1, W, 1).
    """
    if point.shape != shape or not isinstance(point.target, str) or not fallback:
        return None
    output = cast("Node", store.args[0])
    fake = output.meta.get("val")
    if not isinstance(fake, torch.Tensor) or fake.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        return None
    definitions: dict[str, str] = {}
    for line in point.lines:
        try:
            statements = ast.parse(line).body
        except SyntaxError:
            return None
        if (
            len(statements) != 1
            or not isinstance(statement := statements[0], ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(target := statement.targets[0], ast.Name)
            or target.id in definitions
        ):
            return None
        definitions[target.id] = ast.unparse(statement.value)
    uniform = {f"chain_origin_{axis}" for axis, _, _ in plan.axes}
    uniform.update(
        argument.name
        for scalar, argument in cg.device_function._expr_args.items()
        if sympy.ask(sympy.Q.integer(scalar)) is True
    )
    if plan.loop is not None:
        uniform.add("chain_loop_index")
        uniform.update(
            name
            for captured, name in plan.loop.captures().items()
            if isinstance(captured.meta.get("val"), (int, torch.SymInt))
            or isinstance(captured.meta.get("val"), torch.Tensor)
            and captured.meta["val"].ndim == 0
            and captured.meta["val"].dtype in (torch.int32, torch.int64)
        )
    proof = prove_rectangular_leaf(
        point.indices,
        definitions,
        row=point.coordinates[0],
        column=point.coordinates[1],
        uniform_names=uniform,
        tile_shape=shape,
        shape=chain._host_shape(fake),
        strides=tuple(fake.stride()),
        dtype=fake.dtype,
        mask=" & ".join(f"({bound})" for bound in point.bounds) or None,
    )
    if proof is None:
        return None
    tensor = point.target
    pointer = _cute_scalar_pointer_expr(tensor, list(proof.origin_indices))
    byte_span = proof.element_bytes * ((shape[0] - 1) * proof.pitch + shape[1] - 1)
    guards = [proof.guard]
    for axis, (extent, stride) in enumerate(
        zip(proof.shape, proof.strides, strict=True)
    ):
        guards.extend(
            (
                f"{tensor}.shape[{axis}] == {extent}",
                f"{tensor}.layout.stride[{axis}] == {stride}",
            )
        )
    dtype = CompileEnvironment.current().backend.dtype_str(fake.dtype)
    row, column = point.coordinates
    body = (
        f"{prefix}_pointer = {pointer}",
        f"{prefix}_address = cutlass.Int64({prefix}_pointer.toint())",
        f"if ({prefix}_address >= 0) & ({prefix}_address <= {2**63 - 1 - byte_span}) & ({prefix}_address % 16 == 0):",
        f"    {prefix}_global = cute.make_tensor({prefix}_pointer.align(16), cute.make_layout({shape!r}, stride=({proof.pitch}, 1)))",
        f"    {prefix}_values = cute.make_rmem_tensor_like({accumulator}, {dtype})",
        f"    {prefix}_coords = {thread_slice}.partition_C(cute.make_identity_tensor({shape!r}))",
        f"    for {index} in cutlass.range_constexpr(cute.size({accumulator})):",
        f"        {row}, {column} = {prefix}_coords[{index}]",
        chain._indent(point.lines, 8),
        f"        {prefix}_values[{index}] = {dtype}({point.value})",
        f"    {prefix}_destination = {thread_slice}.partition_C({prefix}_global)",
        f"    cute.autovec_copy({prefix}_values, {prefix}_destination)",
        f"    {prefix}_copied = cutlass.Boolean(True)",
    )
    return (
        f"{prefix}_copied = cutlass.Boolean(False)",
        f"if {' & '.join(f'({guard})' for guard in guards)}:",
        chain._indent(body),
        f"if not {prefix}_copied:",
        chain._indent(fallback),
    )
