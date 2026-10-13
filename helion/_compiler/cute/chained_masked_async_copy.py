"""Use a proven complete masked rectangle in the original warp copy schedule.

The existing rectangular-leaf proof owns mask/index semantics. This adapter
only binds its rectangle to an already allocated plain warp operand image;
the original caller still owns commit/wait, publication and lifetime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ...language.memory_ops import _cute_scalar_pointer_expr
from .chained_rectangular_leaf import prove_rectangular_leaf

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan
    from .chained_native_destination import NativeDestination


def copy_g2s_op(config: Mapping[str, object]) -> str:
    """Cache policy for already proven 128-bit chained cp.async copies.

    Use the raw mapping, not Config.cute_async_load_cache's legacy default cg:
    omitted and ca retain the SDK's original ALWAYS atom byte-for-byte.
    Admission belongs to config normalization; no copy/mask proof is relaxed.
    """
    cache = config.get("cute_async_load_cache", "ca")
    if cache == "ca":
        return "cute.nvgpu.cpasync.CopyG2SOp()"
    if cache == "cg":
        return "cute.nvgpu.cpasync.CopyG2SOp(cute.nvgpu.LoadCacheMode.GLOBAL)"
    raise ValueError(f"invalid chained async load cache: {cache!r}")


def emit_masked_affine_copy(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    operand: Node,
    prefix: str,
    role: str,
    shape: tuple[int, int],
    stride: tuple[int, int],
    inner: int,
    dtype: str,
    fallback: list[str],
    *,
    native_destination: NativeDestination | None = None,
) -> list[str] | None:
    from . import chained_matmul as chain

    if native_destination is not None:
        native_destination.check(
            cg, plan, operand, prefix, role, shape, stride, inner, dtype
        )
    leaf = operand
    while leaf.target in chain._VIEWS:
        leaf = cast("Node", leaf.args[0])
    if (
        leaf.target is not chain.memory_ops.load
        or len(leaf.args) < 3
        or leaf.args[2] is None
        or any(item is not None for item in leaf.args[3:4])
        or cast("Node", leaf.args[0]).target is not chain._tracing_ops._host_tensor
        or leaf.meta["val"].dtype != operand.meta["val"].dtype
        or operand.meta["val"].dtype not in (torch.bfloat16, torch.float16)
    ):
        return None
    width, height = shape[inner], shape[1 - inner]
    columns = min(plan.threads, width // 8)
    if (
        not columns
        or columns & (columns - 1)
        or width % 8
        or height % (plan.threads // columns)
        or stride[inner] != 1
        or stride[1 - inner] % 8
    ):
        return None
    coordinates = ("chain_masked_copy_row", "chain_masked_copy_col")
    coords = coordinates if role == "a" else coordinates[::-1]
    expression = chain._Expression(cg, plan, {})
    expression.coordinate_names.update(coordinates)
    try:
        expression.value(operand, coords)
        if len(expression.loaded_inputs) != 1:
            return None
        loaded, leaf_coords, indices, _value = expression.loaded_inputs[0]
        if loaded is not leaf:
            return None
        mask = expression._load_mask(leaf, leaf_coords)
        assert mask is not None
        predicates = (mask, *chain._operand_domain(cg, operand, coords, plan))
    except chain._UnsupportedChain:
        return None
    uniform = set(expression.origins.values())
    # Reuse preparation-leaf ownership: only integer device arguments and
    # typed lexical captures may supplement CTA/grid origins. Unknown local
    # names and data-dependent loads are rejected by recursive expansion.
    uniform.update(
        argument.name
        for scalar, argument in cg.device_function._expr_args.items()
        if sympy.ask(sympy.Q.integer(scalar)) is True
    )
    if plan.loop is not None:
        uniform.update(
            name
            for captured, name in plan.loop.captures().items()
            if isinstance(captured.meta.get("val"), (int, torch.SymInt))
            or isinstance(captured.meta.get("val"), torch.Tensor)
            and captured.meta["val"].ndim == 0
            and captured.meta["val"].dtype in (torch.int32, torch.int64)
        )
    source = cast("Node", leaf.args[0])
    fake = source.meta["val"]
    proof = prove_rectangular_leaf(
        indices,
        expression.definitions,
        row=coordinates[1 - inner],
        column=coordinates[inner],
        uniform_names=uniform,
        tile_shape=(height, width),
        shape=chain._host_shape(fake),
        strides=tuple(fake.stride()),
        dtype=fake.dtype,
        mask=" & ".join(f"({item})" for item in predicates),
    )
    if proof is None:
        return None
    # Dense positive source strides and <2**31 total elements are part of the
    # rectangle proof. Its per-axis bounds therefore also prove every product
    # and partial sum in the *original* signed scalar pointer expression.
    tensor = expression.tensor_name(source)
    tag = f"{prefix}_{role}_masked_async"
    pointer = _cute_scalar_pointer_expr(tensor, list(proof.origin_indices))
    byte_span = proof.element_bytes * ((height - 1) * proof.pitch + width - 1)
    guards = [proof.guard]
    for axis, (size, step) in enumerate(zip(proof.shape, proof.strides, strict=True)):
        guards.extend(
            (
                f"{tensor}.shape[{axis}] == {size}",
                f"{tensor}.layout.stride[{axis}] == {step}",
            )
        )
    checks = " & ".join(f"({item})" for item in guards)
    fast = [
        f"{tag}_pointer = {pointer}",
        f"{tag}_address = cutlass.Int64({tag}_pointer.toint())",
        f"if ({tag}_address >= 0) & ({tag}_address <= {2**63 - 1 - byte_span}) & ({tag}_address % 16 == 0) & ({prefix}_{role}_ptr.toint() % 16 == 0):",
        f"    {tag}_source = cute.make_tensor({tag}_pointer.align(16), cute.make_layout(({height}, {width}), stride=({proof.pitch}, 1)))",
        (
            f"    {tag}_target = {native_destination.name}"
            if native_destination is not None
            else f"    {tag}_target = cute.make_tensor({prefix}_{role}_ptr, cute.make_layout(({height}, {width}), stride=({stride[1 - inner]}, 1)))"
        ),
        f"    {tag}_copy = cute.make_tiled_copy_tv(cute.make_copy_atom({copy_g2s_op(cg.device_function.config.config)}, {dtype}, num_bits_per_copy=128), cute.make_layout(({plan.threads // columns}, {columns}), stride=({columns}, 1)), cute.make_layout((1, 8)))",
        f"    {tag}_thread = {tag}_copy.get_slice(chain_thread)",
        f"    cute.copy({tag}_copy, {tag}_thread.partition_S({tag}_source), {tag}_thread.partition_D({tag}_target))",
        f"    {tag}_copied = cutlass.Boolean(True)",
    ]
    return [
        f"{tag}_copied = cutlass.Boolean(False)",
        f"if {checks}:",
        chain._indent(fast),
        f"if not {tag}_copied:",
        chain._indent(fallback),
    ]
