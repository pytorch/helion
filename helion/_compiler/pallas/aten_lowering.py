"""Pallas-backend ``register_codegen`` handlers for the aten lowerings whose
lowering objects live in ``helion/_compiler/aten_lowering.py``.

Backend-specific codegen bodies live here (not in the backend-neutral
``aten_lowering`` module).  Importing this module runs the
``@<op>_lowering.register_codegen("pallas")`` registrations; ``aten_lowering``
imports it at the bottom so registration keeps the same eager timing as before.
"""

from __future__ import annotations

import ast
import math
from operator import getitem
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch._inductor.codegen.simd import constant_repr
from torch.fx.node import Node
from torch.fx.node import map_arg

from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..aten_lowering import _env_arg
from ..aten_lowering import _node_dtype_kwarg
from ..aten_lowering import _pallas_argreduce
from ..aten_lowering import addmm_lowering
from ..aten_lowering import arange_default_lowering
from ..aten_lowering import argmax_lowering
from ..aten_lowering import argmin_lowering
from ..aten_lowering import baddbmm_lowering
from ..aten_lowering import bmm_lowering
from ..aten_lowering import constant_pad_nd_lowering
from ..aten_lowering import expand_lowering
from ..aten_lowering import iota_lowering
from ..aten_lowering import mm_lowering
from ..aten_lowering import permute_lowering
from ..aten_lowering import reshape_lowering
from ..aten_lowering import sort_lowering
from ..aten_lowering import squeeze_lowering
from ..aten_lowering import topk_lowering
from ..aten_lowering import unsqueeze_lowering
from ..aten_lowering import view_lowering
from ..aten_lowering import where_lowering
from ..compile_environment import CompileEnvironment
from ..matmul_utils import _emit_pallas_matmul
from ..matmul_utils import _needs_f32_accumulator
from . import codegen as pallas_codegen

if TYPE_CHECKING:
    from ..aten_lowering import LoweringContext


@argmax_lowering.register_codegen("pallas")
def codegen_argmax_pallas(ctx: LoweringContext, node: Node) -> ast.AST:
    return _pallas_argreduce(ctx, node, "argmax")


@argmin_lowering.register_codegen("pallas")
def codegen_argmin_pallas(ctx: LoweringContext, node: Node) -> ast.AST:
    return _pallas_argreduce(ctx, node, "argmin")


@where_lowering.register_codegen("pallas")
def codegen_where_pallas(ctx: LoweringContext, node: Node) -> object:
    env = CompileEnvironment.current()
    cond, x, y = map_arg(node.args, lambda arg: _env_arg(ctx, arg))

    def ensure_ast(value: object) -> ast.AST:
        if isinstance(value, ast.AST):
            return value
        if isinstance(value, (int, float, bool)):
            return expr_from_string(constant_repr(value))
        raise AssertionError(f"unsupported where operand: {type(value)!r}")

    cond_ast = ensure_ast(cond)
    x_ast = ensure_ast(x)
    y_ast = ensure_ast(y)
    ShapeDim = int | torch.SymInt

    def tensor_shape(arg: object) -> tuple[ShapeDim, ...] | None:
        if isinstance(arg, Node):
            value = arg.meta.get("val")
            if isinstance(value, torch.Tensor):
                return cast("tuple[ShapeDim, ...]", tuple(value.size()))
        return None

    def shapes_match(
        lhs: tuple[ShapeDim, ...] | None, rhs: tuple[ShapeDim, ...]
    ) -> bool:
        if lhs is None or len(lhs) != len(rhs):
            return False
        for left, right in zip(lhs, rhs, strict=True):
            if not env.known_equal(left, right):
                return False
        return True

    cond_arg = node.args[0]
    output_val = node.meta.get("val")
    if isinstance(cond_arg, Node) and isinstance(output_val, torch.Tensor):
        cond_val = cond_arg.meta.get("val")
        if isinstance(cond_val, torch.Tensor) and cond_val.dtype is torch.bool:
            output_shape = cast("tuple[ShapeDim, ...]", tuple(output_val.size()))
            branch_asts = ((node.args[1], x_ast), (node.args[2], y_ast))
            layout_anchor = next(
                (
                    branch_ast
                    for branch_arg, branch_ast in branch_asts
                    if shapes_match(tensor_shape(branch_arg), output_shape)
                ),
                next(
                    (
                        branch_ast
                        for branch_arg, branch_ast in branch_asts
                        if tensor_shape(branch_arg) is not None
                    ),
                    x_ast,
                ),
            )
            numeric_select = pallas_codegen.numeric_where_expr(
                cond_ast,
                x_ast,
                y_ast,
                output_val.dtype,
                layout_anchor,
            )
            if numeric_select is not None:
                return numeric_select
            if not shapes_match(tuple(cond_val.size()), output_shape):
                cond_ast = pallas_codegen.layout_tied_bf16_mask_expr(
                    cond_ast,
                    layout_anchor,
                )
                cond_ast = expr_from_string(
                    "({cond}) == jnp.array(1, dtype=jnp.bfloat16)",
                    cond=cond_ast,
                )
            else:
                cond_ast = expr_from_string("({cond}) != 0", cond=cond_ast)

    return expr_from_string(
        env.backend.where_expr("{cond}", "{x}", "{y}"),
        cond=cond_ast,
        x=x_ast,
        y=y_ast,
    )


def _pallas_bool_safe_singleton_index(
    tensor: ast.AST,
    input_val: torch.Tensor,
    args: list[str],
    singleton_shape: str | None = None,
) -> ast.AST:
    index = ", ".join(args)
    if input_val.dtype is torch.bool and "None" in args:
        # Mosaic cannot reshape bool vectors directly (jax-ml/jax#37370).
        if singleton_shape is not None:
            return pallas_codegen.numeric_mask_reshape_expr(tensor, singleton_shape)
        tensor = pallas_codegen.numeric_mask_expr(tensor)
    return expr_from_string(f"{{tensor}}[{index}]", tensor=tensor)


def _pallas_singleton_shape(
    input_val: torch.Tensor,
    args: list[str],
    ctx: LoweringContext,
    *,
    compacted_args: bool = False,
) -> str:
    tile_strategy = ctx.cg.device_function.tile_strategy
    if compacted_args:
        input_dims = iter(tile_strategy.shape_dims([*input_val.size()]))
        shape_dims = ["1" if arg == "None" else next(input_dims) for arg in args]
        return f"[{', '.join(shape_dims)}]"

    input_dims = iter(input_val.size())
    shape = [1 if arg == "None" else next(input_dims) for arg in args]
    return tile_strategy.shape_str(shape)


@unsqueeze_lowering.register_codegen("pallas")
def codegen_unsqueeze_pallas(ctx: LoweringContext, node: Node) -> object:
    assert not node.kwargs, "unsqueeze kwargs not supported"
    tensor, dim = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    assert isinstance(dim, int)
    input_node = node.args[0]
    assert isinstance(input_node, Node)
    input_val = input_node.meta["val"]
    assert isinstance(input_val, torch.Tensor)
    ndim = input_val.ndim
    if dim < 0:
        dim += ndim + 1
    assert 0 <= dim <= ndim, f"Invalid dim {dim} for tensor with {ndim} dims"
    args = [":"] * ndim
    args.insert(dim, "None")
    singleton_shape = _pallas_singleton_shape(input_val, args, ctx)
    return _pallas_bool_safe_singleton_index(tensor, input_val, args, singleton_shape)


@squeeze_lowering.register_codegen("pallas")
@view_lowering.register_codegen("pallas")
@reshape_lowering.register_codegen("pallas")
def codegen_view_pallas(ctx: LoweringContext, node: Node) -> object:
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    shape_str = ctx.cg.device_function.tile_strategy.shape_str(
        [*node.meta["val"].size()]
    )
    input_node = node.args[0]
    if isinstance(input_node, Node):
        input_val = input_node.meta.get("val")
        if isinstance(input_val, torch.Tensor) and input_val.dtype is torch.bool:
            tensor = pallas_codegen.numeric_mask_expr(tensor)
    return expr_from_string(f"jnp.reshape({{tensor}}, {shape_str})", tensor=tensor)


def _pad_fill_literal(value: object) -> str:
    """Render a ``constant_pad_nd`` fill value as valid Python source.

    ``repr()`` is wrong for the non-finite floats: it yields bare ``inf`` /
    ``-inf`` / ``nan``, which are not names in the generated module (the
    artifact raises ``NameError: name 'inf' is not defined``). Spell those as
    ``float('inf')`` etc., which is valid in any module and needs no import.
    """
    if isinstance(value, float) and not math.isfinite(value):
        # repr() of the value is itself the bare name (``-inf``), so quote the
        # str() form: float('-inf') / float('inf') / float('nan').
        return f"float({str(value)!r})"
    return repr(value)


@constant_pad_nd_lowering.register_codegen("pallas")
def codegen_constant_pad_nd_pallas(ctx: LoweringContext, node: Node) -> object:
    """``F.pad(x, pad, value)`` (aten.constant_pad_nd) -> inline ``jnp.pad``.

    Mosaic lacks a direct constant_pad_nd lowering, so emit a fused ``jnp.pad``.
    ``pad`` is aten's flat, from-the-last-dim format
    ``[last_lo, last_hi, 2ndlast_lo, 2ndlast_hi, ...]``; convert to jnp's per-dim
    ``((lo, hi), ...)`` ordered from the first dim. Used e.g. to pad a top-k
    output up to a 128-aligned width so the store is Mosaic-tile-aligned on jax
    0.10.0 without computing a wider top-k.
    """
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    pad = [int(p) for p in cast("list[int]", node.args[1])]  # static pad widths
    value = node.args[2] if len(node.args) > 2 else node.kwargs.get("value", 0)
    ndim = node.meta["val"].ndim
    npad = len(pad) // 2
    pad_width = [
        (pad[2 * (ndim - 1 - j)], pad[2 * (ndim - 1 - j) + 1])
        if (ndim - 1 - j) < npad
        else (0, 0)
        for j in range(ndim)
    ]
    pw = "(" + ", ".join(f"({lo}, {hi})" for lo, hi in pad_width) + ")"
    return expr_from_string(
        f"jnp.pad({{t}}, {pw}, mode='constant', constant_values={_pad_fill_literal(value)})",
        t=tensor,
    )


@permute_lowering.register_codegen("pallas")
def codegen_permute_pallas(ctx: LoweringContext, node: Node) -> object:
    from .codegen import maybe_codegen_resident_prep_cache_read

    resident_prep_read = maybe_codegen_resident_prep_cache_read(ctx, node)
    if resident_prep_read is not None:
        return resident_prep_read
    tensor, dims = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    # pyrefly: ignore [not-iterable]
    dims = [*dims]
    return expr_from_string(
        f"jnp.transpose({{tensor}}, {dims!r})",
        tensor=tensor,
    )


@expand_lowering.register_codegen("pallas")
def codegen_expand_pallas(ctx: LoweringContext, node: Node) -> object:
    tensor, _ = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    val = node.meta["val"]
    assert isinstance(val, torch.Tensor)
    shape = [*val.size()]
    # pyrefly: ignore [missing-attribute]
    input_val = node.args[0].meta["val"]
    assert isinstance(input_val, torch.Tensor)
    if input_val.ndim != len(shape):
        tile_strategy = ctx.cg.device_function.tile_strategy
        broadcasting = tile_strategy.broadcast_expand_dims(
            tuple(input_val.shape), tuple(shape)
        )
        if broadcasting:
            singleton_shape = _pallas_singleton_shape(
                input_val, broadcasting, ctx, compacted_args=True
            )
            tensor = _pallas_bool_safe_singleton_index(
                tensor, input_val, broadcasting, singleton_shape
            )
    elif input_val.dtype is torch.bool:
        tensor = pallas_codegen.numeric_mask_expr(tensor)
    shape_str = ctx.cg.device_function.tile_strategy.shape_str(shape)
    return expr_from_string(
        f"jnp.broadcast_to({{tensor}}, {shape_str})",
        tensor=tensor,
    )


def _pallas_dot(ctx: LoweringContext, node: Node, with_acc: bool) -> ast.AST:
    """Generate jnp.dot_general for Pallas backend."""
    if with_acc:
        acc_node_arg, lhs_node_arg, rhs_node_arg = node.args[:3]
        acc, lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
        assert isinstance(acc, ast.AST)
        assert isinstance(lhs, ast.AST)
        assert isinstance(rhs, ast.AST)
    else:
        lhs_node_arg, rhs_node_arg = node.args[:2]
        lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
        assert isinstance(lhs, ast.AST)
        assert isinstance(rhs, ast.AST)
        acc = None

    assert isinstance(lhs_node_arg, Node)
    assert isinstance(rhs_node_arg, Node)
    lhs_dtype = lhs_node_arg.meta["val"].dtype
    rhs_dtype = rhs_node_arg.meta["val"].dtype
    lhs_ndim = lhs_node_arg.meta["val"].ndim
    need_f32_acc = _needs_f32_accumulator(lhs_dtype, rhs_dtype)
    out_dtype = node.meta["val"].dtype if "val" in node.meta else None

    return _emit_pallas_matmul(
        lhs,
        rhs,
        acc=acc if with_acc else None,
        need_f32_acc=need_f32_acc,
        out_dtype=out_dtype,
        lhs_ndim=lhs_ndim,
    )


@bmm_lowering.register_codegen("pallas")
@mm_lowering.register_codegen("pallas")
def codegen_mm_pallas(ctx: LoweringContext, node: Node) -> ast.AST:
    return _pallas_dot(ctx, node, False)


@addmm_lowering.register_codegen("pallas")
def codegen_addmm_pallas(ctx: LoweringContext, node: Node) -> ast.AST:
    return _pallas_dot(ctx, node, True)


@baddbmm_lowering.register_codegen("pallas")
def codegen_baddbmm_pallas(ctx: LoweringContext, node: Node) -> ast.AST:
    return _pallas_dot(ctx, node, True)


def _pallas_iota_expr(
    ctx: LoweringContext,
    *,
    length_arg: object,
    start: object = 0,
    step: object = 1,
    dtype: torch.dtype | None = None,
) -> object:
    dtype = dtype or CompileEnvironment.current().index_dtype
    assert isinstance(dtype, torch.dtype)

    dtype_str = CompileEnvironment.current().backend.dtype_str(dtype)
    expr = f"jnp.arange(0, {{length}}, dtype={dtype_str})"
    if step != 1:
        expr = f"{{step}} * {expr}"
    if start != 0:
        expr = f"{{start}} + {expr}"
    return expr_from_string(
        expr,
        start=ctx.to_ast(start),
        step=ctx.to_ast(step),
        length=ctx.to_ast(length_arg),
    )


@iota_lowering.register_codegen("pallas")
def codegen_iota_pallas(ctx: LoweringContext, node: Node) -> object:
    """Generate jnp.arange for torch.ops.prims.iota.default on Pallas."""
    return _pallas_iota_expr(
        ctx,
        length_arg=node.args[0],
        start=node.kwargs.get("start", 0),
        step=node.kwargs.get("step", 1),
        dtype=_node_dtype_kwarg(node),
    )


@arange_default_lowering.register_codegen("pallas")
def codegen_arange_default_pallas(ctx: LoweringContext, node: Node) -> object:
    return _pallas_iota_expr(
        ctx,
        length_arg=node.args[0],
        dtype=_node_dtype_kwarg(node),
    )


def _pallas_last_dim(node: Node, dim: object) -> None:
    """Assert an aten sort/topk reduces the last dim (all we support on Pallas)."""
    input_node = node.args[0]
    assert isinstance(input_node, Node)
    ndim = input_node.meta["val"].ndim
    assert isinstance(dim, int)
    norm = ndim + dim if dim < 0 else dim
    assert norm == ndim - 1, (
        f"pallas sort/topk only supports the last dim, got dim={dim} (ndim={ndim})"
    )


@topk_lowering.register_codegen("pallas")
def codegen_topk_pallas(ctx: LoweringContext, node: Node) -> object:
    """``torch.topk(x, k, dim=-1, largest=True, sorted=True)`` on Mosaic/TPU.

    ``jax.lax.top_k`` is unimplemented in the Mosaic TPU lowering, so we emit
    ``topk_impl.divide_filter_topk`` -- a tallax-style divide-and-filter top-k
    built only from Mosaic-supported ops (strided slices, iota, where, min/max).
    Being plain jnp emitted inline, it FUSES with the surrounding kernel. Returns
    ``(values desc, int32 indices)`` over the LAST axis (approximate, recall~0.99;
    top-1 exact). ``k`` must be static (use ``hl.specialize(k)``).
    """
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    k = node.args[1]
    if not isinstance(k, int):
        # k can arrive as an fx Node / SymInt rather than a Python int; recover it
        # from the (static) last dim of the output. Requires a static k -- use
        # hl.specialize(k) in the kernel. (The jax_fn path already gives an int.)
        try:
            val = node.meta["val"]
            out0 = val[0] if isinstance(val, (tuple, list)) else val
            k = int(out0.shape[-1])
        except Exception:
            pass
    assert isinstance(k, int), (
        f"pallas topk requires a static int k (use hl.specialize(k)); got {type(k)!r}"
    )
    dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", -1)
    largest = node.args[3] if len(node.args) > 3 else node.kwargs.get("largest", True)
    assert largest, "pallas topk only supports largest=True"
    _pallas_last_dim(node, dim)

    result = ctx.cg.device_function.new_var("topk_result")
    ctx.cg.add_statement(
        statement_from_string(
            f"{result} = _helion_divide_filter_topk({{t}}, {k})", t=tensor
        )
    )
    # torch.topk returns (values, indices); skip the indices expr when unused.
    indices_used = any(
        user.target is getitem and user.args[1] == 1 for user in node.users
    )
    values = expr_from_string(f"{result}[0]")
    indices = expr_from_string(f"{result}[1]") if indices_used else None
    return (values, indices)


@sort_lowering.register_codegen("pallas")
def codegen_sort_pallas(ctx: LoweringContext, node: Node) -> object:
    """``torch.sort(x, dim=-1, descending=False)`` via ``jax.lax.sort``.

    Co-sorts ``(x, iota)`` so the permuted iota gives the indices (argsort), which
    matches torch.sort's (values, indices). Last axis only.
    """
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
    descending = (
        node.args[2] if len(node.args) > 2 else node.kwargs.get("descending", False)
    )
    _pallas_last_dim(node, dim)
    input_node = node.args[0]
    assert isinstance(input_node, Node)
    n = int(input_node.meta["val"].shape[-1])

    indices_used = any(
        user.target is getitem and user.args[1] == 1 for user in node.users
    )
    if not indices_used:
        expr = "jnp.sort({t}, axis=-1)"
        if descending:
            expr = f"jnp.flip({expr}, axis=-1)"
        return (expr_from_string(expr, t=tensor), None)

    # Co-sort values with an index iota to recover argsort indices.
    result = ctx.cg.device_function.new_var("sort_result")
    key = "-({t})" if descending else "{t}"
    ctx.cg.add_statement(
        statement_from_string(
            f"{result} = jax.lax.sort(({key}, "
            f"jnp.broadcast_to(jnp.arange({n}, dtype=jnp.int32), ({{t}}).shape)), "
            f"dimension=-1, num_keys=1)",
            t=tensor,
        )
    )
    values = f"(-{result}[0])" if descending else f"{result}[0]"
    return (expr_from_string(values), expr_from_string(f"{result}[1]"))
