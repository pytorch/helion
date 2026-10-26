"""Triton-backend ``register_codegen`` handlers for the aten lowerings whose
lowering objects live in ``helion/_compiler/aten_lowering.py``.

Backend-specific codegen bodies live here (not in the backend-neutral
``aten_lowering`` module).  Importing this module runs the
``@<op>_lowering.register_codegen("triton")`` registrations; ``aten_lowering``
imports it at the bottom so registration keeps the same eager timing as before.
"""

from __future__ import annotations

import ast
from operator import getitem
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._inductor.utils import triton_type
from torch.fx.node import Node
from torch.fx.node import map_arg

from ... import exc
from ..._compat import is_hip
from ..._utils import next_power_of_2
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..aten_lowering import _env_arg
from ..aten_lowering import _node_dtype_kwarg
from ..aten_lowering import _triton_argreduce
from ..aten_lowering import addmm_lowering
from ..aten_lowering import arange_default_lowering
from ..aten_lowering import argmax_lowering
from ..aten_lowering import argmin_lowering
from ..aten_lowering import baddbmm_lowering
from ..aten_lowering import bmm_lowering
from ..aten_lowering import expand_lowering
from ..aten_lowering import gather_lowering
from ..aten_lowering import iota_lowering
from ..aten_lowering import mm_lowering
from ..aten_lowering import permute_lowering
from ..aten_lowering import reshape_lowering
from ..aten_lowering import sort_lowering
from ..aten_lowering import squeeze_lowering
from ..aten_lowering import stack_lowering
from ..aten_lowering import topk_lowering
from ..aten_lowering import view_dtype_lowering
from ..aten_lowering import view_lowering
from ..compile_environment import CompileEnvironment
from ..compile_environment import _symint_sympy_expr
from ..device_function import find_block_size_symbols
from ..matmul_utils import emit_tl_dot_with_padding

if TYPE_CHECKING:
    from ..aten_lowering import LoweringContext


@argmax_lowering.register_codegen("triton")
def codegen_argmax(ctx: LoweringContext, node: Node) -> ast.AST:
    return _triton_argreduce(ctx, node, "argmax")


@argmin_lowering.register_codegen("triton")
def codegen_argmin(ctx: LoweringContext, node: Node) -> ast.AST:
    return _triton_argreduce(ctx, node, "argmin")


@squeeze_lowering.register_codegen("triton")
@view_lowering.register_codegen("triton")
@reshape_lowering.register_codegen("triton")
def codegen_view(ctx: LoweringContext, node: Node) -> object:
    assert not node.kwargs, "view kwargs not supported"
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    shape_str = ctx.cg.device_function.tile_strategy.shape_str(
        [*node.meta["val"].size()]
    )
    return expr_from_string(f"tl.reshape({{tensor}}, {shape_str})", tensor=tensor)


@view_dtype_lowering.register_codegen("triton")
def codegen_view_dtype(ctx: LoweringContext, node: Node) -> object:
    """Generate tl.cast with bitcast=True for dtype reinterpretation."""
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    target_dtype = node.args[1]
    assert isinstance(target_dtype, torch.dtype)
    return expr_from_string(
        f"tl.cast({{tensor}}, {triton_type(target_dtype)}, bitcast=True)",
        tensor=tensor,
    )


@permute_lowering.register_codegen("triton")
def codegen_permute(ctx: LoweringContext, node: Node) -> object:
    assert not node.kwargs, "getitem kwargs not supported"
    tensor, dims = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    # pyrefly: ignore [not-iterable]
    dims = [*dims]
    assert {*dims} == {*range(len(dims))}, dims
    input_node = node.args[0]
    assert isinstance(input_node, Node)
    input_val = input_node.meta["val"]
    assert isinstance(input_val, torch.Tensor)
    physical_rank = len(
        ctx.cg.device_function.tile_strategy.shape_dims(input_val.shape)
    )
    if physical_rank != len(dims):
        raise exc.InvalidConfig(
            "torch.permute does not support rank-compacted tile inputs. "
            "Disable flatten_loops for the affected tile axes."
        )
    return expr_from_string(
        f"tl.permute({{tensor}}, {dims!r})",
        tensor=tensor,
    )


@stack_lowering.register_codegen("triton")
def codegen_stack(ctx: LoweringContext, node: Node) -> object:
    tensors = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)

    assert isinstance(tensors, (list, tuple))
    # pyrefly: ignore [bad-index]
    tensor_asts = [ctx.env[t] for t in tensors]
    n = len(tensor_asts)

    if n == 0:
        raise ValueError("Cannot stack empty tensor list")

    # Round up to power of 2 for efficient masking
    padded_size = 1 << (n - 1).bit_length()

    # Create index array [0, 1, 2, 3, ...] for tensor selection
    idx = ctx.cg.device_function.new_var("stack_idx")
    ctx.cg.add_statement(statement_from_string(f"{idx} = tl.arange(0, {padded_size})"))

    # Map the logical stack axis to its position after tile compaction.
    input_node = tensors[0]
    assert isinstance(input_node, Node)
    input_val = input_node.meta["val"]
    assert isinstance(input_val, torch.Tensor)
    output_val = node.meta["val"]
    assert isinstance(output_val, torch.Tensor)
    assert isinstance(dim, int)
    logical_rank = input_val.ndim
    if dim < 0:
        dim += logical_rank + 1
    tile_strategy = ctx.cg.device_function.tile_strategy
    input_compacted = tile_strategy.compact_shape(input_val.shape)
    output_compacted = tile_strategy.compact_shape(output_val.shape)
    new_axes = [
        axis
        for axis, compacted in enumerate(output_compacted)
        if compacted.user_indices == [dim]
    ]
    shifted_input = [
        [index if index < dim else index + 1 for index in compacted.user_indices]
        for compacted in input_compacted
    ]
    remaining_output = [
        compacted.user_indices
        for axis, compacted in enumerate(output_compacted)
        if axis not in new_axes
    ]
    if len(new_axes) != 1 or remaining_output != shifted_input:
        raise exc.InvalidConfig(
            "torch.stack requires a stack axis representable by one physical "
            "dimension. Disable flatten_loops for the affected tile axes."
        )
    stack_dim = new_axes[0]
    bidx = ctx.cg.device_function.new_var("broadcast_idx")
    pattern = (
        "["
        + ", ".join(
            ["None"] * stack_dim
            + [":"]
            + ["None"] * (len(output_compacted) - stack_dim - 1)
        )
        + "]"
    )
    ctx.cg.add_statement(statement_from_string(f"{bidx} = {idx}{pattern}"))

    # Expand each input tensor along the stack dimension
    expanded = [ctx.cg.device_function.new_var(f"expanded_{i}") for i in range(n)]
    for var, tensor in zip(expanded, tensor_asts, strict=False):
        tensor_ast = cast("ast.AST", tensor)
        ctx.cg.add_statement(
            statement_from_string(
                f"{var} = tl.expand_dims({{t}}, {stack_dim})", t=tensor_ast
            )
        )

    # Initialize result with zeros
    result = ctx.cg.device_function.new_var("stacked_result")
    ctx.cg.add_statement(
        statement_from_string(f"{result} = tl.zeros_like({expanded[0]})")
    )

    # Select each tensor using masks
    for i in range(n):
        mask = ctx.cg.device_function.new_var(f"mask_{i}")
        ctx.cg.add_statement(statement_from_string(f"{mask} = {bidx} == {i}"))
        ctx.cg.add_statement(
            statement_from_string(
                f"{result} = tl.where({mask}, {expanded[i]}, {result})"
            )
        )

    return expr_from_string(result)


@expand_lowering.register_codegen("triton")
def codegen_expand(ctx: LoweringContext, node: Node) -> object:
    assert not node.kwargs, "expand kwargs not supported"
    tensor, _ = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)
    val = node.meta["val"]
    assert isinstance(val, torch.Tensor)
    shape = [*val.size()]
    # pyrefly: ignore [missing-attribute]
    input_val = node.args[0].meta["val"]
    if input_val.ndim != len(shape):
        tile_strategy = ctx.cg.device_function.tile_strategy
        broadcasting = tile_strategy.broadcast_expand_dims(
            tuple(input_val.shape), tuple(shape)
        )
        if broadcasting:
            tensor = expr_from_string(
                f"{{tensor}}[{', '.join(broadcasting)}]", tensor=tensor
            )
    shape_str = ctx.cg.device_function.tile_strategy.shape_str(shape)
    return expr_from_string(
        f"tl.broadcast_to({{tensor}}, {shape_str})",
        tensor=tensor,
    )


def reduce_3d_dot(ctx: LoweringContext, node: Node, with_acc: bool) -> ast.AST:
    acc = None
    acc_node: Node | None = None
    if with_acc:
        acc, lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
        assert isinstance(acc, ast.AST)
        assert isinstance(node.args[0], Node)
        acc_node = node.args[0]
        lhs_node = node.args[1]
        rhs_node = node.args[2]
    else:
        lhs, rhs = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
        lhs_node = node.args[0]
        rhs_node = node.args[1]
    assert isinstance(lhs, ast.AST)
    assert isinstance(rhs, ast.AST)
    assert isinstance(lhs_node, Node)
    assert isinstance(rhs_node, Node)

    # Check if inputs are FP8 - if so, redirect user to hl.dot()
    lhs_dtype = lhs_node.meta["val"].dtype
    rhs_dtype = rhs_node.meta["val"].dtype
    acc_dtype_meta: torch.dtype | None = None
    if with_acc:
        assert acc_node is not None
        assert isinstance(acc_node, Node)
        acc_dtype_meta = acc_node.meta["val"].dtype
    if lhs_dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and rhs_dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ]:
        raise NotImplementedError(
            "FP8 GEMM via torch API is not supported yet. Please use hl.dot() instead."
        )

    lhs_shape = list(lhs_node.meta["val"].size())
    rhs_shape = list(rhs_node.meta["val"].size())
    acc_shape = (
        list(acc_node.meta["val"].size())
        if (with_acc and acc_node is not None)
        else None
    )

    # Extract expected output dtype from FX node to match PyTorch eager mode behavior
    out_dtype: torch.dtype | None = None
    if "val" in node.meta and isinstance(node.meta["val"], torch.Tensor):
        out_dtype = node.meta["val"].dtype

    return emit_tl_dot_with_padding(
        lhs,
        rhs,
        acc if with_acc else None,
        lhs_dtype,
        rhs_dtype,
        acc_dtype=acc_dtype_meta if with_acc else None,
        out_dtype=out_dtype,
        lhs_shape=lhs_shape,
        rhs_shape=rhs_shape,
        acc_shape=acc_shape,
        lhs_fx_node=lhs_node,
    )


@bmm_lowering.register_codegen("triton")
@mm_lowering.register_codegen("triton")
def codegen_mm(ctx: LoweringContext, node: Node) -> ast.AST:
    assert not node.kwargs, "matmul kwargs not supported"

    return reduce_3d_dot(ctx, node, False)


@addmm_lowering.register_codegen("triton")
def codegen_addmm(ctx: LoweringContext, node: Node) -> ast.AST:
    assert not node.kwargs, "addmm kwargs not supported"
    return reduce_3d_dot(ctx, node, True)


@baddbmm_lowering.register_codegen("triton")
def codegen_baddbmm(ctx: LoweringContext, node: Node) -> ast.AST:
    assert not node.kwargs, "baddbmm kwargs not supported"
    return reduce_3d_dot(ctx, node, True)


def _triton_iota_expr(
    ctx: LoweringContext,
    *,
    length_arg: object,
    start: object = 0,
    step: object = 1,
    dtype: torch.dtype | None = None,
) -> object:
    from ..generate_ast import GenerateAST

    dtype = dtype or CompileEnvironment.current().index_dtype
    assert isinstance(dtype, torch.dtype)

    # FX scalar nodes can retain expressions derived entirely from tunable
    # block sizes. Resolve only those proven config constants here: their
    # ordinary scalar rendering may be a runtime host argument, whereas arange
    # requires a power-of-two compile-time extent. Keep the logical expression
    # intact elsewhere so loads, stores and reductions retain their tail masks.
    length = length_arg.meta["val"] if isinstance(length_arg, Node) else length_arg
    env = CompileEnvironment.current()
    block_id = (
        env.resolve_block_id(length) if isinstance(length, torch.SymInt) else None
    )
    expr: str | None = None
    reused_coordinate = False
    if (
        block_id is not None
        and env.block_sizes[block_id].reduction
        and isinstance(ctx.cg, GenerateAST)
        and ctx.cg.active_device_loops.get(block_id)
    ):
        # A full-axis iota shares the reduction's physical tile and logical
        # coordinate. Reusing that coordinate preserves both padded constexpr
        # capacity and the offset of a rolled reduction's later tiles.
        expr = ctx.cg.index_var(block_id)
        reused_coordinate = True
    elif isinstance(length, torch.SymInt):
        expression = _symint_sympy_expr(length)
        block_symbols, runtime_symbols = find_block_size_symbols(expression)
        replacements: dict[sympy.Basic, sympy.Basic] = {}
        if not runtime_symbols:
            for symbol, block_id in block_symbols.items():
                if env.block_sizes[block_id].reduction:
                    break
                value = ctx.cg.device_function.resolved_block_size(block_id)
                if not isinstance(value, int):
                    break
                if expression == symbol and value == next_power_of_2(value):
                    # A direct block extent already has a constexpr argument,
                    # which can also be used outside a hl.tile loop.
                    block_var = ctx.cg.device_function.block_size_var(
                        env.canonical_block_id(block_id)
                    )
                    assert block_var is not None
                    expr = f"tl.arange(0, {block_var})"
                    break
                replacements[symbol] = sympy.Integer(value)
            else:
                resolved = expression.xreplace(replacements)
                if isinstance(resolved, sympy.Integer):
                    length = int(resolved)

    if expr is None:
        # Pad literal and config-derived non-power-of-two lengths alike.
        length_expr = "{length}"
        if isinstance(length, int):
            length_expr = str(next_power_of_2(length))
        expr = f"tl.arange(0, {length_expr})"
    if step != 1:
        expr = f"{{step}} * {expr}"
    if start != 0:
        expr = f"{{start}} + {expr}"
    if reused_coordinate or dtype != torch.int32:
        expr = f"({expr}).to({triton_type(dtype)})"
    return expr_from_string(
        expr,
        start=ctx.to_ast(start),
        step=ctx.to_ast(step),
        length=ctx.to_ast(length_arg),
    )


@iota_lowering.register_codegen("triton")
def codegen_iota(ctx: LoweringContext, node: Node) -> object:
    """Generate tl.arange for torch.ops.prims.iota.default operations with automatic power-of-2 padding."""
    return _triton_iota_expr(
        ctx,
        length_arg=node.args[0],
        start=node.kwargs.get("start", 0),
        step=node.kwargs.get("step", 1),
        dtype=_node_dtype_kwarg(node),
    )


@arange_default_lowering.register_codegen("triton")
def codegen_arange_default(ctx: LoweringContext, node: Node) -> object:
    return _triton_iota_expr(
        ctx,
        length_arg=node.args[0],
        dtype=_node_dtype_kwarg(node),
    )


@sort_lowering.register_codegen("triton")
def codegen_sort(ctx: LoweringContext, node: Node) -> object:
    """Generate tl.sort-based sort implementation.

    torch.sort(input, dim=-1, descending=False, stable=False) returns (values, indices).
    We implement this using tl.sort for values.
    For indices, we compute the rank of each element to determine its sorted position.

    Note: tl.sort only works on the last dimension currently.
    """
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)

    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
    descending = (
        node.args[2] if len(node.args) > 2 else node.kwargs.get("descending", False)
    )
    # stable arg (node.args[3]) is ignored - tl.sort is stable

    assert isinstance(dim, int), f"sort dim must be int, got {type(dim)}"
    assert isinstance(descending, bool), (
        f"sort descending must be bool, got {type(descending)}"
    )

    # Get the input tensor shape info
    input_val = node.args[0]
    assert isinstance(input_val, Node)
    input_tensor = input_val.meta["val"]
    ndim = input_tensor.ndim

    # Normalize negative dim
    if dim < 0:
        dim = ndim + dim

    # tl.sort only supports sorting on the last dimension
    assert dim == ndim - 1, (
        f"tl.sort only supports sorting on last dimension, got dim={dim}"
    )

    descending_str = "True" if descending else "False"

    # Generate sorted values using tl.sort
    sorted_vals = ctx.cg.device_function.new_var("sorted_vals")
    ctx.cg.add_statement(
        statement_from_string(
            f"{sorted_vals} = tl.sort({{tensor}}, descending={descending_str})",
            tensor=tensor,
        )
    )

    # Skip O(N^2) argsort when indices are not used downstream
    indices_used = any(
        user.target is getitem and user.args[1] == 1 for user in node.users
    )
    if not indices_used:
        return (expr_from_string(sorted_vals), None)

    # For indices, compute argsort using ranking:
    # For each element x[..., i], its rank is count of elements strictly less (or greater for descending)
    # plus count of equal elements with smaller index (for stability).
    # rank[..., i] gives the sorted position of x[..., i], so we need to invert this.
    sorted_indices = ctx.cg.device_function.new_var("sorted_indices")
    rank = ctx.cg.device_function.new_var("rank")
    idx_var = ctx.cg.device_function.new_var("idx")

    # Get size of last dimension (must be power of 2 for tl.sort)
    n = input_tensor.shape[-1]
    env = CompileEnvironment.current()
    n_hint = env.size_hint(n) if isinstance(n, torch.SymInt) else n
    n_pow2 = next_power_of_2(n_hint)

    # Create indices: [0, 1, 2, ..., n-1]
    ctx.cg.add_statement(statement_from_string(f"{idx_var} = tl.arange(0, {n_pow2})"))

    # Set up dimension-specific indexing patterns and comparison operator
    cmp_op = ">" if descending else "<"
    if ndim == 1:
        # 1D: compare [1, n] with [n, 1], reduce over axis 1
        t_a, t_b = "[None, :]", "[:, None]"
        i_a, i_b = "[None, :]", "[:, None]"
        reduce_axis = 1
        # For inverting: [n, 1] == [1, n], reduce axis 0
        r_a, r_b, inv_i_a, _inv_i_b, inv_axis = (
            "[:, None]",
            "[None, :]",
            "[:, None]",
            "[None, :]",
            0,
        )
    elif ndim == 2:
        # 2D: compare [batch, 1, n] with [batch, n, 1], reduce over axis 2
        t_a, t_b = "[:, None, :]", "[:, :, None]"
        i_a, i_b = "[None, None, :]", "[None, :, None]"
        reduce_axis = 2
        # For inverting: [batch, n, 1] == [1, 1, n], reduce axis 1
        r_a, r_b, inv_i_a, _inv_i_b, inv_axis = (
            "[:, :, None]",
            "[None, None, :]",
            "[None, :, None]",
            "[None, None, :]",
            1,
        )
    else:
        raise NotImplementedError

    # Compute rank: count elements that should come before + tie-breaking
    ctx.cg.add_statement(
        statement_from_string(
            f"{rank} = tl.sum(tl.where({{tensor}}{t_a} {cmp_op} {{tensor}}{t_b}, 1, 0), axis={reduce_axis}) + "
            f"tl.sum(tl.where(({{tensor}}{t_a} == {{tensor}}{t_b}) & ({idx_var}{i_a} < {idx_var}{i_b}), 1, 0), axis={reduce_axis})",
            tensor=tensor,
        )
    )

    # Invert the rank permutation: sorted_indices[rank[i]] = i
    ctx.cg.add_statement(
        statement_from_string(
            f"{sorted_indices} = tl.sum(tl.where({rank}{r_a} == {idx_var}{r_b}, {idx_var}{inv_i_a}, 0), axis={inv_axis})"
        )
    )

    # Return as tuple (values, indices)
    return (expr_from_string(sorted_vals), expr_from_string(sorted_indices))


@gather_lowering.register_codegen("triton")
def codegen_gather(ctx: LoweringContext, node: Node) -> object:
    """Generate gather implementation using tl.gather.

    torch.gather(input, dim, index) gathers values along dim using index.
    Both input and index must be already-loaded tiles (not host tensors).
    Uses Triton's tl.gather for the actual gather operation.
    """
    # Validate arguments
    assert not node.kwargs, "gather does not support keyword arguments"
    assert len(node.args) == 3, f"gather expects 3 arguments, got {len(node.args)}"

    input_node = node.args[0]
    dim = node.args[1]
    index_node = node.args[2]

    assert isinstance(input_node, Node), "gather input must be a Node"
    assert isinstance(dim, int), f"gather dim must be int, got {type(dim)}"
    assert isinstance(index_node, Node), "gather index must be a Node"

    input_tensor = input_node.meta["val"]

    # Validate that input is a tensor
    assert isinstance(input_tensor, torch.Tensor), (
        f"gather input must be a tensor, got {type(input_tensor)}"
    )

    ndim = input_tensor.ndim

    # Normalize negative dim
    if dim < 0:
        dim = ndim + dim

    # Validate dim is in range
    assert 0 <= dim < ndim, (
        f"gather dim {dim} out of range for tensor with {ndim} dimensions"
    )

    fn = ctx.cg.device_function

    # Get the input and index AST nodes
    input_ast_raw = _env_arg(ctx, input_node)
    assert isinstance(input_ast_raw, ast.AST)
    input_ast = input_ast_raw

    index_ast_raw = _env_arg(ctx, index_node)
    assert isinstance(index_ast_raw, ast.AST)
    index_ast = index_ast_raw

    result_var = fn.new_var("gather_result")

    ctx.cg.add_statement(
        statement_from_string(
            f"{result_var} = tl.gather({{input}}, {{index}}.to(tl.int32), axis={dim})",
            input=input_ast,
            index=index_ast,
        )
    )

    return expr_from_string(result_var)


@topk_lowering.register_codegen("triton")
def codegen_topk(ctx: LoweringContext, node: Node) -> object:
    """Generate tl.topk-based topk implementation.

    torch.topk(input, k, dim=-1, largest=True, sorted=True) returns (values, indices).
    FP16/BF16/FP32 inputs select ordered value/index keys and gather the
    original payloads. Other dtypes retain value selection and rank-based indices.

    Note: tl.topk/tl.sort only works on the last dimension currently.
    See: https://github.com/triton-lang/triton/blob/main/python/triton/language/standard.py
    """
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)

    k = node.args[1]
    assert isinstance(k, int), f"topk k must be int, got {type(k)}"

    dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", -1)
    largest = node.args[3] if len(node.args) > 3 else node.kwargs.get("largest", True)
    # sorted arg (node.args[4]) is ignored - tl.topk always returns sorted

    assert isinstance(dim, int), f"topk dim must be int, got {type(dim)}"
    assert isinstance(largest, bool), f"topk largest must be bool, got {type(largest)}"

    # Get the input tensor shape info
    input_val = node.args[0]
    assert isinstance(input_val, Node)
    input_tensor = input_val.meta["val"]
    ndim = input_tensor.ndim

    # Normalize negative dim
    if dim < 0:
        dim = ndim + dim

    # tl.topk only supports sorting on the last dimension
    assert dim == ndim - 1, f"tl.topk only supports the last dimension, got dim={dim}"

    # Get size of last dimension
    n = input_tensor.shape[-1]
    env = CompileEnvironment.current()
    n_hint = env.size_hint(n) if isinstance(n, torch.SymInt) else n
    n_pow2 = next_power_of_2(n_hint)
    k_pow2 = next_power_of_2(k)

    if input_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return _codegen_float_topk(ctx, tensor, input_tensor, k_pow2, largest)

    # Generate top-k values using tl.topk (for largest=True) or tl.sort (for largest=False)
    topk_vals = ctx.cg.device_function.new_var("topk_vals")
    if largest:
        # tl.topk returns top k largest elements directly
        ctx.cg.add_statement(
            statement_from_string(
                f"{topk_vals} = tl.topk({{tensor}}, {k_pow2})",
                tensor=tensor,
            )
        )
    else:
        # tl.topk only supports largest=True, so use tl.sort with descending=False
        sorted_vals = ctx.cg.device_function.new_var("sorted_vals")
        ctx.cg.add_statement(
            statement_from_string(
                f"{sorted_vals} = tl.sort({{tensor}}, descending=False)",
                tensor=tensor,
            )
        )
        # Need to gather first k elements from sorted
        k_idx = ctx.cg.device_function.new_var("k_idx")
        idx_n = ctx.cg.device_function.new_var("idx_n")
        ctx.cg.add_statement(statement_from_string(f"{k_idx} = tl.arange(0, {k_pow2})"))
        ctx.cg.add_statement(statement_from_string(f"{idx_n} = tl.arange(0, {n_pow2})"))
        if ndim == 1:
            ctx.cg.add_statement(
                statement_from_string(
                    f"{topk_vals} = tl.sum("
                    f"tl.where(({idx_n}[:, None] == {k_idx}[None, :]) & ({k_idx}[None, :] < {k}), "
                    f"{sorted_vals}[:, None], 0.0), axis=0)"
                )
            )
        else:
            ctx.cg.add_statement(
                statement_from_string(
                    f"{topk_vals} = tl.sum("
                    f"tl.where(({idx_n}[None, :, None] == {k_idx}[None, None, :]) & ({k_idx}[None, None, :] < {k}), "
                    f"{sorted_vals}[:, :, None], 0.0), axis=1)"
                )
            )

    # For indices, compute argsort using ranking approach
    topk_indices = ctx.cg.device_function.new_var("topk_indices")
    rank = ctx.cg.device_function.new_var("rank")
    idx_var = ctx.cg.device_function.new_var("idx")

    ctx.cg.add_statement(statement_from_string(f"{idx_var} = tl.arange(0, {n_pow2})"))

    # Set up dimension-specific indexing patterns and comparison operator
    cmp_op = ">" if largest else "<"
    if ndim == 1:
        t_a, t_b = "[None, :]", "[:, None]"
        i_a, i_b = "[None, :]", "[:, None]"
        reduce_axis = 1
        r_a, r_b, inv_i_a, inv_axis = "[:, None]", "[None, :]", "[:, None]", 0
    elif ndim == 2:
        t_a, t_b = "[:, None, :]", "[:, :, None]"
        i_a, i_b = "[None, None, :]", "[None, :, None]"
        reduce_axis = 2
        r_a, r_b, inv_i_a, inv_axis = (
            "[:, :, None]",
            "[None, None, :]",
            "[None, :, None]",
            1,
        )
    else:
        raise NotImplementedError

    # Compute rank: count elements that should come before + tie-breaking
    ctx.cg.add_statement(
        statement_from_string(
            f"{rank} = tl.sum(tl.where({{tensor}}{t_a} {cmp_op} {{tensor}}{t_b}, 1, 0), axis={reduce_axis}) + "
            f"tl.sum(tl.where(({{tensor}}{t_a} == {{tensor}}{t_b}) & ({idx_var}{i_a} < {idx_var}{i_b}), 1, 0), axis={reduce_axis})",
            tensor=tensor,
        )
    )

    # Invert rank permutation to get sorted indices, then gather first k
    sorted_indices = ctx.cg.device_function.new_var("sorted_indices")
    ctx.cg.add_statement(
        statement_from_string(
            f"{sorted_indices} = tl.sum(tl.where({rank}{r_a} == {idx_var}{r_b}, {idx_var}{inv_i_a}, 0), axis={inv_axis})"
        )
    )

    # Gather first k indices
    k_idx_final = ctx.cg.device_function.new_var("k_idx")
    ctx.cg.add_statement(
        statement_from_string(f"{k_idx_final} = tl.arange(0, {k_pow2})")
    )

    if ndim == 1:
        ctx.cg.add_statement(
            statement_from_string(
                f"{topk_indices} = tl.sum("
                f"tl.where(({idx_var}[:, None] == {k_idx_final}[None, :]) & ({k_idx_final}[None, :] < {k}), "
                f"{sorted_indices}[:, None], 0), axis=0)"
            )
        )
    else:
        ctx.cg.add_statement(
            statement_from_string(
                f"{topk_indices} = tl.sum("
                f"tl.where(({idx_var}[None, :, None] == {k_idx_final}[None, None, :]) & ({k_idx_final}[None, None, :] < {k}), "
                f"{sorted_indices}[:, :, None], 0), axis=1)"
            )
        )

    return (expr_from_string(topk_vals), expr_from_string(topk_indices))


def _codegen_float_topk(
    ctx: LoweringContext,
    tensor: ast.AST,
    input_tensor: torch.Tensor,
    k: int,
    largest: bool,
) -> tuple[ast.AST, ast.AST]:
    """Select ordered value/index keys without quadratic rank tensors.

    FP16/BF16 values promote exactly to FP32 for comparison. Original payloads
    are gathered after selection, preserving signed zeros and NaN payload bits.
    NaNs compare above infinity; equal values select the lowest original index.
    Padding is outside the valid key range in either ordering direction.
    """
    fn = ctx.cg.device_function
    env = CompileEnvironment.current()
    shape = list(input_tensor.shape)
    suffix = "[" + ", ".join(["None"] * (len(shape) - 1) + [":"]) + "]"
    index = fn.new_var("topk_index")
    bits = fn.new_var("topk_bits")
    ordered = fn.new_var("topk_ordered")
    keys = fn.new_var("topk_keys")
    selected = fn.new_var("topk_selected")
    indices = fn.new_var("topk_indices")
    values = fn.new_var("topk_values")

    def emit(source: str) -> None:
        ctx.cg.add_statement(statement_from_string(source, tensor=tensor))

    emit(f"{index} = tl.arange(0, {{tensor}}.shape[-1]).to(tl.uint32)")
    emit(f"{bits} = {{tensor}}.to(tl.float32).to(tl.uint32, bitcast=True)")
    emit(f"{bits} = tl.where(({bits} & 0x7fffffff) == 0, 0, {bits})")
    emit(
        f"{ordered} = tl.where(({bits} & 0x80000000) != 0, {bits} ^ 0xffffffff, {bits} ^ 0x80000000)"
    )
    emit(
        f"{ordered} = tl.where(({bits} & 0x7fffffff) > 0x7f800000, 0xffffffff, {ordered})"
    )
    tie = f"({index} ^ 0xffffffff)" if largest else index
    emit(f"{keys} = ({ordered}.to(tl.uint64) << 32) | {tie}{suffix}.to(tl.uint64)")

    axis = env.resolve_block_id(shape[-1])
    logical = (
        env.block_sizes[axis].numel
        if axis is not None and env.block_sizes[axis].reduction
        else shape[-1]
    )
    masks = [f"({index}{suffix} < {fn.literal_expr(logical)})"]
    # Reuse ordinary live-axis masks for independently tiled axes and tails.
    # A padding value equal to a real infinity/NaN must still lose selection.
    for dim, size in enumerate(shape):
        block = env.resolve_block_id(size)
        if block is not None and (mask := fn.codegen.mask_var(block)) is not None:
            expand = fn.tile_strategy.expand_str(shape, dim)
            masks.append(f"({mask}{expand})")
    invalid = "0" if largest else "0xffffffffffffffff"
    emit(
        f"{keys} = tl.where({' & '.join(masks)}, {keys}, tl.full([], {invalid}, tl.uint64))"
    )
    # Older supported Triton versions only provide descending topk. Invert
    # unsigned keys to select the smallest values with the same primitive.
    topk_expr = (
        f"tl.topk({keys}, {k})"
        if largest
        else f"tl.topk({keys} ^ 0xffffffffffffffff, {k}) ^ 0xffffffffffffffff"
    )
    topk_statement = f"{selected} = {topk_expr}"
    capability = env.config_spec.target_device_capability
    # GB300 measurements favor sort for one row per four-warp program in this
    # crossover range. Multi-row tiles, smaller k, and other targets use topk.
    if (
        env.backend_name == "triton"
        and not is_hip()
        and capability is not None
        and capability[0] == 10
        and fn.config.num_warps == 4
        and 32 <= k <= 512
    ):
        sorted_keys = fn.new_var("topk_sorted")
        selected_shape = ", ".join(
            [*(f"{sorted_keys}.shape[{dim}]" for dim in range(len(shape) - 1)), str(k)]
        )
        sort_statements = [
            f"{sorted_keys} = tl.sort({keys}, descending={largest!r})",
            f"{selected} = tl.gather({sorted_keys}, tl.broadcast_to(tl.arange(0, {k}){suffix}, [{selected_shape}]), axis={len(shape) - 1})",
        ]
        # Use physical extents so dynamic shapes specialize this decision again
        # when their padded width or the number of rows per program changes.
        condition = "{tensor}.shape[-1] == 1024 and {tensor}.numel == 1024"
        sort_body = "\n    ".join(sort_statements)
        emit(f"if {condition}:\n    {sort_body}\nelse:\n    {topk_statement}")
    else:
        # TileIR retains the gather-free top-k path.
        emit(topk_statement)
    decoded = f"{selected}.to(tl.uint32)"
    if largest:
        decoded = f"({decoded} ^ 0xffffffff)"
    emit(f"{indices} = ({decoded}).to(tl.int64)")
    if env.backend_name == "tileir":
        # TileIR cannot legalize gather. Select original unsigned payload bits
        # one output at a time, preserving NaNs/zeros without a K-by-N temporary.
        payloads = fn.new_var("topk_payloads")
        ordinal = fn.new_var("topk_ordinal")
        slot = fn.new_var("topk_slot")
        selected_index = fn.new_var("topk_selected_index")
        payload = fn.new_var("topk_payload")
        dim = len(shape) - 1
        uint = fn.new_var("topk_payload_type")
        # Producers such as division may retain FP32 despite a low-precision
        # FX dtype. Preserve the actual Triton value until the output store.
        emit(
            f"{uint}: tl.constexpr = tl.uint16 if {{tensor}}.dtype.primitive_bitwidth == 16 else tl.uint32"
        )
        emit(f"{payloads} = {{tensor}}.to({uint}, bitcast=True).to(tl.uint32)")
        emit(f"{ordinal} = tl.arange(0, {indices}.shape[-1]){suffix}")
        emit(f"{values} = tl.full({indices}.shape, 0, tl.uint32)")
        emit(
            f"for {slot} in range({k}):\n"
            f"    {selected_index} = tl.max(tl.where({ordinal} == {slot}, {indices}, 0), axis={dim})\n"
            f"    {payload} = tl.max(tl.where({index}{suffix} == tl.expand_dims({selected_index}, {dim}), {payloads}, 0), axis={dim})\n"
            f"    {values} = tl.where({ordinal} == {slot}, tl.expand_dims({payload}, {dim}), {values})"
        )
        emit(f"{values} = {values}.to({uint}).to({{tensor}}.dtype, bitcast=True)")
    else:
        # Dummy K-padding slots are not stored but still participate in tl.gather.
        emit(
            f"{values} = tl.gather({{tensor}}, tl.minimum({indices}, {{tensor}}.shape[-1] - 1).to(tl.int32), axis={len(shape) - 1})"
        )
    return expr_from_string(values), expr_from_string(indices)
