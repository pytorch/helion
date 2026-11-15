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
from .tracing_ops import _tile_mask_expr

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..aten_lowering import LoweringContext
    from ..device_function import DeviceFunction


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
        ctx.cg.device_function.tile_strategy.physical_dims(input_val.shape)
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
    input_dims = tile_strategy.physical_dims(input_val.shape)
    output_dims = tile_strategy.physical_dims(output_val.shape)
    new_axes = [axis for axis, dims in enumerate(output_dims) if dims == [dim]]
    shifted_input = [
        [index if index < dim else index + 1 for index in dims] for dims in input_dims
    ]
    remaining_output = [
        dims for axis, dims in enumerate(output_dims) if axis not in new_axes
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
            ["None"] * stack_dim + [":"] + ["None"] * (len(output_dims) - stack_dim - 1)
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
    """Generate torch.sort(input, dim=-1, descending=False).

    Returns (values, indices), with indices None when they are unused. The
    order is stable: equal elements keep their input order, NaNs order above
    infinity (last ascending, first descending) and signed zeros compare
    equal, as in eager, and the values keep the selected elements' bits. A
    sort whose indices go unused orders -0.0 below +0.0 and NaNs among
    themselves by their bits instead. The lanes that pad the sorted dim to a
    power of two, or past the end of a partial tile, hold no element and
    reach no output.

    Note: tl.sort only works on the last dimension currently.
    """
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    assert isinstance(tensor, ast.AST)

    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
    descending = (
        node.args[2] if len(node.args) > 2 else node.kwargs.get("descending", False)
    )

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
    return _codegen_selection(ctx, node, tensor, input_tensor, None, descending)


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

    axis = fn.tile_strategy.physical_dim([*input_tensor.size()], dim)
    ctx.cg.add_statement(
        statement_from_string(
            f"{result_var} = tl.gather({{input}}, {{index}}.to(tl.int32), axis={axis})",
            input=input_ast,
            index=index_ast,
        )
    )

    return expr_from_string(result_var)


@topk_lowering.register_codegen("triton")
def codegen_topk(ctx: LoweringContext, node: Node) -> object:
    """Generate torch.topk(input, k, dim=-1, largest=True, sorted=True).

    Returns (values, indices), ordered as codegen_sort orders them; the
    selection is always sorted. tl.topk takes a power-of-two k, so the results
    hold next_power_of_2(k) slots, of which only the first k are stored.

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
    return _codegen_selection(
        ctx, node, tensor, input_tensor, next_power_of_2(k), largest
    )


def _codegen_selection(
    ctx: LoweringContext,
    node: Node,
    tensor: ast.AST,
    input_tensor: torch.Tensor,
    k: int | None,
    largest: bool,
) -> tuple[ast.AST, ast.AST | None]:
    """Sort the last dim (k is None) or select its k largest or smallest.

    The lanes outside the elements (_selection_valid) select after every
    element. Values of up to 32 bits select value/index keys
    (_codegen_packed_selection) and 64-bit values select by ranks
    (_codegen_rank_selection), except where the indices go unused and the
    values have an image that orders them alone (_emit_image_selection, see
    below).
    """
    ndim = _selection_rank(ctx, input_tensor)
    indices_used = any(
        user.target is getitem and user.args[1] == 1 for user in node.users
    )

    def emit(source: str) -> None:
        ctx.cg.add_statement(statement_from_string(source, tensor=tensor))

    fn = ctx.cg.device_function
    dtype = input_tensor.dtype
    prefix = "sort" if k is None else "topk"
    index = fn.new_var(f"{prefix}_index")
    valid = fn.new_var(f"{prefix}_valid")
    emit(f"{index} = tl.arange(0, {{tensor}}.shape[-1])")
    emit(f"{valid} = {_selection_valid(fn, list(input_tensor.shape), index, ndim)}")
    # Integers, 64-bit values and the sorts of floats with a bit-exact image
    # order those images when the indices go unused. A floating top-k of up
    # to 32 bits keeps its value/index keys and gathers its k outputs, as the
    # GB300 sort-or-topk dispatch (_keys_selection_source) was measured on
    # those keys. float8 has no bit-exact image.
    if not indices_used and (
        not dtype.is_floating_point
        or (dtype in _INFINITY_BITS and (k is None or dtype.itemsize == 8))
    ):
        width = 64 if dtype.itemsize == 8 else 32
        values = _emit_image_selection(
            emit, fn, dtype, width, valid, k, largest, prefix
        )
        return expr_from_string(values), None
    selection = (
        _codegen_packed_selection if dtype.itemsize <= 4 else _codegen_rank_selection
    )
    values, indices = selection(emit, fn, dtype, k, largest, ndim, prefix, index, valid)
    if not indices_used:
        return expr_from_string(values), None
    return expr_from_string(values), expr_from_string(indices)


def _selection_rank(ctx: LoweringContext, input_tensor: torch.Tensor) -> int:
    """The rank of the emitted value a sort or top-k orders along its last dim.

    flatten_loops makes a multi-dim tile one dimension of the value;
    ``physical_dim`` refuses ordering along a dim of it, which would run
    across the whole tile.
    """
    tile_strategy = ctx.cg.device_function.tile_strategy
    last = input_tensor.ndim - 1
    return tile_strategy.physical_dim([*input_tensor.size()], last) + 1


def _last_dim_index(ndim: int) -> str:
    """The subscript placing a 1-D tensor along the last of ``ndim`` dims."""
    return "[" + ", ".join(["None"] * (ndim - 1) + [":"]) + "]"


def _selection_valid(
    fn: DeviceFunction, shape: list[int | torch.SymInt], index: str, ndim: int
) -> str:
    """The lanes of a selection input that hold elements.

    They lie inside the selected dim's logical extent, which its reduction
    block pads to a power of two, and inside the live masks of the tiled dims
    (_tile_mask_expr), such as the tail of a partial tile.
    """
    env = CompileEnvironment.current()
    axis = env.resolve_block_id(shape[-1])
    logical = (
        env.block_sizes[axis].numel
        if axis is not None and env.block_sizes[axis].reduction
        else shape[-1]
    )
    valid = f"({index}{_last_dim_index(ndim)} < {fn.literal_expr(logical)})"
    tiles = _tile_mask_expr(fn, shape)
    return valid if tiles is None else f"{valid} & {tiles}"


def _emit_image_selection(
    emit: Callable[[str], None],
    fn: DeviceFunction,
    dtype: torch.dtype,
    width: int,
    valid: str,
    k: int | None,
    largest: bool,
    prefix: str,
) -> str:
    """Sort (k is None) or select the ``width``-bit images of ``{tensor}``
    (_emit_ordered_image) alone, and return the variable holding the values
    they decode to, bit for bit.

    Ties need no order, so the lanes outside ``valid`` only have to select no
    earlier than the elements: they take the image that selects last. -0.0
    orders below +0.0, and NaNs order among themselves by their bits.
    """
    image = fn.new_var(f"{prefix}_image")
    selected = fn.new_var(f"{prefix}_selected")
    values = fn.new_var(f"{prefix}_values")
    _emit_ordered_image(emit, fn, dtype, width, image)
    invalid = 0 if largest else (1 << width) - 1
    emit(
        f"{image} = tl.where({valid}, {image}, tl.full([], {invalid:#x}, tl.uint{width}))"
    )
    emit(f"{selected} = {_selection(image, width, k, largest)}")
    _emit_ordered_value(emit, fn, dtype, width, selected, values)
    return values


def _codegen_packed_selection(
    emit: Callable[[str], None],
    fn: DeviceFunction,
    dtype: torch.dtype,
    k: int | None,
    largest: bool,
    ndim: int,
    prefix: str,
    index: str,
    valid: str,
) -> tuple[str, str]:
    """Sort (k is None) or select ordered value/index keys of values of up to
    32 bits, without quadratic rank tensors, and return the variables holding
    the values and the indices.

    A key holds a 32-bit image of the value that orders like torch compares
    it (_emit_canonical_image) above the element's index. Equal values select
    the lowest original index first in either direction. The lanes outside
    ``valid`` take the key that selects last, behind an element equal to an
    infinity or a NaN too. Floating values are read at the selected indices
    (_emit_values_at), preserving signed zeros and NaN payload bits; integer
    values are decoded from their keys.
    """
    suffix = _last_dim_index(ndim)
    ordered = fn.new_var(f"{prefix}_ordered")
    keys = fn.new_var(f"{prefix}_keys")
    selected = fn.new_var(f"{prefix}_selected")
    indices = fn.new_var(f"{prefix}_indices")
    values = fn.new_var(f"{prefix}_values")
    _emit_canonical_image(emit, fn, dtype, 32, ordered)
    position = f"{index}.to(tl.uint32)"
    tie = f"({position} ^ 0xffffffff)" if largest else position
    emit(f"{keys} = ({ordered}.to(tl.uint64) << 32) | {tie}{suffix}.to(tl.uint64)")
    invalid = "0" if largest else "0xffffffffffffffff"
    emit(f"{keys} = tl.where({valid}, {keys}, tl.full([], {invalid}, tl.uint64))")
    emit(_keys_selection_source(fn, keys, selected, k, largest, ndim))
    decoded = f"{selected}.to(tl.uint32)"
    if largest:
        decoded = f"({decoded} ^ 0xffffffff)"
    emit(f"{indices} = ({decoded}).to(tl.int64)")
    if not dtype.is_floating_point:
        # Integer and bool keys hold their values exactly.
        image = f"({selected} >> 32).to(tl.uint32)"
        _emit_ordered_value(emit, fn, dtype, 32, image, values)
    else:
        _emit_values_at(emit, fn, dtype, prefix, index, indices, k, ndim, values)
    return values, indices


# Infinity's bits in the IEEE formats whose ordered images keep every bit.
_INFINITY_BITS = {
    torch.float16: 0x7C00,
    torch.bfloat16: 0x7F80,
    torch.float32: 0x7F800000,
    torch.float64: 0x7FF0000000000000,
}


def _float_layout(dtype: torch.dtype) -> tuple[int, int, int]:
    """The width, sign bit and rotation of a float's ordered image.

    Offset binary (the sign bit flipped, or every bit of a negative value)
    orders bit patterns like their values, but puts the negative NaNs below
    -inf. Rotating it down by their count moves them above the positive NaNs,
    so every NaN orders above infinity and each pattern keeps its own image.
    """
    width = dtype.itemsize * 8
    sign = 1 << (width - 1)
    return width, sign, sign - 1 - _INFINITY_BITS[dtype]


def _emit_ordered_image(
    emit: Callable[[str], None],
    fn: DeviceFunction,
    dtype: torch.dtype,
    width: int,
    image: str,
) -> None:
    """Assign ``image`` an unsigned ``width``-bit image of ``{tensor}`` that
    orders like its values and decodes back to them bit for bit
    (_emit_ordered_value).

    Integers and bools widen. A float images its own bits (_float_layout)
    after a cast to its dtype, which rounds a value a producer kept in FP32
    as the store would; converting FP16 to FP32 would make every NaN one
    quiet NaN on GPUs.
    """
    key = f"tl.uint{width}"
    if not dtype.is_floating_point:
        if dtype.is_signed:
            emit(
                f"{image} = {{tensor}}.to(tl.int{width}).to({key}, bitcast=True) ^ {1 << (width - 1):#x}"
            )
        else:
            emit(f"{image} = {{tensor}}.to({key})")
        return
    bit_width, sign, rotation = _float_layout(dtype)
    ones = (1 << bit_width) - 1
    bits = fn.new_var("ordered_bits")
    widen = f".to({key})" if bit_width < width else ""
    emit(
        f"{bits} = {{tensor}}.to({triton_type(dtype)}).to(tl.uint{bit_width}, bitcast=True){widen}"
    )
    rotated = f"tl.where(({bits} & {sign:#x}) != 0, {bits} ^ {ones:#x}, {bits} ^ {sign:#x}) - {rotation:#x}"
    if bit_width < width:
        rotated = f"({rotated}) & {ones:#x}"
    emit(f"{image} = {rotated}")


def _emit_ordered_value(
    emit: Callable[[str], None],
    fn: DeviceFunction,
    dtype: torch.dtype,
    width: int,
    image: str,
    values: str,
) -> None:
    """Assign ``values`` the elements whose _emit_ordered_image is ``image``."""
    if not dtype.is_floating_point:
        if dtype.is_signed:
            image = f"({image} ^ {1 << (width - 1):#x}).to(tl.int{width}, bitcast=True)"
        emit(f"{values} = {image}.to({triton_type(dtype)})")
        return
    bit_width, sign, rotation = _float_layout(dtype)
    ones = (1 << bit_width) - 1
    unrotated = fn.new_var("ordered_image")
    source = f"{image} + {rotation:#x}"
    if bit_width < width:
        source = f"({source}) & {ones:#x}"
    emit(f"{unrotated} = {source}")
    bits = f"tl.where(({unrotated} & {sign:#x}) != 0, {unrotated} ^ {sign:#x}, {unrotated} ^ {ones:#x})"
    if bit_width < width:
        bits += f".to(tl.uint{bit_width})"
    emit(f"{values} = {bits}.to({triton_type(dtype)}, bitcast=True)")


def _emit_canonical_image(
    emit: Callable[[str], None],
    fn: DeviceFunction,
    dtype: torch.dtype,
    width: int,
    image: str,
) -> None:
    """Assign ``image`` an unsigned ``width``-bit image of ``{tensor}`` that
    orders like torch compares its values: every NaN equal and above
    infinity, and the two zeros equal, so that an index breaks their ties.

    Integers and bools take their ordered images (_emit_ordered_image). A
    float is cast to its dtype first, which rounds a value a producer kept
    in FP32 as the store would, then images the bits of its exact
    ``width``-bit float form in offset binary.
    """
    if not dtype.is_floating_point:
        _emit_ordered_image(emit, fn, dtype, width, image)
        return
    form = {32: torch.float32, 64: torch.float64}[width]
    sign = 1 << (width - 1)
    ones = (1 << width) - 1
    cast = f".to({triton_type(dtype)})"
    if dtype != form:
        cast += f".to({triton_type(form)})"
    bits = fn.new_var("canonical_bits")
    emit(f"{bits} = {{tensor}}{cast}.to(tl.uint{width}, bitcast=True)")
    emit(f"{bits} = tl.where(({bits} & {sign - 1:#x}) == 0, 0, {bits})")
    emit(
        f"{image} = tl.where(({bits} & {sign:#x}) != 0, {bits} ^ {ones:#x}, {bits} ^ {sign:#x})"
    )
    emit(
        f"{image} = tl.where(({bits} & {sign - 1:#x}) > {_INFINITY_BITS[form]:#x}, {ones:#x}, {image})"
    )


def _selection(keys: str, width: int, k: int | None, largest: bool) -> str:
    """The expression sorting the unsigned ``width``-bit ``keys`` along their
    last dim (k is None) or selecting their k largest or smallest, in order."""
    if k is None:
        return f"tl.sort({keys}, descending={largest!r})"
    if largest:
        return f"tl.topk({keys}, {k})"
    # Older supported Triton versions only provide descending topk; the
    # largest complements are the smallest keys.  The complement is an xor
    # with all ones: ``~`` of an unsigned tensor overflows in the interpreter
    # of Triton 3.7 (it fills -1 into the unsigned dtype).
    ones = (1 << width) - 1
    return f"tl.topk({keys} ^ {ones:#x}, {k}) ^ {ones:#x}"


def _keys_selection_source(
    fn: DeviceFunction,
    keys: str,
    selected: str,
    k: int | None,
    largest: bool,
    ndim: int,
) -> str:
    """The statements assigning ``selected`` the _selection of packed
    value/index keys, which sorts rather than tl.topk where GB300 measured
    that faster."""
    env = CompileEnvironment.current()
    statement = f"{selected} = {_selection(keys, 64, k, largest)}"
    capability = env.config_spec.target_device_capability
    # GB300 measurements favor sort for one row per four-warp program in this
    # crossover range. Multi-row tiles, smaller k, and other targets use topk.
    if (
        k is not None
        and env.backend_name == "triton"
        and not is_hip()
        and capability is not None
        and capability[0] == 10
        and fn.config.num_warps == 4
        and 32 <= k <= 512
    ):
        sorted_keys = fn.new_var("topk_sorted")
        selected_shape = ", ".join(
            [*(f"{sorted_keys}.shape[{dim}]" for dim in range(ndim - 1)), str(k)]
        )
        sort_statements = [
            f"{sorted_keys} = {_selection(keys, 64, None, largest)}",
            f"{selected} = tl.gather({sorted_keys}, tl.broadcast_to(tl.arange(0, {k}){_last_dim_index(ndim)}, [{selected_shape}]), axis={ndim - 1})",
        ]
        # Use physical extents so dynamic shapes specialize this decision again
        # when their padded width or the number of rows per program changes.
        condition = "{tensor}.shape[-1] == 1024 and {tensor}.numel == 1024"
        sort_body = "\n    ".join(sort_statements)
        return f"if {condition}:\n    {sort_body}\nelse:\n    {statement}"
    return statement


def _emit_values_at(
    emit: Callable[[str], None],
    fn: DeviceFunction,
    dtype: torch.dtype,
    prefix: str,
    index: str,
    indices: str,
    k: int | None,
    ndim: int,
    values: str,
) -> None:
    """Assign ``values`` the elements of ``{tensor}`` at ``indices`` along its
    last dim, keeping their bits (signed zeros, NaN payloads) in ``dtype``.

    The elements are cast to ``dtype`` first, as the keys are
    (_emit_canonical_image): a producer such as division may keep an FP16,
    BF16 or float8 value in FP32, and the values must be the rounded ones
    the keys ordered, whatever consumes them.

    TileIR cannot legalize gather: there each output slot reduces the payload
    bits of the lane its index names, one slot at a time, without a K-by-N
    temporary. ``index`` numbers the lanes of the last dim.
    """
    elements = f"{{tensor}}.to({triton_type(dtype)})"
    if CompileEnvironment.current().backend_name != "tileir":
        # Dummy K-padding slots are not stored but still participate in tl.gather.
        emit(
            f"{values} = tl.gather({elements}, tl.minimum({indices}, {{tensor}}.shape[-1] - 1).to(tl.int32), axis={ndim - 1})"
        )
        return
    suffix = _last_dim_index(ndim)
    payloads = fn.new_var(f"{prefix}_payloads")
    ordinal = fn.new_var(f"{prefix}_ordinal")
    slot = fn.new_var(f"{prefix}_slot")
    selected_index = fn.new_var(f"{prefix}_selected_index")
    payload = fn.new_var(f"{prefix}_payload")
    dim = ndim - 1
    # The payloads reduce as unsigned integers of at least 32 bits.
    bits = dtype.itemsize * 8
    width = max(bits, 32)
    count = f"{indices}.shape[-1]" if k is None else str(k)
    emit(f"{payloads} = {elements}.to(tl.uint{bits}, bitcast=True).to(tl.uint{width})")
    emit(f"{ordinal} = tl.arange(0, {indices}.shape[-1]){suffix}")
    emit(f"{values} = tl.full({indices}.shape, 0, tl.uint{width})")
    emit(
        f"for {slot} in range({count}):\n"
        f"    {selected_index} = tl.max(tl.where({ordinal} == {slot}, {indices}, 0), axis={dim})\n"
        f"    {payload} = tl.max(tl.where({index}{suffix} == tl.expand_dims({selected_index}, {dim}), {payloads}, 0), axis={dim})\n"
        f"    {values} = tl.where({ordinal} == {slot}, tl.expand_dims({payload}, {dim}), {values})"
    )
    emit(
        f"{values} = {values}.to(tl.uint{bits}).to({triton_type(dtype)}, bitcast=True)"
    )


def _codegen_rank_selection(
    emit: Callable[[str], None],
    fn: DeviceFunction,
    dtype: torch.dtype,
    k: int | None,
    largest: bool,
    ndim: int,
    prefix: str,
    index: str,
    valid: str,
) -> tuple[str, str]:
    """Sort (k is None) or select 64-bit values, which leave no room for an
    index beside them in a key, and return the variables holding the values
    and the indices.

    The indices come from ranks: an element's index lands at its rank among
    the lanes of ``valid``, the elements ordered before it, equal ones
    (_emit_canonical_image) only from lower indices. Ranks are quadratic in
    the dim, which the selections whose indices go unused skip
    (_emit_image_selection). The values are read at the indices
    (_emit_values_at), keeping the selected elements' bits.
    """
    values = fn.new_var(f"{prefix}_values")
    ranked = fn.new_var(f"{prefix}_ranked")
    _emit_canonical_image(emit, fn, dtype, 64, ranked)
    lead = ":, " * (ndim - 1)
    # Candidates along the second-to-last dim, the elements they count along
    # the last.
    candidate, other = f"[{lead}:, None]", f"[{lead}None, :]"
    cmp = ">" if largest else "<"
    before = (
        f"{valid}{other} & (({ranked}{other} {cmp} {ranked}{candidate}) | "
        f"(({ranked}{other} == {ranked}{candidate}) & ({index}[None, :] < {index}[:, None])))"
    )
    ranks = fn.new_var(f"{prefix}_rank")
    order = fn.new_var(f"{prefix}_order")
    emit(f"{ranks} = tl.sum(tl.where({before}, 1, 0), axis={ndim})")
    # Invert the valid lanes' ranks: position r holds the index ranked r.
    emit(
        f"{order} = tl.sum(tl.where({valid}{candidate} & ({ranks}{candidate} == {index}[None, :]), {index}[:, None], 0), axis={ndim - 1})"
    )
    selected = order
    if k is not None:
        slot = fn.new_var(f"{prefix}_slot")
        selected = fn.new_var(f"{prefix}_selected")
        emit(f"{slot} = tl.arange(0, {k})")
        emit(
            f"{selected} = tl.sum(tl.where({index}[:, None] == {slot}[None, :], {order}{candidate}, 0), axis={ndim - 1})"
        )
    # int64, as the packed path's indices.
    indices = fn.new_var(f"{prefix}_indices")
    emit(f"{indices} = {selected}.to(tl.int64)")
    _emit_values_at(emit, fn, dtype, prefix, index, indices, k, ndim, values)
    return values, indices
