"""Triton-backend codegen for ops defined in ``helion.language.matmul_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; ``matmul_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch._subclasses.fake_tensor import FakeTensor

from ...language import _decorators
from ...language import _tracing_ops
from ...language.matmul_ops import dot
from ...language.matmul_ops import dot_scaled
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from ..matmul_utils import _emit_tl_dot_scaled
from ..matmul_utils import _resolve_dim_size
from ..matmul_utils import emit_tl_dot_with_padding
from ..mxfp8 import PackedScaleMatch
from ..mxfp8 import match_packed_scale
from ..mxfp8 import tile_load

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


def _unmask_zero_operand(node: object) -> object:
    while (
        isinstance(node, torch.fx.Node)
        and node.target is _tracing_ops._mask_to
        and len(node.args) == 2
        and not node.kwargs
        and node.args[1] == 0
    ):
        node = node.args[0]
    return node


def _node_ast(state: CodegenState, node: object) -> ast.AST | None:
    if not isinstance(node, torch.fx.Node):
        return None
    value = state.env.get(node)
    return value if isinstance(value, ast.AST) else None


def _host_descriptor_tile(
    state: CodegenState,
    tensor: torch.Tensor,
    indices: tuple[object, object],
    block_shape: tuple[int, int],
    *,
    physical_layout: bool = False,
) -> ast.AST | None:
    """Load one 2D tile through a descriptor, including padded edge tiles."""
    if tensor.ndim != 2:
        return None
    index_asts = tuple(_node_ast(state, index) for index in indices)
    if any(index is None for index in index_asts):
        return None
    env = CompileEnvironment.current()
    layout = env.tensor_descriptor_layout_signature(tensor)
    if (
        env.ptx_capability is None
        or env.ptx_capability[0] != 10
        or layout is None
        or layout[0] is None
        or not all(aligned or dim == layout[0] for dim, aligned in enumerate(layout[1]))
        or not env.tensor_descriptor_base_is_aligned(tensor)
    ):
        return None
    descriptor = state.device_function.tensor_descriptor_arg(
        tensor, list(block_shape), force_host=True
    )
    permutation = descriptor.permutation or [0, 1]
    offsets = []
    for dim in permutation:
        index_ast = index_asts[dim]
        assert index_ast is not None
        offsets.append(
            expr_from_string(
                "tl.cast(tl.min({index}), tl.int32)",
                index=index_ast,
            )
        )
    placeholders = {f"offset_{i}": value for i, value in enumerate(offsets)}
    load = expr_from_string(
        f"{descriptor.name}.load("
        f"[{', '.join(f'{{offset_{i}}}' for i in range(len(offsets)))}])",
        **placeholders,
    )
    if descriptor.permutation is not None and not physical_layout:
        load = expr_from_string(
            f"tl.permute({{value}}, {descriptor.inverse_permutation!r})",
            value=load,
        )
    return load


def _packed_scale_ast(
    state: CodegenState,
    match: PackedScaleMatch,
    tile_shape: tuple[int, int],
    *,
    prefix: str,
) -> ast.AST | None:
    """Load packed words once and expose scales in tcgen05's UTCCP order."""
    block_rows, block_groups = tile_shape
    if block_rows % 128 or block_groups % 4:
        return None
    row_ast = _node_ast(state, match.row)
    word_index_ast = _node_ast(state, match.word_index)
    if row_ast is None or word_index_ast is None:
        return None

    env = CompileEnvironment.current()
    tensor = match.tensor
    try:
        rows, words_per_row = (env.size_hint(value) for value in tensor.shape)
        row_stride, word_stride = (env.size_hint(value) for value in tensor.stride())
    except (AssertionError, TypeError, ValueError):
        return None
    if (
        tensor.dtype != torch.int32
        or row_stride != 1
        or word_stride < rows
        or word_stride % 4
    ):
        return None

    words = block_groups // 4
    row_repetitions = block_rows // 128
    loaded = _host_descriptor_tile(
        state,
        tensor,
        (match.row, match.word_index),
        (block_rows, words),
        physical_layout=True,
    )
    if loaded is None:
        pointer = state.device_function.tensor_arg(tensor).name
        loaded = expr_from_string(
            (
                "tl.load({pointer} + "
                f"({{word_start}} + tl.arange(0, {words}))[:, None] * {word_stride} + "
                f"{{rows_ast}}[None, :] * {row_stride}, "
                f"mask=(({{word_start}} + tl.arange(0, {words}))[:, None] < {words_per_row}) "
                f"& ({{rows_ast}}[None, :] < {rows}), other=2139062143)"
            ),
            pointer=expr_from_string(pointer),
            word_start=expr_from_string(
                "tl.min({word_index})", word_index=word_index_ast
            ),
            rows_ast=row_ast,
        )
    loaded = state.codegen.lift(loaded, dce=True, prefix=f"{prefix}_words")
    unpacked = expr_from_string(
        (
            "(({words_ast}[:, :, None] >> "
            "(tl.arange(0, 4)[None, None, :] * 8)) & 255).to(tl.uint8)"
        ),
        words_ast=loaded,
    )
    unpacked = state.codegen.lift(unpacked, dce=True, prefix=f"{prefix}_unpacked")
    # Packed words arrive as [K/128, rows, four bytes]. First recover the
    # tutorial's rank-5 physical scale tile, then apply its UTCCP transform.
    scale = expr_from_string(
        (
            f"{{scale}}.reshape({words}, {row_repetitions}, 4, 32, 4)"
            ".trans(1, 0, 3, 2, 4)"
            ".trans(0, 3, 2, 1, 4)"
            f".reshape({block_rows}, {block_groups})"
        ),
        scale=unpacked,
    )
    if match.mask is not None:
        mask_ast = _node_ast(state, match.mask)
        if mask_ast is None:
            return None
        scale = expr_from_string(
            "tl.where({mask}, {scale}, 0)", mask=mask_ast, scale=scale
        )
    return scale


def _packed_mxfp8_scales(
    state: CodegenState,
) -> tuple[ast.AST | None, ast.AST, ast.AST | None, ast.AST] | None:
    node = state.fx_node
    if node is None or state.proxy_args[2] != "e4m3" or state.proxy_args[5] != "e4m3":
        return None
    lhs = tile_load(_unmask_zero_operand(node.args[0]), allow_mask=True)
    rhs_node = _unmask_zero_operand(node.args[3])
    if (
        isinstance(rhs_node, torch.fx.Node)
        and rhs_node.target is torch.ops.aten.permute.default
        and len(rhs_node.args) == 2
        and rhs_node.args[1] == [1, 0]
        and not rhs_node.kwargs
    ):
        rhs_node = rhs_node.args[0]
        rhs_transposed = True
    else:
        rhs_transposed = False
    rhs = tile_load(rhs_node, allow_mask=True)
    if lhs is None or rhs is None:
        return None
    lhs_scale_proxy = state.proxy_args[1]
    rhs_scale_proxy = state.proxy_args[4]
    if not isinstance(lhs_scale_proxy, torch.Tensor) or not isinstance(
        rhs_scale_proxy, torch.Tensor
    ):
        return None
    env = CompileEnvironment.current()
    try:
        groups = env.size_hint(lhs[0].shape[1]) // 32
        lhs_shape = tuple(_resolve_dim_size(value) for value in lhs_scale_proxy.shape)
        rhs_shape = tuple(_resolve_dim_size(value) for value in rhs_scale_proxy.shape)
    except (AssertionError, TypeError, ValueError):
        return None
    if (
        len(lhs_shape) != 2
        or len(rhs_shape) != 2
        or not all(type(value) is int for value in (*lhs_shape, *rhs_shape))
        or lhs_shape[1] != rhs_shape[1]
    ):
        return None
    lhs_shape = cast("tuple[int, int]", lhs_shape)
    rhs_shape = cast("tuple[int, int]", rhs_shape)
    split_k = state.config.get("split_k", 1)
    if type(split_k) is not int or split_k < 1:
        return None
    groups_per_split = (groups + split_k - 1) // split_k
    if groups % split_k or lhs_shape[1] % 4 or groups_per_split % lhs_shape[1]:
        return None
    lhs_match = match_packed_scale(node.args[1], None, lhs_shape[1], allow_mask=True)
    rhs_match = match_packed_scale(node.args[4], None, rhs_shape[1], allow_mask=True)
    if lhs_match is None or rhs_match is None:
        return None
    operand_rows = (lhs[1][0], rhs[1][0] if rhs_transposed else rhs[1][1])
    for operand_row, scale_match in zip(
        operand_rows, (lhs_match, rhs_match), strict=True
    ):
        if operand_row != scale_match.row:
            return None
    lhs_scale = _packed_scale_ast(state, lhs_match, lhs_shape, prefix="mxfp8_lhs")
    rhs_scale = _packed_scale_ast(state, rhs_match, rhs_shape, prefix="mxfp8_rhs")
    if lhs_scale is None or rhs_scale is None:
        return None
    lhs_operand = _host_descriptor_tile(
        state,
        lhs[0],
        (lhs[1][0], lhs[1][1]),
        (lhs_shape[0], lhs_shape[1] * 32),
    )
    rhs_operand = _host_descriptor_tile(
        state,
        rhs[0],
        (rhs[1][0], rhs[1][1]),
        (rhs_shape[0], rhs_shape[1] * 32),
    )
    if rhs_operand is not None and rhs_transposed:
        rhs_operand = expr_from_string("tl.permute({value}, [1, 0])", value=rhs_operand)
    return lhs_operand, lhs_scale, rhs_operand, rhs_scale


def _compact_packed_mxfp8_swap(state: CodegenState) -> bool:
    """Whether a packed MXFP8 dot should use Triton's compact swapped geometry."""
    node = state.fx_node
    if node is None or state.proxy_args[2] != "e4m3" or state.proxy_args[5] != "e4m3":
        return False
    lhs = tile_load(_unmask_zero_operand(node.args[0]), allow_mask=True)
    rhs_node = _unmask_zero_operand(node.args[3])
    if (
        not isinstance(rhs_node, torch.fx.Node)
        or rhs_node.target is not torch.ops.aten.permute.default
        or len(rhs_node.args) != 2
        or rhs_node.args[1] != [1, 0]
        or rhs_node.kwargs
    ):
        return False
    rhs = tile_load(rhs_node.args[0], allow_mask=True)
    lhs_scale_proxy = state.proxy_args[1]
    rhs_scale_proxy = state.proxy_args[4]
    if (
        lhs is None
        or rhs is None
        or not isinstance(lhs_scale_proxy, torch.Tensor)
        or not isinstance(rhs_scale_proxy, torch.Tensor)
    ):
        return False
    try:
        lhs_shape = tuple(_resolve_dim_size(value) for value in lhs_scale_proxy.shape)
        rhs_shape = tuple(_resolve_dim_size(value) for value in rhs_scale_proxy.shape)
    except (AssertionError, TypeError, ValueError):
        return False
    if (
        len(lhs_shape) != 2
        or len(rhs_shape) != 2
        or not all(type(value) is int for value in (*lhs_shape, *rhs_shape))
    ):
        return False
    lhs_shape = cast("tuple[int, int]", lhs_shape)
    rhs_shape = cast("tuple[int, int]", rhs_shape)
    if (
        lhs_shape[0] not in (16, 32)
        or rhs_shape[0] < 128
        or lhs_shape[1] != rhs_shape[1]
    ):
        return False
    lhs_match = match_packed_scale(node.args[1], None, lhs_shape[1], allow_mask=True)
    rhs_match = match_packed_scale(node.args[4], None, rhs_shape[1], allow_mask=True)
    return (
        lhs_match is not None
        and rhs_match is not None
        and lhs[1][0] == lhs_match.row
        and rhs[1][0] == rhs_match.row
    )


@_decorators.codegen(dot, "triton")
def _(state: CodegenState) -> object:
    # Get the AST representations of our arguments
    lhs_ast = state.ast_arg(0)
    rhs_ast = state.ast_arg(1)
    acc_ast = state.ast_arg(2)
    assert isinstance(lhs_ast, ast.AST)
    assert isinstance(rhs_ast, ast.AST)

    # Get the dtypes of the inputs from proxy args
    lhs_proxy = state.proxy_args[0]
    assert isinstance(lhs_proxy, FakeTensor), "lhs_proxy must be a FakeTensor"
    rhs_proxy = state.proxy_args[1]
    assert isinstance(rhs_proxy, FakeTensor), "rhs_proxy must be a FakeTensor"
    acc_proxy = state.proxy_args[2] if len(state.proxy_args) > 2 else None
    out_dtype_proxy = state.proxy_args[3] if len(state.proxy_args) > 3 else None

    lhs_dtype = lhs_proxy.dtype
    rhs_dtype = rhs_proxy.dtype
    acc_dtype: torch.dtype | None = None
    if acc_proxy is not None:
        assert isinstance(acc_proxy, FakeTensor), "acc_proxy must be a FakeTensor"
        acc_dtype = acc_proxy.dtype

    out_dtype: torch.dtype | None = None
    if out_dtype_proxy is not None:
        assert isinstance(out_dtype_proxy, torch.dtype), (
            "out_dtype must be a torch.dtype"
        )
        out_dtype = out_dtype_proxy

    # Check if accumulator is None
    is_acc_none = isinstance(acc_ast, ast.Constant) and acc_ast.value is None

    lhs_shape: list[int | torch.SymInt] = list(lhs_proxy.shape)
    rhs_shape: list[int | torch.SymInt] = list(rhs_proxy.shape)
    acc_shape: list[int | torch.SymInt] | None = (
        list(acc_proxy.shape) if acc_proxy is not None else None
    )
    acc_arg = None if is_acc_none else acc_ast
    acc_dtype_arg = acc_dtype if not is_acc_none else None

    lhs_fx_node = state.fx_node.args[0] if state.fx_node is not None else None

    # Perform dot with optional padding
    return emit_tl_dot_with_padding(
        lhs_ast,
        rhs_ast,
        acc_arg,
        lhs_dtype,
        rhs_dtype,
        acc_dtype=acc_dtype_arg,
        lhs_shape=lhs_shape,
        rhs_shape=rhs_shape,
        acc_shape=acc_shape,
        out_dtype=out_dtype,
        lhs_fx_node=lhs_fx_node if isinstance(lhs_fx_node, torch.fx.Node) else None,
    )


@_decorators.codegen(dot_scaled, "triton")
def _(state: CodegenState) -> object:
    lhs_ast = state.ast_arg(0)  # mat1
    lhs_scale_ast = state.ast_arg(1)  # mat1_scale
    lhs_format = state.proxy_args[2]  # "e2m1" etc (string, not AST)
    assert isinstance(lhs_format, str), "lhs_format must be a string"
    rhs_ast = state.ast_arg(3)  # mat2
    rhs_scale_ast = state.ast_arg(4)  # mat2_scale
    rhs_format = state.proxy_args[5]  # "e2m1" etc (string, not AST)
    assert isinstance(rhs_format, str), "rhs_format must be a string"
    acc_ast = state.ast_arg(6)  # acc
    out_dtype_proxy = state.proxy_args[7] if len(state.proxy_args) > 7 else None

    out_dtype: torch.dtype | None = None
    if out_dtype_proxy is not None:
        assert isinstance(out_dtype_proxy, torch.dtype), (
            "out_dtype must be a torch.dtype"
        )
        out_dtype = out_dtype_proxy

    if packed_scales := _packed_mxfp8_scales(state):
        packed_lhs, lhs_scale_ast, packed_rhs, rhs_scale_ast = packed_scales
        if packed_lhs is not None:
            lhs_ast = packed_lhs
        if packed_rhs is not None:
            rhs_ast = packed_rhs

    is_acc_none = isinstance(acc_ast, ast.Constant) and acc_ast.value is None
    acc = None if is_acc_none else acc_ast
    if _compact_packed_mxfp8_swap(state):
        lhs_transposed = expr_from_string("tl.permute({value}, [1, 0])", value=lhs_ast)
        rhs_transposed = expr_from_string("tl.permute({value}, [1, 0])", value=rhs_ast)
        acc_transposed = (
            None
            if acc is None
            else expr_from_string("tl.permute({value}, [1, 0])", value=acc)
        )
        swapped = _emit_tl_dot_scaled(
            rhs_transposed,
            rhs_scale_ast,
            rhs_format,
            lhs_transposed,
            lhs_scale_ast,
            lhs_format,
            acc=acc_transposed,
            out_dtype=out_dtype,
        )
        return expr_from_string("tl.permute({value}, [1, 0])", value=swapped)
    return _emit_tl_dot_scaled(
        lhs_ast,
        lhs_scale_ast,
        lhs_format,
        rhs_ast,
        rhs_scale_ast,
        rhs_format,
        acc=acc,
        out_dtype=out_dtype,
    )
