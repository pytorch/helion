from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch
from torch._inductor.utils import triton_type

from .._compat import min_dot_size
from .ast_extension import expr_from_string
from .compile_environment import CompileEnvironment
from .device_function import DeviceFunction
from .dtype_utils import cast_ast

if TYPE_CHECKING:
    import ast


original_matmul = torch.matmul


def torch_matmul_replacement(
    a: torch.Tensor, b: torch.Tensor, *extra_args: object, **extra_kwargs: object
) -> torch.Tensor:
    if extra_kwargs and "out" in extra_kwargs:
        raise NotImplementedError(
            "torch.matmul(..., out=...) is not supported in Helion kernel"
        )
    if a.dim() != b.dim():
        raise NotImplementedError(
            "torch.matmul with different input tensor dims is not supported in Helion kernel"
        )
    if a.dim() == 2 and b.dim() == 2:
        return original_matmul(a, b)
    if a.dim() == 3 and b.dim() == 3:
        return torch.bmm(a, b)
    raise NotImplementedError(
        "torch.matmul with input tensor dim <2 or >3 is not supported in Helion kernel"
    )


def tensor_matmul_replacement(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch_matmul_replacement(self, other)


def _emit_tl_dot(
    lhs: ast.AST,
    rhs: ast.AST,
    *,
    input_precision: str | None = None,
    acc: ast.AST | None = None,
    out_dtype: torch.dtype | None = None,
) -> ast.AST:
    """Build a tl.dot AST with optional acc/input_precision/out_dtype.

    The caller is responsible for ensuring compatible operand/accumulator
    dtypes for fused accumulation when providing `acc`.
    """
    kwargs = {"lhs": lhs, "rhs": rhs}
    parts = ["tl.dot({lhs}, {rhs}"]
    if acc is not None:
        kwargs["acc"] = acc
        parts.append(", acc={acc}")
    if input_precision:
        parts.append(f", input_precision='{input_precision}'")
    if out_dtype:
        parts.append(f", out_dtype={triton_type(out_dtype)}")
    parts.append(")")
    return expr_from_string("".join(parts), **kwargs)


def _compute_out_dtype(
    mat1_dtype: torch.dtype,
    mat2_dtype: torch.dtype,
    acc_dtype: torch.dtype | None = None,
) -> torch.dtype:
    """Compute the output dtype for dot operation."""
    if acc_dtype is not None:
        # If accumulator is provided, use its dtype
        return acc_dtype

    # When no accumulator is specified:
    # For int8 inputs, default to int32
    if mat1_dtype == torch.int8 or mat2_dtype == torch.int8:
        return torch.int32
    # For all other inputs (including FP8), default to float32
    return torch.float32


def _resolve_dim_size(
    v: int | torch.SymInt | sympy.Expr,
) -> int | torch.SymInt | sympy.Expr:
    """Resolve a dimension size value using both direct resolution and config lookup."""
    if isinstance(v, int):
        return v

    env = CompileEnvironment.current()
    sym_expr = (
        v._sympy_()
        if isinstance(v, torch.SymInt)
        else v
        if isinstance(v, sympy.Expr)
        else None
    )

    if sym_expr is not None:
        replaced = env.shape_env.replace(sym_expr)
        if isinstance(replaced, (int, sympy.Integer, sympy.Number)):
            return int(replaced)

    if not isinstance(v, (torch.SymInt, sympy.Expr)):
        return v

    device_fn = DeviceFunction.current()
    cfg = device_fn.config
    block_idx = env.get_block_id(v)

    if block_idx is None:
        return v

    cfg_value = env.block_sizes[block_idx].from_config(cfg)
    if isinstance(cfg_value, int):
        return cfg_value
    if isinstance(cfg_value, (torch.SymInt, sympy.Expr)):
        sym_expr = (
            cfg_value._sympy_() if isinstance(cfg_value, torch.SymInt) else cfg_value
        )
        replaced = env.shape_env.replace(sym_expr)
        if isinstance(replaced, (int, sympy.Integer, sympy.Number)):
            return int(replaced)

    return v


def _pad_tensor(
    tensor: ast.AST,
    pad_dim: int,
    cur_size: int,
    target_size: int,
    other_dim: int | torch.SymInt | sympy.Expr,
) -> ast.AST:
    """Pad tensor by repeatedly doubling specified dimension."""
    assert pad_dim in (0, 1), f"pad_dim must be 0 or 1, got {pad_dim}"
    x = tensor
    shape_str = DeviceFunction.current().tile_strategy.shape_str
    while cur_size < target_size:
        x = expr_from_string("tl.join({x}, tl.zeros_like({x}))", x=x)
        x = expr_from_string(
            f"tl.permute({{x}}, {'[2, 0, 1]' if pad_dim == 0 else '[0, 2, 1]'})", x=x
        )
        cur_size *= 2
        shape = [cur_size, other_dim] if pad_dim == 0 else [other_dim, cur_size]
        x = expr_from_string(f"tl.reshape({{x}}, {shape_str(shape)})", x=x)
    return x


def emit_tl_dot_with_padding(
    lhs: ast.AST,
    rhs: ast.AST,
    acc: ast.AST | None,
    lhs_dtype: torch.dtype,
    rhs_dtype: torch.dtype,
    *,
    acc_dtype: torch.dtype | None = None,
    out_dtype: torch.dtype | None = None,
    lhs_shape: list[int | torch.SymInt],
    rhs_shape: list[int | torch.SymInt],
    acc_shape: list[int | torch.SymInt] | None = None,
) -> ast.AST:
    device_fn = DeviceFunction.current()
    shape_str = device_fn.tile_strategy.shape_str

    env = CompileEnvironment.current()
    input_precision = env.settings.dot_precision
    config = device_fn.config

    lhs_shape_list = list(lhs_shape)
    rhs_shape_list = list(rhs_shape)
    acc_shape_list = list(acc_shape) if acc_shape is not None else None

    if len(lhs_shape_list) < 2:
        raise ValueError("lhs_shape must have at least two dimensions")
    if len(rhs_shape_list) < 1:
        raise ValueError("rhs_shape must have at least one dimension")

    m, k, n = lhs_shape_list[-2], lhs_shape_list[-1], rhs_shape_list[-1]
    common_dtype = torch.promote_types(lhs_dtype, rhs_dtype)
    lhs_cast, rhs_cast = cast_ast(lhs, common_dtype), cast_ast(rhs, common_dtype)
    m, n, k = (_resolve_dim_size(d) for d in (m, n, k))

    fuse_acc = (
        acc is not None
        and acc_dtype in (common_dtype, torch.float32)
        and (out_dtype is None or out_dtype == acc_dtype)
    )
    acc_out = acc if not fuse_acc else None
    acc_for_dot = acc if fuse_acc else None
    acc_cast_dtype = acc_dtype if not fuse_acc else None

    # Determine the out_dtype to use for tl.dot operation, and whether to
    # explicitly cast the tl.dot result to the expected output dtype
    expected_out_dtype = out_dtype or (
        acc_dtype if fuse_acc else _compute_out_dtype(lhs_dtype, rhs_dtype)
    )
    if expected_out_dtype == torch.float32:
        dot_out_dtype = torch.float32
    elif expected_out_dtype == torch.float16:
        dot_out_dtype = (
            torch.float32
            if common_dtype in {torch.float16, torch.bfloat16} and not fuse_acc
            else torch.float16
        )
    elif common_dtype == torch.int8 and expected_out_dtype == torch.int32:
        dot_out_dtype = torch.int32
    else:
        # Unsupported dtype (like bfloat16), use float32 and cast afterward
        dot_out_dtype = torch.float32

    # Squeeze 3D shapes to 2D when leading dims map to block size 1 for both operands.
    need_squeeze_dim = (
        len(lhs_shape_list) == 3
        and config is not None
        and all(
            block_idx is not None
            and env.block_sizes[block_idx].from_config(config) == 1
            for block_idx in (
                env.get_block_id(lhs_shape_list[0]),
                env.get_block_id(rhs_shape_list[0]),
            )
        )
    )

    if need_squeeze_dim:
        lhs_cast = expr_from_string(
            f"tl.reshape({{x}}, {shape_str(lhs_shape_list[1:])})", x=lhs_cast
        )
        rhs_cast = expr_from_string(
            f"tl.reshape({{x}}, {shape_str(rhs_shape_list[1:])})", x=rhs_cast
        )

        if acc_for_dot is not None:
            assert acc_shape_list is not None
            acc_for_dot = expr_from_string(
                f"tl.reshape({{x}}, {shape_str(acc_shape_list[1:])})",
                x=acc_for_dot,
            )

    min_m, min_n, min_k = min_dot_size(env.device, lhs_dtype, rhs_dtype)
    dims = {
        d: v if isinstance(v, int) else None
        for d, v in zip(("m", "n", "k"), [m, n, k], strict=False)
    }
    min_sizes = {"m": min_m, "n": min_n, "k": min_k}
    pad_needed = {}
    for d in ("m", "n", "k"):
        dim_value = dims[d]
        pad_needed[d] = dim_value is not None and dim_value < min_sizes[d]
    need_padding = any(pad_needed.values())

    if not need_padding:
        result = _emit_tl_dot(
            lhs_cast,
            rhs_cast,
            acc=acc_for_dot,
            input_precision=input_precision,
            out_dtype=dot_out_dtype,
        )
    else:
        lhs_pad, rhs_pad, acc_pad = lhs_cast, rhs_cast, acc_for_dot
        pad_specs = [
            ("lhs", "k", 1, min_k, m),
            ("rhs", "k", 0, min_k, n),
            ("lhs", "m", 0, min_m, min_k if pad_needed["k"] else k),
            ("rhs", "n", 1, min_n, min_k if pad_needed["k"] else k),
        ]
        pads = {"lhs": lhs_cast, "rhs": rhs_cast}
        for target, dim, axis, target_size, other_dim in pad_specs:
            if pad_needed[dim] and (cur := dims[dim]):
                pads[target] = _pad_tensor(
                    pads[target], axis, cur, target_size, other_dim
                )
        lhs_pad, rhs_pad = pads["lhs"], pads["rhs"]

        if acc_for_dot is not None and (pad_needed["m"] or pad_needed["n"]):
            assert acc_pad is not None  # acc_pad == acc_for_dot when we reach here
            acc_pad_specs = [
                ("m", 0, min_m, min_n if pad_needed["n"] and dims["n"] else n),
                ("n", 1, min_n, min_m if pad_needed["m"] and dims["m"] else m),
            ]
            for dim, axis, min_dim, other in acc_pad_specs:
                if pad_needed[dim] and (cur := dims[dim]):
                    acc_pad = _pad_tensor(acc_pad, axis, cur, min_dim, other)

        result = _emit_tl_dot(
            lhs_pad,
            rhs_pad,
            acc=acc_pad,
            input_precision=input_precision,
            out_dtype=dot_out_dtype,
        )

        unpad_specs = [
            (
                "n",
                min_n,
                min_m if pad_needed["m"] else m,
                lambda cur, other: [other, 2, cur],
                [0, 2, 1],
            ),
            (
                "m",
                min_m,
                dims["n"] or n,
                lambda cur, other: [2, cur, other],
                [1, 2, 0],
            ),
        ]
        for dim, min_dim, other, shape_fn, perm in unpad_specs:
            if pad_needed[dim] and (cur := dims[dim]):
                assert dim in ("m", "n"), f"dim must be 'm' or 'n', got {dim}"
                cur_size = min_dim
                while cur_size > cur:
                    cur_size //= 2
                    shape = shape_fn(cur_size, other)
                    result = expr_from_string(
                        f"tl.split(tl.permute(tl.reshape({{x}}, {shape_str(shape)}), {perm}))[0]",
                        x=result,
                    )

    if need_squeeze_dim:
        out_shape = [*lhs_shape_list[:-1], rhs_shape_list[-1]]
        result = expr_from_string(
            f"tl.reshape({{x}}, {shape_str(out_shape)})",
            x=result,
        )

    if acc_cast_dtype is not None:
        result = cast_ast(result, acc_cast_dtype)

    # Explicitly cast to expected output dtype if we used a different out_dtype for tl.dot and haven't already cast
    if dot_out_dtype != expected_out_dtype and acc_cast_dtype != expected_out_dtype:
        assert expected_out_dtype is not None
        result = cast_ast(result, expected_out_dtype)

    return (
        expr_from_string("{acc} + {mm}", acc=acc_out, mm=result)
        if not fuse_acc and acc_out is not None
        else result
    )
