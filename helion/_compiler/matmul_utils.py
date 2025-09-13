from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from torch._inductor.utils import triton_type

from .._compat import min_dot_size
from .ast_extension import expr_from_string
from .compile_environment import CompileEnvironment
from .dtype_utils import cast_ast

if TYPE_CHECKING:
    import ast
    import torch
    import sympy
    from ..runtime.config import Config


def emit_tl_dot(
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
    if acc is not None:
        kwargs["acc"] = acc
    
    parts = ["tl.dot({lhs}, {rhs}"]
    parts.extend([
        f", acc={{acc}}" if acc is not None else "",
        f", input_precision='{input_precision}'" if input_precision else "",
        f", out_dtype={triton_type(out_dtype)}" if out_dtype else "",
        ")"
    ])
    return expr_from_string("".join(parts), **kwargs)


def resolve_dim_to_int(v: object) -> int | None:
    """Resolve a dimension to an integer value if possible.
    
    Handles int, torch.SymInt, sympy expressions, and block size lookups.
    """
    if isinstance(v, int):
        return v
    
    env = CompileEnvironment.current()
    
    # Try to convert torch.SymInt or sympy expression to int if constant
    for attr in ["_sympy_", None]:
        try:
            expr = v._sympy_() if attr and hasattr(v, attr) else v
            if expr is not v:
                expr = env.shape_env.replace(expr)
            else:
                expr = env.shape_env.replace(v)  # type: ignore[arg-type]
            return int(expr)
        except Exception:
            continue
    
    return None


def resolve_dim_with_config(
    v: object, 
    config: Config | None = None,
    env: CompileEnvironment | None = None
) -> object:
    """Resolve a dimension value using both direct resolution and config lookup."""
    if env is None:
        env = CompileEnvironment.current()
    
    if (val := resolve_dim_to_int(v)) is not None:
        return val
    
    if config:
        try:
            import sympy
            import torch
            if isinstance(v, (torch.SymInt, sympy.Expr)) and (idx := env.get_block_id(v)) is not None:
                if (cfg := env.block_sizes[idx].from_config(config)) is not None:
                    return int(cfg)
        except Exception:
            pass
    
    return v


def emit_tl_dot_with_padding(
    lhs: ast.AST,
    rhs: ast.AST,
    acc: ast.AST | None,
    lhs_dtype: torch.dtype,
    rhs_dtype: torch.dtype,
    *,
    m: object,
    n: object,
    k: object,
    shape_str_fn: Callable[[list[object]], str] | None = None,
    acc_dtype: torch.dtype | None = None,
    input_precision: str | None = None,
    out_dtype: torch.dtype | None = None,
) -> ast.AST:
    """Emit a tl.dot operation with automatic padding for small dimensions."""
    device = CompileEnvironment.current().device
    min_m, min_n, min_k = min_dot_size(device, lhs_dtype, rhs_dtype)

    # Resolve dimensions and check padding requirements
    dims = {d: resolve_dim_to_int(v) for d, v in zip("mnk", [m, n, k])}
    pad_needed = {d: dims[d] is not None and dims[d] < min_val 
                  for d, min_val in zip("mnk", [min_m, min_n, min_k])}

    def _pad_tensor(tensor: ast.AST, pad_dim: int, cur_size: int, 
                    target_size: int, other_dim: object) -> ast.AST:
        """Pad tensor by repeatedly doubling specified dimension."""
        x = tensor
        while cur_size < target_size:
            x = expr_from_string("tl.join({x}, tl.zeros_like({x}))", x=x)
            perm = "[2, 0, 1]" if pad_dim == 0 else "[0, 2, 1]"
            x = expr_from_string(f"tl.permute({{x}}, {perm})", x=x)
            cur_size *= 2
            shape = [cur_size, other_dim] if pad_dim == 0 else [other_dim, cur_size]
            shape_str = shape_str_fn(shape) if shape_str_fn else str(shape)
            x = expr_from_string(f"tl.reshape({{x}}, {shape_str})", x=x)
        return x

    # Apply padding
    lhs_pad, rhs_pad = lhs, rhs
    
    if pad_needed['k'] and dims['k']:
        lhs_pad = _pad_tensor(lhs_pad, 1, dims['k'], min_k, m)
        rhs_pad = _pad_tensor(rhs_pad, 0, dims['k'], min_k, n)
    
    if pad_needed['m'] and dims['m']:
        lhs_pad = _pad_tensor(lhs_pad, 0, dims['m'], min_m, min_k if pad_needed['k'] else k)
    
    if pad_needed['n'] and dims['n']:
        rhs_pad = _pad_tensor(rhs_pad, 1, dims['n'], min_n, min_k if pad_needed['k'] else k)

    # Pad accumulator if needed
    acc_pad = acc
    if acc and (pad_needed['m'] or pad_needed['n']):
        for dim, min_dim, other in [('m', min_m, min_n if pad_needed['n'] and dims['n'] else n),
                                     ('n', min_n, min_m if pad_needed['m'] and dims['m'] else m)]:
            if pad_needed[dim] and dims[dim]:
                acc_pad = _pad_tensor(acc_pad, 0 if dim == 'm' else 1, dims[dim], min_dim, other)

    # Perform dot operation
    result = emit_tl_dot(lhs_pad, rhs_pad, acc=acc_pad, 
                        input_precision=input_precision, out_dtype=out_dtype)

    # Unpad result - process in reverse order (n then m)
    for dim in ['n', 'm']:
        if pad_needed[dim] and dims[dim]:
            min_dim = min_n if dim == 'n' else min_m
            other_dim = (min_m if pad_needed['m'] else m) if dim == 'n' else (dims['n'] or n)
            result = _unpad_result(result, dim, dims[dim], min_dim, other_dim, shape_str_fn)

    return cast_ast(result, acc_dtype) if acc_dtype else result


def _unpad_result(
    result: ast.AST,
    dim: str,
    target_size: int,
    padded_size: int,
    other_dim: object,
    shape_str_fn: Callable[[list[object]], str] | None = None
) -> ast.AST:
    """Unpad result tensor back to original size."""
    cur_size = padded_size
    while cur_size > target_size:
        shape = [cur_size // 2, 2, other_dim] if dim == 'm' else [other_dim, 2, cur_size // 2]
        shape_str = shape_str_fn(shape) if shape_str_fn else str(shape).replace("'", "")
        result = expr_from_string(f"tl.reshape({{x}}, {shape_str})", x=result)
        result = expr_from_string("tl.permute({x}, [0, 2, 1])", x=result)
        result = expr_from_string("tl.split({x})[0]", x=result)
        cur_size //= 2
    return result
