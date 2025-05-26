"""
NOTE: This file is only designed to be imported within `patch_inductor_lowerings()` function.
"""

from __future__ import annotations

import torch
from torch._inductor.lowering import register_lowering as register_inductor_lowering


def var_mean_helper_(
    x: torch._inductor.ir.TensorBox,
    *,
    axis: list[int] | None,
    correction: float | None,
    keepdim: bool,
    return_mean: bool,
) -> torch._inductor.ir.IRNode:
    from torch._inductor.lowering import to_dtype
    from torch._inductor.lowering import mean, square, sub, sum_
    from torch._inductor.lowering import _validate_reduction_axis
    from torch._inductor.lowering import sympy_product
    from torch._inductor import ir
    from torch._prims_common import get_computation_dtype
    import sympy

    # Custom var_mean implementation that properly handles block size substitution
    out_dtype = x.get_dtype()
    compute_dtype = get_computation_dtype(out_dtype)
    x = to_dtype(x, compute_dtype, copy=False)
    
    if correction is None:
        correction = 1

    size = x.get_size()
    axis = _validate_reduction_axis(x, axis)
    x_mean = mean(x, axis, keepdim=True)
    if return_mean:
        x_mean.realize()

    diffs = square(sub(x, x_mean))
    sum_result = sum_(diffs, axis, keepdim)

    # Compute denominator with proper handling for block sizes
    denom = sympy_product(size[i] for i in axis)
    
    # Check if we're dividing by a block size that has been rounded up
    from .compile_environment import CompileEnvironment
    from .tile_strategy import TileStrategy
    from .host_function import HostFunction
    from .variable_origin import SymIntOrigin
    
    env = CompileEnvironment.current()
    host_fn = HostFunction.current()
    
    # Check if denom contains block size symbols that need substitution
    if isinstance(denom, sympy.Expr) and denom.free_symbols:
        replacements = {}
        for sym in denom.free_symbols:
            block_idx = TileStrategy.get_block_index(sym)
            if block_idx is not None and block_idx < len(env.block_sizes):
                block_info = env.block_sizes[block_idx]
                if block_info.has_mask():
                    # Create a symbol for the actual dimension size
                    actual_sym = sympy.Symbol(f"s{block_idx}", integer=True, positive=True)
                    # Register this symbol with the host function
                    if actual_sym not in host_fn.expr_to_origin:
                        origin = SymIntOrigin(f"actual_size_{block_idx}")
                        from .host_function import SymbolOrigin
                        expr_origin = SymbolOrigin(
                            origin=origin,
                            fake_value=getattr(block_info.block_size_source, 'actual_value', block_info.size) or block_info.size,
                        )
                        host_fn.expr_to_origin[actual_sym] = expr_origin
                    replacements[sym] = actual_sym
        
        if replacements:
            denom = denom.xreplace(replacements)
    
    if correction:
        denom = sympy.Max(denom - correction, 0)
    
    denom = ir.IndexingConstant(index=denom, dtype=x.get_dtype(), device=x.get_device())
    denom = ir.ExpandView.create(denom, list(sum_result.get_size()))
    from torch._inductor.lowering import div
    x_var = div(sum_result, denom)
    
    # Convert back to original dtype
    x_var = to_dtype(x_var, out_dtype, copy=False)
    
    if not return_mean:
        return (x_var,)

    x_mean = to_dtype(x_mean, out_dtype, copy=False)
    from torch._inductor.lowering import squeeze
    x_mean = x_mean if keepdim else squeeze(x_mean, axis)
    return x_var, x_mean


@register_inductor_lowering(torch.ops.aten.var_mean.correction)  # pyre-ignore[56]
def var_mean(
    x: torch._inductor.ir.TensorBox,
    axis: list[int] | None = None,
    *,
    correction: float | None = None,
    keepdim: bool = False,
) -> torch._inductor.ir.IRNode:
    return var_mean_helper_(
        x, axis=axis, correction=correction, keepdim=keepdim, return_mean=True
    )
