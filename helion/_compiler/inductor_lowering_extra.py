from __future__ import annotations

import contextlib
import functools
from typing import Any
from typing import Callable
from typing import Generator

import torch
from torch._inductor.ir import TensorBox
from torch._inductor.lowering import lowerings as original_lowerings
from torch._inductor.lowering import to_dtype
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND

inductor_lowering_dispatch: dict[Callable[..., Any] | str, Callable[..., Any]] = {}


def create_fp16_to_fp32_unary_fallback_lowering(
    original_op: Callable[..., object],
) -> Callable[..., object]:
    """Create a lowering that converts fp16/bfloat16 inputs to fp32 before calling the operation."""

    @functools.wraps(original_op)
    def fp32_fallback_lowering(x: object) -> object:
        if isinstance(x, TensorBox) and (original_dtype := x.get_dtype()) in (
            torch.float16,
            torch.bfloat16,
        ):
            x_fp32 = to_dtype(x, torch.float32)
            result_fp32 = original_op(x_fp32)
            assert isinstance(result_fp32, TensorBox)
            return to_dtype(result_fp32, original_dtype)
        return original_op(x)

    return fp32_fallback_lowering


# Operations that need fp32 fallbacks due to libdevice/tl_math limitations
FP32_FALLBACK_OPS_UNARY = [
    torch.ops.aten.rsqrt.default,  # pyright: ignore[reportAttributeAccessIssue]
    torch.ops.aten.sqrt.default,  # pyright: ignore[reportAttributeAccessIssue]
    torch.ops.aten.sin.default,  # pyright: ignore[reportAttributeAccessIssue]
    torch.ops.aten.cos.default,  # pyright: ignore[reportAttributeAccessIssue]
    torch.ops.aten.log.default,  # pyright: ignore[reportAttributeAccessIssue]
    torch.ops.aten.tanh.default,  # pyright: ignore[reportAttributeAccessIssue]
    torch.ops.aten.log1p.default,  # pyright: ignore[reportAttributeAccessIssue]
    torch.ops.aten.expm1.default,  # pyright: ignore[reportAttributeAccessIssue]
    torch.ops.aten.exp.default,  # pyright: ignore[reportAttributeAccessIssue]
]

# Register fp32 fallback lowerings for ops that don't support fp16/bfloat16
for op in FP32_FALLBACK_OPS_UNARY:
    inductor_lowering_dispatch[op] = create_fp16_to_fp32_unary_fallback_lowering(
        original_lowerings[op]
    )


@contextlib.contextmanager
def patch_inductor_lowerings() -> Generator[None, Any, Any]:
    """Context manager to temporarily patch the inductor lowering table.

    This is useful for overwriting specific Inductor lowerings without
    affecting the global state, especially in cases where Helion
    is missing support for a specific lowering.
    """
    original_lowerings = torch._inductor.lowering.lowerings.copy()  # pyright: ignore[reportAttributeAccessIssue]
    
    # Also need to patch Helion's aten_lowering_dispatch
    from . import inductor_lowering
    original_aten_dispatch = inductor_lowering.aten_lowering_dispatch.copy()
    
    try:
        torch._inductor.lowering.lowerings.update(inductor_lowering_dispatch)  # pyright: ignore[reportAttributeAccessIssue]
        
        # For prims operations, we need to ensure they're handled in prepare_node_lowering
        # We'll just make sure the torch._inductor.lowering.lowerings has the correct mappings
        # The proper handling will be done by the normal prepare_node_lowering flow
        
        yield
    finally:
        torch._inductor.lowering.lowerings = original_lowerings  # pyright: ignore[reportAttributeAccessIssue]
        inductor_lowering.aten_lowering_dispatch = original_aten_dispatch


def _register_inductor_lowering(
    aten_fn: object,
    decomp_fn: object,
    broadcast: bool,
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND | None,
    convert_input_to_bool: bool,
    lowering_dict: dict[object, Callable[..., object]],
) -> Callable[..., object]:
    from torch._inductor.lowering import fallbacks
    from torch._inductor.lowering import get_overloads
    from torch._inductor.lowering import in_namespace
    from torch._inductor.lowering import transform_args
    from torch._inductor.lowering import (
        validate_ir,  # pyright: ignore[reportPrivateImportUsage]
    )

    @functools.wraps(decomp_fn)  # pyright: ignore[reportArgumentType]
    def wrapped(*args: object, **kwargs: object) -> object:
        args = list(args)  # pyright: ignore[reportAssignmentType]
        kwargs = dict(kwargs)
        unpacked = False
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            unpacked = True
            args = list(  # pyright: ignore[reportAssignmentType]
                args[0]
            )

        if not all(
            (fn in fallbacks or in_namespace(fn, "_c10d_functional"))
            for fn in aten_fn  # pyright: ignore[reportGeneralTypeIssues]
        ):
            # explicitly assert for "out=" ops for better error messages
            assert not any(x == "out" for x in kwargs), "out= ops aren't yet supported"

        args, kwargs = (  # pyright: ignore[reportAssignmentType]
            transform_args(
                args,  # pyright: ignore[reportArgumentType]
                kwargs,
                broadcast,
                type_promotion_kind,
                convert_input_to_bool,
            )
        )

        if unpacked:
            args = [args]  # pyright: ignore[reportAssignmentType]

        out = decomp_fn(  # pyright: ignore[reportCallIssue]
            *args, **kwargs
        )
        validate_ir(out)

        return out

    aten_fn = get_overloads(aten_fn)

    lowering_dict.update(dict.fromkeys(aten_fn, wrapped))
    return wrapped


# Save reference to original inductor_random lowering before we override it
from torch._inductor.lowering import lowerings as inductor_lowerings
_original_inductor_random = inductor_lowerings[torch.ops.prims.inductor_random.default]


def helion_inductor_random_wrapper(size, seed, mode, *, offset=None):
    """Handle RNG operations for Helion with dynamic tile sizes."""
    from .rng_utils import process_dynamic_size
    
    # Process dynamic sizes using utility
    processed_size = process_dynamic_size(size)
    
    # Delegate to original inductor function
    return _original_inductor_random(processed_size, seed, mode) if offset is None else _original_inductor_random(processed_size, seed, mode, offset=offset)


# Dimension registry now handled by rng_utils module
from .rng_utils import _symbol_to_index

def get_dimension_index(sym: object) -> int | None:
    """Get the dimension index for a registered symbol."""
    return _symbol_to_index.get(sym)


# Register the wrapper for inductor_random
inductor_lowering_dispatch[torch.ops.prims.inductor_random.default] = helion_inductor_random_wrapper

# Also need to register inductor_seeds and lookup_seed for RNG to work properly
from torch._inductor import inductor_prims

# Save reference to original inductor_seeds and lookup_seed lowering
_original_inductor_seeds = inductor_lowerings[inductor_prims.seeds]
_original_lookup_seed = inductor_lowerings[inductor_prims.lookup_seed]

# Register seeds and lookup_seed lowerings
inductor_lowering_dispatch[torch.ops.prims.inductor_seeds.default] = _original_inductor_seeds
inductor_lowering_dispatch[torch.ops.prims.inductor_lookup_seed.default] = _original_lookup_seed

# For Helion's aten_lowering_dispatch, we need to add these as well
# This will be done in patch_inductor_lowerings

# Register lookup_seed lowering directly
inductor_lowering_dispatch[torch.ops.prims.inductor_lookup_seed.default] = _original_lookup_seed


# TODO(yf225): Switch to use upstream torch._inductor.lowering.register_lowering() after PyTorch 2.8 is released.
def register_inductor_lowering(
    aten_fn: object,
    broadcast: bool = False,
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND
    | None = ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    convert_input_to_bool: bool = False,
    lowering_dict: dict[Any, Callable[..., Any]] = inductor_lowering_dispatch,
) -> Callable[..., object]:
    return functools.partial(
        _register_inductor_lowering,
        aten_fn,
        broadcast=broadcast,
        type_promotion_kind=type_promotion_kind,
        convert_input_to_bool=convert_input_to_bool,
        lowering_dict=lowering_dict,
    )


def var_mean_helper_(
    x: torch._inductor.ir.TensorBox,  # pyright: ignore[reportAttributeAccessIssue]
    *,
    axis: list[int] | None,
    correction: float | None,
    keepdim: bool,
    return_mean: bool,
) -> torch._inductor.ir.TensorBox:  # pyright: ignore[reportAttributeAccessIssue]
    from torch._inductor.lowering import var_mean_sum_
    from torch._prims_common import get_computation_dtype

    out_dtype = x.get_dtype()
    compute_dtype = get_computation_dtype(out_dtype)
    x = to_dtype(x, compute_dtype, copy=False)

    kwargs = {
        "x": x,
        "axis": axis,
        "correction": correction,
        "keepdim": keepdim,
        "return_mean": return_mean,
    }
    # TODO(yf225): support Welford reduction in Helion, then switch back to use Inductor `var_mean_helper_()`.
    output = var_mean_sum_(**kwargs)
    output = tuple(to_dtype(o, out_dtype, copy=False) for o in output)
    return output[0] if not return_mean else output


@register_inductor_lowering(
    [torch.ops.aten.var.correction],  # pyright: ignore[reportAttributeAccessIssue]
    lowering_dict=inductor_lowering_dispatch,
)
def var_(
    x: torch._inductor.ir.TensorBox,  # pyright: ignore[reportAttributeAccessIssue]
    axis: list[int] | None = None,
    *,
    correction: float | None = None,
    keepdim: bool = False,
) -> torch._inductor.ir.TensorBox:  # pyright: ignore[reportAttributeAccessIssue]
    return var_mean_helper_(
        x,
        axis=axis,
        correction=correction,
        keepdim=keepdim,
        return_mean=False,
    )


@register_inductor_lowering(
    torch.ops.aten.var_mean.correction,  # pyright: ignore[reportAttributeAccessIssue]
    lowering_dict=inductor_lowering_dispatch,
)
def var_mean(
    x: torch._inductor.ir.TensorBox,  # pyright: ignore[reportAttributeAccessIssue]
    axis: list[int] | None = None,
    *,
    correction: float | None = None,
    keepdim: bool = False,
) -> torch._inductor.ir.TensorBox:  # pyright: ignore[reportAttributeAccessIssue]
    return var_mean_helper_(
        x,
        axis=axis,
        correction=correction,
        keepdim=keepdim,
        return_mean=True,
    )
