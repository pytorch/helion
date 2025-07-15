from __future__ import annotations

import contextlib
import functools
from typing import Any
from typing import Callable
from typing import Generator

import torch
from torch._inductor.lowering import to_dtype
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND

inductor_lowering_dispatch: dict[Callable[..., Any] | str, Callable[..., Any]] = {}


@contextlib.contextmanager
def patch_inductor_lowerings() -> Generator[None, Any, Any]:
    """Context manager to temporarily patch the inductor lowering table.

    This is useful for overwriting specific Inductor lowerings without
    affecting the global state, especially in cases where Helion
    is missing support for a specific lowering.
    """
    original_lowerings = torch._inductor.lowering.lowerings.copy()  # pyright: ignore[reportAttributeAccessIssue]
    try:
        torch._inductor.lowering.lowerings.update(inductor_lowering_dispatch)  # pyright: ignore[reportAttributeAccessIssue]
        yield
    finally:
        torch._inductor.lowering.lowerings = original_lowerings  # pyright: ignore[reportAttributeAccessIssue]


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
