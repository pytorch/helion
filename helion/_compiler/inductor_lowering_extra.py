from __future__ import annotations

import contextlib
import functools
from typing import Any
from typing import Callable
from typing import Generator

import torch
from torch._inductor.lowering import to_dtype
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND

inductor_lowering_dispatch: dict[  # pyre-ignore[5]
    Callable[..., Any] | str, Callable[..., Any]
] = {}


@contextlib.contextmanager
def patch_inductor_lowerings() -> Generator[  # pyre-ignore[3]
    None, Any, Any
]:
    """Context manager to temporarily patch the inductor lowering table.

    This is useful for overwriting specific Inductor lowerings without
    affecting the global state, especially in cases where Helion
    is missing support for a specific lowering.
    """
    original_lowerings = torch._inductor.lowering.lowerings.copy()
    try:
        torch._inductor.lowering.lowerings.update(inductor_lowering_dispatch)
        yield
    finally:
        torch._inductor.lowering.lowerings = original_lowerings


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
    from torch._inductor.lowering import validate_ir

    @functools.wraps(decomp_fn)  # pyre-ignore[6]
    def wrapped(*args: Any, **kwargs: Any) -> object:
        args = list(args)  # pyre-ignore[9]
        kwargs = dict(kwargs)
        unpacked = False
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            unpacked = True
            args = list(args[0])  # pyre-ignore[9]

        if not all(
            (fn in fallbacks or in_namespace(fn, "_c10d_functional"))
            for fn in aten_fn  # pyre-ignore[16]
        ):
            # explicitly assert for "out=" ops for better error messages
            assert not any(x == "out" for x in kwargs), "out= ops aren't yet supported"

        args, kwargs = transform_args(  # pyre-ignore[9]
            args,  # pyre-ignore[6]
            kwargs,
            broadcast,
            type_promotion_kind,
            convert_input_to_bool,
        )

        if unpacked:
            args = [args]  # pyre-ignore[9]

        out = decomp_fn(*args, **kwargs)  # pyre-ignore[29]
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
    lowering_dict: dict[  # pyre-ignore[2]
        Any, Callable[..., Any]
    ] = inductor_lowering_dispatch,
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
    x: torch._inductor.ir.TensorBox,
    *,
    axis: list[int] | None,
    correction: float | None,
    keepdim: bool,
    return_mean: bool,
) -> torch._inductor.ir.TensorBox:
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
    [torch.ops.aten.var.correction], lowering_dict=inductor_lowering_dispatch
)
def var_(
    x: torch._inductor.ir.TensorBox,
    axis: list[int] | None = None,
    *,
    correction: float | None = None,
    keepdim: bool = False,
) -> torch._inductor.ir.TensorBox:
    return var_mean_helper_(
        x,
        axis=axis,
        correction=correction,
        keepdim=keepdim,
        return_mean=False,
    )


@register_inductor_lowering(  # pyre-ignore[56]
    torch.ops.aten.var_mean.correction, lowering_dict=inductor_lowering_dispatch
)
def var_mean(
    x: torch._inductor.ir.TensorBox,
    axis: list[int] | None = None,
    *,
    correction: float | None = None,
    keepdim: bool = False,
) -> torch._inductor.ir.TensorBox:
    return var_mean_helper_(
        x,
        axis=axis,
        correction=correction,
        keepdim=keepdim,
        return_mean=True,
    )
