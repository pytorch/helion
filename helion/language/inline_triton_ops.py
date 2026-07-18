from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import inspect
from typing import TYPE_CHECKING
from typing import TypeVar

import torch
from torch.fx import has_side_effect

from .. import exc
from . import _decorators

if TYPE_CHECKING:
    _T = TypeVar("_T")

__all__ = ["inline_triton", "triton_kernel"]


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def inline_triton(
    triton_source: str,
    args: Sequence[object] | Mapping[str, object],
    output_like: _T,
) -> _T:
    """Inline a raw Triton snippet inside a Helion kernel.

    Args:
        triton_source: The Triton code snippet. The last statement must be an
            expression representing the return value. The snippet may be
            indented, and common indentation is stripped automatically.
        args: Positional or keyword placeholders that will be substituted via
            ``str.format`` before code generation. Provide a tuple/list for
            positional placeholders (``{0}``, ``{1}``, ...) or a mapping for
            named placeholders (``{x}``, ``{y}``, ...).
        output_like: Example tensors describing the expected outputs. A single
            tensor indicates a single output; a tuple/list of tensors indicates
            multiple outputs.

    Returns:
        The value(s) produced by the snippet. Matches the structure of
        ``output_like``.
    """
    raise exc.NotInsideKernel


def _validate_args(args: object) -> None:
    if isinstance(args, Mapping):
        return
    if isinstance(args, Sequence) and not isinstance(args, (str, bytes)):
        return
    raise exc.InvalidAPIUsage("inline_triton args must be a tuple/list or a mapping")


def _fake_outputs(output_like: object) -> object:
    if output_like is None:
        return None
    if isinstance(output_like, torch.Tensor):
        return torch.empty_like(output_like)
    if isinstance(output_like, Sequence) and not isinstance(output_like, (str, bytes)):
        outputs = []
        for i, item in enumerate(output_like):
            if not isinstance(item, torch.Tensor):
                raise exc.InvalidAPIUsage(
                    f"output_like[{i}] must be a torch.Tensor, got {type(item)}"
                )
            outputs.append(torch.empty_like(item))
        return type(output_like)(outputs) if isinstance(output_like, tuple) else outputs
    raise exc.InvalidAPIUsage(
        "output_like must be a tensor or a sequence of tensors or None"
    )


@_decorators.register_fake(inline_triton)
def _(
    triton_source: str,
    args: object,
    output_like: object,
) -> object:
    if not isinstance(triton_source, str):
        raise exc.InvalidAPIUsage(
            f"triton_source must be a string, got {type(triton_source)}"
        )
    _validate_args(args)
    return _fake_outputs(output_like)


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def triton_kernel(
    triton_source_or_fn: object,
    args: Sequence[object] | Mapping[str, object],
    output_like: _T,
) -> _T:
    """
    Define (once) and call a @triton.jit function from Helion device code.

    Args:
        triton_source_or_fn: Source for a single @triton.jit function definition,
            or a Python function object defining a @triton.jit kernel.
        args: Positional or keyword placeholders that will be substituted via
            name resolution of Helion variables.
        output_like: Example tensor(s) describing the expected outputs for shape/dtype checks.
    """
    raise exc.NotInsideKernel


@_decorators.register_fake(triton_kernel)
def _(
    triton_source_or_fn: object,
    args: object,
    output_like: object,
) -> object:
    if not (
        isinstance(triton_source_or_fn, str) or inspect.isfunction(triton_source_or_fn)
    ):
        raise exc.InvalidAPIUsage(
            f"triton_kernel expects a string source or a function, got {type(triton_source_or_fn)}"
        )
    _validate_args(args)
    return _fake_outputs(output_like)
