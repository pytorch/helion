"""Autograd integration for Helion kernels.

When ``HELION_EXPERIMENTAL_DIFFERENTIABLE=1`` is set, ``Kernel.__call__``
routes through :func:`call_with_autograd` so that the returned tensor
participates in the autograd graph.  The backward kernel is auto-generated
by :func:`helion.experimental.backward`.
"""

from __future__ import annotations

from typing import Any

import torch

from .. import exc


class _HelionAutogradFunction(torch.autograd.Function):
    """Autograd glue between a Helion forward kernel and its auto-generated
    backward kernel."""

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx: Any,  # noqa: ANN401
        kernel: Any,  # noqa: ANN401
        *args: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        tensors: list[torch.Tensor] = []
        scalars: list[object] = []
        is_tensor: list[bool] = []
        for a in args:
            if isinstance(a, torch.Tensor):
                tensors.append(a)
                is_tensor.append(True)
            else:
                scalars.append(a)
                is_tensor.append(False)

        ctx.save_for_backward(*tensors)
        ctx.kernel = kernel
        ctx.arg_is_tensor = is_tensor
        ctx.scalar_values = scalars
        # Call via bind() to bypass Kernel.__call__ and avoid recursing
        # back into the autograd guard.
        return kernel.bind(args)(*args)

    @staticmethod
    def backward(
        ctx: Any,  # noqa: ANN401
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        import helion.experimental

        saved = ctx.saved_tensors
        kernel = ctx.kernel

        # Rebuild the original arg list with detached saved tensors in their
        # original positions and scalars interleaved.
        full_args: list[object] = []
        ti = 0
        si = 0
        for is_tensor in ctx.arg_is_tensor:
            if is_tensor:
                full_args.append(saved[ti].detach())
                ti += 1
            else:
                full_args.append(ctx.scalar_values[si])
                si += 1

        # enable_grad() is needed because helion.experimental.backward()
        # uses aot_module_simplified on the first call, which requires
        # grad-enabled tensors to extract the backward graph.
        with torch.enable_grad():
            grad_out: torch.Tensor | tuple[torch.Tensor, ...]
            if len(grad_outputs) == 1:
                grad_out = grad_outputs[0]
            else:
                grad_out = grad_outputs
            try:
                grads = helion.experimental.backward(kernel, grad_out, *full_args)
            except exc.AutodiffNotSupported as e:
                raise exc.AutodiffNotSupported(
                    f"{e}\n\nTo work around this, write a manual "
                    f"torch.autograd.Function for this kernel."
                ) from e

        grad_tuple: tuple[torch.Tensor, ...]
        if isinstance(grads, torch.Tensor):
            grad_tuple = (grads,)
        else:
            grad_tuple = grads  # pyrefly: ignore [bad-assignment]
        # None for the prepended kernel arg, then one entry per original arg.
        result: list[torch.Tensor | None] = [None]
        gi = 0
        for is_tensor in ctx.arg_is_tensor:
            if is_tensor and gi < len(grad_tuple):
                result.append(grad_tuple[gi])
                gi += 1
            else:
                result.append(None)
        return tuple(result)


def call_with_autograd(kernel: object, *args: object) -> object:
    """Route a kernel call through autograd.

    Called from ``Kernel.__call__`` when the env var is set and at least
    one tensor argument has ``requires_grad=True``.
    """
    return _HelionAutogradFunction.apply(kernel, *args)
