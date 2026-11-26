"""Trace device assertions without evaluating their predicate during compilation."""

from __future__ import annotations

from contextlib import ExitStack
from contextlib import contextmanager
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental import proxy_tensor

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence
    from contextlib import AbstractContextManager


_ASSERTION_CALLS = (
    torch._assert_async,
    torch.ops.aten._assert_async,
    torch.ops.aten._assert_async.default,
    torch.ops.aten._assert_async.msg,
)


def device_assertion_context(
    fn: object, args: Sequence[object], kwargs: Mapping[str, object]
) -> AbstractContextManager[None]:
    if any(fn is check for check in _ASSERTION_CALLS):
        return _abstract_assertion(
            args[0] if args else kwargs.get("input", kwargs.get("self"))
        )
    return nullcontext()


@contextmanager
def _abstract_assertion(predicate: object) -> Iterator[None]:
    # Keep the real predicate and its FX node. Both FakeTensor and make_fx can
    # otherwise execute a constant false value while tracing an untaken branch.
    with ExitStack() as stack:
        if isinstance(predicate, FakeTensor) and predicate.constant is not None:
            mode = proxy_tensor.get_proxy_mode()
            if mode is not None:
                slot = proxy_tensor.fetch_object_proxy(mode.tracer, predicate)
                if isinstance(slot, proxy_tensor._ProxyTensor):
                    stack.callback(setattr, slot, "constant", slot.constant)
                    slot.constant = None
            stack.callback(setattr, predicate, "constant", predicate.constant)
            predicate.constant = None
        yield
    # The ATen meta kernel checks the call signature, but not the single-element
    # contract of its CPU/CUDA implementation. Do not specialize symbolic sizes.
    if isinstance(predicate, torch.Tensor):
        numel = predicate.numel()
        if isinstance(numel, int) and numel != 1:
            raise RuntimeError("a device assertion requires a tensor with one element")
