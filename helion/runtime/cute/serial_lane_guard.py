"""Current-call contracts for opt-in serial-lane scheduling.

These checks precede *every* launcher invocation, including cache hits. They
never retain tensors or specialize a route using a previous call's pointers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc

if TYPE_CHECKING:
    from collections.abc import Sequence

# Compiler-emitted immutable literals, never sample tensors or bound objects.
Contract = tuple[object, ...]


def validate_trace() -> None:
    # This check must execute *outside* the disabled metadata helpers.
    if (
        torch.compiler.is_compiling()
        or torch.jit.is_tracing()
        or torch.jit.is_scripting()
    ):
        _unsupported("tracing/export is not supported by this schedule")


def _unsupported(reason: str) -> None:
    raise exc.BackendUnsupported("cute", f"serial lane schedule: {reason}")


def _require_cuda(value: torch.Tensor) -> None:
    if value.device.type != "cuda":
        _unsupported("requires CUDA tensors")


def _tensor_span(value: object, contract: Contract) -> tuple[torch.device, int, int]:
    if type(value) is not torch.Tensor:
        _unsupported("requires plain tensor arguments")
    tensor = cast("torch.Tensor", value)
    if tensor.layout != torch.strided:
        _unsupported("requires strided tensor layout")
    if tensor.is_conj() or tensor.is_neg() or tensor.is_quantized:
        _unsupported("lazy conjugate/negative or quantized tensor")
    if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        _unsupported("unsupported tensor dtype")
    _require_cuda(tensor)
    _, shape, stride, dtype, alignment = contract
    if (
        tuple(tensor.shape) != shape
        or tuple(tensor.stride()) != stride
        or str(tensor.dtype) != dtype
        or tensor.ndim == 0
        or any(size <= 0 for size in tensor.shape)
        or any(step < 0 for step in tensor.stride())
    ):
        _unsupported("changed compiled shape, stride or dtype")
    pointer = tensor.data_ptr()
    if pointer % max(cast("int", alignment), tensor.element_size()):
        _unsupported("pointer violates the ordinary launcher alignment")
    storage = tensor.untyped_storage()
    start = storage.data_ptr()
    end = start + storage.nbytes()
    limit = pointer + tensor.element_size() * (
        1
        + sum(
            (size - 1) * step
            for size, step in zip(tensor.shape, tensor.stride(), strict=True)
        )
    )
    if not (0 < start <= pointer < limit <= end < 2**64):
        _unsupported("invalid current backing span")
    return tensor.device, start, end


def _validate(
    values: Sequence[object], contracts: tuple[Contract, ...]
) -> list[tuple[torch.device, int, int] | None]:
    if len(values) != len(contracts):
        _unsupported("missing original or launch dependency")
    spans: list[tuple[torch.device, int, int] | None] = []
    device = None
    for value, contract in zip(values, contracts, strict=True):
        if contract[0] == "tensor":
            span = _tensor_span(value, contract)
            if device is not None and span[0] != device:
                _unsupported("mixed tensor devices")
            device = span[0]
            spans.append(span)
        else:
            _, kind, expected = contract
            if type(value).__name__ != kind or type(value) not in (int, float, bool):
                _unsupported("changed specialized scalar type")
            actual = value.hex() if type(value) is float else value
            if actual != expected:
                _unsupported("changed specialized scalar value")
            spans.append(None)
    return spans


@torch.compiler.disable
def validate_entry(values: tuple[object, ...], contracts: tuple[Contract, ...]) -> None:
    """Called before the generated host can overwrite a public argument."""
    _validate(values, contracts)


@torch.compiler.disable
def select_kernel(
    original: object,
    reordered: object,
    values: tuple[object, ...],
    contracts: tuple[Contract, ...],
    writes: tuple[int, ...],
) -> object:
    """Check actual launch locals, then select without caching the decision."""
    spans = _validate(values, contracts)
    for left, a in enumerate(spans):
        if a is None:
            continue
        for right in range(left + 1, len(spans)):
            b = spans[right]
            if b is None or (left not in writes and right not in writes):
                continue
            if not (a[2] <= b[1] or b[2] <= a[1]):
                return original
    return reordered
