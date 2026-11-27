"""Known starting contents for kernel outputs, so a store that never ran shows.

``torch.empty`` returns whatever block the allocator hands out, usually the
one an earlier launch's result just freed. A test that compares two launches
bit for bit then passes for every element the second launch left unwritten:
the block still holds the first launch's values. Outputs the caller
preallocates, and the outputs of captured calls, which every replay stores
into the same memory, keep an earlier result the same way.

These helpers start every output as a quiet NaN with a payload that neither
the device's default NaN nor any test input carries: ``poisoned_launch`` for
one eager call, ``poison_`` for tensors the test owns (captured outputs ahead
of a replay), ``written_bits`` to read a result back. Outputs are BF16 or
FP32; a tensor of another dtype has no payload, and the helpers that are
handed one say so.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Sequence

# Output dtype -> (integer dtype of the same width, payload bits).
_PAYLOADS = {
    torch.bfloat16: (torch.int16, 0x7FDE),
    torch.float32: (torch.int32, 0x7FC0DEAD),
}


def _payload(tensor: torch.Tensor) -> tuple[torch.dtype, int]:
    """The integer dtype and the payload bits for ``tensor``'s dtype."""
    if tensor.dtype not in _PAYLOADS:
        supported = " and ".join(str(dtype) for dtype in _PAYLOADS)
        raise TypeError(
            f"no poison payload for an output of dtype {tensor.dtype}: "
            f"poisoned outputs are {supported}"
        )
    return _PAYLOADS[tensor.dtype]


def poison_(*tensors: torch.Tensor) -> None:
    """Fill each tensor with the payload of its dtype."""
    for tensor in tensors:
        integer, payload = _payload(tensor)
        tensor.view(integer).fill_(payload)


def written_bits(tensor: torch.Tensor) -> torch.Tensor:
    """A copy of the tensor's bits, once no element holds the payload.

    As integers, equal NaN payloads and signed zeros compare exactly.
    """
    integer, payload = _payload(tensor)
    bits = tensor.view(integer).clone()
    unwritten = (bits == payload).reshape(bits.size(0), -1).any(dim=1).nonzero()
    assert unwritten.numel() == 0, (
        f"no store reached {unwritten.numel()} of {bits.size(0)} leading "
        f"indices, first {unwritten.flatten()[:8].tolist()}"
    )
    return bits


@contextlib.contextmanager
def poisoned_empty() -> Iterator[list[torch.Tensor]]:
    """Poison what ``torch.empty`` returns inside the block.

    Yields the tensors it filled: those of a dtype with a payload. Not for a
    CUDA graph capture, where the fill would become a node of the graph.
    """
    fresh: list[torch.Tensor] = []
    empty: Callable[..., torch.Tensor] = torch.empty

    def poisoned(*args: object, **kwargs: object) -> torch.Tensor:
        tensor = empty(*args, **kwargs)
        if tensor.dtype in _PAYLOADS:
            poison_(tensor)
            fresh.append(tensor)
        return tensor

    with patch.object(torch, "empty", poisoned):
        yield fresh


def poisoned_launch(
    launch: Callable[[], object], preallocated: Sequence[torch.Tensor] = ()
) -> list[torch.Tensor]:
    """Run ``launch`` and return the bits of its outputs, all written by it.

    The outputs the kernel allocates with ``torch.empty`` and the
    ``preallocated`` ones it stores into start as the payload. An output that
    started as neither is refused: it could equal a reference by keeping an
    earlier launch's values. So is an output of a dtype without a payload,
    by a ``TypeError`` that names the dtype.
    """
    poison_(*preallocated)
    with poisoned_empty() as fresh:
        outputs = cast("tuple[torch.Tensor, ...]", launch())
    started = (*fresh, *preallocated)
    bits = []
    for output in outputs:
        # ``torch.empty`` left an output of another dtype as it was.
        _payload(output)
        assert any(output is tensor for tensor in started), (
            "an output did not start poisoned: allocate it with torch.empty "
            "inside the launch or pass it as preallocated"
        )
        bits.append(written_bits(output))
    return bits
