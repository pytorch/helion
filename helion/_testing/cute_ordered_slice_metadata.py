"""Scalar sequence slicing around dependency-proved complete reductions."""

from __future__ import annotations

from types import ModuleType

import torch

import helion
import helion.language as hl

slice_policy = ModuleType("_ordered_slice_policy")
slice_policy.__dict__["stop"] = -1


def opaque_stop() -> int:
    return -1


class OpaqueSequence(ModuleType):
    def __getitem__(self, key: slice) -> tuple[int, ...]:
        return (3, 32, 32)[key]


opaque_sequence = OpaqueSequence("_ordered_opaque_sequence")


@helion.kernel(backend="cute", static_shapes=True)
def sliced_metadata_votes(x: torch.Tensor, mode: hl.constexpr) -> torch.Tensor:
    if mode == "prefix":
        assert x.shape[:-1] == (3, 32)
    elif mode == "bounds":
        assert x.shape[0 : x.ndim - 1 : 1] == (3, 32)
    elif mode == "reverse":
        assert x.shape[::-1] == (32, 32, 3)
    elif mode == "local":
        metadata = (x.size(0), x.size(1), x.size(2))
        assert metadata[::2] == (3, 32)
    elif mode == "mutable":
        assert x.shape[: slice_policy.stop] == (3, 32)
    elif mode == "opaque_bound":
        assert x.shape[: opaque_stop()] == (3, 32)
    elif mode == "opaque_container":
        assert opaque_sequence[:-1] == (3, 32)
    elif mode == "tensor":
        view = x[:-1]
        assert view.size(0) == 2
    partial = torch.empty_like(x)
    output = torch.empty(x.shape[:1], dtype=torch.bool, device=x.device)
    for row in hl.tile(x.size(0), block_size=1):
        values = x[row.id, :, :]
        partial[row.id, :, :] = values + torch.any(values > 2)
    for row in hl.tile(x.size(0), block_size=1):
        values = partial[row.id, :, :]
        output[row.id] = torch.any(values > 0)
    return output
