"""Lower multi-axis any/all through exact integer-extrema reduction chains.

The existing reduction merger retains its backend-specific static-shape,
adjacent-axis, view and alias requirements; this is no new layout admission.
"""

from __future__ import annotations

from typing import Callable
from typing import cast

import torch
from torch._prims_common import canonicalize_dims


def _chain_logical_dims(
    reduction: torch._ops.OpOverload,
    identity: int,
    fallback: Callable[..., object] | None,
) -> Callable[..., object]:
    def decomp(x: torch.Tensor, *args: object, **kwargs: object) -> object:
        dim = args[0] if args else kwargs.get("dim")
        keepdim = args[1] if len(args) > 1 else kwargs.get("keepdim", False)
        if x.ndim > 1 and not isinstance(dim, int):
            dims = list(range(x.ndim)) if dim is None else list(cast("list[int]", dim))
            normalized = list(canonicalize_dims(x.ndim, dims))
            if len(normalized) >= 2 and len(set(normalized)) == len(normalized):
                # Use integer extrema: bool extrema decompose back to
                # any/all, whose native reduction is not available on every
                # backend. Non-byte values become exactly 0 or 1; uint8
                # extrema preserve zero/nonzero truth without a preliminary cast.
                out_dtype = torch.uint8 if x.dtype == torch.uint8 else torch.bool
                if x.dtype != torch.uint8:
                    x = x.to(torch.bool).to(torch.uint8)
                # Extrema do not have eager identities for empty tensors,
                # whereas logical reductions do. Materialize that result before
                # delegating the nonempty multi-axis operation to the existing
                # reduction-chain decomposition and its layout/alias proof.
                if any(
                    isinstance(extent := x.size(axis), int) and extent == 0
                    for axis in normalized
                ):
                    shape = [
                        1 if axis in normalized else extent
                        for axis, extent in enumerate(x.shape)
                        if keepdim or axis not in normalized
                    ]
                    return torch.full(shape, identity, dtype=out_dtype, device=x.device)
                x = reduction(x, normalized, keepdim)
                # Also normalize the integer min identity if a dynamic empty
                # domain is handled by the backend's reduction mask.
                return x.to(torch.bool).to(out_dtype)
        # Preserve the original scalar, empty-axis-list and duplicate-axis
        # decomposition. Invalid axes above are rejected by canonicalize_dims.
        return NotImplemented if fallback is None else fallback(x, *args, **kwargs)

    return decomp


def install_logical_reduction_decompositions(
    table: dict[torch._ops.OpOverload, Callable[..., object]],
) -> None:
    for packet, reduction, identity in (
        (torch.ops.aten.any, torch.ops.aten.amax.default, 0),
        (torch.ops.aten.all, torch.ops.aten.amin.default, 1),
    ):
        for name in ("default", "dims"):
            op = getattr(packet, name)
            table[op] = _chain_logical_dims(reduction, identity, table.get(op))
