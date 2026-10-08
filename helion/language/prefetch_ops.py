from __future__ import annotations

import math
from typing import TYPE_CHECKING

from torch.fx import has_side_effect

from .. import exc
from . import _decorators
from .tile_proxy import Tile

if TYPE_CHECKING:
    import torch

__all__ = ["prefetch"]


@has_side_effect
@_decorators.api(allow_host_tensor=True)
def prefetch(tensor: torch.Tensor, index: list[object]) -> None:
    """Prefetch ``tensor[index]`` into L2 with one bulk prefetch from one thread.

    Scalar indices select leading dimensions; the remaining suffix must be one
    dense byte range of static size. It is issued in place. Prefetching never
    changes values.

    Args:
        tensor: The tensor to prefetch from
        index: Scalar indices into the leading dimensions of ``tensor``
    """
    raise exc.NotInsideKernel


def prefetch_nbytes(tensor: torch.Tensor, index: list[object]) -> int:
    """Byte size of the dense region ``tensor[index]`` selects."""
    if len(index) > tensor.ndim or any(
        isinstance(i, (slice, type(None), list, Tile)) for i in index
    ):
        raise exc.InvalidPrefetchRegion("indices must be scalars")
    dims = [
        *zip(tensor.stride()[len(index) :], tensor.shape[len(index) :], strict=True)
    ]
    if not all(isinstance(v, int) for dim in dims for v in dim):
        raise exc.InvalidPrefetchRegion("sizes and strides must be specialized")
    expected = 1
    for stride, size in sorted(dims):
        if size != 1 and stride != expected:
            raise exc.InvalidPrefetchRegion("the region must be dense")
        expected *= size
    return math.prod(size for _stride, size in dims) * tensor.element_size()


@_decorators.register_fake(prefetch)
def _(tensor: torch.Tensor, index: list[object]) -> None:
    prefetch_nbytes(tensor, index)
    return None


@_decorators.ref(prefetch)
def _(tensor: torch.Tensor, index: list[object]) -> None:
    return None
