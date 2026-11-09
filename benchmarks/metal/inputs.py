"""Deterministic host inputs, hashing, and conversion to each runtime.

Inputs are generated once per cell on the CPU with a seeded generator, in the
dtype the operator runs at.  Every provider and the float64 reference start
from these same bytes.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import mlx.core as mx

    from .manifest import Case

SEED = 20261019


def generator(case: Case) -> torch.Generator:
    """A CPU generator seeded per (kernel, case) so cells are independent."""
    digest = hashlib.sha256(f"{SEED}:{case.kernel}:{case.index}".encode())
    return torch.Generator().manual_seed(int.from_bytes(digest.digest()[:8], "big"))


def input_hash(inputs: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, tensor in inputs.items():
        h.update(
            f"{name}:{tensor.dtype}:{list(tensor.shape)}:{tensor.stride()}".encode()
        )
        h.update(_raw_bytes(tensor.contiguous()))
    return h.hexdigest()


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy().tobytes()
    return tensor.numpy().tobytes()


_MLX_DTYPES = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.uint32: "uint32",
    torch.bool: "bool_",
}


def to_mlx(tensor: torch.Tensor) -> mx.array:
    import mlx.core as mx

    tensor = tensor.contiguous()
    if tensor.dtype == torch.bfloat16:
        return mx.array(tensor.view(torch.uint16).numpy()).view(mx.bfloat16)
    array = mx.array(tensor.numpy())
    assert array.dtype == getattr(mx, _MLX_DTYPES[tensor.dtype])
    return array


def from_mlx(array: mx.array) -> torch.Tensor:
    import mlx.core as mx

    assert isinstance(array, mx.array)
    if array.dtype == mx.bfloat16:
        return torch.from_numpy(np.array(array.view(mx.uint16))).view(torch.bfloat16)
    return torch.from_numpy(np.array(array))
