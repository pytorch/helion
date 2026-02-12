"""Pallas CPU launcher for Helion kernels.

This module is imported lazily to avoid requiring JAX at import time.
"""

from __future__ import annotations

import jax  # pyrefly: ignore[import-error]
from jax.experimental import pallas as pl  # pyrefly: ignore[import-error]
import jax.numpy as jnp  # pyrefly: ignore[import-error]
import numpy as np
import torch


def pallas_launcher(
    pallas_kernel: object,  # Callable taking Ref args
    grid: tuple[int, ...],
    *args: object,
    **kwargs: object,
) -> None:
    """Execute a Pallas kernel on CPU using pallas_call with interpret=True.

    Uses BlockSpec to tile inputs/outputs according to the grid.  Block shape
    is inferred from the output tensor shape and grid dimensions.

    Args:
        pallas_kernel: The Pallas kernel function (takes Ref args).
        grid: Grid dimensions for the kernel launch.
        *args: Tensor args. The last tensor is treated as the output.
    """
    tensor_args = [a for a in args if isinstance(a, torch.Tensor)]
    if not tensor_args:
        return

    input_tensors = tensor_args[:-1]
    output_tensor = tensor_args[-1]

    # Convert PyTorch tensors to JAX arrays
    input_jax = [jnp.array(t.detach().numpy()) for t in input_tensors]
    out_dtype = jnp.dtype(np.dtype(str(output_tensor.dtype).replace("torch.", "")))
    out_shape = jax.ShapeDtypeStruct(output_tensor.shape, out_dtype)

    # Infer block shape from output shape and grid
    # e.g. output (1024,) with grid=(4,) -> block_shape=(256,)
    block_shape = tuple(s // g for s, g in zip(output_tensor.shape, grid, strict=True))
    in_specs = [pl.BlockSpec(block_shape, lambda *idx: idx) for _ in input_jax]
    out_specs = pl.BlockSpec(block_shape, lambda *idx: idx)

    result = pl.pallas_call(
        pallas_kernel,  # pyrefly: ignore[bad-argument-type]
        out_shape=out_shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        interpret=True,
    )(*input_jax)

    # Copy result back to PyTorch output tensor
    output_tensor.copy_(torch.from_numpy(np.asarray(result)))
