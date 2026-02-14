from __future__ import annotations

import contextvars
import os

import torch
import triton

from .. import _compat as _compat  # ensure Triton compatibility patches run
from .config import Config as Config
from .kernel import Kernel as Kernel
from .kernel import kernel as kernel
from .triton_helpers import triton_send_signal as triton_send_signal
from .triton_helpers import triton_wait_multiple_signal as triton_wait_multiple_signal
from .triton_helpers import triton_wait_signal as triton_wait_signal


def _alloc_fn(size: int, alignment: int, stream: int | None) -> torch.Tensor:
    # Dynamically get device from Triton backend
    current_target = triton.runtime.driver.active.get_current_target()
    if current_target is None:
        raise RuntimeError("No active Triton target available")
    backend = current_target.backend
    return torch.empty(size, device=backend, dtype=torch.int8)


def set_triton_allocator() -> None:
    try:
        from triton import set_allocator
        from triton.runtime._allocation import NullAllocator
        from triton.runtime._allocation import _allocator
    except ImportError:
        return
    if isinstance(_allocator, contextvars.ContextVar):
        existing = _allocator.get()
    else:  # older versions of Triton
        existing = _allocator
    # if allocator isn't NullAllocator, we assume it is set by the user
    if isinstance(existing, NullAllocator):
        set_allocator(_alloc_fn)


def get_num_sm(device: torch.device, *, reserved_sms: int = 0) -> int:
    """
    Get the number of streaming multiprocessors (SMs) for the specified device.

    Args:
        device: Device to query.
        reserved_sms: Number of SMs to keep free for other work (e.g., communication
            kernels). Defaults to 0 meaning all device SMs are available to Helion.

    Returns:
        Grid size to use for a persistent kernel on the device after accounting
        for any reserved SMs. Always at least 1.
    """
    available_sms: int
    assert device.type in [
        "cuda",
        "xpu",
        "cpu",
        "mtia",
    ], "TODO: implement for other devices"
    if device.type == "cpu":
        try:
            num_threads = int(torch.get_num_threads())
        except Exception:
            num_threads = 0
        available_sms = num_threads if num_threads > 0 else int(os.cpu_count() or 1)
    elif device.type == "cuda":
        available_sms = torch.cuda.get_device_properties(
            device.index
        ).multi_processor_count
    # TODO(EikanWang): gpu_subslice_count is an out-of-date term. we change update it to XeCore number.
    elif device.type == "xpu":
        available_sms = torch.xpu.get_device_properties(device.index).gpu_subslice_count
    elif device.type == "mtia":
        device_props = torch.mtia.get_device_properties(device.index)
        if "max_grid_height" in device_props and "max_grid_width" in device_props:
            available_sms = (
                device_props["max_grid_height"] * device_props["max_grid_width"]
            )
        else:
            raise RuntimeError(
                f"Unable to determine SM count for MTIA device. "
                f"Available properties: {list(device_props.keys())}"
            )
    else:
        raise NotImplementedError(
            f"get_num_sm not implemented for device type: {device.type}"
        )

    if reserved_sms <= 0:
        return available_sms
    return max(available_sms - reserved_sms, 1)


def default_launcher(
    triton_kernel: triton.JITFunction,
    grid: tuple[int, ...],
    *args: object,
    num_warps: int,
    num_stages: int,
    launch_cooperative_grid: bool = False,
    **kwargs: dict,
) -> object:
    """Default launcher function that executes the kernel immediately."""
    # For both CUDA and MTIA, use the same kernel execution
    return triton_kernel.run(
        *args,
        grid=grid,
        warmup=False,
        num_warps=num_warps,
        num_stages=num_stages,
        launch_cooperative_grid=launch_cooperative_grid,
        **kwargs,
    )


def _helion_cdiv(x: int, y: int) -> int:
    """Ceiling division: ``ceil(x / y)`` using integer arithmetic."""
    return (x + y - 1) // y


def default_pallas_launcher(
    pallas_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    **kwargs: object,
) -> None:
    """Default launcher for Pallas kernels using pallas_call with interpret=True."""
    import functools

    import jax  # pyrefly: ignore[import-error, missing-import]
    from jax.experimental import (  # pyrefly: ignore[import-error, missing-import]
        pallas as pl,
    )
    import numpy as np

    tensor_args = [a for a in args if isinstance(a, torch.Tensor)]
    if not tensor_args:
        return

    input_tensors = tensor_args[:-1]
    output_tensor = tensor_args[-1]

    # Zero-copy: numpy() returns a view of CPU tensor memory
    input_jax = [jax.numpy.asarray(t.detach().numpy()) for t in input_tensors]
    output_jax = jax.numpy.asarray(output_tensor.detach().numpy())

    out_shape = jax.ShapeDtypeStruct(output_tensor.shape, output_jax.dtype)

    # Infer block shape from output shape and grid
    # e.g. output (1024,) with grid=(4,) -> block_shape=(256,)
    block_shape = tuple(s // g for s, g in zip(output_tensor.shape, grid, strict=True))

    n_inputs = len(input_jax)
    # n_inputs + 1 specs: regular inputs + donated output buffer
    in_specs = [
        pl.BlockSpec(block_shape, lambda *idx: idx) for _ in range(n_inputs + 1)
    ]
    out_specs = pl.BlockSpec(block_shape, lambda *idx: idx)

    # Wrap kernel to skip the donated-output ref that input_output_aliases
    # inserts between the regular input refs and the output ref.
    def _kernel(*refs: object) -> None:
        pallas_kernel(*refs[:n_inputs], refs[-1])  # pyrefly: ignore[not-callable]

    # Wrap in jax.jit with donate_argnums so JAX reuses the output buffer
    # in-place via input_output_aliases, avoiding a copy back.
    @functools.partial(jax.jit, donate_argnums=(n_inputs,))
    def _jit_call(*jax_args: object) -> object:
        return pl.pallas_call(
            _kernel,
            out_shape=out_shape,
            in_specs=in_specs,
            out_specs=out_specs,
            input_output_aliases={n_inputs: 0},
            interpret=True,
            grid=grid,
        )(*jax_args)

    result = _jit_call(*input_jax, output_jax)

    # With donate_argnums + input_output_aliases the result may already
    # reside in the output tensor's memory.  Copy back only if needed.
    result_np = np.asarray(result)
    output_np = output_tensor.detach().numpy()
    if not np.shares_memory(result_np, output_np):
        np.copyto(output_np, result_np)
