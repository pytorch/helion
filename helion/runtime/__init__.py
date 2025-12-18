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
        **kwargs,
    )


def pallas_launcher(
    kernel_fn: object,
    grid: tuple[int, ...],
    *args: object,
    **kwargs: dict,
) -> object:
    """Pallas launcher function that wraps the kernel with pallas_call.

    Pallas supports both TPU and GPU backends. The device is automatically
    selected based on JAX's default device (TPU preferred if available,
    otherwise GPU).

    Args:
        kernel_fn: The Pallas kernel function (takes refs as params)
        grid: The grid size for parallel execution
        *args: Arguments in order: input tensors, output tensor, then scalars (block_size, etc.)
    """
    # Import JAX/Pallas lazily to avoid import errors when not available
    try:
        import jax
        import jax.numpy as jnp
        from jax.experimental import pallas as pl
    except ImportError as e:
        raise ImportError(
            "Pallas backend requires JAX to be installed. "
            "Install with: pip install jax jaxlib"
        ) from e

    # Separate tensors from scalars
    tensor_args = []
    scalar_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            tensor_args.append(arg)
        else:
            scalar_args.append(arg)

    # Get block size (first scalar arg)
    block_size = scalar_args[0] if scalar_args else 128

    # Convert PyTorch tensors to JAX arrays
    def to_jax(x: torch.Tensor) -> jnp.ndarray:
        return jnp.array(x.detach().cpu().numpy())

    jax_tensors = [to_jax(t) for t in tensor_args]

    # The last tensor is the output
    # Inputs are all tensors except the last one
    input_tensors = jax_tensors[:-1]
    output_tensor = jax_tensors[-1]

    # Create BlockSpec for each tensor
    # For 1D tensors, use (block_size,) shape with lambda to compute offset
    def make_block_spec(tensor: jnp.ndarray) -> pl.BlockSpec:
        if tensor.ndim == 1:
            return pl.BlockSpec((block_size,), lambda i: (i * block_size,))
        else:
            # For multi-dimensional tensors, only tile the first dimension for now
            block_shape = (block_size,) + tensor.shape[1:]
            return pl.BlockSpec(block_shape, lambda i: (i * block_size,) + (0,) * (tensor.ndim - 1))

    in_specs = [make_block_spec(t) for t in input_tensors]
    out_specs = make_block_spec(output_tensor)

    # With BlockSpec, the kernel receives pre-sliced blocks and doesn't need block size params
    # The scalar args (block sizes) are only used for BlockSpec creation
    # Call pallas_call to create the kernel
    pallas_kernel = pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct(output_tensor.shape, output_tensor.dtype),
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid,
    )

    # Execute the kernel
    result = pallas_kernel(*input_tensors)

    # Convert result back to PyTorch tensor
    # Use np.array with copy=True to ensure writable array
    import numpy as np
    result_np = np.array(result, copy=True)
    result_torch = torch.from_numpy(result_np)

    # Copy result back to the output tensor
    tensor_args[-1].copy_(result_torch)

    return tensor_args[-1]
