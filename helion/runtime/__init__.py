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


def default_pallas_launcher(
    pallas_kernel: object,  # Callable taking Ref args
    grid: tuple[int, ...],
    *args: object,
    **kwargs: object,
) -> None:
    """Default launcher for Pallas kernels using pallas_call with interpret=True.

    Args:
        pallas_kernel: The Pallas kernel function (takes Ref args).
        grid: Grid dimensions for the kernel launch.
        *args: Positional args — tensors and constexpr values.
              Tensor args come first, int args (block sizes) come last.
              The last tensor arg is treated as the output.
    """
    import jax
    from jax.experimental import pallas as pl
    import jax.numpy as jnp
    import numpy as np

    # Separate tensor args from constexpr (int) args
    tensor_args: list[torch.Tensor] = []
    block_sizes: list[int] = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            tensor_args.append(arg)
        elif isinstance(arg, int):
            block_sizes.append(arg)

    if not tensor_args:
        return

    input_tensors = tensor_args[:-1]
    output_tensor = tensor_args[-1]

    # Convert PyTorch tensors to JAX arrays
    input_jax = [jnp.array(t.detach().numpy()) for t in input_tensors]
    out_dtype = jnp.dtype(np.dtype(str(output_tensor.dtype).replace("torch.", "")))

    if len(block_sizes) == 1:
        # Simple 1D case: use BlockSpec for clean tiling
        block_shape = (block_sizes[0],)
        in_specs = [pl.BlockSpec(block_shape, lambda i: (i,)) for _ in input_jax]
        out_specs = pl.BlockSpec(block_shape, lambda i: (i,))
        out_shape = jax.ShapeDtypeStruct(output_tensor.shape, out_dtype)

        result = pl.pallas_call(
            pallas_kernel,  # pyrefly: ignore[bad-argument-type]
            out_shape=out_shape,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            interpret=True,
        )(*input_jax)
    else:
        # Multi-dimensional / matmul case: pass full arrays (no BlockSpec).
        # The kernel has internal loops for grid traversal and reduction.
        # We run with grid=(1,) and let the kernel handle everything.
        n_inputs = len(input_jax)
        out_shape = jax.ShapeDtypeStruct(output_tensor.shape, out_dtype)

        def wrapper_kernel(*refs: object) -> None:  # type: ignore[type-arg]
            # The kernel expects tensor refs + block size ints
            pallas_kernel(*refs, *block_sizes)  # type: ignore[operator]

        result = pl.pallas_call(
            wrapper_kernel,  # pyrefly: ignore[bad-argument-type]
            out_shape=out_shape,
            grid=(1,),
            in_specs=[pl.BlockSpec(None, None)] * n_inputs,
            out_specs=pl.BlockSpec(None, None),
            interpret=True,
        )(*input_jax)

    # Copy result back to PyTorch output tensor
    output_tensor.copy_(torch.from_numpy(np.asarray(result)))
