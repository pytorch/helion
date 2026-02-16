from __future__ import annotations

import contextvars
import linecache
import os
from typing import Any
from typing import cast

import torch

from .. import _compat as _compat  # ensure Triton compatibility patches run
from .. import exc
from .._utils import triton_is_available
from .config import Config as Config
from .kernel import Kernel as Kernel
from .kernel import kernel as kernel

if triton_is_available():
    import triton

    from .triton_helpers import triton_send_signal as triton_send_signal
    from .triton_helpers import (
        triton_wait_multiple_signal as triton_wait_multiple_signal,
    )
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
else:

    def set_triton_allocator() -> None:  # type: ignore[misc]
        pass


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
    triton_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    num_warps: int,
    num_stages: int,
    launch_cooperative_grid: bool = False,
    **kwargs: dict,
) -> object:
    """Default launcher function that executes the kernel immediately."""
    # For both CUDA and MTIA, use the same kernel execution
    return triton_kernel.run(  # type: ignore[union-attr]
        *args,
        grid=grid,
        warmup=False,
        num_warps=num_warps,
        num_stages=num_stages,
        launch_cooperative_grid=launch_cooperative_grid,
        **kwargs,
    )


def default_pallas_launcher(
    pallas_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    **kwargs: object,
) -> None:
    """Default launcher for Pallas kernels using the Mosaic GPU backend.

    Uses ``plgpu.as_torch_kernel`` to call the kernel directly on PyTorch
    tensors without DLPack conversion, copies, or explicit synchronization.
    All tensor arguments (inputs and outputs) are passed as refs; the kernel
    writes to output refs via side effects.
    """
    try:
        torch_kernel = pallas_kernel._torch_kernel  # type: ignore[union-attr]
    except AttributeError:
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import mosaic_gpu as plgpu

        # Wrap kernel with pl.kernel using out_shape=() (side-effect only:
        # the kernel mutates output refs in-place, no new outputs are created).
        wrapped = pl.kernel(
            pallas_kernel,  # pyrefly: ignore[bad-argument-type]
            out_shape=(),
            mesh=plgpu.Mesh(),  # pyrefly: ignore[bad-argument-type]
            compiler_params=plgpu.CompilerParams(),  # pyrefly: ignore[bad-instantiation]
        )
        torch_kernel = pallas_kernel._torch_kernel = plgpu.as_torch_kernel(wrapped)  # type: ignore[union-attr]
    return torch_kernel(*args, **kwargs)


def _torch_dtype_to_cutlass(dtype: torch.dtype) -> object:
    import cutlass

    mapping: dict[torch.dtype, object] = {
        torch.float16: cutlass.Float16,
        torch.float32: cutlass.Float32,
        torch.float64: cutlass.Float64,
        torch.bfloat16: cutlass.BFloat16,
        torch.int8: cutlass.Int8,
        torch.int16: cutlass.Int16,
        torch.int32: cutlass.Int32,
        torch.int64: cutlass.Int64,
        torch.uint8: cutlass.Uint8,
    }
    if dtype not in mapping:
        raise exc.BackendUnsupported("cute", f"dtype: {dtype}")
    return mapping[dtype]


def _normalize_cute_scalar(arg: object) -> tuple[str, object]:
    if isinstance(arg, (bool, torch.SymBool)):
        return ("bool", bool(arg))
    if isinstance(arg, (int, torch.SymInt)):
        return ("int", int(arg))
    if isinstance(arg, (float, torch.SymFloat)):
        return ("float", float(arg))
    raise exc.BackendUnsupported("cute", f"launcher scalar argument type: {type(arg)}")


def _cute_scalar_annotation(kind: str) -> str:
    mapping = {
        "bool": "cutlass.Boolean",
        "int": "cutlass.Int64",
        "float": "cutlass.Float64",
    }
    return mapping[kind]


def _create_cute_wrapper(
    cute_kernel: object,
    schema_key: tuple[tuple[object, ...], ...],
) -> object:
    import cutlass
    import cutlass.cute as cute

    func_name = "_helion_cute_launch"
    params: list[str] = []
    body: list[str] = []
    call_args: list[str] = []

    for i, entry in enumerate(schema_key):
        kind = entry[0]
        if kind == "tensor":
            (_, _dtype, rank) = entry
            assert isinstance(rank, int)
            ptr_name = f"arg{i}_ptr"
            params.append(f"{ptr_name}: cute.Pointer")
            shape_names = [f"arg{i}_shape{d}" for d in range(rank)]
            stride_names = [f"arg{i}_stride{d}" for d in range(rank)]
            params.extend(f"{name}: cutlass.Int64" for name in shape_names)
            params.extend(f"{name}: cutlass.Int64" for name in stride_names)
            shape_tuple = (
                f"({shape_names[0]},)" if rank == 1 else f"({', '.join(shape_names)})"
            )
            stride_tuple = (
                f"({stride_names[0]},)" if rank == 1 else f"({', '.join(stride_names)})"
            )
            body.append(
                f"    arg{i} = cute.make_tensor({ptr_name}, layout=cute.make_layout({shape_tuple}, stride={stride_tuple}))"
            )
            call_args.append(f"arg{i}")
            continue

        assert kind == "scalar"
        (_, scalar_kind) = entry
        assert isinstance(scalar_kind, str)
        scalar_name = f"arg{i}"
        params.append(f"{scalar_name}: {_cute_scalar_annotation(scalar_kind)}")
        call_args.append(scalar_name)

    params.extend(("grid_x: cutlass.Int32", "block_x: cutlass.Int32"))
    body.extend(
        (
            "    _kernel("
            + ", ".join(call_args)
            + ").launch(grid=(grid_x, 1, 1), block=(block_x, 1, 1))",
        )
    )

    source = "\n".join(
        [
            "@cute.jit",
            f"def {func_name}({', '.join(params)}) -> None:",
            *body,
        ]
    )

    namespace: dict[str, Any] = {
        "cutlass": cutlass,
        "cute": cute,
        "_kernel": cute_kernel,
    }
    filename = (
        f"<helion_cute_launcher:{cast('Any', cute_kernel).__name__}:{schema_key!r}>"
    )
    linecache.cache[filename] = (
        len(source),
        None,
        [line + "\n" for line in source.splitlines()],
        filename,
    )
    exec(compile(source, filename, "exec"), namespace)
    return namespace[func_name]


def _get_compiled_cute_launcher(
    cute_kernel: object,
    schema_key: tuple[tuple[object, ...], ...],
    launch_args: tuple[object, ...],
) -> object:
    import cutlass.cute as cute

    try:
        # pyrefly: ignore [missing-attribute]
        return cute_kernel._helion_cute_compiled_launcher
    except AttributeError:
        pass

    wrapper = _create_cute_wrapper(cute_kernel, schema_key)
    compiled = cute.compile(wrapper, *launch_args)
    # pyrefly: ignore [missing-attribute]
    cute_kernel._helion_cute_compiled_launcher = compiled
    return compiled


def _build_cute_schema_and_args(
    args: tuple[object, ...],
    grid_x: int,
    block_x: int,
) -> tuple[tuple[tuple[object, ...], ...], tuple[object, ...]]:
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    schema: list[tuple[object, ...]] = []
    launch_args: list[object] = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            if arg.device.type != "cuda":
                raise exc.BackendUnsupported("cute", "launcher requires CUDA tensors")
            if arg.ndim <= 0:
                raise exc.BackendUnsupported(
                    "cute", "launcher requires tensor rank >= 1"
                )
            schema.append(("tensor", str(arg.dtype), arg.ndim))
            launch_args.append(
                make_ptr(
                    cast("Any", _torch_dtype_to_cutlass(arg.dtype)),
                    arg.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
            )
            launch_args.extend(int(arg.size(d)) for d in range(arg.ndim))
            launch_args.extend(int(arg.stride(d)) for d in range(arg.ndim))
            continue

        scalar_kind, scalar_value = _normalize_cute_scalar(arg)
        schema.append(("scalar", scalar_kind))
        launch_args.append(scalar_value)

    launch_args.extend((grid_x, block_x))
    return tuple(schema), tuple(launch_args)


def default_cute_launcher(
    cute_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    **kwargs: object,
) -> object:
    block = kwargs.pop("block", (256, 1, 1))
    if not isinstance(block, tuple) or len(block) < 1:
        raise ValueError(f"Invalid block specification: {block}")
    if kwargs:
        raise exc.BackendUnsupported("cute", f"launcher kwargs: {sorted(kwargs)}")

    block_x = int(block[0])
    grid_x = int(grid[0])
    schema_key, launch_args = _build_cute_schema_and_args(tuple(args), grid_x, block_x)
    compiled = _get_compiled_cute_launcher(cute_kernel, schema_key, launch_args)
    return cast("Any", compiled)(*launch_args)
