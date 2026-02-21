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
    _output_indices: list[int] | None = None,
    **kwargs: object,
) -> None:
    """Default launcher for Pallas kernels on TPU.

    Uses ``JaxCallable`` from ``torch_tpu`` to compile and run the Pallas
    kernel on TPU.  Output tensors are donated via ``input_output_aliases``
    so the kernel writes directly into their buffers (zero-copy).
    """
    import jax
    from jax.experimental import pallas as pl
    from torch_tpu._internal.pallas.pallas import (  # pyrefly: ignore[missing-import]
        JaxCallable,
    )
    from torch_tpu._internal.pallas.pallas import (  # pyrefly: ignore[missing-import]
        jax_placeholder,
    )

    if _output_indices is None:
        _output_indices = []

    output_set = set(_output_indices)
    input_indices = list(range(len(args)))
    n_inputs = len(input_indices)

    inputs = [args[i] for i in input_indices]
    outputs = [args[i] for i in _output_indices]

    # Detect inplace args: positions that appear in both inputs and outputs.
    # For these, the output ref starts uninitialized on TPU, so we must copy
    # the input ref's data into the output ref before the kernel runs.
    inplace_positions = output_set & set(input_indices)

    out_shapes = tuple(jax_placeholder(out) for out in outputs)  # type: ignore[arg-type]

    # Cache JaxCallable on the kernel function object to avoid global state
    cache = getattr(pallas_kernel, "_pallas_cache", None)
    if cache is None or cache[0] != grid:
        # Create a reordering wrapper so pallas_call's (inputs..., outputs...)
        # ref ordering maps back to the original Helion kernel parameter order.
        # For output args, use the output ref (not the input ref) so that
        # writes go to the output buffer.
        def reordered_kernel(*refs: object) -> None:
            n_kernel_params = len(args)
            original_order: list[object] = [None] * n_kernel_params
            for new_pos, orig_pos in enumerate(input_indices):
                original_order[orig_pos] = refs[new_pos]
            # Output refs override input refs at the same position
            for new_pos, orig_pos in enumerate(_output_indices):
                out_ref = refs[n_inputs + new_pos]
                if orig_pos in inplace_positions:
                    # Inplace: copy input data into the output ref so the
                    # kernel can read the original values.
                    in_ref = refs[input_indices.index(orig_pos)]
                    out_ref[...] = in_ref[...]  # type: ignore[index]
                original_order[orig_pos] = out_ref
            pallas_kernel(*original_order)  # type: ignore[operator]

        out_shape_arg = out_shapes if len(out_shapes) > 1 else out_shapes[0]
        kernel_name = getattr(pallas_kernel, "__name__", "pallas_kernel")

        # Map each output to its corresponding input for XLA buffer reuse
        pallas_aliases = {
            input_indices.index(orig_pos): out_idx
            for out_idx, orig_pos in enumerate(_output_indices)
        }

        jit_fn = jax.jit(
            pl.pallas_call(
                reordered_kernel,
                out_shape=out_shape_arg,
                input_output_aliases=pallas_aliases,
                grid=grid,
            ),
        )

        # Build call_custom_kernel aliases: map input tensor positions to
        # output positions so torch_tpu donates the buffer (zero-copy).
        call_aliases: dict[int, int] = {}
        for out_idx, orig_pos in enumerate(_output_indices):
            input_pos = input_indices.index(orig_pos)
            tensor_pos = sum(
                1
                for i in input_indices[:input_pos]
                if isinstance(args[i], torch.Tensor)
            )
            call_aliases[tensor_pos] = out_idx

        jax_callable = JaxCallable(
            name=kernel_name,
            jit_fn=jit_fn,
            trace_key=f"{kernel_name}_{id(pallas_kernel)}_{grid}",
            input_output_aliases=call_aliases,
        )
        pallas_kernel._pallas_cache = (grid, jax_callable)  # type: ignore[union-attr]
    else:
        _, jax_callable = cache

    # Call with all input tensors (including outputs for donation)
    input_tensors = [inp for inp in inputs if isinstance(inp, torch.Tensor)]
    jax_callable(*input_tensors)


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

    params.extend(
        (
            "grid_x: cutlass.Int32",
            "grid_y: cutlass.Int32",
            "grid_z: cutlass.Int32",
            "block_x: cutlass.Int32",
            "block_y: cutlass.Int32",
            "block_z: cutlass.Int32",
        )
    )
    body.extend(
        (
            "    _kernel("
            + ", ".join(call_args)
            + ").launch(grid=(grid_x, grid_y, grid_z), block=(block_x, block_y, block_z))",
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
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
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

    launch_args.extend((*grid, *block))
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
    if not isinstance(grid, tuple) or len(grid) < 1:
        raise ValueError(f"Invalid grid specification: {grid}")
    if kwargs:
        raise exc.BackendUnsupported("cute", f"launcher kwargs: {sorted(kwargs)}")

    grid_xyz = (
        int(grid[0]),
        int(grid[1]) if len(grid) > 1 else 1,
        int(grid[2]) if len(grid) > 2 else 1,
    )
    block_xyz = (
        int(block[0]),
        int(block[1]) if len(block) > 1 else 1,
        int(block[2]) if len(block) > 2 else 1,
    )

    schema_key, launch_args = _build_cute_schema_and_args(
        tuple(args), grid_xyz, block_xyz
    )
    compiled = _get_compiled_cute_launcher(cute_kernel, schema_key, launch_args)
    return cast("Any", compiled)(*launch_args)
