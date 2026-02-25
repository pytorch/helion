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
    ptx_options: str | None = None,
    launch_cooperative_grid: bool = False,
    **kwargs: dict,
) -> object:
    """Default launcher function that executes the kernel immediately."""
    # For both CUDA and MTIA, use the same kernel execution
    run_kwargs: dict = {
        "grid": grid,
        "warmup": False,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "launch_cooperative_grid": launch_cooperative_grid,
        **kwargs,
    }
    if ptx_options is not None:
        run_kwargs["ptx_options"] = ptx_options
    return triton_kernel.run(  # type: ignore[union-attr]
        *args,
        **run_kwargs,
    )


_TPU_ALIGN_LAST = 128
_TPU_ALIGN_SECOND_LAST = 8


def _pallas_build_block_specs(
    pl: object,
    jnp: object,
    grid: tuple[int, ...],
    args: tuple[object, ...],
    tensor_arg_indices: list[int],
    output_indices: list[int],
) -> tuple[list[object] | None, object | None]:
    """Build ``in_specs`` and ``out_specs`` for ``pl.pallas_call``.

    Infers block shapes from tensor shapes and the Helion grid.  For each
    output dimension, if ``dim_size == grid[g] * block`` for some grid
    dimension *g*, that axis is tiled with the block size.

    TPU alignment constraints (last dim multiple of 128, second-to-last
    multiple of 8, exact division required) are checked; falls back to
    ``(None, None)`` when Helion's grid is incompatible with TPU tiling.
    """
    import jax.numpy as jnp  # pyrefly: ignore[import-error]

    output_set = set(output_indices)

    # Find the largest tensor (input or output) as the reference shape.
    ref_shape: tuple[int, ...] | None = None
    for idx in tensor_arg_indices:
        t = args[idx]
        if isinstance(t, torch.Tensor) and t.ndim > 0:
            if ref_shape is None or len(t.shape) > len(ref_shape):
                ref_shape = tuple(t.shape)

    if ref_shape is None or len(grid) == 0:
        return None, None

    # For each grid dim, find which ref axis it tiles.
    ref_block = list(ref_shape)
    axis_to_grid: dict[int, int] = {}
    used_grid_dims: set[int] = set()

    for g_dim in range(len(grid)):
        if grid[g_dim] <= 1:
            continue
        for ax in range(len(ref_shape)):
            if ax in axis_to_grid:
                continue
            block = ref_shape[ax] // grid[g_dim]
            if block <= 0 or ref_shape[ax] % grid[g_dim] != 0:
                continue
            # Check TPU alignment for this axis.
            # TPU Mosaic XLA layout uses the full dimension as the tiling
            # factor for the last (innermost) dimension, so it cannot be
            # sub-tiled.  Only non-last dimensions can be tiled.
            n_dims = len(ref_shape)
            if ax == n_dims - 1:
                # Last dim cannot be tiled on TPU.
                continue
            if ax == n_dims - 2:
                align = _TPU_ALIGN_SECOND_LAST
            else:
                align = 1
            if block < align or block % align != 0:
                continue
            ref_block[ax] = block
            axis_to_grid[ax] = g_dim
            used_grid_dims.add(g_dim)
            break

    # If we couldn't map all non-trivial grid dims, fall back.
    for g_dim in range(len(grid)):
        if grid[g_dim] > 1 and g_dim not in used_grid_dims:
            return None, None

    ref_block_tuple = tuple(ref_block)

    def _make_spec(tensor: torch.Tensor, is_output: bool) -> object:
        shape = tuple(tensor.shape)
        buf_dims = len(shape)
        ref_dims = len(ref_shape)  # type: ignore[arg-type]
        offset = ref_dims - buf_dims

        block = list(shape)
        tiled_pairs: list[tuple[int, int]] = []

        for ref_ax, g_dim in axis_to_grid.items():
            buf_ax = ref_ax - offset
            if 0 <= buf_ax < buf_dims and shape[buf_ax] == ref_shape[ref_ax]:  # type: ignore[index]
                block[buf_ax] = ref_block_tuple[ref_ax]
                tiled_pairs.append((buf_ax, g_dim))
            elif (
                is_output
                and buf_dims < ref_dims
                and 0 <= ref_ax < buf_dims
                and shape[ref_ax] == ref_shape[ref_ax]  # type: ignore[index]
            ):
                # Left-aligned match for reduction outputs
                block[ref_ax] = ref_block_tuple[ref_ax]
                tiled_pairs.append((ref_ax, g_dim))

        mapping = dict(tiled_pairs)

        def index_map(
            *grid_args: object,
            _m: dict[int, int] = mapping,
            _nd: int = buf_dims,
        ) -> tuple[object, ...]:
            return tuple(
                jnp.int32(grid_args[_m[d]]) if d in _m else jnp.int32(0)
                for d in range(_nd)
            )

        return pl.BlockSpec(tuple(block), index_map)  # type: ignore[union-attr]

    in_specs = []
    for idx in tensor_arg_indices:
        t = args[idx]
        assert isinstance(t, torch.Tensor)
        in_specs.append(_make_spec(t, is_output=(idx in output_set)))

    out_specs_list = []
    for idx in output_indices:
        t = args[idx]
        assert isinstance(t, torch.Tensor)
        out_specs_list.append(_make_spec(t, is_output=True))

    # Validate: check that all block shapes satisfy TPU alignment.
    # For 1D tensors, the block must be the full tensor or a multiple of 128.
    all_specs = in_specs + out_specs_list
    for idx, spec in zip(
        [*tensor_arg_indices, *output_indices], all_specs, strict=True
    ):
        t = args[idx]
        assert isinstance(t, torch.Tensor)
        block_shape = spec.block_shape  # type: ignore[union-attr]
        for d in range(len(block_shape)):
            dim_size = t.shape[d]
            bs = block_shape[d]
            if bs == dim_size:
                continue  # Full dim — always OK
            # Last dim must be a multiple of 128
            if d == len(block_shape) - 1 and bs % _TPU_ALIGN_LAST != 0:
                return None, None
            # Second-to-last must be a multiple of 8
            if d == len(block_shape) - 2 and bs % _TPU_ALIGN_SECOND_LAST != 0:
                return None, None
            # Must divide evenly
            if dim_size % bs != 0:
                return None, None

    out_specs = out_specs_list if len(out_specs_list) > 1 else out_specs_list[0]
    return in_specs, out_specs


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

    # Separate tensor args (which become Pallas refs) from non-tensor args
    # (scalars/ints which are passed to the kernel via closure).
    tensor_arg_indices = [
        i for i in range(len(args)) if isinstance(args[i], torch.Tensor)
    ]
    non_tensor_args: dict[int, object] = {
        i: args[i] for i in range(len(args)) if not isinstance(args[i], torch.Tensor)
    }
    n_tensor_inputs = len(tensor_arg_indices)
    # Map original arg position -> position within tensor-only refs
    arg_to_tensor_pos = {orig: tpos for tpos, orig in enumerate(tensor_arg_indices)}

    outputs = [args[i] for i in _output_indices]

    # Detect inplace args: positions that appear in both inputs and outputs.
    # For these, the output ref starts uninitialized on TPU, so we must copy
    # the input ref's data into the output ref before the kernel runs.
    inplace_positions = output_set & set(tensor_arg_indices)

    out_shapes = tuple(jax_placeholder(out) for out in outputs)  # type: ignore[arg-type]

    # Cache JaxCallable on the kernel function object to avoid global state
    cache = getattr(pallas_kernel, "_pallas_cache", None)
    if cache is None or cache[0] != grid:
        import jax.numpy as jnp

        # Build BlockSpecs so each grid cell sees the correctly-tiled slice.
        # The grid comes from Helion's tile strategy; we infer block shapes
        # by dividing tensor shapes by the grid extents.
        in_specs, out_specs = _pallas_build_block_specs(
            pl, jnp, grid, args, tensor_arg_indices, _output_indices
        )

        # Create a reordering wrapper so pallas_call's (inputs..., outputs...)
        # ref ordering maps back to the original Helion kernel parameter order.
        # pallas_call gives us refs for tensor inputs followed by output refs.
        # Non-tensor args (scalars) are captured from the closure.
        def reordered_kernel(*refs: object) -> None:
            n_kernel_params = len(args)
            original_order: list[object] = [None] * n_kernel_params
            # Assign tensor input refs
            for tensor_pos, orig_pos in enumerate(tensor_arg_indices):
                original_order[orig_pos] = refs[tensor_pos]
            # Assign non-tensor args directly from closure
            for orig_pos, value in non_tensor_args.items():
                original_order[orig_pos] = value
            # Output refs override input refs at the same position
            for out_idx, orig_pos in enumerate(_output_indices):
                out_ref = refs[n_tensor_inputs + out_idx]
                if orig_pos in inplace_positions:
                    # Inplace: copy input data into the output ref so the
                    # kernel can read the original values.
                    in_ref = refs[arg_to_tensor_pos[orig_pos]]
                    out_ref[...] = in_ref[...]  # type: ignore[index]
                original_order[orig_pos] = out_ref
            pallas_kernel(*original_order)  # type: ignore[operator]

        out_shape_arg = out_shapes if len(out_shapes) > 1 else out_shapes[0]
        kernel_name = getattr(pallas_kernel, "__name__", "pallas_kernel")

        # Map each tensor input to its output for XLA buffer reuse
        pallas_aliases = {
            arg_to_tensor_pos[orig_pos]: out_idx
            for out_idx, orig_pos in enumerate(_output_indices)
        }

        pallas_call_kwargs: dict[str, object] = {
            "out_shape": out_shape_arg,
            "input_output_aliases": pallas_aliases,
            "grid": grid,
        }
        if in_specs is not None:
            pallas_call_kwargs["in_specs"] = in_specs
            pallas_call_kwargs["out_specs"] = out_specs

        jit_fn = jax.jit(
            pl.pallas_call(
                reordered_kernel,
                **pallas_call_kwargs,  # type: ignore[arg-type]
            ),
        )

        # Build call_custom_kernel aliases: map input tensor positions to
        # output positions so torch_tpu donates the buffer (zero-copy).
        call_aliases: dict[int, int] = {}
        for out_idx, orig_pos in enumerate(_output_indices):
            call_aliases[arg_to_tensor_pos[orig_pos]] = out_idx

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
    input_tensors = [
        args[i] for i in tensor_arg_indices if isinstance(args[i], torch.Tensor)
    ]
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
