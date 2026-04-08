from __future__ import annotations

from contextlib import suppress
import contextvars
import hashlib
import linecache
import sys
from typing import Any
from typing import cast

import torch

from .. import _compat as _compat  # ensure Triton compatibility patches run
from .. import exc
from .._utils import triton_is_available
from .config import Config as Config
from .kernel import Kernel as Kernel
from .kernel import kernel as kernel

_CUTLASS_SHUTDOWN_PATCHED = False


def _patch_cutlass_jit_shutdown_unload() -> None:
    """Avoid CUDA library unload hangs during interpreter shutdown.

    On current CUTLASS DSL builds, ``CudaDialectJitModule.__del__`` unconditionally
    calls ``cudaLibraryUnload``. On B200 this can hang during Python finalization
    after a CuTe kernel has already finished executing. Skipping that unload during
    interpreter teardown lets the process exit cleanly while preserving the normal
    unload path during regular runtime GC.
    """

    global _CUTLASS_SHUTDOWN_PATCHED
    if _CUTLASS_SHUTDOWN_PATCHED:
        return

    try:
        import cutlass.cutlass_dsl.cuda_jit_executor as cuda_jit_executor
    except ImportError:
        return

    module_type = cuda_jit_executor.CudaDialectJitModule
    if getattr(module_type, "_helion_shutdown_patch", False):
        _CUTLASS_SHUTDOWN_PATCHED = True
        return

    original_del = cast("Any", module_type.__del__)

    def _helion_del(self: object) -> None:
        module = cast("Any", self)
        if sys.is_finalizing():
            with suppress(Exception):
                module._unloaded = True
            return
        original_del(module)

    module_type.__del__ = _helion_del
    module_type._helion_shutdown_patch = True
    _CUTLASS_SHUTDOWN_PATCHED = True


if triton_is_available():
    import triton

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
        "mtia",
        "mps",
    ], "TODO: implement for other devices"
    if device.type == "cuda":
        available_sms = torch.cuda.get_device_properties(
            device.index
        ).multi_processor_count
    # TODO(EikanWang): gpu_subslice_count is an out-of-date term. we change update it to XeCore number.
    elif device.type == "xpu":
        available_sms = torch.xpu.get_device_properties(device.index).gpu_subslice_count
    elif device.type == "mps":
        available_sms = torch.backends.mps.get_core_count()
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
    try:
        return triton_kernel.run(  # type: ignore[union-attr]
            *args,
            **run_kwargs,
        )
    except Exception as error:
        message = str(error)
        if "Cannot make_shape_compatible: incompatible dimensions" in message:
            raise exc.ShapeMismatch("kernel operands", message) from error
        raise


def _pallas_make_block_spec(
    pl: object,
    jnp: object,
    pltpu: object,
    tensor: torch.Tensor,
    entry: tuple[tuple[int | None, ...], tuple[int | tuple[int, int, int] | None, ...]]
    | None,
    should_use_smem: bool = False,
) -> object:
    """Build one ``pl.BlockSpec`` from compile-time ``(block_shape, grid_dims)``."""

    memory_space = None  # default value (pallas will default to VMEM)
    if should_use_smem:
        # pyrefly: ignore[missing-attribute]
        memory_space = pltpu.SMEM

    if entry is None:
        ndim = tensor.ndim
        full_shape = tuple(tensor.shape)

        def index_map_full(*grid_args: object, _nd: int = ndim) -> tuple[object, ...]:
            # pyrefly: ignore[missing-attribute]
            return tuple(jnp.int32(0) for _ in range(_nd))

        return pl.BlockSpec(full_shape, index_map_full, memory_space=memory_space)  # type: ignore[union-attr]

    block_shape_template, grid_dims = entry
    block_shape = tuple(
        min(bs, tensor.shape[d]) if bs is not None else tensor.shape[d]
        for d, bs in enumerate(block_shape_template)
    )

    def _index_for_dim(
        grid_args: tuple[object, ...],
        g: int | tuple[int, int, int] | None,
        jnp: object = jnp,
    ) -> object:
        if g is None:
            return jnp.int32(0)  # pyrefly: ignore[missing-attribute]
        if isinstance(g, tuple):
            # Flat grid decomposition: (grid_dim, stride, num_blocks)
            grid_dim, stride, num_blocks = g
            val = grid_args[grid_dim]
            if stride > 1:
                val = val // stride  # type: ignore[operator]
            val = val % num_blocks  # type: ignore[operator]
            return jnp.int32(val)  # pyrefly: ignore[missing-attribute]
        return jnp.int32(grid_args[g])  # pyrefly: ignore[missing-attribute]

    def index_map(
        *grid_args: object,
        _grid_dims: tuple[int | tuple[int, int, int] | None, ...] = grid_dims,
    ) -> tuple[object, ...]:
        return tuple(_index_for_dim(grid_args, g) for g in _grid_dims)

    return pl.BlockSpec(block_shape, index_map, memory_space=memory_space)  # type: ignore[union-attr]


# Per-tensor block spec info: see ``_pallas_make_block_spec``.
# grid_dims entries are int (direct grid dim), tuple (flat decomposition),
# or None (untiled dim).
_BlockSpecInfo = list[
    tuple[tuple[int | None, ...], tuple[int | tuple[int, int, int] | None, ...]] | None
]


def _pallas_build_block_specs(
    pl: object,
    jnp: object,
    pltpu: object,
    grid: tuple[int, ...],
    args: tuple[object, ...],
    tensor_arg_indices: list[int],
    output_indices: list[int],
    block_spec_info: _BlockSpecInfo | None = None,
    _smem_arg_indices: list[int] | None = None,
    _output_only_set: set[int] | None = None,
) -> tuple[list[object] | None, object | None]:
    """Build ``in_specs`` and ``out_specs`` for ``pl.pallas_call``.

    Output-only tensors (in ``_output_only_set``) get HBM in_specs
    to avoid VMEM pressure while keeping VMEM out_specs for writes.
    """
    if block_spec_info is None or len(grid) == 0:
        return None, None

    output_only_set = _output_only_set or set()

    in_specs = []
    for tensor_pos, idx in enumerate(tensor_arg_indices):
        t = args[idx]
        assert isinstance(t, torch.Tensor)
        if idx in output_only_set:
            in_specs.append(pl.BlockSpec(memory_space=pl.ANY))  # type: ignore[union-attr]
        else:
            should_use_smem = tensor_pos in (_smem_arg_indices or [])
            in_specs.append(
                _pallas_make_block_spec(
                    pl, jnp, pltpu, t, block_spec_info[tensor_pos], should_use_smem
                )
            )

    arg_to_tensor_pos = {orig: tpos for tpos, orig in enumerate(tensor_arg_indices)}
    out_specs_list = []
    for idx in output_indices:
        t = args[idx]
        assert isinstance(t, torch.Tensor)
        # Output-only tensors keep VMEM out_specs so the kernel can write
        # to them; only their in_specs use HBM to avoid VMEM pressure.
        should_use_smem = arg_to_tensor_pos[idx] in (_smem_arg_indices or [])
        out_specs_list.append(
            _pallas_make_block_spec(
                pl,
                jnp,
                pltpu,
                t,
                block_spec_info[arg_to_tensor_pos[idx]],
                should_use_smem,
            )
        )

    out_specs = out_specs_list if len(out_specs_list) > 1 else out_specs_list[0]
    return in_specs, out_specs


def _pallas_build_pipeline_specs(
    pl: object,
    jnp: object,
    pltpu: object,
    grid: tuple[int, ...],
    args: tuple[object, ...],
    tensor_arg_indices: list[int],
    output_indices: list[int],
    block_spec_info: _BlockSpecInfo | None,
    pipeline_arg_indices: list[int] | None,
) -> tuple[list[object], object]:
    """Build in/out specs for pipeline launchers.

    Pipeline-body tensors (listed in *pipeline_arg_indices*) get HBM refs.
    All other tensors get proper BlockSpecs for automatic VMEM prefetch.
    """
    pipeline_set = set(pipeline_arg_indices or [])
    arg_to_tpos = {orig: tpos for tpos, orig in enumerate(tensor_arg_indices)}

    def _spec_for(idx: int) -> object:
        if idx in pipeline_set:
            return pl.BlockSpec(memory_space=pltpu.HBM)  # type: ignore[union-attr]
        if block_spec_info is not None:
            t = args[idx]
            assert isinstance(t, torch.Tensor)
            return _pallas_make_block_spec(
                pl, jnp, pltpu, t, block_spec_info[arg_to_tpos[idx]]
            )
        return pl.BlockSpec(memory_space=pl.ANY)  # type: ignore[union-attr]

    in_specs = [_spec_for(idx) for idx in tensor_arg_indices]
    out_specs_list = [_spec_for(idx) for idx in output_indices]
    out_specs = out_specs_list if len(out_specs_list) > 1 else out_specs_list[0]
    return in_specs, out_specs


def _jax_placeholder_for_tensor(t: torch.Tensor) -> object:
    """Create a JAX ShapeDtypeStruct placeholder for a torch.Tensor.

    Used as a fallback when ``torch_tpu`` is not available (e.g. interpret mode
    on CPU).
    """
    import jax
    from torch._inductor.runtime.runtime_utils import torch_dtype_to_jax_runtime

    jax_dtype = torch_dtype_to_jax_runtime(t.dtype)
    return jax.ShapeDtypeStruct(tuple(t.shape), jax_dtype)


def _pallas_prepare_args(
    args: tuple[object, ...],
    _output_indices: list[int],
) -> tuple[
    set[int],
    list[int],
    dict[int, object],
    int,
    dict[int, int],
    list[object],
    set[int],
    tuple[object, ...],
]:
    """Extract and organize tensor/non-tensor args for Pallas launchers.

    Returns a tuple of:
    - output_set: set of output arg positions
    - tensor_arg_indices: positions of tensor args
    - non_tensor_args: mapping of non-tensor arg positions to values
    - n_tensor_inputs: count of tensor args
    - arg_to_tensor_pos: mapping from original position to tensor-only position
    - outputs: list of output tensors
    - inplace_positions: positions that are both input and output
    - out_shapes: JAX placeholders for output shapes
    """
    from .settings import is_pallas_interpret

    if is_pallas_interpret():
        placeholder_fn = _jax_placeholder_for_tensor
    else:
        from torch_tpu._internal.pallas.pallas import (  # pyrefly: ignore[missing-import]
            jax_placeholder,
        )

        placeholder_fn = jax_placeholder

    output_set = set(_output_indices)
    tensor_arg_indices = [
        i for i in range(len(args)) if isinstance(args[i], torch.Tensor)
    ]
    non_tensor_args: dict[int, object] = {
        i: args[i] for i in range(len(args)) if not isinstance(args[i], torch.Tensor)
    }
    n_tensor_inputs = len(tensor_arg_indices)
    arg_to_tensor_pos = {orig: tpos for tpos, orig in enumerate(tensor_arg_indices)}
    outputs = [args[i] for i in _output_indices]
    inplace_positions = output_set & set(tensor_arg_indices)
    out_shapes = tuple(placeholder_fn(out) for out in outputs)  # type: ignore[arg-type]

    return (
        output_set,
        tensor_arg_indices,
        non_tensor_args,
        n_tensor_inputs,
        arg_to_tensor_pos,
        outputs,
        inplace_positions,
        out_shapes,
    )


def _pallas_make_reordered_kernel(
    pallas_kernel: object,
    args: tuple[object, ...],
    tensor_arg_indices: list[int],
    non_tensor_args: dict[int, object],
    n_tensor_inputs: int,
    _output_indices: list[int],
    inplace_positions: set[int],
    arg_to_tensor_pos: dict[int, int],
    n_extra_refs: int = 0,
    skip_inplace_copy: set[int] | None = None,
    _smem_arg_indices: list[int] | None = None,
) -> object:
    """Create a wrapper kernel that reorders pallas_call refs to the original arg order.

    ``pallas_call`` provides refs as ``[inputs..., outputs...]``, but Helion
    kernels expect the original parameter order.  When *n_extra_refs* > 0
    (e.g. scratch buffers), those trailing refs are appended after the
    reordered args.

    *skip_inplace_copy* is a set of original-arg positions for which the
    initial ``out_ref[...] = in_ref[...]`` copy should be skipped.  This is
    needed for outputs backed by HBM refs (``pl.ANY``) where direct
    load/store is not allowed.  Outputs with VMEM BlockSpecs still get the
    copy so that ``input_output_aliases`` correctly preloads their contents.
    """
    _skip_copy = skip_inplace_copy or set()

    def reordered_kernel(*refs: object) -> None:
        n_kernel_params = len(args)
        original_order: list[object] = [None] * n_kernel_params
        for tensor_pos, orig_pos in enumerate(tensor_arg_indices):
            original_order[orig_pos] = refs[tensor_pos]
        for orig_pos, value in non_tensor_args.items():
            original_order[orig_pos] = value
        for out_idx, orig_pos in enumerate(_output_indices):
            out_ref = refs[n_tensor_inputs + out_idx]
            if orig_pos in inplace_positions and orig_pos not in _skip_copy:
                in_ref = refs[arg_to_tensor_pos[orig_pos]]
                if _smem_arg_indices is not None and orig_pos in _smem_arg_indices:
                    # [...] cannot be used for SMEMs,
                    # TODO(dunfanlu): handle in-place copy for SMEM refs
                    pass
                else:
                    out_ref[...] = in_ref[...]  # type: ignore[index]
            original_order[orig_pos] = out_ref
        extra_refs = refs[n_tensor_inputs + len(_output_indices) :]
        pallas_kernel(*original_order, *extra_refs)  # type: ignore[operator]

    return reordered_kernel


def _pallas_build_callable(
    pallas_kernel: object,
    grid: tuple[int, ...],
    jit_fn: object,
    _output_indices: list[int],
    arg_to_tensor_pos: dict[int, int],
    tensor_arg_indices: list[int],
    cache_attr: str,
    trace_key_suffix: str = "",
) -> object:
    """Build a ``JaxCallable``, cache it on the kernel, and return it.

    When ``torch_tpu`` is available, wraps the function in a ``JaxCallable``
    for efficient torch<->JAX interop.  Otherwise (interpret mode on CPU),
    returns a thin wrapper that converts tensors manually.
    """

    def _make_interpret_callable() -> _PallasInterpretCallable:
        output_tensor_positions = [
            arg_to_tensor_pos[orig_pos] for orig_pos in _output_indices
        ]
        callable_obj = _PallasInterpretCallable(jit_fn, output_tensor_positions)
        setattr(pallas_kernel, cache_attr, (grid, callable_obj, tensor_arg_indices))
        return callable_obj

    if _pallas_interpret_flag():
        return _make_interpret_callable()

    import jax
    from torch_tpu._internal.pallas.pallas import (  # pyrefly: ignore[missing-import]
        JaxCallable,
    )

    kernel_name = getattr(pallas_kernel, "__name__", "pallas_kernel")

    call_aliases: dict[int, int] = {}
    for out_idx, orig_pos in enumerate(_output_indices):
        call_aliases[arg_to_tensor_pos[orig_pos]] = out_idx

    jax_callable = JaxCallable(
        name=kernel_name,
        jit_fn=jax.jit(jit_fn),  # pyrefly: ignore[no-matching-overload]
        trace_key=f"{kernel_name}_{id(pallas_kernel)}_{grid}{trace_key_suffix}",
        input_output_aliases=call_aliases,
    )
    setattr(pallas_kernel, cache_attr, (grid, jax_callable, tensor_arg_indices))
    return jax_callable


class _PallasInterpretCallable:
    """Thin wrapper that converts torch tensors <-> JAX arrays for interpret mode.

    ``pallas_call`` with ``input_output_aliases`` returns new JAX arrays for the
    outputs.  This wrapper copies those results back into the original torch
    output tensors (identified by ``output_tensor_positions``).
    """

    def __init__(
        self,
        jit_fn: object,
        output_tensor_positions: list[int],
    ) -> None:
        self._jit_fn = jit_fn
        self._output_tensor_positions = output_tensor_positions

    def __call__(self, *input_tensors: torch.Tensor) -> None:
        jax_inputs = [_torch_to_jax(t) for t in input_tensors]
        jax_results = self._jit_fn(*jax_inputs)  # type: ignore[operator]
        if not isinstance(jax_results, (tuple, list)):
            jax_results = (jax_results,)
        # Write results back into the original output tensors.
        for out_idx, tensor_pos in enumerate(self._output_tensor_positions):
            out_tensor = input_tensors[tensor_pos]
            result_data = _jax_to_torch(
                jax_results[out_idx], device=out_tensor.device, dtype=out_tensor.dtype
            )
            out_tensor.copy_(result_data)


def _pallas_interpret_flag() -> bool:
    """Return True if ``HELION_PALLAS_INTERPRET=1`` is set.

    As a side effect, registers a synthetic CPU TpuInfo entry so that
    ``emit_pipeline`` / ``fori_loop`` interpret paths don't fail.
    """
    from .settings import is_pallas_interpret

    result = is_pallas_interpret()
    if result:
        _ensure_cpu_tpu_info()
    return result


def _ensure_cpu_tpu_info() -> None:
    """Register a synthetic TpuInfo for ``"cpu"`` so that
    ``emit_pipeline`` / ``fori_loop`` interpret paths don't fail.
    """
    try:
        from jax._src.pallas.mosaic.tpu_info import ChipVersion
        from jax._src.pallas.mosaic.tpu_info import _get_tpu_info_impl
        from jax._src.pallas.mosaic.tpu_info import registry
    except ImportError:
        return
    if "cpu" not in registry:
        registry["cpu"] = lambda: _get_tpu_info_impl(ChipVersion.TPU_7X, 1)


def default_pallas_launcher(
    pallas_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    _output_indices: list[int] | None = None,
    _inplace_indices: list[int] | None = None,
    _block_spec_info: _BlockSpecInfo | None = None,
    _smem_arg_indices: list[int] | None = None,
    **kwargs: object,
) -> None:
    """Default launcher for Pallas kernels on TPU (or CPU with interpret=True).

    Uses ``JaxCallable`` from ``torch_tpu`` to compile and run the Pallas
    kernel on TPU.  When ``torch_tpu`` is not available (interpret mode),
    falls back to direct torch<->JAX conversion.  Output tensors are donated
    via ``input_output_aliases`` so the kernel writes directly into their
    buffers (zero-copy on TPU).

    Output-only tensors (in ``_output_indices`` but not in ``_inplace_indices``)
    get HBM in_specs to avoid VMEM pressure while still being donated.
    """
    if _output_indices is None:
        _output_indices = []

    cache = getattr(pallas_kernel, "_pallas_cache", None)
    if cache is not None and cache[0] == grid:
        _, jax_callable, tensor_arg_indices = cache
    else:
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        import jax.numpy as jnp

        (
            output_set,
            tensor_arg_indices,
            non_tensor_args,
            n_tensor_inputs,
            arg_to_tensor_pos,
            outputs,
            inplace_positions,
            out_shapes,
        ) = _pallas_prepare_args(args, _output_indices)

        # Derive output-only set: outputs not in _inplace_indices.
        inplace_set = (
            set(_inplace_indices)
            if _inplace_indices is not None
            else set(_output_indices)
        )
        output_only_set = set(_output_indices) - inplace_set

        in_specs, out_specs = _pallas_build_block_specs(
            pl,
            jnp,
            pltpu,
            grid,
            args,
            tensor_arg_indices,
            _output_indices,
            _block_spec_info,
            _smem_arg_indices,
            output_only_set,
        )

        reordered_kernel = _pallas_make_reordered_kernel(
            pallas_kernel,
            args,
            tensor_arg_indices,
            non_tensor_args,
            n_tensor_inputs,
            _output_indices,
            inplace_positions,
            arg_to_tensor_pos,
            _smem_arg_indices=_smem_arg_indices,
            skip_inplace_copy=output_only_set,
        )

        out_shape_arg = out_shapes if len(out_shapes) > 1 else out_shapes[0]

        pallas_aliases = {
            arg_to_tensor_pos[orig_pos]: out_idx
            for out_idx, orig_pos in enumerate(_output_indices)
        }

        pallas_call_kwargs: dict[str, object] = {
            "out_shape": out_shape_arg,
            "input_output_aliases": pallas_aliases,
            "grid": grid,
        }
        if _pallas_interpret_flag():
            pallas_call_kwargs["interpret"] = True
        if in_specs is not None:
            pallas_call_kwargs["in_specs"] = in_specs
            pallas_call_kwargs["out_specs"] = out_specs

        jit_fn = pl.pallas_call(
            reordered_kernel,  # pyrefly: ignore[bad-argument-type]
            **pallas_call_kwargs,  # type: ignore[arg-type]
        )

        jax_callable = _pallas_build_callable(
            pallas_kernel,
            grid,
            jit_fn,
            _output_indices,
            arg_to_tensor_pos,
            tensor_arg_indices,
            cache_attr="_pallas_cache",
        )

    input_tensors = [
        cast("torch.Tensor", args[i]).contiguous() for i in tensor_arg_indices
    ]
    jax_callable(*input_tensors)  # type: ignore[operator]


def default_pallas_pipeline_launcher(
    pallas_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    _output_indices: list[int] | None = None,
    _block_spec_info: _BlockSpecInfo | None = None,
    _scratch_shapes: list[tuple[tuple[int, ...], str]] | None = None,
    _pipeline_arg_indices: list[int] | None = None,
    **kwargs: object,
) -> None:
    """Launcher for Pallas kernels using PrefetchScalarGridSpec with scratch memory.

    Used when ``pallas_loop_type='emit_pipeline'``.  Pipeline-body tensors
    (listed in ``_pipeline_arg_indices``) use HBM refs; all other tensors
    get proper BlockSpecs for automatic VMEM prefetch.
    """
    if _output_indices is None:
        _output_indices = []
    if _scratch_shapes is None:
        _scratch_shapes = []

    cache = getattr(pallas_kernel, "_pallas_pipeline_cache", None)
    if cache is not None and cache[0] == grid:
        _, jax_callable, tensor_arg_indices = cache
    else:
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        import jax.numpy as jnp

        (
            output_set,
            tensor_arg_indices,
            non_tensor_args,
            n_tensor_inputs,
            arg_to_tensor_pos,
            outputs,
            inplace_positions,
            out_shapes,
        ) = _pallas_prepare_args(args, _output_indices)

        # Build scratch shapes for VMEM
        _jnp_dtype_map: dict[str, object] = {
            "jnp.float32": jnp.float32,
            "jnp.float16": jnp.float16,
            "jnp.bfloat16": jnp.bfloat16,
            "jnp.int32": jnp.int32,
            "jnp.int16": jnp.int16,
            "jnp.int8": jnp.int8,
            "jnp.uint8": jnp.uint8,
            "jnp.bool_": jnp.bool_,
        }
        scratch_shapes = []
        for scratch_entry in _scratch_shapes:
            if len(scratch_entry) == 3:
                shape, dtype_str, scratch_type = scratch_entry
            else:
                shape, dtype_str = scratch_entry  # type: ignore[misc]
                scratch_type = "vmem"
            if scratch_type == "dma_semaphore":
                scratch_shapes.append(pltpu.SemaphoreType.DMA(()))
            else:
                jnp_dtype = _jnp_dtype_map.get(dtype_str, jnp.float32)
                scratch_shapes.append(
                    pltpu.VMEM(shape, jnp_dtype)  # pyrefly: ignore[bad-argument-type]
                )

        in_specs_list, out_specs = _pallas_build_pipeline_specs(
            pl,
            jnp,
            pltpu,
            grid,
            args,
            tensor_arg_indices,
            _output_indices,
            _block_spec_info,
            _pipeline_arg_indices,
        )

        _pipeline_set = set(_pipeline_arg_indices or [])
        reordered_kernel = _pallas_make_reordered_kernel(
            pallas_kernel,
            args,
            tensor_arg_indices,
            non_tensor_args,
            n_tensor_inputs,
            _output_indices,
            inplace_positions,
            arg_to_tensor_pos,
            n_extra_refs=len(scratch_shapes),
            skip_inplace_copy=_pipeline_set,
        )

        out_shape_arg = out_shapes if len(out_shapes) > 1 else out_shapes[0]

        pallas_aliases = {
            arg_to_tensor_pos[orig_pos]: out_idx
            for out_idx, orig_pos in enumerate(_output_indices)
        }

        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=in_specs_list,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
            grid=grid,
        )

        pallas_call_kwargs: dict[str, object] = {
            "out_shape": out_shape_arg,
            "input_output_aliases": pallas_aliases,
            "grid_spec": grid_spec,
            "compiler_params": pltpu.CompilerParams(  # pyrefly: ignore[bad-instantiation]
                dimension_semantics=tuple("parallel" for _ in grid),
            ),
        }
        if _pallas_interpret_flag():
            pallas_call_kwargs["interpret"] = True

        jit_fn = pl.pallas_call(
            reordered_kernel,  # pyrefly: ignore[bad-argument-type]
            **pallas_call_kwargs,  # type: ignore[arg-type]
        )

        jax_callable = _pallas_build_callable(
            pallas_kernel,
            grid,
            jit_fn,
            _output_indices,
            arg_to_tensor_pos,
            tensor_arg_indices,
            cache_attr="_pallas_pipeline_cache",
            trace_key_suffix="_pipeline",
        )

    input_tensors = [
        cast("torch.Tensor", args[i]).contiguous() for i in tensor_arg_indices
    ]
    jax_callable(*input_tensors)  # type: ignore[operator]


def default_pallas_fori_launcher(
    pallas_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    _output_indices: list[int] | None = None,
    _block_spec_info: _BlockSpecInfo | None = None,
    _scratch_shapes: list[tuple[tuple[int, ...], str | None, str]] | None = None,
    **kwargs: object,
) -> None:
    """Launcher for Pallas kernels using fori_loop with manual DMA.

    Used when ``pallas_loop_type="fori_loop"``.  Passes all tensors as
    ``memory_space=pl.ANY`` (HBM refs) and adds scratch buffers as
    ``pltpu.VMEM`` shapes plus ``pltpu.SemaphoreType.DMA`` for async copies.
    The kernel uses ``jax.lax.fori_loop`` with ``pltpu.make_async_copy``
    internally for DMA control.
    """
    if _output_indices is None:
        _output_indices = []
    if _scratch_shapes is None:
        _scratch_shapes = []

    cache = getattr(pallas_kernel, "_pallas_fori_cache", None)
    if cache is not None and cache[0] == grid:
        _, jax_callable, tensor_arg_indices = cache
    else:
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        import jax.numpy as jnp

        (
            output_set,
            tensor_arg_indices,
            non_tensor_args,
            n_tensor_inputs,
            arg_to_tensor_pos,
            outputs,
            inplace_positions,
            out_shapes,
        ) = _pallas_prepare_args(args, _output_indices)

        # Build scratch shapes: VMEM buffers + DMA semaphores
        _jnp_dtype_map: dict[str, object] = {
            "jnp.float32": jnp.float32,
            "jnp.float16": jnp.float16,
            "jnp.bfloat16": jnp.bfloat16,
            "jnp.int32": jnp.int32,
            "jnp.int16": jnp.int16,
            "jnp.int8": jnp.int8,
            "jnp.uint8": jnp.uint8,
            "jnp.bool_": jnp.bool_,
        }
        scratch_shapes = []
        for shape, dtype_str, scratch_type in _scratch_shapes:
            if scratch_type == "dma_semaphore":
                scratch_shapes.append(pltpu.SemaphoreType.DMA(()))
            else:  # "vmem"
                assert dtype_str is not None
                jnp_dtype = _jnp_dtype_map.get(dtype_str, jnp.float32)
                scratch_shapes.append(
                    pltpu.VMEM(shape, jnp_dtype)  # pyrefly: ignore[bad-argument-type]
                )

        # Build in_specs/out_specs: proper BlockSpecs for outer grid dims,
        # HBM refs for tensors used in the fori_loop body (DMA handles tiling).
        _fori_pipeline_indices = kwargs.get("_pipeline_arg_indices")
        in_specs_list, out_specs = _pallas_build_pipeline_specs(
            pl,
            jnp,
            pltpu,
            grid,
            args,
            tensor_arg_indices,
            _output_indices,
            _block_spec_info,
            _fori_pipeline_indices,  # type: ignore[arg-type]
        )

        _fori_pipeline_set = set(_fori_pipeline_indices or [])  # type: ignore[arg-type]
        reordered_kernel = _pallas_make_reordered_kernel(
            pallas_kernel,
            args,
            tensor_arg_indices,
            non_tensor_args,
            n_tensor_inputs,
            _output_indices,
            inplace_positions,
            arg_to_tensor_pos,
            n_extra_refs=len(scratch_shapes),
            skip_inplace_copy=_fori_pipeline_set,
        )

        out_shape_arg = out_shapes if len(out_shapes) > 1 else out_shapes[0]

        pallas_aliases = {
            arg_to_tensor_pos[orig_pos]: out_idx
            for out_idx, orig_pos in enumerate(_output_indices)
        }

        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=in_specs_list,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
            grid=grid,
        )

        pallas_call_kwargs: dict[str, object] = {
            "out_shape": out_shape_arg,
            "input_output_aliases": pallas_aliases,
            "grid_spec": grid_spec,
            "compiler_params": pltpu.CompilerParams(  # pyrefly: ignore[bad-instantiation]
                dimension_semantics=tuple("parallel" for _ in grid),
            ),
        }
        if _pallas_interpret_flag():
            pallas_call_kwargs["interpret"] = True

        jit_fn = pl.pallas_call(
            reordered_kernel,  # pyrefly: ignore[bad-argument-type]
            **pallas_call_kwargs,  # type: ignore[arg-type]
        )

        jax_callable = _pallas_build_callable(
            pallas_kernel,
            grid,
            jit_fn,
            _output_indices,
            arg_to_tensor_pos,
            tensor_arg_indices,
            cache_attr="_pallas_fori_cache",
            trace_key_suffix="_fori",
        )

    input_tensors = [
        cast("torch.Tensor", args[i]).contiguous() for i in tensor_arg_indices
    ]
    jax_callable(*input_tensors)  # type: ignore[operator]


def _torch_to_jax(t: torch.Tensor) -> object:
    """Convert a torch.Tensor to a JAX array via numpy (for interpret mode on CPU)."""
    import jax.numpy as jnp
    import numpy as np

    return jnp.array(np.asarray(t.detach().cpu()))


def _jax_to_torch(
    arr: object, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Convert a JAX array back to a torch.Tensor via numpy (for interpret mode on CPU)."""
    import numpy as np

    return torch.from_numpy(np.asarray(arr)).to(dtype=dtype, device=device)


def _torch_dtype_to_cutlass(dtype: torch.dtype) -> object:
    _patch_cutlass_jit_shutdown_unload()
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
    _patch_cutlass_jit_shutdown_unload()
    import cutlass
    import cutlass.cute as cute

    kernel_name = getattr(cast("Any", cute_kernel), "__name__", "cute_kernel")
    kernel_tag = f"{kernel_name}_{id(cute_kernel):x}"
    func_name = f"_helion_cute_launch_{kernel_tag}"
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
            f"    _helion_cute_kernel_tag = {kernel_tag!r}",
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
    filename = f"<helion_cute_launcher:{kernel_tag}:{schema_key!r}>"
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
    try:
        # pyrefly: ignore [missing-attribute]
        cache = cute_kernel._helion_cute_compiled_launchers
    except AttributeError:
        cache = {}
        # pyrefly: ignore [missing-attribute]
        cute_kernel._helion_cute_compiled_launchers = cache
    cached = cache.get(schema_key)
    if cached is not None:
        return cached

    wrapper = _create_cute_wrapper(cute_kernel, schema_key)
    cache[schema_key] = wrapper
    return wrapper


def _build_cute_schema_and_args(
    args: tuple[object, ...],
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
) -> tuple[tuple[tuple[object, ...], ...], tuple[object, ...]]:
    _patch_cutlass_jit_shutdown_unload()
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

    if any(dim <= 0 for dim in grid_xyz):
        return None

    schema_key, launch_args = _build_cute_schema_and_args(
        tuple(args), grid_xyz, block_xyz
    )
    compiled = _get_compiled_cute_launcher(cute_kernel, schema_key, launch_args)
    return cast("Any", compiled)(*launch_args)


def default_metal_launcher(
    metal_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    _block_size: int = 256,
    **kwargs: object,
) -> None:
    """Default launcher for Metal kernels on Apple MPS devices.

    Compiles MSL source via ``torch.mps.compile_shader()`` and dispatches
    using the compiled library.  Caches the compiled library on the kernel
    object to avoid recompilation on subsequent calls.

    Only 1D grids are currently supported.
    """
    kwargs.pop("num_warps", None)
    kwargs.pop("num_stages", None)
    if kwargs:
        raise exc.BackendUnsupported(
            "metal", f"unexpected launcher kwargs: {sorted(kwargs)}"
        )

    assert len(grid) == 1, (
        f"Metal launcher only supports 1D grids, got {len(grid)}D: {grid}"
    )

    msl_source, kernel_name = metal_kernel()  # type: ignore[operator]
    source_hash = hashlib.sha256(msl_source.encode()).digest()
    cache = getattr(metal_kernel, "_metal_cache", None)
    if cache is not None and cache[0] == source_hash:
        lib = cache[1]
    else:
        lib = torch.mps.compile_shader(msl_source)  # type: ignore[attr-defined]
        metal_kernel._metal_cache = (source_hash, lib)  # type: ignore[attr-defined]

    tensor_args = [a for a in args if isinstance(a, torch.Tensor)]
    dispatch_fn = getattr(lib, kernel_name)
    total_threads = grid[0] * _block_size
    dispatch_fn(*tensor_args, threads=total_threads, group_size=_block_size)
