from __future__ import annotations

from contextlib import suppress
import contextvars
import importlib
import inspect
import linecache
import os
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import TypedDict
from typing import cast

import torch

from .. import _compat as _compat  # ensure Triton compatibility patches run
from .. import exc
from .._compiler.cute.strategies import tcgen05_default_epilogue_tile_expr
from .._compiler.cute.strategies import tcgen05_explicit_d_store_tile_expr
from .._compiler.cute.strategies import tcgen05_smem_layout_expr
from .._compiler.cute.tcgen05_constants import TCGEN05_DIRECT_ENTRY_CLUSTER_M
from .._compiler.cute.tcgen05_constants import (
    TCGEN05_DIRECT_ENTRY_SHAPE_SETS_BY_BK as _DIRECT_ENTRY_SHAPE_SETS_BY_BK,
)
from .._compiler.cute.tcgen05_constants import (
    TCGEN05_DIRECT_ENTRY_STAGE_TUPLES_BY_BK as _DIRECT_ENTRY_STAGE_TUPLES_BY_BK,
)
from .._compiler.cute.tcgen05_constants import TCGEN05_DIRECT_ENTRY_TOTAL_WORK_CLUSTERS
from .._utils import triton_is_available
from .config import Config as Config
from .kernel import Kernel as Kernel
from .kernel import kernel as kernel

if TYPE_CHECKING:
    from collections.abc import Callable

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


_CACHED_VMEM_LIMIT_BYTES: int | None = None


def _get_vmem_limit_bytes(pltpu: object) -> int:
    """Safely retrieves the TPU VMEM capacity without crashing on hardware locks."""
    global _CACHED_VMEM_LIMIT_BYTES
    if _CACHED_VMEM_LIMIT_BYTES is not None:
        return _CACHED_VMEM_LIMIT_BYTES

    # In interpret mode there is no real TPU; query the synthetic TPU info
    # registered by ``_ensure_cpu_tpu_info`` so the budget matches what real
    # TPU 7X reports rather than falling back to the conservative 16MB default.
    from .settings import is_pallas_interpret

    if is_pallas_interpret():
        try:
            from jax._src.pallas.mosaic.tpu_info import registry

            _CACHED_VMEM_LIMIT_BYTES = registry["cpu"]().vmem_capacity_bytes
            return _CACHED_VMEM_LIMIT_BYTES
        except (ImportError, KeyError, AttributeError):
            pass

    try:
        get_tpu_info = pltpu.get_tpu_info  # pyrefly: ignore[missing-attribute]
        _CACHED_VMEM_LIMIT_BYTES = get_tpu_info().vmem_capacity_bytes
    except Exception:
        # Fallback if JAX fails to acquire the TPU backend lock (e.g., in a precompile fork).
        # Default to 16MB (safe baseline for v4 and v5e per-core VMEM).
        _CACHED_VMEM_LIMIT_BYTES = 16 * 1024 * 1024

    return _CACHED_VMEM_LIMIT_BYTES


def _estimate_pallas_vmem_bytes(
    pl: object,
    pltpu: object,
    in_specs: list[object] | None,
    out_specs: list[object] | object | None,
    scratch_shapes: list[object] | list[Any] | None,
    args: tuple[object, ...],
    tensor_arg_indices: list[int],
    output_indices: list[int],
    pallas_aliases: dict[int, int] | None,
) -> int:
    """Estimates the VMEM required by the Pallas kernel."""
    total_bytes = 0
    in_spec_bytes = [0] * len(tensor_arg_indices)
    out_spec_bytes = [0] * len(output_indices)

    def _bytes_per_element(t: object) -> int:
        import torch

        if isinstance(t, torch.Tensor):
            return t.element_size()

        dtype = getattr(t, "dtype", None)
        if dtype is not None:
            # Works for torch.dtype and np.dtype/jnp.dtype
            itemsize = getattr(dtype, "itemsize", None)
            if itemsize is not None:
                return itemsize

        return 4

    if in_specs:
        for i, idx in enumerate(tensor_arg_indices):
            spec = in_specs[i]
            # pl.BlockSpec will have block_shape and memory_space.
            # HBM is pl.ANY. We only count VMEM (which is not pl.ANY).
            if spec is not None and getattr(spec, "memory_space", None) is not getattr(
                pl, "ANY", None
            ):
                block_shape = getattr(spec, "block_shape", None)
                if block_shape is not None:
                    numel = 1
                    for d in block_shape:
                        numel *= int(d)
                    in_spec_bytes[i] = numel * _bytes_per_element(args[idx])

    if out_specs:
        out_specs_list = (
            out_specs if isinstance(out_specs, (list, tuple)) else [out_specs]
        )
        for i, idx in enumerate(output_indices):
            if i < len(out_specs_list):
                spec = out_specs_list[i]
                if spec is not None and getattr(
                    spec, "memory_space", None
                ) is not getattr(pl, "ANY", None):
                    block_shape = getattr(spec, "block_shape", None)
                    if block_shape is not None:
                        numel = 1
                        for d in block_shape:
                            numel *= int(d)
                        out_spec_bytes[i] = numel * _bytes_per_element(args[idx])

    pallas_aliases = pallas_aliases or {}
    aliased_out_positions = set()
    for in_pos, out_pos in pallas_aliases.items():
        aliased_out_positions.add(out_pos)
        if in_pos < len(in_spec_bytes) and out_pos < len(out_spec_bytes):
            in_spec_bytes[in_pos] = max(in_spec_bytes[in_pos], out_spec_bytes[out_pos])

    for out_pos in aliased_out_positions:
        if out_pos < len(out_spec_bytes):
            out_spec_bytes[out_pos] = 0

    # Pallas pipelines and default launchers natively double buffer their BlockSpecs.
    multiplier = 2
    total_bytes += sum(in_spec_bytes) * multiplier
    total_bytes += sum(out_spec_bytes) * multiplier

    if scratch_shapes:
        for scratch in scratch_shapes:
            if type(scratch).__name__ == "VMEM":
                numel = 1
                shape = getattr(scratch, "shape", ())
                for d in shape:
                    numel *= int(d)
                dtype_size = getattr(getattr(scratch, "dtype", None), "itemsize", 4)
                total_bytes += numel * dtype_size

    return total_bytes


# Per-tensor block spec info: see ``_pallas_make_block_spec``.
# grid_dims entries are int (direct grid dim), tuple (flat decomposition),
# or None (untiled dim).
_BlockSpecInfo = list[
    tuple[tuple[int | None, ...], tuple[int | tuple[int, int, int] | None, ...]] | None
]


class _PallasAtomicInfo(TypedDict):
    target_arg_pos: int
    ops: tuple[str, ...]
    return_used: bool


_PallasAtomicInfos = list[_PallasAtomicInfo]


def _pallas_tensor_pos_map(
    tensor_arg_indices: list[int],
    output_only_indices: list[int] | None,
) -> dict[int, int]:
    all_positions = sorted(set(tensor_arg_indices) | set(output_only_indices or []))
    return {orig: tpos for tpos, orig in enumerate(all_positions)}


def _pallas_grid_dims_used_by_block_spec(
    block_info: tuple[
        tuple[int | None, ...], tuple[int | tuple[int, int, int] | None, ...]
    ],
) -> set[int]:
    used: set[int] = set()
    _, grid_dims = block_info
    for grid_dim in grid_dims:
        if isinstance(grid_dim, int):
            used.add(grid_dim)
        elif isinstance(grid_dim, tuple):
            used.add(grid_dim[0])
    return used


def _pallas_atomic_contributor_plan(
    grid: tuple[int, ...],
    tensor_arg_indices: list[int],
    output_only_indices: list[int],
    output_indices: list[int],
    inplace_indices: list[int] | None,
    block_spec_info: _BlockSpecInfo | None,
    atomic_infos: _PallasAtomicInfos | None,
) -> tuple[dict[int, tuple[int, ...]], tuple[str, ...]]:
    """Plan ordered shared-tile atomic RMW for Pallas outputs.

    For a structural atomic such as split-K, the target BlockSpec maps only
    output-tile dimensions while one or more grid dimensions contribute to the
    same tile. Those contributor dimensions must run sequentially, and the
    output preload must happen only for the first contributor.
    """
    dim_semantics = ["parallel"] * len(grid)
    guarded_copies: dict[int, tuple[int, ...]] = {}
    if not atomic_infos:
        return guarded_copies, tuple(dim_semantics)
    if block_spec_info is None:
        raise RuntimeError(
            "Pallas atomic lowering requires block-spec metadata for atomic targets"
        )

    output_set = set(output_indices)
    inplace_set = set(inplace_indices or [])
    arg_to_tpos = _pallas_tensor_pos_map(tensor_arg_indices, output_only_indices)
    reduction_ops = {
        "atomic_add",
        "atomic_max",
        "atomic_min",
    }
    for info in atomic_infos:
        orig_pos = info["target_arg_pos"]
        ops = info["ops"]
        return_used = info["return_used"]
        if orig_pos not in output_set:
            raise RuntimeError(
                "Pallas atomic target is not an output tensor; cannot lower atomic RMW safely"
            )
        tensor_pos = arg_to_tpos.get(orig_pos)
        if tensor_pos is None or tensor_pos >= len(block_spec_info):
            raise RuntimeError(
                "Pallas atomic lowering requires block-spec metadata for the target tensor"
            )
        block_info = block_spec_info[tensor_pos]
        if block_info is None:
            raise RuntimeError(
                "Pallas atomic lowering requires block-spec metadata for the target tensor"
            )
        used_dims = _pallas_grid_dims_used_by_block_spec(block_info)
        # In split-K, the output tile uses the M/N grid dims; K is a contributor.
        contributor_dims = tuple(
            dim for dim, size in enumerate(grid) if size > 1 and dim not in used_dims
        )
        if not contributor_dims:
            continue
        if orig_pos not in inplace_set:
            # TODO(tcombes): route this through launcher-owned VMEM scratch once
            # shared atomic accumulator substitution is implemented.
            raise RuntimeError(
                "Pallas shared-tile atomics without an aliased input/output "
                "target are not supported by this lowering"
            )
        unsupported = tuple(op for op in ops if op not in reduction_ops)
        if unsupported:
            raise RuntimeError(
                "Pallas shared-tile atomics only support reduction-style ops "
                f"for now; got {unsupported}"
            )
        if return_used:
            raise RuntimeError(
                "Pallas shared-tile atomics with used return values are not supported yet"
            )
        guarded_copies[orig_pos] = contributor_dims
        for dim in contributor_dims:
            dim_semantics[dim] = "arbitrary"
    return guarded_copies, tuple(dim_semantics)


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
    output_only_indices: list[int] | None = None,
) -> tuple[list[object] | None, object | None]:
    """Build ``in_specs`` and ``out_specs`` for ``pl.pallas_call``.

    ``block_spec_info`` is indexed by position among *all* tensor args.
    ``output_only_indices`` lists tensor positions excluded from
    ``tensor_arg_indices``; they are merged back to compute the mapping.
    """
    if block_spec_info is None or len(grid) == 0:
        return None, None

    all_positions = sorted(set(tensor_arg_indices) | set(output_only_indices or []))
    all_arg_to_tensor_pos = {orig: tpos for tpos, orig in enumerate(all_positions)}

    in_specs = []
    for idx in tensor_arg_indices:
        t = args[idx]
        assert isinstance(t, torch.Tensor)
        tensor_pos = all_arg_to_tensor_pos[idx]
        should_use_smem = tensor_pos in (_smem_arg_indices or [])
        in_specs.append(
            _pallas_make_block_spec(
                pl, jnp, pltpu, t, block_spec_info[tensor_pos], should_use_smem
            )
        )

    out_specs_list = []
    for idx in output_indices:
        t = args[idx]
        assert isinstance(t, torch.Tensor)
        tensor_pos = all_arg_to_tensor_pos[idx]
        should_use_smem = tensor_pos in (_smem_arg_indices or [])
        out_specs_list.append(
            _pallas_make_block_spec(
                pl,
                jnp,
                pltpu,
                t,
                block_spec_info[tensor_pos],
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
    block_spec_info: _BlockSpecInfo,
    pipeline_arg_indices: list[int] | None,
    output_only_indices: list[int] | None = None,
    smem_arg_indices: list[int] | None = None,
) -> tuple[list[object], object]:
    """Build in/out specs for pipeline launchers.

    Pipeline-body tensors (listed in *pipeline_arg_indices*) get HBM refs.
    All other tensors get proper BlockSpecs for automatic VMEM prefetch.
    Tensors in *smem_arg_indices* (only ever accessed by scalar index, e.g.
    group offset tables) are placed in SMEM so dynamic scalar reads don't
    require 128-lane alignment proofs against a small VMEM ref.
    """
    pipeline_set = set(pipeline_arg_indices or [])
    smem_set = set(smem_arg_indices or [])
    all_positions = sorted(set(tensor_arg_indices) | set(output_only_indices or []))
    arg_to_tpos = {orig: tpos for tpos, orig in enumerate(all_positions)}

    def _spec_for(idx: int) -> object:
        if idx in pipeline_set:
            return pl.BlockSpec(memory_space=pltpu.HBM)  # type: ignore[union-attr]
        tpos = arg_to_tpos[idx]
        t = args[idx]
        assert isinstance(t, torch.Tensor)
        return _pallas_make_block_spec(
            pl, jnp, pltpu, t, block_spec_info[tpos], tpos in smem_set
        )

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


def _pallas_jnp_dtype_map() -> dict[str, object]:
    import jax.numpy as jnp

    return {
        "jnp.float32": jnp.float32,
        "jnp.float16": jnp.float16,
        "jnp.bfloat16": jnp.bfloat16,
        "jnp.int32": jnp.int32,
        "jnp.int16": jnp.int16,
        "jnp.int8": jnp.int8,
        "jnp.uint8": jnp.uint8,
        "jnp.bool_": jnp.bool_,
    }


def _pallas_check_dtypes(args: tuple[object, ...]) -> None:
    """Raise if any tensor arg uses a dtype unsupported on TPU."""
    from .._compiler.backend import _PALLAS_UNSUPPORTED_DTYPES

    for a in args:
        if isinstance(a, torch.Tensor) and a.dtype in _PALLAS_UNSUPPORTED_DTYPES:
            raise TypeError(
                f"Pallas/TPU does not support {a.dtype} tensors. "
                f"Cast to a 32-bit type before calling the kernel."
            )


def _pallas_prepare_args(
    args: tuple[object, ...],
    _output_indices: list[int],
    _inplace_indices: list[int] | None = None,
    *,
    interpret: bool = False,
) -> tuple[
    list[int],
    list[int],
    dict[int, object],
    int,
    dict[int, int],
    set[int],
    tuple[object, ...],
    dict[int, int],
]:
    """Extract and organize tensor/non-tensor args for Pallas launchers.

    Returns a tuple of:
    - tensor_arg_indices: positions of tensor args passed as pallas_call inputs
    - output_only_indices: positions of output-only tensors (excluded from inputs)
    - non_tensor_args: mapping of non-tensor arg positions to values
    - n_tensor_inputs: count of tensor inputs (excl. output-only)
    - arg_to_tensor_pos: mapping from original position to tensor-only position
    - inplace_positions: positions that are both input and output
    - out_shapes: JAX placeholders for output shapes
    """
    if interpret:
        placeholder_fn = _jax_placeholder_for_tensor
    else:
        from torch_tpu._internal.pallas.pallas import (  # pyrefly: ignore[missing-import]
            jax_placeholder,
        )

        placeholder_fn = jax_placeholder

    output_set = set(_output_indices)
    inplace_set = set(_inplace_indices) if _inplace_indices is not None else output_set
    output_only = output_set - inplace_set

    all_tensor_positions = [
        i for i in range(len(args)) if isinstance(args[i], torch.Tensor)
    ]
    output_only_indices = [i for i in all_tensor_positions if i in output_only]
    tensor_arg_indices = [i for i in all_tensor_positions if i not in output_only]

    non_tensor_args: dict[int, object] = {
        i: args[i] for i in range(len(args)) if not isinstance(args[i], torch.Tensor)
    }
    n_tensor_inputs = len(tensor_arg_indices)
    arg_to_tensor_pos = {orig: tpos for tpos, orig in enumerate(tensor_arg_indices)}
    inplace_positions = output_set & set(tensor_arg_indices)
    out_shapes = tuple(placeholder_fn(args[i]) for i in _output_indices)  # type: ignore[arg-type]

    pallas_aliases = {
        arg_to_tensor_pos[orig_pos]: out_idx
        for out_idx, orig_pos in enumerate(_output_indices)
        if orig_pos in arg_to_tensor_pos
    }

    return (
        tensor_arg_indices,
        output_only_indices,
        non_tensor_args,
        n_tensor_inputs,
        arg_to_tensor_pos,
        inplace_positions,
        out_shapes,
        pallas_aliases,
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
    atomic_copy_guards: dict[int, tuple[int, ...]] | None = None,
) -> object:
    """Create a wrapper kernel that reorders pallas_call refs to the original arg order.

    ``pallas_call`` provides refs as ``[inputs..., outputs...]``, but Helion
    kernels expect the original parameter order.  When *n_extra_refs* > 0
    (e.g. scratch buffers), those trailing refs are appended after the
    reordered args.

    *skip_inplace_copy* is a set of original-arg positions for which the
    initial ``out_ref[...] = in_ref[...]`` copy should be skipped.  Used by
    pipeline/fori launchers for pipeline-body tensors backed by HBM refs
    where direct load/store is not allowed.
    """
    _skip_copy = skip_inplace_copy or set()
    _atomic_copy_guards = atomic_copy_guards or {}
    if _atomic_copy_guards:
        from jax.experimental import pallas as pl

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
                elif orig_pos in _atomic_copy_guards:
                    # Only the first contributor initializes the accumulator;
                    # later contributors must preserve earlier RMW updates.
                    condition = None
                    for dim in _atomic_copy_guards[orig_pos]:
                        dim_is_first = pl.program_id(dim) == 0
                        condition = (
                            dim_is_first
                            if condition is None
                            else condition & dim_is_first
                        )
                    assert condition is not None

                    @pl.when(condition)
                    def _copy_atomic_initial_value(
                        out_ref: object = out_ref, in_ref: object = in_ref
                    ) -> None:
                        out_ref[...] = in_ref[...]  # type: ignore[index]

                else:
                    out_ref[...] = in_ref[...]  # type: ignore[index]
            original_order[orig_pos] = out_ref
        extra_refs = refs[n_tensor_inputs + len(_output_indices) :]
        pallas_kernel(*original_order, *extra_refs)  # type: ignore[operator]

    return reordered_kernel


def _pallas_build_callable(
    pallas_kernel: object,
    grid: tuple[int, ...],
    jit_fn: Callable[..., object],
    _output_indices: list[int],
    arg_to_tensor_pos: dict[int, int],
    tensor_arg_indices: list[int],
    cache_attr: str,
    call_aliases: dict[int, int],
    trace_key_suffix: str = "",
    *,
    interpret: bool = False,
) -> object:
    """Build a ``JaxCallable``, cache it on the kernel, and return it.

    When ``torch_tpu`` is available, wraps the function in a ``JaxCallable``
    for efficient torch<->JAX interop.  Otherwise (interpret mode on CPU),
    returns a thin wrapper that converts tensors manually.
    """

    def _make_interpret_callable() -> _PallasInterpretCallable:
        # Map (out_idx in _output_indices) -> tensor_pos for inplace outputs.
        # out_idx must match jax_results ordering (all outputs), not filtered.
        inplace_output_mapping = [
            (out_idx, arg_to_tensor_pos[orig_pos])
            for out_idx, orig_pos in enumerate(_output_indices)
            if orig_pos in arg_to_tensor_pos
        ]
        callable_obj = _PallasInterpretCallable(jit_fn, inplace_output_mapping)
        setattr(
            pallas_kernel,
            cache_attr,
            (grid, callable_obj, tensor_arg_indices, arg_to_tensor_pos),
        )
        return callable_obj

    if interpret:
        return _make_interpret_callable()

    import jax
    from torch_tpu._internal.pallas.pallas import (  # pyrefly: ignore[missing-import]
        JaxCallable,
    )

    kernel_name = getattr(pallas_kernel, "__name__", "pallas_kernel")

    jax_callable = JaxCallable(
        name=kernel_name,
        jit_fn=jax.jit(jit_fn),
        trace_key=f"{kernel_name}_{id(pallas_kernel)}_{grid}{trace_key_suffix}",
        input_output_aliases=call_aliases,
    )
    setattr(
        pallas_kernel,
        cache_attr,
        (grid, jax_callable, tensor_arg_indices, arg_to_tensor_pos),
    )
    return jax_callable


class _PallasInterpretCallable:
    """Thin wrapper that converts torch tensors <-> JAX arrays for interpret mode.

    In interpret mode, ``pallas_call`` runs on CPU and returns JAX arrays.
    This wrapper:
    1. Converts input torch tensors to JAX arrays
    2. Runs the pallas_call function
    3. For inplace outputs (donated tensors): copies JAX results back into
       the original torch tensors via ``copy_()``
    4. Returns raw JAX results so ``_pallas_invoke_and_return`` can
       handle output-only tensors (which are not in the input list)

    ``inplace_output_mapping`` maps each inplace output to its JAX result:
    a list of ``(out_idx, tensor_pos)`` where ``out_idx`` indexes into
    ``jax_results`` and ``tensor_pos`` indexes into ``input_tensors``.
    """

    def __init__(
        self,
        jit_fn: Callable[..., object],
        inplace_output_mapping: list[tuple[int, int]],
    ) -> None:
        self._jit_fn = jit_fn
        self._inplace_output_mapping = inplace_output_mapping

    def __call__(self, *input_tensors: torch.Tensor) -> tuple[object, ...]:
        jax_inputs = [_torch_to_jax(t) for t in input_tensors]
        jax_results = self._jit_fn(*jax_inputs)  # type: ignore[operator]
        if not isinstance(jax_results, (tuple, list)):
            jax_results = (jax_results,)
        # Write inplace results back into the original output tensors.
        for out_idx, tensor_pos in self._inplace_output_mapping:
            out_tensor = input_tensors[tensor_pos]
            result_data = _jax_to_torch(
                jax_results[out_idx], device=out_tensor.device, dtype=out_tensor.dtype
            )
            out_tensor.copy_(result_data)
        # Return JAX results so output-only tensors can be handled
        # by _pallas_invoke_and_return.
        return tuple(jax_results)


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


def _pallas_invoke_and_return(
    jax_callable: object,
    args: tuple[object, ...],
    tensor_arg_indices: list[int],
    arg_to_tensor_pos: dict[int, int],
    _output_indices: list[int],
    _ds_pad_dims: list[tuple[int, int, int, int]] | None = None,
    _orig_output_tensors: dict[int, torch.Tensor] | None = None,
) -> object:
    """Run the JaxCallable and return output-only results.

    Output-only tensors (those not in ``arg_to_tensor_pos``) are not passed
    as pallas_call inputs, so the JaxCallable returns new buffers for them.
    Returns a single tensor, a tuple of tensors, or None.

    When ``_ds_pad_dims`` is provided, also handles:
    - Copying sliced results back into original (unpadded) in-place output tensors
    - Slicing padded output-only result tensors back to original shapes
    """
    input_tensors = [
        cast("torch.Tensor", args[i]).contiguous() for i in tensor_arg_indices
    ]
    results = jax_callable(*input_tensors)  # type: ignore[operator]
    if results is None:
        return None
    if not isinstance(results, (tuple, list)):
        results = (results,)
    output_only_results = []
    for out_idx, orig_pos in enumerate(_output_indices):
        if orig_pos not in arg_to_tensor_pos:
            result = results[out_idx]
            if not isinstance(result, torch.Tensor):
                # Interpret mode: pallas_call returns JAX arrays, convert to torch.
                # On TPU, JaxCallable returns torch tensors directly.
                out_tensor = cast("torch.Tensor", args[orig_pos])
                # Output-only tensors are allocated with ``device='meta'`` to
                # avoid HBM; interpret mode runs on CPU so the converted
                # tensor lands there.
                device = out_tensor.device
                if device.type == "meta":
                    device = torch.device("cpu")
                result = _jax_to_torch(
                    result,
                    device=device,
                    dtype=out_tensor.dtype,
                )
            output_only_results.append(result)

    # Handle padding copy-back and result slicing
    if _ds_pad_dims and _orig_output_tensors:
        # _ds_pad_dims contains (arg_idx, dim, block_size, extra_pad).
        # Build a map from arg_idx → [(dim, ...)] for padded output args.
        padded_dims_by_arg: dict[int, list[int]] = {}
        for arg_idx, dim, _bs, _extra in _ds_pad_dims:
            if arg_idx in _orig_output_tensors:
                padded_dims_by_arg.setdefault(arg_idx, []).append(dim)

        # Copy sliced results back into original in-place output tensors.
        # Skip output-only tensors (not in arg_to_tensor_pos) — their
        # results come from output_only_results, not from args.
        for arg_idx, orig_tensor in _orig_output_tensors.items():
            if arg_idx not in arg_to_tensor_pos:
                continue
            dims = padded_dims_by_arg.get(arg_idx)
            if not dims:
                continue
            padded = cast("torch.Tensor", args[arg_idx])
            slices = [slice(None)] * padded.ndim
            for dim in dims:
                slices[dim] = slice(None, orig_tensor.shape[dim])
            orig_tensor.copy_(padded[tuple(slices)])

        # Slice padded output-only results back to original shapes
        if output_only_results:
            compacted_idx = 0
            for orig_pos in _output_indices:
                if orig_pos not in arg_to_tensor_pos:
                    orig = _orig_output_tensors.get(orig_pos)
                    dims = padded_dims_by_arg.get(orig_pos)
                    if (
                        orig is not None
                        and dims
                        and compacted_idx < len(output_only_results)
                    ):
                        t = output_only_results[compacted_idx]
                        if isinstance(t, torch.Tensor):
                            slices = [slice(None)] * t.ndim
                            for dim in dims:
                                slices[dim] = slice(None, orig.shape[dim])
                            output_only_results[compacted_idx] = t[tuple(slices)]
                    compacted_idx += 1

    if len(output_only_results) == 1:
        return output_only_results[0]
    return tuple(output_only_results) if output_only_results else None


def _pallas_apply_ds_padding(
    args: tuple[object, ...],
    _output_indices: list[int],
    _ds_pad_dims: list[tuple[int, int, int, int]],
) -> tuple[tuple[object, ...], dict[int, torch.Tensor]]:
    """Pad tensor args so ``pl.ds(offset, block_size)`` never reads OOB.

    ``_ds_pad_dims`` contains ``(arg_index, dim, block_size, extra_pad)``
    tuples.  The pad amount is ``(-tensor.shape[dim]) % block_size +
    extra_pad``, where *extra_pad* accounts for non-zero loop begins.

    Returns the padded args tuple and a dict mapping output arg indices
    to their original (unpadded) tensors for post-call copy-back.
    """
    args_list = list(args)
    orig_output_tensors: dict[int, torch.Tensor] = {}
    output_set = set(_output_indices)
    for arg_idx, dim, block_size, extra_pad in _ds_pad_dims:
        a = args_list[arg_idx]
        if not isinstance(a, torch.Tensor):
            continue
        pad_amount = (-a.shape[dim]) % block_size + extra_pad
        if pad_amount == 0:
            continue
        if arg_idx in output_set and arg_idx not in orig_output_tensors:
            orig_output_tensors[arg_idx] = a
        pad_widths = [0] * (2 * a.ndim)
        pad_widths[2 * (a.ndim - 1 - dim) + 1] = pad_amount
        args_list[arg_idx] = torch.nn.functional.pad(a, pad_widths)
    return tuple(args_list), orig_output_tensors


def default_pallas_launcher(
    pallas_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    _output_indices: list[int] | None = None,
    _inplace_indices: list[int] | None = None,
    _block_spec_info: _BlockSpecInfo | None = None,
    _pallas_atomic_infos: _PallasAtomicInfos | None = None,
    _smem_arg_indices: list[int] | None = None,
    _ds_pad_dims: list[tuple[int, int, int, int]] | None = None,
    _pallas_interpret: bool | None = None,
    **kwargs: object,
) -> object:
    """Default launcher for Pallas kernels on TPU (or CPU with interpret=True).

    Uses ``JaxCallable`` from ``torch_tpu`` to compile and run the Pallas
    kernel on TPU.  When ``torch_tpu`` is not available (interpret mode),
    falls back to direct torch<->JAX conversion.  Output tensors are donated
    via ``input_output_aliases`` so the kernel writes directly into their
    buffers (zero-copy on TPU).

    Output-only tensors (in ``_output_indices`` but not in ``_inplace_indices``)
    are excluded from pallas_call inputs to save VMEM.  Their results are
    returned as torch tensors.
    """
    from .settings import is_pallas_interpret

    interpret = (
        _pallas_interpret if _pallas_interpret is not None else is_pallas_interpret()
    )
    if interpret:
        _ensure_cpu_tpu_info()

    if _output_indices is None:
        _output_indices = []

    _orig_output_tensors: dict[int, torch.Tensor] | None = None
    if _ds_pad_dims:
        args, _orig_output_tensors = _pallas_apply_ds_padding(
            args, _output_indices, _ds_pad_dims
        )

    _pallas_check_dtypes(args)

    cache = getattr(pallas_kernel, "_pallas_cache", None)
    if cache is not None and cache[0] == grid:
        _, jax_callable, tensor_arg_indices, arg_to_tensor_pos = cache
    else:
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        import jax.numpy as jnp

        (
            tensor_arg_indices,
            output_only_indices,
            non_tensor_args,
            n_tensor_inputs,
            arg_to_tensor_pos,
            inplace_positions,
            out_shapes,
            pallas_aliases,
        ) = _pallas_prepare_args(
            args, _output_indices, _inplace_indices, interpret=interpret
        )

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
            output_only_indices,
        )
        atomic_copy_guards, dimension_semantics = _pallas_atomic_contributor_plan(
            grid,
            tensor_arg_indices,
            output_only_indices,
            _output_indices,
            _inplace_indices,
            _block_spec_info,
            _pallas_atomic_infos,
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
            atomic_copy_guards=atomic_copy_guards,
        )

        out_shape_arg = out_shapes if len(out_shapes) > 1 else out_shapes[0]

        estimated_vmem = _estimate_pallas_vmem_bytes(
            pl,
            pltpu,
            in_specs,
            out_specs,
            None,
            args,
            tensor_arg_indices,
            _output_indices,
            pallas_aliases,
        )
        vmem_limit_bytes = _get_vmem_limit_bytes(pltpu)
        if estimated_vmem > vmem_limit_bytes:
            raise RuntimeError(
                f"XLA:TPU compile permanent error. Ran out of memory in memory space vmem. "
                f"Estimated {estimated_vmem / 1e6:.2f}MB exceeds {vmem_limit_bytes / 1e6:.2f}MB vmem capacity."
            )

        pallas_call_kwargs: dict[str, object] = {
            "out_shape": out_shape_arg,
            "grid": grid,
        }
        if any(dim != "parallel" for dim in dimension_semantics):
            pallas_call_kwargs["compiler_params"] = pltpu.CompilerParams(  # pyrefly: ignore[bad-instantiation]
                dimension_semantics=cast("Any", dimension_semantics),
            )
        if interpret:
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
            call_aliases=pallas_aliases,
            interpret=interpret,
        )

    return _pallas_invoke_and_return(
        jax_callable,
        args,
        tensor_arg_indices,
        arg_to_tensor_pos,
        _output_indices,
        _ds_pad_dims,
        _orig_output_tensors,
    )


def default_pallas_pipeline_launcher(
    pallas_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    _output_indices: list[int] | None = None,
    _inplace_indices: list[int] | None = None,
    _block_spec_info: _BlockSpecInfo | None = None,
    _pallas_atomic_infos: _PallasAtomicInfos | None = None,
    _scratch_shapes: list[tuple[tuple[int, ...], str]] | None = None,
    _pipeline_arg_indices: list[int] | None = None,
    _ds_pad_dims: list[tuple[int, int, int, int]] | None = None,
    _smem_arg_indices: list[int] | None = None,
    _pallas_interpret: bool | None = None,
    **kwargs: object,
) -> object:
    """Launcher for Pallas kernels using PrefetchScalarGridSpec with scratch memory.

    Used when ``pallas_loop_type='emit_pipeline'``.  Pipeline-body tensors
    (listed in ``_pipeline_arg_indices``) use HBM refs; all other tensors
    get proper BlockSpecs for automatic VMEM prefetch.
    """
    from .settings import is_pallas_interpret

    interpret = (
        _pallas_interpret if _pallas_interpret is not None else is_pallas_interpret()
    )
    if interpret:
        _ensure_cpu_tpu_info()

    if _output_indices is None:
        _output_indices = []
    if _scratch_shapes is None:
        _scratch_shapes = []

    _orig_output_tensors: dict[int, torch.Tensor] | None = None
    if _ds_pad_dims:
        args, _orig_output_tensors = _pallas_apply_ds_padding(
            args, _output_indices, _ds_pad_dims
        )

    _pallas_check_dtypes(args)

    cache = getattr(pallas_kernel, "_pallas_pipeline_cache", None)
    if cache is not None and cache[0] == grid:
        _, jax_callable, tensor_arg_indices, arg_to_tensor_pos = cache
    else:
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        import jax.numpy as jnp

        (
            tensor_arg_indices,
            output_only_indices,
            non_tensor_args,
            n_tensor_inputs,
            arg_to_tensor_pos,
            inplace_positions,
            out_shapes,
            pallas_aliases,
        ) = _pallas_prepare_args(
            args, _output_indices, _inplace_indices, interpret=interpret
        )

        # Build scratch shapes for VMEM
        _jnp_dtype_map = _pallas_jnp_dtype_map()
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

        assert _block_spec_info is not None, (
            "emit_pipeline launcher requires _block_spec_info from codegen"
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
            output_only_indices,
            smem_arg_indices=_smem_arg_indices,
        )
        atomic_copy_guards, dimension_semantics = _pallas_atomic_contributor_plan(
            grid,
            tensor_arg_indices,
            output_only_indices,
            _output_indices,
            _inplace_indices,
            _block_spec_info,
            _pallas_atomic_infos,
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
            _smem_arg_indices=_smem_arg_indices,
            atomic_copy_guards=atomic_copy_guards,
        )

        out_shape_arg = out_shapes if len(out_shapes) > 1 else out_shapes[0]

        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=in_specs_list,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
            grid=grid,
        )

        estimated_vmem = _estimate_pallas_vmem_bytes(
            pl,
            pltpu,
            in_specs_list,
            out_specs,
            scratch_shapes,
            args,
            tensor_arg_indices,
            _output_indices,
            pallas_aliases,
        )
        vmem_limit_bytes = _get_vmem_limit_bytes(pltpu)
        if estimated_vmem > vmem_limit_bytes:
            raise RuntimeError(
                f"XLA:TPU compile permanent error. Ran out of memory in memory space vmem. "
                f"Estimated {estimated_vmem / 1e6:.2f}MB exceeds {vmem_limit_bytes / 1e6:.2f}MB vmem capacity."
            )

        pallas_call_kwargs: dict[str, object] = {
            "out_shape": out_shape_arg,
            "grid_spec": grid_spec,
            "compiler_params": pltpu.CompilerParams(  # pyrefly: ignore[bad-instantiation]
                dimension_semantics=cast("Any", dimension_semantics),
            ),
        }
        if interpret:
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
            call_aliases=pallas_aliases,
            trace_key_suffix="_pipeline",
            interpret=interpret,
        )

    return _pallas_invoke_and_return(
        jax_callable,
        args,
        tensor_arg_indices,
        arg_to_tensor_pos,
        _output_indices,
        _ds_pad_dims,
        _orig_output_tensors,
    )


def default_pallas_fori_launcher(
    pallas_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    _output_indices: list[int] | None = None,
    _inplace_indices: list[int] | None = None,
    _block_spec_info: _BlockSpecInfo | None = None,
    _pallas_atomic_infos: _PallasAtomicInfos | None = None,
    _scratch_shapes: list[tuple[tuple[int, ...], str | None, str]] | None = None,
    _ds_pad_dims: list[tuple[int, int, int, int]] | None = None,
    _smem_arg_indices: list[int] | None = None,
    _pallas_interpret: bool | None = None,
    **kwargs: object,
) -> object:
    """Launcher for Pallas kernels using fori_loop with manual DMA.

    Used when ``pallas_loop_type="fori_loop"``.  Passes all tensors as
    ``memory_space=pl.ANY`` (HBM refs) and adds scratch buffers as
    ``pltpu.VMEM`` shapes plus ``pltpu.SemaphoreType.DMA`` for async copies.
    The kernel uses ``jax.lax.fori_loop`` with ``pltpu.make_async_copy``
    internally for DMA control.
    """
    from .settings import is_pallas_interpret

    interpret = (
        _pallas_interpret if _pallas_interpret is not None else is_pallas_interpret()
    )
    if interpret:
        _ensure_cpu_tpu_info()

    if _output_indices is None:
        _output_indices = []
    if _scratch_shapes is None:
        _scratch_shapes = []

    _orig_output_tensors: dict[int, torch.Tensor] | None = None
    if _ds_pad_dims:
        args, _orig_output_tensors = _pallas_apply_ds_padding(
            args, _output_indices, _ds_pad_dims
        )

    _pallas_check_dtypes(args)

    cache = getattr(pallas_kernel, "_pallas_fori_cache", None)
    if cache is not None and cache[0] == grid:
        _, jax_callable, tensor_arg_indices, arg_to_tensor_pos = cache
    else:
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        import jax.numpy as jnp

        (
            tensor_arg_indices,
            output_only_indices,
            non_tensor_args,
            n_tensor_inputs,
            arg_to_tensor_pos,
            inplace_positions,
            out_shapes,
            pallas_aliases,
        ) = _pallas_prepare_args(
            args, _output_indices, _inplace_indices, interpret=interpret
        )

        # Build scratch shapes: VMEM buffers + DMA semaphores
        _jnp_dtype_map = _pallas_jnp_dtype_map()
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
        assert _block_spec_info is not None, (
            "fori_loop launcher requires _block_spec_info from codegen"
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
            _fori_pipeline_indices,  # type: ignore[arg-type]
            output_only_indices,
            smem_arg_indices=_smem_arg_indices,
        )
        atomic_copy_guards, dimension_semantics = _pallas_atomic_contributor_plan(
            grid,
            tensor_arg_indices,
            output_only_indices,
            _output_indices,
            _inplace_indices,
            _block_spec_info,
            _pallas_atomic_infos,
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
            _smem_arg_indices=_smem_arg_indices,
            atomic_copy_guards=atomic_copy_guards,
        )

        out_shape_arg = out_shapes if len(out_shapes) > 1 else out_shapes[0]

        grid_spec = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=in_specs_list,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
            grid=grid,
        )

        estimated_vmem = _estimate_pallas_vmem_bytes(
            pl,
            pltpu,
            in_specs_list,
            out_specs,
            scratch_shapes,
            args,
            tensor_arg_indices,
            _output_indices,
            pallas_aliases,
        )
        vmem_limit_bytes = _get_vmem_limit_bytes(pltpu)
        if estimated_vmem > vmem_limit_bytes:
            raise RuntimeError(
                f"XLA:TPU compile permanent error. Ran out of memory in memory space vmem. "
                f"Estimated {estimated_vmem / 1e6:.2f}MB exceeds {vmem_limit_bytes / 1e6:.2f}MB vmem capacity."
            )

        pallas_call_kwargs: dict[str, object] = {
            "out_shape": out_shape_arg,
            "grid_spec": grid_spec,
            "compiler_params": pltpu.CompilerParams(  # pyrefly: ignore[bad-instantiation]
                dimension_semantics=cast("Any", dimension_semantics),
            ),
        }
        if interpret:
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
            call_aliases=pallas_aliases,
            trace_key_suffix="_fori",
            interpret=interpret,
        )

    return _pallas_invoke_and_return(
        jax_callable,
        args,
        tensor_arg_indices,
        arg_to_tensor_pos,
        _output_indices,
        _ds_pad_dims,
        _orig_output_tensors,
    )


def _torch_to_jax(t: torch.Tensor) -> object:
    """Convert a torch.Tensor to a JAX array via DLPack (for interpret mode on CPU)."""
    import jax.numpy as jnp

    return jnp.from_dlpack(t.detach().cpu())


def _jax_to_torch(
    arr: object, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Convert a JAX array back to a torch.Tensor via DLPack (for interpret mode on CPU)."""
    return torch.from_dlpack(arr).to(dtype=dtype, device=device)


_TORCH_DTYPE_TO_CUTLASS: dict[torch.dtype, object] | None = None


def _torch_dtype_to_cutlass(dtype: torch.dtype) -> object:
    global _TORCH_DTYPE_TO_CUTLASS
    mapping: dict[torch.dtype, object] | None = _TORCH_DTYPE_TO_CUTLASS
    if mapping is None:
        _patch_cutlass_jit_shutdown_unload()
        import cutlass

        mapping = {
            torch.float16: cutlass.Float16,
            torch.float32: cutlass.Float32,
            torch.float64: cutlass.Float64,
            torch.bfloat16: cutlass.BFloat16,
            torch.float8_e4m3fn: cutlass.Float8E4M3FN,
            torch.float8_e5m2: cutlass.Float8E5M2,
            torch.float4_e2m1fn_x2: cutlass.Uint8,
            # CuTe does not support i1 global-memory tensors; torch.bool is
            # stored as one byte, so pass bool tensor pointers as uint8 and
            # let load lowering convert nonzero bytes back to cutlass.Boolean
            # registers.
            torch.bool: cutlass.Uint8,
            torch.int8: cutlass.Int8,
            torch.int16: cutlass.Int16,
            torch.int32: cutlass.Int32,
            torch.int64: cutlass.Int64,
            torch.uint8: cutlass.Uint8,
            torch.uint32: cutlass.Uint32,
            torch.uint64: cutlass.Int64,
        }
        _TORCH_DTYPE_TO_CUTLASS = mapping
    cutlass_dtype = mapping.get(dtype)
    if cutlass_dtype is None:
        raise exc.BackendUnsupported("cute", f"dtype: {dtype}")
    return cutlass_dtype


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
        "float": "cutlass.Float32",
    }
    return mapping[kind]


def _cute_kernel_param_is_constexpr(cute_kernel: object) -> tuple[bool, ...]:
    """Return per-parameter Constexpr flags for a ``@cute.kernel``.

    Cached on the kernel object to avoid repeated signature inspection.
    The newer cutlass DSL (>=4.5) enforces region isolation: a runtime scalar
    passed through the wrapper cannot satisfy a kernel parameter declared as
    ``cutlass.Constexpr``.  When the wrapper sees a Constexpr-typed kernel
    parameter, it must propagate the value as a Constexpr (i.e., baked into
    the compiled wrapper) rather than as a runtime ``cutlass.Int64``.
    """
    cached = getattr(cast("Any", cute_kernel), "_helion_cute_param_constexpr", None)
    if cached is not None:
        return cast("tuple[bool, ...]", cached)
    import cutlass

    try:
        sig = inspect.signature(cute_kernel)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        flags: tuple[bool, ...] = ()
    else:
        from typing import get_origin
        from typing import get_type_hints

        # Helion-emitted kernels use ``from __future__ import annotations`` so
        # ``param.annotation`` is the source string. ``get_type_hints`` resolves
        # those strings against the function's globals (which include
        # ``cutlass``).
        try:
            hints = get_type_hints(cute_kernel)  # type: ignore[arg-type]
        except Exception:
            hints = {}
        flags_list: list[bool] = []
        for name, param in sig.parameters.items():
            ann = hints.get(name, param.annotation)
            is_constexpr = ann is cutlass.Constexpr or get_origin(ann) is (
                cutlass.Constexpr
            )
            flags_list.append(is_constexpr)
        flags = tuple(flags_list)
    with suppress(AttributeError, TypeError):
        cast("Any", cute_kernel)._helion_cute_param_constexpr = flags
    return flags


def _append_cute_wrapper_plan(
    body: list[str],
    call_args: list[str],
    plan: dict[str, object],
) -> None:
    def plan_int(key: str, default: int | None = None) -> int:
        value = plan.get(key, default) if default is not None else plan[key]
        assert isinstance(value, int)
        return value

    def plan_optional_int(key: str) -> int | None:
        value = plan.get(key)
        assert value is None or isinstance(value, int)
        return value

    def require_positive_int(value: int | None, name: str) -> int:
        assert type(value) is int, name
        assert value > 0, name
        return value

    def append_tcgen05_epilogue_tma_wrapper(
        *,
        tensor_idx: int,
        bm: int,
        bn: int,
        stage_count: int,
        dtype: str,
        kernel_args: list[str],
        copy_op: str,
        epi_tile_m: int | None = None,
        epi_tile_n: int | None = None,
        d_store_box_n: int | None = None,
    ) -> None:
        assert len(kernel_args) == 2
        explicit_epi_tile = any(
            value is not None for value in (epi_tile_m, epi_tile_n, d_store_box_n)
        )
        if explicit_epi_tile:
            checked_epi_tile_m = require_positive_int(epi_tile_m, "epi_tile_m")
            checked_epi_tile_n = require_positive_int(epi_tile_n, "epi_tile_n")
            checked_d_store_box_n = require_positive_int(d_store_box_n, "d_store_box_n")
            assert checked_epi_tile_n == checked_d_store_box_n
            epi_tile_expr = tcgen05_explicit_d_store_tile_expr(
                checked_epi_tile_m, checked_d_store_box_n
            )
        else:
            epi_tile_expr = tcgen05_default_epilogue_tile_expr(
                bm,
                bn,
                dtype,
                c_layout="cutlass.utils.layout.LayoutEnum.ROW_MAJOR",
            )
        tma_atom, tma_tensor = kernel_args
        epi_tile = f"{tma_atom}_epi_tile"
        smem_layout = f"{tma_atom}_smem_layout"
        cta_v_layout = f"{tma_atom}_cta_v_layout"
        # Keep these layout arguments in sync with the device-side
        # ``make_smem_layout_epi`` calls; the wrapper's TMA atom and the kernel's
        # SMEM staging must slice the same epilogue tile shape.
        body.extend(
            (
                f"    {epi_tile} = {epi_tile_expr}",
                (
                    f"    {smem_layout} = cutlass.utils.blackwell_helpers."
                    "make_smem_layout_epi("
                    f"{dtype}, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, "
                    f"{epi_tile}, {stage_count})"
                ),
                (
                    f"    {cta_v_layout} = cute.composition("
                    f"cute.make_identity_layout(arg{tensor_idx}.shape), {epi_tile})"
                ),
                (
                    f"    {tma_atom}, {tma_tensor} = "
                    "cute.nvgpu.cpasync.make_tiled_tma_atom("
                    f"{copy_op}, "
                    f"arg{tensor_idx}, cute.slice_({smem_layout}, (None, None, 0)), "
                    f"{cta_v_layout})"
                ),
            )
        )
        call_args.extend(kernel_args)

    kind = plan["kind"]
    if kind == "tcgen05_d_tma":
        d_idx = plan_int("d_idx")
        bm = plan_int("bm")
        bn = plan_int("bn")
        c_stage_count = plan_int("c_stage_count")
        output_dtype = str(plan["output_dtype"])
        kernel_args = [str(arg) for arg in cast("list[object]", plan["kernel_args"])]
        append_tcgen05_epilogue_tma_wrapper(
            tensor_idx=d_idx,
            bm=bm,
            bn=bn,
            stage_count=c_stage_count,
            dtype=output_dtype,
            kernel_args=kernel_args,
            copy_op="cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()",
            epi_tile_m=plan_optional_int("epi_tile_m"),
            epi_tile_n=plan_optional_int("epi_tile_n"),
            d_store_box_n=plan_optional_int("d_store_box_n"),
        )
        return
    if kind == "tcgen05_aux_tma":
        c_idx = plan_int("c_idx")
        bm = plan_int("bm")
        bn = plan_int("bn")
        stage_count = plan_int("stage_count")
        input_dtype = str(plan["input_dtype"])
        kernel_args = [str(arg) for arg in cast("list[object]", plan["kernel_args"])]
        append_tcgen05_epilogue_tma_wrapper(
            tensor_idx=c_idx,
            bm=bm,
            bn=bn,
            stage_count=stage_count,
            dtype=input_dtype,
            kernel_args=kernel_args,
            copy_op="cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()",
        )
        return
    if kind != "tcgen05_ab_tma":
        raise exc.BackendUnsupported("cute", f"wrapper plan kind: {kind}")

    lhs_idx_key = "lhs_idx" if "lhs_idx" in plan else "lhsidx"
    rhs_idx_key = "rhs_idx" if "rhs_idx" in plan else "rhsidx"
    lhs_idx = plan_int(lhs_idx_key)
    rhs_idx = plan_int(rhs_idx_key)
    bm = plan_int("bm")
    bn = plan_int("bn")
    bk = plan_int("bk")
    cluster_m = plan_int("cluster_m", 1)
    cluster_n = plan_int("cluster_n", 1)
    input_dtype = str(plan["input_dtype"])
    acc_dtype = str(plan["acc_dtype"])
    ab_stage_count = plan_int("ab_stage_count", 2)
    # Optional ``smem_swizzle_*`` overrides recorded by the device-side
    # codegen when the user opts into a non-default A/B SMEM atom
    # swizzle. When absent the wrapper emits the legacy
    # ``make_smem_layout_a/b`` calls. The no-override wrapper markers
    # are covered by the focused tcgen05 SMEM-swizzle codegen test.
    smem_swizzle_a_raw = plan.get("smem_swizzle_a")
    smem_swizzle_b_raw = plan.get("smem_swizzle_b")
    smem_swizzle_a: int | None = (
        int(smem_swizzle_a_raw) if isinstance(smem_swizzle_a_raw, int) else None
    )
    smem_swizzle_b: int | None = (
        int(smem_swizzle_b_raw) if isinstance(smem_swizzle_b_raw, int) else None
    )
    kernel_args = [str(arg) for arg in cast("list[object]", plan["kernel_args"])]
    assert len(kernel_args) == 4
    tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b = kernel_args

    # CtaGroup.TWO is selected when ``cluster_m == 2 and bm == 256`` —
    # the V=2 path. ``cluster_n`` extends the cluster along the N axis
    # but does not change the V dimension. Cycle 26's
    # ``cluster_m * cluster_n == 2`` test happened to work for
    # cluster_m=2 cluster_n=1 but rejects the canonical Quack-best
    # cluster_m=2 cluster_n=2 4-CTA cluster (product=4). Use
    # ``cluster_m == 2`` directly so cluster_n=2 keeps CtaGroup.TWO.
    cta_group = (
        "cute.nvgpu.tcgen05.CtaGroup.TWO"
        if cluster_m == 2 and bm == 256
        else "cute.nvgpu.tcgen05.CtaGroup.ONE"
    )
    cluster_shape = f"({cluster_m}, {cluster_n}, 1)"
    tiled_mma = f"{tma_atom_a}_tiled_mma"
    cluster_layout_vmnk = f"{tma_atom_a}_cluster_layout_vmnk"
    smem_a_layout = f"{tma_atom_a}_smem_layout"
    smem_b_layout = f"{tma_atom_b}_smem_layout"
    rhs_tma = f"{tma_atom_b}_rhs_tma"
    smem_a_layout_expr = tcgen05_smem_layout_expr(
        tiled_mma=tiled_mma,
        bm=bm,
        bn=bn,
        bk=bk,
        dtype_str=input_dtype,
        num_stages=ab_stage_count,
        operand="a",
        swizzle_override=smem_swizzle_a,
    )
    smem_b_layout_expr = tcgen05_smem_layout_expr(
        tiled_mma=tiled_mma,
        bm=bm,
        bn=bn,
        bk=bk,
        dtype_str=input_dtype,
        num_stages=ab_stage_count,
        operand="b",
        swizzle_override=smem_swizzle_b,
    )
    body.extend(
        (
            (
                f"    {tiled_mma} = cutlass.utils.blackwell_helpers.make_trivial_tiled_mma("
                f"{input_dtype}, "
                "cute.nvgpu.tcgen05.OperandMajorMode.K, "
                "cute.nvgpu.tcgen05.OperandMajorMode.MN, "
                f"{acc_dtype}, "
                f"{cta_group}, "
                f"({bm}, {bn}), "
                "cute.nvgpu.tcgen05.OperandSource.SMEM)"
            ),
            (
                f"    {cluster_layout_vmnk} = cute.tiled_divide("
                f"cute.make_layout({cluster_shape}), ({tiled_mma}.thr_id.shape,))"
            ),
            f"    {smem_a_layout} = {smem_a_layout_expr}",
            f"    {smem_b_layout} = {smem_b_layout_expr}",
            (
                f"    {rhs_tma} = cute.make_tensor("
                f"arg{rhs_idx}.iterator, "
                "layout=cute.make_layout("
                f"(arg{rhs_idx}_shape1, arg{rhs_idx}_shape0), "
                f"stride=(arg{rhs_idx}_stride1, arg{rhs_idx}_stride0)))"
            ),
            f"    {rhs_tma}.mark_layout_dynamic(leading_dim=0)",
            # ``make_tiled_tma_atom_A`` vs ``_B`` asymmetry:
            # - ``_B`` always passes ``cluster_layout_vmnk.shape`` as
            #   its trailing arg (CuTe's signature for B requires the
            #   cluster shape; the cluster_m=1 cluster_n=1 case still
            #   passes the 1×1×1 shape harmlessly).
            # - ``_A`` only adds the same trailing arg when
            #   ``cluster_n > 1``. For the validated cluster_n=1
            #   paths, A's atom is constructed without the cluster
            #   shape while B still receives it. The asymmetry is
            #   intentional: A only needs the cluster shape when N
            #   multicast is active (cluster_n>1). The cluster_n=1
            #   form is pinned by
            #   ``test_tcgen05_role_local_monolithic_codegen_markers``.
            (
                f"    {tma_atom_a}, {tma_tensor_a} = cute.nvgpu.make_tiled_tma_atom_A("
                "cutlass.utils.blackwell_helpers.cluster_shape_to_tma_atom_A("
                f"{cluster_shape}, {tiled_mma}.thr_id), "
                f"arg{lhs_idx}, "
                f"cute.slice_({smem_a_layout}, (None, None, None, 0)), "
                f"({bm}, {bn}, {bk}), {tiled_mma}"
                + (f", {cluster_layout_vmnk}.shape" if cluster_n > 1 else "")
                + ")"
            ),
            # See the asymmetry comment above ``make_tiled_tma_atom_A``
            # for why ``_B`` always passes the cluster shape and ``_A``
            # only does at cluster_n>1.
            (
                f"    {tma_atom_b}, {tma_tensor_b} = cute.nvgpu.make_tiled_tma_atom_B("
                "cutlass.utils.blackwell_helpers.cluster_shape_to_tma_atom_B("
                f"{cluster_shape}, {tiled_mma}.thr_id), "
                f"{rhs_tma}, "
                f"cute.slice_({smem_b_layout}, (None, None, None, 0)), "
                f"({bm}, {bn}, {bk}), {tiled_mma}, {cluster_layout_vmnk}.shape)"
            ),
        )
    )
    call_args.extend(kernel_args)


def _cute_cluster_shape_from_wrapper_plans(
    wrapper_plans: list[dict[str, object]],
) -> tuple[int, int, int] | None:
    cluster_m = 1
    cluster_n = 1
    for plan in wrapper_plans:
        if plan.get("kind") != "tcgen05_ab_tma":
            continue
        plan_cluster_m = plan.get("cluster_m", 1)
        plan_cluster_n = plan.get("cluster_n", 1)
        assert isinstance(plan_cluster_m, int)
        assert isinstance(plan_cluster_n, int)
        cluster_m = max(cluster_m, plan_cluster_m)
        cluster_n = max(cluster_n, plan_cluster_n)
    if cluster_m * cluster_n <= 1:
        return None
    return (cluster_m, cluster_n, 1)


def _cute_cluster_shape(
    cute_kernel: object, wrapper_plans: list[dict[str, object]]
) -> tuple[int, int, int] | None:
    explicit_cluster_shape = getattr(
        cast("Any", cute_kernel), "_helion_cute_cluster_shape", None
    )
    if explicit_cluster_shape is not None:
        if (
            isinstance(explicit_cluster_shape, tuple)
            and len(explicit_cluster_shape) == 3
            and all(isinstance(dim, int) for dim in explicit_cluster_shape)
        ):
            return cast("tuple[int, int, int]", explicit_cluster_shape)
        raise exc.BackendUnsupported(
            "cute",
            f"invalid _helion_cute_cluster_shape: {explicit_cluster_shape!r}",
        )
    return _cute_cluster_shape_from_wrapper_plans(wrapper_plans)


def _create_cute_wrapper(
    cute_kernel: object,
    schema_key: tuple[tuple[object, ...], ...],
    block: tuple[int, int, int],
) -> object:
    _patch_cutlass_jit_shutdown_unload()
    import cutlass
    import cutlass.cute as cute

    cuda_driver = importlib.import_module("cuda.bindings.driver")
    kernel_name = getattr(cast("Any", cute_kernel), "__name__", "cute_kernel")
    kernel_tag = f"{kernel_name}_{id(cute_kernel):x}"
    func_name = f"_helion_cute_launch_{kernel_tag}"
    params: list[str] = []
    body: list[str] = []
    call_args: list[str] = []

    for i, entry in enumerate(schema_key):
        kind = entry[0]
        if kind == "tensor":
            ptr_name = f"arg{i}_ptr"
            params.append(f"{ptr_name}: cute.Pointer")
            if len(entry) == 5:
                # ("tensor", dtype, rank, sizes, strides) — baked layout.
                # Wrapper plans (matmul TMA) also reference
                # ``arg{i}_shape{d}`` / ``arg{i}_stride{d}`` names, so we
                # bind those names to their literal values in the wrapper
                # body before constructing the tensor.
                (_, _dtype, rank, sizes_t, strides_t) = entry
                assert isinstance(rank, int)
                assert isinstance(sizes_t, tuple) and len(sizes_t) == rank
                assert isinstance(strides_t, tuple) and len(strides_t) == rank
                shape_literals = [repr(int(s)) for s in sizes_t]
                stride_literals = [repr(int(s)) for s in strides_t]
                for d, lit in enumerate(shape_literals):
                    body.append(f"    arg{i}_shape{d} = {lit}")
                for d, lit in enumerate(stride_literals):
                    body.append(f"    arg{i}_stride{d} = {lit}")
                shape_tuple = (
                    f"({shape_literals[0]},)"
                    if rank == 1
                    else f"({', '.join(shape_literals)})"
                )
                stride_tuple = (
                    f"({stride_literals[0]},)"
                    if rank == 1
                    else f"({', '.join(stride_literals)})"
                )
                body.append(
                    f"    arg{i} = cute.make_tensor({ptr_name}, layout=cute.make_layout({shape_tuple}, stride={stride_tuple}))"
                )
                call_args.append(f"arg{i}")
                continue
            (_, _dtype, rank) = entry
            assert isinstance(rank, int)
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

        if kind == "scalar_constexpr":
            (_, scalar_kind, _scalar_key_value, scalar_value) = entry
            assert isinstance(scalar_kind, str)
            literal = repr(scalar_value)
            body.append(f"    arg{i} = {literal}")
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
            "stream: CUstream",
        )
    )
    wrapper_plans = [
        cast("dict[str, object]", plan)
        for plan in getattr(cast("Any", cute_kernel), "_helion_cute_wrapper_plans", [])
    ]
    for plan in wrapper_plans:
        _append_cute_wrapper_plan(body, call_args, plan)
    launch_suffix = f", block={block!r}"
    cluster_shape = _cute_cluster_shape(cute_kernel, wrapper_plans)
    if cluster_shape is not None:
        launch_suffix += f", cluster={list(cluster_shape)!r}"
    # G2-H (cute_plan.md, see plan: G2-H CLC): CLC kernels need PDL
    # enabled at the host launch so ``nvvm.clusterlaunchcontrol_try_cancel``
    # returns valid responses. ``use_pdl`` is set on the per-matmul
    # wrapper plan in ``cute_mma._codegen_cute_mma`` when
    # ``Tcgen05PersistenceModel.CLC_PERSISTENT`` is active. Reading
    # from the plan rather than a kernel-level side-channel attribute
    # mirrors how ``cluster_m``/``cluster_n`` flow through this layer.
    if any(plan.get("use_pdl") for plan in wrapper_plans):
        launch_suffix += ", use_pdl=True"
    body.extend(
        (
            f"    _helion_cute_kernel_tag = {kernel_tag!r}",
            "    _kernel("
            + ", ".join(call_args)
            + f").launch(grid=(grid_x, grid_y, grid_z){launch_suffix}, stream=stream)",
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
        "CUstream": cuda_driver.CUstream,
        "_kernel": cute_kernel,
    }
    filename = f"<helion_cute_launcher:{kernel_tag}:{schema_key!r}:{block!r}>"
    linecache.cache[filename] = (
        len(source),
        None,
        [line + "\n" for line in source.splitlines()],
        filename,
    )
    exec(compile(source, filename, "exec"), namespace)
    return namespace[func_name]


# Per-``bk`` stage tuple and tensor shape sets for the TVM-FFI
# direct-entry fast launch path. The validator imports these from
# ``tcgen05_constants`` (see top of file) so the codegen-side
# direct-entry source builder and the runtime-side validator cannot
# drift out of sync.


def _target1_direct_entry_plan(
    cute_kernel: object,
) -> dict[str, object] | None:
    direct_plans = [
        cast("dict[str, object]", plan)
        for plan in getattr(
            cast("Any", cute_kernel), "_helion_cute_direct_entry_plans", []
        )
    ]
    if not direct_plans:
        return None
    if len(direct_plans) != 1:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 direct entry requires exactly one direct-entry plan",
        )
    plan = direct_plans[0]
    if plan.get("kind") != "tcgen05_target1_direct_entry":
        raise exc.BackendUnsupported(
            "cute",
            f"unsupported direct-entry plan kind: {plan.get('kind')!r}",
        )
    return plan


def _plan_int(plan: dict[str, object], key: str) -> int:
    value = plan[key]
    if not isinstance(value, int):
        raise exc.BackendUnsupported(
            "cute",
            f"invalid direct-entry plan {key}: {value!r}",
        )
    return value


def _target1_direct_entry_arg_indices(
    plan: dict[str, object],
) -> tuple[int, ...]:
    # T2 adds a 4th index for the rowvec bias tensor when the plan
    # carries ``bias_idx``. T1/T3/T4/T5 keep the 3-index lhs/rhs/output
    # signature.
    indices: tuple[int, ...] = (
        _plan_int(plan, "lhs_idx"),
        _plan_int(plan, "rhs_idx"),
        _plan_int(plan, "d_idx"),
    )
    if "bias_idx" in plan:
        indices = (*indices, _plan_int(plan, "bias_idx"))
    return indices


def _direct_entry_clustered_grid_k(device: torch.device) -> tuple[int, ...]:
    """Return the accepted ``grid[2]`` set for the clustered direct-entry launch.

    The clustered grid is ``(cluster_m, cluster_n, min(total_clusters,
    num_sms // cluster_m))`` (see the codegen at ``program_id.py``).
    Both the unconstrained ``total_clusters`` value (when ``num_sms``
    is large enough) and the cap ``num_sms // cluster_m`` are valid
    runtime values; the accept set is the union for the validated
    ``total_clusters`` envelope.

    On B200 with 148 SMs and ``cluster_m=2``: T1 yields ``min(64, 74)
    = 64``; T2/T3/T4/T5 each yield ``min(128, 74) = 74``; T6 yields
    ``min(256, 74) = 74``. The validator derives this set from the
    actual CUDA device so different Blackwell SKUs with a different
    ``num_sms`` extend automatically — on a hypothetical larger SKU
    with ``num_sms // cluster_m >= 256`` the T6 runtime ``grid[2] =
    256`` is in the accept set because
    ``TCGEN05_DIRECT_ENTRY_TOTAL_WORK_CLUSTERS`` includes 256.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        # Fall back to the B200 set; tests that mock device access never
        # hit launch-time validation.
        return (64, 74)
    sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    cap = sm_count // TCGEN05_DIRECT_ENTRY_CLUSTER_M
    accepted: set[int] = {cap}
    accepted.update(
        min(total, cap) for total in TCGEN05_DIRECT_ENTRY_TOTAL_WORK_CLUSTERS
    )
    return tuple(sorted(accepted))


def _validate_target1_direct_entry_args(
    plan: dict[str, object],
    args: tuple[object, ...],
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    compile_options: str | None,
) -> tuple[int, ...]:
    if compile_options is None or "--enable-tvm-ffi" not in compile_options.split():
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 direct entry requires --enable-tvm-ffi",
        )
    # The clustered grid is ``(cluster_m, cluster_n, K)`` for some
    # device/shape-dependent ``K``: Target 1 yields ``K = 64`` (matches
    # total work clusters at the T1 shape) and Target 4 caps at
    # ``K = num_sms // cluster_m`` (= 74 on B200). The x-linear
    # ``(128, 1, 1)`` is the legacy generated direct-entry form used
    # inside the compiler-emitted entry point. We pin the accepted
    # ``grid[2]`` set to the two validated values so a runtime grid that
    # diverges from the validated envelope is rejected loudly.
    if block != (256, 1, 1):
        raise exc.BackendUnsupported(
            "cute",
            f"tcgen05 direct entry requires block=(256, 1, 1), got {block}",
        )
    # A4 (cycle-2 review): the clustered ``grid[2]`` accept set is
    # derived from the target tensor's CUDA device's SM count via
    # ``_direct_entry_clustered_grid_k`` so different Blackwell SKUs
    # extend automatically instead of being silently rejected by a
    # B200-hardcoded literal.
    first_tensor = next(
        (a for a in args if isinstance(a, torch.Tensor)),
        None,
    )
    if first_tensor is None:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 direct entry requires at least one tensor argument",
        )
    accepted_clustered_grid_k = _direct_entry_clustered_grid_k(first_tensor.device)
    if not (
        (grid[0] == 2 and grid[1] == 1 and grid[2] in accepted_clustered_grid_k)
        or grid == (128, 1, 1)
    ):
        raise exc.BackendUnsupported(
            "cute",
            f"tcgen05 direct entry requires the validated cluster_m=2 "
            f"launch geometry (clustered grid[2] in "
            f"{accepted_clustered_grid_k!r}), got {grid}",
        )
    bm = _plan_int(plan, "bm")
    bn = _plan_int(plan, "bn")
    bk = _plan_int(plan, "bk")
    cluster_m = _plan_int(plan, "cluster_m")
    cluster_n = _plan_int(plan, "cluster_n")
    ab_stage_count = _plan_int(plan, "ab_stage_count")
    c_stage_count = _plan_int(plan, "c_stage_count")
    if (bm, bn, cluster_m, cluster_n) != (256, 256, 2, 1):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 direct entry requires bm=bn=256 cluster_m=2 cluster_n=1, "
            f"got bm={bm} bn={bn} cluster_m={cluster_m} cluster_n={cluster_n}",
        )
    if (ab_stage_count, c_stage_count) not in _DIRECT_ENTRY_STAGE_TUPLES_BY_BK.get(
        bk, ()
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 direct entry requires a validated stage tuple, got "
            f"bk={bk} (ab,c)=({ab_stage_count},{c_stage_count})",
        )
    if (
        plan.get("input_dtype") != "cutlass.BFloat16"
        or plan.get("output_dtype") != "cutlass.BFloat16"
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 direct entry requires bf16 input/output dtypes",
        )

    indices = _target1_direct_entry_arg_indices(plan)
    # T1/T3/T4/T5 keep the 3-tensor (lhs/rhs/output) signature; T2 adds
    # a 4th rank-1 trailing-axis bias tensor signalled by the plan's
    # ``bias_idx`` (= ``indices[3]``).
    has_bias = len(indices) == 4
    if not has_bias and (indices != (0, 1, 2) or len(args) != 3):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 direct entry currently supports exactly lhs/rhs/output args",
        )
    if has_bias and (indices != (0, 1, 2, 3) or len(args) != 4):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 direct entry with bias requires exactly lhs/rhs/output/bias args",
        )
    matmul_args = args[:3]
    arg_shapes: list[tuple[int, ...]] = []
    for arg in matmul_args:
        if not isinstance(arg, torch.Tensor):
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 direct entry requires tensor arguments",
            )
        _validate_cute_launcher_tensor(arg)
        if arg.dtype is not torch.bfloat16:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 direct entry requires bf16 tensor arguments",
            )
        if arg.ndim != 2 or int(arg.stride(1)) != 1:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 direct entry requires row-major rank-2 tensors",
            )
        arg_shapes.append(tuple(int(arg.size(dim)) for dim in range(arg.ndim)))
    observed_shapes = tuple(arg_shapes)
    # B1 (cycle-3 review): each direct-entry plan carries its
    # validated problem shape so the validator can dispatch on the
    # exact (M, N, K) envelope baked into the plan. T4 and T5 share
    # ``bk=128`` (and the same ``(ab,c)`` stage tuple and cluster
    # geometry), so the bk-keyed shape-set alone cannot distinguish
    # a T4-plan from a T5-plan — the plan-carried ``validated_shape``
    # is what keeps them apart at the validator boundary. The legacy
    # bk-keyed shape-set (``_DIRECT_ENTRY_SHAPE_SETS_BY_BK``) remains
    # as defense-in-depth: the per-plan check below additionally
    # asserts that the plan's ``validated_shape`` is one of the
    # bk-allowed shapes (so a future plan-construction-site bug that
    # forged an off-envelope shape into a plan still gets caught).
    validated_shape_raw = plan.get("validated_shape")
    if not (
        isinstance(validated_shape_raw, (list, tuple))
        and len(validated_shape_raw) == 3
        and all(isinstance(v, int) for v in validated_shape_raw)
    ):
        raise exc.BackendUnsupported(
            "cute",
            f"tcgen05 direct entry plan requires validated_shape, got "
            f"{validated_shape_raw!r}",
        )
    plan_m, plan_n, plan_k = (int(v) for v in validated_shape_raw)
    bk_allowed_shapes = _DIRECT_ENTRY_SHAPE_SETS_BY_BK.get(bk, ())
    expected_lhs = (plan_m, plan_k)
    expected_rhs = (plan_k, plan_n)
    expected_d = (plan_m, plan_n)
    expected_shapes = (expected_lhs, expected_rhs, expected_d)
    if expected_shapes not in bk_allowed_shapes:
        raise exc.BackendUnsupported(
            "cute",
            f"tcgen05 direct entry plan bk={bk} carries unvalidated "
            f"validated_shape={plan_m, plan_n, plan_k}; bk-allowed shape "
            f"envelopes are {bk_allowed_shapes!r}",
        )
    if observed_shapes != expected_shapes:
        raise exc.BackendUnsupported(
            "cute",
            f"tcgen05 direct entry plan validated_shape="
            f"{plan_m, plan_n, plan_k} (bk={bk}) requires shapes "
            f"{expected_shapes!r}, got {observed_shapes!r}",
        )
    if has_bias:
        # The arity check above already required ``len(args) == 4`` when
        # ``has_bias`` is set; pyrefly cannot narrow the tuple type
        # across the conditional, so index the args by ``cast`` below.
        bias_arg = cast("tuple[object, ...]", args)[3]
        if not isinstance(bias_arg, torch.Tensor):
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 direct entry bias requires a tensor argument",
            )
        _validate_cute_launcher_tensor(bias_arg)
        if bias_arg.dtype is not torch.bfloat16:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 direct entry bias requires a bf16 tensor",
            )
        # The bias tensor for T2 is rank-1 with N elements (trailing-axis
        # rowvec broadcast); stride must be 1 so the GMEM load uses
        # contiguous reads. The N-extent matches the matmul's N to keep
        # the bias broadcast aligned with the output tile columns.
        if (
            bias_arg.ndim != 1
            or int(bias_arg.stride(0)) != 1
            or int(bias_arg.size(0)) != plan_n
        ):
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 direct entry bias requires a contiguous rank-1 "
                f"bf16 tensor of shape ({plan_n},), got shape "
                f"{tuple(int(bias_arg.size(d)) for d in range(bias_arg.ndim))!r} "
                f"stride {tuple(int(bias_arg.stride(d)) for d in range(bias_arg.ndim))!r}",
            )
    return indices


def _wrapper_plans_for_direct_entry(
    cute_kernel: object,
    direct_plan: dict[str, object],
) -> list[dict[str, object]]:
    wrapper_plans = [
        cast("dict[str, object]", plan)
        for plan in getattr(cast("Any", cute_kernel), "_helion_cute_wrapper_plans", [])
    ]
    if [plan.get("kind") for plan in wrapper_plans] != [
        "tcgen05_ab_tma",
        "tcgen05_d_tma",
    ]:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 direct entry requires A/B and D TMA wrapper plans",
        )
    ab_plan, d_plan = wrapper_plans
    expected_ab_fields = {
        "lhs_idx": _plan_int(direct_plan, "lhs_idx"),
        "rhs_idx": _plan_int(direct_plan, "rhs_idx"),
        "bm": _plan_int(direct_plan, "bm"),
        "bn": _plan_int(direct_plan, "bn"),
        "bk": _plan_int(direct_plan, "bk"),
        "cluster_m": _plan_int(direct_plan, "cluster_m"),
        "cluster_n": _plan_int(direct_plan, "cluster_n"),
        "ab_stage_count": _plan_int(direct_plan, "ab_stage_count"),
        "input_dtype": direct_plan.get("input_dtype"),
        "acc_dtype": "cutlass.Float32",
    }
    expected_d_fields = {
        "d_idx": _plan_int(direct_plan, "d_idx"),
        "bm": _plan_int(direct_plan, "bm"),
        "bn": _plan_int(direct_plan, "bn"),
        "c_stage_count": _plan_int(direct_plan, "c_stage_count"),
        "output_dtype": direct_plan.get("output_dtype"),
        "epi_tile_m": 128,
        "epi_tile_n": 32,
        "d_store_box_n": 32,
    }
    for key, expected in expected_ab_fields.items():
        if ab_plan.get(key) != expected:
            raise exc.BackendUnsupported(
                "cute",
                f"tcgen05 direct entry wrapper A/B plan mismatch for {key}",
            )
    for key, expected in expected_d_fields.items():
        if d_plan.get(key) != expected:
            raise exc.BackendUnsupported(
                "cute",
                f"tcgen05 direct entry wrapper D plan mismatch for {key}",
            )
    if "smem_swizzle_a" in ab_plan or "smem_swizzle_b" in ab_plan:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 direct entry requires default A/B SMEM wrapper layouts",
        )
    if list(cast("list[object]", direct_plan["ab_kernel_args"])) != list(
        cast("list[object]", ab_plan["kernel_args"])
    ) or list(cast("list[object]", direct_plan["d_kernel_args"])) != list(
        cast("list[object]", d_plan["kernel_args"])
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 direct entry plan does not match wrapper TMA args",
        )
    return wrapper_plans


def _create_cute_direct_entry(
    cute_kernel: object,
    direct_plan: dict[str, object],
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
) -> object:
    _patch_cutlass_jit_shutdown_unload()
    import cutlass
    import cutlass.cute as cute

    wrapper_plans = _wrapper_plans_for_direct_entry(cute_kernel, direct_plan)
    body: list[str] = []
    arg_indices = _target1_direct_entry_arg_indices(direct_plan)
    # The lhs/rhs/output tensors are rank-2; the optional bias tensor
    # for T2 is rank-1 and only contributes a single shape/stride pair.
    for index in arg_indices[:3]:
        body.extend(
            (
                f"    arg{index}_shape0 = arg{index}.shape[0]",
                f"    arg{index}_shape1 = arg{index}.shape[1]",
                f"    arg{index}_stride0 = arg{index}.stride[0]",
                f"    arg{index}_stride1 = arg{index}.stride[1]",
            )
        )
    has_bias = len(arg_indices) == 4
    if has_bias:
        bias_index = arg_indices[3]
        body.extend(
            (
                f"    arg{bias_index}_shape0 = arg{bias_index}.shape[0]",
                f"    arg{bias_index}_stride0 = arg{bias_index}.stride[0]",
            )
        )
    call_args = ["arg0", "arg1", "arg2"]
    if has_bias:
        call_args.append(f"arg{arg_indices[3]}")
    for plan in wrapper_plans:
        _append_cute_wrapper_plan(body, call_args, plan)

    launch_suffix = f", block={block!r}"
    cluster_shape = _cute_cluster_shape(cute_kernel, wrapper_plans)
    if cluster_shape is not None:
        launch_suffix += f", cluster={list(cluster_shape)!r}"
    if any(plan.get("use_pdl") for plan in wrapper_plans):
        launch_suffix += ", use_pdl=True"
    body.extend(
        (
            f"    _helion_cute_kernel_tag = {getattr(cast('Any', cute_kernel), '__name__', 'cute_kernel')!r}",
            "    _kernel("
            + ", ".join(call_args)
            + f").launch(grid={grid!r}{launch_suffix})",
        )
    )

    kernel_name = getattr(cast("Any", cute_kernel), "__name__", "cute_kernel")
    func_name = f"_helion_cute_direct_entry_{kernel_name}_{id(cute_kernel):x}"
    entry_params = ", ".join(f"arg{i}" for i in range(len(arg_indices)))
    source = "\n".join(
        [
            "@cute.jit",
            f"def {func_name}({entry_params}) -> None:",
            *body,
        ]
    )
    namespace: dict[str, Any] = {
        "cutlass": cutlass,
        "cute": cute,
        "_kernel": cute_kernel,
    }
    filename = f"<helion_cute_direct_entry:{kernel_name}:{id(cute_kernel):x}>"
    linecache.cache[filename] = (
        len(source),
        None,
        [line + "\n" for line in source.splitlines()],
        filename,
    )
    exec(compile(source, filename, "exec"), namespace)
    return namespace[func_name]


class _CompiledCuteLauncher:
    """Lazily compile a Helion ``@cute.jit`` wrapper via ``cute.compile``.

    The first call uses ``cute.compile(jit_func, *args)`` to produce a compiled
    callable; subsequent calls invoke the compiled callable directly. This
    bypasses the per-launch ``@cute.jit`` argument-handling/dispatch path,
    matching Quack's pattern (see ``gemm_tvm_ffi_utils.py``). On B200 this
    collapses ~200ms of per-launch host overhead into ~0.1ms.
    """

    __slots__ = ("_compile_options", "_compiled", "_jit_func")

    def __init__(self, jit_func: object, compile_options: str | None) -> None:
        self._jit_func = jit_func
        self._compile_options = compile_options
        self._compiled: object = None

    def __call__(self, *args: object) -> object:
        compiled = self._compiled
        if compiled is None:
            import cutlass.cute as cute

            if self._compile_options is None:
                compiled = cute.compile(self._jit_func, *args)
            else:
                compiled = cute.compile(
                    self._jit_func,
                    *args,
                    options=self._compile_options,
                )
            self._compiled = compiled
        return cast("Any", compiled)(*args)


class _CompiledCuteDirectEntryLauncher:
    """Compile a Target1 direct tensor-entry wrapper with fake tensor args."""

    __slots__ = (
        "_compile_args",
        "_compile_options",
        "_compiled",
        "_jit_func",
        "_runtime_arg_indices",
    )

    def __init__(
        self,
        jit_func: object,
        compile_args: tuple[object, ...],
        runtime_arg_indices: tuple[int, ...],
        compile_options: str,
    ) -> None:
        self._jit_func = jit_func
        self._compile_args = compile_args
        self._runtime_arg_indices = runtime_arg_indices
        self._compile_options = compile_options
        self._compiled: object = None

    def __call__(self, *args: object) -> object:
        compiled = self._compiled
        if compiled is None:
            _patch_cutlass_jit_shutdown_unload()
            import cutlass.cute as cute

            compiled = cute.compile(
                self._jit_func,
                *self._compile_args,
                options=self._compile_options,
            )
            self._compiled = compiled
        launch_args = tuple(args[index] for index in self._runtime_arg_indices)
        return cast("Any", compiled)(*launch_args)


def _make_cute_direct_entry_fake_tensor(arg: torch.Tensor) -> object:
    import cutlass.cute as cute

    dtype = cast("type[Any]", _torch_dtype_to_cutlass(arg.dtype))
    assert dtype is not None
    shape = tuple(cute.sym_int() for _ in range(arg.ndim))
    leading_dim = arg.ndim - 1
    divisibility = max(1, 16 // max(1, arg.element_size()))
    stride = tuple(
        1 if dim == leading_dim else cute.sym_int64(divisibility=divisibility)
        for dim in range(arg.ndim)
    )
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=stride,
        assumed_align=16,
    )


def _get_compiled_cute_direct_entry_launcher(
    cute_kernel: object,
    direct_plan: dict[str, object],
    args: tuple[object, ...],
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    compile_options: str,
) -> object:
    runtime_arg_indices = _validate_target1_direct_entry_args(
        direct_plan,
        args,
        grid,
        block,
        compile_options,
    )
    try:
        # pyrefly: ignore [missing-attribute]
        cache = cute_kernel._helion_cute_direct_entry_launchers
    except AttributeError:
        cache = {}
        # pyrefly: ignore [missing-attribute]
        cute_kernel._helion_cute_direct_entry_launchers = cache
    wrapper_plans = tuple(
        repr(plan)
        for plan in getattr(cast("Any", cute_kernel), "_helion_cute_wrapper_plans", [])
    )
    cluster_shape = getattr(
        cast("Any", cute_kernel), "_helion_cute_cluster_shape", None
    )
    generated_direct_entry = getattr(
        cast("Any", cute_kernel), "_helion_cute_generated_direct_entry", None
    )
    # The compiler-emitted direct entry hard-codes an x-linear
    # ``grid=(128,1,1)`` launch. The validator above also admits the
    # clustered ``(2,1,64)`` form for the runtime descriptor fallback;
    # only the x-linear launch may reuse the generated direct entry.
    if generated_direct_entry is not None and grid != (128, 1, 1):
        generated_direct_entry = None
    cache_key = (
        repr(direct_plan),
        wrapper_plans,
        repr(cluster_shape),
        grid,
        block,
        compile_options,
        id(generated_direct_entry) if generated_direct_entry is not None else None,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    _ensure_cute_dsl_arch_env(args)
    jit_func = (
        generated_direct_entry
        if generated_direct_entry is not None
        else _create_cute_direct_entry(cute_kernel, direct_plan, grid, block)
    )
    compile_args = tuple(
        _make_cute_direct_entry_fake_tensor(cast("torch.Tensor", args[index]))
        for index in runtime_arg_indices
    )
    launcher = _CompiledCuteDirectEntryLauncher(
        jit_func,
        compile_args,
        runtime_arg_indices,
        compile_options,
    )
    cache[cache_key] = launcher
    return launcher


def _get_compiled_cute_launcher(
    cute_kernel: object,
    schema_key: tuple[tuple[object, ...], ...],
    block: tuple[int, int, int],
    compile_options: str | None = None,
    arch_args: tuple[object, ...] | None = None,
) -> object:
    try:
        # pyrefly: ignore [missing-attribute]
        cache = cute_kernel._helion_cute_compiled_launchers
    except AttributeError:
        cache = {}
        # pyrefly: ignore [missing-attribute]
        cute_kernel._helion_cute_compiled_launchers = cache
    wrapper_plans = tuple(
        repr(plan)
        for plan in getattr(cast("Any", cute_kernel), "_helion_cute_wrapper_plans", [])
    )
    cluster_shape = getattr(
        cast("Any", cute_kernel), "_helion_cute_cluster_shape", None
    )
    cache_key = (
        schema_key,
        block,
        wrapper_plans,
        repr(cluster_shape),
        compile_options,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    if arch_args is not None:
        _ensure_cute_dsl_arch_env(arch_args)
    jit_func = _create_cute_wrapper(cute_kernel, schema_key, block)
    launcher = _CompiledCuteLauncher(jit_func, compile_options)
    cache[cache_key] = launcher
    return launcher


_CUTE_LAUNCHER_IMPORTS: tuple[object, ...] | None = None


def _get_cute_launcher_imports() -> tuple[object, ...]:
    global _CUTE_LAUNCHER_IMPORTS
    cached = _CUTE_LAUNCHER_IMPORTS
    if cached is not None:
        return cached
    _patch_cutlass_jit_shutdown_unload()
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr
    import cutlass.torch as cutlass_torch

    cached = (cute.AddressSpace.gmem, make_ptr, cutlass_torch.current_stream)
    _CUTE_LAUNCHER_IMPORTS = cached
    return cached


# Keep the per-kernel launch-argument cache small: production kernels normally
# relaunch one or two stable tensor signatures, while autotune may probe many.
_CUTE_LAUNCH_ARG_CACHE_LIMIT = 8


def _cute_scalar_cache_value(scalar_kind: str, scalar_value: object) -> object:
    return cast("float", scalar_value).hex() if scalar_kind == "float" else scalar_value


def _validate_cute_launcher_tensor(arg: torch.Tensor) -> None:
    if arg.device.type != "cuda":
        raise exc.BackendUnsupported("cute", "launcher requires CUDA tensors")
    if arg.ndim <= 0:
        raise exc.BackendUnsupported("cute", "launcher requires tensor rank >= 1")


def _cute_launch_arg_cache_key(
    cute_kernel: object,
    args: tuple[object, ...],
    grid: tuple[int, int, int],
) -> tuple[object, ...]:
    constexpr_flags = _cute_kernel_param_is_constexpr(cute_kernel)
    key: list[object] = [grid]
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            _validate_cute_launcher_tensor(arg)
            key.append(
                (
                    "tensor",
                    arg.device.type,
                    arg.device.index,
                    str(arg.dtype),
                    arg.ndim,
                    arg.data_ptr(),
                    tuple(int(arg.size(d)) for d in range(arg.ndim)),
                    tuple(int(arg.stride(d)) for d in range(arg.ndim)),
                )
            )
            continue

        scalar_kind, scalar_value = _normalize_cute_scalar(arg)
        scalar_key_value = _cute_scalar_cache_value(scalar_kind, scalar_value)
        is_constexpr = i < len(constexpr_flags) and constexpr_flags[i]
        key.append(
            (
                "scalar_constexpr" if is_constexpr else "scalar",
                scalar_kind,
                scalar_key_value,
            )
        )
    return tuple(key)


def _build_cached_cute_schema_and_args(
    cute_kernel: object,
    args: tuple[object, ...],
    grid: tuple[int, int, int],
) -> tuple[tuple[tuple[object, ...], ...], tuple[object, ...]]:
    cache_key = _cute_launch_arg_cache_key(cute_kernel, args, grid)
    try:
        # pyrefly: ignore [missing-attribute]
        cache = cute_kernel._helion_cute_launch_arg_cache
    except AttributeError:
        cache = {}
        # pyrefly: ignore [missing-attribute]
        cute_kernel._helion_cute_launch_arg_cache = cache
    cached = cache.get(cache_key)
    if cached is not None:
        cache[cache_key] = cache.pop(cache_key)
        return cached

    built = _build_cute_schema_and_args(cute_kernel, args, grid)
    cache[cache_key] = built
    if len(cache) > _CUTE_LAUNCH_ARG_CACHE_LIMIT:
        cache.pop(next(iter(cache)))
    return built


def _build_cute_schema_and_args(
    cute_kernel: object,
    args: tuple[object, ...],
    grid: tuple[int, int, int],
    bake_tensor_shapes: bool = True,
) -> tuple[tuple[tuple[object, ...], ...], tuple[object, ...]]:
    gmem_space, make_ptr_obj, current_stream_obj = _get_cute_launcher_imports()
    make_ptr = cast("Any", make_ptr_obj)
    current_stream = cast("Any", current_stream_obj)
    constexpr_flags = _cute_kernel_param_is_constexpr(cute_kernel)
    # Kernels that emit cute MMA ops (universal matmul fallback or tcgen05
    # TMA wrapper plans) need runtime tensor layouts: the wrapper's
    # ``cute.make_tensor`` feeds into ``.mark_layout_dynamic`` (TMA path) or
    # into in-kernel arithmetic that relies on dynamic shape/stride
    # propagation (universal MMA SMEM-load guards). Baking literal shapes
    # silently miscompiles those paths.
    if bake_tensor_shapes:
        any_obj = cast("Any", cute_kernel)
        disable_bake = bool(
            getattr(any_obj, "_helion_cute_disable_bake_tensor_shapes", False)
            or getattr(any_obj, "_helion_cute_wrapper_plans", None)
        )
        if disable_bake:
            bake_tensor_shapes = False
    schema: list[tuple[object, ...]] = []
    launch_args: list[object] = []
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            _validate_cute_launcher_tensor(arg)
            ndim = arg.ndim
            if ndim <= 0:
                raise exc.BackendUnsupported(
                    "cute", "launcher requires tensor rank >= 1"
                )
            sizes_t = tuple(int(arg.size(d)) for d in range(ndim))
            strides_t = tuple(int(arg.stride(d)) for d in range(ndim))
            launch_args.append(
                make_ptr(
                    cast("Any", _torch_dtype_to_cutlass(arg.dtype)),
                    arg.data_ptr(),
                    gmem_space,
                    assumed_align=16,
                )
            )
            # ``cute.make_layout`` rejects a 0 in any shape dimension, so
            # zero-sized tensors must keep the runtime-shape path.
            if bake_tensor_shapes and all(s > 0 for s in sizes_t):
                # Bake the shape / stride tuple into the schema key.  The
                # generated wrapper substitutes literal Int values for each
                # dimension, so the CuTe DSL sees a fully static tensor
                # layout and the per-load offset arithmetic collapses to
                # constant strides — typically a 2-3x reduction in
                # ``smsp__inst_executed`` for reduction kernels where the
                # inner loop is dominated by stride multiplies.
                schema.append(("tensor", str(arg.dtype), ndim, sizes_t, strides_t))
            else:
                schema.append(("tensor", str(arg.dtype), ndim))
                launch_args.extend(sizes_t)
                launch_args.extend(strides_t)
            continue

        scalar_kind, scalar_value = _normalize_cute_scalar(arg)
        is_constexpr = i < len(constexpr_flags) and constexpr_flags[i]
        if is_constexpr:
            # Bake Constexpr values into the wrapper / cache key. cutlass DSL
            # >=4.5 fails IR verification ("value defined outside the region")
            # if a runtime scalar is fed to a kernel parameter declared as
            # ``cutlass.Constexpr``.
            schema.append(
                (
                    "scalar_constexpr",
                    scalar_kind,
                    _cute_scalar_cache_value(scalar_kind, scalar_value),
                    scalar_value,
                )
            )
        else:
            schema.append(("scalar", scalar_kind))
            launch_args.append(scalar_value)

    launch_args.extend(grid)
    launch_args.append(current_stream())
    return tuple(schema), tuple(launch_args)


_CUTE_DSL_ARCH_CACHE: dict[int, str] = {}
_CUTE_MIN_CUDA_VERSION = "13"


def _require_cuda13_for_cute() -> None:
    from .._compat import requires_cuda_version

    if not requires_cuda_version(_CUTE_MIN_CUDA_VERSION):
        raise exc.BackendUnsupported(
            "cute",
            f"requires CUDA >= {_CUTE_MIN_CUDA_VERSION} "
            f"(found torch.version.cuda={torch.version.cuda!r})",
        )


def _ensure_cute_dsl_arch_env(args: tuple[object, ...]) -> None:
    tensor_args = [arg for arg in args if isinstance(arg, torch.Tensor)]
    if tensor_args:
        device = tensor_args[0].device
        if device.type != "cuda":
            return
        device_index = device.index if device.index is not None else 0
    elif not torch.cuda.is_available():
        return
    else:
        device_index = torch.cuda.current_device()
    _require_cuda13_for_cute()
    desired = _CUTE_DSL_ARCH_CACHE.get(device_index)
    if desired is None:
        if tensor_args:
            with torch.cuda.device(tensor_args[0].device):
                major, minor = torch.cuda.get_device_capability(tensor_args[0].device)
        else:
            major, minor = torch.cuda.get_device_capability()
        # CUTLASS DSL distinguishes post-Hopper arch variants such as
        # sm_90a/sm_100a, while torch.cuda.get_device_capability() only
        # returns major/minor.
        suffix = "a" if major >= 9 else ""
        desired = f"sm_{major}{minor}{suffix}"
        _CUTE_DSL_ARCH_CACHE[device_index] = desired
    if os.environ.get("CUTE_DSL_ARCH") != desired:
        os.environ["CUTE_DSL_ARCH"] = desired


def default_cute_launcher(
    cute_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    **kwargs: object,
) -> object:
    block = kwargs.pop("block", (256, 1, 1))
    cute_compile_options = kwargs.pop("cute_compile_options", None)
    if cute_compile_options is not None and not isinstance(cute_compile_options, str):
        raise ValueError(f"Invalid CuTe compile options: {cute_compile_options!r}")
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

    args_tuple = tuple(args)
    direct_plan = _target1_direct_entry_plan(cute_kernel)
    if direct_plan is not None:
        if cute_compile_options is None:
            raise exc.BackendUnsupported(
                "cute",
                "tcgen05 direct entry requires explicit CuTe compile options",
            )
        direct_launcher = _get_compiled_cute_direct_entry_launcher(
            cute_kernel,
            direct_plan,
            args_tuple,
            grid_xyz,
            block_xyz,
            cute_compile_options,
        )
        return cast("Any", direct_launcher)(*args_tuple)

    schema_key, launch_args = _build_cached_cute_schema_and_args(
        cute_kernel, args_tuple, grid_xyz
    )
    compiled = _get_compiled_cute_launcher(
        cute_kernel,
        schema_key,
        block_xyz,
        compile_options=cute_compile_options,
        arch_args=args_tuple,
    )
    return cast("Any", compiled)(*launch_args)


def default_metal_launcher(
    metal_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    _block_dims: tuple[int, int, int] = (256, 1, 1),
    **kwargs: object,
) -> None:
    """Default launcher for Metal kernels on Apple MPS devices.

    The ``metal_kernel`` is a ``@metal_jit`` decorated function that
    translates its Python AST body to MSL and compiles it via
    ``torch.mps.compile_shader`` on each call.
    This launcher dispatches the compiled kernel with the given grid and
    threadgroup dimensions.

    Uses a 3D threadgroup dispatch model: ``_block_dims`` specifies the
    threadgroup size as ``(x, y, z)``.  The grid specifies the number of
    threadgroups per dimension.
    """
    kwargs.pop("num_warps", None)
    kwargs.pop("num_stages", None)
    if kwargs:
        raise exc.BackendUnsupported(
            "metal", f"unexpected launcher kwargs: {sorted(kwargs)}"
        )

    from .._compiler.metal.metal_launcher import set_required_threads_per_threadgroup

    set_required_threads_per_threadgroup(metal_kernel, _block_dims)
    lib, kernel_name = metal_kernel(*args)  # type: ignore[operator]

    tensor_args = [a for a in args if isinstance(a, torch.Tensor)]
    dispatch_fn = getattr(lib, kernel_name)
    bx, by, bz = _block_dims
    # Pad grid to 3D
    gx = grid[0] if len(grid) > 0 else 1
    gy = grid[1] if len(grid) > 1 else 1
    gz = grid[2] if len(grid) > 2 else 1
    total_threads = (gx * bx, gy * by, gz * bz)
    group_size = (bx, by, bz)
    dispatch_fn(*tensor_args, threads=total_threads, group_size=group_size)
