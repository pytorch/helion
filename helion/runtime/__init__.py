from __future__ import annotations

import base64
from contextlib import suppress
import contextvars
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
import linecache
import logging
import os
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

import torch

from .. import _compat as _compat  # ensure Triton compatibility patches run
from .. import exc
from .._compat import get_num_xcd as get_num_xcd
from .._compiler.cute.strategies import tcgen05_default_epilogue_tile_expr
from .._compiler.cute.strategies import tcgen05_explicit_d_store_tile_expr
from .._compiler.cute.strategies import tcgen05_smem_layout_expr
from .._utils import triton_is_available
from .config import Config as Config
from .kernel import Kernel as Kernel
from .kernel import kernel as kernel
from .settings import is_pallas_interpret as _module_is_pallas_interpret

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable

    import jax

log: logging.Logger = logging.getLogger(__name__)

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
    if device.type == "cpu":
        if not _module_is_pallas_interpret():
            raise AssertionError("TODO: implement for other devices")
        return 1
    if device.type == "tpu":
        return 1

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
        full_shape = tuple(max(s, 1) for s in tensor.shape)

        def index_map_full(*grid_args: object, _nd: int = ndim) -> tuple[object, ...]:
            # pyrefly: ignore[missing-attribute]
            return tuple(jnp.int32(0) for _ in range(_nd))

        return pl.BlockSpec(full_shape, index_map_full, memory_space=memory_space)  # type: ignore[union-attr]

    block_shape_template, grid_dims = entry
    # Clamp to >= 1: empty tensors (zero-work grids) would otherwise produce
    # 0-sized block dims, which the interpret machinery divides by.
    block_shape = tuple(
        max(min(bs, tensor.shape[d]) if bs is not None else tensor.shape[d], 1)
        for d, bs in enumerate(block_shape_template)
    )
    # Block indices past the last block are clamped, matching pallas_call's
    # window clamping (index maps may run past the end, e.g. offset reads).
    max_block_index = tuple(
        max(-(-tensor.shape[d] // bs), 1) - 1 for d, bs in enumerate(block_shape)
    )

    def _index_for_dim(
        grid_args: tuple[object, ...],
        g: int | tuple[int, int, int] | None,
        d: int,
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
        return jnp.minimum(  # pyrefly: ignore[missing-attribute]
            jnp.int32(grid_args[g]),  # pyrefly: ignore[missing-attribute]
            max_block_index[d],
        )

    def index_map(
        *grid_args: object,
        _grid_dims: tuple[int | tuple[int, int, int] | None, ...] = grid_dims,
    ) -> tuple[object, ...]:
        return tuple(_index_for_dim(grid_args, g, d) for d, g in enumerate(_grid_dims))

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


def _ordered_per_token_bytes(operands: list[tuple[tuple[int, ...], int]]) -> int:
    per_tok = 0
    for shape, itemsize in operands:
        trail = 1
        for s in shape[1:]:
            trail *= int(s)
        per_tok += trail * itemsize
    return per_tok


def compact_ordered_budget_capacity(
    operands: list[tuple[tuple[int, ...], int]],
    vmem_bytes: int,
    *,
    prep_operands: list[tuple[tuple[int, ...], int]],
) -> int:
    """VMEM budget capacity for resident ordered operands.

    ``operands`` is ``[(shape, itemsize), ...]`` for every resident ordered
    operand.  Each ``C`` token costs that per-token footprint twice because the
    ``pl.Element`` resident window is double-buffered by Pallas.  ``prep_operands``
    adds any optional persistent prep-cache copies (for today's transpose-cache
    path, one more equivalent copy).  Pass ``[]`` for a resident-only/no-prep
    reduction.
    """
    if not operands:
        return 0
    # resident window (2x double-buffer) + optional prep cache (1x per prep).
    bytes_per_token = 2 * _ordered_per_token_bytes(operands) + _ordered_per_token_bytes(
        prep_operands
    )
    return int(vmem_bytes * 0.5) // bytes_per_token


def compact_ordered_physical_window(
    operands: list[tuple[tuple[int, ...], int]],
    vmem_bytes: int,
    ordered_block: int,
    *,
    prep_operands: list[tuple[tuple[int, ...], int]],
) -> int:
    """Block-aligned physical resident window that fits the VMEM budget.

    :func:`compact_ordered_budget_capacity` gives the largest per-source length the
    VMEM budget allows (the logical bound ``C``).  The resident ``pl.Element``
    window, optional prep-cache scratch, and the refill/reduction ``pl.ds`` slices
    are all tiled by the ordered block, so the allocation must be a block multiple.
    Round the budget DOWN to a block multiple so the allocation never exceeds the
    VMEM budget; cap by the operand extent rounded UP to one block so short tensors
    still get a legal ``pl.Element(block, padding=...)`` window instead of a
    zero-sized allocation.

    Returns 0 when the budget cannot hold one ordered block.  Resident caching is
    an automatic optimization, so the compiler treats that as "inactive" and
    falls back to the streamed ordered loop.
    """
    if not operands:
        return 0
    budget_capacity = compact_ordered_budget_capacity(
        operands, vmem_bytes, prep_operands=prep_operands
    )
    block = max(int(ordered_block), 1)
    budget_physical = (budget_capacity // block) * block
    if budget_physical <= 0:
        return 0
    min_leading = min(max(1, int(shape[0])) for shape, _itemsize in operands)
    extent_physical = ((min_leading + block - 1) // block) * block
    return min(budget_physical, extent_physical)


def _compact_raise_if_range_exceeds_window(
    args: tuple[object, ...],
    ordered_aligned_arg_indices: list[int] | None,
    ordered_offset_arg_index: int,
    active_mask_arg_index: int,
    ordered_window: int,
) -> None:
    """Raise if any active ordered range exceeds the resident window.

    Resident caching holds each source's ordered operand in a compile-time-sized
    VMEM window.  ``ordered_window`` is the exact block-aligned physical extent
    computed once during compile setup and threaded through the launcher; a range
    longer than it would over-read the window.

    ``ordered_aligned_arg_indices`` non-empty means the resident window is active.
    When it is active we MUST be able to bound-check, so a missing/ambiguous
    offset index raises rather than silently returning.  The
    ordered offset supplies ordered lengths; the active-mask offset supplies compact
    lengths, so sources with no compact work are ignored because they produce no
    worklist item and never refill the cache.

    Best-effort magnitude: reached with materialized offsets only on the
    concrete/eager (torch) launch path; under ``jax.jit`` the offsets are tracers
    and the caller guarantees the bound (a future change that sizes the window from
    a caller-provided max per-source length would remove this caveat).
    """
    if not ordered_aligned_arg_indices:
        return  # resident caching inactive (no resident window) -> nothing to guard
    if ordered_window <= 0:
        raise RuntimeError(
            "compact_worklist resident caching: the resident window is active but "
            f"the compiled ordered window is invalid ({ordered_window})."
        )
    if ordered_offset_arg_index < 0 or active_mask_arg_index < 0:
        raise RuntimeError(
            "compact_worklist resident caching: the resident window is active but the "
            "ordered reduction bound or compact active-owner mask is not a "
            "checkable single-offsets (offsets[i+1]-offsets[i]) pattern, so "
            "per-source length cannot be verified against the window."
        )
    offsets = cast("Any", args[ordered_offset_arg_index])
    active_offsets = cast("Any", args[active_mask_arg_index])
    if len(offsets) < 2:  # 0 owners -> no reduction ranges to check
        return
    if len(active_offsets) != len(offsets):
        raise RuntimeError(
            "compact_worklist resident caching: ordered and compact offset arrays have "
            "different owner counts, so the active-owner guard cannot be evaluated."
        )
    ordered_lens = offsets[1:] - offsets[:-1]
    compact_lens = active_offsets[1:] - active_offsets[:-1]
    active = compact_lens > 0
    if not bool(active.any()):
        return
    max_len = int(ordered_lens[active].max())
    if max_len > ordered_window:
        raise RuntimeError(
            f"compact_worklist resident caching: a per-source reduction length "
            f"({max_len}) exceeds the resident window ({ordered_window}, "
            f"VMEM-derived and fixed at compile time), so the range-keyed cache "
            f"would be over-read. "
            f"Reduce the maximum per-source length below the window -- it scales "
            f"with available VMEM / per-token bytes."
        )


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
_PallasCopyGuards = dict[int, tuple[int, ...]]
_PallasDimensionSemantic = Literal["parallel", "arbitrary"]


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


def _pallas_shared_output_plan(
    grid: tuple[int, ...],
    tensor_arg_indices: list[int],
    output_only_indices: list[int],
    output_indices: list[int],
    inplace_indices: set[int],
    block_spec_info: _BlockSpecInfo | None,
) -> tuple[_PallasCopyGuards, tuple[_PallasDimensionSemantic, ...]]:
    """Plan ordered updates for aliased outputs shared by multiple programs."""
    dim_semantics: list[_PallasDimensionSemantic] = ["parallel"] * len(grid)
    copy_guards: _PallasCopyGuards = {}
    if not output_indices or not grid:
        return copy_guards, tuple(dim_semantics)
    if block_spec_info is None:
        return copy_guards, tuple(dim_semantics)

    arg_to_tpos = _pallas_tensor_pos_map(tensor_arg_indices, output_only_indices)
    for orig_pos in output_indices:
        if orig_pos not in inplace_indices:
            continue
        tensor_pos = arg_to_tpos.get(orig_pos)
        if tensor_pos is None or tensor_pos >= len(block_spec_info):
            continue
        block_info = block_spec_info[tensor_pos]
        if block_info is None:
            continue
        used_dims = _pallas_grid_dims_used_by_block_spec(block_info)
        # These programs update the same output tile and must observe one
        # shared accumulator, not a freshly preloaded copy per program.
        shared_dims = tuple(
            dim for dim, size in enumerate(grid) if size > 1 and dim not in used_dims
        )
        if not shared_dims:
            continue
        copy_guards[orig_pos] = shared_dims
        for dim in shared_dims:
            dim_semantics[dim] = "arbitrary"
    return copy_guards, tuple(dim_semantics)


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
    """Build ``in_specs`` and ``out_specs`` for the launcher.

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
                pl, jnp, pltpu, t, block_spec_info[tensor_pos], should_use_smem
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
    hbm_arg_indices: list[int] | None,
    output_only_indices: list[int] | None = None,
    smem_arg_indices: list[int] | None = None,
) -> tuple[list[object], object]:
    """Build in/out specs for the pipeline/scratch path.

    Tensors listed in *hbm_arg_indices* get HBM refs (used by pipeline
    launchers as the outer HBM ref that DMAs into VMEM, and by
    distributed ops that address peer HBM directly).  All other
    tensors get proper BlockSpecs for automatic VMEM prefetch.
    Tensors in *smem_arg_indices* (only ever accessed by scalar index,
    e.g. group offset tables) are placed in SMEM so dynamic scalar
    reads don't require 128-lane alignment proofs against a small
    VMEM ref.
    """
    hbm_set = set(hbm_arg_indices or [])
    smem_set = set(smem_arg_indices or [])
    all_positions = sorted(set(tensor_arg_indices) | set(output_only_indices or []))
    arg_to_tpos = {orig: tpos for tpos, orig in enumerate(all_positions)}

    def _spec_for(idx: int) -> object:
        if idx in hbm_set:
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


@dataclass(slots=True)
class _DirectCallKernel:
    """Pre-captured metadata for a direct ``call_custom_kernel`` invocation.

    Built lazily on the first call of a static-shape Pallas kernel and
    attached to the launcher cache so subsequent calls bypass
    ``JaxCallable.__call__``.  ``sig`` guards against shape changes (mismatch
    falls back to JaxCallable); ``sig_locked`` flips after the first match so
    later calls skip the sig check.  ``invoke`` is a pre-baked dispatch closure
    populated by ``_build_direct_call_invoke``.
    """

    call_custom_kernel: object
    kernel_name: str
    kernel_key: str
    output_shapes: object
    donate_argnums: object
    out_tree: object
    alias_items: tuple[tuple[int, int], ...]
    sig: tuple[object, ...]
    invoke: object
    sig_locked: bool = False


def _build_direct_call_invoke(
    call_custom_kernel: object,
    kernel_name: str,
    kernel_key: str,
    output_shapes: object,
    donate_argnums: object,
    out_tree: object,
    alias_items: tuple[tuple[int, int], ...],
) -> object:
    """Pre-bake a closure that runs the direct-dispatch hot path; two variants
    (no-alias / with-alias) avoid a per-call branch on ``alias_items``."""
    if not alias_items:

        def invoke_no_alias(input_tensors: list[object]) -> object:
            results = call_custom_kernel(  # type: ignore[operator]
                kernel_name,
                kernel_key,
                inputs=input_tensors,
                output_shapes=output_shapes,
                donate_argnums=donate_argnums,
            )
            return out_tree.unflatten(results)  # type: ignore[attr-defined]

        return invoke_no_alias

    def invoke_with_alias(input_tensors: list[object]) -> object:
        results = call_custom_kernel(  # type: ignore[operator]
            kernel_name,
            kernel_key,
            inputs=input_tensors,
            output_shapes=output_shapes,
            donate_argnums=donate_argnums,
        )
        for in_idx, out_idx in alias_items:
            input_tensors[in_idx].copy_(results[out_idx])  # type: ignore[attr-defined]
        return out_tree.unflatten(results)  # type: ignore[attr-defined]

    return invoke_with_alias


_HELION_STATIC_JAX_CALLABLE_CLASS: type | None = None


def _make_helion_static_jax_callable_class() -> type:
    """Build a ``JaxCallable`` subclass that caches torch_tpu's per-call invocation key."""

    global _HELION_STATIC_JAX_CALLABLE_CLASS
    if _HELION_STATIC_JAX_CALLABLE_CLASS is not None:
        return _HELION_STATIC_JAX_CALLABLE_CLASS

    from torch_tpu._internal.pallas import (  # pyrefly: ignore[missing-import]
        tpu_torch_pallas,
    )
    from torch_tpu._internal.pallas.pallas import (  # pyrefly: ignore[missing-import]
        JaxCallable,
    )

    class _HelionStaticJaxCallable(JaxCallable):  # type: ignore[misc, valid-type]
        """``JaxCallable`` subclass with a direct-call snapshot.

        The first call goes through the JaxCallable slow path and
        populates ``_helion_direct_call`` with a pre-captured
        ``_DirectCallKernel``; the launcher hot path picks that up so
        subsequent calls bypass ``JaxCallable.__call__`` entirely.
        """

        __slots__ = ("_helion_direct_call",)

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[misc]
            # Pre-captured metadata so the launcher hot path can bypass
            # this ``__call__`` entirely; populated on the first call.
            self._helion_direct_call: _DirectCallKernel | None = None

        def __call__(self, *args: object, **kwargs: object) -> object:
            # First call goes through the JaxCallable slow path; the launcher
            # snapshot built afterwards lets later calls skip this method.
            result = super().__call__(*args, **kwargs)

            if kwargs or not self.output_shapes:
                return result

            from torch_tpu._internal.pallas.pallas import (  # pyrefly: ignore[missing-import]
                _get_kernel_invocation_key,
            )

            kernel_key = _get_kernel_invocation_key(
                self.trace_key, args, kwargs, self.static_argnums
            )
            cached_entry = self.output_shapes.get(kernel_key)
            if cached_entry is None:
                return result
            output_shapes, out_tree = cached_entry
            sig_tuple = tuple(
                (a.shape, a.dtype)  # type: ignore[attr-defined]
                for a in args
            )
            alias_items = tuple(self.input_output_aliases.items())
            # Stash the launcher-side direct-call structure so the next call
            # can bypass this ``__call__`` entirely.  Pre-bake ``invoke`` now
            # so the hot path skips the attribute walk + kwargs dict alloc.
            invoke = _build_direct_call_invoke(
                tpu_torch_pallas.call_custom_kernel,
                self.name,
                kernel_key,
                output_shapes,
                self.donate_argnums,
                out_tree,
                alias_items,
            )
            self._helion_direct_call = _DirectCallKernel(
                call_custom_kernel=tpu_torch_pallas.call_custom_kernel,
                kernel_name=self.name,
                kernel_key=kernel_key,
                output_shapes=output_shapes,
                donate_argnums=self.donate_argnums,
                out_tree=out_tree,
                alias_items=alias_items,
                sig=sig_tuple,
                invoke=invoke,
            )
            return result

    _HELION_STATIC_JAX_CALLABLE_CLASS = _HelionStaticJaxCallable
    return _HelionStaticJaxCallable


def _pallas_output_only_descriptors(
    _output_indices: list[int],
    arg_to_tensor_pos: dict[int, int],
) -> tuple[tuple[int, int], ...]:
    """Return ``((out_idx, orig_pos), ...)`` for write-only outputs.

    These positions appear in ``_output_indices`` but not in
    ``arg_to_tensor_pos`` — i.e. the kernel produces them as fresh
    buffers rather than aliasing back into an input tensor.  Both the
    torch fast-path (``_LauncherFastPath``) and the JAX-export
    launcher iterate this tuple to pick output-only results out of
    the full ``pallas_call`` result list.
    """
    return tuple(
        (out_idx, orig_pos)
        for out_idx, orig_pos in enumerate(_output_indices)
        if orig_pos not in arg_to_tensor_pos
    )


def _pallas_padded_output_dims_by_arg(
    _ds_pad_dims: list[tuple[int, int, int, int]],
    output_arg_set: frozenset[int] | set[int],
) -> dict[int, list[int]]:
    """Group ``_ds_pad_dims`` entries (arg_idx → padded dims) for output args.

    ``_ds_pad_dims`` carries ``(arg_idx, dim, block_size, extra_pad)``
    tuples for every padded position; this filter keeps only the ones
    whose ``arg_idx`` is in ``output_arg_set`` so callers can slice
    those outputs back to their original shapes.  Both the torch path
    (via ``_LauncherFastPath``) and the JAX-export launcher use this.
    """
    padded_dims_by_arg: dict[int, list[int]] = {}
    for arg_idx, dim, _bs, _extra in _ds_pad_dims:
        if arg_idx in output_arg_set:
            padded_dims_by_arg.setdefault(arg_idx, []).append(dim)
    return padded_dims_by_arg


class _LauncherFastPath:
    """Precomputed per-call state stored on the cached launcher entry."""

    __slots__ = (
        "ds_pad_required",  # bool|None: any non-zero pad? (None til 1st call)
        "ds_pad_orig_output_arg_indices",  # padded outputs that are also inputs
        "output_only_count",  # number of write-only output tensors
        "output_only_descriptors",  # (out_idx, orig_pos) per output-only result
        "padded_output_arg_indices",  # output args that get padded
        "padded_output_dims_by_arg",  # {arg: [padded dims]} (to slice back)
        "tensor_arg_indices_tuple",  # tensor arg positions (tuple = fast iter)
    )

    def __init__(
        self,
        tensor_arg_indices: list[int],
        arg_to_tensor_pos: dict[int, int],
        _output_indices: list[int],
        _ds_pad_dims: list[tuple[int, int, int, int]] | None,
    ) -> None:
        # Tuple iteration is faster than list in the hot-path comprehension.
        self.tensor_arg_indices_tuple: tuple[int, ...] = tuple(tensor_arg_indices)

        self.output_only_descriptors: tuple[tuple[int, int], ...] = (
            _pallas_output_only_descriptors(_output_indices, arg_to_tensor_pos)
        )
        self.output_only_count: int = len(self.output_only_descriptors)

        # ``None`` sentinel: filled in on the first call once we know if any pad is non-zero.
        self.ds_pad_required: bool | None = None

        if _ds_pad_dims:
            self.padded_output_dims_by_arg: dict[int, list[int]] = (
                _pallas_padded_output_dims_by_arg(_ds_pad_dims, set(_output_indices))
            )
            self.padded_output_arg_indices: frozenset[int] = frozenset(
                self.padded_output_dims_by_arg.keys()
            )
            self.ds_pad_orig_output_arg_indices: frozenset[int] = frozenset(
                idx
                for idx in self.padded_output_arg_indices
                if idx in arg_to_tensor_pos
            )
        else:
            self.padded_output_dims_by_arg = {}
            self.padded_output_arg_indices = frozenset()
            self.ds_pad_orig_output_arg_indices = frozenset()


def _pallas_slice_to_orig(
    t: torch.Tensor, dims: list[int], orig_shape: torch.Size
) -> torch.Tensor:
    """Slice a ds-padded tensor back to ``orig_shape`` along ``dims``."""
    slices: list[slice] = [slice(None)] * t.ndim
    for dim in dims:
        slices[dim] = slice(None, orig_shape[dim])
    return t[tuple(slices)]


def _pallas_collect_outputs(
    results: object,
    args: tuple[object, ...],
    output_only_descriptors: Iterable[tuple[int, int]],
    orig_output_tensors: dict[int, torch.Tensor] | None,
    padded_dims_by_arg: dict[int, list[int]],
    inplace_output_arg_indices: Iterable[int],
) -> object:
    """Turn raw kernel ``results`` into the launcher's return value.

    1. Copy each ds-padded in-place output (the kernel wrote into the padded
       ``args`` entry) back into its original unpadded tensor.
    2. Collect the output-only results, converting JAX arrays to torch in
       interpret mode and slicing any ds-padded result back to its true shape.

    ``orig_output_tensors`` maps each padded output arg to its original tensor,
    or is ``None`` when no ds-padding happened (the common case).  Returns
    ``None`` / a single tensor / a tuple, per the number of output-only results.
    """
    if results is None:
        return None
    if not isinstance(results, (tuple, list)):
        results = (results,)

    # (1) Copy padded in-place outputs back into the caller's tensors.
    if orig_output_tensors:
        for arg_idx in inplace_output_arg_indices:
            orig = orig_output_tensors.get(arg_idx)
            dims = padded_dims_by_arg.get(arg_idx)
            if orig is not None and dims:
                padded = cast("torch.Tensor", args[arg_idx])
                orig.copy_(_pallas_slice_to_orig(padded, dims, orig.shape))

    # (2) Collect (and unpad) the output-only results.
    output_only_results: list[object] = []
    for out_idx, orig_pos in output_only_descriptors:
        result = results[out_idx]
        if not isinstance(result, torch.Tensor):
            # Interpret mode: pallas_call returns JAX arrays; convert to torch.
            # Output-only tensors are allocated on device='meta' to avoid HBM,
            # so route the converted tensor to CPU where interpret mode runs.
            out_tensor = cast("torch.Tensor", args[orig_pos])
            device = out_tensor.device
            if device.type == "meta":
                device = torch.device("cpu")
            result = _jax_to_torch(result, device=device, dtype=out_tensor.dtype)
        if orig_output_tensors is not None:
            orig = orig_output_tensors.get(orig_pos)
            dims = padded_dims_by_arg.get(orig_pos)
            if orig is not None and dims and isinstance(result, torch.Tensor):
                result = _pallas_slice_to_orig(result, dims, orig.shape)
        output_only_results.append(result)

    if not output_only_results:
        return None
    if len(output_only_results) == 1:
        return output_only_results[0]
    return tuple(output_only_results)


def _pallas_apply_ds_padding_fast(
    args: tuple[object, ...],
    _ds_pad_dims: list[tuple[int, int, int, int]],
    fast_path: _LauncherFastPath,
    padded_output_arg_indices: frozenset[int],
) -> tuple[tuple[object, ...], dict[int, torch.Tensor] | None, bool]:
    """``_pallas_apply_ds_padding`` with a short-circuit when every pad amount is zero."""
    args_list: list[object] | None = None
    orig_output_tensors: dict[int, torch.Tensor] | None = None
    any_padding = False
    for arg_idx, dim, block_size, extra_pad in _ds_pad_dims:
        a = args[arg_idx] if args_list is None else args_list[arg_idx]
        if not isinstance(a, torch.Tensor):
            continue
        pad_amount = (-a.shape[dim]) % block_size + extra_pad
        if pad_amount == 0:
            continue
        any_padding = True
        if args_list is None:
            args_list = list(args)
        if arg_idx in padded_output_arg_indices:
            if orig_output_tensors is None:
                orig_output_tensors = {}
            if arg_idx not in orig_output_tensors:
                orig_output_tensors[arg_idx] = cast("torch.Tensor", a)
        pad_widths = [0] * (2 * a.ndim)
        pad_widths[2 * (a.ndim - 1 - dim) + 1] = pad_amount
        args_list[arg_idx] = torch.nn.functional.pad(a, pad_widths)
    if fast_path.ds_pad_required is None:
        # First-call precomputation: lock in whether any pad amount is
        # non-zero so subsequent calls can elide the iteration outright.
        fast_path.ds_pad_required = any_padding
    if args_list is None:
        return args, None, False
    return tuple(args_list), orig_output_tensors, True


def _pallas_invoke_and_return_fast(
    jax_callable: object,
    args: tuple[object, ...],
    fast_path: _LauncherFastPath,
    _orig_output_tensors: dict[int, torch.Tensor] | None,
    direct_call: _DirectCallKernel | None = None,
) -> object:
    """Run the JaxCallable (or pre-baked direct call) and collect output-only
    results; ``direct_call`` bypasses ``jax_callable`` when the sig matches."""
    tensor_arg_indices = fast_path.tensor_arg_indices_tuple
    input_tensors = [
        cast("torch.Tensor", args[i]).contiguous() for i in tensor_arg_indices
    ]
    if direct_call is not None:
        # Once the sig matches once, grid-keyed cache + static_shapes makes
        # subsequent sig checks constant-True; skip them on the locked path
        # and call the pre-baked ``invoke`` closure directly.
        if direct_call.sig_locked:
            results = direct_call.invoke(input_tensors)  # type: ignore[operator]
        else:
            # First direct-dispatch call: verify sig, flip the lock on match.
            # Mismatch (dynamic shape reusing cache) falls back to JaxCallable.
            direct_sig: tuple[object, ...] = tuple(
                (a.shape, a.dtype) for a in input_tensors
            )
            if direct_sig == direct_call.sig:
                direct_call.sig_locked = True
                results = direct_call.invoke(input_tensors)  # type: ignore[operator]
            else:
                results = jax_callable(*input_tensors)  # type: ignore[operator]
    else:
        results = jax_callable(*input_tensors)  # type: ignore[operator]

    output_only_count = fast_path.output_only_count
    if output_only_count == 0 and _orig_output_tensors is None:
        # Hottest path: in-place outputs already written through donated aliases.
        return None
    # Hot single-output (matmul) short-circuit: skip result post-processing.
    if (
        output_only_count == 1
        and _orig_output_tensors is None
        and isinstance(results, torch.Tensor)
    ):
        return results

    return _pallas_collect_outputs(
        results,
        args,
        fast_path.output_only_descriptors,
        _orig_output_tensors,
        fast_path.padded_output_dims_by_arg,
        fast_path.ds_pad_orig_output_arg_indices,
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


def _pallas_do_smem_inplace_copy(
    in_ref: object,
    out_ref: object,
    current_indices: tuple[int, ...] = (),
) -> None:
    if len(current_indices) == len(in_ref.shape):  # type: ignore[attr-defined]
        out_ref[current_indices] = in_ref[current_indices]  # type: ignore[index]
        return
    next_dim = len(current_indices)
    for i in range(in_ref.shape[next_dim]):  # type: ignore[attr-defined]
        _pallas_do_smem_inplace_copy(in_ref, out_ref, (*current_indices, i))


def _pallas_inplace_copy(in_ref: object, out_ref: object, *, is_smem: bool) -> None:
    if is_smem:
        _pallas_do_smem_inplace_copy(in_ref, out_ref)
    else:
        out_ref[...] = in_ref[...]  # type: ignore[index]


def _pallas_copy_guard(dims: tuple[int, ...]) -> bool | jax.Array:
    from jax.experimental import pallas as pl

    should_copy = True
    for dim in dims:
        should_copy = should_copy & (pl.program_id(dim) == 0)
    return should_copy


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
    _copy_guards: _PallasCopyGuards | None = None,
) -> object:
    """Create a wrapper kernel that reorders pallas_call refs to the original arg order.

    ``pallas_call`` provides refs as ``[inputs..., outputs...]``, but Helion
    kernels expect the original parameter order.  When *n_extra_refs* > 0
    (e.g. scratch buffers), those trailing refs are appended after the
    reordered args.

    *skip_inplace_copy* is a set of original-arg positions for which the
    initial ``out_ref[...] = in_ref[...]`` copy should be skipped.  Used
    for tensors backed by HBM refs (pipeline/fori-loop outer refs,
    distributed-op targets) where direct load/store is not allowed.
    """
    _skip_copy = skip_inplace_copy or set()
    copy_guards = {
        orig_pos: guard_dims
        for orig_pos, guard_dims in (_copy_guards or {}).items()
        if guard_dims
    }

    def reordered_kernel(*refs: object) -> None:
        from jax.experimental import pallas as pl

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
                is_smem = (
                    _smem_arg_indices is not None and orig_pos in _smem_arg_indices
                )
                copy_guard_dims = copy_guards.get(orig_pos)
                if copy_guard_dims:
                    should_copy = _pallas_copy_guard(copy_guard_dims)

                    @pl.when(should_copy)
                    def _copy_shared_output(
                        out_ref: object = out_ref,
                        in_ref: object = in_ref,
                        is_smem: bool = is_smem,
                    ) -> None:
                        _pallas_inplace_copy(in_ref, out_ref, is_smem=is_smem)

                else:
                    _pallas_inplace_copy(in_ref, out_ref, is_smem=is_smem)
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
        # Seed with ``None`` fast-path slot; launcher overwrites with real ``_LauncherFastPath``.
        setattr(
            pallas_kernel,
            cache_attr,
            (grid, callable_obj, tensor_arg_indices, arg_to_tensor_pos, None),
        )
        return callable_obj

    if interpret:
        return _make_interpret_callable()

    import jax

    kernel_name = getattr(pallas_kernel, "__name__", "pallas_kernel")

    # JaxCallable subclass caches the per-call invocation key (see _make_helion_static_jax_callable_class).
    callable_cls = _make_helion_static_jax_callable_class()
    jax_callable = callable_cls(
        name=kernel_name,
        jit_fn=jax.jit(jit_fn),
        trace_key=f"{kernel_name}_{id(pallas_kernel)}_{grid}{trace_key_suffix}",
        input_output_aliases=call_aliases,
    )
    # Seed with ``None`` fast-path slot; launcher overwrites with real ``_LauncherFastPath``.
    setattr(
        pallas_kernel,
        cache_attr,
        (grid, jax_callable, tensor_arg_indices, arg_to_tensor_pos, None),
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
    4. Returns raw JAX results so ``_pallas_invoke_and_return_fast`` can
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
        # by _pallas_invoke_and_return_fast.
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


def _build_matmul_dot_general_jit_fn(
    spec: dict[str, object],
) -> Callable[..., object]:
    """Build a ``jax.jit(lax.dot_general)`` wrapper replacing the
    ``pl.pallas_call`` for a single-launch (no-tiling) Pallas matmul.

    Same call signature as the ``pl.pallas_call`` it replaces, so torch_tpu's
    dispatch is unchanged.  XLA sees a plain ``dot_general`` op (so
    ``cross_program_prefetch_index`` is reachable), bypassing the Pallas
    ``custom_call`` opacity that blocks the prefetch planner.  ``spec`` (from
    ``_detect_matmul_dot_general_lowering``) carries the lhs/rhs arg positions
    and the f32-accumulator flag.
    """
    import jax
    import jax.lax as lax
    import jax.numpy as jnp

    out_dtype_str = cast("str", spec["out_dtype"])
    out_jnp_dtype = cast("Any", _pallas_jnp_dtype_map().get(out_dtype_str, jnp.float32))
    f32_accumulator = bool(spec.get("f32_accumulator"))
    lhs_idx = int(cast("int", spec["lhs_tensor_arg_index"]))
    rhs_idx = int(cast("int", spec["rhs_tensor_arg_index"]))

    # Accumulate in f32 and cast back only when the output is narrower than f32
    # (bf16/fp16 out); otherwise accumulate straight into the output dtype.
    needs_cast = f32_accumulator and out_jnp_dtype is not jnp.float32
    preferred = jnp.float32 if needs_cast else out_jnp_dtype

    def matmul_fn(*tensor_inputs: Any) -> Any:  # noqa: ANN401
        result = lax.dot_general(
            tensor_inputs[lhs_idx],
            tensor_inputs[rhs_idx],
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=preferred,
        )
        if needs_cast:
            result = lax.convert_element_type(result, out_jnp_dtype)
        return result

    return cast("Callable[..., object]", jax.jit(matmul_fn))


def _pallas_build_scratch_shapes(
    pltpu: object,
    jnp: object,
    scratch_entries: list[object],
) -> list[object]:
    """Translate codegen scratch-shape descriptors into ``pltpu`` objects.

    Each entry is either ``(shape, dtype_str, scratch_type)`` or the
    legacy 2-tuple ``(shape, dtype_str)`` form (``scratch_type``
    defaults to ``"vmem"``).  Supported scratch types: ``"vmem"`` and
    ``"dma_semaphore"``.
    """
    _jnp_dtype_map = _pallas_jnp_dtype_map()
    scratch_shapes: list[object] = []
    for entry in scratch_entries:
        if len(entry) == 3:  # type: ignore[arg-type]
            shape, dtype_str, scratch_type = entry  # type: ignore[misc]
        else:
            shape, dtype_str = entry  # type: ignore[misc]
            scratch_type = "vmem"
        if scratch_type == "dma_semaphore":
            scratch_shapes.append(pltpu.SemaphoreType.DMA(()))  # type: ignore[union-attr]
        else:
            assert dtype_str is not None
            jnp_dtype = _jnp_dtype_map.get(dtype_str, jnp.float32)  # type: ignore[union-attr]
            scratch_shapes.append(
                pltpu.VMEM(shape, jnp_dtype)  # type: ignore[union-attr]  # pyrefly: ignore[bad-argument-type]
            )
    return scratch_shapes


def _pallas_check_vmem_or_raise(
    pl: object,
    pltpu: object,
    in_specs: list[object] | None,
    out_specs: list[object] | object | None,
    scratch_shapes: list[object] | None,
    args: tuple[object, ...],
    tensor_arg_indices: list[int],
    output_indices: list[int],
    pallas_aliases: dict[int, int] | None,
) -> None:
    """Estimate the kernel's VMEM footprint and raise if it exceeds the limit."""
    estimated_vmem = _estimate_pallas_vmem_bytes(
        pl,
        pltpu,
        in_specs,
        out_specs,
        scratch_shapes,
        args,
        tensor_arg_indices,
        output_indices,
        pallas_aliases,
    )
    vmem_limit_bytes = _get_vmem_limit_bytes(pltpu)
    if estimated_vmem > vmem_limit_bytes:
        raise RuntimeError(
            f"XLA:TPU compile permanent error. Ran out of memory in memory space vmem. "
            f"Estimated {estimated_vmem / 1e6:.2f}MB exceeds {vmem_limit_bytes / 1e6:.2f}MB vmem capacity."
        )


def _pallas_pl_kernel_jit_fn(
    pl: object,
    pltpu: object,
    reordered_kernel: object,
    *,
    out_shape_arg: object,
    grid: tuple[int, ...],
    in_specs: list[object],
    out_specs: object,
    scratch_shapes: list[object],
    n_inputs: int,
    n_outputs: int,
    hbm_in_positions: set[int],
    hbm_out_positions: set[int],
    interpret: bool,
) -> object:
    """Build the ``pl.kernel`` jit_fn that drives the Helion device kernel.

    The kernel body receives ANY-space refs ``[inputs..., outputs...,
    scratch...]`` and iterates the grid with ``pltpu.emit_pipeline``.

    - Tensors at *hbm_in_positions* / *hbm_out_positions* are not pipelined:
      ``emit_pipeline`` rejects ANY-space buffer specs, so their raw refs are
      closure-captured and stitched back into position for the kernel to DMA
      manually.
    - Scratch refs are closure-forwarded; the primitive ``emit_pipeline``
      implementation has no ``scratches=`` kwarg.
    - No ``dimension_semantics``: ``TensorCoreMesh`` forbids it, and
      ``emit_pipeline`` runs grid steps sequentially in ascending order.
    """
    out_specs_seq = (
        list(out_specs) if isinstance(out_specs, (list, tuple)) else [out_specs]
    )
    pipeline_grid = grid or (1,)
    n_io = n_inputs + n_outputs
    # Positions (within [inputs..., outputs...]) that go through the
    # pipeline; the rest are raw pass-through refs.
    pipe_positions = [i for i in range(n_inputs) if i not in hbm_in_positions]
    pipe_positions += [
        n_inputs + i for i in range(n_outputs) if i not in hbm_out_positions
    ]
    all_specs = list(in_specs) + out_specs_seq
    pipe_in_specs = [all_specs[p] for p in pipe_positions if p < n_inputs]
    pipe_out_specs = [all_specs[p] for p in pipe_positions if p >= n_inputs]

    def kernel_body(*refs: object) -> None:
        io_any = refs[:n_io]
        # Mixed spaces: VMEM/SMEM buffers and DMA semaphores.
        scratch_refs = refs[n_io:]
        pipe_any = [io_any[p] for p in pipe_positions]

        # block_refs are the per-step windowed buffers (VMEM, or SMEM for
        # SMEM-spec'd args), one per pipe_positions entry.
        def pipeline_body(*block_refs: object) -> None:
            merged = list(io_any)
            for p, block in zip(pipe_positions, block_refs, strict=True):
                merged[p] = block
            reordered_kernel(*merged, *scratch_refs)  # type: ignore[operator]

        pltpu.emit_pipeline(  # type: ignore[union-attr]
            pipeline_body,
            grid=pipeline_grid,
            in_specs=pipe_in_specs,
            out_specs=pipe_out_specs,
        )(*pipe_any)

    mesh = pltpu.create_tensorcore_mesh("core", num_cores=1)  # type: ignore[union-attr]
    # Distributed kernels may run in-kernel collectives (a cross-chip barrier via
    # get_barrier_semaphore); Mosaic requires the enclosing kernel to declare a
    # collective_id for those. Gate it on the ``distributed`` setting so
    # non-distributed kernels are unaffected. dimension_semantics is deliberately
    # NOT set here (TensorCoreMesh forbids it — see the note above).
    compiler_params = (
        pltpu.CompilerParams(collective_id=0)  # type: ignore[union-attr]  # pyrefly: ignore[bad-instantiation]
        if _pallas_is_distributed()
        else None
    )
    return pl.kernel(  # type: ignore[union-attr]
        kernel_body,
        out_shape_arg,
        mesh=mesh,
        scratch_types=scratch_shapes,
        compiler_params=compiler_params,
        interpret=interpret,
    )


@dataclass(slots=True)
class _PallasCompileResult:
    """Bundle returned by :func:`_pallas_compile_jit_fn`.

    Carries the compiled ``pl.kernel`` jit_fn plus all per-arg metadata
    that downstream consumers (``_pallas_build_callable``,
    ``_LauncherFastPath`` setup, the JAX-export launcher) need to wire
    inputs and outputs.  The fields mirror the named portion of the
    ``_pallas_prepare_args`` return tuple so consumers can use them
    directly without re-running argument prep.
    """

    jit_fn: object
    tensor_arg_indices: list[int]
    output_only_indices: list[int]
    arg_to_tensor_pos: dict[int, int]
    inplace_positions: set[int]
    pallas_aliases: dict[int, int]


def _pallas_is_distributed() -> bool:
    """True if the current kernel declares ``distributed=[...]``.

    Distributed kernels may issue in-kernel collectives (e.g. a cross-chip
    ``get_barrier_semaphore`` barrier) whose pallas_call must carry a
    ``collective_id``.  Resolved from the active ``CompileEnvironment`` settings;
    returns False if unavailable so non-distributed kernels are unaffected.
    """
    try:
        from .._compiler.compile_environment import CompileEnvironment

        return bool(CompileEnvironment.current().settings.distributed)
    except Exception:
        return False


def _pallas_compile_jit_fn(
    pallas_kernel: object,
    grid: tuple[int, ...],
    args: tuple[object, ...],
    *,
    _output_indices: list[int],
    _inplace_indices: list[int] | None,
    _block_spec_info: _BlockSpecInfo | None,
    _smem_arg_indices: list[int] | None,
    _scratch_shapes: list[object] | None,
    _hbm_arg_indices: list[int] | None,
    _matmul_dot_general: dict[str, object] | None,
    interpret: bool,
) -> _PallasCompileResult:
    """Build the ``pl.kernel`` jit_fn used by the Pallas launcher.

    The kernel loop shape is driven entirely by the launcher-observable
    inputs:

    - ``_scratch_shapes`` present (VMEM buffers / DMA semaphores) →
      the scratch refs are allocated by ``pl.kernel`` and forwarded to
      the device kernel.  Tensors listed in ``_hbm_arg_indices`` get
      raw pass-through refs (see :func:`_pallas_pl_kernel_jit_fn`), and
      their inplace copy is skipped because you cannot directly index
      an HBM ref.
    - ``_scratch_shapes`` absent → simple ``grid`` + per-arg
      ``BlockSpec`` layout via ``_pallas_build_block_specs``.

    When ``_matmul_dot_general`` is provided (no-tiling matmul configs
    on the unroll / emit_pipeline lowerings), substitutes
    ``jax.jit(lax.dot_general)`` for the Pallas launch and skips the
    VMEM check; XLA's planner streams the contraction so the Pallas
    lowering's VMEM estimate doesn't apply.

    ``args`` must already have any ds-padding applied — this helper
    builds specs from the post-pad shapes.  Returns a
    :class:`_PallasCompileResult` so the torch launcher can wrap the
    jit_fn in a JaxCallable while the JAX-export path calls it directly.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    import jax.numpy as jnp

    if interpret:
        _ensure_cpu_tpu_info()

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

    # Only the copy guards are consumed; the dimension semantics are implied
    # by emit_pipeline's sequential in-order grid execution.
    copy_guards, _ = _pallas_shared_output_plan(
        grid,
        tensor_arg_indices,
        output_only_indices,
        _output_indices,
        inplace_positions,
        _block_spec_info,
    )

    # Two discriminators drive the spec-building path — either forces
    # the pipeline-spec route:
    #   1. ``_hbm_arg_indices`` non-empty: some tensor needs a raw
    #      pass-through ref rather than a plain BlockSpec.
    #   2. ``_scratch_shapes`` non-empty: the kernel registered VMEM
    #      buffers or DMA semaphores.
    needs_pipeline_specs = bool(_hbm_arg_indices) or bool(_scratch_shapes)
    has_scratch = bool(_scratch_shapes)
    if needs_pipeline_specs:
        if _block_spec_info is None:
            # Distributed-only kernels with no tile analysis reach here
            # without a block_spec_info from codegen.  Synthesize a
            # trivial one (one ``None`` per tensor arg + output) so the
            # pipeline spec builder treats every non-HBM tensor as
            # untiled full-buffer.
            all_positions = sorted(
                set(tensor_arg_indices) | set(output_only_indices or [])
            )
            _block_spec_info = [None] * len(all_positions)
        scratch_shapes = _pallas_build_scratch_shapes(pltpu, jnp, _scratch_shapes or [])
        in_specs, out_specs = _pallas_build_pipeline_specs(
            pl,
            jnp,
            pltpu,
            grid,
            args,
            tensor_arg_indices,
            _output_indices,
            _block_spec_info,
            _hbm_arg_indices,
            output_only_indices,
            smem_arg_indices=_smem_arg_indices,
        )
        skip_inplace_copy: set[int] = set(_hbm_arg_indices or [])
    else:
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
        scratch_shapes = []
        skip_inplace_copy = set()

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
        skip_inplace_copy=skip_inplace_copy,
        _smem_arg_indices=_smem_arg_indices,
        _copy_guards=copy_guards,
    )

    out_shape_arg = out_shapes if len(out_shapes) > 1 else out_shapes[0]

    # The VMEM estimate only applies to the ``pl.pallas_call`` lowering.
    # The ``jax.jit(lax.dot_general)`` substitution streams the
    # contraction through XLA's planner, so the pallas_call estimate
    # doesn't apply — skip the check there.
    if _matmul_dot_general is None:
        _pallas_check_vmem_or_raise(
            pl,
            pltpu,
            in_specs,
            out_specs,
            scratch_shapes if has_scratch else None,
            args,
            tensor_arg_indices,
            _output_indices,
            pallas_aliases,
        )

    if _matmul_dot_general is not None:
        # Substitute ``lax.dot_general`` for the Pallas launch on
        # no-tiling matmul configs so XLA sees a regular ``dot`` and
        # can attach ``cross_program_prefetch_index``.
        jit_fn = _build_matmul_dot_general_jit_fn(_matmul_dot_general)
    else:
        # emit_pipeline needs concrete specs: build whole-array specs when
        # the builders returned None (no tiling info or empty grid).  The
        # VMEM estimate above deliberately sees the original specs, not
        # these synthetic full-shape ones.
        launch_in_specs = in_specs
        launch_out_specs = out_specs
        if launch_in_specs is None:
            all_positions = sorted(set(tensor_arg_indices) | set(output_only_indices))
            all_arg_to_tpos = {orig: tpos for tpos, orig in enumerate(all_positions)}
            smem_set = set(_smem_arg_indices or [])

            def _full_spec(idx: int) -> object:
                t = args[idx]
                assert isinstance(t, torch.Tensor)
                return _pallas_make_block_spec(
                    pl, jnp, pltpu, t, None, all_arg_to_tpos[idx] in smem_set
                )

            launch_in_specs = [_full_spec(idx) for idx in tensor_arg_indices]
            out_list = [_full_spec(idx) for idx in _output_indices]
            launch_out_specs = out_list if len(out_list) > 1 else out_list[0]

        hbm_set = set(_hbm_arg_indices or [])
        jit_fn = _pallas_pl_kernel_jit_fn(
            pl,
            pltpu,
            reordered_kernel,
            out_shape_arg=out_shape_arg,
            grid=grid,
            in_specs=launch_in_specs,  # pyrefly: ignore[bad-argument-type]
            out_specs=launch_out_specs,
            scratch_shapes=scratch_shapes,
            n_inputs=len(tensor_arg_indices),
            n_outputs=len(_output_indices),
            hbm_in_positions={
                i for i, idx in enumerate(tensor_arg_indices) if idx in hbm_set
            },
            hbm_out_positions={
                i for i, idx in enumerate(_output_indices) if idx in hbm_set
            },
            interpret=interpret,
        )

    return _PallasCompileResult(
        jit_fn=jit_fn,
        tensor_arg_indices=tensor_arg_indices,
        output_only_indices=output_only_indices,
        arg_to_tensor_pos=arg_to_tensor_pos,
        inplace_positions=inplace_positions,
        pallas_aliases=pallas_aliases,
    )


_PALLAS_CACHE_ATTR = "_pallas_cache"


def _pallas_install_launcher_cache(
    pallas_kernel: object,
    grid: tuple[int, ...],
    args: tuple[object, ...],
    *,
    _output_indices: list[int] | None,
    _inplace_indices: list[int] | None,
    _block_spec_info: _BlockSpecInfo | None,
    _smem_arg_indices: list[int] | None,
    _scratch_shapes: list[object] | None,
    _hbm_arg_indices: list[int] | None,
    _ds_pad_dims: list[tuple[int, int, int, int]] | None,
    _pallas_interpret: bool | None,
    _matmul_dot_general: dict[str, object] | None = None,
) -> tuple[object, ...]:
    """Cache-miss path shared by all Pallas launchers.

    Builds the ``pl.kernel`` jit_fn via :func:`_pallas_compile_jit_fn`
    (whose shape is fully determined by the passed-in kwargs — no loop-type
    discriminator), wraps it in a ``JaxCallable`` (or interpret-mode shim),
    seeds the ``_LauncherFastPath`` slot, stores the result on
    ``pallas_kernel._pallas_cache``, and returns the freshly-installed cache
    tuple so the caller can fall straight through to the shared invoke.
    """
    interpret = (
        _pallas_interpret
        if _pallas_interpret is not None
        else _module_is_pallas_interpret()
    )
    if interpret:
        _ensure_cpu_tpu_info()

    output_indices = _output_indices if _output_indices is not None else []

    # Build the pallas specs from ds-padded shapes on a throwaway copy so
    # ``args`` stays unpadded for the shared invoke below to pad fresh.
    spec_args = args
    if _ds_pad_dims:
        spec_args, _ = _pallas_apply_ds_padding(args, output_indices, _ds_pad_dims)

    _pallas_check_dtypes(spec_args)

    result = _pallas_compile_jit_fn(
        pallas_kernel,
        grid,
        spec_args,
        _output_indices=output_indices,
        _inplace_indices=_inplace_indices,
        _block_spec_info=_block_spec_info,
        _smem_arg_indices=_smem_arg_indices,
        _scratch_shapes=_scratch_shapes,
        _hbm_arg_indices=_hbm_arg_indices,
        _matmul_dot_general=_matmul_dot_general,
        interpret=interpret,
    )

    jax_callable = _pallas_build_callable(
        pallas_kernel,
        grid,
        cast("Callable[..., object]", result.jit_fn),
        output_indices,
        result.arg_to_tensor_pos,
        result.tensor_arg_indices,
        cache_attr=_PALLAS_CACHE_ATTR,
        call_aliases=result.pallas_aliases,
        trace_key_suffix="",
        interpret=interpret,
    )

    fast_path = _LauncherFastPath(
        result.tensor_arg_indices,
        result.arg_to_tensor_pos,
        output_indices,
        _ds_pad_dims,
    )
    cache = (
        grid,
        jax_callable,
        result.tensor_arg_indices,
        result.arg_to_tensor_pos,
        fast_path,
        None,
    )
    setattr(pallas_kernel, _PALLAS_CACHE_ATTR, cache)
    return cache


def _pallas_invoke_cached_launcher(
    pallas_kernel: object,
    cache: tuple[object, ...],
    args: tuple[object, ...],
    *,
    cache_attr: str,
    _ds_pad_dims: list[tuple[int, int, int, int]] | None,
) -> object:
    """Shared fast-invoke tail: lift direct-call snapshot, ds-pad, dispatch."""
    _grid = cache[0]
    jax_callable = cache[1]
    tensor_arg_indices = cast("list[int]", cache[2])
    arg_to_tensor_pos = cast("dict[int, int]", cache[3])
    fast_path = cast("_LauncherFastPath", cache[4])
    direct_call = cast("_DirectCallKernel | None", cache[5])
    if direct_call is None:
        # Lazily lift the direct-call kernel off the JaxCallable subclass.
        direct_call = getattr(jax_callable, "_helion_direct_call", None)
        if direct_call is not None:
            cache = (
                _grid,
                jax_callable,
                tensor_arg_indices,
                arg_to_tensor_pos,
                fast_path,
                direct_call,
            )
            setattr(pallas_kernel, cache_attr, cache)

    _orig_output_tensors: dict[int, torch.Tensor] | None = None
    if _ds_pad_dims and fast_path.ds_pad_required is not False:
        args, _orig_output_tensors, _ = _pallas_apply_ds_padding_fast(
            args,
            _ds_pad_dims,
            fast_path,
            fast_path.padded_output_arg_indices,
        )
    return _pallas_invoke_and_return_fast(
        jax_callable, args, fast_path, _orig_output_tensors, direct_call
    )


def default_pallas_launcher(
    pallas_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    _output_indices: list[int] | None = None,
    _inplace_indices: list[int] | None = None,
    _block_spec_info: _BlockSpecInfo | None = None,
    _smem_arg_indices: list[int] | None = None,
    _scratch_shapes: list[tuple[tuple[int, ...], str | None, str]] | None = None,
    _hbm_arg_indices: list[int] | None = None,
    _ds_pad_dims: list[tuple[int, int, int, int]] | None = None,
    _pallas_interpret: bool | None = None,
    _matmul_dot_general: dict[str, object] | None = None,
    _compact_build_worklist: Callable[..., object] | None = None,
    _compact_offset_arg_indices: list[int] | None = None,
    _compact_metadata_fields: list[str] | None = None,
    _compact_owner_ref_pos: int = 0,
    _compact_num_scalar_prefetch: int = 0,
    _compact_aligned_arg_indices: list[int] | None = None,
    _compact_tile_start_ref_pos: int = 1,
    _compact_block: int = 1,
    # Resident-cache (owner-cache) params: the backstop below reads all of them
    # every call; the three compile-relevant ones are threaded to the install path.
    _compact_ordered_aligned_arg_indices: list[int] | None = None,
    _compact_range_start_ref_pos: int = -1,
    _compact_ordered_offset_arg_index: int = -1,
    _compact_active_mask_arg_index: int = -1,
    _compact_ordered_window: int = 0,
    **kwargs: object,
) -> object:
    """Unified Pallas kernel launcher for TPU (or CPU with interpret=True).

    Dispatch is driven entirely by launcher-observable inputs:

    - ``_compact_build_worklist`` present → the kernel builds a
      dynamic-``num_work`` grid via ``_pallas_compile_compact_jit_fn``
      (compact-worklist path).
    - Otherwise → the standard ``_pallas_compile_jit_fn`` path.
      ``_pallas_compile_jit_fn`` internally chooses between a plain
      ``grid`` + ``BlockSpec`` layout (no scratch) and the
      pipeline/scratch layout based on ``_scratch_shapes``.

    Uses ``JaxCallable`` from ``torch_tpu`` to compile and run the Pallas
    kernel on TPU.  When ``torch_tpu`` is not available (interpret mode),
    falls back to direct torch<->JAX conversion.  Output tensors are donated
    via ``input_output_aliases`` so the kernel writes directly into their
    buffers (zero-copy on TPU).

    Output-only tensors (in ``_output_indices`` but not in ``_inplace_indices``)
    are excluded from pallas_call inputs to save VMEM.  Their results are
    returned as torch tensors.

    Distributed ops (e.g. ``hl.start_async_remote_copy``) register DMA
    semaphore scratch and mark their tensors as HBM.  Those pieces of
    metadata flow through the same ``_scratch_shapes`` /
    ``_hbm_arg_indices`` kwargs the pipeline launcher uses.
    """
    if _compact_build_worklist is not None:
        # Resident-cache correctness backstop: runs EVERY call (the offset arrays are
        # runtime data even when the compiled kernel is cached for this grid), raising
        # rather than silently over-reading the resident window when a source's ordered
        # reduction length exceeds the compile-time window C.  No-ops when resident
        # caching is inactive (empty _compact_ordered_aligned_arg_indices).
        _compact_raise_if_range_exceeds_window(
            args,
            _compact_ordered_aligned_arg_indices,
            _compact_ordered_offset_arg_index,
            _compact_active_mask_arg_index,
            _compact_ordered_window,
        )
    cache = getattr(pallas_kernel, _PALLAS_CACHE_ATTR, None)
    if cache is None or cache[0] != grid:
        if _compact_build_worklist is not None:
            cache = _pallas_install_compact_launcher_cache(
                pallas_kernel,
                grid,
                args,
                _output_indices=_output_indices,
                _inplace_indices=_inplace_indices,
                _block_spec_info=_block_spec_info,
                _smem_arg_indices=_smem_arg_indices,
                _scratch_shapes=cast("list[object] | None", _scratch_shapes),
                _hbm_arg_indices=_hbm_arg_indices,
                _ds_pad_dims=_ds_pad_dims,
                _pallas_interpret=_pallas_interpret,
                _compact_build_worklist=_compact_build_worklist,
                _compact_offset_arg_indices=_compact_offset_arg_indices,
                _compact_metadata_fields=_compact_metadata_fields,
                _compact_owner_ref_pos=_compact_owner_ref_pos,
                _compact_num_scalar_prefetch=_compact_num_scalar_prefetch,
                _compact_aligned_arg_indices=_compact_aligned_arg_indices,
                _compact_tile_start_ref_pos=_compact_tile_start_ref_pos,
                _compact_block=_compact_block,
                _compact_ordered_aligned_arg_indices=_compact_ordered_aligned_arg_indices,
                _compact_range_start_ref_pos=_compact_range_start_ref_pos,
                _compact_ordered_window=_compact_ordered_window,
            )
        else:
            cache = _pallas_install_launcher_cache(
                pallas_kernel,
                grid,
                args,
                _output_indices=_output_indices,
                _inplace_indices=_inplace_indices,
                _block_spec_info=_block_spec_info,
                _smem_arg_indices=_smem_arg_indices,
                _scratch_shapes=cast("list[object] | None", _scratch_shapes),
                _hbm_arg_indices=_hbm_arg_indices,
                _ds_pad_dims=_ds_pad_dims,
                _pallas_interpret=_pallas_interpret,
                _matmul_dot_general=_matmul_dot_general,
            )

    return _pallas_invoke_cached_launcher(
        pallas_kernel,
        cache,
        args,
        cache_attr=_PALLAS_CACHE_ATTR,
        _ds_pad_dims=_ds_pad_dims,
    )


def _pallas_compact_in_out_specs(
    pl: object,
    jnp: object,
    pltpu: object,
    args: tuple[object, ...],
    tensor_arg_indices: list[int],
    output_indices: list[int],
    block_spec_info: _BlockSpecInfo | None,
    smem_set: set[int],
    hbm_set: set[int],
    owner_ref_pos: int,
    aligned_set: set[int] | None = None,
    tile_start_ref_pos: int = 1,
    compact_block: int = 1,
    ordered_aligned_set: set[int] | None = None,
    range_start_ref_pos: int = -1,
    ordered_window: int = 0,
) -> tuple[list[object], object]:
    """Build in/out BlockSpecs for the compact-worklist PrefetchScalarGridSpec.

    Like ``_pallas_build_pipeline_specs`` but: pipelined tensors -> HBM; an
    owner-indexed tensor (its ``grid_dims`` carry the owner grid dim ``0``) gets
    an ``index_map`` that reads ``owner_ids[wid]`` (a scalar-prefetch ref); a
    compact-aligned-load tensor (``aligned_set``) gets a per-tile
    ``pl.Element`` slice at ``tile_start`` so Pallas double-buffers it across work
    items; everything else is full/SMEM.  Under ``PrefetchScalarGridSpec`` every
    ``index_map`` receives ``(wid, *scalar_refs)``.
    """
    aligned_set = aligned_set or set()
    ordered_aligned_set = ordered_aligned_set or set()
    all_positions = sorted(set(tensor_arg_indices) | set(output_indices))
    arg_to_tpos = {orig: tpos for tpos, orig in enumerate(all_positions)}

    def _spec_for(idx: int) -> object:
        if idx in hbm_set:
            return pl.BlockSpec(memory_space=pltpu.HBM)  # type: ignore[union-attr]
        t = args[idx]
        assert isinstance(t, torch.Tensor)
        if idx in aligned_set:
            # compact_aligned_load: slice dim 0 to one tile at tile_start via
            # pl.Element so Pallas prefetches/double-buffers it; other dims full.
            # Use the FULL compact_block (not min with the tensor length): the
            # body slices ``pl.ds(0, compact_block)``, and the ``padding=(0,
            # compact_block)`` lets the read overshoot the tensor end (handles a
            # short compact dimension).  Clamping the block here would mismatch the
            # body's pl.ds size and slice out of bounds.
            #
            # tile_start is the RAW compact offset (base + tile*block), NOT
            # sublane-aligned-down: packed rows can start at arbitrary
            # offsets.  pl.Element is exactly the mechanism that tolerates a
            # dynamic, possibly-unaligned start -- Mosaic lowers an element-indexed
            # block to a dynamic-offset (un)masked access, where a plain int block
            # dim would require an aligned, fixed-grid block.  Verified bitwise ==
            # fori_loop on unaligned/random offsets; this holds for both the load
            # and the full-block store.
            block = compact_block
            elt = pl.Element(block, padding=(0, block))  # type: ignore[union-attr]
            block_shape = (elt, *(pl.Element(s) for s in t.shape[1:]))  # type: ignore[union-attr]

            def aligned_index_map(
                wid: object,
                *scalar_refs: object,
                _pos: int = tile_start_ref_pos,
                _nd: int = t.ndim,
            ) -> tuple[object, ...]:
                tile_start = scalar_refs[_pos][wid]  # type: ignore[index]
                return (tile_start, *(jnp.int32(0) for _ in range(_nd - 1)))  # type: ignore[union-attr]

            return pl.BlockSpec(block_shape, aligned_index_map)  # type: ignore[union-attr]
        if idx in ordered_aligned_set:
            # Resident caching: per-range resident window sized ``ordered_window`` (C)
            # at ``range_start`` -- the fori body reads it at the local ordered-tile
            # offset (offset - range_start).  padding=(0, C) tolerates reads past
            # the range tail (same as the compact_aligned_load window).  Keying
            # on range_start lets Pallas dedup the load across same-range tiles.
            assert ordered_window > 0
            assert range_start_ref_pos >= 0
            oblock = ordered_window
            oelt = pl.Element(oblock, padding=(0, oblock))  # type: ignore[union-attr]
            oblock_shape = (
                oelt,
                *(pl.Element(s) for s in t.shape[1:]),  # type: ignore[union-attr]
            )

            def ordered_index_map(
                wid: object,
                *scalar_refs: object,
                _pos: int = range_start_ref_pos,
                _nd: int = t.ndim,
            ) -> tuple[object, ...]:
                start = scalar_refs[_pos][wid]  # type: ignore[index]
                return (
                    start,
                    *(jnp.int32(0) for _ in range(_nd - 1)),  # type: ignore[union-attr]
                )

            return pl.BlockSpec(  # type: ignore[union-attr]
                oblock_shape, ordered_index_map
            )
        entry = block_spec_info[arg_to_tpos[idx]] if block_spec_info else None
        if entry is not None:
            block_shape_template, grid_dims = entry
            if any(isinstance(g, int) for g in grid_dims):
                block_shape = tuple(
                    min(bs, t.shape[d]) if bs is not None else t.shape[d]
                    for d, bs in enumerate(block_shape_template)
                )

                def index_map(
                    wid: object,
                    *scalar_refs: object,
                    _gd: tuple[object, ...] = grid_dims,
                    _pos: int = owner_ref_pos,
                ) -> tuple[object, ...]:
                    owner = scalar_refs[_pos][wid]  # type: ignore[index]
                    return tuple(
                        owner if g == 0 else jnp.int32(0)  # type: ignore[union-attr]
                        for g in _gd
                    )

                mem = pltpu.SMEM if idx in smem_set else None  # type: ignore[union-attr]
                return pl.BlockSpec(block_shape, index_map, memory_space=mem)  # type: ignore[union-attr]
        return _pallas_make_block_spec(pl, jnp, pltpu, t, entry, idx in smem_set)

    in_specs = [_spec_for(idx) for idx in tensor_arg_indices]
    out_list = [_spec_for(idx) for idx in output_indices]
    out_specs = out_list if len(out_list) > 1 else out_list[0]
    return in_specs, out_specs


def _pallas_make_compact_reordered_kernel(
    pallas_kernel: object,
    args: tuple[object, ...],
    tensor_arg_indices: list[int],
    non_tensor_args: dict[int, object],
    n_tensor_inputs: int,
    _output_indices: list[int],
    n_scalar_prefetch: int,
) -> object:
    """Reordered kernel for the compact-worklist launcher.

    The launcher passes refs as ``[scalar_refs..., inputs..., outputs..., scratch...]``;
    the generated device function expects
    ``(inputs..., outputs..., scratch..., metadata_refs...)`` (the metadata refs
    are ``wrapper_only_params``, appended last).  Strip the N leading scalar refs
    and re-append them after scratch.
    """

    def reordered_kernel(*refs: object) -> None:
        scalar_refs = refs[:n_scalar_prefetch]
        body_refs = refs[n_scalar_prefetch:]
        n_kernel_params = len(args)
        original_order: list[object] = [None] * n_kernel_params
        for tensor_pos, orig_pos in enumerate(tensor_arg_indices):
            original_order[orig_pos] = body_refs[tensor_pos]
        for orig_pos, value in non_tensor_args.items():
            original_order[orig_pos] = value
        for out_idx, orig_pos in enumerate(_output_indices):
            original_order[orig_pos] = body_refs[n_tensor_inputs + out_idx]
        scratch_refs = body_refs[n_tensor_inputs + len(_output_indices) :]
        pallas_kernel(*original_order, *scratch_refs, *scalar_refs)  # type: ignore[operator]

    return reordered_kernel


def _pallas_compile_compact_jit_fn(
    pallas_kernel: object,
    args: tuple[object, ...],
    *,
    _output_indices: list[int],
    _inplace_indices: list[int] | None,
    _block_spec_info: _BlockSpecInfo | None,
    _scratch_shapes: list[object] | None,
    _smem_arg_indices: list[int] | None,
    _hbm_arg_indices: list[int] | None,
    build_worklist: Callable[..., object],
    offset_arg_indices: list[int],
    metadata_fields: list[str],
    owner_ref_pos: int,
    num_scalar_prefetch: int,
    aligned_arg_indices: list[int] | None = None,
    tile_start_ref_pos: int = 1,
    compact_block: int = 1,
    ordered_aligned_arg_indices: list[int] | None = None,
    range_start_ref_pos: int = -1,
    ordered_window: int = 0,
    interpret: bool = False,
) -> _PallasCompileResult:
    """Build the compact-worklist jit_fn: build metadata in-jit -> dynamic grid."""
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

    scratch_shapes = _pallas_build_scratch_shapes(pltpu, jnp, _scratch_shapes or [])
    smem_set = set(_smem_arg_indices or [])
    hbm_set = set(_hbm_arg_indices or [])
    in_specs, out_specs = _pallas_compact_in_out_specs(
        pl,
        jnp,
        pltpu,
        args,
        tensor_arg_indices,
        _output_indices,
        _block_spec_info,
        smem_set,
        hbm_set,
        owner_ref_pos,
        set(aligned_arg_indices or []),
        tile_start_ref_pos,
        compact_block,
        set(ordered_aligned_arg_indices or []),
        range_start_ref_pos,
        ordered_window,
    )
    reordered_kernel = _pallas_make_compact_reordered_kernel(
        pallas_kernel,
        args,
        tensor_arg_indices,
        non_tensor_args,
        n_tensor_inputs,
        _output_indices,
        num_scalar_prefetch,
    )
    out_shape_arg = out_shapes if len(out_shapes) > 1 else out_shapes[0]
    # NOTE: the shared _pallas_check_vmem_or_raise estimator does not yet
    # understand pl.Element block shapes (compact_aligned_load), so it is not
    # applied here; adding pl.Element support to the estimator is a follow-up.
    # Offsets-tensor positions within the tensor-arg list (jit_fn input order).
    offset_tpos = [arg_to_tensor_pos[i] for i in offset_arg_indices]

    def jit_fn(*jax_inputs: object) -> object:
        offsets = [jax_inputs[tp] for tp in offset_tpos]
        metadata = build_worklist(*offsets)
        num_work = metadata.num_work  # type: ignore[attr-defined]
        scalar_prefetch = [getattr(metadata, f) for f in metadata_fields]
        call = pl.pallas_call(  # type: ignore[union-attr]
            reordered_kernel,  # pyrefly: ignore[bad-argument-type]
            out_shape=out_shape_arg,
            grid_spec=pltpu.PrefetchScalarGridSpec(  # type: ignore[union-attr]
                num_scalar_prefetch=num_scalar_prefetch,
                # num_work is a traced scalar; for an empty batch (total == 0) it
                # is 0, i.e. a dynamic grid=(0,).  Verified that Mosaic accepts the
                # zero-grid launch and returns the empty output (no special-case
                # skip needed).
                grid=(num_work,),
                in_specs=in_specs,
                out_specs=out_specs,
                scratch_shapes=scratch_shapes,  # pyrefly: ignore[bad-argument-type]
            ),
            compiler_params=pltpu.CompilerParams(  # pyrefly: ignore[bad-instantiation]
                # "arbitrary" (NOT "parallel") is load-bearing for correctness,
                # for two reasons:
                #
                # 1. Input reuse: Pallas reuses an unchanged owner-indexed input
                #    block across consecutive same-owner work items -- the builder
                #    orders work items by owner, so all compact tiles for one owner
                #    share the same owner-indexed fetches.
                #
                # 2. Ordered-overwrite store (the precondition the whole kernel
                #    rests on): each work item's VALID output rows are disjoint,
                #    BUT the store is a masked FULL-block pl.Element write, so a
                #    partial last tile overwrites the next owner's leading rows
                #    with masked-zero padding (owners are packed contiguously at
                #    unaligned offsets, so the write regions overlap even though
                #    valid-row ownership does not).  "arbitrary" tells
                #    Mosaic the grid iterations may have dependencies, so it runs
                #    them sequentially in grid order rather than reordering or
                #    pipelining them.  The builder emits work items in ascending
                #    (owner, tile) order, so the next owner's first tile is a
                #    LATER iteration that re-writes those rows with the correct
                #    values -- and being later + serial, it deterministically
                #    wins.  With "parallel", Mosaic could reorder/overlap
                #    iterations and the spill would race the re-write.
                #
                # KNOWN RISK: this rests on Mosaic running an "arbitrary" 1-D grid
                # strictly sequentially in ascending order and never reordering
                # it.  That is the documented meaning of
                # "arbitrary" and is bitwise-verified vs fori today, but it is a
                # scheduling-contract dependency -- if a future Mosaic relaxes it,
                # the spilling store would silently corrupt.  The robust fix is the
                # deferred exact pl.ds(tile_start, tile_extent) / emit_pipeline +
                # BoundedSlice store; a future correctness-mode toggle could select
                # it for validation.  Detection also now restricts to packed (so
                # work order == row order); see detect_compact_worklist_plan.
                dimension_semantics=("arbitrary",),
                # Resident caching sizes its physical window from _get_vmem_limit_bytes
                # during backend setup.  This 128MiB floor is ONLY a Mosaic compile
                # ceiling so TPU7x accepts that already-sized allocation; do not use
                # it to choose C.  A streamed compact_worklist kernel keeps the
                # platform default ceiling.
                vmem_limit_bytes=(
                    max(_get_vmem_limit_bytes(pltpu), 128 * 1024 * 1024)
                    if ordered_aligned_arg_indices
                    else _get_vmem_limit_bytes(pltpu)
                ),
            ),
            interpret=interpret,
        )
        return call(*scalar_prefetch, *jax_inputs)

    return _PallasCompileResult(
        jit_fn=jit_fn,
        tensor_arg_indices=tensor_arg_indices,
        output_only_indices=output_only_indices,
        arg_to_tensor_pos=arg_to_tensor_pos,
        inplace_positions=inplace_positions,
        pallas_aliases=pallas_aliases,
    )


def _pallas_install_compact_launcher_cache(
    pallas_kernel: object,
    grid: tuple[int, ...],
    args: tuple[object, ...],
    *,
    _output_indices: list[int] | None,
    _inplace_indices: list[int] | None,
    _block_spec_info: _BlockSpecInfo | None,
    _smem_arg_indices: list[int] | None,
    _scratch_shapes: list[object] | None,
    _hbm_arg_indices: list[int] | None,
    _ds_pad_dims: list[tuple[int, int, int, int]] | None,
    _pallas_interpret: bool | None,
    _compact_build_worklist: Callable[..., object],
    _compact_offset_arg_indices: list[int] | None,
    _compact_metadata_fields: list[str] | None,
    _compact_owner_ref_pos: int,
    _compact_num_scalar_prefetch: int,
    _compact_aligned_arg_indices: list[int] | None,
    _compact_tile_start_ref_pos: int,
    _compact_block: int,
    # Resident-cache (owner-cache) compile params; default to inactive so a
    # non-resident compact kernel compiles unchanged.
    _compact_ordered_aligned_arg_indices: list[int] | None = None,
    _compact_range_start_ref_pos: int = -1,
    _compact_ordered_window: int = 0,
) -> tuple[object, ...]:
    """Cache-miss path for compact-worklist Pallas kernels.

    Mirror of :func:`_pallas_install_launcher_cache`, but calls
    :func:`_pallas_compile_compact_jit_fn` (which builds the worklist
    metadata in-jit from the offset args, then feeds the traced
    ``num_work`` to a dynamic ``grid=(num_work,)`` with scalar-prefetch
    metadata).  Compact needs its own compile function because the grid
    is dynamic; everything else — JaxCallable wrap, cache slot,
    ``_LauncherFastPath`` seed, downstream invoke path — is identical
    to the standard install.
    """
    interpret = (
        _pallas_interpret
        if _pallas_interpret is not None
        else _module_is_pallas_interpret()
    )
    if interpret:
        _ensure_cpu_tpu_info()
    output_indices = _output_indices if _output_indices is not None else []

    spec_args = args
    if _ds_pad_dims:
        spec_args, _ = _pallas_apply_ds_padding(args, output_indices, _ds_pad_dims)
    _pallas_check_dtypes(spec_args)

    result = _pallas_compile_compact_jit_fn(
        pallas_kernel,
        spec_args,
        _output_indices=output_indices,
        _inplace_indices=_inplace_indices,
        _block_spec_info=_block_spec_info,
        _scratch_shapes=_scratch_shapes,
        _smem_arg_indices=_smem_arg_indices,
        _hbm_arg_indices=_hbm_arg_indices,
        build_worklist=_compact_build_worklist,
        offset_arg_indices=_compact_offset_arg_indices or [],
        metadata_fields=_compact_metadata_fields or [],
        owner_ref_pos=_compact_owner_ref_pos,
        num_scalar_prefetch=_compact_num_scalar_prefetch,
        aligned_arg_indices=_compact_aligned_arg_indices or [],
        tile_start_ref_pos=_compact_tile_start_ref_pos,
        compact_block=_compact_block,
        ordered_aligned_arg_indices=_compact_ordered_aligned_arg_indices or [],
        range_start_ref_pos=_compact_range_start_ref_pos,
        ordered_window=_compact_ordered_window,
        interpret=interpret,
    )

    jax_callable = _pallas_build_callable(
        pallas_kernel,
        grid,
        cast("Callable[..., object]", result.jit_fn),
        output_indices,
        result.arg_to_tensor_pos,
        result.tensor_arg_indices,
        cache_attr=_PALLAS_CACHE_ATTR,
        call_aliases=result.pallas_aliases,
        trace_key_suffix="",
        interpret=interpret,
    )
    fast_path = _LauncherFastPath(
        result.tensor_arg_indices,
        result.arg_to_tensor_pos,
        output_indices,
        _ds_pad_dims,
    )
    cache = (
        grid,
        jax_callable,
        result.tensor_arg_indices,
        result.arg_to_tensor_pos,
        fast_path,
        None,
    )
    setattr(pallas_kernel, _PALLAS_CACHE_ATTR, cache)
    return cache


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
    num_sm: int | None = None,
) -> None:
    def plan_int(key: str, default: int | None = None) -> int:
        value = plan.get(key, default) if default is not None else plan[key]
        assert isinstance(value, int)
        return value

    def plan_optional_int(key: str) -> int | None:
        value = plan.get(key)
        assert value is None or isinstance(value, int)
        return value

    def plan_optional_str(key: str) -> str | None:
        value = plan.get(key)
        assert value is None or isinstance(value, str)
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
        epi_tile_raw_expr: str | None = None,
    ) -> None:
        assert len(kernel_args) == 2
        explicit_epi_tile = any(
            value is not None for value in (epi_tile_m, epi_tile_n, d_store_box_n)
        )
        if epi_tile_raw_expr is not None:
            # The bm=128 CtaGroup.TWO family threads the device-exact (N-mode
            # permuted) epilogue-tile expression verbatim so the host TMA-store
            # atom is built from the same layout the device r2s copy writes
            # through. The plain ``epi_tile_m/n`` integer keys cannot express
            # the permutation. See ``tcgen05_two_cta_m128_epilogue_tile_expr``.
            assert not explicit_epi_tile
            epi_tile_expr = epi_tile_raw_expr
        elif explicit_epi_tile:
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
    if kind == "helion_small_biased_attention":
        batch = plan_int("batch")
        seq = plan_int("seq")
        body.extend(
            [
                f"    grid_x = cutlass.Int32({seq})",
                f"    grid_y = cutlass.Int32({batch})",
                "    grid_z = cutlass.Int32(1)",
            ]
        )
        return

    if kind == "helion_flash":
        # Fused tcgen05 flash-attention host setup: reorder Helion's (B, S, D)
        # tensors to the reference (S, D, B) / (D, S, B) layouts, build the two
        # tiled_mma (QK from SMEM, PV with OperandSource.TMEM) and the three TMA
        # atoms, then append all kernel args. Mirrors
        # ``.notes/spikes/fa_tcgen05_spike.py`` host_function (3D-batched variant
        # validated standalone).
        q_idx = plan_int("q_idx")
        k_idx = plan_int("k_idx")
        v_idx = plan_int("v_idx")
        o_idx = plan_int("o_idx")
        lse_idx = plan_optional_int("lse_idx")
        bias_idx = plan_optional_int("bias_idx")
        alibi_idx = plan_optional_int("alibi_idx")
        document_idx = plan_optional_int("document_idx")
        seq = plan_int("seq")
        head_dim = plan_int("head_dim")
        batch = plan_int("batch")
        scale_log2 = plan["scale_log2"]
        assert isinstance(scale_log2, float)
        score_bias_scale = plan.get("score_bias_scale", 0.0)
        assert isinstance(score_bias_scale, float)
        alibi_count = plan_int("alibi_count", default=batch)
        document_batch = plan_int("document_batch", default=batch)
        document_heads_per_batch = plan_int("document_heads_per_batch", default=1)
        kv_stage = plan_int("kv_stage")
        q_stage = plan_int("q_stage", default=1)
        num_kv = (seq + 127) // 128
        # Static-persistent scheduler: total_tiles = num_bh * num_m_tiles (the
        # flat tile-id space the device-body strided while loop walks). When
        # persistent, the host clamps grid_x down to min(total_tiles, num_SMs)
        # so each SM gets one CTA that strides over many work tiles.
        persistent = bool(plan.get("persistent"))
        total_tiles = plan_int("total_tiles", default=batch * (seq // 128))
        pass_dynamic_tile_counts = plan.get("topology") != "fa4"
        hd = head_dim
        dtype = str(plan.get("dtype", "cutlass.Float16"))
        assert dtype in ("cutlass.Float16", "cutlass.BFloat16")
        # (S, D, B) views over the existing (B, S, D) row-major buffers.
        bw = "cutlass.utils.blackwell_helpers"
        qkd = f"(128, 128, {hd})"
        pvd = f"(128, {hd}, 128)"
        sdb = f"cute.make_layout(({seq}, {hd}, {batch}), stride=({hd}, 1, {seq * hd}))"
        dsb = f"cute.make_layout(({hd}, {seq}, {batch}), stride=(1, {hd}, {seq * hd}))"
        ssb = (
            f"cute.make_layout(({seq}, {seq}, {batch}), stride=({seq}, 1, {seq * seq}))"
        )
        sb = f"cute.make_layout(({seq}, {batch}), stride=(1, {seq}))"
        majk = "cute.nvgpu.OperandMajorMode.K"
        cg1 = "cute.nvgpu.tcgen05.CtaGroup.ONE"
        sel = "cute.select"
        flash_lines = [
            f"_flash_mQ = cute.make_tensor(arg{q_idx}.iterator, {sdb})",
            f"_flash_mK = cute.make_tensor(arg{k_idx}.iterator, {sdb})",
            # V is MN-major: (D, S, B).
            f"_flash_mV = cute.make_tensor(arg{v_idx}.iterator, {dsb})",
            f"_flash_mO = cute.make_tensor(arg{o_idx}.iterator, {sdb})",
            f"_flash_qk_mma = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majk}, {majk}, cutlass.Float32, {cg1}, (128, 128))",
            f"_flash_pv_mma = {bw}.make_trivial_tiled_mma({dtype}, {dtype}, {majk}, cute.nvgpu.OperandMajorMode.MN, cutlass.Float32, {cg1}, (128, {hd}), cute.nvgpu.tcgen05.OperandSource.TMEM)",
            f"_flash_qsl = {bw}.make_smem_layout_a(_flash_qk_mma, {qkd}, {dtype}, {q_stage})",
            # K/V are multi-stage TMA rings (Stage 3); the stage count must match
            # the device-body kv_stage + the SharedStorage MemRange depths.
            f"_flash_ksl = {bw}.make_smem_layout_b(_flash_qk_mma, {qkd}, {dtype}, {kv_stage})",
            f"_flash_vsl = {bw}.make_smem_layout_b(_flash_pv_mma, {pvd}, {dtype}, {kv_stage})",
            f"_flash_ptl = {bw}.make_smem_layout_a(_flash_pv_mma, {pvd}, {dtype}, 1)",
            f"_flash_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp({cg1})",
            f"_flash_tma_q, _flash_mQt = cute.nvgpu.make_tiled_tma_atom_A(_flash_op, _flash_mQ, {sel}(_flash_qsl, mode=[0, 1, 2]), {qkd}, _flash_qk_mma)",
            f"_flash_tma_k, _flash_mKt = cute.nvgpu.make_tiled_tma_atom_B(_flash_op, _flash_mK, {sel}(_flash_ksl, mode=[0, 1, 2]), {qkd}, _flash_qk_mma)",
            f"_flash_tma_v, _flash_mVt = cute.nvgpu.make_tiled_tma_atom_B(_flash_op, _flash_mV, {sel}(_flash_vsl, mode=[0, 1, 2]), {pvd}, _flash_pv_mma)",
            f"_flash_scale_log2 = cutlass.Float32({scale_log2!r})",
            f"_flash_num_kv_tiles = cutlass.Int32({num_kv})",
        ]
        if bias_idx is not None:
            flash_lines.extend(
                [
                    f"_flash_mBias = cute.make_tensor(arg{bias_idx}.iterator, {ssb})",
                    f"_flash_score_bias_scale = cutlass.Float32({score_bias_scale!r})",
                ]
            )
        if alibi_idx is not None:
            flash_lines.extend(
                [
                    (
                        f"_flash_mAlibi = cute.make_tensor(arg{alibi_idx}.iterator, "
                        f"cute.make_layout(({alibi_count},), stride=(1,)))"
                    ),
                    f"_flash_num_alibi = cutlass.Int32({alibi_count})",
                ]
            )
        if document_idx is not None:
            sdoc = f"cute.make_layout(({seq}, {document_batch}), stride=(1, {seq}))"
            flash_lines.extend(
                [
                    f"_flash_mDoc = cute.make_tensor(arg{document_idx}.iterator, {sdoc})",
                    (
                        "_flash_doc_heads_per_batch = "
                        f"cutlass.Int32({document_heads_per_batch})"
                    ),
                ]
            )
        if pass_dynamic_tile_counts:
            flash_lines.extend(
                [
                    f"_flash_num_bh = cutlass.Int32({batch})",
                    f"_flash_total_tiles = cutlass.Int32({total_tiles})",
                ]
            )
        if lse_idx is not None:
            flash_lines.append(
                f"_flash_mLSE = cute.make_tensor(arg{lse_idx}.iterator, {sb})"
            )
        epi_tma = bool(plan.get("epi_tma"))
        if epi_tma:
            # Lever A: build the O TMA STORE atom (fa4-only). The O smem layout is
            # the per-Q-tile (128, hd) epilogue tile, 2-staged so BOTH adjacent
            # Q-tiles' outputs fit (matches the fa4 q_stage=2 sQ region the corr
            # epilogue reuses). ``make_tiled_tma_atom(S2G, ...)`` returns the
            # TMA-adjusted mO as ``_flash_mOt`` (mirrors the Q/K/V load atoms).
            otile = f"(128, {hd})"
            flash_lines.extend(
                [
                    (
                        f"_flash_osl = {bw}.make_smem_layout_epi("
                        f"{dtype}, cutlass.utils.layout.LayoutEnum.ROW_MAJOR, {otile}, 2)"
                    ),
                    (
                        f"_flash_o_cta_v = cute.composition("
                        f"cute.make_identity_layout(_flash_mO.shape), {otile})"
                    ),
                    (
                        "_flash_tma_o, _flash_mOt = "
                        "cute.nvgpu.cpasync.make_tiled_tma_atom("
                        "cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), _flash_mO, "
                        "cute.select(_flash_osl, mode=[0, 1]), _flash_o_cta_v)"
                    ),
                ]
            )
        else:
            # mO stays the (S, D, B) view (no TMA atom; the epilogue uses
            # autovec_copy straight to gmem).
            flash_lines.append("_flash_mOt = _flash_mO")
        body.extend(f"    {line}" for line in flash_lines)
        if persistent:
            # Cap the flat grid at num_SMs (computed host-side from the q tensor's
            # device at wrapper-build time and baked as a literal). grid_y/grid_z
            # stay 1 (already true for the flat flash grid). The device-body
            # strided while loop then covers all total_tiles work items.
            assert num_sm is not None and num_sm > 0
            grid_cap = min(total_tiles, num_sm)
            body.append(f"    grid_x = cutlass.Int32({grid_cap})")
        elif plan.get("topology") == "fa4":
            # The fa4 topology processes a PAIR of adjacent 128-row Q-tiles per
            # CTA, so it needs exactly total_tiles (= batch * seq // 256) CTAs.
            # The default root grid would launch batch * seq // 128; override it
            # to the halved fa4 tile count.
            body.append(f"    grid_x = cutlass.Int32({total_tiles})")
        call_args.extend(
            [
                "_flash_qk_mma",
                "_flash_pv_mma",
                "_flash_tma_q",
                "_flash_mQt",
                "_flash_tma_k",
                "_flash_mKt",
                "_flash_tma_v",
                "_flash_mVt",
                "_flash_mOt",
                "_flash_qsl",
                "_flash_ksl",
                "_flash_vsl",
                "_flash_ptl",
                "_flash_scale_log2",
                "_flash_num_kv_tiles",
            ]
        )
        if pass_dynamic_tile_counts:
            call_args.extend(["_flash_num_bh", "_flash_total_tiles"])
        if lse_idx is not None:
            call_args.append("_flash_mLSE")
        if bias_idx is not None:
            call_args.extend(["_flash_mBias", "_flash_score_bias_scale"])
        if alibi_idx is not None:
            call_args.extend(["_flash_mAlibi", "_flash_num_alibi"])
        if document_idx is not None:
            call_args.extend(["_flash_mDoc", "_flash_doc_heads_per_batch"])
        if epi_tma:
            call_args.extend(["_flash_tma_o", "_flash_osl"])
        return
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
            epi_tile_raw_expr=plan_optional_str("epi_tile_raw_expr"),
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
    # K-major (column-major / K-contiguous) B. Absent on the MN-major
    # (row-major B) default path.
    b_k_major = bool(plan.get("b_k_major"))
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
    #
    # The bm=128 CtaGroup.TWO family (fp8 small-grid) cannot be derived from
    # ``bm == 256`` here, so the device codegen records the resolved decision
    # on the plan as ``use_2cta_instrs``. Honor it when present; fall back to
    # the legacy bm==256 derivation for golden-stable older plans.
    plan_use_2cta = plan.get("use_2cta_instrs")
    if plan_use_2cta is not None:
        assert isinstance(plan_use_2cta, bool)
        use_2cta_instrs = plan_use_2cta
    else:
        use_2cta_instrs = cluster_m == 2 and bm == 256
    cta_group = (
        "cute.nvgpu.tcgen05.CtaGroup.TWO"
        if use_2cta_instrs
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
        b_k_major=b_k_major,
    )
    body.extend(
        (
            (
                f"    {tiled_mma} = cutlass.utils.blackwell_helpers.make_trivial_tiled_mma("
                f"{input_dtype}, "
                f"{input_dtype}, "
                "cute.nvgpu.OperandMajorMode.K, "
                + (
                    "cute.nvgpu.OperandMajorMode.K, "
                    if b_k_major
                    else "cute.nvgpu.OperandMajorMode.MN, "
                )
                + f"{acc_dtype}, "
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
            # B is viewed as (N, K). For row-major B (MN-major) the N axis
            # (position 0) is contiguous; for column-major B (K-major, native
            # fp8 layout) the K axis (position 1) is contiguous.
            f"    {rhs_tma}.mark_layout_dynamic(leading_dim={1 if b_k_major else 0})",
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
    num_sm: int | None = None,
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
        _append_cute_wrapper_plan(body, call_args, plan, num_sm=num_sm)
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
    # The fa4 flash topology (16-warp/512-thread) uses ``cute.arch.setmaxregister``
    # for per-warp register reallocation (softmax warps inc to 200; mma/corr/load/empty
    # dec). ptxas only emits the ``EIATTR_REG_RECONFIG`` that HONORS those ``setmaxnreg``
    # ops when the kernel declares ``min_blocks_per_mp`` (>= 1); WITHOUT it ptxas
    # SILENTLY DROPS every setmaxnreg and all warps are stuck at the static uniform
    # split -- so the softmax warp never reaches its 200-reg grant and spills its
    # resident row to local memory. fa4 already pins 1 CTA/SM (512 threads + TMEM = 1
    # tcgen05 unit/SM + smem near the cap), so ``min_blocks_per_mp=1`` matches its real
    # occupancy and enables the reallocation (=1 avoids the smem-carveout path >1 would
    # trigger). NOT applied to ws_overlap (256-thread): forcing 1 CTA/SM there cuts its
    # 2-blocks/SM occupancy and regresses it ~4pp.
    if any(plan.get("topology") == "fa4" for plan in wrapper_plans):
        launch_suffix += ", min_blocks_per_mp=1"
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


class _CompiledCuteLauncher:
    """Lazily compile a Helion ``@cute.jit`` wrapper via ``cute.compile``.

    The first call uses ``cute.compile(jit_func, *args)`` to produce a compiled
    callable; subsequent calls invoke the compiled callable directly. This
    bypasses the per-launch ``@cute.jit`` argument-handling/dispatch path,
    matching Quack's pattern (see ``gemm_tvm_ffi_utils.py``). On B200 this
    collapses ~200ms of per-launch host overhead into ~0.1ms.

    When ``cache_key`` is provided, the lowered IR module of the compiled
    kernel is persisted under ``CUTE_DSL_CACHE_DIR`` and reloaded on a later
    process, skipping recompilation.  ``cute.compile`` forces the CuTe DSL's
    own ``no_cache=True`` path, so Helion drives the on-disk cache itself: it
    writes the post-pass ``ir_module`` bytecode (plus a small JSON sidecar
    holding the mangled entry symbol) and, on a hit, reconstructs a runnable
    ``CudaDialectJitCompiledFunction`` by JIT-loading the stored module.
    Any failure in the cache layer falls back to a plain ``cute.compile``.
    """

    __slots__ = ("_cache_key", "_compile_options", "_compiled", "_jit_func")

    def __init__(
        self,
        jit_func: object,
        compile_options: str | None,
        cache_key: str | None = None,
    ) -> None:
        self._jit_func = jit_func
        self._compile_options = compile_options
        self._compiled: object = None
        self._cache_key = cache_key

    def __call__(self, *args: object) -> object:
        compiled = self._compiled
        if compiled is None:
            import cutlass.cute as cute

            compiled = None
            if self._cache_key is not None:
                compiled = self._reload_from_disk()
            if compiled is None:
                if self._compile_options is None:
                    compiled = cute.compile(self._jit_func, *args)
                else:
                    compiled = cute.compile(
                        self._jit_func,
                        *args,
                        options=self._compile_options,
                    )
                if self._cache_key is not None:
                    self._persist_to_disk(compiled)
            self._compiled = compiled
        return cast("Any", compiled)(*args)

    def persist_compiled(self) -> None:
        """Persist the already-compiled module into the current on-disk cache dir.

        Used by ``finalize_ephemeral_cache``: the artifact written during
        autotuning died with the ephemeral dir, but the compiled module is
        still in memory and ``_cache_file_paths`` resolves the destination
        from the (now restored) ``CUTE_DSL_CACHE_DIR`` at call time.
        """
        if self._cache_key is not None and self._compiled is not None:
            self._persist_to_disk(self._compiled)

    def _cache_file_paths(self) -> tuple[str, str, str]:
        from cutlass.base_dsl.cache_helpers import get_default_generated_ir_path

        cache_dir = get_default_generated_ir_path("CUTE_DSL")
        mlir = os.path.join(cache_dir, f"cute_dsl_{self._cache_key}.mlir")
        meta = os.path.join(cache_dir, f"cute_dsl_{self._cache_key}.json")
        return cache_dir, mlir, meta

    def _persist_to_disk(self, compiled: object) -> None:
        try:
            from cutlass.base_dsl.cache_helpers import save_ir
            from cutlass.base_dsl.cache_helpers import write_bytecode_with_crc32

            ir_module = getattr(compiled, "ir_module", None)
            function_name = getattr(compiled, "function_name", None)
            if ir_module is None or function_name is None:
                return
            cache_dir, _mlir, meta = self._cache_file_paths()
            os.makedirs(cache_dir, exist_ok=True)
            save_ir(
                "CUTE_DSL",
                ir_module,
                str(self._cache_key),
                output_dir=cache_dir,
                as_bytecode=True,
                bytecode_writer=lambda f: write_bytecode_with_crc32(f, ir_module),
            )
            # Atomic sidecar with the mangled entry symbol (process-dependent,
            # so it cannot be recomputed and must be stored alongside the IR).
            tmp = f"{meta}.tmp.{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(
                    {
                        "function_name": function_name,
                        "has_gpu_module": bool(
                            getattr(compiled, "has_gpu_module", True)
                        ),
                    },
                    f,
                )
            os.replace(tmp, meta)
        except (ImportError, OSError):
            # Old cutlass or an unwritable cache dir; just recompile next time.
            log.debug(
                "CuTe disk-cache persist failed for key %s",
                self._cache_key,
                exc_info=True,
            )

    def _reload_from_disk(self) -> object:
        try:
            from cutlass.base_dsl.cache_helpers import load_ir
            from cutlass.base_dsl.cache_helpers import read_bytecode_and_check_crc32
            from cutlass.cutlass_dsl.cuda_jit_executor import (
                CudaDialectJitCompiledFunction,
            )
            from cutlass.cutlass_dsl.cutlass import CuTeDSL

            _cache_dir, mlir, meta = self._cache_file_paths()
            if not (os.path.exists(mlir) and os.path.exists(meta)):
                return None
            with open(meta) as f:
                metadata = json.load(f)
            function_name = metadata["function_name"]
            # The parsed Module holds an internal reference to the ir.Context
            # that load_ir opened, so it stays valid after load_ir returns even
            # though its ``with ir.Context()`` block has already exited.
            _, module = load_ir(
                mlir,
                asBytecode=True,
                bytecode_reader=read_bytecode_and_check_crc32,
            )
            dsl = CuTeDSL._get_dsl()
            engine = dsl.compiler_provider.jit(
                module, shared_libs=dsl.get_shared_libs()
            )
            capi_func = engine.lookup(function_name)
            # The signature is reconstructable from the wrapper, so it does not
            # need to be persisted.
            wrapped = getattr(self._jit_func, "__wrapped__", self._jit_func)
            signature = inspect.signature(cast("Any", wrapped), eval_str=True)
            # Empty kernel_info / default extra-arg state is correct only for the
            # non-experimental ``cute.compile`` path Helion uses here; the
            # experimental DSL would populate these from module attributes.
            return CudaDialectJitCompiledFunction(
                module,
                engine,
                capi_func,
                signature,
                function_name,
                {},
                False,
                None,
                has_gpu_module=bool(metadata.get("has_gpu_module", True)),
            )
        except Exception:
            # Any cutlass-internal change or corrupt artifact -> recompile.
            return None


_TVM_FFI_COMPILE_OPTION = "--enable-tvm-ffi"


def _merge_tvm_ffi_compile_option(compile_options: str | None) -> str:
    """Ensure ``--enable-tvm-ffi`` is present in *compile_options*.

    The generic launcher always benefits from the FFI bridge (it skips
    CUTLASS-DSL's per-arg cast/pointer work). Other flags such as
    ``--generate-line-info`` may already be present (e.g. when the
    autotuner picks ``tcgen05_cubin_lineinfo=True``), so we splice rather
    than replace.
    """
    if compile_options is None:
        return _TVM_FFI_COMPILE_OPTION
    tokens = compile_options.split()
    if _TVM_FFI_COMPILE_OPTION in tokens:
        return compile_options
    tokens.append(_TVM_FFI_COMPILE_OPTION)
    return " ".join(tokens)


def _get_compiled_cute_launcher(
    cute_kernel: object,
    schema_key: tuple[tuple[object, ...], ...],
    block: tuple[int, int, int],
    compile_options: str | None = None,
    arch_args: tuple[object, ...] | None = None,
) -> object:
    # Always ensure ``--enable-tvm-ffi`` is present on the generic launcher
    # path: the generated wrapper signature (``cute.Pointer`` + scalars) is
    # TVM-FFI compatible and the FFI bridge bypasses CUTLASS-DSL's per-arg
    # cast/pointer work in ``generate_execution_args``. We merge rather
    # than replace because other flags (e.g. ``--generate-line-info`` when
    # ``tcgen05_cubin_lineinfo`` is True) can already be in
    # ``compile_options``.
    compile_options = _merge_tvm_ffi_compile_option(compile_options)
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
    # Persistent flash kernels bake the device SM count into the wrapper grid
    # clamp; resolve it from the first tensor arg's device so the cache key (and
    # the baked literal) stay device-correct across GPUs.
    num_sm: int | None = None
    if arch_args is not None:
        for arg in arch_args:
            if isinstance(arg, torch.Tensor) and arg.device.type == "cuda":
                num_sm = get_num_sm(arg.device)
                break
    cache_key = (
        schema_key,
        block,
        wrapper_plans,
        repr(cluster_shape),
        compile_options,
        num_sm,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    if arch_args is not None:
        _ensure_cute_dsl_arch_env(arch_args)
    jit_func = _create_cute_wrapper(cute_kernel, schema_key, block, num_sm=num_sm)
    disk_cache_key = _cute_disk_cache_key(
        cute_kernel,
        schema_key,
        block,
        wrapper_plans,
        cluster_shape,
        compile_options,
        num_sm,
    )
    launcher = _CompiledCuteLauncher(
        jit_func, compile_options, cache_key=disk_cache_key
    )
    cache[cache_key] = launcher
    return launcher


def _cute_cache_relevant_env() -> tuple[tuple[str, str], ...]:
    """Return CuTe DSL env vars that can change the compiled IR.

    The CuTe DSL folds *every* one of its ``CUTE_DSL_*`` env vars into its own
    module hash (e.g. ``CUTE_DSL_ENABLE_ASSERTIONS``, ``CUTE_DSL_LINEINFO``,
    ``CUTE_DSL_KEEP``, the tvm-ffi flags), so any of them can alter the
    persisted artifact.  We snapshot the whole set (so future flags are covered
    too) and only exclude the cache *location* ``CUTE_DSL_CACHE_DIR`` — that
    selects where artifacts live (autotuning uses an ephemeral dir) and must not
    affect the key.  Including an env var that does not actually affect codegen
    only costs an occasional missed cache hit, never a wrong-kernel reload.
    """
    return tuple(
        sorted(
            (k, v)
            for k, v in os.environ.items()
            if k.startswith("CUTE_DSL_") and k != "CUTE_DSL_CACHE_DIR"
        )
    )


def _cute_disk_cache_key(
    cute_kernel: object,
    schema_key: tuple[tuple[object, ...], ...],
    block: tuple[int, int, int],
    wrapper_plans: tuple[object, ...],
    cluster_shape: object,
    compile_options: str | None,
    num_sm: int | None = None,
) -> str | None:
    """Compute a stable cross-process key for the on-disk CuTe compile cache.

    Returns ``None`` (disabling the on-disk cache) when the generated-source
    hash is unavailable.  The key must be computable *before* the kernel is
    compiled (so a hit can skip recompilation), so it is derived from the
    inputs that determine the lowered IR rather than from the IR itself:
    generated device-kernel source, full input specialization (dtypes, ranks,
    baked shapes/strides, constexpr values), launch shape (block/cluster), CuTe
    compile options, the IR-affecting ``CUTE_DSL_*`` env vars (target SM arch
    among them), and the cutlass version.

    ``num_sm`` is the device SM count the persistent flash wrapper bakes into
    its grid clamp as a literal (``cute.compile`` lowers that literal into the
    persisted ``ir_module``).  The env-var arch capture only distinguishes the
    target *arch*, not the SM *count*, so two same-arch GPUs with different SM
    counts would otherwise collide on one on-disk artifact carrying the wrong
    grid clamp.  It is included unconditionally to match the in-memory cache
    key; for non-persistent kernels num_sm does not affect codegen, so it only
    costs an occasional cross-GPU miss, never a wrong-kernel reload.
    """
    source_hash = getattr(cute_kernel, "_helion_cute_source_hash", None)
    if source_hash is None:
        return None
    try:
        import cutlass

        cutlass_version = getattr(cutlass, "__version__", "")
    except Exception:
        cutlass_version = ""
    payload = repr(
        (
            "helion-cute-cache-v1",
            source_hash,
            schema_key,
            block,
            wrapper_plans,
            repr(cluster_shape),
            compile_options or "",
            _cute_cache_relevant_env(),
            cutlass_version,
            num_sm,
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return base64.b32encode(digest).decode().rstrip("=")


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


def _cute_current_stream() -> object:
    """Sample the *current* CUDA stream for a cute kernel launch.

    Must be called fresh on every launch and never cached: under CUDA graph
    capture ``torch.cuda.current_stream()`` is redirected to a dedicated capture
    stream, so a stream baked into the cached launch args (during eager warmup)
    would make the kernel launch on the wrong, non-capturing stream — the graph
    then records no work and replays as a no-op (empty-graph capture). Sampling
    here keeps the launch on whatever stream is current at call time.
    """
    _gmem, _make_ptr, current_stream_obj = _get_cute_launcher_imports()
    return cast("Any", current_stream_obj)()


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
    # NOTE: the returned launch args deliberately EXCLUDE the CUDA stream. The
    # stream is the only launch arg that is not a pure function of
    # (grid, tensor metadata, scalars), so it must not be baked into the cached
    # args — the caller appends a freshly sampled ``_cute_current_stream()`` on
    # every launch (see ``default_cute_launcher``). Caching the stream would
    # break CUDA graph capture (empty-graph / no-op replay).
    gmem_space, make_ptr_obj, _current_stream_obj = _get_cute_launcher_imports()
    make_ptr = cast("Any", make_ptr_obj)
    constexpr_flags = _cute_kernel_param_is_constexpr(cute_kernel)
    # Kernels that emit cute MMA ops (universal matmul fallback or tcgen05
    # TMA wrapper plans) need runtime tensor layouts: the wrapper's
    # ``cute.make_tensor`` feeds into ``.mark_layout_dynamic`` (TMA path) or
    # into in-kernel arithmetic that relies on dynamic shape/stride
    # propagation (universal MMA SMEM-load guards). Baking literal shapes
    # silently miscompiles those paths.
    if bake_tensor_shapes:
        any_obj = cast("Any", cute_kernel)
        wrapper_plans = getattr(any_obj, "_helion_cute_wrapper_plans", None)
        wrapper_plans_disable_bake = bool(wrapper_plans) and not all(
            plan.get("kind") == "helion_small_biased_attention"
            for plan in wrapper_plans
        )
        disable_bake = bool(
            getattr(any_obj, "_helion_cute_disable_bake_tensor_shapes", False)
            or wrapper_plans_disable_bake
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
    # The stream is intentionally NOT appended here; it is sampled fresh per
    # launch by the caller so CUDA graph capture sees the capture stream.
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
    # Append the CUDA stream fresh on every launch (never cached): under CUDA
    # graph capture the current stream is the capture stream, so the kernel must
    # be issued there and not on a stale stream baked into the cached args.
    return cast("Any", compiled)(*launch_args, _cute_current_stream())


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
