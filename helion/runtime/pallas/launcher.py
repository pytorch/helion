"""Pallas/TPU runtime launchers and their compile core.

Holds `default_pallas_launcher` (the torch-tensor / TorchTPU launcher that
generated Pallas code invokes) plus the shared block-spec / compile / caching
helpers it and the pure-JAX export path (`helion.runtime.pallas.jax_export`)
build on.

Depends only on ``torch`` and ``jax`` -- no other ``helion`` module -- so the
ahead-of-time precompiler can bulk-export this file verbatim into a standalone
kernel with zero Helion runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import inspect
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import Protocol
from typing import TypeGuard
from typing import cast

import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Mapping

    import jax


class _TorchTensorOrJaxArray(Protocol):
    """Structural type for a Pallas tensor arg -- a ``torch.Tensor`` or a
    ``jax.Array``. The shared compile core reads only ``shape``/``ndim``/``dtype``
    so it works with either, regardless of which launcher/mode drives it."""

    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def dtype(self) -> object: ...


# Dtypes Mosaic/Pallas cannot lower. Kept here (rather than imported from the
# compiler) so the launcher stays dependency-free; the compiler applies the same
# check at codegen time.
_PALLAS_UNSUPPORTED_DTYPES = frozenset({torch.int64, torch.uint64, torch.float64})


def _pallas_interpret_enabled() -> bool:
    """Whether Pallas interpret mode (CPU, no TPU) is on, per the
    ``HELION_PALLAS_INTERPRET`` env var.

    Dependency-free stand-in for ``helion.runtime.settings.is_pallas_interpret``
    at the launcher's call sites: the launcher runs at kernel-execution time,
    where no ``CompileEnvironment`` is active, so ``is_pallas_interpret`` already
    reduces to this env-var read (``Settings.pallas_interpret`` itself defaults
    from the same variable).
    """
    value = os.environ.get("HELION_PALLAS_INTERPRET")
    if value is None or (value := value.strip()) == "":
        return False
    return value.lower() in ("1", "true", "yes", "on", "y", "t")


def _is_torch_tensor_or_jax_array(x: object) -> TypeGuard[_TorchTensorOrJaxArray]:
    """True for a tensor-like kernel arg -- a ``torch.Tensor`` or a ``jax.Array``
    -- and False for Python scalars / other non-array args.

    Duck-typed so the shared compile core classifies args identically whether it
    is driven with torch tensors (torch launcher), ``_JaxExportTensor`` adapters
    (jax_fn runtime), or raw ``jax.Array``s (jax_fn precompiled standalone).
    """
    return hasattr(x, "shape") and hasattr(x, "dtype")


def _pallas_make_block_spec(
    pl: object,
    jnp: object,
    pltpu: object,
    tensor: _TorchTensorOrJaxArray,
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


def _get_vmem_limit_bytes(pltpu: object, interpret: bool) -> int:
    """Safely retrieves the TPU VMEM capacity without crashing on hardware locks.

    ``interpret`` picks the synthetic CPU TPU-info budget over a real-TPU query.
    Callers pass the compile-time ``settings.pallas_interpret`` (compiler) or the
    runtime env-var value (launcher), so this helper stays dependency-free.
    """
    global _CACHED_VMEM_LIMIT_BYTES
    if _CACHED_VMEM_LIMIT_BYTES is not None:
        return _CACHED_VMEM_LIMIT_BYTES

    # In interpret mode there is no real TPU; query the synthetic TPU info
    # registered by ``_ensure_cpu_tpu_info`` so the budget matches what real
    # TPU 7X reports rather than falling back to the conservative 16MB default.
    if interpret:
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
    resident window is double-buffered by Pallas.  ``prep_operands``
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
    VMEM budget allows (the logical bound ``C``).  The resident window, optional
    prep-cache scratch, and the refill/reduction ``pl.ds`` slices are all tiled by
    the ordered block, so the allocation must be a block multiple.  Round the
    budget DOWN to a block multiple so the allocation never exceeds the VMEM
    budget; cap by the operand extent rounded UP to one block so a short tensor
    still gets a whole block rather than a zero-sized allocation -- including an
    operand with no rows, which keeps resident caching (and therefore
    ``pallas_loop_type='unroll'``) admissible for an all-empty ordered
    reduction.

    Returns 0 when the budget cannot hold one ordered block. The selected loop
    policy decides whether that is an invalid resident config or irrelevant to a
    streamed config.
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
    # Floored to 1 so an operand with NO rows still admits a one-block window:
    # returning 0 would make the resident decision inactive, and
    # pallas_loop_type='unroll' rejects that outright, so a legal all-empty
    # ordered reduction would stop compiling.  The window is made safe instead by
    # padding one dummy row for such an operand (see _compute_pad_info).
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
        # ``dtype.itemsize`` works for torch.dtype, jnp.dtype and np.dtype, so
        # this stays torch-free -- letting the jax-only standalone inline the
        # VMEM check unchanged.
        itemsize = getattr(getattr(t, "dtype", None), "itemsize", None)
        return itemsize if itemsize is not None else 4

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
        assert _is_torch_tensor_or_jax_array(t)
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
        assert _is_torch_tensor_or_jax_array(t)
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
        assert _is_torch_tensor_or_jax_array(t)
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


def _pallas_jax_placeholder(a: object) -> object:
    """A ``jax.ShapeDtypeStruct`` output placeholder from a ``jax.Array``'s
    shape/dtype.

    Torch-free -- the default placeholder for the compile core, used by the
    jax_fn runtime and the jax_fn precompiled standalone. Typed ``object`` so it
    unifies with ``_pallas_torch_placeholder`` under one placeholder callable.
    """
    import jax

    arr = cast("_TorchTensorOrJaxArray", a)
    return jax.ShapeDtypeStruct(tuple(int(s) for s in arr.shape), arr.dtype)


def _pallas_torch_placeholder(a: object, *, interpret: bool) -> object:
    """Output placeholder for the torch launcher: ``torch_tpu``'s
    ``jax_placeholder`` on TPU (the exact layout its JaxCallable dispatch needs),
    else a dtype-mapped ``ShapeDtypeStruct`` fallback (interpret / no torch_tpu).
    """
    t = cast("torch.Tensor", a)
    if not interpret:
        try:
            from torch_tpu._internal.pallas.pallas import (  # pyrefly: ignore[missing-import]
                jax_placeholder,
            )

            return jax_placeholder(t)
        except ImportError:
            pass
    return _jax_placeholder_for_tensor(t)


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
    mlir_fingerprint: str
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
    mlir_fingerprint: str,
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
                mlir_fingerprint,
                inputs=input_tensors,
                output_shapes=output_shapes,
                donate_argnums=donate_argnums,
            )
            return out_tree.unflatten(results)  # type: ignore[attr-defined]

        return invoke_no_alias

    def invoke_with_alias(input_tensors: list[object]) -> object:
        results = call_custom_kernel(  # type: ignore[operator]
            kernel_name,
            mlir_fingerprint,
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
            mlir_fingerprint = self.kernel_key_to_mlir_fingerprint[kernel_key]
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
                mlir_fingerprint,
                output_shapes,
                self.donate_argnums,
                out_tree,
                alias_items,
            )
            self._helion_direct_call = _DirectCallKernel(
                call_custom_kernel=tpu_torch_pallas.call_custom_kernel,
                kernel_name=self.name,
                mlir_fingerprint=mlir_fingerprint,
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
    placeholder_fn: Callable[[object], object] | None = None,
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
    # Default to the torch-free jax placeholder; the torch launcher injects its
    # torch_tpu placeholder. ``interpret`` is retained for signature stability.
    if placeholder_fn is None:
        placeholder_fn = _pallas_jax_placeholder

    output_set = set(_output_indices)
    inplace_set = set(_inplace_indices) if _inplace_indices is not None else output_set
    output_only = output_set - inplace_set

    all_tensor_positions = [
        i for i in range(len(args)) if _is_torch_tensor_or_jax_array(args[i])
    ]
    output_only_indices = [i for i in all_tensor_positions if i in output_only]
    tensor_arg_indices = [i for i in all_tensor_positions if i not in output_only]

    non_tensor_args: dict[int, object] = {
        i: args[i]
        for i in range(len(args))
        if not _is_torch_tensor_or_jax_array(args[i])
    }
    n_tensor_inputs = len(tensor_arg_indices)
    arg_to_tensor_pos = {orig: tpos for tpos, orig in enumerate(tensor_arg_indices)}
    inplace_positions = output_set & set(tensor_arg_indices)
    out_shapes = tuple(placeholder_fn(args[i]) for i in _output_indices)

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
        from jax._src.pallas.mosaic.tpu_info import get_tpu_info_for_chip
        from jax._src.pallas.mosaic.tpu_info import registry
    except ImportError:
        return
    if "cpu" not in registry:
        registry["cpu"] = lambda: get_tpu_info_for_chip(ChipVersion.TPU_7X, 1)


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
            precision="default",
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
            scratch_shapes.append(pltpu.SemaphoreType.DMA(shape))  # type: ignore[union-attr]
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
    vmem_limit_bytes = _get_vmem_limit_bytes(pltpu, _pallas_interpret_enabled())
    if estimated_vmem > vmem_limit_bytes:
        raise RuntimeError(
            f"XLA:TPU compile permanent error. Ran out of memory in memory space vmem. "
            f"Estimated {estimated_vmem / 1e6:.2f}MB exceeds {vmem_limit_bytes / 1e6:.2f}MB vmem capacity."
        )


def _pallas_kernel_scratch_kwarg(pl: object) -> str:
    """Name of ``pl.kernel``'s scratch-types kwarg for the installed pallas.

    jax renamed it ``scratch_shapes`` -> ``scratch_types`` in 0.10.1; accept
    whichever the installed pallas exposes so Helion runs on both (e.g. a TPU
    serve pinned to jax 0.10.0). Used by every ``pl.kernel`` call site (simple,
    pipelined, and compact-worklist) so all launch paths stay version-robust.
    """
    return (
        "scratch_types"
        if "scratch_types" in inspect.signature(pl.kernel).parameters  # type: ignore[union-attr]
        else "scratch_shapes"
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
    input_output_aliases: dict[int, int],
    interpret: bool,
    collective_id: int | None,
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
    out_shape_seq = (
        list(out_shape_arg)
        if n_outputs > 1 and isinstance(out_shape_arg, (list, tuple))
        else [out_shape_arg]
    )
    hbm_alias_input_by_output = {
        out_pos: in_pos
        for in_pos, out_pos in input_output_aliases.items()
        if in_pos in hbm_in_positions and out_pos in hbm_out_positions
    }
    hbm_alias_input_positions = set(hbm_alias_input_by_output.values())
    kernel_input_positions = [
        pos for pos in range(n_inputs) if pos not in hbm_alias_input_positions
    ]
    kernel_output_positions = [
        pos for pos in range(n_outputs) if pos not in hbm_alias_input_by_output
    ]
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

    mesh = pltpu.create_tensorcore_mesh("_helion_core", num_cores=1)  # type: ignore[union-attr]
    scratch_kw = _pallas_kernel_scratch_kwarg(pl)

    # ``jax.new_ref`` models physical HBM aliases for real TPU execution, but
    # interpret-mode refs do not expose the same memory-space behavior. Remote
    # DMA itself is unsupported in interpret mode, so retain the direct IO path
    # used before HBM aliasing was added.
    if interpret:

        def interpret_kernel_body(*refs: object) -> None:
            io_any = refs[:n_io]
            scratch_refs = refs[n_io:]
            pipe_any = [io_any[p] for p in pipe_positions]

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

        return pl.kernel(  # type: ignore[union-attr]
            interpret_kernel_body,
            out_shape_arg,
            mesh=mesh,
            interpret=True,
            **{scratch_kw: scratch_shapes},
        )

    kernel_out_shapes = [out_shape_seq[pos] for pos in kernel_output_positions]
    kernel_out_shape: object
    if not kernel_out_shapes:
        kernel_out_shape = ()
    elif len(kernel_out_shapes) == 1:
        kernel_out_shape = kernel_out_shapes[0]
    else:
        kernel_out_shape = tuple(kernel_out_shapes)

    def make_kernel(alias_refs: Mapping[int, object]) -> object:
        def kernel_body(*refs: object) -> None:
            input_refs = iter(refs[: len(kernel_input_positions)])
            output_start = len(kernel_input_positions)
            output_end = output_start + len(kernel_output_positions)
            output_refs = iter(refs[output_start:output_end])
            scratch_refs = refs[output_end:]

            io_any: list[object] = []
            for position in range(n_inputs):
                io_any.append(
                    alias_refs[position]
                    if position in hbm_alias_input_positions
                    else next(input_refs)
                )
            for position in range(n_outputs):
                alias_input = hbm_alias_input_by_output.get(position)
                io_any.append(
                    alias_refs[alias_input]
                    if alias_input is not None
                    else next(output_refs)
                )
            assert len(io_any) == n_io
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

        kernel_kwargs: dict[str, object] = {scratch_kw: scratch_shapes}
        if collective_id is not None:
            kernel_kwargs["compiler_params"] = pltpu.CompilerParams(  # type: ignore[union-attr]
                collective_id=collective_id
            )
        return pl.kernel(  # type: ignore[union-attr]
            kernel_body,
            kernel_out_shape,
            mesh=mesh,
            interpret=interpret,
            **kernel_kwargs,
        )

    if not hbm_alias_input_positions:
        return make_kernel({})

    import jax

    def ref_jit_fn(*inputs: object) -> object:
        alias_refs = {
            position: jax.new_ref(inputs[position])
            for position in hbm_alias_input_positions
        }
        kernel_inputs = [inputs[position] for position in kernel_input_positions]
        kernel = cast("Callable[..., object]", make_kernel(alias_refs))
        kernel_results = kernel(*kernel_inputs)
        if not kernel_output_positions:
            kernel_results_seq: list[object] = []
        elif len(kernel_output_positions) == 1:
            kernel_results_seq = [kernel_results]
        else:
            kernel_results_seq = list(cast("Iterable[object]", kernel_results))

        result_iter = iter(kernel_results_seq)
        results = []
        for position in range(n_outputs):
            alias_input = hbm_alias_input_by_output.get(position)
            results.append(
                alias_refs[alias_input][...]
                if alias_input is not None
                else next(result_iter)
            )
        return results[0] if len(results) == 1 else tuple(results)

    return ref_jit_fn


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
    _collective_id: int | None,
    interpret: bool,
    placeholder_fn: Callable[[object], object] | None = None,
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
        args,
        _output_indices,
        _inplace_indices,
        interpret=interpret,
        placeholder_fn=placeholder_fn,
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
        assert _block_spec_info is not None, (
            "pallas pipeline / scratch kernels require _block_spec_info from codegen"
        )
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
                assert _is_torch_tensor_or_jax_array(t)
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
            input_output_aliases=pallas_aliases,
            interpret=interpret,
            collective_id=_collective_id,
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
_PALLAS_SCRATCH_KEY_ATTR = "_pallas_scratch_key"


def _pallas_jax_call(
    pallas_kernel: object,
    grid: tuple[int, ...],
    jax_args: tuple[object, ...],
    *,
    output_indices: list[int],
    inplace_indices: list[int] | None,
    block_spec_info: _BlockSpecInfo | None,
    scratch_shapes: list[object] | None,
    hbm_arg_indices: list[int] | None,
    smem_arg_indices: list[int] | None,
    collective_id: int | None,
    interpret: bool,
    compact: dict[str, object] | None = None,
    orig_shapes: dict[int, tuple[int, ...]] | None = None,
    ds_pad_dims: list[tuple[int, int, int, int]] | None = None,
    return_all_outputs: bool = False,
) -> list[object]:
    """Drive the shared compile core (``pl.kernel``) + jit_fn on raw ``jax.Array``s
    and return the output JAX array(s).

    The single JAX launch path shared by the jax_fn runtime launcher
    (``default_pallas_jax_launcher``, which wraps this with adapter unwrap/rewrap)
    and the jax_fn precompiled standalone -- neither touches torch. ``compact``
    carries the compact-worklist kwargs when the kernel uses that lowering.
    """
    if compact is not None:
        if collective_id is not None:
            raise RuntimeError(
                "Pallas remote_barrier is not supported by compact_worklist kernels"
            )
        result = _pallas_compile_compact_jit_fn(
            pallas_kernel,
            jax_args,
            _output_indices=output_indices,
            _inplace_indices=inplace_indices,
            _block_spec_info=block_spec_info,
            _scratch_shapes=scratch_shapes,
            _smem_arg_indices=smem_arg_indices,
            _hbm_arg_indices=hbm_arg_indices,
            interpret=interpret,
            **cast("dict[str, Any]", compact),
        )
    else:
        result = _pallas_compile_jit_fn(
            pallas_kernel,
            grid,
            jax_args,
            _output_indices=output_indices,
            _inplace_indices=inplace_indices,
            _block_spec_info=block_spec_info,
            _smem_arg_indices=smem_arg_indices,
            _scratch_shapes=scratch_shapes,
            _hbm_arg_indices=hbm_arg_indices,
            _matmul_dot_general=None,
            _collective_id=collective_id,
            interpret=interpret,
        )

    jax_inputs = [jax_args[i] for i in result.tensor_arg_indices]
    jax_results = result.jit_fn(*jax_inputs)  # type: ignore[operator]
    if not isinstance(jax_results, (tuple, list)):
        jax_results = (jax_results,)

    # In-place positions alias back into the caller's buffer on the torch path,
    # but JAX has no in-place mutation -- when every output is in-place, surface
    # them as fresh values (mirrors the torch fast-path descriptor list).
    descriptors = ()
    if not return_all_outputs:
        descriptors = _pallas_output_only_descriptors(
            output_indices, result.arg_to_tensor_pos
        )
    if return_all_outputs or not descriptors:
        descriptors = tuple(enumerate(output_indices))

    output_results: list[object] = [jax_results[out_idx] for out_idx, _ in descriptors]
    output_orig_pos: list[int] = [orig_pos for _, orig_pos in descriptors]

    # Slice padded outputs back to their original shapes (ds-pad), reusing the
    # torch fast-path's arg->padded-dims grouping; JAX arrays index identically.
    if ds_pad_dims and orig_shapes:
        padded_dims_by_arg = _pallas_padded_output_dims_by_arg(
            ds_pad_dims, set(orig_shapes.keys())
        )
        for i, orig_pos in enumerate(output_orig_pos):
            dims = padded_dims_by_arg.get(orig_pos)
            orig_shape = orig_shapes.get(orig_pos)
            if dims and orig_shape is not None:
                output_results[i] = _pallas_slice_to_orig(
                    cast("torch.Tensor", output_results[i]),
                    dims,
                    cast("torch.Size", orig_shape),
                )

    return output_results


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
    _collective_id: int | None,
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
        else _pallas_interpret_enabled()
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
        _collective_id=_collective_id,
        interpret=interpret,
        placeholder_fn=functools.partial(
            _pallas_torch_placeholder, interpret=interpret
        ),
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
    _collective_id: int | None = None,
    _uses_remote_copy: bool = False,
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
    """
    if _compact_build_worklist is not None and _collective_id is not None:
        raise RuntimeError(
            "Pallas remote_barrier is not supported by compact_worklist kernels"
        )
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
    scratch_key = tuple(
        (tuple(shape), dtype, kind) for shape, dtype, kind in (_scratch_shapes or [])
    )
    cache = getattr(pallas_kernel, _PALLAS_CACHE_ATTR, None)
    if (
        cache is None
        or cache[0] != grid
        or getattr(pallas_kernel, _PALLAS_SCRATCH_KEY_ATTR, None) != scratch_key
    ):
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
                _collective_id=_collective_id,
                _matmul_dot_general=_matmul_dot_general,
            )
        setattr(pallas_kernel, _PALLAS_SCRATCH_KEY_ATTR, scratch_key)

    return _pallas_invoke_cached_launcher(
        pallas_kernel,
        cache,
        args,
        cache_attr=_PALLAS_CACHE_ATTR,
        _ds_pad_dims=_ds_pad_dims,
    )


def _compact_window_block_spec(
    t: object,
    window: int,
    ref_pos: int,
    scalar_refs: tuple[object, ...],
) -> object:
    """BlockSpec for one compact-worklist window: up to ``window`` rows of dim 0
    at the runtime row offset in ``scalar_refs[ref_pos]``, other dims full.

    The offset is a RAW compact row (an owner base plus a tile step), never
    aligned down to a sublane boundary, so the block dim must accept a dynamic,
    possibly-unaligned start.

    ``pl.BoundedSlice`` accepts one AND lets the index map choose the transfer
    SIZE, which is what keeps the window inside the tensor.  That matters
    because a compact/jagged tensor is normally allocated with exactly
    ``sum(lengths)`` rows, so a fixed full-window transfer runs off the end on
    the last window of the final range -- a BoundsCheck core halt on
    ``dma.hbm_to_vmem``, plus a matching store past the output.  Pallas does not
    prevent that by itself: it clamps a block dim against the array only when
    that dim is TILED (jax's Mosaic ``_create_bounded_slice`` returns
    ``ds(start, size)`` verbatim when the dim has no tiling, which is the case
    for the leading dim of a ``(rows, ...)`` tensor).  A ``pl.Element`` window
    cannot express the clamp -- its size is fixed and its ``padding=`` never
    reaches that computation -- and would instead need the CALLER's tensor
    padded, costing a full copy of every windowed operand.

    The START is clamped too, and both clamps are held at or above zero:
    worklist entries past ``num_work`` are padded with a repeated owner but an
    unbounded group offset, so they can record a start beyond the tensor.  A
    resident operand with no rows of its own is given one dummy row instead (see
    ``_zero_row_resident_pad_info``), so the clamp always has an in-bounds row
    and this slice is never empty.

    The kernel body still slices ``pl.ds(0, window)`` off a window-sized ref and
    masks the tail, so a short transfer only leaves already-masked rows stale.
    """
    from jax.experimental import pallas as pl
    import jax.numpy as jnp

    ndim = int(t.ndim)  # type: ignore[attr-defined]
    block_shape = (pl.BoundedSlice(window), *t.shape[1:])  # type: ignore[union-attr,attr-defined]

    def index_map(
        wid: object,
        _pos: int = ref_pos,
        _nd: int = ndim,
        _rows: int = int(t.shape[0]),  # type: ignore[attr-defined]
        _window: int = window,
    ) -> tuple[object, ...]:
        start = scalar_refs[_pos][wid]  # type: ignore[index]
        start = jnp.clip(start, 0, max(_rows - 1, 0))
        size = jnp.clip(jnp.int32(_rows) - start, 0, _window)
        return (pl.ds(start, size), *(jnp.int32(0) for _ in range(_nd - 1)))

    return pl.BlockSpec(block_shape, index_map)  # type: ignore[union-attr]


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
    scalar_refs: tuple[object, ...],
    aligned_set: set[int] | None = None,
    tile_start_ref_pos: int = 1,
    compact_block: int = 1,
    ordered_aligned_set: set[int] | None = None,
    range_start_ref_pos: int = -1,
    ordered_window: int = 0,
) -> tuple[list[object], object]:
    """Build in/out BlockSpecs for the compact-worklist emit_pipeline.

    Like ``_pallas_build_pipeline_specs`` but: pipelined tensors -> HBM; an
    owner-indexed tensor (its ``grid_dims`` carry the owner grid dim ``0``) gets
    an ``index_map`` that reads ``owner_ids[wid]`` (a ``scalar_refs`` table); a
    compact-aligned-load tensor (``aligned_set``) gets a per-tile
    ``pl.BoundedSlice`` window at ``tile_start`` so Pallas double-buffers it
    across work items; everything else is full/SMEM.  ``scalar_refs`` are the SMEM refs
    holding the worklist metadata tables; every ``index_map`` receives only
    ``wid`` and closes over them.
    """
    aligned_set = aligned_set or set()
    ordered_aligned_set = ordered_aligned_set or set()
    all_positions = sorted(set(tensor_arg_indices) | set(output_indices))
    arg_to_tpos = {orig: tpos for tpos, orig in enumerate(all_positions)}

    def _spec_for(idx: int) -> object:
        if idx in hbm_set:
            return pl.BlockSpec(memory_space=pltpu.HBM)  # type: ignore[union-attr]
        t = args[idx]
        assert _is_torch_tensor_or_jax_array(t)
        if idx in aligned_set:
            # compact_aligned_load / compact_exact_store: one compact_block-row
            # window at tile_start, so Pallas prefetches and double-buffers both
            # the load and the store write-back across work items.  Window
            # sizing and the tensor-end hazard: _compact_window_block_spec.
            # Windowed on dim 0 only: emit_pipeline rejects it on the lane dim.
            return _compact_window_block_spec(
                t, compact_block, tile_start_ref_pos, scalar_refs
            )
        if idx in ordered_aligned_set:
            # Resident caching: per-range resident window sized ``ordered_window``
            # (C) at ``range_start`` -- the fori body reads it at the local
            # ordered-tile offset (offset - range_start).  Keying on range_start
            # lets Pallas dedup the load across same-range tiles.
            assert ordered_window > 0
            assert range_start_ref_pos >= 0
            return _compact_window_block_spec(
                t, ordered_window, range_start_ref_pos, scalar_refs
            )
        entry = block_spec_info[arg_to_tpos[idx]] if block_spec_info else None
        if entry is not None:
            block_shape_template, grid_dims = entry
            if any(isinstance(g, int) for g in grid_dims):
                block_shape = tuple(
                    max(min(bs, t.shape[d]) if bs is not None else t.shape[d], 1)
                    for d, bs in enumerate(block_shape_template)
                )

                def index_map(
                    wid: object,
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
    placeholder_fn: Callable[[object], object] | None = None,
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
        args,
        _output_indices,
        _inplace_indices,
        interpret=interpret,
        placeholder_fn=placeholder_fn,
    )

    scratch_shapes = _pallas_build_scratch_shapes(pltpu, jnp, _scratch_shapes or [])
    smem_set = set(_smem_arg_indices or [])
    hbm_set = set(_hbm_arg_indices or [])
    aligned_set = set(aligned_arg_indices or [])
    ordered_set = set(ordered_aligned_arg_indices or [])
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
    # understand the compact window's block shape (compact_aligned_load), so it
    # is not applied here; teaching the estimator about it is a follow-up.
    # Offsets-tensor positions within the tensor-arg list (jit_fn input order).
    offset_tpos = [arg_to_tensor_pos[i] for i in offset_arg_indices]

    n_inputs = len(tensor_arg_indices)
    n_outputs = len(_output_indices)
    n_kernel_scratch = len(scratch_shapes)
    hbm_in_positions = {i for i, idx in enumerate(tensor_arg_indices) if idx in hbm_set}
    hbm_out_positions = {i for i, idx in enumerate(_output_indices) if idx in hbm_set}

    # Flat positions within [inputs..., outputs...].
    n_io = n_inputs + n_outputs
    pass_positions = hbm_in_positions | {n_inputs + p for p in hbm_out_positions}
    pipe_positions = [p for p in range(n_io) if p not in pass_positions]

    def jit_fn(*jax_inputs: object) -> object:
        offsets = [jax_inputs[tp] for tp in offset_tpos]
        metadata = build_worklist(*offsets)
        num_work = metadata.num_work  # type: ignore[attr-defined]
        scalar_prefetch = [getattr(metadata, f) for f in metadata_fields]

        # The BlockSpec index maps and the generated kernel read the
        # worklist tables from SMEM.
        smem_types = [
            pltpu.SMEM(tuple(a.shape), a.dtype)  # type: ignore[union-attr]
            for a in scalar_prefetch
        ]
        all_scratch: list[object] = [*scratch_shapes, *smem_types]

        def kernel_body(*refs: object) -> None:
            scalar_any = refs[:num_scalar_prefetch]
            io_any = refs[num_scalar_prefetch : num_scalar_prefetch + n_io]
            rest = refs[num_scalar_prefetch + n_io :]
            kernel_scratch = rest[:n_kernel_scratch]
            scalar_smem = rest[n_kernel_scratch:]
            for src, dst in zip(scalar_any, scalar_smem, strict=True):
                pltpu.sync_copy(src, dst)  # type: ignore[union-attr]

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
                tuple(scalar_smem),
                aligned_set,
                tile_start_ref_pos,
                compact_block,
                ordered_set,
                range_start_ref_pos,
                ordered_window,
            )
            out_specs_seq = (
                list(out_specs) if isinstance(out_specs, (list, tuple)) else [out_specs]
            )
            all_specs = list(in_specs) + out_specs_seq
            pipe_in_specs = [all_specs[p] for p in pipe_positions if p < n_inputs]
            pipe_out_specs = [all_specs[p] for p in pipe_positions if p >= n_inputs]
            pipe_any = [io_any[p] for p in pipe_positions]

            def pipeline_body(*block_refs: object) -> None:
                merged = list(io_any)
                for p, block in zip(pipe_positions, block_refs, strict=True):
                    merged[p] = block
                reordered_kernel(*scalar_smem, *merged, *kernel_scratch)  # type: ignore[operator]

            # Correctness relies on emit_pipeline running grid steps
            # sequentially in ascending order (a later work item re-writes
            # the previous item's spilled store rows) and re-fetching an
            # input block only when its index map changes (owner-indexed
            # k/v reuse).
            pltpu.emit_pipeline(  # type: ignore[union-attr]
                pipeline_body,
                # num_work may be 0 (empty batch): zero steps, empty output.
                grid=(num_work,),
                in_specs=pipe_in_specs,
                out_specs=pipe_out_specs,
            )(*pipe_any)

        mesh = pltpu.create_tensorcore_mesh("_helion_core", num_cores=1)  # type: ignore[union-attr]
        call = pl.kernel(  # type: ignore[union-attr]
            kernel_body,
            out_shape_arg,
            mesh=mesh,
            # scratch kwarg name is version-dependent (scratch_shapes vs
            # scratch_types); pass whichever this pallas exposes.
            **{_pallas_kernel_scratch_kwarg(pl): all_scratch},  # pyrefly: ignore[bad-argument-type]
            compiler_params=pltpu.CompilerParams(  # pyrefly: ignore[bad-instantiation]
                # The device capacity is also the largest scoped-Vmem XLA will
                # accept, so never request more: a bare kernel call is clamped
                # down with a warning, but the same request inside a shard_map
                # is rejected outright (INVALID_ARGUMENT; see
                # https://openxla.org/xla/errors/error_0200).
                # Resident caching sizes its physical window conservatively
                # from this capacity during backend setup.
                vmem_limit_bytes=_get_vmem_limit_bytes(pltpu, interpret),
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
        else _pallas_interpret_enabled()
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
        placeholder_fn=functools.partial(
            _pallas_torch_placeholder, interpret=interpret
        ),
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
