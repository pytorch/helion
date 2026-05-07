"""Multi-cell read-modify-write accumulator for Pallas atomic targets.

`hl.atomic_*` on Pallas is not a hardware atomic; the TensorCore has no
concurrent writers to a given HBM cell.  The semantics we need are
"accumulate across grid cells of the same output tile".  This module
plans and wraps that pattern at the launcher level.

The flow:

1. :func:`_grid_rmw_plan` inspects the atomic target list and the
   ``_BlockSpecInfo`` from compile time.  For each target it finds grid
   axes that are unmapped to the target's tile and have ``grid[i] > 1``;
   those drive accumulation and become ``"arbitrary"``.  Returns
   ``(dim_sem, scratch_tiles)``.

2. The launcher feeds ``set(scratch_tiles)`` into ``skip_inplace_copy``
   so the reordered kernel's HBM->VMEM refresh doesn't clobber the
   accumulator.

3. :func:`_grid_rmw_apply` allocates a persistent VMEM scratch per RMW
   target (f32 for bf16/f16 to avoid per-cell rounding) and wraps the
   kernel: on the first ``"arbitrary"`` cell, copy HBM into the scratch;
   substitute the scratch ref for the out_ref inside the kernel body;
   on the last cell, commit the scratch back to HBM.

4. The launcher passes the wrap to ``pl.pallas_call`` with the returned
   scratches in ``scratch_shapes`` and ``dim_sem`` in
   ``compiler_params.dimension_semantics``.

Single TensorCore only.  ``pl.core_map`` would need a cross-core barrier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

if TYPE_CHECKING:
    from . import _BlockSpecInfo


def _grid_rmw_plan(
    grid: tuple[int, ...],
    args: tuple[object, ...],
    rmw_indices: list[int] | None,
    block_spec_info: _BlockSpecInfo | None,
    arg_to_tensor_pos: dict[int, int],
) -> tuple[tuple[str, ...], dict[int, tuple[int, ...]]]:
    """Plan VMEM scratches for RMW outputs hit by multiple grid cells.

    Returns ``(dim_sem, scratch_tiles)``.  ``scratch_tiles`` maps each
    ``orig_pos`` that needs a scratch to its tile shape; empty when no
    output needs accumulation, in which case ``dim_sem`` is all
    ``"parallel"``.
    """
    n = len(grid)
    parallel = tuple("parallel" for _ in grid)
    if not rmw_indices or block_spec_info is None:
        return parallel, {}

    arb: set[int] = set()
    scratch_tiles: dict[int, tuple[int, ...]] = {}
    for orig_pos in rmw_indices:
        tpos = arg_to_tensor_pos.get(orig_pos)
        if tpos is None or tpos >= len(block_spec_info):
            continue
        info = block_spec_info[tpos]
        if info is None:
            continue
        bshape, grid_mapping = info
        mapped = {gd for gd in grid_mapping if isinstance(gd, int)}
        unmapped = {i for i in range(n) if i not in mapped and grid[i] > 1}
        if not unmapped:
            continue
        arb |= unmapped
        t = args[orig_pos]
        assert isinstance(t, torch.Tensor)
        scratch_tiles[orig_pos] = tuple(
            bs if bs is not None else t.shape[d] for d, bs in enumerate(bshape)
        )
    if not scratch_tiles:
        return parallel, {}
    dim_sem = tuple("arbitrary" if i in arb else "parallel" for i in range(n))
    return dim_sem, scratch_tiles


def _grid_rmw_apply(
    inner: object,
    scratch_tiles: dict[int, tuple[int, ...]],
    dim_sem: tuple[str, ...],
    args: tuple[object, ...],
    arg_to_tensor_pos: dict[int, int],
    output_indices: list[int],
    n_tensor_inputs: int,
    grid: tuple[int, ...],
    pltpu: object,
) -> tuple[object, list[object]]:
    """Wrap *inner* with cross-cell scratch substitution.

    Returns ``(wrapped_kernel, extras)``.  No-op when ``scratch_tiles``
    is empty: returns ``(inner, [])``.

    bf16/f16 targets get an f32 scratch so per-cell sums don't round.
    The atomic codegen casts the value to ``prev.dtype`` so the in-kernel
    RMW stays in scratch dtype.  ``convert_element_type`` is a no-op
    when dtypes match.
    """
    if not scratch_tiles:
        return inner, []

    from torch._inductor.runtime.runtime_utils import torch_dtype_to_jax_runtime

    extras: list[object] = []
    out_to_extra: dict[int, int] = {}
    out_to_in: dict[int, int] = {}
    for out_idx, orig_pos in enumerate(output_indices):
        tile = scratch_tiles.get(orig_pos)
        if tile is None:
            continue
        t = cast("torch.Tensor", args[orig_pos])
        sd = torch.float32 if t.dtype in (torch.bfloat16, torch.float16) else t.dtype
        extras.append(pltpu.VMEM(tile, torch_dtype_to_jax_runtime(sd)))  # type: ignore[union-attr]
        out_to_extra[out_idx] = len(out_to_extra)
        out_to_in[out_idx] = arg_to_tensor_pos[orig_pos]

    arb_dims = [i for i, s in enumerate(dim_sem) if s == "arbitrary"]
    n_extras = len(extras)

    def wrapped(*refs: object) -> None:
        from jax import lax
        from jax.experimental import pallas as pl
        import jax.numpy as jnp

        inner_end = len(refs) - n_extras
        scratches = {oi: refs[inner_end + off] for oi, off in out_to_extra.items()}
        patched = list(refs[:inner_end])
        originals: dict[int, object] = {}
        for oi, s in scratches.items():
            slot = n_tensor_inputs + oi
            originals[oi] = patched[slot]
            patched[slot] = s

        is_first: Any = jnp.bool_(True)
        is_last: Any = jnp.bool_(True)
        for d in arb_dims:
            is_first = is_first & (pl.program_id(d) == 0)
            is_last = is_last & (pl.program_id(d) == (grid[d] - 1))

        @pl.when(is_first)  # type: ignore[arg-type]
        def _init() -> None:
            for oi, s in scratches.items():
                in_ref = refs[out_to_in[oi]]
                s[...] = lax.convert_element_type(in_ref[...], s.dtype)  # type: ignore[index,attr-defined]

        inner(*patched)  # type: ignore[operator]

        @pl.when(is_last)  # type: ignore[arg-type]
        def _commit() -> None:
            for oi, s in scratches.items():
                originals[oi][...] = lax.convert_element_type(s[...], originals[oi].dtype)  # type: ignore[index,attr-defined]

    return wrapped, extras
