"""JAX worklist builder for the ``compact_worklist`` Pallas loop type.

Pure JAX and unit-testable on CPU, with no dependency on the Helion compiler.

The generated, per-kernel ``_build_worklist(offsets...)`` (emitted by the
compiler, see ``helion/_compiler/pallas/compact_worklist.py``) computes the
per-owner ``base``/``length`` (and optional dependent ``dep_base``/``dep_len``)
arrays as ``jnp`` expressions, then calls :func:`flatten_worklist` here to turn
the ragged owner x tile product into a flat, padded worklist.  Everything runs
inside the launcher's internal ``jax.jit`` with **nothing ``.item()``-ed**, so
there is no device->host sync: ``num_work`` is a traced ``jax.Array`` fed
straight into ``grid=(num_work,)``; ``UPPER`` is static during JAX tracing and
may be derived from input shapes plus a user-provided tile extent bound.

The padding scheme mirrors tokamax's ``gmm`` (``jnp.repeat(...,
total_repeat_length=UPPER)`` + ``num_work = cnt.sum()`` fed to ``grid=``): the
metadata arrays have the static shape ``UPPER`` so every BlockSpec ``index_map``
is in-bounds by construction, while only the first ``num_work`` grid points run,
so the padded ``[num_work, UPPER)`` tail never executes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

from .._utils import cdiv

if TYPE_CHECKING:
    from jax import Array


class CompactWorkMetadata(NamedTuple):
    """Scalar-prefetch metadata describing one compact worklist.

    Each of the array fields has the static shape ``[UPPER]``; entry ``i``
    describes the ``i``-th compact tile (work item).  ``num_work`` is the
    *traced* number of work items actually produced (``<= UPPER``); only the
    first ``num_work`` grid points execute, so the padded tail of every array is
    never indexed.

    The dependent-range fields (``range_start``/``range_len``) are ``None`` for
    the dense-KV case (no ordered inner axis) and carry the raw per-work-item
    ordered range (e.g. ``kv_begin``/``kv_len``) for the fully jagged case.  The
    ordered length is stored **raw** -- never clamped up -- so the inner
    ``fori_loop`` trip count ``cdiv(kv_len, KV_BLOCK)`` is 0 for an empty range
    and the finalizer writes the identity accumulator.
    """

    owner_ids: Array  # int32[UPPER]   (per work item: which owner produced it)
    tile_starts: Array  # int32[UPPER] (absolute compact-tile start row)
    tile_extents: Array  # int32[UPPER] (valid rows in the tile, in [1, block])
    range_start: Array | None  # int32[UPPER] | None  (ordered range begin)
    range_len: Array | None  # int32[UPPER] | None    (ordered range length, RAW)
    num_work: Array  # int32[]  TRACED scalar (never .item()-ed)


def packed_upper_bound(total_compact: int, num_owners: int, block: int) -> int:
    """Static megablocks bound on the number of compact tiles, PACKED ranges only.

    Tight/safe **only when** owner ranges are packed/non-overlapping
    (``sum(lengths) == total_compact``): then each owner wastes at most one
    partial leading tile, so the worst case is ``cdiv(total_compact, block) +
    num_owners - 1`` (tokamax: ``tiles_m + num_groups - 1``).  ``total_compact``
    (e.g. ``total_q``) and ``num_owners`` (e.g. ``num_seq``) are static tensor
    shapes, so this is a static Python int and no per-owner max length is needed.

    For general (possibly-overlapping) ranges this bound can UNDER-count
    ``num_work``. A per-owner ``max_extent`` instead gives the conservative
    ``num_owners * cdiv(max_extent, block)`` bound; without one, compact-worklist
    detection requires packed ranges before using this helper.
    """
    return cdiv(total_compact, block) + num_owners - 1


def flatten_worklist(
    base: Array,
    length: Array,
    block: int,
    UPPER: int,
    *,
    dep_base: Array | None = None,
    dep_len: Array | None = None,
) -> CompactWorkMetadata:
    """Flatten a ragged ``owner x tile`` product into a padded compact worklist.

    ``base`` and ``length`` are ``int`` arrays of shape ``[P]`` (one entry per
    owner): ``base[p]`` is the absolute start row of owner ``p``'s compact slab
    and ``length[p]`` its row count.  The owner is split into
    ``cdiv(length[p], block)`` tiles; the flat list of all such tiles across all
    owners is the worklist.

    ``dep_base``/``dep_len`` (optional, ``[P]``) carry a per-owner ordered range
    (e.g. KV) that is gathered per work item and stored **raw** (never clamped).

    Returns a :class:`CompactWorkMetadata` whose arrays are padded to the static
    shape ``[UPPER]`` and whose ``num_work`` is a traced scalar.
    """
    # Local import (against the usual "no local imports" style) is deliberate: it
    # keeps this module importable on a host without JAX -- only the actual
    # builder call needs jnp, and that only runs inside the Pallas/TPU path.
    import jax.numpy as jnp

    base = jnp.asarray(base, jnp.int32)
    length = jnp.asarray(length, jnp.int32)
    block_i = jnp.int32(block)

    # Per-owner tile count -- parallel level, NOT clamped to a minimum of 1.
    cnt = (length + block_i - 1) // block_i  # [P]
    off = jnp.cumsum(cnt) - cnt  # exclusive scan: first work-item index per owner
    num_work = cnt.sum().astype(jnp.int32)  # TRACED scalar; never .item()-ed

    p = cnt.shape[0]
    # work[i] = owner that produced work item i; padded to UPPER with repeated
    # (valid) indices -- the [num_work, UPPER) tail is never indexed at runtime.
    # LOAD-BEARING: ``UPPER >= num_work`` (the caller's static bound) -- otherwise
    # total_repeat_length truncates the worklist and tiles are SILENTLY DROPPED.
    work = jnp.repeat(jnp.arange(p, dtype=jnp.int32), cnt, total_repeat_length=UPPER)

    # tile_in[i] = index of this tile within its owner (0-based).
    tile_in = jnp.arange(UPPER, dtype=jnp.int32) - off[work]
    tile_starts = (base[work] + tile_in * block_i).astype(jnp.int32)
    # Last tile of an owner is partial; clamp the extent to [.., block].  In the
    # valid range [0, num_work) this is always in [1, block]; clip(min=0) is
    # defensive for the padded tail (where length-tile_in*block can go negative),
    # which the runtime grid never indexes but keeps the array non-negative.
    tile_extents = jnp.clip(length[work] - tile_in * block_i, 0, block_i).astype(
        jnp.int32
    )

    range_start = None if dep_base is None else jnp.asarray(dep_base, jnp.int32)[work]
    range_len = None if dep_len is None else jnp.asarray(dep_len, jnp.int32)[work]

    return CompactWorkMetadata(
        owner_ids=work,
        tile_starts=tile_starts,
        tile_extents=tile_extents,
        range_start=range_start,
        range_len=range_len,
        num_work=num_work,
    )
