"""The CuTe slice resolver never binds a ``:`` to a tile position's block.

``_cute_resolve_active_slice_block_id`` (helion/language/memory_ops.py)
matches a full slice's block by EXTENT.  In the attention backward's
preprocess (``delta = sum(o * do)`` over the row beside
``dq[tile_b, tile_h, tile_m, :] = hl.zeros(...)``) with the batch extent
equal to the head dim under ``static_shapes``, the ``:`` over D matched the
batch tile's block too.  When the slice's own block's index was not active
at the store (a rolled reduction's exists only inside its ``for roffset``
loop; a persistent one's lanes closed before the store) the resolver fell
back to the one active candidate, the batch block, although the subscript's
first position already indexes it: the store's column became the batch
tile's index and one element per row was written, silently, delta correct
(b64 h12 s512 d64: 393,216 of 25,165,824 elements; the same with the
reduction persistent).  A block a tile position takes is never a slice's
now.  Alone, that rule rejected such a site loudly ("indexing dimension is
not active in this scope"); with the rolled block's chunk lane loop
(``_cute_make_rolled_replica``, the commit on top) the slice takes that
loop's index instead and the site lowers.  The GPU companion checks the
forms that lower store every element on both backends.
"""

from __future__ import annotations

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_lane_loop_distribution import _accesses
from test.test_cute_lane_loop_distribution import _escaping_names
from test.test_cute_lane_loop_distribution import _kernel_function
from test.test_cute_lane_loop_distribution import _loops

import helion
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


def _row_sum_then_zero_rows_fn(
    o: torch.Tensor, do: torch.Tensor, dq: torch.Tensor
) -> torch.Tensor:
    """The attention backward's preprocess shape: ``delta = sum(o * do)`` over
    the value head dim beside a full-slice zero of the query gradient's rows
    (another head dim, or the same one)."""
    B, H, M, _ = o.size()
    D = dq.size(-1)
    delta = torch.empty((B, H, M), dtype=torch.float32, device=o.device)
    for tile_b, tile_h, tile_m in hl.tile([B, H, M]):
        o_rows = o[tile_b, tile_h, tile_m, :].to(torch.float32)
        do_rows = do[tile_b, tile_h, tile_m, :].to(torch.float32)
        delta[tile_b, tile_h, tile_m] = torch.sum(o_rows * do_rows, dim=-1)
        dq[tile_b, tile_h, tile_m, :] = hl.zeros(
            [tile_b, tile_h, tile_m, D], dtype=dq.dtype
        )
    return delta


def _row_sum_then_fill_rows_fn(
    o: torch.Tensor, do: torch.Tensor, dq: torch.Tensor
) -> torch.Tensor:
    B, H, M, _ = o.size()
    D = dq.size(-1)
    delta = torch.empty((B, H, M), dtype=torch.float32, device=o.device)
    for tile_b, tile_h, tile_m in hl.tile([B, H, M]):
        o_rows = o[tile_b, tile_h, tile_m, :].to(torch.float32)
        do_rows = do[tile_b, tile_h, tile_m, :].to(torch.float32)
        delta[tile_b, tile_h, tile_m] = torch.sum(o_rows * do_rows, dim=-1)
        dq[tile_b, tile_h, tile_m, :] = hl.full(
            [tile_b, tile_h, tile_m, D], 1.0, dtype=dq.dtype
        )
    return delta


def _row_sum_then_scalar_rows_fn(
    o: torch.Tensor, do: torch.Tensor, dq: torch.Tensor
) -> torch.Tensor:
    B, H, M, _ = o.size()
    delta = torch.empty((B, H, M), dtype=torch.float32, device=o.device)
    for tile_b, tile_h, tile_m in hl.tile([B, H, M]):
        o_rows = o[tile_b, tile_h, tile_m, :].to(torch.float32)
        do_rows = do[tile_b, tile_h, tile_m, :].to(torch.float32)
        delta[tile_b, tile_h, tile_m] = torch.sum(o_rows * do_rows, dim=-1)
        dq[tile_b, tile_h, tile_m, :] = 0.0
    return delta


def _row_sum_then_rowvec_rows_fn(
    o: torch.Tensor, do: torch.Tensor, row: torch.Tensor, dq: torch.Tensor
) -> torch.Tensor:
    B, H, M, _ = o.size()
    D = dq.size(-1)
    delta = torch.empty((B, H, M), dtype=torch.float32, device=o.device)
    for tile_b, tile_h, tile_m in hl.tile([B, H, M]):
        o_rows = o[tile_b, tile_h, tile_m, :].to(torch.float32)
        do_rows = do[tile_b, tile_h, tile_m, :].to(torch.float32)
        delta[tile_b, tile_h, tile_m] = torch.sum(o_rows * do_rows, dim=-1)
        dq[tile_b, tile_h, tile_m, :] = row[None, None, None, :] + hl.zeros(
            [tile_b, tile_h, tile_m, D], dtype=dq.dtype
        )
    return delta


def _row_sum_then_zero_flagged_rows_fn(
    o: torch.Tensor, do: torch.Tensor, flag: torch.Tensor, dq: torch.Tensor
) -> torch.Tensor:
    """The masked form: only the rows ``flag`` marks are zeroed (the rows no
    KV block adds into, in the backward preprocess)."""
    B, H, M, _ = o.size()
    D = dq.size(-1)
    delta = torch.empty((B, H, M), dtype=torch.float32, device=o.device)
    for tile_b, tile_h, tile_m in hl.tile([B, H, M]):
        o_rows = o[tile_b, tile_h, tile_m, :].to(torch.float32)
        do_rows = do[tile_b, tile_h, tile_m, :].to(torch.float32)
        delta[tile_b, tile_h, tile_m] = torch.sum(o_rows * do_rows, dim=-1)
        absent = flag[tile_b, tile_h, tile_m] != 0
        hl.store(
            dq,
            [tile_b, tile_h, tile_m, slice(None)],
            hl.zeros([tile_b, tile_h, tile_m, D], dtype=dq.dtype),
            extra_mask=absent[:, :, :, None],
        )
    return delta


ROW_SUM_STORE_FNS = {
    "zeros": _row_sum_then_zero_rows_fn,
    "ones": _row_sum_then_fill_rows_fn,
    "scalar": _row_sum_then_scalar_rows_fn,
    "rowvec": _row_sum_then_rowvec_rows_fn,
    "masked": _row_sum_then_zero_flagged_rows_fn,
}
SMALL_BATCH_SHAPE = (2, 2, 256)
# The batch extent equal to the head dim: headdims' cell at a smaller head
# count, and the same at 128.
BATCH_IS_HEAD_DIM_SHAPES = {64: (64, 2, 256), 128: (128, 2, 256)}


def row_sum_store_kernel(variant: str, backend: str = "cute") -> object:
    return helion.kernel(
        ROW_SUM_STORE_FNS[variant],
        backend=backend,
        static_shapes=True,
        autotune_effort="none",
    )


def row_sum_store_args(
    variant: str,
    d: int,
    dv: int,
    shape: tuple[int, int, int],
    device: str = "cpu",
) -> tuple[torch.Tensor, ...]:
    """``(o, do, [row | flag], dq)`` for one variant (``dq`` in the returned
    tuple's last slot); ``shape`` is ``(B, H, M)``."""
    b, h, m = shape
    o = torch.empty((b, h, m, dv), dtype=torch.bfloat16, device=device)
    do = torch.empty((b, h, m, dv), dtype=torch.bfloat16, device=device)
    dq = torch.empty((b, h, m, d), dtype=torch.float32, device=device)
    if variant == "rowvec":
        return (o, do, torch.empty((d,), dtype=torch.float32, device=device), dq)
    if variant == "masked":
        return (o, do, torch.empty((b, h, m), dtype=torch.int32, device=device), dq)
    return (o, do, dq)


def row_sum_store_config(bound: object, *, rolled: int | None) -> dict[str, object]:
    """The default config, its reduction loops rolled to ``rolled``-lane chunks
    (None: persistent, every lane of the row on the thread's registers)."""
    config = dict(bound.config_spec.default_config().config)  # pyrefly: ignore [missing-attribute]
    loops = config.get("reduction_loops")
    if isinstance(loops, list) and loops:
        config["reduction_loops"] = [rolled] * len(loops)
    return config


def _render(
    variant: str, d: int, shape: tuple[int, int, int], rolled: int | None
) -> str:
    args = row_sum_store_args(variant, d, d, shape)
    with _mock_cuda_unavailable():
        bound = _cpu_bind(row_sum_store_kernel(variant), args)
        config = row_sum_store_config(bound, rolled=rolled)
        return bound.to_code(helion.Config(**config))


def dq_store_terms(code: str) -> list[list[str]]:
    """The index terms (one per tensor dim) of every store into ``dq``."""
    terms = []
    for line in code.splitlines():
        if "dq.iterator" not in line or ".store(" not in line:
            continue
        address = line.split("dq.iterator + ", 1)[1].split(").store(", 1)[0]
        terms.append(
            [
                part.split(" * cutlass.Int32(dq.layout.stride[")[0]
                for part in address.split(" + ")
            ]
        )
    return terms


BROADCAST_VARIANTS = ["zeros", "ones", "masked"]
REPLAYED_VARIANTS = ["scalar", "rowvec"]


@pytest.mark.parametrize("variant", BROADCAST_VARIANTS)
@pytest.mark.parametrize("rolled", ["persistent", "rolled"])
@pytest.mark.parametrize(("extent", "chunk"), [(64, 16), (128, 32)])
def test_full_slice_store_at_a_batch_extent_equal_to_the_head_dim_takes_the_chunk_lane_loop(
    variant: str, rolled: str, extent: int, chunk: int
) -> None:
    """The broadcast store's column has no active index of its own at the
    site (the reduction's index lives inside its rolled loop, or its
    persistent lanes closed before the store); the batch tile's block has
    the same extent and is active, but the subscript's first position
    already indexes it.  Before: silently bound to the batch block, one
    element per row, the column equal to the batch index.  With the
    resolver rule alone: rejected.  Now the slice takes the rolled block's
    chunk lane loop: every ``dq`` store's column term is that loop's index
    and never the first term, the store sits in the loop, and every name is
    read where a definition is in scope."""
    code = _render(
        variant,
        extent,
        BATCH_IS_HEAD_DIM_SHAPES[extent],
        chunk if rolled == "rolled" else None,
    )
    stores = dq_store_terms(code)
    assert stores, code
    for terms in stores:
        assert len(terms) == 4, terms
        assert terms[0] == "cutlass.Int32(indices_0)", terms
        assert terms[3] != terms[0], terms
        assert terms[3].startswith("cutlass.Int32(rindex_lane_"), terms
    function = _kernel_function(code)
    assert not _escaping_names(function), code
    (replica,) = _loops(function, "synthetic_lane_")
    assert _accesses(replica, "dq"), code


@pytest.mark.parametrize("variant", REPLAYED_VARIANTS)
@pytest.mark.parametrize("rolled", ["persistent", "rolled"])
@pytest.mark.parametrize(("extent", "chunk"), [(64, 16), (128, 32)])
def test_full_slice_store_at_a_batch_extent_equal_to_the_head_dim_never_takes_the_batch_index(
    variant: str, rolled: str, extent: int, chunk: int
) -> None:
    """The scalar and row-vector stores are emitted where the slice's own
    index is active (replayed inside the reduction's loop, or on the row
    vector's lanes): they lower, and every ``dq`` store's column term is
    that index, never the batch tile's (the first term)."""
    code = _render(
        variant,
        extent,
        BATCH_IS_HEAD_DIM_SHAPES[extent],
        chunk if rolled == "rolled" else None,
    )
    stores = dq_store_terms(code)
    assert stores, code
    for terms in stores:
        assert len(terms) == 4, terms
        assert terms[0] == "cutlass.Int32(indices_0)", terms
        assert terms[3] != terms[0], terms
        assert terms[3] in {"cutlass.Int32(indices_3)", "cutlass.Int32(rindex_3)"}, (
            terms
        )


@pytest.mark.parametrize(
    ("variant", "extent"),
    [(v, 64) for v in sorted(ROW_SUM_STORE_FNS)]
    + [(v, 128) for v in REPLAYED_VARIANTS],
)
def test_full_slice_store_at_a_small_batch_takes_the_slices_own_block(
    variant: str, extent: int
) -> None:
    """The control: with the batch extent apart from the head dim the slice's
    block is the only candidate (the shape every earlier sweep used)."""
    code = _render(variant, extent, SMALL_BATCH_SHAPE, None)
    stores = dq_store_terms(code)
    assert stores, code
    for terms in stores:
        assert len(terms) == 4, terms
        assert terms[3] != terms[0], terms
        assert terms[3] in {"cutlass.Int32(indices_3)", "cutlass.Int32(rindex_3)"}, (
            terms
        )
