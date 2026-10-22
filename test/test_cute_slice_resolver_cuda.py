"""GPU numerics for the CuTe slice resolver (see the GPU-free companion)."""

from __future__ import annotations

import pytest
import torch

from test.test_cute_slice_resolver import BATCH_IS_HEAD_DIM_SHAPES
from test.test_cute_slice_resolver import REPLAYED_VARIANTS
from test.test_cute_slice_resolver import ROW_SUM_STORE_FNS
from test.test_cute_slice_resolver import SMALL_BATCH_SHAPE
from test.test_cute_slice_resolver import row_sum_store_config
from test.test_cute_slice_resolver import row_sum_store_kernel

import helion
from helion._testing import skipUnlessBackends

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

CUDA_DEVICE = "cuda"


# The pairings that lower on the generic path with a persistent reduction:
# the broadcast forms (zeros, ones, masked) only at head dim 64 and a batch
# apart from it; the replayed forms (scalar, rowvec) everywhere.
_CASES = [
    *[(v, SMALL_BATCH_SHAPE, 64, 64) for v in sorted(ROW_SUM_STORE_FNS)],
    *[(v, SMALL_BATCH_SHAPE, 128, 64) for v in sorted(ROW_SUM_STORE_FNS)],
    *[(v, SMALL_BATCH_SHAPE, 128, 128) for v in REPLAYED_VARIANTS],
    *[(v, BATCH_IS_HEAD_DIM_SHAPES[64], 64, 64) for v in REPLAYED_VARIANTS],
    *[(v, BATCH_IS_HEAD_DIM_SHAPES[128], 128, 128) for v in REPLAYED_VARIANTS],
]


@pytest.mark.parametrize(
    ("variant", "shape", "d", "dv"),
    _CASES,
    ids=[f"{v}-b{s[0]}-d{d}-dv{dv}" for v, s, d, dv in _CASES],
)
def test_full_slice_store_beside_a_row_reduction_stores_every_element(
    variant: str, shape: tuple[int, int, int], d: int, dv: int
) -> None:
    """The attention backward's preprocess shape on the generic lowering: a
    full-slice store of a broadcast value (zeros, ones, a scalar, a row
    vector, zeros under a per-row mask) over head dim ``d`` beside
    ``sum(o * do)`` over head dim ``dv`` in one ``hl.tile`` body, the
    reduction persistent.  With the batch extent equal to the head dim the
    slice must take the reduction's block, not the batch tile's: every
    element is stored, matching torch and the Triton backend's store
    bitwise."""
    torch.manual_seed(0)
    b, h, m = shape
    o = torch.randn((b, h, m, dv), dtype=torch.bfloat16, device=CUDA_DEVICE)
    do = torch.randn_like(o)
    row = torch.randn((d,), dtype=torch.float32, device=CUDA_DEVICE)
    flag = (torch.rand((b, h, m), device=CUDA_DEVICE) < 0.4).to(torch.int32)
    expected_delta = (o.float() * do.float()).sum(-1)
    outputs: dict[str, torch.Tensor] = {}
    for backend in ("cute", "triton"):
        fill = 7.0 if variant == "masked" else float("nan")
        dq = torch.full((b, h, m, d), fill, dtype=torch.float32, device=CUDA_DEVICE)
        extra: tuple[torch.Tensor, ...] = {"rowvec": (row,), "masked": (flag,)}.get(
            variant, ()
        )
        args = (o, do, *extra, dq)
        bound = row_sum_store_kernel(variant, backend).bind(args)  # pyrefly: ignore [missing-attribute]
        config = row_sum_store_config(bound, rolled=None)
        delta = bound.compile_config(helion.Config(**config))(*args)
        torch.cuda.synchronize()
        torch.testing.assert_close(delta, expected_delta, rtol=1e-3, atol=1e-2)
        if variant == "masked":
            absent = (flag != 0)[..., None].expand_as(dq)
            assert torch.all(dq[absent] == 0), backend
            assert torch.all(dq[~absent] == 7.0), backend
        else:
            assert not torch.isnan(dq).any(), backend
            if variant == "rowvec":
                assert torch.equal(dq, row.expand_as(dq)), backend
            else:
                assert torch.all(dq == (1.0 if variant == "ones" else 0.0)), backend
        outputs[backend] = dq.clone()
    # delta is a bf16-product sum whose order differs per backend; dq is
    # the stored broadcast value and must agree bitwise.
    assert torch.equal(outputs["cute"], outputs["triton"])
