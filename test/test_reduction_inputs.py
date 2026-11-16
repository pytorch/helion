from __future__ import annotations

import pytest
import torch

from helion._testing.reduction_inputs import exact_fp32_sum_input

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "shape,transpose",
    [
        ((6, 24, 64), False),
        ((6, 33, 33), False),
        ((6, 1, 33), False),
        ((6, 33, 33), True),
    ],
)
def test_exact_sum_inputs_cover_reduction_orders(shape, transpose):
    x = exact_fp32_sum_input(shape)
    if transpose:
        x = x.transpose(1, 2)
    flat = x.flatten(1)
    reference = x.to(torch.float64).sum((1, 2))
    assert ((flat > 0).any(1) & (flat < 0).any(1)).all()
    # Every axis varies, including the noncontiguous case. The unit axis is
    # intentionally exempt, as in the native dimension-merging test.
    for axis, size in enumerate(x.shape):
        if size > 1:
            assert x.diff(dim=axis).ne(0).any()

    results = [
        flat.sum(1),
        flat.flip(1).sum(1),
        x.sum(1).sum(1),
        x.sum(2).sum(1),
    ]
    for chunk in (16, 256):
        lanes = torch.zeros((shape[0], chunk), dtype=torch.float32)
        partials = []
        for block in flat.split(chunk, dim=1):
            # Model the generated loop's per-lane accumulation before its
            # final reduction, including zero padding of the final block.
            lanes[:, : block.size(1)] += block
            partials.append(block.sum(1))
        partials = torch.stack(partials)
        # Dropping or duplicating any whole chunk must remain observable.
        assert partials.ne(0).all()
        results.extend((lanes.sum(1), partials.sum(0)))
    for result in results:
        torch.testing.assert_close(result.double(), reference, rtol=0, atol=0)


def test_exact_sum_inputs_preserve_global_rng_state():
    before = torch.random.get_rng_state()
    first = exact_fp32_sum_input((6, 24, 64))
    second = exact_fp32_sum_input((6, 24, 64))
    assert torch.equal(torch.random.get_rng_state(), before)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
