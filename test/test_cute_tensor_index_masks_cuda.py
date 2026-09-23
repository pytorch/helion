from __future__ import annotations

from examples.jagged_dense_bmm import jagged_dense_bmm
import pytest
import torch

import helion
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])

CUDA_DEVICE = "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_scalar_jagged_reduction_tail_ignores_backing_storage() -> None:
    torch.manual_seed(20260912)
    batch, rows, k, n, bk = 16, 319, 32, 32, 64
    lengths = torch.tensor([19, 0, 40] + [20] * 13, dtype=torch.int64)
    offsets = torch.zeros(batch + 1, dtype=torch.int64, device=CUDA_DEVICE)
    offsets[1:].copy_(lengths.cumsum(0))
    # Poison only storage outside the contiguous logical view. Keeping the
    # backing allocation alive removes allocator history from this test.
    backing = torch.empty(rows * k + bk, device=CUDA_DEVICE)
    a = backing[: rows * k].view(rows, k)
    a.normal_()
    b = torch.randn((batch, k, n), device=CUDA_DEVICE)
    bias = torch.randn((batch, n), device=CUDA_DEVICE)
    arguments = (offsets, a, b, bias)
    original = tuple(value.clone() for value in arguments)
    assert a.is_contiguous() and a.stride() == (k, 1)
    assert a.data_ptr() == backing.data_ptr()

    left, right = a.double(), b.double()
    pieces = []
    begin = 0
    for group, count in enumerate(lengths.tolist()):
        end = begin + count
        pieces.append((left[begin:end] @ right[group]).float() + bias[group])
        begin = end
    expected = torch.cat(pieces)
    assert torch.isfinite(expected).all()

    kernel = helion.kernel(
        jagged_dense_bmm.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        dot_precision="ieee",
    )
    bound = kernel.bind(arguments)
    config = helion.Config(
        block_sizes=[1, 32, n, bk],
        num_threads=[0, 4, 32, 1],
        cute_vector_widths=[1] * 4,
        cute_lane_layouts=["blocked"] * 4,
    )
    source = bound.to_code(config)
    assert "dot_acc" in source
    assert "cute.gemm(" not in source
    assert "cvt_f32_tf32" not in source
    bound.set_config(config)

    outputs = []
    for padding in (0.0, float("nan"), 0.0):
        backing[rows * k :].fill_(padding)
        output = bound(*arguments)
        outputs.append(output)
        torch.testing.assert_close(output, expected, rtol=2e-4, atol=2e-4)
        for before, after in zip(original, arguments, strict=True):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert len({output.data_ptr() for output in outputs}) == len(outputs)
    for output in outputs:
        torch.testing.assert_close(output, expected, rtol=2e-4, atol=2e-4)
