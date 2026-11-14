"""Native wrapping and guarded prefetch for scalar table-index recipes."""

from __future__ import annotations

import operator

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing.cute_mma_dag_scalar_indices import scalar_index_args
from helion._testing.cute_mma_dag_scalar_indices import scalar_index_config
from helion._testing.cute_mma_dag_scalar_indices import scalar_index_matmul
from helion._testing.cute_mma_dag_scalar_indices import wrap_scalar_index as _wrap


@skipIfNotCUDA()
@skipIfCudaCapabilityLessThan((9, 0))
@pytest.mark.parametrize(
    "mode,dtype,variant,stages,comparison,scalar_rhs",
    [
        ("warp", torch.int32, "reverse", 1, "", False),
        ("tcgen05", torch.int64, "reverse", 3, "", False),
        ("warp", torch.int64, "wrap", 3, "", False),
        ("tcgen05", torch.int32, "wrap", 1, "", False),
        ("warp", torch.int32, "nested", 3, "", False),
        ("tcgen05", torch.int64, "nested", 1, "", False),
        ("warp", torch.int64, "floor", 3, "", False),
        ("tcgen05", torch.int32, "floor", 1, "", False),
        # Both physical lowerings cover signed extrema, including explicit
        # narrowing before the mixed-width compare and changed-input replay.
        ("warp", torch.int32, "extrema", 3, "", False),
        ("tcgen05", torch.int64, "extrema_narrow", 1, "", False),
        # Every comparison crosses true/false during the prefetch ring; pair
        # widths, owners and overloads rather than taking their full product.
        ("warp", torch.int32, "compare", 3, "eq", False),
        ("tcgen05", torch.int64, "compare", 3, "ne", True),
        ("warp", torch.int64, "compare", 3, "lt", True),
        ("tcgen05", torch.int32, "compare", 3, "le", False),
        ("warp", torch.int32, "compare", 3, "gt", True),
        ("tcgen05", torch.int64, "compare", 3, "ge", False),
    ],
)
def test_typed_index_wrapping_order_and_guarded_replay(
    mode, dtype, variant, stages, comparison, scalar_rhs
):
    if mode == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the tcgen05 fixture targets SM100")
    args = scalar_index_args(
        dtype, variant, DEVICE, comparison=comparison, scalar_rhs=scalar_rhs
    )
    (
        a,
        pages,
        order,
        bounds,
        offsets,
        _variant,
        divisor,
        _comparison,
        _scalar_rhs,
        literal,
    ) = args
    a.copy_(
        (torch.arange(a.numel(), device=DEVICE).reshape(a.shape) % 7 - 3).to(a.dtype)
    )
    pages.copy_(
        (torch.arange(pages.numel(), device=DEVICE).reshape(pages.shape) % 13 - 6).to(
            pages.dtype
        )
    )
    bits = 32 if dtype == torch.int32 else 64
    limit = (1 << (bits - 1)) - 1
    orders = ([4, 2, 0, 3, 1], [1, 3, 4, 0, 2])
    order.copy_(order.new_tensor(orders[0]))
    bounds.copy_(bounds.new_tensor([1, 4]))
    offsets.copy_(offsets.new_tensor([limit - 1, -5]))
    bound = helion.kernel(scalar_index_matmul, backend="cute", static_shapes=True).bind(
        args
    )
    config = scalar_index_config(mode, stages)
    compiled = bound.compile_config(config)
    compiled(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(*args)
    for iteration, (begin, end) in enumerate([(1, 4), (0, 5), (1024, 1024), (2, 4)]):
        indices = orders[iteration % 2]
        order.copy_(order.new_tensor(indices))
        bounds.copy_(bounds.new_tensor([begin, end]))
        offset_values = [limit - iteration, -5 + iteration]
        if variant == "compare":
            offset_values = [(-1, 1), (2, 2), (-3, 0), (limit, 1)][iteration]
        if variant == "extrema_narrow" and iteration == 1:
            offset_values[0] = 2**32 + 5
        offsets.copy_(offsets.new_tensor(offset_values))
        if iteration == 1:
            a.neg_()
            pages.add_(1)
        expected = torch.zeros((64, 64), dtype=torch.float32, device=DEVICE)
        for step in range(begin, end):
            if variant == "reverse":
                index = _wrap(_wrap(_wrap(begin + end, bits) - 1, bits) - step, bits)
            elif variant == "floor":
                index = (_wrap(offset_values[0] + step, bits) // divisor) % len(indices)
            elif variant in ("extrema", "extrema_narrow"):
                added = _wrap(offset_values[0] + step, bits)
                if variant == "extrema_narrow":
                    added = _wrap(added, 32)
                index = min(
                    max(added, offset_values[1]), _wrap(offset_values[0] - step, bits)
                ) % len(indices)
            elif variant == "compare":
                left = _wrap(offset_values[0] + step, bits)
                right = literal if scalar_rhs else offset_values[1]
                predicate = getattr(operator, comparison)(left, right)
                index = (step + int(predicate)) % len(indices)
            else:
                index = _wrap(
                    _wrap(_wrap(offset_values[0] + step, bits) * 3, bits)
                    - offset_values[1],
                    bits,
                ) % len(indices)
            page = indices[index]
            if variant == "nested":
                page = indices[(page + 1) % len(indices)]
            expected = expected * 0.5 + a.float() @ pages[page].float()
        direct = compiled(*args)
        torch.testing.assert_close(direct, expected, rtol=0, atol=0)
        captured.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
