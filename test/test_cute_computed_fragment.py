from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_recurrence(x: torch.Tensor, steps: hl.constexpr):
    batches = x.size(0)
    size = hl.specialize(x.size(1))
    out = torch.empty_like(x)
    mask_out = torch.empty_like(x, dtype=torch.bool)
    for batch in hl.tile(batches, block_size=1):
        index = hl.arange(size)
        mask = index[:, None] > index[None, :]
        value = torch.where(mask, x[batch, :, :], 0.0)
        mask_out[batch, :, :] = mask[None, :, :]
        for _ in range(steps):
            value = hl.dot(value, value.transpose(-2, -1)) + value
        out[batch, :, :] = value
    return out, mask_out


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_broadcast_dot(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor):
    batches = x.size(0)
    rows = hl.specialize(x.size(1))
    columns = y.size(2)
    contraction = hl.specialize(x.size(2))
    assert contraction == y.size(1)
    out = torch.empty((batches, rows, columns), device=x.device, dtype=x.dtype)
    for batch in hl.tile(batches, block_size=1):
        index = hl.arange(contraction)
        left = x[batch, :, index]
        right = y[batch, index, :] * scale[batch, index, None]
        out[batch, :, :] = hl.dot(left, right)
    return out


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("size,steps", [(5, 0), (5, 1), (8, 3)])
def test_static_fragment_axes_transpose_and_loop_carry(size, steps):
    torch.manual_seed(123)
    x = torch.randn((3, size, size), device=DEVICE) * 0.1
    expected_mask = (
        torch.arange(size, device=DEVICE)[:, None]
        > torch.arange(size, device=DEVICE)[None, :]
    )
    expected = torch.where(expected_mask, x, 0.0)
    for _ in range(steps):
        expected = expected @ expected.transpose(-2, -1) + expected
    actual, mask = _fragment_recurrence(x, steps)
    torch.testing.assert_close(mask, expected_mask.expand_as(mask))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_parallel_carries(x: torch.Tensor, y: torch.Tensor, steps: hl.constexpr):
    size = hl.specialize(x.size(1))
    out = torch.empty_like(x)
    other = torch.empty_like(y)
    for batch in hl.tile(x.size(0), block_size=1):
        index = hl.arange(size)
        eye = (index[:, None] == index[None, :]).to(x.dtype)
        left = x[batch, :, :] + eye[None, :, :]
        right = y[batch, :, :]
        for _ in range(steps):
            left, right = right, left + right
        out[batch, :, :] = left
        other[batch, :, :] = right
    return out, other


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("steps", [1, 3])
def test_fragment_parallel_carry_aliases_and_lazy_updates(steps):
    x = torch.randn((3, 5, 5), device=DEVICE)
    y = torch.randn_like(x)
    left, right = x + torch.eye(5, device=DEVICE), y
    for _ in range(steps):
        left, right = right, left + right
    actual = _fragment_parallel_carries(x, y, steps)
    torch.testing.assert_close(actual, (left, right))


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_half_carry_dot(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    flags: torch.Tensor,
    steps: hl.constexpr,
):
    out = torch.empty_like(x, dtype=torch.float32)
    for batch in hl.tile(x.size(0), block_size=1):
        state = x[batch, :, :].float()
        for _ in range(steps):
            if flags[batch.begin] > 0:
                product = hl.dot(z[batch, :, :], state.to(z.dtype))
                state = hl.dot(y[batch, :, :], product.to(y.dtype), acc=state)
            else:
                state = state * 0.5
        out[batch, :, :] = state
    return out


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("steps", [1, 3])
def test_half_dot_operand_reads_resident_loop_carry(dtype, steps):
    x = torch.randn((3, 3, 7), device=DEVICE, dtype=dtype) * 0.1
    y = torch.randn((3, 3, 5), device=DEVICE, dtype=dtype) * 0.1
    z = torch.randn((3, 5, 3), device=DEVICE, dtype=dtype) * 0.1
    flags = torch.tensor([0, 1, 1], device=DEVICE)
    expected = x.float()
    for _ in range(steps):
        product = z.float() @ expected.to(dtype).float()
        updated = y.float() @ product.to(dtype).float() + expected
        expected = torch.where(flags[:, None, None] > 0, updated, expected * 0.5)
    torch.testing.assert_close(
        _fragment_half_carry_dot(x, y, z, flags, steps),
        expected,
        atol=1e-6,
        rtol=1e-5,
    )


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_scaled_half_dot(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty((x.size(0), x.size(1), y.size(2)), device=x.device)
    for batch in hl.tile(x.size(0), block_size=1):
        left = x[batch, :, :] * 0.5
        for columns in hl.tile(y.size(2), block_size=4):
            out[batch, :, columns] = hl.dot(left, y[batch, :, columns])
    return out


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batched_half_dot_keeps_repeated_operand_axes_independent(dtype):
    x = torch.randn((3, 5, 5), device=DEVICE, dtype=dtype).transpose(-2, -1)
    y = torch.randn((3, 7, 5), device=DEVICE, dtype=dtype).transpose(-2, -1)
    torch.testing.assert_close(
        _fragment_scaled_half_dot(x, y),
        (x * 0.5).float() @ y.float(),
        atol=2e-5,
        rtol=1e-5,
    )


@skipUnlessBackends(["cute"])
def test_computed_batched_dot_noncontiguous_broadcast():
    torch.manual_seed(124)
    x = torch.randn((3, 7, 5), device=DEVICE).transpose(-2, -1)
    y = torch.randn((3, 9, 7), device=DEVICE).transpose(-2, -1)
    scale = torch.randn((3, 7), device=DEVICE)
    actual = _fragment_broadcast_dot(x, y, scale)
    torch.testing.assert_close(
        actual, x @ (y * scale[:, :, None]), atol=2e-5, rtol=1e-5
    )


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_batched_load_dot(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty((x.size(0), x.size(1), y.size(2)), device=x.device)
    for batch in hl.tile(x.size(0), block_size=1):
        out[batch, :, :] = hl.dot(x[batch, :, :], y[batch, :, :])
    return out


@skipUnlessBackends(["cute"])
def test_fragment_batched_dot_of_direct_loads():
    x = torch.randn((3, 5, 7), device=DEVICE)
    y = torch.randn((3, 7, 9), device=DEVICE)
    torch.testing.assert_close(_fragment_batched_load_dot(x, y), x @ y)


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_select(x: torch.Tensor, flags: torch.Tensor, scale: float):
    batches = x.size(0)
    size = hl.specialize(x.size(1))
    out = torch.empty_like(x)
    for batch in hl.tile(batches, block_size=1):
        index = hl.arange(size)
        eye = (index[:, None] == index[None, :]).to(x.dtype)
        value = x[batch, :, :] + eye[None, :, :]
        if flags[batch.begin] > 0:
            value = hl.dot(value, value.transpose(-2, -1))
        else:
            value = value * scale
        out[batch, :, :] = value
    return out


@skipUnlessBackends(["cute"])
def test_fragment_uniform_branch_and_offset_view():
    torch.manual_seed(125)
    x = torch.randn((4, 11, 11), device=DEVICE)[:, 1:11:2, 2:7]
    flags = torch.tensor([0, 1, 1, 0], device=DEVICE)
    actual = _fragment_select(x, flags, 0.125)
    value = x + torch.eye(5, device=DEVICE)
    expected = torch.where(
        flags[:, None, None] > 0, value @ value.transpose(-2, -1), value * 0.125
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_subrange(x: torch.Tensor, out: torch.Tensor):
    batches = x.size(0)
    rows = hl.specialize(x.size(1))
    for batch in hl.tile(batches, block_size=1):
        index = hl.arange(rows)
        eye = (index[:, None] == index[None, :]).to(x.dtype)
        for columns in hl.tile(2, x.size(2) - 3, block_size=4):
            out[batch, :, columns] = hl.dot(eye[None, :, :], x[batch, :, columns]) + 1.0
    return out


@skipUnlessBackends(["cute"])
def test_fragment_tiled_subrange_keeps_sentinels():
    x = torch.randn((3, 5, 14), device=DEVICE)
    out = torch.full_like(x, -123.0)
    actual = _fragment_subrange(x, out)
    expected = torch.full_like(x, -123.0)
    expected[:, :, 2:-3] = x[:, :, 2:-3] + 1.0
    torch.testing.assert_close(actual, expected)


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_sum(x: torch.Tensor):
    batches = x.size(0)
    size = hl.specialize(x.size(1))
    rows = torch.empty((batches, size, 1), device=x.device)
    columns = torch.empty((batches, 1, size), device=x.device)
    for batch in hl.tile(batches, block_size=1):
        index = hl.arange(size)
        value = torch.where(index[:, None] >= index[None, :], x[batch, :, :], 0.0)
        product = hl.dot(value, value.transpose(-2, -1))
        rows[batch, :, :] = product.sum(-1, keepdim=True)
        columns[batch, :, :] = product.sum(-2, keepdim=True)
    return rows, columns


@skipUnlessBackends(["cute"])
def test_fragment_sum_on_both_matrix_axes():
    x = torch.randn((3, 7, 7), device=DEVICE)
    rows, columns = _fragment_sum(x)
    value = torch.tril(x)
    product = value @ value.transpose(-2, -1)
    torch.testing.assert_close(
        rows, product.sum(-1, keepdim=True), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        columns, product.sum(-2, keepdim=True), atol=1e-5, rtol=1e-5
    )


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_scan(x: torch.Tensor, reverse: hl.constexpr):
    batches = x.size(0)
    out = torch.empty_like(x)
    for batch in hl.tile(batches, block_size=1):
        value = x[batch, :, :] * 2.0 + 1.0
        out[batch, :, :] = hl.cumsum(value, dim=1, reverse=bool(reverse))
    return out


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("reverse", [False, True])
def test_fragment_computed_scan_nonlast_axis(reverse):
    x = torch.randn((3, 7, 5), device=DEVICE)
    value = x * 2 + 1
    expected = value.flip(1).cumsum(1).flip(1) if reverse else value.cumsum(1)
    torch.testing.assert_close(_fragment_scan(x, reverse), expected)


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_scalar_fill(x: torch.Tensor, scales: torch.Tensor):
    size = hl.specialize(x.size(1))
    out = torch.empty_like(x)
    for batch in hl.tile(x.size(0), block_size=1):
        first = scales[batch.begin, 0]
        filled = hl.full([size, size], first, dtype=x.dtype)
        second = scales[batch.begin, 1]
        value = x[batch, :, :] * second
        out[batch, :, :] = hl.dot(filled[None, :, :], value)
    return out


@skipUnlessBackends(["cute"])
def test_fragment_dynamic_fill_keeps_scalar_storage():
    x = torch.randn((3, 5, 5), device=DEVICE)
    scales = torch.tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], device=DEVICE)
    filled = scales[:, 0, None, None].expand_as(x)
    expected = filled @ (x * scales[:, 1, None, None])
    torch.testing.assert_close(_fragment_scalar_fill(x, scales), expected)


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_masked_sum(x: torch.Tensor):
    rows = hl.specialize(x.size(1))
    out = torch.empty((x.size(0), rows), device=x.device)
    for batch in hl.tile(x.size(0), block_size=1):
        index = hl.arange(rows)
        eye = (index[:, None] == index[None, :]).to(x.dtype)
        total = (
            hl.zeros([batch, rows], dtype=torch.float32) + eye.sum(-1).float()[None, :]
        )
        for columns in hl.tile(x.size(2), block_size=4):
            total = total + (x[batch, :, columns] + 1.0).sum(-1).float()
        out[batch, :] = total
    return out


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fragment_lowprecision_sum_masks_computed_tail(dtype):
    x = torch.randn((3, 8, 7), device=DEVICE, dtype=dtype)
    expected = torch.ones((3, 8), device=DEVICE)
    for start in range(0, 7, 4):
        expected += (x[:, :, start : start + 4] + 1).sum(-1).float()
    torch.testing.assert_close(_fragment_masked_sum(x), expected)


@helion.kernel(backend="cute", autotune_effort="none")
def _fragment_masks(
    x: torch.Tensor, flags: torch.Tensor, out: torch.Tensor, unchanged: torch.Tensor
):
    size = hl.specialize(x.size(1))
    for batch in hl.tile(x.size(0), block_size=1):
        index = hl.arange(size)
        eye = (index[:, None] == index[None, :]).to(x.dtype)
        disabled = hl.full([], False, dtype=torch.bool)
        value = hl.load(x, [batch, slice(None), slice(None)], extra_mask=disabled) + 3.0
        result = hl.dot(eye[None, :, :], value)
        hl.store(
            out,
            [batch, slice(None), slice(None)],
            result,
            extra_mask=flags[batch.begin] > 0,
        )
        hl.store(
            unchanged, [batch, slice(None), slice(None)], result, extra_mask=disabled
        )
    return out, unchanged


@skipUnlessBackends(["cute"])
def test_fragment_scalar_memory_masks():
    x = torch.randn((3, 5, 5), device=DEVICE)
    flags = torch.tensor([0, 1, 0], device=DEVICE)
    out, unchanged = _fragment_masks(
        x, flags, torch.full_like(x, -7), torch.full_like(x, -9)
    )
    expected = torch.full_like(x, -7)
    expected[1] = 3
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(unchanged, torch.full_like(x, -9))


@skipUnlessBackends(["cute"])
def test_fragment_rejects_oversized_shared_storage_before_launch():
    with FakeTensorMode():
        x = torch.empty((1, 512, 512), device=DEVICE)
        bound = _fragment_recurrence.bind((x, 1))
    with pytest.raises(exc.InvalidConfig, match="shared bytes, exceeding"):
        bound.to_code(bound.config_spec.default_config())
