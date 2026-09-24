from __future__ import annotations

import ast

import pytest
import torch

from .test_cute_grid_launch_extents import _code
from .test_cute_grid_launch_extents import _launch_block
import helion
from helion._testing import DEVICE
from helion._testing import _example_kernel
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _serial_rows(
    x: torch.Tensor, row_scale: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    block = hl.register_block_size(x.shape[0])
    columns = hl.specialize(x.shape[1])
    for group in hl.tile(x.shape[0], block_size=block):
        for row in hl.tile(group.begin, group.end):
            value = x[row, :].float() * row_scale[row][:, None]
            total = torch.sum(value, dim=-1) / columns
            out[row, :] = (value + total[:, None]).to(x.dtype)
    return out


def _config() -> helion.Config:
    return helion.Config(block_sizes=[32, 1], num_warps=4, num_stages=3)


@pytest.mark.parametrize("columns", [8, 16, 32])
@pytest.mark.parametrize("rows", [160, 169])
def test_serial_row_root_uses_physical_launch_axes(rows: int, columns: int) -> None:
    args = (
        torch.empty((rows, columns)),
        torch.empty((rows,)),
        torch.empty((rows, columns)),
    )
    code = _code(_serial_rows, args, _config())
    assert _launch_block(code) == (columns, 1, 1)
    # The root row dimension has no remaining per-thread work; its logical
    # width32 must not enlarge the feature axis mapped onto threadIdx.x.
    thread_axes = {
        node.slice.value
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "thread_idx"
        and isinstance(node.slice, ast.Constant)
    }
    assert thread_axes == {0}


@pytest.mark.parametrize("rows", [160, 1152000])
def test_original_layernorm_backward_keeps_feature_extent(rows: int) -> None:
    kernel = _example_kernel("layer_norm", fn_name="layer_norm_bwd")
    args = (
        torch.empty((rows, 16), dtype=torch.float16),
        torch.empty((rows, 16), dtype=torch.float16),
        torch.empty((rows,), dtype=torch.float32),
        torch.empty((rows,), dtype=torch.float32),
        torch.empty((16,), dtype=torch.float16),
        True,
    )
    code = _code(kernel, args, _config())
    assert _launch_block(code) == (16, 1, 1)
    assert "threads_in_group=16" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("columns", [8, 16, 32])
@pytest.mark.parametrize("rows", [160, 169])
def test_serial_rows_preserve_boundaries_and_canaries(rows: int, columns: int) -> None:
    # Padding makes accidental feature>=N accesses deterministic and checks
    # the trailing OOB writes instead of relying on allocator contents.
    x_storage = torch.arange(
        (rows + 2) * columns, device=DEVICE, dtype=torch.float32
    ).view(rows + 2, columns)
    x = x_storage[1:-1]
    # Extra feature lanes would read the next row's x with this row's scale,
    # so every CTA boundary is numerically wrong even if another CTA writes
    # the same output location. The trailing canary also detects the OOB.
    row_scale = torch.arange(1, rows + 1, device=DEVICE, dtype=torch.float32)
    storage = torch.full_like(x_storage, -12345)
    out = storage[1:-1]
    saved = x_storage.clone()
    saved_scale = row_scale.clone()
    value = x * row_scale[:, None]
    expected = value + value.mean(dim=-1, keepdim=True)
    run = _serial_rows._bind_isolated((x, row_scale, out)).compile_config(_config())
    for _ in range(3):
        out.fill_(float("nan"))
        result = run(x, row_scale, out)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        torch.testing.assert_close(result[::32], expected[::32], atol=0, rtol=0)
        assert torch.all(storage[0] == -12345)
        assert torch.all(storage[-1] == -12345)
        torch.testing.assert_close(x_storage, saved, atol=0, rtol=0)
        torch.testing.assert_close(row_scale, saved_scale, atol=0, rtol=0)
