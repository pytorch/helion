"""Test a single skipped test case to see error."""
from __future__ import annotations

import os
os.environ.setdefault('HELION_AUTOTUNE_EFFORT', 'none')

import torch
from torch._inductor.utils import run_and_get_code
from helion._testing import DEVICE, count_triton_kernels
import helion
import helion.language as hl


@helion.kernel(static_shapes=True, autotune_effort='none', allow_fuse_with_inductor_ops=True)
def elementwise_3d(x, scale):  # pyrefly: ignore
    """Simple 3D elementwise kernel for testing 3D transpose fusion."""
    d1, d2, d3 = x.size()
    out = torch.empty_like(x)

    for tile_d1 in hl.tile(d1):
        for tile_d2 in hl.tile(d2):
            x_tile = x[tile_d1, tile_d2, :]
            out[tile_d1, tile_d2, :] = x_tile * scale

    return out, 42


def test_3d_transpose_prologue_swap_last_two_dims():
    """3D transpose prologue x.permute(0, 2, 1) -> kernel - pure 3D transpose."""
    d1, d2, d3 = 8, 16, 32
    x = torch.randn(d1, d3, d2, device=DEVICE, dtype=torch.float32)
    x_scale = torch.randn(d1, d2, d3, device=DEVICE, dtype=torch.float32)
    kernel_scale = 2.0

    def f(x, x_scale):
        x_permuted = x.permute(0, 2, 1)  # [D1, D2, D3]
        x_scaled = x_permuted * x_scale  # [D1, D2, D3]
        x_transformed = torch.relu(x_scaled)  # [D1, D2, D3]
        out, info = elementwise_3d(x_transformed, kernel_scale)
        return out, info

    inputs = (x, x_scale)

    # Eager
    out_eager, info_eager = f(*inputs)
    print(f'Eager info: {info_eager}')
    assert info_eager == 42

    # Compiled
    compiled_f = torch.compile(f, fullgraph=True, backend='inductor')
    result, source_codes = run_and_get_code(compiled_f, *inputs)
    out_compiled, info_compiled = result

    torch.testing.assert_close(out_compiled, out_eager, rtol=1e-2, atol=1e-2)
    print(f'Compiled info: {info_compiled}')
    print(f'Out shape: {list(out_compiled.shape)}')
    assert info_compiled == 42
    assert list(out_compiled.shape) == [d1, d2, d3]

    kernel_count, _ = count_triton_kernels(source_codes)
    print(f'Kernel count: {kernel_count}')
    assert kernel_count == 1, f"Expected 1 kernel, got {kernel_count}"


if __name__ == "__main__":
    test_3d_transpose_prologue_swap_last_two_dims()
    print("PASSED!")
