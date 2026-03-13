# Auto-generated standalone Triton kernel for 'rms_norm'.
# No Helion dependency required at runtime.

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _default_launcher(
    triton_kernel,
    grid,
    *args,
    num_warps,
    num_stages,
    launch_cooperative_grid=False,
    **kwargs,
):
    return triton_kernel.run(
        *args,
        grid=grid,
        warmup=False,
        num_warps=num_warps,
        num_stages=num_stages,
        launch_cooperative_grid=launch_cooperative_grid,
        **kwargs,
    )


# =================================================================
# Config 0
# =================================================================

@triton.jit
def _helion_rms_norm_c0(x, weight, out, out_stride_0, out_stride_1, weight_stride_0, x_stride_0, x_stride_1, n, eps, _RDIM_SIZE_1: tl.constexpr):
    # src[07_aot_standalone.py:14]: for tile_m in hl.tile(m):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    indices_1 = tl.arange(0, _RDIM_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < n
    # src[07_aot_standalone.py:15]: row = x[tile_m, :].to(torch.float32)
    load = tl.load(x + (indices_0[:, None] * x_stride_0 + indices_1[None, :] * x_stride_1), mask_1[None, :], other=0)
    v_0 = tl.cast(load, tl.float32)
    # src[07_aot_standalone.py:16]: rms = torch.sqrt(torch.mean(row * row, dim=-1) + eps)
    v_1 = v_0 * v_0
    mean_extra = tl.cast(tl.sum(v_1, 1), tl.float32)
    v_2 = tl.cast(n, tl.float32)
    v_3 = mean_extra / v_2
    v_4 = v_3 + eps
    v_5 = tl.sqrt_rn(v_4)
    # src[07_aot_standalone.py:17]: out[tile_m, :] = (row / rms[:, None] * weight[:].to(torch.float32)).to(out.dtype)
    subscript = v_5[:, None]
    v_6 = v_0 / subscript
    load_1 = tl.load(weight + indices_1 * weight_stride_0, mask_1, other=0)
    v_7 = tl.cast(load_1, tl.float32)
    v_8 = v_7[None, :]
    v_9 = v_6 * v_8
    v_10 = tl.cast(v_9, tl.bfloat16)
    tl.store(out + (indices_0[:, None] * out_stride_0 + indices_1[None, :] * out_stride_1), v_10, mask_1[None, :])

def _rms_norm_c0(x: torch.Tensor, weight: torch.Tensor, eps: float=1e-05, *, _launcher=_default_launcher):
    # src[07_aot_standalone.py:12]: m, n = x.size()
    m, n = x.size()
    # src[07_aot_standalone.py:13]: out = torch.empty_like(x)
    out = torch.empty_like(x)
    # src[07_aot_standalone.py:14]: for tile_m in hl.tile(m):
    _RDIM_SIZE_1 = triton.next_power_of_2(n)
    # src[07_aot_standalone.py:14]: for tile_m in hl.tile(m):
    # src[07_aot_standalone.py:15]:     row = x[tile_m, :].to(torch.float32)
    # src[07_aot_standalone.py:16]:     rms = torch.sqrt(torch.mean(row * row, dim=-1) + eps)
    # src[07_aot_standalone.py:14-17]: ...
    _launcher(_helion_rms_norm_c0, (m,), x, weight, out, out.stride(0), out.stride(1), weight.stride(0), x.stride(0), x.stride(1), n, eps, _RDIM_SIZE_1, num_warps=16, num_stages=1)
    # src[07_aot_standalone.py:18]: return out
    return out

def rms_norm(*args, **kwargs):
    return [
        _rms_norm_c0,
    ][0](*args, **kwargs)
