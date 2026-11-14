"""The arguments of the nvfp4 GEMV example's fp4-input kernel.

Shared by the CuTe tests that run it (``test/test_cute_tile_packing.py``,
``test/test_cute_fp4_pair_dot.py``, ``test/test_cute_device_loop_lane_unroll.py``)
and one that renders it on CPU
(``test/test_cute_device_loop_lane_unroll_codegen.py``).
"""

from __future__ import annotations

import torch

from . import DEVICE
from . import import_example

_nvfp4_gemv = import_example("nvfp4_gemv")


def nvfp4_gemv_args(
    rows: int, kbytes: int, *, random: bool = True
) -> tuple[tuple[object, ...], torch.Tensor | None]:
    """The arguments of ``_nvfp4_gemv_fp4in_body`` for ``rows`` rows of
    ``kbytes`` packed fp4 bytes (groups of eight), on ``DEVICE``, and the
    reference result.

    ``random=False`` gives uninitialized CPU tensors of the same shapes and
    dtypes and no reference: enough to bind and render on CPU.
    """
    groups = kbytes // 8
    device = DEVICE if random else None
    if random:
        torch.manual_seed(0)
        weight = torch.randint(0, 256, (rows, kbytes), dtype=torch.uint8, device=device)
        weight = weight.view(torch.float4_e2m1fn_x2)
        x = torch.randint(0, 256, (kbytes,), dtype=torch.uint8, device=device)
        x = x.view(torch.float4_e2m1fn_x2)
        wscale = _nvfp4_gemv.make_fp8_scales((rows, groups), device)
        xscale = _nvfp4_gemv.make_fp8_scales((groups,), device)
        expected = _nvfp4_gemv.reference_nvfp4_gemv_fp4in(weight, x, wscale, xscale)
    else:
        weight = torch.empty((rows, kbytes), dtype=torch.float4_e2m1fn_x2)
        x = torch.empty((kbytes,), dtype=torch.float4_e2m1fn_x2)
        wscale = torch.empty(
            (_nvfp4_gemv.swizzled_scale_numel(rows, groups),), dtype=torch.float8_e4m3fn
        )
        xscale = torch.empty(
            (_nvfp4_gemv.swizzled_scale_numel(1, groups),), dtype=torch.float8_e4m3fn
        )
        expected = None
    out = torch.empty(rows, dtype=torch.bfloat16, device=device)
    args = (
        weight.view(rows, groups, 8),
        x.view(groups, 8),
        wscale.reshape(-1),
        xscale.reshape(-1),
        out,
        1.0,
    )
    return args, expected
