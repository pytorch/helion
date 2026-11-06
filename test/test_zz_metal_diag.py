"""TEMPORARY diagnostic for the Metal CI row (do not merge).

Fails on purpose so the job log carries, for the kernels of
``TestMppStagingSpread.test_hoisted_operands``: the normalized config, the
host code's launch block dims, the dealt lane loops of the MSL, and the full
host code and MSL, to diff between origin/main and the CuTe refactor stack.
"""

from __future__ import annotations

import hashlib
import platform
import re
import sys

import pytest
import torch

import helion

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Metal diagnostic runs on macOS only"
)


def _render(kernel: helion.Kernel, args: tuple, config: helion.Config | None) -> str:
    from test import test_metal_mpp_matmul as m
    from torch._inductor.codecache import PyCodeCache

    from helion._compiler.metal.metal_jit import _MetalKernel

    bound = kernel.bind(args)
    config = config or bound.config_spec.default_config()
    code = bound.to_code(config)
    module = PyCodeCache.load(code)
    getattr(module, kernel.fn.__name__)(*args)
    torch.mps.synchronize()
    (msl,) = [
        k.msl_source
        for k in vars(module).values()
        if isinstance(k, _MetalKernel) and k.msl_source is not None
    ]
    dims = re.search(r"_block_dims=\((\d+), (\d+), (\d+)\)", code)
    lines = [
        f"### kernel {kernel.fn.__name__}",
        f"config: {dict(config)}",
        f"block_dims: {dims.groups() if dims else None}",
        f"spread_loops: {[loop[:3] for loop in m._spread_loops(msl)]}",
        (
            f"code sha {hashlib.sha256(code.encode()).hexdigest()[:10]} "
            f"msl sha {hashlib.sha256(msl.encode()).hexdigest()[:10]}"
        ),
        "--- HOST CODE ---",
        code,
        "--- MSL ---",
        msl,
    ]
    return "\n".join(lines)


def test_metal_diag_dump() -> None:
    from test import test_metal_mpp_matmul as m

    h = torch.float16
    out = [
        (
            f"python {sys.version.split()[0]} torch {torch.__version__} "
            f"{platform.platform()} mps={torch.backends.mps.get_name()}"
        ),
    ]
    x, g, bias = (
        torch.randn(37, 72, dtype=h),
        torch.randn(37, dtype=h),
        torch.randn(72, dtype=h),
    )
    w = torch.randn(72, 40, dtype=h)
    out.append(
        _render(
            m.gated_lhs,
            tuple(t.to("mps") for t in (x, g, bias, w)),
            helion.Config(block_sizes=[32, 32], num_warps=4),
        )
    )
    x2, w2 = torch.randn(40, 64, dtype=h), torch.randn(64, 48, dtype=h)
    for config in (
        None,
        helion.Config(block_sizes=[16, 16], num_threads=[16, 0], num_warps=1),
    ):
        out.append(_render(m.scaled_colon_rhs, (x2.to("mps"), w2.to("mps")), config))
    pytest.fail("METAL DIAG BEGIN\n" + "\n".join(out) + "\nMETAL DIAG END")
