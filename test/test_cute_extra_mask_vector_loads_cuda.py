"""GPU numerics for CuTe vector loads under ``hl.load(extra_mask=...)``.

Companion to ``test_cute_extra_mask_vector_loads.py``: the masked concatenation
example runs with a vectorized column tile where its ``extra_mask`` terms are
lane-uniform (packets) and where they are not (scalar fallback), and a kernel
whose mask changes inside every packet.
"""

from __future__ import annotations

from examples.concatenate import concat2d_dim1
import pytest
import torch

import helion
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

CUDA_DEVICE = "cuda"


def _run(kernel: object, args: tuple[object, ...], **config: object) -> torch.Tensor:
    bound = kernel.bind(args)  # pyrefly: ignore [missing-attribute]
    return bound.compile_config(helion.Config.from_dict(config))(*args)


@pytest.mark.parametrize(
    ("n1", "n2", "dtype", "vec_width"),
    [
        (512, 768, torch.float32, 4),  # both loads vectorize
        (500, 512, torch.float32, 4),  # 500 % 4 == 0: both loads vectorize
        (512, 768, torch.bfloat16, 8),  # 16-byte bf16 packets
        (502, 300, torch.float32, 4),  # misaligned bound: scalar fallback
        (500, 512, torch.bfloat16, 8),  # 500 % 8 != 0: scalar fallback
    ],
)
def test_concat_masked_matches_torch_cat(
    n1: int, n2: int, dtype: torch.dtype, vec_width: int
) -> None:
    kernel = helion.kernel(
        concat2d_dim1.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    x = torch.randn((37, n1), dtype=dtype, device=CUDA_DEVICE)
    y = torch.randn((37, n2), dtype=dtype, device=CUDA_DEVICE)
    out = _run(
        kernel,
        (x, y),
        block_sizes=[1, 2048],
        num_threads=[0, 2048 // vec_width],
        cute_vector_widths=[1, vec_width],
    )
    torch.testing.assert_close(out, torch.cat([x, y], dim=1), rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _keep_even_columns(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(out.size()):
        out[tile0, tile1] = hl.load(
            x, [tile0, tile1], extra_mask=(tile1.index % 2 == 0)[None, :]
        )
    return out


def test_lane_varying_extra_mask_matches_reference() -> None:
    x = torch.randn((5, 1000), device=CUDA_DEVICE)
    out = _run(
        _keep_even_columns,
        (x,),
        block_sizes=[1, 1024],
        num_threads=[0, 256],
        cute_vector_widths=[1, 4],
    )
    keep = (torch.arange(1000, device=CUDA_DEVICE) % 2 == 0)[None, :]
    torch.testing.assert_close(out, torch.where(keep, x, 0.0), rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _prefix_columns(x: torch.Tensor, last: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile0, tile1 in hl.tile(out.size()):
        out[tile0, tile1] = hl.load(
            x, [tile0, tile1], extra_mask=(tile1.index <= last)[None, :]
        )
    return out


@pytest.mark.parametrize("last", [511, 512])
def test_inclusive_bound_matches_reference(last: int) -> None:
    x = torch.randn((3, 1024), device=CUDA_DEVICE)
    out = _run(
        _prefix_columns,
        (x, last),
        block_sizes=[1, 1024],
        num_threads=[0, 256],
        cute_vector_widths=[1, 4],
    )
    keep = (torch.arange(1024, device=CUDA_DEVICE) <= last)[None, :]
    torch.testing.assert_close(out, torch.where(keep, x, 0.0), rtol=0, atol=0)
