"""GPU tests: a transposing kernel's loads staged in a shared-memory image
(``cute_staged_load``: packet fill, swizzled image, relabeled threads and
CTAs) read exactly what eager reads, in bf16 and fp32, beside a direct load
of another tensor, and where the pass declines (a partial tile)."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
import helion.language as hl


def _copy(x: torch.Tensor, out: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile(out.size()):
        out[tile_z, tile_y, tile_x] = x[tile_z, tile_y, tile_x]


def _copy2d(x: torch.Tensor, out: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile(out.size()):
        out[tile_y, tile_x] = x[tile_y, tile_x]


def _copy4d(x: torch.Tensor, out: torch.Tensor) -> None:
    for tile_n, tile_h, tile_w, tile_c in hl.tile(out.size()):
        out[tile_n, tile_h, tile_w, tile_c] = x[tile_n, tile_h, tile_w, tile_c]


def _scaled(x: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile(out.size()):
        out[tile_z, tile_y, tile_x] = x[tile_z, tile_y, tile_x] * 2.0 + b[tile_x]


def _config(dtype: torch.dtype) -> dict[str, object]:
    # 16-byte stores along x: 8 bf16 or 4 fp32 per thread.
    vec = 16 // dtype.itemsize
    return {
        "block_sizes": [1, 64, 8 * vec],
        "num_threads": [1, 16, 8],
        "cute_vector_widths": [1, 1, vec],
        "cute_staged_load": True,
    }


@pytest.mark.parametrize(
    ("dtype", "shape", "scaled", "staged"),
    [
        # Every CTA stages; bf16 and fp32 (an image row of 128 bytes each).
        (torch.bfloat16, (3, 128, 192), False, True),
        (torch.float32, (2, 128, 96), True, True),
        # y of 100 in tiles of 64: masked loads, the pass declines.
        (torch.bfloat16, (2, 100, 192), False, False),
    ],
)
def test_staged_loads_match_eager(
    dtype: torch.dtype, shape: tuple[int, int, int], scaled: bool, staged: bool
) -> None:
    z, y, x = shape
    source = torch.randn(z, x, y, device=DEVICE, dtype=dtype).transpose(1, 2)
    out = torch.full(shape, 12345.0, device=DEVICE, dtype=dtype)
    kernel = helion.kernel(
        _scaled if scaled else _copy,
        backend="cute",
        static_shapes=True,
        config=helion.Config.from_dict(_config(dtype)),
    )
    b = torch.randn(x, device=DEVICE, dtype=dtype)
    args = (source, b, out) if scaled else (source, out)
    bound = kernel.bind(args)
    code = bound.to_code(bound.config_spec.normalized_config(kernel.configs[0]))
    assert ("_cta_image" in code) == staged
    kernel(*args)
    expected = source * 2.0 + b if scaled else source
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "case", ["rank2_fp32", "nchw_to_nhwc_bf16", "element_per_thread"]
)
def test_other_ranks_match_eager(case: str) -> None:
    if case == "rank2_fp32":
        source = torch.randn(768, 512, device=DEVICE).t()
        out = torch.full((512, 768), 12345.0, device=DEVICE)
        fn, config = (
            _copy2d,
            {
                "block_sizes": [32, 32],
                "num_threads": [8, 8],
                "cute_vector_widths": [1, 4],
            },
        )
    elif case == "element_per_thread":
        # A thread per element of both tiles: the relabeled coordinates
        # open the body, ahead of the indices the grid strategy names there.
        source = torch.randn(
            2, 192, 128, device=DEVICE, dtype=torch.bfloat16
        ).transpose(1, 2)
        out = torch.full((2, 128, 192), 12345.0, device=DEVICE, dtype=torch.bfloat16)
        fn = _copy
        config = {"block_sizes": [1, 32, 32], "num_threads": [1, 32, 32]}
    else:
        source = torch.randn(
            4, 128, 16, 64, device=DEVICE, dtype=torch.bfloat16
        ).permute(0, 2, 3, 1)
        out = torch.full((4, 16, 64, 128), 12345.0, device=DEVICE, dtype=torch.bfloat16)
        fn = _copy4d
        config = {
            "block_sizes": [1, 1, 64, 64],
            "num_threads": [1, 1, 16, 8],
            "cute_vector_widths": [1, 1, 1, 8],
        }
    kernel = helion.kernel(
        fn,
        backend="cute",
        static_shapes=True,
        config=helion.Config.from_dict({**config, "cute_staged_load": True}),
    )
    bound = kernel.bind((source, out))
    code = bound.to_code(bound.config_spec.normalized_config(kernel.configs[0]))
    assert "_cta_image" in code
    kernel(source, out)
    torch.testing.assert_close(out, source, rtol=0, atol=0)
