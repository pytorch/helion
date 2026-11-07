from __future__ import annotations

import importlib

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA


def _check_full_domain_decode(
    columns: int,
    scale_group: int,
    scale_row_stride: int,
    scale_swizzle: bool,
    transpose: bool,
) -> None:
    """Compare every byte/scale pair and graph replay using raw BF16 bits.

    Replay mixes exceptional and ordinary scales within warps to force the
    uniform fallback. Bit comparison includes NaN and signed infinity cases.
    """
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the decoder uses SM100 tensor memory and scaled S2F6")

    cuda = importlib.import_module("cuda.bindings.driver")
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from helion._testing.cute_paged_attention_decode import launch

    tiles = (65536 + 128 * columns - 1) // (128 * columns)
    codes = torch.arange(
        tiles * 128 * columns, device=DEVICE, dtype=torch.int32
    ).reshape(tiles, 128, columns)
    raw = (codes % 256).to(torch.uint8).view(torch.int8)
    scales = (codes[:, :, ::scale_group] // 256).to(torch.uint8).contiguous()
    output = torch.empty((tiles, 128, columns // 2), device=DEVICE, dtype=torch.int32)
    args = [from_dlpack(t, assumed_align=16) for t in (raw, scales, output)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        launch,
        *args,
        stream,
        scale_row_stride,
        scale_swizzle,
        transpose,
        columns,
        scale_group,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    for changed in (False, True):
        if changed:
            raw.copy_(
                ((codes * 37 + (codes // 128) * 13 + 11) % 256)
                .to(torch.uint8)
                .view(torch.int8)
            )
            scales.copy_(
                (
                    (
                        codes[:, :, ::scale_group] // 256 * 73
                        + (torch.arange(128, device=DEVICE)[None, :, None] % 2) * 127
                        + torch.arange(columns // scale_group, device=DEVICE)[
                            None, None, :
                        ]
                        * 17
                        + 29
                    )
                    % 256
                ).to(torch.uint8)
            )
        graph.replay()
        expected = (
            raw.float()
            * torch.exp2(scales.float().repeat_interleave(scale_group, -1) - 127.0)
        ).to(torch.bfloat16)
        if transpose:
            expected = expected.transpose(-1, -2)
        expected = expected.contiguous()
        assert torch.equal(
            output.view(torch.bfloat16).view(torch.int16), expected.view(torch.int16)
        )


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "scale_row_stride, scale_swizzle", [(4, False), (8, False), (4, True)]
)
def test_transposed_byte_decode_full_domain_and_replay(
    scale_row_stride: int, scale_swizzle: bool
) -> None:
    """Native packed V staging and padded/composed indexed-load fallbacks."""
    _check_full_domain_decode(128, 32, scale_row_stride, scale_swizzle, True)


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "columns, scale_group, padding, swizzle",
    [
        (128, 32, 0, False),
        (64, 32, 0, False),
        (64, 16, 0, False),
        (128, 16, 0, False),
        (128, 64, 0, False),
        (96, 32, 0, False),
        (128, 32, 4, False),
        (128, 32, 0, True),
    ],
)
def test_row_byte_decode_grouped_stores(
    columns: int, scale_group: int, padding: int, swizzle: bool
) -> None:
    """Group batching, original store widths, and noncontiguous scale layouts."""
    _check_full_domain_decode(
        columns, scale_group, columns // scale_group + padding, swizzle, False
    )


@skipIfNotCUDA()
def test_packed_scale_guard_full_domain() -> None:
    """Exercise pair guards and every adjacent-byte position in the word guard."""
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the decoder uses SM100 tensor memory and scaled S2F6")
    cuda = importlib.import_module("cuda.bindings.driver")
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from helion._testing.cute_paged_attention_decode import launch_packed_guard

    low = torch.arange(65536, device=DEVICE, dtype=torch.int64)
    pairs = (low | 0xA5A50000).to(torch.int32)
    output = torch.empty((65536, 6), device=DEVICE, dtype=torch.int32)
    args = [from_dlpack(t, assumed_align=16) for t in (pairs, output)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(launch_packed_guard, *args, stream)
    expected_pair = ((low & 255) <= 247) & (((low >> 8) & 255) <= 247)
    for high in (torch.full_like(low, 0xA5A5), (low * 197 + 11) & 65535):
        words = low | (high << 16)
        pairs.copy_(words.to(torch.int32))
        compiled(*args, stream)
        assert torch.equal(output[:, 0].bool(), expected_pair)
        assert torch.equal(output[:, 1].bool(), expected_pair)
        # A byte permutation cannot change whether every byte is in range.
        # The second input mixes valid and exceptional high-byte values.
        expected_word = torch.ones_like(low, dtype=torch.bool)
        for byte in range(4):
            expected_word &= ((words >> (8 * byte)) & 255) <= 247
        for rotation in range(4):
            assert torch.equal(output[:, 2 + rotation].bool(), expected_word)
