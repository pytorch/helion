"""Evaluate CuTe's element-to-byte swizzle conversion entirely on the CPU."""

from __future__ import annotations

import ctypes
import itertools
from unittest.mock import patch

import numpy as np
import pytest

from test._cute_binding import _mock_cuda_unavailable

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


@cute.jit
def _translated_addresses(
    bits: cutlass.Constexpr,
    count: cutlass.Int32,
    address: cutlass.Int64,
) -> None:
    elements = cute.make_composed_layout(
        cute.make_swizzle(bits, 3, 3), 0, cute.make_layout((131072,), stride=(1,))
    )
    byte_swizzle = cute.recast_layout(8, cutlass.TFloat32.width, elements).inner
    bytes_layout = cute.make_composed_layout(
        byte_swizzle, 0, cute.make_layout((524288,), stride=(1,))
    )
    output = cute.make_tensor(
        cute.make_ptr(
            cutlass.Int32, address, cute.AddressSpace.generic, assumed_align=4
        ),
        cute.make_layout((count,), stride=(1,)),
    )
    for index in cutlass.range(count):
        output[index] = cute.crd2idx(index * 4, bytes_layout)


@pytest.mark.parametrize("bits", [1, 2, 3])
def test_recast_tf32_swizzle_preserves_vectors_and_stage_bounds(bits: int) -> None:
    # Include every supported A/B physical major order, K tile, stage count,
    # and the four phases possible with 1024-byte shared allocation alignment.
    count = (4 * 128 * 128 * 4 + 3072) // 4
    storage = (ctypes.c_int32 * count)()
    address = ctypes.addressof(storage)
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        # This is a host JIT function. It evaluates layout address arithmetic and
        # writes ordinary CPU memory; it contains no GPU kernel or launch.
        compiled = cute.compile(
            _translated_addresses,
            bits,
            cutlass.Int32(count),
            cutlass.Int64(address),
            options="--gpu-arch sm_100a",
        )
        compiled(count, address)

    observed = np.ctypeslib.as_array(storage)
    element_offsets = np.arange(count, dtype=np.int64)
    expected = 4 * (
        element_offsets ^ ((element_offsets >> 3) & (((1 << bits) - 1) << 3))
    )
    np.testing.assert_array_equal(observed, expected)
    vectors = observed.reshape(-1, 4)
    assert np.all(vectors[:, 0] % 16 == 0)
    np.testing.assert_array_equal(
        vectors - vectors[:, :1],
        np.broadcast_to(np.array([0, 4, 8, 12]), vectors.shape),
    )
    for rows, major, stages, base in itertools.product(
        (16, 32, 64, 128), (16, 32, 64, 128), (1, 2, 3, 4), (0, 1024, 2048, 3072)
    ):
        if min(3, (major // 8).bit_length() - 1) != bits:
            continue
        tile_elements = rows * major
        for stage in range(stages):
            begin = base // 4 + stage * tile_elements
            end = begin + tile_elements
            assert np.all(observed[begin:end] >= begin * 4)
            assert np.all(observed[begin:end] < end * 4)
