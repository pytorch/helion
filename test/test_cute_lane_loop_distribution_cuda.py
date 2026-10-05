"""GPU numerics for CuTe lane-loop distribution (see the GPU-free companion)."""

from __future__ import annotations

from examples.concatenate import concat2d_dim1_simple
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
    ("shape", "config"),
    [
        # Rolled x slice next to a persistent y slice (the study winner).
        (
            (2048, 512, 768),
            {
                "block_sizes": [1],
                "num_threads": [0, 4, 32],
                "reduction_loops": [256],
                "cute_vector_widths": [4, 2, 8],
                "cute_lane_layouts": ["blocked", "blocked", "blocked"],
            },
        ),
        # Two persistent slices with synthetic lanes each.
        (
            (256, 128, 256),
            {
                "block_sizes": [1],
                "num_threads": [0, 32, 32],
                "reduction_loops": [None],
                "cute_vector_widths": [1, 1, 1],
            },
        ),
        # Several rows per CTA, x slice fully threaded, y slice lane looped.
        (
            (256, 128, 256),
            {
                "block_sizes": [4],
                "num_threads": [4, 128, 2],
                "reduction_loops": [None],
                "cute_vector_widths": [1, 1, 1],
            },
        ),
        # The default config at the study's shape 0.
        (
            (256, 128, 256),
            {
                "block_sizes": [32],
                "num_threads": [0, 0, 0],
                "reduction_loops": [32],
                "cute_vector_widths": [1, 1, 1],
            },
        ),
    ],
)
def test_concat_simple_matches_torch_cat(
    shape: tuple[int, int, int], config: dict[str, object]
) -> None:
    m, n1, n2 = shape
    kernel = helion.kernel(
        concat2d_dim1_simple.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    x = torch.randn((m, n1), device=CUDA_DEVICE)
    y = torch.randn((m, n2), device=CUDA_DEVICE)
    out = _run(kernel, (x, y), **config)
    torch.testing.assert_close(out, torch.cat([x, y], dim=1), rtol=0, atol=0)


_ROW_CONFIG = {
    "block_sizes": [1, 1024],
    "num_threads": [0, 256],
    "cute_vector_widths": [1, 4],
}
# Four rows per thread around the vector lane loop: two live lane loops.
_NESTED_CONFIG = {
    "block_sizes": [4, 256],
    "num_threads": [1, 64],
    "cute_vector_widths": [1, 4],
}
# A thread-owned leading axis, a plain lane loop and the vector lane loop.
_NESTED_3D_CONFIG = {
    "block_sizes": [2, 4, 256],
    "num_threads": [2, 1, 64],
    "cute_vector_widths": [1, 1, 4],
}
_CONFIG_IDS = ["one_loop", "two_loops", "three_dims"]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gather_and_echo(
    idx: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty([idx.size(0), w.size(1)], dtype=w.dtype, device=w.device)
    echo = torch.empty([idx.size(0)], dtype=idx.dtype, device=idx.device)
    for tile0, tile1 in hl.tile(out.size()):
        rows = idx[tile0]
        echo[tile0] = rows
        out[tile0, tile1] = w[rows, tile1]
    return out, echo


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _gather_and_echo_3d(
    idx: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(
        [idx.size(0), w.size(1), w.size(2)], dtype=w.dtype, device=w.device
    )
    echo = torch.empty([idx.size(0)], dtype=idx.dtype, device=idx.device)
    for tile0, tile1, tile2 in hl.tile(out.size()):
        rows = idx[tile0]
        echo[tile0] = rows
        out[tile0, tile1, tile2] = w[rows, tile1, tile2]
    return out, echo


@pytest.mark.parametrize(
    ("kernel", "shapes", "config"),
    [
        (_gather_and_echo, ((8,), (16, 1024)), _ROW_CONFIG),
        (_gather_and_echo, ((8,), (16, 1024)), _NESTED_CONFIG),
        (_gather_and_echo_3d, ((4,), (16, 4, 256)), _NESTED_3D_CONFIG),
    ],
    ids=_CONFIG_IDS,
)
def test_gathered_row_read_twice_matches_reference(
    kernel: object,
    shapes: tuple[tuple[int, ...], tuple[int, ...]],
    config: dict[str, object],
) -> None:
    idx_shape, w_shape = shapes
    idx = torch.randint(0, w_shape[0], idx_shape, device=CUDA_DEVICE)
    w = torch.randn(w_shape, device=CUDA_DEVICE)
    out, echo = _run(kernel, (idx, w), **config)
    torch.testing.assert_close(out, w[idx], rtol=0, atol=0)
    torch.testing.assert_close(echo, idx, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_flag_and_count(
    x: torch.Tensor, flags: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    cnt = torch.empty_like(flags)
    for tile0, tile1 in hl.tile(out.size()):
        f = flags[tile0]
        out[tile0, tile1] = hl.load(x, [tile0, tile1], extra_mask=(f > 0)[:, None])
        cnt[tile0] = f + 1
    return out, cnt


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_flag_and_count_3d(
    x: torch.Tensor, flags: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    cnt = torch.empty_like(flags)
    for tile0, tile1, tile2 in hl.tile(out.size()):
        f = flags[tile0]
        out[tile0, tile1, tile2] = hl.load(
            x, [tile0, tile1, tile2], extra_mask=(f > 0)[:, None, None]
        )
        cnt[tile0] = f + 1
    return out, cnt


@pytest.mark.parametrize(
    ("kernel", "shape", "config"),
    [
        (_row_flag_and_count, (8, 1024), _ROW_CONFIG),
        (_row_flag_and_count, (8, 1024), _NESTED_CONFIG),
        (_row_flag_and_count_3d, (4, 4, 256), _NESTED_3D_CONFIG),
    ],
    ids=_CONFIG_IDS,
)
def test_row_flag_reused_after_the_masked_load_matches_reference(
    kernel: object, shape: tuple[int, ...], config: dict[str, object]
) -> None:
    x = torch.randn(shape, device=CUDA_DEVICE)
    flags = torch.randint(-1, 2, shape[:1], dtype=torch.int32, device=CUDA_DEVICE)
    out, cnt = _run(kernel, (x, flags), **config)
    keep = (flags > 0).reshape(-1, *([1] * (len(shape) - 1)))
    torch.testing.assert_close(out, torch.where(keep, x, 0.0), rtol=0, atol=0)
    torch.testing.assert_close(cnt, flags + 1, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _read_then_zero(x: torch.Tensor) -> torch.Tensor:
    # ``x`` holds the rows to copy followed by as many spare rows.  Each tile
    # zeroes the spare row of its first row, which no tile reads, so the
    # result does not depend on cross-thread timing.
    rows = x.size(0) // 2
    out = torch.empty([rows, x.size(1)], dtype=x.dtype, device=x.device)
    for tile0, tile1 in hl.tile(out.size()):
        v = x[tile0, tile1]
        x[tile0.begin + rows, 0] = 0.0
        out[tile0, tile1] = v
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _read_then_zero_3d(x: torch.Tensor) -> torch.Tensor:
    rows = x.size(1) // 2
    out = torch.empty([x.size(0), rows, x.size(2)], dtype=x.dtype, device=x.device)
    for tile0, tile1, tile2 in hl.tile(out.size()):
        v = x[tile0, tile1, tile2]
        x[tile0, tile1.begin + rows, 0] = 0.0
        out[tile0, tile1, tile2] = v
    return out


@pytest.mark.parametrize(
    ("kernel", "shape", "axis", "config"),
    [
        (_read_then_zero, (16, 1024), 0, _ROW_CONFIG),
        (_read_then_zero, (16, 256), 0, _NESTED_CONFIG),
        (_read_then_zero_3d, (2, 8, 256), 1, _NESTED_3D_CONFIG),
    ],
    ids=_CONFIG_IDS,
)
def test_store_between_a_hoisted_load_and_its_use_matches_reference(
    kernel: object, shape: tuple[int, ...], axis: int, config: dict[str, list[int]]
) -> None:
    # The store into ``x`` follows the packet loop nest (see the GPU-free
    # companion); the copied rows and the zeroed spare rows must both match.
    x = torch.randn(shape, device=CUDA_DEVICE)
    original = x.clone()
    out = _run(kernel, (x,), **config)
    rows = shape[axis] // 2
    torch.testing.assert_close(out, original.narrow(axis, 0, rows), rtol=0, atol=0)
    expected = original.clone()
    block = config["block_sizes"][axis]
    for begin in range(0, rows, block):
        expected.select(axis, rows + begin)[..., 0] = 0.0
    torch.testing.assert_close(x, expected, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _convert_bytes_then_zero(packed: torch.Tensor) -> torch.Tensor:
    # Each program's tile spans its whole row, so the element it zeroes is
    # overwritten by its own tile store; the result does not depend on
    # cross-program timing.
    out = torch.empty(packed.shape, dtype=torch.bfloat16, device=packed.device)
    for tile0, tile1 in hl.tile(packed.shape):
        y = packed[tile0, tile1].to(torch.bfloat16)
        out[tile0.begin, 0] = 0.0
        out[tile0, tile1] = y
    return out


# One row per program; 32 threads own four bytes each per lane iteration, so
# a 256-wide row takes two lane iterations per thread.
_BYTE_CASES = [
    pytest.param(columns, packet_flush, id=f"{lanes}-{protocol}")
    for lanes, columns in (("one_lane", 128), ("two_lanes", 256))
    for protocol, packet_flush in (("values", False), ("packet", True))
]


@pytest.mark.parametrize(("columns", "packet_flush"), _BYTE_CASES)
def test_store_between_a_byte_conversion_and_its_flush_matches_reference(
    columns: int, packet_flush: bool
) -> None:
    # The zeroing store precedes the lane loop (see the GPU-free companion),
    # so the flush overwrites the zero.  Kept in the nest, the second lane
    # iteration zeroed the element again after the first iteration's flush.
    packed = torch.randint(
        -128, 128, (8, columns), dtype=torch.int8, device=CUDA_DEVICE
    )
    packed[:, 0] = -3  # The zeroed element must differ from its value.
    config = helion.Config(
        block_sizes=[1, columns],
        num_threads=[0, 32],
        cute_vector_widths=[1, 4],
        cute_signed_bitfield_bf16=packet_flush,
    )
    bound = _convert_bytes_then_zero.bind((packed,))
    code = bound.to_code(config)
    assert ("_cute_signed_bitfield_to_bf16_packed(" in code) is packet_flush, code
    out = bound.compile_config(config)(packed)
    torch.testing.assert_close(out, packed.to(torch.bfloat16), rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _copy_zero_copy(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Each program's tile spans its whole row, so the element it zeroes was
    # written by its own tile store; the result does not depend on
    # cross-program timing.
    out = torch.empty(packed.shape, dtype=torch.bfloat16, device=packed.device)
    out2 = torch.empty(packed.shape, dtype=torch.bfloat16, device=packed.device)
    for tile0, tile1 in hl.tile(packed.shape):
        y = packed[tile0, tile1].to(torch.bfloat16)
        out[tile0, tile1] = y
        out[tile0.begin, 0] = 0.0
        out2[tile0, tile1] = y
    return out, out2


@pytest.mark.parametrize("packet_flush", [False, True], ids=["values", "packet"])
def test_store_after_a_per_lane_store_of_its_tensor_matches_reference(
    packet_flush: bool,
) -> None:
    # The zeroing store follows the loop (see the GPU-free companion): the
    # copy's element is zero and the second copy is untouched, over two lane
    # iterations per thread.
    packed = torch.randint(-128, 128, (8, 256), dtype=torch.int8, device=CUDA_DEVICE)
    packed[:, 0] = -3
    config = helion.Config(
        block_sizes=[1, 256],
        num_threads=[0, 32],
        cute_vector_widths=[1, 4],
        cute_signed_bitfield_bf16=packet_flush,
    )
    out, out2 = _copy_zero_copy.bind((packed,)).compile_config(config)(packed)
    expected = packed.to(torch.bfloat16)
    torch.testing.assert_close(out2, expected, rtol=0, atol=0)
    expected[:, 0] = 0.0
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _read_then_copy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros_like(x)
    first = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile0, tile1 in hl.tile(x.size()):
        first[tile0] = out[tile0, 0]
        out[tile0, tile1] = x[tile0, tile1]
    return out, first


def test_read_before_a_flushed_store_matches_reference() -> None:
    # ``first`` reads the zero-initialized output before the copy overwrites
    # it; a read emitted after the flushed copy would see ``x`` instead.
    x = torch.randn((8, 1024), device=CUDA_DEVICE)
    bound = _read_then_copy.bind((x,))
    out, first = bound.compile_config(helion.Config.from_dict(_ROW_CONFIG))(x)
    torch.testing.assert_close(out, x, rtol=0, atol=0)
    torch.testing.assert_close(first, torch.zeros_like(first), rtol=0, atol=0)
