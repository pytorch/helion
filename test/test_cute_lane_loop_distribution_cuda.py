"""GPU numerics for CuTe lane-loop distribution (see the GPU-free companion)."""

from __future__ import annotations

from examples.concatenate import concat2d_dim1_simple
import pytest
import torch

from test.test_cute_lane_loop_distribution import _BACKWARDS_KERNELS
from test.test_cute_lane_loop_distribution import _COLUMN_LANES_ON_ONE_THREAD_CONFIG
from test.test_cute_lane_loop_distribution import _DEVICE_LOOP_COLUMNS_CONFIG
from test.test_cute_lane_loop_distribution import _DEVICE_LOOP_ROWS_PER_THREAD_CONFIG
from test.test_cute_lane_loop_distribution import _EIGHT_LANES_PER_THREAD_CONFIG
from test.test_cute_lane_loop_distribution import _INNER_LANES_CONFIG
from test.test_cute_lane_loop_distribution import _INNER_THREADED_COLUMNS_CONFIG
from test.test_cute_lane_loop_distribution import _NESTED_CONFIG
from test.test_cute_lane_loop_distribution import _NESTED_SCALAR_CONFIG
from test.test_cute_lane_loop_distribution import _ONE_COLUMN_PER_THREAD_CONFIG
from test.test_cute_lane_loop_distribution import _ONE_ELEMENT_PER_THREAD_CONFIG
from test.test_cute_lane_loop_distribution import _ONE_PACKET_PER_THREAD_CONFIG
from test.test_cute_lane_loop_distribution import _REPEATED_DEVICE_LOOP_KERNELS
from test.test_cute_lane_loop_distribution import _REPEATED_INNER_LOOP_CONFIG
from test.test_cute_lane_loop_distribution import _REPEATED_INNER_LOOP_WIDE_CONFIG
from test.test_cute_lane_loop_distribution import _ROW_LANES_ON_ONE_THREAD_CONFIG
from test.test_cute_lane_loop_distribution import _ROWS_ON_TWO_LANES_CONFIG
from test.test_cute_lane_loop_distribution import _ROWS_ONE_PER_THREAD_CONFIG
from test.test_cute_lane_loop_distribution import _SHARED_BLOCK_SIZE_CONFIG
from test.test_cute_lane_loop_distribution import _SHORT_BLOCK_ON_ONE_THREAD_CONFIG
from test.test_cute_lane_loop_distribution import _SHORT_SIBLING_LOOP_CONFIG
from test.test_cute_lane_loop_distribution import _THREADED_PLANES_CONFIG
from test.test_cute_lane_loop_distribution import _THREADED_ROWS_CONFIG
from test.test_cute_lane_loop_distribution import _THREADED_ROWS_SCALAR_CONFIG
from test.test_cute_lane_loop_distribution import _THREADED_TILE_CONFIG
from test.test_cute_lane_loop_distribution import _THREE_LOOPS_CONFIG
from test.test_cute_lane_loop_distribution import _THREE_LOOPS_SCALAR_CONFIG
from test.test_cute_lane_loop_distribution import _TWO_COLUMNS_PER_THREAD_CONFIG
from test.test_cute_lane_loop_distribution import _TWO_PACKETS_PER_THREAD_CONFIG
from test.test_cute_lane_loop_distribution import _VECTORIZED_DEVICE_LOOP_COLUMNS_CONFIG
from test.test_cute_lane_loop_distribution import _carry_previous_column
from test.test_cute_lane_loop_distribution import _carry_to_the_next_tile
from test.test_cute_lane_loop_distribution import _copy_then_zero_first_column
from test.test_cute_lane_loop_distribution import (
    _copy_then_zero_first_column_in_inner_tile,
)
from test.test_cute_lane_loop_distribution import _copy_then_zero_first_row
from test.test_cute_lane_loop_distribution import _double_first_column_beside_copy
from test.test_cute_lane_loop_distribution import _first_column_then_update_all
from test.test_cute_lane_loop_distribution import _first_column_then_update_all_3d
from test.test_cute_lane_loop_distribution import _first_row_increment_beside_copy
from test.test_cute_lane_loop_distribution import _first_row_scaled_beside_copy
from test.test_cute_lane_loop_distribution import _first_row_then_update_all
from test.test_cute_lane_loop_distribution import _first_row_then_update_all_plus_one
from test.test_cute_lane_loop_distribution import _first_row_to_vector
from test.test_cute_lane_loop_distribution import _gather_rows_then_zero
from test.test_cute_lane_loop_distribution import _gather_then_zero
from test.test_cute_lane_loop_distribution import _increment_segment_rows
from test.test_cute_lane_loop_distribution import _inner_column_zero_then_update_all
from test.test_cute_lane_loop_distribution import _inner_first_column_then_update_all
from test.test_cute_lane_loop_distribution import _read_fourth_next_then_store
from test.test_cute_lane_loop_distribution import _read_modify_write_in_a_device_loop
from test.test_cute_lane_loop_distribution import _read_next_column_then_store
from test.test_cute_lane_loop_distribution import _read_next_then_store
from test.test_cute_lane_loop_distribution import (
    _read_previous_row_at_the_next_tiles_column_then_store,
)
from test.test_cute_lane_loop_distribution import (
    _read_previous_row_then_store_in_a_device_loop,
)
from test.test_cute_lane_loop_distribution import _read_previous_then_store
from test.test_cute_lane_loop_distribution import _read_reversed_then_store
from test.test_cute_lane_loop_distribution import _read_wrapped_then_store
from test.test_cute_lane_loop_distribution import _repeat_first_column_update
from test.test_cute_lane_loop_distribution import _short_loop_then_read_ahead_and_store
from test.test_cute_lane_loop_distribution import _six_loops_over_one_block_size
from test.test_cute_lane_loop_distribution import _store_flattened_rows
from test.test_cute_lane_loop_distribution import _store_flattened_rows_overlapping
from test.test_cute_lane_loop_distribution import (
    _store_flattened_rows_overlapping_dynamic,
)
from test.test_cute_lane_loop_distribution import _store_rows_shifted_by_the_iteration
from test.test_cute_lane_loop_distribution import _store_then_read_moving_backwards
from test.test_cute_lane_loop_distribution import _two_loops_over_one_block_size
from test.test_cute_lane_loop_distribution import _two_loops_over_one_block_size_war
from test.test_cute_lane_loop_distribution import (
    _uniform_read_modify_write_in_a_device_loop,
)
from test.test_cute_lane_loop_distribution import (
    _update_under_a_loop_indexed_by_its_begin,
)
from test.test_cute_lane_loop_distribution import (
    _zero_first_column_then_copy_in_inner_tile,
)
from test.test_cute_lane_loop_distribution import _zero_first_column_then_copy_plus_one
from test.test_cute_lane_loop_distribution import _zero_first_column_then_read
from test.test_cute_lane_loop_distribution import _zero_first_row_then_copy
from test.test_cute_lane_loop_distribution import _zero_first_row_then_copy_plus_one

import helion
from helion import exc
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
    # cross-program timing.  The last column belongs to the last lane
    # iteration of the last thread: a zeroing store left inside the loop is
    # overwritten by that iteration's flush, whereas the first column is
    # flushed in the first iteration and re-zeroed by the second.
    out = torch.empty(packed.shape, dtype=torch.bfloat16, device=packed.device)
    out2 = torch.empty(packed.shape, dtype=torch.bfloat16, device=packed.device)
    for tile0, tile1 in hl.tile(packed.shape):
        y = packed[tile0, tile1].to(torch.bfloat16)
        out[tile0, tile1] = y
        out[tile0.begin, 255] = 0.0
        out2[tile0, tile1] = y
    return out, out2


@pytest.mark.parametrize("packet_flush", [False, True], ids=["values", "packet"])
def test_store_after_a_per_lane_store_of_its_tensor_matches_reference(
    packet_flush: bool,
) -> None:
    # The zeroing store follows the loop (see the GPU-free companion): the
    # copy's last element is zero and the second copy is untouched, over two
    # lane iterations per thread.
    packed = torch.randint(-128, 128, (8, 256), dtype=torch.int8, device=CUDA_DEVICE)
    packed[:, 255] = -3
    config = helion.Config(
        block_sizes=[1, 256],
        num_threads=[0, 32],
        cute_vector_widths=[1, 4],
        cute_signed_bitfield_bf16=packet_flush,
    )
    out, out2 = _copy_zero_copy.bind((packed,)).compile_config(config)(packed)
    expected = packed.to(torch.bfloat16)
    torch.testing.assert_close(out2, expected, rtol=0, atol=0)
    expected[:, 255] = 0.0
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


def _small_integers(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randint(0, 5, shape, device=CUDA_DEVICE).float()


@pytest.mark.parametrize(
    "config", [_NESTED_CONFIG, _NESTED_SCALAR_CONFIG], ids=["vector", "scalar"]
)
def test_masked_tile_uniform_accesses_around_per_lane_stores_reject_the_config(
    config: dict[str, object],
) -> None:
    # On a partial tile the first row's load and the zeroing store read the
    # row mask and stay inside the row loop, where they would repeat around
    # the other rows' stores (see the GPU-free companion).
    x = _small_integers((6, 250))
    with pytest.raises(
        exc.BackendUnsupported, match="lane-invariant load of x would repeat"
    ):
        _run(_first_row_then_update_all, (x,), **config)
    out = torch.full((6, 250), -7.0, device=CUDA_DEVICE)
    with pytest.raises(
        exc.BackendUnsupported, match="lane-invariant store to out would repeat"
    ):
        _run(_zero_first_row_then_copy, (x, out), **config)


@pytest.mark.parametrize(
    "config", [_NESTED_CONFIG, _NESTED_SCALAR_CONFIG], ids=["vector", "scalar"]
)
def test_masked_tile_uniform_store_after_per_lane_stores_matches_reference(
    config: dict[str, object],
) -> None:
    x = _small_integers((6, 250))
    out = torch.full((6, 250), -7.0, device=CUDA_DEVICE)
    result = _run(_copy_then_zero_first_row, (x, out), **config)
    expected = x.clone()
    expected[::4] = 0.0
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_tile_uniform_store_in_an_inner_tile_loop_matches_reference() -> None:
    x = _small_integers((8, 512))
    out = torch.full((8, 512), -7.0, device=CUDA_DEVICE)
    with pytest.raises(
        exc.BackendUnsupported, match="lane-invariant store to out would repeat"
    ):
        _run(
            _zero_first_column_then_copy_in_inner_tile, (x, out), **_INNER_LANES_CONFIG
        )
    result = _run(
        _copy_then_zero_first_column_in_inner_tile, (x, out), **_INNER_LANES_CONFIG
    )
    expected = x.clone()
    expected[:, ::256] = 0.0
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_placed_nest_repeating_a_tile_uniform_access_rejects_the_config(
    shape: tuple[int, int],
) -> None:
    # A loop-invariant constant leaves the lane loops, so these bodies are
    # emitted as placements; the first row's access still runs once per row
    # iteration of the loop the copy's packet nests it in (see the GPU-free
    # companion).
    x = _small_integers(shape)
    with pytest.raises(
        exc.BackendUnsupported, match="lane-invariant load of x would repeat"
    ):
        _run(_first_row_then_update_all_plus_one, (x,), **_NESTED_CONFIG)
    out = torch.full(shape, -7.0, device=CUDA_DEVICE)
    with pytest.raises(
        exc.BackendUnsupported, match="lane-invariant store to out would repeat"
    ):
        _run(_zero_first_row_then_copy_plus_one, (x, out), **_NESTED_CONFIG)
    with pytest.raises(
        exc.BackendUnsupported, match="lane-invariant load of x would repeat"
    ):
        args = (x, _small_integers(shape), out)
        _run(_first_row_increment_beside_copy, args, **_NESTED_CONFIG)


@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_scaled_first_row_beside_a_copy_matches_reference(
    shape: tuple[int, int],
) -> None:
    x = _small_integers(shape)
    out = torch.full(shape, -7.0, device=CUDA_DEVICE)
    y = torch.full(shape, -7.0, device=CUDA_DEVICE)
    _run(_first_row_scaled_beside_copy, (x, out, y), **_NESTED_CONFIG)
    expected = torch.full(shape, -7.0, device=CUDA_DEVICE)
    expected[::4] = 2 * x[::4]
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(y, x, rtol=0, atol=0)


@pytest.mark.parametrize(
    "config",
    [
        _NESTED_CONFIG,
        _NESTED_SCALAR_CONFIG,
        {**_NESTED_SCALAR_CONFIG, "num_threads": [2, 64]},
        {
            **_NESTED_CONFIG,
            "num_threads": [2, 64],
            "cute_lane_layouts": ["strided", "strided"],
        },
    ],
    ids=["vector", "scalar", "two_row_threads", "strided"],
)
def test_first_row_to_vector_stores_a_first_row_value_in_every_column(
    config: dict[str, object],
) -> None:
    # Guarded by the row mask as well, the first row's load was zero in the
    # lanes past the last row, and the store of ``first`` (guarded by the
    # column mask only) raced that zero in.
    x = torch.randint(1, 6, (6, 250), device=CUDA_DEVICE).float()
    first = torch.full((250,), -7.0, device=CUDA_DEVICE)
    out = torch.full((6, 250), -7.0, device=CUDA_DEVICE)
    result = _run(_first_row_to_vector, (x, first, out), **config)
    torch.testing.assert_close(result, x, rtol=0, atol=0)
    assert bool(((first == x[0]) | (first == x[4])).all()), first


# ---------------------------------------------------------------------------
# Accesses that threads sharing a tile axis could reorder are separated by a
# block-wide barrier (see the GPU-free companion).  These races surfaced as
# wrong values before the barriers; the ones on a full tile depend on timing.


def _first_of_each_tile(extent: int, block: int) -> torch.Tensor:
    return (torch.arange(extent, device=CUDA_DEVICE) // block) * block


@pytest.mark.parametrize(
    "config",
    [
        _NESTED_CONFIG,
        _NESTED_SCALAR_CONFIG,
        {**_NESTED_CONFIG, "num_threads": [2, 64]},
        {
            **_NESTED_CONFIG,
            "num_threads": [2, 64],
            "cute_lane_layouts": ["strided", "strided"],
        },
    ],
    ids=["vector", "scalar", "two_row_threads", "strided"],
)
@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_uniform_read_before_per_lane_stores_on_a_shared_axis_matches_reference(
    shape: tuple[int, int], config: dict[str, object]
) -> None:
    x = _small_integers(shape)
    expected = x + x[:, _first_of_each_tile(shape[1], 256)] + 1.0
    result = _run(_first_column_then_update_all, (x,), **config)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "config",
    [_THREADED_ROWS_CONFIG, _THREADED_ROWS_SCALAR_CONFIG],
    ids=["vector", "scalar"],
)
@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_uniform_read_on_a_thread_axis_without_a_lane_loop_matches_reference(
    shape: tuple[int, int], config: dict[str, object]
) -> None:
    x = _small_integers(shape)
    expected = x + x[_first_of_each_tile(shape[0], 4)] + 1.0
    result = _run(_first_row_then_update_all_plus_one, (x,), **config)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("config", "block", "shapes"),
    [
        (_THREE_LOOPS_SCALAR_CONFIG, 128, [(4, 8, 256), (3, 6, 250)]),
        (_THREE_LOOPS_CONFIG, 128, [(4, 8, 256), (3, 6, 250)]),
        (_THREADED_PLANES_CONFIG, 8, [(4, 256, 16), (3, 250, 13)]),
    ],
    ids=["two_threads_per_plane", "threaded_planes", "planes_on_y_threads"],
)
def test_uniform_read_hoisted_out_of_the_innermost_shared_loop_matches_reference(
    config: dict[str, object], block: int, shapes: list[tuple[int, int, int]]
) -> None:
    # Review 8's kernel: wrong under every one of these configs without the
    # barrier (the column axis spans several warps).
    for shape in shapes:
        x = _small_integers(shape)
        expected = x + x[:, :, _first_of_each_tile(shape[2], block)] + 1.0
        result = _run(_first_column_then_update_all_3d, (x,), **config)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "config", [_NESTED_CONFIG, _NESTED_SCALAR_CONFIG], ids=["vector", "scalar"]
)
@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_uniform_load_then_uniform_store_of_one_element_matches_reference(
    shape: tuple[int, int], config: dict[str, object]
) -> None:
    # Every column thread doubles the first column's element once.
    x = _small_integers(shape)
    out = torch.randint(1, 6, shape, device=CUDA_DEVICE).float()
    y = torch.full(shape, -7.0, device=CUDA_DEVICE)
    expected = out.clone()
    expected[:, ::256] *= 2.0
    result = _run(_double_first_column_beside_copy, (x, out, y), **config)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(result, x, rtol=0, atol=0)


@pytest.mark.parametrize(
    "config", [_NESTED_CONFIG, _NESTED_SCALAR_CONFIG], ids=["vector", "scalar"]
)
@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_uniform_store_before_per_lane_stores_matches_reference(
    shape: tuple[int, int], config: dict[str, object]
) -> None:
    x = _small_integers(shape)
    out = torch.full(shape, -7.0, device=CUDA_DEVICE)
    result = _run(_zero_first_column_then_copy_plus_one, (x, out), **config)
    torch.testing.assert_close(result, x + 1.0, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_idempotent_uniform_stores_match_reference(shape: tuple[int, int]) -> None:
    x = _small_integers(shape)
    out = torch.full(shape, -7.0, device=CUDA_DEVICE)
    result = _run(_copy_then_zero_first_column, (x, out), **_NESTED_CONFIG)
    expected = x.clone()
    expected[:, ::256] = 0.0
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    out = torch.randint(1, 6, shape, device=CUDA_DEVICE).float()
    y = torch.full(shape, -7.0, device=CUDA_DEVICE)
    result = _run(_zero_first_column_then_read, (x, out, y), **_NESTED_CONFIG)
    expected = out.clone()
    expected[:, ::256] = 0.0
    torch.testing.assert_close(result, expected + x, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_device_loop_body_with_a_thread_axis_matches_reference(
    shape: tuple[int, int],
) -> None:
    x = _small_integers(shape)
    expected = x + x[:, _first_of_each_tile(shape[1], 64)] + 1.0
    result = _run(
        _inner_first_column_then_update_all, (x,), **_INNER_THREADED_COLUMNS_CONFIG
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_loop_invariant_read_in_a_device_loop_matches_reference(
    shape: tuple[int, int],
) -> None:
    # Iteration j + 1's read of column 0 sees iteration 0's update of it.
    x = _small_integers(shape)
    expected = x.clone()
    for start in range(0, shape[1], 64):
        first = expected[:, 0].clone()
        expected[:, start : start + 64] += first[:, None] + 1.0
    result = _run(
        _inner_column_zero_then_update_all, (x,), **_INNER_THREADED_COLUMNS_CONFIG
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_gathered_read_before_a_per_lane_store_matches_reference(
    shape: tuple[int, int],
) -> None:
    # One thread per element: every thread's gather of the reversed column
    # precedes every thread's store to its own column.  (With lane loops the
    # config is rejected: the CPU test.)  The columns are reversed within
    # each tile: the barrier orders the threads of one block, and a gather
    # from another block's columns would race with that block's stores.
    x = _small_integers(shape)
    width = _THREADED_TILE_CONFIG["block_sizes"][1]
    perm = torch.cat(
        [
            torch.arange(
                min(start + width, shape[1]) - 1, start - 1, -1, device=CUDA_DEVICE
            )
            for start in range(0, shape[1], width)
        ]
    )
    y = torch.full(shape, -7.0, device=CUDA_DEVICE)
    expected = x[:, perm]
    result = _run(_gather_then_zero, (x, perm, y), **_THREADED_TILE_CONFIG)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert not x.any()


def _compiled(kernel: object, args: tuple[object, ...], **config: object) -> object:
    bound = kernel.bind(args)  # pyrefly: ignore [missing-attribute]
    return bound.compile_config(helion.Config.from_dict(config))


_RUNS = 5


@pytest.mark.parametrize("shape", [(8, 257), (6, 251), (16, 1025)])
def test_carry_from_the_previous_tile_matches_reference(
    shape: tuple[int, int],
) -> None:
    # Iteration j's uniform read of column begin - 1 is iteration j - 1's
    # per-thread store of it: the barrier closing the body orders them.
    # Several runs: without it the outcome depends on the warps' timing.
    run = _compiled(
        _carry_previous_column, (_small_integers(shape),), **_DEVICE_LOOP_COLUMNS_CONFIG
    )
    for _ in range(_RUNS):
        x = _small_integers(shape)
        expected = x.clone()
        for start in range(1, shape[1], 64):
            carry = expected[:, start - 1].clone()
            expected[:, start : start + 64] += carry[:, None]
        torch.testing.assert_close(run(x), expected, rtol=0, atol=0)  # pyrefly: ignore [not-callable]


@pytest.mark.parametrize("shape", [(8, 257), (6, 251), (16, 1025)])
def test_carry_to_the_next_tile_matches_reference(shape: tuple[int, int]) -> None:
    # Iteration j stores the first column of tile j + 1 per thread, which
    # iteration j + 1 reads uniformly.
    run = _compiled(
        _carry_to_the_next_tile,
        (_small_integers(shape),),
        **_DEVICE_LOOP_COLUMNS_CONFIG,
    )
    n = shape[1] - 1
    for _ in range(_RUNS):
        x = _small_integers(shape)
        expected = x.clone()
        for start in range(0, n, 64):
            carry = expected[:, start].clone()
            expected[:, start + 1 : min(start + 64, n) + 1] += carry[:, None]
        torch.testing.assert_close(run(x), expected, rtol=0, atol=0)  # pyrefly: ignore [not-callable]


@pytest.mark.parametrize(
    "config",
    [_REPEATED_INNER_LOOP_CONFIG, _REPEATED_INNER_LOOP_WIDE_CONFIG],
    ids=["two_row_threads", "one_row_thread"],
)
@pytest.mark.parametrize("shape", [(8, 64), (6, 60), (8, 128)])
def test_repeated_inner_loop_matches_reference(
    shape: tuple[int, int], config: dict[str, object]
) -> None:
    # Outer iteration j + 1's inner loop reads column 0, which outer
    # iteration j's inner loop stored from one thread: the barrier closing the
    # outer body orders them (with one inner iteration nothing else does).
    block = config["block_sizes"][2]  # pyrefly: ignore [bad-index, unsupported-operation]
    run = _compiled(_repeat_first_column_update, (_small_integers(shape),), **config)
    for _ in range(_RUNS):
        x = _small_integers(shape)
        expected = x.clone()
        for _repeat in range(8):
            for start in range(0, shape[1], block):
                first = expected[:, start].clone()
                expected[:, start : start + block] += first[:, None]
        torch.testing.assert_close(run(x), expected, rtol=0, atol=0)  # pyrefly: ignore [not-callable]


@pytest.mark.parametrize("shape", [(8, 320), (6, 314), (16, 1088)])
def test_two_loops_over_one_block_size_match_reference(shape: tuple[int, int]) -> None:
    # Loop 2's uniform read of the next tile's first column is loop 1's
    # per-thread update of it (and the mirror: loop 2's per-thread update
    # must not overtake loop 1's read): the barrier between the loops.
    n = shape[1] - 64
    columns = torch.arange(n, device=CUDA_DEVICE)
    source = (columns // 64) * 64 + 64
    run = _compiled(
        _two_loops_over_one_block_size,
        (_small_integers(shape), _small_integers(shape)),
        **_SHARED_BLOCK_SIZE_CONFIG,
    )
    mirror = _compiled(
        _two_loops_over_one_block_size_war,
        (_small_integers(shape), _small_integers(shape)),
        **_SHARED_BLOCK_SIZE_CONFIG,
    )
    for _ in range(_RUNS):
        x = _small_integers(shape)
        y = torch.full(shape, -7.0, device=CUDA_DEVICE)
        updated = x.clone()
        updated[:, :n] += 1.0
        expected = y.clone()
        expected[:, :n] = updated[:, source]
        torch.testing.assert_close(run(x, y), expected, rtol=0, atol=0)  # pyrefly: ignore [not-callable]
        torch.testing.assert_close(x, updated, rtol=0, atol=0)
        x = _small_integers(shape)
        y = torch.full(shape, -7.0, device=CUDA_DEVICE)
        expected = y.clone()
        expected[:, :n] = x[:, source]
        updated = x.clone()
        updated[:, :n] += 1.0
        torch.testing.assert_close(mirror(x, y), expected, rtol=0, atol=0)  # pyrefly: ignore [not-callable]
        torch.testing.assert_close(x, updated, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(8, 320), (6, 314)])
def test_six_loops_over_one_block_size_match_reference(shape: tuple[int, int]) -> None:
    # Loop k + 1's uniform read of the next tile's first column is loop k's
    # per-thread update of it: the barrier between each pair of loops.
    n = shape[1] - 64
    columns = torch.arange(n, device=CUDA_DEVICE)
    source = (columns // 64) * 64 + 64
    run = _compiled(
        _six_loops_over_one_block_size,
        (_small_integers(shape), _small_integers(shape)),
        **_SHARED_BLOCK_SIZE_CONFIG,
    )
    for _ in range(_RUNS):
        x = _small_integers(shape)
        y = torch.full(shape, -7.0, device=CUDA_DEVICE)
        updated = x.clone()
        updated[:, :n] += 1.0
        for _loop in range(4):
            updated[:, :n] = updated[:, :n] + updated[:, source]
        expected = y.clone()
        expected[:, :n] = updated[:, source]
        torch.testing.assert_close(run(x, y), expected, rtol=0, atol=0)  # pyrefly: ignore [not-callable]
        torch.testing.assert_close(x, updated, rtol=0, atol=0)


def _backwards_reference(
    kernel: object, x: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """``out`` after the program, its device loop iterated in order (blocks of 32 columns)."""
    x = x.clone()
    out = out.clone()
    for start in range(0, x.shape[0], 8):
        rows = slice(start, start + 8)
        for begin in range(0, 2048, 32):
            if kernel is _store_then_read_moving_backwards:
                x[start, 32 + 2164 - begin] = -1.0
            value = x[start, 2100 - begin].clone()
            out[rows, begin : begin + 32] = out[rows, begin : begin + 32] * 0.0 + value
            if kernel is not _store_then_read_moving_backwards:
                x[start, 32 + 2164 - begin] = -1.0
    return out


@pytest.mark.parametrize("kernel", _BACKWARDS_KERNELS)
@pytest.mark.parametrize("shape", [(8, 4096), (16, 4096)])
def test_accesses_moving_backwards_match_reference(
    kernel: object, shape: tuple[int, int]
) -> None:
    # Iteration j + 3's uniform store lands on the column iteration j read
    # uniformly: the barrier closing the body keeps every thread's read of
    # iteration j ahead of any thread's store of iteration j + 3 (without it
    # a warp three iterations ahead overwrote the column in every run).
    run = _compiled(
        kernel,
        (_small_integers(shape), _small_integers(shape)),
        **_DEVICE_LOOP_ROWS_PER_THREAD_CONFIG,
    )
    for _ in range(_RUNS):
        x = _small_integers(shape)
        out = torch.full(shape, -7.0, device=CUDA_DEVICE)
        expected = _backwards_reference(kernel, x, out)
        torch.testing.assert_close(run(x, out), expected, rtol=0, atol=0)  # pyrefly: ignore [not-callable]


@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_uniform_read_modify_write_in_a_device_loop_matches_reference(
    shape: tuple[int, int],
) -> None:
    # One row per thread: the barrier between the uniform read and the
    # uniform store of the element orders the threads.
    run = _compiled(
        _uniform_read_modify_write_in_a_device_loop,
        (_small_integers(shape), _small_integers(shape)),
        **_DEVICE_LOOP_ROWS_PER_THREAD_CONFIG,
    )
    for _ in range(_RUNS):
        x = _small_integers(shape)
        out = torch.full(shape, -7.0, device=CUDA_DEVICE)
        expected_x = x.clone()
        expected = out.clone()
        for start in range(0, shape[0], 8):
            rows = slice(start, min(start + 8, shape[0]))
            for begin in range(0, shape[1], 32):
                value = expected_x[start, begin].clone()
                expected[rows, begin : begin + 32] = (
                    expected[rows, begin : begin + 32] * 0.0 + value
                )
                expected_x[start, begin] = value + 1.0
        torch.testing.assert_close(run(x, out), expected, rtol=0, atol=0)  # pyrefly: ignore [not-callable]
        torch.testing.assert_close(x, expected_x, rtol=0, atol=0)


def test_uniform_read_modify_write_in_a_repeated_device_loop_rejects_the_config() -> (
    None
):
    # Rows over two threads and a lane loop of two: the grid's lane loop runs
    # the device loop whole once per lane, and the second pass would read the
    # elements the first pass incremented (accepted, it doubled every
    # increment in every run).
    shape = (8, 256)
    with pytest.raises(exc.BackendUnsupported, match="repeated whole once per"):
        _run(
            _uniform_read_modify_write_in_a_device_loop,
            (_small_integers(shape), _small_integers(shape)),
            **_DEVICE_LOOP_COLUMNS_CONFIG,
        )


@pytest.mark.parametrize(
    ("kernel", "config"),
    [
        pytest.param(
            _gather_rows_then_zero, _ROW_LANES_ON_ONE_THREAD_CONFIG, id="rows"
        ),
        pytest.param(
            _gather_then_zero, _COLUMN_LANES_ON_ONE_THREAD_CONFIG, id="columns"
        ),
    ],
)
def test_gathered_store_conflict_on_a_single_thread_axis_rejects_the_config(
    kernel: object, config: dict[str, object]
) -> None:
    # The lanes of the gathered axis run on one thread: no barrier orders
    # them as the program does, so the config is rejected (accepted, it ran
    # every lane past the middle on the zeroed elements).
    shape = (8, 256)
    x = _small_integers(shape)
    if kernel is _gather_rows_then_zero:
        i = torch.arange(shape[0], device=CUDA_DEVICE)
        index = (i // 2) * 2 + (1 - i % 2)
    else:
        index = torch.arange(shape[1] - 1, -1, -1, device=CUDA_DEVICE)
    y = torch.full(shape, -7.0, device=CUDA_DEVICE)
    with pytest.raises(exc.BackendUnsupported, match="across the lanes"):
        _run(kernel, (x, index, y), **config)


def _mapped_reference(
    kernel: object, x: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``y`` and ``x`` after the program: the mapped read of ``x``, then ``x = v``."""
    if kernel is _read_next_then_store:
        y = torch.cat([x[1:], torch.zeros_like(x[:1])])
    elif kernel is _read_reversed_then_store:
        y = x.flip(0)
    elif kernel is _read_wrapped_then_store:
        y = x.roll(-1, 0)
    else:
        y = torch.cat([x[:, 1:], torch.zeros_like(x[:, :1])], dim=1)
    return y, v.clone()


_MAPPED_KERNELS_1D = [
    pytest.param(_read_next_then_store, id="next"),
    pytest.param(_read_reversed_then_store, id="reversed"),
    pytest.param(_read_wrapped_then_store, id="wrapped"),
]


@pytest.mark.parametrize("kernel", _MAPPED_KERNELS_1D)
def test_mapped_read_beside_the_store_matches_reference(kernel: object) -> None:
    # One element per thread: the neighbour's, the reversed or the wrapped
    # element is another thread's, and the barrier between the read and the
    # store keeps the read ahead of that thread's store.  Several runs: the
    # outcome without it depends on the warps' timing.
    shape = (256,)
    run = _compiled(
        kernel,
        tuple(_small_integers(shape) for _ in range(3)),
        **_ONE_ELEMENT_PER_THREAD_CONFIG,
    )
    for _ in range(_RUNS):
        x = _small_integers(shape)
        v = _small_integers(shape)
        y = torch.full(shape, -7.0, device=CUDA_DEVICE)
        expected_y, expected_x = _mapped_reference(kernel, x, v)
        run(x, v, y)  # pyrefly: ignore [not-callable]
        torch.testing.assert_close(y, expected_y, rtol=0, atol=0)
        torch.testing.assert_close(x, expected_x, rtol=0, atol=0)


@pytest.mark.parametrize("kernel", _MAPPED_KERNELS_1D)
def test_mapped_read_in_a_lane_loop_rejects_the_config(kernel: object) -> None:
    # Eight lanes per thread in one rolled loop: lane 7 reads the next
    # thread's first element after that thread's lane 0 stored it (the
    # reversed and wrapped reads meet other lanes likewise), which no
    # barrier orders; the config is rejected (accepted, it was wrong in
    # every run).
    shape = (256,)
    with pytest.raises(exc.BackendUnsupported, match="across the lanes"):
        _run(
            kernel,
            tuple(_small_integers(shape) for _ in range(3)),
            **_EIGHT_LANES_PER_THREAD_CONFIG,
        )


@pytest.mark.parametrize("shape", [(8, 64), (6, 60), (8, 128)])
def test_next_column_read_beside_the_store_matches_reference(
    shape: tuple[int, int],
) -> None:
    # The stencil neighbour along the column threads, rows over two threads
    # and two lanes: the barrier between the read and the store orders the
    # neighbour thread's store after the read.
    run = _compiled(
        _read_next_column_then_store,
        tuple(_small_integers(shape) for _ in range(3)),
        **_ONE_COLUMN_PER_THREAD_CONFIG,
    )
    for _ in range(_RUNS):
        x = _small_integers(shape)
        v = _small_integers(shape)
        y = torch.full(shape, -7.0, device=CUDA_DEVICE)
        expected_y, expected_x = _mapped_reference(_read_next_column_then_store, x, v)
        run(x, v, y)  # pyrefly: ignore [not-callable]
        torch.testing.assert_close(y, expected_y, rtol=0, atol=0)
        torch.testing.assert_close(x, expected_x, rtol=0, atol=0)
    with pytest.raises(exc.BackendUnsupported, match="across the lanes"):
        _run(
            _read_next_column_then_store,
            tuple(_small_integers(shape) for _ in range(3)),
            **_TWO_COLUMNS_PER_THREAD_CONFIG,
        )


@pytest.mark.parametrize("shape", [(8, 2304), (32, 2304)])
def test_short_sibling_loop_over_one_block_size_matches_reference(
    shape: tuple[int, int],
) -> None:
    # Iteration j + 2's uniform store lands on the column iteration j read
    # uniformly: the barrier closing the body keeps every thread's read of
    # iteration j ahead of any thread's store of iteration j + 2 (without it
    # a warp two iterations ahead overwrote the column in every run).
    n = shape[1] - 256
    run = _compiled(
        _short_loop_then_read_ahead_and_store,
        tuple(_small_integers(shape) for _ in range(3)),
        **_SHORT_SIBLING_LOOP_CONFIG,
    )
    for _ in range(_RUNS):
        x = _small_integers(shape)
        z = torch.ones(shape, device=CUDA_DEVICE)
        out = torch.full(shape, -7.0, device=CUDA_DEVICE)
        expected_x = x.clone()
        expected = out.clone()
        for start in range(0, shape[0], 8):
            rows = slice(start, start + 8)
            for begin in range(0, n, 64):
                value = expected_x[start, begin + 128].clone()
                expected[rows, begin : begin + 64] = (
                    expected[rows, begin : begin + 64] * 0.0 + value
                )
                expected_x[start, begin] = -1.0
        torch.testing.assert_close(run(x, z, out), expected, rtol=0, atol=0)  # pyrefly: ignore [not-callable]
        torch.testing.assert_close(x, expected_x, rtol=0, atol=0)
        assert not z[:, :128].any()


@pytest.mark.parametrize("shape", [(256,), (1024,)])
def test_packet_store_beside_the_previous_elements_read_matches_reference(
    shape: tuple[int],
) -> None:
    # One packet of four per thread: vector lane 0 reads the previous
    # thread's element 3, which that thread's flush stores; the barrier
    # between the read and the flush orders the threads.
    run = _compiled(
        _read_previous_then_store,
        tuple(_small_integers(shape) for _ in range(3)),
        **_ONE_PACKET_PER_THREAD_CONFIG,
    )
    for _ in range(_RUNS):
        x = _small_integers(shape)
        v = _small_integers(shape)
        y = torch.full(shape, -7.0, device=CUDA_DEVICE)
        expected_y = torch.cat([torch.zeros_like(x[:1]), x[:-1]])
        run(x, v, y)  # pyrefly: ignore [not-callable]
        torch.testing.assert_close(y, expected_y, rtol=0, atol=0)
        torch.testing.assert_close(x, v, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("kernel", "shape"),
    [
        pytest.param(_read_previous_then_store, (256,), id="previous"),
        pytest.param(_read_fourth_next_then_store, (250,), id="fourth_masked"),
    ],
)
def test_mapped_read_across_the_packets_of_a_thread_rejects_the_config(
    kernel: object, shape: tuple[int]
) -> None:
    # Two packets per thread, rolled: lane 1 reads lane 0's element 3 of
    # its own thread (stored already), or thread t + 1's lane 0 element in
    # its own vector lane; no barrier orders the lanes (accepted, both were
    # wrong in every run).
    with pytest.raises(exc.BackendUnsupported, match="across the lanes"):
        _run(
            kernel,
            tuple(_small_integers(shape) for _ in range(3)),
            **_TWO_PACKETS_PER_THREAD_CONFIG,
        )


def _repeated_device_loop_reference(
    kernel: object, args: tuple[torch.Tensor, ...], block: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(the result, x)`` after the program: rows in tiles of ``block``, columns in tiles of 32."""
    if kernel is _read_previous_row_then_store_in_a_device_loop:
        x, v, y = (a.clone() for a in args)
        m, n = x.shape
        x0 = x.clone()
        for start in range(0, m, block):
            end = min(start + block, m)
            for begin in range(0, n, 32):
                columns = slice(begin, min(begin + 32, n))
                y[start, columns] = 0.0
                y[start + 1 : end, columns] = x0[start : end - 1, columns]
                x[start:end, columns] = v[start:end, columns]
        return y, x
    x, out = (a.clone() for a in args)
    m, n = x.shape
    for start in range(0, m, block):
        end = min(start + block, m)
        for begin in range(0, n, 32):
            columns = slice(begin, min(begin + 32, n))
            value = x[start, begin].clone()
            out[start:end, columns] = value
            x[start:end, columns] += 1.0
    return out, x


@pytest.mark.parametrize(("kernel", "args"), _REPEATED_DEVICE_LOOP_KERNELS)
@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_per_lane_accesses_mapped_differently_in_a_device_loop_match_reference(
    kernel: object, args: tuple[torch.Tensor, ...], shape: tuple[int, int]
) -> None:
    # One row per thread: the other row's element is another thread's, and
    # the barrier between the read and the store orders the threads.
    run = _compiled(
        kernel,
        tuple(_small_integers(shape) for _ in args),
        **_DEVICE_LOOP_ROWS_PER_THREAD_CONFIG,
    )
    for _ in range(_RUNS):
        inputs = tuple(_small_integers(shape) for _ in args)
        expected, expected_x = _repeated_device_loop_reference(kernel, inputs, 8)
        result = run(*inputs)  # pyrefly: ignore [not-callable]
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(inputs[0], expected_x, rtol=0, atol=0)


@pytest.mark.parametrize(("kernel", "args"), _REPEATED_DEVICE_LOOP_KERNELS)
def test_per_lane_accesses_mapped_differently_in_a_repeated_device_loop_reject_the_config(
    kernel: object, args: tuple[torch.Tensor, ...]
) -> None:
    # Rows over two threads and a lane loop of two: lane 1's pass reads the
    # rows lane 0's pass stored, in every iteration (accepted, both kernels
    # were wrong in every run).
    shape = (8, 256)
    with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
        _run(
            kernel,
            tuple(_small_integers(shape) for _ in args),
            **_DEVICE_LOOP_COLUMNS_CONFIG,
        )


def test_previous_row_read_at_the_next_tiles_column_rejects_the_config() -> None:
    # Rows over two threads and a lane loop of two: within one iteration the
    # read (the previous row at the next tile's first column) and the store
    # are apart, but lane 1's pass reads in iteration j what lane 0's pass
    # stored in iteration j + 1 (accepted, wrong in every run).
    with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
        _run(
            _read_previous_row_at_the_next_tiles_column_then_store,
            tuple(_small_integers((8, 320)) for _ in range(3)),
            **_DEVICE_LOOP_COLUMNS_CONFIG,
        )


@pytest.mark.parametrize("shape", [(8, 320), (6, 314)], ids=["full", "partial"])
def test_previous_row_read_at_the_next_tiles_column_matches_reference(
    shape: tuple[int, int],
) -> None:
    # One row per thread: the previous row is another thread's, and the
    # barrier closing the body keeps iteration j's read ahead of iteration
    # j + 1's stores.
    n = shape[1] - 64
    run = _compiled(
        _read_previous_row_at_the_next_tiles_column_then_store,
        tuple(_small_integers(shape) for _ in range(3)),
        **_DEVICE_LOOP_ROWS_PER_THREAD_CONFIG,
    )
    for _ in range(_RUNS):
        x, v, y = (_small_integers(shape) for _ in range(3))
        expected_x = x.clone()
        expected_y = y.clone()
        for start in range(0, shape[0], 8):
            end = min(start + 8, shape[0])
            for begin in range(0, n, 32):
                columns = slice(begin, min(begin + 32, n))
                value = torch.zeros(end - start, device=CUDA_DEVICE)
                value[1:] = expected_x[start : end - 1, begin + 32]
                if start > 0:
                    value[0] = expected_x[start - 1, begin + 32]
                expected_y[start:end, columns] = value[:, None]
                expected_x[start:end, columns] = v[start:end, columns]
        torch.testing.assert_close(run(x, v, y), expected_y, rtol=0, atol=0)  # pyrefly: ignore [not-callable]
        torch.testing.assert_close(x, expected_x, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_read_modify_write_in_a_vectorized_device_loop_matches_reference(
    shape: tuple[int, int],
) -> None:
    # Rows on two lanes, the columns one packet of four per thread: the
    # packet load and the flush inside the repeated device loop are the
    # lane's own row, accepted without a barrier.
    run = _compiled(
        _read_modify_write_in_a_device_loop,
        tuple(_small_integers(shape) for _ in range(3)),
        **_VECTORIZED_DEVICE_LOOP_COLUMNS_CONFIG,
    )
    for _ in range(_RUNS):
        x, v, y = (_small_integers(shape) for _ in range(3))
        expected_y = x + 1.0
        torch.testing.assert_close(run(x, v, y), expected_y, rtol=0, atol=0)  # pyrefly: ignore [not-callable]
        torch.testing.assert_close(x, v, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(8, 2048), (32, 2048)])
def test_update_under_a_loop_indexed_by_its_begin_matches_reference(
    shape: tuple[int, int],
) -> None:
    # The later loop over the short block runs two iterations; its nested
    # update adds two to every element (with the block's dead lane loop of
    # 64 around it, accepted before, it added 128).
    run = _compiled(
        _update_under_a_loop_indexed_by_its_begin,
        (_small_integers(shape), _small_integers((shape[0], 64))),
        **_SHORT_BLOCK_ON_ONE_THREAD_CONFIG,
    )
    for _ in range(_RUNS):
        x = _small_integers(shape)
        z = _small_integers((shape[0], 64))
        expected = x + 2.0
        torch.testing.assert_close(run(x, z), expected, rtol=0, atol=0)  # pyrefly: ignore [not-callable]
        assert not z.any()


def _shifted_rows_reference(
    v: torch.Tensor, x: torch.Tensor, block: int
) -> torch.Tensor:
    expected = x.clone()
    m, n = v.shape
    for start in range(0, m, block):
        end = min(start + block, m)
        for j in range(n):
            expected[start + j : end + j] = v[start:end, j]
    return expected


def test_folding_store_in_a_repeated_device_loop_rejects_the_config() -> None:
    # Rows over two threads and a lane loop of two: lane 1's pass stores
    # iteration j's value over lane 0's iteration j + 1's (accepted, wrong
    # in every run).
    with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
        _run(
            _store_rows_shifted_by_the_iteration,
            (_small_integers((8, 256)), _small_integers((264,))),
            **_ROWS_ON_TWO_LANES_CONFIG,
        )


@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_folding_store_per_thread_matches_reference(shape: tuple[int, int]) -> None:
    # One row per thread: thread r + 1's iteration j and thread r's
    # iteration j + 1 store one element, ordered by the barrier closing the
    # body.
    m, n = shape
    run = _compiled(
        _store_rows_shifted_by_the_iteration,
        (_small_integers(shape), _small_integers((m + n,))),
        **_ROWS_ONE_PER_THREAD_CONFIG,
    )
    for _ in range(_RUNS):
        v = _small_integers(shape)
        x = torch.full((m + n,), -7.0, device=CUDA_DEVICE)
        expected = _shifted_rows_reference(v, x, 8)
        torch.testing.assert_close(run(v, x), expected, rtol=0, atol=0)  # pyrefly: ignore [not-callable]


@pytest.mark.parametrize(
    "config",
    [_DEVICE_LOOP_COLUMNS_CONFIG, _DEVICE_LOOP_ROWS_PER_THREAD_CONFIG],
    ids=["lanes", "threads"],
)
@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_flattened_store_with_a_masked_tail_matches_reference(
    shape: tuple[int, int], config: dict[str, object]
) -> None:
    # The column tile's mask bounds the flattened address below the row
    # stride: accepted without a barrier on the lanes and across the
    # threads' iterations, and right (the padded columns of a partial tile
    # are never stored).
    m, n = shape
    run = _compiled(
        _store_flattened_rows,
        (_small_integers(shape), _small_integers((m * n,))),
        **config,
    )
    for _ in range(_RUNS):
        v = _small_integers(shape)
        out = torch.full((m * n,), -7.0, device=CUDA_DEVICE)
        torch.testing.assert_close(run(v, out), v.reshape(-1), rtol=0, atol=0)  # pyrefly: ignore [not-callable]


def _overlapping_rows_reference(
    v: torch.Tensor, out: torch.Tensor, rows: int, columns: int
) -> torch.Tensor:
    expected = out.clone()
    m, n = v.shape
    stride = n - 8
    for start in range(0, m, rows):
        end = min(start + rows, m)
        for first in range(0, n, columns):
            last = min(first + columns, n)
            for r in range(start, end):
                expected[stride * r + first : stride * r + last] = v[r, first:last]
    return expected


def test_overlapping_flattened_store_rejects_the_config() -> None:
    # Rows over two threads and a lane loop of two: lane 1's pass stores
    # its row's first eight columns over lane 0's row's last eight, which
    # the program ordered the other way (accepted, wrong in every run).
    with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
        _run(
            _store_flattened_rows_overlapping,
            (_small_integers((6, 250)), _small_integers((6 * 250,))),
            **_DEVICE_LOOP_COLUMNS_CONFIG,
        )


@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_overlapping_flattened_store_per_thread_matches_reference(
    shape: tuple[int, int],
) -> None:
    # One row per thread: thread r + 1's iteration 0 and thread r's last
    # iteration store one element, ordered by the barrier closing the body.
    m, n = shape
    run = _compiled(
        _store_flattened_rows_overlapping,
        (_small_integers(shape), _small_integers((m * n,))),
        **_DEVICE_LOOP_ROWS_PER_THREAD_CONFIG,
    )
    for _ in range(_RUNS):
        v = _small_integers(shape)
        out = torch.full((m * n,), -7.0, device=CUDA_DEVICE)
        expected = _overlapping_rows_reference(v, out, 8, 32)
        torch.testing.assert_close(run(v, out), expected, rtol=0, atol=0)  # pyrefly: ignore [not-callable]


def test_dynamic_overlapping_flattened_store_rejects_the_config() -> None:
    # The stride is a size argument: paired through it as a uniform value,
    # the fold is found (accepted and wrong in every run when the host's
    # unpacking of the shape left the size an unknown).
    with pytest.raises(exc.BackendUnsupported, match="lanes' passes interleave"):
        _run(
            _store_flattened_rows_overlapping_dynamic,
            (_small_integers((6, 250)), _small_integers((6 * 250,))),
            **_DEVICE_LOOP_COLUMNS_CONFIG,
        )


@pytest.mark.parametrize("shape", [(8, 256), (6, 250)], ids=["full", "partial"])
def test_dynamic_overlapping_flattened_store_per_thread_matches_reference(
    shape: tuple[int, int],
) -> None:
    m, n = shape
    run = _compiled(
        _store_flattened_rows_overlapping_dynamic,
        (_small_integers(shape), _small_integers((m * n,))),
        **_DEVICE_LOOP_ROWS_PER_THREAD_CONFIG,
    )
    for _ in range(_RUNS):
        v = _small_integers(shape)
        out = torch.full((m * n,), -7.0, device=CUDA_DEVICE)
        expected = _overlapping_rows_reference(v, out, 8, 32)
        torch.testing.assert_close(run(v, out), expected, rtol=0, atol=0)  # pyrefly: ignore [not-callable]


def test_segment_rows_through_a_uniform_start_match_reference() -> None:
    # Three segments of eight rows, each updated in place by its block with
    # the rows on two threads and two lanes: the start every lane shares
    # makes the rows one lane's, so the config is accepted and right.
    offsets = torch.tensor([0, 8, 16], dtype=torch.int32, device=CUDA_DEVICE)
    run = _compiled(
        _increment_segment_rows,
        (offsets, _small_integers((8, 128)), _small_integers((24 * 128,))),
        **_DEVICE_LOOP_COLUMNS_CONFIG,
    )
    for _ in range(_RUNS):
        v = _small_integers((8, 128))
        x = _small_integers((24 * 128,))
        expected = (x.reshape(24, 128) + v.repeat(3, 1)).reshape(-1)
        torch.testing.assert_close(run(offsets, v, x.clone()), expected, rtol=0, atol=0)  # pyrefly: ignore [not-callable]
