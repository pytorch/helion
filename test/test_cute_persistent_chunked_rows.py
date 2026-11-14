"""Persistent rows of several V-wide packets per thread (chunked lanes).

A persistent reduction whose threads each hold more elements than one packet
used to drop the configured vector width and walk the row in scalar steps
``threads`` elements apart.  Each thread now holds ``E / V`` packets
``threads * V`` elements apart (Triton's persistent layout: consecutive threads
read consecutive packets): one constexpr lane loop walks all of them, the
branch-local persistent vectorizer places one packet per chunk and flushes one
constant slice of the collected values per chunk, and a scalar access of
another tensor on the row (rows that are not packet aligned) spans
``threads * V`` elements per warp instruction instead of the whole row.  A
row stored in place keeps its chunks (each lane's element is its own), the
exact division's redo reads the chunks' packets, and a scanned row keeps the
cyclic lanes its parallel scan walks.  The kernels render on CPU; the
``cuda`` tests run the chunked rows against eager.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._compiler.cute.autotune.heuristics.cute import CuteReductionTileHeuristic
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_codegen import with_exact_division
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]

REDO = "_cute_exact_div_redo_f32("


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _shifted_softmax(content: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    # XLNet's relative-shift softmax: ``position`` rows start one element
    # further along for every row, so only ``content`` rows are packet aligned.
    rows, width = content.shape
    out = torch.empty_like(content)
    for row in hl.tile(rows):
        x = content[row, :].to(torch.float32) + position[row, :].to(torch.float32)
        e = torch.exp(x - torch.amax(x, dim=-1, keepdim=True))
        out[row, :] = (e / e.sum(-1, keepdim=True)).to(content.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _shifted_rows(
    content: torch.Tensor, position: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Only order-independent reductions: every layout gives the same bits.
    rows, width = content.shape
    out = torch.empty_like(content)
    finite = torch.empty(content.shape, dtype=torch.bool, device=content.device)
    for row in hl.tile(rows):
        x = content[row, :].to(torch.float32) + position[row, :].to(torch.float32)
        shifted = x - torch.amax(x, dim=-1, keepdim=True)
        out[row, :] = shifted.to(content.dtype)
        finite[row, :] = shifted > -4.0
    return out, finite


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_softmax(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        data = x[row, :].to(torch.float32)
        e = torch.exp(data - torch.amax(data, dim=-1, keepdim=True))
        out[row, :] = (e / e.sum(-1, keepdim=True)).to(x.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _softmax_in_place(x: torch.Tensor) -> torch.Tensor:
    for row in hl.tile(x.size(0)):
        data = x[row, :].to(torch.float32)
        e = torch.exp(data - torch.amax(data, dim=-1, keepdim=True))
        x[row, :] = (e / e.sum(-1, keepdim=True)).to(x.dtype)
    return x


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _rows_div(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    # The division sweep reads the row's packets.
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        out[row, :] = x[row, :] / s[row][:, None]
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_cumsum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        out[row, :] = torch.cumsum(x[row, :], dim=1)
    return out


def _argmax_combine(
    left_value: torch.Tensor,
    left_index: torch.Tensor,
    right_value: torch.Tensor,
    right_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    take_right = right_value > left_value
    return (
        torch.where(take_right, right_value, left_value),
        torch.where(take_right, right_index, left_index),
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_cumulative_argmax(
    x: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # A scan of a tuple: the graph gives the scan node no value of its own.
    top = torch.empty_like(x)
    at = torch.empty_like(x, dtype=torch.int32)
    for row in hl.tile(x.size(0)):
        values = x[row, :]
        indices = positions[:].to(torch.float32).unsqueeze(0).expand_as(values)
        top_values, top_indices = hl.associative_scan(
            _argmax_combine, (values, indices), dim=1
        )
        top[row, :] = top_values
        at[row, :] = top_indices.to(torch.int32)
    return top, at


def _shifted_args(
    rows: int, width: int, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    content = torch.randn((rows, width), dtype=torch.bfloat16, device=device)
    storage = torch.randn((rows, 2 * width), dtype=torch.bfloat16, device=device)
    position = storage.as_strided((rows, width), (2 * width - 1, 1), width)
    return content, position


def _config(
    bound: BoundKernel, vec_width: int, *, row_block: int = 1, threads: int = 32
) -> helion.Config:
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    spec = bound.config_spec
    return helion.Config(
        block_sizes=[row_block],
        num_threads=[
            threads if block == reduction else row_block
            for block in spec.num_threads.valid_block_ids()
        ],
        reduction_loops=[None] * len(spec.reduction_loops),
        cute_vector_widths=[
            vec_width if block == reduction else 1
            for block in spec.cute_vector_widths.valid_block_ids()
        ],
        cute_reduction_reloads=[
            "register_prefetch" if block == reduction else "auto"
            for block in spec.cute_reduction_reloads.valid_block_ids()
        ],
    )


def _code(
    kernel: Kernel[object],
    args: tuple[object, ...],
    vec_width: int,
    *,
    row_block: int = 1,
    exact_division: bool = False,
) -> str:
    if exact_division:
        kernel = with_exact_division(kernel)
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(kernel, args)
        config = _config(bound, vec_width, row_block=row_block)
        assert bound.config_spec.normalized_config(config).reduction_loops == [None]
        return bound.to_code(config)


@pytest.mark.parametrize("vec_width", [2, 4, 8])
def test_each_chunk_takes_one_packet(vec_width: int) -> None:
    chunks = 16 // vec_width
    packet = f"ir.VectorType.get([{vec_width}], cutlass.Uint16.mlir_type)"
    # With ``cute_exact_division`` too: the store sweep collects its packets
    # in lists, which a redo of the loop would append to twice, so the exact
    # division keeps its plain form there and the layout is the same.
    for exact_division in (False, True):
        code = _code(
            _shifted_softmax,
            _shifted_args(4, 512),
            vec_width,
            exact_division=exact_division,
        )
        # 512 elements over 32 threads: one constexpr loop over each thread's 16.
        assert "for synthetic_lane_1 in cutlass.range_constexpr(16):" in code, code
        assert (
            f"synthetic_lane_1 // {vec_width} * {32 * vec_width}"
            f" + synthetic_lane_1 % {vec_width}"
        ) in code, code
        # ``content`` takes one packet per chunk, read again by the later sweeps.
        assert code.count(packet) == chunks, code
        # ``position`` rows are not packet aligned: its loads stay scalar.
        assert "position.iterator" in code and ".load()" in code, code
        assert REDO not in code, code
        # Each chunk is flushed once.
        for chunk in range(chunks):
            lanes = f"[{chunk * vec_width}:{(chunk + 1) * vec_width}])"
            assert code.count(lanes) == 1, code


def test_one_packet_per_thread_keeps_its_layout() -> None:
    code = _code(_shifted_softmax, _shifted_args(4, 512), 16)
    assert "for synthetic_lane_1 in cutlass.range_constexpr(16):" in code, code
    assert "reduction_lane_base_1 + cutlass.Int32(synthetic_lane_1)" in code, code
    assert "synthetic_lane_1 //" not in code, code
    assert code.count("ir.VectorType.get([16], cutlass.Uint16.mlir_type)") == 1, code


@pytest.mark.parametrize("vec_width", [2, 4, 8])
def test_an_in_place_row_keeps_its_chunks(vec_width: int) -> None:
    # Each lane loads and stores its own element of the row
    # (``_lane_offsets_separate``), so the lane split may reorder them.
    code = _code(
        _softmax_in_place, (torch.empty((4, 512), dtype=torch.bfloat16),), vec_width
    )
    assert f"synthetic_lane_1 // {vec_width} * {32 * vec_width}" in code, code
    packet = f"ir.VectorType.get([{vec_width}], cutlass.Uint16.mlir_type)"
    assert code.count(packet) == 16 // vec_width, code


@pytest.mark.parametrize("vec_width", [2, 4, 8])
def test_exact_division_reads_each_chunk_packet_once(vec_width: int) -> None:
    code = _code(
        _rows_div,
        (torch.empty((4, 512)), torch.empty(4)),
        vec_width,
        exact_division=True,
    )
    read = f"[synthetic_lane_1 // {vec_width}][synthetic_lane_1 % {vec_width}]"
    # The packet-store sweep keeps the plain IEEE division: no redo copy.
    assert code.count(read) == 1, code
    assert REDO not in code, code


def test_a_row_whose_packets_do_not_fit_keeps_cyclic_lanes() -> None:
    # 16 fp32 elements are 64 bytes, past the widest packet (32 bytes on
    # SM100): chunks would walk the row in scalar lanes 16 apart, a 2 KB
    # span per warp instruction, where the cyclic lanes read 128 bytes.
    code = _code(_row_softmax, (torch.empty((4, 1024)),), 16)
    assert "synthetic_lane_1 //" not in code, code
    assert "cutlass.Int32(synthetic_lane_1) * 32" in code, code
    assert "ir.VectorType.get(" not in code, code


@pytest.mark.parametrize(("dtype", "kept"), [(torch.float32, 2), (torch.bfloat16, 1)])
def test_normalization_keeps_the_unroll_of_cyclic_lanes(
    dtype: torch.dtype, kept: int
) -> None:
    # 32 lanes per thread at V=16: the fp32 row keeps its cyclic lanes, which
    # the unroll takes; the bf16 row's 32-byte packets fit, and its chunks'
    # constexpr loop takes none (``fits_row_packets``).
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(_row_softmax, (torch.empty((4, 1024), dtype=dtype),))
        config = _config(bound, 16)
        config.config["cute_lane_unroll"] = 2
        normalized = bound.config_spec.normalized_config(config)
        assert normalized.config["cute_lane_unroll"] == kept
        code = bound.to_code(normalized)
    assert ("_lane_unroll_loads_" in code) == (kept > 1), code


@pytest.mark.parametrize("vec_width", [4, 8])
def test_a_scanned_row_keeps_cyclic_lanes(vec_width: int) -> None:
    # The parallel scan walks cyclic lanes (``cute_lane_axis``); chunks would
    # leave the row to the serial rescan.  Normalization knows the scanned
    # row (``cute_scanned_block_ids``) and keeps the unroll its lanes take.
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(_row_cumsum, (torch.empty((4, 512)),))
        config = _config(bound, vec_width)
        config.config["cute_lane_unroll"] = 2
        normalized = bound.config_spec.normalized_config(config)
        assert normalized.config["cute_lane_unroll"] == 2
        code = bound.to_code(normalized)
    assert "cutlass.Int32(synthetic_lane_1) * 32" in code, code
    assert "synthetic_lane_1 //" not in code, code
    assert "for scan_i in range(" not in code, code
    assert "cute.arch.shuffle_sync_up(" in code, code
    assert "_lane_unroll_loads_" in code, code


def test_a_tuple_scan_records_its_scanned_row() -> None:
    # Normalization reads the scanned row from the scan's input, as the scan
    # of a tuple has no value of its own.
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(
            _row_cumulative_argmax,
            (torch.empty((4, 512)), torch.empty(512, dtype=torch.int32)),
        )
        code = bound.to_code(bound.config_spec.default_config())
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    assert bound.config_spec.cute_scanned_block_ids == {reduction}
    assert "cute.arch.shuffle_sync_up(" in code, code


@pytest.mark.parametrize(("threads", "vec_width"), [(1, 8), (4, 2)])
def test_a_row_past_the_chunk_lane_cap_keeps_cyclic_lanes(
    threads: int, vec_width: int
) -> None:
    # 1024 and 256 lanes per thread, past ``MAX_VECTOR_ROW_LANES``: chunks
    # would load every packet ahead of a constexpr loop that unrolls each
    # lane, so the row keeps its runtime loop of cyclic lanes, and
    # normalization keeps the unroll that loop takes.
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(_row_softmax, (torch.empty((4, 1024), dtype=torch.bfloat16),))
        config = _config(bound, vec_width, threads=threads)
        config.config["cute_lane_unroll"] = 2
        normalized = bound.config_spec.normalized_config(config)
        assert normalized.config["cute_lane_unroll"] == 2
        code = bound.to_code(normalized)
    lanes = 1024 // threads
    assert f"for synthetic_lane_1 in range({lanes}):" in code, code
    assert f"cutlass.range_constexpr({lanes})" not in code, code
    assert "synthetic_lane_1 //" not in code, code
    assert "ir.VectorType.get(" not in code, code
    assert "_lane_unroll_loads_" in code, code


def test_a_partial_row_tile_guards_each_chunk() -> None:
    # 37 rows in tiles of two: the last tile's second row is masked, as a
    # whole, for every chunk of its packets and flushes.
    rows = torch.empty((37, 512), dtype=torch.bfloat16)
    code = _code(_row_softmax, (rows,), 8, row_block=2)
    packet = "ir.VectorType.get([8], cutlass.Uint16.mlir_type)"
    assert code.count(packet) == 2, code
    assert code.count(" if mask_0 else cutlass.Int32(0))") == 2, code
    # One flush per chunk (the masked dividend keeps the plain division, so
    # there is no redo copy).
    assert code.count("if mask_0:\n") == 2, code
    assert code.count("_cute_store_u16_vec(") == 2, code


@pytest.mark.parametrize(
    ("dtype", "width", "vec_width"),
    [
        (torch.bfloat16, 512, 8),
        (torch.float32, 256, 4),
        # 32 threads would hold four packets each: the rolled row takes 64.
        (torch.bfloat16, 1024, None),
    ],
)
def test_one_warp_two_packet_row_is_seeded_persistent(
    dtype: torch.dtype, width: int, vec_width: int | None
) -> None:
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(_row_softmax, (torch.empty((64, width), dtype=dtype),))
        host = bound.host_function
        assert host is not None
        env, device_ir = bound.env, host.device_ir
        seeds = CuteReductionTileHeuristic.get_seed_configs(env, device_ir)
        assert seeds is not None
        rolled = CuteReductionTileHeuristic.two_packet_rolled_row_seed_config(
            env, device_ir
        )
        assert rolled is not None
        persistent = CuteReductionTileHeuristic.two_packet_persistent_row_seed_config(
            env, rolled
        )
        # The row's max takes the native warp reduction on the persistent row
        # alone (the kernel's structure makes the knob searchable).
        redux = [seed for seed in seeds if seed.config.get("cute_warp_fp32_redux")]
        if vec_width is None:
            assert persistent is None
            assert redux == []
            return
        assert persistent is not None
        # The rolled row's geometry, held in registers, right after it.
        assert persistent.config == rolled.config | {"reduction_loops": [None]}
        assert redux == [seeds[seeds.index(rolled) + 1]]
        assert redux[0].config == persistent.config | {"cute_warp_fp32_redux": True}
        code = bound.to_code(redux[0])
    assert f"synthetic_lane_1 // {vec_width} * {32 * vec_width}" in code, code
    assert "_cute_warp_reduce_max_redux(" in code, code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_floor(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        data = x[row, :].to(torch.float32)
        out[row, :] = (data - torch.amin(data, dim=-1, keepdim=True)).to(x.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_flags(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        data = x[row, :].to(torch.float32)
        hit = torch.any(data > 2.0, dim=-1, keepdim=True)
        out[row, :] = torch.where(hit, -data, data).to(x.dtype)
    return out


@functools.cache
def _cuda_shifted() -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    args = _shifted_args(64, 512, DEVICE)
    content, position = (arg.float() for arg in args)
    x = content + position
    expected = torch.softmax(x, dim=-1).to(torch.bfloat16)
    return args, expected


@pytest.mark.cuda
@skipIfNotCUDA()
@pytest.mark.parametrize("vec_width", [1, 4, 8, 16])
def test_cuda_chunked_rows_match_eager(vec_width: int) -> None:
    args, expected = _cuda_shifted()
    bound = _shifted_softmax._bind_isolated(args)
    out = bound.compile_config(_config(bound, vec_width))(*args)
    torch.testing.assert_close(out, expected, atol=2e-5, rtol=1e-2)


@pytest.mark.cuda
@skipIfNotCUDA()
@pytest.mark.parametrize("kernel", [_row_softmax, _row_floor, _row_flags])
def test_cuda_persistent_row_seed_with_native_redux_matches_eager(
    kernel: Kernel,
) -> None:
    """The one-warp persistent row seed with ``cute_warp_fp32_redux``: a max,
    a min and an ``any`` per row (``redux.sync`` on the SM100 family)."""
    x = torch.randn((64, 512), dtype=torch.bfloat16, device=DEVICE) * 2
    bound = kernel._bind_isolated((x,))
    host = bound.host_function
    assert host is not None
    rolled = CuteReductionTileHeuristic.two_packet_rolled_row_seed_config(
        bound.env, host.device_ir
    )
    assert rolled is not None
    persistent = CuteReductionTileHeuristic.two_packet_persistent_row_seed_config(
        bound.env, rolled
    )
    assert persistent is not None
    config = helion.Config(**persistent.config, cute_warp_fp32_redux=True)
    out = bound.compile_config(config)(x)
    data = x.float()
    if kernel is _row_softmax:
        expected = torch.softmax(data, dim=-1).to(torch.bfloat16)
        torch.testing.assert_close(out, expected, atol=2e-5, rtol=1e-2)
        return
    if kernel is _row_floor:
        expected = data - data.amin(dim=-1, keepdim=True)
    else:
        expected = torch.where((data > 2.0).any(dim=-1, keepdim=True), -data, data)
    assert torch.equal(out, expected.to(torch.bfloat16))


@pytest.mark.cuda
@skipIfNotCUDA()
def test_cuda_chunked_rows_keep_order_independent_bits() -> None:
    # bf16 and bool packets of every chunk width against the scalar lanes.
    args = _shifted_args(64, 512, DEVICE)
    bound = _shifted_rows._bind_isolated(args)
    reference = bound.compile_config(_config(bound, 1))(*args)
    for vec_width in (2, 4, 8):
        config = _config(bound, vec_width)
        code = bound.to_code(config)
        assert "_cute_store_u8_vec(" in code and "_cute_store_u16_vec(" in code
        out, finite = bound.compile_config(config)(*args)
        assert torch.equal(out.view(torch.int16), reference[0].view(torch.int16))
        assert torch.equal(finite, reference[1])


@pytest.mark.cuda
@skipIfNotCUDA()
def test_cuda_partial_row_tile_matches_eager() -> None:
    x = torch.randn((37, 512), dtype=torch.bfloat16, device=DEVICE)
    bound = _row_softmax._bind_isolated((x,))
    out = bound.compile_config(_config(bound, 8, row_block=2))(x)
    expected = torch.softmax(x.float(), dim=-1).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, atol=2e-5, rtol=1e-2)


@pytest.mark.cuda
@skipIfNotCUDA()
@pytest.mark.parametrize("vec_width", [2, 4, 8])
def test_cuda_in_place_chunked_rows_match_eager(vec_width: int) -> None:
    x = torch.randn((64, 512), dtype=torch.bfloat16, device=DEVICE)
    expected = torch.softmax(x.float(), dim=-1).to(torch.bfloat16)
    bound = _softmax_in_place._bind_isolated((x,))
    bound.compile_config(_config(bound, vec_width))(x)
    torch.testing.assert_close(x, expected, atol=2e-5, rtol=1e-2)


@pytest.mark.cuda
@skipIfNotCUDA()
def test_cuda_chunked_rows_divide_exactly() -> None:
    # Rows whose dividends leave the admitted window take the redo.
    g = torch.Generator().manual_seed(0)
    x = torch.randn((64, 512), generator=g) * 3
    s = torch.rand(64, generator=g) + 0.5
    x[0, 0] = float("inf")
    x[1, :] = 1e-40
    x[2, 100] = float("nan")
    s[3] = 0.0
    s[4] = 2.0**120
    x[4, ::7] = 1e-3
    x[5, :] = 1e38
    s[5] = 0.3
    x, s = x.to(DEVICE), s.to(DEVICE)
    expected = (x / s[:, None]).view(torch.int32)
    bound = with_exact_division(_rows_div)._bind_isolated((x, s))
    for vec_width in (4, 8):
        out = bound.compile_config(_config(bound, vec_width))(x, s)
        assert torch.equal(out.view(torch.int32), expected), vec_width
