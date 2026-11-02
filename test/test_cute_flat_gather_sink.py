"""Flattened gathers vectorized along a tile axis by the V-loop sink.

``hl.load(x_flat, [(starts[:, None] + tile_k.index[None, :])[:, :, None] * M
+ tile_m.index[None, None, :]])`` gathers rows of a flattened 2-D tensor; the
index is ``row + tile_m.index`` so consecutive ``tile_m`` lanes are
consecutive elements.  With a vector width on ``tile_m`` and
``cute_vloop_sink`` the load lowering emits the pointer with the lane index
as its own term (``cute/flat_gather_lane.py``), the sink loads every row as
one predicated 16-byte packet (``cute/masked_packet_load.py``) and packs the
per-lane stores of the same shape into one 16-byte store.

CPU code generation only; the nested-row seed tests run the kernels.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.jagged_mean import jagged_mean_kernel
from examples.jagged_softmax import jagged_softmax_kernel
from examples.jagged_sum import jagged_sum_kernel
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = skipUnlessBackends(["cute"])

cutlass = pytest.importorskip("cutlass")


@pytest.fixture
def cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU-only test")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


def _code(kernel: helion.Kernel, args: tuple[object, ...], config: dict) -> str:
    bound = _cpu_bind(kernel, args)
    normalized = bound.config_spec.normalized_config(helion.Config.from_dict(config))
    with patch(
        "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
        return_value=232448,
    ):
        return bound.to_code(normalized)


def _jagged_args(columns: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty((731, columns)), torch.empty(18, dtype=torch.int64)


def _loops(code: str, target: str) -> list[ast.For]:
    return [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == target
    ]


SUM_CONFIG = {
    "block_sizes": [1, 128, 128],
    "num_threads": [1, 32, 8],
    "cute_vector_widths": [1, 4, 1],
    "cute_lane_layouts": ["strided", "blocked", "strided"],
    "cute_vloop_sink": True,
    "cute_lane_unroll": 8,
    "cute_min_blocks_per_mp": 1,
}

SOFTMAX_CONFIG = {
    "block_sizes": [1, 128, 128, 128],
    "num_threads": [1, 32, 8, 8],
    "cute_vector_widths": [1, 4, 1, 1],
    "cute_lane_layouts": ["strided", "blocked", "strided", "strided"],
    "cute_vloop_sink": True,
    "cute_lane_unroll": 8,
    "cute_min_blocks_per_mp": 1,
}


def test_jagged_sum_gathers_rows_as_predicated_packets(cpu_only: None) -> None:
    kernel = helion.kernel(jagged_sum_kernel.fn, backend="cute")
    code = _code(kernel, _jagged_args(), SUM_CONFIG)
    # The gather's pointer carries the feature index as its own term, so the
    # sink finds the vectorized axis and loads V features per row at once:
    # 16 rows per thread (128 / 8 threads), unrolled by 8, loaded before any
    # is consumed.
    assert "x_flat.layout.stride[0])).load()" not in code
    assert code.count("_cute_load_v16b_if(x_flat.iterator") == 8
    (lane_loop,) = _loops(code, "lane_2")
    assert ast.unparse(lane_loop.iter) == "range(2)"
    first_consume = next(
        index for index, stmt in enumerate(lane_loop.body) if isinstance(stmt, ast.For)
    )
    loads = [
        index
        for index, stmt in enumerate(lane_loop.body)
        if "_cute_load_v16b_if" in ast.unparse(stmt)
    ]
    assert len(loads) == 8 and max(loads) < first_consume
    # The masked rows are zero-filled by the predicated load itself; no
    # per-lane select repeats the fill.
    assert "if mask_2_u0 else cutlass.Float32(0)" not in code
    # Per-lane state lives in V-element fragments and the per-row-tile
    # combine across the jagged threads runs once for all V lanes.
    assert "row_sums_frag = cute.make_rmem_tensor(4, cutlass.Float32)" in code
    assert code.count("_cute_grouped_reduce_shared_two_stage_fragment(") == 1
    assert "block=(32, 8, 1)" in code


def test_jagged_softmax_vectorizes_both_passes_and_packs_the_stores(
    cpu_only: None,
) -> None:
    kernel = helion.kernel(jagged_softmax_kernel.fn, backend="cute", fast_math=True)
    code = _code(kernel, _jagged_args(), SOFTMAX_CONFIG)
    assert "x_flat.layout.stride[0])).load()" not in code
    # Pass one's sum sweep re-reads the rows its max sweep loaded: the input
    # is never written, so the sweep reads the first sweep's packets from a
    # register fragment (16 rows x 4 lanes per thread) instead; pass two
    # loads the rows again.
    assert code.count("_cute_load_v16b_if(x_flat.iterator") == 16
    assert "_vsink_cache = cute.make_rmem_tensor(64, cutlass.Uint32)" in code
    assert code.count("_vsink_cache[(lane_2 * 8 + 7) * 4 + vec_lane_1]") == 1
    # The online-softmax state is a fragment per lane and the two combines
    # of a row tile run once for all V lanes each.
    assert "block_max_frag = cute.make_rmem_tensor(4, cutlass.Float32)" in code
    assert "block_L_frag = cute.make_rmem_tensor(4, cutlass.Float32)" in code
    assert code.count("_cute_grouped_reduce_shared_two_stage_fragment(") == 2
    # The max's -inf fill is not a zero fill: its select stays.
    assert "else cutlass.Float32(float('-inf'))" in code
    # Pass two's per-lane stores into the fresh output (proven disjoint from
    # the gathered input) flush as one 16-byte packet per row under the
    # row's own mask, and its loads move above those stores.
    assert code.count("_cute_store_u32_vec(out.iterator") == 8
    assert ".store(cutlass.Float32(v_" not in code
    assert "_vsink_store_vals_u0.append(" in code
    assert "block=(32, 8, 1)" in code


def test_a_jagged_feature_tile_loads_straddling_chunks_at_the_chunk_mask(
    cpu_only: None,
) -> None:
    """``hl.jagged_tile(feature_counts)`` has no static extent: a chunk may
    straddle it.  The row width is an ``int`` argument, whose packet residue
    the binding pins into its cache key, so the packet is loaded under the
    mask of its first lane; the lanes past the extent feed only the row fold,
    which is re-masked against its entry value instead of each packet."""
    kernel = helion.kernel(jagged_mean_kernel.fn, backend="cute", fast_math=True)
    x, offsets = _jagged_args()
    counts = torch.empty(17, dtype=torch.int32)
    bound = _cpu_bind(kernel, (x, offsets, counts, 128))
    assert (
        "cute_flat_gather_int_factor_residue_v1"
        in bound.env.runtime_input_specializations
    )
    code = _code(kernel, (x, offsets, counts, 128), SUM_CONFIG)
    assert "_vsink_chunk_mask = lane_base_1 < feature_counts" in code
    assert code.count("_cute_load_v16b_if(x_flat.iterator") == 8
    assert ", mask_2_u0 and _vsink_chunk_mask, ir.VectorType" in code
    assert "if mask_2_u0 and mask_1_u0 else cutlass.Float32(0)" not in code
    assert (
        "sum_1_lane_acc_frag[vec_lane_1] = sum_1_lane_acc_frag[vec_lane_1] if mask_1_u0"
        " else sum_1_lane_acc_entry_frag[vec_lane_1]"
    ) in code
    assert "x_flat.layout.stride[0])).load()" not in code
    # The vectorized seeds lead for this kernel too.
    seeds = bound.config_spec.compiler_seed_configs
    assert seeds and seeds[0].config.get("cute_vloop_sink") is True


def test_an_int_row_width_that_is_not_a_packet_multiple_keeps_the_scalar_gather(
    cpu_only: None,
) -> None:
    kernel = helion.kernel(jagged_mean_kernel.fn, backend="cute", fast_math=True)
    x, offsets = _jagged_args(132)
    counts = torch.empty(17, dtype=torch.int32)
    code = _code(kernel, (x, offsets, counts, 130), SUM_CONFIG)
    assert "_cute_load_v16b_if" not in code
    assert "x_flat.layout.stride[0])).load()" in code


def test_a_scaled_lane_term_keeps_the_scalar_gather(cpu_only: None) -> None:
    @helion.kernel(backend="cute")
    def strided_gather(x_data: torch.Tensor, x_offsets: torch.Tensor) -> torch.Tensor:
        M = x_data.shape[1]
        num_rows = x_offsets.size(0) - 1
        out = torch.zeros([num_rows, M // 2], dtype=x_data.dtype, device=x_data.device)
        x_flat = x_data.view(-1)
        for tile_b in hl.tile(num_rows):
            starts = x_offsets[tile_b]
            ends = x_offsets[tile_b.index + 1]
            nnz = ends - starts
            for tile_m in hl.tile(M // 2):
                acc = hl.zeros([tile_b, tile_m], dtype=x_data.dtype)
                for tile_k in hl.jagged_tile(nnz):
                    base = starts[:, None] + tile_k.index[None, :]
                    # Every other feature: the lane coordinate is scaled, so
                    # consecutive lanes are not consecutive elements.
                    flat = base[:, :, None] * M + 2 * tile_m.index[None, None, :]
                    acc = acc + hl.load(x_flat, [flat]).sum(dim=1)
                out[tile_b, tile_m] = acc
        return out

    code = _code(strided_gather, _jagged_args(), SUM_CONFIG)
    assert "_cute_load_v16b_if" not in code
    assert "x_flat.layout.stride[0])).load()" in code


def test_an_odd_row_width_keeps_the_scalar_gather(cpu_only: None) -> None:
    # 130 features: a row's start is not a multiple of the packet, so the
    # packets would straddle rows (and the extent is not a multiple of V).
    kernel = helion.kernel(jagged_sum_kernel.fn, backend="cute")
    code = _code(
        kernel,
        _jagged_args(130),
        {**SUM_CONFIG, "block_sizes": [1, 256, 128]},
    )
    assert "_cute_load_v16b_if" not in code
    assert "x_flat.layout.stride[0])).load()" in code


def test_a_wide_lane_loads_two_packets_per_row_in_one_sweep(cpu_only: None) -> None:
    kernel = helion.kernel(jagged_sum_kernel.fn, backend="cute")
    code = _code(
        kernel,
        _jagged_args(),
        {**SUM_CONFIG, "num_threads": [1, 16, 8], "cute_vector_widths": [1, 8, 1]},
    )
    # Eight features per lane span two consecutive 16-byte packets: both
    # load under the row's predicate in one call, so the 128-column tile is
    # covered by 16 threads in a single sweep over the rows instead of two
    # 64-column sweeps, each a dependent round of loads and a combine.
    assert code.count("_cute_load_v32b_if(x_flat.iterator") == 8
    assert "_cute_load_v16b_if(" not in code
    (feature_loop,) = _loops(code, "lane_1")
    assert ast.unparse(feature_loop.iter) == "range(1)"
    assert "row_sums_frag = cute.make_rmem_tensor(8, cutlass.Float32)" in code
    assert code.count("_cute_grouped_reduce_shared_two_stage_fragment(") == 1
    assert code.count("_cute_store_u32_vec(out.iterator") == 1
    assert "block=(16, 8, 1)" in code


MEAN_CONFIG = {
    "block_sizes": [1, 64, 128],
    "num_threads": [1, 16, 32],
    "cute_vector_widths": [1, 4, 1],
    "cute_lane_layouts": ["strided", "blocked", "strided"],
    "cute_vloop_sink": True,
    "cute_lane_unroll": 4,
    "cute_min_blocks_per_mp": 1,
}


def _mean_args() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    return (
        torch.empty((4096, 64)),
        torch.empty(65, dtype=torch.int64),
        torch.empty(64, dtype=torch.int32),
        64,
    )


def test_a_lane_invariant_select_conjunct_moves_onto_the_fold(cpu_only: None) -> None:
    kernel = helion.kernel(jagged_mean_kernel.fn, backend="cute", fast_math=True)
    code = _code(kernel, _mean_args(), MEAN_CONFIG)
    # jagged_mean's feature tile is jagged: the gather's select guard is
    # ``row_mask and feature_mask`` with the feature mask invariant across the
    # rows.  Its only consumer is the row fold, so the select is gone from the
    # strip (no per-packet select behind a load, which made ptxas reuse one
    # destination register and serialize the rows) and the fold is re-masked
    # against its entry value after each update instead.
    assert "if mask_2_u0 and mask_1_u0 else" not in code
    assert (
        "x_slice_u0 = cutlass.Uint32(_vsink_vec_u0[vec_lane_1]).bitcast(cutlass.Float32)"
        in code
    )
    assert (
        "sum_1_lane_acc_entry_frag[vec_lane_1] = sum_1_lane_acc_frag[vec_lane_1]"
        in code
    )
    assert (
        "sum_1_lane_acc_frag[vec_lane_1] = sum_1_lane_acc_frag[vec_lane_1] if mask_1_u0"
        " else sum_1_lane_acc_entry_frag[vec_lane_1]"
    ) in code
    # The cross-thread combine of the fold is still hoisted out of the V-loop.
    assert code.count("_cute_grouped_reduce_shared_two_stage_fragment(") == 1
    assert "block=(16, 32, 1)" in code


@helion.kernel(fast_math=True, backend="cute")
def _jagged_sum_and_sumsq(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
    x_feature_counts: torch.Tensor,
    max_M: int,
) -> torch.Tensor:
    num_rows = x_offsets.size(0) - 1
    out = torch.zeros([num_rows, max_M], dtype=x_data.dtype, device=x_data.device)
    x_flat = x_data.view(-1)
    for tile_b in hl.tile(num_rows):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        nnz = ends - starts
        feature_counts = x_feature_counts[tile_b]
        for tile_m in hl.jagged_tile(feature_counts):
            row_sums = hl.zeros([tile_b, tile_m], dtype=x_data.dtype)
            row_sq = hl.zeros([tile_b, tile_m], dtype=x_data.dtype)
            for tile_k in hl.jagged_tile(nnz):
                flat_indices = (starts[:, None] + tile_k.index[None, :])[
                    :, :, None
                ] * max_M
                flat_indices = flat_indices + tile_m.index[None, None, :]
                x_slice = hl.load(x_flat, [flat_indices])
                row_sums = row_sums + x_slice.sum(dim=1)
                row_sq = row_sq + (x_slice * x_slice).sum(dim=1)
            out[tile_b, tile_m] = row_sums + row_sq
    return out


def test_a_second_consumer_keeps_the_select(cpu_only: None) -> None:
    # ``x_slice`` feeds two folds (its sum and its square's sum): the loaded
    # value is observed other than through one identity-matched fold, so the
    # per-lane select stays on the packet.
    code = _code(_jagged_sum_and_sumsq, _mean_args(), MEAN_CONFIG)
    assert "_cute_load_v16b_if(x_flat.iterator" in code
    assert "and mask_1_u0 else cutlass.Float32(0)" in code
    assert "_entry" not in code


@helion.kernel(fast_math=True, backend="cute")
def _jagged_max_kernel(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
    x_feature_counts: torch.Tensor,
    max_M: int,
) -> torch.Tensor:
    num_rows = x_offsets.size(0) - 1
    out = torch.zeros([num_rows, max_M], dtype=x_data.dtype, device=x_data.device)
    x_flat = x_data.view(-1)
    for tile_b in hl.tile(num_rows):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        nnz = ends - starts
        feature_counts = x_feature_counts[tile_b]
        for tile_m in hl.jagged_tile(feature_counts):
            row_max = hl.full([tile_b, tile_m], float("-inf"), dtype=x_data.dtype)
            for tile_k in hl.jagged_tile(nnz):
                flat_indices = (starts[:, None] + tile_k.index[None, :])[
                    :, :, None
                ] * max_M
                flat_indices = flat_indices + tile_m.index[None, None, :]
                x_slice = hl.load(x_flat, [flat_indices])
                row_max = torch.maximum(row_max, x_slice.amax(dim=1))
            out[tile_b, tile_m] = row_max
    return out


def _jagged_inputs(
    rows: int, length: int, features: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    torch.manual_seed(0)
    lengths = torch.randint(1, length + 1, (rows,), device=DEVICE)
    offsets = torch.nn.functional.pad(lengths.cumsum(0), (1, 0))
    counts = torch.randint(1, features + 1, (rows,), device=DEVICE, dtype=torch.int32)
    x = torch.randn(int(offsets[-1]), features, device=DEVICE)
    # Rows past feature_counts hold values the fold must not observe.
    x[:, features // 2 :] += 1000.0
    return x, offsets, counts, features


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fold_mask_hoist_matches_the_reference_on_a_jagged_feature_tile() -> None:
    from examples.jagged_mean import reference_jagged_mean_kernel_pytorch

    x, offsets, counts, features = _jagged_inputs(64, 128, 64)
    kernel = helion.kernel(jagged_mean_kernel.fn, backend="cute", fast_math=True)
    bound = kernel.bind((x, offsets, counts, features))
    config = helion.Config(**MEAN_CONFIG)
    code = bound.to_code(config)
    assert "sum_1_lane_acc_entry_frag" in code
    out = bound.compile_config(config)(x, offsets, counts, features)
    expected = reference_jagged_mean_kernel_pytorch(x, offsets, counts, features)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-3)
    # A max fold over the same jagged feature tile (its select carries -inf).
    x, offsets, counts, features = _jagged_inputs(64, 128, 64)
    bound = _jagged_max_kernel.bind((x, offsets, counts, features))
    out = bound.compile_config(config)(x, offsets, counts, features)
    expected = torch.zeros_like(out)
    for row in range(64):
        rows = x[int(offsets[row]) : int(offsets[row + 1])]
        valid = int(counts[row])
        expected[row, :valid] = rows[:, :valid].amax(dim=0)
    torch.testing.assert_close(out, expected)


@helion.kernel(backend="cute", fast_math=True)
def _jagged_softmax_into(
    x_data: torch.Tensor, x_offsets: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """``jagged_softmax_kernel`` storing into a user tensor instead of its own."""
    num_rows, M = x_offsets.size(0) - 1, x_data.size(1)
    x_flat = x_data.view(-1)
    for tile_b in hl.tile(num_rows):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        seqlens = ends - starts
        for tile_m in hl.tile(M):
            block_max = hl.full([tile_b, tile_m], float("-inf"), dtype=x_data.dtype)
            block_new_max = hl.full([tile_b, tile_m], float("-inf"), dtype=x_data.dtype)
            block_L = hl.full([tile_b, tile_m], 0.0, dtype=x_data.dtype)
            for tile_k in hl.jagged_tile(seqlens):
                base_indices = starts[:, None] + tile_k.index[None, :]
                flat_indices = (
                    base_indices[:, :, None] * M + tile_m.index[None, None, :]
                )
                x_slice = hl.load(x_flat, [flat_indices])
                slice_max = x_slice.amax(dim=1)
                block_new_max = torch.maximum(block_max, slice_max)
                block_L *= torch.exp(block_max - block_new_max)
                block_L += torch.exp(x_slice - block_new_max[:, None, :]).sum(dim=1)
                block_max = block_new_max
            for tile_k in hl.jagged_tile(seqlens):
                base_indices = starts[:, None] + tile_k.index[None, :]
                flat_indices = (
                    base_indices[:, :, None] * M + tile_m.index[None, None, :]
                )
                x_slice = hl.load(x_flat, [flat_indices])
                block_out = (
                    torch.exp(x_slice - block_max[:, None, :]) / block_L[:, None, :]
                )
                hl.store(y, [flat_indices], block_out)
    return y


def test_a_store_into_a_possible_alias_disables_the_packet_cache(
    cpu_only: None,
) -> None:
    # Pass one's sum sweep re-reads the rows its max sweep loaded.  When the
    # output ``y`` shares its storage with ``x_data`` (whose flat view the
    # sweeps read), the kernel's stores into ``y`` may change those rows, so
    # the second sweep may not be served from the first sweep's registers:
    # it re-issues its loads.  "Never written" is not proven by name but by
    # the kernel's disjointness proof (``proven_disjoint_tensor_pairs``).
    x, offsets = _jagged_args()
    code = _code(_jagged_softmax_into, (x, offsets, x.view(-1)), SOFTMAX_CONFIG)
    assert "_vsink_cache" not in code
    # Both sweeps of pass one load their own packets (8 + 8); pass two's
    # loads stay scalar, in order with its stores into the aliasing ``y``.
    assert code.count("_cute_load_v16b_if(x_flat.iterator") == 16
    # A separately stored ``y`` is proven disjoint for this binding (the
    # storage-span predicate is part of the bound kernel's cache key), as is
    # jagged_softmax's own freshly allocated ``out``: the cache stays.
    code = _code(
        _jagged_softmax_into, (x, offsets, torch.empty(731 * 128)), SOFTMAX_CONFIG
    )
    assert "_vsink_cache = cute.make_rmem_tensor(64, cutlass.Uint32)" in code
    kernel = helion.kernel(jagged_softmax_kernel.fn, backend="cute", fast_math=True)
    code = _code(kernel, _jagged_args(), SOFTMAX_CONFIG)
    assert "_vsink_cache = cute.make_rmem_tensor(64, cutlass.Uint32)" in code
    assert code.count("_cute_load_v16b_if(x_flat.iterator") == 16


@helion.kernel(backend="cute", fast_math=True)
def _jagged_mean_with_row_store(
    x_data: torch.Tensor,
    x_offsets: torch.Tensor,
    x_feature_counts: torch.Tensor,
    max_M: int,
    y: torch.Tensor,
) -> torch.Tensor:
    num_rows = x_offsets.size(0) - 1
    out = torch.zeros([num_rows, max_M], dtype=x_data.dtype, device=x_data.device)
    x_flat = x_data.view(-1)
    for tile_b in hl.tile(num_rows):
        starts = x_offsets[tile_b]
        ends = x_offsets[tile_b.index + 1]
        nnz = ends - starts
        feature_counts = x_feature_counts[tile_b]
        for tile_m in hl.jagged_tile(feature_counts):
            row_sums = hl.zeros([tile_b, tile_m], dtype=x_data.dtype)
            for tile_k in hl.jagged_tile(nnz):
                flat_indices = (starts[:, None] + tile_k.index[None, :])[
                    :, :, None
                ] * max_M
                flat_indices = flat_indices + tile_m.index[None, None, :]
                x_slice = hl.load(x_flat, [flat_indices])
                row_sums = row_sums + x_slice.sum(dim=1)
                # A store into ``y`` on every row, reading neither the
                # gathered value nor the fold.
                y[tile_b, tile_k, tile_m] = hl.full(
                    [tile_b, tile_k, tile_m], 1.0, dtype=x_data.dtype
                )
            out[tile_b, tile_m] = row_sums
    return out


def test_an_effectful_row_loop_keeps_the_select_on_its_scalar_load(
    cpu_only: None,
) -> None:
    # The row loop stores into ``y`` on every row while ``y`` shares its
    # storage with the input: the gather could not be widened across that
    # store, and a fold-mask hoist on the scalar load would read rows past
    # ``nnz`` unconditionally.  The backend refuses the schedule before the
    # sink runs (the lane reduction would reorder a possibly aliasing write);
    # the sink itself drops a marked select only once its load was widened
    # into a predicated packet (``_is_lane_extract``).
    x, offsets, counts, features = _mean_args()
    with pytest.raises(helion.exc.BackendUnsupported, match="aliasing write"):
        _code(
            _jagged_mean_with_row_store,
            (x, offsets, counts, features, x.view(64, 64, 64)),
            MEAN_CONFIG,
        )
    # With a separately stored ``y`` (proven disjoint) the loads are widened
    # and the hoist applies as for jagged_mean.
    code = _code(
        _jagged_mean_with_row_store,
        (x, offsets, counts, features, torch.empty((64, 64, 64))),
        MEAN_CONFIG,
    )
    assert "_cute_load_v16b_if(x_flat" in code
    assert "if mask_2_u0 and mask_1_u0 else" not in code
    assert "_entry" in code


def test_an_extremum_fold_keeps_its_select(cpu_only: None) -> None:
    # Only the zero identity is admitted: a dropped select would let the rows
    # past the row mask contribute the packet's zero fill, which is not the
    # identity of a max.  The max kernel keeps every select and no entry
    # value is saved.
    code = _code(_jagged_max_kernel, _mean_args(), MEAN_CONFIG)
    assert "_entry" not in code
    assert "float('-inf')" in code
