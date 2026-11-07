"""Native integer-boundary masks and both attention recurrences."""

from __future__ import annotations

from typing import cast

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing.cuda_graph import cuda_graph_for_replay
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")
driver = pytest.importorskip("cuda.bindings.driver")

logical_attention_mask = pytest.importorskip(
    "helion._compiler.cute._paged_attention_compute"
).logical_attention_mask
logical_attention_tile_is_interior = pytest.importorskip(
    "helion._compiler.cute._paged_attention_compute"
).logical_attention_tile_is_interior


@cute.kernel
def _mask_kernel(
    bases,
    lengths,
    output,
    interior,
    coordinate_dtype: cutlass.Constexpr,
    window: cutlass.Constexpr,
):
    case = cute.arch.block_idx()[0]
    thread = cute.arch.thread_idx()[0]
    if thread == 0:
        interior[case] = cutlass.Int32(
            logical_attention_tile_is_interior(
                bases[case], 128, lengths[case], 40, 5, 8, window, coordinate_dtype
            )
        )
    for query in cutlass.range_constexpr(40):
        output[case, query, thread] = cutlass.Int32(
            logical_attention_mask(
                bases[case],
                thread,
                lengths[case],
                query,
                5,
                8,
                window,
                coordinate_dtype,
            )
        )


@cute.jit
def _mask_launch(
    bases,
    lengths,
    output,
    interior,
    coordinate_dtype: cutlass.Constexpr,
    window: cutlass.Constexpr,
    stream: driver.CUstream,
):
    _mask_kernel(bases, lengths, output, interior, coordinate_dtype, window).launch(
        grid=(cute.size(bases), 1, 1), block=(128, 1, 1), stream=stream
    )


@skipIfNotCUDA()
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("window", [None, 31, 2047])
def test_native_mask_preserves_wrapping_before_promotion(dtype, window):
    # One tiny launch covers subtraction overflow, row-add overflow, wide and
    # negative key positions, then changed-input replay at both index widths.
    bases = torch.tensor(
        [
            0,
            0,
            2**31 - 64,
            -128,
            0,
            0,
            384,
            2**32,
            128,
            2**63 - 64,
            -(2**31),
            0,
            -(2**31) + 2,
            1,
            0,
        ],
        device=DEVICE,
    )
    lengths = torch.tensor(
        [
            -(2**31),
            -(2**31) + 4,
            2**31 - 1,
            0,
            5,
            128,
            400,
            2**31 - 1,
            400,
            2**31 - 1,
            -(2**31) + 4,
            3000,
            -(2**31) + 2050,
            2049,
            2049,
        ],
        dtype=torch.int32,
        device=DEVICE,
    )
    # Case 12 has a partially wrapped window lower bound in int32:
    # earlier query rows wrap to MAX, while the last rows remain near MIN.
    cases = len(bases)
    output = torch.empty((cases, 40, 128), dtype=torch.int32, device=DEVICE)
    interior = torch.empty((cases,), dtype=torch.int32, device=DEVICE)
    tensors = tuple(
        cute.runtime.from_dlpack(value) for value in (bases, lengths, output, interior)
    )
    scalar_type = cutlass.Int64 if dtype == torch.int64 else cutlass.Int32
    stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(_mask_launch, *tensors, scalar_type, window, stream)
    compiled(*tensors, stream)
    with cuda_graph_for_replay() as graph:
        with torch.cuda.graph(graph):
            capture_stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
            compiled(*tensors, capture_stream)
        for changed in (False, True):
            if changed:
                bases.add_(64)
                lengths.copy_(
                    torch.tensor(
                        [
                            -(2**31) + 4,
                            -(2**31) + 10,
                            2**31 - 3,
                            9,
                            8,
                            65,
                            500,
                            2**31 - 1,
                            450,
                            2**31 - 1,
                            -(2**31),
                            3100,
                            -(2**31) + 2051,
                            2113,
                            2113,
                        ],
                        dtype=torch.int32,
                        device=DEVICE,
                    )
                )
            graph.replay()
            # Case 8 has keys 128..255 and query positions 395..399; after mutation,
            # keys 192..319 and positions 445..449. Window 31 excludes both tiles.
            assert bool(interior[8]) is (window != 31)
            # With window 2047, case 13 starts exactly at the last query row
            # lower bound; case 14 starts one key earlier, including after replay.
            assert bool(interior[13]) is (window != 31)
            assert bool(interior[14]) is (window is None)
            assert bool(interior[12]) is (
                window is None or (window == 2047 and dtype == torch.int64)
            )
            for case in range(cases):
                # The scalar Int32 subtraction must complete before the vector's
                # index dtype controls promotion in the following addition.
                positions = (lengths[case] - 5) + torch.arange(
                    40, dtype=dtype, device=DEVICE
                ) // 8
                keys = bases[case].to(dtype) + torch.arange(
                    128, dtype=dtype, device=DEVICE
                )
                expected = keys[None, :] <= positions[:, None]
                if window is not None:
                    expected &= keys[None, :] >= positions[:, None] - window
                torch.testing.assert_close(output[case], expected.int(), rtol=0, atol=0)
                assert not bool(interior[case]) or bool(expected.all())


@skipIfNotCUDA()
@skipIfRefEager("requires the selected native attention route")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("per_key", [False, True])
def test_attention_mask_boundaries_keep_owner_and_replay(dtype, per_key):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the paged pipeline uses SM100 tensor memory")
    rows = 40 if per_key else 64
    args = list(paged_attention_args(device=DEVICE, rows=rows))
    q, k, v, sk, sv, pages, lengths = (cast("torch.Tensor", arg) for arg in args[:7])
    q.zero_()
    k.zero_()
    v.fill_(1)
    sk.fill_(127)
    sv.fill_(127)
    pages.zero_()
    lengths.copy_(torch.tensor([-(2**31) + 4, 63], dtype=torch.int32, device=DEVICE))
    args[-1] = -1
    args.extend((torch.tensor([[0, 1], [0, 1]], device=DEVICE), per_key))
    kernel = helion.kernel(
        paged_online_attention.fn, backend="cute", static_shapes=True, index_dtype=dtype
    )
    bound = kernel.bind(tuple(args))
    with bound.env:
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[64],
                cute_mma_dag="tcgen05",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
                cute_mma_dag_pass_warps="attention",
                cute_min_blocks_per_mp=1,
            )
        )
    owner = "paged_attention_partial_body" if per_key else "paged_attention_body"
    assert owner + "(" in bound.to_code(config)
    compiled = bound.compile_config(config)
    compiled(*args)
    with cuda_graph_for_replay() as graph:
        with torch.cuda.graph(graph):
            replay = compiled(*args)
        for changed in (False, True):
            if changed:
                lengths.copy_(
                    torch.tensor([-(2**31), 0], dtype=torch.int32, device=DEVICE)
                )
                v.fill_(2)
            actual = compiled(*args)
            graph.replay()
            for item in range(4):
                positions = (lengths[item // 2] - 5) + torch.arange(
                    rows, dtype=dtype, device=DEVICE
                ) // 8
                visible = (
                    torch.arange(128, dtype=dtype, device=DEVICE)[None, :]
                    <= positions[:, None]
                )
                count = visible.sum(1)
                expected_o = (
                    torch.where(count > 0, 2.0 if changed else 1.0, 0.0)[:, None]
                    .expand(rows, 128)
                    .bfloat16()
                )
                expected_lse = count.float().log()
                for output, lse in (actual, replay):
                    torch.testing.assert_close(output[item], expected_o, rtol=0, atol=0)
                    torch.testing.assert_close(
                        lse[item], expected_lse, rtol=0.00001, atol=0.00001
                    )
