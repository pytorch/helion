from __future__ import annotations

from itertools import accumulate

from examples.jagged_dense_bmm import jagged_dense_bmm
import pytest
import torch

import helion
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])

CUDA_DEVICE = "cuda"


def _reference(
    lengths: list[int],
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # CPU FP64 avoids both the candidate conversion and global CUDA TF32 settings.
    left, right, addend = (value.cpu().double() for value in (jagged, dense, bias))
    result = torch.empty((left.size(0), right.size(2)), dtype=torch.float64)
    magnitude = torch.empty_like(result)
    start = 0
    for group, length in enumerate(lengths):
        stop = start + length
        result[start:stop] = left[start:stop] @ right[group] + addend[group]
        magnitude[start:stop] = left[start:stop].abs() @ right[group].abs()
        start = stop
    assert start == left.size(0)
    return result, magnitude


def _check_output(
    actual: torch.Tensor,
    expected: torch.Tensor,
    magnitude: torch.Tensor,
    *,
    rounded: bool,
) -> None:
    assert actual.dtype is torch.float32 and actual.shape == expected.shape
    actual_cpu = actual.cpu().double()
    if rounded:
        # One TF32 ulp per input permits truncation as well as RN/RNA.
        # 0.0021 exceeds 2*2**-10 + 2**-20 plus FP32 accumulation error
        # for these K<=256 cases. The dyadic passes below require exact output.
        allowance = magnitude * 0.0021 + 1e-5
        error = (actual_cpu - expected).abs()
        assert torch.all(error <= allowance), (error / allowance).max().item()
    else:
        torch.testing.assert_close(actual_cpu, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "selection,reduction,columns,groups,offset_dtype",
    [
        pytest.param(
            {"cute_grouped_rna_warps": 1, "cute_grouped_rna_block_k": 64},
            256,
            68,
            5,
            torch.int32,
            id="rna-dynamic-tail-ring-wrap",
        ),
        pytest.param(
            {
                "cute_grouped_rna_warps": 8,
                "cute_grouped_rna_block_k": 32,
                "cute_grouped_descriptor_policy": "wrapped",
            },
            128,
            128,
            5,
            torch.int64,
            id="rna-wrapped",
        ),
        pytest.param(
            {
                "cute_mma_f32_conversion": "tma_rn",
                "cute_grouped_rna_block_k": 32,
                "cute_grouped_ab_stages": 2,
            },
            128,
            68,
            37,
            torch.int64,
            id="tma-rn-dynamic-warp-prefix-tail",
        ),
        pytest.param(
            {
                "cute_mma_f32_conversion": "tma_rn",
                "cute_grouped_rna_block_k": 32,
                "cute_grouped_ab_stages": 2,
                "cute_grouped_ctas_per_sm": 2,
                "cute_grouped_descriptor_policy": "wrapped",
                "cute_grouped_prefix_scan": "block",
            },
            128,
            128,
            65,
            torch.int64,
            id="tma-rn-wrapped-block-prefix-two-ctas",
        ),
        pytest.param(
            {"cute_mma_f32_conversion": "warp_raw"},
            128,
            68,
            5,
            torch.int64,
            id="raw-warp-k-refill-n-tail",
        ),
    ],
)
def test_grouped_fp32_modes_numerics_and_graph_replay(
    selection: dict[str, object],
    reduction: int,
    columns: int,
    groups: int,
    offset_dtype: torch.dtype,
) -> None:
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("grouped native schedules require SM100")
    # Unequal, empty, and partial row tiles; the larger case spans prefix warps.
    lengths = [0, 1, 0, 129, 257] + [1] * (groups - 5)
    rows = sum(lengths)
    generator = torch.Generator().manual_seed(7123)

    def values(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randint(-8, 9, shape, generator=generator).float() / 16

    offsets = torch.tensor(
        list(accumulate(lengths, initial=0)), dtype=offset_dtype, device=CUDA_DEVICE
    )
    jagged = values((rows, reduction)).to(CUDA_DEVICE)
    dense = values((groups, reduction, columns)).to(CUDA_DEVICE)
    bias = values((groups, columns)).to(CUDA_DEVICE)
    inputs = (offsets, jagged, dense, bias)
    kernel = helion.kernel(
        jagged_dense_bmm.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        dot_precision="tf32",
        cute_region_fission=True,
        cute_full_slice_matmul_tiling=True,
        cute_segmented_matmul_tiling=True,
        cute_flatten_nested_reductions=True,
        cute_materialize_transformed_operands=True,
    )
    bound = kernel.bind(inputs)
    config = bound.config_spec.normalized_config(
        helion.Config.from_dict(
            bound.config_spec._base_default_config().config | selection
        )
    )
    for key, value in selection.items():
        assert config[key] == value
    # A numerical pass from a silent fallback would not cover the selected mode.
    assert "_helion_cute_native_grouped_plan" in bound.to_code(config)
    bound.set_config(config)
    expected, magnitude = _reference(lengths, jagged, dense, bias)
    first = bound(*inputs)
    _check_output(first, expected, magnitude, rounded=False)
    second = bound(*inputs)
    assert first.data_ptr() != second.data_ptr()
    first.fill_(float("nan"))
    _check_output(second, expected, magnitude, rounded=False)

    # Compile and warm outside capture, retaining the original allocating caller.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        bound(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = bound(*inputs)

    for rounded in (False, True):
        lengths = list(reversed(lengths))
        offsets.copy_(
            torch.tensor(list(accumulate(lengths, initial=0)), dtype=offset_dtype)
        )
        jagged.copy_(values((rows, reduction)))
        dense.copy_(values((groups, reduction, columns)))
        bias.copy_(values((groups, columns)))
        if rounded:
            # Finite values around TF32 rounding boundaries, including cancellation.
            jagged.add_(2**-12)
            dense.add_(2**-12)
        expected, magnitude = _reference(lengths, jagged, dense, bias)
        captured.fill_(float("nan"))
        graph.replay()
        _check_output(captured, expected, magnitude, rounded=rounded)
