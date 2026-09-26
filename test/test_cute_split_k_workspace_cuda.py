from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from examples.matmul_split_k import matmul_split_k
import pytest
import torch

from test.test_cute_collective_chunk_seeds import _direct_chunk_matmul
from test.test_cute_split_k_workspace import _fp32_bias
from test.test_cute_split_k_workspace_protocol import _matrix_bias

import helion
from helion.autotuner.benchmarking import _make_cudagraph_replay


@dataclass(frozen=True)
class WorkspaceCase:
    name: str
    shape: tuple[int, int, int]
    tile: tuple[int, int, int]
    parameter: str
    value: int
    stages: int
    kind: str = "original"
    bias: bool = False
    dtype: torch.dtype = torch.float16
    padding: int = 0


WORKSPACE_CASES = (
    WorkspaceCase(
        "original-plain", (128, 32768, 256), (128, 128, 64), "split_k", 32, 12
    ),
    WorkspaceCase(
        "original-bias",
        (128, 32768, 256),
        (128, 128, 64),
        "split_k",
        32,
        12,
        bias=True,
    ),
    WorkspaceCase("n64-deep", (128, 4096, 256), (128, 64, 64), "split_k", 16, 16),
    WorkspaceCase(
        "persistent-multiwave",
        (512, 2048, 512),
        (128, 128, 64),
        "split_k",
        16,
        2,
        padding=8,
    ),
    WorkspaceCase(
        "bf16-fp32-padded",
        (256, 4096, 256),
        (256, 128, 128),
        "partitions",
        8,
        2,
        kind="fp32",
        bias=True,
        dtype=torch.bfloat16,
        padding=8,
    ),
    WorkspaceCase(
        "direct-three-partitions",
        (128, 192, 128),
        (128, 128, 64),
        "tile_extent",
        64,
        2,
        kind="direct",
    ),
    WorkspaceCase(
        "matrix-bias-three-partitions",
        (128, 192, 128),
        (128, 128, 64),
        "tile_extent",
        64,
        2,
        kind="matrix",
        bias=True,
        padding=8,
    ),
)


def workspace_case(case: WorkspaceCase, device: str):
    """Shared CPU-preflight/GPU input construction, including aligned views."""
    m, k, n = case.shape
    offset = case.padding
    a = torch.empty((m, k + offset), dtype=case.dtype, device=device)[:, offset:]
    b = torch.empty((k, n + offset), dtype=case.dtype, device=device)[:, offset:]
    if case.kind == "matrix":
        bias = torch.empty((m, n * 2), dtype=case.dtype, device=device)[:, ::2]
        kernel, arguments = _matrix_bias, (a, b, bias)
    else:
        bias = torch.empty((n * 3,), dtype=case.dtype, device=device)[::3]
        if case.kind == "fp32":
            kernel, arguments = _fp32_bias, (a, b, bias)
        elif case.kind == "direct":
            kernel, arguments = _direct_chunk_matmul, (a, b)
        else:
            kernel = helion.kernel(
                matmul_split_k.fn,
                backend="cute",
                static_shapes=True,
                autotune_effort="none",
            )
            arguments = (
                (a, b, lambda acc, tile: acc + bias[tile[1]]) if case.bias else (a, b)
            )
    config = helion.Config.from_dict(
        {
            "block_sizes": list(case.tile),
            case.parameter: case.value,
            "cute_split_k_workspace": True,
            "cute_split_k_stages": case.stages,
        }
    )
    return kernel._bind_isolated(arguments), config, arguments, (a, b, bias)


def ordered_reduction(
    partials: torch.Tensor, bias: torch.Tensor | None, dtype: torch.dtype
) -> torch.Tensor:
    """The emitted four partition-lane sums, including the partition-zero bias."""
    partitions, rows, columns = partials.shape
    lanes = torch.zeros((4, rows, columns), dtype=torch.float32, device=partials.device)
    for partition in range(partitions):
        value = partials[partition]
        if partition == 0 and bias is not None:
            value = value + bias.float()
        lane = partition % 4
        lanes[lane].copy_(lanes[lane] + value)
    return ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])).to(dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("case", WORKSPACE_CASES, ids=lambda case: case.name)
def test_native_split_k_workspace_partials_and_poisoned_graph(case: WorkspaceCase):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(2026091371)
    bound, config, arguments, (a, b, bias) = workspace_case(case, "cuda")
    code = bound.to_code(config)
    assert "_helion_cute_split_k_workspace" in code
    bound.set_config(config)
    partial_buffers: list[torch.Tensor] = []
    original_empty = torch.empty

    def capture_empty(*args: Any, **kwargs: Any):
        result = original_empty(*args, **kwargs)
        if result.ndim == 3 and result.dtype == torch.float32:
            partial_buffers.append(result)
        return result

    def run():
        before = len(partial_buffers)
        result = bound(*arguments)
        assert len(partial_buffers) == before + 1
        return result, partial_buffers[-1]

    def check(result, partials, *, integer_inputs=False):
        p, m, n = partials.shape
        chunk = a.size(1) // p
        expected_partials = torch.bmm(
            a.as_strided((p, m, chunk), (chunk, *a.stride())).float(),
            b.as_strided((p, chunk, n), (chunk * b.stride(0), *b.stride())).float(),
        )
        torch.testing.assert_close(
            partials,
            expected_partials,
            atol=0 if integer_inputs else 1e-3,
            rtol=0 if integer_inputs else 1e-3,
        )
        torch.testing.assert_close(
            result,
            ordered_reduction(partials, bias if case.bias else None, result.dtype),
            atol=0,
            rtol=0,
        )
        reference = a.float() @ b.float()
        if case.bias:
            reference = reference + bias.float()
        # The archived example's unchanged numerical contract.
        torch.testing.assert_close(
            result, reference.to(result.dtype), atol=1, rtol=1e-2
        )

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with patch("torch.empty", side_effect=capture_empty):
            # Integer products and sums are exactly representable in FP32;
            # checking every partial detects lost/duplicated partition ownership.
            a.random_(-2, 3)
            b.random_(-2, 3)
            bias.random_(-2, 3)
            eager, partials = run()
            check(eager, partials, integer_inputs=True)
            replay = _make_cudagraph_replay(run)
            for _iteration in range(4):
                a.normal_()
                b.normal_()
                bias.normal_()
                eager, partials = run()
                check(eager, partials)
                # Sixteen replays per mutation: 64 poisoned graph executions.
                for _repeat in range(16):
                    result, partials = replay()
                    check(result, partials)
                    result.fill_(float("nan"))
                    partials.fill_(float("nan"))
            a.zero_()
            b.zero_()
            bias.zero_()
            result, partials = replay()
            torch.testing.assert_close(result, torch.zeros_like(result), atol=0, rtol=0)
            torch.testing.assert_close(
                partials, torch.zeros_like(partials), atol=0, rtol=0
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


FALLBACK_SHAPES = ((32, 4096, 64), (64, 16384, 128), (65, 530, 71), (128, 0, 128))


def fallback_case(shape, device):
    m, k, n = shape
    a = torch.empty((m, k), dtype=torch.float16, device=device)
    b = torch.empty((k, n), dtype=torch.float16, device=device)
    kernel = helion.kernel(
        matmul_split_k.fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = kernel._bind_isolated((a, b))
    assert not bound.config_spec.cute_split_k_workspace_available
    config = helion.Config(
        block_sizes=[32, 64, 32],
        split_k=4,
        num_threads=[2, 64, 1],
        cute_collective_mma=k != 0,
        cute_collective_compute="warp",
        cute_collective_copy="async_cached",
        cute_collective_stages=4 if k else 1,
    )
    return bound, config, (a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape", FALLBACK_SHAPES)
def test_split_k_workspace_fallback_keeps_fp32_and_resets_graph(shape):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(2026091372)
    bound, config, (a, b) = fallback_case(shape, "cuda")
    source = bound.to_code(config)
    assert "_helion_cute_split_k_workspace" not in source
    assert "dtype=torch.float32" in source and "return out.to(torch.float16)" in source
    bound.set_config(config)
    a.normal_()
    b.normal_()
    replay = _make_cudagraph_replay(lambda: bound(a, b))
    for _iteration in range(4):
        a.normal_()
        b.normal_()
        expected = (a.float() @ b.float()).half()
        result = replay()
        torch.testing.assert_close(result, expected, atol=1, rtol=1e-2)
        result.fill_(float("nan"))
    a.zero_()
    b.zero_()
    result = replay()
    torch.testing.assert_close(result, torch.zeros_like(result), atol=0, rtol=0)
