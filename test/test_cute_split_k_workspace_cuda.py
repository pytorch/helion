from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest
import torch

from test.test_cute_collective_chunk_seeds import _direct_chunk_matmul
from test.test_cute_split_k_workspace import _fp32_bias
from test.test_cute_split_k_workspace import _plain_store_bias_matmul
from test.test_cute_split_k_workspace import _plain_store_matmul
from test.test_cute_split_k_workspace_protocol import _matrix_bias

import helion
from helion._testing import import_example
from helion._testing import skipUnlessBackends
from helion.autotuner.benchmarking import _make_cudagraph_replay

matmul_split_k = import_example("matmul_split_k").matmul_split_k

pytestmark = skipUnlessBackends(["cute"])


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
        "original-bf16",
        (128, 32768, 256),
        (128, 128, 64),
        "split_k",
        32,
        12,
        dtype=torch.bfloat16,
    ),
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
        "plain-store-bf16",
        (256, 4096, 256),
        (128, 128, 64),
        "cute_split_k_partitions",
        8,
        4,
        kind="store",
        dtype=torch.bfloat16,
    ),
    WorkspaceCase(
        "plain-store-bias-fp16",
        (128, 8192, 512),
        (128, 128, 64),
        "cute_split_k_partitions",
        16,
        6,
        kind="store",
        bias=True,
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
        elif case.kind == "store":
            kernel, arguments = (
                (_plain_store_bias_matmul, (a, b, bias))
                if case.bias
                else (_plain_store_matmul, (a, b))
            )
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


ATOMIC_EPILOGUE_CASES = (
    # The s2 catalogue winner's geometry: one 64x64 epilogue subtile per CTA.
    WorkspaceCase(
        "atomic-fp16-bias",
        (128, 32768, 256),
        (128, 64, 128),
        "split_k",
        16,
        6,
        bias=True,
    ),
    WorkspaceCase(
        "atomic-fp16-plain", (128, 32768, 256), (128, 64, 128), "split_k", 16, 6
    ),
    # 64x128 CTA tiles: two subtiles whose columns interleave (the (32, 2)
    # epilogue tiler), the exactness case below checks every output once.
    WorkspaceCase(
        "atomic-fp16-n128-bias",
        (128, 32768, 256),
        (128, 128, 64),
        "split_k",
        16,
        12,
        bias=True,
    ),
    WorkspaceCase(
        "atomic-fp16-exact",
        (128, 2048, 256),
        (128, 128, 64),
        "split_k",
        4,
        12,
        bias=True,
    ),
    WorkspaceCase(
        "atomic-bf16-plain",
        (128, 4096, 256),
        (128, 64, 64),
        "split_k",
        16,
        16,
        dtype=torch.bfloat16,
    ),
    # fp32 output (``red.global.add.v2.f32``), 128-row CTA tiles, padded bias.
    WorkspaceCase(
        "atomic-fp32-bias",
        (256, 4096, 256),
        (256, 128, 128),
        "partitions",
        8,
        2,
        kind="fp32",
        bias=True,
        padding=8,
    ),
    # A persistent grid with several tiles per CTA pair (the staging buffers
    # alternate under the atomics).
    WorkspaceCase(
        "atomic-fp16-persistent",
        (512, 2048, 512),
        (128, 128, 64),
        "split_k",
        16,
        2,
        bias=True,
    ),
)


def _rounded_partition_reference(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None, partitions: int
) -> torch.Tensor:
    """What the source's ``hl.atomic_add`` computes when every partial is
    rounded to the output dtype on its way in (exact for small integers)."""
    m, k = a.shape
    chunk = k // partitions
    total = torch.zeros((m, b.size(1)), dtype=torch.float32, device=a.device)
    for partition in range(partitions):
        partial = (
            a[:, partition * chunk : (partition + 1) * chunk].float()
            @ b[partition * chunk : (partition + 1) * chunk].float()
        )
        if partition == 0 and bias is not None:
            partial = partial + bias.float()
        total = total + partial.to(a.dtype).float()
    return total.to(a.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("case", ATOMIC_EPILOGUE_CASES, ids=lambda case: case.name)
def test_native_split_k_atomic_epilogue_reduces_into_the_filled_output(
    case: WorkspaceCase,
):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(2026103101)
    bound, config, arguments, (a, b, bias) = workspace_case(case, "cuda")
    config = helion.Config.from_dict(
        config.config | {"cute_split_k_atomic_epilogue": True}
    )
    code = bound.to_code(config)
    assert "'epilogue': 'atomic'" in code
    assert "_reduce_partitions" not in code
    assert code.count("PyCodeCache.load(") == 1
    assert (
        "_helion_zero_fill_split_k" in code
        and "torch.zeros" not in code.split("def ")[-1].split("return")[0]
    )
    bound.set_config(config)
    output_dtype = torch.float32 if case.kind == "fp32" else case.dtype
    partitions = case.value

    def run():
        return bound(*arguments)

    def check(result, *, exact: bool):
        assert result.dtype is output_dtype
        reference = a.float() @ b.float()
        if case.bias:
            reference = reference + bias.float()
        if exact:
            torch.testing.assert_close(
                result,
                _rounded_partition_reference(
                    a, b, bias if case.bias else None, partitions
                ).to(output_dtype),
                atol=0,
                rtol=0,
            )
        # The example's numerical contract (atol=1) for the half outputs, a
        # per-partition rounding error budget for bfloat16.
        tolerance = {
            torch.float16: {"atol": 1, "rtol": 1e-2},
            torch.bfloat16: {"atol": 4, "rtol": 5e-2},
            torch.float32: {"atol": 1e-2, "rtol": 1e-3},
        }[output_dtype]
        torch.testing.assert_close(result, reference.to(output_dtype), **tolerance)

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        # Small integers: every partial and every partial sum is exact in the
        # output dtype (|values| <= chunk <= 2048 and sums <= K <= 2048 for the
        # exact case), so the result is order independent.
        a.random_(-1, 2)
        b.random_(-1, 2)
        bias.random_(-1, 2)
        check(run(), exact=case.name == "atomic-fp16-exact")
        replay = _make_cudagraph_replay(run)
        for _iteration in range(3):
            a.normal_()
            b.normal_()
            bias.normal_()
            check(run(), exact=False)
            for _repeat in range(8):
                result = replay()
                check(result, exact=False)
                # The fill kernel must zero the output on every replay.
                result.fill_(float("nan"))
        a.zero_()
        b.zero_()
        bias.zero_()
        result = replay()
        torch.testing.assert_close(result, torch.zeros_like(result), atol=0, rtol=0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


FALLBACK_SHAPES = ((32, 4096, 64), (64, 16384, 128), (65, 530, 71), (128, 0, 128))


def fallback_case(shape, device):
    m, k, n = shape
    a = torch.empty((m, k), dtype=torch.float16, device=device)
    b = torch.empty((k, n), dtype=torch.float16, device=device)
    # The fallback under test is the fp32-promoted form of the float16 atomic
    # output (the Setting defaults to accumulating in float16 in place).
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_promote_half_atomic_outputs=True,
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
