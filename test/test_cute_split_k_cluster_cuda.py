from __future__ import annotations

from typing import Any
from typing import cast

from examples.matmul_split_k import matmul_split_k
import pytest
import torch

from test.test_cute_split_k_workspace import _fp32_bias

import helion
from helion._compiler.autotuner_heuristics.cute_split_k_cluster import cluster_carrier
from helion._testing import skipUnlessBackends

CUDA_DEVICE = "cuda"
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    skipUnlessBackends(["cute"]),
]


def _check_cluster(
    shape: tuple[int, int, int],
    *,
    bias: bool,
    finalizer_warps: int,
    fp32_output: bool = False,
) -> None:
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Cluster shared memory requires SM90 or newer")
    m, k, n = shape
    dtype = torch.bfloat16 if fp32_output else torch.float16
    device = torch.device(CUDA_DEVICE)
    generator = torch.Generator(device=device).manual_seed(7021)
    # Offset by a complete vector to exercise aligned, padded input row strides.
    padding = 8 if k > 4096 or fp32_output else 0
    a = torch.empty((m, k + padding), dtype=dtype, device=device)[:, padding:]
    b = torch.empty((k, n + padding), dtype=dtype, device=device)[:, padding:]
    bias_tensor = torch.empty((n * 3,), dtype=dtype, device=device)[::3]

    def refresh() -> None:
        a.normal_(std=k**-0.5, generator=generator)
        b.normal_(generator=generator)
        bias_tensor.normal_(generator=generator)

    refresh()
    kernel = helion.kernel(
        _fp32_bias.fn if fp32_output else matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_full_slice_matmul_tiling=False,
        cute_region_fission=False,
    )
    args: tuple[object, ...]
    if fp32_output:
        args = (a, b, bias_tensor)
    elif bias:
        args = (a, b, lambda acc, tile: acc + bias_tensor[tile[1]])
    else:
        args = (a, b)
    bound = kernel.bind(args)
    with bound.env, bound.host_function:
        config = cluster_carrier(bound.env, bound.host_function.device_ir)
    assert config is not None
    config.config["cute_split_k_finalizer_warps"] = finalizer_warps
    bound.set_config(config)
    assert bound._run is not None
    compiled = cast("Any", bound._run)
    metadata = compiled._helion_cute_split_k_schedule
    assert metadata["cluster"] == (8, 1, 1)
    assert metadata.get("finalizer_warps", 1) == finalizer_warps

    def check(output: torch.Tensor) -> None:
        expected = a.double() @ b.double()
        if bias:
            expected = expected + bias_tensor.double()
        tolerance = 2e-4 if fp32_output else 2e-3
        torch.testing.assert_close(
            output,
            expected.to(torch.float32 if fp32_output else dtype),
            rtol=tolerance,
            atol=tolerance,
        )

    first = compiled(*args)
    second = compiled(*args)
    assert first.data_ptr() != second.data_ptr()
    check(first)
    check(second)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_output = compiled(*args)
    for _ in range(2):
        refresh()
        replay_output.fill_(float("nan"))
        graph.replay()
        check(replay_output)


@pytest.mark.parametrize("shape", ((32, 4096, 64), (48, 8192, 80)))
@pytest.mark.parametrize("bias", (False, True))
@pytest.mark.parametrize("finalizer_warps", (1, 4))
def test_cluster_finalizers_execute_and_replay(
    shape: tuple[int, int, int], bias: bool, finalizer_warps: int
) -> None:
    _check_cluster(shape, bias=bias, finalizer_warps=finalizer_warps)


@pytest.mark.parametrize("finalizer_warps", (1, 4))
def test_cluster_bfloat16_fp32_output_and_strided_bias(finalizer_warps: int) -> None:
    _check_cluster(
        (48, 8192, 80), bias=True, finalizer_warps=finalizer_warps, fp32_output=True
    )
