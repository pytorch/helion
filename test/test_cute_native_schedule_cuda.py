from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from examples.grouped_gemm import grouped_gemm_jagged
import pytest
import torch

from test.test_cute_epilogue_fanout import paired
from test.test_cute_shared_rhs_grouped import _plans

import helion
from helion._compiler.autotuner_heuristics.cute import grouped_row_union_carrier
from helion._compiler.autotuner_heuristics.cute import (
    grouped_row_union_cluster4_carrier,
)
from helion._compiler.cute.epilogue_fanout import FANOUT_CONFIG_KEY
from helion._compiler.cute.grouped_row_union import CONFIG_KEY as ROW_UNION_KEY
from helion._compiler.cute.grouped_row_union import RESIDENT_CTAS_KEY
from helion._compiler.cute.grouped_row_union import SCHEDULE_KEY
from helion._compiler.cute.grouped_row_union import TRANSPOSED_SCHEDULE
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from typing import Literal

    from helion.runtime.kernel import BoundKernel


def _require_sm100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("these native schedules require SM100")


def _fanout_bound(
    args: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cluster_m: int,
    mode: Literal["off", "shared"],
) -> tuple[BoundKernel, helion.Config]:
    kernel = helion.kernel(
        paired, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = kernel.bind(args)
    config = helion.Config(
        block_sizes=[128 * cluster_m, 128, 64],
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=cluster_m,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=4 if cluster_m == 1 else 2,
        tcgen05_num_epi_warps=4,
    )
    config.config[FANOUT_CONFIG_KEY] = mode
    source = bound.to_code(config)
    ab, first, second = _plans(source)
    assert ab["kind"] == "tcgen05_ab_tma" and ab["cluster_m"] == cluster_m
    assert first["kind"] == second["kind"] == "tcgen05_d_tma"
    assert first["d_idx"] != second["d_idx"]
    assert "cute.gemm(" in source
    # Prove that the shared case executes one TMEM read for both output stores.
    assert source.count("cute.copy(tcgen05_tiled_copy_t2r") == (
        1 if mode == "shared" else 2
    )
    return bound, config


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cluster_m", [1, 2])
def test_native_shared_fanout_cuda_replay(dtype: torch.dtype, cluster_m: int) -> None:
    _require_sm100()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(93211)
    # More logical tiles than persistent CTAs exercise ACC and C-ring reuse.
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    m, n, k = 256 * (sm_count // 16 + 1), 2048, 128
    left, right, residual = (
        torch.randn(shape, device=device, dtype=dtype, generator=generator) * 0.125
        for shape in ((m, k), (k, n), (m, n))
    )
    args = (left, right, residual)
    original = tuple(tensor.clone() for tensor in args)
    serial, serial_config = _fanout_bound(args, cluster_m, "off")
    shared, shared_config = _fanout_bound(args, cluster_m, "shared")
    serial.set_config(serial_config)
    shared.set_config(shared_config)

    def check(result: tuple[torch.Tensor, torch.Tensor]) -> None:
        gate = torch.sigmoid(left.float() @ right.float()).to(dtype)
        expected = (gate, residual * gate)
        assert result[0].data_ptr() != result[1].data_ptr()
        for actual, reference in zip(result, expected, strict=True):
            assert actual.shape == (m, n) and actual.dtype == dtype
            torch.testing.assert_close(actual, reference, rtol=3e-2, atol=3e-3)

    control = serial(*args)
    torch.testing.assert_close(args, original, rtol=0, atol=0)
    eager = shared(*args)
    torch.testing.assert_close(args, original, rtol=0, atol=0)
    check(control)
    check(eager)
    torch.testing.assert_close(eager, control, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = shared(*args)
    for replay in range(3):
        if replay == 2:
            for tensor, saved in zip(args, original, strict=True):
                tensor.copy_(saved)
        else:
            for tensor in args:
                tensor.normal_(std=0.125, generator=generator)
        before = tuple(tensor.clone() for tensor in args)
        for output in captured:
            output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(args, before, rtol=0, atol=0)
        check(captured)
        torch.testing.assert_close(captured, serial(*args), rtol=0, atol=0)
        torch.testing.assert_close(args, before, rtol=0, atol=0)


def _row_union_bound(
    args: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    schedule: Literal["cluster4", "resident2"],
) -> tuple[BoundKernel, helion.Config]:
    kernel = helion.kernel(
        grouped_gemm_jagged.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        cute_segmented_matmul_tiling=True,
    )
    bound = kernel.bind(args)
    assert bound.host_function is not None
    with bound.env:
        carrier = (
            grouped_row_union_cluster4_carrier
            if schedule == "cluster4"
            else grouped_row_union_carrier
        )
        config = carrier(bound.env, bound.host_function.device_ir)
        assert config is not None
        config.config[ROW_UNION_KEY] = True
        if schedule == "resident2":
            config.config[RESIDENT_CTAS_KEY] = 2
        normalized = bound.config_spec.normalized_config(config)
    assert normalized[ROW_UNION_KEY] is True
    source = bound.to_code(config)
    assert "cute.gemm(" in source and "row_union_store_pred" in source
    ab = _plans(source)[0]
    assert ab["kind"] == "tcgen05_ab_tma"
    if schedule == "cluster4":
        assert normalized[SCHEDULE_KEY] == TRANSPOSED_SCHEDULE
        assert ab["row_union_schedule"] == TRANSPOSED_SCHEDULE
        assert (ab["cluster_m"], ab["cluster_n"]) == (2, 2)
        assert "_NUM_SM // 4" in source
    else:
        assert normalized[RESIDENT_CTAS_KEY] == 2
        assert (ab["bm"], ab["bn"], ab["bk"]) == (128, 64, 128)
        assert "_NUM_SM * 2" in source
    return bound, config


@pytest.mark.parametrize("schedule", ["cluster4", "resident2"])
def test_native_row_union_cuda_replay(
    schedule: Literal["cluster4", "resident2"],
) -> None:
    _require_sm100()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(93213)
    # Allocation satisfies both profiles; individual intervals have partial tiles.
    m, n, k = 5120, 512, 512
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16, generator=generator)
    b = torch.randn((k, n), device=device, dtype=torch.bfloat16, generator=generator)
    a.mul_(0.125)
    b.mul_(0.125)
    full = [0, 13, 13, 479, 1280, 1280, 3819, 5001, m]
    offsets = torch.tensor(full, device=device, dtype=torch.int32)
    args = (a, b, offsets)
    bound, config = _row_union_bound(args, schedule)
    bound.set_config(config)

    def check(result: torch.Tensor, routing: list[int]) -> None:
        assert result.shape == (m, n) and result.dtype == torch.bfloat16
        # The allocating example leaves rows outside the interval union undefined.
        valid = torch.zeros(m, dtype=torch.bool, device=device)
        for first, last in itertools.pairwise(routing):
            begin, end = max(0, first), min(m, last)
            if end > begin:
                valid[begin:end] = True
        expected = (a.float() @ b.float()).to(torch.bfloat16)
        torch.testing.assert_close(result[valid], expected[valid], rtol=3e-2, atol=3e-3)

    before = tuple(tensor.clone() for tensor in args)
    check(bound(*args), full)
    torch.testing.assert_close(args, before, rtol=0, atol=0)
    # Capture a partial worklist, then require new rows on replay. A stale
    # captured worklist would leave those required rows poisoned.
    offsets.copy_(offsets.new_tensor([13, 479, 479, 479, 479, 479, 479, 479, 479]))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = bound(*args)
    for routing in (
        [m - 509, m - 19, 128, 769, 769, 64, 97, 97, 97],
        [-31, 193, 193, m + 67, m + 67, m + 67, -17, -17, -17],
        [7] * 9,
        full,
    ):
        a.normal_(std=0.125, generator=generator)
        b.normal_(std=0.125, generator=generator)
        offsets.copy_(offsets.new_tensor(routing))
        before = tuple(tensor.clone() for tensor in args)
        captured.fill_(float("nan"))
        graph.replay()
        check(captured, routing)
        torch.testing.assert_close(args, before, rtol=0, atol=0)
