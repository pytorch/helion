from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from examples.grouped_gemm import grouped_gemm_jagged
import pytest
import torch

from test._cute_dense_tcgen05 import vector_scaled_mm
from test.test_cute_epilogue_fanout import bias_gelu_aux
from test.test_cute_epilogue_fanout import gelu_aux_scaled
from test.test_cute_epilogue_fanout import paired
from test.test_cute_shared_rhs_grouped import _plans
from test.test_cute_tcgen05_fixed_chain import _batched_gemm

import helion
from helion._compiler.autotuner_heuristics.cute import grouped_row_union_carrier
from helion._compiler.autotuner_heuristics.cute import (
    grouped_row_union_cluster4_carrier,
)
from helion._compiler.autotuner_heuristics.cute import (
    grouped_row_union_paired_clc_carrier,
)
from helion._compiler.cute.epilogue_fanout import FANOUT_CONFIG_KEY
from helion._compiler.cute.grouped_row_union import AB_STAGES_KEY
from helion._compiler.cute.grouped_row_union import CONFIG_KEY as ROW_UNION_KEY
from helion._compiler.cute.grouped_row_union import PAIRED_CLC_SCHEDULE
from helion._compiler.cute.grouped_row_union import RESIDENT_CTAS_KEY
from helion._compiler.cute.grouped_row_union import SCHEDULE_KEY
from helion._compiler.cute.grouped_row_union import TRANSPOSED_SCHEDULE
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Callable
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
    schedule: Literal["cluster4", "resident2", "paired"],
    *,
    ab_stages: int | None = None,
    pdl: bool = False,
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
        carrier = {
            "cluster4": grouped_row_union_cluster4_carrier,
            "paired": grouped_row_union_paired_clc_carrier,
            "resident2": grouped_row_union_carrier,
        }[schedule]
        config = carrier(bound.env, bound.host_function.device_ir)
        assert config is not None
        config.config[ROW_UNION_KEY] = True
        if schedule == "resident2":
            config.config[RESIDENT_CTAS_KEY] = 2
        if ab_stages is not None:
            config.config[AB_STAGES_KEY] = ab_stages
        if pdl:
            config.config["cute_pdl"] = True
        normalized = bound.config_spec.normalized_config(config)
    assert normalized[ROW_UNION_KEY] is True
    assert normalized.config.get(AB_STAGES_KEY) == ab_stages
    source = bound.to_code(config)
    assert "cute.gemm(" in source and "row_union_store_pred" in source
    ab = _plans(source)[0]
    assert ab["kind"] == "tcgen05_ab_tma"
    if ab_stages is not None:
        # The searched physical ring depth, carried by the plan into the
        # launcher's descriptor contract.
        assert (ab["ab_stage_count"], ab["row_union_ab_stages"]) == (
            ab_stages,
            ab_stages,
        )
        assert f"PipelineTmaUmma.create(num_stages={ab_stages}," in source
    if pdl:
        # No entry wait: the TMA-load and epilogue roles (and the paired
        # profile's coverage warp) wait themselves ahead of their reads.
        assert "'use_pdl': True" in source
        kernel_body = source[source.index("@cute.kernel") :]
        first = kernel_body.splitlines()[2].strip()
        assert first != "cute.arch.griddepcontrol_wait()", first
        assert source.count("cute.arch.griddepcontrol_wait()") >= 2
    else:
        assert "'use_pdl': True" not in source
    if schedule == "cluster4":
        assert normalized[SCHEDULE_KEY] == TRANSPOSED_SCHEDULE
        assert ab["row_union_schedule"] == TRANSPOSED_SCHEDULE
        assert (ab["cluster_m"], ab["cluster_n"]) == (2, 2)
        assert "_MAX_ACTIVE_CLUSTERS" in source and "_NUM_SM // 4" not in source
    elif schedule == "paired":
        assert normalized[SCHEDULE_KEY] == PAIRED_CLC_SCHEDULE
        assert ab["row_union_schedule"] == PAIRED_CLC_SCHEDULE
    else:
        assert normalized[RESIDENT_CTAS_KEY] == 2
        assert (ab["bm"], ab["bn"], ab["bk"]) == (128, 64, 128)
        assert "_NUM_SM * 2" in source
    return bound, config


@pytest.mark.parametrize(
    ("schedule", "ab_stages"),
    [("cluster4", None), ("cluster4", 4), ("cluster4", 6), ("resident2", None)],
)
def test_native_row_union_cuda_replay(
    schedule: Literal["cluster4", "resident2"], ab_stages: int | None
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
    bound, config = _row_union_bound(args, schedule, ab_stages=ab_stages)
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


def _row_union_check(
    result: torch.Tensor,
    routing: list[int],
    a: torch.Tensor,
    b: torch.Tensor,
) -> None:
    m, n = a.size(0), b.size(1)
    assert result.shape == (m, n) and result.dtype == torch.bfloat16
    valid = torch.zeros(m, dtype=torch.bool, device=a.device)
    for first, last in itertools.pairwise(routing):
        begin, end = max(0, first), min(m, last)
        if end > begin:
            valid[begin:end] = True
    assert bool(valid.any())
    expected = (a.float() @ b.float()).to(torch.bfloat16)
    torch.testing.assert_close(result[valid], expected[valid], rtol=3e-2, atol=3e-3)


@pytest.mark.parametrize("schedule", ["cluster4", "paired"])
def test_native_row_union_dependent_launch_cuda(
    schedule: Literal["cluster4", "paired"],
) -> None:
    """``cute_pdl`` on the row-union profiles: the group offsets are read
    behind the epilogue role's wait (the coverage warp's on the paired
    profile), not at kernel entry.  A producer that releases its dependents
    at entry and rewrites the offsets late runs ahead of every launch, in
    the stream and as two dependent launches inside one CUDA graph with the
    offsets rewritten between them; a read ahead of the wait would see the
    other layout's rows."""
    _require_sm100()
    from test.test_cute_pdl_cuda import _produce

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(93217)
    if schedule == "cluster4":
        m, n, k = 5120, 512, 512
        layouts = (
            [0, 640, 1280, 1920, 2560, 2560, 2560, 2560, 2560],
            [2560, 3200, 3840, 4480, 5120, 5120, 5120, 5120, 5120],
        )
    else:
        m, n, k = 768, 256, 512
        layouts = (
            [0, 64, 128, 192, 256, 320, 384] + [384] * 25,
            [384, 448, 512, 576, 640, 704, 768] + [768] * 25,
        )
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16, generator=generator)
    b = torch.randn((k, n), device=device, dtype=torch.bfloat16, generator=generator)
    a.mul_(0.125)
    b.mul_(0.125)
    sources = [
        torch.tensor(layout, device=device, dtype=torch.int32) for layout in layouts
    ]
    offsets = sources[0].clone()
    args = (a, b, offsets)
    bound, config = _row_union_bound(args, schedule, pdl=True)
    bound.set_config(config)
    for i in range(16):
        j = i & 1
        a.normal_(std=0.125, generator=generator)
        _produce(sources[j], offsets, delay=4096)
        _row_union_check(bound(*args), layouts[j], a, b)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _produce(sources[0], offsets, delay=4096)
        first = bound(*args)
        _produce(sources[1], offsets, delay=4096)
        second = bound(*args)
    for _ in range(3):
        a.normal_(std=0.125, generator=generator)
        first.fill_(float("nan"))
        second.fill_(float("nan"))
        graph.replay()
        _row_union_check(first, layouts[0], a, b)
        _row_union_check(second, layouts[1], a, b)
        assert torch.equal(offsets, sources[1])


def _two_output_bound(
    fn: Callable[..., object],
    args: tuple[torch.Tensor, ...],
    *,
    cluster: tuple[int, int],
    mode: Literal["off", "shared"],
) -> BoundKernel:
    cluster_m, cluster_n = cluster
    kernel = helion.kernel(
        fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = kernel.bind(args)
    config = helion.Config(
        block_sizes=[128 * cluster_m, 128, 64],
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=cluster_m,
        tcgen05_cluster_n=cluster_n,
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=4 if cluster_m == 1 else 2,
        tcgen05_num_epi_warps=4,
    )
    config.config[FANOUT_CONFIG_KEY] = mode
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    assert normalized.config.get(FANOUT_CONFIG_KEY, "off") == mode
    source = bound.to_code(normalized)
    ab, first, second = _plans(source)
    assert ab["kind"] == "tcgen05_ab_tma"
    assert (ab["cluster_m"], ab["cluster_n"]) == cluster
    assert first["kind"] == second["kind"] == "tcgen05_d_tma"
    assert source.count("cute.copy(tcgen05_tiled_copy_t2r") == (
        1 if mode == "shared" else 2
    )
    bound.set_config(normalized)
    return bound


@pytest.mark.parametrize("cluster", [(1, 1), (2, 1), (2, 2)])
def test_native_gelu_aux_shared_fanout_cuda(cluster: tuple[int, int]) -> None:
    """The cuBLASLt GELU_AUX pattern (``aux = pre.to(dt); out = gelu(pre)``,
    a row-vector bias in the shared FP32 prefix) and its scaled variant (a
    rank-0 device scalar in the second chain) under the shared two-output
    epilogue, on one CTA, the M-paired CTA group and the 2x2 cluster of
    pairs: bit-identical to the two-traversal kernel and within eager
    tolerance of torch."""
    _require_sm100()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(93219)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    m, n, k = 256 * (sm_count // 16 + 1), 2048, 128
    dtype = torch.float16
    left, right = (
        torch.randn(shape, device=device, dtype=dtype, generator=generator) * 0.125
        for shape in ((m, k), (k, n))
    )
    bias = torch.randn((n,), device=device, dtype=dtype, generator=generator)
    out_scale = torch.tensor(0.75, device=device)
    pre = (left.float() @ right.float()) * 1.25 + bias.float()
    for fn, args, expected in (
        (
            bias_gelu_aux,
            (left, right, bias),
            (torch.nn.functional.gelu(pre).to(dtype), pre.to(dtype)),
        ),
        (
            gelu_aux_scaled,
            (left, right, bias, out_scale),
            ((torch.nn.functional.gelu(pre) * 0.75).to(dtype), pre.to(dtype)),
        ),
    ):
        serial = _two_output_bound(fn, args, cluster=cluster, mode="off")
        shared = _two_output_bound(fn, args, cluster=cluster, mode="shared")
        control = serial(*args)
        result = shared(*args)
        torch.cuda.synchronize()
        for actual, reference in zip(result, expected, strict=True):
            assert actual.shape == (m, n) and actual.dtype == dtype
            torch.testing.assert_close(actual, reference, rtol=3e-2, atol=3e-3)
        torch.testing.assert_close(result, control, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("shape", "bn", "cluster_n"),
    [
        # Odd batch count, three N tiles (cluster_n=1 only).
        ((7, 384, 192, 512), 64, 1),
        # The 2x2 cluster of pairs: whole N-tile pairs.
        ((8, 256, 256, 512), 64, 2),
        ((5, 256, 256, 256), 128, 2),
    ],
)
def test_native_batched_two_cta_paired_cuda(
    shape: tuple[int, int, int, int], bn: int, cluster_n: int
) -> None:
    """Batched GEMMs on the explicit CtaGroup.TWO family: the batch rides the
    scheduler's first dimension (both CTAs of a pair decode one batch)."""
    _require_sm100()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(93221)
    batch, m, n, k = shape
    a = torch.randn(
        (batch, m, k), device=device, dtype=torch.float16, generator=generator
    )
    b = torch.randn(
        (batch, k, n), device=device, dtype=torch.float16, generator=generator
    )
    a.mul_(0.125)
    b.mul_(0.125)
    kernel = helion.kernel(
        _batched_gemm, backend="cute", static_shapes=True, autotune_effort="none"
    )
    bound = kernel.bind((a, b))
    requested = helion.Config(
        block_sizes=[1, 128, bn, 64],
        loop_orders=[[0, 1, 2]],
        pid_type="persistent_interleaved",
        tcgen05_cta_group="two",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=cluster_n,
        tcgen05_ab_stages=4,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
    )
    with bound.env:
        config = bound.config_spec.normalized_config(requested)
    assert config.config["tcgen05_cta_group"] == "two"
    assert config.config["tcgen05_cluster_n"] == cluster_n
    source = bound.to_code(config)
    assert f"_helion_cute_cluster_shape = (2, {cluster_n}, 1)" in source
    assert "CtaGroup.TWO" in source
    assert f"PersistentTileSchedulerParams(({batch} * 2, " in source
    bound.set_config(config)
    expected = torch.bmm(a.float(), b.float()).to(torch.float16)
    for _ in range(2):
        result = bound(a, b)
        torch.cuda.synchronize()
        torch.testing.assert_close(result, expected, rtol=3e-2, atol=3e-3)


def test_native_fp8_pair_cluster_n2_cuda() -> None:
    """fp8 CTA pairs in the 2x2 cluster (A multicast across the N
    neighbours), the vector-scaled GEMM of the pipeline SMEM tests."""
    _require_sm100()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(93223)
    m, n, k = 256, 2048, 4096
    a = (
        torch.randn((m, k), device=device, dtype=torch.float32, generator=generator)
        * 0.25
    ).to(torch.float8_e4m3fn)
    b = (
        (
            torch.randn((n, k), device=device, dtype=torch.float32, generator=generator)
            * 0.25
        )
        .to(torch.float8_e4m3fn)
        .T
    )
    scale_a = torch.rand((m,), device=device, generator=generator) + 0.5
    scale_b = torch.rand((n,), device=device, generator=generator) + 0.5
    args = (a, b, scale_a, scale_b)
    bound = vector_scaled_mm.bind(args)
    requested = helion.Config.from_dict(
        {
            "block_sizes": [128, 128, 128],
            "tcgen05_cta_group": "two",
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 2,
            "pid_type": "persistent_blocked",
            "tcgen05_ab_stages": 2,
            "tcgen05_c_stages": 2,
            "tcgen05_acc_stages": 2,
        }
    )
    with bound.env:
        config = bound.config_spec.normalized_config(requested)
    assert config.config["tcgen05_cta_group"] == "two"
    assert config.config["tcgen05_cluster_n"] == 2
    source = bound.to_code(config)
    assert "_helion_cute_cluster_shape = (2, 2, 1)" in source
    assert "CtaGroup.TWO" in source
    bound.set_config(config)
    expected = (a.float() @ b.float()) * scale_a[:, None] * scale_b[None, :]
    for _ in range(2):
        result = bound(*args)
        torch.cuda.synchronize()
        assert result.dtype == torch.bfloat16
        torch.testing.assert_close(result.float(), expected, rtol=3e-2, atol=3e-2)
