"""Native role geometry must survive generic per-region thread budgeting."""

from __future__ import annotations

import ast
from collections import Counter
from contextlib import contextmanager
import math
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.int4_gemm import matmul_bf16_int4
from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_fwd
import pytest
import torch

from test.test_cute_full_slice_matmul import _bind
from test.test_cute_fuse_mm_accumulation import _cpu_target
from test.test_cute_materialized_fission import _se_args
from test.test_cute_materialized_fission import _sources

import helion
from helion._compiler.cute.device_state import CuteDeviceFunctionState
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion._compiler.cute.device_state import CuteTcgen05MatmulPlan
    from helion.runtime.kernel import BoundKernel


@pytest.fixture(autouse=True)
def _cpu_b200() -> Iterator[None]:
    with (
        _cpu_target(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        yield


@contextmanager
def _capture_plans() -> Iterator[
    list[tuple[CuteDeviceFunctionState, CuteTcgen05MatmulPlan]]
]:
    plans: list[tuple[CuteDeviceFunctionState, CuteTcgen05MatmulPlan]] = []
    original = CuteDeviceFunctionState.register_tcgen05_matmul_plan

    def register(state: CuteDeviceFunctionState, plan: CuteTcgen05MatmulPlan) -> None:
        original(state, plan)
        plans.append((state, plan))

    with patch.object(
        CuteDeviceFunctionState, "register_tcgen05_matmul_plan", register
    ):
        yield plans


def _launch_shape(source: str) -> tuple[int, int, int]:
    blocks = [
        ast.literal_eval(keyword.value)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "block"
    ]
    assert len(blocks) == 1
    return blocks[0]


def _assert_role_ownership(source: str, plan: CuteTcgen05MatmulPlan) -> None:
    shape = _launch_shape(source)
    assert shape == plan.block_shape
    assert math.prod(shape) == 32 * plan.launched_warp_count
    assert plan.physical_m_threads == 32
    roles: Counter[str] = Counter()
    for thread in range(math.prod(shape)):
        warp = thread // 32
        if warp < plan.epi_warp_count:
            roles["epilogue"] += 1
        elif warp == plan.exec_warp_id:
            roles["mma"] += 1
        elif plan.ab_load_warp_begin <= warp < plan.ab_load_warp_end:
            roles["ab"] += 1
        else:
            roles["unused"] += 1
    assert roles == {
        "epilogue": 32 * plan.epi_warp_count,
        "mma": 32,
        "ab": 32 * plan.ab_load_warp_count,
    }
    # The monolithic role policy gives epi/exec threads 256 registers and
    # producers 120. Extra physical warps made the old 64x6 launch require
    # 67,840 registers, so setmaxregister_increase could not complete.
    assert (roles["epilogue"] + roles["mma"]) * 256 + roles["ab"] * 120 <= 65536


def _materialized_bound(dtype: torch.dtype) -> BoundKernel:
    kernel = helion.kernel(
        matmul_bf16_int4.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        cute_region_fission=True,
        cute_materialize_transformed_operands=True,
    )
    bound = kernel._bind_isolated(
        (
            torch.empty((512, 1024), dtype=dtype),
            torch.empty((512, 512), dtype=torch.int8),
        )
    )
    assert bound.env.cute_fission_plan is not None
    return bound


def _materialized_config(
    *, producer_order: list[int], m_tile: int = 128, n_tile: int = 16
) -> helion.Config:
    return helion.Config(
        block_sizes=[512, 32, m_tile, n_tile, 64],
        num_threads=[32, 32, 0, 0, 0],
        cute_lane_layouts=["strided", "blocked", "blocked", "blocked", "blocked"],
        cute_vector_widths=[2, 4, 1, 1, 1],
        loop_orders=[producer_order, [0, 1]],
        l2_groupings=[64, 32],
        pid_type="persistent_interleaved",
        cute_collective_mma=False,
        tcgen05_ab_stages=6,
        tcgen05_acc_stages=1,
        tcgen05_c_stages=4,
        tcgen05_l2_swizzle_size=4,
        tcgen05_cluster_m=1,
        tcgen05_cluster_n=1,
        tcgen05_num_epi_warps=4,
        tcgen05_strategy="role_local_monolithic",
        tcgen05_persistence_model="static_persistent",
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("producer_order", [[0, 1], [1, 0]])
@pytest.mark.parametrize("n_tile", [16, 32])
def test_materialized_narrow_consumer_uses_native_role_geometry(
    dtype: torch.dtype, producer_order: list[int], n_tile: int
) -> None:
    bound = _materialized_bound(dtype)
    config = _materialized_config(producer_order=producer_order, n_tile=n_tile)
    before_config = config.config.copy()
    with _capture_plans() as plans:
        sources = _sources(bound.to_code(config))
    assert config.config == before_config
    assert len(sources) == 2
    assert len(plans) == 1
    assert _launch_shape(sources[0]) == (32, 32, 1)
    assert "cute.gemm(" not in sources[0]
    assert "cute.nvgpu.tcgen05.CtaGroup.ONE" in sources[1]
    _assert_role_ownership(sources[1], plans[0][1])


def test_native_consumer_geometry_does_not_change_materialization_region() -> None:
    bound = _materialized_bound(torch.bfloat16)
    narrow = _materialized_config(producer_order=[1, 0], n_tile=16)
    wide = _materialized_config(producer_order=[1, 0], n_tile=128)
    narrow_sources = _sources(bound.to_code(narrow))
    wide_sources = _sources(bound.to_code(wide))
    assert narrow_sources[0] == wide_sources[0]
    assert narrow_sources[1] != wide_sources[1]
    assert _launch_shape(narrow_sources[1]) == _launch_shape(wide_sources[1])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_wide_m_two_cta_consumer_keeps_each_ctas_role_geometry(
    dtype: torch.dtype,
) -> None:
    bound = _materialized_bound(dtype)
    config = _materialized_config(producer_order=[0, 1], m_tile=256)
    config = helion.Config.from_dict(config.config | {"tcgen05_cluster_m": 2})
    with _capture_plans() as plans:
        sources = _sources(bound.to_code(config))
    assert len(plans) == 1
    plan = plans[0][1]
    assert (plan.bm, plan.bn) == (256, 16)
    assert plan.is_two_cta
    assert _launch_shape(sources[0]) == (32, 32, 1)
    assert "cute.nvgpu.tcgen05.CtaGroup.TWO" in sources[1]
    _assert_role_ownership(sources[1], plan)


@pytest.mark.parametrize(
    "tiles", [[128, 16, 16, 128, 64, 32], [128, 32, 32, 128, 16, 16]]
)
def test_two_native_regions_keep_independent_geometry_and_plans(
    tiles: list[int],
) -> None:
    bound = _bind(
        squeeze_and_excitation_net_fwd.fn,
        _se_args((512, 256, 64)),
        fission=True,
    )
    config = helion.Config(
        block_sizes=tiles,
        num_threads=[0] * 6,
        loop_orders=[[0, 1], [0, 1]],
        pid_type="flat",
        cute_collective_mma=False,
        tcgen05_cluster_m=1,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=1,
        tcgen05_c_stages=2,
    )
    with _capture_plans() as plans:
        sources = _sources(bound.to_code(config))
    assert len(sources) == len(plans) == 2
    assert plans[0][0] is not plans[1][0]
    for source, (_, plan), offset in zip(sources, plans, [0, 3], strict=True):
        assert (plan.bm, plan.bn, plan.bk) == tuple(tiles[offset : offset + 3])
        assert plan.output_offsets == (
            f"tile_offset_{offset}",
            f"tile_offset_{offset + 1}",
        )
        _assert_role_ownership(source, plan)
