from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from ._cute_binding import _cpu_bind
from ._cute_binding import _mock_cuda_unavailable
from .cute_population_contracts import _target
from .test_cute_grouped_fp32_output import _grouped_fp32_output
import helion
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.runtime.cute.launcher import _append_cute_wrapper_plan
from helion.runtime.cute.launcher import _tcgen05_grouped_dynamic_tensormap_workspace


def _config(multiplier: int) -> helion.Config:
    return helion.Config(
        block_sizes=[256, 128, 64],
        pid_type="persistent_interleaved",
        num_sm_multiplier=multiplier,
        tcgen05_cluster_m=1,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=5,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
        tcgen05_grouped_mode="worklist_nm",
        tcgen05_grouped_worklist_source_m_tile=32,
    )


@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_grouped_multiplier_owns_actual_grid_workspace(
    offset_dtype: torch.dtype,
) -> None:
    args = (
        torch.empty((640, 128), dtype=torch.bfloat16),
        torch.empty((4, 256, 128), dtype=torch.bfloat16),
        torch.tensor([0, 0, 1, 129, 640], dtype=offset_dtype),
    )
    with _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped_fp32_output, args)
        config = bound.config_spec.normalized_config(_config(2))
        assert config["num_sm_multiplier"] == 2
        code = bound.to_code(config)
    ast.parse(code)
    assert "_NUM_SM * 2" in code
    assert "cute.arch.grid_dim()" in code
    assert "tcgen05_grouped_tensormap_workspace_idx" in code
    assert "StaticPersistentGroupTileScheduler.create" in code
    assert "'num_sm_multiplier': 2" in code
    assert code.count(".relinquish_alloc_permit()") == 1
    assert code.index(".relinquish_alloc_permit()") < code.index(
        "StaticPersistentGroupTileScheduler.create"
    )
    wrapper_assignment = next(
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "_helion_cute_wrapper_plans"
    )
    plans = ast.literal_eval(wrapper_assignment.value)
    grouped = next(
        plan for plan in plans if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    body: list[str] = []
    arguments: list[str] = []
    _append_cute_wrapper_plan(body, arguments, grouped, num_sm=148)
    assert "cutlass.Int32(296)" in "\n".join(body)


def test_grouped_multiplier_workspace_cache_matches_expanded_grid() -> None:
    kernel = SimpleNamespace()
    device = torch.device("cuda", 0)
    with (
        patch("helion.runtime.cute.launcher.get_num_sm", return_value=148),
        patch(
            "helion.runtime.cute.launcher._cuda_stream_capture_context",
            return_value=(7, None),
        ),
        patch(
            "helion.runtime.cute.launcher.torch.empty",
            side_effect=lambda shape, **kwargs: torch.zeros(
                shape, dtype=kwargs["dtype"]
            ),
        ),
    ):
        first = _tcgen05_grouped_dynamic_tensormap_workspace(
            kernel, device=device, tensormap_count=3
        )
        second = _tcgen05_grouped_dynamic_tensormap_workspace(
            kernel, device=device, tensormap_count=3, num_sm_multiplier=2
        )
        again = _tcgen05_grouped_dynamic_tensormap_workspace(
            kernel, device=device, tensormap_count=3, num_sm_multiplier=2
        )
    assert first.shape == (148, 3, 16)
    assert second.shape == (296, 3, 16)
    assert again is second and first is not second


@pytest.mark.parametrize(
    "change",
    [
        {"num_sm_multiplier": True},
        {"num_sm_multiplier": 2.0},
        {"num_sm_multiplier": 3},
        {"tcgen05_cluster_m": 2},
        {"tcgen05_cluster_n": 2},
        {"tcgen05_grouped_mode": "worklist"},
        {"tcgen05_grouped_runtime_direct": True},
        {"tcgen05_grouped_static_reserved_sms": 1},
    ],
)
def test_grouped_multiplier_keeps_cooperative_contracts(change: dict) -> None:
    config = dict(_config(2).config)
    config.update(change)
    assert not CuteTcgen05Config._grouped_grid_multiplier_supported(config)


@skipUnlessBackends(["cute"])
def test_grouped_multiplier_live_offsets() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("grouped tcgen05 requires SM100-family")
    groups, rows, width, columns = 128, 2, 128, 1024
    generator = torch.Generator(device=DEVICE).manual_seed(731)
    a = torch.randn(
        (groups * rows, width), generator=generator, device=DEVICE
    ).bfloat16()
    b = torch.randn(
        (groups, columns, width), generator=generator, device=DEVICE
    ).bfloat16()
    offsets = torch.arange(groups + 1, device=DEVICE, dtype=torch.int32) * rows
    bound = _grouped_fp32_output.bind((a, b, offsets))
    bound.set_config(_config(2))
    expected = torch.bmm(a.view(groups, rows, width).float(), b.float().transpose(1, 2))
    expected = expected.view(groups * rows, columns)
    actual = bound(a, b, offsets)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    offsets[1] = 0
    expected[: 2 * rows] = a[: 2 * rows].float() @ b[1].float().T
    changed = bound(a, b, offsets)
    torch.testing.assert_close(changed, expected, atol=2e-4, rtol=2e-4)
