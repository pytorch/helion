from __future__ import annotations

import ast

import pytest
import torch

from ._cute_binding import _cpu_bind
from ._cute_binding import _mock_cuda_unavailable
from .cute_population_contracts import _target
from .test_cute_grouped_fp32_output import _grouped_fp32_output
from .test_cute_grouped_grid_multiplier import _config
import helion
from helion import exc
from helion.runtime.cute.launcher import _append_cute_wrapper_plan


@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_global_weights_host_descriptor_matches_device_coordinates(
    offset_dtype: torch.dtype,
) -> None:
    args = (
        torch.empty((640, 128), dtype=torch.bfloat16),
        torch.empty((4, 256, 128), dtype=torch.bfloat16),
        torch.tensor([0, 0, 1, 129, 640], dtype=offset_dtype),
    )
    config = dict(_config(1).config)
    config["tcgen05_grouped_full_allocation_a"] = True
    with _target(), _mock_cuda_unavailable():
        code = _cpu_bind(_grouped_fp32_output, args).to_code(
            helion.Config.from_dict(config)
        )
    parsed = ast.parse(code)
    assignment = next(
        node
        for node in ast.walk(parsed)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "_helion_cute_wrapper_plans"
    )
    plans = ast.literal_eval(assignment.value)
    ab = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert ab["full_allocation_a"] is True
    assert ab["dynamic_ab_tensormaps"] is True
    body: list[str] = []
    arguments: list[str] = []
    _append_cute_wrapper_plan(body, arguments, ab, num_sm=148)
    source = ab["lhs_idx"]
    assert f"arg{source}_shape0 * arg{source}_shape1" in "\n".join(body)
    assert "tcgen05_grouped_group_idx * cutlass.Int32(2)" in code
    assert "cute.domain_offset((tcgen05_grouped_global_m_start" in code
    assert code.count(".update_tensormap(") == 1
    descriptor_arguments = [
        keyword.value.id
        for node in ast.walk(parsed)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "tma_desc_ptr" and isinstance(keyword.value, ast.Name)
    ]
    assert descriptor_arguments
    assert all("_d_tensormap_desc_ptr" in name for name in descriptor_arguments)


def test_global_weights_keeps_feature_tail_on_dynamic_path() -> None:
    args = (
        torch.empty((640, 128), dtype=torch.bfloat16),
        torch.empty((4, 192, 128), dtype=torch.bfloat16),
        torch.tensor([0, 0, 1, 129, 640], dtype=torch.int32),
    )
    with _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped_fp32_output, args)
        config = dict(_config(1).config)
        bound.to_code(helion.Config.from_dict(config))
        config["tcgen05_grouped_full_allocation_a"] = True
        with pytest.raises(exc.BackendUnsupported, match="whole output feature"):
            bound.to_code(helion.Config.from_dict(config))
