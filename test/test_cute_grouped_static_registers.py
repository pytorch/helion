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
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_WARP_REGISTER_REALLOCATION_CONFIG_KEY as REGISTER_REALLOCATION,
)


@pytest.mark.parametrize("multiplier", [1, 2])
def test_grouped_static_registers_preserve_pipeline_lifecycle(multiplier: int) -> None:
    args = (
        torch.empty((640, 128), dtype=torch.bfloat16),
        torch.empty((4, 256, 128), dtype=torch.bfloat16),
        torch.tensor([0, 0, 1, 129, 640], dtype=torch.int32),
    )
    with _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped_fp32_output, args)
        original = dict(_config(multiplier).config)
        default_code = bound.to_code(helion.Config.from_dict(original))
        original[REGISTER_REALLOCATION] = False
        config = bound.config_spec.normalized_config(helion.Config.from_dict(original))
        assert config[REGISTER_REALLOCATION] is False
        static_code = bound.to_code(config)
    ast.parse(static_code)
    assert "cute.arch.setmaxregister_decrease(120)" in default_code
    assert "cute.arch.setmaxregister_increase(240)" in default_code
    assert "setmaxregister_" not in static_code
    # A register allocation choice must preserve every MMA and synchronization
    # operation, including one TMEM allocation and its complete drain/free.
    default_tree = ast.parse(default_code)
    static_tree = ast.parse(static_code)
    for node in ast.walk(default_tree):
        if isinstance(node, ast.If) and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr.startswith("setmaxregister_")
            for statement in node.body
            for call in ast.walk(statement)
        ):
            node.body = [ast.Pass()]
    # Remove the empty register-role predicates before comparing device bodies.
    for tree in (default_tree, static_tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_"):
                node.body = [
                    statement
                    for statement in node.body
                    if not (
                        isinstance(statement, ast.If)
                        and len(statement.body) == 1
                        and isinstance(statement.body[0], ast.Pass)
                    )
                ]
    assert ast.dump(default_tree) == ast.dump(static_tree)
    assert static_code.count(".allocate(") == 1
    assert static_code.count(".relinquish_alloc_permit()") == 1
    assert static_code.count(".free(") == 1


@pytest.mark.parametrize(
    "change",
    [
        {"tcgen05_cluster_m": 2},
        {"tcgen05_cluster_n": 2},
        {"tcgen05_grouped_mode": "worklist"},
        {"tcgen05_grouped_runtime_direct": True},
        {"tcgen05_grouped_static_reserved_sms": 1},
        {"tcgen05_grouped_full_coverage": "dense"},
        {"tcgen05_grouped_worklist_mma_m": 64},
    ],
)
def test_static_registers_keep_independent_worklist_contract(change: dict) -> None:
    config = dict(_config(2).config)
    config.update(change)
    assert not CuteTcgen05Config._grouped_static_registers_supported(config)


def test_static_registers_reject_conflicting_consumer_ceiling() -> None:
    args = (
        torch.empty((640, 128), dtype=torch.bfloat16),
        torch.empty((4, 256, 128), dtype=torch.bfloat16),
        torch.tensor([0, 0, 1, 129, 640], dtype=torch.int32),
    )
    with _target(), _mock_cuda_unavailable():
        bound = _cpu_bind(_grouped_fp32_output, args)
        config = dict(_config(2).config)
        config[REGISTER_REALLOCATION] = False
        config["tcgen05_consumer_regs"] = 240
        with pytest.raises(exc.InvalidConfig, match="requires warp register"):
            bound.config_spec.normalized_config(helion.Config.from_dict(config))
