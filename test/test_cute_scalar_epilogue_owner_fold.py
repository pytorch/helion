from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from ._cute_binding import _cpu_bind
from ._cute_binding import _forbid_native_compile
from ._cute_binding import _mock_cuda_unavailable
from .cute_population_contracts import _target
from .test_cute_rolled_online_epilogue import _config as row_config
from .test_cute_rolled_online_epilogue import _program
from .test_cute_rolled_online_epilogue import row_logsumexp
import helion
from helion import exc
from helion._compiler.cute import scalar_epilogue


def _config(owner: bool | None = True, *, rows: int = 1) -> helion.Config:
    config = row_config(rows=rows, threads=256, cluster=8)
    config.config.update(
        cute_lane_layouts=["strided", "blocked"],
        cute_register_row=True,
        cute_register_row_tensor_copy=True,
        cute_warp_fp32_redux=True,
    )
    if owner is not None:
        config.config["cute_scalar_epilogue_owner_fold"] = owner
    return config


def _pair_call(device: ast.FunctionDef) -> ast.Call:
    return next(
        node
        for node in ast.walk(device)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "_cute_grouped_reduce_cluster_warp_online_pair"
    )


def test_proved_final_result_gets_owner_keyword() -> None:
    code, device = _program(columns=128256, config=_config())
    call = _pair_call(device)
    assert any(
        keyword.arg == "owner_only" and ast.unparse(keyword.value) == "True"
        for keyword in call.keywords
    )
    assert code.count("_cute_complete_row_tensor_copy(") == 1
    assert "group_span=256, cluster_n=8" in code


def test_default_and_false_preserve_original_kernel_code() -> None:
    missing, _ = _program(columns=128256, config=_config(None))
    disabled, _ = _program(columns=128256, config=_config(False))
    assert missing == disabled
    assert "owner_only=" not in missing


def test_multirow_mapping_retains_all_receivers() -> None:
    code, _ = _program(columns=32000, config=_config(rows=2))
    assert "owner_only=" not in code


@pytest.mark.parametrize(
    "mutation", ["prefix_write", "thread_value", "later_collective"]
)
def test_unproved_scalar_suffix_retains_all_receivers(mutation: str) -> None:
    original = scalar_epilogue.lower_scalar_epilogue

    def change(body, *args):
        if mutation == "prefix_write":
            body.insert(
                0, ast.parse("(x.iterator + 0).store(cutlass.BFloat16(1))").body[0]
            )
        elif mutation == "thread_value":
            body.append(ast.parse("varying = cute.arch.thread_idx()[0]").body[0])
        else:
            body.append(
                ast.parse(
                    "later = cute.arch.warp_reduction_sum(cutlass.Float32(1))"
                ).body[0]
            )
        return original(body, *args)

    with patch.object(scalar_epilogue, "lower_scalar_epilogue", change):
        code, _ = _program(columns=128256, config=_config())
    assert "owner_only=" not in code


def test_flag_requires_scalar_epilogue_proof_control() -> None:
    x = torch.empty((3, 32000), dtype=torch.bfloat16)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(row_logsumexp, (x,))
        config = _config()
        config.config["cute_scalar_epilogue"] = False
        with pytest.raises(exc.InvalidConfig, match="requires cute_scalar_epilogue"):
            bound.config_spec.normalized_config(config)
        repaired = helion.Config.from_dict(config.config)
        bound.config_spec.normalize(repaired, _fix_invalid=True)
        assert repaired["cute_scalar_epilogue_owner_fold"] is False


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_flag_is_boolean(value) -> None:
    x = torch.empty((3, 32000), dtype=torch.bfloat16)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(row_logsumexp, (x,))
        config = _config()
        config.config["cute_scalar_epilogue_owner_fold"] = value
        with pytest.raises(exc.InvalidConfig, match="must be a boolean"):
            bound.config_spec.normalized_config(config)


def test_helper_communication_completes_before_owner_branch() -> None:
    source = Path(scalar_epilogue.__file__).with_name("reduce_helpers.py")
    module = ast.parse(source.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_cute_grouped_reduce_cluster_warp_online_pair_body"
    )
    wait = next(
        index
        for index, node in enumerate(function.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "cute.arch.mbarrier_wait"
    )
    fold = next(
        index
        for index, node in enumerate(function.body)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "fold_active"
    )
    assert wait < fold
    assert any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "store_shared_remote_f32x2"
        for statement in function.body[:wait]
        for node in ast.walk(statement)
    )
    initialized = {
        ast.unparse(node.targets[0])
        for node in function.body[wait + 1 : fold]
        if isinstance(node, ast.Assign)
    }
    assert {"group_max", "group_sum"} <= initialized
    assert isinstance(function.body[-1], ast.Return)
