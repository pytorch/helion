from __future__ import annotations

import ast

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_logsumexp(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    for row in hl.tile(x.shape[0]):
        values = x[row, :].float()
        maximum = values.amax(-1)
        denominator = torch.exp(values - maximum[:, None]).sum(-1)
        out[row] = denominator.log() + maximum
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_shifted_other(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    for row in hl.tile(x.shape[0]):
        maximum = x[row, :].float().amax(-1)
        denominator = torch.exp(y[row, :].float() - maximum[:, None]).sum(-1)
        out[row] = denominator.log() + maximum
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def row_early_maximum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    for row in hl.tile(x.shape[0]):
        values = x[row, :].float()
        maximum = values.amax(-1)
        early = maximum - x[row, 0].float()
        denominator = torch.exp(values - maximum[:, None]).sum(-1)
        out[row] = early + denominator.log()
    return out


def _config(rows=1, threads=256, cluster=2, *, online=True, epilogue=True):
    return helion.Config(
        block_sizes=[rows],
        num_threads=[rows, threads],
        reduction_loops=[4096],
        cute_vector_widths=[8, 1],
        cute_reduction_reloads=["register"],
        cute_cluster_n=cluster,
        cute_cluster_warp_online=online,
        cute_scalar_epilogue=epilogue,
    )


def _program(kernel=row_logsumexp, *, rows=3, columns=32000, config=None):
    x = torch.empty((rows, columns), dtype=torch.bfloat16)
    arguments = (x, torch.empty_like(x)) if kernel is row_shifted_other else (x,)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(kernel, arguments)
        code = bound.to_code(config or _config())
    tree = ast.parse(code)
    device = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            ast.unparse(decorator) == "cute.kernel" for decorator in node.decorator_list
        )
    )
    return code, device


@pytest.mark.parametrize("columns", [32000, 65544, 128256, 151936])
def test_warp_pair_preserves_masked_row_extent(columns):
    code, device = _program(columns=columns)
    assert code.count("_cute_grouped_reduce_cluster_warp_online_pair(") == 1
    assert "_cute_grouped_reduce_cluster(" not in code
    assert f"< {columns}" in code
    guards = [node for node in device.body if isinstance(node, ast.If)]
    owner = next(
        node for node in guards if "block_idx_in_cluster" in ast.unparse(node.test)
    )
    assert "== 0" in ast.unparse(owner.test)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("_cute_grouped_reduce")
        for node in ast.walk(owner)
    )


@pytest.mark.parametrize(
    "online,epilogue", [(False, False), (True, False), (False, True)]
)
def test_schedule_choices_are_independent(online, epilogue):
    code, device = _program(config=_config(online=online, epilogue=epilogue))
    assert ("_cute_grouped_reduce_cluster_warp_online_pair(" in code) is online
    assert (
        any(
            isinstance(node, ast.If)
            and "block_idx_in_cluster" in ast.unparse(node.test)
            for node in device.body
        )
        is epilogue
    )


@pytest.mark.parametrize("kernel", [row_shifted_other, row_early_maximum])
def test_warp_pair_rejects_unproved_frame_changes(kernel):
    code, device = _program(kernel)
    assert "_cute_grouped_reduce_cluster_warp_online_pair(" not in code
    assert code.count("_cute_grouped_reduce_cluster(") == 2


def test_multirow_reduction_retains_its_thread_ownership():
    code, device = _program(rows=7, config=_config(rows=4, threads=64))
    assert "_cute_grouped_reduce_cluster_warp_online_pair(" not in code
    assert not any(
        isinstance(node, ast.If) and "block_idx_in_cluster" in ast.unparse(node.test)
        for node in device.body
    )


def test_independent_cached_sums_keep_the_proved_pair():
    config = _config()
    config.config["cute_independent_reduction"] = True
    code, device = _program(config=config)
    assert "_cute_grouped_reduce_cluster_warp_online_pair(" in code
    initializers = [
        statement
        for statement in device.body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id.startswith("_cute_rolled_sum_")
    ]
    assert len(initializers) == 8


@pytest.mark.parametrize("columns,cluster", [(32000, 2), (128256, 8), (151936, 16)])
def test_register_row_uses_one_fp32_tensor_and_two_tensor_reductions(columns, cluster):
    config = _config(cluster=cluster)
    config.config["cute_register_row"] = True
    code, device = _program(columns=columns, config=config)
    assert "_fuse_cache_" not in code
    row = next(
        node.value
        for node in device.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_register_row"
    )
    assert isinstance(row, ast.Call)
    assert ast.unparse(row.func) == "cute.make_rmem_tensor"
    assert isinstance(row.args[0], ast.Constant)
    assert isinstance(row.args[0].value, int)
    assert 2 <= row.args[0].value <= 128
    assert ast.unparse(row.args[1]) == "cutlass.Float32"
    assert ".reduce(cute.ReductionOp.MAX" in code
    assert ".reduce(cute.ReductionOp.ADD" in code
    assert "for roffset_1 in range(" not in code
    assert "_cute_grouped_reduce_cluster_warp_online_pair(" in code


@pytest.mark.parametrize("kernel", [row_shifted_other, row_early_maximum])
def test_register_row_rejects_an_unproved_successive_reduction(kernel):
    config = _config()
    config.config["cute_register_row"] = True
    code, device = _program(kernel, config=config)
    assert "_register_row =" not in code
    assert ".reduce(cute.ReductionOp.MAX" not in code


@pytest.mark.parametrize("native", [False, True])
def test_native_fp32_redux_reaches_both_unconditional_collectives(native):
    config = _config()
    config.config["cute_register_row"] = True
    config.config["cute_warp_fp32_redux"] = native
    code, device = _program(config=config)
    marked = [
        node
        for node in ast.walk(device)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("_cute_grouped_reduce_")
        and any(
            keyword.arg == "fp32_redux"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
    ]
    assert len(marked) == (2 if native else 0)
    guards = [node for node in device.body if isinstance(node, ast.If)]
    owner = next(
        node for node in guards if "block_idx_in_cluster" in ast.unparse(node.test)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("_cute_grouped_reduce_")
        for node in ast.walk(owner)
    )


@skipIfNotCUDA()
@pytest.mark.cuda
@pytest.mark.parametrize("columns", [32000, 65544])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@skipUnlessBackends(["cute"])
def test_warp_pair_and_owner_match_logsumexp(columns, dtype):
    x = torch.randn((3, columns), device=DEVICE, dtype=dtype)
    config = _config()
    if dtype == torch.float32:
        config.config["cute_vector_widths"] = [4, 1]
    code, actual = code_and_output(row_logsumexp, (x,), **config.config)
    torch.testing.assert_close(
        actual, x.double().logsumexp(-1).float(), atol=1e-5, rtol=2e-5
    )
    assert "_cute_grouped_reduce_cluster_warp_online_pair(" in code
