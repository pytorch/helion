from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_full_slice_matmul import _bind

from helion._compiler.ast_read_writes import dead_assignment_elimination
from helion._compiler.cute.memory_ops import _pure_epilogue_ancestors
from helion._compiler.cute.mma_support import CuteMmaSupport
from helion._compiler.generate_ast import GenerateAST
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import memory_ops

if TYPE_CHECKING:
    from collections.abc import Generator

    from helion._compiler.device_function import DeviceFunction


@pytest.fixture(autouse=True)
def _native_cpu_target() -> Generator[None, None, None]:
    support = CuteMmaSupport(
        universal=True, warp_f16bf16=True, warpgroup_f16bf16=True, tcgen05_f16bf16=True
    )
    with (
        _target(),
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        patch(
            "helion._compiler.cute.mma_support.get_cute_mma_support",
            return_value=support,
        ),
        patch(
            "helion._compiler.cute.cute_mma.get_cute_mma_support", return_value=support
        ),
        patch(
            "helion._compiler.cute.tcgen05_config.CuteTcgen05Config.per_cta_smem_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


def _max_calls(source: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "cute.math.max"
    ]


def _rounded_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((a.size(0), b.size(1)), device=a.device, dtype=a.dtype)
    for row, column in hl.tile((a.size(0), b.size(1))):
        product = a[row, :] @ b[:, column]
        out[row, column] = torch.relu(product)
    return out


def _inputs(
    shape: tuple[int, int, int], dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n, k = shape
    return torch.empty((m, n), dtype=dtype), torch.empty((n, k), dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(128, 256, 64), (73, 88, 56)])
@skipUnlessBackends(["cute"])
def test_full_slice_rounded_relu_only_emits_the_fused_expression(
    dtype: torch.dtype, shape: tuple[int, int, int]
) -> None:
    bound = _bind(_rounded_relu, _inputs(shape, dtype))
    source = bound.to_code(bound.config_spec.default_config())
    assert "cute.gemm(" in source and "tcgen05_" in source
    # The real ReLU is evaluated on the rounded TMEM value. Its old scalar
    # duplicate combined the FP32 accumulator placeholder with a 16-bit zero,
    # causing a CuTe type error even though nothing read its result.
    assert not _max_calls(source)
    relu_steps = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id.startswith("tcgen05_chain_step")
            for target in node.targets
        )
        and any(
            isinstance(child, ast.Compare)
            and any(isinstance(op, ast.Gt) for op in child.ops)
            for child in ast.walk(node.value)
        )
    ]
    assert relu_steps
    assert all("_helion_fullslice_acc" not in ast.unparse(node) for node in relu_steps)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@skipUnlessBackends(["cute"])
def test_unfused_scalar_relu_remains_live(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HELION_CUTE_MMA_IMPL", "universal")
    bound = _bind(_rounded_relu, _inputs((128, 256, 64), dtype))
    first = bound.to_code(bound.config_spec.default_config())
    assert "tcgen05_" not in first
    assert _max_calls(first)


def test_epilogue_dce_keeps_shared_cse_values_and_unowned_effects() -> None:
    graph = torch.fx.Graph()
    tensor = graph.placeholder("tensor")
    rounded = graph.call_function(
        torch.ops.prims.convert_element_type.default, (tensor, torch.bfloat16)
    )
    relu = graph.call_function(torch.ops.aten.relu.default, (rounded,))
    body = cast(
        "list[ast.AST]",
        ast.parse(
            "shared = cutlass.BFloat16(tensor)\n"
            "dead = cute.math.max(shared, cutlass.BFloat16(0))\n"
            "unowned = effectful()\n"
            "output.store(shared)\n"
        ).body,
    )
    cg = object.__new__(GenerateAST)
    cg.device_function = cast("DeviceFunction", SimpleNamespace(dce_vars=[]))
    cg._statements_by_owner_node_id = {
        id(rounded): [(body, body[0])],
        id(relu): [(body, body[1])],
    }
    cg.allow_dead_assignments_owned_by_nodes(_pure_epilogue_ancestors(relu))
    dead_assignment_elimination(body, cg.device_function.dce_vars)
    remaining = ast.unparse(
        ast.Module(body=cast("list[ast.stmt]", body), type_ignores=[])
    )
    assert "dead =" not in remaining
    assert "shared = cutlass.BFloat16(tensor)" in remaining
    assert "unowned = effectful()" in remaining
    assert "output.store(shared)" in remaining


def test_epilogue_dce_stops_at_memory_mutation_and_random_operations() -> None:
    graph = torch.fx.Graph()
    tensor = graph.placeholder("tensor")
    loaded = graph.call_function(memory_ops.load, (tensor, (slice(None),)))
    mutated = graph.call_function(torch.ops.aten.add_.Tensor, (loaded, 1))
    random = graph.call_function(torch.ops.aten.rand_like.default, (mutated,))
    for leaf in (loaded, mutated, random):
        relu = graph.call_function(torch.ops.aten.relu.default, (leaf,))
        assert _pure_epilogue_ancestors(relu) == (relu,)
