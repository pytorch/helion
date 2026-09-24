from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_scan_cache import _cpu_code
from .test_cute_chained_scan_cache import _inputs
from .test_cute_chained_scan_cache import _scan_cached_chain
import helion
from helion import exc
from helion._compiler.cute import chained_matmul
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.device_ir import GraphInfo

pytestmark = skipUnlessBackends(["cute"])
_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_NAMES = {
    torch.bfloat16: "cutlass.BFloat16",
    torch.float16: "cutlass.Float16",
    torch.float32: "cutlass.Float32",
}


def _typed_inputs(
    device: str | torch.device, length: int, dtype: torch.dtype
) -> tuple[torch.Tensor, ...]:
    args = _inputs(device, length, 2)
    # Keep independently typed sources and noncontiguous physical strides.
    delta = args[3].to(dtype).repeat_interleave(2, dim=1)[:, ::2]
    other = args[4].to(torch.float32).repeat_interleave(2, dim=1)[:, ::2]
    return (*args[:3], delta, other, args[5])


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("length", [49, 64])
def test_scan_cache_original_dtype_codegen(dtype: torch.dtype, length: int) -> None:
    code = _cpu_code(_typed_inputs("cpu", length, dtype), "multiple")
    tree = ast.parse(code)
    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    caches = {
        name: value
        for name, value in assignments.items()
        if name.startswith("chain_scan_0_input_")
    }
    assert len(caches) == 2
    cache_dtypes = []
    for name, tensor in caches.items():
        assert isinstance(tensor, ast.Call)
        allocation = tensor.args[0]
        assert isinstance(allocation, ast.Call)
        assert ast.unparse(allocation.func) == "cute.arch.alloc_smem"
        cache_dtype = ast.unparse(allocation.args[0])
        cache_dtypes.append(cache_dtype)
        assert ast.literal_eval(allocation.args[1]) == 64
        assert allocation.keywords[0].arg == "alignment"
        assert ast.literal_eval(allocation.keywords[0].value) == 128
        stores = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Subscript) and ast.unparse(target.value) == name
                for target in node.targets
            )
        ]
        assert len(stores) == 1
        stored = stores[0].body if isinstance(stores[0], ast.IfExp) else stores[0]
        assert isinstance(stored, ast.Call)
        assert ast.unparse(stored.func) == cache_dtype
        # The cache stores exactly the raw leaf register, not its FP32 scan
        # conversion or product. This also prevents a duplicate global read.
        raw = stored.args[0]
        assert isinstance(raw, ast.Name)
        loaded = assignments[raw.id]
        assert isinstance(loaded, ast.IfExp)
        assert isinstance(loaded.body, ast.Call)
        assert isinstance(loaded.body.func, ast.Attribute)
        assert loaded.body.func.attr == "load"
        assert isinstance(loaded.orelse, ast.Call)
        assert ast.unparse(loaded.orelse.func) == cache_dtype
    assert sorted(cache_dtypes) == sorted([_NAMES[dtype], "cutlass.Float32"])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_scan_cache_original_dtype_exact_shared_admission(dtype: torch.dtype) -> None:
    initialized_before = torch.cuda.is_initialized()
    args = _typed_inputs("cpu", 49, dtype)
    allocations = (
        2 * (16 * 64 + 8 * 64),
        2 * (64 * 64 + 8 * 64),
        4 * 16 * (64 + 4),
        4 * 16 * (32 + 4),
        4 * (64 + 4),
        64 * args[3].element_size(),
        64 * args[4].element_size(),
    )
    capacity = sum((size + 127) // 128 * 128 for size in allocations)
    original = chained_matmul.plan_chained_matmul
    decisions: list[ChainedMatmulPlan | None] = []

    def observe_plan(graphs: Sequence[GraphInfo]) -> ChainedMatmulPlan | None:
        plan = original(graphs)
        decisions.append(plan)
        return plan

    with patch.object(
        chained_matmul, "plan_chained_matmul", side_effect=observe_plan
    ) as planner:
        code = _cpu_code(args, "multiple", capacity=capacity)
        assert "chain_1_mma" in code
        assert planner.call_count == 1
        assert len(decisions) == 1 and decisions[0] is not None
        decisions.clear()
        planner.reset_mock()
        # A fallback may reject for a different reason after this planner
        # declines the chain. Check the actual admission decision directly.
        with pytest.raises(exc.BackendUnsupported):
            _cpu_code(args, "multiple", capacity=capacity - 1)
        assert planner.call_count == 1
        assert decisions == [None]
    assert torch.cuda.is_initialized() is initialized_before


@pytest.mark.parametrize("dtype", _DTYPES)
def test_scan_cache_original_dtype_runtime(dtype: torch.dtype) -> None:
    args = _typed_inputs(DEVICE, 49, dtype)
    originals = tuple(value.clone() for value in args)
    _scan_cached_chain.reset()
    fn = _scan_cached_chain.bind((*args, "multiple")).compile_config(
        helion.Config(cute_chained_mma_schedule="cp_async_register_reuse_scan")
    )
    actual = fn(*args, "multiple")
    a, b, v, delta, other, permutation = args
    decay = (delta[:, :49].double() * other[:, :49].double()).cumsum(-1)
    weights = (a.double() @ b.double().transpose(-1, -2)) * (
        decay[:, :, None] - decay[:, None, :]
    ).exp()
    weights *= other[:, None, :49].double()
    expected = torch.tril(weights).to(v.dtype).double() @ v.double()
    expected += delta[:, :49, None].double()
    torch.testing.assert_close(actual, expected.to(v.dtype), atol=0.0001, rtol=0.02)
    repeated = fn(*args, "multiple")
    torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
    assert repeated.data_ptr() != actual.data_ptr()
    for value, original in zip(args, originals, strict=True):
        torch.testing.assert_close(value, original, atol=0, rtol=0)
