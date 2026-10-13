from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_pipeline import _config
import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _ready_sequence(a, b, c, initial):
    steps, rows, reduction = a.shape
    columns = c.shape[-1]
    history = torch.empty((steps, rows, columns), device=a.device, dtype=torch.float32)
    final = torch.empty_like(initial)
    for rr, cc in hl.tile([rows, columns], block_size=[32, 128]):
        state = initial[rr, cc]
        for step in hl.tile(steps, block_size=1):
            kk, jj = hl.arange(reduction), hl.arange(32)
            prepared = hl.dot(a[step.id, rr, kk], b[step.id, kk, jj]).to(a.dtype)
            state = hl.dot(prepared, c[step.id, jj, cc], acc=state)
            history[step.id, rr, cc] = state
        final[rr, cc] = state
    return history, final


def _inputs(device, dtype, steps=3):
    torch.manual_seed(9471)
    return (
        *(
            torch.randn(shape, device=device, dtype=dtype) * 0.125
            for shape in ((steps, 32, 16), (steps, 16, 32), (steps, 32, 128))
        ),
        torch.randn((32, 128), device=device) * 0.125,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_frontier_view_is_published_before_async_operand_use_cpu(dtype):
    args = _inputs("cpu", dtype)
    with _cpu_codegen():
        bound = _ready_sequence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_ready_sequence, args)):
            source = bound.to_code(_config(16, pipeline=True))
    assert "chain_1_b = chain_prepared_" in source
    assert "chain_1_b_1_step" not in source
    assert "chain_1_b_ptr" not in source
    roles = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.If) and ast.unparse(node.test) == "chain_thread < 384"
    )
    producer = ast.unparse(ast.Module(body=roles.body, type_ignores=[]))
    assert (
        producer.rindex("cute.arch.fence_view_async_shared()")
        < producer.rindex("chain_prep_barrier.arrive_and_wait()")
        < producer.rindex("chain_sync.arrive_mbarrier(")
    )
    assert "chain_prepared_0_ptr = cute.recast_ptr(chain_frame" in source
    assert "chain_prepared_0_layout.inner" in source


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("steps", [0, 1, 5])
def test_prepared_operand_gpu_matches_ordinary_fill_bitwise(dtype, steps):
    args = _inputs(DEVICE, dtype, steps)
    saved = tuple(value.clone() for value in args)
    config = _config(16, pipeline=True)
    bound = _ready_sequence._bind_isolated(args)
    with (
        bound.env.use_runtime_arg_values(_runtime_values(_ready_sequence, args)),
        patch(
            "helion._compiler.cute.chained_prepared_operands.plan_prepared_operands",
            return_value=(),
        ),
    ):
        ordinary = bound.compile_config(config)
    native_bound = _ready_sequence._bind_isolated(args)
    with native_bound.env.use_runtime_arg_values(
        _runtime_values(_ready_sequence, args)
    ):
        assert "chain_1_b = chain_prepared_" in native_bound.to_code(config)
        native = native_bound.compile_config(config)
    assert native is not ordinary
    expected = ordinary(*args)
    actual = native(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(native(*args), actual, atol=0, rtol=0)
    torch.testing.assert_close(args, saved, atol=0, rtol=0)
