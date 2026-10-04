from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_pipeline import _config
import helion
from helion._compiler.cute import chained_prepared_operands
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _tail_domain_sequence(a, b, c, initial):
    rows, reduction = a.shape
    steps, columns = c.shape
    history = torch.empty(
        ((steps + 31) // 32, rows, columns), device=a.device, dtype=torch.float32
    )
    final = torch.empty_like(initial)
    for rr, cc in hl.tile([rows, columns], block_size=[32, 128]):
        state = initial[rr, cc]
        for time in hl.tile(steps, block_size=32):
            kk = hl.arange(reduction)
            product = hl.dot(a[rr, kk], b[kk, time])
            prepared = torch.reciprocal(product).to(a.dtype)
            state = hl.dot(prepared, c[time, cc], acc=state)
            history[time.id, rr, cc] = state
        final[rr, cc] = state
    return history, final


def _inputs(device, dtype, steps=5):
    torch.manual_seed(9472)
    # Positive valid products are finite and nonzero. Padded products are zero,
    # so their reciprocals expose a dropped reduction-axis mask as 0 * inf.
    return (
        *(
            torch.rand(shape, device=device, dtype=dtype) + 0.25
            for shape in (
                (32, 16),
                (16, steps),
                (steps, 128),
            )
        ),
        torch.randn((32, 128), device=device) * 0.125,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_complete_native_image_with_inherited_tail_keeps_original_b_mask_cpu(dtype):
    args = _inputs("cpu", dtype)
    config = _config(16, pipeline=True)
    original = chained_prepared_operands.plan_prepared_operands
    candidates = []

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None
        candidates.extend(result)
        return result

    with _cpu_codegen():
        bound = _tail_domain_sequence._bind_isolated(args)
        with (
            bound.env.use_runtime_arg_values(
                _runtime_values(_tail_domain_sequence, args)
            ),
            patch.object(chained_prepared_operands, "plan_prepared_operands", observe),
        ):
            source = bound.to_code(config)
    # The byte/shape proof admits this complete image. Only the emission-time
    # domain proof can see that its original B producer must mask the time tail.
    assert candidates
    assert all(
        item.stage == 1 and item.physical_shape == (32, 32) for item in candidates
    )
    name = candidates[0].buffer.name
    assert f"chain_1_b = {name}" not in source
    assert "chain_1_b_ptr = chain_b_workspace" in source
    assert "chain_1_b_1_step" in source
    assert any(
        isinstance(node, ast.IfExp)
        and {"chain_loop_index", "chain_loop_end"}
        <= {item.id for item in ast.walk(node.test) if isinstance(item, ast.Name)}
        for assignment in ast.walk(ast.parse(source))
        if isinstance(assignment, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "chain_1_b"
            for target in assignment.targets
        )
        for node in ast.walk(assignment.value)
    )
    # Do not instead zero the exact frontier: other users may require its raw
    # padded reciprocal. Keep its ordinary logical view and mask only B filling.
    assert f"{name}_layout = cute.tile_to_shape(" not in source
    assert f"{name} = cute.make_tensor(" in source


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize(
    "dtype,steps", [(torch.bfloat16, 5), (torch.float16, 5), (torch.bfloat16, 97)]
)
def test_prepared_operand_domain_gpu_matches_ordinary_fill_without_tail_nan(
    dtype, steps
):
    args = _inputs(DEVICE, dtype, steps)
    saved = tuple(value.clone() for value in args)
    config = _config(16, pipeline=True)
    ordinary_bound = _tail_domain_sequence._bind_isolated(args)
    with (
        ordinary_bound.env.use_runtime_arg_values(
            _runtime_values(_tail_domain_sequence, args)
        ),
        patch.object(
            chained_prepared_operands, "plan_prepared_operands", return_value=()
        ),
    ):
        ordinary = ordinary_bound.compile_config(config)
    candidate_bound = _tail_domain_sequence._bind_isolated(args)
    with candidate_bound.env.use_runtime_arg_values(
        _runtime_values(_tail_domain_sequence, args)
    ):
        candidate = candidate_bound.compile_config(config)
    expected = ordinary(*args)
    assert all(torch.isfinite(value).all() for value in expected)
    actual = candidate(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(candidate(*args), actual, atol=0, rtol=0)
    torch.testing.assert_close(args, saved, atol=0, rtol=0)
