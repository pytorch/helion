from __future__ import annotations

import ast
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_loop_tmem_transport import _inputs
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_pipeline import _config
import helion
from helion._compiler.cute import chained_loop_tmem_transport as transport
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _late_side_input_sequence(a, b, c, d, initial):
    steps, rows, reduction = a.shape
    width = initial.shape[-1]
    history = torch.empty((steps, rows, width), device=a.device)
    side_history = torch.empty_like(history)
    final = torch.empty_like(initial)
    for rr in hl.tile(rows, block_size=128):
        state = initial[rr, :]
        for step in hl.tile(steps, block_size=1):
            kk, jj = hl.arange(reduction), hl.arange(width)
            prepared = hl.dot(c[step.id, jj, kk], d[step.id, kk, jj]).to(a.dtype)
            source = hl.dot(a[step.id, rr, kk], b[step.id, kk, jj], acc=state)
            side = hl.dot(a[step.id, rr, kk], b[step.id, kk, jj], acc=state * 2)
            side_history[step.id, rr, jj] = side
            image = (source + side).to(a.dtype)
            state = hl.dot(image, prepared, out_dtype=torch.float32)
            history[step.id, rr, jj] = state
        final[rr, :] = state
    return history, final, side_history


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _masked_packed_sequence(b, c, d, initial):
    rows, width = initial.shape
    reduction = c.shape[-1]
    steps = b.shape[-1]
    history = torch.empty(((steps + 31) // 32, rows, width), device=b.device)
    final = torch.empty_like(initial)
    for rr in hl.tile(rows, block_size=128):
        state = initial[rr, :]
        for time in hl.tile(steps, block_size=32):
            kk, jj = hl.arange(reduction), hl.arange(width)
            prepared = hl.dot(c[time, kk], d[kk, jj]).to(b.dtype)
            source = hl.dot(state.to(b.dtype), b[jj, time])
            # Explicit intermediate narrowing must survive eager packing. A
            # padded zero source has an infinite reciprocal before its mask.
            image = (
                torch.reciprocal(source).to(torch.float16).to(torch.float32).to(b.dtype)
            )
            state = hl.dot(image, prepared, out_dtype=torch.float32)
            history[time.id, rr, jj] = state
        final[rr, :] = state
    return history, final


def _observe(kernel, args):
    candidates, selected = [], []
    original_candidates = transport.plan_loop_tmem_bridges
    original_prepare = transport.prepare_loop_tmem_transports

    def observe_candidates(*args):
        result = original_candidates(*args)
        assert result is not None
        candidates.extend(result)
        return result

    def observe_prepare(*args):
        result = original_prepare(*args)
        selected.extend(result[0])
        return result

    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        with (
            bound.env.use_runtime_arg_values(_runtime_values(kernel, args)),
            patch.object(transport, "plan_loop_tmem_bridges", observe_candidates),
            patch.object(transport, "prepare_loop_tmem_transports", observe_prepare),
        ):
            source = bound.to_code(_config(16, pipeline=True, consumer_warps=8))
    return source, candidates, selected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_eager_packing_rejects_an_unavailable_recurrent_dot_side_input_cpu(dtype):
    source, candidates, selected = _observe(
        _late_side_input_sequence, _inputs("cpu", dtype)
    )
    # The first source is structurally exclusive to the final A expression,
    # but its later side input is not an available fragment/frontier. Storing
    # the side input excludes a second candidate: duplicate-slot rejection
    # must not accidentally substitute for the readiness guard tested here.
    assert {candidate.source_stage for candidate in candidates} == {1}
    assert all(candidate.destination_group.stages == (3,) for candidate in candidates)
    assert selected == []
    assert "chain_1_result_row" in source
    assert "chain_2_result_row" in source
    assert "chain_3_a_ptr = chain_a_workspace" in source
    assert "chain_tmem_barrier" not in source
    assert "_transport_values" not in source


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,steps", [(129, 32), (128, 5)])
def test_eager_packing_keeps_inherited_domains_and_source_narrowing_cpu(
    dtype, rows, steps
):
    torch.manual_seed(9581)
    args = (
        *(
            torch.rand(shape, dtype=dtype) + 0.25
            for shape in ((32, steps), (steps, 16), (16, 32))
        ),
        torch.ones((rows, 32), dtype=torch.float32),
    )
    source, candidates, selected = _observe(_masked_packed_sequence, args)
    assert len(candidates) == len(selected) == 1
    packed = selected[0]
    assert packed.slot.candidate.source_stage == 1
    assert packed.slot.candidate.destination_group.stages == (2,)
    assert "chain_1_result_row" not in source
    assert "chain_2_a_ptr" not in source
    assignments = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "chain_1_transport_values"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    masked = assignments[0].value
    assert isinstance(masked, ast.IfExp)
    expected = ast.parse(packed.masked_value, mode="eval").body
    assert isinstance(expected, ast.IfExp)
    # Full-function emission can rename scalar SSA temporaries in the value;
    # the domain coordinates and the original mask must remain unchanged.
    assert ast.unparse(masked.test) == ast.unparse(expected.test)
    assert isinstance(masked.body, ast.Call)
    assert ast.unparse(masked.body.func) == packed.dtype
    names = {node.id for node in ast.walk(masked.test) if isinstance(node, ast.Name)}
    if rows == 129:
        assert "chain_1_transport_row" in names
        assert any(name.startswith("chain_origin_") for name in names)
        assert "129" in ast.unparse(masked.test)
    else:
        assert {
            "chain_loop_index",
            "chain_loop_end",
            "chain_1_transport_column",
        } <= names
    assert ast.unparse(masked.orelse).endswith("(0)")
    expression = "\n".join(packed.expression_lines)
    assert "cutlass.Float16(" in expression
    assert "cutlass.Float32(" in expression
    if dtype == torch.bfloat16:
        assert "cutlass.BFloat16(" in expression


def _compare_original(kernel, args, *, packed):
    saved = tuple(value.clone() for value in args)
    config = _config(16, pipeline=True, consumer_warps=8)
    original_bound = kernel._bind_isolated(args)
    with (
        original_bound.env.use_runtime_arg_values(_runtime_values(kernel, args)),
        patch.object(transport, "plan_loop_tmem_bridges", return_value=()),
    ):
        original = original_bound.compile_config(config)
    bound = kernel._bind_isolated(args)
    with bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
        source = bound.to_code(config)
        assert ("_transport_values" in source) is packed
        compiled = bound.compile_config(config)
    actual = compiled(*args)
    assert all(torch.isfinite(value).all() for value in actual)
    torch.testing.assert_close(actual, original(*args), atol=0, rtol=0)
    torch.testing.assert_close(compiled(*args), actual, atol=0, rtol=0)
    torch.testing.assert_close(args, saved, atol=0, rtol=0)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,steps", [(129, 32), (128, 5), (129, 97)])
def test_loop_transport_gpu_masks_padded_infinities_and_preserves_casts(
    dtype, rows, steps
):
    torch.manual_seed(9581)
    args = (
        *(
            torch.rand(shape, device=DEVICE, dtype=dtype) + 0.25
            for shape in ((32, steps), (steps, 16), (16, 32))
        ),
        torch.ones((rows, 32), device=DEVICE, dtype=torch.float32),
    )
    _compare_original(_masked_packed_sequence, args, packed=True)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_loop_transport_gpu_unavailable_side_input_keeps_original_fallback(dtype):
    _compare_original(_late_side_input_sequence, _inputs(DEVICE, dtype), packed=False)
