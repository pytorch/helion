from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_carry_transport import _resident_carry_sequence
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_pipeline import _config
import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transposed_carry_sequence(a, b, c, d, initial):
    steps, rows, reduction = a.shape
    width = initial.shape[-1]
    history = torch.empty((steps, rows, width), device=a.device)
    final = torch.empty_like(initial)
    for rr, nn in hl.tile([rows, width], block_size=[32, 128]):
        state = initial[rr, nn]
        for step in hl.tile(steps, block_size=1):
            kk = hl.arange(reduction)
            prepared = hl.dot(c[step.id, rr, kk], d[step.id, kk, rr]).to(a.dtype)
            projected = hl.dot(prepared, state.to(a.dtype), out_dtype=torch.float32)
            state = hl.dot(a[step.id, rr, kk], b[step.id, kk, nn], acc=state * 0.5)
            history[step.id, rr, nn] = projected
        final[rr, nn] = state
    return history, final


def _inputs(device, dtype, steps, transposed):
    torch.manual_seed(9751)
    rows, columns = (32, 128) if transposed else (129, 32)
    return (
        *(
            torch.randn(shape, device=device, dtype=dtype) * 0.125
            for shape in (
                (steps, rows, 16),
                (steps, 16, columns),
                (steps, 32, 16),
                (steps, 16, 32),
            )
        ),
        torch.randn((rows, columns), device=device) * 0.125,
    )


@pytest.mark.parametrize("transposed", [False, True])
def test_carry_physical_orientation_and_ragged_snapshot_domain_cpu(transposed):
    kernel = _transposed_carry_sequence if transposed else _resident_carry_sequence
    source = _source(
        kernel,
        _inputs("cpu", torch.bfloat16, 3, transposed),
        _config(16, pipeline=True, consumer_warps=8),
    )
    assert "_snapshot_values" in source
    assert "chain_2_seed_2_load" in source
    assert "chain_2_c[" not in source
    if transposed:
        assert "chain_1_a_ptr" not in source
    else:
        snapshot_line = next(
            line
            for line in source.splitlines()
            if "_snapshot_values[" in line and " = " in line
        )
        assert "129" in snapshot_line and "else cutlass.BFloat16(0)" in snapshot_line


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("steps", [0, 5])
def test_carry_gpu_transposed_and_ragged_tiles_preserve_exact_state(
    transposed, dtype, steps
):
    kernel = _transposed_carry_sequence if transposed else _resident_carry_sequence
    args = _inputs(DEVICE, dtype, steps, transposed)
    saved = tuple(value.clone() for value in args)
    config = _config(16, pipeline=True, consumer_warps=8)
    bound = kernel._bind_isolated(args)
    with (
        bound.env.use_runtime_arg_values(_runtime_values(kernel, args)),
        patch(
            "helion._compiler.cute.chained_loop_tmem_carry.plan_loop_tmem_carry",
            return_value=None,
        ),
    ):
        ordinary = bound.compile_config(config)
    resident_bound = kernel._bind_isolated(args)
    with resident_bound.env.use_runtime_arg_values(_runtime_values(kernel, args)):
        assert "_snapshot_values" in resident_bound.to_code(config)
        resident = resident_bound.compile_config(config)
    actual = resident(*args)
    torch.testing.assert_close(actual, ordinary(*args), atol=0, rtol=0)
    torch.testing.assert_close(resident(*args), actual, atol=0, rtol=0)
    torch.testing.assert_close(args, saved, atol=0, rtol=0)
