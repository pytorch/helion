from __future__ import annotations

import ast

import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_pipeline import _config
from .test_cute_chained_register_binding import _polynomial_loop
from helion import Config
from helion import exc
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends


def _inputs(device, dtype, steps, *, keep_entry=True):
    torch.manual_seed(9362)
    return (
        torch.randn((steps, 16, 16), device=device, dtype=dtype) * 0.125,
        torch.randn((steps, 16, 16), device=device, dtype=dtype) * 0.125,
        torch.randn((16, 16), device=device, dtype=torch.float32) * 0.125,
        keep_entry,
    )


def _island_config(cohorts: int, enabled: bool) -> Config:
    config = _config(16, pipeline=True, consumer_warps=4)
    config.config.update(
        cute_chained_preparation_cohorts=cohorts,
        cute_chained_preparation_unroll=1,
        cute_chained_register_islands=enabled,
    )
    return config


def _check_source(source: str, cohorts: int, enabled: bool) -> None:
    assert ("chain_cohort =" in source) is (cohorts == 3)
    tree = ast.parse(source)
    islands = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "chain_prep_thread < 32"
        and "chain_register_island" in ast.unparse(node)
    ]
    assert len(islands) == int(enabled)
    if enabled:
        body = ast.unparse(islands[0])
        assert body.count("cute.gemm(") == 3
        assert "chain_prep_barrier.arrive_and_wait()" not in body
        assert "cutlass.Float32" in body
    else:
        assert "chain_register_island" not in source


@pytest.mark.parametrize("cohorts", [1, 3])
@pytest.mark.parametrize("steps", [0, 1, 4, 7])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_register_island_public_polynomial_preflight_cpu(dtype, steps, cohorts):
    args = _inputs("cpu", dtype, steps)
    off, on = (_island_config(cohorts, enabled) for enabled in (False, True))
    assert {key for key in off.config if off.config[key] != on.config[key]} == {
        "cute_chained_register_islands"
    }
    for enabled, config in ((False, off), (True, on)):
        source = _source(_polynomial_loop, args, config)
        _check_source(source, cohorts, enabled)


@pytest.mark.parametrize("cohorts", [1, 3])
def test_register_island_public_alias_rejects_early_zero_cpu(cohorts):
    args = _inputs("cpu", torch.bfloat16, 4, keep_entry=False)
    source = _source(_polynomial_loop, args, _island_config(cohorts, False))
    _check_source(source, cohorts, False)
    with pytest.raises(exc.BackendUnsupported, match="register"):
        _source(_polynomial_loop, args, _island_config(cohorts, True))


def _assert_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    integer = torch.int32 if actual.dtype == torch.float32 else torch.int16
    assert torch.equal(
        actual.contiguous().view(integer), expected.contiguous().view(integer)
    )


def _assert_outputs(actual, expected, steps: int) -> None:
    assert len(actual) == len(expected) == 2
    assert actual[0].shape == (steps, 16, 16)
    assert actual[1].shape == (16, 16)
    for value, reference in zip(actual, expected, strict=True):
        assert value.dtype == torch.float32
        _assert_bits(value, reference)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("cohorts", [1, 3])
@pytest.mark.parametrize("steps", [0, 1, 4, 7])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_register_island_polynomial_bitwise_replay_and_inputs_gpu(
    dtype, steps, cohorts
):
    args = _inputs(DEVICE, dtype, steps)
    inputs = args[:3]
    saved = tuple(value.clone() for value in inputs)
    compiled = []
    for enabled in (False, True):
        config = _island_config(cohorts, enabled)
        bound = _polynomial_loop._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_polynomial_loop, args)):
            _check_source(bound.to_code(config), cohorts, enabled)
            compiled.append(bound.compile_config(config))
    assert compiled[0] is not compiled[1]
    expected = tuple(value.clone() for value in compiled[0](*args))
    for implementation in compiled:
        actual = implementation(*args)
        _assert_outputs(actual, expected, steps)
        _assert_outputs(implementation(*args), expected, steps)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = implementation(*args)
        for _ in range(2):
            graph.replay()
            _assert_outputs(captured, expected, steps)
        for value, original in zip(inputs, saved, strict=True):
            _assert_bits(value, original)
    if steps == 0:
        _assert_bits(expected[1], saved[2])
