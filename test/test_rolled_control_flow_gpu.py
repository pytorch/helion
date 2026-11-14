from __future__ import annotations

import math

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing.rolled_control_flow import carried_reduction
from helion._testing.rolled_control_flow import else_only_carried_reduction
from helion._testing.rolled_control_flow import two_axis_carried_reductions


def _input(width: int) -> torch.Tensor:
    # Three loop iterations, two opposite branch outcomes, and a partial
    # final reduction chunk. Integer-valued floats make the oracle exact.
    shape = (3, 2, width)
    return (
        torch.arange(math.prod(shape), device=DEVICE, dtype=torch.float32)
        .remainder(31)
        .sub(15)
        .reshape(shape)
    )


def _expected(x: torch.Tensor, conditional: bool = False) -> torch.Tensor:
    value = x.sum(0)
    if conditional:
        value[1:] = -value[1:]
    return value - value.sum(-1, keepdim=True)


@onlyBackends(["triton", "cute"])
@skipIfRefEager("checks explicit reduction rolling")
def test_else_only_carried_reduction_both_outcomes(tmp_path, record_property):
    x = _input(131)
    bound = else_only_carried_reduction.bind((x,))
    config = bound.config_spec.default_config()
    config.config["reduction_loops"] = [32]
    source = tmp_path / "generated.py"
    source.write_text(bound.to_code(config))
    compiled = bound.compile_config(config)
    raw = tmp_path / "raw.pt"
    record_property("raw_bits", str(raw))
    record_property("generated_source", str(source))
    observations = []
    for delta in (0, 3):
        x.add_(delta)
        before = x.clone()
        value = x[0] * 0
        value[1] += x[:, 1].sum(0)
        expected = value - value.sum(-1, keepdim=True)
        assert torch.count_nonzero(expected[0]).item() == 0
        assert torch.count_nonzero(expected[1]).item() > 0
        actual = compiled(x)
        observations.append(
            {
                "delta": delta,
                "actual": actual.cpu(),
                "expected": expected.cpu(),
                "before": before.cpu(),
                "after": x.cpu(),
            }
        )
        torch.save(observations, raw)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x, before, rtol=0, atol=0)


@onlyBackends(["triton", "cute"])
@skipIfRefEager("checks explicit reduction rolling")
@pytest.mark.parametrize("conditional", (False, True))
def test_loop_carried_values_across_replayed_sweeps(conditional):
    x = _input(131)
    bound = carried_reduction.bind((x, conditional))
    config = bound.config_spec.default_config()
    config.config["reduction_loops"] = [32]
    compiled = bound.compile_config(config)
    for delta in (0, 3):
        x.add_(delta)
        before = x.clone()
        expected = _expected(x, conditional)
        actual = compiled(x, conditional)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x, before, rtol=0, atol=0)


@onlyBackends(["triton", "cute"])
@skipIfRefEager("checks explicit reduction rolling")
def test_independent_axes_reset_loop_carried_values():
    x, y = _input(131), _input(97)
    bound = two_axis_carried_reductions.bind((x, y))
    config = bound.config_spec.default_config()
    config.config["reduction_loops"] = [32, 32]
    compiled = bound.compile_config(config)
    for delta in (0, 3):
        x.add_(delta)
        y.sub_(delta)
        before = (x.clone(), y.clone())
        expected = (_expected(x), _expected(y))
        actual = compiled(x, y)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close((x, y), before, rtol=0, atol=0)
