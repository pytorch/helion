from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import onlyBackends
from helion._testing.host_float_control_flow import aliased_host_float
from helion._testing.host_float_control_flow import rebound_host_float
import helion.language as hl

SCALAR = 1e16


@onlyBackends(["triton", "cute"])
def test_device_float_alias_keeps_its_value_across_loop_updates():
    x = torch.ones((1, 4), device=DEVICE)
    bound = aliased_host_float.bind((x, 1.0))
    config = bound.config_spec.default_config()
    config.config["block_sizes"] = [1, 1, 4]
    bound.set_config(config)
    for alpha in (1.0, -2.0, 1.0):
        torch.testing.assert_close(bound(x, alpha), x * (alpha * 17), rtol=0, atol=0)


@pytest.mark.parametrize("branch", [False, True])
@onlyBackends(["triton", "cute"])
def test_rebound_host_float_resets_for_each_device_iteration(branch):
    x = torch.ones((1, 4, 4), device=DEVICE)
    kernel = helion.kernel(rebound_host_float.fn, autotune_effort="none")
    bound = kernel.bind((x, 1.0, branch))
    config = bound.config_spec.default_config()
    config.config["block_sizes"] = [1, 1, 4] if branch else [1, 1, 4, 4]
    bound.set_config(config)
    compiled = bound.compile_config(config)
    for alpha, fill in ((1.0, 1), (2.0, 1), (-2.0, -1), (1.0, 1)):
        x.fill_(fill)
        original = x.clone()
        factor = 2 if not branch or fill > 0 else 1
        actual = bound(x, alpha, branch)
        torch.testing.assert_close(actual, x * (alpha * factor), rtol=0, atol=0)
        torch.testing.assert_close(x, original, rtol=0, atol=0)
        assert bound.compile_config(config) is compiled


@helion.kernel(autotune_effort="none")
def host_float_outputs(x, runtime, flag):
    host = (SCALAR + 1) - SCALAR
    if flag:
        merged = 1.0
    else:
        merged = host
    out = torch.empty((9, x.numel()), device=x.device, dtype=x.dtype)
    for tile in hl.tile(x.numel()):
        out[0, tile] = x[tile] * host
        out[1, tile] = x[tile] * merged
        out[2, tile] = x[tile] * ((SCALAR + 1) - SCALAR)
        out[3, tile] = x[tile] * ((runtime + 1) - runtime)
        out[4, tile] = x[tile] * ((SCALAR + 1) * runtime - SCALAR)
        out[5, tile] = x[tile] * -(+SCALAR)
        out[6, tile] = x[tile] * (runtime + 1e309)
        out[7, tile] = x[tile] * (runtime + -1e309)
        out[8, tile] = x[tile] * (runtime + (1e309 - 1e309))
    return out


@onlyBackends(["triton", "cute"])
def test_cached_host_float_outputs_follow_python_evaluation():
    backing = torch.ones(34, device=DEVICE, dtype=torch.float32)
    original = backing.clone()
    x = backing[::2]
    kernel = helion.kernel(host_float_outputs.fn, autotune_effort="none")
    bound = kernel.bind((x, 1.0, False))
    config = bound.config_spec.default_config()
    bound.set_config(config)
    compiled = bound.compile_config(config)
    cases = [
        (1e16, 1.0, False),
        (1.0, 1e16, True),
        (-0.0, -0.0, False),
        (0.0, 0.0, True),
        (float("inf"), 1.0, False),
        (1e16, -1e16, True),
        (1e16, 1.0, False),
    ]
    for scalar, runtime, flag in cases:
        with patch.dict(kernel.fn.__globals__, SCALAR=scalar):
            values = [
                (scalar + 1) - scalar,
                1.0 if flag else (scalar + 1) - scalar,
                (scalar + 1) - scalar,
                (runtime + 1) - runtime,
                (scalar + 1) * runtime - scalar,
                -(+scalar),
                runtime + float("inf"),
                runtime - float("inf"),
                runtime + (float("inf") - float("inf")),
            ]
            expected = torch.stack([x * value for value in values])
            actual = bound(x, runtime, flag)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            zero = expected == 0
            assert torch.equal(
                torch.signbit(actual[zero]), torch.signbit(expected[zero])
            )
            assert bound.compile_config(config) is compiled
    torch.testing.assert_close(backing, original, rtol=0, atol=0)
