from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.tunable_fragments import load_host_function
from helion.autotuner import EnumFragment
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(autotune_effort="none")
def branch_registration(x: torch.Tensor, flag: bool, choices: hl.constexpr):
    if flag:
        first = hl.register_tunable("factor", EnumFragment(choices))
        scale = first + 1
    else:
        second = hl.register_tunable("factor", EnumFragment(choices))
        scale = second + 2
    third = hl.register_tunable("factor", EnumFragment(choices))
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * (scale + third)
    return out


@helion.kernel(autotune_effort="none")
def helper_branch_registration(x: torch.Tensor, flag: bool, choices: hl.constexpr):
    def selected():
        return hl.register_tunable("factor", EnumFragment(choices))

    if flag:
        first = selected()
        scale = first + 1
    else:
        second = selected()
        scale = second + 2
    third = selected()
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * (scale + third)
    return out


@pytest.mark.parametrize("kernel", [branch_registration, helper_branch_registration])
@pytest.mark.parametrize("choices", [(2, 4), (2.5, 4.5)])
@onlyBackends(["triton", "cute"])
def test_branch_local_registration_does_not_escape(kernel, choices):
    x = torch.empty(17)
    with cpu_render_target():
        bound = cpu_bind(kernel, (x, True, choices))
        for selected in choices:
            config = bound.config_spec.default_config()
            config.config["factor"] = selected
            calls = []

            def observe_launch(
                kernel, grid, source, output, *scalars, records=calls, **kwargs
            ):
                assert source is x
                records.append(scalars)

            host = load_host_function(bound, config, observe_launch)
            for flag in (True, False, True):
                calls.clear()
                host(x, flag, choices)
                assert len(calls) == 1
                assert sum(calls[0]) == 2 * selected + (1 if flag else 2)
