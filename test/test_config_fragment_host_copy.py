from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.tunable_fragments import check_host_selected_value
from helion.autotuner import BooleanFragment
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(autotune_effort="none")
def host_fragment_copy(x, fragment: hl.constexpr, container: hl.constexpr):
    copied = fragment
    if container:
        selected = copied[0]
    else:
        selected = copied
    value = hl.register_tunable("value", selected)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + value
    return out


@pytest.mark.parametrize("container", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
@onlyBackends(["triton", "cute"])
def test_host_fragment_copy_keeps_executable_argument(container, wrapped):
    fragment = BooleanFragment()
    argument = (fragment,) if container else fragment
    if wrapped:
        argument = hl.constexpr(argument)
    args = (torch.empty(17), argument, container)
    kernel = helion.kernel(host_fragment_copy.fn, autotune_effort="none")
    with cpu_render_target():
        bound = kernel.bind(args)
        assert type(bound.config_spec.user_defined_tunables["value"]) is BooleanFragment
        source = bound.to_code(bound.config_spec.default_config())
        compile(source, "<fragment_host_copy>", "exec")
        check_host_selected_value(bound, args, (False, True))
        # Any generated hl.constexpr read must carry its ordinary import;
        # the launch observer's namespace does not establish that property.
        parsed = ast.parse(source)
        if any(isinstance(n, ast.Name) and n.id == "hl" for n in ast.walk(parsed)):
            assert any(
                isinstance(n, ast.Import)
                and any(
                    a.name == "helion.language" and a.asname == "hl" for a in n.names
                )
                for n in parsed.body
            )
