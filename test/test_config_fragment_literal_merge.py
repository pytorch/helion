from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion.autotuner import EnumFragment
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(autotune_effort="none")
def literal_branch_domain(x, runtime, mode: hl.constexpr, registration: hl.constexpr):
    if mode == "zero":
        if runtime > 0:
            value = 0.0
        else:
            value = -0.0
    elif mode == "int_float":
        if runtime > 0:
            value = 1
        else:
            value = 1.0
    elif mode == "float_int":
        if runtime > 0:
            value = 1.0
        else:
            value = 1
    elif mode == "bool_int":
        if runtime > 0:
            value = True
        else:
            value = 1
    elif mode == "same_float":
        if runtime > 0:
            value = 1.0
        else:
            value = 1.0
    else:
        if runtime > 0:
            value = 1
        else:
            value = 1
    if registration:
        factor = hl.register_tunable("factor", EnumFragment((0.0, value)))
    else:
        factor = value
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] * factor
    return out


@pytest.mark.parametrize("mode", ["zero", "int_float", "float_int", "bool_int"])
def test_distinct_literal_arms_do_not_certify_fragment_domain(mode):
    kernel = helion.kernel(literal_branch_domain.fn, autotune_effort="none")
    with (
        cpu_render_target(),
        pytest.warns(FutureWarning, match="Symbolic config fragment domains"),
    ):
        bound = kernel.bind((torch.empty(17), 1, mode, True))
        choice = bound.config_spec.user_defined_tunables["factor"].choices[1]
        assert isinstance(choice, (torch.SymInt, torch.SymFloat, torch.SymBool))
        assert bound.env.resolve_fragment_global(choice) is None
        config = bound.config_spec.default_config()
        config.config["factor"] = 2.0
        ast.parse(bound.to_code(config))


@pytest.mark.parametrize("mode", ["same_float", "same_int"])
def test_identical_typed_literal_arms_keep_concrete_domain(mode):
    kernel = helion.kernel(literal_branch_domain.fn, autotune_effort="none")
    with cpu_render_target():
        bound = kernel.bind((torch.empty(17), 1, mode, True))
        choice = bound.config_spec.user_defined_tunables["factor"].choices[1]
        assert type(choice) is (float if mode == "same_float" else int)
        assert choice == 1
        ast.parse(bound.to_code(bound.config_spec.default_config()))


@pytest.mark.parametrize(
    "mode", ["zero", "int_float", "float_int", "bool_int", "same_float", "same_int"]
)
def test_ordinary_scalar_branches_remain_renderable(mode):
    kernel = helion.kernel(literal_branch_domain.fn, autotune_effort="none")
    with cpu_render_target():
        bound = kernel.bind((torch.empty(17), 1, mode, False))
        compile(
            bound.to_code(bound.config_spec.default_config()), "<scalar_branch>", "exec"
        )
