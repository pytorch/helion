from __future__ import annotations

import ast
from dataclasses import dataclass
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion.autotuner import BooleanFragment
from helion.autotuner.config_fragment import ConfigSpecFragment
import helion.language as hl
from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only
ANCHOR = BooleanFragment()


@helion.kernel(autotune_effort="none")
def _aliased_compare(x, value: hl.constexpr):
    fragment = value
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        if fragment == ANCHOR:
            out[tile] = x[tile] + 2.0
        else:
            out[tile] = x[tile] + 5.0
    return out


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_host_alias_retains_argument_comparison_guard(wrapped):
    kernel = helion.kernel(_aliased_compare.fn, autotune_effort="none")
    tensor = torch.empty(17)

    def args(value):
        return tensor, hl.constexpr(value) if wrapped else value

    def observe(bound, *values):
        bound.to_code()
        bound._run = lambda *unused: bound
        return bound._run(*values)

    with cpu_render_target(), patch.object(BoundKernel, "__call__", observe):
        first = kernel(*args(ANCHOR))
        other = args(BooleanFragment())
        second = kernel(*other)
        fresh = helion.kernel(_aliased_compare.fn, autotune_effort="none")
        assert second.to_code() == fresh.bind(other).to_code()
        assert first.to_code() != second.to_code()
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            assert kernel(*args(ANCHOR)) is first
            assert kernel(*args(BooleanFragment())) is second


@dataclass(frozen=True)
class _ValueFragment(ConfigSpecFragment):
    value: int = 7

    def default(self):
        return self.value


@helion.kernel(autotune_effort="none")
def _return_alias(x, value: hl.constexpr):
    fragment = value
    for tile in hl.tile(x.numel()):
        x[tile] = x[tile] + 1.0
    return fragment


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_host_alias_returns_original_custom_fragment(wrapped):
    fragment = _ValueFragment()
    value = hl.constexpr(fragment) if wrapped else fragment
    tensor = torch.empty(1)
    with cpu_render_target():
        kernel = helion.kernel(_return_alias.fn, autotune_effort="none")
        bound = kernel.bind((tensor, value))
        source = ast.parse(bound.to_code())
    host = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "_return_alias"
    )
    launcher = Mock()
    namespace = {"torch": torch, "hl": hl}
    for node in source.body:
        if isinstance(node, ast.FunctionDef) and node is not host:
            namespace[node.name] = object()
    for default in host.args.kw_defaults:
        if isinstance(default, ast.Name):
            namespace[default.id] = launcher
    module = ast.Module(
        [ast.ImportFrom("__future__", [ast.alias("annotations")], 0), host], []
    )
    # Run actual generated host code, replacing only the native launch.
    exec(
        compile(ast.fix_missing_locations(module), "<fragment_alias>", "exec"),
        namespace,
    )
    assert namespace[host.name](tensor, value) is fragment
    assert launcher.call_count == 1


@helion.kernel(autotune_effort="none")
def _return_container_alias(x, values: hl.constexpr, nested: hl.constexpr):
    sequence = values
    if nested:
        fragment = sequence[0][0]
    else:
        fragment = sequence[0]
    for tile in hl.tile(x.numel()):
        x[tile] = x[tile] + 1.0
    return fragment, sequence


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize(
    "nested,wrapped,custom",
    [
        (False, False, False),
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ],
)
def test_host_container_alias_returns_original_fragment_and_container(
    nested, wrapped, custom
):
    fragment = _ValueFragment() if custom else BooleanFragment()
    container = ((fragment,),) if nested else (fragment,)
    value = hl.constexpr(container) if wrapped else container
    tensor = torch.empty(1)
    with cpu_render_target():
        kernel = helion.kernel(_return_container_alias.fn, autotune_effort="none")
        bound = kernel.bind((tensor, value, nested))
        source = ast.parse(bound.to_code())
    host = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "_return_container_alias"
    )
    launcher = Mock()
    namespace = {"torch": torch}
    imports = [
        node
        for node in source.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and any(
            alias.name == "helion.language" and alias.asname == "hl"
            for alias in node.names
        )
    ]
    assert len(imports) == 1
    for node in source.body:
        if isinstance(node, ast.FunctionDef) and node is not host:
            namespace[node.name] = object()
    for default in host.args.kw_defaults:
        if isinstance(default, ast.Name):
            namespace[default.id] = launcher
    module = ast.Module(
        [ast.ImportFrom("__future__", [ast.alias("annotations")], 0), *imports, host],
        [],
    )
    exec(
        compile(ast.fix_missing_locations(module), "<container_alias>", "exec"),
        namespace,
    )
    actual_fragment, actual_container = namespace[host.name](tensor, value, nested)
    assert actual_fragment is fragment
    assert actual_container is container
    assert launcher.call_count == 1
