from __future__ import annotations

import ast
from types import ModuleType
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

from test.test_cute_epilogue_fanout import _opaque_metadata
from test.test_cute_epilogue_fanout import cuda_trace  # noqa: F401
from test.test_cute_epilogue_fanout import (
    test_typed_host_freshness_rejects_effects_and_rebinding as check_alias_effect,
)

import helion
from helion._compiler.cute.epilogue_fanout import _fresh_returned_tensors
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import register_block_size as choose_block
from helion.language import specialize as specialize_shape

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from helion._compiler.host_function import HostFunction
    from helion.runtime.kernel import BoundKernel


def known_assignments(left: torch.Tensor) -> torch.Tensor:
    shape = hl.specialize(left.shape)
    block = hl.register_block_size(left.size(0))
    result = torch.empty(shape, dtype=left.dtype, device=left.device)
    for row in hl.tile(left.size(0), block_size=block):
        result[row] = left[row]
    return result


def known_aliases(left: torch.Tensor) -> torch.Tensor:
    shape = specialize_shape(left.shape)
    block = choose_block(1, left.size(0))
    result = torch.empty(shape, dtype=left.dtype, device=left.device)
    for row in hl.tile(left.size(0), block_size=block):
        result[row] = left[row]
    return result


def known_expression(left: torch.Tensor) -> torch.Tensor:
    hl.specialize(left.shape)
    result = torch.empty_like(left)
    for row in hl.tile(left.size(0)):
        result[row] = left[row]
    return result


def nested_callback(left: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(left)
    token = hl.specialize(_opaque_metadata(result, left))
    for row in hl.tile(left.size(0) + token):
        result[row] = left[row]
    return result


class _SpecializeProperty(ModuleType):
    reads = 0

    @property
    def specialize(self) -> Callable[..., object]:
        self.reads += 1
        return specialize_shape


class _SpecializeImpostor(ModuleType):
    reads = 0

    def specialize(self, value: object) -> object:
        self.reads += 1
        return value


_property_provider = _SpecializeProperty("_fresh_metadata_property")
_impostor_provider = _SpecializeImpostor("_fresh_metadata_impostor")


def effectful_property(left: torch.Tensor) -> torch.Tensor:
    shape = _property_provider.specialize(left.shape)
    result = torch.empty(shape, dtype=left.dtype, device=left.device)
    for row in hl.tile(left.size(0)):
        result[row] = left[row]
    return result


def same_spelled_impostor(left: torch.Tensor) -> torch.Tensor:
    shape = _impostor_provider.specialize(left.shape)
    result = torch.empty(shape, dtype=left.dtype, device=left.device)
    for row in hl.tile(left.size(0)):
        result[row] = left[row]
    return result


def bind_metadata(function: Callable[..., object]) -> BoundKernel[Any]:
    kernel = helion.kernel(
        function, backend="cute", static_shapes=False, autotune_effort="full"
    )
    return kernel._bind_isolated((torch.empty((64,), dtype=torch.bfloat16),))


@pytest.mark.parametrize(
    "function", (known_assignments, known_aliases, known_expression)
)
def test_known_builtin_metadata_retains_freshness(
    function: Callable[..., object],
) -> None:
    bound = bind_metadata(function)
    host = bound.host_function
    assert host is not None
    with bound.env, host:
        found = _fresh_returned_tensors(host)
    assert len(found) == 1
    (output,) = found
    assert output.dtype is torch.bfloat16 and output.shape == (64,)


@pytest.mark.parametrize(
    "function", (nested_callback, effectful_property, same_spelled_impostor)
)
def test_untrusted_metadata_still_declines(function: Callable[..., object]) -> None:
    reads = _property_provider.reads + _impostor_provider.reads
    bound = bind_metadata(function)
    host = bound.host_function
    assert host is not None
    if function is not nested_callback:
        assert _property_provider.reads + _impostor_provider.reads > reads
    with bound.env, host:
        assert not _fresh_returned_tensors(host)


@pytest.mark.parametrize("fault", ("register_expression", "specialize_after_root"))
def test_builtin_statement_scope_does_not_expand(fault: str) -> None:
    bound = bind_metadata(known_assignments)
    host = bound.host_function
    assert host is not None
    name = "register_block_size" if fault == "register_expression" else "specialize"
    call = next(
        stmt.value
        for stmt in host.body
        if isinstance(stmt, ast.Assign)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == name
    )
    root = next(i for i, stmt in enumerate(host.body) if isinstance(stmt, ast.For))
    insertion = root if fault == "register_expression" else root + 1
    altered = cast(
        "HostFunction",
        SimpleNamespace(
            params=host.params,
            body=[
                *host.body[:insertion],
                ast.Expr(value=call),
                *host.body[insertion:],
            ],
        ),
    )
    with bound.env, host:
        assert len(_fresh_returned_tensors(host)) == 1
        assert not _fresh_returned_tensors(altered)


@pytest.mark.parametrize(
    "statement",
    (
        "gate_out.set_(left)",
        "opaque(gate_out)",
        "alias = gate_out.view(-1)",
        "gate_out = left",
        "left = gate_out",
        "gate_out[0] = 1",
    ),
)
def test_existing_alias_and_effect_rejections(statement: str) -> None:
    check_alias_effect(statement)
