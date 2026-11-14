from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import pytest

from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._compiler.cute.paged_attention_transport import (
    plan_paged_attention_transport,
)
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_optional_lse import optional_lse_fixture

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("form", ["reduced", "per_key", "mapped"])
@pytest.mark.parametrize("return_lse", [False, True])
def test_optional_lse_preserves_native_route(form: str, return_lse: bool) -> None:
    kernel, args = optional_lse_fixture(form, return_lse)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(cast("Kernel[object]", kernel), args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
            assert program is not None
            semantic = match_paged_attention(program)
            assert semantic is not None
            assert (semantic.lse is not None) == return_lse
            transport = plan_paged_attention_transport(bound.env, semantic)
            assert transport is not None
            if form == "mapped":
                assert transport.direct_io is not None
                assert (transport.direct_io.lse is not None) == return_lse
            config = bound.config_spec.normalized_config(optional_lse_config())
        source = bound.to_code(config)
    body = (
        "paged_attention_body" if form == "reduced" else "paged_attention_partial_body"
    )
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == body
    ]
    assert len(calls) == 1
    lse_argument = calls[0].args[3]
    if return_lse:
        assert isinstance(lse_argument, ast.Name)
    else:
        assert isinstance(lse_argument, ast.Constant) and lse_argument.value is None
