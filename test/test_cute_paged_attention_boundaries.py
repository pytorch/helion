from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import pytest

from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._compiler.cute.paged_attention_transport import (
    paged_attention_schedule_facts,
)
from helion._compiler.cute.paged_attention_transport import (
    plan_paged_attention_transport,
)
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("constant", ["query_len", "group", "window"])
def test_attention_declines_promoting_integer_literals(constant: str) -> None:
    args = list(paged_attention_args())
    position = {"query_len": 7, "group": 8, "window": 9}[constant]
    # Keep the wrapped group divisor nonzero, so the test isolates literal
    # promotion rather than an invalid division.
    args[position] = 2**32 + (constant == "group")
    with cpu_render_target():
        bound = cpu_bind(cast("Kernel[object]", paged_online_attention), tuple(args))
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            program = match_mma_dag(bound.host_function.device_ir.graphs)
            assert program is not None
            semantic = match_paged_attention(program)
            assert semantic is not None
            assert plan_paged_attention_transport(bound.env, semantic) is not None
            assert paged_attention_schedule_facts(bound.env, semantic) is None
