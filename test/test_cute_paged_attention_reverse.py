from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import pytest
import sympy
import torch

import helion
from helion._compiler.cute.analysis.paged_attention_match import match_paged_attention
from helion._compiler.cute.paged_attention_transport import (
    plan_paged_attention_transport,
)
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_paged_attention import copy_paged_attention_program
from helion._testing.cute_paged_reference import reference_attention_bound
from helion._testing.cute_paged_reference import reverse_attention_fixture

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("form", ["mapped", "per_key"])
@pytest.mark.parametrize("query_len", [1, 4, 5])
def test_reverse_source_preserves_its_typed_page_and_mask(form, query_len):
    bound, program = reference_attention_bound(form, query_len, reverse_pages=True)
    plan = match_paged_attention(program)
    assert plan is not None and plan.coordinates.reflected is not None
    config = optional_lse_config()
    config.config.update(block_sizes=[helion.next_power_of_2(query_len * 8)])
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    assert "reflected_dtype=cutlass.Int32" in source
    assert "reference_scores=True" in source
    assert "paged_attention_partial_body(" in source


@pytest.mark.parametrize(
    "mutation",
    [
        "one",
        "total",
        "bounds",
        "alpha",
        "key_product",
        "page_divisor",
        "subpage_divisor",
        "page_source",
    ],
)
def test_reverse_proof_rejects_mismatched_coordinates(mutation):
    bound, original = reference_attention_bound("mapped", 5, reverse_pages=True)
    program = copy_paged_attention_program(original)
    plan = match_paged_attention(program)
    assert plan is not None and plan.coordinates.reflected is not None
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        assert plan_paged_attention_transport(bound.env, plan) is not None
    page_index = plan.page_table.subscripts[1].expr
    subpage_index = plan.key.raw.subscripts[1].expr
    assert isinstance(page_index, sympy.Symbol)
    assert isinstance(subpage_index, sympy.Symbol)
    token = plan.coordinates.reflected.value
    prefix = token.args[0]
    assert isinstance(prefix, torch.fx.Node)
    total = prefix.args[0]
    assert isinstance(total, torch.fx.Node)
    total = total.args[0]  # source's multiply by one
    assert isinstance(total, torch.fx.Node)
    if mutation == "one":
        prefix.args = (prefix.args[0], 2)
    elif mutation == "total":
        total.target = torch.ops.aten.sub.Tensor
    elif mutation == "bounds":
        total.args = (total.args[0], total.args[0])
    elif mutation == "alpha":
        token.kwargs = {"alpha": 2}
    elif mutation == "key_product":
        product = plan.mask.key_position.args[1]
        assert isinstance(product, torch.fx.Node)
        product.args = (product.args[0], 64)
    elif mutation == "page_divisor":
        node = program.index_values[page_index]
        node.args = (node.args[0], 2)
    elif mutation == "subpage_divisor":
        node = program.index_values[subpage_index]
        node.args = (node.args[0], 2)
    else:
        node = program.index_values[page_index]
        node.args = (prefix, 1)
    with bound.env, bound.host_function:
        changed = match_paged_attention(program)
        if mutation in ("page_divisor", "subpage_divisor", "page_source"):
            assert changed is not None and changed.coordinates.reflected is not None
            assert plan_paged_attention_transport(bound.env, changed) is None
        else:
            assert changed is None


def test_reverse_bound_tensors_keep_their_int64_dtype():
    kernel, args = reverse_attention_fixture("per_key", 4, True)
    args = list(args)
    args[10] = torch.empty((2, 2), dtype=torch.int64)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(cast("Kernel[object]", kernel), tuple(args))
        source = bound.to_code(optional_lse_config())
    assert "reflected_dtype=cutlass.Int64" in source


@pytest.mark.parametrize("form", ["mapped", "per_key"])
def test_reverse_source_without_reference_scores(form):
    query_len = 5
    kernel, args = reverse_attention_fixture(
        form, query_len, True, reference_scores=False
    )
    config = optional_lse_config()
    config.config.update(block_sizes=[helion.next_power_of_2(query_len * 8)])
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(cast("Kernel[object]", kernel), args)
        source = bound.to_code(config)
    assert "reflected_dtype=cutlass.Int32" in source
    assert "reference_scores=True" not in source
    assert "paged_attention_partial_body(" in source
