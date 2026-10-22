from __future__ import annotations

import ast

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_atomic_ops import _simulate_register_load_program
from test.test_cute_int64_value_network import _integer_data
from test.test_cute_int64_value_network import _value_selection

import helion


def _codegen(x, k, largest, threads):
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_value_selection, (x, k, largest, "topk"))
        config = bound.config_spec.default_config()
        config.config.update(
            block_sizes=[1],
            cute_fragment_topk_network=True,
            cute_fragment_cta_topk=True,
            cute_fragment_threads=threads,
        )
        return bound, config, bound.to_code(config)


@pytest.mark.parametrize(
    "width,k,threads", [(17, 7, 128), (65, 65, 128), (257, 129, 256)]
)
@pytest.mark.parametrize("largest", [True, False])
def test_cta_int64_sort_preserves_signed_extrema_padding_and_partial_outputs(
    width, k, threads, largest
):
    x = _integer_data(width)
    before = x.clone()
    expected = torch.sort(x, dim=-1, descending=largest).values[..., :k]
    _bound, _config, source = _codegen(x, k, largest, threads)
    assert "fragment_cta_sort_keys" in source
    for order in (list(range(threads)), list(reversed(range(threads)))):
        values = torch.full_like(expected, -999)
        _simulate_register_load_program(
            source, x, threads, host_tensors={"values": values}, lane_order=order
        )
        assert torch.equal(values, expected)
    assert torch.equal(x, before)


def test_cta_sort_typed_dependency_and_no_index_result_allocation():
    x = _integer_data(64)
    bound, config, source = _codegen(x, 31, True, 128)
    assert (
        bound.config_spec.default_config().get("cute_fragment_cta_topk", False) is False
    )
    missing = helion.Config.from_dict(
        dict(config.config) | {"cute_fragment_topk_network": False}
    )
    with pytest.raises(helion.exc.InvalidConfig, match="requires"):
        bound.config_spec.normalized_config(missing)
    assert "fragment_cta_sort_keys" in source
    assert "fragment_topk_selected_column" not in source
    group = next(
        group
        for group in bound.config_spec.compiler_coverage_groups
        if group.key == "cute_fragment_cta_topk"
    )
    assert any(
        dependency.key == "cute_fragment_topk_network" and dependency.value is True
        for dependency in group.dependencies
    )


def test_cta_sort_full_1024_network_source_has_bounded_storage_and_retirement():
    x = _integer_data(1000)
    _bound, _config, source = _codegen(x, 1000, True, 256)
    assert "cute.make_rmem_tensor((4,), cutlass.Int64)" in source
    parsed = ast.parse(source)
    loops = [
        node
        for node in ast.walk(parsed)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("fragment_cta_sort_row")
    ]
    assert len(loops) == 1
    barriers = [
        node
        for node in loops[0].body
        if isinstance(node, ast.Expr) and "sync_threads" in ast.unparse(node)
    ]
    assert len(barriers) == 25
