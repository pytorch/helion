from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from helion import exc
from helion._compiler.cute.static_indexing import shape_only_indices
from helion._compiler.cute.static_indexing import static_tensor_index
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_static_indexing import static_index_full_axis
from helion._testing.cute_static_indexing import static_index_full_axis_first
from helion._testing.cute_static_indexing import static_index_full_axis_identity
from helion._testing.cute_static_indexing import static_index_halves
from helion._testing.cute_static_indexing import static_index_mixed
from helion._testing.cute_static_indexing import static_index_register
from helion._testing.cute_static_indexing import static_index_select
from helion._testing.cute_static_indexing import static_index_shape_only
from helion._testing.cute_static_indexing import static_index_tree

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "target,args,kwargs,expected",
    [
        (torch.ops.aten.slice.Tensor, (-1, -7, 20, 3), {}, [1, 4, 7]),
        (
            torch.ops.aten.slice.Tensor,
            (),
            {"dim": 1, "start": 2, "end": 7, "step": 2},
            [2, 4, 6],
        ),
        (torch.ops.aten.slice.Tensor, (1, -40, -2), {}, [0, 1, 2, 3, 4, 5]),
        (torch.ops.aten.slice.Tensor, (1, 6, 2), {}, []),
        (torch.ops.aten.select.int, (-1, -2), {}, 6),
    ],
)
def test_static_coordinate_indices(target, args, kwargs, expected):
    graph = Graph()
    source = graph.placeholder("source")
    values = torch.arange(16).reshape(2, 8)
    source.meta["val"] = values
    node = graph.call_function(target, (source, *args), kwargs)
    original = target(values, *args, **kwargs)
    matched = static_tensor_index(node)
    assert matched is not None
    mapped_source, indices = matched
    assert mapped_source is source
    actual = values[tuple(indices)]
    torch.testing.assert_close(actual, original)
    torch.testing.assert_close(actual, values[:, expected])


@pytest.mark.parametrize("bounds", [(0, 8, 1), (-20, 20, 1)])
def test_concrete_full_slice_preserves_scalar_coordinates(bounds):
    graph = Graph()
    source = graph.placeholder("source")
    values = torch.arange(16).reshape(2, 8)
    source.meta["val"] = values
    node = graph.call_function(torch.ops.aten.slice.Tensor, (source, 1, *bounds))
    matched = static_tensor_index(node)
    assert matched is not None and shape_only_indices(*matched)
    torch.testing.assert_close(values[tuple(matched[1])], values)


def test_symbolic_extent_only_accepts_the_full_slice():
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(static_index_halves, (torch.empty(3, 128),))
        with bound.env, bound.host_function:
            symbolic = bound.env.shape_env.create_unbacked_symint()
            graph = Graph()
            source = graph.placeholder("source")
            source.meta["val"] = torch.empty((symbolic, 8))
            full = graph.call_function(
                torch.ops.aten.slice.Tensor, (source, 0, 0, 2**63 - 1)
            )
            partial = graph.call_function(
                torch.ops.aten.slice.Tensor, (source, 0, 1, 4)
            )
            selected = graph.call_function(torch.ops.aten.select.int, (source, 0, 0))
            assert static_tensor_index(full) == (source, [slice(None), slice(None)])
            assert static_tensor_index(partial) is None
            assert static_tensor_index(selected) is None


def test_invalid_or_runtime_indices_are_not_canonicalized():
    graph = Graph()
    source = graph.placeholder("source")
    source.meta["val"] = torch.empty(2, 8)
    runtime_index = graph.placeholder("runtime_index")
    for target, args in (
        (torch.ops.aten.slice.Tensor, (1, runtime_index, 4)),
        (torch.ops.aten.slice.Tensor, (1, 0, 4, 0)),
        (torch.ops.aten.slice.Tensor, (1, 0, 4, -1)),
        (torch.ops.aten.select.int, (1, 8)),
        (torch.ops.aten.select.int, (2, 0)),
    ):
        assert static_tensor_index(graph.call_function(target, (source, *args))) is None


@pytest.mark.parametrize(
    "kernel",
    [static_index_tree, static_index_mixed, static_index_halves, static_index_select],
)
def test_computed_views_render_through_complete_owner(kernel):
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, (torch.arange(3 * 128).reshape(3, 128).float(),))
        source = bound.to_code(bound.config_spec.default_config())
    assert "fragment_smem" in source


@pytest.mark.parametrize("kernel", [static_index_halves, static_index_select])
def test_static_views_do_not_fall_back_to_unproved_scalar_coordinates(kernel):
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, (torch.empty(3, 128),))
        config = bound.config_spec.default_config()
        with (
            patch(
                "helion._compiler.cute.computed_fragment.computed_fragment_supported",
                return_value=False,
            ),
            patch(
                "helion._compiler.cute.register_region.plan_register_region",
                return_value=None,
            ),
            pytest.raises(
                exc.BackendUnsupported,
                match="static tensor indexing requires a complete coordinate owner",
            ),
        ):
            bound.to_code(config)


@pytest.mark.parametrize("select_singleton,width", [(False, 8), (True, 1)])
def test_shape_only_static_views_render_without_complete_owner(select_singleton, width):
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(
            static_index_shape_only, (torch.empty(3, width), select_singleton)
        )
        with (
            patch(
                "helion._compiler.cute.computed_fragment.computed_fragment_supported",
                return_value=False,
            ),
            patch(
                "helion._compiler.cute.register_region.plan_register_region",
                return_value=None,
            ),
        ):
            source = bound.to_code(bound.config_spec.default_config())
    assert "fragment_smem" not in source


def test_static_indexing_preserves_register_coordinate_owner():
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(static_index_register, (torch.empty(3, 8, 4),))
        source = bound.to_code(bound.config_spec.default_config())
    assert "_cute_execute_register_plan" in source
    assert "fragment_smem" not in source


@pytest.mark.parametrize("mode", ["halves", "last", "first"])
@pytest.mark.parametrize("check", ["analysis", "validation"])
def test_full_axis_indexing_refuses_reduction_rolling(mode, check):
    # None of these full-axis loads specialize the input width. A 32-element
    # chunk cannot represent the indexed coordinates of the entire row, even
    # when selecting element zero would be in bounds within every chunk.
    width = 128 if mode == "halves" else 127
    args = (torch.arange(3 * width).reshape(3, width).float(),)
    kernel = static_index_full_axis_first
    if mode != "first":
        kernel = static_index_full_axis
        args = (*args, mode == "last")
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, args)
        config = bound.config_spec.default_config()
        if check == "analysis":
            # Refusing a rolled candidate must retain the working persistent
            # lowering, including the non-power-of-two last-element case.
            assert "fragment_smem" in bound.to_code(config)
            assert not bound.config_spec.reduction_loops.valid_block_ids()
        else:
            config.config["reduction_loops"] = [32]
            # Reject during configuration validation, before splitting the IR
            # or reaching an incidental complete-owner lowering failure.
            with pytest.raises(exc.InvalidConfig):
                bound._normalized_config_copy(config)


@pytest.mark.parametrize("explicit_end", [False, True])
def test_full_axis_identity_requires_preserved_rdim(explicit_end):
    # A literal stop equal to the logical width can erase the reduction symbol:
    # its hint is 128 for this 127-element axis. A bare slice retains the symbol.
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(
            static_index_full_axis_identity, (torch.empty(3, 127), explicit_end)
        )
        assert (
            bool(bound.config_spec.reduction_loops.valid_block_ids())
            is not explicit_end
        )
        config = bound.config_spec.default_config()
        config.config["reduction_loops"] = [32]
        if explicit_end:
            with pytest.raises(exc.InvalidConfig):
                bound._normalized_config_copy(config)
        else:
            assert bound.to_code(config)
