from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipUnlessCuteAvailable
from helion._testing.cute_static_indexing import mixed_views
from helion._testing.cute_static_indexing import ordered_tree
from helion._testing.cute_static_indexing import plain_halves
from helion._testing.cute_static_indexing import plain_last
from helion._testing.cute_static_indexing import plain_select
from helion._testing.cute_static_indexing import shape_only_view
from helion._testing.cute_static_indexing import static_index_full_axis
from helion._testing.cute_static_indexing import static_index_full_axis_first
from helion._testing.cute_static_indexing import static_index_full_axis_identity
from helion._testing.cute_static_indexing import static_index_halves
from helion._testing.cute_static_indexing import static_index_mixed
from helion._testing.cute_static_indexing import static_index_register
from helion._testing.cute_static_indexing import static_index_select
from helion._testing.cute_static_indexing import static_index_shape_only
from helion._testing.cute_static_indexing import static_index_tree


@skipUnlessCuteAvailable("requires CuTe DSL")
def test_static_index_tree_preserves_fp32_addition_order():
    generator = torch.Generator().manual_seed(634)
    values = torch.randn(7, 128, generator=generator)
    # Deliberately discriminates the prescribed tree from reassociation.
    values[0].zero_()
    values[0, 0] = 2**24
    values[0, 16] = 1
    values[0, 32] = -(2**24)
    values[1].fill_(-0.0)
    values[2, 7] = float("nan")
    values[3, 8] = float("inf")
    values[4, 8] = -float("inf")
    expected = ordered_tree(values)
    assert expected[0].item() == 0
    assert (values[0, 0] + (values[0, 16] + values[0, 32])).item() == 1
    actual = static_index_tree(values.to(DEVICE)).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    finite = torch.isfinite(expected)
    assert torch.equal(
        actual[finite].view(torch.int32), expected[finite].view(torch.int32)
    )


@skipUnlessCuteAvailable("requires CuTe DSL")
def test_static_indexing_composes_computed_and_transposed_views():
    values = torch.arange(3 * 128, dtype=torch.int32).reshape(3, 128) - 170
    actual = static_index_mixed(values.to(DEVICE)).cpu()
    torch.testing.assert_close(actual, mixed_views(values), rtol=0, atol=0)


@skipUnlessCuteAvailable("requires CuTe DSL")
def test_narrowing_computed_vectors_and_scalar_selection():
    values = torch.arange(3 * 128, dtype=torch.int32).reshape(3, 128) - 170
    device_values = values.to(DEVICE)
    torch.testing.assert_close(
        static_index_halves(device_values).cpu(), plain_halves(values), rtol=0, atol=0
    )
    torch.testing.assert_close(
        static_index_select(device_values).cpu(), plain_select(values), rtol=0, atol=0
    )


@skipUnlessCuteAvailable("requires CuTe DSL")
def test_static_indexing_preserves_register_permutation():
    values = torch.arange(3 * 8 * 4, dtype=torch.int32).reshape(3, 8, 4) - 47
    expected = torch.maximum(values, values[:, torch.arange(8) ^ 1])[:, :, ::2]
    actual = static_index_register(values.to(DEVICE)).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("mode,width", [("halves", 128), ("last", 127), ("first", 127)])
def test_full_axis_loads_use_logical_extent(mode, width):
    values = torch.arange(3 * width, dtype=torch.int32).reshape(3, width) - 170
    if mode == "first":
        expected = plain_select(values)
        actual = static_index_full_axis_first(values.to(DEVICE)).cpu()
    else:
        expected = plain_last(values) if mode == "last" else plain_halves(values)
        actual = static_index_full_axis(values.to(DEVICE), mode == "last").cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("select_singleton,width", [(False, 8), (True, 1)])
def test_shape_only_views_preserve_scalar_values(select_singleton, width):
    values = torch.arange(3 * width, dtype=torch.int32).reshape(3, width) - 7
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
        actual = static_index_shape_only(values.to(DEVICE), select_singleton).cpu()
    torch.testing.assert_close(
        actual, shape_only_view(values, select_singleton), rtol=0, atol=0
    )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("explicit_end", [False, True])
def test_full_axis_identity_preserves_partial_chunk_values(explicit_end):
    values = torch.arange(3 * 127, device=DEVICE, dtype=torch.float32).reshape(3, 127)
    bound = static_index_full_axis_identity.bind((values, explicit_end))
    config = bound.config_spec.default_config()
    if not explicit_end:
        config.config["reduction_loops"] = [32]
    bound.set_config(config)
    torch.testing.assert_close(bound(values, explicit_end), values * 2, rtol=0, atol=0)
