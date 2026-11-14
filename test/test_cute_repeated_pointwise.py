from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_output_addresses import generated_output_address_coverage
from helion._testing.cute_repeated_axes import full_slice_add
from helion._testing.cute_repeated_axes import full_slice_row_sum
from helion._testing.cute_repeated_axes import full_slice_tiled_dot
from helion._testing.cute_repeated_axes import grid_full_slice_add
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True)
def _independently_tiled_add(x: torch.Tensor):
    out = torch.empty_like(x)
    for b, m, n in hl.tile(x.shape, block_size=[1, 16, 16]):
        out[b, m, n] = x[b, m, n] + 1.0
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _computed_square_dot(x: torch.Tensor):
    out = torch.empty_like(x)
    for batch in hl.grid(x.size(0)):
        value = x[batch, :, :] + 1.0
        out[batch, :, :] = hl.dot(value, value.T)
    return out


def _render(kernel, shape, **options):
    with cpu_target():
        bound = cpu_bind(kernel, (torch.empty(shape),))
        config = bound.config_spec.default_config()
        config.config.update(options)
        return bound.to_code(config)


@pytest.mark.parametrize("size", [16, 40])
def test_full_slice_pointwise_uses_complete_coordinates(size):
    threads = 128
    source = _render(full_slice_add, (1, size, size), cute_fragment_threads=threads)
    assert "fragment_thread" in source
    assert "fragment_buffer" in source
    assert generated_output_address_coverage(
        source, "out", 1, threads
    ) == dict.fromkeys(range(size * size), 1)


def test_rectangular_full_slices_keep_scalar_owner():
    source = _render(full_slice_add, (1, 16, 32))
    assert "fragment_thread" not in source


def test_equal_sizes_with_independent_tiles_keep_scalar_owner():
    source = _render(_independently_tiled_add, (1, 16, 16))
    assert "fragment_thread" not in source


def test_computed_square_dot_keeps_complete_owner():
    source = _render(_computed_square_dot, (1, 16, 16))
    assert "fragment_thread" in source


def test_repeated_axis_scalar_guard_remains_active():
    with (
        patch(
            "helion._compiler.cute.computed_fragment.codegen_computed_fragment_root",
            return_value=False,
        ),
        pytest.raises(exc.BackendUnsupported, match="to two axes of one tensor"),
    ):
        _render(full_slice_add, (1, 16, 16))


@pytest.mark.parametrize("epilogue", [False, True])
def test_repeated_axis_tiled_dot_uses_complete_coordinates(epilogue):
    with cpu_target():
        q = torch.empty((2, 16, 32), dtype=torch.bfloat16)
        bound = cpu_bind(full_slice_tiled_dot, (q, torch.empty_like(q), epilogue))
        source = bound.to_code(helion.Config(block_sizes=[1, 16]))
    assert "fragment_thread" in source


def test_tiled_dot_scalar_guard_remains_active():
    with cpu_target():
        q = torch.empty((2, 16, 32), dtype=torch.bfloat16)
        bound = cpu_bind(full_slice_tiled_dot, (q, torch.empty_like(q), False))
        with (
            patch(
                "helion._compiler.cute.computed_fragment.codegen_computed_fragment_root",
                return_value=False,
            ),
            pytest.raises(exc.BackendUnsupported, match="to two axes of one tensor"),
        ):
            bound.to_code(helion.Config(block_sizes=[1, 16]))


@pytest.mark.parametrize(
    "threads,register_loads", [(64, False), (256, False), (128, True)]
)
def test_grid_full_slices_admit_complete_coordinate_configs(threads, register_loads):
    with cpu_target():
        bound = cpu_bind(grid_full_slice_add, (torch.empty((3, 16, 16)),))
        assert bound.config_spec.cute_fragment_thread_root_ids == {0}
        assert bound.config_spec.cute_fragment_register_load_root_ids == {0}
        assert bound.config_spec.cute_fragment_packet_load_root_ids == {0}
        assert bound.config_spec.cute_fragment_vector_io_root_ids == {0}
        assert bound.config_spec.cute_fragment_unrolled_loops_root_ids == {0}
        default = bound.config_spec.default_config()
        assert "fragment_thread" not in bound.to_code(default)
        config = helion.Config(**default.config)
        config.config.update(
            cute_fragment_threads=threads,
            cute_fragment_register_loads=register_loads,
        )
        source = bound.to_code(config)
    assert "fragment_thread" in source
    assert f"block=({threads}, 1, 1)" in source
    assert generated_output_address_coverage(
        source, "out", 3, threads
    ) == dict.fromkeys(range(3 * 16 * 16), 1)


@pytest.mark.parametrize("size", [16, 40])
def test_repeated_axis_reduction_preserves_each_output_row(size):
    kernel = helion.kernel(
        full_slice_row_sum.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    source = _render(kernel, (3, size, size))
    assert "fragment_thread" in source
    assert generated_output_address_coverage(source, "out", 3, 128) == dict.fromkeys(
        range(3 * size), 1
    )


def test_dynamic_repeated_axis_reduction_specializes_shape_and_stride():
    kernel = helion.kernel(
        full_slice_row_sum.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
    )
    backing = torch.empty((3, 16, 35))
    inputs = [
        backing[:, :, 1:33:2],
        torch.empty((3, 40, 83))[:, :, 1:81:2],
        torch.empty((3, 16, 16)),
    ]
    with cpu_target():
        bounds = [kernel.bind((value,)) for value in inputs]
        assert len({id(bound) for bound in bounds}) == 3
        assert kernel.bind((backing.clone()[:, :, 1:33:2],)) is bounds[0]
        for bound in bounds:
            assert (
                "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts
            )
            assert "fragment_thread" in bound.to_code(
                bound.config_spec.default_config()
            )
