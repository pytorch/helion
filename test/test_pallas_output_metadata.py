from __future__ import annotations

import struct

import pytest
import torch

import helion
from helion._testing import onlyBackends
from helion._testing import skipIfPallasTpu
from helion.autotuner import PowerOfTwoFragment
import helion.language as hl
from helion.runtime.pallas.launcher import _LauncherFastPath
from helion.runtime.pallas.launcher import _pallas_collect_outputs
from helion.runtime.pallas.launcher import _pallas_invoke_and_return_fast
from helion.runtime.pallas.launcher import default_pallas_launcher

pytestmark = pytest.mark.cpu_only
META_DEVICE = torch.device("meta")


@helion.kernel(static_shapes=True, autotune_effort="none")
def _copy_with_metadata(x, requires: hl.constexpr):
    out = torch.empty_like(x, requires_grad=requires)
    for tile in hl.tile(x.shape):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _conditional_output(x, flag: int):
    if flag > 0:
        x.t_()
    output_size = x.size(0)
    out = torch.empty((output_size,), dtype=x.dtype, device=x.device)
    for row in hl.tile(output_size):
        out[row] = 1.0
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _transpose_then_add_inplace(x, flag: int):
    if flag > 0:
        x.t_()
    for tile in hl.tile(x.shape):
        x[tile] = x[tile] + 1
    return x


@helion.kernel(static_shapes=True, autotune_effort="none")
def _specialized_conditional_output(x, flag: hl.constexpr):
    output_size = x.size(1) if flag > 0 else x.size(0)
    if flag > 0:
        x.t_()
    out = torch.empty((output_size,), dtype=x.dtype, device=x.device)
    for row in hl.tile(output_size):
        out[row] = 1.0
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _scalar_sized_output(x, size: int, specialize_size: hl.constexpr):
    if specialize_size:
        size = hl.specialize(size)
    out = torch.empty((size,), dtype=x.dtype, device=x.device)
    for tile in hl.tile(size):
        out[tile] = 1.0
    return out


@helion.kernel(static_shapes=True, autotune_effort="none")
def _configured_output(x):
    size = hl.register_tunable("size", PowerOfTwoFragment(1, 4, 2))
    out = torch.empty((size,), dtype=x.dtype, device=x.device)
    for tile in hl.tile(size):
        out[tile] = 1.0
    return out


@helion.kernel(static_shapes=False, autotune_effort="none")
def _scalar_value_output(x, value: float):
    out = torch.empty_like(x)
    for tile in hl.tile(x.shape):
        out[tile] = value
    return out


@onlyBackends(["pallas"])
@skipIfPallasTpu("executes the Pallas CPU interpreter with CPU input tensors")
def test_runtime_scalar_float_values_reuse_launcher_entry():
    config = helion.Config(block_sizes=[128])
    kernel = helion.kernel(
        _scalar_value_output.fn,
        backend="pallas",
        static_shapes=False,
        autotune_effort="none",
        config=config,
    )
    nan1 = struct.unpack("!d", bytes.fromhex("7ff8000020000000"))[0]
    nan2 = struct.unpack("!d", bytes.fromhex("7ff8000040000000"))[0]
    x = torch.zeros(128)
    first_bound = None
    first_cache = None
    for value in (0.0, -0.0, 0.0, nan1, nan2, nan2):
        out = kernel(x, value)
        expected = torch.full_like(x, value)
        torch.testing.assert_close(out, expected, equal_nan=True)
        torch.testing.assert_close(torch.signbit(out), torch.signbit(expected))
        bound = kernel.bind((x, value))
        if first_bound is None:
            first_bound = bound
        assert bound is first_bound
        compiled = bound.compile_config(config)
        caches = [
            fn._pallas_cache
            for fn in compiled.__globals__.values()
            if getattr(fn, "_pallas_cache", None) is not None
        ]
        assert len(caches) == 1
        if first_cache is None:
            first_cache = caches[0]
        assert caches[0] is first_cache


@onlyBackends(["pallas"])
@skipIfPallasTpu("executes the Pallas CPU interpreter with CPU input tensors")
def test_captured_scalar_float_bits_select_distinct_launcher_entries():
    import jax.numpy as jnp

    def scalar_fill(out, value):
        out[:] = jnp.full(out.shape, value, dtype=jnp.float32)

    placeholder = torch.empty((128,), device=META_DEVICE)
    nan1 = struct.unpack("!d", bytes.fromhex("7ff8000020000000"))[0]
    nan2 = struct.unpack("!d", bytes.fromhex("7ff8000040000000"))[0]
    previous_bits = None
    previous_cache = None
    for value in (0.0, -0.0, 0.0, nan1, nan2, nan2):
        out = default_pallas_launcher(
            scalar_fill,
            (1,),
            placeholder,
            value,
            _output_indices=[0],
            _inplace_indices=[],
            _pallas_interpret=True,
        )
        expected = torch.full((128,), value)
        torch.testing.assert_close(out, expected, equal_nan=True)
        torch.testing.assert_close(torch.signbit(out), torch.signbit(expected))
        cache = scalar_fill._pallas_cache
        bits = struct.pack("!d", value)
        if bits == previous_bits:
            assert cache is previous_cache
        else:
            assert cache is not previous_cache
        previous_bits = bits
        previous_cache = cache


@onlyBackends(["pallas"])
@skipIfPallasTpu("executes the Pallas CPU interpreter with CPU input tensors")
def test_configured_output_extents_remain_supported():
    x = torch.ones(8)
    bound = _configured_output.bind((x,))
    for size in (1, 4, 2, 1):
        config = helion.Config(block_sizes=[4], size=size)
        out = bound.compile_config(config)(x)
        torch.testing.assert_close(out, torch.ones(size))


@onlyBackends(["pallas"])
@skipIfPallasTpu("executes the Pallas CPU interpreter with CPU input tensors")
def test_runtime_scalar_outputs_follow_normal_dispatch():
    first_bound = None
    for size in (8, 4, 8):
        x = torch.ones(8)
        bound = _scalar_sized_output.bind((x, size, False))
        if first_bound is None:
            first_bound = bound
        assert bound is first_bound
        out = _scalar_sized_output(x, size, False)
        torch.testing.assert_close(out, torch.ones(size))


@onlyBackends(["pallas"])
@skipIfPallasTpu("executes the Pallas CPU interpreter with CPU input tensors")
def test_specialized_scalar_outputs_follow_normal_dispatch():
    for size in (8, 4, 8):
        out = _scalar_sized_output(torch.ones(8), size, True)
        torch.testing.assert_close(out, torch.ones(size))


@onlyBackends(["pallas"])
@skipIfPallasTpu("executes the Pallas CPU interpreter with CPU input tensors")
@pytest.mark.parametrize("static_shapes", [False, True])
def test_conditional_output_metadata_follows_normal_dispatch(static_shapes):
    kernel = helion.kernel(
        _conditional_output.fn,
        backend="pallas",
        static_shapes=static_shapes,
        autotune_effort="none",
        config=helion.Config(block_sizes=[8]),
    )
    first_bound = None
    for flag in (1, 0, 1):
        x = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        bound = kernel.bind((x, flag))
        if first_bound is None:
            first_bound = bound
        assert bound is first_bound
        out = kernel(x, flag)
        torch.testing.assert_close(out, torch.ones(x.size(0)))


@onlyBackends(["pallas"])
@skipIfPallasTpu("executes the Pallas CPU interpreter with CPU input tensors")
def test_host_transpose_inplace_writes_back_across_cached_calls():
    kernel = helion.kernel(
        _transpose_then_add_inplace.fn,
        backend="pallas",
        static_shapes=True,
        autotune_effort="none",
        config=helion.Config(block_sizes=[8, 8]),
    )
    first_bound = None
    for flag in (1, 0, 1, 1):
        x = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        storage_alias = x.view(-1)
        expected = x.t() + 1 if flag > 0 else x + 1
        bound = kernel.bind((x, flag))
        if first_bound is None:
            first_bound = bound
        assert bound is first_bound
        actual = kernel(x, flag)
        assert actual is x
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            storage_alias, torch.arange(32, dtype=torch.float32) + 1, rtol=0, atol=0
        )


@onlyBackends(["pallas"])
@skipIfPallasTpu("executes the Pallas CPU interpreter with CPU input tensors")
def test_specializing_control_before_mutation_preserves_output_shape():
    for flag in (1, 0, 1):
        x = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        out = _specialized_conditional_output(x, flag)
        torch.testing.assert_close(out, torch.ones(x.size(0)))


@onlyBackends(["pallas"])
@skipIfPallasTpu("executes the Pallas CPU interpreter with CPU input tensors")
@pytest.mark.parametrize("size,stride,requires", [(1, 2, False), (4, 1, True)])
def test_output_preserves_authored_allocation_metadata(size, stride, requires):
    x = torch.empty_strided((size,), (stride,))
    x.fill_(2)
    expected = torch.empty_like(x, requires_grad=requires)
    out = _copy_with_metadata(x, requires)
    torch.testing.assert_close(out, x + 1)
    assert out.stride() == expected.stride()
    assert out.requires_grad is requires


@onlyBackends(["pallas"])
@skipIfPallasTpu("executes the Pallas CPU interpreter with CPU input tensors")
@pytest.mark.parametrize("size", [0, 1])
def test_empty_and_singleton_jax_results_preserve_strides(size):
    import jax.numpy as jnp

    result = jnp.full((size,), 3, dtype=jnp.float32)
    placeholder = torch.empty_strided((size,), (2,), device=META_DEVICE)
    out = _pallas_collect_outputs(result, (placeholder,), ((0, 0),), None, {}, ())
    torch.testing.assert_close(out, torch.full((size,), 3.0))
    assert out.stride() == placeholder.stride()


@onlyBackends(["pallas"])
def test_matching_torch_result_is_returned_without_copy():
    result = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    placeholder = torch.empty_like(result, device=META_DEVICE)
    out = _pallas_collect_outputs(result, (placeholder,), ((0, 0),), None, {}, ())
    assert out is result


@onlyBackends(["pallas"])
def test_empty_grid_preserves_output_metadata_and_inplace_values():
    inplace = torch.arange(8, dtype=torch.float32)
    placeholder = torch.empty_strided(
        (4, 0), (3, 2), device=META_DEVICE, requires_grad=True
    )

    def should_not_run(*args):
        raise AssertionError("an empty grid must not execute the kernel")

    out = default_pallas_launcher(
        should_not_run,
        (1, 0),
        inplace,
        placeholder,
        _output_indices=[0, 1],
        _inplace_indices=[0],
        _pallas_interpret=True,
    )
    assert out.shape == placeholder.shape
    assert out.stride() == placeholder.stride()
    assert out.requires_grad
    assert out.device == inplace.device
    torch.testing.assert_close(inplace, torch.arange(8, dtype=torch.float32))
    assert not hasattr(should_not_run, "_pallas_cache")


@onlyBackends(["pallas"])
def test_empty_grid_keeps_default_inplace_return_convention():
    output = torch.empty((0,))

    def should_not_run(*args):
        raise AssertionError("an empty grid must not execute the kernel")

    assert (
        default_pallas_launcher(
            should_not_run,
            (0,),
            output,
            _output_indices=[0],
            _pallas_interpret=True,
        )
        is None
    )


@onlyBackends(["pallas"])
@pytest.mark.parametrize(
    "stride,requires", [((1, 2), True), ((3, 1), True), ((3, 1), False)]
)
def test_single_torch_fast_result_preserves_output_metadata(stride, requires):
    result = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    placeholder = torch.empty_strided(
        result.shape, stride, device=META_DEVICE, requires_grad=requires
    )
    fast_path = _LauncherFastPath([], {}, [0], None)
    out = _pallas_invoke_and_return_fast(
        lambda: result, (placeholder,), fast_path, None
    )
    torch.testing.assert_close(out, result)
    assert out.stride() == stride
    assert out.requires_grad is requires
    if stride == result.stride():
        assert out is result


@onlyBackends(["pallas"])
@skipIfPallasTpu("executes the Pallas CPU interpreter with CPU input tensors")
def test_dynamic_input_geometry_rebuilds_output_descriptors():
    config = helion.Config(block_sizes=[128])
    kernel = helion.kernel(
        _copy_with_metadata.fn,
        backend="pallas",
        static_shapes=False,
        autotune_effort="none",
        config=config,
    )
    first_bound = None
    previous_size = None
    previous_cache = None
    for size in (65, 70, 128, 70, 65, 65):
        x = torch.arange(size, dtype=torch.float32)
        out = kernel(x, False)
        torch.testing.assert_close(out, x + 1)
        bound = kernel.bind((x, False))
        if first_bound is None:
            first_bound = bound
        assert bound is first_bound
        compiled = bound.compile_config(config)
        caches = [
            value._pallas_cache
            for value in compiled.__globals__.values()
            if getattr(value, "_pallas_cache", None) is not None
        ]
        assert len(caches) == 1
        if size == previous_size:
            assert caches[0] is previous_cache
        else:
            assert caches[0] is not previous_cache
        previous_size = size
        previous_cache = caches[0]


@onlyBackends(["pallas"])
@skipIfPallasTpu("executes the Pallas CPU interpreter with CPU input tensors")
def test_static_output_metadata_cache_is_reused():
    config = helion.Config(block_sizes=[128])
    kernel = helion.kernel(
        _copy_with_metadata.fn,
        backend="pallas",
        static_shapes=True,
        autotune_effort="none",
        config=config,
    )
    x = torch.arange(65, dtype=torch.float32)
    torch.testing.assert_close(kernel(x, False), x + 1)
    compiled = kernel.bind((x, False)).compile_config(config)
    holders = [
        value
        for value in compiled.__globals__.values()
        if getattr(value, "_helion_output_meta_cache_0", None) is not None
    ]
    assert len(holders) == 1
    descriptor = holders[0]._helion_output_meta_cache_0
    torch.testing.assert_close(kernel(x + 2, False), x + 3)
    assert holders[0]._helion_output_meta_cache_0 is descriptor


@onlyBackends(["pallas"])
def test_padded_torch_result_uses_original_output_metadata():
    result = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    original = torch.empty_strided(
        (2, 3), (1, 2), device=META_DEVICE, requires_grad=True
    )
    padded = torch.empty((2, 4), device=META_DEVICE)
    out = _pallas_collect_outputs(
        result, (padded,), ((0, 0),), {0: original}, {0: [1]}, ()
    )
    torch.testing.assert_close(out, result[:, :3])
    assert out.shape == original.shape
    assert out.stride() == original.stride()
    assert out.requires_grad
