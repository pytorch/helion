from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import helion
from helion._testing import onlyBackends
from helion._testing import skipIfPallasTpu
import helion.language as hl
from helion.runtime.pallas.launcher import _build_direct_call_invoke
from helion.runtime.pallas.launcher import _DirectCallKernel
from helion.runtime.pallas.launcher import _LauncherFastPath
from helion.runtime.pallas.launcher import _pallas_apply_ds_padding_fast
from helion.runtime.pallas.launcher import _pallas_invoke_and_return_fast
from helion.runtime.pallas.launcher import _PallasInterpretCallable

pytestmark = pytest.mark.cpu_only
META_DEVICE = torch.device("meta")


@helion.kernel(static_shapes=True, autotune_effort="none")
def _add_inplace(x, y):
    for tile in hl.tile(x.shape):
        x[tile] = x[tile] + y[tile]
    return x


@onlyBackends(["pallas"])
@skipIfPallasTpu("executes the Pallas CPU interpreter with CPU input tensors")
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "stepped"])
def test_inplace_kernel_preserves_original_storage(layout):
    kernel = helion.kernel(
        _add_inplace.fn,
        backend="pallas",
        static_shapes=True,
        autotune_effort="none",
        config=helion.Config(block_sizes=[8, 8]),
    )
    first_bound = None
    for delta in (0, 10):
        base = torch.arange(48, dtype=torch.float32).reshape(6, 8) + delta
        expected_base = base.clone()
        if layout == "transpose":
            x, expected = base.t(), expected_base.t()
        elif layout == "stepped":
            x, expected = base[1:5, ::2], expected_base[1:5, ::2]
        else:
            x, expected = base, expected_base
        y = torch.full_like(x, 3)
        expected.add_(y)
        metadata = (x.stride(), x.storage_offset(), x.untyped_storage().data_ptr())
        bound = kernel.bind((x, y))
        if first_bound is None:
            first_bound = bound
        assert bound is first_bound
        actual = kernel(x, y)
        assert actual is x
        assert metadata == (
            x.stride(),
            x.storage_offset(),
            x.untyped_storage().data_ptr(),
        )
        torch.testing.assert_close(base, expected_base, rtol=0, atol=0)
        torch.testing.assert_close(y, torch.full_like(y, 3), rtol=0, atol=0)


@onlyBackends(["pallas"])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "padded"])
def test_direct_call_writeback_preserves_donation_and_padding(layout):
    """Exercise the native dispatch wrapper with a CPU custom-kernel substitute."""
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    if layout != "contiguous":
        x = x.t()
    expected = x + 2
    readonly = torch.arange(6, dtype=torch.float32).reshape(3, 2).t()
    readonly_before = readonly.clone()
    readonly_version = readonly._version
    x_version = x._version
    output = torch.empty_like(x, device=META_DEVICE)
    pad_dims = [(0, 0, 4, 0), (2, 0, 4, 0)] if layout == "padded" else None
    fast = _LauncherFastPath([0, 1], {0: 0, 1: 1}, [0, 2], pad_dims)
    args = (x, readonly, output)
    originals = None
    if pad_dims:
        args, originals, padded = _pallas_apply_ds_padding_fast(
            args, pad_dims, fast, fast.padded_output_arg_indices
        )
        assert padded
    calls = []

    def custom_kernel(name, fingerprint, *, inputs, **kwargs):
        assert name == "inplace" and fingerprint == "fingerprint"
        assert kwargs["donate_argnums"] == (0,)
        assert all(value.is_contiguous() for value in inputs)
        if layout == "contiguous":
            assert inputs[0] is x
        calls.append(inputs)
        updated = inputs[0] + 2
        return [updated, updated * 3]

    invoke = _build_direct_call_invoke(
        custom_kernel,
        "inplace",
        "fingerprint",
        None,
        (0,),
        SimpleNamespace(unflatten=tuple),
        ((0, 0),),
        [],
    )
    direct = _DirectCallKernel(
        call_custom_kernel=custom_kernel,
        kernel_name="inplace",
        mlir_fingerprint="fingerprint",
        output_shapes=None,
        donate_argnums=(0,),
        out_tree=None,
        alias_items=((0, 0),),
        sig=tuple((arg.shape, arg.dtype) for arg in args[:2]),
        invoke=invoke,
    )

    def unexpected_fallback(*inputs):
        raise AssertionError("the matching direct call must run")

    result = _pallas_invoke_and_return_fast(
        unexpected_fallback, args, fast, originals, direct
    )
    assert len(calls) == 1 and direct.sig_locked
    torch.testing.assert_close(x, expected, rtol=0, atol=0)
    torch.testing.assert_close(result, expected * 3, rtol=0, atol=0)
    torch.testing.assert_close(readonly, readonly_before, rtol=0, atol=0)
    assert readonly._version == readonly_version
    # Exactly one writeback, including the contiguous case's existing alias copy.
    assert x._version == x_version + 1


@onlyBackends(["pallas"])
@skipIfPallasTpu("executes the Pallas CPU interpreter with CPU input tensors")
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_contiguous_writeback_keeps_ambient_grad_mode(grad_enabled):
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3).t().requires_grad_()
    expected = x.detach() + 2
    callable_obj = _PallasInterpretCallable(lambda value: value + 2, [(0, 0)])
    fast = _LauncherFastPath([0], {0: 0}, [0], None)
    with torch.set_grad_enabled(grad_enabled):
        if grad_enabled:
            # The original is a leaf requiring grad, just as for a contiguous
            # input written by the existing interpreter alias handler.
            with pytest.raises(RuntimeError, match="leaf Variable"):
                _pallas_invoke_and_return_fast(callable_obj, (x,), fast, None)
        else:
            assert (
                _pallas_invoke_and_return_fast(callable_obj, (x,), fast, None) is None
            )
            assert x.requires_grad
            torch.testing.assert_close(x, expected, rtol=0, atol=0)
