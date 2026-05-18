"""Parity tests for the C bind cache.

The C extension's ``helion._C.tensor_key`` mirrors the static-shape branch
of ``helion.runtime.kernel._tensor_key`` and the
``_base_specialization_key`` fast path. This file checks that:

1. The two implementations return equal keys for representative
   shapes / dtypes / strides.
2. ``Kernel.bind(args)`` returns the **same** ``BoundKernel`` object
   whether the C bind path is enabled or not — i.e. the cache key is
   stable across the two implementations.

Skipped when the C extension didn't build (e.g. PyTorch-on-MTIA), since
in that case the C/Python parity question is moot.
"""

from __future__ import annotations

import importlib

import pytest
import torch

import helion
import helion._C as _helion_c
import helion.language as hl
from helion.runtime.settings import _get_backend

_kernel_module = importlib.import_module("helion.runtime.kernel")

# The ``Kernel.bind`` parity tests construct Helion kernels that rely on
# the Triton launcher path; skip the file under non-Triton backends.
pytestmark = [
    pytest.mark.skipif(
        not _helion_c.available, reason="helion._C extension not built in this env"
    ),
    pytest.mark.skipif(
        _get_backend() != "triton",
        reason="C bind cache tests exercise the Triton kernel launch path",
    ),
]


@helion.kernel(static_shapes=True)
def _identity_two(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


def _py_tensor_key(t: torch.Tensor) -> object:
    """Call the original Python ``_tensor_key`` with C bypass disabled."""
    saved = _kernel_module._c_tensor_key
    _kernel_module._c_tensor_key = None
    try:

        class _Fake:
            class settings:
                static_shapes = True
                index_dtype = None

        return _kernel_module._tensor_key(_Fake, t)
    finally:
        _kernel_module._c_tensor_key = saved


@pytest.mark.parametrize(
    "shape",
    [
        (1024, 1024),
        (128, 64),
        (4, 8, 16),
    ],
)
def test_c_tensor_key_shapes(shape: tuple[int, ...]) -> None:
    """Same shape, default dtype: C key matches Python key."""
    t = torch.empty(shape, dtype=torch.float32, device="cpu")
    assert _helion_c.tensor_key(t) == _py_tensor_key(t)


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float32, torch.int64],
)
def test_c_tensor_key_dtypes(dtype: torch.dtype) -> None:
    """Common dtypes round-trip equally in the C and Python paths."""
    t = torch.empty((1024, 1024), dtype=dtype, device="cpu")
    assert _helion_c.tensor_key(t) == _py_tensor_key(t)


def test_c_tensor_key_non_contiguous_strides() -> None:
    """Strides that don't match a contiguous layout must be reflected."""
    t = torch.empty((16, 32), dtype=torch.float32, device="cpu")
    sliced = t[:, ::2]  # stride (32, 2)
    c_key = _helion_c.tensor_key(sliced)
    py_key = _py_tensor_key(sliced)
    assert c_key == py_key
    # Sanity: stride is preserved.
    assert c_key[2] == (32, 2)  # type: ignore[index]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_kernel_bind_parity_contiguous() -> None:
    """Kernel.bind() returns the same BoundKernel with and without C."""
    x = torch.empty(1024, 1024, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(1024, 1024, dtype=torch.bfloat16, device="cuda")

    # Warm the cache with the default path (C on).
    bk_c = _identity_two.bind((x, y))

    # Disable the C path and re-bind — should hit the same entry.
    saved = _kernel_module._c_tensor_key
    _kernel_module._c_tensor_key = None
    try:
        bk_py = _identity_two.bind((x, y))
    finally:
        _kernel_module._c_tensor_key = saved

    assert bk_c is bk_py


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_kernel_bind_parity_view() -> None:
    """Bind parity must hold for views (non-contiguous tensors)."""
    base = torch.empty(2048, 1024, dtype=torch.bfloat16, device="cuda")
    x = base[:1024]  # contiguous view but with different storage
    y = base[1024:]

    bk_c = _identity_two.bind((x, y))

    saved = _kernel_module._c_tensor_key
    _kernel_module._c_tensor_key = None
    try:
        bk_py = _identity_two.bind((x, y))
    finally:
        _kernel_module._c_tensor_key = saved

    assert bk_c is bk_py


def test_c_tensor_key_fallback_for_non_tensor() -> None:
    """Non-tensor inputs return ``None`` so callers fall back."""
    assert _helion_c.tensor_key("not a tensor") is None
    assert _helion_c.tensor_key(42) is None
    assert _helion_c.tensor_key(None) is None
    assert _helion_c.tensor_key([1, 2, 3]) is None
    # Sanity: the C launcher type is exposed and directly callable so the
    # Python fast launcher can install it straight into kwdefaults.
    assert isinstance(_helion_c.CompiledLauncher, type)
    assert callable(_helion_c.CompiledLauncher)
