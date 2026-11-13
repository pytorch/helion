"""CPU stand-ins for the CuTe launcher's pointer/schema tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from helion.runtime.cute import launcher

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest


def mock_cute_launcher_runtime(
    monkeypatch: pytest.MonkeyPatch,
    make_ptr: Callable[..., object],
    *,
    constexpr: tuple[bool, ...] = (),
) -> None:
    """Keep real tensor metadata while replacing the CUDA pointer adapter."""
    monkeypatch.setattr(
        launcher, "_get_cute_launcher_imports", lambda: (None, make_ptr, None)
    )
    monkeypatch.setattr(launcher, "_validate_cute_launcher_tensor", lambda tensor: None)
    monkeypatch.setattr(launcher, "_torch_dtype_to_cutlass", str)
    monkeypatch.setattr(
        launcher, "_cute_kernel_param_is_constexpr", lambda kernel: constexpr
    )
