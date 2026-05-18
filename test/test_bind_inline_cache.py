"""Tests for the monomorphic bind inline cache.

The cache lives on ``Kernel`` and stores the last
``(args_tuple, bound_kernel, run_callable)`` triple. On the hot path,
``Kernel.__call__`` and ``Kernel.bind`` short-circuit to the cached
``BoundKernel`` if the new args interchangeably match the cached ones —
identity for tensors, equality for non-tensors. These tests pin down
the integration invariants of the cache update and refresh logic.
"""

from __future__ import annotations

import pytest
import torch

import helion
import helion.language as hl
from helion.runtime.settings import _get_backend

# Inline-cache end-to-end tests execute kernels via the Triton launcher;
# skip the file under non-Triton backends.
pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for kernel execution"
    ),
    pytest.mark.skipif(
        _get_backend() != "triton",
        reason="inline cache tests exercise the Triton kernel launch path",
    ),
]


@helion.kernel(static_shapes=True)
def _add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


def test_inline_cache_populated_after_first_call() -> None:
    """Calling a kernel populates ``_bind_inline_cache``."""
    _add_kernel.reset()
    assert _add_kernel._bind_inline_cache is None
    x = torch.randn(64, device="cuda")
    y = torch.randn(64, device="cuda")
    _add_kernel(x, y)
    ic = _add_kernel._bind_inline_cache
    assert ic is not None
    cached_args, cached_bound, cached_run = ic
    assert cached_args[0] is x
    assert cached_args[1] is y
    assert cached_bound is not None
    # ``cached_run`` should be the compiled wrapper, populated by ``set_config``.
    assert cached_run is not None


def test_inline_cache_hit_returns_same_bound_kernel() -> None:
    """Repeated calls with same args reuse the same BoundKernel object."""
    _add_kernel.reset()
    x = torch.randn(64, device="cuda")
    y = torch.randn(64, device="cuda")
    _add_kernel(x, y)
    first_bound = _add_kernel._bind_inline_cache[1]
    for _ in range(5):
        _add_kernel(x, y)
    second_bound = _add_kernel._bind_inline_cache[1]
    assert first_bound is second_bound


def test_inline_cache_correctness_repeated() -> None:
    """Numerical correctness across many calls."""
    _add_kernel.reset()
    x = torch.randn(64, device="cuda")
    y = torch.randn(64, device="cuda")
    for _ in range(8):
        out = _add_kernel(x, y)
        torch.testing.assert_close(out, x + y)


def test_inline_cache_swap_tensors_same_shape() -> None:
    """Calling with a different tensor of the same shape doesn't break:
    we miss the inline cache, fall through to the full bind path (which
    hits the BoundKernel dict cache), and produce correct output. The
    inline cache then refreshes to the new args."""
    _add_kernel.reset()
    x1 = torch.randn(64, device="cuda")
    y1 = torch.randn(64, device="cuda")
    x2 = torch.randn(64, device="cuda")
    y2 = torch.randn(64, device="cuda")

    out1 = _add_kernel(x1, y1)
    torch.testing.assert_close(out1, x1 + y1)
    bound1 = _add_kernel._bind_inline_cache[1]

    out2 = _add_kernel(x2, y2)
    torch.testing.assert_close(out2, x2 + y2)
    bound2 = _add_kernel._bind_inline_cache[1]

    # Same shape ⇒ same BoundKernel (resolved through the slow bind dict).
    assert bound1 is bound2
    # But the inline-cache args slot now points at the latest tensors.
    cached_args = _add_kernel._bind_inline_cache[0]
    assert cached_args[0] is x2
    assert cached_args[1] is y2


def test_reset_clears_inline_cache() -> None:
    """``Kernel.reset()`` wipes the inline cache so the next call recompiles."""
    _add_kernel.reset()
    x = torch.randn(64, device="cuda")
    y = torch.randn(64, device="cuda")
    _add_kernel(x, y)
    assert _add_kernel._bind_inline_cache is not None
    _add_kernel.reset()
    assert _add_kernel._bind_inline_cache is None


def test_set_config_refreshes_inline_cache_run_slot() -> None:
    """When ``set_config`` re-pins ``_run`` on the cached BoundKernel, the
    inline cache's third slot (``_run`` shortcut) must update so the next
    ``Kernel.__call__`` short-circuit dispatches the fresh compiled wrapper.
    """
    _add_kernel.reset()
    x = torch.randn(64, device="cuda")
    y = torch.randn(64, device="cuda")
    _add_kernel(x, y)
    original_run = _add_kernel._bind_inline_cache[2]
    assert original_run is not None

    bound = _add_kernel._bind_inline_cache[1]
    # Re-pin set_config with the same config to force ``_run`` reassignment.
    bound.set_config(bound._config)
    refreshed_run = _add_kernel._bind_inline_cache[2]
    # ``set_config`` produces a fresh compiled callable each time; the inline
    # cache slot should track it so ``Kernel.__call__`` doesn't dispatch the
    # stale closure.
    assert refreshed_run is bound._run
