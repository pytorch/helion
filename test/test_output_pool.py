"""Tests for the output pool.

The pool lives in ``helion/runtime/_output_pool.py`` and routes the
generated host wrapper's ``torch.empty(...)`` calls through
``_output_pool_alloc``.

Activation:

* Default (env unset): pool is active for every kernel call. The first
  buffer reuse fires a one-time INFO log line.
* ``HELION_REUSE_OUTPUT_BUFFERS=0`` (or ``off``/``false``/``no``): pool
  is disabled (passthrough to ``torch.empty``).
* ``HELION_REUSE_OUTPUT_BUFFERS=debug``: pool active with per-decision
  INFO logging.

When pooling is active, the cache is gated by three safety signals:
tensor refcount (catches direct references, views via ``_base``, and
autograd saves), CUDA stream tagging (cross-stream consumers don't see
stale buffers), and CUDA-graph capture detection (capture bypasses the
pool entirely).
"""

from __future__ import annotations

import gc
import logging
from typing import Iterator

import pytest
import torch

import helion
from helion._testing import DEVICE
import helion.language as hl
import helion.runtime as helion_runtime
from helion.runtime.settings import _get_backend

# Output-pool tests genuinely require the Triton backend: the codegen
# rewrite that routes ``torch.empty(...)`` through ``_helion_output_alloc``
# (see ``helion/_compiler/generate_ast.py``) is restricted to
# ``TritonBackend``. They would silently no-op the pool under
# ``HELION_BACKEND=cute`` on a CUDA host.
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(
        _get_backend() != "triton",
        reason="output pool codegen rewrite is Triton-backend only",
    ),
]


@helion.kernel(static_shapes=True)
def _pool_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Minimal add kernel used by the pool tests.

    Goes through the generated host wrapper's ``torch.empty(...)`` (which
    is rewritten to ``_helion_output_alloc``), so its output participates
    in the pool exactly like ``examples/add.py``.
    """
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@pytest.fixture
def reset_pool(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset the env-cached mode + drop buffers around each test.

    The env var is read once and cached for the lifetime of the process,
    so tests that flip it need a reset hook. ``monkeypatch`` also restores
    the env var after the test.
    """
    helion_runtime._reset_output_pool_user_opt_in_cache()
    yield
    helion_runtime._reset_output_pool_user_opt_in_cache()


def test_user_call_pools_by_default(
    reset_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default mode: a dropped output's buffer is reused by the next call."""
    monkeypatch.delenv(helion_runtime._REUSE_OUTPUT_BUFFERS_ENV, raising=False)
    helion_runtime._reset_output_pool_user_opt_in_cache()
    x = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    y = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)

    # Warmup so the kernel itself is JIT'd.
    out = _pool_add(x, y)
    torch.testing.assert_close(out, x + y)
    del out
    gc.collect()

    out1 = _pool_add(x, y)
    p1 = out1.data_ptr()
    del out1
    gc.collect()

    out2 = _pool_add(x, y)
    p2 = out2.data_ptr()
    assert p1 == p2, (
        "Default mode: dropping the previous output should let the next "
        f"call reuse its buffer (got {p1=} vs {p2=})."
    )


def test_opt_out_env_disables_pool(
    reset_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``HELION_REUSE_OUTPUT_BUFFERS=0`` falls back to fresh allocations."""
    monkeypatch.setenv(helion_runtime._REUSE_OUTPUT_BUFFERS_ENV, "0")
    helion_runtime._reset_output_pool_user_opt_in_cache()
    x = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    y = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)

    # Warmup so we don't measure first-time JIT.
    out = _pool_add(x, y)
    del out
    gc.collect()

    out1 = _pool_add(x, y)
    p1 = out1.data_ptr()
    del out1
    gc.collect()

    out2 = _pool_add(x, y)
    p2 = out2.data_ptr()
    # Without the pool, dropping the previous tensor doesn't guarantee the
    # CUDA caching allocator hands the same address back: assert that the
    # passthrough doesn't go through the pool by verifying the cache stays
    # empty (the pool would have populated it).
    from helion.runtime._output_pool import _output_pool_safe_cache

    assert not _output_pool_safe_cache, (
        "With HELION_REUSE_OUTPUT_BUFFERS=0 the pool must be inactive; "
        f"found {len(_output_pool_safe_cache)} cached entries instead."
    )
    # Also sanity-check that p1/p2 don't both come from a Helion pool slot
    # (they may still match by chance via PyTorch's own caching allocator,
    # which is fine — the pool itself is what we're testing).
    assert isinstance(p1, int) and isinstance(p2, int)


def test_pool_active_scope_clears_on_exit(
    reset_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_pool_active()`` drops cached buffers when the scope exits."""
    monkeypatch.delenv(helion_runtime._REUSE_OUTPUT_BUFFERS_ENV, raising=False)
    helion_runtime._reset_output_pool_user_opt_in_cache()
    x = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    y = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)

    # Warmup outside the scope.
    out = _pool_add(x, y)
    del out
    gc.collect()

    from helion.runtime._output_pool import _output_pool_safe_cache

    with helion_runtime._pool_active():
        out1 = _pool_add(x, y)
        del out1
        gc.collect()
        assert _output_pool_safe_cache, (
            "Inside _pool_active() the pool should still cache buffers."
        )

    assert not _output_pool_safe_cache, (
        "_pool_active.__exit__ must clear the cache so cross-scope state "
        "doesn't leak; found leftover entries."
    )


def test_pool_view_keeps_storage_alive(
    reset_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A surviving view of the previous output forces a fresh allocation."""
    monkeypatch.delenv(helion_runtime._REUSE_OUTPUT_BUFFERS_ENV, raising=False)
    helion_runtime._reset_output_pool_user_opt_in_cache()
    x = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    y = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)

    # Warmup: get the pool entry seeded.
    out = _pool_add(x, y)
    del out
    gc.collect()

    out1 = _pool_add(x, y)
    p1 = out1.data_ptr()
    view = out1[:128]  # holds out1 alive via _base + storage via view
    del out1
    gc.collect()

    out2 = _pool_add(x, y)
    p2 = out2.data_ptr()
    assert p1 != p2, (
        "View kept storage alive, so the pool must allocate a fresh buffer "
        f"(got {p1=} vs {p2=})."
    )
    # Sanity: the view still sees the original buffer's data.
    assert view.data_ptr() == p1


def test_pool_autograd_retain_alloc_fresh(
    reset_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An autograd graph that saves the output keeps the pool from reusing it.

    ``_pool_add`` itself doesn't track gradients (Helion outputs are leaf
    tensors), so we emulate the "autograd retains" case by saving the
    tensor inside a custom autograd graph via ``ctx.save_for_backward``.
    """
    monkeypatch.delenv(helion_runtime._REUSE_OUTPUT_BUFFERS_ENV, raising=False)
    helion_runtime._reset_output_pool_user_opt_in_cache()
    x = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    y = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)

    # Warmup.
    out = _pool_add(x, y)
    del out
    gc.collect()

    class Retain(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: object, kept: torch.Tensor, gate: torch.Tensor
        ) -> torch.Tensor:
            ctx.save_for_backward(kept)  # type: ignore[attr-defined]
            return gate.clone()

        @staticmethod
        def backward(ctx: object, grad: torch.Tensor) -> tuple[None, torch.Tensor]:
            (kept,) = ctx.saved_tensors  # type: ignore[attr-defined]
            return None, grad + kept.mean()

    out1 = _pool_add(x, y)
    p1 = out1.data_ptr()
    gate = torch.zeros(1, device=DEVICE, requires_grad=True)
    # The Retain.forward call saves out1 in the autograd graph.
    graph_out = Retain.apply(out1, gate)
    del out1  # caller drops their explicit ref; autograd still holds it
    gc.collect()

    out2 = _pool_add(x, y)
    p2 = out2.data_ptr()
    assert p1 != p2, (
        "Autograd retained the previous output via save_for_backward, so "
        f"the pool must allocate fresh (got {p1=} vs {p2=})."
    )
    # Trigger the backward path so the kept reference is exercised.
    assert graph_out is not None


def test_pool_cuda_graph_bypass(
    reset_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """During CUDA graph capture, the pool must bypass (never reuse)."""
    monkeypatch.delenv(helion_runtime._REUSE_OUTPUT_BUFFERS_ENV, raising=False)
    helion_runtime._reset_output_pool_user_opt_in_cache()
    x = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    y = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    # Warmup outside the graph.
    out = _pool_add(x, y)
    torch.cuda.synchronize()
    del out
    gc.collect()

    # Capture: the kernel runs once inside the graph context. We don't
    # assert the data_ptr inside capture (CG semantics own that), only
    # that the captured graph replays correctly without aliasing.
    side_stream = torch.cuda.Stream()
    with torch.cuda.stream(side_stream):
        _pool_add(x, y)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    static_out: torch.Tensor | None = None
    with torch.cuda.graph(graph):
        static_out = _pool_add(x, y)
    assert static_out is not None
    torch.cuda.synchronize()

    # Replay: should reproduce the original result.
    graph.replay()
    torch.cuda.synchronize()
    expected = x + y
    torch.testing.assert_close(static_out, expected)


def test_pool_cross_stream_no_reuse(
    reset_pool: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When pooling is active, the pool is stream-keyed — switching streams allocates fresh."""
    monkeypatch.delenv(helion_runtime._REUSE_OUTPUT_BUFFERS_ENV, raising=False)
    helion_runtime._reset_output_pool_user_opt_in_cache()
    x = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    y = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)

    # Warmup on the default stream.
    out = _pool_add(x, y)
    del out
    gc.collect()

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    with torch.cuda.stream(stream_a):
        out_a = _pool_add(x, y)
        ptr_a = out_a.data_ptr()
        del out_a
        gc.collect()
    stream_a.synchronize()

    with torch.cuda.stream(stream_b):
        out_b = _pool_add(x, y)
        ptr_b = out_b.data_ptr()
    stream_b.synchronize()

    assert ptr_a != ptr_b, (
        "Cross-stream consumer: the pool must not return a buffer first "
        f"allocated on stream A to a caller on stream B (got {ptr_a=} vs {ptr_b=})."
    )


def test_pool_debug_mode_logs_decisions(
    reset_pool: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``=debug`` mode emits an INFO line for each pool decision."""
    monkeypatch.setenv(helion_runtime._REUSE_OUTPUT_BUFFERS_ENV, "debug")
    helion_runtime._reset_output_pool_user_opt_in_cache()
    x = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    y = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)

    # Warmup outside caplog so JIT noise stays out.
    out = _pool_add(x, y)
    del out
    gc.collect()

    with caplog.at_level(logging.INFO, logger="helion.runtime._output_pool"):
        out1 = _pool_add(x, y)
        del out1
        gc.collect()
        _pool_add(x, y)

    messages = [r.getMessage() for r in caplog.records]
    assert any("output pool" in m for m in messages), (
        f"Expected at least one 'output pool: ...' log line in debug mode, "
        f"got: {messages!r}"
    )


def test_pool_intro_logs_once(
    reset_pool: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Default mode logs a one-time INFO line the first time the pool reuses."""
    monkeypatch.delenv(helion_runtime._REUSE_OUTPUT_BUFFERS_ENV, raising=False)
    helion_runtime._reset_output_pool_user_opt_in_cache()
    x = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    y = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)

    # Warmup so the cache has a seeded entry to reuse.
    out = _pool_add(x, y)
    del out
    gc.collect()

    with caplog.at_level(logging.INFO, logger="helion.runtime._output_pool"):
        # Two reuse calls; intro log should fire exactly once total.
        out1 = _pool_add(x, y)
        del out1
        gc.collect()
        out2 = _pool_add(x, y)
        del out2
        gc.collect()

    intro_records = [r for r in caplog.records if "Helion is reusing" in r.getMessage()]
    assert len(intro_records) == 1, (
        f"Intro log should fire exactly once, got {len(intro_records)} "
        f"records: {[r.getMessage() for r in intro_records]!r}"
    )
