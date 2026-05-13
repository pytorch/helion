"""Pytest plugin used by ``scripts/cute_autotune_sweep.py``.

Drives an existing test case through the *autotune* code path instead of
``ConfigSpec.default_config()``. The plugin patches
``helion._testing.code_and_output`` and ``helion._testing.output_only``
so each test invocation runs the kernel under
``bound.autotune(args, force=True)`` instead of compiling the spec
default. The selected config and a small set of generated-code markers
are written to ``HELION_AUTOTUNE_SWEEP_RESULT_JSON``.

The patch only fires on tests that go through ``check_example`` /
``code_and_output`` / ``output_only``. Tests that build a
``helion.Config(...)`` and call the kernel directly (e.g.
``test_matmul_bwd``, ``test_addmm_bwd``, ``test_split_k_barrier_accuracy``)
bypass these helpers and would run their forced config rather than
autotune; the sweep driver's curated node-ID list intentionally
excludes them.

``@skipIfCute`` tests are skipped at pytest collection. ``@xfailIfCute``
tests *run* — pytest only records the xfail outcome after the test
body has executed, so a CuTe-unsupported example can still spend GPU
time and crash the context. The sweep driver's curated list keeps
``@xfailIfCute`` cases out so the sweep does not chase known-bad paths.

Loaded with ``pytest -p scripts.cute_autotune_sweep_plugin`` from the
sweep harness; not part of the ordinary test suite.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

import pytest


def _scan_codegen_markers(code: str) -> dict[str, bool]:
    """Scan a generated kernel for the canonical CuTe codegen markers."""
    return {
        "uses_tcgen05": "cute.nvgpu.tcgen05" in code,
        "uses_tcgen05_two_cta": "CtaGroup.TWO" in code,
        "uses_tma_umma_pipeline": "PipelineTmaUmma" in code,
    }


def _summarize_config(config: object) -> dict[str, Any]:
    """Reduce a helion.Config to a JSON-stable dict."""
    cfg = getattr(config, "config", None)
    if not isinstance(cfg, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = repr(v)
    return out


@pytest.fixture(autouse=True)
def _patch_helion_check_example_to_autotune(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patch ``code_and_output`` / ``output_only`` to autotune each test.

    The original helpers call ``_bound_test_config(bound, **kwargs)`` which
    returns either the user-provided kwargs config or the spec default
    (``default_config``) — neither path autotunes. The replacements below
    delegate to ``bound.autotune(args, force=True)`` and run the bound
    kernel against the resulting config, ignoring the per-test ``**kwargs``
    (e.g. ``block_sizes=...``) so we exercise the autotune surface
    directly. Tests that build a ``helion.Config(...)`` themselves and
    pass it via ``bound.compile_config`` (e.g. ``test_split_k_barrier_accuracy``)
    bypass these helpers and are unaffected.

    Stores the autotune outcome on the pytest item so
    ``pytest_runtest_logreport`` can record it.
    """
    sweep_state: dict[str, Any] = {}
    request.node.cute_sweep_state = sweep_state

    from helion import _testing as ht

    # ``_run_bound_kernel`` is an underscore-prefixed helper (private
    # by convention) but it is the exact code path the original
    # ``code_and_output`` / ``output_only`` use to compile and execute
    # a bound kernel against a config. Re-implementing it here would
    # silently drift when ``helion._testing`` changes its error
    # reporting / sync logic; reach-in is intentional for this dev
    # script.
    from helion._testing import _run_bound_kernel
    from helion._testing import is_ref_mode_enabled

    def _autotune_and_run(
        bound: Any,  # noqa: ANN401  -- BoundKernel is internal/dynamic
        args: tuple[Any, ...],
        emit_code: bool,
    ) -> tuple[str | None, Any]:
        try:
            t0 = time.monotonic()
            config = bound.autotune(args, force=True)
        except Exception as exc:
            sweep_state.setdefault("autotune_errors", []).append(
                f"{type(exc).__name__}: {exc}"
            )
            raise
        sweep_state["autotune_seconds"] = time.monotonic() - t0
        sweep_state["selected_config"] = _summarize_config(config)
        try:
            code_for_markers = bound.to_triton_code(config)
        except Exception as exc:
            # Record the marker-scan failure so the JSONL distinguishes
            # "marker scan failed" from "no marker present". Without
            # this the sweep reports "0 tcgen05 hits" for both cases
            # and a codegen / to_triton_code bug hides as a sanity
            # miss rather than a recorded failure.
            sweep_state["codegen_marker_scan_error"] = f"{type(exc).__name__}: {exc}"
            sweep_state["codegen_markers"] = None
        else:
            sweep_state["codegen_markers"] = _scan_codegen_markers(code_for_markers)
        return _run_bound_kernel(bound, args, config, emit_code=emit_code)

    def _patched_code_and_output(
        fn: Any,  # noqa: ANN401  -- helion.Kernel
        args: tuple[Any, ...],
        **_kwargs: Any,  # noqa: ANN401  -- swallow per-test config kwargs
    ) -> tuple[str, Any]:
        bound = fn.bind(args)
        if is_ref_mode_enabled(bound.kernel.settings):
            return _orig_code_and_output(fn, args, **_kwargs)
        code, result = _autotune_and_run(bound, args, emit_code=True)
        assert code is not None
        return code, result

    def _patched_output_only(
        fn: Any,  # noqa: ANN401  -- helion.Kernel
        args: tuple[Any, ...],
        **_kwargs: Any,  # noqa: ANN401  -- swallow per-test config kwargs
    ) -> Any:  # noqa: ANN401  -- arbitrary kernel return
        bound = fn.bind(args)
        if is_ref_mode_enabled(bound.kernel.settings):
            return _orig_output_only(fn, args, **_kwargs)
        _code, result = _autotune_and_run(bound, args, emit_code=False)
        return result

    _orig_code_and_output = ht.code_and_output
    _orig_output_only = ht.output_only
    monkeypatch.setattr(ht, "code_and_output", _patched_code_and_output)
    monkeypatch.setattr(ht, "output_only", _patched_output_only)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Append a JSON record per test to ``HELION_AUTOTUNE_SWEEP_RESULT_JSON``.

    Only the ``call`` phase is recorded; setup/teardown phases would
    duplicate entries. Non-passing outcomes still record ``outcome`` plus
    the captured longrepr so the harness can grep across runs.
    """
    if report.when != "call":
        return
    out_path = os.environ.get("HELION_AUTOTUNE_SWEEP_RESULT_JSON")
    if not out_path:
        return
    sweep_state: dict[str, Any] = getattr(report, "_cute_sweep_state", {}) or {}
    record: dict[str, Any] = {
        "nodeid": report.nodeid,
        "outcome": report.outcome,  # "passed" | "failed" | "skipped"
        "duration_seconds": report.duration,
        "selected_config": sweep_state.get("selected_config"),
        "codegen_markers": sweep_state.get("codegen_markers"),
        "codegen_marker_scan_error": sweep_state.get("codegen_marker_scan_error"),
        "autotune_seconds": sweep_state.get("autotune_seconds"),
        "autotune_errors": sweep_state.get("autotune_errors"),
    }
    if report.outcome != "passed":
        longrepr = getattr(report, "longrepr", None)
        record["error_summary"] = str(longrepr)[:4000] if longrepr else None
    with Path(out_path).open("a") as fh:
        fh.write(json.dumps(record) + "\n")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> None:
    """Pipe per-test sweep state through to ``pytest_runtest_logreport``.

    Under ``hookwrapper=True`` pytest reads the result from
    ``outcome.get_result()`` after the inner ``yield``; the wrapper
    must not return a value.
    """
    outcome = yield
    report: pytest.TestReport = outcome.get_result()
    if call.when == "call":
        report._cute_sweep_state = getattr(item, "cute_sweep_state", {})  # type: ignore[attr-defined]
