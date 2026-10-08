from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import Any
import warnings

from _pytest._io.saferepr import saferepr
import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

# The subtest parameters a worker sent as their repr (see below).
_REPR_KWARGS_KEY = "_helion.subtest_repr_kwargs"
_EXECNET_SCALARS = (type(None), bool, int, float, complex, str, bytes)


def _execnet_serializable(value: object) -> bool:
    """Whether execnet, which dispatches on the exact type, can send ``value``."""
    if type(value) in _EXECNET_SCALARS:
        return True
    if isinstance(value, (list, tuple, set, frozenset)):
        return type(value) in (list, tuple, set, frozenset) and all(
            _execnet_serializable(item) for item in value
        )
    if isinstance(value, dict):
        return type(value) is dict and all(
            _execnet_serializable(key) and _execnet_serializable(item)
            for key, item in value.items()
        )
    return False


class _SubtestParameterRepr(str):
    """A subtest parameter received as its repr, displayed as the object was."""

    __slots__ = ()

    def __repr__(self) -> str:
        return str(self)


@pytest.hookimpl(wrapper=True)
def pytest_report_to_serializable(
    config: pytest.Config, report: pytest.TestReport
) -> Generator[None, dict[str, Any] | None, dict[str, Any] | None]:
    # A subtest report carries its ``subTest(...)`` keyword arguments as they
    # were passed (a dtype, a Config, a sympy Integer), and a pytest-xdist
    # worker sends the report through execnet, which serializes only builtin
    # types: the send raised DumpError and failed the test.  Send each such
    # parameter as its repr, which is all the report shows of it.
    data = yield
    context = data.get("_subtest.context") if data is not None else None
    if context is not None:
        kwargs = context["kwargs"]
        unsendable = [
            key for key, value in kwargs.items() if not _execnet_serializable(value)
        ]
        if unsendable:
            context["kwargs"] = {
                key: saferepr(value) if key in unsendable else value
                for key, value in kwargs.items()
            }
            data[_REPR_KWARGS_KEY] = unsendable
    return data


@pytest.hookimpl(wrapper=True)
def pytest_report_from_serializable(
    config: pytest.Config, data: dict[str, Any]
) -> Generator[None, pytest.TestReport | None, pytest.TestReport | None]:
    keys = data.pop(_REPR_KWARGS_KEY, ())
    report = yield
    if keys:
        # A subtest report (pytest's own, or pytest-subtests' before pytest 9).
        kwargs = report.context.kwargs  # pyrefly: ignore [missing-attribute]
        for key in keys:
            kwargs[key] = _SubtestParameterRepr(kwargs[key])
    return report


def pytest_configure() -> None:
    # The final-verification rebench re-times the top configs for a full 5s each
    # by default (HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS=5000), which alone
    # pushes the autotuner tests well past the suite's 60s per-test timeout and
    # gets their xdist worker killed. Clamp it to the floor (200ms) so the step
    # still runs (and stays covered) without dominating the test runtime.
    os.environ.setdefault("HELION_AUTOTUNE_FINAL_REBENCHMARK_TARGET_MS", "200")
    # The same pass re-times the top 8 configs (32 for cute); every timed
    # iteration also zeroes the 256 MiB benchmark cache. Two finalists keep
    # the pass covered while it stays cheap on runners that share one GPU
    # between xdist workers. Tests of the default clear this key first.
    os.environ.setdefault("HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K", "2")

    # The device-us re-rank needs the TPU profiler plane; under interpret
    # (CPU) it burns ~100 traced calls per candidate just to return inf.
    if os.environ.get("HELION_PALLAS_INTERPRET") == "1":
        os.environ.setdefault("HELION_AUTOTUNE_PALLAS_RANK_BY", "wall_time")

    # TODO(tcombes): remove this once Pallas RNG generation avoids int64.
    # JAX x64 is disabled on TPU, so RNG-generated int64s are truncated and
    # spam Pallas test logs with one warning per generated statement.
    warnings.filterwarnings(
        "ignore",
        message=(
            "Explicitly requested dtype int64 requested in .* is not available, "
            "and will be truncated to dtype int32.*"
        ),
        category=UserWarning,
    )
