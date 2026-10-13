"""Explicit bounded native admission before an unchanged parent measurement."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import signal
import sys
import threading
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

from . import native_pool as pool

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Sequence

    from ..config import Config
    from ..kernel import BoundKernel


class RequiredNativePrecompileError(RuntimeError):
    """The required compile boundary could not be established safely."""


@contextmanager
def require_cached_native(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    from . import launcher

    misses: list[str] = []
    token = launcher._REQUIRED_NATIVE_CACHE_MISSES.set(misses)
    try:
        yield
    finally:
        launcher._REQUIRED_NATIVE_CACHE_MISSES.reset(token)
    if misses:
        # A normal benchmark exception handler may have consumed the first error.
        raise RequiredNativePrecompileError(f"Required native cache missed: {misses}")


def provider_statuses(
    record: dict[str, Any], count: int
) -> tuple[list[bool], list[Literal["ok", "error", "timeout"]]]:
    rows = record["native_results"]
    if len(rows) != count or [row["index"] for row in rows] != list(range(count)):
        raise RequiredNativePrecompileError("Native results lost positional identity")
    statuses: list[Literal["ok", "error", "timeout"]] = [row["status"] for row in rows]
    if any(status not in ("ok", "error", "timeout") for status in statuses):
        raise RequiredNativePrecompileError("Native results have no terminal status")
    return [status == "ok" for status in statuses], statuses


def _outcomes(
    requests: Sequence[pool.NativeRequest], stats: dict[str, Any]
) -> dict[str, str]:
    by_key = {request.spec["cache_key"]: request for request in requests}
    outcomes: dict[str, str] = {}
    for process in stats["processes"]:
        for completed in process["completed_requests"]:
            key = completed["cache_key"]
            request = by_key[key]
            request.check()
            if request.compiled._compiled is None:
                raise RequiredNativePrecompileError(
                    f"Verified native cache failed to reload: {key}"
                )
            outcomes[key] = "ok"
    for failure in stats["compile_failures"]:
        key = failure["cache_key"]
        if key not in by_key or key in outcomes:
            raise RequiredNativePrecompileError(
                "Native compile failure lost request identity"
            )
        outcomes[key] = "error"
    if "request_timeout" in stats:
        failure = stats["request_timeout"]
        assigned = requests[failure["worker"] :: stats["workers"]]
        key = assigned[failure["request"]].spec["cache_key"]
        if key in outcomes:
            raise RequiredNativePrecompileError(
                "Native timeout conflicts with a terminal result"
            )
        outcomes[key] = "timeout"
    return outcomes


def warm_required_native_batch(
    bound: BoundKernel,
    configs: Sequence[Config],
    fns: Sequence[Callable[..., object]],
    args: Sequence[object],
    *,
    budget_exceeded: Callable[[], bool],
) -> dict[str, Any]:
    from ..kernel import BoundKernel

    if (
        type(bound) is not BoundKernel
        or bound.settings.backend != "cute"
        or not bound.settings.static_shapes
        or bound.env.process_group_name is not None
        or bound.env.cute_fission_plan is not None
        or bound.env.cute_resolved_wrapper_plans
        or bound.env.device.type != "cuda"
        or sys.platform != "linux"
        or sys.flags.optimize
        or threading.current_thread() is not threading.main_thread()
        or signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL
        or not 1 <= bound.settings.autotune_cute_native_workers <= 32
        or bound.settings.autotune_compile_timeout <= 0
        or bound.settings.autotune_benchmark_subprocess
        or len(configs) != len(fns)
    ):
        raise RequiredNativePrecompileError(
            "Required native precompile is unsupported for this invocation"
        )
    started = time.monotonic()
    requests = []
    unique: dict[str, pool.NativeRequest] = {}
    outcomes: dict[str, str] = {}
    for index, (config, fn) in enumerate(zip(configs, fns, strict=True)):
        try:
            request = pool.capture(bound, config, fn, args, include_cached=True)
        except Exception as error:
            raise RequiredNativePrecompileError(
                f"Required native capture failed at candidate {index}"
            ) from error
        if request is None:
            raise RequiredNativePrecompileError(
                f"Required native capture declined candidate {index}"
            )
        request.check()
        requests.append(request)
        key = request.spec["cache_key"]
        unique.setdefault(key, request)
    cache_reloads = 0
    for key, request in unique.items():
        if request.compiled._compiled is not None:
            outcomes[key] = "ok"
            continue
        files = request.compiled._cache_file_paths()[1:]
        present = [Path(path).is_file() for path in files]
        if any(present):
            if not all(present):
                raise RequiredNativePrecompileError(
                    f"Incomplete required native cache: {key}"
                )
            compiled = request.compiled._reload_from_disk()
            if compiled is None:
                raise RequiredNativePrecompileError(
                    f"Required native cache failed to reload: {key}"
                )
            request.compiled._compiled = compiled
            cache_reloads += 1
            outcomes[key] = "ok"
    cache_hits = len(outcomes)
    pending = [request for key, request in unique.items() if key not in outcomes]
    rounds: list[dict[str, Any]] = []

    def run_round(selected: Sequence[pool.NativeRequest]) -> None:
        stats: dict[str, Any] = {
            "status": "started",
            "requests": len(selected),
            "workers": 0,
            "reaps": [],
        }
        rounds.append(stats)
        try:
            pool._compile_native_requests(
                bound, selected, stats, budget_exceeded=budget_exceeded
            )
        except Exception as error:
            raise RequiredNativePrecompileError(
                "Required native worker/authority boundary failed"
            ) from error
        outcomes.update(_outcomes(selected, stats))

    if pending and budget_exceeded():
        raise RequiredNativePrecompileError(
            "Budget expired before required native precompile"
        )
    if pending:
        run_round(pending)
    # The first deadline also drains peers. Never call those unattempted or
    # interrupted requests timeouts: give each exactly one fresh bounded round.
    for request in pending:
        key = request.spec["cache_key"]
        if key in outcomes:
            continue
        if budget_exceeded():
            raise RequiredNativePrecompileError(
                "Budget expired before required native retry"
            )
        run_round([request])
        if key not in outcomes:
            raise RequiredNativePrecompileError(
                f"Bounded retry has no terminal outcome: {key}"
            )
    rows = []
    for index, request in enumerate(requests):
        key = request.spec["cache_key"]
        status = outcomes[key]
        if status == "ok":
            request.check()
            if request.compiled._compiled is None:
                request.compiled._compiled = request.compiled._reload_from_disk()
                if request.compiled._compiled is None:
                    raise RequiredNativePrecompileError(
                        f"Native source alias failed to reload: {key}"
                    )
                cache_reloads += 1
        rows.append(
            {
                "index": index,
                "cache_key": key,
                "source_sha256": request.spec["source_sha256"],
                "status": status,
            }
        )
    return {
        "status": "required_completed",
        "requests": len(unique),
        "cache_hits": cache_hits,
        "workers": max((row["workers"] for row in rounds), default=0),
        "verified_requests": sum(row["verified_requests"] for row in rounds),
        "ready_requests": sum(status == "ok" for status in outcomes.values()),
        "reloads": cache_reloads + sum(row["reloads"] for row in rounds),
        "native_results": rows,
        "rounds": rounds,
        "reaps": [reap for row in rounds for reap in row["reaps"]],
        "processes": [process for row in rounds for process in row["processes"]],
        "wall_seconds": time.monotonic() - started,
    }
