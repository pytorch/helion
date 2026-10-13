from __future__ import annotations

import ast
import inspect
from pathlib import Path
import signal
import textwrap
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import pytest
import torch

from helion.runtime.cute import launcher
from helion.runtime.cute import native_pool as pool
from helion.runtime.cute import native_required as required
from helion.runtime.kernel import BoundKernel
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence


def _bound() -> BoundKernel:
    bound: Any = object.__new__(BoundKernel)
    bound.kernel = SimpleNamespace(
        settings=SimpleNamespace(
            backend="cute",
            static_shapes=True,
            autotune_cute_native_workers=2,
            autotune_cute_native_required=True,
            autotune_compile_timeout=600,
            autotune_benchmark_subprocess=False,
        )
    )
    bound._env = SimpleNamespace(
        process_group_name=None,
        cute_fission_plan=None,
        cute_resolved_wrapper_plans=(),
        device=torch.device("cuda:0"),
    )
    return bound


def _requests(tmp_path: Path, count: int) -> list[Any]:
    requests = []
    for index in range(count):
        directory = tmp_path / str(index)
        directory.mkdir()
        request = SimpleNamespace(
            spec={"cache_key": str(index), "source_sha256": f"source-{index}"},
            checks=0,
        )

        def check(request: SimpleNamespace = request) -> None:
            request.checks += 1

        request.check = check
        request.compiled = SimpleNamespace(
            _compiled=None,
            _cache_file_paths=lambda directory=directory: (
                str(directory),
                str(directory / "ir"),
                str(directory / "metadata"),
            ),
            _reload_from_disk=lambda: object(),
        )
        requests.append(request)
    return requests


def _capture(monkeypatch: pytest.MonkeyPatch, requests: list[Any]) -> BoundKernel:
    bound = _bound()

    def capture(
        b: BoundKernel,
        config: int,
        fn: Callable[..., object],
        args: Sequence[object],
        *,
        include_cached: bool = False,
    ) -> SimpleNamespace:
        assert b is bound and include_cached and args == ("original",)
        return requests[config]

    monkeypatch.setattr(pool, "capture", capture)
    monkeypatch.setattr(required.signal, "getsignal", lambda s: signal.SIG_DFL)
    return bound


def _invoke(bound: BoundKernel, count: int) -> dict[str, Any]:
    return pool.warm_native_batch(
        bound,
        cast("Any", list(range(count))),
        [lambda: None] * count,
        ("original",),
        budget_exceeded=lambda: False,
    )


def _complete(selected: list[Any], stats: dict[str, Any]) -> None:
    for request in selected:
        request.compiled._compiled = object()
    stats.update(
        status="warmed",
        workers=min(2, len(selected)),
        reaps=[{"index": 0, "pid": 100, "exitcode": 0}],
        processes=[
            {
                "completed_requests": [
                    {"cache_key": r.spec["cache_key"]} for r in selected
                ]
            }
        ],
        compile_failures=[],
        verified_requests=len(selected),
        reloads=len(selected),
    )


def test_required_setting_default_and_bool_validation() -> None:
    assert Settings().autotune_cute_native_required is False
    assert Settings(autotune_cute_native_required=True).autotune_cute_native_required
    for value in (1, "true", None):
        with pytest.raises(ValueError, match="must be a boolean"):
            Settings(autotune_cute_native_required=cast("Any", value))


@pytest.mark.parametrize("terminal", ["timeout", "error"])
def test_only_cancelled_peers_get_one_singleton_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    terminal: str,
) -> None:
    requests = _requests(tmp_path, 4)
    bound = _capture(monkeypatch, requests)
    calls = []

    def run(
        b: BoundKernel,
        selected: list[Any],
        stats: dict[str, Any],
        *,
        budget_exceeded: Callable[[], bool],
    ) -> None:
        assert b is bound and b.settings.autotune_compile_timeout == 600
        calls.append([r.spec["cache_key"] for r in selected])
        if len(calls) == 1:
            _complete([requests[1], requests[3]], stats)
            stats["workers"] = 2
            stats["reaps"] = [
                {"index": 0, "pid": 100, "exitcode": -15},
                {"index": 1, "pid": 101, "exitcode": 0},
            ]
            if terminal == "timeout":
                # Worker0 owns0,2. Its second request timed out;0's killed-worker
                # checkpoint is untrusted and must be retried, never published.
                stats["request_timeout"] = {"worker": 0, "request": 1, "pid": 100}
            else:
                stats["reaps"][0]["exitcode"] = 0
                stats["compile_failures"] = [{"cache_key": "0"}]
        else:
            assert len(selected) == 1
            _complete(selected, stats)

    monkeypatch.setattr(pool, "_compile_native_requests", run)
    result = _invoke(bound, 4)
    assert calls == [["0", "1", "2", "3"], ["0" if terminal == "timeout" else "2"]]
    statuses = [row["status"] for row in result["native_results"]]
    assert statuses == (
        ["ok", "ok", "timeout", "ok"]
        if terminal == "timeout"
        else ["error", "ok", "ok", "ok"]
    )
    assert all(r.checks for r in requests)
    assert len(result["rounds"]) == 2


def test_retry_timeout_is_terminal_without_second_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requests = _requests(tmp_path, 2)
    bound = _capture(monkeypatch, requests)
    calls = []

    def run(
        b: BoundKernel, selected: list[Any], stats: dict[str, Any], **kw: object
    ) -> None:
        calls.append([r.spec["cache_key"] for r in selected])
        stats.update(
            workers=len(selected),
            reaps=[],
            processes=[],
            compile_failures=[],
            verified_requests=0,
            reloads=0,
            request_timeout={"worker": 0, "request": 0},
        )

    monkeypatch.setattr(pool, "_compile_native_requests", run)
    result = _invoke(bound, 2)
    assert calls == [["0", "1"], ["1"]]
    assert [row["status"] for row in result["native_results"]] == ["timeout", "timeout"]


def test_cache_hit_survives_and_cold_singleton_is_supported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requests = _requests(tmp_path, 2)
    requests[0].compiled._compiled = object()
    bound = _capture(monkeypatch, requests)
    calls = []

    def run(
        b: BoundKernel, selected: list[Any], stats: dict[str, Any], **kw: object
    ) -> None:
        calls.append([r.spec["cache_key"] for r in selected])
        _complete(selected, stats)

    monkeypatch.setattr(pool, "_compile_native_requests", run)
    result = _invoke(bound, 2)
    assert calls == [["1"]] and result["cache_hits"] == 1
    assert required.provider_statuses(result, 2) == ([True, True], ["ok", "ok"])


@pytest.mark.parametrize("failure", ["capture", "reload", "worker", "verified_reload"])
def test_required_boundary_failures_are_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    requests = _requests(tmp_path, 1)
    bound = _capture(monkeypatch, requests)
    if failure == "capture":
        monkeypatch.setattr(pool, "capture", lambda *args, **kwargs: None)
    elif failure == "reload":
        for path in requests[0].compiled._cache_file_paths()[1:]:
            Path(path).write_text("present")
        requests[0].compiled._reload_from_disk = lambda: None

    def run(
        b: BoundKernel, selected: list[Any], stats: dict[str, Any], **kw: object
    ) -> None:
        if failure == "worker":
            raise ValueError("original worker authority error")
        _complete(selected, stats)
        selected[0].compiled._compiled = None

    monkeypatch.setattr(pool, "_compile_native_requests", run)
    with pytest.raises(required.RequiredNativePrecompileError):
        _invoke(bound, 1)


def test_source_aliases_keep_all_original_result_positions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requests = _requests(tmp_path, 2)
    requests[1].spec = dict(requests[0].spec)
    bound = _capture(monkeypatch, requests)
    calls = []

    def run(
        b: BoundKernel, selected: list[Any], stats: dict[str, Any], **kw: object
    ) -> None:
        calls.append(selected)
        _complete(selected, stats)

    monkeypatch.setattr(pool, "_compile_native_requests", run)
    result = _invoke(bound, 2)
    assert len(calls) == 1 and calls[0] == [requests[0]]
    assert [row["index"] for row in result["native_results"]] == [0, 1]
    assert result["requests"] == 1
    assert requests[1].compiled._compiled is not None


def test_actual_provider_required_status_branch() -> None:
    from helion.autotuner.benchmark_provider import LocalBenchmarkProvider

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(LocalBenchmarkProvider.benchmark))
    )
    method = tree.body[0]
    assert isinstance(method, ast.FunctionDef)
    node = next(
        n
        for n in method.body
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "self.settings.autotune_cute_native_required"
    )
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace: dict[str, Any] = {
        "__package__": "helion.autotuner",
        "self": SimpleNamespace(
            settings=SimpleNamespace(autotune_cute_native_required=True)
        ),
        "native_record": {
            "native_results": [
                {"index": 0, "status": "ok"},
                {"index": 1, "status": "timeout"},
                {"index": 2, "status": "error"},
            ]
        },
        "configs": [0, 1, 2],
    }
    exec(compile(module, "<actual-provider-native-status>", "exec"), namespace)
    assert namespace["is_workings"] == [True, False, False]
    assert namespace["precompile_status"] == ["ok", "timeout", "error"]
    namespace["self"].settings = SimpleNamespace(
        autotune_cute_native_required=False, autotune_precompile=None
    )
    exec(compile(module, "<actual-provider-native-status>", "exec"), namespace)
    assert namespace["is_workings"] == [True, True, True]
    assert namespace["precompile_status"] == ["ok", "ok", "ok"]


def _actual_launcher_call() -> Callable[..., object]:
    tree = ast.parse(Path(launcher.__file__).read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "_CompiledCuteLauncher"
    )
    node = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"
    )
    namespace = dict(vars(launcher))
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            "<actual-cache-call>",
            "exec",
        ),
        namespace,
    )
    return namespace["__call__"]


def test_actual_launcher_cache_gate_and_restoration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cutlass.cute as cute

    calls = []
    monkeypatch.setattr(
        cute,
        "compile",
        lambda *args, **kwargs: calls.append(True) or (lambda *args: "compiled"),
    )
    monkeypatch.setattr(
        launcher._CompiledCuteLauncher, "_reload_from_disk", lambda self: None
    )
    monkeypatch.setattr(
        launcher._CompiledCuteLauncher, "_persist_to_disk", lambda *args: None
    )
    call = _actual_launcher_call()
    compiled = launcher._CompiledCuteLauncher(object(), None, "key")
    with (
        pytest.raises(required.RequiredNativePrecompileError),
        required.require_cached_native(True),
        pytest.raises(required.RequiredNativePrecompileError),
    ):
        # Even swallowing the immediate error cannot turn it into a score.
        call(compiled)
    assert not calls and launcher._REQUIRED_NATIVE_CACHE_MISSES.get() is None
    with required.require_cached_native(False):
        assert call(compiled) == "compiled"
    assert calls == [True]
    with required.require_cached_native(True):
        assert call(compiled) == "compiled"
    assert calls == [True]


def test_cache_gate_restores_after_original_exception() -> None:
    failure = RuntimeError("original")
    with pytest.raises(RuntimeError) as observed, required.require_cached_native(True):
        raise failure
    assert observed.value is failure
    assert launcher._REQUIRED_NATIVE_CACHE_MISSES.get() is None


def test_required_budget_exhaustion_is_not_a_candidate_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requests = _requests(tmp_path, 1)
    bound = _capture(monkeypatch, requests)

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Expired budget must not start another worker")

    monkeypatch.setattr(pool, "_compile_native_requests", unexpected)
    with pytest.raises(required.RequiredNativePrecompileError, match="Budget expired"):
        pool.warm_native_batch(
            bound,
            cast("Any", [0]),
            [lambda: None],
            ("original",),
            budget_exceeded=lambda: True,
        )


@pytest.mark.parametrize("unsupported", ["workers", "subprocess", "dynamic"])
def test_required_unsupported_invocations_cannot_fall_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, unsupported: str
) -> None:
    requests = _requests(tmp_path, 1)
    bound = _capture(monkeypatch, requests)
    if unsupported == "workers":
        bound.settings.autotune_cute_native_workers = 0
    elif unsupported == "subprocess":
        bound.settings.autotune_benchmark_subprocess = True
    else:
        bound.settings.static_shapes = False
    with pytest.raises(required.RequiredNativePrecompileError, match="unsupported"):
        _invoke(bound, 1)
    assert requests[0].checks == 0
