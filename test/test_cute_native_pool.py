from __future__ import annotations

import ast
import ctypes
import gc
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import threading
import time
import traceback
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch
import weakref

import pytest
import torch

from helion.runtime.config import Config
from helion.runtime.cute import launcher
from helion.runtime.cute import native_pool as pool
from helion.runtime.cute.native_pool_metadata import DriverMetadata
from helion.runtime.cute.native_pool_protocol import NativeRequestTimeout
from helion.runtime.cute.native_pool_protocol import freeze
from helion.runtime.cute.native_pool_protocol import run_workers
from helion.runtime.cute.native_pool_protocol import sha
from helion.runtime.cute.native_pool_protocol import thaw
from helion.runtime.cute.native_pool_protocol import write
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.cute.launcher import _CuteLaunchArgCacheEntry
    from helion.runtime.kernel import BoundKernel

LINUX = pytest.mark.skipif(
    sys.platform != "linux", reason="owned POSIX spawned workers"
)
UNBLOCK = "import signal; signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT,signal.SIGTERM}); "


def host(x: object, *, _launcher: Callable[..., object] | None = None) -> object:
    output = torch.empty_like(x)
    view = x.view(x.size(0), -1)
    _launcher(host, (1,), view, output, block=(128, 1, 1))
    return output


def conditional_host(
    x: object, *, _launcher: Callable[..., object] | None = None
) -> object:
    if x.size(0):
        _launcher(host, (1,), x)


def compute_host(
    x: object, *, _launcher: Callable[..., object] | None = None
) -> object:
    output = torch.exp(x)
    _launcher(host, (1,), output)


def two_launch_host(
    x: object, *, _launcher: Callable[..., object] | None = None
) -> object:
    _launcher(host, (1,), x)
    _launcher(host, (1,), x)


def test_settings_default_count_and_bounds() -> None:
    assert Settings().autotune_cute_native_workers == 0
    for value in (0, 1, 2, 32):
        assert (
            Settings(autotune_cute_native_workers=value).autotune_cute_native_workers
            == value
        )
    for value in (True, -1, 33, 1.0, "2"):
        with pytest.raises(ValueError):
            Settings(autotune_cute_native_workers=value)


def test_host_eligibility_and_metadata_only_dispatch() -> None:
    assert pool.host_supported(host)
    for fn in (conditional_host, compute_host, two_launch_host, object()):
        assert not pool.host_supported(fn)
    x = torch.ones((3, 4))
    before = x.clone()
    calls = []
    with pool.MetadataHostOnly():
        result = host(x, _launcher=lambda *args, **kwargs: calls.append(args))
    assert len(calls) == 1 and result.shape == x.shape
    assert torch.equal(x, before)
    for operation in (lambda: x.add_(1), lambda: torch.exp(x), lambda: x.clone()):
        with pool.MetadataHostOnly(), pytest.raises(ValueError, match="computation"):
            operation()
    with patch.dict(host.__globals__, {"len": lambda x: 0}):
        assert not pool.host_supported(host)


def test_codec_preserves_float_bits_and_containers() -> None:
    value = (
        ("tensor", "torch.bfloat16", 2, (3, 4), (4, 1), 2),
        {"a": [-0.0, None, True]},
    )
    restored = thaw(json.loads(json.dumps(freeze(value))))
    assert restored == value and restored[1]["a"][0].hex() == "-0x0.0p+0"
    for unsupported in (object(), float("nan"), {1: 2}):
        with pytest.raises(TypeError):
            freeze(unsupported)


def test_schema_argument_positions() -> None:
    x = torch.empty(2, 4, dtype=torch.bfloat16)
    dynamic = SimpleNamespace(
        schema=(
            ("tensor", "torch.bfloat16", 2, 2),
            ("scalar", "float"),
            ("scalar_constexpr", "int", 7),
        ),
        launch_args=(object(), 2, 4, 4, 1, 2.5, 3, 1, 1),
    )
    result = pool.arguments_for_worker((x, 2.5, 7), dynamic)
    assert result[0] == {"kind": "pointer", "dtype": "BFloat16", "alignment": 2}
    assert [thaw(row["value"]) for row in result[1:]] == [2, 4, 4, 1, 2.5, 3, 1, 1]
    static = SimpleNamespace(
        schema=(("tensor", "torch.bfloat16", 2, (2, 4), (4, 1), 2),),
        launch_args=(object(), 3, 1, 1),
    )
    assert len(pool.arguments_for_worker((x,), static)) == 4
    dynamic.launch_args += (0,)
    with pytest.raises(AssertionError):
        pool.arguments_for_worker((x, 2.5, 7), dynamic)


def test_owner_facts_detect_value_view_storage_mutations() -> None:
    for operation in (
        lambda t: t.add_(1),
        lambda t: t.transpose_(0, 1),
        lambda t: t.set_(torch.empty_like(t)),
    ):
        x = torch.ones(2, 4)
        before = pool.owner_facts(x)
        operation(x)
        assert pool.owner_facts(x) != before
    x = torch.ones(2, 4)
    ref = weakref.ref(x)
    retained = pool.NativeRequest(
        {}, None, (x,), (x.untyped_storage(),), (pool.owner_facts(x),), ()
    )
    del x
    gc.collect()
    assert ref() is retained.owners[0]
    with torch.inference_mode():
        inference = torch.ones(1)
    with pytest.raises(RuntimeError):
        pool.owner_facts(inference)


def test_kernel_metadata_only_ignores_original_memo() -> None:
    kernel = SimpleNamespace(
        _helion_cute_source_hash="abc",
        _helion_cute_compiled_launchers={object(): object()},
    )
    first = pool.kernel_attributes(kernel)
    kernel._helion_cute_compiled_launchers[0] = object()
    assert pool.kernel_attributes(kernel) == first
    kernel._helion_cute_pointer_alignment = 4
    assert pool.kernel_attributes(kernel) != first
    kernel._helion_cute_unknown = object()
    with pytest.raises(TypeError):
        pool.kernel_attributes(kernel)


def test_request_owner_schema_source_environment_revalidation(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("pass\n")
    tensor = torch.ones(2, 4)
    kernel = SimpleNamespace(_helion_cute_source_hash="abc")
    compiled = SimpleNamespace(
        _cache_key="key", _compile_options="opts", _jit_func=object()
    )
    launch = SimpleNamespace(schema=(("x",),))
    spec = {
        "source": str(source),
        "source_sha256": sha(source),
        "environment": freeze(()),
        "cache_key": "key",
        "options": "opts",
        "attributes": pool.kernel_attributes(kernel),
        "schema": freeze(launch.schema),
        "wrapper_ast": "ast",
        "block": [128, 1, 1],
        "num_sm": 152,
    }
    request = pool.NativeRequest(
        spec,
        compiled,
        (tensor,),
        (tensor.untyped_storage(),),
        (pool.owner_facts(tensor),),
        (kernel, launch, (tensor,), host, (tensor,), None),
    )
    with (
        patch.object(launcher, "_cute_cache_relevant_env", return_value=()),
        patch.object(pool, "wrapper_ast", return_value="ast"),
        patch.object(
            launcher,
            "_cute_compiled_launcher_discriminator",
            return_value=((), "opts", 152),
        ),
        patch.object(launcher, "_cute_disk_cache_key", return_value="key"),
    ):
        request.check()
        kernel._helion_cute_source_hash = "changed"
        with pytest.raises(AssertionError):
            request.check()
        kernel._helion_cute_source_hash = "abc"
        tensor.add_(1)
        with pytest.raises(AssertionError):
            request.check()


def test_cache_pair_validation_rejects_before_publication(tmp_path: Path) -> None:
    ir, meta = (tmp_path / "cute_dsl_abc.mlir", tmp_path / "cute_dsl_abc.json")
    ir.write_bytes(b"original cache bytes")
    symbol = {"function_name": "entry", "has_gpu_module": True}
    write(meta, symbol)
    checks = []
    request = SimpleNamespace(
        spec={"cache_key": "abc", "source_sha256": "source"},
        check=lambda: checks.append(1),
    )
    row = {
        "cache_key": "abc",
        "source_sha256": "source",
        "ir": str(ir),
        "metadata": str(meta),
        "ir_sha256": sha(ir),
        "metadata_sha256": sha(meta),
        "symbol": symbol,
    }
    assert pool.verify_result(request, row, tmp_path) == [ir, meta]
    assert checks == [1]
    for change in (
        {"cache_key": "foreign"},
        {"ir_sha256": "wrong"},
        {"metadata": str(ir)},
        {"symbol": {"function_name": "wrong", "has_gpu_module": True}},
    ):
        with pytest.raises(AssertionError):
            pool.verify_result(request, dict(row, **change), tmp_path)
    symbol["has_gpu_module"] = False
    write(meta, symbol)
    row["metadata_sha256"] = sha(meta)
    with pytest.raises(AssertionError):
        pool.verify_result(request, row, tmp_path)


def test_optional_capture_failure_leaves_original_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from helion.runtime.kernel import BoundKernel

    fake = object.__new__(BoundKernel)
    fake.kernel = SimpleNamespace(
        settings=SimpleNamespace(
            backend="cute",
            static_shapes=True,
            autotune_cute_native_workers=2,
            autotune_cute_native_required=False,
            autotune_compile_timeout=1,
        )
    )
    fake._env = SimpleNamespace(
        process_group_name=None,
        cute_fission_plan=None,
        cute_resolved_wrapper_plans=(),
        device=torch.device("cuda:0"),
    )
    original = RuntimeError("original candidate failure")

    def failed_capture(*args: object) -> object:
        raise original

    monkeypatch.setattr(pool, "capture", failed_capture)
    monkeypatch.setattr(pool.sys, "platform", "linux")
    monkeypatch.setattr(pool.signal, "getsignal", lambda s: signal.SIG_DFL)
    result = pool.warm_native_batch(
        fake, [1, 2], [host, host], (), budget_exceeded=lambda: False
    )
    assert result["status"] == "unsupported"
    with pytest.raises(RuntimeError) as failure:
        failed_capture()
    assert failure.value is original
    monkeypatch.setattr(
        pool, "capture", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        pool.warm_native_batch(
            fake, [1, 2], [host, host], (), budget_exceeded=lambda: False
        )


def test_provider_hook_retains_order_and_continuation() -> None:
    from helion.autotuner.benchmark_provider import LocalBenchmarkProvider

    source = ast.parse(
        textwrap.dedent(inspect.getsource(LocalBenchmarkProvider.benchmark))
    )
    fn = source.body[0]
    hook = next(
        node
        for node in fn.body
        if isinstance(node, ast.If)
        and "autotune_cute_native_workers" in ast.unparse(node.test)
    )
    before = fn.body[: fn.body.index(hook)]
    assert any(
        isinstance(n, ast.Assign)
        and ast.unparse(n.targets[0]) == "compiled"
        and (ast.unparse(n.value) == "unique_compiled")
        for n in before
    )
    required = fn.body[fn.body.index(hook) + 2]
    assert isinstance(required, ast.If)
    assert ast.unparse(required.test) == "self.settings.autotune_cute_native_required"
    legacy = required.orelse[0]
    assert isinstance(legacy, ast.If)
    assert ast.unparse(legacy.test) == "self.settings.autotune_precompile"
    method = ast.FunctionDef(
        name="run_hook",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=n) for n in ("self", "configs", "fns")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[hook, ast.Return(ast.Name("fns", ast.Load()))],
        decorator_list=[],
    )
    scope = {"__package__": "helion.autotuner"}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            "<actual-native-provider-hook>",
            "exec",
        ),
        scope,
    )
    provider = SimpleNamespace(
        settings=SimpleNamespace(backend="cute", autotune_cute_native_workers=2),
        kernel=object(),
        args=(1,),
        _budget_exceeded_synced=lambda: False,
        _cute_native_batch_records=[],
    )
    seen = []
    with patch.object(
        pool,
        "warm_native_batch",
        side_effect=lambda *a, **k: seen.append(a) or {"status": "warmed"},
    ):
        fns = [host, compute_host]
        assert scope["run_hook"](provider, [1, 2], fns) is fns
    assert seen[0][1:] == ([1, 2], fns, (1,))
    assert provider._cute_native_batch_records == [{"status": "warmed"}]


def commands(texts: list[str]) -> object:
    return [
        ([sys.executable, "-B", "-c", UNBLOCK + text], dict(os.environ))
        for text in texts
    ]


@LINUX
def test_distinct_requests_reuse_two_owned_processes(tmp_path: Path) -> None:
    scripts = []
    for i in range(2):
        path = tmp_path / f"process-{i}"
        path.mkdir()
        scripts.append(
            f"import os,json,pathlib; p=pathlib.Path({str(path)!r}); rows=[]\nfor j in range(2):\n rows.append([os.getpid(),j,{i}*2+j])\n(p/'rows.json').write_text(json.dumps(rows))"
        )
    reaps = run_workers(commands(scripts), tmp_path, workers=2, timeout=3)
    assert len({x["pid"] for x in reaps}) == 2 and all(
        x["exitcode"] == 0 for x in reaps
    )
    rows = [
        json.loads((tmp_path / f"process-{i}/rows.json").read_text()) for i in range(2)
    ]
    assert all(r[0][0] == r[1][0] for r in rows)
    assert len({x[2] for r in rows for x in r}) == 4


@LINUX
@pytest.mark.parametrize("failure", ["exit", "timeout", "spawn", "budget"])
def test_failure_drains_only_owned_children(tmp_path: Path, failure: str) -> None:
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        cmds = commands(
            [
                "raise SystemExit(3)"
                if failure == "exit"
                else "import time;time.sleep(5)",
                "import time;time.sleep(5)",
            ]
        )
        if failure == "spawn":
            cmds[1] = (["/nonexistent-native-worker"], dict(os.environ))
        n = [0]

        def budget() -> object:
            n[0] += 1
            return failure == "budget" and n[0] > 1

        with pytest.raises((RuntimeError, TimeoutError, FileNotFoundError)):
            run_workers(
                cmds,
                tmp_path,
                workers=2,
                timeout=0.15 if failure == "timeout" else 3,
                budget_exceeded=budget,
            )
        record = json.loads((tmp_path / "REAPS.json").read_text())
        assert record["all_direct_children_reaped"] and record["records"]
        for row in record["records"]:
            with pytest.raises(ChildProcessError):
                os.waitpid(row["pid"], os.WNOHANG)
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait()


@LINUX
def test_interrupt_drains_then_propagates(tmp_path: Path) -> None:
    prior = signal.getsignal(signal.SIGTERM)
    timer = threading.Timer(0.15, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            run_workers(
                commands(["import time;time.sleep(5)"]), tmp_path, workers=1, timeout=3
            )
    finally:
        timer.cancel()
        timer.join()
    assert signal.getsignal(signal.SIGTERM) is prior
    assert json.loads((tmp_path / "REAPS.json").read_text())[
        "all_direct_children_reaped"
    ]


@LINUX
def test_per_request_timeout_does_not_reset_on_same_index(tmp_path: Path) -> None:
    (tmp_path / "process-0").mkdir()
    path = str(tmp_path / "process-0/ACTIVE.json")
    script = f"import time,json,pathlib;p=pathlib.Path({path!r});p.write_text(json.dumps(dict(index=0)));time.sleep(.1);p.write_text(json.dumps(dict(index=1)));time.sleep(5)"
    with pytest.raises(NativeRequestTimeout, match="startup/request") as failure:
        run_workers(
            commands([script]), tmp_path, workers=1, timeout=0.2, request_counts=[2]
        )
    assert json.loads((tmp_path / "REAPS.json").read_text())[
        "all_direct_children_reaped"
    ]
    assert failure.value.worker == 0 and failure.value.request == 1
    assert failure.value.pid > 0


def test_observer_narrow_loader_and_version_only(tmp_path: Path) -> None:

    class Library:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __getitem__(self, name: str) -> object:

            def query(pointer: object) -> object:
                pointer._obj.value = 13010
                return 0

            return query

    loader = tmp_path / "loader.py"
    scope = {"ctypes": ctypes}
    exec(
        compile(
            "def _select_and_load_cutlass_ir_toolkit():\n v=ctypes.c_int(0)\n result=ctypes.CDLL('libcuda.so.1').cuDriverGetVersion(ctypes.byref(v))\n return result,v.value\n",
            str(loader),
            "exec",
        ),
        scope,
    )
    calls = []
    with (
        patch.object(ctypes, "CDLL", Library),
        DriverMetadata(loader, allow_query=True, calls=calls) as observer,
    ):
        assert scope["_select_and_load_cutlass_ir_toolkit"]() == (0, 13010)
        with pytest.raises(AssertionError):
            scope["_select_and_load_cutlass_ir_toolkit"]()
        for name in ("cuInit", "cuLaunchKernel", "cuCtxCreate"):
            with pytest.raises(RuntimeError):
                getattr(ctypes.CDLL("libcuda.so.1"), name)
        with pytest.raises(RuntimeError):
            assert callable(ctypes.CDLL("libcudart.so.13").cudaMalloc)
    assert len(observer.queries) == 1 and len(calls) == 4


def test_worker_import_order_and_hard_exit() -> None:
    tree = ast.parse(Path(pool.__file__).with_name("native_pool_worker.py").read_text())
    main = next(
        x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "main"
    )
    ordered = ast.unparse(main)
    assert (
        ordered.index("observer.__enter__()")
        < ordered.index("importlib.import_module('cutlass')")
        < ordered.index("metadata = observer.finish()")
        < ordered.index("guard.__enter__()")
        < ordered.index("_compile_requests(")
    )
    assert "os._exit(" in ast.unparse(main.body[-1].finalbody[-1])
    assert "status='failed'" in ast.unparse(main.body[-1].handlers[0])


def worker_functions() -> dict[str, object]:
    """Run actual worker control flow with inert compilation, without SDK imports."""
    tree = ast.parse(Path(pool.__file__).with_name("native_pool_worker.py").read_text())
    names = {"NativeCompileError", "_compile_native", "_compile_requests"}
    body = [
        ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0)
    ]
    body.extend(
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    )
    scope = {"time": time, "traceback": traceback, "write": write}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            "<actual-native-worker-control>",
            "exec",
        ),
        scope,
    )
    return scope


@pytest.mark.parametrize("failure", [RuntimeError, MemoryError, KeyboardInterrupt])
def test_only_compile_exceptions_are_recoverable(failure: type[BaseException]) -> None:
    scope = worker_functions()
    original = failure("original compile error")

    def compile_fn(*args: object, **kwargs: object) -> object:
        raise original

    expected = scope["NativeCompileError"] if failure is RuntimeError else failure
    with pytest.raises(expected) as error:
        scope["_compile_native"](compile_fn, object(), [], "options")
    assert (
        error.value.__cause__ if failure is RuntimeError else error.value
    ) is original


@pytest.mark.parametrize("failure", ["compile", "guard", "persistence"])
def test_worker_retains_checked_prefix_and_stops_after_failure(
    tmp_path: Path, failure: str
) -> None:
    scope = worker_functions()
    specs = [{"cache_key": str(i), "source_sha256": f"source{i}"} for i in range(3)]
    result = {"status": "running", "requests": [], "retained_compilations": 0}
    retained = []
    seen = []
    violated = []

    def compile_request(spec: object, owners: list[object]) -> object:
        seen.append(spec)
        if len(seen) == 1:
            owners.append(object())
            return dict(spec, compile_seconds=0.1)
        # The first success is durable before compilation of the second starts.
        saved = json.loads((tmp_path / "RESULT.json").read_text())
        assert saved["requests"] == [dict(specs[0], compile_seconds=0.1)]
        assert saved["retained_compilations"] == 1
        if failure == "persistence":
            raise RuntimeError("cache persistence is not a compile exception")
        if failure == "guard":
            violated.append(True)

        def compile_fn(*args: object) -> object:
            raise ValueError("candidate IR rejected")

        return scope["_compile_native"](compile_fn, object(), [], None)

    def check() -> None:
        assert not violated, "forbidden operation"

    scope["compile_request"] = compile_request
    if failure == "compile":
        scope["_compile_requests"](specs, tmp_path, retained, result, check)
        saved = json.loads((tmp_path / "RESULT.json").read_text())
        assert saved["status"] == "compile_error"
        assert saved["compile_failure"]["index"] == 1
        assert saved["compile_failure"]["cache_key"] == "1"
        assert saved["compile_failure"]["error_type"] == "ValueError"
    else:
        with pytest.raises(AssertionError if failure == "guard" else RuntimeError):
            scope["_compile_requests"](specs, tmp_path, retained, result, check)
        assert result["status"] == "running"  # main marks the worker failed.
    assert seen == specs[:2] and len(retained) == 1


@pytest.mark.parametrize(
    "mode",
    [
        "normal",
        "corrupt",
        "compile_error",
        "wrong_failure",
        "guard",
        "reload",
        "timeout",
        "timeout_compile_error",
        "timeout_corrupt",
        "timeout_metadata",
        "timeout_guard",
        "timeout_exit",
        "timeout_bad_reaps",
        "timeout_bad_active",
        "startup_timeout",
        "budget_timeout",
        "interrupt",
    ],
)
def test_batch_publication_drains_and_preserves_verified_successes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    from helion.runtime.kernel import BoundKernel

    bound = object.__new__(BoundKernel)
    bound.kernel = SimpleNamespace(
        settings=SimpleNamespace(
            backend="cute",
            static_shapes=True,
            autotune_cute_native_workers=2,
            autotune_cute_native_required=False,
            autotune_compile_timeout=3,
        )
    )
    bound._env = SimpleNamespace(
        process_group_name=None,
        cute_fission_plan=None,
        cute_resolved_wrapper_plans=(),
        device=torch.device("cuda:0"),
    )
    requests = []
    reloaded = []
    drained = []
    timeout_mode = "timeout" in mode or mode == "interrupt"
    for index in range(4):
        directory = tmp_path / str(index)
        directory.mkdir()
        compiled = SimpleNamespace(
            _compiled=None,
            _cache_file_paths=lambda d=directory, i=index: (
                str(d),
                str(d / f"cute_dsl_{i}.mlir"),
                str(d / f"cute_dsl_{i}.json"),
            ),
            _reload_from_disk=lambda i=index: (
                reloaded.append(i)
                or (None if mode == "reload" and i == 0 else object())
            ),
        )
        requests.append(
            SimpleNamespace(
                spec={"cache_key": str(index), "source_sha256": f"source{index}"},
                compiled=compiled,
                check=lambda: None,
            )
        )
    monkeypatch.setattr(pool, "capture", lambda b, c, f, a: requests[c])
    monkeypatch.setattr(pool, "sdk_identity", lambda: {"metadata": {"version": 1}})
    monkeypatch.setattr(pool.sys, "platform", "linux")
    monkeypatch.setattr(pool.signal, "getsignal", lambda s: signal.SIG_DFL)
    original_verify = pool.verify_result

    def verify_after_drain(*args: object) -> object:
        assert drained and not reloaded
        return original_verify(*args)

    monkeypatch.setattr(pool, "verify_result", verify_after_drain)

    def inert_workers(
        commands: object, output: Path, **kwargs: object
    ) -> list[dict[str, object]]:
        records = []
        for i in range(2):
            directory = output / f"process-{i}"
            specs = json.loads((directory / "BATCH.json").read_text())["requests"]
            assert kwargs["request_counts"] == [2, 2]
            assert commands[i][1]["CUDA_VISIBLE_DEVICES"] == ""
            partial = i == 0 and mode in (
                "compile_error",
                "wrong_failure",
                "timeout_compile_error",
            )
            killed = timeout_mode and i == 1
            rows = []
            for spec in specs[:1] if partial or killed else specs:
                key = spec["cache_key"]
                ir = directory / "cache" / f"cute_dsl_{key}.mlir"
                meta = directory / "cache" / f"cute_dsl_{key}.json"
                ir.write_bytes(b"original cache payload")
                symbol = {"function_name": f"symbol{key}", "has_gpu_module": True}
                write(meta, symbol)
                rows.append(
                    {
                        "cache_key": key,
                        "source_sha256": spec["source_sha256"],
                        "ir": str(ir),
                        "metadata": str(meta),
                        "ir_sha256": "wrong"
                        if mode == "corrupt"
                        and i == 1
                        or mode == "timeout_corrupt"
                        and i == 0
                        else sha(ir),
                        "metadata_sha256": sha(meta),
                        "symbol": symbol,
                        "compile_seconds": 0.01,
                    }
                )
            write(
                directory / "RESULT.json",
                {
                    "status": "running"
                    if killed
                    else "compile_error"
                    if partial
                    else "passed",
                    "compile_failure": {
                        "index": 1,
                        "cache_key": "foreign"
                        if mode == "wrong_failure"
                        else specs[-1]["cache_key"],
                        "source_sha256": specs[-1]["source_sha256"],
                        "error_type": "RuntimeError",
                        "error": "candidate compilation failed",
                    },
                    "forbidden_calls": ["cuInit"]
                    if mode == "guard" or mode == "timeout_guard" and killed
                    else [],
                    "driver_metadata": {
                        "version": 2 if mode == "timeout_metadata" and i == 0 else 1
                    },
                    "retained_compilations": len(rows),
                    "pid": 10 + i,
                    "requests": rows,
                    "startup_seconds": 0.1,
                    "seconds": 0.2,
                    "guards": 956,
                },
            )
            records.append(
                {
                    "index": i,
                    "pid": 10 + i,
                    "exitcode": (-11 if mode == "timeout_exit" else -signal.SIGTERM)
                    if killed
                    else 0,
                    **({"cleanup": True} if killed else {}),
                }
            )
            write(
                directory / "ACTIVE.json",
                {"index": 0 if mode == "timeout_bad_active" else 1},
            )
        assert not reloaded
        drained.append(True)
        if timeout_mode:
            write(
                output / "REAPS.json",
                {
                    "records": records,
                    "all_direct_children_reaped": mode != "timeout_bad_reaps",
                },
            )
            if mode in ("startup_timeout", "budget_timeout"):
                raise TimeoutError(mode)
            if mode == "interrupt":
                raise KeyboardInterrupt
            raise NativeRequestTimeout(1, 11, 1, 3)
        return records

    monkeypatch.setattr(pool, "run_workers", inert_workers)
    if mode == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            pool.warm_native_batch(
                bound, [0, 1, 2, 3], [host] * 4, (), budget_exceeded=lambda: False
            )
        assert drained and not reloaded
        return
    result = pool.warm_native_batch(
        bound, [0, 1, 2, 3], [host] * 4, (), budget_exceeded=lambda: False
    )
    rejected = (
        mode in ("corrupt", "wrong_failure", "guard")
        or timeout_mode
        and mode not in ("timeout", "timeout_compile_error")
    )
    expected = (
        []
        if rejected
        else [0]
        if mode == "timeout_compile_error"
        else [0, 2]
        if mode == "timeout"
        else [0, 1, 3]
        if mode == "compile_error"
        else [0, 2, 1, 3]
    )
    assert reloaded == expected
    assert result["status"] == (
        "serial_fallback"
        if rejected
        else "partially_warmed"
        if mode in ("compile_error", "reload", "timeout", "timeout_compile_error")
        else "warmed"
    )
    for i in range(4):
        assert (tmp_path / str(i) / f"cute_dsl_{i}.mlir").exists() == (i in expected)
    if mode == "compile_error":
        assert result["reloads"] == 3 and len(result["compile_failures"]) == 1
        assert result["processes"][0]["completed_requests"][0]["cache_key"] == "0"
    if mode == "reload":
        assert result["reloads"] == 3 and requests[0].compiled._compiled is None
    if mode in ("timeout", "timeout_compile_error"):
        assert result["excluded_timeout_workers"] == [1]
        assert result["request_timeout"]["request"] == 1
        assert (
            requests[1].compiled._compiled is None
            and requests[3].compiled._compiled is None
        )


DTYPE_CAPTURE_CONFIG = Config()


def dtype_host(
    x: torch.Tensor,
    columns: int,
    dtype: torch.dtype,
    *,
    _launcher: Callable[..., object] | None = None,
) -> torch.Tensor:
    assert _launcher is not None
    output = torch.empty((x.size(0), columns), dtype=dtype, device=x.device)
    _launcher(dtype_host, (1,), x, output, block=(128, 1, 1))
    return output


def dtype_capture_bound() -> BoundKernel:
    return cast(
        "BoundKernel",
        SimpleNamespace(
            _normalized_config_copy=lambda config: config,
            _compile_cache={DTYPE_CAPTURE_CONFIG: dtype_host},
            get_cached_path=lambda config: str(Path(__file__).resolve()),
        ),
    )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.half, torch.bfloat16, torch.float32]
)
def test_exact_dtype_reaches_original_metadata_host_and_native_schema(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    x = torch.zeros(2, 4)
    original = (x, 4, dtype)
    seen: list[tuple[object, ...]] = []

    def schema(
        kernel: object, values: tuple[object, ...], grid: object
    ) -> SimpleNamespace:
        assert kernel is dtype_host and grid == (1, 1, 1)
        assert values[0] is original[0]
        assert isinstance(values[1], torch.Tensor)
        assert values[1].dtype is original[2]
        seen.append(values)
        return SimpleNamespace(
            schema=tuple(
                ("tensor", str(v.dtype), v.ndim, tuple(v.shape), tuple(v.stride()), 16)
                for v in (cast("torch.Tensor", value) for value in values)
            ),
            launch_args=(object(), object(), 1, 1, 1),
            stream_schedule=None,
            owned_tensors=(),
            grouped_static_metadata=(),
        )

    def compiled(
        kernel: object, schema: object, block: object, options: object, values: object
    ) -> SimpleNamespace:
        assert kernel is dtype_host and block == (128, 1, 1) and options is None
        assert values is seen[0]
        # Exercise the real already-compiled terminal path without native work.
        return SimpleNamespace(_compiled=object())

    monkeypatch.setattr(launcher, "_build_cute_schema_and_args", schema)
    monkeypatch.setattr(launcher, "_get_compiled_cute_launcher", compiled)
    assert pool.host_supported(dtype_host)
    assert (
        pool.capture(dtype_capture_bound(), DTYPE_CAPTURE_CONFIG, dtype_host, original)
        is None
    )
    assert len(seen) == 1
    assert original[0] is x and original[2] is dtype
    assert torch.count_nonzero(x).item() == 0


class IntSubclass(int):
    pass


@pytest.mark.parametrize(
    "unsupported",
    [
        object(),
        "torch.float16",
        torch.device("cpu"),
        [torch.float16],
        {"dtype": torch.float16},
        IntSubclass(4),
    ],
)
def test_existing_unsupported_host_argument_types_still_decline(
    monkeypatch: pytest.MonkeyPatch, unsupported: object
) -> None:
    def forbidden_schema(*args: object) -> None:
        pytest.fail("unsupported host argument reached the native schema")

    monkeypatch.setattr(launcher, "_build_cute_schema_and_args", forbidden_schema)
    assert (
        pool.capture(
            dtype_capture_bound(),
            DTYPE_CAPTURE_CONFIG,
            dtype_host,
            (torch.zeros(2, 4), 4, unsupported),
        )
        is None
    )


def test_tensor_subclass_still_declines(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_schema(*args: object) -> None:
        pytest.fail("tensor subclass reached the native schema")

    monkeypatch.setattr(launcher, "_build_cute_schema_and_args", forbidden_schema)
    value = torch.nn.Parameter(torch.zeros(2, 4))
    assert (
        pool.capture(
            dtype_capture_bound(),
            DTYPE_CAPTURE_CONFIG,
            dtype_host,
            (value, 4, torch.float16),
        )
        is None
    )


def test_host_dtype_does_not_expand_native_scalar_codec() -> None:
    launch = SimpleNamespace(
        schema=(("scalar", "dtype"),),
        launch_args=(torch.float16, 1, 1, 1),
    )
    with pytest.raises(TypeError, match="Unsupported compile metadata"):
        pool.arguments_for_worker(
            (torch.float16,), cast("_CuteLaunchArgCacheEntry", launch)
        )
