"""Opt-in native cache warming for a finite provider batch.

The parent owns real host arguments, schema and cache identity. Fresh spawned
workers only compile pointer/scalar wrappers; they never execute kernels. All
workers exit before ordinary accuracy validation or timing starts. Unsupported
hosts and speculative failures leave the existing serial launch path intact.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
import inspect
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import textwrap
import threading
import time
import types
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ..._compile_time import measure
from .native_pool_protocol import NativeRequestTimeout
from .native_pool_protocol import freeze
from .native_pool_protocol import run_workers
from .native_pool_protocol import sha
from .native_pool_protocol import wrapper_ast
from .native_pool_protocol import write

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from ..config import Config
    from ..kernel import BoundKernel
    from .launcher import _CuteLaunchArgCacheEntry

log = logging.getLogger(__name__)


def owner_facts(tensor: torch.Tensor) -> tuple[object, ...]:
    storage = tensor.untyped_storage()
    return (
        id(tensor),
        storage._cdata,
        storage.data_ptr(),
        storage.nbytes(),
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.dtype,
        tensor.device,
        tensor._version,
    )


def host_supported(fn: object) -> bool:
    """Only straight-line generated allocation/metadata code and one launcher."""
    if not isinstance(fn, types.FunctionType):
        return False
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False
    host = tree.body[0]
    if any(
        isinstance(
            n,
            (
                ast.If,
                ast.For,
                ast.While,
                ast.With,
                ast.Try,
                ast.Lambda,
                ast.ListComp,
                ast.GeneratorExp,
                ast.Await,
                ast.Yield,
            ),
        )
        for n in ast.walk(host)
    ):
        return False
    if fn.__globals__.get("torch") is not torch:
        return False
    for name in ("int", "len", "tuple", "min", "max"):
        if fn.__globals__.get(name, getattr(builtins, name)) is not getattr(
            builtins, name
        ):
            return False
    launches = 0
    for node in ast.walk(ast.Module(body=host.body, type_ignores=[])):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name == "_launcher":
            launches += 1
        elif (
            name
            in (
                "torch.empty",
                "torch.empty_like",
                "torch.empty_strided",
                "int",
                "len",
                "tuple",
                "min",
                "max",
            )
            or isinstance(node.func, ast.Attribute)
            and node.func.attr
            in ("size", "stride", "view", "reshape", "as_strided", "numel", "dim")
        ):
            continue
        elif name == "_next_power_of_2":
            from ..._utils import next_power_of_2

            if fn.__globals__.get(name) is not next_power_of_2:
                return False
        else:
            return False
    return launches == 1


class MetadataHostOnly(TorchDispatchMode):
    def __torch_dispatch__(
        self,
        func: Callable[..., Any],
        types: object,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        allowed = {
            torch.ops.aten.empty.memory_format,
            torch.ops.aten.empty_strided.default,
            torch.ops.aten.empty_like.default,
            torch.ops.aten.view.default,
            torch.ops.aten._unsafe_view.default,
            torch.ops.aten.as_strided.default,
            torch.ops.aten.alias.default,
        }
        if func not in allowed:
            raise ValueError(f"Native precompile host contains computation: {func}")
        return func(*args, **kwargs or {})


def kernel_attributes(kernel: object) -> object:
    # Launcher-owned execution caches can appear after the first invocation.
    # They do not define compilation; every other CuTe attribute still does.
    runtime_caches = {
        "_helion_cute_compiled_launchers",
        "_helion_cute_launch_arg_cache",
        "_helion_cute_fastpath",
        "_helion_cute_last_launch_cache",
    }
    return freeze(
        {
            k: v
            for k, v in vars(kernel).items()
            if k.startswith("_helion_cute_") and k not in runtime_caches
        }
    )


@dataclass
class NativeRequest:
    spec: dict[str, Any]
    compiled: Any
    owners: tuple[torch.Tensor, ...]
    storages: tuple[object, ...]
    facts: tuple[tuple[object, ...], ...]
    retained: tuple[Any, ...]

    def check(self) -> None:
        from . import launcher

        assert tuple(owner_facts(t) for t in self.owners) == self.facts
        assert sha(self.spec["source"]) == self.spec["source_sha256"]
        assert freeze(launcher._cute_cache_relevant_env()) == self.spec["environment"]
        assert self.compiled._cache_key == self.spec["cache_key"]
        assert self.compiled._compile_options == self.spec["options"]
        kernel, launch, original_args, fn, values, compile_options = self.retained[:6]
        assert kernel_attributes(kernel) == self.spec["attributes"]
        assert freeze(launch.schema) == self.spec["schema"]
        assert wrapper_ast(self.compiled._jit_func, kernel) == self.spec["wrapper_ast"]
        _, options, num_sm = launcher._cute_compiled_launcher_discriminator(
            launch.schema,
            tuple(self.spec["block"]),
            compile_options,
            values,
            pointer_alignment=launcher._cute_pointer_alignment(kernel),
        )
        assert (options, num_sm) == (self.spec["options"], self.spec["num_sm"])
        assert (
            launcher._cute_disk_cache_key(
                kernel,
                launch.schema,
                tuple(self.spec["block"]),
                (),
                None,
                options,
                num_sm,
            )
            == self.spec["cache_key"]
        )


def arguments_for_worker(
    args: Sequence[object], launch: _CuteLaunchArgCacheEntry
) -> list[dict[str, Any]]:
    """Describe the original schema's pointer positions without patching make_ptr."""
    from . import launcher

    assert len(launch.schema) == len(args)
    result: list[dict[str, Any]] = []
    cursor = 0
    for arg, entry in zip(args, launch.schema, strict=True):
        kind = entry[0]
        if kind == "tensor":
            assert type(arg) is torch.Tensor
            result.append(
                {
                    "kind": "pointer",
                    "dtype": cast(
                        "Any", launcher._torch_dtype_to_cutlass(arg.dtype)
                    ).__name__,
                    "alignment": launcher._cute_schema_pointer_alignment(entry),
                }
            )
            cursor += 1
            if len(entry) in (3, 4):
                for _ in range(2 * arg.ndim):
                    result.append(
                        {"kind": "scalar", "value": freeze(launch.launch_args[cursor])}
                    )
                    cursor += 1
        elif kind == "scalar":
            result.append(
                {"kind": "scalar", "value": freeze(launch.launch_args[cursor])}
            )
            cursor += 1
        else:
            assert kind == "scalar_constexpr"
    assert len(launch.launch_args) - cursor == 3
    result.extend(
        {"kind": "scalar", "value": freeze(v)} for v in launch.launch_args[cursor:]
    )
    return result


def capture(
    bound: BoundKernel,
    config: Config,
    fn: Callable[..., object],
    args: Sequence[object],
    *,
    include_cached: bool = False,
) -> NativeRequest | None:
    from . import launcher
    from .chained_startup import validate_arguments
    from .horizontal_state import validate_horizontal_state_kernel

    normalized = bound._normalized_config_copy(config)
    if bound._compile_cache.get(normalized) is not fn or not host_supported(fn):
        return None
    path = Path(cast("types.FunctionType", fn).__code__.co_filename).resolve()
    if bound.get_cached_path(normalized) != str(path):
        return None
    # Dtypes are immutable host metadata; native values still use the schema codec.
    if any(
        type(v) not in (torch.Tensor, torch.dtype, int, float, bool, type(None))
        for v in args
    ):
        return None
    private = args
    got = []

    def intercept(
        kernel: object, grid: tuple[int, ...], *values: object, **kwargs: object
    ) -> object:
        assert not got and set(kwargs) <= {"block", "cute_compile_options"}
        assert not getattr(kernel, "_helion_cute_wrapper_plans", ())
        assert not getattr(kernel, "_helion_cute_cluster_shape", ())
        validate_arguments(kernel, values)
        launcher._validate_chained_leaf_arguments(kernel, values)
        validate_horizontal_state_kernel(kernel, values)
        grid = tuple(grid) + (1,) * (3 - len(grid))
        block = cast(
            "tuple[int, int, int]",
            tuple(cast("tuple[int, ...]", kwargs.get("block", (256, 1, 1)))),
        )
        assert len(grid) == len(block) == 3 and all(
            type(x) is int and x > 0 for x in (*grid, *block)
        )
        launch = launcher._build_cute_schema_and_args(kernel, values, grid)
        assert (
            launch.stream_schedule is None
            and (not launch.owned_tensors)
            and (not launch.grouped_static_metadata)
        )
        params = arguments_for_worker(values, launch)
        attrs = kernel_attributes(kernel)
        compiled = cast(
            "Any",
            launcher._get_compiled_cute_launcher(
                kernel,
                launch.schema,
                block,
                cast("str | None", kwargs.get("cute_compile_options")),
                values,
            ),
        )
        if compiled._compiled is not None and not include_cached:
            got.append(None)
            return
        assert compiled._cache_key is not None
        if not include_cached and all(
            Path(x).is_file() for x in compiled._cache_file_paths()[1:]
        ):
            got.append(None)
            return
        _, options, num_sm = launcher._cute_compiled_launcher_discriminator(
            launch.schema,
            block,
            cast("str | None", kwargs.get("cute_compile_options")),
            values,
            pointer_alignment=launcher._cute_pointer_alignment(kernel),
        )
        spec = {
            "source": str(path),
            "source_sha256": sha(path),
            "host": cast("types.FunctionType", fn).__name__,
            "kernel": cast("Any", kernel).__name__,
            "config": freeze(normalized.config),
            "attributes": attrs,
            "schema": freeze(launch.schema),
            "block": block,
            "arguments": params,
            "options": options,
            "num_sm": num_sm,
            "cache_key": compiled._cache_key,
            "environment": freeze(launcher._cute_cache_relevant_env()),
            "wrapper_ast": wrapper_ast(compiled._jit_func, kernel),
        }
        owners = tuple(v for v in values if type(v) is torch.Tensor)
        assert owners and all(t.device.type == "cuda" for t in owners)
        got.append(
            NativeRequest(
                spec,
                compiled,
                owners,
                tuple(t.untyped_storage() for t in owners),
                tuple(owner_facts(t) for t in owners),
                (
                    kernel,
                    launch,
                    private,
                    fn,
                    values,
                    cast("str | None", kwargs.get("cute_compile_options")),
                ),
            )
        )

    with MetadataHostOnly():
        output = fn(*private, _launcher=intercept)
    assert len(got) == 1
    request = got[0]
    if request is not None:
        request.retained += (output,)
    return request


def sdk_identity() -> dict[str, Any]:
    import importlib

    from cuda.bindings import driver

    importlib.import_module("cutlass")
    loader = Path(
        cast("str", sys.modules["cutlass._mlir._mlir_libs"].__file__)
    ).resolve()
    extension = Path(
        cast("str", sys.modules["cutlass._mlir._mlir_libs._cutlass_ir"].__file__)
    ).resolve()
    assert extension.name.startswith("_cutlass_ir.cu")
    toolkit = int(extension.name.split(".cu", 1)[1].split(".", 1)[0])
    status, version = driver.cuDriverGetVersion()
    assert int(status) == 0 and version // 1000 >= toolkit
    runtime = loader.parents[4] / f"cu{toolkit}/lib/libcute_dsl_runtime.so"
    libs = os.environ.get("CUTE_DSL_LIBS", "")
    assert runtime.is_file() and str(runtime) in libs.split(":")
    metadata = {
        "queries": [{"result": 0, "driver_version": version}],
        "query_count": 1,
        "toolkit": toolkit,
        "extension": str(extension),
        "runtime_library": str(runtime),
        "cute_dsl_libs": libs,
    }
    return {"loader": str(loader), "loader_sha256": sha(loader), "metadata": metadata}


def verify_result(
    request: NativeRequest, row: dict[str, Any], directory: Path
) -> list[Path]:
    assert row["cache_key"] == request.spec["cache_key"]
    assert row["source_sha256"] == request.spec["source_sha256"]
    checked = []
    for field, suffix in (("ir", ".mlir"), ("metadata", ".json")):
        path = Path(row[field]).resolve()
        assert path.parent == directory.resolve()
        assert path.name == "cute_dsl_" + request.spec["cache_key"] + suffix
        assert sha(path) == row[field + "_sha256"]
        checked.append(path)
    sidecar = json.loads(checked[1].read_text())
    assert set(sidecar) == {"function_name", "has_gpu_module"}
    assert sidecar == row["symbol"] and sidecar["has_gpu_module"] is True
    assert isinstance(sidecar["function_name"], str) and sidecar["function_name"]
    request.check()
    return checked


def _record_worker_failure(
    stats: dict[str, Any],
    output: Path,
    requests: Sequence[NativeRequest],
    workers: int,
) -> None:
    """Retain small diagnostic records after drain, before temporary cleanup.

    Worker results here are reported metadata, not verified cache artifacts.
    This path never publishes or reloads any completed compilation.
    """
    diagnostics: dict[str, Any] = {"processes": [], "read_errors": []}
    stats["failure_diagnostics"] = diagnostics

    def read(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            return value
        except (OSError, ValueError) as error:
            diagnostics["read_errors"].append(f"{path.name}: {type(error).__name__}")
            return {}

    reaps = read(output / "REAPS.json")
    stats["reaps"] = reaps.get("records", [])
    diagnostics["supervisor"] = reaps
    for index in range(workers):
        directory = output / f"process-{index}"
        assigned = requests[index::workers]
        active = read(directory / "ACTIVE.json")
        result = read(directory / "RESULT.json")
        rows = result.get("requests", [])
        process: dict[str, Any] = {
            "index": index,
            "reported_status": result.get("status"),
            "reported_pid": result.get("pid"),
            "reported_completed_count": len(rows),
            "reported_completed_requests": [
                {
                    k: row.get(k)
                    for k in ("cache_key", "source_sha256", "compile_seconds")
                }
                for row in rows
            ],
            "parent_verified": False,
            "publication_authorized": False,
        }
        request_index = active.get("index")
        if type(request_index) is int and 0 <= request_index < len(assigned):
            spec = assigned[request_index].spec
            process["active"] = {
                "worker_request_index": request_index,
                "batch_request_index": index + request_index * workers,
                "cache_key": spec["cache_key"],
                "source_sha256": spec["source_sha256"],
                "source": spec["source"],
                "reported_started": active.get("started"),
            }
        elif active:
            process["invalid_active_index"] = True
        diagnostics["processes"].append(process)


def _request_timeout_reaps(
    output: Path,
    workers: int,
    requests: Sequence[NativeRequest],
    error: NativeRequestTimeout,
) -> list[dict[str, Any]]:
    """Accept only the original supervisor's complete, expected deadline drain."""
    receipt = json.loads((output / "REAPS.json").read_text())
    assert receipt["all_direct_children_reaped"] is True
    records = receipt["records"]
    assert type(records) is list and len(records) == workers
    assert all(type(row["index"]) is int for row in records)
    assert {row["index"] for row in records} == set(range(workers))
    assert all(type(row["pid"]) is int and row["pid"] > 0 for row in records)
    assert len({row["pid"] for row in records}) == workers
    assert all(
        type(row["exitcode"]) is int
        and (
            row["exitcode"] == 0
            or row["exitcode"] in (-signal.SIGTERM, -signal.SIGKILL)
            and row.get("cleanup") is True
        )
        for row in records
    )
    assert 0 <= error.worker < workers
    assert (
        next(row["pid"] for row in records if row["index"] == error.worker) == error.pid
    )
    assert 0 <= error.request < len(requests[error.worker :: workers])
    active = json.loads(
        (output / f"process-{error.worker}" / "ACTIVE.json").read_text()
    )
    assert type(active["index"]) is int and active["index"] == error.request
    return records


def _compile_native_requests(
    bound: BoundKernel,
    requests: Sequence[NativeRequest],
    stats: dict[str, Any],
    *,
    budget_exceeded: Callable[[], bool],
) -> None:
    """Run one existing native worker round with unchanged verification/publication."""
    identity = sdk_identity()
    workers = min(bound.settings.autotune_cute_native_workers, len(requests))
    stats.update(
        status="started",
        requests=len(requests),
        workers=workers,
        parent_sdk_verification=identity,
        reloads=0,
    )
    runtime = str(Path(__file__).resolve().parents[3])
    script = Path(__file__).with_name("native_pool_worker.py")
    with tempfile.TemporaryDirectory(prefix="helion-native-batch-") as temporary:
        output = Path(temporary)
        commands = []
        for index in range(workers):
            directory = output / f"process-{index}"
            directory.mkdir()
            cache = directory / "cache"
            cache.mkdir()
            specs = [r.spec for r in requests[index::workers]]
            batch = {
                "requests": specs,
                "sdk": identity,
                "runtime": runtime,
                "sys_path": [
                    runtime,
                    *[str(Path(v or os.getcwd()).resolve()) for v in sys.path],
                ],
            }
            write(directory / "BATCH.json", batch)
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="",
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONUNBUFFERED="1",
                CUTE_DSL_CACHE_DIR=str(cache),
            )
            commands.append(
                (
                    [
                        sys.executable,
                        "-B",
                        str(script),
                        str(directory / "BATCH.json"),
                    ],
                    env,
                )
            )
        try:
            stats["reaps"] = run_workers(
                commands,
                output,
                workers=workers,
                timeout=bound.settings.autotune_compile_timeout,
                request_counts=[len(requests[i::workers]) for i in range(workers)],
                budget_exceeded=budget_exceeded,
            )
        except Exception as error:
            # run_workers has already terminated and reaped its children.
            # Keep evidence before TemporaryDirectory removes their files.
            try:
                _record_worker_failure(stats, output, requests, workers)
            except Exception as diagnostic_error:
                stats["diagnostic_error"] = type(diagnostic_error).__name__
            if not isinstance(error, NativeRequestTimeout):
                raise
            stats["reaps"] = _request_timeout_reaps(output, workers, requests, error)
            stats["request_timeout"] = {
                "worker": error.worker,
                "pid": error.pid,
                "request": error.request,
                "failure": str(error),
            }
            stats["excluded_timeout_workers"] = []
        verified = []
        stats["processes"] = []
        stats["compile_failures"] = []
        for index in range(workers):
            directory = output / f"process-{index}"
            reap = next(x for x in stats["reaps"] if x["index"] == index)
            killed = "request_timeout" in stats and reap["exitcode"] != 0
            if killed and not (directory / "RESULT.json").exists():
                # A peer can still be importing when the deadline drains it.
                stats["excluded_timeout_workers"].append(index)
                continue
            result = json.loads((directory / "RESULT.json").read_text())
            assert result["status"] in (
                ("running", "passed", "compile_error")
                if killed
                else ("passed", "compile_error")
            ) and (not result["forbidden_calls"])
            assert result["driver_metadata"] == identity["metadata"]
            assert result["pid"] == reap["pid"]
            assert type(result["requests"]) is list
            assert type(result["retained_compilations"]) is int
            assert len(result["requests"]) == result["retained_compilations"]
            if killed:
                # Check observable failures, but never authorize a killed
                # worker's checkpoint without its final guard/exit check.
                assigned = requests[index::workers]
                assert len(result["requests"]) <= len(assigned)
                for row, request in zip(
                    result["requests"],
                    assigned[: len(result["requests"])],
                    strict=True,
                ):
                    assert row["cache_key"] == request.spec["cache_key"]
                    assert row["source_sha256"] == request.spec["source_sha256"]
                stats["excluded_timeout_workers"].append(index)
                continue
            stats["processes"].append(
                {
                    k: result[k]
                    for k in (
                        "pid",
                        "startup_seconds",
                        "seconds",
                        "guards",
                        "status",
                    )
                }
            )
            stats["processes"][-1]["compile_seconds"] = [
                row["compile_seconds"] for row in result["requests"]
            ]
            assigned = requests[index::workers]
            completed = len(result["requests"])
            assert completed == result["retained_compilations"]
            if result["status"] == "passed":
                assert completed == len(assigned)
            else:
                failure = result["compile_failure"]
                assert completed < len(assigned)
                assert type(failure["index"]) is int
                assert failure["index"] == completed
                failed = assigned[completed]
                assert failure["cache_key"] == failed.spec["cache_key"]
                assert failure["source_sha256"] == failed.spec["source_sha256"]
                stats["compile_failures"].append(failure)
            for request, row in zip(
                assigned[:completed], result["requests"], strict=True
            ):
                verified.append(
                    (request, verify_result(request, row, directory / "cache"))
                )
            stats["processes"][-1]["completed_requests"] = [
                {k: row[k] for k in ("cache_key", "source_sha256", "compile_seconds")}
                for row in result["requests"]
            ]
        # Verify every worker artifact before publishing the first pair.
        for request, files in verified:
            request.check()
            directory, ir, meta = request.compiled._cache_file_paths()
            Path(directory).mkdir(parents=True, exist_ok=True)
            for source, destination in zip(files, (ir, meta), strict=True):
                tmp = Path(destination + f".pool.{os.getpid()}")
                shutil.copyfile(source, tmp)
                os.replace(tmp, destination)
            compiled = request.compiled._reload_from_disk()
            if compiled is not None:
                request.compiled._compiled = compiled
                stats["reloads"] += 1
        stats["verified_requests"] = len(verified)
        stats["status"] = (
            "warmed"
            if stats["reloads"] == len(requests)
            else "partially_warmed"
            if stats["reloads"]
            else "serial_fallback"
        )


def warm_native_batch(
    bound: object,
    configs: Sequence[Config],
    fns: Sequence[Callable[..., object]],
    args: Sequence[object],
    *,
    budget_exceeded: Callable[[], bool],
) -> dict[str, Any]:
    """Best-effort cache warming. No candidate failure is recorded here."""
    from ..kernel import BoundKernel

    if type(bound) is BoundKernel and bound.settings.autotune_cute_native_required:
        from .native_required import warm_required_native_batch

        return warm_required_native_batch(
            bound, configs, fns, args, budget_exceeded=budget_exceeded
        )

    stats: dict[str, Any] = {
        "status": "unsupported",
        "requests": 0,
        "workers": 0,
        "reaps": [],
    }
    if (
        type(bound) is not BoundKernel
        or bound.settings.backend != "cute"
        or (not bound.settings.static_shapes)
        or (bound.env.process_group_name is not None)
        or (bound.env.cute_fission_plan is not None)
        or bound.env.cute_resolved_wrapper_plans
        or (bound.env.device.type != "cuda")
        or (sys.platform != "linux")
        or sys.flags.optimize
        or (threading.current_thread() is not threading.main_thread())
        or (signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL)
        or (not 1 <= bound.settings.autotune_cute_native_workers <= 32)
        or (bound.settings.autotune_compile_timeout <= 0)
        or (len(fns) < 2)
    ):
        return stats
    started = time.monotonic()
    with measure("CuTe.native_pool_wall"):
        try:
            requests = []
            keys = set()
            for config, fn in zip(configs, fns, strict=True):
                if budget_exceeded():
                    return stats
                try:
                    request = capture(bound, config, fn, args)
                except Exception:
                    log.debug("CuTe native capture unavailable", exc_info=True)
                    continue
                if request is not None and request.spec["cache_key"] not in keys:
                    keys.add(request.spec["cache_key"])
                    requests.append(request)
            if len(requests) < 2:
                return stats
            _compile_native_requests(
                bound, requests, stats, budget_exceeded=budget_exceeded
            )
        except Exception as error:
            stats["failure"] = f"{type(error).__name__}: {error}"
            stats["status"] = "serial_fallback"
            log.debug(
                "CuTe native batch unavailable; retaining serial launch", exc_info=True
            )
    stats["wall_seconds"] = time.monotonic() - started
    return stats
