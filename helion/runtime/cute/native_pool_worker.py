"""One fresh interpreter, several native requests, no device execution."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from .native_pool_metadata import DriverMetadata
    from .native_pool_protocol import sha
    from .native_pool_protocol import thaw
    from .native_pool_protocol import wrapper_ast
    from .native_pool_protocol import write
else:
    from native_pool_metadata import DriverMetadata
    from native_pool_protocol import sha
    from native_pool_protocol import thaw
    from native_pool_protocol import wrapper_ast
    from native_pool_protocol import write


_Compiled = TypeVar("_Compiled")


class NativeCompileError(Exception):
    """A candidate failed inside cute.compile, before cache publication."""


def _compile_native(
    compile_fn: Callable[..., _Compiled],
    jit: object,
    args: list[object],
    options: object,
) -> _Compiled:
    try:
        if options is None:
            return compile_fn(jit, *args)
        return compile_fn(jit, *args, options=options)
    except MemoryError:
        raise
    except Exception as error:
        raise NativeCompileError from error


def _compile_requests(
    specs: list[dict[str, Any]],
    output: Path,
    retained: list[object],
    result: dict[str, Any],
    check: Callable[[], None],
) -> None:
    for index, spec in enumerate(specs):
        write(output / "ACTIVE.json", {"index": index, "started": time.monotonic()})
        check()
        try:
            row = compile_request(spec, retained)
        except NativeCompileError as error:
            # Do not reuse a possibly tainted SDK context after compilation fails.
            # Only the already checked prefix survives, and only if guards still
            # hold. Import, identity, persistence and process failures stay fatal.
            check()
            result.update(
                status="compile_error",
                compile_failure={
                    "index": index,
                    "cache_key": spec["cache_key"],
                    "source_sha256": spec["source_sha256"],
                    "error_type": type(error.__cause__).__name__,
                    "error": traceback.format_exc(),
                },
            )
            write(output / "RESULT.json", result)
            return
        check()
        result["requests"].append(row)
        result["retained_compilations"] = len(retained)
        # Persist each completed request before starting the next compilation.
        write(output / "RESULT.json", result)
    result["status"] = "passed"


def compile_request(spec: dict[str, Any], retained: list[object]) -> dict[str, Any]:
    from cuda.bindings.driver import CUstream
    import cutlass
    from cutlass import cute
    from cutlass.cute.runtime import make_ptr

    from helion._compiler.cute.backend import CuteBackend
    from helion.runtime.cute import launcher

    assert sha(spec["source"]) == spec["source_sha256"]
    name = "_helion_native_pool_" + spec["cache_key"]
    module_spec = importlib.util.spec_from_file_location(name, spec["source"])
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    text = Path(spec["source"]).read_text()
    CuteBackend().annotate_compiled_module(module, text, spec["host"])
    kernel = getattr(module, spec["kernel"])
    for key, value in cast("dict[str, object]", thaw(spec["attributes"])).items():
        setattr(kernel, key, value)
    schema, block = (
        cast("tuple[tuple[object, ...], ...]", thaw(spec["schema"])),
        tuple(spec["block"]),
    )
    assert launcher._cute_cache_relevant_env() == thaw(spec["environment"])
    jit = launcher._create_cute_wrapper(kernel, schema, block, spec["num_sm"])
    assert wrapper_ast(cast("Any", jit), kernel) == spec["wrapper_ast"]
    key = launcher._cute_disk_cache_key(
        kernel, schema, block, (), None, spec["options"], spec["num_sm"]
    )
    assert key == spec["cache_key"]
    compiled_launcher = launcher._CompiledCuteLauncher(jit, spec["options"], key)
    args: list[object] = []
    for item in spec["arguments"]:
        if item["kind"] == "pointer":
            args.append(
                make_ptr(
                    getattr(cutlass, item["dtype"]),
                    0,
                    cute.AddressSpace.gmem,
                    assumed_align=item["alignment"],
                )
            )
        else:
            assert item["kind"] == "scalar"
            args.append(thaw(item["value"]))
    args.append(CUstream(0))
    started = time.monotonic()
    compiled = _compile_native(cute.compile, jit, args, spec["options"])
    seconds = time.monotonic() - started
    compiled_launcher._persist_to_disk(compiled)
    _, ir, metadata = compiled_launcher._cache_file_paths()
    sidecar = json.loads(Path(metadata).read_text())
    assert sidecar == {"function_name": compiled.function_name, "has_gpu_module": True}
    assert compiled.has_gpu_module
    retained.append((module, kernel, jit, args, compiled, compiled_launcher))
    return {
        "cache_key": key,
        "source_sha256": spec["source_sha256"],
        "ir": ir,
        "metadata": metadata,
        "ir_sha256": sha(ir),
        "metadata_sha256": sha(metadata),
        "symbol": sidecar,
        "compile_seconds": seconds,
    }


def main(batch_path: Path) -> None:
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and (not sys.flags.optimize)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
    batch = json.loads(batch_path.read_text())
    output = batch_path.parent
    calls = []
    result: dict[str, Any] = {"status": "failed", "pid": os.getpid(), "requests": []}
    started = time.monotonic()
    try:
        assert sha(batch["sdk"]["loader"]) == batch["sdk"]["loader_sha256"]
        observer = DriverMetadata(batch["sdk"]["loader"], allow_query=True, calls=calls)
        observer.__enter__()
        sys.path[:0] = batch["sys_path"]
        importlib.import_module("cutlass")
        metadata = observer.finish()
        assert metadata == batch["sdk"]["metadata"]
        import torch

        import helion
        from helion.runtime.cute.native_pool_guards import worker_guards

        assert str(Path(helion.__file__).resolve().parent.parent) == batch["runtime"]
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()
        guard = worker_guards(calls)
        guard_count = guard.__enter__()
        ready = time.monotonic()
        retained = []

        def check() -> None:
            assert observer.finish() == metadata and (not calls)
            assert not torch.cuda.is_initialized()

        result.update(
            status="running",
            guards=guard_count,
            forbidden_calls=calls,
            driver_metadata=metadata,
            retained_compilations=0,
            startup_seconds=ready - started,
        )
        _compile_requests(batch["requests"], output, retained, result, check)
        result["seconds"] = time.monotonic() - started
    except BaseException:
        result.update(
            status="failed", error=traceback.format_exc(), forbidden_calls=calls
        )
    finally:
        write(output / "RESULT.json", result)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if result["status"] in ("passed", "compile_error") else 1)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
