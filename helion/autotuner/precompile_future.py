from __future__ import annotations

import collections
import contextlib
import dataclasses
import importlib.util
import inspect
import linecache
import multiprocessing as mp
from multiprocessing import connection
import os
from pathlib import Path
import pickle
import signal
import sys
import time
import traceback
import types
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import NoReturn
from typing import cast
import uuid

from .. import exc
from ..runtime.precompile_shim import already_compiled
from ..runtime.precompile_shim import already_compiled_fail
from ..runtime.precompile_shim import make_precompiler
from .benchmarking import clear_jit_fast_path_caches
from .benchmarking import synchronize_device
from .kernel_args import load_trusted_kernel_args
from .logger import SUPPRESSED_TRITON_CODE_MSG
from .logger import capture_output
from .logger import classify_triton_exception
from .logger import format_triton_compile_failure
from .logger import log_generated_triton_code_debug
from .logger import match_unrecoverable_runtime_error
from .logger import maybe_dump_triton_failure
from .process_utils import isolate_forked_worker
from .process_utils import signal_process_tree
from .process_utils import start_isolated_process_group
from .process_utils import start_without_collection
from .progress_bar import iter_with_progress

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from ..runtime.kernel import CompiledConfig
    from ..runtime.settings import Settings
    from .base_search import _AutotunableKernel
    from .logger import AutotuningLogger


@dataclasses.dataclass
class PrecompileContext:
    """Narrow context that PrecompileFuture uses instead of a back-reference
    to the full search object.

    Attributes:
        settings: Autotuning settings (compile timeout, ignore_errors, etc.).
        log: Logger for warnings/debug messages.
        kernel: The kernel being autotuned (used for error reporting).
        args: The kernel arguments (used for repro logging on failure).
        jobs: Maximum number of concurrent precompile processes.
        report_failures: Whether a failed precompile is reported: logged,
            dumped, a repro printed, an unexpected error raised. Off for a
            caller that compiles the config in-process next and reports its
            failure there (the accuracy baseline): whether the compile
            finished, outran the timeout or crashed its child
            (``PrecompileFuture.exit_signal``) and the child's error
            (``failure_error()``, which the baseline checks for an
            unrecoverable one) are left, and an unrecoverable runtime error
            from the fork precompile's host code is still raised.
    """

    settings: Settings
    log: AutotuningLogger
    kernel: _AutotunableKernel
    args: Sequence[object]
    jobs: int
    report_failures: bool = True


def _write_result_file(result_path: str, message: dict[str, object]) -> None:
    tmp_path = f"{result_path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(message, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, result_path)


class _ExtractedLaunchArgs(Exception):
    """Exception that carries kernel launch arguments for precompiler extraction."""

    def __init__(
        self,
        kernel: object,
        grid: tuple[int, ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.grid = grid
        self.args = args
        self.kwargs = kwargs


@dataclasses.dataclass
class SerializedCompiledFunction:
    function_name: str
    source_code: str
    filename: str | None
    module_name: str | None
    source_hash: str | None = None
    # (origin_module_name, origin_file_path) for every kernel-source module the
    # generated code imports (e.g. `import <origin> as _source_module` for a
    # module-global like a dtype constant). The worker re-registers these from
    # their files before exec so the import resolves even when the origin module
    # name is not importable in a fresh process (notebook / exec / import-by-path
    # kernels, whose synthetic module name only exists in the parent).
    source_modules: list[tuple[str, str]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class RemoteError:
    exc_type: str
    exc_module: str | None
    # The exception's str(), which to_exception() rebuilds it from. Its args
    # would read as their tuple when its class formats its own message
    # (Triton's CompilationError: source, AST node and error), and a child's
    # args that pickle cannot serialize would lose its whole result.
    message: str
    traceback: str | None
    classification: str | None
    captured_output: str | None = None

    def to_exception(self) -> Exception:
        exc_cls = types.new_class(self.exc_type, (Exception,))
        exc_cls.__module__ = self.exc_module or __name__
        exc_obj = exc_cls(self.message)
        exc_obj.remote_traceback = self.traceback
        # The child's classification of the real error, which no type rule
        # can match on this stand-in of its class.
        exc_obj.remote_classification = self.classification
        return exc_obj


def _serialize_compiled_fn(fn: CompiledConfig) -> SerializedCompiledFunction:
    if "<locals>" in getattr(fn, "__qualname__", ""):
        raise RuntimeError("Unable to serialize nested compiled functions")
    module_name = getattr(fn, "__module__", None)
    module = sys.modules.get(module_name) if module_name is not None else None
    filename: str | None = None
    source_code: str | None = None
    if module is not None:
        filename = getattr(module, "__file__", None)
        if filename is not None and os.path.exists(filename):
            source_code = Path(filename).read_text(encoding="utf-8")
        if source_code is None:
            with contextlib.suppress(OSError, TypeError):
                source_code = inspect.getsource(module)
    if source_code is None:
        raise RuntimeError("Unable to capture source for compiled kernel")
    source_hash = getattr(fn, "_helion_cute_source_hash", None)
    if not isinstance(source_hash, str):
        source_hash = None
    fn_globals = getattr(fn, "__globals__", None)
    if isinstance(fn_globals, dict) and not isinstance(
        getattr(fn, "_helion_cute_kernels", None), tuple
    ):
        # A conditional bundle can retain the legacy primary kernel name for
        # its fallback. That kernel has its own hash; it is not the host
        # bundle's source identity.
        kernel = fn_globals.get(f"_helion_{fn.__name__}")
        value = getattr(kernel, "_helion_cute_source_hash", None)
        if isinstance(value, str):
            source_hash = value
    return SerializedCompiledFunction(
        function_name=fn.__name__,
        source_code=source_code,
        filename=filename,
        module_name=module_name,
        source_hash=source_hash,
        source_modules=_collect_source_modules(module),
    )


def _collect_source_modules(module: types.ModuleType | None) -> list[tuple[str, str]]:
    """Capture (name, file) for every kernel-origin module the generated module
    imports into its globals.

    Helion codegen emits ``import <origin> as _source_module`` (and
    ``_global_source<N>`` for other modules) whenever the kernel references a
    module-global. The worker re-execs the generated source in a fresh process,
    so any such import must resolve there. A kernel defined in a normal importable
    module is fine, but one loaded by path / exec / notebook lives under a
    synthetic module name that only exists in the parent process. Shipping the
    origin module's real file lets the worker re-register it under the same name.
    """
    if module is None:
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in vars(module).values():
        if not isinstance(value, types.ModuleType):
            continue
        name = getattr(value, "__name__", None)
        file = getattr(value, "__file__", None)
        if not isinstance(name, str) or name in seen:
            continue
        if not isinstance(file, str) or not os.path.exists(file):
            continue
        # Skip modules a fresh worker can import on its own (real packages like
        # torch/triton, or an example on PYTHONPATH) -- shipping/re-exec'ing them
        # would be wasteful or harmful. A by-path / notebook / exec module lives
        # under a synthetic name that only the parent's sys.modules knows about,
        # so it is NOT freshly importable and gets shipped with its real file.
        if _is_freshly_importable(name, file):
            continue
        seen.add(name)
        out.append((name, file))
    return out


def _is_freshly_importable(name: str, loaded_file: str) -> bool:
    """Would a fresh process import ``name`` and get the file we actually loaded?

    ``importlib.util.find_spec`` normally returns the cached ``sys.modules``
    entry, which would mask synthetic by-path modules. Temporarily drop the
    cached entry (restoring it afterwards) so the lookup reflects what a worker
    -- with an empty module cache -- would resolve.
    """
    saved = sys.modules.pop(name, None)
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        spec = None
    finally:
        if saved is not None:
            sys.modules[name] = saved
    origin = getattr(spec, "origin", None)
    if not isinstance(origin, str):
        return False
    return os.path.realpath(origin) == os.path.realpath(loaded_file)


def _register_source_module(name: str, file: str) -> None:
    """Re-register a kernel-origin module from its file under ``name`` so the
    generated code's ``import <name> as _source_module`` resolves in the worker.

    No-op if the name is already present (e.g. a real importable package the
    serializer chose not to ship, or a sibling job already loaded it)."""
    if name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(name, file)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so self-references / decorators resolve by name.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Never leave a partially initialized module behind.  A long-lived
        # benchmark worker may otherwise let a later generated wrapper import
        # the broken module successfully and fail only when it reads a missing
        # global, incorrectly making a valid candidate look broken.
        sys.modules.pop(name, None)
        raise


def _load_compiled_fn(fn_spec: SerializedCompiledFunction) -> CompiledConfig:
    for origin_name, origin_file in fn_spec.source_modules:
        _register_source_module(origin_name, origin_file)
    module_name = f"_helion_autotune_subprocess_{uuid.uuid4().hex}"
    module = types.ModuleType(module_name)
    module.__file__ = fn_spec.filename or "<helion-autotune-subprocess>"
    module.__loader__ = None
    module.__package__ = None
    sys.modules[module_name] = module
    try:
        exec(
            compile(fn_spec.source_code, module.__file__, "exec"),
            module.__dict__,
        )
        fn = getattr(module, fn_spec.function_name, None)
        if fn is None:
            raise RuntimeError(
                f"Unable to locate compiled kernel '{fn_spec.function_name}' "
                "in generated module"
            )
        cute_kernel = module.__dict__.get(f"_helion_{fn_spec.function_name}")
        if (
            cute_kernel is not None
            and fn_spec.source_hash is not None
            and not isinstance(getattr(fn, "_helion_cute_kernels", None), tuple)
        ):
            # Backend annotation is runtime metadata and is not present in the
            # generated source. Carry the parent's exact value because
            # PyCodeCache treats strip-equivalent source as one file key.
            with contextlib.suppress(AttributeError, TypeError):
                cute_kernel._helion_cute_source_hash = fn_spec.source_hash
        if fn_spec.source_hash is not None:
            with contextlib.suppress(AttributeError, TypeError):
                fn._helion_cute_source_hash = fn_spec.source_hash
        return fn
    except BaseException:
        sys.modules.pop(module_name, None)
        raise


def _unload_compiled_fn(fn: CompiledConfig) -> None:
    """Release caches and the temporary module created by ``_load_compiled_fn``."""
    clear_jit_fast_path_caches(fn)
    module_name = getattr(fn, "__module__", "")
    if module_name.startswith("_helion_autotune_subprocess_"):
        module = sys.modules.get(module_name)
        if module is not None:
            kernels = getattr(fn, "_helion_cute_kernels", None)
            if not isinstance(kernels, tuple):
                kernels = (module.__dict__.get(f"_helion_{fn.__name__}"),)
            for cute_kernel in kernels:
                if cute_kernel is None:
                    continue
                launchers = getattr(
                    cute_kernel, "_helion_cute_compiled_launchers", None
                )
                if isinstance(launchers, dict):
                    for launcher in launchers.values():
                        jit_func = getattr(launcher, "_jit_func", None)
                        wrapped = getattr(jit_func, "__wrapped__", jit_func)
                        code = getattr(wrapped, "__code__", None)
                        filename = getattr(code, "co_filename", "")
                        if filename.startswith("<helion_cute_launcher:"):
                            linecache.cache.pop(filename, None)
                for cache_name in (
                    "_helion_cute_compiled_launchers",
                    "_helion_cute_launch_arg_cache",
                ):
                    cache = getattr(cute_kernel, cache_name, None)
                    if isinstance(cache, dict):
                        cache.clear()
        sys.modules.pop(module_name, None)


def _detach_error(error: BaseException) -> None:
    """Drop what ``error`` keeps alive of the code that raised it: the
    tracebacks along its chain, whose frames hold the host code's tensors,
    and an ``_ExtractedLaunchArgs`` in it, which holds the launch's
    arguments (a bind error raised while handling one)."""
    seen: set[int] = set()
    pending: list[BaseException] = [error]
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        node.__traceback__ = None
        if isinstance(node.__cause__, _ExtractedLaunchArgs):
            node.__cause__ = None
        if isinstance(node.__context__, _ExtractedLaunchArgs):
            node.__context__ = None
        pending += [link for link in (node.__cause__, node.__context__) if link]


def cached_compile_failure_error() -> RuntimeError:
    """Why a kernel an earlier compile in this process cached but could not
    load (``already_compiled_fail``) fails again: the precompile phase and the
    in-process compile report the same error."""
    return RuntimeError(
        "an earlier compile of this kernel in this process could not load it"
    )


def _write_error_result(
    result_path: str, exc: Exception, decorator: str, captured_output: str
) -> None:
    """Report a precompile child's error to the parent: its type, its str()
    (its arguments may not pickle; an exception that formats its own message
    reads as a tuple of them) and its classification."""
    with contextlib.suppress(Exception):
        try:
            classification = classify_triton_exception(exc)
        except Exception:
            classification = None
        _write_result_file(
            result_path,
            {
                "status": "error",
                "traceback": "".join(traceback.format_exception(exc)),
                "decorator": decorator,
                "exc_type": type(exc).__name__,
                "exc_module": type(exc).__module__,
                "message": str(exc),
                "classification": classification,
                "captured_output": captured_output or None,
            },
        )


def _run_kernel_in_subprocess_spawn(
    fn_spec: SerializedCompiledFunction,
    args_path: str,
    result_path: str,
    decorator: str,
) -> None:
    start_isolated_process_group()
    # Exit status 0 only after the "ok" result (see the fork child below).
    status = 1
    _cap: list[str] = [""]
    try:
        fn = _load_compiled_fn(fn_spec)
        args = load_trusted_kernel_args(args_path)
        assert isinstance(args, (tuple, list))
        synchronize_device()
        with capture_output() as _cap:
            # Keep asynchronous device buffers alive until execution completes.
            _output = fn(*args)
        synchronize_device()
        _write_result_file(result_path, {"status": "ok"})
        status = 0
    except Exception as exc:
        _write_error_result(result_path, exc, decorator, _cap[0])
    finally:
        os._exit(status)


def _prepare_precompiler_for_fork(
    fn: CompiledConfig,
    args: Sequence[object],
    config: Config,
    kernel: _AutotunableKernel,
) -> Callable[[], bool]:
    def extract_launcher(
        triton_kernel: object,
        grid: tuple[int, ...],
        *launch_args: object,
        **launch_kwargs: object,
    ) -> NoReturn:
        raise _ExtractedLaunchArgs(triton_kernel, grid, launch_args, launch_kwargs)

    try:
        fn(*args, _launcher=extract_launcher)
        raise RuntimeError("Expected _ExtractedLaunchArgs to be raised")
    except _ExtractedLaunchArgs as extracted:
        return make_precompiler(
            cast("Any", extracted.kernel),
            config,
            cast("BoundKernel", kernel),
        )(*extracted.args, **extracted.kwargs)


def _run_kernel_in_subprocess_fork(
    precompiler: Callable[[], bool],
    config: Config,
    kernel: _AutotunableKernel,
    result_path: str,
    decorator: str,
) -> None:
    start_isolated_process_group()
    isolate_forked_worker()
    # Exit status 0 tells the parent the config compiled, so it is set only
    # once the "ok" result is written: an exception the handler below does
    # not catch (SystemExit, KeyboardInterrupt) still exits nonzero.
    status = 1
    _cap: list[str] = [""]
    try:
        with capture_output() as _cap:
            precompiler()
        _write_result_file(result_path, {"status": "ok"})
        status = 0
    except Exception as exc:
        _write_error_result(result_path, exc, decorator, _cap[0])
    finally:
        os._exit(status)


@dataclasses.dataclass
class PrecompileFuture:
    """
    Wraps a child process where we are precompiling a kernel.

    Attributes:
        search (BaseSearch): The search object that initiated the precompilation.
        config (Config): The configuration to be precompiled.
        process (mp.Process | None): The process running the precompilation.
        timeout (float): The timeout for the precompilation.
        start_time (float): The time when the precompilation started.
        end_time (float | None): The time when the precompilation ended.
        ok (bool | None): The result of the precompilation (True if successful, False otherwise).
    """

    ctx: PrecompileContext
    config: Config
    process: mp.Process | None
    timeout: float
    # Set when the process is actually started. For queued futures this is None.
    start_time: float | None = None
    end_time: float | None = None
    ok: bool | None = None
    result_path: str | None = None
    _result_received: bool = False
    remote_error: RemoteError | None = None
    _remote_error_handled: bool = False
    # The error that rejected the config before a child started: preparing
    # the fork precompile (running the host code to find the kernel's launch,
    # binding its arguments) raised it, detached (``_detach_error``), or a
    # kernel an earlier compile in this process failed to load
    # (``cached_compile_failure_error``).
    prepare_error: Exception | None = None
    failure_reason: Literal["ok", "error", "timeout"] | None = None
    # The signal that ended the child when it died before writing a result:
    # a crash in the backend compiler (an LLVM or MLIR abort, a segfault)
    # rather than an error it reported or the deadline.
    exit_signal: int | None = None

    @property
    def elapsed(self) -> float:
        """Return the elapsed time since the start of the precompilation."""
        if self.start_time is None:
            return 0.0
        if self.end_time is not None:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def seconds_left(self) -> float:
        """Return the number of seconds left before the timeout."""
        if self.end_time is not None:
            return 0
        if self.start_time is None:
            return self.timeout
        return self.timeout - (time.time() - self.start_time)

    def is_alive(self) -> bool:
        """Check if the precompilation process is still alive."""
        if (p := self.process) is None:
            return False
        return p.is_alive()

    @property
    def started(self) -> bool:
        """Whether the process has been started."""
        return self.start_time is not None

    def start(self) -> None:
        """Start the underlying process and set the timer if not already started."""
        if self.process is None or self.started:
            return
        self.start_time = time.time()
        start_without_collection(self.process)

    @staticmethod
    def skip(
        ctx: PrecompileContext,
        config: Config,
        ok: bool,
        prepare_error: Exception | None = None,
    ) -> PrecompileFuture:
        """Dummy precompile future that is already done; ``prepare_error`` is
        the error that rejected the config, when known."""
        ts = time.time()
        return PrecompileFuture(
            ctx=ctx,
            config=config,
            process=None,
            timeout=0,
            ok=ok,
            start_time=ts,
            end_time=ts,
            result_path=None,
            _result_received=True,
            remote_error=None,
            _remote_error_handled=True,
            prepare_error=prepare_error,
            failure_reason="ok" if ok else "error",
        )

    @staticmethod
    def create(
        ctx: PrecompileContext,
        config: Config,
        fn: CompiledConfig,
        args: Sequence[object],
        result_path: str,
        args_path: str | None,
    ) -> PrecompileFuture:
        """Create a PrecompileFuture by spawning or forking a subprocess.

        Handles fork-vs-spawn mode selection, serialization, and process
        construction.  Returns a done ``skip`` future when no child runs: in
        fork mode when the outcome is known without one -- a kernel an earlier
        compile in this process cached, compiled or not
        (``cached_compile_failure_error``), or an error that keeps the child
        from starting (``prepare_error``) -- and in spawn mode, without
        ``report_failures``, for a kernel that cannot be serialized, which the
        caller compiles in-process.
        """
        mode = ctx.settings.autotune_precompile
        decorator = ctx.kernel.format_kernel_decorator(config, ctx.settings)

        if mode == "spawn":
            mp_ctx = mp.get_context("spawn")
            assert args_path is not None
            try:
                fn_spec = _serialize_compiled_fn(fn)
            except RuntimeError as err:
                if not ctx.report_failures:
                    # The caller compiles the config in-process next.
                    return PrecompileFuture.skip(ctx, config, False)
                raise exc.AutotuneError(
                    "Failed to serialize compiled kernel for spawn precompile."
                    ' Set HELION_AUTOTUNE_PRECOMPILE="fork" to fall back to fork mode.'
                ) from err
            process = cast(
                "mp.Process",
                mp_ctx.Process(
                    target=_run_kernel_in_subprocess_spawn,
                    args=(fn_spec, args_path, result_path, decorator),
                ),
            )
            process.daemon = True
        else:
            try:
                precompiler = _prepare_precompiler_for_fork(
                    fn, args, config, ctx.kernel
                )
            except Exception as e:
                action = classify_triton_exception(e)
                if ctx.report_failures:
                    maybe_dump_triton_failure(ctx.kernel, config, e)
                    if not ctx.settings.autotune_ignore_errors:
                        log_generated_triton_code_debug(
                            ctx.log,
                            ctx.kernel,
                            config,
                            prefix=f"Generated Triton code for {decorator}:",
                        )
                        log = ctx.log.debug if action == "debug" else ctx.log.warning
                        log(
                            f"Helion autotuner precompile error for {decorator}.",
                            SUPPRESSED_TRITON_CODE_MSG,
                            exc_info=True,
                        )
                _detach_error(e)
                if match_unrecoverable_runtime_error(e):
                    raise
                if (
                    action == "raise"
                    and ctx.report_failures
                    and not ctx.settings.autotune_ignore_errors
                ):
                    raise
                return PrecompileFuture.skip(ctx, config, False, prepare_error=e)
            if precompiler is already_compiled or precompiler is already_compiled_fail:
                # Compiled earlier in this process, so the outcome is known
                # (failed: a kernel ``_triton_compile`` compiled but could not
                # load). A child would ignore the sentinel's result and report
                # success.
                return PrecompileFuture.skip(
                    ctx,
                    config,
                    precompiler is already_compiled,
                    prepare_error=(
                        cached_compile_failure_error()
                        if precompiler is already_compiled_fail
                        else None
                    ),
                )
            mp_ctx = mp.get_context("fork")
            process = cast(
                "mp.Process",
                mp_ctx.Process(
                    target=_run_kernel_in_subprocess_fork,
                    args=(precompiler, config, ctx.kernel, result_path, decorator),
                ),
            )
            process.daemon = True
        return PrecompileFuture(
            ctx=ctx,
            config=config,
            process=process,
            timeout=ctx.settings.autotune_compile_timeout,
            result_path=result_path,
        )

    def __call__(self) -> bool:
        """Wait for the precompilation to finish and return true on success."""
        if self.ok is not None:
            return self.ok
        process = self.process
        assert process is not None
        try:
            # Start now if not already started (single-future path)
            if not self.started:
                self.start()
            process.join(self.seconds_left())
        finally:
            self._mark_complete()
        self._consume_result(raise_on_raise=True)
        assert self.ok is not None
        return self.ok

    def failure_error(self) -> Exception | None:
        """Why the precompile rejected the config, for the compile-failure
        summaries: the deadline it outran, the signal its child died from, the
        child's error, or the error that kept a child from starting or that a
        kernel compiled earlier in this process failed with
        (``prepare_error``); None when none is known (a spawn-mode kernel
        that cannot be serialized, without ``report_failures``)."""
        if self.failure_reason == "timeout":
            return TimeoutError(
                f"compile exceeded autotune_compile_timeout={self.timeout}s "
                "(HELION_AUTOTUNE_COMPILE_TIMEOUT) in the precompile subprocess"
            )
        if self.exit_signal is not None:
            return ChildProcessError(
                f"the precompile subprocess died from signal {self.exit_signal} "
                f"({signal.strsignal(self.exit_signal) or 'unknown'}) before "
                "reporting a result"
            )
        if self.remote_error is not None:
            return self.remote_error.to_exception()
        return self.prepare_error

    @staticmethod
    def wait_for_all(
        futures: list[PrecompileFuture],
        desc: str | None = None,
    ) -> list[bool]:
        """
        Wait for all precompile futures to complete.

        Args:
            futures: A list of PrecompileFuture objects.
            desc: Optional description used for the progress display.

        Returns:
            A list of boolean values indicating completion status.
        """
        progress = iter_with_progress(
            range(len(futures)),
            total=len(futures),
            description=desc,
            enabled=desc is not None,
        )
        next(progress, None)  # display the progress bar immediately
        progress_left = len(futures)
        remaining = [f for f in futures if f.ok is None]
        try:
            while remaining:
                remaining = PrecompileFuture._wait_for_all_step(remaining)
                while progress_left > len(remaining):
                    next(progress, None)
                    progress_left -= 1
        except BaseException:
            PrecompileFuture._cancel_all(futures)
            raise
        result = []
        for f in futures:
            assert f.ok is not None
            if f.failure_reason is None:
                f.failure_reason = "ok" if f.ok else "error"
            result.append(f.ok)
        return result

    @staticmethod
    def _wait_for_all_step(
        futures: list[PrecompileFuture],
    ) -> list[PrecompileFuture]:
        """Start up to the concurrency cap, wait for progress, and return remaining futures."""
        cap = futures[0].ctx.jobs if futures else 1
        running = [f for f in futures if f.started and f.ok is None and f.is_alive()]

        # Start queued futures up to the cap
        queued = collections.deque(f for f in futures if not f.started and f.ok is None)
        while len(running) < cap and queued:
            job = queued.popleft()
            job.start()
            if job.is_alive():
                running.append(job)

        # Wait for at least one to finish or time out
        timeout = min([f.seconds_left() for f in running], default=0.0)
        handles = [f.process.sentinel for f in running if f.process is not None]
        if handles and timeout > 0:
            connection.wait(handles, timeout)
        remaining: list[PrecompileFuture] = []
        for f in futures:
            if f.ok is not None:
                continue
            if f.started and (not f.is_alive() or f.seconds_left() <= 0):
                f._mark_complete()
                f._consume_result(raise_on_raise=True)
            else:
                remaining.append(f)
        return remaining

    @staticmethod
    def _cancel_all(futures: Iterable[PrecompileFuture]) -> None:
        """Cancel any futures that have not completed."""
        active = [future for future in futures if future.ok is None]
        for future in active:
            with contextlib.suppress(Exception):
                future._kill_without_wait()
        for future in active:
            with contextlib.suppress(Exception):
                future.cancel()

    def _kill_without_wait(self) -> None:
        """Issue a hard kill to the underlying process without waiting for exit."""
        process = self.process
        if process is None or not self.started:
            return
        with contextlib.suppress(Exception):
            signal_process_tree(process, signal.SIGKILL)

    def cancel(self) -> None:
        """Terminate the underlying process (if any) without waiting for success."""
        self.end_time = time.time()
        process = self.process
        if process is not None:
            if self.started:
                with contextlib.suppress(Exception):
                    signal_process_tree(process, signal.SIGKILL)
                    process.join()
        if self.ok is None:
            self.ok = False
        if self.failure_reason is None:
            self.failure_reason = "error"
        self._consume_result(raise_on_raise=False)

    def _mark_complete(self) -> bool:
        """
        Mark the precompile future as complete and kill the process if needed.

        Returns:
            True if the precompilation was successful, False otherwise.
        """
        self.end_time = time.time()
        process = self.process
        assert process is not None
        # If the process hasn't been started yet (shouldn't happen in normal flow),
        # start and immediately terminate to maintain invariants.
        if not self.started:
            self.start()
        if not process.is_alive():
            self.ok = process.exitcode == 0
            if not self.ok:
                signal_process_tree(process, signal.SIGKILL)
            reported = self._result_received or (
                self.result_path is not None and os.path.exists(self.result_path)
            )
            self._consume_result(raise_on_raise=False)
            if self.ok:
                self.failure_reason = "ok"
            elif self.failure_reason is None:
                self.failure_reason = "error"
            exitcode = process.exitcode
            if not reported and exitcode is not None and exitcode < 0:
                self.exit_signal = -exitcode
            return self.ok
        signal_process_tree(process, signal.SIGTERM)
        process.join(10)
        msg = f"Timeout after {self.elapsed:.0f}s compiling {self.config}"
        report = (
            self.ctx.report_failures and not self.ctx.settings.autotune_ignore_errors
        )
        if process.is_alive():
            if report:
                self.ctx.log.warning(
                    msg,
                    "(SIGKILL required)",
                )
        elif report:
            self.ctx.log.warning(msg)
        # The worker can exit on SIGTERM before a compiler grandchild does.
        # Always hard-kill the process group after the grace period.
        signal_process_tree(process, signal.SIGKILL)
        process.join()

        self.ok = False
        self.failure_reason = "timeout"
        self._consume_result(raise_on_raise=False)
        return False

    def _consume_result(self, *, raise_on_raise: bool) -> None:
        if not self._result_received and self.result_path is not None:
            message_data: dict[str, object] | None = None
            try:
                with open(self.result_path, "rb") as f:
                    message_data = pickle.load(f)
            except FileNotFoundError:
                message_data = None
            except Exception as err:
                if self.remote_error is None:
                    self.remote_error = RemoteError(
                        exc_type=type(err).__name__,
                        exc_module=type(err).__module__,
                        message=str(err),
                        traceback=None,
                        classification="warn",
                    )
            finally:
                with contextlib.suppress(Exception):
                    os.remove(self.result_path)
            if message_data is None:
                if self.failure_reason == "timeout":
                    # Timeout warnings have already been emitted; suppress secondary EOF logs.
                    self.remote_error = None
                    self._remote_error_handled = True
                elif self.remote_error is None:
                    self.remote_error = RemoteError(
                        exc_type="EOFError",
                        exc_module=__name__,
                        message="No result received from subprocess.",
                        traceback=None,
                        classification="debug",
                    )
            elif message_data["status"] == "ok":
                if self.ok is None:
                    self.ok = True
                assert self.remote_error is None
            else:
                self.remote_error = RemoteError(
                    exc_type=cast("str", message_data["exc_type"]),
                    exc_module=cast("str | None", message_data["exc_module"]),
                    message=cast("str", message_data["message"]),
                    traceback=cast("str | None", message_data["traceback"]),
                    classification=cast("str | None", message_data["classification"]),
                    captured_output=cast(
                        "str | None", message_data.get("captured_output")
                    ),
                )
                self.ok = False
            self.result_path = None
            self._result_received = True

        error = self.remote_error
        if error is None or self._remote_error_handled:
            return
        if not self.ctx.report_failures:
            # The caller compiles the config in-process next and reports it.
            self._remote_error_handled = True
            return
        exc_obj = error.to_exception()
        maybe_dump_triton_failure(
            self.ctx.kernel,
            self.config,
            exc_obj,
            remote_traceback=error.traceback,
            captured_output=error.captured_output,
        )
        classification = error.classification or classify_triton_exception(exc_obj)
        ignore_errors = self.ctx.settings.autotune_ignore_errors
        if ignore_errors:
            classification = "debug"
        # A fork child only compiles the kernel, so an error there, even one
        # the classifier does not expect ("raise", such as Triton's frontend
        # CompilationError), fails only this config: it is logged as a warning
        # and the config skipped, as the benchmark worker skips a config it
        # cannot compile. A spawned child also runs the generated host code
        # and the kernel; an unexpected error there still raises.
        if (
            classification == "raise"
            and self.ctx.settings.autotune_precompile == "spawn"
        ):
            if raise_on_raise:
                self._remote_error_handled = True
                decorator = self.ctx.kernel.format_kernel_decorator(
                    self.config, self.ctx.settings
                )
                log_generated_triton_code_debug(
                    self.ctx.log,
                    self.ctx.kernel,
                    self.config,
                    prefix=f"Generated Triton code for {decorator}:",
                )
                self.ctx.kernel.maybe_log_repro(
                    self.ctx.log.error, self.ctx.args, self.config
                )
                raise exc.TritonError(
                    error=f"{type(exc_obj).__qualname__}: {exc_obj}",
                    decorator=decorator,
                    code=SUPPRESSED_TRITON_CODE_MSG,
                ) from exc_obj
            return

        decorator = self.ctx.kernel.format_kernel_decorator(
            self.config, self.ctx.settings
        )
        log_generated_triton_code_debug(
            self.ctx.log,
            self.ctx.kernel,
            self.config,
            prefix=f"Generated Triton code for {decorator}:",
        )
        if not ignore_errors:
            # A fork child's unexpected error (above) is shown by default; the
            # traceback, through Triton's frames, stays at debug level.
            log = (
                self.ctx.log.warning
                if classification == "raise"
                else self.ctx.log.debug
            )
            log(format_triton_compile_failure(self.config, exc_obj, self.ctx.kernel))
            if error.traceback:
                self.ctx.log.debug(f"Remote traceback (subprocess):\n{error.traceback}")
            # The repro of a compile error ("warn": ptxas, PassManager) is a
            # warning too, as on the in-process path, so HELION_PRINT_REPRO=1
            # shows it although the error itself is logged at debug level.
            self.ctx.kernel.maybe_log_repro(
                self.ctx.log.warning
                if classification in ("raise", "warn")
                else self.ctx.log.debug,
                self.ctx.args,
                self.config,
            )
        self._remote_error_handled = True
