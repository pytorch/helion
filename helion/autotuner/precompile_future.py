from __future__ import annotations

import collections
import contextlib
import dataclasses
import inspect
import multiprocessing as mp
from multiprocessing import connection
import os
from pathlib import Path
import pickle
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

import torch

from .. import exc
from ..runtime.precompile_shim import already_compiled
from ..runtime.precompile_shim import make_precompiler
from .benchmarking import synchronize_device
from .logger import SUPPRESSED_TRITON_CODE_MSG
from .logger import capture_output
from .logger import classify_triton_exception
from .logger import format_triton_compile_failure
from .logger import log_generated_triton_code_debug
from .logger import maybe_dump_triton_failure
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
    """

    settings: Settings
    log: AutotuningLogger
    kernel: _AutotunableKernel
    args: Sequence[object]
    jobs: int


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


@dataclasses.dataclass
class RemoteError:
    exc_type: str
    exc_module: str | None
    exc_args: tuple[object, ...]
    traceback: str | None
    classification: str | None
    captured_output: str | None = None

    def to_exception(self) -> Exception:
        exc_cls = types.new_class(self.exc_type, (Exception,))
        exc_cls.__module__ = self.exc_module or __name__
        exc_obj = exc_cls(*self.exc_args)
        exc_obj.remote_traceback = self.traceback
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
    return SerializedCompiledFunction(
        function_name=fn.__name__,
        source_code=source_code,
        filename=filename,
        module_name=module_name,
    )


def _load_compiled_fn(fn_spec: SerializedCompiledFunction) -> CompiledConfig:
    module_name = f"_helion_autotune_subprocess_{uuid.uuid4().hex}"
    module = types.ModuleType(module_name)
    module.__file__ = fn_spec.filename or "<helion-autotune-subprocess>"
    module.__loader__ = None
    module.__package__ = None
    sys.modules[module_name] = module
    exec(
        compile(fn_spec.source_code, module.__file__, "exec"),
        module.__dict__,
    )
    fn = getattr(module, fn_spec.function_name, None)
    if fn is None:
        raise RuntimeError(
            f"Unable to locate compiled kernel '{fn_spec.function_name}' in generated module"
        )
    return fn


def _run_kernel_in_subprocess_spawn(
    fn_spec: SerializedCompiledFunction,
    args_path: str,
    result_path: str,
    decorator: str,
) -> None:
    status = 0
    _cap: list[str] = [""]
    try:
        fn = _load_compiled_fn(fn_spec)
        args = torch.load(args_path)
        assert isinstance(args, (tuple, list))
        synchronize_device(None)
        with capture_output() as _cap:
            result = fn(*args)
        synchronize_device(result)
        _write_result_file(result_path, {"status": "ok"})
    except Exception as exc:
        status = 1
        with contextlib.suppress(Exception):
            try:
                exc_args = tuple(exc.args)
            except Exception:
                exc_args = (str(exc),)
            try:
                classification = classify_triton_exception(exc)
            except Exception:
                classification = None
            _write_result_file(
                result_path,
                {
                    "status": "error",
                    "traceback": traceback.format_exc(),
                    "decorator": decorator,
                    "exc_type": type(exc).__name__,
                    "exc_module": type(exc).__module__,
                    "exc_args": exc_args,
                    "classification": classification,
                    "captured_output": _cap[0] or None,
                },
            )
    finally:
        os._exit(status)


def _prepare_precompiler_for_fork(
    fn: CompiledConfig,
    args: Sequence[object],
    config: Config,
    kernel: _AutotunableKernel,
    decorator: str,
    logger: AutotuningLogger,
) -> Callable[[], bool] | None:
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
        precompiler = make_precompiler(
            cast("Any", extracted.kernel),
            config,
            cast("BoundKernel", kernel),
        )(*extracted.args, **extracted.kwargs)
        if precompiler is already_compiled:
            return None
        return precompiler
    except Exception as e:
        maybe_dump_triton_failure(kernel, config, e)
        log_generated_triton_code_debug(
            logger,
            kernel,
            config,
            prefix=f"Generated Triton code for {decorator}:",
        )
        logger.warning(
            "Helion autotuner precompile error for %s. %s",
            decorator,
            SUPPRESSED_TRITON_CODE_MSG,
            exc_info=True,
        )
        raise


def _run_kernel_in_subprocess_fork(
    precompiler: Callable[[], None],
    config: Config,
    kernel: _AutotunableKernel,
    result_path: str,
    decorator: str,
) -> None:
    status = 0
    _cap: list[str] = [""]
    try:
        with capture_output() as _cap:
            precompiler()
        _write_result_file(result_path, {"status": "ok"})
    except Exception as exc:
        status = 1
        with contextlib.suppress(Exception):
            try:
                exc_args = tuple(exc.args)
            except Exception:
                exc_args = (str(exc),)
            try:
                classification = classify_triton_exception(exc)
            except Exception:
                classification = None
            _write_result_file(
                result_path,
                {
                    "status": "error",
                    "traceback": traceback.format_exc(),
                    "decorator": decorator,
                    "exc_type": type(exc).__name__,
                    "exc_module": type(exc).__module__,
                    "exc_args": exc_args,
                    "classification": classification,
                    "captured_output": _cap[0] or None,
                },
            )
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
    failure_reason: Literal["ok", "error", "timeout"] | None = None

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
        self.process.start()

    @staticmethod
    def skip(ctx: PrecompileContext, config: Config, ok: bool) -> PrecompileFuture:
        """Dummy precompile future that is already done."""
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
        construction.  Returns a ``skip`` future when the kernel is already
        compiled (fork mode only).
        """
        mode = ctx.settings.autotune_precompile
        decorator = ctx.kernel.format_kernel_decorator(config, ctx.settings)

        if mode == "spawn":
            mp_ctx = mp.get_context("spawn")
            assert args_path is not None
            try:
                fn_spec = _serialize_compiled_fn(fn)
            except RuntimeError as err:
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
            precompiler = _prepare_precompiler_for_fork(
                fn, args, config, ctx.kernel, decorator, ctx.log
            )
            if precompiler is None:
                return PrecompileFuture.skip(ctx, config, True)
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
        if process.is_alive():
            with contextlib.suppress(Exception):
                process.kill()

    def cancel(self) -> None:
        """Terminate the underlying process (if any) without waiting for success."""
        self.end_time = time.time()
        process = self.process
        if process is not None:
            if self.started:
                with contextlib.suppress(Exception):
                    if process.is_alive():
                        process.kill()
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
            self._consume_result(raise_on_raise=False)
            if self.ok:
                self.failure_reason = "ok"
            elif self.failure_reason is None:
                self.failure_reason = "error"
            return self.ok
        process.terminate()
        process.join(10)
        msg = f"Timeout after {self.elapsed:.0f}s compiling {self.config}"
        if process.is_alive():
            if not self.ctx.settings.autotune_ignore_errors:
                self.ctx.log.warning(
                    msg,
                    "(SIGKILL required)",
                )
            process.kill()
            process.join()
        else:
            if not self.ctx.settings.autotune_ignore_errors:
                self.ctx.log.warning(msg)

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
                        exc_args=(str(err),),
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
                        exc_args=("No result received from subprocess.",),
                        traceback=None,
                        classification="debug",
                    )
            elif message_data["status"] == "ok":
                if self.ok is None:
                    self.ok = True
                assert self.remote_error is None
            else:
                exc_args_obj = message_data["exc_args"]
                if isinstance(exc_args_obj, tuple):
                    exc_args_tuple: tuple[object, ...] = exc_args_obj
                else:
                    exc_args_tuple = tuple(cast("Iterable[object]", exc_args_obj))
                self.remote_error = RemoteError(
                    exc_type=cast("str", message_data["exc_type"]),
                    exc_module=cast("str | None", message_data["exc_module"]),
                    exc_args=exc_args_tuple,
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
        if classification == "raise":
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
        formatted = format_triton_compile_failure(self.config, exc_obj, self.ctx.kernel)
        if error.traceback:
            formatted = (
                f"{formatted}\nRemote traceback (spawned process):\n{error.traceback}"
            )
        if classification == "warn":
            self.ctx.log.warning(formatted)
            self.ctx.kernel.maybe_log_repro(
                self.ctx.log.warning, self.ctx.args, self.config
            )
        elif not ignore_errors:
            self.ctx.log.debug(formatted)
            self.ctx.kernel.maybe_log_repro(
                self.ctx.log.debug, self.ctx.args, self.config
            )
        self._remote_error_handled = True
