from __future__ import annotations

import contextlib
import csv
import hashlib
import itertools
import logging
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import traceback
from types import TracebackType
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterator
from typing import Literal
from typing import NamedTuple
from typing import TypeAlias
from typing import TypeVar
from typing_extensions import Self

from torch._inductor.runtime.triton_compat import OutOfResources
from torch._inductor.runtime.triton_compat import PTXASError

if TYPE_CHECKING:
    from _csv import _writer as CsvWriter
    import io

    from ..runtime.config import Config
    from ..runtime.settings import Settings
    from .base_search import _AutotunableKernel

else:
    CsvWriter = Any  # type: ignore[assignment]

SinkSelf = TypeVar("SinkSelf", bound="AutotuneLogSink")
ExcInfoParam: TypeAlias = (
    bool
    | BaseException
    | tuple[type[BaseException], BaseException, TracebackType | None]
    | None
)


class _ElapsedFormatter(logging.Formatter):
    def __init__(self, elapsed_fn: Callable[[], int]) -> None:
        super().__init__()
        self._elapsed_fn = elapsed_fn

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        elapsed = self._elapsed_fn()
        return f"[{elapsed}s] {record.getMessage()}"


class AutotuningLogger:
    """
    A self-contained logger that does not propagate to the root logger and
    prints each record to stderr in the form:

        [<elapsed>s] <message>

    where *elapsed* is the whole-second wall-clock time since the logger
    instance was created.

    Takes lambdas as arguments, which are called when the log is emitted.
    """

    _count: itertools.count[int] = itertools.count()

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        level = settings.autotune_log_level
        self.level = level
        self._logger: logging.Logger = logging.getLogger(
            f"{__name__}.{next(self._count)}"
        )
        self._logger.setLevel(level)
        self._logger.propagate = False
        self._start_time: float = time.perf_counter()
        self._extra_handlers: list[logging.Handler] = []
        self._active_handlers: list[logging.Handler] = []
        self._log_sink: AutotuneLogSink | None = None
        self.reset()

    def reset(self) -> None:
        self._start_time = time.perf_counter()
        for handler in list(self._active_handlers):
            self._logger.removeHandler(handler)
        self._active_handlers = []
        self._register_handler(self._make_stream_handler())
        for handler in self._extra_handlers:
            self._register_handler(handler)

    def add_handler(self, handler: logging.Handler) -> None:
        if handler in self._extra_handlers:
            return
        self._extra_handlers.append(handler)
        self._register_handler(handler)

    def remove_handler(self, handler: logging.Handler) -> None:
        if handler in self._extra_handlers:
            self._extra_handlers.remove(handler)
        if handler in self._active_handlers:
            self._active_handlers.remove(handler)
            self._logger.removeHandler(handler)

    @contextlib.contextmanager
    def autotune_logging(
        self, base_path: str | None = None
    ) -> Iterator[AutotuneLogSink | None]:
        """Attach an :class:`AutotuneLogSink` for the duration of a tuning run."""

        path = base_path or self._settings.autotune_log
        if not path:
            yield None
            return
        with AutotuneLogSink(path) as sink:
            self._attach_sink(sink)
            sink.start_run()
            try:
                yield sink
            finally:
                sink.end_run()
                self._detach_sink()

    def record_autotune_entry(self, entry: AutotuneLogEntry) -> None:
        """Write a structured autotune log entry when a sink is active."""

        if self._log_sink is None:
            return
        self._log_sink.record(entry)

    def _attach_sink(self, sink: AutotuneLogSink) -> None:
        self._log_sink = sink
        self.add_handler(sink.handler)

    def _detach_sink(self) -> None:
        sink = self._log_sink
        if sink is None:
            return
        self.remove_handler(sink.handler)
        self._log_sink = None

    def _elapsed_seconds(self) -> int:
        return int(time.perf_counter() - self._start_time)

    def _configure_handler(self, handler: logging.Handler) -> None:
        handler.setFormatter(_ElapsedFormatter(self._elapsed_seconds))

    def _register_handler(self, handler: logging.Handler) -> None:
        self._configure_handler(handler)
        self._logger.addHandler(handler)
        self._active_handlers.append(handler)

    def _make_stream_handler(self) -> logging.Handler:
        return logging.StreamHandler(sys.stderr)

    def __call__(
        self,
        *msg: str | Callable[[], str],
        level: int = logging.INFO,
        exc_info: ExcInfoParam = None,
        stacklevel: int | None = None,
    ) -> None:
        """
        Log a message at a specified log level.

        Args:
            msg: The message(s) to log. Can be strings or callables that return strings.
            level: The log level for the message.
            exc_info: Optional exception info forwarded to ``logging.Logger``.
            stacklevel: Optional stack level forwarded to ``logging.Logger``.
        """
        if level >= self.level:
            message = " ".join(map(_maybe_call, msg))
            if stacklevel is not None:
                if exc_info is not None:
                    self._logger.log(
                        level,
                        message,
                        exc_info=exc_info,
                        stacklevel=stacklevel,
                    )
                else:
                    self._logger.log(
                        level,
                        message,
                        stacklevel=stacklevel,
                    )
            else:
                if exc_info is not None:
                    self._logger.log(level, message, exc_info=exc_info)
                else:
                    self._logger.log(level, message)

    def error(
        self,
        *msg: str | Callable[[], str],
        exc_info: ExcInfoParam = None,
        stacklevel: int | None = None,
    ) -> None:
        return self(
            *msg,
            level=logging.ERROR,
            exc_info=exc_info,
            stacklevel=stacklevel,
        )

    def warning(
        self,
        *msg: str | Callable[[], str],
        exc_info: ExcInfoParam = None,
        stacklevel: int | None = None,
    ) -> None:
        return self(
            *msg,
            level=logging.WARNING,
            exc_info=exc_info,
            stacklevel=stacklevel,
        )

    def debug(
        self,
        *msg: str | Callable[[], str],
        exc_info: ExcInfoParam = None,
        stacklevel: int | None = None,
    ) -> None:
        return self(
            *msg,
            level=logging.DEBUG,
            exc_info=exc_info,
            stacklevel=stacklevel,
        )


def _maybe_call(fn: Callable[[], str] | str) -> str:
    """
    Call a callable or return the string directly.

    Args:
        fn: A callable that returns a string or a string.

    Returns:
        The resulting string.
    """
    if callable(fn):
        return fn()
    return fn


class AutotuneLogEntry(NamedTuple):
    generation: int
    status: str
    perf_ms: float | None
    compile_time: float | None
    config: Config


class AutotuneLogSink:
    """
    Writes autotune results to CSV and connects autotune logs to a file handler.
    """

    def __init__(self, base_path: str) -> None:
        self._base_path = Path(base_path)
        self.csv_path = self._base_path.with_suffix(".csv")
        self.log_path = self._base_path.with_suffix(".log")
        self._csv_file: io.TextIOWrapper | None = None
        self._csv_writer: CsvWriter | None = None
        self._log_handler: logging.FileHandler | None = None
        self._run_start_time: float | None = None
        self._config_counter: int = 0

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def handler(self) -> logging.Handler:
        assert self._log_handler is not None, "Log sink not opened"
        return self._log_handler

    def open(self) -> None:
        if self._csv_writer is not None:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = self.csv_path.open("w", encoding="utf-8", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            [
                "timestamp_s",
                "config_index",
                "generation",
                "status",
                "perf_ms",
                "compile_time_s",
                "config",
            ]
        )
        self._csv_file.flush()
        handler = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        self._log_handler = handler

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
        self._csv_file = None
        self._csv_writer = None
        if self._log_handler is not None:
            self._log_handler.flush()
            self._log_handler.close()
        self._log_handler = None
        self._run_start_time = None
        self._config_counter = 0

    def start_run(self) -> None:
        self._run_start_time = time.perf_counter()
        self._config_counter = 0

    def end_run(self) -> None:
        self._run_start_time = None
        self._config_counter = 0

    def record(self, entry: AutotuneLogEntry) -> None:
        if self._csv_writer is None:
            return
        self._config_counter += 1
        timestamp_field = ""
        if self._run_start_time is not None:
            timestamp = time.perf_counter() - self._run_start_time
            timestamp_field = f"{timestamp:.2f}"
        perf_field = ""
        if entry.perf_ms is not None and math.isfinite(entry.perf_ms):
            perf_field = f"{entry.perf_ms:.6f}"
        compile_field = ""
        if entry.compile_time is not None:
            compile_field = f"{entry.compile_time:.2f}"
        self._csv_writer.writerow(
            [
                timestamp_field,
                self._config_counter,
                entry.generation,
                entry.status,
                perf_field,
                compile_field,
                str(entry.config),
            ]
        )
        if self._csv_file is not None:
            self._csv_file.flush()


SUPPRESSED_TRITON_CODE_MSG = (
    "Enable HELION_AUTOTUNE_LOG_LEVEL=DEBUG to log generated Triton code."
)


def log_generated_triton_code_debug(
    logger: AutotuningLogger,
    kernel: _AutotunableKernel,
    config: Config,
    *,
    prefix: str | None = None,
) -> None:
    """
    Emit the generated Triton code at debug level if the logger allows it.

    Args:
        logger: Logger that should receive the message.
        kernel: Kernel whose Triton code should be logged.
        config: Config used to generate the Triton code.
        prefix: Optional prefix for the log message.
    """
    logger.debug(
        lambda: _format_triton_code(kernel, config, prefix or "Generated Triton code:")
    )


def _format_triton_code(kernel: _AutotunableKernel, config: Config, prefix: str) -> str:
    code = kernel.to_triton_code(config)
    if code is None:
        return f"{prefix}\n<no source code available>"
    return f"{prefix}\n{code}"


_FAILURE_DUMP_ENV = "HELION_AUTOTUNER_TRITON_FAILURE_DUMP"


def _get_failure_dump_dir() -> str | None:
    value = os.environ.get(_FAILURE_DUMP_ENV)
    if value is None or (value := value.strip()) == "":
        return None
    return value


def maybe_dump_triton_failure(
    bound_kernel: _AutotunableKernel,
    config: Config,
    err: BaseException,
    *,
    remote_traceback: str | None = None,
    captured_output: str | None = None,
) -> None:
    """Dump a Triton failure report if HELION_AUTOTUNER_TRITON_FAILURE_DUMP is set."""
    dump_dir = _get_failure_dump_dir()
    if not dump_dir:
        return
    try:
        triton_code = bound_kernel.to_triton_code(config) or "# (no Triton code)"
    except Exception:
        triton_code = "# (failed to regenerate Triton code)"
    dump_triton_failure(
        dump_dir,
        triton_code,
        err,
        remote_traceback=remote_traceback,
        captured_output=captured_output,
    )


def format_triton_compile_failure(
    config: Config, err: BaseException, kernel: _AutotunableKernel
) -> str:
    return (
        "Triton compile failed. This likely indicates a bug in Triton. "
        "Skipping failing config.\n"
        f"Config: {kernel.format_kernel_decorator(config, kernel.settings)}\n"
        f"Error: {type(err).__name__}: {err}\n"
        f"{SUPPRESSED_TRITON_CODE_MSG}"
    )


# Common logic to decide how to surface Triton errors
_EXPECTED_TRITON_ERRORS_RE: re.Pattern[str] = re.compile(
    "|".join(
        map(
            re.escape,
            [
                "[CUDA]: invalid argument",  # CUDA Error
                "PassManager::run failed",  # Triton Error
                "TServiceRouterException",  # Remote compile failed
                "triton.compiler.errors.CompilationError",  # Triton CompilationError
                "out of resource: shared memory",  # Triton shared memory OOM
                "ZE_RESULT_ERROR_INVALID_KERNEL_NAME",  # Level Zero compile failed
                "exceeds triton maximum tensor numel",  # needs smaller config
                "Resource temporarily unavailable",  # LLVM Error
                "too many blocks in cooperative launch",  # CUDA cooperative launch limit
                "too many resources requested for launch",  # Triton resouce error
            ],
        )
    )
)

_UNRECOVERABLE_RUNTIME_ERROR_RE: re.Pattern[str] = re.compile(
    "|".join(
        map(
            re.escape,
            [
                "illegal memory access",
                "misaligned address",
                "unspecified launch failure",
                "illegal instruction",
            ],
        )
    ),
    re.IGNORECASE,
)


def match_unrecoverable_runtime_error(err: BaseException) -> bool:
    return bool(_UNRECOVERABLE_RUNTIME_ERROR_RE.search(str(err)))


def classify_triton_exception(err: BaseException) -> Literal["raise", "warn", "debug"]:
    """
    Classify a Triton compile/runtime exception during autotuning.

    Returns one of:
      - "raise": unexpected error, caller should raise
      - "warn": notable expected error (e.g., PassManager pipeline failure)
      - "debug": benign/expected error; caller can log at debug level
    """
    # Known exception types first
    if isinstance(err, OutOfResources):
        return "debug"
    # Different PTXASError classes may be raised from different modules; match by name as well
    if isinstance(err, PTXASError) or err.__class__.__name__ == "PTXASError":
        return "warn"

    msg = str(err)
    if "PassManager::run failed" in msg:
        return "warn"
    if _EXPECTED_TRITON_ERRORS_RE.search(msg) or match_unrecoverable_runtime_error(err):
        return "debug"
    return "raise"


# Regex to strip filesystem paths and timestamps from dump content before hashing,
# so that equivalent failures on different machines or at different times deduplicate.
# Matches absolute paths with at least two components (e.g. /tmp/foo, /home/user/bar.py).
_PATH_RE = re.compile(r"(?:/[\w][\w.-]*){2,}")
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[^\s]*")


def _stable_hash(content: str) -> str:
    """Hash *content* after stripping filesystem paths and timestamps."""
    normalized = _PATH_RE.sub("PATH", content)
    normalized = _TIMESTAMP_RE.sub("TIME", normalized)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


@contextlib.contextmanager
def capture_output() -> Iterator[list[str]]:
    """
    Context manager that captures stdout and stderr written during its scope.

    Yields a single-element list; after the block exits ``result[0]`` contains
    the captured text.  The original file descriptors are restored on exit even
    if an exception is raised.

    Captures at the file-descriptor level (via a temporary file) so that C/C++
    output from Triton's LLVM/MLIR compiler passes is captured in addition to
    Python-level ``print()`` calls.  A temporary file is used instead of a pipe
    to avoid deadlocks when the captured output exceeds the pipe buffer size.
    """
    result: list[str] = [""]
    sys.stdout.flush()
    sys.stderr.flush()

    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    tmp_fd, tmp_path = tempfile.mkstemp()
    try:
        os.dup2(tmp_fd, stdout_fd)
        os.dup2(tmp_fd, stderr_fd)
        os.close(tmp_fd)
        yield result
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        try:
            result[0] = Path(tmp_path).read_text(encoding="utf-8", errors="replace")
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


def dump_triton_failure(
    dump_dir: str,
    triton_code: str,
    err: BaseException,
    *,
    remote_traceback: str | None = None,
    captured_output: str | None = None,
) -> str | None:
    """
    Write a self-contained Triton failure report to *dump_dir*.

    The file contains the full Triton source code, the Triton version, and the
    error message / traceback.  The filename is derived from a content hash
    (with filesystem paths and timestamps stripped) so that identical failures
    naturally deduplicate across runs.

    Returns the path of the written file, or ``None`` if writing failed.
    """
    import triton

    triton_version = getattr(triton, "__version__", "unknown")

    error_type = type(err).__qualname__
    error_msg = str(err)
    if remote_traceback:
        tb_text = remote_traceback
    else:
        tb_text = "".join(traceback.format_exception(type(err), err, err.__traceback__))

    parts: list[str] = [
        (
            f"# Triton failure report\n"
            f"# triton version: {triton_version}\n"
            f"# error: {error_type}: {error_msg}\n"
            f"\n"
            f"# --- Triton source code ---\n"
            f"{triton_code}\n"
            f"\n"
            f"# --- Error ---\n"
            f"# {error_type}: {error_msg}\n"
            f"#\n"
            f"# --- Traceback ---\n"
        ),
    ]
    parts.extend(f"# {line}\n" for line in tb_text.splitlines())

    if captured_output:
        parts.append("#\n# --- Captured stdout/stderr ---\n")
        parts.extend(f"# {line}\n" for line in captured_output.splitlines())

    report = "".join(parts)
    file_hash = _stable_hash(report)
    filename = f"{file_hash}.py"
    dump_path = Path(dump_dir)
    try:
        dump_path.mkdir(parents=True, exist_ok=True)
        dest = dump_path / filename
        if not dest.exists():
            dest.write_text(report, encoding="utf-8")
        return str(dest)
    except Exception:
        return None
