from __future__ import annotations

import atexit
from collections import defaultdict
import contextlib
import functools
import operator
import os
import sys
import threading
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Generator
from typing import TypeVar

if TYPE_CHECKING:
    from .autotuner.metrics import AutotuneMetrics

_F = TypeVar("_F", bound=Callable[..., Any])


def _is_enabled() -> bool:
    """Check if compile time measurement is enabled via HELION_MEASURE_COMPILE_TIME=1 or HELION_PERF_LOG."""
    return (
        os.environ.get("HELION_MEASURE_COMPILE_TIME", "0") == "1"
        or os.environ.get("HELION_PERF_LOG", "0").strip().lower() != "0"
    )


def _perf_log_mode() -> str:
    """Return the HELION_PERF_LOG mode: '0', 'summary', or other."""
    return os.environ.get("HELION_PERF_LOG", "0").strip().lower()


class CompileTimeTracker:
    """
    Thread-safe tracker for compilation time measurements.

    When HELION_MEASURE_COMPILE_TIME=1 is set, this tracks time spent in various
    compilation phases and prints a summary table at program exit.

    When HELION_PERF_LOG=summary is set, it also prints a [HELION_PERF] summary
    in the same format as Triton's [TRITON_PERF] for easy side-by-side comparison.
    """

    _instance: CompileTimeTracker | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._timings: defaultdict[str, float] = defaultdict(float)
        self._call_counts: defaultdict[str, int] = defaultdict(int)
        self._active_timers: dict[int, list[tuple[str, float]]] = {}
        self._timer_lock = threading.Lock()
        self._printed = False
        # Wall-clock tracking for perf log mode
        self._create_time = time.perf_counter()
        self._first_call_time: float | None = None
        self._last_call_time: float = 0.0
        self._autotune_records: list[AutotuneMetrics] = []
        self._process_start: float | None = None
        try:
            clock_ticks = os.sysconf("SC_CLK_TCK")
            with open("/proc/self/stat") as f:
                fields = f.read().split(")")[-1].split()
                starttime_ticks = int(fields[19])
            with open("/proc/stat") as f:
                for line in f:
                    if line.startswith("btime "):
                        btime = int(line.split()[1])
                        break
            process_start_epoch = btime + starttime_ticks / clock_ticks
            epoch_offset = time.time() - time.perf_counter()
            self._process_start = process_start_epoch - epoch_offset
        except Exception:
            self._process_start = None

    @classmethod
    def instance(cls) -> CompileTimeTracker:
        """Get the singleton instance of CompileTimeTracker."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = CompileTimeTracker()
                    if _is_enabled():
                        if os.environ.get("HELION_MEASURE_COMPILE_TIME", "0") == "1":
                            atexit.register(cls._instance.print_report)
                        if _perf_log_mode() == "summary":
                            atexit.register(cls._instance.print_perf_summary)
                        # Register hook to capture autotuning metrics
                        from .autotuner.metrics import register_post_autotune_hook

                        register_post_autotune_hook(cls._instance._on_autotune_done)
        return cls._instance

    def start(self, name: str) -> None:
        """Start timing a named section."""
        if not _is_enabled():
            return
        tid = threading.get_ident()
        with self._timer_lock:
            if tid not in self._active_timers:
                self._active_timers[tid] = []
            self._active_timers[tid].append((name, time.perf_counter()))

    def stop(self, name: str) -> float:
        """Stop timing a named section and return elapsed time."""
        if not _is_enabled():
            return 0.0
        end_time = time.perf_counter()
        tid = threading.get_ident()
        with self._timer_lock:
            if tid not in self._active_timers or not self._active_timers[tid]:
                return 0.0
            started_name, start_time = self._active_timers[tid].pop()
            if started_name != name:
                # Mismatched start/stop - put it back and warn
                self._active_timers[tid].append((started_name, start_time))
                return 0.0
            elapsed = end_time - start_time
            self._timings[name] += elapsed
            self._call_counts[name] += 1
            # Track wall-clock boundaries for top-level calls
            if self._first_call_time is None:
                self._first_call_time = start_time
            self._last_call_time = end_time
            return elapsed

    @contextlib.contextmanager
    def measure(self, name: str) -> Generator[None, None, None]:
        """Context manager to measure time for a named section."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def record(self, name: str, elapsed: float) -> None:
        """Directly record a timing measurement."""
        if not _is_enabled():
            return
        with self._timer_lock:
            self._timings[name] += elapsed
            self._call_counts[name] += 1

    # Define the hierarchy of timings (parent -> children)
    _HIERARCHY: ClassVar[dict[str, list[str]]] = {
        "Kernel.bind": ["BoundKernel.create_host_function"],
        "BoundKernel.create_host_function": [
            "HostFunction.parse_ast",
            "HostFunction.unroll_static_loops",
            "HostFunction.propagate_types",
            "HostFunction.finalize_config_spec",
            "HostFunction.lower_to_device_ir",
        ],
        "BoundKernel.set_config": [
            "BoundKernel.to_triton_code",
            "BoundKernel.PyCodeCache.load",
        ],
        "BoundKernel.to_triton_code": [
            "BoundKernel.generate_ast",
            "BoundKernel.unparse",
        ],
        "BoundKernel.autotune": [
            "BoundKernel.to_triton_code",
            "BoundKernel.PyCodeCache.load",
        ],
        "default_launcher": [
            "static_launch.first_call",
            "static_launch.hot_path",
            "triton_standard_launch",
        ],
    }

    # Top-level phases (not nested in anything else)
    _TOP_LEVEL: ClassVar[list[str]] = [
        "Kernel.bind",
        "BoundKernel.set_config",
        "BoundKernel.autotune",
        "BoundKernel.kernel_call",
        "default_launcher",
    ]

    # Hierarchy for perf log output (name, indent level)
    _PERF_HIERARCHY: ClassVar[list[tuple[str, int]]] = [
        ("BoundKernel.__call__", 0),
        ("Kernel.bind", 1),
        ("BoundKernel.create_host_function", 2),
        ("HostFunction.parse_ast", 3),
        ("HostFunction.propagate_types", 3),
        ("HostFunction.lower_to_device_ir", 3),
        ("BoundKernel.set_config", 1),
        ("BoundKernel.compile_config", 2),
        ("BoundKernel.to_triton_code", 3),
        ("BoundKernel.generate_ast", 4),
        ("BoundKernel.unparse", 4),
        ("BoundKernel.PyCodeCache.load", 3),
        ("BoundKernel.autotune", 1),
        ("default_launcher", 1),
        ("static_launch.first_call", 2),
        ("static_launch.hot_path", 2),
        ("triton_standard_launch", 2),
    ]

    def print_report(self) -> None:
        """Print a formatted timing report to stderr."""
        if self._printed or not self._timings:
            return
        self._printed = True

        # Calculate top-level total (what user actually perceives)
        top_level_total = sum(self._timings.get(name, 0.0) for name in self._TOP_LEVEL)

        if top_level_total == 0:
            # Fallback if no top-level timings
            top_level_total = sum(self._timings.values())

        # Print header
        print("\n", file=sys.stderr)
        print("=" * 85, file=sys.stderr)
        print("HELION COMPILE TIME BREAKDOWN", file=sys.stderr)
        print("=" * 85, file=sys.stderr)
        print(
            f"{'Section':<50}  {'Time':>10}  {'%':>6}  {'Calls':>6}",
            file=sys.stderr,
        )
        print("-" * 85, file=sys.stderr)

        # Print hierarchically
        printed = set()
        for top_name in self._TOP_LEVEL:
            if top_name in self._timings:
                self._print_section(top_name, 0, top_level_total, printed)

        # Print any remaining sections not in hierarchy
        remaining = sorted(
            [(name, t) for name, t in self._timings.items() if name not in printed],
            key=operator.itemgetter(1),
            reverse=True,
        )
        if remaining:
            print("-" * 85, file=sys.stderr)
            print("(other sections)", file=sys.stderr)
            for name, elapsed in remaining:
                self._print_line(name, elapsed, top_level_total, 0)

        print("-" * 85, file=sys.stderr)
        print(
            f"{'WALL CLOCK TOTAL':<50}  {top_level_total:>9.3f}s  {100.0:>5.1f}%",
            file=sys.stderr,
        )
        print("=" * 85, file=sys.stderr)
        print(file=sys.stderr)

    def _print_section(
        self, name: str, indent: int, total: float, printed: set[str]
    ) -> None:
        """Print a section and its children recursively."""
        if name in printed or name not in self._timings:
            return
        printed.add(name)

        elapsed = self._timings[name]
        self._print_line(name, elapsed, total, indent)

        # Print children
        children = self._HIERARCHY.get(name, [])
        for child in children:
            self._print_section(child, indent + 1, total, printed)

    def _print_line(self, name: str, elapsed: float, total: float, indent: int) -> None:
        """Print a single timing line with indentation."""
        pct = (elapsed / total * 100) if total > 0 else 0
        calls = self._call_counts[name]
        prefix = "  " * indent + ("└─ " if indent > 0 else "")
        display_name = prefix + name
        print(
            f"{display_name:<50}  {elapsed:>9.3f}s  {pct:>5.1f}%  {calls:>6}",
            file=sys.stderr,
        )

    def print_perf_summary(self) -> None:
        """Print a [HELION_PERF] summary matching Triton's [TRITON_PERF] format."""
        if not self._timings:
            return

        P = "[HELION_PERF]"
        known = {name for name, _ in self._PERF_HIERARCHY}

        print(f"\n{P} === Process Summary ===", file=sys.stderr)
        for cat, indent in self._PERF_HIERARCHY:
            total_time = self._timings.get(cat)
            if total_time is None:
                continue
            n = self._call_counts[cat]
            avg = total_time / n if n else 0
            prefix = "  " * indent
            label = f"{prefix}{cat}"
            print(
                f"{P} {label:42s}: {total_time * 1000:10.1f}ms total, {n:5d} calls, avg {avg * 1000:.1f}ms",
                file=sys.stderr,
            )

        # Print any categories not in the known hierarchy
        for cat in sorted(self._timings.keys()):
            if cat in known:
                continue
            total_time = self._timings[cat]
            n = self._call_counts[cat]
            avg = total_time / n if n else 0
            print(
                f"{P} {cat:42s}: {total_time * 1000:10.1f}ms total, {n:5d} calls, avg {avg * 1000:.1f}ms",
                file=sys.stderr,
            )

        # Wall-clock accounting
        summary_time = time.perf_counter()
        top_level = sum(self._timings.get(name, 0.0) for name in ("BoundKernel.__call__",))
        if self._first_call_time is not None:
            before_first = self._first_call_time - self._create_time
        else:
            before_first = 0.0
        after_last = summary_time - self._last_call_time if self._last_call_time else 0.0
        wall_since_import = summary_time - self._create_time
        between_calls = wall_since_import - before_first - top_level - after_last

        print(f"{P} ---", file=sys.stderr)
        if self._process_start is not None:
            proc_to_import = self._create_time - self._process_start
            total_wall = summary_time - self._process_start
            print(
                f"{P} Process total wall time                  : {total_wall * 1000:10.1f}ms",
                file=sys.stderr,
            )
            print(
                f"{P}   python start → helion import           : {proc_to_import * 1000:10.1f}ms",
                file=sys.stderr,
            )
            print(
                f"{P}   helion import → first kernel call      : {before_first * 1000:10.1f}ms",
                file=sys.stderr,
            )
        print(
            f"{P}   BoundKernel.__call__ (top-level)       : {top_level * 1000:10.1f}ms",
            file=sys.stderr,
        )
        print(
            f"{P}   between kernel calls (host/test work)  : {between_calls * 1000:10.1f}ms",
            file=sys.stderr,
        )
        print(
            f"{P}   after last kernel call                 : {after_last * 1000:10.1f}ms",
            file=sys.stderr,
        )
        # Autotuning summary
        if self._autotune_records:
            total_at_time = sum(r.autotune_time for r in self._autotune_records)
            total_configs = sum(r.num_configs_tested for r in self._autotune_records)
            total_compile_fail = sum(r.num_compile_failures for r in self._autotune_records)
            total_accuracy_fail = sum(r.num_accuracy_failures for r in self._autotune_records)
            n_at = len(self._autotune_records)
            print(f"{P} --- Autotuning ---", file=sys.stderr)
            print(
                f"{P}   autotune calls                       : {n_at:6d}",
                file=sys.stderr,
            )
            print(
                f"{P}   total autotune time                  : {total_at_time * 1000:10.1f}ms",
                file=sys.stderr,
            )
            print(
                f"{P}   configs tested                       : {total_configs:6d}",
                file=sys.stderr,
            )
            print(
                f"{P}   compile failures                     : {total_compile_fail:6d}",
                file=sys.stderr,
            )
            print(
                f"{P}   accuracy failures                    : {total_accuracy_fail:6d}",
                file=sys.stderr,
            )
            for i, r in enumerate(self._autotune_records):
                print(
                    f"{P}   [{i}] {r.kernel_name:30s}: {r.autotune_time * 1000:8.1f}ms, "
                    f"{r.num_configs_tested} configs, best={r.best_perf_ms:.3f}ms",
                    file=sys.stderr,
                )
        print(f"{P} === End Summary ===", file=sys.stderr, flush=True)

    def _on_autotune_done(self, metrics: AutotuneMetrics) -> None:
        """Hook called after each autotuning run to capture metrics."""
        self._autotune_records.append(metrics)

    def reset(self) -> None:
        """Reset all timing data."""
        with self._timer_lock:
            self._timings.clear()
            self._call_counts.clear()
            self._active_timers.clear()
            self._printed = False


def get_tracker() -> CompileTimeTracker:
    """Get the global CompileTimeTracker instance."""
    return CompileTimeTracker.instance()


@contextlib.contextmanager
def measure(name: str) -> Generator[None, None, None]:
    """
    Context manager to measure compilation time for a named section.

    Usage:
        with measure("phase_name"):
            # code to measure

    Only active when HELION_MEASURE_COMPILE_TIME=1 is set.
    """
    tracker = get_tracker()
    tracker.start(name)
    try:
        yield
    finally:
        tracker.stop(name)


def timed(name: str | None = None) -> Callable[[_F], _F]:
    """
    Decorator to measure compilation time for a function.

    Usage:
        @timed("my_function")
        def my_function(...):
            ...

        # Or use function name automatically:
        @timed()
        def my_function(...):
            ...

    Only active when HELION_MEASURE_COMPILE_TIME=1 is set.
    """

    def decorator(fn: _F) -> _F:
        section_name = name if name is not None else fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> object:
            if not _is_enabled():
                return fn(*args, **kwargs)
            tracker = get_tracker()
            tracker.start(section_name)
            try:
                return fn(*args, **kwargs)
            finally:
                tracker.stop(section_name)

        return wrapper  # type: ignore[return-value]

    return decorator


def record(name: str, elapsed: float) -> None:
    """
    Record a timing measurement directly.

    Usage:
        start = time.perf_counter()
        # ... do work ...
        record("work", time.perf_counter() - start)

    Only active when HELION_MEASURE_COMPILE_TIME=1 is set.
    """
    get_tracker().record(name, elapsed)


def print_report() -> None:
    """Manually print the timing report."""
    get_tracker().print_report()


def reset() -> None:
    """Reset all timing data."""
    get_tracker().reset()
