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
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Generator
from typing import TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])


def _is_enabled() -> bool:
    """Check if compile time measurement is enabled via HELION_MEASURE_COMPILE_TIME=1."""
    return os.environ.get("HELION_MEASURE_COMPILE_TIME", "0") == "1"


class CompileTimeTracker:
    """
    Thread-safe tracker for compilation time measurements.

    When HELION_MEASURE_COMPILE_TIME=1 is set, this tracks time spent in various
    compilation phases and prints a summary table at program exit.
    """

    _instance: CompileTimeTracker | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._timings: defaultdict[str, float] = defaultdict(float)
        self._call_counts: defaultdict[str, int] = defaultdict(int)
        self._active_timers: dict[int, list[tuple[str, float]]] = {}
        self._timer_lock = threading.Lock()
        self._printed = False

    @classmethod
    def instance(cls) -> CompileTimeTracker:
        """Get the singleton instance of CompileTimeTracker."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = CompileTimeTracker()
                    if _is_enabled():
                        atexit.register(cls._instance.print_report)
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
    }

    # Top-level phases (not nested in anything else)
    _TOP_LEVEL: ClassVar[list[str]] = [
        "Kernel.bind",
        "BoundKernel.set_config",
        "BoundKernel.autotune",
        "BoundKernel.kernel_call",
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
