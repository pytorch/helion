"""Render autotune candidates in forked worker processes.

CuTe code generation is pure Python that takes seconds per config (15-22 s
for a 128K-row flash kernel), has no Triton-style precompile entry point and
needs only the bound kernel and the config; the CuTe DSL JIT happens later,
at the first call in the parent. A batch of candidates therefore renders in
children forked one per config, up to ``jobs`` ahead of the autotuner, which
loads, JIT-compiles and times the earlier configs meanwhile.

A child forked after the parent initialised CUDA must not touch the driver
(torch refuses with "Cannot re-initialize CUDA in forked subprocess"). Code
generation reads a few device facts through ``torch.cuda``: the current
device index, device properties, capability and name (the MMA-support probe,
shared-memory budgets, SM counts). The parent captures them for the kernel's
device before forking (:meth:`CudaDeviceFacts.capture`) and the child serves
them from that snapshot (:meth:`CudaDeviceFacts.install`). Any other CUDA
call in a child still fails loudly; the parent then renders that config
itself, so a new CUDA dependency in code generation degrades to the serial
path rather than to wrong code. Code generation is a pure function of the
bound kernel and the config, so a worker's source is byte-identical to the
parent's; the facts it records on the kernel are not (see
``BoundKernel.prime_rendered_code``). A worker renders from the parent's
state at fork time: with ``static_shapes=False`` the parent's shape-env
guards accumulate as the search renders, but they only widen the dispatch
key, never the generated source, and the config the search finally selects
is rendered again in the parent, after the search and outside its budget.
"""

from __future__ import annotations

import contextlib
import dataclasses
import multiprocessing as mp
import os
import pickle
import signal
import time
import traceback
from typing import TYPE_CHECKING
from typing import Callable
from typing import Literal

import torch

from .logger import capture_output
from .precompile_future import _write_result_file
from .process_utils import signal_process_tree
from .process_utils import start_isolated_process_group

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel
    from .logger import AutotuningLogger

# Without HELION_AUTOTUNE_PRECOMPILE_JOBS the pool uses this many workers at
# most: each forked child copies the parent pages it touches while rendering
# (about 1 GB on a flash kernel), and a batch rarely benefits from more.
DEFAULT_MAX_JOBS = 32

_DeviceLike = int | str | torch.device | None
RenderStatus = Literal[
    "pending", "ok", "error", "timeout", "unavailable", "cancelled", "cached"
]


@dataclasses.dataclass(frozen=True)
class CudaDeviceFacts:
    """The ``torch.cuda`` facts code generation reads, captured before a fork."""

    current: int
    properties: dict[int, object]
    capability: dict[int, tuple[int, int]]
    name: dict[int, str]

    @staticmethod
    def capture(device: torch.device) -> CudaDeviceFacts | None:
        """Snapshot the facts of ``device`` and the current device (None off CUDA)."""
        if device.type != "cuda" or not torch.cuda.is_available():
            return None
        current = torch.cuda.current_device()
        indices = sorted({current, current if device.index is None else device.index})
        return CudaDeviceFacts(
            current=current,
            properties={i: torch.cuda.get_device_properties(i) for i in indices},
            capability={i: torch.cuda.get_device_capability(i) for i in indices},
            name={i: torch.cuda.get_device_name(i) for i in indices},
        )

    def index(self, device: _DeviceLike) -> int:
        """The captured index ``device`` refers to (``None`` is the current one)."""
        if device is None:
            index = self.current
        elif isinstance(device, int):
            index = device
        else:
            if isinstance(device, str):
                device = torch.device(device)
            index = self.current if device.index is None else device.index
        if index not in self.properties:
            raise RuntimeError(
                f"CUDA device {index} was not captured before forking the render worker"
            )
        return index

    def install(self) -> None:
        """Serve the snapshot from ``torch.cuda`` in a forked render worker.

        Process-wide and never undone: the worker exits after rendering.
        """
        facts = self
        torch.cuda.current_device = lambda: facts.current  # pyrefly: ignore[bad-assignment]
        torch.cuda.get_device_properties = (  # pyrefly: ignore[bad-assignment]
            lambda device=None: facts.properties[facts.index(device)]
        )
        torch.cuda.get_device_capability = (  # pyrefly: ignore[bad-assignment]
            lambda device=None: facts.capability[facts.index(device)]
        )
        torch.cuda.get_device_name = (  # pyrefly: ignore[bad-assignment]
            lambda device=None: facts.name[facts.index(device)]
        )


def _render_in_child(
    kernel: BoundKernel,
    config: Config,
    facts: CudaDeviceFacts | None,
    result_path: str,
) -> None:
    """Fork target: render ``config`` and leave the source at ``result_path``."""
    start_isolated_process_group()
    status = 0
    captured: list[str] = [""]
    try:
        if facts is not None:
            facts.install()
        started = time.perf_counter()
        with capture_output() as captured:
            code = kernel.to_triton_code(
                config, emit_repro_caller=kernel.settings.print_output_code
            )
        _write_result_file(
            result_path,
            {"status": "ok", "code": code, "elapsed": time.perf_counter() - started},
        )
    except Exception as error:
        status = 1
        with contextlib.suppress(Exception):
            _write_result_file(
                result_path,
                {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "captured_output": captured[0] or None,
                },
            )
    finally:
        os._exit(status)


@dataclasses.dataclass
class RenderFuture:
    """One config's render in a forked worker."""

    config: Config
    result_path: str
    timeout: float
    process: mp.process.BaseProcess | None = None
    start_time: float | None = None
    status: RenderStatus = "pending"
    code: str | None = None
    # The worker's wall time for the render (code generation only).
    elapsed: float | None = None
    error: str | None = None

    def seconds_left(self) -> float:
        if self.start_time is None:
            return self.timeout
        return self.timeout - (time.time() - self.start_time)

    def running(self) -> bool:
        return (
            self.process is not None
            and self.status == "pending"
            and self.process.is_alive()
        )

    def wait(self) -> None:
        """Block until the render finished or timed out and record the outcome."""
        process = self.process
        if process is None or self.status != "pending":
            return
        process.join(max(0.0, self.seconds_left()))
        if process.is_alive():
            signal_process_tree(process, signal.SIGTERM)
            process.join(10)
            signal_process_tree(process, signal.SIGKILL)
            process.join()
            self.status = "timeout"
            self._remove_result()
            return
        message: dict[str, object]
        try:
            with open(self.result_path, "rb") as f:
                message = pickle.load(f)
        except FileNotFoundError:
            message = {
                "status": "error",
                "error": "the render worker exited without writing a result",
            }
        finally:
            self._remove_result()
        if message["status"] == "ok":
            self.status = "ok"
            self.code = str(message["code"])
            self.elapsed = float(message["elapsed"])  # pyrefly: ignore[bad-argument-type]
        else:
            self.status = "error"
            self.error = str(message.get("traceback") or message.get("error"))

    def cancel(self) -> None:
        """Kill a render still in flight (its source is no longer wanted)."""
        process = self.process
        if process is not None and self.status == "pending":
            if process.is_alive():
                signal_process_tree(process, signal.SIGKILL)
            process.join()
            self.status = "cancelled"
        self._remove_result()

    def _remove_result(self) -> None:
        with contextlib.suppress(OSError):
            os.remove(self.result_path)


class RenderAhead:
    """Render a batch's configs in forked workers, ``jobs`` ahead of a consumer.

    ``take(i)`` returns the i-th config's future, complete: it starts workers
    for the configs up to ``jobs`` positions ahead (never more than ``jobs``
    alive at once) and waits for the i-th. The consumer takes configs in
    order, as the provider's streaming loop does, so a budget cut wastes at
    most ``jobs`` renders; ``close`` kills whatever is still running and
    starts no more. Configs the kernel already compiled get no worker
    (``cached``). When a worker cannot be forked (an OS limit, a daemonic
    parent that may not have children, ...) the remaining futures turn
    ``unavailable``; whatever a future's status, the consumer still compiles
    the config in-process, loading the worker's source when there is one.
    ``take`` never raises for a worker problem.
    """

    def __init__(
        self,
        kernel: BoundKernel,
        configs: Sequence[Config],
        *,
        jobs: int,
        timeout: float,
        facts: CudaDeviceFacts | None,
        result_path: Callable[[], str],
        log: AutotuningLogger,
        already_compiled: Callable[[Config], bool] = lambda config: False,
    ) -> None:
        self.kernel = kernel
        self.jobs = max(1, jobs)
        self.facts = facts
        self.log = log
        self.futures = [
            RenderFuture(
                config=config,
                result_path=result_path(),
                timeout=timeout,
                status="cached" if already_compiled(config) else "pending",
            )
            for config in configs
        ]
        self._next = 0
        # ``unavailable``: a fork failed, no more workers for this kernel;
        # ``closed``: the consumer is done with this batch.
        self.unavailable = False
        self.closed = False

    def take(self, index: int) -> RenderFuture:
        self._start_ahead(index)
        future = self.futures[index]
        future.wait()
        return future

    def close(self) -> None:
        self.closed = True
        for future in self.futures:
            future.cancel()

    def _start_ahead(self, index: int) -> None:
        limit = min(len(self.futures), index + self.jobs)
        while self._next < limit and not (self.unavailable or self.closed):
            if sum(1 for future in self.futures[: self._next] if future.running()) >= (
                self.jobs
            ):
                return
            future = self.futures[self._next]
            if future.status == "pending":
                self._start(future)
            self._next += 1

    def _start(self, future: RenderFuture) -> None:
        try:
            process = mp.get_context("fork").Process(
                target=_render_in_child,
                args=(self.kernel, future.config, self.facts, future.result_path),
            )
            process.daemon = True
            process.start()
        except Exception as error:
            self.unavailable = True
            for pending in self.futures[self._next :]:
                if pending.status == "pending":
                    pending.status = "unavailable"
            self.log.warning(
                f"Could not fork a render worker ({type(error).__name__}: "
                f"{error}); rendering autotune candidates in-process."
            )
            return
        future.process = process
        future.start_time = time.time()
