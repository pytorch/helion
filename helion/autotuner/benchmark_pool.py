from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable
from typing import Literal
from typing import NamedTuple
from typing import TypeVar
from typing import cast

import torch
from torch.utils._pytree import tree_map_only

from .benchmark_job import PrecompileJob
from .benchmark_worker import BenchmarkSubprocessError
from .benchmark_worker import BenchmarkWorkerPool

if TYPE_CHECKING:
    from ..runtime.config import Config
    from ..runtime.kernel import CompiledConfig
    from .logger import AutotuningLogger
    from .metrics import AutotuneMetrics
    from .precompile_future import SerializedCompiledFunction

_T = TypeVar("_T")


class PoolPrecompileResult(NamedTuple):
    is_workings: list[bool]
    statuses: list[Literal["ok", "error", "timeout"]]
    compile_times: list[float | None]


def estimate_tree_bytes(obj: object) -> int:
    """Estimate pytree tensor storage, counting shared storage once."""
    total = 0
    seen_ptrs: set[int] = set()

    def _accumulate(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal total
        size = tensor.element_size() * tensor.numel()
        try:
            storage = tensor.untyped_storage()
        except RuntimeError:
            pass
        else:
            ptr = storage.data_ptr()
            if ptr in seen_ptrs:
                return tensor
            seen_ptrs.add(ptr)
            size = storage.nbytes()
        total += size
        return tensor

    tree_map_only(torch.Tensor, _accumulate, obj)
    return total


class PoolBenchmarkManager:
    """Owns the long-lived worker pool for one autotune call."""

    def __init__(
        self,
        *,
        num_workers: int,
        log: AutotuningLogger,
        autotune_metrics: AutotuneMetrics,
    ) -> None:
        self._pool = BenchmarkWorkerPool(num_workers)
        self._log = log
        self._autotune_metrics = autotune_metrics
        self._precompile_worker_by_fn: dict[int, int] = {}

    def shutdown(self) -> None:
        self._pool.shutdown()
        self._precompile_worker_by_fn.clear()

    def worker_index_for_fn(self, fn: Callable[..., object]) -> int:
        return self._precompile_worker_by_fn.get(id(fn), 0)

    def run_on(self, worker_index: int, job: Callable[[], _T], timeout: float) -> _T:
        return self._pool.run_on(worker_index, job, timeout=timeout)

    def precompile(
        self,
        configs: list[Config],
        fns: list[CompiledConfig],
        *,
        args_path: str,
        timeout: float,
        desc: str | None,
        serialize_fn: Callable[[CompiledConfig], SerializedCompiledFunction | None],
    ) -> PoolPrecompileResult:
        """Compile each config in the worker pool."""
        jobs: list[PrecompileJob | None] = []
        for fn in fns:
            fn_spec = serialize_fn(fn)
            jobs.append(
                PrecompileJob(fn_spec=fn_spec, args_path=args_path)
                if fn_spec is not None
                else None
            )

        live_idxs = [i for i, job in enumerate(jobs) if job is not None]
        live_jobs = cast("list[Callable[[], object]]", [jobs[i] for i in live_idxs])
        self._pool.start_all(limit=len(live_jobs))
        live_results = self._pool.map_jobs(live_jobs, timeout=timeout)

        is_workings = [False] * len(configs)
        statuses: list[Literal["ok", "error", "timeout"]] = ["error"] * len(configs)
        compile_times: list[float | None] = [None] * len(configs)
        for idx, job in enumerate(jobs):
            if job is None:
                self._log.debug(
                    f"Precompile worker could not serialize {configs[idx]!r}"
                )
                self._autotune_metrics.num_compile_failures += 1

        for idx, result in zip(live_idxs, live_results, strict=True):
            compile_times[idx] = result.elapsed
            job_result = result.result
            if isinstance(job_result, BaseException):
                statuses[idx] = (
                    "timeout"
                    if isinstance(job_result, BenchmarkSubprocessError)
                    and "timeout" in str(job_result).lower()
                    else "error"
                )
                self._log.debug(
                    f"Precompile worker failed for {configs[idx]!r}: "
                    f"{type(job_result).__name__}: {job_result}"
                )
                self._autotune_metrics.num_compile_failures += 1
            elif job_result is True:
                is_workings[idx] = True
                statuses[idx] = "ok"
                self._precompile_worker_by_fn[id(fns[idx])] = result.worker_index
            else:
                self._log.debug(
                    f"Precompile worker returned failure for {configs[idx]!r}: "
                    f"{job_result!r}"
                )
                self._autotune_metrics.num_compile_failures += 1

        if desc:
            self._log(f"{desc} 100% via worker pool ({len(live_idxs)} configs)")
        return PoolPrecompileResult(is_workings, statuses, compile_times)
