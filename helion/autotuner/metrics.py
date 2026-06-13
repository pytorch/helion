from __future__ import annotations

import dataclasses
import hashlib
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

_post_autotune_hooks: list[Callable[[AutotuneMetrics], None]] = []


def register_post_autotune_hook(hook: Callable[[AutotuneMetrics], None]) -> None:
    _post_autotune_hooks.append(hook)


def remove_post_autotune_hook(hook: Callable[[AutotuneMetrics], None]) -> None:
    _post_autotune_hooks.remove(hook)


def _run_post_autotune_hooks(metrics: AutotuneMetrics) -> None:
    for hook in _post_autotune_hooks:
        hook(metrics)


@dataclasses.dataclass
class AutotuneMetrics:
    _start_time: float = dataclasses.field(default_factory=time.perf_counter)
    num_configs_tested: int = 0
    num_compile_failures: int = 0
    num_accuracy_failures: int = 0
    num_generations: int = 0
    autotune_time: float = 0.0
    best_perf_ms: float = 0.0
    kernel_id: str = ""
    kernel_name: str = ""
    kernel_source: str = ""
    input_shapes: str = ""
    hardware: str = ""
    random_seed: int = 0
    search_algorithm: str = ""

    def finalize(self) -> None:
        self.autotune_time = time.perf_counter() - self._start_time

    def to_dict(self) -> dict[str, object]:
        return {
            "kernel_id": self.kernel_id,
            "kernel_name": self.kernel_name,
            "kernel_source": self.kernel_source,
            "input_shapes": self.input_shapes,
            "hardware": self.hardware,
            "random_seed": self.random_seed,
            "search_algorithm": self.search_algorithm,
            "num_configs_tested": self.num_configs_tested,
            "num_compile_failures": self.num_compile_failures,
            "num_accuracy_failures": self.num_accuracy_failures,
            "num_generations": self.num_generations,
            "autotune_time": self.autotune_time,
            "best_perf_ms": self.best_perf_ms,
        }


@dataclasses.dataclass
class KernelMetadata:
    """Per-run identity for the kernel being autotuned.

    Appended (one JSON record per run) to the ``<autotune_log>.meta.jsonl``
    sidecar that sits next to the per-config CSV telemetry. The CSV records each
    config and its result; this provides the stable identity needed to group
    those rows across runs.

    Two content-derived ids tie the data together:

    - ``kernel_id`` is the foreign key for the *kernel* (a hash of the kernel
      source and code-generation settings); it is shape/dtype independent, so a
      kernel autotuned at several input shapes shares one ``kernel_id``.
    - ``run_id`` is the foreign key for a single autotune *invocation*: a hash of
      ``(kernel_id, input_shapes, dtypes, hardware)``. Because every CSV row is
      also stamped with ``run_id``, rows join to exactly one meta record on
      ``run_id`` (a clean many-to-one), which lets a config's measured perf be
      attributed to the specific shape/dtype/hardware it was measured at.

    ``run_id`` is derived from the other fields in :meth:`__post_init__` when not
    provided. ``kernel_source`` carries the full source text for analysis.
    """

    kernel_id: str = ""
    kernel_name: str = ""
    kernel_source: str = ""
    input_shapes: str = ""
    dtypes: str = ""
    hardware: str = ""
    run_id: str = ""

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = self._compute_run_id()

    def _compute_run_id(self) -> str:
        """Stable id for this ``(kernel, input shapes, dtypes, hardware)`` run.

        Content-derived so the same invocation produces the same ``run_id``
        across processes and CI runs (enabling dedup/aggregation). The fields are
        joined with a delimiter that cannot appear inside them to avoid
        boundary collisions.
        """
        payload = (
            f"{self.kernel_id}\x00{self.input_shapes}\x00{self.dtypes}\x00"
            f"{self.hardware}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "kernel_id": self.kernel_id,
            "kernel_name": self.kernel_name,
            "kernel_source": self.kernel_source,
            "input_shapes": self.input_shapes,
            "dtypes": self.dtypes,
            "hardware": self.hardware,
        }
