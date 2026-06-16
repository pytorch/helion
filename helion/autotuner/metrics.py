from __future__ import annotations

import dataclasses
import functools
import hashlib
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from .._hardware import HardwareInfoRecord

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


def _codegen_signature(settings: dict[str, object] | None) -> str:
    """Join codegen-affecting settings into a stable string for run_id.

    Iterates CODEGEN_AFFECTING_SETTINGS tagged at field definition, so any
    codegen change alters run_id. Order is alphabetical for stability across
    field reorders; missing keys render as None. Full settings are stored
    separately in metadata.
    """
    if not settings:
        return ""
    from ..runtime.settings import CODEGEN_AFFECTING_SETTINGS

    return ", ".join(
        f"{name}={settings.get(name)}" for name in CODEGEN_AFFECTING_SETTINGS
    )


@dataclasses.dataclass
class KernelMetadata:
    """Per-run identity for the kernel being autotuned.

    Appended (one JSON record per run) to the ``<autotune_log>.meta.jsonl``
    sidecar that sits next to the per-config CSV telemetry. The CSV records each
    config and its result; this record provides the kernel context (source,
    shapes, dtypes, hardware, settings) those rows join back to.

    ``run_id`` is the single foreign key for an autotune *invocation*: a direct
    content hash of ``(kernel_source, codegen-settings signature, input_shapes,
    dtypes, hardware)``. The same invocation produces the same ``run_id`` across
    processes and CI runs (enabling dedup/aggregation), and any change to the
    kernel, codegen-affecting settings, shapes, dtypes, or hardware changes it.
    Because every CSV row is also stamped with ``run_id``, rows join to exactly
    one meta record (a clean many-to-one).

    ``run_id`` is a :func:`functools.cached_property` computed on first access.
    ``kernel_source`` carries the full source text and ``settings`` the full
    reproduction context for analysis. ``hardware_info`` is a best-effort
    structured hardware/runtime snapshot for the cost model; it is descriptive
    only and deliberately excluded from ``run_id`` so software-version bumps do
    not fragment the dataset.
    """

    kernel_name: str = ""
    kernel_source: str = ""
    input_shapes: str = ""
    dtypes: str = ""
    hardware: str = ""
    hardware_info: HardwareInfoRecord | None = None
    settings: dict[str, object] | None = None

    @functools.cached_property
    def run_id(self) -> str:
        """Stable, content-derived id for this run, computed once and cached.

        Hashed from ``(kernel_source, codegen-settings signature, input_shapes,
        dtypes, hardware)`` joined with a delimiter that cannot appear inside the
        fields (so boundaries can't collide). Content-derived, so the same
        invocation yields the same ``run_id`` across processes and CI runs.
        """
        payload = (
            f"{self.kernel_source}\x00{_codegen_signature(self.settings)}\x00"
            f"{self.input_shapes}\x00{self.dtypes}\x00{self.hardware}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "kernel_name": self.kernel_name,
            "kernel_source": self.kernel_source,
            "input_shapes": self.input_shapes,
            "dtypes": self.dtypes,
            "hardware": self.hardware,
            "hardware_info": self.hardware_info,
            "settings": self.settings,
        }
