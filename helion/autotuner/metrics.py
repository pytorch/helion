from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

_post_autotune_hooks: list[Callable[[AutotuneMetrics], None]] = []
_kernel_metadata_hooks: list[Callable[[KernelMetadata], None]] = []


def register_post_autotune_hook(hook: Callable[[AutotuneMetrics], None]) -> None:
    _post_autotune_hooks.append(hook)


def remove_post_autotune_hook(hook: Callable[[AutotuneMetrics], None]) -> None:
    _post_autotune_hooks.remove(hook)


def _run_post_autotune_hooks(metrics: AutotuneMetrics) -> None:
    for hook in _post_autotune_hooks:
        hook(metrics)


def register_kernel_metadata_hook(hook: Callable[[KernelMetadata], None]) -> None:
    """Register a hook fired whenever a kernel is bound to a config and compiled.

    Unlike :func:`register_post_autotune_hook`, this fires on every path that
    activates a config (autotune, on-disk cache hit, and default config), so it
    captures kernels that never go through the autotuner.
    """
    _kernel_metadata_hooks.append(hook)


def remove_kernel_metadata_hook(hook: Callable[[KernelMetadata], None]) -> None:
    _kernel_metadata_hooks.remove(hook)


def _run_kernel_metadata_hooks(metadata: KernelMetadata) -> None:
    for hook in _kernel_metadata_hooks:
        hook(metadata)


@dataclasses.dataclass
class AutotuneMetrics:
    _start_time: float = dataclasses.field(default_factory=time.perf_counter)
    num_configs_tested: int = 0
    num_compile_failures: int = 0
    num_accuracy_failures: int = 0
    num_generations: int = 0
    autotune_time: float = 0.0
    best_perf_ms: float = 0.0
    kernel_idx: int = -1
    kernel_name: str = ""
    kernel_source_hash: str = ""
    input_shapes: str = ""
    hardware: str = ""
    random_seed: int = 0
    search_algorithm: str = ""

    def finalize(self) -> None:
        self.autotune_time = time.perf_counter() - self._start_time

    def to_dict(self) -> dict[str, object]:
        return {
            "kernel_idx": self.kernel_idx,
            "kernel_name": self.kernel_name,
            "kernel_source_hash": self.kernel_source_hash,
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
    """Metadata for a single kernel that has been bound to a config and compiled.

    Emitted on every path that activates a config so that cache hits and
    default-config runs are captured in addition to autotuned runs. ``path``
    records which path produced the record ("autotune", "default", or
    "explicit").
    """

    kernel_idx: int = -1
    kernel_name: str = ""
    kernel_source_hash: str = ""
    config: str = ""
    input_shapes: str = ""
    dtypes: str = ""
    hardware: str = ""
    path: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "kernel_idx": self.kernel_idx,
            "kernel_name": self.kernel_name,
            "kernel_source_hash": self.kernel_source_hash,
            "config": self.config,
            "input_shapes": self.input_shapes,
            "dtypes": self.dtypes,
            "hardware": self.hardware,
            "path": self.path,
        }
