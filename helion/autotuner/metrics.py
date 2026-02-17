from __future__ import annotations

import dataclasses
import time

_autotune_metrics: list[dict[str, object]] = []


def register_autotune_metrics(metrics: AutotuneMetrics) -> None:
    _autotune_metrics.append(metrics.to_dict())


def get_all_autotune_metrics() -> list[dict[str, object]]:
    return list(_autotune_metrics)


def clear_autotune_metrics() -> None:
    _autotune_metrics.clear()


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
    input_shapes: str = ""
    hardware: str = ""
    random_seed: int = 0

    def finalize(self) -> None:
        self.autotune_time = time.perf_counter() - self._start_time

    def to_dict(self) -> dict[str, object]:
        configs_per_second = (
            self.num_configs_tested / self.autotune_time
            if self.autotune_time > 0
            else 0.0
        )
        return {
            "kernel_name": self.kernel_name,
            "input_shapes": self.input_shapes,
            "hardware": self.hardware,
            "random_seed": self.random_seed,
            "num_configs_tested": self.num_configs_tested,
            "num_compile_failures": self.num_compile_failures,
            "num_accuracy_failures": self.num_accuracy_failures,
            "num_generations": self.num_generations,
            "autotune_time": self.autotune_time,
            "best_perf_ms": self.best_perf_ms,
            "configs_per_second": configs_per_second,
        }
