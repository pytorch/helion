from __future__ import annotations

import dataclasses
import functools
import hashlib
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from .._compiler.device_ir import DeviceIR


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


# Only codegen/perf-affecting settings belong in run_id; full settings stay in
# metadata. Keep sorted for a stable wire format.
_CODEGEN_SETTINGS: tuple[str, ...] = (
    "allow_warp_specialize",
    "backend",
    "debug_dtype_asserts",
    "dot_precision",
    "fast_math",
    "index_dtype",
    "pallas_interpret",
    "persistent_reserved_sms",
    "static_shapes",
    "triton_do_not_specialize",
)


def _codegen_signature(settings: dict[str, object] | None) -> str:
    """Generates a stable, reproducible run_id string based on code-generation
    settings. Iterates through a fixed, sorted list of settings. Missing keys
    are marked as None. Any change to a codegen-affecting setting automatically
    changes the run_id. The complete settings are stored separately in the
    metadata.
    """
    if not settings:
        return ""
    return ", ".join(f"{name}={settings.get(name)}" for name in _CODEGEN_SETTINGS)


@dataclasses.dataclass
class KernelMetadata:
    """A metadata record containing the full context (source, shapes, dtypes, hardware,
    settings) for an autotune run computed lazily on first access. Saved as a single JSON
    line in <autotune_log>.meta.jsonl, acting as a sidecar to the configuration CSV.
    Telemetry CSV rows map many-to-one to this metadata record using run_id as the foreign
    key.
    """

    kernel_name: str = ""
    kernel_source: str = ""
    input_shapes: str = ""
    dtypes: str = ""
    hardware: str = ""
    settings: dict[str, object] | None = None
    # Derived artifact source; never part of dataclass identity or run_id.
    _device_ir: DeviceIR | None = dataclasses.field(
        default=None, repr=False, compare=False, hash=False
    )

    @functools.cached_property
    def run_id(self) -> str:
        """Stable content hash for joining CSV rows to sidecar metadata."""
        # _device_ir is derived from these fields, so it must stay out of identity.
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
            "settings": self.settings,
            "ir_graph": self.ir_graph,
        }

    @functools.cached_property
    def ir_graph(self) -> dict[str, object] | None:
        if self._device_ir is None:
            return None
        from ._metadata.ir_features import _has_networkx_node_link
        from ._metadata.ir_features import extract_ir_graph

        # Old/absent networkx degrades to None (extract_ir_graph would raise); a
        # missing optional dep must not drop the whole record via the end_run guard.
        if not _has_networkx_node_link():
            return None
        return extract_ir_graph(self._device_ir)
