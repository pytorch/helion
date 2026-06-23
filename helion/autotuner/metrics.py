from __future__ import annotations

import dataclasses
import functools
import hashlib
import logging
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


# Codegen/perf-affecting settings hashed into run_id, so invocations that would
# generate different code never collide. Kept as a small local tuple (not Settings
# field metadata) so settings.py stays untouched; the full settings are still
# recorded verbatim in the .meta.jsonl record. Sorted for a stable wire format.
# Every value is a scalar/enum/torch.dtype, so run_id is reproducible across
# processes (no callables/paths/seed leak in).
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
    """Join codegen-affecting settings into a stable, reproducible string for run_id.

    Iterates the fixed local set of codegen-affecting settings, so any codegen
    change alters run_id. Order is the tuple's (sorted) order for a stable wire
    format; missing keys render as None. Full settings are stored separately in
    metadata.
    """
    if not settings:
        return ""
    return ", ".join(f"{name}={settings.get(name)}" for name in _CODEGEN_SETTINGS)


@dataclasses.dataclass
class KernelMetadata:
    """Per-run identity for the autotuned kernel; a passive shell over its sources.

    Appended (one JSON record per run) to the ``<autotune_log>.meta.jsonl``
    sidecar that sits next to the per-config CSV telemetry. The CSV records each
    config and its result; this record provides the kernel context (source,
    shapes, dtypes, hardware, settings) those rows join back to, plus the
    config-independent ``ir_graph`` device-IR dump. ``ir_graph`` is extracted
    lazily in :meth:`to_dict` from the held ``_device_ir`` (the dataset-only write
    path), so the object itself just carries source references; it is a derived
    artifact (a function of ``run_id``), excluded from the ``run_id`` hash.

    ``run_id`` is the single foreign key for an autotune *invocation*: a direct
    content hash of ``(kernel_source, codegen-settings signature, input_shapes,
    dtypes, hardware)``. The same invocation produces the same ``run_id`` across
    processes and CI runs (enabling dedup/aggregation), and any change to the
    kernel, codegen-affecting settings, shapes, dtypes, or hardware changes it.
    Because every CSV row is also stamped with ``run_id``, rows join to exactly
    one meta record (a clean many-to-one).

    ``run_id`` is a :func:`functools.cached_property` computed on first access.
    ``kernel_source`` carries the full source text and ``settings`` the full
    reproduction context for analysis.
    """

    kernel_name: str = ""
    kernel_source: str = ""
    input_shapes: str = ""
    dtypes: str = ""
    hardware: str = ""
    settings: dict[str, object] | None = None
    # Non-serialized source ref: the config-independent device IR. ir_graph is
    # extracted from it lazily in to_dict() (the dataset-only write path at run
    # end), keeping this object a passive shell. Excluded from run_id (a derived
    # artifact, not identity); repr/compare/hash off so the large IR object never
    # bloats repr or participates in dataclass equality/hashing.
    _device_ir: DeviceIR | None = dataclasses.field(
        default=None, repr=False, compare=False, hash=False
    )

    @functools.cached_property
    def run_id(self) -> str:
        """Stable, content-derived id for this run, computed once and cached.

        Hashed from ``(kernel_source, codegen-settings signature, input_shapes,
        dtypes, hardware)`` joined with a delimiter that cannot appear inside the
        fields (so boundaries can't collide). Content-derived, so the same
        invocation yields the same ``run_id`` across processes and CI runs.
        """
        # Hash exactly these five identity fields. The device IR (_device_ir) is
        # intentionally NOT included: its ir_graph dump is a derived artifact (a
        # function of run_id), not identity, so hashing it would be circular.
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
            "ir_graph": self._ir_graph(),
        }

    def _ir_graph(self) -> dict[str, object] | None:
        """Lazily extract the device-IR node-link dump (dataset-only, best-effort).

        Called from :meth:`to_dict` at run end (after codegen). Safe to defer: the
        autotune loop never mutates the original device IR -- codegen specializes
        on copies (``DeviceIR.build_codegen_graphs``) -- so the dump matches a
        pre-codegen extraction (block dims stay symbolic). Returns ``None`` when
        there is no IR or extraction fails, so the telemetry write never breaks
        autotuning.
        """
        if self._device_ir is None:
            return None
        from ._metadata.ir_features import extract_ir_graph

        try:
            return extract_ir_graph(self._device_ir)
        except Exception:
            # Best-effort: never break autotuning. Visible at HELION_AUTOTUNE_LOG_LEVEL
            # =DEBUG so silent degradation (e.g. a changed _compiler API) is traceable.
            logging.getLogger(__name__).debug(
                "device-IR extraction failed; recording ir_graph=None", exc_info=True
            )
            return None
