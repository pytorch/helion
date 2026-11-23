# ruff: noqa: ANN401
"""Shared helpers for the better-benchmark Inductor-vs-Helion comparison.

better-benchmark (https://github.com/eellison/better-benchmark) extracts the
fused regions TorchInductor compiles from real models as standalone repros
(``repros/canonical/<repro_id>/repro.py``). Each module under ``kernels/`` is a
hand port of one repro to Helion: it defines the Helion kernel(s) equivalent to
the Triton kernel(s) Inductor generates for that repro and a ``forward(inputs)``
that takes the repro's ``make_inputs()`` list and returns what
``Repro.forward`` returns.

Every run gets freshly generated, seeded inputs (``make_inputs``) rather than
copies, so strides (including expanded stride-0 dims) and the storage shared
by better-benchmark's alias groups are exactly what the repro specifies.

The better-benchmark checkout is found through ``BETTER_BENCHMARK_ROOT``,
defaulting to a ``better-benchmark`` directory next to this repository. The
ports read the repros' inputs by position, so they hold for the commit
``selection.json`` names (``better_benchmark_commit``); ``bb_path`` warns once
when the checkout is elsewhere.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import functools
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pwd
import re
import statistics
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
import warnings

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence
    import types

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = Path(__file__).resolve().parent
KERNELS_DIR = PACKAGE_DIR / "kernels"
SELECTION_PATH = PACKAGE_DIR / "selection.json"
BB_ROOT = Path(
    os.environ.get("BETTER_BENCHMARK_ROOT", REPO_ROOT.parent / "better-benchmark")
)
_PATTERN_HASH = re.compile(r"_([0-9a-f]{12})$")


def bb_path() -> Path:
    """The better-benchmark checkout, put on ``sys.path`` for its modules."""
    if not (BB_ROOT / "repro_harness.py").exists():
        raise FileNotFoundError(
            f"better-benchmark checkout not found at {BB_ROOT}; "
            "set BETTER_BENCHMARK_ROOT"
        )
    if str(BB_ROOT) not in sys.path:
        sys.path.insert(0, str(BB_ROOT))
    _check_bb_revision()
    return BB_ROOT


@functools.cache
def _check_bb_revision() -> None:
    """Warn, once, when the checkout is not at the commit the ports and
    ``selection.json`` were made against: a re-captured repro can reorder or
    reshape the inputs the ports read by position.  Uncommitted changes
    (``git_revision``'s ``-dirty``) do not move it off the commit."""
    pinned = json.loads(SELECTION_PATH.read_text()).get("better_benchmark_commit")
    have = git_revision(BB_ROOT)
    if pinned and have and have.removesuffix("-dirty") != pinned:
        warnings.warn(
            f"better-benchmark checkout {BB_ROOT} is at {have}; the ports and "
            f"selection.json were made against {pinned}",
            stacklevel=3,
        )


def family(repro_id: str) -> str:
    """The repro's family: its directory name without the pattern hash."""
    return _PATTERN_HASH.sub("", repro_id)


def pattern_hash(repro_id: str) -> str | None:
    match = _PATTERN_HASH.search(repro_id)
    return None if match is None else match.group(1)


def config_record_path(
    configs_dir: Path, repro_id: str, shape: str, backend: str
) -> Path:
    """Where ``autotune.py`` records one port's tuned decorators."""
    return configs_dir / f"{repro_id}__{shape}__{backend}.json"


def bench_record_path(bench_dir: Path, repro_id: str, shape: str) -> Path:
    """Where ``bench.py`` records one (repro, shape) point."""
    return bench_dir / f"{repro_id}__{shape}.json"


def bench_repeat_path(bench_dir: Path, repro_id: str, shape: str, repeat: int) -> Path:
    """Where ``campaign.py --bench-repeats`` has the ``repeat``-th bench
    process of a point (from 1) record it; ``bench_record_path`` is then the
    point's aggregate record (``aggregate_bench_records``)."""
    return bench_dir / f"{repro_id}__{shape}.r{repeat}.json"


def is_repeat_record(path: Path) -> bool:
    """Whether ``path`` names one process's record of a repeated bench
    (``bench_repeat_path``), which stands for its point only through the
    aggregate record."""
    return re.search(r"\.r\d+\.json$", path.name) is not None


# The timings of a repeated bench that the aggregate record gives as medians
# over the processes (``aggregate_bench_records``).
AGGREGATED_TIMINGS = ("bb_us", "cold_us", "rot_us", "kernel_us", "kernel_span_us")


def spans_overlap(kernel_us: float, kernel_span_us: float) -> bool:
    """Whether the profiler spans of an implementation's kernels overlap
    (dependent launches): their total, ``kernel_span_us``, is more than the
    total of the kernels' steps, ``kernel_us``, by over 1% of itself, which
    is beyond what the profiler's clock gives two kernels in a row. A bench
    record's ``kernel_spans_overlap``."""
    return kernel_span_us - kernel_us > 0.01 * kernel_span_us


# The Inductor compiles ``bench.py`` can time (``inductor_options``). A
# comparison takes the fastest of those a record holds, in each process of a
# repeated bench.
INDUCTOR_IMPLS = ("inductor", "inductor_cd", "inductor_max")


def _merged_activity(records: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The foreign-activity samples of several jobs as one job's
    (``watch_foreign_gpu_activity``): their counts and seconds added, the
    longest gap, every process under its name with its highest utilization;
    None when one of the jobs was not sampled. ``unknown_allowed`` is the
    failed samples that the jobs allow, each its own
    (``unknown_samples_allowed``), added up: the allowance of one job
    applied to the sums would fail samples that cover every job."""
    activities: list[dict[str, Any]] = []
    for record in records:
        activity = record.get("foreign_gpu_activity")
        if not activity:
            return None
        activities.append(activity)
    processes: dict[str, dict[str, Any]] = {}
    for activity in activities:
        for name, entry in activity["processes"].items():
            seen = processes.setdefault(name, {"max_sm": 0, "active_s": 0.0})
            seen["max_sm"] = max(seen["max_sm"], entry["max_sm"])
            seen["active_s"] = round(seen["active_s"] + entry["active_s"], 1)
            if "busy_s" in entry:
                seen["busy_s"] = round(seen.get("busy_s", 0.0) + entry["busy_s"], 1)
    merged: dict[str, Any] = {
        "interval_s": activities[0]["interval_s"],
        "samples": sum(a["samples"] for a in activities),
        "unknown_samples": sum(a["unknown_samples"] for a in activities),
        "unknown_allowed": sum(unknown_samples_allowed(a) for a in activities),
        "processes": processes,
    }
    for key, combine in (("watched_s", sum), ("max_gap_s", max)):
        if all(key in activity for activity in activities):
            merged[key] = round(combine(a[key] for a in activities), 1)
    return merged


def _all_or_any_false(values: Sequence[Any]) -> bool | None:
    """A verdict over several processes: False when one process says False,
    True when every one says True, None otherwise (some did not judge)."""
    if any(value is False for value in values):
        return False
    return True if all(value is True for value in values) else None


def aggregate_bench_records(
    processes: dict[str, dict[str, Any]], digests: dict[str, str]
) -> dict[str, Any]:
    """The aggregate record of a point that ``campaign.py --bench-repeats``
    benched in several fresh processes: ``processes`` are their ok records
    by file name, in the order of the repeats, ``digests`` the SHA-256 of
    each file (the campaign aggregates again when they change).

    A process counts unless foreign work contaminated its record or its
    activity samples do not cover it (``left_out`` says why; when none
    counts, all do and the aggregate is flagged ``contaminated``, like the
    cleanest attempt of a job that was contaminated every time). The
    aggregate is a bench record: each implementation's entry is that of its
    representative process, the one whose ``cold_us`` is the median or
    nearest to it, with

    - ``cold_us``, ``bb_us``, ``rot_us``, ``kernel_us`` and
      ``kernel_span_us`` replaced by the median over the counted processes
      (``AGGREGATED_TIMINGS``; the throughputs follow, and
      ``kernel_spans_overlap`` is of the two medians: a timing that one
      process lacks has no median, and the flag goes with either total),
      and the per-round fields left to the processes' own records.
      ``kernels`` stays the representative's (``representative`` names its
      record), so its entries need not add up to ``kernel_us``;
    - ``per_process``: every counted process's status, verdicts and timings
      in order, and for an Inductor compile the configs its kernels ended
      on in each (``inductor_configs``: coordinate descent tunes again in
      every process);
    - ``status`` ok only when the implementation is ok and timed in every
      counted process, ``numerics_pass`` and ``code_matches_autotune`` false
      when any process says so: a result that one of several processes does
      not reproduce is not a result.

    ``report.py`` takes the ratio of two implementations from
    ``per_process``: the ratio in each process (the fastest of the
    Inductor compiles of that process over the port's time in the same
    process), then the median of those ratios. The ratio of the medians is
    not that number: the processes' Inductor picks differ, and a ratio is
    only of timings that one process interleaved. The baseline of a process
    is of the compiles that are usable in that process (its ``per_process``
    status and verdict), whatever the aggregate's ``status`` of a compile
    says: one that failed in a single process is missing from that
    process's baseline alone."""
    # A process made without the sampling counts, as a record made without
    # it does where the report admits those.
    left_out = {
        name: issue
        for name, record in processes.items()
        if (issue := activity_issue(record, include_unsampled=True)) is not None
    }
    counted = {name: r for name, r in processes.items() if name not in left_out}
    contaminated = not counted
    if contaminated:
        counted = dict(processes)
    names = list(counted)
    records = list(counted.values())
    first = records[0]
    out: dict[str, Any] = {
        key: first[key]
        for key in ("repro", "shape", "gpu", "versions", "outputs", "total_bytes")
        if key in first
    }
    out["status"] = "ok"
    out["aggregate"] = {
        "processes": len(records),
        "counted": names,
        "left_out": left_out,
        "records": digests,
        "gpus": [record.get("gpu") for record in records],
        "placement_seeds": [
            (record.get("placement") or {}).get("seed") for record in records
        ],
    }
    if contaminated:
        out["contaminated"] = True
    total_bytes = out.get("total_bytes")
    impls: dict[str, Any] = {}
    for impl in dict.fromkeys(impl for record in records for impl in record["impls"]):
        infos = [
            record["impls"].get(impl) or {"status": "not timed"} for record in records
        ]
        per_process: dict[str, list[Any]] = {
            "status": [info.get("status") for info in infos],
            "numerics_pass": [info.get("numerics_pass") for info in infos],
        }
        medians: dict[str, float] = {}
        for metric in AGGREGATED_TIMINGS:
            values = [info.get(metric) for info in infos]
            if all(value is None for value in values):
                continue
            per_process[metric] = values
            timed = [value for value in values if value is not None]
            if len(timed) == len(values):
                medians[metric] = statistics.median(timed)
        if impl in INDUCTOR_IMPLS:
            per_process["inductor_configs"] = [
                info.get("inductor_configs") for info in infos
            ]
        # The representative: the process whose cold_us is nearest the median.
        cold = medians.get("cold_us")
        index = 0
        if cold is not None:
            index = min(
                range(len(infos)), key=lambda i: abs(per_process["cold_us"][i] - cold)
            )
        info = {
            key: value
            for key, value in infos[index].items()
            if not key.endswith(("_runs_us", "_order_medians_us", "_order_mins_us"))
            and key not in AGGREGATED_TIMINGS
            and not key.endswith("_gbps")
        }
        info.update(medians)
        info.pop("kernel_spans_overlap", None)
        if "kernel_us" in medians and "kernel_span_us" in medians:
            info["kernel_spans_overlap"] = spans_overlap(
                medians["kernel_us"], medians["kernel_span_us"]
            )
        if total_bytes:
            for metric, value in medians.items():
                # The throughputs bench.py records.
                if value > 0 and metric != "kernel_span_us":
                    info[metric.replace("_us", "_gbps")] = total_bytes / value / 1e3
        info["representative"] = names[index]
        info["per_process"] = per_process
        failed = [
            f"{name}: {status}"
            for name, status in zip(names, per_process["status"], strict=True)
            if status != "ok"
        ]
        if "cold_us" in per_process and "cold_us" not in medians:
            failed.append("cold_us is missing in a process")
        if failed:
            info["status"] = "error"
            info["error"] = f"not ok in every process ({'; '.join(failed)})"
        for verdict in ("numerics_pass", "code_matches_autotune"):
            values = [i.get(verdict) for i in infos]
            if any(value is not None for value in values):
                info[verdict] = _all_or_any_false(values)
        impls[impl] = info
    out["impls"] = impls
    out["foreign_gpu_activity"] = _merged_activity(records)
    for key in ("foreign_gpu_processes_before", "foreign_gpu_processes_after"):
        # None in a process: nvidia-smi could not tell.
        listed: set[str] | None = set()
        for record in records:
            found_processes = record.get(key)
            if found_processes is None or listed is None:
                listed = None
            else:
                listed.update(found_processes)
        out[key] = None if listed is None else sorted(listed)
    if "rotate_inputs" in first:
        out["rotate_inputs"] = first["rotate_inputs"]
    out["wall_s"] = round(sum(record.get("wall_s") or 0.0 for record in records), 1)
    return out


# The implementations ``bench.py`` times, in the order of a forward round (its
# ``--impls``). ``helion_<backend>`` is the Helion port pinned to that
# backend's autotune record.
ALL_IMPLS = (
    "inductor",
    "inductor_cd",
    "inductor_max",
    "oracle",
    "helion_triton",
    "helion_cute",
    "memcopy",
)


def repro_dir(repro_id: str) -> Path:
    path = bb_path() / "repros" / "canonical" / repro_id
    if not (path / "repro.py").exists():
        raise FileNotFoundError(f"no repro.py in {path}")
    return path


def load_repro_module(repro_id: str) -> types.ModuleType:
    bb_path()
    import oracle_harness  # pyrefly: ignore [missing-import]

    return oracle_harness.load_repro_module(repro_dir(repro_id))


def shape_configs(repro_id: str) -> dict[str, dict[str, Any]]:
    """``{shape_hash: config}`` for every shape point of a repro."""
    bb_path()
    import repro_harness  # pyrefly: ignore [missing-import]

    configs = repro_harness.load_shape_configs(str(repro_dir(repro_id) / "repro.py"))
    # Labels are ``<model>_<shape_hash>``; the hash alone names the point.
    return {label.rsplit("_", 1)[-1]: cfg for label, cfg in configs.items()}


@functools.cache
def selected_shapes() -> dict[str, str]:
    """``{repro_id: shape_hash}`` of the points the study measures
    (``selection.json``)."""
    shapes: dict[str, str] = {}
    for entry in json.loads(SELECTION_PATH.read_text())["kernels"]:
        shapes.setdefault(entry["repro"], entry["shape"])
    return shapes


def default_shape(repro_id: str) -> str:
    """The shape point a script uses without ``--shape``: the repro's point in
    ``selection.json``, or its first shape point for a repro outside it."""
    return selected_shapes().get(repro_id) or next(iter(shape_configs(repro_id)))


def seed_everything(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_inputs(repro_id: str, shape: str, seed: int = 0) -> list[Any]:
    """The repro's inputs at one shape point, generated deterministically.

    Call it once per run: the same seed regenerates identical values with the
    repro's exact strides and aliasing, which copying a previous set cannot
    guarantee (a clone densifies expanded dims and splits shared storage).
    """
    bb_path()
    import repro_harness  # pyrefly: ignore [missing-import]

    configs = shape_configs(repro_id)
    if shape not in configs:
        raise KeyError(f"{repro_id} has no shape point {shape}; have {list(configs)}")
    seed_everything(seed)
    return repro_harness.make_inputs_from_config(configs[shape])


def input_storages(inputs: Sequence[Any]) -> dict[int, int]:
    """``{address: bytes}`` of the distinct storages behind the tensors of
    one input set, in the order of the inputs (views of one buffer are one
    storage, whatever part of it they show)."""
    import torch

    storages = {}
    for value in inputs:
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
    return storages


def input_bytes(inputs: Sequence[Any]) -> int:
    """The device memory behind one input set: the bytes of the distinct
    storages of its tensors."""
    return sum(input_storages(inputs).values())


def compile_policy(repro_id: str, shape: str) -> dict[str, Any]:
    bb_path()
    import repro_harness  # pyrefly: ignore [missing-import]

    return repro_harness.compile_policy_from_config(shape_configs(repro_id)[shape])


def inductor_options(impl: str) -> dict[str, Any]:
    """The ``torch.compile`` options an Inductor implementation adds to the
    shape point's compile policy: none for ``inductor``,
    ``coordinate_descent_tuning`` for ``inductor_cd``, and for
    ``inductor_max`` what ``torch.compile(mode="max-autotune")`` sets, as
    PyTorch lists them, without the mode's CUDA-graph wrapper
    (``triton.cudagraphs``). The wrapper replays the compiled function as a
    CUDA graph of its own, which cannot run inside the CUDA graphs the timers
    capture every implementation in; the kernels and their tuning are those
    of the mode (the same options as ``mode="max-autotune-no-cudagraphs"``).
    ``torch.compile`` takes a mode or options, not both, and the compile
    policy is options."""
    if impl == "inductor":
        return {}
    if impl == "inductor_cd":
        return {"coordinate_descent_tuning": True}
    if impl == "inductor_max":
        import torch._inductor

        options = dict(torch._inductor.list_mode_options("max-autotune"))
        del options["triton.cudagraphs"]
        return options
    raise ValueError(f"not an Inductor implementation: {impl!r}")


def compile_inductor(
    repro_id: str, shape: str, instance: Any, *, impl: str = "inductor"
) -> Callable[..., Any]:
    """``torch.compile`` of the repro as better-benchmark compiles it (the
    shape point's compile policy), with the options of the Inductor
    implementation ``impl`` (``inductor_options``). Compiles with different
    options coexist in one process (each wraps its own backend), so no
    dynamo reset is needed in between."""
    bb_path()
    import repro_harness  # pyrefly: ignore [missing-import]

    options = inductor_options(impl)
    return repro_harness.compile_repro(
        instance,
        compile_policy=compile_policy(repro_id, shape),
        **({"options": options} if options else {}),
    )


@contextlib.contextmanager
def inductor_kernels_launched() -> Iterator[list[Any]]:
    """The Triton kernels (Inductor's ``CachingAutotuner`` objects) that the
    Inductor-compiled code called in the block launches, in the order of
    their first launch: ``inductor_kernel_configs`` reads the configs they
    ended on. Taken at the launches, not from the modules Inductor has
    loaded: those hold the kernels of every compile of the process, the
    default one's beside the tuned one's under the same names, and the
    variants that a mix-order reduction's compile benchmarks its split
    sizes with. The compile happens at the first call of the compiled
    code, which launches those variants too (their benchmark runs are left
    out here, their warm-up launches are not): take the block around a
    later call."""
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    launched: list[Any] = []
    run = CachingAutotuner.run

    def recording(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not kwargs.get("benchmark_run") and not any(
            kernel is self for kernel in launched
        ):
            launched.append(self)
        return run(self, *args, **kwargs)

    CachingAutotuner.run = recording  # type: ignore[method-assign]
    try:
        yield launched
    finally:
        CachingAutotuner.run = run  # type: ignore[method-assign]


def inductor_kernel_configs(launched: Sequence[Any]) -> list[dict[str, Any]]:
    """What a bench record says of each Triton kernel that an Inductor
    compile launches (``inductor_kernels_launched``, after a call: a kernel
    settles on one launcher at its first launch, where coordinate descent
    tunes it): its name, the launcher's config (``block_sizes``: the
    config's kwargs, such as ``XBLOCK``, ``R0_BLOCK`` and, for a mix-order
    reduction, ``RSPLIT_SIZE``; ``num_warps``; ``num_stages``), whether
    coordinate descent found it, the split size the kernel was generated
    for (``rsplit_size``, mix-order reductions only) and the registers,
    spills and shared memory of the compiled kernel. A kernel that holds
    another number of launchers than one lists them all (``launchers``)."""
    kernels = []
    for kernel in launched:
        meta = kernel.inductor_meta
        configs = [
            {
                "block_sizes": dict(launcher.config.kwargs),
                "num_warps": launcher.config.num_warps,
                "num_stages": launcher.config.num_stages,
                "found_by_coordesc": bool(
                    getattr(launcher.config, "found_by_coordesc", False)
                ),
                "n_regs": launcher.n_regs,
                "n_spills": launcher.n_spills,
                "shared": launcher.shared,
            }
            for launcher in kernel.launchers
        ]
        entry: dict[str, Any] = {
            "name": meta.get("kernel_name", kernel.fn.__name__),
            "grid_type": meta.get("grid_type"),
        }
        if meta.get("RSPLIT_SIZE") is not None:
            entry["rsplit_size"] = meta["RSPLIT_SIZE"]
        if len(configs) == 1:
            entry.update(configs[0])
        else:
            entry["launchers"] = configs
        kernels.append(entry)
    return kernels


def load_oracle(repro_id: str, inputs: list[Any]) -> Callable[[list[Any]], Any]:
    """The repro's hand-written oracle (``oracle.py``), dispatched for the
    shape of ``inputs``."""
    bb_path()
    import oracle_harness  # pyrefly: ignore [missing-import]

    _, forward, _ = oracle_harness._load_oracle_module(repro_dir(repro_id))
    resolved, _ = oracle_harness.resolve_oracle(forward, inputs)
    return resolved


def flatten_outputs(value: Any) -> list[Any]:
    bb_path()
    import oracle_harness  # pyrefly: ignore [missing-import]

    return oracle_harness._normalize_outputs(value)


def port_path(repro_id: str) -> Path:
    return KERNELS_DIR / f"{repro_id}.py"


@contextlib.contextmanager
def env_override(**values: str) -> Iterator[None]:
    """Set environment variables for the duration of the block."""
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def load_port(
    repro_id: str,
    backend: str,
    pinned: dict[str, str] | None = None,
) -> types.ModuleType:
    """Import the Helion port of ``repro_id`` with its kernels on ``backend``.

    Kernels take their backend from ``HELION_BACKEND`` when they are
    decorated, so each backend gets its own module instance. ``pinned`` maps a
    kernel's global name to the ``@helion.kernel(...)`` decorator the
    autotuner reported for it (config plus the codegen settings it ran with);
    every kernel is then rebuilt from that decorator, which reproduces the
    tuned kernel without searching.

    An unpinned port's kernels run their default configs
    (``HELION_AUTOTUNE_EFFORT=none``) unless the environment asks for a
    search: it names an effort, or ``HELION_FORCE_AUTOTUNE`` is true.
    """
    import torch

    import helion

    path = port_path(repro_id)
    if not path.exists():
        raise FileNotFoundError(f"no Helion port at {path}")
    # Settings read HELION_BACKEND (and HELION_FORCE_AUTOTUNE) when a
    # decorator expression is evaluated: the module's own decorators and the
    # pinned ones below both run under this override. A Triton decorator
    # names no backend, so the override is what makes it a Triton kernel.
    #
    # An unpinned port searches on its first call. The harness's callers say
    # which they mean (check_port.py: HELION_AUTOTUNE_EFFORT=none;
    # autotune.py and the campaign: an effort, with HELION_FORCE_AUTOTUNE=1).
    # Any other script that loads a port (a comparison written beside a
    # port, say) gets the default config, not a full search on a GPU that
    # others are timing on. Both variables are read the way the kernels'
    # settings read them: an empty effort is no effort, and "off" forces
    # nothing (settings made under the override, like the kernels': the
    # caller's HELION_BACKEND does not matter to a port).
    effort: dict[str, str] = {}
    if pinned is None and not os.environ.get("HELION_AUTOTUNE_EFFORT", "").strip():
        with env_override(HELION_BACKEND=backend):
            if not helion.Settings().force_autotune:
                effort["HELION_AUTOTUNE_EFFORT"] = "none"
    with env_override(HELION_BACKEND=backend, **effort):
        name = f"_bb_port_{repro_id}_{backend}"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        kernels = port_kernels(module)
        if not kernels:
            raise ValueError(f"{path} defines no @helion.kernel")
        for kernel_name, kernel in kernels.items():
            if kernel.settings.backend != backend:
                raise ValueError(
                    f"{repro_id}.{kernel_name} pins "
                    f"backend={kernel.settings.backend!r}; ports must leave the "
                    "backend to HELION_BACKEND"
                )
        if pinned is None:
            return module
        missing = set(kernels) - set(pinned)
        if missing:
            raise KeyError(f"no pinned config for {sorted(missing)} of {repro_id}")
        namespace = {"helion": helion, "torch": torch}
        # A pinned kernel must not search again, even in a process that
        # forces autotuning for its other kernels.
        with env_override(HELION_FORCE_AUTOTUNE="0"):
            for kernel_name, kernel in kernels.items():
                decorator = pinned[kernel_name].strip()
                if not decorator.startswith("@helion.kernel("):
                    raise ValueError(
                        f"unexpected decorator for {kernel_name}: {decorator}"
                    )
                rebuilt = eval(decorator[1:], namespace)(kernel.fn)
                if rebuilt.settings.force_autotune:
                    raise RuntimeError(f"pinned {kernel_name} would still autotune")
                if rebuilt.settings.backend != backend:
                    raise ValueError(
                        f"pinned decorator of {kernel_name} is for "
                        f"{rebuilt.settings.backend!r}, not {backend!r}"
                    )
                setattr(module, kernel_name, rebuilt)
    return module


def autotune_record(
    configs_dir: Path, repro_id: str, shape: str, backend: str
) -> dict[str, Any]:
    """The record of one port's autotune (``autotune.py``'s), which must be
    ok: what a bench pins the port to."""
    path = config_record_path(configs_dir, repro_id, shape, backend)
    record = json.loads(path.read_text())
    if record.get("status") != "ok":
        raise RuntimeError(f"autotune record {path} is not ok")
    return record


# The variable that gives a search its random seed. A campaign gives every
# autotune job one (``campaign.job_env``), and ``autotune.py`` records it with
# the rest of the job's HELION_* environment (the record's ``env``).
SEED_VARIABLE = "HELION_AUTOTUNE_RANDOM_SEED"
# The variable through which ``campaign.py --seeds-from`` tells an autotune
# job the run that its seed was taken from, which ``autotune.py`` records as
# ``seed_from``.
SEED_FROM = "BB_SEED_FROM"
# What ``campaign.py --configs-from`` has copied into a run, in its run
# directory (see the comment at ``campaign.COPIED_CONFIGS``).
COPIED_CONFIGS = "configs_copied.json"


def holds_copy(entry: dict[str, Any] | None, digest: str) -> bool:
    """Whether a record of a run with the SHA-256 ``digest`` is a copy that
    ``campaign.py --configs-from`` made, by the record's entry in the run's
    ``COPIED_CONFIGS`` (None: it has none): the copy the entry names
    (``sha256``), or the one that copy was to replace (``replaces``), which
    the run still holds when the campaign died between writing the entry
    and making the copy."""
    return entry is not None and digest in (entry["sha256"], entry.get("replaces"))


def record_seed(record: dict[str, Any]) -> int | None:
    """The random seed that the search of an autotune record was given
    (``SEED_VARIABLE`` in the record's ``env``); None when the record names
    none, or a blank one, which Helion takes for none: a search that is
    given no seed draws one from the clock."""
    seed = (record.get("env") or {}).get(SEED_VARIABLE)
    return int(seed) if seed is not None and seed.strip() else None


def standing_seed(
    run_dir: Path, repro_id: str, shape: str, backend: str
) -> tuple[int | None, str | None]:
    """The seed of the search whose autotune record stands in the run
    ``run_dir`` for one port; or None, with why the run names none.

    The record that stands is the ok one in the run's ``configs``, whichever
    attempt or run made it: the first clean attempt, the cleanest one kept
    after contaminated attempts (the campaign keeps that attempt's record as
    it is, where the job's last ``job_start`` event names another seed), or
    a record ``plan_run.py`` reused from an earlier run. Each carries the
    seed of its own search in ``env`` (``record_seed``). A provisional
    record stands in for an autotune that has not run, so it is no search of
    this run to repeat, and a record that names no seed is of a search that
    was given none (``autotune.py`` run by hand)."""
    path = config_record_path(run_dir / "configs", repro_id, shape, backend)
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None, "no autotune record"
    if record.get("status") != "ok":
        return None, "its autotune record is not ok"
    if record.get("provisional"):
        return None, "its autotune record is provisional: not a search of that run"
    seed = record_seed(record)
    if seed is None:
        return None, "its autotune record names no seed"
    return seed, None


def load_autotuned_port(
    configs_dir: Path, repro_id: str, shape: str, backend: str
) -> tuple[types.ModuleType, dict[str, Any]]:
    """The port with every kernel pinned to the decorator its autotune
    record (``autotune_record``) holds, and the record."""
    record = autotune_record(configs_dir, repro_id, shape, backend)
    pinned = per_kernel(record["kernels"], "decorator")
    return load_port(repro_id, backend, pinned=pinned), record


def per_kernel(kernels: dict[str, dict[str, Any]], field: str) -> dict[str, Any]:
    """``field`` of each kernel of an autotune record (its ``kernels``), by
    the kernel's name: ``decorator``, what the kernel is pinned to, or
    ``code_sha256``, the digest of the code it then generates."""
    return {name: kernel[field] for name, kernel in kernels.items()}


def timed_record_fields(tuned: dict[str, Any]) -> dict[str, Any]:
    """The fields with which a bench record names the autotune record
    ``tuned`` that a port's timing pinned: its kernels' code digests and
    whether it was provisional or reused.  An autotune that runs after the
    bench replaces the record, and ``report.py`` counts the timing only while
    the record holds the same kernels."""
    return {
        "autotune_code_sha256": per_kernel(tuned["kernels"], "code_sha256"),
        "autotune_provisional": bool(tuned.get("provisional")),
        "autotune_reused_from": tuned.get("reused_from"),
    }


def timed_kernels_held(info: dict[str, Any], tuned: dict[str, Any] | None) -> bool:
    """Whether the autotune record ``tuned`` (None: the point has no ok one)
    holds the kernels that a port's entry in a bench record, ``info``, was
    timed on: by their code digests (``timed_record_fields``), or, for an
    entry made before ``bench.py`` recorded those, by their decorators."""
    kernels = (tuned or {}).get("kernels", {})
    if "autotune_code_sha256" in info:
        return info["autotune_code_sha256"] == per_kernel(kernels, "code_sha256")
    return info.get("decorators") == per_kernel(kernels, "decorator")


def timed_on_record(info: dict[str, Any], tuned: dict[str, Any] | None) -> bool:
    """Whether a port's entry in a bench record, ``info``, is of the autotune
    record ``tuned`` (None: the point has no ok one), so that a bench of the
    point now would pin what that one pinned: the same kernels
    (``timed_kernels_held``), provisional or not as ``tuned`` is (an entry
    made before ``bench.py`` recorded that is told by its kernels alone).
    ``bench.py`` names the record before it loads the port, so an entry that
    names none found no ok record, and is of ``tuned`` only when there is
    still none."""
    named = "autotune_code_sha256" in info or "decorators" in info
    if tuned is None or not named:
        return tuned is None and not named
    return timed_kernels_held(info, tuned) and (
        "autotune_provisional" not in info
        or info["autotune_provisional"] == bool(tuned.get("provisional"))
    )


def port_kernels(module: types.ModuleType) -> dict[str, Any]:
    """The module-level Helion kernels of a port, by global name."""
    import helion

    return {
        name: value
        for name, value in vars(module).items()
        if isinstance(value, helion.Kernel)
    }


def _bound_kernel(name: str, kernel: Any) -> Any:
    """The single configured bound kernel of ``kernel`` (a port calls each
    kernel with one signature)."""
    bounds = [b for b in kernel._bound_kernels.values() if b._config is not None]
    if not bounds:
        raise RuntimeError(f"kernel {name} was never called by forward()")
    if len({b._config.to_json() for b in bounds}) > 1:
        raise RuntimeError(
            f"kernel {name} was bound with {len(bounds)} signatures and "
            "different configs; give each call site its own kernel"
        )
    return bounds[0]


def generated_code(bound: Any) -> str:
    """The code a configured bound kernel compiled and ran: the module it
    loaded, rendered again with ``to_code`` only when none is cached."""
    path = bound.get_cached_path(bound._config)
    return Path(path).read_text() if path else bound.to_code(bound._config)


def code_digests(module: types.ModuleType) -> dict[str, str]:
    """``{kernel: sha256 of its generated code}`` for a port's kernels."""
    out = {}
    for name, kernel in port_kernels(module).items():
        code = generated_code(_bound_kernel(name, kernel))
        out[name] = hashlib.sha256(code.encode()).hexdigest()
    return out


def bound_kernel_records(module: types.ModuleType) -> dict[str, dict[str, Any]]:
    """The tuned decorator, config and generated-code digest of every kernel
    a port's ``forward`` bound."""
    digests = code_digests(module)
    records: dict[str, dict[str, Any]] = {}
    for name, kernel in port_kernels(module).items():
        bound = _bound_kernel(name, kernel)
        records[name] = {
            "decorator": bound.format_kernel_decorator(bound._config, bound.settings),
            "config": bound._config.to_json(),
            "code_sha256": digests[name],
        }
    return records


def pin_roundtrip(
    module: types.ModuleType, records: dict[str, dict[str, Any]]
) -> dict[str, bool]:
    """Whether each kernel of a pinned port generates the recorded code."""
    digests = code_digests(module)
    return {name: digests.get(name) == r["code_sha256"] for name, r in records.items()}


def order_balanced(
    rounds: Sequence[float], statistic: Callable[[list[float]], float]
) -> tuple[float, list[float]]:
    """The mean of ``statistic`` over the rounds a timer ran in each order,
    and the two values, forward order first (one value for a single round).

    ``bench.time_impls`` runs the implementations in forward order in even
    rounds and in reverse order in odd ones. A reading depends on what ran
    right before it, by up to a microsecond for a small kernel, so a
    statistic over all rounds leans to the order with more of them: the
    median of 5 cold rounds (forward, reverse, forward, reverse, forward)
    was a forward-order reading."""
    if not rounds:
        raise ValueError("order_balanced needs at least one round")
    per_order = [
        statistic(list(rounds[start::2])) for start in (0, 1) if rounds[start::2]
    ]
    return sum(per_order) / len(per_order), per_order


def capture_graph(fn: Callable[[], Any]) -> tuple[Any, Any]:
    """A CUDA graph of ``fn()``, captured the way ``bench.py`` times it, and
    what ``fn`` returned during the capture (its outputs, which every replay
    rewrites). Warm ``fn`` up before capturing it.

    The garbage collector is off during the capture: a collection there can
    release CUDA state of earlier captures that dead reference cycles hold,
    which invalidates the capture (``cudaErrorStreamCaptureInvalidated``).
    ``bench.py`` hit that now and then when a dropout oracle's capture-only
    kernel compiled during the capture and its allocations set off a
    collection."""
    import torch

    from helion.runtime import cute_cuda_graph

    stream = torch.cuda.current_stream()
    try:
        with gc_paused(), cute_cuda_graph() as graph:
            result = fn()
    except Exception:
        # A failed capture can leave the capture stream current.
        torch.cuda.set_stream(stream)
        raise
    return graph, result


@contextlib.contextmanager
def gc_paused() -> Iterator[None]:
    """No garbage collection while the block runs (see ``capture_graph``)."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


# What the checks fill free device memory with before a call they judge
# (``poison_free_memory``). ``torch.empty`` hands a call the block an earlier
# call just freed, with what that call wrote there: the outputs of the
# previous implementation of the same repro, which have the same sizes and
# are correct. A kernel that skips part of its output then returns the right
# values in that part, and every check passes it (a ConvBert port variant
# that never wrote a ninth of its output passed the gate with error 0.0).
# Bytes of 0xFF read as a NaN in the float dtypes and as -1 in the signed
# integer ones (the two fp8 "fnuz" formats are the exception: their only NaN
# is 0x80, and 0xFF is a finite value, -240.0 in float8_e4m3fnuz and
# -57344.0 in float8_e5m2fnuz; no output of the ported repros has those
# dtypes). An output that holds such elements after the call is run
# again on memory filled with the second byte: an element that nothing wrote
# holds the first byte, then the second (``check_written``).
POISON_BYTE = 0xFF
POISON_RECHECK_BYTE = 0x00


class UnwrittenOutput(AssertionError):
    """Part of an output is memory the implementation never wrote: never
    acceptable, whatever values that memory happens to hold."""


class UnfilledMemory(RuntimeError):
    """A call took device memory from the driver in every one of its tries
    (:func:`call_on_poisoned_memory`), so its outputs may lie in memory that
    the fill never reached: what the call left unwritten there cannot be
    told from what it wrote, and the call cannot be judged."""


class _DeviceBytes:
    """``size`` bytes of device memory at ``address``, as ``torch.as_tensor``
    takes them (the CUDA array interface): a uint8 tensor over memory that it
    does not own."""

    def __init__(self, address: int, size: int) -> None:
        self.__cuda_array_interface__ = {
            "shape": (size,),
            "typestr": "|u1",
            "data": (address, False),
            "version": 2,
        }


def _fill_device_bytes(address: int, size: int, byte: int) -> None:
    import torch

    torch.as_tensor(_DeviceBytes(address, size), device="cuda").fill_(byte)


def _allocator_blocks(*states: str) -> dict[int, int]:
    """``{address: bytes}`` of the blocks in one of ``states`` that
    PyTorch's CUDA caching allocator holds on the current device, in every
    memory pool (a CUDA graph capture allocates in a pool of its own); none
    before the process has used CUDA. A block is ``active_allocated`` while
    a tensor holds it, ``inactive`` when it is free, and
    ``active_pending_free`` in between: freed while another stream still
    used it (``Tensor.record_stream``), it stays with the allocator until
    that stream's work is done and the next allocation notices."""
    import torch

    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        return {}
    device = torch.cuda.current_device()
    return {
        block["address"]: block["size"]
        for segment in torch.cuda.memory_snapshot()
        if segment["device"] == device
        for block in segment["blocks"]
        if block["state"] in states
    }


def allocated_blocks() -> set[int]:
    """The addresses of the device memory blocks that live tensors hold now:
    what :func:`poison_new_outputs` takes to tell the memory a call
    allocated from the memory it was given."""
    return set(_allocator_blocks("active_allocated"))


def poison_free_memory(byte: int = POISON_BYTE) -> int:
    """Fill the device memory that PyTorch's allocator holds free with
    ``byte``, so that what the next call allocates there starts from it and
    not from what an earlier call left (``POISON_BYTE``); the bytes filled.

    The blocks are filled where they are, through their addresses, and none
    is allocated: the allocator hands out the same blocks as without the
    fill. That reaches the free blocks of a CUDA graph's private pool too,
    which no allocation outside the capture can take (intermediate buffers
    that the captured call freed; every replay writes them before it reads
    them). Memory that the allocator takes fresh from the driver afterwards
    is not covered: :func:`call_on_poisoned_memory` calls again when that
    happened, and ``bench.py`` fills the outputs of a captured call itself
    (:func:`poison_new_outputs`).

    A block whose free waits for work on another stream is filled as well
    (``active_pending_free``, see :func:`_allocator_blocks`): the next
    allocation can hand it out, and it takes nothing from the driver for
    it. The device is idle by then, so nothing reads or writes the block
    any more."""
    import torch

    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        return 0
    # First, so that the work the pending frees wait for is done.
    torch.cuda.synchronize()
    free = _allocator_blocks("inactive", "active_pending_free")
    for address, size in free.items():
        _fill_device_bytes(address, size, byte)
    if free:
        torch.cuda.synchronize()
    return sum(free.values())


def poison_new_outputs(outputs: Any, held: set[int], byte: int = POISON_BYTE) -> None:
    """Fill with ``byte`` the storage of every tensor of ``outputs`` that
    was allocated since ``held`` was taken (:func:`allocated_blocks`). For
    the outputs of a call captured in a CUDA graph, between the capture and
    the first replay: the capture allocated them and wrote nothing, so the
    replay finds them as the driver handed them out. An output in memory
    from before the call (a view of an input, the input that the repro
    writes into, a buffer of the module) is left as it is."""
    import torch

    new = _allocator_blocks("active_allocated")
    filled = set(held)
    for tensor in flatten_outputs(outputs):
        storage = tensor.untyped_storage()
        address = storage.data_ptr()
        if tensor.is_cuda and address in new and address not in filled:
            filled.add(address)
            _fill_device_bytes(address, storage.nbytes(), byte)
    torch.cuda.synchronize()


def _device_allocations() -> int:
    """How many times PyTorch's allocator has taken memory from the driver
    on the current device (0 before the process has used CUDA). The count
    tells a call that took fresh memory where the reserved bytes do not: an
    allocator that finds no room releases its cached blocks to the driver
    and asks again, as does a call that empties the cache itself (a CUDA
    graph capture does), and the bytes it holds afterwards can be those
    from before the call, or fewer."""
    import torch

    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return torch.cuda.memory_stats()["num_device_alloc"]
    return 0


def call_on_poisoned_memory(
    call: Callable[[list[Any]], Any],
    fresh_inputs: Callable[[], list[Any]],
    byte: int = POISON_BYTE,
    tries: int = 3,
) -> Any:
    """``call(fresh_inputs())`` on free memory filled with ``byte``
    (:func:`poison_free_memory`; the inputs are made first, so they are not
    part of it). A call during which the allocator took memory from the
    driver may hold outputs or intermediate buffers there, which the fill
    did not reach: its outputs are dropped and it runs again on new inputs,
    now on the blocks that it left free, up to ``tries`` calls in all. A
    call that takes memory from the driver every time (one that keeps what
    it allocates, say, or that empties the allocator's cache) raises
    :class:`UnfilledMemory`: fresh device memory reads as zeros on a B200,
    so an element left unwritten there would pass wherever the right value
    is zero."""
    for _ in range(tries):
        # The earlier call's outputs and inputs go before the fill: this
        # call starts from the allocator state of the one before it, with
        # the blocks that one took from the driver free and filled.
        outputs = inputs = None
        inputs = fresh_inputs()
        poison_free_memory(byte)
        taken = _device_allocations()
        outputs = call(inputs)
        if _device_allocations() == taken:
            return outputs
    raise UnfilledMemory(
        f"the call took fresh device memory in each of {tries} calls: its "
        "outputs may lie in memory that was not filled before it"
    )


def forward_on_poisoned_memory(
    port: Any, repro_id: str, shape: str, byte: int = POISON_BYTE
) -> Any:
    """The port's outputs on fresh inputs of the shape point, from a call
    that found the free device memory filled with ``byte``
    (:func:`call_on_poisoned_memory`): what it returns is what it wrote, not
    what an earlier call, of this config or another, left in the blocks that
    ``torch.empty`` recycles. What ``check_port.py`` and ``autotune.py``
    judge, and what :func:`check_written` has them run again."""
    import torch

    with torch.no_grad():
        outputs = call_on_poisoned_memory(
            port.forward, lambda: make_inputs(repro_id, shape), byte
        )
    torch.cuda.synchronize()
    return outputs


def _holds_byte(part: Any, byte: int) -> Any:
    """Which elements of ``part`` (not complex) consist of the byte ``byte``
    alone. Read through an integer view of the elements' own size, which no
    layout needs a copy for: a copy of a bool tensor would make each byte 0
    or 1."""
    import torch

    size = part.element_size()
    integer = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[size]
    pattern = int.from_bytes(bytes([byte]) * size, "little", signed=size > 1)
    return part.view(integer) == pattern


def _poison_left(tensor: Any, byte: int, also: tuple[Any, int] | None = None) -> int:
    """How many elements of ``tensor`` consist of the byte ``byte`` alone (a
    complex element: both of its parts); with ``also`` (another tensor of
    the same shape and dtype, and a byte), how many of those are that byte
    alone in the other tensor too. Counted slice by slice
    (``CHECK_SLICE_ELEMENTS``)."""
    import torch

    pairs = tensor.is_complex()
    other, other_byte = (None, 0) if also is None else also
    if pairs:
        tensor = torch.view_as_real(tensor)
        other = None if other is None else torch.view_as_real(other)
    count = 0
    for part in _slices(tensor.shape):
        found = _holds_byte(tensor[part], byte)
        if other is not None:
            found &= _holds_byte(other[part], other_byte)
        if pairs:
            found = found.all(dim=-1)
        count += int(found.sum())
    return count


def check_written(outputs: Any, rerun: Callable[[int], Any]) -> None:
    """Raise :class:`UnwrittenOutput` when an implementation left part of
    its outputs unwritten. ``outputs`` are those of a call on memory filled
    with ``POISON_BYTE`` (:func:`call_on_poisoned_memory`, or ``bench.py``'s
    captured replay); ``rerun(byte)`` returns those of the same call on
    memory filled with ``byte``.

    An output without an element of that byte alone was written in full.
    Otherwise the implementation runs once more, on ``POISON_RECHECK_BYTE``:
    an element that nothing writes holds the first byte in the first
    outputs and the second byte in the second, while a value that the
    implementation computes (an integer -1, say) cannot follow the fill.
    The comparison is of the outputs' bytes, so it holds for every dtype,
    for outputs whose values no reference fixes (random numbers) and for
    values within any tolerance of the right ones (a zero).

    Its limit: the fill is of the memory that is free before the call. An
    output allocated in a block that the same call freed earlier (an
    intermediate buffer of its own) starts from what the call wrote there,
    and an element left unwritten in it follows neither fill. It shows only
    as a wrong value, so it passes if the intermediate held the output's
    right values there (in ``bench.py``'s replay too, which writes the
    intermediate again after the fill)."""
    first = flatten_outputs(outputs)
    suspects = [
        index for index, tensor in enumerate(first) if _poison_left(tensor, POISON_BYTE)
    ]
    if not suspects:
        return
    second = flatten_outputs(rerun(POISON_RECHECK_BYTE))
    unwritten = {}
    for index in suspects:
        a = first[index]
        b = second[index] if index < len(second) else None
        if b is None or a.shape != b.shape or a.dtype != b.dtype:
            raise StructureMismatch(
                f"output {index}: its shape or dtype differs between two calls"
            )
        if count := _poison_left(a, POISON_BYTE, (b, POISON_RECHECK_BYTE)):
            unwritten[index] = count
    if unwritten:
        raise UnwrittenOutput(
            "; ".join(
                f"output {index}: {count} of {first[index].numel()} elements were "
                "never written (they hold whatever the memory held before the call)"
                for index, count in unwritten.items()
            )
        )


# Outputs can have billions of elements (sum_b2873b4c9052's has 2**31, 16 GiB
# as float64), so the numerics checks below compare them in slices of at most
# this many elements: their temporaries, float64 conversions included, then
# take a few times 128 MiB whatever the output's size.
CHECK_SLICE_ELEMENTS = 1 << 24


def _slices(shape: Sequence[int]) -> Iterator[tuple[int | slice, ...]]:
    """Indices that split a tensor of ``shape`` into views of at most
    ``CHECK_SLICE_ELEMENTS`` elements, in row-major order: runs of whole rows
    of the first dim, and a row longer than that split along its own dims."""
    if not shape:
        yield ()
        return
    row = math.prod(shape[1:])
    if row <= CHECK_SLICE_ELEMENTS:
        step = max(CHECK_SLICE_ELEMENTS // max(row, 1), 1)
        for start in range(0, shape[0], step):
            yield (slice(start, start + step),)
        return
    for index in range(shape[0]):
        for rest in _slices(shape[1:]):
            yield (index, *rest)


def _nan_max(a: float, b: float) -> float:
    """``max`` that keeps a NaN, as ``Tensor.max`` does."""
    return a if math.isnan(a) else b if math.isnan(b) else max(a, b)


def _count_mismatches(a: Any, e: Any) -> int:
    """Elements where ``a != e``, counted slice by slice."""
    return sum(int((a[part] != e[part]).sum()) for part in _slices(e.shape))


def _slice_errors(a: Any, e: Any) -> tuple[float, float, bool, bool, bool]:
    """One slice of :func:`output_errors`: the max abs error and the max of
    ``|e|`` over ``e``'s finite elements, whether there are any, whether the
    NaN patterns differ, and whether an infinite element of ``e`` is not the
    same infinity in ``a``. The float64 temporaries are copies, updated in
    place, so a float64 output itself is left alone."""
    import torch

    ref = e.to(torch.float64, copy=True)
    nan_mismatch = bool((torch.isnan(a) != torch.isnan(ref)).any())
    inf_mismatch = bool((torch.isinf(ref) & (a != ref)).any())
    nonfinite = ~torch.isfinite(ref)
    if bool(nonfinite.all()):
        return 0.0, 0.0, False, nan_mismatch, inf_mismatch
    max_abs = float(
        a.to(torch.float64, copy=True)
        .sub_(ref)
        .abs_()
        .masked_fill_(nonfinite, 0.0)
        .max()
    )
    absmax = float(ref.abs_().masked_fill_(nonfinite, 0.0).max())
    return max_abs, absmax, True, nan_mismatch, inf_mismatch


def _stride_mismatches(a: Any, e: Any) -> list[int]:
    """The dims of ``e`` longer than one along which ``a`` (of ``e``'s shape)
    has another stride; the other dims' strides carry no meaning, as
    ``torch._prims_common.check_significant_strides`` has it, and outputs
    with at most one element have none."""
    if e.numel() <= 1:
        return []
    return [d for d in range(e.dim()) if e.shape[d] > 1 and a.stride(d) != e.stride(d)]


def output_errors(actual: Any, expected: Any) -> list[dict[str, Any]]:
    """Per-output max abs / rel error of ``actual`` against ``expected``,
    computed slice by slice (``CHECK_SLICE_ELEMENTS``); a complex output's
    over its real and imaginary parts.  Raises :class:`StructureMismatch`
    when the output counts differ, as :func:`check_outputs` does."""
    import torch

    a_list = flatten_outputs(actual)
    e_list = flatten_outputs(expected)
    if len(a_list) != len(e_list):
        raise StructureMismatch(f"{len(a_list)} outputs, expected {len(e_list)}")
    rows = []
    for index, (a, e) in enumerate(zip(a_list, e_list, strict=True)):
        shape_match = list(a.shape) == list(e.shape)
        row: dict[str, Any] = {
            "idx": index,
            "shape": list(e.shape),
            "dtype": str(e.dtype).replace("torch.", ""),
            "shape_match": shape_match,
            "dtype_match": a.dtype == e.dtype,
            "stride_match": shape_match and not _stride_mismatches(a, e),
        }
        if shape_match and e.numel():
            if e.is_complex() and a.is_complex():
                # A float64 copy of a complex tensor drops its imaginary part.
                a, e = torch.view_as_real(a), torch.view_as_real(e)
            if e.is_floating_point() or e.is_complex():
                max_abs = absmax = 0.0
                any_finite = nan_mismatch = inf_mismatch = False
                for part in _slices(e.shape):
                    part_abs, part_absmax, part_finite, part_nan, part_inf = (
                        _slice_errors(a[part], e[part])
                    )
                    if part_finite:
                        any_finite = True
                        max_abs = _nan_max(max_abs, part_abs)
                        absmax = max(absmax, part_absmax)
                    nan_mismatch = nan_mismatch or part_nan
                    inf_mismatch = inf_mismatch or part_inf
                if any_finite:
                    row["max_abs"] = max_abs
                    # Relative to the output's magnitude, not elementwise:
                    # near-zero elements make elementwise ratios meaningless.
                    row["ref_absmax"] = absmax
                    row["max_rel"] = max_abs / max(absmax, 1e-30)
                else:
                    row["max_abs"] = row["max_rel"] = 0.0
                row["nan_mismatch"] = nan_mismatch
                row["inf_mismatch"] = inf_mismatch
            else:
                row["mismatches"] = _count_mismatches(a, e)
        rows.append(row)
    return rows


class StructureMismatch(AssertionError):
    """Wrong output count, shape, dtype or strides: never acceptable."""


class ExactMismatch(AssertionError):
    """An integer or boolean output differs: never acceptable."""


class FloatMismatch(AssertionError):
    """A float output is outside the loose eager tolerance; the anchored gate
    decides whether that is drift (Inductor-style fp32 intermediates vs
    eager's per-op rounding) or an error."""


def check_outputs(
    actual: Any,
    expected: Any,
    *,
    rtol: float = 2e-2,
    atol: float = 2e-2,
    stochastic: set[int] | frozenset[int] = frozenset(),
    check_strides: bool = True,
    inductor: Callable[[], Any] | None = None,
    vs_inductor: list[dict[str, Any]] | None = None,
) -> None:
    """Compare against eager; raises one of the mismatch classes above.
    Every output needs eager's shape and dtype, and with ``check_strides``
    its strides along the dims longer than one (the others carry no meaning,
    as ``torch._prims_common.check_significant_strides`` has it; outputs with
    at most one element are free).
    Inductor may pad an output's strides (comprehensive padding), so its
    baseline is compared without them; :func:`output_errors` records the
    mismatch. Outputs in ``stochastic`` (random number dependent: eager draws
    them from its global generator) are not compared by value here;
    :class:`AnchoredGate` compares them with Inductor's.

    An integer or boolean output that differs from eager's is compared with
    the default Inductor compile's when that compile's own output differs
    from eager's too (:func:`_inductor_reference`): eager leaves such a value
    to its implementation (the order of ``topk``'s tied values, say) and
    Inductor's kernels, which the ports mirror, settle it another way.
    ``inductor`` returns that compile's outputs on the same inputs and is
    called only for such an output; ``vs_inductor`` gets an entry for each
    output that passed this way. Without ``inductor``, and for every output
    Inductor reproduces, eager is the only reference."""
    a_list = flatten_outputs(actual)
    e_list = flatten_outputs(expected)
    if len(a_list) != len(e_list):
        raise StructureMismatch(f"{len(a_list)} outputs, expected {len(e_list)}")
    float_errors = []
    for index, (a, e) in enumerate(zip(a_list, e_list, strict=True)):
        if a.shape != e.shape:
            raise StructureMismatch(
                f"output {index}: shape {list(a.shape)} != {list(e.shape)}"
            )
        if a.dtype != e.dtype:
            raise StructureMismatch(f"output {index}: dtype {a.dtype} != {e.dtype}")
        if check_strides and (dims := _stride_mismatches(a, e)):
            raise StructureMismatch(
                f"output {index}: strides {a.stride()} != {e.stride()} (dims {dims})"
            )
        if index in stochastic:
            continue
        if e.is_floating_point() or e.is_complex():
            if (error := _not_close(a, e, rtol=rtol, atol=atol)) is not None:
                float_errors.append(f"output {index}: {error}")
        elif mismatches := _count_mismatches(a, e):
            message = f"output {index}: {mismatches} of {e.numel()} elements differ"
            if inductor is None:
                raise ExactMismatch(message)
            baseline = flatten_outputs(inductor())
            # Only an output of eager's structure can stand in for eager's.
            if len(baseline) != len(e_list) or baseline[index].shape != e.shape:
                raise ExactMismatch(message)
            inductor_mismatches = _inductor_reference(a, e, baseline[index], message)
            if vs_inductor is not None:
                vs_inductor.append(
                    {
                        "idx": index,
                        "reference": "inductor",
                        "elements": e.numel(),
                        "mismatches_vs_eager": mismatches,
                        "inductor_mismatches_vs_eager": inductor_mismatches,
                    }
                )
    if float_errors:
        raise FloatMismatch("\n".join(float_errors))


def _inductor_reference(a: Any, e: Any, baseline: Any, message: str) -> int:
    """Judge an integer or boolean output ``a`` that differs from eager's
    ``e`` (``message`` says in how many elements) against the default
    Inductor compile's ``baseline``. Inductor's output is the reference only
    when it differs from eager's itself, and then ``a`` must equal it element
    for element: a port that does what Inductor's kernels do reproduces
    their output, so any other difference stays an :class:`ExactMismatch`.
    Returns the number of elements in which Inductor's output differs from
    eager's, which is why it is the reference."""
    inductor_mismatches = _count_mismatches(baseline, e)
    if not inductor_mismatches:
        # Inductor reproduces eager, so every implementation has to.
        raise ExactMismatch(message)
    if differing := _count_mismatches(a, baseline):
        raise ExactMismatch(
            f"{message} from eager's and {differing} from the default Inductor "
            f"compile's, which differs from eager's in {inductor_mismatches}"
        )
    return inductor_mismatches


def _slice_not_close(
    a: Any, e: Any, *, rtol: float, atol: float
) -> tuple[int, float, float] | None:
    """One slice of :func:`_not_close`: how many elements are outside the
    tolerance and their greatest absolute and relative differences, or None
    when every element is within it."""
    import torch

    matches = torch.isclose(a, e, rtol=rtol, atol=atol, equal_nan=True)
    if bool(matches.all()):
        return None
    wide = torch.promote_types(e.dtype, torch.float64)
    abs_diff = (a.to(wide) - e.to(wide)).abs().masked_fill_(matches, 0.0)
    rel_diff = (abs_diff / e.to(wide).abs()).masked_fill_(matches, 0.0)
    return int((~matches).sum()), float(abs_diff.max()), float(rel_diff.max())


def _not_close(a: Any, e: Any, *, rtol: float, atol: float) -> str | None:
    """Why two float tensors of one shape and dtype are not close, or None:
    ``torch.testing.assert_close``'s verdict (``torch.isclose`` with
    ``equal_nan``), taken slice by slice. An output of one slice gets
    ``assert_close``'s message, a larger one the same statistics over the
    whole output, without the elements' indices."""
    import torch

    parts = list(_slices(e.shape))
    if len(parts) <= 1:
        try:
            torch.testing.assert_close(
                a, e, rtol=rtol, atol=atol, equal_nan=True, check_dtype=False
            )
        except AssertionError as error:
            return str(error)
        return None
    mismatches = 0
    greatest_abs = greatest_rel = 0.0
    for part in parts:
        found = _slice_not_close(a[part], e[part], rtol=rtol, atol=atol)
        if found is not None:
            mismatches += found[0]
            greatest_abs = _nan_max(greatest_abs, found[1])
            greatest_rel = _nan_max(greatest_rel, found[2])
    if not mismatches:
        return None
    return (
        "Tensor-likes are not close!\n\n"
        f"Mismatched elements: {mismatches} / {e.numel()} "
        f"({mismatches / e.numel():.1%})\n"
        f"Greatest absolute difference: {greatest_abs} (up to {atol} allowed)\n"
        f"Greatest relative difference: {greatest_rel} (up to {rtol} allowed)"
    )


def stochastic_outputs(repro_id: str, shape: str, instance: Any) -> set[int]:
    """Indices of the repro's outputs that depend on random numbers."""
    bb_path()
    import oracle_harness  # pyrefly: ignore [missing-import]

    return set(
        oracle_harness.detect_stochastic_outputs(
            instance, make_inputs(repro_id, shape, seed=0)
        )
    )


def inductor_outputs(repro_id: str, shape: str, instance: Any) -> Callable[[], Any]:
    """The default Inductor compile's outputs on the repro's inputs, as
    :func:`check_outputs` takes them (``inductor``): compiled and run by the
    first call and kept for the later ones, so a job that compiles no
    Inductor code otherwise (``autotune.py``) compiles it only for a port
    with an integer or boolean output that differs from eager's."""
    import torch

    @functools.cache
    def outputs() -> Any:
        compiled = compile_inductor(repro_id, shape, instance)
        with torch.no_grad():
            return compiled(*make_inputs(repro_id, shape))

    return outputs


def ulp_of(dtype: Any, magnitude: Any) -> Any:
    """Spacing of ``dtype``'s representable values at ``magnitude`` (a
    float64 tensor): one unit in the last place, per element."""
    import torch

    info = torch.finfo(dtype)
    # In place on one copy: outputs can have billions of elements.
    return magnitude.clamp_min(info.tiny).log2_().floor_().exp2_().mul_(info.eps)


def _peak(values: Any) -> float:
    return float(values.max()) if values.numel() else 0.0


def _leaf_first_pass(
    candidate: Any, baseline: Any, reference: Any
) -> tuple[int, bool, float, float]:
    """One slice of :func:`_anchored_leaf`'s first pass: the elements where
    the baseline is not finite but the reference is, whether the candidate's
    NaN pattern differs from the reference's, and the baseline error and
    ``|ref|`` maxima over the reference's finite elements."""
    import torch

    ref = reference.double()
    finite = torch.isfinite(ref)
    baseline_nonfinite = int((finite & ~torch.isfinite(baseline)).sum())
    nan_differs = bool((torch.isnan(candidate) != torch.isnan(ref)).any())
    err_base = _peak((baseline.double() - ref).abs_().where(finite, 0.0))
    absmax = _peak(ref.abs().where(finite, 0.0))
    return baseline_nonfinite, nan_differs, err_base, absmax


def _leaf_second_pass(
    candidate: Any, reference: Any, threshold: float
) -> tuple[float, int, bool]:
    """One slice of :func:`_anchored_leaf`'s second pass: the candidate's
    error maximum, the elements past the threshold but within one unit in the
    last place, and whether every element passes."""
    import torch

    ref = reference.double()
    finite = torch.isfinite(ref)
    err = (candidate.double() - ref).abs_().where(finite, 0.0)
    ok = err <= threshold
    one_ulp_flips = 0
    if candidate.dtype in (torch.bfloat16, torch.float16):
        within_ulp = err <= ulp_of(candidate.dtype, ref.abs())
        one_ulp_flips = int((~ok & within_ulp).sum())
        ok |= within_ulp
    # Infinities must match exactly.
    infinite = ~(finite | torch.isnan(ref))
    inf_ok = bool((candidate[infinite].double() == ref[infinite]).all())
    return _peak(err), one_ulp_flips, bool(ok.all()) and inf_ok


def _anchored_leaf(
    candidate: Any, baseline: Any, reference: Any
) -> tuple[dict[str, Any], bool]:
    """One float output of the gate: better-benchmark's threshold
    ``max(3 * baseline_error, 1e-5 * max(|ref|max, 1))``, loosened so that an
    element may also be off by one unit in the last place of the output's
    dtype at its own magnitude. Without that allowance the gate has no
    headroom whenever Inductor's output rounds exactly like the reference
    (baseline error 0, common since the reference keeps the graph's casts):
    a different but equally valid fp32 summation order flips the bf16
    rounding of elements near a rounding midpoint. A baseline that is not
    finite where the reference is fails the output for every candidate: its
    error would make the threshold infinite (every candidate passes) or NaN
    (none does).

    The outputs are compared slice by slice (``CHECK_SLICE_ELEMENTS``), in
    two passes: the threshold needs the whole output's baseline error."""
    parts = list(_slices(reference.shape))
    baseline_nonfinite = 0
    nan_differs = False
    err_base = absmax = 0.0
    for part in parts:
        part_nonfinite, part_nan, part_err, part_absmax = _leaf_first_pass(
            candidate[part], baseline[part], reference[part]
        )
        baseline_nonfinite += part_nonfinite
        nan_differs = nan_differs or part_nan
        err_base = max(err_base, part_err)
        absmax = max(absmax, part_absmax)
    if baseline_nonfinite:
        return {
            "pass": False,
            "error": "Inductor's output is not finite where the reference is",
            "baseline_nonfinite": baseline_nonfinite,
        }, False
    if nan_differs:
        return {"pass": False, "error": "NaN pattern differs from the reference"}, False
    threshold = max(3.0 * err_base, 1e-5 * max(absmax, 1.0))
    err_candidate = 0.0
    one_ulp_flips = 0
    passed = True
    for part in parts:
        part_err, part_flips, part_passed = _leaf_second_pass(
            candidate[part], reference[part], threshold
        )
        err_candidate = max(err_candidate, part_err)
        one_ulp_flips += part_flips
        passed = passed and part_passed
    fields = {
        "err_candidate": err_candidate,
        "err_baseline": err_base,
        "threshold": threshold,
        "ref_absmax": absmax,
        "one_ulp_flips": one_ulp_flips,
        "pass": passed,
    }
    return fields, passed


def _stochastic_leaf(candidate: Any, baseline: Any) -> tuple[dict[str, Any], bool]:
    """One random-number dependent output against Inductor's: a mask (or any
    integer output) must match exactly, a float output must pass the gate
    with Inductor's output as its reference (zero baseline error: within
    ``1e-5 * max(|ref|max, 1)`` or one unit in the last place). A mask's
    record also counts the elements only the candidate or only Inductor
    keeps."""
    import torch

    fields: dict[str, Any]
    if candidate.is_floating_point():
        fields, passed = _anchored_leaf(candidate, baseline, baseline)
    else:
        mismatches = _count_mismatches(candidate, baseline)
        passed = mismatches == 0
        fields = {"mismatches": mismatches, "pass": passed}
    if candidate.dtype == torch.bool:
        kept = only_candidate = only_inductor = 0
        for part in _slices(candidate.shape):
            mask, inductor_mask = candidate[part], baseline[part]
            kept += int(mask.count_nonzero())
            only_candidate += int((mask & ~inductor_mask).sum())
            only_inductor += int((inductor_mask & ~mask).sum())
        fields["kept"] = kept / candidate.numel() if candidate.numel() else 0.0
        fields["kept_only_by_candidate"] = only_candidate
        fields["kept_only_by_inductor"] = only_inductor
    return {"reference": "inductor", **fields}, passed


class AnchoredGate:
    """A loosened variant of better-benchmark's anchored oracle gate, with
    the references computed once so every candidate (each config of each
    implementation) can be judged: each float output's error against the
    reference must be within ``max(3 * inductor_error, 1e-5 * max(|ref|max,
    1))`` (better-benchmark's threshold), or within one unit in the last
    place of the output dtype at that element (the loosening,
    ``_anchored_leaf``).

    The reference is built the way better-benchmark builds its "fp64"
    reference: the eager module, its parameters and buffers converted to
    float64, run on float64 copies of the float inputs (``ref_precision``
    ``"f64_inputs"``). The graph's own dtype conversions stay, so the
    reference is float64 only up to the graph's first cast to a narrower
    dtype. Most graphs convert to fp32 right after their loads and back to
    the output dtype before their stores, so for them the reference is eager
    fp32 math rounded like the output; ``ref_dtype`` records the dtype each
    float reference output comes out in. When an op has no float64 kernel,
    the reference is the eager module on the original inputs
    (``ref_precision`` ``"eager"``); any other failure of the float64 run
    falls back the same way, except running out of memory, which raises, and
    ``ref_fallback`` in each verdict says why.

    better-benchmark skips the outputs that differ between two eager runs
    (``stochastic``). Those that repeat under the same seed depend on random
    numbers (``random``) and are compared with the default Inductor compile's
    instead (``_stochastic_leaf``): eager draws them from its global
    generator, while Inductor draws them from the repro's seed-table input
    with one Philox4x32-10 evaluation per element at its row-major flat
    index, keeping word 0 (``tl.rand``). The Triton backend's ``hl.rand`` and
    the CuTe backend's with ``cute_rng_stream="word0"`` draw the same numbers,
    so a port's masks match Inductor's bit for bit. The oracles of the
    dropout repros draw the same numbers when captured in a CUDA graph (how
    ``bench.py`` times and checks them) but, like eager, round each sample
    to bf16 before comparing it with bf16(0.1), where Inductor (without
    ``emulate_precision_casts``) and the ports compare the fp32 sample: their
    masks drop the samples just above the threshold, about 2**-12 of the
    elements, that Inductor keeps. ``check(..., gate_random=False)`` records
    such an implementation's random outputs against Inductor's without
    letting them decide the verdict. The other stochastic outputs have a
    nondeterministic summation order (atomic accumulation) and are judged
    against the reference like the deterministic outputs.

    The baseline's call and every candidate's start from free memory filled
    with ``POISON_BYTE`` (``poison_free_memory``), so a float element that a
    call leaves unwritten is a NaN (a finite value in the fp8 "fnuz"
    formats) and not what the candidate judged before it left in the block.
    The gate judges values; ``check_written`` tells an unwritten element
    whatever its right value is and in every dtype, and the checks run it
    beside the gate."""

    def __init__(
        self, repro_id: str, shape: str, instance: Any, compiled: Callable[..., Any]
    ) -> None:
        import torch

        bb_path()
        import oracle_harness  # pyrefly: ignore [missing-import]

        self._repro_id = repro_id
        self._shape = shape
        # Outputs that differ between two eager runs: those that depend on
        # random numbers, and those with a nondeterministic summation order
        # (atomic accumulation). Only the former repeat under the same seed.
        self.stochastic: set[int] = set(
            oracle_harness.detect_stochastic_outputs(instance, self.inputs())
        )
        self.random: set[int] = set()
        if self.stochastic:
            runs = []
            for _ in range(2):
                # Inputs first: generating them reseeds (make_inputs).
                inputs = self.inputs()
                seed_everything(1)
                with torch.no_grad():
                    outputs = flatten_outputs(instance(*inputs))
                runs.append({index: outputs[index] for index in self.stochastic})
            self.random = {
                index
                for index in self.stochastic
                if bool(
                    (
                        (runs[0][index] == runs[1][index])
                        | (runs[0][index].isnan() & runs[1][index].isnan())
                    ).all()
                )
            }
        self.ref_precision = "f64_inputs"
        self.ref_fallback: str | None = None
        try:
            self.reference = self._fp64_reference(instance)
        except torch.OutOfMemoryError:
            # Not a missing float64 kernel: a fallback here would make the
            # reference depend on the size of the outputs.
            raise
        except Exception as error:
            # Ops without float64 kernels: better-benchmark falls back to the
            # eager module on the original inputs (its "f32"). Other failures
            # fall back too; the verdicts keep the reason.
            self.ref_fallback = f"{type(error).__name__}: {error}"[:500]
        if self.ref_fallback is not None:
            # Out of the handler, whose traceback holds the float64 run's
            # frames and with them its float64 copies.
            self.ref_precision = "eager"
            inputs = self.inputs()
            seed_everything(1)
            with torch.no_grad():
                self.reference = flatten_outputs(instance(*inputs))
        inputs = self.inputs()
        seed_everything(1)
        # The baseline's outputs must be its own too (``POISON_BYTE``): a
        # NaN that it leaves where the reference is finite fails the output.
        poison_free_memory()
        with torch.no_grad():
            self.baseline = flatten_outputs(compiled(*inputs))

    def _fp64_reference(self, instance: Any) -> list[Any]:
        """The ``f64_inputs`` reference outputs. The float64 module and inputs
        (a float64 copy of every float input) are freed on return, before the
        baseline runs."""
        import torch

        reference = copy.deepcopy(instance).double()
        fp64_inputs = [
            x.double() if isinstance(x, torch.Tensor) and x.is_floating_point() else x
            for x in self.inputs()
        ]
        seed_everything(1)
        with torch.no_grad():
            return flatten_outputs(reference(*fp64_inputs))

    def inputs(self) -> list[Any]:
        return make_inputs(self._repro_id, self._shape, seed=0)

    def check(
        self, candidate: Callable[[list[Any]], Any], *, gate_random: bool = True
    ) -> dict[str, Any]:
        import torch

        inputs = self.inputs()
        seed_everything(1)
        # The candidate's outputs start from poisoned memory, not from the
        # block an earlier candidate freed with its outputs still in it
        # (``POISON_BYTE``): float elements that the candidate leaves
        # unwritten are NaN, which fails them unless the reference is NaN
        # there too (``check_written`` finds those as well, and the fp8
        # "fnuz" elements, where the fill is a finite value).
        poison_free_memory()
        with torch.no_grad():
            outputs = flatten_outputs(candidate(inputs))
        if len(outputs) != len(self.reference):
            return {"pass": False, "error": "output count differs from the reference"}
        result: dict[str, Any] = {
            "pass": True,
            "per_output": [],
            "ref_precision": self.ref_precision,
        }
        if self.ref_fallback is not None:
            result["ref_fallback"] = self.ref_fallback
        for index, (cand, base, ref) in enumerate(
            zip(outputs, self.baseline, self.reference, strict=True)
        ):
            record: dict[str, Any] = {"idx": index}
            if index in self.random:
                fields, passed = _stochastic_leaf(cand, base)
                record.update(fields)
                if gate_random:
                    result["pass"] = result["pass"] and passed
                else:
                    record["gated"] = False
            elif not (ref.is_floating_point() or ref.is_complex()):
                record["skip"] = "non_float"
            else:
                record["ref_dtype"] = str(ref.dtype).replace("torch.", "")
                if ref.is_complex():
                    # The real and imaginary parts, as float outputs.
                    cand, base, ref = map(torch.view_as_real, (cand, base, ref))
                fields, passed = _anchored_leaf(cand, base, ref)
                record.update(fields)
                result["pass"] = result["pass"] and passed
            result["per_output"].append(record)
        return result


def _nvidia_smi(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _float_or_none(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def gpu_info() -> dict[str, Any]:
    """Identity and state of this process's first visible GPU (informational;
    fields nvidia-smi cannot report are None)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    fields = (
        "index",
        "uuid",
        "name",
        "power.limit",
        "driver_version",
        "temperature.gpu",
    )
    out = _nvidia_smi(
        "-i",
        visible,
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    )
    rows = list(csv.reader((out or "").splitlines(), skipinitialspace=True))
    if not rows or len(rows[0]) != len(fields):
        return {"error": f"nvidia-smi gave {out!r}"}
    index, uuid, name, power, driver, temp = rows[0]
    return {
        "physical_index": index,
        "uuid": uuid,
        "name": name,
        "power_limit_w": _float_or_none(power),
        "driver": driver,
        "temperature_c": _float_or_none(temp),
    }


def gpu_temperature() -> float | None:
    return gpu_info().get("temperature_c")


def _process_owner(pid: int) -> str:
    try:
        return pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except (OSError, KeyError):
        return "?"


def proc_stats(uid: int | None = None) -> Iterator[tuple[int, list[str]]]:
    """``(pid, fields)`` of each process under /proc, of the account ``uid``
    only when given: the fields of its ``stat`` after the command name, its
    state first, then its parent pid and its process group.  A process that
    exits meanwhile is left out."""
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            if uid is not None and entry.stat().st_uid != uid:
                continue
            stat = Path(entry.path, "stat").read_text()
        except OSError:
            continue
        # The command name, in parentheses, may hold spaces and parentheses.
        yield int(entry.name), stat.rsplit(")", 1)[1].split()


def process_tree(*roots: int) -> set[int]:
    """``roots`` and their descendants, by the parent pids under /proc.  A
    job's process group does not hold all of its processes: the autotuner's
    benchmark worker and its precompile and render children start sessions
    of their own (``os.setsid``)."""
    children: dict[int, list[int]] = {}
    for pid, fields in proc_stats():
        children.setdefault(int(fields[1]), []).append(pid)
    tree = set(roots)
    stack = list(roots)
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in tree:
                tree.add(child)
                stack.append(child)
    return tree


# The environment variable whose value marks a campaign job's processes
# (``campaign.job_env``): they all inherit it, those whose parents have
# exited included.
JOB_MARKER = "BB_CAMPAIGN_JOB"


def marked_processes(marker: str) -> set[int]:
    """The processes of this account whose environment holds
    ``JOB_MARKER=marker`` (a zombie's environment reads as empty)."""
    wanted = f"{JOB_MARKER}={marker}".encode()
    found = set()
    for pid, _ in proc_stats(os.getuid()):
        try:
            environ = Path(f"/proc/{pid}/environ").read_bytes()
        except OSError:
            continue
        if wanted in environ.split(b"\0"):
            found.add(pid)
    return found


def own_processes() -> set[int]:
    """This job's processes: its tree (:func:`process_tree`) and, in a
    campaign, the trees of the processes that carry its marker
    (``JOB_MARKER``): a worker whose parent has exited is init's child, out
    of the tree, but still the job's own."""
    marker = os.environ.get(JOB_MARKER)
    return process_tree(os.getpid(), *(marked_processes(marker) if marker else ()))


def foreign_gpu_processes() -> list[str] | None:
    """``user:pid`` of the processes on this job's GPU that are not its own
    (:func:`own_processes`: the autotuner's workers are the job's own); None
    when nvidia-smi cannot tell."""
    uuid = gpu_info().get("uuid")
    out = _nvidia_smi("--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader")
    if uuid is None or out is None:
        return None
    own = own_processes()
    foreign = []
    for gpu, pid in csv.reader(out.splitlines(), skipinitialspace=True):
        if gpu != uuid or not pid.isdigit() or int(pid) in own:
            continue
        if not Path(f"/proc/{pid}").exists():
            # Exited since the query (the job's own short-lived workers do).
            continue
        foreign.append(f"{_process_owner(int(pid))}:{pid}")
    return foreign


# Seconds between the samples of watch_foreign_gpu_activity.
FOREIGN_ACTIVITY_INTERVAL_S = 1.0


def _pmon_sample(uuid: str) -> list[tuple[int, int]] | None:
    """``(pid, SM utilization in percent)`` of each process ``nvidia-smi
    pmon`` reports on the GPU (``-``, no activity, reads as 0), or None when
    nvidia-smi fails or prints something else."""
    out = _nvidia_smi("pmon", "-i", uuid, "-c", "1", "-s", "u")
    if out is None:
        return None
    columns: list[str] = []
    rows = []
    for line in out.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "#":
            # The column names, then their units.
            if not columns:
                columns = fields[1:]
            continue
        if (
            "pid" not in columns
            or "sm" not in columns
            or len(fields) <= max(columns.index("pid"), columns.index("sm"))
        ):
            # Something else, or a row cut short.
            return None
        pid, sm = fields[columns.index("pid")], fields[columns.index("sm")]
        if not pid.isdigit():
            # The row pmon prints for a GPU without processes.
            continue
        rows.append((int(pid), int(sm) if sm.isdigit() else 0))
    return rows


@contextlib.contextmanager
def watch_foreign_gpu_activity(record: dict[str, Any]) -> Iterator[None]:
    """Record in ``record["foreign_gpu_activity"]`` how much processes other
    than this job's own (:func:`own_processes`) computed on its GPU while the
    block runs, from ``nvidia-smi pmon`` sampled every
    ``FOREIGN_ACTIVITY_INTERVAL_S``: for each ``user:pid``, the highest SM
    utilization (percent), the seconds of the samples that saw it above zero
    (``active_s``) and of those that saw it at ``CONTAMINATION_SM_PCT`` or
    more (``busy_s``). Unlike the processes :func:`foreign_gpu_processes`
    lists, this tells a foreign job that computes from a resident process
    that idles. A pid is classified when it first shows up, against the
    job's processes at that sample; one that has exited by then is left out,
    since nothing can tell whose it was (the job's own short-lived workers
    come and go).
    Samples that nvidia-smi fails to give, or that fail to be read, count in
    ``unknown_samples``; sampling never fails the job and never stops before
    it ends.
    ``watched_s`` is how long the block ran, and ``max_gap_s`` the longest
    stretch of it without a sample that read the GPU's processes (from the
    start to the first such sample, between two, and from the last to the
    end): :func:`activity_coverage_issue` checks that the samples cover the
    job.

    The sampling goes on through the job's timing, where foreign work spoils
    a record and only these samples see it. That costs one nvidia-smi query
    a second, for the job's own GPU; the campaign's scheduler, whose queries
    cover every GPU, queries only when it can start a job.
    ``BB_FOREIGN_ACTIVITY=0`` in the environment turns the sampling off, to
    measure what it costs the timings for instance; the record then holds
    None."""
    if os.environ.get("BB_FOREIGN_ACTIVITY") == "0":
        record["foreign_gpu_activity"] = None
        yield
        return
    uuid: str | None = None
    own: set[int] = set()
    foreign: dict[int, dict[str, Any]] = {}
    counts = {"samples": 0, "unknown_samples": 0}
    coverage = {"watched_s": 0.0, "max_gap_s": 0.0}
    stop = threading.Event()

    def sample(elapsed_s: float) -> bool:
        """Read the GPU's processes once; whether nvidia-smi told them."""
        nonlocal uuid
        if uuid is None:
            # Asked again until nvidia-smi answers: one failed query must not
            # leave every sample of the job unknown.
            uuid = gpu_info().get("uuid")
        rows = _pmon_sample(uuid) if uuid else None
        if rows is None:
            return False
        if any(pid not in own and pid not in foreign for pid, _ in rows):
            own.update(own_processes())
        for pid, sm in rows:
            if pid in own:
                continue
            if pid not in foreign:
                if not Path(f"/proc/{pid}").exists():
                    # Exited before the processes above were read: a worker
                    # of the job's own would pass for foreign and contaminate
                    # the record.
                    continue
                # The owner once, not at every sample: the lookup reads
                # /proc and the user database while the autotuner forks.
                foreign[pid] = {
                    "user": _process_owner(pid),
                    "max_sm": 0,
                    "active_s": 0.0,
                    "busy_s": 0.0,
                }
            entry = foreign[pid]
            entry["max_sm"] = max(entry["max_sm"], sm)
            if sm > 0:
                entry["active_s"] += elapsed_s
            if sm >= CONTAMINATION_SM_PCT:
                entry["busy_s"] += elapsed_s
        return True

    def poll() -> None:
        started = last = told = time.monotonic()

        def take() -> None:
            nonlocal last, told
            now = time.monotonic()
            counts["samples"] += 1
            try:
                ok = sample(now - last)
            except Exception:
                # Nothing a sample raises may end the sampling: a thread
                # that died would leave a record that reads clean.
                ok = False
            last = now
            if ok:
                done = time.monotonic()
                coverage["max_gap_s"] = max(coverage["max_gap_s"], done - told)
                told = done
            else:
                counts["unknown_samples"] += 1

        while not stop.wait(FOREIGN_ACTIVITY_INTERVAL_S):
            take()
        take()
        coverage["max_gap_s"] = max(coverage["max_gap_s"], time.monotonic() - told)
        coverage["watched_s"] = time.monotonic() - started

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        record["foreign_gpu_activity"] = {
            "interval_s": FOREIGN_ACTIVITY_INTERVAL_S,
            **counts,
            **{key: round(value, 1) for key, value in coverage.items()},
            "processes": {
                f"{entry['user']}:{pid}": {
                    "max_sm": entry["max_sm"],
                    "active_s": round(entry["active_s"], 1),
                    "busy_s": round(entry["busy_s"], 1),
                }
                for pid, entry in sorted(foreign.items())
            },
        }


def write_json(path: Path, record: dict[str, Any]) -> None:
    """Write a record whole: a reader never sees half of it."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, default=str) + "\n")
    tmp.replace(path)


# A bench record is contaminated when a process outside its job reached this
# SM utilization (percent) in one of the job's nvidia-smi pmon samples
# (watch_foreign_gpu_activity); idle resident processes read 0.
CONTAMINATION_SM_PCT = 5
# An autotune record only when such processes computed at that utilization
# (their ``busy_s``, summed) for this share of the job and these seconds at
# least: a blip moves a few of a long autotune's timings, and running hours
# of autotuning again over it costs more than it saves.  (Records made before
# ``busy_s`` keep the one-sample rule.)
AUTOTUNE_CONTAMINATION_SHARE = 0.01
AUTOTUNE_CONTAMINATION_MIN_S = 10.0


# A record's activity samples cover its job when at most this many of them
# failed (or one in twenty, if that is more) and no stretch of the job longer
# than ``MAX_ACTIVITY_GAP_S`` went without a sample that read the GPU.
MAX_UNKNOWN_SAMPLES = 2
MAX_ACTIVITY_GAP_S = 30.0
# A record made before the gaps were recorded: a sample every so many seconds
# at least over its ``wall_s`` (the interval plus an ``nvidia-smi pmon`` call,
# which takes up to about 5 s on a loaded machine).
MAX_SAMPLE_PERIOD_S = 6.0


def unknown_samples_allowed(activity: dict[str, Any]) -> int:
    """How many of a job's activity samples may have failed
    (``MAX_UNKNOWN_SAMPLES``, or one in twenty of them). The samples of
    several jobs merged into one, as an aggregate bench record holds them,
    name what their jobs allowed, added up (``unknown_allowed``): each
    process of a repeated bench has the allowance of a job."""
    allowed = activity.get("unknown_allowed")
    if allowed is None:
        allowed = max(MAX_UNKNOWN_SAMPLES, activity["samples"] // 20)
    return allowed


def activity_coverage_issue(record: dict[str, Any]) -> str | None:
    """Why ``record``'s foreign-activity samples do not cover its job (no
    samples, too many that failed, or a stretch of the job without one), or
    None when they do. A job's samples cover all of it only while the
    sampling thread runs: one that died, or nvidia-smi calls that kept
    failing, would otherwise leave a record that reads clean."""
    activity = record.get("foreign_gpu_activity")
    if not activity:
        return "no activity samples"
    samples, unknown = activity["samples"], activity["unknown_samples"]
    if unknown >= samples:
        return f"none of {samples} samples read the GPU"
    if unknown > unknown_samples_allowed(activity):
        return f"{unknown} of {samples} samples failed"
    gap = activity.get("max_gap_s")
    if gap is not None:
        if gap > MAX_ACTIVITY_GAP_S:
            return f"{gap:.0f} s without a sample"
    elif (wall_s := record.get("wall_s")) and wall_s > samples * MAX_SAMPLE_PERIOD_S:
        return f"{samples} samples over {wall_s:.0f} s"
    return None


def _contaminating(record: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    """The processes that contaminate ``record`` (``CONTAMINATION_SM_PCT``,
    and for an autotune record -- one without a bench record's ``impls`` --
    ``AUTOTUNE_CONTAMINATION_SHARE`` and ``AUTOTUNE_CONTAMINATION_MIN_S``),
    or None when its samples do not cover its job."""
    if activity_coverage_issue(record) is not None:
        return None
    activity = record["foreign_gpu_activity"]
    found = {
        name: entry
        for name, entry in activity["processes"].items()
        if entry["max_sm"] >= CONTAMINATION_SM_PCT
    }
    # Records made before busy_s keep the any-sample rule, and with it the
    # verdicts of the campaigns that made them.
    if found and "impls" not in record and all("busy_s" in e for e in found.values()):
        busy_s = sum(entry["busy_s"] for entry in found.values())
        job_s = activity.get("watched_s", record.get("wall_s")) or 0.0
        if busy_s < max(
            AUTOTUNE_CONTAMINATION_MIN_S, AUTOTUNE_CONTAMINATION_SHARE * job_s
        ):
            return {}
    return found


def foreign_activity(record: dict[str, Any]) -> list[str] | None:
    """The processes outside a job that computed on its GPU while it ran, as
    ``user:pid (max SM%, seconds active)``; None when the record cannot tell
    (made before the sampling existed, or its samples do not cover the job:
    :func:`activity_coverage_issue`)."""
    found = _contaminating(record)
    if found is None:
        return None
    return [
        f"{name} ({entry['max_sm']}% SM, {entry['active_s']} s)"
        for name, entry in found.items()
    ]


def activity_issue(
    record: dict[str, Any], include_unsampled: bool = False
) -> str | None:
    """Why ``record``'s timings or configs cannot count: processes outside
    its job contaminated it (:func:`foreign_activity`, or the campaign kept
    it, flagged ``contaminated``, as the cleanest of contaminated attempts),
    or its samples do not cover its job (:func:`activity_coverage_issue`);
    None when they can.  ``include_unsampled`` admits a record made without
    the sampling (no ``foreign_gpu_activity`` at all: before it existed, or
    with ``BB_FOREIGN_ACTIVITY=0``), never one whose samples miss its job."""
    found = foreign_activity(record)
    if found:
        return f"contaminated: {', '.join(found)}"
    if record.get("contaminated"):
        return "contaminated: the cleanest of its attempts"
    if found is None and not (
        include_unsampled and record.get("foreign_gpu_activity") is None
    ):
        return activity_coverage_issue(record)
    return None


def contamination_rank(record: dict[str, Any]) -> float:
    """How much foreign processes computed while a job ran, for keeping the
    cleanest of several contaminated attempts: the contaminating processes'
    seconds active, each weighted by its peak SM share."""
    found = _contaminating(record) or {}
    return sum(entry["max_sm"] / 100 * entry["active_s"] for entry in found.values())


def git_revision(path: Path) -> str | None:
    try:
        rev = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{rev}{'-dirty' if dirty else ''}"


def versions() -> dict[str, Any]:
    import importlib.metadata

    import torch

    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda_runtime": torch.version.cuda,
        "helion_git": git_revision(REPO_ROOT),
        "better_benchmark_git": git_revision(BB_ROOT),
    }
    for package in ("triton", "nvidia-cutlass-dsl"):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = None
    return out
