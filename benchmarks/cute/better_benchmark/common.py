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


def compile_policy(repro_id: str, shape: str) -> dict[str, Any]:
    bb_path()
    import repro_harness  # pyrefly: ignore [missing-import]

    return repro_harness.compile_policy_from_config(shape_configs(repro_id)[shape])


def compile_inductor(
    repro_id: str, shape: str, instance: Any, *, cd: bool
) -> Callable[..., Any]:
    """``torch.compile`` of the repro as better-benchmark compiles it (the
    shape point's compile policy), with ``coordinate_descent_tuning`` when
    ``cd``. Compiles with different options coexist in one process (each
    wraps its own backend), so no dynamo reset is needed in between."""
    bb_path()
    import repro_harness  # pyrefly: ignore [missing-import]

    options = {"coordinate_descent_tuning": True} if cd else None
    return repro_harness.compile_repro(
        instance,
        compile_policy=compile_policy(repro_id, shape),
        **({"options": options} if options else {}),
    )


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
    with env_override(HELION_BACKEND=backend):
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


def load_autotuned_port(
    configs_dir: Path, repro_id: str, shape: str, backend: str
) -> tuple[types.ModuleType, dict[str, Any]]:
    """The port with every kernel pinned to the decorator its autotune
    record (``autotune.py``'s, which must be ok) holds, and the record."""
    path = config_record_path(configs_dir, repro_id, shape, backend)
    record = json.loads(path.read_text())
    if record.get("status") != "ok":
        raise RuntimeError(f"autotune record {path} is not ok")
    pinned = {name: kernel["decorator"] for name, kernel in record["kernels"].items()}
    return load_port(repro_id, backend, pinned=pinned), record


def timed_record_fields(tuned: dict[str, Any]) -> dict[str, Any]:
    """The fields with which a bench record names the autotune record
    ``tuned`` that a port's timing pinned: its kernels' code digests and
    whether it was provisional or reused.  An autotune that runs after the
    bench replaces the record, and ``report.py`` counts the timing only while
    the record holds the same kernels."""
    return {
        "autotune_code_sha256": {
            name: kernel["code_sha256"] for name, kernel in tuned["kernels"].items()
        },
        "autotune_provisional": bool(tuned.get("provisional")),
        "autotune_reused_from": tuned.get("reused_from"),
    }


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
    :class:`AnchoredGate` compares them with Inductor's."""
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
            raise ExactMismatch(
                f"output {index}: {mismatches} of {e.numel()} elements differ"
            )
    if float_errors:
        raise FloatMismatch("\n".join(float_errors))


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
    against the reference like the deterministic outputs."""

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
    if unknown > max(MAX_UNKNOWN_SAMPLES, samples // 20):
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
