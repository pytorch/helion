"""Re-tune the pretuned linear-attention config tables.

The linattn benchmark (benchmarks/run_linattn.py) runs with
HELION_AUTOTUNE_CACHE=AOTAutotuneCache, so every helion.experimental.linear_attention
kernel call takes its config from the table beside that module,
_helion_aot_linear_attention_<device>_<arch>.py, instead of autotuning. A table goes
stale when code generation changes under it (#3859 started honoring its
tensor_descriptor slots) or when a kernel's call signature changes, after which the
call only finds a nearest-shape fallback. This regenerates the current GPU's table:

1. Every (variant, shape) the linattn suite runs becomes a job: a subprocess
   repeats the job's Helion calls (the accuracy pass, then forward and
   forward+backward) with autotuning on and records each call's winning config
   under the table's key, its tensor shapes plus its None/bool arguments.
2. Jobs that share kernels at one shape run back to back on one GPU, so later jobs
   reuse earlier tuning through the local autotune cache. The groups spread over
   --gpus.
3. The records replace the table's entries; the file's preamble is kept.

A call the autotuner refuses to tune (AutotuneError, e.g. when its default and
reference configs disagree) keeps the table's existing config. Records and logs go
to --work-dir, so an interrupted run resumes where it stopped.

Usage:
    python -m benchmarks.tune_linattn --gpus 0,1,2,3,4,5,6,7
    python -m benchmarks.tune_linattn --gpus 0,1 --kernel kda,delta_rule

At HELION_AUTOTUNE_EFFORT=full (the default here) the full sweep takes about 4.5
hours on 8 H100s.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
from typing import TYPE_CHECKING

from benchmarks.run_linattn import FUSED_PREAMBLE_VARIANTS
from benchmarks.run_linattn import SHAPES
from benchmarks.run_linattn import VARIANTS
from benchmarks.run_linattn import VARLEN_SHAPES
from benchmarks.run_linattn import VARLEN_VARIANTS
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

REPO = Path(__file__).resolve().parents[1]
# Variants that share kernels with each other but few with the rest.
DELTA_FAMILY = ["kda", "kda_fused", "delta_rule", "gated_delta_rule"]

CallKey = tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]
Config = dict[str, object]


def call_key(args: Sequence[object]) -> CallKey:
    """The table's key for a call: its tensor shapes and its None/bool arguments
    (the same as _call_key in the table modules)."""
    shapes = tuple(
        tuple(int(d) for d in a.shape) for a in args if isinstance(a, torch.Tensor)
    )
    flags = tuple(repr(a) for a in args if a is None or isinstance(a, bool))
    return shapes, flags


def default_table() -> Path:
    """The table the AOT cache reads on this GPU."""
    from helion._hardware import get_hardware_info
    import helion.experimental.linear_attention as engine

    hw = get_hardware_info()
    name = f"_helion_aot_linear_attention_{hw.device_kind}_{hw.compute_capability}.py"
    return Path(engine.__file__).with_name(name)


def load_table(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(variant: str, shape: int, out: Path, table: Path) -> None:
    """Run one job's Helion calls with autotuning on and write every engine call's
    config to ``out`` as {kernel: {repr(call key): config}}."""
    import helion
    from helion import exc
    from helion._testing import DEVICE
    from helion._testing import import_example
    import helion._testing.linear_attention_harness as lah
    import helion.experimental.linear_attention as engine
    from helion.runtime.kernel import Kernel

    existing = load_table(table) if table.exists() else None
    engine_file = inspect.getsourcefile(engine)
    records: dict[str, dict[str, Config]] = {}
    call = Kernel.__call__

    def recording_call(self: Kernel, *args: object, **kwargs: object) -> object:
        if inspect.getsourcefile(self.fn) != engine_file:
            return call(self, *args, **kwargs)
        nargs = self.normalize_args(*args, **kwargs)
        key = repr(call_key(nargs))
        try:
            result = call(self, *args, **kwargs)
        except exc.AutotuneError:
            if existing is None:
                raise
            print(f"keeping the table's config for {self.name} {key}", flush=True)
            config = getattr(existing, f"autotune_{self.name}")(*nargs)
            self.bind(nargs).set_config(helion.Config(**config))
            result = call(self, *args, **kwargs)
        config = self.bind(nargs)._config
        assert config is not None
        records.setdefault(self.name, {}).setdefault(key, dict(config.config))
        return result

    Kernel.__call__ = recording_call  # pyrefly: ignore[bad-assignment]

    base = FUSED_PREAMBLE_VARIANTS.get(variant) or VARLEN_VARIANTS.get(variant, variant)
    harness = import_example(f"linear.example_{base}").HARNESS
    if variant in VARLEN_VARIANTS:
        _, lens, h, d = VARLEN_SHAPES[shape]
        b, t = len(lens), sum(lens)
        kw: dict[str, object] = {"varlen": True, "varlen_lengths": lens}
    else:
        _, b, t, h, d = SHAPES[shape]
        kw = {"fused_preamble": True} if variant in FUSED_PREAMBLE_VARIANTS else {}
    # run_linattn.py's accuracy pass, then the Helion half of its timing loop
    # (linear_attention_harness._time_config).
    print("accuracy:", harness.accuracy([(b, h, t, d, d)], **kw), flush=True)
    inputs = harness.make_inputs(
        b, h, t, d, d, dtype=lah.DTYPE, device=DEVICE, requires_grad=not kw, **kw
    )
    harness.helion_fwd(inputs, lah.BENCH_C)
    if not inputs.preamble:
        grad_out = torch.randn(b, h, t, d, device=DEVICE, dtype=lah.DTYPE)
        harness.helion_fb(inputs, grad_out, lah.BENCH_C)
    torch.cuda.synchronize()
    out.write_text(json.dumps(records, indent=1))
    print(f"recorded {sum(map(len, records.values()))} calls", flush=True)


def job_groups(variants: Sequence[str]) -> list[list[tuple[str, int]]]:
    """(variant, shape index) jobs, grouped so that jobs sharing kernels at one
    shape run in order; the largest shapes come first."""
    order = sorted(range(len(SHAPES)), key=lambda i: -math.prod(SHAPES[i][1:]))
    dense = [v for v in variants if v not in VARLEN_VARIANTS]
    families = [
        [v for v in dense if v in DELTA_FAMILY],
        [v for v in dense if v not in DELTA_FAMILY],
    ]
    groups = [[(v, i) for v in fam] for i in order for fam in families if fam]
    varlen = [
        (v, i)
        for v in variants
        if v in VARLEN_VARIANTS
        for i in range(len(VARLEN_SHAPES))
    ]
    if varlen:
        groups.insert(min(2, len(groups)), varlen)
    return groups


def run_jobs(
    groups: list[list[tuple[str, int]]], gpus: list[str], work_dir: Path, table: Path
) -> list[str]:
    """Run each group on the next free GPU; returns the jobs that failed."""
    env = {k: v for k, v in os.environ.items() if k != "HELION_AUTOTUNE_CACHE"}
    env.setdefault("HELION_AUTOTUNE_EFFORT", "full")
    env["HELION_CACHE_DIR"] = str(work_dir / "cache")
    free: queue.Queue[str] = queue.Queue()
    for gpu in gpus:
        free.put(gpu)
    failed: list[str] = []

    def run_group(group: list[tuple[str, int]]) -> None:
        gpu = free.get()
        try:
            for variant, shape in group:
                name = f"{variant}_{shape}"
                out = work_dir / "records" / f"{name}.json"
                if out.exists():
                    continue
                cmd = [sys.executable, "-m", "benchmarks.tune_linattn"]
                cmd += [
                    "--table",
                    str(table),
                    "--record",
                    variant,
                    str(shape),
                    str(out),
                ]
                start = time.time()
                with open(work_dir / "logs" / f"{name}.log", "w") as log:
                    rc = subprocess.call(
                        cmd,
                        cwd=REPO,
                        env={**env, "CUDA_VISIBLE_DEVICES": gpu},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                print(
                    f"gpu {gpu}: {name} rc={rc} {time.time() - start:.0f}s", flush=True
                )
                if rc:
                    failed.append(name)
        finally:
            free.put(gpu)

    with ThreadPoolExecutor(len(gpus)) as pool:
        list(pool.map(run_group, groups))
    return failed


def table_section(name: str, entries: list[tuple[CallKey, Config]]) -> str:
    keys = "\n".join(f"    {key!r}," for key, _ in entries)
    configs = "\n".join(f"    {config!r}," for _, config in entries)
    return f'''_KEYS_{name} = [
{keys}
]

_CONFIGS_{name} = [
{configs}
]


def key_{name}(*args) -> int:
    """Config index for the given args (also the cache key)."""
    return _select(_KEYS_{name}, args)


def autotune_{name}(*args) -> dict:
    """Config dict for the given args."""
    return _CONFIGS_{name}[key_{name}(*args)]
'''


def write_table(records: Path, table: Path, keep_existing: bool) -> None:
    """Rewrite ``table`` from the record files, keeping its preamble. Where two jobs
    recorded different configs for one call key, the first job by name wins. With
    ``keep_existing``, calls no record covers keep their current entries."""
    merged: dict[str, dict[CallKey, Config]] = {}
    if keep_existing:
        old = load_table(table)
        for attr in dir(old):
            if attr.startswith("_KEYS_"):
                name = attr.removeprefix("_KEYS_")
                configs = getattr(old, f"_CONFIGS_{name}")
                merged[name] = dict(zip(getattr(old, attr), configs, strict=True))
    fresh: dict[str, dict[CallKey, Config]] = {}
    disagreements = 0
    for path in sorted(records.glob("*.json")):
        for name, calls in json.loads(path.read_text()).items():
            for key, config in calls.items():
                kept = fresh.setdefault(name, {}).setdefault(
                    ast.literal_eval(key), config
                )
                disagreements += kept != config
    for name, calls in fresh.items():
        merged.setdefault(name, {}).update(calls)
    source = table.read_text()
    preamble = source[: source.index("\n_KEYS_") + 1]
    sections = [
        table_section(name, sorted(calls.items(), key=lambda e: (e[0][1], e[0][0])))
        for name, calls in sorted(merged.items())
    ]
    table.write_text(preamble + "\n\n".join(sections))
    print(
        f"wrote {table}: {sum(map(len, merged.values()))} calls over {len(merged)} "
        f"kernels ({disagreements} calls recorded by several jobs kept the first config)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-tune the linear-attention AOT config table for this GPU."
    )
    parser.add_argument(
        "--gpus", default="0", help="comma-separated GPU indices to tune on"
    )
    parser.add_argument(
        "--kernel",
        default=",".join(VARIANTS),
        help="comma-separated variants to re-tune (default: all); with a subset, "
        "calls the subset does not make keep their table entries",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/helion_tune_linattn"),
        help="records, logs and the autotune cache; reused to resume",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=None,
        help="table to rewrite (default: the one the AOT cache reads on this GPU)",
    )
    parser.add_argument("--record", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    table: Path = args.table or default_table()
    if args.record:
        variant, shape, out = args.record
        record(variant, int(shape), Path(out), table)
        return

    variants = [v.strip() for v in args.kernel.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown kernels: {unknown}; choose from {VARIANTS}")
    if not table.exists():
        raise SystemExit(f"{table} does not exist; copy a sibling table there first")
    for sub in ("records", "logs", "cache"):
        (args.work_dir / sub).mkdir(parents=True, exist_ok=True)
    groups = job_groups(variants)
    failed = run_jobs(groups, args.gpus.split(","), args.work_dir, table)
    if failed:
        raise SystemExit(f"failed jobs (logs in {args.work_dir}/logs): {failed}")
    write_table(args.work_dir / "records", table, set(variants) != set(VARIANTS))


if __name__ == "__main__":
    main()
