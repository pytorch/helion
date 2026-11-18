"""Cold full autotune of one Helion port on one backend, in this process.

Runs the port's ``forward`` once on the repro's inputs, which autotunes every
Helion kernel it calls (in call order, each on its real arguments), then
records each kernel's winning ``@helion.kernel(...)`` decorator. The record is
checked by rebuilding the kernels from those decorators and comparing their
generated code and outputs, which is how ``bench.py`` pins them later. That
call of the pinned port starts from poisoned memory, and a pick that leaves
part of an output unwritten fails the record (``common.check_written``): the
search's own calls left the allocator's free blocks full of right results.

Run it through ``campaign.py``, which gives every job a fresh process, a
random seed (a fresh one, or with ``--seeds-from`` the one another run's
autotune of the port had: the record's ``seed_from`` names that run), private
cache directories and a file for the search's per-config log; standalone::

    python benchmarks/cute/better_benchmark/autotune.py \\
        var_mean_004f9a9ea198 --backend cute --output /tmp/cfg.json

Either way the autotune is cold: ``HELION_FORCE_AUTOTUNE=1`` unless the
environment sets it (an empty value sets nothing), so no config comes from
Helion's autotune cache. It is a full search (``HELION_AUTOTUNE_EFFORT=full``)
unless the environment names another effort: ``common.load_port`` runs an
unpinned port's default configs when neither variable asks for a search, so
the script names the effort also where the cache is allowed
(``HELION_FORCE_AUTOTUNE=0``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    # A cold autotune: no config from Helion's autotune cache. And a search
    # also when the environment allows the cache: load_port runs an unpinned
    # port's default configs unless an effort or the force is set. An empty
    # value sets neither, for load_port as for Helion's settings.
    for name, value in (
        ("HELION_FORCE_AUTOTUNE", "1"),
        ("HELION_AUTOTUNE_EFFORT", "full"),
    ):
        if not os.environ.get(name, "").strip():
            os.environ[name] = value

from benchmarks.cute.better_benchmark import common
import torch

from helion.autotuner.metrics import register_post_autotune_hook


def check(
    actual: object, expected: object, args: argparse.Namespace, record: dict[str, Any]
) -> None:
    """Structure and exact outputs must match eager; float drift beyond the
    loose tolerance is recorded (bench.py runs the anchored gate). An integer
    or boolean output that the default Inductor compile's does not reproduce
    either must equal Inductor's instead (``common.check_outputs``), which
    the record says (``exact_vs_inductor``): the only case in which this job
    compiles the repro with Inductor."""
    vs_inductor: list[dict[str, Any]] = []
    try:
        common.check_outputs(
            actual,
            expected,
            rtol=args.rtol,
            atol=args.atol,
            stochastic=args.stochastic,
            inductor=args.inductor,
            vs_inductor=vs_inductor,
        )
    except common.FloatMismatch as error:
        record.setdefault("eager_allclose_warnings", []).append(str(error)[:1000])
    if vs_inductor:
        record["exact_vs_inductor"] = vs_inductor


def new_record(repro_id: str, shape: str, backend: str) -> dict[str, Any]:
    """The record of an autotune as it starts: its port, the job's HELION_*
    environment (``env``: the search's random seed among it, which
    ``common.record_seed`` reads back, and the file of its per-config log)
    and, under ``campaign.py --seeds-from``, the run whose autotune of this
    port had that seed (``seed_from``, from ``common.SEED_FROM``)."""
    record: dict[str, Any] = {
        "repro": repro_id,
        "shape": shape,
        "backend": backend,
        "status": "error",
        "env": {k: v for k, v in os.environ.items() if k.startswith("HELION_")},
        "gpu": common.gpu_info(),
        "versions": common.versions(),
    }
    if common.SEED_FROM in os.environ:
        record["seed_from"] = os.environ[common.SEED_FROM]
    return record


def run(args: argparse.Namespace) -> dict[str, Any]:
    shape = args.shape or common.default_shape(args.repro_id)
    record = new_record(args.repro_id, shape, args.backend)
    started = time.monotonic()
    try:
        torch.cuda.set_device(0)
        autotunes: list[dict[str, Any]] = []
        register_post_autotune_hook(lambda m: autotunes.append(m.to_dict()))
        instance = common.load_repro_module(args.repro_id).Repro()
        with torch.no_grad():
            expected = instance(*common.make_inputs(args.repro_id, shape))
        args.stochastic = common.stochastic_outputs(args.repro_id, shape, instance)
        args.inductor = common.inductor_outputs(args.repro_id, shape, instance)
        port = common.load_port(args.repro_id, args.backend)
        tuned_at = time.monotonic()
        with torch.no_grad():
            actual = port.forward(common.make_inputs(args.repro_id, shape))
            torch.cuda.synchronize()
        record["autotune_s"] = round(time.monotonic() - tuned_at, 1)
        record["autotunes"] = autotunes
        record["errors_vs_eager"] = common.output_errors(actual, expected)
        check(actual, expected, args, record)
        kernels = common.bound_kernel_records(port)
        record["kernels"] = kernels
        # Round trip: rebuild from the recorded decorators, as bench.py will.
        pinned = common.load_port(
            args.repro_id,
            args.backend,
            pinned={name: k["decorator"] for name, k in kernels.items()},
        )

        again = common.forward_on_poisoned_memory(pinned, args.repro_id, shape)
        common.check_written(
            again,
            lambda byte: common.forward_on_poisoned_memory(
                pinned, args.repro_id, shape, byte
            ),
        )
        check(again, expected, args, record)
        record["pin_roundtrip"] = common.pin_roundtrip(pinned, kernels)
        if not all(record["pin_roundtrip"].values()):
            raise RuntimeError(
                f"pinned decorators generate different code: {record['pin_roundtrip']}"
            )
        record["status"] = "ok"
    except Exception:
        record["error"] = traceback.format_exc()
    finally:
        record["wall_s"] = round(time.monotonic() - started, 1)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("repro_id")
    parser.add_argument(
        "--shape",
        default=None,
        help="shape hash (default: the repro's selection.json point, else its first)",
    )
    parser.add_argument("--backend", required=True, choices=("cute", "triton"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    args = parser.parse_args()
    activity: dict[str, Any] = {}
    with common.watch_foreign_gpu_activity(activity):
        record = run(args)
    record.update(activity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(args.output, record)
    print(f"{record['status']} {args.repro_id} {args.backend} -> {args.output}")
    if record["status"] != "ok":
        print(record.get("error", ""), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
