"""Cold full autotune of one Helion port on one backend, in this process.

Runs the port's ``forward`` once on the repro's inputs, which autotunes every
Helion kernel it calls (in call order, each on its real arguments), then
records each kernel's winning ``@helion.kernel(...)`` decorator. The record is
checked by rebuilding the kernels from those decorators and comparing their
generated code and outputs, which is how ``bench.py`` pins them later.

Run it through ``campaign.py``, which gives every job a fresh process, a fresh
random seed and private cache directories; standalone::

    python benchmarks/cute/better_benchmark/autotune.py \\
        var_mean_004f9a9ea198 --backend cute --output /tmp/cfg.json

Either way the autotune is cold: ``HELION_FORCE_AUTOTUNE=1`` unless the
environment sets it, so no config comes from Helion's autotune cache.
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
    # A cold autotune: no config from Helion's autotune cache.
    os.environ.setdefault("HELION_FORCE_AUTOTUNE", "1")

from benchmarks.cute.better_benchmark import common
import torch

from helion.autotuner.metrics import register_post_autotune_hook


def check(
    actual: object, expected: object, args: argparse.Namespace, record: dict[str, Any]
) -> None:
    """Structure and exact outputs must match eager; float drift beyond the
    loose tolerance is recorded (bench.py runs the anchored gate)."""
    try:
        common.check_outputs(
            actual,
            expected,
            rtol=args.rtol,
            atol=args.atol,
            stochastic=args.stochastic,
        )
    except common.FloatMismatch as error:
        record.setdefault("eager_allclose_warnings", []).append(str(error)[:1000])


def run(args: argparse.Namespace) -> dict[str, Any]:
    shape = args.shape or common.default_shape(args.repro_id)
    record: dict[str, Any] = {
        "repro": args.repro_id,
        "shape": shape,
        "backend": args.backend,
        "status": "error",
        "env": {k: v for k, v in os.environ.items() if k.startswith("HELION_")},
        "gpu": common.gpu_info(),
        "versions": common.versions(),
    }
    started = time.monotonic()
    try:
        torch.cuda.set_device(0)
        autotunes: list[dict[str, Any]] = []
        register_post_autotune_hook(lambda m: autotunes.append(m.to_dict()))
        instance = common.load_repro_module(args.repro_id).Repro()
        with torch.no_grad():
            expected = instance(*common.make_inputs(args.repro_id, shape))
        args.stochastic = common.stochastic_outputs(args.repro_id, shape, instance)
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
        with torch.no_grad():
            again = pinned.forward(common.make_inputs(args.repro_id, shape))
        torch.cuda.synchronize()
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
