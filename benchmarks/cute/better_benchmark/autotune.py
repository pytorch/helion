"""Cold full autotune of one Helion port on one backend, in this process.

Runs the port's ``forward`` once on the repro's inputs, which autotunes every
Helion kernel it calls (in call order, each on its real arguments), then
records each kernel's winning ``@helion.kernel(...)`` decorator. A kernel
that the port runs behind earlier kernels is tuned behind them
(``--regime port``, the CuTe backend's default; ``common.kernels_in_sequence``;
Triton ports default to ``alone``, ``common.default_regime``): the search
runs those kernels, at the configs they were just tuned to, between the L2
flush and every timed call of a candidate, so the candidate finds its
operands as the port's earlier kernels leave them and is timed alone from
there. The port's first kernel is timed alone after a flush, which is how
the port runs it. ``--regime alone`` times every kernel that way, as every
autotune did before; the record names the regime and, per kernel, how its
search was timed and why (``regime``, ``regimes``). The record is
checked by rebuilding the kernels from those decorators and comparing their
generated code and outputs, which is how ``bench.py`` pins them later. That
call of the pinned port starts from poisoned memory, and a pick that leaves
part of an output unwritten fails the record (``common.check_written``): the
search's own calls left the allocator's free blocks full of right results.
The record holds two digests of each kernel's code (``common.code_digests``):
of the whole text (``code_sha256``) and of the text without the comments
that quote the port's source (``code_nosrc_sha256``), which is the one the
kernel is pinned by, so a comment in the port or a line added above a kernel
leaves the record standing.

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


def new_record(
    repro_id: str, shape: str, backend: str, regime: str | None = None
) -> dict[str, Any]:
    """The record of an autotune as it starts: its port, how its searches
    time their candidates (``regime``, ``common.REGIMES``; None: the
    backend's default, ``common.default_regime``), the job's
    HELION_* environment (``env``: the search's random seed among it, which
    ``common.record_seed`` reads back, and the file of its per-config log)
    and, under ``campaign.py --seeds-from``, the run whose autotune of this
    port had that seed (``seed_from``, from ``common.SEED_FROM``)."""
    record: dict[str, Any] = {
        "repro": repro_id,
        "shape": shape,
        "backend": backend,
        "regime": regime or common.default_regime(backend),
        "status": "error",
        "env": {k: v for k, v in os.environ.items() if k.startswith("HELION_")},
        "gpu": common.gpu_info(),
        "versions": common.versions(),
    }
    if common.SEED_FROM in os.environ:
        record["seed_from"] = os.environ[common.SEED_FROM]
    return record


def searched_regimes(
    regimes: dict[str, dict[str, Any]], autotunes: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """``regimes`` (``common.kernels_in_sequence``) with what each kernel's
    search says of itself: ``prelude_steps``, the earlier kernel calls its
    final selection ran between the flush and every timed call (0: it timed
    the kernel alone; absent: no search ran, or none with a final
    selection).

    Every search is held to the kernel call it ran in (``port_call`` in its
    metrics, from ``common.searching_call``: the port's name for the kernel,
    the call's regime and the calls before it), not to the kernel's first
    call: a kernel called with a second signature searches again, behind
    more calls. A search that ran another number of earlier calls than its
    call was given was not tuned in the regime the record claims, which is
    an error. The entry of a kernel keeps its first such search's count."""
    for metrics in autotunes:
        call = metrics.get("port_call")
        selection = metrics.get("final_selection")
        if call is None or selection is None:
            continue
        steps = selection.get("prelude_steps", 0)
        expected = call["behind"] if call["regime"] == "port" else 0
        if steps != expected:
            raise RuntimeError(
                f"{call['kernel']} was to be tuned behind {expected} earlier "
                f"kernel call(s); its search ran {steps}"
            )
        regimes[call["kernel"]].setdefault("prelude_steps", steps)
    return regimes


def run(args: argparse.Namespace) -> dict[str, Any]:
    shape = args.shape or common.default_shape(args.repro_id)
    record = new_record(args.repro_id, shape, args.backend, args.regime)
    regime = record["regime"]
    # Who chose to tune every kernel alone, for the record's reasons.
    alone_reason = (
        "--regime alone"
        if args.regime == "alone"
        else f"the {args.backend} backend's default regime"
    )
    started = time.monotonic()
    try:
        torch.cuda.set_device(0)
        autotunes: list[dict[str, Any]] = []
        # (a search runs inside the kernel call that started it, which
        # ``common.kernels_in_sequence`` names)
        register_post_autotune_hook(
            lambda m: autotunes.append(
                {**m.to_dict(), "port_call": common.searching_call.get()}
            )
        )
        instance = common.load_repro_module(args.repro_id).Repro()
        with torch.no_grad():
            expected = instance(*common.make_inputs(args.repro_id, shape))
        args.stochastic = common.stochastic_outputs(args.repro_id, shape, instance)
        args.inductor = common.inductor_outputs(args.repro_id, shape, instance)
        port = common.load_port(args.repro_id, args.backend)
        tuned_at = time.monotonic()
        regimes: dict[str, dict[str, Any]] = {}
        with (
            torch.no_grad(),
            common.kernels_in_sequence(port, regime, regimes, alone_reason),
        ):
            actual = port.forward(common.make_inputs(args.repro_id, shape))
            torch.cuda.synchronize()
        record["autotune_s"] = round(time.monotonic() - tuned_at, 1)
        record["autotunes"] = autotunes
        record["regimes"] = searched_regimes(regimes, autotunes)
        record["errors_vs_eager"] = common.output_errors(actual, expected)
        check(actual, expected, args, record)
        kernels = common.bound_kernel_records(port)
        record["kernels"] = kernels
        # A kernel of the port's module that forward bound without going
        # through its name there (an alias taken at import, a closure) ran
        # outside the wrappers.
        unfollowed = sorted(
            name
            for name in common.port_kernels(port)
            if name in kernels and name not in regimes
        )
        if unfollowed and regime == "port":
            # The kernels after it were tuned behind preludes that lack its
            # calls, and their entries count the calls before them wrongly.
            raise RuntimeError(
                f"forward ran {unfollowed} other than through their names in "
                "the port's module, so the harness could not follow them: the "
                "port regime needs every kernel call (PORTING.md)"
            )
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
    parser.add_argument(
        "--regime",
        choices=common.REGIMES,
        default=None,
        help="how the searches time their candidates: 'port' times a kernel "
        "that runs behind earlier kernels of its port behind those kernels "
        "(the L2 state the port runs it in); 'alone' times every kernel alone "
        "after an L2 flush, as autotunes did before the regime existed "
        "(default: port for --backend cute, alone for triton, whose "
        "event-timed searches were not measured in the port regime)",
    )
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
