# ruff: noqa: ANN401
"""Time one better-benchmark repro under every implementation, interleaved.

Implementations (``--impls``):

- ``inductor``: ``torch.compile`` of the repro, default Inductor config
- ``inductor_cd``: the same with ``coordinate_descent_tuning=True``
- ``oracle``: the repro's hand-written reference kernel (``oracle.py``)
- ``helion_triton`` / ``helion_cute``: the Helion port with the kernels pinned
  to the decorators a cold full autotune reported (``autotune.py``); the
  record names that autotune record's code digests and whether it was
  provisional (``autotune_code_sha256``, ``autotune_provisional``), so
  ``report.py`` tells the timing from one of a later autotune of the port
- ``memcopy``: better-benchmark's speed-of-light proxy, a copy of the repro's
  effective bytes (``byte_accounting.count_bytes_effective``)

All implementations live in one process on one GPU, each on its own freshly
generated inputs, and are timed with the same timers, alternating between
implementations every round, in forward order in even rounds and in reverse
order in odd ones. A reading depends on what ran right before it (by up to a
microsecond for a small kernel), so each round-based timing is the mean of
its forward-order and reverse-order values, and the default round counts are
even:

- ``bb_us``: better-benchmark's methodology, ``do_bench(min)`` of a
  single-call CUDA graph replay (L2 flushed by do_bench before every replay;
  includes one graph launch), the minimum over each order's rounds
  (``bb_order_mins_us``, forward first), averaged.
- ``cold_us``: Helion's autotuner regime, ``interleaved_bench`` with the L2
  flushed before every call inside a batched CUDA graph and the flush-only
  time subtracted (no launch-latency quantization), the median over each
  order's rounds (``cold_order_medians_us``), averaged. ``cold_runs_us``
  keeps every round; ``report.py --rebalance`` recomputes both timings from
  the rounds of records made before (whose ``cold_us`` was the median of 5
  rounds, a forward-order reading).
- ``kernel_us``: summed device time of the kernels one replay launches
  (torch.profiler; back-to-back replays, so L2-warm), with per-kernel names.

Correctness is checked on what the timers run, a replay of the
implementation captured in a CUDA graph (the oracles of the dropout repros
take another path when they are not captured). Every implementation must
match eager PyTorch's output count, shapes and dtypes, and eager's integer
outputs exactly (except those that depend on random numbers). The Helion
ports must also match eager's strides; Inductor and the oracles may differ
(Inductor pads some output strides), which ``errors_vs_eager`` records. Float
outputs outside eager's loose tolerance (rtol = atol = 0.1) are only noted
(``eager_allclose_warning``): eager rounds every intermediate the graph
computes in bf16, so it can sit far from Inductor and every other fp32
implementation (pointwise_238e96847398 evaluates its whole
cancellation-dominated graph in bf16; sum_sum_sum_d2c97d17f3dc rounds
intermediates to bf16 and its outputs reach 1e18). The coordinate-descent
compile, the oracle and the Helion ports also go through the anchored
numerics gate (``common.AnchoredGate``, a loosened variant of
better-benchmark's) with the default Inductor compile as the error baseline,
which compares the outputs that depend on random numbers with that
compile's. The default compile goes through it too: judged against itself,
it fails only where its output is not finite and the reference is, which
fails every other implementation as well. The oracle's random outputs are
recorded but not gated, as
better-benchmark does: the dropout oracles draw Inductor's random numbers
under capture but round them like eager, so about 2**-12 of their mask
elements differ from Inductor's by design. ``numerics_pass`` records the
verdict (reports leave failing implementations out). The processes outside
the job on its GPU are listed before and after the timing, and ``nvidia-smi
pmon`` samples their SM utilization once a second through the whole run
(``foreign_gpu_activity``; ``BB_FOREIGN_ACTIVITY=0`` turns it off), so a
record tells a foreign job that computed from an idle resident process.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import statistics
import sys
import time
import traceback
from typing import Any
from typing import Callable
from typing import cast

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.cute.better_benchmark import common
import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile

from helion.autotuner.benchmarking import interleaved_bench
from helion.autotuner.benchmarking import steady_graph_dispatch

ALL_IMPLS = (
    "inductor",
    "inductor_cd",
    "oracle",
    "helion_triton",
    "helion_cute",
    "memcopy",
)
# Judged by the gate against the default Inductor compile ("inductor"),
# which the gate judges too: against itself, that fails only an output that
# is not finite where the reference is (common._anchored_leaf).
GATED_IMPLS = ("inductor", "inductor_cd", "oracle", "helion_triton", "helion_cute")


def _profile_kernels(replay: Callable[[], None], calls: int = 10) -> dict[str, Any]:
    replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            replay()
        torch.cuda.synchronize()
    per_name: dict[str, list[float]] = {}
    for event in prof.events() or []:
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        per_name.setdefault(event.name, []).append(event.time_range.elapsed_us())
    kernels = []
    total = 0.0
    for name, durations in per_name.items():
        per_call = len(durations) / calls
        median = statistics.median(durations)
        total += median * per_call
        kernels.append({"name": name[:200], "us": median, "per_call": per_call})
    kernels.sort(key=lambda k: -k["us"] * k["per_call"])
    return {"kernel_us": total, "kernels": kernels}


def _replayed(
    call: Callable[[list[Any]], Any], fresh_inputs: Callable[[], list[Any]]
) -> Callable[[list[Any]], Any]:
    """``call`` run the way the timers run it: warmed up eagerly (on other
    inputs: a repro can write into its inputs), then captured in a CUDA graph
    on the given inputs and replayed once. The checks judge this, not an
    eager call, because an implementation may take another path under
    capture: the oracles of the dropout repros draw their random numbers in
    the kernel from Inductor's seed (``tl.rand``) while capturing, and load
    a random tensor matching eager's otherwise."""

    def run(inputs: list[Any]) -> Any:
        call(fresh_inputs())
        torch.cuda.synchronize()
        graph, outputs = common.capture_graph(lambda: call(inputs))
        graph.replay()
        torch.cuda.synchronize()
        return outputs

    return run


def do_bench(fn: Callable[[], object], **kwargs: object) -> object:
    """``triton.testing.do_bench``, imported when first called: the test
    module imports this one on lanes whose torch build ships no Triton."""
    from triton.testing import do_bench as triton_do_bench

    return triton_do_bench(fn, **kwargs)


def time_impls(
    fns: dict[str, Callable[[], Any]],
    *,
    rounds: int,
    rep_ms: int,
    cold_rounds: int,
) -> dict[str, dict[str, Any]]:
    """Time ``fns`` with both timers; an implementation that cannot be
    warmed up or captured is reported with an error, the others still run."""
    out: dict[str, dict[str, Any]] = {name: {} for name in fns}
    note = steady_graph_dispatch()
    graphs = {}
    for name, fn in fns.items():
        out[name]["graph_state_warmup"] = note
        try:
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            graph, _ = common.capture_graph(fn)
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            graphs[name] = graph
        except Exception:
            out[name]["timing_error"] = traceback.format_exc()
            torch.cuda.synchronize()
    names = list(graphs)
    # Settle a power-capped GPU on its sustained clock before timing.
    started = time.perf_counter()
    while time.perf_counter() - started < 0.5:
        for name in names:
            graphs[name].replay()
    torch.cuda.synchronize()
    bb_runs: dict[str, list[float]] = {name: [] for name in names}
    for r in range(rounds):
        order = names if r % 2 == 0 else names[::-1]
        for name in order:
            ms = do_bench(graphs[name].replay, warmup=25, rep=rep_ms, return_mode="min")
            bb_runs[name].append(float(cast("float", ms)) * 1000.0)
    cold_runs: dict[str, list[float]] = {name: [] for name in names}
    for r in range(cold_rounds):
        order = names if r % 2 == 0 else names[::-1]
        timings = interleaved_bench(
            [fns[name] for name in order],
            repeat=100,
            default_cudagraph=True,
            l2_flush_mode="call",
        )
        for name, ms in zip(order, timings, strict=True):
            cold_runs[name].append(ms * 1000.0)
    for name in names:
        bb_us, bb_order = common.order_balanced(bb_runs[name], min)
        cold_us, cold_order = common.order_balanced(cold_runs[name], statistics.median)
        out[name].update(
            bb_us=bb_us,
            bb_order_mins_us=bb_order,
            bb_runs_us=bb_runs[name],
            cold_us=cold_us,
            cold_order_medians_us=cold_order,
            cold_runs_us=cold_runs[name],
        )
        out[name].update(_profile_kernels(graphs[name].replay))
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    shape = args.shape or common.default_shape(args.repro_id)
    impls = [i for i in args.impls.split(",") if i]
    record: dict[str, Any] = {
        "repro": args.repro_id,
        "shape": shape,
        "status": "error",
        "gpu": common.gpu_info(),
        "versions": common.versions(),
        "impls": {},
    }
    started = time.monotonic()
    try:
        torch.cuda.set_device(0)
        torch._dynamo.reset()
        common.bb_path()
        import byte_accounting  # pyrefly: ignore [missing-import]

        def inputs() -> list[Any]:
            return common.make_inputs(args.repro_id, shape, seed=0)

        instance = common.load_repro_module(args.repro_id).Repro()
        with torch.no_grad():
            expected = instance(*inputs())
        record["outputs"] = [
            {"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype)}
            for t in common.flatten_outputs(expected)
        ]
        total_bytes = int(byte_accounting.count_bytes_effective(instance, inputs()))
        record["total_bytes"] = total_bytes

        fns: dict[str, Callable[[], Any]] = {}
        default_compile: Callable | None = None
        gate: common.AnchoredGate | None = None

        def inductor_default() -> Callable:
            nonlocal default_compile
            if default_compile is None:
                default_compile = common.compile_inductor(
                    args.repro_id, shape, instance, cd=False
                )
            return default_compile

        def numerics_gate() -> common.AnchoredGate:
            """The gate, with the default Inductor compile as its baseline."""
            nonlocal gate
            if gate is None:
                gate = common.AnchoredGate(
                    args.repro_id, shape, instance, inductor_default()
                )
            return gate

        for impl in impls:
            info: dict[str, Any] = {"status": "error"}
            record["impls"][impl] = info
            try:
                if impl == "memcopy":
                    n = max(total_bytes // 8, 256)
                    src = torch.empty(n, dtype=torch.float32, device="cuda")
                    dst = torch.empty_like(src)
                    fns[impl] = lambda src=src, dst=dst: torch.add(src, 1, out=dst)
                    info["status"] = "ok"
                    continue
                port = None
                tuned: dict[str, Any] = {}
                if impl == "inductor":
                    compiled = inductor_default()
                    call = lambda inp, c=compiled: c(*inp)  # noqa: E731
                    info["numerics_baseline"] = True
                elif impl == "inductor_cd":
                    compiled = common.compile_inductor(
                        args.repro_id, shape, instance, cd=True
                    )
                    call = lambda inp, c=compiled: c(*inp)  # noqa: E731
                elif impl == "oracle":
                    call = common.load_oracle(args.repro_id, inputs())
                elif impl.startswith("helion_"):
                    backend = impl.removeprefix("helion_")
                    info["autotune_record"] = str(
                        common.config_record_path(
                            args.configs_dir, args.repro_id, shape, backend
                        )
                    )
                    port, tuned = common.load_autotuned_port(
                        args.configs_dir, args.repro_id, shape, backend
                    )
                    info["decorators"] = {
                        n: k["decorator"] for n, k in tuned["kernels"].items()
                    }
                    info.update(common.timed_record_fields(tuned))
                    call = port.forward
                else:
                    raise ValueError(f"unknown implementation {impl}")
                replay = _replayed(call, inputs)
                with torch.no_grad():
                    actual = replay(inputs())
                info["errors_vs_eager"] = common.output_errors(actual, expected)
                try:
                    common.check_outputs(
                        actual,
                        expected,
                        rtol=1e-1,
                        atol=1e-1,
                        stochastic=numerics_gate().stochastic,
                        check_strides=port is not None,
                    )
                except common.FloatMismatch as error:
                    # Low-precision eager can sit far from the exact result
                    # (it rounds every intermediate); the anchored gate below
                    # judges floating-point values. Structure and exact
                    # (integer, bool) mismatches stay fatal.
                    info["eager_allclose_warning"] = str(error)[:1000]
                if port is not None:
                    roundtrip = common.pin_roundtrip(port, tuned["kernels"])
                    info["code_matches_autotune"] = all(roundtrip.values())
                if impl in GATED_IMPLS:
                    # The dropout oracles round their random samples to bf16
                    # before the threshold comparison, as eager does; Inductor
                    # and the ports do not (common.AnchoredGate). Their
                    # random outputs are recorded against Inductor's, not
                    # gated.
                    verdict = numerics_gate().check(
                        replay, gate_random=impl != "oracle"
                    )
                    info["anchored"] = verdict
                    info["numerics_pass"] = bool(verdict["pass"])
                run_inputs = inputs()
                fns[impl] = lambda call=call, run_inputs=run_inputs: call(run_inputs)
                info["status"] = "ok"
            except Exception:
                info["error"] = traceback.format_exc()
                print(f"[{impl}] {info['error']}", file=sys.stderr, flush=True)
        record["foreign_gpu_processes_before"] = common.foreign_gpu_processes()
        record["temperature_before_c"] = common.gpu_temperature()
        # interleaved_bench captures graphs of its own: no garbage collection
        # while timing (see common.capture_graph).
        gc.collect()
        with torch.no_grad(), common.gc_paused():
            timings = time_impls(
                fns,
                rounds=args.rounds,
                rep_ms=args.rep_ms,
                cold_rounds=args.cold_rounds,
            )
        for impl, values in timings.items():
            info = record["impls"][impl]
            info.update(values)
            if "timing_error" in values:
                info["status"] = "error"
                continue
            if total_bytes:
                for key in ("bb_us", "cold_us", "kernel_us"):
                    us = values.get(key)
                    if us and us > 0:
                        info[key.replace("_us", "_gbps")] = total_bytes / us / 1e3
        record["temperature_after_c"] = common.gpu_temperature()
        record["foreign_gpu_processes_after"] = common.foreign_gpu_processes()
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
    parser.add_argument("--configs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--impls", default=",".join(ALL_IMPLS))
    # Even round counts: as many rounds in each order (module docstring).
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--rep-ms", type=int, default=200)
    parser.add_argument("--cold-rounds", type=int, default=6)
    args = parser.parse_args()
    for flag, count in (("--rounds", args.rounds), ("--cold-rounds", args.cold_rounds)):
        if count <= 0 or count % 2:
            parser.error(f"{flag} {count}: a positive, even count is needed")
    activity: dict[str, Any] = {}
    with common.watch_foreign_gpu_activity(activity):
        record = run(args)
    record.update(activity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(args.output, record)
    summary = {
        impl: (
            f"{info['bb_us']:.2f}/{info['cold_us']:.2f}us"
            if "bb_us" in info
            else info["status"]
        )
        for impl, info in record["impls"].items()
    }
    print(f"{record['status']} {args.repro_id}[{record['shape']}] {summary}")
    if record["status"] != "ok":
        print(record.get("error", ""), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
