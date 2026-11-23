# ruff: noqa: ANN401
"""Time one better-benchmark repro under every implementation, interleaved.

Implementations (``--impls``):

- ``inductor``: ``torch.compile`` of the repro, default Inductor config
- ``inductor_cd``: the same with ``coordinate_descent_tuning=True``
- ``inductor_max``: the same with the options of
  ``torch.compile(mode="max-autotune")`` (``max_autotune`` and
  ``coordinate_descent_tuning``; ``common.inductor_options`` says why the
  mode's CUDA-graph wrapper is left out). The comparison is against the
  fastest Inductor compile of the record
- ``oracle``: the repro's hand-written reference kernel (``oracle.py``)
- ``helion_triton`` / ``helion_cute``: the Helion port with the kernels pinned
  to the decorators a cold full autotune reported (``autotune.py``); the
  record names that autotune record's code digests and whether it was
  provisional (``autotune_code_sha256``, ``autotune_code_nosrc_sha256``
  where the record holds the digests without source-line comments,
  ``autotune_provisional``), so ``report.py`` tells the timing from one of
  a later autotune of the port. It names them before the port loads
  (``pinned_port``): an entry without them found no ok autotune record,
  whatever else failed. An entry that ran also says whether the pinned
  kernels generated the code the record holds (``code_matches_autotune``,
  ``common.pin_roundtrip``): the same text but for its source-line
  comments, or the same whole text for a record made before autotune
  records held the digest without them
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
- ``cold_us``: the regime Helion's autotuner searches in (CuTe's final
  selection times trace kernel durations instead, see the README),
  ``interleaved_bench`` with the L2 flushed before every call inside a
  batched CUDA graph and the flush-only time subtracted (no launch-latency
  quantization), the median over each order's rounds
  (``cold_order_medians_us``), averaged. ``cold_runs_us``
  keeps every round; ``report.py --rebalance`` recomputes both timings from
  the rounds of records made before (whose ``cold_us`` was the median of 5
  rounds, a forward-order reading).
- ``kernel_us``: summed device time of the kernels one replay launches
  (torch.profiler; back-to-back replays, so L2-warm), with per-kernel names
  and times (``kernels``). A kernel's time is its step, the end of its
  profiler span minus the later of its start and the latest end before it
  (``kernel_steps``): the spans of dependent launches overlap, and their
  sum counted the overlap twice. ``kernel_span_us`` and each kernel's
  ``span_us`` keep the spans.
- ``rot_us`` (only with ``--rotate-inputs N``): ``cold_us``'s regime on
  inputs no earlier call left in L2. ``cold_us`` replays every call on the
  same input tensors, and lines loaded with an ``L2::evict_last`` hint
  (Triton's ``eviction_policy="evict_last"``, which Inductor emits on the
  operands it expects to read again; CuTe's ``l2_last`` load policies) stay
  in L2 through the 256 MB flush, so an implementation that hints its loads
  reads part of its inputs warm there. Here every implementation gets N
  private input sets of its own (one implementation's hinted lines cannot
  serve another's reads) and is timed as a graph of N x [L2 flush; call on
  the next set] minus a graph of the N flushes (``rotating_bench``), in
  ``--cold-rounds`` rounds that alternate the order like ``cold_us``'s:
  ``rot_runs_us``, ``rot_order_medians_us`` and ``rot_sets`` go with it. A
  cross-check for ``cold_us``, not a proof of coldness (see the README);
  only for ports whose inputs are small enough for hinted lines to matter
  and for N sets per implementation to fit, and that ``cold_us`` times in
  batches too (``rotation_skip_reason``; the record's ``rotate_inputs`` says
  what was decided and why).

The record says what ran right behind each Helion port's batch in the rounds
of each order (``behind_helion``) and, with a warning, which Inductor
compiles a listing puts there, or first in an order that a port ends
(``carryover_risk``): on some ports the batch right behind a Helion port's
reads up to a microsecond a call higher than behind anything else (README,
Timers), and the listing of ``common.ALL_IMPLS`` keeps the oracle, the
memcopy and the other port there.

The record also holds the baseline two ways (``fastest_inductor``,
``common.fastest_inductor``): the fastest of the Inductor compiles, which
the reports divide by, and the same without what choosing among noisy
readings takes off.

The entries of the Inductor compiles name the config that each of
their Triton kernels ended on (``inductor_configs``,
``common.inductor_kernel_configs``): coordinate descent tunes again in every
process and its pick moves a reading. The buffers a process times lie at a
placement drawn for that process (``Placement``; ``placement`` in the
record), because a small kernel's reading depends on it and the allocation
order of a process would otherwise repeat one placement, another for each
place in ``--impls``. ``campaign.py --bench-repeats`` takes several
processes of a point for both reasons (README, Baseline).

Correctness is checked on what the timers run, a replay of the
implementation captured in a CUDA graph (the oracles of the dropout repros
take another path when they are not captured). The replay starts from
poisoned memory, and an implementation that leaves part of an output
unwritten is an error and is not timed (``_replayed``,
``common.check_written``). Every implementation must
match eager PyTorch's output count, shapes and dtypes, and eager's integer
outputs exactly (except those that depend on random numbers). An integer or
boolean output that the default Inductor compile itself does not reproduce
(eager leaves the order of ``topk``'s tied values to its implementation, and
Inductor's decomposition returns them in index order) may equal that
compile's output instead, element for element. The implementation's entry
then says so (``exact_vs_inductor``: the output, its size, and in how many
elements the implementation's and Inductor's outputs differ from eager's);
the default compile is its own reference there, and an oracle that makes
eager's ATen call for such an op equals eager as before. The Helion
ports must also match eager's strides; Inductor and the oracles may differ
(Inductor pads some output strides), which ``errors_vs_eager`` records. Float
outputs outside eager's loose tolerance (rtol = atol = 0.1) are only noted
(``eager_allclose_warning``): eager rounds every intermediate the graph
computes in bf16, so it can sit far from Inductor and every other fp32
implementation (pointwise_238e96847398 evaluates its whole
cancellation-dominated graph in bf16; sum_sum_sum_d2c97d17f3dc rounds
intermediates to bf16 and its outputs reach 1e18). The coordinate-descent
and max-autotune compiles, the oracle and the Helion ports also go through
the anchored numerics gate (``common.AnchoredGate``, a loosened variant of
better-benchmark's) with the default Inductor compile as the error
baseline, which compares the outputs that depend on random numbers with
that compile's. The default compile goes through it too: judged against itself,
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

The Inductor kernels are compiled by the Triton of the environment the
bench runs in, and the record names it (``triton_backend``): the
distribution, its version and commit, and the switches in effect, Meta's
warp specialization in fbtriton (``TRITON_USE_META_WS``) and the Tile IR
backend of the Triton-to-tile-IR fork (``ENABLE_TILE``). A bench of such a
baseline never passes for one when it is not (the README's Other Triton
builds): it ends before compiling when its switch cannot be in effect or
its cache directories hold another build's kernels (``preflight``), and an
Inductor compile whose launched binaries were not compiled by this Triton
through the switch is an error (``common.artifact_issue``;
``inductor_configs`` names each binary's ``artifact``). Where that is the
default compile, the numerics gate has no anchor, as where that compile
fails: no implementation is judged by another build's kernel, whether or
where ``--impls`` lists ``inductor``.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import math
from pathlib import Path
import random
import secrets
import statistics
import sys
import time
import traceback
from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import cast

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.cute.better_benchmark import common
import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile

from helion.autotuner.benchmarking import _BATCHED_REPLAY_BELOW_US
from helion.autotuner.benchmarking import _batched_flushed_replays
from helion.autotuner.benchmarking import _make_l2_cache_clearer
from helion.autotuner.benchmarking import _positive_median
from helion.autotuner.benchmarking import interleaved_bench
from helion.autotuner.benchmarking import steady_graph_dispatch

# Judged by the gate against the default Inductor compile ("inductor"),
# which the gate judges too: against itself, that fails only an output that
# is not finite where the reference is (common._anchored_leaf).
GATED_IMPLS = (*common.INDUCTOR_IMPLS, "oracle", "helion_triton", "helion_cute")

# What the rounds of ``cold_us`` and of ``rot_us`` share. Untimed replays run
# for ``SETTLE_S`` before the timed rounds: the GPU idles while the
# implementations are checked and while the input sets are made, and a
# power-capped one then starts on a higher clock than it holds under the
# rounds. ``COLD_REPEAT`` is the ``repeat`` of a round's ``interleaved_bench``
# (``cold_round``), its sample count before batching.
SETTLE_S = 0.5
COLD_REPEAT = 100

# ``--rotate-inputs`` applies to ports whose inputs total at most this many
# times the device's L2 (``--rotate-max-l2``). A hint keeps at most an L2's
# worth of lines resident, so beyond twice the L2 less than half of a port's
# inputs can be warm whatever it hints, while the private sets would cost
# sets x implementations x input bytes of device memory: the selection's
# multi-GB ports (4.3 GB of inputs) would need 206 GB for 8 sets of 6
# implementations, more than a B200 has. On a B200 (L2: 126.5 MiB,
# ``torch.cuda.get_device_properties(0).L2_cache_size``) the default admits
# inputs up to 265 MB, at most 12.7 GB of sets.
ROTATE_MAX_L2_MULTIPLE = 2.0
# ... and whose private sets fit this share of the free device memory (a
# device with less memory than a B200, or one that other work fills).
ROTATE_MEMORY_FRACTION = 0.25
# ... and that ``cold_us`` times in batches, as ``rot_us`` is timed: from this
# per-call time of the fastest implementation on, ``interleaved_bench`` times
# single replays instead, each with a graph launch (about 4.5 us on a B200)
# and with idle time between the calls, and the two timers would differ by
# more than the rotation: by the launch, and by the clock a power-capped GPU
# holds under each load. On any_amax_sum_e232f7a3d0d8 (201 MB of inputs, a
# 750 W-capped B200) Inductor read 88 to 101 us in ``cold_us`` from one
# process to the next and 90 to 103 us in batches of 8 as the SM clock moved
# between 1485 and 1935 MHz under the cap (84 us at 1965 MHz before the cap
# set in), the memcopy 61 us throughout once the launch is taken off.
ROTATE_BELOW_US = _BATCHED_REPLAY_BELOW_US
# The fewest samples of a round that times batches (``rotating_samples``).
BATCHED_MIN_SAMPLES = 20


def kernel_steps(spans: list[tuple[float, float]]) -> list[float]:
    """What each of ``spans`` (the start and end of a kernel in the profile,
    in the order they started) adds to the time its stream is busy: its end
    minus the later of its start and the latest end before it, and never
    less than nothing.

    For kernels that run one after the other that is the span. A kernel
    launched as a programmatic dependent launch (CuTe's ``cute_pdl``) starts
    its span while its predecessor still runs, where it waits for the
    predecessor's writes, so its span holds the rest of the predecessor's
    run too (the README has a port's numbers)."""
    steps = []
    done = -math.inf
    for start, end in spans:
        steps.append(max(end - max(start, done), 0.0))
        done = max(done, end)
    return steps


def _profile_kernels(replay: Callable[[], None], calls: int = 10) -> dict[str, Any]:
    """The device time of the kernels one replay launches, from a profile of
    ``calls`` back-to-back replays (L2-warm): ``kernels`` lists each kernel
    name's ``us``, the median of its steps (``kernel_steps``), the median of
    its profiler spans (``span_us``) and its launches a call, by their share
    of ``kernel_us``, the steps' total for one call. ``kernel_span_us`` is
    the spans' total, which counts the overlap of dependent launches twice
    (what ``kernel_us`` was before the steps); ``kernel_spans_overlap`` says
    whether the two differ."""
    replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            replay()
        torch.cuda.synchronize()
    events = sorted(
        (
            (event.time_range.start, event.time_range.end, event.name)
            for event in prof.events() or []
            if event.device_type == torch.autograd.DeviceType.CUDA
        ),
    )
    steps = kernel_steps([(start, end) for start, end, _ in events])
    per_name: dict[str, list[tuple[float, float]]] = {}
    for (start, end, name), step in zip(events, steps, strict=True):
        per_name.setdefault(name, []).append((step, end - start))
    kernels = []
    total = total_span = 0.0
    for name, timings in per_name.items():
        per_call = len(timings) / calls
        step = statistics.median(step for step, _ in timings)
        span = statistics.median(span for _, span in timings)
        total += step * per_call
        total_span += span * per_call
        kernels.append(
            {"name": name[:200], "us": step, "span_us": span, "per_call": per_call}
        )
    kernels.sort(key=lambda k: -k["us"] * k["per_call"])
    return {
        "kernel_us": total,
        "kernel_span_us": total_span,
        "kernel_spans_overlap": common.spans_overlap(total, total_span),
        "kernels": kernels,
    }


def _replayed(
    call: Callable[[list[Any]], Any], fresh_inputs: Callable[[], list[Any]]
) -> Callable[..., Any]:
    """``call`` run the way the timers run it: warmed up eagerly (on other
    inputs: a repro can write into its inputs), then captured in a CUDA graph
    on the given inputs and replayed once. The checks judge this, not an
    eager call, because an implementation may take another path under
    capture: the oracles of the dropout repros draw their random numbers in
    the kernel from Inductor's seed (``tl.rand``) while capturing, and load
    a random tensor matching eager's otherwise.

    The replay runs on poisoned memory (``common.POISON_BYTE``; ``poison``
    names another byte, for ``common.check_written``). The capture allocates
    the call's outputs and writes nothing, so between the capture and the
    replay the outputs that the call allocated are filled with the byte, and
    so is the memory the allocator holds free, the buffers that the captured
    call freed among it: what the replay does not write is then that byte,
    not the zeros of fresh device memory or what an earlier call left."""

    def run(inputs: list[Any], poison: int = common.POISON_BYTE) -> Any:
        call(fresh_inputs())
        torch.cuda.synchronize()
        held = common.allocated_blocks()
        graph, outputs = common.capture_graph(lambda: call(inputs))
        common.poison_free_memory(poison)
        common.poison_new_outputs(outputs, held, poison)
        graph.replay()
        torch.cuda.synchronize()
        return outputs

    return run


def pinned_port(
    info: dict[str, Any], configs_dir: Path, repro_id: str, shape: str, backend: str
) -> tuple[Any, dict[str, Any]]:
    """The port of ``repro_id`` on ``backend`` with its kernels pinned to
    their autotune record, and that record. ``info``, the port's entry in
    the bench record, names the record first (its path, its decorators and
    ``common.timed_record_fields``), so an entry of a port that then fails
    to load or to run still says what it was pinned to, and an entry
    without those fields says that the run had no ok record
    (``common.timed_on_record``)."""
    info["autotune_record"] = str(
        common.config_record_path(configs_dir, repro_id, shape, backend)
    )
    tuned = common.autotune_record(configs_dir, repro_id, shape, backend)
    info["decorators"] = common.per_kernel(tuned["kernels"], "decorator")
    info.update(common.timed_record_fields(tuned))
    return common.load_port(repro_id, backend, pinned=info["decorators"]), tuned


def do_bench(fn: Callable[[], object], **kwargs: object) -> object:
    """``triton.testing.do_bench``, imported when first called: the test
    module imports this one on lanes whose torch build ships no Triton."""
    from triton.testing import do_bench as triton_do_bench

    return triton_do_bench(fn, **kwargs)


def cold_round(fns: list[Callable[[], Any]]) -> list[float]:
    """One round of ``cold_us``: the per-call time of each of ``fns`` in
    milliseconds, by ``interleaved_bench`` on CUDA graphs with the L2 flushed
    before every call."""
    return interleaved_bench(
        fns, repeat=COLD_REPEAT, default_cudagraph=True, l2_flush_mode="call"
    )


def settle(replay: Callable[[], object]) -> None:
    """Run ``replay`` untimed for ``SETTLE_S``, which settles a power-capped
    GPU on the clock it sustains under the rounds that follow."""
    started = time.perf_counter()
    while time.perf_counter() - started < SETTLE_S:
        replay()


def alternating_rounds(
    names: list[str], rounds: int, round_ms: Callable[[list[str]], list[float]]
) -> dict[str, list[float]]:
    """The readings of ``rounds`` rounds over the implementations ``names``,
    in microseconds, round by round for each: ``round_ms`` times them in the
    order it is given (their per-call milliseconds in that order), which is
    ``names`` in even rounds and its reverse in odd ones. Every round-based
    timer takes its rounds through here, so all alternate alike."""
    runs: dict[str, list[float]] = {name: [] for name in names}
    for r in range(rounds):
        order = names if r % 2 == 0 else names[::-1]
        for name, ms in zip(order, round_ms(order), strict=True):
            runs[name].append(ms * 1000.0)
    return runs


def behind_helion(names: list[str]) -> dict[str, dict[str, str | None]]:
    """What runs right behind each Helion port's batch in the rounds of each
    order, for the implementations ``names`` in listing order:
    ``{"forward": {port: follower}, "reverse": {...}}``, the follower None
    where the port is the last of that order: the first implementation of
    the order follows it there, at the start of the next sample, behind the
    flush-only graph where the round times batches (ports whose fastest
    implementation takes under 64 us a call; a longer port is timed replay
    by replay, each behind a flush of its own, and has no such graph).

    On some ports the batch right behind a Helion port's reads up to a
    microsecond a call higher than behind any other implementation (README,
    Timers), and the two orders give a port its two listing neighbours as
    followers and no other."""
    return {
        label: {
            name: order[place + 1] if place + 1 < len(order) else None
            for place, name in enumerate(order)
            if name.startswith("helion_")
        }
        for label, order in (("forward", names), ("reverse", names[::-1]))
    }


def carryover_risk(names: list[str]) -> list[str]:
    """The Inductor compiles that ``names`` lists so that their batch is the
    next one replayed after a Helion port's in some round: right behind it
    (``behind_helion``), each as ``"<compile> behind <port> in <order>
    rounds"``, or, where the port is the last of an order, as the first of
    that order: ``"<compile> behind <port> at the start of the next sample
    in <order> rounds"`` (with the flush-only graph of a batched round
    between them, up to a quarter of what the port leaves was still read:
    README, Timers). Their readings, and with them the
    fastest Inductor compile's, can be high by what the port leaves behind.
    ``common.ALL_IMPLS`` has the oracle between the compiles and the ports
    and the memcopy last: no risk."""
    orders = {"forward": names, "reverse": names[::-1]}
    risks = []
    for label, followers in behind_helion(names).items():
        for port, follower in followers.items():
            later = follower is None
            if later:
                # The last of an order: its first follows, a sample later.
                follower = orders[label][0]
            if follower in common.INDUCTOR_IMPLS:
                when = "at the start of the next sample " if later else ""
                risks.append(f"{follower} behind {port} {when}in {label} rounds")
    return risks


def timed_implementations(timings: dict[str, dict[str, Any]]) -> list[str]:
    """The implementations that ``time_impls`` timed, in listing order:
    those of ``timings`` without a ``timing_error``. The rounds are over
    these alone, so they decide who runs behind whom (``note_carryover``),
    and the rotating pass times the same."""
    return [impl for impl, values in timings.items() if "timing_error" not in values]


def note_carryover(record: dict[str, Any], timed: list[str]) -> None:
    """Put into a bench ``record`` what ran right behind each Helion port in
    the rounds over the implementations ``timed`` (``behind_helion``) and
    which Inductor compiles that listing puts there (``carryover_risk``),
    and warn of each of those on stderr: its reading, and with it the
    fastest Inductor compile's, is not the one a listing with the oracle
    between them gives."""
    record["behind_helion"] = behind_helion(timed)
    record["carryover_risk"] = carryover_risk(timed)
    for risk in record["carryover_risk"]:
        print(
            f"[carryover] {risk}: a batch right behind a Helion port's can "
            "read up to 1 us a call high (a quarter of that behind the "
            "flush-only graph of a batched round), and the fastest Inductor "
            "compile with it. "
            "List the oracle between the Inductor compiles and the Helion "
            "ports and the memcopy last, as common.ALL_IMPLS does (README, "
            "Timers).",
            file=sys.stderr,
            flush=True,
        )


def cold_fields(prefix: str, runs: list[float]) -> dict[str, Any]:
    """What a cold timer records of an implementation's rounds, for
    ``cold_us`` and ``rot_us`` alike: ``<prefix>_us``, the median over each
    order's rounds, averaged (``common.order_balanced``), then those medians
    (``<prefix>_order_medians_us``) and the rounds (``<prefix>_runs_us``)."""
    value, orders = common.order_balanced(runs, statistics.median)
    return {
        f"{prefix}_us": value,
        f"{prefix}_order_medians_us": orders,
        f"{prefix}_runs_us": runs,
    }


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
    settle(lambda: [graphs[name].replay() for name in names])
    torch.cuda.synchronize()

    def bb_ms(name: str) -> float:
        ms = do_bench(graphs[name].replay, warmup=25, rep=rep_ms, return_mode="min")
        return float(cast("float", ms))

    bb_runs = alternating_rounds(
        names, rounds, lambda order: [bb_ms(name) for name in order]
    )
    cold_runs = alternating_rounds(
        names, cold_rounds, lambda order: cold_round([fns[name] for name in order])
    )
    for name in names:
        bb_us, bb_order = common.order_balanced(bb_runs[name], min)
        out[name].update(
            bb_us=bb_us,
            bb_order_mins_us=bb_order,
            bb_runs_us=bb_runs[name],
            **cold_fields("cold", cold_runs[name]),
        )
        out[name].update(_profile_kernels(graphs[name].replay))
    return out


# A placement spacer is this many 2 MiB pages at most (``Placement``, which
# says how few places that amounts to).
PLACEMENT_PAGE_BYTES = 2 << 20
PLACEMENT_MAX_PAGES = 128


class Placement:
    """Where the buffers that a process times lie in device memory, drawn
    from ``seed`` (None: where the process's allocation order puts them, as
    in the records made before this).

    A small kernel's reading depends on where its inputs and outputs lie.
    One CuTe config of var_mean_c3ea8f7ce5c6 (48 MB of traffic) read
    differently on six input sets of the same values in one process, each
    timed after every other in turn, and so did Inductor's kernel on its
    six, the same sets slow and fast for both. The allocation order of a
    process is the same every time, so the same placement comes back in
    every process and reads as an effect of the implementation's place in
    ``--impls``: the CuTe port of that point read lower listed fourth than
    listed second. What ran before a kernel was not it there: the same
    graph took the same time after each of the other implementations' (with
    the picks of that time; ``behind_helion`` says when it can be). The
    readings are in README.md ("Baseline: several bench processes").

    ``shift`` releases the allocator's cached blocks and holds a block of a
    random number of 2 MiB pages, so the buffers allocated next come from
    the driver, somewhere else in every process: before each
    implementation's timed inputs and before the timers capture their
    graphs. One process still times one placement of each implementation;
    the placements of several processes differ (``campaign.py
    --bench-repeats``).

    The draw is of few places. The driver lays device memory out in chunks
    of 32 MiB (a B200): the first buffer after a spacer of p pages lay 16 *
    ceil(p / 16) pages below where it lies without one (and in place after
    a spacer of 5 to 8 pages, which the driver put elsewhere). So a buffer
    of 32 MiB or more lands on one of 9 chunk starts over the 128 spacer
    sizes, and a smaller one also on a page inside the spacer's last chunk
    where that leaves it room (15 places for a 3 MB buffer). The record
    therefore names what the draw came to, not the draw alone
    (``placement``: each spacer's pages and device address, and the
    addresses of the storages that each implementation's timed call
    reads, ``timed_inputs``; the latter also under ``--fixed-placement``)."""

    def __init__(self, seed: int | None) -> None:
        self.seed = seed
        self._random = None if seed is None else random.Random(seed)
        self._spacers: list[torch.Tensor] = []
        self._timed_inputs: dict[str, list[int]] = {}

    def shift(self) -> None:
        if self._random is None:
            return
        torch.cuda.empty_cache()
        pages = self._random.randrange(1, PLACEMENT_MAX_PAGES + 1)
        self._spacers.append(
            torch.empty(pages * PLACEMENT_PAGE_BYTES, dtype=torch.uint8, device="cuda")
        )

    def placed(self, impl: str, inputs: list[Any]) -> None:
        """Note where the tensors that ``impl``'s timed call reads lie: the
        address of each distinct storage among ``inputs``."""
        self._timed_inputs[impl] = list(common.input_storages(inputs))

    def record(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "spacer_pages": [
                spacer.numel() // PLACEMENT_PAGE_BYTES for spacer in self._spacers
            ],
            "spacer_addresses": [spacer.data_ptr() for spacer in self._spacers],
            "timed_inputs": self._timed_inputs,
        }


class Rotating(NamedTuple):
    """What ``--rotate-inputs`` needs of an implementation: its call on an
    input set, what makes it a fresh, private set, and the bytes of one set
    (None: the port's inputs)."""

    call: Callable[[list[Any]], Any]
    fresh: Callable[[], list[Any]]
    set_bytes: int | None = None


def rotation_skip_reason(
    input_bytes: int,
    sets_bytes: int,
    *,
    fastest_impl: str,
    fastest_us: float,
    l2_bytes: int,
    free_bytes: int,
    max_l2_multiple: float = ROTATE_MAX_L2_MULTIPLE,
) -> str | None:
    """Why the rotating timer leaves a port out, or None when it applies:
    ``input_bytes`` is one input set of the port, ``sets_bytes`` what the
    private sets of every timed implementation take together,
    ``fastest_impl`` the fastest of them by ``cold_us`` (the memcopy, where
    it is timed and the ports take longer: ``interleaved_bench`` sizes
    ``cold_us``'s batches by the fastest function it is given too) and
    ``fastest_us`` its ``cold_us``, ``l2_bytes`` the device's L2 size and
    ``free_bytes`` its free memory (``ROTATE_MAX_L2_MULTIPLE``,
    ``ROTATE_BELOW_US``, ``ROTATE_MEMORY_FRACTION``)."""
    limit = max_l2_multiple * l2_bytes
    if input_bytes > limit:
        return (
            f"inputs of {input_bytes / 1e6:.1f} MB exceed {max_l2_multiple:g} x "
            f"the {l2_bytes / 1e6:.1f} MB L2 ({limit / 1e6:.1f} MB), the limit "
            "for private input sets (--rotate-max-l2)"
        )
    if fastest_us >= ROTATE_BELOW_US:
        return (
            f"the fastest implementation, {fastest_impl}, takes "
            f"{fastest_us:.1f} us a call: from "
            f"{ROTATE_BELOW_US:g} us cold_us times single replays, not the "
            "batches of flushed calls that rot_us would time"
        )
    budget = ROTATE_MEMORY_FRACTION * free_bytes
    if sets_bytes > budget:
        return (
            f"the private input sets need {sets_bytes / 1e9:.2f} GB, over "
            f"{ROTATE_MEMORY_FRACTION:.0%} of the {free_bytes / 1e9:.2f} GB of "
            "free device memory"
        )
    return None


def rotating_bench(
    fns: list[Callable[[], object]],
    clear_cache: Callable[[], None],
    *,
    calls: int,
    repeat: int,
) -> list[float]:
    """One round of the rotating timer: the per-call time of each of ``fns``
    in milliseconds, timed the way ``interleaved_bench`` times a batch in
    the per-call flushed regime, with batches of exactly ``calls`` calls.

    Each function is captured as one CUDA graph of ``calls`` x
    [``clear_cache``; ``fn()``] and the flushes alone as another
    (``helion.autotuner.benchmarking``'s capture, which ``interleaved_bench``
    uses). A sample replays every function's graph, in the order given, and
    then the flush-only graph, each between a pair of CUDA events; a
    function's time is the median over the samples of (its replay - the
    flush replay of the same sample) / ``calls``, the positive samples'
    when that median is not positive. ``fns`` are stateful: each call reads
    the next of the function's input sets, so a graph of as many calls as
    there are sets reads each set once per replay.

    ``interleaved_bench`` itself cannot time them: it sizes a batch from the
    kernel's time (1 to 16 calls, a single call above 64 us), so a graph
    would rotate through that many of the sets, or replay one."""
    built = _batched_flushed_replays(fns, clear_cache, calls)
    if built is None:
        raise RuntimeError(f"could not capture graphs of {calls} flushed calls")
    replays, flush_replay = built
    assert flush_replay is not None
    timed = [*replays, flush_replay]
    events = [
        [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(repeat)
        ]
        for _ in timed
    ]
    for sample in range(repeat):
        for replay, pairs in zip(timed, events, strict=True):
            start, end = pairs[sample]
            start.record()
            replay()
            end.record()
    torch.cuda.synchronize()
    *elapsed, flush = (
        [start.elapsed_time(end) for start, end in pairs] for pairs in events
    )
    return [
        _positive_median(
            [
                (ms - flush_ms) / calls
                for ms, flush_ms in zip(samples, flush, strict=True)
            ]
        )
        for samples in elapsed
    ]


def rotating_samples(calls: int) -> int:
    """The samples of one ``rot_us`` round, each a replay of a batch of
    ``calls`` calls: as many as a ``cold_us`` round (``cold_round``) takes of
    a batch that long, so both timers average as many executions of a
    kernel. ``interleaved_bench`` has that rule inside it (its ``repeat``
    over the batch size, and at least 20 samples); the module's device test
    holds this count against the samples it takes."""
    return max(min(COLD_REPEAT, BATCHED_MIN_SAMPLES), math.ceil(COLD_REPEAT / calls))


def _on_the_next_set(
    call: Callable[[list[Any]], Any], sets: list[list[Any]]
) -> Callable[[], object]:
    """``call`` on the next of ``sets`` each time, round and round."""
    inputs = itertools.cycle(sets)
    return lambda: call(next(inputs))


def time_rotating(
    calls: dict[str, Callable[[list[Any]], Any]],
    sets: dict[str, list[list[Any]]],
    *,
    rounds: int,
) -> dict[str, dict[str, Any]]:
    """``rot_us`` of every implementation: ``calls[impl]`` on the input sets
    ``sets[impl]`` (as many for every implementation, private to it), one
    set after the other, in ``rounds`` rounds of ``rotating_bench`` that
    alternate the order of the implementations as ``time_impls``'s cold
    rounds do (``alternating_rounds``), after untimed ones (``settle``); the
    median over each order's rounds, averaged (``cold_fields``)."""
    names = list(calls)
    (count,) = {len(sets[name]) for name in names}
    fns: dict[str, Callable[[], object]] = {}
    for name in names:
        fns[name] = _on_the_next_set(calls[name], sets[name])
        # Every set once, eagerly: nothing compiles or allocates for the
        # first time inside a capture.
        for _ in range(count):
            fns[name]()
    torch.cuda.synchronize()
    clear_cache = _make_l2_cache_clearer()
    samples = rotating_samples(count)

    def rotating_round(order: list[str]) -> list[float]:
        return rotating_bench(
            [fns[name] for name in order], clear_cache, calls=count, repeat=samples
        )

    settle(lambda: rotating_round(names))
    runs = alternating_rounds(names, rounds, rotating_round)
    return {
        name: {**cold_fields("rot", runs[name]), "rot_sets": count} for name in names
    }


def rotate_inputs(
    args: argparse.Namespace,
    record: dict[str, Any],
    rotating: dict[str, Rotating],
    input_bytes: int,
) -> None:
    """The ``--rotate-inputs`` pass over the implementations the other
    timers timed (``rotating``; ``input_bytes``: one set of the port's
    inputs): decide whether the port takes it (``rotation_skip_reason``),
    make the private sets, time them (``time_rotating``) and record
    ``rot_us`` beside the other timings. The decision and its numbers go to
    ``record["rotate_inputs"]``, and so does a failure of the pass: the
    other timings stand without it."""
    count = args.rotate_inputs
    decision: dict[str, Any] = {
        "sets": count,
        "rounds": args.cold_rounds,
        "input_bytes": input_bytes,
        "max_l2_multiple": args.rotate_max_l2,
        "status": "error",
    }
    record["rotate_inputs"] = decision
    # Everything that can fail from here on is inside the try, the device
    # queries included: the record keeps the timings it has.
    try:
        fastest = min(
            rotating, key=lambda name: record["impls"][name]["cold_us"], default=None
        )
        free_bytes, _total = torch.cuda.mem_get_info()
        decision.update(
            fastest_impl=fastest,
            fastest_cold_us=(
                None if fastest is None else record["impls"][fastest]["cold_us"]
            ),
            sets_bytes=count
            * sum(
                input_bytes if impl.set_bytes is None else impl.set_bytes
                for impl in rotating.values()
            ),
            l2_cache_bytes=torch.cuda.get_device_properties(0).L2_cache_size,
            free_bytes=free_bytes,
        )
        reason = (
            "no implementation was timed"
            if fastest is None
            else rotation_skip_reason(
                input_bytes,
                decision["sets_bytes"],
                fastest_impl=fastest,
                fastest_us=decision["fastest_cold_us"],
                l2_bytes=decision["l2_cache_bytes"],
                free_bytes=free_bytes,
                max_l2_multiple=args.rotate_max_l2,
            )
        )
        if reason is not None:
            decision.update(status="skipped", skipped=reason)
            return
        sets = {
            name: [impl.fresh() for _ in range(count)]
            for name, impl in rotating.items()
        }
        torch.cuda.synchronize()
        gc.collect()
        with torch.no_grad(), common.gc_paused():
            timings = time_rotating(
                {name: impl.call for name, impl in rotating.items()},
                sets,
                rounds=args.cold_rounds,
            )
        total_bytes = record.get("total_bytes")
        for name, values in timings.items():
            info = record["impls"][name]
            info.update(values)
            if total_bytes and values["rot_us"] > 0:
                info["rot_gbps"] = total_bytes / values["rot_us"] / 1e3
        decision["status"] = "ok"
    except Exception:
        decision["error"] = traceback.format_exc()
        print(f"[rotate-inputs] {decision['error']}", file=sys.stderr, flush=True)


def preflight(backend: dict[str, Any] | None) -> dict[str, str]:
    """What a bench checks before it compiles anything, so that a run
    started as a baseline (``TRITON_USE_META_WS=1`` under fbtriton,
    ``ENABLE_TILE=1`` under the Triton-to-tile-IR fork) is never a stock
    run in disguise: the switch can be in effect in the installed Triton
    (``backend``: ``common.triton_backend``;
    ``common.check_triton_switches``), and the cache directories are this
    Triton build's and switch state's alone (``common.claim_caches``, whose
    directories it returns). ``common.TritonSwitchError`` otherwise. Each
    Inductor compile's launched kernels are held against the build once
    they exist (``common.artifact_issue``, in ``run``)."""
    common.check_triton_switches(backend)
    return common.claim_caches(backend)


def new_record(
    args: argparse.Namespace, backend: dict[str, Any] | None
) -> dict[str, Any]:
    """A bench's record before anything ran, with the status of a bench
    that did not finish: its point (the default shape where none was
    given), the GPU, the versions, and the Triton build that compiles the
    Inductor kernels with the switches in effect (``backend``: the
    process's ``common.triton_backend``, which ``main`` asks once)."""
    return {
        "repro": args.repro_id,
        "shape": args.shape or common.default_shape(args.repro_id),
        "status": "error",
        "gpu": common.gpu_info(),
        "versions": common.versions(backend),
        "triton_backend": backend,
        "impls": {},
    }


def run(args: argparse.Namespace, backend: dict[str, Any] | None) -> dict[str, Any]:
    """The bench's record (``new_record``, filled)."""
    impls = [i for i in args.impls.split(",") if i]
    record = new_record(args, backend)
    shape = record["shape"]
    started = time.monotonic()
    seed = args.placement_seed
    if seed is None and not args.fixed_placement:
        seed = secrets.randbits(31)
    placement = Placement(None if args.fixed_placement else seed)
    try:
        torch.cuda.set_device(0)
        torch._dynamo.reset()
        common.bb_path()
        import byte_accounting  # pyrefly: ignore [missing-import]

        def inputs() -> list[Any]:
            return common.make_inputs(args.repro_id, shape, seed=0)

        instance = common.load_repro_module(args.repro_id).Repro()
        first_inputs = inputs()
        # What one set of the port's inputs takes, for --rotate-inputs.
        input_bytes = common.input_bytes(first_inputs)
        with torch.no_grad():
            expected = instance(*first_inputs)
        del first_inputs
        record["outputs"] = [
            {"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype)}
            for t in common.flatten_outputs(expected)
        ]
        total_bytes = int(byte_accounting.count_bytes_effective(instance, inputs()))
        record["total_bytes"] = total_bytes

        fns: dict[str, Callable[[], Any]] = {}
        rotating: dict[str, Rotating] = {}
        default_compile: Callable | None = None
        gate: common.AnchoredGate | None = None
        # Whether the kernels that the default Inductor compile launches
        # were held against this build (common.artifact_issue), and why it
        # was refused if it was: a kernel of another build or switch state
        # anchors no gate, so nothing is judged by it.
        anchor_checked = False
        refused_anchor: str | None = None

        def inductor_default() -> Callable:
            nonlocal default_compile
            if default_compile is None:
                default_compile = common.compile_inductor(
                    args.repro_id, shape, instance
                )
            return default_compile

        def launched_configs(call: Callable) -> list[dict[str, Any]]:
            """The config each Triton kernel of an Inductor compile ended
            on (``common.inductor_kernel_configs``), from a call of its own
            once the compile has settled: the first call compiled, and
            launched the variants a mix-order reduction's compile tries
            its split sizes with."""
            with common.inductor_kernels_launched() as launched, torch.no_grad():
                call(inputs())
            return common.inductor_kernel_configs(launched)

        def check_anchor() -> None:
            """Hold the default compile's kernels against this build before
            the gate is anchored on it, where the listing has not done so:
            it leaves ``inductor`` out, or has another implementation
            before it."""
            nonlocal anchor_checked, refused_anchor
            if backend is not None:
                compiled = inductor_default()
                with torch.no_grad():
                    compiled(*inputs())
                refused_anchor = common.artifact_issue(
                    backend, launched_configs(lambda inp: compiled(*inp))
                )
            anchor_checked = True

        def numerics_gate() -> common.AnchoredGate:
            """The gate, with the default Inductor compile as its baseline."""
            nonlocal gate
            if not anchor_checked:
                check_anchor()
            if refused_anchor is not None:
                raise common.TritonSwitchError(
                    "the numerics gate has no anchor: the default Inductor "
                    f"compile was refused ({refused_anchor})"
                )
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
                    placement.shift()
                    src = torch.empty(n, dtype=torch.float32, device="cuda")
                    dst = torch.empty_like(src)
                    placement.placed(impl, [src, dst])
                    fns[impl] = lambda src=src, dst=dst: torch.add(src, 1, out=dst)
                    # Private sources; the stores keep their one buffer, as
                    # the outputs of the other implementations' calls do.
                    rotating[impl] = Rotating(
                        lambda inp, dst=dst: torch.add(inp[0], 1, out=dst),
                        lambda n=n: [
                            torch.empty(n, dtype=torch.float32, device="cuda")
                        ],
                        src.nbytes,
                    )
                    info["status"] = "ok"
                    continue
                port = None
                tuned: dict[str, Any] = {}
                if impl == "inductor":
                    compiled = inductor_default()
                    call = lambda inp, c=compiled: c(*inp)  # noqa: E731
                    info["numerics_baseline"] = True
                elif impl in common.INDUCTOR_IMPLS:
                    compiled = common.compile_inductor(
                        args.repro_id, shape, instance, impl=impl
                    )
                    call = lambda inp, c=compiled: c(*inp)  # noqa: E731
                elif impl == "oracle":
                    call = common.load_oracle(args.repro_id, inputs())
                elif impl.startswith("helion_"):
                    port, tuned = pinned_port(
                        info,
                        args.configs_dir,
                        args.repro_id,
                        shape,
                        impl.removeprefix("helion_"),
                    )
                    call = port.forward
                else:
                    raise ValueError(f"unknown implementation {impl}")
                replay = _replayed(call, inputs)
                with torch.no_grad():
                    actual = replay(inputs())
                if impl in common.INDUCTOR_IMPLS:
                    # The config each of the compile's Triton kernels ended
                    # on: coordinate descent tunes again in every process,
                    # and its pick moves a reading (README, Baseline).
                    info["inductor_configs"] = launched_configs(call)
                    # A cached kernel of another Triton build or switch
                    # state is not this run's: an error, not a timing.
                    issue = common.artifact_issue(backend, info["inductor_configs"])
                    if impl == "inductor":
                        # The gate's anchor, held here where it is listed
                        # before whatever the gate judges.
                        anchor_checked, refused_anchor = True, issue
                    if issue is not None:
                        raise common.TritonSwitchError(issue)
                info["errors_vs_eager"] = common.output_errors(actual, expected)
                # Every element of every output must be the replay's own: an
                # implementation that left some unwritten is not timed.
                with torch.no_grad():
                    common.check_written(
                        actual, lambda byte, replay=replay: replay(inputs(), byte)
                    )
                vs_inductor: list[dict[str, Any]] = []
                try:
                    common.check_outputs(
                        actual,
                        expected,
                        rtol=1e-1,
                        atol=1e-1,
                        stochastic=numerics_gate().stochastic,
                        check_strides=port is not None,
                        inductor=lambda: numerics_gate().baseline,
                        vs_inductor=vs_inductor,
                    )
                except common.FloatMismatch as error:
                    # Low-precision eager can sit far from the exact result
                    # (it rounds every intermediate); the anchored gate below
                    # judges floating-point values. Structure and exact
                    # (integer, bool) mismatches stay fatal, except where the
                    # default Inductor compile is the exact reference.
                    info["eager_allclose_warning"] = str(error)[:1000]
                if vs_inductor:
                    info["exact_vs_inductor"] = vs_inductor
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
                placement.shift()
                run_inputs = inputs()
                placement.placed(impl, run_inputs)
                fns[impl] = lambda call=call, run_inputs=run_inputs: call(run_inputs)
                rotating[impl] = Rotating(call, inputs)
                info["status"] = "ok"
            except Exception:
                info["error"] = traceback.format_exc()
                print(f"[{impl}] {info['error']}", file=sys.stderr, flush=True)
        record["foreign_gpu_processes_before"] = common.foreign_gpu_processes()
        record["temperature_before_c"] = common.gpu_temperature()
        # interleaved_bench captures graphs of its own: no garbage collection
        # while timing (see common.capture_graph).
        gc.collect()
        placement.shift()
        record["placement"] = placement.record()
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
        # The rounds were over these: what ran right behind each Helion
        # port among them (README, Timers), and what the rotating pass times.
        timed = timed_implementations(timings)
        note_carryover(record, timed)
        if args.rotate_inputs:
            # After the other timers, which thus run as they do without it.
            rotate_inputs(
                args, record, {impl: rotating[impl] for impl in timed}, input_bytes
            )
        # The baseline two ways: the fastest Inductor compile, and the same
        # without the selection among noisy readings (rot_us included).
        record["fastest_inductor"] = common.fastest_inductor(record["impls"])
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
    parser.add_argument("--impls", default=",".join(common.ALL_IMPLS))
    # Even round counts: as many rounds in each order (module docstring).
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--rep-ms", type=int, default=200)
    parser.add_argument("--cold-rounds", type=int, default=6)
    # One or the other: a seed with --fixed-placement would draw nothing.
    placed = parser.add_mutually_exclusive_group()
    placed.add_argument(
        "--placement-seed",
        type=int,
        default=None,
        metavar="SEED",
        help="the seed that the placement of the timed buffers is drawn from "
        "(Placement; the record names it): give a record's to time its "
        "placement again (default: a fresh one in every process)",
    )
    placed.add_argument(
        "--fixed-placement",
        action="store_true",
        help="leave the timed buffers where the process's allocation order "
        "puts them, the same in every process, as in the records made before "
        "the placement was drawn",
    )
    parser.add_argument(
        "--rotate-inputs",
        type=int,
        default=0,
        metavar="N",
        help="also time every implementation on N private input sets of its "
        "own, one per call of a flushed batch (rot_us, in --cold-rounds "
        "rounds): cold_us without the L2 lines that hinted loads keep through "
        "the flush. Only for ports whose inputs fit --rotate-max-l2 and whose "
        "fastest implementation cold_us times in batches; 8 is the count the "
        "study used (default: off)",
    )
    parser.add_argument(
        "--rotate-max-l2",
        type=float,
        default=ROTATE_MAX_L2_MULTIPLE,
        metavar="X",
        help="--rotate-inputs leaves out ports whose inputs total more than X "
        "times the device's L2 size (default: %(default)s)",
    )
    args = parser.parse_args()
    for flag, count in (("--rounds", args.rounds), ("--cold-rounds", args.cold_rounds)):
        if count <= 0 or count % 2:
            parser.error(f"{flag} {count}: a positive, even count is needed")
    if args.rotate_inputs < 0 or args.rotate_inputs == 1:
        parser.error(
            f"--rotate-inputs {args.rotate_inputs}: at least 2 input sets are "
            "needed (0 turns it off)"
        )
    if args.rotate_max_l2 <= 0:
        parser.error(f"--rotate-max-l2 {args.rotate_max_l2}: a positive factor")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Asked once: it lists every installed distribution.
    backend = common.triton_backend()
    try:
        caches = preflight(backend)
    except common.TritonSwitchError as error:
        # Before anything compiles: the record says why there are no timings.
        record = new_record(args, backend)
        record["error"] = f"TritonSwitchError: {error}"
        common.write_json(args.output, record)
        print(record["error"], file=sys.stderr)
        sys.exit(1)
    activity: dict[str, Any] = {}
    with common.watch_foreign_gpu_activity(activity):
        record = run(args, backend)
    record.update(activity)
    record["cache_directories"] = caches
    common.write_json(args.output, record)
    summary = {
        impl: (
            f"{info['bb_us']:.2f}/{info['cold_us']:.2f}"
            + (f"/{info['rot_us']:.2f}" if "rot_us" in info else "")
            + "us"
            if "bb_us" in info
            else info["status"]
        )
        for impl, info in record["impls"].items()
    }
    print(f"{record['status']} {args.repro_id}[{record['shape']}] {summary}")
    rotation = record.get("rotate_inputs")
    if rotation is not None and rotation["status"] != "ok":
        reason = rotation.get("skipped") or rotation.get("error", "").strip()[-300:]
        print(f"rot_us {rotation['status']}: {reason}")
    if record["status"] != "ok":
        print(record.get("error", ""), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
