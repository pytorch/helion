"""``SpyreBackend``: a Helion backend emitting torch-spyre OpSpecs.

Plan 1.1: the emission seam is the *per-node lowering* one
(inductor_lowering.py:1642-1662), where a codegen result that is not an
``ast.AST`` passes straight through to the interpreter env. So each compute node
returns an ``OpSpec``-producing ``Operand`` rather than an expression string, and
``inductor_op_overrides`` is deliberately left unimplemented -- a node that
reaches Inductor's own pointwise fallback is a bug and must fail loudly.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING
from typing import ClassVar

from ... import exc
from ..backend import Backend
from .emitter import SpyreEmitter

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    import ast
    from collections.abc import Sequence

    # ``torch`` is annotations-only in this module. Every runtime use -- the
    # device-timer probes' ``torch.zeros``/``compile``/``profiler``, and the
    # generated launcher's own import -- already has a function-scope import.
    # Sibling ``_compiler`` modules keep ``torch`` at module level and pass
    # TC002 because they touch it at import time; this one does not.
    import torch

    from ...autotuner.config_spec import BlockSizeSpec
    from ...runtime.config import Config
    from ..backend import InductorOpOverrides
    from ..device_ir import GraphInfo
    from ..inductor_lowering import Lowering
    from ..tile_dispatch import TileStrategyDispatch


#: Widest stick any supported dtype uses: 64 elements at fp16/bf16
#: (torch_spyre._C.get_elem_in_stick), 32 at fp32. Used as the floor for a
#: trailing block size so no config can split a stick.
_MAX_ELEMS_PER_STICK = 64

#: Usable LX scratchpad, in bytes. **A real byte budget, not a buffer count.**
#:
#: Taken from torch-spyre's own constants rather than measured, so the two
#: cannot drift (`scratchpad/allocator.py:207-213`):
#:
#:     _LX_PHYSICAL_CAPACITY_BYTES         = 2 << 20   # 2 MiB
#:     _LX_PROGRAM_DEBUG_RESERVATION_BYTES = 64 << 10  # 64 KiB
#:     _LX_TRACKER_CAPACITY_BYTES          = 2 MiB - 64 KiB
#:
#: scaled by `config.dxp_lx_frac_avail` (default **0.2**, `config.py:164`), the
#: fraction the frontend may use. 0.2 x (2 MiB - 64 KiB) = **406 KB**, which is
#: the figure §9 quoted as "the nominal budget" -- and it turns out to have
#: been the right one all along.
#:
#: ~~_MAX_LX_BUFFERS = 2: a count, not a byte budget; 3 concurrent LX buffers
#: fail at launch at any size.~~ **That was bug #15.** The failures were an
#: *addressing* bug in this backend, not a hardware ceiling: the emitter handed
#: out `{"lx": 0}`, `{"lx": 1}`, `{"lx": 2}` as slot indices while SDSC reads
#: the value as a byte address, so `{"lx": 2}` meant "2 bytes in" -- a
#: misaligned address, which is what the hardware rejected. At properly
#: aligned, non-overlapping addresses **6 concurrent LX buffers run**, and
#: torch-spyre's own softmax uses 4 (`golden/ref_softmax.py`: 0, 256, 1024,
#: 1280). See `repros/lx_allocation_is_an_address.py` and
#: state-of-the-world §8.9.
_LX_CAPACITY_BYTES = int((2 << 20) - (64 << 10)) * 2 // 10

#: Alignment every LX address must satisfy. torch-spyre calls this
#: `_LX_ALLOCATION_GRANULARITY_BYTES` (`allocator.py:214`) and it is **128** --
#: which independently confirms the rule measured from the outside
#: (`k % 128 in {0, 1}` for an accepted `{"lx": k}`). It is the same 128 as
#: `layout.STICK_BYTES`, so the emitter aligns LX the way it already aligns the
#: pool.
_LX_ALIGNMENT_BYTES = 128

#: Upper bound of the ``spyre_lx_buffers`` search. 8 rather than unbounded
#: because the byte budget is the real limit and a request beyond it simply
#: spills -- so a larger range would add configs that behave identically.
#: Measured: 6 concurrent 8 KB buffers run, and 406323 / 65536 = 6 buffers at
#: the 128x256 fp16 size the drivers use.
_LX_SEARCH_MAX = 8

#: Default *residency preference*: how many intermediates to keep in LX rather
#: than spill to `hbm_pool`. A performance knob now, not a correctness ceiling
#: -- the correctness limit is `_LX_CAPACITY_BYTES`, enforced by
#: `SpyreEmitter._lx_fits`. Kept at 2 as the default so an unspecified config
#: reproduces the previously-shipped behaviour; the autotuner may now search
#: above it.
_MAX_LX_BUFFERS = 2

#: A Spyre launch costs milliseconds and the device is a shared, serialized
#: resource, so keep the sample count low: autotuning sweeps dozens of configs.
_BENCH_WARMUP = 1
_BENCH_REPS = 3

#: Launches submitted per timed sample, before one synchronize drains them.
#:
#: A Spyre launch is **asynchronous**: ``spyre_stream.cpp:261`` wraps
#: ``launchOperationCompute``, so timing a call measures the enqueue and the
#: loop overlaps submit(N+1) with compute(N). Timing one call at a time
#: therefore measures the *queue*, not the kernel -- it reports ~26-30 us for
#: matmuls from 0.07 to 4.3 GFLOP, a 64x work range, which at the top end would
#: be 158 TFLOP/s. It also **inverts** the ranking the autotuner needs: on a
#: 512x512 matmul it rates the 1-core config at 18 us and the 16-core config at
#: 23-60 us, when draining shows 181 us against 62 us.
#:
#: Batching before the synchronize amortizes the one-off sync cost while still
#: emptying the queue, so a per-launch figure stays meaningful. 8 is enough to
#: separate every config of that matmul with disjoint IQRs while keeping a
#: sweep affordable: it costs 8 launches per sample, not 8x the wall time,
#: because the submissions pipeline.
_BENCH_BATCH = 8


def _drained(fn: object, batch: int = _BENCH_BATCH) -> float:
    """Microseconds for one launch of ``fn``, with the device queue drained.

    Submits ``batch`` launches, synchronizes once, and divides. The
    synchronize is what makes this a measurement of the device rather than of
    the submission path.
    """
    import time

    import torch

    start = time.perf_counter()
    for _ in range(batch):
        fn()
    torch.accelerator.synchronize()
    return (time.perf_counter() - start) * 1e3 / batch


def _spyre_activity() -> object | None:
    """``ProfilerActivity`` for the Spyre device, by whichever name exists.

    ``PrivateUse1`` and ``SPYRE`` are the **same enum value** here (verified:
    they compare equal), so the order is only about which attribute a given
    torch build exposes.
    """
    import torch

    activity = torch.profiler.ProfilerActivity
    return getattr(activity, "PrivateUse1", None) or getattr(activity, "SPYRE", None)


#: Marker identifying an emitted Spyre kernel among a profile's device
#: activities. Every kernel torch-spyre launches is named
#: ``spyre_kernel_v1_...`` (the ``v1`` is the SDSC bundle ABI), so this is the
#: name torch-spyre itself chose rather than a heuristic.
_KERNEL_ACTIVITY_MARKER = "spyre_kernel"


def _kernel_activities(prof: object) -> list[tuple[float, float]]:
    """``(start, end)`` of each **emitted kernel** activity, in microseconds.

    **This filter is the whole point, and T6.1 is why it exists.** With the
    profiler finally compiled in (§8.3), a real profile turned out to contain
    three kinds of device activity, not one. Measured on a 128x256 fp16 add,
    batch 8:

    ====================== === =============
    activity                 n    per launch
    ====================== === =============
    ``Memset (Device)``      9      4.84 us
    ``spyre_kernel_v1_...``  8    **4.23 us**
    ``Memcpy (HtoD)``        8      3.52 us
    ====================== === =============

    Only the middle row is the kernel. The other two are the *caller* staging
    inputs and zeroing the output -- work the harness chose, not work the
    config under test is responsible for. Counting them made
    ``_device_span_ms`` report **2.99x** the kernel's time and
    ``_device_time_ms`` **13.96x** it.

    That is not a constant offset that would cancel in a ranking: staging
    traffic scales with shape and layout, so including it corrupts the
    autotuner's comparison in a shape-dependent way. ``repros/
    device_timer_counts_kernel_only.py`` is the negative control.

    Activities with no ``time_range`` are skipped: AIUPTI leaves
    ``duration_us`` as ``None`` on these events, so the interval is the only
    usable source and an event without one carries no timing at all.
    """
    spans: list[tuple[float, float]] = []
    for event in getattr(prof, "events", list)():
        if "CPU" in str(getattr(event, "device_type", "")):
            continue
        if _KERNEL_ACTIVITY_MARKER not in str(getattr(event, "name", "")):
            continue
        rng = getattr(event, "time_range", None)
        if rng is None:
            continue
        spans.append((float(rng.start), float(rng.end)))
    return spans


def _device_time_ms(prof: object) -> float:
    """Summed **kernel** device work in a profile, 0.0 when there is none.

    Device *work* rather than elapsed device time -- the two differ only when
    activities overlap, which is what ``_device_span_ms`` handles.

    Derived from the kernel activities' own intervals rather than from
    ``key_averages()``, and T6.1 found two reasons that matters:

    1. ``key_averages()`` aggregates **every** operator, including the
       harness's ``Memcpy``/``Memset``, which this must not charge to the
       kernel (see :func:`_kernel_activities`).
    2. The previous implementation summed **both** ``device_time_total`` and
       ``self_device_time_total`` for every row. Those overlap by construction
       for a tree of operators, so it double-counted: measured per launch,
       ``sum(device_time_total) = 65.75 us``,
       ``sum(self_device_time_total) = 27.49 us``, and the reported figure was
       their sum, 93.24 us -- against a true kernel time of 4.21 us.
    """
    return sum(end - start for start, end in _kernel_activities(prof)) / 1e3


def _device_span_ms(prof: object) -> float:
    """Elapsed device time: the union of the **kernel** activity intervals.

    The runtime owns many streams per device -- stream 0 default, 1-32 low
    priority, 33-64 high priority, 65+ host compute
    (``spyre_stream.cpp:95-104``) -- and ``handleKernelActivity`` stamps each
    activity with its ``stream_id`` (``AiuptiActivityHandlers.cpp:343``). So
    activities **can overlap**, and summing their durations would over-count
    the time actually spent. Merging the intervals gives the span the device
    was busy regardless of how the work was spread across streams.

    Only kernel activities are merged (:func:`_kernel_activities`). Merging
    every device activity was worse than merely over-counting: a ``Memcpy``
    immediately before the kernel merges *with* it, so the gap between the two
    was charged to the kernel as well. Measured 2.99x the true kernel time
    before the filter.

    **The span-vs-sum distinction is not yet exercised on hardware.** With the
    filter in place, every shape in T6.1's sweep reports span == sum, because
    these kernels put all their activities on one stream (``device_index`` 0
    throughout). The preference for the span remains right in principle -- a
    sum would double-count genuinely overlapping streams -- but it is verified
    only against the stubbed profile in
    ``repros/bench_source_selection.py`` check 4. Treat it as
    untested-in-practice rather than validated.

    Returns 0.0 when no kernel activities are present.
    """
    spans = _kernel_activities(prof)
    if not spans:
        return 0.0
    spans.sort()
    merged = 0.0
    lo, hi = spans[0]
    for start, end in spans[1:]:
        if start > hi:
            merged += hi - lo
            lo, hi = start, end
        else:
            hi = max(hi, end)
    merged += hi - lo
    return merged / 1e3


@functools.cache
def _device_profiler_available() -> bool:
    """Whether torch-spyre was built with its AIUPTI/Kineto device profiler.

    torch-spyre ships a real device profiler (``torch_spyre/csrc/profiler/``,
    ``handleKernelActivity`` at ``AiuptiActivityHandlers.cpp:328`` emits
    ``CONCURRENT_KERNEL`` with device ``start``/``end`` timestamps), gated at
    build time by ``USE_SPYRE_PROFILER`` -- which defaults to 1
    (``torch-spyre/setup.py:83``) but was **off** for the wheel installed here.

    Detection is a real profile of a trivial launch, because nothing cheaper is
    trustworthy. Measured on this build:

    * ``torch.autograd._supported_activities()`` contains ``SPYRE`` **even
      when no profiler is registered** -- it reflects torch's Kineto support,
      not torch-spyre's. Useless as a probe.
    * every profiled event is ``DeviceType.CPU`` and the summed device time is
      exactly ``0.0``.

    So: non-zero device time is the signal. ~4 ms, once per process, and the
    result cannot change within one.

    Any failure means no usable device timing, so the answer is False and the
    caller falls back -- a benchmark must never raise.

    **The broad ``except`` below is deliberate and stays.** T4's sweep
    narrowed the two broad handlers that *swallowed* a decision
    (``emitter._config_int``'s config lookup and
    ``record_input_layout``'s layout recording, both of which made the backend
    silently assume the one value their knob exists to stop it assuming). This
    one is different in kind: the whole function *is* a feature probe whose
    contract is a bool, and the thing it probes is an optional build-time
    feature of torch-spyre that is currently absent (§8.0). Narrowing it would
    mean enumerating the failure modes of a profiler that is not compiled in.
    """
    activity = _spyre_activity()
    if activity is None:
        return False
    try:
        import torch

        x = torch.zeros(64, 64, dtype=torch.float16).to("spyre")
        probe = torch.compile(lambda t: t + t)
        probe(x)  # warm, so compilation is outside the profile
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, activity]
        ) as prof:
            probe(x)
        return _device_time_ms(prof) > 0.0
    except Exception:
        return False


def _device_timed(fn: object, batch: int = _BENCH_BATCH) -> float | None:
    """Milliseconds of **device** time per launch of ``fn``, or None.

    Prefers the **elapsed** device span (``_device_span_ms``) over summed
    per-op device work. The runtime has up to 64 compute streams plus
    host-compute streams (``spyre_stream.cpp:95-104``) and each activity
    carries a ``stream_id``, so activities can overlap and a plain sum would
    over-count time the device spent. Merging the intervals measures how long
    the device was busy, which is what a config should be ranked on, and the
    two coincide when everything runs on one stream.

    Falls back to the summed form when the profile exposes no usable
    ``time_range`` -- still device time, just work rather than span.

    Returns None if the profile yields no device time at all, so a caller can
    fall back rather than silently report 0.0 and make every config look equal.

    **VERIFIED ON HARDWARE 2026-10-28** (T6.1), after the rebuild that put the
    profiler in the wheel (§8.3). ~~UNVERIFIED: the wheel installed here was
    built USE_SPYRE_PROFILER=0, so this path has never run against a real
    profiler.~~ What the scale test established, measured on a 256x range of
    work:

    * device time responds **15.0x** to a 256x change in elements, against the
      drained wall clock's **2.6x** -- so it is measuring work, which was the
      open question;
    * the gap to the drained clock is the launch overhead, directly: **89%** of
      the wall clock at 64x128 falling to **34%** at 1024x2048;
    * ``key_averages().device_time_total`` *is* populated, so that was a real
      data source -- but it aggregates the harness's copies too, which is why
      ``_device_time_ms`` no longer uses it (see there).

    The span-vs-sum choice is still **untested in practice**: with the kernel
    filter in place every measured shape reports span == sum, because all
    activities land on ``device_index`` 0. It is checked only against the
    stubbed profile in ``repros/bench_source_selection.py`` check 4.
    """
    activity = _spyre_activity()
    if activity is None:
        return None
    import torch

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, activity]
    ) as prof:
        for _ in range(batch):
            fn()
        torch.accelerator.synchronize()
    total = _device_span_ms(prof) or _device_time_ms(prof)
    return total / batch if total > 0.0 else None


def _bench_once(fn: object) -> float:
    """One timing sample in milliseconds, device time when it is available.

    The selection is logged once per process so a reviewer, or a future
    session reading an autotune log, can tell which path produced a number.
    """
    if _device_profiler_available():
        sample = _device_timed(fn)
        if sample is not None:
            _log_bench_source("device (AIUPTI/Kineto CONCURRENT_KERNEL)")
            return sample
    _log_bench_source("drained wall clock (no device profiler in this build)")
    return _drained(fn)


@functools.cache
def _log_bench_source(what: str) -> None:
    log.info("spyre benchmark source: %s", what)


def _sencores() -> int:
    """Cores available to one kernel (``SENCORES``, default 32)."""
    from torch_spyre._inductor import config as spyre_config

    return int(spyre_config.sencores)


#: Device for a synthesized scalar constant in the generated host wrapper.
_DEVICE_EXPR = '"spyre"'


class SpyreBackend(Backend):
    """Emits torch-spyre ``OpSpec``/``LoopSpec`` objects per Device IR node."""

    _SUPPORTED_CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "block_sizes",
            "range_unroll_factors",
            "spyre_lx_buffers",
            "spyre_relayout_split",
        }
    )

    def __init__(self) -> None:
        super().__init__()
        self.emitter = SpyreEmitter()
        #: id(fake tensor) -> stick axis of the real input it stands for.
        self._input_stick_dims: dict[int, int] = {}
        #: id(fake tensor) -> host rank, so output_stick_dim can match by rank.
        self._input_ranks: dict[int, int] = {}

    @property
    def name(self) -> str:
        return "spyre"

    @property
    def experimental(self) -> bool:
        return True

    def validate_environment(self) -> None:
        try:
            import torch_spyre  # noqa: F401
            from torch_spyre._inductor.op_spec import OpSpec  # noqa: F401
        except ImportError as e:
            raise exc.BackendUnsupported(
                self.name,
                "torch-spyre is not installed; install it with"
                " `pip install torch-spyre`",
            ) from e

    # -- what Helion must not assume on this backend ----------------------

    @property
    def pad_factory_tensors_to_power_of_2(self) -> bool:
        # Plan 4.3: Spyre wants stick multiples, not powers of two.
        return False

    @property
    def requires_shape_specialized_module(self) -> bool:
        # Plan 5.2: the generated module caches a compiled Spyre kernel keyed by
        # spec hash, so distinct static_shapes specializations that happen to
        # emit identical source must not share it.
        return True

    def supports_config_key(self, key: str) -> bool:
        return key in self._SUPPORTED_CONFIG_KEYS

    def backend_tunables(self) -> dict[str, object]:
        """Expose LX residency to the autotuner instead of hard-coding it.

        How many intermediates live in LX before the rest spill to ``hbm_pool``
        is a real performance decision, and the right value is kernel-dependent
        -- so it must be searched, not baked in. Nothing in the stack diagnoses
        asking for too much: the specs validate, the bundle builds, the backend
        compiler succeeds, and the kernel then fails *at launch* with
        ``RAS::RUNTIMESCHEDULER::ComputeHardwareError``, which also poisons the
        device stream for the rest of the process.

        The ceiling is a **count, not a byte budget**: measured on this device,
        ~~3 concurrent LX intermediates fail at both 8 KB each and 64 KB each,
        an 8x difference in footprint with the same wall, so the search space is
        capped at ``_MAX_LX_BUFFERS``.~~ That wall was bug #15 -- an addressing
        bug here, not hardware. The search now runs to ``_LX_SEARCH_MAX`` and
        the emitter spills to ``hbm_pool`` when the LX *byte* budget is
        exhausted, so no config in the range can fail as a hardware fault.
        """
        from ...autotuner.config_spec import IntegerFragment

        return {
            # low=0 (everything in hbm_pool) .. high=_LX_SEARCH_MAX, default
            # _MAX_LX_BUFFERS (2, the previously-shipped behaviour).
            #
            # ~~high was _MAX_LX_BUFFERS, because 3 concurrent LX buffers
            # failed at launch at any size.~~ That ceiling was bug #15 -- an
            # addressing bug in this backend, not a hardware limit (see
            # ``_LX_CAPACITY_BYTES``). torch-spyre's own softmax uses **four**
            # concurrent LX buffers, so a cap of 2 made the reference kernel
            # inexpressible. The real limit is now the byte budget, enforced
            # per allocation by ``SpyreEmitter._lx_fits``, which spills to
            # ``hbm_pool`` instead of failing.
            "spyre_lx_buffers": IntegerFragment(0, _LX_SEARCH_MAX, _MAX_LX_BUFFERS),
            # Cores a ReStickifyOpHBM splits its leading non-stick dim over.
            #
            # This was the frozen literal ``(2, 4, 1)`` inside
            # ``emitter._relayout_split`` -- "the reference uses a small split",
            # first of the three that divides. That is a work-division decision
            # neither user-written nor autotuned, i.e. exactly the governing
            # invariant (§1) bent in the way the assessment calls "the clearest
            # next hardening target" (summary deviation 3).
            #
            # Exposing it as a knob is also the lesson of bug #5, which was
            # found *only* because a frozen constant was turned into one: a
            # decision hard-coded hides bugs in the path it never takes. The
            # default is 2, so the previous behaviour is still what an
            # unspecified config gets; the difference is that the other values
            # are now reachable and benchmarked rather than unreachable.
            #
            # A relayout needs a real split: with every dim unsplit the backend
            # scheduler rejects the program ("Expect valid lower and upper
            # bound parameters"), so low=1 is the floor and
            # ``_relayout_split`` falls back to a divisor when the requested
            # value does not divide the extent.
            "spyre_relayout_split": IntegerFragment(1, 8, 2),
        }

    def supports_precompile(self) -> bool:
        return False

    def get_interleaved_bench(self) -> object:
        """Interleaved timing with the device queue drained; the default needs
        Triton.

        ``autotuner/benchmarking.py:383`` imports ``triton.runtime`` for its
        cache-clearing and device-interface handles, which do not exist here.
        The hook's own docstring (`_compiler/backend.py:603-604`) says backends
        without Triton event timing should override.

        Interleaving exists to cancel slow drift between candidates, so the
        sweeps are round-robin over the candidate functions rather than
        candidate-at-a-time. Each candidate's turn goes through
        ``_bench_once``: real device time when torch-spyre carries its
        profiler, otherwise a *drained* batch (see ``_BENCH_BATCH``). Without
        the synchronize this timed the submission queue and ranked candidates
        backwards.
        """

        def spyre_interleaved_bench(
            fns: object,
            *,
            repeat: int = 3,
            desc: object = None,
            max_total_ms: object = None,
            **kwargs: object,
        ) -> list[float]:
            fns = list(fns)
            for fn in fns:
                fn()

            totals = [0.0] * len(fns)
            sweeps = max(1, min(int(repeat), _BENCH_REPS))
            for _ in range(sweeps):
                for i, fn in enumerate(fns):
                    totals[i] += _bench_once(fn)
            return [t / sweeps for t in totals]

        return spyre_interleaved_bench

    def get_do_bench(self) -> object:
        """Device time when the profiler is built in, drained wall clock if not.

        ``autotuner/benchmarking.py`` times with CUDA events and streams
        (`:105-114`), which do not exist for the ``spyre`` device.

        ``_bench_once`` prefers torch-spyre's AIUPTI/Kineto device activity,
        which is real device time rather than a host clock. That profiler is
        gated by ``USE_SPYRE_PROFILER`` (default 1) and is **absent from the
        wheel installed here**, so the fallback is what runs today: a Spyre
        launch is asynchronous, so a clock around one call measures submission
        and is flat in the problem size, and each sample is a drained batch
        instead. See ``_BENCH_BATCH`` for the measurements that forced it, and
        ``_device_profiler_available`` for how the two are told apart.
        """

        def spyre_do_bench(
            fn: object,
            warmup: int = 25,
            rep: int = 100,
            grad_to_none: object = None,
            quantiles: object = None,
            return_mode: str = "mean",
            process_group_name: object = None,
            **kwargs: object,
        ) -> float | tuple[float, ...]:
            import statistics

            for _ in range(_BENCH_WARMUP):
                _bench_once(fn)

            samples: list[float] = [_bench_once(fn) for _ in range(_BENCH_REPS)]

            if return_mode == "all":
                return tuple(samples)
            if quantiles:
                ordered = sorted(samples)
                return tuple(
                    ordered[min(int(q * (len(ordered) - 1)), len(ordered) - 1)]
                    for q in quantiles
                )
            reducer = {
                "min": min,
                "max": max,
                "median": statistics.median,
                "mean": statistics.fmean,
            }.get(return_mode, statistics.fmean)
            return float(reducer(samples))

        return spyre_do_bench

    def adjust_block_size_constraints(
        self,
        block_specs: list[object],
        ndim: int,
        block_sizes: list[object] | None = None,
        kernel_tensor_sizes: dict[tuple[object, ...], int] | None = None,
        min_element_bits: int = 32,
    ) -> None:
        """Keep the autotuner off block sizes that split a stick.

        Plan 4.3: the trailing axis is the within-stick axis and is never
        divided -- "a stick is the unit of transfer, so splitting it across
        cores would split a stick" (codegen/opspec_utils.py:515-516). A sub-stick
        trailing block is not caught by ``validate_op_specs``; it surfaces as an
        opaque ``SubprocException`` from the backend compiler, so bound the
        search instead of discovering it late.

        The floor is the widest stick any supported dtype uses (64 elements at
        fp16/bf16; fp32 uses 32 and is therefore also satisfied).

        The same floors also keep the **core budget** reachable. The product of
        the per-dim splits is the core count, bounded by ``sencores`` (32), and
        ``autotune_config_is_viable`` already rejects candidates that overshoot
        -- but the autotuner's *reference* config is taken straight from these
        fragment defaults (``benchmark_provider.py:876``) and never goes through
        that check. On 512x512 the default landed on ``block_sizes=[32, 64]``,
        which asks for 16*8 = 128 cores and dies in ``dbo-opt`` with SIGABRT.
        The generic recovery cannot help: it *shrinks* block sizes
        (``_compute_reference_baseline``), and on a fixed extent a smaller block
        means **more** cores, so every retry moves further over budget (128 ->
        256 -> 512). Raising the floors here fixes it at the source, since
        ``BlockSizeSpec._fragment`` clamps its default to ``min_size``.
        """
        from ...autotuner.config_spec import BlockSizeSpec

        specs = [s for s in block_specs if isinstance(s, BlockSizeSpec)]
        if len(specs) < 2:
            return
        for trailing in specs[1:]:
            # Never raise the floor above the dim itself: a dim narrower than
            # one stick (e.g. N=32 fp16) occupies a single partial stick, which
            # is not a split and is therefore legal. Forcing 64 there makes the
            # block indivisible by the extent and nothing can satisfy it.
            extent = getattr(trailing, "size_hint", None)
            if isinstance(extent, int) and extent < _MAX_ELEMS_PER_STICK:
                continue
            trailing.update_min(_MAX_ELEMS_PER_STICK)

        self._raise_floors_for_core_budget(specs)

    @staticmethod
    def _raise_floors_for_core_budget(specs: list[BlockSizeSpec]) -> None:
        """Raise block-size floors until the minimum core count fits sencores.

        At each floor the split of a dim is ``extent // min_size``, so the
        product over dims is the fewest cores any in-bounds config can ask for.
        While that exceeds the budget, double the floor of the dim currently
        contributing the most splits -- halving its core count -- which
        converges and keeps the floors as low as the budget allows.
        """
        budget = _sencores()

        def splits(spec: object) -> int:
            extent = int(getattr(spec, "size_hint", 0) or 0)
            floor = max(1, int(getattr(spec, "min_size", 1) or 1))
            return max(1, extent // floor) if extent > 0 else 1

        for _ in range(32):
            counts = [splits(s) for s in specs]
            cores = 1
            for c in counts:
                cores *= c
            if cores <= budget:
                return
            # Double the floor of the worst offender; a power-of-two floor is
            # required (update_min asserts it).
            worst = specs[counts.index(max(counts))]
            floor = max(1, int(getattr(worst, "min_size", 1) or 1))
            extent = int(getattr(worst, "size_hint", 0) or 0)
            if floor >= extent:
                # Already one core on this dim; nothing further to give.
                return
            worst.update_min(floor * 2)

    def autotune_config_is_viable(self, config_spec: object, config: Config) -> bool:
        """Reject configs Spyre cannot express, before anything is emitted.

        Three rules, each one a hard constraint rather than a preference:

        * **Exact division.** Spyre work division and coarse-tile counts are
          integers; a block size that does not divide its extent has no valid
          ``iteration_space`` split (`wsr/coarse_tile.py:1384-1395` raises on
          this for coarse tiles). The emitter already fails loudly, but a
          rejected *config* must never reach codegen.
        * **Stick alignment.** As in ``adjust_block_size_constraints``: a
          trailing block below one stick splits a stick.
        * **One reduction split.** Plan 4.1: at most one reduction dim may be
          split across cores (`work_division.py:328-331`), since a K-split
          builds a PSUM chain the emitter does not model.
        * **Core budget.** The product of the per-dim splits is the number of
          cores the kernel asks for, bounded by ``config.sencores``
          (`_inductor/config.py:166`, default 32; also the ``max_cores`` the
          work divider itself uses at `work_division.py:2761`). Overshooting is
          not caught by ``validate_op_specs`` -- it surfaces as an opaque
          ``SubprocException`` from the backend compiler, which an autotune
          sweep hits constantly as it shrinks block sizes.
        """
        lx_buffers = config.config.get("spyre_lx_buffers")
        if lx_buffers is not None and not (0 <= int(lx_buffers) <= _LX_SEARCH_MAX):
            # ~~Above the measured ceiling the kernel fails at launch, not
            # here.~~ There is no such ceiling (bug #15); this now only screens
            # values outside the searched range. Exceeding the LX *byte* budget
            # is not a viability question at all, because the emitter spills to
            # ``hbm_pool`` rather than failing -- see ``_lx_fits``.
            return False

        block_sizes = config.block_sizes
        if not block_sizes:
            return True

        # config_spec is the authority here, not CompileEnvironment.current():
        # this hook is called from the autotuner's search loop, where there is
        # no active compile environment, so reading the ambient one silently
        # accepted every candidate.
        spec = getattr(config_spec, "block_sizes", None)
        if spec is None:
            return True
        infos = [spec.block_id_lookup(bid) for bid in spec.valid_block_ids()]
        if len(infos) != len(block_sizes):
            # A config that does not describe this kernel's block ids is not
            # viable. ruff's B905 flagged the bare ``zip`` here: it would
            # silently *truncate* to the shorter list, so a config missing a
            # trailing block size would be screened against only its leading
            # dims and pass. Returning False is the honest answer -- and this
            # hook's contract is a bool, so the ``strict=True`` that B905 wants
            # cannot raise here; the length check is what makes it safe.
            return False

        # **Which axis carries the sticks.** Read from the one authority rather
        # than assumed to be the trailing one. ``_input_stick_dims`` is
        # populated by ``record_input_layout`` at bind time and persists across
        # autotuner trials, so it is available here even though there is no
        # active compile environment -- which is what makes the per-op fact
        # usable in a per-config check.
        #
        # This resolves the tension the plan flagged: the stick axis is an
        # *op*-level fact while viability is *per config*. A kernel-wide axis is
        # the conservative reading of it, and it is sound here because every
        # rank-matching tensor had to agree for ``output_stick_dim`` to return
        # anything at all -- a mixture gives ``None`` and
        # ``kernel_stick_dim`` falls back to the trailing dim, which is exactly
        # the old behaviour. So this is never *weaker* than before, and it is
        # stronger precisely in the dim-0-sticked case that was wrong.
        stick_position = self.kernel_stick_dim(len(block_sizes))

        reduction_splits = 0
        cores = 1
        for position, (info, block) in enumerate(zip(infos, block_sizes, strict=True)):
            extent = int(getattr(info, "size_hint", 0) or 0)
            value = int(block)
            if extent <= 0:
                continue
            if value <= 0 or extent % value != 0:
                # Work division must be exact.
                return False
            if position == stick_position:
                # **The stick axis may be divided only down to whole sticks.**
                # This screen is the clamp that used to live in
                # ``compute_ops._iteration_space``, moved here because a clamp
                # in codegen makes the autotuner benchmark a work division it
                # did not choose -- the governing invariant (§1: "never solved
                # behind their back") bent in the one place the autotuner is
                # supposed to be in charge.
                #
                # Measured before the move, on a dim-0-sticked 128x256 fp16
                # kernel: configs asking for c0 splits of 2, 4 and 8 all
                # emitted a split of 2, and ``autotune_config_is_viable``
                # returned True for all three. So the search space contained
                # three configs that were the same kernel, and the autotuner
                # spent launches distinguishing them.
                sticks = max(1, -(-extent // _MAX_ELEMS_PER_STICK))
                if extent // value > sticks:
                    return False
            elif value < _MAX_ELEMS_PER_STICK and extent > _MAX_ELEMS_PER_STICK:
                # A block below one stick on a *non*-stick axis splits a stick
                # too -- unless the whole dim is narrower than a stick, where
                # it cannot. (This was guarded as ``position > 0``, i.e. with
                # the trailing axis assumed to be the stick axis; for a
                # dim-0-sticked kernel that guarded the wrong axis entirely.)
                return False
            if getattr(info, "reduction", False) and extent // value > 1:
                reduction_splits += 1
            cores *= extent // value

        # At most one reduction dim may be split across cores.
        if reduction_splits > 1:
            return False
        return cores <= _sencores()

    # -- the emission seam ------------------------------------------------

    def pre_codegen(
        self,
        graphs: list[GraphInfo],
        config: Config,
        tile_strategy: TileStrategyDispatch,
    ) -> None:
        self.emitter.reset()

    def record_input_layout(self, real: torch.Tensor, fake: torch.Tensor) -> None:
        """Remember each real input's stick axis before it is fakified.

        Plan 5.3 says input layouts are *read*, not chosen -- but by codegen
        time every tensor is a ``FakeTensor`` whose ``device_tensor_layout()``
        returns ``None``, so the real layout is unrecoverable there. Without
        this the backend silently assumed the default trailing-dim stick axis
        for every input, which is wrong numerics (not a crash) for a
        transposed operand and means no ReStickifyOpHBM is ever emitted.
        """
        from .layout import stick_dim_of

        # Narrowed to ``BackendUnsupported``, which is the only thing
        # ``stick_dim_of`` raises by design: an unsupported dtype (via
        # ``device_dtype``) or a ``device_size`` that cannot be reconciled with
        # the host shape. Both of those genuinely do fail again later, at the
        # point where the operand is used and the message can name it.
        #
        # A bare ``except Exception`` here was the wrong shape for this hook in
        # particular: silently not recording a layout makes the backend assume
        # the default trailing-dim stick axis, which is **bug #6 and #11's
        # failure mode** -- wrong numerics with nothing raised, not a crash. So
        # anything that is not a described limitation now propagates.
        try:
            self._input_stick_dims[id(fake)] = stick_dim_of(real)
            self._input_ranks[id(fake)] = real.dim()
        except exc.BackendUnsupported:
            pass

    def input_stick_dim(self, fake: torch.Tensor) -> int | None:
        return self._input_stick_dims.get(id(fake))

    def output_stick_dim(self, rank: int) -> int | None:
        """Stick axis every rank-``rank`` kernel tensor agrees on, if any.

        A kernel's output tensor is recorded by ``record_input_layout`` like any
        other argument (``torch.empty_like(x)`` in the kernel body is still a
        host tensor passed to the launcher), but which recorded entry *is* the
        output is not known here. When every rank-matching tensor shares one
        stick axis, that axis is unambiguous and is the right target; a mixture
        means a genuine conflict, which the relayout path resolves per operand.

        **The set-resolve is deliberate and `None` is a real answer**, not a
        fallback. §9 item 3 asked for this to get `stick_dim_of`'s treatment
        (that was one forward scan, and it gave bug #11). The difference is
        that `stick_dim_of` had to *choose* between candidates that both fitted
        one tensor's layout, where guessing was the bug; here a mixture is
        several tensors genuinely disagreeing, and the per-operand relayout
        path is the right resolution rather than a tie-break. What this does
        share with bug #11 is that a *square* shape is the configuration that
        exposes it -- `stick_dim_of` returns the tie-broken trailing dim for
        every square tensor, so a square kernel records one axis for all of
        them and this returns that axis whether or not it is the real one. The
        gate for that is `repros/output_stick_dim_square.py`.
        """
        axes = {
            dim
            for key, dim in self._input_stick_dims.items()
            if self._input_ranks.get(key) == rank
        }
        return axes.pop() if len(axes) == 1 else None

    def kernel_stick_dim(self, rank: int) -> int:
        """**The** stick axis for a rank-``rank`` operand of this kernel.

        One authority, used by every caller that needs "which axis carries the
        sticks": the pointwise agreement axis (`compute_ops._output_stick_dim`),
        the stick-split bound in `_iteration_space`, and
        `autotune_config_is_viable`'s stick-alignment screen. Before this, the
        first two used the real recorded axis while the third assumed the
        trailing one (`position > 0`), so for a dim-0-sticked kernel the
        viability check guarded **the wrong axis entirely** -- review finding
        (d) and §9 item 3 seen from two sides.

        Falls back to the trailing dim when nothing was recorded, which is
        torch-spyre's own default tiled layout and the same choice
        `stick_dim_of` makes for a tensor with no device layout to read.
        """
        recorded = self.output_stick_dim(rank)
        return rank - 1 if recorded is None else recorded

    def pre_inductor_lowering(self, node: torch.fx.Node) -> Lowering | None:
        """Install a Spyre lowering for every compute node.

        Preempts Inductor's ``PointwiseLowering``/``ReductionLowering``, which
        would funnel the node's result through ``_unpack_opsvalue``
        (inductor_lowering.py:1415-1419) and assert on a non-string.
        """
        if node.op != "call_function":
            return None
        from .compute_ops import SpyreOpLowering
        from .optable import spyre_op_name

        if spyre_op_name(node.target) is None:
            return None
        return SpyreOpLowering(self, node.target)

    # -- module shape (plan 5.1) -----------------------------------------

    # -- the vestigial device body ---------------------------------------
    #
    # Plan 5.1: for Spyre there is no device Python -- the payload is the spec
    # literal. Helion still emits a device FunctionDef and walks the tile
    # strategy to build it, so these hooks exist only to keep that walk
    # producing *syntactically legal* Python that is never executed. Nothing
    # here feeds an OpSpec: the iteration space is derived from the node's own
    # shape in compute_ops, not from these index variables.

    def index_type_str(self, index_dtype: torch.dtype) -> str:
        return "int"

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        return f"_spyre_pid({dim})"

    def grid_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        return f"({offset_var})"

    def loop_index_expr(
        self, offset_var: str, block_size_var: str, dtype: str, *, axis: int
    ) -> str:
        return f"({offset_var})"

    def arange_expr(
        self,
        offsets_var: str,
        lid: str,
        block_size_var: str,
        dtype: str,
        *,
        axis: int = 0,
    ) -> str:
        return f"({offsets_var})"

    def cast_expr(self, expr_str: str, dtype_str: str) -> str:
        return expr_str

    def expands_broadcast_dims(self) -> bool:
        # Shape lives in device_size/device_coordinates, not in an expression,
        # so a Triton-style [None, :] broadcast-expand is meaningless here.
        # Same move FlyDSL makes (flydsl/backend.py:294).
        return False

    @property
    def function_decorator(self) -> str:
        return ""

    def function_decorator_for_args(self, args: Sequence[object]) -> str:
        """Emit the spec literal, then decorate nothing.

        Plan 5.1: this runs from ``codegen_function_def`` -- after the device
        body has been walked (so the spec tree is complete) and while
        ``module_statements`` is still open (so the literal can be inserted
        without a core edit). The literal itself is printed by torch-spyre's own
        ``_codegen_op_spec_list`` (spyre_kernel.py:1646); plan 5.1 is explicit
        that we must not write a second printer.
        """
        from ..ast_extension import statement_from_string
        from ..device_function import DeviceFunction
        from .spec_source import spec_source

        device_function = DeviceFunction.current()
        emitter = self.emitter
        specs = emitter.specs()
        if not specs:
            raise exc.BackendUnsupported(self.name, "kernel emitted no OpSpecs")

        statements = device_function.codegen.module_statements
        statements.append(statement_from_string(f"_SPYRE_SPECS = {spec_source(specs)}"))
        statements.append(
            statement_from_string(f"_SPYRE_POOL_SIZE = {emitter.pool_size}")
        )
        return ""

    @property
    def constexpr_type(self) -> str:
        return "int"

    @property
    def default_launcher_name(self) -> str:
        return "_default_spyre_launcher"

    @property
    def library_imports(self) -> dict[str, str]:
        return {
            "torch": "import torch",
            "sympify": "from sympy import sympify",
            "DataFormats": "from torch_spyre._C import DataFormats",
            "ElementArrangement": "from torch_spyre._C import ElementArrangement",
            "OpSpec": "from torch_spyre._inductor.op_spec import OpSpec",
            "LoopSpec": "from torch_spyre._inductor.op_spec import LoopSpec",
            "TensorArg": "from torch_spyre._inductor.op_spec import TensorArg",
            "IndirectAccess": (
                "from torch_spyre._inductor.op_spec import IndirectAccess"
            ),
            "_default_spyre_launcher": (
                "from helion.runtime import default_spyre_launcher"
                " as _default_spyre_launcher"
            ),
        }

    @staticmethod
    def reserved_launch_param_names() -> frozenset[str]:
        return frozenset({"_spyre_specs", "_spyre_pool_size"})

    # -- hooks that are structurally wrong here --------------------------
    #
    # Each of these is a *string*-valued expression hook. On this backend a
    # value is a buffer, not an expression, so a string here would be silently
    # meaningless. Plan 6.4: the carry is structural, emitted by the
    # _for_loop handler, so the reduction string hooks stay unsupported.

    def dtype_str(self, dtype: torch.dtype) -> str:
        from .layout import device_dtype

        return str(device_dtype(dtype))

    def acc_type(self, dtype: torch.dtype) -> str:
        raise exc.BackendUnsupported(
            self.name, "acc_type: Spyre accumulators are buffers, not type strings"
        )

    def inductor_op_overrides(self) -> InductorOpOverrides:
        # Plan 1.1: deliberately unimplemented. Reaching this means a compute
        # node escaped pre_inductor_lowering.
        raise exc.BackendUnsupported(
            self.name,
            "inductor_op_overrides: every compute node must be claimed by"
            " pre_inductor_lowering",
        )

    def reduction_combine_expr(self, *args: object, **kwargs: object) -> str:
        raise exc.BackendUnsupported(
            self.name, "reduction_combine_expr: the carry is an explicit buffer"
        )

    def reduction_expr(self, *args: object, **kwargs: object) -> str:
        raise exc.BackendUnsupported(
            self.name, "reduction_expr: the carry is an explicit buffer"
        )

    def build_launcher_args(
        self,
        args: list[str],
        *,
        tensor_host_args: list[str],
        has_rng_ops: bool,
        config: Config,
        has_barrier: bool,
        sorted_args: Sequence[object] | None = None,
    ) -> list[str]:
        """Pass the host tensors positionally, in OpSpec ``arg_index`` order.

        The device body is vestigial here, so it references none of the kernel
        tensors and Helion registers no device args for them: ``args``,
        ``tensor_host_args`` and ``sorted_args`` all arrive empty, and the
        emitted call would be ``_launcher(kernel, grid)`` with no tensors --
        which fails at runtime with "symbolic_args tensor_id out of range".

        The emitter's own buffer table is the source of truth instead. Each
        graph input/output was registered from Helion's ``tensor_arg`` table, so
        its ``name`` *is* the host wrapper's variable name and its ``arg_index``
        is the position SDSC expects. A scalar constant the emitter synthesized
        (an accumulator zero-fill has no host tensor) is materialized here.
        """
        if has_rng_ops:
            raise exc.BackendUnsupported(self.name, "RNG ops")

        consts = {index for index, _, _ in self.emitter.const_args}
        launch: list[tuple[int, str]] = []
        for operand in self.emitter.args:
            if operand.arg_index < 0:
                continue
            if operand.arg_index in consts:
                launch.append(
                    (
                        operand.arg_index,
                        (
                            f"torch.zeros((), dtype={operand.dtype!r},"
                            f" device={_DEVICE_EXPR})"
                        ),
                    )
                )
            elif operand.name:
                launch.append((operand.arg_index, operand.name))
            else:
                raise exc.BackendUnsupported(
                    self.name, f"kernel arg {operand.arg_index} has no host name"
                )
        return [name for _, name in sorted(launch)]

    def scalar_arg_preamble(self, arg: object) -> list[ast.AST]:
        return []
