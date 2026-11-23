"""Tune each Helion kernel of a compiled graph behind the kernel that runs
before it.

A graph's later kernels meet their operands as the kernel before them left
them: the sums over y of a GELU backward read the gradient its pointwise kernel
has just written, much of it still in L2. Searched alone, after an L2 flush, a
kernel picks a config for a state it never runs in. So the generated code wraps
every kernel (``in_sequence``): the first call of each one, during the graph's
first run, is timed behind the call before it (``benchmark_prelude``'s
``timed_behind``, the regime of better-benchmark's ports), and later calls go
to the kernel itself. In the output code::

    helion_fused_sum_view_1 = PyCodeCache.load(
        helion_fused_sum_view_1_source, "7c1f0d2a9b3e5f64"
    ).kernel
    helion_fused_sum_view_1 = helion_in_sequence(
        helion_fused_sum_view_1,
        globals(),
        "helion_fused_sum_view_1",
        after="helion_fused_add_0",
    )

``after`` is the kernel whose first call is the call before, named where the
code is written (``codegen.sequence``), and the module is loaded under a key
of its predecessors (``module_key``): None, and the kernel is tuned alone,
when another launch comes between the two (an extern kernel) or the call
before is a later call of its kernel, which runs bare. The first call is
timed behind the call the module recorded last only when that call is
``after``'s: a kernel in a branch the run skips leaves another kernel's
call there.

The prelude is the one call before, not every earlier call: a kernel's search
replays its prelude before each timed call, so a longer one would make the
searches of a long graph's last kernels slower without changing the state
they find much. A search behind a prelude caches its winner under the
prelude's fingerprint (``local_cache._search_policy_hash``).
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING
from typing import Any

from ..autotuner.benchmark_prelude import KernelSequence
from ..autotuner.benchmark_prelude import PreludeRefused
from ..autotuner.benchmark_prelude import PreludeSearchFailed
from ..autotuner.benchmark_prelude import PreludeUnsupported
from ..autotuner.benchmark_prelude import timed_behind
from ..autotuner.benchmark_provider import benchmark_prelude_unsupported
from ..runtime.ref_mode import is_ref_mode_enabled
from ..runtime.settings import default_autotuner_fn

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..runtime.kernel import BoundKernel
    from ..runtime.kernel import Kernel

log = logging.getLogger(__name__)


def times_short_kernels(bound: BoundKernel[Any]) -> bool:
    """Whether ``bound``'s backend has its searches read a short kernel's
    candidates from the CUDA activity trace
    (``Backend.autotune_short_kernel_below_us``), as CuTe's does. A search
    that reads them between CUDA events, as Triton's does, reads a short
    kernel behind a prelude in the event clock's steps of about 2 us, the
    host launch included: better-benchmark's harness read a 6 us max pool at
    18 to 19 us that way, uncorrelated with how it runs, and tunes such a
    backend's kernels alone (``default_regime``). This asks the backend, not
    the search: on CuTe, a search that reads no initial population (the
    LLM-guided one) keeps the event timer and is still timed behind a
    prelude."""
    spec = bound.config_spec
    return spec.backend.autotune_short_kernel_below_us(spec) is not None


class GraphKernels:
    """The kernel calls of one generated module's first run: the first call
    that ran last, which the next kernel's first call is timed behind.  A
    first run that fails between the two (an allocation out of memory)
    leaves the call kept, with its tensors, until the next first call, in a
    later run, which is then timed behind it."""

    def __init__(self) -> None:
        self.registered = 0
        self.launched = 0
        # the kernels some kernel's first call is timed behind (``after``)
        self.wanted: set[str] = set()
        # (kernel name, compiled function, arguments, output) of the last call
        self.last: (
            tuple[str, Callable[..., object], tuple[object, ...], object] | None
        ) = None

    def launch(
        self,
        kernel: Kernel[Any],
        args: tuple[object, ...],
        index: int,
        name: str,
        after: str | None,
    ) -> object:
        """The first call of ``kernel``, defined ``index``-th in the module as
        ``name``, timed behind the first call of kernel ``after`` where that
        is the call before and the search can take it."""
        # Whatever this call does, the call before it is no one's prelude
        # after it: taken now, it is let go also when binding the kernel or
        # building the prelude fails, which leaves the bare kernel in the
        # module's globals, and nothing would replace it.
        last, self.last = self.last, None
        call_args = kernel.normalize_args(*args)
        prelude = None
        bound = kernel.bind(call_args)
        if is_ref_mode_enabled(bound.settings):
            # Reference mode (HELION_INTERPRET=1) runs the kernel eagerly:
            # there is no search to time behind a prelude, and no config to
            # record for one (compile_config() would raise "no config
            # provided and no implicit config available").
            self.launched += 1
            return kernel(*args)
        # A search that cannot time behind a prelude (HELION_AUTOTUNE_L2_FLUSH=0,
        # graph replays, timing in this process, a distributed kernel:
        # benchmark_prelude_unsupported; a custom autotuner, whose provider
        # may not run one) would refuse it, and one that reads its candidates
        # between CUDA events could not resolve a short kernel behind it
        # (times_short_kernels): the kernel is tuned alone, as without the
        # sequence.
        if (
            after is not None
            and last is not None
            and last[0] == after
            and bound.settings.autotuner_fn is default_autotuner_fn
            and benchmark_prelude_unsupported(bound.settings, bound.config_spec, bound)
            is None
            and times_short_kernels(bound)
        ):
            sequence = KernelSequence()
            sequence.record(*last[1:])
            # helion.inductor's kernels return None, so no call before has an
            # output a later argument could stand for; should one, a call it
            # cannot replay leaves this kernel tuned alone.
            with contextlib.suppress(PreludeUnsupported):
                prelude = sequence.prelude(call_args)
        # The prelude holds what it needs of the call before.
        del last
        try:
            with timed_behind(prelude):
                output = kernel(*args)
        except (PreludeRefused, PreludeSearchFailed) as error:
            # The search could not time behind the call before: the provider
            # refused the prelude (it cannot reach the benchmark worker), or
            # the prelude failed there and left the search without a config
            # (out of memory with its arguments). The kernel is tuned alone,
            # as without the sequence. Any other failure is the kernel's own.
            log.warning(
                "Tuning %s alone: its search behind the kernel before it "
                "failed (%s: %s)",
                kernel.name,
                type(error).__name__,
                error,
            )
            output = kernel(*args)
        self.launched += 1
        # This call is kept for a prelude only while a kernel may still be
        # timed behind it: not when no kernel names it, every kernel has run,
        # or the last one the module defines has. (A kernel that skipped the
        # first run, one in a branch not taken, is tuned alone when it first
        # runs, not behind a call of an earlier run.)
        if (
            name in self.wanted
            and self.launched < self.registered
            and index < self.registered - 1
        ):
            compiled = bound.compile_config(allow_print=False)
            self.last = (name, compiled, tuple(call_args), output)
        return output


class SequencedKernel:
    """A kernel until its first call, which ``GraphKernels`` times behind the
    call before it; the call then puts the kernel itself in the module's
    globals, so the generated code calls it directly from then on."""

    def __init__(
        self,
        kernel: Kernel[Any],
        graph: GraphKernels,
        module_globals: dict[str, object],
        name: str,
        after: str | None,
    ) -> None:
        self.kernel = kernel
        self.graph = graph
        self.module_globals = module_globals
        self.name = name
        self.after = after
        # the order in which the module defines its kernels
        self.index = graph.registered
        graph.registered += 1
        if after is not None:
            graph.wanted.add(after)

    def __call__(self, *args: object) -> object:
        self.module_globals[self.name] = self.kernel
        return self.graph.launch(self.kernel, args, self.index, self.name, self.after)


def in_sequence(
    kernel: Kernel[Any],
    module_globals: dict[str, object],
    name: str,
    after: str | None = None,
) -> SequencedKernel:
    """``kernel``, defined as ``name`` in the generated module whose globals
    are ``module_globals``, wrapped for its first call, which is timed behind
    the first call of kernel ``after`` or, None, alone (module docstring)."""
    graph = module_globals.setdefault("_helion_graph_kernels", GraphKernels())
    assert isinstance(graph, GraphKernels)
    return SequencedKernel(kernel, graph, module_globals, name, after)
