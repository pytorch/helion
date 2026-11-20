"""Time a search's candidates behind the kernels that run before them.

Internal to the autotuner and its harnesses: not a setting, a decorator
argument or an environment variable, and free to change.

The per-call flushed regimes (``call`` and ``kernel``, see
``benchmarking._l2_flush_mode``) time a candidate alone: L2 is flushed, then
the candidate runs on arguments nothing has touched since. That is the state
the first kernel of a sequence finds. A later kernel finds another one: the
kernels before it have just written (or read) its operands, so they are in
L2, wholly when they fit and their tail when they do not, and the lines the
flush dirtied are partly gone. Tile shapes and layouts can rank in opposite
orders in the two states. Of 19 configs of a padded 3x3 max pool behind the
BatchNorm that wrote its 12.8 MB input, the one fastest alone (8.1 us) was
among the last five in the sequence, 10% behind the sequence's fastest,
which read 17% slower alone (B200; rank correlation between the two states
-0.81). A search that times such a kernel alone picks for a state the
kernel never runs in.

A *prelude* names the work that precedes the candidate: the earlier kernel
calls of the sequence, each a compiled function with its arguments.
``timed_behind(prelude)`` makes every search started inside it run that work
between the L2 flush and each timed call of a candidate, and hand the
candidate the tensors the work produced. The timed quantity stays the
candidate alone (``benchmarking.do_bench`` says how each timer leaves the
prelude out). ``KernelSequence`` builds preludes for a harness that runs
kernels one after another::

    sequence = KernelSequence()
    for kernel, args in calls:  # args: positional, defaults filled in
        with timed_behind(sequence.prelude(args)):
            output = kernel(*args)  # searches here on its first call
        sequence.record(compiled_function_of(kernel), args, output)

The arguments are the ones a compiled function takes, as the search of the
call has them (``Kernel.normalize_args``: positional, with the defaults
filled in): the worker calls the steps' functions and every candidate on
them, and a search refuses a prelude whose timed call is not its own
(``BenchmarkPrelude.mismatch``).

Without a prelude (none named, or one with no steps) nothing changes: no
code of this module runs in a timer.

A search with a prelude takes every reading in its benchmark worker, as a
search that reads its candidates from the CUDA activity trace does: the
in-process timers have no prelude, and a reading without one is on another
scale. It therefore needs what the worker needs (subprocess benchmarking of
a single kernel on CUDA event or trace timers, one input bank, an NVIDIA
device) and a regime that flushes per call;
``benchmarking.check_prelude_regime`` and ``LocalBenchmarkProvider`` refuse
the rest, CUDA graph replays among it (a replay runs the candidate on the
arguments of its capture, not on the ones the work has just made).

What a prelude's work must be, and what it does to a search:

- The steps launch the same kernels on every run, as compiled kernels on
  fixed arguments do: the trace timer finds a candidate's kernels by the
  number of events the flush and the work issue before its marker, learned
  from two runs. Work whose two runs differ has no trace reading
  (``benchmarking.prelude_trace_unready_reason``).
- The worker holds one copy of the prelude's tensors for the whole search.
  Every run makes the steps' outputs anew, but a tensor the caller gave
  that a step or the candidate writes in place stays written for the next
  run and the next candidate, as when a sequence is replayed on the same
  buffers. Readings do not depend on that unless a kernel's time depends on
  its operands' values. The accuracy check is not behind the prelude: it
  runs every candidate alone on the search's own arguments, as always.
- Every window starts by displacing the L2 lines that earlier windows
  pinned (``benchmarking._release_l2_pins``), the final selection's too,
  whose finalists share the prelude's tensors instead of private copies.
- A search whose backend's searches can read candidates from the CUDA
  activity trace moves to it after its initial population, whatever the
  kernel's length (``PopulationBasedSearch._settle_candidate_timer``).
  Behind work that the device finishes before the host has queued the
  candidate, a sample between two CUDA events also holds the candidate's
  launch on the host (``benchmarking.do_bench``); a search of a backend
  that reads by events only keeps that offset, alike for every config.
- The worker loads the prelude and compiles its steps in the first job
  that names it, inside that job's compile deadline, as it loads the
  kernel arguments. It holds the caller's tensors a second time (the
  candidate's own arguments stay loaded for the accuracy check) and two
  generations of the steps' outputs.
- A prelude that does not load, or whose work fails before a candidate's
  first call, fails that candidate's job with ``PreludeFailed``: the
  candidate has no reading, and the next candidate's job tries again.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.utils._pytree import tree_flatten
from torch.utils._pytree import tree_unflatten

from .kernel_args import _input_tree_leaves
from .kernel_args import _make_portable_args
from .kernel_args import resolve_portable_args
from .precompile_future import _load_compiled_fn
from .precompile_future import _serialize_compiled_fn
from .tensor_serialization import save_tensor_tree

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Sequence


class PreludeUnsupported(Exception):
    """A prelude cannot stand for this call (an argument, or an earlier
    call's, views an earlier output as another dtype), or cannot reach the
    benchmark worker."""


class PreludeFailed(RuntimeError):
    """A prelude did not load in the benchmark worker, or its work raised
    before the candidate's first call: the failure is the prelude's, not the
    candidate's, whose job has no reading."""


@dataclasses.dataclass(frozen=True, slots=True)
class PreludeOutput:
    """Stand-in, among the arguments of a prelude step or of the timed call,
    for a tensor that an earlier step produces: tensor ``leaf`` (in pytree
    order) of the output of step ``step``, viewed with ``size``, ``stride``
    and ``storage_offset`` (the consumer may take a view of the output).
    ``dtype`` is the output's, where the maker knows it (a sequence does):
    a search compares it with its own argument's. It is a note on the
    stand-in, not part of what it stands for: two stand-ins for one view
    are equal with or without it."""

    step: int
    leaf: int
    size: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: torch.dtype | None = dataclasses.field(default=None, compare=False)

    def resolve(self, outputs: Sequence[Sequence[torch.Tensor]]) -> torch.Tensor:
        tensor = outputs[self.step][self.leaf]
        if (
            tuple(tensor.shape) == self.size
            and tuple(tensor.stride()) == self.stride
            and tensor.storage_offset() == self.storage_offset
        ):
            return tensor
        return tensor.as_strided(self.size, self.stride, self.storage_offset)


@dataclasses.dataclass(frozen=True, eq=False)
class PreludeStep:
    """One earlier kernel call: the compiled function of the kernel at the
    config it runs with, and its arguments (``PreludeOutput`` where an
    earlier step produces one)."""

    fn: Callable[..., object]
    args: tuple[object, ...]


def _tensor_leaves(value: object) -> list[torch.Tensor]:
    leaves, _spec = tree_flatten(value)
    return [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]


def _resolved(
    args: Sequence[object], outputs: Sequence[Sequence[torch.Tensor]]
) -> list[object]:
    """``args`` with every ``PreludeOutput`` replaced by its tensor."""
    leaves, spec = tree_flatten(list(args))
    return cast(
        "list[object]",
        tree_unflatten(
            [
                leaf.resolve(outputs) if isinstance(leaf, PreludeOutput) else leaf
                for leaf in leaves
            ],
            spec,
        ),
    )


class LoadedPrelude:
    """A prelude ready to run in this process: ``run`` makes the steps'
    calls and keeps the timed call's arguments, ``timed(fn)`` is the timed
    call of a candidate on them. One instance serves every candidate of a
    search (the work does not depend on the candidate), so a timer may key
    what it learns about the work on the instance."""

    def __init__(
        self,
        steps: Sequence[tuple[Callable[..., object], Sequence[object]]],
        args: Sequence[object],
    ) -> None:
        self._steps = list(steps)
        self._args = list(args)
        # The outputs of the latest run stay alive until the next one has
        # made its own, so a run never writes where the timed call still
        # reads.
        self._outputs: list[list[torch.Tensor]] = []
        self.args: list[object] | None = None

    def __len__(self) -> int:
        return len(self._steps)

    def run(self) -> None:
        outputs: list[list[torch.Tensor]] = []
        for fn, args in self._steps:
            outputs.append(_tensor_leaves(fn(*_resolved(args, outputs))))
        self.args = _resolved(self._args, outputs)
        self._outputs = outputs

    def timed(self, fn: Callable[..., object]) -> Callable[[], object]:
        def call() -> object:
            assert self.args is not None, "the prelude has not run"
            return fn(*self.args)

        return call


@dataclasses.dataclass(frozen=True, eq=False)
class BenchmarkPrelude:
    """The work a search's candidates are timed behind: ``steps``, and the
    timed call's arguments ``args`` (``PreludeOutput`` where a step produces
    one; the other arguments are the caller's tensors, which the steps'
    arguments may alias, as in the sequence)."""

    steps: tuple[PreludeStep, ...]
    args: tuple[object, ...]

    def loaded(self) -> LoadedPrelude:
        """The prelude as this process runs it (the steps' own functions)."""
        return LoadedPrelude([(step.fn, step.args) for step in self.steps], self.args)

    def mismatch(self, args: Sequence[object]) -> str | None:
        """How the timed call's arguments differ from ``args``, the arguments
        of a search that would time its candidates behind this prelude; None
        when they are that call's. The worker runs every candidate on the
        prelude's arguments, so they must be the ones the search specialized
        and checked its candidates on: the same tree, every tensor with the
        search's shape, strides and dtype (a stand-in with its view's), and
        every other leaf of the same type, equal where it is a plain value."""
        mine, my_spec = tree_flatten(list(self.args))
        theirs, their_spec = tree_flatten(list(args))
        if my_spec != their_spec:
            return (
                f"its timed call takes {len(self.args)} argument(s) of another "
                f"structure than the search's {len(args)}"
            )
        for index, (own, other) in enumerate(zip(mine, theirs, strict=True)):
            if isinstance(other, torch.Tensor):
                if isinstance(own, PreludeOutput):
                    dtype = other.dtype if own.dtype is None else own.dtype
                    layout = (own.size, own.stride, dtype)
                elif isinstance(own, torch.Tensor):
                    layout = (tuple(own.shape), tuple(own.stride()), own.dtype)
                else:
                    return f"leaf {index} is no tensor where the search's is one"
                expected = (tuple(other.shape), tuple(other.stride()), other.dtype)
                if layout != expected:
                    return (
                        f"tensor leaf {index} is {layout}, the search's {expected} "
                        "(shape, strides, dtype)"
                    )
            elif isinstance(own, (PreludeOutput, torch.Tensor)):
                return f"leaf {index} is a tensor where the search's is none"
            elif type(own) is not type(other) or (
                isinstance(other, (bool, int, float, str))
                and own != other
                and not (own != own and other != other)  # (two NaNs)
            ):
                return f"leaf {index} is {own!r}, the search's {other!r}"
        return None

    def save(self, path: str) -> None:
        """Write the prelude for the benchmark worker (``load_prelude``): the
        steps' functions as the worker loads a candidate's, and one tree of
        all arguments, so tensors that share storage across the steps and
        the timed call still share it there. Raises ``PreludeUnsupported``
        when a function or an argument cannot cross."""
        trees = [list(self.args), *(list(step.args) for step in self.steps)]
        try:
            specs = [_serialize_compiled_fn(step.fn) for step in self.steps]
            portable = _make_portable_args(trees)
            save_tensor_tree(
                {"functions": specs, "args": trees if portable is None else portable},
                path,
            )
        except Exception as error:
            raise PreludeUnsupported(
                f"the benchmark prelude cannot reach the benchmark worker: "
                f"{type(error).__name__}: {error}"
            ) from error


@functools.cache
def load_prelude(path: str) -> LoadedPrelude:
    """The prelude ``BenchmarkPrelude.save`` wrote, once per process: every
    job of the worker runs the same steps on the same tensors. The file is
    a trusted temporary artifact of the parent autotuner process, as the
    kernel arguments are (``kernel_args.load_trusted_kernel_args``)."""
    payload = torch.load(path, weights_only=False)
    trees = resolve_portable_args(payload["args"])
    functions = [_load_compiled_fn(spec) for spec in payload["functions"]]
    return LoadedPrelude(
        list(zip(functions, cast("list[list[object]]", trees[1:]), strict=True)),
        cast("list[object]", trees[0]),
    )


class KernelSequence:
    """The kernel calls of a sequence so far, from which the prelude of the
    next kernel is built (module docstring). Tensors are followed by their
    storage: an argument whose storage is an earlier call's output becomes
    that output (a ``PreludeOutput``, with the argument's view of it), and
    every other tensor stays itself, so a buffer that one call fills and a
    later one reads is the same buffer in the prelude. A view of an output
    as another dtype has no stand-in, and neither has an output that a
    callback or a record among the arguments holds (a closure cell, a
    default, a field: no pytree leaf): the kernel about to be called gets
    no prelude (``PreludeUnsupported``), and neither does any call after it.
    The call that read such a tensor cannot be replayed on what the earlier
    calls make (it would read, and write, the tensor it was given, a copy
    the later calls never see), so the sequence stops making preludes
    there. The sequence keeps the outputs alive: an address is one tensor
    for as long as it is used. Arguments are given as a compiled function
    takes them (module docstring)."""

    def __init__(self) -> None:
        self._steps: list[PreludeStep] = []
        # storage address -> (step, leaf, the output tensor)
        self._produced: dict[int, tuple[int, int, torch.Tensor]] = {}
        # storage addresses of the tensors the sequence was given
        self._given: set[int] = set()
        # why no later call gets a prelude (a recorded call that cannot be
        # replayed), None while every call can
        self._unreplayable: str | None = None

    def __len__(self) -> int:
        return len(self._steps)

    def _stand_ins(self, args: Sequence[object], *, given: bool) -> tuple[object, ...]:
        leaves, spec = tree_flatten(list(args))
        # A callback or a record among the arguments may hold a tensor too
        # (a closure cell, a default, a field). It is no pytree leaf, so it
        # cannot become a stand-in: where it is an earlier output, the call
        # would be replayed on the copy it was given, also when the same
        # output is an argument of its own beside it.
        for leaf in leaves:
            if isinstance(leaf, torch.Tensor):
                continue
            for held in _input_tree_leaves([leaf]):
                if (
                    isinstance(held, torch.Tensor)
                    and held.numel() > 0
                    and held.untyped_storage().data_ptr() in self._produced
                ):
                    reason = (
                        "a callback or a record among the arguments holds the "
                        "output of an earlier kernel call"
                    )
                    if not given:
                        raise PreludeUnsupported(reason)
                    self._unreplayable = f"an earlier kernel call's arguments: {reason}"
        out: list[object] = []
        for leaf in leaves:
            if not isinstance(leaf, torch.Tensor) or leaf.numel() == 0:
                out.append(leaf)
                continue
            address = leaf.untyped_storage().data_ptr()
            produced = self._produced.get(address)
            if produced is None:
                if given:
                    self._given.add(address)
                out.append(leaf)
                continue
            step, index, tensor = produced
            if leaf.dtype != tensor.dtype:
                if given:
                    # A recorded call that read an earlier output as another
                    # dtype: the calls after it get no prelude (class
                    # docstring), so the tensor it keeps is never replayed.
                    self._unreplayable = (
                        f"an earlier kernel call read the {tensor.dtype} output "
                        f"of a call before it as {leaf.dtype}"
                    )
                    out.append(leaf)
                    continue
                raise PreludeUnsupported(
                    f"an argument views the {tensor.dtype} output of an earlier "
                    f"kernel call as {leaf.dtype}"
                )
            out.append(
                PreludeOutput(
                    step,
                    index,
                    tuple(int(size) for size in leaf.shape),
                    tuple(int(stride) for stride in leaf.stride()),
                    int(leaf.storage_offset()),
                    leaf.dtype,
                )
            )
        return tuple(cast("list[object]", tree_unflatten(out, spec)))

    def prelude(self, args: Sequence[object]) -> BenchmarkPrelude:
        """The prelude of a kernel about to be called with ``args``: every
        call recorded so far, in order. ``PreludeUnsupported`` when a call
        cannot be replayed or an argument has no stand-in."""
        if self._unreplayable is not None:
            raise PreludeUnsupported(self._unreplayable)
        return BenchmarkPrelude(tuple(self._steps), self._stand_ins(args, given=False))

    def record(
        self, fn: Callable[..., object], args: Sequence[object], output: object
    ) -> None:
        """Add a call that has run: ``fn`` is the compiled function of the
        kernel at the config it ran with, ``output`` what it returned."""
        step = len(self._steps)
        self._steps.append(PreludeStep(fn, self._stand_ins(args, given=True)))
        for index, tensor in enumerate(_tensor_leaves(output)):
            if tensor.numel() == 0:
                continue
            address = tensor.untyped_storage().data_ptr()
            # An output that is one of the call's arguments (written in
            # place) or an earlier output stays what it was.
            if address not in self._given and address not in self._produced:
                self._produced[address] = (step, index, tensor)


_active: contextvars.ContextVar[BenchmarkPrelude | None] = contextvars.ContextVar(
    "helion_benchmark_prelude", default=None
)


@contextlib.contextmanager
def timed_behind(prelude: BenchmarkPrelude | None) -> Iterator[None]:
    """Time the candidates of every search started inside the block behind
    ``prelude`` (module docstring). None, or a prelude without steps, names
    nothing: searches inside time their candidates alone, as outside."""
    if prelude is not None and not prelude.steps:
        prelude = None
    token = _active.set(prelude)
    try:
        yield
    finally:
        _active.reset(token)


def active_prelude() -> BenchmarkPrelude | None:
    """The prelude of the enclosing ``timed_behind`` block, None outside."""
    return _active.get()
