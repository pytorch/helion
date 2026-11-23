"""helion.inductor's kernels tuned behind the kernel before them
(``helion.inductor.kernel_sequence``), with stand-in kernels on CPU;
``test_inductor_backend.py`` checks a compiled graph's on a GPU."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from helion import exc
from helion._compat import requires_torch_version

if not requires_torch_version("2.16"):
    # helion.inductor raises ImportError below torch 2.16 (helion/inductor/__init__.py).
    pytest.skip("helion.inductor needs torch 2.16", allow_module_level=True)

# After the check: torch 2.13 and 2.14 may not have it.
from torch._inductor.codegen.wrapper import CommentLine

from helion.autotuner.benchmark_prelude import PreludeRefused
from helion.autotuner.benchmark_prelude import PreludeSearchFailed
from helion.autotuner.benchmark_prelude import active_prelude
from helion.inductor import kernel_sequence
from helion.inductor.codegen.scheduling import HelionScheduling
from helion.inductor.codegen.sequence import module_key
from helion.inductor.codegen.sequence import note_call
from helion.inductor.codegen.sequence import predecessor
from helion.inductor.kernel_sequence import GraphKernels
from helion.inductor.kernel_sequence import in_sequence
from helion.runtime.ref_mode import RefMode
from helion.runtime.settings import default_autotuner_fn

pytestmark = pytest.mark.cpu_only


def fill(buffer: torch.Tensor, value: float) -> None:
    buffer.fill_(value)


def double_into(x: torch.Tensor, out: torch.Tensor) -> None:
    out.copy_(x * 2)


@pytest.fixture(autouse=True)
def timers_take_a_prelude(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stand-in kernels' searches could time behind a prelude
    (``benchmark_prelude_unsupported``, which the provider tests check) and
    read short kernels from the trace (``times_short_kernels``)."""
    monkeypatch.setattr(
        kernel_sequence,
        "benchmark_prelude_unsupported",
        lambda settings, config_spec, kernel: None,
    )
    monkeypatch.setattr(kernel_sequence, "times_short_kernels", lambda bound: True)


class _Kernel:
    """A stand-in for a Helion kernel: its calls note the prelude they are
    timed behind, and its compiled function is ``fn``."""

    def __init__(
        self,
        fn: object,
        autotuner_fn: object = default_autotuner_fn,
        ref_mode: RefMode = RefMode.OFF,
    ) -> None:
        self.fn = fn
        self.autotuner_fn = autotuner_fn
        self.ref_mode = ref_mode
        self.preludes: list[object] = []
        self.name = getattr(fn, "__name__", "kernel")

    def normalize_args(self, *args: object) -> tuple[object, ...]:
        return args

    def __call__(self, *args: object) -> object:
        self.preludes.append(active_prelude())
        return self.fn(*args)  # pyrefly: ignore [not-callable]

    def bind(self, args: tuple[object, ...]) -> SimpleNamespace:
        def compile_config(allow_print: bool) -> object:
            if self.ref_mode != RefMode.OFF:
                # (no config: the call ran eagerly)
                raise RuntimeError("no config provided and no implicit config")
            return self.fn

        return SimpleNamespace(
            compile_config=compile_config,
            settings=SimpleNamespace(
                autotuner_fn=self.autotuner_fn, ref_mode=self.ref_mode
            ),
            config_spec=None,
        )


def _module(autotuner_fn: object = default_autotuner_fn) -> dict[str, object]:
    module: dict[str, object] = {}
    for name, fn, after in (("first", fill, None), ("second", double_into, "first")):
        module[name] = in_sequence(  # pyrefly: ignore
            _Kernel(fn, autotuner_fn), module, name, after=after
        )
    return module


def test_a_first_call_is_timed_behind_the_call_before() -> None:
    module = _module()
    first, second = module["first"], module["second"]
    buffer, out = torch.zeros(4), torch.zeros(4)
    module["first"](buffer, 3.0)  # pyrefly: ignore [not-callable]
    module["second"](buffer, out)  # pyrefly: ignore [not-callable]
    assert torch.equal(out, torch.full([4], 6.0))
    assert first.kernel.preludes == [None]  # pyrefly: ignore [missing-attribute]
    (prelude,) = second.kernel.preludes  # pyrefly: ignore [missing-attribute]
    (step,) = prelude.steps
    # The buffer the first call filled is the one the second reads.
    assert step.fn is fill and step.args[0] is buffer and prelude.args[0] is buffer
    # The module calls the kernels themselves from now on, and keeps no call.
    assert module["first"] is first.kernel  # pyrefly: ignore [missing-attribute]
    assert module["second"] is second.kernel  # pyrefly: ignore [missing-attribute]
    graph = module["_helion_graph_kernels"]
    assert isinstance(graph, GraphKernels) and graph.last is None


def test_later_calls_skip_the_sequence() -> None:
    module = _module()
    first = module["first"]
    buffer = torch.zeros(4)
    for value in (1.0, 2.0):
        module["first"](buffer, value)  # pyrefly: ignore [not-callable]
    # The second call went to the kernel itself, outside any prelude block.
    assert first.kernel.preludes == [None, None]  # pyrefly: ignore [missing-attribute]
    graph = module["_helion_graph_kernels"]
    assert isinstance(graph, GraphKernels) and graph.launched == 1


def _second_kernel_preludes(module: dict[str, object]) -> list[object]:
    """The preludes the second kernel's first call was timed behind."""
    buffer, out = torch.zeros(4), torch.zeros(4)
    module["first"](buffer, 3.0)  # pyrefly: ignore [not-callable]
    module["second"](buffer, out)  # pyrefly: ignore [not-callable]
    assert torch.equal(out, torch.full([4], 6.0))
    return module["second"].preludes  # pyrefly: ignore [missing-attribute]


def test_a_kernel_whose_search_cannot_time_behind_a_prelude_is_tuned_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HELION_AUTOTUNE_L2_FLUSH=0, graph replays, timing in this process or a
    distributed kernel (``benchmark_prelude_unsupported``), and a custom
    autotuner, whose provider may not run a prelude: the search would refuse
    one, so the second kernel is called outside any prelude block."""
    assert _second_kernel_preludes(_module(autotuner_fn=lambda *args: None)) == [None]
    monkeypatch.setattr(
        kernel_sequence,
        "benchmark_prelude_unsupported",
        lambda settings, config_spec, kernel: "the regime does not flush per call",
    )
    assert _second_kernel_preludes(_module()) == [None]


def test_a_kernel_that_skips_the_first_run_is_tuned_alone_later() -> None:
    """Three kernels, the middle one in a branch the first run does not
    take: the call kept for a prelude goes once the last kernel the module
    defines has run, and the middle kernel's first call, in a later run, is
    tuned alone rather than behind a call of the first run."""
    module: dict[str, object] = {}
    for name, fn, after in (
        ("first", fill, None),
        ("branch", double_into, "first"),
        ("last", fill, "branch"),
    ):
        module[name] = in_sequence(  # pyrefly: ignore
            _Kernel(fn), module, name, after=after
        )
    branch, last = module["branch"], module["last"]
    buffer, out = torch.zeros(4), torch.zeros(4)
    module["first"](buffer, 3.0)  # pyrefly: ignore [not-callable]
    module["last"](buffer, 1.0)  # pyrefly: ignore [not-callable]
    # The call before the last kernel's is the first kernel's, not the
    # branch's it names: tuned alone.
    assert last.kernel.preludes == [None]  # pyrefly: ignore [missing-attribute]
    graph = module["_helion_graph_kernels"]
    assert isinstance(graph, GraphKernels) and graph.last is None
    module["branch"](buffer, out)  # pyrefly: ignore [not-callable]
    assert branch.kernel.preludes == [None]  # pyrefly: ignore [missing-attribute]


def test_a_kernel_whose_search_reads_events_is_tuned_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A search that reads its candidates between CUDA events (Triton's)
    could not resolve a short kernel behind the call before: the second
    kernel is tuned alone."""
    monkeypatch.setattr(kernel_sequence, "times_short_kernels", lambda bound: False)
    module = _module()
    buffer, out = torch.zeros(4), torch.zeros(4)
    module["first"](buffer, 3.0)  # pyrefly: ignore [not-callable]
    second = module["second"]
    module["second"](buffer, out)  # pyrefly: ignore [not-callable]
    assert second.kernel.preludes == [None]  # pyrefly: ignore [missing-attribute]


def test_a_kernel_after_no_kernel_is_tuned_alone() -> None:
    """``after=None`` (another launch came between the two calls, or the call
    before was a later call of its kernel): the kernel is tuned alone, and
    the call before it, which no kernel names, is not kept."""
    module: dict[str, object] = {}
    for name, fn in (("first", fill), ("second", double_into)):
        module[name] = in_sequence(_Kernel(fn), module, name)  # pyrefly: ignore
    second = module["second"]
    graph = module["_helion_graph_kernels"]
    assert isinstance(graph, GraphKernels) and not graph.wanted
    buffer, out = torch.zeros(4), torch.zeros(4)
    module["first"](buffer, 3.0)  # pyrefly: ignore [not-callable]
    assert graph.last is None
    module["second"](buffer, out)  # pyrefly: ignore [not-callable]
    assert second.kernel.preludes == [None]  # pyrefly: ignore [missing-attribute]


@pytest.mark.parametrize(
    "failure",
    [
        PreludeSearchFailed(": 3 of the 3 tested configs had no reading"),
        PreludeRefused("a benchmark prelude requires arguments that can cross"),
    ],
)
def test_a_search_that_fails_behind_the_prelude_is_tuned_alone(
    failure: Exception,
) -> None:
    """A prelude that failed in the benchmark worker and left the search
    without a config, or that the provider refused: the kernel is tuned
    alone."""

    class _Fragile(_Kernel):
        def __call__(self, *args: object) -> object:
            if active_prelude() is not None:
                self.preludes.append(active_prelude())
                raise failure
            return super().__call__(*args)

    module: dict[str, object] = {}
    module["first"] = in_sequence(_Kernel(fill), module, "first")  # pyrefly: ignore
    second = module["second"] = in_sequence(  # pyrefly: ignore
        _Fragile(double_into), module, "second", after="first"
    )
    buffer, out = torch.zeros(4), torch.zeros(4)
    module["first"](buffer, 3.0)  # pyrefly: ignore [not-callable]
    module["second"](buffer, out)  # pyrefly: ignore [not-callable]
    assert torch.equal(out, torch.full([4], 6.0))
    preludes = second.kernel.preludes  # pyrefly: ignore [missing-attribute]
    assert len(preludes) == 2 and preludes[0] is not None and preludes[1] is None


@pytest.mark.parametrize(
    "failure", [exc.NoConfigFound(), exc.InvalidAPIUsage("a codegen error")]
)
def test_a_kernel_that_fails_behind_the_prelude_on_its_own_is_not_tuned_again(
    failure: Exception,
) -> None:
    """A search without a working config, or a kernel that does not compile,
    fails behind the prelude as it would alone: it is not searched again."""

    class _Broken(_Kernel):
        def __call__(self, *args: object) -> object:
            self.preludes.append(active_prelude())
            raise failure

    module: dict[str, object] = {}
    module["first"] = in_sequence(_Kernel(fill), module, "first")  # pyrefly: ignore
    second = module["second"] = in_sequence(  # pyrefly: ignore
        _Broken(double_into), module, "second", after="first"
    )
    buffer, out = torch.zeros(4), torch.zeros(4)
    module["first"](buffer, 3.0)  # pyrefly: ignore [not-callable]
    with pytest.raises(type(failure)):
        module["second"](buffer, out)  # pyrefly: ignore [not-callable]
    preludes = second.kernel.preludes  # pyrefly: ignore [missing-attribute]
    assert len(preludes) == 1 and preludes[0] is not None
    # The failed call keeps the first kernel's call (its arguments) no longer.
    graph = module["_helion_graph_kernels"]
    assert isinstance(graph, GraphKernels) and graph.last is None


def test_a_kernel_that_fails_to_bind_lets_the_call_before_go() -> None:
    """A first call that fails before it runs (its kernel does not bind)
    keeps the call before it, and that call's tensors, no longer."""

    class _Unbound(_Kernel):
        def bind(self, args: tuple[object, ...]) -> SimpleNamespace:
            raise exc.InvalidAPIUsage("a binding error")

    module: dict[str, object] = {}
    module["first"] = in_sequence(_Kernel(fill), module, "first")  # pyrefly: ignore
    module["second"] = in_sequence(  # pyrefly: ignore
        _Unbound(double_into), module, "second", after="first"
    )
    buffer, out = torch.zeros(4), torch.zeros(4)
    module["first"](buffer, 3.0)  # pyrefly: ignore [not-callable]
    graph = module["_helion_graph_kernels"]
    assert isinstance(graph, GraphKernels) and graph.last is not None
    with pytest.raises(exc.InvalidAPIUsage):
        module["second"](buffer, out)  # pyrefly: ignore [not-callable]
    assert graph.last is None


class _Wrapper:
    """The ``lines`` of Inductor's wrapper codegen and its root (a graph
    partition's or a subgraph's is its parent's), as ``codegen.sequence``
    reads them."""

    def __init__(self, parent: _Wrapper | None = None) -> None:
        self.lines: list[object] = []
        self.parent = parent

    def get_root_graph(self) -> _Wrapper:
        return self if self.parent is None else self.parent.get_root_graph()

    def call(self, name: str) -> None:
        self.lines.append(f"{name}(buf0, buf1)")
        note_call(self, name, ["buf0", "buf1"])  # pyrefly: ignore [bad-argument-type]


def _after(wrapper: _Wrapper) -> str | None:
    return predecessor(wrapper)  # pyrefly: ignore [bad-argument-type]


def test_a_first_call_is_named_behind_the_helion_call_just_before_it() -> None:
    """Frees, comments and ``del`` lines between two Helion kernel calls
    launch nothing: the second kernel's first call is timed behind the
    first's."""
    wrapper = _Wrapper()
    assert _after(wrapper) is None
    wrapper.call("helion_k0")
    wrapper.lines += [
        CommentLine("# Topologically Sorted Source Nodes: [sum_1]"),
        "del buf0",
        "",
    ]
    assert _after(wrapper) == "helion_k0"


def test_a_first_call_after_another_launch_is_tuned_alone() -> None:
    """An extern kernel between the two calls, or a later call of the kernel
    before (the runtime records only first calls: ``K(a); K(b); L(c)``
    would time ``L`` behind ``K(a)``): no kernel is named."""
    wrapper = _Wrapper()
    wrapper.call("helion_k0")
    wrapper.lines.append("extern_kernels.mm(buf1, arg1_1, out=buf2)")
    assert _after(wrapper) is None
    wrapper.call("helion_k1")
    assert _after(wrapper) == "helion_k1"
    wrapper.call("helion_k1")
    assert _after(wrapper) is None


def test_a_first_call_across_graph_partitions_is_tuned_alone() -> None:
    """Partitions of one graph share its kernels: K's call in the second
    partition is not its first, and the call before a partition's first
    call is in another wrapper, across the op that split the graph."""
    root = _Wrapper()
    first, second = _Wrapper(root), _Wrapper(root)
    first.call("helion_k")
    assert _after(second) is None
    second.call("helion_k")
    assert _after(second) is None
    second.call("helion_l")
    assert _after(second) == "helion_l"


def test_a_kernel_behind_another_loads_a_module_of_its_own() -> None:
    """Graphs with one kernel source share its PyCodeCache module, and so
    the config the first to call it selects: a kernel timed behind another
    loads its module under the other kernel's source."""
    alone = HelionScheduling.format_kernel_definition("helion_k", "src")
    assert "helion_k = PyCodeCache.load(helion_k_source).kernel" in alone
    assert 'helion_in_sequence(helion_k, globals(), "helion_k", after=None)' in alone
    behind = HelionScheduling.format_kernel_definition(
        "helion_l", "src", "helion_k", "0123456789abcdef"
    )
    assert (
        "helion_l = PyCodeCache.load(helion_l_source, '0123456789abcdef').kernel"
        in behind
    )
    assert "after='helion_k'" in behind


def test_reference_mode_runs_the_kernels_without_a_sequence() -> None:
    """HELION_INTERPRET=1 runs each kernel eagerly: no config to record
    for a prelude, so no kernel is timed behind another."""
    module: dict[str, object] = {}
    for name, fn, after in (("first", fill, None), ("second", double_into, "first")):
        module[name] = in_sequence(  # pyrefly: ignore
            _Kernel(fn, ref_mode=RefMode.EAGER), module, name, after=after
        )
    second = module["second"]
    buffer, out = torch.zeros(4), torch.zeros(4)
    module["first"](buffer, 3.0)  # pyrefly: ignore [not-callable]
    module["second"](buffer, out)  # pyrefly: ignore [not-callable]
    assert torch.equal(out, torch.full([4], 6.0))
    assert second.kernel.preludes == [None]  # pyrefly: ignore [missing-attribute]


def test_a_kernel_behind_another_keys_its_module_on_that_kernels_key() -> None:
    """C behind B behind A and C behind B behind D: B runs at the config
    selected behind A in one graph and behind D in the other, so the two
    Cs load modules of their own, though B's source is the same."""
    keys = []
    for first in ("a_source", "d_source"):
        wrapper = _Wrapper()
        module_key(wrapper, "helion_b", "helion_x", first, [])  # pyrefly: ignore
        keys.append(module_key(wrapper, "helion_c", "helion_b", "b_source", []))  # pyrefly: ignore
    assert keys[0] != keys[1]
    # A kernel behind one that was tuned alone keys on its source alone.
    wrapper = _Wrapper()
    alone = module_key(wrapper, "helion_c", "helion_b", "b_source", [])  # pyrefly: ignore
    assert alone not in keys and len(alone) == 16


def test_a_kernel_behind_another_keys_its_module_on_the_buffers_they_share() -> None:
    """B in ``A(x, tmp); B(tmp, out)`` reads the buffer A has just written,
    and in ``A(x, tmp); B(y, out)`` one A did not touch: the two load modules
    of their own, though the sources are the same.  B passing A's buffer
    through a view shares it as well."""

    def key(b_args: list[str]) -> str:
        wrapper = _Wrapper()
        note_call(wrapper, "helion_a", ["x", "tmp"])  # pyrefly: ignore
        return module_key(wrapper, "helion_b", "helion_a", "a_source", b_args)  # pyrefly: ignore

    behind = key(["tmp", "out"])
    assert behind != key(["y", "out"])
    assert behind == key(["tmp.view(4, 4)", "out"])
