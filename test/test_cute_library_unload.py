"""Where a compiled CuTe module's CUDA library is unloaded.

The CuTe DSL unloads it from ``CudaDialectJitModule.__del__``, wherever the
collector runs before interpreter shutdown.  CUDA refuses ``cudaLibraryUnload`` while
a CUDA graph capture is underway (in the default capture mode from every thread)
and the refused call invalidates the capture, so the benchmark worker unloads a
candidate's libraries when its job ends (``_unload_compiled_fn``) and the
finalizer makes no driver call while a capture may be underway.  Both ask
and call under the capture gate, which a ``cute_cuda_graph`` capture's
startup waits for, so that none begins between the question and the call.
These tests run the DSL's real module and executor classes over a recording
stand-in for the driver; the GPU cases are in
``test_cute_launcher_capture_ownership.py``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import logging
import sys
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch
import weakref

from cutlass.base_dsl.jit_executor import JitExecutor
import cutlass.cutlass_dsl.cuda_jit_executor as cuda_jit_executor
import pytest
import torch
from torch._inductor.codecache import PyCodeCache

import helion
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.zero_fill_kernels import zero_fill_kernel_name
from helion._compiler.zero_fill_kernels import zero_fill_kernel_names
from helion._testing.cute_capture_gate import STEP_TIMEOUT_S
from helion._testing.cute_capture_gate import CaptureOnAnotherThread
from helion._testing.cute_capture_gate import capture_stand_ins
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_zero_fill import COLUMN_SUM_CONFIG
from helion._testing.cute_zero_fill import atomic_column_sum
from helion.autotuner.precompile_future import SerializedCompiledFunction
from helion.autotuner.precompile_future import _load_compiled_fn
from helion.autotuner.precompile_future import _module_kernels
from helion.autotuner.precompile_future import _serialize_compiled_fn
from helion.autotuner.precompile_future import _source_hash_globals
from helion.autotuner.precompile_future import _unload_compiled_fn
import helion.language as hl
from helion.runtime.cute import compile_cache
from helion.runtime.cute import launch_caches
from helion.runtime.cute import launcher

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

pytestmark = pytest.mark.cpu_only

_CANDIDATE_SOURCE = (
    "class Kernel:\n"
    "    pass\n"
    "_helion_call = Kernel()\n"
    "def call():\n"
    "    return _helion_call\n"
)
# A candidate whose host zero fill launches a second kernel of the generated
# module (``cute_pdl``), under the name the code generator gives it.
_ZERO_FILL_CANDIDATE_SOURCE = (
    "class Kernel:\n"
    "    pass\n"
    "_helion_call = Kernel()\n"
    "_helion_zero_fill_0_helion_call = Kernel()\n"
    "def call():\n"
    "    return _helion_call, _helion_zero_fill_0_helion_call\n"
)


@dataclasses.dataclass(eq=False)
class _Library:
    """A library handle that records its unload among the events of the test
    that loaded it, wherever the unload is issued.  ``during_unload`` runs
    inside the driver call, before it is recorded."""

    name: str
    events: list[str]
    during_unload: Callable[[], None] | None = None

    def __int__(self) -> int:
        # The handle's value: ``CudaDialectJitModule.unload`` gives it to the
        # DSL's unload subscribers (``_notify_library_unload``) ahead of the
        # driver call, and swallows the error of a handle without one.
        return id(self)


class _Driver:
    """``cuda.bindings.runtime`` as the DSL's executor module sees it, with a
    ``cudaLibraryUnload`` that records instead of calling the driver."""

    def __init__(self, real: object) -> None:
        self._real = real

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)

    def cudaLibraryUnload(self, library: _Library) -> tuple[int]:
        if library.during_unload is not None:
            library.during_unload()
        library.events.append(f"unload {library.name}")
        return (0,)


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """The driver calls a test's modules issue, in order.  The collector is
    off for the test, so a module is finalized where the test says."""
    compile_cache.patch_cutlass_jit_shutdown_unload()
    # Before the stand-in takes the driver's place: garbage of earlier tests
    # (real modules, in a process that also ran GPU tests) is finalized
    # through the real driver.
    gc.collect()
    monkeypatch.setattr(
        cuda_jit_executor, "cuda_runtime", _Driver(cuda_jit_executor.cuda_runtime)
    )
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield []
    finally:
        gc.collect()
        if enabled:
            gc.enable()


class _Compiled(SimpleNamespace):
    """A compiled function; a call launches nothing, and raises ``fault``
    when the test gave it one."""

    fault: Exception | None = None

    def __call__(self, *args: object) -> None:
        if self.fault is not None:
            raise self.fault


def _launched(
    events: list[str],
    library: str,
    during_unload: Callable[[], None] | None = None,
) -> _Compiled:
    """A compiled function as its first launch leaves it: the DSL's module
    holding a loaded library, and its executor, which refers to itself, so
    that only the collector frees the two."""
    module = cuda_jit_executor.CudaDialectJitModule(
        None,
        SimpleNamespace(restype=None),
        None,
        [_Library(library, events, during_unload)],
    )
    return _Compiled(
        jit_module=module, _default_executor=JitExecutor(module, None, False)
    )


def _drop(compiled: SimpleNamespace) -> weakref.ReferenceType[object]:
    """Make ``compiled``'s module garbage that waits for the collector."""
    module = weakref.ref(compiled.jit_module)
    compiled.__dict__.clear()
    assert module() is not None
    return module


def _give_launcher(
    kernel: object,
    events: list[str],
    label: str = "",
    during_unload: Callable[[], None] | None = None,
) -> list[weakref.ReferenceType[object]]:
    """Leave ``kernel`` as a launch in the benchmark worker does: a compiled
    launcher holding a disk-reloaded function and the C-API twin of a fresh
    compile, each with a loaded library (``<label>reloaded``,
    ``<label>twin``), and the kernel's relaunch state.  ``during_unload``
    runs inside the unload of the first of the two.  Returns the two
    modules."""
    launcher = compile_cache.CompiledCuteLauncher(lambda: None, None)
    launcher._compiled = _launched(events, f"{label}reloaded", during_unload)
    launcher._capi_compiled = _launched(events, f"{label}twin")
    kernel._helion_cute_compiled_launchers = {"launch": launcher}  # type: ignore[attr-defined]
    kernel._helion_cute_last_launch_cache = SimpleNamespace(compiled=launcher)  # type: ignore[attr-defined]
    kernel._helion_cute_fastpath = SimpleNamespace(  # type: ignore[attr-defined]
        executor=launcher._compiled._default_executor
    )
    return [
        weakref.ref(launcher._compiled.jit_module),
        weakref.ref(launcher._capi_compiled.jit_module),
    ]


def _released(kernel: object) -> bool:
    """Whether ``_unload_compiled_fn`` emptied ``kernel``'s launcher cache
    and dropped its relaunch state."""
    return (
        kernel._helion_cute_compiled_launchers == {}  # type: ignore[attr-defined]
        and not hasattr(kernel, "_helion_cute_last_launch_cache")
        and not hasattr(kernel, "_helion_cute_fastpath")
    )


def _candidate(
    events: list[str],
    launched_on: tuple[int, ...] = (),
    during_unload: Callable[[], None] | None = None,
) -> tuple[Callable[[], object], list[weakref.ReferenceType[object]]]:
    """A candidate the benchmark worker loaded from the on-disk compile cache
    and launched, once with each device of ``launched_on`` current.
    ``during_unload`` runs inside the unload of its first library."""
    fn = _load_compiled_fn(
        SerializedCompiledFunction("call", _CANDIDATE_SOURCE, None, None)
    )
    kernel = fn()
    modules = _give_launcher(kernel, events, during_unload=during_unload)
    (launcher,) = kernel._helion_cute_compiled_launchers.values()
    for device in launched_on:
        with _device(device, events):
            launcher()
    return fn, modules


def _launch_that_raises(kernel: object, device: int, events: list[str]) -> None:
    """Launch ``kernel``'s compiled function with ``device`` current, as a
    launch that faults: the call raises."""
    (launcher,) = kernel._helion_cute_compiled_launchers.values()  # type: ignore[attr-defined]
    launcher._compiled.fault = RuntimeError(
        "CUDA error: an illegal memory access was encountered"
    )
    with (
        _device(device, events),
        pytest.raises(RuntimeError, match="illegal memory access"),
    ):
        launcher()


@contextlib.contextmanager
def _callers_stream() -> Iterator[None]:
    """The stream ``cute_cuda_graph`` notes at entry and sets again when
    its capture raises, without a device."""
    with (
        patch.object(torch.cuda, "current_stream", lambda: None),
        patch.object(torch.cuda, "set_stream", lambda stream: None),
    ):
        yield


@contextlib.contextmanager
def _managed_capture() -> Iterator[None]:
    """``cute_cuda_graph`` around a stand-in for ``torch.cuda.graph``: the
    scope during which the process counts a capture as underway."""
    with (
        patch.object(launch_caches, "CuteCUDAGraph", SimpleNamespace),
        patch.object(torch.cuda, "graph", lambda graph, **_: contextlib.nullcontext()),
        _callers_stream(),
        launch_caches.cute_cuda_graph(),
    ):
        yield


@contextlib.contextmanager
def _stream_capturing(capturing: bool | Exception) -> Iterator[None]:
    """An initialized CUDA whose current stream reads ``capturing``."""

    def query() -> bool:
        if isinstance(capturing, Exception):
            raise capturing
        return capturing

    with (
        patch.object(torch.cuda, "is_initialized", lambda: True),
        patch.object(torch.cuda, "is_current_stream_capturing", query),
    ):
        yield


@contextlib.contextmanager
def _device(
    index: int,
    events: list[str],
    *,
    synchronizes: bool = True,
    during_synchronize: Callable[[], None] | None = None,
) -> Iterator[None]:
    """An initialized CUDA with device ``index`` current.  A synchronize
    records the device it waits for among ``events`` and, unless
    ``synchronizes``, fails as after a launch that faulted;
    ``during_synchronize`` runs inside the wait, before it is recorded."""

    def synchronize(device: int | None = None) -> None:
        if during_synchronize is not None:
            during_synchronize()
        events.append(f"synchronize {index if device is None else device}")
        if not synchronizes:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

    with (
        _stream_capturing(False),
        patch.object(torch.cuda, "current_device", lambda: index),
        patch.object(torch.cuda, "synchronize", synchronize),
    ):
        yield


class _TorchGraph:
    """``torch.cuda.graph`` in the order of its entry: it synchronizes,
    collects (``torch.compiler.config.force_cudagraph_gc``; releases before
    that option always did) and only then begins the capture.  ``fails``
    names the step that raises; the begin and the end are recorded with
    whether the capture counted as underway there."""

    def __init__(self, events: list[str], fails: str = "") -> None:
        self.events = events
        self.fails = fails

    def __call__(self, graph: object, **_: object) -> _TorchGraph:
        return self

    def __enter__(self) -> None:
        self.events.append("synchronize")
        gc.collect()
        if self.fails == "begin":
            raise RuntimeError("capture_begin failed")
        counted = launch_caches.capture_may_be_active()
        self.events.append("capture_begin" if counted else "capture_begin, uncounted")

    def __exit__(self, *exc_info: object) -> None:
        counted = launch_caches.capture_may_be_active()
        self.events.append("capture_end" if counted else "capture_end, uncounted")
        if self.fails == "end":
            raise RuntimeError("capture_end failed")


@contextlib.contextmanager
def _torch_capture(events: list[str], fails: str = "") -> Iterator[None]:
    """``cute_cuda_graph`` around ``_TorchGraph``."""
    with (
        patch.object(launch_caches, "CuteCUDAGraph", SimpleNamespace),
        patch.object(torch.cuda, "graph", _TorchGraph(events, fails)),
        _callers_stream(),
        launch_caches.cute_cuda_graph(),
    ):
        yield


_CAPTURES = {
    "managed capture underway": _managed_capture,
    "current stream capturing": lambda: _stream_capturing(True),
    "capture query fails": lambda: _stream_capturing(RuntimeError("no context")),
}


def test_unload_compiled_fn_unloads_the_libraries_when_the_job_ends(
    events: list[str],
) -> None:
    fn, modules = _candidate(events)
    kernel = fn()

    _unload_compiled_fn(fn)

    assert events == ["unload reloaded", "unload twin"]
    assert _released(kernel)
    # The modules are garbage that only the collector frees ...
    assert all(module() is not None for module in modules)
    del fn, kernel
    gc.collect()
    assert all(module() is None for module in modules)
    # ... and its finalizers find nothing left to unload.
    assert events == ["unload reloaded", "unload twin"]


def test_release_drops_the_relaunch_state_the_launcher_sets() -> None:
    """The names ``drop_relaunch_state`` goes by are the launcher's own: the
    launcher sets and reads a kernel's relaunch state under the constants
    the drop goes by, and those are the attributes the launch tests find on
    a kernel the launcher launched and clear to make it launch afresh
    (``test_cute_fast_relaunch.py``)."""
    assert launch_caches.RELAUNCH_STATE_ATTRIBUTES == (
        launch_caches.LAST_LAUNCH_CACHE_ATTRIBUTE,
        launch_caches.FASTPATH_ATTRIBUTE,
    )
    assert launch_caches.RELAUNCH_STATE_ATTRIBUTES == (
        "_helion_cute_last_launch_cache",
        "_helion_cute_fastpath",
    )
    assert launcher._LAST_LAUNCH_CACHE_ATTRIBUTE == "_helion_cute_last_launch_cache"
    assert launcher._FASTPATH_ATTRIBUTE == "_helion_cute_fastpath"
    kernel = SimpleNamespace(
        _helion_cute_last_launch_cache=object(),
        _helion_cute_fastpath=object(),
        _helion_cute_launch_arg_cache={},
    )
    launch_caches.drop_relaunch_state(kernel)
    assert vars(kernel) == {"_helion_cute_launch_arg_cache": {}}
    # (a kernel that never relaunched has none)
    launch_caches.drop_relaunch_state(kernel)


def test_unload_compiled_fn_unloads_every_kernel_of_the_module(
    events: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A candidate with a host zero fill launches a second kernel of its
    generated module: its libraries, launcher cache and relaunch state go at
    job end as the main kernel's do, so no library waits for a collection
    inside the next job's capture and stays loaded there."""
    fn = _load_compiled_fn(
        SerializedCompiledFunction("call", _ZERO_FILL_CANDIDATE_SOURCE, None, None)
    )
    main, fill = fn()
    modules = _give_launcher(main, events, "main ")
    modules += _give_launcher(fill, events, "fill ")
    unloads = [
        "unload main reloaded",
        "unload main twin",
        "unload fill reloaded",
        "unload fill twin",
    ]

    _unload_compiled_fn(fn)

    assert events == unloads
    assert _released(main) and _released(fill)

    monkeypatch.setattr(compile_cache, "_SKIPPED_CAPTURE_UNLOADS", 0, raising=False)
    del fn, main, fill
    with _managed_capture():
        gc.collect()
    assert all(module() is None for module in modules)
    assert events == unloads
    assert compile_cache.skipped_capture_unloads() == 0


def test_unload_compiled_fn_finds_the_zero_fill_kernel_of_generated_code(
    events: list[str],
) -> None:
    """The names the teardown goes by are the code generator's: the module
    of a ``cute_pdl`` kernel with a host zero fill defines the fill kernel
    beside the kernel proper, and both are released."""
    with (
        mock_cuda_unavailable(),
        patch(
            "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = cpu_bind(
            atomic_column_sum, (torch.empty(1024, 1024, dtype=torch.bfloat16),)
        )
        source = bound.to_code(
            bound.config_spec.normalized_config(
                helion.Config.from_dict({**COLUMN_SUM_CONFIG, "cute_pdl": True})
            )
        )
    name = bound.kernel.name
    assert f"_launcher(_helion_{name}, " in source
    assert f"_launcher(_helion_zero_fill_0_helion_{name}, " in source
    fn = _load_compiled_fn(SerializedCompiledFunction(name, source, None, None))
    # The generator's name is the one the rule's own builder gives.
    assert zero_fill_kernel_names(fn.__globals__, name) == [
        zero_fill_kernel_name(0, f"_helion_{name}")
    ]
    kernels = [
        fn.__globals__[f"_helion_{name}"],
        fn.__globals__[f"_helion_zero_fill_0_helion_{name}"],
    ]
    for label, kernel in zip(("main ", "fill "), kernels, strict=True):
        _give_launcher(kernel, events, label)

    _unload_compiled_fn(fn)

    assert events == [
        "unload main reloaded",
        "unload main twin",
        "unload fill reloaded",
        "unload fill twin",
    ]
    assert all(_released(kernel) for kernel in kernels)


def test_one_rule_finds_a_modules_zero_fill_kernels() -> None:
    """The backend, which annotates a module's kernels and reads their
    compile-cache keys, the record of the kernels that carried the module's
    hash in the parent, and the release of a job's kernels find a module's
    zero-fill kernels by one rule, the name the code generator gives the
    fill kernels of the kernel named after the function. A fill kernel is a
    kernel of the module to all three. A kernel under another name, and the
    fill kernel of another kernel, is one to none of them, although each
    holds a compiled launcher and carries the module's hash as the fill
    kernel does. The rule is the names its builder builds and no other: the
    fill kernel of a kernel whose name ends in this one's, and names with
    anything but one decimal number in the number's place, are not found.
    Nor is the device kernel of a kernel whose own name begins with
    ``zero_fill_``: it begins with the prefix and is that kernel, not a fill
    kernel of it. That the code generator names its fill kernels with the
    builder is seen on generated code
    (``test_unload_compiled_fn_finds_the_zero_fill_kernel_of_generated_code``)."""
    module_hash = "the module's source hash"
    fill_name = "_helion_zero_fill_0_helion_call"
    second_name = "_helion_zero_fill_10_helion_call"
    foreign_name = "_helion_zero_fill_0_helion_other"
    assert zero_fill_kernel_name(0, "_helion_call") == fill_name
    assert zero_fill_kernel_name(10, "_helion_call") == second_name
    assert zero_fill_kernel_name(12, "_helion__private") == (
        "_helion_zero_fill_12_helion__private"
    )
    for own in ("zero_fill_x", "zero_fill_7", "zero_fill_0_helion_add"):
        assert zero_fill_kernel_names({f"_helion_{own}": object()}, own) == []
    # The number is compared as text: more digits than ``int`` converts
    # are answered like any other number.
    long_name = f"_helion_zero_fill_{'9' * 5000}_helion_call"
    assert zero_fill_kernel_names({long_name: object()}, "call") == [long_name]
    near_misses = (
        "_helion_zero_fill_0_helion_helion_call",
        "_helion_zero_fill_0_helion_my_call",
        "_helion_zero_fill__helion_call",
        "_helion_zero_fill_x_helion_call",
        "_helion_zero_fill_00_helion_call",
        "_helion_zero_fill_0_1_helion_call",
        "_helion_zero_fill_0_helion_call_",
        "x_helion_zero_fill_0_helion_call",
    )
    source = (
        "class Kernel:\n"
        "    def __init__(self):\n"
        "        self._helion_cute_compiled_launchers = {'launch': object()}\n"
        f"        self._helion_cute_source_hash = {module_hash!r}\n"
        "_helion_call = Kernel()\n"
        f"{fill_name} = Kernel()\n"
        f"{second_name} = Kernel()\n"
        "_helion_other = Kernel()\n"
        f"{foreign_name} = Kernel()\n"
        + "".join(f"{name} = Kernel()\n" for name in near_misses)
        + "def call():\n"
        "    pass\n"
        f"call._helion_cute_source_hash = {module_hash!r}\n"
    )
    fn = _load_compiled_fn(SerializedCompiledFunction("call", source, None, None))
    try:
        loaded = fn.__globals__
        kernel, fill, second, other, foreign = (
            loaded[name]
            for name in (
                "_helion_call",
                fill_name,
                second_name,
                "_helion_other",
                foreign_name,
            )
        )
        assert all(name in loaded for name in near_misses)
        assert zero_fill_kernel_names(loaded, "call") == [fill_name, second_name]
        assert zero_fill_kernel_names(loaded, "other") == [foreign_name]
        mine = [id(kernel), id(fill), id(second)]
        compiled = CuteBackend._compiled_kernels(fn, "call")
        assert [id(found) for found in compiled] == mine
        assert _source_hash_globals(fn, loaded, module_hash) == (
            fill_name,
            second_name,
        )
        released = _module_kernels(fn, sys.modules[fn.__module__])
        assert [id(found) for found in released] == mine
        assert len({*mine, id(other), id(foreign)}) == 5
    finally:
        _unload_compiled_fn(fn)


def test_a_job_keys_the_zero_fill_kernel_in_the_compile_cache_as_its_parent_does(
    events: list[str],
) -> None:
    """The backend gives the fill kernel of a ``cute_pdl`` module the
    module's source hash, which is what its on-disk compile-cache key starts
    from. The copy a worker job loads gets the hash for the same kernels, so
    a second job that launches the fill kernel finds the entry the first
    wrote (the parent's key, in every job). Without the names the fill
    kernel has no key and every job compiles it again."""
    with (
        mock_cuda_unavailable(),
        patch(
            "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = cpu_bind(
            atomic_column_sum, (torch.empty(1024, 1024, dtype=torch.bfloat16),)
        )
        source = bound.to_code(
            bound.config_spec.normalized_config(
                helion.Config.from_dict({**COLUMN_SUM_CONFIG, "cute_pdl": True})
            )
        )
    name = bound.kernel.name
    fill_name = f"_helion_zero_fill_0_helion_{name}"
    parent = PyCodeCache.load(source)
    bound.env.backend.annotate_compiled_module(parent, source, name)
    spec = _serialize_compiled_fn(getattr(parent, name))
    assert spec.source_hash is not None
    assert spec.source_hash_globals == (fill_name,)

    def key(kernel: object) -> str | None:
        plans = tuple(
            repr(plan) for plan in getattr(kernel, "_helion_cute_wrapper_plans", ())
        )
        return launcher._cute_disk_cache_key(
            kernel, (), (128, 1, 1), plans, None, "--enable-tvm-ffi", 148
        )

    parent_keys = [
        key(getattr(parent, f"_helion_{name}")),
        key(getattr(parent, fill_name)),
    ]
    assert None not in parent_keys
    # The key names the kernel: the two share the module's hash, and this
    # fixture hands both one schema and one block.
    assert parent_keys[0] != parent_keys[1]
    for _job in range(2):
        fn = _load_compiled_fn(spec)
        try:
            assert [
                key(fn.__globals__[f"_helion_{name}"]),
                key(fn.__globals__[fill_name]),
            ] == parent_keys
        finally:
            _unload_compiled_fn(fn)
    # What a job computed before the names were sent: the main kernel's key
    # and none for the fill kernel.
    fn = _load_compiled_fn(dataclasses.replace(spec, source_hash_globals=()))
    try:
        assert key(fn.__globals__[f"_helion_{name}"]) == parent_keys[0]
        assert key(fn.__globals__[fill_name]) is None
    finally:
        _unload_compiled_fn(fn)


def test_unload_compiled_fn_waits_for_the_devices_of_its_launches(
    events: list[str],
) -> None:
    """The job has left its device when it releases its candidate: the wait
    is for the devices the launcher launched on, not for the current one."""
    fn, _modules = _candidate(events, launched_on=(2, 1, 2))

    with _device(0, events):
        _unload_compiled_fn(fn)

    assert events == [
        "synchronize 1",
        "synchronize 2",
        "unload reloaded",
        "unload twin",
    ]


@pytest.mark.parametrize(
    "unsafe",
    ["device does not synchronize", "device of a launch that raised", *_CAPTURES],
)
def test_unload_compiled_fn_leaves_unsafe_unloads_to_the_finalizer(
    events: list[str], unsafe: str
) -> None:
    """After a launch that failed its device may not synchronize (kernels of
    the libraries may still run), and inside a capture the unload and the
    synchronize would each invalidate it: no driver call then, and the
    dropped modules' finalizers unload once that is safe."""
    if unsafe == "device of a launch that raised":
        # The only launch raises, with device 1 current.  The launcher took
        # note of the device before it launched: the wait is for that
        # device, where the failed launch may have left kernels running.
        fn, modules = _candidate(events)
        _launch_that_raises(fn(), 1, events)
    else:
        fn, modules = _candidate(events, launched_on=(1,))
    kernel = fn()

    if unsafe in ("device does not synchronize", "device of a launch that raised"):
        with _device(0, events, synchronizes=False):
            _unload_compiled_fn(fn)
        assert events == ["synchronize 1"]
        events.clear()
    else:
        with _device(0, events), _CAPTURES[unsafe]():
            _unload_compiled_fn(fn)
        assert events == []
    assert _released(kernel)
    assert all(module() is not None for module in modules)

    del fn, kernel
    gc.collect()
    assert all(module() is None for module in modules)
    assert sorted(events) == ["unload reloaded", "unload twin"]


def test_unload_libraries_logs_the_wait_that_failed(
    events: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    """A job end that leaves its libraries to the finalizer says so at
    DEBUG, with the devices it waited for and CUDA's error."""
    fn, _modules = _candidate(events, launched_on=(2, 1))
    with (
        caplog.at_level(logging.DEBUG, logger=compile_cache.log.name),
        _device(0, events, synchronizes=False),
    ):
        _unload_compiled_fn(fn)
    assert events == ["synchronize 1"]
    (record,) = [
        record for record in caplog.records if record.name == compile_cache.log.name
    ]
    assert record.levelno == logging.DEBUG
    assert "on (1, 2) did not synchronize" in record.getMessage()
    assert record.exc_info is not None
    assert "illegal memory access" in str(record.exc_info[1])


def test_unload_libraries_raises_what_is_no_cuda_error(events: list[str]) -> None:
    """Only CUDA's error of the wait, a ``RuntimeError``, is a device that
    does not synchronize.  Another exception is raised: nothing is
    unloaded, the capture gate is free again, and the launcher keeps its
    functions, which a later release unloads."""
    fn, _modules = _candidate(events, launched_on=(1,))

    def bug() -> None:
        raise TypeError("not a CUDA error")

    with (
        _device(0, events, during_synchronize=bug),
        pytest.raises(TypeError, match="not a CUDA error"),
    ):
        _unload_compiled_fn(fn)
    assert events == []
    assert not launch_caches._CAPTURE_GATE.locked()

    with _device(0, events):
        _unload_compiled_fn(fn)
    assert events == ["synchronize 1", "unload reloaded", "unload twin"]


def test_dropping_a_kernel_unloads_nothing_in_the_process_that_compiled_it(
    events: list[str],
) -> None:
    """Which modules the collector finalizes at all.  ``PyCodeCache.load``
    registers the module a bound kernel compiles in ``sys.modules``, where it
    stays when the kernel is dropped, and the module's device kernels hold
    their compiled launchers: the process that compiled a kernel unloads
    none of its libraries, whatever it captures.  The modules that die are
    the temporary ones a worker job loads and releases
    (``_load_compiled_fn``, ``_unload_compiled_fn``)."""

    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def dropped(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.size(0)):
            out[tile] = x[tile] + 1
        return out

    with (
        mock_cuda_unavailable(),
        patch(
            "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = cpu_bind(dropped, (torch.empty(4096),))
        fn = bound.compile_config(
            bound.config_spec.normalized_config(
                helion.Config.from_dict({"block_sizes": [256], "num_threads": [256]})
            )
        )
    generated = fn.__module__
    device_kernel = f"_helion_{fn.__name__}"
    modules = _give_launcher(fn.__globals__[device_kernel], events)

    del dropped, bound, fn
    gc.collect()

    assert events == []
    assert all(module() is not None for module in modules)
    kernel = getattr(sys.modules[generated], device_kernel)
    # The stand-in libraries go with the test.
    del kernel._helion_cute_compiled_launchers
    launch_caches.drop_relaunch_state(kernel)


@pytest.mark.parametrize("capture", _CAPTURES)
def test_finalizer_makes_no_driver_call_while_a_capture_may_be_underway(
    events: list[str],
    capture: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    before = _drop(_launched(events, "before"))
    gc.collect()
    assert before() is None
    assert events == ["unload before"]

    monkeypatch.setattr(compile_cache, "_SKIPPED_CAPTURE_UNLOADS", 0, raising=False)
    first = _drop(_launched(events, "first"))
    second = _drop(_launched(events, "second"))
    unloaded = _launched(events, "unloaded")
    unloaded.jit_module.unload()
    unloaded = _drop(unloaded)
    with (
        caplog.at_level(logging.WARNING, logger=compile_cache.log.name),
        _CAPTURES[capture](),
    ):
        gc.collect()
        assert first() is None and second() is None and unloaded() is None
        assert events == ["unload before", "unload unloaded"]
    # Two modules kept their libraries, counted, with one warning; a module
    # unloaded before it died is not one of them.
    assert compile_cache.skipped_capture_unloads() == 2
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == compile_cache.log.name
    ]
    assert len(warnings) == 1 and "stays loaded" in warnings[0]
    # ... and the warning says what unloads such libraries.
    assert "gc.collect() outside a capture" in warnings[0]

    after = _drop(_launched(events, "after"))
    with _stream_capturing(False):
        gc.collect()
    assert after() is None
    assert events == ["unload before", "unload unloaded", "unload after"]
    assert compile_cache.skipped_capture_unloads() == 2


def test_collection_at_capture_entry_unloads_ahead_of_the_count(
    events: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """torch collects on capture entry when asked to
    (``TORCH_CUDAGRAPH_GC``), with the capture already counted.
    ``cute_cuda_graph`` collects first under that option, ahead of the
    count: a dead module is unloaded there, as outside any capture, not left
    loaded as one the collector reaches inside the capture is."""
    monkeypatch.setattr(compile_cache, "_SKIPPED_CAPTURE_UNLOADS", 0, raising=False)
    monkeypatch.setattr(torch.compiler.config, "force_cudagraph_gc", True)
    dead = _drop(_launched(events, "dead"))

    with _torch_capture(events):
        assert dead() is None
        assert events == ["unload dead", "synchronize", "capture_begin"]
        assert compile_cache.skipped_capture_unloads() == 0
        assert launch_caches.capture_may_be_active()

    assert events == ["unload dead", "synchronize", "capture_begin", "capture_end"]
    assert not launch_caches.capture_may_be_active()


def test_module_finalized_inside_torchs_entry_keeps_its_library(
    events: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The capture is counted from ahead of torch's entry, where it begins.
    What the collector finalizes inside that entry makes no driver call,
    though this thread's capture has not begun: another thread's may have.
    (Without the option ``cute_cuda_graph`` collects nothing first; the
    stand-in's entry collects as torch's did before the option existed.)"""
    monkeypatch.setattr(compile_cache, "_SKIPPED_CAPTURE_UNLOADS", 0, raising=False)
    monkeypatch.setattr(torch.compiler.config, "force_cudagraph_gc", False)
    dead = _drop(_launched(events, "dead"))

    with _torch_capture(events):
        assert dead() is None
        assert events == ["synchronize", "capture_begin"]
        assert compile_cache.skipped_capture_unloads() == 1

    assert events == ["synchronize", "capture_begin", "capture_end"]
    assert not launch_caches.capture_may_be_active()


@pytest.mark.parametrize("fails", ["begin", "body", "end"])
def test_capture_count_ends_with_a_capture_that_raises(
    events: list[str], fails: str
) -> None:
    """A capture whose entry, body or end raises is counted no longer, the
    one that failed to begin included, and the capture gate is free: later
    finalizers unload again."""
    error = ZeroDivisionError if fails == "body" else RuntimeError
    with pytest.raises(error), _torch_capture(events, fails):
        assert launch_caches.capture_may_be_active()
        if fails == "body":
            raise ZeroDivisionError
    assert not launch_caches.capture_may_be_active()
    assert not launch_caches._CAPTURE_GATE.locked()
    # torch's exit ran for a capture that began, with the count still up.
    assert events == ["synchronize"] + (
        [] if fails == "begin" else ["capture_begin", "capture_end"]
    )

    dropped = _drop(_launched(events, "dropped"))
    gc.collect()
    assert dropped() is None
    assert events[-1] == "unload dropped"


class _InterruptedCount:
    """``_capture_underway`` when an interrupt arrives as it is entered."""

    def __enter__(self) -> None:
        raise KeyboardInterrupt

    def __exit__(self, *exc_info: object) -> None:
        raise AssertionError("a context that was not entered is not exited")


def test_no_capture_begins_when_an_exception_arrives_at_the_count(
    events: list[str],
) -> None:
    """The count is taken ahead of torch's entry: an exception that arrives
    there (an interrupt) leaves no capture begun and no graph marked as the
    one being captured."""
    with (
        patch.object(launch_caches, "_capture_underway", _InterruptedCount),
        pytest.raises(KeyboardInterrupt),
        _torch_capture(events),
    ):
        pytest.fail("the body ran")
    assert events == []
    assert not launch_caches.capture_may_be_active()
    assert launch_caches.ACTIVE_CUDA_GRAPH.get() is None


def test_capture_that_starts_during_an_unload_begins_behind_it(
    events: list[str],
) -> None:
    """``unload_libraries`` asks whether a capture is underway, then waits
    for its devices and unloads.  A ``cute_cuda_graph`` capture that another
    thread starts in between (here inside the wait) begins behind the
    unloads: the question and the driver calls hold the capture gate, which
    the capture's count waits for.  Begun inside, the capture would be
    invalidated by the calls that follow."""
    fn, _modules = _candidate(events, launched_on=(1,))

    with capture_stand_ins(events) as (gate, graph):
        capture = CaptureOnAnotherThread(gate, graph)
        try:
            with _device(
                0, events, during_synchronize=capture.start_inside_a_driver_call
            ):
                _unload_compiled_fn(fn)
        finally:
            capture.finish()

    assert events == [
        "synchronize 1",
        "unload reloaded",
        "unload twin",
        "capture_begin",
        "capture_end",
    ]


def test_capture_that_starts_inside_an_unload_begins_behind_the_last_one(
    events: list[str],
) -> None:
    """The unloads hold the capture gate as the wait ahead of them does: a
    capture that another thread starts inside the first library's unload
    begins behind the second's.  Unloaded behind the gate, the second
    library's unload would land in the capture."""
    with capture_stand_ins(events) as (gate, graph):
        capture = CaptureOnAnotherThread(gate, graph)
        fn, _modules = _candidate(
            events,
            launched_on=(1,),
            during_unload=capture.start_inside_a_driver_call,
        )
        try:
            with _device(0, events):
                _unload_compiled_fn(fn)
        finally:
            capture.finish()

    assert events == [
        "synchronize 1",
        "unload reloaded",
        "unload twin",
        "capture_begin",
        "capture_end",
    ]


class _CaptureStartedByTheQuestion:
    """``torch.cuda.is_current_stream_capturing`` as a thread asks it that
    wants to know whether a capture is underway: the first time, another
    thread starts a ``cute_cuda_graph`` capture inside the question, which
    then answers that the stream is not capturing.  A question asked ahead
    of the capture gate is stale by the time the gate is taken; asked under
    it, the capture waits for the gate and begins behind the driver calls
    the answer allowed."""

    def __init__(self, capture: CaptureOnAnotherThread) -> None:
        self.capture = capture
        self.asked = 0

    def __call__(self) -> bool:
        self.asked += 1
        if self.asked == 1:
            self.capture.start_inside_a_driver_call()
        return False


def test_capture_that_starts_inside_the_unloads_question_begins_behind_it(
    events: list[str],
) -> None:
    """``unload_libraries`` asks under the capture gate: a capture that
    another thread starts while the question is being answered begins
    behind the wait and the unloads."""
    fn, _modules = _candidate(events, launched_on=(1,))

    with capture_stand_ins(events) as (gate, graph):
        capture = CaptureOnAnotherThread(gate, graph)
        question = _CaptureStartedByTheQuestion(capture)
        try:
            with (
                _device(0, events),
                patch.object(torch.cuda, "is_current_stream_capturing", question),
            ):
                _unload_compiled_fn(fn)
        finally:
            capture.finish()

    assert question.asked == 1
    assert events == [
        "synchronize 1",
        "unload reloaded",
        "unload twin",
        "capture_begin",
        "capture_end",
    ]


def test_capture_that_starts_inside_a_finalizers_question_begins_behind_it(
    events: list[str],
) -> None:
    """The finalizer asks under the capture gate as well: a capture that
    another thread starts while its question is being answered begins once
    the library is unloaded."""
    with capture_stand_ins(events) as (gate, graph):
        capture = CaptureOnAnotherThread(gate, graph)
        question = _CaptureStartedByTheQuestion(capture)
        dead = _drop(_launched(events, "dead"))
        try:
            with (
                _stream_capturing(False),
                patch.object(torch.cuda, "is_current_stream_capturing", question),
            ):
                gc.collect()
        finally:
            capture.finish()

    assert dead() is None and question.asked == 1
    assert events == ["unload dead", "capture_begin", "capture_end"]


def test_abandoned_capture_finalized_on_the_gates_holder_does_not_wait(
    events: list[str],
) -> None:
    """A capture that was entered and never left (its owner dropped in a
    reference cycle) is ended by the collector, which closes the context's
    generator wherever it runs: also on a thread that holds the capture
    gate for driver calls of its own.  The end of a capture lowers the
    count without the gate, so that thread goes on, and the count is back
    at zero."""
    with capture_stand_ins(events) as (gate, _graph):
        stack = contextlib.ExitStack()
        stack.enter_context(launch_caches.cute_cuda_graph())
        cycle: list[object] = [stack]
        cycle.append(cycle)
        del stack, cycle
        assert events == ["capture_begin"]
        assert launch_caches.capture_may_be_active()
        with launch_caches.outside_captures(wait=True) as safe:
            # The abandoned capture is still counted: no driver call here.
            assert not safe
            # The collector, on the thread that holds the gate.
            gc.collect()
            assert events == ["capture_begin", "capture_end"]
            assert not launch_caches.capture_may_be_active()
        assert not gate.deadlocks
        # The gate is free and the next block may call the driver.
        with launch_caches.outside_captures(wait=False) as safe:
            assert safe
    assert launch_caches._CAPTURES_UNDERWAY == 0


class _FinalizingLock:
    """A lock of the kind of the count's own
    (``launch_caches._CAPTURE_COUNT_LOCK``), a new one, whose next entry
    drops what ``abandoned`` holds right behind the acquisition: an entered
    capture's last reference, which finalizes the capture there, inside the
    lock's region, as the collector can at any allocation."""

    def __init__(self, events: list[str]) -> None:
        self._lock = type(launch_caches._CAPTURE_COUNT_LOCK)()
        self.abandoned: list[object] = []
        self.events = events

    def __enter__(self) -> bool:
        self._lock.acquire()
        if self.abandoned:
            self.events.append("finalization inside the count's region")
            self.abandoned.clear()
        return True

    def __exit__(self, *exc_info: object) -> None:
        self._lock.release()


def test_abandoned_capture_finalized_inside_the_counts_region_does_not_wait(
    events: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An abandoned capture can also be finalized on a thread that is
    inside the count lock's own region: here on the thread that has just
    taken the lock to count a second capture.  The end of the abandoned
    capture takes the lock again on that thread, so the lock is re-entrant:
    the thread goes on, counts its capture (one underway inside it) and
    leaves none behind."""
    # Restored when the test ends, also where the thread below never does.
    monkeypatch.setattr(launch_caches, "_CAPTURES_UNDERWAY", 0)
    lock = _FinalizingLock(events)
    underway: list[int] = []
    errors: list[BaseException] = []

    def second_capture() -> None:
        try:
            abandoned = launch_caches.cute_cuda_graph()
            abandoned.__enter__()
            underway.append(launch_caches._CAPTURES_UNDERWAY)
            lock.abandoned.append(abandoned)
            del abandoned
            with launch_caches.cute_cuda_graph():
                underway.append(launch_caches._CAPTURES_UNDERWAY)
            underway.append(launch_caches._CAPTURES_UNDERWAY)
        except BaseException as error:
            errors.append(error)

    with (
        capture_stand_ins(events) as (gate, _graph),
        patch.object(launch_caches, "_CAPTURE_COUNT_LOCK", lock),
    ):
        thread = threading.Thread(target=second_capture, daemon=True)
        thread.start()
        # A lock that is not re-entrant leaves the thread waiting for
        # itself: the test fails here instead of hanging.
        thread.join(STEP_TIMEOUT_S)
        assert not thread.is_alive(), "the thread waits for the count lock it holds"
        assert not errors, errors
        assert not gate.deadlocks
    assert underway == [1, 1, 0]
    assert events == [
        "capture_begin",
        "finalization inside the count's region",
        "capture_end",
        "capture_begin",
        "capture_end",
    ]


def test_unload_waits_for_another_threads_driver_calls(events: list[str]) -> None:
    """The job-end unload is its owner's one chance: where another thread
    holds the capture gate for driver calls of its own (a spare's fill, a
    finalizer's unload), ``unload_libraries`` waits for it and then
    unloads, instead of leaving the libraries to a finalizer."""
    fn, _modules = _candidate(events, launched_on=(1,))
    holding = threading.Event()
    errors: list[BaseException] = []

    with capture_stand_ins(events) as (gate, _graph):

        def other_thread() -> None:
            try:
                with launch_caches.outside_captures(wait=True) as safe:
                    assert safe
                    holding.set()
                    # Until the unload starts to wait for the gate.
                    assert gate.waited.wait(STEP_TIMEOUT_S)
                    events.append("the other thread's driver call")
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=other_thread, daemon=True)
        thread.start()
        try:
            assert holding.wait(STEP_TIMEOUT_S)
            with _device(0, events):
                _unload_compiled_fn(fn)
        finally:
            gate.waited.set()
            thread.join(STEP_TIMEOUT_S)
        assert not thread.is_alive() and not errors and not gate.deadlocks

    assert events == [
        "the other thread's driver call",
        "synchronize 1",
        "unload reloaded",
        "unload twin",
    ]


def test_capture_that_starts_during_a_finalizers_unload_begins_behind_it(
    events: list[str],
) -> None:
    """The finalizer's question and its unload hold the capture gate too: a
    capture that another thread starts inside the driver call begins once
    the call has returned."""
    with capture_stand_ins(events) as (gate, graph):
        capture = CaptureOnAnotherThread(gate, graph)
        dead = _drop(_launched(events, "dead", capture.start_inside_a_driver_call))
        try:
            gc.collect()
        finally:
            capture.finish()

    assert dead() is None
    assert events == ["unload dead", "capture_begin", "capture_end"]


def test_finalizer_on_the_thread_that_counts_its_capture_makes_no_driver_call(
    events: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The collector runs a finalizer wherever an allocation crosses its
    threshold, also on the thread that holds the capture gate for its
    capture's count.  The finalizer does not wait for the gate (the thread
    would wait for itself) and makes no driver call there, before the count
    has risen: the module keeps its library, as one finalized inside the
    capture does."""
    monkeypatch.setattr(compile_cache, "_SKIPPED_CAPTURE_UNLOADS", 0, raising=False)
    dead = _drop(_launched(events, "dead"))
    collections: list[int] = []

    def collect() -> None:
        # The collector, as the gate is taken for the count.
        if not collections:
            assert not launch_caches.capture_may_be_active()
            collections.append(gc.collect())

    with capture_stand_ins(events, taken=collect) as (gate, _graph):
        with launch_caches.cute_cuda_graph():
            assert dead() is None
            assert events == ["capture_begin"]
        assert not gate.deadlocks

    assert collections
    assert events == ["capture_begin", "capture_end"]
    assert compile_cache.skipped_capture_unloads() == 1


def test_finalizer_does_not_wait_for_another_threads_capture_count(
    events: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finalizer that runs while another thread holds the capture gate
    for its capture's count leaves the library loaded and returns: it
    neither waits for that thread nor calls the driver while that capture
    may be beginning."""
    monkeypatch.setattr(compile_cache, "_SKIPPED_CAPTURE_UNLOADS", 0, raising=False)
    dead = _drop(_launched(events, "dead"))
    holding, released = threading.Event(), threading.Event()
    gates: list[object] = []

    def hold() -> None:
        # The other thread, with the gate taken for its count: until the
        # test lets it go on, or a thread starts to wait for the gate.
        if not holding.is_set():
            holding.set()
            (gate,) = gates
            deadline = STEP_TIMEOUT_S
            while not (released.is_set() or gate.waited.is_set()) and deadline > 0:
                released.wait(0.001)
                deadline -= 0.001

    with capture_stand_ins(events, taken=hold) as (gate, graph):
        gates.append(gate)
        capture = CaptureOnAnotherThread(gate, graph)
        capture.start()
        try:
            assert holding.wait(STEP_TIMEOUT_S)
            gc.collect()
            assert dead() is None
            # No driver call, no wait, and the capture has not begun.
            assert events == []
            assert not gate.waited.is_set()
            assert compile_cache.skipped_capture_unloads() == 1
        finally:
            released.set()
            capture.finish()

    assert events == ["capture_begin", "capture_end"]


def test_finalizer_still_makes_no_driver_call_at_interpreter_shutdown(
    events: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(compile_cache, "_SKIPPED_CAPTURE_UNLOADS", 0, raising=False)
    module = cuda_jit_executor.CudaDialectJitModule(
        None, SimpleNamespace(restype=None), None, [_Library("library", events)]
    )
    # Not garbage of a cycle: the finalizer runs at the ``del``, and no
    # collection runs other objects' finalizers under the patch.
    with patch.object(sys, "is_finalizing", lambda: True):
        del module
    assert events == []
    assert compile_cache.skipped_capture_unloads() == 0
