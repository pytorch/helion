"""CPU coverage for pointer alignment and CuTe launch cache compatibility."""

from __future__ import annotations

import ctypes
import dataclasses
import inspect
import threading
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import MagicMock
import weakref

import pytest
import torch

from helion.runtime.cute import launcher


@dataclasses.dataclass
class _Pointer:
    address: int
    alignment: int
    cell: ctypes.c_uint64 = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        assert self.address % self.alignment == 0
        self.cell = ctypes.c_uint64(self.address)


@dataclasses.dataclass
class _Stream:
    value: int


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> list[_Pointer]:
    pointers: list[_Pointer] = []

    def make_ptr(
        dtype: object, address: int, space: object, *, assumed_align: int
    ) -> _Pointer:
        pointer = _Pointer(address, assumed_align)
        pointers.append(pointer)
        return pointer

    monkeypatch.setattr(
        launcher, "_get_cute_launcher_imports", lambda: (None, make_ptr, None)
    )
    monkeypatch.setattr(launcher, "_validate_cute_launcher_tensor", lambda tensor: None)
    monkeypatch.setattr(launcher, "_torch_dtype_to_cutlass", str)
    monkeypatch.setattr(launcher, "_cute_kernel_param_is_constexpr", lambda kernel: ())
    return pointers


@pytest.mark.parametrize("bake", [False, True])
@pytest.mark.parametrize(
    "dtype,offset,alignment",
    [
        (torch.uint8, 1, 1),
        (torch.float16, 0, 16),
        (torch.float16, 1, 2),
        (torch.float16, 2, 4),
        (torch.float16, 4, 8),
        (torch.float16, 8, 16),
        (torch.float32, 1, 4),
        (torch.float64, 1, 8),
    ],
)
def test_actual_view_pointer_alignment_enters_schema(
    fake_runtime: list[_Pointer],
    bake: bool,
    dtype: torch.dtype,
    offset: int,
    alignment: int,
) -> None:
    tensor = torch.empty(64, dtype=dtype)[offset : offset + 16]
    built = launcher._build_cute_schema_and_args(
        SimpleNamespace(), (tensor,), (1, 1, 1), bake_tensor_shapes=bake
    )
    pointer = cast("_Pointer", built.launch_args[0])
    assert pointer.address == tensor.data_ptr()
    assert pointer.alignment == alignment
    legacy: tuple[object, ...] = ("tensor", str(dtype), 1)
    if bake:
        legacy = (*legacy, (16,), (1,))
    assert built.schema == ((legacy if alignment == 16 else (*legacy, alignment)),)
    assert launcher._cute_schema_pointer_alignment(built.schema[0]) == alignment


def test_empty_tensor_zero_pointer_remains_valid(fake_runtime: list[_Pointer]) -> None:
    tensor = torch.empty(0, dtype=torch.float16)
    built = launcher._build_cute_schema_and_args(
        SimpleNamespace(), (tensor,), (1, 1, 1)
    )
    pointer = cast("_Pointer", built.launch_args[0])
    assert pointer.address == 0
    assert pointer.alignment == 16
    assert built.schema == (("tensor", "torch.float16", 1),)


def test_real_cute_constructor_accepts_a_naturally_aligned_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass")
    monkeypatch.setattr(launcher, "_validate_cute_launcher_tensor", lambda tensor: None)
    monkeypatch.setattr(launcher, "_cute_kernel_param_is_constexpr", lambda kernel: ())
    tensor = torch.empty(33, dtype=torch.float16)[1:]
    built = launcher._build_cute_schema_and_args(
        SimpleNamespace(), (tensor,), (1, 1, 1)
    )
    pointer = cast("Any", built.launch_args[0])
    assert pointer.__cache_key__[-1] == 2
    assert pointer.__tvm_ffi_opaque_ptr__() == tensor.data_ptr()


@pytest.mark.parametrize("bake", [False, True])
def test_pointer_and_compiled_caches_distinguish_alignment(
    fake_runtime: list[_Pointer], monkeypatch: pytest.MonkeyPatch, bake: bool
) -> None:
    kernel = SimpleNamespace(
        _helion_cute_disable_bake_tensor_shapes=not bake,
        _helion_cute_source_hash="alignment-test",
    )
    storage = torch.empty(64, dtype=torch.float16)
    aligned = storage[:16]
    unaligned = storage[1:17]
    fresh_aligned = torch.empty_like(aligned)
    first = launcher._build_cached_cute_schema_and_args(kernel, (aligned,), (1, 1, 1))
    same = launcher._build_cached_cute_schema_and_args(
        kernel, (aligned.view_as(aligned),), (1, 1, 1)
    )
    changed = launcher._build_cached_cute_schema_and_args(
        kernel, (unaligned,), (1, 1, 1)
    )
    fresh = launcher._build_cached_cute_schema_and_args(
        kernel, (fresh_aligned,), (1, 1, 1)
    )
    assert first is same
    assert first is not changed
    assert first.schema == fresh.schema
    assert first.schema != changed.schema
    monkeypatch.setattr(
        launcher, "_create_cute_wrapper", lambda *args, **kwargs: object()
    )
    compiled = launcher._get_compiled_cute_launcher(kernel, first.schema, (32, 1, 1))
    assert compiled is launcher._get_compiled_cute_launcher(
        kernel, fresh.schema, (32, 1, 1)
    )
    other = launcher._get_compiled_cute_launcher(kernel, changed.schema, (32, 1, 1))
    assert compiled is not other
    assert isinstance(compiled, launcher._CompiledCuteLauncher)
    assert isinstance(other, launcher._CompiledCuteLauncher)
    assert compiled._cache_key is not None
    assert compiled._cache_key != other._cache_key
    guard = launcher._cute_last_launch_arg_guard(kernel, (aligned,), (1, 1, 1))
    assert guard.matches(kernel, (aligned.view_as(aligned),), (1, 1, 1))
    assert not guard.matches(kernel, (unaligned,), (1, 1, 1))


@pytest.mark.parametrize("runtime_extent", [False, True])
def test_wrapper_side_tensor_alignment_is_specialized(
    fake_runtime: list[_Pointer], monkeypatch: pytest.MonkeyPatch, runtime_extent: bool
) -> None:
    layout = torch.empty(2, dtype=torch.int32)
    workspace = torch.empty(21, dtype=torch.int32)[1:].view(2, 10)
    result = launcher._Tcgen05GroupedStaticMetadataResult(
        problem_sizes=None if runtime_extent else workspace,
        starts=None if runtime_extent else layout,
        total_clusters=2,
        runtime_tile_records=workspace if runtime_extent else None,
    )
    entry = launcher._Tcgen05GroupedStaticMetadataCacheEntry(
        layout_ref=weakref.ref(layout),
        n_sizes_ref=None,
        k_sizes_ref=None,
        has_m_tail=False,
        has_n_tail=False,
        result=result,
    )
    plan = {
        "kind": "tcgen05_grouped_static_persistent",
        "problem_sizes_arg": "workspace",
        "starts_arg": "starts",
        "runtime_tile_records_arg": "workspace",
        "total_clusters_arg": "total_clusters",
    }
    kernel = SimpleNamespace(_helion_cute_wrapper_plans=[plan])
    monkeypatch.setattr(
        launcher, "_tcgen05_grouped_static_layout_arg", lambda *args: layout
    )
    monkeypatch.setattr(
        launcher, "_build_tcgen05_grouped_static_metadata", lambda *args: entry
    )
    built = launcher._build_cute_schema_and_args(kernel, (layout,), (1, 1, 1))
    pointer = next(
        pointer for pointer in fake_runtime if pointer.address == workspace.data_ptr()
    )
    assert pointer.alignment == 4
    workspace_schema = next(
        schema for schema in built.schema if schema[1] == "workspace"
    )
    kind = (
        "wrapper_tensor_runtime_leading_extent" if runtime_extent else "wrapper_tensor"
    )
    assert workspace_schema == (
        kind,
        "workspace",
        "torch.int32",
        2,
        (10,) if runtime_extent else (2, 10),
        (10, 1),
        4,
    )


@pytest.mark.parametrize(
    "schema,expected",
    [
        (("tensor", "torch.float16", 1, 2), "arg0_shape0: cutlass.Int64"),
        (("tensor", "torch.float16", 1, (16,), (1,), 2), "arg0_shape0 = 16"),
        (
            ("wrapper_tensor", "workspace", "torch.int32", 1, (16,), (1,), 4),
            "cute.make_layout((16,), stride=(1,))",
        ),
        (
            (
                "wrapper_tensor_runtime_leading_extent",
                "workspace",
                "torch.int32",
                2,
                (10,),
                (10, 1),
                4,
            ),
            "workspace_shape0: cutlass.Int64",
        ),
    ],
)
def test_wrapper_codegen_accepts_alignment_specialized_schemas(
    schema: tuple[object, ...], expected: str
) -> None:
    pytest.importorskip("cutlass")
    wrapper = launcher._create_cute_wrapper(SimpleNamespace(), (schema,), (32, 1, 1))
    assert expected in inspect.getsource(cast("Any", wrapper))


def _cuda_tensor(pointer: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    tensor = MagicMock(spec=torch.Tensor)
    tensor.device = torch.device("cuda", 0)
    tensor.dtype = dtype
    tensor.ndim = 1
    tensor.data_ptr.return_value = pointer
    tensor.shape = torch.Size((16,))
    tensor.size.side_effect = lambda dim=None: (16,) if dim is None else 16
    tensor.stride.side_effect = lambda dim=None: (1,) if dim is None else 1
    return cast("torch.Tensor", tensor)


class _ExecutionArgs:
    def __init__(self, by_reference: bool) -> None:
        self.by_reference = by_reference

    def generate_execution_args(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> tuple[list[object], list[object]]:
        result: list[object] = []
        for arg in args:
            if isinstance(arg, _Pointer):
                result.append(
                    ctypes.addressof(arg.cell) if self.by_reference else arg.address
                )
            elif isinstance(arg, _Stream):
                result.append(arg.value)
            else:
                result.append(arg)
        return result, list(args)


class _Executor:
    def __init__(self, by_reference: bool, pointer_index: int = 0) -> None:
        self.by_reference = by_reference
        self.pointer_index = pointer_index
        self.calls: list[tuple[int, int]] = []

    def run_compiled_program(self, args: list[object]) -> str:
        first = cast("int", args[self.pointer_index])
        pointer = (
            ctypes.c_uint64.from_address(first).value if self.by_reference else first
        )
        self.calls.append((pointer, cast("int", args[-1])))
        return "launched"


def _compiled(executor: object, by_reference: bool) -> launcher._CompiledCuteLauncher:
    compiled = launcher._CompiledCuteLauncher(object(), None)
    compiled._compiled = SimpleNamespace(
        execution_args=_ExecutionArgs(by_reference), _default_executor=executor
    )
    return compiled


def _capi_executor_class() -> type:
    """A C-API ``JitExecutor`` whose entry point records its packed arguments.

    ``JitExecutor.__init__`` needs a compiled module; the fields that
    ``_get_invoke_packed_args`` reads are set directly instead.  The recorded
    stream is the last marshalled argument, followed by the cuda-result slot.
    """
    jit_executor = pytest.importorskip("cutlass.base_dsl.jit_executor")

    class _CapiExecutor(jit_executor.JitExecutor):
        def __init__(
            self, by_reference: bool, pointer_index: int = 0, error_code: int = 0
        ) -> None:
            self.by_reference = by_reference
            self.pointer_index = pointer_index
            self.error_code = error_code
            self.cuda_result = ctypes.c_int32(0)
            self._has_cuda_result = True
            self._cuda_result_addr = ctypes.addressof(self.cuda_result)
            self._kernel_ptrs = None
            self._num_extra_args = 1
            self._tls = threading.local()
            self.calls: list[tuple[int, int]] = []
            self.run_compiled_program_calls = 0

        def capi_func(self, packed: ctypes.Array[ctypes.c_void_p]) -> None:
            first = cast("int", packed[self.pointer_index])
            pointer = (
                ctypes.c_uint64.from_address(first).value
                if self.by_reference
                else first
            )
            self.calls.append((pointer, cast("int", packed[len(packed) - 2])))
            result_address = cast("int", packed[len(packed) - 1])
            ctypes.c_int32.from_address(result_address).value = self.error_code

        def run_compiled_program(self, exe_args: list[object]) -> int | None:
            self.run_compiled_program_calls += 1
            return 0

    return _CapiExecutor


def _require_probe_visible_cells(pointers: list[_Pointer]) -> None:
    """The fast relaunch probe only dereferences by-reference cells above the
    launcher's host address floor, which fits the Linux user-space layout the
    CUDA-only fast path runs under. macOS arm64 allocates every host cell
    below it, so by-reference detection cannot be exercised there."""
    if any(
        ctypes.addressof(pointer.cell) <= launcher._CUTE_HOST_CELL_ADDRESS_FLOOR
        for pointer in pointers
    ):
        pytest.skip("host ctypes cells sit below the fast relaunch address floor")


def _mock_cuda_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = launcher.importlib.import_module
    monkeypatch.setattr(
        launcher.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(CUstream=_Stream)
            if name == "cuda.bindings.driver"
            else original_import(name)
        ),
    )
    # CPU-only torch builds (Metal, XPU, TPU, Pallas interpret) do not define
    # the CUDA stream accessor; create it for the mocked launch either way.
    monkeypatch.setattr(
        torch._C,
        "_cuda_getCurrentRawStream",
        lambda device: 0xD00,
        raising=False,
    )


@pytest.mark.parametrize("by_reference", [False, True])
@pytest.mark.parametrize(
    "initial,replacement,hit",
    [
        (0x1000, 0x2000, True),
        (0x1000, 0x2002, False),
        (0x1002, 0x2006, True),
        (0x1002, 0x2000, True),
    ],
)
def test_fast_relaunch_respects_compiled_alignment_before_patching(
    fake_runtime: list[_Pointer],
    monkeypatch: pytest.MonkeyPatch,
    by_reference: bool,
    initial: int,
    replacement: int,
    hit: bool,
) -> None:
    _mock_cuda_driver(monkeypatch)
    kernel = SimpleNamespace()
    original = _cuda_tensor(initial)
    launch = launcher._build_cute_schema_and_args(kernel, (original,), (1, 1, 1))
    executor = _Executor(by_reference)
    fastpath = launcher._cute_build_fast_relaunch(
        kernel,
        (original,),
        (1, 1, 1),
        (32, 1, 1),
        None,
        launch,
        _compiled(executor, by_reference),
    )
    if by_reference:
        _require_probe_visible_cells(fake_runtime)
    assert fastpath is not None
    expected_alignment = 16 if initial == 0x1000 else 2
    assert all(pointer.alignment == expected_alignment for pointer in fake_runtime)
    result = fastpath.try_launch(
        (_cuda_tensor(replacement),), (1, 1, 1), (32, 1, 1), None
    )
    assert result == ((True, "launched") if hit else (False, None))
    assert executor.calls == ([(replacement, 0xD00)] if hit else [])
    if not hit:
        assert fastpath.try_launch((original,), (1, 1, 1), (32, 1, 1), None) == (
            True,
            "launched",
        )
        assert executor.calls == [(initial, 0xD00)]


@pytest.mark.parametrize("by_reference", [False, True])
def test_fast_relaunch_enters_capi_executor_with_owned_packed_args(
    fake_runtime: list[_Pointer], monkeypatch: pytest.MonkeyPatch, by_reference: bool
) -> None:
    executor = _capi_executor_class()(by_reference, pointer_index=1)
    _mock_cuda_driver(monkeypatch)
    kernel = SimpleNamespace()
    args = (3, _cuda_tensor(0x1000), _cuda_tensor(0x5000))
    launch = launcher._build_cute_schema_and_args(kernel, args, (1, 1, 1))
    fastpath = launcher._cute_build_fast_relaunch(
        kernel,
        args,
        (1, 1, 1),
        (32, 1, 1),
        None,
        launch,
        _compiled(executor, by_reference),
    )
    if by_reference:
        _require_probe_visible_cells(fake_runtime)
    assert fastpath is not None
    assert fastpath.packed_args is not None
    # The owned copy ends with the executor's cuda-result slot and holds the
    # by-value stream handle in the slot the probe found.
    assert len(fastpath.packed_args) == len(fastpath.exe_args) + 1
    assert fastpath.packed_args[len(fastpath.exe_args)] == executor._cuda_result_addr
    assert executor.calls == []

    assert fastpath.try_launch(
        (3, _cuda_tensor(0x2000), _cuda_tensor(0x6000)), (1, 1, 1), (32, 1, 1), None
    ) == (True, 0)
    assert executor.calls == [(0x2000, 0xD00)]
    # A second call with another pointer and stream is written before launch.
    monkeypatch.setattr(torch._C, "_cuda_getCurrentRawStream", lambda device: 0xE00)
    assert fastpath.try_launch(
        (3, _cuda_tensor(0x3000), _cuda_tensor(0x6000)), (1, 1, 1), (32, 1, 1), None
    ) == (True, 0)
    assert executor.calls == [(0x2000, 0xD00), (0x3000, 0xE00)]
    assert executor.run_compiled_program_calls == 0
    # Metadata misses still never touch the executor.
    assert fastpath.try_launch(
        (3, _cuda_tensor(0x3002), _cuda_tensor(0x6000)), (1, 1, 1), (32, 1, 1), None
    ) == (False, None)
    assert len(executor.calls) == 2


def test_fast_relaunch_capi_executor_raises_cuda_errors(
    fake_runtime: list[_Pointer], monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = _capi_executor_class()(False, error_code=1)
    common = pytest.importorskip("cutlass.base_dsl.common")
    _mock_cuda_driver(monkeypatch)
    kernel = SimpleNamespace()
    original = _cuda_tensor(0x1000)
    launch = launcher._build_cute_schema_and_args(kernel, (original,), (1, 1, 1))
    fastpath = launcher._cute_build_fast_relaunch(
        kernel,
        (original,),
        (1, 1, 1),
        (32, 1, 1),
        None,
        launch,
        _compiled(executor, False),
    )
    assert fastpath is not None
    with pytest.raises(common.DSLCudaRuntimeError, match="CUDA_ERROR_INVALID_VALUE"):
        fastpath.try_launch((_cuda_tensor(0x2000),), (1, 1, 1), (32, 1, 1), None)
    assert executor.calls == [(0x2000, 0xD00)]


@pytest.mark.parametrize("mode", ("eligible", "excluded-plan", "rejit-fails"))
def test_fastpath_builder_rehosts_fresh_tvm_ffi_compiles_only_when_eligible(
    fake_runtime: list[_Pointer], monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    tvm_ffi_provider = pytest.importorskip("cutlass.cutlass_dsl.tvm_ffi_provider")
    _mock_cuda_driver(monkeypatch)
    # What a fresh ``cute.compile`` under --enable-tvm-ffi returns: no
    # JitExecutor, so the fastpath cannot be probed against it directly.
    fresh = object.__new__(tvm_ffi_provider.TVMFFIJitCompiledFunctionBase)
    fresh.ir_module = object()
    fresh.function_name = "fresh_entry"
    fresh.has_gpu_module = True
    executor = _Executor(False)
    rehosted = SimpleNamespace(
        execution_args=_ExecutionArgs(False),
        _default_executor=None,
        to=lambda device=None: executor,
    )
    rejit_calls: list[tuple[object, str, bool]] = []

    def rejit(
        self: object, module: object, function_name: str, *, has_gpu_module: bool
    ) -> object:
        rejit_calls.append((module, function_name, has_gpu_module))
        if mode == "rejit-fails":
            raise RuntimeError("no host JIT here")
        return rehosted

    monkeypatch.setattr(
        launcher._CompiledCuteLauncher, "_jit_cuda_dialect_function", rejit
    )
    kernel = SimpleNamespace(
        _helion_cute_wrapper_plans=(
            [{"kind": "gathered_mma_tma"}] if mode == "excluded-plan" else []
        )
    )
    original = _cuda_tensor(0x1000)
    launch = launcher._build_cute_schema_and_args(kernel, (original,), (1, 1, 1))
    compiled = launcher._CompiledCuteLauncher(object(), None)
    compiled._compiled = fresh
    fastpath = launcher._cute_build_fast_relaunch(
        kernel, (original,), (1, 1, 1), (32, 1, 1), None, launch, compiled
    )
    if mode == "excluded-plan":
        # Kernels outside the fastpath keep the tvm-ffi launch and never pay
        # for the re-JIT.
        assert fastpath is None and rejit_calls == []
        assert compiled._compiled is fresh and compiled._capi_compiled is None
        return
    assert rejit_calls == [(fresh.ir_module, "fresh_entry", True)]
    if mode == "rejit-fails":
        assert fastpath is None and compiled._compiled is fresh
        assert compiled._capi_compiled is None
        return
    assert fastpath is not None
    # The C-API twin serves the fast relaunch; the tvm-ffi function keeps
    # serving the full path and the twin is memoised for later builds.
    assert compiled._compiled is fresh
    assert compiled._capi_compiled is rehosted
    assert compiled.capi_function() is rehosted and len(rejit_calls) == 1
    assert fastpath.executor is executor
    assert fastpath.try_launch(
        (_cuda_tensor(0x2000),), (1, 1, 1), (32, 1, 1), None
    ) == (True, "launched")
    assert executor.calls == [(0x2000, 0xD00)]


def test_fast_relaunch_wraps_capi_entry_failures_like_run_compiled_program(
    fake_runtime: list[_Pointer], monkeypatch: pytest.MonkeyPatch
) -> None:
    common = pytest.importorskip("cutlass.base_dsl.common")
    executor = _capi_executor_class()(False)

    def crash(packed: object) -> None:
        raise OSError("entry point faulted")

    executor.capi_func = crash
    _mock_cuda_driver(monkeypatch)
    kernel = SimpleNamespace()
    original = _cuda_tensor(0x1000)
    launch = launcher._build_cute_schema_and_args(kernel, (original,), (1, 1, 1))
    fastpath = launcher._cute_build_fast_relaunch(
        kernel,
        (original,),
        (1, 1, 1),
        (32, 1, 1),
        None,
        launch,
        _compiled(executor, False),
    )
    assert fastpath is not None and fastpath.packed_args is not None
    with pytest.raises(common.DSLRuntimeError, match="Runtime Crash") as info:
        fastpath.try_launch((_cuda_tensor(0x2000),), (1, 1, 1), (32, 1, 1), None)
    assert isinstance(info.value.cause, OSError)


def test_direct_launch_setup_failure_keeps_run_compiled_program(
    fake_runtime: list[_Pointer], monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = _capi_executor_class()(False)
    # A packing-internals change in cutlass must not disable the whole
    # fastpath; only the direct C entry is given up.
    monkeypatch.setattr(
        type(executor),
        "_get_invoke_packed_args",
        lambda self, exe_args: (_ for _ in ()).throw(AttributeError("renamed")),
    )
    _mock_cuda_driver(monkeypatch)
    kernel = SimpleNamespace()
    original = _cuda_tensor(0x1000)
    launch = launcher._build_cute_schema_and_args(kernel, (original,), (1, 1, 1))
    fastpath = launcher._cute_build_fast_relaunch(
        kernel,
        (original,),
        (1, 1, 1),
        (32, 1, 1),
        None,
        launch,
        _compiled(executor, False),
    )
    assert fastpath is not None and fastpath.packed_args is None
    assert fastpath.try_launch(
        (_cuda_tensor(0x2000),), (1, 1, 1), (32, 1, 1), None
    ) == (True, 0)
    assert executor.run_compiled_program_calls == 1 and executor.calls == []


def test_fastpath_builder_refuses_a_pointer_weaker_than_compiled_schema(
    fake_runtime: list[_Pointer], monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_cuda_driver(monkeypatch)
    kernel = SimpleNamespace()
    launch = launcher._build_cute_schema_and_args(
        kernel, (_cuda_tensor(0x1000),), (1, 1, 1)
    )
    executor = _Executor(False)
    fastpath = launcher._cute_build_fast_relaunch(
        kernel,
        (_cuda_tensor(0x1002),),
        (1, 1, 1),
        (32, 1, 1),
        None,
        launch,
        _compiled(executor, False),
    )
    assert fastpath is None
    assert executor.calls == []
    assert len(fake_runtime) == 1


@pytest.mark.parametrize("by_reference", [False, True])
def test_fastpath_tracks_each_pointer_alignment_among_scalar_arguments(
    fake_runtime: list[_Pointer], monkeypatch: pytest.MonkeyPatch, by_reference: bool
) -> None:
    _mock_cuda_driver(monkeypatch)
    kernel = SimpleNamespace()
    args = (3, _cuda_tensor(0x1002), 7, _cuda_tensor(0x5000))
    launch = launcher._build_cute_schema_and_args(kernel, args, (1, 1, 1))
    executor = _Executor(by_reference, pointer_index=1)
    fastpath = launcher._cute_build_fast_relaunch(
        kernel,
        args,
        (1, 1, 1),
        (32, 1, 1),
        None,
        launch,
        _compiled(executor, by_reference),
    )
    if by_reference:
        _require_probe_visible_cells(fake_runtime)
    assert fastpath is not None
    assert [pointer.alignment for pointer in fake_runtime] == [2, 16, 2, 16, 2, 16]
    assert fastpath.try_launch(
        (3, _cuda_tensor(0x2002), 7, _cuda_tensor(0x6002)),
        (1, 1, 1),
        (32, 1, 1),
        None,
    ) == (False, None)
    assert executor.calls == []
    assert fastpath.try_launch(
        (3, _cuda_tensor(0x2002), 7, _cuda_tensor(0x6000)),
        (1, 1, 1),
        (32, 1, 1),
        None,
    ) == (True, "launched")
    assert executor.calls == [(0x2002, 0xD00)]


@pytest.mark.parametrize("by_reference", [False, True])
def test_fast_relaunch_raw_launch_compares_the_host_call_as_spelled(
    fake_runtime: list[_Pointer], monkeypatch: pytest.MonkeyPatch, by_reference: bool
) -> None:
    # The entry hands the relaunch the grid tuple and keyword arguments as
    # the generated host code spells them; anything else (another grid,
    # block, an option, a missing record) goes through the normalizing path.
    _mock_cuda_driver(monkeypatch)
    kernel = SimpleNamespace()
    original = _cuda_tensor(0x1000)
    launch = launcher._build_cute_schema_and_args(kernel, (original,), (1, 1, 1))
    executor = _Executor(by_reference)
    fastpath = launcher._cute_build_fast_relaunch(
        kernel,
        (original,),
        (1, 1, 1),
        (32, 1, 1),
        None,
        launch,
        _compiled(executor, by_reference),
    )
    if by_reference:
        _require_probe_visible_cells(fake_runtime)
    assert fastpath is not None
    assert fastpath.cxx_guards is None  # mocks are not torch.Tensor objects
    miss = (False, None)
    tensor = _cuda_tensor(0x2000)
    assert fastpath.try_launch_raw((tensor,), (1,), {"block": (32, 1, 1)}) == miss
    fastpath.raw_grid = (1,)
    fastpath.raw_kwargs = {"block": (32, 1, 1)}
    assert fastpath.try_launch_raw((tensor,), (1,), {"block": (32, 1, 1)}) == (
        True,
        "launched",
    )
    assert executor.calls == [(0x2000, 0xD00)]
    assert fastpath.try_launch_raw((tensor,), (2,), {"block": (32, 1, 1)}) == miss
    assert fastpath.try_launch_raw((tensor,), (1,), {"block": (64, 1, 1)}) == miss
    assert (
        fastpath.try_launch_raw(
            (tensor,), (1,), {"block": (32, 1, 1), "cute_compile_options": "--x"}
        )
        == miss
    )
    assert fastpath.try_launch_raw((tensor, 3), (1,), {"block": (32, 1, 1)}) == miss
    assert executor.calls == [(0x2000, 0xD00)]


def test_fast_relaunch_scalar_guards_match_type_and_value(
    fake_runtime: list[_Pointer], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Scalars compare as (type, value) first; a value the exact compare
    # cannot settle (NaN) falls back to the normalizing guard.
    _mock_cuda_driver(monkeypatch)
    kernel = SimpleNamespace()
    args = (3, _cuda_tensor(0x1000), 0.5, float("nan"))
    launch = launcher._build_cute_schema_and_args(kernel, args, (1, 1, 1))
    executor = _Executor(False, pointer_index=1)
    fastpath = launcher._cute_build_fast_relaunch(
        kernel, args, (1, 1, 1), (32, 1, 1), None, launch, _compiled(executor, False)
    )
    assert fastpath is not None
    assert [(index, kind) for index, kind, _value in fastpath.scalar_fast] == [
        (0, int),
        (2, float),
        (3, float),
    ]

    def relaunch(*scalars: object) -> tuple[bool, object]:
        a, b, c = scalars
        return fastpath.try_launch(
            (a, _cuda_tensor(0x2000), b, c), (1, 1, 1), (32, 1, 1), None
        )

    assert relaunch(3, 0.5, float("nan")) == (True, "launched")
    assert relaunch(4, 0.5, float("nan")) == (False, None)
    assert relaunch(3.0, 0.5, float("nan")) == (False, None)
    assert relaunch(True, 0.5, float("nan")) == (False, None)
    assert relaunch(3, 0.25, float("nan")) == (False, None)
    assert relaunch(3, 0.5, 1.0) == (False, None)
    # NaNs compare equal through the normalizing guard (as they always did).
    assert relaunch(3, 0.5, -float("nan")) == (True, "launched")
    assert executor.calls == [(0x2000, 0xD00), (0x2000, 0xD00)]


@pytest.mark.parametrize("pinned,other", [(0.0, -0.0), (-0.0, 0.0)])
def test_fast_relaunch_distinguishes_the_sign_of_a_zero_scalar(
    fake_runtime: list[_Pointer],
    monkeypatch: pytest.MonkeyPatch,
    pinned: float,
    other: float,
) -> None:
    # -0.0 == 0.0, but the float travels by value and is pinned in the
    # relaunch's argument array -- a division by it would produce infinities of
    # the wrong sign.  The normalizing guard's hex compare distinguished them;
    # so does the exact compare now.
    _mock_cuda_driver(monkeypatch)
    kernel = SimpleNamespace()
    args = (pinned, _cuda_tensor(0x1000))
    launch = launcher._build_cute_schema_and_args(kernel, args, (1, 1, 1))
    executor = _Executor(False, pointer_index=1)
    fastpath = launcher._cute_build_fast_relaunch(
        kernel, args, (1, 1, 1), (32, 1, 1), None, launch, _compiled(executor, False)
    )
    assert fastpath is not None
    assert fastpath.try_launch(
        (other, _cuda_tensor(0x2000)), (1, 1, 1), (32, 1, 1), None
    ) == (False, None)
    assert fastpath.try_launch(
        (pinned, _cuda_tensor(0x2000)), (1, 1, 1), (32, 1, 1), None
    ) == (True, "launched")
    assert executor.calls == [(0x2000, 0xD00)]


def test_the_raw_launch_is_recorded_only_for_the_fastpaths_own_launch(
    fake_runtime: list[_Pointer], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The entry recorded the raw (grid, kwargs) of whichever call first found
    # none, so a concurrent first launch with another grid could leave its
    # spelling on a fastpath pinned to a different grid, and the raw relaunch
    # then launched the pinned grid.
    _mock_cuda_driver(monkeypatch)
    kernel = SimpleNamespace()
    original = _cuda_tensor(0x1000)
    launch = launcher._build_cute_schema_and_args(kernel, (original,), (2, 1, 1))
    executor = _Executor(False)
    fastpath = launcher._cute_build_fast_relaunch(
        kernel,
        (original,),
        (2, 1, 1),
        (32, 1, 1),
        None,
        launch,
        _compiled(executor, False),
    )
    assert fastpath is not None and fastpath.raw_grid is None
    kernel._helion_cute_fastpath = fastpath
    block = {"block": (32, 1, 1)}
    launcher._cute_record_raw_launch(kernel, (1,), block, (1, 1, 1), (32, 1, 1), None)
    assert fastpath.raw_grid is None and fastpath.raw_kwargs is None
    launcher._cute_record_raw_launch(kernel, (2,), block, (2, 1, 1), (64, 1, 1), None)
    assert fastpath.raw_grid is None
    launcher._cute_record_raw_launch(kernel, (2,), block, (2, 1, 1), (32, 1, 1), "--x")
    assert fastpath.raw_grid is None
    launcher._cute_record_raw_launch(kernel, (2,), block, (2, 1, 1), (32, 1, 1), None)
    assert fastpath.raw_grid == (2,) and fastpath.raw_kwargs == block
    # The first record stays.
    launcher._cute_record_raw_launch(kernel, (2, 1), block, (2, 1, 1), (32, 1, 1), None)
    assert fastpath.raw_grid == (2,)
    assert fastpath.try_launch_raw((_cuda_tensor(0x2000),), (1,), block) == (
        False,
        None,
    )
    assert fastpath.try_launch_raw((_cuda_tensor(0x2000),), (2,), block) == (
        True,
        "launched",
    )


def test_fast_relaunch_cxx_guard_accelerates_but_python_guards_decide(
    fake_runtime: list[_Pointer], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A passing C++ check (type, dtype, device, requires_grad, dispatch keys,
    # sizes, strides) stands in for the Python metadata guards; a failing
    # one (e.g. requires_grad differs) hands the decision to them.
    _mock_cuda_driver(monkeypatch)
    kernel = SimpleNamespace()
    original = _cuda_tensor(0x1000)
    launch = launcher._build_cute_schema_and_args(kernel, (original,), (1, 1, 1))
    executor = _Executor(False)
    fastpath = launcher._cute_build_fast_relaunch(
        kernel,
        (original,),
        (1, 1, 1),
        (32, 1, 1),
        None,
        launch,
        _compiled(executor, False),
    )
    assert fastpath is not None
    seen: list[tuple[object, ...]] = []

    def check(*tensors: object) -> bool:
        seen.append(tensors)
        return False

    fastpath.cxx_check = check
    other_dtype = _cuda_tensor(0x2000, dtype=torch.float32)
    assert fastpath.try_launch((other_dtype,), (1, 1, 1), (32, 1, 1), None) == (
        False,
        None,
    )
    same = _cuda_tensor(0x2000)
    assert fastpath.try_launch((same,), (1, 1, 1), (32, 1, 1), None) == (
        True,
        "launched",
    )
    assert seen == [(other_dtype,), (same,)]
    fastpath.cxx_check = lambda *tensors: True
    assert fastpath.try_launch((other_dtype,), (1, 1, 1), (32, 1, 1), None) == (
        True,
        "launched",
    )
    # The alignment of the compiled schema is checked on the gathered
    # pointers whichever guard decided.
    assert fastpath.try_launch(
        (_cuda_tensor(0x2002),), (1, 1, 1), (32, 1, 1), None
    ) == (False, None)
    assert executor.calls == [(0x2000, 0xD00), (0x2000, 0xD00)]


def test_cxx_tensor_guards_pin_dtype_shape_stride_and_device() -> None:
    guards_module = pytest.importorskip("torch._C._dynamo.guards")
    assert hasattr(guards_module, "TensorGuards")
    a = torch.empty((4, 8), dtype=torch.float16)
    b = torch.empty((8,), dtype=torch.float32)
    tensor_guards = (
        (0, "cpu", None, a.dtype, (4, 8), (8, 1), 16),
        (2, "cpu", None, b.dtype, (8,), (1,), 16),
    )
    cxx = launcher._cute_cxx_tensor_guards((a, 1, b), tensor_guards)
    assert cxx is not None
    check = cast("Any", cxx).check
    assert check(a, b)
    assert check(torch.empty_like(a), torch.empty_like(b))
    assert not check(a[:, :4], b)
    assert not check(a.t().contiguous().t(), b)
    assert not check(a.to(torch.bfloat16), b)
    assert not check(a, b.requires_grad_(True))
    assert not check(a, 3)
    # Mocks (no torch.Tensor type) leave the Python guards alone.
    assert (
        launcher._cute_cxx_tensor_guards(
            (_cuda_tensor(0x1000),), ((0, "cuda", 0, torch.float16, (16,), (1,), 16),)
        )
        is None
    )


def test_standalone_alignment_guard_uses_the_classifiers_32_byte_residue() -> None:
    pytest.importorskip("cutlass.cute")
    from helion._compiler.cute.memory_ops import _CUTE_ALIGNMENT_MAX_BYTES
    from helion.runtime.cute import standalone_launcher as standalone

    # The exported residues come from the compiler's classifier: the two
    # moduli must agree or an export accepts pointers of another class.
    assert standalone._STANDALONE_ALIGNMENT_BYTES == _CUTE_ALIGNMENT_MAX_BYTES == 32
    backing = torch.empty(64, dtype=torch.float32)
    residue = backing.data_ptr() % 32
    shifted16 = backing[4:]  # +16 bytes: another 32-byte class
    shifted32 = backing[8:]  # +32 bytes: the same class
    standalone._standalone_validate_alignment((backing, shifted32), (residue, residue))
    with pytest.raises(ValueError, match="alignment"):
        standalone._standalone_validate_alignment((shifted16,), (residue,))
    # A workload exported for a residue-16 input accepts that input ...
    standalone._standalone_validate_alignment((shifted16,), ((residue + 16) % 32,))
    # ... and not the residue-0 one (a 16-byte check conflated the two).
    with pytest.raises(ValueError, match="alignment"):
        standalone._standalone_validate_alignment((backing,), ((residue + 16) % 32,))
