"""CPU coverage for pointer alignment and CuTe launch cache compatibility."""

from __future__ import annotations

import ctypes
import dataclasses
import inspect
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


def _compiled(
    executor: _Executor, by_reference: bool
) -> launcher._CompiledCuteLauncher:
    compiled = launcher._CompiledCuteLauncher(object(), None)
    compiled._compiled = SimpleNamespace(
        execution_args=_ExecutionArgs(by_reference), _default_executor=executor
    )
    return compiled


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
