from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

from helion import exc
from helion.runtime.cute import chained_rectangular_leaf as leaf
from helion.runtime.cute import launcher
from helion.runtime.cute.tma_tile import append_tma_tile

_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@pytest.fixture(autouse=True)
def _ordinary_cuda_tensors_without_initialization():
    initialized = torch.cuda.is_initialized()
    with (
        patch.object(
            torch.Tensor, "device", property(lambda self: torch.device("cuda", 0))
        ),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
    ):
        yield
    assert torch.cuda.is_initialized() == initialized


def _plan(dtype=torch.bfloat16) -> dict[str, Any]:
    return {
        "kind": leaf.KIND,
        "source_idx": 0,
        "write_indices": [2, 3],
        "shape": (2, 32, 64),
        "strides": (2048, 64, 1),
        "dtype": str(dtype).removeprefix("torch."),
        "rows": 64,
        "columns": 64,
        "tile": (16, 32),
        "kernel_args": ["rect_atom", "rect_tensor"],
    }


def _args(dtype=torch.bfloat16) -> tuple[object, ...]:
    return (
        torch.empty((2, 32, 64), dtype=dtype),
        torch.empty(64, dtype=torch.float32),
        torch.empty((64, 8), dtype=dtype),
        torch.empty((64, 64), dtype=torch.float32),
        3,
    )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_current_fresh_and_aligned_offset_sources(dtype):
    args = list(_args(dtype))
    original = args[0]
    assert isinstance(original, torch.Tensor)
    offset = 16 // original.element_size()
    storage = torch.empty(original.numel() + offset, dtype=dtype)
    sources = (
        original,
        original.clone(),
        storage[offset:].view(original.shape),
    )
    assert len({source.data_ptr() for source in sources}) == 3
    for source in sources:
        args[0] = source
        leaf.validate_plan(_plan(dtype), tuple(args))


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("kind", ("shape", "stride", "dtype", "unaligned"))
def test_current_source_metadata_and_pointer_must_match(dtype, kind):
    args = list(_args(dtype))
    if kind == "shape":
        args[0] = torch.empty((32, 2, 64), dtype=dtype)
    elif kind == "stride":
        args[0] = torch.empty((2, 32, 128), dtype=dtype)[..., ::2]
    elif kind == "dtype":
        args[0] = torch.empty((2, 32, 64), dtype=torch.float64)
    else:
        args[0] = torch.empty(4097, dtype=dtype)[1:].view(2, 32, 64)
    with pytest.raises(exc.BackendUnsupported):
        leaf.validate_plan(_plan(dtype), tuple(args))


@pytest.mark.parametrize("position", (0, 1, 2, 3))
def test_every_current_tensor_reachable_span_is_checked(position):
    args = _args()
    value = args[position]
    assert isinstance(value, torch.Tensor)
    value.untyped_storage().resize_(1)
    with pytest.raises(exc.BackendUnsupported):
        leaf.validate_plan(_plan(), args)


@pytest.mark.parametrize("position", (0, 1, 2, 3))
def test_every_current_tensor_must_be_ordinary(position):
    args = list(_args())
    args[position] = torch.nn.Parameter(torch.empty((2, 32, 64)))
    with pytest.raises(exc.BackendUnsupported):
        leaf.validate_plan(_plan(), tuple(args))


@pytest.mark.parametrize("position", (0, 1, 2, 3))
def test_every_current_tensor_must_be_on_source_cuda_device(position):
    args = _args()
    foreign = args[position]
    with (
        patch.object(
            torch.Tensor,
            "device",
            property(lambda self: torch.device("cuda", 1 if self is foreign else 0)),
        ),
        pytest.raises(exc.BackendUnsupported),
    ):
        leaf.validate_plan(_plan(), args)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("write_index", (2, 3))
@pytest.mark.parametrize("read_index", (0, 1))
def test_each_write_excludes_all_read_storage_even_cross_dtype(
    dtype, write_index, read_index
):
    args = list(_args(dtype))
    read = args[read_index]
    assert isinstance(read, torch.Tensor)
    args[write_index] = read.view(torch.uint8)
    with pytest.raises(exc.BackendUnsupported, match="alias"):
        leaf.validate_plan(_plan(dtype), tuple(args))


def test_second_write_cannot_alias_first_write():
    args = list(_args())
    args[3] = args[2]
    with pytest.raises(exc.BackendUnsupported, match="alias"):
        leaf.validate_plan(_plan(), tuple(args))


def test_disjoint_logical_views_still_share_excluded_storage():
    args = list(_args())
    storage = torch.empty(8192, dtype=torch.bfloat16)
    args[0] = storage[:4096].view(2, 32, 64)
    args[3] = storage[4096:]
    with pytest.raises(exc.BackendUnsupported, match="alias"):
        leaf.validate_plan(_plan(), tuple(args))


def test_read_read_alias_is_allowed_and_noncontiguous_outputs_are_valid():
    args = list(_args())
    source = args[0]
    assert isinstance(source, torch.Tensor)
    args[1] = source.view(torch.uint8)
    args[2] = torch.empty((64, 16), dtype=torch.bfloat16)[:, ::2]
    args[3] = torch.empty((64, 64)).T
    leaf.validate_plan(_plan(), tuple(args))


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("empty_index", (1, 2, 3))
def test_empty_nondescriptor_reads_and_writes_are_valid(dtype, empty_index):
    args = list(_args(dtype))
    args[empty_index] = torch.empty((0, 64), dtype=dtype)
    leaf.validate_plan(_plan(dtype), tuple(args))


def test_empty_write_with_nonempty_aliasing_storage_still_rejects():
    args = list(_args())
    source = args[0]
    assert isinstance(source, torch.Tensor)
    args[3] = source[:0]
    with pytest.raises(exc.BackendUnsupported, match="alias"):
        leaf.validate_plan(_plan(), tuple(args))


@pytest.mark.parametrize(
    "field,value",
    (
        ("source_idx", -1),
        ("source_idx", 99),
        ("source_idx", True),
        ("source_idx", 4),
        ("write_indices", []),
        ("write_indices", [2, 2]),
        ("write_indices", [0, 2]),
        ("write_indices", [-1, 2]),
        ("write_indices", [2, 99]),
        ("write_indices", [2, True]),
        ("write_indices", [2, 4]),
    ),
)
def test_index_schema_is_consistent(field, value):
    plan = _plan()
    plan[field] = value
    with pytest.raises(exc.BackendUnsupported):
        leaf.validate_plan(plan, _args())


@pytest.mark.parametrize("dtype", _DTYPES)
def test_compact_descriptor_view_matches_storage_element_count(dtype):
    plan = _plan(dtype)
    plan["rows"] += 1
    with pytest.raises(exc.BackendUnsupported):
        leaf.validate_plan(plan, _args(dtype))


@pytest.mark.parametrize("field", ("rows", "columns"))
@pytest.mark.parametrize("value", (0, -1, True, 64.0, "64", None))
def test_descriptor_dimensions_require_positive_strict_integers(field, value):
    plan = _plan()
    plan[field] = value
    with pytest.raises(exc.BackendUnsupported):
        leaf.validate_plan(plan, _args())


def test_exact_metadata_does_not_license_holey_descriptor_source():
    args = list(_args())
    source = torch.empty((2, 32, 128), dtype=torch.bfloat16)[..., ::2]
    args[0] = source
    plan = _plan()
    plan["strides"] = source.stride()
    with pytest.raises(exc.BackendUnsupported):
        leaf.validate_plan(plan, tuple(args))


@pytest.mark.parametrize("route", ("direct", "cached", "last_launch"))
def test_current_guards_precede_every_launcher_cache_and_schema(route):
    plan = _plan()
    kernel = SimpleNamespace(_helion_cute_wrapper_plans=[plan])
    args = list(_args())
    leaf.validate_plan(plan, tuple(args))
    args[3] = args[1]
    with (
        patch.object(
            launcher,
            "_get_cute_launcher_imports",
            side_effect=AssertionError("builder"),
        ),
        patch.object(
            launcher,
            "_cute_dynamic_tensormap_contexts",
            side_effect=AssertionError("cache"),
        ),
        patch.object(
            launcher,
            "_cute_last_launch_cache_entry",
            side_effect=AssertionError("cache"),
        ),
        pytest.raises(exc.BackendUnsupported, match="alias"),
    ):
        if route == "direct":
            launcher._build_cute_schema_and_args(kernel, tuple(args), (1, 1, 1))
        elif route == "cached":
            launcher._build_cached_cute_schema_and_args(kernel, tuple(args), (1, 1, 1))
        else:
            launcher.default_cute_launcher(kernel, (1,), *args, block=(128, 1, 1))


def test_descriptor_kind_is_excluded_from_pointer_patch_fast_relaunch():
    compiled = launcher._CompiledCuteLauncher(object(), None)
    compiled._compiled = SimpleNamespace(
        execution_args=SimpleNamespace(generate_execution_args=lambda: None),
        _default_executor=SimpleNamespace(run_compiled_program=lambda: None),
    )
    launch = cast("Any", SimpleNamespace(owned_tensors=(), grouped_static_metadata=()))
    kernel = SimpleNamespace(_helion_cute_wrapper_plans=[_plan()])
    with (
        patch.object(launcher, "_tcgen05_grouped_static_plans", return_value=()),
        patch.object(launcher, "_cute_dynamic_tensormap_contexts", return_value=()),
        patch.object(
            launcher,
            "_get_cute_launcher_imports",
            side_effect=AssertionError("fastpath"),
        ),
    ):
        assert (
            launcher._cute_build_fast_relaunch(
                kernel, _args(), (1, 1, 1), (128, 1, 1), None, launch, compiled
            )
            is None
        )


@pytest.mark.parametrize("position", (1, 2, 3))
@pytest.mark.parametrize("view", ("sparse", "conjugate", "negative"))
def test_nondescriptor_arguments_keep_ordinary_strided_value_semantics(position, view):
    args = list(_args())
    if view == "sparse":
        args[position] = torch.empty(64).to_sparse()
    elif view == "conjugate":
        args[position] = torch.empty(64, dtype=torch.complex64).conj()
    else:
        args[position] = torch._neg_view(torch.empty(64))
    with pytest.raises(exc.BackendUnsupported):
        leaf.validate_plan(_plan(), tuple(args))


def test_source_requires_cuda_even_when_all_tensors_agree():
    with (
        patch.object(
            torch.Tensor, "device", property(lambda self: torch.device("cpu"))
        ),
        pytest.raises(exc.BackendUnsupported),
    ):
        leaf.validate_plan(_plan(), _args())


@pytest.mark.parametrize("dtype", _DTYPES)
def test_aligned_source_and_unaligned_valid_output_offsets(dtype):
    args = list(_args(dtype))
    args[2] = torch.empty(513, dtype=dtype)[1:].view(64, 8)
    args[3] = torch.empty(4097, dtype=torch.float32)[1:].view(64, 64)
    leaf.validate_plan(_plan(dtype), tuple(args))


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("tile", ((16, 16), (32, 32), (64, 64), (32, 128)))
def test_wrapper_exactly_delegates_typed_native_builder(dtype, tile):
    plan = _plan(dtype)
    plan["tile"] = tile
    expected, expected_args = ["    previous = 1"], ["previous_arg"]
    append_tma_tile(
        expected,
        expected_args,
        source_index=0,
        atom="rect_atom",
        tensor="rect_tensor",
        rows=64,
        columns=64,
        tile=tile,
        dtype=dtype,
    )
    body, args = ["    previous = 1"], ["previous_arg"]
    with patch.object(leaf, "append_tma_tile", wraps=append_tma_tile) as shared:
        leaf.append_wrapper(body, args, plan)
    shared.assert_called_once()
    assert body == expected and args == expected_args
