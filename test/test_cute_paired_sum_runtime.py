from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

from test._cute_binding import _mock_cuda_unavailable

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from test._cute_paired_sum_model import reference_tree
from test._cute_paired_sum_model import run_literal

from helion.runtime.cute import paired_sum
from helion.runtime.cute.paired_sum import plan_for_pair
from helion.runtime.cute.paired_sum import sum_plan
from helion.runtime.cute.paired_sum import try_paired_sum_cast

if TYPE_CHECKING:
    from collections.abc import Iterator

    from test._cute_paired_sum_model import Value


_requires_pinned_aten_build = pytest.mark.skipif(
    torch.version.git_version != paired_sum._ATEN_SUM_GIT,
    reason="host sum ATen reuse requires the pinned PyTorch build",
)


@pytest.fixture(scope="module", autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield


@pytest.mark.parametrize("layout", ("mapped", "narrow"))
@pytest.mark.parametrize(
    "rows,columns,stride",
    [
        (64, 1024, 1024),
        (0, 129, 129),
        (1, 1, 1),
        (1, 17, 17),
        (13, 127, 127),
        (65, 130, 130),
        (65, 128, 132),
        (257, 132, 132),
        (64, 0, 1),
    ],
)
def test_literal_device_ast_has_exact_tree_and_unique_ownership(
    layout: str, rows: int, columns: int, stride: int
) -> None:
    plan = sum_plan(rows, columns, stride, 16, layout)
    assert plan is not None
    result = run_literal(plan, stride=stride)
    assert result["loads"] == 2 * rows * columns
    assert result["stores"] == 2 * columns


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_barrier",
        "ascending_tree",
        "missing_row_guard",
        "missing_column_guard",
        "cast_partial",
        "reuse_left_output",
        "wrong_source",
    ],
)
def test_literal_tree_proof_rejects_incorrect_device_mutations(mutation: str) -> None:
    rows, columns = (65, 132) if "guard" in mutation else (64, 128)
    plan = sum_plan(rows, columns, columns, 16, "narrow")
    assert plan is not None
    with pytest.raises((AssertionError, KeyError)):
        run_literal(plan, mutation=mutation)


@pytest.mark.parametrize("layout", ("mapped", "narrow"))
def test_byte_limit_and_cross_cta_fallback(layout: str) -> None:
    assert sum_plan(2, 128, 536870784, 16, layout) is not None
    assert sum_plan(2, 128, 536870785, 16, layout) is None
    assert sum_plan(64, 128, 2**24, 16, layout) is None
    assert sum_plan(32, 128, 2**24, 16, layout) is not None
    assert sum_plan(1, 2**29, 2**29, 16, layout) is not None
    assert sum_plan(1, 2**29 + 1, 2**29 + 1, 16, layout) is None
    assert sum_plan(65536, 128, 128, 16, layout) is None
    # Even a singleton's unused row stride must fit the emitted Int32 cast.
    assert sum_plan(1, 128, 2**40, 16, layout) is None


def test_byte_split_changes_addition_tree_before_bfloat16_cast() -> None:
    def evaluate(value: Value, row_base: int = 0) -> np.float32:
        if value.op == "positive_zero":
            return np.float32(0)
        if value.op == "input":
            return np.float32({0: 1e8, 4: -1e8, 16: 1}.get(value.args[1] + row_base, 0))
        assert value.op == "add_rn_f32"
        return np.float32(
            evaluate(value.args[0], row_base) + evaluate(value.args[1], row_base)
        )

    # The installed iterator divides [64,128], stride=(2**24,1), into two
    # [32,128] reductions. The first split's result is accumulated into output.
    unsplit = evaluate(reference_tree(64, 16, 0, 0))
    first = evaluate(reference_tree(32, 1, 0, 0))
    second = evaluate(reference_tree(32, 1, 0, 0), 32)
    split = np.float32(first + second)
    assert float(unsplit) == 0
    assert float(split) == 1
    assert torch.tensor([unsplit, split]).to(torch.bfloat16).tolist() == [0, 1]


def _forbidden(*args: object, **kwargs: object) -> None:
    raise AssertionError("unsupported binding must execute original host operations")


def _try(left: torch.Tensor, right: torch.Tensor) -> object:
    return try_paired_sum_cast(
        left,
        right,
        torch.bfloat16,
        dtype_is_tensor=False,
        layout="narrow",
        _launcher=_forbidden,
    )


def test_cpu_fallback_and_autograd_remain_original_operations() -> None:
    left = torch.randn(17, 20, requires_grad=True)
    right = torch.randn(17, 20, requires_grad=True)
    assert _try(left, right) is None
    outputs = left.sum(0).to(torch.bfloat16), right.sum(0).to(torch.bfloat16)
    sum(output.float().sum() for output in outputs).backward()
    assert torch.equal(left.grad, torch.ones_like(left))
    assert torch.equal(right.grad, torch.ones_like(right))


def test_forward_ad_negative_view_and_subclass_decline() -> None:
    tensor = torch.ones(3, 8)
    assert not paired_sum._ordinary_tensor(torch._neg_view(tensor))
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(tensor, torch.ones_like(tensor))
        assert not paired_sum._ordinary_tensor(dual)

    class Subclass(torch.Tensor):
        pass

    assert not paired_sum._ordinary_tensor(tensor.as_subclass(Subclass))
    assert not paired_sum._ordinary_tensor(torch._to_functional_tensor(tensor))


def test_unknown_dispatch_modes_decline_before_any_tensor_operation() -> None:
    tensor = torch.ones(3, 8)

    class FunctionMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            raise AssertionError("guard must not invoke custom torch functions")

    class DispatchMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            raise AssertionError("guard must not invoke custom dispatch")

    with FunctionMode():
        assert _try(tensor, tensor) is None
    with DispatchMode():
        assert _try(tensor, tensor) is None


def test_dtype_alignment_and_read_alias_proof() -> None:
    tensor = torch.empty(64, 128)
    plan = plan_for_pair(tensor, tensor, torch.bfloat16, "narrow")
    assert plan is not None  # Read aliases do not create an output dependency.
    shifted = torch.empty(64 * 128 + 1)[1:].view(64, 128)
    assert plan_for_pair(tensor, shifted, torch.bfloat16, "narrow") is None
    assert plan_for_pair(tensor, tensor[:, ::2], torch.bfloat16, "narrow") is None
    assert plan_for_pair(tensor, tensor, torch.float16, "narrow") is None
    assert (
        plan_for_pair(tensor.requires_grad_(), tensor, torch.bfloat16, "narrow") is None
    )


@_requires_pinned_aten_build
def test_unknown_aten_build_headers_and_missing_source_fall_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paired_sum._compatible_aten_sum.cache_clear()
    try:
        assert paired_sum._compatible_aten_sum()
        monkeypatch.setattr(torch.version, "git_version", "unknown library build")
        paired_sum._compatible_aten_sum.cache_clear()
        assert not paired_sum._compatible_aten_sum()
        monkeypatch.setattr(torch.version, "git_version", paired_sum._ATEN_SUM_GIT)
        monkeypatch.setattr(torch, "__file__", str(tmp_path / "__init__.py"))
        paired_sum._compatible_aten_sum.cache_clear()
        assert not paired_sum._compatible_aten_sum()
        for relative in paired_sum._ATEN_SUM_HEADERS:
            filename = tmp_path / "include" / relative
            filename.parent.mkdir(parents=True, exist_ok=True)
            filename.write_text("changed reduction tree")
        paired_sum._compatible_aten_sum.cache_clear()
        assert not paired_sum._compatible_aten_sum()
    finally:
        paired_sum._compatible_aten_sum.cache_clear()


@_requires_pinned_aten_build
def test_aten_sources_are_not_read_on_repeated_calls() -> None:
    paired_sum._compatible_aten_sum.cache_clear()
    try:
        with patch.object(
            Path, "read_bytes", autospec=True, side_effect=Path.read_bytes
        ) as read:
            assert paired_sum._compatible_aten_sum()
            assert paired_sum._compatible_aten_sum()
            assert read.call_count == len(paired_sum._ATEN_SUM_HEADERS)
    finally:
        paired_sum._compatible_aten_sum.cache_clear()


@pytest.mark.parametrize("columns", (0, 20))
def test_host_helper_allocates_fresh_outputs_and_reuses_the_supplied_launcher(
    columns: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    left, right = torch.ones(3, columns), torch.ones(3, columns)
    calls = []
    # Real Tensor objects/storage/methods, with only the Python device metadata
    # replaced. The injected launcher records ABI; it never executes CUDA.
    monkeypatch.setattr(
        torch.Tensor, "device", property(lambda self: torch.device("cuda"))
    )
    monkeypatch.setattr(paired_sum, "_compatible_aten_sum", lambda: True)

    def launch(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    outputs = [
        try_paired_sum_cast(
            left,
            right,
            torch.bfloat16,
            dtype_is_tensor=False,
            layout="narrow",
            _launcher=launch,
        )
        for _ in range(2)
    ]
    assert all(output is not None for output in outputs)
    tensors = [tensor for pair in outputs for tensor in pair]
    assert len({id(tensor) for tensor in tensors}) == 4
    assert all(
        tensor.shape == (columns,) and tensor.dtype == torch.bfloat16
        for tensor in tensors
    )
    assert all(not tensor.requires_grad for tensor in tensors)
    if columns:
        assert len({tensor.data_ptr() for tensor in tensors}) == 4
        assert len(calls) == 2
        assert calls[0][0][0] is calls[1][0][0]
        assert calls[0][0][2] is left and calls[0][0][3] is right
        assert (
            calls[0][1]["block"]
            == sum_plan(3, columns, columns, left.data_ptr(), "narrow").block
        )
    else:
        assert not calls
