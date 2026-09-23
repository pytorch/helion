from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

from test._cute_binding import _mock_cuda_unavailable

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from test._cute_paired_sum_model import run_literal

from helion.runtime.cute import single_sum
from helion.runtime.cute.paired_sum import sum_plan
from helion.runtime.cute.single_sum import plan_for_single
from helion.runtime.cute.single_sum import try_single_sum_cast

if TYPE_CHECKING:
    from collections.abc import Iterator


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
    (
        (64, 1024, 1024),
        (0, 129, 129),
        (64, 0, 1),
        (1, 1, 1),
        (1, 17, 17),
        (65, 130, 130),
        (65, 131, 131),
        (257, 132, 132),
        (65, 128, 132),
        (65, 131, 136),
    ),
)
def test_single_literal_tree_and_ownership(
    layout: str, rows: int, columns: int, stride: int
) -> None:
    plan = sum_plan(rows, columns, stride, 16, layout)
    assert plan is not None
    result = run_literal(plan, stride=stride, single=True, output_dtype="float16")
    assert result["loads"] == rows * columns
    assert result["stores"] == columns


@pytest.mark.parametrize("dtype", ("bfloat16", "float32"))
def test_other_single_casts_keep_the_same_fp32_tree(dtype: str) -> None:
    plan = sum_plan(65, 130, 130, 16, "narrow")
    assert plan is not None
    assert run_literal(plan, single=True, output_dtype=dtype)["single_final_cast"]


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_barrier",
        "ascending_tree",
        "missing_row_guard",
        "missing_column_guard",
        "cast_partial",
    ),
)
def test_single_tree_rejects_broken_ownership_order_or_early_cast(
    mutation: str,
) -> None:
    plan = sum_plan(65, 132, 132, 16, "mapped")
    assert plan is not None
    with pytest.raises((AssertionError, KeyError)):
        run_literal(plan, single=True, output_dtype="float16", mutation=mutation)


def test_fp16_rounding_at_all_finite_adjacent_midpoints_and_neighbors() -> None:
    # The device's RN opcode is checked by offline compilation. Here an integer
    # adjacency/tie-even oracle checks every finite FP16 boundary independently.
    codes = torch.arange(0x7BFF, dtype=torch.int32)
    low = codes.to(torch.int16).view(torch.float16).float()
    high = (codes + 1).to(torch.int16).view(torch.float16).float()
    middle = ((low.double() + high.double()) * 0.5).float()
    below = torch.nextafter(middle, torch.full_like(middle, -float("inf")))
    above = torch.nextafter(middle, torch.full_like(middle, float("inf")))
    tie = torch.where(codes % 2 == 0, codes, codes + 1)
    for sign in (1, -1):
        sign_bit = 0 if sign == 1 else 0x8000
        for values, expected in ((below, codes), (middle, tie), (above, codes + 1)):
            actual = (values * sign).to(torch.float16).view(torch.int16).to(
                torch.int32
            ) & 0xFFFF
            assert torch.equal(actual, expected | sign_bit)
    special = torch.tensor(
        [0.0, -0.0, 65504.0, -65504.0, 65520.0, -65520.0, float("inf"), -float("inf")]
    )
    expected = torch.tensor(
        [0, 0x8000, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7C00, 0xFC00], dtype=torch.int32
    )
    assert torch.equal(
        special.half().view(torch.int16).to(torch.int32) & 0xFFFF, expected
    )
    assert torch.isnan(torch.tensor(float("nan")).half())


def _forbidden(*args: object, **kwargs: object) -> None:
    raise AssertionError("fallback must retain the original host operations")


def _try(tensor: torch.Tensor) -> torch.Tensor | None:
    return try_single_sum_cast(
        tensor,
        torch.float16,
        dtype_is_tensor=False,
        layout="narrow",
        _launcher=_forbidden,
    )


def test_cpu_fallback_preserves_autograd_and_aliases() -> None:
    tensor = torch.randn(17, 20, requires_grad=True)
    assert _try(tensor) is None
    output = tensor.sum(0).to(torch.float16)
    output.float().sum().backward()
    assert torch.equal(tensor.grad, torch.ones_like(tensor))
    assert output.data_ptr() != tensor.data_ptr()


def test_dispatch_and_unknown_tensor_methods_decline_before_operations() -> None:
    tensor = torch.ones(3, 8)

    class FunctionMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            raise AssertionError("guard called a custom function")

    class DispatchMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            raise AssertionError("guard called a custom dispatch")

    with FunctionMode():
        assert _try(tensor) is None
    with DispatchMode():
        assert _try(tensor) is None
    for method in ("sum", "to", "new_empty"):
        with patch.object(torch.Tensor, method, _forbidden):
            assert _try(tensor) is None


def test_admission_preserves_byte_limit_autograd_and_alignment_tree() -> None:
    tensor = torch.empty(64, 128)
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        assert plan_for_single(tensor, dtype, "narrow") is not None
    shifted = torch.empty(64 * 128 + 1)[1:].view(64, 128)
    shifted_plan = plan_for_single(shifted, torch.float16, "narrow")
    assert shifted_plan is not None and shifted_plan.vector == 1
    assert plan_for_single(tensor[:, ::2], torch.float16, "narrow") is None
    assert plan_for_single(tensor, torch.float64, "narrow") is None
    assert plan_for_single(tensor.requires_grad_(), torch.float16, "narrow") is None
    assert sum_plan(64, 128, 2**24, 16, "narrow") is None
    assert sum_plan(32, 128, 2**24, 16, "narrow") is not None
    assert sum_plan(65536, 128, 128, 16, "narrow") is None


@pytest.mark.parametrize("layout", ("mapped", "narrow"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("stride", (1, 4))
def test_singleton_cancellation_domain_retains_original_sum_and_cast(
    layout: str, dtype: torch.dtype, stride: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The original [64,1] fixture is still covered: it must now retain ATen's
    # input-reduction tree, whose cancellation result differs from our y tree.
    tensor = torch.zeros(64, stride)[:, :1]
    tensor[0, 0], tensor[4, 0], tensor[16, 0] = 1e8, -1e8, 1
    assert plan_for_single(tensor, dtype, layout) is None
    monkeypatch.setattr(
        torch.Tensor, "device", property(lambda self: torch.device("cuda"))
    )
    monkeypatch.setattr(single_sum, "_compatible_aten_sum", _forbidden)
    monkeypatch.setattr(single_sum, "_compatible_aten_half_cast", _forbidden)
    assert (
        try_single_sum_cast(
            tensor,
            dtype,
            dtype_is_tensor=False,
            layout=layout,
            _launcher=_forbidden,
        )
        is None
    )


def test_half_header_identity_unknown_build_and_no_repeated_file_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    single_sum._compatible_aten_half_cast.cache_clear()
    try:
        with patch.object(
            Path, "read_bytes", autospec=True, side_effect=Path.read_bytes
        ) as read:
            assert single_sum._compatible_aten_half_cast()
            assert single_sum._compatible_aten_half_cast()
            assert read.call_count == 1
        monkeypatch.setattr(torch, "__file__", str(tmp_path / "__init__.py"))
        single_sum._compatible_aten_half_cast.cache_clear()
        assert not single_sum._compatible_aten_half_cast()
        header = tmp_path / "include" / single_sum._ATEN_HALF_HEADER
        header.parent.mkdir(parents=True)
        header.write_text("unknown half conversion")
        single_sum._compatible_aten_half_cast.cache_clear()
        assert not single_sum._compatible_aten_half_cast()
    finally:
        single_sum._compatible_aten_half_cast.cache_clear()


@pytest.mark.parametrize("columns", (0, 20))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_single_fresh_output_launch_order_and_fallback_gates(
    columns: int, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    tensor = torch.ones(3, columns)
    owner = torch.empty(columns, dtype=dtype)
    monkeypatch.setattr(
        torch.Tensor, "device", property(lambda self: torch.device("cuda"))
    )
    monkeypatch.setattr(single_sum, "_compatible_aten_sum", lambda: True)
    monkeypatch.setattr(single_sum, "_compatible_aten_half_cast", lambda: True)
    calls = []

    def launch(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    outputs = [
        try_single_sum_cast(
            tensor, owner, dtype_is_tensor=True, layout="narrow", _launcher=launch
        )
        for _ in range(2)
    ]
    assert all(output is not None for output in outputs)
    assert len({id(output) for output in outputs}) == 2
    assert all(
        output.shape == (columns,) and output.dtype == dtype for output in outputs
    )
    assert all(not output.requires_grad for output in outputs)
    if columns:
        assert len({output.data_ptr() for output in outputs}) == 2
        assert len(calls) == 2 and calls[0][0][0] is calls[1][0][0]
        assert all(call[0][2] is tensor for call in calls)
        assert all(
            call[0][3] is output for call, output in zip(calls, outputs, strict=True)
        )
    else:
        assert not calls
    monkeypatch.setattr(single_sum, "_compatible_aten_sum", lambda: False)
    assert _try(tensor) is None
    monkeypatch.setattr(single_sum, "_compatible_aten_sum", lambda: True)
    monkeypatch.setattr(single_sum, "_compatible_aten_half_cast", lambda: False)
    assert _try(tensor) is None


def test_runtime_subclasses_duals_and_nonordinary_views_are_not_admitted() -> None:
    tensor = torch.ones(3, 8)

    class Subclass(torch.Tensor):
        pass

    for value in (
        tensor.as_subclass(Subclass),
        torch._neg_view(tensor),
        torch._to_functional_tensor(tensor),
    ):
        assert _try(value) is None
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(tensor, torch.ones_like(tensor))
        assert _try(dual) is None
