from __future__ import annotations

import importlib.util
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.test_cute_register_region import _float32_extrema_input_bits
from test.test_cute_register_region import _plan

import helion
from helion._testing import skipUnlessCuteAvailable
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _numeric_register_extrema(left: torch.Tensor, right: torch.Tensor):
    minimum = torch.empty_like(left)
    maximum = torch.empty_like(left)
    for row in hl.grid(left.size(0)):
        lhs = left[row, :, :]
        rhs = right[row, :, :]
        groups = hl.arange(left.size(1))[:, None]
        peer = torch.gather(rhs, 0, (groups ^ 1).expand(-1, left.size(2)))
        minimum[row, :, :] = torch.fmin(lhs, peer)
        maximum[row, :, :] = torch.fmax(lhs, peer)
    return minimum, maximum


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _numeric_pointwise(left: torch.Tensor, right: torch.Tensor):
    out = torch.empty_like(left, dtype=right.dtype)
    for tile in hl.tile(left.size(0)):
        out[tile] = torch.fmin(left[tile], right[tile])
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _numeric_broadcast(left: torch.Tensor, right: torch.Tensor):
    out = torch.empty_like(left, dtype=right.dtype)
    for rows, columns in hl.tile([left.size(0), left.size(1)]):
        out[rows, columns] = torch.fmax(left[rows, columns], right[columns][None, :])
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
def test_numeric_extrema_preserved_in_register_region(dtype):
    left = torch.arange(96, dtype=dtype).reshape(3, 4, 8)
    right = left + 1
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound, config, plan = _plan(_numeric_register_extrema, (left, right))
        assert plan is not None
        source = bound.to_code(config)
    assert "_cute_execute_register_plan" in source
    for operation in ("aten.fmin.default", "aten.fmax.default"):
        assert operation in source
    assert "aten.isnan.default" not in source


def test_numeric_extrema_pointwise_promotion():
    left = torch.arange(256, dtype=torch.bfloat16)
    right = torch.arange(256, dtype=torch.float32) - 1
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_numeric_pointwise, (left, right))
        source = bound.to_code(helion.Config(block_sizes=[128]))
    assert "_cute_numeric_extremum" in source
    assert "minimum=" in source


def test_numeric_extrema_broadcast_promotion():
    left = torch.ones((31, 33), dtype=torch.bfloat16)
    right = torch.ones(33, dtype=torch.float32)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_numeric_broadcast, (left, right))
        source = bound.to_code(helion.Config(block_sizes=[8, 32]))
    assert "_cute_numeric_extremum" in source
    assert "minimum=" in source


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_numeric_extrema_register_native_ieee_values():
    left_bits, right_bits = _float32_extrema_input_bits()
    left = left_bits.to(torch.int32).view(torch.float32).to("cuda")
    right = right_bits.to(torch.int32).view(torch.float32).to("cuda")
    outputs = _numeric_register_extrema(left, right)
    peer_bits = right_bits[:, torch.arange(4) ^ 1, :]

    def is_nan(bits):
        return ((bits & 0x7F800000) == 0x7F800000) & ((bits & 0x007FFFFF) != 0)

    left_nan, right_nan = is_nan(left_bits), is_nan(peer_bits)
    both_nan = left_nan & right_nan
    left_order = torch.where(
        (left_bits & 0x80000000) != 0, left_bits ^ 0xFFFFFFFF, left_bits ^ 0x80000000
    )
    right_order = torch.where(
        (peer_bits & 0x80000000) != 0, peer_bits ^ 0xFFFFFFFF, peer_bits ^ 0x80000000
    )
    for actual, ordered_left in zip(
        outputs, (left_order <= right_order, left_order >= right_order), strict=True
    ):
        choose_left = right_nan | (~left_nan & ordered_left)
        expected = torch.where(choose_left, left_bits, peer_bits)
        bits = actual.cpu().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        assert torch.equal(is_nan(bits), both_nan)
        assert torch.equal(bits[~both_nan], expected[~both_nan])
    for tensor, original in ((left, left_bits), (right, right_bits)):
        assert torch.equal(
            tensor.cpu().view(torch.int32).to(torch.int64) & 0xFFFFFFFF, original
        )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_numeric_extrema_pointwise_native_promotion():
    left = torch.linspace(-4, 4, 257, device="cuda", dtype=torch.bfloat16)
    right = torch.linspace(3, -3, 257, device="cuda", dtype=torch.float32)
    right[0] = torch.nan
    left[1] = torch.nan
    torch.testing.assert_close(_numeric_pointwise(left, right), torch.fmin(left, right))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_numeric_extrema_broadcast_native_promotion():
    left = torch.linspace(-4, 4, 31 * 33, device="cuda", dtype=torch.bfloat16)
    left = left.reshape(31, 33)
    right = torch.linspace(3, -3, 33, device="cuda", dtype=torch.float32)
    right[0] = torch.nan
    left[1, 2] = torch.nan
    torch.testing.assert_close(
        _numeric_broadcast(left, right), torch.fmax(left, right[None, :])
    )


@pytest.mark.parametrize("minimum", [True, False])
@pytest.mark.parametrize("mixed", [False, True])
def test_numeric_tensor_helper_retains_sdk_contract(tmp_path, minimum, mixed):
    cutlass = pytest.importorskip("cutlass")
    import cutlass.cute as cute

    path = tmp_path / "numeric_tensor_contract.py"
    path.write_text(
        "import cutlass\n"
        "import cutlass.cute as cute\n"
        "from helion.runtime.cute.register_tensor import _cute_numeric_extremum\n"
        "@cute.kernel\n"
        "def kernel(left, right, out):\n"
        "    layout = cute.make_layout((4,))\n"
        "    lhs = cute.make_tensor(left.iterator, layout).load()\n"
        + (
            "    rhs = right.iterator.load()\n"
            if mixed
            else "    rhs = cute.make_tensor(right.iterator, layout).load()\n"
        )
        + f"    result = _cute_numeric_extremum(lhs, rhs, minimum={minimum})\n"
        "    cute.make_tensor(out.iterator, layout).store(result)\n"
        "@cute.jit\n"
        "def entry(left, right, out):\n"
        "    kernel(left, right, out).launch(grid=(1,), block=(32,))\n"
    )
    spec = importlib.util.spec_from_file_location("numeric_tensor_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tensors = [
        cute.runtime.make_fake_tensor(
            cutlass.Float32, (32, 4), (4, 1), assumed_align=16
        )
        for _ in range(3)
    ]
    initialized = torch.cuda.is_initialized()
    name = "min" if minimum else "max"
    operation = getattr(cute.math, name)
    with patch.object(cute.math, name, wraps=operation) as sdk:
        if mixed:
            with pytest.raises(TypeError):
                cute.compile.to_precompiled_mlir(
                    module.entry, *tensors, options="--gpu-arch=sm_100a"
                )
        else:
            result = cute.compile.to_precompiled_mlir(
                module.entry, *tensors, options="--gpu-arch=sm_100a"
            )
            assert result.get_bitcode()
        assert sdk.call_count == 1
        assert sdk.call_args.kwargs["propagate_nan"] is False
    assert torch.cuda.is_initialized() == initialized
