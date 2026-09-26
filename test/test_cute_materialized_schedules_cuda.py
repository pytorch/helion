"""Execute the explicit packed and row-resident materialized schedules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from examples.int4_gemm import matmul_bf16_int4
import pytest
import torch

import helion
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Callable

CUDA_DEVICE = "cuda"


def _linear_pair(left, hidden, expansion, auxiliary):
    rows, inner = left.size()
    middle_width = hidden.size(1)
    columns = expansion.size(1)
    result = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    middle = torch.empty((rows, middle_width), dtype=left.dtype, device=left.device)
    carrier = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    for row in hl.tile(rows):
        for column in hl.tile(middle_width):
            first = hl.zeros([row, column], dtype=torch.float32)
            for reduction in hl.tile(inner):
                first = torch.addmm(
                    first, left[row, reduction], hidden[reduction, column]
                )
            middle[row, column] = first
        for column in hl.tile(columns):
            value = hl.zeros([row, column], dtype=torch.float32)
            for reduction in hl.tile(middle_width):
                value = torch.addmm(
                    value, middle[row, reduction], expansion[reduction, column]
                )
            carrier[row, column] = -value
            result[row, column] = carrier[row, column] * auxiliary[row, column]
    return middle, carrier, result


def _nonlinear_pair(left, hidden, expansion, auxiliary):
    rows, inner = left.size()
    middle_width = hidden.size(1)
    columns = expansion.size(1)
    result = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    middle = torch.empty((rows, middle_width), dtype=left.dtype, device=left.device)
    carrier = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    for row in hl.tile(rows):
        for column in hl.tile(middle_width):
            first = hl.zeros([row, column], dtype=torch.float32)
            for reduction in hl.tile(inner):
                first = torch.addmm(
                    first, left[row, reduction], hidden[reduction, column]
                )
            middle[row, column] = torch.relu(first)
        for column in hl.tile(columns):
            value = hl.zeros([row, column], dtype=torch.float32)
            for reduction in hl.tile(middle_width):
                value = torch.addmm(
                    value, middle[row, reduction], expansion[reduction, column]
                )
            carrier[row, column] = torch.sigmoid(value)
            result[row, column] = auxiliary[row, column] * carrier[row, column]
    return middle, carrier, result


def _check_outputs(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
    *,
    exact: bool,
) -> None:
    assert len(actual) == len(expected)
    assert len({value.data_ptr() for value in actual}) == len(actual)
    for value, reference in zip(actual, expected, strict=True):
        # Dyadic linear cases have exact FP32 sums. The nonlinear case also
        # covers the scalar exp approximation and its final FP16 rounding.
        torch.testing.assert_close(
            value.cpu(), reference, rtol=0 if exact else 2e-3, atol=0 if exact else 1e-3
        )


def _exercise_graph(
    call: Callable[[], tuple[torch.Tensor, ...]],
    inputs: tuple[torch.Tensor, ...],
    reference: Callable[[], tuple[torch.Tensor, ...]],
    update: Callable[[int], None],
    *,
    exact: bool,
) -> None:
    first, second = call(), call()
    expected = reference()
    _check_outputs(first, expected, exact=exact)
    _check_outputs(second, expected, exact=exact)
    assert {value.data_ptr() for value in first}.isdisjoint(
        value.data_ptr() for value in (*second, *inputs)
    )
    for value in first:
        value.fill_(float("nan"))
    _check_outputs(second, expected, exact=exact)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = call()
    for iteration in (1, 2):
        update(iteration)
        expected = reference()
        _check_outputs(call(), expected, exact=exact)
        for _ in range(2):
            for value in captured:
                value.fill_(float("nan"))
            graph.replay()
            _check_outputs(captured, expected, exact=exact)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("reduction", (64, 128, 256))
def test_packed_schedule_numerics_and_graph_replay(reduction: int) -> None:
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("packed BF16 warp MMA requires SM80 or later")
    generator = torch.Generator().manual_seed(1549)
    a_cpu = torch.empty((32, reduction), dtype=torch.bfloat16)
    packed_cpu = torch.empty((reduction // 2, 12), dtype=torch.int8)
    inputs = (a_cpu.to(CUDA_DEVICE), packed_cpu.to(CUDA_DEVICE))

    def update(iteration: int) -> None:
        a_cpu.copy_(torch.randint(-4, 5, a_cpu.shape, generator=generator).float() / 8)
        # Cover every byte, hence both signed nibbles, on each replay.
        packed_cpu.copy_(
            ((torch.arange(packed_cpu.numel()) * 73 + iteration * 19) % 256 - 128)
            .to(torch.int8)
            .view_as(packed_cpu)
        )
        inputs[0].copy_(a_cpu)
        inputs[1].copy_(packed_cpu)

    def reference() -> tuple[torch.Tensor, ...]:
        packed = packed_cpu.to(torch.int32)
        low = packed & 15
        low = torch.where(low >= 8, low - 16, low)
        unpacked = torch.stack((low, packed >> 4), dim=1).reshape(reduction, 12)
        return ((a_cpu.double() @ unpacked.double()).to(torch.bfloat16),)

    update(0)
    kernel = helion.kernel(
        matmul_bf16_int4.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        fast_math=False,
        cute_region_fission=True,
        cute_full_slice_matmul_tiling=True,
        cute_segmented_matmul_tiling=True,
        cute_flatten_nested_reductions=True,
        cute_materialize_transformed_operands=True,
    )
    bound = kernel.bind(inputs)
    config = bound.config_spec.normalized_config(
        helion.Config.from_dict(
            bound.config_spec.default_config().config
            | {"cute_materialized_operand_schedule": "warp_narrow4"}
        )
    )
    assert config["cute_materialized_operand_schedule"] == "warp_narrow4"
    assert "def _helion_packed_operand(" in bound.to_code(config)
    bound.set_config(config)
    _exercise_graph(lambda: (bound(*inputs),), inputs, reference, update, exact=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ("warp_rows2", "warp_rows2_matrix"))
@pytest.mark.parametrize("nonlinear", (False, True), ids=("linear", "nonlinear"))
def test_row_schedules_numerics_and_graph_replay(mode: str, nonlinear: bool) -> None:
    major = torch.cuda.get_device_capability()[0]
    if major < (9 if mode == "warp_rows2_matrix" else 8):
        pytest.skip("selected row schedule requires newer matrix instructions")
    generator = torch.Generator().manual_seed(1553)
    # Nine CTAs include an odd final row. All three independently returned
    # allocations must be complete, including after graph output poisoning.
    host_inputs = tuple(
        torch.empty(shape, dtype=torch.float16)
        for shape in ((17, 32), (32, 64), (64, 128), (17, 128))
    )
    inputs = tuple(value.to(CUDA_DEVICE) for value in host_inputs)

    def update(iteration: int) -> None:
        for host, device in zip(host_inputs, inputs, strict=True):
            host.copy_(
                torch.randint(-4, 5, host.shape, generator=generator).float() / 16
            )
            device.copy_(host)

    def reference() -> tuple[torch.Tensor, ...]:
        left, hidden, expansion, auxiliary = (value.double() for value in host_inputs)
        first = left @ hidden
        middle = (torch.relu(first) if nonlinear else first).half()
        second = middle.double() @ expansion
        carrier = (torch.sigmoid(second) if nonlinear else -second).half()
        result = (carrier.double() * auxiliary).half()
        return middle, carrier, result

    update(0)
    kernel = helion.kernel(
        _nonlinear_pair if nonlinear else _linear_pair,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        fast_math=False,
        cute_region_fission=True,
        cute_full_slice_matmul_tiling=True,
        cute_segmented_matmul_tiling=True,
        cute_flatten_nested_reductions=True,
        cute_materialize_transformed_operands=True,
    )
    bound = kernel.bind(inputs)
    config = bound.config_spec.normalized_config(
        helion.Config.from_dict(
            bound.config_spec.default_config().config
            | {"cute_materialized_schedule": mode, "cute_min_blocks_per_mp": 1}
        )
    )
    assert config["cute_materialized_schedule"] == mode
    source = bound.to_code(config)
    assert "def _helion_row_resident(" in source
    assert ("def _row_matrix_store(" in source) == (mode == "warp_rows2_matrix")
    bound.set_config(config)
    _exercise_graph(
        lambda: bound(*inputs), inputs, reference, update, exact=not nonlinear
    )
