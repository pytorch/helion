from __future__ import annotations

from itertools import accumulate
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._compiler.cute import tcgen05_flat_grouped_kernel
from helion._compiler.cute.tcgen05_grouped_descriptors import DESCRIPTOR_KEY
from helion._compiler.cute.tcgen05_grouped_descriptors import ELEMENT_BYTES
from helion._compiler.cute.tcgen05_grouped_descriptors import LARGE_ALLOC_MIN_BYTES
from helion._compiler.cute.tcgen05_grouped_descriptors import WRAPPED
from helion._testing import import_example
from helion._testing.cuda_page_touch import PageTouches
from helion._testing.cuda_page_touch import tensor_at_block_end
from helion._testing.cute_flat_grouped import native_kernel_source

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

jagged_dense_bmm = import_example("jagged_dense_bmm").jagged_dense_bmm

CUDA_DEVICE = "cuda"

# selection, reduction, columns, groups, offset dtype
_MODES: dict[str, tuple[dict[str, object], int, int, int, torch.dtype]] = {
    "rna-dynamic-tail-ring-wrap": (
        {"cute_grouped_rna_warps": 1, "cute_grouped_rna_block_k": 64},
        256,
        68,
        5,
        torch.int32,
    ),
    "rna-wrapped": (
        {
            "cute_grouped_rna_warps": 8,
            "cute_grouped_rna_block_k": 32,
            "cute_grouped_descriptor_policy": "wrapped",
        },
        128,
        128,
        5,
        torch.int64,
    ),
    "tma-rn-dynamic-warp-prefix-tail": (
        {
            "cute_mma_f32_conversion": "tma_rn",
            "cute_grouped_rna_block_k": 32,
            "cute_grouped_ab_stages": 2,
        },
        128,
        68,
        37,
        torch.int64,
    ),
    "tma-rn-wrapped-block-prefix-two-ctas": (
        {
            "cute_mma_f32_conversion": "tma_rn",
            "cute_grouped_rna_block_k": 32,
            "cute_grouped_ab_stages": 2,
            "cute_grouped_ctas_per_sm": 2,
            "cute_grouped_descriptor_policy": "wrapped",
            "cute_grouped_prefix_scan": "block",
        },
        128,
        128,
        65,
        torch.int64,
    ),
    "raw-warp-k-refill-n-tail": (
        {"cute_mma_f32_conversion": "warp_raw"},
        128,
        68,
        5,
        torch.int64,
    ),
}
# One compile per mode for the module: the shapes are static, so every test
# launches the mode's kernel with its own tensors. With each kernel: the source
# of its native grouped module.
_BOUND: dict[str, tuple[BoundKernel, str]] = {}


def _lengths(groups: int) -> list[int]:
    # Unequal, empty, and partial row tiles; the larger case spans prefix warps.
    return [0, 1, 0, 129, 257] + [1] * (groups - 5)


def _bind(
    selection: dict[str, object],
    inputs: tuple[torch.Tensor, ...],
    *,
    optional: tuple[str, ...] = (),
) -> tuple[BoundKernel, str]:
    """Bind and select ``selection``; a binding may normalize the
    ``optional`` keys away. Returns the kernel and the source of its native
    grouped module."""
    kernel = helion.kernel(
        jagged_dense_bmm.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        dot_precision="tf32",
        cute_region_fission=True,
        cute_full_slice_matmul_tiling=True,
        cute_segmented_matmul_tiling=True,
        cute_flatten_nested_reductions=True,
        cute_materialize_transformed_operands=True,
    )
    bound = kernel.bind(inputs)
    config = bound.config_spec.normalized_config(
        helion.Config.from_dict(
            bound.config_spec._base_default_config().config | selection
        )
    )
    for key, value in selection.items():
        assert config.get(key, value if key in optional else None) == value
    code = bound.to_code(config)
    # A numerical pass from a silent fallback would not cover the selected mode.
    assert "_helion_cute_native_grouped_plan" in code
    bound.set_config(config)
    return bound, native_kernel_source(code)


def _bound(mode: str, inputs: tuple[torch.Tensor, ...]) -> tuple[BoundKernel, str]:
    if mode not in _BOUND:
        _BOUND[mode] = _bind(_MODES[mode][0], inputs)
    return _BOUND[mode]


# The dynamic policy's kernel imports the helper that clears a template's
# allocation class, and calls it on the row-packed operand's template.
_HELPER_IMPORT = (
    "from helion._compiler.cute.tensormap_alloc_class import "
    "select_small_alloc_class_every_lane"
)
_OPERAND_CLEAR = (
    "select_small_alloc_class_every_lane(tcgen05_grouped_tensormap_b_smem_ptr, 5)"
)


def _require_grouped_native_device() -> None:
    # The targets the compiler admits these kernels for (prove_flat_grouped_rna).
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("grouped native schedules require an SM10x device")


def _reference(
    lengths: list[int],
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # CPU FP64 avoids both the candidate conversion and global CUDA TF32 settings.
    left, right, addend = (value.cpu().double() for value in (jagged, dense, bias))
    result = torch.empty((left.size(0), right.size(2)), dtype=torch.float64)
    magnitude = torch.empty_like(result)
    start = 0
    for group, length in enumerate(lengths):
        stop = start + length
        result[start:stop] = left[start:stop] @ right[group] + addend[group]
        magnitude[start:stop] = left[start:stop].abs() @ right[group].abs()
        start = stop
    assert start == left.size(0)
    return result, magnitude


def _check_output(
    actual: torch.Tensor,
    expected: torch.Tensor,
    magnitude: torch.Tensor,
    *,
    rounded: bool,
) -> None:
    assert actual.dtype is torch.float32 and actual.shape == expected.shape
    actual_cpu = actual.cpu().double()
    if rounded:
        # One TF32 ulp per input permits truncation as well as RN/RNA.
        # 0.0021 exceeds 2*2**-10 + 2**-20 plus FP32 accumulation error
        # for these K<=256 cases. The dyadic passes below require exact output.
        allowance = magnitude * 0.0021 + 1e-5
        error = (actual_cpu - expected).abs()
        assert torch.all(error <= allowance), (error / allowance).max().item()
    else:
        torch.testing.assert_close(actual_cpu, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", _MODES)
def test_grouped_fp32_modes_numerics_and_graph_replay(mode: str) -> None:
    _require_grouped_native_device()
    _selection, reduction, columns, groups, offset_dtype = _MODES[mode]
    lengths = _lengths(groups)
    rows = sum(lengths)
    generator = torch.Generator().manual_seed(7123)

    def values(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randint(-8, 9, shape, generator=generator).float() / 16

    offsets = torch.tensor(
        list(accumulate(lengths, initial=0)), dtype=offset_dtype, device=CUDA_DEVICE
    )
    jagged = values((rows, reduction)).to(CUDA_DEVICE)
    dense = values((groups, reduction, columns)).to(CUDA_DEVICE)
    bias = values((groups, columns)).to(CUDA_DEVICE)
    inputs = (offsets, jagged, dense, bias)
    bound, _native = _bound(mode, inputs)
    expected, magnitude = _reference(lengths, jagged, dense, bias)
    first = bound(*inputs)
    _check_output(first, expected, magnitude, rounded=False)
    second = bound(*inputs)
    assert first.data_ptr() != second.data_ptr()
    first.fill_(float("nan"))
    _check_output(second, expected, magnitude, rounded=False)

    # Compile and warm outside capture, retaining the original allocating caller.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        bound(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = bound(*inputs)

    for rounded in (False, True):
        lengths = list(reversed(lengths))
        offsets.copy_(
            torch.tensor(list(accumulate(lengths, initial=0)), dtype=offset_dtype)
        )
        jagged.copy_(values((rows, reduction)))
        dense.copy_(values((groups, reduction, columns)))
        bias.copy_(values((groups, columns)))
        if rounded:
            # Finite values around TF32 rounding boundaries, including cancellation.
            jagged.add_(2**-12)
            dense.add_(2**-12)
        expected, magnitude = _reference(lengths, jagged, dense, bias)
        captured.fill_(float("nan"))
        graph.replay()
        _check_output(captured, expected, magnitude, rounded=rounded)


def _touches_past_the_row_packed_operand(
    selection: dict[str, object],
    lengths: list[int],
    reduction: int,
    columns: int,
    offset_dtype: torch.dtype,
    *,
    optional: tuple[str, ...] = (),
    mode: str | None = None,
) -> tuple[PageTouches, str]:
    """Launch with the row-packed operand at the end of a UVM block: what the
    kernel touched behind it (its results are checked) and the source of its
    native grouped module. ``mode`` shares that mode's compiled kernel."""
    groups, rows = len(lengths), sum(lengths)
    generator = torch.Generator().manual_seed(7123)

    def values(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randint(-8, 9, shape, generator=generator).float() / 16

    packed = values((rows, reduction))
    offsets = torch.tensor(
        list(accumulate(lengths, initial=0)), dtype=offset_dtype, device=CUDA_DEVICE
    )
    dense = values((groups, reduction, columns)).to(CUDA_DEVICE)
    bias = values((groups, columns)).to(CUDA_DEVICE)
    expected, magnitude = _reference(lengths, packed, dense, bias)
    inputs = (offsets, packed.to(CUDA_DEVICE), dense, bias)
    bound, native = (
        _bind(selection, inputs, optional=optional)
        if mode is None
        else _bound(mode, inputs)
    )
    with tensor_at_block_end((rows, reduction)) as jagged:
        jagged.write(packed)
        actual = bound(offsets, jagged.tensor, dense, bias)
        touches = jagged.touches()
        _check_output(actual, expected, magnitude, rounded=False)
    if not touches.observable:
        pytest.skip("the kernel's page touches are not observable here")
    return touches, native


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "mode",
    ["rna-dynamic-tail-ring-wrap", "tma-rn-dynamic-warp-prefix-tail", "rna-wrapped"],
)
def test_grouped_fp32_loads_stay_inside_the_row_packed_operand(mode: str) -> None:
    """A one-row group at the end of the operand: its loads leave the group.

    The dynamic policy published that group's TensorMap with the large
    allocation class of the whole operand, and the out-of-bounds requests of
    its loads went up to 128 KiB past the operand: an illegal memory access
    when the allocator had unmapped what follows. Its kernel now clears the
    class in the template it rewrites. The wrapped policy is admitted
    because its operand spans that window.
    """
    _require_grouped_native_device()
    selection, reduction, columns, groups, offset_dtype = _MODES[mode]
    lengths = list(reversed(_lengths(groups)))
    assert [length for length in lengths if length][-1] == 1
    assert sum(lengths) * reduction * ELEMENT_BYTES >= LARGE_ALLOC_MIN_BYTES
    touches, native = _touches_past_the_row_packed_operand(
        selection, lengths, reduction, columns, offset_dtype, mode=mode
    )
    assert touches.past_end_pages == 0, (
        f"read up to {touches.past_end_reach} bytes past the row-packed operand"
    )
    # The kernel under test clears the operand's template.
    assert (_OPERAND_CLEAR in native) is (selection.get(DESCRIPTOR_KEY) != WRAPPED)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rewritten_template_keeps_the_large_class_without_the_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Why the kernels clear the selector themselves, and the positive
    control of the observation above: the rewrite (``update_tensormap``)
    leaves the class as the template was encoded, under CuTe DSL 4.8.1 too.
    The same kernel, rendered with a helper that emits nothing in place of
    the clear, loads the one-row group at the end of the operand through a
    large-class TensorMap, and its out-of-bounds requests are seen behind
    the operand. A driver whose encoder does not select the large class for
    this operand leaves nothing to observe, and the test skips there.
    """
    _require_grouped_native_device()
    mode = "rna-dynamic-tail-ring-wrap"
    selection, reduction, columns, groups, offset_dtype = _MODES[mode]
    template = tcgen05_flat_grouped_kernel._KERNEL
    assert template.count(_HELPER_IMPORT) == 1
    # Another source than the mode's own, so nothing compiled for one
    # serves the other.
    monkeypatch.setattr(
        tcgen05_flat_grouped_kernel,
        "_KERNEL",
        template.replace(
            _HELPER_IMPORT,
            "select_small_alloc_class_every_lane = lambda template, warp: None",
        ),
    )
    touches, native = _touches_past_the_row_packed_operand(
        selection,
        list(reversed(_lengths(groups))),
        reduction,
        columns,
        offset_dtype,
    )
    assert _OPERAND_CLEAR in native and _HELPER_IMPORT not in native
    if touches.past_end_reach == 0:
        pytest.skip(
            "the driver did not encode this operand's TensorMap in the large "
            "allocation class: no over-read to observe without the clear"
        )
    assert touches.past_end_reach <= LARGE_ALLOC_MIN_BYTES


# 128 fp32 per row: 256 rows are the 131072 bytes a large-class descriptor
# may reach from its address, which the wrapped policy needs of its buffers.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "lengths,admitted",
    [
        pytest.param([126, 129, 0, 0, 1], True, id="spans-the-window"),
        pytest.param([100, 0, 0, 0, 1], False, id="under-the-window"),
    ],
)
def test_wrapped_policy_needs_buffers_that_span_the_large_class_window(
    lengths: list[int], admitted: bool
) -> None:
    """The driver encodes a wrapped TensorMap large whatever the buffer holds.

    Under the window the policy read up to 128 KiB past the row-packed
    operand on every launch; such a binding now keeps the dynamic policy
    (the config key is normalized away). At exactly the window the policy is
    admitted and its loads stay inside.
    """
    _require_grouped_native_device()
    selection, reduction, columns, _groups, offset_dtype = _MODES["rna-wrapped"]
    operand_bytes = sum(lengths) * reduction * ELEMENT_BYTES
    assert (operand_bytes >= LARGE_ALLOC_MIN_BYTES) is admitted
    assert (operand_bytes == LARGE_ALLOC_MIN_BYTES) is admitted
    touches, native = _touches_past_the_row_packed_operand(
        selection,
        lengths,
        reduction,
        columns,
        offset_dtype,
        optional=(DESCRIPTOR_KEY,),
    )
    assert touches.past_end_pages == 0, (
        f"read up to {touches.past_end_reach} bytes past the row-packed operand"
    )
    assert ("TensorMapManager" not in native) is admitted
