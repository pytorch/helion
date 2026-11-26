"""Opt-in batch-one Gemma 4 26B-A4B CuTe preset for the 148-SM B200.

The default gemma4_a4b_moe remains the original Triton AOT entry. Both
entries compile the same DSL function and return the same seven tensors.
"""

from __future__ import annotations

from pretuned_kernels.megakernels.gemma4_a4b_moe import (
    _helion_aot_gemma4_a4b_moe_cuda_sm100 as _triton_preset,
)
from pretuned_kernels.megakernels.gemma4_a4b_moe._helion_aot_gemma4_a4b_moe_cute_cuda_sm100 import (
    CONFIG as _CONFIG,
)
from pretuned_kernels.megakernels.gemma4_a4b_moe.gemma4_a4b_moe import gemma4_a4b_moe
from pretuned_kernels.megakernels.gemma4_a4b_moe.gemma4_a4b_moe import (
    use_cudagraph as use_cudagraph,
)
import torch
from torch._subclasses.fake_tensor import is_fake

import helion


def _supports_device(device: torch.device | int) -> bool:
    properties = torch.cuda.get_device_properties(device)
    return (properties.major, properties.minor, properties.multi_processor_count) == (
        10,
        0,
        148,
    )


def _preset_key(*args: object) -> int:
    """Check concrete binding and eager dispatch; Inductor runs its generated wrapper."""
    count = len(_triton_preset._TENSOR_SIGNATURES)
    if (
        len(args) != count + 2
        or type(args[-2]) is not int
        or type(args[-1]) is not float
        or args[-2:] != _triton_preset._STATIC_ARGS
    ):
        raise ValueError("Gemma CuTe requires the published top_k and eps")
    device = None
    storages = set()
    spans: list[tuple[int, int]] = []
    for tensor, (shape, dtype) in zip(
        args[:count], _triton_preset._TENSOR_SIGNATURES, strict=True
    ):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.layout != torch.strided
            or tensor.device.type != "cuda"
            or tensor.dtype != dtype
            or tuple(tensor.shape) != shape
            or tensor.is_neg()
            # Fake CUDA tensors used for output inference have no addresses.
            # The concrete launcher specializes the actual packet alignment.
            or (not is_fake(tensor) and tensor.data_ptr() % 16 != 0)
        ):
            raise ValueError(
                "Gemma CuTe requires the aligned, contiguous BF16 model operands"
            )
        storage = tensor.untyped_storage()
        if storage in storages:
            raise ValueError("Gemma CuTe requires independently stored model operands")
        storages.add(storage)
        if not is_fake(tensor):
            # DLPack can expose the same allocation with independent storage
            # wrappers. Check occupied bytes as well as ordinary shared views.
            start = storage.data_ptr()
            end = start + storage.nbytes()
            if any(
                start < other_end and other_start < end
                for other_start, other_end in spans
            ):
                raise ValueError(
                    "Gemma CuTe requires independently stored model operands"
                )
            spans.append((start, end))
        # A size-one dimension can be contiguous with an arbitrary stride;
        # match the published specialization, including those strides.
        stride = 1
        for extent, actual in zip(
            reversed(shape), reversed(tensor.stride()), strict=True
        ):
            if actual != stride:
                raise ValueError("Gemma CuTe requires the original row-major strides")
            stride *= extent
        if device is None:
            device = tensor.device
        elif device != tensor.device:
            raise ValueError("Gemma CuTe requires all model operands on one device")
    if device is None or device.index != torch.cuda.current_device():
        raise ValueError("Gemma CuTe requires the operand device to be current")
    if not _supports_device(device):
        raise ValueError("Gemma CuTe requires a 148-SM SM100 B200")
    return 0


gemma4_a4b_moe_cute = helion.kernel(
    gemma4_a4b_moe.fn,
    backend="cute",
    # The whole model opts into approximate exp, division, tanh and rsqrt
    # (including flush-to-zero) as well as relaxed BF16 MMA accumulation.
    fast_math=True,
    static_shapes=gemma4_a4b_moe.settings.static_shapes,
    config=helion.Config.from_dict(_CONFIG),
    key=_preset_key,
)


def _assert_outputs_close(
    actual: tuple[torch.Tensor, ...], reference: tuple[torch.Tensor, ...]
) -> None:
    """Check the seven model outputs, including exact routed expert IDs."""
    assert len(actual) == len(reference) == 7
    for i, (got, expected) in enumerate(zip(actual, reference, strict=True)):
        assert got.shape == expected.shape and got.dtype == expected.dtype
        if i == 3:
            torch.testing.assert_close(got, expected, atol=0, rtol=0)
        elif i in (1, 2):
            torch.testing.assert_close(got, expected, atol=5e-5, rtol=2e-4)
        else:
            expected_f32 = expected.float()
            diff = (got.float() - expected_f32).abs()
            assert diff.norm() <= 0.025 * expected_f32.norm(), f"output {i}: L2 error"
            assert diff.amax() <= 0.025 * expected_f32.abs().amax(), (
                f"output {i}: maximum error"
            )


@torch.inference_mode()
def main(verbose: bool = True) -> dict:
    """Benchmark CuTe against the shipped Triton preset on three B200 routing cases."""
    from pretuned_kernels import _bench
    from pretuned_kernels.megakernels.gemma4_a4b_moe import gemma4_a4b_moe as model

    if not torch.cuda.is_available() or not _supports_device(
        torch.cuda.current_device()
    ):
        reason = "Gemma 4 CuTe is pretuned for a full 148-SM B200"
        if verbose:
            print(f"Skipping gemma4_a4b_moe_cute: {reason}")
        return {"skipped": reason}
    tensors = model._make_inputs()
    args = model._kernel_args(tensors)
    cute_graph, actual = _bench.capture_cuda_graph(lambda: gemma4_a4b_moe_cute(*args))
    triton_graph, expected = _bench.capture_cuda_graph(lambda: gemma4_a4b_moe(*args))
    seen = set()
    cases = model._make_routing_cases()

    def make_calls(case: tuple[str, torch.Tensor]) -> tuple:
        label, residual = case
        tensors["residual"].copy_(residual)
        triton_graph.replay()
        cute_graph.replay()
        _assert_outputs_close(actual, expected)
        seen.add(tuple(sorted(actual[3][0].tolist())))
        return cute_graph.replay, [("helion_triton", triton_graph.replay)], label

    results = _bench.run_sweep(
        cases,
        make_calls,
        use_cudagraph=False,
        pre_captured_cudagraph=True,
        thermal_warmup_ms=5_000,
        shape_header="routing",
        verbose=verbose,
    )
    assert len(seen) == len(cases), "routing cases did not select distinct expert sets"
    return results


if __name__ == "__main__":
    main()
