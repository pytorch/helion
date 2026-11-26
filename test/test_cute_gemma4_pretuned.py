"""Exercise the opt-in public Gemma preset on its complete original geometry."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from pretuned_kernels import run as pretuned_benchmarks
from pretuned_kernels.megakernels.gemma4_a4b_moe import (
    _helion_aot_gemma4_a4b_moe_cuda_sm100 as aot,
)
from pretuned_kernels.megakernels.gemma4_a4b_moe import gemma4_a4b_moe as model
from pretuned_kernels.megakernels.gemma4_a4b_moe.gemma4_a4b_moe_cute import (
    _assert_outputs_close,
)
from pretuned_kernels.megakernels.gemma4_a4b_moe.gemma4_a4b_moe_cute import _preset_key
from pretuned_kernels.megakernels.gemma4_a4b_moe.gemma4_a4b_moe_cute import (
    gemma4_a4b_moe_cute,
)
import pytest
import torch
from torch._functorch import config as functorch_config
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion._compat import supports_torch_compile_template_lowering
from helion._compiler.ast_extension import unparse
from helion._testing import float32_matmul_precision
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_render_target

pytestmark = pytest.mark.cpu_only


def test_b200_benchmark_discovers_both_gemma_presets() -> None:
    for name in ("gemma4_a4b_moe", "gemma4_a4b_moe_cute"):
        assert name in pretuned_benchmarks.KERNELS
        # Hardware selection is derived from the shipped config file and must
        # skip an H100 before importing either benchmark or touching a device.
        record = pretuned_benchmarks.run_kernel(name, "h100")
        assert record["pretuned_hardware"] == ["b200"] and "skipped" in record


def test_benchmark_skips_partial_b200_before_allocating_weights() -> None:
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(major=10, minor=0, multi_processor_count=74),
        ),
        patch.object(
            model, "_make_inputs", side_effect=AssertionError("allocated weights")
        ),
    ):
        result = pretuned_benchmarks.run_kernel("gemma4_a4b_moe_cute", "b200")
    assert "148-SM" in result["skipped"] and "error" not in result


def test_preset_preserves_the_published_triton_kernel() -> None:
    assert gemma4_a4b_moe_cute.fn is model.gemma4_a4b_moe.fn
    assert model.gemma4_a4b_moe.settings.backend == "triton"
    assert gemma4_a4b_moe_cute.settings.backend == "cute"
    with pytest.raises(ValueError, match="top_k and eps"):
        gemma4_a4b_moe_cute.bind((torch.empty(()),) * 9 + (8.0, model.EPS))


def test_traced_preset_checks_layout_without_reading_a_fake_tensor_address() -> None:
    with (
        functorch_config.patch(fake_tensor_allow_unsafe_data_ptr_access=False),
        FakeTensorMode(),
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(major=10, minor=0, multi_processor_count=148),
        ),
    ):
        device = torch.device("cuda", 0)
        tensors = [
            torch.empty(shape, dtype=dtype, device=device)
            for shape, dtype in aot._TENSOR_SIGNATURES
        ]
        assert _preset_key(*tensors, *aot._STATIC_ARGS) == 0
        tensors[-1] = tensors[1].view_as(tensors[1])
        with pytest.raises(ValueError, match="independently stored"):
            _preset_key(*tensors, *aot._STATIC_ARGS)


def test_stored_config_renders_the_complete_public_projection() -> None:
    if not supports_torch_compile_template_lowering():
        pytest.skip("Inductor template rendering requires a supported PyTorch")
    from helion._compiler._inductor.template_buffer import render_inductor_ast

    with FakeTensorMode(allow_non_fake_inputs=True), torch.device("cpu"):
        operands = tuple(torch.empty(s, dtype=d) for s, d in aot._TENSOR_SIGNATURES)
    entry = helion.kernel(
        gemma4_a4b_moe_cute.fn,
        backend="cute",
        fast_math=gemma4_a4b_moe_cute.settings.fast_math,
        static_shapes=gemma4_a4b_moe_cute.settings.static_shapes,
    )
    with cpu_render_target(
        fake_device_sm_count=148,
        smem_budget_bytes=232448,
        reduction_smem_budget_bytes=232448,
    ):
        bound = cpu_bind(entry, operands + aot._STATIC_ARGS)
        code = unparse(
            render_inductor_ast(
                bound, gemma4_a4b_moe_cute.configs[0], symbolic_input_layouts=False
            )
        )
    assert "_stage_epilogue_" in code and "_cute_staged_indexed_row_dot(" in code


@pytest.fixture(scope="module")
def b200_inputs() -> dict[str, torch.Tensor]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    if (properties.major, properties.minor, properties.multi_processor_count) != (
        10,
        0,
        148,
    ):
        pytest.skip("The Gemma preset targets the 148-SM SM100 B200")
    return model._make_inputs()


def _eager_outputs(t: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Evaluate the selected experts in PyTorch, including the model's BF16 roundings."""

    def rms(value: torch.Tensor) -> torch.Tensor:
        f32 = value.float()
        return (f32 * (f32.square().mean(dim=-1, keepdim=True) + model.EPS).rsqrt()).to(
            value.dtype
        )

    normalized = rms(t["residual"])
    router_input = normalized * t["root_size"] * t["router_scale"]
    logits = router_input.float() @ t["router_weight"].float().T
    selected, ids = logits.topk(model.TOP_K)
    weights = selected.softmax(-1) * t["per_expert_scale"][ids].float()
    expert_input = (normalized * t["pre_ff_norm_weight"]).float()
    activations = []
    expert_outputs = []
    for slot, expert in enumerate(ids[0].tolist()):
        gate_up = (
            expert_input @ t["expert_gate_up_weight"][expert].float().T
        ).bfloat16()
        gate, up = gate_up.chunk(2, dim=-1)
        activation = (
            torch.nn.functional.gelu(gate.float(), approximate="tanh").bfloat16() * up
        )
        down = activation.float() @ t["expert_down_weight"][expert].float().T
        activations.append(activation)
        expert_outputs.append((down * weights[:, slot : slot + 1]).bfloat16())
    experts = torch.stack(expert_outputs, dim=1)
    reduced = experts.float().sum(dim=1).bfloat16()
    return (
        rms(reduced) * t["post_ff_norm_weight"],
        logits,
        weights,
        ids.int(),
        torch.stack(activations, dim=1),
        experts,
        reduced,
    )


@pytest.mark.cuda
def test_b200_preset_matches_all_seven_outputs(
    b200_inputs: dict[str, torch.Tensor],
) -> None:
    # One native launch for the PR lane; graph replay and Inductor stay below.
    args = model._kernel_args(b200_inputs)
    with float32_matmul_precision("highest"):
        expected = _eager_outputs(b200_inputs)
    _assert_outputs_close(gemma4_a4b_moe_cute(*args), expected)


@pytest.mark.cuda
@pytest.mark.slow
def test_b200_preset_replays_all_published_outputs_on_changed_routes(
    b200_inputs: dict[str, torch.Tensor],
) -> None:
    tensors = b200_inputs
    args = model._kernel_args(tensors)
    routes = model._make_routing_cases()
    residual = tensors["residual"]
    # Warm the driver occupancy admission on the same executable as capture.
    gemma4_a4b_moe_cute(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = gemma4_a4b_moe_cute(*args)
    routed = set()
    for _, case in (*routes, routes[0]):
        residual.copy_(case)
        expected = model.gemma4_a4b_moe(*args)
        graph.replay()
        _assert_outputs_close(outputs, expected)
        routed.add(tuple(outputs[3].flatten().tolist()))
    assert len(routed) == len(routes)
    # Dynamo's output inference sees fake CUDA tensors. Both the composite
    # dispatcher and Inductor's concrete template keep the published outputs.
    for backend in (
        ("eager", "inductor") if supports_torch_compile_template_lowering() else ()
    ):
        compiled = torch.compile(gemma4_a4b_moe_cute, backend=backend, fullgraph=True)
        actual = compiled(*args)
        assert len(actual) == len(outputs)
        for got, ref in zip(actual, outputs, strict=True):
            torch.testing.assert_close(got, ref, atol=0, rtol=0)
        if backend != "inductor":
            continue
        # Inductor's generated wrapper does not call the eager key again.
        # Changing a read-only operand must nevertheless change every
        # dependent result exactly as if the same weights were separately owned.
        pre_norm = tensors["pre_ff_norm_weight"]
        distinct = (*args[:8], pre_norm.clone(), *args[9:])
        expected = gemma4_a4b_moe_cute(*distinct)
        for view in (pre_norm.detach(), torch.from_dlpack(pre_norm)):
            with pytest.raises(ValueError, match="independently stored"):
                gemma4_a4b_moe_cute(*args[:8], view, *args[9:])
            actual = compiled(*args[:8], view, *args[9:])
            for got, ref in zip(actual, expected, strict=True):
                torch.testing.assert_close(got, ref, atol=0, rtol=0)
    # The same already-warm tensor tuple must still belong to the active
    # device. No allocation or launch is attempted on the mocked device.
    with (
        patch.object(
            torch.cuda, "current_device", return_value=residual.device.index + 1
        ),
        pytest.raises(ValueError, match="current"),
    ):
        gemma4_a4b_moe_cute(*args)
    # A warm shape cache must not bypass the effective pointer alignment.
    weak = residual.new_empty(residual.numel() + 1)[1:].view_as(residual)
    with pytest.raises(ValueError, match="aligned"):
        gemma4_a4b_moe_cute(weak, *args[1:])
