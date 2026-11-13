from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_codegen import tcgen05_target as _target
from helion._testing.cute_fragment_layout import _fragment_singleton_scan

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("static_shapes", [False, True])
@pytest.mark.parametrize("steps", [0, 2])
def test_fragment_singleton_scan_codegen_and_rebind(static_shapes, steps):
    kernel = helion.kernel(
        _fragment_singleton_scan.fn,
        backend="cute",
        static_shapes=static_shapes,
        autotune_effort="none",
        disable_autotuner_heuristics=True,
    )
    shapes = [(3, 5, 7), (3, 9, 3), (3, 1, 4)]
    inputs = [
        torch.randn(shape).transpose(-2, -1).contiguous().transpose(-2, -1)
        for shape in shapes
    ]
    with (
        mock_cuda_unavailable(),
        _target(),
        forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 0)),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 0),
        ),
        patch("helion.runtime.get_num_sm", return_value=148),
        patch(
            "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bounds = [kernel.bind((value, steps)) for value in inputs]
        assert len({id(bound) for bound in bounds}) == len(inputs)
        # The canonical third input publishes layout guards and invalidates
        # bindings created before that schema was complete. Rebind every input
        # under the final schema before checking reuse by metadata-equivalent clones.
        bounds = [kernel.bind((value, steps)) for value in inputs]
        assert len({id(bound) for bound in bounds}) == len(inputs)
        for value, bound in zip(inputs, bounds, strict=True):
            assert (
                kernel.bind((value.clone(memory_format=torch.preserve_format), steps))
                is bound
            )
            if not static_shapes:
                assert (
                    "input_tensor_metadata"
                    in bound.env.compiler_fact_specialization_facts
                )
            code = bound.to_code(bound.config_spec.default_config())
            assert "fragment_index" in code and "SmemAllocator" in code
        with ExitStack() as stack:
            for index, bound in enumerate(bounds):
                stack.enter_context(patch.object(bound, "_run", return_value=index))
                stack.enter_context(patch.object(bound, "_prepare_direct_call"))
            for index, value in enumerate(inputs):
                assert bounds[0](value, steps) == index
