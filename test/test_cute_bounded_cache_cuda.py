from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path

from examples.aot_example import row_softmax
import pytest
import torch
from torch._inductor.codecache import PyCodeCache

import helion
from helion._testing import skipUnlessBackends
from helion.autotuner.benchmarking import _make_cudagraph_replay
from helion.runtime import default_cute_launcher

pytestmark = skipUnlessBackends(["cute"])


@dataclass(frozen=True)
class CacheCase:
    name: str
    shape: tuple[int, int]
    dtype: torch.dtype
    static: bool
    width: int
    threads: int
    vector: int
    rows: int = 1
    layout: str = "dense"
    pattern: str = "finite"
    fast: bool = True


CASES = (
    CacheCase("original0", (256, 256), torch.bfloat16, False, 256, 64, 2),
    CacheCase("original1", (1024, 1024), torch.bfloat16, False, 1024, 128, 4),
    CacheCase("original2", (4096, 4096), torch.bfloat16, False, 4096, 128, 8),
    CacheCase("fp16-dynamic-tail", (17, 513), torch.float16, False, 4096, 128, 8),
    CacheCase("fp16-static-tail", (17, 513), torch.float16, True, 4096, 128, 8),
    CacheCase("bf16-static-tail", (17, 513), torch.bfloat16, True, 4096, 128, 8),
    CacheCase("fp32-dynamic-tail", (17, 513), torch.float32, False, 4096, 128, 4),
    CacheCase("fp32-static-tail", (17, 513), torch.float32, True, 4096, 128, 4),
    CacheCase("rows2", (17, 509), torch.float16, False, 1024, 64, 8, rows=2),
    CacheCase("rows4", (17, 513), torch.bfloat16, False, 1024, 32, 8, rows=4),
    CacheCase(
        "transpose", (17, 255), torch.float16, False, 512, 32, 4, layout="transpose"
    ),
    CacheCase(
        "column-slice", (17, 513), torch.bfloat16, False, 1024, 64, 8, layout="slice"
    ),
    CacheCase(
        "unaligned", (17, 513), torch.float32, False, 4096, 128, 4, layout="unaligned"
    ),
    CacheCase(
        "padded", (17, 1007), torch.bfloat16, False, 1024, 64, 8, layout="padded"
    ),
    CacheCase(
        "odd-stride", (17, 513), torch.bfloat16, False, 1024, 64, 8, layout="odd"
    ),
    CacheCase(
        "bf16-nonfinite",
        (17, 513),
        torch.bfloat16,
        False,
        4096,
        128,
        8,
        pattern="nonfinite",
    ),
    CacheCase(
        "fp32-nonfinite",
        (17, 513),
        torch.float32,
        False,
        4096,
        128,
        4,
        pattern="nonfinite",
    ),
    CacheCase("zero-columns", (5, 0), torch.bfloat16, False, 4096, 128, 8, fast=False),
    CacheCase("oversized", (5, 4097), torch.bfloat16, False, 4096, 128, 8, fast=False),
    CacheCase(
        "first-tile-negative-infinity",
        (5, 8192),
        torch.float32,
        False,
        4096,
        128,
        4,
        pattern="first-inf",
        fast=False,
    ),
    CacheCase(
        "later-tile-negative-infinity",
        (5, 8192),
        torch.float32,
        False,
        4096,
        128,
        4,
        pattern="later-inf",
        fast=False,
    ),
)


def case_stride(case: CacheCase) -> tuple[int, int]:
    rows, columns = case.shape
    if case.layout == "transpose":
        return 1, rows
    if case.layout == "slice":
        return 2 * columns, 2
    if case.layout == "padded":
        return ((columns + 7) // 8 + 2) * 8, 1
    if case.layout == "odd":
        return columns + 2, 1
    return max(columns, 1), 1


class GuardedTensor:
    def __init__(self, shape, stride, dtype, device, *, residue=0):
        span = (
            1 + sum((size - 1) * step for size, step in zip(shape, stride, strict=True))
            if all(shape)
            else 0
        )
        self.offset = 128 + residue
        self.storage = torch.full(
            (self.offset + span + 128,), -17.25, dtype=dtype, device=device
        )
        self.tensor = self.storage.as_strided(shape, stride, self.offset)
        self.active = torch.zeros_like(self.storage, dtype=torch.bool)
        self.active.as_strided(shape, stride, self.offset).fill_(True)

    def poison(self):
        self.storage.fill_(-17.25)

    def check_holes(self):
        assert torch.all(self.storage[~self.active] == -17.25).item()


def case_config(case: CacheCase, mode="bounded_layout"):
    return helion.Config(
        block_sizes=[case.rows, case.width],
        num_threads=[case.rows if case.rows > 1 else 0, case.threads],
        cute_vector_widths=[1, case.vector],
        cute_lane_layouts=["blocked", "strided"],
        cute_reduction_sequence=mode,
        cute_cluster_n=1,
    )


def case_kernel(case: CacheCase, *, fast_math: bool = False):
    return helion.kernel(
        row_softmax.fn,
        backend="cute",
        static_shapes=case.static,
        autotune_effort="none",
        fast_math=fast_math,
        cute_full_slice_matmul_tiling=True,
        cute_segmented_matmul_tiling=True,
        cute_flatten_nested_reductions=True,
    )


def case_binding_input(case: CacheCase, tensor: torch.Tensor) -> torch.Tensor:
    if case.shape[1] != 0:
        return tensor
    # The unchanged frontend rejects amax on an empty tracing example. Test
    # the generated dynamic host's existing zero-trip fallback from a positive
    # binding, as opposed to changing that frontend admission contract.
    return torch.empty(
        (case.shape[0], case.threads * case.vector + 1),
        dtype=case.dtype,
        device=tensor.device,
    )


def _fill(tensor, case, generation):
    generator = torch.Generator(device=tensor.device).manual_seed(20260913 + generation)
    tensor.copy_(
        torch.randn(
            case.shape, dtype=case.dtype, device=tensor.device, generator=generator
        )
    )
    if case.pattern == "nonfinite":
        tensor[0].fill_(float("-inf"))
        tensor[1].fill_(float("inf"))
        tensor[2, 0] = float("nan")
        tensor[3].zero_()
        tensor[4].fill_(-0.0)
        tensor[5, 0] = float("inf")
        tensor[6, 0] = float("-inf")
    elif case.pattern == "first-inf":
        tensor[:, : case.width] = float("-inf")
    elif case.pattern == "later-inf":
        tensor[:, case.width :] = float("-inf")


def _check_exact(actual, expected):
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    bits = torch.int32 if actual.dtype == torch.float32 else torch.int16
    keep = ~torch.isnan(expected)
    # Also compares infinities and the sign of zero; NaN payloads are not an
    # observable promise of the original math helpers.
    assert torch.equal(
        actual.contiguous().view(bits)[keep], expected.contiguous().view(bits)[keep]
    )


def _load(bound, source):
    module = PyCodeCache.load(source)
    bound.env.backend.annotate_compiled_module(module, source, "row_softmax")
    return module


def run_guarded_cache_case(case: CacheCase, *, fast_math: bool = False):
    retained = []
    kernel = case_kernel(case, fast_math=fast_math)
    checks = []
    for replacement in range(2):
        inputs = GuardedTensor(
            case.shape,
            case_stride(case),
            case.dtype,
            "cuda",
            residue=int(case.layout == "unaligned"),
        )
        _fill(inputs.tensor, case, replacement * 3)
        bound = kernel.bind((case_binding_input(case, inputs.tensor),))
        config = case_config(case)
        source = bound.to_code(config)
        original = bound.to_code(case_config(case, "scalar"))
        module = _load(bound, source)
        control = _load(bound, original)
        functions = {
            node.name: node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
        }
        original_functions = {
            node.name: node
            for node in ast.parse(original).body
            if isinstance(node, ast.FunctionDef)
        }
        assert ast.dump(functions["_helion_row_softmax"]) == ast.dump(
            original_functions["_helion_row_softmax"]
        )
        if case.fast:
            assert "_helion_row_softmax_bounded" in functions
        output = torch.empty_like(inputs.tensor)
        outputs = GuardedTensor(case.shape, tuple(output.stride()), case.dtype, "cuda")
        retained.append((inputs, outputs))
        if replacement and inputs.tensor.numel():
            assert inputs.tensor.data_ptr() != retained[0][0].tensor.data_ptr()
            assert outputs.tensor.data_ptr() != retained[0][1].tensor.data_ptr()
        selected = []

        def protected(
            implementation,
            grid,
            *arguments,
            inputs=inputs,
            outputs=outputs,
            selected=selected,
            **keywords,
        ):
            assert arguments[0] is inputs.tensor
            assert isinstance(arguments[1], torch.Tensor)
            assert tuple(arguments[1].shape) == case.shape
            assert tuple(arguments[1].stride()) == tuple(outputs.tensor.stride())
            selected.append(implementation.__name__)
            return default_cute_launcher(
                implementation,
                grid,
                arguments[0],
                outputs.tensor,
                *arguments[2:],
                **keywords,
            )

        def invoke(module=module, inputs=inputs, outputs=outputs, protected=protected):
            module.row_softmax(inputs.tensor, _launcher=protected)
            return outputs.tensor

        invoke()
        _check_exact(outputs.tensor, control.row_softmax(inputs.tensor))
        inputs.check_holes()
        outputs.check_holes()
        replay = _make_cudagraph_replay(invoke)
        for mutation in range(3):
            _fill(inputs.tensor, case, replacement * 3 + mutation)
            expected = control.row_softmax(inputs.tensor)
            snapshot = inputs.storage.clone()
            outputs.poison()
            replay()
            torch.cuda.synchronize()
            _check_exact(outputs.tensor, expected)
            assert torch.equal(
                inputs.storage.view(torch.uint8), snapshot.view(torch.uint8)
            )
            inputs.check_holes()
            outputs.check_holes()
            if case.pattern == "finite" and inputs.tensor.numel():
                # The unchanged study tolerance remains an additional reference
                # check; the scalar generated control above is byte exact.
                torch.testing.assert_close(
                    outputs.tensor,
                    inputs.tensor.float().softmax(dim=1).to(case.dtype),
                    rtol=0.01,
                    atol=0.1,
                )
            checks.append(
                {
                    "replacement": replacement,
                    "mutation": mutation,
                    "exact": True,
                    "holes": True,
                }
            )
        expected_name = (
            "_helion_row_softmax_bounded" if case.fast else "_helion_row_softmax"
        )
        assert set(selected) == {expected_name}
        if artifact_root := os.environ.get("HELION_BOUNDED_CACHE_ARTIFACT_DIR"):
            artifact = Path(artifact_root) / case.name
            artifact.mkdir(parents=True, exist_ok=True)
            (artifact / "generated.py").write_text(source)
            (artifact / "control.py").write_text(original)
            (artifact / "config.json").write_text(
                json.dumps(dict(bound.config_spec.normalized_config(config)), indent=2)
                + "\n"
            )
            (artifact / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_guarded_cache_exact_outputs_holes_and_mutated_graphs(case: CacheCase):
    run_guarded_cache_case(case)
