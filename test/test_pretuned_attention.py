"""CPU checks for exact attention recipes; no CUDA allocation or compilation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from pretuned_kernels import run
from pretuned_kernels.attention import _helion_aot_attention_cuda_sm103 as heuristic
from pretuned_kernels.attention import attention
import pytest
import torch

import helion
from helion._hardware import HardwareInfo
from helion.autotuner import aot_cache
from helion.autotuner.aot_kernel import _flatten_key_value
from helion.autotuner.aot_kernel import make_aot_key
from helion.autotuner.search_space_logger import canonical_config_id

# The final FULL-search returns, in the immutable twelve-row matrix order.
EXPECTED = [
    ((1, 8, 12288, 128, "float16", False), "eba99c205cbf8ed0"),
    ((2, 32, 16384, 128, "float16", False), "eaef17b4e5af6f2d"),
    ((2, 32, 32768, 64, "float16", False), "5709d41b7951347a"),
    ((2, 32, 65536, 64, "float16", True), "b660aa50ecc2e55b"),
    ((2, 32, 2048, 64, "float16", False), "8df30045e26d568b"),
    ((2, 32, 4096, 64, "float16", True), "b29cf18a6dc5483a"),
    ((8, 32, 8192, 128, "bfloat16", False), "1fe8bac622ec2f94"),
    ((1, 1, 8192, 128, "float16", False), "8b336d6f6767cce7"),
    ((2, 16, 8192, 128, "float16", False), "0ed0c82334cf554a"),
    ((1, 8, 12288, 128, "float16", True), "ec9f8a9b85fcdd5d"),
    ((4, 32, 4096, 128, "bfloat16", False), "f092788bbe9c9935"),
    ((4, 32, 4096, 128, "bfloat16", True), "f77c17b1b03848e4"),
]


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple[int, ...] = (1, 8, 12288, 128)
    dtype: torch.dtype = torch.float16
    device: torch.device = field(default_factory=lambda: torch.device("cuda:0"))
    contiguous: bool = True

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def is_contiguous(self) -> bool:
        return self.contiguous


def _args(metadata: TensorMetadata) -> tuple[torch.Tensor, ...]:
    # Scope checks inspect metadata only. No FakeTensor CUDA initialization is
    # necessary to exercise these public-key contracts.
    return cast("tuple[torch.Tensor, ...]", (metadata, metadata, metadata))


@pytest.mark.parametrize(("shape", "config_id"), EXPECTED)
def test_exact_returned_config_and_nonaliasing(shape, config_id):
    *key, causal = shape
    select = (
        heuristic.autotune_causal_attention_output
        if causal
        else heuristic.autotune_attention_output
    )
    config = select(*key)
    assert canonical_config_id(helion.Config(**config)) == config_id
    config["block_sizes"][0] = 999
    config["cute_reduction_reloads"][0] = "changed"
    assert canonical_config_id(helion.Config(**select(*key))) == config_id


def test_complete_scope_and_backend():
    assert [shape for shape, _ in EXPECTED] == attention.SHAPES
    assert len(set(attention.SHAPES)) == 12
    assert set(heuristic._DENSE_KEYS) == {
        shape[:5] for shape, _ in EXPECTED if not shape[-1]
    }
    assert set(heuristic._CAUSAL_KEYS) == {
        shape[:5] for shape, _ in EXPECTED if shape[-1]
    }
    for kernel in (attention.attention_output, attention.causal_attention_output):
        assert kernel.settings.backend == "cute"
        assert kernel.settings.static_shapes is True
        assert kernel.settings.autotune_cache == "AOTAutotuneCache"
    assert attention.use_cudagraph() is False


@pytest.mark.parametrize("name", ("attention_output", "causal_attention_output"))
def test_frontend_body_matches_original(name):
    original = Path(__file__).resolve().parent.parent / "examples/attention.py"
    bodies = []
    for path in (original, Path(attention.__file__)):
        tree = ast.parse(path.read_text())
        fn = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        bodies.append(ast.dump(ast.Module(body=fn.body, type_ignores=[])))
    assert bodies[0] == bodies[1]


@pytest.mark.parametrize(("shape", "_config_id"), EXPECTED)
def test_aot_user_key_and_scalar_heuristic_contract(monkeypatch, shape, _config_id):
    batch, heads, seq, dim, dtype_name, causal = shape
    metadata = TensorMetadata(
        shape=(batch, heads, seq, dim),
        dtype={"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name],
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 3))
    user_key = attention._causal_key if causal else attention._dense_key
    key = user_key(*_args(metadata))
    assert key == shape[:5]
    assert _flatten_key_value(key) == list(shape[:5])
    name = "causal_attention_output" if causal else "attention_output"
    selector = (
        heuristic.key_causal_attention_output
        if causal
        else heuristic.key_attention_output
    )
    runtime_key = make_aot_key(attention.__file__, name, user_key=user_key)
    monkeypatch.setattr(runtime_key, "_load_key_function", lambda: selector)
    assert runtime_key(*_args(metadata)) == selector(*key)


@pytest.mark.parametrize(
    ("index", "replacement", "message"),
    [
        (0, TensorMetadata(shape=(8, 12288, 128)), "four-dimensional"),
        (1, TensorMetadata(shape=(1, 8, 8192, 128)), "equal Q/K/V shapes"),
        (0, TensorMetadata(dtype=torch.float32), "float16 and bfloat16"),
        (2, TensorMetadata(dtype=torch.bfloat16), "matching Q/K/V dtypes"),
        (1, TensorMetadata(device=torch.device("cuda:1")), "same CUDA device"),
        (0, TensorMetadata(device=torch.device("cpu")), "same CUDA device"),
        (2, TensorMetadata(contiguous=False), "contiguous BHND"),
    ],
)
def test_scope_rejected_even_without_heuristic(
    monkeypatch, index, replacement, message
):
    def no_device_query(device):
        pytest.fail("Invalid metadata should be rejected before CUDA inspection")

    monkeypatch.setattr(torch.cuda, "get_device_capability", no_device_query)
    runtime_key = make_aot_key(
        attention.__file__, "attention_output", user_key=attention._dense_key
    )
    monkeypatch.setattr(runtime_key, "_load_key_function", lambda: None)
    args = list(_args(TensorMetadata()))
    args[index] = cast("torch.Tensor", replacement)
    with pytest.raises(ValueError, match=message):
        runtime_key(*args)


@pytest.mark.parametrize("capability", [(9, 0), (10, 0), (12, 0)])
def test_unsupported_hardware(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    with pytest.raises(ValueError, match="sm103 only"):
        attention._dense_key(*_args(TensorMetadata()))


@pytest.mark.parametrize("causal", [False, True])
def test_no_nearest_shape_or_dtype_fallback(causal):
    select = (
        heuristic.autotune_causal_attention_output
        if causal
        else heuristic.autotune_attention_output
    )
    for key in [(1, 8, 12289, 128, "float16"), (1, 8, 12288, 128, "bfloat16")]:
        with pytest.raises(ValueError, match="No pretuned"):
            select(*key)
    with pytest.raises(ValueError, match="No pretuned"):
        attention._input_key(
            *_args(TensorMetadata(shape=(1, 8, 12289, 128))), causal=causal
        )


def test_dense_and_causal_are_distinct():
    key = (1, 8, 12288, 128, "float16")
    assert heuristic.autotune_attention_output(
        *key
    ) != heuristic.autotune_causal_attention_output(*key)


def test_sm103_discovery_and_runner_scope(monkeypatch):
    monkeypatch.delenv("HELION_HEURISTIC_DIR", raising=False)
    for compute, expected in [("sm103", True), ("sm100", False), ("sm120", False)]:
        hardware = HardwareInfo("cuda", "test", "13.0", compute)
        monkeypatch.setattr(
            aot_cache, "get_hardware_info", lambda hardware=hardware: hardware
        )
        monkeypatch.setattr(aot_cache, "_heuristic_file_cache", {})
        found = aot_cache.find_heuristic_file(attention.__file__, "attention_output")
        assert (found is not None) is expected
        if found is not None:
            assert found.name == "_helion_aot_attention_cuda_sm103.py"
    assert "attention" in run.KERNELS
    assert run._supported_hardware("attention") == {"sm103"}
    assert "skipped" in run.run_kernel("attention", "b200")


def test_main_checks_all_shapes_before_benchmark(monkeypatch):
    checks = []
    monkeypatch.setattr(attention.sys, "path", attention.sys.path.copy())
    monkeypatch.setattr(
        torch,
        "Generator",
        lambda **kwargs: SimpleNamespace(manual_seed=lambda seed: None),
    )
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: None)
    monkeypatch.setattr(attention, "attention_output", lambda *args: "actual")
    monkeypatch.setattr(attention, "causal_attention_output", lambda *args: "actual")
    monkeypatch.setattr(attention, "_dense_reference", lambda *args: "expected")
    monkeypatch.setattr(attention, "_causal_reference", lambda *args: "expected")
    monkeypatch.setattr(
        attention,
        "_check_attention_accuracy",
        lambda a, b: checks.append((a, b)),
    )

    def sweep(shapes, make_calls, **kwargs):
        assert kwargs["use_cudagraph"] is False
        for index, shape in enumerate(shapes):
            call, baselines, _ = make_calls(shape)
            assert len(checks) == index + 1
            assert call() == "actual"
            assert baselines[0][0] == "cuDNN SDPA"
        return {"total": len(shapes)}

    monkeypatch.setitem(
        attention.sys.modules, "_bench", SimpleNamespace(run_sweep=sweep)
    )
    assert attention.main(verbose=False) == {"total": 12}
    assert checks == [("actual", "expected")] * 12


@pytest.mark.parametrize(
    ("dtype", "accepted", "rejected"),
    [
        (torch.float16, [0.0005, 1.0009765625], [0.002, 1.00390625]),
        (torch.bfloat16, [0.0025, 1.0078125], [0.01, 1.0234375]),
    ],
)
def test_accuracy_checks_absolute_and_relative_error(dtype, accepted, rejected):
    expected = torch.tensor([0.0, 1.0], dtype=dtype)
    attention._check_attention_accuracy(torch.tensor(accepted, dtype=dtype), expected)
    for index in range(2):
        actual = expected.clone()
        actual[index] = rejected[index]
        with pytest.raises(AssertionError):
            attention._check_attention_accuracy(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_accuracy_rejects_nonfinite_outputs(dtype, value):
    bad = torch.tensor([value], dtype=dtype)
    good = torch.zeros_like(bad)
    for actual, expected in ((bad, good), (good, bad), (bad, bad)):
        with pytest.raises(AssertionError, match="must be finite"):
            attention._check_attention_accuracy(actual, expected)
