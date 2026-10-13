"""CPU checks for policy-preserving MXFP8 benchmark configuration loading."""

from __future__ import annotations

from contextlib import nullcontext
from importlib.abc import MetaPathFinder
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import torch

import helion
from helion.runtime.cute_structural_config import CuteStructuralConfig
from helion.runtime.cute_structural_policy import CuteStructuralPolicy

if TYPE_CHECKING:
    from collections.abc import Sequence
    from importlib.machinery import ModuleSpec


class _BlockExternalKernelLibrary(MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname == "helion_kernel_library" or fullname.startswith(
            "helion_kernel_library."
        ):
            raise ModuleNotFoundError("Kernel library is absent from standalone CI")
        return None


@pytest.fixture(autouse=True)
def block_external_kernel_library(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(sys.modules):
        if name == "helion_kernel_library" or name.startswith("helion_kernel_library."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(
        sys, "meta_path", [_BlockExternalKernelLibrary(), *sys.meta_path]
    )


def install_fake_campaign(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    """Only reproduce external module paths and a packing callable's interface."""
    gemm_path = root / "repro" / "baselines" / "gb300" / "gemm.py"
    gemm_path.parent.mkdir(parents=True)
    gemm_path.write_text("# fake campaign module\n")
    modules = {
        name: ModuleType(name)
        for name in (
            "repro",
            "repro.baselines",
            "repro.baselines.gb300",
            "repro.baselines.gb300.gemm",
        )
    }
    modules["repro"].__dict__["baselines"] = modules["repro.baselines"]
    modules["repro.baselines"].__dict__["gb300"] = modules["repro.baselines.gb300"]
    modules["repro.baselines.gb300"].__dict__["gemm"] = modules[
        "repro.baselines.gb300.gemm"
    ]
    modules["repro.baselines.gb300.gemm"].__dict__.update(
        __file__=str(gemm_path), pack_scales=Mock(side_effect=lambda tensor: tensor)
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    scripts = gemm_path.parents[2] / ".venv-gb300-main-20261012" / "bin"
    scripts.mkdir(parents=True)
    (scripts / "ninja").write_text("# fake executable marker\n")
    return scripts


def load_benchmark() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "benchmarks/cute/mxfp8_hillclimb.py"
    spec = importlib.util.spec_from_file_location("mxfp8_hillclimb_cpu", path)
    assert spec is not None and spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    return benchmark


@pytest.mark.parametrize("envelope_kind", ["winner", "result", "nested"])
def test_pinned_config_retains_policy(tmp_path: Path, envelope_kind: str) -> None:
    initialized = torch.cuda.is_initialized()
    benchmark = load_benchmark()
    expected = CuteStructuralConfig(
        helion.Config(block_sizes=[16, 128], cute_scaled_mma=True),
        CuteStructuralPolicy(cute_full_slice_matmul_tiling=True),
    )
    payload = expected.to_dict()
    if envelope_kind == "result":
        payload = {"config": dict(expected.config), "config_envelope": payload}
    elif envelope_kind == "nested":
        payload = {"config": payload}
    artifact = tmp_path / "winner.json"
    artifact.write_text(json.dumps(payload))
    loaded = benchmark.load_pinned_config(artifact)
    assert isinstance(loaded, CuteStructuralConfig)
    assert loaded.to_dict() == expected.to_dict()
    assert torch.cuda.is_initialized() is initialized


def test_campaign_tool_path_keeps_active_and_baseline_scripts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = load_benchmark()
    scripts = install_fake_campaign(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "active" / "bin" / "python"))
    monkeypatch.setenv("PATH", "/sentinel/bin")
    directories = benchmark.initialize_tool_path()
    assert directories[0] == str(Path(sys.executable).parent)
    assert (
        benchmark.os.environ["PATH"].split(benchmark.os.pathsep)[-1] == "/sentinel/bin"
    )
    assert str(scripts) in directories
    assert Path(scripts, "ninja").is_file(), (
        "Campaign dependency scripts must expose ninja"
    )


def test_unsupported_optional_flashinfer_baseline_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialized = torch.cuda.is_initialized()
    benchmark = load_benchmark()
    install_fake_campaign(monkeypatch, tmp_path)
    fake_flashinfer = ModuleType("flashinfer")
    fake_flashinfer.__dict__.update(
        mm_mxfp8=Mock(return_value=torch.empty((1, 8), dtype=torch.bfloat16))
    )
    fake_tuner = ModuleType("flashinfer.autotuner")
    fake_tuner.__dict__.update(
        AutoTuner=SimpleNamespace(
            get=lambda: SimpleNamespace(
                clear_cache=Mock(),
                load_configs=Mock(side_effect=AssertionError("No tactic cache exists")),
            )
        ),
        autotune=Mock(return_value=nullcontext()),
    )
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.autotuner", fake_tuner)
    tensors = (
        torch.empty((1, 32), dtype=torch.float8_e4m3fn),
        torch.empty((8, 32), dtype=torch.float8_e4m3fn),
        torch.full((1, 1), 127, dtype=torch.uint8),
        torch.full((8, 1), 127, dtype=torch.uint8),
    )
    cache = tmp_path / "unsupported.flashinfer.json"
    calls, metadata = benchmark.build_baselines(
        tensors,
        ["flashinfer-cute"],
        SimpleNamespace(mode="autotune", flashinfer_cache=cache),
    )
    assert calls == {}
    assert metadata["flashinfer-cute"]["status"] == "SKIP"
    assert "no supported CuTe MXFP8 tactic" in metadata["flashinfer-cute"]["reason"]
    assert not cache.exists()
    skip_receipt = cache.with_suffix(".skip.json")
    assert skip_receipt.is_file()
    verify_calls, verify_metadata = benchmark.build_baselines(
        tensors,
        ["flashinfer-cute"],
        SimpleNamespace(mode="verify", flashinfer_cache=cache),
    )
    assert verify_calls == {}
    assert verify_metadata["flashinfer-cute"]["status"] == "SKIP"
    assert (
        verify_metadata["flashinfer-cute"]["reason"]
        == metadata["flashinfer-cute"]["reason"]
    )
    for mismatched in (
        (
            torch.empty((2, 32), dtype=torch.float8_e4m3fn),
            tensors[1],
            torch.empty((2, 1), dtype=torch.uint8),
            tensors[3],
        ),
        (tensors[0], tensors[1], tensors[2].view(torch.float8_e8m0fnu), tensors[3]),
    ):
        with pytest.raises(RuntimeError, match="unsupported-tactic receipt"):
            benchmark.build_baselines(
                mismatched,
                ["flashinfer-cute"],
                SimpleNamespace(mode="verify", flashinfer_cache=cache),
            )
    assert torch.cuda.is_initialized() is initialized


def test_public_provider_backend_dispatch_is_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = load_benchmark()
    monkeypatch.setenv("HELION_BACKEND", "outer-backend")
    monkeypatch.setenv("HELION_AOT_MODE", "outer-mode")
    seen = []
    tensor = torch.empty((1,))

    def operation(value: torch.Tensor) -> torch.Tensor:
        seen.append(
            (
                benchmark.os.environ["HELION_BACKEND"],
                benchmark.os.environ["HELION_AOT_MODE"],
            )
        )
        return value

    triton_call = benchmark.public_library_call(operation, (tensor,), "triton")
    published_call = benchmark.public_library_call(operation, (tensor,), "cute")
    for call in (triton_call, published_call, triton_call, published_call):
        assert call() is tensor
        assert benchmark.os.environ["HELION_BACKEND"] == "outer-backend"
        assert benchmark.os.environ["HELION_AOT_MODE"] == "outer-mode"
    assert seen == [("triton", "evaluate"), ("cute", "evaluate")] * 2

    def failed_operation(value: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("provider failure")

    with pytest.raises(RuntimeError, match="provider failure"):
        benchmark.public_library_call(failed_operation, (tensor,), "cute")()
    assert benchmark.os.environ["HELION_BACKEND"] == "outer-backend"
    assert benchmark.os.environ["HELION_AOT_MODE"] == "outer-mode"
