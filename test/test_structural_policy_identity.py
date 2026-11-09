"""Pins every persisted identity string of the CuTe structural policy.

``test/data/cute_structural_policy_identity.json`` was generated at abce7886a,
before the backend-neutral :mod:`helion.runtime.structural_policy` protocol
existed.  Cached ``.best_config`` files, AOT heuristic modules, measurement
rows and disk-cache keys written by that version must read back and be
reproduced byte-for-byte by every later version; a diff here is a cache
identity change, not a refactoring detail.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

import helion
from helion._compiler.backend_registry import get_backend_class
from helion._compiler.cute import _codegen_modules
from helion.autotuner.aot_structural_policy import attach_model_policy
from helion.autotuner.aot_structural_policy import measurement_config_hash
from helion.autotuner.aot_structural_policy import measurement_metadata
from helion.autotuner.aot_structural_policy import measurement_policy
from helion.autotuner.aot_structural_policy import model_configs
from helion.autotuner.aot_structural_policy import policy_suffix
from helion.autotuner.metrics import KernelMetadata
from helion.runtime.cute.structural_config import CuteStructuralConfig
from helion.runtime.cute.structural_config import decode_structural_policy
from helion.runtime.cute.structural_policy import CUTE_STRUCTURAL_SETTINGS
from helion.runtime.cute.structural_policy import CuteStructuralPolicy
from helion.runtime.kernel import BoundKernel
from helion.runtime.kernel import Kernel
from helion.runtime.settings import Settings
from helion.runtime.structural_policy import StructuralConfig

FIXTURE = Path(__file__).parent / "data" / "cute_structural_policy_identity.json"


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE.read_text())


def _policies() -> dict[str, CuteStructuralPolicy]:
    return {
        "default": CuteStructuralPolicy(),
        "full": CuteStructuralPolicy(**dict.fromkeys(CUTE_STRUCTURAL_SETTINGS, True)),
        "mixed": CuteStructuralPolicy(
            cute_region_fission=True, cute_flatten_nested_reductions=True
        ),
    }


def test_policy_envelopes_and_digests_are_frozen() -> None:
    fixture = _fixture()
    assert list(CUTE_STRUCTURAL_SETTINGS) == fixture["settings"]
    expected = fixture["policies"]
    for name, policy in _policies().items():
        assert policy.to_dict() == expected[name]["dict"], name
        assert policy.to_json() == expected[name]["json"], name
        assert policy.identity() == expected[name]["identity"], name
        # The persisted envelope reads back to an equal policy (both decoders).
        assert CuteStructuralPolicy.from_dict(expected[name]["dict"]) == policy
        assert decode_structural_policy(json.loads(expected[name]["json"])) == policy


def test_config_envelope_is_frozen() -> None:
    fixture = _fixture()
    config = helion.Config(**fixture["config"])
    envelope = CuteStructuralConfig(config, _policies()["mixed"])
    expected = fixture["envelope"]
    assert envelope.to_json() == expected["json"]
    assert envelope.identity() == expected["identity"]
    assert repr(envelope) == expected["repr"]
    assert envelope.to_dict() == expected["dict"]
    restored = CuteStructuralConfig.from_json(expected["json"])
    assert restored == envelope
    assert dict(restored.config) == dict(config)
    assert isinstance(envelope, StructuralConfig)


def test_aot_measurement_and_model_identities_are_frozen() -> None:
    fixture = _fixture()
    policy = _policies()["mixed"]
    config = helion.Config(**fixture["config"])
    metadata = measurement_metadata(policy)
    assert metadata == fixture["measurement_metadata"]
    assert list(metadata) == [
        "aot_policy_version",
        "cute_structural_policy",
        "cute_structural_policy_id",
    ]
    assert measurement_policy(metadata) == policy
    assert measurement_config_hash(config, policy) == fixture["measurement_config_hash"]
    assert (
        measurement_config_hash(config, None)
        == fixture["measurement_config_hash_no_policy"]
    )
    assert policy_suffix(policy) == fixture["policy_suffix"]
    model = attach_model_policy(fixture["model_raw"], {"k": [config]}, policy)
    assert model == fixture["model"]
    # The frozen model source still loads through the current reader.
    namespace: dict[str, object] = {}
    exec(compile(model, "<model>", "exec"), namespace)
    (recorded,) = model_configs(SimpleNamespace(**namespace), "k", policy)
    assert recorded == CuteStructuralConfig(config, policy)
    assert namespace["autotune_k"](torch.empty(64)) == recorded


def test_kernel_metadata_and_cache_keys_are_frozen() -> None:
    fixture = _fixture()
    policy = _policies()["mixed"]
    metadata = KernelMetadata(
        kernel_name="k",
        kernel_source="def k(x): return x",
        input_shapes="[(4,)]",
        dtypes="['torch.float32']",
        hardware="cpu",
        settings={"backend": "cute"},
        cute_structural_policy=policy,
    )
    assert metadata.run_id == fixture["kernel_metadata_run_id"]
    assert metadata.to_dict() == fixture["kernel_metadata_dict"]
    # Disk-cache and specialization-key entries (kernel.py).
    bound = SimpleNamespace(kernel=SimpleNamespace(cute_structural_policy=policy))
    assert (
        BoundKernel.extra_cache_key(bound)  # type: ignore[arg-type]
        == f"cute_structural_policy:{policy.identity()}"
    )

    def fn(x: torch.Tensor) -> torch.Tensor:
        return x

    kernel = Kernel(
        fn,
        settings=Settings(backend="cute", autotune_effort="none"),
        cute_structural_policy=policy,
    )
    key = kernel._base_specialization_key((torch.empty(4),))
    assert key[0] == ("cute_structural_policy", policy.identity())


def test_structural_policy_type_hook_names_the_cute_policy() -> None:
    assert get_backend_class("cute")().structural_policy_type() is CuteStructuralPolicy
    assert get_backend_class("triton")().structural_policy_type() is None
    settings = Settings(backend="cute", cute_region_fission=True)
    assert (
        CuteStructuralPolicy.from_settings(settings)
        == settings.get_cute_structural_policy()
        == CuteStructuralPolicy(cute_region_fission=True)
    )


def test_runtime_hooks_module_imports_first_and_is_not_a_codegen_module() -> None:
    # ``runtime_hooks`` imports ``helion.runtime.kernel`` at module level, so
    # it must stay out of helion's own import chain: importable as the very
    # first helion import of a fresh interpreter ...
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(root))
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import importlib; "
                "hooks = importlib.import_module('helion._compiler.cute.runtime_hooks'); "
                "kernel = importlib.import_module('helion.runtime.kernel'); "
                "assert hooks.PreparedMetadataSpecializationExtractor "
                "is kernel.PreparedMetadataSpecializationExtractor"
            ),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # ... and never reloaded by the codegen-module registry (a reload would
    # fork the helper classes the tests patch and kernel.py isinstance-checks).
    assert "runtime_hooks" not in vars(_codegen_modules)
