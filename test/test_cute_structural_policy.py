from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import os
import pickle
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion.autotuner.local_cache import _search_policy_json_value
from helion.autotuner.local_cache import _UncacheableSearchPolicy
import helion.language as hl
from helion.runtime.cute_structural_policy import CUTE_STRUCTURAL_SETTINGS
from helion.runtime.cute_structural_policy import CuteStructuralOrigins
from helion.runtime.cute_structural_policy import CuteStructuralPolicy
from helion.runtime.settings import Settings


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch):
    for name in CUTE_STRUCTURAL_SETTINGS:
        monkeypatch.delenv(f"HELION_{name.upper()}", raising=False)
    monkeypatch.delenv("HELION_BACKEND", raising=False)
    monkeypatch.setenv("HELION_AUTOTUNE_RANDOM_SEED", "0")
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield


@pytest.mark.parametrize("name", CUTE_STRUCTURAL_SETTINGS)
@pytest.mark.parametrize("raw", [None, "", "  ", "0", "1", " off ", " YeS "])
@pytest.mark.parametrize("override", [None, False, True])
def test_effective_values_and_origins(name, raw, override, monkeypatch):
    if raw is not None:
        monkeypatch.setenv(f"HELION_{name.upper()}", raw)
    kwargs = {} if override is None else {name: override}
    settings = Settings(backend="cute", **kwargs)
    if override is not None:
        expected, origin = override, "explicit"
    elif raw is None or not raw.strip():
        expected, origin = False, "default"
    else:
        expected, origin = raw.strip().lower() in {"1", "yes"}, "environment"
    assert getattr(settings, name) is expected
    assert settings.to_dict()[name] is expected
    assert getattr(settings.get_cute_structural_policy(), name) is expected
    assert getattr(settings.cute_structural_origins, name) == origin
    for other in CUTE_STRUCTURAL_SETTINGS:
        if other != name:
            assert getattr(settings, other) is False
            assert getattr(settings.cute_structural_origins, other) == "default"


@pytest.mark.parametrize("name", CUTE_STRUCTURAL_SETTINGS)
def test_invalid_environment_and_explicit_precedence(name, monkeypatch):
    monkeypatch.setenv(f"HELION_{name.upper()}", "not-a-bool")
    with pytest.raises(ValueError, match=f"HELION_{name.upper()}"):
        Settings(backend="cute")
    for value in (False, True):
        settings = Settings(backend="cute", **{name: value})
        assert getattr(settings, name) is value
        assert getattr(settings.cute_structural_origins, name) == "explicit"


@pytest.mark.parametrize("name", CUTE_STRUCTURAL_SETTINGS)
def test_value_and_origin_use_one_environment_read(name, monkeypatch):
    key = f"HELION_{name.upper()}"
    monkeypatch.setenv(key, "1")
    real_get = os.environ.get
    reads = []

    def changing_get(var_name, default=None):
        value = real_get(var_name, default)
        if var_name == key:
            reads.append(value)
            os.environ.pop(key, None)
        return value

    monkeypatch.setattr(os.environ, "get", changing_get)
    settings = Settings(backend="cute")
    assert reads == ["1"]
    assert getattr(settings, name) is True
    assert getattr(settings.cute_structural_origins, name) == "environment"
    snapshot = settings.get_cute_structural_policy()
    os.environ[key] = "0"
    assert settings.get_cute_structural_policy() == snapshot
    assert settings.copy().get_cute_structural_policy() == snapshot
    assert reads == ["1"]


@pytest.mark.parametrize("backend", ["cute", "triton", "pallas", "metal"])
@pytest.mark.parametrize("name", CUTE_STRUCTURAL_SETTINGS)
@pytest.mark.parametrize("raw", [None, "0", "1"])
def test_direct_same_value_and_changed_writes_are_explicit(
    backend, name, raw, monkeypatch
):
    if raw is not None:
        monkeypatch.setenv(f"HELION_{name.upper()}", raw)
    settings = Settings(backend=backend)
    previous = settings.cute_structural_origins
    policy = settings.get_cute_structural_policy()
    setattr(settings, name, getattr(settings, name))
    assert getattr(settings.cute_structural_origins, name) == "explicit"
    assert getattr(previous, name) != "explicit"
    assert settings.get_cute_structural_policy() == policy
    assert settings.get_cute_structural_policy().identity() == policy.identity()
    setattr(settings, name, not getattr(settings, name))
    assert getattr(settings.cute_structural_origins, name) == "explicit"
    assert getattr(policy, name) is not getattr(settings, name)
    assert settings.get_cute_structural_policy().identity() != policy.identity()


@pytest.mark.parametrize("make_copy", [copy.copy, copy.deepcopy, Settings.copy])
def test_provenance_preserving_copies_capture_environment(make_copy, monkeypatch):
    monkeypatch.setenv("HELION_CUTE_REGION_FISSION", "1")
    monkeypatch.setenv("HELION_CUTE_SEGMENTED_MATMUL_TILING", "0")
    original = Settings(backend="cute", cute_full_slice_matmul_tiling=False)
    monkeypatch.setenv("HELION_CUTE_REGION_FISSION", "0")
    monkeypatch.setenv("HELION_CUTE_SEGMENTED_MATMUL_TILING", "1")
    copied = make_copy(original)
    assert copied is not original
    assert copied == original
    assert copied.cute_structural_origins == original.cute_structural_origins
    assert copied.get_cute_structural_policy() == original.get_cute_structural_policy()
    copied.cute_region_fission = False
    assert original.cute_region_fission is True
    assert original.cute_structural_origins.cute_region_fission == "environment"
    assert copied.cute_structural_origins.cute_region_fission == "explicit"


@pytest.mark.parametrize("name", CUTE_STRUCTURAL_SETTINGS)
@pytest.mark.parametrize("enabled", [False, True])
def test_copy_overrides_preserve_other_origins(name, enabled, monkeypatch):
    monkeypatch.setenv(f"HELION_{name.upper()}", str(int(enabled)))
    original = Settings(backend="cute")
    for changed in (enabled, not enabled):
        copied = original.copy(**{name: changed, "static_shapes": False})
        assert getattr(copied, name) is changed
        assert copied.static_shapes is False
        assert original.static_shapes is True
        assert getattr(copied.cute_structural_origins, name) == "explicit"
        assert getattr(original.cute_structural_origins, name) == "environment"
        for other in CUTE_STRUCTURAL_SETTINGS:
            if other != name:
                assert getattr(copied.cute_structural_origins, other) == "default"
    copied = Settings(backend="cute").copy(**{name: False})
    assert getattr(copied.cute_structural_origins, name) == "explicit"


def test_default_false_copy_override_is_explicit():
    original = Settings(backend="cute")
    copied = original.copy(cute_region_fission=False)
    assert original.cute_structural_origins.cute_region_fission == "default"
    assert copied.cute_structural_origins.cute_region_fission == "explicit"
    assert original == copied
    assert original.get_cute_structural_policy() == copied.get_cute_structural_policy()


@pytest.mark.parametrize(
    "make_copy",
    [dataclasses.replace, lambda settings: Settings(**settings.to_dict())],
)
def test_flattened_reconstruction_is_explicit(make_copy, monkeypatch):
    monkeypatch.setenv("HELION_CUTE_REGION_FISSION", "1")
    original = Settings(backend="cute")
    for name in CUTE_STRUCTURAL_SETTINGS:
        monkeypatch.setenv(
            f"HELION_{name.upper()}", str(int(not getattr(original, name)))
        )
    copied = make_copy(original)
    assert copied == original
    assert copied.get_cute_structural_policy() == original.get_cute_structural_policy()
    assert all(
        getattr(copied.cute_structural_origins, name) == "explicit"
        for name in CUTE_STRUCTURAL_SETTINGS
    )
    overridden = dataclasses.replace(original, cute_region_fission=False)
    assert overridden.cute_region_fission is False
    assert overridden.cute_structural_origins.cute_region_fission == "explicit"


def test_ordinary_fields_copy_depth_and_validation_are_preserved():
    original = Settings(backend="triton", autotune_search_acf=["one"])
    for make_copy in (copy.copy, Settings.copy, dataclasses.replace):
        copied = make_copy(original)
        assert copied.autotune_search_acf is original.autotune_search_acf
    deep = copy.deepcopy(original)
    assert deep.autotune_search_acf == original.autotune_search_acf
    assert deep.autotune_search_acf is not original.autotune_search_acf
    origins = original.cute_structural_origins
    original.fast_math = True
    original.autotune_compile_timeout = 17
    assert original.cute_structural_origins is origins
    with pytest.raises(ValueError, match="autotune_best_of_k"):
        original.copy(autotune_best_of_k=0)
    with pytest.raises(TypeError, match="unexpected keyword"):
        original.copy(unknown_setting=True)
    with pytest.raises(TypeError, match="unexpected keyword"):
        original.copy(_cute_structural_origins=origins)


@pytest.mark.parametrize("protocol", [2, 4, pickle.HIGHEST_PROTOCOL])
def test_pickle_and_copy_restore_origins_without_user_writes(protocol, monkeypatch):
    monkeypatch.setenv("HELION_CUTE_REGION_FISSION", "1")
    original = Settings(backend="cute", cute_segmented_matmul_tiling=False)
    original.extra_state = {"self": original}
    restored = pickle.loads(pickle.dumps(original, protocol=protocol))
    assert restored == original
    assert restored.cute_structural_origins == original.cute_structural_origins
    assert restored.extra_state["self"] is restored
    deep = copy.deepcopy(original)
    assert deep.extra_state["self"] is deep
    assert deep.cute_structural_origins == original.cute_structural_origins


def test_legacy_effective_only_serialized_state_is_explicit():
    original = Settings(backend="cute", cute_region_fission=True)
    namespace, slots = original.__reduce_ex__(4)[2]
    assert "_cute_structural_origins" in namespace
    restored = Settings.__new__(Settings)
    restored.__setstate__((None, slots))
    assert restored == original
    assert all(
        getattr(restored.cute_structural_origins, name) == "explicit"
        for name in CUTE_STRUCTURAL_SETTINGS
    )
    restored.cute_region_fission = False
    assert original.cute_region_fission is True


def test_origin_metadata_does_not_change_public_fields_or_cache_serialization():
    implicit = Settings(backend="triton")
    explicit = Settings(**implicit.to_dict())
    assert implicit.cute_structural_origins != explicit.cute_structural_origins
    assert implicit == explicit
    assert repr(implicit) == repr(explicit)
    assert dataclasses.asdict(implicit) == dataclasses.asdict(explicit)
    assert implicit.to_dict() == explicit.to_dict()
    # The generic cache serializer still declines the existing callable setting;
    # adding provenance must neither add dataclass fields nor change this path.
    for settings in (implicit, explicit):
        with pytest.raises(_UncacheableSearchPolicy, match="builtins.function"):
            _search_policy_json_value(settings)
    assert {field.name for field in dataclasses.fields(implicit)}.isdisjoint(
        {
            "_cute_structural_origins",
            "cute_structural_origins",
            "cute_structural_policy",
        }
    )
    for name in CUTE_STRUCTURAL_SETTINGS:
        assert implicit.to_dict()[name] is False
    for settings in (implicit, explicit):
        env = CompileEnvironment(torch.device("cpu"), settings)
        assert (
            env.config_spec.cache_fingerprint_hash()
            == CompileEnvironment(
                torch.device("cpu"), implicit
            ).config_spec.cache_fingerprint_hash()
        )


def _pointwise(x):
    output = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        output[tile] = x[tile] + 1
    return output


def test_kernel_settings_object_and_kwargs_contract_is_unchanged(monkeypatch):
    monkeypatch.setenv("HELION_CUTE_REGION_FISSION", "1")
    settings = Settings(backend="triton", cute_region_fission=False)
    kernel = helion.kernel(_pointwise, settings=settings)
    assert kernel.settings is settings
    assert kernel.settings.cute_structural_origins.cute_region_fission == "explicit"
    with pytest.raises(AssertionError, match="settings must be the only"):
        helion.kernel(_pointwise, settings=settings, cute_region_fission=True)
    explicit = helion.kernel(_pointwise, backend="triton", cute_region_fission=False)
    assert explicit.settings.cute_region_fission is False
    assert explicit.settings.cute_structural_origins.cute_region_fission == "explicit"
    delayed = helion.kernel(backend="triton")
    monkeypatch.setenv("HELION_CUTE_REGION_FISSION", "0")
    captured = delayed(_pointwise)
    assert captured.settings.cute_region_fission is True
    assert (
        captured.settings.cute_structural_origins.cute_region_fission == "environment"
    )


def test_triton_codegen_and_binding_keys_ignore_origin():
    implicit = helion.kernel(_pointwise, backend="triton", autotune_effort="none")
    explicit = helion.kernel(
        _pointwise, settings=Settings(**implicit.settings.to_dict())
    )
    args = (torch.empty(37),)
    left = implicit._bind_isolated(args)
    right = explicit._bind_isolated(args)
    config = helion.Config(block_sizes=[32], num_warps=4)
    assert (
        left.config_spec.cache_fingerprint_hash()
        == right.config_spec.cache_fingerprint_hash()
    )
    assert left.extra_cache_key() == right.extra_cache_key()
    assert left.to_code(config) == right.to_code(config)
    assert not implicit.configs and not explicit.configs


def test_policy_envelope_is_not_a_flattened_config():
    config = helion.Config(block_sizes=[16, 32], loop_orders=[[0, 1]])
    snapshot = copy.deepcopy(config.config)
    policy = CuteStructuralPolicy(cute_full_slice_matmul_tiling=True)
    assert set(policy.to_dict()) == {"schema", "version", "settings"}
    assert config.config == snapshot
    with pytest.raises(ValueError, match="envelope"):
        CuteStructuralPolicy.from_dict(config.config)
    with pytest.raises(ValueError, match="envelope"):
        CuteStructuralPolicy.from_dict(dataclasses.asdict(policy))


def test_all_policies_have_stable_independent_serialization_and_identity():
    identities = set()
    for flags in itertools.product((False, True), repeat=5):
        policy = CuteStructuralPolicy(
            **dict(zip(CUTE_STRUCTURAL_SETTINGS, flags, strict=True))
        )
        envelope = policy.to_dict()
        # The wire form is complete; order and environment do not affect identity.
        assert set(envelope["settings"]) == set(CUTE_STRUCTURAL_SETTINGS)
        assert (
            tuple(envelope["settings"][name] for name in CUTE_STRUCTURAL_SETTINGS)
            == flags
        )
        roundtrip = CuteStructuralPolicy.from_dict(json.loads(policy.to_json()))
        assert roundtrip == policy
        reversed_envelope = dict(reversed(list(envelope.items())))
        reversed_envelope["settings"] = dict(
            reversed(list(envelope["settings"].items()))
        )
        assert (
            CuteStructuralPolicy.from_dict(reversed_envelope).identity()
            == policy.identity()
        )
        identities.add(policy.identity())
    assert len(identities) == 32


def test_policy_and_origin_snapshots_are_immutable_and_detached():
    settings = Settings(backend="cute")
    policy = settings.get_cute_structural_policy()
    origins = settings.cute_structural_origins
    identity = policy.identity()
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.cute_region_fission = True
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.version = 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        origins.cute_region_fission = "explicit"
    envelope = policy.to_dict()
    restored = CuteStructuralPolicy.from_dict(envelope)
    envelope["settings"]["cute_region_fission"] = True
    envelope["version"] = 99
    assert policy.identity() == restored.identity() == identity
    settings.cute_region_fission = True
    assert policy.cute_region_fission is False
    assert origins.cute_region_fission == "default"
    assert settings.get_cute_structural_policy().cute_region_fission is True


@pytest.mark.parametrize("version", [None, True, 0, 2, 1.0, "1"])
def test_unknown_or_untyped_policy_versions_are_rejected(version):
    envelope = CuteStructuralPolicy().to_dict()
    envelope["version"] = version
    with pytest.raises(ValueError, match="version"):
        CuteStructuralPolicy.from_dict(envelope)


@pytest.mark.parametrize("name", CUTE_STRUCTURAL_SETTINGS)
@pytest.mark.parametrize("value", [None, 0, 1, "false", [], {}])
def test_policy_requires_real_boolean_fields(name, value):
    with pytest.raises(TypeError, match="must be a bool"):
        CuteStructuralPolicy(**{name: value})
    envelope = CuteStructuralPolicy().to_dict()
    envelope["settings"][name] = value
    with pytest.raises(TypeError, match="must be a bool"):
        CuteStructuralPolicy.from_dict(envelope)


@pytest.mark.parametrize("change", ["schema", "extra", "missing", "flattened", "field"])
def test_incomplete_and_unrelated_envelopes_are_rejected(change):
    envelope = CuteStructuralPolicy().to_dict()
    if change == "schema":
        envelope["schema"] = "unrelated"
    elif change == "extra":
        envelope["config"] = {"block_sizes": [16]}
    elif change == "missing":
        envelope.pop("version")
    elif change == "flattened":
        envelope["settings"] = [False] * 5
    else:
        envelope["settings"].pop("cute_region_fission")
    with pytest.raises(ValueError):
        CuteStructuralPolicy.from_dict(envelope)


@pytest.mark.parametrize("value", [None, [], {}, "implicit", True])
def test_origins_cannot_contain_mutable_or_unknown_values(value):
    with pytest.raises(TypeError, match="Invalid origin"):
        CuteStructuralOrigins(cute_region_fission=value)
