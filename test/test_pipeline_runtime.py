from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from helion.autotuner import aot_cache
from helion.runtime import pipeline as runtime_pipeline
from helion.runtime.config import Config
from helion.runtime.cute_structural_config import CuteStructuralConfig
from helion.runtime.cute_structural_config import StructuralPolicyError
from helion.runtime.cute_structural_policy import CuteStructuralPolicy
from helion.runtime.kernel import BoundKernel
from helion.runtime.pipeline import PipelineConfig
from helion.runtime.pipeline import PipelineScope
from helion.runtime.pipeline import _active_pipeline


def _run(scope, kernel, args):
    context = (
        scope.activate()
        if _active_pipeline.get() is None
        else contextlib.nullcontext(scope)
    )
    with context:
        scope.prepare(kernel, args)
        return kernel.bind(args)._run(*args)


def _kernel():
    def operation(x):
        return x

    old_config = Config(num_warps=4)
    old_run = object()
    compiled = []

    def compile_config(config, **kwargs):
        compiled.append(config.num_warps)
        return lambda x: x + config.num_warps

    bound = SimpleNamespace(
        _config=old_config,
        _run=old_run,
        env=SimpleNamespace(
            process_group_name=None, backend=SimpleNamespace(name="triton")
        ),
        config_spec=SimpleNamespace(
            structural_fingerprint_hash=lambda **kwargs: "test",
            autotune_reference_config=lambda: Config(num_warps=4),
        ),
        compile_config=compile_config,
    )
    kernel = SimpleNamespace(
        fn=operation,
        name="operation",
        configs=[old_config],
        settings=SimpleNamespace(distributed=False, autotune_search_acf=None),
        normalize_args=lambda *args: args,
        bind=lambda args: bound,
        kernel_source=lambda: "def operation(x): return x",
    )
    bound.kernel = kernel
    bound.settings = kernel.settings
    return kernel, bound, compiled


@pytest.fixture
def aot_kernel(monkeypatch, tmp_path):
    kernel, bound, compiled = _kernel()
    bound._config = None
    kernel.configs = []
    kernel.settings.autotune_cache = "AOTAutotuneCache"
    heuristic = tmp_path / "_helion_aot_operation.py"
    monkeypatch.setattr(aot_cache, "find_heuristic_file", lambda *a, **k: heuristic)
    return kernel, bound, compiled, heuristic


def test_scope_changes_only_local_config_and_restores_after_error():
    kernel, bound, compiled = _kernel()
    original = bound._config, bound._run
    x = torch.ones(8)
    scope = PipelineScope(PipelineConfig())
    with pytest.raises(ValueError, match="caller failure"), scope.activate():
        torch.testing.assert_close(_run(scope, kernel, (x,)), x + 4)
        raise ValueError("caller failure")
    assert _active_pipeline.get() is None
    assert (bound._config, bound._run) == original
    key = scope.trace[0]
    alternative = scope.bundle.with_config(key, Config(num_warps=8))
    with alternative.activate() as configured:
        torch.testing.assert_close(_run(configured, kernel, (x,)), x + 8)
    assert compiled == [4, 8]
    assert (bound._config, bound._run) == original


def test_value_changes_share_key_but_shape_changes_do_not():
    kernel, _, _ = _kernel()
    scope = PipelineScope(PipelineConfig())
    _run(scope, kernel, (torch.ones(8),))
    _run(scope, kernel, (torch.zeros(8),))
    _run(scope, kernel, (torch.zeros(16),))
    assert scope.trace[0] == scope.trace[1]
    assert scope.trace[0] != scope.trace[2]
    assert len(scope.stages) == 2


def test_new_key_on_shared_binding_uses_saved_aot_config(aot_kernel):
    kernel, bound, compiled, heuristic = aot_kernel
    heuristic.write_text(
        "def autotune_operation(*args):\n    return {'num_warps': 4}\n"
    )
    original = bound._config, bound._run
    discovery = PipelineScope(PipelineConfig())
    _run(discovery, kernel, (torch.ones(8),))
    key = discovery.trace[0]
    candidate = discovery.bundle.with_config(key, Config(num_warps=8))
    with candidate.activate(allow_new=True) as prepared:
        torch.testing.assert_close(
            _run(prepared, kernel, (torch.ones(8),)), torch.ones(8) + 8
        )
        torch.testing.assert_close(
            _run(prepared, kernel, (torch.ones(16),)), torch.ones(16) + 4
        )
        assert next(reversed(prepared.stages.values())).config_source == (
            f"saved_aot:{heuristic}"
        )
    assert (bound._config, bound._run) == original


def test_freeze_reuses_known_compilation_and_rejects_new_shape_or_binding():
    kernel, bound, compiled = _kernel()
    scope = PipelineScope(PipelineConfig())
    x = torch.ones(8)
    _run(scope, kernel, (x,))
    scope.freeze()
    _run(scope, kernel, (x,))
    assert compiled == [4]
    with pytest.raises(RuntimeError, match="Unseen pipeline stage"):
        _run(scope, kernel, (torch.ones(16),))
    kernel.bind = lambda args: SimpleNamespace(**vars(bound))
    with pytest.raises(RuntimeError, match="compilation after freeze"):
        _run(scope, kernel, (x,))


def test_direct_bound_call_uses_its_run_without_scope_dispatch():
    bound = object.__new__(BoundKernel)
    bound._cache_managed = False
    bound._run = lambda value: value + 1

    def unexpected_dispatch(*args):
        pytest.fail("BoundKernel calls must not reenter pipeline dispatch")

    token = _active_pipeline.set(SimpleNamespace(prepare=unexpected_dispatch))
    try:
        assert bound(4) == 5
    finally:
        _active_pipeline.reset(token)


def test_nested_scope_is_rejected_without_losing_outer_scope():
    first, second = PipelineScope(PipelineConfig()), PipelineScope(PipelineConfig())
    with first.activate():
        with pytest.raises(RuntimeError, match="Nested pipeline"), second.activate():
            pytest.fail("nested scope entered")
        assert _active_pipeline.get() is first
    assert _active_pipeline.get() is None


def test_bundle_serialization_does_not_alias_user_config():
    original = Config(block_sizes=[64])
    bundle = PipelineConfig({"key": original, "other": original})
    original.config["block_sizes"][0] = 128
    assert bundle.configs["key"].block_sizes == [64]
    assert PipelineConfig.from_dict(bundle.to_dict()).digest() == bundle.digest()
    scope = PipelineScope(bundle)
    scope.configs["key"].block_sizes[0] = 256
    assert bundle.configs["key"].block_sizes == [64]
    assert scope.configs["other"].block_sizes == [64]
    returned = scope.bundle
    scope.configs["key"].block_sizes[0] = 512
    assert returned.configs["key"].block_sizes == [256]


def test_custom_sharing_key_is_preserved_when_applying_a_returned_bundle():
    kernel, _, _ = _kernel()

    def key_fn(kernel, args):
        return "shared-across-shapes"

    scope = PipelineScope(PipelineConfig(), key_fn=key_fn)
    _run(scope, kernel, (torch.ones(8),))
    bundle = scope.bundle
    with bundle.activate() as applied:
        _run(applied, kernel, (torch.ones(16),))
    assert applied.trace == scope.trace
    restored = PipelineConfig.from_dict(bundle.to_dict(), key_fn=key_fn)
    with restored.activate() as applied:
        _run(applied, kernel, (torch.ones(32),))
    assert applied.trace == scope.trace


def test_saved_aot_config_is_pinned_even_when_autotune_cache_is_skipped(
    monkeypatch, aot_kernel
):
    kernel, bound, compiled, heuristic = aot_kernel
    kernel._aot_user_key = lambda x: (x.numel(),)
    heuristic.write_text(
        "def autotune_operation(length):\n"
        "    assert length == 8\n"
        "    return {'num_warps': 8}\n"
    )
    monkeypatch.setenv("HELION_SKIP_CACHE", "1")
    scope = PipelineScope(PipelineConfig())
    _run(scope, kernel, (torch.ones(8),))
    stage = next(iter(scope.stages.values()))
    assert stage.config.num_warps == 8
    assert stage.config_source == f"saved_aot:{heuristic}"
    assert compiled == [8]
    assert bound._config is None


def test_incompatible_config_spaces_have_distinct_default_keys():
    first, _, _ = _kernel()
    second, bound, _ = _kernel()
    bound.config_spec.structural_fingerprint_hash = lambda **kwargs: "other-space"
    scope = PipelineScope(PipelineConfig())
    _run(scope, first, (torch.ones(8),))
    _run(scope, second, (torch.ones(8),))
    assert scope.trace[0] != scope.trace[1]


def test_custom_key_cannot_force_incompatible_config_spaces_to_share():
    first, _, _ = _kernel()
    second, bound, _ = _kernel()
    bound.config_spec.structural_fingerprint_hash = lambda **kwargs: "other-space"
    scope = PipelineScope(PipelineConfig(), key_fn=lambda kernel, args: "shared")
    _run(scope, first, (torch.ones(8),))
    with pytest.raises(ValueError, match="incompatible"):
        _run(scope, second, (torch.ones(8),))


@pytest.mark.parametrize("statement", ["return None", "raise KeyError('new shape')"])
def test_missing_aot_shape_uses_explicit_default_without_nested_autotuning(
    aot_kernel, statement
):
    kernel, bound, compiled, heuristic = aot_kernel
    heuristic.write_text(f"def autotune_operation(*args):\n    {statement}\n")
    scope = PipelineScope(PipelineConfig())
    _run(scope, kernel, (torch.ones(8),))
    assert compiled == [4]
    assert next(iter(scope.stages.values())).config_source == "reference_default"
    assert bound._config is None


@pytest.mark.parametrize(
    "saved", ["matching", "different", "unversioned", "different_manifest"]
)
def test_saved_aot_config_respects_bound_structural_policy(
    monkeypatch, aot_kernel, saved
):
    kernel, bound, compiled, heuristic = aot_kernel
    bound.env.backend.name = "cute"
    policy = CuteStructuralPolicy(cute_region_fission=True)
    recorded = CuteStructuralConfig(
        Config(num_warps=8), policy if saved == "matching" else CuteStructuralPolicy()
    )
    selected = recorded.config if saved == "unversioned" else recorded
    source = f"import helion\ndef autotune_operation(*args):\n    return {selected!r}\n"
    if saved == "different_manifest":
        manifest = {
            "schema": "helion.aot.structural_model",
            "version": 1,
            "policy": recorded.policy.to_dict(),
            "policy_id": recorded.policy.identity(),
            "configs": {kernel.name: [recorded.to_dict()]},
        }
        source = (
            f"_HELION_AOT_STRUCTURAL_MANIFEST = {manifest!r}\n"
            "def autotune_operation(*args):\n"
            "    raise AssertionError('selector ran before manifest validation')\n"
        )
    heuristic.write_text(source)

    def find_heuristic(*args, **kwargs):
        assert kwargs["policy"] is policy
        return heuristic

    def forbid_autotuning(*args, **kwargs):
        pytest.fail("pipeline resolution started an autotuner")

    monkeypatch.setattr(aot_cache, "find_heuristic_file", find_heuristic)
    monkeypatch.setattr(aot_cache.AOTAutotuneCache, "__init__", forbid_autotuning)
    monkeypatch.setattr(BoundKernel, "autotune", forbid_autotuning)
    monkeypatch.setattr(
        runtime_pipeline, "bound_structural_policy", lambda bound: policy
    )
    scope = PipelineScope(PipelineConfig())
    if saved == "matching":
        _run(scope, kernel, (torch.ones(8),))
        assert next(iter(scope.stages.values())).config.num_warps == 8
        assert compiled == [8]
    else:
        with pytest.raises(
            StructuralPolicyError, match="different CuTe structural policy"
        ):
            _run(scope, kernel, (torch.ones(8),))
        assert compiled == []
    assert bound._config is None
