from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from helion.runtime.config import Config
from helion.runtime.kernel import BoundKernel
from helion.runtime.kernel import Kernel
from helion.runtime.pipeline import PipelineConfig
from helion.runtime.pipeline import PipelineScope
from helion.runtime.pipeline import _active_pipeline


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


def test_scope_changes_only_local_config_and_restores_after_error():
    kernel, bound, compiled = _kernel()
    original = bound._config, bound._run
    x = torch.ones(8)
    scope = PipelineScope(PipelineConfig())
    with pytest.raises(ValueError, match="caller failure"), scope.activate():
        torch.testing.assert_close(scope.call(kernel, (x,)), x + 4)
        raise ValueError("caller failure")
    assert _active_pipeline.get() is None
    assert (bound._config, bound._run) == original
    key = scope.trace[0]
    alternative = scope.bundle.with_config(key, Config(num_warps=8))
    with alternative.activate() as configured:
        torch.testing.assert_close(configured.call(kernel, (x,)), x + 8)
    assert compiled == [4, 8]
    assert (bound._config, bound._run) == original


def test_value_changes_share_key_but_shape_changes_do_not():
    kernel, _, _ = _kernel()
    scope = PipelineScope(PipelineConfig())
    scope.call(kernel, (torch.ones(8),))
    scope.call(kernel, (torch.zeros(8),))
    scope.call(kernel, (torch.zeros(16),))
    assert scope.trace[0] == scope.trace[1]
    assert scope.trace[0] != scope.trace[2]
    assert len(scope.stages) == 2


def test_freeze_reuses_known_compilation_and_rejects_new_shape_or_binding():
    kernel, bound, compiled = _kernel()
    scope = PipelineScope(PipelineConfig())
    x = torch.ones(8)
    scope.call(kernel, (x,))
    scope.freeze()
    scope.call(kernel, (x,))
    assert compiled == [4]
    with pytest.raises(RuntimeError, match="Unseen pipeline stage"):
        scope.call(kernel, (torch.ones(16),))
    kernel.bind = lambda args: SimpleNamespace(**vars(bound))
    with pytest.raises(RuntimeError, match="compilation after freeze"):
        scope.call(kernel, (x,))


def test_native_scope_precedes_kernel_and_bound_prepared_fast_paths():
    scope = SimpleNamespace(call=lambda kernel, args: (kernel, args))
    token = _active_pipeline.set(scope)
    try:
        kernel = object.__new__(Kernel)
        bound = object.__new__(BoundKernel)
        bound.kernel = kernel
        # Neither object has ordinary dispatch state; reaching a fast path
        # instead of the scope would fail before returning this result.
        assert kernel(3) == (kernel, (3,))
        assert bound(4) == (kernel, (4,))
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
    bundle = PipelineConfig({"key": original})
    original.config["block_sizes"][0] = 128
    assert bundle.configs["key"].block_sizes == [64]
    assert PipelineConfig.from_dict(bundle.to_dict()).digest() == bundle.digest()


def test_custom_sharing_key_is_preserved_when_applying_a_returned_bundle():
    kernel, _, _ = _kernel()

    def key_fn(kernel, args):
        return "shared-across-shapes"

    scope = PipelineScope(PipelineConfig(), key_fn=key_fn)
    scope.call(kernel, (torch.ones(8),))
    bundle = scope.bundle.copy()
    with bundle.activate() as applied:
        applied.call(kernel, (torch.ones(16),))
    assert applied.trace == scope.trace
    restored = PipelineConfig.from_dict(bundle.to_dict(), key_fn=key_fn)
    with restored.activate() as applied:
        applied.call(kernel, (torch.ones(32),))
    assert applied.trace == scope.trace


def test_saved_aot_config_is_pinned_even_when_autotune_cache_is_skipped(
    monkeypatch, tmp_path
):
    from helion.autotuner import aot_cache

    kernel, bound, compiled = _kernel()
    bound._config = None
    kernel.configs = []
    kernel.settings.autotune_cache = "AOTAutotuneCache"
    kernel._aot_user_key = lambda x: (x.numel(),)
    heuristic = tmp_path / "_helion_aot_operation.py"
    heuristic.write_text(
        "def autotune_operation(length):\n"
        "    assert length == 8\n"
        "    return {'num_warps': 8}\n"
    )
    monkeypatch.setattr(aot_cache, "find_heuristic_file", lambda *a, **k: heuristic)
    monkeypatch.setenv("HELION_SKIP_CACHE", "1")
    scope = PipelineScope(PipelineConfig())
    scope.call(kernel, (torch.ones(8),))
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
    scope.call(first, (torch.ones(8),))
    scope.call(second, (torch.ones(8),))
    assert scope.trace[0] != scope.trace[1]


def test_custom_key_cannot_force_incompatible_config_spaces_to_share():
    first, _, _ = _kernel()
    second, bound, _ = _kernel()
    bound.config_spec.structural_fingerprint_hash = lambda **kwargs: "other-space"
    scope = PipelineScope(PipelineConfig(), key_fn=lambda kernel, args: "shared")
    scope.call(first, (torch.ones(8),))
    with pytest.raises(ValueError, match="incompatible"):
        scope.call(second, (torch.ones(8),))


@pytest.mark.parametrize("statement", ["return None", "raise KeyError('new shape')"])
def test_missing_aot_shape_uses_explicit_default_without_nested_autotuning(
    monkeypatch, tmp_path, statement
):
    from helion.autotuner import aot_cache

    kernel, bound, compiled = _kernel()
    bound._config = None
    kernel.configs = []
    kernel.settings.autotune_cache = "AOTAutotuneCache"
    heuristic = tmp_path / "_helion_aot_operation.py"
    heuristic.write_text(f"def autotune_operation(*args):\n    {statement}\n")
    monkeypatch.setattr(aot_cache, "find_heuristic_file", lambda *a, **k: heuristic)
    scope = PipelineScope(PipelineConfig())
    scope.call(kernel, (torch.ones(8),))
    assert compiled == [4]
    assert next(iter(scope.stages.values())).config_source == "reference_default"
    assert bound._config is None
