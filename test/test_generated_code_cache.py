from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

import helion
from helion._utils import counters
from helion.autotuner.base_search import BaseAutotuner
import helion.language as hl
from helion.runtime import generated_code_cache as cache
from helion.runtime.kernel import KernelCompiler
from helion.runtime.kernel import PyCodeCache

if TYPE_CHECKING:
    from pathlib import Path


@helion.kernel(backend="triton")
def _cached_scale(x: torch.Tensor, scale: int) -> torch.Tensor:
    scale = hl.specialize(scale)
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] * scale
    return out


_SHIFT = 1


class _OpaqueState:
    enabled = True


_OPAQUE = _OpaqueState()


@helion.kernel(backend="triton")
def _cached_opaque(x: torch.Tensor) -> torch.Tensor:
    if _OPAQUE.enabled:
        return x
    return x


@helion.kernel(backend="triton")
def _cached_shift(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + _SHIFT
    return out


def _bound(x: torch.Tensor, scale: int = 2, **settings: object):
    kernel = helion.kernel(backend="triton", generated_code_cache=True, **settings)(
        _cached_scale.fn
    )
    return kernel.bind((x, scale))


@pytest.fixture
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HELION_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("HELION_SKIP_CACHE", raising=False)
    # CPU tests exercise the real frontend/codegen, but never import Triton.
    monkeypatch.setattr(cache, "helion_key", lambda: "helion-v1")
    monkeypatch.setattr(cache, "torch_key_wrapper", lambda: "torch-v1")
    monkeypatch.setattr(cache, "triton_key_wrapper", lambda: "triton-v1")
    counters["generated_code_cache"].clear()
    return tmp_path


def _source_loader(root: Path, sources: list[str]):
    def load(source: str, *, extra: str = ""):
        sources.append(source)
        path = root / f"module_{len(sources)}.py"
        path.write_text(source)
        # Importing/launching the generated module belongs to the CUDA test.
        return SimpleNamespace(__file__=str(path), _cached_scale=lambda *args: None)

    return load


def test_fresh_binding_skips_codegen(cache_root: Path) -> None:
    config = helion.Config(block_sizes=[32])
    x = torch.ones(16)
    first = _bound(x)
    sources: list[str] = []
    with patch.object(
        PyCodeCache, "load", side_effect=_source_loader(cache_root, sources)
    ):
        first.set_config(config)
        with patch.object(
            KernelCompiler, "compile", autospec=True, side_effect=KernelCompiler.compile
        ) as compile_fn:
            second = _bound(x)
        assert compile_fn.call_count == 0
        with patch.object(
            second, "to_triton_code", side_effect=AssertionError("codegen cache miss")
        ):
            second.set_config(config)
    assert sources[0] == sources[1]
    assert counters["generated_code_cache"] == {"miss": 1, "hit": 1, "frontend_hit": 1}


def test_only_selected_config_is_saved(cache_root: Path) -> None:
    bound = _bound(torch.ones(16))
    winner = helion.Config(block_sizes=[32])
    other = helion.Config(block_sizes=[64])
    sources: list[str] = []
    with patch.object(
        PyCodeCache, "load", side_effect=_source_loader(cache_root, sources)
    ):
        bound.compile_config(winner)
        bound.compile_config(other)
        assert not (cache_root / "generated_code").exists()
        with patch.object(
            bound, "to_triton_code", side_effect=AssertionError("winner regenerated")
        ):
            bound.set_config(winner)
        fresh = _bound(torch.ones(16))
        with patch.object(
            fresh, "to_triton_code", side_effect=AssertionError("winner cache miss")
        ):
            fresh.set_config(winner)
    assert len(sources) == 3
    assert sources[0] == sources[2]
    entries = list((cache_root / "generated_code").glob("*.json"))
    assert len(entries) == 1
    assert json.loads(entries[0].read_text())["source"] == sources[0]


def test_search_seed_does_not_change_source_key(cache_root: Path) -> None:
    first = _bound(torch.ones(16), autotune_random_seed=1)
    second = _bound(torch.ones(16), autotune_random_seed=2)
    config = helion.Config(block_sizes=[32])
    key = cache.generated_code_cache_key(first, first._normalized_config_copy(config))
    assert key is not None
    assert key == cache.generated_code_cache_key(
        second, second._normalized_config_copy(config)
    )


def test_dynamic_shapes_skip_frontend(cache_root: Path) -> None:
    config = helion.Config(block_sizes=[32])
    x = torch.ones(16)
    sources: list[str] = []
    with patch.object(
        PyCodeCache, "load", side_effect=_source_loader(cache_root, sources)
    ):
        _bound(x, static_shapes=False).set_config(config)
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
        ):
            fresh = _bound(x, static_shapes=False)
            fresh.set_config(config)
    assert sources[0] == sources[1]


def test_other_config_materializes_frontend(cache_root: Path) -> None:
    x = torch.ones(16)
    sources: list[str] = []
    with patch.object(
        PyCodeCache, "load", side_effect=_source_loader(cache_root, sources)
    ):
        _bound(x).set_config(helion.Config(block_sizes=[32]))
        fresh = _bound(x)
        with patch.object(
            KernelCompiler, "compile", autospec=True, side_effect=KernelCompiler.compile
        ) as compile_fn:
            fresh.set_config(helion.Config(block_sizes=[64]))
        assert compile_fn.call_count == 1
    assert sources[0] != sources[1]


def test_frontend_introspection_after_inputs_expire(cache_root: Path) -> None:
    sources: list[str] = []
    with patch.object(
        PyCodeCache, "load", side_effect=_source_loader(cache_root, sources)
    ):
        _bound(torch.ones(16)).set_config(helion.Config(block_sizes=[32]))
    fresh = _bound(torch.ones(16))
    assert "tl.store" in fresh.to_code(helion.Config(block_sizes=[64]))


def test_preset_selector_still_runs(cache_root: Path) -> None:
    calls = []
    config = helion.Config(block_sizes=[32])

    class PresetSearch(BaseAutotuner):
        def autotune(self, *, skip_cache=False):
            calls.append(skip_cache)
            return config

    def selector(bound, args):
        return PresetSearch()

    x = torch.ones(16)
    sources: list[str] = []
    with patch.object(
        PyCodeCache, "load", side_effect=_source_loader(cache_root, sources)
    ):
        _bound(x, static_shapes=False, autotuner_fn=selector).autotune(
            (x, 2), force=False
        )
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
        ):
            _bound(x, static_shapes=False, autotuner_fn=selector).autotune(
                (x, 2), force=False
            )
    assert calls == [False, False]


def test_changed_decorator_config_takes_precedence(cache_root: Path) -> None:
    x = torch.ones(16)
    first = helion.Config(block_sizes=[32])
    second = helion.Config(block_sizes=[64])
    selector = Mock(side_effect=AssertionError("explicit config ignored"))
    sources: list[str] = []
    with patch.object(
        PyCodeCache, "load", side_effect=_source_loader(cache_root, sources)
    ):
        _bound(x, config=first, autotuner_fn=selector).autotune((x, 2), force=False)
        fresh = _bound(x, config=second, autotuner_fn=selector)
        with patch.object(
            KernelCompiler, "compile", autospec=True, side_effect=KernelCompiler.compile
        ) as compile_fn:
            assert fresh.autotune((x, 2), force=False) == second
        assert compile_fn.call_count == 1
    assert sources[0] != sources[1]


def test_cached_kernel_call_and_reset(cache_root: Path) -> None:
    x = torch.ones(16)
    sources: list[str] = []
    with patch.object(
        PyCodeCache, "load", side_effect=_source_loader(cache_root, sources)
    ):
        _bound(x).set_config(helion.Config(block_sizes=[32]))
        with patch.object(
            KernelCompiler, "compile", side_effect=AssertionError("frontend ran")
        ):
            fresh = _bound(x)
            assert fresh.kernel(x, 2) is None
            with pytest.raises(TypeError, match="missing a required argument"):
                fresh(x)
            fresh.kernel.reset()
            assert fresh(x, 2) is None
        with patch.object(
            KernelCompiler, "compile", autospec=True, side_effect=KernelCompiler.compile
        ) as compile_fn:
            rebound = fresh.kernel.bind((x, 3))
        assert compile_fn.call_count == 1
        assert rebound is not fresh


@pytest.mark.parametrize("damaged", ["binding", "source"])
def test_broken_binding_artifact_compiles_normally(
    cache_root: Path, damaged: str
) -> None:
    x = torch.ones(16)
    sources: list[str] = []
    with patch.object(
        PyCodeCache, "load", side_effect=_source_loader(cache_root, sources)
    ):
        _bound(x).set_config(helion.Config(block_sizes=[32]))
        directory = cache_root / "generated_code"
        path = (
            next((directory / "bindings").glob("*.json"))
            if damaged == "binding"
            else next(directory.glob("*.json"))
        )
        path.write_text("{")
        with patch.object(
            KernelCompiler, "compile", autospec=True, side_effect=KernelCompiler.compile
        ) as compile_fn:
            fresh = _bound(x)
        assert compile_fn.call_count == 1
        fresh.set_config(helion.Config(block_sizes=[32]))


def test_skip_cache_does_not_publish_binding(
    cache_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    x = torch.ones(16)
    bound = _bound(x)
    sources: list[str] = []
    monkeypatch.setenv("HELION_SKIP_CACHE", "1")
    with patch.object(
        PyCodeCache, "load", side_effect=_source_loader(cache_root, sources)
    ):
        bound.set_config(helion.Config(block_sizes=[32]))
    assert not (cache_root / "generated_code").exists()


def test_opaque_dependency_keeps_frontend(cache_root: Path) -> None:
    kernel = helion.kernel(_cached_opaque.fn, generated_code_cache=True)
    assert cache.exact_input_key(kernel, (torch.ones(16),)) is None


def test_global_values_invalidate_frontend_identity(
    cache_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def bind():
        kernel = helion.kernel(backend="triton", generated_code_cache=True)(
            _cached_shift.fn
        )
        return kernel.bind((torch.ones(16),))

    first = bind()
    config = helion.Config(block_sizes=[32])
    key = cache.generated_code_cache_key(first, first._normalized_config_copy(config))
    monkeypatch.setitem(_cached_shift.fn.__globals__, "_SHIFT", 2)
    second = bind()
    other = cache.generated_code_cache_key(
        second, second._normalized_config_copy(config)
    )
    # Even runtime globals conservatively invalidate a pre-frontend artifact.
    assert key is not None and key != other
    assert first.to_code(config) == second.to_code(config)


@pytest.mark.parametrize(
    "change",
    ["shape", "stride", "dtype", "scale", "config", "settings", "version", "hardware"],
)
def test_key_invalidation(
    cache_root: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    config = helion.Config(block_sizes=[32])
    x = torch.ones(16)
    first = _bound(x)
    key = cache.generated_code_cache_key(first, first._normalized_config_copy(config))
    settings: dict[str, object] = {}
    scale = 2
    if change == "shape":
        x = torch.ones(32)
    elif change == "stride":
        x = torch.ones(32)[::2]
    elif change == "dtype":
        x = x.to(torch.float64)
    elif change == "scale":
        scale = 3
    elif change == "config":
        config = helion.Config(block_sizes=[64])
    elif change == "settings":
        settings["fast_math"] = True
    elif change == "version":
        monkeypatch.setattr(cache, "triton_key_wrapper", lambda: "triton-v2")
    else:
        monkeypatch.setattr(cache, "get_device_name", lambda device: "different GPU")
    second = _bound(x, scale, **settings)
    other = cache.generated_code_cache_key(
        second, second._normalized_config_copy(config)
    )
    assert key is not None and other is not None and key != other


@pytest.mark.parametrize("skip", ["disabled", "descriptor", "skip_cache"])
def test_ineligible_configs_use_codegen(
    cache_root: Path, monkeypatch: pytest.MonkeyPatch, skip: str
) -> None:
    bound = _bound(torch.ones(16))
    if skip == "disabled":
        bound.settings.generated_code_cache = False
    if skip == "skip_cache":
        monkeypatch.setenv("HELION_SKIP_CACHE", "1")
    config = bound._normalized_config_copy(helion.Config(block_sizes=[32]))
    if skip == "descriptor":
        config = helion.Config(block_sizes=[32], indexing="tensor_descriptor")
    assert cache.generated_code_cache_key(bound, config) is None
    assert not (cache_root / "generated_code").exists()


@pytest.mark.parametrize(
    "contents",
    ["{", "null", "{}", '{"source": 1}', '{"source": "pass", "sha256": "bad"}'],
)
def test_corrupt_entry_is_a_miss(cache_root: Path, contents: str) -> None:
    directory = cache_root / "generated_code"
    directory.mkdir()
    (directory / "test.json").write_text(contents)
    assert cache.load_generated_code("test") is None
    cache.save_generated_code("test", "def add(x):\n    return x + 1\n")
    assert cache.load_generated_code("test") == "def add(x):\n    return x + 1\n"


def test_atomic_writes_and_failed_publish(cache_root: Path) -> None:
    sources = [f"def f():\n    return {i}\n" for i in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda source: cache.save_generated_code("shared", source), sources
            )
        )
    assert cache.load_generated_code("shared") in sources
    directory = cache_root / "generated_code"
    assert sorted(path.name for path in directory.iterdir()) == ["shared.json"]
    with patch.object(cache.os, "replace", side_effect=OSError("read-only cache")):
        cache.save_generated_code("shared", "replacement")
    assert cache.load_generated_code("shared") in sources
    assert sorted(path.name for path in directory.iterdir()) == ["shared.json"]


def test_cpu_reload_in_fresh_process(cache_root: Path) -> None:
    bound = _bound(torch.ones(16))
    sources: list[str] = []
    with patch.object(
        PyCodeCache, "load", side_effect=_source_loader(cache_root, sources)
    ):
        bound.set_config(helion.Config(block_sizes=[32]))
    code = """
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location(sys.argv[2], sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
for name in ("helion_key", "torch_key_wrapper", "triton_key_wrapper"):
    setattr(module.cache, name, lambda name=name: {"helion_key": "helion-v1", "torch_key_wrapper": "torch-v1", "triton_key_wrapper": "triton-v1"}[name])
sources = []
with patch.object(module.KernelCompiler, "compile", side_effect=AssertionError("frontend cache miss")), patch.object(module.PyCodeCache, "load", side_effect=module._source_loader(Path(sys.argv[3]), sources)):
    bound = module._bound(torch.ones(16))
    with patch.object(bound, "to_triton_code", side_effect=AssertionError("codegen cache miss")):
        bound.set_config(module.helion.Config(block_sizes=[32]))
assert module.counters["generated_code_cache"]["hit"] == 1
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            __file__,
            _cached_scale.fn.__module__,
            str(cache_root),
        ],
        check=True,
        timeout=30,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")
@pytest.mark.parametrize("static_shapes", [True, False])
def test_cuda_reload_in_fresh_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, static_shapes: bool
) -> None:
    monkeypatch.setenv("HELION_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("HELION_SKIP_CACHE", raising=False)
    x = torch.randn(16, device=torch.device("cuda"))
    bound = _bound(x, static_shapes=static_shapes)
    bound.set_config(helion.Config(block_sizes=[32]))
    torch.testing.assert_close(bound(x, 2), x * 2)
    code = """
import importlib.util
import sys
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location(sys.argv[2], sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
x = torch.randn(16, device=torch.device("cuda"))
with patch.object(module.KernelCompiler, "compile", side_effect=AssertionError("frontend cache miss")):
    bound = module._bound(x, static_shapes=sys.argv[3] == "True")
    with patch.object(bound, "to_triton_code", side_effect=AssertionError("codegen cache miss")):
        bound.set_config(module.helion.Config(block_sizes=[32]))
        torch.testing.assert_close(bound(x, 2), x * 2)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = bound(x, 2)
        x.fill_(3)
        graph.replay()
        torch.testing.assert_close(out, x * 2)
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            __file__,
            _cached_scale.fn.__module__,
            str(static_shapes),
        ],
        check=True,
        timeout=30,
    )
