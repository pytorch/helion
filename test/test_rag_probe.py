from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import helion
from helion.autotuner.local_cache import _load_best_config
from helion.autotuner.rag import ExactHit
from helion.autotuner.rag import ExactMiss
from helion.autotuner.rag import ExactReadError
from helion.autotuner.rag import probe_exact_cache
from helion.autotuner.rag import rag_enabled
from helion.autotuner.rag import rag_enabled_env
from helion.runtime.settings import Settings


class _FakeCache:
    def __init__(self, *, result=None, exc=None):
        self._result = result
        self._exc = exc

    def get_or_raise(self):
        if self._exc is not None:
            raise self._exc
        return self._result


# --- typed probe -----------------------------------------------------------
def test_probe_hit():
    config = object()
    result = probe_exact_cache(_FakeCache(result=config))
    assert result == ExactHit(config=config)


def test_probe_miss():
    assert probe_exact_cache(_FakeCache(result=None)) == ExactMiss()


def test_probe_read_error_is_not_a_miss():
    result = probe_exact_cache(_FakeCache(exc=ValueError("boom")))
    assert isinstance(result, ExactReadError)
    assert "ValueError" in result.error and "boom" in result.error


# --- loader: miss (None) vs error (raise) ----------------------------------
def test_load_best_config_missing_returns_none(tmp_path):
    assert _load_best_config(tmp_path / "absent.best_config") is None


def test_load_best_config_corrupt_raises(tmp_path):
    p = tmp_path / "corrupt.best_config"
    p.write_text("not json{")
    with pytest.raises(json.JSONDecodeError):
        _load_best_config(p)


def test_load_best_config_missing_key_raises(tmp_path):
    p = tmp_path / "nokey.best_config"
    p.write_text(json.dumps({"not_config": 1}))
    with pytest.raises(KeyError):
        _load_best_config(p)


def test_load_best_config_roundtrip(tmp_path):
    cfg = helion.Config(block_sizes=[32])
    p = tmp_path / "ok.best_config"
    p.write_text(json.dumps({"config": cfg.to_json()}))
    loaded = _load_best_config(p)
    assert loaded is not None
    assert loaded.config == cfg.config


# --- kill switch -----------------------------------------------------------
def test_rag_enabled_reads_settings_flag():
    assert rag_enabled(SimpleNamespace(autotune_rag_enabled=True))
    assert not rag_enabled(SimpleNamespace(autotune_rag_enabled=False))


def test_rag_enabled_env(monkeypatch):
    monkeypatch.delenv("HELION_RAG_ENABLED", raising=False)
    assert not rag_enabled_env()
    monkeypatch.setenv("HELION_RAG_ENABLED", "1")
    assert rag_enabled_env()
    monkeypatch.setenv("HELION_RAG_ENABLED", "0")
    assert not rag_enabled_env()


# --- decoupled cache-access settings ---------------------------------------
def test_cache_access_defaults_preserve_behavior(monkeypatch):
    for var in (
        "HELION_RAG_ENABLED",
        "HELION_AUTOTUNE_EXACT_READ",
        "HELION_AUTOTUNE_BEST_AVAILABLE_READ",
        "HELION_AUTOTUNE_CACHE_WRITE",
    ):
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    assert s.autotune_rag_enabled is False
    assert s.autotune_exact_read is True
    assert s.autotune_best_available_read is True
    assert s.autotune_cache_write is True


def test_cache_access_env_overrides(monkeypatch):
    monkeypatch.setenv("HELION_RAG_ENABLED", "1")
    monkeypatch.setenv("HELION_AUTOTUNE_EXACT_READ", "0")
    monkeypatch.setenv("HELION_AUTOTUNE_BEST_AVAILABLE_READ", "0")
    monkeypatch.setenv("HELION_AUTOTUNE_CACHE_WRITE", "0")
    s = Settings()
    assert s.autotune_rag_enabled is True
    assert s.autotune_exact_read is False
    assert s.autotune_best_available_read is False
    assert s.autotune_cache_write is False
