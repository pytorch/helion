"""Lookup tiers and patch apply behavior."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from types import SimpleNamespace

import helion_rag.corpus as corpus
import helion_rag.lookup as lookup_mod
import helion_rag.patch as patch_mod

from ._fixtures import DTYPES
from ._fixtures import SHAPES
from ._fixtures import SRC


@contextmanager
def _env(**kw: str | None):
    old: dict[str, str | None] = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        index_dir=tmp_path / "index",
        manifest_path=None,
        hardware_family=None,
        embed_model="unused",
    )


def _write_generation(
    cfg: SimpleNamespace,
    family: str,
    *,
    exact: dict,
    runids: dict | None = None,
    with_faiss: bool = True,
) -> Path:
    fam_dir = cfg.index_dir / family
    gen = fam_dir / "generations" / "000000"
    gen.mkdir(parents=True)
    (gen / "exact.json").write_text(json.dumps(exact), encoding="utf-8")
    (gen / "runids.json").write_text(json.dumps(runids or {}), encoding="utf-8")
    if with_faiss:
        (gen / "index.faiss").write_text("placeholder", encoding="utf-8")
    (fam_dir / "current").write_text("000000\n", encoding="utf-8")
    return gen


def test_lookup_tier0_exact_hit(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    key = corpus._workload_key(SRC, SHAPES, DTYPES, {}, "h100")
    hit = {
        "best_config": {"block_size": 16},
        "best_config_id": "cfg0",
        "run_id": "RUN1",
        "ref": {"family": "h100", "source_file": "add.meta.jsonl", "run_id": "RUN1"},
        "tier0_eligible": True,
    }
    _write_generation(cfg, "h100", exact={key: hit}, runids={"RUN1": key})

    exact = lookup_mod.lookup(SRC, SHAPES, DTYPES, "h100", cfg=cfg)

    assert exact["tier"] == 0
    assert exact["best_config"] == {"block_size": 16}
    assert exact["best_config_id"] == "cfg0"


def test_lookup_returns_tier2_for_unknown_hardware_and_missing_index(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)

    def fake_resolve(hw, cfg_obj):
        # mimic token matching without torch fallback
        return "h100" if "h100" in hw.lower() else None

    unknown = lookup_mod.lookup(
        SRC,
        SHAPES,
        DTYPES,
        "mystery accelerator",
        cfg=cfg,
        _resolve_family_fn=fake_resolve,
    )
    missing = lookup_mod.lookup(
        SRC, SHAPES, DTYPES, "h100", cfg=cfg, _resolve_family_fn=fake_resolve
    )

    assert unknown == {"tier": 2, "family": None}
    assert missing == {"tier": 2, "family": "h100"}


def test_lookup_tier1_uses_faiss_neighbors(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_generation(cfg, "h100", exact={})

    doc = SimpleNamespace(
        metadata={
            "kernel_name": "add",
            "input_shapes": SHAPES,
            "dtypes": DTYPES,
            "top_n": [{"config": {"block_size": 16}, "median": 1.0}],
            "ref": {
                "family": "h100",
                "source_file": "add.meta.jsonl",
                "run_id": "RUN1",
            },
        }
    )

    class FakeStore:
        def similarity_search_with_score(self, query: str, k: int):
            assert query == SRC.strip()
            assert k == 3
            return [(doc, 0.95)]

    def fake_loader(cfg_arg, family):
        assert family == "h100"
        return FakeStore()

    with _env(HELION_RAG_SIM_THRESHOLD="0.5"):
        result = lookup_mod.lookup(
            SRC, SHAPES, DTYPES, "h100", k=3, cfg=cfg, _index_loader=fake_loader
        )

    assert result["tier"] == 1
    assert result["family"] == "h100"
    assert result["neighbors"] == [
        {
            "kernel_name": "add",
            "input_shapes": SHAPES,
            "dtypes": DTYPES,
            "top_n": [{"config": {"block_size": 16}, "median": 1.0}],
            "ref": {
                "family": "h100",
                "source_file": "add.meta.jsonl",
                "run_id": "RUN1",
            },
            "score": 0.95,
        }
    ]


def test_merge_seed_configs_keeps_user_first_and_dedups() -> None:
    user = [{"block_size": 16}, {"block_size": 32}]
    rag = [{"block_size": 16}, {"block_size": 64}]

    assert lookup_mod.merge_seed_configs(user, rag) == [
        {"block_size": 16},
        {"block_size": 32},
        {"block_size": 64},
    ]


def test_patch_disabled_and_missing_metadata_fall_back() -> None:
    bound = SimpleNamespace()
    calls: list[tuple[object, tuple]] = []

    def original(kernel, args, *rest, **kwargs):
        calls.append((kernel, tuple(args)))
        return "original"

    def raising_extract(*_args, **_kwargs):
        raise AssertionError("called")

    with _env(HELION_RAG_ENABLED=None):
        assert (
            patch_mod.apply(
                bound, original, ("x",), (), {}, _extract_fn=raising_extract
            )
            == "original"
        )

    with _env(HELION_RAG_ENABLED="1"):
        assert (
            patch_mod.apply(
                bound, original, ("y",), (), {}, _extract_fn=lambda *_: None
            )
            == "original"
        )
    assert calls == [(bound, ("x",)), (bound, ("y",))]


def test_patch_tier0_sets_config_without_autotune() -> None:
    config = object()
    bound = SimpleNamespace(configs=[])
    bound.set_config = lambda cfg: bound.configs.append(cfg)

    def original(*_args, **_kwargs):
        raise AssertionError("original autotune should not run")

    def fake_extract(_kernel, _args):
        return {
            "kernel_source": SRC,
            "shapes": SHAPES,
            "dtypes": DTYPES,
            "hardware": "h100",
            "settings": {},
        }

    def fake_lookup(*_args, **_kwargs):
        return {"tier": 0, "best_config": config}

    with _env(HELION_RAG_ENABLED="1"):
        assert (
            patch_mod.apply(
                bound,
                original,
                ("x",),
                (),
                {},
                _extract_fn=fake_extract,
                _lookup_fn=fake_lookup,
            )
            is None
        )
    assert bound.configs == [config]


def test_patch_tier1_uses_temporary_seed_configs() -> None:
    user_seed = object()
    rag_seed = object()
    settings = SimpleNamespace(autotune_seed_configs=[user_seed])
    bound = SimpleNamespace(settings=settings)
    seen_during_original: list[list[object]] = []

    def original(_kernel, _args, *rest, **kwargs):
        seen_during_original.append(list(settings.autotune_seed_configs))
        return "autotuned"

    def fake_extract(_kernel, _args):
        return {
            "kernel_source": SRC,
            "shapes": SHAPES,
            "dtypes": DTYPES,
            "hardware": "h100",
            "settings": {},
        }

    def fake_lookup(*_args, **_kwargs):
        return {"tier": 1, "neighbors": [{"top_n": [{"config": rag_seed}]}]}

    with _env(HELION_RAG_ENABLED="1"):
        assert (
            patch_mod.apply(
                bound,
                original,
                ("x",),
                (),
                {},
                _extract_fn=fake_extract,
                _lookup_fn=fake_lookup,
            )
            == "autotuned"
        )
    assert seen_during_original == [[user_seed, rag_seed]]
    assert settings.autotune_seed_configs == [user_seed]
