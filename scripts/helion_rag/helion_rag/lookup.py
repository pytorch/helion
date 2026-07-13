"""Tiered lookup: exact match, then similar kernels, then miss."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types

from helion_rag._util import _log
from helion_rag._util import _sim_threshold
from helion_rag.config import _config
import helion_rag.corpus as corpus
import helion_rag.hardware as hardware
import helion_rag.index as index_mod
from helion_rag.manifest import load_manifest


def _load_manifest_opt(cfg) -> dict | None:
    """Load manifest if configured, else None."""
    return (
        load_manifest(cfg.manifest_path)
        if getattr(cfg, "manifest_path", None)
        else None
    )


def _normalize_config(c: dict) -> str:
    return json.dumps(c, sort_keys=True, default=str)


def merge_seed_configs(user_seeds: list[dict], rag_seeds: list[dict]) -> list[dict]:
    """Combine user and RAG seed configs, user first, deduped."""
    seen: dict[str, dict] = {}
    for c in (*(user_seeds or []), *(rag_seeds or [])):
        k = _normalize_config(c)
        if k not in seen:
            seen[k] = c
    return list(seen.values())


def _resolve_family(hardware_str: str, cfg) -> str | None:
    return hardware.resolve_family(
        device=hardware_str,
        manifest=_load_manifest_opt(cfg),
        env_family=getattr(cfg, "hardware_family", None),
    )


def _tier2(family: str | None = None, msg: str | None = None) -> dict:
    if msg:
        _log(msg)
    return {"tier": 2, "family": family}


def _lookup_impl(
    kernel_source: str,
    shapes: str,
    dtypes: str,
    hardware_str: str,
    settings: dict | None = None,
    k: int = 8,
    cfg=None,
    *,
    _index_loader=None,
    _resolve_family_fn=None,
) -> dict:
    """Try exact match, then vector similarity, else Tier 2 miss."""
    cfg = cfg or _config()
    resolve_fn = _resolve_family_fn or _resolve_family
    family = resolve_fn(hardware_str, cfg)
    if family is None:
        return _tier2(
            None, f"lookup: unrecognized hardware {hardware_str!r}; Tier 2 miss"
        )

    fam_dir = Path(cfg.index_dir) / family
    key = corpus._workload_key(kernel_source, shapes, dtypes, settings or {}, family)
    if not index_mod._index_present(fam_dir):
        return _tier2(
            family, f"lookup: no index bundle for family {family}; Tier 2 miss"
        )

    exact = index_mod.exact_map_for(cfg, family)
    hit = exact.get(key)
    if hit and hit.get("tier0_eligible"):
        return {
            "tier": 0,
            "family": family,
            "best_config": hit.get("best_config"),
            "best_config_id": hit.get("best_config_id"),
            "run_id": hit.get("run_id"),
            "ref": hit.get("ref"),
        }

    index_loader = _index_loader or index_mod.load_index
    vs = index_loader(cfg, family)
    hits = vs.similarity_search_with_score(kernel_source.strip(), k=k)
    if not hits:
        return _tier2(family)

    top_score = hits[0][1]
    threshold = _sim_threshold()
    if top_score < threshold:
        return _tier2(
            family,
            f"lookup: top-1 similarity {top_score:.4f} < {threshold} for {family}; Tier 2",
        )

    neighbors = [
        {
            "kernel_name": doc.metadata.get("kernel_name"),
            "input_shapes": doc.metadata.get("input_shapes"),
            "dtypes": doc.metadata.get("dtypes"),
            "top_n": doc.metadata.get("top_n"),
            "ref": doc.metadata.get("ref"),
            "score": float(score),
        }
        for doc, score in hits
    ]
    return {"tier": 1, "family": family, "neighbors": neighbors}


def lookup(
    kernel_source: str,
    shapes: str,
    dtypes: str,
    hardware: str,
    settings: dict | None = None,
    k: int = 8,
    cfg=None,
    *,
    _index_loader=None,
    _resolve_family_fn=None,
) -> dict:
    """Run Tier-0 exact then Tier-1 similarity lookup."""
    return _lookup_impl(
        kernel_source,
        shapes,
        dtypes,
        hardware,
        settings=settings,
        k=k,
        cfg=cfg or _config(),
        _index_loader=_index_loader,
        _resolve_family_fn=_resolve_family_fn,
    )


# Making module callable so helion_rag.lookup(...) works.
class _CallableModule(types.ModuleType):
    def __call__(self, *args, **kwargs):
        return lookup(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
