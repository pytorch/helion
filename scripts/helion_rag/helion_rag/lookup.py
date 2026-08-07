"""Tiered lookup: exact match, then similar kernels, then miss."""

from __future__ import annotations

import os
from pathlib import Path

from helion_rag import signing
from helion_rag._util import _log
from helion_rag._util import _sim_threshold
from helion_rag.config import _config
import helion_rag.corpus as corpus
import helion_rag.hardware as hardware
import helion_rag.index as index_mod
from helion_rag.embedding_text import query_text
from helion_rag.manifest import load_manifest
from helion_rag.shape_distance import shape_distance
from helion_rag.shape_distance import shape_relevance

# Semantic over-fetch depth before shape reranking. Measured on the published
# 46-workload H100 study: the deepest pre-rerank rank reranking ever promoted
# into the final top-3 was 39 (p50 3, p95 21), so 64 leaves ~1.6x headroom. It
# does bind -- 8 of those workloads had more than 64 candidates above the
# similarity threshold -- so revisit this if the corpus grows well beyond the
# ~150 workloads per hardware family it was measured at.
_TIER1_POOL = 64


def _load_manifest_opt(cfg) -> dict | None:
    """Load manifest if configured, else None."""
    return load_manifest(cfg.manifest_path) if cfg.manifest_path else None


def _resolve_family(hardware_str: str, cfg) -> str | None:
    return hardware.resolve_family(
        device=hardware_str,
        manifest=_load_manifest_opt(cfg),
        env_family=cfg.hardware_family,
    )


def _tier2(
    family: str | None = None,
    msg: str | None = None,
    artifact_identity: dict[str, str] | None = None,
) -> dict:
    if msg:
        _log(msg)
    result: dict[str, object] = {"tier": 2, "family": family}
    if artifact_identity is not None:
        result["artifact_identity"] = artifact_identity
    return result


def _artifact_identity(
    loaded: index_mod.VerifiedFamilyGeneration,
    embed_model: str,
) -> dict[str, str]:
    artifacts = loaded.manifest["artifacts"]
    assert isinstance(artifacts, dict)
    return {
        "index_id": loaded.generation_dir.name,
        "manifest_id": signing.sha256_hex(
            signing.canonical_json_bytes(loaded.manifest)
        ),
        "corpus_id": str(artifacts["metadata.json"]),
        "model_id": embed_model,
        "tokenizer_id": embed_model,
    }


def _distinct_kernel_neighbors(neighbors: list[dict], k: int) -> list[dict]:
    """Keep the best-ranked workload for each distinct kernel type."""
    selected = []
    seen = set()
    for neighbor in neighbors:
        kernel_name = neighbor.get("kernel_name")
        if kernel_name in seen:
            continue
        seen.add(kernel_name)
        selected.append(neighbor)
        if len(selected) >= k:
            break
    return selected


def lookup(
    kernel_source: str,
    shapes: str,
    dtypes: str,
    hardware: str,
    settings: dict | None = None,
    kernel_name: str = "",
    k: int | None = None,
    cfg=None,
    propagate_artifact_errors: bool = False,
    require_generation_pin: bool = False,
) -> dict:
    """Try Tier-0 exact match, then Tier-1 vector similarity, else Tier-2 miss."""
    cfg = cfg or _config()
    k = k if k is not None else int(os.environ.get("HELION_RAG_K", "8"))
    family = _resolve_family(hardware, cfg)
    if family is None:
        return _tier2(None, f"lookup: unrecognized hardware {hardware!r}; Tier 2 miss")
    if require_generation_pin and cfg.generation_id is None:
        raise index_mod.GenerationPinError(
            "live lookup requires an explicit generation pin"
        )

    fam_dir = Path(cfg.index_dir) / family
    if cfg.generation_id is None and not index_mod._index_present(fam_dir):
        return _tier2(
            family, f"lookup: no index bundle for family {family}; Tier 2 miss"
        )

    # Verify the signed generation before reading any artifact; fail closed to
    # Tier-2 on any missing key / signature / hash failure.
    try:
        public_key = signing.load_trusted_public_key()
        loaded = index_mod.load_verified_family_generation(cfg, family, public_key)
    except signing.ArtifactVerificationError as exc:
        if propagate_artifact_errors:
            raise
        return _tier2(
            family, f"lookup: artifact verification failed for {family}: {exc}"
        )
    safe_idx = loaded.index
    extra = loaded.extra_json
    artifact_identity = _artifact_identity(loaded, cfg.embed_model)

    key = corpus._workload_key(kernel_source, shapes, dtypes, settings or {}, family)
    exact_obj = extra.get("exact.json")
    exact = exact_obj if isinstance(exact_obj, dict) else {}
    hit = exact.get(key)
    if hit and hit.get("tier0_eligible"):
        return {
            "tier": 0,
            "family": family,
            "best_config": hit.get("best_config"),
            "best_config_id": hit.get("best_config_id"),
            "run_id": hit.get("run_id"),
            "ref": hit.get("ref"),
            "tier0_identity_combo": hit.get("tier0_identity_combo"),
            "tier0_collision_count": hit.get("tier0_collision_count"),
            "artifact_identity": artifact_identity,
        }

    # Over-fetch a pool rather than k: shape reranking below reorders the
    # semantic hits, so the final top-k can come from anywhere in the pool.
    query_vec = index_mod._embeddings(cfg).embed_query(
        query_text(kernel_source, shapes, dtypes, kernel_name, cfg.embed_text)
    )
    hits = safe_idx.search(query_vec, _TIER1_POOL)
    if not hits:
        return _tier2(family, artifact_identity=artifact_identity)

    top_score = hits[0][0]
    threshold = _sim_threshold()
    if top_score < threshold:
        return _tier2(
            family,
            f"lookup: top-1 similarity {top_score:.4f} < {threshold} for {family}; Tier 2",
            artifact_identity=artifact_identity,
        )

    neighbors = []
    for score, md in hits:
        if score < threshold:
            continue
        distance = shape_distance(shapes, md.get("input_shapes") or "")
        shape_score = shape_relevance(distance)
        neighbors.append(
            {
                "kernel_name": md.get("kernel_name"),
                "input_shapes": md.get("input_shapes"),
                "dtypes": md.get("dtypes"),
                "top_n": md.get("top_n"),
                "ref": md.get("ref"),
                "score": float(score),
                "shape_distance": distance,
                "relevance": (
                    float(score) * shape_score if shape_score else float(score)
                ),
            }
        )
    if os.environ.get("HELION_RAG_SHAPE_RERANK", "0") == "1":
        neighbors.sort(
            key=lambda neighbor: (
                neighbor["shape_distance"],
                -neighbor["score"],
                (neighbor.get("top_n") or [{"median": float("inf")}])[0]["median"],
            )
        )
    selected = (
        _distinct_kernel_neighbors(neighbors, k)
        if os.environ.get("HELION_RAG_DISTINCT_KERNELS") == "1"
        else neighbors[:k]
    )
    return {
        "tier": 1,
        "family": family,
        "neighbors": selected,
        "artifact_identity": artifact_identity,
    }
