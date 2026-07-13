"""FAISS index with atomic generation swap per hardware family."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
from typing import Callable
from typing import TypeVar

from helion_rag._util import INDEX_FILE
from helion_rag._util import _die
from helion_rag._util import _log
from helion_rag.corpus import _dedup_by_key
from helion_rag.corpus import _exact_map
from helion_rag.corpus import _group_by_family
from helion_rag.corpus import _runid_map
from helion_rag.corpus import load_corpus

T = TypeVar("T")

_CURRENT = "current"
_GENERATIONS = "generations"
_KEEP_GENERATIONS = 2


def _device() -> str:
    """Use HELION_RAG_EMBED_DEVICE if set, else pick cuda if available else cpu."""
    if forced := os.environ.get("HELION_RAG_EMBED_DEVICE"):
        return forced
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _embeddings(cfg):
    """HuggingFace embedding model with cosine norm, on chosen device."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=cfg.embed_model,
        model_kwargs={"device": _device()},
        encode_kwargs={"normalize_embeddings": True},
    )


def _to_documents(records: list):
    """Turn corpus records into Documents with metadata needed for Tier-1."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content=r["embed_text"],
            metadata={
                "record_id": r["workload_key"],
                "kernel_name": r["kernel_name"],
                "input_shapes": r["input_shapes"],
                "dtypes": r["dtypes"],
                "family": r["family"],
                "workload_key": r["workload_key"],
                "run_id": r["run_id"],
                "top_n": r["top_n"],
                "ref": asdict(r["ref"]),
            },
        )
        for r in records
    ]


def _gens_dir(family_index_dir: Path) -> Path:
    return Path(family_index_dir) / _GENERATIONS


def _next_gen_id(gens_dir: Path) -> str:
    """Next numeric generation id, zero padded to 6 digits."""
    if not gens_dir.is_dir():
        return "000000"
    nums = (int(d.name) for d in gens_dir.iterdir() if d.is_dir() and d.name.isdigit())
    return f"{max(nums, default=-1) + 1:06d}"


def commit_generation(family_index_dir: Path, populate: Callable[[Path], None]) -> Path:
    """Write a new generation to temp dir, then atomically swap current pointer."""
    family_index_dir = Path(family_index_dir)
    gens = _gens_dir(family_index_dir)
    gens.mkdir(parents=True, exist_ok=True)

    gen_id = _next_gen_id(gens)
    tmp = gens / f".tmp-{os.getpid()}-{gen_id}"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        populate(tmp)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    final = gens / gen_id
    os.replace(tmp, final)
    _set_current(family_index_dir, gen_id)
    _gc_old_generations(family_index_dir)
    return final


def _set_current(family_index_dir: Path, gen_id: str) -> None:
    """Write current file atomically."""
    ptr = Path(family_index_dir) / _CURRENT
    tmp = Path(family_index_dir) / f".{_CURRENT}.tmp"
    tmp.write_text(f"{gen_id}\n", encoding="utf-8")
    os.replace(tmp, ptr)


def resolve_current(family_index_dir: Path) -> Path | None:
    """Return path to current generation, or None if missing."""
    ptr = Path(family_index_dir) / _CURRENT
    if not ptr.is_file():
        return None
    gen_id = ptr.read_text(encoding="utf-8").strip()
    if not gen_id:
        return None
    gen = _gens_dir(family_index_dir) / gen_id
    return gen if gen.is_dir() else None


def load_current(family_index_dir: Path, loader: Callable[[Path], T]) -> T:
    """Load from current generation or raise FileNotFoundError."""
    gen = resolve_current(family_index_dir)
    if gen is None:
        raise FileNotFoundError(f"no current generation under {family_index_dir}")
    return loader(gen)


def _gc_old_generations(family_index_dir: Path) -> None:
    """Keep only newest 2 generations plus current."""
    gens = _gens_dir(family_index_dir)
    if not gens.is_dir():
        return
    cur = resolve_current(family_index_dir)
    numeric = sorted(
        (d for d in gens.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
        reverse=True,
    )
    keep = set(numeric[:_KEEP_GENERATIONS])
    if cur:
        keep.add(cur)
    for d in numeric[_KEEP_GENERATIONS:]:
        if d not in keep:
            shutil.rmtree(d, ignore_errors=True)


def _index_present(family_index_dir: Path) -> bool:
    """True if current generation has FAISS index file."""
    gen = resolve_current(Path(family_index_dir))
    return bool(gen and (gen / INDEX_FILE).is_file())


def build_family_index(cfg, family: str, records: list) -> Path:
    """Build FAISS vector store and exact maps for one family, then swap current."""
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    runids = _runid_map(records)
    records = _dedup_by_key(records)
    emb = _embeddings(cfg)
    vs = FAISS.from_documents(
        _to_documents(records),
        emb,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    exact = _exact_map(records)

    def _populate(gen_dir: Path) -> None:
        vs.save_local(str(gen_dir))
        (gen_dir / "exact.json").write_text(json.dumps(exact), encoding="utf-8")
        (gen_dir / "runids.json").write_text(json.dumps(runids), encoding="utf-8")

    fam_dir = Path(cfg.index_dir) / family
    gen = commit_generation(fam_dir, _populate)
    _log(
        f"{family}: wrote generation {gen.name} ({len(exact)} workloads, {len(runids)} run_ids) to {fam_dir}"
    )
    return gen


def build_index(cfg=None, force: bool = False) -> None:
    """Build indexes for all families found in corpus and writeback."""
    from helion_rag.config import _config

    cfg = cfg or _config()
    records = load_corpus(cfg.corpus_dir)
    records += load_corpus(cfg.writeback_dir, required=False)
    by_family = _group_by_family(records)
    Path(cfg.index_dir).mkdir(parents=True, exist_ok=True)
    for family, fam_records in sorted(by_family.items()):
        fam_dir = Path(cfg.index_dir) / family
        if _index_present(fam_dir) and not force:
            _log(f"{family}: index exists, use --force to rebuild; skipping")
            continue
        _log(f"{family}: embedding {len(fam_records)} records …")
        build_family_index(cfg, family, fam_records)


def load_index(cfg, family: str):
    """Load FAISS store for family from current generation."""
    from langchain_community.vectorstores import FAISS

    fam_dir = Path(cfg.index_dir) / family
    if not _index_present(fam_dir):
        _die(
            f"no index for family {family} under {fam_dir}; run `python -m helion_rag index`."
        )

    def _loader(gen_dir: Path):
        return FAISS.load_local(
            str(gen_dir),
            _embeddings(cfg),
            allow_dangerous_deserialization=True,
        )

    return load_current(fam_dir, _loader)


def exact_map_for(cfg, family: str) -> dict:
    """Load exact.json map for Tier-0 exact match."""
    fam_dir = Path(cfg.index_dir) / family

    def _loader(gen_dir: Path) -> dict:
        return json.loads((gen_dir / "exact.json").read_text(encoding="utf-8"))

    return load_current(fam_dir, _loader)
