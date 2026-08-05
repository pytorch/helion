"""FAISS index with atomic generation swap per hardware family."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import shutil
from typing import TYPE_CHECKING
from typing import Callable
from typing import TypeVar

from helion_rag import safe_index
from helion_rag import signing
from helion_rag._util import INDEX_FILE
from helion_rag._util import _die
from helion_rag._util import _log
from helion_rag.corpus import _dedup_by_key
from helion_rag.corpus import _exact_map
from helion_rag.corpus import _group_by_family
from helion_rag.corpus import _runid_map
from helion_rag.corpus import load_corpus

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

T = TypeVar("T")

_CURRENT = "current"
_GENERATIONS = "generations"
_BUILD_LOCK = ".build.lock"


class GenerationPinError(signing.ArtifactVerificationError):
    """A selected directory does not contain its declared signed generation."""


@dataclass(frozen=True)
class VerifiedFamilyGeneration:
    """Index data and identity from one signature-verified generation."""

    index: safe_index.SafeIndex
    extra_json: dict[str, object]
    generation_dir: Path
    manifest: dict[str, object]


def _device() -> str:
    """Use HELION_RAG_EMBED_DEVICE if set, else pick cuda if available else cpu."""
    if forced := os.environ.get("HELION_RAG_EMBED_DEVICE"):
        return forced
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _embeddings(cfg):
    """HuggingFace embedding model with cosine norm, on chosen device."""
    from langchain_huggingface import HuggingFaceEmbeddings

    if cfg.model_revision != cfg.tokenizer_revision:
        raise ValueError(
            "the embedding loader cannot independently pin different model and "
            "tokenizer revisions"
        )
    model_kwargs = {"device": _device()}
    if cfg.model_revision is not None:
        model_kwargs["revision"] = cfg.model_revision
    return HuggingFaceEmbeddings(
        model_name=cfg.embed_model,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )


def _metadata_for(r: dict) -> dict:
    """Per-row metadata stored beside the vector (replaces the pickled docstore)."""
    return {
        "record_id": r["workload_key"],
        "kernel_name": r["kernel_name"],
        "input_shapes": r["input_shapes"],
        "dtypes": r["dtypes"],
        "family": r["family"],
        "workload_key": r["workload_key"],
        "run_id": r["run_id"],
        "top_n": r["top_n"],
        "ref": asdict(r["ref"]),
    }


def _gens_dir(family_index_dir: Path) -> Path:
    return Path(family_index_dir) / _GENERATIONS


def _next_gen_id(gens_dir: Path) -> str:
    """Next numeric generation id, zero padded to 6 digits."""
    if not gens_dir.is_dir():
        return "000000"
    nums = (int(d.name) for d in gens_dir.iterdir() if d.is_dir() and d.name.isdigit())
    return f"{max(nums, default=-1) + 1:06d}"


@contextmanager
def _generation_lock(family_index_dir: Path) -> Iterator[None]:
    """Reject a concurrent builder for one hardware family.

    ``flock`` is advisory and automatically released if the process exits. All
    generation writers in this package acquire it before building or publishing.
    """
    family_index_dir = Path(family_index_dir)
    family_index_dir.mkdir(parents=True, exist_ok=True)
    with (family_index_dir / _BUILD_LOCK).open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"index build already in progress for {family_index_dir}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _recover_orphaned_generations(family_index_dir: Path) -> None:
    """Recover interrupted publication while holding the family build lock.

    Temporary directories are incomplete and removed. A numeric directory has
    already passed ``populate`` and the atomic final rename, so the newest one is
    promoted when it is newer than ``current`` or the pointer is missing. Numeric
    IDs encode publication order; manually rolling ``current`` back is unsupported.
    """
    family_index_dir = Path(family_index_dir)
    gens = _gens_dir(family_index_dir)
    gens.mkdir(parents=True, exist_ok=True)
    for tmp in gens.glob(".tmp-*"):
        if tmp.is_dir():
            shutil.rmtree(tmp, ignore_errors=True)

    numeric = [d for d in gens.iterdir() if d.is_dir() and d.name.isdigit()]
    if not numeric:
        return
    newest = max(numeric, key=lambda d: int(d.name))
    current = resolve_current(family_index_dir)
    if current is None or int(newest.name) > int(current.name):
        _set_current(family_index_dir, newest.name)
        _log(f"recovered generation {newest.name} under {family_index_dir}")


def _commit_generation_locked(
    family_index_dir: Path, populate: Callable[[Path, str], None]
) -> Path:
    """Publish one generation while the caller holds the family build lock.

    ``populate`` receives the staging directory and the final numeric generation
    id (so a signed manifest can record the id even though files are written into
    a temporary directory before the atomic rename).
    """
    gens = _gens_dir(family_index_dir)
    gens.mkdir(parents=True, exist_ok=True)

    gen_id = _next_gen_id(gens)
    tmp = gens / f".tmp-{os.getpid()}-{gen_id}"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        populate(tmp, gen_id)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    final = gens / gen_id
    os.replace(tmp, final)
    _set_current(family_index_dir, gen_id)
    _gc_old_generations(family_index_dir)
    return final


def commit_generation(
    family_index_dir: Path, populate: Callable[[Path, str], None]
) -> Path:
    """Publish one generation with single-writer locking and crash recovery."""
    family_index_dir = Path(family_index_dir)
    with _generation_lock(family_index_dir):
        _recover_orphaned_generations(family_index_dir)
        return _commit_generation_locked(family_index_dir, populate)


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


def resolve_generation(
    family_index_dir: Path, generation_id: str | None = None
) -> Path | None:
    """Resolve a pinned generation, or ``current`` when no pin is configured."""
    family_index_dir = Path(family_index_dir)
    if generation_id is None:
        return resolve_current(family_index_dir)
    if len(generation_id) != 6 or not generation_id.isdigit():
        raise GenerationPinError(f"invalid generation pin {generation_id!r}")
    generation = _gens_dir(family_index_dir) / generation_id
    if not generation.is_dir():
        raise signing.MissingArtifactError(
            f"pinned generation {generation_id!r} not found under {family_index_dir}"
        )
    return generation


def load_current(family_index_dir: Path, loader: Callable[[Path], T]) -> T:
    """Load from current generation or raise FileNotFoundError."""
    gen = resolve_current(family_index_dir)
    if gen is None:
        raise FileNotFoundError(f"no current generation under {family_index_dir}")
    return loader(gen)


def _gc_old_generations(family_index_dir: Path) -> None:
    """Retain completed generations until a reference-aware policy exists.

    Confirmation manifests and replay ledgers pin numeric generation IDs. Without
    a registry of those references, age-based deletion would make valid signed
    experiments irreproducible.
    """


def _index_present(family_index_dir: Path) -> bool:
    """True if current generation has FAISS index file."""
    gen = resolve_current(Path(family_index_dir))
    return bool(gen and (gen / INDEX_FILE).is_file())


def build_family_index(cfg, family: str, records: list) -> Path:
    """Embed, sign, and publish one family index under the single-writer lock.

    Requires the designated publisher's Ed25519 private key
    (``HELION_RAG_PRIVATE_KEY_PATH``); the winning artifacts are persisted with
    ``faiss.write_index`` + canonical JSON and covered by a signed manifest, never
    pickled.
    """
    import numpy as np

    runids = _runid_map(records)
    records = _dedup_by_key(records)
    if not records:
        _die(f"{family}: no records to index")
    fam_dir = Path(cfg.index_dir) / family
    private_key = signing.load_publisher_private_key()
    with _generation_lock(fam_dir):
        _recover_orphaned_generations(fam_dir)
        emb = _embeddings(cfg)
        vectors = np.asarray(
            emb.embed_documents([r["embed_text"] for r in records]), dtype="float32"
        )
        idx = safe_index.build_safe_index(vectors, [_metadata_for(r) for r in records])
        exact = _exact_map(records)

        def _populate(gen_dir: Path, gen_id: str) -> None:
            safe_index.save_generation(
                gen_dir,
                idx,
                private_key,
                generation_id=gen_id,
                extra_json={"exact.json": exact, "runids.json": runids},
            )

        gen = _commit_generation_locked(fam_dir, _populate)
    _log(
        f"{family}: wrote generation {gen.name} ({len(exact)} workloads, "
        f"{len(runids)} run_ids) to {fam_dir}"
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


def load_verified_family_generation(
    cfg, family: str, public_key: Ed25519PublicKey
) -> VerifiedFamilyGeneration:
    """Verify and load one generation while retaining its trusted identity."""
    fam_dir = Path(cfg.index_dir) / family
    generation = resolve_generation(fam_dir, cfg.generation_id)
    if generation is None:
        raise FileNotFoundError(f"no current generation under {fam_dir}")
    manifest = signing.verify_generation(generation, public_key)
    signed_id = manifest.get("generation_id")
    if signed_id != generation.name:
        raise GenerationPinError(
            f"signed generation identity {signed_id!r} does not match selected "
            f"generation {generation.name!r}"
        )
    index, extra_json = safe_index.load_generation(generation, public_key)
    return VerifiedFamilyGeneration(index, extra_json, generation, manifest)


def load_family_generation(
    cfg, family: str, public_key: Ed25519PublicKey
) -> tuple[safe_index.SafeIndex, dict[str, object]]:
    """Verify and load the configured generation for a family (safe, no pickle).

    Returns ``(SafeIndex, extra_json)`` where ``extra_json`` holds the verified
    ``exact.json`` (Tier-0) and ``runids.json``. Raises an
    :class:`~helion_rag.signing.ArtifactVerificationError` subclass on any
    signature/hash failure so the caller fails closed to baseline.
    """
    loaded = load_verified_family_generation(cfg, family, public_key)
    return loaded.index, loaded.extra_json


def republish_generation(
    cfg,
    family: str,
    source_generation_id: str,
    private_key: Ed25519PrivateKey,
    public_key: Ed25519PublicKey,
) -> Path:
    """Roll forward: republish a known-good generation as a new signed generation.

    Rollback never moves the ``current`` pointer backward (unsupported); instead
    the last-good artifacts are copied into a fresh, higher-numbered generation and
    re-signed under the new id, which then becomes current.
    """
    fam_dir = Path(cfg.index_dir) / family
    src = _gens_dir(fam_dir) / source_generation_id
    if not src.is_dir():
        _die(
            f"{family}: source generation {source_generation_id} not found under {src}"
        )
    manifest = signing.verify_generation(src, public_key)
    signed_id = manifest.get("generation_id")
    if signed_id != source_generation_id or signed_id != src.name:
        raise GenerationPinError(
            f"signed generation identity {signed_id!r} does not match source "
            f"generation {source_generation_id!r}"
        )
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, dict)
    names = list(artifacts)

    def _populate(gen_dir: Path, gen_id: str) -> None:
        for name in names:
            shutil.copy2(src / name, gen_dir / name)
        signing.write_signed_manifest(gen_id, gen_dir, names, private_key)

    with _generation_lock(fam_dir):
        _recover_orphaned_generations(fam_dir)
        return _commit_generation_locked(fam_dir, _populate)
