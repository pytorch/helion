"""Safe FAISS persistence: ``faiss.read_index`` + signed, hash-verified JSON (§3.2).

This replaces the unsafe ``FAISS.load_local(..., allow_dangerous_deserialization=True)``
path (which unpickles a docstore). A generation directory holds:

* ``index.faiss`` — raw vectors, written/read with ``faiss.write_index`` /
  ``faiss.read_index`` (no pickle).
* ``metadata.json`` — canonical JSON, one entry per row, replacing the pickled
  docstore.
* optional extra JSON artifacts (e.g. ``exact.json``, ``runids.json``).
* ``manifest.json`` + ``manifest.sig`` — an Ed25519-signed hash manifest over all
  of the above (see :mod:`helion_rag.signing`).

Loading verifies the signature and every artifact hash *before* reading any
vector data; any failure raises a typed
:class:`~helion_rag.signing.ArtifactVerificationError` so the caller fails closed
to baseline tuning. FAISS and numpy are imported lazily so importing this module
stays cheap.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from . import signing

if TYPE_CHECKING:
    import numpy as np
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"
_RESERVED = {INDEX_FILE, METADATA_FILE, signing.MANIFEST_NAME, signing.SIGNATURE_NAME}


@dataclasses.dataclass
class SafeIndex:
    """A raw FAISS inner-product index plus per-row metadata."""

    index: Any  # faiss.Index (faiss ships no type stubs)
    metadata: list[dict]

    def search(
        self, query_vec: np.ndarray | Sequence[float], k: int
    ) -> list[tuple[float, dict]]:
        """Return up to ``k`` ``(cosine_score, metadata)`` pairs, best first."""
        import faiss
        import numpy as np

        if self.index.ntotal == 0:
            return []
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q)
        scores, rows = self.index.search(q, min(k, self.index.ntotal))
        out: list[tuple[float, dict]] = []
        for score, row in zip(scores[0], rows[0]):
            if row < 0:
                continue
            out.append((float(score), self.metadata[int(row)]))
        return out


def build_safe_index(vectors: np.ndarray, metadata: list[dict]) -> SafeIndex:
    """Build a cosine (normalized inner-product) index from row vectors."""
    import faiss
    import numpy as np

    vecs = np.asarray(vectors, dtype="float32")
    if vecs.ndim != 2:
        raise ValueError(f"vectors must be 2-D, got shape {vecs.shape}")
    if len(metadata) != vecs.shape[0]:
        raise ValueError(
            f"metadata length {len(metadata)} != vector count {vecs.shape[0]}"
        )
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return SafeIndex(index=index, metadata=list(metadata))


def save_generation(
    gen_dir: Path,
    safe_index: SafeIndex,
    private_key: Ed25519PrivateKey,
    *,
    generation_id: str | None = None,
    extra_json: dict[str, object] | None = None,
) -> dict:
    """Write and sign a generation directory; return the manifest.

    ``generation_id`` defaults to the directory name; pass it explicitly when the
    files are written into a temporary staging directory before an atomic rename.
    Extra JSON artifacts (e.g. ``exact.json``) are written canonically and covered
    by the signed manifest.
    """
    import faiss

    gen_dir = Path(gen_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(safe_index.index, str(gen_dir / INDEX_FILE))
    (gen_dir / METADATA_FILE).write_bytes(
        signing.canonical_json_bytes(safe_index.metadata)
    )
    names = [INDEX_FILE, METADATA_FILE]
    for name, obj in (extra_json or {}).items():
        if name in _RESERVED:
            raise ValueError(f"extra artifact name {name!r} is reserved")
        (gen_dir / name).write_bytes(signing.canonical_json_bytes(obj))
        names.append(name)
    gid = generation_id if generation_id is not None else gen_dir.name
    return signing.write_signed_manifest(gid, gen_dir, names, private_key)


def load_generation(
    gen_dir: Path, public_key: Ed25519PublicKey
) -> tuple[SafeIndex, dict[str, object]]:
    """Verify then load a generation: returns ``(SafeIndex, extra_json)``.

    Verification (signature + every artifact hash) happens before any vector data
    is read; a failure raises an ``ArtifactVerificationError`` subclass.
    """
    import faiss

    gen_dir = Path(gen_dir)
    manifest = signing.verify_generation(gen_dir, public_key)
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, dict)
    for required in (INDEX_FILE, METADATA_FILE):
        if required not in artifacts:
            raise signing.MissingArtifactError(
                f"required artifact {required!r} is absent from the signed manifest"
            )
    try:
        index = faiss.read_index(str(gen_dir / INDEX_FILE))
    except RuntimeError as exc:
        raise signing.ArtifactFormatError("malformed FAISS index artifact") from exc
    metadata_payload = (gen_dir / METADATA_FILE).read_bytes()
    try:
        metadata = json.loads(metadata_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise signing.ArtifactFormatError(
            "metadata artifact is not valid JSON"
        ) from exc
    if not isinstance(metadata, list) or not all(
        isinstance(row, dict) for row in metadata
    ):
        raise signing.ArtifactFormatError(
            "metadata artifact must be a list of JSON objects"
        )
    if metadata_payload != signing.canonical_json_bytes(metadata):
        raise signing.ArtifactFormatError("metadata artifact JSON is not canonical")
    if index.ntotal != len(metadata):
        raise signing.ArtifactFormatError(
            f"FAISS vector count {index.ntotal} does not match metadata row count "
            f"{len(metadata)}"
        )
    extra: dict[str, object] = {}
    for name in artifacts:
        if name in _RESERVED:
            continue
        payload = (gen_dir / name).read_bytes()
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise signing.ArtifactFormatError(
                f"extra artifact {name!r} is not valid JSON"
            ) from exc
        if payload != signing.canonical_json_bytes(value):
            raise signing.ArtifactFormatError(
                f"extra artifact {name!r} JSON is not canonical"
            )
        extra[name] = value
    return SafeIndex(index=index, metadata=metadata), extra
