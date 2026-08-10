"""Read HELION_RAG_* env vars into a simple Config object."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from helion_rag._util import DEFAULT_EMBED_MODEL

_HOME_RAG = Path.home() / "helion-rag"


@dataclass
class Config:
    """Paths, hardware family, and model name for a RAG run."""

    embed_model: str
    data_dir: Path
    index_dir: Path
    writeback_dir: Path
    manifold_base: str = ""
    manifest_path: Path | None = None
    hardware_family: str | None = None
    autotune_log_dir: Path | None = None
    uploads_dir: Path | None = None

    @property
    def corpus_dir(self) -> Path:
        """Where extracted corpus lives under data_dir."""
        return self.data_dir / "corpus"


def _config() -> Config:
    """Build Config from env, using defaults under ~/helion-rag."""

    def p(name: str, default: Path) -> Path:
        v = os.environ.get(name)
        return Path(v) if v else default

    def opt(name: str) -> Path | None:
        v = os.environ.get(name)
        return Path(v) if v else None

    return Config(
        embed_model=os.environ.get("HELION_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        data_dir=p("HELION_RAG_DATA_DIR", _HOME_RAG / "ci_artifacts"),
        index_dir=p("HELION_RAG_INDEX_DIR", _HOME_RAG / "rag_index"),
        writeback_dir=p("HELION_RAG_WRITEBACK_DIR", _HOME_RAG / "rag_writeback"),
        manifold_base=os.environ.get("HELION_RAG_MANIFOLD_BASE", ""),
        manifest_path=opt("HELION_RAG_MANIFEST"),
        hardware_family=os.environ.get("HELION_RAG_HARDWARE_FAMILY") or None,
        autotune_log_dir=opt("HELION_RAG_AUTOTUNE_LOG_DIR"),
        uploads_dir=opt("HELION_RAG_UPLOADS_DIR"),
    )
