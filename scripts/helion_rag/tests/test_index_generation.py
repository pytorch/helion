"""Generation publication policy and crash recovery."""

from __future__ import annotations

from pathlib import Path

import helion_rag.index as index_mod
import pytest


def test_generation_lock_rejects_concurrent_builder(tmp_path: Path) -> None:
    family_dir = tmp_path / "h100"

    with index_mod._generation_lock(family_dir):
        with pytest.raises(RuntimeError, match="already in progress"):
            with index_mod._generation_lock(family_dir):
                pass


def test_recover_orphaned_generation_and_temp_dir(tmp_path: Path) -> None:
    family_dir = tmp_path / "h100"
    generations = family_dir / "generations"
    (generations / "000000").mkdir(parents=True)
    (generations / "000001").mkdir()
    stale_tmp = generations / ".tmp-1234-000002"
    stale_tmp.mkdir()
    (family_dir / "current").write_text("000000\n", encoding="utf-8")

    index_mod._recover_orphaned_generations(family_dir)

    assert not stale_tmp.exists()
    assert (family_dir / "current").read_text(encoding="utf-8") == "000001\n"
