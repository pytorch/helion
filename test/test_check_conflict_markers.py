"""The commit hook checks file contents without depending on Git merge state."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_conflict_markers.py"
spec = importlib.util.spec_from_file_location("check_conflict_markers", SCRIPT)
assert spec is not None and spec.loader is not None
conflicts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conflicts)


@pytest.mark.parametrize("length", [7, 12])
def test_conflict_markers_without_git_state(tmp_path, capsys, length):
    path = tmp_path / "outside-a-repository.txt"
    path.write_text(
        "<" * length
        + " HEAD\nleft\n"
        + "|" * length
        + " base\nancestor\n"
        + "=" * length
        + "\nright\n"
        + ">" * length
        + " incoming\n"
    )
    assert conflicts.main([str(path)]) == 1
    assert capsys.readouterr().out.splitlines() == [
        f"{path}:{line}: unresolved conflict marker" for line in (1, 3, 7)
    ]


def test_headings_and_quoted_marker_examples_are_not_conflicts(tmp_path, capsys):
    path = tmp_path / "notes.rst"
    path.write_text('Example\n=======\nTesting\n=======\n"<<<<<<< HEAD"\n')
    assert conflicts.main([str(path)]) == 0
    assert not capsys.readouterr().out


@pytest.mark.parametrize("marker", ["<<<<<<<", "|||||||", ">>>>>>>"])
def test_partial_conflict_marker_is_still_reported(tmp_path, capsys, marker):
    path = tmp_path / "partial.txt"
    path.write_bytes((marker + "\r\n").encode())
    assert conflicts.main([str(path)]) == 1
    assert f"{path}:1:" in capsys.readouterr().out
