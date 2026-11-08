"""Tests of scripts/autoreview.py's reviewer selection, with fake ``claude``
and ``codex`` executables on PATH."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import types

pytestmark = pytest.mark.cpu_only

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "autoreview.py"


def _load() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("autoreview", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


autoreview = _load()

FAKE_CLAUDE = f"""#!{sys.executable}
import json
print(json.dumps({{"result": "claude says LGTM", "session_id": "s1", "is_error": False}}))
"""
FAKE_CODEX = f"""#!{sys.executable}
import sys
args = sys.argv[1:]
with open(args[args.index("-o") + 1], "w") as f:
    f.write("codex says LGTM")
"""


@pytest.fixture
def review(tmp_path, monkeypatch, capsys):
    """Run autoreview.main with only the named fake reviewers on PATH; returns
    (exit status, stdout, stderr)."""
    monkeypatch.delenv(autoreview.SESSION_MARKER, raising=False)
    monkeypatch.setattr(autoreview, "LOGS_BASE", tmp_path / "logs")
    monkeypatch.setattr(autoreview, "REPO_ROOT", tmp_path)

    def run(*installed: str, args: tuple[str, ...] = ()) -> tuple[int, str, str]:
        bin_dir = tmp_path / ("bin-" + "-".join(installed))
        bin_dir.mkdir(exist_ok=True)
        for name in installed:
            path = bin_dir / name
            path.write_text({"claude": FAKE_CLAUDE, "codex": FAKE_CODEX}[name])
            path.chmod(0o755)
        monkeypatch.setenv("PATH", str(bin_dir))
        monkeypatch.setattr(sys, "argv", ["autoreview.py", "--no-meta", *args, "x.py"])
        code = autoreview.main()
        out = capsys.readouterr()
        return code, out.out, out.err

    return run


def test_both_reviewers(review):
    code, out, err = review("claude", "codex")
    # Codex's review is not "LGTM", so Claude combines the two.
    assert code == 0 and "claude says LGTM" in out and "WARNING" not in err


def test_codex_missing_runs_claude_alone_with_a_warning(review):
    code, out, err = review("claude")
    assert code == 0 and out.strip() == "claude says LGTM"
    assert "WARNING: the codex CLI is not installed, so only Claude reviewed" in err
    assert "Tell the user" in err


def test_claude_missing_runs_codex_alone_with_a_warning(review):
    code, out, err = review("codex")
    assert code == 0 and out.strip() == "codex says LGTM"
    assert "WARNING: the claude CLI is not installed, so only Codex reviewed" in err
    assert "resume session" not in err


def test_no_codex_is_a_choice_not_a_warning(review):
    code, out, err = review("claude", args=("--no-codex",))
    assert code == 0 and "WARNING" not in err


def test_neither_reviewer_fails(review):
    code, _, err = review()
    assert code == 2 and "neither the claude nor the codex CLI is installed" in err
    code, _, err = review("codex", args=("--no-codex",))
    assert code == 2 and "--no-codex was given" in err


def test_fix_needs_claude(review):
    with pytest.raises(SystemExit) as info:
        review("codex", args=("--fix",))
    assert info.value.code == 2


def test_logs_record_each_reviewer(review, tmp_path):
    review("codex")
    (log_dir,) = (tmp_path / "logs").iterdir()
    assert (log_dir / "codex-review.txt").read_text() == "codex says LGTM"
    assert not (log_dir / "claude-review.txt").exists()
