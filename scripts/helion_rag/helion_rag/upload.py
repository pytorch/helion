"""Package local autotune runs for upload, tracking what's already sent."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from helion_rag._util import _log
from helion_rag.corpus import _parse_record


def select_unuploaded_runs(
    all_run_ids: list[str], uploaded_runs_dir: Path | str, reupload: bool = False
) -> list[str]:
    """Return run IDs not yet marked uploaded, preserving order."""
    uploaded_runs_dir = Path(uploaded_runs_dir)
    uniq = list(dict.fromkeys(all_run_ids))
    if reupload:
        return uniq
    return [r for r in uniq if not (uploaded_runs_dir / f"{r}.json").is_file()]


def build_batch_manifest(run_ids: list[str], family: str, contributor: str) -> dict:
    """Small JSON describing what's in the zip."""
    return {"family": family, "contributor": contributor, "run_ids": list(run_ids)}


def record_upload(
    run_ids: list[str],
    archive_sha256: str,
    uploaded_runs_dir: Path | str,
    uploaded_archives_dir: Path | str,
) -> None:
    """Write markers so we don't re-upload same runs."""
    runs_dir, arch_dir = Path(uploaded_runs_dir), Path(uploaded_archives_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    arch_dir.mkdir(parents=True, exist_ok=True)

    payload = json.dumps
    for rid in run_ids:
        (runs_dir / f"{rid}.json").write_text(
            payload({"run_id": rid, "archive_sha256": archive_sha256}), encoding="utf-8"
        )
    (arch_dir / f"{archive_sha256}.json").write_text(
        payload({"archive_sha256": archive_sha256, "run_ids": list(run_ids)}),
        encoding="utf-8",
    )


def _records_from_logs(log_dir: Path | str) -> list[tuple[Path, list[dict]]]:
    log_dir = Path(log_dir)
    files = []
    for path in sorted(log_dir.rglob("*.meta.jsonl")):
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"{path}: metadata record must be an object")
            run_id = record["run_id"]
            if not isinstance(run_id, str) or not run_id:
                raise ValueError(f"{path}: run_id must be a non-empty string")
            records.append(record)
        files.append((path, records))
    return files


def upload(
    *,
    autotune_log_dir: Path | str,
    uploads_dir: Path | str,
    family: str,
    contributor: str,
    manifold_put=None,
    dry_run: bool = False,
    reupload: bool = False,
) -> dict:
    """Zip new runs and optionally call manifold_put. Dry run just reports."""
    autotune_log_dir, uploads_dir = Path(autotune_log_dir), Path(uploads_dir)
    runs_dir = uploads_dir / "uploaded-runs"
    archives_dir = uploads_dir / "uploaded-archives"

    log_records = _records_from_logs(autotune_log_dir)
    for path, records in log_records:
        for record in records:
            if record["configs"]:
                _parse_record(record, family, path.name)
    all_run_ids = list(
        dict.fromkeys(
            record["run_id"] for _, records in log_records for record in records
        )
    )
    todo = select_unuploaded_runs(all_run_ids, runs_dir, reupload)
    manifest = build_batch_manifest(todo, family, contributor)
    base = {"run_ids": todo, "manifest": manifest}

    if dry_run:
        _log(f"upload --dry-run: {len(todo)} unuploaded run(s) for {family}: {todo}")
        return {**base, "dry_run": True}
    if not todo:
        _log("upload: nothing to upload")
        return {**base, "run_ids": []}

    uploads_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=uploads_dir, prefix=".contrib-", suffix=".zip", delete=False
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("batch-manifest.json", json.dumps(manifest, indent=2))
            todo_set = set(todo)
            for path, records in log_records:
                selected = [
                    record for record in records if record["run_id"] in todo_set
                ]
                if selected:
                    data = "".join(f"{json.dumps(record)}\n" for record in selected)
                    zf.writestr(str(path.relative_to(autotune_log_dir)), data)

        sha = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
        archive_path = uploads_dir / f"contrib-{family}-{sha}.zip"
        if archive_path.exists():
            tmp_path.unlink()
        else:
            tmp_path.replace(archive_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    res = {**base, "archive_sha256": sha, "archive_path": str(archive_path)}

    if not manifold_put:
        _log(
            f"upload: built {archive_path.name} but no manifold_put; not marking uploaded"
        )
        return {**res, "run_ids": [], "uploaded": False}

    manifold_put(
        archive_path, f"contrib/{family}/{contributor}/{sha[:12]}/{archive_path.name}"
    )
    record_upload(todo, sha, runs_dir, archives_dir)
    _log(
        f"upload: contributed {len(todo)} run(s) as {archive_path.name} (sha {sha[:12]})"
    )
    return {**res, "uploaded": True}
