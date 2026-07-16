"""CLI entry point for python -m helion_rag. Dispatches to subcommands."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import sys

from helion_rag.config import _config
import helion_rag.lookup as lookup
import helion_rag.patch as patch


def _contributor() -> str:
    """Who ran upload, for attribution in manifest."""
    return getpass.getuser()


def _manifold_put(manifold_base: str):
    """Return a function that runs `manifold put`."""
    import subprocess

    def _put(archive_path: Path, dest: str) -> None:
        subprocess.run(
            ["manifold", "put", str(archive_path), f"{manifold_base}/{dest}"],
            check=True,
        )

    return _put


def _manifold_get_file(src: str, dest: str) -> bool:
    """Run `manifold get`; return True on success, False if the path is absent."""
    import subprocess

    return subprocess.run(["manifold", "get", src, dest]).returncode == 0


def _manifold_put_file(src: str, dest: str) -> None:
    """Run `manifold put` to a full destination path."""
    import subprocess

    subprocess.run(["manifold", "put", src, dest], check=True)


def _load_env_file(path: str | Path) -> None:
    """Parse simple shell export lines into os.environ."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        # strip quotes
        if (v.startswith("'") and v.endswith("'")) or (
            v.startswith('"') and v.endswith('"')
        ):
            v = v[1:-1]
        os.environ.setdefault(k, v)


def main(argv: list[str] | None = None) -> int:
    """Parse args and run extract, index, lookup, ingest or upload."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file")
    pre_args, remaining = pre.parse_known_args(argv)
    if pre_args.env_file:
        _load_env_file(pre_args.env_file)
    elif os.environ.get("HELION_RAG_ENV_FILE"):
        _load_env_file(os.environ["HELION_RAG_ENV_FILE"])

    p = argparse.ArgumentParser(
        prog="helion_rag", description="Helion RAG over CI artifacts."
    )
    p.add_argument("--env-file", help="Path to env.sh to load")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("extract", help="unzip benchmark zips into corpus tree")
    pi = sub.add_parser("index", help="build per-family generation index")
    pi.add_argument("--force", action="store_true")

    pl = sub.add_parser("lookup", help="tiered autotuning-assist lookup")
    pl.add_argument("--kernel-source-file", required=True)
    pl.add_argument("--shapes", required=True)
    pl.add_argument("--dtypes", required=True)
    pl.add_argument("--hardware", required=True)
    pl.add_argument("--k", type=int, default=5)
    pl.add_argument("--settings-json")

    sub.add_parser("ingest", help="ingest autotune logs into writeback")
    pu = sub.add_parser("upload", help="upload unuploaded runs to Manifold")
    pu.add_argument("--dry-run", action="store_true")
    pu.add_argument("--reupload", action="store_true")

    pp = sub.add_parser(
        "patch-helion", help="append RAG hook to Helion runtime kernel.py"
    )
    pp.add_argument(
        "--target",
        default="helion/runtime/kernel.py",
        help="Path to kernel.py to patch",
    )

    sh = sub.add_parser("setup-helper", help="helpers for setup script")
    sh_sub = sh.add_subparsers(dest="action", required=True)
    ci = sh_sub.add_parser("check-import", help="check helion import matches repo")
    ci.add_argument("--repo", required=True)
    sy = sh_sub.add_parser("synthesize", help="synthesize manifest from families")
    sy.add_argument("--out", required=True)
    sy.add_argument("families", nargs="+")
    va = sh_sub.add_parser("validate", help="validate manifest")
    va.add_argument("--manifest", required=True)
    rf = sh_sub.add_parser("resolve-family", help="resolve hardware family")
    rf.add_argument("--manifest", required=True)
    rf.add_argument("--device")
    ir = sh_sub.add_parser("is-represented", help="check family in manifest")
    ir.add_argument("--manifest", required=True)
    ir.add_argument("--family", required=True)
    ap = sh_sub.add_parser("artifact-path", help="get artifact path for family")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--family", required=True)
    pm = sh_sub.add_parser(
        "publish-manifest", help="register a family in the shared manifest"
    )
    pm.add_argument("--manifold-base", required=True)
    pm.add_argument("--family", required=True)
    pm.add_argument("--artifact-path")
    pm.add_argument(
        "--alias", action="append", default=[], help="extra alias, repeatable"
    )

    args = p.parse_args(argv)
    if args.env_file:
        _load_env_file(args.env_file)
    cfg = _config()

    if args.cmd == "extract":
        from helion_rag.corpus import extract_corpus

        n = extract_corpus(cfg.data_dir, cfg.corpus_dir)
        print(f"extracted {n} corpus file(s) into {cfg.corpus_dir}", file=sys.stderr)
        return 0

    if args.cmd == "index":
        from helion_rag.index import build_index

        build_index(cfg, force=args.force)
        return 0

    if args.cmd == "lookup":
        src = Path(args.kernel_source_file).read_text(encoding="utf-8")
        settings = json.loads(args.settings_json) if args.settings_json else None
        res = lookup(
            src,
            args.shapes,
            args.dtypes,
            args.hardware,
            settings=settings,
            k=args.k,
            cfg=cfg,
        )
        print(f"tier {res.get('tier')}", file=sys.stderr)
        print(json.dumps(res, indent=2, default=str))
        return 0

    if args.cmd == "ingest":
        from helion_rag.ingest import ingest as run_ingest

        if not cfg.hardware_family:
            print("HELION_RAG_HARDWARE_FAMILY must be set for ingest", file=sys.stderr)
            return 2
        if not cfg.autotune_log_dir:
            print("HELION_RAG_AUTOTUNE_LOG_DIR must be set for ingest", file=sys.stderr)
            return 2
        ledger = cfg.writeback_dir.parent / "ingest-ledger.json"
        summary = run_ingest(
            autotune_log_dir=cfg.autotune_log_dir,
            writeback_dir=cfg.writeback_dir,
            family=cfg.hardware_family,
            ledger_path=ledger,
            cfg=cfg,
        )
        print(json.dumps(summary, default=str))
        return 0

    if args.cmd == "upload":
        from helion_rag.upload import upload as run_upload

        if not cfg.hardware_family:
            print("HELION_RAG_HARDWARE_FAMILY must be set for upload", file=sys.stderr)
            return 2
        manifold_put = None
        if not args.dry_run:
            if not cfg.manifold_base:
                print(
                    "HELION_RAG_MANIFOLD_BASE must be set for upload", file=sys.stderr
                )
                return 2
            manifold_put = _manifold_put(cfg.manifold_base)
        uploads_dir = cfg.uploads_dir or (cfg.writeback_dir.parent / "uploads")
        summary = run_upload(
            autotune_log_dir=cfg.autotune_log_dir,
            uploads_dir=uploads_dir,
            family=cfg.hardware_family,
            contributor=_contributor(),
            manifold_put=manifold_put,
            dry_run=args.dry_run,
            reupload=args.reupload,
        )
        print(json.dumps(summary, default=str))
        return 0

    if args.cmd == "patch-helion":
        target = Path(args.target)
        written = patch.write_hook(target)
        print(
            f"{'patched' if written else 'already present'} {target}", file=sys.stderr
        )
        return 0

    if args.cmd == "setup-helper":
        from helion_rag import setup_helpers as shm

        act = args.action
        if act == "check-import":
            return shm.check_helion_import(args.repo)
        if act == "synthesize":
            shm.synthesize_manifest(args.out, args.families)
            return 0
        if act == "validate":
            return shm.validate_manifest_cli(args.manifest)
        if act == "resolve-family":
            out = shm.resolve_family_cli(args.manifest, args.device)
            print(out)
            return 0
        if act == "is-represented":
            return 0 if shm.is_represented(args.manifest, args.family) else 1
        if act == "artifact-path":
            print(shm.artifact_path(args.manifest, args.family))
            return 0
        if act == "publish-manifest":
            summary = shm.publish_manifest(
                args.manifold_base,
                args.family,
                manifold_get=_manifold_get_file,
                manifold_put=_manifold_put_file,
                artifact_path=args.artifact_path,
                aliases=args.alias,
            )
            print(json.dumps(summary))
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
