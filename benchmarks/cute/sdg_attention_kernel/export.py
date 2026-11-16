"""Export committed SDG source and dependencies for an isolated evaluator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import zipfile

_SOURCE = Path("benchmarks/cute/sdg_attention_kernel")


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repo)


def _files(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink in export source: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return result


def export_candidate(
    repo: Path, profiles: Path, dependencies: Path, output: Path
) -> Path:
    """Build from committed bytes; keep failed assemblies out of the output path."""
    repo, profiles, dependencies, output = (
        path.resolve() for path in (repo, profiles, dependencies, output)
    )
    if output.exists():
        raise ValueError(f"Export output already exists: {output}")
    if not dependencies.is_dir():
        raise ValueError(f"Missing candidate dependency directory: {dependencies}")
    changed = _git(repo, "diff", "--name-only", "HEAD", "--", "helion", str(_SOURCE))
    if changed:
        raise ValueError(
            "Commit the compiler and benchmark source changes before exporting:\n"
            + changed.decode()
        )
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    compiler_tree = _git(repo, "rev-parse", "HEAD:helion").decode().strip()
    benchmark_tree = _git(repo, "rev-parse", f"HEAD:{_SOURCE}").decode().strip()
    archive = _git(repo, "archive", "--format=zip", "HEAD", "helion", str(_SOURCE))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temp:
        temporary = Path(temp)
        archived = temporary / "source"
        with zipfile.ZipFile(io.BytesIO(archive)) as stream:
            # Git preserves symlink modes in ZIP metadata. Reject them before
            # extraction instead of exporting their link text as Python source.
            for member in stream.infolist():
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError(f"Symlink in export source: {member.filename}")
            stream.extractall(archived)
        source = archived / _SOURCE
        spec = importlib.util.spec_from_file_location(
            "_sdg_export_profile_contract", source / "profile_contract.py"
        )
        assert spec is not None and spec.loader is not None
        contract = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contract)
        contract.load_profiles(profiles)
        stage = temporary / "candidate"
        stage.mkdir()
        vendor = stage / "_vendor"
        vendor.mkdir()
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".git")
        for dependency in dependencies.iterdir():
            if (
                dependency.name == "helion"
                or dependency.name.startswith("helion-")
                and dependency.name.endswith(".dist-info")
                or dependency.name in ("__pycache__", ".git")
                or dependency.suffix == ".pyc"
            ):
                continue
            target = vendor / dependency.name
            if dependency.is_dir():
                shutil.copytree(dependency, target, ignore=ignore)
            else:
                shutil.copy2(dependency, target)
        shutil.move(str(archived / "helion"), vendor / "helion")
        for name in (
            "__init__.py",
            "headwise.py",
            "recurrence.py",
            "rowwise.py",
            "profile_contract.py",
            "bootstrap.py",
            "README.md",
        ):
            shutil.copy2(source / name, stage / name)
        shutil.copy2(source / "kernel.py", stage / "implementation.py")
        shutil.copy2(source / "entry.py", stage / "kernel.py")
        shutil.copy2(profiles, stage / "configs.json")
        files = _files(stage)
        manifest = {
            "schema_version": 2,
            "compiler_commit": head,
            "helion_git_tree": compiler_tree,
            "benchmark_git_tree": benchmark_tree,
            "benchmark_source_files": _files(source),
            "profiles_sha256": files["configs.json"],
            "files": files,
            "origin_fingerprint_semantics": (
                "profiles.authored_source_sha256 identifies either a supported historical "
                "tuning source or the current family source file. It does not certify "
                "the entire compiler/dependency bundle. "
                "benchmark_source_files and files record the current committed "
                "and shipped bytes independently."
            ),
            "compilation": "Ordinary Helion compilation from authored source and stored configs",
        }
        (stage / "provenance.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if output.exists():
            raise ValueError(f"Export output appeared during assembly: {output}")
        stage.rename(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument(
        "--dependencies",
        required=True,
        type=Path,
        help="Trusted candidate _vendor directory; its Helion source/metadata are replaced",
    )
    args = parser.parse_args()
    try:
        output = export_candidate(
            Path(__file__).resolve().parents[3],
            args.profiles,
            args.dependencies,
            args.output,
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"SDG export failed: {error}\n")
    print(output)


if __name__ == "__main__":
    main()
