#!/usr/bin/env python3
"""Compare example Triton codegen against the pre-stack baseline.

The existing ``test/test_examples.py`` suite owns the representative inputs and
configs for the examples.  This script runs that suite in the current checkout
and in a detached baseline worktree, intercepts every ``BoundKernel`` Triton
source emission, and compares the resulting source byte-for-byte.

Source-location comments are disabled in both runs because their absolute file
paths necessarily differ between worktrees.  The executable generated source
is otherwise compared without normalization.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import dataclasses
import difflib
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING
from typing import Protocol
from unittest import mock

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion.runtime.config import Config


@dataclasses.dataclass(frozen=True)
class CaptureRecord:
    nodeid: str
    invocation: int
    kernel: str
    source_file: str | None
    root_count: int | None
    config: str
    code_file: str
    sha256: str

    @property
    def key(self) -> tuple[str, int]:
        return self.nodeid, self.invocation


class _PytestItem(Protocol):
    nodeid: str
    path: Path


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *args),
        text=True,
    ).strip()


def _repo_root() -> Path:
    return Path(_git_output(Path.cwd(), "rev-parse", "--show-toplevel"))


def _stable_config(config: object) -> str:
    value = getattr(config, "config", config)
    try:
        return json.dumps(value, sort_keys=True, default=repr)
    except TypeError:
        return repr(value)


class _CapturePlugin:
    def __init__(self, root: Path, artifacts: Path) -> None:
        self.root = root
        self.artifacts = artifacts
        self.current_nodeid = "<session>"
        self.records: list[CaptureRecord] = []
        self._invocations: defaultdict[str, int] = defaultdict(int)
        self._seen: set[tuple[str, str, str, str]] = set()
        self.collected_tests = 0

    def pytest_collection_modifyitems(self, items: list[object]) -> None:
        self.collected_tests = len(items)

    def pytest_runtest_setup(self, item: _PytestItem) -> None:
        _path, separator, suffix = item.nodeid.partition("::")
        stable_path = item.path.name
        self.current_nodeid = f"{stable_path}::{suffix}" if separator else stable_path

    def pytest_runtest_teardown(self) -> None:
        self.current_nodeid = "<session>"

    def record(self, bound: object, config: object, code: str) -> None:
        kernel_object = getattr(bound, "kernel", None)
        kernel_name = str(getattr(kernel_object, "name", type(bound).__name__))
        kernel_fn = getattr(kernel_object, "fn", None)
        source_path = (
            inspect.getsourcefile(kernel_fn) if kernel_fn is not None else None
        )
        source_file: str | None = None
        if source_path is not None:
            resolved_source = Path(source_path).resolve()
            try:
                source_file = str(resolved_source.relative_to(self.root))
            except ValueError:
                source_file = str(resolved_source)

        host_function = getattr(bound, "host_function", None)
        device_ir = getattr(host_function, "device_ir", None)
        root_ids = getattr(device_ir, "root_ids", None)
        root_count = len(root_ids) if root_ids is not None else None
        config_text = _stable_config(config)
        digest = hashlib.sha256(code.encode()).hexdigest()
        duplicate_key = (self.current_nodeid, kernel_name, config_text, digest)
        if duplicate_key in self._seen:
            return
        self._seen.add(duplicate_key)

        invocation = self._invocations[self.current_nodeid]
        self._invocations[self.current_nodeid] += 1
        code_file = f"{len(self.records):05d}.py"
        (self.artifacts / code_file).write_bytes(code.encode())
        self.records.append(
            CaptureRecord(
                nodeid=self.current_nodeid,
                invocation=invocation,
                kernel=kernel_name,
                source_file=source_file,
                root_count=root_count,
                config=config_text,
                code_file=code_file,
                sha256=digest,
            )
        )


def _capture_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--test-target", action="append", dest="test_targets")
    parser.add_argument("--require-aot-heuristic", action="store_true")
    args, pytest_args = parser.parse_known_args(argv)

    root = args.root.resolve()
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ["HELION_BACKEND"] = "triton"
    os.environ["HELION_OUTPUT_ORIGIN_LINES"] = "0"
    if args.require_aot_heuristic:
        os.environ["HELION_AOT_MODE"] = "evaluate"
        os.environ["HELION_AUTOTUNE_EFFORT"] = "full"
        os.environ["HELION_FORCE_AUTOTUNE"] = "0"
        os.environ["HELION_SKIP_CACHE"] = "0"
        os.environ["HELION_INTERPRET"] = "0"
        os.environ["HELION_AOT_DATA_DIR"] = str(artifacts / "aot-data")
        os.environ.pop("HELION_AUTOTUNE_CONFIG_OVERRIDES", None)
        os.environ.pop("HELION_HEURISTIC_DIR", None)
        os.environ.pop("TRITON_INTERPRET", None)

    import pytest

    from helion.runtime.kernel import BoundKernel

    plugin = _CapturePlugin(root, artifacts)
    original = BoundKernel.to_triton_code

    def capture_to_triton_code(
        bound: BoundKernel[object],
        config: Config | dict[str, object] | None = None,
        *,
        emit_repro_caller: bool = False,
        output_origin_lines: bool | None = None,
    ) -> str:
        code = original(
            bound,
            config,
            emit_repro_caller=emit_repro_caller,
            output_origin_lines=(
                False if output_origin_lines is None else output_origin_lines
            ),
        )
        plugin.record(bound, config, code)
        return code

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                BoundKernel,
                "to_triton_code",
                capture_to_triton_code,
            )
        )
        if args.require_aot_heuristic:
            from helion.autotuner.aot_cache import AOTAutotuneCache

            original_get_heuristic_config = AOTAutotuneCache._get_heuristic_config

            def require_heuristic_config(
                cache: AOTAutotuneCache,
                heuristic_args: Sequence[object] | None = None,
            ) -> Config | None:
                heuristic_file = cache._find_heuristic_file()
                kernel_source = Path(cache.kernel.kernel.__code__.co_filename).resolve()
                if (
                    heuristic_file is None
                    or heuristic_file.resolve().parent != kernel_source.parent
                ):
                    kernel_name = cache.kernel.kernel.name
                    raise RuntimeError(
                        f"No adjacent checked-in AOT heuristic found for "
                        f"{kernel_name!r}"
                    )
                config = original_get_heuristic_config(cache, heuristic_args)
                if config is None:
                    kernel_name = cache.kernel.kernel.name
                    raise RuntimeError(
                        f"No checked-in AOT heuristic selected a config for "
                        f"{kernel_name!r}"
                    )
                return config

            stack.enter_context(
                mock.patch.object(
                    AOTAutotuneCache,
                    "_get_heuristic_config",
                    require_heuristic_config,
                )
            )
        exit_code = int(
            pytest.main(
                [*(args.test_targets or ["test/test_examples.py"]), "-q", *pytest_args],
                plugins=[plugin],
            )
        )
    manifest = {
        "root": str(root),
        "collected_tests": plugin.collected_tests,
        "pytest_exit_code": exit_code,
        "records": [dataclasses.asdict(record) for record in plugin.records],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return exit_code


def _load_manifest(path: Path) -> tuple[dict[str, object], list[CaptureRecord]]:
    payload = json.loads(path.read_text())
    records = [CaptureRecord(**record) for record in payload["records"]]
    return payload, records


def _print_diff(
    baseline_code: str,
    candidate_code: str,
    *,
    label: str,
    max_lines: int,
) -> None:
    lines = list(
        difflib.unified_diff(
            baseline_code.splitlines(keepends=True),
            candidate_code.splitlines(keepends=True),
            fromfile=f"baseline:{label}",
            tofile=f"candidate:{label}",
        )
    )
    sys.stdout.writelines(lines[:max_lines])
    if len(lines) > max_lines:
        print(f"... diff truncated after {max_lines} lines")


def _compare_captures(
    baseline_manifest: Path,
    candidate_manifest: Path,
    baseline_artifacts: Path,
    candidate_artifacts: Path,
    *,
    max_diffs: int,
    max_diff_lines: int,
    suite_name: str,
    require_same_collection: bool,
    required_kernels: Sequence[str],
) -> bool:
    baseline_metadata, baseline_records = _load_manifest(baseline_manifest)
    candidate_metadata, candidate_records = _load_manifest(candidate_manifest)
    if (
        require_same_collection
        and baseline_metadata["collected_tests"]
        != candidate_metadata["collected_tests"]
    ):
        print(
            "Collected test count differs: "
            f"baseline={baseline_metadata['collected_tests']} "
            f"candidate={candidate_metadata['collected_tests']}",
            file=sys.stderr,
        )
        return False

    baseline_by_key = {record.key: record for record in baseline_records}
    candidate_by_key = {record.key: record for record in candidate_records}
    missing_required_kernels: dict[str, list[str]] = {}
    required_kernel_set = set(required_kernels)
    for label, records in (
        ("baseline", baseline_records),
        ("candidate", candidate_records),
    ):
        missing_kernels = sorted(
            required_kernel_set - {record.kernel for record in records}
        )
        if missing_kernels:
            missing_required_kernels[label] = missing_kernels
            print(
                f"{label.capitalize()} did not capture required kernels: "
                f"{missing_kernels}",
                file=sys.stderr,
            )
    missing = sorted(baseline_by_key.keys() - candidate_by_key.keys())
    added = sorted(candidate_by_key.keys() - baseline_by_key.keys())
    if missing:
        print(f"Missing candidate captures: {missing[:10]}", file=sys.stderr)
    if added:
        print(f"Additional candidate captures: {added[:10]}", file=sys.stderr)

    mismatches: list[tuple[CaptureRecord, CaptureRecord]] = []
    metadata_mismatches: list[tuple[CaptureRecord, CaptureRecord]] = []
    for key in sorted(baseline_by_key.keys() & candidate_by_key.keys()):
        baseline = baseline_by_key[key]
        candidate = candidate_by_key[key]
        if (
            baseline.kernel,
            baseline.source_file,
            baseline.root_count,
        ) != (
            candidate.kernel,
            candidate.source_file,
            candidate.root_count,
        ):
            metadata_mismatches.append((baseline, candidate))
            continue
        if baseline.sha256 != candidate.sha256:
            mismatches.append((baseline, candidate))

    for baseline, candidate in metadata_mismatches[:max_diffs]:
        print(
            f"Capture identity changed for {baseline.key}:\n"
            f"  baseline: {baseline.kernel}, {baseline.source_file}, "
            f"roots={baseline.root_count}\n"
            f"  candidate: {candidate.kernel}, {candidate.source_file}, "
            f"roots={candidate.root_count}",
            file=sys.stderr,
        )

    for baseline, candidate in mismatches[:max_diffs]:
        label = f"{baseline.nodeid}#{baseline.invocation}:{baseline.kernel}"
        print(f"Generated source differs for {label}", file=sys.stderr)
        _print_diff(
            (baseline_artifacts / baseline.code_file).read_text(),
            (candidate_artifacts / candidate.code_file).read_text(),
            label=label,
            max_lines=max_diff_lines,
        )

    if metadata_mismatches or mismatches:
        print(
            f"Found {len(metadata_mismatches)} capture-identity changes and "
            f"{len(mismatches)} byte differences.",
            file=sys.stderr,
        )

    compared = len(baseline_by_key.keys() & candidate_by_key.keys())
    source_files = {
        record.source_file for record in candidate_records if record.source_file
    }
    multi_root = [record for record in candidate_records if record.root_count != 1]
    print(
        f"Compared {compared} generated programs from "
        f"{len(source_files)} {suite_name} source files."
    )
    if multi_root:
        multi_root_labels = sorted(
            {
                f"{record.nodeid} ({record.kernel}, roots={record.root_count})"
                for record in multi_root
            }
        )
        print(
            f"Note: {len(multi_root)} captures came from kernels with a root count "
            "other than one."
        )
        for label in multi_root_labels:
            print(f"  {label}")
    return not (
        missing_required_kernels
        or missing
        or added
        or metadata_mismatches
        or mismatches
    )


def compare_main(
    argv: Sequence[str],
    *,
    test_target: str | Sequence[str] = "test/test_examples.py",
    suite_name: str = "example",
    default_pytest_args: Sequence[str] = (),
    description: str | None = __doc__,
    require_same_collection: bool = True,
    capture_args: Sequence[str] = (),
    required_kernels: Sequence[str] = (),
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--baseline-ref",
        help="Git ref to compare against (default: merge-base with origin/main)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for both pytest runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Keep manifests and generated sources in this directory",
    )
    parser.add_argument("--max-diffs", type=int, default=3)
    parser.add_argument("--max-diff-lines", type=int, default=200)
    args, pytest_args = parser.parse_known_args(argv)
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    pytest_args = [*default_pytest_args, *pytest_args]

    candidate_root = _repo_root()
    test_targets = (test_target,) if isinstance(test_target, str) else test_target
    baseline_ref = args.baseline_ref or _git_output(
        candidate_root, "merge-base", "HEAD", "origin/main"
    )
    baseline_ref = _git_output(candidate_root, "rev-parse", baseline_ref)
    script = Path(__file__).resolve()

    temporary_output = args.output_dir is None
    output_root = (
        Path(tempfile.mkdtemp(prefix="helion-example-codegen-"))
        if temporary_output
        else args.output_dir.resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_root = output_root / "baseline-worktree"
    baseline_artifacts = output_root / "baseline"
    candidate_artifacts = output_root / "candidate"
    baseline_manifest = baseline_artifacts / "manifest.json"
    candidate_manifest = candidate_artifacts / "manifest.json"

    environment = os.environ.copy()
    environment["HELION_BACKEND"] = "triton"
    environment["HELION_OUTPUT_ORIGIN_LINES"] = "0"
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    worktree_added = False
    try:
        print(f"Baseline: {baseline_ref}")
        print(f"Candidate: {candidate_root}")
        _run(
            (
                "git",
                "-C",
                str(candidate_root),
                "worktree",
                "add",
                "--detach",
                str(baseline_root),
                baseline_ref,
            ),
            cwd=candidate_root,
        )
        worktree_added = True

        for label, root, artifacts, manifest in (
            ("baseline", baseline_root, baseline_artifacts, baseline_manifest),
            ("candidate", candidate_root, candidate_artifacts, candidate_manifest),
        ):
            artifacts.mkdir(parents=True, exist_ok=True)
            run_environment = environment.copy()
            run_environment["PYTHONPATH"] = str(root)
            run_environment["TORCHINDUCTOR_CACHE_DIR"] = str(artifacts / "inductor")
            run_environment["TRITON_CACHE_DIR"] = str(artifacts / "triton")
            print(f"Running {label} {suite_name} suite...")
            _run(
                (
                    args.python,
                    str(script),
                    "_capture",
                    "--root",
                    str(root),
                    "--artifacts",
                    str(artifacts),
                    "--manifest",
                    str(manifest),
                    *(
                        arg
                        for target in test_targets
                        for arg in ("--test-target", target)
                    ),
                    *capture_args,
                    *pytest_args,
                ),
                cwd=root,
                env=run_environment,
            )

        identical = _compare_captures(
            baseline_manifest,
            candidate_manifest,
            baseline_artifacts,
            candidate_artifacts,
            max_diffs=args.max_diffs,
            max_diff_lines=args.max_diff_lines,
            suite_name=suite_name,
            require_same_collection=require_same_collection,
            required_kernels=required_kernels,
        )
        if identical:
            print(f"All captured {suite_name} generated sources are byte-identical.")
            return 0
        print(
            f"{suite_name.capitalize()} generated source comparison failed.",
            file=sys.stderr,
        )
        return 1
    finally:
        if worktree_added:
            subprocess.run(
                (
                    "git",
                    "-C",
                    str(candidate_root),
                    "worktree",
                    "remove",
                    "--force",
                    str(baseline_root),
                ),
                cwd=candidate_root,
                check=False,
            )
        if temporary_output:
            shutil.rmtree(output_root, ignore_errors=True)
        else:
            print(f"Artifacts: {output_root}")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "_capture":
        return _capture_main(sys.argv[2:])
    return compare_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
