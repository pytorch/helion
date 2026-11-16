from __future__ import annotations

import ast
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
from types import ModuleType
from typing import Any

import pytest
import torch

from helion._testing import skipUnlessBackends

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]
_SOURCE = Path(__file__).resolve().parents[1] / "benchmarks/cute/sdg_attention_kernel"


def _module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXPORT = _module("_sdg_test_export", _SOURCE / "export.py")
_CONTRACT = _module("_sdg_test_contract", _SOURCE / "profile_contract.py")
_BOOTSTRAP = _module("_sdg_test_bootstrap", _SOURCE / "bootstrap.py")


def test_bootstrap_without_public_namespace_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Python 3.10 creates namespace loaders but does not export their class
    # from importlib.machinery. Collection and namespace validation must work.
    monkeypatch.delattr(importlib.machinery, "NamespaceLoader", raising=False)
    bootstrap = _module("_sdg_bootstrap_no_public_loader", _SOURCE / "bootstrap.py")
    monkeypatch.setattr(bootstrap, "_PREFIXES", ("nvidia_cutlass_dsl",))
    spec = importlib.machinery.ModuleSpec(
        "nvidia_cutlass_dsl", loader=None, is_package=True
    )
    namespace = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, namespace)
    bootstrap._verify_loaded_origins(tmp_path, allow_namespace=True)
    namespace.__loader__ = importlib.machinery.SourceFileLoader(
        spec.name, str(tmp_path / "unverified.py")
    )
    with pytest.raises(RuntimeError, match="already loaded package"):
        bootstrap._verify_loaded_origins(tmp_path, allow_namespace=True)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


@pytest.fixture
def bundle_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    source = repo / "benchmarks/cute/sdg_attention_kernel"
    shutil.copytree(_SOURCE, source, ignore=shutil.ignore_patterns("__pycache__"))
    # Exercise the real exporter/bootstrap with small import-only dependencies.
    # Full authored kernels and actual SDK startup are validated separately.
    (source / "kernel.py").write_text("def run():\n    return 17\n")
    (repo / "helion").mkdir()
    (repo / "helion/__init__.py").write_text("MARKER = 'committed'\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "helion", "benchmarks")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "fixture",
    )
    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profiles": [
                    {
                        "name": "fixture",
                        "inputs": {
                            name: {
                                "shape": [1],
                                "stride": [1],
                                "dtype": "torch.float32",
                            }
                            for name in (
                                "q",
                                "k",
                                "v",
                                "ks",
                                "vs",
                                "table",
                                "lengths",
                                "out",
                            )
                        },
                        "scale_storage_offset": 0,
                        "max_kv_len": 128,
                        "window_left": None,
                        "return_lse": True,
                        "family": "rowwise_extended",
                        "config": {"rescale_threshold": 64},
                        "authored_source_sha256": _CONTRACT.ORIGIN_FINGERPRINTS[
                            "rowwise_extended"
                        ],
                    }
                ],
            }
        )
    )
    vendor = tmp_path / "dependencies"
    sdk = vendor / "nvidia_cutlass_dsl/dsl_packages/cutlass"
    sdk.mkdir(parents=True)
    (sdk / "__init__.py").write_text("MARKER = 'vendor'\n")
    (sdk / "cutlass_dsl.py").write_text("MARKER = 'vendor-child'\n")
    (vendor / "nvidia_cutlass_dsl_packages.pth").write_text(
        "import sys, os, nvidia_cutlass_dsl; sys.path.insert(0, "
        "os.path.join(nvidia_cutlass_dsl.__path__[0], 'dsl_packages'))\n"
    )
    (vendor / "extra").mkdir()
    (vendor / "extra/sdg_pth_test.py").write_text("MARKER = 'vendor-extra'\n")
    (vendor / "extra.pth").write_text("extra\n")
    for name, version in (("nvidia_cutlass_dsl", "4.7.0"), ("helion", "0.0.1")):
        metadata = vendor / f"{name}-{version}.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
    (vendor / "helion").mkdir()
    (vendor / "helion/__init__.py").write_text("raise AssertionError('stale Helion')\n")
    return repo, profiles, vendor


def _export(inputs: tuple[Path, Path, Path], output: Path) -> Path:
    return _EXPORT.export_candidate(*inputs, output)


def test_committed_export_identity_and_metadata(
    bundle_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    repo, profiles, vendor = bundle_inputs
    (repo / "benchmarks/cute/sdg_attention_kernel/untracked.py").write_text(
        "not archived"
    )
    output = _export(bundle_inputs, tmp_path / "candidate")
    manifest = json.loads((output / "provenance.json").read_text())
    assert manifest["compiler_commit"] == _git(repo, "rev-parse", "HEAD")
    assert manifest["helion_git_tree"] == _git(repo, "rev-parse", "HEAD:helion")
    assert manifest["benchmark_git_tree"] == _git(
        repo, "rev-parse", "HEAD:benchmarks/cute/sdg_attention_kernel"
    )
    assert (
        manifest["profiles_sha256"] == hashlib.sha256(profiles.read_bytes()).hexdigest()
    )
    assert "untracked.py" not in manifest["benchmark_source_files"]
    assert not list((output / "_vendor").glob("helion-*.dist-info"))
    assert "committed" in (output / "_vendor/helion/__init__.py").read_text()
    assert (vendor / "helion-0.0.1.dist-info").exists()
    actual = _EXPORT._files(output)
    actual.pop("provenance.json")
    assert manifest["files"] == actual
    for relative, digest in manifest["benchmark_source_files"].items():
        original = repo / "benchmarks/cute/sdg_attention_kernel" / relative
        assert hashlib.sha256(original.read_bytes()).hexdigest() == digest


def test_run_dispatch_preserves_callers_and_checks_storage(
    bundle_inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Test the public Python gateway on CPU. The selected device kernel is an
    # observation boundary; its numerical GPU coverage lives in the native tests.
    q = torch.empty((1, 2, 8, 128), dtype=torch.bfloat16)
    out = torch.empty_like(q)
    bk = torch.zeros(128 * 132, dtype=torch.uint8)
    bv = torch.empty_like(bk)
    k = bk.view(torch.int8).as_strided((1, 128, 1, 128), (16896, 128, 128, 1))
    v = bv.view(torch.int8).as_strided(k.shape, k.stride())
    ks = bk.as_strided((1, 128, 1, 4), (16896, 4, 4, 1), 128 * 128)
    vs = bv.as_strided(ks.shape, ks.stride(), ks.storage_offset())
    table = torch.zeros((1, 1), dtype=torch.int32)
    lengths = torch.full((1,), 128, dtype=torch.int32)
    tensors = (q, k, v, ks, vs, table, lengths, out)
    profiles = json.loads(bundle_inputs[1].read_text())
    profile = profiles["profiles"][0]
    profile["inputs"] = {
        name: {
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in zip(_CONTRACT.INPUT_NAMES, tensors, strict=True)
    }
    profile["scale_storage_offset"] = ks.storage_offset()
    bundle_inputs[1].write_text(json.dumps(profiles))
    monkeypatch.setenv("SDG_ATTENTION_PROFILES", str(bundle_inputs[1]))
    spec = importlib.util.spec_from_file_location(
        "_sdg_run_selection",
        _SOURCE / "kernel.py",
        submodule_search_locations=[str(_SOURCE)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    observed = []
    result = (out, torch.empty((1, 2, 8)))

    def configured(name):
        assert name == "fixture"

        def launch(*args):
            observed.append(args)
            return result

        return launch

    monkeypatch.setattr(module, "configured_kernel", configured)

    def invoke(inputs=tensors, *, max_kv_len=128):
        return module.run(
            *inputs[:7],
            max_kv_len=max_kv_len,
            window_left=None,
            return_lse=True,
            out=inputs[7],
        )

    assert module.profile_names() == ("fixture",)
    for length in (128, 0):
        lengths.fill_(length)
        assert invoke() is result
        actual = observed[-1]
        assert all(
            left is right for left, right in zip(actual[:7], tensors[:7], strict=True)
        )
        assert actual[7:10] == (None, 128, True)
        assert actual[10] is out
    with pytest.raises(ValueError, match="Unsupported SDG attention layout"):
        invoke(max_kv_len=256)
    with pytest.raises(ValueError, match="share a device"):
        invoke((q.to("meta"), *tensors[1:]))
    split_scales = (
        ks.clone()
        .new_empty(bk.shape)
        .as_strided(ks.shape, ks.stride(), ks.storage_offset())
    )
    with pytest.raises(ValueError, match="scales must share their backing storage"):
        invoke((q, k, v, split_scales, vs, table, lengths, out))
    with pytest.raises(ValueError, match="storage must be distinct"):
        invoke((*tensors[:7], q))
    with pytest.raises(ValueError, match="storage must be distinct"):
        invoke((q, k, k, ks, ks, table, lengths, out))
    assert len(observed) == 2


def test_export_rejects_committed_symlinks(bundle_inputs, tmp_path):
    repo, profiles, vendor = bundle_inputs
    (repo / "helion/linked.py").symlink_to("__init__.py")
    _git(repo, "add", "helion/linked.py")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "symlink fixture",
    )
    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match="Symlink in export source"):
        _EXPORT.export_candidate(repo, profiles, vendor, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "relative", ["helion/__init__.py", "benchmarks/cute/sdg_attention_kernel/kernel.py"]
)
def test_dirty_source_rejected(
    bundle_inputs: tuple[Path, Path, Path], tmp_path: Path, relative: str
) -> None:
    path = bundle_inputs[0] / relative
    path.write_text(path.read_text() + "# dirty\n")
    with pytest.raises(ValueError, match="Commit the compiler and benchmark"):
        _export(bundle_inputs, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


def test_bad_profile_and_existing_output_leave_no_partial_export(
    bundle_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output = tmp_path / "candidate"
    profile = json.loads(bundle_inputs[1].read_text())
    profile["profiles"][0]["authored_source_sha256"] = "wrong"
    bundle_inputs[1].write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="tuning-origin fingerprint"):
        _export(bundle_inputs, output)
    assert not output.exists() and not list(tmp_path.glob(".candidate-*"))
    output.mkdir()
    marker = output / "keep"
    marker.write_text("existing")
    with pytest.raises(ValueError, match="already exists"):
        _export(bundle_inputs, output)
    assert marker.read_text() == "existing"


def test_profile_lineage_and_legacy_threshold(
    bundle_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    path = bundle_inputs[1]
    data = json.loads(path.read_text())
    assert _CONTRACT.load_profiles(path)[0]["config"]["rescale_threshold"] == 64
    for family in ("rowwise", "rowwise_extended"):
        profile = data["profiles"][0]
        profile["family"] = family
        for fingerprint in (
            _CONTRACT.ORIGIN_FINGERPRINTS[family],
            _CONTRACT.source_fingerprint(family),
        ):
            profile["authored_source_sha256"] = fingerprint
            for value in _CONTRACT.ROWWISE_THRESHOLDS:
                profile["config"]["rescale_threshold"] = value
                path.write_text(json.dumps(data))
                assert (
                    _CONTRACT.load_profiles(path)[0]["config"]["rescale_threshold"]
                    == value
                )
        for value in (3, 64.0, True):
            profile["config"]["rescale_threshold"] = value
            path.write_text(json.dumps(data))
            with pytest.raises(ValueError, match="supported integer threshold"):
                _CONTRACT.load_profiles(path)
    with pytest.raises(
        ValueError, match="Missing SDG profiles.*SDG_ATTENTION_PROFILES"
    ):
        _CONTRACT.load_profiles(tmp_path / "absent.json")
    for malformed in ([], {"schema_version": 1, "profiles": [None]}):
        path.write_text(json.dumps(malformed))
        with pytest.raises(ValueError, match="SDG profile"):
            _CONTRACT.load_profiles(path)


@pytest.mark.slow
def test_bootstrap_real_pth_origins_and_negative_controls(
    bundle_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    output = _export(bundle_inputs, tmp_path / "candidate")
    ambient = tmp_path / "ambient"
    (ambient / "nvidia_cutlass_dsl").mkdir(parents=True)
    (ambient / "sdg_pth_test.py").write_text("MARKER = 'ambient-extra'\n")
    for name in ("cutlass", "helion"):
        directory = ambient / name
        directory.mkdir()
        (directory / "__init__.py").write_text("MARKER = 'ambient'\n")
    script = textwrap.dedent("""
        import importlib.util, json, os, sys
        from pathlib import Path
        package, ambient, mode = map(str, sys.argv[1:])
        # -S retains stdlib only: model an ambient SDK installation deliberately.
        sys.path.insert(0, ambient)
        import nvidia_cutlass_dsl
        if mode in ('cutlass', 'helion'):
            __import__(mode)
        if mode == 'fileless':
            import types
            sys.modules['cutlass.unowned'] = types.ModuleType('cutlass.unowned')
        if mode == 'stale':
            import py_compile
            for relative, old, new in (
                ('implementation.py', b'return 17', b'return 99'),
                ('bootstrap.py', b'"schema_version"', b'"schema_versioN"'),
                ('_vendor/extra/sdg_pth_test.py', b'vendor-extra', b'vendor-stale'),
            ):
                path = Path(package) / relative
                original, stat = path.read_bytes(), path.stat()
                assert old in original and len(old) == len(new)
                path.write_bytes(original.replace(old, new))
                py_compile.compile(str(path), doraise=True)
                path.write_bytes(original)
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        if mode == 'environment':
            os.environ['SDG_ATTENTION_PROFILES'] = str(Path(package) / 'configs.json')
        if mode == 'metadata':
            metadata = Path(ambient) / 'helion-0.1.dist-info'
            metadata.mkdir()
            (metadata / 'METADATA').write_text('Name: helion\\nVersion: 0.1\\n')
        spec = importlib.util.spec_from_file_location('candidate', Path(package) / '__init__.py', submodule_search_locations=[package])
        module = importlib.util.module_from_spec(spec)
        sys.modules['candidate'] = module
        spec.loader.exec_module(module)
        from candidate.kernel import run
        assert run() == 17
        import cutlass, helion
        assert cutlass.MARKER == 'vendor' and helion.MARKER == 'committed'
        import sdg_pth_test
        assert sdg_pth_test.MARKER == 'vendor-extra'
        namespace = sys.modules['nvidia_cutlass_dsl']
        assert all(Path(p).is_relative_to(Path(package) / '_vendor') for p in namespace.__path__)
        assert Path(sys.pycache_prefix).is_dir()
        print(sys.pycache_prefix)
    """)
    environment = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "SDG_ATTENTION_PROFILES")
    }
    for mode, expected in (
        ("normal", None),
        ("stale", None),
        ("cutlass", "already loaded package"),
        ("helion", "already loaded package"),
        ("fileless", "already loaded package"),
        ("environment", "Unset SDG_ATTENTION_PROFILES"),
        ("metadata", "ambient Helion distribution"),
    ):
        result = subprocess.run(
            [sys.executable, "-S", "-B", "-c", script, str(output), str(ambient), mode],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if expected is None:
            assert result.returncode == 0, result.stderr
            assert result.stdout.strip()
            assert not Path(result.stdout.strip()).exists()
        else:
            assert result.returncode != 0 and expected in result.stderr, result.stderr
    for relative in ("configs.json", "implementation.py"):
        path = output / relative
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    "-B",
                    "-c",
                    script,
                    str(output),
                    str(ambient),
                    "normal",
                ],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            assert result.returncode != 0 and "bytes changed" in result.stderr, (
                result.stderr
            )
        finally:
            path.write_bytes(original)


@pytest.mark.parametrize("family", ["recurrence", "rowwise_extended", "headwise"])
def test_fixed_profile_selection_ignores_force_and_rejects_overrides(
    bundle_inputs: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    family: str,
) -> None:
    import torch

    import helion
    from helion._testing.cute_codegen import cpu_bind
    from helion._testing.cute_codegen import cpu_target

    profiles = json.loads(bundle_inputs[1].read_text())
    profile = profiles["profiles"][0]
    profile["family"] = family
    profile["authored_source_sha256"] = _CONTRACT.ORIGIN_FINGERPRINTS[family]
    config: dict[str, Any] = {} if family == "headwise" else {"block_sizes": [16]}
    if family == "rowwise_extended":
        config["rescale_threshold"] = 64
    profile["config"] = config
    bundle_inputs[1].write_text(json.dumps(profiles))
    monkeypatch.setenv("SDG_ATTENTION_PROFILES", str(bundle_inputs[1]))
    monkeypatch.setenv("HELION_FORCE_AUTOTUNE", "1")
    monkeypatch.setenv("HELION_AUTOTUNE_EFFORT", "none")
    monkeypatch.setenv("HELION_AUTOTUNE_FORCE_PERSISTENT", "1")
    monkeypatch.delenv("HELION_AUTOTUNE_CONFIG_OVERRIDES", raising=False)
    spec = importlib.util.spec_from_file_location(
        "_sdg_fixed_selection",
        _SOURCE / "kernel.py",
        submodule_search_locations=[str(_SOURCE)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    kernel = module.configured_kernel("fixture")
    assert not kernel.settings.force_autotune
    assert not kernel.settings.autotune_force_persistent
    with cpu_target():
        q = torch.empty((1, 2, 8, 128), dtype=torch.bfloat16)
        out = torch.empty_like(q)
        bk = torch.empty(128 * 132, dtype=torch.uint8)
        bv = torch.empty_like(bk)
        k = bk.view(torch.int8).as_strided((1, 128, 1, 128), (16896, 128, 128, 1))
        v = bv.view(torch.int8).as_strided(k.shape, k.stride())
        ks = bk.as_strided((1, 128, 1, 4), (16896, 4, 4, 1), 128 * 128)
        vs = bv.as_strided(ks.shape, ks.stride(), 128 * 128)
        table = torch.zeros((1, 1), dtype=torch.int32)
        lengths = torch.full((1,), 128, dtype=torch.int32)
        bound = cpu_bind(
            kernel, (q, k, v, ks, vs, table, lengths, None, 128, True, out)
        )
        assert bound._user_provided_config() == helion.Config(**config)
        # All three authored programs must survive ordinary binding and codegen;
        # importing their functions alone does not validate an exported profile.
        ast.parse(bound.to_code())
    overrides = tmp_path / "overrides.json"
    overrides.write_text("{}")
    monkeypatch.setenv("HELION_AUTOTUNE_CONFIG_OVERRIDES", str(overrides))
    assert module.configured_kernel("fixture") is kernel
    for setting in ('{"num_warps": 16}', str(overrides)):
        overrides.write_text('{"num_warps": 16}')
        monkeypatch.setenv("HELION_AUTOTUNE_CONFIG_OVERRIDES", setting)
        with pytest.raises(ValueError, match="fixed SDG profiles"):
            module.configured_kernel("fixture")


def test_extension_children_require_their_real_vendored_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vendor = tmp_path / "vendor"
    extension = vendor / "cutlass/native.so"
    parent = ModuleType("cutlass.native")
    parent.__file__ = str(extension)
    parent.__loader__ = importlib.machinery.ExtensionFileLoader(
        parent.__name__, str(extension)
    )
    child = ModuleType("cutlass.native.child")
    grandchild = ModuleType("cutlass.native.child.nested")
    parent.child = child
    child.nested = grandchild
    for module in (parent, child, grandchild):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    assert _BOOTSTRAP._extension_owns_child(child.__name__, child, vendor)
    assert _BOOTSTRAP._extension_owns_child(grandchild.__name__, grandchild, vendor)
    assert not _BOOTSTRAP._extension_owns_child(
        child.__name__, ModuleType(child.__name__), vendor
    )
    parent.__loader__ = importlib.machinery.SourceFileLoader(
        parent.__name__, str(extension)
    )
    assert not _BOOTSTRAP._extension_owns_child(child.__name__, child, vendor)
    parent.__loader__ = importlib.machinery.ExtensionFileLoader(
        parent.__name__, str(extension)
    )
    parent.__file__ = str(tmp_path / "ambient/native.so")
    assert not _BOOTSTRAP._extension_owns_child(child.__name__, child, vendor)


def test_export_rejects_incomplete_and_ambiguous_dispatch_profiles(
    bundle_inputs: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    profiles = bundle_inputs[1]
    original = json.loads(profiles.read_text())
    malformed = []
    for field in ("name", "inputs", "max_kv_len", "window_left"):
        data = copy.deepcopy(original)
        del data["profiles"][0][field]
        malformed.append(data)
    for duplicate_name in (True, False):
        data = copy.deepcopy(original)
        duplicate = copy.deepcopy(data["profiles"][0])
        if duplicate_name:
            duplicate["max_kv_len"] += 1
        else:
            duplicate["name"] = "different-name"
        data["profiles"].append(duplicate)
        malformed.append(data)
    for data in malformed:
        profiles.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="SDG.*profile"):
            _export(bundle_inputs, tmp_path / "candidate")
        assert not (tmp_path / "candidate").exists()
